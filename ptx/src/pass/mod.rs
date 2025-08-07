use ptx_parser as ast;
use quick_error::quick_error;
use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::{
    borrow::Cow,
    cell::RefCell,
    collections::{hash_map, HashMap, HashSet},
    ffi::CString,
    fmt::{self, Display, Formatter, Write},
    iter,
    path::Path,
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

pub(crate) mod debug_integration;
mod deparamize_functions;
pub(crate) mod emit_llvm;
pub(crate) mod emit_tosa_mlir;
mod expand_operands;
mod fix_special_registers2;
mod hoist_globals;
mod insert_explicit_load_store;
mod insert_implicit_conversions2;
pub(crate) mod mlir_debug_framework;
pub(crate) mod mlir_debugger_integration;
mod normalize_identifiers2;
mod normalize_predicates2;
mod replace_instructions_with_function_calls;
mod replace_known_functions;
mod resolve_function_pointers;

// Re-export necessary types for emit_llvm.rs
pub use debug_integration::DebugAwarePtxContext;

#[cfg(feature = "amd")]
static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../../lib/zluda_ptx_impl.bc");
#[cfg(feature = "intel")]
static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../../lib/zluda_ptx_ze_impl.bc");
#[cfg(not(any(feature = "amd", feature = "intel")))]
static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../../lib/zluda_ptx_impl.bc");
const ZLUDA_PTX_PREFIX: &'static str = "__zluda_ptx_impl_";

quick_error! {
    #[derive(Debug)]
    pub enum TranslateError {
        UnknownSymbol {}
        UnreachableCodeError {}
        Todo
        InvalidSymbolFormat {}
        OutOfRangeCall {}
        UntypedSymbol {}
        MismatchedType {}
        Unreachable {}
        UnexpectedError(message: String) {}
        LLVMValidationError(message: String) {}
        MissingId {}
    }
}

pub fn to_mlir_module<'input>(ast: ast::Module<'input>) -> Result<String, TranslateError> {
    // Direct PTX to Linalg MLIR conversion
    let mut flat_resolver = GlobalStringIdentResolver2::<'input>::new(SpirvWord(1));
    let mut scoped_resolver = ScopedResolver::new(&mut flat_resolver);
    eprintln!("ZLUDA DEBUG: Created scoped_resolver");
    let sreg_map = SpecialRegistersMap2::new(&mut scoped_resolver)?;
    eprintln!("ZLUDA DEBUG: Created sreg_map");
    // eprintln!("=== AST before normalize_identifiers2 ===");
    // eprintln!("{:#?}", ast.directives);
    let directives = normalize_identifiers2::run(&mut scoped_resolver, ast.directives)?;
    // eprintln!("=== AST after normalize_identifiers2 ===");
    // eprintln!("{:#?}", directives);
    eprintln!("ZLUDA DEBUG: Completed normalize_identifiers2");
    // eprintln!("=== AST before replace_known_functions ===");
    // eprintln!("{:#?}", directives);
    let directives = replace_known_functions::run(&flat_resolver, directives);
    // eprintln!("=== AST after replace_known_functions ===");
    // eprintln!("{:#?}", directives);
    eprintln!("ZLUDA DEBUG: Completed replace_known_functions");
    // eprintln!("=== AST before normalize_predicates2 ===");
    // eprintln!("{:#?}", directives);
    let directives = normalize_predicates2::run(&mut flat_resolver, directives)?;
    // eprintln!("=== AST after normalize_predicates2 ===");
    // eprintln!("{:#?}", directives);
    eprintln!("ZLUDA DEBUG: Completed normalize_predicates2");
    // eprintln!("=== AST before resolve_function_pointers ===");
    // eprintln!("{:#?}", directives);
    let directives = resolve_function_pointers::run(directives)?;
    // eprintln!("=== AST after resolve_function_pointers ===");
    // eprintln!("{:#?}", directives);
    eprintln!("ZLUDA DEBUG: Completed resolve_function_pointers");
    // eprintln!("=== AST before fix_special_registers2 ===");
    // eprintln!("{:#?}", directives);
    let directives: Vec<
        Directive2<
            '_,
            ptx_parser::Instruction<ptx_parser::ParsedOperand<SpirvWord>>,
            ptx_parser::ParsedOperand<SpirvWord>,
        >,
    > = fix_special_registers2::run(&mut flat_resolver, &sreg_map, directives)?;
    // eprintln!("=== AST after fix_special_registers2 ===");
    // eprintln!("{:#?}", directives);
    eprintln!("ZLUDA DEBUG: Completed fix_special_registers2");
    // eprintln!("=== AST before expand_operands ===");
    // eprintln!("{:#?}", directives);
    let directives = expand_operands::run(&mut flat_resolver, directives)?;
    // eprintln!("=== AST after expand_operands ===");
    // eprintln!("{:#?}", directives);
    eprintln!("ZLUDA DEBUG: Completed expand_operands");
    // eprintln!("=== AST before deparamize_functions ===");
    // eprintln!("{:#?}", directives);
    let directives = deparamize_functions::run(&mut flat_resolver, directives)?;
    // eprintln!("=== AST after deparamize_functions ===");
    // eprintln!("{:#?}", directives);
    eprintln!("ZLUDA DEBUG: Completed deparamize_functions");
    // Skip insert_explicit_load_store for TOSA MLIR - it adds unnecessary complexity
    // let directives = insert_explicit_load_store::run(&mut flat_resolver, directives)?;
    eprintln!("ZLUDA DEBUG: Skipped insert_explicit_load_store for TOSA MLIR");
    // eprintln!("=== AST before insert_implicit_conversions2 ===");
    // eprintln!("{:#?}", directives);
    let directives = insert_implicit_conversions2::run(&mut flat_resolver, directives)?;
    // eprintln!("=== AST after insert_implicit_conversions2 ===");
    // eprintln!("{:#?}", directives);
    eprintln!("ZLUDA DEBUG: Completed insert_implicit_conversions2");
    // eprintln!("=== AST before replace_instructions_with_function_calls ===");
    // eprintln!("{:#?}", directives);
    let directives = replace_instructions_with_function_calls::run(&mut flat_resolver, directives)?;
    // eprintln!("=== AST after replace_instructions_with_function_calls ===");
    // eprintln!("{:#?}", directives);
    eprintln!("ZLUDA DEBUG: Completed replace_instructions_with_function_calls");
    // eprintln!("=== AST before hoist_globals ===");
    // eprintln!("{:#?}", directives);
    let directives = hoist_globals::run(directives)?;
    // eprintln!("=== AST after hoist_globals ===");
    // eprintln!("{:#?}", directives);
    eprintln!("ZLUDA DEBUG: Completed hoist_globals");

    eprintln!("=== AST before mlir codegen ===");
    eprintln!("{:#?}", directives);
    // Convert directly to Linalg MLIR
    let mlir_code = emit_tosa_mlir::run(flat_resolver, directives)?;
    eprintln!("ZLUDA DEBUG: Completed emit_linalg_mlir");

    Ok(mlir_code)
}

pub fn to_llvm_module<'input>(ast: ast::Module<'input>) -> Result<Module, TranslateError> {
    let mut flat_resolver = GlobalStringIdentResolver2::<'input>::new(SpirvWord(1));
    let mut scoped_resolver = ScopedResolver::new(&mut flat_resolver);
    eprintln!("ZLUDA DEBUG: Created scoped_resolver");
    let sreg_map = SpecialRegistersMap2::new(&mut scoped_resolver)?;
    eprintln!("ZLUDA DEBUG: Created sreg_map");
    let directives = normalize_identifiers2::run(&mut scoped_resolver, ast.directives)?;
    eprintln!("ZLUDA DEBUG: Completed normalize_identifiers2");
    let directives = replace_known_functions::run(&flat_resolver, directives);
    eprintln!("ZLUDA DEBUG: Completed replace_known_functions");
    let directives = normalize_predicates2::run(&mut flat_resolver, directives)?;
    eprintln!("ZLUDA DEBUG: Completed normalize_predicates2");
    let directives = resolve_function_pointers::run(directives)?;
    eprintln!("ZLUDA DEBUG: Completed resolve_function_pointers");
    let directives: Vec<
        Directive2<
            '_,
            ptx_parser::Instruction<ptx_parser::ParsedOperand<SpirvWord>>,
            ptx_parser::ParsedOperand<SpirvWord>,
        >,
    > = fix_special_registers2::run(&mut flat_resolver, &sreg_map, directives)?;
    eprintln!("ZLUDA DEBUG: Completed fix_special_registers2");
    let directives = expand_operands::run(&mut flat_resolver, directives)?;
    eprintln!("ZLUDA DEBUG: Completed expand_operands");
    let directives = deparamize_functions::run(&mut flat_resolver, directives)?;
    eprintln!("ZLUDA DEBUG: Completed deparamize_functions");
    let directives = insert_explicit_load_store::run(&mut flat_resolver, directives)?;
    eprintln!("ZLUDA DEBUG: Completed insert_explicit_load_store");
    let directives = insert_implicit_conversions2::run(&mut flat_resolver, directives)?;
    eprintln!("ZLUDA DEBUG: Completed insert_implicit_conversions2");
    let directives = replace_instructions_with_function_calls::run(&mut flat_resolver, directives)?;
    eprintln!("ZLUDA DEBUG: Completed replace_instructions_with_function_calls");
    let directives = hoist_globals::run(directives)?;
    eprintln!("ZLUDA DEBUG: Completed hoist_globals");
    let llvm_ir = emit_llvm::run(flat_resolver, directives)?;
    eprintln!("ZLUDA DEBUG: Completed emit_llvm");
    Ok(Module {
        llvm_ir,
        kernel_info: HashMap::new(),
    })
}

pub struct Module {
    pub llvm_ir: emit_llvm::MemoryBuffer,
    pub kernel_info: HashMap<String, KernelInfo>,
}

impl Module {
    pub fn linked_bitcode(&self) -> &[u8] {
        ZLUDA_PTX_IMPL
    }

    pub fn print_to_string(&self) -> Result<String, String> {
        use std::fs;
        use std::io::Write;
        use std::process::Command;

        // Create a temporary file to store the bitcode
        let temp_bc_path = "/tmp/zluda_temp.bc";
        let temp_ll_path = "/tmp/zluda_temp.ll";

        // Write the LLVM IR to a temp file
        fs::write(temp_bc_path, &*self.llvm_ir)
            .map_err(|e| format!("Failed to write temporary bitcode file: {}", e))?;

        // Use llvm-dis to convert the bitcode to text
        let llvm_dis_output = Command::new("llvm-dis-20")
            .arg(temp_bc_path)
            .arg("-o")
            .arg(temp_ll_path)
            .output()
            .map_err(|e| format!("Failed to execute llvm-dis: {}", e))?;

        if !llvm_dis_output.status.success() {
            return Err(format!(
                "llvm-dis failed: {}",
                String::from_utf8_lossy(&llvm_dis_output.stderr)
            ));
        }

        // Read the resulting text file
        let ir_text = fs::read_to_string(temp_ll_path)
            .map_err(|e| format!("Failed to read disassembled LLVM IR: {}", e))?;

        // Clean up temp files
        // let _ = fs::remove_file(temp_bc_path);
        // let _ = fs::remove_file(temp_ll_path);

        Ok(ir_text)
    }
}

/// PTX to LLVM to PTX compilation with debug info for SASS mapping
pub fn to_llvm_module_with_debug_round_trip<'input>(
    ast: ast::Module<'input>,
) -> Result<
    (
        Module,
        String,
        HashMap<u64, crate::debug::PtxSourceLocation>,
    ),
    TranslateError,
> {
    // First compile PTX to LLVM with debug info preserved
    let llvm_module = to_llvm_module(ast)?;

    // Get the LLVM IR as text for debugging
    let llvm_ir_text = llvm_module.print_to_string().map_err(|e| {
        TranslateError::UnexpectedError(format!("Failed to convert LLVM to string: {}", e))
    })?;

    // Save LLVM IR with debug info to /tmp
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let llvm_ir_path = format!("/tmp/ptx_debug_{}_llvm.ll", timestamp);
    std::fs::write(&llvm_ir_path, &llvm_ir_text).map_err(|e| {
        TranslateError::UnexpectedError(format!(
            "Failed to write LLVM IR to {}: {}",
            llvm_ir_path, e
        ))
    })?;
    eprintln!(
        "ZLUDA DEBUG: Saved LLVM IR with debug info to: {}",
        llvm_ir_path
    );

    // Save bitcode to /tmp
    let bitcode_path = format!("/tmp/ptx_debug_{}_llvm.bc", timestamp);
    std::fs::write(&bitcode_path, &*llvm_module.llvm_ir).map_err(|e| {
        TranslateError::UnexpectedError(format!(
            "Failed to write bitcode to {}: {}",
            bitcode_path, e
        ))
    })?;
    eprintln!("ZLUDA DEBUG: Saved LLVM bitcode to: {}", bitcode_path);

    // For now, we'll need to parse the bitcode back to get an LLVMModuleRef
    // This is not ideal but necessary given the current Module structure
    unsafe {
        use llvm_zluda::bit_reader::*;
        use llvm_zluda::core::*;
        use std::ffi::CString;

        // Create a new LLVM context for conversion
        let context = LLVMContextCreate();

        // Create memory buffer from the module's bitcode
        let bitcode_data = &llvm_module.llvm_ir;
        let mem_buf = LLVMCreateMemoryBufferWithMemoryRangeCopy(
            bitcode_data.as_ptr() as *const i8,
            bitcode_data.len(),
            CString::new("module").unwrap().as_ptr(),
        );

        // Parse bitcode into module
        let mut module_ref = std::ptr::null_mut();
        let mut err_msg = std::ptr::null_mut();
        let result = LLVMParseBitcodeInContext(context, mem_buf, &mut module_ref, &mut err_msg);

        if result != 0 {
            let error = if !err_msg.is_null() {
                let err_str = std::ffi::CStr::from_ptr(err_msg)
                    .to_str()
                    .unwrap_or("Unknown error");
                LLVMDisposeMessage(err_msg);
                err_str.to_string()
            } else {
                "Failed to parse bitcode".to_string()
            };
            LLVMContextDispose(context);
            return Err(TranslateError::UnexpectedError(format!(
                "Failed to parse LLVM bitcode: {}",
                error
            )));
        }

        // Use LLVM's NVPTX backend to convert to PTX
        eprintln!("ZLUDA DEBUG: Using LLVM NVPTX backend for PTX generation...");

        // Write LLVM IR to a temporary file
        let llvm_temp_path = format!("/tmp/zluda_llvm_{}.ll", timestamp);
        std::fs::write(&llvm_temp_path, &llvm_ir_text).map_err(|e| {
            TranslateError::UnexpectedError(format!("Failed to write temp LLVM: {}", e))
        })?;

        // Use llc to convert LLVM IR to PTX with full DWARF debug info
        let ptx_temp_path = format!("/tmp/zluda_ptx_{}.ptx", timestamp);
        let llc_output = std::process::Command::new("llc-20")
            .args(&[
                "-march=nvptx64",
                "-mcpu=sm_61", // Use newer compute capability for better debug support
                "-filetype=asm", // Generate assembly (PTX)
                "-O0",         // No optimization to preserve debug info
                &llvm_temp_path,
                "-o",
                &ptx_temp_path,
            ])
            .output()
            .map_err(|e| {
                TranslateError::UnexpectedError(format!("Failed to execute llc: {}", e))
            })?;

        if !llc_output.status.success() {
            let stderr = String::from_utf8_lossy(&llc_output.stderr);
            return Err(TranslateError::UnexpectedError(format!(
                "llc failed: {}",
                stderr
            )));
        }

        // Read the generated PTX
        let ptx_text = std::fs::read_to_string(ptx_temp_path).map_err(|e| {
            TranslateError::UnexpectedError(format!("Failed to read generated PTX: {}", e))
        })?;

        Ok((llvm_module, ptx_text, HashMap::new()))
    }
}

/// PTX to LLVM to PTX compilation with debug info and custom filename
pub fn to_llvm_module_with_debug_round_trip_and_filename<'input>(
    ast: ast::Module<'input>,
    source_filename: &str,
) -> Result<
    (
        Module,
        String,
        HashMap<u64, crate::debug::PtxSourceLocation>,
    ),
    TranslateError,
> {
    // First compile PTX to LLVM with debug info preserved and custom filename
    let llvm_module = to_llvm_module_with_filename(ast, source_filename)?;

    // Get the LLVM IR as text for debugging
    let llvm_ir_text = llvm_module.print_to_string().map_err(|e| {
        TranslateError::UnexpectedError(format!("Failed to convert LLVM to string: {}", e))
    })?;

    // Save LLVM IR with debug info to /tmp
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let llvm_ir_path = format!("/tmp/ptx_debug_{}_llvm.ll", timestamp);
    std::fs::write(&llvm_ir_path, &llvm_ir_text).map_err(|e| {
        TranslateError::UnexpectedError(format!(
            "Failed to write LLVM IR to {}: {}",
            llvm_ir_path, e
        ))
    })?;
    eprintln!(
        "ZLUDA DEBUG: Saved LLVM IR with debug info to: {}",
        llvm_ir_path
    );

    // Save bitcode to /tmp
    let bitcode_path = format!("/tmp/ptx_debug_{}_llvm.bc", timestamp);
    std::fs::write(&bitcode_path, &*llvm_module.llvm_ir).map_err(|e| {
        TranslateError::UnexpectedError(format!(
            "Failed to write bitcode to {}: {}",
            bitcode_path, e
        ))
    })?;
    eprintln!("ZLUDA DEBUG: Saved LLVM bitcode to: {}", bitcode_path);

    // For now, we'll need to parse the bitcode back to get an LLVMModuleRef
    // This is not ideal but necessary given the current Module structure
    unsafe {
        use llvm_zluda::bit_reader::*;
        use llvm_zluda::core::*;
        use std::ffi::CString;

        // Create a new LLVM context for conversion
        let context = LLVMContextCreate();

        // Create memory buffer from the module's bitcode
        let bitcode_data = &llvm_module.llvm_ir;
        let mem_buf = LLVMCreateMemoryBufferWithMemoryRangeCopy(
            bitcode_data.as_ptr() as *const i8,
            bitcode_data.len(),
            CString::new("module").unwrap().as_ptr(),
        );

        // Parse bitcode into module
        let mut module_ref = std::ptr::null_mut();
        let mut err_msg = std::ptr::null_mut();
        let result = LLVMParseBitcodeInContext(context, mem_buf, &mut module_ref, &mut err_msg);

        if result != 0 {
            let error = if !err_msg.is_null() {
                let err_str = std::ffi::CStr::from_ptr(err_msg)
                    .to_str()
                    .unwrap_or("Unknown error");
                LLVMDisposeMessage(err_msg);
                err_str.to_string()
            } else {
                "Failed to parse bitcode".to_string()
            };
            LLVMContextDispose(context);
            return Err(TranslateError::UnexpectedError(format!(
                "Failed to parse LLVM bitcode: {}",
                error
            )));
        }

        // Use LLVM's NVPTX backend to convert to PTX
        eprintln!("ZLUDA DEBUG: Using LLVM NVPTX backend for PTX generation...");

        // Write LLVM IR to a temporary file
        let llvm_temp_path = format!("/tmp/zluda_llvm_{}.ll", timestamp);
        std::fs::write(&llvm_temp_path, &llvm_ir_text).map_err(|e| {
            TranslateError::UnexpectedError(format!("Failed to write temp LLVM: {}", e))
        })?;

        // Use llc to convert LLVM IR to PTX with full DWARF debug info
        let ptx_temp_path = format!("/tmp/zluda_ptx_{}.ptx", timestamp);
        let llc_output = std::process::Command::new("llc-20")
            .args(&[
                "-march=nvptx64",
                "-mcpu=sm_61", // Use newer compute capability for better debug support
                "-filetype=asm", // Generate assembly (PTX)
                "-O0",         // No optimization to preserve debug info
                &llvm_temp_path,
                "-o",
                &ptx_temp_path,
            ])
            .output()
            .map_err(|e| {
                TranslateError::UnexpectedError(format!("Failed to execute llc: {}", e))
            })?;

        if !llc_output.status.success() {
            let stderr = String::from_utf8_lossy(&llc_output.stderr);
            return Err(TranslateError::UnexpectedError(format!(
                "llc failed: {}",
                stderr
            )));
        }

        // Read the generated PTX
        let ptx_text = std::fs::read_to_string(ptx_temp_path).map_err(|e| {
            TranslateError::UnexpectedError(format!("Failed to read generated PTX: {}", e))
        })?;

        Ok((llvm_module, ptx_text, HashMap::new()))
    }
}

/// PTX to LLVM compilation with custom filename for debug info
pub fn to_llvm_module_with_filename<'input>(
    ast: ast::Module<'input>,
    source_filename: &str,
) -> Result<Module, TranslateError> {
    let mut id_defs = GlobalStringIdentResolver2::new(SpirvWord(1));
    eprintln!("ZLUDA DEBUG: Created scoped_resolver");
    let mut scoped_resolver = ScopedResolver::new(&mut id_defs);
    eprintln!("ZLUDA DEBUG: Created sreg_map");
    let sreg_map = SpecialRegistersMap2::new(&mut scoped_resolver)?;
    eprintln!("ZLUDA DEBUG: Completed normalize_identifiers2");
    let directives = normalize_identifiers2::run(&mut scoped_resolver, ast.directives)?;
    eprintln!("ZLUDA DEBUG: Completed replace_known_functions");
    let directives = replace_known_functions::run(&mut id_defs, directives);
    eprintln!("ZLUDA DEBUG: Completed normalize_predicates2");
    let directives = normalize_predicates2::run(&mut id_defs, directives)?;
    eprintln!("ZLUDA DEBUG: Completed resolve_function_pointers");
    let directives = resolve_function_pointers::run(directives);
    eprintln!("ZLUDA DEBUG: Completed fix_special_registers2");
    let directives = fix_special_registers2::run(&mut id_defs, &sreg_map, directives.unwrap())?;
    eprintln!("ZLUDA DEBUG: Completed expand_operands");
    let directives = expand_operands::run(&mut id_defs, directives)?;
    eprintln!("ZLUDA DEBUG: Completed deparamize_functions");
    let directives = deparamize_functions::run(&mut id_defs, directives)?;
    eprintln!("ZLUDA DEBUG: Completed insert_explicit_load_store");
    let directives = insert_explicit_load_store::run(&mut id_defs, directives)?;
    eprintln!("ZLUDA DEBUG: Completed insert_implicit_conversions2");
    let directives = insert_implicit_conversions2::run(&mut id_defs, directives)?;
    eprintln!("ZLUDA DEBUG: Completed replace_instructions_with_function_calls");
    let directives = replace_instructions_with_function_calls::run(&mut id_defs, directives)?;
    eprintln!("ZLUDA DEBUG: Completed hoist_globals");
    let directives = hoist_globals::run(directives);
    eprintln!("ZLUDA DEBUG: Completed emit_llvm");
    let llvm_ir = emit_llvm::run_with_filename(id_defs, directives.unwrap(), source_filename)?;
    Ok(Module {
        llvm_ir,
        kernel_info: HashMap::new(),
    })
}

#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub arguments_sizes: Vec<(usize, bool)>,
    pub uses_shared_mem: bool,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone, EnumIter)]
enum PtxSpecialRegister {
    Tid,
    Ntid,
    Ctaid,
    Nctaid,
    Clock,
    LanemaskLt,
}

impl PtxSpecialRegister {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tid => "%tid",
            Self::Ntid => "%ntid",
            Self::Ctaid => "%ctaid",
            Self::Nctaid => "%nctaid",
            Self::Clock => "%clock",
            Self::LanemaskLt => "%lanemask_lt",
        }
    }

    fn get_type(self) -> ast::Type {
        match self {
            PtxSpecialRegister::Tid
            | PtxSpecialRegister::Ntid
            | PtxSpecialRegister::Ctaid
            | PtxSpecialRegister::Nctaid => ast::Type::Vector(4, self.get_function_return_type()),
            _ => ast::Type::Scalar(self.get_function_return_type()),
        }
    }

    fn get_function_return_type(self) -> ast::ScalarType {
        match self {
            PtxSpecialRegister::Tid => ast::ScalarType::U32,
            PtxSpecialRegister::Ntid => ast::ScalarType::U32,
            PtxSpecialRegister::Ctaid => ast::ScalarType::U32,
            PtxSpecialRegister::Nctaid => ast::ScalarType::U32,
            PtxSpecialRegister::Clock => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskLt => ast::ScalarType::U32,
        }
    }

    fn get_function_input_type(self) -> Option<ast::ScalarType> {
        match self {
            PtxSpecialRegister::Tid
            | PtxSpecialRegister::Ntid
            | PtxSpecialRegister::Ctaid
            | PtxSpecialRegister::Nctaid => Some(ast::ScalarType::U8),
            PtxSpecialRegister::Clock | PtxSpecialRegister::LanemaskLt => None,
        }
    }

    fn get_unprefixed_function_name(self) -> &'static str {
        match self {
            PtxSpecialRegister::Tid => "sreg_tid",
            PtxSpecialRegister::Ntid => "sreg_ntid",
            PtxSpecialRegister::Ctaid => "sreg_ctaid",
            PtxSpecialRegister::Nctaid => "sreg_nctaid",
            PtxSpecialRegister::Clock => "sreg_clock",
            PtxSpecialRegister::LanemaskLt => "sreg_lanemask_lt",
        }
    }
}

#[cfg(debug_assertions)]
fn error_unreachable() -> TranslateError {
    unreachable!()
}

#[cfg(not(debug_assertions))]
fn error_unreachable() -> TranslateError {
    TranslateError::UnreachableCodeError
}

#[cfg(debug_assertions)]
fn error_todo() -> TranslateError {
    unreachable!()
}

#[cfg(not(debug_assertions))]
fn error_todo() -> TranslateError {
    TranslateError::Todo
}

#[cfg(debug_assertions)]
fn error_unknown_symbol() -> TranslateError {
    panic!()
}

#[cfg(not(debug_assertions))]
fn error_unknown_symbol() -> TranslateError {
    TranslateError::UnknownSymbol
}

#[cfg(debug_assertions)]
fn error_mismatched_type() -> TranslateError {
    panic!()
}

#[cfg(not(debug_assertions))]
fn error_mismatched_type() -> TranslateError {
    TranslateError::InvalidSymbolFormat
}

#[derive(Debug, Clone)]
enum Statement<I, P: ast::Operand> {
    Label(SpirvWord),
    Variable(ast::Variable<P::Ident>),
    Instruction(I),
    // New: Predicated instruction execution
    PredicatedInstruction {
        predicate: SpirvWord,
        negated: bool,
        instruction: I,
    },
    Conversion(ImplicitConversion),
    Constant(ConstantDefinition),
    RetValue(ast::RetData, Vec<(SpirvWord, ast::Type)>),
    PtrAccess(PtrAccess<P>),
    RepackVector(RepackVectorDetails),
    FunctionPointer(FunctionPointerDetails),
    VectorRead(VectorRead),
    VectorWrite(VectorWrite),
}

impl<T: ast::Operand<Ident = SpirvWord>> Statement<ast::Instruction<T>, T> {
    fn visit_map<To: ast::Operand<Ident = SpirvWord>, Err>(
        self,
        visitor: &mut impl ast::VisitorMap<T, To, Err>,
    ) -> std::result::Result<Statement<ast::Instruction<To>, To>, Err> {
        Ok(match self {
            Statement::Instruction(i) => {
                return ast::visit_map(i, visitor).map(Statement::Instruction)
            }
            Statement::Label(label) => {
                Statement::Label(visitor.visit_ident(label, None, false, false)?)
            }
            Statement::Variable(var) => {
                let name = visitor.visit_ident(
                    var.name,
                    Some((&var.v_type, var.state_space)),
                    true,
                    false,
                )?;
                Statement::Variable(ast::Variable {
                    align: var.align,
                    v_type: var.v_type,
                    state_space: var.state_space,
                    name,
                    array_init: var.array_init,
                })
            }
            Statement::PredicatedInstruction {
                predicate,
                negated,
                instruction,
            } => {
                let predicate = visitor.visit_ident(
                    predicate,
                    Some((&ast::ScalarType::Pred.into(), ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                let instruction = ast::visit_map(instruction, visitor)?;
                Statement::PredicatedInstruction {
                    predicate,
                    negated,
                    instruction,
                }
            }
            Statement::Conversion(ImplicitConversion {
                src,
                dst,
                from_type,
                to_type,
                from_space,
                to_space,
                kind,
            }) => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((&to_type, ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(
                    src,
                    Some((&from_type, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::Conversion(ImplicitConversion {
                    src,
                    dst,
                    from_type,
                    to_type,
                    from_space,
                    to_space,
                    kind,
                })
            }
            Statement::Constant(ConstantDefinition { dst, typ, value }) => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((&typ.into(), ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                Statement::Constant(ConstantDefinition { dst, typ, value })
            }
            Statement::RetValue(data, value) => {
                let value = value
                    .into_iter()
                    .map(|(ident, type_)| {
                        Ok((
                            visitor.visit_ident(
                                ident,
                                Some((&type_, ast::StateSpace::Local)),
                                false,
                                false,
                            )?,
                            type_,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Statement::RetValue(data, value)
            }
            Statement::PtrAccess(PtrAccess {
                underlying_type,
                state_space,
                dst,
                ptr_src,
                offset_src,
            }) => {
                let dst =
                    visitor.visit_ident(dst, Some((&underlying_type, state_space)), true, false)?;
                let ptr_src = visitor.visit_ident(
                    ptr_src,
                    Some((&underlying_type, state_space)),
                    false,
                    false,
                )?;
                let offset_src = visitor.visit(
                    offset_src,
                    Some((
                        &ast::Type::Scalar(ast::ScalarType::S64),
                        ast::StateSpace::Reg,
                    )),
                    false,
                    false,
                )?;
                Statement::PtrAccess(PtrAccess {
                    underlying_type,
                    state_space,
                    dst,
                    ptr_src,
                    offset_src,
                })
            }
            Statement::VectorRead(VectorRead {
                scalar_type,
                vector_width,
                scalar_dst: dst,
                vector_src,
                member,
            }) => {
                let scalar_t = scalar_type.into();
                let vector_t = ast::Type::Vector(vector_width, scalar_type);
                let dst: SpirvWord = visitor.visit_ident(
                    dst,
                    Some((&scalar_t, ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(
                    vector_src,
                    Some((&vector_t, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::VectorRead(VectorRead {
                    scalar_type,
                    vector_width,
                    scalar_dst: dst,
                    vector_src: src,
                    member,
                })
            }
            Statement::VectorWrite(VectorWrite {
                scalar_type,
                vector_width,
                vector_dst,
                vector_src,
                scalar_src,
                member,
            }) => {
                let scalar_t = scalar_type.into();
                let vector_t = ast::Type::Vector(vector_width, scalar_type);
                let vector_dst = visitor.visit_ident(
                    vector_dst,
                    Some((&vector_t, ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let vector_src = visitor.visit_ident(
                    vector_src,
                    Some((&vector_t, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                let scalar_src = visitor.visit_ident(
                    scalar_src,
                    Some((&scalar_t, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::VectorWrite(VectorWrite {
                    vector_dst,
                    vector_src,
                    scalar_src,
                    scalar_type,
                    vector_width,
                    member,
                })
            }
            Statement::RepackVector(RepackVectorDetails {
                is_extract,
                typ,
                packed,
                unpacked,
                relaxed_type_check,
            }) => {
                let (packed, unpacked) = if is_extract {
                    let unpacked = unpacked
                        .into_iter()
                        .map(|ident| {
                            visitor.visit_ident(
                                ident,
                                Some((&typ.into(), ast::StateSpace::Reg)),
                                true,
                                relaxed_type_check,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let packed = visitor.visit_ident(
                        packed,
                        Some((
                            &ast::Type::Vector(unpacked.len() as u8, typ),
                            ast::StateSpace::Reg,
                        )),
                        false,
                        false,
                    )?;
                    (packed, unpacked)
                } else {
                    let packed = visitor.visit_ident(
                        packed,
                        Some((
                            &ast::Type::Vector(unpacked.len() as u8, typ),
                            ast::StateSpace::Reg,
                        )),
                        true,
                        false,
                    )?;
                    let unpacked = unpacked
                        .into_iter()
                        .map(|ident| {
                            visitor.visit_ident(
                                ident,
                                Some((&typ.into(), ast::StateSpace::Reg)),
                                false,
                                relaxed_type_check,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    (packed, unpacked)
                };
                Statement::RepackVector(RepackVectorDetails {
                    is_extract,
                    typ,
                    packed,
                    unpacked,
                    relaxed_type_check,
                })
            }
            Statement::FunctionPointer(FunctionPointerDetails { dst, src }) => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((
                        &ast::Type::Scalar(ast::ScalarType::U64),
                        ast::StateSpace::Reg,
                    )),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(src, None, false, false)?;
                Statement::FunctionPointer(FunctionPointerDetails { dst, src })
            }
        })
    }
}

#[derive(Debug, Clone)]
struct ImplicitConversion {
    src: SpirvWord,
    dst: SpirvWord,
    from_type: ast::Type,
    to_type: ast::Type,
    from_space: ast::StateSpace,
    to_space: ast::StateSpace,
    kind: ConversionKind,
}

#[derive(Debug, Clone, PartialEq)]
enum ConversionKind {
    Default,
    // zero-extend/chop/bitcast depending on types
    SignExtend,
    BitToPtr,
    PtrToPtr,
    AddressOf,
}

#[derive(Debug, Clone)]
struct ConstantDefinition {
    pub dst: SpirvWord,
    pub typ: ast::ScalarType,
    pub value: ast::ImmediateValue,
}

#[derive(Debug, Clone)]
pub struct PtrAccess<T> {
    pub underlying_type: ast::Type,
    pub state_space: ast::StateSpace,
    pub dst: SpirvWord,
    pub ptr_src: SpirvWord,
    pub offset_src: T,
}

#[derive(Debug, Clone)]
struct RepackVectorDetails {
    is_extract: bool,
    typ: ast::ScalarType,
    packed: SpirvWord,
    unpacked: Vec<SpirvWord>,
    relaxed_type_check: bool,
}

#[derive(Debug, Clone)]
struct FunctionPointerDetails {
    dst: SpirvWord,
    src: SpirvWord,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct SpirvWord(u32);

impl From<u32> for SpirvWord {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
impl From<SpirvWord> for u32 {
    fn from(value: SpirvWord) -> Self {
        value.0
    }
}

impl ast::Operand for SpirvWord {
    type Ident = Self;

    fn from_ident(ident: Self::Ident) -> Self {
        ident
    }
}

type ExpandedStatement = Statement<ast::Instruction<SpirvWord>, SpirvWord>;

type NormalizedStatement = Statement<
    (
        Option<ast::PredAt<SpirvWord>>,
        ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ),
    ast::ParsedOperand<SpirvWord>,
>;

#[derive(Debug, Clone)]
enum Directive2<'input, Instruction, Operand: ast::Operand> {
    Variable(ast::LinkingDirective, ast::Variable<SpirvWord>),
    Method(Function2<'input, Instruction, Operand>),
}

#[derive(Debug, Clone)]
struct Function2<'input, Instruction, Operand: ast::Operand> {
    pub func_decl: ast::MethodDeclaration<'input, SpirvWord>,
    pub globals: Vec<ast::Variable<SpirvWord>>,
    pub body: Option<Vec<Statement<Instruction, Operand>>>,
    import_as: Option<String>,
    tuning: Vec<ast::TuningDirective>,
    linkage: ast::LinkingDirective,
}

type NormalizedDirective2<'input> = Directive2<
    'input,
    (
        Option<ast::PredAt<SpirvWord>>,
        ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ),
    ast::ParsedOperand<SpirvWord>,
>;

type NormalizedFunction2<'input> = Function2<
    'input,
    (
        Option<ast::PredAt<SpirvWord>>,
        ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ),
    ast::ParsedOperand<SpirvWord>,
>;

type UnconditionalDirective<'input> = Directive2<
    'input,
    ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ast::ParsedOperand<SpirvWord>,
>;

type UnconditionalFunction<'input> = Function2<
    'input,
    ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ast::ParsedOperand<SpirvWord>,
>;

struct GlobalStringIdentResolver2<'input> {
    pub(crate) current_id: SpirvWord,
    pub(crate) ident_map: FxHashMap<SpirvWord, IdentEntry<'input>>,
}

impl<'input> GlobalStringIdentResolver2<'input> {
    fn new(spirv_word: SpirvWord) -> Self {
        Self {
            current_id: spirv_word,
            ident_map: FxHashMap::default(),
        }
    }

    fn register_named(
        &mut self,
        name: Cow<'input, str>,
        type_space: Option<(ast::Type, ast::StateSpace)>,
    ) -> SpirvWord {
        let new_id = self.current_id;
        self.ident_map.insert(
            new_id,
            IdentEntry {
                name: Some(name),
                type_space,
            },
        );
        self.current_id.0 += 1;
        new_id
    }

    fn register_unnamed(&mut self, type_space: Option<(ast::Type, ast::StateSpace)>) -> SpirvWord {
        let new_id = self.current_id;
        self.ident_map.insert(
            new_id,
            IdentEntry {
                name: None,
                type_space,
            },
        );
        self.current_id.0 += 1;
        new_id
    }

    fn get_typed(&self, id: SpirvWord) -> Result<&(ast::Type, ast::StateSpace), TranslateError> {
        match self.ident_map.get(&id) {
            Some(IdentEntry {
                type_space: Some(type_space),
                ..
            }) => Ok(type_space),
            _ => Err(error_unknown_symbol()),
        }
    }
}

struct IdentEntry<'input> {
    name: Option<Cow<'input, str>>,
    type_space: Option<(ast::Type, ast::StateSpace)>,
}

struct ScopedResolver<'input, 'b> {
    flat_resolver: &'b mut GlobalStringIdentResolver2<'input>,
    scopes: Vec<ScopeMarker<'input>>,
}

impl<'input, 'b> ScopedResolver<'input, 'b> {
    fn new(flat_resolver: &'b mut GlobalStringIdentResolver2<'input>) -> Self {
        Self {
            flat_resolver,
            scopes: vec![ScopeMarker::new()],
        }
    }

    fn start_scope(&mut self) {
        self.scopes.push(ScopeMarker::new());
    }

    fn end_scope(&mut self) {
        let scope = self.scopes.pop().unwrap();
        scope.flush(self.flat_resolver);
    }

    fn add_or_get_in_current_scope_untyped(
        &mut self,
        name: &'input str,
    ) -> Result<SpirvWord, TranslateError> {
        let current_scope = self.scopes.last_mut().unwrap();
        Ok(
            match current_scope.name_to_ident.entry(Cow::Borrowed(name)) {
                hash_map::Entry::Occupied(occupied_entry) => {
                    let ident = *occupied_entry.get();
                    let entry = current_scope
                        .ident_map
                        .get(&ident)
                        .ok_or_else(|| error_unreachable())?;
                    if entry.type_space.is_some() {
                        return Err(error_unknown_symbol());
                    }
                    ident
                }
                hash_map::Entry::Vacant(vacant_entry) => {
                    let new_id = self.flat_resolver.current_id;
                    self.flat_resolver.current_id.0 += 1;
                    vacant_entry.insert(new_id);
                    current_scope.ident_map.insert(
                        new_id,
                        IdentEntry {
                            name: Some(Cow::Borrowed(name)),
                            type_space: None,
                        },
                    );
                    new_id
                }
            },
        )
    }

    fn add(
        &mut self,
        name: Cow<'input, str>,
        type_space: Option<(ast::Type, ast::StateSpace)>,
    ) -> Result<SpirvWord, TranslateError> {
        let result = self.flat_resolver.current_id;
        self.flat_resolver.current_id.0 += 1;
        let current_scope = self.scopes.last_mut().unwrap();
        if current_scope
            .name_to_ident
            .insert(name.clone(), result)
            .is_some()
        {
            return Err(error_unknown_symbol());
        }
        current_scope.ident_map.insert(
            result,
            IdentEntry {
                name: Some(name),
                type_space,
            },
        );
        Ok(result)
    }

    fn get(&mut self, name: &str) -> Result<SpirvWord, TranslateError> {
        self.scopes
            .iter()
            .rev()
            .find_map(|resolver| resolver.name_to_ident.get(name).copied())
            .ok_or_else(|| error_unreachable())
    }

    fn get_in_current_scope(&self, label: &'input str) -> Result<SpirvWord, TranslateError> {
        let current_scope = self.scopes.last().unwrap();
        current_scope
            .name_to_ident
            .get(label)
            .copied()
            .ok_or_else(|| error_unreachable())
    }
}

struct ScopeMarker<'input> {
    ident_map: FxHashMap<SpirvWord, IdentEntry<'input>>,
    name_to_ident: FxHashMap<Cow<'input, str>, SpirvWord>,
}

impl<'input> ScopeMarker<'input> {
    fn new() -> Self {
        Self {
            ident_map: FxHashMap::default(),
            name_to_ident: FxHashMap::default(),
        }
    }

    fn flush(self, resolver: &mut GlobalStringIdentResolver2<'input>) {
        resolver.ident_map.extend(self.ident_map);
    }
}

struct SpecialRegistersMap2 {
    reg_to_id: FxHashMap<PtxSpecialRegister, SpirvWord>,
    id_to_reg: FxHashMap<SpirvWord, PtxSpecialRegister>,
}

impl SpecialRegistersMap2 {
    fn new(resolver: &mut ScopedResolver) -> Result<Self, TranslateError> {
        let mut result = SpecialRegistersMap2 {
            reg_to_id: FxHashMap::default(),
            id_to_reg: FxHashMap::default(),
        };
        for sreg in PtxSpecialRegister::iter() {
            let text = sreg.as_str();
            let id = resolver.add(
                Cow::Borrowed(text),
                Some((sreg.get_type(), ast::StateSpace::Reg)),
            )?;
            result.reg_to_id.insert(sreg, id);
            result.id_to_reg.insert(id, sreg);
        }
        Ok(result)
    }

    fn get(&self, id: SpirvWord) -> Option<PtxSpecialRegister> {
        self.id_to_reg.get(&id).copied()
    }

    fn generate_declarations<'a, 'input>(
        resolver: &'a mut GlobalStringIdentResolver2<'input>,
    ) -> impl ExactSizeIterator<
        Item = (
            PtxSpecialRegister,
            ast::MethodDeclaration<'input, SpirvWord>,
        ),
    > + 'a {
        PtxSpecialRegister::iter().map(|sreg| {
            let external_fn_name = [ZLUDA_PTX_PREFIX, sreg.get_unprefixed_function_name()].concat();
            let name =
                ast::MethodName::Func(resolver.register_named(Cow::Owned(external_fn_name), None));
            let return_type = sreg.get_function_return_type();
            let input_type = sreg.get_function_input_type();
            (
                sreg,
                ast::MethodDeclaration {
                    return_arguments: vec![ast::Variable {
                        align: None,
                        v_type: return_type.into(),
                        state_space: ast::StateSpace::Reg,
                        name: resolver
                            .register_unnamed(Some((return_type.into(), ast::StateSpace::Reg))),
                        array_init: Vec::new(),
                    }],
                    name: name,
                    input_arguments: input_type
                        .into_iter()
                        .map(|type_| ast::Variable {
                            align: None,
                            v_type: type_.into(),
                            state_space: ast::StateSpace::Reg,
                            name: resolver
                                .register_unnamed(Some((type_.into(), ast::StateSpace::Reg))),
                            array_init: Vec::new(),
                        })
                        .collect::<Vec<_>>(),
                    shared_mem: None,
                },
            )
        })
    }
}

#[derive(Debug, Clone)]
pub struct VectorRead {
    pub scalar_type: ast::ScalarType,
    pub vector_width: u8,
    pub scalar_dst: SpirvWord,
    pub vector_src: SpirvWord,
    pub member: u8,
}

#[derive(Debug, Clone)]
pub struct VectorWrite {
    pub scalar_type: ast::ScalarType,
    pub vector_width: u8,
    pub vector_dst: SpirvWord,
    pub vector_src: SpirvWord,
    pub scalar_src: SpirvWord,
    pub member: u8,
}

fn scalar_to_ptx_name(this: ast::ScalarType) -> &'static str {
    match this {
        ast::ScalarType::B8 => "b8",
        ast::ScalarType::B16 => "b16",
        ast::ScalarType::B32 => "b32",
        ast::ScalarType::B64 => "b64",
        ast::ScalarType::B128 => "b128",
        ast::ScalarType::U8 => "u8",
        ast::ScalarType::U16 => "u16",
        ast::ScalarType::U16x2 => "u16x2",
        ast::ScalarType::U32 => "u32",
        ast::ScalarType::U64 => "u64",
        ast::ScalarType::S8 => "s8",
        ast::ScalarType::S16 => "s16",
        ast::ScalarType::S16x2 => "s16x2",
        ast::ScalarType::S32 => "s32",
        ast::ScalarType::S64 => "s64",
        ast::ScalarType::F16 => "f16",
        ast::ScalarType::F16x2 => "f16x2",
        ast::ScalarType::F32 => "f32",
        ast::ScalarType::F64 => "f64",
        ast::ScalarType::BF16 => "bf16",
        ast::ScalarType::BF16x2 => "bf16x2",
        ast::ScalarType::Pred => "pred",
    }
}

type UnconditionalStatement =
    Statement<ast::Instruction<ast::ParsedOperand<SpirvWord>>, ast::ParsedOperand<SpirvWord>>;

impl From<SpirvWord> for String {
    fn from(word: SpirvWord) -> Self {
        format!("_{}", word.0)
    }
}

impl std::fmt::Display for SpirvWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for SpirvWord {
    fn as_ref(&self) -> &str {
        // This is a bit of a hack since we can't actually return a reference
        // to the formatted string, we'll use a thread-local static string
        thread_local! {
            static THREAD_LOCAL_BUFFER: std::cell::RefCell<String> = std::cell::RefCell::new(String::new());
        }

        THREAD_LOCAL_BUFFER.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            buffer.clear();
            write!(buffer, "_{}", self.0).unwrap();
            // This is unsafe because we're returning a reference to a string that might change
            // if the same thread accesses the same thread-local storage before this reference
            // is used. For our purposes this should be safe enough since we're only using it
            // immediately for lookups.
            unsafe { std::mem::transmute::<&str, &str>(&buffer) }
        })
    }
}
