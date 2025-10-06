use ptx_parser as ast;
use quick_error::quick_error;
use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::{
    borrow::Cow,
    collections::{hash_map, HashMap},
    ffi::CString,
    fmt::Write,
    iter,
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

pub(crate) mod debug_integration;
mod deparamize_functions;
pub(crate) mod emit_llvm;
pub(crate) mod emit_tosa_mlir;
mod expand_operands;
mod fix_special_registers;
mod fix_special_registers2;
mod hoist_globals;
mod insert_explicit_load_store;
mod insert_implicit_conversions2;
pub(crate) mod mlir_debug_framework;
pub(crate) mod mlir_debugger_integration;
mod insert_post_saturation;
mod instruction_mode_to_global_mode;
pub mod llvm;
mod normalize_basic_blocks;
mod normalize_identifiers;
mod normalize_identifiers2;
mod normalize_predicates2;
mod remove_unreachable_basic_blocks;
mod replace_instructions_with_function_calls;
mod replace_instructions_with_functions;
mod replace_instructions_with_functions_fp_required;
mod replace_known_functions;
mod resolve_function_pointers;

#[cfg(test)]
mod test;

#[cfg(feature = "amd")]
static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../../lib/zluda_ptx_impl.bc");
#[cfg(feature = "intel")]
static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../../lib/zluda_ptx_ze_impl.bc");
#[cfg(not(any(feature = "amd", feature = "intel")))]
static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../../lib/zluda_ptx_impl.bc");
const ZLUDA_PTX_PREFIX: &'static str = "__zluda_ptx_impl_";

quick_error! {
    #[derive(Debug, strum_macros::AsRefStr)]
    pub enum TranslateError {
        UnknownSymbol(symbol: String) {
            display("Unknown symbol: \"{}\"", symbol)
        }
        UntypedSymbol {}
        MismatchedType {}
        Unreachable {}
        Todo(msg: String) {
            display("TODO: {}", msg)
        }
        UnexpectedError(msg: String) {
            display("Unexpected error: {}", msg)
        }
    }
}

/// GPU attributes needed at compile time.
#[derive(Copy, Clone)]
pub struct Attributes {
    /// Clock frequency in kHz.
    pub clock_rate: u32,
}

pub fn to_llvm_module<'input>(
    ast: ast::Module<'input>,
    attributes: Attributes,
    mut on_pass_end: impl FnMut(&str),
) -> Result<Module, TranslateError> {
    let mut flat_resolver = GlobalStringIdentResolver2::<'input>::new(SpirvWord(1));
    let mut scoped_resolver = ScopedResolver::new(&mut flat_resolver);
    let sreg_map = SpecialRegistersMap::new(&mut scoped_resolver)?;
    let directives = normalize_identifiers::run(&mut scoped_resolver, ast.directives)?;
    on_pass_end("normalize_identifiers");
    let directives = replace_known_functions::run(&mut flat_resolver, directives);
    on_pass_end("replace_known_functions");
    let directives = normalize_predicates2::run(&mut flat_resolver, directives)?;
    on_pass_end("normalize_predicates2");
    let directives = resolve_function_pointers::run(directives)?;
    on_pass_end("resolve_function_pointers");
    let directives = fix_special_registers::run(&mut flat_resolver, &sreg_map, directives)?;
    on_pass_end("fix_special_registers");
    let directives = expand_operands::run(&mut flat_resolver, directives)?;
    on_pass_end("expand_operands");
    let directives = insert_post_saturation::run(&mut flat_resolver, directives)?;
    on_pass_end("insert_post_saturation");
    let directives = deparamize_functions::run(&mut flat_resolver, directives)?;
    on_pass_end("deparamize_functions");
    let directives =
        replace_instructions_with_functions_fp_required::run(&mut flat_resolver, directives)?;
    on_pass_end("replace_instructions_with_functions_fp_required");
    let directives = normalize_basic_blocks::run(&mut flat_resolver, directives)?;
    on_pass_end("normalize_basic_blocks");
    let directives = remove_unreachable_basic_blocks::run(directives)?;
    on_pass_end("remove_unreachable_basic_blocks");
    let directives = instruction_mode_to_global_mode::run(&mut flat_resolver, directives)?;
    on_pass_end("instruction_mode_to_global_mode");
    let directives = insert_explicit_load_store::run(&mut flat_resolver, directives)?;
    on_pass_end("insert_explicit_load_store");
    let directives = insert_implicit_conversions2::run(&mut flat_resolver, directives)?;
    on_pass_end("insert_implicit_conversions2");
    let directives = replace_instructions_with_functions::run(&mut flat_resolver, directives)?;
    on_pass_end("replace_instructions_with_functions");
    let directives = hoist_globals::run(directives)?;
    on_pass_end("hoist_globals");
    let context = llvm::Context::new();
    let llvm_ir = llvm::emit::run(&context, flat_resolver, directives)?;
    let attributes_ir = llvm::attributes::run(&context, attributes)?;
    on_pass_end("emit_llvm");
    Ok(Module {
        llvm_ir,
        attributes_ir,
        kernel_info: HashMap::new(),
        _context: context,
    })
}

pub struct Module {
    pub llvm_ir: llvm::Module,
    pub attributes_ir: llvm::Module,
    pub kernel_info: HashMap<String, KernelInfo>,
    _context: llvm::Context,
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
        let bitcode = self.llvm_ir.write_bitcode_to_memory();
        fs::write(temp_bc_path, &*bitcode)
            .map_err(|e| format!("Failed to write temporary bitcode file: {}", e))?;

        // Use llvm-dis to convert the bitcode to text
        // Try to use the built LLVM tools first, fall back to system
        let llvm_dis_cmd = option_env!("LLVM_DIS_PATH")
            .unwrap_or("llvm-dis");

        let llvm_dis_output = Command::new(&llvm_dis_cmd)
            .arg(temp_bc_path)
            .arg("-o")
            .arg(temp_ll_path)
            .output()
            .map_err(|e| format!("Failed to execute llvm-dis ({}): {}", llvm_dis_cmd, e))?;

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
    let llvm_module = to_llvm_module(ast, Attributes { clock_rate: 2124000 }, |_| {})?;

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
    let bitcode_buf = llvm_module.llvm_ir.write_bitcode_to_memory();
    std::fs::write(&bitcode_path, &*bitcode_buf).map_err(|e| {
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
        let bitcode_data = &*bitcode_buf;
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
        // Try to use the built LLVM tools first, fall back to system
        let llc_cmd = option_env!("LLC_PATH")
            .unwrap_or("llc");

        let ptx_temp_path = format!("/tmp/zluda_ptx_{}.ptx", timestamp);
        let llc_output = std::process::Command::new(&llc_cmd)
            .args(&[
                "-march=nvptx64",
                "-mcpu=sm_61", // Use newer compute capability for better debug support
                "-filetype=asm", // Generate assembly (PTX)
                "-O0", // No optimization to preserve debug info
                &llvm_temp_path,
                "-o",
                &ptx_temp_path,
            ])
            .output()
            .map_err(|e| {
                TranslateError::UnexpectedError(format!("Failed to execute llc ({}): {}", llc_cmd, e))
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
    let bitcode_buf = llvm_module.llvm_ir.write_bitcode_to_memory();
    std::fs::write(&bitcode_path, &*bitcode_buf).map_err(|e| {
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
        let bitcode_data = &*bitcode_buf;
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
        // Try to use the built LLVM tools first, fall back to system
        let llc_cmd = option_env!("LLC_PATH")
            .unwrap_or("llc");

        let ptx_temp_path = format!("/tmp/zluda_ptx_{}.ptx", timestamp);
        let llc_output = std::process::Command::new(&llc_cmd)
            .args(&[
                "-march=nvptx64",
                "-mcpu=sm_61", // Use newer compute capability for better debug support
                "-filetype=asm", // Generate assembly (PTX)
                "-O0", // No optimization to preserve debug info
                &llvm_temp_path,
                "-o",
                &ptx_temp_path,
            ])
            .output()
            .map_err(|e| {
                TranslateError::UnexpectedError(format!("Failed to execute llc ({}): {}", llc_cmd, e))
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
/// This is now just a wrapper around to_llvm_module with default attributes
pub fn to_llvm_module_with_filename<'input>(
    ast: ast::Module<'input>,
    _source_filename: &str,
) -> Result<Module, TranslateError> {
    // For now, just use the regular compilation path with default attributes
    // TODO: Pass source_filename to the debug info generation
    to_llvm_module(ast, Attributes { clock_rate: 2124000 }, |_| {})
}

#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub arguments_sizes: Vec<(usize, bool)>,
    pub uses_shared_mem: bool,
}

/// Convert PTX to MLIR (TOSA dialect)
pub fn to_mlir_module<'input>(_ast: ast::Module<'input>) -> Result<String, TranslateError> {
    Err(TranslateError::Todo(
        "to_mlir_module - MLIR backend requires additional passes that are not yet integrated".to_string(),
    ))
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone, EnumIter)]
enum PtxSpecialRegister {
    Tid,
    Ntid,
    Ctaid,
    Nctaid,
    Clock,
    LanemaskLt,
    LanemaskGe,
    Laneid,
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
            Self::LanemaskGe => "%lanemask_ge",
            Self::Laneid => "%laneid",
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
            PtxSpecialRegister::LanemaskGe => ast::ScalarType::U32,
            PtxSpecialRegister::Laneid => ast::ScalarType::U32,
        }
    }

    fn get_function_input_type(self) -> Option<ast::ScalarType> {
        match self {
            PtxSpecialRegister::Tid
            | PtxSpecialRegister::Ntid
            | PtxSpecialRegister::Ctaid
            | PtxSpecialRegister::Nctaid => Some(ast::ScalarType::U8),
            PtxSpecialRegister::Clock
            | PtxSpecialRegister::LanemaskLt
            | PtxSpecialRegister::LanemaskGe
            | PtxSpecialRegister::Laneid => None,
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
            PtxSpecialRegister::LanemaskGe => "sreg_lanemask_ge",
            PtxSpecialRegister::Laneid => "sreg_laneid",
        }
    }
}

#[cfg(debug_assertions)]
fn error_unreachable() -> TranslateError {
    unreachable!()
}

#[cfg(not(debug_assertions))]
fn error_unreachable() -> TranslateError {
    TranslateError::Unreachable
}

#[cfg(debug_assertions)]
fn error_todo_msg<T: Into<String>>(msg: T) -> TranslateError {
    unreachable!("{}", msg.into())
}

#[cfg(not(debug_assertions))]
fn error_todo_msg<T: Into<String>>(msg: T) -> TranslateError {
    TranslateError::Todo(msg.into())
}

#[cfg(debug_assertions)]
fn error_todo() -> TranslateError {
    unreachable!()
}

#[cfg(not(debug_assertions))]
fn error_todo() -> TranslateError {
    TranslateError::Todo("".to_string())
}

#[cfg(debug_assertions)]
fn error_unknown_symbol<T: Into<String>>(symbol: T) -> TranslateError {
    panic!("Unknown symbol: \"{}\"", symbol.into())
}

#[cfg(not(debug_assertions))]
fn error_unknown_symbol<T: Into<String>>(symbol: T) -> TranslateError {
    TranslateError::UnknownSymbol(symbol.into())
}

#[cfg(debug_assertions)]
fn error_mismatched_type() -> TranslateError {
    panic!("Mismatched type")
}

#[cfg(not(debug_assertions))]
fn error_mismatched_type() -> TranslateError {
    TranslateError::MismatchedType
}

#[derive(Debug)]
enum Statement<I, P: ast::Operand> {
    Label(SpirvWord),
    Variable(ast::Variable<P::Ident>),
    Instruction(I),
    // SPIR-V compatible replacement for PTX predicates
    Conditional(BrachCondition),
    Conversion(ImplicitConversion),
    Constant(ConstantDefinition),
    RetValue(ast::RetData, Vec<(SpirvWord, ast::Type)>),
    PtrAccess(PtrAccess<P>),
    RepackVector(RepackVectorDetails),
    FunctionPointer(FunctionPointerDetails),
    VectorRead(VectorRead),
    VectorWrite(VectorWrite),
    SetMode(ModeRegister),
    // This instruction is a nop, it serves as a marker to indicate that the
    // next instruction requires certain floating-point modes to be set.
    // Some transcendentals compile to a sequence of instructions that
    // require certain modes to be set _mid-function_.
    // See replace_instructions_with_functions_fp_required pass for details
    FpModeRequired {
        ftz_f32: Option<bool>,
        rnd_f32: Option<ast::RoundingMode>,
    },
    FpSaturate {
        dst: SpirvWord,
        src: SpirvWord,
        type_: ast::ScalarType,
    },
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
enum ModeRegister {
    Denormal {
        f32: bool,
        f16f64: bool,
    },
    Rounding {
        f32: ast::RoundingMode,
        f16f64: ast::RoundingMode,
    },
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
                    Some((&var.info.v_type, var.info.state_space)),
                    true,
                    false,
                )?;
                Statement::Variable(ast::Variable {
                    info: ast::VariableInfo {
                        align: var.info.align,
                        v_type: var.info.v_type,
                        state_space: var.info.state_space,
                        array_init: var.info.array_init,
                    },
                    name,
                })
            }
            Statement::Conditional(conditional) => {
                let predicate = visitor.visit_ident(
                    conditional.predicate,
                    Some((&ast::ScalarType::Pred.into(), ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                let if_true = visitor.visit_ident(conditional.if_true, None, false, false)?;
                let if_false = visitor.visit_ident(conditional.if_false, None, false, false)?;
                Statement::Conditional(BrachCondition {
                    predicate,
                    if_true,
                    if_false,
                })
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
            Statement::SetMode(mode_register) => Statement::SetMode(mode_register),
            Statement::FpSaturate { dst, src, type_ } => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((&type_.into(), ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(
                    src,
                    Some((&type_.into(), ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::FpSaturate { dst, src, type_ }
            }
            Statement::FpModeRequired { ftz_f32, rnd_f32 } => {
                Statement::FpModeRequired { ftz_f32, rnd_f32 }
            }
        })
    }
}

#[derive(Debug)]
struct BrachCondition {
    predicate: SpirvWord,
    if_true: SpirvWord,
    if_false: SpirvWord,
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

impl std::fmt::Display for ImplicitConversion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "zluda.convert_implicit{}{}{}{}{}",
            self.kind, self.to_space, self.to_type, self.from_space, self.from_type
        )
    }
}

#[derive(Debug, PartialEq, Clone, strum_macros::Display)]
#[strum(serialize_all = "snake_case", prefix = ".")]
enum ConversionKind {
    Default,
    // zero-extend/chop/bitcast depending on types
    SignExtend,
    BitToPtr,
    PtrToPtr,
    AddressOf,
}

#[derive(Debug)]
struct ConstantDefinition {
    pub dst: SpirvWord,
    pub typ: ast::ScalarType,
    pub value: ast::ImmediateValue,
}

impl std::fmt::Display for ConstantDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "zluda.constant{} {}", self.typ, self.value)
    }
}

#[derive(Debug)]
pub struct PtrAccess<T> {
    underlying_type: ast::Type,
    state_space: ast::StateSpace,
    dst: SpirvWord,
    ptr_src: SpirvWord,
    offset_src: T,
}

#[derive(Debug)]
struct RepackVectorDetails {
    is_extract: bool,
    typ: ast::ScalarType,
    packed: SpirvWord,
    unpacked: Vec<SpirvWord>,
    relaxed_type_check: bool,
}

impl std::fmt::Display for RepackVectorDetails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let extract = if self.is_extract {
            ".extract"
        } else {
            ".composite"
        };
        let relaxed = if self.relaxed_type_check {
            ".relaxed"
        } else {
            ""
        };
        write!(f, "zluda.repack_vector{}{}{}", extract, relaxed, self.typ)
    }
}

#[derive(Debug)]
struct FunctionPointerDetails {
    dst: SpirvWord,
    src: SpirvWord,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct SpirvWord(u32);

impl std::fmt::Display for SpirvWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

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

enum Directive2<Instruction, Operand: ast::Operand> {
    Variable(ast::LinkingDirective, ast::Variable<SpirvWord>),
    Method(Function2<Instruction, Operand>),
}

struct Function2<Instruction, Operand: ast::Operand> {
    pub return_arguments: Vec<ast::Variable<Operand::Ident>>,
    pub name: Operand::Ident,
    pub input_arguments: Vec<ast::Variable<Operand::Ident>>,
    pub body: Option<Vec<Statement<Instruction, Operand>>>,
    is_kernel: bool,
    import_as: Option<String>,
    tuning: Vec<ast::TuningDirective>,
    linkage: ast::LinkingDirective,
    flush_to_zero_f32: bool,
    flush_to_zero_f16f64: bool,
    rounding_mode_f32: ast::RoundingMode,
    rounding_mode_f16f64: ast::RoundingMode,
}

type NormalizedDirective2 = Directive2<
    (
        Option<ast::PredAt<SpirvWord>>,
        ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ),
    ast::ParsedOperand<SpirvWord>,
>;

type NormalizedFunction2 = Function2<
    (
        Option<ast::PredAt<SpirvWord>>,
        ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ),
    ast::ParsedOperand<SpirvWord>,
>;

type UnconditionalDirective =
    Directive2<ast::Instruction<ast::ParsedOperand<SpirvWord>>, ast::ParsedOperand<SpirvWord>>;

type UnconditionalFunction =
    Function2<ast::Instruction<ast::ParsedOperand<SpirvWord>>, ast::ParsedOperand<SpirvWord>>;

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
            _ => Err(error_unknown_symbol(format!("{:?}", id))),
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
                        return Err(error_unknown_symbol(name));
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
            return Err(error_unknown_symbol(name));
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
            .ok_or_else(|| error_unknown_symbol(name))
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

struct SpecialRegistersMap {
    reg_to_id: FxHashMap<PtxSpecialRegister, SpirvWord>,
    id_to_reg: FxHashMap<SpirvWord, PtxSpecialRegister>,
}

impl SpecialRegistersMap {
    fn new(resolver: &mut ScopedResolver) -> Result<Self, TranslateError> {
        let mut result = SpecialRegistersMap {
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

    fn len() -> usize {
        PtxSpecialRegister::iter().len()
    }

    fn foreach_declaration<'a, 'input>(
        resolver: &'a mut GlobalStringIdentResolver2<'input>,
        mut fn_: impl FnMut(
            PtxSpecialRegister,
            (
                Vec<ast::Variable<SpirvWord>>,
                SpirvWord,
                Vec<ast::Variable<SpirvWord>>,
            ),
        ),
    ) {
        for sreg in PtxSpecialRegister::iter() {
            let external_fn_name = [ZLUDA_PTX_PREFIX, sreg.get_unprefixed_function_name()].concat();
            let name = resolver.register_named(Cow::Owned(external_fn_name), None);
            let return_type = sreg.get_function_return_type();
            let input_type = sreg.get_function_input_type();
            let return_arguments = vec![ast::Variable {
                info: ast::VariableInfo {
                    align: None,
                    v_type: return_type.into(),
                    state_space: ast::StateSpace::Reg,
                    array_init: Vec::new(),
                },
                name: resolver.register_unnamed(Some((return_type.into(), ast::StateSpace::Reg))),
            }];
            let input_arguments = input_type
                .into_iter()
                .map(|type_| ast::Variable {
                    info: ast::VariableInfo {
                        align: None,
                        v_type: type_.into(),
                        state_space: ast::StateSpace::Reg,
                        array_init: Vec::new(),
                    },
                    name: resolver.register_unnamed(Some((type_.into(), ast::StateSpace::Reg))),
                })
                .collect::<Vec<_>>();
            fn_(sreg, (return_arguments, name, input_arguments));
        }
    }

    fn generate_declarations<'input>(
        resolver: &mut GlobalStringIdentResolver2<'input>,
    ) -> Vec<(PtxSpecialRegister, (Vec<ast::Variable<SpirvWord>>, SpirvWord, Vec<ast::Variable<SpirvWord>>))> {
        let mut result = Vec::new();
        Self::foreach_declaration(resolver, |sreg, decl| {
            result.push((sreg, decl));
        });
        result
    }
}

#[derive(Debug)]
pub struct VectorRead {
    scalar_type: ast::ScalarType,
    vector_width: u8,
    scalar_dst: SpirvWord,
    vector_src: SpirvWord,
    member: u8,
}

#[derive(Debug)]
pub struct VectorWrite {
    scalar_type: ast::ScalarType,
    vector_width: u8,
    vector_dst: SpirvWord,
    vector_src: SpirvWord,
    scalar_src: SpirvWord,
    member: u8,
}

type SpecialRegistersMap2 = SpecialRegistersMap;

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
        ast::ScalarType::E4m3x2 => "e4m3x2",
        ast::ScalarType::E5m2x2 => "e5m2x2",
    }
}

type UnconditionalStatement =
    Statement<ast::Instruction<ast::ParsedOperand<SpirvWord>>, ast::ParsedOperand<SpirvWord>>;

impl From<SpirvWord> for String {
    fn from(word: SpirvWord) -> Self {
        format!("_{}", word.0)
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
