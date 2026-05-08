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

pub(crate) mod debug_integration;
mod deparamize_functions;
pub(crate) mod emit_llvm;
pub mod emit_pacc_vcix;
pub mod emit_tmatmul_asm;
pub(crate) mod emit_tosa_mlir;
mod expand_operands;
mod fix_special_registers;
mod fix_special_registers2;
mod hoist_globals;
mod insert_explicit_load_store;
mod insert_implicit_conversions2;
mod insert_post_saturation;
mod instruction_mode_to_global_mode;
pub mod llvm;
pub(crate) mod mlir_debug_framework;
pub(crate) mod mlir_debugger_integration;
mod normalize_basic_blocks;
mod normalize_identifiers;
mod normalize_identifiers2;
mod normalize_predicates2;
pub mod ptx_to_tmatmul;
mod remove_unreachable_basic_blocks;
mod replace_instructions_with_function_calls;
mod replace_instructions_with_functions;
mod replace_instructions_with_functions_fp_required;
mod replace_known_functions;
mod resolve_function_pointers;
pub mod tmatmul_algorithm_tree;

#[cfg(test)]
mod test;

#[cfg(feature = "amd")]
static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../../lib/zluda_ptx_impl.bc");
#[cfg(feature = "intel")]
static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../../lib/zluda_ptx_ze_impl.bc");
#[cfg(not(any(feature = "amd", feature = "intel")))]
static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../../lib/zluda_ptx_impl.bc");
const ZLUDA_PTX_PREFIX: &'static str = "__zluda_ptx_impl_";

/// `ggml_*` and `micro_kernel_*` symbols are implemented by the linked PACC
/// operator bitcode compiled from the llama.cpp-side operator sources. Keep
/// their names stable so the later ELF/device-side link step can resolve them
/// directly instead of rewriting them into the generic ZLUDA helper namespace.
pub(crate) fn is_passthrough_external_symbol(name: &str) -> bool {
    name.starts_with("ggml_") || name.starts_with("micro_kernel_")
}

pub(crate) fn lower_external_symbol_name<'input>(name: Cow<'input, str>) -> Cow<'input, str> {
    if is_passthrough_external_symbol(name.as_ref()) {
        name
    } else {
        Cow::Owned(format!("{}{}", ZLUDA_PTX_PREFIX, name))
    }
}

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
    /// Generate DWARF debug information
    pub emit_debug_info: bool,
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
    #[cfg(feature = "pacc")]
    let directives = {
        on_pass_end("instruction_mode_to_global_mode(skipped_pacc)");
        directives
    };
    #[cfg(not(feature = "pacc"))]
    let directives = {
        let directives = instruction_mode_to_global_mode::run(&mut flat_resolver, directives)?;
        on_pass_end("instruction_mode_to_global_mode");
        directives
    };
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
    #[cfg(feature = "pacc")]
    llvm_ir.force_all_function_call_conv(0);
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
        let llvm_dis_cmd = option_env!("LLVM_DIS_PATH").unwrap_or("llvm-dis");

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

/// Inject minimal debug metadata into LLVM IR to enable PTX debug directives
fn inject_minimal_debug_metadata(llvm_ir: &str) -> String {
    let mut output = String::with_capacity(llvm_ir.len() + 2000);

    // Find where functions are defined and add !dbg references
    let mut in_function = false;
    let mut func_count = 0;

    for line in llvm_ir.lines() {
        // Detect function definitions and add !dbg metadata
        if line.contains("define ") && line.contains("@") {
            let modified_line = if line.trim_end().ends_with('{') {
                line.replace('{', &format!("!dbg !{} {{", func_count + 4))
            } else {
                format!("{} !dbg !{}", line, func_count + 4)
            };
            output.push_str(&modified_line);
            output.push('\n');
            in_function = true;
            func_count += 1;
        } else {
            output.push_str(line);
            output.push('\n');
        }
    }

    // Append debug metadata at the end
    output.push_str("\n!llvm.dbg.cu = !{!0}\n");
    output.push_str("!llvm.module.flags = !{!2, !3}\n\n");
    output.push_str("!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: \"hetGPU PTX Compiler\", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)\n");
    output.push_str("!1 = !DIFile(filename: \"kernel.ptx\", directory: \".\")\n");
    output.push_str("!2 = !{i32 2, !\"Dwarf Version\", i32 2}\n");
    output.push_str("!3 = !{i32 2, !\"Debug Info Version\", i32 3}\n");

    // Add subprogram metadata for each function
    for i in 0..func_count {
        output.push_str(&format!("!{} = distinct !DISubprogram(name: \"kernel_{}\", scope: !1, file: !1, line: {}, type: !{}, scopeLine: {}, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)\n",
            i + 4, i, i + 1, i + 100, i + 1));
        output.push_str(&format!("!{} = !DISubroutineType(types: !{{}})\n", i + 100));
    }

    output
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
    let llvm_module = to_llvm_module(
        ast,
        Attributes {
            clock_rate: 2124000,
            emit_debug_info: true,
        },
        |_| {},
    )?;

    // Get the LLVM IR as text for debugging
    let mut llvm_ir_text = llvm_module.print_to_string().map_err(|e| {
        TranslateError::UnexpectedError(format!("Failed to convert LLVM to string: {}", e))
    })?;

    // Inject minimal debug metadata to enable .loc/.file directives in PTX
    // TODO: Proper debug info generation in emit_llvm
    llvm_ir_text = inject_minimal_debug_metadata(&llvm_ir_text);

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
            bitcode_data.as_ptr().cast(),
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
        let llc_cmd = option_env!("LLC_PATH").unwrap_or("llc");

        let ptx_temp_path = format!("/tmp/zluda_ptx_{}.ptx", timestamp);
        let llc_output = std::process::Command::new(&llc_cmd)
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
                TranslateError::UnexpectedError(format!(
                    "Failed to execute llc ({}): {}",
                    llc_cmd, e
                ))
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
            bitcode_data.as_ptr().cast(),
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
        let llc_cmd = option_env!("LLC_PATH").unwrap_or("llc");

        let ptx_temp_path = format!("/tmp/zluda_ptx_{}.ptx", timestamp);
        let llc_output = std::process::Command::new(&llc_cmd)
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
                TranslateError::UnexpectedError(format!(
                    "Failed to execute llc ({}): {}",
                    llc_cmd, e
                ))
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
    to_llvm_module(
        ast,
        Attributes {
            clock_rate: 2124000,
            emit_debug_info: false,
        },
        |_| {},
    )
}

#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub arguments_sizes: Vec<(usize, bool)>,
    pub uses_shared_mem: bool,
}

/// Convert PTX to MLIR (TOSA dialect)
pub fn to_mlir_module<'input>(_ast: ast::Module<'input>) -> Result<String, TranslateError> {
    Err(TranslateError::Todo(
        "to_mlir_module - MLIR backend requires additional passes that are not yet integrated"
            .to_string(),
    ))
}

/// Convert PTX AST to TMatmul assembly - HIGH-LEVEL API
///
/// This is the main entry point for compiling PTX to TMatmul assembly.
/// It handles the complete compilation pipeline.
///
/// # Example
/// ```no_run
/// use ptx_parser;
/// use ptx::pass;
///
/// let ptx_source = r#"
///     .visible .entry kernel(.param .u64 input) {
///         .reg .f32 %r1, %r2;
///         ld.param.u64 %r1, [input];
///         add.f32 %r2, %r1, %r1;
///         ret;
///     }
/// "#;
///
/// let assembly = pass::ptx_to_tmatmul_assembly(ptx_source).unwrap();
/// println!("{}", assembly);
/// ```
pub fn ptx_to_tmatmul_assembly(ptx_source: &str) -> Result<String, TranslateError> {
    ptx_to_tmatmul::ptx_to_tmatmul(ptx_source).map_err(|e| TranslateError::UnexpectedError(e))
}

/// Convert PTX AST directly to TMatmul assembly with custom memory mappings
pub fn ptx_ast_to_tmatmul_assembly<'input>(
    ast: ast::Module<'input>,
    custom_memory_map: Option<HashMap<String, emit_tmatmul_asm::MemoryLocation>>,
) -> Result<ptx_to_tmatmul::TMatmulCompilationResult, TranslateError> {
    let mut compiler = ptx_to_tmatmul::PtxToTMatmulCompiler::new();

    // Setup standard or custom memory mappings
    if let Some(mem_map) = custom_memory_map {
        for (symbol, location) in mem_map {
            compiler.map_memory(&symbol, location);
        }
    } else {
        compiler.setup_standard_memory_map();
    }

    compiler
        .compile_module(ast)
        .map_err(|e| TranslateError::UnexpectedError(e))
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
enum PtxSpecialRegister {
    Tid,
    Ntid,
    Ctaid,
    Nctaid,
    Clock,
    LanemaskEq,
    LanemaskLt,
    LanemaskLe,
    LanemaskGe,
    LanemaskGt,
    Laneid,
    Envreg(u8),
}

impl PtxSpecialRegister {
    fn iter() -> impl Iterator<Item = Self> {
        const FIXED: [PtxSpecialRegister; 11] = [
            PtxSpecialRegister::Tid,
            PtxSpecialRegister::Ntid,
            PtxSpecialRegister::Ctaid,
            PtxSpecialRegister::Nctaid,
            PtxSpecialRegister::Clock,
            PtxSpecialRegister::LanemaskEq,
            PtxSpecialRegister::LanemaskLt,
            PtxSpecialRegister::LanemaskLe,
            PtxSpecialRegister::LanemaskGe,
            PtxSpecialRegister::LanemaskGt,
            PtxSpecialRegister::Laneid,
        ];
        FIXED
            .into_iter()
            .chain((0u8..=31).map(PtxSpecialRegister::Envreg))
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Tid => "%tid",
            Self::Ntid => "%ntid",
            Self::Ctaid => "%ctaid",
            Self::Nctaid => "%nctaid",
            Self::Clock => "%clock",
            Self::LanemaskEq => "%lanemask_eq",
            Self::LanemaskLt => "%lanemask_lt",
            Self::LanemaskLe => "%lanemask_le",
            Self::LanemaskGe => "%lanemask_ge",
            Self::LanemaskGt => "%lanemask_gt",
            Self::Laneid => "%laneid",
            Self::Envreg(0) => "%envreg0",
            Self::Envreg(1) => "%envreg1",
            Self::Envreg(2) => "%envreg2",
            Self::Envreg(3) => "%envreg3",
            Self::Envreg(4) => "%envreg4",
            Self::Envreg(5) => "%envreg5",
            Self::Envreg(6) => "%envreg6",
            Self::Envreg(7) => "%envreg7",
            Self::Envreg(8) => "%envreg8",
            Self::Envreg(9) => "%envreg9",
            Self::Envreg(10) => "%envreg10",
            Self::Envreg(11) => "%envreg11",
            Self::Envreg(12) => "%envreg12",
            Self::Envreg(13) => "%envreg13",
            Self::Envreg(14) => "%envreg14",
            Self::Envreg(15) => "%envreg15",
            Self::Envreg(16) => "%envreg16",
            Self::Envreg(17) => "%envreg17",
            Self::Envreg(18) => "%envreg18",
            Self::Envreg(19) => "%envreg19",
            Self::Envreg(20) => "%envreg20",
            Self::Envreg(21) => "%envreg21",
            Self::Envreg(22) => "%envreg22",
            Self::Envreg(23) => "%envreg23",
            Self::Envreg(24) => "%envreg24",
            Self::Envreg(25) => "%envreg25",
            Self::Envreg(26) => "%envreg26",
            Self::Envreg(27) => "%envreg27",
            Self::Envreg(28) => "%envreg28",
            Self::Envreg(29) => "%envreg29",
            Self::Envreg(30) => "%envreg30",
            Self::Envreg(31) => "%envreg31",
            Self::Envreg(_) => unreachable!(),
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
            PtxSpecialRegister::LanemaskEq => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskLt => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskLe => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskGe => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskGt => ast::ScalarType::U32,
            PtxSpecialRegister::Laneid => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg(_) => ast::ScalarType::U32,
        }
    }

    fn get_function_input_type(self) -> Option<ast::ScalarType> {
        match self {
            PtxSpecialRegister::Tid
            | PtxSpecialRegister::Ntid
            | PtxSpecialRegister::Ctaid
            | PtxSpecialRegister::Nctaid => Some(ast::ScalarType::U8),
            PtxSpecialRegister::Clock
            | PtxSpecialRegister::LanemaskEq
            | PtxSpecialRegister::LanemaskLt
            | PtxSpecialRegister::LanemaskLe
            | PtxSpecialRegister::LanemaskGe
            | PtxSpecialRegister::LanemaskGt
            | PtxSpecialRegister::Laneid
            | PtxSpecialRegister::Envreg(_) => None,
        }
    }

    fn get_unprefixed_function_name(self) -> &'static str {
        match self {
            PtxSpecialRegister::Tid => "sreg_tid",
            PtxSpecialRegister::Ntid => "sreg_ntid",
            PtxSpecialRegister::Ctaid => "sreg_ctaid",
            PtxSpecialRegister::Nctaid => "sreg_nctaid",
            PtxSpecialRegister::Clock => "sreg_clock",
            PtxSpecialRegister::LanemaskEq => "sreg_lanemask_eq",
            PtxSpecialRegister::LanemaskLt => "sreg_lanemask_lt",
            PtxSpecialRegister::LanemaskLe => "sreg_lanemask_le",
            PtxSpecialRegister::LanemaskGe => "sreg_lanemask_ge",
            PtxSpecialRegister::LanemaskGt => "sreg_lanemask_gt",
            PtxSpecialRegister::Laneid => "sreg_laneid",
            PtxSpecialRegister::Envreg(0) => "sreg_envreg0",
            PtxSpecialRegister::Envreg(1) => "sreg_envreg1",
            PtxSpecialRegister::Envreg(2) => "sreg_envreg2",
            PtxSpecialRegister::Envreg(3) => "sreg_envreg3",
            PtxSpecialRegister::Envreg(4) => "sreg_envreg4",
            PtxSpecialRegister::Envreg(5) => "sreg_envreg5",
            PtxSpecialRegister::Envreg(6) => "sreg_envreg6",
            PtxSpecialRegister::Envreg(7) => "sreg_envreg7",
            PtxSpecialRegister::Envreg(8) => "sreg_envreg8",
            PtxSpecialRegister::Envreg(9) => "sreg_envreg9",
            PtxSpecialRegister::Envreg(10) => "sreg_envreg10",
            PtxSpecialRegister::Envreg(11) => "sreg_envreg11",
            PtxSpecialRegister::Envreg(12) => "sreg_envreg12",
            PtxSpecialRegister::Envreg(13) => "sreg_envreg13",
            PtxSpecialRegister::Envreg(14) => "sreg_envreg14",
            PtxSpecialRegister::Envreg(15) => "sreg_envreg15",
            PtxSpecialRegister::Envreg(16) => "sreg_envreg16",
            PtxSpecialRegister::Envreg(17) => "sreg_envreg17",
            PtxSpecialRegister::Envreg(18) => "sreg_envreg18",
            PtxSpecialRegister::Envreg(19) => "sreg_envreg19",
            PtxSpecialRegister::Envreg(20) => "sreg_envreg20",
            PtxSpecialRegister::Envreg(21) => "sreg_envreg21",
            PtxSpecialRegister::Envreg(22) => "sreg_envreg22",
            PtxSpecialRegister::Envreg(23) => "sreg_envreg23",
            PtxSpecialRegister::Envreg(24) => "sreg_envreg24",
            PtxSpecialRegister::Envreg(25) => "sreg_envreg25",
            PtxSpecialRegister::Envreg(26) => "sreg_envreg26",
            PtxSpecialRegister::Envreg(27) => "sreg_envreg27",
            PtxSpecialRegister::Envreg(28) => "sreg_envreg28",
            PtxSpecialRegister::Envreg(29) => "sreg_envreg29",
            PtxSpecialRegister::Envreg(30) => "sreg_envreg30",
            PtxSpecialRegister::Envreg(31) => "sreg_envreg31",
            PtxSpecialRegister::Envreg(_) => unreachable!(),
        }
    }
}

#[track_caller]
#[cfg(debug_assertions)]
fn error_unreachable() -> TranslateError {
    let loc = std::panic::Location::caller();
    let bt = std::backtrace::Backtrace::force_capture();
    TranslateError::UnexpectedError(format!(
        "unreachable path at {}:{}:{}\nbacktrace:\n{}",
        loc.file(),
        loc.line(),
        loc.column(),
        bt
    ))
}

#[cfg(not(debug_assertions))]
fn error_unreachable() -> TranslateError {
    TranslateError::Unreachable
}

#[track_caller]
#[cfg(debug_assertions)]
fn error_todo_msg<T: Into<String>>(msg: T) -> TranslateError {
    let loc = std::panic::Location::caller();
    TranslateError::Todo(format!(
        "{} @ {}:{}:{}",
        msg.into(),
        loc.file(),
        loc.line(),
        loc.column()
    ))
}

#[cfg(not(debug_assertions))]
fn error_todo_msg<T: Into<String>>(msg: T) -> TranslateError {
    TranslateError::Todo(msg.into())
}

#[track_caller]
#[cfg(debug_assertions)]
fn error_todo() -> TranslateError {
    let loc = std::panic::Location::caller();
    TranslateError::Todo(format!(
        "todo path at {}:{}:{}",
        loc.file(),
        loc.line(),
        loc.column()
    ))
}

#[cfg(not(debug_assertions))]
fn error_todo() -> TranslateError {
    TranslateError::Todo("".to_string())
}

#[cfg(debug_assertions)]
fn error_unknown_symbol<T: Into<String>>(symbol: T) -> TranslateError {
    let loc = std::panic::Location::caller();
    TranslateError::UnknownSymbol(format!(
        "{} (at {}:{}:{})",
        symbol.into(),
        loc.file(),
        loc.line(),
        loc.column()
    ))
}

#[cfg(not(debug_assertions))]
fn error_unknown_symbol<T: Into<String>>(symbol: T) -> TranslateError {
    TranslateError::UnknownSymbol(symbol.into())
}

#[cfg(debug_assertions)]
fn error_mismatched_type() -> TranslateError {
    TranslateError::MismatchedType
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
                return ast::visit_map(i, visitor).map(Statement::Instruction);
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
        11 + 32
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
    ) -> Vec<(
        PtxSpecialRegister,
        (
            Vec<ast::Variable<SpirvWord>>,
            SpirvWord,
            Vec<ast::Variable<SpirvWord>>,
        ),
    )> {
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
