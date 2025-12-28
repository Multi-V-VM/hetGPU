// #![feature(str_from_raw_parts)]

pub mod checkpoint;
pub mod checkpoint_integration;
pub mod debug;
pub mod dwarf_validation;
pub mod pass;
pub mod state_recovery;
#[cfg(test)]
mod test;

pub use pass::llvm::bitcode_to_ir;
pub use pass::to_llvm_module;
pub use pass::to_llvm_module_with_debug_round_trip;
pub use pass::to_llvm_module_with_filename;
pub use pass::to_mlir_module;
pub use pass::Attributes;
pub use pass::Module;
pub use pass::TranslateError;

// Export SASS ↔ PTX mapping types for hetGPU runtime integration
pub use debug::{
    // Core mapping types
    SassInstruction,
    SassLineMapping,
    SassPtxMapper,
    // CUBIN debug info
    CubinDebugInfo,
    CubinSymbol,
    CubinSymbolType,
    DebugLineEntry,
    // hetGPU runtime interface
    HetGpuDebugInterface,
    RuntimeBreakpoint,
    Watchpoint,
    WatchType,
    StepMode,
    ExecutionContext,
    StackFrame,
    // PTX reconstruction
    PtxReconstructor,
    PtxExecutionState,
    // Original types
    PtxSourceLocation,
    TargetInstruction,
    DwarfMappingEntry,
    VariableLocation,
    PtxDwarfBuilder,
    PtxStateRecovery,
    // GPU trap and checkpoint types
    TrapReason,
    GpuCheckpointState,
    GpuTrapHandler,
    CheckpointBreakpoint,
    ThreadCheckpointState,
    MemoryRegion,
    MemorySpace,
    KernelArgument,
    ResumeInfo,
    CheckpointManager,
};

pub use state_recovery::{
    PtxStateRecoveryManager,
    ExecutionState,
    VariableValue,
    ThreadState,
    MemorySnapshot,
    Breakpoint,
    CallFrame,
};

use std::collections::HashMap;

/// PTX to LLVM to PTX round-trip with SASS mapping
///
/// This function compiles PTX through LLVM and back to PTX, extracting
/// debug information for SASS ↔ PTX mapping during hetGPU runtime.
///
/// Returns:
/// - Module: The compiled LLVM module
/// - String: Regenerated PTX with debug info
/// - HashMap<u64, u64>: SASS address to PTX line mapping (line << 32 | column)
pub fn ptx_to_llvm_to_ptx_with_sass_mapping(
    text: &str,
) -> Result<(pass::Module, String, HashMap<u64, u64>), TranslateError> {
    // Parse PTX source
    let ast = ptx_parser::parse_module_checked(text)
        .map_err(|e| TranslateError::Todo(format!("PTX parse error: {:?}", e)))?;

    // Compile to LLVM with debug info, then back to PTX
    let (module, regenerated_ptx, debug_mappings) = pass::to_llvm_module_with_debug_round_trip(ast)?;

    // Convert debug mappings to SASS address → PTX line format
    // Each entry maps SASS address to (line << 32 | column)
    let sass_to_ptx: HashMap<u64, u64> = debug_mappings
        .iter()
        .map(|(sass_addr, loc)| (*sass_addr, ((loc.line as u64) << 32) | (loc.column as u64)))
        .collect();

    Ok((module, regenerated_ptx, sass_to_ptx))
}

/// Create a SASS-PTX mapper from cuobjdump output
///
/// Example usage:
/// ```ignore
/// let output = std::process::Command::new("cuobjdump")
///     .args(&["-sass", "-lineinfo", "kernel.cubin"])
///     .output()?;
/// let mapper = create_sass_ptx_mapper_from_cuobjdump(
///     &String::from_utf8_lossy(&output.stdout),
///     Some(ptx_source)
/// )?;
///
/// // Query PTX location from SASS address during breakpoint
/// if let Some(loc) = mapper.sass_to_ptx_location(sass_addr) {
///     println!("Breakpoint hit at {}:{}", loc.file, loc.line);
/// }
/// ```
pub fn create_sass_ptx_mapper_from_cuobjdump(
    cuobjdump_output: &str,
    ptx_source: Option<String>,
) -> Result<debug::SassPtxMapper, String> {
    let mut mapper = match ptx_source {
        Some(src) => debug::SassPtxMapper::with_ptx_source(src),
        None => debug::SassPtxMapper::new(),
    };
    mapper.parse_cuobjdump_output(cuobjdump_output)?;
    Ok(mapper)
}

/// Create a hetGPU debug interface for runtime debugging
///
/// Example usage:
/// ```ignore
/// let mut debug_iface = create_hetgpu_debug_interface("kernel.cubin", Some(ptx_source))?;
///
/// // Set breakpoint at PTX line
/// debug_iface.set_breakpoint_at_ptx_line("kernel.ptx", 42)?;
///
/// // During execution, query PTX location from SASS address
/// if let Some((ptx_loc, bp)) = debug_iface.handle_breakpoint_hit(sass_addr) {
///     println!("Hit breakpoint #{} at {}:{}", bp.id, ptx_loc.file, ptx_loc.line);
/// }
/// ```
#[cfg(unix)]
pub fn create_hetgpu_debug_interface(
    cubin_path: &str,
    ptx_source: Option<String>,
) -> Result<debug::HetGpuDebugInterface, String> {
    let mut iface = debug::HetGpuDebugInterface::new();
    iface.load_from_cubin(cubin_path, ptx_source)?;
    Ok(iface)
}

#[cfg(not(unix))]
pub fn create_hetgpu_debug_interface(
    _cubin_path: &str,
    _ptx_source: Option<String>,
) -> Result<debug::HetGpuDebugInterface, String> {
    Err("hetGPU debug interface only supported on Unix systems".to_string())
}

/// Stub function for backward compatibility
/// TODO: Implement proper LLVM to SPIRV conversion
pub fn llvm_to_spirv_robust(_llvm_ir: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Err("llvm_to_spirv_robust not implemented".into())
}
