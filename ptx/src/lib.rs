// #![feature(str_from_raw_parts)]

pub mod checkpoint;
pub mod checkpoint_integration;
pub mod debug;
pub mod dwarf_validation;
pub mod pass;
pub mod sass;
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
    CheckpointBreakpoint,
    CheckpointManager,
    // CUBIN debug info
    CubinDebugInfo,
    CubinSymbol,
    CubinSymbolType,
    DebugLineEntry,
    DwarfMappingEntry,
    ExecutionContext,
    GpuCheckpointState,
    GpuTrapHandler,
    // hetGPU runtime interface
    HetGpuDebugInterface,
    KernelArgument,
    MemoryRegion,
    MemorySpace,
    PtxDwarfBuilder,
    PtxExecutionState,
    // PTX reconstruction
    PtxReconstructor,
    // Original types
    PtxSourceLocation,
    PtxStateRecovery,
    ResumeInfo,
    RuntimeBreakpoint,
    // Core mapping types
    SassInstruction,
    SassLineMapping,
    SassPtxMapper,
    StackFrame,
    StepMode,
    TargetInstruction,
    ThreadCheckpointState,
    // GPU trap and checkpoint types
    TrapReason,
    VariableLocation,
    WatchType,
    Watchpoint,
};

pub use state_recovery::{
    Breakpoint, CallFrame, ExecutionState, MemorySnapshot, PtxStateRecoveryManager, ThreadState,
    VariableValue,
};

// Export enhanced SASS analysis types
pub use sass::{
    // Helper functions
    classify_opcode,
    get_memory_space,
    get_ptx_template,
    is_cubin,
    CompilationUnitInfo,
    ControlFlowAnalyzer,
    CubinConstant,
    CubinKernel,
    CubinParseError,
    // CUBIN parser
    CubinParser,
    CubinParserBuilder,
    CubinSection,
    DebugFunctionInfo,
    DebugVariableInfo,
    DisassemblerError,
    DwarfParseError,
    // DWARF parser
    DwarfParser,
    // Enhanced SASS instruction with full semantics
    EnhancedSassInstruction,
    InlineStats,
    MappingConfidence,
    ParsedCubin,
    ParsedDebugInfo,
    PtxEquivalent,
    PtxTemplate,
    SassControlCodes,
    SassDataType,
    // SASS disassembler
    SassDisassembler,
    SassInlineConfig,
    SassInlineStrategy,
    // SASS → LLVM inlining
    SassInliner,
    SassKernelBuilder,
    SassMemorySpace,
    SassMetadataExtractor,
    SassOpcodeClass,
    SassOperand,
    SassRegister,
    SmVersion,
    TextDisassemblyParser,
    VariableLocationExpr,
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
    let (module, regenerated_ptx, debug_mappings) =
        pass::to_llvm_module_with_debug_round_trip(ast)?;

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

// ============================================================================
// Native CUBIN/SASS Analysis Functions
// ============================================================================

/// Parse a CUBIN file and extract kernels, debug info, and SASS code
///
/// This function provides native CUBIN parsing without requiring cuobjdump.
///
/// Example usage:
/// ```ignore
/// let cubin_data = std::fs::read("kernel.cubin")?;
/// let parsed = parse_cubin_native(&cubin_data)?;
///
/// for kernel in &parsed.kernels {
///     println!("Kernel: {} ({} bytes, {} registers)",
///         kernel.name, kernel.size, kernel.num_registers);
/// }
/// ```
pub fn parse_cubin_native(cubin_data: &[u8]) -> Result<sass::ParsedCubin, sass::CubinParseError> {
    let parser = sass::CubinParser::new(cubin_data.to_vec());
    parser.parse()
}

/// Disassemble SASS code from a kernel
///
/// Example usage:
/// ```ignore
/// let parsed = parse_cubin_native(&cubin_data)?;
/// let kernel = &parsed.kernels[0];
/// let instructions = disassemble_sass(&kernel.code, kernel.sm_version, kernel.address)?;
///
/// for inst in &instructions {
///     println!("{}", inst);
///     if let Some(ptx) = inst.generate_ptx() {
///         println!("  -> {}", ptx);
///     }
/// }
/// ```
pub fn disassemble_sass(
    code: &[u8],
    sm_version: u32,
    base_address: u64,
) -> Result<Vec<sass::EnhancedSassInstruction>, sass::DisassemblerError> {
    let disasm = sass::SassDisassembler::new(sm_version)?;
    Ok(disasm.disassemble(code, base_address))
}

/// Parse cuobjdump text output into enhanced SASS instructions
///
/// This is a convenience wrapper around TextDisassemblyParser.
pub fn parse_cuobjdump_to_enhanced(output: &str) -> Vec<sass::EnhancedSassInstruction> {
    sass::TextDisassemblyParser::parse_cuobjdump_output(output)
}

/// Create an enhanced SASS-PTX mapper from CUBIN file
///
/// This combines native CUBIN parsing with DWARF debug info extraction
/// to create a high-precision SASS ↔ PTX mapping.
///
/// Example usage:
/// ```ignore
/// let cubin_data = std::fs::read("kernel.cubin")?;
/// let mapper = create_enhanced_sass_mapper(&cubin_data, Some(ptx_source))?;
///
/// // Get all instructions for a kernel with PTX source context
/// for inst in mapper.get_kernel_instructions("my_kernel") {
///     println!("{:04x}: {} -> {}:{}",
///         inst.address, inst.opcode,
///         inst.ptx_file.as_deref().unwrap_or("?"),
///         inst.ptx_line.unwrap_or(0));
/// }
/// ```
pub fn create_enhanced_sass_mapper(
    cubin_data: &[u8],
    ptx_source: Option<String>,
) -> Result<EnhancedSassMapper, Box<dyn std::error::Error>> {
    let mut mapper = EnhancedSassMapper::new();
    mapper.load_cubin(cubin_data)?;
    if let Some(src) = ptx_source {
        mapper.set_ptx_source(src);
    }
    Ok(mapper)
}

/// Enhanced SASS-PTX mapper with native parsing
pub struct EnhancedSassMapper {
    /// Parsed CUBIN data
    parsed_cubin: Option<sass::ParsedCubin>,
    /// Disassembled instructions per kernel
    kernel_instructions: HashMap<String, Vec<sass::EnhancedSassInstruction>>,
    /// PTX source
    ptx_source: Option<String>,
    /// SASS address to instruction index
    address_index: HashMap<u64, (String, usize)>,
}

impl EnhancedSassMapper {
    pub fn new() -> Self {
        Self {
            parsed_cubin: None,
            kernel_instructions: HashMap::new(),
            ptx_source: None,
            address_index: HashMap::new(),
        }
    }

    /// Load and parse CUBIN data
    pub fn load_cubin(&mut self, cubin_data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Parse CUBIN structure
        let parsed = parse_cubin_native(cubin_data)?;

        // Disassemble each kernel
        for kernel in &parsed.kernels {
            let disasm = sass::SassDisassembler::new(kernel.sm_version)?;
            let mut instructions = disasm.disassemble(&kernel.code, kernel.address);

            // Apply debug line mappings if available
            for inst in &mut instructions {
                if let Some(debug_info) = parsed.debug_lines.get(&inst.address) {
                    inst.ptx_file = Some(debug_info.file.clone());
                    inst.ptx_line = Some(debug_info.line);
                    inst.ptx_column = Some(debug_info.column);
                }
                inst.function_name = Some(kernel.name.clone());
            }

            // Analyze control flow and data dependencies
            sass::ControlFlowAnalyzer::find_basic_blocks(&mut instructions);
            sass::ControlFlowAnalyzer::analyze_data_flow(&mut instructions);

            // Build address index
            for (idx, inst) in instructions.iter().enumerate() {
                self.address_index
                    .insert(inst.address, (kernel.name.clone(), idx));
            }

            self.kernel_instructions
                .insert(kernel.name.clone(), instructions);
        }

        self.parsed_cubin = Some(parsed);
        Ok(())
    }

    /// Set PTX source for context retrieval
    pub fn set_ptx_source(&mut self, source: String) {
        self.ptx_source = Some(source);
    }

    /// Get instructions for a kernel
    pub fn get_kernel_instructions(&self, kernel_name: &str) -> &[sass::EnhancedSassInstruction] {
        self.kernel_instructions
            .get(kernel_name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Look up instruction by SASS address
    pub fn get_instruction_at(&self, address: u64) -> Option<&sass::EnhancedSassInstruction> {
        let (kernel, idx) = self.address_index.get(&address)?;
        self.kernel_instructions.get(kernel)?.get(*idx)
    }

    /// Get PTX source line content
    pub fn get_ptx_line(&self, line: u32) -> Option<&str> {
        self.ptx_source
            .as_ref()
            .and_then(|source| source.lines().nth((line.saturating_sub(1)) as usize))
    }

    /// Get all kernel names
    pub fn get_kernel_names(&self) -> Vec<&str> {
        self.kernel_instructions
            .keys()
            .map(|s| s.as_str())
            .collect()
    }

    /// Get parsed CUBIN info
    pub fn get_cubin_info(&self) -> Option<&sass::ParsedCubin> {
        self.parsed_cubin.as_ref()
    }

    /// Find instructions at a PTX line
    pub fn find_instructions_at_ptx_line(
        &self,
        file: &str,
        line: u32,
    ) -> Vec<&sass::EnhancedSassInstruction> {
        let mut results = Vec::new();
        for instructions in self.kernel_instructions.values() {
            for inst in instructions {
                if inst.ptx_line == Some(line) {
                    if let Some(ref f) = inst.ptx_file {
                        if f.contains(file) || file.contains(f) {
                            results.push(inst);
                        }
                    }
                }
            }
        }
        results
    }

    /// Generate PTX reconstruction for all instructions
    pub fn generate_ptx_reconstruction(&self, kernel_name: &str) -> String {
        let mut output = String::new();
        let instructions = self.get_kernel_instructions(kernel_name);

        output.push_str(&format!(
            "// PTX reconstruction for kernel: {}\n",
            kernel_name
        ));
        output.push_str("// Generated from SASS analysis\n\n");

        let mut current_line = 0u32;
        for inst in instructions {
            // Add source context comment
            if let Some(line) = inst.ptx_line {
                if line != current_line {
                    current_line = line;
                    if let Some(src_line) = self.get_ptx_line(line) {
                        output.push_str(&format!("\n// Line {}: {}\n", line, src_line.trim()));
                    }
                }
            }

            // Add SASS instruction as comment
            output.push_str(&format!("// SASS {:04x}: {}\n", inst.address, inst.opcode));

            // Add PTX equivalent if available
            if let Some(ptx) = inst.generate_ptx() {
                output.push_str(&format!("    {}\n", ptx));
            }
        }

        output
    }
}

impl Default for EnhancedSassMapper {
    fn default() -> Self {
        Self::new()
    }
}
