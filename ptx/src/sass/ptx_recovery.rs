//! PTX Recovery from SASS using Debug Information
//!
//! This module provides functionality to recover PTX source code from SASS
//! instructions using DWARF debug information embedded in CUBIN files.
//!
//! Key features:
//! - Extract PTX source lines from embedded debug info
//! - Map SASS addresses to PTX source locations
//! - Reconstruct PTX from SASS semantics when source not available
//! - Generate annotated PTX with SASS correlation

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs;
use std::path::Path;

use super::cubin_parser::{CubinParser, DebugLineInfo, ParsedCubin};
use super::disassembler::SassDisassembler;
use super::dwarf_parser::{DebugFunctionInfo, DwarfParser, ParsedDebugInfo};
use super::instruction::{EnhancedSassInstruction, SassDataType, SassOpcodeClass, SassOperand};

// ============================================================================
// PTX Recovery Structures
// ============================================================================

/// PTX source line with metadata
#[derive(Debug, Clone)]
pub struct PtxSourceLine {
    /// Line number in original PTX
    pub line_number: u32,
    /// PTX source text
    pub source: String,
    /// File name
    pub file: String,
    /// Is this a statement (vs. directive)
    pub is_statement: bool,
    /// Column number
    pub column: u32,
}

/// Recovered PTX function
#[derive(Debug, Clone)]
pub struct RecoveredPtxFunction {
    /// Function name
    pub name: String,
    /// Mangled/linkage name
    pub linkage_name: Option<String>,
    /// Start address in SASS
    pub start_address: u64,
    /// End address in SASS
    pub end_address: u64,
    /// PTX source lines (line_number -> source)
    pub ptx_lines: BTreeMap<u32, PtxSourceLine>,
    /// SASS to PTX mapping (SASS address -> PTX line number)
    pub sass_to_ptx: HashMap<u64, u32>,
    /// Reconstructed PTX (if original not available)
    pub reconstructed_ptx: Option<String>,
    /// Original PTX source (if available from embedded data)
    pub original_ptx: Option<String>,
}

/// Complete PTX recovery result
#[derive(Debug, Clone)]
pub struct PtxRecoveryResult {
    /// Recovered functions
    pub functions: Vec<RecoveredPtxFunction>,
    /// Global PTX header (version, target, etc.)
    pub ptx_header: String,
    /// Debug information used
    pub debug_info: Option<ParsedDebugInfo>,
    /// Source files referenced
    pub source_files: Vec<String>,
    /// Recovery statistics
    pub stats: RecoveryStats,
}

/// Recovery statistics
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Total SASS instructions
    pub total_sass_instructions: u64,
    /// Instructions with PTX mapping
    pub mapped_instructions: u64,
    /// Instructions reconstructed from semantics
    pub reconstructed_instructions: u64,
    /// Unmapped instructions
    pub unmapped_instructions: u64,
    /// Number of PTX lines recovered
    pub ptx_lines_recovered: u64,
}

// ============================================================================
// PTX Reconstructor
// ============================================================================

/// Reconstructs PTX from SASS instructions
pub struct PtxReconstructor {
    /// SM version for instruction semantics
    sm_version: u32,
    /// Register allocation tracking
    registers: HashMap<String, String>, // SASS reg -> PTX reg
    /// Predicate tracking
    predicates: HashMap<String, String>,
    /// Next virtual register number
    next_vreg: u32,
}

impl PtxReconstructor {
    /// Create a new PTX reconstructor
    pub fn new(sm_version: u32) -> Self {
        Self {
            sm_version,
            registers: HashMap::new(),
            predicates: HashMap::new(),
            next_vreg: 0,
        }
    }

    /// Reconstruct PTX for a single SASS instruction
    pub fn reconstruct_instruction(&mut self, inst: &EnhancedSassInstruction) -> String {
        match inst.opcode_class {
            SassOpcodeClass::IntegerArithmetic => self.reconstruct_int_arith(inst),
            SassOpcodeClass::FloatArithmetic => self.reconstruct_float_arith(inst),
            SassOpcodeClass::GlobalLoad => self.reconstruct_global_load(inst),
            SassOpcodeClass::GlobalStore => self.reconstruct_global_store(inst),
            SassOpcodeClass::SharedLoad => self.reconstruct_shared_load(inst),
            SassOpcodeClass::SharedStore => self.reconstruct_shared_store(inst),
            SassOpcodeClass::Move => self.reconstruct_move(inst),
            SassOpcodeClass::IntegerComparison | SassOpcodeClass::FloatComparison => {
                self.reconstruct_compare(inst)
            }
            SassOpcodeClass::Branch | SassOpcodeClass::ConditionalBranch => {
                self.reconstruct_branch(inst)
            }
            SassOpcodeClass::Barrier => self.reconstruct_barrier(inst),
            SassOpcodeClass::SpecialRegRead => self.reconstruct_special_reg(inst),
            SassOpcodeClass::IntegerConversion | SassOpcodeClass::FloatConversion => {
                self.reconstruct_conversion(inst)
            }
            SassOpcodeClass::Exit => "ret;".to_string(),
            _ => format!("// SASS: {} {}", inst.opcode, self.format_operands(inst)),
        }
    }

    fn reconstruct_int_arith(&mut self, inst: &EnhancedSassInstruction) -> String {
        let op = match inst.opcode.as_str() {
            "IADD" | "IADD3" => "add",
            "ISUB" => "sub",
            "IMUL" | "IMAD" => {
                if inst.opcode.contains("MAD") {
                    "mad"
                } else {
                    "mul"
                }
            }
            "IAND" | "LOP3" => "and",
            "IOR" => "or",
            "IXOR" => "xor",
            "ISHL" | "SHL" => "shl",
            "ISHR" | "SHR" => "shr",
            _ => return format!("// INT: {} {}", inst.opcode, self.format_operands(inst)),
        };

        let data_type = self.get_ptx_type(inst);
        let dest = self.format_dest(inst);
        let sources = self.format_sources(inst);

        if op == "mad" && sources.len() >= 3 {
            format!(
                "{}.{} {}, {}, {}, {};",
                op, data_type, dest, sources[0], sources[1], sources[2]
            )
        } else if sources.len() >= 2 {
            format!(
                "{}.{} {}, {}, {};",
                op, data_type, dest, sources[0], sources[1]
            )
        } else if !sources.is_empty() {
            format!("{}.{} {}, {};", op, data_type, dest, sources[0])
        } else {
            format!("// {}: missing operands", inst.opcode)
        }
    }

    fn reconstruct_float_arith(&mut self, inst: &EnhancedSassInstruction) -> String {
        let op = match inst.opcode.as_str() {
            "FADD" => "add",
            "FSUB" => "sub",
            "FMUL" | "FFMA" => {
                if inst.opcode.contains("FMA") {
                    "fma"
                } else {
                    "mul"
                }
            }
            "FDIV" => "div",
            "FABS" => "abs",
            "FNEG" => "neg",
            "FMIN" => "min",
            "FMAX" => "max",
            "FSQRT" | "MUFU" => "sqrt",
            "FSIN" => "sin",
            "FCOS" => "cos",
            "FLG2" => "lg2",
            "FEX2" => "ex2",
            _ => return format!("// FP: {} {}", inst.opcode, self.format_operands(inst)),
        };

        let data_type = self.get_ptx_type(inst);
        let dest = self.format_dest(inst);
        let sources = self.format_sources(inst);

        if op == "fma" && sources.len() >= 3 {
            format!(
                "{}.rn.{} {}, {}, {}, {};",
                op, data_type, dest, sources[0], sources[1], sources[2]
            )
        } else if sources.len() >= 2 {
            format!(
                "{}.{} {}, {}, {};",
                op, data_type, dest, sources[0], sources[1]
            )
        } else if !sources.is_empty() {
            format!("{}.{} {}, {};", op, data_type, dest, sources[0])
        } else {
            format!("// {}: missing operands", inst.opcode)
        }
    }

    fn reconstruct_global_load(&mut self, inst: &EnhancedSassInstruction) -> String {
        let data_type = self.get_ptx_type(inst);
        let dest = self.format_dest(inst);
        let addr = self.format_memory_operand(inst);

        format!("ld.global.{} {}, [{}];", data_type, dest, addr)
    }

    fn reconstruct_global_store(&mut self, inst: &EnhancedSassInstruction) -> String {
        let data_type = self.get_ptx_type(inst);
        let addr = self.format_memory_operand(inst);
        let src = if !inst.src_operands.is_empty() {
            self.format_operand(&inst.src_operands[0])
        } else {
            "?".to_string()
        };

        format!("st.global.{} [{}], {};", data_type, addr, src)
    }

    fn reconstruct_shared_load(&mut self, inst: &EnhancedSassInstruction) -> String {
        let data_type = self.get_ptx_type(inst);
        let dest = self.format_dest(inst);
        let addr = self.format_memory_operand(inst);

        format!("ld.shared.{} {}, [{}];", data_type, dest, addr)
    }

    fn reconstruct_shared_store(&mut self, inst: &EnhancedSassInstruction) -> String {
        let data_type = self.get_ptx_type(inst);
        let addr = self.format_memory_operand(inst);
        let src = if !inst.src_operands.is_empty() {
            self.format_operand(&inst.src_operands[0])
        } else {
            "?".to_string()
        };

        format!("st.shared.{} [{}], {};", data_type, addr, src)
    }

    fn reconstruct_move(&mut self, inst: &EnhancedSassInstruction) -> String {
        let data_type = self.get_ptx_type(inst);
        let dest = self.format_dest(inst);
        let src = if !inst.src_operands.is_empty() {
            self.format_operand(&inst.src_operands[0])
        } else {
            "?".to_string()
        };

        format!("mov.{} {}, {};", data_type, dest, src)
    }

    fn reconstruct_compare(&mut self, inst: &EnhancedSassInstruction) -> String {
        let cmp_op = inst
            .modifiers
            .iter()
            .find(|m| ["LT", "LE", "GT", "GE", "EQ", "NE"].contains(&m.as_str()))
            .map(|s| s.to_lowercase())
            .unwrap_or_else(|| "eq".to_string());

        let data_type = self.get_ptx_type(inst);
        let dest = self.format_dest(inst);
        let sources = self.format_sources(inst);

        if sources.len() >= 2 {
            format!(
                "setp.{}.{} {}, {}, {};",
                cmp_op, data_type, dest, sources[0], sources[1]
            )
        } else {
            format!("// CMP: {} {}", inst.opcode, self.format_operands(inst))
        }
    }

    fn reconstruct_branch(&mut self, inst: &EnhancedSassInstruction) -> String {
        // Check for predicate
        let pred = inst
            .predicate
            .as_ref()
            .map(|p| match p {
                SassOperand::Predicate { register, negated } => {
                    let neg = if *negated { "!" } else { "" };
                    format!("@{}%p{} ", neg, register.number)
                }
                _ => String::new(),
            })
            .unwrap_or_default();

        // Get target
        let target = inst
            .src_operands
            .iter()
            .find_map(|op| {
                if let SassOperand::Immediate(imm) = op {
                    Some(format!("L_{:x}", imm))
                } else if let SassOperand::Label(lbl) = op {
                    Some(lbl.clone())
                } else if let SassOperand::Address(addr) = op {
                    Some(format!("L_{:x}", addr))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "?".to_string());

        format!("{}bra {};", pred, target)
    }

    fn reconstruct_barrier(&mut self, inst: &EnhancedSassInstruction) -> String {
        if inst.opcode.contains("SYNC") || inst.opcode == "BAR" {
            "bar.sync 0;".to_string()
        } else {
            format!("// BARRIER: {}", inst.opcode)
        }
    }

    fn reconstruct_special_reg(&mut self, inst: &EnhancedSassInstruction) -> String {
        let dest = self.format_dest(inst);

        // Map SASS special register names to PTX
        let special_reg = inst
            .src_operands
            .iter()
            .find_map(|op| {
                if let SassOperand::SpecialRegister(name) = op {
                    Some(self.map_special_register(name))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "%tid.x".to_string());

        format!("mov.u32 {}, {};", dest, special_reg)
    }

    fn reconstruct_conversion(&mut self, inst: &EnhancedSassInstruction) -> String {
        let dest = self.format_dest(inst);
        let src = if !inst.src_operands.is_empty() {
            self.format_operand(&inst.src_operands[0])
        } else {
            "?".to_string()
        };

        // Determine source and dest types from modifiers
        let (src_type, dst_type) = self.get_conversion_types(inst);

        format!("cvt.{}.{} {}, {};", dst_type, src_type, dest, src)
    }

    fn map_special_register(&self, sass_name: &str) -> String {
        match sass_name {
            "SR_TID.X" | "SR_TIDX" => "%tid.x".to_string(),
            "SR_TID.Y" | "SR_TIDY" => "%tid.y".to_string(),
            "SR_TID.Z" | "SR_TIDZ" => "%tid.z".to_string(),
            "SR_CTAID.X" | "SR_CTAIDX" => "%ctaid.x".to_string(),
            "SR_CTAID.Y" | "SR_CTAIDY" => "%ctaid.y".to_string(),
            "SR_CTAID.Z" | "SR_CTAIDZ" => "%ctaid.z".to_string(),
            "SR_NTID.X" | "SR_NTIDX" => "%ntid.x".to_string(),
            "SR_NTID.Y" | "SR_NTIDY" => "%ntid.y".to_string(),
            "SR_NTID.Z" | "SR_NTIDZ" => "%ntid.z".to_string(),
            "SR_NCTAID.X" | "SR_NCTAIDX" => "%nctaid.x".to_string(),
            "SR_NCTAID.Y" | "SR_NCTAIDY" => "%nctaid.y".to_string(),
            "SR_NCTAID.Z" | "SR_NCTAIDZ" => "%nctaid.z".to_string(),
            "SR_LANEID" => "%laneid".to_string(),
            "SR_WARPID" => "%warpid".to_string(),
            "SR_CLOCK" | "SR_CLOCK_LO" => "%clock".to_string(),
            "SR_CLOCK_HI" => "%clock_hi".to_string(),
            _ => format!("%{}", sass_name.to_lowercase()),
        }
    }

    fn get_ptx_type(&self, inst: &EnhancedSassInstruction) -> String {
        if let Some(dt) = &inst.data_type {
            match dt {
                SassDataType::U8 => "u8",
                SassDataType::U16 => "u16",
                SassDataType::U32 => "u32",
                SassDataType::U64 => "u64",
                SassDataType::U128 => "b128", // PTX doesn't have u128, use b128
                SassDataType::S8 => "s8",
                SassDataType::S16 => "s16",
                SassDataType::S32 => "s32",
                SassDataType::S64 => "s64",
                SassDataType::F16 => "f16",
                SassDataType::F32 => "f32",
                SassDataType::F64 => "f64",
                SassDataType::BF16 => "bf16",
                SassDataType::TF32 => "tf32",
                SassDataType::FP8E4M3 => "e4m3",
                SassDataType::FP8E5M2 => "e5m2",
                SassDataType::B8 => "b8",
                SassDataType::B16 => "b16",
                SassDataType::B32 => "b32",
                SassDataType::B64 => "b64",
                SassDataType::B128 => "b128",
                SassDataType::Pred => "pred",
                SassDataType::Unknown => "b32",
            }
            .to_string()
        } else {
            // Infer from opcode class
            match inst.opcode_class {
                SassOpcodeClass::FloatArithmetic => "f32",
                SassOpcodeClass::IntegerArithmetic => "s32",
                SassOpcodeClass::IntegerComparison | SassOpcodeClass::FloatComparison => "pred",
                _ => "b32",
            }
            .to_string()
        }
    }

    fn get_conversion_types(&self, inst: &EnhancedSassInstruction) -> (String, String) {
        // Parse from modifiers like I2F, F2I, etc.
        let src_type = inst
            .modifiers
            .iter()
            .find_map(|m| {
                if m.starts_with("F32") || m == "F" {
                    Some("f32")
                } else if m.starts_with("F64") {
                    Some("f64")
                } else if m.starts_with("S32") || m == "S" {
                    Some("s32")
                } else if m.starts_with("U32") || m == "U" {
                    Some("u32")
                } else {
                    None
                }
            })
            .unwrap_or("s32");

        let dst_type = if inst.opcode.contains("I2F") || inst.opcode.contains("2F") {
            "f32"
        } else if inst.opcode.contains("F2I") || inst.opcode.contains("2I") {
            "s32"
        } else {
            "s32"
        };

        (src_type.to_string(), dst_type.to_string())
    }

    fn format_dest(&self, inst: &EnhancedSassInstruction) -> String {
        if let Some(dest) = inst.dest_operands.first() {
            self.format_operand(dest)
        } else {
            "%r0".to_string()
        }
    }

    fn format_sources(&self, inst: &EnhancedSassInstruction) -> Vec<String> {
        inst.src_operands
            .iter()
            .map(|op| self.format_operand(op))
            .collect()
    }

    fn format_operands(&self, inst: &EnhancedSassInstruction) -> String {
        let dests: Vec<String> = inst
            .dest_operands
            .iter()
            .map(|op| self.format_operand(op))
            .collect();
        let srcs: Vec<String> = inst
            .src_operands
            .iter()
            .map(|op| self.format_operand(op))
            .collect();

        if dests.is_empty() {
            srcs.join(", ")
        } else {
            format!("{}, {}", dests.join(", "), srcs.join(", "))
        }
    }

    fn format_operand(&self, op: &SassOperand) -> String {
        match op {
            SassOperand::Register(reg) => {
                // Handle uniform registers by prefix
                if reg.prefix.starts_with("UR") || reg.prefix == "U" {
                    format!("%ur{}", reg.number)
                } else if reg.prefix.starts_with("P") {
                    format!("%p{}", reg.number)
                } else {
                    format!("%r{}", reg.number)
                }
            }
            SassOperand::Predicate {
                register,
                negated: _,
            } => format!("%p{}", register.number),
            SassOperand::Immediate(imm) => format!("{}", imm),
            SassOperand::FloatImmediate(f) => format!("{}", f),
            SassOperand::ConstantBank { bank, offset } => format!("c[{}][{}]", bank, offset),
            SassOperand::Memory { base, offset, .. } => {
                if *offset != 0 {
                    format!(
                        "[%r{}+{}]",
                        base.as_ref().map(|r| r.number).unwrap_or(0),
                        offset
                    )
                } else {
                    format!("[%r{}]", base.as_ref().map(|r| r.number).unwrap_or(0))
                }
            }
            SassOperand::SpecialRegister(name) => self.map_special_register(name),
            SassOperand::Address(addr) => format!("0x{:x}", addr),
            SassOperand::Label(lbl) => lbl.clone(),
            SassOperand::Barrier(n) => format!("barrier_{}", n),
        }
    }

    fn format_memory_operand(&self, inst: &EnhancedSassInstruction) -> String {
        // Find memory operand in sources or destinations
        for op in inst.src_operands.iter().chain(inst.dest_operands.iter()) {
            if let SassOperand::Memory { base, offset, .. } = op {
                let base_reg = base
                    .as_ref()
                    .map(|r| format!("%r{}", r.number))
                    .unwrap_or_default();
                return if *offset != 0 {
                    format!("{}+{}", base_reg, offset)
                } else {
                    base_reg
                };
            }
        }

        // Fallback to first source operand
        if let Some(op) = inst.src_operands.first() {
            self.format_operand(op)
        } else {
            "?".to_string()
        }
    }
}

// ============================================================================
// PTX Recovery Engine
// ============================================================================

/// Main PTX recovery engine
pub struct PtxRecoveryEngine {
    /// CUBIN data
    cubin_data: Vec<u8>,
    /// Parsed CUBIN
    parsed_cubin: Option<ParsedCubin>,
    /// Debug information
    debug_info: Option<ParsedDebugInfo>,
    /// PTX reconstructor
    reconstructor: PtxReconstructor,
    /// PTX source cache (file -> content)
    ptx_source_cache: HashMap<String, String>,
}

impl PtxRecoveryEngine {
    /// Create a new PTX recovery engine from CUBIN data
    pub fn new(cubin_data: Vec<u8>) -> Result<Self, PtxRecoveryError> {
        let parser = CubinParser::new(cubin_data.clone());
        let parsed_cubin = parser
            .parse()
            .map_err(|e| PtxRecoveryError::CubinParse(e.to_string()))?;

        let dwarf_parser = DwarfParser::new(&cubin_data);
        let debug_info = dwarf_parser.parse().ok();

        let sm_version = parsed_cubin.sm_version;

        Ok(Self {
            cubin_data,
            parsed_cubin: Some(parsed_cubin),
            debug_info,
            reconstructor: PtxReconstructor::new(sm_version),
            ptx_source_cache: HashMap::new(),
        })
    }

    /// Create from CUBIN file path
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, PtxRecoveryError> {
        let data = fs::read(path.as_ref()).map_err(|e| PtxRecoveryError::IoError(e.to_string()))?;
        Self::new(data)
    }

    /// Add PTX source file for recovery
    pub fn add_ptx_source<P: AsRef<Path>>(&mut self, path: P) -> Result<(), PtxRecoveryError> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| PtxRecoveryError::IoError(e.to_string()))?;
        let name = path
            .as_ref()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.ptx")
            .to_string();
        self.ptx_source_cache.insert(name, content);
        Ok(())
    }

    /// Add PTX source content directly
    pub fn add_ptx_content(&mut self, name: String, content: String) {
        self.ptx_source_cache.insert(name, content);
    }

    /// Recover PTX for all kernels
    pub fn recover_all(&mut self) -> Result<PtxRecoveryResult, PtxRecoveryError> {
        let cubin = self
            .parsed_cubin
            .as_ref()
            .ok_or_else(|| PtxRecoveryError::NoCubin)?;

        let sm_version = cubin.sm_version;

        // Collect kernel info first to avoid borrow issues
        let kernel_info: Vec<(String, u64, Vec<u8>)> = cubin
            .kernels
            .iter()
            .map(|k| (k.name.clone(), k.address, k.code.clone()))
            .collect();

        let mut result = PtxRecoveryResult {
            functions: Vec::new(),
            ptx_header: self.generate_ptx_header(sm_version),
            debug_info: self.debug_info.clone(),
            source_files: Vec::new(),
            stats: RecoveryStats::default(),
        };

        // Collect source files
        if let Some(ref info) = self.debug_info {
            result.source_files = info.file_table.values().cloned().collect();
        }

        // Disassemble and recover each kernel
        let disasm = SassDisassembler::new(sm_version)
            .map_err(|e| PtxRecoveryError::Disassembly(e.to_string()))?;

        for (name, address, code) in kernel_info {
            let instructions = disasm.disassemble(&code, address);
            let recovered = self.recover_function(&name, address, &instructions)?;
            result.stats.total_sass_instructions += instructions.len() as u64;
            result.stats.mapped_instructions += recovered.sass_to_ptx.len() as u64;
            result.functions.push(recovered);
        }

        result.stats.unmapped_instructions =
            result.stats.total_sass_instructions - result.stats.mapped_instructions;

        Ok(result)
    }

    /// Recover PTX for a single function
    pub fn recover_function(
        &mut self,
        name: &str,
        start_address: u64,
        instructions: &[EnhancedSassInstruction],
    ) -> Result<RecoveredPtxFunction, PtxRecoveryError> {
        let mut func = RecoveredPtxFunction {
            name: name.to_string(),
            linkage_name: None,
            start_address,
            end_address: instructions
                .last()
                .map(|i| i.address)
                .unwrap_or(start_address),
            ptx_lines: BTreeMap::new(),
            sass_to_ptx: HashMap::new(),
            reconstructed_ptx: None,
            original_ptx: None,
        };

        // Try to get function debug info
        if let Some(ref debug_info) = self.debug_info {
            // Find matching function
            for f in &debug_info.functions {
                if f.name == name || f.linkage_name.as_deref() == Some(name) {
                    func.linkage_name = f.linkage_name.clone();
                    break;
                }
            }

            // Build SASS -> PTX mapping from debug line info
            for inst in instructions {
                if let Some(line_info) = debug_info.line_mappings.get(&inst.address) {
                    func.sass_to_ptx.insert(inst.address, line_info.line);

                    // Try to get source line
                    if let Some(source) = self.get_source_line(&line_info.file, line_info.line) {
                        func.ptx_lines
                            .entry(line_info.line)
                            .or_insert(PtxSourceLine {
                                line_number: line_info.line,
                                source,
                                file: line_info.file.clone(),
                                is_statement: line_info.is_statement,
                                column: line_info.column,
                            });
                    }
                }
            }
        }

        // Reconstruct PTX from SASS semantics
        let mut reconstructed_lines = Vec::new();
        reconstructed_lines.push(format!(".visible .entry {} (", name));
        reconstructed_lines.push("    // Parameters would go here".to_string());
        reconstructed_lines.push(")".to_string());
        reconstructed_lines.push("{".to_string());
        reconstructed_lines.push("    .reg .b32 %r<256>;".to_string());
        reconstructed_lines.push("    .reg .pred %p<8>;".to_string());
        reconstructed_lines.push(String::new());

        for inst in instructions {
            let ptx = self.reconstructor.reconstruct_instruction(inst);

            // Add comment with original SASS
            let comment = format!(
                "    // 0x{:04x}: {} {}",
                inst.address,
                inst.opcode,
                inst.src_operands
                    .iter()
                    .map(|op| format!("{:?}", op))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            reconstructed_lines.push(comment);
            reconstructed_lines.push(format!("    {}", ptx));
        }

        reconstructed_lines.push("}".to_string());
        func.reconstructed_ptx = Some(reconstructed_lines.join("\n"));

        Ok(func)
    }

    /// Get a source line from cached PTX files
    fn get_source_line(&self, file: &str, line: u32) -> Option<String> {
        // Try exact match first
        if let Some(content) = self.ptx_source_cache.get(file) {
            return content
                .lines()
                .nth((line - 1) as usize)
                .map(|s| s.to_string());
        }

        // Try matching by filename only
        let basename = Path::new(file)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(file);

        for (cached_name, content) in &self.ptx_source_cache {
            let cached_basename = Path::new(cached_name)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(cached_name);

            if cached_basename == basename {
                return content
                    .lines()
                    .nth((line - 1) as usize)
                    .map(|s| s.to_string());
            }
        }

        None
    }

    /// Generate PTX header with version and target
    fn generate_ptx_header(&self, sm_version: u32) -> String {
        format!(
            ".version 7.0\n\
             .target sm_{}\n\
             .address_size 64\n",
            sm_version
        )
    }

    /// Get mapping from SASS address to PTX line
    pub fn get_ptx_for_sass(&self, sass_address: u64) -> Option<&DebugLineInfo> {
        self.debug_info.as_ref()?.line_mappings.get(&sass_address)
    }

    /// Get all line mappings
    pub fn get_line_mappings(&self) -> Option<&HashMap<u64, DebugLineInfo>> {
        self.debug_info.as_ref().map(|d| &d.line_mappings)
    }

    /// Get function boundaries
    pub fn get_functions(&self) -> Option<&Vec<DebugFunctionInfo>> {
        self.debug_info.as_ref().map(|d| &d.functions)
    }
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug)]
pub enum PtxRecoveryError {
    CubinParse(String),
    Disassembly(String),
    IoError(String),
    NoCubin,
    NoDebugInfo,
}

impl fmt::Display for PtxRecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PtxRecoveryError::CubinParse(msg) => write!(f, "CUBIN parse error: {}", msg),
            PtxRecoveryError::Disassembly(msg) => write!(f, "Disassembly error: {}", msg),
            PtxRecoveryError::IoError(msg) => write!(f, "I/O error: {}", msg),
            PtxRecoveryError::NoCubin => write!(f, "No CUBIN data"),
            PtxRecoveryError::NoDebugInfo => write!(f, "No debug info available"),
        }
    }
}

impl std::error::Error for PtxRecoveryError {}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Recover PTX from CUBIN file
pub fn recover_ptx_from_cubin<P: AsRef<Path>>(
    cubin_path: P,
    ptx_paths: &[&Path],
) -> Result<PtxRecoveryResult, PtxRecoveryError> {
    let mut engine = PtxRecoveryEngine::from_file(cubin_path)?;

    for path in ptx_paths {
        engine.add_ptx_source(path)?;
    }

    engine.recover_all()
}

/// Get SASS to PTX mapping from CUBIN
pub fn get_sass_ptx_mapping(
    cubin_data: &[u8],
) -> Result<HashMap<u64, DebugLineInfo>, PtxRecoveryError> {
    let parser = DwarfParser::new(cubin_data);
    let info = parser
        .parse()
        .map_err(|e| PtxRecoveryError::CubinParse(e.to_string()))?;
    Ok(info.line_mappings)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_reconstructor_basic() {
        let mut recon = PtxReconstructor::new(75);

        let inst = EnhancedSassInstruction {
            address: 0,
            opcode: "FADD".to_string(),
            opcode_class: SassOpcodeClass::FloatArithmetic,
            modifiers: vec![],
            data_type: Some(SassDataType::F32),
            dest_operands: vec![SassOperand::Register(
                super::super::instruction::SassRegister {
                    prefix: "R".to_string(),
                    number: 0,
                    is_64bit: false,
                },
            )],
            src_operands: vec![
                SassOperand::Register(super::super::instruction::SassRegister {
                    prefix: "R".to_string(),
                    number: 1,
                    is_64bit: false,
                }),
                SassOperand::Register(super::super::instruction::SassRegister {
                    prefix: "R".to_string(),
                    number: 2,
                    is_64bit: false,
                }),
            ],
            predicate: None,
            encoding: None,
            latency: 0,
            throughput: 0.0,
            functional_unit: None,
            ptx_template: None,
            ptx_file: None,
            ptx_line: None,
            function_name: None,
            data_dependencies: vec![],
            basic_block_id: None,
        };

        let ptx = recon.reconstruct_instruction(&inst);
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("%r0"));
        assert!(ptx.contains("%r1"));
        assert!(ptx.contains("%r2"));
    }

    #[test]
    fn test_special_register_mapping() {
        let recon = PtxReconstructor::new(75);
        assert_eq!(recon.map_special_register("SR_TID.X"), "%tid.x");
        assert_eq!(recon.map_special_register("SR_CTAID.Y"), "%ctaid.y");
        assert_eq!(recon.map_special_register("SR_LANEID"), "%laneid");
    }
}
