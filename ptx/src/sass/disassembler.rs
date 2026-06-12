//! Native SASS disassembler for NVIDIA GPU machine code
//!
//! This module provides instruction decoding for SASS (Streaming ASSembler),
//! the native GPU assembly language for NVIDIA GPUs.
//!
//! SASS instruction encoding varies by GPU architecture:
//! - SM 5.x (Maxwell): 64-bit instructions with separate control words
//! - SM 6.x (Pascal): 64-bit instructions with control codes
//! - SM 7.x (Volta/Turing): 128-bit instructions with integrated control
//! - SM 8.x (Ampere): 128-bit instructions
//! - SM 9.x (Hopper): 128-bit instructions with new opcodes

use std::collections::HashMap;
use std::fmt;

use super::instruction::*;

// ============================================================================
// Architecture-Specific Encoding
// ============================================================================

/// SM architecture version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmVersion {
    /// Maxwell (GTX 9xx)
    Sm50,
    Sm52,
    Sm53,
    /// Pascal (GTX 10xx)
    Sm60,
    Sm61,
    Sm62,
    /// Volta/Turing (GTX 20xx, RTX 20xx)
    Sm70,
    Sm72,
    Sm75,
    /// Ampere (RTX 30xx)
    Sm80,
    Sm86,
    Sm87,
    /// Ada Lovelace (RTX 40xx)
    Sm89,
    /// Hopper
    Sm90,
    /// Blackwell
    Sm100,
    Sm120,
    Sm120a,
}

impl SmVersion {
    /// Parse from version number
    pub fn from_version(version: u32) -> Option<Self> {
        match version {
            50 => Some(SmVersion::Sm50),
            52 => Some(SmVersion::Sm52),
            53 => Some(SmVersion::Sm53),
            60 => Some(SmVersion::Sm60),
            61 => Some(SmVersion::Sm61),
            62 => Some(SmVersion::Sm62),
            70 => Some(SmVersion::Sm70),
            72 => Some(SmVersion::Sm72),
            75 => Some(SmVersion::Sm75),
            80 => Some(SmVersion::Sm80),
            86 => Some(SmVersion::Sm86),
            87 => Some(SmVersion::Sm87),
            89 => Some(SmVersion::Sm89),
            90 => Some(SmVersion::Sm90),
            100 => Some(SmVersion::Sm100),
            120 => Some(SmVersion::Sm120),
            121 => Some(SmVersion::Sm120a),
            _ => None,
        }
    }

    /// Get instruction size in bytes for this architecture
    pub fn instruction_size(&self) -> usize {
        match self {
            SmVersion::Sm50 | SmVersion::Sm52 | SmVersion::Sm53 => 8, // 64-bit
            SmVersion::Sm60 | SmVersion::Sm61 | SmVersion::Sm62 => 8, // 64-bit
            _ => 16,                                                  // 128-bit for Volta and later
        }
    }

    /// Does this architecture use 128-bit instructions?
    pub fn uses_128bit(&self) -> bool {
        match self {
            SmVersion::Sm50
            | SmVersion::Sm52
            | SmVersion::Sm53
            | SmVersion::Sm60
            | SmVersion::Sm61
            | SmVersion::Sm62 => false,
            _ => true,
        }
    }
}

// ============================================================================
// SASS Disassembler
// ============================================================================

/// SASS disassembler
pub struct SassDisassembler {
    /// Target SM version
    sm_version: SmVersion,
    /// Opcode table for this architecture
    opcode_table: OpcodeTable,
}

/// Opcode lookup table
struct OpcodeTable {
    /// Map from opcode bits to opcode name
    opcodes: HashMap<u16, &'static str>,
    /// SM version
    sm_version: SmVersion,
}

impl OpcodeTable {
    fn new(sm_version: SmVersion) -> Self {
        let mut opcodes = HashMap::new();

        // Populate with common opcodes
        // Note: These are approximate - real encoding varies by SM version
        // The opcode is typically in bits [63:52] or similar positions

        // Memory operations
        opcodes.insert(0x380, "LDG"); // Global load
        opcodes.insert(0x381, "LDG.E");
        opcodes.insert(0x385, "STG"); // Global store
        opcodes.insert(0x386, "STG.E");
        opcodes.insert(0x388, "LDS"); // Shared load
        opcodes.insert(0x389, "STS"); // Shared store
        opcodes.insert(0x38C, "LDL"); // Local load
        opcodes.insert(0x38D, "STL"); // Local store
        opcodes.insert(0x390, "LDC"); // Constant load

        // Atomics
        opcodes.insert(0x3A8, "ATOMG"); // Global atomic
        opcodes.insert(0x3A9, "ATOMS"); // Shared atomic
        opcodes.insert(0x3AC, "RED"); // Reduction

        // Integer arithmetic
        opcodes.insert(0x210, "IADD");
        opcodes.insert(0x211, "IADD3");
        opcodes.insert(0x214, "IMAD");
        opcodes.insert(0x215, "IMAD.WIDE");
        opcodes.insert(0x218, "IMUL");
        opcodes.insert(0x21C, "ISETP");
        opcodes.insert(0x21D, "ISET");
        opcodes.insert(0x220, "IABS");
        opcodes.insert(0x221, "INEG");

        // Floating point
        opcodes.insert(0x300, "FADD");
        opcodes.insert(0x304, "FMUL");
        opcodes.insert(0x308, "FFMA");
        opcodes.insert(0x30C, "FSETP");
        opcodes.insert(0x30D, "FSET");
        opcodes.insert(0x310, "FABS");
        opcodes.insert(0x311, "FNEG");
        opcodes.insert(0x318, "MUFU"); // Multi-function unit

        // Double precision
        opcodes.insert(0x320, "DADD");
        opcodes.insert(0x324, "DMUL");
        opcodes.insert(0x328, "DFMA");

        // Logic
        opcodes.insert(0x200, "LOP");
        opcodes.insert(0x201, "LOP3");
        opcodes.insert(0x204, "SHL");
        opcodes.insert(0x205, "SHR");
        opcodes.insert(0x208, "BFE");
        opcodes.insert(0x209, "BFI");
        opcodes.insert(0x20C, "FLO");
        opcodes.insert(0x20D, "POPC");

        // Conversion
        opcodes.insert(0x340, "I2I");
        opcodes.insert(0x344, "I2F");
        opcodes.insert(0x348, "F2I");
        opcodes.insert(0x34C, "F2F");

        // Data movement
        opcodes.insert(0x100, "MOV");
        opcodes.insert(0x101, "MOV32I");
        opcodes.insert(0x104, "PRMT");
        opcodes.insert(0x108, "SEL");
        opcodes.insert(0x10C, "SHFL");

        // Special registers
        opcodes.insert(0x110, "S2R");
        opcodes.insert(0x114, "CS2R");

        // Control flow
        opcodes.insert(0x000, "BRA");
        opcodes.insert(0x001, "BRX");
        opcodes.insert(0x004, "JMP");
        opcodes.insert(0x005, "JMX");
        opcodes.insert(0x008, "CAL");
        opcodes.insert(0x009, "JCAL");
        opcodes.insert(0x00C, "RET");
        opcodes.insert(0x010, "EXIT");

        // Synchronization
        opcodes.insert(0x020, "BAR");
        opcodes.insert(0x024, "DEPBAR");
        opcodes.insert(0x028, "MEMBAR");

        // Predicates
        opcodes.insert(0x180, "PSETP");
        opcodes.insert(0x184, "PLOP3");
        opcodes.insert(0x188, "P2R");
        opcodes.insert(0x18C, "R2P");

        // Texture
        opcodes.insert(0x400, "TEX");
        opcodes.insert(0x404, "TLD");
        opcodes.insert(0x408, "TLD4");
        opcodes.insert(0x40C, "TXQ");

        // Half precision
        opcodes.insert(0x360, "HADD2");
        opcodes.insert(0x364, "HMUL2");
        opcodes.insert(0x368, "HFMA2");

        // Tensor core
        opcodes.insert(0x3C0, "HMMA");
        opcodes.insert(0x3C4, "IMMA");

        // No-op
        opcodes.insert(0x7FF, "NOP");

        Self {
            opcodes,
            sm_version,
        }
    }

    fn lookup(&self, opcode_bits: u16) -> &'static str {
        self.opcodes.get(&opcode_bits).copied().unwrap_or("UNK")
    }
}

impl SassDisassembler {
    /// Create a new disassembler for the specified SM version
    pub fn new(sm_version: u32) -> Result<Self, DisassemblerError> {
        let version = SmVersion::from_version(sm_version)
            .ok_or(DisassemblerError::UnsupportedVersion(sm_version))?;

        Ok(Self {
            sm_version: version,
            opcode_table: OpcodeTable::new(version),
        })
    }

    /// Disassemble a block of SASS code
    pub fn disassemble(&self, code: &[u8], base_address: u64) -> Vec<EnhancedSassInstruction> {
        let mut instructions = Vec::new();
        let mut offset = 0;
        let inst_size = self.sm_version.instruction_size();

        while offset + inst_size <= code.len() {
            let inst = self.decode_instruction(&code[offset..], base_address + offset as u64);
            instructions.push(inst);
            offset += inst_size;
        }

        instructions
    }

    /// Decode a single instruction
    fn decode_instruction(&self, bytes: &[u8], address: u64) -> EnhancedSassInstruction {
        if self.sm_version.uses_128bit() {
            self.decode_128bit(bytes, address)
        } else {
            self.decode_64bit(bytes, address)
        }
    }

    /// Decode a 64-bit instruction (Maxwell/Pascal)
    fn decode_64bit(&self, bytes: &[u8], address: u64) -> EnhancedSassInstruction {
        if bytes.len() < 8 {
            return self.create_invalid_instruction(address);
        }

        let encoding = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);

        // Extract opcode (typically bits 63:52 or similar)
        // Note: This is simplified - real encoding is more complex
        let opcode_bits = ((encoding >> 52) & 0xFFF) as u16;
        let opcode_name = self.opcode_table.lookup(opcode_bits);

        // Extract predicate (typically bits 51:49 or similar)
        let pred_bits = ((encoding >> 16) & 0x7) as u8;
        let pred_neg = ((encoding >> 19) & 0x1) != 0;

        let predicate = if pred_bits < 7 {
            Some(SassOperand::Predicate {
                register: SassRegister::new("P", pred_bits as u32),
                negated: pred_neg,
            })
        } else {
            None // PT (always true)
        };

        // Extract destination register (simplified)
        let dest_reg = ((encoding >> 0) & 0xFF) as u32;
        let dest = if dest_reg < 255 {
            vec![SassOperand::Register(SassRegister::new("R", dest_reg))]
        } else {
            vec![SassOperand::Register(SassRegister::new("RZ", 255))]
        };

        // Extract source registers (simplified)
        let src1_reg = ((encoding >> 8) & 0xFF) as u32;
        let src2_reg = ((encoding >> 20) & 0xFF) as u32;
        let srcs = vec![
            SassOperand::Register(SassRegister::new("R", src1_reg)),
            SassOperand::Register(SassRegister::new("R", src2_reg)),
        ];

        // Determine opcode class
        let opcode_class = classify_opcode(opcode_name);

        EnhancedSassInstruction {
            opcode: opcode_name.to_string(),
            instruction_text: format!(
                "/*{:04x}*/ {} R{}, R{}, R{} ;",
                address, opcode_name, dest_reg, src1_reg, src2_reg
            ),
            address,
            size: 8,
            encoding_lo: encoding,
            encoding_hi: None,
            opcode_class,
            memory_space: get_memory_space(opcode_name),
            data_type: None,
            vector_width: 1,
            predicate,
            dest_operands: dest,
            src_operands: srcs,
            modifiers: Vec::new(),
            control_codes: SassControlCodes::default(),
            ptx_template: get_ptx_template(opcode_name),
            ptx_equivalent: None,
            ptx_file: None,
            ptx_line: None,
            ptx_column: None,
            function_name: None,
            data_dependencies: Vec::new(),
            basic_block_id: None,
        }
    }

    /// Decode a 128-bit instruction (Volta and later)
    fn decode_128bit(&self, bytes: &[u8], address: u64) -> EnhancedSassInstruction {
        if bytes.len() < 16 {
            return self.create_invalid_instruction(address);
        }

        let encoding_lo = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let encoding_hi = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);

        // For 128-bit instructions:
        // - Bits [63:0] contain the main instruction
        // - Bits [127:64] contain control codes (stall, barriers, reuse)

        // Extract opcode (simplified - real position varies)
        let opcode_bits = ((encoding_lo >> 52) & 0xFFF) as u16;
        let opcode_name = self.opcode_table.lookup(opcode_bits);

        // Extract control codes from upper 64 bits
        let stall_count = ((encoding_hi >> 0) & 0xF) as u8;
        let yield_flag = ((encoding_hi >> 4) & 0x1) != 0;
        let write_barrier = ((encoding_hi >> 5) & 0x7) as u8;
        let read_barrier = ((encoding_hi >> 8) & 0x7) as u8;
        let wait_barrier = ((encoding_hi >> 11) & 0x3F) as u8;
        let reuse_flags = ((encoding_hi >> 17) & 0xF) as u8;

        // Extract predicate
        let pred_bits = ((encoding_lo >> 16) & 0x7) as u8;
        let pred_neg = ((encoding_lo >> 19) & 0x1) != 0;

        let predicate = if pred_bits < 7 {
            Some(SassOperand::Predicate {
                register: SassRegister::new("P", pred_bits as u32),
                negated: pred_neg,
            })
        } else {
            None
        };

        // Extract registers (simplified)
        let dest_reg = ((encoding_lo >> 0) & 0xFF) as u32;
        let src1_reg = ((encoding_lo >> 8) & 0xFF) as u32;
        let src2_reg = ((encoding_lo >> 20) & 0xFF) as u32;
        let src3_reg = ((encoding_lo >> 28) & 0xFF) as u32;

        let dest = if dest_reg < 255 {
            vec![SassOperand::Register(SassRegister::new("R", dest_reg))]
        } else {
            vec![SassOperand::Register(SassRegister::new("RZ", 255))]
        };

        let mut srcs = vec![SassOperand::Register(SassRegister::new("R", src1_reg))];
        if src2_reg < 255 {
            srcs.push(SassOperand::Register(SassRegister::new("R", src2_reg)));
        }
        if src3_reg < 255 {
            srcs.push(SassOperand::Register(SassRegister::new("R", src3_reg)));
        }

        // Determine opcode class and other semantic info
        let opcode_class = classify_opcode(opcode_name);
        let memory_space = get_memory_space(opcode_name);

        EnhancedSassInstruction {
            opcode: opcode_name.to_string(),
            instruction_text: format!(
                "/*{:04x}*/ {} R{}, R{}, R{} ;",
                address, opcode_name, dest_reg, src1_reg, src2_reg
            ),
            address,
            size: 16,
            encoding_lo,
            encoding_hi: Some(encoding_hi),
            opcode_class,
            memory_space,
            data_type: None,
            vector_width: 1,
            predicate,
            dest_operands: dest,
            src_operands: srcs,
            modifiers: Vec::new(),
            control_codes: SassControlCodes {
                stall_count,
                yield_flag,
                write_barrier,
                read_barrier,
                wait_barrier_mask: wait_barrier,
                reuse_flags,
            },
            ptx_template: get_ptx_template(opcode_name),
            ptx_equivalent: None,
            ptx_file: None,
            ptx_line: None,
            ptx_column: None,
            function_name: None,
            data_dependencies: Vec::new(),
            basic_block_id: None,
        }
    }

    /// Create an invalid/unknown instruction marker
    fn create_invalid_instruction(&self, address: u64) -> EnhancedSassInstruction {
        EnhancedSassInstruction {
            opcode: "???".to_string(),
            instruction_text: format!("/*{:04x}*/ ???", address),
            address,
            size: self.sm_version.instruction_size() as u8,
            encoding_lo: 0,
            encoding_hi: None,
            opcode_class: SassOpcodeClass::Unknown,
            memory_space: None,
            data_type: None,
            vector_width: 1,
            predicate: None,
            dest_operands: Vec::new(),
            src_operands: Vec::new(),
            modifiers: Vec::new(),
            control_codes: SassControlCodes::default(),
            ptx_template: PtxTemplate::NoEquivalent,
            ptx_equivalent: None,
            ptx_file: None,
            ptx_line: None,
            ptx_column: None,
            function_name: None,
            data_dependencies: Vec::new(),
            basic_block_id: None,
        }
    }

    /// Get the SM version
    pub fn sm_version(&self) -> SmVersion {
        self.sm_version
    }
}

// ============================================================================
// Text-based Disassembly Parser (for cuobjdump output)
// ============================================================================

/// Parser for cuobjdump-style text output
pub struct TextDisassemblyParser;

impl TextDisassemblyParser {
    /// Parse a single instruction line from cuobjdump output
    pub fn parse_instruction_line(line: &str) -> Option<EnhancedSassInstruction> {
        let line = line.trim();

        // Format: /*0050*/ @P0 LDG.E.U32 R0, [R2.64+0x10] ;

        // Extract address from /*xxxx*/
        if !line.starts_with("/*") {
            return None;
        }

        let addr_end = line.find("*/")?;
        let addr_str = line.get(2..addr_end)?.trim();
        let address = u64::from_str_radix(addr_str, 16).ok()?;

        // Get rest of line after address
        let rest = line
            .get(addr_end + 2..)?
            .trim()
            .trim_end_matches(';')
            .trim();

        // Check for predicate (@P0, @!P1, etc.)
        let (predicate, instruction_part) = if rest.starts_with('@') {
            let space_idx = rest.find(' ').unwrap_or(rest.len());
            let pred_str = rest.get(..space_idx)?;
            let pred_op = SassOperand::parse(pred_str)?;
            (Some(pred_op), rest.get(space_idx..)?.trim())
        } else {
            (None, rest)
        };

        // Split into opcode+modifiers and operands
        let parts: Vec<&str> = instruction_part.split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }

        let opcode_full = parts[0];
        // Split opcode from modifiers (e.g., "LDG.E.U32" -> "LDG", [".E", ".U32"])
        let opcode_parts: Vec<&str> = opcode_full.split('.').collect();
        let opcode = opcode_parts[0].to_string();
        let modifiers: Vec<String> = opcode_parts[1..].iter().map(|s| s.to_string()).collect();

        // Parse data type from modifiers
        let data_type = modifiers
            .iter()
            .find_map(|m| SassDataType::from_modifier(m));

        // Parse operands
        let operands_str = parts[1..].join(" ");
        let operand_strs: Vec<&str> = operands_str.split(',').map(|s| s.trim()).collect();

        // First operand is typically destination
        let dest_operands: Vec<SassOperand> = if !operand_strs.is_empty() {
            SassOperand::parse(operand_strs[0]).into_iter().collect()
        } else {
            Vec::new()
        };

        // Rest are source operands
        let src_operands: Vec<SassOperand> = operand_strs[1..]
            .iter()
            .filter_map(|s| SassOperand::parse(s))
            .collect();

        // Determine opcode class
        let opcode_class = classify_opcode(&opcode);
        let memory_space = get_memory_space(&opcode);

        Some(EnhancedSassInstruction {
            opcode: opcode.clone(),
            instruction_text: line.to_string(),
            address,
            size: 16, // Assume 128-bit
            encoding_lo: 0,
            encoding_hi: None,
            opcode_class,
            memory_space,
            data_type,
            vector_width: 1,
            predicate,
            dest_operands,
            src_operands,
            modifiers,
            control_codes: SassControlCodes::default(),
            ptx_template: get_ptx_template(&opcode),
            ptx_equivalent: None,
            ptx_file: None,
            ptx_line: None,
            ptx_column: None,
            function_name: None,
            data_dependencies: Vec::new(),
            basic_block_id: None,
        })
    }

    /// Parse full cuobjdump -sass -lineinfo output
    pub fn parse_cuobjdump_output(output: &str) -> Vec<EnhancedSassInstruction> {
        let mut instructions = Vec::new();
        let mut current_file = String::from("kernel.ptx");
        let mut current_line = 0u32;
        let mut current_function = None;

        for line in output.lines() {
            let line = line.trim();

            // Parse function header: "Function : kernel_name"
            if line.starts_with("Function :") || line.starts_with("function :") {
                if let Some(name) = line.split(':').nth(1) {
                    current_function = Some(name.trim().to_string());
                }
                continue;
            }

            // Parse file reference: ## File "kernel.ptx", line 16
            if line.contains("## File") {
                if let Some(file_part) = line.split('"').nth(1) {
                    current_file = file_part.to_string();
                }
                if let Some(line_part) = line.split("line ").nth(1) {
                    if let Ok(line_num) = line_part
                        .trim_end_matches(|c: char| !c.is_ascii_digit())
                        .parse::<u32>()
                    {
                        current_line = line_num;
                    }
                }
                continue;
            }

            // Parse line marker: ## Line 16
            if let Some(line_marker) = line.strip_prefix("## Line ") {
                if let Ok(line_num) = line_marker.trim().parse::<u32>() {
                    current_line = line_num;
                }
                continue;
            }

            // Parse instruction
            if line.starts_with("/*") {
                if let Some(mut inst) = Self::parse_instruction_line(line) {
                    inst.ptx_file = Some(current_file.clone());
                    inst.ptx_line = Some(current_line);
                    inst.function_name = current_function.clone();
                    instructions.push(inst);
                }
            }
        }

        instructions
    }
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug)]
pub enum DisassemblerError {
    UnsupportedVersion(u32),
    InvalidInstruction(u64),
    ParseError(String),
}

impl fmt::Display for DisassemblerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisassemblerError::UnsupportedVersion(v) => write!(f, "Unsupported SM version: {}", v),
            DisassemblerError::InvalidInstruction(addr) => {
                write!(f, "Invalid instruction at 0x{:x}", addr)
            }
            DisassemblerError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for DisassemblerError {}

// ============================================================================
// Control Flow Analysis
// ============================================================================

/// Control flow analyzer for SASS code
pub struct ControlFlowAnalyzer;

impl ControlFlowAnalyzer {
    /// Identify basic blocks in a sequence of instructions
    pub fn find_basic_blocks(instructions: &mut [EnhancedSassInstruction]) {
        if instructions.is_empty() {
            return;
        }

        // Build set of branch targets
        let mut branch_targets: std::collections::HashSet<u64> = std::collections::HashSet::new();

        for inst in instructions.iter() {
            if inst.opcode_class.is_control_flow() {
                // Check for branch/jump targets in source operands
                for op in &inst.src_operands {
                    if let SassOperand::Address(addr) = op {
                        branch_targets.insert(*addr);
                    }
                    if let SassOperand::Label(label) = op {
                        // Would need to resolve label to address
                    }
                }
            }
        }

        // Assign basic block IDs
        let mut current_block = 0u32;
        let mut new_block_next = true;

        for inst in instructions.iter_mut() {
            // Start new block if:
            // 1. This is a branch target
            // 2. Previous instruction was a control flow instruction
            if new_block_next || branch_targets.contains(&inst.address) {
                current_block += 1;
                new_block_next = false;
            }

            inst.basic_block_id = Some(current_block);

            // Next instruction starts new block if this is control flow
            if inst.opcode_class.is_control_flow() {
                new_block_next = true;
            }
        }
    }

    /// Compute data dependencies between instructions
    pub fn analyze_data_flow(instructions: &mut [EnhancedSassInstruction]) {
        // Map from register to the address of instruction that last wrote it
        let mut last_writer: HashMap<String, u64> = HashMap::new();

        for inst in instructions.iter_mut() {
            // Add dependencies for source operands
            for src in &inst.src_operands {
                if let SassOperand::Register(reg) = src {
                    let key = format!("{}{}", reg.prefix, reg.number);
                    if let Some(&writer_addr) = last_writer.get(&key) {
                        inst.data_dependencies.push(writer_addr);
                    }
                }
            }

            // Update last writer for destination operands
            for dest in &inst.dest_operands {
                if let SassOperand::Register(reg) = dest {
                    if !reg.is_zero {
                        let key = format!("{}{}", reg.prefix, reg.number);
                        last_writer.insert(key, inst.address);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sm_version_parse() {
        assert!(SmVersion::from_version(61).is_some());
        assert!(SmVersion::from_version(86).is_some());
        assert!(SmVersion::from_version(99).is_none());
    }

    #[test]
    fn test_instruction_size() {
        assert_eq!(SmVersion::Sm61.instruction_size(), 8);
        assert_eq!(SmVersion::Sm86.instruction_size(), 16);
    }

    #[test]
    fn test_text_parse_instruction() {
        let line = "/*0050*/ LDG.E.U32 R0, [R2.64] ;";
        let inst = TextDisassemblyParser::parse_instruction_line(line).unwrap();

        assert_eq!(inst.address, 0x50);
        assert_eq!(inst.opcode, "LDG");
        assert!(inst.modifiers.contains(&"E".to_string()));
        assert!(inst.modifiers.contains(&"U32".to_string()));
        assert_eq!(inst.opcode_class, SassOpcodeClass::GlobalLoad);
    }

    #[test]
    fn test_text_parse_with_predicate() {
        let line = "/*0100*/ @P0 BRA 0x200 ;";
        let inst = TextDisassemblyParser::parse_instruction_line(line).unwrap();

        assert_eq!(inst.address, 0x100);
        assert_eq!(inst.opcode, "BRA");
        assert!(inst.predicate.is_some());
    }

    #[test]
    fn test_cuobjdump_parse() {
        let output = r#"
Function : test_kernel
## File "kernel.ptx", line 10
/*0000*/ MOV R1, c[0x0][0x20] ;
## Line 12
/*0010*/ LDG.E.U64 R2, [R4] ;
/*0020*/ FADD R0, R1, R2 ;
"#;

        let instructions = TextDisassemblyParser::parse_cuobjdump_output(output);
        assert_eq!(instructions.len(), 3);

        assert_eq!(instructions[0].ptx_line, Some(10));
        assert_eq!(instructions[1].ptx_line, Some(12));
        assert_eq!(instructions[2].ptx_line, Some(12));

        assert_eq!(
            instructions[0].function_name,
            Some("test_kernel".to_string())
        );
    }

    #[test]
    fn test_basic_block_detection() {
        let mut instructions = vec![
            EnhancedSassInstruction::new("MOV".to_string(), 0x00),
            EnhancedSassInstruction::new("ADD".to_string(), 0x10),
            EnhancedSassInstruction::new("BRA".to_string(), 0x20),
            EnhancedSassInstruction::new("MOV".to_string(), 0x30),
        ];

        ControlFlowAnalyzer::find_basic_blocks(&mut instructions);

        assert_eq!(instructions[0].basic_block_id, Some(1));
        assert_eq!(instructions[1].basic_block_id, Some(1));
        assert_eq!(instructions[2].basic_block_id, Some(1));
        assert_eq!(instructions[3].basic_block_id, Some(2)); // New block after BRA
    }
}
