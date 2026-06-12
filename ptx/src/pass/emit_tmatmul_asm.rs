// emit_tmatmul_asm.rs - TMatmul assembly code generator
// This module generates tmatmul assembly from MLIR tmatmul dialect operations

use std::collections::HashMap;
use std::fmt::Write;

/// Number of vector registers available in hardware (v0-v7)
const NUM_REGISTERS: usize = 8;

/// Spill slot information
#[derive(Clone, Debug)]
struct SpillSlot {
    /// Memory location name (e.g., "SPILL_0", "SPILL_1")
    memory_name: String,
    /// SSA value stored in this slot
    ssa_value: String,
}

/// Register allocator for tmatmul vector registers (v0-v7) with spilling support
pub struct RegisterAllocator {
    /// Maps MLIR SSA values to physical registers
    value_to_reg: HashMap<String, String>,
    /// Tracks which registers are currently in use
    allocated_regs: [bool; NUM_REGISTERS],
    /// Next free register to try
    next_reg: usize,
    /// Spill slots for values that don't fit in registers
    spill_slots: HashMap<String, SpillSlot>,
    /// Next available spill slot index
    next_spill_slot: usize,
    /// Pending spill operations (to be emitted before next instruction)
    pending_spills: Vec<(String, String)>, // (reg, memory_name)
    /// Pending reload operations
    pending_reloads: Vec<(String, String)>, // (memory_name, reg)
}

impl RegisterAllocator {
    pub fn new() -> Self {
        Self {
            value_to_reg: HashMap::new(),
            allocated_regs: [false; NUM_REGISTERS],
            next_reg: 0,
            spill_slots: HashMap::new(),
            next_spill_slot: 0,
            pending_spills: Vec::new(),
            pending_reloads: Vec::new(),
        }
    }

    /// Allocate a new register for an SSA value, with spilling if needed
    pub fn allocate(&mut self, ssa_value: &str) -> Result<String, String> {
        // Check if already allocated in register
        if let Some(reg) = self.value_to_reg.get(ssa_value) {
            return Ok(reg.clone());
        }

        // Check if spilled - need to reload
        if let Some(slot) = self.spill_slots.get(ssa_value).cloned() {
            // Need to reload from spill slot
            // First, get a register (may cause another spill)
            let reg = self.allocate_physical_register(ssa_value)?;
            // Queue reload operation
            self.pending_reloads
                .push((slot.memory_name.clone(), reg.clone()));
            // Remove from spill slots
            self.spill_slots.remove(ssa_value);
            return Ok(reg);
        }

        // Fresh allocation
        self.allocate_physical_register(ssa_value)
    }

    /// Allocate a physical register, spilling if necessary
    fn allocate_physical_register(&mut self, ssa_value: &str) -> Result<String, String> {
        // Find lowest free register
        for reg_idx in 0..NUM_REGISTERS {
            if !self.allocated_regs[reg_idx] {
                let reg_name = format!("v{}", reg_idx);
                self.allocated_regs[reg_idx] = true;
                self.value_to_reg
                    .insert(ssa_value.to_string(), reg_name.clone());
                self.next_reg = (reg_idx + 1) % NUM_REGISTERS;
                return Ok(reg_name);
            }
        }

        // All registers used - need to spill one
        // Use simple LRU-like strategy: spill the oldest allocated register
        // Find a victim (pick the one with highest index to preserve common values in low regs)
        let victim_idx = (self.next_reg + NUM_REGISTERS - 1) % NUM_REGISTERS;
        let victim_reg = format!("v{}", victim_idx);

        // Find which SSA value is in this register
        let victim_ssa = self
            .value_to_reg
            .iter()
            .find(|(_, r)| **r == victim_reg)
            .map(|(s, _)| s.clone());

        if let Some(victim_ssa) = victim_ssa {
            // Create spill slot
            let spill_name = format!("SPILL_{}", self.next_spill_slot);
            self.next_spill_slot += 1;

            // Queue spill operation
            self.pending_spills
                .push((victim_reg.clone(), spill_name.clone()));

            // Record spill slot
            self.spill_slots.insert(
                victim_ssa.clone(),
                SpillSlot {
                    memory_name: spill_name,
                    ssa_value: victim_ssa.clone(),
                },
            );

            // Free the victim from register mapping
            self.value_to_reg.remove(&victim_ssa);
        }

        // Now allocate the freed register for the new value
        self.allocated_regs[victim_idx] = true;
        self.value_to_reg
            .insert(ssa_value.to_string(), victim_reg.clone());
        self.next_reg = (victim_idx + 1) % NUM_REGISTERS;

        Ok(victim_reg)
    }

    /// Free a register
    pub fn free(&mut self, ssa_value: &str) {
        if let Some(reg) = self.value_to_reg.remove(ssa_value) {
            if let Some(idx) = reg.strip_prefix('v').and_then(|s| s.parse::<usize>().ok()) {
                if idx < NUM_REGISTERS {
                    self.allocated_regs[idx] = false;
                }
            }
        }
        // Also remove from spill slots if present
        self.spill_slots.remove(ssa_value);
    }

    /// Get register for an SSA value
    pub fn get_register(&self, ssa_value: &str) -> Option<&String> {
        self.value_to_reg.get(ssa_value)
    }

    /// Check if a value is spilled
    pub fn is_spilled(&self, ssa_value: &str) -> bool {
        self.spill_slots.contains_key(ssa_value)
    }

    /// Get pending spill operations and clear them
    pub fn take_pending_spills(&mut self) -> Vec<(String, String)> {
        std::mem::take(&mut self.pending_spills)
    }

    /// Get pending reload operations and clear them
    pub fn take_pending_reloads(&mut self) -> Vec<(String, String)> {
        std::mem::take(&mut self.pending_reloads)
    }

    /// Get number of spill slots used
    pub fn spill_count(&self) -> usize {
        self.next_spill_slot
    }
}

/// Memory location for DDR memory references
#[derive(Debug, Clone)]
pub enum MemoryLocation {
    X,
    WG,
    WF,
    WC,
    WO,
    WU1,
    WU2,
    WN,
    OH,
    B,
    O,
    TempVec,
    Custom(String),
}

impl std::fmt::Display for MemoryLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryLocation::X => write!(f, "X"),
            MemoryLocation::WG => write!(f, "WG"),
            MemoryLocation::WF => write!(f, "WF"),
            MemoryLocation::WC => write!(f, "WC"),
            MemoryLocation::WO => write!(f, "WO"),
            MemoryLocation::WU1 => write!(f, "WU1"),
            MemoryLocation::WU2 => write!(f, "WU2"),
            MemoryLocation::WN => write!(f, "WN"),
            MemoryLocation::OH => write!(f, "oH"),
            MemoryLocation::B => write!(f, "B"),
            MemoryLocation::O => write!(f, "O"),
            MemoryLocation::TempVec => write!(f, "TEMP_VEC"),
            MemoryLocation::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// TMatmul assembly instruction
#[derive(Debug, Clone)]
pub enum TMatmulInstruction {
    // Memory operations
    LoadVector {
        dst: String,
        src: MemoryLocation,
    },
    StoreVector {
        src: String,
        dst: MemoryLocation,
    },

    // Arithmetic operations
    Add {
        dst: String,
        src1: String,
        src2: String,
    },
    Sub {
        dst: String,
        src1: String,
        src2: String,
    },
    Mul {
        dst: String,
        src1: String,
        src2: String,
    },
    Div {
        dst: String,
        src1: String,
        src2: String,
    },

    // Activation functions
    Sigmoid {
        dst: String,
        src: String,
    },
    ComplementSigmoid {
        dst: String,
        src: String,
    },
    SiLU {
        dst: String,
        src: String,
    },

    // Special operations
    Norm {
        dst: String,
        src: String,
    },
    TMatmulImport {
        src: String,
    },
    TMatmulGo {
        weights: MemoryLocation,
    },
    TMatmulExport {
        dst: String,
    },

    // Comments and labels
    Comment(String),
    Label(String),
}

impl std::fmt::Display for TMatmulInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TMatmulInstruction::LoadVector { dst, src } => {
                write!(f, "    ldv    {},{}", dst, src)
            }
            TMatmulInstruction::StoreVector { src, dst } => {
                write!(f, "    sv    {},{}", src, dst)
            }
            TMatmulInstruction::Add { dst, src1, src2 } => {
                write!(f, "    add    {},{},{}", dst, src1, src2)
            }
            TMatmulInstruction::Sub { dst, src1, src2 } => {
                write!(f, "    sub    {},{},{}", dst, src1, src2)
            }
            TMatmulInstruction::Mul { dst, src1, src2 } => {
                write!(f, "    mul    {},{},{}", dst, src1, src2)
            }
            TMatmulInstruction::Div { dst, src1, src2 } => {
                write!(f, "    div    {},{},{}", dst, src1, src2)
            }
            TMatmulInstruction::Sigmoid { dst, src } => {
                write!(f, "    sig    {},{}", dst, src)
            }
            TMatmulInstruction::ComplementSigmoid { dst, src } => {
                write!(f, "    csig    {},{}", dst, src)
            }
            TMatmulInstruction::SiLU { dst, src } => {
                write!(f, "    silu    {},{}", dst, src)
            }
            TMatmulInstruction::Norm { dst, src } => {
                write!(f, "    norm    {},{}", dst, src)
            }
            TMatmulInstruction::TMatmulImport { src } => {
                write!(f, "    tmatmul_import    {}", src)
            }
            TMatmulInstruction::TMatmulGo { weights } => {
                write!(f, "    tmatmul_go    {}", weights)
            }
            TMatmulInstruction::TMatmulExport { dst } => {
                write!(f, "    tmatmul_export    {}", dst)
            }
            TMatmulInstruction::Comment(text) => {
                write!(f, "    ; {}", text)
            }
            TMatmulInstruction::Label(name) => {
                write!(f, "{}:", name)
            }
        }
    }
}

/// TMatmul assembly program
pub struct TMatmulProgram {
    pub instructions: Vec<TMatmulInstruction>,
}

impl TMatmulProgram {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    pub fn add_instruction(&mut self, inst: TMatmulInstruction) {
        self.instructions.push(inst);
    }

    pub fn add_comment(&mut self, comment: &str) {
        self.instructions
            .push(TMatmulInstruction::Comment(comment.to_string()));
    }

    pub fn add_section_separator(&mut self, section_name: &str) {
        self.instructions
            .push(TMatmulInstruction::Comment("".to_string()));
        self.instructions.push(TMatmulInstruction::Comment(format!(
            "-------------------------------------------------------"
        )));
        self.instructions
            .push(TMatmulInstruction::Comment(format!(" {}", section_name)));
        self.instructions.push(TMatmulInstruction::Comment(format!(
            "-------------------------------------------------------"
        )));
        self.instructions
            .push(TMatmulInstruction::Comment("".to_string()));
    }

    /// Generate assembly code
    pub fn to_assembly(&self) -> String {
        let mut output = String::new();

        // Header
        writeln!(
            &mut output,
            ";-------------------------------------------------------"
        )
        .unwrap();
        writeln!(&mut output, "; vector registers: v0..v7").unwrap();
        writeln!(&mut output, ";").unwrap();
        writeln!(
            &mut output,
            "; DDR: X, WG, WF, WC, WO, WU1, WU2, WN, oH, B, O, TEMP_VEC"
        )
        .unwrap();
        writeln!(&mut output, ";").unwrap();
        writeln!(&mut output, "; instructions:").unwrap();
        writeln!(&mut output, "; memory: ldv sv").unwrap();
        writeln!(&mut output, "; operations: add sub mul div sig csig silu").unwrap();
        writeln!(
            &mut output,
            "; misc: norm tmatmul_import tmatmul_go tmatmul_export"
        )
        .unwrap();
        writeln!(
            &mut output,
            ";-------------------------------------------------------"
        )
        .unwrap();
        writeln!(&mut output).unwrap();

        // Instructions
        for inst in &self.instructions {
            writeln!(&mut output, "{}", inst).unwrap();
        }

        output
    }
}

/// MLIR to TMatmul assembly converter
pub struct TMatmulCodegen {
    program: TMatmulProgram,
    reg_allocator: RegisterAllocator,
    mem_map: HashMap<String, MemoryLocation>,
}

impl TMatmulCodegen {
    pub fn new() -> Self {
        Self {
            program: TMatmulProgram::new(),
            reg_allocator: RegisterAllocator::new(),
            mem_map: HashMap::new(),
        }
    }

    /// Map an MLIR symbol to a memory location
    pub fn map_memory(&mut self, symbol: &str, location: MemoryLocation) {
        self.mem_map.insert(symbol.to_string(), location);
    }

    /// Emit pending spill instructions (save registers to memory before they're reused)
    fn emit_pending_spills(&mut self) {
        let spills = self.reg_allocator.take_pending_spills();
        for (reg, mem_name) in spills {
            self.program
                .add_comment(&format!("Spill {} to {}", reg, mem_name));
            self.program
                .add_instruction(TMatmulInstruction::StoreVector {
                    src: reg,
                    dst: MemoryLocation::Custom(mem_name),
                });
        }
    }

    /// Emit pending reload instructions (restore registers from memory)
    fn emit_pending_reloads(&mut self) {
        let reloads = self.reg_allocator.take_pending_reloads();
        for (mem_name, reg) in reloads {
            self.program
                .add_comment(&format!("Reload {} from {}", reg, mem_name));
            self.program
                .add_instruction(TMatmulInstruction::LoadVector {
                    dst: reg,
                    src: MemoryLocation::Custom(mem_name),
                });
        }
    }

    /// Ensure an SSA value is in a register (allocate/reload if needed)
    /// Returns the register name
    fn ensure_in_register(&mut self, ssa_value: &str) -> Result<String, String> {
        let reg = self.reg_allocator.allocate(ssa_value)?;
        // Emit any pending reloads immediately
        self.emit_pending_reloads();
        Ok(reg)
    }

    /// Generate code for an MLIR tmatmul operation
    pub fn emit_operation(
        &mut self,
        op: &str,
        operands: &[&str],
        results: &[&str],
    ) -> Result<(), String> {
        match op {
            "tmatmul.ldv" => {
                // Emit any pending spills first
                self.emit_pending_spills();
                let dst = self.reg_allocator.allocate(results[0])?;
                self.emit_pending_spills(); // Emit spills from this allocation
                let src_mem = self
                    .mem_map
                    .get(operands[0])
                    .cloned()
                    .unwrap_or(MemoryLocation::Custom(operands[0].to_string()));
                self.program
                    .add_instruction(TMatmulInstruction::LoadVector { dst, src: src_mem });
            }
            "tmatmul.sv" => {
                // Get source first (may need reload)
                let src = self.ensure_in_register(operands[0])?;
                self.emit_pending_spills();
                let dst_mem = self
                    .mem_map
                    .get(operands[1])
                    .cloned()
                    .unwrap_or(MemoryLocation::Custom(operands[1].to_string()));
                self.program
                    .add_instruction(TMatmulInstruction::StoreVector { src, dst: dst_mem });
            }
            "tmatmul.add" => {
                // Get operands first (may need reload), BEFORE allocating dst
                let src1 = self.ensure_in_register(operands[0])?;
                let src2 = self.ensure_in_register(operands[1])?;
                self.emit_pending_spills();
                // Now allocate dst (may spill, but we have src1/src2 names saved)
                let dst = self.reg_allocator.allocate(results[0])?;
                self.emit_pending_spills();
                self.program
                    .add_instruction(TMatmulInstruction::Add { dst, src1, src2 });
            }
            "tmatmul.mul" => {
                let src1 = self.ensure_in_register(operands[0])?;
                let src2 = self.ensure_in_register(operands[1])?;
                self.emit_pending_spills();
                let dst = self.reg_allocator.allocate(results[0])?;
                self.emit_pending_spills();
                self.program
                    .add_instruction(TMatmulInstruction::Mul { dst, src1, src2 });
            }
            "tmatmul.sub" => {
                let src1 = self.ensure_in_register(operands[0])?;
                let src2 = self.ensure_in_register(operands[1])?;
                self.emit_pending_spills();
                let dst = self.reg_allocator.allocate(results[0])?;
                self.emit_pending_spills();
                self.program
                    .add_instruction(TMatmulInstruction::Sub { dst, src1, src2 });
            }
            "tmatmul.div" => {
                let src1 = self.ensure_in_register(operands[0])?;
                let src2 = self.ensure_in_register(operands[1])?;
                self.emit_pending_spills();
                let dst = self.reg_allocator.allocate(results[0])?;
                self.emit_pending_spills();
                self.program
                    .add_instruction(TMatmulInstruction::Div { dst, src1, src2 });
            }
            "tmatmul.sig" => {
                let src = self.ensure_in_register(operands[0])?;
                self.emit_pending_spills();
                let dst = self.reg_allocator.allocate(results[0])?;
                self.emit_pending_spills();
                self.program
                    .add_instruction(TMatmulInstruction::Sigmoid { dst, src });
            }
            "tmatmul.csig" => {
                let src = self.ensure_in_register(operands[0])?;
                self.emit_pending_spills();
                let dst = self.reg_allocator.allocate(results[0])?;
                self.emit_pending_spills();
                self.program
                    .add_instruction(TMatmulInstruction::ComplementSigmoid { dst, src });
            }
            "tmatmul.silu" => {
                let src = self.ensure_in_register(operands[0])?;
                self.emit_pending_spills();
                let dst = self.reg_allocator.allocate(results[0])?;
                self.emit_pending_spills();
                self.program
                    .add_instruction(TMatmulInstruction::SiLU { dst, src });
            }
            "tmatmul.norm" => {
                let src = self.ensure_in_register(operands[0])?;
                self.emit_pending_spills();
                let dst = self.reg_allocator.allocate(results[0])?;
                self.emit_pending_spills();
                self.program
                    .add_instruction(TMatmulInstruction::Norm { dst, src });
            }
            "tmatmul.tmatmul_import" => {
                let src = self.ensure_in_register(operands[0])?;
                self.emit_pending_spills();
                self.program
                    .add_instruction(TMatmulInstruction::TMatmulImport { src });
            }
            "tmatmul.tmatmul_go" => {
                let weights = self
                    .mem_map
                    .get(operands[0])
                    .cloned()
                    .unwrap_or(MemoryLocation::Custom(operands[0].to_string()));
                self.program
                    .add_instruction(TMatmulInstruction::TMatmulGo { weights });
            }
            "tmatmul.tmatmul_export" => {
                self.emit_pending_spills();
                let dst = self.reg_allocator.allocate(results[0])?;
                self.emit_pending_spills();
                self.program
                    .add_instruction(TMatmulInstruction::TMatmulExport { dst });
            }
            "tmatmul.matmul" => {
                // Combined operation: import -> go -> export
                let input_reg = self.ensure_in_register(operands[0])?;
                self.emit_pending_spills();
                let weights = self
                    .mem_map
                    .get(operands[1])
                    .cloned()
                    .unwrap_or(MemoryLocation::Custom(operands[1].to_string()));
                let output_reg = self.reg_allocator.allocate(results[0])?;
                self.emit_pending_spills();

                self.program
                    .add_instruction(TMatmulInstruction::TMatmulImport { src: input_reg });
                self.program
                    .add_instruction(TMatmulInstruction::TMatmulGo { weights });
                self.program
                    .add_instruction(TMatmulInstruction::TMatmulExport { dst: output_reg });
            }
            _ => {
                return Err(format!("Unknown operation: {}", op));
            }
        }

        Ok(())
    }

    pub fn add_comment(&mut self, comment: &str) {
        self.program.add_comment(comment);
    }

    pub fn add_section(&mut self, section_name: &str) {
        self.program.add_section_separator(section_name);
    }

    /// Get the generated assembly code
    pub fn get_assembly(&self) -> String {
        self.program.to_assembly()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_allocation() {
        let mut allocator = RegisterAllocator::new();

        let r1 = allocator.allocate("%0").unwrap();
        assert_eq!(r1, "v0");

        let r2 = allocator.allocate("%1").unwrap();
        assert_eq!(r2, "v1");

        // Check idempotent allocation
        let r1_again = allocator.allocate("%0").unwrap();
        assert_eq!(r1_again, "v0");
    }

    #[test]
    fn test_simple_codegen() {
        let mut codegen = TMatmulCodegen::new();

        codegen.map_memory("X", MemoryLocation::X);
        codegen.map_memory("oH", MemoryLocation::OH);

        codegen.add_section("TEST SECTION");
        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.norm", &["%0"], &["%1"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.sv", &["%1", "oH"], &[])
            .unwrap();

        let asm = codegen.get_assembly();
        assert!(asm.contains("ldv"));
        assert!(asm.contains("norm"));
        assert!(asm.contains("sv"));
    }
}
