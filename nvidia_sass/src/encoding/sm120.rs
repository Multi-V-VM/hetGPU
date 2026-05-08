use std::collections::HashMap;
use std::sync::LazyLock;

use crate::types::*;

// ---------------------------------------------------------------------------
// Opcode table: mnemonic -> 12-bit primary opcode value
// ---------------------------------------------------------------------------
static OPCODE_TABLE: LazyLock<HashMap<&'static str, u16>> = LazyLock::new(|| {
    let entries: &[(&str, u16)] = &[
        // Memory
        ("LDG", 0x380),
        ("LDG.E", 0x381),
        ("STG", 0x385),
        ("STG.E", 0x386),
        ("LDS", 0x388),
        ("STS", 0x389),
        ("LDL", 0x38C),
        ("STL", 0x38D),
        ("LDC", 0x390),
        // Atomics
        ("ATOMG", 0x3A8),
        ("ATOMS", 0x3A9),
        ("RED", 0x3AC),
        // Integer
        ("IADD", 0x210),
        ("IADD3", 0x211),
        ("IMAD", 0x214),
        ("IMAD.WIDE", 0x215),
        ("IMUL", 0x218),
        ("ISETP", 0x21C),
        ("ISET", 0x21D),
        ("IABS", 0x220),
        ("INEG", 0x221),
        // Float
        ("FADD", 0x300),
        ("FMUL", 0x304),
        ("FFMA", 0x308),
        ("FSETP", 0x30C),
        ("FSET", 0x30D),
        ("FABS", 0x310),
        ("FNEG", 0x311),
        ("MUFU", 0x318),
        // Double
        ("DADD", 0x320),
        ("DMUL", 0x324),
        ("DFMA", 0x328),
        // Logic
        ("LOP", 0x200),
        ("LOP3", 0x201),
        ("SHL", 0x204),
        ("SHR", 0x205),
        ("BFE", 0x208),
        ("BFI", 0x209),
        ("FLO", 0x20C),
        ("POPC", 0x20D),
        // Conversion
        ("I2I", 0x340),
        ("I2F", 0x344),
        ("F2I", 0x348),
        ("F2F", 0x34C),
        // Data movement
        ("MOV", 0x100),
        ("MOV32I", 0x101),
        ("PRMT", 0x104),
        ("SEL", 0x108),
        ("SHFL", 0x10C),
        // Special registers
        ("S2R", 0x110),
        ("CS2R", 0x114),
        // Control flow
        ("BRA", 0x000),
        ("BRX", 0x001),
        ("JMP", 0x004),
        ("JMX", 0x005),
        ("CAL", 0x008),
        ("JCAL", 0x009),
        ("RET", 0x00C),
        ("EXIT", 0x010),
        // Sync
        ("BAR", 0x020),
        ("DEPBAR", 0x024),
        ("MEMBAR", 0x028),
        // Predicates
        ("PSETP", 0x180),
        ("PLOP3", 0x184),
        ("P2R", 0x188),
        ("R2P", 0x18C),
        // Half
        ("HADD2", 0x360),
        ("HMUL2", 0x364),
        ("HFMA2", 0x368),
        // Tensor
        ("HMMA", 0x3C0),
        ("IMMA", 0x3C4),
        // Texture
        ("TEX", 0x400),
        ("TLD", 0x404),
        ("TLD4", 0x408),
        ("TXQ", 0x40C),
        // NOP
        ("NOP", 0x7FF),
    ];
    entries.iter().cloned().collect()
});

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Encode a full 128-bit SASS instruction (lower 64 = instruction, upper 64 = control).
pub fn encode(inst: &SassInst) -> Result<[u8; 16], NvSassError> {
    let inst_lo = encode_instruction_bits(inst)?;
    let ctrl_hi = super::control_codes::encode(&inst.control);
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&inst_lo.to_le_bytes());
    bytes[8..].copy_from_slice(&ctrl_hi.to_le_bytes());
    Ok(bytes)
}

/// Look up the 12-bit opcode value for a mnemonic string.
/// Tries exact match first, then strips everything after the first '.' and retries.
pub fn lookup_opcode_bits(mnemonic: &str) -> Result<u16, NvSassError> {
    lookup_opcode(mnemonic)
}

// ---------------------------------------------------------------------------
// Opcode lookup
// ---------------------------------------------------------------------------

fn lookup_opcode(mnemonic: &str) -> Result<u16, NvSassError> {
    // Exact match
    if let Some(&op) = OPCODE_TABLE.get(mnemonic) {
        return Ok(op);
    }
    // Strip modifiers after first dot and retry base
    if let Some(dot_pos) = mnemonic.find('.') {
        let base = &mnemonic[..dot_pos];
        if let Some(&op) = OPCODE_TABLE.get(base) {
            return Ok(op);
        }
    }
    Err(NvSassError::EncodingError {
        opcode: mnemonic.to_string(),
        msg: "unknown opcode mnemonic".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Predicate encoding  [19:16]
// ---------------------------------------------------------------------------

fn encode_predicate(pred: &Option<Predicate>) -> u64 {
    match pred {
        Some(p) => {
            let reg_bits = p.reg.encode_pred() as u64;
            let neg_bit = if p.negated { 1u64 } else { 0u64 };
            (neg_bit << 19) | (reg_bits << 16)
        }
        None => {
            // Default: PT (true predicate, register 7), not negated
            7u64 << 16
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level instruction bit encoder  (lower 64 bits)
// ---------------------------------------------------------------------------

fn encode_instruction_bits(inst: &SassInst) -> Result<u64, NvSassError> {
    match inst.opcode.class {
        OpcodeClass::Alu3 => encode_alu3(inst),
        OpcodeClass::Alu2 => encode_alu2(inst),
        OpcodeClass::Fma => encode_fma(inst),
        OpcodeClass::Load => encode_load(inst),
        OpcodeClass::Store => encode_store(inst),
        OpcodeClass::Branch => encode_branch(inst),
        OpcodeClass::Comparison => encode_comparison(inst),
        OpcodeClass::Sync => encode_sync(inst),
        OpcodeClass::Special => encode_special(inst),
        OpcodeClass::Nop => encode_nop(inst),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Place 12-bit opcode into bits [63:52].
fn opcode_field(mnemonic: &str) -> Result<u64, NvSassError> {
    let op = lookup_opcode(mnemonic)?;
    Ok((op as u64) << 52)
}

/// Encode a destination register into bits [7:0].
fn dst_field(inst: &SassInst) -> u64 {
    match &inst.dst {
        Some(r) => r.encode_gpr() as u64,
        None => 255u64, // RZ
    }
}

/// Encode source register from Operand into the given bit position.
fn src_reg_field(op: &Operand, shift: u32) -> u64 {
    match op {
        Operand::Reg(r) => (r.encode_gpr() as u64) << shift,
        _ => 255u64 << shift, // RZ for non-register operands
    }
}

/// Encode a 20-bit immediate into bits [27:8] (standard immediate field).
fn imm20_field(val: i32) -> u64 {
    let raw = (val as u32) & 0xFFFFF;
    (raw as u64) << 20
}

/// Encode an Imm32 into bits [51:20].
fn imm32_field(val: i32) -> u64 {
    let raw = val as u32;
    (raw as u64) << 20
}

// ---------------------------------------------------------------------------
// Per-class encoders
// ---------------------------------------------------------------------------

/// ALU with 3 operands: dst, src1, src2
/// Layout: [63:52]=opcode, [19:16]=pred, [7:0]=dst, [15:8]=src1, [27:20]=src2
fn encode_alu3(inst: &SassInst) -> Result<u64, NvSassError> {
    let mut bits: u64 = 0;
    bits |= opcode_field(inst.opcode.mnemonic)?;
    bits |= encode_predicate(&inst.pred);
    bits |= dst_field(inst);

    if inst.srcs.len() >= 1 {
        bits |= src_reg_field(&inst.srcs[0], 8);
    }
    if inst.srcs.len() >= 2 {
        match &inst.srcs[1] {
            Operand::Imm20(v) => bits |= imm20_field(*v),
            other => bits |= src_reg_field(other, 20),
        }
    }
    // Third source into [35:28] if present (e.g. IADD3)
    if inst.srcs.len() >= 3 {
        bits |= src_reg_field(&inst.srcs[2], 28);
    }

    Ok(bits)
}

/// ALU with 2 operands: dst, src1 (e.g. MOV, IABS, INEG, POPC, FLO, FABS, FNEG)
/// Layout: [63:52]=opcode, [19:16]=pred, [7:0]=dst, [15:8]=src1
fn encode_alu2(inst: &SassInst) -> Result<u64, NvSassError> {
    let mut bits: u64 = 0;
    bits |= opcode_field(inst.opcode.mnemonic)?;
    bits |= encode_predicate(&inst.pred);
    bits |= dst_field(inst);

    if inst.srcs.len() >= 1 {
        match &inst.srcs[0] {
            Operand::Imm32(v) => bits |= imm32_field(*v),
            Operand::Imm20(v) => bits |= imm20_field(*v),
            other => bits |= src_reg_field(other, 8),
        }
    }

    Ok(bits)
}

/// FMA: dst, src1, src2, src3
/// Layout: [63:52]=opcode, [19:16]=pred, [7:0]=dst, [15:8]=src1, [27:20]=src2, [35:28]=src3
fn encode_fma(inst: &SassInst) -> Result<u64, NvSassError> {
    let mut bits: u64 = 0;
    bits |= opcode_field(inst.opcode.mnemonic)?;
    bits |= encode_predicate(&inst.pred);
    bits |= dst_field(inst);

    if inst.srcs.len() >= 1 {
        bits |= src_reg_field(&inst.srcs[0], 8);
    }
    if inst.srcs.len() >= 2 {
        match &inst.srcs[1] {
            Operand::Imm20(v) => bits |= imm20_field(*v),
            other => bits |= src_reg_field(other, 20),
        }
    }
    if inst.srcs.len() >= 3 {
        bits |= src_reg_field(&inst.srcs[2], 28);
    }

    Ok(bits)
}

/// Load: dst, [base + offset]
/// Layout: [63:52]=opcode, [19:16]=pred, [7:0]=dst, [15:8]=base, [51:20]=offset (sign-extended 32)
fn encode_load(inst: &SassInst) -> Result<u64, NvSassError> {
    let mut bits: u64 = 0;
    bits |= opcode_field(inst.opcode.mnemonic)?;
    bits |= encode_predicate(&inst.pred);
    bits |= dst_field(inst);

    if let Some(src) = inst.srcs.first() {
        match src {
            Operand::Memory { base, offset } => {
                bits |= (base.encode_gpr() as u64) << 8;
                let off_bits = (*offset as u32) as u64;
                bits |= (off_bits & 0xFFFFFFFF) << 20;
            }
            Operand::Reg(r) => {
                bits |= (r.encode_gpr() as u64) << 8;
            }
            Operand::SReg(sr) => {
                // S2R: special register code in [27:20]
                bits |= (sr.encode() as u64) << 20;
            }
            _ => {}
        }
    }

    Ok(bits)
}

/// Store: [base + offset], data
/// Layout: [63:52]=opcode, [19:16]=pred, [7:0]=data, [15:8]=base, [51:20]=offset
fn encode_store(inst: &SassInst) -> Result<u64, NvSassError> {
    let mut bits: u64 = 0;
    bits |= opcode_field(inst.opcode.mnemonic)?;
    bits |= encode_predicate(&inst.pred);

    // For stores, first operand is the memory address, second is the data register
    if let Some(addr) = inst.srcs.first() {
        match addr {
            Operand::Memory { base, offset } => {
                bits |= (base.encode_gpr() as u64) << 8;
                let off_bits = (*offset as u32) as u64;
                bits |= (off_bits & 0xFFFFFFFF) << 20;
            }
            _ => {}
        }
    }

    // Data register in dst field [7:0]
    bits |= dst_field(inst);

    Ok(bits)
}

/// Branch / control flow: BRA, BRX, JMP, JMX, CAL, JCAL, RET, EXIT
/// Layout: [63:52]=opcode, [19:16]=pred, [51:20]=target
fn encode_branch(inst: &SassInst) -> Result<u64, NvSassError> {
    let mut bits: u64 = 0;
    bits |= opcode_field(inst.opcode.mnemonic)?;
    bits |= encode_predicate(&inst.pred);

    // Branch target in first source
    if let Some(src) = inst.srcs.first() {
        match src {
            Operand::BranchTarget(target) => {
                bits |= ((*target as u64) & 0xFFFFFFFF) << 20;
            }
            Operand::Reg(r) => {
                bits |= (r.encode_gpr() as u64) << 8;
            }
            _ => {}
        }
    }

    Ok(bits)
}

/// Comparison: ISETP, ISET, FSETP, FSET, PSETP, PLOP3
/// Layout: [63:52]=opcode, [19:16]=pred, [7:0]=dst (predicate destination encoded as GPR slot),
///         [15:8]=src1, [27:20]=src2
fn encode_comparison(inst: &SassInst) -> Result<u64, NvSassError> {
    let mut bits: u64 = 0;
    bits |= opcode_field(inst.opcode.mnemonic)?;
    bits |= encode_predicate(&inst.pred);
    bits |= dst_field(inst);

    if inst.srcs.len() >= 1 {
        bits |= src_reg_field(&inst.srcs[0], 8);
    }
    if inst.srcs.len() >= 2 {
        match &inst.srcs[1] {
            Operand::Imm20(v) => bits |= imm20_field(*v),
            other => bits |= src_reg_field(other, 20),
        }
    }

    Ok(bits)
}

/// Sync: BAR, DEPBAR, MEMBAR
/// Layout: [63:52]=opcode, [19:16]=pred, [7:0]=barrier id or similar
fn encode_sync(inst: &SassInst) -> Result<u64, NvSassError> {
    let mut bits: u64 = 0;
    bits |= opcode_field(inst.opcode.mnemonic)?;
    bits |= encode_predicate(&inst.pred);

    // Barrier ID in dst field if present
    bits |= dst_field(inst);

    if inst.srcs.len() >= 1 {
        bits |= src_reg_field(&inst.srcs[0], 8);
    }

    Ok(bits)
}

/// Special: S2R, CS2R, MUFU, conversion ops, PRMT, SEL, SHFL, P2R, R2P
/// Layout: [63:52]=opcode, [19:16]=pred, [7:0]=dst, [15:8]=src1,
///         [27:20]=special register code or MUFU subop
fn encode_special(inst: &SassInst) -> Result<u64, NvSassError> {
    let mut bits: u64 = 0;
    bits |= opcode_field(inst.opcode.mnemonic)?;
    bits |= encode_predicate(&inst.pred);
    bits |= dst_field(inst);

    if let Some(src) = inst.srcs.first() {
        match src {
            Operand::SReg(sr) => {
                // S2R / CS2R: special register index in [27:20]
                bits |= (sr.encode() as u64) << 20;
            }
            Operand::Reg(r) => {
                bits |= (r.encode_gpr() as u64) << 8;
            }
            Operand::Imm20(v) => {
                bits |= imm20_field(*v);
            }
            _ => {}
        }
    }

    // Encode MUFU subop in [27:20] if present
    for m in &inst.modifiers {
        if let Modifier::MufuOp(op) = m {
            let code: u8 = match op {
                MufuOp::Rcp => 0,
                MufuOp::Rsq => 1,
                MufuOp::Sin => 2,
                MufuOp::Cos => 3,
                MufuOp::Ex2 => 4,
                MufuOp::Lg2 => 5,
                MufuOp::Rcp64h => 6,
                MufuOp::Rsq64h => 7,
            };
            bits |= (code as u64) << 20;
        }
    }

    // Additional sources
    if inst.srcs.len() >= 2 {
        bits |= src_reg_field(&inst.srcs[1], 20);
    }

    Ok(bits)
}

/// NOP: opcode 0x7FF, predicate PT
fn encode_nop(inst: &SassInst) -> Result<u64, NvSassError> {
    let mut bits: u64 = 0;
    bits |= opcode_field(inst.opcode.mnemonic)?;
    bits |= encode_predicate(&inst.pred);
    Ok(bits)
}

// ---------------------------------------------------------------------------
// Internal tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_lookup_exact() {
        assert_eq!(lookup_opcode("IADD3").unwrap(), 0x211);
        assert_eq!(lookup_opcode("NOP").unwrap(), 0x7FF);
        assert_eq!(lookup_opcode("LDG").unwrap(), 0x380);
        assert_eq!(lookup_opcode("EXIT").unwrap(), 0x010);
    }

    #[test]
    fn test_opcode_lookup_with_modifier_strip() {
        // "LDG.E" is in the table explicitly
        assert_eq!(lookup_opcode("LDG.E").unwrap(), 0x381);
        // "IADD.SAT" should strip to "IADD" and find 0x210
        assert_eq!(lookup_opcode("IADD.SAT").unwrap(), 0x210);
    }

    #[test]
    fn test_opcode_lookup_unknown() {
        assert!(lookup_opcode("ZZZZZ").is_err());
    }

    #[test]
    fn test_nop_encoding() {
        let inst = SassInst {
            opcode: Opcode {
                mnemonic: "NOP",
                class: OpcodeClass::Nop,
            },
            dst: None,
            srcs: vec![],
            pred: None,
            modifiers: vec![],
            control: ControlCodes::default(),
        };
        let bits = encode_instruction_bits(&inst).unwrap();
        // Opcode in [63:52] = 0x7FF
        let opcode_val = (bits >> 52) & 0xFFF;
        assert_eq!(opcode_val, 0x7FF);
        // Predicate: PT=7 in [18:16], not negated [19]=0
        let pred_val = (bits >> 16) & 0xF;
        assert_eq!(pred_val, 7);
    }

    #[test]
    fn test_predicate_encoding_pt() {
        let bits = encode_predicate(&None);
        assert_eq!((bits >> 16) & 0x7, 7); // PT
        assert_eq!((bits >> 19) & 1, 0); // not negated
    }

    #[test]
    fn test_predicate_encoding_p2() {
        let p = Some(Predicate {
            reg: Reg::P(2),
            negated: false,
        });
        let bits = encode_predicate(&p);
        assert_eq!((bits >> 16) & 0x7, 2);
        assert_eq!((bits >> 19) & 1, 0);
    }

    #[test]
    fn test_predicate_encoding_negated() {
        let p = Some(Predicate {
            reg: Reg::P(3),
            negated: true,
        });
        let bits = encode_predicate(&p);
        assert_eq!((bits >> 16) & 0x7, 3);
        assert_eq!((bits >> 19) & 1, 1);
    }
}
