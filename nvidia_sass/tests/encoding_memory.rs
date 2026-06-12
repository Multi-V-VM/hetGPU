use nvidia_sass::encoding::sm120;
use nvidia_sass::types::*;

// Helper: encode an instruction and return the lower 64 instruction bits as u64
fn encode_inst_bits(inst: &SassInst) -> u64 {
    let bytes = sm120::encode(inst).expect("encoding should succeed");
    u64::from_le_bytes(bytes[..8].try_into().unwrap())
}

fn default_control() -> ControlCodes {
    ControlCodes::default()
}

// ---------------------------------------------------------------------------
// LDG (Load class)
// ---------------------------------------------------------------------------

#[test]
fn test_ldg_encodes_base_and_offset() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "LDG",
            class: OpcodeClass::Load,
        },
        dst: Some(Reg::R(4)),
        srcs: vec![Operand::Memory {
            base: Reg::R(2),
            offset: 0x10,
        }],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);

    // Opcode
    let opcode = (bits >> 52) & 0xFFF;
    assert_eq!(opcode, 0x380, "LDG opcode should be 0x380");

    // dst in [7:0]
    let dst = bits & 0xFF;
    assert_eq!(dst, 4, "dst should be R4 = 4");

    // base in [15:8]
    let base = (bits >> 8) & 0xFF;
    assert_eq!(
        base, 2,
        "base register R2 should encode as 2 in bits [15:8]"
    );

    // offset in bits starting at [20]
    let offset = (bits >> 20) & 0xFFFFFFFF;
    assert_eq!(offset, 0x10, "offset 0x10 should be in bits [51:20]");
}

// ---------------------------------------------------------------------------
// STG (Store class)
// ---------------------------------------------------------------------------

#[test]
fn test_stg_encodes_data_and_addr() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "STG",
            class: OpcodeClass::Store,
        },
        dst: Some(Reg::R(6)), // data register
        srcs: vec![Operand::Memory {
            base: Reg::R(2),
            offset: 0x20,
        }],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);

    // Opcode
    let opcode = (bits >> 52) & 0xFFF;
    assert_eq!(opcode, 0x385, "STG opcode should be 0x385");

    // data register in [7:0] (dst field)
    let data = bits & 0xFF;
    assert_eq!(data, 6, "data register R6 should be in bits [7:0]");

    // base in [15:8]
    let base = (bits >> 8) & 0xFF;
    assert_eq!(base, 2, "base register R2 should be in bits [15:8]");

    // offset
    let offset = (bits >> 20) & 0xFFFFFFFF;
    assert_eq!(offset, 0x20, "offset 0x20 should be in bits [51:20]");
}

// ---------------------------------------------------------------------------
// S2R (Special class)
// ---------------------------------------------------------------------------

#[test]
fn test_s2r_tid_x() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "S2R",
            class: OpcodeClass::Special,
        },
        dst: Some(Reg::R(0)),
        srcs: vec![Operand::SReg(SpecialReg::TidX)],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);

    // Opcode
    let opcode = (bits >> 52) & 0xFFF;
    assert_eq!(opcode, 0x110, "S2R opcode should be 0x110");

    // dst
    let dst = bits & 0xFF;
    assert_eq!(dst, 0, "dst should be R0 = 0");

    // SR_TID.X = 0x21 in [27:20]
    let sreg = (bits >> 20) & 0xFF;
    assert_eq!(sreg, 0x21, "SR_TID.X should encode as 0x21 in bits [27:20]");
}

// ---------------------------------------------------------------------------
// BRA (Branch class)
// ---------------------------------------------------------------------------

#[test]
fn test_bra_encodes_target() {
    let target: u32 = 0x1234;
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "BRA",
            class: OpcodeClass::Branch,
        },
        dst: None,
        srcs: vec![Operand::BranchTarget(target)],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);

    // Opcode
    let opcode = (bits >> 52) & 0xFFF;
    assert_eq!(opcode, 0x000, "BRA opcode should be 0x000");

    // Target in [51:20]
    let encoded_target = (bits >> 20) & 0xFFFFFFFF;
    assert_eq!(
        encoded_target, 0x1234,
        "branch target 0x1234 should be in bits [51:20]"
    );
}

// ---------------------------------------------------------------------------
// EXIT (Branch class)
// ---------------------------------------------------------------------------

#[test]
fn test_exit_encodes() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "EXIT",
            class: OpcodeClass::Branch,
        },
        dst: None,
        srcs: vec![],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);

    // Opcode
    let opcode = (bits >> 52) & 0xFFF;
    assert_eq!(opcode, 0x010, "EXIT opcode should be 0x010");

    // Predicate should be PT (7), not negated
    let pred_reg = (bits >> 16) & 0x7;
    let pred_neg = (bits >> 19) & 1;
    assert_eq!(pred_reg, 7, "default predicate should be PT=7");
    assert_eq!(pred_neg, 0, "default predicate should not be negated");
}
