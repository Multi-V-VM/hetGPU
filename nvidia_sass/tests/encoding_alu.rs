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
// IADD3 tests
// ---------------------------------------------------------------------------

#[test]
fn test_iadd3_encodes_opcode_bits() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "IADD3",
            class: OpcodeClass::Alu3,
        },
        dst: Some(Reg::R(0)),
        srcs: vec![
            Operand::Reg(Reg::R(1)),
            Operand::Reg(Reg::R(2)),
            Operand::Reg(Reg::R(3)),
        ],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);
    let opcode = (bits >> 52) & 0xFFF;
    assert_eq!(opcode, 0x211, "IADD3 opcode should be 0x211");
}

#[test]
fn test_iadd3_encodes_dst_register() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "IADD3",
            class: OpcodeClass::Alu3,
        },
        dst: Some(Reg::R(10)),
        srcs: vec![
            Operand::Reg(Reg::R(1)),
            Operand::Reg(Reg::R(2)),
            Operand::Reg(Reg::R(3)),
        ],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);
    let dst = bits & 0xFF;
    assert_eq!(
        dst, 10,
        "dst register R10 should encode as 10 in bits [7:0]"
    );
}

#[test]
fn test_iadd3_encodes_src1_register() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "IADD3",
            class: OpcodeClass::Alu3,
        },
        dst: Some(Reg::R(0)),
        srcs: vec![
            Operand::Reg(Reg::R(5)),
            Operand::Reg(Reg::R(2)),
            Operand::Reg(Reg::R(3)),
        ],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);
    let src1 = (bits >> 8) & 0xFF;
    assert_eq!(
        src1, 5,
        "src1 register R5 should encode as 5 in bits [15:8]"
    );
}

#[test]
fn test_iadd3_encodes_src2_register() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "IADD3",
            class: OpcodeClass::Alu3,
        },
        dst: Some(Reg::R(0)),
        srcs: vec![
            Operand::Reg(Reg::R(1)),
            Operand::Reg(Reg::R(7)),
            Operand::Reg(Reg::R(3)),
        ],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);
    let src2 = (bits >> 20) & 0xFF;
    assert_eq!(
        src2, 7,
        "src2 register R7 should encode as 7 in bits [27:20]"
    );
}

// ---------------------------------------------------------------------------
// FFMA (FMA class - three sources)
// ---------------------------------------------------------------------------

#[test]
fn test_ffma_encodes_three_sources() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "FFMA",
            class: OpcodeClass::Fma,
        },
        dst: Some(Reg::R(4)),
        srcs: vec![
            Operand::Reg(Reg::R(1)),
            Operand::Reg(Reg::R(2)),
            Operand::Reg(Reg::R(3)),
        ],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);

    // Opcode
    let opcode = (bits >> 52) & 0xFFF;
    assert_eq!(opcode, 0x308, "FFMA opcode should be 0x308");

    // dst
    let dst = bits & 0xFF;
    assert_eq!(dst, 4);

    // src1 in [15:8]
    let src1 = (bits >> 8) & 0xFF;
    assert_eq!(src1, 1);

    // src2 in [27:20]
    let src2 = (bits >> 20) & 0xFF;
    assert_eq!(src2, 2);

    // src3 in [35:28]
    let src3 = (bits >> 28) & 0xFF;
    assert_eq!(src3, 3, "FFMA src3 should be in bits [35:28]");
}

// ---------------------------------------------------------------------------
// Predicate tests
// ---------------------------------------------------------------------------

#[test]
fn test_predicated_instruction() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "IADD3",
            class: OpcodeClass::Alu3,
        },
        dst: Some(Reg::R(0)),
        srcs: vec![
            Operand::Reg(Reg::R(1)),
            Operand::Reg(Reg::R(2)),
            Operand::Reg(Reg::R(3)),
        ],
        pred: Some(Predicate {
            reg: Reg::P(2),
            negated: false,
        }),
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);
    let pred_reg = (bits >> 16) & 0x7;
    let pred_neg = (bits >> 19) & 1;
    assert_eq!(
        pred_reg, 2,
        "predicate register P2 should encode as 2 in bits [18:16]"
    );
    assert_eq!(pred_neg, 0, "predicate should not be negated");
}

#[test]
fn test_negated_predicate() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "IADD3",
            class: OpcodeClass::Alu3,
        },
        dst: Some(Reg::R(0)),
        srcs: vec![
            Operand::Reg(Reg::R(1)),
            Operand::Reg(Reg::R(2)),
            Operand::Reg(Reg::R(3)),
        ],
        pred: Some(Predicate {
            reg: Reg::P(5),
            negated: true,
        }),
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);
    let pred_reg = (bits >> 16) & 0x7;
    let pred_neg = (bits >> 19) & 1;
    assert_eq!(pred_reg, 5, "predicate register P5 should encode as 5");
    assert_eq!(pred_neg, 1, "negated predicate should set bit [19]");
}

// ---------------------------------------------------------------------------
// MOV (Alu2 class)
// ---------------------------------------------------------------------------

#[test]
fn test_mov_encodes() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "MOV",
            class: OpcodeClass::Alu2,
        },
        dst: Some(Reg::R(8)),
        srcs: vec![Operand::Reg(Reg::R(9))],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);
    let opcode = (bits >> 52) & 0xFFF;
    assert_eq!(opcode, 0x100, "MOV opcode should be 0x100");
    let dst = bits & 0xFF;
    assert_eq!(dst, 8);
    let src1 = (bits >> 8) & 0xFF;
    assert_eq!(src1, 9);
}

// ---------------------------------------------------------------------------
// RZ register
// ---------------------------------------------------------------------------

#[test]
fn test_rz_register_encodes_as_255() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "MOV",
            class: OpcodeClass::Alu2,
        },
        dst: Some(Reg::RZ),
        srcs: vec![Operand::Reg(Reg::RZ)],
        pred: None,
        modifiers: vec![],
        control: default_control(),
    };
    let bits = encode_inst_bits(&inst);
    let dst = bits & 0xFF;
    assert_eq!(dst, 255, "RZ should encode as 255 in dst field");
    let src1 = (bits >> 8) & 0xFF;
    assert_eq!(src1, 255, "RZ should encode as 255 in src1 field");
}
