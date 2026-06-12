use nvidia_sass::roundtrip;
use nvidia_sass::types::*;

#[test]
fn test_roundtrip_iadd3() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "IADD3",
            class: OpcodeClass::Alu3,
        },
        dst: Some(Reg::R(4)),
        srcs: vec![Operand::Reg(Reg::R(5)), Operand::Reg(Reg::R(6))],
        pred: None,
        modifiers: vec![],
        control: ControlCodes {
            stall: 4,
            yield_flag: false,
            write_barrier: 7,
            read_barrier: 7,
            wait_mask: 0,
            reuse: 0,
        },
    };
    roundtrip::validate_roundtrip(&inst, 120).unwrap();
}

#[test]
fn test_roundtrip_mov() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "MOV",
            class: OpcodeClass::Alu2,
        },
        dst: Some(Reg::R(1)),
        srcs: vec![Operand::Reg(Reg::R(2))],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    };
    roundtrip::validate_roundtrip(&inst, 120).unwrap();
}

#[test]
fn test_roundtrip_nop() {
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
    roundtrip::validate_roundtrip(&inst, 120).unwrap();
}

#[test]
fn test_roundtrip_control_codes() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "IADD3",
            class: OpcodeClass::Alu3,
        },
        dst: Some(Reg::R(0)),
        srcs: vec![Operand::Reg(Reg::R(1)), Operand::Reg(Reg::R(2))],
        pred: Some(Predicate {
            reg: Reg::P(3),
            negated: true,
        }),
        modifiers: vec![],
        control: ControlCodes {
            stall: 7,
            yield_flag: true,
            write_barrier: 2,
            read_barrier: 3,
            wait_mask: 0b101,
            reuse: 0b1010,
        },
    };
    roundtrip::validate_roundtrip(&inst, 120).unwrap();
}

#[test]
fn test_roundtrip_ldg() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "LDG",
            class: OpcodeClass::Load,
        },
        dst: Some(Reg::R(10)),
        srcs: vec![Operand::Memory {
            base: Reg::R(2),
            offset: 0x40,
        }],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    };
    roundtrip::validate_roundtrip(&inst, 120).unwrap();
}

#[test]
fn test_roundtrip_exit() {
    let inst = SassInst {
        opcode: Opcode {
            mnemonic: "EXIT",
            class: OpcodeClass::Branch,
        },
        dst: None,
        srcs: vec![],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    };
    roundtrip::validate_roundtrip(&inst, 120).unwrap();
}
