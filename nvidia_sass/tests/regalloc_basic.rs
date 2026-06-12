use nvidia_sass::regalloc;
use nvidia_sass::types::*;

fn make_virtual_sequence() -> Vec<SassInst> {
    vec![
        SassInst {
            opcode: Opcode {
                mnemonic: "S2R",
                class: OpcodeClass::Special,
            },
            dst: Some(Reg::R(200)),
            srcs: vec![Operand::SReg(SpecialReg::TidX)],
            pred: None,
            modifiers: vec![],
            control: ControlCodes::default(),
        },
        SassInst {
            opcode: Opcode {
                mnemonic: "LDG",
                class: OpcodeClass::Load,
            },
            dst: Some(Reg::R(201)),
            srcs: vec![Operand::Memory {
                base: Reg::R(200),
                offset: 0,
            }],
            pred: None,
            modifiers: vec![],
            control: ControlCodes::default(),
        },
        SassInst {
            opcode: Opcode {
                mnemonic: "IADD3",
                class: OpcodeClass::Alu3,
            },
            dst: Some(Reg::R(202)),
            srcs: vec![Operand::Reg(Reg::R(201)), Operand::Reg(Reg::R(201))],
            pred: None,
            modifiers: vec![],
            control: ControlCodes::default(),
        },
        SassInst {
            opcode: Opcode {
                mnemonic: "STG",
                class: OpcodeClass::Store,
            },
            dst: None,
            srcs: vec![
                Operand::Memory {
                    base: Reg::R(200),
                    offset: 0,
                },
                Operand::Reg(Reg::R(202)),
            ],
            pred: None,
            modifiers: vec![],
            control: ControlCodes::default(),
        },
        SassInst {
            opcode: Opcode {
                mnemonic: "EXIT",
                class: OpcodeClass::Branch,
            },
            dst: None,
            srcs: vec![],
            pred: None,
            modifiers: vec![],
            control: ControlCodes::default(),
        },
    ]
}

#[test]
fn test_regalloc_maps_virtual_to_physical() {
    let virtual_insts = make_virtual_sequence();
    let (physical_insts, _) = regalloc::allocate(&virtual_insts).unwrap();
    for inst in &physical_insts {
        if let Some(Reg::R(n)) = inst.dst {
            assert!(n < 200, "dst register {} should be physical", n);
        }
        for src in &inst.srcs {
            match src {
                Operand::Reg(Reg::R(n)) => {
                    assert!(*n < 200, "src register {} should be physical", n)
                }
                Operand::Memory {
                    base: Reg::R(n), ..
                } => assert!(*n < 200, "base register {} should be physical", n),
                _ => {}
            }
        }
    }
}

#[test]
fn test_regalloc_preserves_instruction_count() {
    let virtual_insts = make_virtual_sequence();
    let (physical_insts, _) = regalloc::allocate(&virtual_insts).unwrap();
    assert_eq!(physical_insts.len(), virtual_insts.len());
}

#[test]
fn test_regalloc_reports_register_count() {
    let virtual_insts = make_virtual_sequence();
    let (_, num_regs) = regalloc::allocate(&virtual_insts).unwrap();
    assert!(num_regs >= 3, "need at least 3 registers, got {}", num_regs);
    assert!(num_regs <= 255);
}
