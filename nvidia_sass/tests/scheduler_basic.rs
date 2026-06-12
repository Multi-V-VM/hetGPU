use nvidia_sass::scheduler;
use nvidia_sass::types::*;

#[test]
fn test_scheduler_sets_stall_for_load_dependency() {
    let insts = vec![
        SassInst {
            opcode: Opcode {
                mnemonic: "LDG",
                class: OpcodeClass::Load,
            },
            dst: Some(Reg::R(0)),
            srcs: vec![Operand::Memory {
                base: Reg::R(1),
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
            dst: Some(Reg::R(2)),
            srcs: vec![Operand::Reg(Reg::R(0)), Operand::Reg(Reg::R(3))],
            pred: None,
            modifiers: vec![],
            control: ControlCodes::default(),
        },
    ];
    let scheduled = scheduler::schedule(&insts);
    let iadd_ctrl = &scheduled[1].control;
    assert!(
        iadd_ctrl.stall > 0 || iadd_ctrl.wait_mask != 0,
        "dependent instruction needs stall or barrier wait"
    );
}

#[test]
fn test_scheduler_independent_instructions() {
    let insts = vec![
        SassInst {
            opcode: Opcode {
                mnemonic: "IADD3",
                class: OpcodeClass::Alu3,
            },
            dst: Some(Reg::R(0)),
            srcs: vec![Operand::Reg(Reg::R(1)), Operand::Reg(Reg::R(2))],
            pred: None,
            modifiers: vec![],
            control: ControlCodes::default(),
        },
        SassInst {
            opcode: Opcode {
                mnemonic: "IADD3",
                class: OpcodeClass::Alu3,
            },
            dst: Some(Reg::R(3)),
            srcs: vec![Operand::Reg(Reg::R(4)), Operand::Reg(Reg::R(5))],
            pred: None,
            modifiers: vec![],
            control: ControlCodes::default(),
        },
    ];
    let scheduled = scheduler::schedule(&insts);
    assert!(
        scheduled[1].control.stall <= 2,
        "independent instructions should have low stall, got {}",
        scheduled[1].control.stall
    );
}

#[test]
fn test_scheduler_load_gets_write_barrier() {
    let insts = vec![SassInst {
        opcode: Opcode {
            mnemonic: "LDG",
            class: OpcodeClass::Load,
        },
        dst: Some(Reg::R(0)),
        srcs: vec![Operand::Memory {
            base: Reg::R(1),
            offset: 0,
        }],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    }];
    let scheduled = scheduler::schedule(&insts);
    assert_ne!(
        scheduled[0].control.write_barrier, 7,
        "LDG should get a write barrier assigned"
    );
}

#[test]
fn test_scheduler_preserves_instruction_count() {
    let insts = vec![
        SassInst {
            opcode: Opcode {
                mnemonic: "NOP",
                class: OpcodeClass::Nop,
            },
            dst: None,
            srcs: vec![],
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
    ];
    let scheduled = scheduler::schedule(&insts);
    assert_eq!(scheduled.len(), 2);
}
