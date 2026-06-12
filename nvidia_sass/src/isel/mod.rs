pub mod patterns;

use crate::types::*;

pub fn select_add_i32(dst: u8, src1: u8, src2: u8) -> SassInst {
    SassInst {
        opcode: Opcode {
            mnemonic: "IADD3",
            class: OpcodeClass::Alu3,
        },
        dst: Some(Reg::R(dst)),
        srcs: vec![Operand::Reg(Reg::R(src1)), Operand::Reg(Reg::R(src2))],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    }
}

pub fn select_fma_f32(dst: u8, src1: u8, src2: u8, src3: u8) -> SassInst {
    SassInst {
        opcode: Opcode {
            mnemonic: "FFMA",
            class: OpcodeClass::Fma,
        },
        dst: Some(Reg::R(dst)),
        srcs: vec![
            Operand::Reg(Reg::R(src1)),
            Operand::Reg(Reg::R(src2)),
            Operand::Reg(Reg::R(src3)),
        ],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    }
}

pub fn select_add_f32(dst: u8, src1: u8, src2: u8) -> SassInst {
    SassInst {
        opcode: Opcode {
            mnemonic: "FADD",
            class: OpcodeClass::Alu3,
        },
        dst: Some(Reg::R(dst)),
        srcs: vec![Operand::Reg(Reg::R(src1)), Operand::Reg(Reg::R(src2))],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    }
}

pub fn select_mul_f32(dst: u8, src1: u8, src2: u8) -> SassInst {
    SassInst {
        opcode: Opcode {
            mnemonic: "FMUL",
            class: OpcodeClass::Alu3,
        },
        dst: Some(Reg::R(dst)),
        srcs: vec![Operand::Reg(Reg::R(src1)), Operand::Reg(Reg::R(src2))],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    }
}

pub fn select_load_global(dst: u8, addr: u8, offset: i32) -> SassInst {
    SassInst {
        opcode: Opcode {
            mnemonic: "LDG",
            class: OpcodeClass::Load,
        },
        dst: Some(Reg::R(dst)),
        srcs: vec![Operand::Memory {
            base: Reg::R(addr),
            offset,
        }],
        pred: None,
        modifiers: vec![Modifier::DataType(DataType::U32)],
        control: ControlCodes::default(),
    }
}

pub fn select_store_global(addr: u8, offset: i32, data: u8) -> SassInst {
    SassInst {
        opcode: Opcode {
            mnemonic: "STG",
            class: OpcodeClass::Store,
        },
        dst: None,
        srcs: vec![
            Operand::Memory {
                base: Reg::R(addr),
                offset,
            },
            Operand::Reg(Reg::R(data)),
        ],
        pred: None,
        modifiers: vec![Modifier::DataType(DataType::U32)],
        control: ControlCodes::default(),
    }
}

pub fn select_mov(dst: u8, src: u8) -> SassInst {
    SassInst {
        opcode: Opcode {
            mnemonic: "MOV",
            class: OpcodeClass::Alu2,
        },
        dst: Some(Reg::R(dst)),
        srcs: vec![Operand::Reg(Reg::R(src))],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    }
}

pub fn select_branch(target: u32) -> SassInst {
    SassInst {
        opcode: Opcode {
            mnemonic: "BRA",
            class: OpcodeClass::Branch,
        },
        dst: None,
        srcs: vec![Operand::BranchTarget(target)],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    }
}

pub fn select_exit() -> SassInst {
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
    }
}

pub fn select_bar_sync(barrier_id: u32) -> SassInst {
    SassInst {
        opcode: Opcode {
            mnemonic: "BAR",
            class: OpcodeClass::Sync,
        },
        dst: None,
        srcs: vec![Operand::Imm20(barrier_id as i32)],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    }
}

pub fn select_special_reg(dst: u8, sreg: SpecialReg) -> SassInst {
    SassInst {
        opcode: Opcode {
            mnemonic: "S2R",
            class: OpcodeClass::Special,
        },
        dst: Some(Reg::R(dst)),
        srcs: vec![Operand::SReg(sreg)],
        pred: None,
        modifiers: vec![],
        control: ControlCodes::default(),
    }
}

pub fn select_isetp(pred_dst: u8, src1: u8, src2: u8, cmp: CmpOp) -> SassInst {
    SassInst {
        opcode: Opcode {
            mnemonic: "ISETP",
            class: OpcodeClass::Comparison,
        },
        dst: Some(Reg::P(pred_dst)),
        srcs: vec![Operand::Reg(Reg::R(src1)), Operand::Reg(Reg::R(src2))],
        pred: None,
        modifiers: vec![Modifier::CmpOp(cmp)],
        control: ControlCodes::default(),
    }
}

pub fn select_nop() -> SassInst {
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
    }
}
