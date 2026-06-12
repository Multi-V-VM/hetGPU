use nvidia_sass::isel;
use nvidia_sass::types::*;

#[test]
fn test_isel_add_i32() {
    let result = isel::select_add_i32(0, 1, 2);
    assert_eq!(result.opcode.mnemonic, "IADD3");
    assert_eq!(result.dst, Some(Reg::R(0)));
    assert_eq!(result.srcs.len(), 2);
}

#[test]
fn test_isel_fma_f32() {
    let result = isel::select_fma_f32(10, 11, 12, 13);
    assert_eq!(result.opcode.mnemonic, "FFMA");
    assert_eq!(result.srcs.len(), 3);
}

#[test]
fn test_isel_load_global() {
    let result = isel::select_load_global(5, 2, 0x40);
    assert_eq!(result.opcode.mnemonic, "LDG");
    assert_eq!(result.opcode.class, OpcodeClass::Load);
}

#[test]
fn test_isel_store_global() {
    let result = isel::select_store_global(4, 0, 8);
    assert_eq!(result.opcode.mnemonic, "STG");
    assert_eq!(result.opcode.class, OpcodeClass::Store);
}

#[test]
fn test_isel_branch() {
    let result = isel::select_branch(0x200);
    assert_eq!(result.opcode.mnemonic, "BRA");
}

#[test]
fn test_isel_exit() {
    let result = isel::select_exit();
    assert_eq!(result.opcode.mnemonic, "EXIT");
}

#[test]
fn test_isel_tid_x() {
    let result = isel::select_special_reg(3, SpecialReg::TidX);
    assert_eq!(result.opcode.mnemonic, "S2R");
    match &result.srcs[0] {
        Operand::SReg(SpecialReg::TidX) => {}
        other => panic!("expected SReg TidX, got {:?}", other),
    }
}

#[test]
fn test_isel_isetp() {
    let result = isel::select_isetp(0, 1, 2, CmpOp::Lt);
    assert_eq!(result.opcode.mnemonic, "ISETP");
    assert_eq!(result.dst, Some(Reg::P(0)));
}
