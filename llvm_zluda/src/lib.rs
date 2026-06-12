#![allow(non_upper_case_globals)]
use llvm_sys::prelude::*;
use llvm_sys::target::LLVMTargetDataRef;
pub use llvm_sys::*;

// 添加LLVMDbgRecordRef类型
pub type LLVMDbgRecordRef = *mut llvm_sys::LLVMOpaqueDbgRecord;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMZludaAtomicRMWBinOp {
    LLVMZludaAtomicRMWBinOpXchg = 0,
    LLVMZludaAtomicRMWBinOpAdd = 1,
    LLVMZludaAtomicRMWBinOpSub = 2,
    LLVMZludaAtomicRMWBinOpAnd = 3,
    LLVMZludaAtomicRMWBinOpNand = 4,
    LLVMZludaAtomicRMWBinOpOr = 5,
    LLVMZludaAtomicRMWBinOpXor = 6,
    LLVMZludaAtomicRMWBinOpMax = 7,
    LLVMZludaAtomicRMWBinOpMin = 8,
    LLVMZludaAtomicRMWBinOpUMax = 9,
    LLVMZludaAtomicRMWBinOpUMin = 10,
    LLVMZludaAtomicRMWBinOpFAdd = 11,
    LLVMZludaAtomicRMWBinOpFSub = 12,
    LLVMZludaAtomicRMWBinOpFMax = 13,
    LLVMZludaAtomicRMWBinOpFMin = 14,
    LLVMZludaAtomicRMWBinOpUIncWrap = 15,
    LLVMZludaAtomicRMWBinOpUDecWrap = 16,
}

// Backport from LLVM 19
pub const LLVMZludaFastMathAllowReassoc: ::std::ffi::c_uint = 1 << 0;
pub const LLVMZludaFastMathNoNaNs: ::std::ffi::c_uint = 1 << 1;
pub const LLVMZludaFastMathNoInfs: ::std::ffi::c_uint = 1 << 2;
pub const LLVMZludaFastMathNoSignedZeros: ::std::ffi::c_uint = 1 << 3;
pub const LLVMZludaFastMathAllowReciprocal: ::std::ffi::c_uint = 1 << 4;
pub const LLVMZludaFastMathAllowContract: ::std::ffi::c_uint = 1 << 5;
pub const LLVMZludaFastMathApproxFunc: ::std::ffi::c_uint = 1 << 6;
pub const LLVMZludaFastMathNone: ::std::ffi::c_uint = 0;
pub const LLVMZludaFastMathAll: ::std::ffi::c_uint = LLVMZludaFastMathAllowReassoc
    | LLVMZludaFastMathNoNaNs
    | LLVMZludaFastMathNoInfs
    | LLVMZludaFastMathNoSignedZeros
    | LLVMZludaFastMathAllowReciprocal
    | LLVMZludaFastMathAllowContract
    | LLVMZludaFastMathApproxFunc;

pub type LLVMZludaFastMathFlags = std::ffi::c_uint;

extern "C" {
    pub fn LLVMParseCommandLineOptions(
        argc: ::std::os::raw::c_int,
        argv: *const *const ::std::os::raw::c_char,
        overview: *const ::std::os::raw::c_char,
    );

    pub fn LLVMZludaBuildAlloca(
        B: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        AddrSpace: u32,
        Name: *const u8,
    ) -> LLVMValueRef;

    pub fn LLVMZludaBuildAtomicRMW(
        B: LLVMBuilderRef,
        op: LLVMZludaAtomicRMWBinOp,
        PTR: LLVMValueRef,
        Val: LLVMValueRef,
        scope: *const u8,
        ordering: LLVMAtomicOrdering,
    ) -> LLVMValueRef;

    pub fn LLVMZludaBuildAtomicCmpXchg(
        B: LLVMBuilderRef,
        Ptr: LLVMValueRef,
        Cmp: LLVMValueRef,
        New: LLVMValueRef,
        scope: *const u8,
        SuccessOrdering: LLVMAtomicOrdering,
        FailureOrdering: LLVMAtomicOrdering,
    ) -> LLVMValueRef;

    pub fn LLVMZludaSetFastMathFlags(FPMathInst: LLVMValueRef, FMF: LLVMZludaFastMathFlags);

    pub fn LLVMZludaBuildFence(
        B: LLVMBuilderRef,
        ordering: LLVMAtomicOrdering,
        scope: *const u8,
        Name: *const u8,
    ) -> LLVMValueRef;

    // DWARF Debug Info functions
    pub fn LLVMZludaSetCurrentDebugLocation(Builder: LLVMBuilderRef, L: LLVMMetadataRef);

    pub fn LLVMZludaInsertDeclareAtEnd(
        Builder: LLVMBuilderRef,
        Storage: LLVMValueRef,
        VarInfo: LLVMMetadataRef,
        Expr: LLVMMetadataRef,
        DL: LLVMMetadataRef,
        InsertAtEnd: LLVMBasicBlockRef,
    ) -> LLVMValueRef;

    pub fn LLVMZludaDIBuilderInsertDeclareRecordAtEnd(
        Builder: LLVMDIBuilderRef,
        Storage: LLVMValueRef,
        VarInfo: LLVMMetadataRef,
        Expr: LLVMMetadataRef,
        DL: LLVMMetadataRef,
        InsertAtEnd: LLVMBasicBlockRef,
    ) -> LLVMDbgRecordRef;

    pub fn LLVMZludaSizeOfTypeInBits(TD: LLVMTargetDataRef, Ty: LLVMTypeRef) -> u64;

    pub fn LLVMZludaSetAtomic(
        MemAccessInst: LLVMValueRef,
        Ordering: LLVMAtomicOrdering,
        Scope: *const u8,
    );

    // LLVM Target initialization functions (for standalone binaries)
    pub fn LLVMInitializeX86TargetInfo();
    pub fn LLVMInitializeX86Target();
    pub fn LLVMInitializeX86TargetMC();
    pub fn LLVMInitializeX86AsmPrinter();
    pub fn LLVMInitializeX86AsmParser();

    pub fn LLVMInitializeNVPTXTargetInfo();
    pub fn LLVMInitializeNVPTXTarget();
    pub fn LLVMInitializeNVPTXTargetMC();
    pub fn LLVMInitializeNVPTXAsmPrinter();
    // Note: NVPTX doesn't have AsmParser

    // LLVM Bitcode writing
    pub fn LLVMWriteBitcodeToMemoryBuffer(M: LLVMModuleRef) -> LLVMMemoryBufferRef;

    // LLVM Module verification
    pub fn LLVMVerifyModule(
        M: LLVMModuleRef,
        Action: LLVMVerifierFailureAction,
        OutMessage: *mut *mut ::std::os::raw::c_char,
    ) -> LLVMBool;

    // Value type checking
    pub fn LLVMIsAInstruction(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstant(Val: LLVMValueRef) -> LLVMValueRef;
}

/// Verifier failure actions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LLVMVerifierFailureAction {
    LLVMAbortProcessAction = 0,
    LLVMPrintMessageAction = 1,
    LLVMReturnStatusAction = 2,
}

/// Initialize all available targets for code generation
#[inline]
pub unsafe fn LLVM_InitializeAllTargetInfos() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeNVPTXTargetInfo();
}

#[inline]
pub unsafe fn LLVM_InitializeAllTargets() {
    LLVMInitializeX86Target();
    LLVMInitializeNVPTXTarget();
}

#[inline]
pub unsafe fn LLVM_InitializeAllTargetMCs() {
    LLVMInitializeX86TargetMC();
    LLVMInitializeNVPTXTargetMC();
}

#[inline]
pub unsafe fn LLVM_InitializeAllAsmPrinters() {
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeNVPTXAsmPrinter();
}

#[inline]
pub unsafe fn LLVM_InitializeAllAsmParsers() {
    LLVMInitializeX86AsmParser();
    // NVPTX doesn't have AsmParser
}
