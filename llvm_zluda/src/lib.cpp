#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <llvm-c/Core.h>
#include <llvm-c/DebugInfo.h>
#include <llvm-c/Target.h>
#include <llvm-c/Types.h>
#include <llvm/ADT/PointerUnion.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DebugProgramInstruction.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#pragma GCC diagnostic pop

using namespace llvm;

typedef enum {
  LLVMZludaAtomicRMWBinOpXchg, /**< Set the new value and return the one old */
  LLVMZludaAtomicRMWBinOpAdd,  /**< Add a value and return the old one */
  LLVMZludaAtomicRMWBinOpSub,  /**< Subtract a value and return the old one */
  LLVMZludaAtomicRMWBinOpAnd,  /**< And a value and return the old one */
  LLVMZludaAtomicRMWBinOpNand, /**< Not-And a value and return the old one */
  LLVMZludaAtomicRMWBinOpOr,   /**< OR a value and return the old one */
  LLVMZludaAtomicRMWBinOpXor,  /**< Xor a value and return the old one */
  LLVMZludaAtomicRMWBinOpMax,  /**< Sets the value if it's greater than the
                            original using a signed comparison and return
                            the old one */
  LLVMZludaAtomicRMWBinOpMin,  /**< Sets the value if it's Smaller than the
                            original using a signed comparison and return
                            the old one */
  LLVMZludaAtomicRMWBinOpUMax, /**< Sets the value if it's greater than the
                           original using an unsigned comparison and return
                           the old one */
  LLVMZludaAtomicRMWBinOpUMin, /**< Sets the value if it's greater than the
                            original using an unsigned comparison and return
                            the old one */
  LLVMZludaAtomicRMWBinOpFAdd, /**< Add a floating point value and return the
                            old one */
  LLVMZludaAtomicRMWBinOpFSub, /**< Subtract a floating point value and return
                          the old one */
  LLVMZludaAtomicRMWBinOpFMax, /**< Sets the value if it's greater than the
                           original using an floating point comparison and
                           return the old one */
  LLVMZludaAtomicRMWBinOpFMin, /**< Sets the value if it's smaller than the
                           original using an floating point comparison and
                           return the old one */
  LLVMZludaAtomicRMWBinOpUIncWrap, /**< Increments the value, wrapping back to
                               zero when incremented above input value */
  LLVMZludaAtomicRMWBinOpUDecWrap, /**< Decrements the value, wrapping back to
                               the input value when decremented below zero */
} LLVMZludaAtomicRMWBinOp;

static llvm::AtomicRMWInst::BinOp
mapFromLLVMRMWBinOp(LLVMZludaAtomicRMWBinOp BinOp) {
  switch (BinOp) {
  case LLVMZludaAtomicRMWBinOpXchg:
    return llvm::AtomicRMWInst::Xchg;
  case LLVMZludaAtomicRMWBinOpAdd:
    return llvm::AtomicRMWInst::Add;
  case LLVMZludaAtomicRMWBinOpSub:
    return llvm::AtomicRMWInst::Sub;
  case LLVMZludaAtomicRMWBinOpAnd:
    return llvm::AtomicRMWInst::And;
  case LLVMZludaAtomicRMWBinOpNand:
    return llvm::AtomicRMWInst::Nand;
  case LLVMZludaAtomicRMWBinOpOr:
    return llvm::AtomicRMWInst::Or;
  case LLVMZludaAtomicRMWBinOpXor:
    return llvm::AtomicRMWInst::Xor;
  case LLVMZludaAtomicRMWBinOpMax:
    return llvm::AtomicRMWInst::Max;
  case LLVMZludaAtomicRMWBinOpMin:
    return llvm::AtomicRMWInst::Min;
  case LLVMZludaAtomicRMWBinOpUMax:
    return llvm::AtomicRMWInst::UMax;
  case LLVMZludaAtomicRMWBinOpUMin:
    return llvm::AtomicRMWInst::UMin;
  case LLVMZludaAtomicRMWBinOpFAdd:
    return llvm::AtomicRMWInst::FAdd;
  case LLVMZludaAtomicRMWBinOpFSub:
    return llvm::AtomicRMWInst::FSub;
  case LLVMZludaAtomicRMWBinOpFMax:
    return llvm::AtomicRMWInst::FMax;
  case LLVMZludaAtomicRMWBinOpFMin:
    return llvm::AtomicRMWInst::FMin;
  case LLVMZludaAtomicRMWBinOpUIncWrap:
    return llvm::AtomicRMWInst::UIncWrap;
  case LLVMZludaAtomicRMWBinOpUDecWrap:
    return llvm::AtomicRMWInst::UDecWrap;
  }

  llvm_unreachable("Invalid LLVMZludaAtomicRMWBinOp value!");
}

static AtomicOrdering mapFromLLVMOrdering(LLVMAtomicOrdering Ordering) {
  switch (Ordering) {
  case LLVMAtomicOrderingNotAtomic:
    return AtomicOrdering::NotAtomic;
  case LLVMAtomicOrderingUnordered:
    return AtomicOrdering::Unordered;
  case LLVMAtomicOrderingMonotonic:
    return AtomicOrdering::Monotonic;
  case LLVMAtomicOrderingAcquire:
    return AtomicOrdering::Acquire;
  case LLVMAtomicOrderingRelease:
    return AtomicOrdering::Release;
  case LLVMAtomicOrderingAcquireRelease:
    return AtomicOrdering::AcquireRelease;
  case LLVMAtomicOrderingSequentiallyConsistent:
    return AtomicOrdering::SequentiallyConsistent;
  }

  llvm_unreachable("Invalid LLVMAtomicOrdering value!");
}

typedef unsigned LLVMFastMathFlags;

static FastMathFlags mapFromLLVMFastMathFlags(LLVMFastMathFlags FMF) {
  FastMathFlags NewFMF;
  NewFMF.setAllowReassoc((FMF & LLVMFastMathAllowReassoc) != 0);
  NewFMF.setNoNaNs((FMF & LLVMFastMathNoNaNs) != 0);
  NewFMF.setNoInfs((FMF & LLVMFastMathNoInfs) != 0);
  NewFMF.setNoSignedZeros((FMF & LLVMFastMathNoSignedZeros) != 0);
  NewFMF.setAllowReciprocal((FMF & LLVMFastMathAllowReciprocal) != 0);
  NewFMF.setAllowContract((FMF & LLVMFastMathAllowContract) != 0);
  NewFMF.setApproxFunc((FMF & LLVMFastMathApproxFunc) != 0);

  return NewFMF;
}

LLVM_C_EXTERN_C_BEGIN

LLVMValueRef LLVMZludaBuildAlloca(LLVMBuilderRef B, LLVMTypeRef Ty,
                                  unsigned AddrSpace, const char *Name) {
  return llvm::wrap(llvm::unwrap(B)->CreateAlloca(llvm::unwrap(Ty), AddrSpace,
                                                  nullptr, Name));
}

LLVMValueRef LLVMZludaBuildAtomicRMW(LLVMBuilderRef B,
                                     LLVMZludaAtomicRMWBinOp op,
                                     LLVMValueRef PTR, LLVMValueRef Val,
                                     char *scope, LLVMAtomicOrdering ordering) {
  auto builder = llvm::unwrap(B);
  LLVMContext &context = builder->getContext();
  llvm::AtomicRMWInst::BinOp intop = mapFromLLVMRMWBinOp(op);

  // Map AMD-specific sync scopes to NVPTX-compatible ones
  std::string nvptx_scope;
  if (scope && strlen(scope) > 0) {
    std::string scope_str(scope);
    if (scope_str == "agent-one-as") {
      nvptx_scope = ""; // Use default/system scope for NVPTX
    } else if (scope_str == "workgroup-one-as") {
      nvptx_scope = ""; // Use default scope for NVPTX
    } else if (scope_str == "one-as") {
      nvptx_scope = ""; // Use default scope for NVPTX
    } else {
      nvptx_scope = scope_str;
    }
  }

  return llvm::wrap(builder->CreateAtomicRMW(
      intop, llvm::unwrap(PTR), llvm::unwrap(Val), llvm::MaybeAlign(),
      mapFromLLVMOrdering(ordering),
      nvptx_scope.empty() ? llvm::SyncScope::System
                          : context.getOrInsertSyncScopeID(nvptx_scope)));
}

LLVMValueRef LLVMZludaBuildAtomicCmpXchg(LLVMBuilderRef B, LLVMValueRef Ptr,
                                         LLVMValueRef Cmp, LLVMValueRef New,
                                         char *scope,
                                         LLVMAtomicOrdering SuccessOrdering,
                                         LLVMAtomicOrdering FailureOrdering) {
  auto builder = llvm::unwrap(B);
  LLVMContext &context = builder->getContext();
  return wrap(builder->CreateAtomicCmpXchg(
      unwrap(Ptr), unwrap(Cmp), unwrap(New), MaybeAlign(),
      mapFromLLVMOrdering(SuccessOrdering),
      mapFromLLVMOrdering(FailureOrdering),
      context.getOrInsertSyncScopeID(scope)));
}

void LLVMZludaSetFastMathFlags(LLVMValueRef FPMathInst, LLVMFastMathFlags FMF) {
  Value *P = unwrap<Value>(FPMathInst);
  cast<Instruction>(P)->setFastMathFlags(mapFromLLVMFastMathFlags(FMF));
}

void LLVMZludaBuildFence(LLVMBuilderRef B, LLVMAtomicOrdering Ordering,
                         char *scope, const char *Name) {
  auto builder = llvm::unwrap(B);
  LLVMContext &context = builder->getContext();
  builder->CreateFence(mapFromLLVMOrdering(Ordering),
                       context.getOrInsertSyncScopeID(scope), Name);
}

void LLVMZludaSetCurrentDebugLocation(LLVMBuilderRef Builder,
                                      LLVMMetadataRef L) {
  IRBuilder<> *B = unwrap(Builder);
  if (L) {
    B->SetCurrentDebugLocation(DebugLoc(unwrap<DILocation>(L)));
  } else {
    B->SetCurrentDebugLocation(DebugLoc());
  }
}

LLVMValueRef
LLVMZludaInsertDeclareAtEnd(LLVMBuilderRef Builder, LLVMValueRef Storage,
                            LLVMMetadataRef VarInfo, LLVMMetadataRef Expr,
                            LLVMMetadataRef DL, LLVMBasicBlockRef InsertAtEnd) {
  IRBuilder<> *B = unwrap(Builder);
  Module *M = B->GetInsertBlock()->getModule();

  // DIB.insertDeclare returns a DbgInstPtr (PointerUnion<Instruction*,
  // DbgRecord*>) But in C bindings, we need a LLVMValueRef (which wraps a
  // Value*) So we need to create the intrinsic directly to get an Instruction*

  Function *DeclareFn = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::dbg_declare, ArrayRef<Type *>());
  if (!DeclareFn)
    return nullptr;

  IRBuilder<> IB(unwrap(InsertAtEnd));

  // Carefully construct arguments with correct types
  Value *StorageVal = unwrap(Storage);
  MetadataAsValue *VarInfoVal =
      MetadataAsValue::get(B->getContext(), unwrap(VarInfo));
  MetadataAsValue *ExprVal =
      MetadataAsValue::get(B->getContext(), unwrap(Expr));

  // Create arguments array
  Value *Args[] = {StorageVal, VarInfoVal, ExprVal};

  // Create the call instruction
  CallInst *Call = IB.CreateCall(DeclareFn, Args);

  // Set debug location if available
  if (DL)
    Call->setDebugLoc(DebugLoc(unwrap<DILocation>(DL)));

  return wrap(Call);
}

// Implement LLVMZludaDIBuilderInsertDeclareRecordAtEnd for record-based debug
// declarations
extern "C" LLVMDbgRecordRef LLVMZludaDIBuilderInsertDeclareRecordAtEnd(
    LLVMDIBuilderRef Builder, LLVMValueRef Storage, LLVMMetadataRef VarInfo,
    LLVMMetadataRef Expr, LLVMMetadataRef DL, LLVMBasicBlockRef InsertAtEnd) {
  return LLVMDIBuilderInsertDeclareRecordAtEnd(Builder, Storage, VarInfo, Expr,
                                               DL, InsertAtEnd);
}

unsigned long long LLVMZludaSizeOfTypeInBits(LLVMTargetDataRef TD,
                                             LLVMTypeRef Ty) {
  return LLVMSizeOfTypeInBits(TD, Ty);
}

void LLVMZludaSetAtomic(LLVMValueRef MemAccessInst, LLVMAtomicOrdering Ordering,
                        char *Scope) {
  Value *Inst = unwrap<Value>(MemAccessInst);
  LLVMContext &Context = Inst->getContext();

  if (auto *LI = dyn_cast<LoadInst>(Inst)) {
    LI->setAtomic(mapFromLLVMOrdering(Ordering));
    if (Scope && strlen(Scope) > 0) {
      LI->setSyncScopeID(Context.getOrInsertSyncScopeID(Scope));
    }
  } else if (auto *SI = dyn_cast<StoreInst>(Inst)) {
    SI->setAtomic(mapFromLLVMOrdering(Ordering));
    if (Scope && strlen(Scope) > 0) {
      SI->setSyncScopeID(Context.getOrInsertSyncScopeID(Scope));
    }
  }
}

LLVM_C_EXTERN_C_END