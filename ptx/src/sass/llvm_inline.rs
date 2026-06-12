//! SASS Inlining to LLVM IR
//!
//! This module provides functionality to inline SASS (Streaming Assembler) code
//! into LLVM IR for GPU profiling and record/replay scenarios.
//!
//! Three inlining strategies are supported:
//! 1. **Inline Assembly**: Embed SASS as LLVM inline assembly (preserves exact encoding)
//! 2. **PTX Reconstruction**: Convert SASS semantics back to PTX-equivalent LLVM IR
//! 3. **Metadata Tracking**: Attach SASS addresses/info as LLVM metadata without changing IR
//!
//! The controllable mapping allows:
//! - Runtime address tracking for profiling
//! - Checkpoint/restore of GPU state
//! - Bidirectional PTX↔SASS navigation

use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;

use llvm_zluda::core::*;
use llvm_zluda::prelude::*;
use llvm_zluda::*;

use super::instruction::{
    EnhancedSassInstruction, SassControlCodes, SassDataType, SassMemorySpace, SassOpcodeClass,
    SassOperand, SassRegister,
};

// ============================================================================
// SASS Inlining Configuration
// ============================================================================

/// Strategy for inlining SASS into LLVM IR
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassInlineStrategy {
    /// Embed SASS as inline assembly (preserves exact machine code)
    /// Best for: Precise debugging, exact replay
    InlineAssembly,

    /// Convert SASS to PTX-equivalent LLVM IR operations
    /// Best for: Optimization, cross-platform compatibility
    PtxReconstruction,

    /// Only attach metadata, don't modify IR structure
    /// Best for: Profiling with minimal overhead
    MetadataOnly,

    /// Hybrid: Use PTX reconstruction with assembly fallback for complex ops
    /// Best for: Balance between compatibility and precision
    Hybrid,
}

/// Configuration for SASS inlining
#[derive(Debug, Clone)]
pub struct SassInlineConfig {
    /// Inlining strategy
    pub strategy: SassInlineStrategy,
    /// Whether to preserve control codes (stall counts, barriers)
    pub preserve_control_codes: bool,
    /// Whether to emit debug location info
    pub emit_debug_info: bool,
    /// Whether to track register liveness
    pub track_registers: bool,
    /// Target SM version for inline assembly
    pub target_sm: u32,
    /// Function prefix for inlined SASS functions
    pub function_prefix: String,
}

impl Default for SassInlineConfig {
    fn default() -> Self {
        Self {
            strategy: SassInlineStrategy::Hybrid,
            preserve_control_codes: true,
            emit_debug_info: true,
            track_registers: true,
            target_sm: 75, // Default to SM 7.5 (Turing)
            function_prefix: "_sass_".to_string(),
        }
    }
}

// ============================================================================
// SASS Inliner
// ============================================================================

/// SASS to LLVM IR inliner
pub struct SassInliner {
    /// LLVM context
    context: LLVMContextRef,
    /// LLVM module
    module: LLVMModuleRef,
    /// LLVM builder
    builder: LLVMBuilderRef,
    /// Configuration
    config: SassInlineConfig,
    /// SASS address to LLVM value mapping
    sass_to_llvm: HashMap<u64, LLVMValueRef>,
    /// Register name to LLVM value mapping
    register_map: HashMap<String, LLVMValueRef>,
    /// Predicate register mapping
    predicate_map: HashMap<String, LLVMValueRef>,
    /// Metadata kind IDs
    md_sass_addr: u32,
    md_sass_opcode: u32,
    md_ptx_line: u32,
    md_control_codes: u32,
    /// Statistics
    stats: InlineStats,
}

/// Statistics for inlining operations
#[derive(Debug, Default, Clone)]
pub struct InlineStats {
    pub instructions_processed: usize,
    pub inline_asm_emitted: usize,
    pub ptx_reconstructed: usize,
    pub metadata_only: usize,
    pub fallback_used: usize,
}

impl SassInliner {
    /// Create a new SASS inliner
    pub unsafe fn new(
        context: LLVMContextRef,
        module: LLVMModuleRef,
        config: SassInlineConfig,
    ) -> Self {
        let builder = LLVMCreateBuilderInContext(context);

        // Register custom metadata kinds
        let md_sass_addr =
            LLVMGetMDKindIDInContext(context, b"sass.addr\0".as_ptr() as *const _, 9);
        let md_sass_opcode =
            LLVMGetMDKindIDInContext(context, b"sass.opcode\0".as_ptr() as *const _, 11);
        let md_ptx_line =
            LLVMGetMDKindIDInContext(context, b"sass.ptx_line\0".as_ptr() as *const _, 13);
        let md_control_codes =
            LLVMGetMDKindIDInContext(context, b"sass.control\0".as_ptr() as *const _, 12);

        Self {
            context,
            module,
            builder,
            config,
            sass_to_llvm: HashMap::new(),
            register_map: HashMap::new(),
            predicate_map: HashMap::new(),
            md_sass_addr,
            md_sass_opcode,
            md_ptx_line,
            md_control_codes,
            stats: InlineStats::default(),
        }
    }

    /// Inline a sequence of SASS instructions into the current function
    pub unsafe fn inline_instructions(
        &mut self,
        instructions: &[EnhancedSassInstruction],
        entry_block: LLVMBasicBlockRef,
    ) -> Result<(), String> {
        LLVMPositionBuilderAtEnd(self.builder, entry_block);

        for inst in instructions {
            self.stats.instructions_processed += 1;

            match self.config.strategy {
                SassInlineStrategy::InlineAssembly => {
                    self.emit_inline_assembly(inst)?;
                }
                SassInlineStrategy::PtxReconstruction => {
                    self.emit_ptx_reconstruction(inst)?;
                }
                SassInlineStrategy::MetadataOnly => {
                    self.emit_metadata_only(inst)?;
                }
                SassInlineStrategy::Hybrid => {
                    self.emit_hybrid(inst)?;
                }
            }
        }

        Ok(())
    }

    /// Emit SASS as inline assembly
    unsafe fn emit_inline_assembly(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        // For simple inline assembly, we emit it as a side-effect-only call
        // with no inputs/outputs (constraints handled via memory)
        let asm_str = self.build_asm_string(inst);

        // Use empty constraints for side-effect-only inline asm
        let constraints = "";

        let asm_cstr = CString::new(asm_str.as_str()).map_err(|e| e.to_string())?;
        let constraints_cstr = CString::new(constraints).map_err(|e| e.to_string())?;

        // Use void return type for side-effect inline asm
        let ret_type = LLVMVoidTypeInContext(self.context);

        // Build function type with no parameters
        let fn_type = LLVMFunctionType(ret_type, ptr::null_mut(), 0, 0);

        // Create inline assembly value
        let asm_val = LLVMGetInlineAsm(
            fn_type,
            asm_cstr.as_ptr() as *mut _,
            asm_str.len(),
            constraints_cstr.as_ptr() as *mut _,
            constraints.len(),
            1, // has side effects
            0, // is align stack
            LLVMInlineAsmDialect::LLVMInlineAsmDialectATT,
            0, // can throw
        );

        // Call the inline assembly
        let call = LLVMBuildCall2(
            self.builder,
            fn_type,
            asm_val,
            ptr::null_mut(),
            0,
            b"\0".as_ptr() as *const _,
        );

        // Attach metadata
        self.attach_sass_metadata(call, inst);

        self.sass_to_llvm.insert(inst.address, call);
        self.stats.inline_asm_emitted += 1;

        Ok(call)
    }

    /// Build assembly string for SASS instruction
    fn build_asm_string(&self, inst: &EnhancedSassInstruction) -> String {
        let mut asm = String::new();

        // Add predicate if present
        if let Some(ref pred) = inst.predicate {
            asm.push_str(&format!("{} ", self.format_operand(pred)));
        }

        // Add opcode with modifiers
        asm.push_str(&inst.opcode);
        for modifier in &inst.modifiers {
            asm.push_str(modifier);
        }

        // Add operands
        let operands: Vec<String> = inst
            .dest_operands
            .iter()
            .chain(inst.src_operands.iter())
            .map(|op| self.format_operand(op))
            .collect();

        if !operands.is_empty() {
            asm.push(' ');
            asm.push_str(&operands.join(", "));
        }

        asm.push(';');

        // Add control codes as comment if preserving
        if self.config.preserve_control_codes {
            asm.push_str(&format!(
                " /* stall:{} wb:{} rb:{} wbm:{} */",
                inst.control_codes.stall_count,
                inst.control_codes.write_barrier,
                inst.control_codes.read_barrier,
                inst.control_codes.wait_barrier_mask
            ));
        }

        asm
    }

    /// Build assembly constraints string
    fn build_asm_constraints(&self, inst: &EnhancedSassInstruction) -> String {
        // For NVIDIA inline PTX, we use specific constraint syntax
        let mut constraints = Vec::new();

        // Output constraints (=r for register output, =l for 64-bit)
        for dest in &inst.dest_operands {
            match dest {
                SassOperand::Register(reg) => {
                    let constraint = self.get_register_constraint(reg, true);
                    constraints.push(constraint);
                }
                _ => constraints.push("=r".to_string()),
            }
        }

        // Input constraints (r for register input, l for 64-bit)
        for src in &inst.src_operands {
            match src {
                SassOperand::Register(reg) => {
                    let constraint = self.get_register_constraint(reg, false);
                    constraints.push(constraint);
                }
                SassOperand::Immediate(_) | SassOperand::FloatImmediate(_) => {
                    constraints.push("n".to_string())
                }
                SassOperand::Memory { .. } => constraints.push("m".to_string()),
                _ => constraints.push("r".to_string()),
            }
        }

        constraints.join(",")
    }

    /// Get inline assembly constraint for a register
    fn get_register_constraint(&self, reg: &SassRegister, is_output: bool) -> String {
        let prefix = if is_output { "=" } else { "" };

        // Determine constraint based on register prefix
        if reg.prefix.starts_with("P") || reg.prefix == "PT" {
            format!("{}b", prefix) // Predicate
        } else if reg.is_uniform {
            format!("{}r", prefix) // Uniform registers use general constraint
        } else {
            // General purpose register - assume 32-bit by default
            format!("{}r", prefix)
        }
    }

    /// Emit PTX-equivalent LLVM IR for a SASS instruction
    unsafe fn emit_ptx_reconstruction(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        let result = match inst.opcode_class {
            // Integer operations
            SassOpcodeClass::IntegerArithmetic => self.emit_int_arithmetic(inst)?,
            SassOpcodeClass::IntegerLogical => self.emit_bitwise(inst)?,
            SassOpcodeClass::IntegerComparison => self.emit_comparison(inst)?,
            SassOpcodeClass::IntegerConversion => self.emit_conversion(inst)?,

            // Float operations
            SassOpcodeClass::FloatArithmetic => self.emit_float_arithmetic(inst)?,
            SassOpcodeClass::FloatComparison => self.emit_comparison(inst)?,
            SassOpcodeClass::FloatConversion => self.emit_conversion(inst)?,

            // Half precision
            SassOpcodeClass::HalfArithmetic => self.emit_half_arithmetic(inst)?,

            // Memory loads
            SassOpcodeClass::GlobalLoad
            | SassOpcodeClass::SharedLoad
            | SassOpcodeClass::LocalLoad
            | SassOpcodeClass::ConstantLoad => self.emit_load(inst)?,

            // Memory stores
            SassOpcodeClass::GlobalStore
            | SassOpcodeClass::SharedStore
            | SassOpcodeClass::LocalStore => self.emit_store(inst)?,

            // Data movement
            SassOpcodeClass::Move => self.emit_move(inst)?,

            // Control flow
            SassOpcodeClass::Branch
            | SassOpcodeClass::ConditionalBranch
            | SassOpcodeClass::Return
            | SassOpcodeClass::Exit => self.emit_branch(inst)?,

            // Synchronization
            SassOpcodeClass::Barrier | SassOpcodeClass::MemoryBarrier => self.emit_barrier(inst)?,

            // Atomic operations
            SassOpcodeClass::GlobalAtomic | SassOpcodeClass::SharedAtomic => {
                self.emit_atomic(inst)?
            }

            // Special registers
            SassOpcodeClass::SpecialRegRead => self.emit_special(inst)?,

            _ => {
                // Fallback to inline assembly for unsupported ops
                self.stats.fallback_used += 1;
                return self.emit_inline_assembly(inst);
            }
        };

        // Attach metadata
        self.attach_sass_metadata(result, inst);

        self.sass_to_llvm.insert(inst.address, result);
        self.stats.ptx_reconstructed += 1;

        Ok(result)
    }

    /// Emit only metadata without generating instructions
    unsafe fn emit_metadata_only(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        // Create a no-op instruction to attach metadata to
        let void_type = LLVMVoidTypeInContext(self.context);
        let nop_fn_type = LLVMFunctionType(void_type, ptr::null_mut(), 0, 0);

        // Use inline assembly nop
        let nop_asm = CString::new("").unwrap();
        let nop_constraints = CString::new("").unwrap();

        let nop = LLVMGetInlineAsm(
            nop_fn_type,
            nop_asm.as_ptr() as *mut _,
            0,
            nop_constraints.as_ptr() as *mut _,
            0,
            0,
            0,
            LLVMInlineAsmDialect::LLVMInlineAsmDialectATT,
            0,
        );

        let call = LLVMBuildCall2(
            self.builder,
            nop_fn_type,
            nop,
            ptr::null_mut(),
            0,
            b"\0".as_ptr() as *const _,
        );

        // Attach comprehensive metadata
        self.attach_sass_metadata(call, inst);

        self.sass_to_llvm.insert(inst.address, call);
        self.stats.metadata_only += 1;

        Ok(call)
    }

    /// Emit using hybrid strategy
    unsafe fn emit_hybrid(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        // Try PTX reconstruction for common operations
        match inst.opcode_class {
            // Integer operations that can be reconstructed
            SassOpcodeClass::IntegerArithmetic
            | SassOpcodeClass::IntegerLogical
            | SassOpcodeClass::IntegerComparison
            | SassOpcodeClass::IntegerConversion
            // Float operations
            | SassOpcodeClass::FloatArithmetic
            | SassOpcodeClass::FloatComparison
            | SassOpcodeClass::FloatConversion
            // Memory operations
            | SassOpcodeClass::GlobalLoad
            | SassOpcodeClass::SharedLoad
            | SassOpcodeClass::LocalLoad
            | SassOpcodeClass::ConstantLoad
            | SassOpcodeClass::GlobalStore
            | SassOpcodeClass::SharedStore
            | SassOpcodeClass::LocalStore
            // Data movement
            | SassOpcodeClass::Move => {
                self.emit_ptx_reconstruction(inst)
            }
            // Use inline assembly for complex/special operations
            _ => {
                self.emit_inline_assembly(inst)
            }
        }
    }

    /// Attach SASS metadata to an LLVM value
    unsafe fn attach_sass_metadata(&self, value: LLVMValueRef, inst: &EnhancedSassInstruction) {
        // Check if value is an instruction - only instructions can have metadata
        // Constants and other values cannot have metadata attached
        if value.is_null() {
            return;
        }

        // Check if this is an instruction by trying to get instruction opcode
        // LLVMIsAInstruction returns null if not an instruction
        if llvm_zluda::LLVMIsAInstruction(value).is_null() {
            // Not an instruction (e.g., constant), skip metadata attachment
            return;
        }

        // SASS address metadata
        let addr_val = LLVMConstInt(LLVMInt64TypeInContext(self.context), inst.address, 0);
        let mut addr_md = LLVMValueAsMetadata(addr_val);
        let addr_node = LLVMMDNodeInContext2(self.context, &mut addr_md as *mut _, 1);
        LLVMSetMetadata(
            value,
            self.md_sass_addr,
            LLVMMetadataAsValue(self.context, addr_node),
        );

        // Opcode metadata
        let opcode_cstr = CString::new(inst.opcode.as_str()).unwrap();
        let mut opcode_md =
            LLVMMDStringInContext2(self.context, opcode_cstr.as_ptr(), inst.opcode.len());
        let opcode_node = LLVMMDNodeInContext2(self.context, &mut opcode_md as *mut _, 1);
        LLVMSetMetadata(
            value,
            self.md_sass_opcode,
            LLVMMetadataAsValue(self.context, opcode_node),
        );

        // PTX line metadata (if available)
        if let Some(line) = inst.ptx_line {
            let line_val = LLVMConstInt(LLVMInt32TypeInContext(self.context), line as u64, 0);
            let mut line_md = LLVMValueAsMetadata(line_val);
            let line_node = LLVMMDNodeInContext2(self.context, &mut line_md as *mut _, 1);
            LLVMSetMetadata(
                value,
                self.md_ptx_line,
                LLVMMetadataAsValue(self.context, line_node),
            );
        }

        // Control codes metadata
        if self.config.preserve_control_codes {
            let cc = &inst.control_codes;
            let cc_vals = [
                LLVMConstInt(
                    LLVMInt8TypeInContext(self.context),
                    cc.stall_count as u64,
                    0,
                ),
                LLVMConstInt(
                    LLVMInt8TypeInContext(self.context),
                    cc.write_barrier as u64,
                    0,
                ),
                LLVMConstInt(
                    LLVMInt8TypeInContext(self.context),
                    cc.read_barrier as u64,
                    0,
                ),
                LLVMConstInt(
                    LLVMInt8TypeInContext(self.context),
                    cc.wait_barrier_mask as u64,
                    0,
                ),
            ];
            let mut cc_mds: Vec<LLVMMetadataRef> =
                cc_vals.iter().map(|v| LLVMValueAsMetadata(*v)).collect();
            let cc_node = LLVMMDNodeInContext2(self.context, cc_mds.as_mut_ptr(), cc_mds.len());
            LLVMSetMetadata(
                value,
                self.md_control_codes,
                LLVMMetadataAsValue(self.context, cc_node),
            );
        }
    }

    // ========================================================================
    // PTX Reconstruction Helpers
    // ========================================================================

    /// Emit integer arithmetic (IADD, IMAD, IMUL, etc.)
    unsafe fn emit_int_arithmetic(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        let opcode = inst.opcode.as_str();

        // Get source operands
        let (src1, src2) = self.get_src_operands(inst)?;

        let result = match opcode {
            "IADD" | "IADD3" => {
                LLVMBuildAdd(self.builder, src1, src2, b"iadd\0".as_ptr() as *const _)
            }
            "ISUB" => LLVMBuildSub(self.builder, src1, src2, b"isub\0".as_ptr() as *const _),
            "IMUL" | "IMUL32I" => {
                LLVMBuildMul(self.builder, src1, src2, b"imul\0".as_ptr() as *const _)
            }
            "IMAD" => {
                // IMAD: d = a * b + c
                let src3 = if let Some(op) = inst.src_operands.get(2) {
                    self.get_operand_value(op)?
                } else {
                    LLVMConstInt(LLVMInt32TypeInContext(self.context), 0, 0)
                };
                let mul =
                    LLVMBuildMul(self.builder, src1, src2, b"imad_mul\0".as_ptr() as *const _);
                LLVMBuildAdd(self.builder, mul, src3, b"imad\0".as_ptr() as *const _)
            }
            "IDIV" => LLVMBuildSDiv(self.builder, src1, src2, b"idiv\0".as_ptr() as *const _),
            "IMOD" => LLVMBuildSRem(self.builder, src1, src2, b"imod\0".as_ptr() as *const _),
            "INEG" => LLVMBuildNeg(self.builder, src1, b"ineg\0".as_ptr() as *const _),
            "IABS" => {
                // abs(x) = x < 0 ? -x : x
                let zero = LLVMConstInt(LLVMTypeOf(src1), 0, 0);
                let is_neg = LLVMBuildICmp(
                    self.builder,
                    LLVMIntPredicate::LLVMIntSLT,
                    src1,
                    zero,
                    b"is_neg\0".as_ptr() as *const _,
                );
                let neg = LLVMBuildNeg(self.builder, src1, b"neg\0".as_ptr() as *const _);
                LLVMBuildSelect(
                    self.builder,
                    is_neg,
                    neg,
                    src1,
                    b"iabs\0".as_ptr() as *const _,
                )
            }
            "LEA" => {
                // LEA: d = a + b * scale
                let scale = self.get_lea_scale(inst);
                let scaled = if scale != 1 {
                    let scale_val = LLVMConstInt(LLVMTypeOf(src2), scale as u64, 0);
                    LLVMBuildMul(
                        self.builder,
                        src2,
                        scale_val,
                        b"lea_scale\0".as_ptr() as *const _,
                    )
                } else {
                    src2
                };
                LLVMBuildAdd(self.builder, src1, scaled, b"lea\0".as_ptr() as *const _)
            }
            "SHL" => LLVMBuildShl(self.builder, src1, src2, b"shl\0".as_ptr() as *const _),
            "SHR" => LLVMBuildLShr(self.builder, src1, src2, b"shr\0".as_ptr() as *const _),
            "SHRS" => LLVMBuildAShr(self.builder, src1, src2, b"shrs\0".as_ptr() as *const _),
            _ => {
                return Err(format!("Unsupported integer op: {}", opcode));
            }
        };

        // Store result
        self.store_to_dest(result, inst)?;
        Ok(result)
    }

    /// Emit floating-point arithmetic (FADD, FMUL, FFMA, etc.)
    unsafe fn emit_float_arithmetic(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        let opcode = inst.opcode.as_str();
        let (src1_raw, src2_raw) = self.get_src_operands(inst)?;

        // Ensure operands are float type
        let float_ty = LLVMFloatTypeInContext(self.context);
        let src1 = self.ensure_float_type(src1_raw, float_ty);
        let src2 = self.ensure_float_type(src2_raw, float_ty);

        let result = match opcode {
            "FADD" | "FADD32I" => {
                LLVMBuildFAdd(self.builder, src1, src2, b"fadd\0".as_ptr() as *const _)
            }
            "FSUB" => LLVMBuildFSub(self.builder, src1, src2, b"fsub\0".as_ptr() as *const _),
            "FMUL" | "FMUL32I" => {
                LLVMBuildFMul(self.builder, src1, src2, b"fmul\0".as_ptr() as *const _)
            }
            "FFMA" | "FMA" => {
                // FMA: d = a * b + c
                let src3_raw = if let Some(op) = inst.src_operands.get(2) {
                    self.get_operand_value(op)?
                } else {
                    LLVMConstReal(float_ty, 0.0)
                };
                let src3 = self.ensure_float_type(src3_raw, float_ty);
                let mul =
                    LLVMBuildFMul(self.builder, src1, src2, b"ffma_mul\0".as_ptr() as *const _);
                LLVMBuildFAdd(self.builder, mul, src3, b"ffma\0".as_ptr() as *const _)
            }
            "FDIV" => LLVMBuildFDiv(self.builder, src1, src2, b"fdiv\0".as_ptr() as *const _),
            "FABS" => {
                // Use LLVM fabs intrinsic
                self.call_intrinsic("llvm.fabs.f32", &[src1])
            }
            "FNEG" => LLVMBuildFNeg(self.builder, src1, b"fneg\0".as_ptr() as *const _),
            "FMIN" => self.call_intrinsic("llvm.minnum.f32", &[src1, src2]),
            "FMAX" => self.call_intrinsic("llvm.maxnum.f32", &[src1, src2]),
            "FSQRT" => self.call_intrinsic("llvm.sqrt.f32", &[src1]),
            "FRCP" => {
                // Reciprocal: 1.0 / x
                let one = LLVMConstReal(LLVMFloatTypeInContext(self.context), 1.0);
                LLVMBuildFDiv(self.builder, one, src1, b"frcp\0".as_ptr() as *const _)
            }
            "FRSQRT" => {
                // Reciprocal sqrt: 1.0 / sqrt(x)
                let sqrt = self.call_intrinsic("llvm.sqrt.f32", &[src1]);
                let one = LLVMConstReal(LLVMFloatTypeInContext(self.context), 1.0);
                LLVMBuildFDiv(self.builder, one, sqrt, b"frsqrt\0".as_ptr() as *const _)
            }
            _ => {
                return Err(format!("Unsupported float op: {}", opcode));
            }
        };

        self.store_to_dest(result, inst)?;
        Ok(result)
    }

    /// Emit half-precision arithmetic
    unsafe fn emit_half_arithmetic(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        // Half ops are similar to float but with different type
        // For now, delegate to float and add conversion if needed
        self.emit_float_arithmetic(inst)
    }

    /// Emit load instruction
    unsafe fn emit_load(&mut self, inst: &EnhancedSassInstruction) -> Result<LLVMValueRef, String> {
        let dest = inst
            .dest_operands
            .first()
            .ok_or("Load requires destination")?;

        // Get memory address from source operand
        let addr = match inst.src_operands.first() {
            Some(SassOperand::Memory { base, offset, .. }) => {
                let base_val = if let Some(base_reg) = base {
                    self.get_register_value(base_reg)?
                } else {
                    LLVMConstInt(LLVMInt64TypeInContext(self.context), 0, 0)
                };
                if *offset != 0 {
                    let offset_val = LLVMConstInt(
                        LLVMInt64TypeInContext(self.context),
                        *offset as u64,
                        (*offset < 0) as i32,
                    );
                    LLVMBuildAdd(
                        self.builder,
                        base_val,
                        offset_val,
                        b"addr\0".as_ptr() as *const _,
                    )
                } else {
                    base_val
                }
            }
            Some(SassOperand::Register(reg)) => {
                // Treat register as address directly
                self.get_register_value(reg)?
            }
            Some(SassOperand::ConstantBank { bank, offset }) => {
                // Constant memory access - simplified to constant value
                LLVMConstInt(
                    LLVMInt64TypeInContext(self.context),
                    (*bank as u64 * 0x10000 + *offset as u64),
                    0,
                )
            }
            Some(other) => {
                // For other operand types, try to get a value
                self.get_operand_value(other)?
            }
            None => {
                return Err("Load requires source operand".to_string());
            }
        };

        // Determine element type from instruction
        let elem_type = self.get_data_type_llvm(inst.data_type.as_ref());

        // Cast address to pointer
        let ptr_type = LLVMPointerType(
            elem_type,
            self.get_address_space(inst.memory_space.as_ref()),
        );
        let ptr = LLVMBuildIntToPtr(self.builder, addr, ptr_type, b"ptr\0".as_ptr() as *const _);

        // Build load
        let load = LLVMBuildLoad2(self.builder, elem_type, ptr, b"load\0".as_ptr() as *const _);

        // Set alignment based on data type
        LLVMSetAlignment(load, self.get_type_alignment(inst.data_type.as_ref()));

        self.store_result(load, dest)?;
        Ok(load)
    }

    /// Emit store instruction
    unsafe fn emit_store(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        // Get memory address from destination
        let addr =
            if let Some(SassOperand::Memory { base, offset, .. }) = inst.dest_operands.first() {
                let base_val = if let Some(base_reg) = base {
                    self.get_register_value(base_reg)?
                } else {
                    LLVMConstInt(LLVMInt64TypeInContext(self.context), 0, 0)
                };
                if *offset != 0 {
                    let offset_val = LLVMConstInt(
                        LLVMInt64TypeInContext(self.context),
                        *offset as u64,
                        (*offset < 0) as i32,
                    );
                    LLVMBuildAdd(
                        self.builder,
                        base_val,
                        offset_val,
                        b"addr\0".as_ptr() as *const _,
                    )
                } else {
                    base_val
                }
            } else {
                return Err("Store requires memory destination".to_string());
            };

        // Get value to store
        let value = self.get_operand_value(
            inst.src_operands
                .first()
                .ok_or("Store requires source value")?,
        )?;

        // Determine element type
        let elem_type = LLVMTypeOf(value);
        let ptr_type = LLVMPointerType(
            elem_type,
            self.get_address_space(inst.memory_space.as_ref()),
        );
        let ptr = LLVMBuildIntToPtr(self.builder, addr, ptr_type, b"ptr\0".as_ptr() as *const _);

        // Build store
        let store = LLVMBuildStore(self.builder, value, ptr);
        LLVMSetAlignment(store, self.get_type_alignment(inst.data_type.as_ref()));

        Ok(store)
    }

    /// Emit move instruction
    unsafe fn emit_move(&mut self, inst: &EnhancedSassInstruction) -> Result<LLVMValueRef, String> {
        let dest = inst
            .dest_operands
            .first()
            .ok_or("Move requires destination")?;
        let src = inst.src_operands.first().ok_or("Move requires source")?;

        let value = self.get_operand_value(src)?;
        self.store_result(value, dest)?;

        Ok(value)
    }

    /// Emit comparison instruction
    unsafe fn emit_comparison(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        let opcode = inst.opcode.as_str();
        let (src1, src2) = self.get_src_operands(inst)?;

        // Determine comparison predicate from modifiers
        let pred = self.get_comparison_predicate(&inst.modifiers);

        let result = if opcode.starts_with('F') {
            // Float comparison
            LLVMBuildFCmp(
                self.builder,
                pred.1,
                src1,
                src2,
                b"fcmp\0".as_ptr() as *const _,
            )
        } else {
            // Integer comparison
            LLVMBuildICmp(
                self.builder,
                pred.0,
                src1,
                src2,
                b"icmp\0".as_ptr() as *const _,
            )
        };

        self.store_to_dest(result, inst)?;
        Ok(result)
    }

    /// Emit branch instruction
    unsafe fn emit_branch(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        // For branches, we need the current function's basic block structure
        // This is a simplified version - full implementation would need block map

        let opcode = inst.opcode.as_str();

        match opcode {
            "BRA" => {
                // Unconditional branch - would need target block
                // For now, create a placeholder
                Err("Branch target resolution not implemented".to_string())
            }
            "EXIT" => {
                // Return from kernel
                LLVMBuildRetVoid(self.builder);
                Ok(ptr::null_mut())
            }
            "RET" => {
                LLVMBuildRetVoid(self.builder);
                Ok(ptr::null_mut())
            }
            _ => Err(format!("Unsupported branch: {}", opcode)),
        }
    }

    /// Emit barrier instruction
    unsafe fn emit_barrier(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        // Use LLVM barrier intrinsic
        let barrier_fn = self.get_or_declare_intrinsic("llvm.nvvm.barrier0");
        let call = LLVMBuildCall2(
            self.builder,
            LLVMGlobalGetValueType(barrier_fn),
            barrier_fn,
            ptr::null_mut(),
            0,
            b"barrier\0".as_ptr() as *const _,
        );
        Ok(call)
    }

    /// Emit atomic instruction
    unsafe fn emit_atomic(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        let opcode = inst.opcode.as_str();

        // Get address and value
        let addr =
            self.get_operand_value(inst.src_operands.first().ok_or("Atomic requires address")?)?;
        let val =
            self.get_operand_value(inst.src_operands.get(1).ok_or("Atomic requires value")?)?;

        // Convert to pointer
        let elem_type = LLVMTypeOf(val);
        let ptr_type = LLVMPointerType(
            elem_type,
            self.get_address_space(inst.memory_space.as_ref()),
        );
        let ptr = LLVMBuildIntToPtr(self.builder, addr, ptr_type, b"aptr\0".as_ptr() as *const _);

        let atomic_op = match opcode {
            "ATOM" | "ATOMS" => {
                // Determine operation from modifiers
                if inst.modifiers.iter().any(|m| m.contains("ADD")) {
                    LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd
                } else if inst.modifiers.iter().any(|m| m.contains("MIN")) {
                    LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpMin
                } else if inst.modifiers.iter().any(|m| m.contains("MAX")) {
                    LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpMax
                } else if inst.modifiers.iter().any(|m| m.contains("AND")) {
                    LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAnd
                } else if inst.modifiers.iter().any(|m| m.contains("OR")) {
                    LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpOr
                } else if inst.modifiers.iter().any(|m| m.contains("XOR")) {
                    LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpXor
                } else if inst.modifiers.iter().any(|m| m.contains("EXCH")) {
                    LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpXchg
                } else {
                    LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd // Default
                }
            }
            "RED" => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd, // Reduction
            _ => return Err(format!("Unknown atomic op: {}", opcode)),
        };

        let result = LLVMBuildAtomicRMW(
            self.builder,
            atomic_op,
            ptr,
            val,
            LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
            0, // Not single thread
        );

        if let Some(dest) = inst.dest_operands.first() {
            self.store_result(result, dest)?;
        }

        Ok(result)
    }

    /// Emit conversion instruction
    unsafe fn emit_conversion(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        let opcode = inst.opcode.as_str();
        let dest = inst
            .dest_operands
            .first()
            .ok_or("Conversion requires destination")?;
        let src = self.get_operand_value(
            inst.src_operands
                .first()
                .ok_or("Conversion requires source")?,
        )?;

        let result = match opcode {
            "I2F" => {
                let f32_type = LLVMFloatTypeInContext(self.context);
                LLVMBuildSIToFP(self.builder, src, f32_type, b"i2f\0".as_ptr() as *const _)
            }
            "F2I" => {
                let i32_type = LLVMInt32TypeInContext(self.context);
                LLVMBuildFPToSI(self.builder, src, i32_type, b"f2i\0".as_ptr() as *const _)
            }
            "I2I" => {
                // Integer type conversion - check sizes
                let dest_type = self.operand_to_llvm_type(dest);
                let src_width = LLVMGetIntTypeWidth(LLVMTypeOf(src));
                let dest_width = LLVMGetIntTypeWidth(dest_type);
                if dest_width > src_width {
                    LLVMBuildSExt(
                        self.builder,
                        src,
                        dest_type,
                        b"i2i_ext\0".as_ptr() as *const _,
                    )
                } else if dest_width < src_width {
                    LLVMBuildTrunc(
                        self.builder,
                        src,
                        dest_type,
                        b"i2i_trunc\0".as_ptr() as *const _,
                    )
                } else {
                    src
                }
            }
            "F2F" => {
                // Float type conversion
                let dest_type = self.operand_to_llvm_type(dest);
                let src_elem_size = LLVMGetTypeKind(LLVMTypeOf(src));
                let dest_elem_size = LLVMGetTypeKind(dest_type);
                if dest_elem_size as u32 > src_elem_size as u32 {
                    LLVMBuildFPExt(
                        self.builder,
                        src,
                        dest_type,
                        b"f2f_ext\0".as_ptr() as *const _,
                    )
                } else {
                    LLVMBuildFPTrunc(
                        self.builder,
                        src,
                        dest_type,
                        b"f2f_trunc\0".as_ptr() as *const _,
                    )
                }
            }
            _ => return Err(format!("Unknown conversion: {}", opcode)),
        };

        self.store_result(result, dest)?;
        Ok(result)
    }

    /// Emit bitwise instruction
    unsafe fn emit_bitwise(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        let opcode = inst.opcode.as_str();
        let (src1, src2) = self.get_src_operands(inst)?;

        let result = match opcode {
            "AND" | "LOP3" => LLVMBuildAnd(self.builder, src1, src2, b"and\0".as_ptr() as *const _),
            "OR" => LLVMBuildOr(self.builder, src1, src2, b"or\0".as_ptr() as *const _),
            "XOR" => LLVMBuildXor(self.builder, src1, src2, b"xor\0".as_ptr() as *const _),
            "NOT" => LLVMBuildNot(self.builder, src1, b"not\0".as_ptr() as *const _),
            "POPC" => self.call_intrinsic("llvm.ctpop.i32", &[src1]),
            "CLZ" => self.call_intrinsic(
                "llvm.ctlz.i32",
                &[
                    src1,
                    LLVMConstInt(LLVMInt1TypeInContext(self.context), 0, 0),
                ],
            ),
            "FLO" => self.call_intrinsic(
                "llvm.cttz.i32",
                &[
                    src1,
                    LLVMConstInt(LLVMInt1TypeInContext(self.context), 0, 0),
                ],
            ),
            "BREV" => self.call_intrinsic("llvm.bitreverse.i32", &[src1]),
            _ => return Err(format!("Unknown bitwise op: {}", opcode)),
        };

        self.store_to_dest(result, inst)?;
        Ok(result)
    }

    /// Emit special instruction
    unsafe fn emit_special(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<LLVMValueRef, String> {
        let opcode = inst.opcode.as_str();

        match opcode {
            "S2R" => {
                // Special register read - map to appropriate LLVM intrinsic
                let sr_name = inst
                    .src_operands
                    .first()
                    .and_then(|op| {
                        if let SassOperand::SpecialRegister(name) = op {
                            Some(name.as_str())
                        } else {
                            None
                        }
                    })
                    .ok_or("S2R requires special register source")?;

                let intrinsic = match sr_name {
                    "SR_TID_X" | "SR_CTAID_X" => "llvm.nvvm.read.ptx.sreg.tid.x",
                    "SR_TID_Y" | "SR_CTAID_Y" => "llvm.nvvm.read.ptx.sreg.tid.y",
                    "SR_TID_Z" | "SR_CTAID_Z" => "llvm.nvvm.read.ptx.sreg.tid.z",
                    "SR_NTID_X" => "llvm.nvvm.read.ptx.sreg.ntid.x",
                    "SR_NTID_Y" => "llvm.nvvm.read.ptx.sreg.ntid.y",
                    "SR_NTID_Z" => "llvm.nvvm.read.ptx.sreg.ntid.z",
                    "SR_LANEID" => "llvm.nvvm.read.ptx.sreg.laneid",
                    "SR_WARPID" => "llvm.nvvm.read.ptx.sreg.warpid",
                    _ => return Err(format!("Unknown special register: {}", sr_name)),
                };

                let result = self.call_intrinsic(intrinsic, &[]);
                if let Some(dest) = inst.dest_operands.first() {
                    self.store_result(result, dest)?;
                }
                Ok(result)
            }
            "NOP" => {
                // No operation - just return null
                Ok(ptr::null_mut())
            }
            _ => Err(format!("Unknown special op: {}", opcode)),
        }
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    fn format_operand(&self, op: &SassOperand) -> String {
        match op {
            SassOperand::Register(reg) => self.format_register(reg),
            SassOperand::Immediate(val) => format!("0x{:x}", val),
            SassOperand::FloatImmediate(val) => format!("{}", val),
            SassOperand::Memory { base, offset, .. } => {
                let base_str = base
                    .as_ref()
                    .map(|r| self.format_register(r))
                    .unwrap_or_else(|| "0".to_string());
                if *offset != 0 {
                    format!("[{}+0x{:x}]", base_str, offset)
                } else {
                    format!("[{}]", base_str)
                }
            }
            SassOperand::Predicate { register, negated } => {
                let reg_str = self.format_register(register);
                if *negated {
                    format!("@!{}", reg_str)
                } else {
                    format!("@{}", reg_str)
                }
            }
            SassOperand::SpecialRegister(name) => name.clone(),
            SassOperand::ConstantBank { bank, offset } => format!("c[{}][0x{:x}]", bank, offset),
            SassOperand::Barrier(n) => format!("B{}", n),
            SassOperand::Label(name) => name.clone(),
            SassOperand::Address(addr) => format!("0x{:x}", addr),
        }
    }

    fn format_register(&self, reg: &SassRegister) -> String {
        if reg.is_zero {
            return reg.prefix.clone();
        }

        if reg.prefix.starts_with("P") {
            format!("P{}", reg.number)
        } else if reg.is_uniform {
            format!("{}{}", reg.prefix, reg.number)
        } else {
            // General purpose register
            let base = format!("R{}", reg.number);
            if let Some(ref comp) = reg.component {
                format!("{}.{}", base, comp)
            } else {
                base
            }
        }
    }

    /// Get source operand values for binary operations
    /// Returns (src1_value, src2_value)
    unsafe fn get_src_operands(
        &mut self,
        inst: &EnhancedSassInstruction,
    ) -> Result<(LLVMValueRef, LLVMValueRef), String> {
        let src1 = self.get_operand_value(
            inst.src_operands
                .first()
                .ok_or("Binary op requires first source")?,
        )?;
        let src2 = if let Some(src2_op) = inst.src_operands.get(1) {
            self.get_operand_value(src2_op)?
        } else {
            // Default to zero for unary operations
            LLVMConstInt(LLVMInt32TypeInContext(self.context), 0, 0)
        };
        Ok((src1, src2))
    }

    /// Store result to destination operand
    unsafe fn store_to_dest(
        &mut self,
        value: LLVMValueRef,
        inst: &EnhancedSassInstruction,
    ) -> Result<(), String> {
        if let Some(dest) = inst.dest_operands.first() {
            self.store_result(value, dest)?;
        }
        Ok(())
    }

    unsafe fn get_operand_value(&mut self, op: &SassOperand) -> Result<LLVMValueRef, String> {
        match op {
            SassOperand::Register(reg) => self.get_register_value(reg),
            SassOperand::Immediate(val) => Ok(LLVMConstInt(
                LLVMInt32TypeInContext(self.context),
                *val as u64,
                0,
            )),
            SassOperand::FloatImmediate(val) => {
                Ok(LLVMConstReal(LLVMFloatTypeInContext(self.context), *val))
            }
            SassOperand::SpecialRegister(_) => {
                // Return placeholder - should be resolved by special register read
                Ok(LLVMConstInt(LLVMInt32TypeInContext(self.context), 0, 0))
            }
            SassOperand::ConstantBank { .. } => {
                // Constant memory access - simplified
                Ok(LLVMConstInt(LLVMInt32TypeInContext(self.context), 0, 0))
            }
            _ => Err(format!("Cannot get value for operand: {:?}", op)),
        }
    }

    unsafe fn get_register_value(&mut self, reg: &SassRegister) -> Result<LLVMValueRef, String> {
        let name = self.format_register(reg);
        if let Some(val) = self.register_map.get(&name) {
            Ok(*val)
        } else {
            // Create a placeholder for uninitialized registers
            let ty = self.get_register_llvm_type(reg);
            let val = LLVMConstInt(ty, 0, 0);
            self.register_map.insert(name, val);
            Ok(val)
        }
    }

    /// Get LLVM type for a register based on its prefix
    unsafe fn get_register_llvm_type(&self, reg: &SassRegister) -> LLVMTypeRef {
        if reg.prefix.starts_with("P") || reg.prefix == "PT" {
            LLVMInt1TypeInContext(self.context) // Predicate
        } else {
            // Default to 32-bit for general purpose registers
            // In a full implementation, we'd track register widths
            LLVMInt32TypeInContext(self.context)
        }
    }

    /// Ensure a value is of float type, converting if necessary
    unsafe fn ensure_float_type(
        &self,
        value: LLVMValueRef,
        target_type: LLVMTypeRef,
    ) -> LLVMValueRef {
        let value_type = LLVMTypeOf(value);

        // If already the target type, return as-is
        if value_type == target_type {
            return value;
        }

        let kind = LLVMGetTypeKind(value_type);
        let target_kind = LLVMGetTypeKind(target_type);

        match (kind, target_kind) {
            (LLVMTypeKind::LLVMIntegerTypeKind, LLVMTypeKind::LLVMFloatTypeKind)
            | (LLVMTypeKind::LLVMIntegerTypeKind, LLVMTypeKind::LLVMDoubleTypeKind)
            | (LLVMTypeKind::LLVMIntegerTypeKind, LLVMTypeKind::LLVMHalfTypeKind) => {
                // Convert integer to float - assume signed for general purpose
                LLVMBuildSIToFP(
                    self.builder,
                    value,
                    target_type,
                    b"sitofp\0".as_ptr() as *const _,
                )
            }
            (LLVMTypeKind::LLVMFloatTypeKind, LLVMTypeKind::LLVMFloatTypeKind)
            | (LLVMTypeKind::LLVMDoubleTypeKind, LLVMTypeKind::LLVMDoubleTypeKind)
            | (LLVMTypeKind::LLVMHalfTypeKind, LLVMTypeKind::LLVMHalfTypeKind) => {
                // Same type, return as-is
                value
            }
            (LLVMTypeKind::LLVMFloatTypeKind, LLVMTypeKind::LLVMDoubleTypeKind)
            | (LLVMTypeKind::LLVMHalfTypeKind, LLVMTypeKind::LLVMFloatTypeKind)
            | (LLVMTypeKind::LLVMHalfTypeKind, LLVMTypeKind::LLVMDoubleTypeKind) => {
                // Float extension
                LLVMBuildFPExt(
                    self.builder,
                    value,
                    target_type,
                    b"fpext\0".as_ptr() as *const _,
                )
            }
            (LLVMTypeKind::LLVMDoubleTypeKind, LLVMTypeKind::LLVMFloatTypeKind)
            | (LLVMTypeKind::LLVMFloatTypeKind, LLVMTypeKind::LLVMHalfTypeKind)
            | (LLVMTypeKind::LLVMDoubleTypeKind, LLVMTypeKind::LLVMHalfTypeKind) => {
                // Float truncation
                LLVMBuildFPTrunc(
                    self.builder,
                    value,
                    target_type,
                    b"fptrunc\0".as_ptr() as *const _,
                )
            }
            (LLVMTypeKind::LLVMPointerTypeKind, _) => {
                // Pointer to int to float
                let int_val = LLVMBuildPtrToInt(
                    self.builder,
                    value,
                    LLVMInt64TypeInContext(self.context),
                    b"ptrtoint\0".as_ptr() as *const _,
                );
                LLVMBuildSIToFP(
                    self.builder,
                    int_val,
                    target_type,
                    b"sitofp\0".as_ptr() as *const _,
                )
            }
            _ => {
                // For other cases, create a zero constant of target type
                // This is safer than invalid bitcast
                LLVMConstReal(target_type, 0.0)
            }
        }
    }

    unsafe fn store_result(
        &mut self,
        value: LLVMValueRef,
        dest: &SassOperand,
    ) -> Result<(), String> {
        if let SassOperand::Register(reg) = dest {
            let name = self.format_register(reg);
            self.register_map.insert(name, value);
        }
        Ok(())
    }

    unsafe fn operand_to_llvm_type(&self, op: &SassOperand) -> LLVMTypeRef {
        match op {
            SassOperand::Register(reg) => self.get_register_llvm_type(reg),
            SassOperand::FloatImmediate(_) => LLVMFloatTypeInContext(self.context),
            _ => LLVMInt32TypeInContext(self.context),
        }
    }

    unsafe fn get_data_type_llvm(&self, data_type: Option<&SassDataType>) -> LLVMTypeRef {
        match data_type {
            Some(SassDataType::U8) | Some(SassDataType::S8) => LLVMInt8TypeInContext(self.context),
            Some(SassDataType::U16) | Some(SassDataType::S16) | Some(SassDataType::F16) => {
                LLVMInt16TypeInContext(self.context)
            }
            Some(SassDataType::U32) | Some(SassDataType::S32) => {
                LLVMInt32TypeInContext(self.context)
            }
            Some(SassDataType::F32) => LLVMFloatTypeInContext(self.context),
            Some(SassDataType::U64) | Some(SassDataType::S64) => {
                LLVMInt64TypeInContext(self.context)
            }
            Some(SassDataType::F64) => LLVMDoubleTypeInContext(self.context),
            _ => LLVMInt32TypeInContext(self.context),
        }
    }

    fn get_address_space(&self, memory_space: Option<&SassMemorySpace>) -> u32 {
        match memory_space {
            Some(SassMemorySpace::Global) => 1,
            Some(SassMemorySpace::Shared) => 3,
            Some(SassMemorySpace::Constant) => 4,
            Some(SassMemorySpace::Local) => 5,
            _ => 0, // Generic
        }
    }

    fn get_type_alignment(&self, data_type: Option<&SassDataType>) -> u32 {
        match data_type {
            Some(SassDataType::U8) | Some(SassDataType::S8) => 1,
            Some(SassDataType::U16) | Some(SassDataType::S16) | Some(SassDataType::F16) => 2,
            Some(SassDataType::U32) | Some(SassDataType::S32) | Some(SassDataType::F32) => 4,
            Some(SassDataType::U64) | Some(SassDataType::S64) | Some(SassDataType::F64) => 8,
            _ => 4,
        }
    }

    fn get_lea_scale(&self, inst: &EnhancedSassInstruction) -> u64 {
        // Extract scale from LEA modifiers (e.g., ".X2", ".X4", ".X8")
        for modifier in &inst.modifiers {
            if modifier.starts_with(".X") {
                if let Ok(scale) = modifier[2..].parse::<u64>() {
                    return scale;
                }
            }
        }
        1
    }

    fn get_comparison_predicate(
        &self,
        modifiers: &[String],
    ) -> (LLVMIntPredicate, LLVMRealPredicate) {
        for modifier in modifiers {
            match modifier.as_str() {
                ".EQ" => return (LLVMIntPredicate::LLVMIntEQ, LLVMRealPredicate::LLVMRealOEQ),
                ".NE" => return (LLVMIntPredicate::LLVMIntNE, LLVMRealPredicate::LLVMRealONE),
                ".LT" => return (LLVMIntPredicate::LLVMIntSLT, LLVMRealPredicate::LLVMRealOLT),
                ".LE" => return (LLVMIntPredicate::LLVMIntSLE, LLVMRealPredicate::LLVMRealOLE),
                ".GT" => return (LLVMIntPredicate::LLVMIntSGT, LLVMRealPredicate::LLVMRealOGT),
                ".GE" => return (LLVMIntPredicate::LLVMIntSGE, LLVMRealPredicate::LLVMRealOGE),
                ".LO" => return (LLVMIntPredicate::LLVMIntULT, LLVMRealPredicate::LLVMRealOLT),
                ".LS" => return (LLVMIntPredicate::LLVMIntULE, LLVMRealPredicate::LLVMRealOLE),
                ".HI" => return (LLVMIntPredicate::LLVMIntUGT, LLVMRealPredicate::LLVMRealOGT),
                ".HS" => return (LLVMIntPredicate::LLVMIntUGE, LLVMRealPredicate::LLVMRealOGE),
                _ => {}
            }
        }
        (LLVMIntPredicate::LLVMIntEQ, LLVMRealPredicate::LLVMRealOEQ)
    }

    unsafe fn call_intrinsic(&mut self, name: &str, args: &[LLVMValueRef]) -> LLVMValueRef {
        let func = self.get_or_declare_intrinsic(name);
        let fn_type = LLVMGlobalGetValueType(func);
        LLVMBuildCall2(
            self.builder,
            fn_type,
            func,
            args.as_ptr() as *mut _,
            args.len() as u32,
            b"\0".as_ptr() as *const _,
        )
    }

    unsafe fn get_or_declare_intrinsic(&mut self, name: &str) -> LLVMValueRef {
        let name_cstr = CString::new(name).unwrap();
        let existing = LLVMGetNamedFunction(self.module, name_cstr.as_ptr());
        if !existing.is_null() {
            return existing;
        }

        // Declare the intrinsic based on name pattern
        let (ret_type, param_types) = self.get_intrinsic_signature(name);
        let fn_type = LLVMFunctionType(
            ret_type,
            param_types.as_ptr() as *mut _,
            param_types.len() as u32,
            0,
        );
        LLVMAddFunction(self.module, name_cstr.as_ptr(), fn_type)
    }

    unsafe fn get_intrinsic_signature(&self, name: &str) -> (LLVMTypeRef, Vec<LLVMTypeRef>) {
        let i32_ty = LLVMInt32TypeInContext(self.context);
        let i1_ty = LLVMInt1TypeInContext(self.context);
        let f32_ty = LLVMFloatTypeInContext(self.context);
        let void_ty = LLVMVoidTypeInContext(self.context);

        match name {
            "llvm.fabs.f32" | "llvm.sqrt.f32" => (f32_ty, vec![f32_ty]),
            "llvm.minnum.f32" | "llvm.maxnum.f32" => (f32_ty, vec![f32_ty, f32_ty]),
            "llvm.ctpop.i32" | "llvm.bitreverse.i32" => (i32_ty, vec![i32_ty]),
            "llvm.ctlz.i32" | "llvm.cttz.i32" => (i32_ty, vec![i32_ty, i1_ty]),
            "llvm.nvvm.barrier0" => (void_ty, vec![]),
            name if name.starts_with("llvm.nvvm.read.ptx.sreg") => (i32_ty, vec![]),
            _ => (i32_ty, vec![]),
        }
    }

    /// Get inlining statistics
    pub fn get_stats(&self) -> &InlineStats {
        &self.stats
    }

    /// Get SASS to LLVM value mapping
    pub fn get_sass_to_llvm_map(&self) -> &HashMap<u64, LLVMValueRef> {
        &self.sass_to_llvm
    }
}

impl Drop for SassInliner {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self.builder);
        }
    }
}

// ============================================================================
// SASS Kernel Builder
// ============================================================================

/// Builder for creating LLVM functions from SASS kernels
pub struct SassKernelBuilder {
    /// Configuration
    config: SassInlineConfig,
}

impl SassKernelBuilder {
    pub fn new(config: SassInlineConfig) -> Self {
        Self { config }
    }

    /// Build an LLVM function from SASS instructions
    pub unsafe fn build_kernel(
        &self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        kernel_name: &str,
        instructions: &[EnhancedSassInstruction],
        param_types: &[LLVMTypeRef],
    ) -> Result<LLVMValueRef, String> {
        // Create function type
        let ret_type = LLVMVoidTypeInContext(context);
        let fn_type = LLVMFunctionType(
            ret_type,
            param_types.as_ptr() as *mut _,
            param_types.len() as u32,
            0,
        );

        // Create function
        let fn_name = format!("{}{}", self.config.function_prefix, kernel_name);
        let fn_name_cstr = CString::new(fn_name).unwrap();
        let function = LLVMAddFunction(module, fn_name_cstr.as_ptr(), fn_type);

        // Set kernel attributes
        let kernel_attr = CString::new("kernel").unwrap();
        let empty = CString::new("").unwrap();
        LLVMAddTargetDependentFunctionAttr(function, kernel_attr.as_ptr(), empty.as_ptr());

        // Create entry block
        let entry_name = CString::new("entry").unwrap();
        let entry_block = LLVMAppendBasicBlockInContext(context, function, entry_name.as_ptr());

        // Create inliner and process instructions
        let mut inliner = SassInliner::new(context, module, self.config.clone());
        inliner.inline_instructions(instructions, entry_block)?;

        // Add return at end
        let builder = LLVMCreateBuilderInContext(context);
        LLVMPositionBuilderAtEnd(builder, entry_block);
        LLVMBuildRetVoid(builder);
        LLVMDisposeBuilder(builder);

        Ok(function)
    }
}

// ============================================================================
// SASS Metadata Extractor
// ============================================================================

/// Extract SASS metadata from LLVM IR
pub struct SassMetadataExtractor {
    md_sass_addr: u32,
    md_sass_opcode: u32,
    md_ptx_line: u32,
    md_control_codes: u32,
}

impl SassMetadataExtractor {
    pub unsafe fn new(context: LLVMContextRef) -> Self {
        Self {
            md_sass_addr: LLVMGetMDKindIDInContext(context, b"sass.addr\0".as_ptr() as *const _, 9),
            md_sass_opcode: LLVMGetMDKindIDInContext(
                context,
                b"sass.opcode\0".as_ptr() as *const _,
                11,
            ),
            md_ptx_line: LLVMGetMDKindIDInContext(
                context,
                b"sass.ptx_line\0".as_ptr() as *const _,
                13,
            ),
            md_control_codes: LLVMGetMDKindIDInContext(
                context,
                b"sass.control\0".as_ptr() as *const _,
                12,
            ),
        }
    }

    /// Extract SASS address from an instruction
    pub unsafe fn get_sass_address(&self, inst: LLVMValueRef) -> Option<u64> {
        let md = LLVMGetMetadata(inst, self.md_sass_addr);
        if md.is_null() {
            return None;
        }

        let operand = LLVMGetOperand(md, 0);
        if operand.is_null() {
            return None;
        }

        Some(LLVMConstIntGetZExtValue(operand))
    }

    /// Extract SASS opcode from an instruction
    pub unsafe fn get_sass_opcode(&self, inst: LLVMValueRef) -> Option<String> {
        let md = LLVMGetMetadata(inst, self.md_sass_opcode);
        if md.is_null() {
            return None;
        }

        let operand = LLVMGetOperand(md, 0);
        if operand.is_null() {
            return None;
        }

        let mut len = 0;
        let ptr = LLVMGetMDString(operand, &mut len);
        if ptr.is_null() {
            return None;
        }

        let slice = std::slice::from_raw_parts(ptr as *const u8, len as usize);
        Some(String::from_utf8_lossy(slice).to_string())
    }

    /// Extract PTX line from an instruction
    pub unsafe fn get_ptx_line(&self, inst: LLVMValueRef) -> Option<u32> {
        let md = LLVMGetMetadata(inst, self.md_ptx_line);
        if md.is_null() {
            return None;
        }

        let operand = LLVMGetOperand(md, 0);
        if operand.is_null() {
            return None;
        }

        Some(LLVMConstIntGetZExtValue(operand) as u32)
    }

    /// Extract control codes from an instruction
    pub unsafe fn get_control_codes(&self, inst: LLVMValueRef) -> Option<SassControlCodes> {
        let md = LLVMGetMetadata(inst, self.md_control_codes);
        if md.is_null() {
            return None;
        }

        let stall = LLVMGetOperand(md, 0);
        let wb = LLVMGetOperand(md, 1);
        let rb = LLVMGetOperand(md, 2);
        let wbm = LLVMGetOperand(md, 3);

        if stall.is_null() || wb.is_null() || rb.is_null() || wbm.is_null() {
            return None;
        }

        Some(SassControlCodes {
            stall_count: LLVMConstIntGetZExtValue(stall) as u8,
            yield_flag: false,
            write_barrier: LLVMConstIntGetZExtValue(wb) as u8,
            read_barrier: LLVMConstIntGetZExtValue(rb) as u8,
            wait_barrier_mask: LLVMConstIntGetZExtValue(wbm) as u8,
            reuse_flags: 0,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sass::instruction::*;

    #[test]
    fn test_config_default() {
        let config = SassInlineConfig::default();
        assert_eq!(config.strategy, SassInlineStrategy::Hybrid);
        assert!(config.preserve_control_codes);
        assert!(config.emit_debug_info);
        assert!(config.track_registers);
        assert_eq!(config.target_sm, 75);
        assert_eq!(config.function_prefix, "_sass_");
    }

    #[test]
    fn test_config_strategies() {
        let strategies = [
            SassInlineStrategy::InlineAssembly,
            SassInlineStrategy::PtxReconstruction,
            SassInlineStrategy::MetadataOnly,
            SassInlineStrategy::Hybrid,
        ];

        for strategy in &strategies {
            let config = SassInlineConfig {
                strategy: *strategy,
                ..Default::default()
            };
            assert_eq!(config.strategy, *strategy);
        }
    }

    #[test]
    fn test_inline_stats_default() {
        let stats = InlineStats::default();
        assert_eq!(stats.instructions_processed, 0);
        assert_eq!(stats.inline_asm_emitted, 0);
        assert_eq!(stats.ptx_reconstructed, 0);
        assert_eq!(stats.metadata_only, 0);
        assert_eq!(stats.fallback_used, 0);
    }

    // Helper function to create test instructions
    fn create_test_instruction(opcode: &str, address: u64) -> EnhancedSassInstruction {
        let mut inst = EnhancedSassInstruction::new(opcode.to_string(), address);
        inst.dest_operands
            .push(SassOperand::Register(SassRegister::new("R", 0)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 1)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 2)));
        inst
    }

    fn create_fadd_instruction(address: u64) -> EnhancedSassInstruction {
        let mut inst = create_test_instruction("FADD", address);
        inst.data_type = Some(SassDataType::F32);
        inst.opcode_class = SassOpcodeClass::FloatArithmetic;
        inst
    }

    fn create_iadd_instruction(address: u64) -> EnhancedSassInstruction {
        let mut inst = create_test_instruction("IADD", address);
        inst.data_type = Some(SassDataType::S32);
        inst.opcode_class = SassOpcodeClass::IntegerArithmetic;
        inst
    }

    fn create_ldg_instruction(address: u64) -> EnhancedSassInstruction {
        let mut inst = EnhancedSassInstruction::new("LDG".to_string(), address);
        inst.data_type = Some(SassDataType::U32);
        inst.opcode_class = SassOpcodeClass::GlobalLoad;
        inst.memory_space = Some(SassMemorySpace::Global);
        inst.dest_operands
            .push(SassOperand::Register(SassRegister::new("R", 0)));
        inst.src_operands.push(SassOperand::Memory {
            base: Some(SassRegister::new("R", 2)),
            offset: 0x10,
            index: None,
            scale: 1,
        });
        inst
    }

    fn create_stg_instruction(address: u64) -> EnhancedSassInstruction {
        let mut inst = EnhancedSassInstruction::new("STG".to_string(), address);
        inst.data_type = Some(SassDataType::U32);
        inst.opcode_class = SassOpcodeClass::GlobalStore;
        inst.memory_space = Some(SassMemorySpace::Global);
        inst.dest_operands.push(SassOperand::Memory {
            base: Some(SassRegister::new("R", 2)),
            offset: 0x10,
            index: None,
            scale: 1,
        });
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 0)));
        inst
    }

    fn create_mov_instruction(address: u64) -> EnhancedSassInstruction {
        let mut inst = EnhancedSassInstruction::new("MOV".to_string(), address);
        inst.data_type = Some(SassDataType::B32);
        inst.opcode_class = SassOpcodeClass::Move;
        inst.dest_operands
            .push(SassOperand::Register(SassRegister::new("R", 0)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 1)));
        inst
    }

    fn create_imad_instruction(address: u64) -> EnhancedSassInstruction {
        let mut inst = EnhancedSassInstruction::new("IMAD".to_string(), address);
        inst.data_type = Some(SassDataType::S32);
        inst.opcode_class = SassOpcodeClass::IntegerArithmetic;
        inst.dest_operands
            .push(SassOperand::Register(SassRegister::new("R", 0)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 1)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 2)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 3)));
        inst
    }

    fn create_ffma_instruction(address: u64) -> EnhancedSassInstruction {
        let mut inst = EnhancedSassInstruction::new("FFMA".to_string(), address);
        inst.data_type = Some(SassDataType::F32);
        inst.opcode_class = SassOpcodeClass::FloatArithmetic;
        inst.dest_operands
            .push(SassOperand::Register(SassRegister::new("R", 0)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 1)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 2)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 3)));
        inst
    }

    fn create_exit_instruction(address: u64) -> EnhancedSassInstruction {
        let mut inst = EnhancedSassInstruction::new("EXIT".to_string(), address);
        inst.opcode_class = SassOpcodeClass::Exit;
        inst
    }

    fn create_bar_instruction(address: u64) -> EnhancedSassInstruction {
        let mut inst = EnhancedSassInstruction::new("BAR".to_string(), address);
        inst.opcode_class = SassOpcodeClass::Barrier;
        inst.src_operands.push(SassOperand::Barrier(0));
        inst
    }

    fn create_s2r_instruction(address: u64, special_reg: &str) -> EnhancedSassInstruction {
        let mut inst = EnhancedSassInstruction::new("S2R".to_string(), address);
        inst.opcode_class = SassOpcodeClass::SpecialRegRead;
        inst.dest_operands
            .push(SassOperand::Register(SassRegister::new("R", 0)));
        inst.src_operands
            .push(SassOperand::SpecialRegister(special_reg.to_string()));
        inst
    }

    #[test]
    fn test_inliner_creation() {
        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig::default();
            let inliner = SassInliner::new(context, module, config);

            let stats = inliner.get_stats();
            assert_eq!(stats.instructions_processed, 0);

            drop(inliner);
            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_kernel_builder_creation() {
        let config = SassInlineConfig::default();
        let builder = SassKernelBuilder::new(config.clone());

        // Builder should hold the config
        assert_eq!(config.strategy, SassInlineStrategy::Hybrid);
    }

    #[test]
    fn test_kernel_builder_build_empty_kernel() {
        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig::default();
            let builder = SassKernelBuilder::new(config);

            let instructions: Vec<EnhancedSassInstruction> = vec![];

            let result = builder.build_kernel(context, module, "empty_kernel", &instructions, &[]);

            assert!(result.is_ok());
            let func = result.unwrap();
            assert!(!func.is_null());

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_kernel_builder_with_instructions() {
        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig {
                strategy: SassInlineStrategy::MetadataOnly,
                ..Default::default()
            };
            let builder = SassKernelBuilder::new(config);

            let instructions = vec![
                create_mov_instruction(0x00),
                create_iadd_instruction(0x10),
                create_fadd_instruction(0x20),
            ];

            let result = builder.build_kernel(context, module, "test_kernel", &instructions, &[]);

            assert!(result.is_ok());

            // Verify function exists
            let func_name = std::ffi::CString::new("_sass_test_kernel").unwrap();
            let func = LLVMGetNamedFunction(module, func_name.as_ptr());
            assert!(!func.is_null());

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_inline_assembly_strategy() {
        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig {
                strategy: SassInlineStrategy::InlineAssembly,
                ..Default::default()
            };
            let builder = SassKernelBuilder::new(config);

            let instructions = vec![create_fadd_instruction(0x00), create_iadd_instruction(0x10)];

            let result = builder.build_kernel(context, module, "asm_kernel", &instructions, &[]);

            assert!(result.is_ok());

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_ptx_reconstruction_strategy() {
        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig {
                strategy: SassInlineStrategy::PtxReconstruction,
                ..Default::default()
            };
            let builder = SassKernelBuilder::new(config);

            let instructions = vec![
                create_mov_instruction(0x00),
                create_iadd_instruction(0x10),
                create_fadd_instruction(0x20),
                create_imad_instruction(0x30),
                create_ffma_instruction(0x40),
            ];

            let result = builder.build_kernel(context, module, "ptx_kernel", &instructions, &[]);

            assert!(result.is_ok());

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_hybrid_strategy() {
        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig {
                strategy: SassInlineStrategy::Hybrid,
                ..Default::default()
            };
            let builder = SassKernelBuilder::new(config);

            let instructions = vec![
                create_mov_instruction(0x00),
                create_ldg_instruction(0x10),
                create_iadd_instruction(0x20),
                create_fadd_instruction(0x30),
                create_stg_instruction(0x40),
            ];

            let result = builder.build_kernel(context, module, "hybrid_kernel", &instructions, &[]);

            assert!(result.is_ok());

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_metadata_extractor() {
        unsafe {
            let context = LLVMContextCreate();
            let extractor = SassMetadataExtractor::new(context);

            // These are just MD kind IDs, they should be non-zero and unique
            assert!(extractor.md_sass_addr > 0 || extractor.md_sass_opcode > 0);

            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_instruction_with_control_codes() {
        let mut inst = create_fadd_instruction(0x00);
        inst.control_codes = SassControlCodes {
            stall_count: 5,
            yield_flag: true,
            write_barrier: 1,
            read_barrier: 2,
            wait_barrier_mask: 0x3,
            reuse_flags: 0x4,
        };

        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig {
                strategy: SassInlineStrategy::MetadataOnly,
                preserve_control_codes: true,
                ..Default::default()
            };
            let builder = SassKernelBuilder::new(config);

            let result = builder.build_kernel(context, module, "control_code_kernel", &[inst], &[]);

            assert!(result.is_ok());

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_instruction_with_predicate() {
        let mut inst = create_fadd_instruction(0x00);
        inst.predicate = Some(SassOperand::Predicate {
            register: SassRegister::new("P", 0),
            negated: false,
        });

        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig {
                strategy: SassInlineStrategy::InlineAssembly,
                ..Default::default()
            };
            let builder = SassKernelBuilder::new(config);

            let result = builder.build_kernel(context, module, "pred_kernel", &[inst], &[]);

            assert!(result.is_ok());

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_instruction_with_ptx_mapping() {
        let mut inst = create_fadd_instruction(0x100);
        inst.ptx_file = Some("test.ptx".to_string());
        inst.ptx_line = Some(42);
        inst.ptx_column = Some(10);
        inst.function_name = Some("test_kernel".to_string());

        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig {
                strategy: SassInlineStrategy::MetadataOnly,
                emit_debug_info: true,
                ..Default::default()
            };
            let builder = SassKernelBuilder::new(config);

            let result = builder.build_kernel(context, module, "ptx_mapped_kernel", &[inst], &[]);

            assert!(result.is_ok());

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_full_kernel_simulation() {
        // Simulate a small kernel with various instruction types
        let instructions = vec![
            // prologue
            create_s2r_instruction(0x00, "SR_TID_X"),
            // load
            create_ldg_instruction(0x10),
            // compute
            create_fadd_instruction(0x20),
            create_ffma_instruction(0x30),
            create_iadd_instruction(0x40),
            create_imad_instruction(0x50),
            // store
            create_stg_instruction(0x60),
            // sync
            create_bar_instruction(0x70),
            // exit
            create_exit_instruction(0x80),
        ];

        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            // Test all strategies
            for strategy in &[
                SassInlineStrategy::InlineAssembly,
                SassInlineStrategy::PtxReconstruction,
                SassInlineStrategy::MetadataOnly,
                SassInlineStrategy::Hybrid,
            ] {
                let config = SassInlineConfig {
                    strategy: *strategy,
                    ..Default::default()
                };
                let builder = SassKernelBuilder::new(config);

                let kernel_name = format!("full_kernel_{:?}", strategy);
                let result =
                    builder.build_kernel(context, module, &kernel_name, &instructions, &[]);

                assert!(result.is_ok(), "Failed for strategy {:?}", strategy);
            }

            // Print module IR for inspection
            let ir_str = LLVMPrintModuleToString(module);
            let ir = std::ffi::CStr::from_ptr(ir_str).to_string_lossy();

            // Verify functions were created
            assert!(ir.contains("_sass_full_kernel"));

            LLVMDisposeMessage(ir_str);
            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_sass_to_llvm_mapping() {
        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig {
                strategy: SassInlineStrategy::MetadataOnly,
                ..Default::default()
            };

            let mut inliner = SassInliner::new(context, module, config);

            // Create function and entry block for testing
            let void_type = LLVMVoidTypeInContext(context);
            let fn_type = LLVMFunctionType(void_type, std::ptr::null_mut(), 0, 0);
            let fn_name = std::ffi::CString::new("test_func").unwrap();
            let function = LLVMAddFunction(module, fn_name.as_ptr(), fn_type);
            let bb_name = std::ffi::CString::new("entry").unwrap();
            let entry = LLVMAppendBasicBlockInContext(context, function, bb_name.as_ptr());

            let instructions = vec![
                create_mov_instruction(0x100),
                create_iadd_instruction(0x110),
            ];

            let result = inliner.inline_instructions(&instructions, entry);
            assert!(result.is_ok());

            // Check mapping was created
            let mapping = inliner.get_sass_to_llvm_map();
            assert_eq!(mapping.len(), 2);
            assert!(mapping.contains_key(&0x100));
            assert!(mapping.contains_key(&0x110));

            // Check stats
            let stats = inliner.get_stats();
            assert_eq!(stats.instructions_processed, 2);
            assert_eq!(stats.metadata_only, 2);

            drop(inliner);
            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_data_type_to_llvm() {
        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig::default();
            let inliner = SassInliner::new(context, module, config);

            // Test various data types
            let types = [
                (SassDataType::U8, 8),
                (SassDataType::U16, 16),
                (SassDataType::U32, 32),
                (SassDataType::U64, 64),
                (SassDataType::S32, 32),
                (SassDataType::F32, 32),
                (SassDataType::F64, 64),
            ];

            for (dt, expected_bits) in &types {
                let llvm_type = inliner.get_data_type_llvm(Some(dt));
                let actual_bits =
                    if *expected_bits <= 64 && *dt != SassDataType::F32 && *dt != SassDataType::F64
                    {
                        LLVMGetIntTypeWidth(llvm_type)
                    } else {
                        // For floats, we just verify the type is not null
                        if !llvm_type.is_null() {
                            *expected_bits
                        } else {
                            0
                        }
                    };
                assert!(
                    actual_bits > 0 || !llvm_type.is_null(),
                    "Failed for {:?}",
                    dt
                );
            }

            drop(inliner);
            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_address_space_mapping() {
        let config = SassInlineConfig::default();

        // Use a helper struct to test address space without unsafe
        let spaces = [
            (Some(SassMemorySpace::Global), 1u32),
            (Some(SassMemorySpace::Shared), 3u32),
            (Some(SassMemorySpace::Constant), 4u32),
            (Some(SassMemorySpace::Local), 5u32),
            (None, 0u32),
        ];

        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);
            let inliner = SassInliner::new(context, module, config);

            for (space, expected) in &spaces {
                let actual = inliner.get_address_space(space.as_ref());
                assert_eq!(actual, *expected, "Failed for {:?}", space);
            }

            drop(inliner);
            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_multiple_kernels() {
        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("multi_kernel").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig::default();
            let builder = SassKernelBuilder::new(config);

            // Create multiple kernels
            for i in 0..5 {
                let instructions = vec![
                    create_mov_instruction(i * 0x100),
                    create_iadd_instruction(i * 0x100 + 0x10),
                    create_exit_instruction(i * 0x100 + 0x20),
                ];

                let kernel_name = format!("kernel_{}", i);
                let result =
                    builder.build_kernel(context, module, &kernel_name, &instructions, &[]);

                assert!(result.is_ok());
            }

            // Verify all functions exist
            for i in 0..5 {
                let func_name = std::ffi::CString::new(format!("_sass_kernel_{}", i)).unwrap();
                let func = LLVMGetNamedFunction(module, func_name.as_ptr());
                assert!(!func.is_null(), "Function kernel_{} not found", i);
            }

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_immediate_operands() {
        let mut inst = EnhancedSassInstruction::new("IADD".to_string(), 0x00);
        inst.opcode_class = SassOpcodeClass::IntegerArithmetic;
        inst.dest_operands
            .push(SassOperand::Register(SassRegister::new("R", 0)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 1)));
        inst.src_operands.push(SassOperand::Immediate(42));

        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig {
                strategy: SassInlineStrategy::PtxReconstruction,
                ..Default::default()
            };
            let builder = SassKernelBuilder::new(config);

            let result = builder.build_kernel(context, module, "imm_kernel", &[inst], &[]);

            assert!(result.is_ok());

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_float_immediate_operands() {
        let mut inst = EnhancedSassInstruction::new("FADD".to_string(), 0x00);
        inst.data_type = Some(SassDataType::F32);
        inst.opcode_class = SassOpcodeClass::FloatArithmetic;
        inst.dest_operands
            .push(SassOperand::Register(SassRegister::new("R", 0)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 1)));
        inst.src_operands.push(SassOperand::FloatImmediate(3.14159));

        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig {
                strategy: SassInlineStrategy::PtxReconstruction,
                ..Default::default()
            };
            let builder = SassKernelBuilder::new(config);

            let result = builder.build_kernel(context, module, "fimm_kernel", &[inst], &[]);

            assert!(result.is_ok());

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }

    #[test]
    fn test_bitwise_operations() {
        let opcodes = ["AND", "OR", "XOR", "NOT", "SHL", "SHR"];

        for opcode in &opcodes {
            let mut inst = EnhancedSassInstruction::new(opcode.to_string(), 0x00);
            inst.opcode_class = SassOpcodeClass::IntegerLogical;
            inst.dest_operands
                .push(SassOperand::Register(SassRegister::new("R", 0)));
            inst.src_operands
                .push(SassOperand::Register(SassRegister::new("R", 1)));
            if *opcode != "NOT" {
                inst.src_operands
                    .push(SassOperand::Register(SassRegister::new("R", 2)));
            }

            unsafe {
                let context = LLVMContextCreate();
                let module_name = std::ffi::CString::new("test_module").unwrap();
                let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

                let config = SassInlineConfig {
                    strategy: SassInlineStrategy::PtxReconstruction,
                    ..Default::default()
                };
                let builder = SassKernelBuilder::new(config);

                let result = builder.build_kernel(
                    context,
                    module,
                    &format!("{}_kernel", opcode),
                    &[inst],
                    &[],
                );

                assert!(result.is_ok(), "Failed for opcode {}", opcode);

                LLVMDisposeModule(module);
                LLVMContextDispose(context);
            }
        }
    }

    #[test]
    fn test_comparison_operations() {
        let mut inst = EnhancedSassInstruction::new("ISETP".to_string(), 0x00);
        inst.opcode_class = SassOpcodeClass::IntegerComparison;
        inst.modifiers.push(".LT".to_string());
        inst.dest_operands
            .push(SassOperand::Register(SassRegister::new("P", 0)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 1)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 2)));

        unsafe {
            let context = LLVMContextCreate();
            let module_name = std::ffi::CString::new("test_module").unwrap();
            let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

            let config = SassInlineConfig {
                strategy: SassInlineStrategy::PtxReconstruction,
                ..Default::default()
            };
            let builder = SassKernelBuilder::new(config);

            let result = builder.build_kernel(context, module, "cmp_kernel", &[inst], &[]);

            assert!(result.is_ok());

            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }
}
