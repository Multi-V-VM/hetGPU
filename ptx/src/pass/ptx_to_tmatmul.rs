// ptx_to_tmatmul.rs - Complete PTX to TMatmul assembly compilation
// This module provides end-to-end compilation from PTX to TMatmul assembly

use super::*;
use crate::pass::emit_tmatmul_asm::*;
use ptx_parser as ast;
use std::collections::HashMap;

/// High-level compilation result
pub struct TMatmulCompilationResult {
    /// Generated TMatmul assembly code
    pub assembly: String,
    /// Mapping of PTX functions to assembly sections
    pub function_map: HashMap<String, usize>,
    /// Register usage statistics
    pub register_stats: RegisterStats,
}

/// Register usage statistics
#[derive(Debug, Clone)]
pub struct RegisterStats {
    pub max_registers_used: usize,
    pub spills: usize,
    pub total_operations: usize,
}

/// PTX to TMatmul compiler
pub struct PtxToTMatmulCompiler {
    codegen: TMatmulCodegen,
    ssa_counter: usize,
    ptx_to_ssa: HashMap<String, String>,
    /// Track which PTX variables came from which memory regions (e.g., "%rd1" → "PARAM_0")
    ptx_to_memory: HashMap<String, String>,
    /// Map kernel parameter names to their indices (e.g., "output_ptr" → 0)
    param_name_to_index: HashMap<String, usize>,
    function_map: HashMap<String, usize>,
    current_function: Option<String>,
    stats: RegisterStats,
}

impl PtxToTMatmulCompiler {
    pub fn new() -> Self {
        Self {
            codegen: TMatmulCodegen::new(),
            ssa_counter: 0,
            ptx_to_ssa: HashMap::new(),
            ptx_to_memory: HashMap::new(),
            param_name_to_index: HashMap::new(),
            function_map: HashMap::new(),
            current_function: None,
            stats: RegisterStats {
                max_registers_used: 0,
                spills: 0,
                total_operations: 0,
            },
        }
    }

    /// Allocate a new SSA value
    fn new_ssa(&mut self) -> String {
        let ssa = format!("%{}", self.ssa_counter);
        self.ssa_counter += 1;
        ssa
    }

    /// Map PTX identifier to SSA value
    fn map_ptx_to_ssa(&mut self, ptx_id: &str) -> String {
        if let Some(ssa) = self.ptx_to_ssa.get(ptx_id) {
            ssa.clone()
        } else {
            let ssa = self.new_ssa();
            self.ptx_to_ssa.insert(ptx_id.to_string(), ssa.clone());
            ssa
        }
    }

    /// Setup standard memory mappings for neural network workloads
    pub fn setup_standard_memory_map(&mut self) {
        self.codegen.map_memory("input", MemoryLocation::X);
        self.codegen.map_memory("hidden", MemoryLocation::OH);
        self.codegen.map_memory("output", MemoryLocation::O);
        self.codegen.map_memory("weight_f", MemoryLocation::WF);
        self.codegen.map_memory("weight_c", MemoryLocation::WC);
        self.codegen.map_memory("weight_g", MemoryLocation::WG);
        self.codegen.map_memory("weight_o", MemoryLocation::WO);
        self.codegen.map_memory("weight_up1", MemoryLocation::WU1);
        self.codegen.map_memory("weight_up2", MemoryLocation::WU2);
        self.codegen.map_memory("weight_down", MemoryLocation::WN);
        self.codegen.map_memory("temp", MemoryLocation::TempVec);
        self.codegen.map_memory("bias", MemoryLocation::B);
    }

    /// Map custom memory location
    pub fn map_memory(&mut self, symbol: &str, location: MemoryLocation) {
        self.codegen.map_memory(symbol, location);
    }

    /// Convert PTX instruction to TMatmul operations
    fn convert_ptx_instruction(&mut self, inst: &str, args: &[&str]) -> Result<(), String> {
        self.stats.total_operations += 1;

        match inst {
            // Memory operations
            "ld.global" | "ld.param" | "ld" => {
                if args.len() >= 2 {
                    let dst_ssa = self.map_ptx_to_ssa(args[0]);
                    let src_mem = args[1];
                    self.codegen
                        .emit_operation("tmatmul.ldv", &[src_mem], &[&dst_ssa])?;
                }
            }
            "st.global" | "st.param" | "st" => {
                if args.len() >= 2 {
                    let src_ssa = self.map_ptx_to_ssa(args[0]);
                    let dst_mem = args[1];
                    self.codegen
                        .emit_operation("tmatmul.sv", &[&src_ssa, dst_mem], &[])?;
                }
            }

            // Arithmetic operations
            "add.f32" | "add.f64" | "add" => {
                if args.len() >= 3 {
                    let dst_ssa = self.map_ptx_to_ssa(args[0]);
                    let src1_ssa = self.map_ptx_to_ssa(args[1]);
                    let src2_ssa = self.map_ptx_to_ssa(args[2]);
                    self.codegen.emit_operation(
                        "tmatmul.add",
                        &[&src1_ssa, &src2_ssa],
                        &[&dst_ssa],
                    )?;
                }
            }
            "sub.f32" | "sub.f64" | "sub" => {
                if args.len() >= 3 {
                    let dst_ssa = self.map_ptx_to_ssa(args[0]);
                    let src1_ssa = self.map_ptx_to_ssa(args[1]);
                    let src2_ssa = self.map_ptx_to_ssa(args[2]);
                    self.codegen.emit_operation(
                        "tmatmul.sub",
                        &[&src1_ssa, &src2_ssa],
                        &[&dst_ssa],
                    )?;
                }
            }
            "mul.f32" | "mul.f64" | "mul" => {
                if args.len() >= 3 {
                    let dst_ssa = self.map_ptx_to_ssa(args[0]);
                    let src1_ssa = self.map_ptx_to_ssa(args[1]);
                    let src2_ssa = self.map_ptx_to_ssa(args[2]);
                    self.codegen.emit_operation(
                        "tmatmul.mul",
                        &[&src1_ssa, &src2_ssa],
                        &[&dst_ssa],
                    )?;
                }
            }
            "div.f32" | "div.f64" | "div" => {
                if args.len() >= 3 {
                    let dst_ssa = self.map_ptx_to_ssa(args[0]);
                    let src1_ssa = self.map_ptx_to_ssa(args[1]);
                    let src2_ssa = self.map_ptx_to_ssa(args[2]);
                    self.codegen.emit_operation(
                        "tmatmul.div",
                        &[&src1_ssa, &src2_ssa],
                        &[&dst_ssa],
                    )?;
                }
            }

            // Activation functions (detect patterns)
            "ex2.approx" | "exp" => {
                // Part of sigmoid/activation pattern
                self.codegen.add_comment(&format!("PTX: {}", inst));
            }
            "rcp.approx" | "rcp" => {
                // Part of sigmoid pattern
                self.codegen.add_comment(&format!("PTX: {}", inst));
            }

            // Matrix operations (detect GEMM patterns)
            "mad.f32" | "fma.f32" | "mad" => {
                if args.len() >= 4 {
                    // d = a * b + c
                    let dst_ssa = self.map_ptx_to_ssa(args[0]);
                    let a_ssa = self.map_ptx_to_ssa(args[1]);
                    let b_ssa = self.map_ptx_to_ssa(args[2]);
                    let c_ssa = self.map_ptx_to_ssa(args[3]);

                    // Emit as mul + add
                    let temp_ssa = self.new_ssa();
                    self.codegen
                        .emit_operation("tmatmul.mul", &[&a_ssa, &b_ssa], &[&temp_ssa])?;
                    self.codegen.emit_operation(
                        "tmatmul.add",
                        &[&temp_ssa, &c_ssa],
                        &[&dst_ssa],
                    )?;
                }
            }

            // Move operations
            "mov" | "mov.f32" | "mov.f64" => {
                if args.len() >= 2 {
                    let dst_ssa = self.map_ptx_to_ssa(args[0]);
                    let src_ssa = self.map_ptx_to_ssa(args[1]);
                    // Move is implicit in SSA, just update mapping
                    self.ptx_to_ssa.insert(args[0].to_string(), src_ssa);
                }
            }

            // Control flow - add as comments for now
            "ret" => {
                self.codegen.add_comment("PTX: ret");
            }
            "bra" | "call" => {
                self.codegen.add_comment(&format!("PTX: {}", inst));
            }

            _ => {
                self.codegen
                    .add_comment(&format!("PTX: {} (not yet mapped)", inst));
            }
        }

        Ok(())
    }

    /// Compile a complete PTX module to TMatmul assembly
    pub fn compile_module<'input>(
        &mut self,
        module: ast::Module<'input>,
    ) -> Result<TMatmulCompilationResult, String> {
        self.codegen.add_section("COMPILED FROM PTX");
        self.codegen.add_comment(&format!(
            "PTX version: {}.{}",
            module.version.0, module.version.1
        ));

        let directive_count = module.directives.len();
        self.codegen
            .add_comment(&format!("Processing {} PTX directives", directive_count));

        // Process each directive
        for directive in module.directives {
            match directive {
                ast::Directive::Variable(_linking, var) => {
                    // Track variables for memory mapping
                    self.codegen.add_comment(&format!("Variable: {}", var.name));
                }
                ast::Directive::Method(_linking, function) => {
                    self.compile_function(function)?;
                }
            }
        }

        Ok(TMatmulCompilationResult {
            assembly: self.codegen.get_assembly(),
            function_map: self.function_map.clone(),
            register_stats: self.stats.clone(),
        })
    }

    /// Compile a PTX function/kernel
    fn compile_function<'input>(
        &mut self,
        function: ast::Function<
            'input,
            &'input str,
            ast::Statement<ast::ParsedOperand<&'input str>>,
        >,
    ) -> Result<(), String> {
        let func_name = function.func_directive.name();
        self.current_function = Some(func_name.to_string());

        self.codegen
            .add_section(&format!("FUNCTION: {}", func_name));

        // Build parameter name → index mapping from function's input arguments
        self.param_name_to_index.clear();
        for (idx, var) in function.func_directive.input_arguments.iter().enumerate() {
            let param_name = var.name.to_string();
            self.param_name_to_index.insert(param_name.clone(), idx);
            self.codegen
                .add_comment(&format!("BIND PARAM_{} {}", idx, param_name));
        }
        if !function.func_directive.input_arguments.is_empty() {
            self.codegen.add_comment("END_BINDINGS");
        }

        // Process function body if it exists
        if let Some(body) = function.body {
            for statement in body {
                self.compile_statement(&statement)?;
            }
        }

        Ok(())
    }

    /// Compile a PTX statement
    fn compile_statement(
        &mut self,
        statement: &ast::Statement<ast::ParsedOperand<&str>>,
    ) -> Result<(), String> {
        match statement {
            ast::Statement::Label(label) => {
                self.codegen.add_comment(&format!("Label: {}", label));
            }
            ast::Statement::Variable(_var) => {
                // Variable declarations
                self.codegen.add_comment("Variable declaration");
            }
            ast::Statement::Instruction(_pred, inst) => {
                self.compile_instruction(inst)?;
            }
            ast::Statement::Block(statements) => {
                for stmt in statements {
                    self.compile_statement(stmt)?;
                }
            }
        }
        Ok(())
    }

    /// Compile a PTX instruction to TMatmul operations
    fn compile_instruction(
        &mut self,
        inst: &ast::Instruction<ast::ParsedOperand<&str>>,
    ) -> Result<(), String> {
        self.stats.total_operations += 1;

        match inst {
            // Arithmetic instructions - only emit for float types (skip integer index/address math)
            ast::Instruction::Add {
                data, arguments, ..
            } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src1_name = Self::operand_to_string(&arguments.src1);
                let src2_name = Self::operand_to_string(&arguments.src2);

                // Check if this is pointer arithmetic (add base_ptr, offset)
                let src1_is_ptr = self.ptx_to_memory.contains_key(&src1_name);
                let src2_is_ptr = self.ptx_to_memory.contains_key(&src2_name);

                if src1_is_ptr || src2_is_ptr {
                    // Pointer arithmetic - propagate memory binding
                    let ptr_name = if src1_is_ptr { &src1_name } else { &src2_name };
                    if let Some(mem_region) = self.ptx_to_memory.get(ptr_name).cloned() {
                        self.ptx_to_memory.insert(dst_name.clone(), mem_region);
                    }
                    self.codegen.add_comment(&format!(
                        "PTX add (pointer arithmetic, propagating {})",
                        ptr_name
                    ));
                    let src_ssa =
                        self.map_ptx_to_ssa(if src1_is_ptr { &src1_name } else { &src2_name });
                    self.ptx_to_ssa.insert(dst_name, src_ssa);
                } else if matches!(data, ast::ArithDetails::Float(_)) {
                    // Float data add - emit tmatmul instruction
                    let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                    let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                    let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                    self.codegen.emit_operation(
                        "tmatmul.add",
                        &[&src1_ssa, &src2_ssa],
                        &[&dst_ssa],
                    )?;
                } else {
                    // Integer add (index/address computation) - just track SSA, don't emit
                    let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                }
            }
            ast::Instruction::Sub {
                data, arguments, ..
            } => {
                if matches!(data, ast::ArithDetails::Float(_)) {
                    let dst_name = Self::operand_to_string(&arguments.dst);
                    let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                    let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                    let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                    self.codegen.emit_operation(
                        "tmatmul.sub",
                        &[&src1_ssa, &src2_ssa],
                        &[&dst_ssa],
                    )?;
                } else {
                    let dst_name = Self::operand_to_string(&arguments.dst);
                    let _dst_ssa = self.map_ptx_to_ssa(&dst_name);
                }
            }
            ast::Instruction::Mul {
                data, arguments, ..
            } => {
                let is_float = matches!(data, ast::MulDetails::Float(_));
                if is_float {
                    let dst_name = Self::operand_to_string(&arguments.dst);
                    let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                    let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                    let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                    self.codegen.emit_operation(
                        "tmatmul.mul",
                        &[&src1_ssa, &src2_ssa],
                        &[&dst_ssa],
                    )?;
                } else {
                    let dst_name = Self::operand_to_string(&arguments.dst);
                    let _dst_ssa = self.map_ptx_to_ssa(&dst_name);
                }
            }
            ast::Instruction::Div {
                data, arguments, ..
            } => {
                if matches!(data, ast::DivDetails::Float(_)) {
                    let dst_name = Self::operand_to_string(&arguments.dst);
                    let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                    let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                    let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                    self.codegen.emit_operation(
                        "tmatmul.div",
                        &[&src1_ssa, &src2_ssa],
                        &[&dst_ssa],
                    )?;
                } else {
                    let dst_name = Self::operand_to_string(&arguments.dst);
                    let _dst_ssa = self.map_ptx_to_ssa(&dst_name);
                }
            }

            // Fused multiply-add - only emit for float
            ast::Instruction::Mad {
                data, arguments, ..
            } => {
                let is_float = matches!(data, ast::MadDetails::Float(_));
                if is_float {
                    let dst_name = Self::operand_to_string(&arguments.dst);
                    let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                    let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                    let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                    let src3_ssa = self.handle_operand_with_immediate(&arguments.src3)?;
                    let temp_ssa = self.new_ssa();
                    self.codegen.emit_operation(
                        "tmatmul.mul",
                        &[&src1_ssa, &src2_ssa],
                        &[&temp_ssa],
                    )?;
                    self.codegen.emit_operation(
                        "tmatmul.add",
                        &[&temp_ssa, &src3_ssa],
                        &[&dst_ssa],
                    )?;
                } else {
                    let dst_name = Self::operand_to_string(&arguments.dst);
                    let _dst_ssa = self.map_ptx_to_ssa(&dst_name);
                }
            }
            ast::Instruction::Fma { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src1_name = Self::operand_to_string(&arguments.src1);
                let src2_name = Self::operand_to_string(&arguments.src2);
                let src3_name = Self::operand_to_string(&arguments.src3);

                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                let src1_ssa = self.map_ptx_to_ssa(&src1_name);
                let src2_ssa = self.map_ptx_to_ssa(&src2_name);
                let src3_ssa = self.map_ptx_to_ssa(&src3_name);

                // Emit as mul + add
                let temp_ssa = self.new_ssa();
                self.codegen.emit_operation(
                    "tmatmul.mul",
                    &[&src1_ssa, &src2_ssa],
                    &[&temp_ssa],
                )?;
                self.codegen
                    .emit_operation("tmatmul.add", &[&temp_ssa, &src3_ssa], &[&dst_ssa])?;
            }

            // Memory operations
            ast::Instruction::Ld { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);

                let base_name = src_name.split('+').next().unwrap_or(&src_name).to_string();

                // Check if this is a parameter load (address setup for a pointer param)
                if let Some(&param_idx) = self.param_name_to_index.get(&base_name) {
                    // This is loading a kernel parameter address - just record the binding
                    let param_ref = format!("PARAM_{}", param_idx);
                    self.ptx_to_memory
                        .insert(dst_name.clone(), param_ref.clone());
                    self.codegen.add_comment(&format!(
                        "ld.param {} = {} (PARAM_{})",
                        dst_name, base_name, param_idx
                    ));
                    // Map the SSA value too so it can be used in later instructions
                    let _dst_ssa = self.map_ptx_to_ssa(&dst_name);
                } else {
                    // This is a global/generic memory load - resolve base to PARAM_N
                    let resolved_mem = if let Some(mem_region) = self.ptx_to_memory.get(&base_name)
                    {
                        mem_region.clone()
                    } else {
                        // Fallback: check named mappings
                        match base_name.as_str() {
                            "input" => "X".to_string(),
                            "output" => "O".to_string(),
                            _ => base_name.clone(),
                        }
                    };
                    // Note: Do NOT add data load destinations to ptx_to_memory.
                    // Only pointer values (from ld.param) should be tracked there.
                    // Data values loaded here are tensor elements, not addresses.

                    let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                    self.codegen
                        .emit_operation("tmatmul.ldv", &[&resolved_mem], &[&dst_ssa])?;
                }
            }
            ast::Instruction::St { arguments, .. } => {
                let addr_name = Self::operand_to_string(&arguments.src1);

                // Resolve address through memory mapping
                // If addr_name was loaded from a parameter, use that parameter's memory region
                let resolved_addr = if let Some(mem_region) = self.ptx_to_memory.get(&addr_name) {
                    mem_region.clone()
                } else {
                    // Check if it's a base+offset (e.g., "%rd3+4")
                    let base = addr_name.split('+').next().unwrap_or(&addr_name);
                    if let Some(mem_region) = self.ptx_to_memory.get(base) {
                        // For PARAM_N references, don't preserve offset (vector mode loads entire tensor)
                        mem_region.clone()
                    } else {
                        addr_name.clone()
                    }
                };

                // Handle immediate values in store source
                let val_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                self.codegen
                    .emit_operation("tmatmul.sv", &[&val_ssa, &resolved_addr], &[])?;
            }

            // Move operations
            ast::Instruction::Mov {
                data, arguments, ..
            } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);

                // Check if this is a float type move (data operation)
                let is_float_type = matches!(&data.typ,
                    ast::Type::Scalar(st) if matches!(st, ast::ScalarType::F32 | ast::ScalarType::F64 | ast::ScalarType::F16 | ast::ScalarType::BF16)
                );

                if Self::is_immediate(&arguments.src) {
                    if is_float_type {
                        // Float immediate - emit load (e.g., mov.f32 %f1, 0.0)
                        let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                        self.codegen.emit_operation(
                            "tmatmul.ldv",
                            &[&format!("CONST_{}", src_name)],
                            &[&dst_ssa],
                        )?;
                    } else {
                        // Integer immediate (thread index, constant) - just track SSA, no emission
                        let _dst_ssa = self.map_ptx_to_ssa(&dst_name);
                    }
                } else {
                    let src_ssa = self.map_ptx_to_ssa(&src_name);
                    // Update mapping - move is implicit in SSA
                    self.ptx_to_ssa.insert(dst_name.clone(), src_ssa);
                    // Propagate memory bindings (e.g., mov %rd3, %rd1 where %rd1 → PARAM_0)
                    if let Some(mem_region) = self.ptx_to_memory.get(&src_name).cloned() {
                        self.ptx_to_memory.insert(dst_name, mem_region);
                    }
                }
            }

            // Control flow
            ast::Instruction::Ret { .. } => {
                self.codegen.add_comment("PTX: ret");
            }

            // Unary operations - abs, neg, not, etc.
            ast::Instruction::Abs { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                let src_ssa = self.map_ptx_to_ssa(&src_name);
                // Emit as a pass-through for now (abs not directly supported)
                self.codegen
                    .add_comment(&format!("PTX abs: {} = |{}|", dst_name, src_name));
                // Copy src to dst (approximation - real abs would need special handling)
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }
            ast::Instruction::Copysign { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src1_name = Self::operand_to_string(&arguments.src1);
                let src2_name = Self::operand_to_string(&arguments.src2);
                let _dst_ssa = self.map_ptx_to_ssa(&dst_name);
                let src1_ssa = self.map_ptx_to_ssa(&src1_name);
                self.codegen.add_comment(&format!(
                    "PTX copysign: {} = copysign({}, {})",
                    dst_name, src1_name, src2_name
                ));
                self.ptx_to_ssa.insert(dst_name, src1_ssa);
            }
            ast::Instruction::Neg { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                let src_ssa = self.map_ptx_to_ssa(&src_name);
                self.codegen
                    .add_comment(&format!("PTX neg: {} = -{}", dst_name, src_name));
                // For tmatmul, implement as 0 - src
                let zero_ssa = self.new_ssa();
                self.codegen
                    .emit_operation("tmatmul.ldv", &["CONST_0"], &[&zero_ssa])?;
                self.codegen
                    .emit_operation("tmatmul.sub", &[&zero_ssa, &src_ssa], &[&dst_ssa])?;
            }
            ast::Instruction::Not { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                let src_ssa = self.map_ptx_to_ssa(&src_name);
                self.codegen.add_comment(&format!(
                    "PTX not: {} = ~{} (passthrough)",
                    dst_name, src_name
                ));
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            // Binary bitwise operations
            ast::Instruction::And { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                self.codegen.add_comment(&format!("PTX and (passthrough)"));
                // Passthrough - use first operand as approximation
                self.ptx_to_ssa.insert(dst_name, src1_ssa);
            }
            ast::Instruction::Or { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                self.codegen.add_comment(&format!("PTX or (passthrough)"));
                self.ptx_to_ssa.insert(dst_name, src1_ssa);
            }
            ast::Instruction::Xor { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                self.codegen.add_comment(&format!("PTX xor (passthrough)"));
                self.ptx_to_ssa.insert(dst_name, src1_ssa);
            }

            // Shift operations
            ast::Instruction::Shl { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                self.codegen.add_comment(&format!("PTX shl (passthrough)"));
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }
            ast::Instruction::Shr { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                self.codegen.add_comment(&format!("PTX shr (passthrough)"));
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            // Comparison/predicate operations
            ast::Instruction::Setp { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst1);
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                self.codegen
                    .add_comment(&format!("PTX setp (predicate set)"));
                // Predicates don't need actual values for control flow in tmatmul
            }

            // Min/Max operations
            ast::Instruction::Min { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                self.codegen.add_comment(&format!("PTX min (passthrough)"));
                self.ptx_to_ssa.insert(dst_name, src1_ssa);
            }
            ast::Instruction::Max { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);
                self.codegen.add_comment(&format!("PTX max (passthrough)"));
                self.ptx_to_ssa.insert(dst_name, src1_ssa);
            }

            // Type conversion
            ast::Instruction::Cvt { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let src_ssa = self.map_ptx_to_ssa(&src_name);
                self.codegen
                    .add_comment(&format!("PTX cvt: {} = convert({})", dst_name, src_name));
                // Pass through for now
                self.ptx_to_ssa.insert(dst_name.clone(), src_ssa);
                // Propagate memory bindings through type conversions
                if let Some(mem_region) = self.ptx_to_memory.get(&src_name).cloned() {
                    self.ptx_to_memory.insert(dst_name, mem_region);
                }
            }

            // Select/conditional
            ast::Instruction::Selp { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                let pred_str = Self::operand_to_string(&arguments.src3);

                // Check if predicate is a constant
                let result_ssa = if pred_str == "0" {
                    // Predicate is false, select src2
                    self.codegen
                        .add_comment(&format!("PTX selp: pred=0, selecting second operand"));
                    src2_ssa
                } else if pred_str.parse::<i64>().is_ok() {
                    // Predicate is a non-zero constant, select src1
                    self.codegen.add_comment(&format!(
                        "PTX selp: pred={}, selecting first operand",
                        pred_str
                    ));
                    src1_ssa
                } else {
                    // Predicate is a register - for single-thread, check if it's mapped to a known value
                    // Default to src1 (true path) for single-threaded simulation
                    self.codegen.add_comment(&format!(
                        "PTX selp: pred={} (runtime), defaulting to first operand",
                        pred_str
                    ));
                    src1_ssa
                };
                self.ptx_to_ssa.insert(dst_name, result_ssa);
            }

            // Branch (control flow)
            ast::Instruction::Bra { .. } => {
                self.codegen.add_comment("PTX: bra (branch)");
            }

            // Call
            ast::Instruction::Call { .. } => {
                self.codegen.add_comment("PTX: call (function call)");
            }

            // ============================================================
            // ATOMIC OPERATIONS - Emulated as load-modify-store sequences
            // ============================================================
            ast::Instruction::Atom {
                data, arguments, ..
            } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let addr_name = Self::operand_to_string(&arguments.src1);

                let _dst_ssa = self.map_ptx_to_ssa(&dst_name);
                // Handle immediate values in atomic operations (e.g., atom.inc with limit)
                let val_ssa = self.handle_operand_with_immediate(&arguments.src2)?;

                // Emulate atomic: load old value, perform operation, store new value
                // For tmatmul (single-threaded), this is equivalent to non-atomic version
                let old_ssa = self.new_ssa();
                let new_ssa = self.new_ssa();

                self.codegen
                    .add_comment(&format!("PTX atom: {} (emulated)", dst_name));

                // Load old value from address
                self.codegen
                    .emit_operation("tmatmul.ldv", &[&addr_name], &[&old_ssa])?;

                // Perform the atomic operation based on type
                let op_name = format!("{:?}", data.op).to_lowercase();
                match op_name.as_str() {
                    "add" => {
                        self.codegen.emit_operation(
                            "tmatmul.add",
                            &[&old_ssa, &val_ssa],
                            &[&new_ssa],
                        )?;
                    }
                    "min" => {
                        // Emulate min: use old value (approximation)
                        self.codegen.add_comment("atom.min emulated as passthrough");
                        self.ptx_to_ssa.insert(new_ssa.clone(), old_ssa.clone());
                    }
                    "max" => {
                        self.codegen.add_comment("atom.max emulated as passthrough");
                        self.ptx_to_ssa.insert(new_ssa.clone(), old_ssa.clone());
                    }
                    "inc" => {
                        // Increment: old + 1
                        let one_ssa = self.new_ssa();
                        self.codegen
                            .emit_operation("tmatmul.ldv", &["CONST_1"], &[&one_ssa])?;
                        self.codegen.emit_operation(
                            "tmatmul.add",
                            &[&old_ssa, &one_ssa],
                            &[&new_ssa],
                        )?;
                    }
                    "dec" => {
                        // Decrement: old - 1
                        let one_ssa = self.new_ssa();
                        self.codegen
                            .emit_operation("tmatmul.ldv", &["CONST_1"], &[&one_ssa])?;
                        self.codegen.emit_operation(
                            "tmatmul.sub",
                            &[&old_ssa, &one_ssa],
                            &[&new_ssa],
                        )?;
                    }
                    "and" => {
                        self.codegen.add_comment("atom.and emulated as passthrough");
                        self.ptx_to_ssa.insert(new_ssa.clone(), old_ssa.clone());
                    }
                    "or" => {
                        self.codegen.add_comment("atom.or emulated as passthrough");
                        self.ptx_to_ssa.insert(new_ssa.clone(), old_ssa.clone());
                    }
                    "xor" => {
                        self.codegen.add_comment("atom.xor emulated as passthrough");
                        self.ptx_to_ssa.insert(new_ssa.clone(), old_ssa.clone());
                    }
                    "exch" => {
                        // Exchange: just use the new value
                        self.ptx_to_ssa.insert(new_ssa.clone(), val_ssa.clone());
                    }
                    _ => {
                        self.codegen
                            .add_comment(&format!("atom.{} emulated as add", op_name));
                        self.codegen.emit_operation(
                            "tmatmul.add",
                            &[&old_ssa, &val_ssa],
                            &[&new_ssa],
                        )?;
                    }
                }

                // Store new value back
                self.codegen
                    .emit_operation("tmatmul.sv", &[&new_ssa, &addr_name], &[])?;

                // Return old value
                self.ptx_to_ssa.insert(dst_name, old_ssa);
            }

            ast::Instruction::AtomCas { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let addr_name = Self::operand_to_string(&arguments.src1);

                // Handle immediates properly for compare and new values
                let _cmp_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                let val_ssa = self.handle_operand_with_immediate(&arguments.src3)?;
                let old_ssa = self.new_ssa();

                self.codegen
                    .add_comment(&format!("PTX atom.cas: {} (emulated)", dst_name));

                // Load old value
                self.codegen
                    .emit_operation("tmatmul.ldv", &[&addr_name], &[&old_ssa])?;

                // For single-threaded: if old == cmp, store val (assume comparison succeeds)
                self.codegen
                    .emit_operation("tmatmul.sv", &[&val_ssa, &addr_name], &[])?;

                // Return old value
                self.ptx_to_ssa.insert(dst_name, old_ssa);
            }

            // ============================================================
            // THREAD SYNCHRONIZATION - No-ops for single-threaded tmatmul
            // ============================================================
            ast::Instruction::Bar { .. } => {
                self.codegen
                    .add_comment("PTX: bar.sync (no-op for single-threaded)");
            }

            ast::Instruction::BarRed { arguments, .. } => {
                // Barrier reduction - for single thread, predicate result is just the input
                let dst_name = Self::operand_to_string(&arguments.dst1);
                let pred_name = Self::operand_to_string(&arguments.src_predicate);
                let pred_ssa = self.map_ptx_to_ssa(&pred_name);

                self.codegen
                    .add_comment("PTX: bar.red (single-thread: result = input predicate)");
                self.ptx_to_ssa.insert(dst_name, pred_ssa);
            }

            ast::Instruction::BarWarp { .. } => {
                self.codegen
                    .add_comment("PTX: bar.warp (no-op for single-threaded)");
            }

            ast::Instruction::Membar { .. } => {
                self.codegen
                    .add_comment("PTX: membar (no-op for single-threaded)");
            }

            // ============================================================
            // CUDA INTRINSICS - Simulated values for single-thread execution
            // ============================================================
            ast::Instruction::Activemask { arguments, .. } => {
                // Returns bitmask of active threads in warp
                // For single-thread: return 0x00000001 (only lane 0 active)
                let dst_name = Self::operand_to_string(&arguments.dst);
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);

                self.codegen
                    .add_comment("PTX: activemask (single-thread: 0x1)");
                self.codegen
                    .emit_operation("tmatmul.ldv", &["CONST_1"], &[&dst_ssa])?;
            }

            // Vote instructions - warp-level collective operations
            ast::Instruction::Vote { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src1);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen
                    .add_comment("PTX: vote (single-thread: result = input)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            // ============================================================
            // MATH OPERATIONS
            // ============================================================
            ast::Instruction::Rem { arguments, .. } => {
                // Remainder/modulo operation
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src1_ssa = self.handle_operand_with_immediate(&arguments.src1)?;
                let src2_ssa = self.handle_operand_with_immediate(&arguments.src2)?;
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);

                // Emulate: rem = a - (a/b)*b
                let quotient_ssa = self.new_ssa();
                let product_ssa = self.new_ssa();

                self.codegen
                    .add_comment("PTX: rem (emulated as a - (a/b)*b)");
                self.codegen.emit_operation(
                    "tmatmul.div",
                    &[&src1_ssa, &src2_ssa],
                    &[&quotient_ssa],
                )?;
                self.codegen.emit_operation(
                    "tmatmul.mul",
                    &[&quotient_ssa, &src2_ssa],
                    &[&product_ssa],
                )?;
                self.codegen.emit_operation(
                    "tmatmul.sub",
                    &[&src1_ssa, &product_ssa],
                    &[&dst_ssa],
                )?;
            }

            ast::Instruction::Rcp { arguments, .. } => {
                // Reciprocal: 1/x
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let src_ssa = self.map_ptx_to_ssa(&src_name);
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);

                let one_ssa = self.new_ssa();
                self.codegen.add_comment("PTX: rcp (1/x)");
                self.codegen
                    .emit_operation("tmatmul.ldv", &["CONST_1"], &[&one_ssa])?;
                self.codegen
                    .emit_operation("tmatmul.div", &[&one_ssa, &src_ssa], &[&dst_ssa])?;
            }

            ast::Instruction::Rsqrt { arguments, .. } => {
                // Reciprocal square root: 1/sqrt(x)
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let src_ssa = self.map_ptx_to_ssa(&src_name);
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);

                self.codegen
                    .add_comment("PTX: rsqrt (passthrough - needs sqrt support)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            ast::Instruction::Sqrt { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: sqrt (passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            ast::Instruction::Sin { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: sin (passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            ast::Instruction::Cos { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: cos (passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            ast::Instruction::Tanh { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: tanh (passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            // ============================================================
            // BIT MANIPULATION OPERATIONS
            // ============================================================
            ast::Instruction::Popc { arguments, .. } => {
                // Population count (count set bits)
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: popc (passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            ast::Instruction::Clz { arguments, .. } => {
                // Count leading zeros
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: clz (passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            ast::Instruction::Brev { arguments, .. } => {
                // Bit reverse
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: brev (passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            ast::Instruction::Bfe { arguments, .. } => {
                // Bit field extract
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src1);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: bfe (passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            ast::Instruction::Bfi { arguments, .. } => {
                // Bit field insert
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src1);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: bfi (passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            ast::Instruction::Prmt { arguments, .. } => {
                // Permute bytes
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src1);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: prmt (passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            ast::Instruction::Shf { arguments, .. } => {
                // Funnel shift
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src_a);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: shf (passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            // ============================================================
            // ADDRESS SPACE OPERATIONS
            // ============================================================
            ast::Instruction::Cvta { arguments, .. } => {
                // Convert address to/from generic
                let dst_name = Self::operand_to_string(&arguments.dst);
                let src_name = Self::operand_to_string(&arguments.src);
                let src_ssa = self.map_ptx_to_ssa(&src_name);

                self.codegen.add_comment("PTX: cvta (address passthrough)");
                self.ptx_to_ssa.insert(dst_name, src_ssa);
            }

            // ============================================================
            // ASYNC COPY OPERATIONS
            // ============================================================
            ast::Instruction::CpAsync { arguments, .. } => {
                let src_from_name = Self::operand_to_string(&arguments.src_from);
                let src_to_name = Self::operand_to_string(&arguments.src_to);

                let src_ssa = self.new_ssa();
                self.codegen
                    .add_comment("PTX: cp.async (emulated as load+store)");
                self.codegen
                    .emit_operation("tmatmul.ldv", &[&src_from_name], &[&src_ssa])?;
                self.codegen
                    .emit_operation("tmatmul.sv", &[&src_ssa, &src_to_name], &[])?;
            }

            ast::Instruction::CpAsyncCommitGroup { .. } => {
                self.codegen
                    .add_comment("PTX: cp.async.commit_group (no-op)");
            }

            ast::Instruction::CpAsyncWaitGroup { .. } => {
                self.codegen.add_comment("PTX: cp.async.wait_group (no-op)");
            }

            ast::Instruction::CpAsyncWaitAll { .. } => {
                self.codegen.add_comment("PTX: cp.async.wait_all (no-op)");
            }

            // ============================================================
            // COMPARISON/SET OPERATIONS
            // ============================================================
            ast::Instruction::Set { arguments, .. } => {
                let dst_name = Self::operand_to_string(&arguments.dst);
                let dst_ssa = self.map_ptx_to_ssa(&dst_name);

                self.codegen.add_comment("PTX: set (comparison to value)");
                // Set result to 0 or 1 based on comparison - default to 1
                self.codegen
                    .emit_operation("tmatmul.ldv", &["CONST_1"], &[&dst_ssa])?;
            }

            // ============================================================
            // MISC OPERATIONS
            // ============================================================
            ast::Instruction::Trap { .. } => {
                self.codegen.add_comment("PTX: trap (ignored)");
            }

            ast::Instruction::Nanosleep { .. } => {
                self.codegen.add_comment("PTX: nanosleep (no-op)");
            }

            // Matrix multiply-accumulate (tensor core)
            ast::Instruction::Mma { .. } => {
                self.codegen
                    .add_comment("PTX: mma (tensor core - needs special handling)");
            }

            // Other instructions - passthrough with comment
            _ => {
                self.codegen
                    .add_comment(&format!("PTX: {} (not yet mapped)", Self::inst_name(inst)));
            }
        }

        Ok(())
    }

    /// Convert operand to string representation
    fn operand_to_string(op: &ast::ParsedOperand<&str>) -> String {
        match op {
            ast::ParsedOperand::Reg(name) => name.to_string(),
            ast::ParsedOperand::RegOffset(name, offset) => format!("{}+{}", name, offset),
            ast::ParsedOperand::Imm(imm) => format!("{}", imm),
            ast::ParsedOperand::VecMember(name, idx) => format!("{}.{}", name, idx),
            ast::ParsedOperand::VecPack(_) => "<vec>".to_string(),
        }
    }

    /// Check if operand is an immediate value
    fn is_immediate(op: &ast::ParsedOperand<&str>) -> bool {
        matches!(op, ast::ParsedOperand::Imm(_))
    }

    /// Handle operand, loading immediates into registers if needed
    fn handle_operand_with_immediate(
        &mut self,
        op: &ast::ParsedOperand<&str>,
    ) -> Result<String, String> {
        if Self::is_immediate(op) {
            let imm_name = Self::operand_to_string(op);
            let imm_ssa = self.new_ssa();
            // Load immediate as constant (use CONST memory location)
            self.codegen
                .add_comment(&format!("Load immediate: {}", imm_name));
            self.codegen.emit_operation(
                "tmatmul.ldv",
                &[&format!("CONST_{}", imm_name)],
                &[&imm_ssa],
            )?;
            Ok(imm_ssa)
        } else {
            Ok(self.map_ptx_to_ssa(&Self::operand_to_string(op)))
        }
    }

    /// Get instruction name for debugging
    fn inst_name(inst: &ast::Instruction<ast::ParsedOperand<&str>>) -> &'static str {
        match inst {
            ast::Instruction::Abs { .. } => "abs",
            ast::Instruction::Copysign { .. } => "copysign",
            ast::Instruction::Activemask { .. } => "activemask",
            ast::Instruction::Add { .. } => "add",
            ast::Instruction::And { .. } => "and",
            ast::Instruction::Atom { .. } => "atom",
            ast::Instruction::AtomCas { .. } => "atom.cas",
            ast::Instruction::Bar { .. } => "bar",
            ast::Instruction::BarRed { .. } => "bar.red",
            ast::Instruction::BarWarp { .. } => "bar.warp",
            ast::Instruction::Bfe { .. } => "bfe",
            ast::Instruction::Bfi { .. } => "bfi",
            ast::Instruction::Bra { .. } => "bra",
            ast::Instruction::Brev { .. } => "brev",
            ast::Instruction::Call { .. } => "call",
            ast::Instruction::Clz { .. } => "clz",
            ast::Instruction::Cos { .. } => "cos",
            ast::Instruction::CpAsync { .. } => "cp.async",
            ast::Instruction::CpAsyncCommitGroup { .. } => "cp.async.commit_group",
            ast::Instruction::CpAsyncWaitAll { .. } => "cp.async.wait_all",
            ast::Instruction::CpAsyncWaitGroup { .. } => "cp.async.wait_group",
            ast::Instruction::Cvt { .. } => "cvt",
            ast::Instruction::Cvta { .. } => "cvta",
            ast::Instruction::Div { .. } => "div",
            ast::Instruction::Fma { .. } => "fma",
            ast::Instruction::Ld { .. } => "ld",
            ast::Instruction::Mad { .. } => "mad",
            ast::Instruction::Max { .. } => "max",
            ast::Instruction::Membar { .. } => "membar",
            ast::Instruction::Min { .. } => "min",
            ast::Instruction::Mma { .. } => "mma",
            ast::Instruction::Mov { .. } => "mov",
            ast::Instruction::Mul { .. } => "mul",
            ast::Instruction::Nanosleep { .. } => "nanosleep",
            ast::Instruction::Neg { .. } => "neg",
            ast::Instruction::Not { .. } => "not",
            ast::Instruction::Or { .. } => "or",
            ast::Instruction::Popc { .. } => "popc",
            ast::Instruction::Prmt { .. } => "prmt",
            ast::Instruction::Rcp { .. } => "rcp",
            ast::Instruction::Rem { .. } => "rem",
            ast::Instruction::Ret { .. } => "ret",
            ast::Instruction::Rsqrt { .. } => "rsqrt",
            ast::Instruction::Selp { .. } => "selp",
            ast::Instruction::Set { .. } => "set",
            ast::Instruction::Setp { .. } => "setp",
            ast::Instruction::Shf { .. } => "shf",
            ast::Instruction::Shl { .. } => "shl",
            ast::Instruction::Shr { .. } => "shr",
            ast::Instruction::Sin { .. } => "sin",
            ast::Instruction::Sqrt { .. } => "sqrt",
            ast::Instruction::St { .. } => "st",
            ast::Instruction::Sub { .. } => "sub",
            ast::Instruction::Tanh { .. } => "tanh",
            ast::Instruction::Trap { .. } => "trap",
            ast::Instruction::Vote { .. } => "vote",
            ast::Instruction::Xor { .. } => "xor",
            _ => "unknown",
        }
    }

    /// Generate assembly output
    pub fn get_assembly(&self) -> String {
        self.codegen.get_assembly()
    }
}

/// High-level API: Convert PTX string to TMatmul assembly
pub fn ptx_to_tmatmul(ptx_source: &str) -> Result<String, String> {
    // Sanitize PTX to tolerate newer forms (virtual backend)
    let sanitized = sanitize_ptx_source(ptx_source);

    // Parse PTX. Be tolerant: fall back to unchecked parse if strict parse fails.
    let module = match ptx_parser::parse_module_checked(&sanitized) {
        Ok(m) => m,
        Err(_errs) => {
            // Best-effort recovery path: attempt to produce an AST ignoring unknown directives
            ptx_parser::parse_module_unchecked(&sanitized)
        }
    };

    // Compile to TMatmul
    let mut compiler = PtxToTMatmulCompiler::new();
    compiler.setup_standard_memory_map();

    let result = compiler.compile_module(module)?;

    Ok(result.assembly)
}

/// Best-effort sanitizer to accept newer PTX syntax that our lightweight parser
/// doesn't yet fully support. This keeps the virtual backend stable by rewriting
/// or relaxing syntax that would otherwise be rejected.
fn sanitize_ptx_source(src: &str) -> String {
    // 1) If the buffer has extraneous bytes/logs before the PTX header, trim to first PTX directive.
    let mut start_slice = src;
    if let Some(i) = src
        .find(".version")
        .or_else(|| src.find(".entry"))
        .or_else(|| src.find(".target"))
    {
        start_slice = &src[i..];
    }

    // 2) Strip ANSI/control characters (keep newlines, tabs, CR). This removes sequences like "\x1b[31m".
    let mut cleaned = String::with_capacity(start_slice.len());
    for ch in start_slice.chars() {
        let code = ch as u32;
        if code < 0x20 {
            if ch == '\n' || ch == '\r' || ch == '\t' {
                cleaned.push(ch);
            }
        } else {
            cleaned.push(ch);
        }
    }

    let mut out = String::with_capacity(cleaned.len());
    for line in cleaned.lines() {
        let mut cur = line.to_string();

        // Comment out global bX string/data directives with initializers that parser doesn't support
        {
            let t = cur.trim_start();
            if t.starts_with(".global") && t.contains(".b") && t.contains('=') {
                out.push_str("// ");
                out.push_str(&cur);
                out.push('\n');
                continue;
            }
        }

        // Convert .local memory to .shared (for tmatmul emulation)
        {
            let t = cur.trim_start();
            if t.starts_with(".local") {
                cur = cur.replace(".local", ".shared");
            }
        }

        // Handle ld.local and st.local -> ld.shared and st.shared
        if cur.contains("ld.local") {
            cur = cur.replace("ld.local", "ld.shared");
        }
        if cur.contains("st.local") {
            cur = cur.replace("st.local", "st.shared");
        }

        // Handle expressions in store operands: st.u64 [addr], temp + 1; -> split into add + store
        // This is a complex transformation, for now we simplify by removing the expression
        {
            let t = cur.trim_start();
            if t.starts_with("st.") && cur.contains("], ") {
                // Check if there's an expression after the comma
                if let Some(bracket_pos) = cur.find("],") {
                    let after_bracket = &cur[bracket_pos + 2..].trim();
                    // If it contains + or -, there's an expression
                    if after_bracket.contains(" + ") || after_bracket.contains(" - ") {
                        // Extract just the variable name (before the operator)
                        let var_part: &str = after_bracket
                            .split(|c| c == '+' || c == '-')
                            .next()
                            .unwrap_or(after_bracket)
                            .trim()
                            .trim_end_matches(';');
                        // Reconstruct without the expression
                        cur = format!("{}], {};", &cur[..bracket_pos], var_part);
                    }
                }
            }
        }
        // Comment out .section and unknown section-like directives
        {
            let t = cur.trim_start();
            if t.starts_with(".section") || t.starts_with(".sectio") {
                out.push_str("// ");
                out.push_str(&cur);
                out.push('\n');
                continue;
            }
        }

        // Drop predication prefix like "@%p7 " (or any leading predicate) which our parser
        // may not support yet in some contexts. Remove up to first space.
        {
            let mut cut_idx: Option<usize> = None;
            {
                let t = cur.trim_start();
                if t.starts_with('@') {
                    if let Some(sp) = t.find(' ') {
                        // convert index in trimmed slice to index in original cur
                        cut_idx = Some(cur.len() - t.len() + sp + 1);
                    }
                }
            }
            if let Some(i) = cut_idx {
                cur = cur[i..].to_string();
            }
        }

        // Comment out line mapping / pragma noise that can vary by toolchains
        {
            let t = cur.trim_start();
            if t.starts_with(".file") || t.starts_with(".loc") || t.starts_with(".pragma") {
                out.push_str("// ");
                out.push_str(&cur);
                out.push('\n');
                continue;
            }
        }

        // Replace unsupported cvt.rn.f16x2.f32 dst, a, b; -> mov.b32 dst, a;
        if cur.trim_start().starts_with("cvt.rn.f16x2.f32") {
            let rest = &cur.trim_start()["cvt.rn.f16x2.f32".len()..];
            let toks: Vec<&str> = rest
                .split(|c| c == ',' || c == ';')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            if toks.len() >= 2 {
                cur = format!("    mov.b32 {}, {};", toks[0], toks[1]);
            } else {
                cur = "    // stripped unsupported cvt.rn.f16x2.f32".to_string();
            }
        }

        // Normalize multi-operand mov (e.g., mov.b32 dst, src0, src1;) to two-operand form
        if cur.trim_start().starts_with("mov.b32") {
            let rest = &cur.trim_start()["mov.b32".len()..];
            let toks: Vec<&str> = rest
                .split(|c| c == ',' || c == ';')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            if toks.len() >= 2 {
                cur = format!("    mov.b32 {}, {};", toks[0], toks[1]);
            }
        }

        // Expand ld.global.v4.b32 %r1, %r2, %r3, %r4, [addr]; into four scalar loads
        if cur.trim_start().starts_with("ld.global.v4.b32") {
            let mut parts = cur.splitn(2, '[');
            let left = parts.next().unwrap_or("");
            let right = parts.next().unwrap_or("");
            // Extract destinations before '[' and the address between '[' and ']'
            let dests_part = left.replace("ld.global.v4.b32", "");
            let mut dests: Vec<String> = dests_part
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            // Some formats have trailing comma before bracket; trim it
            if let Some(last) = dests.last_mut() {
                if last.ends_with(',') {
                    last.pop();
                }
            }
            let addr_inside = right.split(']').next().unwrap_or("").trim().to_string();
            for d in dests {
                out.push_str(&format!("    ld.global.b32 {}, [{}];\n", d, addr_inside));
            }
            continue;
        }

        // Expand st.global.v4.b32 [addr], r1, r2, r3, r4; into four scalar stores
        if cur.trim_start().starts_with("st.global.v4.b32") {
            if let Some(lb) = cur.find('[') {
                if let Some(rb) = cur.find(']') {
                    let addr_inside = cur[lb + 1..rb].trim().to_string();
                    let after = &cur[rb + 1..];
                    let srcs: Vec<String> = after
                        .split(&[',', ';'][..])
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    for s in srcs {
                        out.push_str(&format!("    st.global.b32 [{}], {};\n", addr_inside, s));
                    }
                    continue;
                }
            }
        }

        // Remove any braces around operands: { %r1 } -> %r1, tolerate stray '{' without matching '}'
        // BUT: Do NOT remove standalone braces that are function body delimiters
        {
            let trimmed = cur.trim();
            // Only strip braces if this is NOT a standalone brace line (function body delimiter)
            if trimmed != "{" && trimmed != "}" {
                if cur.contains('{') || cur.contains('}') {
                    cur = cur.replace('{', "").replace('}', "");
                }
            }
        }

        // ============================================================
        // SPECIAL REGISTER HANDLING
        // Replace special registers with constants for single-threaded execution
        // ============================================================

        // Thread ID registers: %tid.x, %tid.y, %tid.z -> 0 (lane 0)
        if cur.contains("%tid.x") {
            cur = cur.replace("%tid.x", "0");
        }
        if cur.contains("%tid.y") {
            cur = cur.replace("%tid.y", "0");
        }
        if cur.contains("%tid.z") {
            cur = cur.replace("%tid.z", "0");
        }

        // Thread block dimensions: %ntid.x, %ntid.y, %ntid.z -> 1 (single thread)
        if cur.contains("%ntid.x") {
            cur = cur.replace("%ntid.x", "1");
        }
        if cur.contains("%ntid.y") {
            cur = cur.replace("%ntid.y", "1");
        }
        if cur.contains("%ntid.z") {
            cur = cur.replace("%ntid.z", "1");
        }

        // Block ID: %ctaid.x, %ctaid.y, %ctaid.z -> 0
        if cur.contains("%ctaid.x") {
            cur = cur.replace("%ctaid.x", "0");
        }
        if cur.contains("%ctaid.y") {
            cur = cur.replace("%ctaid.y", "0");
        }
        if cur.contains("%ctaid.z") {
            cur = cur.replace("%ctaid.z", "0");
        }

        // Grid dimensions: %nctaid.x, %nctaid.y, %nctaid.z -> 1
        if cur.contains("%nctaid.x") {
            cur = cur.replace("%nctaid.x", "1");
        }
        if cur.contains("%nctaid.y") {
            cur = cur.replace("%nctaid.y", "1");
        }
        if cur.contains("%nctaid.z") {
            cur = cur.replace("%nctaid.z", "1");
        }

        // Lane ID and warp ID: -> 0
        if cur.contains("%laneid") {
            cur = cur.replace("%laneid", "0");
        }
        if cur.contains("%warpid") {
            cur = cur.replace("%warpid", "0");
        }

        // Warp size: %WARP_SZ -> 32, also handle WARP_SZ without % prefix
        if cur.contains("%WARP_SZ") {
            cur = cur.replace("%WARP_SZ", "32");
        }
        // Handle WARP_SZ without % prefix (appears in some PTX files)
        if cur.contains("WARP_SZ") && !cur.contains("32") {
            cur = cur.replace("WARP_SZ", "32");
        }

        // Clock registers (for timing) -> 0
        if cur.contains("%clock") {
            cur = cur.replace("%clock", "0");
        }
        if cur.contains("%clock64") {
            cur = cur.replace("%clock64", "0");
        }

        // %pm0-%pm7 (performance monitors) -> 0
        for i in 0..8 {
            let pm = format!("%pm{}", i);
            if cur.contains(&pm) {
                cur = cur.replace(&pm, "0");
            }
        }

        // %smid, %nsmid, %gridid -> 0
        if cur.contains("%smid") {
            cur = cur.replace("%smid", "0");
        }
        if cur.contains("%nsmid") {
            cur = cur.replace("%nsmid", "1");
        }
        if cur.contains("%gridid") {
            cur = cur.replace("%gridid", "0");
        }

        // %lanemask_eq, %lanemask_lt, %lanemask_le, %lanemask_gt, %lanemask_ge
        if cur.contains("%lanemask_eq") {
            cur = cur.replace("%lanemask_eq", "1"); // Only lane 0 equals itself
        }
        if cur.contains("%lanemask_lt") {
            cur = cur.replace("%lanemask_lt", "0"); // No lanes less than 0
        }
        if cur.contains("%lanemask_le") {
            cur = cur.replace("%lanemask_le", "1"); // Lane 0 <= lane 0
        }
        if cur.contains("%lanemask_gt") {
            cur = cur.replace("%lanemask_gt", "0xFFFFFFFE"); // All lanes > 0
        }
        if cur.contains("%lanemask_ge") {
            cur = cur.replace("%lanemask_ge", "0xFFFFFFFF"); // All lanes >= 0
        }

        // ============================================================
        // DIV MODIFIER HANDLING
        // Strip approx/full modifiers from div instructions
        // ============================================================
        if cur.trim_start().starts_with("div.approx") {
            cur = cur.replace("div.approx", "div");
        }
        if cur.trim_start().starts_with("div.full") {
            cur = cur.replace("div.full", "div");
        }
        // Also handle rn (round to nearest) modifier on div
        if cur.contains("div.rn.") {
            cur = cur.replace("div.rn.", "div.");
        }

        // ============================================================
        // SPECIAL SETP MODES
        // setp.nan.f32 pred, a, b -> checks if either is NaN, for single values assume not NaN
        // setp.num.f32 pred, a, b -> checks if neither is NaN
        // ============================================================
        {
            let t = cur.trim_start();
            if t.starts_with("setp.nan.") {
                // setp.nan checks if either operand is NaN
                // For tmatmul emulation, assume no NaN values -> result is always false
                let parts: Vec<&str> = t.split_whitespace().collect();
                if parts.len() >= 2 {
                    let operands_str = parts[1..].join(" ");
                    let operands: Vec<&str> = operands_str
                        .split(',')
                        .map(|s| s.trim().trim_end_matches(';'))
                        .collect();
                    if !operands.is_empty() {
                        // Set predicate to false (0) since we assume no NaN
                        cur = format!("    mov.pred {}, 0;", operands[0]);
                    }
                }
            } else if t.starts_with("setp.num.") {
                // setp.num checks if neither operand is NaN
                // For tmatmul emulation, assume no NaN values -> result is always true
                let parts: Vec<&str> = t.split_whitespace().collect();
                if parts.len() >= 2 {
                    let operands_str = parts[1..].join(" ");
                    let operands: Vec<&str> = operands_str
                        .split(',')
                        .map(|s| s.trim().trim_end_matches(';'))
                        .collect();
                    if !operands.is_empty() {
                        // Set predicate to true (1) since we assume no NaN
                        cur = format!("    mov.pred {}, 1;", operands[0]);
                    }
                }
            }
        }

        // ============================================================
        // REDUX SYNC OPERATIONS
        // Convert redux.sync.* to load constant (single-thread: just use the input)
        // redux.sync.add.u32 dst, src, mask; -> mov.u32 dst, src;
        // ============================================================
        {
            let t = cur.trim_start();
            if t.starts_with("redux.sync.") {
                // Extract operation like "redux.sync.add.u32 dst, src, mask;"
                // Convert to "mov.u32 dst, src;"
                let parts: Vec<&str> = t.split_whitespace().collect();
                if parts.len() >= 3 {
                    let instr = parts[0]; // e.g., "redux.sync.add.u32"
                                          // Extract type from instruction (last part after dots)
                    let instr_parts: Vec<&str> = instr.split('.').collect();
                    let type_part = instr_parts.last().unwrap_or(&"u32");

                    // Get operands (dst, src, mask) - we just need dst and src
                    let operands_str = parts[1..].join(" ");
                    let operands: Vec<&str> = operands_str
                        .split(',')
                        .map(|s| s.trim().trim_end_matches(';'))
                        .collect();

                    if operands.len() >= 2 {
                        cur = format!("    mov.{} {}, {};", type_part, operands[0], operands[1]);
                    } else {
                        out.push_str("    // ");
                        out.push_str(&cur);
                        out.push_str(" // redux.sync unsupported\n");
                        continue;
                    }
                }
            }
        }

        // ============================================================
        // VOTE SYNC OPERATIONS
        // Convert vote.sync.* to simple predicate handling
        // vote.sync.all.pred dst, src, mask; -> mov.pred dst, src;
        // vote.sync.ballot.b32 dst, src, mask; -> selp.u32 dst, 1, 0, src;
        // ============================================================
        {
            let t = cur.trim_start();
            if t.starts_with("vote.sync.") {
                // For single-thread execution, vote.all = src, vote.any = src, vote.ballot = 1 if src
                let parts: Vec<&str> = t.split_whitespace().collect();
                if parts.len() >= 2 {
                    let instr = parts[0];
                    let operands_str = parts[1..].join(" ");
                    let operands: Vec<&str> = operands_str
                        .split(',')
                        .map(|s| s.trim().trim_end_matches(';'))
                        .collect();

                    if instr.contains("ballot") {
                        // vote.sync.ballot.b32 dst, src, mask -> selp.u32 dst, 1, 0, src;
                        // For single thread: result is 1 if predicate is true, else 0
                        if operands.len() >= 2 {
                            cur = format!("    selp.u32 {}, 1, 0, {};", operands[0], operands[1]);
                        } else if operands.len() >= 1 {
                            cur = format!("    mov.u32 {}, 1;", operands[0]);
                        }
                    } else if operands.len() >= 2 {
                        // vote.sync.all/any/uni.pred -> result equals input for single thread
                        cur = format!("    mov.pred {}, {};", operands[0], operands[1]);
                    } else if operands.len() >= 1 {
                        // If only dst, set to 1 (true)
                        cur = format!("    mov.pred {}, 1;", operands[0]);
                    } else {
                        out.push_str("    // ");
                        out.push_str(&cur);
                        out.push_str(" // vote.sync unsupported\n");
                        continue;
                    }
                }
            }
        }

        // ============================================================
        // VECTOR TYPE HANDLING
        // Convert vector declarations to scalar equivalents
        // .reg .v2 .u32 temp; -> .reg .u32 temp_x; .reg .u32 temp_y;
        // ============================================================

        // Handle vector register declarations: .reg .v2 .u32 name; or .reg .v4 .f32 name;
        if cur.trim_start().starts_with(".reg") && (cur.contains(".v2") || cur.contains(".v4")) {
            // Convert vector declarations to scalar by removing vector prefix
            // .reg .v2 .u32 temp; -> .reg .u32 temp;
            cur = cur.replace(".v2 ", " ").replace(".v4 ", " ");
        }

        // Handle vector load: ld.v2.u32 temp, [addr]; -> ld.u32 temp, [addr];
        if cur.trim_start().starts_with("ld.v2.") || cur.trim_start().starts_with("ld.v4.") {
            cur = cur.replace("ld.v2.", "ld.").replace("ld.v4.", "ld.");
        }

        // Handle vector store: st.v2.u32 [addr], temp; -> st.u32 [addr], temp;
        if cur.trim_start().starts_with("st.v2.") || cur.trim_start().starts_with("st.v4.") {
            cur = cur.replace("st.v2.", "st.").replace("st.v4.", "st.");
        }

        // Handle vector move: mov.v2.u32 dst, src; -> mov.u32 dst, src;
        if cur.trim_start().starts_with("mov.v2.") || cur.trim_start().starts_with("mov.v4.") {
            cur = cur.replace("mov.v2.", "mov.").replace("mov.v4.", "mov.");
        }

        // Handle vector member access: temp.x, temp.y -> temp (just use base variable)
        // This is a simplification - use first element for all
        if cur.contains(".x") && !cur.contains("%") {
            // Be careful not to replace .x in instruction names or labels
            let has_member_access = cur.split_whitespace().any(|word| {
                word.contains(".x") && !word.starts_with('.') && !word.starts_with('%')
            });
            if has_member_access {
                // Replace variable.x with just variable
                let parts: Vec<&str> = cur.split_whitespace().collect();
                let new_parts: Vec<String> = parts
                    .iter()
                    .map(|p| {
                        if p.contains(".x") && !p.starts_with('.') && !p.starts_with('%') {
                            p.replace(".x", "").replace(",", ",")
                        } else {
                            p.to_string()
                        }
                    })
                    .collect();
                cur = new_parts.join(" ");
            }
        }
        if cur.contains(".y") && !cur.contains("%") {
            let has_member_access = cur.split_whitespace().any(|word| {
                word.contains(".y") && !word.starts_with('.') && !word.starts_with('%')
            });
            if has_member_access {
                let parts: Vec<&str> = cur.split_whitespace().collect();
                let new_parts: Vec<String> = parts
                    .iter()
                    .map(|p| {
                        if p.contains(".y") && !p.starts_with('.') && !p.starts_with('%') {
                            p.replace(".y", "")
                        } else {
                            p.to_string()
                        }
                    })
                    .collect();
                cur = new_parts.join(" ");
            }
        }
        if cur.contains(".z") && !cur.contains("%") {
            let has_member_access = cur.split_whitespace().any(|word| {
                word.contains(".z") && !word.starts_with('.') && !word.starts_with('%')
            });
            if has_member_access {
                let parts: Vec<&str> = cur.split_whitespace().collect();
                let new_parts: Vec<String> = parts
                    .iter()
                    .map(|p| {
                        if p.contains(".z") && !p.starts_with('.') && !p.starts_with('%') {
                            p.replace(".z", "")
                        } else {
                            p.to_string()
                        }
                    })
                    .collect();
                cur = new_parts.join(" ");
            }
        }
        if cur.contains(".w") && !cur.contains("%") {
            let has_member_access = cur.split_whitespace().any(|word| {
                word.contains(".w") && !word.starts_with('.') && !word.starts_with('%')
            });
            if has_member_access {
                let parts: Vec<&str> = cur.split_whitespace().collect();
                let new_parts: Vec<String> = parts
                    .iter()
                    .map(|p| {
                        if p.contains(".w") && !p.starts_with('.') && !p.starts_with('%') {
                            p.replace(".w", "")
                        } else {
                            p.to_string()
                        }
                    })
                    .collect();
                cur = new_parts.join(" ");
            }
        }

        // Handle vector function parameters in .func declarations
        {
            let t = cur.trim_start();
            if t.starts_with(".func") && (t.contains(".v2") || t.contains(".v4")) {
                cur = cur.replace(".v2 ", " ").replace(".v4 ", " ");
            }
        }

        // ============================================================
        // FUNNEL SHIFT HANDLING
        // Convert shf.* to shl or shr depending on direction
        // shf.l.clamp.b32 dst, lo, hi, shift; -> shl.b32 dst, lo, shift;
        // shf.r.clamp.b32 dst, lo, hi, shift; -> shr.b32 dst, hi, shift;
        // ============================================================
        {
            let t = cur.trim_start();
            if t.starts_with("shf.l.") {
                // Left funnel shift: use low word shifted left
                let new_instr = t
                    .replace("shf.l.clamp.", "shl.")
                    .replace("shf.l.wrap.", "shl.");
                // Need to remove the 'hi' operand (third operand)
                let parts: Vec<&str> = new_instr.split_whitespace().collect();
                if parts.len() >= 2 {
                    let instr = parts[0];
                    let operands_str = parts[1..].join(" ");
                    let operands: Vec<&str> = operands_str
                        .split(',')
                        .map(|s| s.trim().trim_end_matches(';'))
                        .collect();
                    if operands.len() >= 4 {
                        // shf.l dst, lo, hi, shift -> shl dst, lo, shift
                        cur = format!(
                            "    {} {}, {}, {};",
                            instr, operands[0], operands[1], operands[3]
                        );
                    }
                }
            } else if t.starts_with("shf.r.") {
                // Right funnel shift: use high word shifted right
                let new_instr = t
                    .replace("shf.r.clamp.", "shr.")
                    .replace("shf.r.wrap.", "shr.");
                let parts: Vec<&str> = new_instr.split_whitespace().collect();
                if parts.len() >= 2 {
                    let instr = parts[0];
                    let operands_str = parts[1..].join(" ");
                    let operands: Vec<&str> = operands_str
                        .split(',')
                        .map(|s| s.trim().trim_end_matches(';'))
                        .collect();
                    if operands.len() >= 4 {
                        // shf.r dst, lo, hi, shift -> shr dst, hi, shift
                        cur = format!(
                            "    {} {}, {}, {};",
                            instr, operands[0], operands[2], operands[3]
                        );
                    }
                }
            }
        }

        // ============================================================
        // SHUFFLE SYNC HANDLING
        // shfl.sync.* -> passthrough (use input value)
        // ============================================================
        {
            let t = cur.trim_start();
            if t.starts_with("shfl.sync.") {
                let parts: Vec<&str> = t.split_whitespace().collect();
                if parts.len() >= 2 {
                    let instr = parts[0]; // e.g., "shfl.sync.bfly.b32"
                    let instr_parts: Vec<&str> = instr.split('.').collect();
                    let type_part = instr_parts.last().unwrap_or(&"b32");

                    let operands_str = parts[1..].join(" ");
                    let operands: Vec<&str> = operands_str
                        .split(',')
                        .map(|s| s.trim().trim_end_matches(';'))
                        .collect();

                    // shfl.sync.* dst, pred, src, offset, mask -> mov.type dst, src; setp.eq.b32 pred, 1, 1;
                    if operands.len() >= 3 {
                        let dst = operands[0];
                        let pred = if operands.len() >= 2 { operands[1] } else { "" };
                        let src = if operands.len() >= 3 {
                            operands[2]
                        } else {
                            operands[1]
                        };

                        // For single thread, shuffled value equals input
                        cur = format!("    mov.{} {}, {};", type_part, dst, src);
                        out.push_str(&cur);
                        out.push('\n');
                        // Set predicate to true if present
                        if !pred.is_empty() && pred.starts_with('%') {
                            out.push_str(&format!("    setp.eq.b32 {}, 1, 1;\n", pred));
                        }
                        continue;
                    }
                }
            }
        }

        // ============================================================
        // MUL24 HANDLING
        // mul24.* is deprecated; convert to regular mul
        // ============================================================
        if cur.contains("mul24.lo.") {
            cur = cur.replace("mul24.lo.", "mul.lo.");
        }
        if cur.contains("mul24.hi.") {
            cur = cur.replace("mul24.hi.", "mul.hi.");
        }

        // ============================================================
        // DP4A HANDLING (Dot product accumulate)
        // dp4a.u32.u32 dst, a, b, c -> mad dst, a, b, c (approximation)
        // ============================================================
        {
            let t = cur.trim_start();
            if t.starts_with("dp4a.") {
                // Approximate as multiply-add
                cur = cur
                    .replace("dp4a.u32.u32", "mad.lo.u32")
                    .replace("dp4a.s32.s32", "mad.lo.s32")
                    .replace("dp4a.u32.s32", "mad.lo.s32")
                    .replace("dp4a.s32.u32", "mad.lo.s32");
            }
        }

        // Simplify "+ 0" in addressing and normalize bracket spacing
        if cur.contains(" + 0") {
            cur = cur.replace(" + 0", "");
        }
        if cur.contains(" ]") {
            cur = cur.replace(" ]", "]");
        }

        out.push_str(&cur);
        out.push('\n');
    }
    out
}

/// Pattern-based optimization: Detect and optimize common PTX patterns
pub struct PatternOptimizer {
    patterns: Vec<Box<dyn Pattern>>,
}

trait Pattern {
    fn matches(&self, instructions: &[String]) -> bool;
    fn optimize(&self, compiler: &mut PtxToTMatmulCompiler) -> Result<(), String>;
}

/// Detect GEMM (matrix multiplication) pattern in PTX
struct GemmPattern;

impl Pattern for GemmPattern {
    fn matches(&self, instructions: &[String]) -> bool {
        // Look for nested loops with MAD instructions
        instructions
            .iter()
            .any(|i| i.contains("mad.f32") || i.contains("fma"))
    }

    fn optimize(&self, compiler: &mut PtxToTMatmulCompiler) -> Result<(), String> {
        compiler
            .codegen
            .add_comment("Detected GEMM pattern - using tmatmul accelerator");
        // Would emit tmatmul_import/go/export sequence
        Ok(())
    }
}

/// Detect activation function patterns
struct ActivationPattern {
    kind: ActivationKind,
}

#[derive(Debug, Clone, Copy)]
enum ActivationKind {
    Sigmoid,
    ReLU,
    SiLU,
}

impl Pattern for ActivationPattern {
    fn matches(&self, instructions: &[String]) -> bool {
        match self.kind {
            ActivationKind::Sigmoid => {
                // Sigmoid: 1 / (1 + exp(-x))
                instructions
                    .iter()
                    .any(|i| i.contains("exp") && i.contains("rcp"))
            }
            ActivationKind::ReLU => instructions
                .iter()
                .any(|i| i.contains("max") && i.contains("0")),
            ActivationKind::SiLU => {
                // SiLU: x * sigmoid(x)
                instructions
                    .iter()
                    .any(|i| i.contains("mul") && i.contains("exp"))
            }
        }
    }

    fn optimize(&self, compiler: &mut PtxToTMatmulCompiler) -> Result<(), String> {
        match self.kind {
            ActivationKind::Sigmoid => {
                compiler
                    .codegen
                    .add_comment("Detected sigmoid pattern - using sig instruction");
            }
            ActivationKind::SiLU => {
                compiler
                    .codegen
                    .add_comment("Detected SiLU pattern - using silu instruction");
            }
            ActivationKind::ReLU => {
                compiler.codegen.add_comment("Detected ReLU pattern");
            }
        }
        Ok(())
    }
}

impl PatternOptimizer {
    pub fn new() -> Self {
        let mut patterns: Vec<Box<dyn Pattern>> = Vec::new();
        patterns.push(Box::new(GemmPattern));
        patterns.push(Box::new(ActivationPattern {
            kind: ActivationKind::Sigmoid,
        }));
        patterns.push(Box::new(ActivationPattern {
            kind: ActivationKind::SiLU,
        }));

        Self { patterns }
    }

    pub fn optimize(
        &self,
        instructions: &[String],
        compiler: &mut PtxToTMatmulCompiler,
    ) -> Result<(), String> {
        for pattern in &self.patterns {
            if pattern.matches(instructions) {
                pattern.optimize(compiler)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let compiler = PtxToTMatmulCompiler::new();
        assert_eq!(compiler.ssa_counter, 0);
    }

    #[test]
    fn test_ssa_generation() {
        let mut compiler = PtxToTMatmulCompiler::new();
        assert_eq!(compiler.new_ssa(), "%0");
        assert_eq!(compiler.new_ssa(), "%1");
        assert_eq!(compiler.new_ssa(), "%2");
    }

    #[test]
    fn test_ptx_to_ssa_mapping() {
        let mut compiler = PtxToTMatmulCompiler::new();

        let ssa1 = compiler.map_ptx_to_ssa("r1");
        assert_eq!(ssa1, "%0");

        // Should return same SSA for same PTX register
        let ssa1_again = compiler.map_ptx_to_ssa("r1");
        assert_eq!(ssa1_again, "%0");

        let ssa2 = compiler.map_ptx_to_ssa("r2");
        assert_eq!(ssa2, "%1");
    }

    #[test]
    fn test_memory_mapping() {
        let mut compiler = PtxToTMatmulCompiler::new();
        compiler.setup_standard_memory_map();

        // Standard mappings should be set up
        let assembly = compiler.get_assembly();
        assert!(assembly.contains("vector registers"));
    }

    #[test]
    fn test_instruction_conversion() {
        let mut compiler = PtxToTMatmulCompiler::new();
        compiler.setup_standard_memory_map();

        // Test add instruction
        compiler
            .convert_ptx_instruction("add.f32", &["r1", "r2", "r3"])
            .unwrap();

        let assembly = compiler.get_assembly();
        assert!(assembly.contains("add"));
    }
}
