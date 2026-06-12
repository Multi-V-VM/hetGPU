// tmatmul_algorithm_tree.rs - Algorithm tree generation for TMatmul accelerator
// Rust reimplementation of ternary_matmul's algorithm_tree.py
// Integrated with hetGPU's tmatmul backend

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Hardware configuration for the TMatmul accelerator
#[derive(Debug, Clone)]
pub struct TMatmulConfig {
    /// Vector dimension (D x D ternary matrices)
    pub d: usize,
    /// Number of parallel cores
    pub num_cores: usize,
    /// Number of vector registers (v0-v7)
    pub num_vector_registers: usize,
    /// Fixed-point precision in bits
    pub fixed_point_precision: usize,
    /// Fixed-point exponent (for scaling)
    pub fixed_point_exponent: i32,
    /// Vector parallelism
    pub vector_parallelism: usize,
    /// TMatmul parallelism
    pub tmatmul_parallelism: usize,
    /// LUT parallelism
    pub lut_parallelism: usize,
    /// Memory bandwidth (bytes per cycle)
    pub memory_bandwidth: usize,
    /// Clock period in seconds
    pub clock_period: f64,
}

impl Default for TMatmulConfig {
    fn default() -> Self {
        Self {
            d: 64,
            num_cores: 1,
            num_vector_registers: 8,
            fixed_point_precision: 16,
            fixed_point_exponent: -8,
            vector_parallelism: 16,
            tmatmul_parallelism: 64,
            lut_parallelism: 8,
            memory_bandwidth: 32,
            clock_period: 1e-9,
        }
    }
}

impl TMatmulConfig {
    /// Size of a DxD ternary matrix in bytes (packed 2-bit encoding)
    pub fn matrix_size_in_bytes(&self) -> usize {
        (self.d * self.d * 2) / 8
    }

    /// Size of a D-element vector in bytes
    pub fn vector_size_in_bytes(&self) -> usize {
        self.d * self.num_cores * (self.fixed_point_precision / 8)
    }

    /// Maximum memory bandwidth per cycle
    pub fn max_bytes_per_cycle(&self) -> usize {
        self.memory_bandwidth
    }
}

/// Abstract vector node - represents a logical vector in the algorithm
#[derive(Debug, Clone)]
pub struct AbstractVectorNode {
    pub size: usize,
    pub instruction_vector_ids: Vec<usize>,
}

/// Abstract operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstractOperation {
    Add,
    Sub,
    Mul,
    Sig,
    CSig,
    SiLU,
    RmsNorm,
    TMatmul,
    Ldv,
    Sv,
}

impl fmt::Display for AbstractOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbstractOperation::Add => write!(f, "add"),
            AbstractOperation::Sub => write!(f, "sub"),
            AbstractOperation::Mul => write!(f, "mul"),
            AbstractOperation::Sig => write!(f, "sig"),
            AbstractOperation::CSig => write!(f, "csig"),
            AbstractOperation::SiLU => write!(f, "silu"),
            AbstractOperation::RmsNorm => write!(f, "rms_norm"),
            AbstractOperation::TMatmul => write!(f, "tmatmul"),
            AbstractOperation::Ldv => write!(f, "ldv"),
            AbstractOperation::Sv => write!(f, "sv"),
        }
    }
}

/// Abstract operation node
#[derive(Debug, Clone)]
pub struct AbstractOperationNode {
    pub operation: AbstractOperation,
    pub input_vector_ids: Vec<usize>,
    pub output_vector_ids: Vec<usize>,
    pub other_info: HashMap<String, OperationInfo>,
}

/// Operation info value types
#[derive(Debug, Clone)]
pub enum OperationInfo {
    String(String),
    Int(i64),
    Float(f64),
}

impl OperationInfo {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            OperationInfo::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            OperationInfo::Int(i) => Some(*i),
            _ => None,
        }
    }
}

/// Instruction-level vector node
#[derive(Debug, Clone)]
pub struct InstructionVectorNode {
    pub fake: bool,
}

/// Instruction-level operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstructionOperation {
    Add,
    Sub,
    Mul,
    Sig,
    CSig,
    SiLU,
    RmsAccumulate,
    RmsFinishAccumulate,
    RmsNorm,
    TMatmulImport,
    TMatmulGo,
    TMatmulExport,
    Ldv,
    Sv,
}

impl fmt::Display for InstructionOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstructionOperation::Add => write!(f, "add"),
            InstructionOperation::Sub => write!(f, "sub"),
            InstructionOperation::Mul => write!(f, "mul"),
            InstructionOperation::Sig => write!(f, "sig"),
            InstructionOperation::CSig => write!(f, "csig"),
            InstructionOperation::SiLU => write!(f, "silu"),
            InstructionOperation::RmsAccumulate => write!(f, "rms_accumulate"),
            InstructionOperation::RmsFinishAccumulate => write!(f, "rms_finish_accumulate"),
            InstructionOperation::RmsNorm => write!(f, "rms_norm"),
            InstructionOperation::TMatmulImport => write!(f, "tmatmul_import"),
            InstructionOperation::TMatmulGo => write!(f, "tmatmul_go"),
            InstructionOperation::TMatmulExport => write!(f, "tmatmul_export"),
            InstructionOperation::Ldv => write!(f, "ldv"),
            InstructionOperation::Sv => write!(f, "sv"),
        }
    }
}

/// Instruction-level operation node
#[derive(Debug, Clone)]
pub struct InstructionOperationNode {
    pub operation: InstructionOperation,
    pub input_vector_ids: Vec<usize>,
    pub output_vector_ids: Vec<usize>,
    pub other_info: HashMap<String, OperationInfo>,
}

/// Register swap information
#[derive(Debug, Clone)]
pub struct RegisterSwapInfo {
    pub previous_register_address: usize,
    pub swap_out_before: usize,
    pub swap_address_label: String,
    pub swap_in_before: Option<usize>,
    pub new_register_address: Option<usize>,
}

/// Register assignment for a vector
#[derive(Debug, Clone)]
pub struct RegisterAssignment {
    pub initial_register_address: Option<usize>,
    pub swaps: Vec<RegisterSwapInfo>,
}

impl RegisterAssignment {
    pub fn new(initial: Option<usize>) -> Self {
        Self {
            initial_register_address: initial,
            swaps: Vec::new(),
        }
    }

    /// Get register name at a given program counter
    pub fn register_name_at_pc(&self, pc: usize) -> String {
        let mut cur_reg = self.initial_register_address;
        for s in &self.swaps {
            if let Some(swap_in) = s.swap_in_before {
                if pc >= swap_in {
                    cur_reg = s.new_register_address;
                }
            }
        }
        match cur_reg {
            Some(r) => format!("v{}", r),
            None => "v?".to_string(),
        }
    }
}

/// Main algorithm tree structure
pub struct AlgorithmTree {
    pub config: TMatmulConfig,
    pub abstract_vectors: Vec<AbstractVectorNode>,
    pub abstract_operations: Vec<AbstractOperationNode>,
    pub abstract_vector_address_labels: Vec<String>,
    pub abstract_matrix_address_labels: Vec<String>,
    pub instruction_vectors: Vec<InstructionVectorNode>,
    pub instruction_operations: Vec<InstructionOperationNode>,
    pub instruction_vector_address_labels: Vec<String>,
    pub instruction_matrix_address_labels: Vec<String>,
}

impl AlgorithmTree {
    pub fn new(config: TMatmulConfig) -> Self {
        Self {
            config,
            abstract_vectors: Vec::new(),
            abstract_operations: Vec::new(),
            abstract_vector_address_labels: Vec::new(),
            abstract_matrix_address_labels: Vec::new(),
            instruction_vectors: Vec::new(),
            instruction_operations: Vec::new(),
            instruction_vector_address_labels: Vec::new(),
            instruction_matrix_address_labels: Vec::new(),
        }
    }

    /// Generate matrix block name from base name and indices
    pub fn matrix_name_to_block_name(base: &str, r: usize, c: usize) -> String {
        format!("{}_block_{}_{}", base, r, c)
    }

    /// Generate vector slice name from base name and index
    pub fn vector_name_to_slice_name(base: &str, s: usize) -> String {
        format!("{}_slice_{}", base, s)
    }

    /// Create a new abstract vector
    pub fn new_abstract_vector(&mut self, size: usize) -> usize {
        let num_new_instruction_vectors = (size + self.config.d - 1) / self.config.d;
        let mut new_instruction_vector_ids = Vec::new();

        for _ in 0..num_new_instruction_vectors {
            new_instruction_vector_ids.push(self.new_instruction_vector(false));
        }

        self.abstract_vectors.push(AbstractVectorNode {
            size,
            instruction_vector_ids: new_instruction_vector_ids,
        });

        self.abstract_vectors.len() - 1
    }

    /// Create a new instruction vector (internal)
    fn new_instruction_vector(&mut self, fake: bool) -> usize {
        self.instruction_vectors
            .push(InstructionVectorNode { fake });
        self.instruction_vectors.len() - 1
    }

    /// Create a new instruction operation (internal)
    fn new_instruction_operation(
        &mut self,
        operation: InstructionOperation,
        input_ids: Vec<usize>,
        output_ids: Vec<usize>,
        other_info: Option<HashMap<String, OperationInfo>>,
    ) -> usize {
        // Validate vector IDs
        for id in &input_ids {
            assert!(
                *id < self.instruction_vectors.len(),
                "Invalid input vector ID"
            );
        }
        for id in &output_ids {
            assert!(
                *id < self.instruction_vectors.len(),
                "Invalid output vector ID"
            );
        }

        // Check for output overlap
        let output_set: HashSet<_> = output_ids.iter().collect();
        for op in &self.instruction_operations {
            let existing_outputs: HashSet<_> = op.output_vector_ids.iter().collect();
            assert!(
                output_set.is_disjoint(&existing_outputs),
                "Instruction output vector overlap detected"
            );
        }

        self.instruction_operations.push(InstructionOperationNode {
            operation,
            input_vector_ids: input_ids,
            output_vector_ids: output_ids,
            other_info: other_info.unwrap_or_default(),
        });

        self.instruction_operations.len() - 1
    }

    /// Create a new abstract operation
    pub fn new_abstract_operation(
        &mut self,
        operation: AbstractOperation,
        input_ids: Vec<usize>,
        output_ids: Vec<usize>,
        other_info: Option<HashMap<String, OperationInfo>>,
    ) -> usize {
        let info = other_info.unwrap_or_default();

        // Validate abstract vector IDs
        for id in &input_ids {
            assert!(
                *id < self.abstract_vectors.len(),
                "Invalid abstract input vector ID"
            );
        }
        for id in &output_ids {
            assert!(
                *id < self.abstract_vectors.len(),
                "Invalid abstract output vector ID"
            );
        }

        // Check for output overlap
        let output_set: HashSet<_> = output_ids.iter().collect();
        for op in &self.abstract_operations {
            let existing_outputs: HashSet<_> = op.output_vector_ids.iter().collect();
            assert!(
                output_set.is_disjoint(&existing_outputs),
                "Abstract output vector overlap detected"
            );
        }

        match operation {
            AbstractOperation::Add
            | AbstractOperation::Sub
            | AbstractOperation::Mul
            | AbstractOperation::Sig
            | AbstractOperation::CSig
            | AbstractOperation::SiLU => {
                self.emit_elementwise_operation(operation, &input_ids, &output_ids);
            }
            AbstractOperation::RmsNorm => {
                self.emit_rms_norm_operation(&input_ids, &output_ids, &info);
            }
            AbstractOperation::TMatmul => {
                self.emit_tmatmul_operation(&input_ids, &output_ids, &info);
            }
            AbstractOperation::Ldv | AbstractOperation::Sv => {
                self.emit_memory_operation(operation, &input_ids, &output_ids, &info);
            }
        }

        self.abstract_operations.push(AbstractOperationNode {
            operation,
            input_vector_ids: input_ids,
            output_vector_ids: output_ids,
            other_info: info,
        });

        self.abstract_operations.len() - 1
    }

    /// Emit element-wise operations (add, sub, mul, sig, csig, silu)
    fn emit_elementwise_operation(
        &mut self,
        operation: AbstractOperation,
        input_ids: &[usize],
        output_ids: &[usize],
    ) {
        let inst_op = match operation {
            AbstractOperation::Add => InstructionOperation::Add,
            AbstractOperation::Sub => InstructionOperation::Sub,
            AbstractOperation::Mul => InstructionOperation::Mul,
            AbstractOperation::Sig => InstructionOperation::Sig,
            AbstractOperation::CSig => InstructionOperation::CSig,
            AbstractOperation::SiLU => InstructionOperation::SiLU,
            _ => unreachable!(),
        };

        // Get instruction vectors for all inputs
        let input_inst_vecs: Vec<_> = input_ids
            .iter()
            .map(|&id| self.abstract_vectors[id].instruction_vector_ids.clone())
            .collect();

        // Get instruction vectors for outputs
        let output_inst_vecs: Vec<_> = output_ids
            .iter()
            .map(|&id| self.abstract_vectors[id].instruction_vector_ids.clone())
            .collect();

        // Zip instruction vectors together
        if !input_inst_vecs.is_empty() && !output_inst_vecs.is_empty() {
            let num_slices = input_inst_vecs[0].len();
            for slice_idx in 0..num_slices {
                let inst_inputs: Vec<_> = input_inst_vecs.iter().map(|v| v[slice_idx]).collect();
                let inst_outputs: Vec<_> = output_inst_vecs.iter().map(|v| v[slice_idx]).collect();
                self.new_instruction_operation(inst_op, inst_inputs, inst_outputs, None);
            }
        }
    }

    /// Emit RMS normalization operation
    fn emit_rms_norm_operation(
        &mut self,
        input_ids: &[usize],
        output_ids: &[usize],
        info: &HashMap<String, OperationInfo>,
    ) {
        let rms_length = info
            .get("rms_length")
            .and_then(|v| v.as_int())
            .unwrap_or(self.config.d as i64) as usize;

        let rms_num_slices = (rms_length + self.config.d - 1) / self.config.d;

        let mut rms_base_info = HashMap::new();
        rms_base_info.insert(
            "rms_id".to_string(),
            OperationInfo::Int(self.abstract_operations.len() as i64),
        );
        rms_base_info.insert(
            "rms_num_slices".to_string(),
            OperationInfo::Int(rms_num_slices as i64),
        );

        // Collect instruction vector IDs first to avoid borrow issues
        let input_inst_vec_ids: Vec<Vec<usize>> = input_ids
            .iter()
            .map(|&id| self.abstract_vectors[id].instruction_vector_ids.clone())
            .collect();

        // Create fake accumulate vectors
        let mut fake_accumulate_vectors = Vec::new();
        for inst_vecs in &input_inst_vec_ids {
            for &inst_vec_id in inst_vecs {
                let new_fake = self.new_instruction_vector(true);
                fake_accumulate_vectors.push(new_fake);
                self.new_instruction_operation(
                    InstructionOperation::RmsAccumulate,
                    vec![inst_vec_id],
                    vec![new_fake],
                    Some(rms_base_info.clone()),
                );
            }
        }

        // Create fake RMS value
        let fake_rms_value = self.new_instruction_vector(true);
        let mut finish_info = rms_base_info.clone();
        finish_info.insert(
            "rms_length".to_string(),
            OperationInfo::Int(rms_length as i64),
        );
        self.new_instruction_operation(
            InstructionOperation::RmsFinishAccumulate,
            fake_accumulate_vectors,
            vec![fake_rms_value],
            Some(finish_info),
        );

        // Collect output instruction vector IDs
        let output_inst_vec_ids: Vec<Vec<usize>> = output_ids
            .iter()
            .map(|&id| self.abstract_vectors[id].instruction_vector_ids.clone())
            .collect();

        // Emit RMS norm for each slice
        for (input_vecs, output_vecs) in input_inst_vec_ids.iter().zip(output_inst_vec_ids.iter()) {
            for (&input_inst_id, &output_inst_id) in input_vecs.iter().zip(output_vecs.iter()) {
                self.new_instruction_operation(
                    InstructionOperation::RmsNorm,
                    vec![input_inst_id, fake_rms_value],
                    vec![output_inst_id],
                    Some(rms_base_info.clone()),
                );
            }
        }
    }

    /// Emit TMatmul operation
    fn emit_tmatmul_operation(
        &mut self,
        input_ids: &[usize],
        output_ids: &[usize],
        info: &HashMap<String, OperationInfo>,
    ) {
        let address_label = info
            .get("address_label")
            .and_then(|v| v.as_string())
            .unwrap_or("matrix")
            .to_string();

        self.abstract_matrix_address_labels
            .push(address_label.clone());

        let num_rows = info
            .get("NumRows")
            .and_then(|v| v.as_int())
            .unwrap_or(self.config.d as i64) as usize;
        let num_columns = info
            .get("NumColumns")
            .and_then(|v| v.as_int())
            .unwrap_or(self.config.d as i64) as usize;

        let num_row_blocks = (num_rows + self.config.d - 1) / self.config.d;
        let num_column_blocks = (num_columns + self.config.d - 1) / self.config.d;

        // Clone instruction vector IDs to avoid borrow issues
        let input_inst_vec_ids = self.abstract_vectors[input_ids[0]]
            .instruction_vector_ids
            .clone();
        let output_inst_vec_ids = self.abstract_vectors[output_ids[0]]
            .instruction_vector_ids
            .clone();

        // Create import fake vectors
        let mut import_fakes = Vec::with_capacity(num_column_blocks);
        for _ in 0..num_column_blocks {
            import_fakes.push(self.new_instruction_vector(true));
        }

        // Create export fake vectors
        let mut export_fakes = Vec::with_capacity(num_row_blocks);
        for _ in 0..num_row_blocks {
            let mut row = Vec::with_capacity(num_column_blocks);
            for _ in 0..num_column_blocks {
                row.push(self.new_instruction_vector(true));
            }
            export_fakes.push(row);
        }

        // Create addend registers
        let mut addend_registers = Vec::with_capacity(num_row_blocks);
        for _ in 0..num_row_blocks {
            let mut row = Vec::with_capacity(num_column_blocks);
            for _ in 0..num_column_blocks {
                row.push(self.new_instruction_vector(false));
            }
            addend_registers.push(row);
        }

        // Create sum registers (for num_column_blocks > 2)
        let mut sum_registers: Vec<Vec<usize>> = Vec::with_capacity(num_row_blocks);
        if num_column_blocks > 2 {
            for _ in 0..num_row_blocks {
                let mut row = Vec::with_capacity(num_column_blocks.saturating_sub(2));
                for _ in 0..num_column_blocks.saturating_sub(2) {
                    row.push(self.new_instruction_vector(false));
                }
                sum_registers.push(row);
            }
        } else {
            for _ in 0..num_row_blocks {
                sum_registers.push(Vec::new());
            }
        }

        for c in 0..num_column_blocks {
            let mut tmatmul_base_info = HashMap::new();
            tmatmul_base_info.insert(
                "tmatmul_id".to_string(),
                OperationInfo::String(format!("id_{}_c_{}", self.abstract_operations.len(), c)),
            );
            tmatmul_base_info.insert(
                "tmatmul_num_row_blocks".to_string(),
                OperationInfo::Int(num_row_blocks as i64),
            );

            // Import
            self.new_instruction_operation(
                InstructionOperation::TMatmulImport,
                vec![input_inst_vec_ids[c]],
                vec![import_fakes[c]],
                Some(tmatmul_base_info.clone()),
            );

            for r in 0..num_row_blocks {
                let block_label = Self::matrix_name_to_block_name(&address_label, r, c);
                self.instruction_matrix_address_labels
                    .push(block_label.clone());

                let mut info_go = tmatmul_base_info.clone();
                info_go.insert(
                    "address_label".to_string(),
                    OperationInfo::String(block_label),
                );

                // Go
                self.new_instruction_operation(
                    InstructionOperation::TMatmulGo,
                    vec![import_fakes[c]],
                    vec![export_fakes[r][c]],
                    Some(info_go),
                );

                // Export and accumulate
                if num_column_blocks == 1 {
                    // No summing needed
                    self.new_instruction_operation(
                        InstructionOperation::TMatmulExport,
                        vec![export_fakes[r][c]],
                        vec![output_inst_vec_ids[r]],
                        Some(tmatmul_base_info.clone()),
                    );
                } else if num_column_blocks == 2 {
                    // Sum directly to output
                    self.new_instruction_operation(
                        InstructionOperation::TMatmulExport,
                        vec![export_fakes[r][c]],
                        vec![addend_registers[r][c]],
                        Some(tmatmul_base_info.clone()),
                    );
                    if c == 1 {
                        self.new_instruction_operation(
                            InstructionOperation::Add,
                            vec![addend_registers[r][c - 1], addend_registers[r][c]],
                            vec![output_inst_vec_ids[r]],
                            Some(tmatmul_base_info.clone()),
                        );
                    }
                } else {
                    // Need sum registers
                    self.new_instruction_operation(
                        InstructionOperation::TMatmulExport,
                        vec![export_fakes[r][c]],
                        vec![addend_registers[r][c]],
                        Some(tmatmul_base_info.clone()),
                    );

                    if c == 0 {
                        // First column, nothing to add yet
                    } else if c == 1 {
                        self.new_instruction_operation(
                            InstructionOperation::Add,
                            vec![addend_registers[r][c - 1], addend_registers[r][c]],
                            vec![sum_registers[r][c - 1]],
                            Some(tmatmul_base_info.clone()),
                        );
                    } else if c == num_column_blocks - 1 {
                        // Last column
                        self.new_instruction_operation(
                            InstructionOperation::Add,
                            vec![sum_registers[r][c - 2], addend_registers[r][c]],
                            vec![output_inst_vec_ids[r]],
                            Some(tmatmul_base_info.clone()),
                        );
                    } else {
                        self.new_instruction_operation(
                            InstructionOperation::Add,
                            vec![sum_registers[r][c - 2], addend_registers[r][c]],
                            vec![sum_registers[r][c - 1]],
                            Some(tmatmul_base_info.clone()),
                        );
                    }
                }
            }
        }
    }

    /// Emit memory operation (ldv, sv)
    fn emit_memory_operation(
        &mut self,
        operation: AbstractOperation,
        input_ids: &[usize],
        output_ids: &[usize],
        info: &HashMap<String, OperationInfo>,
    ) {
        let address_label = info
            .get("address_label")
            .and_then(|v| v.as_string())
            .unwrap_or("mem")
            .to_string();

        self.abstract_vector_address_labels
            .push(address_label.clone());

        let vector_ids = if operation == AbstractOperation::Sv {
            input_ids
        } else {
            output_ids
        };

        // Collect instruction vector IDs to avoid borrow issues
        let inst_vecs: Vec<Vec<usize>> = vector_ids
            .iter()
            .map(|&av| self.abstract_vectors[av].instruction_vector_ids.clone())
            .collect();

        for inst_vec_ids in inst_vecs {
            for (j, iv) in inst_vec_ids.iter().enumerate() {
                let slice_label = Self::vector_name_to_slice_name(&address_label, j);
                self.instruction_vector_address_labels
                    .push(slice_label.clone());

                let mut slice_info = HashMap::new();
                slice_info.insert(
                    "address_label".to_string(),
                    OperationInfo::String(slice_label),
                );

                if operation == AbstractOperation::Ldv {
                    self.new_instruction_operation(
                        InstructionOperation::Ldv,
                        vec![],
                        vec![*iv],
                        Some(slice_info),
                    );
                } else {
                    self.new_instruction_operation(
                        InstructionOperation::Sv,
                        vec![*iv],
                        vec![],
                        Some(slice_info),
                    );
                }
            }
        }
    }

    /// Get operation duration in cycles
    pub fn operation_duration(&self, operation: InstructionOperation, parallel: bool) -> usize {
        match operation {
            InstructionOperation::Add => 3 * self.config.d / self.config.vector_parallelism,
            InstructionOperation::Sub => 3 * self.config.d / self.config.vector_parallelism,
            InstructionOperation::Mul => 3 * self.config.d / self.config.vector_parallelism,
            InstructionOperation::Sig => 2 * self.config.d / self.config.lut_parallelism,
            InstructionOperation::CSig => 2 * self.config.d / self.config.lut_parallelism,
            InstructionOperation::SiLU => 2 * self.config.d / self.config.lut_parallelism,
            InstructionOperation::RmsAccumulate => self.config.d / self.config.vector_parallelism,
            InstructionOperation::RmsFinishAccumulate => 4 * self.config.fixed_point_precision,
            InstructionOperation::RmsNorm => 2 * self.config.d / self.config.vector_parallelism,
            InstructionOperation::TMatmulImport => self.config.d / self.config.vector_parallelism,
            InstructionOperation::TMatmulGo => {
                if parallel {
                    1
                } else {
                    let unbounded = self.config.d * self.config.d / self.config.tmatmul_parallelism;
                    let bounded =
                        self.config.matrix_size_in_bytes() / self.config.max_bytes_per_cycle();
                    unbounded.max(bounded)
                }
            }
            InstructionOperation::TMatmulExport => self.config.d / self.config.vector_parallelism,
            InstructionOperation::Ldv => {
                let unbounded = self.config.d / self.config.vector_parallelism;
                let bounded =
                    self.config.vector_size_in_bytes() / self.config.max_bytes_per_cycle();
                unbounded.max(bounded)
            }
            InstructionOperation::Sv => {
                let unbounded = self.config.d / self.config.vector_parallelism;
                let bounded =
                    self.config.vector_size_in_bytes() / self.config.max_bytes_per_cycle();
                unbounded.max(bounded)
            }
        }
    }

    /// Get instruction operation dependencies
    pub fn get_instruction_operation_dependencies(&self) -> Vec<HashSet<usize>> {
        let num_ops = self.instruction_operations.len();
        let mut dependencies = vec![HashSet::new(); num_ops];

        // Build producer map: vector_id -> operation that produces it
        let mut producer_map: HashMap<usize, usize> = HashMap::new();
        for (op_idx, op) in self.instruction_operations.iter().enumerate() {
            for &out_id in &op.output_vector_ids {
                producer_map.insert(out_id, op_idx);
            }
        }

        // Calculate dependencies recursively
        fn calc_deps(
            op_idx: usize,
            ops: &[InstructionOperationNode],
            producer_map: &HashMap<usize, usize>,
            deps: &mut Vec<HashSet<usize>>,
        ) -> HashSet<usize> {
            if !deps[op_idx].is_empty() {
                return deps[op_idx].clone();
            }

            let mut result = HashSet::new();
            for &input_id in &ops[op_idx].input_vector_ids {
                if let Some(&producer_idx) = producer_map.get(&input_id) {
                    result.insert(producer_idx);
                    result.extend(calc_deps(producer_idx, ops, producer_map, deps));
                }
            }
            deps[op_idx] = result.clone();
            result
        }

        for i in 0..num_ops {
            calc_deps(
                i,
                &self.instruction_operations,
                &producer_map,
                &mut dependencies,
            );
        }

        dependencies
    }

    /// Create instruction ordering using scheduling algorithm
    pub fn create_instruction_ordering(&self) -> Vec<usize> {
        let total_ops = self.instruction_operations.len();
        let dependencies = self.get_instruction_operation_dependencies();
        let mut ordering = Vec::new();
        let mut scheduled: HashSet<usize> = HashSet::new();

        // Track tmatmul and rms state
        let mut active_tmatmul_id: Option<String> = None;
        let mut tmatmul_num_row_blocks_remaining = 0;
        let mut previous_tmatmul_op: Option<InstructionOperation> = None;

        while ordering.len() < total_ops {
            let next_op = self.next_instruction_to_add(
                &ordering,
                &dependencies,
                &scheduled,
                &mut active_tmatmul_id,
                &mut tmatmul_num_row_blocks_remaining,
                &mut previous_tmatmul_op,
            );

            match next_op {
                Some(op_idx) => {
                    ordering.push(op_idx);
                    scheduled.insert(op_idx);

                    // Update tmatmul state
                    let op = &self.instruction_operations[op_idx];
                    match op.operation {
                        InstructionOperation::TMatmulImport => {
                            active_tmatmul_id = op
                                .other_info
                                .get("tmatmul_id")
                                .and_then(|v| v.as_string())
                                .map(|s| s.to_string());
                            tmatmul_num_row_blocks_remaining =
                                op.other_info
                                    .get("tmatmul_num_row_blocks")
                                    .and_then(|v| v.as_int())
                                    .unwrap_or(0) as usize;
                            previous_tmatmul_op = Some(InstructionOperation::TMatmulImport);
                        }
                        InstructionOperation::TMatmulGo => {
                            previous_tmatmul_op = Some(InstructionOperation::TMatmulGo);
                        }
                        InstructionOperation::TMatmulExport => {
                            tmatmul_num_row_blocks_remaining =
                                tmatmul_num_row_blocks_remaining.saturating_sub(1);
                            if tmatmul_num_row_blocks_remaining == 0 {
                                active_tmatmul_id = None;
                            }
                            previous_tmatmul_op = Some(InstructionOperation::TMatmulExport);
                        }
                        _ => {}
                    }
                }
                None => {
                    eprintln!("Warning: Could not schedule all instructions");
                    break;
                }
            }
        }

        ordering
    }

    /// Find next instruction to add to ordering
    fn next_instruction_to_add(
        &self,
        partial_ordering: &[usize],
        dependencies: &[HashSet<usize>],
        scheduled: &HashSet<usize>,
        active_tmatmul_id: &mut Option<String>,
        tmatmul_num_row_blocks_remaining: &mut usize,
        previous_tmatmul_op: &mut Option<InstructionOperation>,
    ) -> Option<usize> {
        let all_ops: HashSet<usize> = (0..self.instruction_operations.len()).collect();
        let mut candidates: HashSet<usize> = all_ops.difference(scheduled).copied().collect();

        // Find available inputs
        let mut available_inputs: HashSet<usize> = HashSet::new();
        for &op_idx in partial_ordering {
            for &out_id in &self.instruction_operations[op_idx].output_vector_ids {
                available_inputs.insert(out_id);
            }
        }

        // Filter to runnable operations (all inputs available)
        candidates.retain(|&op_idx| {
            let op = &self.instruction_operations[op_idx];
            op.input_vector_ids.iter().all(|&in_id| {
                available_inputs.contains(&in_id) || self.instruction_vectors[in_id].fake
            })
        });

        // Apply tmatmul state constraints
        if let Some(ref active_id) = active_tmatmul_id {
            // Filter tmatmul ops to those matching active ID
            candidates.retain(|&op_idx| {
                let op = &self.instruction_operations[op_idx];
                match op.operation {
                    InstructionOperation::TMatmulImport
                    | InstructionOperation::TMatmulGo
                    | InstructionOperation::TMatmulExport => op
                        .other_info
                        .get("tmatmul_id")
                        .and_then(|v| v.as_string())
                        .map(|s| s == active_id)
                        .unwrap_or(false),
                    _ => true,
                }
            });

            // Apply state machine constraints
            if let Some(prev_op) = previous_tmatmul_op {
                match prev_op {
                    InstructionOperation::TMatmulImport => {
                        // After import: only go is allowed
                        candidates.retain(|&op_idx| {
                            let op = &self.instruction_operations[op_idx];
                            op.operation != InstructionOperation::TMatmulImport
                                && op.operation != InstructionOperation::TMatmulExport
                        });
                    }
                    InstructionOperation::TMatmulGo => {
                        // After go: only export is allowed for tmatmul ops
                        candidates.retain(|&op_idx| {
                            let op = &self.instruction_operations[op_idx];
                            op.operation != InstructionOperation::TMatmulImport
                                && op.operation != InstructionOperation::TMatmulGo
                        });
                    }
                    InstructionOperation::TMatmulExport => {
                        // After export: can do import (new column) or go (same column)
                        candidates.retain(|&op_idx| {
                            let op = &self.instruction_operations[op_idx];
                            op.operation != InstructionOperation::TMatmulExport
                        });
                    }
                    _ => {}
                }
            }
        } else {
            // No active tmatmul: can't run go or export
            candidates.retain(|&op_idx| {
                let op = &self.instruction_operations[op_idx];
                op.operation != InstructionOperation::TMatmulGo
                    && op.operation != InstructionOperation::TMatmulExport
            });
        }

        // Prioritize tmatmul_import and tmatmul_go
        let tmatmul_priority: Vec<_> = candidates
            .iter()
            .filter(|&&op_idx| {
                let op = &self.instruction_operations[op_idx];
                op.operation == InstructionOperation::TMatmulImport
                    || op.operation == InstructionOperation::TMatmulGo
            })
            .copied()
            .collect();

        if !tmatmul_priority.is_empty() {
            return Some(tmatmul_priority[0]);
        }

        // Return any available candidate
        candidates.into_iter().next()
    }

    /// Create register mapping for instruction ordering
    pub fn create_register_mapping(&self, ordering: &[usize]) -> Vec<RegisterAssignment> {
        let num_regs = self.config.num_vector_registers;
        let mut registers: Vec<Option<usize>> = vec![None; num_regs];
        let mut assignments: Vec<RegisterAssignment> = self
            .instruction_vectors
            .iter()
            .map(|_| RegisterAssignment::new(None))
            .collect();

        for (pc, &op_id) in ordering.iter().enumerate() {
            let op = &self.instruction_operations[op_id];

            // Handle inputs that need to be swapped back in
            for &vin_id in &op.input_vector_ids {
                if self.instruction_vectors[vin_id].fake {
                    continue;
                }
                if !registers.contains(&Some(vin_id)) {
                    // Need to swap in
                    let (dst_reg, needs_swap_out) =
                        self.allocate_register(&mut registers, ordering, pc);

                    if needs_swap_out {
                        if let Some(victim_id) = registers[dst_reg] {
                            assignments[victim_id].swaps.push(RegisterSwapInfo {
                                previous_register_address: dst_reg,
                                swap_out_before: pc,
                                swap_address_label: format!("swap_pc_{}_vec_{}", pc, victim_id),
                                swap_in_before: None,
                                new_register_address: None,
                            });
                        }
                    }

                    if let Some(last_swap) = assignments[vin_id].swaps.last_mut() {
                        last_swap.swap_in_before = Some(pc);
                        last_swap.new_register_address = Some(dst_reg);
                    }
                    registers[dst_reg] = Some(vin_id);
                }
            }

            // Handle outputs
            for &vout_id in &op.output_vector_ids {
                if self.instruction_vectors[vout_id].fake {
                    continue;
                }

                let (dst_reg, needs_swap_out) =
                    self.allocate_register(&mut registers, ordering, pc);

                if needs_swap_out {
                    if let Some(victim_id) = registers[dst_reg] {
                        assignments[victim_id].swaps.push(RegisterSwapInfo {
                            previous_register_address: dst_reg,
                            swap_out_before: pc,
                            swap_address_label: format!("swap_pc_{}_vec_{}", pc, victim_id),
                            swap_in_before: None,
                            new_register_address: None,
                        });
                    }
                }

                assignments[vout_id].initial_register_address = Some(dst_reg);
                registers[dst_reg] = Some(vout_id);
            }
        }

        assignments
    }

    /// Allocate a register, returns (register_index, needs_swap_out)
    fn allocate_register(
        &self,
        registers: &mut [Option<usize>],
        ordering: &[usize],
        pc: usize,
    ) -> (usize, bool) {
        // Find free register
        for (i, r) in registers.iter().enumerate() {
            if r.is_none() {
                return (i, false);
            }

            // Check if register value is no longer needed
            let vec_id = r.unwrap();
            let mut still_needed = false;
            for &future_op_id in &ordering[pc..] {
                let future_op = &self.instruction_operations[future_op_id];
                if future_op.input_vector_ids.contains(&vec_id) {
                    still_needed = true;
                    break;
                }
            }
            if !still_needed {
                return (i, false);
            }
        }

        // Need to spill - find victim with latest next use
        let mut best_victim = 0;
        let mut best_time = 0;

        for (i, r) in registers.iter().enumerate() {
            if let Some(vec_id) = r {
                let mut time_until_use = 0;
                for &future_op_id in &ordering[pc..] {
                    let future_op = &self.instruction_operations[future_op_id];
                    if future_op.input_vector_ids.contains(vec_id) {
                        break;
                    }
                    time_until_use += self.operation_duration(future_op.operation, true);
                }
                if time_until_use > best_time {
                    best_time = time_until_use;
                    best_victim = i;
                }
            }
        }

        (best_victim, true)
    }

    /// Create assembly tokens from ordering and register mapping
    pub fn create_assembly_tokens(
        &self,
        ordering: &[usize],
        register_mapping: &[RegisterAssignment],
    ) -> Vec<Vec<String>> {
        let mut out = Vec::new();

        for (pc, &op_id) in ordering.iter().enumerate() {
            let op = &self.instruction_operations[op_id];

            // Emit swap outs
            for (vec_id, assign) in register_mapping.iter().enumerate() {
                for s in &assign.swaps {
                    if s.swap_out_before == pc {
                        out.push(vec![
                            "sv".to_string(),
                            format!("v{}", s.previous_register_address),
                            s.swap_address_label.clone(),
                        ]);
                    }
                }
            }

            // Emit swap ins
            for assign in register_mapping.iter() {
                for s in &assign.swaps {
                    if s.swap_in_before == Some(pc) {
                        if let Some(new_reg) = s.new_register_address {
                            out.push(vec![
                                "ldv".to_string(),
                                format!("v{}", new_reg),
                                s.swap_address_label.clone(),
                            ]);
                        }
                    }
                }
            }

            // Emit instruction
            let tokens = match op.operation {
                InstructionOperation::Add
                | InstructionOperation::Sub
                | InstructionOperation::Mul => {
                    vec![
                        op.operation.to_string(),
                        register_mapping[op.output_vector_ids[0]].register_name_at_pc(pc),
                        register_mapping[op.input_vector_ids[0]].register_name_at_pc(pc),
                        register_mapping[op.input_vector_ids[1]].register_name_at_pc(pc),
                    ]
                }
                InstructionOperation::Sig
                | InstructionOperation::CSig
                | InstructionOperation::SiLU => {
                    vec![
                        op.operation.to_string(),
                        register_mapping[op.output_vector_ids[0]].register_name_at_pc(pc),
                        register_mapping[op.input_vector_ids[0]].register_name_at_pc(pc),
                    ]
                }
                InstructionOperation::RmsAccumulate => {
                    vec![
                        "rms_accumulate".to_string(),
                        register_mapping[op.input_vector_ids[0]].register_name_at_pc(pc),
                    ]
                }
                InstructionOperation::RmsNorm => {
                    vec![
                        "rms_norm".to_string(),
                        register_mapping[op.output_vector_ids[0]].register_name_at_pc(pc),
                        register_mapping[op.input_vector_ids[0]].register_name_at_pc(pc),
                    ]
                }
                InstructionOperation::RmsFinishAccumulate => {
                    let rms_length = op
                        .other_info
                        .get("rms_length")
                        .and_then(|v| v.as_int())
                        .unwrap_or(0);
                    vec!["rms_finish_accumulate".to_string(), rms_length.to_string()]
                }
                InstructionOperation::TMatmulImport => {
                    vec![
                        "tmatmul_import".to_string(),
                        register_mapping[op.input_vector_ids[0]].register_name_at_pc(pc),
                    ]
                }
                InstructionOperation::TMatmulGo => {
                    let addr = op
                        .other_info
                        .get("address_label")
                        .and_then(|v| v.as_string())
                        .unwrap_or("unknown");
                    vec!["tmatmul_go".to_string(), addr.to_string()]
                }
                InstructionOperation::TMatmulExport => {
                    vec![
                        "tmatmul_export".to_string(),
                        register_mapping[op.output_vector_ids[0]].register_name_at_pc(pc),
                    ]
                }
                InstructionOperation::Ldv => {
                    let addr = op
                        .other_info
                        .get("address_label")
                        .and_then(|v| v.as_string())
                        .unwrap_or("unknown");
                    vec![
                        "ldv".to_string(),
                        register_mapping[op.output_vector_ids[0]].register_name_at_pc(pc),
                        addr.to_string(),
                    ]
                }
                InstructionOperation::Sv => {
                    let addr = op
                        .other_info
                        .get("address_label")
                        .and_then(|v| v.as_string())
                        .unwrap_or("unknown");
                    vec![
                        "sv".to_string(),
                        register_mapping[op.input_vector_ids[0]].register_name_at_pc(pc),
                        addr.to_string(),
                    ]
                }
            };
            out.push(tokens);
        }

        out
    }

    /// Print summary of the algorithm tree
    pub fn print_summary(&self) {
        println!("=== AlgorithmTree Summary ===");
        println!("Abstract vectors: {}", self.abstract_vectors.len());
        println!("Instruction vectors: {}", self.instruction_vectors.len());
        println!("Abstract operations: {}", self.abstract_operations.len());
        println!(
            "Instruction operations: {}",
            self.instruction_operations.len()
        );

        println!("\nAbstract Operations:");
        for (i, op) in self.abstract_operations.iter().enumerate() {
            println!(
                "  [{}] {}: inputs={:?}, outputs={:?}",
                i, op.operation, op.input_vector_ids, op.output_vector_ids
            );
        }

        println!("\nOperation durations:");
        println!(
            "  add delay = {}",
            self.operation_duration(InstructionOperation::Add, false)
        );
        println!(
            "  tmatmul_go parallel delay = {}",
            self.operation_duration(InstructionOperation::TMatmulGo, true)
        );
        println!(
            "  tmatmul_go nonparallel delay = {}",
            self.operation_duration(InstructionOperation::TMatmulGo, false)
        );
    }

    /// Generate assembly code string
    pub fn generate_assembly(&self) -> String {
        let ordering = self.create_instruction_ordering();
        let register_mapping = self.create_register_mapping(&ordering);
        let tokens = self.create_assembly_tokens(&ordering, &register_mapping);

        let mut output = String::new();
        output.push_str("; TMatmul Assembly - Generated by hetGPU\n");
        output.push_str("; Vector registers: v0..v7\n");
        output.push_str(";\n");

        for token_line in tokens {
            let line = format!("    {}\n", token_line.join(" "));
            output.push_str(&line);
        }

        output
    }
}

// =============================================================================
// PTX Integration Module
// =============================================================================

/// PTX to AlgorithmTree converter
/// Bridges PTX AST operations to TMatmul algorithm tree for scheduling
pub struct PtxToAlgorithmTreeConverter {
    /// The algorithm tree being built
    pub tree: AlgorithmTree,
    /// Mapping from PTX variable names to abstract vector IDs
    var_to_vector: HashMap<String, usize>,
    /// Track vector sizes from PTX declarations
    var_sizes: HashMap<String, usize>,
    /// Default vector size when not specified
    default_vector_size: usize,
}

impl PtxToAlgorithmTreeConverter {
    /// Create a new converter with given configuration
    pub fn new(config: TMatmulConfig) -> Self {
        Self {
            tree: AlgorithmTree::new(config),
            var_to_vector: HashMap::new(),
            var_sizes: HashMap::new(),
            default_vector_size: 64,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(TMatmulConfig::default())
    }

    /// Set default vector size
    pub fn set_default_vector_size(&mut self, size: usize) {
        self.default_vector_size = size;
    }

    /// Declare a vector variable
    pub fn declare_vector(&mut self, name: &str, size: usize) -> usize {
        if let Some(&id) = self.var_to_vector.get(name) {
            return id;
        }
        let id = self.tree.new_abstract_vector(size);
        self.var_to_vector.insert(name.to_string(), id);
        self.var_sizes.insert(name.to_string(), size);
        id
    }

    /// Get or create vector for a variable
    pub fn get_or_create_vector(&mut self, name: &str) -> usize {
        if let Some(&id) = self.var_to_vector.get(name) {
            return id;
        }
        let size = self
            .var_sizes
            .get(name)
            .copied()
            .unwrap_or(self.default_vector_size);
        self.declare_vector(name, size)
    }

    /// Load vector from memory
    pub fn emit_load(&mut self, dst: &str, address_label: &str) -> usize {
        let dst_id = self.get_or_create_vector(dst);
        let mut info = HashMap::new();
        info.insert(
            "address_label".to_string(),
            OperationInfo::String(address_label.to_string()),
        );
        self.tree
            .new_abstract_operation(AbstractOperation::Ldv, vec![], vec![dst_id], Some(info))
    }

    /// Store vector to memory
    pub fn emit_store(&mut self, src: &str, address_label: &str) -> usize {
        let src_id = self.get_or_create_vector(src);
        let mut info = HashMap::new();
        info.insert(
            "address_label".to_string(),
            OperationInfo::String(address_label.to_string()),
        );
        self.tree
            .new_abstract_operation(AbstractOperation::Sv, vec![src_id], vec![], Some(info))
    }

    /// Element-wise add: dst = src1 + src2
    pub fn emit_add(&mut self, dst: &str, src1: &str, src2: &str) -> usize {
        let dst_id = self.get_or_create_vector(dst);
        let src1_id = self.get_or_create_vector(src1);
        let src2_id = self.get_or_create_vector(src2);
        self.tree.new_abstract_operation(
            AbstractOperation::Add,
            vec![src1_id, src2_id],
            vec![dst_id],
            None,
        )
    }

    /// Element-wise subtract: dst = src1 - src2
    pub fn emit_sub(&mut self, dst: &str, src1: &str, src2: &str) -> usize {
        let dst_id = self.get_or_create_vector(dst);
        let src1_id = self.get_or_create_vector(src1);
        let src2_id = self.get_or_create_vector(src2);
        self.tree.new_abstract_operation(
            AbstractOperation::Sub,
            vec![src1_id, src2_id],
            vec![dst_id],
            None,
        )
    }

    /// Element-wise multiply: dst = src1 * src2
    pub fn emit_mul(&mut self, dst: &str, src1: &str, src2: &str) -> usize {
        let dst_id = self.get_or_create_vector(dst);
        let src1_id = self.get_or_create_vector(src1);
        let src2_id = self.get_or_create_vector(src2);
        self.tree.new_abstract_operation(
            AbstractOperation::Mul,
            vec![src1_id, src2_id],
            vec![dst_id],
            None,
        )
    }

    /// Sigmoid activation: dst = sigmoid(src)
    pub fn emit_sigmoid(&mut self, dst: &str, src: &str) -> usize {
        let dst_id = self.get_or_create_vector(dst);
        let src_id = self.get_or_create_vector(src);
        self.tree
            .new_abstract_operation(AbstractOperation::Sig, vec![src_id], vec![dst_id], None)
    }

    /// SiLU activation: dst = silu(src)
    pub fn emit_silu(&mut self, dst: &str, src: &str) -> usize {
        let dst_id = self.get_or_create_vector(dst);
        let src_id = self.get_or_create_vector(src);
        self.tree
            .new_abstract_operation(AbstractOperation::SiLU, vec![src_id], vec![dst_id], None)
    }

    /// RMS normalization: dst = rms_norm(src)
    pub fn emit_rms_norm(&mut self, dst: &str, src: &str, rms_length: Option<usize>) -> usize {
        let dst_id = self.get_or_create_vector(dst);
        let src_id = self.get_or_create_vector(src);
        let mut info = HashMap::new();
        if let Some(len) = rms_length {
            info.insert("rms_length".to_string(), OperationInfo::Int(len as i64));
        }
        self.tree.new_abstract_operation(
            AbstractOperation::RmsNorm,
            vec![src_id],
            vec![dst_id],
            if info.is_empty() { None } else { Some(info) },
        )
    }

    /// Ternary matrix multiplication: dst = W @ src
    pub fn emit_tmatmul(
        &mut self,
        dst: &str,
        src: &str,
        weight_label: &str,
        num_rows: usize,
        num_columns: usize,
    ) -> usize {
        let dst_id = self.get_or_create_vector(dst);
        let src_id = self.get_or_create_vector(src);

        let mut info = HashMap::new();
        info.insert(
            "address_label".to_string(),
            OperationInfo::String(weight_label.to_string()),
        );
        info.insert("NumRows".to_string(), OperationInfo::Int(num_rows as i64));
        info.insert(
            "NumColumns".to_string(),
            OperationInfo::Int(num_columns as i64),
        );

        self.tree.new_abstract_operation(
            AbstractOperation::TMatmul,
            vec![src_id],
            vec![dst_id],
            Some(info),
        )
    }

    /// Fused multiply-add: dst = src1 * src2 + src3
    /// Emits as mul followed by add
    pub fn emit_fma(&mut self, dst: &str, src1: &str, src2: &str, src3: &str) -> usize {
        // Create temp vector for intermediate result
        let temp_name = format!("_fma_temp_{}", self.var_to_vector.len());
        let temp_id = self.declare_vector(&temp_name, self.default_vector_size);

        let src1_id = self.get_or_create_vector(src1);
        let src2_id = self.get_or_create_vector(src2);
        let src3_id = self.get_or_create_vector(src3);
        let dst_id = self.get_or_create_vector(dst);

        // mul: temp = src1 * src2
        self.tree.new_abstract_operation(
            AbstractOperation::Mul,
            vec![src1_id, src2_id],
            vec![temp_id],
            None,
        );

        // add: dst = temp + src3
        self.tree.new_abstract_operation(
            AbstractOperation::Add,
            vec![temp_id, src3_id],
            vec![dst_id],
            None,
        )
    }

    /// Build an FFN layer pattern: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    pub fn emit_ffn_layer(
        &mut self,
        input: &str,
        output: &str,
        hidden_dim: usize,
        intermediate_dim: usize,
        gate_weights: &str,
        up_weights: &str,
        down_weights: &str,
    ) {
        // Declare vectors with proper sizes
        let gate_name = format!("_ffn_gate_{}", self.var_to_vector.len());
        let up_name = format!("_ffn_up_{}", self.var_to_vector.len() + 1);
        let gate_silu_name = format!("_ffn_gate_silu_{}", self.var_to_vector.len() + 2);
        let intermediate_name = format!("_ffn_intermediate_{}", self.var_to_vector.len() + 3);

        self.declare_vector(input, hidden_dim);
        self.declare_vector(&gate_name, intermediate_dim);
        self.declare_vector(&up_name, intermediate_dim);
        self.declare_vector(&gate_silu_name, intermediate_dim);
        self.declare_vector(&intermediate_name, intermediate_dim);
        self.declare_vector(output, hidden_dim);

        // gate = gate_proj(x)
        self.emit_tmatmul(
            &gate_name,
            input,
            gate_weights,
            intermediate_dim,
            hidden_dim,
        );

        // up = up_proj(x)
        self.emit_tmatmul(&up_name, input, up_weights, intermediate_dim, hidden_dim);

        // gate_silu = silu(gate)
        self.emit_silu(&gate_silu_name, &gate_name);

        // intermediate = gate_silu * up
        self.emit_mul(&intermediate_name, &gate_silu_name, &up_name);

        // output = down_proj(intermediate)
        self.emit_tmatmul(
            output,
            &intermediate_name,
            down_weights,
            hidden_dim,
            intermediate_dim,
        );
    }

    /// Build a transformer attention layer pattern
    pub fn emit_attention_projection(
        &mut self,
        input: &str,
        output: &str,
        query_weights: &str,
        key_weights: &str,
        value_weights: &str,
        hidden_dim: usize,
        head_dim: usize,
    ) {
        let q_name = format!("_attn_q_{}", self.var_to_vector.len());
        let k_name = format!("_attn_k_{}", self.var_to_vector.len() + 1);
        let v_name = format!("_attn_v_{}", self.var_to_vector.len() + 2);

        self.declare_vector(&q_name, head_dim);
        self.declare_vector(&k_name, head_dim);
        self.declare_vector(&v_name, head_dim);

        // Q = Wq @ input
        self.emit_tmatmul(&q_name, input, query_weights, head_dim, hidden_dim);

        // K = Wk @ input
        self.emit_tmatmul(&k_name, input, key_weights, head_dim, hidden_dim);

        // V = Wv @ input
        self.emit_tmatmul(&v_name, input, value_weights, head_dim, hidden_dim);

        // For now, copy V to output (simplified - real attention is more complex)
        // In practice you'd need softmax(QK^T/sqrt(d)) @ V
        let v_id = self.var_to_vector[&v_name];
        let output_id = self.get_or_create_vector(output);
        self.tree.new_abstract_operation(
            AbstractOperation::Add,
            vec![v_id, v_id],
            vec![output_id],
            None,
        );
    }

    /// Generate assembly from the built tree
    pub fn generate_assembly(&self) -> String {
        self.tree.generate_assembly()
    }

    /// Get summary of the built tree
    pub fn print_summary(&self) {
        self.tree.print_summary();
    }

    /// Consume and return the built algorithm tree
    pub fn into_tree(self) -> AlgorithmTree {
        self.tree
    }
}

/// Pattern matcher for recognizing neural network operations in PTX
pub struct PtxPatternMatcher {
    /// Activation function patterns
    activation_patterns: Vec<ActivationPattern>,
    /// Matrix operation patterns
    matmul_patterns: Vec<MatmulPattern>,
}

/// Recognized activation function pattern
#[derive(Debug, Clone)]
pub struct ActivationPattern {
    pub name: String,
    pub operation: AbstractOperation,
    /// PTX instruction sequence that matches this pattern
    pub instruction_sequence: Vec<String>,
}

/// Recognized matrix multiplication pattern
#[derive(Debug, Clone)]
pub struct MatmulPattern {
    pub name: String,
    /// Expected loop structure
    pub is_loop_matmul: bool,
    /// Uses tensor core MMA instructions
    pub uses_tensor_core: bool,
}

impl PtxPatternMatcher {
    pub fn new() -> Self {
        let mut matcher = Self {
            activation_patterns: Vec::new(),
            matmul_patterns: Vec::new(),
        };
        matcher.register_default_patterns();
        matcher
    }

    fn register_default_patterns(&mut self) {
        // Sigmoid pattern: 1 / (1 + exp(-x))
        self.activation_patterns.push(ActivationPattern {
            name: "sigmoid".to_string(),
            operation: AbstractOperation::Sig,
            instruction_sequence: vec![
                "neg".to_string(),
                "ex2.approx".to_string(),
                "add".to_string(),
                "rcp.approx".to_string(),
            ],
        });

        // SiLU pattern: x * sigmoid(x)
        self.activation_patterns.push(ActivationPattern {
            name: "silu".to_string(),
            operation: AbstractOperation::SiLU,
            instruction_sequence: vec![
                "neg".to_string(),
                "ex2.approx".to_string(),
                "add".to_string(),
                "rcp.approx".to_string(),
                "mul".to_string(),
            ],
        });

        // Tensor core matmul pattern
        self.matmul_patterns.push(MatmulPattern {
            name: "mma_matmul".to_string(),
            is_loop_matmul: true,
            uses_tensor_core: true,
        });

        // Loop-based matmul pattern
        self.matmul_patterns.push(MatmulPattern {
            name: "loop_matmul".to_string(),
            is_loop_matmul: true,
            uses_tensor_core: false,
        });
    }

    /// Check if instruction sequence matches an activation pattern
    pub fn match_activation(&self, instructions: &[&str]) -> Option<&ActivationPattern> {
        for pattern in &self.activation_patterns {
            if instructions.len() >= pattern.instruction_sequence.len() {
                let matches = instructions
                    .iter()
                    .zip(pattern.instruction_sequence.iter())
                    .all(|(inst, pat)| inst.starts_with(pat));
                if matches {
                    return Some(pattern);
                }
            }
        }
        None
    }

    /// Check if a function name suggests matrix multiplication
    pub fn is_matmul_function(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.contains("gemm")
            || lower.contains("matmul")
            || lower.contains("linear")
            || lower.contains("dense")
            || lower.contains("fc")
    }

    /// Check if a function name suggests activation
    pub fn is_activation_function(&self, name: &str) -> Option<AbstractOperation> {
        let lower = name.to_lowercase();
        if lower.contains("relu") {
            // ReLU not directly supported, approximate with sig
            Some(AbstractOperation::Sig)
        } else if lower.contains("sigmoid") {
            Some(AbstractOperation::Sig)
        } else if lower.contains("silu") || lower.contains("swish") {
            Some(AbstractOperation::SiLU)
        } else if lower.contains("gelu") {
            // GELU approximated as SiLU
            Some(AbstractOperation::SiLU)
        } else {
            None
        }
    }
}

impl Default for PtxPatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = TMatmulConfig::default();
        assert_eq!(config.d, 64);
        assert_eq!(config.num_vector_registers, 8);
    }

    #[test]
    fn test_simple_add() {
        let config = TMatmulConfig::default();
        let mut tree = AlgorithmTree::new(config);

        // Create input vectors
        let a = tree.new_abstract_vector(64);
        let b = tree.new_abstract_vector(64);
        let c = tree.new_abstract_vector(64);

        // Load inputs
        let mut info_a = HashMap::new();
        info_a.insert(
            "address_label".to_string(),
            OperationInfo::String("A".to_string()),
        );
        tree.new_abstract_operation(AbstractOperation::Ldv, vec![], vec![a], Some(info_a));

        let mut info_b = HashMap::new();
        info_b.insert(
            "address_label".to_string(),
            OperationInfo::String("B".to_string()),
        );
        tree.new_abstract_operation(AbstractOperation::Ldv, vec![], vec![b], Some(info_b));

        // Add
        tree.new_abstract_operation(AbstractOperation::Add, vec![a, b], vec![c], None);

        // Store output
        let mut info_c = HashMap::new();
        info_c.insert(
            "address_label".to_string(),
            OperationInfo::String("C".to_string()),
        );
        tree.new_abstract_operation(AbstractOperation::Sv, vec![c], vec![], Some(info_c));

        tree.print_summary();

        let asm = tree.generate_assembly();
        assert!(asm.contains("ldv"));
        assert!(asm.contains("add"));
        assert!(asm.contains("sv"));
    }

    #[test]
    fn test_tmatmul() {
        let config = TMatmulConfig {
            d: 32,
            ..Default::default()
        };
        let mut tree = AlgorithmTree::new(config);

        // Create vectors for 64-element input/output (2 blocks)
        let input = tree.new_abstract_vector(64);
        let output = tree.new_abstract_vector(64);

        // Load input
        let mut info_in = HashMap::new();
        info_in.insert(
            "address_label".to_string(),
            OperationInfo::String("X".to_string()),
        );
        tree.new_abstract_operation(AbstractOperation::Ldv, vec![], vec![input], Some(info_in));

        // TMatmul: 64x64 matrix
        let mut info_mat = HashMap::new();
        info_mat.insert(
            "address_label".to_string(),
            OperationInfo::String("W".to_string()),
        );
        info_mat.insert("NumRows".to_string(), OperationInfo::Int(64));
        info_mat.insert("NumColumns".to_string(), OperationInfo::Int(64));
        tree.new_abstract_operation(
            AbstractOperation::TMatmul,
            vec![input],
            vec![output],
            Some(info_mat),
        );

        // Store output
        let mut info_out = HashMap::new();
        info_out.insert(
            "address_label".to_string(),
            OperationInfo::String("Y".to_string()),
        );
        tree.new_abstract_operation(AbstractOperation::Sv, vec![output], vec![], Some(info_out));

        tree.print_summary();

        let asm = tree.generate_assembly();
        println!("{}", asm);
        assert!(asm.contains("tmatmul_import"));
        assert!(asm.contains("tmatmul_go"));
        assert!(asm.contains("tmatmul_export"));
    }

    #[test]
    fn test_ptx_converter_basic() {
        let mut converter = PtxToAlgorithmTreeConverter::with_defaults();

        // Load inputs
        converter.emit_load("a", "A_mem");
        converter.emit_load("b", "B_mem");

        // Add
        converter.emit_add("c", "a", "b");

        // Store
        converter.emit_store("c", "C_mem");

        let asm = converter.generate_assembly();
        println!("PTX Converter Basic:\n{}", asm);
        assert!(asm.contains("ldv"));
        assert!(asm.contains("add"));
        assert!(asm.contains("sv"));
    }

    #[test]
    fn test_ptx_converter_fma() {
        let mut converter = PtxToAlgorithmTreeConverter::with_defaults();

        converter.emit_load("a", "A");
        converter.emit_load("b", "B");
        converter.emit_load("c", "C");

        // d = a * b + c
        converter.emit_fma("d", "a", "b", "c");

        converter.emit_store("d", "D");

        let asm = converter.generate_assembly();
        println!("PTX Converter FMA:\n{}", asm);
        assert!(asm.contains("mul"));
        assert!(asm.contains("add"));
    }

    #[test]
    fn test_ptx_converter_tmatmul() {
        let config = TMatmulConfig {
            d: 32,
            ..Default::default()
        };
        let mut converter = PtxToAlgorithmTreeConverter::new(config);

        converter.declare_vector("x", 64);
        converter.declare_vector("y", 64);

        converter.emit_load("x", "X_mem");
        converter.emit_tmatmul("y", "x", "W", 64, 64);
        converter.emit_store("y", "Y_mem");

        let asm = converter.generate_assembly();
        println!("PTX Converter TMatmul:\n{}", asm);
        assert!(asm.contains("tmatmul_import"));
        assert!(asm.contains("tmatmul_go"));
        assert!(asm.contains("tmatmul_export"));
    }

    #[test]
    fn test_ptx_converter_ffn_layer() {
        let config = TMatmulConfig {
            d: 32,
            ..Default::default()
        };
        let mut converter = PtxToAlgorithmTreeConverter::new(config);

        converter.emit_load("input", "Input");
        converter.emit_ffn_layer(
            "input", "output", 64,  // hidden_dim
            128, // intermediate_dim
            "W_gate", "W_up", "W_down",
        );
        converter.emit_store("output", "Output");

        converter.print_summary();

        let asm = converter.generate_assembly();
        println!("PTX Converter FFN Layer:\n{}", asm);
        // Should have 3 tmatmul operations
        assert!(asm.matches("tmatmul_go").count() >= 3);
        // Should have silu
        assert!(asm.contains("silu"));
    }

    #[test]
    fn test_pattern_matcher() {
        let matcher = PtxPatternMatcher::new();

        // Test matmul detection
        assert!(matcher.is_matmul_function("cublas_gemm"));
        assert!(matcher.is_matmul_function("MatMulKernel"));
        assert!(matcher.is_matmul_function("linear_forward"));
        assert!(!matcher.is_matmul_function("relu_kernel"));

        // Test activation detection
        assert_eq!(
            matcher.is_activation_function("sigmoid"),
            Some(AbstractOperation::Sig)
        );
        assert_eq!(
            matcher.is_activation_function("silu_kernel"),
            Some(AbstractOperation::SiLU)
        );
        assert_eq!(
            matcher.is_activation_function("gelu"),
            Some(AbstractOperation::SiLU)
        );
    }

    #[test]
    fn test_activation_pattern_matching() {
        let matcher = PtxPatternMatcher::new();

        // Sigmoid pattern
        let sigmoid_seq = vec!["neg", "ex2.approx.f32", "add.f32", "rcp.approx.f32"];
        let pattern = matcher.match_activation(&sigmoid_seq);
        assert!(pattern.is_some());
        assert_eq!(pattern.unwrap().name, "sigmoid");

        // SiLU pattern
        let silu_seq = vec![
            "neg",
            "ex2.approx.f32",
            "add.f32",
            "rcp.approx.f32",
            "mul.f32",
        ];
        let pattern = matcher.match_activation(&silu_seq);
        assert!(pattern.is_some());
        assert_eq!(pattern.unwrap().name, "silu");
    }
}
