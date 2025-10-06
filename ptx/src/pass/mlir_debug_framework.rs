// mlir_debug_framework.rs - Universal MLIR Debug Framework
// This module provides comprehensive debug information support for multiple MLIR dialects
// including TOSA, TTIR (Tenstorrent), and future dialects.

use super::*;
use ptx_parser as ast;
use std::collections::{BTreeMap, HashMap};
use std::fmt;

/// Universal debug location that works across all MLIR dialects
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniversalDebugLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub instruction_name: String,
    pub dialect: MlirDialect,
    pub optimization_level: OptimizationLevel,
}

/// Supported MLIR dialects
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirDialect {
    TOSA,
    TTIR,
    Linalg,
    Arith,
    MemRef,
    Func,
    EmitC,
    Custom(String),
}

/// Optimization levels for debug info
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,       // -O0
    Less,       // -O1
    Default,    // -O2
    Aggressive, // -O3
}

/// Enhanced type information for debug metadata
#[derive(Clone)]
pub struct DebugTypeInfo {
    pub ptx_type: ast::Type,
    pub mlir_type: String,
    pub size_bits: u64,
    pub alignment: u32,
    pub memory_space: ast::StateSpace,
    pub is_tensor: bool,
    pub tensor_shape: Option<Vec<i64>>,
    pub element_type: Option<String>,
}

impl fmt::Debug for DebugTypeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DebugTypeInfo")
            .field("ptx_type", &"<ptx_type>") // Placeholder since ast::Type doesn't implement Debug
            .field("mlir_type", &self.mlir_type)
            .field("size_bits", &self.size_bits)
            .field("alignment", &self.alignment)
            .field("memory_space", &self.memory_space)
            .field("is_tensor", &self.is_tensor)
            .field("tensor_shape", &self.tensor_shape)
            .field("element_type", &self.element_type)
            .finish()
    }
}

/// Variable debug information with enhanced metadata
#[derive(Debug, Clone)]
pub struct EnhancedVariableDebugInfo {
    pub name: String,
    pub type_info: DebugTypeInfo,
    pub scope: String,
    pub location: UniversalDebugLocation,
    pub lifetime: VariableLifetime,
    pub usage_pattern: UsagePattern,
    pub memory_layout: MemoryLayout,
}

/// Variable lifetime tracking
#[derive(Debug, Clone)]
pub enum VariableLifetime {
    Function,    // Lives for the entire function
    Block,       // Lives for a specific block
    Instruction, // Temporary (single instruction)
    Parameter,   // Function parameter
    Global,      // Global variable
}

/// Usage pattern analysis
#[derive(Debug, Clone)]
pub struct UsagePattern {
    pub read_count: u32,
    pub write_count: u32,
    pub is_loop_invariant: bool,
    pub is_vectorizable: bool,
    pub memory_access_pattern: MemoryAccessPattern,
}

#[derive(Debug, Clone)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided(i64),
    Broadcast,
    Reduction,
}

/// Memory layout information
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    pub address_space: ast::StateSpace,
    pub alignment: u32,
    pub size_bytes: u64,
    pub is_contiguous: bool,
    pub stride_info: Option<Vec<i64>>,
}

/// Optimization remarks and annotations
#[derive(Debug, Clone)]
pub struct OptimizationRemark {
    pub remark_type: RemarkType,
    pub message: String,
    pub location: UniversalDebugLocation,
    pub pass_name: String,
    pub severity: Severity,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone)]
pub enum RemarkType {
    Vectorization,
    Inlining,
    LoopOptimization,
    MemoryOptimization,
    TensorOptimization,
    DialectConversion,
    Performance,
    Warning,
}

#[derive(Debug, Clone)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Note,
}

/// Universal MLIR debug context
pub struct UniversalMlirDebugContext {
    pub debug_enabled: bool,
    pub current_dialect: MlirDialect,
    pub optimization_level: OptimizationLevel,
    pub source_files: HashMap<String, SourceFileInfo>,
    pub debug_locations: Vec<UniversalDebugLocation>,
    pub variables: HashMap<SpirvWord, EnhancedVariableDebugInfo>,
    pub scopes: BTreeMap<String, DebugScope>,
    pub optimization_remarks: Vec<OptimizationRemark>,
    pub type_registry: HashMap<String, DebugTypeInfo>,
    pub instruction_mapping: HashMap<String, InstructionDebugInfo>,
    pub performance_annotations: Vec<PerformanceAnnotation>,
}

#[derive(Debug, Clone)]
pub struct SourceFileInfo {
    pub path: String,
    pub content: Option<String>,
    pub checksum: Option<String>,
    pub language: SourceLanguage,
}

#[derive(Debug, Clone)]
pub enum SourceLanguage {
    PTX,
    CUDA,
    HIP,
    SYCL,
    OpenCL,
}

#[derive(Debug, Clone)]
pub struct DebugScope {
    pub name: String,
    pub parent: Option<String>,
    pub children: Vec<String>,
    pub location: UniversalDebugLocation,
    pub variables: Vec<SpirvWord>,
    pub scope_type: ScopeType,
}

#[derive(Debug, Clone)]
pub enum ScopeType {
    Function,
    Block,
    Loop,
    Conditional,
}

#[derive(Debug, Clone)]
pub struct InstructionDebugInfo {
    pub ptx_instruction: String,
    pub mlir_operations: Vec<String>,
    pub transformation_notes: Vec<String>,
    pub performance_impact: PerformanceImpact,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone)]
pub struct PerformanceImpact {
    pub latency_cycles: Option<u32>,
    pub throughput_ops_per_cycle: Option<f64>,
    pub memory_bandwidth_gb_s: Option<f64>,
    pub compute_intensity: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub register_pressure: u32,
    pub memory_usage_bytes: u64,
    pub vector_units_used: u32,
    pub special_function_units: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PerformanceAnnotation {
    pub location: UniversalDebugLocation,
    pub annotation_type: PerformanceAnnotationType,
    pub value: f64,
    pub unit: String,
    pub context: String,
}

#[derive(Debug, Clone)]
pub enum PerformanceAnnotationType {
    ExecutionTime,
    MemoryBandwidth,
    CacheHitRate,
    VectorizationFactor,
    ParallelizationDegree,
    EnergyConsumption,
}

impl UniversalMlirDebugContext {
    pub fn new(dialect: MlirDialect, optimization_level: OptimizationLevel) -> Self {
        Self {
            debug_enabled: true,
            current_dialect: dialect,
            optimization_level,
            source_files: HashMap::new(),
            debug_locations: Vec::new(),
            variables: HashMap::new(),
            scopes: BTreeMap::new(),
            optimization_remarks: Vec::new(),
            type_registry: HashMap::new(),
            instruction_mapping: HashMap::new(),
            performance_annotations: Vec::new(),
        }
    }

    /// Add a source file to the debug context
    pub fn add_source_file(&mut self, filename: &str, content: Option<String>) {
        let checksum = content.as_ref().map(|c| {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            c.hash(&mut hasher);
            format!("{:x}", hasher.finish())
        });

        self.source_files.insert(
            filename.to_string(),
            SourceFileInfo {
                path: filename.to_string(),
                content,
                checksum,
                language: SourceLanguage::PTX,
            },
        );
    }

    /// Create a debug location with comprehensive metadata
    pub fn create_debug_location(
        &mut self,
        file: &str,
        line: u32,
        column: u32,
        instruction_name: &str,
    ) -> UniversalDebugLocation {
        let location = UniversalDebugLocation {
            file: file.to_string(),
            line,
            column,
            instruction_name: instruction_name.to_string(),
            dialect: self.current_dialect.clone(),
            optimization_level: self.optimization_level.clone(),
        };

        self.debug_locations.push(location.clone());
        location
    }

    /// Add enhanced variable debug information
    pub fn add_variable_debug_info(
        &mut self,
        var_id: SpirvWord,
        name: &str,
        ptx_type: &ast::Type,
        mlir_type: &str,
        scope: &str,
        location: UniversalDebugLocation,
    ) {
        let type_info = DebugTypeInfo {
            ptx_type: ptx_type.clone(),
            mlir_type: mlir_type.to_string(),
            size_bits: self.calculate_type_size_bits(ptx_type),
            alignment: self.calculate_type_alignment(ptx_type),
            memory_space: self.extract_memory_space(ptx_type),
            is_tensor: mlir_type.contains("tensor"),
            tensor_shape: self.extract_tensor_shape(mlir_type),
            element_type: self.extract_element_type(mlir_type),
        };

        let var_info = EnhancedVariableDebugInfo {
            name: name.to_string(),
            type_info,
            scope: scope.to_string(),
            location,
            lifetime: VariableLifetime::Function, // Default, can be refined
            usage_pattern: UsagePattern {
                read_count: 0,
                write_count: 0,
                is_loop_invariant: false,
                is_vectorizable: true,
                memory_access_pattern: MemoryAccessPattern::Sequential,
            },
            memory_layout: MemoryLayout {
                address_space: self.extract_memory_space(ptx_type),
                alignment: self.calculate_type_alignment(ptx_type),
                size_bytes: self.calculate_type_size_bits(ptx_type) / 8,
                is_contiguous: true,
                stride_info: None,
            },
        };

        self.variables.insert(var_id, var_info);
    }

    /// Add optimization remark
    pub fn add_optimization_remark(
        &mut self,
        remark_type: RemarkType,
        message: &str,
        location: UniversalDebugLocation,
        pass_name: &str,
        severity: Severity,
    ) {
        let remark = OptimizationRemark {
            remark_type,
            message: message.to_string(),
            location,
            pass_name: pass_name.to_string(),
            severity,
            suggested_fix: None,
        };

        self.optimization_remarks.push(remark);
    }

    /// Add performance annotation
    pub fn add_performance_annotation(
        &mut self,
        location: UniversalDebugLocation,
        annotation_type: PerformanceAnnotationType,
        value: f64,
        unit: &str,
        context: &str,
    ) {
        let annotation = PerformanceAnnotation {
            location,
            annotation_type,
            value,
            unit: unit.to_string(),
            context: context.to_string(),
        };

        self.performance_annotations.push(annotation);
    }

    /// Generate MLIR location attribute string
    pub fn format_location_attribute(&self, location: &UniversalDebugLocation) -> String {
        // Generate proper MLIR location attribute that goes after the operation
        format!(
            "loc(\"{}\":{}:{})",
            location.file, location.line, location.column
        )
    }

    /// Generate MLIR location definition for module header
    pub fn format_location_definition(&self, location: &UniversalDebugLocation) -> String {
        let loc_id = format!(
            "loc{}_{}_{}",
            location.line,
            location.column,
            location
                .file
                .replace(".", "_")
                .replace("/", "_")
                .replace("-", "_")
        );
        format!(
            "#loc_{} = loc(\"{}\":{}:{})",
            loc_id, location.file, location.line, location.column
        )
    }

    /// Generate MLIR location reference for use in operations
    pub fn format_location_reference(&self, location: &UniversalDebugLocation) -> String {
        let loc_id = format!(
            "loc{}_{}_{}",
            location.line,
            location.column,
            location
                .file
                .replace(".", "_")
                .replace("/", "_")
                .replace("-", "_")
        );
        format!("loc(#loc_{})", loc_id)
    }

    /// Generate comprehensive debug summary
    pub fn generate_debug_summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("// ===== UNIVERSAL MLIR DEBUG SUMMARY =====\n");
        summary.push_str(&format!("// Dialect: {:?}\n", self.current_dialect));
        summary.push_str(&format!(
            "// Optimization Level: {:?}\n",
            self.optimization_level
        ));
        summary.push_str(&format!(
            "// Total Debug Locations: {}\n",
            self.debug_locations.len()
        ));
        summary.push_str(&format!("// Total Variables: {}\n", self.variables.len()));
        summary.push_str(&format!("// Total Scopes: {}\n", self.scopes.len()));
        summary.push_str(&format!(
            "// Optimization Remarks: {}\n",
            self.optimization_remarks.len()
        ));
        summary.push_str(&format!(
            "// Performance Annotations: {}\n",
            self.performance_annotations.len()
        ));
        summary.push_str("\n");

        // Source files
        summary.push_str("// Source Files:\n");
        for (file, info) in &self.source_files {
            summary.push_str(&format!(
                "//   - {} ({})\n",
                file,
                info.language.debug_name()
            ));
            if let Some(ref checksum) = info.checksum {
                summary.push_str(&format!("//     Checksum: {}\n", checksum));
            }
        }
        summary.push_str("\n");

        // Type registry
        summary.push_str("// Type Registry:\n");
        for (name, type_info) in &self.type_registry {
            summary.push_str(&format!(
                "//   - {}: {} ({} bits, {:?})\n",
                name, type_info.mlir_type, type_info.size_bits, type_info.memory_space
            ));
        }
        summary.push_str("\n");

        // Variables with enhanced info
        summary.push_str("// Variables with Debug Info:\n");
        for (id, var_info) in &self.variables {
            summary.push_str(&format!(
                "//   - {}: {} ({})\n",
                var_info.name, var_info.type_info.mlir_type, id.0
            ));
            summary.push_str(&format!(
                "//     Lifetime: {:?}, Usage: R{}/W{}\n",
                var_info.lifetime,
                var_info.usage_pattern.read_count,
                var_info.usage_pattern.write_count
            ));
            summary.push_str(&format!(
                "//     Memory: {:?}, {} bytes\n",
                var_info.memory_layout.address_space, var_info.memory_layout.size_bytes
            ));
        }
        summary.push_str("\n");

        // Optimization remarks
        if !self.optimization_remarks.is_empty() {
            summary.push_str("// Optimization Remarks:\n");
            for remark in &self.optimization_remarks {
                summary.push_str(&format!(
                    "//   [{:?}] {}: {}\n",
                    remark.severity, remark.pass_name, remark.message
                ));
                summary.push_str(&format!(
                    "//     at {}:{}:{}\n",
                    remark.location.file, remark.location.line, remark.location.column
                ));
            }
            summary.push_str("\n");
        }

        // Performance annotations
        if !self.performance_annotations.is_empty() {
            summary.push_str("// Performance Annotations:\n");
            for annotation in &self.performance_annotations {
                summary.push_str(&format!(
                    "//   {:?}: {:.2} {} ({})\n",
                    annotation.annotation_type,
                    annotation.value,
                    annotation.unit,
                    annotation.context
                ));
            }
            summary.push_str("\n");
        }

        summary.push_str("// ===== END DEBUG SUMMARY =====\n");
        summary
    }

    // Helper methods
    fn calculate_type_size_bits(&self, ptx_type: &ast::Type) -> u64 {
        match ptx_type {
            ast::Type::Scalar(scalar_type) => match scalar_type {
                ast::ScalarType::U8 | ast::ScalarType::S8 | ast::ScalarType::B8 => 8,
                ast::ScalarType::U16
                | ast::ScalarType::S16
                | ast::ScalarType::B16
                | ast::ScalarType::F16
                | ast::ScalarType::BF16 => 16,
                ast::ScalarType::U32
                | ast::ScalarType::S32
                | ast::ScalarType::B32
                | ast::ScalarType::F32 => 32,
                ast::ScalarType::U64
                | ast::ScalarType::S64
                | ast::ScalarType::B64
                | ast::ScalarType::F64 => 64,
                ast::ScalarType::Pred => 1,
                ast::ScalarType::U16x2 | ast::ScalarType::S16x2 => 32,
                ast::ScalarType::F16x2 | ast::ScalarType::BF16x2 => 32,
                ast::ScalarType::B128 => 128,
                ast::ScalarType::E4m3x2 | ast::ScalarType::E5m2x2 => 16, // Two 8-bit floats
            },
            ast::Type::Vector(count, scalar_type) => {
                let scalar_size = self.calculate_type_size_bits(&ast::Type::Scalar(*scalar_type));
                *count as u64 * scalar_size
            }
            ast::Type::Array(_, scalar_type, dimensions) => {
                let scalar_size = self.calculate_type_size_bits(&ast::Type::Scalar(*scalar_type));
                let total_elements: u64 = dimensions.iter().map(|&d| d as u64).product();
                total_elements * scalar_size
            }
            ast::Type::Pointer(scalar_type, _) => {
                // Pointer size is typically 64 bits, but element type affects stride calculations
                64
            }
        }
    }

    fn calculate_type_alignment(&self, ptx_type: &ast::Type) -> u32 {
        match ptx_type {
            ast::Type::Scalar(scalar_type) => match scalar_type {
                ast::ScalarType::U8 | ast::ScalarType::S8 | ast::ScalarType::B8 => 1,
                ast::ScalarType::U16
                | ast::ScalarType::S16
                | ast::ScalarType::B16
                | ast::ScalarType::F16
                | ast::ScalarType::BF16 => 2,
                ast::ScalarType::U32
                | ast::ScalarType::S32
                | ast::ScalarType::B32
                | ast::ScalarType::F32 => 4,
                ast::ScalarType::U64
                | ast::ScalarType::S64
                | ast::ScalarType::B64
                | ast::ScalarType::F64 => 8,
                ast::ScalarType::Pred => 1,
                ast::ScalarType::U16x2 | ast::ScalarType::S16x2 => 4,
                ast::ScalarType::F16x2 | ast::ScalarType::BF16x2 => 4,
                ast::ScalarType::B128 => 16,
                ast::ScalarType::E4m3x2 | ast::ScalarType::E5m2x2 => 2, // Two 8-bit floats
            },
            ast::Type::Vector(count, scalar_type) => {
                let scalar_align = self.calculate_type_alignment(&ast::Type::Scalar(*scalar_type));
                std::cmp::min(*count as u32 * scalar_align, 16) // Cap at 16-byte alignment
            }
            ast::Type::Array(_, scalar_type, _) => {
                self.calculate_type_alignment(&ast::Type::Scalar(*scalar_type))
            }
            ast::Type::Pointer(_, _) => 8, // 64-bit pointer alignment
        }
    }

    fn extract_memory_space(&self, ptx_type: &ast::Type) -> ast::StateSpace {
        match ptx_type {
            ast::Type::Pointer(_, state_space) => *state_space,
            _ => ast::StateSpace::Reg, // Default for non-pointer types
        }
    }

    fn extract_tensor_shape(&self, mlir_type: &str) -> Option<Vec<i64>> {
        if mlir_type.contains("tensor<") {
            // Parse tensor shape from MLIR type string
            // Example: "tensor<32x32xf32>" -> [32, 32]
            if let Some(start) = mlir_type.find('<') {
                if let Some(end) = mlir_type.rfind('x') {
                    let shape_str = &mlir_type[start + 1..end];
                    let dimensions: Result<Vec<i64>, _> = shape_str
                        .split('x')
                        .map(|s| s.trim().parse::<i64>())
                        .collect();
                    return dimensions.ok();
                }
            }
        }
        None
    }

    fn extract_element_type(&self, mlir_type: &str) -> Option<String> {
        if mlir_type.contains("tensor<") {
            // Extract element type from tensor
            if let Some(last_x) = mlir_type.rfind('x') {
                if let Some(end) = mlir_type.find('>') {
                    return Some(mlir_type[last_x + 1..end].to_string());
                }
            }
        }
        None
    }
}

impl fmt::Display for MlirDialect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlirDialect::TOSA => write!(f, "TOSA"),
            MlirDialect::TTIR => write!(f, "TTIR"),
            MlirDialect::Linalg => write!(f, "Linalg"),
            MlirDialect::Arith => write!(f, "Arith"),
            MlirDialect::MemRef => write!(f, "MemRef"),
            MlirDialect::Func => write!(f, "Func"),
            MlirDialect::EmitC => write!(f, "EmitC"),
            MlirDialect::Custom(name) => write!(f, "{}", name),
        }
    }
}

impl SourceLanguage {
    fn debug_name(&self) -> &'static str {
        match self {
            SourceLanguage::PTX => "PTX",
            SourceLanguage::CUDA => "CUDA",
            SourceLanguage::HIP => "HIP",
            SourceLanguage::SYCL => "SYCL",
            SourceLanguage::OpenCL => "OpenCL",
        }
    }
}

/// MLIR Debugger Integration trait
pub trait MlirDebuggerIntegration {
    /// Export debug information in GDB-compatible format
    fn export_gdb_debug_info(&self) -> Vec<u8>;

    /// Export debug information in LLDB-compatible format
    fn export_lldb_debug_info(&self) -> Vec<u8>;

    /// Export debug information in MLIR debugger format
    fn export_mlir_debug_info(&self) -> String;

    /// Set breakpoint at specific location
    fn set_breakpoint(&mut self, location: &UniversalDebugLocation) -> Result<u32, String>;

    /// Get variable value at runtime
    fn get_variable_value(&self, var_id: SpirvWord) -> Option<String>;

    /// Step through execution
    fn step_instruction(&mut self) -> Result<UniversalDebugLocation, String>;
}

impl MlirDebuggerIntegration for UniversalMlirDebugContext {
    fn export_gdb_debug_info(&self) -> Vec<u8> {
        // Generate DWARF debug information compatible with GDB
        let mut dwarf_data = Vec::new();

        // Basic DWARF header (simplified)
        dwarf_data.extend_from_slice(b"DWARF_DEBUG_INFO");

        // Add compilation unit
        for (filename, file_info) in &self.source_files {
            let cu_header = format!("CU: {} ({})\n", filename, file_info.language.debug_name());
            dwarf_data.extend_from_slice(cu_header.as_bytes());
        }

        // Add variable information
        for (var_id, var_info) in &self.variables {
            let var_entry = format!(
                "VAR: {} {} {} @ {}:{}:{}\n",
                var_id.0,
                var_info.name,
                var_info.type_info.mlir_type,
                var_info.location.file,
                var_info.location.line,
                var_info.location.column
            );
            dwarf_data.extend_from_slice(var_entry.as_bytes());
        }

        dwarf_data
    }

    fn export_lldb_debug_info(&self) -> Vec<u8> {
        // Generate LLDB-compatible debug information
        let mut lldb_data = Vec::new();

        lldb_data.extend_from_slice(b"LLDB_DEBUG_INFO\n");

        // Module information
        let module_info = format!(
            "module: {} dialect, {} optimization\n",
            self.current_dialect,
            match self.optimization_level {
                OptimizationLevel::None => "O0",
                OptimizationLevel::Less => "O1",
                OptimizationLevel::Default => "O2",
                OptimizationLevel::Aggressive => "O3",
            }
        );
        lldb_data.extend_from_slice(module_info.as_bytes());

        // Function and variable information
        for scope in self.scopes.values() {
            let scope_info = format!(
                "scope: {} type={:?} @ {}:{}:{}\n",
                scope.name,
                scope.scope_type,
                scope.location.file,
                scope.location.line,
                scope.location.column
            );
            lldb_data.extend_from_slice(scope_info.as_bytes());
        }

        lldb_data
    }

    fn export_mlir_debug_info(&self) -> String {
        let mut mlir_debug = String::new();

        mlir_debug.push_str("// MLIR Debug Information Export\n");
        mlir_debug.push_str(&format!(
            "// Generated for {} dialect\n",
            self.current_dialect
        ));
        mlir_debug.push_str("\n");

        // Location definitions
        mlir_debug.push_str("// Location Definitions:\n");
        for (i, location) in self.debug_locations.iter().enumerate() {
            mlir_debug.push_str(&format!(
                "#loc{} = {}\n",
                i,
                self.format_location_attribute(location)
            ));
        }
        mlir_debug.push_str("\n");

        // Type definitions
        mlir_debug.push_str("// Type Definitions:\n");
        for (name, type_info) in &self.type_registry {
            mlir_debug.push_str(&format!(
                "// {} : {} ({} bits)\n",
                name, type_info.mlir_type, type_info.size_bits
            ));
        }
        mlir_debug.push_str("\n");

        // Performance annotations
        if !self.performance_annotations.is_empty() {
            mlir_debug.push_str("// Performance Annotations:\n");
            for annotation in &self.performance_annotations {
                mlir_debug.push_str(&format!(
                    "// {:?}: {:.2} {} at {}:{}:{}\n",
                    annotation.annotation_type,
                    annotation.value,
                    annotation.unit,
                    annotation.location.file,
                    annotation.location.line,
                    annotation.location.column
                ));
            }
            mlir_debug.push_str("\n");
        }

        mlir_debug
    }

    fn set_breakpoint(&mut self, location: &UniversalDebugLocation) -> Result<u32, String> {
        // Simulate breakpoint setting
        let breakpoint_id = self.debug_locations.len() as u32 + 1000;

        let remark = OptimizationRemark {
            remark_type: RemarkType::Warning,
            message: format!(
                "Breakpoint {} set at {}:{}:{}",
                breakpoint_id, location.file, location.line, location.column
            ),
            location: location.clone(),
            pass_name: "debugger".to_string(),
            severity: Severity::Info,
            suggested_fix: None,
        };

        self.optimization_remarks.push(remark);
        Ok(breakpoint_id)
    }

    fn get_variable_value(&self, var_id: SpirvWord) -> Option<String> {
        self.variables.get(&var_id).map(|var_info| {
            format!(
                "{}: {} ({})",
                var_info.name, var_info.type_info.mlir_type, var_info.memory_layout.size_bytes
            )
        })
    }

    fn step_instruction(&mut self) -> Result<UniversalDebugLocation, String> {
        if !self.debug_locations.is_empty() {
            Ok(self.debug_locations[0].clone())
        } else {
            Err("No debug locations available".to_string())
        }
    }
}
