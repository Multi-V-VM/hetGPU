// Static Analysis Component for Delta Checkpoint System
// Analyzes PTX AST, LLVM IR, and compilation artifacts to identify changes
// between compilation states for incremental checkpointing

use crate::delta_checkpoint::{
    AstChanges, AstNodeInfo, AstNodeModification, BasicBlockDelta, CfgChanges, CfgEdge,
    IrChanges, IrInstructionInfo, IrInstructionModification, StaticAnalysisDeltas,
    SymbolChanges, SymbolInfo, SymbolModification, ScopeInfo, BlockInfo,
};
use crate::debug::PtxSourceLocation;
use crate::TranslateError;
use ptx_parser as ast;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Advanced static analyzer for PTX compilation deltas
pub struct StaticDeltaAnalyzer {
    /// Previous AST state for comparison
    previous_ast_state: Option<AstState>,
    /// Previous IR state for comparison  
    previous_ir_state: Option<IrState>,
    /// Previous symbol table state
    previous_symbol_state: Option<SymbolTableState>,
    /// Configuration for analysis depth
    analysis_config: AnalysisConfig,
}

/// Configuration for static analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Enable deep AST comparison (slower but more accurate)
    pub deep_ast_analysis: bool,
    /// Track control flow graph changes
    pub track_cfg_changes: bool,
    /// Monitor optimization passes impact
    pub track_optimization_impact: bool,
    /// Maximum depth for recursive AST analysis
    pub max_ast_depth: usize,
}

/// AST state snapshot for comparison
#[derive(Debug, Clone)]
pub struct AstState {
    pub directives: Vec<DirectiveInfo>,
    pub functions: HashMap<String, FunctionInfo>,
    pub variables: HashMap<String, VariableInfo>,
    pub node_hashes: HashMap<String, u64>,
    pub source_locations: HashMap<String, PtxSourceLocation>,
}

/// LLVM IR state snapshot
#[derive(Debug, Clone)]
pub struct IrState {
    pub functions: HashMap<String, IrFunctionInfo>,
    pub basic_blocks: HashMap<String, IrBasicBlockInfo>,
    pub instructions: HashMap<String, IrInstructionInfo>,
    pub globals: HashMap<String, IrGlobalInfo>,
    pub metadata: HashMap<String, IrMetadataInfo>,
}

/// Symbol table state snapshot
#[derive(Debug, Clone)]
pub struct SymbolTableState {
    pub symbols: HashMap<String, SymbolInfo>,
    pub scopes: HashMap<String, ScopeInfo>,
    pub type_definitions: HashMap<String, TypeInfo>,
}

/// PTX directive information
#[derive(Debug, Clone)]
pub struct DirectiveInfo {
    pub directive_type: String,
    pub name: Option<String>,
    pub parameters: Vec<String>,
    pub hash: u64,
    pub source_location: Option<PtxSourceLocation>,
}

/// PTX function information
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub parameters: Vec<ParameterInfo>,
    pub return_type: Option<String>,
    pub body_hash: u64,
    pub instruction_count: usize,
    pub uses_shared_memory: bool,
}

/// PTX variable information
#[derive(Debug, Clone)]
pub struct VariableInfo {
    pub name: String,
    pub var_type: String,
    pub state_space: String,
    pub alignment: Option<u32>,
    pub is_array: bool,
    pub array_size: Option<usize>,
}

/// Function parameter information
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub name: String,
    pub param_type: String,
    pub state_space: String,
}

/// LLVM IR function information
#[derive(Debug, Clone)]
pub struct IrFunctionInfo {
    pub name: String,
    pub return_type: String,
    pub parameters: Vec<String>,
    pub basic_block_count: usize,
    pub instruction_count: usize,
    pub attributes: HashMap<String, String>,
}

/// LLVM IR basic block information
#[derive(Debug, Clone)]
pub struct IrBasicBlockInfo {
    pub name: String,
    pub function: String,
    pub instructions: Vec<String>,
    pub predecessors: Vec<String>,
    pub successors: Vec<String>,
    pub terminator_type: String,
}

/// LLVM IR global information
#[derive(Debug, Clone)]
pub struct IrGlobalInfo {
    pub name: String,
    pub global_type: String,
    pub linkage: String,
    pub initial_value: Option<String>,
}

/// LLVM IR metadata information
#[derive(Debug, Clone)]
pub struct IrMetadataInfo {
    pub id: String,
    pub metadata_type: String,
    pub content: String,
}

/// Type definition information
#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub type_name: String,
    pub kind: String,
    pub size_bits: usize,
    pub alignment: usize,
}

impl StaticDeltaAnalyzer {
    /// Create new static delta analyzer
    pub fn new() -> Self {
        Self {
            previous_ast_state: None,
            previous_ir_state: None,
            previous_symbol_state: None,
            analysis_config: AnalysisConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: AnalysisConfig) -> Self {
        Self {
            previous_ast_state: None,
            previous_ir_state: None,
            previous_symbol_state: None,
            analysis_config: config,
        }
    }

    /// Analyze PTX source changes and generate static deltas
    pub fn analyze_ptx_changes(
        &mut self,
        previous_source: &str,
        current_source: &str,
    ) -> Result<StaticAnalysisDeltas, TranslateError> {
        // Parse both sources
        let previous_ast = self.parse_ptx_safely(previous_source)?;
        let current_ast = self.parse_ptx_safely(current_source)?;

        // Extract AST states
        let previous_state = self.extract_ast_state(&previous_ast)?;
        let current_state = self.extract_ast_state(&current_ast)?;

        // Compare and generate deltas
        let ast_changes = self.compare_ast_states(&previous_state, &current_state)?;

        // Store current state for next comparison
        self.previous_ast_state = Some(current_state);

        Ok(StaticAnalysisDeltas {
            ast_changes,
            ir_changes: IrChanges::empty(),
            symbol_changes: SymbolChanges::empty(),
            cfg_changes: CfgChanges::empty(),
        })
    }

    /// Analyze LLVM IR changes between compilation states
    pub fn analyze_ir_changes(
        &mut self,
        previous_ir: &str,
        current_ir: &str,
    ) -> Result<IrChanges, TranslateError> {
        let previous_state = self.extract_ir_state(previous_ir)?;
        let current_state = self.extract_ir_state(current_ir)?;

        let ir_changes = self.compare_ir_states(&previous_state, &current_state)?;

        self.previous_ir_state = Some(current_state);
        Ok(ir_changes)
    }

    /// Analyze symbol table changes
    pub fn analyze_symbol_changes(
        &mut self,
        previous_symbols: &HashMap<String, SymbolInfo>,
        current_symbols: &HashMap<String, SymbolInfo>,
    ) -> Result<SymbolChanges, TranslateError> {
        let mut added_symbols = HashMap::new();
        let mut modified_symbols = HashMap::new();
        let mut removed_symbols = HashSet::new();

        // Find added and modified symbols
        for (name, current_symbol) in current_symbols {
            if let Some(previous_symbol) = previous_symbols.get(name) {
                if !self.symbols_equal(previous_symbol, current_symbol) {
                    modified_symbols.insert(
                        name.clone(),
                        self.create_symbol_modification(previous_symbol, current_symbol),
                    );
                }
            } else {
                added_symbols.insert(name.clone(), current_symbol.clone());
            }
        }

        // Find removed symbols
        for name in previous_symbols.keys() {
            if !current_symbols.contains_key(name) {
                removed_symbols.insert(name.clone());
            }
        }

        Ok(SymbolChanges {
            added_symbols,
            modified_symbols,
            removed_symbols,
            scope_changes: HashMap::new(), // TODO: Implement scope tracking
        })
    }

    /// Analyze control flow graph changes
    pub fn analyze_cfg_changes(
        &mut self,
        previous_cfg: &ControlFlowGraph,
        current_cfg: &ControlFlowGraph,
    ) -> Result<CfgChanges, TranslateError> {
        if !self.analysis_config.track_cfg_changes {
            return Ok(CfgChanges::empty());
        }

        let mut added_edges = Vec::new();
        let mut removed_edges = Vec::new();
        let mut modified_blocks = HashMap::new();

        // Compare edges
        for edge in &current_cfg.edges {
            if !previous_cfg.edges.contains(edge) {
                added_edges.push(edge.clone());
            }
        }

        for edge in &previous_cfg.edges {
            if !current_cfg.edges.contains(edge) {
                removed_edges.push(edge.clone());
            }
        }

        // Compare blocks
        for (block_name, current_block) in &current_cfg.blocks {
            if let Some(previous_block) = previous_cfg.blocks.get(block_name) {
                if !self.blocks_equal(previous_block, current_block) {
                    modified_blocks.insert(
                        block_name.clone(),
                        BlockInfo {
                            instruction_count: current_block.instruction_count,
                            predecessor_count: current_block.predecessors.len(),
                            successor_count: current_block.successors.len(),
                        },
                    );
                }
            }
        }

        Ok(CfgChanges {
            added_edges,
            removed_edges,
            modified_blocks,
        })
    }

    /// Calculate hash for AST node
    pub fn calculate_ast_hash<T: Hash>(&self, node: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        node.hash(&mut hasher);
        hasher.finish()
    }

    /// Calculate semantic hash for PTX instruction
    pub fn calculate_instruction_hash(
        &self,
        instruction: &ast::Instruction<ast::ParsedOperand<u32>>,
    ) -> u64 {
        let mut hasher = DefaultHasher::new();
        
        // Hash instruction opcode
        std::mem::discriminant(instruction).hash(&mut hasher);
        
        // Hash operands (simplified)
        match instruction {
            ast::Instruction::Add { data, arguments } => {
                data.hash(&mut hasher);
                arguments.len().hash(&mut hasher);
            }
            ast::Instruction::Mov { data, arguments } => {
                data.hash(&mut hasher);
                arguments.len().hash(&mut hasher);
            }
            ast::Instruction::Ld { data, arguments } => {
                data.hash(&mut hasher);
                arguments.len().hash(&mut hasher);
            }
            ast::Instruction::St { data, arguments } => {
                data.hash(&mut hasher);
                arguments.len().hash(&mut hasher);
            }
            // Add more instruction types as needed
            _ => {
                // Generic hash for unknown instructions
                "unknown".hash(&mut hasher);
            }
        }
        
        hasher.finish()
    }

    /// Generate optimization impact analysis
    pub fn analyze_optimization_impact(
        &self,
        before_optimization: &IrState,
        after_optimization: &IrState,
    ) -> OptimizationImpactAnalysis {
        let mut analysis = OptimizationImpactAnalysis::new();

        // Analyze instruction count changes
        let before_count: usize = before_optimization.functions.values()
            .map(|f| f.instruction_count)
            .sum();
        let after_count: usize = after_optimization.functions.values()
            .map(|f| f.instruction_count)
            .sum();

        analysis.instruction_count_delta = after_count as i64 - before_count as i64;
        analysis.instruction_count_ratio = if before_count > 0 {
            after_count as f64 / before_count as f64
        } else {
            1.0
        };

        // Analyze basic block changes
        let before_bb_count: usize = before_optimization.functions.values()
            .map(|f| f.basic_block_count)
            .sum();
        let after_bb_count: usize = after_optimization.functions.values()
            .map(|f| f.basic_block_count)
            .sum();

        analysis.basic_block_count_delta = after_bb_count as i64 - before_bb_count as i64;

        // Classify optimization type
        if analysis.instruction_count_delta < 0 {
            analysis.optimization_type = OptimizationType::CodeReduction;
        } else if analysis.basic_block_count_delta < 0 {
            analysis.optimization_type = OptimizationType::ControlFlowSimplification;
        } else {
            analysis.optimization_type = OptimizationType::Other;
        }

        analysis
    }

    // Private helper methods

    fn parse_ptx_safely(&self, source: &str) -> Result<ast::Module, TranslateError> {
        ptx_parser::parse_module_checked(source)
            .map_err(|e| TranslateError::UnexpectedError(format!("PTX parse error: {:?}", e)))
    }

    fn extract_ast_state(&self, module: &ast::Module) -> Result<AstState, TranslateError> {
        let mut state = AstState {
            directives: Vec::new(),
            functions: HashMap::new(),
            variables: HashMap::new(),
            node_hashes: HashMap::new(),
            source_locations: HashMap::new(),
        };

        // Extract directives
        for (index, directive) in module.directives.iter().enumerate() {
            let directive_info = self.extract_directive_info(directive, index)?;
            state.directives.push(directive_info);
        }

        Ok(state)
    }

    fn extract_directive_info(
        &self,
        directive: &ast::Directive,
        index: usize,
    ) -> Result<DirectiveInfo, TranslateError> {
        let directive_type = match directive {
            ast::Directive::Variable(..) => "variable",
            ast::Directive::Method(..) => "method",
        }.to_string();

        let hash = self.calculate_directive_hash(directive);
        let directive_id = format!("directive_{}", index);

        Ok(DirectiveInfo {
            directive_type,
            name: None, // TODO: Extract actual names
            parameters: Vec::new(), // TODO: Extract parameters
            hash,
            source_location: None, // TODO: Extract source locations
        })
    }

    fn calculate_directive_hash(&self, directive: &ast::Directive) -> u64 {
        let mut hasher = DefaultHasher::new();
        std::mem::discriminant(directive).hash(&mut hasher);
        hasher.finish()
    }

    fn extract_ir_state(&self, ir_text: &str) -> Result<IrState, TranslateError> {
        // Simplified IR parsing - in a real implementation, this would
        // use LLVM's IR parser or a custom parser
        let mut state = IrState {
            functions: HashMap::new(),
            basic_blocks: HashMap::new(),
            instructions: HashMap::new(),
            globals: HashMap::new(),
            metadata: HashMap::new(),
        };

        // Parse IR text line by line (simplified)
        for (line_num, line) in ir_text.lines().enumerate() {
            if line.starts_with("define ") {
                // Function definition
                let func_info = self.parse_function_definition(line, line_num)?;
                state.functions.insert(func_info.name.clone(), func_info);
            } else if line.contains(':') && !line.starts_with(';') {
                // Basic block or instruction
                if line.ends_with(':') {
                    // Basic block label
                    let block_name = line.trim_end_matches(':').to_string();
                    state.basic_blocks.insert(
                        block_name.clone(),
                        IrBasicBlockInfo {
                            name: block_name,
                            function: "unknown".to_string(),
                            instructions: Vec::new(),
                            predecessors: Vec::new(),
                            successors: Vec::new(),
                            terminator_type: "unknown".to_string(),
                        },
                    );
                }
            }
        }

        Ok(state)
    }

    fn parse_function_definition(
        &self,
        line: &str,
        line_num: usize,
    ) -> Result<IrFunctionInfo, TranslateError> {
        // Simplified function parsing
        let parts: Vec<&str> = line.split_whitespace().collect();
        let name = parts.get(2)
            .and_then(|s| s.split('(').next())
            .unwrap_or("unknown")
            .to_string();

        Ok(IrFunctionInfo {
            name,
            return_type: parts.get(1).unwrap_or("void").to_string(),
            parameters: Vec::new(),
            basic_block_count: 0,
            instruction_count: 0,
            attributes: HashMap::new(),
        })
    }

    fn compare_ast_states(
        &self,
        previous: &AstState,
        current: &AstState,
    ) -> Result<AstChanges, TranslateError> {
        let mut added_nodes = HashMap::new();
        let mut modified_nodes = HashMap::new();
        let mut removed_nodes = HashSet::new();
        let mut node_hash_changes = HashMap::new();

        // Compare directive counts and hashes
        if current.directives.len() != previous.directives.len() {
            // Directives added or removed
            for (index, directive) in current.directives.iter().enumerate() {
                let node_id = format!("directive_{}", index);
                if index >= previous.directives.len() {
                    // New directive
                    added_nodes.insert(
                        node_id,
                        AstNodeInfo {
                            node_type: directive.directive_type.clone(),
                            hash: directive.hash,
                            source_location: directive.source_location.clone().unwrap_or(
                                PtxSourceLocation {
                                    file: "unknown.ptx".to_string(),
                                    line: index as u32 + 1,
                                    column: 0,
                                    instruction_offset: index,
                                }
                            ),
                            children_count: 0,
                        },
                    );
                }
            }
        }

        // Compare existing directives for modifications
        let min_len = std::cmp::min(previous.directives.len(), current.directives.len());
        for index in 0..min_len {
            let prev_directive = &previous.directives[index];
            let curr_directive = &current.directives[index];
            
            if prev_directive.hash != curr_directive.hash {
                let node_id = format!("directive_{}", index);
                modified_nodes.insert(
                    node_id.clone(),
                    AstNodeModification {
                        field_changes: HashMap::new(), // TODO: Track specific field changes
                        hash_before: prev_directive.hash,
                        hash_after: curr_directive.hash,
                    },
                );
                node_hash_changes.insert(node_id, curr_directive.hash);
            }
        }

        Ok(AstChanges {
            added_nodes,
            modified_nodes,
            removed_nodes,
            node_hash_changes,
        })
    }

    fn compare_ir_states(
        &self,
        previous: &IrState,
        current: &IrState,
    ) -> Result<IrChanges, TranslateError> {
        let mut added_instructions = HashMap::new();
        let mut modified_instructions = HashMap::new();
        let mut removed_instructions = HashSet::new();
        let mut basic_block_changes = HashMap::new();

        // Compare functions
        for (func_name, current_func) in &current.functions {
            if let Some(previous_func) = previous.functions.get(func_name) {
                if current_func.instruction_count != previous_func.instruction_count {
                    // Instructions were added or removed in this function
                    basic_block_changes.insert(
                        func_name.clone(),
                        BasicBlockDelta {
                            added_instructions: Vec::new(), // TODO: Track specific instructions
                            removed_instructions: Vec::new(),
                            predecessors_changed: false,
                            successors_changed: false,
                        },
                    );
                }
            } else {
                // New function - all its instructions are new
                for i in 0..current_func.instruction_count {
                    let instr_id = format!("{}_{}", func_name, i);
                    added_instructions.insert(
                        instr_id,
                        IrInstructionInfo {
                            opcode: "unknown".to_string(),
                            operands: Vec::new(),
                            basic_block: func_name.clone(),
                            debug_location: None,
                        },
                    );
                }
            }
        }

        // Find removed functions/instructions
        for func_name in previous.functions.keys() {
            if !current.functions.contains_key(func_name) {
                if let Some(prev_func) = previous.functions.get(func_name) {
                    for i in 0..prev_func.instruction_count {
                        let instr_id = format!("{}_{}", func_name, i);
                        removed_instructions.insert(instr_id);
                    }
                }
            }
        }

        Ok(IrChanges {
            added_instructions,
            modified_instructions,
            removed_instructions,
            basic_block_changes,
        })
    }

    fn symbols_equal(&self, prev: &SymbolInfo, curr: &SymbolInfo) -> bool {
        prev.symbol_type == curr.symbol_type &&
        prev.scope == curr.scope &&
        prev.size == curr.size
    }

    fn create_symbol_modification(
        &self,
        prev: &SymbolInfo,
        curr: &SymbolInfo,
    ) -> SymbolModification {
        SymbolModification {
            type_changed: prev.symbol_type != curr.symbol_type,
            scope_changed: prev.scope != curr.scope,
            location_changed: prev.location != curr.location,
        }
    }

    fn blocks_equal(&self, prev: &BlockInfo, curr: &BlockInfo) -> bool {
        prev.instruction_count == curr.instruction_count &&
        prev.predecessor_count == curr.predecessor_count &&
        prev.successor_count == curr.successor_count
    }
}

/// Control flow graph representation
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    pub blocks: HashMap<String, BlockInfo>,
    pub edges: Vec<CfgEdge>,
}

/// Optimization impact analysis results
#[derive(Debug, Clone)]
pub struct OptimizationImpactAnalysis {
    pub instruction_count_delta: i64,
    pub instruction_count_ratio: f64,
    pub basic_block_count_delta: i64,
    pub optimization_type: OptimizationType,
    pub estimated_performance_impact: f64,
}

/// Types of optimizations detected
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    CodeReduction,
    ControlFlowSimplification,
    InstructionCombining,
    DeadCodeElimination,
    LoopOptimization,
    Other,
}

impl OptimizationImpactAnalysis {
    pub fn new() -> Self {
        Self {
            instruction_count_delta: 0,
            instruction_count_ratio: 1.0,
            basic_block_count_delta: 0,
            optimization_type: OptimizationType::Other,
            estimated_performance_impact: 0.0,
        }
    }
}

impl AnalysisConfig {
    pub fn default() -> Self {
        Self {
            deep_ast_analysis: true,
            track_cfg_changes: true,
            track_optimization_impact: true,
            max_ast_depth: 10,
        }
    }

    pub fn fast() -> Self {
        Self {
            deep_ast_analysis: false,
            track_cfg_changes: false,
            track_optimization_impact: false,
            max_ast_depth: 5,
        }
    }

    pub fn comprehensive() -> Self {
        Self {
            deep_ast_analysis: true,
            track_cfg_changes: true,
            track_optimization_impact: true,
            max_ast_depth: 20,
        }
    }
}

// Empty implementations for delta types
impl IrChanges {
    pub fn empty() -> Self {
        Self {
            added_instructions: HashMap::new(),
            modified_instructions: HashMap::new(),
            removed_instructions: HashSet::new(),
            basic_block_changes: HashMap::new(),
        }
    }
}

impl SymbolChanges {
    pub fn empty() -> Self {
        Self {
            added_symbols: HashMap::new(),
            modified_symbols: HashMap::new(),
            removed_symbols: HashSet::new(),
            scope_changes: HashMap::new(),
        }
    }
}

impl CfgChanges {
    pub fn empty() -> Self {
        Self {
            added_edges: Vec::new(),
            removed_edges: Vec::new(),
            modified_blocks: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_analyzer_creation() {
        let analyzer = StaticDeltaAnalyzer::new();
        assert!(analyzer.previous_ast_state.is_none());
        assert!(analyzer.analysis_config.deep_ast_analysis);
    }

    #[test]
    fn test_ptx_changes_analysis() {
        let mut analyzer = StaticDeltaAnalyzer::new();
        
        let prev_source = r#"
.version 7.0
.target sm_50
.entry kernel1() {
    ret;
}
"#;
        
        let curr_source = r#"
.version 7.0
.target sm_50
.entry kernel1() {
    mov.u32 %r0, 42;
    ret;
}
"#;

        let result = analyzer.analyze_ptx_changes(prev_source, curr_source);
        assert!(result.is_ok());
        
        let deltas = result.unwrap();
        // Should detect changes in the function body
        assert!(!deltas.ast_changes.modified_nodes.is_empty() || 
                !deltas.ast_changes.added_nodes.is_empty());
    }

    #[test]
    fn test_optimization_impact_analysis() {
        let analyzer = StaticDeltaAnalyzer::new();
        
        let before = IrState {
            functions: {
                let mut map = HashMap::new();
                map.insert("test".to_string(), IrFunctionInfo {
                    name: "test".to_string(),
                    return_type: "void".to_string(),
                    parameters: Vec::new(),
                    basic_block_count: 2,
                    instruction_count: 10,
                    attributes: HashMap::new(),
                });
                map
            },
            basic_blocks: HashMap::new(),
            instructions: HashMap::new(),
            globals: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        let after = IrState {
            functions: {
                let mut map = HashMap::new();
                map.insert("test".to_string(), IrFunctionInfo {
                    name: "test".to_string(),
                    return_type: "void".to_string(),
                    parameters: Vec::new(),
                    basic_block_count: 2,
                    instruction_count: 8, // Optimized - removed 2 instructions
                    attributes: HashMap::new(),
                });
                map
            },
            basic_blocks: HashMap::new(),
            instructions: HashMap::new(),
            globals: HashMap::new(),
            metadata: HashMap::new(),
        };

        let analysis = analyzer.analyze_optimization_impact(&before, &after);
        assert_eq!(analysis.instruction_count_delta, -2);
        assert_eq!(analysis.optimization_type, OptimizationType::CodeReduction);
    }

    #[test]
    fn test_hash_calculation() {
        let analyzer = StaticDeltaAnalyzer::new();
        
        // Test consistent hashing
        let test_string = "test_data";
        let hash1 = analyzer.calculate_ast_hash(&test_string);
        let hash2 = analyzer.calculate_ast_hash(&test_string);
        assert_eq!(hash1, hash2);
        
        // Test different inputs produce different hashes
        let different_string = "different_test_data";
        let hash3 = analyzer.calculate_ast_hash(&different_string);
        assert_ne!(hash1, hash3);
    }
}