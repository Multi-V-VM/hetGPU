// Delta Checkpoint System for PTX Compilation
// Combines static analysis (AST/IR changes) with dynamic analysis (runtime state)
// to create incremental checkpoints that only store deltas between compilation states

use crate::checkpoint::{
    CheckpointError, CheckpointManager, CompilationStage, CompileOptions, PerformanceStats,
};
use crate::debug::{DwarfMappingEntry, PtxSourceLocation, TargetInstruction, VariableLocation};
use crate::TranslateError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

/// Delta checkpoint that stores only changes between compilation states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaCheckpoint {
    /// Base checkpoint metadata
    pub metadata: DeltaCheckpointMetadata,
    /// Static analysis deltas (AST/IR changes)
    pub static_deltas: StaticAnalysisDeltas,
    /// Dynamic analysis deltas (runtime state changes)
    pub dynamic_deltas: DynamicAnalysisDeltas,
    /// Compilation stage transition
    pub stage_transition: StageTransition,
    /// Performance delta metrics
    pub performance_delta: PerformanceDelta,
}

/// Metadata for delta checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaCheckpointMetadata {
    pub id: String,
    pub timestamp: u64,
    pub base_checkpoint_id: Option<String>,
    pub delta_size_bytes: usize,
    pub compression_ratio: f64,
    pub created_at: String,
    pub description: String,
}

/// Static analysis changes between checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticAnalysisDeltas {
    /// AST node changes (added, modified, removed)
    pub ast_changes: AstChanges,
    /// LLVM IR instruction deltas
    pub ir_changes: IrChanges,
    /// Symbol table modifications
    pub symbol_changes: SymbolChanges,
    /// Control flow graph updates
    pub cfg_changes: CfgChanges,
}

/// Dynamic analysis runtime state changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicAnalysisDeltas {
    /// Variable state changes
    pub variable_deltas: HashMap<String, VariableStateDelta>,
    /// Memory region updates
    pub memory_deltas: Vec<MemoryDelta>,
    /// Register allocation changes
    pub register_deltas: HashMap<String, RegisterDelta>,
    /// Debug info mapping updates
    pub debug_mapping_deltas: Vec<DebugMappingDelta>,
}

/// AST changes between checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstChanges {
    pub added_nodes: HashMap<String, AstNodeInfo>,
    pub modified_nodes: HashMap<String, AstNodeModification>,
    pub removed_nodes: HashSet<String>,
    pub node_hash_changes: HashMap<String, u64>,
}

/// LLVM IR instruction changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrChanges {
    pub added_instructions: HashMap<String, IrInstructionInfo>,
    pub modified_instructions: HashMap<String, IrInstructionModification>,
    pub removed_instructions: HashSet<String>,
    pub basic_block_changes: HashMap<String, BasicBlockDelta>,
}

/// Symbol table changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolChanges {
    pub added_symbols: HashMap<String, SymbolInfo>,
    pub modified_symbols: HashMap<String, SymbolModification>,
    pub removed_symbols: HashSet<String>,
    pub scope_changes: HashMap<String, ScopeInfo>,
}

/// Control flow graph changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgChanges {
    pub added_edges: Vec<CfgEdge>,
    pub removed_edges: Vec<CfgEdge>,
    pub modified_blocks: HashMap<String, BlockInfo>,
}

/// Variable state change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableStateDelta {
    pub name: String,
    pub previous_location: Option<VariableLocation>,
    pub new_location: VariableLocation,
    pub value_changed: bool,
    pub type_changed: bool,
    pub scope_changed: bool,
}

/// Memory region change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDelta {
    pub address: u64,
    pub size: u32,
    pub operation: MemoryOperation,
    pub content_hash_before: Option<u64>,
    pub content_hash_after: u64,
}

/// Register allocation change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterDelta {
    pub register_name: String,
    pub previous_value: Option<String>,
    pub new_value: String,
    pub allocation_changed: bool,
}

/// Debug mapping change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugMappingDelta {
    pub ptx_location: PtxSourceLocation,
    pub operation: MappingOperation,
    pub previous_mapping: Option<DwarfMappingEntry>,
    pub new_mapping: Option<DwarfMappingEntry>,
}

/// Stage transition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTransition {
    pub from_stage: CompilationStage,
    pub to_stage: CompilationStage,
    pub transition_time_ms: u64,
    pub intermediate_states: Vec<CompilationStage>,
}

/// Performance metrics delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDelta {
    pub time_delta_ms: i64,
    pub memory_delta_bytes: i64,
    pub ir_size_delta_bytes: i64,
    pub spirv_size_delta_bytes: i64,
    pub optimization_impact: f64,
}

/// AST node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstNodeInfo {
    pub node_type: String,
    pub hash: u64,
    pub source_location: PtxSourceLocation,
    pub children_count: usize,
}

/// AST node modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstNodeModification {
    pub field_changes: HashMap<String, String>,
    pub hash_before: u64,
    pub hash_after: u64,
}

/// LLVM IR instruction information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrInstructionInfo {
    pub opcode: String,
    pub operands: Vec<String>,
    pub basic_block: String,
    pub debug_location: Option<PtxSourceLocation>,
}

/// LLVM IR instruction modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrInstructionModification {
    pub opcode_changed: bool,
    pub operands_changed: HashMap<usize, String>,
    pub attributes_changed: HashMap<String, String>,
}

/// Basic block delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlockDelta {
    pub added_instructions: Vec<String>,
    pub removed_instructions: Vec<String>,
    pub predecessors_changed: bool,
    pub successors_changed: bool,
}

/// Symbol information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    pub symbol_type: String,
    pub scope: String,
    pub size: usize,
    pub location: VariableLocation,
}

/// Symbol modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolModification {
    pub type_changed: bool,
    pub scope_changed: bool,
    pub location_changed: bool,
}

/// Scope information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeInfo {
    pub scope_type: String,
    pub parent_scope: Option<String>,
    pub symbols: HashSet<String>,
}

/// Control flow graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgEdge {
    pub from_block: String,
    pub to_block: String,
    pub edge_type: String,
}

/// Basic block information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockInfo {
    pub instruction_count: usize,
    pub predecessor_count: usize,
    pub successor_count: usize,
}

/// Memory operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOperation {
    Allocated,
    Deallocated,
    Modified,
    Accessed,
}

/// Debug mapping operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MappingOperation {
    Added,
    Modified,
    Removed,
}

/// Delta checkpoint manager that combines static and dynamic analysis
pub struct DeltaCheckpointManager {
    /// Base checkpoint manager for full checkpoints
    base_manager: CheckpointManager,
    /// Delta checkpoint storage
    delta_checkpoints: HashMap<String, DeltaCheckpoint>,
    /// Static analyzer for AST/IR changes
    static_analyzer: StaticAnalyzer,
    /// Dynamic analyzer for runtime state changes
    dynamic_analyzer: DynamicAnalyzer,
    /// Compression settings
    compression_enabled: bool,
    /// Maximum delta chain length before creating new base
    max_delta_chain: usize,
}

impl DeltaCheckpointManager {
    /// Create new delta checkpoint manager
    pub fn new<P: AsRef<std::path::Path>>(
        checkpoint_dir: P,
        enable_compression: bool,
    ) -> Result<Self, std::io::Error> {
        let base_manager = CheckpointManager::new(checkpoint_dir)?;
        
        Ok(Self {
            base_manager,
            delta_checkpoints: HashMap::new(),
            static_analyzer: StaticAnalyzer::new(),
            dynamic_analyzer: DynamicAnalyzer::new(),
            compression_enabled: enable_compression,
            max_delta_chain: 10,
        })
    }

    /// Create delta checkpoint by comparing with previous state
    pub fn create_delta_checkpoint(
        &mut self,
        current_state: &CompilationState,
        previous_checkpoint_id: Option<&str>,
        description: String,
    ) -> Result<String, CheckpointError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let delta_id = format!("delta_{}_{}", timestamp, rand::random::<u16>());

        // Analyze static changes
        let static_deltas = if let Some(prev_id) = previous_checkpoint_id {
            if let Some(prev_checkpoint) = self.get_base_checkpoint(prev_id) {
                self.static_analyzer.analyze_changes(&prev_checkpoint.ptx_source, &current_state.ptx_source)?
            } else {
                StaticAnalysisDeltas::empty()
            }
        } else {
            StaticAnalysisDeltas::empty()
        };

        // Analyze dynamic changes
        let dynamic_deltas = if let Some(prev_id) = previous_checkpoint_id {
            self.dynamic_analyzer.analyze_runtime_changes(
                prev_id,
                &current_state.debug_mappings,
                &current_state.variable_states,
            )?
        } else {
            DynamicAnalysisDeltas::empty()
        };

        // Calculate performance delta
        let performance_delta = self.calculate_performance_delta(
            previous_checkpoint_id,
            &current_state.performance_stats,
        )?;

        // Create stage transition info
        let stage_transition = StageTransition {
            from_stage: if let Some(prev_id) = previous_checkpoint_id {
                self.get_previous_stage(prev_id).unwrap_or(CompilationStage::PtxParsing)
            } else {
                CompilationStage::PtxParsing
            },
            to_stage: current_state.stage.clone(),
            transition_time_ms: performance_delta.time_delta_ms as u64,
            intermediate_states: Vec::new(),
        };

        // Calculate delta size
        let delta_size = self.calculate_delta_size(&static_deltas, &dynamic_deltas);
        
        let metadata = DeltaCheckpointMetadata {
            id: delta_id.clone(),
            timestamp,
            base_checkpoint_id: previous_checkpoint_id.map(|s| s.to_string()),
            delta_size_bytes: delta_size,
            compression_ratio: if self.compression_enabled { 0.7 } else { 1.0 },
            created_at: format!("{}:{}:{}", file!(), line!(), column!()),
            description,
        };

        let delta_checkpoint = DeltaCheckpoint {
            metadata,
            static_deltas,
            dynamic_deltas,
            stage_transition,
            performance_delta,
        };

        self.delta_checkpoints.insert(delta_id.clone(), delta_checkpoint);
        Ok(delta_id)
    }

    /// Restore compilation state from delta chain
    pub fn restore_from_delta_chain(
        &self,
        delta_id: &str,
    ) -> Result<CompilationState, CheckpointError> {
        // Find the delta chain to the base checkpoint
        let delta_chain = self.build_delta_chain(delta_id)?;
        
        // Start with base checkpoint
        let base_id = &delta_chain[0];
        let mut state = self.get_base_checkpoint_state(base_id)?;

        // Apply deltas in sequence
        for delta_id in &delta_chain[1..] {
            if let Some(delta) = self.delta_checkpoints.get(delta_id) {
                state = self.apply_delta(&state, delta)?;
            }
        }

        Ok(state)
    }

    /// Apply static analysis deltas to identify changes
    pub fn apply_static_deltas(
        &self,
        state: &mut CompilationState,
        deltas: &StaticAnalysisDeltas,
    ) -> Result<(), CheckpointError> {
        // Apply AST changes
        for (node_id, node_info) in &deltas.ast_changes.added_nodes {
            state.ast_nodes.insert(node_id.clone(), node_info.clone());
        }
        
        for node_id in &deltas.ast_changes.removed_nodes {
            state.ast_nodes.remove(node_id);
        }

        // Apply IR changes
        for (instr_id, instr_info) in &deltas.ir_changes.added_instructions {
            state.ir_instructions.insert(instr_id.clone(), instr_info.clone());
        }

        for instr_id in &deltas.ir_changes.removed_instructions {
            state.ir_instructions.remove(instr_id);
        }

        // Apply symbol changes
        for (symbol_id, symbol_info) in &deltas.symbol_changes.added_symbols {
            state.symbols.insert(symbol_id.clone(), symbol_info.clone());
        }

        for symbol_id in &deltas.symbol_changes.removed_symbols {
            state.symbols.remove(symbol_id);
        }

        Ok(())
    }

    /// Apply dynamic analysis deltas for runtime state recovery
    pub fn apply_dynamic_deltas(
        &self,
        state: &mut CompilationState,
        deltas: &DynamicAnalysisDeltas,
    ) -> Result<(), CheckpointError> {
        // Apply variable state changes
        for (var_name, var_delta) in &deltas.variable_deltas {
            state.variable_states.insert(var_name.clone(), var_delta.new_location.clone());
        }

        // Apply memory deltas
        for memory_delta in &deltas.memory_deltas {
            match memory_delta.operation {
                MemoryOperation::Allocated => {
                    state.memory_regions.insert(
                        memory_delta.address,
                        MemoryRegion {
                            address: memory_delta.address,
                            size: memory_delta.size,
                            content_hash: memory_delta.content_hash_after,
                        }
                    );
                }
                MemoryOperation::Deallocated => {
                    state.memory_regions.remove(&memory_delta.address);
                }
                MemoryOperation::Modified => {
                    if let Some(region) = state.memory_regions.get_mut(&memory_delta.address) {
                        region.content_hash = memory_delta.content_hash_after;
                    }
                }
                MemoryOperation::Accessed => {
                    // Track access patterns if needed
                }
            }
        }

        // Apply register deltas
        for (reg_name, reg_delta) in &deltas.register_deltas {
            state.register_values.insert(reg_name.clone(), reg_delta.new_value.clone());
        }

        // Apply debug mapping deltas
        for mapping_delta in &deltas.debug_mapping_deltas {
            match mapping_delta.operation {
                MappingOperation::Added => {
                    if let Some(ref new_mapping) = mapping_delta.new_mapping {
                        state.debug_mappings.push(new_mapping.clone());
                    }
                }
                MappingOperation::Removed => {
                    state.debug_mappings.retain(|m| 
                        m.ptx_location != mapping_delta.ptx_location
                    );
                }
                MappingOperation::Modified => {
                    if let Some(ref new_mapping) = mapping_delta.new_mapping {
                        if let Some(existing) = state.debug_mappings.iter_mut().find(|m| 
                            m.ptx_location == mapping_delta.ptx_location
                        ) {
                            *existing = new_mapping.clone();
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get optimization suggestions based on delta analysis
    pub fn get_optimization_suggestions(&self, delta_id: &str) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        if let Some(delta) = self.delta_checkpoints.get(delta_id) {
            // Analyze static deltas for optimization opportunities
            if delta.static_deltas.ir_changes.added_instructions.len() > 100 {
                suggestions.push(OptimizationSuggestion {
                    category: "IR Bloat".to_string(),
                    description: "High number of added IR instructions detected. Consider enabling -O2 optimization.".to_string(),
                    estimated_impact: 0.3,
                });
            }

            // Analyze dynamic deltas for runtime issues
            if delta.dynamic_deltas.memory_deltas.len() > 50 {
                suggestions.push(OptimizationSuggestion {
                    category: "Memory Churn".to_string(),
                    description: "High memory allocation/deallocation activity. Consider memory pooling.".to_string(),
                    estimated_impact: 0.4,
                });
            }

            // Performance regression detection
            if delta.performance_delta.time_delta_ms > 1000 {
                suggestions.push(OptimizationSuggestion {
                    category: "Performance Regression".to_string(),
                    description: "Significant increase in compilation time detected.".to_string(),
                    estimated_impact: 0.6,
                });
            }
        }

        suggestions
    }

    /// Compress delta chain to reduce storage
    pub fn compress_delta_chain(&mut self, chain_start: &str) -> Result<String, CheckpointError> {
        let delta_chain = self.build_delta_chain(chain_start)?;
        
        if delta_chain.len() <= 2 {
            return Ok(chain_start.to_string()); // No compression needed
        }

        // Restore full state
        let full_state = self.restore_from_delta_chain(chain_start)?;
        
        // Create new base checkpoint
        let compressed_id = self.base_manager.create_checkpoint(
            full_state.ptx_source,
            full_state.stage,
            format!("Compressed delta chain from {}", chain_start),
        );

        // Remove old delta chain
        for delta_id in &delta_chain[1..] {
            self.delta_checkpoints.remove(delta_id);
        }

        Ok(compressed_id)
    }

    // Helper methods
    
    fn get_base_checkpoint(&self, _id: &str) -> Option<&crate::checkpoint::CompilationCheckpoint> {
        // Implementation would retrieve base checkpoint
        None
    }

    fn get_base_checkpoint_state(&self, _id: &str) -> Result<CompilationState, CheckpointError> {
        // Implementation would convert base checkpoint to state
        Err(CheckpointError::CheckpointNotFound("Not implemented".to_string()))
    }

    fn build_delta_chain(&self, delta_id: &str) -> Result<Vec<String>, CheckpointError> {
        let mut chain = Vec::new();
        let mut current_id = delta_id.to_string();

        loop {
            if let Some(delta) = self.delta_checkpoints.get(&current_id) {
                chain.push(current_id.clone());
                if let Some(base_id) = &delta.metadata.base_checkpoint_id {
                    current_id = base_id.clone();
                } else {
                    break;
                }
            } else {
                // Reached base checkpoint
                chain.push(current_id);
                break;
            }
        }

        chain.reverse();
        Ok(chain)
    }

    fn apply_delta(
        &self,
        state: &CompilationState,
        delta: &DeltaCheckpoint,
    ) -> Result<CompilationState, CheckpointError> {
        let mut new_state = state.clone();
        
        // Apply static deltas
        self.apply_static_deltas(&mut new_state, &delta.static_deltas)?;
        
        // Apply dynamic deltas
        self.apply_dynamic_deltas(&mut new_state, &delta.dynamic_deltas)?;
        
        // Update stage
        new_state.stage = delta.stage_transition.to_stage.clone();
        
        Ok(new_state)
    }

    fn get_previous_stage(&self, _prev_id: &str) -> Option<CompilationStage> {
        // Implementation would retrieve previous stage
        Some(CompilationStage::PtxParsing)
    }

    fn calculate_performance_delta(
        &self,
        _prev_id: Option<&str>,
        _current_stats: &PerformanceStats,
    ) -> Result<PerformanceDelta, CheckpointError> {
        Ok(PerformanceDelta {
            time_delta_ms: 0,
            memory_delta_bytes: 0,
            ir_size_delta_bytes: 0,
            spirv_size_delta_bytes: 0,
            optimization_impact: 0.0,
        })
    }

    fn calculate_delta_size(
        &self,
        _static_deltas: &StaticAnalysisDeltas,
        _dynamic_deltas: &DynamicAnalysisDeltas,
    ) -> usize {
        // Implementation would calculate serialized size
        1024
    }
}

/// Compilation state for delta analysis
#[derive(Debug, Clone)]
pub struct CompilationState {
    pub ptx_source: String,
    pub stage: CompilationStage,
    pub performance_stats: PerformanceStats,
    pub debug_mappings: Vec<DwarfMappingEntry>,
    pub variable_states: HashMap<String, VariableLocation>,
    pub ast_nodes: HashMap<String, AstNodeInfo>,
    pub ir_instructions: HashMap<String, IrInstructionInfo>,
    pub symbols: HashMap<String, SymbolInfo>,
    pub memory_regions: HashMap<u64, MemoryRegion>,
    pub register_values: HashMap<String, String>,
}

/// Memory region information
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub address: u64,
    pub size: u32,
    pub content_hash: u64,
}

/// Static analyzer for AST and IR changes
pub struct StaticAnalyzer {
    previous_ast_hashes: HashMap<String, u64>,
    previous_ir_hashes: HashMap<String, u64>,
}

impl StaticAnalyzer {
    pub fn new() -> Self {
        Self {
            previous_ast_hashes: HashMap::new(),
            previous_ir_hashes: HashMap::new(),
        }
    }

    pub fn analyze_changes(
        &mut self,
        _prev_source: &str,
        _current_source: &str,
    ) -> Result<StaticAnalysisDeltas, CheckpointError> {
        // Implementation would parse and compare AST/IR
        Ok(StaticAnalysisDeltas::empty())
    }
}

/// Dynamic analyzer for runtime state changes
pub struct DynamicAnalyzer {
    previous_variable_states: HashMap<String, VariableLocation>,
    previous_memory_state: HashMap<u64, u64>,
}

impl DynamicAnalyzer {
    pub fn new() -> Self {
        Self {
            previous_variable_states: HashMap::new(),
            previous_memory_state: HashMap::new(),
        }
    }

    pub fn analyze_runtime_changes(
        &mut self,
        _prev_checkpoint_id: &str,
        _current_debug_mappings: &[DwarfMappingEntry],
        _current_variables: &HashMap<String, VariableLocation>,
    ) -> Result<DynamicAnalysisDeltas, CheckpointError> {
        // Implementation would compare runtime states
        Ok(DynamicAnalysisDeltas::empty())
    }
}

/// Optimization suggestion based on delta analysis
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub category: String,
    pub description: String,
    pub estimated_impact: f64,
}

// Implementation of empty constructors for delta types
impl StaticAnalysisDeltas {
    pub fn empty() -> Self {
        Self {
            ast_changes: AstChanges {
                added_nodes: HashMap::new(),
                modified_nodes: HashMap::new(),
                removed_nodes: HashSet::new(),
                node_hash_changes: HashMap::new(),
            },
            ir_changes: IrChanges {
                added_instructions: HashMap::new(),
                modified_instructions: HashMap::new(),
                removed_instructions: HashSet::new(),
                basic_block_changes: HashMap::new(),
            },
            symbol_changes: SymbolChanges {
                added_symbols: HashMap::new(),
                modified_symbols: HashMap::new(),
                removed_symbols: HashSet::new(),
                scope_changes: HashMap::new(),
            },
            cfg_changes: CfgChanges {
                added_edges: Vec::new(),
                removed_edges: Vec::new(),
                modified_blocks: HashMap::new(),
            },
        }
    }
}

impl DynamicAnalysisDeltas {
    pub fn empty() -> Self {
        Self {
            variable_deltas: HashMap::new(),
            memory_deltas: Vec::new(),
            register_deltas: HashMap::new(),
            debug_mapping_deltas: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_delta_checkpoint_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = DeltaCheckpointManager::new(temp_dir.path(), true).unwrap();

        let state = CompilationState {
            ptx_source: ".version 7.0\n.target sm_50\n.entry test() { ret; }".to_string(),
            stage: CompilationStage::PtxParsing,
            performance_stats: PerformanceStats::default(),
            debug_mappings: Vec::new(),
            variable_states: HashMap::new(),
            ast_nodes: HashMap::new(),
            ir_instructions: HashMap::new(),
            symbols: HashMap::new(),
            memory_regions: HashMap::new(),
            register_values: HashMap::new(),
        };

        let delta_id = manager.create_delta_checkpoint(
            &state,
            None,
            "Initial delta checkpoint".to_string(),
        ).unwrap();

        assert!(manager.delta_checkpoints.contains_key(&delta_id));
    }

    #[test]
    fn test_optimization_suggestions() {
        let temp_dir = TempDir::new().unwrap();
        let manager = DeltaCheckpointManager::new(temp_dir.path(), true).unwrap();

        // Create a delta with performance regression
        let mut delta = DeltaCheckpoint {
            metadata: DeltaCheckpointMetadata {
                id: "test_delta".to_string(),
                timestamp: 0,
                base_checkpoint_id: None,
                delta_size_bytes: 0,
                compression_ratio: 1.0,
                created_at: "test".to_string(),
                description: "test".to_string(),
            },
            static_deltas: StaticAnalysisDeltas::empty(),
            dynamic_deltas: DynamicAnalysisDeltas::empty(),
            stage_transition: StageTransition {
                from_stage: CompilationStage::PtxParsing,
                to_stage: CompilationStage::LlvmGeneration,
                transition_time_ms: 0,
                intermediate_states: Vec::new(),
            },
            performance_delta: PerformanceDelta {
                time_delta_ms: 1500, // Performance regression
                memory_delta_bytes: 0,
                ir_size_delta_bytes: 0,
                spirv_size_delta_bytes: 0,
                optimization_impact: 0.0,
            },
        };

        // Add many IR instructions to trigger optimization suggestion
        for i in 0..150 {
            delta.static_deltas.ir_changes.added_instructions.insert(
                format!("instr_{}", i),
                IrInstructionInfo {
                    opcode: "add".to_string(),
                    operands: vec!["r0".to_string(), "r1".to_string()],
                    basic_block: "entry".to_string(),
                    debug_location: None,
                },
            );
        }

        let mut manager = manager;
        manager.delta_checkpoints.insert("test_delta".to_string(), delta);

        let suggestions = manager.get_optimization_suggestions("test_delta");
        assert!(!suggestions.is_empty());
        
        // Should have both IR bloat and performance regression suggestions
        assert!(suggestions.len() >= 2);
    }
}