// PTX State Recovery Tool
// This module provides functionality to recover PTX program state at arbitrary execution points

use crate::debug::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// State recovery manager for PTX debugging
#[derive(Debug, Serialize, Deserialize)]
pub struct PtxStateRecoveryManager {
    /// All debug mappings from PTX source to target architectures
    debug_mappings: Vec<DwarfMappingEntry>,

    /// Current execution state
    current_execution_state: Option<ExecutionState>,

    /// Breakpoints set by user
    breakpoints: Vec<Breakpoint>,

    /// Call stack for function debugging
    call_stack: Vec<CallFrame>,
}

/// Current execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    /// Current PTX source location
    current_location: PtxSourceLocation,

    /// Variable values at current location
    variable_state: HashMap<String, VariableValue>,

    /// Thread/warp state for GPU debugging
    thread_state: ThreadState,

    /// Memory state snapshots
    memory_snapshots: HashMap<String, MemorySnapshot>,
}

/// Variable value at execution point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Pointer(u64),
    Vector(Vec<VariableValue>),
    Unknown,
}

/// Thread/warp state for GPU debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadState {
    pub thread_id: (u32, u32, u32), // (x, y, z)
    pub block_id: (u32, u32, u32),  // (x, y, z)
    pub warp_id: u32,
    pub lane_id: u32,
    pub active_mask: u64, // Which threads in warp are active
}

/// Memory snapshot at execution point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub address_space: String, // global, shared, local, etc.
    pub base_address: u64,
    pub size: usize,
    pub data: Vec<u8>,
    pub timestamp: u64,
}

/// Breakpoint for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    pub id: u32,
    pub location: PtxSourceLocation,
    pub condition: Option<String>, // Optional condition expression
    pub hit_count: u32,
    pub enabled: bool,
}

/// Call frame for stack trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallFrame {
    pub function_name: String,
    pub location: PtxSourceLocation,
    pub local_variables: HashMap<String, VariableValue>,
}

impl PtxStateRecoveryManager {
    /// Create new state recovery manager
    pub fn new() -> Self {
        Self {
            debug_mappings: Vec::new(),
            current_execution_state: None,
            breakpoints: Vec::new(),
            call_stack: Vec::new(),
        }
    }

    /// Load debug mappings from file
    pub fn load_debug_mappings<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        self.debug_mappings = serde_json::from_str(&content)?;
        Ok(())
    }

    /// Save debug mappings to file
    pub fn save_debug_mappings<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(&self.debug_mappings)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Set current execution state
    pub fn set_execution_state(&mut self, state: ExecutionState) {
        self.current_execution_state = Some(state);
    }

    /// Get current execution state
    pub fn get_execution_state(&self) -> Option<&ExecutionState> {
        self.current_execution_state.as_ref()
    }

    /// Add breakpoint
    pub fn add_breakpoint(
        &mut self,
        location: PtxSourceLocation,
        condition: Option<String>,
    ) -> u32 {
        let id = self.breakpoints.len() as u32;
        let breakpoint = Breakpoint {
            id,
            location,
            condition,
            hit_count: 0,
            enabled: true,
        };
        self.breakpoints.push(breakpoint);
        id
    }

    /// Remove breakpoint
    pub fn remove_breakpoint(&mut self, id: u32) -> bool {
        if let Some(pos) = self.breakpoints.iter().position(|bp| bp.id == id) {
            self.breakpoints.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if execution should stop at current location
    pub fn should_break_at_location(
        &mut self,
        location: &PtxSourceLocation,
    ) -> Option<&Breakpoint> {
        for breakpoint in &mut self.breakpoints {
            if breakpoint.enabled && breakpoint.location == *location {
                breakpoint.hit_count += 1;
                // TODO: Evaluate condition if present
                return Some(breakpoint);
            }
        }
        None
    }

    /// Find PTX source location from target address
    pub fn find_ptx_location_from_target(
        &self,
        target_arch: &str,
        address: u64,
    ) -> Option<&PtxSourceLocation> {
        for mapping in &self.debug_mappings {
            for target_inst in &mapping.target_instructions {
                match (target_arch, target_inst) {
                    (
                        "amd_gcn",
                        TargetInstruction::AmdGcn {
                            address: inst_addr, ..
                        },
                    )
                    | (
                        "sass",
                        TargetInstruction::Sass {
                            address: inst_addr, ..
                        },
                    ) => {
                        if *inst_addr == address {
                            return Some(&mapping.ptx_location);
                        }
                    }
                    ("spirv", TargetInstruction::IntelSpirv { opcode, .. }) => {
                        // SPIRV doesn't have direct addresses, use opcode matching
                        // This would need runtime integration for proper mapping
                        if *opcode as u64 == address {
                            return Some(&mapping.ptx_location);
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Get variable value at current location
    pub fn get_variable_value(&self, var_name: &str) -> Option<&VariableValue> {
        self.current_execution_state
            .as_ref()?
            .variable_state
            .get(var_name)
    }

    /// Set variable value (for debugging/testing)
    pub fn set_variable_value(&mut self, var_name: String, value: VariableValue) {
        if let Some(ref mut state) = self.current_execution_state {
            state.variable_state.insert(var_name, value);
        }
    }

    /// Get memory snapshot
    pub fn get_memory_snapshot(&self, address_space: &str) -> Option<&MemorySnapshot> {
        self.current_execution_state
            .as_ref()?
            .memory_snapshots
            .get(address_space)
    }

    /// Take memory snapshot
    pub fn take_memory_snapshot(
        &mut self,
        address_space: String,
        base_address: u64,
        data: Vec<u8>,
    ) {
        if let Some(ref mut state) = self.current_execution_state {
            let snapshot = MemorySnapshot {
                address_space: address_space.clone(),
                base_address,
                size: data.len(),
                data,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };
            state.memory_snapshots.insert(address_space, snapshot);
        }
    }

    /// Push call frame
    pub fn push_call_frame(&mut self, function_name: String, location: PtxSourceLocation) {
        let frame = CallFrame {
            function_name,
            location,
            local_variables: HashMap::new(),
        };
        self.call_stack.push(frame);
    }

    /// Pop call frame
    pub fn pop_call_frame(&mut self) -> Option<CallFrame> {
        self.call_stack.pop()
    }

    /// Get call stack
    pub fn get_call_stack(&self) -> &[CallFrame] {
        &self.call_stack
    }

    /// Generate state dump for debugging
    pub fn generate_state_dump(&self) -> String {
        let mut dump = String::new();

        dump.push_str("=== PTX State Recovery Dump ===\n\n");

        if let Some(ref state) = self.current_execution_state {
            dump.push_str(&format!(
                "Current Location: {}:{}:{}\n",
                state.current_location.file,
                state.current_location.line,
                state.current_location.column
            ));

            dump.push_str(&format!(
                "Thread ID: ({}, {}, {})\n",
                state.thread_state.thread_id.0,
                state.thread_state.thread_id.1,
                state.thread_state.thread_id.2
            ));

            dump.push_str(&format!(
                "Block ID: ({}, {}, {})\n",
                state.thread_state.block_id.0,
                state.thread_state.block_id.1,
                state.thread_state.block_id.2
            ));

            dump.push_str(&format!(
                "Warp ID: {}, Lane ID: {}\n",
                state.thread_state.warp_id, state.thread_state.lane_id
            ));

            dump.push_str(&format!(
                "Active Mask: 0x{:016x}\n\n",
                state.thread_state.active_mask
            ));

            dump.push_str("Variables:\n");
            for (name, value) in &state.variable_state {
                dump.push_str(&format!("  {} = {:?}\n", name, value));
            }

            dump.push_str("\nMemory Snapshots:\n");
            for (space, snapshot) in &state.memory_snapshots {
                dump.push_str(&format!(
                    "  {}: {} bytes at 0x{:016x}\n",
                    space, snapshot.size, snapshot.base_address
                ));
            }
        } else {
            dump.push_str("No current execution state\n");
        }

        dump.push_str("\nCall Stack:\n");
        for (i, frame) in self.call_stack.iter().enumerate() {
            dump.push_str(&format!(
                "  #{}: {} at {}:{}:{}\n",
                i,
                frame.function_name,
                frame.location.file,
                frame.location.line,
                frame.location.column
            ));
        }

        dump.push_str(&format!("\nBreakpoints ({}):\n", self.breakpoints.len()));
        for bp in &self.breakpoints {
            let status = if bp.enabled { "enabled" } else { "disabled" };
            dump.push_str(&format!(
                "  #{}: {}:{}:{} ({}, hit {} times)\n",
                bp.id, bp.location.file, bp.location.line, bp.location.column, status, bp.hit_count
            ));
        }

        dump.push_str(&format!(
            "\nDebug Mappings: {} entries\n",
            self.debug_mappings.len()
        ));

        dump
    }

    /// Export state for external debugger
    pub fn export_gdb_compatible_info(&self) -> String {
        let mut output = String::new();

        // Generate GDB-compatible source location info
        if let Some(ref state) = self.current_execution_state {
            output.push_str(&format!("*stopped,reason=\"breakpoint-hit\",disp=\"keep\",bkptno=\"1\",frame={{addr=\"0x{:016x}\",func=\"unknown\",args=[],file=\"{}\",fullname=\"{}\",line=\"{}\"}}\n",
                0, // TODO: get actual address
                state.current_location.file,
                state.current_location.file,
                state.current_location.line));
        }

        output
    }

    /// Import state from external debugger
    pub fn import_debugger_state(
        &mut self,
        debugger_data: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Parse common debugger formats (GDB MI, VS Code DAP, etc.)
        // For now, implement basic JSON import
        if let Ok(state) = serde_json::from_str::<ExecutionState>(debugger_data) {
            self.current_execution_state = Some(state);
        }
        Ok(())
    }
}

impl Default for PtxStateRecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for creating test states
#[cfg(test)]
impl PtxStateRecoveryManager {
    pub fn create_test_state() -> Self {
        let mut manager = Self::new();

        // Add test execution state
        let state = ExecutionState {
            current_location: PtxSourceLocation {
                file: "test.ptx".to_string(),
                line: 42,
                column: 10,
                instruction_offset: 100,
            },
            variable_state: {
                let mut vars = HashMap::new();
                vars.insert("tid".to_string(), VariableValue::Integer(123));
                vars.insert("result".to_string(), VariableValue::Float(3.14));
                vars
            },
            thread_state: ThreadState {
                thread_id: (1, 0, 0),
                block_id: (0, 0, 0),
                warp_id: 0,
                lane_id: 1,
                active_mask: 0xffffffff,
            },
            memory_snapshots: HashMap::new(),
        };

        manager.set_execution_state(state);
        manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_recovery_basic() {
        let mut manager = PtxStateRecoveryManager::new();

        // Test breakpoint management
        let location = PtxSourceLocation {
            file: "test.ptx".to_string(),
            line: 10,
            column: 5,
            instruction_offset: 50,
        };

        let bp_id = manager.add_breakpoint(location.clone(), None);
        assert_eq!(bp_id, 0);

        // Test breakpoint hit
        let hit_bp = manager.should_break_at_location(&location);
        assert!(hit_bp.is_some());
        assert_eq!(hit_bp.unwrap().hit_count, 1);

        // Test breakpoint removal
        assert!(manager.remove_breakpoint(bp_id));
        assert!(!manager.remove_breakpoint(bp_id)); // Should fail second time
    }

    #[test]
    fn test_variable_state() {
        let mut manager = PtxStateRecoveryManager::create_test_state();

        // Test getting variable value
        let tid_value = manager.get_variable_value("tid");
        assert!(matches!(tid_value, Some(VariableValue::Integer(123))));

        // Test setting variable value
        manager.set_variable_value("new_var".to_string(), VariableValue::Boolean(true));
        let new_var_value = manager.get_variable_value("new_var");
        assert!(matches!(new_var_value, Some(VariableValue::Boolean(true))));
    }

    #[test]
    fn test_state_dump() {
        let manager = PtxStateRecoveryManager::create_test_state();
        let dump = manager.generate_state_dump();

        assert!(dump.contains("Current Location: test.ptx:42:10"));
        assert!(dump.contains("tid = Integer(123)"));
        assert!(dump.contains("Thread ID: (1, 0, 0)"));
    }
}
