//! GPU Record and Replay (gpu_rr) - rr-style debugging for GPU kernels
//!
//! This tool provides record/replay functionality for GPU kernel execution,
//! enabling deterministic debugging, time-travel debugging, and execution analysis.
//!
//! Usage:
//!   gpu_rr record [OPTIONS] <cubin>          Record kernel execution
//!   gpu_rr replay [OPTIONS] <recording>      Replay a recorded session
//!   gpu_rr analyze [OPTIONS] <recording>     Analyze a recording
//!   gpu_rr checkpoint [OPTIONS] <recording>  Create/manage checkpoints
//!
//! Examples:
//!   # Record kernel execution
//!   gpu_rr record --output trace.gpur kernel.cubin
//!
//!   # Replay with breakpoints
//!   gpu_rr replay --break 0x100 trace.gpur
//!
//!   # Analyze execution hotspots
//!   gpu_rr analyze --hotspots trace.gpur

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use ptx::sass::{
    CubinParser, EnhancedSassInstruction, PtxReconstructor, PtxRecoveryEngine, SassDisassembler,
    SassOpcodeClass, TextDisassemblyParser,
};

// ============================================================================
// Data Structures for Recording
// ============================================================================

/// GPU Recording file format version
const RECORDING_VERSION: u32 = 1;

/// Magic number for GPU recording files
const RECORDING_MAGIC: [u8; 4] = [b'G', b'P', b'U', b'R'];

/// Complete GPU recording containing all execution data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRecording {
    /// Recording metadata
    pub header: RecordingHeader,
    /// Recorded kernels
    pub kernels: Vec<KernelRecording>,
    /// Global memory snapshots
    pub memory_snapshots: Vec<MemorySnapshot>,
    /// Execution timeline
    pub timeline: ExecutionTimeline,
    /// Debug symbols if available
    pub debug_info: Option<DebugInfo>,
}

/// Recording header with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingHeader {
    /// Magic number
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Recording timestamp
    pub timestamp: u64,
    /// Recording ID
    pub id: String,
    /// Description
    pub description: String,
    /// Source CUBIN path
    pub cubin_path: Option<String>,
    /// GPU device info
    pub device_info: DeviceInfo,
    /// Recording configuration
    pub config: RecordingConfig,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// SM version (e.g., 75 for sm_75)
    pub sm_version: u32,
    /// Total global memory in bytes
    pub global_memory: u64,
    /// Number of SMs
    pub sm_count: u32,
    /// Max threads per block
    pub max_threads_per_block: u32,
    /// Warp size
    pub warp_size: u32,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            name: "Unknown GPU".to_string(),
            sm_version: 75,
            global_memory: 8 * 1024 * 1024 * 1024, // 8GB
            sm_count: 40,
            max_threads_per_block: 1024,
            warp_size: 32,
        }
    }
}

/// Recording configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingConfig {
    /// Record register values
    pub record_registers: bool,
    /// Record memory accesses
    pub record_memory: bool,
    /// Record predicate values
    pub record_predicates: bool,
    /// Memory snapshot frequency (every N instructions)
    pub memory_snapshot_frequency: u64,
    /// Instruction-level recording
    pub instruction_level: bool,
    /// Record control flow
    pub record_control_flow: bool,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            record_registers: true,
            record_memory: true,
            record_predicates: true,
            memory_snapshot_frequency: 1000,
            instruction_level: true,
            record_control_flow: true,
        }
    }
}

/// Recording for a single kernel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelRecording {
    /// Kernel name
    pub name: String,
    /// Kernel entry address
    pub entry_address: u64,
    /// Grid dimensions (blocks)
    pub grid_dim: [u32; 3],
    /// Block dimensions (threads)
    pub block_dim: [u32; 3],
    /// Shared memory size
    pub shared_mem_size: u32,
    /// SASS instructions
    pub instructions: Vec<EnhancedSassInstruction>,
    /// Instruction execution records
    pub execution_records: Vec<InstructionRecord>,
    /// Thread execution traces (per-warp)
    pub warp_traces: Vec<WarpTrace>,
    /// Kernel parameters
    pub parameters: Vec<KernelParameter>,
    /// Execution statistics
    pub stats: KernelStats,
}

/// Record of a single instruction execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionRecord {
    /// Instruction address
    pub address: u64,
    /// Execution sequence number
    pub sequence: u64,
    /// Warp ID that executed this
    pub warp_id: u32,
    /// Active thread mask
    pub active_mask: u64,
    /// Register state before execution
    pub registers_before: Option<RegisterState>,
    /// Register state after execution
    pub registers_after: Option<RegisterState>,
    /// Memory access (if any)
    pub memory_access: Option<MemoryAccess>,
    /// Predicate value (if predicated)
    pub predicate_value: Option<bool>,
    /// Control flow decision (for branches)
    pub control_flow: Option<ControlFlowRecord>,
    /// Cycle count (if available)
    pub cycles: Option<u64>,
}

/// Register state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterState {
    /// General purpose registers (R0-R255)
    pub general: BTreeMap<u8, u64>,
    /// Predicate registers (P0-P7)
    pub predicates: u8,
    /// Uniform registers (UR0-UR63)
    pub uniform: BTreeMap<u8, u64>,
    /// Special registers
    pub special: BTreeMap<String, u64>,
}

impl RegisterState {
    pub fn new() -> Self {
        Self {
            general: BTreeMap::new(),
            predicates: 0,
            uniform: BTreeMap::new(),
            special: BTreeMap::new(),
        }
    }

    pub fn set_general(&mut self, reg: u8, value: u64) {
        self.general.insert(reg, value);
    }

    pub fn get_general(&self, reg: u8) -> u64 {
        *self.general.get(&reg).unwrap_or(&0)
    }

    pub fn set_predicate(&mut self, pred: u8, value: bool) {
        if value {
            self.predicates |= 1 << pred;
        } else {
            self.predicates &= !(1 << pred);
        }
    }

    pub fn get_predicate(&self, pred: u8) -> bool {
        (self.predicates & (1 << pred)) != 0
    }
}

/// Memory access record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccess {
    /// Access type
    pub access_type: MemoryAccessType,
    /// Memory space
    pub space: MemorySpace,
    /// Address
    pub address: u64,
    /// Size in bytes
    pub size: u32,
    /// Data (for stores or value read)
    pub data: Vec<u8>,
    /// Thread mask for vectorized accesses
    pub thread_mask: u64,
}

/// Memory access type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryAccessType {
    Load,
    Store,
    Atomic,
    Prefetch,
}

/// Memory space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemorySpace {
    Global,
    Shared,
    Local,
    Constant,
    Texture,
    Surface,
}

/// Control flow record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlFlowRecord {
    /// Branch type
    pub branch_type: BranchType,
    /// Source address
    pub source: u64,
    /// Target address
    pub target: u64,
    /// Taken mask (which threads took the branch)
    pub taken_mask: u64,
    /// Fall-through mask
    pub fallthrough_mask: u64,
    /// Is this a divergent branch
    pub is_divergent: bool,
}

/// Branch type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchType {
    Conditional,
    Unconditional,
    Call,
    Return,
    Exit,
    Sync,
}

/// Warp-level execution trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpTrace {
    /// Warp ID
    pub warp_id: u32,
    /// Block ID
    pub block_id: [u32; 3],
    /// Initial active mask
    pub initial_active_mask: u64,
    /// Instruction sequence
    pub instruction_sequence: Vec<u64>, // Sequence numbers
    /// Divergence points
    pub divergence_points: Vec<DivergencePoint>,
    /// Reconvergence points
    pub reconvergence_points: Vec<u64>,
    /// Total instructions executed
    pub total_instructions: u64,
    /// Stall cycles
    pub stall_cycles: u64,
}

/// Divergence point in warp execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergencePoint {
    /// Address where divergence occurred
    pub address: u64,
    /// Sequence number
    pub sequence: u64,
    /// Taken mask
    pub taken_mask: u64,
    /// Not-taken mask
    pub not_taken_mask: u64,
    /// Reconvergence address
    pub reconvergence_address: u64,
}

/// Kernel parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelParameter {
    /// Parameter name (if known)
    pub name: Option<String>,
    /// Parameter offset in constant memory
    pub offset: u32,
    /// Parameter size
    pub size: u32,
    /// Parameter value
    pub value: Vec<u8>,
}

/// Kernel execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelStats {
    /// Total instructions executed
    pub total_instructions: u64,
    /// Instructions by opcode class
    pub instructions_by_class: HashMap<String, u64>,
    /// Memory operations
    pub memory_ops: MemoryStats,
    /// Control flow stats
    pub control_flow_stats: ControlFlowStats,
    /// Execution time (if available)
    pub execution_time_ns: Option<u64>,
    /// Total cycles
    pub total_cycles: Option<u64>,
}

/// Memory operation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub global_loads: u64,
    pub global_stores: u64,
    pub shared_loads: u64,
    pub shared_stores: u64,
    pub local_loads: u64,
    pub local_stores: u64,
    pub constant_loads: u64,
    pub atomics: u64,
    pub total_bytes_loaded: u64,
    pub total_bytes_stored: u64,
}

/// Control flow statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ControlFlowStats {
    pub branches: u64,
    pub divergent_branches: u64,
    pub calls: u64,
    pub returns: u64,
    pub barriers: u64,
    pub exits: u64,
}

/// Memory snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Snapshot ID
    pub id: u64,
    /// Sequence number when snapshot was taken
    pub sequence: u64,
    /// Memory regions
    pub regions: Vec<MemoryRegion>,
    /// Timestamp
    pub timestamp: u64,
}

/// Memory region in snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegion {
    /// Base address
    pub base: u64,
    /// Size
    pub size: u64,
    /// Memory space
    pub space: MemorySpace,
    /// Data (potentially compressed)
    pub data: Vec<u8>,
    /// Is data compressed
    pub compressed: bool,
}

/// Execution timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTimeline {
    /// Timeline events
    pub events: Vec<TimelineEvent>,
    /// Total duration in nanoseconds
    pub total_duration_ns: u64,
}

/// Timeline event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Event type
    pub event_type: TimelineEventType,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Associated data
    pub data: String,
}

/// Timeline event type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimelineEventType {
    KernelLaunch,
    KernelComplete,
    MemoryAlloc,
    MemoryFree,
    MemoryCopy,
    Synchronize,
    Checkpoint,
}

/// Debug information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugInfo {
    /// PTX source (if available)
    pub ptx_source: Option<String>,
    /// SASS to PTX mapping
    pub sass_ptx_mapping: HashMap<u64, PtxLocation>,
    /// Function names
    pub functions: HashMap<u64, String>,
    /// Source file mapping
    pub source_files: Vec<String>,
}

/// PTX source location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtxLocation {
    pub file_index: u32,
    pub line: u32,
    pub column: u32,
}

// ============================================================================
// Recorder Implementation
// ============================================================================

/// GPU execution recorder
pub struct GpuRecorder {
    /// Recording in progress
    recording: GpuRecording,
    /// Current sequence number
    sequence: u64,
    /// Recording configuration
    #[allow(dead_code)]
    config: RecordingConfig,
    /// Current kernel being recorded
    current_kernel: Option<KernelRecording>,
}

impl GpuRecorder {
    /// Create a new recorder
    pub fn new(config: RecordingConfig, description: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let id = format!("gpu_recording_{}_{:04x}", timestamp, rand::random::<u16>());

        let recording = GpuRecording {
            header: RecordingHeader {
                magic: RECORDING_MAGIC,
                version: RECORDING_VERSION,
                timestamp,
                id: id.clone(),
                description,
                cubin_path: None,
                device_info: DeviceInfo::default(),
                config: config.clone(),
            },
            kernels: Vec::new(),
            memory_snapshots: Vec::new(),
            timeline: ExecutionTimeline {
                events: Vec::new(),
                total_duration_ns: 0,
            },
            debug_info: None,
        };

        Self {
            recording,
            sequence: 0,
            config,
            current_kernel: None,
        }
    }

    /// Set device info
    pub fn set_device_info(&mut self, info: DeviceInfo) {
        self.recording.header.device_info = info;
    }

    /// Start recording a kernel
    pub fn start_kernel(&mut self, name: String, instructions: Vec<EnhancedSassInstruction>) {
        let kernel = KernelRecording {
            name,
            entry_address: instructions.first().map(|i| i.address).unwrap_or(0),
            grid_dim: [1, 1, 1],
            block_dim: [1, 1, 1],
            shared_mem_size: 0,
            instructions,
            execution_records: Vec::new(),
            warp_traces: Vec::new(),
            parameters: Vec::new(),
            stats: KernelStats {
                total_instructions: 0,
                instructions_by_class: HashMap::new(),
                memory_ops: MemoryStats::default(),
                control_flow_stats: ControlFlowStats::default(),
                execution_time_ns: None,
                total_cycles: None,
            },
        };

        self.current_kernel = Some(kernel);

        // Add timeline event
        self.recording.timeline.events.push(TimelineEvent {
            event_type: TimelineEventType::KernelLaunch,
            timestamp_ns: self.get_timestamp_ns(),
            data: self.current_kernel.as_ref().unwrap().name.clone(),
        });
    }

    /// Record an instruction execution
    pub fn record_instruction(
        &mut self,
        address: u64,
        warp_id: u32,
        active_mask: u64,
        registers_before: Option<RegisterState>,
        registers_after: Option<RegisterState>,
    ) {
        let record = InstructionRecord {
            address,
            sequence: self.sequence,
            warp_id,
            active_mask,
            registers_before,
            registers_after,
            memory_access: None,
            predicate_value: None,
            control_flow: None,
            cycles: None,
        };

        if let Some(ref mut kernel) = self.current_kernel {
            kernel.execution_records.push(record);
            kernel.stats.total_instructions += 1;
        }

        self.sequence += 1;
    }

    /// Record a memory access
    pub fn record_memory_access(
        &mut self,
        address: u64,
        mem_address: u64,
        access_type: MemoryAccessType,
        space: MemorySpace,
        size: u32,
        data: Vec<u8>,
    ) {
        if let Some(ref mut kernel) = self.current_kernel {
            if let Some(last_record) = kernel.execution_records.last_mut() {
                if last_record.address == address {
                    last_record.memory_access = Some(MemoryAccess {
                        access_type,
                        space,
                        address: mem_address,
                        size,
                        data,
                        thread_mask: last_record.active_mask,
                    });

                    // Update stats
                    match (access_type, space) {
                        (MemoryAccessType::Load, MemorySpace::Global) => {
                            kernel.stats.memory_ops.global_loads += 1;
                            kernel.stats.memory_ops.total_bytes_loaded += size as u64;
                        }
                        (MemoryAccessType::Store, MemorySpace::Global) => {
                            kernel.stats.memory_ops.global_stores += 1;
                            kernel.stats.memory_ops.total_bytes_stored += size as u64;
                        }
                        (MemoryAccessType::Load, MemorySpace::Shared) => {
                            kernel.stats.memory_ops.shared_loads += 1;
                        }
                        (MemoryAccessType::Store, MemorySpace::Shared) => {
                            kernel.stats.memory_ops.shared_stores += 1;
                        }
                        (MemoryAccessType::Atomic, _) => {
                            kernel.stats.memory_ops.atomics += 1;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Record a branch/control flow
    pub fn record_control_flow(
        &mut self,
        address: u64,
        branch_type: BranchType,
        target: u64,
        taken_mask: u64,
        fallthrough_mask: u64,
    ) {
        if let Some(ref mut kernel) = self.current_kernel {
            if let Some(last_record) = kernel.execution_records.last_mut() {
                if last_record.address == address {
                    let is_divergent =
                        taken_mask.count_ones() > 0 && fallthrough_mask.count_ones() > 0;

                    last_record.control_flow = Some(ControlFlowRecord {
                        branch_type,
                        source: address,
                        target,
                        taken_mask,
                        fallthrough_mask,
                        is_divergent,
                    });

                    // Update stats
                    kernel.stats.control_flow_stats.branches += 1;
                    if is_divergent {
                        kernel.stats.control_flow_stats.divergent_branches += 1;
                    }
                }
            }
        }
    }

    /// Complete kernel recording
    pub fn complete_kernel(&mut self) {
        if let Some(kernel) = self.current_kernel.take() {
            self.recording.kernels.push(kernel);

            self.recording.timeline.events.push(TimelineEvent {
                event_type: TimelineEventType::KernelComplete,
                timestamp_ns: self.get_timestamp_ns(),
                data: self.recording.kernels.last().unwrap().name.clone(),
            });
        }
    }

    /// Take a memory snapshot
    pub fn snapshot_memory(&mut self, regions: Vec<MemoryRegion>) {
        let snapshot = MemorySnapshot {
            id: self.recording.memory_snapshots.len() as u64,
            sequence: self.sequence,
            regions,
            timestamp: self.get_timestamp_ns(),
        };

        self.recording.memory_snapshots.push(snapshot);

        self.recording.timeline.events.push(TimelineEvent {
            event_type: TimelineEventType::Checkpoint,
            timestamp_ns: self.get_timestamp_ns(),
            data: format!(
                "Memory snapshot {}",
                self.recording.memory_snapshots.len() - 1
            ),
        });
    }

    /// Set debug info
    pub fn set_debug_info(&mut self, debug_info: DebugInfo) {
        self.recording.debug_info = Some(debug_info);
    }

    /// Finalize and get the recording
    pub fn finalize(mut self) -> GpuRecording {
        // Complete any in-progress kernel
        if self.current_kernel.is_some() {
            self.complete_kernel();
        }

        // Set total duration
        if let (Some(first), Some(last)) = (
            self.recording.timeline.events.first(),
            self.recording.timeline.events.last(),
        ) {
            self.recording.timeline.total_duration_ns = last.timestamp_ns - first.timestamp_ns;
        }

        self.recording
    }

    /// Save recording to file
    pub fn save(&self, path: &PathBuf) -> io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.recording)?;
        Ok(())
    }

    fn get_timestamp_ns(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

// ============================================================================
// Replayer Implementation
// ============================================================================

/// GPU execution replayer
pub struct GpuReplayer {
    /// Recording being replayed
    recording: GpuRecording,
    /// Current position in the recording
    position: ReplayPosition,
    /// Breakpoints
    breakpoints: Vec<Breakpoint>,
    /// Current register state
    register_state: RegisterState,
    /// Replay mode
    #[allow(dead_code)]
    mode: ReplayMode,
}

/// Position in replay
#[derive(Debug, Clone, Default)]
pub struct ReplayPosition {
    /// Current kernel index
    pub kernel_index: usize,
    /// Current instruction record index
    pub record_index: usize,
    /// Current sequence number
    pub sequence: u64,
}

/// Breakpoint
#[derive(Debug, Clone)]
pub struct Breakpoint {
    /// Breakpoint ID
    pub id: u32,
    /// Breakpoint type
    pub bp_type: BreakpointType,
    /// Is enabled
    pub enabled: bool,
    /// Hit count
    pub hit_count: u64,
}

/// Breakpoint type
#[derive(Debug, Clone)]
pub enum BreakpointType {
    /// Break at address
    Address(u64),
    /// Break at sequence number
    Sequence(u64),
    /// Break on memory access
    MemoryAccess { address: u64, size: u32 },
    /// Break on condition
    Condition(String),
}

/// Replay mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayMode {
    /// Step by step
    Step,
    /// Continue until breakpoint
    Continue,
    /// Reverse step
    ReverseStep,
    /// Reverse continue
    ReverseContinue,
}

/// Replay result
#[derive(Debug)]
pub enum ReplayResult {
    /// Reached end of recording
    EndOfRecording,
    /// Hit breakpoint
    Breakpoint(u32),
    /// Stepped to next instruction
    Stepped,
    /// Reached start of recording (for reverse)
    StartOfRecording,
    /// Error
    Error(String),
}

impl GpuReplayer {
    /// Load recording from file
    pub fn load(path: &PathBuf) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let recording: GpuRecording = serde_json::from_reader(reader)?;

        Ok(Self {
            recording,
            position: ReplayPosition::default(),
            breakpoints: Vec::new(),
            register_state: RegisterState::new(),
            mode: ReplayMode::Step,
        })
    }

    /// Get current instruction record
    pub fn current_record(&self) -> Option<&InstructionRecord> {
        self.recording
            .kernels
            .get(self.position.kernel_index)?
            .execution_records
            .get(self.position.record_index)
    }

    /// Get current kernel
    pub fn current_kernel(&self) -> Option<&KernelRecording> {
        self.recording.kernels.get(self.position.kernel_index)
    }

    /// Get instruction at current position
    pub fn current_instruction(&self) -> Option<&EnhancedSassInstruction> {
        let record = self.current_record()?;
        let kernel = self.current_kernel()?;
        kernel
            .instructions
            .iter()
            .find(|i| i.address == record.address)
    }

    /// Add a breakpoint
    pub fn add_breakpoint(&mut self, bp_type: BreakpointType) -> u32 {
        let id = self.breakpoints.len() as u32;
        self.breakpoints.push(Breakpoint {
            id,
            bp_type,
            enabled: true,
            hit_count: 0,
        });
        id
    }

    /// Remove a breakpoint
    pub fn remove_breakpoint(&mut self, id: u32) -> bool {
        if let Some(pos) = self.breakpoints.iter().position(|bp| bp.id == id) {
            self.breakpoints.remove(pos);
            true
        } else {
            false
        }
    }

    /// Step forward one instruction
    pub fn step(&mut self) -> ReplayResult {
        // Get record data we need before mutating
        let record_data = self.current_record().map(|r| {
            (
                r.address,
                r.sequence,
                r.registers_after.clone(),
                r.memory_access.clone(),
            )
        });

        let Some((address, sequence, registers_after, memory_access)) = record_data else {
            return ReplayResult::EndOfRecording;
        };

        // Update register state if available
        if let Some(regs) = registers_after {
            self.register_state = regs;
        }

        // Check breakpoints
        for bp in &mut self.breakpoints {
            if bp.enabled {
                let hit = match &bp.bp_type {
                    BreakpointType::Address(addr) => address == *addr,
                    BreakpointType::Sequence(seq) => sequence == *seq,
                    BreakpointType::MemoryAccess {
                        address: mem_addr,
                        size: _,
                    } => {
                        if let Some(ref access) = memory_access {
                            access.address <= *mem_addr
                                && *mem_addr < access.address + access.size as u64
                        } else {
                            false
                        }
                    }
                    BreakpointType::Condition(_) => false,
                };
                if hit {
                    bp.hit_count += 1;
                    return ReplayResult::Breakpoint(bp.id);
                }
            }
        }

        // Move to next instruction
        self.position.record_index += 1;
        self.position.sequence = sequence + 1;

        // Check if we need to move to next kernel
        let kernel_len = self
            .current_kernel()
            .map(|k| k.execution_records.len())
            .unwrap_or(0);
        if self.position.record_index >= kernel_len {
            self.position.kernel_index += 1;
            self.position.record_index = 0;

            if self.position.kernel_index >= self.recording.kernels.len() {
                return ReplayResult::EndOfRecording;
            }
        }

        ReplayResult::Stepped
    }

    /// Step backward one instruction
    pub fn reverse_step(&mut self) -> ReplayResult {
        if self.position.record_index == 0 {
            if self.position.kernel_index == 0 {
                return ReplayResult::StartOfRecording;
            }
            self.position.kernel_index -= 1;
            let kernel_len = self
                .current_kernel()
                .map(|k| k.execution_records.len())
                .unwrap_or(1);
            self.position.record_index = kernel_len.saturating_sub(1);
        } else {
            self.position.record_index -= 1;
        }

        // Get register state and sequence before mutating
        let record_data = self
            .current_record()
            .map(|r| (r.registers_before.clone(), r.sequence));

        // Restore register state
        if let Some((registers_before, sequence)) = record_data {
            if let Some(regs) = registers_before {
                self.register_state = regs;
            }
            self.position.sequence = sequence;
        }

        ReplayResult::Stepped
    }

    /// Continue execution until breakpoint or end
    pub fn continue_execution(&mut self) -> ReplayResult {
        loop {
            match self.step() {
                ReplayResult::Stepped => continue,
                other => return other,
            }
        }
    }

    /// Reverse continue until breakpoint or start
    pub fn reverse_continue(&mut self) -> ReplayResult {
        loop {
            match self.reverse_step() {
                ReplayResult::Stepped => {
                    // Get record data for breakpoint checking
                    let record_data = self
                        .current_record()
                        .map(|r| (r.address, r.sequence, r.memory_access.clone()));

                    // Check breakpoints
                    if let Some((address, sequence, memory_access)) = record_data {
                        for bp in &mut self.breakpoints {
                            if bp.enabled {
                                let hit = match &bp.bp_type {
                                    BreakpointType::Address(addr) => address == *addr,
                                    BreakpointType::Sequence(seq) => sequence == *seq,
                                    BreakpointType::MemoryAccess {
                                        address: mem_addr,
                                        size: _,
                                    } => {
                                        if let Some(ref access) = memory_access {
                                            access.address <= *mem_addr
                                                && *mem_addr < access.address + access.size as u64
                                        } else {
                                            false
                                        }
                                    }
                                    BreakpointType::Condition(_) => false,
                                };
                                if hit {
                                    bp.hit_count += 1;
                                    return ReplayResult::Breakpoint(bp.id);
                                }
                            }
                        }
                    }
                    continue;
                }
                other => return other,
            }
        }
    }

    /// Go to specific sequence number
    pub fn goto_sequence(&mut self, target_sequence: u64) -> ReplayResult {
        // Reset to start
        self.position = ReplayPosition::default();

        // Search for the sequence
        for (ki, kernel) in self.recording.kernels.iter().enumerate() {
            for (ri, record) in kernel.execution_records.iter().enumerate() {
                if record.sequence == target_sequence {
                    self.position.kernel_index = ki;
                    self.position.record_index = ri;
                    self.position.sequence = target_sequence;

                    if let Some(ref regs) = record.registers_before {
                        self.register_state = regs.clone();
                    }

                    return ReplayResult::Stepped;
                }
            }
        }

        ReplayResult::Error(format!("Sequence {} not found", target_sequence))
    }

    /// Get register state
    pub fn get_registers(&self) -> &RegisterState {
        &self.register_state
    }

    /// Get recording info
    pub fn get_recording(&self) -> &GpuRecording {
        &self.recording
    }

    /// Get current position
    pub fn get_position(&self) -> &ReplayPosition {
        &self.position
    }

    #[allow(dead_code)]
    fn check_breakpoint(&self, bp_type: &BreakpointType, record: &InstructionRecord) -> bool {
        match bp_type {
            BreakpointType::Address(addr) => record.address == *addr,
            BreakpointType::Sequence(seq) => record.sequence == *seq,
            BreakpointType::MemoryAccess { address, size: _ } => {
                if let Some(ref access) = record.memory_access {
                    access.address <= *address && *address < access.address + access.size as u64
                } else {
                    false
                }
            }
            BreakpointType::Condition(_) => false, // TODO: Implement condition evaluation
        }
    }
}

// ============================================================================
// Analyzer Implementation
// ============================================================================

/// Recording analyzer
pub struct RecordingAnalyzer {
    recording: GpuRecording,
}

/// Analysis report
#[derive(Debug)]
pub struct AnalysisReport {
    /// Summary statistics
    pub summary: AnalysisSummary,
    /// Hotspots
    pub hotspots: Vec<Hotspot>,
    /// Memory access patterns
    pub memory_patterns: MemoryPatternAnalysis,
    /// Divergence analysis
    pub divergence_analysis: DivergenceAnalysis,
    /// Performance recommendations
    pub recommendations: Vec<String>,
}

/// Summary statistics
#[derive(Debug, Serialize)]
pub struct AnalysisSummary {
    pub total_kernels: usize,
    pub total_instructions: u64,
    pub total_memory_ops: u64,
    pub total_branches: u64,
    pub divergent_branch_ratio: f64,
    pub memory_efficiency: f64,
}

/// Execution hotspot
#[derive(Debug)]
pub struct Hotspot {
    /// Address
    pub address: u64,
    /// Execution count
    pub count: u64,
    /// Percentage of total
    pub percentage: f64,
    /// Instruction opcode
    pub opcode: String,
    /// Average cycles (if available)
    pub avg_cycles: Option<f64>,
}

/// Memory access pattern analysis
#[derive(Debug)]
pub struct MemoryPatternAnalysis {
    /// Coalesced access ratio
    pub coalesced_ratio: f64,
    /// Bank conflict ratio
    pub bank_conflict_ratio: f64,
    /// Cache hit ratio (estimated)
    pub cache_hit_ratio: f64,
    /// Memory bandwidth utilization
    pub bandwidth_utilization: f64,
}

/// Divergence analysis
#[derive(Debug)]
pub struct DivergenceAnalysis {
    /// Total branches
    pub total_branches: u64,
    /// Divergent branches
    pub divergent_branches: u64,
    /// Average active threads
    pub avg_active_threads: f64,
    /// Worst divergence points
    pub worst_divergence_points: Vec<(u64, f64)>, // (address, divergence ratio)
}

impl RecordingAnalyzer {
    /// Create analyzer from recording
    pub fn new(recording: GpuRecording) -> Self {
        Self { recording }
    }

    /// Load from file
    pub fn load(path: &PathBuf) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let recording: GpuRecording = serde_json::from_reader(reader)?;
        Ok(Self::new(recording))
    }

    /// Analyze the recording
    pub fn analyze(&self) -> AnalysisReport {
        let summary = self.compute_summary();
        let hotspots = self.find_hotspots(10);
        let memory_patterns = self.analyze_memory_patterns();
        let divergence_analysis = self.analyze_divergence();
        let recommendations =
            self.generate_recommendations(&summary, &memory_patterns, &divergence_analysis);

        AnalysisReport {
            summary,
            hotspots,
            memory_patterns,
            divergence_analysis,
            recommendations,
        }
    }

    fn compute_summary(&self) -> AnalysisSummary {
        let total_kernels = self.recording.kernels.len();
        let mut total_instructions = 0u64;
        let mut total_memory_ops = 0u64;
        let mut total_branches = 0u64;
        let mut divergent_branches = 0u64;

        for kernel in &self.recording.kernels {
            total_instructions += kernel.stats.total_instructions;
            total_memory_ops += kernel.stats.memory_ops.global_loads
                + kernel.stats.memory_ops.global_stores
                + kernel.stats.memory_ops.shared_loads
                + kernel.stats.memory_ops.shared_stores;
            total_branches += kernel.stats.control_flow_stats.branches;
            divergent_branches += kernel.stats.control_flow_stats.divergent_branches;
        }

        let divergent_branch_ratio = if total_branches > 0 {
            divergent_branches as f64 / total_branches as f64
        } else {
            0.0
        };

        AnalysisSummary {
            total_kernels,
            total_instructions,
            total_memory_ops,
            total_branches,
            divergent_branch_ratio,
            memory_efficiency: 0.8, // Placeholder
        }
    }

    fn find_hotspots(&self, limit: usize) -> Vec<Hotspot> {
        let mut address_counts: HashMap<u64, (u64, String)> = HashMap::new();

        for kernel in &self.recording.kernels {
            for record in &kernel.execution_records {
                let entry = address_counts
                    .entry(record.address)
                    .or_insert((0, String::new()));
                entry.0 += 1;
                if entry.1.is_empty() {
                    if let Some(inst) = kernel
                        .instructions
                        .iter()
                        .find(|i| i.address == record.address)
                    {
                        entry.1 = inst.opcode.clone();
                    }
                }
            }
        }

        let total: u64 = address_counts.values().map(|(c, _)| c).sum();

        let mut hotspots: Vec<_> = address_counts
            .into_iter()
            .map(|(addr, (count, opcode))| Hotspot {
                address: addr,
                count,
                percentage: if total > 0 {
                    count as f64 / total as f64 * 100.0
                } else {
                    0.0
                },
                opcode,
                avg_cycles: None,
            })
            .collect();

        hotspots.sort_by(|a, b| b.count.cmp(&a.count));
        hotspots.truncate(limit);
        hotspots
    }

    fn analyze_memory_patterns(&self) -> MemoryPatternAnalysis {
        // Simplified analysis
        MemoryPatternAnalysis {
            coalesced_ratio: 0.75,
            bank_conflict_ratio: 0.05,
            cache_hit_ratio: 0.6,
            bandwidth_utilization: 0.7,
        }
    }

    fn analyze_divergence(&self) -> DivergenceAnalysis {
        let mut total_branches = 0u64;
        let mut divergent_branches = 0u64;
        let mut total_active_threads = 0u64;
        let mut total_records = 0u64;
        let mut divergence_points: HashMap<u64, (u64, u64)> = HashMap::new(); // (divergent, total)

        for kernel in &self.recording.kernels {
            for record in &kernel.execution_records {
                total_records += 1;
                total_active_threads += record.active_mask.count_ones() as u64;

                if let Some(ref cf) = record.control_flow {
                    if matches!(cf.branch_type, BranchType::Conditional) {
                        total_branches += 1;
                        let entry = divergence_points.entry(record.address).or_insert((0, 0));
                        entry.1 += 1;

                        if cf.is_divergent {
                            divergent_branches += 1;
                            entry.0 += 1;
                        }
                    }
                }
            }
        }

        let avg_active_threads = if total_records > 0 {
            total_active_threads as f64 / total_records as f64
        } else {
            0.0
        };

        let mut worst_points: Vec<_> = divergence_points
            .into_iter()
            .map(|(addr, (div, total))| (addr, div as f64 / total as f64))
            .collect();
        worst_points.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        worst_points.truncate(5);

        DivergenceAnalysis {
            total_branches,
            divergent_branches,
            avg_active_threads,
            worst_divergence_points: worst_points,
        }
    }

    fn generate_recommendations(
        &self,
        summary: &AnalysisSummary,
        memory: &MemoryPatternAnalysis,
        divergence: &DivergenceAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if summary.divergent_branch_ratio > 0.3 {
            recommendations.push(
                "High branch divergence detected. Consider restructuring control flow or using predication."
                    .to_string(),
            );
        }

        if memory.coalesced_ratio < 0.5 {
            recommendations.push(
                "Low memory coalescing. Consider reorganizing data layout for better coalescing."
                    .to_string(),
            );
        }

        if memory.bank_conflict_ratio > 0.2 {
            recommendations.push(
                "High shared memory bank conflicts. Consider padding shared memory arrays."
                    .to_string(),
            );
        }

        if divergence.avg_active_threads < 16.0 {
            recommendations.push(
                "Low warp occupancy. Consider increasing parallelism or reducing divergence."
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("No significant performance issues detected.".to_string());
        }

        recommendations
    }

    /// Print analysis report
    pub fn print_report(&self, report: &AnalysisReport) {
        println!("=== GPU Recording Analysis Report ===\n");

        println!("Summary:");
        println!("  Total kernels: {}", report.summary.total_kernels);
        println!(
            "  Total instructions: {}",
            report.summary.total_instructions
        );
        println!(
            "  Total memory operations: {}",
            report.summary.total_memory_ops
        );
        println!("  Total branches: {}", report.summary.total_branches);
        println!(
            "  Divergent branch ratio: {:.2}%",
            report.summary.divergent_branch_ratio * 100.0
        );
        println!();

        println!("Top {} Hotspots:", report.hotspots.len());
        for (i, hotspot) in report.hotspots.iter().enumerate() {
            println!(
                "  {}. 0x{:08x} {:12} - {} executions ({:.2}%)",
                i + 1,
                hotspot.address,
                hotspot.opcode,
                hotspot.count,
                hotspot.percentage
            );
        }
        println!();

        println!("Memory Patterns:");
        println!(
            "  Coalesced access ratio: {:.2}%",
            report.memory_patterns.coalesced_ratio * 100.0
        );
        println!(
            "  Bank conflict ratio: {:.2}%",
            report.memory_patterns.bank_conflict_ratio * 100.0
        );
        println!(
            "  Cache hit ratio: {:.2}%",
            report.memory_patterns.cache_hit_ratio * 100.0
        );
        println!(
            "  Bandwidth utilization: {:.2}%",
            report.memory_patterns.bandwidth_utilization * 100.0
        );
        println!();

        println!("Divergence Analysis:");
        println!(
            "  Total branches: {}",
            report.divergence_analysis.total_branches
        );
        println!(
            "  Divergent branches: {}",
            report.divergence_analysis.divergent_branches
        );
        println!(
            "  Average active threads: {:.1}",
            report.divergence_analysis.avg_active_threads
        );
        if !report
            .divergence_analysis
            .worst_divergence_points
            .is_empty()
        {
            println!("  Worst divergence points:");
            for (addr, ratio) in &report.divergence_analysis.worst_divergence_points {
                println!("    0x{:08x}: {:.2}% divergent", addr, ratio * 100.0);
            }
        }
        println!();

        println!("Recommendations:");
        for rec in &report.recommendations {
            println!("  - {}", rec);
        }
    }
}

// ============================================================================
// CLI Implementation
// ============================================================================

/// CLI arguments
struct Args {
    command: Command,
    verbose: bool,
}

enum Command {
    Record {
        input: String,
        output: PathBuf,
        config: RecordingConfig,
    },
    Replay {
        input: PathBuf,
        breakpoints: Vec<u64>,
        interactive: bool,
    },
    Analyze {
        input: PathBuf,
        output: Option<PathBuf>,
        hotspots: bool,
    },
    Info {
        input: PathBuf,
    },
    Ptx {
        input: PathBuf,
        output: Option<PathBuf>,
        address: Option<u64>,
    },
}

fn parse_args() -> Result<Args, String> {
    let mut argv: Vec<String> = std::env::args().collect();
    argv.remove(0);

    if argv.is_empty() {
        print_help();
        std::process::exit(0);
    }

    let mut verbose = false;
    let mut i = 0;

    // Parse global options
    while i < argv.len() {
        match argv[i].as_str() {
            "-v" | "--verbose" => {
                verbose = true;
                argv.remove(i);
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => i += 1,
        }
    }

    if argv.is_empty() {
        return Err("No command specified".to_string());
    }

    let command = match argv[0].as_str() {
        "record" => parse_record_command(&argv[1..])?,
        "replay" => parse_replay_command(&argv[1..])?,
        "analyze" => parse_analyze_command(&argv[1..])?,
        "info" => parse_info_command(&argv[1..])?,
        "ptx" => parse_ptx_command(&argv[1..])?,
        other => return Err(format!("Unknown command: {}", other)),
    };

    Ok(Args { command, verbose })
}

fn parse_record_command(args: &[String]) -> Result<Command, String> {
    let mut input = String::new();
    let mut output = PathBuf::from("recording.gpur");
    let mut config = RecordingConfig::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err("--output requires an argument".to_string());
                }
                output = PathBuf::from(&args[i]);
            }
            "--no-registers" => config.record_registers = false,
            "--no-memory" => config.record_memory = false,
            "--snapshot-freq" => {
                i += 1;
                if i >= args.len() {
                    return Err("--snapshot-freq requires an argument".to_string());
                }
                config.memory_snapshot_frequency =
                    args[i].parse().map_err(|_| "Invalid frequency")?;
            }
            arg if arg.starts_with('-') => {
                return Err(format!("Unknown option: {}", arg));
            }
            _ => {
                if input.is_empty() {
                    input = args[i].clone();
                }
            }
        }
        i += 1;
    }

    if input.is_empty() {
        return Err("Input file required".to_string());
    }

    Ok(Command::Record {
        input,
        output,
        config,
    })
}

fn parse_replay_command(args: &[String]) -> Result<Command, String> {
    let mut input = PathBuf::new();
    let mut breakpoints = Vec::new();
    let mut interactive = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-b" | "--break" => {
                i += 1;
                if i >= args.len() {
                    return Err("--break requires an argument".to_string());
                }
                let addr = if args[i].starts_with("0x") {
                    u64::from_str_radix(&args[i][2..], 16).map_err(|_| "Invalid address")?
                } else {
                    args[i].parse().map_err(|_| "Invalid address")?
                };
                breakpoints.push(addr);
            }
            "-i" | "--interactive" => interactive = true,
            arg if arg.starts_with('-') => {
                return Err(format!("Unknown option: {}", arg));
            }
            _ => {
                if input.as_os_str().is_empty() {
                    input = PathBuf::from(&args[i]);
                }
            }
        }
        i += 1;
    }

    if input.as_os_str().is_empty() {
        return Err("Input file required".to_string());
    }

    Ok(Command::Replay {
        input,
        breakpoints,
        interactive,
    })
}

fn parse_analyze_command(args: &[String]) -> Result<Command, String> {
    let mut input = PathBuf::new();
    let mut output = None;
    let mut hotspots = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err("--output requires an argument".to_string());
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "--no-hotspots" => hotspots = false,
            arg if arg.starts_with('-') => {
                return Err(format!("Unknown option: {}", arg));
            }
            _ => {
                if input.as_os_str().is_empty() {
                    input = PathBuf::from(&args[i]);
                }
            }
        }
        i += 1;
    }

    if input.as_os_str().is_empty() {
        return Err("Input file required".to_string());
    }

    Ok(Command::Analyze {
        input,
        output,
        hotspots,
    })
}

fn parse_info_command(args: &[String]) -> Result<Command, String> {
    if args.is_empty() {
        return Err("Input file required".to_string());
    }
    Ok(Command::Info {
        input: PathBuf::from(&args[0]),
    })
}

fn parse_ptx_command(args: &[String]) -> Result<Command, String> {
    let mut input = PathBuf::new();
    let mut output = None;
    let mut address = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err("--output requires an argument".to_string());
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "-a" | "--address" => {
                i += 1;
                if i >= args.len() {
                    return Err("--address requires an argument".to_string());
                }
                let addr = if args[i].starts_with("0x") {
                    u64::from_str_radix(&args[i][2..], 16).map_err(|_| "Invalid address")?
                } else {
                    args[i].parse().map_err(|_| "Invalid address")?
                };
                address = Some(addr);
            }
            arg if arg.starts_with('-') => {
                return Err(format!("Unknown option: {}", arg));
            }
            _ => {
                if input.as_os_str().is_empty() {
                    input = PathBuf::from(&args[i]);
                }
            }
        }
        i += 1;
    }

    if input.as_os_str().is_empty() {
        return Err("Input file required".to_string());
    }

    Ok(Command::Ptx {
        input,
        output,
        address,
    })
}

fn print_help() {
    println!(
        r#"GPU Record and Replay (gpu_rr) - rr-style debugging for GPU kernels

USAGE:
    gpu_rr [OPTIONS] <COMMAND>

COMMANDS:
    record      Record GPU kernel execution
    replay      Replay a recorded session
    analyze     Analyze a recording
    info        Show recording information
    ptx         Recover/reconstruct PTX from SASS using debug info

OPTIONS:
    -v, --verbose    Verbose output
    -h, --help       Show this help message

RECORD OPTIONS:
    -o, --output <file>       Output recording file (default: recording.gpur)
    --no-registers            Don't record register values
    --no-memory               Don't record memory accesses
    --snapshot-freq <N>       Memory snapshot frequency (default: 1000)

REPLAY OPTIONS:
    -b, --break <addr>        Set breakpoint at address (can be used multiple times)
    -i, --interactive         Interactive replay mode

ANALYZE OPTIONS:
    -o, --output <file>       Output report file (JSON)
    --no-hotspots             Don't analyze hotspots

PTX OPTIONS:
    -o, --output <file>       Output PTX file
    -a, --address <addr>      Show PTX for specific SASS address

EXAMPLES:
    # Record kernel execution from CUBIN
    gpu_rr record kernel.cubin -o trace.gpur

    # Replay with breakpoint
    gpu_rr replay --break 0x100 trace.gpur

    # Analyze execution hotspots
    gpu_rr analyze --hotspots trace.gpur

    # Show recording info
    gpu_rr info trace.gpur

    # Recover PTX from CUBIN (uses debug info)
    gpu_rr ptx kernel.cubin -o recovered.ptx

    # Reconstruct PTX from text SASS
    gpu_rr ptx cuobjdump_output.txt -o reconstructed.ptx
"#
    );
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    match args.command {
        Command::Record {
            input,
            output,
            config,
        } => run_record(&input, &output, config, args.verbose),
        Command::Replay {
            input,
            breakpoints,
            interactive,
        } => run_replay(&input, breakpoints, interactive, args.verbose),
        Command::Analyze {
            input,
            output,
            hotspots,
        } => run_analyze(&input, output.as_ref(), hotspots, args.verbose),
        Command::Info { input } => run_info(&input, args.verbose),
        Command::Ptx {
            input,
            output,
            address,
        } => run_ptx(&input, output.as_ref(), address, args.verbose),
    }
}

fn run_record(
    input: &str,
    output: &PathBuf,
    config: RecordingConfig,
    verbose: bool,
) -> Result<(), String> {
    if verbose {
        eprintln!("Recording from: {}", input);
        eprintln!("Output: {:?}", output);
    }

    // Load CUBIN or SASS text
    let content = fs::read(input).map_err(|e| format!("Failed to read input: {}", e))?;

    let (instructions, kernel_name): (Vec<EnhancedSassInstruction>, String) =
        if content.starts_with(&[0x7f, b'E', b'L', b'F']) {
            // ELF/CUBIN file
            let parser = CubinParser::new(content);
            let cubin = parser
                .parse()
                .map_err(|e| format!("Failed to parse CUBIN: {}", e))?;

            if cubin.kernels.is_empty() {
                return Err("No kernels found in CUBIN".to_string());
            }

            let kernel = &cubin.kernels[0];
            let disasm = SassDisassembler::new(cubin.sm_version)
                .map_err(|e| format!("Disassembler error: {}", e))?;
            let instructions = disasm.disassemble(&kernel.code, kernel.address);
            (instructions, kernel.name.clone())
        } else {
            // Text SASS (cuobjdump output)
            let text = String::from_utf8_lossy(&content);
            let instructions = TextDisassemblyParser::parse_cuobjdump_output(&text);

            if instructions.is_empty() {
                return Err("No instructions found in input".to_string());
            }

            // Get function name from first instruction
            let kernel_name = instructions
                .first()
                .and_then(|i| i.function_name.clone())
                .unwrap_or_else(|| "kernel".to_string());

            (instructions, kernel_name)
        };

    if verbose {
        eprintln!(
            "Found kernel '{}' with {} instructions",
            kernel_name,
            instructions.len()
        );
    }

    // Create recorder
    let mut recorder = GpuRecorder::new(config, format!("Recording of {}", kernel_name));
    recorder.recording.header.cubin_path = Some(input.to_string());

    // Record instructions (simulated execution)
    recorder.start_kernel(kernel_name.clone(), instructions.clone());

    // Simulate execution by recording each instruction
    for (i, inst) in instructions.iter().enumerate() {
        let regs_before = RegisterState::new();
        let mut regs_after = RegisterState::new();

        // Simulate register state changes
        for (j, dest) in inst.dest_operands.iter().enumerate() {
            if let ptx::sass::SassOperand::Register(reg) = dest {
                regs_after.set_general(reg.number as u8, (i * 100 + j) as u64);
            }
        }

        recorder.record_instruction(
            inst.address,
            0,          // warp_id
            0xFFFFFFFF, // all threads active
            Some(regs_before),
            Some(regs_after),
        );

        // Record memory accesses
        match inst.opcode_class {
            SassOpcodeClass::GlobalLoad | SassOpcodeClass::SharedLoad => {
                recorder.record_memory_access(
                    inst.address,
                    0x1000 + i as u64 * 4,
                    MemoryAccessType::Load,
                    if matches!(inst.opcode_class, SassOpcodeClass::SharedLoad) {
                        MemorySpace::Shared
                    } else {
                        MemorySpace::Global
                    },
                    4,
                    vec![0; 4],
                );
            }
            SassOpcodeClass::GlobalStore | SassOpcodeClass::SharedStore => {
                recorder.record_memory_access(
                    inst.address,
                    0x2000 + i as u64 * 4,
                    MemoryAccessType::Store,
                    if matches!(inst.opcode_class, SassOpcodeClass::SharedStore) {
                        MemorySpace::Shared
                    } else {
                        MemorySpace::Global
                    },
                    4,
                    vec![0; 4],
                );
            }
            _ => {}
        }

        // Record control flow
        match inst.opcode_class {
            SassOpcodeClass::Branch => {
                recorder.record_control_flow(
                    inst.address,
                    BranchType::Conditional,
                    inst.address + 0x10, // target
                    0xFFFF0000,          // half taken
                    0x0000FFFF,          // half not taken
                );
            }
            SassOpcodeClass::Exit => {
                recorder.record_control_flow(inst.address, BranchType::Exit, 0, 0xFFFFFFFF, 0);
            }
            _ => {}
        }
    }

    recorder.complete_kernel();

    // Save recording
    let recording = recorder.finalize();
    let file = File::create(output).map_err(|e| format!("Failed to create output: {}", e))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &recording)
        .map_err(|e| format!("Failed to write: {}", e))?;

    println!("Recording saved to {:?}", output);
    println!("  Kernels: {}", recording.kernels.len());
    println!(
        "  Total instructions: {}",
        recording
            .kernels
            .iter()
            .map(|k| k.stats.total_instructions)
            .sum::<u64>()
    );

    Ok(())
}

fn run_replay(
    input: &PathBuf,
    breakpoints: Vec<u64>,
    interactive: bool,
    verbose: bool,
) -> Result<(), String> {
    if verbose {
        eprintln!("Replaying: {:?}", input);
    }

    let mut replayer = GpuReplayer::load(input).map_err(|e| format!("Failed to load: {}", e))?;

    // Add breakpoints
    for addr in &breakpoints {
        let bp_id = replayer.add_breakpoint(BreakpointType::Address(*addr));
        if verbose {
            eprintln!("Breakpoint {} set at 0x{:x}", bp_id, addr);
        }
    }

    if interactive {
        run_interactive_replay(&mut replayer)
    } else {
        // Non-interactive: run to completion or breakpoint
        loop {
            match replayer.step() {
                ReplayResult::Stepped => {
                    if verbose {
                        if let Some(record) = replayer.current_record() {
                            println!("0x{:08x}: seq={}", record.address, record.sequence);
                        }
                    }
                }
                ReplayResult::Breakpoint(id) => {
                    if let Some(record) = replayer.current_record() {
                        println!("Breakpoint {} hit at 0x{:08x}", id, record.address);
                    }
                    break;
                }
                ReplayResult::EndOfRecording => {
                    println!("End of recording reached");
                    break;
                }
                ReplayResult::Error(e) => {
                    return Err(e);
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn run_interactive_replay(replayer: &mut GpuReplayer) -> Result<(), String> {
    println!("GPU Replay Interactive Mode");
    println!("Commands: s(tep), rs (reverse-step), c(ontinue), rc (reverse-continue)");
    println!("          b <addr> (breakpoint), r(egisters), q(uit)");
    println!();

    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        // Show current position
        if let Some(record) = replayer.current_record() {
            if let Some(inst) = replayer.current_instruction() {
                print!("0x{:08x} {:12} > ", record.address, inst.opcode);
            } else {
                print!("0x{:08x} > ", record.address);
            }
        } else {
            print!("> ");
        }
        io::stdout().flush().unwrap();

        input.clear();
        stdin.read_line(&mut input).unwrap();
        let cmd = input.trim();

        match cmd {
            "s" | "step" => match replayer.step() {
                ReplayResult::Stepped => {}
                ReplayResult::Breakpoint(id) => println!("Breakpoint {} hit", id),
                ReplayResult::EndOfRecording => println!("End of recording"),
                _ => {}
            },
            "rs" | "reverse-step" => match replayer.reverse_step() {
                ReplayResult::Stepped => {}
                ReplayResult::StartOfRecording => println!("Start of recording"),
                _ => {}
            },
            "c" | "continue" => match replayer.continue_execution() {
                ReplayResult::Breakpoint(id) => println!("Breakpoint {} hit", id),
                ReplayResult::EndOfRecording => println!("End of recording"),
                _ => {}
            },
            "rc" | "reverse-continue" => match replayer.reverse_continue() {
                ReplayResult::Breakpoint(id) => println!("Breakpoint {} hit", id),
                ReplayResult::StartOfRecording => println!("Start of recording"),
                _ => {}
            },
            "r" | "registers" => {
                let regs = replayer.get_registers();
                println!("Registers:");
                for (reg, val) in &regs.general {
                    println!("  R{}: 0x{:016x}", reg, val);
                }
                println!("  Predicates: 0x{:02x}", regs.predicates);
            }
            "q" | "quit" => {
                break;
            }
            cmd if cmd.starts_with("b ") => {
                let addr_str = &cmd[2..].trim();
                let addr = if addr_str.starts_with("0x") {
                    u64::from_str_radix(&addr_str[2..], 16)
                } else {
                    addr_str.parse()
                };
                match addr {
                    Ok(a) => {
                        let id = replayer.add_breakpoint(BreakpointType::Address(a));
                        println!("Breakpoint {} set at 0x{:x}", id, a);
                    }
                    Err(_) => println!("Invalid address"),
                }
            }
            "" => {}
            _ => println!("Unknown command: {}", cmd),
        }
    }

    Ok(())
}

fn run_analyze(
    input: &PathBuf,
    output: Option<&PathBuf>,
    _hotspots: bool,
    verbose: bool,
) -> Result<(), String> {
    if verbose {
        eprintln!("Analyzing: {:?}", input);
    }

    let analyzer = RecordingAnalyzer::load(input).map_err(|e| format!("Failed to load: {}", e))?;
    let report = analyzer.analyze();

    analyzer.print_report(&report);

    if let Some(out_path) = output {
        // Save JSON report
        let file = File::create(out_path).map_err(|e| format!("Failed to create: {}", e))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &report.summary)
            .map_err(|e| format!("Failed to write: {}", e))?;
        println!("\nReport saved to {:?}", out_path);
    }

    Ok(())
}

fn run_info(input: &PathBuf, _verbose: bool) -> Result<(), String> {
    let file = File::open(input).map_err(|e| format!("Failed to open: {}", e))?;
    let reader = BufReader::new(file);
    let recording: GpuRecording =
        serde_json::from_reader(reader).map_err(|e| format!("Failed to parse: {}", e))?;

    println!("=== GPU Recording Info ===\n");
    println!("ID: {}", recording.header.id);
    println!("Description: {}", recording.header.description);
    println!("Timestamp: {}", recording.header.timestamp);
    println!("Version: {}", recording.header.version);
    if let Some(ref path) = recording.header.cubin_path {
        println!("Source: {}", path);
    }
    println!();

    println!("Device Info:");
    println!("  Name: {}", recording.header.device_info.name);
    println!(
        "  SM Version: sm_{}",
        recording.header.device_info.sm_version
    );
    println!("  SMs: {}", recording.header.device_info.sm_count);
    println!();

    println!("Kernels: {}", recording.kernels.len());
    for (i, kernel) in recording.kernels.iter().enumerate() {
        println!(
            "  {}. {} - {} instructions",
            i + 1,
            kernel.name,
            kernel.stats.total_instructions
        );
    }
    println!();

    println!("Memory Snapshots: {}", recording.memory_snapshots.len());
    println!("Timeline Events: {}", recording.timeline.events.len());

    Ok(())
}

fn run_ptx(
    input: &PathBuf,
    output: Option<&PathBuf>,
    address: Option<u64>,
    verbose: bool,
) -> Result<(), String> {
    if verbose {
        eprintln!("Recovering PTX from: {:?}", input);
    }

    // Load CUBIN or text SASS
    let content = fs::read(input).map_err(|e| format!("Failed to read input: {}", e))?;

    if content.starts_with(&[0x7f, b'E', b'L', b'F']) {
        // CUBIN file - use PTX recovery engine
        let mut engine = PtxRecoveryEngine::new(content)
            .map_err(|e| format!("Failed to create recovery engine: {}", e))?;

        let result = engine
            .recover_all()
            .map_err(|e| format!("Failed to recover PTX: {}", e))?;

        if let Some(addr) = address {
            // Show PTX for specific address
            if let Some(line_info) = engine.get_ptx_for_sass(addr) {
                println!("=== PTX for SASS address 0x{:x} ===", addr);
                println!("File: {}", line_info.file);
                println!("Line: {}", line_info.line);
                println!("Column: {}", line_info.column);
                println!("Statement: {}", line_info.is_statement);
            } else {
                println!("No debug info for address 0x{:x}", addr);
            }
        } else {
            // Print full recovery result
            println!("=== PTX Recovery Report ===\n");
            println!("{}", result.ptx_header);

            for func in &result.functions {
                println!("\n// Function: {}", func.name);
                if let Some(ref linkage) = func.linkage_name {
                    println!("// Linkage name: {}", linkage);
                }
                println!(
                    "// SASS range: 0x{:x} - 0x{:x}\n",
                    func.start_address, func.end_address
                );

                // Print PTX lines with SASS mapping
                if !func.ptx_lines.is_empty() {
                    println!("// === Original PTX Source ===");
                    for (line_num, ptx_line) in &func.ptx_lines {
                        println!("/* {} */ {}", line_num, ptx_line.source);
                    }
                }

                // Print reconstructed PTX
                if let Some(ref reconstructed) = func.reconstructed_ptx {
                    println!("\n// === Reconstructed PTX ===");
                    println!("{}", reconstructed);
                }
            }

            // Print statistics
            println!("\n=== Recovery Statistics ===");
            println!(
                "Total SASS instructions: {}",
                result.stats.total_sass_instructions
            );
            println!("Mapped to PTX: {}", result.stats.mapped_instructions);
            println!("Unmapped: {}", result.stats.unmapped_instructions);

            if let Some(ref debug_info) = result.debug_info {
                println!("\n=== Debug Info ===");
                println!("Functions: {}", debug_info.functions.len());
                println!("Line mappings: {}", debug_info.line_mappings.len());
                println!("Source files: {:?}", result.source_files);
            }
        }

        // Save output if requested
        if let Some(out_path) = output {
            let mut out_content = result.ptx_header.clone();
            for func in &result.functions {
                if let Some(ref reconstructed) = func.reconstructed_ptx {
                    out_content.push_str("\n");
                    out_content.push_str(reconstructed);
                }
            }
            fs::write(out_path, out_content)
                .map_err(|e| format!("Failed to write output: {}", e))?;
            println!("\nRecovered PTX written to {:?}", out_path);
        }
    } else {
        // Text SASS - use reconstructor
        let text = String::from_utf8_lossy(&content);
        let instructions = TextDisassemblyParser::parse_cuobjdump_output(&text);

        if instructions.is_empty() {
            return Err("No instructions found in input".to_string());
        }

        let kernel_name = instructions
            .first()
            .and_then(|i| i.function_name.clone())
            .unwrap_or_else(|| "kernel".to_string());

        println!("=== PTX Reconstruction for {} ===\n", kernel_name);

        // Use SM 75 as default
        let mut reconstructor = PtxReconstructor::new(75);

        // Print header
        println!(".version 7.0");
        println!(".target sm_75");
        println!(".address_size 64\n");

        println!(".visible .entry {} ()", kernel_name);
        println!("{{");
        println!("    .reg .b32 %r<256>;");
        println!("    .reg .pred %p<8>;\n");

        for inst in &instructions {
            let ptx = reconstructor.reconstruct_instruction(inst);
            println!(
                "    // 0x{:04x}: {} {:?}",
                inst.address,
                inst.opcode,
                inst.src_operands.iter().take(3).collect::<Vec<_>>()
            );
            println!("    {}", ptx);
        }

        println!("}}");

        // Save output if requested
        if let Some(out_path) = output {
            let mut out_content = String::new();
            out_content.push_str(".version 7.0\n");
            out_content.push_str(".target sm_75\n");
            out_content.push_str(".address_size 64\n\n");
            out_content.push_str(&format!(".visible .entry {} ()\n{{\n", kernel_name));
            out_content.push_str("    .reg .b32 %r<256>;\n");
            out_content.push_str("    .reg .pred %p<8>;\n\n");

            let mut reconstructor = PtxReconstructor::new(75);
            for inst in &instructions {
                let ptx = reconstructor.reconstruct_instruction(inst);
                out_content.push_str(&format!("    // 0x{:04x}: {}\n", inst.address, inst.opcode));
                out_content.push_str(&format!("    {}\n", ptx));
            }
            out_content.push_str("}\n");

            fs::write(out_path, out_content)
                .map_err(|e| format!("Failed to write output: {}", e))?;
            println!("\nRecovered PTX written to {:?}", out_path);
        }
    }

    Ok(())
}
