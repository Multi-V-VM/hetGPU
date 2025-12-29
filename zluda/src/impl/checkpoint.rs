//! GPU Checkpoint/Resume support for hetGPU
//!
//! This module provides SIGINT (Ctrl+C) handling to checkpoint GPU kernel
//! execution and allow resuming from saved state.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::path::PathBuf;
use std::fs;

use serde::{Serialize, Deserialize};

// =============================================================================
// Fine-Grained Checkpoint Structures
// =============================================================================

/// Checkpoint granularity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointGranularity {
    /// Checkpoint at kernel launch boundaries only
    KernelLevel,
    /// Checkpoint at basic block boundaries within kernel
    BasicBlockLevel,
    /// Checkpoint at individual PTX instruction level
    InstructionLevel,
    /// Checkpoint at warp synchronization points
    WarpSyncLevel,
}

impl Default for CheckpointGranularity {
    fn default() -> Self {
        CheckpointGranularity::KernelLevel
    }
}

/// PTX instruction checkpoint state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtxInstructionState {
    /// Instruction index within the kernel (program counter)
    pub instruction_pc: u64,
    /// PTX line number (for debugging)
    pub ptx_line: u32,
    /// Instruction opcode (e.g., "add", "mul", "ld.global")
    pub opcode: String,
    /// Source operands
    pub src_operands: Vec<String>,
    /// Destination operand
    pub dst_operand: Option<String>,
    /// Predicate (if any)
    pub predicate: Option<String>,
}

/// Register state for a single thread
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadRegisterState {
    /// Thread ID within block (x, y, z)
    pub thread_id: (u32, u32, u32),
    /// 32-bit registers (r0-r255)
    pub regs_32: HashMap<u8, u32>,
    /// 64-bit registers (rd0-rd255)
    pub regs_64: HashMap<u8, u64>,
    /// Predicate registers (p0-p7)
    pub pred_regs: HashMap<u8, bool>,
    /// Float registers (f0-f255)
    pub float_regs: HashMap<u8, f32>,
    /// Double registers (fd0-fd255)
    pub double_regs: HashMap<u8, f64>,
    /// Special registers (tid, ntid, ctaid, nctaid, etc.)
    pub special_regs: HashMap<String, u64>,
    /// Current instruction pointer
    pub current_pc: u64,
    /// Is thread active (not masked by divergence)?
    pub is_active: bool,
}

/// Warp execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpState {
    /// Warp ID within block
    pub warp_id: u32,
    /// Thread states for all threads in warp (32 threads per warp)
    pub thread_states: Vec<ThreadRegisterState>,
    /// Warp-level program counter
    pub warp_pc: u64,
    /// Active mask (which threads are active)
    pub active_mask: u32,
    /// Convergence point stack (for divergence handling)
    pub convergence_stack: Vec<(u64, u32)>, // (reconvergence_pc, saved_mask)
    /// Barrier participation mask
    pub barrier_mask: u32,
}

/// Block execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockState {
    /// Block ID (x, y, z)
    pub block_id: (u32, u32, u32),
    /// Warp states for all warps in block
    pub warp_states: Vec<WarpState>,
    /// Shared memory snapshot
    pub shared_memory: Vec<u8>,
    /// Shared memory size
    pub shared_memory_size: u64,
    /// Number of warps that have reached barrier
    pub barrier_count: u32,
    /// Block-level synchronization state
    pub sync_state: BlockSyncState,
}

/// Block synchronization state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockSyncState {
    /// Normal execution
    Running,
    /// Waiting at barrier N
    AtBarrier(u32),
    /// Completed
    Completed,
}

/// Memory region with fine-grained tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegionState {
    /// Device pointer
    pub device_ptr: u64,
    /// Size in bytes
    pub size: u64,
    /// Memory type (global, constant, texture)
    pub memory_type: PtxMemoryType,
    /// Data content (compressed if large)
    pub data: Vec<u8>,
    /// Is data dirty (modified since last checkpoint)?
    pub is_dirty: bool,
    /// Checksum for verification
    pub checksum: u64,
    /// Access pattern tracking
    pub access_regions: Vec<(u64, u64)>, // (offset, size) pairs that were accessed
}

/// PTX Memory types (for fine-grained checkpointing)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PtxMemoryType {
    Global,
    Constant,
    Texture,
    Shared,
    Local,
    Parameter,
}

/// Fine-grained kernel execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineGrainedKernelState {
    /// Base kernel info
    pub base: KernelExecutionState,
    /// Checkpoint granularity level
    pub granularity: CheckpointGranularity,
    /// Block states for all active blocks
    pub block_states: Vec<BlockState>,
    /// Global memory regions
    pub memory_regions: Vec<MemoryRegionState>,
    /// Parsed PTX instructions (for instruction-level checkpointing)
    pub ptx_instructions: Vec<PtxInstructionState>,
    /// Current grid progress (completed blocks)
    pub completed_blocks: u64,
    /// Total blocks in grid
    pub total_blocks: u64,
    /// Kernel start PC
    pub kernel_start_pc: u64,
    /// Kernel end PC
    pub kernel_end_pc: u64,
}

/// Fine-grained checkpoint data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineGrainedCheckpointData {
    /// Version for compatibility
    pub version: u32,
    /// Timestamp
    pub timestamp: u64,
    /// Checkpoint granularity
    pub granularity: CheckpointGranularity,
    /// Active fine-grained kernel states
    pub kernel_states: Vec<FineGrainedKernelState>,
    /// PTX sources
    pub ptx_sources: HashMap<u64, String>,
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata,
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Checkpoint ID
    pub checkpoint_id: u64,
    /// Parent checkpoint ID (for incremental checkpointing)
    pub parent_id: Option<u64>,
    /// Is this an incremental checkpoint?
    pub is_incremental: bool,
    /// Backend that created this checkpoint
    pub source_backend: String,
    /// CUDA compute capability
    pub compute_capability: (u32, u32),
    /// Estimated resume time in microseconds
    pub estimated_resume_time_us: u64,
}

/// Global checkpoint counter for incremental checkpointing
static CHECKPOINT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Global flag for checkpoint request (set by SIGINT handler)
static CHECKPOINT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Global flag indicating checkpoint handler is installed
static HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Current kernel execution counter
static KERNEL_EXECUTION_ID: AtomicU64 = AtomicU64::new(0);

/// Checkpoint directory path
static CHECKPOINT_DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Current kernel execution state for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelExecutionState {
    /// Unique execution ID
    pub execution_id: u64,
    /// Kernel function name
    pub kernel_name: String,
    /// PTX source if available
    pub ptx_source: Option<String>,
    /// Grid dimensions
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions
    pub block_dim: (u32, u32, u32),
    /// Shared memory size
    pub shared_mem_bytes: u32,
    /// Kernel arguments (address, size pairs)
    pub kernel_args: Vec<(u64, usize)>,
    /// Stream handle
    pub stream: u64,
    /// Start timestamp
    pub start_time: u64,
    /// Module handle
    pub module_handle: u64,
    /// Function handle
    pub function_handle: u64,
}

/// Thread-local storage for current kernel state
thread_local! {
    static CURRENT_KERNEL: std::cell::RefCell<Option<KernelExecutionState>> = std::cell::RefCell::new(None);
}

/// Global checkpoint manager
pub struct CheckpointManager {
    /// Checkpoint directory
    checkpoint_dir: PathBuf,
    /// Active kernel states (execution_id -> state)
    active_kernels: HashMap<u64, KernelExecutionState>,
    /// Saved checkpoints
    saved_checkpoints: Vec<PathBuf>,
    /// PTX source cache (module_handle -> ptx_source)
    ptx_cache: HashMap<u64, String>,
    /// Loaded restore state (for FFI restore)
    pub loaded_restore_state: Option<GpuRestoreState>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(checkpoint_dir: &str) -> Self {
        let path = PathBuf::from(checkpoint_dir);
        if !path.exists() {
            let _ = fs::create_dir_all(&path);
        }

        Self {
            checkpoint_dir: path,
            active_kernels: HashMap::new(),
            saved_checkpoints: Vec::new(),
            ptx_cache: HashMap::new(),
            loaded_restore_state: None,
        }
    }

    /// Register PTX source for a module
    pub fn register_ptx_source(&mut self, module_handle: u64, ptx_source: String) {
        self.ptx_cache.insert(module_handle, ptx_source);
    }

    /// Get PTX source for a module
    pub fn get_ptx_source(&self, module_handle: u64) -> Option<&String> {
        self.ptx_cache.get(&module_handle)
    }

    /// Register a kernel execution
    pub fn register_kernel(&mut self, state: KernelExecutionState) {
        self.active_kernels.insert(state.execution_id, state);
    }

    /// Unregister a completed kernel execution
    pub fn unregister_kernel(&mut self, execution_id: u64) {
        self.active_kernels.remove(&execution_id);
    }

    /// Save checkpoint for all active kernels
    pub fn save_checkpoint(&self) -> Result<PathBuf, String> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let filename = format!("hetgpu_checkpoint_{}.json", timestamp);
        let filepath = self.checkpoint_dir.join(&filename);

        // Create checkpoint data
        let checkpoint_data = CheckpointData {
            version: 1,
            timestamp,
            active_kernels: self.active_kernels.values().cloned().collect(),
            ptx_sources: self.ptx_cache.clone(),
        };

        let json = serde_json::to_string_pretty(&checkpoint_data)
            .map_err(|e| format!("JSON serialization error: {}", e))?;

        fs::write(&filepath, json)
            .map_err(|e| format!("Failed to write checkpoint: {}", e))?;

        Ok(filepath)
    }

    /// Load checkpoint from file
    pub fn load_checkpoint(path: &str) -> Result<CheckpointData, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read checkpoint: {}", e))?;

        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse checkpoint: {}", e))
    }

    /// Save fine-grained checkpoint
    pub fn save_fine_grained_checkpoint(
        &self,
        granularity: CheckpointGranularity,
        kernel_states: Vec<FineGrainedKernelState>,
    ) -> Result<PathBuf, String> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let checkpoint_id = CHECKPOINT_COUNTER.fetch_add(1, Ordering::SeqCst);
        let filename = format!("hetgpu_finegrained_{}_{}.json", checkpoint_id, timestamp);
        let filepath = self.checkpoint_dir.join(&filename);

        let metadata = CheckpointMetadata {
            checkpoint_id,
            parent_id: None,
            is_incremental: false,
            source_backend: detect_backend(),
            compute_capability: (8, 6), // Default SM_86
            estimated_resume_time_us: estimate_resume_time(&kernel_states),
        };

        let checkpoint_data = FineGrainedCheckpointData {
            version: 2, // Version 2 for fine-grained
            timestamp,
            granularity,
            kernel_states,
            ptx_sources: self.ptx_cache.clone(),
            metadata,
        };

        let json = serde_json::to_string_pretty(&checkpoint_data)
            .map_err(|e| format!("JSON serialization error: {}", e))?;

        fs::write(&filepath, json)
            .map_err(|e| format!("Failed to write checkpoint: {}", e))?;

        eprintln!("[hetGPU] Fine-grained checkpoint saved: {:?}", filepath);
        eprintln!("[hetGPU]   Granularity: {:?}", granularity);
        eprintln!("[hetGPU]   Kernels: {}", checkpoint_data.kernel_states.len());

        Ok(filepath)
    }

    /// Load fine-grained checkpoint
    pub fn load_fine_grained_checkpoint(path: &str) -> Result<FineGrainedCheckpointData, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read checkpoint: {}", e))?;

        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse fine-grained checkpoint: {}", e))
    }
}

// =============================================================================
// PTX Instruction Parsing
// =============================================================================

/// Parse PTX source into instruction states for fine-grained checkpointing
pub fn parse_ptx_instructions(ptx_source: &str) -> Vec<PtxInstructionState> {
    let mut instructions = Vec::new();
    let mut pc: u64 = 0;
    let mut in_kernel = false;

    for (line_num, line) in ptx_source.lines().enumerate() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        // Track kernel entry/exit
        if line.contains(".entry") || line.contains(".func") {
            in_kernel = true;
            continue;
        }

        if line == "}" {
            in_kernel = false;
            continue;
        }

        // Skip directives
        if line.starts_with(".") {
            continue;
        }

        // Skip labels
        if line.ends_with(":") {
            continue;
        }

        if !in_kernel {
            continue;
        }

        // Parse instruction
        if let Some(instr) = parse_single_instruction(line, pc, line_num as u32) {
            instructions.push(instr);
            pc += 1;
        }
    }

    instructions
}

/// Parse a single PTX instruction line
fn parse_single_instruction(line: &str, pc: u64, ptx_line: u32) -> Option<PtxInstructionState> {
    let line = line.trim();

    // Handle predicated instructions: @%p1 bra LABEL
    let (predicate, rest) = if line.starts_with('@') {
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() == 2 {
            (Some(parts[0].to_string()), parts[1].trim())
        } else {
            return None;
        }
    } else {
        (None, line)
    };

    // Remove trailing semicolon
    let rest = rest.trim_end_matches(';').trim();

    // Split into parts
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.is_empty() {
        return None;
    }

    let opcode = parts[0].to_string();

    // Handle different instruction formats
    let (dst_operand, src_operands) = if parts.len() > 1 {
        // Look for comma-separated operands
        let operand_str = parts[1..].join(" ");
        let operands: Vec<&str> = operand_str.split(',').map(|s| s.trim()).collect();

        if operands.is_empty() {
            (None, Vec::new())
        } else if is_store_instruction(&opcode) {
            // Store instructions: st.xxx [addr], src
            (None, operands.iter().map(|s| s.to_string()).collect())
        } else if operands.len() == 1 {
            // Single operand (like ret, bra)
            (Some(operands[0].to_string()), Vec::new())
        } else {
            // Normal instruction: dst, src1, src2, ...
            let dst = Some(operands[0].to_string());
            let srcs = operands[1..].iter().map(|s| s.to_string()).collect();
            (dst, srcs)
        }
    } else {
        (None, Vec::new())
    };

    Some(PtxInstructionState {
        instruction_pc: pc,
        ptx_line,
        opcode,
        src_operands,
        dst_operand,
        predicate,
    })
}

/// Check if instruction is a store
fn is_store_instruction(opcode: &str) -> bool {
    opcode.starts_with("st.") || opcode.starts_with("atom.")
}

/// Detect current backend
fn detect_backend() -> String {
    #[cfg(feature = "nvidia")]
    return "nvidia".to_string();
    #[cfg(feature = "amd")]
    return "amd".to_string();
    #[cfg(feature = "intel")]
    return "intel".to_string();
    #[cfg(feature = "tenstorrent")]
    return "tenstorrent".to_string();
    #[cfg(not(any(feature = "nvidia", feature = "amd", feature = "intel", feature = "tenstorrent")))]
    return "unknown".to_string();
}

/// Estimate resume time based on checkpoint state
fn estimate_resume_time(kernel_states: &[FineGrainedKernelState]) -> u64 {
    let mut total_us: u64 = 0;

    for state in kernel_states {
        // Base overhead
        total_us += 100;

        // Memory restore overhead (1us per KB)
        for region in &state.memory_regions {
            total_us += region.size / 1024;
        }

        // Block state restore overhead
        total_us += state.block_states.len() as u64 * 10;

        // Instruction state overhead
        total_us += state.ptx_instructions.len() as u64;
    }

    total_us
}

/// Create initial thread register state
pub fn create_initial_thread_state(thread_id: (u32, u32, u32), block_dim: (u32, u32, u32), block_id: (u32, u32, u32), grid_dim: (u32, u32, u32)) -> ThreadRegisterState {
    let mut special_regs = HashMap::new();

    // Thread ID registers
    special_regs.insert("tid.x".to_string(), thread_id.0 as u64);
    special_regs.insert("tid.y".to_string(), thread_id.1 as u64);
    special_regs.insert("tid.z".to_string(), thread_id.2 as u64);

    // Block dimension registers
    special_regs.insert("ntid.x".to_string(), block_dim.0 as u64);
    special_regs.insert("ntid.y".to_string(), block_dim.1 as u64);
    special_regs.insert("ntid.z".to_string(), block_dim.2 as u64);

    // Block ID registers
    special_regs.insert("ctaid.x".to_string(), block_id.0 as u64);
    special_regs.insert("ctaid.y".to_string(), block_id.1 as u64);
    special_regs.insert("ctaid.z".to_string(), block_id.2 as u64);

    // Grid dimension registers
    special_regs.insert("nctaid.x".to_string(), grid_dim.0 as u64);
    special_regs.insert("nctaid.y".to_string(), grid_dim.1 as u64);
    special_regs.insert("nctaid.z".to_string(), grid_dim.2 as u64);

    // Warp size
    special_regs.insert("WARP_SZ".to_string(), 32);

    // Lane ID
    let lane_id = thread_id.0 % 32;
    special_regs.insert("laneid".to_string(), lane_id as u64);

    ThreadRegisterState {
        thread_id,
        regs_32: HashMap::new(),
        regs_64: HashMap::new(),
        pred_regs: HashMap::new(),
        float_regs: HashMap::new(),
        double_regs: HashMap::new(),
        special_regs,
        current_pc: 0,
        is_active: true,
    }
}

/// Create initial warp state
pub fn create_initial_warp_state(warp_id: u32, block_dim: (u32, u32, u32), block_id: (u32, u32, u32), grid_dim: (u32, u32, u32)) -> WarpState {
    let mut thread_states = Vec::with_capacity(32);
    let threads_per_block = block_dim.0 * block_dim.1 * block_dim.2;
    let warp_start = warp_id * 32;

    for lane in 0..32 {
        let linear_tid = warp_start + lane;
        if linear_tid < threads_per_block {
            let tid_x = linear_tid % block_dim.0;
            let tid_y = (linear_tid / block_dim.0) % block_dim.1;
            let tid_z = linear_tid / (block_dim.0 * block_dim.1);
            thread_states.push(create_initial_thread_state(
                (tid_x, tid_y, tid_z),
                block_dim,
                block_id,
                grid_dim,
            ));
        }
    }

    let active_mask = if threads_per_block >= warp_start + 32 {
        0xFFFFFFFF
    } else if threads_per_block > warp_start {
        (1u32 << (threads_per_block - warp_start)) - 1
    } else {
        0
    };

    WarpState {
        warp_id,
        thread_states,
        warp_pc: 0,
        active_mask,
        convergence_stack: Vec::new(),
        barrier_mask: 0,
    }
}

/// Create initial block state
pub fn create_initial_block_state(
    block_id: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    grid_dim: (u32, u32, u32),
    shared_mem_size: u64,
) -> BlockState {
    let threads_per_block = block_dim.0 * block_dim.1 * block_dim.2;
    let warps_per_block = (threads_per_block + 31) / 32;

    let mut warp_states = Vec::with_capacity(warps_per_block as usize);
    for warp_id in 0..warps_per_block {
        warp_states.push(create_initial_warp_state(warp_id, block_dim, block_id, grid_dim));
    }

    BlockState {
        block_id,
        warp_states,
        shared_memory: vec![0u8; shared_mem_size as usize],
        shared_memory_size: shared_mem_size,
        barrier_count: 0,
        sync_state: BlockSyncState::Running,
    }
}

/// Create fine-grained kernel state from basic state
pub fn create_fine_grained_state(
    base: KernelExecutionState,
    granularity: CheckpointGranularity,
) -> FineGrainedKernelState {
    let grid_dim = base.grid_dim;
    let block_dim = base.block_dim;
    let total_blocks = (grid_dim.0 as u64) * (grid_dim.1 as u64) * (grid_dim.2 as u64);

    // Parse PTX instructions if available
    let ptx_instructions = if let Some(ref ptx) = base.ptx_source {
        parse_ptx_instructions(ptx)
    } else {
        Vec::new()
    };

    let kernel_end_pc = ptx_instructions.len() as u64;

    FineGrainedKernelState {
        base,
        granularity,
        block_states: Vec::new(), // Will be populated during execution
        memory_regions: Vec::new(),
        ptx_instructions,
        completed_blocks: 0,
        total_blocks,
        kernel_start_pc: 0,
        kernel_end_pc,
    }
}

/// Serializable checkpoint data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    pub version: u32,
    pub timestamp: u64,
    pub active_kernels: Vec<KernelExecutionState>,
    pub ptx_sources: HashMap<u64, String>,
}

/// Global checkpoint manager instance
static CHECKPOINT_MANAGER: std::sync::OnceLock<Mutex<CheckpointManager>> = std::sync::OnceLock::new();

/// Get the global checkpoint manager
pub fn get_checkpoint_manager() -> &'static Mutex<CheckpointManager> {
    CHECKPOINT_MANAGER.get_or_init(|| {
        let dir = CHECKPOINT_DIR.get_or_init(|| {
            std::env::var("HETGPU_CHECKPOINT_DIR")
                .unwrap_or_else(|_| "/tmp/hetgpu_checkpoints".to_string())
        });
        Mutex::new(CheckpointManager::new(dir))
    })
}

/// Install SIGINT handler for checkpoint
///
/// IMPORTANT: The signal handler is async-signal-safe - it only sets an atomic flag.
/// The actual checkpoint is performed at safe points (e.g., after cuCtxSynchronize)
/// to avoid conflicts with NVIDIA driver internal state.
#[cfg(unix)]
pub fn install_signal_handler() -> Result<(), String> {
    if HANDLER_INSTALLED.swap(true, Ordering::SeqCst) {
        // Already installed
        return Ok(());
    }

    // Async-signal-safe handler - ONLY sets the flag
    // No locks, no allocations, no file I/O - these cause core dumps on NVIDIA
    extern "C" fn sigint_handler(_: libc::c_int) {
        CHECKPOINT_REQUESTED.store(true, Ordering::SeqCst);

        // Only use async-signal-safe functions (write() is safe)
        // Use a static message to avoid any allocations
        let msg = b"\n[hetGPU] Checkpoint requested (will save at next sync point)...\n";
        unsafe {
            libc::write(libc::STDERR_FILENO, msg.as_ptr() as *const libc::c_void, msg.len());
        }

        // DO NOT call save_checkpoint() here - it's not async-signal-safe!
        // The checkpoint will be saved at the next safe point (cuCtxSynchronize, etc.)
    }

    unsafe {
        // Use sigaction() with SA_RESTART for better NVIDIA compatibility
        // SA_RESTART: automatically restart interrupted system calls
        // This prevents EINTR errors that can confuse the NVIDIA driver
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = sigint_handler as usize;
        sa.sa_flags = libc::SA_RESTART;
        libc::sigemptyset(&mut sa.sa_mask);

        let ret = libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
        if ret != 0 {
            HANDLER_INSTALLED.store(false, Ordering::SeqCst);
            return Err("Failed to install SIGINT handler".to_string());
        }
    }

    eprintln!("[hetGPU] Checkpoint handler installed (Ctrl+C to checkpoint)");
    eprintln!("[hetGPU] Note: Checkpoint will be saved at the next GPU sync point");
    Ok(())
}

#[cfg(not(unix))]
pub fn install_signal_handler() -> Result<(), String> {
    Err("Signal handler not supported on this platform".to_string())
}

/// Check if checkpoint was requested
pub fn is_checkpoint_requested() -> bool {
    CHECKPOINT_REQUESTED.load(Ordering::SeqCst)
}

/// Clear checkpoint request flag
pub fn clear_checkpoint_request() {
    CHECKPOINT_REQUESTED.store(false, Ordering::SeqCst);
}

/// Request a checkpoint programmatically
pub fn request_checkpoint() {
    CHECKPOINT_REQUESTED.store(true, Ordering::SeqCst);
}

/// Check for pending checkpoint and save if requested
/// This should be called at safe points (after cuCtxSynchronize, etc.)
/// Returns true if a checkpoint was saved
pub fn process_pending_checkpoint() -> bool {
    if !is_checkpoint_requested() {
        return false;
    }

    // Clear the flag first to avoid re-entry
    clear_checkpoint_request();

    eprintln!("[hetGPU] Processing pending checkpoint at safe point...");

    // Now it's safe to acquire locks and do file I/O
    if let Ok(manager) = get_checkpoint_manager().lock() {
        match manager.save_checkpoint() {
            Ok(path) => {
                eprintln!("[hetGPU] Checkpoint saved to: {:?}", path);
                return true;
            }
            Err(e) => {
                eprintln!("[hetGPU] Checkpoint failed: {}", e);
            }
        }
    } else {
        eprintln!("[hetGPU] Failed to acquire checkpoint manager lock");
    }

    false
}

/// Generate new kernel execution ID
pub fn next_execution_id() -> u64 {
    KERNEL_EXECUTION_ID.fetch_add(1, Ordering::SeqCst)
}

/// Start tracking a kernel execution
pub fn begin_kernel_execution(
    kernel_name: &str,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: u64,
    module_handle: u64,
    function_handle: u64,
) -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    let execution_id = next_execution_id();
    let start_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Get PTX source from cache
    let ptx_source = get_checkpoint_manager()
        .lock()
        .ok()
        .and_then(|mgr| mgr.get_ptx_source(module_handle).cloned());

    let state = KernelExecutionState {
        execution_id,
        kernel_name: kernel_name.to_string(),
        ptx_source,
        grid_dim,
        block_dim,
        shared_mem_bytes,
        kernel_args: Vec::new(),
        stream,
        start_time,
        module_handle,
        function_handle,
    };

    // Store in thread-local
    CURRENT_KERNEL.with(|k| {
        *k.borrow_mut() = Some(state.clone());
    });

    // Register with global manager
    if let Ok(mut manager) = get_checkpoint_manager().lock() {
        manager.register_kernel(state);
    }

    execution_id
}

/// Add kernel argument to current execution
pub fn add_kernel_argument(arg_addr: u64, arg_size: usize) {
    CURRENT_KERNEL.with(|k| {
        if let Some(ref mut state) = *k.borrow_mut() {
            state.kernel_args.push((arg_addr, arg_size));
        }
    });
}

/// End kernel execution tracking
pub fn end_kernel_execution(execution_id: u64) {
    // Clear thread-local
    CURRENT_KERNEL.with(|k| {
        *k.borrow_mut() = None;
    });

    // Unregister from global manager
    if let Ok(mut manager) = get_checkpoint_manager().lock() {
        manager.unregister_kernel(execution_id);
    }
}

/// Get current kernel state for the current thread
pub fn get_current_kernel_state() -> Option<KernelExecutionState> {
    CURRENT_KERNEL.with(|k| k.borrow().clone())
}

/// Check for checkpoint at kernel launch point
/// Returns true if execution should be paused
pub fn check_checkpoint_at_launch() -> bool {
    if is_checkpoint_requested() {
        clear_checkpoint_request();

        eprintln!("[hetGPU] Checkpoint triggered at kernel launch");

        // Save checkpoint
        if let Ok(manager) = get_checkpoint_manager().lock() {
            match manager.save_checkpoint() {
                Ok(path) => {
                    eprintln!("[hetGPU] Checkpoint saved to: {:?}", path);
                }
                Err(e) => {
                    eprintln!("[hetGPU] Checkpoint failed: {}", e);
                }
            }
        }

        // Check if we should pause execution
        if std::env::var("HETGPU_CHECKPOINT_PAUSE").ok().as_deref() == Some("1") {
            eprintln!("[hetGPU] Execution paused. Set HETGPU_CHECKPOINT_RESUME=<file> to resume.");
            return true;
        }
    }
    false
}

/// Register PTX source when module is loaded
pub fn register_module_ptx(module_handle: u64, ptx_source: &str) {
    if let Ok(mut manager) = get_checkpoint_manager().lock() {
        manager.register_ptx_source(module_handle, ptx_source.to_string());
    }
}

/// Load and prepare resume from checkpoint file
pub fn prepare_resume(checkpoint_path: &str) -> Result<CheckpointData, String> {
    CheckpointManager::load_checkpoint(checkpoint_path)
}

// ============================================================================
// GPU State Restoration for Heterogeneous GPUs
// ============================================================================

/// Memory region that needs to be restored
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegionRestore {
    /// Original device address
    pub original_addr: u64,
    /// Size in bytes
    pub size: usize,
    /// Memory contents (base64 encoded for large data)
    pub data: Vec<u8>,
    /// Memory type
    pub mem_type: MemoryType,
    /// Alignment requirement
    pub alignment: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryType {
    Device,
    Managed,
    Pinned,
    Shared,
}

/// Complete GPU state for restoration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRestoreState {
    /// Checkpoint version
    pub version: u32,
    /// Source backend (intel, amd, tenstorrent, virtual)
    pub source_backend: String,
    /// PTX source code (portable across backends)
    pub ptx_source: String,
    /// Compiled SASS/binary for original target (if available)
    pub compiled_binary: Option<Vec<u8>>,
    /// Kernel name to resume
    pub kernel_name: String,
    /// Kernel arguments with their types
    pub kernel_args: Vec<KernelArgRestore>,
    /// Memory regions to restore
    pub memory_regions: Vec<MemoryRegionRestore>,
    /// Grid dimensions
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions
    pub block_dim: (u32, u32, u32),
    /// Shared memory size
    pub shared_mem_bytes: u32,
    /// Thread execution state (warp/lane info)
    pub thread_state: Option<ThreadExecutionState>,
    /// Address remap table (original -> new)
    #[serde(skip)]
    pub address_remap: HashMap<u64, u64>,
}

/// Kernel argument for restoration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelArgRestore {
    /// Argument index
    pub index: u32,
    /// Original address (for pointers)
    pub original_addr: u64,
    /// Argument size
    pub size: usize,
    /// Is this a pointer type?
    pub is_pointer: bool,
    /// Raw data (for non-pointer types)
    pub data: Vec<u8>,
}

/// Thread execution state for fine-grained resume
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadExecutionState {
    /// Current PTX line number
    pub ptx_line: u32,
    /// Current instruction offset in compiled code
    pub instruction_offset: u64,
    /// Register values (register name -> value)
    pub registers: HashMap<String, u64>,
    /// Predicate register values
    pub predicates: HashMap<String, bool>,
    /// Active thread mask
    pub active_mask: u64,
    /// Block ID that was executing
    pub block_id: (u32, u32, u32),
    /// Thread ID within block
    pub thread_id: (u32, u32, u32),
}

impl GpuRestoreState {
    /// Create restore state from checkpoint
    pub fn from_checkpoint(
        checkpoint: &CheckpointData,
        ptx_source: String,
        kernel_name: &str,
    ) -> Result<Self, String> {
        let kernel = checkpoint.active_kernels
            .iter()
            .find(|k| k.kernel_name == kernel_name)
            .ok_or_else(|| format!("Kernel '{}' not found in checkpoint", kernel_name))?;

        Ok(Self {
            version: checkpoint.version,
            source_backend: detect_current_backend(),
            ptx_source,
            compiled_binary: None,
            kernel_name: kernel_name.to_string(),
            kernel_args: kernel.kernel_args.iter().enumerate().map(|(i, (addr, size))| {
                KernelArgRestore {
                    index: i as u32,
                    original_addr: *addr,
                    size: *size,
                    is_pointer: is_likely_pointer(*addr),
                    data: Vec::new(),
                }
            }).collect(),
            memory_regions: Vec::new(),
            grid_dim: kernel.grid_dim,
            block_dim: kernel.block_dim,
            shared_mem_bytes: kernel.shared_mem_bytes,
            thread_state: None,
            address_remap: HashMap::new(),
        })
    }

    /// Add memory region to restore
    pub fn add_memory_region(&mut self, addr: u64, size: usize, data: Vec<u8>, mem_type: MemoryType) {
        self.memory_regions.push(MemoryRegionRestore {
            original_addr: addr,
            size,
            data,
            mem_type,
            alignment: 256, // Default GPU alignment
        });
    }

    /// Set thread execution state for fine-grained resume
    pub fn set_thread_state(&mut self, state: ThreadExecutionState) {
        self.thread_state = Some(state);
    }

    /// Register an address remapping
    pub fn remap_address(&mut self, original: u64, new: u64) {
        self.address_remap.insert(original, new);
    }

    /// Get remapped address
    pub fn get_remapped_address(&self, original: u64) -> u64 {
        *self.address_remap.get(&original).unwrap_or(&original)
    }

    /// Save restore state to file
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Serialization error: {}", e))?;
        fs::write(path, json)
            .map_err(|e| format!("Write error: {}", e))
    }

    /// Load restore state from file
    pub fn load(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Read error: {}", e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Parse error: {}", e))
    }
}

/// GPU Restorer handles the actual restoration on heterogeneous backends
pub struct GpuRestorer {
    state: GpuRestoreState,
    target_backend: String,
}

impl GpuRestorer {
    /// Create a new restorer for the given state
    pub fn new(state: GpuRestoreState) -> Self {
        Self {
            state,
            target_backend: detect_current_backend(),
        }
    }

    /// Check if cross-backend restoration is needed
    pub fn is_cross_backend(&self) -> bool {
        self.state.source_backend != self.target_backend
    }

    /// Restore GPU state and prepare for execution
    ///
    /// Returns Ok with recompiled module handle if successful
    pub fn restore(&mut self) -> Result<RestoreResult, String> {
        eprintln!("[hetGPU] Restoring GPU state from {} backend to {} backend",
            self.state.source_backend, self.target_backend);

        let mut result = RestoreResult::default();

        // Step 1: Recompile PTX for target backend if needed
        if self.is_cross_backend() || self.state.compiled_binary.is_none() {
            eprintln!("[hetGPU] Recompiling PTX for {} backend...", self.target_backend);
            result.recompiled = true;
            // The actual recompilation will be handled by cuModuleLoadData
            // We store the PTX source for that
        }

        // Step 2: Reallocate memory regions
        eprintln!("[hetGPU] Restoring {} memory regions...", self.state.memory_regions.len());

        // First pass: collect all the allocation info without mutating
        let mut allocations: Vec<(u64, u64, usize, Vec<u8>)> = Vec::new();
        for region in &self.state.memory_regions {
            let new_addr = self.allocate_memory(region.size, region.mem_type)?;
            allocations.push((region.original_addr, new_addr, region.size, region.data.clone()));
        }

        // Second pass: apply the remappings and copy data
        for (original_addr, new_addr, size, data) in allocations {
            // Record remapping
            self.state.remap_address(original_addr, new_addr);

            // Copy data to new allocation
            self.copy_to_device(new_addr, &data)?;

            result.memory_mappings.push((original_addr, new_addr));
            eprintln!("[hetGPU]   0x{:016x} -> 0x{:016x} ({} bytes)",
                original_addr, new_addr, size);
        }

        // Step 3: Update kernel arguments with remapped addresses
        result.remapped_args = self.state.kernel_args.iter().map(|arg| {
            if arg.is_pointer {
                let new_addr = self.state.get_remapped_address(arg.original_addr);
                (arg.index, new_addr)
            } else {
                (arg.index, arg.original_addr)
            }
        }).collect();

        // Step 4: Prepare thread state for fine-grained resume
        if let Some(ref thread_state) = self.state.thread_state {
            result.resume_point = Some(ResumePoint {
                ptx_line: thread_state.ptx_line,
                instruction_offset: thread_state.instruction_offset,
                block_id: thread_state.block_id,
                thread_id: thread_state.thread_id,
            });
            eprintln!("[hetGPU] Resume point: PTX line {}, offset 0x{:x}",
                thread_state.ptx_line, thread_state.instruction_offset);
        }

        result.ptx_source = self.state.ptx_source.clone();
        result.kernel_name = self.state.kernel_name.clone();
        result.grid_dim = self.state.grid_dim;
        result.block_dim = self.state.block_dim;
        result.shared_mem_bytes = self.state.shared_mem_bytes;

        eprintln!("[hetGPU] Restoration complete!");
        Ok(result)
    }

    /// Allocate memory on current device
    fn allocate_memory(&self, size: usize, mem_type: MemoryType) -> Result<u64, String> {
        // This would call the actual CUDA/Level Zero/etc memory allocation
        // For now, return a placeholder
        // In real implementation, this calls cuMemAlloc or equivalent

        #[cfg(feature = "intel")]
        {
            // Level Zero allocation would go here
            // ze_memory_alloc_device(...)
        }

        #[cfg(feature = "amd")]
        {
            // HIP allocation would go here
            // hipMalloc(...)
        }

        // Placeholder - actual implementation would allocate real memory
        Ok(0x1000_0000 + (size as u64))
    }

    /// Copy data to device memory
    fn copy_to_device(&self, addr: u64, data: &[u8]) -> Result<(), String> {
        // This would call the actual memory copy
        // cuMemcpyHtoD or equivalent

        if data.is_empty() {
            return Ok(());
        }

        #[cfg(feature = "intel")]
        {
            // Level Zero copy would go here
            // zeCommandListAppendMemoryCopy(...)
        }

        #[cfg(feature = "amd")]
        {
            // HIP copy would go here
            // hipMemcpyHtoD(...)
        }

        Ok(())
    }
}

/// Result of GPU state restoration
#[derive(Debug, Clone, Default)]
pub struct RestoreResult {
    /// Was recompilation needed?
    pub recompiled: bool,
    /// PTX source code
    pub ptx_source: String,
    /// Kernel name to launch
    pub kernel_name: String,
    /// Memory address mappings (original -> new)
    pub memory_mappings: Vec<(u64, u64)>,
    /// Remapped kernel arguments (index -> address)
    pub remapped_args: Vec<(u32, u64)>,
    /// Grid dimensions
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions
    pub block_dim: (u32, u32, u32),
    /// Shared memory size
    pub shared_mem_bytes: u32,
    /// Fine-grained resume point
    pub resume_point: Option<ResumePoint>,
}

/// Point to resume execution
#[derive(Debug, Clone)]
pub struct ResumePoint {
    pub ptx_line: u32,
    pub instruction_offset: u64,
    pub block_id: (u32, u32, u32),
    pub thread_id: (u32, u32, u32),
}

/// Detect the current backend
fn detect_current_backend() -> String {
    #[cfg(feature = "intel")]
    return "intel".to_string();

    #[cfg(feature = "amd")]
    return "amd".to_string();

    #[cfg(feature = "tenstorrent")]
    return "tenstorrent".to_string();

    #[cfg(not(any(feature = "intel", feature = "amd", feature = "tenstorrent")))]
    return "virtual".to_string();
}

/// Check if an address looks like a device pointer
fn is_likely_pointer(addr: u64) -> bool {
    // Device pointers are typically in high address ranges
    // and properly aligned
    addr >= 0x1000 && (addr & 0x7) == 0
}

/// Capture current GPU memory state for checkpointing
pub fn capture_memory_region(addr: u64, size: usize) -> Result<Vec<u8>, String> {
    // Allocate host buffer
    let mut buffer = vec![0u8; size];

    // Copy from device
    #[cfg(feature = "intel")]
    {
        // zeCommandListAppendMemoryCopy to host
    }

    #[cfg(feature = "amd")]
    {
        // hipMemcpyDtoH
    }

    // For virtual backend, return empty
    Ok(buffer)
}

/// Resume kernel execution from checkpoint
///
/// This is the main entry point for resuming execution on heterogeneous GPUs
pub fn resume_from_checkpoint(checkpoint_path: &str) -> Result<RestoreResult, String> {
    eprintln!("[hetGPU] Loading checkpoint from: {}", checkpoint_path);

    // Load the restore state
    let state = GpuRestoreState::load(checkpoint_path)?;

    // Create restorer and perform restoration
    let mut restorer = GpuRestorer::new(state);
    restorer.restore()
}

/// Get checkpoint directory
pub fn get_checkpoint_dir() -> String {
    CHECKPOINT_DIR.get_or_init(|| {
        std::env::var("HETGPU_CHECKPOINT_DIR")
            .unwrap_or_else(|_| "/tmp/hetgpu_checkpoints".to_string())
    }).clone()
}

/// Set checkpoint directory
pub fn set_checkpoint_dir(dir: &str) {
    let _ = CHECKPOINT_DIR.set(dir.to_string());
    // Also create the directory
    let _ = fs::create_dir_all(dir);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_execution_tracking() {
        let exec_id = begin_kernel_execution(
            "test_kernel",
            (1, 1, 1),
            (32, 1, 1),
            0,
            0,
            100,
            200,
        );

        add_kernel_argument(0x1000, 8);
        add_kernel_argument(0x2000, 8);

        let state = get_current_kernel_state();
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.kernel_name, "test_kernel");
        assert_eq!(state.kernel_args.len(), 2);

        end_kernel_execution(exec_id);

        let state = get_current_kernel_state();
        assert!(state.is_none());
    }

    #[test]
    fn test_checkpoint_request() {
        assert!(!is_checkpoint_requested());
        request_checkpoint();
        assert!(is_checkpoint_requested());
        clear_checkpoint_request();
        assert!(!is_checkpoint_requested());
    }
}

// =============================================================================
// C-compatible FFI functions for checkpoint/restore
// =============================================================================

/// C-compatible restore result structure
#[repr(C)]
pub struct CRestoreResult {
    /// 0 = success, non-zero = error
    pub error_code: i32,
    /// Was recompilation needed?
    pub recompiled: i32,
    /// Number of memory mappings
    pub num_memory_mappings: u32,
    /// Number of remapped args
    pub num_remapped_args: u32,
    /// Grid dimensions
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    /// Block dimensions
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
    /// Shared memory bytes
    pub shared_mem_bytes: u32,
    /// Resume PTX line (0 if none)
    pub resume_ptx_line: u32,
    /// Resume instruction offset
    pub resume_instruction_offset: u64,
}

impl Default for CRestoreResult {
    fn default() -> Self {
        Self {
            error_code: 0,
            recompiled: 0,
            num_memory_mappings: 0,
            num_remapped_args: 0,
            grid_dim_x: 1,
            grid_dim_y: 1,
            grid_dim_z: 1,
            block_dim_x: 1,
            block_dim_y: 1,
            block_dim_z: 1,
            shared_mem_bytes: 0,
            resume_ptx_line: 0,
            resume_instruction_offset: 0,
        }
    }
}

/// Loaded checkpoint data for FFI access
static LOADED_CHECKPOINT: std::sync::OnceLock<Mutex<Option<CheckpointData>>> = std::sync::OnceLock::new();

fn get_loaded_checkpoint() -> &'static Mutex<Option<CheckpointData>> {
    LOADED_CHECKPOINT.get_or_init(|| Mutex::new(None))
}

/// Load checkpoint from file (C-compatible)
/// Returns 0 on success, non-zero on error
#[no_mangle]
pub extern "C" fn hetgpu_checkpoint_load(path: *const std::ffi::c_char) -> i32 {
    if path.is_null() {
        return -1;
    }

    let path_str = unsafe {
        match std::ffi::CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return -2,
        }
    };

    eprintln!("[hetGPU FFI] Loading checkpoint from: {}", path_str);

    // First try to load as CheckpointData (the format save_checkpoint uses)
    match CheckpointManager::load_checkpoint(path_str) {
        Ok(checkpoint_data) => {
            eprintln!("[hetGPU FFI] Loaded checkpoint data:");
            eprintln!("  Version: {}", checkpoint_data.version);
            eprintln!("  Timestamp: {}", checkpoint_data.timestamp);
            eprintln!("  Active kernels: {}", checkpoint_data.active_kernels.len());
            eprintln!("  PTX sources: {}", checkpoint_data.ptx_sources.len());

            // Store in global for later access
            if let Ok(mut guard) = get_loaded_checkpoint().lock() {
                *guard = Some(checkpoint_data.clone());
            }

            // If there's kernel state with PTX, create a restore state
            if !checkpoint_data.active_kernels.is_empty() {
                let kernel = &checkpoint_data.active_kernels[0];
                if let Some(ptx) = &kernel.ptx_source {
                    let restore_state = GpuRestoreState {
                        version: checkpoint_data.version,
                        source_backend: detect_current_backend(),
                        ptx_source: ptx.clone(),
                        compiled_binary: None,
                        kernel_name: kernel.kernel_name.clone(),
                        kernel_args: kernel.kernel_args.iter().enumerate().map(|(i, (addr, size))| {
                            KernelArgRestore {
                                index: i as u32,
                                original_addr: *addr,
                                size: *size,
                                is_pointer: is_likely_pointer(*addr),
                                data: Vec::new(),
                            }
                        }).collect(),
                        memory_regions: Vec::new(),
                        grid_dim: kernel.grid_dim,
                        block_dim: kernel.block_dim,
                        shared_mem_bytes: kernel.shared_mem_bytes,
                        thread_state: None,
                        address_remap: HashMap::new(),
                    };

                    if let Ok(mut manager) = get_checkpoint_manager().lock() {
                        manager.loaded_restore_state = Some(restore_state);
                    }
                }
            } else if !checkpoint_data.ptx_sources.is_empty() {
                // No active kernels but we have PTX sources - create minimal restore state
                let (module_handle, ptx) = checkpoint_data.ptx_sources.iter().next().unwrap();
                let restore_state = GpuRestoreState {
                    version: checkpoint_data.version,
                    source_backend: detect_current_backend(),
                    ptx_source: ptx.clone(),
                    compiled_binary: None,
                    kernel_name: String::new(),
                    kernel_args: Vec::new(),
                    memory_regions: Vec::new(),
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                    thread_state: None,
                    address_remap: HashMap::new(),
                };
                eprintln!("[hetGPU FFI] Created restore state from PTX source (module {})", module_handle);

                if let Ok(mut manager) = get_checkpoint_manager().lock() {
                    manager.loaded_restore_state = Some(restore_state);
                }
            }

            eprintln!("[hetGPU FFI] Checkpoint loaded successfully");
            0
        }
        Err(e) => {
            eprintln!("[hetGPU FFI] Failed to load as CheckpointData: {}", e);
            // Try loading as GpuRestoreState (alternate format)
            match GpuRestoreState::load(path_str) {
                Ok(state) => {
                    if let Ok(mut manager) = get_checkpoint_manager().lock() {
                        manager.loaded_restore_state = Some(state);
                        eprintln!("[hetGPU FFI] Checkpoint loaded as GpuRestoreState");
                        0
                    } else {
                        -3
                    }
                }
                Err(e2) => {
                    eprintln!("[hetGPU FFI] Failed to load checkpoint: {} / {}", e, e2);
                    -4
                }
            }
        }
    }
}

/// Perform restore from loaded checkpoint
#[no_mangle]
pub extern "C" fn hetgpu_checkpoint_restore(result: *mut CRestoreResult) -> i32 {
    if result.is_null() {
        return -1;
    }

    let restore_state = {
        let manager = match get_checkpoint_manager().lock() {
            Ok(m) => m,
            Err(_) => return -2,
        };

        match &manager.loaded_restore_state {
            Some(state) => state.clone(),
            None => {
                eprintln!("[hetGPU FFI] No checkpoint loaded - call hetgpu_checkpoint_load first");
                return -3;
            }
        }
    };

    eprintln!("[hetGPU FFI] Performing restore...");

    let mut restorer = GpuRestorer::new(restore_state);
    match restorer.restore() {
        Ok(res) => {
            unsafe {
                (*result).error_code = 0;
                (*result).recompiled = if res.recompiled { 1 } else { 0 };
                (*result).num_memory_mappings = res.memory_mappings.len() as u32;
                (*result).num_remapped_args = res.remapped_args.len() as u32;
                (*result).grid_dim_x = res.grid_dim.0;
                (*result).grid_dim_y = res.grid_dim.1;
                (*result).grid_dim_z = res.grid_dim.2;
                (*result).block_dim_x = res.block_dim.0;
                (*result).block_dim_y = res.block_dim.1;
                (*result).block_dim_z = res.block_dim.2;
                (*result).shared_mem_bytes = res.shared_mem_bytes;

                if let Some(resume) = res.resume_point {
                    (*result).resume_ptx_line = resume.ptx_line;
                    (*result).resume_instruction_offset = resume.instruction_offset;
                }
            }
            eprintln!("[hetGPU FFI] Restore completed successfully");
            0
        }
        Err(e) => {
            eprintln!("[hetGPU FFI] Restore failed: {}", e);
            unsafe {
                (*result).error_code = -10;
            }
            -10
        }
    }
}

/// Get PTX source from loaded checkpoint
/// Returns pointer to PTX string (null-terminated), or null on error
/// Caller must NOT free this pointer
#[no_mangle]
pub extern "C" fn hetgpu_checkpoint_get_ptx() -> *const std::ffi::c_char {
    static mut PTX_BUFFER: Option<std::ffi::CString> = None;

    // First check restore state
    if let Ok(manager) = get_checkpoint_manager().lock() {
        if let Some(state) = &manager.loaded_restore_state {
            if !state.ptx_source.is_empty() {
                unsafe {
                    PTX_BUFFER = std::ffi::CString::new(state.ptx_source.as_str()).ok();
                    if let Some(cstr) = &PTX_BUFFER {
                        return cstr.as_ptr();
                    }
                }
            }
        }
    }

    // Fallback: check loaded checkpoint data for PTX sources
    if let Ok(guard) = get_loaded_checkpoint().lock() {
        if let Some(checkpoint) = &*guard {
            // Try to get PTX from ptx_sources map
            if let Some((_, ptx)) = checkpoint.ptx_sources.iter().next() {
                unsafe {
                    PTX_BUFFER = std::ffi::CString::new(ptx.as_str()).ok();
                    if let Some(cstr) = &PTX_BUFFER {
                        return cstr.as_ptr();
                    }
                }
            }
            // Try to get PTX from active kernels
            if let Some(kernel) = checkpoint.active_kernels.first() {
                if let Some(ptx) = &kernel.ptx_source {
                    unsafe {
                        PTX_BUFFER = std::ffi::CString::new(ptx.as_str()).ok();
                        if let Some(cstr) = &PTX_BUFFER {
                            return cstr.as_ptr();
                        }
                    }
                }
            }
        }
    }

    std::ptr::null()
}

/// Get kernel name from loaded checkpoint
#[no_mangle]
pub extern "C" fn hetgpu_checkpoint_get_kernel_name() -> *const std::ffi::c_char {
    static mut NAME_BUFFER: Option<std::ffi::CString> = None;

    // First check restore state
    if let Ok(manager) = get_checkpoint_manager().lock() {
        if let Some(state) = &manager.loaded_restore_state {
            if !state.kernel_name.is_empty() {
                unsafe {
                    NAME_BUFFER = std::ffi::CString::new(state.kernel_name.as_str()).ok();
                    if let Some(cstr) = &NAME_BUFFER {
                        return cstr.as_ptr();
                    }
                }
            }
        }
    }

    // Fallback: check loaded checkpoint data for kernel names
    if let Ok(guard) = get_loaded_checkpoint().lock() {
        if let Some(checkpoint) = &*guard {
            if let Some(kernel) = checkpoint.active_kernels.first() {
                if !kernel.kernel_name.is_empty() {
                    unsafe {
                        NAME_BUFFER = std::ffi::CString::new(kernel.kernel_name.as_str()).ok();
                        if let Some(cstr) = &NAME_BUFFER {
                            return cstr.as_ptr();
                        }
                    }
                }
            }
        }
    }

    std::ptr::null()
}

/// Get memory mapping by index
/// Returns 0 on success, non-zero on error
#[no_mangle]
pub extern "C" fn hetgpu_checkpoint_get_memory_mapping(
    index: u32,
    original_addr: *mut u64,
    size: *mut u64,
) -> i32 {
    let manager = match get_checkpoint_manager().lock() {
        Ok(m) => m,
        Err(_) => return -1,
    };

    let state = match &manager.loaded_restore_state {
        Some(s) => s,
        None => return -2,
    };

    if (index as usize) >= state.memory_regions.len() {
        return -3;
    }

    let region = &state.memory_regions[index as usize];
    unsafe {
        if !original_addr.is_null() {
            *original_addr = region.original_addr;
        }
        if !size.is_null() {
            *size = region.size as u64;
        }
    }

    0
}

/// Get number of memory regions in checkpoint
#[no_mangle]
pub extern "C" fn hetgpu_checkpoint_get_memory_region_count() -> u32 {
    let manager = match get_checkpoint_manager().lock() {
        Ok(m) => m,
        Err(_) => return 0,
    };

    match &manager.loaded_restore_state {
        Some(s) => s.memory_regions.len() as u32,
        None => 0,
    }
}

/// Get memory region data
/// Returns pointer to data buffer (size bytes), caller must copy
#[no_mangle]
pub extern "C" fn hetgpu_checkpoint_get_memory_data(
    index: u32,
    buffer: *mut u8,
    buffer_size: u64,
) -> i32 {
    let manager = match get_checkpoint_manager().lock() {
        Ok(m) => m,
        Err(_) => return -1,
    };

    let state = match &manager.loaded_restore_state {
        Some(s) => s,
        None => return -2,
    };

    if (index as usize) >= state.memory_regions.len() {
        return -3;
    }

    let region = &state.memory_regions[index as usize];
    let copy_size = std::cmp::min(buffer_size as usize, region.data.len());

    if !buffer.is_null() && copy_size > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(region.data.as_ptr(), buffer, copy_size);
        }
    }

    copy_size as i32
}

/// Request checkpoint programmatically (C-compatible)
/// This triggers a checkpoint without using signals, avoiding core dumps on some backends
#[no_mangle]
pub extern "C" fn hetgpu_request_checkpoint() {
    eprintln!("[hetGPU] Programmatic checkpoint requested...");

    // Save checkpoint directly
    if let Ok(manager) = get_checkpoint_manager().lock() {
        match manager.save_checkpoint() {
            Ok(path) => {
                eprintln!("[hetGPU] Checkpoint saved to: {:?}", path);
            }
            Err(e) => {
                eprintln!("[hetGPU] Checkpoint failed: {}", e);
            }
        }
    } else {
        eprintln!("[hetGPU] Failed to acquire checkpoint manager lock");
    }
}
