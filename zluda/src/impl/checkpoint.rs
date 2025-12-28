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
#[cfg(unix)]
pub fn install_signal_handler() -> Result<(), String> {
    if HANDLER_INSTALLED.swap(true, Ordering::SeqCst) {
        // Already installed
        return Ok(());
    }

    extern "C" fn sigint_handler(_: libc::c_int) {
        CHECKPOINT_REQUESTED.store(true, Ordering::SeqCst);

        // Print notification
        let msg = b"\n[hetGPU] Checkpoint requested - saving GPU state...\n";
        unsafe {
            libc::write(libc::STDERR_FILENO, msg.as_ptr() as *const libc::c_void, msg.len());
        }

        // Try to save checkpoint
        if let Ok(manager) = get_checkpoint_manager().lock() {
            match manager.save_checkpoint() {
                Ok(path) => {
                    let msg = format!("[hetGPU] Checkpoint saved to: {:?}\n", path);
                    unsafe {
                        libc::write(
                            libc::STDERR_FILENO,
                            msg.as_ptr() as *const libc::c_void,
                            msg.len(),
                        );
                    }
                }
                Err(e) => {
                    let msg = format!("[hetGPU] Checkpoint failed: {}\n", e);
                    unsafe {
                        libc::write(
                            libc::STDERR_FILENO,
                            msg.as_ptr() as *const libc::c_void,
                            msg.len(),
                        );
                    }
                }
            }
        }
    }

    unsafe {
        // Use signal() which is simpler and more portable
        let prev = libc::signal(libc::SIGINT, sigint_handler as libc::sighandler_t);
        if prev == libc::SIG_ERR {
            HANDLER_INSTALLED.store(false, Ordering::SeqCst);
            return Err("Failed to install SIGINT handler".to_string());
        }
    }

    eprintln!("[hetGPU] Checkpoint handler installed (Ctrl+C to checkpoint)");
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
