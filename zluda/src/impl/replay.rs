//! Kernel-centric Record/Replay System for Heterogeneous GPU Debugging
//!
//! This module provides functionality to:
//! 1. Record kernel launches and memory state during execution
//! 2. Replay recorded traces on different GPU backends
//! 3. Validate outputs for debugging and profiling
//!
//! The system uses incremental/dirty page tracking to minimize trace size.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// =============================================================================
// Constants
// =============================================================================

/// Magic number for replay trace files: "HETR" in ASCII
pub const REPLAY_TRACE_MAGIC: u32 = 0x48455452;

/// Current replay trace version
pub const REPLAY_TRACE_VERSION: u32 = 1;

/// Page size for dirty tracking (4KB)
pub const PAGE_SIZE: usize = 4096;

/// Default checkpoint directory
pub const DEFAULT_TRACE_DIR: &str = "/tmp/hetgpu_traces";

// =============================================================================
// Trace File Structures
// =============================================================================

/// Compression type for memory data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Zstd,
    Lz4,
}

impl Default for CompressionType {
    fn default() -> Self {
        CompressionType::None
    }
}

/// Target backend for replay
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    Nvidia,
    Amd,
    Intel,
    Tenstorrent,
    Virtual,
}

impl Default for Backend {
    fn default() -> Self {
        Backend::Virtual
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Nvidia => write!(f, "nvidia"),
            Backend::Amd => write!(f, "amd"),
            Backend::Intel => write!(f, "intel"),
            Backend::Tenstorrent => write!(f, "tenstorrent"),
            Backend::Virtual => write!(f, "virtual"),
        }
    }
}

/// Header for replay trace files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayTraceHeader {
    /// Magic number (should be REPLAY_TRACE_MAGIC)
    pub magic: u32,
    /// Trace format version
    pub version: u32,
    /// Unix timestamp when trace was created
    pub timestamp: u64,
    /// Source backend where trace was recorded
    pub source_backend: Backend,
    /// GPU device name
    pub device_name: String,
    /// CUDA compute capability (major, minor)
    pub compute_capability: (u32, u32),
    /// Total number of kernel launches
    pub total_kernels: u64,
    /// Total number of memory operations
    pub total_memory_ops: u64,
    /// Compression used for memory data
    pub compression: CompressionType,
}

impl Default for ReplayTraceHeader {
    fn default() -> Self {
        Self {
            magic: REPLAY_TRACE_MAGIC,
            version: REPLAY_TRACE_VERSION,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            source_backend: Backend::default(),
            device_name: String::new(),
            compute_capability: (0, 0),
            total_kernels: 0,
            total_memory_ops: 0,
            compression: CompressionType::default(),
        }
    }
}

/// Complete replay trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayTrace {
    /// Trace header
    pub header: ReplayTraceHeader,
    /// PTX sources by module handle
    pub ptx_sources: HashMap<u64, String>,
    /// Memory allocations
    pub allocations: Vec<TrackedAllocation>,
    /// Recorded kernel launches
    pub kernel_launches: Vec<RecordedKernelLaunch>,
}

impl Default for ReplayTrace {
    fn default() -> Self {
        Self {
            header: ReplayTraceHeader::default(),
            ptx_sources: HashMap::new(),
            allocations: Vec::new(),
            kernel_launches: Vec::new(),
        }
    }
}

impl ReplayTrace {
    /// Create a new empty trace
    pub fn new(source_backend: Backend, device_name: &str) -> Self {
        let mut trace = Self::default();
        trace.header.source_backend = source_backend;
        trace.header.device_name = device_name.to_string();
        trace
    }

    /// Save trace to file
    pub fn save(&self, path: &str) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| format!("Failed to serialize trace: {}", e))?;
        Ok(())
    }

    /// Load trace from file
    pub fn load(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = BufReader::new(file);
        let trace: ReplayTrace = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to deserialize trace: {}", e))?;

        // Validate magic number
        if trace.header.magic != REPLAY_TRACE_MAGIC {
            return Err("Invalid trace file: bad magic number".to_string());
        }

        Ok(trace)
    }
}

// =============================================================================
// Kernel Launch Recording
// =============================================================================

/// A single recorded kernel launch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedKernelLaunch {
    /// Unique sequence number for ordering
    pub sequence_id: u64,
    /// Timestamp (nanoseconds since trace start)
    pub timestamp_ns: u64,
    /// Kernel name
    pub kernel_name: String,
    /// Module handle (for PTX lookup)
    pub module_handle: u64,
    /// Function handle
    pub function_handle: u64,
    /// Grid dimensions (x, y, z)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block_dim: (u32, u32, u32),
    /// Shared memory size in bytes
    pub shared_mem_bytes: u32,
    /// Stream handle
    pub stream_handle: u64,
    /// Kernel arguments
    pub arguments: Vec<KernelArgument>,
    /// Pre-execution memory snapshots (input state)
    pub input_memory: Vec<MemorySnapshot>,
    /// Post-execution memory snapshots (output state for validation)
    pub output_memory: Vec<MemorySnapshot>,
    /// Execution duration in nanoseconds
    pub execution_duration_ns: Option<u64>,
}

impl RecordedKernelLaunch {
    /// Create a new recorded kernel launch
    pub fn new(
        sequence_id: u64,
        timestamp_ns: u64,
        kernel_name: &str,
        module_handle: u64,
        function_handle: u64,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream_handle: u64,
    ) -> Self {
        Self {
            sequence_id,
            timestamp_ns,
            kernel_name: kernel_name.to_string(),
            module_handle,
            function_handle,
            grid_dim,
            block_dim,
            shared_mem_bytes,
            stream_handle,
            arguments: Vec::new(),
            input_memory: Vec::new(),
            output_memory: Vec::new(),
            execution_duration_ns: None,
        }
    }
}

/// Kernel argument type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArgumentType {
    Scalar,
    DevicePointer,
    HostPointer,
    Texture,
    Surface,
}

/// Kernel argument descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelArgument {
    /// Argument index (0-based)
    pub index: u32,
    /// Size in bytes
    pub size: usize,
    /// Argument type
    pub arg_type: ArgumentType,
    /// For pointer args: the device address
    pub device_addr: Option<u64>,
    /// For scalar args: raw bytes (base64 encoded in JSON)
    pub scalar_data: Option<Vec<u8>>,
}

// =============================================================================
// Memory Snapshot Structures
// =============================================================================

/// Snapshot type for incremental tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnapshotType {
    /// Complete memory dump
    Full,
    /// Only dirty pages/regions
    Incremental,
    /// Reference to a previous snapshot (for unchanged memory)
    Reference {
        /// ID of the referenced snapshot
        snapshot_id: u64,
    },
}

/// Memory snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Unique snapshot ID
    pub snapshot_id: u64,
    /// Original device address
    pub device_addr: u64,
    /// Total allocation size
    pub allocation_size: u64,
    /// Snapshot type
    pub snapshot_type: SnapshotType,
    /// For full snapshots: complete data (base64 in JSON)
    pub full_data: Option<Vec<u8>>,
    /// For incremental snapshots: dirty regions only
    pub dirty_regions: Option<Vec<DirtyRegion>>,
    /// CRC32 checksum for validation
    pub checksum: u32,
}

impl MemorySnapshot {
    /// Create a full memory snapshot
    pub fn full(snapshot_id: u64, device_addr: u64, data: Vec<u8>) -> Self {
        let checksum = crc32_checksum(&data);
        Self {
            snapshot_id,
            device_addr,
            allocation_size: data.len() as u64,
            snapshot_type: SnapshotType::Full,
            full_data: Some(data),
            dirty_regions: None,
            checksum,
        }
    }

    /// Create an incremental memory snapshot
    pub fn incremental(
        snapshot_id: u64,
        device_addr: u64,
        allocation_size: u64,
        dirty_regions: Vec<DirtyRegion>,
    ) -> Self {
        // Compute checksum over all dirty region data
        let mut all_data = Vec::new();
        for region in &dirty_regions {
            all_data.extend_from_slice(&region.data);
        }
        let checksum = crc32_checksum(&all_data);

        Self {
            snapshot_id,
            device_addr,
            allocation_size,
            snapshot_type: SnapshotType::Incremental,
            full_data: None,
            dirty_regions: Some(dirty_regions),
            checksum,
        }
    }
}

/// Dirty memory region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirtyRegion {
    /// Offset from base address
    pub offset: u64,
    /// Size of dirty region
    pub size: u64,
    /// Dirty data
    pub data: Vec<u8>,
}

/// Simple CRC32 checksum
fn crc32_checksum(data: &[u8]) -> u32 {
    // Simple checksum implementation
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

// =============================================================================
// Memory Allocation Tracking
// =============================================================================

/// Allocation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationType {
    Device,
    Managed,
    Pinned,
    Mapped,
    Host,
}

/// Tracked memory allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedAllocation {
    /// Device address
    pub device_addr: u64,
    /// Allocation size in bytes
    pub size: u64,
    /// Timestamp when allocated
    pub allocation_time: u64,
    /// Type of allocation
    pub allocation_type: AllocationType,
    /// Page-level dirty bits (one bit per 4KB page)
    #[serde(skip)]
    pub dirty_bitmap: Vec<u64>,
    /// Last snapshot sequence ID for this allocation
    pub last_snapshot_seq: u64,
    /// Shadow copy for dirty detection (not serialized)
    #[serde(skip)]
    pub shadow_copy: Option<Vec<u8>>,
}

impl TrackedAllocation {
    /// Create a new tracked allocation
    pub fn new(device_addr: u64, size: u64, allocation_type: AllocationType) -> Self {
        let num_pages = ((size as usize) + PAGE_SIZE - 1) / PAGE_SIZE;
        let bitmap_words = (num_pages + 63) / 64;

        Self {
            device_addr,
            size,
            allocation_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            allocation_type,
            dirty_bitmap: vec![0u64; bitmap_words],
            last_snapshot_seq: 0,
            shadow_copy: None,
        }
    }

    /// Get the number of pages in this allocation
    pub fn num_pages(&self) -> usize {
        ((self.size as usize) + PAGE_SIZE - 1) / PAGE_SIZE
    }

    /// Mark a page as dirty
    pub fn mark_page_dirty(&mut self, page_index: usize) {
        if page_index < self.num_pages() {
            let word_idx = page_index / 64;
            let bit_idx = page_index % 64;
            if word_idx < self.dirty_bitmap.len() {
                self.dirty_bitmap[word_idx] |= 1u64 << bit_idx;
            }
        }
    }

    /// Check if a page is dirty
    pub fn is_page_dirty(&self, page_index: usize) -> bool {
        if page_index >= self.num_pages() {
            return false;
        }
        let word_idx = page_index / 64;
        let bit_idx = page_index % 64;
        if word_idx < self.dirty_bitmap.len() {
            (self.dirty_bitmap[word_idx] & (1u64 << bit_idx)) != 0
        } else {
            false
        }
    }

    /// Clear all dirty bits
    pub fn clear_dirty_bits(&mut self) {
        for word in &mut self.dirty_bitmap {
            *word = 0;
        }
    }

    /// Mark all pages as dirty
    pub fn mark_all_dirty(&mut self) {
        for word in &mut self.dirty_bitmap {
            *word = !0u64;
        }
    }
}

// =============================================================================
// Dirty Page Tracker
// =============================================================================

/// Dirty tracking method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirtyTrackingMethod {
    /// Full memory dump each time (no optimization)
    FullDump,
    /// Software shadow copy comparison
    SoftwareShadow,
    /// Checksum-based dirty detection
    Checksum,
}

/// Dirty page tracker
pub struct DirtyPageTracker {
    /// Tracked allocations by device address
    allocations: HashMap<u64, TrackedAllocation>,
    /// Dirty tracking method
    method: DirtyTrackingMethod,
    /// Snapshot ID counter
    snapshot_counter: AtomicU64,
}

impl DirtyPageTracker {
    /// Create a new dirty page tracker
    pub fn new(method: DirtyTrackingMethod) -> Self {
        Self {
            allocations: HashMap::new(),
            method,
            snapshot_counter: AtomicU64::new(0),
        }
    }

    /// Track a new allocation
    pub fn track_allocation(&mut self, device_addr: u64, size: u64, alloc_type: AllocationType) {
        let mut allocation = TrackedAllocation::new(device_addr, size, alloc_type);

        // Initialize shadow copy if using software shadow method
        if self.method == DirtyTrackingMethod::SoftwareShadow {
            allocation.shadow_copy = Some(vec![0u8; size as usize]);
        }

        // Mark all pages as dirty initially
        allocation.mark_all_dirty();

        self.allocations.insert(device_addr, allocation);
        eprintln!(
            "[hetGPU Replay] Tracking allocation: addr=0x{:x}, size={}, type={:?}",
            device_addr, size, alloc_type
        );
    }

    /// Stop tracking an allocation
    pub fn untrack_allocation(&mut self, device_addr: u64) {
        if self.allocations.remove(&device_addr).is_some() {
            eprintln!(
                "[hetGPU Replay] Untracked allocation: addr=0x{:x}",
                device_addr
            );
        }
    }

    /// Mark a memory region as dirty
    pub fn mark_dirty(&mut self, device_addr: u64, size: u64) {
        // Find the allocation containing this address
        for (base_addr, allocation) in &mut self.allocations {
            let end_addr = base_addr + allocation.size;
            if device_addr >= *base_addr && device_addr < end_addr {
                // Calculate page range
                let offset = device_addr - base_addr;
                let start_page = (offset as usize) / PAGE_SIZE;
                let end_page = ((offset + size) as usize + PAGE_SIZE - 1) / PAGE_SIZE;

                for page in start_page..end_page.min(allocation.num_pages()) {
                    allocation.mark_page_dirty(page);
                }
                break;
            }
        }
    }

    /// Get next snapshot ID
    fn next_snapshot_id(&self) -> u64 {
        self.snapshot_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Capture memory snapshot for an allocation
    /// This requires a callback to read device memory
    pub fn capture_snapshot<F>(
        &mut self,
        device_addr: u64,
        read_device_memory: F,
    ) -> Option<MemorySnapshot>
    where
        F: Fn(u64, u64) -> Option<Vec<u8>>,
    {
        // Get snapshot ID and method before mutable borrow
        let snapshot_id = self.next_snapshot_id();
        let method = self.method;

        let allocation = self.allocations.get_mut(&device_addr)?;
        let alloc_size = allocation.size;

        match method {
            DirtyTrackingMethod::FullDump => {
                // Always capture full memory
                let data = read_device_memory(device_addr, alloc_size)?;
                Some(MemorySnapshot::full(snapshot_id, device_addr, data))
            }
            DirtyTrackingMethod::SoftwareShadow => {
                // Compare with shadow copy and capture only dirty regions
                let current = read_device_memory(device_addr, alloc_size)?;
                let shadow = allocation.shadow_copy.as_ref()?;

                let dirty_regions = detect_dirty_regions_static(&current, shadow, allocation);

                if dirty_regions.is_empty() {
                    // No changes - return reference to previous snapshot
                    let last_seq = allocation.last_snapshot_seq;
                    Some(MemorySnapshot {
                        snapshot_id,
                        device_addr,
                        allocation_size: alloc_size,
                        snapshot_type: SnapshotType::Reference {
                            snapshot_id: last_seq,
                        },
                        full_data: None,
                        dirty_regions: None,
                        checksum: 0,
                    })
                } else {
                    // Update shadow copy
                    allocation.shadow_copy = Some(current);
                    allocation.clear_dirty_bits();
                    allocation.last_snapshot_seq = snapshot_id;

                    Some(MemorySnapshot::incremental(
                        snapshot_id,
                        device_addr,
                        alloc_size,
                        dirty_regions,
                    ))
                }
            }
            DirtyTrackingMethod::Checksum => {
                // Use bitmap-based dirty tracking
                let current = read_device_memory(device_addr, alloc_size)?;
                let mut dirty_regions = Vec::new();

                let mut region_start: Option<usize> = None;
                let num_pages = allocation.num_pages();
                for page_idx in 0..num_pages {
                    if allocation.is_page_dirty(page_idx) {
                        if region_start.is_none() {
                            region_start = Some(page_idx * PAGE_SIZE);
                        }
                    } else if let Some(start) = region_start {
                        let end = page_idx * PAGE_SIZE;
                        dirty_regions.push(DirtyRegion {
                            offset: start as u64,
                            size: (end - start) as u64,
                            data: current[start..end.min(current.len())].to_vec(),
                        });
                        region_start = None;
                    }
                }

                // Handle trailing region
                if let Some(start) = region_start {
                    dirty_regions.push(DirtyRegion {
                        offset: start as u64,
                        size: (current.len() - start) as u64,
                        data: current[start..].to_vec(),
                    });
                }

                let last_seq = allocation.last_snapshot_seq;
                allocation.clear_dirty_bits();
                allocation.last_snapshot_seq = snapshot_id;

                if dirty_regions.is_empty() {
                    Some(MemorySnapshot {
                        snapshot_id,
                        device_addr,
                        allocation_size: alloc_size,
                        snapshot_type: SnapshotType::Reference {
                            snapshot_id: last_seq,
                        },
                        full_data: None,
                        dirty_regions: None,
                        checksum: 0,
                    })
                } else {
                    Some(MemorySnapshot::incremental(
                        snapshot_id,
                        device_addr,
                        alloc_size,
                        dirty_regions,
                    ))
                }
            }
        }
    }

    /// Get all tracked allocations
    pub fn get_allocations(&self) -> Vec<&TrackedAllocation> {
        self.allocations.values().collect()
    }

    /// Check if an address is tracked
    pub fn is_tracked(&self, device_addr: u64) -> bool {
        self.allocations.contains_key(&device_addr)
    }
}

/// Detect dirty regions by comparing current memory with shadow copy (standalone function)
fn detect_dirty_regions_static(
    current: &[u8],
    shadow: &[u8],
    allocation: &TrackedAllocation,
) -> Vec<DirtyRegion> {
    let mut dirty_regions = Vec::new();
    let mut region_start: Option<usize> = None;

    for page_idx in 0..allocation.num_pages() {
        let page_start = page_idx * PAGE_SIZE;
        let page_end = (page_start + PAGE_SIZE).min(current.len());

        if page_end > shadow.len() {
            break;
        }

        let is_dirty = &current[page_start..page_end] != &shadow[page_start..page_end];

        if is_dirty {
            if region_start.is_none() {
                region_start = Some(page_start);
            }
        } else if let Some(start) = region_start {
            dirty_regions.push(DirtyRegion {
                offset: start as u64,
                size: (page_start - start) as u64,
                data: current[start..page_start].to_vec(),
            });
            region_start = None;
        }
    }

    // Handle trailing dirty region
    if let Some(start) = region_start {
        dirty_regions.push(DirtyRegion {
            offset: start as u64,
            size: (current.len() - start) as u64,
            data: current[start..].to_vec(),
        });
    }

    dirty_regions
}

// =============================================================================
// Recording Session
// =============================================================================

/// Recording configuration
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    /// Enable dirty page tracking
    pub dirty_tracking: bool,
    /// Dirty tracking method
    pub dirty_method: DirtyTrackingMethod,
    /// Record pre-launch memory state (inputs)
    pub capture_inputs: bool,
    /// Record post-launch memory state (outputs for validation)
    pub capture_outputs: bool,
    /// Maximum memory to record per kernel (0 = unlimited)
    pub max_memory_per_kernel: usize,
    /// Compression for memory data
    pub compression: CompressionType,
    /// Output file path
    pub output_path: PathBuf,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            dirty_tracking: true,
            dirty_method: DirtyTrackingMethod::SoftwareShadow,
            capture_inputs: true,
            capture_outputs: true,
            max_memory_per_kernel: 0, // unlimited
            compression: CompressionType::None,
            output_path: PathBuf::from(DEFAULT_TRACE_DIR).join("trace.hetr"),
        }
    }
}

impl RecordingConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("HETGPU_RECORD_OUTPUT") {
            config.output_path = PathBuf::from(val);
        }

        if let Ok(val) = std::env::var("HETGPU_RECORD_INPUTS") {
            config.capture_inputs = val == "1" || val.to_lowercase() == "true";
        }

        if let Ok(val) = std::env::var("HETGPU_RECORD_OUTPUTS") {
            config.capture_outputs = val == "1" || val.to_lowercase() == "true";
        }

        if let Ok(val) = std::env::var("HETGPU_RECORD_DIRTY_METHOD") {
            config.dirty_method = match val.to_lowercase().as_str() {
                "full" => DirtyTrackingMethod::FullDump,
                "shadow" => DirtyTrackingMethod::SoftwareShadow,
                "checksum" => DirtyTrackingMethod::Checksum,
                _ => DirtyTrackingMethod::SoftwareShadow,
            };
        }

        config
    }
}

/// Recording session state
pub struct RecordingSession {
    /// Is recording active?
    active: AtomicBool,
    /// Trace being recorded
    trace: Mutex<ReplayTrace>,
    /// Dirty page tracker
    dirty_tracker: Mutex<DirtyPageTracker>,
    /// Sequence counter for kernel launches
    sequence: AtomicU64,
    /// Start time of recording
    start_time: Instant,
    /// Recording configuration
    config: RecordingConfig,
    /// Current kernel being recorded (for post-launch capture)
    current_launch: Mutex<Option<RecordedKernelLaunch>>,
}

impl RecordingSession {
    /// Create a new recording session
    pub fn new(config: RecordingConfig, source_backend: Backend, device_name: &str) -> Self {
        Self {
            active: AtomicBool::new(false),
            trace: Mutex::new(ReplayTrace::new(source_backend, device_name)),
            dirty_tracker: Mutex::new(DirtyPageTracker::new(config.dirty_method)),
            sequence: AtomicU64::new(0),
            start_time: Instant::now(),
            config,
            current_launch: Mutex::new(None),
        }
    }

    /// Start recording
    pub fn start(&self) {
        self.active.store(true, Ordering::SeqCst);
        eprintln!("[hetGPU Replay] Recording started");
    }

    /// Stop recording and save trace
    pub fn stop(&self) -> Result<PathBuf, String> {
        self.active.store(false, Ordering::SeqCst);

        let trace = self.trace.lock().map_err(|e| e.to_string())?;

        // Ensure output directory exists
        if let Some(parent) = self.config.output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create output directory: {}", e))?;
        }

        // Save trace
        let path = self.config.output_path.to_string_lossy().to_string();
        trace.save(&path)?;

        eprintln!(
            "[hetGPU Replay] Recording stopped. Trace saved to: {}",
            path
        );
        eprintln!(
            "[hetGPU Replay] Recorded {} kernel launches",
            trace.kernel_launches.len()
        );

        Ok(self.config.output_path.clone())
    }

    /// Check if recording is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    /// Get elapsed time since recording started
    pub fn elapsed_ns(&self) -> u64 {
        self.start_time.elapsed().as_nanos() as u64
    }

    /// Get next sequence ID
    pub fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst)
    }

    /// Register PTX source
    pub fn register_ptx(&self, module_handle: u64, ptx_source: &str) {
        if let Ok(mut trace) = self.trace.lock() {
            trace
                .ptx_sources
                .insert(module_handle, ptx_source.to_string());
        }
    }

    /// Track memory allocation
    pub fn track_allocation(&self, device_addr: u64, size: u64, alloc_type: AllocationType) {
        if let Ok(mut tracker) = self.dirty_tracker.lock() {
            tracker.track_allocation(device_addr, size, alloc_type);
        }

        if let Ok(mut trace) = self.trace.lock() {
            trace
                .allocations
                .push(TrackedAllocation::new(device_addr, size, alloc_type));
        }
    }

    /// Untrack memory allocation
    pub fn untrack_allocation(&self, device_addr: u64) {
        if let Ok(mut tracker) = self.dirty_tracker.lock() {
            tracker.untrack_allocation(device_addr);
        }
    }

    /// Mark memory as dirty
    pub fn mark_dirty(&self, device_addr: u64, size: u64) {
        if let Ok(mut tracker) = self.dirty_tracker.lock() {
            tracker.mark_dirty(device_addr, size);
        }
    }

    /// Record kernel pre-launch
    pub fn record_pre_launch(
        &self,
        kernel_name: &str,
        module_handle: u64,
        function_handle: u64,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream_handle: u64,
        arguments: Vec<KernelArgument>,
    ) -> u64 {
        let sequence_id = self.next_sequence();
        let timestamp_ns = self.elapsed_ns();

        let mut launch = RecordedKernelLaunch::new(
            sequence_id,
            timestamp_ns,
            kernel_name,
            module_handle,
            function_handle,
            grid_dim,
            block_dim,
            shared_mem_bytes,
            stream_handle,
        );
        launch.arguments = arguments;

        // Capture input memory if configured
        // (This would need device memory read callback - placeholder for now)

        if let Ok(mut current) = self.current_launch.lock() {
            *current = Some(launch);
        }

        sequence_id
    }

    /// Record kernel post-launch
    pub fn record_post_launch(&self, _sequence_id: u64, execution_time_ns: u64) {
        if let Ok(mut current) = self.current_launch.lock() {
            if let Some(mut launch) = current.take() {
                launch.execution_duration_ns = Some(execution_time_ns);

                // Capture output memory if configured
                // (This would need device memory read callback - placeholder for now)

                // Add to trace
                if let Ok(mut trace) = self.trace.lock() {
                    trace.header.total_kernels += 1;
                    trace.kernel_launches.push(launch);
                }
            }
        }
    }

    /// Get recording statistics
    pub fn get_stats(&self) -> RecordingStats {
        let trace = self.trace.lock().ok();
        let tracker = self.dirty_tracker.lock().ok();

        RecordingStats {
            kernels_recorded: trace
                .as_ref()
                .map(|t| t.kernel_launches.len() as u64)
                .unwrap_or(0),
            allocations_tracked: tracker
                .as_ref()
                .map(|t| t.get_allocations().len() as u64)
                .unwrap_or(0),
            recording_duration_ns: self.elapsed_ns(),
            is_active: self.is_active(),
        }
    }
}

/// Recording statistics
#[derive(Debug, Clone)]
pub struct RecordingStats {
    pub kernels_recorded: u64,
    pub allocations_tracked: u64,
    pub recording_duration_ns: u64,
    pub is_active: bool,
}

// =============================================================================
// Global Recording Session
// =============================================================================

static RECORDING_SESSION: OnceLock<Mutex<Option<RecordingSession>>> = OnceLock::new();

fn get_recording_session() -> &'static Mutex<Option<RecordingSession>> {
    RECORDING_SESSION.get_or_init(|| Mutex::new(None))
}

/// Check if recording is enabled via environment
pub fn is_recording_enabled() -> bool {
    std::env::var("HETGPU_RECORD")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Check if recording is currently active
pub fn is_recording_active() -> bool {
    if let Ok(guard) = get_recording_session().lock() {
        if let Some(session) = &*guard {
            return session.is_active();
        }
    }
    false
}

/// Start recording session
pub fn start_recording(source_backend: Backend, device_name: &str) -> Result<(), String> {
    let config = RecordingConfig::from_env();

    let session = RecordingSession::new(config, source_backend, device_name);
    session.start();

    let mut guard = get_recording_session().lock().map_err(|e| e.to_string())?;
    *guard = Some(session);

    Ok(())
}

/// Stop recording and save trace
pub fn stop_recording() -> Result<PathBuf, String> {
    let guard = get_recording_session().lock().map_err(|e| e.to_string())?;

    if let Some(session) = &*guard {
        session.stop()
    } else {
        Err("No active recording session".to_string())
    }
}

/// Register PTX source for recording
pub fn record_ptx_source(module_handle: u64, ptx_source: &str) {
    if let Ok(guard) = get_recording_session().lock() {
        if let Some(session) = &*guard {
            if session.is_active() {
                session.register_ptx(module_handle, ptx_source);
            }
        }
    }
}

/// Track allocation for recording
pub fn record_allocation(device_addr: u64, size: u64, alloc_type: AllocationType) {
    if let Ok(guard) = get_recording_session().lock() {
        if let Some(session) = &*guard {
            if session.is_active() {
                session.track_allocation(device_addr, size, alloc_type);
            }
        }
    }
}

/// Untrack allocation for recording
pub fn record_deallocation(device_addr: u64) {
    if let Ok(guard) = get_recording_session().lock() {
        if let Some(session) = &*guard {
            if session.is_active() {
                session.untrack_allocation(device_addr);
            }
        }
    }
}

/// Mark memory as dirty for recording
pub fn record_memory_dirty(device_addr: u64, size: u64) {
    if let Ok(guard) = get_recording_session().lock() {
        if let Some(session) = &*guard {
            if session.is_active() {
                session.mark_dirty(device_addr, size);
            }
        }
    }
}

/// Record kernel pre-launch
pub fn record_kernel_pre_launch(
    kernel_name: &str,
    module_handle: u64,
    function_handle: u64,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream_handle: u64,
    arguments: Vec<KernelArgument>,
) -> u64 {
    if let Ok(guard) = get_recording_session().lock() {
        if let Some(session) = &*guard {
            if session.is_active() {
                return session.record_pre_launch(
                    kernel_name,
                    module_handle,
                    function_handle,
                    grid_dim,
                    block_dim,
                    shared_mem_bytes,
                    stream_handle,
                    arguments,
                );
            }
        }
    }
    0
}

/// Record kernel post-launch
pub fn record_kernel_post_launch(sequence_id: u64, execution_time_ns: u64) {
    if let Ok(guard) = get_recording_session().lock() {
        if let Some(session) = &*guard {
            if session.is_active() {
                session.record_post_launch(sequence_id, execution_time_ns);
            }
        }
    }
}

/// Get recording statistics
pub fn get_recording_stats() -> Option<RecordingStats> {
    if let Ok(guard) = get_recording_session().lock() {
        if let Some(session) = &*guard {
            return Some(session.get_stats());
        }
    }
    None
}

// =============================================================================
// Replay Engine (Phase 4 placeholder)
// =============================================================================

/// Replay configuration
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Step-by-step mode (pause after each kernel)
    pub step_mode: bool,
    /// Validate outputs against recorded values
    pub validate_outputs: bool,
    /// Tolerance for floating-point comparison
    pub fp_tolerance: f64,
    /// Skip kernels that don't match filter
    pub kernel_filter: Option<String>,
    /// Maximum kernels to replay
    pub max_kernels: Option<usize>,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            step_mode: false,
            validate_outputs: true,
            fp_tolerance: 1e-6,
            kernel_filter: None,
            max_kernels: None,
        }
    }
}

/// Validation result for a single kernel
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub kernel_sequence: u64,
    pub kernel_name: String,
    pub passed: bool,
    pub mismatches: Vec<MemoryMismatch>,
    pub execution_time_recorded: u64,
    pub execution_time_replayed: u64,
}

/// Memory mismatch details
#[derive(Debug, Clone)]
pub struct MemoryMismatch {
    pub address: u64,
    pub offset: u64,
    pub size: usize,
    pub expected_checksum: u32,
    pub actual_checksum: u32,
}

/// Replay results summary
#[derive(Debug, Clone, Default)]
pub struct ReplayResults {
    pub total_kernels: u64,
    pub passed_kernels: u64,
    pub failed_kernels: u64,
    pub total_time_recorded_ns: u64,
    pub total_time_replayed_ns: u64,
}

/// Reconstructed memory state during replay
#[derive(Debug, Clone)]
pub struct ReconstructedMemory {
    /// Device address (replayed)
    pub device_addr: u64,
    /// Memory content
    pub data: Vec<u8>,
    /// Last snapshot ID applied
    pub last_snapshot_id: u64,
}

/// Compiled module for replay
#[derive(Debug)]
pub struct CompiledModule {
    /// Original module handle from trace
    pub original_handle: u64,
    /// PTX source code
    pub ptx_source: String,
    /// Compiled module handle for target backend (opaque)
    pub target_handle: Option<u64>,
    /// Kernel function handles by name
    pub kernel_handles: HashMap<String, u64>,
}

/// Replay engine for cross-backend kernel replay
pub struct ReplayEngine {
    /// Loaded trace
    trace: ReplayTrace,
    /// Target backend
    target_backend: Backend,
    /// Address remapping table (original -> replayed)
    address_map: HashMap<u64, u64>,
    /// Reverse address map (replayed -> original)
    reverse_address_map: HashMap<u64, u64>,
    /// Reconstructed memory state
    memory_state: HashMap<u64, ReconstructedMemory>,
    /// Compiled modules for target backend
    compiled_modules: HashMap<u64, CompiledModule>,
    /// Current replay position
    current_position: usize,
    /// Replay configuration
    config: ReplayConfig,
    /// Validation results
    validation_results: Vec<ValidationResult>,
    /// Whether the engine has been initialized
    initialized: bool,
    /// Step mode callback (called after each kernel)
    step_callback: Option<Box<dyn Fn(&RecordedKernelLaunch, &ValidationResult) -> bool + Send>>,
}

impl ReplayEngine {
    /// Load trace for replay
    pub fn load(trace_path: &str, target_backend: Backend) -> Result<Self, String> {
        let trace = ReplayTrace::load(trace_path)?;

        eprintln!(
            "[hetGPU Replay] Loaded trace from: {} ({} kernels, {} allocations)",
            trace_path,
            trace.kernel_launches.len(),
            trace.allocations.len()
        );
        eprintln!(
            "[hetGPU Replay] Source backend: {:?}, Target backend: {:?}",
            trace.header.source_backend, target_backend
        );

        Ok(Self {
            trace,
            target_backend,
            address_map: HashMap::new(),
            reverse_address_map: HashMap::new(),
            memory_state: HashMap::new(),
            compiled_modules: HashMap::new(),
            current_position: 0,
            config: ReplayConfig::default(),
            validation_results: Vec::new(),
            initialized: false,
            step_callback: None,
        })
    }

    /// Set replay configuration
    pub fn set_config(&mut self, config: ReplayConfig) {
        self.config = config;
    }

    /// Set step mode callback
    pub fn set_step_callback<F>(&mut self, callback: F)
    where
        F: Fn(&RecordedKernelLaunch, &ValidationResult) -> bool + Send + 'static,
    {
        self.step_callback = Some(Box::new(callback));
    }

    /// Get trace header info
    pub fn get_trace_info(&self) -> &ReplayTraceHeader {
        &self.trace.header
    }

    /// Get total number of kernels to replay
    pub fn total_kernels(&self) -> usize {
        self.trace.kernel_launches.len()
    }

    /// Get current replay position
    pub fn current_position(&self) -> usize {
        self.current_position
    }

    /// Check if replay is complete
    pub fn is_complete(&self) -> bool {
        self.current_position >= self.trace.kernel_launches.len()
    }

    /// Get validation results
    pub fn get_validation_results(&self) -> &[ValidationResult] {
        &self.validation_results
    }

    /// Get address mapping
    pub fn get_address_map(&self) -> &HashMap<u64, u64> {
        &self.address_map
    }

    /// Initialize the replay engine
    ///
    /// This prepares the target backend for replay:
    /// 1. Allocates memory for all recorded allocations
    /// 2. Compiles PTX sources for target backend
    /// 3. Reconstructs initial memory state
    pub fn initialize(&mut self) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }

        eprintln!("[hetGPU Replay] Initializing replay engine...");

        // Phase 1: Allocate memory on target backend
        self.allocate_memory()?;

        // Phase 2: Compile PTX modules for target backend
        self.compile_modules()?;

        // Phase 3: Reconstruct initial memory state
        self.reconstruct_initial_memory()?;

        self.initialized = true;
        eprintln!("[hetGPU Replay] Initialization complete");

        Ok(())
    }

    /// Allocate memory on target backend
    fn allocate_memory(&mut self) -> Result<(), String> {
        eprintln!(
            "[hetGPU Replay] Allocating {} memory regions",
            self.trace.allocations.len()
        );

        for allocation in &self.trace.allocations {
            let original_addr = allocation.device_addr;
            let size = allocation.size;

            // For now, we use a virtual address mapping
            // In a real implementation, this would call the target backend's malloc
            let new_addr = self.allocate_on_target(size, allocation.allocation_type)?;

            self.address_map.insert(original_addr, new_addr);
            self.reverse_address_map.insert(new_addr, original_addr);

            // Initialize memory state tracker
            self.memory_state.insert(
                new_addr,
                ReconstructedMemory {
                    device_addr: new_addr,
                    data: vec![0u8; size as usize],
                    last_snapshot_id: 0,
                },
            );

            eprintln!(
                "[hetGPU Replay] Mapped 0x{:x} -> 0x{:x} ({} bytes, {:?})",
                original_addr, new_addr, size, allocation.allocation_type
            );
        }

        Ok(())
    }

    /// Allocate memory on target backend (abstracted)
    fn allocate_on_target(&self, size: u64, _alloc_type: AllocationType) -> Result<u64, String> {
        // Virtual allocation for now - returns a unique address
        // In real implementation, this would call:
        // - cudaMalloc for NVIDIA
        // - hipMalloc for AMD
        // - zeMemAllocDevice for Intel
        // - tt_malloc for Tenstorrent
        static VIRTUAL_ADDR: AtomicU64 = AtomicU64::new(0x1000_0000_0000);
        let addr = VIRTUAL_ADDR.fetch_add(
            ((size + 4095) / 4096) * 4096, // Page-aligned
            Ordering::SeqCst,
        );
        Ok(addr)
    }

    /// Compile PTX modules for target backend
    fn compile_modules(&mut self) -> Result<(), String> {
        eprintln!(
            "[hetGPU Replay] Compiling {} PTX modules for {:?}",
            self.trace.ptx_sources.len(),
            self.target_backend
        );

        for (module_handle, ptx_source) in &self.trace.ptx_sources {
            let module = self.compile_ptx_for_target(*module_handle, ptx_source)?;
            self.compiled_modules.insert(*module_handle, module);
        }

        Ok(())
    }

    /// Compile PTX for target backend
    fn compile_ptx_for_target(
        &self,
        original_handle: u64,
        ptx_source: &str,
    ) -> Result<CompiledModule, String> {
        // Cross-backend PTX compilation
        // For AMD/Intel/Tenstorrent, this would need PTX-to-target translation

        let mut module = CompiledModule {
            original_handle,
            ptx_source: ptx_source.to_string(),
            target_handle: None,
            kernel_handles: HashMap::new(),
        };

        match self.target_backend {
            Backend::Nvidia => {
                // Native PTX compilation via cuModuleLoadData
                // In real implementation: call CUDA driver API
                module.target_handle = Some(original_handle);
                eprintln!(
                    "[hetGPU Replay] Module 0x{:x}: Native NVIDIA compilation",
                    original_handle
                );
            }
            Backend::Amd => {
                // PTX -> HIP translation
                // Would use ptx2hip or similar translator
                eprintln!(
                    "[hetGPU Replay] Module 0x{:x}: PTX->HIP translation (simulated)",
                    original_handle
                );
                module.target_handle = Some(original_handle | 0xA5D0_0000_0000);
            }
            Backend::Intel => {
                // PTX -> SPIR-V translation
                // Would use nvptx2spirv or similar
                eprintln!(
                    "[hetGPU Replay] Module 0x{:x}: PTX->SPIR-V translation (simulated)",
                    original_handle
                );
                module.target_handle = Some(original_handle | 0x0E1E_0000_0000);
            }
            Backend::Tenstorrent => {
                // PTX -> Tenstorrent kernel translation
                eprintln!(
                    "[hetGPU Replay] Module 0x{:x}: PTX->TT translation (simulated)",
                    original_handle
                );
                module.target_handle = Some(original_handle | 0x7700_0000_0000);
            }
            Backend::Virtual => {
                // Virtual backend - just store PTX for analysis
                eprintln!(
                    "[hetGPU Replay] Module 0x{:x}: Virtual backend (no compilation)",
                    original_handle
                );
                module.target_handle = Some(original_handle);
            }
        }

        // Extract kernel names from PTX
        module.kernel_handles = self.extract_kernel_names(&module.ptx_source);

        Ok(module)
    }

    /// Extract kernel names from PTX source
    fn extract_kernel_names(&self, ptx_source: &str) -> HashMap<String, u64> {
        let mut kernels = HashMap::new();
        let mut handle_counter = 0u64;

        // Parse PTX for .entry directives
        for line in ptx_source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(".entry") || trimmed.starts_with(".visible .entry") {
                // Extract kernel name
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                for (i, part) in parts.iter().enumerate() {
                    if *part == ".entry" && i + 1 < parts.len() {
                        let name = parts[i + 1].trim_end_matches('(');
                        kernels.insert(name.to_string(), handle_counter);
                        handle_counter += 1;
                        break;
                    }
                }
            }
        }

        kernels
    }

    /// Reconstruct initial memory state from trace
    fn reconstruct_initial_memory(&mut self) -> Result<(), String> {
        eprintln!("[hetGPU Replay] Reconstructing initial memory state");

        // Find all input memory snapshots from the first kernel launch
        // and apply them to initialize memory
        let snapshots: Vec<MemorySnapshot> = self
            .trace
            .kernel_launches
            .first()
            .map(|launch| launch.input_memory.clone())
            .unwrap_or_default();

        for snapshot in &snapshots {
            self.apply_snapshot(snapshot)?;
        }

        Ok(())
    }

    /// Apply a memory snapshot to reconstructed memory
    fn apply_snapshot(&mut self, snapshot: &MemorySnapshot) -> Result<(), String> {
        let original_addr = snapshot.device_addr;
        let new_addr = self
            .address_map
            .get(&original_addr)
            .copied()
            .ok_or_else(|| format!("No address mapping for 0x{:x}", original_addr))?;

        let memory = self
            .memory_state
            .get_mut(&new_addr)
            .ok_or_else(|| format!("No memory state for 0x{:x}", new_addr))?;

        match &snapshot.snapshot_type {
            SnapshotType::Full => {
                if let Some(data) = &snapshot.full_data {
                    if data.len() <= memory.data.len() {
                        memory.data[..data.len()].copy_from_slice(data);
                    }
                }
            }
            SnapshotType::Incremental => {
                if let Some(regions) = &snapshot.dirty_regions {
                    for region in regions {
                        let start = region.offset as usize;
                        let end = start + region.size as usize;
                        if end <= memory.data.len() && region.data.len() == region.size as usize {
                            memory.data[start..end].copy_from_slice(&region.data);
                        }
                    }
                }
            }
            SnapshotType::Reference { snapshot_id: _ } => {
                // Reference to previous snapshot - no changes needed
            }
        }

        memory.last_snapshot_id = snapshot.snapshot_id;
        Ok(())
    }

    /// Remap kernel arguments to use new addresses
    fn remap_arguments(&self, args: &[KernelArgument]) -> Vec<KernelArgument> {
        args.iter()
            .map(|arg| {
                let mut new_arg = arg.clone();
                if let Some(orig_addr) = arg.device_addr {
                    if let Some(&new_addr) = self.address_map.get(&orig_addr) {
                        new_arg.device_addr = Some(new_addr);
                    }
                }
                new_arg
            })
            .collect()
    }

    /// Replay the next kernel
    pub fn replay_next(&mut self) -> Result<Option<ValidationResult>, String> {
        if !self.initialized {
            return Err("Replay engine not initialized. Call initialize() first.".to_string());
        }

        if self.is_complete() {
            return Ok(None);
        }

        // Clone the kernel launch data to avoid borrow checker issues
        let launch = self.trace.kernel_launches[self.current_position].clone();

        // Check kernel filter
        if let Some(ref filter) = self.config.kernel_filter {
            if !launch.kernel_name.contains(filter) {
                self.current_position += 1;
                return self.replay_next(); // Skip to next
            }
        }

        eprintln!(
            "[hetGPU Replay] Replaying kernel {}/{}: {}",
            self.current_position + 1,
            self.total_kernels(),
            launch.kernel_name
        );

        // Apply input memory snapshots
        for snapshot in &launch.input_memory {
            self.apply_snapshot(snapshot)?;
        }

        // Remap arguments
        let remapped_args = self.remap_arguments(&launch.arguments);

        // Execute kernel on target backend
        let start_time = Instant::now();
        let execution_result = self.execute_kernel(&launch, &remapped_args);
        let execution_time = start_time.elapsed().as_nanos() as u64;

        // Validate outputs if configured
        let validation = if self.config.validate_outputs {
            self.validate_kernel_output(&launch, execution_time)
        } else {
            ValidationResult {
                kernel_sequence: launch.sequence_id,
                kernel_name: launch.kernel_name.clone(),
                passed: execution_result.is_ok(),
                mismatches: Vec::new(),
                execution_time_recorded: launch.execution_duration_ns.unwrap_or(0),
                execution_time_replayed: execution_time,
            }
        };

        // Store result
        self.validation_results.push(validation.clone());

        // Call step callback if in step mode
        if self.config.step_mode {
            if let Some(ref callback) = self.step_callback {
                let should_continue = callback(&launch, &validation);
                if !should_continue {
                    return Ok(Some(validation));
                }
            }
        }

        self.current_position += 1;

        // Check max kernels limit
        if let Some(max) = self.config.max_kernels {
            if self.current_position >= max {
                eprintln!("[hetGPU Replay] Reached max kernels limit: {}", max);
            }
        }

        Ok(Some(validation))
    }

    /// Execute a kernel on target backend
    fn execute_kernel(
        &self,
        launch: &RecordedKernelLaunch,
        _remapped_args: &[KernelArgument],
    ) -> Result<(), String> {
        // Get compiled module
        let _module = self
            .compiled_modules
            .get(&launch.module_handle)
            .ok_or_else(|| format!("Module 0x{:x} not compiled", launch.module_handle))?;

        // In real implementation, this would:
        // 1. Look up kernel function in compiled module
        // 2. Set up arguments with remapped addresses
        // 3. Launch kernel with grid/block dimensions
        // 4. Synchronize stream

        match self.target_backend {
            Backend::Nvidia => {
                // cuLaunchKernel(...)
                eprintln!(
                    "  [NVIDIA] Launch {}<<<({},{},{}), ({},{},{})>>>",
                    launch.kernel_name,
                    launch.grid_dim.0,
                    launch.grid_dim.1,
                    launch.grid_dim.2,
                    launch.block_dim.0,
                    launch.block_dim.1,
                    launch.block_dim.2
                );
            }
            Backend::Amd => {
                // hipLaunchKernel(...)
                eprintln!(
                    "  [AMD/HIP] Launch {}<<<({},{},{}), ({},{},{})>>>",
                    launch.kernel_name,
                    launch.grid_dim.0,
                    launch.grid_dim.1,
                    launch.grid_dim.2,
                    launch.block_dim.0,
                    launch.block_dim.1,
                    launch.block_dim.2
                );
            }
            Backend::Intel => {
                // zeCommandListAppendLaunchKernel(...)
                eprintln!(
                    "  [Intel/L0] Launch {} with {} groups",
                    launch.kernel_name,
                    launch.grid_dim.0 * launch.grid_dim.1 * launch.grid_dim.2
                );
            }
            Backend::Tenstorrent => {
                // tt_launch(...)
                eprintln!(
                    "  [Tenstorrent] Launch {} on {} cores",
                    launch.kernel_name,
                    launch.grid_dim.0 * launch.grid_dim.1 * launch.grid_dim.2
                );
            }
            Backend::Virtual => {
                // No actual execution
                eprintln!("  [Virtual] Simulated launch {}", launch.kernel_name);
            }
        }

        Ok(())
    }

    /// Validate kernel output against recorded values
    fn validate_kernel_output(
        &self,
        launch: &RecordedKernelLaunch,
        execution_time: u64,
    ) -> ValidationResult {
        let mut mismatches = Vec::new();

        // Compare output memory with recorded snapshots
        for snapshot in &launch.output_memory {
            if let Some(&new_addr) = self.address_map.get(&snapshot.device_addr) {
                if let Some(memory) = self.memory_state.get(&new_addr) {
                    // Compare checksums
                    let actual_checksum = crc32_checksum(&memory.data);
                    if actual_checksum != snapshot.checksum {
                        mismatches.push(MemoryMismatch {
                            address: snapshot.device_addr,
                            offset: 0,
                            size: memory.data.len(),
                            expected_checksum: snapshot.checksum,
                            actual_checksum,
                        });
                    }
                }
            }
        }

        let passed = mismatches.is_empty();
        if passed {
            eprintln!(
                "  ✓ Validation passed (recorded: {}ns, replayed: {}ns)",
                launch.execution_duration_ns.unwrap_or(0),
                execution_time
            );
        } else {
            eprintln!(
                "  ✗ Validation failed: {} memory mismatches",
                mismatches.len()
            );
            for mismatch in &mismatches {
                eprintln!(
                    "    - 0x{:x}: expected checksum 0x{:08x}, got 0x{:08x}",
                    mismatch.address, mismatch.expected_checksum, mismatch.actual_checksum
                );
            }
        }

        ValidationResult {
            kernel_sequence: launch.sequence_id,
            kernel_name: launch.kernel_name.clone(),
            passed,
            mismatches,
            execution_time_recorded: launch.execution_duration_ns.unwrap_or(0),
            execution_time_replayed: execution_time,
        }
    }

    /// Replay all kernels
    pub fn replay_all(&mut self) -> Result<ReplayResults, String> {
        if !self.initialized {
            self.initialize()?;
        }

        eprintln!(
            "[hetGPU Replay] Replaying all {} kernels",
            self.total_kernels()
        );

        let mut results = ReplayResults::default();
        results.total_kernels = self.total_kernels() as u64;

        while !self.is_complete() {
            if let Some(max) = self.config.max_kernels {
                if self.current_position >= max {
                    break;
                }
            }

            match self.replay_next()? {
                Some(validation) => {
                    results.total_time_recorded_ns += validation.execution_time_recorded;
                    results.total_time_replayed_ns += validation.execution_time_replayed;

                    if validation.passed {
                        results.passed_kernels += 1;
                    } else {
                        results.failed_kernels += 1;
                    }
                }
                None => break,
            }
        }

        eprintln!("[hetGPU Replay] Replay complete:");
        eprintln!(
            "  Total: {}, Passed: {}, Failed: {}",
            results.total_kernels, results.passed_kernels, results.failed_kernels
        );
        eprintln!(
            "  Recorded time: {}ms, Replayed time: {}ms",
            results.total_time_recorded_ns / 1_000_000,
            results.total_time_replayed_ns / 1_000_000
        );

        Ok(results)
    }

    /// Reset replay to beginning
    pub fn reset(&mut self) {
        self.current_position = 0;
        self.validation_results.clear();
        eprintln!("[hetGPU Replay] Reset to beginning");
    }

    /// Seek to a specific kernel position
    pub fn seek(&mut self, position: usize) -> Result<(), String> {
        if position >= self.total_kernels() {
            return Err(format!(
                "Position {} out of range (max: {})",
                position,
                self.total_kernels()
            ));
        }

        // Collect allocation info first
        let alloc_info: Vec<(u64, u64)> = self
            .trace
            .allocations
            .iter()
            .filter_map(|a| {
                self.address_map
                    .get(&a.device_addr)
                    .map(|&new_addr| (new_addr, a.size))
            })
            .collect();

        // Reset memory state
        self.memory_state.clear();
        for (new_addr, size) in alloc_info {
            self.memory_state.insert(
                new_addr,
                ReconstructedMemory {
                    device_addr: new_addr,
                    data: vec![0u8; size as usize],
                    last_snapshot_id: 0,
                },
            );
        }

        // Collect all snapshots to apply
        let all_snapshots: Vec<MemorySnapshot> = (0..=position)
            .flat_map(|i| self.trace.kernel_launches[i].input_memory.clone())
            .collect();

        // Apply snapshots
        for snapshot in &all_snapshots {
            self.apply_snapshot(snapshot)?;
        }

        self.current_position = position;
        eprintln!("[hetGPU Replay] Seeked to position {}", position);

        Ok(())
    }

    /// Get memory content at an address
    pub fn read_memory(&self, original_addr: u64) -> Option<&[u8]> {
        let new_addr = self.address_map.get(&original_addr)?;
        let memory = self.memory_state.get(new_addr)?;
        Some(&memory.data)
    }

    /// Get kernel launch info by sequence ID
    pub fn get_kernel_launch(&self, sequence_id: u64) -> Option<&RecordedKernelLaunch> {
        self.trace
            .kernel_launches
            .iter()
            .find(|l| l.sequence_id == sequence_id)
    }

    /// Compare memory at an address with a snapshot
    pub fn diff_memory(&self, original_addr: u64, snapshot: &MemorySnapshot) -> Vec<DirtyRegion> {
        let Some(new_addr) = self.address_map.get(&original_addr) else {
            return Vec::new();
        };
        let Some(memory) = self.memory_state.get(new_addr) else {
            return Vec::new();
        };

        // Get expected data from snapshot
        let expected = match &snapshot.snapshot_type {
            SnapshotType::Full => snapshot.full_data.clone().unwrap_or_default(),
            SnapshotType::Incremental => {
                // Build expected from dirty regions
                let mut data = vec![0u8; snapshot.allocation_size as usize];
                if let Some(regions) = &snapshot.dirty_regions {
                    for region in regions {
                        let start = region.offset as usize;
                        let end = (start + region.size as usize).min(data.len());
                        if region.data.len() >= (end - start) {
                            data[start..end].copy_from_slice(&region.data[..(end - start)]);
                        }
                    }
                }
                data
            }
            SnapshotType::Reference { .. } => return Vec::new(),
        };

        // Find differences
        let allocation = TrackedAllocation::new(
            original_addr,
            memory.data.len() as u64,
            AllocationType::Device,
        );
        detect_dirty_regions_static(&memory.data, &expected, &allocation)
    }
}

// =============================================================================
// Debugging Features (Phase 6)
// =============================================================================

/// Execution timeline entry for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    /// Kernel sequence ID
    pub sequence_id: u64,
    /// Kernel name
    pub kernel_name: String,
    /// Start time (nanoseconds from trace start)
    pub start_time_ns: u64,
    /// Duration (nanoseconds)
    pub duration_ns: u64,
    /// Grid dimensions
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions
    pub block_dim: (u32, u32, u32),
    /// Number of input memory regions
    pub num_inputs: usize,
    /// Number of output memory regions
    pub num_outputs: usize,
    /// Total input memory size
    pub input_bytes: u64,
    /// Total output memory size
    pub output_bytes: u64,
    /// Validation status (if replayed)
    pub validation_status: Option<bool>,
}

/// Execution timeline for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTimeline {
    /// Trace metadata
    pub source_backend: String,
    pub device_name: String,
    pub timestamp: u64,
    /// Total execution time
    pub total_time_ns: u64,
    /// Timeline entries
    pub entries: Vec<TimelineEntry>,
    /// Memory allocation timeline
    pub allocations: Vec<AllocationEvent>,
}

/// Memory allocation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    /// Event type: "alloc" or "free"
    pub event_type: String,
    /// Device address
    pub device_addr: u64,
    /// Size (0 for free)
    pub size: u64,
    /// Timestamp
    pub timestamp_ns: u64,
}

/// Memory diff report for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDiffReport {
    /// Original device address
    pub device_addr: u64,
    /// Allocation size
    pub allocation_size: u64,
    /// Number of differences
    pub num_differences: usize,
    /// Total bytes different
    pub bytes_different: u64,
    /// Difference regions
    pub diff_regions: Vec<DiffRegion>,
}

/// Single difference region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffRegion {
    /// Offset from base
    pub offset: u64,
    /// Size of difference
    pub size: u64,
    /// Expected bytes (truncated for display)
    pub expected_preview: String,
    /// Actual bytes (truncated for display)
    pub actual_preview: String,
}

/// Breakpoint configuration
#[derive(Debug, Clone)]
pub struct Breakpoint {
    /// Breakpoint type
    pub bp_type: BreakpointType,
    /// Is breakpoint enabled?
    pub enabled: bool,
    /// Hit count
    pub hit_count: u64,
}

/// Breakpoint types
#[derive(Debug, Clone)]
pub enum BreakpointType {
    /// Break at specific kernel sequence ID
    KernelSequence(u64),
    /// Break at kernel name (substring match)
    KernelName(String),
    /// Break when memory address is accessed
    MemoryAccess(u64),
    /// Break after N kernels
    KernelCount(u64),
}

/// Debug session state
pub struct DebugSession {
    /// Engine being debugged
    pub engine: ReplayEngine,
    /// Breakpoints
    pub breakpoints: Vec<Breakpoint>,
    /// Is paused?
    pub paused: bool,
    /// Last validation result
    pub last_result: Option<ValidationResult>,
}

impl DebugSession {
    /// Create new debug session
    pub fn new(engine: ReplayEngine) -> Self {
        Self {
            engine,
            breakpoints: Vec::new(),
            paused: false,
            last_result: None,
        }
    }

    /// Add a breakpoint
    pub fn add_breakpoint(&mut self, bp_type: BreakpointType) -> usize {
        let bp = Breakpoint {
            bp_type,
            enabled: true,
            hit_count: 0,
        };
        self.breakpoints.push(bp);
        self.breakpoints.len() - 1
    }

    /// Remove a breakpoint
    pub fn remove_breakpoint(&mut self, index: usize) -> bool {
        if index < self.breakpoints.len() {
            self.breakpoints.remove(index);
            true
        } else {
            false
        }
    }

    /// Step to next kernel
    pub fn step(&mut self) -> Result<Option<ValidationResult>, String> {
        self.engine.replay_next()
    }

    /// Continue until breakpoint or end
    pub fn continue_execution(&mut self) -> Result<Option<ValidationResult>, String> {
        loop {
            if self.engine.is_complete() {
                return Ok(None);
            }

            // Check breakpoints before execution
            if self.should_break() {
                self.paused = true;
                return Ok(self.last_result.clone());
            }

            match self.engine.replay_next()? {
                Some(result) => {
                    self.last_result = Some(result.clone());
                    // Check breakpoints after execution
                    if self.should_break_post(&result) {
                        self.paused = true;
                        return Ok(Some(result));
                    }
                }
                None => return Ok(None),
            }
        }
    }

    /// Check if we should break before execution
    fn should_break(&mut self) -> bool {
        let position = self.engine.current_position();
        let launch = match self.engine.trace.kernel_launches.get(position) {
            Some(l) => l,
            None => return false,
        };

        for bp in &mut self.breakpoints {
            if !bp.enabled {
                continue;
            }

            let should_break = match &bp.bp_type {
                BreakpointType::KernelSequence(seq) => launch.sequence_id == *seq,
                BreakpointType::KernelName(name) => launch.kernel_name.contains(name),
                BreakpointType::KernelCount(count) => position as u64 >= *count,
                BreakpointType::MemoryAccess(addr) => launch
                    .arguments
                    .iter()
                    .any(|arg| arg.device_addr.map_or(false, |a| a == *addr)),
            };

            if should_break {
                bp.hit_count += 1;
                return true;
            }
        }

        false
    }

    /// Check if we should break after execution
    fn should_break_post(&self, result: &ValidationResult) -> bool {
        // Break on validation failure
        !result.passed
    }

    /// Get current kernel info
    pub fn current_kernel(&self) -> Option<&RecordedKernelLaunch> {
        self.engine
            .trace
            .kernel_launches
            .get(self.engine.current_position())
    }

    /// Print memory at address
    pub fn print_memory(&self, original_addr: u64, offset: usize, len: usize) -> Option<String> {
        let data = self.engine.read_memory(original_addr)?;
        let end = (offset + len).min(data.len());
        if offset >= data.len() {
            return None;
        }

        let slice = &data[offset..end];
        let mut output = format!(
            "Memory at 0x{:x}+0x{:x} ({} bytes):\n",
            original_addr,
            offset,
            end - offset
        );

        // Hex dump
        for (i, chunk) in slice.chunks(16).enumerate() {
            output.push_str(&format!("{:08x}  ", offset + i * 16));
            for (j, byte) in chunk.iter().enumerate() {
                output.push_str(&format!("{:02x} ", byte));
                if j == 7 {
                    output.push(' ');
                }
            }
            // ASCII representation
            output.push_str(" |");
            for byte in chunk {
                let c = if *byte >= 0x20 && *byte < 0x7f {
                    *byte as char
                } else {
                    '.'
                };
                output.push(c);
            }
            output.push_str("|\n");
        }

        Some(output)
    }
}

impl ReplayEngine {
    /// Generate execution timeline from trace
    pub fn generate_timeline(&self) -> ExecutionTimeline {
        let mut entries = Vec::new();

        for launch in &self.trace.kernel_launches {
            let input_bytes: u64 = launch.input_memory.iter().map(|s| s.allocation_size).sum();
            let output_bytes: u64 = launch.output_memory.iter().map(|s| s.allocation_size).sum();

            // Find validation status if available
            let validation_status = self
                .validation_results
                .iter()
                .find(|r| r.kernel_sequence == launch.sequence_id)
                .map(|r| r.passed);

            entries.push(TimelineEntry {
                sequence_id: launch.sequence_id,
                kernel_name: launch.kernel_name.clone(),
                start_time_ns: launch.timestamp_ns,
                duration_ns: launch.execution_duration_ns.unwrap_or(0),
                grid_dim: launch.grid_dim,
                block_dim: launch.block_dim,
                num_inputs: launch.input_memory.len(),
                num_outputs: launch.output_memory.len(),
                input_bytes,
                output_bytes,
                validation_status,
            });
        }

        // Build allocation events
        let allocations: Vec<AllocationEvent> = self
            .trace
            .allocations
            .iter()
            .map(|a| AllocationEvent {
                event_type: "alloc".to_string(),
                device_addr: a.device_addr,
                size: a.size,
                timestamp_ns: a.allocation_time,
            })
            .collect();

        let total_time_ns = entries
            .iter()
            .map(|e| e.start_time_ns + e.duration_ns)
            .max()
            .unwrap_or(0);

        ExecutionTimeline {
            source_backend: format!("{:?}", self.trace.header.source_backend),
            device_name: self.trace.header.device_name.clone(),
            timestamp: self.trace.header.timestamp,
            total_time_ns,
            entries,
            allocations,
        }
    }

    /// Export timeline to JSON file
    pub fn export_timeline_json(&self, path: &str) -> Result<(), String> {
        let timeline = self.generate_timeline();
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &timeline)
            .map_err(|e| format!("Failed to write JSON: {}", e))?;
        eprintln!("[hetGPU Replay] Exported timeline to: {}", path);
        Ok(())
    }

    /// Generate memory diff report for a specific address
    pub fn generate_memory_diff_report(
        &self,
        original_addr: u64,
        expected: &[u8],
    ) -> Option<MemoryDiffReport> {
        let actual = self.read_memory(original_addr)?;

        let allocation =
            TrackedAllocation::new(original_addr, actual.len() as u64, AllocationType::Device);
        let dirty_regions = detect_dirty_regions_static(actual, expected, &allocation);

        let mut diff_regions = Vec::new();
        let mut bytes_different = 0u64;

        for region in &dirty_regions {
            bytes_different += region.size;

            // Create preview strings (first 16 bytes)
            let preview_len = (region.size as usize).min(16);
            let offset = region.offset as usize;

            let expected_preview = if offset + preview_len <= expected.len() {
                format!("{:02x?}", &expected[offset..offset + preview_len])
            } else {
                "N/A".to_string()
            };

            let actual_preview = if offset + preview_len <= actual.len() {
                format!("{:02x?}", &actual[offset..offset + preview_len])
            } else {
                "N/A".to_string()
            };

            diff_regions.push(DiffRegion {
                offset: region.offset,
                size: region.size,
                expected_preview,
                actual_preview,
            });
        }

        Some(MemoryDiffReport {
            device_addr: original_addr,
            allocation_size: actual.len() as u64,
            num_differences: diff_regions.len(),
            bytes_different,
            diff_regions,
        })
    }

    /// Print human-readable summary
    pub fn print_summary(&self) {
        eprintln!("\n=== hetGPU Replay Summary ===");
        eprintln!("Source Backend: {:?}", self.trace.header.source_backend);
        eprintln!("Target Backend: {:?}", self.target_backend);
        eprintln!("Device Name: {}", self.trace.header.device_name);
        eprintln!("Total Kernels: {}", self.total_kernels());
        eprintln!("Current Position: {}", self.current_position);
        eprintln!("Memory Allocations: {}", self.trace.allocations.len());
        eprintln!("PTX Modules: {}", self.trace.ptx_sources.len());

        if !self.validation_results.is_empty() {
            let passed = self.validation_results.iter().filter(|r| r.passed).count();
            let failed = self.validation_results.len() - passed;
            eprintln!("\n=== Validation Results ===");
            eprintln!("Passed: {}", passed);
            eprintln!("Failed: {}", failed);

            if failed > 0 {
                eprintln!("\nFailed Kernels:");
                for result in self.validation_results.iter().filter(|r| !r.passed) {
                    eprintln!(
                        "  - {}: {} mismatches",
                        result.kernel_name,
                        result.mismatches.len()
                    );
                }
            }
        }
        eprintln!("=============================\n");
    }

    /// Get trace for direct access (for debugging)
    pub fn get_trace(&self) -> &ReplayTrace {
        &self.trace
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracked_allocation() {
        let mut alloc = TrackedAllocation::new(0x1000, 8192, AllocationType::Device);
        assert_eq!(alloc.num_pages(), 2);

        // Test dirty tracking
        alloc.mark_page_dirty(0);
        assert!(alloc.is_page_dirty(0));
        assert!(!alloc.is_page_dirty(1));

        alloc.mark_all_dirty();
        assert!(alloc.is_page_dirty(0));
        assert!(alloc.is_page_dirty(1));

        alloc.clear_dirty_bits();
        assert!(!alloc.is_page_dirty(0));
        assert!(!alloc.is_page_dirty(1));
    }

    #[test]
    fn test_memory_snapshot_full() {
        let data = vec![1u8, 2, 3, 4, 5];
        let snapshot = MemorySnapshot::full(1, 0x1000, data.clone());

        assert_eq!(snapshot.snapshot_id, 1);
        assert_eq!(snapshot.device_addr, 0x1000);
        assert_eq!(snapshot.allocation_size, 5);
        assert!(matches!(snapshot.snapshot_type, SnapshotType::Full));
        assert_eq!(snapshot.full_data, Some(data));
    }

    #[test]
    fn test_dirty_page_tracker() {
        let mut tracker = DirtyPageTracker::new(DirtyTrackingMethod::Checksum);

        tracker.track_allocation(0x1000, 8192, AllocationType::Device);
        assert!(tracker.is_tracked(0x1000));

        tracker.mark_dirty(0x1000, 100);

        tracker.untrack_allocation(0x1000);
        assert!(!tracker.is_tracked(0x1000));
    }

    #[test]
    fn test_crc32_checksum() {
        let data = b"Hello, World!";
        let crc = crc32_checksum(data);
        assert_ne!(crc, 0);

        // Same data should produce same checksum
        let crc2 = crc32_checksum(data);
        assert_eq!(crc, crc2);

        // Different data should produce different checksum
        let crc3 = crc32_checksum(b"Different data");
        assert_ne!(crc, crc3);
    }

    #[test]
    fn test_replay_trace_serialization() {
        let trace = ReplayTrace::new(Backend::Nvidia, "Test GPU");

        // Save to temp file
        let temp_path = "/tmp/test_trace.hetr";
        trace.save(temp_path).unwrap();

        // Load back
        let loaded = ReplayTrace::load(temp_path).unwrap();
        assert_eq!(loaded.header.magic, REPLAY_TRACE_MAGIC);
        assert_eq!(loaded.header.device_name, "Test GPU");

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }
}
