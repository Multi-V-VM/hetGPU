//! C FFI Interface for hetGPU Record/Replay System
//!
//! This module provides C-compatible functions for external tools to:
//! - Start/stop recording sessions
//! - Load and replay traces
//! - Query recording/replay statistics
//! - Export debugging data

use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

use super::replay::{
    self, AllocationType, Backend, CompressionType, DirtyTrackingMethod, RecordingConfig,
    ReplayConfig, ReplayEngine, ReplayTrace,
};

// =============================================================================
// Opaque Handle Types
// =============================================================================

/// Opaque handle for recording session
pub type HetGpuRecordingHandle = *mut c_void;

/// Opaque handle for replay session
pub type HetGpuReplayHandle = *mut c_void;

// =============================================================================
// C-Compatible Structures
// =============================================================================

/// Recording configuration (C-compatible)
#[repr(C)]
pub struct CRecordingConfig {
    /// Enable dirty tracking (1 = true, 0 = false)
    pub dirty_tracking: i32,
    /// Dirty tracking method: 0=FullDump, 1=SoftwareShadow, 2=Checksum
    pub dirty_method: i32,
    /// Capture input memory (1 = true, 0 = false)
    pub capture_inputs: i32,
    /// Capture output memory (1 = true, 0 = false)
    pub capture_outputs: i32,
    /// Maximum memory to capture per kernel (0 = unlimited)
    pub max_memory_per_kernel: u64,
    /// Compression: 0=None, 1=Zstd, 2=Lz4
    pub compression: i32,
}

impl Default for CRecordingConfig {
    fn default() -> Self {
        Self {
            dirty_tracking: 1,
            dirty_method: 1, // SoftwareShadow
            capture_inputs: 1,
            capture_outputs: 1,
            max_memory_per_kernel: 0,
            compression: 0,
        }
    }
}

impl From<CRecordingConfig> for RecordingConfig {
    fn from(c: CRecordingConfig) -> Self {
        RecordingConfig {
            dirty_tracking: c.dirty_tracking != 0,
            dirty_method: match c.dirty_method {
                0 => DirtyTrackingMethod::FullDump,
                1 => DirtyTrackingMethod::SoftwareShadow,
                2 => DirtyTrackingMethod::Checksum,
                _ => DirtyTrackingMethod::SoftwareShadow,
            },
            capture_inputs: c.capture_inputs != 0,
            capture_outputs: c.capture_outputs != 0,
            max_memory_per_kernel: c.max_memory_per_kernel as usize,
            compression: match c.compression {
                1 => CompressionType::Zstd,
                2 => CompressionType::Lz4,
                _ => CompressionType::None,
            },
            ..Default::default()
        }
    }
}

/// Recording statistics (C-compatible)
#[repr(C)]
pub struct CRecordingStats {
    /// Number of kernels recorded
    pub kernels_recorded: u64,
    /// Number of memory allocations tracked
    pub allocations_tracked: u64,
    /// Recording duration in nanoseconds
    pub recording_duration_ns: u64,
    /// Whether recording is active (1 = true, 0 = false)
    pub is_active: i32,
}

/// Replay configuration (C-compatible)
#[repr(C)]
pub struct CReplayConfig {
    /// Step mode (pause after each kernel)
    pub step_mode: i32,
    /// Validate outputs
    pub validate_outputs: i32,
    /// Floating-point tolerance (as bits, reinterpret as f64)
    pub fp_tolerance_bits: u64,
    /// Maximum kernels to replay (0 = unlimited)
    pub max_kernels: u64,
}

impl Default for CReplayConfig {
    fn default() -> Self {
        Self {
            step_mode: 0,
            validate_outputs: 1,
            fp_tolerance_bits: 1e-6f64.to_bits(),
            max_kernels: 0,
        }
    }
}

/// Validation result for a single kernel (C-compatible)
#[repr(C)]
pub struct CValidationResult {
    /// Kernel sequence ID
    pub kernel_sequence: u64,
    /// Whether validation passed (1 = true, 0 = false)
    pub passed: i32,
    /// Number of memory mismatches
    pub num_mismatches: u32,
    /// Recorded execution time in nanoseconds
    pub execution_time_recorded_ns: u64,
    /// Replayed execution time in nanoseconds
    pub execution_time_replayed_ns: u64,
}

/// Replay results summary (C-compatible)
#[repr(C)]
pub struct CReplayResults {
    /// Total kernels replayed
    pub total_kernels: u64,
    /// Kernels that passed validation
    pub passed_kernels: u64,
    /// Kernels that failed validation
    pub failed_kernels: u64,
    /// Total recorded execution time in nanoseconds
    pub total_time_recorded_ns: u64,
    /// Total replayed execution time in nanoseconds
    pub total_time_replayed_ns: u64,
}

/// Memory diff information (C-compatible)
#[repr(C)]
pub struct CMemoryDiff {
    /// Base address
    pub base_addr: u64,
    /// Size in bytes
    pub size: u64,
    /// Number of differences found
    pub num_differences: u32,
    /// Offset of first difference
    pub first_diff_offset: u64,
}

/// Trace header information (C-compatible)
#[repr(C)]
pub struct CTraceInfo {
    /// Trace version
    pub version: u32,
    /// Timestamp when trace was created
    pub timestamp: u64,
    /// Source backend: 0=Nvidia, 1=Amd, 2=Intel, 3=Tenstorrent, 4=Virtual
    pub source_backend: i32,
    /// Total number of kernel launches
    pub total_kernels: u64,
    /// Total number of memory operations
    pub total_memory_ops: u64,
}

// =============================================================================
// Recording API
// =============================================================================

/// Start a recording session
///
/// # Arguments
/// * `output_path` - Path where trace file will be saved (null-terminated C string)
/// * `config` - Recording configuration (can be null for defaults)
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_recording_start(
    output_path: *const c_char,
    config: *const CRecordingConfig,
) -> i32 {
    let path = if output_path.is_null() {
        "/tmp/hetgpu_traces/trace.hetr".to_string()
    } else {
        match unsafe { CStr::from_ptr(output_path) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return -1,
        }
    };

    // Set environment variable for the recording path
    std::env::set_var("HETGPU_RECORD", "1");
    std::env::set_var("HETGPU_RECORD_OUTPUT", &path);

    if !config.is_null() {
        let c_config = unsafe { &*config };
        std::env::set_var(
            "HETGPU_RECORD_INPUTS",
            if c_config.capture_inputs != 0 {
                "1"
            } else {
                "0"
            },
        );
        std::env::set_var(
            "HETGPU_RECORD_OUTPUTS",
            if c_config.capture_outputs != 0 {
                "1"
            } else {
                "0"
            },
        );
    }

    // Detect backend (default to NVIDIA for now)
    let backend = Backend::Nvidia;
    let device_name = "Unknown GPU";

    match replay::start_recording(backend, device_name) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

/// Stop recording and save trace file
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_recording_stop() -> i32 {
    match replay::stop_recording() {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Check if recording is active
///
/// # Returns
/// 1 if recording is active, 0 otherwise
#[no_mangle]
pub extern "C" fn hetgpu_recording_is_active() -> i32 {
    if replay::is_recording_active() {
        1
    } else {
        0
    }
}

/// Get recording statistics
///
/// # Arguments
/// * `stats` - Output structure for statistics
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_recording_get_stats(stats: *mut CRecordingStats) -> i32 {
    if stats.is_null() {
        return -1;
    }

    match replay::get_recording_stats() {
        Some(s) => {
            unsafe {
                (*stats).kernels_recorded = s.kernels_recorded;
                (*stats).allocations_tracked = s.allocations_tracked;
                (*stats).recording_duration_ns = s.recording_duration_ns;
                (*stats).is_active = if s.is_active { 1 } else { 0 };
            }
            0
        }
        None => -2,
    }
}

// =============================================================================
// Replay API
// =============================================================================

/// Load a trace file for replay
///
/// # Arguments
/// * `trace_path` - Path to trace file (null-terminated C string)
/// * `target_backend` - Target backend: 0=Nvidia, 1=Amd, 2=Intel, 3=Tenstorrent, 4=Virtual
///
/// # Returns
/// Handle to replay session, or null on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_load(
    trace_path: *const c_char,
    target_backend: i32,
) -> HetGpuReplayHandle {
    if trace_path.is_null() {
        return ptr::null_mut();
    }

    let path = match unsafe { CStr::from_ptr(trace_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let backend = match target_backend {
        0 => Backend::Nvidia,
        1 => Backend::Amd,
        2 => Backend::Intel,
        3 => Backend::Tenstorrent,
        _ => Backend::Virtual,
    };

    match ReplayEngine::load(path, backend) {
        Ok(engine) => Box::into_raw(Box::new(engine)) as HetGpuReplayHandle,
        Err(_) => ptr::null_mut(),
    }
}

/// Get trace information
///
/// # Arguments
/// * `handle` - Replay session handle
/// * `info` - Output structure for trace info
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_get_trace_info(
    handle: HetGpuReplayHandle,
    info: *mut CTraceInfo,
) -> i32 {
    if handle.is_null() || info.is_null() {
        return -1;
    }

    let engine = unsafe { &*(handle as *const ReplayEngine) };
    let header = engine.get_trace_info();

    unsafe {
        (*info).version = header.version;
        (*info).timestamp = header.timestamp;
        (*info).source_backend = match header.source_backend {
            Backend::Nvidia => 0,
            Backend::Amd => 1,
            Backend::Intel => 2,
            Backend::Tenstorrent => 3,
            Backend::Virtual => 4,
        };
        (*info).total_kernels = header.total_kernels;
        (*info).total_memory_ops = header.total_memory_ops;
    }

    0
}

/// Get total number of kernels in trace
///
/// # Arguments
/// * `handle` - Replay session handle
///
/// # Returns
/// Number of kernels, or 0 on error
#[no_mangle]
pub extern "C" fn hetgpu_replay_get_kernel_count(handle: HetGpuReplayHandle) -> u64 {
    if handle.is_null() {
        return 0;
    }

    let engine = unsafe { &*(handle as *const ReplayEngine) };
    engine.total_kernels() as u64
}

/// Get current replay position
///
/// # Arguments
/// * `handle` - Replay session handle
///
/// # Returns
/// Current position (0-based index), or 0 on error
#[no_mangle]
pub extern "C" fn hetgpu_replay_get_position(handle: HetGpuReplayHandle) -> u64 {
    if handle.is_null() {
        return 0;
    }

    let engine = unsafe { &*(handle as *const ReplayEngine) };
    engine.current_position() as u64
}

/// Check if replay is complete
///
/// # Arguments
/// * `handle` - Replay session handle
///
/// # Returns
/// 1 if complete, 0 if not complete or on error
#[no_mangle]
pub extern "C" fn hetgpu_replay_is_complete(handle: HetGpuReplayHandle) -> i32 {
    if handle.is_null() {
        return 0;
    }

    let engine = unsafe { &*(handle as *const ReplayEngine) };
    if engine.is_complete() {
        1
    } else {
        0
    }
}

/// Free replay session
///
/// # Arguments
/// * `handle` - Replay session handle to free
#[no_mangle]
pub extern "C" fn hetgpu_replay_free(handle: HetGpuReplayHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle as *mut ReplayEngine));
        }
    }
}

/// Initialize replay engine
///
/// This prepares the target backend for replay:
/// - Allocates memory for all recorded allocations
/// - Compiles PTX sources for target backend
/// - Reconstructs initial memory state
///
/// # Arguments
/// * `handle` - Replay session handle
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_initialize(handle: HetGpuReplayHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }

    let engine = unsafe { &mut *(handle as *mut ReplayEngine) };
    match engine.initialize() {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

/// Set replay configuration
///
/// # Arguments
/// * `handle` - Replay session handle
/// * `config` - Replay configuration
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_set_config(
    handle: HetGpuReplayHandle,
    config: *const CReplayConfig,
) -> i32 {
    if handle.is_null() || config.is_null() {
        return -1;
    }

    let engine = unsafe { &mut *(handle as *mut ReplayEngine) };
    let c_config = unsafe { &*config };

    engine.set_config(replay::ReplayConfig {
        step_mode: c_config.step_mode != 0,
        validate_outputs: c_config.validate_outputs != 0,
        fp_tolerance: f64::from_bits(c_config.fp_tolerance_bits),
        kernel_filter: None,
        max_kernels: if c_config.max_kernels == 0 {
            None
        } else {
            Some(c_config.max_kernels as usize)
        },
    });

    0
}

/// Replay the next kernel
///
/// # Arguments
/// * `handle` - Replay session handle
/// * `result` - Output structure for validation result (can be null)
///
/// # Returns
/// 1 if a kernel was replayed
/// 0 if replay is complete
/// Negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_next(
    handle: HetGpuReplayHandle,
    result: *mut CValidationResult,
) -> i32 {
    if handle.is_null() {
        return -1;
    }

    let engine = unsafe { &mut *(handle as *mut ReplayEngine) };
    match engine.replay_next() {
        Ok(Some(validation)) => {
            if !result.is_null() {
                unsafe {
                    (*result).kernel_sequence = validation.kernel_sequence;
                    (*result).passed = if validation.passed { 1 } else { 0 };
                    (*result).num_mismatches = validation.mismatches.len() as u32;
                    (*result).execution_time_recorded_ns = validation.execution_time_recorded;
                    (*result).execution_time_replayed_ns = validation.execution_time_replayed;
                }
            }
            1
        }
        Ok(None) => 0,
        Err(_) => -2,
    }
}

/// Replay all kernels
///
/// # Arguments
/// * `handle` - Replay session handle
/// * `results` - Output structure for replay results
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_all(
    handle: HetGpuReplayHandle,
    results: *mut CReplayResults,
) -> i32 {
    if handle.is_null() {
        return -1;
    }

    let engine = unsafe { &mut *(handle as *mut ReplayEngine) };
    match engine.replay_all() {
        Ok(replay_results) => {
            if !results.is_null() {
                unsafe {
                    (*results).total_kernels = replay_results.total_kernels;
                    (*results).passed_kernels = replay_results.passed_kernels;
                    (*results).failed_kernels = replay_results.failed_kernels;
                    (*results).total_time_recorded_ns = replay_results.total_time_recorded_ns;
                    (*results).total_time_replayed_ns = replay_results.total_time_replayed_ns;
                }
            }
            0
        }
        Err(_) => -2,
    }
}

/// Reset replay to beginning
///
/// # Arguments
/// * `handle` - Replay session handle
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_reset(handle: HetGpuReplayHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }

    let engine = unsafe { &mut *(handle as *mut ReplayEngine) };
    engine.reset();
    0
}

/// Seek to a specific kernel position
///
/// # Arguments
/// * `handle` - Replay session handle
/// * `position` - Target position (0-based)
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_seek(handle: HetGpuReplayHandle, position: u64) -> i32 {
    if handle.is_null() {
        return -1;
    }

    let engine = unsafe { &mut *(handle as *mut ReplayEngine) };
    match engine.seek(position as usize) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

/// Read memory content at an address
///
/// # Arguments
/// * `handle` - Replay session handle
/// * `original_addr` - Original device address from trace
/// * `buffer` - Output buffer
/// * `buffer_size` - Size of output buffer
///
/// # Returns
/// Number of bytes copied, or negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_read_memory(
    handle: HetGpuReplayHandle,
    original_addr: u64,
    buffer: *mut u8,
    buffer_size: u64,
) -> i64 {
    if handle.is_null() || buffer.is_null() {
        return -1;
    }

    let engine = unsafe { &*(handle as *const ReplayEngine) };
    match engine.read_memory(original_addr) {
        Some(data) => {
            let copy_len = data.len().min(buffer_size as usize);
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buffer, copy_len);
            }
            copy_len as i64
        }
        None => -2,
    }
}

/// Enable step mode for debugging
///
/// # Arguments
/// * `handle` - Replay session handle
/// * `enabled` - 1 to enable, 0 to disable
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_set_step_mode(handle: HetGpuReplayHandle, enabled: i32) -> i32 {
    if handle.is_null() {
        return -1;
    }

    let engine = unsafe { &mut *(handle as *mut ReplayEngine) };
    let mut config = replay::ReplayConfig::default();
    config.step_mode = enabled != 0;
    engine.set_config(config);
    0
}

/// Export execution timeline to JSON file
///
/// # Arguments
/// * `handle` - Replay session handle
/// * `output_path` - Path for JSON output file
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_export_timeline_json(
    handle: HetGpuReplayHandle,
    output_path: *const c_char,
) -> i32 {
    if handle.is_null() || output_path.is_null() {
        return -1;
    }

    let path = match unsafe { CStr::from_ptr(output_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let engine = unsafe { &*(handle as *const ReplayEngine) };
    match engine.export_timeline_json(path) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

/// Print replay summary
///
/// # Arguments
/// * `handle` - Replay session handle
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn hetgpu_replay_print_summary(handle: HetGpuReplayHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }

    let engine = unsafe { &*(handle as *const ReplayEngine) };
    engine.print_summary();
    0
}

/// Get validation results count
///
/// # Arguments
/// * `handle` - Replay session handle
///
/// # Returns
/// Number of validation results, or 0 on error
#[no_mangle]
pub extern "C" fn hetgpu_replay_get_validation_count(handle: HetGpuReplayHandle) -> u64 {
    if handle.is_null() {
        return 0;
    }

    let engine = unsafe { &*(handle as *const ReplayEngine) };
    engine.get_validation_results().len() as u64
}

/// Get number of passed validations
///
/// # Arguments
/// * `handle` - Replay session handle
///
/// # Returns
/// Number of passed validations
#[no_mangle]
pub extern "C" fn hetgpu_replay_get_passed_count(handle: HetGpuReplayHandle) -> u64 {
    if handle.is_null() {
        return 0;
    }

    let engine = unsafe { &*(handle as *const ReplayEngine) };
    engine
        .get_validation_results()
        .iter()
        .filter(|r| r.passed)
        .count() as u64
}

/// Get number of failed validations
///
/// # Arguments
/// * `handle` - Replay session handle
///
/// # Returns
/// Number of failed validations
#[no_mangle]
pub extern "C" fn hetgpu_replay_get_failed_count(handle: HetGpuReplayHandle) -> u64 {
    if handle.is_null() {
        return 0;
    }

    let engine = unsafe { &*(handle as *const ReplayEngine) };
    engine
        .get_validation_results()
        .iter()
        .filter(|r| !r.passed)
        .count() as u64
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get version string for the replay system
///
/// # Returns
/// Pointer to static null-terminated version string
#[no_mangle]
pub extern "C" fn hetgpu_replay_get_version() -> *const c_char {
    static VERSION: &[u8] = b"hetGPU Replay v1.0.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Check if replay feature is available
///
/// # Returns
/// 1 if available, 0 if not
#[no_mangle]
pub extern "C" fn hetgpu_replay_is_available() -> i32 {
    1
}

// =============================================================================
// Environment Variable Control
// =============================================================================

/// Enable recording via environment variable
/// This is an alternative to calling hetgpu_recording_start()
///
/// # Arguments
/// * `output_path` - Path for trace output (can be null for default)
///
/// # Returns
/// 0 on success
#[no_mangle]
pub extern "C" fn hetgpu_replay_enable_recording_env(output_path: *const c_char) -> i32 {
    std::env::set_var("HETGPU_RECORD", "1");

    if !output_path.is_null() {
        if let Ok(path) = unsafe { CStr::from_ptr(output_path) }.to_str() {
            std::env::set_var("HETGPU_RECORD_OUTPUT", path);
        }
    }

    0
}

/// Disable recording via environment variable
///
/// # Returns
/// 0 on success
#[no_mangle]
pub extern "C" fn hetgpu_replay_disable_recording_env() -> i32 {
    std::env::set_var("HETGPU_RECORD", "0");
    0
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recording_config_default() {
        let config = CRecordingConfig::default();
        assert_eq!(config.dirty_tracking, 1);
        assert_eq!(config.dirty_method, 1);
        assert_eq!(config.capture_inputs, 1);
        assert_eq!(config.capture_outputs, 1);
    }

    #[test]
    fn test_version() {
        let version = hetgpu_replay_get_version();
        assert!(!version.is_null());
        let version_str = unsafe { CStr::from_ptr(version) }.to_str().unwrap();
        assert!(version_str.contains("hetGPU"));
    }

    #[test]
    fn test_is_available() {
        assert_eq!(hetgpu_replay_is_available(), 1);
    }
}
