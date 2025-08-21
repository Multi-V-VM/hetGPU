use std::sync::{Arc, Mutex, Once};
use std::ffi::{c_void, c_int, c_char};
use std::ptr;
use std::collections::HashMap;
use super::nccl_fault_tolerance::*;

// Global fault tolerance manager instance
static mut FAULT_TOLERANCE_MANAGER: Option<Arc<Mutex<NcclFaultToleranceManager>>> = None;
static INIT: Once = Once::new();

// Initialize the fault tolerance manager
pub fn initialize_fault_tolerance() {
    INIT.call_once(|| {
        unsafe {
            let config = FaultToleranceConfig::default();
            let mut manager = NcclFaultToleranceManager::new(config);
            manager.start_health_monitoring();
            FAULT_TOLERANCE_MANAGER = Some(Arc::new(Mutex::new(manager)));
        }
    });
}

// Get the fault tolerance manager
fn get_manager() -> Option<Arc<Mutex<NcclFaultToleranceManager>>> {
    unsafe { FAULT_TOLERANCE_MANAGER.clone() }
}

// NCCL data types for wrapping
type ncclDataType_t = c_int;
type ncclRedOp_t = c_int;

// Function pointers for original NCCL functions
struct NcclOriginalFunctions {
    ncclCommInitRank: Option<unsafe extern "C" fn(*mut ncclComm_t, c_int, ncclUniqueId, c_int) -> ncclResult_t>,
    ncclCommInitAll: Option<unsafe extern "C" fn(*mut ncclComm_t, c_int, *const c_int) -> ncclResult_t>,
    ncclCommDestroy: Option<unsafe extern "C" fn(ncclComm_t) -> ncclResult_t>,
    ncclAllReduce: Option<unsafe extern "C" fn(*const c_void, *mut c_void, usize, ncclDataType_t, ncclRedOp_t, ncclComm_t, *mut c_void) -> ncclResult_t>,
    ncclBroadcast: Option<unsafe extern "C" fn(*const c_void, *mut c_void, usize, ncclDataType_t, c_int, ncclComm_t, *mut c_void) -> ncclResult_t>,
    ncclReduce: Option<unsafe extern "C" fn(*const c_void, *mut c_void, usize, ncclDataType_t, ncclRedOp_t, c_int, ncclComm_t, *mut c_void) -> ncclResult_t>,
    ncclAllGather: Option<unsafe extern "C" fn(*const c_void, *mut c_void, usize, ncclDataType_t, ncclComm_t, *mut c_void) -> ncclResult_t>,
    ncclReduceScatter: Option<unsafe extern "C" fn(*const c_void, *mut c_void, usize, ncclDataType_t, ncclRedOp_t, ncclComm_t, *mut c_void) -> ncclResult_t>,
    ncclGetErrorString: Option<unsafe extern "C" fn(ncclResult_t) -> *const c_char>,
}

static mut ORIGINAL_FUNCTIONS: Option<NcclOriginalFunctions> = None;

// Load original NCCL functions
pub fn load_original_functions() {
    unsafe {
        // Try to load NCCL library - first try without RTLD_NOLOAD to actually load it
        let mut lib = libc::dlopen(b"libnccl.so.2\0".as_ptr() as *const c_char, libc::RTLD_LAZY);
        if lib.is_null() {
            // Try alternative name
            lib = libc::dlopen(b"libnccl.so\0".as_ptr() as *const c_char, libc::RTLD_LAZY);
            if lib.is_null() {
                eprintln!("[NCCL Hook] Warning: Could not load original NCCL library");
                return;
            }
        }
        
        eprintln!("[NCCL Hook] Successfully loaded NCCL library");
        
        ORIGINAL_FUNCTIONS = Some(NcclOriginalFunctions {
            ncclCommInitRank: std::mem::transmute(libc::dlsym(lib, b"ncclCommInitRank\0".as_ptr() as *const c_char)),
            ncclCommInitAll: std::mem::transmute(libc::dlsym(lib, b"ncclCommInitAll\0".as_ptr() as *const c_char)),
            ncclCommDestroy: std::mem::transmute(libc::dlsym(lib, b"ncclCommDestroy\0".as_ptr() as *const c_char)),
            ncclAllReduce: std::mem::transmute(libc::dlsym(lib, b"ncclAllReduce\0".as_ptr() as *const c_char)),
            ncclBroadcast: std::mem::transmute(libc::dlsym(lib, b"ncclBroadcast\0".as_ptr() as *const c_char)),
            ncclReduce: std::mem::transmute(libc::dlsym(lib, b"ncclReduce\0".as_ptr() as *const c_char)),
            ncclAllGather: std::mem::transmute(libc::dlsym(lib, b"ncclAllGather\0".as_ptr() as *const c_char)),
            ncclReduceScatter: std::mem::transmute(libc::dlsym(lib, b"ncclReduceScatter\0".as_ptr() as *const c_char)),
            ncclGetErrorString: std::mem::transmute(libc::dlsym(lib, b"ncclGetErrorString\0".as_ptr() as *const c_char)),
        });
        
        // Don't close the library to keep symbols available
        // libc::dlclose(lib);
    }
}

// Wrapped NCCL functions with fault tolerance

// Commented out to avoid duplicate symbol with nccl_fault_tolerant.rs
// Use nccl_fault_tolerant.rs version instead
pub unsafe extern "C" fn ncclCommInitRank_hooks(
    comm: *mut ncclComm_t,
    nranks: c_int,
    commId: ncclUniqueId,
    rank: c_int,
) -> ncclResult_t {
    eprintln!("[NCCL Hook] ncclCommInitRank called with nranks={}, rank={}", nranks, rank);
    
    initialize_fault_tolerance();
    load_original_functions();  // Ensure original functions are loaded
    
    // Call original function
    let result = if let Some(ref funcs) = ORIGINAL_FUNCTIONS {
        if let Some(orig_fn) = funcs.ncclCommInitRank {
            eprintln!("[NCCL Hook] Calling original ncclCommInitRank");
            orig_fn(comm, nranks, commId, rank)
        } else {
            eprintln!("[NCCL Hook] Original ncclCommInitRank not found");
            ncclResult_t::ncclInternalError
        }
    } else {
        eprintln!("[NCCL Hook] No original functions loaded, using fallback");
        // Fallback implementation
        *comm = Box::into_raw(Box::new(ncclComm { _private: [] }));
        ncclResult_t::ncclSuccess
    };
    
    // Register with fault tolerance manager
    if result == ncclResult_t::ncclSuccess {
        if let Some(manager) = get_manager() {
            let mut mgr = manager.lock().unwrap();
            let ranks: Vec<i32> = (0..nranks).collect();
            mgr.register_communicator(*comm, commId, ranks);
            mgr.register_gpu(rank, rank);
        }
    }
    
    result
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommInitAll(
    comms: *mut ncclComm_t,
    ndevs: c_int,
    devlist: *const c_int,
) -> ncclResult_t {
    eprintln!("[NCCL Hook] ncclCommInitAll called with ndevs={}", ndevs);
    
    initialize_fault_tolerance();
    load_original_functions();
    
    // Call original function
    let result = if let Some(ref funcs) = ORIGINAL_FUNCTIONS {
        if let Some(orig_fn) = funcs.ncclCommInitAll {
            eprintln!("[NCCL Hook] Calling original ncclCommInitAll");
            orig_fn(comms, ndevs, devlist)
        } else {
            eprintln!("[NCCL Hook] Original ncclCommInitAll not found");
            ncclResult_t::ncclInternalError
        }
    } else {
        eprintln!("[NCCL Hook] No original functions loaded, using fallback");
        // Fallback implementation
        for i in 0..ndevs {
            let comm = comms.offset(i as isize);
            *comm = Box::into_raw(Box::new(ncclComm { _private: [] }));
        }
        ncclResult_t::ncclSuccess
    };
    
    result
}

// Commented out to avoid duplicate symbol with nccl_fault_tolerant.rs
// Use nccl_fault_tolerant.rs version instead
pub unsafe extern "C" fn ncclCommDestroy_hooks(comm: ncclComm_t) -> ncclResult_t {
    eprintln!("[NCCL Hook] ncclCommDestroy called");
    
    // Call original function
    let result = if let Some(ref funcs) = ORIGINAL_FUNCTIONS {
        if let Some(orig_fn) = funcs.ncclCommDestroy {
            orig_fn(comm)
        } else {
            ncclResult_t::ncclInternalError
        }
    } else {
        // Fallback cleanup
        if !comm.is_null() {
            Box::from_raw(comm);
        }
        ncclResult_t::ncclSuccess
    };
    
    result
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommCount(
    comm: ncclComm_t,
    count: *mut c_int,
) -> ncclResult_t {
    eprintln!("[NCCL Hook] ncclCommCount called");
    // This is a stub - would need actual implementation
    if !count.is_null() {
        *count = 1;  // Default to 1 for single GPU
    }
    ncclResult_t::ncclSuccess
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t {
    eprintln!("[NCCL Hook] ncclCommAbort called");
    ncclCommDestroy_hooks(comm)
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommGetAsyncError(
    comm: ncclComm_t,
    asyncError: *mut ncclResult_t,
) -> ncclResult_t {
    eprintln!("[NCCL Hook] ncclCommGetAsyncError called");
    if !asyncError.is_null() {
        *asyncError = ncclResult_t::ncclSuccess;
    }
    ncclResult_t::ncclSuccess
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommRegister(
    comm: ncclComm_t,
    buff: *mut c_void,
    size: usize,
    handle: *mut *mut c_void,
) -> ncclResult_t {
    eprintln!("[NCCL Hook] ncclCommRegister called with size={}", size);
    // Stub implementation
    if !handle.is_null() {
        *handle = buff;
    }
    ncclResult_t::ncclSuccess
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommDeregister(
    comm: ncclComm_t,
    handle: *mut c_void,
) -> ncclResult_t {
    eprintln!("[NCCL Hook] ncclCommDeregister called");
    // Stub implementation
    ncclResult_t::ncclSuccess
}

// Commented out to avoid duplicate symbol with nccl_fault_tolerant.rs
// Use nccl_fault_tolerant.rs version instead
pub unsafe extern "C" fn ncclAllReduce_hooks(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    eprintln!("[NCCL Hook] ncclAllReduce called with count={}", count);
    // Heartbeat update
    if let Some(manager) = get_manager() {
        let mgr = manager.lock().unwrap();
        // Update heartbeat for current GPU (would need actual rank info)
        mgr.update_heartbeat(0);
    }
    
    // Call original function with error handling
    let mut result = if let Some(ref funcs) = ORIGINAL_FUNCTIONS {
        if let Some(orig_fn) = funcs.ncclAllReduce {
            orig_fn(sendbuff, recvbuff, count, datatype, op, comm, stream)
        } else {
            ncclResult_t::ncclInternalError
        }
    } else {
        // Fallback: simple memcpy for single GPU case
        if !sendbuff.is_null() && !recvbuff.is_null() {
            let size = count * get_datatype_size(datatype);
            std::ptr::copy_nonoverlapping(sendbuff as *const u8, recvbuff as *mut u8, size);
        }
        ncclResult_t::ncclSuccess
    };
    
    // Handle errors with fault tolerance
    if result != ncclResult_t::ncclSuccess {
        if let Some(manager) = get_manager() {
            let mgr = manager.lock().unwrap();
            result = mgr.handle_nccl_error(result, comm);
            
            // If recovery was successful, retry the operation
            if result == ncclResult_t::ncclSuccess {
                if let Some(ref funcs) = ORIGINAL_FUNCTIONS {
                    if let Some(orig_fn) = funcs.ncclAllReduce {
                        result = orig_fn(sendbuff, recvbuff, count, datatype, op, comm, stream);
                    }
                }
            }
        }
    }
    
    result
}

// Commented out to avoid duplicate symbol with nccl_fault_tolerant.rs
// Use nccl_fault_tolerant.rs version instead
pub unsafe extern "C" fn ncclBroadcast_hooks(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    // Similar implementation to ncclAllReduce
    let mut result = if let Some(ref funcs) = ORIGINAL_FUNCTIONS {
        if let Some(orig_fn) = funcs.ncclBroadcast {
            orig_fn(sendbuff, recvbuff, count, datatype, root, comm, stream)
        } else {
            ncclResult_t::ncclInternalError
        }
    } else {
        // Fallback implementation
        if !sendbuff.is_null() && !recvbuff.is_null() {
            let size = count * get_datatype_size(datatype);
            std::ptr::copy_nonoverlapping(sendbuff as *const u8, recvbuff as *mut u8, size);
        }
        ncclResult_t::ncclSuccess
    };
    
    // Handle errors
    if result != ncclResult_t::ncclSuccess {
        if let Some(manager) = get_manager() {
            let mgr = manager.lock().unwrap();
            result = mgr.handle_nccl_error(result, comm);
        }
    }
    
    result
}

#[no_mangle]
pub unsafe extern "C" fn ncclReduce(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    let mut result = if let Some(ref funcs) = ORIGINAL_FUNCTIONS {
        if let Some(orig_fn) = funcs.ncclReduce {
            orig_fn(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
        } else {
            ncclResult_t::ncclInternalError
        }
    } else {
        // Fallback implementation
        if !sendbuff.is_null() && !recvbuff.is_null() {
            let size = count * get_datatype_size(datatype);
            std::ptr::copy_nonoverlapping(sendbuff as *const u8, recvbuff as *mut u8, size);
        }
        ncclResult_t::ncclSuccess
    };
    
    // Handle errors
    if result != ncclResult_t::ncclSuccess {
        if let Some(manager) = get_manager() {
            let mgr = manager.lock().unwrap();
            result = mgr.handle_nccl_error(result, comm);
        }
    }
    
    result
}

// Commented out to avoid duplicate symbol with nccl_fault_tolerant.rs
// Use nccl_fault_tolerant.rs version instead
pub unsafe extern "C" fn ncclGetErrorString_hooks(result: ncclResult_t) -> *const c_char {
    if let Some(ref funcs) = ORIGINAL_FUNCTIONS {
        if let Some(orig_fn) = funcs.ncclGetErrorString {
            return orig_fn(result);
        }
    }
    
    // Fallback error strings
    match result {
        ncclResult_t::ncclSuccess => b"Success\0".as_ptr() as *const c_char,
        ncclResult_t::ncclUnhandledCudaError => b"Unhandled CUDA error\0".as_ptr() as *const c_char,
        ncclResult_t::ncclSystemError => b"System error\0".as_ptr() as *const c_char,
        ncclResult_t::ncclInternalError => b"Internal error\0".as_ptr() as *const c_char,
        ncclResult_t::ncclInvalidArgument => b"Invalid argument\0".as_ptr() as *const c_char,
        ncclResult_t::ncclInvalidUsage => b"Invalid usage\0".as_ptr() as *const c_char,
        _ => b"Unknown error\0".as_ptr() as *const c_char,
    }
}

// Helper function to get datatype size
fn get_datatype_size(datatype: ncclDataType_t) -> usize {
    // NCCL data type sizes
    const ncclInt8: ncclDataType_t = 0;
    const ncclChar: ncclDataType_t = 0;
    const ncclUint8: ncclDataType_t = 1;
    const ncclInt32: ncclDataType_t = 2;
    const ncclInt: ncclDataType_t = 2;
    const ncclUint32: ncclDataType_t = 3;
    const ncclInt64: ncclDataType_t = 4;
    const ncclUint64: ncclDataType_t = 5;
    const ncclFloat16: ncclDataType_t = 6;
    const ncclHalf: ncclDataType_t = 6;
    const ncclFloat32: ncclDataType_t = 7;
    const ncclFloat: ncclDataType_t = 7;
    const ncclFloat64: ncclDataType_t = 8;
    const ncclDouble: ncclDataType_t = 8;
    const ncclBfloat16: ncclDataType_t = 9;
    
    match datatype {
        ncclInt8 | ncclChar | ncclUint8 => 1,
        ncclFloat16 | ncclHalf | ncclBfloat16 => 2,
        ncclInt32 | ncclInt | ncclUint32 | ncclFloat32 | ncclFloat => 4,
        ncclInt64 | ncclUint64 | ncclFloat64 | ncclDouble => 8,
        _ => 1,
    }
}

// API for setting recovery strategy
#[no_mangle]
pub extern "C" fn nccl_set_recovery_strategy(strategy: c_int) {
    if let Some(manager) = get_manager() {
        let mut mgr = manager.lock().unwrap();
        let strategy = match strategy {
            0 => RecoveryStrategy::ExcludeAndRebuild,
            1 => RecoveryStrategy::RecoverAndRetry,
            2 => RecoveryStrategy::CheckpointRestore,
            3 => RecoveryStrategy::DynamicReconfiguration,
            _ => RecoveryStrategy::ExcludeAndRebuild,
        };
        mgr.set_recovery_strategy(strategy);
    }
}

// API for manual checkpoint
#[no_mangle]
pub extern "C" fn nccl_save_checkpoint(comm: ncclComm_t, data: *const u8, size: usize) {
    if let Some(manager) = get_manager() {
        let mgr = manager.lock().unwrap();
        if !data.is_null() && size > 0 {
            let checkpoint = unsafe {
                std::slice::from_raw_parts(data, size).to_vec()
            };
            mgr.save_checkpoint(comm, checkpoint);
        }
    }
}

// API for registering custom error hook
#[no_mangle]
pub extern "C" fn nccl_register_error_callback(
    callback: extern "C" fn(ncclResult_t, ncclComm_t),
) {
    if let Some(manager) = get_manager() {
        let mgr = manager.lock().unwrap();
        mgr.register_error_hook(move |error, comm| {
            callback(error, comm);
        });
    }
}

// Module for libc functions
mod libc {
    use std::ffi::c_char;
    use std::ffi::c_void;
    use std::ffi::c_int;
    
    pub const RTLD_LAZY: c_int = 0x00001;
    pub const RTLD_NOLOAD: c_int = 0x00004;
    
    extern "C" {
        pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        pub fn dlclose(handle: *mut c_void) -> c_int;
    }
}