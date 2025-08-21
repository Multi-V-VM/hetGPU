use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::ffi::{c_void, c_int, c_char, CString};
use std::ptr;
use std::thread;
use cuda_types::cuda::*;

// NCCL类型定义 - 使用usize作为句柄以避免Send/Sync问题
type NcclComm = usize;
type NcclUniqueId = [u8; 128];
type CudaStream = *mut c_void;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NcclResult {
    Success = 0,
    UnhandledCudaError = 1,
    SystemError = 2,
    InternalError = 3,
    InvalidArgument = 4,
    InvalidUsage = 5,
    NumResults = 6,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum NcclDataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum NcclRedOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

// 设备健康状态
#[derive(Clone, Debug)]
struct DeviceHealth {
    device_id: i32,
    health_score: i32,  // 0-100, 100 is healthy
    failure_count: u32,
    last_check: Instant,
}

// 容错上下文
#[derive(Clone)]
struct FaultTolerantContext {
    current_device: i32,
    backup_device: i32,
    active_comm: NcclComm,
    backup_comm: Option<NcclComm>,
    rank: i32,
    nranks: i32,
    comm_id: NcclUniqueId,
    is_migrating: Arc<Mutex<bool>>,
}

// Safe to share between threads
unsafe impl Send for FaultTolerantContext {}
unsafe impl Sync for FaultTolerantContext {}

// 全局状态
lazy_static::lazy_static! {
    static ref COMM_CONTEXTS: RwLock<HashMap<NcclComm, Arc<Mutex<FaultTolerantContext>>>> = 
        RwLock::new(HashMap::new());
    static ref DEVICE_HEALTH: RwLock<Vec<DeviceHealth>> = RwLock::new(Vec::new());
    static ref ORIGINAL_NCCL: RwLock<Option<NcclFunctions>> = RwLock::new(None);
}

// 原始NCCL函数指针
struct NcclFunctions {
    comm_init_rank: unsafe extern "C" fn(*mut *mut c_void, c_int, *const NcclUniqueId, c_int) -> NcclResult,
    all_reduce: unsafe extern "C" fn(*const c_void, *mut c_void, usize, NcclDataType, NcclRedOp, *mut c_void, CudaStream) -> NcclResult,
    broadcast: unsafe extern "C" fn(*const c_void, *mut c_void, usize, NcclDataType, c_int, *mut c_void, CudaStream) -> NcclResult,
    all_gather: unsafe extern "C" fn(*const c_void, *mut c_void, usize, NcclDataType, *mut c_void, CudaStream) -> NcclResult,
    reduce_scatter: unsafe extern "C" fn(*const c_void, *mut c_void, usize, NcclDataType, NcclRedOp, *mut c_void, CudaStream) -> NcclResult,
    comm_destroy: unsafe extern "C" fn(*mut c_void) -> NcclResult,
    comm_abort: unsafe extern "C" fn(*mut c_void) -> NcclResult,
    get_error_string: unsafe extern "C" fn(NcclResult) -> *const c_char,
}

// 初始化NCCL钩子
pub fn initialize_nccl_hooks() {
    eprintln!("[NCCL-FT-Rust] Initializing NCCL Fault Tolerance Hooks");
    
    // 加载原始NCCL库
    unsafe {
        let lib = libloading::Library::new("libnccl.so.2")
            .or_else(|_| libloading::Library::new("/usr/local/cuda/lib64/libnccl.so.2"))
            .or_else(|_| libloading::Library::new("libnccl.so"))
            .or_else(|_| libloading::Library::new("/usr/lib/x86_64-linux-gnu/libnccl.so.2"));
        
        if let Ok(lib) = lib {
            // 使用 unwrap_or 提供默认值，避免 unwrap 导致崩溃
            let comm_init_rank = lib.get(b"ncclCommInitRank")
                .map(|f| *f)
                .unwrap_or_else(|e| {
                    eprintln!("[NCCL-FT-Rust] WARNING: Failed to load ncclCommInitRank: {}", e);
                    panic!("Critical function ncclCommInitRank not found");
                });
            
            let all_reduce = lib.get(b"ncclAllReduce")
                .map(|f| *f)
                .unwrap_or_else(|e| {
                    eprintln!("[NCCL-FT-Rust] WARNING: Failed to load ncclAllReduce: {}", e);
                    panic!("Critical function ncclAllReduce not found");
                });
            
            let broadcast = lib.get(b"ncclBroadcast")
                .map(|f| *f)
                .unwrap_or_else(|e| {
                    eprintln!("[NCCL-FT-Rust] WARNING: Failed to load ncclBroadcast: {}", e);
                    panic!("Critical function ncclBroadcast not found");
                });
            
            let all_gather = lib.get(b"ncclAllGather")
                .map(|f| *f)
                .unwrap_or_else(|e| {
                    eprintln!("[NCCL-FT-Rust] WARNING: Failed to load ncclAllGather: {}", e);
                    panic!("Critical function ncclAllGather not found");
                });
            
            let reduce_scatter = lib.get(b"ncclReduceScatter")
                .map(|f| *f)
                .unwrap_or_else(|e| {
                    eprintln!("[NCCL-FT-Rust] WARNING: Failed to load ncclReduceScatter: {}", e);
                    panic!("Critical function ncclReduceScatter not found");
                });
            
            let comm_destroy = lib.get(b"ncclCommDestroy")
                .map(|f| *f)
                .unwrap_or_else(|e| {
                    eprintln!("[NCCL-FT-Rust] WARNING: Failed to load ncclCommDestroy: {}", e);
                    panic!("Critical function ncclCommDestroy not found");
                });
            
            let comm_abort = lib.get(b"ncclCommAbort")
                .map(|f| *f)
                .unwrap_or_else(|e| {
                    eprintln!("[NCCL-FT-Rust] WARNING: Failed to load ncclCommAbort: {}", e);
                    panic!("Critical function ncclCommAbort not found");
                });
            
            let get_error_string = lib.get(b"ncclGetErrorString")
                .map(|f| *f)
                .unwrap_or_else(|e| {
                    eprintln!("[NCCL-FT-Rust] WARNING: Failed to load ncclGetErrorString: {}", e);
                    panic!("Critical function ncclGetErrorString not found");
                });
            
            let funcs = NcclFunctions {
                comm_init_rank,
                all_reduce,
                broadcast,
                all_gather,
                reduce_scatter,
                comm_destroy,
                comm_abort,
                get_error_string,
            };
            
            *ORIGINAL_NCCL.write().unwrap() = Some(funcs);
            std::mem::forget(lib); // 保持库加载
            eprintln!("[NCCL-FT-Rust] Successfully loaded original NCCL library");
        } else {
            eprintln!("[NCCL-FT-Rust] ERROR: Failed to load NCCL library");
        }
    }
    
    // 初始化设备健康状态
    let mut device_count: c_int = 0;
    unsafe {
        super::nvidia_backend::cudaGetDeviceCount(&mut device_count);
    }
    
    let mut health_vec = Vec::new();
    for i in 0..device_count {
        health_vec.push(DeviceHealth {
            device_id: i,
            health_score: 100,
            failure_count: 0,
            last_check: Instant::now(),
        });
    }
    *DEVICE_HEALTH.write().unwrap() = health_vec;
    
    eprintln!("[NCCL-FT-Rust] Found {} CUDA devices", device_count);
    
    // 启动健康监控线程
    start_health_monitor();
}

// 查找最佳备用设备
fn find_best_backup_device(current_device: i32) -> Option<i32> {
    let health = DEVICE_HEALTH.read().unwrap();
    let mut best_device = None;
    let mut best_score = 0;
    
    for device in health.iter() {
        if device.device_id == current_device {
            continue;
        }
        
        // 检查设备可用性
        unsafe {
            super::nvidia_backend::cudaSetDevice(device.device_id);
            // 使用cudaStreamSynchronize(0)代替cudaDeviceSynchronize
            if super::nvidia_backend::cudaStreamSynchronize(ptr::null_mut()) == CUresult::SUCCESS {
                if device.health_score > best_score {
                    best_score = device.health_score;
                    best_device = Some(device.device_id);
                }
            }
        }
    }
    
    best_device
}

// 执行设备迁移
fn perform_device_migration(
    ctx: &mut FaultTolerantContext,
    data: *mut c_void,
    size: usize
) -> Result<(), String> {
    eprintln!("[NCCL-FT-Rust] Starting migration from device {} to device {}", 
             ctx.current_device, ctx.backup_device);
    
    *ctx.is_migrating.lock().unwrap() = true;
    
    // 1. 分配备用设备内存
    let mut backup_buffer: *mut c_void = ptr::null_mut();
    unsafe {
        super::nvidia_backend::cudaSetDevice(ctx.backup_device);
        let err = super::nvidia_backend::cudaMalloc(&mut backup_buffer, size);
        if err != CUresult::SUCCESS {
            *ctx.is_migrating.lock().unwrap() = false;
            return Err(format!("Failed to allocate memory on backup device: {:?}", err));
        }
    }
    
    // 2. 复制数据到备用设备
    unsafe {
        let err = super::nvidia_backend::cudaMemcpy(
            backup_buffer,
            data,
            size,
            1 // cudaMemcpyDeviceToDevice
        );
        if err != CUresult::SUCCESS {
            super::nvidia_backend::cudaFree(backup_buffer);
            *ctx.is_migrating.lock().unwrap() = false;
            return Err(format!("Failed to copy data to backup device: {:?}", err));
        }
    }
    
    // 3. 创建新的NCCL通信器
    if ctx.backup_comm.is_none() {
        unsafe {
            super::nvidia_backend::cudaSetDevice(ctx.backup_device);
            if let Some(nccl) = ORIGINAL_NCCL.read().unwrap().as_ref() {
                let mut new_comm: *mut c_void = ptr::null_mut();
                let result = (nccl.comm_init_rank)(
                    &mut new_comm,
                    ctx.nranks,
                    &ctx.comm_id,
                    ctx.rank
                );
                if result != NcclResult::Success {
                    super::nvidia_backend::cudaFree(backup_buffer);
                    *ctx.is_migrating.lock().unwrap() = false;
                    return Err(format!("Failed to create backup communicator: {:?}", result));
                }
                ctx.backup_comm = Some(new_comm as usize);
            }
        }
    }
    
    // 4. 切换到备用通信器
    if let Some(backup) = ctx.backup_comm {
        let old_comm = ctx.active_comm;
        ctx.active_comm = backup;
        ctx.backup_comm = Some(old_comm);
    }
    
    // 5. 交换设备ID
    std::mem::swap(&mut ctx.current_device, &mut ctx.backup_device);
    
    // 6. 更新设备健康状态
    {
        let mut health = DEVICE_HEALTH.write().unwrap();
        if let Some(device) = health.iter_mut().find(|d| d.device_id == ctx.backup_device) {
            device.failure_count += 1;
            device.health_score = (device.health_score - 10).max(0);
        }
    }
    
    *ctx.is_migrating.lock().unwrap() = false;
    
    eprintln!("[NCCL-FT-Rust] Migration completed successfully");
    Ok(())
}

// NCCL通信初始化钩子
#[allow(non_snake_case)]
pub unsafe extern "C" fn ncclCommInitRank(
    comm: *mut *mut c_void,
    nranks: c_int,
    comm_id: NcclUniqueId,  // Changed to pass by value
    rank: c_int
) -> NcclResult {
    eprintln!("[NCCL-FT-Rust] ncclCommInitRank called");
    eprintln!("[NCCL-FT-Rust]   comm ptr: {:p}", comm);
    eprintln!("[NCCL-FT-Rust]   nranks: {}", nranks);
    eprintln!("[NCCL-FT-Rust]   comm_id: first 8 bytes: {:?}", &comm_id[0..8]);
    eprintln!("[NCCL-FT-Rust]   rank: {}", rank);
    
    // 验证输入参数
    if comm.is_null() {
        eprintln!("[NCCL-FT-Rust] ERROR: comm pointer is null");
        return NcclResult::InvalidArgument;
    }
    
    if nranks <= 0 || rank < 0 || rank >= nranks {
        eprintln!("[NCCL-FT-Rust] ERROR: Invalid rank ({}) or nranks ({})", rank, nranks);
        return NcclResult::InvalidArgument;
    }
    
    let nccl_guard = ORIGINAL_NCCL.read().unwrap();
    let nccl = match nccl_guard.as_ref() {
        Some(n) => n,
        None => {
            eprintln!("[NCCL-FT-Rust] ERROR: Original NCCL functions not loaded");
            return NcclResult::SystemError;
        }
    };
    
    // 获取当前设备
    let mut current_device: c_int = 0;
    let cuda_result = super::nvidia_backend::cudaGetDevice(&mut current_device);
    if cuda_result != CUresult::SUCCESS {
        eprintln!("[NCCL-FT-Rust] ERROR: Failed to get current device: {:?}", cuda_result);
        return NcclResult::UnhandledCudaError;
    }
    
    // comm_id已经是按值传递的
    
    // 创建容错上下文
    let mut ctx = FaultTolerantContext {
        current_device,
        backup_device: find_best_backup_device(current_device).unwrap_or(-1),
        active_comm: 0,
        backup_comm: None,
        rank,
        nranks,
        comm_id: comm_id,
        is_migrating: Arc::new(Mutex::new(false)),
    };
    
    // 初始化主通信器
    let mut comm_ptr: *mut c_void = ptr::null_mut();
    let result = (nccl.comm_init_rank)(&mut comm_ptr, nranks, &comm_id, rank);
    
    if result != NcclResult::Success {
        eprintln!("[NCCL-FT-Rust] Failed to initialize primary communicator: {:?}", result);
        return result;
    }
    
    if comm_ptr.is_null() {
        eprintln!("[NCCL-FT-Rust] ERROR: Communicator pointer is null after initialization");
        return NcclResult::InternalError;
    }
    
    ctx.active_comm = comm_ptr as usize;
    *comm = comm_ptr;
    
    // 保存上下文
    COMM_CONTEXTS.write().unwrap().insert(
        comm_ptr as usize,
        Arc::new(Mutex::new(ctx))
    );
    
    eprintln!("[NCCL-FT-Rust] Communicator initialized successfully with fault tolerance");
    result
}

// AllReduce容错包装
#[allow(non_snake_case)]
pub unsafe extern "C" fn ncclAllReduce(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: NcclDataType,
    op: NcclRedOp,
    comm: usize,
    stream: CudaStream
) -> NcclResult {
    let nccl_guard = ORIGINAL_NCCL.read().unwrap();
    let nccl = match nccl_guard.as_ref() {
        Some(n) => n,
        None => return NcclResult::SystemError,
    };
    
    // 查找容错上下文
    let ctx_arc = {
        let contexts = COMM_CONTEXTS.read().unwrap();
        contexts.get(&(comm as usize)).cloned()
    };
    
    let ctx_arc = match ctx_arc {
        Some(c) => c,
        None => {
            // 没有容错上下文，直接调用原始函数
            return (nccl.all_reduce)(sendbuff, recvbuff, count, datatype, op, comm as *mut c_void, stream);
        }
    };
    
    // 计算数据大小
    let element_size = match datatype {
        NcclDataType::Int8 | NcclDataType::Uint8 => 1,
        NcclDataType::Float16 => 2,
        NcclDataType::Int32 | NcclDataType::Uint32 | NcclDataType::Float32 => 4,
        NcclDataType::Int64 | NcclDataType::Uint64 | NcclDataType::Float64 => 8,
    };
    let data_size = count * element_size;
    
    // 带容错的执行
    let max_retries = 3;
    let mut retry_count = 0;
    let mut result = NcclResult::Success;
    
    while retry_count < max_retries {
        // 等待迁移完成
        while *ctx_arc.lock().unwrap().is_migrating.lock().unwrap() {
            thread::sleep(Duration::from_millis(1));
        }
        
        let ctx = ctx_arc.lock().unwrap();
        super::nvidia_backend::cudaSetDevice(ctx.current_device);
        let active_comm = ctx.active_comm as *mut c_void;
        drop(ctx); // 释放锁
        
        result = (nccl.all_reduce)(sendbuff, recvbuff, count, datatype, op, active_comm, stream);
        
        if result == NcclResult::Success {
            // 成功，提高健康分数
            let mut health = DEVICE_HEALTH.write().unwrap();
            let ctx = ctx_arc.lock().unwrap();
            if let Some(device) = health.iter_mut().find(|d| d.device_id == ctx.current_device) {
                device.health_score = (device.health_score + 2).min(100);
            }
            break;
        }
        
        eprintln!("[NCCL-FT-Rust] AllReduce failed: {:?} (attempt {}/{})",
                 result, retry_count + 1, max_retries);
        
        // 检查是否需要迁移
        if result == NcclResult::UnhandledCudaError || result == NcclResult::SystemError {
            let mut ctx = ctx_arc.lock().unwrap();
            if ctx.backup_device >= 0 {
                eprintln!("[NCCL-FT-Rust] Attempting device migration...");
                
                // 创建临时缓冲区保存数据
                let mut temp_buffer: *mut c_void = ptr::null_mut();
                super::nvidia_backend::cudaMalloc(&mut temp_buffer, data_size);
                super::nvidia_backend::cudaMemcpy(temp_buffer, sendbuff, data_size, 1);
                
                if perform_device_migration(&mut ctx, temp_buffer, data_size).is_ok() {
                    // 迁移成功
                    eprintln!("[NCCL-FT-Rust] Migration successful, retrying operation");
                } else {
                    super::nvidia_backend::cudaFree(temp_buffer);
                    break;
                }
            } else {
                eprintln!("[NCCL-FT-Rust] No backup device available");
                break;
            }
        }
        
        retry_count += 1;
        thread::sleep(Duration::from_millis(100 * retry_count as u64));
    }
    
    result
}

// Broadcast容错包装
#[allow(non_snake_case)]
pub unsafe extern "C" fn ncclBroadcast(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: NcclDataType,
    root: c_int,
    comm: usize,
    stream: CudaStream
) -> NcclResult {
    let nccl_guard = ORIGINAL_NCCL.read().unwrap();
    let nccl = match nccl_guard.as_ref() {
        Some(n) => n,
        None => return NcclResult::SystemError,
    };
    
    // 查找容错上下文
    let ctx_arc = {
        let contexts = COMM_CONTEXTS.read().unwrap();
        contexts.get(&(comm as usize)).cloned()
    };
    
    let ctx_arc = match ctx_arc {
        Some(c) => c,
        None => {
            return (nccl.broadcast)(sendbuff, recvbuff, count, datatype, root, comm as *mut c_void, stream);
        }
    };
    
    // 类似AllReduce的容错处理
    let max_retries = 3;
    let mut retry_count = 0;
    let mut result = NcclResult::Success;
    
    while retry_count < max_retries {
        while *ctx_arc.lock().unwrap().is_migrating.lock().unwrap() {
            thread::sleep(Duration::from_millis(1));
        }
        
        let ctx = ctx_arc.lock().unwrap();
        super::nvidia_backend::cudaSetDevice(ctx.current_device);
        let active_comm = ctx.active_comm as *mut c_void;
        drop(ctx);
        
        result = (nccl.broadcast)(sendbuff, recvbuff, count, datatype, root, active_comm, stream);
        
        if result == NcclResult::Success {
            break;
        }
        
        eprintln!("[NCCL-FT-Rust] Broadcast failed: {:?} (attempt {}/{})",
                 result, retry_count + 1, max_retries);
        
        retry_count += 1;
        thread::sleep(Duration::from_millis(100 * retry_count as u64));
    }
    
    result
}

// 通信器销毁
#[allow(non_snake_case)]
pub unsafe extern "C" fn ncclCommDestroy(comm: usize) -> NcclResult {
    let nccl_guard = ORIGINAL_NCCL.read().unwrap();
    let nccl = match nccl_guard.as_ref() {
        Some(n) => n,
        None => return NcclResult::SystemError,
    };
    
    // 查找并删除容错上下文
    let ctx_arc = COMM_CONTEXTS.write().unwrap().remove(&(comm as usize));
    
    if let Some(ctx_arc) = ctx_arc {
        let ctx = ctx_arc.lock().unwrap();
        
        // 销毁两个通信器
        (nccl.comm_destroy)(ctx.active_comm as *mut c_void);
        if let Some(backup) = ctx.backup_comm {
            (nccl.comm_destroy)(backup as *mut c_void);
        }
        
        eprintln!("[NCCL-FT-Rust] Communicator destroyed");
        NcclResult::Success
    } else {
        (nccl.comm_destroy)(comm as *mut c_void)
    }
}

// 错误字符串
#[allow(non_snake_case)]
pub unsafe extern "C" fn ncclGetErrorString(result: NcclResult) -> *const c_char {
    if let Some(nccl) = ORIGINAL_NCCL.read().unwrap().as_ref() {
        return (nccl.get_error_string)(result);
    }
    
    let s = match result {
        NcclResult::Success => "no error",
        NcclResult::UnhandledCudaError => "unhandled cuda error",
        NcclResult::SystemError => "system error",
        NcclResult::InternalError => "internal error",
        NcclResult::InvalidArgument => "invalid argument",
        NcclResult::InvalidUsage => "invalid usage",
        NcclResult::NumResults => "num results",
    };
    
    s.as_ptr() as *const c_char
}

// 健康监控线程
fn health_monitor_thread() {
    loop {
        thread::sleep(Duration::from_secs(5));
        
        let mut health = DEVICE_HEALTH.write().unwrap();
        for device in health.iter_mut() {
            unsafe {
                super::nvidia_backend::cudaSetDevice(device.device_id);
                // 使用cudaStreamSynchronize(0)代替cudaDeviceSynchronize
                let err = super::nvidia_backend::cudaStreamSynchronize(ptr::null_mut());
                
                if err != CUresult::SUCCESS {
                    device.health_score = (device.health_score - 20).max(0);
                    eprintln!("[NCCL-FT-Rust] Device {} health decreased to {}",
                             device.device_id, device.health_score);
                } else {
                    // 缓慢恢复健康分数
                    device.health_score = (device.health_score + 1).min(100);
                }
                
                device.last_check = Instant::now();
            }
        }
    }
}

// 启动健康监控
fn start_health_monitor() {
    thread::spawn(|| {
        health_monitor_thread();
    });
}