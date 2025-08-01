//! NVIDIA 后端实现 - 直接转发到原生 CUDA API
//! 
//! 这个模块在检测到 NVIDIA GPU 时，直接调用系统的 CUDA 库，
//! 但同时保持我们的动态内存跟踪功能。

// 集成完整版内存跟踪器
use crate::r#impl::simple_memory_tracer::{get_simple_tracer, track_memory_copy, track_memory_set, AccessType, AllocationType};
use cuda_types::cuda::*;
use std::ffi::{c_void, CStr};
use std::ptr;
use std::sync::{Mutex, Once, RwLock, Arc};
use std::collections::HashMap;
use lazy_static::lazy_static;
use libc;

/// Thread-safe wrapper for raw library handle
struct LibraryHandle(*mut c_void);

unsafe impl Send for LibraryHandle {}
unsafe impl Sync for LibraryHandle {}

/// NVIDIA CUDA 库的动态加载句柄
struct NvidiaCudaLibrary {
    /// libcuda.so 的句柄
    handle: LibraryHandle,
    /// 原生 CUDA 函数指针
    functions: CudaFunctions,
}

/// 设备特定的上下文信息
#[derive(Clone)]
struct DeviceContext {
    device_id: CUdevice,
    context: CUcontext,
    is_primary: bool,
}

/// 线程安全的设备上下文管理器
struct DeviceContextManager {
    contexts: HashMap<i32, DeviceContext>,
    current_device: Option<i32>,
}

/// 原生 CUDA 函数指针结构
#[derive(Debug)]
struct CudaFunctions {
    // 初始化和版本
    cu_init: Option<unsafe extern "C" fn(flags: u32) -> CUresult>,
    cu_driver_get_version: Option<unsafe extern "C" fn(driver_version: *mut i32) -> CUresult>,
    
    // 设备管理
    cu_device_get_count: Option<unsafe extern "C" fn(count: *mut i32) -> CUresult>,
    cu_device_get: Option<unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult>,
    cu_device_get_name: Option<unsafe extern "C" fn(name: *mut i8, len: i32, dev: CUdevice) -> CUresult>,
    cu_device_get_attribute: Option<unsafe extern "C" fn(pi: *mut i32, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult>,
    cu_device_compute_capability: Option<unsafe extern "C" fn(major: *mut i32, minor: *mut i32, dev: CUdevice) -> CUresult>,
    cu_device_total_mem_v2: Option<unsafe extern "C" fn(bytes: *mut usize, dev: CUdevice) -> CUresult>,
    cu_device_get_properties: Option<unsafe extern "C" fn(prop: *mut CUdevprop, dev: CUdevice) -> CUresult>,
    cu_device_get_uuid: Option<unsafe extern "C" fn(uuid: *mut CUuuid, dev: CUdevice) -> CUresult>,
    cu_device_get_uuid_v2: Option<unsafe extern "C" fn(uuid: *mut CUuuid, dev: CUdevice) -> CUresult>,
    cu_device_get_luid: Option<unsafe extern "C" fn(luid: *mut i8, device_node_mask: *mut u32, dev: CUdevice) -> CUresult>,
    
    // 上下文管理
    cu_ctx_create_v2: Option<unsafe extern "C" fn(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult>,
    cu_ctx_destroy_v2: Option<unsafe extern "C" fn(ctx: CUcontext) -> CUresult>,
    cu_ctx_push_current_v2: Option<unsafe extern "C" fn(ctx: CUcontext) -> CUresult>,
    cu_ctx_pop_current_v2: Option<unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult>,
    cu_ctx_set_current: Option<unsafe extern "C" fn(ctx: CUcontext) -> CUresult>,
    cu_ctx_get_current: Option<unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult>,
    cu_ctx_get_device: Option<unsafe extern "C" fn(device: *mut CUdevice) -> CUresult>,
    cu_ctx_synchronize: Option<unsafe extern "C" fn() -> CUresult>,
    cu_ctx_set_limit: Option<unsafe extern "C" fn(limit: CUlimit, value: usize) -> CUresult>,
    cu_ctx_get_limit: Option<unsafe extern "C" fn(pvalue: *mut usize, limit: CUlimit) -> CUresult>,
    cu_device_primary_ctx_retain: Option<unsafe extern "C" fn(pctx: *mut CUcontext, dev: CUdevice) -> CUresult>,
    cu_device_primary_ctx_release: Option<unsafe extern "C" fn(dev: CUdevice) -> CUresult>,
    
    // 内存管理
    cu_mem_alloc_v2: Option<unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult>,
    cu_mem_free_v2: Option<unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult>,
    cu_memcpy_hto_d_v2: Option<unsafe extern "C" fn(dst_device: CUdeviceptr, src_host: *const c_void, byte_count: usize) -> CUresult>,
    cu_memcpy_dto_h_v2: Option<unsafe extern "C" fn(dst_host: *mut c_void, src_device: CUdeviceptr, byte_count: usize) -> CUresult>,
    cu_memcpy_dto_d_v2: Option<unsafe extern "C" fn(dst_device: CUdeviceptr, src_device: CUdeviceptr, byte_count: usize) -> CUresult>,
    cu_mem_get_address_range_v2: Option<unsafe extern "C" fn(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult>,
    cu_memset_d8_v2: Option<unsafe extern "C" fn(dstDevice: CUdeviceptr, uc: u8, n: usize) -> CUresult>,
    cu_memset_d32_v2: Option<unsafe extern "C" fn(dstDevice: CUdeviceptr, ui: u32, n: usize) -> CUresult>,
    cu_mem_alloc_host_v2: Option<unsafe extern "C" fn(pp: *mut *mut c_void, bytesize: usize) -> CUresult>,
    cu_mem_free_host: Option<unsafe extern "C" fn(p: *mut c_void) -> CUresult>,
    cu_mem_host_register_v2: Option<unsafe extern "C" fn(p: *mut c_void, bytesize: usize, flags: u32) -> CUresult>,
    cu_mem_host_unregister: Option<unsafe extern "C" fn(p: *mut c_void) -> CUresult>,
    cu_pointer_get_attribute: Option<unsafe extern "C" fn(data: *mut c_void, attribute: CUpointer_attribute, ptr: CUdeviceptr) -> CUresult>,
    
    // 模块和函数管理
    cu_module_load_data: Option<unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult>,
    cu_module_load_data_ex: Option<unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void, num_options: u32, options: *mut CUjit_option, option_values: *mut *mut c_void) -> CUresult>,
    cu_module_unload: Option<unsafe extern "C" fn(hmod: CUmodule) -> CUresult>,
    cu_module_get_function: Option<unsafe extern "C" fn(hfunc: *mut CUfunction, hmod: CUmodule, name: *const i8) -> CUresult>,
    cu_func_get_attribute: Option<unsafe extern "C" fn(pi: *mut i32, attrib: CUfunction_attribute, hfunc: CUfunction) -> CUresult>,
    cu_launch_kernel: Option<unsafe extern "C" fn(f: CUfunction, grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32, block_dim_x: u32, block_dim_y: u32, block_dim_z: u32, shared_mem_bytes: u32, h_stream: CUstream, kernel_params: *mut *mut c_void, extra: *mut *mut c_void) -> CUresult>,
    
    // 流管理
    cu_stream_create: Option<unsafe extern "C" fn(ph_stream: *mut CUstream, flags: u32) -> CUresult>,
    cu_stream_destroy_v2: Option<unsafe extern "C" fn(h_stream: CUstream) -> CUresult>,
    cu_stream_synchronize: Option<unsafe extern "C" fn(h_stream: CUstream) -> CUresult>,
    cu_stream_wait_event: Option<unsafe extern "C" fn(h_stream: CUstream, h_event: CUevent, flags: u32) -> CUresult>,
    
    // 事件管理
    cu_event_create: Option<unsafe extern "C" fn(ph_event: *mut CUevent, flags: u32) -> CUresult>,
    cu_event_destroy_v2: Option<unsafe extern "C" fn(h_event: CUevent) -> CUresult>,
    cu_event_record: Option<unsafe extern "C" fn(h_event: CUevent, h_stream: CUstream) -> CUresult>,
    cu_event_synchronize: Option<unsafe extern "C" fn(h_event: CUevent) -> CUresult>,
    cu_event_elapsed_time: Option<unsafe extern "C" fn(pms: *mut f32, h_start: CUevent, h_end: CUevent) -> CUresult>,
}

lazy_static! {
    /// 全局 NVIDIA CUDA 库实例
    static ref NVIDIA_CUDA: RwLock<Option<Arc<NvidiaCudaLibrary>>> = RwLock::new(None);
    /// 设备上下文管理器
    static ref DEVICE_MANAGER: Mutex<DeviceContextManager> = Mutex::new(DeviceContextManager {
        contexts: HashMap::new(),
        current_device: None,
    });
}

/// 确保后端只初始化一次的标志
static INIT_ONCE: Once = Once::new();

/// 库构造函数实现 - 在动态库加载时自动调用
#[no_mangle]
pub extern "C" fn __zluda_lib_init() {
    INIT_ONCE.call_once(|| {
        eprintln!("[NvidiaBackend] LD_PRELOAD 检测到库加载，开始自动初始化 NVIDIA 后端...");
        
        // 尝试初始化 NVIDIA 后端
        match initialize_nvidia_backend() {
            Ok(()) => {
                eprintln!("[NvidiaBackend] LD_PRELOAD 自动初始化成功！");
                eprintln!("[NvidiaBackend] 现在所有 CUDA 调用将转发到原生 NVIDIA 库并进行内存跟踪");
            },
            Err(e) => {
                eprintln!("[NvidiaBackend] LD_PRELOAD 自动初始化失败: {}", e);
                eprintln!("[NvidiaBackend] 将使用其他后端或返回错误");
            }
        }
    });
}

/// 使用 ctor 属性确保在库加载时调用初始化函数
#[cfg_attr(any(target_os = "linux", target_os = "android"), link_section = ".init_array")]
#[cfg_attr(target_os = "freebsd", link_section = ".init_array")]
#[cfg_attr(target_os = "netbsd", link_section = ".init_array")]
#[cfg_attr(target_os = "openbsd", link_section = ".init_array")]
#[cfg_attr(target_os = "macos", link_section = "__DATA,__mod_init_func")]
#[cfg_attr(target_os = "ios", link_section = "__DATA,__mod_init_func")]
#[cfg_attr(target_os = "windows", link_section = ".CRT$XIB")]
#[used]
static LIBRARY_CTOR: extern "C" fn() = __zluda_lib_init;

impl CudaFunctions {
    fn new() -> Self {
        Self {
            cu_init: None,
            cu_driver_get_version: None,
            cu_device_get_count: None,
            cu_device_get: None,
            cu_device_get_name: None,
            cu_device_get_attribute: None,
            cu_device_compute_capability: None,
            cu_device_total_mem_v2: None,
            cu_device_get_properties: None,
            cu_device_get_uuid: None,
            cu_device_get_uuid_v2: None,
            cu_device_get_luid: None,
            cu_ctx_create_v2: None,
            cu_ctx_destroy_v2: None,
            cu_ctx_push_current_v2: None,
            cu_ctx_pop_current_v2: None,
            cu_ctx_set_current: None,
            cu_ctx_get_current: None,
            cu_ctx_get_device: None,
            cu_ctx_synchronize: None,
            cu_ctx_set_limit: None,
            cu_ctx_get_limit: None,
            cu_device_primary_ctx_retain: None,
            cu_device_primary_ctx_release: None,
            cu_mem_alloc_v2: None,
            cu_mem_free_v2: None,
            cu_memcpy_hto_d_v2: None,
            cu_memcpy_dto_h_v2: None,
            cu_memcpy_dto_d_v2: None,
            cu_mem_get_address_range_v2: None,
            cu_memset_d8_v2: None,
            cu_memset_d32_v2: None,
            cu_mem_alloc_host_v2: None,
            cu_mem_free_host: None,
            cu_mem_host_register_v2: None,
            cu_mem_host_unregister: None,
            cu_pointer_get_attribute: None,
            cu_module_load_data: None,
            cu_module_load_data_ex: None,
            cu_module_unload: None,
            cu_module_get_function: None,
            cu_func_get_attribute: None,
            cu_launch_kernel: None,
            cu_stream_create: None,
            cu_stream_destroy_v2: None,
            cu_stream_synchronize: None,
            cu_stream_wait_event: None,
            cu_event_create: None,
            cu_event_destroy_v2: None,
            cu_event_record: None,
            cu_event_synchronize: None,
            cu_event_elapsed_time: None,
        }
    }
}

impl NvidiaCudaLibrary {
    /// 尝试加载系统上的原生 CUDA 库
    fn try_load() -> Result<Self, String> {
        unsafe {
            // 尝试加载系统的 libcuda.so
            let lib_names = [
                "/usr/lib/x86_64-linux-gnu/libcuda.so.1\0",
                "/usr/lib64/libcuda.so.1\0", 
                "/usr/local/cuda/lib64/libcuda.so.1\0",
                "/opt/cuda/lib64/libcuda.so.1\0",
                "libcuda.so.1\0",
                "libcuda.so\0",
            ];

            let mut handle: *mut c_void = ptr::null_mut();
            
            for lib_name in &lib_names {
                handle = libc::dlopen(lib_name.as_ptr() as *const i8, libc::RTLD_LAZY);
                if !handle.is_null() {
                    eprintln!("[NvidiaBackend] 成功加载 CUDA 库: {}", 
                             CStr::from_ptr(lib_name.as_ptr() as *const i8).to_string_lossy());
                    break;
                }
            }

            if handle.is_null() {
                return Err("无法找到系统 CUDA 库".to_string());
            }

            let mut functions = CudaFunctions::new();
            
            // 加载所有必需的 CUDA 函数
            macro_rules! load_function {
                ($field:ident, $name:expr) => {
                    let name_cstr = concat!($name, "\0");
                    let func_ptr = libc::dlsym(handle, name_cstr.as_ptr() as *const i8);
                    if !func_ptr.is_null() {
                        functions.$field = Some(std::mem::transmute(func_ptr));
                    } else {
                        eprintln!("[NvidiaBackend] 警告: 无法加载函数 {}", $name);
                    }
                };
            }

            // 加载所有 CUDA 函数
            load_function!(cu_init, "cuInit");
            load_function!(cu_driver_get_version, "cuDriverGetVersion");
            load_function!(cu_device_get_count, "cuDeviceGetCount");
            load_function!(cu_device_get, "cuDeviceGet");
            load_function!(cu_device_get_name, "cuDeviceGetName");
            load_function!(cu_device_get_attribute, "cuDeviceGetAttribute");
            load_function!(cu_device_compute_capability, "cuDeviceComputeCapability");
            load_function!(cu_device_total_mem_v2, "cuDeviceTotalMem_v2");
            load_function!(cu_device_get_properties, "cuDeviceGetProperties");
            load_function!(cu_device_get_uuid, "cuDeviceGetUuid");
            load_function!(cu_device_get_uuid_v2, "cuDeviceGetUuid_v2");
            load_function!(cu_device_get_luid, "cuDeviceGetLuid");
            
            load_function!(cu_ctx_create_v2, "cuCtxCreate_v2");
            load_function!(cu_ctx_destroy_v2, "cuCtxDestroy_v2");
            load_function!(cu_ctx_push_current_v2, "cuCtxPushCurrent_v2");
            load_function!(cu_ctx_pop_current_v2, "cuCtxPopCurrent_v2");
            load_function!(cu_ctx_set_current, "cuCtxSetCurrent");
            load_function!(cu_ctx_get_current, "cuCtxGetCurrent");
            load_function!(cu_ctx_get_device, "cuCtxGetDevice");
            load_function!(cu_ctx_synchronize, "cuCtxSynchronize");
            load_function!(cu_ctx_set_limit, "cuCtxSetLimit");
            load_function!(cu_ctx_get_limit, "cuCtxGetLimit");
            load_function!(cu_device_primary_ctx_retain, "cuDevicePrimaryCtxRetain");
            load_function!(cu_device_primary_ctx_release, "cuDevicePrimaryCtxRelease");
            
            load_function!(cu_mem_alloc_v2, "cuMemAlloc_v2");
            load_function!(cu_mem_free_v2, "cuMemFree_v2");
            load_function!(cu_memcpy_hto_d_v2, "cuMemcpyHtoD_v2");
            load_function!(cu_memcpy_dto_h_v2, "cuMemcpyDtoH_v2");
            load_function!(cu_memcpy_dto_d_v2, "cuMemcpyDtoD_v2");
            load_function!(cu_mem_get_address_range_v2, "cuMemGetAddressRange_v2");
            load_function!(cu_memset_d8_v2, "cuMemsetD8_v2");
            load_function!(cu_memset_d32_v2, "cuMemsetD32_v2");
            load_function!(cu_mem_alloc_host_v2, "cuMemAllocHost_v2");
            load_function!(cu_mem_free_host, "cuMemFreeHost");
            load_function!(cu_mem_host_register_v2, "cuMemHostRegister_v2");
            load_function!(cu_mem_host_unregister, "cuMemHostUnregister");
            load_function!(cu_pointer_get_attribute, "cuPointerGetAttribute");
            
            load_function!(cu_module_load_data, "cuModuleLoadData");
            load_function!(cu_module_load_data_ex, "cuModuleLoadDataEx");
            load_function!(cu_module_unload, "cuModuleUnload");
            load_function!(cu_module_get_function, "cuModuleGetFunction");
            load_function!(cu_func_get_attribute, "cuFuncGetAttribute");
            load_function!(cu_launch_kernel, "cuLaunchKernel");
            
            load_function!(cu_stream_create, "cuStreamCreate");
            load_function!(cu_stream_destroy_v2, "cuStreamDestroy_v2");
            load_function!(cu_stream_synchronize, "cuStreamSynchronize");
            load_function!(cu_stream_wait_event, "cuStreamWaitEvent");
            
            load_function!(cu_event_create, "cuEventCreate");
            load_function!(cu_event_destroy_v2, "cuEventDestroy_v2");
            load_function!(cu_event_record, "cuEventRecord");
            load_function!(cu_event_synchronize, "cuEventSynchronize");
            load_function!(cu_event_elapsed_time, "cuEventElapsedTime");

            Ok(NvidiaCudaLibrary {
                handle: LibraryHandle(handle),
                functions,
            })
        }
    }
}

/// 初始化 NVIDIA 后端
pub fn initialize_nvidia_backend() -> Result<(), String> {
    // 使用读锁检查是否已初始化
    {
        let cuda_lib = NVIDIA_CUDA.read().unwrap();
        if cuda_lib.is_some() {
            return Ok(()); // 已经初始化
        }
    }
    
    // 需要初始化，获取写锁
    let mut cuda_lib = NVIDIA_CUDA.write().unwrap();
    
    // 双重检查，防止竞争条件
    if cuda_lib.is_some() {
        return Ok(());
    }

    match NvidiaCudaLibrary::try_load() {
        Ok(lib) => {
            eprintln!("[NvidiaBackend] NVIDIA 后端初始化成功");
            eprintln!("[NvidiaBackend] 将直接转发 CUDA 调用到原生库，同时保持内存跟踪功能");
            *cuda_lib = Some(Arc::new(lib));
            Ok(())
        }
        Err(e) => {
            eprintln!("[NvidiaBackend] 无法初始化 NVIDIA 后端: {}", e);
            Err(e)
        }
    }
}

/// 检查 NVIDIA 后端是否可用
pub fn is_nvidia_backend_available() -> bool {
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    cuda_lib.is_some()
}

/// 验证库句柄是否仍然有效
fn is_library_handle_valid(handle: *mut c_void) -> bool {
    if handle.is_null() {
        return false;
    }
    
    // 尝试获取一个简单的符号来测试句柄有效性
    unsafe {
        let test_symbol = libc::dlsym(handle, "cuInit\0".as_ptr() as *const i8);
        !test_symbol.is_null()
    }
}

/// 确保 CUDA 上下文已创建（支持指定设备）
fn ensure_cuda_context_for_device(device_id: Option<i32>) -> Result<(), CUresult> {
    let target_device = device_id.unwrap_or(0);
    eprintln!("[NvidiaBackend] ensure_cuda_context_for_device(device={})", target_device);
    
    // 首先确保 CUDA 已初始化
    let init_result = cuInit(0);
    if init_result != CUresult::SUCCESS {
        eprintln!("[NvidiaBackend] cuInit 失败: {:?}", init_result);
        return Err(init_result);
    }
    
    // 验证设备数量
    let mut device_count = 0i32;
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_get_count {
            let result = unsafe { func(&mut device_count) };
            if result != CUresult::SUCCESS {
                eprintln!("[NvidiaBackend] cuDeviceGetCount 失败: {:?}", result);
                return Err(result);
            }
        } else {
            eprintln!("[NvidiaBackend] cuDeviceGetCount 函数未加载");
            return Err(CUresult::ERROR_NOT_INITIALIZED);
        }
        
        if device_count == 0 {
            eprintln!("[NvidiaBackend] 没有找到 CUDA 设备");
            return Err(CUresult::ERROR_NO_DEVICE);
        }
        
        if target_device >= device_count {
            eprintln!("[NvidiaBackend] 设备 {} 不存在，总共只有 {} 个设备", target_device, device_count);
            return Err(CUresult::ERROR_INVALID_DEVICE);
        }
        
        // 使用设备管理器来处理上下文
        drop(cuda_lib); // 释放读锁
        
        let mut device_manager = DEVICE_MANAGER.lock().unwrap();
        let cuda_lib = NVIDIA_CUDA.read().unwrap();
        if let Some(ref lib) = cuda_lib.as_ref() {
            match device_manager.set_current_device(target_device, lib) {
                Ok(()) => {
                    eprintln!("[NvidiaBackend] 设备 {} 上下文设置成功", target_device);
                    Ok(())
                }
                Err(e) => {
                    eprintln!("[NvidiaBackend] 设置设备 {} 上下文失败: {:?}", target_device, e);
                    Err(e)
                }
            }
        } else {
            eprintln!("[NvidiaBackend] CUDA 库未初始化");
            Err(CUresult::ERROR_NOT_INITIALIZED)
        }
    } else {
        eprintln!("[NvidiaBackend] CUDA 库未初始化");
        Err(CUresult::ERROR_NOT_INITIALIZED)
    }
}

/// 确保 CUDA 上下文已创建（使用默认设备）
fn ensure_cuda_context() -> Result<(), CUresult> {
    ensure_cuda_context_for_device(None)
}

/// 检查并重新初始化 NVIDIA 后端（如果需要）
pub fn ensure_nvidia_backend_available() -> bool {
    // 首先用读锁检查
    {
        let cuda_lib = NVIDIA_CUDA.read().unwrap();
        if let Some(ref lib) = *cuda_lib {
            // 检查库句柄是否仍然有效
            if is_library_handle_valid(lib.handle.0) {
                return true;
            } else {
                eprintln!("[NvidiaBackend] 检测到库句柄无效，需要重新加载");
            }
        }
    }
    
    // 需要重新初始化，获取写锁
    {
        let mut cuda_lib = NVIDIA_CUDA.write().unwrap();
        // 再次检查，防止竞争条件
        if let Some(ref lib) = *cuda_lib {
            if is_library_handle_valid(lib.handle.0) {
                return true;
            }
        }
        // 清理无效状态
        *cuda_lib = None;
    }
    
    eprintln!("[NvidiaBackend] 后端需要重新初始化...");
    match initialize_nvidia_backend() {
        Ok(()) => {
            eprintln!("[NvidiaBackend] 重新初始化成功");
            true
        }
        Err(e) => {
            eprintln!("[NvidiaBackend] 重新初始化失败: {}", e);
            false
        }
    }
}

/// 安全的内存跟踪器操作
fn safe_track_alloc(address: u64, size: usize, alloc_type: AllocationType) {
    const MAX_RETRIES: u32 = 3;
    for attempt in 0..MAX_RETRIES {
        if let Ok(mut tracer) = get_simple_tracer().try_lock() {
            tracer.track_alloc_with_type(address, size, alloc_type);
            return;
        }
        // 如果锁定失败，等待一小段时间再重试
        std::thread::sleep(std::time::Duration::from_micros(100 * (attempt + 1) as u64));
    }
    eprintln!("[NvidiaBackend] 警告: 无法锁定内存跟踪器，分配跟踪失败 (addr=0x{:x}, size={})", address, size);
}

fn safe_track_free(address: u64) {
    const MAX_RETRIES: u32 = 3;
    for attempt in 0..MAX_RETRIES {
        if let Ok(mut tracer) = get_simple_tracer().try_lock() {
            tracer.track_free(address);
            return;
        }
        std::thread::sleep(std::time::Duration::from_micros(100 * (attempt + 1) as u64));
    }
    eprintln!("[NvidiaBackend] 警告: 无法锁定内存跟踪器，释放跟踪失败 (addr=0x{:x})", address);
}

fn safe_track_memory_access(address: u64, size: usize, access_type: AccessType) {
    if let Ok(mut tracer) = get_simple_tracer().try_lock() {
        tracer.track_memory_access(address, size, access_type);
    }
    // 对于内存访问跟踪，失败时不打印警告避免日志过多
}

/// 设备上下文管理函数
impl DeviceContextManager {
    fn ensure_device_context(&mut self, device_id: i32, cuda_lib: &NvidiaCudaLibrary) -> Result<CUcontext, CUresult> {
        // 如果已存在该设备的上下文，直接返回
        if let Some(ctx) = self.contexts.get(&device_id) {
            return Ok(ctx.context);
        }
        
        // 创建新的设备上下文
        let device = device_id as CUdevice;
        let mut context: CUcontext = CUcontext(ptr::null_mut());
        
        // 使用主上下文
        if let Some(func) = cuda_lib.functions.cu_device_primary_ctx_retain {
            let result = unsafe { func(&mut context, device) };
            if result != CUresult::SUCCESS {
                eprintln!("[NvidiaBackend] 为设备 {} 创建主上下文失败: {:?}", device_id, result);
                return Err(result);
            }
        } else {
            return Err(CUresult::ERROR_NOT_INITIALIZED);
        }
        
        // 保存上下文信息
        let device_ctx = DeviceContext {
            device_id: device,
            context,
            is_primary: true,
        };
        
        self.contexts.insert(device_id, device_ctx);
        self.current_device = Some(device_id);
        
        eprintln!("[NvidiaBackend] 为设备 {} 创建上下文成功", device_id);
        Ok(context)
    }
    
    fn set_current_device(&mut self, device_id: i32, cuda_lib: &NvidiaCudaLibrary) -> Result<(), CUresult> {
        // 确保设备上下文存在
        let context = self.ensure_device_context(device_id, cuda_lib)?;
        
        // 设置当前上下文
        if let Some(func) = cuda_lib.functions.cu_ctx_set_current {
            let result = unsafe { func(context) };
            if result == CUresult::SUCCESS {
                self.current_device = Some(device_id);
                Ok(())
            } else {
                Err(result)
            }
        } else {
            Err(CUresult::ERROR_NOT_INITIALIZED)
        }
    }
}

// CUDA API 转发函数实现

/// cuInit - 初始化 CUDA
pub fn cuInit(flags: u32) -> CUresult {
    eprintln!("[NvidiaBackend] cuInit(flags={})", flags);
    
    // 确保 NVIDIA 后端可用（处理重新加载情况）
    if !ensure_nvidia_backend_available() {
        eprintln!("[NvidiaBackend] 无法初始化或重新初始化 NVIDIA 后端");
        return CUresult::ERROR_NOT_INITIALIZED;
    }
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_init {
            let result = unsafe { func(flags) };
            eprintln!("[NvidiaBackend] cuInit 结果: {:?}", result);
            return result;
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuDriverGetVersion - 获取驱动版本
pub fn cuDriverGetVersion(driver_version: *mut i32) -> CUresult {
    eprintln!("[NvidiaBackend] cuDriverGetVersion()");
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_driver_get_version {
            let result = unsafe { func(driver_version) };
            if result == CUresult::SUCCESS && !driver_version.is_null() {
                let version = unsafe { *driver_version };
                eprintln!("[NvidiaBackend] CUDA 驱动版本: {}", version);
            }
            return result;
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuDeviceGetCount - 获取设备数量
pub fn cuDeviceGetCount(count: *mut i32) -> CUresult {
    eprintln!("[NvidiaBackend] cuDeviceGetCount()");
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_get_count {
            let result = unsafe { func(count) };
            if result == CUresult::SUCCESS && !count.is_null() {
                let device_count = unsafe { *count };
                eprintln!("[NvidiaBackend] 检测到 {} 个 CUDA 设备", device_count);
            }
            return result;
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuDeviceGet - 获取设备句柄
pub fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult {
    eprintln!("[NvidiaBackend] cuDeviceGet(ordinal={})", ordinal);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_get {
            return unsafe { func(device, ordinal) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuMemAlloc_v2 - 分配设备内存 (带跟踪)
pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    eprintln!("[NvidiaBackend] cuMemAlloc_v2(size={})", bytesize);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_mem_alloc_v2 {
            let result = unsafe { func(dptr, bytesize) };
            
            // 如果分配成功，添加到内存跟踪器
            if result == CUresult::SUCCESS && !dptr.is_null() {
                let address = unsafe { (*dptr).0 as u64 };
                safe_track_alloc(address, bytesize, AllocationType::Device);
                eprintln!("[NvidiaBackend] 成功分配 {} 字节在地址 0x{:x}", bytesize, address);
            }
            
            return result;
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuMemFree_v2 - 释放设备内存 (带跟踪)
pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    let address = dptr.0 as u64;
    eprintln!("[NvidiaBackend] cuMemFree_v2(ptr=0x{:x})", address);
    
    // 先更新跟踪器
    safe_track_free(address);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_mem_free_v2 {
            let result = unsafe { func(dptr) };
            if result == CUresult::SUCCESS {
                eprintln!("[NvidiaBackend] 成功释放地址 0x{:x} 的内存", address);
            }
            return result;
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuMemcpyHtoD_v2 - 主机到设备内存复制 (带跟踪)
pub fn cuMemcpyHtoD_v2(dst_device: CUdeviceptr, src_host: *const c_void, byte_count: usize) -> CUresult {
    let address = dst_device.0 as u64;
    eprintln!("[NvidiaBackend] cuMemcpyHtoD_v2(dst=0x{:x}, size={})", address, byte_count);
    
    // 跟踪内存写入
    safe_track_memory_access(address, byte_count, AccessType::Write);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_memcpy_hto_d_v2 {
            return unsafe { func(dst_device, src_host, byte_count) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuMemcpyDtoH_v2 - 设备到主机内存复制 (带跟踪)
pub fn cuMemcpyDtoH_v2(dst_host: *mut c_void, src_device: CUdeviceptr, byte_count: usize) -> CUresult {
    let address = src_device.0 as u64;
    eprintln!("[NvidiaBackend] cuMemcpyDtoH_v2(src=0x{:x}, size={})", address, byte_count);
    
    // 跟踪内存读取
    safe_track_memory_access(address, byte_count, AccessType::Read);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_memcpy_dto_h_v2 {
            return unsafe { func(dst_host, src_device, byte_count) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuMemcpyDtoD_v2 - 设备到设备内存复制 (带跟踪)
pub fn cuMemcpyDtoD_v2(dst_device: CUdeviceptr, src_device: CUdeviceptr, byte_count: usize) -> CUresult {
    let dst_address = dst_device.0 as u64;
    let src_address = src_device.0 as u64;
    eprintln!("[NvidiaBackend] cuMemcpyDtoD_v2(dst=0x{:x}, src=0x{:x}, size={})", dst_address, src_address, byte_count);
    
    // 跟踪内存复制（读源地址，写目标地址）
    track_memory_copy(dst_address, src_address, byte_count);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_memcpy_dto_d_v2 {
            return unsafe { func(dst_device, src_device, byte_count) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuMemGetAddressRange_v2 - 获取内存地址范围
pub fn cuMemGetAddressRange_v2(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult {
    let address = dptr.0 as u64;
    eprintln!("[NvidiaBackend] cuMemGetAddressRange_v2(ptr=0x{:x})", address);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_mem_get_address_range_v2 {
            return unsafe { func(pbase, psize, dptr) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuMemsetD8_v2 - 设置设备内存 (8位) 
pub fn cuMemsetD8_v2(dst_device: CUdeviceptr, uc: u8, n: usize) -> CUresult {
    let address = dst_device.0 as u64;
    eprintln!("[NvidiaBackend] cuMemsetD8_v2(dst=0x{:x}, value={}, count={})", address, uc, n);
    
    // 跟踪内存写入
    track_memory_set(address, n);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_memset_d8_v2 {
            return unsafe { func(dst_device, uc, n) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuMemsetD32_v2 - 设置设备内存 (32位)
pub fn cuMemsetD32_v2(dst_device: CUdeviceptr, ui: u32, n: usize) -> CUresult {
    let address = dst_device.0 as u64;
    eprintln!("[NvidiaBackend] cuMemsetD32_v2(dst=0x{:x}, value={}, count={})", address, ui, n);
    
    // 跟踪内存写入 
    track_memory_set(address, n * 4); // 32位 = 4字节
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_memset_d32_v2 {
            return unsafe { func(dst_device, ui, n) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuMemAllocHost_v2 - 分配主机内存
pub fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> CUresult {
    eprintln!("[NvidiaBackend] cuMemAllocHost_v2(size={})", bytesize);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_mem_alloc_host_v2 {
            let result = unsafe { func(pp, bytesize) };
            
            // 跟踪主机内存分配
            if result == CUresult::SUCCESS && !pp.is_null() {
                let ptr = unsafe { *pp };
                let address = ptr as u64;
                safe_track_alloc(address, bytesize, AllocationType::Host);
                eprintln!("[NvidiaBackend] 成功分配主机内存 {} 字节在地址 0x{:x}", bytesize, address);
            }
            
            return result;
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuMemFreeHost - 释放主机内存
pub fn cuMemFreeHost(p: *mut c_void) -> CUresult {
    let address = p as u64;
    eprintln!("[NvidiaBackend] cuMemFreeHost(ptr=0x{:x})", address);
    
    // 跟踪主机内存释放
    safe_track_free(address);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_mem_free_host {
            return unsafe { func(p) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuModuleLoadData - 加载模块
pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    eprintln!("[NvidiaBackend] cuModuleLoadData()");
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_module_load_data {
            return unsafe { func(module, image) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuModuleUnload - 卸载模块
pub fn cuModuleUnload(hmod: CUmodule) -> CUresult {
    eprintln!("[NvidiaBackend] cuModuleUnload()");
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_module_unload {
            return unsafe { func(hmod) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuModuleGetFunction - 获取函数
pub fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const i8) -> CUresult {
    let func_name = if !name.is_null() {
        unsafe { CStr::from_ptr(name).to_string_lossy().to_string() }
    } else {
        "unknown".to_string()
    };
    eprintln!("[NvidiaBackend] cuModuleGetFunction(name={})", func_name);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_module_get_function {
            return unsafe { func(hfunc, hmod, name) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuLaunchKernel - 启动内核
pub fn cuLaunchKernel(
    f: CUfunction,
    grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
    block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
    shared_mem_bytes: u32,
    h_stream: CUstream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void
) -> CUresult {
    eprintln!("[NvidiaBackend] cuLaunchKernel(grid=[{},{},{}], block=[{},{},{}], shared={})", 
             grid_dim_x, grid_dim_y, grid_dim_z,
             block_dim_x, block_dim_y, block_dim_z,
             shared_mem_bytes);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_launch_kernel {
            return unsafe { func(f, grid_dim_x, grid_dim_y, grid_dim_z,
                                block_dim_x, block_dim_y, block_dim_z,
                                shared_mem_bytes, h_stream, kernel_params, extra) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cuPointerGetAttribute - 获取指针属性
pub fn cuPointerGetAttribute(data: *mut c_void, attribute: CUpointer_attribute, ptr: CUdeviceptr) -> CUresult {
    let address = ptr.0 as u64;
    eprintln!("[NvidiaBackend] cuPointerGetAttribute(ptr=0x{:x}, attr={:?})", address, attribute);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_pointer_get_attribute {
            return unsafe { func(data, attribute, ptr) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// cudaMallocHost - CUDA Runtime API 主机内存分配
pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> CUresult {
    eprintln!("[NvidiaBackend] cudaMallocHost(size={})", size);
    
    // 确保 CUDA 已初始化并设置默认上下文
    if let Err(result) = ensure_cuda_context() {
        return result;
    }
    
    // 转发到 cuMemAllocHost_v2
    let result = cuMemAllocHost_v2(ptr, size);
    
    if result == CUresult::SUCCESS {
        eprintln!("[NvidiaBackend] cudaMallocHost 成功分配 {} 字节", size);
    } else {
        eprintln!("[NvidiaBackend] cudaMallocHost 失败: {:?}", result);
    }
    
    result
}

/// cudaFreeHost - CUDA Runtime API 主机内存释放
pub fn cudaFreeHost(ptr: *mut c_void) -> CUresult {
    eprintln!("[NvidiaBackend] cudaFreeHost(ptr=0x{:x})", ptr as u64);
    
    // 转发到 cuMemFreeHost
    let result = cuMemFreeHost(ptr);
    
    if result == CUresult::SUCCESS {
        eprintln!("[NvidiaBackend] cudaFreeHost 成功释放内存");
    } else {
        eprintln!("[NvidiaBackend] cudaFreeHost 失败: {:?}", result);
    }
    
    result
}

/// cudaMalloc - CUDA Runtime API 设备内存分配
pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> CUresult {
    eprintln!("[NvidiaBackend] cudaMalloc(size={})", size);
    
    // 确保 CUDA 已初始化并设置默认上下文
    if let Err(result) = ensure_cuda_context() {
        return result;
    }
    
    // 转换指针类型并转发到 cuMemAlloc_v2
    let result = cuMemAlloc_v2(devPtr as *mut CUdeviceptr, size);
    
    if result == CUresult::SUCCESS {
        eprintln!("[NvidiaBackend] cudaMalloc 成功分配 {} 字节设备内存", size);
    } else {
        eprintln!("[NvidiaBackend] cudaMalloc 失败: {:?}", result);
    }
    
    result
}

/// cudaFree - CUDA Runtime API 设备内存释放
pub fn cudaFree(devPtr: *mut c_void) -> CUresult {
    eprintln!("[NvidiaBackend] cudaFree(ptr=0x{:x})", devPtr as u64);
    
    // 转换指针类型并转发到 cuMemFree_v2
    let device_ptr = CUdeviceptr_v2(devPtr as *mut _);
    let result = cuMemFree_v2(device_ptr);
    
    if result == CUresult::SUCCESS {
        eprintln!("[NvidiaBackend] cudaFree 成功释放设备内存");
    } else {
        eprintln!("[NvidiaBackend] cudaFree 失败: {:?}", result);
    }
    
    result
}

/// cudaMemcpy - CUDA Runtime API 内存复制
pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> CUresult {
    eprintln!("[NvidiaBackend] cudaMemcpy(dst=0x{:x}, src=0x{:x}, size={}, kind={})", 
             dst as u64, src as u64, count, kind);
    
    // kind: 0=HostToHost, 1=HostToDevice, 2=DeviceToHost, 3=DeviceToDevice
    let result = match kind {
        1 => {
            // HostToDevice
            let device_ptr = CUdeviceptr_v2(dst as *mut _);
            cuMemcpyHtoD_v2(device_ptr, src, count)
        }
        2 => {
            // DeviceToHost
            let device_ptr = CUdeviceptr_v2(src as *mut _);
            cuMemcpyDtoH_v2(dst, device_ptr, count)
        }
        3 => {
            // DeviceToDevice
            let dst_device = CUdeviceptr_v2(dst as *mut _);
            let src_device = CUdeviceptr_v2(src as *mut _);
            cuMemcpyDtoD_v2(dst_device, src_device, count)
        }
        _ => {
            eprintln!("[NvidiaBackend] 不支持的 cudaMemcpy 类型: {}", kind);
            CUresult::ERROR_INVALID_VALUE
        }
    };
    
    if result == CUresult::SUCCESS {
        eprintln!("[NvidiaBackend] cudaMemcpy 成功复制 {} 字节", count);
    } else {
        eprintln!("[NvidiaBackend] cudaMemcpy 失败: {:?}", result);
    }
    
    result
}

// 其他 CUDA API 的转发函数可以按需添加...

/// 内存跟踪器管理函数

/// 获取内存统计信息
pub fn get_memory_statistics() -> Option<(u64, u64, u64, u64, u64)> {
    if let Ok(tracer) = get_simple_tracer().try_lock() {
        Some(tracer.get_stats())
    } else {
        None
    }
}

/// 执行内存泄漏检测
pub fn detect_memory_leaks() -> u64 {
    if let Ok(mut tracer) = get_simple_tracer().try_lock() {
        tracer.detect_memory_leaks();
        tracer.get_complete_stats().leak_count
    } else {
        0
    }
}

/// 打印内存报告
pub fn print_memory_report() {
    if let Ok(tracer) = get_simple_tracer().try_lock() {
        tracer.print_report();
    }
}

/// 导出详细内存报告到文件
pub fn export_memory_report(filename: &str) -> Result<(), std::io::Error> {
    if let Ok(tracer) = get_simple_tracer().try_lock() {
        tracer.export_report(filename)
    } else {
        Err(std::io::Error::new(std::io::ErrorKind::Other, "Unable to access memory tracer"))
    }
}

/// 获取脏页面数量
pub fn get_dirty_pages_count() -> u64 {
    if let Ok(tracer) = get_simple_tracer().try_lock() {
        tracer.get_dirty_pages_count()
    } else {
        0
    }
}

/// 设置跟踪器配置  
pub fn configure_memory_tracer(enable_dirty_tracking: bool, enable_leak_detection: bool, page_size: usize) {
    if let Ok(mut tracer) = get_simple_tracer().try_lock() {
        use crate::r#impl::simple_memory_tracer::TrackerConfig;
        let config = TrackerConfig {
            enable_dirty_tracking,
            enable_access_tracking: true,
            enable_pattern_analysis: true,
            enable_leak_detection,
            max_history_size: 10000,
            page_size,
            leak_detection_threshold: 60000, // 1 minute
            report_interval_ms: 30000, // 30 seconds
        };
        tracer.set_config(config);
        eprintln!("[NvidiaBackend] 内存跟踪器配置已更新");
    }
}

// 上下文管理函数
pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult {
    eprintln!("[NvidiaBackend] cuCtxSetCurrent()");
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_ctx_set_current {
            return unsafe { func(ctx) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuCtxSynchronize() -> CUresult {
    eprintln!("[NvidiaBackend] cuCtxSynchronize()");
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_ctx_synchronize {
            return unsafe { func() };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuCtxSetLimit(limit: CUlimit, value: usize) -> CUresult {
    eprintln!("[NvidiaBackend] cuCtxSetLimit(limit={:?}, value={})", limit, value);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_ctx_set_limit {
            return unsafe { func(limit, value) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuCtxGetLimit(pvalue: *mut usize, limit: CUlimit) -> CUresult {
    eprintln!("[NvidiaBackend] cuCtxGetLimit(limit={:?})", limit);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_ctx_get_limit {
            return unsafe { func(pvalue, limit) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

// 设备管理函数
pub fn cuDeviceGetName(name: *mut i8, len: i32, dev: CUdevice) -> CUresult {
    eprintln!("[NvidiaBackend] cuDeviceGetName(dev={})", dev);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_get_name {
            let result = unsafe { func(name, len, dev) };
            if result == CUresult::SUCCESS && !name.is_null() && len > 0 {
                let device_name = unsafe { CStr::from_ptr(name).to_string_lossy() };
                eprintln!("[NvidiaBackend] 设备名称: {}", device_name);
            }
            return result;
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuDeviceGetAttribute(pi: *mut i32, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult {
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_get_attribute {
            return unsafe { func(pi, attrib, dev) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuDeviceComputeCapability(major: *mut i32, minor: *mut i32, dev: CUdevice) -> CUresult {
    eprintln!("[NvidiaBackend] cuDeviceComputeCapability(dev={})", dev);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_compute_capability {
            let result = unsafe { func(major, minor, dev) };
            if result == CUresult::SUCCESS && !major.is_null() && !minor.is_null() {
                let maj = unsafe { *major };
                let min = unsafe { *minor };
                eprintln!("[NvidiaBackend] 计算能力: {}.{}", maj, min);
            }
            return result;
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult {
    eprintln!("[NvidiaBackend] cuDeviceTotalMem_v2(dev={})", dev);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_total_mem_v2 {
            let result = unsafe { func(bytes, dev) };
            if result == CUresult::SUCCESS && !bytes.is_null() {
                let total_mem = unsafe { *bytes };
                eprintln!("[NvidiaBackend] 设备总内存: {:.2} GB", total_mem as f64 / (1024.0 * 1024.0 * 1024.0));
            }
            return result;
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuDeviceGetProperties(prop: *mut CUdevprop, dev: CUdevice) -> CUresult {
    eprintln!("[NvidiaBackend] cuDeviceGetProperties(dev={})", dev);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_get_properties {
            return unsafe { func(prop, dev) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuDeviceGetUuid(uuid: *mut CUuuid, dev: CUdevice) -> CUresult {
    eprintln!("[NvidiaBackend] cuDeviceGetUuid(dev={})", dev);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_get_uuid {
            return unsafe { func(uuid, dev) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuDeviceGetUuid_v2(uuid: *mut CUuuid, dev: CUdevice) -> CUresult {
    eprintln!("[NvidiaBackend] cuDeviceGetUuid_v2(dev={})", dev);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_get_uuid_v2 {
            return unsafe { func(uuid, dev) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuDeviceGetLuid(luid: *mut i8, device_node_mask: *mut u32, dev: CUdevice) -> CUresult {
    eprintln!("[NvidiaBackend] cuDeviceGetLuid(dev={})", dev);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_get_luid {
            return unsafe { func(luid, device_node_mask, dev) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    eprintln!("[NvidiaBackend] cuDevicePrimaryCtxRetain(dev={})", dev);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_primary_ctx_retain {
            return unsafe { func(pctx, dev) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult {
    eprintln!("[NvidiaBackend] cuDevicePrimaryCtxRelease(dev={})", dev);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_device_primary_ctx_release {
            return unsafe { func(dev) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

pub fn cuFuncGetAttribute(pi: *mut i32, attrib: CUfunction_attribute, hfunc: CUfunction) -> CUresult {
    eprintln!("[NvidiaBackend] cuFuncGetAttribute(attr={:?})", attrib);
    
    let cuda_lib = NVIDIA_CUDA.read().unwrap();
    if let Some(ref lib) = cuda_lib.as_ref() {
        if let Some(func) = lib.functions.cu_func_get_attribute {
            return unsafe { func(pi, attrib, hfunc) };
        }
    }
    
    CUresult::ERROR_NOT_INITIALIZED
}

/// 切换到指定的CUDA设备（多卡支持）
pub fn cuDeviceSetCurrent(device: CUdevice) -> CUresult {
    eprintln!("[NvidiaBackend] cuDeviceSetCurrent(device={})", device);
    
    // 确保后端可用
    if !ensure_nvidia_backend_available() {
        return CUresult::ERROR_NOT_INITIALIZED;
    }
    
    // 使用新的设备管理系统
    match ensure_cuda_context_for_device(Some(device as i32)) {
        Ok(()) => CUresult::SUCCESS,
        Err(e) => e,
    }
}

/// 获取当前CUDA设备
pub fn cuDeviceGetCurrent(device: *mut CUdevice) -> CUresult {
    eprintln!("[NvidiaBackend] cuDeviceGetCurrent()");
    
    if device.is_null() {
        return CUresult::ERROR_INVALID_VALUE;
    }
    
    let device_manager = DEVICE_MANAGER.lock().unwrap();
    if let Some(current_device) = device_manager.current_device {
        unsafe { *device = current_device as CUdevice };
        CUresult::SUCCESS
    } else {
        // 如果没有设置当前设备，默认返回设备0
        unsafe { *device = 0 };
        CUresult::SUCCESS
    }
}

impl Drop for NvidiaCudaLibrary {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.0.is_null() {
                libc::dlclose(self.handle.0);
                eprintln!("[NvidiaBackend] CUDA 库已卸载 - 句柄将失效");
                // 注意：不要在这里清理全局状态，让 ensure_nvidia_backend_available 检测并处理
            }
        }
    }
}