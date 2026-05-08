//! NVIDIA CUDA Runtime bindings via dynamic loading of libcuda.so
//! This crate provides direct passthrough to NVIDIA's CUDA driver API

use cuda_types::cuda::*;
use libc::{c_char, c_int, c_uint, c_void, size_t};
use std::ffi::CStr;
use std::ptr;
use std::sync::OnceLock;

// Thread-safe wrapper for library handle
struct LibHandle(*mut c_void);
unsafe impl Send for LibHandle {}
unsafe impl Sync for LibHandle {}

// Library handle
static CUDA_LIB: OnceLock<LibHandle> = OnceLock::new();

// Function pointer types
type CuInitFn = unsafe extern "C" fn(c_uint) -> CUresult;
type CuDeviceGetCountFn = unsafe extern "C" fn(*mut c_int) -> CUresult;
type CuDeviceGetFn = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type CuDeviceGetNameFn = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type CuDeviceTotalMemFn = unsafe extern "C" fn(*mut size_t, CUdevice) -> CUresult;
type CuDeviceGetAttributeFn =
    unsafe extern "C" fn(*mut c_int, CUdevice_attribute, CUdevice) -> CUresult;
type CuCtxCreateFn = unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
type CuCtxDestroyFn = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxSynchronizeFn = unsafe extern "C" fn() -> CUresult;
type CuCtxPushCurrentFn = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxPopCurrentFn = unsafe extern "C" fn(*mut CUcontext) -> CUresult;
type CuCtxGetCurrentFn = unsafe extern "C" fn(*mut CUcontext) -> CUresult;
type CuCtxSetCurrentFn = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuMemAllocFn = unsafe extern "C" fn(*mut CUdeviceptr, size_t) -> CUresult;
type CuMemFreeFn = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type CuMemcpyHtoDFn = unsafe extern "C" fn(CUdeviceptr, *const c_void, size_t) -> CUresult;
type CuMemcpyDtoHFn = unsafe extern "C" fn(*mut c_void, CUdeviceptr, size_t) -> CUresult;
type CuMemcpyDtoDFn = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, size_t) -> CUresult;
type CuMemsetD8Fn = unsafe extern "C" fn(CUdeviceptr, u8, size_t) -> CUresult;
type CuMemsetD32Fn = unsafe extern "C" fn(CUdeviceptr, u32, size_t) -> CUresult;
type CuModuleLoadDataFn = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type CuModuleLoadDataExFn = unsafe extern "C" fn(
    *mut CUmodule,
    *const c_void,
    c_uint,
    *mut CUjit_option,
    *mut *mut c_void,
) -> CUresult;
type CuModuleUnloadFn = unsafe extern "C" fn(CUmodule) -> CUresult;
type CuModuleGetFunctionFn =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type CuModuleGetGlobalFn =
    unsafe extern "C" fn(*mut CUdeviceptr, *mut size_t, CUmodule, *const c_char) -> CUresult;
type CuLaunchKernelFn = unsafe extern "C" fn(
    CUfunction,
    c_uint,
    c_uint,
    c_uint, // grid
    c_uint,
    c_uint,
    c_uint,           // block
    c_uint,           // shared mem
    CUstream,         // stream
    *mut *mut c_void, // kernel params
    *mut *mut c_void, // extra
) -> CUresult;
type CuStreamCreateFn = unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult;
type CuStreamDestroyFn = unsafe extern "C" fn(CUstream) -> CUresult;
type CuStreamSynchronizeFn = unsafe extern "C" fn(CUstream) -> CUresult;
type CuEventCreateFn = unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult;
type CuEventDestroyFn = unsafe extern "C" fn(CUevent) -> CUresult;
type CuEventRecordFn = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
type CuEventSynchronizeFn = unsafe extern "C" fn(CUevent) -> CUresult;
type CuEventElapsedTimeFn = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;
type CuGetErrorStringFn = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type CuGetErrorNameFn = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type CuDriverGetVersionFn = unsafe extern "C" fn(*mut c_int) -> CUresult;
type CuFuncGetAttributeFn =
    unsafe extern "C" fn(*mut c_int, CUfunction_attribute, CUfunction) -> CUresult;
type CuFuncSetAttributeFn =
    unsafe extern "C" fn(CUfunction, CUfunction_attribute, c_int) -> CUresult;
type CuDevicePrimaryCtxRetainFn = unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;
type CuDevicePrimaryCtxReleaseFn = unsafe extern "C" fn(CUdevice) -> CUresult;
type CuDevicePrimaryCtxGetStateFn =
    unsafe extern "C" fn(CUdevice, *mut c_uint, *mut c_int) -> CUresult;
type CuPointerGetAttributeFn =
    unsafe extern "C" fn(*mut c_void, CUpointer_attribute, CUdeviceptr) -> CUresult;
type CuCtxGetDeviceFn = unsafe extern "C" fn(*mut CUdevice) -> CUresult;
type CuCtxSetLimitFn = unsafe extern "C" fn(CUlimit, size_t) -> CUresult;
type CuCtxGetLimitFn = unsafe extern "C" fn(*mut size_t, CUlimit) -> CUresult;
type CuDeviceComputeCapabilityFn =
    unsafe extern "C" fn(*mut c_int, *mut c_int, CUdevice) -> CUresult;
type CuDeviceGetUuidFn = unsafe extern "C" fn(*mut CUuuid, CUdevice) -> CUresult;
type CuDeviceGetLuidFn = unsafe extern "C" fn(*mut c_char, *mut c_uint, CUdevice) -> CUresult;
type CuMemGetAddressRangeFn =
    unsafe extern "C" fn(*mut CUdeviceptr, *mut size_t, CUdeviceptr) -> CUresult;

// Function pointers struct
pub struct NvidiaCudaFunctions {
    pub cuInit: Option<CuInitFn>,
    pub cuDeviceGetCount: Option<CuDeviceGetCountFn>,
    pub cuDeviceGet: Option<CuDeviceGetFn>,
    pub cuDeviceGetName: Option<CuDeviceGetNameFn>,
    pub cuDeviceTotalMem_v2: Option<CuDeviceTotalMemFn>,
    pub cuDeviceGetAttribute: Option<CuDeviceGetAttributeFn>,
    pub cuCtxCreate_v2: Option<CuCtxCreateFn>,
    pub cuCtxDestroy_v2: Option<CuCtxDestroyFn>,
    pub cuCtxSynchronize: Option<CuCtxSynchronizeFn>,
    pub cuCtxPushCurrent_v2: Option<CuCtxPushCurrentFn>,
    pub cuCtxPopCurrent_v2: Option<CuCtxPopCurrentFn>,
    pub cuCtxGetCurrent: Option<CuCtxGetCurrentFn>,
    pub cuCtxSetCurrent: Option<CuCtxSetCurrentFn>,
    pub cuMemAlloc_v2: Option<CuMemAllocFn>,
    pub cuMemFree_v2: Option<CuMemFreeFn>,
    pub cuMemcpyHtoD_v2: Option<CuMemcpyHtoDFn>,
    pub cuMemcpyDtoH_v2: Option<CuMemcpyDtoHFn>,
    pub cuMemcpyDtoD_v2: Option<CuMemcpyDtoDFn>,
    pub cuMemsetD8_v2: Option<CuMemsetD8Fn>,
    pub cuMemsetD32_v2: Option<CuMemsetD32Fn>,
    pub cuModuleLoadData: Option<CuModuleLoadDataFn>,
    pub cuModuleLoadDataEx: Option<CuModuleLoadDataExFn>,
    pub cuModuleUnload: Option<CuModuleUnloadFn>,
    pub cuModuleGetFunction: Option<CuModuleGetFunctionFn>,
    pub cuModuleGetGlobal_v2: Option<CuModuleGetGlobalFn>,
    pub cuLaunchKernel: Option<CuLaunchKernelFn>,
    pub cuStreamCreate: Option<CuStreamCreateFn>,
    pub cuStreamDestroy_v2: Option<CuStreamDestroyFn>,
    pub cuStreamSynchronize: Option<CuStreamSynchronizeFn>,
    pub cuEventCreate: Option<CuEventCreateFn>,
    pub cuEventDestroy_v2: Option<CuEventDestroyFn>,
    pub cuEventRecord: Option<CuEventRecordFn>,
    pub cuEventSynchronize: Option<CuEventSynchronizeFn>,
    pub cuEventElapsedTime: Option<CuEventElapsedTimeFn>,
    pub cuGetErrorString: Option<CuGetErrorStringFn>,
    pub cuGetErrorName: Option<CuGetErrorNameFn>,
    pub cuDriverGetVersion: Option<CuDriverGetVersionFn>,
    pub cuFuncGetAttribute: Option<CuFuncGetAttributeFn>,
    pub cuFuncSetAttribute: Option<CuFuncSetAttributeFn>,
    pub cuDevicePrimaryCtxRetain: Option<CuDevicePrimaryCtxRetainFn>,
    pub cuDevicePrimaryCtxRelease_v2: Option<CuDevicePrimaryCtxReleaseFn>,
    pub cuDevicePrimaryCtxGetState: Option<CuDevicePrimaryCtxGetStateFn>,
    pub cuPointerGetAttribute: Option<CuPointerGetAttributeFn>,
    pub cuCtxGetDevice: Option<CuCtxGetDeviceFn>,
    pub cuCtxSetLimit: Option<CuCtxSetLimitFn>,
    pub cuCtxGetLimit: Option<CuCtxGetLimitFn>,
    pub cuDeviceComputeCapability: Option<CuDeviceComputeCapabilityFn>,
    pub cuDeviceGetUuid: Option<CuDeviceGetUuidFn>,
    pub cuDeviceGetLuid: Option<CuDeviceGetLuidFn>,
    pub cuMemGetAddressRange_v2: Option<CuMemGetAddressRangeFn>,
}

static CUDA_FUNCS: OnceLock<NvidiaCudaFunctions> = OnceLock::new();

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *mut c_char;
}

const RTLD_NOW: c_int = 0x2;
const RTLD_GLOBAL: c_int = 0x100;

/// Initialize the NVIDIA CUDA library
pub fn init() -> Result<(), String> {
    let _ = CUDA_FUNCS.get_or_init(|| {
        let lib_handle = load_cuda_library();
        let lib = lib_handle.0;
        if lib.is_null() {
            eprintln!("[hetGPU-nvidia] Failed to load libcuda.so");
            return NvidiaCudaFunctions::empty();
        }
        let _ = CUDA_LIB.set(lib_handle);

        unsafe {
            NvidiaCudaFunctions {
                cuInit: load_fn(lib, "cuInit"),
                cuDeviceGetCount: load_fn(lib, "cuDeviceGetCount"),
                cuDeviceGet: load_fn(lib, "cuDeviceGet"),
                cuDeviceGetName: load_fn(lib, "cuDeviceGetName"),
                cuDeviceTotalMem_v2: load_fn(lib, "cuDeviceTotalMem_v2"),
                cuDeviceGetAttribute: load_fn(lib, "cuDeviceGetAttribute"),
                cuCtxCreate_v2: load_fn(lib, "cuCtxCreate_v2"),
                cuCtxDestroy_v2: load_fn(lib, "cuCtxDestroy_v2"),
                cuCtxSynchronize: {
                    let f = load_fn(lib, "cuCtxSynchronize");
                    eprintln!("[nvidia-sys] loaded cuCtxSynchronize: {:?}", f.is_some());
                    f
                },
                cuCtxPushCurrent_v2: load_fn(lib, "cuCtxPushCurrent_v2"),
                cuCtxPopCurrent_v2: load_fn(lib, "cuCtxPopCurrent_v2"),
                cuCtxGetCurrent: load_fn(lib, "cuCtxGetCurrent"),
                cuCtxSetCurrent: load_fn(lib, "cuCtxSetCurrent"),
                cuMemAlloc_v2: load_fn(lib, "cuMemAlloc_v2"),
                cuMemFree_v2: load_fn(lib, "cuMemFree_v2"),
                cuMemcpyHtoD_v2: load_fn(lib, "cuMemcpyHtoD_v2"),
                cuMemcpyDtoH_v2: load_fn(lib, "cuMemcpyDtoH_v2"),
                cuMemcpyDtoD_v2: load_fn(lib, "cuMemcpyDtoD_v2"),
                cuMemsetD8_v2: load_fn(lib, "cuMemsetD8_v2"),
                cuMemsetD32_v2: load_fn(lib, "cuMemsetD32_v2"),
                cuModuleLoadData: load_fn(lib, "cuModuleLoadData"),
                cuModuleLoadDataEx: load_fn(lib, "cuModuleLoadDataEx"),
                cuModuleUnload: load_fn(lib, "cuModuleUnload"),
                cuModuleGetFunction: load_fn(lib, "cuModuleGetFunction"),
                cuModuleGetGlobal_v2: load_fn(lib, "cuModuleGetGlobal_v2"),
                cuLaunchKernel: load_fn(lib, "cuLaunchKernel"),
                cuStreamCreate: load_fn(lib, "cuStreamCreate"),
                cuStreamDestroy_v2: load_fn(lib, "cuStreamDestroy_v2"),
                cuStreamSynchronize: load_fn(lib, "cuStreamSynchronize"),
                cuEventCreate: load_fn(lib, "cuEventCreate"),
                cuEventDestroy_v2: load_fn(lib, "cuEventDestroy_v2"),
                cuEventRecord: load_fn(lib, "cuEventRecord"),
                cuEventSynchronize: load_fn(lib, "cuEventSynchronize"),
                cuEventElapsedTime: load_fn(lib, "cuEventElapsedTime"),
                cuGetErrorString: load_fn(lib, "cuGetErrorString"),
                cuGetErrorName: load_fn(lib, "cuGetErrorName"),
                cuDriverGetVersion: load_fn(lib, "cuDriverGetVersion"),
                cuFuncGetAttribute: load_fn(lib, "cuFuncGetAttribute"),
                cuFuncSetAttribute: load_fn(lib, "cuFuncSetAttribute"),
                cuDevicePrimaryCtxRetain: load_fn(lib, "cuDevicePrimaryCtxRetain"),
                cuDevicePrimaryCtxRelease_v2: load_fn(lib, "cuDevicePrimaryCtxRelease_v2"),
                cuDevicePrimaryCtxGetState: load_fn(lib, "cuDevicePrimaryCtxGetState"),
                cuPointerGetAttribute: load_fn(lib, "cuPointerGetAttribute"),
                cuCtxGetDevice: load_fn(lib, "cuCtxGetDevice"),
                cuCtxSetLimit: load_fn(lib, "cuCtxSetLimit"),
                cuCtxGetLimit: load_fn(lib, "cuCtxGetLimit"),
                cuDeviceComputeCapability: load_fn(lib, "cuDeviceComputeCapability"),
                cuDeviceGetUuid: load_fn(lib, "cuDeviceGetUuid"),
                cuDeviceGetLuid: load_fn(lib, "cuDeviceGetLuid"),
                cuMemGetAddressRange_v2: load_fn(lib, "cuMemGetAddressRange_v2"),
            }
        }
    });

    Ok(())
}

fn load_cuda_library() -> LibHandle {
    let paths = [
        b"/usr/lib/x86_64-linux-gnu/libcuda.so\0".as_ptr() as *const c_char,
        b"/usr/lib64/libcuda.so\0".as_ptr() as *const c_char,
        b"/usr/local/cuda/lib64/libcuda.so\0".as_ptr() as *const c_char,
        b"libcuda.so.1\0".as_ptr() as *const c_char,
        b"libcuda.so\0".as_ptr() as *const c_char,
    ];

    for path in paths {
        let lib = unsafe { dlopen(path, RTLD_NOW | RTLD_GLOBAL) };
        if !lib.is_null() {
            let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
            eprintln!("[hetGPU-nvidia] Loaded CUDA library from: {}", path_str);
            return LibHandle(lib);
        }
    }

    LibHandle(ptr::null_mut())
}

unsafe fn load_fn<T>(lib: *mut c_void, name: &str) -> Option<T> {
    let name_cstr = std::ffi::CString::new(name).ok()?;
    let ptr = dlsym(lib, name_cstr.as_ptr());
    if ptr.is_null() {
        None
    } else {
        Some(std::mem::transmute_copy(&ptr))
    }
}

impl NvidiaCudaFunctions {
    fn empty() -> Self {
        Self {
            cuInit: None,
            cuDeviceGetCount: None,
            cuDeviceGet: None,
            cuDeviceGetName: None,
            cuDeviceTotalMem_v2: None,
            cuDeviceGetAttribute: None,
            cuCtxCreate_v2: None,
            cuCtxDestroy_v2: None,
            cuCtxSynchronize: None,
            cuCtxPushCurrent_v2: None,
            cuCtxPopCurrent_v2: None,
            cuCtxGetCurrent: None,
            cuCtxSetCurrent: None,
            cuMemAlloc_v2: None,
            cuMemFree_v2: None,
            cuMemcpyHtoD_v2: None,
            cuMemcpyDtoH_v2: None,
            cuMemcpyDtoD_v2: None,
            cuMemsetD8_v2: None,
            cuMemsetD32_v2: None,
            cuModuleLoadData: None,
            cuModuleLoadDataEx: None,
            cuModuleUnload: None,
            cuModuleGetFunction: None,
            cuModuleGetGlobal_v2: None,
            cuLaunchKernel: None,
            cuStreamCreate: None,
            cuStreamDestroy_v2: None,
            cuStreamSynchronize: None,
            cuEventCreate: None,
            cuEventDestroy_v2: None,
            cuEventRecord: None,
            cuEventSynchronize: None,
            cuEventElapsedTime: None,
            cuGetErrorString: None,
            cuGetErrorName: None,
            cuDriverGetVersion: None,
            cuFuncGetAttribute: None,
            cuFuncSetAttribute: None,
            cuDevicePrimaryCtxRetain: None,
            cuDevicePrimaryCtxRelease_v2: None,
            cuDevicePrimaryCtxGetState: None,
            cuPointerGetAttribute: None,
            cuCtxGetDevice: None,
            cuCtxSetLimit: None,
            cuCtxGetLimit: None,
            cuDeviceComputeCapability: None,
            cuDeviceGetUuid: None,
            cuDeviceGetLuid: None,
            cuMemGetAddressRange_v2: None,
        }
    }
}

/// Get the loaded CUDA functions
pub fn get_cuda_funcs() -> Option<&'static NvidiaCudaFunctions> {
    CUDA_FUNCS.get()
}

// Wrapper functions that call into the real CUDA library

pub fn cuInit(flags: c_uint) -> i32 {
    let _ = init();
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuInit {
            let result = unsafe { f(flags) };
            return cuda_result_to_int(result);
        }
    }
    999 // Not initialized error code
}

fn cuda_result_to_int(result: CUresult) -> i32 {
    match result {
        Ok(()) => 0,
        Err(e) => e.0.get() as i32, // Return the actual error code
    }
}

pub fn cuDeviceGetCount(count: *mut c_int) -> i32 {
    eprintln!("[hetGPU-nvidia] cuDeviceGetCount called");
    if let Some(funcs) = get_cuda_funcs() {
        eprintln!("[hetGPU-nvidia] got cuda funcs");
        if let Some(f) = funcs.cuDeviceGetCount {
            eprintln!("[hetGPU-nvidia] calling real cuDeviceGetCount");
            let result = unsafe { f(count) };
            let count_val = unsafe { *count };
            eprintln!(
                "[hetGPU-nvidia] cuDeviceGetCount returned {:?}, count={}",
                result, count_val
            );
            return cuda_result_to_int(result);
        } else {
            eprintln!("[hetGPU-nvidia] cuDeviceGetCount function not loaded");
        }
    } else {
        eprintln!("[hetGPU-nvidia] CUDA funcs not initialized");
    }
    999
}

pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceGet {
            let result = unsafe { f(device, ordinal) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceGetName {
            let result = unsafe { f(name, len, dev) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuDeviceTotalMem_v2(bytes: *mut size_t, dev: CUdevice) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceTotalMem_v2 {
            let result = unsafe { f(bytes, dev) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuDeviceGetAttribute(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceGetAttribute {
            let result = unsafe { f(pi, attrib, dev) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDevicePrimaryCtxRetain {
            let result = unsafe { f(pctx, dev) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDevicePrimaryCtxRelease_v2 {
            let result = unsafe { f(dev) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDevicePrimaryCtxGetState {
            let result = unsafe { f(dev, flags, active) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuCtxGetDevice(dev: *mut CUdevice) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxGetDevice {
            let result = unsafe { f(dev) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuCtxSetLimit(limit: CUlimit, value: size_t) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxSetLimit {
            let result = unsafe { f(limit, value) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuCtxGetLimit(pvalue: *mut size_t, limit: CUlimit) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxGetLimit {
            let result = unsafe { f(pvalue, limit) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> i32 {
    eprintln!(
        "[hetGPU-nvidia] cuCtxCreate_v2 called: flags={}, dev={}",
        flags, dev
    );
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxCreate_v2 {
            let result = unsafe { f(pctx, flags, dev) };
            if !pctx.is_null() {
                let ctx_val = unsafe { (*pctx).0 };
                eprintln!(
                    "[hetGPU-nvidia] cuCtxCreate_v2 result: {:?}, ctx={:?}",
                    result, ctx_val
                );
            } else {
                eprintln!(
                    "[hetGPU-nvidia] cuCtxCreate_v2 result: {:?}, pctx was null",
                    result
                );
            }
            return cuda_result_to_int(result);
        } else {
            eprintln!("[hetGPU-nvidia] cuCtxCreate_v2 function not loaded");
        }
    } else {
        eprintln!("[hetGPU-nvidia] CUDA funcs not initialized");
    }
    999
}

pub fn cuCtxCreate_v3(
    pctx: *mut CUcontext,
    params: *mut c_void,
    num_params: c_int,
    flags: c_uint,
    dev: CUdevice,
) -> i32 {
    eprintln!("[hetGPU-nvidia] cuCtxCreate_v3 called, falling back to v2");
    cuCtxCreate_v2(pctx, flags, dev)
}

pub fn cuCtxPushCurrent_v2(ctx: CUcontext) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxPushCurrent_v2 {
            let result = unsafe { f(ctx) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxPopCurrent_v2 {
            let result = unsafe { f(pctx) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxGetCurrent {
            let result = unsafe { f(pctx) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuDeviceComputeCapability(major: *mut c_int, minor: *mut c_int, dev: CUdevice) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceComputeCapability {
            let result = unsafe { f(major, minor, dev) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuDeviceGetUuid(uuid: *mut CUuuid, dev: CUdevice) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceGetUuid {
            let result = unsafe { f(uuid, dev) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuDeviceGetLuid(luid: *mut c_char, device_node_mask: *mut c_uint, dev: CUdevice) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceGetLuid {
            let result = unsafe { f(luid, device_node_mask, dev) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: size_t) -> i32 {
    eprintln!("[nvidia-sys] cuMemAlloc_v2 called: bytesize={}", bytesize);
    if let Some(funcs) = get_cuda_funcs() {
        eprintln!("[nvidia-sys] got cuda funcs");
        if let Some(f) = funcs.cuMemAlloc_v2 {
            eprintln!("[nvidia-sys] calling real cuMemAlloc_v2");
            let result = unsafe { f(dptr, bytesize) };
            let int_result = cuda_result_to_int(result);
            if int_result != 0 {
                eprintln!(
                    "[nvidia-sys] cuMemAlloc_v2 FAILED: result={:?}, int_result={}",
                    result, int_result
                );
            } else if !dptr.is_null() {
                let ptr_val = unsafe { *dptr };
                eprintln!("[nvidia-sys] cuMemAlloc_v2 OK: ptr={:?}", ptr_val);
            }
            return int_result;
        } else {
            eprintln!("[nvidia-sys] cuMemAlloc_v2 function NOT LOADED");
        }
    } else {
        eprintln!("[nvidia-sys] CUDA funcs not initialized");
    }
    999
}

pub fn cuMemFree_v2(dptr: CUdeviceptr) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuMemFree_v2 {
            let result = unsafe { f(dptr) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytecount: size_t) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuMemcpyHtoD_v2 {
            let result = unsafe { f(dst, src, bytecount) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytecount: size_t) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuMemcpyDtoH_v2 {
            let result = unsafe { f(dst, src, bytecount) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuMemGetAddressRange_v2(
    pbase: *mut CUdeviceptr,
    psize: *mut size_t,
    dptr: CUdeviceptr,
) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuMemGetAddressRange_v2 {
            let result = unsafe { f(pbase, psize, dptr) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuMemsetD8_v2(dst: CUdeviceptr, uc: u8, n: size_t) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuMemsetD8_v2 {
            let result = unsafe { f(dst, uc, n) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuMemsetD32_v2(dst: CUdeviceptr, ui: u32, n: size_t) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuMemsetD32_v2 {
            let result = unsafe { f(dst, ui, n) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuModuleLoadData {
            let result = unsafe { f(module, image) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuModuleUnload(hmod: CUmodule) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuModuleUnload {
            let result = unsafe { f(hmod) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuModuleGetFunction {
            let result = unsafe { f(hfunc, hmod, name) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuFuncGetAttribute(pi: *mut c_int, attrib: CUfunction_attribute, hfunc: CUfunction) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuFuncGetAttribute {
            let result = unsafe { f(pi, attrib, hfunc) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    kernelParams: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(func) = funcs.cuLaunchKernel {
            let result = unsafe {
                func(
                    f,
                    gridDimX,
                    gridDimY,
                    gridDimZ,
                    blockDimX,
                    blockDimY,
                    blockDimZ,
                    sharedMemBytes,
                    hStream,
                    kernelParams,
                    extra,
                )
            };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuCtxSynchronize() -> i32 {
    eprintln!("[nvidia-sys] cuCtxSynchronize called");
    if let Some(funcs) = get_cuda_funcs() {
        eprintln!("[nvidia-sys] cuCtxSynchronize: got cuda funcs");
        if let Some(f) = funcs.cuCtxSynchronize {
            eprintln!("[nvidia-sys] cuCtxSynchronize: calling real function");
            let result = unsafe { f() };
            let ret = cuda_result_to_int(result);
            eprintln!("[nvidia-sys] cuCtxSynchronize returned: {}", ret);
            return ret;
        } else {
            eprintln!("[nvidia-sys] cuCtxSynchronize: function pointer is None!");
        }
    } else {
        eprintln!("[nvidia-sys] cuCtxSynchronize: get_cuda_funcs returned None!");
    }
    999
}

pub fn cuCtxSetCurrent(ctx: CUcontext) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxSetCurrent {
            let result = unsafe { f(ctx) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuCtxDestroy_v2(ctx: CUcontext) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxDestroy_v2 {
            let result = unsafe { f(ctx) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuDriverGetVersion(version: *mut c_int) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDriverGetVersion {
            let result = unsafe { f(version) };
            return cuda_result_to_int(result);
        }
    }
    999
}

pub fn cuPointerGetAttribute(
    data: *mut c_void,
    attrib: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> i32 {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuPointerGetAttribute {
            let result = unsafe { f(data, attrib, ptr) };
            return cuda_result_to_int(result);
        }
    }
    999
}
