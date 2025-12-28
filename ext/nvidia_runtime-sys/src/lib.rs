//! NVIDIA CUDA Runtime bindings via dynamic loading of libcuda.so
//! This crate provides direct passthrough to NVIDIA's CUDA driver API

use cuda_types::cuda::*;
use libc::{c_char, c_int, c_uint, c_void, size_t};
use std::ffi::CStr;
use std::ptr;
use std::sync::OnceLock;

// Library handle
static CUDA_LIB: OnceLock<*mut c_void> = OnceLock::new();

// Function pointer types
type CuInitFn = unsafe extern "C" fn(c_uint) -> CUresult;
type CuDeviceGetCountFn = unsafe extern "C" fn(*mut c_int) -> CUresult;
type CuDeviceGetFn = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type CuDeviceGetNameFn = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type CuDeviceTotalMemFn = unsafe extern "C" fn(*mut size_t, CUdevice) -> CUresult;
type CuDeviceGetAttributeFn = unsafe extern "C" fn(*mut c_int, CUdevice_attribute, CUdevice) -> CUresult;
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
type CuModuleLoadDataExFn = unsafe extern "C" fn(*mut CUmodule, *const c_void, c_uint, *mut CUjit_option, *mut *mut c_void) -> CUresult;
type CuModuleUnloadFn = unsafe extern "C" fn(CUmodule) -> CUresult;
type CuModuleGetFunctionFn = unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type CuModuleGetGlobalFn = unsafe extern "C" fn(*mut CUdeviceptr, *mut size_t, CUmodule, *const c_char) -> CUresult;
type CuLaunchKernelFn = unsafe extern "C" fn(
    CUfunction,
    c_uint, c_uint, c_uint,  // grid
    c_uint, c_uint, c_uint,  // block
    c_uint,                   // shared mem
    CUstream,                 // stream
    *mut *mut c_void,         // kernel params
    *mut *mut c_void,         // extra
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
type CuFuncGetAttributeFn = unsafe extern "C" fn(*mut c_int, CUfunction_attribute, CUfunction) -> CUresult;
type CuFuncSetAttributeFn = unsafe extern "C" fn(CUfunction, CUfunction_attribute, c_int) -> CUresult;
type CuDevicePrimaryCtxRetainFn = unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;
type CuDevicePrimaryCtxReleaseFn = unsafe extern "C" fn(CUdevice) -> CUresult;
type CuDevicePrimaryCtxGetStateFn = unsafe extern "C" fn(CUdevice, *mut c_uint, *mut c_int) -> CUresult;
type CuPointerGetAttributeFn = unsafe extern "C" fn(*mut c_void, CUpointer_attribute, CUdeviceptr) -> CUresult;

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
        let lib = load_cuda_library();
        if lib.is_null() {
            eprintln!("[hetGPU-nvidia] Failed to load libcuda.so");
            return NvidiaCudaFunctions::empty();
        }

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
                cuCtxSynchronize: load_fn(lib, "cuCtxSynchronize"),
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
            }
        }
    });

    Ok(())
}

fn load_cuda_library() -> *mut c_void {
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
            return lib;
        }
    }

    ptr::null_mut()
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
        }
    }
}

/// Get the loaded CUDA functions
pub fn get_cuda_funcs() -> Option<&'static NvidiaCudaFunctions> {
    CUDA_FUNCS.get()
}

// Wrapper functions that call into the real CUDA library

pub unsafe fn cuInit(flags: c_uint) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuInit {
            return f(flags);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuDeviceGetCount(count: *mut c_int) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceGetCount {
            return f(count);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceGet {
            return f(device, ordinal);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceGetName {
            return f(name, len, dev);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuDeviceTotalMem_v2(bytes: *mut size_t, dev: CUdevice) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceTotalMem_v2 {
            return f(bytes, dev);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuDeviceGetAttribute(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDeviceGetAttribute {
            return f(pi, attrib, dev);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxCreate_v2 {
            return f(pctx, flags, dev);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxDestroy_v2 {
            return f(ctx);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuCtxSynchronize() -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxSynchronize {
            return f();
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxPushCurrent_v2 {
            return f(ctx);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxPopCurrent_v2 {
            return f(pctx);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxGetCurrent {
            return f(pctx);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuCtxSetCurrent {
            return f(ctx);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: size_t) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuMemAlloc_v2 {
            return f(dptr, bytesize);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuMemFree_v2 {
            return f(dptr);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytecount: size_t) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuMemcpyHtoD_v2 {
            return f(dst, src, bytecount);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytecount: size_t) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuMemcpyDtoH_v2 {
            return f(dst, src, bytecount);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, bytecount: size_t) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuMemcpyDtoD_v2 {
            return f(dst, src, bytecount);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuModuleLoadData {
            return f(module, image);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuModuleLoadDataEx(
    module: *mut CUmodule,
    image: *const c_void,
    numOptions: c_uint,
    options: *mut CUjit_option,
    optionValues: *mut *mut c_void,
) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuModuleLoadDataEx {
            return f(module, image, numOptions, options, optionValues);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuModuleUnload(hmod: CUmodule) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuModuleUnload {
            return f(hmod);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuModuleGetFunction {
            return f(hfunc, hmod, name);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuLaunchKernel(
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
) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(func) = funcs.cuLaunchKernel {
            return func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuStreamCreate(phStream: *mut CUstream, flags: c_uint) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuStreamCreate {
            return f(phStream, flags);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuStreamDestroy_v2 {
            return f(hStream);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuStreamSynchronize(hStream: CUstream) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuStreamSynchronize {
            return f(hStream);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDriverGetVersion {
            return f(driverVersion);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDevicePrimaryCtxRetain {
            return f(pctx, dev);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}

pub unsafe fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult {
    if let Some(funcs) = get_cuda_funcs() {
        if let Some(f) = funcs.cuDevicePrimaryCtxRelease_v2 {
            return f(dev);
        }
    }
    CUresult::ERROR_NOT_INITIALIZED
}
