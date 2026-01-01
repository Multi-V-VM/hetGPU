pub(crate) mod r#impl;
// Import necessary for FromCuda
use crate::r#impl::FromCuda;
// Import std::ptr for null_mut
use std::ptr;
// Import Ze types
#[cfg(feature = "intel")]
use ze_runtime_sys::ze_device_handle_t;
// Import CUerror for Result
use cuda_types::cuda::CUerror;
// Define Result type to match FromCuda error return type
type Result<T> = std::result::Result<T, CUerror>;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

#[cfg(unix)]
use libc::{dlsym, RTLD_DEFAULT};

// Note: The cudart shim symbols are force-linked via --whole-archive in build.rs,
// so we don't need an explicit anchor function here.

#[no_mangle]
pub unsafe extern "C" fn cudaDriverGetVersion(driver_version: *mut i32) -> i32 {
    if driver_version.is_null() {
        return 1; // cudaErrorInvalidValue
    }

    *driver_version = std::cmp::max(cuda_types::cuda::CUDA_VERSION as i32, 13000);
    0
}

// Note: cuGetProcAddress/cuGetProcAddress_v2/cuGetExportTable are provided
// via cuda_function_declarations! and implemented in r#impl::driver.

#[no_mangle]
pub unsafe extern "C" fn cudaGetDeviceCount(count: *mut i32) -> i32 {
    eprintln!("[hetGPU] cudaGetDeviceCount called");
    if count.is_null() {
        eprintln!("[hetGPU] cudaGetDeviceCount: count is null");
        return 1; // cudaErrorInvalidValue
    }

    match crate::r#impl::driver::init(0) {
        Ok(_) => eprintln!("[hetGPU] cudaGetDeviceCount: init succeeded"),
        Err(e) => {
            eprintln!("[hetGPU] cudaGetDeviceCount: init failed with {:?}", e);
            *count = 0;
            return 0;
        }
    }

    *count = crate::r#impl::driver::global_state()
        .map(|state| {
            eprintln!("[hetGPU] cudaGetDeviceCount: {} devices", state.devices.len());
            state.devices.len() as i32
        })
        .unwrap_or(0);
    0
}

// Get device handle by index using the ordinal-to-handle mapping from driver.rs
#[cfg(feature = "intel")]
fn get_device_handle_by_index(index: usize) -> Result<ze_device_handle_t> {
    crate::r#impl::driver::get_ze_handle_by_ordinal(index as i32)
}

// Fix implementation of FromCuda for ze_device_handle_t
#[cfg(feature = "intel")]
impl FromCuda<'_, *mut i32> for *mut ze_device_handle_t {
    fn from_cuda(_: &*mut i32) -> Result<Self> {
        // Simplified implementation - just a placeholder
        Ok(ptr::null_mut())
    }
}

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                // Always log unimplemented calls to help debug initialization issues
                eprintln!("[hetGPU] Unimplemented CUDA call: {}", stringify!($fn_name));
                crate::r#impl::unimplemented()
            }
        )*
    };
}

#[cfg(feature = "amd")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id).unwrap()),*).unwrap();
                Ok(())
            }
        )*
    };
}
#[cfg(feature = "intel")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                // Debug log for all implemented function calls
                eprintln!("[hetGPU] {} called", stringify!($fn_name));

                // Convert arguments with error handling
                let result = (|| -> std::result::Result<_, CUerror> {
                    let backend_ret = cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name )($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                    Ok(crate::r#impl::into_cu_result(backend_ret))
                })();

                match result {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[hetGPU] {} failed with conversion error: {:?}", stringify!($fn_name), e);
                        Err(e)
                    }
                }
            }
        )*
    };
}
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, i32> for ze_device_handle_t {
    fn from_cuda(cuda_value: &'a i32) -> Result<Self> {
        // Logic to convert i32 to ze_device_handle_t
        if *cuda_value < 0 {
            return Err(CUerror::INVALID_VALUE); // Return an error, not CUresult
        }

        // Get device handle by index
        let device_handle = get_device_handle_by_index(*cuda_value as usize)?;
        Ok(device_handle)
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, cuda_types::cuda::CUdeviceptr_v2> for cuda_types::cuda::CUdeviceptr_v2 {
    fn from_cuda(cuda_value: &'a cuda_types::cuda::CUdeviceptr_v2) -> Result<Self> {
        // Logic to validate CUdeviceptr_v2
        if unsafe { cuda_value.0 as i64 } < 0 {
            return Err(CUerror::INVALID_HANDLE); // Return an error, not CUresult
        }

        Ok(*cuda_value)
    }
}
#[cfg(feature = "amd")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id).unwrap()),*).unwrap();
                Ok(())
            }
        )*
    };
}

#[cfg(feature = "intel")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                let backend_ret = cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name )($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                crate::r#impl::into_cu_result(backend_ret)
            }
        )*
    };
}

#[cfg(feature = "tenstorrent")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
            }
        )*
    };
}

#[cfg(feature = "tenstorrent")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
            }
        )*
    };
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
            }
        )*
    };
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
            }
        )*
    };
}

// NVIDIA backend macros - pass through to real libcuda.so
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)
            }
        )*
    };
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)
            }
        )*
    };
}

cuda_base::cuda_function_declarations!(
    unimplemented,
    implemented
        <= [
            cuCtxCreate_v2,
            cuCtxDestroy_v2,
            cuCtxGetLimit,
            cuCtxSetCurrent,
            cuCtxSetLimit,
            cuCtxSynchronize,
            cuCtxGetCurrent,
            cuCtxGetDevice,
            cuCtxPushCurrent_v2,
            cuCtxPopCurrent_v2,
            cuDeviceComputeCapability,
            cuDeviceGet,
            cuDeviceGetAttribute,
            cuDeviceGetCount,
            cuDeviceGetLuid,
            cuDeviceGetName,
            cuDevicePrimaryCtxRelease,
            cuDevicePrimaryCtxRetain,
            cuDevicePrimaryCtxGetState,
            cuDeviceGetProperties,
            cuDeviceGetUuid,
            cuDeviceGetUuid_v2,
            cuDeviceTotalMem_v2,
            cuDriverGetVersion,
            cuFuncGetAttribute,
            cuInit,
            cuMemAlloc_v2,
            cuMemFree_v2,
            cuMemcpyDtoH_v2,
            cuMemcpyHtoD_v2,
            cuModuleGetFunction,
            cuModuleGetLoadingMode,
            cuModuleLoadData,
            cuModuleUnload,
            cuPointerGetAttribute,
            cuMemGetAddressRange_v2,
            cuMemsetD32_v2,
            cuMemsetD8_v2,
            // Provide custom implementations in r#impl::driver
            cuGetProcAddress,
            cuGetProcAddress_v2,
            cuGetExportTable,
            cuGetErrorString,
            cuGetErrorName
        ],
    implemented_in_function <= [cuLaunchKernel, cuLaunchKernelEx,]
);
// cuGetErrorString/cuGetErrorName are implemented via r#impl::driver
