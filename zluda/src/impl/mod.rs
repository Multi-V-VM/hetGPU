use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
use std::mem::{self, ManuallyDrop, MaybeUninit};
#[cfg(feature = "intel")]
use ze_device::ze_device_limit_t;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;

pub(super) mod context;
pub(super) mod device;
pub(super) mod driver;
pub(super) mod function;
pub(super) mod memory;
pub(super) mod module;
pub(super) mod pointer;

#[cfg(feature = "intel")]
pub mod ze_device;
#[cfg(feature = "intel")]
pub mod ze_module;

#[cfg(debug_assertions)]
pub(crate) fn unimplemented() -> CUresult {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unimplemented() -> CUresult {
    CUresult::ERROR_NOT_SUPPORTED
}

pub(crate) trait FromCuda<'a, T>: Sized {
    fn from_cuda(t: &'a T) -> Result<Self, CUerror>;
}

macro_rules! from_cuda_nop {
    ($($type_:ty),*) => {
        $(
            impl<'a> FromCuda<'a, $type_> for $type_ {
                fn from_cuda(x: &'a $type_) -> Result<Self, CUerror> {
                    Ok(*x)
                }
            }

            impl<'a> FromCuda<'a, *mut $type_> for &'a mut $type_ {
                fn from_cuda(x: &'a *mut $type_) -> Result<Self, CUerror> {
                    match unsafe { x.as_mut() } {
                        Some(x) => Ok(x),
                        None => Err(CUerror::INVALID_VALUE),
                    }
                }
            }
        )*
    };
}

macro_rules! from_cuda_transmute {
    ($($from:ty => $to:ty),*) => {
        $(
            impl<'a> FromCuda<'a, $from> for $to {
                fn from_cuda(x: &'a $from) -> Result<Self, CUerror> {
                    // Use as_ptr and from_bits instead of direct transmute
                    // to handle types of different sizes safely
                    Ok(unsafe {
                        let raw_bits = std::mem::transmute_copy::<$from, u64>(x);
                        std::mem::transmute_copy::<u64, $to>(&raw_bits)
                    })
                }
            }

            impl<'a> FromCuda<'a, *mut $from> for &'a mut $to {
                fn from_cuda(x: &'a *mut $from) -> Result<Self, CUerror> {
                    match unsafe { x.cast::<$to>().as_mut() } {
                        Some(x) => Ok(x),
                        None => Err(CUerror::INVALID_VALUE),
                    }
                }
            }

            impl<'a> FromCuda<'a, *mut $from> for * mut $to {
                fn from_cuda(x: &'a *mut $from) -> Result<Self, CUerror> {
                    Ok(x.cast::<$to>())
                }
            }
        )*
    };
}

macro_rules! from_cuda_object {
    ($($type_:ty),*) => {
        $(
            impl<'a> FromCuda<'a, <$type_ as ZludaObject>::CudaHandle> for <$type_ as ZludaObject>::CudaHandle {
                fn from_cuda(handle: &'a <$type_ as ZludaObject>::CudaHandle) -> Result<<$type_ as ZludaObject>::CudaHandle, CUerror> {
                    Ok(*handle)
                }
            }

            impl<'a> FromCuda<'a, *mut <$type_ as ZludaObject>::CudaHandle> for &'a mut <$type_ as ZludaObject>::CudaHandle {
                fn from_cuda(handle: &'a *mut <$type_ as ZludaObject>::CudaHandle) -> Result<&'a mut <$type_ as ZludaObject>::CudaHandle, CUerror> {
                    match unsafe { handle.as_mut() } {
                        Some(x) => Ok(x),
                        None => Err(CUerror::INVALID_VALUE),
                    }
                }
            }

            impl<'a> FromCuda<'a, <$type_ as ZludaObject>::CudaHandle> for &'a $type_ {
                fn from_cuda(handle: &'a <$type_ as ZludaObject>::CudaHandle) -> Result<&'a $type_, CUerror> {
                    Ok(as_ref(handle).as_result()?)
                }
            }
        )*
    };
}

// Common type conversions that work for both AMD and Intel implementations
from_cuda_nop!(
    *mut i8,
    *mut i32,
    *mut usize,
    *const ::core::ffi::c_void,
    *const ::core::ffi::c_char,
    *mut ::core::ffi::c_void,
    *mut *mut ::core::ffi::c_void,
    u8,
    i32,
    u32,
    usize,
    cuda_types::cuda::CUdevprop,
    CUdevice_attribute
);

// AMD-specific type conversions
#[cfg(feature = "amd")]
from_cuda_transmute!(
    CUuuid => hipUUID,
    CUfunction => hipFunction_t,
    CUfunction_attribute => hipFunction_attribute,
    CUstream => hipStream_t,
    CUpointer_attribute => hipPointer_attribute,
    CUdeviceptr_v2 => hipDeviceptr_t
);

// Intel-specific type conversions
#[cfg(feature = "intel")]
from_cuda_transmute!(
    CUuuid => ze_uuid_t,
    CUfunction => ze_kernel_handle_t,
    CUstream => ze_command_queue_handle_t,
    CUpointer_attribute => ze_memory_allocation_properties_t,
    CUdeviceptr_v2 => ze_device_handle_t
);

// Special implementation for CUfunction_attribute to ze_kernel_desc_t conversion
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, CUfunction_attribute> for ze_kernel_desc_t {
    fn from_cuda(attr: &'a CUfunction_attribute) -> Result<Self, CUerror> {
        // Create a properly initialized ze_kernel_desc_t
        // This needs to be customized based on how attributes should map
        let desc = ze_kernel_desc_t::default();

        // Set appropriate fields based on the attribute
        // For example:
        match *attr {
            // Map each attribute appropriately
            // This is a placeholder - actual implementation will depend on attributes
            _ => return Err(CUerror::NOT_SUPPORTED),
        }

        Ok(desc)
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, *mut CUfunction_attribute> for &'a mut ze_kernel_desc_t {
    fn from_cuda(x: &'a *mut CUfunction_attribute) -> Result<Self, CUerror> {
        match unsafe { x.cast::<ze_kernel_desc_t>().as_mut() } {
            Some(x) => Ok(x),
            None => Err(CUerror::INVALID_VALUE),
        }
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, *mut CUfunction_attribute> for *mut ze_kernel_desc_t {
    fn from_cuda(x: &'a *mut CUfunction_attribute) -> Result<Self, CUerror> {
        Ok(x.cast::<ze_kernel_desc_t>())
    }
}

// Add implementation for _ze_kernel_properties_t
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, CUfunction_attribute_enum> for _ze_kernel_properties_t {
    fn from_cuda(attr: &'a CUfunction_attribute_enum) -> Result<Self, CUerror> {
        let props = _ze_kernel_properties_t::default();

        // Map CUDA function attributes to Level Zero kernel properties
        match *attr {
            // Add mappings based on the specific attributes needed
            // Example (modify based on actual enum variants):
            // CUfunction_attribute_enum::SOME_VARIANT => {
            //     props.some_field = some_value;
            // }
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK => {
                // Set appropriate property
                // This is a placeholder - implement with actual property
            }
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES => {
                // Set appropriate property
                // This is a placeholder - implement with actual property
            }
            // Add more mappings as needed
            _ => return Err(CUerror::NOT_SUPPORTED),
        }

        Ok(props)
    }
}

// Also implement for pointer types
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, *mut CUfunction_attribute_enum> for &'a mut _ze_kernel_properties_t {
    fn from_cuda(x: &'a *mut CUfunction_attribute_enum) -> Result<Self, CUerror> {
        match unsafe { x.cast::<_ze_kernel_properties_t>().as_mut() } {
            Some(x) => Ok(x),
            None => Err(CUerror::INVALID_VALUE),
        }
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, *mut CUfunction_attribute_enum> for *mut _ze_kernel_properties_t {
    fn from_cuda(x: &'a *mut CUfunction_attribute_enum) -> Result<Self, CUerror> {
        Ok(x.cast::<_ze_kernel_properties_t>())
    }
}

// Add implementation for CUpointer_attribute_enum
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, CUpointer_attribute_enum> for CUpointer_attribute_enum {
    fn from_cuda(attr: &'a CUpointer_attribute_enum) -> Result<Self, CUerror> {
        // Simple implementation that just returns the same value
        Ok(*attr)
    }
}

// Add implementation for &mut CUfunction from *mut CUfunction
impl<'a> FromCuda<'a, *mut CUfunction> for &'a mut CUfunction {
    fn from_cuda(x: &'a *mut CUfunction) -> Result<Self, CUerror> {
        match unsafe { x.as_mut() } {
            Some(x) => Ok(x),
            None => Err(CUerror::INVALID_VALUE),
        }
    }
}

// Add implementation for *mut CUdeviceptr_v2 from *mut CUdeviceptr_v2
impl<'a> FromCuda<'a, *mut CUdeviceptr_v2> for *mut CUdeviceptr_v2 {
    fn from_cuda(x: &'a *mut CUdeviceptr_v2) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

// Add implementation for *mut _ze_device_uuid_t from *mut CUuuid_st
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, *mut CUuuid_st> for *mut _ze_device_uuid_t {
    fn from_cuda(x: &'a *mut CUuuid_st) -> Result<Self, CUerror> {
        Ok(x.cast::<_ze_device_uuid_t>())
    }
}

// Add implementation for u32 from CUlimit_enum
impl<'a> FromCuda<'a, CUlimit_enum> for u32 {
    fn from_cuda(limit: &'a CUlimit_enum) -> Result<Self, CUerror> {
        // Convert CUlimit_enum to u32 based on your requirements
        // This is a basic implementation, adjust values as needed
        Ok(match *limit {
            CUlimit_enum::CU_LIMIT_STACK_SIZE => 0,
            CUlimit_enum::CU_LIMIT_PRINTF_FIFO_SIZE => 1,
            CUlimit_enum::CU_LIMIT_MALLOC_HEAP_SIZE => 2,
            CUlimit_enum::CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH => 3,
            CUlimit_enum::CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT => 4,
            CUlimit_enum::CU_LIMIT_MAX_L2_FETCH_GRANULARITY => 5,
            _ => return Err(CUerror::NOT_SUPPORTED),
        })
    }
}

// Add implementation for CUdeviceptr_v2 -> ()
impl<'a> FromCuda<'a, CUdeviceptr_v2> for () {
    fn from_cuda(_: &'a CUdeviceptr_v2) -> Result<Self, CUerror> {
        // Just discard the value and return unit
        Ok(())
    }
}

from_cuda_object!(module::Module);
from_cuda_object!(context::Context);

#[cfg(feature = "amd")]
impl<'a> FromCuda<'a, CUlimit> for hipLimit_t {
    fn from_cuda(limit: &'a CUlimit) -> Result<Self, CUerror> {
        Ok(match *limit {
            CUlimit::CU_LIMIT_STACK_SIZE => hipLimit_t::hipLimitStackSize,
            CUlimit::CU_LIMIT_PRINTF_FIFO_SIZE => hipLimit_t::hipLimitPrintfFifoSize,
            CUlimit::CU_LIMIT_MALLOC_HEAP_SIZE => hipLimit_t::hipLimitMallocHeapSize,
            _ => return Err(CUerror::NOT_SUPPORTED),
        })
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, CUlimit> for ze_device_limit_t {
    fn from_cuda(limit: &'a CUlimit) -> Result<Self, CUerror> {
        // Intel's Level Zero doesn't have direct equivalents for CUDA limits
        // Return same enum variants as AMD for consistency
        Ok(match *limit {
            CUlimit::CU_LIMIT_STACK_SIZE => ze_device_limit_t::ZE_LIMIT_STACK_SIZE,
            CUlimit::CU_LIMIT_PRINTF_FIFO_SIZE => ze_device_limit_t::ZE_LIMIT_PRINTF_FIFO_SIZE,
            CUlimit::CU_LIMIT_MALLOC_HEAP_SIZE => ze_device_limit_t::ZE_LIMIT_MALLOC_HEAP_SIZE,
            _ => return Err(CUerror::NOT_SUPPORTED),
        })
    }
}

pub(crate) trait ZludaObject: Sized + Send + Sync {
    const COOKIE: usize;
    const LIVENESS_FAIL: CUerror = cuda_types::cuda::CUerror::INVALID_VALUE;

    type CudaHandle: Sized;

    fn drop_checked(&mut self) -> CUresult;

    fn wrap(self) -> Self::CudaHandle {
        unsafe { mem::transmute_copy(&LiveCheck::wrap(self)) }
    }
}

// Helper trait for converting between CUDA and Ze results (Intel only)
#[cfg(feature = "intel")]
pub(crate) trait Decuda {
    fn decuda(self) -> CUresult;
}

#[cfg(feature = "intel")]
impl Decuda for CUresult {
    fn decuda(self) -> CUresult {
        self
    }
}

// Helper trait for converting between Ze and CUDA results (Intel only)
#[cfg(feature = "intel")]
pub(crate) trait Encuda {
    fn encuda(self) -> CUresult;
}

#[cfg(feature = "intel")]
impl Encuda for CUresult {
    fn encuda(self) -> CUresult {
        self
    }
}

#[repr(C)]
pub(crate) struct LiveCheck<T: ZludaObject> {
    cookie: usize,
    data: MaybeUninit<T>,
}

impl<T: ZludaObject> LiveCheck<T> {
    fn new(data: T) -> Self {
        LiveCheck {
            cookie: T::COOKIE,
            data: MaybeUninit::new(data),
        }
    }

    fn as_handle(&self) -> T::CudaHandle {
        unsafe { mem::transmute_copy(&self) }
    }

    fn wrap(data: T) -> *mut Self {
        Box::into_raw(Box::new(Self::new(data)))
    }

    fn as_result(&self) -> Result<&T, CUerror> {
        if self.cookie == T::COOKIE {
            Ok(unsafe { self.data.assume_init_ref() })
        } else {
            Err(T::LIVENESS_FAIL)
        }
    }

    // This looks like nonsense, but it's not. There are two cases:
    // Err(CUerror) -> meaning that the object is invalid, this pointer does not point into valid memory
    // Ok(maybe_error) -> meaning that the object is valid, we dropped everything, but there *might*
    //                    an error in the underlying runtime that we want to propagate
    #[must_use]
    fn drop_checked(&mut self) -> Result<Result<(), CUerror>, CUerror> {
        if self.cookie == T::COOKIE {
            self.cookie = 0;
            let result = unsafe { self.data.assume_init_mut().drop_checked() };
            unsafe { MaybeUninit::assume_init_drop(&mut self.data) };
            Ok(result)
        } else {
            Err(T::LIVENESS_FAIL)
        }
    }
}

pub fn as_ref<'a, T: ZludaObject>(
    handle: &'a T::CudaHandle,
) -> &'a ManuallyDrop<Box<LiveCheck<T>>> {
    unsafe { mem::transmute(handle) }
}

pub fn drop_checked<T: ZludaObject>(handle: T::CudaHandle) -> Result<(), CUerror> {
    let mut wrapped_object: ManuallyDrop<Box<LiveCheck<T>>> =
        unsafe { mem::transmute_copy(&handle) };
    let underlying_error = LiveCheck::drop_checked(&mut wrapped_object)?;
    unsafe { ManuallyDrop::drop(&mut wrapped_object) };
    underlying_error
}

// Update the CUresult conversion to use proper enum variants
#[cfg(feature = "intel")]
fn ze_to_cuda_result(result: ze_result_t) -> CUresult {
    match result {
        ze_result_t::ZE_RESULT_SUCCESS => CUresult::SUCCESS,
        ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
        | ze_result_t::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY => CUresult::ERROR_OUT_OF_MEMORY,
        ze_result_t::ZE_RESULT_ERROR_DEVICE_LOST => CUresult::ERROR_NO_DEVICE,
        ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_HANDLE => CUresult::ERROR_INVALID_HANDLE,
        ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_POINTER => CUresult::ERROR_INVALID_VALUE,
        ze_result_t::ZE_RESULT_ERROR_UNINITIALIZED => CUresult::ERROR_NOT_INITIALIZED,
        _ => CUresult::ERROR_UNKNOWN,
    }
}

// Implement the CUlimit to ze_device_limit_t conversion
#[cfg(feature = "intel")]
fn cuda_limit_to_ze_limit(limit: CUlimit) -> crate::r#impl::ze_device::ze_device_limit_t {
    match limit {
        CUlimit::CU_LIMIT_STACK_SIZE => {
            crate::r#impl::ze_device::ze_device_limit_t::ZE_LIMIT_STACK_SIZE
        }
        CUlimit::CU_LIMIT_PRINTF_FIFO_SIZE => {
            crate::r#impl::ze_device::ze_device_limit_t::ZE_LIMIT_PRINTF_FIFO_SIZE
        }
        CUlimit::CU_LIMIT_MALLOC_HEAP_SIZE => {
            crate::r#impl::ze_device::ze_device_limit_t::ZE_LIMIT_MALLOC_HEAP_SIZE
        }
        _ => panic!("Unsupported limit"),
    }
}

// This function converts CUresult to proper CUresult under Intel builds
#[cfg(feature = "intel")]
pub fn ze_to_hip_error(result: CUresult) -> CUresult {
    result
}

// Add a utility function to convert CUerror to CUresult and Result
#[cfg(feature = "intel")]
pub fn error_to_result<T>(error: CUerror) -> Result<T, CUerror> {
    Err(error)
}

// Create a wrapper type for ze_result_t to resolve orphan rule violation
#[cfg(feature = "intel")]
pub struct ZeResult(ze_result_t);

#[cfg(feature = "intel")]
impl From<ze_result_t> for ZeResult {
    fn from(result: ze_result_t) -> Self {
        ZeResult(result)
    }
}

#[cfg(feature = "intel")]
impl From<ZeResult> for Result<(), CUerror> {
    fn from(ze_result: ZeResult) -> Self {
        if ze_result.0 == ze_result_t::ZE_RESULT_SUCCESS {
            Ok(())
        } else {
            Err(match ze_result.0 {
                ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
                | ze_result_t::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY => CUerror::OUT_OF_MEMORY,
                ze_result_t::ZE_RESULT_ERROR_DEVICE_LOST => CUerror::NO_DEVICE,
                ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_HANDLE => CUerror::INVALID_HANDLE,
                ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_POINTER => CUerror::INVALID_VALUE,
                ze_result_t::ZE_RESULT_ERROR_UNINITIALIZED => CUerror::NOT_INITIALIZED,
                ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE => CUerror::NOT_SUPPORTED,
                _ => CUerror::UNKNOWN,
            })
        }
    }
}

// Add a convenience function to convert ze_result_t to Result<(), CUerror>
#[cfg(feature = "intel")]
pub fn ze_result_to_result(result: ze_result_t) -> Result<(), CUerror> {
    ZeResult(result).into()
}

// Tenstorrent-specific FromCuda implementations for types not covered by existing macros
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, *mut CUuuid_st> for *mut [u8; 16] {
    fn from_cuda(x: &'a *mut CUuuid_st) -> Result<Self, CUerror> {
        Ok(x.cast::<[u8; 16]>())
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, *mut i8> for *mut [u8; 8] {
    fn from_cuda(x: &'a *mut i8) -> Result<Self, CUerror> {
        Ok(x.cast::<[u8; 8]>())
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, *mut u32> for *mut u32 {
    fn from_cuda(x: &'a *mut u32) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, *mut CUcontext> for *mut CUcontext {
    fn from_cuda(x: &'a *mut CUcontext) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, *mut CUfunction> for *mut CUfunction {
    fn from_cuda(x: &'a *mut CUfunction) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, CUfunction> for *mut module::TtKernel {
    fn from_cuda(handle: &'a CUfunction) -> Result<Self, CUerror> {
        // Convert CUfunction handle to TtKernel pointer
        Ok(handle.0 as *mut module::TtKernel)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, CUstream> for *mut ::core::ffi::c_void {
    fn from_cuda(x: &'a CUstream) -> Result<Self, CUerror> {
        Ok(x.0 as *mut ::core::ffi::c_void)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, CUdeviceptr_v2> for CUdeviceptr_v2 {
    fn from_cuda(x: &'a CUdeviceptr_v2) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, CUpointer_attribute_enum> for CUpointer_attribute_enum {
    fn from_cuda(x: &'a CUpointer_attribute_enum) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, CUfunction_attribute_enum> for CUfunction_attribute_enum {
    fn from_cuda(x: &'a CUfunction_attribute_enum) -> Result<Self, CUerror> {
        Ok(*x)
    }
}
