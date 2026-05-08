#[cfg(feature = "intel")]
use crate::r#impl::ZeResult;
use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
use nvidia_runtime_sys;
use std::{ffi::c_void, ptr};
#[cfg(feature = "intel")]
use ze_runtime_sys::*;
#[cfg(feature = "amd")]
pub(crate) unsafe fn get_attribute(
    data: *mut c_void,
    attribute: hipPointer_attribute,
    ptr: hipDeviceptr_t,
) -> hipError_t {
    if data == ptr::null_mut() {
        return hipError_t::ErrorInvalidValue;
    }
    match attribute {
        // TODO: implement by getting device ordinal & allocation start,
        // then go through every context for that device
        hipPointer_attribute::HIP_POINTER_ATTRIBUTE_CONTEXT => hipError_t::ErrorNotSupported,
        hipPointer_attribute::HIP_POINTER_ATTRIBUTE_MEMORY_TYPE => {
            let mut hip_result = hipMemoryType(0);
            hipPointerGetAttribute(
                (&mut hip_result as *mut hipMemoryType).cast::<c_void>(),
                attribute,
                ptr,
            )?;
            let cuda_result = memory_type(hip_result)?;
            unsafe { *(data.cast()) = cuda_result };
            Ok(())
        }
        _ => unsafe { hipPointerGetAttribute(data, attribute, ptr) },
    }
}
#[cfg(feature = "amd")]
fn memory_type(cu: hipMemoryType) -> Result<CUmemorytype, hipErrorCode_t> {
    match cu {
        hipMemoryType::hipMemoryTypeHost => Ok(CUmemorytype::CU_MEMORYTYPE_HOST),
        hipMemoryType::hipMemoryTypeDevice => Ok(CUmemorytype::CU_MEMORYTYPE_DEVICE),
        hipMemoryType::hipMemoryTypeArray => Ok(CUmemorytype::CU_MEMORYTYPE_ARRAY),
        hipMemoryType::hipMemoryTypeUnified => Ok(CUmemorytype::CU_MEMORYTYPE_UNIFIED),
        _ => Err(hipErrorCode_t::InvalidValue),
    }
}
#[cfg(feature = "intel")]
pub(crate) unsafe fn get_attribute(
    data: *mut c_void,
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> CUresult {
    if data == ptr::null_mut() {
        return CUresult::ERROR_INVALID_VALUE;
    }

    // Get the current context for querying memory properties
    let ze_context = match super::context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // Check if we're in virtual device mode (device handle is null)
    // In virtual mode, pointers are from alloc_zeroed (host memory), not Level Zero
    let is_virtual_device = ze_context.device.0.is_null();

    match attribute {
        // TODO: implement context attribute for Intel devices
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT => {
            // For Intel, we can get the context from the memory allocation
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = unsafe {
                zeMemGetAllocProperties(
                    ze_context.context,
                    ptr.0 as *const c_void,
                    &mut alloc_props,
                    ptr::null_mut(),
                )
            };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return super::ze_to_cuda_result(result);
            }

            // Store the context handle
            *(data.cast::<ze_context_handle_t>()) = ze_context.context;
            CUresult::SUCCESS
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE => {
            // Virtual device: memory is host-allocated via alloc_zeroed
            if is_virtual_device {
                *(data.cast()) = CUmemorytype::CU_MEMORYTYPE_HOST;
                return CUresult::SUCCESS;
            }

            // Real device: Query memory attributes using Level Zero memory APIs
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = unsafe {
                zeMemGetAllocProperties(
                    ze_context.context,
                    ptr.0 as *const c_void,
                    &mut alloc_props,
                    ptr::null_mut(),
                )
            };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return super::ze_to_cuda_result(result);
            }

            // Convert Level Zero memory type to CUDA memory type
            let cuda_type = match alloc_props.type_ {
                ze_memory_type_t::ZE_MEMORY_TYPE_HOST => CUmemorytype::CU_MEMORYTYPE_HOST,
                ze_memory_type_t::ZE_MEMORY_TYPE_DEVICE => CUmemorytype::CU_MEMORYTYPE_DEVICE,
                ze_memory_type_t::ZE_MEMORY_TYPE_SHARED => CUmemorytype::CU_MEMORYTYPE_UNIFIED,
                _ => return CUresult::ERROR_INVALID_VALUE,
            };

            *(data.cast()) = cuda_type;
            CUresult::SUCCESS
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_POINTER => {
            // Virtual device or real device: just return the pointer as-is
            // For virtual device, this is a host pointer from alloc_zeroed
            // For real device, this is a Level Zero device pointer
            *(data.cast::<CUdeviceptr>()) = ptr;
            CUresult::SUCCESS
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_HOST_POINTER => {
            // For host-mapped memory, need to query base address
            let mut base_ptr = ptr::null_mut();
            let mut size = 0;

            let result = unsafe {
                zeMemGetAddressRange(
                    ze_context.context,
                    ptr.0 as *const c_void,
                    &mut base_ptr,
                    &mut size,
                )
            };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                *(data.cast::<*mut c_void>()) = ptr::null_mut();
            } else {
                *(data.cast::<*mut c_void>()) = base_ptr;
            }

            CUresult::SUCCESS
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_IS_MANAGED => {
            // Check if memory is managed (shared memory in Level Zero)
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = unsafe {
                zeMemGetAllocProperties(
                    ze_context.context,
                    ptr.0 as *const c_void,
                    &mut alloc_props,
                    ptr::null_mut(),
                )
            };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return super::ze_to_cuda_result(result);
            }

            // In Level Zero, shared memory is considered managed
            *(data.cast::<i32>()) = if alloc_props.type_ == ze_memory_type_t::ZE_MEMORY_TYPE_SHARED
            {
                1
            } else {
                0
            };

            CUresult::SUCCESS
        }

        // For other attributes, handle based on Intel capabilities or return unsupported
        _ => CUresult::ERROR_NOT_SUPPORTED,
    }
}

#[cfg(feature = "amd")]
fn memory_type_amd(cu: CUmemorytype) -> Result<CUmemorytype, CUresult> {
    match cu {
        CUmemorytype::CU_MEMORYTYPE_HOST => Ok(CUmemorytype::CU_MEMORYTYPE_HOST),
        CUmemorytype::CU_MEMORYTYPE_DEVICE => Ok(CUmemorytype::CU_MEMORYTYPE_DEVICE),
        CUmemorytype::CU_MEMORYTYPE_ARRAY => Ok(CUmemorytype::CU_MEMORYTYPE_ARRAY),
        CUmemorytype::CU_MEMORYTYPE_UNIFIED => Ok(CUmemorytype::CU_MEMORYTYPE_UNIFIED),
        _ => Err(CUresult::ERROR_ALREADY_ACQUIRED),
    }
}

#[cfg(feature = "intel")]
fn memory_type_intel(ze_type: ze_memory_type_t) -> Result<CUmemorytype, ze_result_t> {
    match ze_type {
        ze_memory_type_t::ZE_MEMORY_TYPE_HOST => Ok(CUmemorytype::CU_MEMORYTYPE_HOST),
        ze_memory_type_t::ZE_MEMORY_TYPE_DEVICE => Ok(CUmemorytype::CU_MEMORYTYPE_DEVICE),
        ze_memory_type_t::ZE_MEMORY_TYPE_SHARED => Ok(CUmemorytype::CU_MEMORYTYPE_UNIFIED),
        _ => Err(ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT),
    }
}

#[cfg(feature = "intel")]
pub(crate) unsafe fn get_pointer_attribute(
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> CUresult {
    // Get the current context
    let ze_context = match super::context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    match attribute {
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT => {
            // Return the current context
            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE => {
            // Query memory type
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = zeMemGetAllocProperties(
                ze_context.context,
                ptr.0 as *const c_void,
                &mut alloc_props,
                ptr::null_mut(),
            );

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return super::ze_to_cuda_result(result);
            }

            // Convert Level Zero memory type to CUDA memory type
            let cuda_type = match alloc_props.type_ {
                ze_memory_type_t::ZE_MEMORY_TYPE_HOST => CUmemorytype::CU_MEMORYTYPE_HOST,
                ze_memory_type_t::ZE_MEMORY_TYPE_DEVICE => CUmemorytype::CU_MEMORYTYPE_DEVICE,
                ze_memory_type_t::ZE_MEMORY_TYPE_SHARED => CUmemorytype::CU_MEMORYTYPE_UNIFIED,
                _ => return ZeResult(ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT).into(),
            };

            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_POINTER => {
            // Device pointer is the same in Level Zero
            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_HOST_POINTER => {
            // Query host pointer
            let mut base_ptr = ptr::null_mut();
            let mut size = 0;

            let result = zeMemGetAddressRange(
                ze_context.context,
                ptr.0 as *const c_void,
                &mut base_ptr,
                &mut size,
            );

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return super::ze_to_cuda_result(result);
            }

            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_IS_MANAGED => {
            // Check if memory is managed (shared memory in Level Zero)
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = zeMemGetAllocProperties(
                ze_context.context,
                ptr.0 as *const c_void,
                &mut alloc_props,
                ptr::null_mut(),
            );

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return super::ze_to_cuda_result(result);
            }

            Ok(())
        }

        // For other attributes, handle based on Intel capabilities or return unsupported
        _ => ZeResult(ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE).into(),
    }
}

// Tenstorrent pointer implementation
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) unsafe fn get_attribute(
    data: *mut c_void,
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr_v2,
) -> CUresult {
    if data == ptr::null_mut() {
        return Err(CUerror::INVALID_VALUE);
    }

    // Get the current Tenstorrent context
    let _tt_context = match super::context::get_current_tt() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    match attribute {
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT => {
            // For Tenstorrent, return the current context handle
            // In a real implementation, this would query the context associated with the pointer
            *(data.cast::<CUcontext>()) =
                super::context::peek_current().unwrap_or(CUcontext(ptr::null_mut()));
            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE => {
            // For Tenstorrent, assume device memory
            // In a real implementation, this would query the memory type from the runtime
            *(data.cast::<CUmemorytype>()) = CUmemorytype::CU_MEMORYTYPE_DEVICE;
            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_POINTER => {
            // Return the device pointer as-is
            *(data.cast::<CUdeviceptr_v2>()) = ptr;
            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_HOST_POINTER => {
            // For Tenstorrent, host pointers would need runtime query
            // For now, return null as most allocations are device-only
            *(data.cast::<*mut c_void>()) = ptr::null_mut();
            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_IS_MANAGED => {
            // Tenstorrent doesn't have managed memory in the CUDA sense
            *(data.cast::<i32>()) = 0;
            Ok(())
        }

        // For other attributes, return unsupported
        _ => Err(CUerror::NOT_SUPPORTED),
    }
}

// NVIDIA backend pointer implementations - passthrough to real libcuda.so
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) unsafe fn get_attribute(
    data: *mut c_void,
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> CUresult {
    let result = nvidia_runtime_sys::cuPointerGetAttribute(data, attribute, ptr);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

// ─── PACC pointer attribute stub ──────────────────────────────────────────────
#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_attribute(
    data: *mut ::core::ffi::c_void,
    attribute: cuda_types::cuda::CUpointer_attribute,
    ptr: cuda_types::cuda::CUdeviceptr,
) -> cuda_types::cuda::CUresult {
    use cuda_types::cuda::*;
    if data.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    match attribute {
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT => {
            unsafe {
                *(data.cast::<CUcontext>()) =
                    super::context::peek_current().unwrap_or(CUcontext(::std::ptr::null_mut()));
            }
            Ok(())
        }
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE => {
            unsafe {
                *(data.cast::<CUmemorytype>()) = CUmemorytype::CU_MEMORYTYPE_DEVICE;
            }
            Ok(())
        }
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_POINTER => {
            unsafe {
                *(data.cast::<CUdeviceptr>()) = ptr;
            }
            Ok(())
        }
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_HOST_POINTER => {
            unsafe {
                *(data.cast::<*mut ::core::ffi::c_void>()) = ::std::ptr::null_mut();
            }
            Ok(())
        }
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_IS_MANAGED => {
            unsafe {
                *(data.cast::<i32>()) = 0;
            }
            Ok(())
        }
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_MAPPED => {
            unsafe {
                *(data.cast::<i32>()) = 1;
            }
            Ok(())
        }
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_BUFFER_ID => {
            unsafe {
                *(data.cast::<u64>()) = ptr.0 as u64;
            }
            Ok(())
        }
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL => {
            unsafe {
                *(data as *mut i32) = 0;
            }
            Ok(())
        }
        _ => Err(CUerror::INVALID_VALUE),
    }
}
