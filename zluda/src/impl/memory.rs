use super::context;
#[cfg(feature = "intel")]
use crate::r#impl::ze_to_cuda_result;
#[cfg(feature = "intel")]
use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
use std::mem;
use std::ptr;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;

#[cfg(feature = "amd")]
pub(crate) fn alloc_v2(dptr: *mut hipDeviceptr_t, bytesize: usize) -> hipError_t {
    unsafe { hipMalloc(dptr.cast(), bytesize) }?;
    // TODO: parametrize for non-Geekbench
    unsafe { hipMemsetD8(*dptr, 0, bytesize) }
}

#[cfg(feature = "intel")]
pub(crate) fn alloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    // Get the current ZE context
    let ze_context = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // Allocate device memory using ZE API
    let mut device_desc = ze_device_mem_alloc_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        pNext: std::ptr::null_mut(),
        flags: 0,
        ordinal: 0,
    };

    let mut device_ptr = std::ptr::null_mut();
    let result = unsafe {
        zeMemAllocDevice(
            ze_context.context,
            &device_desc,
            bytesize,
            1, // alignment
            ze_context.device,
            &mut device_ptr,
        )
    };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return ze_to_cuda_result(result);
    }

    // Store the device pointer in the output parameter
    unsafe {
        *dptr = cuda_types::cuda::CUdeviceptr_v2(device_ptr);
    }

    // Initialize memory to zero (common CUDA behavior)
    unsafe {
        set_d8_v2(*dptr, 0, bytesize)?;
    }

    CUresult::SUCCESS
}

#[cfg(feature = "amd")]
pub(crate) fn free_v2(dptr: hipDeviceptr_t) -> hipError_t {
    unsafe { hipFree(dptr.0) }
}

#[cfg(feature = "intel")]
pub(crate) fn free_v2(dptr: CUdeviceptr) -> CUresult {
    // Get the current ZE context
    let ze_context = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // Validate the pointer
    if dptr == CUdeviceptr_v2(ptr::null_mut()) {
        return CUresult::ERROR_INVALID_VALUE;
    }

    // Free the memory using ZE API
    let result = unsafe { zeMemFree(ze_context.context, dptr.0 as *mut std::ffi::c_void) };
    ze_to_cuda_result(result)
}

#[cfg(feature = "amd")]
pub(crate) fn copy_dto_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    unsafe { hipMemcpyDtoH(dst_host, src_device, byte_count) }
}

#[cfg(feature = "intel")]
pub(crate) fn copy_dto_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    // Get current context
    let ctx = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // Get a command list
    let command_list = match get_immediate_command_list(&ctx) {
        Ok(cl) => cl,
        Err(e) => return e,
    };

    // Append copy command
    let result = unsafe {
        zeCommandListAppendMemoryCopy(
            command_list,
            dst_host,
            src_device.0 as *mut std::ffi::c_void,
            byte_count,
            ze_event_handle_t(ptr::null_mut()),  // No wait event
            0,                                   // Number of wait events
            *ze_event_handle_t(ptr::null_mut()), // No signal event
        )
    };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return CUresult::ERROR_INVALID_VALUE;
    }

    // Close and execute the command list
    execute_immediate_command_list(&ctx, command_list)
}

#[cfg(feature = "amd")]
pub(crate) fn copy_hto_d_v2(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> hipError_t {
    unsafe { hipMemcpyHtoD(dst_device, src_host.cast_mut(), byte_count) }
}

#[cfg(feature = "intel")]
pub(crate) fn copy_hto_d_v2(
    dst_device: CUdeviceptr,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> CUresult {
    // Get current context
    let ctx = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // Get a command list
    let command_list = match get_immediate_command_list(&ctx) {
        Ok(cl) => cl,
        Err(e) => return e,
    };

    // Append copy command
    let result = unsafe {
        zeCommandListAppendMemoryCopy(
            command_list,
            dst_device.0 as *mut std::ffi::c_void,
            src_host as *mut ::core::ffi::c_void,
            byte_count,
            ze_event_handle_t(ptr::null_mut()),  // No wait event
            0,                                   // Number of wait events
            *ze_event_handle_t(ptr::null_mut()), // No signal event
        )
    };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return CUresult::ERROR_INVALID_VALUE;
    }

    // Close and execute the command list
    execute_immediate_command_list(&ctx, command_list)
}

#[cfg(feature = "amd")]
pub(crate) fn get_address_range_v2(
    pbase: *mut hipDeviceptr_t,
    psize: *mut usize,
    dptr: hipDeviceptr_t,
) -> hipError_t {
    unsafe { hipMemGetAddressRange(pbase, psize, dptr) }
}

#[cfg(feature = "intel")]
pub(crate) fn get_address_range_v2(
    pbase: *mut CUdeviceptr,
    psize: *mut usize,
    dptr: CUdeviceptr,
) -> CUresult {
    // Intel Level Zero doesn't have a direct equivalent to hipMemGetAddressRange
    // In a production implementation, you would need to track allocations and their sizes
    // For now, return the same pointer as the base and assume we don't know the size

    if !pbase.is_null() {
        unsafe {
            *pbase = dptr;
        }
    }

    if !psize.is_null() {
        // We don't know the size, so use 0 or query it from allocation tracking in a real implementation
        unsafe {
            *psize = 0;
        }
    }

    CUresult::SUCCESS
}

#[cfg(feature = "amd")]
pub(crate) fn set_d32_v2(dst: hipDeviceptr_t, ui: ::core::ffi::c_uint, n: usize) -> hipError_t {
    unsafe { hipMemsetD32(dst, ui, n) }
}

#[cfg(feature = "intel")]
pub(crate) fn set_d32_v2(dst: CUdeviceptr, ui: ::core::ffi::c_uint, n: usize) -> CUresult {
    // Get current context
    let ctx = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // Get a command list
    let command_list = match get_immediate_command_list(&ctx) {
        Ok(cl) => cl,
        Err(e) => return e,
    };

    // Append fill command
    let result = unsafe {
        zeCommandListAppendMemoryFill(
            command_list,
            dst.0,
            &ui as *const _ as *const ::core::ffi::c_void,
            std::mem::size_of::<::core::ffi::c_uint>(),
            n * std::mem::size_of::<::core::ffi::c_uint>(),
            ze_event_handle_t(ptr::null_mut()),  // No wait event
            0,                                   // Number of wait events
            *ze_event_handle_t(ptr::null_mut()), // No signal event
        )
    };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return CUresult::ERROR_INVALID_VALUE;
    }

    // Close and execute the command list
    execute_immediate_command_list(&ctx, command_list)
}

#[cfg(feature = "amd")]
pub(crate) fn set_d8_v2(dst: hipDeviceptr_t, value: ::core::ffi::c_uchar, n: usize) -> hipError_t {
    unsafe { hipMemsetD8(dst, value, n) }
}

#[cfg(feature = "intel")]
pub(crate) fn set_d8_v2(dst: CUdeviceptr, value: ::core::ffi::c_uchar, n: usize) -> CUresult {
    // Get current context
    let ctx = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // Get a command list
    let command_list = match get_immediate_command_list(&ctx) {
        Ok(cl) => cl,
        Err(e) => return e,
    };

    // Append fill command
    let result = unsafe {
        zeCommandListAppendMemoryFill(
            command_list,
            dst.0 as *mut std::ffi::c_void,
            &value as *const _ as *const ::core::ffi::c_void,
            std::mem::size_of::<::core::ffi::c_uchar>(),
            n,
            ze_event_handle_t(ptr::null_mut()), // No wait event
            0,                                  // Number of wait events
            ze_event_handle_t(ptr::null_mut()), // No signal event
        )
    };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return CUresult::ERROR_INVALID_VALUE;
    }

    // Close and execute the command list
    execute_immediate_command_list(&ctx, command_list)
}

// Helper functions for Intel Level Zero implementation

#[cfg(feature = "intel")]
fn get_immediate_command_list(
    ctx: &context::Context,
) -> Result<ze_command_list_handle_t, CUresult> {
    // Create a new immediate command list
    let desc = ze_command_list_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        pNext: ptr::null(),
        commandQueueGroupOrdinal: 0,
        flags: 0,
    };

    let mut command_list = ptr::null_mut();
    let result = unsafe { zeCommandListCreate(ctx.context, ctx.device, &desc, command_list) };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return Err(CUresult::ERROR_INVALID_VALUE);
    }

    let handle = ze_command_list_handle_t(command_list);

    // Track the command list in the context
    ctx.add_command_list(handle);

    Ok(handle)
}

#[cfg(feature = "intel")]
fn execute_immediate_command_list(
    ctx: &context::Context,
    command_list: ze_command_list_handle_t,
) -> CUresult {
    // Create a command queue
    let queue_desc = ze_command_queue_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        pNext: ptr::null(),
        ordinal: 0,
        index: 0,
        flags: 0,
        mode: ze_command_queue_mode_t::ZE_COMMAND_QUEUE_MODE_DEFAULT,
        priority: ze_command_queue_priority_t::ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
    };

    let mut command_queue = ptr::null_mut();
    let result =
        unsafe { zeCommandQueueCreate(ctx.context, ctx.device, &queue_desc, command_queue) };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        // Clean up command list
        unsafe { zeCommandListDestroy(command_list) };
        return CUresult::ERROR_INVALID_VALUE;
    }

    let queue_handle = ze_command_queue_handle_t(*command_queue.0);

    // Track the command queue in the context
    ctx.add_command_queue(queue_handle);

    // Close the command list
    let result = unsafe { zeCommandListClose(command_list) };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        // Clean up resources
        unsafe {
            zeCommandListDestroy(command_list);
            zeCommandQueueDestroy(queue_handle);
        }
        ctx.remove_command_list(command_list);
        ctx.remove_command_queue(queue_handle);
        return CUresult::ERROR_INVALID_VALUE;
    }

    // Execute the command list
    let result = unsafe {
        zeCommandQueueExecuteCommandLists(
            queue_handle,
            1,
            &command_list,
            ze_fence_handle_t(ptr::null_mut()),
        )
    };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        // Clean up resources
        unsafe {
            zeCommandListDestroy(command_list);
            zeCommandQueueDestroy(queue_handle);
        }
        ctx.remove_command_list(command_list);
        ctx.remove_command_queue(queue_handle);
        return CUresult::ERROR_INVALID_VALUE;
    }

    // Synchronize the queue
    let result = unsafe { zeCommandQueueSynchronize(queue_handle, u64::MAX) };

    // Clean up resources
    unsafe {
        zeCommandListDestroy(command_list);
        zeCommandQueueDestroy(queue_handle);
    }
    ctx.remove_command_list(command_list);
    ctx.remove_command_queue(queue_handle);

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return CUresult::ERROR_INVALID_VALUE;
    }

    CUresult::SUCCESS
}
