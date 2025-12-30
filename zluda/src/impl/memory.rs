use crate::r#impl::context;
#[cfg(feature = "intel")]
use crate::r#impl::ze_to_cuda_result;
#[cfg(any(feature = "intel", feature = "nvidia"))]
use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
use nvidia_runtime_sys;
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
        Err(e) => {
            return Err(e);
        }
    };

    // Check if this is a virtual device (null handle)
    if ze_context.device.0.is_null() {
        // Virtual device: use host memory allocation
        // IMPORTANT: Initialize to zero so PyTorch zeros() operations work correctly
        use std::alloc::{alloc_zeroed, Layout};

        if bytesize == 0 {
            unsafe {
                *dptr = cuda_types::cuda::CUdeviceptr_v2(0x1 as *mut _);
            }
            return Ok(());
        }

        let layout = Layout::from_size_align(bytesize, 64).map_err(|_| CUerror::OUT_OF_MEMORY)?;

        // Use alloc_zeroed to initialize memory to zero
        // This makes torch.zeros() work correctly without kernel execution
        let host_ptr = unsafe { alloc_zeroed(layout) };
        if host_ptr.is_null() {
            return Err(CUerror::OUT_OF_MEMORY);
        }

        unsafe {
            *dptr = cuda_types::cuda::CUdeviceptr_v2(host_ptr as *mut _);
            crate::r#impl::hetgpu_debug!(
                "[DEBUG alloc_v2] Virtual device allocated {} bytes at host_ptr={:p}, stored as CUdeviceptr={:p}",
                bytesize,
                host_ptr,
                (*dptr).0
            );
        }

        return Ok(());
    }

    // Real Level Zero device: use ZE API
    let device_desc = ze_device_mem_alloc_desc_t {
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

    Ok(())
}

#[cfg(feature = "amd")]
pub(crate) fn free_v2(dptr: hipDeviceptr_t) -> hipError_t {
    unsafe { hipFree(dptr.0) }
}

#[cfg(feature = "intel")]
pub(crate) fn free_v2(dptr: CUdeviceptr) -> CUresult {
    // Validate the pointer
    if dptr == CUdeviceptr_v2(ptr::null_mut()) {
        return Ok(());
    }

    // Special case for zero-size allocations
    if dptr.0 == 0x1 as *mut _ {
        return Ok(());
    }

    // Get the current ZE context
    let ze_context = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // Check if this is a virtual device (null handle)
    if ze_context.device.0.is_null() {
        // Virtual device: free host memory
        use std::alloc::{dealloc, Layout};

        // We don't know the original size, so we can't properly deallocate
        // For now, just leak the memory (this is for testing only)
        // In a real implementation, we'd track allocations
        // TODO: Track allocations to properly deallocate
        return Ok(());
    }

    // Real Level Zero device: use ZE API
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

    // Check if this is a virtual device (null handle)
    if ctx.device.0.is_null() {
        // Virtual device: simple memcpy from device (host) memory to host memory
        if src_device.0.is_null() || src_device.0 == 0x1 as *mut _ {
            return Err(CUerror::INVALID_VALUE);
        }
        if dst_host.is_null() {
            return Err(CUerror::INVALID_VALUE);
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                src_device.0 as *const u8,
                dst_host as *mut u8,
                byte_count,
            );
        }
        return Ok(());
    }

    // Real Level Zero device: use ZE API
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
            ze_event_handle_t(ptr::null_mut()), // No wait event
            0,                                  // Number of wait events
            &mut ze_event_handle_t(ptr::null_mut()), // No signal event
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

    // Check if this is a virtual device (null handle)
    if ctx.device.0.is_null() {
        // Virtual device: simple memcpy from host memory to device (host) memory
        if dst_device.0.is_null() || dst_device.0 == 0x1 as *mut _ {
            return Err(CUerror::INVALID_VALUE);
        }
        if src_host.is_null() {
            return Err(CUerror::INVALID_VALUE);
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                src_host as *const u8,
                dst_device.0 as *mut u8,
                byte_count,
            );
        }
        return Ok(());
    }

    // Real Level Zero device: use ZE API
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
            ze_event_handle_t(ptr::null_mut()), // No wait event
            0,                                  // Number of wait events
            &mut ze_event_handle_t(ptr::null_mut()), // No signal event
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
    unsafe { hipMemsetD32(dst, ui.try_into().unwrap(), n) }
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
            ze_event_handle_t(ptr::null_mut()), // No wait event
            0,                                  // Number of wait events
            &mut ze_event_handle_t(ptr::null_mut()), // No signal event
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
    crate::r#impl::hetgpu_debug!(
        "[DEBUG set_d8_v2] Called with dst={:p}, value={}, n={}",
        dst.0,
        value,
        n
    );

    // Get current context
    let ctx = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => {
            crate::r#impl::hetgpu_debug!("[DEBUG set_d8_v2] Failed to get context: {:?}", e);
            return Err(e);
        }
    };

    // Check if this is a virtual device (null handle)
    if ctx.device.0.is_null() {
        crate::r#impl::hetgpu_debug!("[DEBUG set_d8_v2] Using virtual device path");
        // Virtual device: use memset on host memory
        if dst.0.is_null() || dst.0 == 0x1 as *mut _ {
            crate::r#impl::hetgpu_debug!("[DEBUG set_d8_v2] Skipping null/sentinel pointer");
            return Ok(());
        }

        unsafe {
            std::ptr::write_bytes(dst.0 as *mut u8, value, n);
        }
        crate::r#impl::hetgpu_debug!(
            "[DEBUG set_d8_v2] Successfully set {} bytes to {}",
            n,
            value
        );
        return Ok(());
    }

    // Real Level Zero device: use ZE API
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
            &mut ze_event_handle_t(ptr::null_mut()), // No signal event
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

    let command_list = ptr::null_mut();
    let result = unsafe { zeCommandListCreate(ctx.context, ctx.device, &desc, command_list) };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return Err(CUresult::ERROR_INVALID_VALUE);
    }

    unsafe {
        let handle = ze_command_list_handle_t((*command_list).0);

        // Track the command list in the context
        ctx.add_command_list(handle);

        Ok(handle)
    }
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

    let command_queue = ptr::null_mut();
    let result =
        unsafe { zeCommandQueueCreate(ctx.context, ctx.device, &queue_desc, command_queue) };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        // Clean up command list
        unsafe { zeCommandListDestroy(command_list) };
        return CUresult::ERROR_INVALID_VALUE;
    }

    let queue_handle = ze_command_queue_handle_t(unsafe { (*command_queue).0 });

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

// Tenstorrent implementations
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
use cuda_types::cuda::*;
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
use tt_runtime_sys::*;

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn alloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    // Get the current TT context
    let tt_context = match context::get_current_tt() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // For Tenstorrent, we'll simulate allocation by storing the size
    // In a real implementation, this would allocate device memory
    unsafe {
        *dptr = CUdeviceptr_v2(bytesize as *mut _);
    }

    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn free_v2(dptr: CUdeviceptr) -> CUresult {
    // For Tenstorrent, memory is automatically managed
    // In a real implementation, this would free device memory
    let _ = dptr; // Suppress unused warning
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn copy_dto_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    // For Tenstorrent, implement device to host copy
    // In a real implementation, this would copy from device memory to host
    let _ = (dst_host, src_device, byte_count); // Suppress unused warnings
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn copy_hto_d_v2(
    dst_device: CUdeviceptr,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> CUresult {
    // For Tenstorrent, implement host to device copy
    // In a real implementation, this would copy from host memory to device
    let _ = (dst_device, src_host, byte_count); // Suppress unused warnings
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_address_range_v2(
    base: *mut CUdeviceptr,
    size: *mut usize,
    dptr: CUdeviceptr,
) -> CUresult {
    // For Tenstorrent, implement address range query
    // In a real implementation, this would return the base and size of the allocation
    unsafe {
        if !base.is_null() {
            *base = dptr;
        }
        if !size.is_null() {
            *size = dptr.0 as usize; // Use the stored size from alloc_v2
        }
    }
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn set_d32_v2(dst: CUdeviceptr, ui: ::core::ffi::c_uint, n: usize) -> CUresult {
    // For Tenstorrent, implement 32-bit memory set
    // In a real implementation, this would set device memory to the specified value
    let _ = (dst, ui, n); // Suppress unused warnings
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn set_d8_v2(dst: CUdeviceptr, value: ::core::ffi::c_uchar, n: usize) -> CUresult {
    // For Tenstorrent, implement 8-bit memory set
    // In a real implementation, this would set device memory to the specified value
    let _ = (dst, value, n); // Suppress unused warnings
    Ok(())
}

// NVIDIA backend memory implementations - passthrough to real libcuda.so
#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
pub(crate) fn alloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    eprintln!("[hetGPU memory] alloc_v2 called: bytesize={}", bytesize);

    // Ensure we have a valid context by using primary context
    // First check if there's a current context
    let mut current_ctx: CUcontext = CUcontext(ptr::null_mut());
    let ctx_result = nvidia_runtime_sys::cuCtxGetCurrent(&mut current_ctx);
    eprintln!("[hetGPU memory] cuCtxGetCurrent returned {}, ctx={:?}", ctx_result, current_ctx.0);

    if ctx_result != 0 || current_ctx.0.is_null() {
        eprintln!("[hetGPU memory] No current context, trying to get primary context for device 0");
        // Try to retain and set primary context for device 0
        let mut pctx: CUcontext = CUcontext(ptr::null_mut());
        let retain_result = nvidia_runtime_sys::cuDevicePrimaryCtxRetain(&mut pctx, 0);
        eprintln!("[hetGPU memory] cuDevicePrimaryCtxRetain returned {}, ctx={:?}", retain_result, pctx.0);
        if retain_result == 0 && !pctx.0.is_null() {
            let set_result = nvidia_runtime_sys::cuCtxSetCurrent(pctx);
            eprintln!("[hetGPU memory] cuCtxSetCurrent returned {}", set_result);
        }
    }

    let result = nvidia_runtime_sys::cuMemAlloc_v2(dptr, bytesize);
    eprintln!("[hetGPU memory] cuMemAlloc_v2 returned: {}", result);
    if result != 0 {
        eprintln!("[hetGPU memory] cuMemAlloc_v2 FAILED with CUDA error {}", result);
        // Convert CUDA error codes properly
        return match result {
            1 => Err(CUerror::INVALID_VALUE),
            2 => Err(CUerror::OUT_OF_MEMORY),
            201 => Err(CUerror::INVALID_CONTEXT),
            400 => Err(CUerror::INVALID_HANDLE),
            999 => {
                eprintln!("[hetGPU memory] Error 999 means cuMemAlloc function not loaded!");
                Err(CUerror::NOT_INITIALIZED)
            },
            _ => Err(CUerror::UNKNOWN),
        };
    }
    let ptr_val = unsafe { *dptr };
    eprintln!("[hetGPU memory] alloc_v2 success: ptr={:?}", ptr_val);
    Ok(())
}

#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
pub(crate) fn free_v2(dptr: CUdeviceptr) -> CUresult {
    let result = nvidia_runtime_sys::cuMemFree_v2(dptr);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
pub(crate) fn copy_dto_h_v2(dst_host: *mut ::core::ffi::c_void, src_device: CUdeviceptr, byte_count: usize) -> CUresult {
    let result = nvidia_runtime_sys::cuMemcpyDtoH_v2(dst_host, src_device, byte_count);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
pub(crate) fn copy_hto_d_v2(dst_device: CUdeviceptr, src_host: *const ::core::ffi::c_void, byte_count: usize) -> CUresult {
    let result = nvidia_runtime_sys::cuMemcpyHtoD_v2(dst_device, src_host, byte_count);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
pub(crate) fn get_address_range_v2(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult {
    let result = nvidia_runtime_sys::cuMemGetAddressRange_v2(pbase, psize, dptr);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
pub(crate) fn set_d8_v2(dst_device: CUdeviceptr, uc: u8, n: usize) -> CUresult {
    let result = nvidia_runtime_sys::cuMemsetD8_v2(dst_device, uc, n);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
pub(crate) fn set_d32_v2(dst_device: CUdeviceptr, ui: u32, n: usize) -> CUresult {
    let result = nvidia_runtime_sys::cuMemsetD32_v2(dst_device, ui, n);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}
