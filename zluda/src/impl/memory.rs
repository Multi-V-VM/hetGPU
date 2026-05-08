use crate::r#impl::context;
#[cfg(feature = "intel")]
use crate::r#impl::ze_to_cuda_result;
#[cfg(any(feature = "intel", feature = "nvidia"))]
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
use std::ptr;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;

#[cfg(feature = "intel")]
use std::collections::HashMap;
/// Global allocation tracker for virtual backend.
/// Maps pointer addresses to their allocation sizes in bytes.
/// Used by invoke_emulator_bridge to determine safe read sizes.
#[cfg(feature = "intel")]
use std::sync::Mutex;

#[cfg(feature = "intel")]
lazy_static::lazy_static! {
    pub(crate) static ref VIRTUAL_ALLOC_MAP: Mutex<HashMap<usize, usize>> = Mutex::new(HashMap::new());
}

/// Look up the allocation containing the given address.
/// Returns Some((remaining_bytes, base_addr)) if the address falls within a tracked allocation.
/// `remaining_bytes` is how many bytes are available from `addr` to the end of the allocation.
/// Returns None if the address is not within any tracked allocation.
#[cfg(feature = "intel")]
pub(crate) fn get_alloc_size(addr: usize) -> Option<usize> {
    if let Ok(map) = VIRTUAL_ALLOC_MAP.lock() {
        // Check for exact match first (fast path)
        if let Some(&size) = map.get(&addr) {
            return Some(size);
        }
        // Check if addr falls within any allocation range
        for (&base, &size) in map.iter() {
            if addr >= base && addr < base + size {
                // Return remaining bytes from addr to end of allocation
                return Some(size - (addr - base));
            }
        }
        None
    } else {
        None
    }
}
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

        // Track allocation for safe reads in emulator bridge
        if let Ok(mut map) = VIRTUAL_ALLOC_MAP.lock() {
            map.insert(host_ptr as usize, bytesize);
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
        // Virtual device: free host memory using tracked allocation size
        use std::alloc::{dealloc, Layout};

        let addr = dptr.0 as usize;
        if let Ok(mut map) = VIRTUAL_ALLOC_MAP.lock() {
            if let Some(bytesize) = map.remove(&addr) {
                if let Ok(layout) = Layout::from_size_align(bytesize, 64) {
                    unsafe {
                        dealloc(dptr.0 as *mut u8, layout);
                    }
                }
            }
        }
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
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn alloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    eprintln!("[hetGPU memory] alloc_v2 called: bytesize={}", bytesize);

    // Ensure we have a valid context by using primary context
    // First check if there's a current context
    let mut current_ctx: CUcontext = CUcontext(ptr::null_mut());
    let ctx_result = nvidia_runtime_sys::cuCtxGetCurrent(&mut current_ctx);
    eprintln!(
        "[hetGPU memory] cuCtxGetCurrent returned {}, ctx={:?}",
        ctx_result, current_ctx.0
    );

    if ctx_result != 0 || current_ctx.0.is_null() {
        eprintln!("[hetGPU memory] No current context, trying to get primary context for device 0");
        // Try to retain and set primary context for device 0
        let mut pctx: CUcontext = CUcontext(ptr::null_mut());
        let retain_result = nvidia_runtime_sys::cuDevicePrimaryCtxRetain(&mut pctx, 0);
        eprintln!(
            "[hetGPU memory] cuDevicePrimaryCtxRetain returned {}, ctx={:?}",
            retain_result, pctx.0
        );
        if retain_result == 0 && !pctx.0.is_null() {
            let set_result = nvidia_runtime_sys::cuCtxSetCurrent(pctx);
            eprintln!("[hetGPU memory] cuCtxSetCurrent returned {}", set_result);
        }
    }

    let result = nvidia_runtime_sys::cuMemAlloc_v2(dptr, bytesize);
    eprintln!("[hetGPU memory] cuMemAlloc_v2 returned: {}", result);
    if result != 0 {
        eprintln!(
            "[hetGPU memory] cuMemAlloc_v2 FAILED with CUDA error {}",
            result
        );
        // Convert CUDA error codes properly
        return match result {
            1 => Err(CUerror::INVALID_VALUE),
            2 => Err(CUerror::OUT_OF_MEMORY),
            201 => Err(CUerror::INVALID_CONTEXT),
            400 => Err(CUerror::INVALID_HANDLE),
            999 => {
                eprintln!("[hetGPU memory] Error 999 means cuMemAlloc function not loaded!");
                Err(CUerror::NOT_INITIALIZED)
            }
            _ => Err(CUerror::UNKNOWN),
        };
    }
    let ptr_val = unsafe { *dptr };
    eprintln!("[hetGPU memory] alloc_v2 success: ptr={:?}", ptr_val);

    // Record allocation for replay
    super::replay::record_allocation(
        ptr_val.0 as u64,
        bytesize as u64,
        super::replay::AllocationType::Device,
    );

    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn free_v2(dptr: CUdeviceptr) -> CUresult {
    // Record deallocation for replay
    super::replay::record_deallocation(dptr.0 as u64);

    let result = nvidia_runtime_sys::cuMemFree_v2(dptr);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn copy_dto_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    let result = nvidia_runtime_sys::cuMemcpyDtoH_v2(dst_host, src_device, byte_count);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn copy_hto_d_v2(
    dst_device: CUdeviceptr,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> CUresult {
    let result = nvidia_runtime_sys::cuMemcpyHtoD_v2(dst_device, src_host, byte_count);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }

    // Mark destination memory as dirty for replay tracking
    super::replay::record_memory_dirty(dst_device.0 as u64, byte_count as u64);

    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn get_address_range_v2(
    pbase: *mut CUdeviceptr,
    psize: *mut usize,
    dptr: CUdeviceptr,
) -> CUresult {
    let result = nvidia_runtime_sys::cuMemGetAddressRange_v2(pbase, psize, dptr);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn set_d8_v2(dst_device: CUdeviceptr, uc: u8, n: usize) -> CUresult {
    let result = nvidia_runtime_sys::cuMemsetD8_v2(dst_device, uc, n);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn set_d32_v2(dst_device: CUdeviceptr, ui: u32, n: usize) -> CUresult {
    let result = nvidia_runtime_sys::cuMemsetD32_v2(dst_device, ui, n);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

// ─── PACC memory API ─────────────────────────────────────────────────────────
#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
use cuda_types::cuda::*;

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
enum PaccAllocKind {
    Host { align: usize },
    Driver { bo: pacc_runtime_sys::PaccBoMap },
    SharedDdr,
    SharedDdrIpc { map_ptr: usize, map_len: usize },
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
struct PaccAlloc {
    size: usize,
    phys: u64,
    kind: PaccAllocKind,
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
static PACC_ALLOC_MAP: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<u64, PaccAlloc>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
const PACC_IPC_HANDLE_BYTES: usize = 64;

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
const PACC_IPC_HANDLE_MAGIC: u64 = 0x4845_5447_5055_4943; // "HETGPUIC"

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
struct PaccSharedDdrArena {
    ptr: usize,
    phys: u64,
    size: usize,
    cursor: usize,
    heap_end: usize,
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe impl Send for PaccSharedDdrArena {}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
static PACC_SHARED_DDR_ARENA: std::sync::LazyLock<std::sync::Mutex<Option<PaccSharedDdrArena>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(None));

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_parse_u64_text(value: &str) -> Option<u64> {
    let value = value.trim();
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        value.parse::<u64>().ok()
    }
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_read_u64(path: &str) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|v| pacc_parse_u64_text(&v))
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_read_helper_u64_at(path: &str, offset: u64) -> Option<u64> {
    use std::os::unix::fs::FileExt;

    let file = std::fs::OpenOptions::new().read(true).open(path).ok()?;
    let mut buf = [0u8; 8];
    file.read_exact_at(&mut buf, offset).ok()?;
    Some(u64::from_le_bytes(buf))
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_shared_ddr_base_from_helper() -> Option<u64> {
    const SHARED_DDR_BASE_INFO_OFF: u64 = 0x0200_4000;
    let path = pacc_helper_path_for_device0();
    pacc_read_helper_u64_at(&path, SHARED_DDR_BASE_INFO_OFF).filter(|&v| v != 0)
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_shared_ddr_base() -> Option<u64> {
    pacc_read_u64("/sys/kernel/debug/hetgpu_pacc_mbox_ddr_coh/shared_ddr_base")
        .or_else(|| pacc_read_u64("/sys/kernel/debug/hetgpu_pacc_mbox_ddr/shared_ddr_base"))
        .or_else(|| pacc_read_u64("/sys/kernel/debug/hetgpu_pacc_mbox_full/shared_ddr_base"))
        .or_else(|| pacc_read_u64("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_base"))
        .or_else(|| pacc_shared_ddr_base_from_helper())
        .or_else(|| {
            std::env::var("HETGPU_PACC_SHARED_DDR_BASE")
                .ok()
                .and_then(|v| pacc_parse_u64_text(&v))
        })
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_shared_ddr_bytes() -> Option<usize> {
    pacc_read_u64("/sys/kernel/debug/hetgpu_pacc_mbox_ddr_coh/shared_ddr_size")
        .or_else(|| pacc_read_u64("/sys/kernel/debug/hetgpu_pacc_mbox_ddr/shared_ddr_size"))
        .or_else(|| pacc_read_u64("/sys/kernel/debug/hetgpu_pacc_mbox_full/shared_ddr_size"))
        .or_else(|| pacc_read_u64("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_size"))
        .or_else(|| pacc_read_u64("/sys/module/hetgpu_pacc_mbox_ddr_coh/parameters/shared_ddr_size"))
        .or_else(|| pacc_read_u64("/sys/module/hetgpu_pacc_mbox_ddr/parameters/shared_ddr_size"))
        .or_else(|| {
            std::env::var("HETGPU_PACC_SHARED_DDR_BYTES")
                .ok()
                .and_then(|v| pacc_parse_u64_text(&v))
        })
        .and_then(|v| usize::try_from(v).ok())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| pacc_parse_u64_text(&v))
        .and_then(|v| usize::try_from(v).ok())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_helper_path_for_device0() -> String {
    match std::env::var("HETGPU_PACC_MBOX_DEVICE") {
        Ok(pattern) if pattern.contains("%d") => pattern.replace("%d", "0"),
        Ok(path) => path,
        Err(_) => "/dev/hetgpu_pacc_mbox_ddr_coh0".to_string(),
    }
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_alloc_trace(tag: &'static [u8]) {
    let enabled = unsafe {
        let env = libc::getenv(b"HETGPU_PACC_ALLOC_TRACE\0".as_ptr() as *const libc::c_char);
        !env.is_null() && *env == b'1' as libc::c_char
    };
    if enabled {
        unsafe {
            let _ = libc::write(libc::STDERR_FILENO, tag.as_ptr() as *const libc::c_void, tag.len());
            let _ = libc::write(libc::STDERR_FILENO, b"\n".as_ptr() as *const libc::c_void, 1);
        }
    }
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_align_up(value: usize, align: usize) -> Option<usize> {
    if align == 0 {
        return Some(value);
    }
    value.checked_add(align - 1).map(|v| v - (v % align))
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_shared_ddr_kernel_reserve(bytes: usize) -> usize {
    if let Some(reserve) = pacc_parse_env_usize("HETGPU_PACC_SHARED_DEVICE_MEM_KERNEL_RESERVE") {
        return reserve.min(bytes);
    }
    let slot_count = pacc_parse_env_usize("HETGPU_PACC_KERNEL_TOTAL_SLOTS")
        .or_else(|| pacc_parse_env_usize("HETGPU_PACC_KERNEL_SLOT_COUNT"))
        .unwrap_or(4)
        .max(1);
    let slot_bytes = pacc_parse_env_usize("HETGPU_PACC_KERNEL_SLOT_BYTES")
        .or_else(|| pacc_parse_env_usize("HETGPU_PACC_KERNEL_DEFAULT_SLOT_BYTES"))
        .unwrap_or(64 * 1024 * 1024)
        .max(64 * 1024);
    slot_count.saturating_mul(slot_bytes).min(bytes)
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_alloc_shared_ddr(bytesize: usize) -> Result<(u64, PaccAlloc), CUerror> {
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;

    pacc_alloc_trace(b"[pacc_alloc] shared entry");
    let mut guard = PACC_SHARED_DDR_ARENA.lock().map_err(|_| CUerror::UNKNOWN)?;
    pacc_alloc_trace(b"[pacc_alloc] shared lock");
    if guard.is_none() {
        pacc_alloc_trace(b"[pacc_alloc] shared init");
        let phys = pacc_shared_ddr_base().ok_or(CUerror::OUT_OF_MEMORY)?;
        pacc_alloc_trace(b"[pacc_alloc] shared base");
        let size = pacc_shared_ddr_bytes().ok_or(CUerror::OUT_OF_MEMORY)?;
        pacc_alloc_trace(b"[pacc_alloc] shared bytes");
        let heap_offset = pacc_parse_env_usize("HETGPU_PACC_SHARED_DEVICE_MEM_HEAP_OFFSET")
            .unwrap_or(0);
        if heap_offset >= size || heap_offset % 4096 != 0 {
            return Err(CUerror::OUT_OF_MEMORY);
        }
        pacc_alloc_trace(b"[pacc_alloc] shared heap offset");
        let window_bytes = pacc_parse_env_usize("HETGPU_PACC_SHARED_DEVICE_MEM_HEAP_BYTES")
            .unwrap_or(size - heap_offset)
            .min(size - heap_offset);
        if window_bytes == 0 || window_bytes % 4096 != 0 {
            return Err(CUerror::OUT_OF_MEMORY);
        }
        pacc_alloc_trace(b"[pacc_alloc] shared window");
        let path = pacc_helper_path_for_device0();
        pacc_alloc_trace(b"[pacc_alloc] shared path");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|_| CUerror::OUT_OF_MEMORY)?;
        pacc_alloc_trace(b"[pacc_alloc] shared open");
        let ptr = unsafe {
            pacc_alloc_trace(b"[pacc_alloc] shared mmap before");
            libc::mmap(
                std::ptr::null_mut(),
                window_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                heap_offset as libc::off_t,
            )
        };
        pacc_alloc_trace(b"[pacc_alloc] shared mmap after");
        if ptr == libc::MAP_FAILED {
            return Err(CUerror::OUT_OF_MEMORY);
        }

        let control_reserved = if heap_offset == 0 { 4usize * 0x2000usize } else { 0 };
        let kernel_reserved = pacc_shared_ddr_kernel_reserve(window_bytes);
        let heap_end = window_bytes.saturating_sub(kernel_reserved);
        pacc_alloc_trace(b"[pacc_alloc] shared reserve");
        if heap_end <= control_reserved {
            unsafe {
                libc::munmap(ptr, window_bytes);
            }
            return Err(CUerror::OUT_OF_MEMORY);
        }
        if std::env::var("HETGPU_PACC_LOG_MEMORY").ok().as_deref() == Some("1") {
            eprintln!(
                "[PACC Backend] shared-DDR CUDA heap mmap {} phys=0x{:x} offset=0x{:x} bytes={} heap=[0x{:x},0x{:x}) kernel_reserve={}",
                path, phys.saturating_add(heap_offset as u64), heap_offset, window_bytes, control_reserved, heap_end, kernel_reserved
            );
        }
        pacc_alloc_trace(b"[pacc_alloc] shared guard before");
        *guard = Some(PaccSharedDdrArena {
            ptr: ptr as usize,
            phys: phys.saturating_add(heap_offset as u64),
            size: window_bytes,
            cursor: control_reserved,
            heap_end,
        });
        pacc_alloc_trace(b"[pacc_alloc] shared guard after");
    }

    let arena = guard.as_mut().ok_or(CUerror::OUT_OF_MEMORY)?;
    pacc_alloc_trace(b"[pacc_alloc] shared arena");
    let alloc_size = bytesize.max(1);
    let offset = pacc_align_up(arena.cursor, 256).ok_or(CUerror::OUT_OF_MEMORY)?;
    let end = offset
        .checked_add(alloc_size)
        .ok_or(CUerror::OUT_OF_MEMORY)?;
    if end > arena.heap_end || end > arena.size {
        return Err(CUerror::OUT_OF_MEMORY);
    }
    arena.cursor = end;
    let addr = (arena.ptr + offset) as u64;
    let phys = arena.phys.saturating_add(offset as u64);
    pacc_alloc_trace(b"[pacc_alloc] shared alloc ready");
    if std::env::var("HETGPU_PACC_SHARED_DEVICE_MEM_ZERO")
        .ok()
        .as_deref()
        == Some("1")
    {
        pacc_alloc_trace(b"[pacc_alloc] shared zero before");
        unsafe {
            std::ptr::write_bytes(addr as *mut u8, 0, alloc_size);
        }
        pacc_alloc_trace(b"[pacc_alloc] shared zero after");
    }
    pacc_alloc_trace(b"[pacc_alloc] shared return");
    Ok((
        addr,
        PaccAlloc {
            size: alloc_size,
            phys,
            kind: PaccAllocKind::SharedDdr,
        },
    ))
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_alloc_host(bytesize: usize) -> Result<(u64, PaccAlloc), CUerror> {
    use std::alloc::{alloc_zeroed, Layout};

    let align = 64;
    let alloc_size = bytesize.max(1);
    let layout = Layout::from_size_align(alloc_size, align).map_err(|_| CUerror::OUT_OF_MEMORY)?;
    let ptr = unsafe { alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(CUerror::OUT_OF_MEMORY);
    }
    Ok((
        ptr as u64,
        PaccAlloc {
            size: alloc_size,
            phys: 0,
            kind: PaccAllocKind::Host { align },
        },
    ))
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn alloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    pacc_alloc_trace(b"[pacc_alloc] alloc_v2 entry");
    if dptr.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let real_driver_alloc =
        std::env::var("HETGPU_PACC_REAL_DEVICE_MEM").ok().as_deref() == Some("1");
    let shared_device_mem = std::env::var("HETGPU_PACC_SHARED_DEVICE_MEM")
        .ok()
        .as_deref()
        == Some("1");
    let allow_host_device_mem = std::env::var("HETGPU_PACC_ALLOW_HOST_DEVICE_MEM")
        .ok()
        .as_deref()
        == Some("1");
    pacc_alloc_trace(b"[pacc_alloc] alloc_v2 env");

    let (addr, alloc) = if real_driver_alloc {
        pacc_alloc_trace(b"[pacc_alloc] alloc_v2 real driver");
        match pacc_runtime_sys::PaccDevice::open(0).and_then(|dev| dev.bo_alloc_map(bytesize)) {
            Ok(mut bo) => {
                let phys = bo.phys();
                let cuda_ptr = bo.as_mut_slice().as_mut_ptr() as u64;
                (
                    cuda_ptr,
                    PaccAlloc {
                        size: bytesize,
                        phys,
                        kind: PaccAllocKind::Driver { bo },
                    },
                )
            }
            Err(e) => {
                eprintln!("[PACC Backend] cuMemAlloc real device memory failed: {}", e);
                return Err(CUerror::OUT_OF_MEMORY);
            }
        }
    } else if shared_device_mem {
        pacc_alloc_trace(b"[pacc_alloc] alloc_v2 shared before");
        match pacc_alloc_shared_ddr(bytesize) {
            Ok((addr, alloc)) => {
                pacc_alloc_trace(b"[pacc_alloc] alloc_v2 shared ok");
                (addr, alloc)
            }
            Err(e) if allow_host_device_mem => {
                pacc_alloc_trace(b"[pacc_alloc] alloc_v2 shared fallback");
                if std::env::var("HETGPU_PACC_LOG_MEMORY").ok().as_deref() == Some("1") {
                    eprintln!(
                        "[PACC Backend] shared-DDR CUDA memory failed; falling back to host-backed memory"
                    );
                }
                pacc_alloc_host(bytesize).map_err(|_| e)?
            }
            Err(e) => return Err(e),
        }
    } else if allow_host_device_mem {
        pacc_alloc_trace(b"[pacc_alloc] alloc_v2 host");
        if std::env::var("HETGPU_PACC_LOG_MEMORY").ok().as_deref() == Some("1") {
            eprintln!(
                "[PACC Backend] HETGPU_PACC_ALLOW_HOST_DEVICE_MEM=1: using host-backed CUDA memory"
            );
        }
        pacc_alloc_host(bytesize)?
    } else {
        eprintln!(
            "[PACC Backend] refusing host-backed CUDA memory; set HETGPU_PACC_REAL_DEVICE_MEM=1 \
             for driver memory or HETGPU_PACC_ALLOW_HOST_DEVICE_MEM=1 only for non-PACC debugging"
        );
        return Err(CUerror::OUT_OF_MEMORY);
    };

    pacc_alloc_trace(b"[pacc_alloc] alloc_v2 map before");
    PACC_ALLOC_MAP.lock().unwrap().insert(addr, alloc);
    pacc_alloc_trace(b"[pacc_alloc] alloc_v2 map after");
    unsafe {
        *dptr = CUdeviceptr_v2(addr as *mut _);
    }
    pacc_alloc_trace(b"[pacc_alloc] alloc_v2 done");
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn free_v2(dptr: CUdeviceptr) -> CUresult {
    let addr = dptr.0 as u64;
    if addr == 0 {
        return Ok(());
    }
    let alloc = PACC_ALLOC_MAP.lock().unwrap().remove(&addr);
    if let Some(alloc) = alloc {
        match alloc.kind {
            PaccAllocKind::Host { align } => {
                if let Ok(layout) = std::alloc::Layout::from_size_align(alloc.size.max(1), align) {
                    unsafe {
                        std::alloc::dealloc(addr as *mut u8, layout);
                    }
                }
            }
            PaccAllocKind::Driver { .. } => {
                // Dropping PaccBoMap unmaps the userspace view. The current
                // driver does not expose a safe free ioctl; ioctl nr=3 is BO
                // submit and is deliberately safety-gated in pacc_runtime_sys.
            }
            PaccAllocKind::SharedDdr => {
                // Shared-DDR allocations come from a monotonic process-local
                // arena. The mmap stays alive until process exit so outstanding
                // kernel bindings cannot dangle.
            }
            PaccAllocKind::SharedDdrIpc { map_ptr, map_len } => {
                if map_ptr != 0 && map_len != 0 {
                    unsafe {
                        libc::munmap(map_ptr as *mut libc::c_void, map_len);
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn copy_dto_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    if dst_host.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let addr = src_device.0 as u64;
    {
        let mut map = PACC_ALLOC_MAP.lock().unwrap();
        if let Some((base, alloc)) = map.iter_mut().find(|(base, alloc)| {
            let start = **base;
            let end = start.saturating_add(alloc.size as u64);
            addr >= start && addr < end
        }) {
            let offset = addr.saturating_sub(*base) as usize;
            if offset
                .checked_add(byte_count)
                .map_or(true, |end| end > alloc.size)
            {
                return Err(CUerror::INVALID_VALUE);
            }
            if let PaccAllocKind::Driver { bo } = &mut alloc.kind {
                let src = &bo.as_mut_slice()[offset..offset + byte_count];
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst_host as *mut u8, byte_count);
                }
                return Ok(());
            }
        }
    }
    if addr >= 0x1000 && addr < 0x1_0000_0000 {
        return Err(CUerror::INVALID_VALUE);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(src_device.0 as *const u8, dst_host as *mut u8, byte_count);
    }
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn copy_hto_d_v2(
    dst_device: CUdeviceptr,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> CUresult {
    if src_host.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let addr = dst_device.0 as u64;
    {
        let mut map = PACC_ALLOC_MAP.lock().unwrap();
        if let Some((base, alloc)) = map.iter_mut().find(|(base, alloc)| {
            let start = **base;
            let end = start.saturating_add(alloc.size as u64);
            addr >= start && addr < end
        }) {
            let offset = addr.saturating_sub(*base) as usize;
            if offset
                .checked_add(byte_count)
                .map_or(true, |end| end > alloc.size)
            {
                return Err(CUerror::INVALID_VALUE);
            }
            if let PaccAllocKind::Driver { bo } = &mut alloc.kind {
                let dst = &mut bo.as_mut_slice()[offset..offset + byte_count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_host as *const u8,
                        dst.as_mut_ptr(),
                        byte_count,
                    );
                }
                bo.flush().map_err(|_| CUerror::UNKNOWN)?;
                return Ok(());
            }
        }
    }
    if addr >= 0x1000 && addr < 0x1_0000_0000 {
        return Err(CUerror::INVALID_VALUE);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(src_host as *const u8, dst_device.0 as *mut u8, byte_count);
    }
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_address_range_v2(
    pbase: *mut CUdeviceptr,
    psize: *mut usize,
    dptr: CUdeviceptr,
) -> CUresult {
    let addr = dptr.0 as u64;
    let map = PACC_ALLOC_MAP.lock().unwrap();
    if let Some((base, alloc)) = map.iter().find(|(base, alloc)| {
        let start = **base;
        let end = start.saturating_add(alloc.size as u64);
        addr >= start && addr < end
    }) {
        if !pbase.is_null() {
            unsafe {
                *pbase = CUdeviceptr_v2((*base) as *mut _);
            }
        }
        if !psize.is_null() {
            unsafe {
                *psize = alloc.size;
            }
        }
        Ok(())
    } else {
        Err(CUerror::INVALID_VALUE)
    }
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn set_d32_v2(dst: CUdeviceptr, ui: ::core::ffi::c_uint, n: usize) -> CUresult {
    let addr = dst.0 as u64;
    let byte_count = n
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CUerror::INVALID_VALUE)?;
    {
        let mut map = PACC_ALLOC_MAP.lock().unwrap();
        if let Some((base, alloc)) = map.iter_mut().find(|(base, alloc)| {
            let start = **base;
            let end = start.saturating_add(alloc.size as u64);
            addr >= start && addr < end
        }) {
            let offset = addr.saturating_sub(*base) as usize;
            if offset
                .checked_add(byte_count)
                .map_or(true, |end| end > alloc.size)
            {
                return Err(CUerror::INVALID_VALUE);
            }
            if let PaccAllocKind::Driver { bo } = &mut alloc.kind {
                let dst = &mut bo.as_mut_slice()[offset..offset + byte_count];
                for chunk in dst.chunks_exact_mut(std::mem::size_of::<u32>()) {
                    chunk.copy_from_slice(&ui.to_ne_bytes());
                }
                bo.flush().map_err(|_| CUerror::UNKNOWN)?;
                return Ok(());
            }
        }
    }
    if addr >= 0x1000 && addr < 0x1_0000_0000 {
        return Err(CUerror::INVALID_VALUE);
    }
    if ui == 0 {
        unsafe {
            std::ptr::write_bytes(dst.0 as *mut u32, 0, n);
        }
    } else {
        let slice = unsafe { std::slice::from_raw_parts_mut(dst.0 as *mut u32, n) };
        slice.fill(ui);
    }
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn set_d8_v2(dst: CUdeviceptr, value: ::core::ffi::c_uchar, n: usize) -> CUresult {
    let addr = dst.0 as u64;
    {
        let mut map = PACC_ALLOC_MAP.lock().unwrap();
        if let Some((base, alloc)) = map.iter_mut().find(|(base, alloc)| {
            let start = **base;
            let end = start.saturating_add(alloc.size as u64);
            addr >= start && addr < end
        }) {
            let offset = addr.saturating_sub(*base) as usize;
            if offset.checked_add(n).map_or(true, |end| end > alloc.size) {
                return Err(CUerror::INVALID_VALUE);
            }
            if let PaccAllocKind::Driver { bo } = &mut alloc.kind {
                bo.as_mut_slice()[offset..offset + n].fill(value);
                bo.flush().map_err(|_| CUerror::UNKNOWN)?;
                return Ok(());
            }
        }
    }
    if addr >= 0x1000 && addr < 0x1_0000_0000 {
        return Err(CUerror::INVALID_VALUE);
    }
    if n != 0 {
        unsafe {
            std::ptr::write_bytes(dst.0 as *mut u8, value, n);
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn pacc_resolve_device_addr(ptr: *const ::core::ffi::c_void) -> Option<u64> {
    let addr = ptr as u64;
    let map = PACC_ALLOC_MAP.lock().unwrap();
    map.iter().find_map(|(base, alloc)| {
        let start = *base;
        let end = start.saturating_add(alloc.size as u64);
        if addr >= start && addr < end {
            let offset = addr.saturating_sub(start);
            match &alloc.kind {
                PaccAllocKind::Driver { .. } => Some(alloc.phys.saturating_add(offset)),
                PaccAllocKind::SharedDdr | PaccAllocKind::SharedDdrIpc { .. } => {
                    Some(alloc.phys.saturating_add(offset))
                }
                PaccAllocKind::Host { .. } => Some(addr),
            }
        } else {
            None
        }
    })
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn pacc_driver_physical_addr(addr: u64) -> Option<u64> {
    let map = PACC_ALLOC_MAP.lock().unwrap();
    map.iter().find_map(|(base, alloc)| {
        let start = *base;
        let end = start.saturating_add(alloc.size as u64);
        if addr >= start && addr < end {
            let offset = addr.saturating_sub(start);
            match &alloc.kind {
                PaccAllocKind::Driver { .. } => Some(alloc.phys.saturating_add(offset)),
                PaccAllocKind::SharedDdr | PaccAllocKind::SharedDdrIpc { .. } => {
                    Some(alloc.phys.saturating_add(offset))
                }
                PaccAllocKind::Host { .. } => None,
            }
        } else {
            None
        }
    })
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn pacc_shared_ddr_physical_addr(addr: u64) -> Option<u64> {
    let map = PACC_ALLOC_MAP.lock().unwrap();
    map.iter().find_map(|(base, alloc)| {
        let start = *base;
        let end = start.saturating_add(alloc.size as u64);
        if addr >= start && addr < end {
            let offset = addr.saturating_sub(start);
            match &alloc.kind {
                PaccAllocKind::SharedDdr | PaccAllocKind::SharedDdrIpc { .. } => {
                    Some(alloc.phys.saturating_add(offset))
                }
                PaccAllocKind::Driver { .. } | PaccAllocKind::Host { .. } => None,
            }
        } else {
            None
        }
    })
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn pacc_allocation_remaining_addr(addr: u64) -> Option<usize> {
    let map = PACC_ALLOC_MAP.lock().unwrap();
    map.iter().find_map(|(base, alloc)| {
        let start = *base;
        let end = start.saturating_add(alloc.size as u64);
        if addr >= start && addr < end {
            Some((end - addr) as usize)
        } else {
            None
        }
    })
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_ipc_write_u64(buf: &mut [u8; PACC_IPC_HANDLE_BYTES], offset: usize, value: u64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn pacc_ipc_read_u64(buf: &[u8], offset: usize) -> Option<u64> {
    let bytes: [u8; 8] = buf.get(offset..offset + 8)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_ipc_get_mem_handle(
    ptr: *const ::core::ffi::c_void,
    handle: *mut ::core::ffi::c_void,
    handle_len: usize,
) -> i32 {
    if ptr.is_null() || handle.is_null() || handle_len < PACC_IPC_HANDLE_BYTES {
        return 1;
    }

    let addr = ptr as u64;
    let map = PACC_ALLOC_MAP.lock().unwrap();
    let found = map.iter().find_map(|(base, alloc)| {
        let start = *base;
        let end = start.saturating_add(alloc.size as u64);
        if addr >= start && addr < end {
            let offset = addr.saturating_sub(start);
            if matches!(
                &alloc.kind,
                PaccAllocKind::SharedDdr | PaccAllocKind::SharedDdrIpc { .. }
            ) {
                Some((
                    alloc.phys.saturating_add(offset),
                    alloc.size.saturating_sub(offset as usize),
                ))
            } else {
                None
            }
        } else {
            None
        }
    });
    drop(map);

    let Some((phys, size)) = found else {
        return 1;
    };
    if size == 0 {
        return 1;
    }

    let mut encoded = [0u8; PACC_IPC_HANDLE_BYTES];
    pacc_ipc_write_u64(&mut encoded, 0, PACC_IPC_HANDLE_MAGIC);
    pacc_ipc_write_u64(&mut encoded, 8, 1);
    pacc_ipc_write_u64(&mut encoded, 16, phys);
    pacc_ipc_write_u64(&mut encoded, 24, size as u64);
    std::ptr::copy_nonoverlapping(encoded.as_ptr(), handle as *mut u8, encoded.len());
    0
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_ipc_open_mem_handle(
    dev_ptr: *mut *mut ::core::ffi::c_void,
    handle: *const ::core::ffi::c_void,
    _flags: ::core::ffi::c_uint,
) -> i32 {
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;

    if dev_ptr.is_null() || handle.is_null() {
        return 1;
    }

    let encoded = std::slice::from_raw_parts(handle as *const u8, PACC_IPC_HANDLE_BYTES);
    if pacc_ipc_read_u64(encoded, 0) != Some(PACC_IPC_HANDLE_MAGIC)
        || pacc_ipc_read_u64(encoded, 8) != Some(1)
    {
        return 1;
    }
    let phys = match pacc_ipc_read_u64(encoded, 16) {
        Some(v) if v != 0 => v,
        _ => return 1,
    };
    let size = match pacc_ipc_read_u64(encoded, 24).and_then(|v| usize::try_from(v).ok()) {
        Some(v) if v != 0 => v,
        _ => return 1,
    };

    let shared_base = match pacc_shared_ddr_base() {
        Some(v) => v,
        None => return 1,
    };
    let shared_bytes = match pacc_shared_ddr_bytes() {
        Some(v) => v,
        None => return 1,
    };
    if phys < shared_base {
        return 1;
    }
    let ddr_offset = phys - shared_base;
    if ddr_offset
        .checked_add(size as u64)
        .map_or(true, |end| end > shared_bytes as u64)
    {
        return 1;
    }

    let page = 4096usize;
    let page_offset = (ddr_offset as usize / page) * page;
    let page_delta = ddr_offset as usize - page_offset;
    let map_len = match page_delta
        .checked_add(size)
        .and_then(|v| pacc_align_up(v, page))
    {
        Some(v) if v != 0 => v,
        _ => return 1,
    };
    let path = pacc_helper_path_for_device0();
    let file = match OpenOptions::new().read(true).write(true).open(&path) {
        Ok(v) => v,
        Err(_) => return 1,
    };
    let map_ptr = libc::mmap(
        std::ptr::null_mut(),
        map_len,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED,
        file.as_raw_fd(),
        page_offset as libc::off_t,
    );
    if map_ptr == libc::MAP_FAILED {
        return 1;
    }

    let addr = (map_ptr as usize).saturating_add(page_delta);
    PACC_ALLOC_MAP.lock().unwrap().insert(
        addr as u64,
        PaccAlloc {
            size,
            phys,
            kind: PaccAllocKind::SharedDdrIpc {
                map_ptr: map_ptr as usize,
                map_len,
            },
        },
    );
    if std::env::var("HETGPU_PACC_LOG_MEMORY").ok().as_deref() == Some("1") {
        eprintln!(
            "[PACC Backend] cudaIpcOpenMemHandle shared-DDR {} phys=0x{:x} offset=0x{:x} size={} -> {:p}",
            path, phys, ddr_offset, size, addr as *mut ::core::ffi::c_void
        );
    }
    *dev_ptr = addr as *mut ::core::ffi::c_void;
    0
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_ipc_close_mem_handle(
    ptr: *mut ::core::ffi::c_void,
) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    let addr = ptr as u64;
    let mut map = PACC_ALLOC_MAP.lock().unwrap();
    let key = if map.contains_key(&addr) {
        Some(addr)
    } else {
        map.iter().find_map(|(base, alloc)| {
            let start = *base;
            let end = start.saturating_add(alloc.size as u64);
            if addr >= start && addr < end {
                Some(*base)
            } else {
                None
            }
        })
    };
    let Some(key) = key else {
        return 0;
    };
    if let Some(alloc) = map.remove(&key) {
        if let PaccAllocKind::SharedDdrIpc { map_ptr, map_len } = alloc.kind {
            if map_ptr != 0 && map_len != 0 {
                libc::munmap(map_ptr as *mut libc::c_void, map_len);
            }
        }
    }
    0
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_resolve_device_addr(ptr: *const ::core::ffi::c_void) -> u64 {
    pacc_resolve_device_addr(ptr).unwrap_or(ptr as u64)
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_is_device_ptr(ptr: *const ::core::ffi::c_void) -> i32 {
    let addr = ptr as u64;
    let map = PACC_ALLOC_MAP.lock().unwrap();
    if map.iter().any(|(base, alloc)| {
        let start = *base;
        let end = start.saturating_add(alloc.size as u64);
        addr >= start
            && addr < end
            && matches!(
                &alloc.kind,
                PaccAllocKind::Driver { .. }
                    | PaccAllocKind::SharedDdr
                    | PaccAllocKind::SharedDdrIpc { .. }
            )
    }) {
        1
    } else {
        0
    }
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_allocation_remaining(
    ptr: *const ::core::ffi::c_void,
) -> usize {
    let addr = ptr as u64;
    if addr == 0 {
        return 0;
    }
    pacc_allocation_remaining_addr(addr).unwrap_or(0)
}
