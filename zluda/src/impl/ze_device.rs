use super::ze_to_cuda_result;
use cuda_types::cuda::*;
use std::mem;
use ze_runtime_sys::*;

/// Enumeration of device limit types
pub enum ze_device_limit_t {
    ZE_LIMIT_STACK_SIZE,
    ZE_LIMIT_PRINTF_FIFO_SIZE,
    ZE_LIMIT_MALLOC_HEAP_SIZE,
}

/// Convert CUDA limit to ZE limit
pub fn cuda_to_ze_limit(limit: CUlimit) -> Result<ze_device_limit_t, CUerror> {
    match limit {
        CUlimit::CU_LIMIT_STACK_SIZE => Ok(ze_device_limit_t::ZE_LIMIT_STACK_SIZE),
        CUlimit::CU_LIMIT_PRINTF_FIFO_SIZE => Ok(ze_device_limit_t::ZE_LIMIT_PRINTF_FIFO_SIZE),
        CUlimit::CU_LIMIT_MALLOC_HEAP_SIZE => Ok(ze_device_limit_t::ZE_LIMIT_MALLOC_HEAP_SIZE),
        _ => Err(CUerror::INVALID_VALUE),
    }
}

/// Get device limit
pub fn get_device_limit(
    device: ze_device_handle_t,
    limit: ze_device_limit_t,
) -> Result<usize, ze_result_t> {
    let mut value: usize = 0;

    // Call appropriate ZE function based on limit type
    match limit {
        ze_device_limit_t::ZE_LIMIT_STACK_SIZE => {
            // Query stack size limit from device properties
            let mut properties = ze_device_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
                pNext: std::ptr::null_mut(),
                // Initialize minimal required fields
                flags: 0,
                type_: ze_device_type_t::ZE_DEVICE_TYPE_GPU,
                vendorId: 0,
                deviceId: 0,
                uuid: ze_device_uuid_t {
                    id: [0; ZE_MAX_DEVICE_UUID_SIZE as usize],
                },
                maxMemAllocSize: 0,
                // Skip initializing the rest of the fields
                ..unsafe { mem::zeroed() }
            };

            let result = unsafe { zeDeviceGetProperties(device, &mut properties) };
            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return Err(result);
            }

            // Use an appropriate property for stack size
            value = properties.maxMemAllocSize as usize / 8;
        }

        ze_device_limit_t::ZE_LIMIT_PRINTF_FIFO_SIZE => {
            // For printf buffer size, use a sensible default or query from device
            value = 1024 * 1024; // 1MB default for printf buffer
        }

        ze_device_limit_t::ZE_LIMIT_MALLOC_HEAP_SIZE => {
            // For heap size, query memory properties
            let mut mem_props = ze_device_memory_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES,
                pNext: std::ptr::null_mut(),
                flags: 0,
                maxClockRate: 0,
                maxBusWidth: 0,
                totalSize: 0,
                name: [0; ZE_MAX_DEVICE_NAME as usize],
            };

            let mut count = 1u32;
            let result = unsafe { zeDeviceGetMemoryProperties(device, &mut count, &mut mem_props) };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return Err(result);
            }

            // Use half of total memory for heap by default
            value = mem_props.totalSize as usize / 2;
        }
    }

    Ok(value)
}

/// Set device limit
pub fn set_device_limit(
    _device: ze_device_handle_t,
    limit: ze_device_limit_t,
    _value: usize,
) -> Result<(), ze_result_t> {
    // Implementing device limit setting is more complex and depends on the specific limit
    // For many limits, this might not be directly supported in Level Zero

    match limit {
        ze_device_limit_t::ZE_LIMIT_STACK_SIZE => {
            // Level Zero may not support directly setting stack size
            // We would need to implement a workaround or return unsupported
            return Err(ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
        }

        ze_device_limit_t::ZE_LIMIT_PRINTF_FIFO_SIZE => {
            // Similar to stack size, this might not be directly supported
            return Err(ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
        }

        ze_device_limit_t::ZE_LIMIT_MALLOC_HEAP_SIZE => {
            // Heap size might be configurable through specific APIs
            // For now, returning unsupported
            return Err(ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
        }
    }
}

/// CUDA device limit API implementation
pub fn cuda_device_limit(dev: CUdevice, limit: CUlimit, value: *mut usize) -> CUresult {
    // Get current ZE context and device
    let ze_context = match super::context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // Convert CUDA limit to ZE limit
    let ze_limit = match cuda_to_ze_limit(limit) {
        Ok(l) => l,
        Err(e) => return CUresult::ERROR_INVALID_VALUE,
    };

    // Get the limit value
    match get_device_limit(ze_context.device, ze_limit) {
        Ok(limit_value) => {
            unsafe { *value = limit_value };
            CUresult::SUCCESS
        }
        Err(e) => ze_to_cuda_result(e),
    }
}

/// CUDA set device limit API implementation
pub fn cuda_set_device_limit(limit: CUlimit, value: usize) -> CUresult {
    // Get current ZE context and device
    let ze_context = match super::context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // Convert CUDA limit to ZE limit
    let ze_limit = match cuda_to_ze_limit(limit) {
        Ok(l) => l,
        Err(e) => return CUresult::ERROR_INVALID_VALUE,
    };

    // Set the limit value
    match set_device_limit(ze_context.device, ze_limit, value) {
        Ok(_) => CUresult::SUCCESS,
        Err(e) => ze_to_cuda_result(e),
    }
}
