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

// Add this function to get device handle by index
#[cfg(feature = "intel")]
fn get_device_handle_by_index(index: usize) -> Result<ze_device_handle_t> {
    // Implementation depends on how you access devices in your system
    // This is a placeholder - replace with actual implementation
    Ok(unsafe { std::mem::zeroed() })
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
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
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
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
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

#[cfg(feature = "nvidia")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                crate::r#impl::nvidia_backend::$fn_name($($arg_id),*)
            }
        )*
    };
}

#[cfg(feature = "nvidia")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                crate::r#impl::nvidia_backend::$fn_name($($arg_id),*)
            }
        )*
    };
}

cuda_base::cuda_function_declarations!(
    unimplemented,
    implemented
        <= [
            cuCtxGetLimit,
            cuCtxSetCurrent,
            cuCtxSetLimit,
            cuCtxSynchronize,
            cuDeviceComputeCapability,
            cuDeviceGet,
            cuDeviceGetAttribute,
            cuDeviceGetCount,
            cuDeviceGetLuid,
            cuDeviceGetName,
            cuDevicePrimaryCtxRelease,
            cuDevicePrimaryCtxRetain,
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
            cuModuleLoadData,
            cuModuleUnload,
            cuPointerGetAttribute,
            cuMemGetAddressRange_v2,
            cuMemsetD32_v2,
            cuMemsetD8_v2
        ],
    implemented_in_function <= [cuLaunchKernel,]
);

// CUDA Runtime API 函数导出 (仅在 nvidia feature 启用时)
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaMallocHost(ptr: *mut *mut std::ffi::c_void, size: usize) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaMallocHost(ptr, size)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaFreeHost(ptr)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaMalloc(devPtr, size)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaFree(devPtr: *mut std::ffi::c_void) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaFree(devPtr)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: std::ffi::c_int) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaMemcpy(dst, src, count, kind)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaMemset(devPtr: *mut std::ffi::c_void, value: std::ffi::c_int, count: usize) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaMemset(devPtr, value, count)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaGetDevice(device: *mut std::ffi::c_int) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaGetDevice(device)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaSetDevice(device: std::ffi::c_int) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaSetDevice(device)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaGetDeviceCount(count: *mut std::ffi::c_int) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaGetDeviceCount(count)
}

// CUDA Stream 管理函数
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaStreamCreate(pStream: *mut *mut std::ffi::c_void) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaStreamCreate(pStream)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaStreamDestroy(stream)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaStreamSynchronize(stream)
}

// CUDA 内存信息函数
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaMemGetInfo(free, total)
}

// CUDA 设备属性函数
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaGetDeviceProperties(prop: *mut std::ffi::c_void, device: std::ffi::c_int) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaGetDeviceProperties(prop, device)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaGetDeviceProperties_v2(prop: *mut std::ffi::c_void, device: std::ffi::c_int) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaGetDeviceProperties(prop, device)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaDeviceGetAttribute(value: *mut std::ffi::c_int, attr: std::ffi::c_int, device: std::ffi::c_int) -> cuda_types::cuda::CUresult {
    crate::r#impl::nvidia_backend::cudaDeviceGetAttribute(value, attr, device)
}

// CUDA Error handling functions
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaGetErrorString(error: std::ffi::c_int) -> *const std::ffi::c_char {
    crate::r#impl::nvidia_backend::cudaGetErrorString(error)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn cudaGetErrorName(error: std::ffi::c_int) -> *const std::ffi::c_char {
    crate::r#impl::nvidia_backend::cudaGetErrorName(error)
}

// NCCL Live Migration Functions
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn ncclCommInitRank(
    comm: *mut *mut std::ffi::c_void,
    nranks: std::ffi::c_int,
    comm_id: *const u8,  // 修正为指针，匹配C调用约定
    rank: std::ffi::c_int
) -> std::ffi::c_int {
    eprintln!("[NCCL-LibEntry] ncclCommInitRank called: rank={}, nranks={}", rank, nranks);
    
    // 将指针转换为128字节数组
    let mut comm_id_array = [0u8; 128];
    if !comm_id.is_null() {
        std::ptr::copy_nonoverlapping(comm_id, comm_id_array.as_mut_ptr(), 128);
        eprintln!("[NCCL-LibEntry] comm_id first 8 bytes: {:?}", &comm_id_array[0..8]);
    }
    
    // 调用实时迁移系统
    crate::r#impl::nccl_live_migration::ncclCommInitRank_with_fault_tolerance(
        comm,
        nranks,
        comm_id_array,
        rank
    ) as std::ffi::c_int
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn ncclAllReduce(
    sendbuff: *const std::ffi::c_void,
    recvbuff: *mut std::ffi::c_void,
    count: usize,
    datatype: std::ffi::c_int,
    op: std::ffi::c_int,
    comm: *mut std::ffi::c_void,
    stream: *mut std::ffi::c_void
) -> std::ffi::c_int {
    eprintln!("[NCCL-LibEntry] ncclAllReduce called with count={}", count);
    
    // 转换参数类型并调用实时迁移系统
    let nccl_datatype = match datatype {
        0 => crate::r#impl::nccl_live_migration::NcclDataType::Int8,
        1 => crate::r#impl::nccl_live_migration::NcclDataType::Uint8,
        2 => crate::r#impl::nccl_live_migration::NcclDataType::Int32,
        3 => crate::r#impl::nccl_live_migration::NcclDataType::Uint32,
        4 => crate::r#impl::nccl_live_migration::NcclDataType::Int64,
        5 => crate::r#impl::nccl_live_migration::NcclDataType::Uint64,
        6 => crate::r#impl::nccl_live_migration::NcclDataType::Float16,
        7 => crate::r#impl::nccl_live_migration::NcclDataType::Float32,
        8 => crate::r#impl::nccl_live_migration::NcclDataType::Float64,
        _ => crate::r#impl::nccl_live_migration::NcclDataType::Float32,
    };
    
    let nccl_op = match op {
        0 => crate::r#impl::nccl_live_migration::NcclRedOp::Sum,
        1 => crate::r#impl::nccl_live_migration::NcclRedOp::Prod,
        2 => crate::r#impl::nccl_live_migration::NcclRedOp::Max,
        3 => crate::r#impl::nccl_live_migration::NcclRedOp::Min,
        4 => crate::r#impl::nccl_live_migration::NcclRedOp::Avg,
        _ => crate::r#impl::nccl_live_migration::NcclRedOp::Sum,
    };
    
    crate::r#impl::nccl_live_migration::ncclAllReduce_with_fault_tolerance(
        sendbuff,
        recvbuff,
        count,
        nccl_datatype,
        nccl_op,
        comm,
        stream,
    ) as std::ffi::c_int
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn ncclBroadcast(
    sendbuff: *const std::ffi::c_void,
    recvbuff: *mut std::ffi::c_void,
    count: usize,
    datatype: std::ffi::c_int,
    root: std::ffi::c_int,
    comm: *mut std::ffi::c_void,
    stream: *mut std::ffi::c_void
) -> std::ffi::c_int {
    use crate::r#impl::nccl_fault_tolerant::NcclDataType;
    
    let datatype = std::mem::transmute::<std::ffi::c_int, NcclDataType>(datatype);
    
    crate::r#impl::nccl_fault_tolerant::ncclBroadcast(
        sendbuff, recvbuff, count, datatype, root, comm as usize, stream
    ) as std::ffi::c_int
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn ncclCommDestroy(comm: *mut std::ffi::c_void) -> std::ffi::c_int {
    crate::r#impl::nccl_fault_tolerant::ncclCommDestroy(comm as usize) as std::ffi::c_int
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn ncclGetErrorString(result: std::ffi::c_int) -> *const std::ffi::c_char {
    use crate::r#impl::nccl_fault_tolerant::NcclResult;
    
    let result = std::mem::transmute::<std::ffi::c_int, NcclResult>(result);
    crate::r#impl::nccl_fault_tolerant::ncclGetErrorString(result)
}
