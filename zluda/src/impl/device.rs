use super::context;
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
use std::{ffi::c_void, mem, ptr};
#[cfg(feature = "intel")]
use ze_runtime_sys::*;

const PROJECT_SUFFIX: &[u8] = b" [ZLUDA]\0";
pub const COMPUTE_CAPABILITY_MAJOR: i32 = 8;
pub const COMPUTE_CAPABILITY_MINOR: i32 = 0;

#[cfg(feature = "amd")]
pub(crate) fn compute_capability(major: &mut i32, minor: &mut i32, _dev: hipDevice_t) -> CUresult {
    *major = COMPUTE_CAPABILITY_MAJOR;
    *minor = COMPUTE_CAPABILITY_MINOR;
    Ok(())
}
#[cfg(feature = "amd")]
macro_rules! remap_attribute {
    ($attrib:expr => $([ $($word:expr)* ]),*,) => {
        match $attrib {
            $(
                paste::paste! { CUdevice_attribute:: [< CU_DEVICE_ATTRIBUTE $(_ $word:upper)* >] } => {
                    paste::paste! { hipDeviceAttribute_t:: [< hipDeviceAttribute $($word:camel)* >] }
                }
            )*
            _ => return Err(hipErrorCode_t::NotSupported)
        }
    }
}

#[cfg(feature = "intel")]
pub(crate) fn compute_capability(
    major: &mut i32,
    minor: &mut i32,
    _dev: ze_device_handle_t,
) -> CUresult {
    *major = COMPUTE_CAPABILITY_MAJOR;
    *minor = COMPUTE_CAPABILITY_MINOR;
    Ok(())
}

#[cfg(feature = "amd")]
pub(crate) fn get(device: *mut hipDevice_t, ordinal: i32) -> hipError_t {
    unsafe { hipDeviceGet(device, ordinal) }
}

#[cfg(feature = "intel")]
pub(crate) fn get(device: *mut i32, ordinal: i32) -> CUresult {
    eprintln!("[hetGPU] cuDeviceGet called with ordinal={}", ordinal);
    // Use global_state to get device (supports virtual device fallback)
    let devices = super::driver::global_state()?;
    if ordinal < 0 || ordinal >= devices.devices.len() as i32 {
        eprintln!("[hetGPU] cuDeviceGet: INVALID_DEVICE (ordinal out of range)");
        return Err(CUerror::INVALID_DEVICE);
    }
    unsafe { *device = ordinal };
    eprintln!("[hetGPU] cuDeviceGet: success, device={}", ordinal);
    Ok(())
}

#[cfg(feature = "amd")]
#[allow(warnings)]
trait DeviceAttributeNames {
    const hipDeviceAttributeGpuOverlap: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeDeviceOverlap;
    const hipDeviceAttributeMaximumTexture1DWidth: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeMaxTexture1DWidth;
    const hipDeviceAttributeMaximumTexture2DWidth: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeMaxTexture2DWidth;
    const hipDeviceAttributeMaximumTexture2DHeight: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeMaxTexture2DHeight;
    const hipDeviceAttributeMaximumTexture3DWidth: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeMaxTexture3DWidth;
    const hipDeviceAttributeMaximumTexture3DHeight: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeMaxTexture3DHeight;
    const hipDeviceAttributeMaximumTexture3DDepth: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeMaxTexture3DDepth;
    const hipDeviceAttributeGlobalMemoryBusWidth: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeMemoryBusWidth;
    const hipDeviceAttributeMaxThreadsPerMultiprocessor: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeMaxThreadsPerMultiProcessor;
    const hipDeviceAttributeAsyncEngineCount: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeConcurrentKernels;
    const hipDeviceAttributePciDomainId: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributePciDomainID;
    const hipDeviceAttributeMultiGpuBoard: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeIsMultiGpuBoard;
    const hipDeviceAttributeMultiGpuBoardGroupId: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeMultiGpuBoardGroupID;
    const hipDeviceAttributeMaxSharedMemoryPerBlockOptin: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeSharedMemPerBlockOptin;
}

#[cfg(feature = "intel")]
#[allow(warnings)]
trait DeviceAttributeNames {
    const CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP;
    const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH;
    const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH;
    const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT;
    const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH;
    const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT;
    const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH;
    const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH;
    const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR;
    const CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT;
    const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID;
    const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD;
    const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID;
    const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: CUdevice_attribute =
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN;
}

#[cfg(feature = "amd")]
impl DeviceAttributeNames for hipDeviceAttribute_t {}

#[cfg(feature = "intel")]
impl DeviceAttributeNames for CUdevice_attribute {}

#[cfg(feature = "amd")]
pub(crate) fn get_attribute(
    pi: &mut i32,
    attrib: CUdevice_attribute,
    dev_idx: hipDevice_t,
) -> hipError_t {
    fn get_device_prop(
        pi: &mut i32,
        dev_idx: hipDevice_t,
        f: impl FnOnce(&hipDeviceProp_tR0600) -> i32,
    ) -> hipError_t {
        let mut props = unsafe { mem::zeroed() };
        unsafe { hipGetDeviceProperties(&mut props, dev_idx)? };
        *pi = f(&props);
        Ok(())
    }
    match attrib {
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE => {
            *pi = 32;
            return Ok(());
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TCC_DRIVER => {
            *pi = 0;
            return Ok(());
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture2DLayered[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture2DLayered[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture2DLayered[2])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture1DLayered[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture1DLayered[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER => {
            return get_device_prop(pi, dev_idx, |props| {
                (props.maxTexture2DGather[0] > 0 && props.maxTexture2DGather[1] > 0) as i32
            })
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture2DGather[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture2DGather[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture3DAlt[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture3DAlt[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture3DAlt[2])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxTextureCubemap)
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxTextureCubemapLayered[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS => {
            return get_device_prop(pi, dev_idx, |props| props.maxTextureCubemapLayered[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurface1D)
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurface2D[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurface2D[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurface3D[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurface3D[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurface3D[2])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurface1DLayered[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurface1DLayered[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurface2DLayered[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurface2DLayered[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurface2DLayered[2])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurfaceCubemap)
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurfaceCubemapLayered[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS => {
            return get_device_prop(pi, dev_idx, |props| props.maxSurfaceCubemapLayered[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture1DLinear)
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture2DLinear[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture2DLinear[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture2DLinear[2])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture2DMipmap[0])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture2DMipmap[1])
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => {
            *pi = COMPUTE_CAPABILITY_MAJOR;
            return Ok(());
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => {
            *pi = COMPUTE_CAPABILITY_MINOR;
            return Ok(());
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH => {
            return get_device_prop(pi, dev_idx, |props| props.maxTexture1DMipmap)
        }
        _ => {}
    }
    let attrib = remap_attribute! {
        attrib =>
        [MAX THREADS PER BLOCK],
        [MAX BLOCK DIM X],
        [MAX BLOCK DIM Y],
        [MAX BLOCK DIM Z],
        [MAX GRID DIM X],
        [MAX GRID DIM Y],
        [MAX GRID DIM Z],
        [MAX SHARED MEMORY PER BLOCK],
        [TOTAL CONSTANT MEMORY],
        //[WARP SIZE],
        [MAX PITCH],
        [MAX REGISTERS PER BLOCK],
        [CLOCK RATE],
        [TEXTURE ALIGNMENT],
        [GPU OVERLAP],
        [MULTIPROCESSOR COUNT],
        [KERNEL EXEC TIMEOUT],
        [INTEGRATED],
        [CAN MAP HOST MEMORY],
        [COMPUTE MODE],
        [MAXIMUM TEXTURE1D WIDTH],
        [MAXIMUM TEXTURE2D WIDTH],
        [MAXIMUM TEXTURE2D HEIGHT],
        [MAXIMUM TEXTURE3D WIDTH],
        [MAXIMUM TEXTURE3D HEIGHT],
        [MAXIMUM TEXTURE3D DEPTH],
        //[MAXIMUM TEXTURE2D LAYERED WIDTH],
        //[MAXIMUM TEXTURE2D LAYERED HEIGHT],
        //[MAXIMUM TEXTURE2D LAYERED LAYERS],
        //[MAXIMUM TEXTURE2D ARRAY WIDTH],
        //[MAXIMUM TEXTURE2D ARRAY HEIGHT],
        //[MAXIMUM TEXTURE2D ARRAY NUMSLICES],
        [SURFACE ALIGNMENT],
        [CONCURRENT KERNELS],
        [ECC ENABLED],
        [PCI BUS ID],
        [PCI DEVICE ID],
        //[TCC DRIVER],
        [MEMORY CLOCK RATE],
        [GLOBAL MEMORY BUS WIDTH],
        [L2 CACHE SIZE],
        [MAX THREADS PER MULTIPROCESSOR],
        [ASYNC ENGINE COUNT],
        [UNIFIED ADDRESSING],
        //[MAXIMUM TEXTURE1D LAYERED WIDTH],
        //[MAXIMUM TEXTURE1D LAYERED LAYERS],
        //[CAN TEX2D GATHER],
        //[MAXIMUM TEXTURE2D GATHER WIDTH],
        //[MAXIMUM TEXTURE2D GATHER HEIGHT],
        //[MAXIMUM TEXTURE3D WIDTH ALTERNATE],
        //[MAXIMUM TEXTURE3D HEIGHT ALTERNATE],
        //[MAXIMUM TEXTURE3D DEPTH ALTERNATE],
        [PCI DOMAIN ID],
        [TEXTURE PITCH ALIGNMENT],
        //[MAXIMUM TEXTURECUBEMAP WIDTH],
        //[MAXIMUM TEXTURECUBEMAP LAYERED WIDTH],
        //[MAXIMUM TEXTURECUBEMAP LAYERED LAYERS],
        //[MAXIMUM SURFACE1D WIDTH],
        //[MAXIMUM SURFACE2D WIDTH],
        //[MAXIMUM SURFACE2D HEIGHT],
        //[MAXIMUM SURFACE3D WIDTH],
        //[MAXIMUM SURFACE3D HEIGHT],
        //[MAXIMUM SURFACE3D DEPTH],
        //[MAXIMUM SURFACE1D LAYERED WIDTH],
        //[MAXIMUM SURFACE1D LAYERED LAYERS],
        //[MAXIMUM SURFACE2D LAYERED WIDTH],
        //[MAXIMUM SURFACE2D LAYERED HEIGHT],
        //[MAXIMUM SURFACE2D LAYERED LAYERS],
        //[MAXIMUM SURFACECUBEMAP WIDTH],
        //[MAXIMUM SURFACECUBEMAP LAYERED WIDTH],
        //[MAXIMUM SURFACECUBEMAP LAYERED LAYERS],
        //[MAXIMUM TEXTURE1D LINEAR WIDTH],
        //[MAXIMUM TEXTURE2D LINEAR WIDTH],
        //[MAXIMUM TEXTURE2D LINEAR HEIGHT],
        //[MAXIMUM TEXTURE2D LINEAR PITCH],
        //[MAXIMUM TEXTURE2D MIPMAPPED WIDTH],
        //[MAXIMUM TEXTURE2D MIPMAPPED HEIGHT],
        //[COMPUTE CAPABILITY MAJOR],
        //[COMPUTE CAPABILITY MINOR],
        //[MAXIMUM TEXTURE1D MIPMAPPED WIDTH],
        [STREAM PRIORITIES SUPPORTED],
        [GLOBAL L1 CACHE SUPPORTED],
        [LOCAL L1 CACHE SUPPORTED],
        [MAX SHARED MEMORY PER MULTIPROCESSOR],
        [MAX REGISTERS PER MULTIPROCESSOR],
        [MANAGED MEMORY],
        [MULTI GPU BOARD],
        [MULTI GPU BOARD GROUP ID],
        [HOST NATIVE ATOMIC SUPPORTED],
        [SINGLE TO DOUBLE PRECISION PERF RATIO],
        [PAGEABLE MEMORY ACCESS],
        [CONCURRENT MANAGED ACCESS],
        [COMPUTE PREEMPTION SUPPORTED],
        [CAN USE HOST POINTER FOR REGISTERED MEM],
        //[CAN USE STREAM MEM OPS],
        [COOPERATIVE LAUNCH],
        [COOPERATIVE MULTI DEVICE LAUNCH],
        [MAX SHARED MEMORY PER BLOCK OPTIN],
        //[CAN FLUSH REMOTE WRITES],
        [HOST REGISTER SUPPORTED],
        [PAGEABLE MEMORY ACCESS USES HOST PAGE TABLES],
        [DIRECT MANAGED MEM ACCESS FROM HOST],
        //[VIRTUAL ADDRESS MANAGEMENT SUPPORTED],
        [VIRTUAL MEMORY MANAGEMENT SUPPORTED],
        //[HANDLE TYPE POSIX FILE DESCRIPTOR SUPPORTED],
        //[HANDLE TYPE WIN32 HANDLE SUPPORTED],
        //[HANDLE TYPE WIN32 KMT HANDLE SUPPORTED],
        //[MAX BLOCKS PER MULTIPROCESSOR],
        //[GENERIC COMPRESSION SUPPORTED],
        //[MAX PERSISTING L2 CACHE SIZE],
        //[MAX ACCESS POLICY WINDOW SIZE],
        //[GPU DIRECT RDMA WITH CUDA VMM SUPPORTED],
        //[RESERVED SHARED MEMORY PER BLOCK],
        //[SPARSE CUDA ARRAY SUPPORTED],
        //[READ ONLY HOST REGISTER SUPPORTED],
        //[TIMELINE SEMAPHORE INTEROP SUPPORTED],
        [MEMORY POOLS SUPPORTED],
        //[GPU DIRECT RDMA SUPPORTED],
        //[GPU DIRECT RDMA FLUSH WRITES OPTIONS],
        //[GPU DIRECT RDMA WRITES ORDERING],
        //[MEMPOOL SUPPORTED HANDLE TYPES],
        //[CLUSTER LAUNCH],
        //[DEFERRED MAPPING CUDA ARRAY SUPPORTED],
        //[CAN USE 64 BIT STREAM MEM OPS],
        //[CAN USE STREAM WAIT VALUE NOR],
        //[DMA BUF SUPPORTED],
        //[IPC EVENT SUPPORTED],
        //[MEM SYNC DOMAIN COUNT],
        //[TENSOR MAP ACCESS SUPPORTED],
        //[HANDLE TYPE FABRIC SUPPORTED],
        //[UNIFIED FUNCTION POINTERS],
        //[NUMA CONFIG],
        //[NUMA ID],
        //[MULTICAST SUPPORTED],
        //[MPS ENABLED],
        //[HOST NUMA ID],
    };
    unsafe { hipDeviceGetAttribute(pi, attrib, dev_idx) }
}

#[cfg(feature = "intel")]
pub(crate) fn get_attribute(
    pi: &mut i32,
    attrib: CUdevice_attribute,
    dev_idx: ze_device_handle_t,
) -> ze_result_t {
    eprintln!(
        "[hetGPU] get_attribute called: attrib={:?}, dev_idx={:?}",
        attrib, dev_idx
    );
    // Check if this is a virtual device (null handle) - return GPU-like defaults
    let is_virtual = dev_idx.0.is_null();
    eprintln!("[hetGPU] is_virtual={}", is_virtual);

    if is_virtual {
        // Virtual device - return GPU-like values without querying hardware
        match attrib {
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => *pi = COMPUTE_CAPABILITY_MAJOR,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => *pi = COMPUTE_CAPABILITY_MINOR,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE => *pi = 32,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK => *pi = 1024,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X => *pi = 1024,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y => *pi = 1024,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z => *pi = 64,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X => *pi = 2147483647,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y => *pi = 65535,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z => *pi = 65535,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => *pi = 49152, // 48KB
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY => *pi = 65536,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH => *pi = 2147483647,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK => *pi = 65536,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE => *pi = 1410000, // 1.41 GHz
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT => *pi = 512,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => *pi = 80,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT => *pi = 0,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED => *pi = 0,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE => *pi = 0,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH => *pi = 131072,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH => *pi = 131072,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT => *pi = 65536,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH => *pi = 16384,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT => *pi = 16384,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH => *pi = 16384,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ECC_ENABLED => *pi = 0,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID => *pi = 0,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID => *pi = 0,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE => *pi = 5001000, // 5 GHz
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH => *pi = 384, // bits
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE => *pi = 6291456, // 6MB
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR => *pi = 2048,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT => *pi = 2,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID => *pi = 0,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR => *pi = 102400, // 100KB
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR => *pi = 65536,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD => *pi = 0,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID => *pi = 0,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO => *pi = 2,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN => *pi = 49152,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED => *pi = 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED => *pi = 1,
            _ => return ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE,
        }
        return ze_result_t::ZE_RESULT_SUCCESS;
    }

    // Real device - query hardware properties
    let mut props: ze_device_properties_t = unsafe { mem::zeroed() };
    props.stype = ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    unsafe {
        match zeDeviceGetProperties(dev_idx, &mut props) {
            ze_result_t::ZE_RESULT_SUCCESS => {}
            e => return e,
        }
    }

    match attrib {
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => {
            *pi = COMPUTE_CAPABILITY_MAJOR;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => {
            *pi = COMPUTE_CAPABILITY_MINOR;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE => {
            *pi = 32;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK => {
            *pi = props.numThreadsPerEU as i32 * 32; // Estimate based on EU count
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X => {
            *pi = props.numThreadsPerEU as i32 * 8; // Estimate based on EU count
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y => {
            *pi = 65535;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z => {
            *pi = 65535;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X => {
            *pi = 65535;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y => {
            *pi = 65535;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z => {
            *pi = 65535;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => {
            *pi = props.maxMemAllocSize as i32;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY => {
            *pi = 65536; // Default size for constant memory
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH => {
            *pi = props.maxMemAllocSize as i32;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK => {
            *pi = 64; // Default value since there's no direct equivalent
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE => {
            *pi = props.coreClockRate as i32;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => {
            // For TMatmul backend, return fixed value to avoid SIGFPE
            #[cfg(feature = "tmatmul")]
            {
                eprintln!("[hetGPU] MULTIPROCESSOR_COUNT: returning 80 (tmatmul feature)");
                *pi = 80;
            }
            #[cfg(not(feature = "tmatmul"))]
            {
                eprintln!(
                    "[hetGPU] MULTIPROCESSOR_COUNT: returning {} (intel path)",
                    props.numEUsPerSubslice
                );
                *pi = props.numEUsPerSubslice as i32;
            }
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED => {
            *pi = 0;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE => {
            *pi = 0;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH => {
            *pi = 16384; // Default value
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH => {
            *pi = 16384; // Default value
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT => {
            *pi = 16384; // Default value
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH => {
            *pi = 2048; // Default value
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT => {
            *pi = 2048; // Default value
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH => {
            *pi = 2048; // Default value
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ECC_ENABLED => {
            *pi = 0;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID => {
            *pi = 0; // Default value
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID => {
            *pi = 0; // Default value
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE => {
            *pi = 0; // Not available in Level Zero API
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH => {
            *pi = 0; // Not available in Level Zero API
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE => {
            *pi = props.maxMemAllocSize as i32 / 10; // Estimate
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR => {
            // For TMatmul backend, return fixed value to avoid SIGFPE
            #[cfg(feature = "tmatmul")]
            {
                eprintln!(
                    "[hetGPU] MAX_THREADS_PER_MULTIPROCESSOR: returning 2048 (tmatmul feature)"
                );
                *pi = 2048;
            }
            #[cfg(not(feature = "tmatmul"))]
            {
                eprintln!(
                    "[hetGPU] MAX_THREADS_PER_MULTIPROCESSOR: returning {} (intel path)",
                    props.numThreadsPerEU
                );
                *pi = props.numThreadsPerEU as i32;
            }
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID => {
            *pi = 0; // Default value
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR => {
            *pi = props.maxMemAllocSize as i32;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR => {
            *pi = 64; // Default value
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD => {
            *pi = 0;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID => {
            *pi = 0;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO => {
            *pi = 2;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN => {
            *pi = props.maxMemAllocSize as i32;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED => {
            *pi = 1;
            ze_result_t::ZE_RESULT_SUCCESS
        }
        _ => ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE,
    }
}

#[cfg(feature = "amd")]
pub(crate) fn get_uuid(uuid: *mut hipUUID, device: hipDevice_t) -> hipError_t {
    unsafe { hipDeviceGetUuid(uuid, device) }
}

#[cfg(feature = "intel")]
pub(crate) fn get_uuid(uuid: *mut ze_device_uuid_t, device: ze_device_handle_t) -> ze_result_t {
    let mut props: ze_device_properties_t = unsafe { mem::zeroed() };
    props.stype = ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    unsafe {
        match zeDeviceGetProperties(device, &mut props) {
            ze_result_t::ZE_RESULT_SUCCESS => {}
            e => return e,
        }
    }

    // Copy UUID from device properties
    unsafe {
        ptr::copy_nonoverlapping(props.uuid.id.as_ptr(), (*uuid).id.as_mut_ptr(), 16);
    }

    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(feature = "amd")]
pub(crate) fn get_uuid_v2(uuid: *mut hipUUID, device: hipDevice_t) -> hipError_t {
    get_uuid(uuid, device)
}

#[cfg(feature = "intel")]
pub(crate) fn get_uuid_v2(uuid: *mut ze_device_uuid_t, device: ze_device_handle_t) -> ze_result_t {
    get_uuid(uuid, device)
}

#[cfg(feature = "amd")]
pub(crate) fn get_luid(
    luid: *mut ::core::ffi::c_char,
    device_node_mask: &mut ::core::ffi::c_uint,
    dev: hipDevice_t,
) -> hipError_t {
    let luid = unsafe {
        luid.cast::<[i8; 8]>()
            .as_mut()
            .ok_or(hipErrorCode_t::InvalidValue)
    }?;
    let mut properties = unsafe { mem::zeroed() };
    unsafe { hipGetDeviceProperties(&mut properties, dev) }?;
    *luid = properties.luid;
    *device_node_mask = properties.luidDeviceNodeMask;
    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn get_luid(
    luid: *mut ::core::ffi::c_char,
    device_node_mask: &mut ::core::ffi::c_uint,
    dev: ze_device_handle_t,
) -> ze_result_t {
    let luid = unsafe {
        luid.cast::<[i8; 8]>()
            .as_mut()
            .ok_or(ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT)
    }
    .unwrap();

    let mut props: ze_device_properties_t = unsafe { mem::zeroed() };
    props.stype = ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    unsafe {
        let _ = zeDeviceGetProperties(dev, &mut props);
    }

    // Set LUID and device node mask
    unsafe {
        ptr::copy_nonoverlapping(props.uuid.id.as_ptr(), luid.as_mut_ptr() as *mut u8, 8);
    }
    *device_node_mask = 1;

    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(feature = "amd")]
pub(crate) fn get_name(
    name: *mut ::core::ffi::c_char,
    len: ::core::ffi::c_int,
    dev: hipDevice_t,
) -> CUresult {
    unsafe { hipDeviceGetName(name, len, dev).unwrap() };
    let len = len as usize;
    let buffer = unsafe { std::slice::from_raw_parts(name, len) };
    let first_zero = buffer.iter().position(|c| *c == 0);
    let first_zero = if let Some(x) = first_zero {
        x
    } else {
        return Ok(());
    };
    if (first_zero + PROJECT_SUFFIX.len()) > len {
        return Ok(());
    }
    unsafe {
        ptr::copy_nonoverlapping(
            PROJECT_SUFFIX.as_ptr() as _,
            name.add(first_zero),
            PROJECT_SUFFIX.len(),
        )
    };
    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn get_name(
    name: *mut ::core::ffi::c_char,
    len: ::core::ffi::c_int,
    dev: ze_device_handle_t,
) -> CUresult {
    // Check if this is a virtual device
    if dev.0.is_null() {
        let device_name = b"Virtual GPU (TMatmul)\0";
        let copy_len = std::cmp::min(device_name.len(), len as usize);
        unsafe {
            ptr::copy_nonoverlapping(
                device_name.as_ptr() as *const ::core::ffi::c_char,
                name,
                copy_len,
            );
        }
        return Ok(());
    }

    // Real device - query hardware
    let mut props: ze_device_properties_t = unsafe { mem::zeroed() };
    props.stype = ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    unsafe {
        let result = zeDeviceGetProperties(dev, &mut props);
        if result != ze_result_t::ZE_RESULT_SUCCESS {
            return Err(CUerror::UNKNOWN);
        }
    }

    let name_len = props.name.len();
    if name_len >= len as usize {
        return Ok(());
    }

    unsafe {
        ptr::copy_nonoverlapping(props.name.as_ptr(), name, name_len);
    }

    let len = len as usize;
    let buffer = unsafe { std::slice::from_raw_parts(name, len) };
    let first_zero = buffer.iter().position(|c| *c == 0);
    let first_zero = if let Some(x) = first_zero {
        x
    } else {
        return Ok(());
    };
    if (first_zero + PROJECT_SUFFIX.len()) > len {
        return Ok(());
    }
    unsafe {
        ptr::copy_nonoverlapping(
            PROJECT_SUFFIX.as_ptr() as _,
            name.add(first_zero),
            PROJECT_SUFFIX.len(),
        )
    };
    Ok(())
}

#[cfg(feature = "amd")]
pub(crate) fn total_mem_v2(bytes: *mut usize, dev: hipDevice_t) -> hipError_t {
    unsafe { hipDeviceTotalMem(bytes, dev) }
}

#[cfg(feature = "intel")]
pub(crate) fn total_mem_v2(bytes: *mut usize, dev: ze_device_handle_t) -> ze_result_t {
    if bytes.is_null() {
        return ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    let mut props: ze_device_properties_t = unsafe { mem::zeroed() };
    props.stype = ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    let result = unsafe { zeDeviceGetProperties(dev, &mut props) };

    // If Level Zero query fails (e.g., virtual device), return 16GB as default
    if result != ze_result_t::ZE_RESULT_SUCCESS || props.maxMemAllocSize == 0 {
        unsafe {
            *bytes = 16 * 1024 * 1024 * 1024; // 16GB virtual device memory
        }
    } else {
        unsafe {
            *bytes = props.maxMemAllocSize as usize;
        }
    }

    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(feature = "amd")]
pub(crate) fn get_properties(prop: &mut CUdevprop, dev: hipDevice_t) -> hipError_t {
    let mut hip_props = unsafe { mem::zeroed() };
    unsafe { hipGetDeviceProperties(&mut hip_props, dev) }?;
    prop.maxThreadsPerBlock = hip_props.maxThreadsPerBlock;
    prop.maxThreadsDim = hip_props.maxThreadsDim;
    prop.maxGridSize = hip_props.maxGridSize;
    prop.totalConstantMemory = clamp_usize(hip_props.totalConstMem);
    prop.SIMDWidth = 32;
    prop.memPitch = clamp_usize(hip_props.memPitch);
    prop.regsPerBlock = hip_props.regsPerBlock;
    prop.clockRate = hip_props.clockRate;
    prop.textureAlign = clamp_usize(hip_props.textureAlignment);
    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn get_properties(prop: &mut CUdevprop, dev: ze_device_handle_t) -> ze_result_t {
    let mut props: ze_device_properties_t = unsafe { mem::zeroed() };
    props.stype = ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    unsafe {
        let _ = zeDeviceGetProperties(dev, &mut props);
    }

    prop.maxThreadsPerBlock = props.numEUsPerSubslice as i32 * props.numThreadsPerEU as i32;
    prop.maxThreadsDim = [prop.maxThreadsPerBlock, 65535, 65535];
    prop.maxGridSize = [65535, 65535, 65535];
    prop.totalConstantMemory = clamp_usize(props.maxMemAllocSize as usize);
    prop.SIMDWidth = 32;
    prop.memPitch = clamp_usize(props.maxMemAllocSize as usize);
    prop.regsPerBlock = props.numThreadsPerEU as i32;
    prop.clockRate = props.coreClockRate as i32;
    prop.textureAlign = 1;

    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(feature = "amd")]
pub(crate) fn get_count(count: &mut ::core::ffi::c_int) -> hipError_t {
    unsafe { hipGetDeviceCount(count) }
}

#[cfg(feature = "intel")]
pub(crate) fn get_count(count: &mut ::core::ffi::c_int) -> ze_result_t {
    eprintln!("[hetGPU] cuDeviceGetCount (driver API) called");
    // Use global_state to get device count (this creates virtual device if needed)
    match super::driver::global_state() {
        Ok(state) => {
            *count = state.devices.len() as ::core::ffi::c_int;
            eprintln!("[hetGPU] cuDeviceGetCount: success, count={}", *count);
            ze_result_t::ZE_RESULT_SUCCESS
        }
        Err(e) => {
            eprintln!("[hetGPU] cuDeviceGetCount: failed with {:?}", e);
            *count = 0;
            ze_result_t::ZE_RESULT_ERROR_DEVICE_LOST
        }
    }
}

fn clamp_usize(x: usize) -> i32 {
    usize::min(x, i32::MAX as usize) as i32
}

#[cfg(feature = "amd")]
pub(crate) fn primary_context_retain(
    pctx: &mut CUcontext,
    hip_dev: hipDevice_t,
) -> Result<(), CUerror> {
    let (ctx, raw_ctx) = context::get_primary(hip_dev)?;
    {
        let mut mutable_ctx = ctx.mutable.lock().map_err(|_| CUerror::UNKNOWN)?;
        mutable_ctx.ref_count += 1;
    }
    *pctx = raw_ctx;
    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn primary_context_retain(pctx: &mut CUcontext, dev: i32) -> Result<(), CUerror> {
    eprintln!("[hetGPU] cuDevicePrimaryCtxRetain called with dev={}", dev);
    // Get device from global state using ordinal
    let device_obj = super::driver::device(dev)?;
    let (ctx, raw_ctx) = device_obj.primary_context();
    {
        let mut mutable_ctx = ctx.mutable.lock().map_err(|_| CUerror::UNKNOWN)?;
        mutable_ctx.ref_count += 1;
    }
    *pctx = raw_ctx;
    eprintln!(
        "[hetGPU] cuDevicePrimaryCtxRetain: success, ctx={:?}",
        raw_ctx
    );
    Ok(())
}

#[cfg(feature = "amd")]
pub(crate) fn primary_context_release(hip_dev: hipDevice_t) -> Result<(), CUerror> {
    let (ctx, _) = context::get_primary(hip_dev)?;
    {
        let mut mutable_ctx = ctx.mutable.lock().map_err(|_| CUerror::UNKNOWN)?;
        if mutable_ctx.ref_count == 0 {
            return Err(CUerror::INVALID_CONTEXT);
        }
        mutable_ctx.ref_count -= 1;
        if mutable_ctx.ref_count == 0 {
            // Clean up all resources owned by this context
            // When the reference count drops to zero, we should clean up all resources
            // owned by this context to prevent memory leaks
            #[cfg(feature = "intel")]
            {
                // Clean up streams
                mutable_ctx.streams.clear();

                // Clean up modules
                mutable_ctx.modules.clear();

                // Clean up memory allocations
                for ptr in mutable_ctx.memory.drain() {
                    unsafe { hipFree(ptr) }.ok();
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn primary_context_release(dev: i32) -> Result<(), CUerror> {
    // Get device from global state using ordinal
    let device_obj = super::driver::device(dev)?;
    let (ctx, _) = device_obj.primary_context();
    {
        let mut mutable_ctx = ctx.mutable.lock().map_err(|_| CUerror::UNKNOWN)?;
        if mutable_ctx.ref_count == 0 {
            return Err(CUerror::INVALID_CONTEXT);
        }
        mutable_ctx.ref_count -= 1;
        if mutable_ctx.ref_count == 0 {
            // Clean up all resources owned by this context
            // When the reference count drops to zero, we should clean up all resources
            // owned by this context to prevent memory leaks
            #[cfg(feature = "intel")]
            {
                // Clean up command queues
                let queues: Vec<_> = mutable_ctx._command_queues.iter().copied().collect();
                for queue in queues {
                    let _ = unsafe { zeCommandQueueDestroy(queue) };
                }
                mutable_ctx._command_queues.clear();

                // Clean up command lists
                let lists: Vec<_> = mutable_ctx._command_lists.iter().copied().collect();
                for list in lists {
                    let _ = unsafe { zeCommandListDestroy(list) };
                }
                mutable_ctx._command_lists.clear();

                // Clean up modules
                let modules: Vec<_> = mutable_ctx._modules.iter().copied().collect();
                for module in modules {
                    let _ = unsafe { zeModuleDestroy(module) };
                }
                mutable_ctx._modules.clear();

                // Clean up memory allocations
                let allocations: Vec<_> = mutable_ctx._allocations.iter().copied().collect();
                for ptr in allocations {
                    let _ = unsafe { zeMemFree(ctx.context, ptr as *mut c_void) };
                }
                mutable_ctx._allocations.clear();
            }
        }
    }
    Ok(())
}

// Add trait for converting ze_result_t to CUresult
#[cfg(feature = "intel")]
trait ZeResultExt {
    fn to_cu_result(self) -> CUresult;
}

#[cfg(feature = "intel")]
impl ZeResultExt for ze_result_t {
    fn to_cu_result(self) -> CUresult {
        match self {
            ze_result_t::ZE_RESULT_SUCCESS => Ok(()),
            ze_result_t::ZE_RESULT_ERROR_DEVICE_LOST => Err(CUerror::DEVICE_NOT_LICENSED),
            ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY => Err(CUerror::OUT_OF_MEMORY),
            ze_result_t::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY => Err(CUerror::OUT_OF_MEMORY),
            ze_result_t::ZE_RESULT_ERROR_MODULE_BUILD_FAILURE => Err(CUerror::INVALID_PTX),
            ze_result_t::ZE_RESULT_ERROR_MODULE_LINK_FAILURE => Err(CUerror::INVALID_PTX),
            ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT => Err(CUerror::INVALID_VALUE),
            ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_HANDLE => Err(CUerror::INVALID_HANDLE),
            ze_result_t::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE => Err(CUerror::INVALID_CONTEXT),
            ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_POINTER => Err(CUerror::INVALID_VALUE),
            ze_result_t::ZE_RESULT_ERROR_INVALID_SIZE => Err(CUerror::INVALID_VALUE),
            ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_SIZE => Err(CUerror::INVALID_VALUE),
            ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE => Err(CUerror::NOT_SUPPORTED),
            ze_result_t::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT => {
                Err(CUerror::INVALID_HANDLE)
            }
            ze_result_t::ZE_RESULT_ERROR_INVALID_ENUMERATION => Err(CUerror::INVALID_VALUE),
            ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION => Err(CUerror::NOT_SUPPORTED),
            ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT => Err(CUerror::INVALID_VALUE),
            ze_result_t::ZE_RESULT_ERROR_INVALID_NATIVE_BINARY => Err(CUerror::INVALID_PTX),
            ze_result_t::ZE_RESULT_ERROR_INVALID_GLOBAL_NAME => Err(CUerror::NOT_FOUND),
            ze_result_t::ZE_RESULT_ERROR_INVALID_KERNEL_NAME => Err(CUerror::NOT_FOUND),
            ze_result_t::ZE_RESULT_ERROR_INVALID_FUNCTION_NAME => Err(CUerror::NOT_FOUND),
            ze_result_t::ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION => {
                Err(CUerror::INVALID_VALUE)
            }
            ze_result_t::ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION => {
                Err(CUerror::INVALID_VALUE)
            }
            ze_result_t::ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX => {
                Err(CUerror::INVALID_VALUE)
            }
            ze_result_t::ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE => {
                Err(CUerror::INVALID_VALUE)
            }
            ze_result_t::ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE => {
                Err(CUerror::INVALID_VALUE)
            }
            ze_result_t::ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED => Err(CUerror::INVALID_PTX),
            ze_result_t::ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE => Err(CUerror::INVALID_VALUE),
            ze_result_t::ZE_RESULT_ERROR_OVERLAPPING_REGIONS => Err(CUerror::INVALID_VALUE),
            _ => Err(CUerror::UNKNOWN),
        }
    }
}

// Tenstorrent device function implementations
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn compute_capability(major: &mut i32, minor: &mut i32, _dev: i32) -> CUresult {
    *major = COMPUTE_CAPABILITY_MAJOR;
    *minor = COMPUTE_CAPABILITY_MINOR;
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get(device: *mut i32, ordinal: i32) -> CUresult {
    let devices = super::driver::global_state()?;
    if ordinal < 0 || ordinal >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }
    unsafe { *device = ordinal };
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_attribute(
    pi: *mut ::core::ffi::c_int,
    attrib: CUdevice_attribute,
    dev: i32,
) -> CUresult {
    if pi.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    let result = match attrib {
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK => 1024,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X => 1024,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y => 1024,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z => 64,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X => 65535,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y => 65535,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z => 65535,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE => 32,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH => 2147483647,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE => 1000000,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT => 512,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => 108,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH => 4096,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT => 4096,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH => 4096,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE => 1000000,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH => 256,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE => 1048576,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR => 2048,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT => 2,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => {
            COMPUTE_CAPABILITY_MAJOR
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => {
            COMPUTE_CAPABILITY_MINOR
        }
        _ => return Err(CUerror::INVALID_VALUE),
    };

    unsafe { *pi = result };
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_count(count: &mut ::core::ffi::c_int) -> CUresult {
    let devices = super::driver::global_state()?;
    *count = devices.devices.len() as ::core::ffi::c_int;
    if let Ok(override_count) = std::env::var("HETGPU_PACC_VISIBLE_DEVICES") {
        if let Ok(parsed) = override_count.parse::<i32>() {
            *count = parsed.clamp(1, 4);
        }
    }
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_name(
    name: *mut ::core::ffi::c_char,
    len: ::core::ffi::c_int,
    dev: i32,
) -> CUresult {
    if name.is_null() || len <= 0 {
        return Err(CUerror::INVALID_VALUE);
    }

    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    let device_name = b"Tenstorrent Device\0";
    let copy_len = std::cmp::min(device_name.len(), len as usize - 1);

    unsafe {
        std::ptr::copy_nonoverlapping(device_name.as_ptr(), name as *mut u8, copy_len);
        *name.add(copy_len) = 0;
    }

    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_uuid(uuid: *mut [u8; 16], dev: i32) -> CUresult {
    if uuid.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    // Generate a deterministic UUID based on device ID
    let mut device_uuid = [0u8; 16];
    device_uuid[0..4].copy_from_slice(&(dev as u32).to_le_bytes());
    device_uuid[4..8].copy_from_slice(b"TTTT"); // Tenstorrent marker
    device_uuid[8..16].copy_from_slice(b"ZLUDA001"); // ZLUDA version

    unsafe { *uuid = device_uuid };
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_uuid_v2(uuid: *mut [u8; 16], dev: i32) -> CUresult {
    get_uuid(uuid, dev)
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_luid(
    luid: *mut [u8; 8],
    device_node_mask: *mut ::core::ffi::c_uint,
    dev: i32,
) -> CUresult {
    if luid.is_null() || device_node_mask.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    // Generate LUID based on device ID
    let mut device_luid = [0u8; 8];
    device_luid[0..4].copy_from_slice(&(dev as u32).to_le_bytes());
    device_luid[4..8].copy_from_slice(b"TTTT");

    unsafe {
        *luid = device_luid;
        *device_node_mask = 1u32 << (dev as u32 % 32);
    }

    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn total_mem_v2(bytes: *mut usize, dev: i32) -> CUresult {
    if bytes.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    // Return 8GB as placeholder for Tenstorrent device memory
    unsafe { *bytes = 8 * 1024 * 1024 * 1024 };
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_properties(prop: &mut CUdevprop, dev: i32) -> CUresult {
    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    prop.maxThreadsPerBlock = 1024;
    prop.maxThreadsDim = [1024, 1024, 64];
    prop.maxGridSize = [65535, 65535, 65535];
    prop.sharedMemPerBlock = 65536;
    prop.totalConstantMemory = 65536;
    prop.SIMDWidth = 32;
    prop.memPitch = 2147483647;
    prop.regsPerBlock = 65536;
    prop.clockRate = 1000;
    prop.textureAlign = 512;

    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn primary_context_retain(pctx: *mut CUcontext, dev: i32) -> CUresult {
    if pctx.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let (primary_ctx, handle) = super::context::get_primary_tt(dev)?;
    primary_ctx.increment_ref_count();

    unsafe { *pctx = handle };
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn primary_context_release(dev: i32) -> CUresult {
    let (primary_ctx, _) = super::context::get_primary_tt(dev)?;
    let ref_count = primary_ctx.decrement_ref_count();

    if ref_count == 0 {
        // Clean up context when reference count reaches zero
        primary_ctx.destroy()?;
    }

    Ok(())
}

// TMatmul device function implementations
#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn compute_capability(major: &mut i32, minor: &mut i32, _dev: i32) -> CUresult {
    *major = COMPUTE_CAPABILITY_MAJOR;
    *minor = COMPUTE_CAPABILITY_MINOR;
    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get(device: *mut i32, ordinal: i32) -> CUresult {
    let devices = super::driver::global_state()?;
    if ordinal < 0 || ordinal >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }
    unsafe { *device = ordinal };
    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_attribute(
    pi: *mut ::core::ffi::c_int,
    attrib: CUdevice_attribute,
    dev: i32,
) -> CUresult {
    if pi.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    // Return GPU-like values compatible with Triton and other libraries
    let result = match attrib {
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK => 1024,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X => 1024,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y => 1024,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z => 64,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X => 2147483647,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y => 65535,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z => 65535,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => 49152, // 48KB - typical GPU value
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE => 32,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH => 2147483647,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE => 1410000, // 1.41 GHz
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT => 512,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => 80, // Typical for modern GPUs
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH => 131072,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH => 131072,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH => 16384,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT => 16384,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH => 16384,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE => 5001000, // 5 GHz
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH => 384, // bits
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE => 6291456,     // 6MB
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR => 2048,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT => 2,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => {
            COMPUTE_CAPABILITY_MAJOR
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => {
            COMPUTE_CAPABILITY_MINOR
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ECC_ENABLED => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID => 0x02 + dev,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TCC_DRIVER => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR => 102400, // 100KB
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID => 0,
        _ => return Err(CUerror::INVALID_VALUE),
    };

    unsafe { *pi = result };
    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_count(count: &mut ::core::ffi::c_int) -> CUresult {
    let devices = super::driver::global_state()?;
    *count = devices.devices.len() as ::core::ffi::c_int;
    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_name(
    name: *mut ::core::ffi::c_char,
    len: ::core::ffi::c_int,
    dev: i32,
) -> CUresult {
    if name.is_null() || len <= 0 {
        return Err(CUerror::INVALID_VALUE);
    }

    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    let device_name = b"TMatmul Virtual GPU\0";
    let copy_len = std::cmp::min(device_name.len(), len as usize - 1);

    unsafe {
        std::ptr::copy_nonoverlapping(device_name.as_ptr(), name as *mut u8, copy_len);
        *name.add(copy_len) = 0;
    }

    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_uuid(uuid: *mut [u8; 16], dev: i32) -> CUresult {
    if uuid.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    let mut device_uuid = [0u8; 16];
    device_uuid[0..4].copy_from_slice(&(dev as u32).to_le_bytes());
    device_uuid[4..8].copy_from_slice(b"TMAT"); // TMatmul marker
    device_uuid[8..16].copy_from_slice(b"ZLUDA001");

    unsafe { *uuid = device_uuid };
    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_uuid_v2(uuid: *mut [u8; 16], dev: i32) -> CUresult {
    get_uuid(uuid, dev)
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_luid(
    luid: *mut [u8; 8],
    device_node_mask: *mut ::core::ffi::c_uint,
    dev: i32,
) -> CUresult {
    if luid.is_null() || device_node_mask.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    let mut device_luid = [0u8; 8];
    device_luid[0..4].copy_from_slice(&(dev as u32).to_le_bytes());
    device_luid[4..8].copy_from_slice(b"TMAT");

    unsafe {
        *luid = device_luid;
        *device_node_mask = 1u32 << (dev as u32 % 32);
    }

    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn total_mem_v2(bytes: *mut usize, dev: i32) -> CUresult {
    if bytes.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    // Return 16GB as virtual device memory
    unsafe { *bytes = 16 * 1024 * 1024 * 1024 };
    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_properties(prop: &mut CUdevprop, dev: i32) -> CUresult {
    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    prop.maxThreadsPerBlock = 1024;
    prop.maxThreadsDim = [1024, 1024, 64];
    prop.maxGridSize = [2147483647, 65535, 65535];
    prop.sharedMemPerBlock = 49152; // 48KB
    prop.totalConstantMemory = 65536;
    prop.SIMDWidth = 32;
    prop.memPitch = 2147483647;
    prop.regsPerBlock = 65536;
    prop.clockRate = 1410000;
    prop.textureAlign = 512;

    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn primary_context_retain(pctx: *mut CUcontext, dev: i32) -> CUresult {
    if pctx.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let (ctx, raw_ctx) = super::context::get_primary_tmatmul(dev)?;
    {
        let mut mutable_ctx = ctx.mutable.lock().map_err(|_| CUerror::UNKNOWN)?;
        mutable_ctx.ref_count += 1;
    }

    unsafe { *pctx = raw_ctx };
    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn primary_context_release(dev: i32) -> CUresult {
    let (ctx, _) = super::context::get_primary_tmatmul(dev)?;
    {
        let mut mutable_ctx = ctx.mutable.lock().map_err(|_| CUerror::UNKNOWN)?;
        if mutable_ctx.ref_count == 0 {
            return Err(CUerror::INVALID_CONTEXT);
        }
        mutable_ctx.ref_count -= 1;
    }

    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn primary_context_get_state(dev: i32, flags: &mut u32, active: &mut i32) -> CUresult {
    crate::r#impl::hetgpu_debug!("[hetGPU DEBUG] cuDevicePrimaryCtxGetState(dev={})", dev);

    // For virtual device, report that context exists and is active
    if dev < 0 {
        crate::r#impl::hetgpu_debug!(
            "[hetGPU DEBUG] cuDevicePrimaryCtxGetState: INVALID_DEVICE (dev < 0)"
        );
        return Err(CUerror::INVALID_DEVICE);
    }

    let devices = match super::driver::global_state() {
        Ok(d) => d,
        Err(e) => {
            crate::r#impl::hetgpu_debug!(
                "[hetGPU DEBUG] cuDevicePrimaryCtxGetState: failed to get global_state: {:?}",
                e
            );
            return Err(e);
        }
    };

    if dev >= devices.devices.len() as i32 {
        crate::r#impl::hetgpu_debug!(
            "[hetGPU DEBUG] cuDevicePrimaryCtxGetState: INVALID_DEVICE (dev >= device_count)"
        );
        return Err(CUerror::INVALID_DEVICE);
    }

    // Report default flags (0) and active (1)
    *flags = 0;
    *active = 1;

    crate::r#impl::hetgpu_debug!(
        "[hetGPU DEBUG] cuDevicePrimaryCtxGetState: SUCCESS (flags={}, active={})",
        *flags,
        *active
    );
    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn primary_context_get_state(dev: i32, flags: &mut u32, active: &mut i32) -> CUresult {
    // For TMatmul virtual device, report that context exists and is active
    if dev < 0 {
        return Err(CUerror::INVALID_DEVICE);
    }

    let devices = super::driver::global_state()?;
    if dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }

    // Report default flags (0) and active (1)
    *flags = 0;
    *active = 1;

    Ok(())
}

// NVIDIA backend device function implementations - passthrough to real libcuda.so
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn compute_capability(major: &mut i32, minor: &mut i32, dev: i32) -> CUresult {
    let result = nvidia_runtime_sys::cuDeviceComputeCapability(major, minor, dev);
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
pub(crate) fn get(device: *mut i32, ordinal: i32) -> CUresult {
    let result = nvidia_runtime_sys::cuDeviceGet(device, ordinal);
    if result != 0 {
        return Err(CUerror::INVALID_DEVICE);
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
pub(crate) fn get_attribute(pi: &mut i32, attrib: CUdevice_attribute, dev: i32) -> CUresult {
    let result = nvidia_runtime_sys::cuDeviceGetAttribute(pi, attrib, dev);
    if result != 0 {
        return Err(CUerror::INVALID_VALUE);
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
pub(crate) fn get_count(count: &mut ::core::ffi::c_int) -> CUresult {
    eprintln!("[hetGPU device] get_count called");
    let result = nvidia_runtime_sys::cuDeviceGetCount(count);
    eprintln!(
        "[hetGPU device] nvidia_runtime_sys::cuDeviceGetCount returned {}, count={}",
        result, *count
    );
    if result != 0 {
        eprintln!("[hetGPU device] get_count returning error");
        return Err(CUerror::UNKNOWN);
    }
    eprintln!("[hetGPU device] get_count returning Ok, count={}", *count);
    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn get_name(
    name: *mut ::core::ffi::c_char,
    len: ::core::ffi::c_int,
    dev: i32,
) -> CUresult {
    let result = nvidia_runtime_sys::cuDeviceGetName(name, len, dev);
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
pub(crate) fn get_uuid(uuid: *mut cuda_types::cuda::CUuuid, dev: i32) -> CUresult {
    let result = nvidia_runtime_sys::cuDeviceGetUuid(uuid, dev);
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
pub(crate) fn get_uuid_v2(uuid: *mut cuda_types::cuda::CUuuid, dev: i32) -> CUresult {
    get_uuid(uuid, dev)
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn get_luid(
    luid: *mut ::core::ffi::c_char,
    device_node_mask: &mut ::core::ffi::c_uint,
    dev: i32,
) -> CUresult {
    let result = nvidia_runtime_sys::cuDeviceGetLuid(luid, device_node_mask, dev);
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
pub(crate) fn total_mem_v2(bytes: *mut usize, dev: i32) -> CUresult {
    let result = nvidia_runtime_sys::cuDeviceTotalMem_v2(bytes, dev);
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
pub(crate) fn get_properties(prop: &mut CUdevprop, dev: i32) -> CUresult {
    // CUDA doesn't have a direct cuDeviceGetProperties, so we build from attributes
    let mut pi: i32 = 0;

    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        dev,
    );
    prop.maxThreadsPerBlock = pi;

    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
        dev,
    );
    prop.maxThreadsDim[0] = pi;
    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
        dev,
    );
    prop.maxThreadsDim[1] = pi;
    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
        dev,
    );
    prop.maxThreadsDim[2] = pi;

    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        dev,
    );
    prop.maxGridSize[0] = pi;
    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
        dev,
    );
    prop.maxGridSize[1] = pi;
    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        dev,
    );
    prop.maxGridSize[2] = pi;

    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        dev,
    );
    prop.sharedMemPerBlock = pi;

    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
        dev,
    );
    prop.totalConstantMemory = pi;

    prop.SIMDWidth = 32;

    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH,
        dev,
    );
    prop.memPitch = pi;

    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        dev,
    );
    prop.regsPerBlock = pi;

    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
        dev,
    );
    prop.clockRate = pi;

    nvidia_runtime_sys::cuDeviceGetAttribute(
        &mut pi,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
        dev,
    );
    prop.textureAlign = pi;

    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn primary_context_retain(pctx: &mut CUcontext, dev: i32) -> CUresult {
    let result = nvidia_runtime_sys::cuDevicePrimaryCtxRetain(pctx, dev);
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
pub(crate) fn primary_context_release(dev: i32) -> CUresult {
    let result = nvidia_runtime_sys::cuDevicePrimaryCtxRelease_v2(dev);
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
pub(crate) fn primary_context_get_state(dev: i32, flags: &mut u32, active: &mut i32) -> CUresult {
    let result = nvidia_runtime_sys::cuDevicePrimaryCtxGetState(dev, flags, active);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

// ============================================================================
// PACC device function implementations (SiFive Intelligence XM / RISC-V IME)
// ============================================================================

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn compute_capability(major: &mut i32, minor: &mut i32, _dev: i32) -> CUresult {
    *major = COMPUTE_CAPABILITY_MAJOR;
    *minor = COMPUTE_CAPABILITY_MINOR;
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get(device: *mut i32, ordinal: i32) -> CUresult {
    let devices = super::driver::global_state()?;
    if ordinal < 0 || ordinal >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }
    unsafe { *device = ordinal };
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_count(count: &mut ::core::ffi::c_int) -> CUresult {
    let devices = super::driver::global_state()?;
    *count = devices.devices.len() as ::core::ffi::c_int;
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_name(
    name: *mut ::core::ffi::c_char,
    len: ::core::ffi::c_int,
    dev: i32,
) -> CUresult {
    if name.is_null() || len <= 0 {
        return Err(CUerror::INVALID_VALUE);
    }
    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }
    let device_name = b"SiFive Intelligence XM PACC\0";
    let copy_len = std::cmp::min(device_name.len(), len as usize - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(device_name.as_ptr(), name as *mut u8, copy_len);
        *name.add(copy_len) = 0;
    }
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_uuid(uuid: *mut [u8; 16], dev: i32) -> CUresult {
    if uuid.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }
    let mut device_uuid = [0u8; 16];
    device_uuid[0..4].copy_from_slice(&(dev as u32).to_le_bytes());
    device_uuid[4..8].copy_from_slice(b"PACC");
    device_uuid[8..16].copy_from_slice(b"ZLUDA001");
    unsafe { *uuid = device_uuid };
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_uuid_v2(uuid: *mut [u8; 16], dev: i32) -> CUresult {
    get_uuid(uuid, dev)
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_luid(
    luid: *mut [u8; 8],
    device_node_mask: *mut ::core::ffi::c_uint,
    dev: i32,
) -> CUresult {
    if luid.is_null() || device_node_mask.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }
    let mut device_luid = [0u8; 8];
    device_luid[0..4].copy_from_slice(&(dev as u32).to_le_bytes());
    device_luid[4..8].copy_from_slice(b"PACC");
    unsafe {
        *luid = device_luid;
        *device_node_mask = 1u32 << (dev as u32 % 32);
    }
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn total_mem_v2(bytes: *mut usize, dev: i32) -> CUresult {
    if bytes.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }
    unsafe { *bytes = 4 * 1024 * 1024 * 1024 };
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_properties(prop: &mut CUdevprop, dev: i32) -> CUresult {
    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }
    prop.maxThreadsPerBlock = 1024;
    prop.maxThreadsDim = [1024, 1024, 64];
    prop.maxGridSize = [65535, 65535, 65535];
    prop.sharedMemPerBlock = 65536;
    prop.totalConstantMemory = 65536;
    prop.SIMDWidth = 32;
    prop.memPitch = 2147483647;
    prop.regsPerBlock = 65536;
    prop.clockRate = 1000;
    prop.textureAlign = 512;
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn primary_context_retain(pctx: *mut CUcontext, dev: i32) -> CUresult {
    if pctx.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let (primary_ctx, handle) = super::context::get_primary_pacc(dev)?;
    primary_ctx.increment_ref_count();
    unsafe { *pctx = handle };
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn primary_context_release(dev: i32) -> CUresult {
    let (primary_ctx, _) = super::context::get_primary_pacc(dev)?;
    let ref_count = primary_ctx.decrement_ref_count();
    if ref_count == 0 {
        primary_ctx.destroy()?;
    }
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_attribute(
    pi: *mut ::core::ffi::c_int,
    attrib: CUdevice_attribute,
    dev: i32,
) -> CUresult {
    if pi.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let devices = super::driver::global_state()?;
    if dev < 0 || dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }
    let result = match attrib {
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK => 1024,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X => 1024,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y => 1024,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z => 64,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X => 65535,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y => 65535,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z => 65535,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE => 32,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH => 2147483647,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE => 1000000,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT => 512,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT => 65536,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH => 4096,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT => 4096,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH => 4096,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE => 1000000,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH => 128,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE => 131072,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR => 2048,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING => 1,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => {
            COMPUTE_CAPABILITY_MAJOR
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => {
            COMPUTE_CAPABILITY_MINOR
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID => 0x02 + dev,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID => 0,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID => 0,
        _ => return Err(CUerror::INVALID_VALUE),
    };
    unsafe { *pi = result };
    Ok(())
}

// ─── PACC primary_context_get_state ──────────────────────────────────────────
#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn primary_context_get_state(dev: i32, flags: &mut u32, active: &mut i32) -> CUresult {
    if dev < 0 {
        return Err(CUerror::INVALID_DEVICE);
    }
    let devices = super::driver::global_state()?;
    if dev >= devices.devices.len() as i32 {
        return Err(CUerror::INVALID_DEVICE);
    }
    *flags = 0;
    *active = 1;
    Ok(())
}
