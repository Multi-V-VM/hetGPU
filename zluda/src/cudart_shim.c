#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(HETGPU_DEBUG_LOGS)
#define HETGPU_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
#define HETGPU_LOG(...) ((void)0)
#endif

// Minimal shim for missing CUDA Runtime API symbols expected by
// PyTorch CUDA libraries when running with hetGPU. We export only
// the symbols that are missing from the packaged libcudart, with
// safe no-op behavior.

// Return type matches cudaError_t ABI (int). 0 means success.
typedef int cudaError_t;
typedef int CUresult;
typedef void* CUcontext;
typedef int CUdevice;

// In CUDA v2 API variants, CUdeviceptr is an opaque handle that wraps a host/device pointer.
// For our virtual backend we model it as a plain pointer-sized value.
typedef void* CUdeviceptr;

// Forward declarations for CUDA Driver API functions (implemented in Rust)
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;

extern CUresult cuInit(unsigned int flags);
extern CUresult cuDeviceGet(CUdevice* device, int ordinal);
extern CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
extern CUresult cuCtxSetCurrent(CUcontext ctx);
extern CUresult cuCtxGetCurrent(CUcontext* pctx);
extern CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);
extern CUresult cuMemFree_v2(CUdeviceptr dptr);
extern CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
extern CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
extern CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N);
extern CUresult cuModuleLoadData(CUmodule* module, const void* image);
extern CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
extern CUresult cuLaunchKernel(CUfunction f,
                               unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                               unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                               unsigned int sharedMemBytes,
                               CUstream hStream,
                               void** kernelParams,
                               void** extra);

// Opaque graph node type (avoid including CUDA headers).
typedef void* cudaGraphNode_t;
typedef void* cudaGraph_t;
typedef int cudaGraphNodeType;
typedef void* cudaStream_t;
typedef int cudaStreamCaptureStatus;
typedef int cudaStreamCaptureMode;
typedef void* cudaEvent_t;
typedef void* cudaGraphExec_t;
typedef void* cudaGraphNode_t; // already defined
typedef void* cudaDeviceProp_t; // opaque placeholder for device properties struct
typedef void* cudaMemPool_t;
typedef void* cudaUserObject_t;
typedef void* cudaFunction_t;
typedef struct { unsigned int x, y, z; } dim3;
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* userData);
typedef void (*cudaHostFn_t)(void* userData);
typedef int cudaMemcpyKind; // use int placeholder

typedef struct {
    void* payload;
    cudaHostFn_t destroy;
    unsigned int refcount;
    unsigned int flags;
} HetGPUUserObject;

// Provide a stub for cudaGraphNodeGetDependencies. We simply report
// zero dependencies and return success. This satisfies symbol lookup
// and avoids unnecessary runtime failures in code paths that only
// probe for availability.
cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node,
                                         cudaGraphNode_t* pDependencies,
                                         size_t* pNumDependencies) {
    (void)node;
    (void)pDependencies;
    if (pNumDependencies) {
        *pNumDependencies = 0;
    }
    return 0; // cudaSuccess
}

// Return dummy node type and success
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) {
    (void)node;
    if (pType) {
        *pType = 0; // unspecified type
    }
    return 0;
}

// Create an empty node stub: return success and a null node
cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode,
                                  cudaGraph_t graph,
                                  const cudaGraphNode_t* pDependencies,
                                  size_t numDependencies) {
    (void)graph;
    (void)pDependencies;
    (void)numDependencies;
    if (pGraphNode) {
        *pGraphNode = (cudaGraphNode_t)0;
    }
    return 0;
}

// Stream capture info APIs (stubs)
cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream,
                                     cudaStreamCaptureStatus* pStatus,
                                     unsigned long long* pId) {
    (void)stream;
    if (pStatus) *pStatus = 0; // cudaStreamCaptureStatusNone
    if (pId) *pId = 0ULL;
    return 0;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream,
                                        cudaStreamCaptureStatus* pStatus,
                                        unsigned long long* pId,
                                        cudaGraph_t* phGraph,
                                        const cudaGraphNode_t** ppDependencies,
                                        size_t* pNumDependencies) {
    (void)stream;
    if (pStatus) *pStatus = 0;
    if (pId) *pId = 0ULL;
    if (phGraph) *phGraph = (cudaGraph_t)0;
    if (ppDependencies) *ppDependencies = NULL;
    if (pNumDependencies) *pNumDependencies = 0;
    return 0;
}

cudaError_t cudaStreamGetCaptureInfo_v3(cudaStream_t stream,
                                        cudaStreamCaptureStatus* pStatus,
                                        unsigned long long* pId,
                                        cudaGraph_t* phGraph,
                                        const cudaGraphNode_t** ppDependencies,
                                        size_t* pNumDependencies,
                                        unsigned long long flags) {
    (void)flags;
    return cudaStreamGetCaptureInfo_v2(stream, pStatus, pId, phGraph, ppDependencies, pNumDependencies);
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream,
                                  cudaStreamCaptureStatus* pStatus) {
    (void)stream;
    if (pStatus) *pStatus = 0; // None
    return 0;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
                                   cudaStreamCaptureMode mode) {
    (void)stream; (void)mode;
    return 0;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream,
                                 cudaGraph_t* pGraph) {
    (void)stream;
    if (pGraph) *pGraph = (cudaGraph_t)0;
    return 0;
}

// Basic stream create/destroy
cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    if (pStream) *pStream = (cudaStream_t)0; return 0;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
    (void)flags; if (pStream) *pStream = (cudaStream_t)0; return 0;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) { (void)stream; return 0; }

// Legacy callback API
cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void* userData,
                                  unsigned int flags) {
    (void)stream; (void)flags;
    if (callback) {
        // Invoke synchronously with success to satisfy callers
        callback(stream, 0, userData);
    }
    return 0;
}

// Launch host function on stream (stub: invoke synchronously)
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
    (void)stream;
    if (fn) fn(userData);
    return 0;
}

// Update capture dependencies (stub)
cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream,
                                               cudaGraphNode_t* dependencies,
                                               size_t numDependencies,
                                               unsigned int updateFlags) {
    (void)stream; (void)dependencies; (void)numDependencies; (void)updateFlags;
    return 0;
}

cudaError_t cudaStreamUpdateCaptureDependencies_v2(cudaStream_t stream,
                                                  cudaGraphNode_t* dependencies,
                                                  size_t numDependencies,
                                                  unsigned long long updateFlags) {
    return cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, (unsigned int)updateFlags);
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream,
                                         unsigned int flags,
                                         int priority) {
    (void)flags; (void)priority;
    if (pStream) *pStream = (cudaStream_t)0;
    return 0;
}

// Event API stubs
cudaError_t cudaEventCreate(cudaEvent_t* event) {
    if (event) *event = (cudaEvent_t)0;
    return 0;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    (void)flags;
    if (event) *event = (cudaEvent_t)0;
    return 0;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    (void)event; (void)stream; return 0;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
    (void)event; (void)stream; (void)flags; return 0;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    (void)event; return 0;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    (void)event; return 0;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    (void)event; return 0;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    (void)start; (void)end; if (ms) *ms = 0.0f; return 0;
}

// Error query APIs
const char* cudaGetErrorString(cudaError_t error) {
    if (error == 0) return "cudaSuccess";
    return "cudaErrorUnknown";
}

const char* cudaGetErrorName(cudaError_t error) {
    if (error == 0) return "cudaSuccess";
    return "cudaErrorUnknown";
}

// Device/runtime info APIs
// Note: cudaGetDeviceCount and cudaDriverGetVersion are already implemented in Rust (lib.rs)
// so we don't define them here to avoid duplicate symbol errors

// Full cudaDeviceProp struct matching CUDA 11.x/12.x layout
// This must match PyTorch's expectations exactly
typedef struct {
    char   name[256];                // 0-255
    char   uuid[16];                 // 256-271 (cudaUUID_t)
    char   luid[8];                  // 272-279
    unsigned int luidDeviceNodeMask; // 280-283
    int    _padding1;                // 284-287 (alignment)
    size_t totalGlobalMem;           // 288-295
    size_t sharedMemPerBlock;        // 296-303
    int    regsPerBlock;             // 304-307
    int    warpSize;                 // 308-311
    size_t memPitch;                 // 312-319
    int    maxThreadsPerBlock;       // 320-323
    int    maxThreadsDim[3];         // 324-335
    int    maxGridSize[3];           // 336-347
    int    clockRate;                // 348-351
    size_t totalConstMem;            // 352-359
    int    major;                    // 360-363 ← This is the key field!
    int    minor;                    // 364-367 ← This is the key field!
    int    textureAlignment;         // 368-371
    int    texturePitchAlignment;    // 372-375
    int    deviceOverlap;            // 376-379
    int    multiProcessorCount;      // 380-383
    int    kernelExecTimeoutEnabled; // 384-387
    int    integrated;               // 388-391
    int    canMapHostMemory;         // 392-395
    int    computeMode;              // 396-399
    int    maxTexture1D;             // 400-403
    int    maxTexture1DMipmap;       // 404-407
    int    maxTexture1DLinear;       // 408-411
    int    maxTexture2D[2];          // 412-419
    int    maxTexture2DMipmap[2];    // 420-427
    int    maxTexture2DLinear[3];    // 428-439
    int    maxTexture2DGather[2];    // 440-447
    int    maxTexture3D[3];          // 448-459
    int    maxTexture3DAlt[3];       // 460-471
    int    maxTextureCubemap;        // 472-475
    int    maxTexture1DLayered[2];   // 476-483
    int    maxTexture2DLayered[3];   // 484-495
    int    maxTextureCubemapLayered[2]; // 496-503
    int    maxSurface1D;             // 504-507
    int    maxSurface2D[2];          // 508-515
    int    maxSurface3D[3];          // 516-527
    int    maxSurface1DLayered[2];   // 528-535
    int    maxSurface2DLayered[3];   // 536-547
    int    maxSurfaceCubemap;        // 548-551
    int    maxSurfaceCubemapLayered[2]; // 552-559
    size_t surfaceAlignment;         // 560-567
    int    concurrentKernels;        // 568-571
    int    ECCEnabled;               // 572-575
    int    pciBusID;                 // 576-579
    int    pciDeviceID;              // 580-583
    int    pciDomainID;              // 584-587
    int    tccDriver;                // 588-591
    int    asyncEngineCount;         // 592-595
    int    unifiedAddressing;        // 596-599
    int    memoryClockRate;          // 600-603
    int    memoryBusWidth;           // 604-607
    int    l2CacheSize;              // 608-611
    int    persistingL2CacheMaxSize; // 612-615
    int    maxThreadsPerMultiProcessor; // 616-619
    int    streamPrioritiesSupported;   // 620-623
    int    globalL1CacheSupported;      // 624-627
    int    localL1CacheSupported;       // 628-631
    size_t sharedMemPerMultiprocessor;  // 632-639
    int    regsPerMultiprocessor;       // 640-643
    int    managedMemory;               // 644-647
    int    isMultiGpuBoard;             // 648-651
    int    multiGpuBoardGroupID;        // 652-655
    int    hostNativeAtomicSupported;   // 656-659
    int    singleToDoublePrecisionPerfRatio; // 660-663
    int    pageableMemoryAccess;        // 664-667
    int    concurrentManagedAccess;     // 668-671
    int    computePreemptionSupported;  // 672-675
    int    canUseHostPointerForRegisteredMem; // 676-679
    int    cooperativeLaunch;           // 680-683
    int    cooperativeMultiDeviceLaunch; // 684-687
    size_t sharedMemPerBlockOptin;      // 688-695
    int    pageableMemoryAccessUsesHostPageTables; // 696-699
    int    directManagedMemAccessFromHost; // 700-703
} cudaDeviceProp_full;

cudaError_t cudaGetDeviceProperties(cudaDeviceProp_t prop, int device) {
    if (!prop) return 1; // cudaErrorInvalidValue

    // Fill full struct matching CUDA 11.x/12.x layout
    cudaDeviceProp_full p;
    memset(&p, 0, sizeof(p));

    // Device name
    const char* name = "Virtual GPU (hetGPU sm_80)";
    strncpy(p.name, name, sizeof(p.name) - 1);

    // Memory properties
    p.totalGlobalMem = 4ULL * 1024 * 1024 * 1024;  // 4GB
    p.sharedMemPerBlock = 48 * 1024;               // 48KB
    p.sharedMemPerMultiprocessor = 64 * 1024;      // 64KB
    p.totalConstMem = 64 * 1024;                   // 64KB
    p.memPitch = 2147483647;
    p.surfaceAlignment = 512;

    // Compute resources
    p.regsPerBlock = 65536;
    p.regsPerMultiprocessor = 65536;
    p.warpSize = 32;
    p.maxThreadsPerBlock = 1024;
    p.maxThreadsPerMultiProcessor = 1536;
    p.multiProcessorCount = 80;  // Like A100

    // Thread/block dimensions
    p.maxThreadsDim[0] = 1024;
    p.maxThreadsDim[1] = 1024;
    p.maxThreadsDim[2] = 64;
    p.maxGridSize[0] = 2147483647;
    p.maxGridSize[1] = 65535;
    p.maxGridSize[2] = 65535;

    // Clock rates
    p.clockRate = 1410000;        // 1.41 GHz
    p.memoryClockRate = 1215000;  // 1.215 GHz
    p.memoryBusWidth = 5120;      // 5120-bit (like A100)

    // Compute capability - THE KEY FIELDS!
    p.major = 8;
    p.minor = 0;

    // Cache properties
    p.l2CacheSize = 40 * 1024 * 1024;  // 40MB
    p.persistingL2CacheMaxSize = 40 * 1024 * 1024;

    // Capabilities
    p.concurrentKernels = 1;
    p.ECCEnabled = 0;
    p.asyncEngineCount = 2;
    p.unifiedAddressing = 1;
    p.managedMemory = 1;
    p.computePreemptionSupported = 1;
    p.cooperativeLaunch = 1;
    p.cooperativeMultiDeviceLaunch = 0;
    p.pageableMemoryAccess = 1;
    p.concurrentManagedAccess = 1;
    p.canUseHostPointerForRegisteredMem = 1;
    p.directManagedMemAccessFromHost = 1;
    p.globalL1CacheSupported = 1;
    p.localL1CacheSupported = 1;

    // Texture limits (conservative defaults)
    p.maxTexture1D = 131072;
    p.maxTexture2D[0] = 131072;
    p.maxTexture2D[1] = 65536;
    p.maxTexture3D[0] = 16384;
    p.maxTexture3D[1] = 16384;
    p.maxTexture3D[2] = 16384;

    // PCI info (fake but valid)
    p.pciBusID = 0;
    p.pciDeviceID = 0;
    p.pciDomainID = 0;

    // Copy full struct to caller's buffer
    memcpy(prop, &p, sizeof(p));

    fprintf(stderr, "[cudart_shim] cudaGetDeviceProperties: name='%s' cc=%d.%d (offset major=%zu, minor=%zu)\n",
            p.name, p.major, p.minor,
            offsetof(cudaDeviceProp_full, major),
            offsetof(cudaDeviceProp_full, minor));

    (void)device;
    return 0;
}

// v2 API variant - just calls the base implementation
cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp_t prop, int device) {
    return cudaGetDeviceProperties(prop, device);
}

// Global to track current device
static int current_device = 0;

cudaError_t cudaSetDevice(int device) {
    // For virtual device support, be permissive
    if (device < 0) {
        return 1; // cudaErrorInvalidDevice
    }

    // Get the CUDA device handle
    CUdevice cu_device;
    CUresult result = cuDeviceGet(&cu_device, device);
    if (result != 0) {
        // For virtual device, still set current_device and succeed
        HETGPU_LOG("[cudart_shim] cudaSetDevice(%d): cuDeviceGet failed (%d), continuing with virtual device\n", device, result);
        current_device = device;
        return 0; // Success for virtual device
    }

    // Retain the primary context for this device
    CUcontext ctx;
    result = cuDevicePrimaryCtxRetain(&ctx, cu_device);
    if (result != 0) {
        // For virtual device, still set current_device and succeed
        HETGPU_LOG("[cudart_shim] cudaSetDevice(%d): cuDevicePrimaryCtxRetain failed (%d), continuing with virtual device\n", device, result);
        current_device = device;
        return 0; // Success for virtual device
    }

    // Set it as the current context
    result = cuCtxSetCurrent(ctx);
    if (result != 0) {
        // For virtual device, still set current_device and succeed
        HETGPU_LOG("[cudart_shim] cudaSetDevice(%d): cuCtxSetCurrent failed (%d), continuing with virtual device\n", device, result);
        current_device = device;
        return 0; // Success for virtual device
    }

    current_device = device;
    return 0;
}

cudaError_t cudaGetDevice(int* device) {
    HETGPU_LOG("[cudart_shim] cudaGetDevice called, returning device %d\n", current_device);
    if (device) *device = current_device;
    return 0;
}

cudaError_t cudaRuntimeGetVersion(int* version) {
    if (version) *version = 12080; return 0;
}

cudaError_t cudaDeviceSynchronize(void) { return 0; }
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { (void)stream; return 0; }
cudaError_t cudaStreamQuery(cudaStream_t stream) { (void)stream; return 0; }
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    (void)stream; (void)event; (void)flags; return 0;
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority) {
    (void)stream; if (priority) *priority = 0; return 0;
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
    (void)device; (void)peerDevice; if (canAccessPeer) *canAccessPeer = 0; return 0;
}
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    (void)peerDevice; (void)flags; return 0;
}

cudaError_t cudaDeviceSetLimit(int limit, size_t value) {
    (void)limit;
    (void)value;
    return 0;
}

// Device attribute query
cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device) {
    if (!value) return 1; // cudaErrorInvalidValue

    // Common CUDA device attributes (from cuda_runtime_api.h)
    enum cudaDeviceAttr {
        cudaDevAttrMaxThreadsPerBlock = 1,
        cudaDevAttrMaxBlockDimX = 2,
        cudaDevAttrMaxBlockDimY = 3,
        cudaDevAttrMaxBlockDimZ = 4,
        cudaDevAttrMaxGridDimX = 5,
        cudaDevAttrMaxGridDimY = 6,
        cudaDevAttrMaxGridDimZ = 7,
        cudaDevAttrMaxSharedMemoryPerBlock = 8,
        cudaDevAttrTotalConstantMemory = 9,
        cudaDevAttrWarpSize = 10,
        cudaDevAttrMaxPitch = 11,
        cudaDevAttrMaxRegistersPerBlock = 12,
        cudaDevAttrClockRate = 13,
        cudaDevAttrTextureAlignment = 14,
        cudaDevAttrMultiProcessorCount = 16,
        cudaDevAttrComputeCapabilityMajor = 75,
        cudaDevAttrComputeCapabilityMinor = 76,
    };

    // Return realistic values for GPU-like device
    switch (attr) {
        case cudaDevAttrMaxThreadsPerBlock: *value = 1024; break;
        case cudaDevAttrMaxBlockDimX: *value = 1024; break;
        case cudaDevAttrMaxBlockDimY: *value = 1024; break;
        case cudaDevAttrMaxBlockDimZ: *value = 64; break;
        case cudaDevAttrMaxGridDimX: *value = 2147483647; break;
        case cudaDevAttrMaxGridDimY: *value = 65535; break;
        case cudaDevAttrMaxGridDimZ: *value = 65535; break;
        case cudaDevAttrMaxSharedMemoryPerBlock: *value = 49152; break;
        case cudaDevAttrTotalConstantMemory: *value = 65536; break;
        case cudaDevAttrWarpSize: *value = 32; break;
        case cudaDevAttrMaxPitch: *value = 2147483647; break;
        case cudaDevAttrMaxRegistersPerBlock: *value = 65536; break;
        case cudaDevAttrClockRate: *value = 1410000; break;  // kHz
        case cudaDevAttrTextureAlignment: *value = 512; break;
        case cudaDevAttrMultiProcessorCount: *value = 80; break;  // SM count (A100-like)
        case cudaDevAttrComputeCapabilityMajor: *value = 8; break;
        case cudaDevAttrComputeCapabilityMinor: *value = 0; break;
        default:
            // Generic non-zero default to avoid divide-by-zero
            *value = 1;
            break;
    }

    (void)device;
    return 0;
}

// Host memory APIs
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
    (void)flags;
    if (!pHost) return 1; // cudaErrorInvalidValue
    // Allocate page-aligned host memory; treat as "pinned" for our virtual device
    if (size == 0) {
        *pHost = (void*)0x1; // sentinel non-null
        return 0;
    }
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    void* ptr = NULL;
    // 64-byte alignment is sufficient for most purposes
    if (posix_memalign(&ptr, 64, size) != 0) {
        *pHost = NULL;
        return 2; // cudaErrorMemoryAllocation (approximate)
    }
#else
    void* ptr = malloc(size);
    if (!ptr) { *pHost = NULL; return 2; }
#endif
    memset(ptr, 0, size);
    *pHost = ptr;
    return 0;
}
cudaError_t cudaFreeHost(void* pHost) {
    if (!pHost || pHost == (void*)0x1) return 0;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    free(pHost);
#else
    free(pHost);
#endif
    return 0;
}
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) { (void)ptr; (void)size; (void)flags; return 0; }
cudaError_t cudaHostUnregister(void* ptr) { (void)ptr; return 0; }

cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) {
    (void)flags;
    if (pDevice) {
        *pDevice = pHost;
    }
    return 0;
}

// PCI bus id helper
cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) {
    (void)device; if (pciBusId && len>0) pciBusId[0] = '\0'; return 0;
}

// Pointer attributes
cudaError_t cudaPointerGetAttributes(void* attributes, const void* ptr) {
    (void)attributes; (void)ptr; return 0;
}

// IPC APIs
cudaError_t cudaIpcGetEventHandle(void* handle, cudaEvent_t event) { (void)handle; (void)event; return 0; }
cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, void* handle) { if (event) *event = (cudaEvent_t)0; (void)handle; return 0; }
cudaError_t cudaIpcGetMemHandle(void* handle, void* devPtr) { (void)handle; (void)devPtr; return 0; }
cudaError_t cudaIpcOpenMemHandle(void** devPtr, void* handle, unsigned int flags) { if (devPtr) *devPtr = (void*)0; (void)handle; (void)flags; return 0; }
cudaError_t cudaIpcCloseMemHandle(void* devPtr) { (void)devPtr; return 0; }

// Graph APIs (additional)
cudaError_t cudaGraphDestroy(cudaGraph_t graph) { (void)graph; return 0; }
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) { (void)graphExec; return 0; }
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) { (void)graphExec; (void)stream; return 0; }
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* graphExec, cudaGraph_t graph, void* errNode_out, char* logBuffer, size_t bufferSize, unsigned long long flags) {
    (void)graph; (void)errNode_out; (void)logBuffer; (void)bufferSize; (void)flags; if (graphExec) *graphExec = (cudaGraphExec_t)0; return 0;
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* graphExec,
                                 cudaGraph_t graph,
                                 cudaGraphNode_t* pErrorNode,
                                 char* logBuffer,
                                 size_t bufferSize) {
    (void)graph; (void)pErrorNode; (void)logBuffer; (void)bufferSize;
    if (graphExec) *graphExec = (cudaGraphExec_t)0;
    return 0;
}
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) {
    (void)graph; (void)nodes; if (numNodes) *numNodes = 0; return 0;
}
cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) { (void)graph; (void)path; (void)flags; return 0; }

cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode,
                                        cudaGraph_t graph,
                                        const cudaGraphNode_t* pDependencies,
                                        size_t numDependencies,
                                        cudaEvent_t event) {
    (void)graph; (void)pDependencies; (void)numDependencies; (void)event;
    if (pGraphNode) {
        *pGraphNode = (cudaGraphNode_t)0;
    }
    return 0;
}

cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode,
                                      cudaGraph_t graph,
                                      const cudaGraphNode_t* pDependencies,
                                      size_t numDependencies,
                                      cudaEvent_t event) {
    (void)graph; (void)pDependencies; (void)numDependencies; (void)event;
    if (pGraphNode) {
        *pGraphNode = (cudaGraphNode_t)0;
    }
    return 0;
}

cudaError_t cudaGraphAddDependencies(cudaGraph_t graph,
                                     const cudaGraphNode_t* from,
                                     const cudaGraphNode_t* to,
                                     size_t numDependencies) {
    (void)graph;
    (void)from;
    (void)to;
    (void)numDependencies;
    return 0;
}

cudaError_t cudaGraphAddDependencies_v2(cudaGraph_t graph,
                                        const cudaGraphNode_t* from,
                                        const cudaGraphNode_t* to,
                                        size_t numDependencies) {
    return cudaGraphAddDependencies(graph, from, to, numDependencies);
}

cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph,
                                      void* object,
                                      unsigned int count) {
    (void)graph;
    (void)object;
    (void)count;
    return 0;
}

cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph,
                                       void* object,
                                       unsigned int count) {
    (void)graph;
    (void)object;
    (void)count;
    return 0;
}

cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out,
                                 void* ptr,
                                 cudaHostFn_t destroy,
                                 unsigned int initialRefcount,
                                 unsigned int flags) {
    if (!object_out) {
        return 1; // cudaErrorInvalidValue
    }

    HetGPUUserObject* obj = (HetGPUUserObject*)malloc(sizeof(HetGPUUserObject));
    if (!obj) {
        *object_out = NULL;
        return 2; // cudaErrorMemoryAllocation
    }

    obj->payload = ptr;
    obj->destroy = destroy;
    obj->flags = flags;
    obj->refcount = (initialRefcount == 0) ? 1U : initialRefcount;

    *object_out = (cudaUserObject_t)obj;
    return 0;
}

cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) {
    if (!object) {
        return 1; // cudaErrorInvalidValue
    }

    HetGPUUserObject* obj = (HetGPUUserObject*)object;
    if (count == 0) {
        count = 1;
    }
    obj->refcount += count;
    return 0;
}

cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) {
    if (!object) {
        return 0;
    }

    HetGPUUserObject* obj = (HetGPUUserObject*)object;
    if (count == 0) {
        count = 1;
    }

    if (count >= obj->refcount) {
        if (obj->destroy) {
            obj->destroy(obj->payload);
        }
        free(obj);
    } else {
        obj->refcount -= count;
    }
    return 0;
}

// Occupancy/API helpers
cudaError_t cudaFuncSetAttribute(const void* func, int attr, int value) { (void)func; (void)attr; (void)value; return 0; }
cudaError_t cudaFuncGetAttributes(void* attr, const void* func) { (void)attr; (void)func; return 0; }
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    // Return a conservative, non-zero occupancy to avoid divide-by-zero
    // in frameworks that use this to size reductions (e.g., PyTorch).
    // We don't model real hardware here, so pick the safest minimal value.
    if (numBlocks) {
        *numBlocks = 1;
    }
    (void)func; (void)blockSize; (void)dynamicSMemSize; (void)flags;
    return 0;
}

// Provide the older API variant as an alias to the WithFlags version.
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, 0);
}
// Estimate reasonable defaults for potential block size selection
cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, const void* func, size_t dynamicSMemSize, int blockSizeLimit) {
    (void)func; (void)dynamicSMemSize;
    if (blockSize) *blockSize = (blockSizeLimit > 0) ? blockSizeLimit : 256;
    if (minGridSize) *minGridSize = 1;
    return 0;
}
typedef size_t (*cudaOccSMemSizeFn)(int);
cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMem(int* minGridSize, int* blockSize, const void* func, cudaOccSMemSizeFn blockSizeToDynamicSMemSize, int blockSizeLimit) {
    (void)func; (void)blockSizeToDynamicSMemSize;
    if (blockSize) *blockSize = (blockSizeLimit > 0) ? blockSizeLimit : 256;
    if (minGridSize) *minGridSize = 1;
    return 0;
}
cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) { if (mode) *mode = 0; return 0; }
cudaError_t cudaLaunchKernelExC(const void* params) { (void)params; return 0; }

// Internal CUDA launch/config registries (stubs)
cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    (void)gridDim; (void)blockDim; (void)sharedMem; (void)stream; return 0;
}

cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, cudaStream_t* stream) {
    if (gridDim) { gridDim->x = gridDim->y = gridDim->z = 1; }
    if (blockDim) { blockDim->x = blockDim->y = blockDim->z = 1; }
    if (sharedMem) { *sharedMem = 0; }
    if (stream) { *stream = (cudaStream_t)0; }
    return 0;
}

// Forward declaration for cuLaunchKernel from driver API
typedef void* CUfunction;
typedef void* CUstream;
extern CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra
);

// Fat binary registration - map host function pointers to Driver API handles
#define MAX_MODULES 64
#define MAX_FUNCTIONS 1024

typedef struct {
    CUmodule module;
    void* fatCubinHandle;
} RegisteredModule;

typedef struct {
    void* hostFun;           // Host function pointer (from PyTorch)
    CUfunction cuFunc;       // Driver API function handle
    char name[256];          // Kernel name for debugging
    CUmodule module;         // Parent module
} RegisteredFunction;

static RegisteredModule g_modules[MAX_MODULES];
static int g_module_count = 0;
static RegisteredFunction g_functions[MAX_FUNCTIONS];
static int g_function_count = 0;

// Some code paths call the runtime API cudaLaunchKernel (not the internal __cudaLaunchKernel).
// Provide a wrapper that forwards to our internal hook and mark it used so the
// symbol is always exported even if the linker tries to fold identical bodies.
__attribute__((used))
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    uintptr_t raw = (uintptr_t)func;
    const void* normalized = (const void*)(raw & ~(uintptr_t)0x7);
    if (normalized != func) {
        HETGPU_LOG("[cudart_shim] cudaLaunchKernel normalized function pointer %p -> %p\n", func, normalized);
    }
    return __cudaLaunchKernel(normalized, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t __cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    HETGPU_LOG("[cudart_shim] __cudaLaunchKernel intercepted!\n");
    HETGPU_LOG("  func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), sharedMem=%zu\n",
            func, gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z, sharedMem);
#if defined(HETGPU_DEBUG_LOGS)
    fflush(stderr);
#endif

    if (func == NULL) {
        fprintf(stderr, "[cudart_shim] ERROR: NULL function pointer\n");
        return 1;  // cudaErrorInvalidValue
    }

    // Look up the function in our registration table
    CUfunction cuFunc = NULL;
    const char* funcName = "<unknown>";

    for (int i = 0; i < g_function_count; i++) {
        if (g_functions[i].hostFun == func) {
            cuFunc = g_functions[i].cuFunc;
            funcName = g_functions[i].name;
            fprintf(stderr, "[cudart_shim] Found registered function '%s': %p -> %p\n",
                    funcName, func, cuFunc);
            break;
        }
    }

    if (cuFunc == NULL) {
        // Function not found in registry - might be from older code path
        fprintf(stderr, "[cudart_shim] Function %p not in registry, using as-is\n", func);
        cuFunc = (CUfunction)func;
    }

    // Forward to Driver API cuLaunchKernel
    // This routes through our Rust implementation in function.rs
    // which has PTX extraction and cocotb execution support
    CUresult result = cuLaunchKernel(
        cuFunc,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        (unsigned int)sharedMem,
        (CUstream)stream,
        args,
        NULL  // extra parameters
    );

    fprintf(stderr, "[cudart_shim] cuLaunchKernel('%s') returned: %d\n", funcName, result);
    fflush(stderr);
#endif

    return (cudaError_t)result;
}

void** __cudaRegisterFatBinary(void* fatCubin) {
    fprintf(stderr, "[cudart_shim] __cudaRegisterFatBinary called with %p\n", fatCubin);

    if (!fatCubin) {
        fprintf(stderr, "[cudart_shim] ERROR: NULL fatCubin!\n");
        static void* dummy = NULL;
        return &dummy;
    }

    // Fat binary starts with magic number followed by version
    unsigned int* magic = (unsigned int*)fatCubin;
    fprintf(stderr, "[cudart_shim] Fat binary magic: 0x%08x\n", magic[0]);

    // Expected magic: 0x466243B1 for CUDA fat binary (little-endian "1BCF")
    // Version is in magic[1]

    // Fat binary format (simplified):
    // struct __fatBinC_Wrapper_t {
    //     int magic;           // 0x466243B1
    //     int version;         // 1
    //     void* data;          // pointer to __fatBinC_t
    //     void* filename;
    // }

    void** wrapper = (void**)fatCubin;
    void* data = (magic[0] == 0x466243B1 && wrapper[1]) ? wrapper[1] : (void*)((char*)fatCubin + 16);

    // For now, try to extract the first CUBIN we can find
    // Real implementation would parse the full fat binary structure

    // Try to load as raw binary via Driver API
    // This will trigger our PTX extraction in module.rs
    CUmodule module = NULL;

    // NOTE: Don't call cuInit here - let PyTorch handle initialization
    // PyTorch will call cuInit/cudaGetDeviceCount before registration
    // Calling it here causes "initialization error" conflicts

    // Try to load the data as a module
    // cuModuleLoadData expects either PTX or CUBIN
    CUresult result = cuModuleLoadData(&module, data);

    if (result != 0) {
        fprintf(stderr, "[cudart_shim] cuModuleLoadData failed: %d\n", result);
        fprintf(stderr, "[cudart_shim] Trying offset +16...\n");
        result = cuModuleLoadData(&module, (char*)fatCubin + 16);
    }

    if (result != 0) {
        fprintf(stderr, "[cudart_shim] cuModuleLoadData failed again: %d\n", result);
        fprintf(stderr, "[cudart_shim] Module load failed, but continuing with placeholder\n");
        // Don't fail completely - return a handle even if load fails
        // This allows the rest of the code to continue
    } else {
        fprintf(stderr, "[cudart_shim] Successfully loaded module: %p\n", module);
    }

    // Store the module
    if (g_module_count < MAX_MODULES) {
        g_modules[g_module_count].module = module;
        g_modules[g_module_count].fatCubinHandle = fatCubin;
        g_module_count++;

        fprintf(stderr, "[cudart_shim] Registered module %d (total: %d)\n",
                g_module_count - 1, g_module_count);
    }

    // Return the module handle as the fatCubinHandle
    // PyTorch will pass this back to __cudaRegisterFunction
    static void* handle_storage[MAX_MODULES];
    handle_storage[g_module_count - 1] = (void*)module;
    return &handle_storage[g_module_count - 1];
}

void __cudaRegisterFatBinaryEnd(void** fatCubinHandle) {
    fprintf(stderr, "[cudart_shim] __cudaRegisterFatBinaryEnd called\n");
    (void)fatCubinHandle;
}

void __cudaUnregisterFatBinary(void** fatCubinHandle) {
    fprintf(stderr, "[cudart_shim] __cudaUnregisterFatBinary called\n");
    (void)fatCubinHandle;
}

void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun,
                            const char* deviceName, int thread_limit, void* tid, void* bid,
                            void* bDim, void* gDim, void* wSize) {
    (void)deviceFun; (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;

    if (!fatCubinHandle || !hostFun || !deviceName) {
        fprintf(stderr, "[cudart_shim] __cudaRegisterFunction: invalid arguments\n");
        return;
    }

    CUmodule module = (CUmodule)(*fatCubinHandle);
    fprintf(stderr, "[cudart_shim] __cudaRegisterFunction: hostFun=%p, name='%s', module=%p\n",
            hostFun, deviceName, module);

    // Get the function from the module
    CUfunction func = NULL;
    CUresult result = cuModuleGetFunction(&func, module, deviceName);

    if (result != 0) {
        fprintf(stderr, "[cudart_shim] cuModuleGetFunction('%s') failed: %d\n", deviceName, result);
        // Continue anyway - func will be NULL, which we handle in launch
    } else {
        fprintf(stderr, "[cudart_shim] Got function '%s': %p\n", deviceName, func);
    }

    // Store the mapping
    if (g_function_count < MAX_FUNCTIONS) {
        g_functions[g_function_count].hostFun = (void*)hostFun;
        g_functions[g_function_count].cuFunc = func;
        g_functions[g_function_count].module = module;
        strncpy(g_functions[g_function_count].name, deviceName, 255);
        g_functions[g_function_count].name[255] = '\0';

        fprintf(stderr, "[cudart_shim] Registered function %d: %p -> %p ('%s')\n",
                g_function_count, hostFun, func, deviceName);

        g_function_count++;
    } else {
        fprintf(stderr, "[cudart_shim] WARNING: Function table full!\n");
    }
}

void __cudaRegisterVar(void** fatCubinHandle,
                       char* hostVar,
                       char* deviceAddress,
                       const char* deviceName,
                       int ext,
                       size_t size,
                       int constant,
                       int global) {
    (void)fatCubinHandle; (void)hostVar; (void)deviceAddress; (void)deviceName;
    (void)ext; (void)size; (void)constant; (void)global;
}

void* __cudaGetKernel(const void* f) { return (void*)f; }

cudaError_t __cudaInitModule(void** fatCubinHandle) {
    (void)fatCubinHandle;
    return 0;
}

// Driver entry point query
cudaError_t cudaGetDriverEntryPoint(const char* symbol,
                                   void** funcPtr,
                                   int driverVersion,
                                   unsigned long long flags) {
    // CUDA 12 introduced cudaGetDriverEntryPoint as a thin wrapper
    // over the versioned API. We defer to the ByVersion variant so
    // both entry points share the same behavior in the shim.
    return cudaGetDriverEntryPointByVersion(symbol, funcPtr, driverVersion, flags);
}

cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol,
                                             void** funcPtr,
                                             int driverVersion,
                                             unsigned long long flags) {
    (void)symbol; (void)driverVersion; (void)flags;
    if (funcPtr) *funcPtr = (void*)0;
    return 0;
}

// Last error query
cudaError_t cudaGetLastError(void) { return 0; }

cudaError_t cudaPeekAtLastError(void) { return 0; }

// Mempool APIs (stubs)
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) {
    (void)device; if (memPool) *memPool = (cudaMemPool_t)0; return 0;
}

// Profiler stubs
cudaError_t cudaProfilerStart(void) { return 0; }
cudaError_t cudaProfilerStop(void) { return 0; }

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) {
    (void)memPool; (void)minBytesToKeep; return 0;
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, int attr, void* value) {
    (void)memPool; (void)attr; (void)value; return 0;
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, int attr, const void* value) {
    (void)memPool; (void)attr; (void)value; return 0;
}

cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const void* descList, size_t count) {
    (void)memPool; (void)descList; (void)count; return 0;
}

cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool, const void* poolProps) {
    (void)poolProps;
    if (memPool) {
        *memPool = (cudaMemPool_t)0;
    }
    return 0;
}

cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) {
    (void)memPool;
    return 0;
}

cudaError_t cudaMallocFromPoolAsync(void** ptr,
                                    size_t size,
                                    cudaMemPool_t memPool,
                                    cudaStream_t stream) {
    (void)memPool;
    (void)stream;
    return cudaMalloc(ptr, size);
}

// Memory info
cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    const size_t sixteen_gb = (size_t)16 * 1024 * 1024 * 1024ULL;
    if (free) *free = sixteen_gb;
    if (total) *total = sixteen_gb;
    return 0;
}

// Basic memory/runtime APIs - forward to driver API for proper tracking
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (!devPtr) return 1; // cudaErrorInvalidValue

    // Ensure a current context exists (PyTorch may not call cudaSetDevice first)
    CUcontext cur = NULL;
    (void)cuCtxGetCurrent(&cur);
    if (cur == NULL) {
        int dev = 0;
        (void)cudaGetDevice(&dev);
        (void)cudaSetDevice(dev);
        (void)cuCtxGetCurrent(&cur);
    }

    CUdeviceptr dptr = 0;
    CUresult result = cuMemAlloc_v2(&dptr, size);
    if (result != 0) {
        HETGPU_LOG("[cudart_shim] cudaMalloc(%zu) cuMemAlloc_v2 failed: %d; falling back to host alloc\n", size, result);
        // Fallback: host allocation (zeroed)
        void* ptr = NULL;
        if (size > 0) {
            ptr = aligned_alloc(64, ((size + 63) / 64) * 64);
            if (ptr) memset(ptr, 0, size);
        } else {
            ptr = (void*)0x1; // sentinel
        }
        *devPtr = ptr;
        return ptr ? 0 : 2; // cudaErrorMemoryAllocation if NULL
    }
    *devPtr = (void*)dptr;
    HETGPU_LOG("[cudart_shim] cudaMalloc(%zu) -> %p\n", size, *devPtr);
    return 0;
}

cudaError_t cudaFree(void* devPtr) {
    if (!devPtr || devPtr == (void*)0x1) return 0;
    // Try driver free first
    CUresult result = cuMemFree_v2((CUdeviceptr)devPtr);
    if (result != 0) {
        // Not a driver-managed pointer, fallback to host free
        HETGPU_LOG("[cudart_shim] cudaFree(%p) cuMemFree_v2 failed: %d; freeing as host ptr\n", devPtr, result);
        free(devPtr);
        return 0;
    }
    return 0;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if (!dst || !src || count == 0) return 0;
    // Treat all kinds as host memcpy in virtual backend
    // This covers H2D/D2H by virtue of using host-backed "device" pointers
    memcpy(dst, src, count);
    return 0;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    (void)stream; return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) {
    (void)dstDevice; (void)srcDevice; (void)stream;
    if (dst && src && count > 0) {
        memcpy(dst, src, count);
    }
    return 0;
}

cudaError_t cudaMemcpyToSymbol(const void* symbol,
                               const void* src,
                               size_t count,
                               size_t offset,
                               cudaMemcpyKind kind) {
    (void)kind;
    if (!symbol || !src || count == 0) {
        return 0;
    }
    unsigned char* dst_bytes = (unsigned char*)(uintptr_t)symbol;
    memcpy(dst_bytes + offset, src, count);
    return 0;
}

cudaError_t cudaMemcpyToSymbolAsync(const void* symbol,
                                    const void* src,
                                    size_t count,
                                    size_t offset,
                                    cudaMemcpyKind kind,
                                    cudaStream_t stream) {
    (void)stream;
    return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t cudaMemcpyFromSymbol(void* dst,
                                 const void* symbol,
                                 size_t count,
                                 size_t offset,
                                 cudaMemcpyKind kind) {
    (void)kind;
    if (!dst || !symbol || count == 0) {
        return 0;
    }
    const unsigned char* src_bytes = (const unsigned char*)(uintptr_t)symbol;
    memcpy(dst, src_bytes + offset, count);
    return 0;
}

cudaError_t cudaMemcpyFromSymbolAsync(void* dst,
                                      const void* symbol,
                                      size_t count,
                                      size_t offset,
                                      cudaMemcpyKind kind,
                                      cudaStream_t stream) {
    (void)stream;
    return cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
}

cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol) {
    if (!devPtr) {
        return 1; // cudaErrorInvalidValue
    }
    *devPtr = (void*)(uintptr_t)symbol;
    return 0;
}

cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol) {
    (void)symbol;
    if (size) {
        *size = 0;
    }
    return 0;
}

cudaError_t cudaGetFuncBySymbol(cudaFunction_t* functionPtr, const void* symbol) {
    if (!functionPtr) {
        return 1; // cudaErrorInvalidValue
    }
    *functionPtr = (cudaFunction_t)(uintptr_t)symbol;
    return 0;
}

cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
    (void)stream; return cudaMalloc(devPtr, size);
}

cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    (void)stream; return cudaFree(devPtr);
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    if (!devPtr || devPtr == (void*)0x1 || count == 0) return 0;
    memset(devPtr, (unsigned char)value, count);
    return 0;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    (void)stream;
    return cudaMemset(devPtr, value, count);
}

// Device stream priority range (stub)
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority,
                                             int* greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = 0;
    return 0; // cudaSuccess
}
