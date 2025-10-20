#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
extern CUresult cuDeviceGet(CUdevice* device, int ordinal);
extern CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
extern CUresult cuCtxSetCurrent(CUcontext ctx);
extern CUresult cuCtxGetCurrent(CUcontext* pctx);
extern CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);
extern CUresult cuMemFree_v2(CUdeviceptr dptr);
extern CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
extern CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
extern CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N);

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
typedef struct { unsigned int x, y, z; } dim3;
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* userData);
typedef void (*cudaHostFn_t)(void* userData);
typedef int cudaMemcpyKind; // use int placeholder

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

// Minimal subset of cudaDeviceProp to populate capability and a few basics.
// This matches the leading fields of CUDA's cudaDeviceProp across versions.
typedef struct {
    char   name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int    regsPerBlock;
    int    warpSize;
    size_t memPitch;
    int    maxThreadsPerBlock;
    int    maxThreadsDim[3];
    int    maxGridSize[3];
    int    clockRate;
    size_t totalConstMem;
    int    major;
    int    minor;
} cudaDeviceProp_min;

cudaError_t cudaGetDeviceProperties(cudaDeviceProp_t prop, int device) {
    if (!prop) return 1; // cudaErrorInvalidValue

    // Fill a minimal struct and copy into caller memory
    cudaDeviceProp_min p;
    memset(&p, 0, sizeof(p));

    // Device name
    const char* name = "Virtual GPU (hetGPU sm_80)";
    strncpy(p.name, name, sizeof(p.name) - 1);

    // Conservative, GPU-like defaults
    p.warpSize = 32;
    p.maxThreadsPerBlock = 1024;
    p.maxThreadsDim[0] = 1024; p.maxThreadsDim[1] = 1024; p.maxThreadsDim[2] = 64;
    p.maxGridSize[0] = 2147483647; p.maxGridSize[1] = 65535; p.maxGridSize[2] = 65535;
    p.clockRate = 1410000; // kHz
    p.totalConstMem = 64 * 1024;

    // Compute capability (match driver attribute path)
    p.major = 8;
    p.minor = 0;

    memcpy(prop, &p, sizeof(p));
    fprintf(stderr, "[cudart_shim] cudaGetDeviceProperties: name='%s' cc=%d.%d\n", p.name, p.major, p.minor);

    // Defensive: also stamp major/minor at several known offsets used by
    // different CUDA headers to avoid layout mismatches.
    // These offsets are relative to the start of cudaDeviceProp and cover
    // common placements (immediately after totalConstMem).
    {
        unsigned char* base = (unsigned char*)prop;
        const int major = 8, minor = 0;
        // Try multiple known offsets to accommodate struct layout differences.
        size_t cand_major[] = { 316, 320, 328, 332, 336, 344 };
        for (size_t i = 0; i < sizeof(cand_major)/sizeof(cand_major[0]); ++i) {
            // Write major, then minor adjacent
            *(int*)(base + cand_major[i]) = major;
            *(int*)(base + cand_major[i] + 4) = minor;
        }
    }
    (void)device;
    return 0;
}

// Global to track current device
static int current_device = 0;

cudaError_t cudaSetDevice(int device) {
    // Get the CUDA device handle
    CUdevice cu_device;
    CUresult result = cuDeviceGet(&cu_device, device);
    if (result != 0) {
        return (cudaError_t)result;
    }

    // Retain the primary context for this device
    CUcontext ctx;
    result = cuDevicePrimaryCtxRetain(&ctx, cu_device);
    if (result != 0) {
        return (cudaError_t)result;
    }

    // Set it as the current context
    result = cuCtxSetCurrent(ctx);
    if (result != 0) {
        return (cudaError_t)result;
    }

    current_device = device;
    return 0;
}

cudaError_t cudaGetDevice(int* device) {
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

// Device attribute query
cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device) {
    if (!value) return 1; // cudaErrorInvalidValue
    fprintf(stderr, "[cudart_shim] cudaDeviceGetAttribute(attr=%d, dev=%d)\n", attr, device);
    // Provide sane defaults; explicitly handle common CC queries
    if (attr == 75 /* cudaDevAttrComputeCapabilityMajor */) { *value = 8; return 0; }
    if (attr == 76 /* cudaDevAttrComputeCapabilityMinor */) { *value = 0; return 0; }
    // Generic non-zero default to avoid divide-by-zero in upstream code
    *value = 1;
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

// Some code paths call the runtime API cudaLaunchKernel (not the internal __cudaLaunchKernel).
// Provide a wrapper that forwards to our internal hook.
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    return __cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t __cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    fprintf(stderr, "[cudart_shim] __cudaLaunchKernel intercepted!\n");
    fprintf(stderr, "  func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), sharedMem=%zu\n",
            func, gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z, sharedMem);
    fflush(stderr);

    // Forward to driver API cuLaunchKernel
    // This routes through our Rust implementation in function.rs
    CUresult result = cuLaunchKernel(
        (CUfunction)func,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        (unsigned int)sharedMem,
        (CUstream)stream,
        args,
        NULL  // extra parameters
    );

    fprintf(stderr, "[cudart_shim] cuLaunchKernel returned: %d\n", result);
    fflush(stderr);

    return (cudaError_t)result;
}

void** __cudaRegisterFatBinary(void* fatCubin) { (void)fatCubin; static void* handle; return &handle; }
void __cudaRegisterFatBinaryEnd(void** fatCubinHandle) { (void)fatCubinHandle; }
void __cudaUnregisterFatBinary(void** fatCubinHandle) { (void)fatCubinHandle; }
void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun, const char* deviceName, int thread_limit, void* tid, void* bid, void* bDim, void* gDim, void* wSize) {
    (void)fatCubinHandle; (void)hostFun; (void)deviceFun; (void)deviceName; (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;
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

// Driver entry point query
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
        fprintf(stderr, "[cudart_shim] cudaMalloc(%zu) cuMemAlloc_v2 failed: %d; falling back to host alloc\n", size, result);
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
    fprintf(stderr, "[cudart_shim] cudaMalloc(%zu) -> %p\n", size, *devPtr);
    return 0;
}

cudaError_t cudaFree(void* devPtr) {
    if (!devPtr || devPtr == (void*)0x1) return 0;
    // Try driver free first
    CUresult result = cuMemFree_v2((CUdeviceptr)devPtr);
    if (result != 0) {
        // Not a driver-managed pointer, fallback to host free
        fprintf(stderr, "[cudart_shim] cudaFree(%p) cuMemFree_v2 failed: %d; freeing as host ptr\n", devPtr, result);
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
