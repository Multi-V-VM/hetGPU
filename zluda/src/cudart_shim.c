#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <execinfo.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <ucontext.h>
#include <dlfcn.h>

// SIGFPE handler - log first occurrence then disable FP exceptions to continue
static int sigfpe_count = 0;

static void sigfpe_handler(int sig, siginfo_t *info, void *ucontext_raw) {
    sigfpe_count++;
    if (sigfpe_count <= 3) {
        fprintf(stderr, "\n[hetGPU] SIGFPE #%d caught (expected with virtual device no-op kernels)\n", sigfpe_count);
        if (sigfpe_count == 1) {
            void* array[10];
            size_t size = backtrace(array, 10);
            backtrace_symbols_fd(array, size, STDERR_FILENO);
        }
        fprintf(stderr, "[hetGPU] Continuing execution (output values may be NaN/Inf)\n");
    }

    // On x86_64, advance RIP past the faulting div/idiv instruction
    // This allows the program to continue with undefined results
    ucontext_t *uc = (ucontext_t *)ucontext_raw;
    // Skip 2-3 bytes (typical div/idiv instruction length on x86_64)
    // Look at the instruction to determine length
    unsigned char *rip = (unsigned char *)uc->uc_mcontext.gregs[REG_RIP];
    int skip = 2; // default: 2-byte div instruction
    if (rip[0] == 0xF7 || rip[0] == 0xF6) {
        // div/idiv with modrm byte - at least 2 bytes
        unsigned char modrm = rip[1];
        skip = 2;
        // Add SIB byte if present
        if ((modrm & 0x07) == 0x04 && (modrm & 0xC0) != 0xC0) skip++;
        // Add displacement
        if ((modrm & 0xC0) == 0x40) skip++;
        else if ((modrm & 0xC0) == 0x80) skip += 4;
    } else if (rip[0] == 0x48 || rip[0] == 0x49) {
        // REX prefix + div/idiv
        skip = 3;
        unsigned char modrm = rip[2];
        if ((modrm & 0x07) == 0x04 && (modrm & 0xC0) != 0xC0) skip++;
        if ((modrm & 0xC0) == 0x40) skip++;
        else if ((modrm & 0xC0) == 0x80) skip += 4;
    }
    uc->uc_mcontext.gregs[REG_RIP] += skip;
    // Clear the divide-by-zero result register to prevent cascading errors
    uc->uc_mcontext.gregs[REG_RAX] = 0;
    uc->uc_mcontext.gregs[REG_RDX] = 0;
}

__attribute__((constructor))
static void install_sigfpe_handler(void) {
    struct sigaction sa;
    sa.sa_sigaction = sigfpe_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGFPE, &sa, NULL);
    fprintf(stderr, "[hetGPU] SIGFPE handler installed (non-fatal)\n");
}

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

// Forward declarations for functions called before they're defined
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol,
                                             void** funcPtr,
                                             int driverVersion,
                                             unsigned long long flags);
cudaError_t __cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);

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
    if (pStream) *pStream = (cudaStream_t)0;
    return 0;
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
// Declare driver API functions (defined in Rust) that we call through to
extern int cuDeviceGetCount(int* count);
extern int cuDriverGetVersion(int* version);
extern int cuInit(unsigned int flags);

cudaError_t cudaGetDeviceCount(int* count) {
    if (!count) return 1; // cudaErrorInvalidValue
    fprintf(stderr, "[hetGPU] cudaGetDeviceCount called\n");
    cuInit(0);
    int result = cuDeviceGetCount(count);
    fprintf(stderr, "[hetGPU] cudaGetDeviceCount: %d devices\n", count ? *count : -1);
    return (result == 0) ? 0 : 2; // cudaSuccess or cudaErrorMemoryAllocation
}

cudaError_t cudaDriverGetVersion(int* version) {
    if (!version) return 1;
    cuDriverGetVersion(version);
    return 0;
}

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
    fprintf(stderr, "[hetGPU] cudaSetDevice(%d) called\n", device);
    // For virtual device support, be permissive
    if (device < 0) {
        fprintf(stderr, "[hetGPU] cudaSetDevice: invalid device\n");
        return 1; // cudaErrorInvalidDevice
    }

    // Get the CUDA device handle
    CUdevice cu_device;
    CUresult result = cuDeviceGet(&cu_device, device);
    fprintf(stderr, "[hetGPU] cudaSetDevice: cuDeviceGet returned %d, cu_device=%d\n", result, cu_device);
    if (result != 0) {
        // For virtual device, still set current_device and succeed
        fprintf(stderr, "[hetGPU] cudaSetDevice(%d): cuDeviceGet failed (%d), continuing with virtual device\n", device, result);
        current_device = device;
        return 0; // Success for virtual device
    }

    // Retain the primary context for this device
    CUcontext ctx;
    result = cuDevicePrimaryCtxRetain(&ctx, cu_device);
    fprintf(stderr, "[hetGPU] cudaSetDevice: cuDevicePrimaryCtxRetain returned %d, ctx=%p\n", result, ctx);
    if (result != 0) {
        // For virtual device, still set current_device and succeed
        fprintf(stderr, "[hetGPU] cudaSetDevice(%d): cuDevicePrimaryCtxRetain failed (%d), continuing with virtual device\n", device, result);
        current_device = device;
        return 0; // Success for virtual device
    }

    // Set it as the current context
    result = cuCtxSetCurrent(ctx);
    fprintf(stderr, "[hetGPU] cudaSetDevice: cuCtxSetCurrent returned %d\n", result);
    if (result != 0) {
        // For virtual device, still set current_device and succeed
        fprintf(stderr, "[hetGPU] cudaSetDevice(%d): cuCtxSetCurrent failed (%d), continuing with virtual device\n", device, result);
        current_device = device;
        return 0; // Success for virtual device
    }

    current_device = device;
    fprintf(stderr, "[hetGPU] cudaSetDevice: success\n");
    return 0;
}

cudaError_t cudaGetDevice(int* device) {
    fprintf(stderr, "[hetGPU] cudaGetDevice called, returning device %d\n", current_device);
    if (device) *device = current_device;
    return 0;
}

cudaError_t cudaRuntimeGetVersion(int* version) {
    if (version) *version = 12080;
    return 0;
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
    // Full list to prevent any missing attributes causing issues
    int result_value = 1;  // Default non-zero
    switch (attr) {
        case 1:  // cudaDevAttrMaxThreadsPerBlock
            *value = 1024; break;
        case 2:  // cudaDevAttrMaxBlockDimX
            *value = 1024; break;
        case 3:  // cudaDevAttrMaxBlockDimY
            *value = 1024; break;
        case 4:  // cudaDevAttrMaxBlockDimZ
            *value = 64; break;
        case 5:  // cudaDevAttrMaxGridDimX
            *value = 2147483647; break;
        case 6:  // cudaDevAttrMaxGridDimY
            *value = 65535; break;
        case 7:  // cudaDevAttrMaxGridDimZ
            *value = 65535; break;
        case 8:  // cudaDevAttrMaxSharedMemoryPerBlock
            *value = 49152; break;
        case 9:  // cudaDevAttrTotalConstantMemory
            *value = 65536; break;
        case 10: // cudaDevAttrWarpSize
            *value = 32; break;
        case 11: // cudaDevAttrMaxPitch
            *value = 2147483647; break;
        case 12: // cudaDevAttrMaxRegistersPerBlock
            *value = 65536; break;
        case 13: // cudaDevAttrClockRate
            *value = 1410000; break;
        case 14: // cudaDevAttrTextureAlignment
            *value = 512; break;
        case 15: // cudaDevAttrGpuOverlap
            *value = 1; break;
        case 16: // cudaDevAttrMultiProcessorCount
            *value = 80; break;
        case 17: // cudaDevAttrKernelExecTimeout
            *value = 0; break;
        case 18: // cudaDevAttrIntegrated
            *value = 0; break;
        case 19: // cudaDevAttrCanMapHostMemory
            *value = 1; break;
        case 20: // cudaDevAttrComputeMode
            *value = 0; break;
        case 21: // cudaDevAttrMaxTexture1DWidth
            *value = 131072; break;
        case 22: // cudaDevAttrMaxTexture2DWidth
            *value = 131072; break;
        case 23: // cudaDevAttrMaxTexture2DHeight
            *value = 65536; break;
        case 24: // cudaDevAttrMaxTexture3DWidth
            *value = 16384; break;
        case 25: // cudaDevAttrMaxTexture3DHeight
            *value = 16384; break;
        case 26: // cudaDevAttrMaxTexture3DDepth
            *value = 16384; break;
        case 32: // cudaDevAttrConcurrentKernels
            *value = 1; break;
        case 33: // cudaDevAttrEccEnabled
            *value = 0; break;
        case 34: // cudaDevAttrPciBusId
            *value = 0; break;
        case 35: // cudaDevAttrPciDeviceId
            *value = 0; break;
        case 36: // cudaDevAttrTccDriver
            *value = 0; break;
        case 37: // cudaDevAttrMemoryClockRate
            *value = 1215000; break;
        case 38: // cudaDevAttrGlobalMemoryBusWidth
            *value = 5120; break;
        case 39: // cudaDevAttrL2CacheSize
            *value = 41943040; break;  // 40MB
        case 40: // cudaDevAttrMaxThreadsPerMultiProcessor
            *value = 2048; break;  // CRITICAL for occupancy calculations!
        case 41: // cudaDevAttrAsyncEngineCount
            *value = 2; break;
        case 42: // cudaDevAttrUnifiedAddressing
            *value = 1; break;
        case 44: // cudaDevAttrMaxTexture1DLayeredWidth
            *value = 32768; break;
        case 45: // cudaDevAttrMaxTexture1DLayeredLayers
            *value = 2048; break;
        case 53: // cudaDevAttrMaxTexture2DGatherWidth
            *value = 32768; break;
        case 54: // cudaDevAttrMaxTexture2DGatherHeight
            *value = 32768; break;
        case 59: // cudaDevAttrMaxTexture2DLinearWidth
            *value = 131072; break;
        case 60: // cudaDevAttrMaxTexture2DLinearHeight
            *value = 65000; break;
        case 61: // cudaDevAttrMaxTexture2DLinearPitch
            *value = 2097120; break;
        case 62: // cudaDevAttrMaxTexture2DMipmappedWidth
            *value = 32768; break;
        case 63: // cudaDevAttrMaxTexture2DMipmappedHeight
            *value = 32768; break;
        case 75: // cudaDevAttrComputeCapabilityMajor
            *value = 8; break;
        case 76: // cudaDevAttrComputeCapabilityMinor
            *value = 0; break;
        case 77: // cudaDevAttrMaxTexture1DMipmappedWidth
            *value = 32768; break;
        case 78: // cudaDevAttrStreamPrioritiesSupported
            *value = 1; break;
        case 79: // cudaDevAttrGlobalL1CacheSupported
            *value = 1; break;
        case 80: // cudaDevAttrLocalL1CacheSupported
            *value = 1; break;
        case 81: // cudaDevAttrMaxSharedMemoryPerMultiprocessor
            *value = 167936; break;  // 164KB for sm_80
        case 82: // cudaDevAttrMaxRegistersPerMultiprocessor
            *value = 65536; break;
        case 83: // cudaDevAttrManagedMemory
            *value = 1; break;
        case 84: // cudaDevAttrIsMultiGpuBoard
            *value = 0; break;
        case 85: // cudaDevAttrMultiGpuBoardGroupID
            *value = 0; break;
        case 86: // cudaDevAttrHostNativeAtomicSupported
            *value = 1; break;
        case 87: // cudaDevAttrSingleToDoublePrecisionPerfRatio
            *value = 2; break;
        case 88: // cudaDevAttrPageableMemoryAccess
            *value = 1; break;
        case 89: // cudaDevAttrConcurrentManagedAccess
            *value = 1; break;
        case 90: // cudaDevAttrComputePreemptionSupported
            *value = 1; break;
        case 91: // cudaDevAttrCanUseHostPointerForRegisteredMem
            *value = 1; break;
        case 95: // cudaDevAttrCooperativeLaunch
            *value = 1; break;
        case 96: // cudaDevAttrCooperativeMultiDeviceLaunch
            *value = 0; break;
        case 97: // cudaDevAttrMaxSharedMemoryPerBlockOptin
            *value = 167936; break;
        case 99: // cudaDevAttrCanFlushRemoteWrites
            *value = 0; break;
        case 100: // cudaDevAttrHostRegisterSupported
            *value = 1; break;
        case 101: // cudaDevAttrPageableMemoryAccessUsesHostPageTables
            *value = 0; break;
        case 102: // cudaDevAttrDirectManagedMemAccessFromHost
            *value = 1; break;
        default:
            // Generic non-zero default to avoid divide-by-zero
            *value = 1;
            fprintf(stderr, "[cudart_shim] cudaDeviceGetAttribute: unknown attr=%d, returning 1\n", attr);
            break;
    }

    fprintf(stderr, "[cudart_shim] cudaDeviceGetAttribute(attr=%d) = %d\n", attr, *value);
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

// PCI bus id helpers
cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) {
    (void)device; if (pciBusId && len>0) pciBusId[0] = '\0'; return 0;
}

cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) {
    (void)pciBusId;
    if (device) *device = 0;  // Return device 0 for any PCI bus ID
    return 0;
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

// cudaFuncAttributes structure - must match CUDA header layout
typedef struct {
    size_t sharedSizeBytes;         // Shared memory per block
    size_t constSizeBytes;          // Constant memory size
    size_t localSizeBytes;          // Local memory per thread
    int maxThreadsPerBlock;         // Max threads per block for this function
    int numRegs;                    // Number of registers used
    int ptxVersion;                 // PTX version
    int binaryVersion;              // Binary version
    int cacheModeCA;                // Cache mode
    int maxDynamicSharedSizeBytes;  // Max dynamic shared memory
    int preferredShmemCarveout;     // Preferred shared memory carveout
    int clusterDimMustBeSet;        // Cluster dimension requirement
    int requiredClusterWidth;       // Required cluster width
    int requiredClusterHeight;      // Required cluster height
    int requiredClusterDepth;       // Required cluster depth
    int clusterSchedulingPolicyPreference; // Cluster scheduling preference
    int nonPortableClusterSizeAllowed;     // Non-portable cluster size flag
    int reserved[16];               // Reserved for future use
} cudaFuncAttributes;

// Occupancy/API helpers
cudaError_t cudaFuncSetAttribute(const void* func, int attr, int value) { (void)func; (void)attr; (void)value; return 0; }
cudaError_t cudaFuncGetAttributes(void* attr, const void* func) {
    (void)func;
    if (!attr) return 1; // cudaErrorInvalidValue

    // Initialize with reasonable default values to prevent division by zero
    cudaFuncAttributes* attrs = (cudaFuncAttributes*)attr;
    memset(attrs, 0, sizeof(cudaFuncAttributes));

    // Critical values that must be non-zero to avoid SIGFPE
    attrs->maxThreadsPerBlock = 1024;       // Standard max for sm_80
    attrs->numRegs = 32;                    // Conservative register count
    attrs->sharedSizeBytes = 0;             // Static shared memory
    attrs->constSizeBytes = 0;              // Constant memory
    attrs->localSizeBytes = 0;              // Local memory per thread
    attrs->ptxVersion = 80;                 // PTX 8.0
    attrs->binaryVersion = 80;              // Binary version 8.0
    attrs->cacheModeCA = 0;                 // Default cache mode
    attrs->maxDynamicSharedSizeBytes = 48 * 1024;  // 48KB dynamic shared
    attrs->preferredShmemCarveout = -1;     // Driver default

    fprintf(stderr, "[cudart_shim] cudaFuncGetAttributes: maxThreadsPerBlock=%d, numRegs=%d\n",
            attrs->maxThreadsPerBlock, attrs->numRegs);

    return 0;
}
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
#define MAX_MODULES 512
#define MAX_FUNCTIONS 32768

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
    // Compare with normalization (mask off low 3 bits) since cudaLaunchKernel
    // and __cudaRegisterFunction may use different pointer representations
    CUfunction cuFunc = NULL;
    const char* funcName = "<unknown>";
    uintptr_t func_normalized = (uintptr_t)func & ~(uintptr_t)0x7;

    for (int i = 0; i < g_function_count; i++) {
        uintptr_t registered_normalized = (uintptr_t)g_functions[i].hostFun & ~(uintptr_t)0x7;
        if (g_functions[i].hostFun == func || registered_normalized == func_normalized) {
            cuFunc = g_functions[i].cuFunc;
            funcName = g_functions[i].name;
            fprintf(stderr, "[cudart_shim] Found registered function '%s': %p -> %p\n",
                    funcName, func, cuFunc);
            break;
        }
    }

    if (cuFunc == NULL) {
        // Function not found in registry - this is a function that wasn't registered via
        // __cudaRegisterFunction (e.g., dynamically loaded or from a different module).
        // For the virtual backend, we can't execute it, so just return success.
        // WARNING: This leaves output tensors uninitialized which may cause SIGFPE!
        fprintf(stderr, "[cudart_shim] WARNING: Function %p not in registry - skipping (output uninitialized!)\n", func);
        fprintf(stderr, "[cudart_shim] This may cause SIGFPE in downstream operations like softmax\n");
        return 0;  // cudaSuccess
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

#if defined(HETGPU_DEBUG_LOGS)
    fprintf(stderr, "[cudart_shim] cuLaunchKernel('%s') returned: %d\n", funcName, result);
    fflush(stderr);
#endif

    return (cudaError_t)result;
}

// Helper to find .so file path from memory address using /proc/self/maps
static int find_so_from_address(const void* addr, char* path, size_t path_size) {
    FILE* maps = fopen("/proc/self/maps", "r");
    if (!maps) return 0;

    char line[1024];
    unsigned long target = (unsigned long)addr;

    while (fgets(line, sizeof(line), maps)) {
        unsigned long start, end;
        char perms[5];
        unsigned long offset;
        int dev_major, dev_minor;
        unsigned long inode;
        char filepath[512] = {0};

        int n = sscanf(line, "%lx-%lx %4s %lx %x:%x %lu %511[^\n]",
                       &start, &end, perms, &offset, &dev_major, &dev_minor, &inode, filepath);

        if (n >= 7 && target >= start && target < end) {
            // Skip if no filepath or if it's a special mapping
            if (n < 8 || filepath[0] == '\0' || filepath[0] == '[') {
                continue;
            }
            // Skip whitespace at start of filepath
            char* fp = filepath;
            while (*fp == ' ') fp++;

            // Check if it's a .so file
            if (strstr(fp, ".so")) {
                strncpy(path, fp, path_size - 1);
                path[path_size - 1] = '\0';
                fclose(maps);
                return 1;
            }
        }
    }

    fclose(maps);
    return 0;
}

// Cache for extracted PTX from .so files
#define MAX_CACHED_SO 16
#define MAX_PTX_PER_SO 512
static struct {
    char so_path[512];
    char ptx_dir[256];
    int ptx_count;
    char ptx_files[MAX_PTX_PER_SO][256];
} g_ptx_cache[MAX_CACHED_SO];
static int g_ptx_cache_count = 0;

// Extract all PTX from a .so file using cuobjdump
static const char* extract_ptx_from_so(const char* so_path) {
    // Check cache first
    for (int i = 0; i < g_ptx_cache_count; i++) {
        if (strcmp(g_ptx_cache[i].so_path, so_path) == 0) {
            fprintf(stderr, "[cudart_shim] Using cached PTX from %s\n", so_path);
            return g_ptx_cache[i].ptx_dir;
        }
    }

    if (g_ptx_cache_count >= MAX_CACHED_SO) {
        fprintf(stderr, "[cudart_shim] PTX cache full, cannot add more .so files\n");
        return NULL;
    }

    // Create cache entry
    int cache_idx = g_ptx_cache_count++;
    strncpy(g_ptx_cache[cache_idx].so_path, so_path, 511);

    // Create output directory
    snprintf(g_ptx_cache[cache_idx].ptx_dir, sizeof(g_ptx_cache[cache_idx].ptx_dir),
             "/tmp/hetgpu_so_ptx_%d", cache_idx);
    mkdir(g_ptx_cache[cache_idx].ptx_dir, 0755);

    fprintf(stderr, "[cudart_shim] Extracting PTX from %s to %s\n",
            so_path, g_ptx_cache[cache_idx].ptx_dir);

    // Run cuobjdump to extract all PTX
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "cd %s && /usr/local/cuda-12.8/bin/cuobjdump --extract-ptx all '%s' 2>&1",
             g_ptx_cache[cache_idx].ptx_dir, so_path);

    FILE* p = popen(cmd, "r");
    if (p) {
        char line[256];
        while (fgets(line, sizeof(line), p)) {
            // Just consume output, maybe log some of it
            if (strstr(line, "Extracting")) {
                fprintf(stderr, "[cudart_shim] %s", line);
            }
        }
        int ret = pclose(p);
        fprintf(stderr, "[cudart_shim] cuobjdump on .so returned: %d\n", WEXITSTATUS(ret));
    }

    // Count extracted PTX files
    char find_cmd[512];
    snprintf(find_cmd, sizeof(find_cmd),
             "find %s -name '*.ptx' -type f 2>/dev/null",
             g_ptx_cache[cache_idx].ptx_dir);

    FILE* find_p = popen(find_cmd, "r");
    if (find_p) {
        char ptx_path[256];
        g_ptx_cache[cache_idx].ptx_count = 0;
        while (fgets(ptx_path, sizeof(ptx_path), find_p) &&
               g_ptx_cache[cache_idx].ptx_count < MAX_PTX_PER_SO) {
            // Remove trailing newline
            size_t len = strlen(ptx_path);
            if (len > 0 && ptx_path[len-1] == '\n') {
                ptx_path[len-1] = '\0';
            }
            if (strlen(ptx_path) > 0) {
                strncpy(g_ptx_cache[cache_idx].ptx_files[g_ptx_cache[cache_idx].ptx_count],
                        ptx_path, 255);
                g_ptx_cache[cache_idx].ptx_count++;
            }
        }
        pclose(find_p);
    }

    fprintf(stderr, "[cudart_shim] Extracted %d PTX files from %s\n",
            g_ptx_cache[cache_idx].ptx_count, so_path);

    return g_ptx_cache[cache_idx].ptx_dir;
}

// Load PTX from a file and return as allocated string
static char* load_ptx_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0 || size > 100*1024*1024) {
        fclose(f);
        return NULL;
    }

    char* data = (char*)malloc(size + 1);
    if (!data) {
        fclose(f);
        return NULL;
    }

    size_t read = fread(data, 1, size, f);
    data[read] = '\0';
    fclose(f);

    if (out_size) *out_size = read;
    return data;
}

// Track which PTX file index we're on for round-robin loading
static int g_ptx_file_index = 0;

// Find a specific PTX file that matches a pattern (or get next one in sequence)
// Returns allocated PTX string, caller must free
static char* find_matching_ptx(const char* ptx_dir, const char* pattern, size_t* out_size) {
    // Find cache entry
    for (int i = 0; i < g_ptx_cache_count; i++) {
        if (strcmp(g_ptx_cache[i].ptx_dir, ptx_dir) == 0) {
            if (g_ptx_cache[i].ptx_count == 0) {
                fprintf(stderr, "[cudart_shim] No PTX files in cache for %s\n", ptx_dir);
                return NULL;
            }

            // If pattern provided, try to find matching file
            if (pattern && strlen(pattern) > 0) {
                for (int j = 0; j < g_ptx_cache[i].ptx_count; j++) {
                    if (strstr(g_ptx_cache[i].ptx_files[j], pattern)) {
                        fprintf(stderr, "[cudart_shim] Found matching PTX file: %s\n",
                                g_ptx_cache[i].ptx_files[j]);
                        return load_ptx_file(g_ptx_cache[i].ptx_files[j], out_size);
                    }
                }
            }

            // No pattern match, use round-robin to get next PTX file
            // This distributes modules across different PTX files
            int idx = g_ptx_file_index % g_ptx_cache[i].ptx_count;
            g_ptx_file_index++;

            fprintf(stderr, "[cudart_shim] Using PTX file %d/%d: %s\n",
                    idx + 1, g_ptx_cache[i].ptx_count,
                    g_ptx_cache[i].ptx_files[idx]);
            return load_ptx_file(g_ptx_cache[i].ptx_files[idx], out_size);
        }
    }
    return NULL;
}

// Concatenate all PTX files into one big PTX string (DISABLED - too slow for large libs)
// Use find_matching_ptx instead for targeted loading
static char* concatenate_all_ptx(const char* ptx_dir, size_t* out_size) {
    (void)ptx_dir;
    (void)out_size;
    fprintf(stderr, "[cudart_shim] WARNING: concatenate_all_ptx disabled (too slow), use find_matching_ptx\n");
    return NULL;
}

// Fat binary structures
typedef struct {
    unsigned int magic;      // 0xBA55ED50
    unsigned short version;  // 0x01
    unsigned short header_size;
    unsigned long files_size;
} FatbinHeader;

typedef struct {
    unsigned short kind;             // 0x01 = PTX, 0x02 = ELF/CUBIN  (offset 0)
    unsigned short version;          // version                       (offset 2)
    unsigned int header_size;        // size of this header           (offset 4)
    unsigned int padded_payload_size;// payload size with padding      (offset 8)
    unsigned int unknown0;           // unknown                        (offset 12)
    unsigned int payload_size;       // actual size of payload data   (offset 16)
    unsigned int unknown1;           // unknown                        (offset 20)
    unsigned int unknown2;           // unknown                        (offset 24)
    unsigned int sm_version;         // sm version (e.g. 0x78=120)    (offset 28)
    unsigned int bit_width;          // 32 or 64                      (offset 32)
    unsigned int unknown3;           // unknown                        (offset 36)
    unsigned long unknown4;          // unknown                        (offset 40)
    unsigned long unknown5;          // unknown                        (offset 48)
    unsigned long uncompressed_payload; // decompressed size (0=not compressed) (offset 56)
} FatbinFileHeader;

// LZ4 decompression (provided by Rust FFI wrapper in lib.rs)
extern int hetgpu_lz4_decompress(const char* src, char* dst, int compressedSize, int dstCapacity);

#define FATBIN_MAGIC 0xBA55ED50
#define FATBIN_KIND_PTX 0x01
#define FATBIN_KIND_ELF 0x02

void** __cudaRegisterFatBinary(void* fatCubin) {
    fprintf(stderr, "[cudart_shim] __cudaRegisterFatBinary called with %p\n", fatCubin);

    if (!fatCubin) {
        fprintf(stderr, "[cudart_shim] ERROR: NULL fatCubin!\n");
        static void* dummy = NULL;
        return &dummy;
    }

    // Fat binary starts with magic number followed by version
    unsigned int* wrapper_magic = (unsigned int*)fatCubin;
    fprintf(stderr, "[cudart_shim] Fat binary wrapper magic: 0x%08x\n", wrapper_magic[0]);

    // Parse FatbincWrapper:
    // struct FatbincWrapper {
    //     unsigned int magic;      // 0x466243B1 (offset 0, 4 bytes)
    //     unsigned int version;    // 1 or 2    (offset 4, 4 bytes)
    //     void* data;              // pointer to FatbinHeader (offset 8, 8 bytes)
    //     void* filename_or_fatbins;           (offset 16, 8 bytes)
    // }
    // So data pointer is at offset 8
    void* fatbin_header_ptr = NULL;

    if (wrapper_magic[0] == 0x466243B1) {
        // Read the data pointer which is at offset 8
        void** data_ptr_location = (void**)((char*)fatCubin + 8);
        fatbin_header_ptr = *data_ptr_location;
    } else {
        // Fallback: assume data is offset by 16 bytes
        fatbin_header_ptr = (char*)fatCubin + 16;
    }

    fprintf(stderr, "[cudart_shim] FatbinHeader pointer: %p\n", fatbin_header_ptr);

    // Parse FatbinHeader to find the actual CUBIN/PTX payload
    FatbinHeader* fatbin_header = (FatbinHeader*)fatbin_header_ptr;
    void* payload = NULL;
    size_t payload_size = 0;
    int payload_needs_free = 0;

    if (fatbin_header && fatbin_header->magic == FATBIN_MAGIC) {
        fprintf(stderr, "[cudart_shim] Valid FatbinHeader found (magic: 0x%08x)\n", fatbin_header->magic);
        fprintf(stderr, "[cudart_shim] Header size: %u, Files size: %lu\n",
                fatbin_header->header_size, fatbin_header->files_size);

        // Start of file headers
        unsigned char* file_ptr = (unsigned char*)fatbin_header + fatbin_header->header_size;
        unsigned char* end_ptr = file_ptr + fatbin_header->files_size;

        // Iterate through file headers to find PTX or CUBIN
        while (file_ptr < end_ptr) {
            FatbinFileHeader* file_header = (FatbinFileHeader*)file_ptr;

            fprintf(stderr, "[cudart_shim] File entry: kind=0x%04x, header_size=%u, payload_size=%u, padded=%u, sm=%u, uncompressed=%lu\n",
                    file_header->kind, file_header->header_size,
                    file_header->payload_size, file_header->padded_payload_size,
                    file_header->sm_version, file_header->uncompressed_payload);

            // Prefer PTX over CUBIN
            if (file_header->kind == FATBIN_KIND_PTX && payload == NULL) {
                unsigned char* ptx_payload = file_ptr + file_header->header_size;
                size_t raw_size = file_header->payload_size;
                size_t uncompressed_size = file_header->uncompressed_payload;
                fprintf(stderr, "[cudart_shim] Found PTX payload at offset +%lu, compressed_size=%zu, uncompressed_size=%zu, sm=%u\n",
                        (unsigned long)((char*)ptx_payload - (char*)fatbin_header_ptr),
                        raw_size, uncompressed_size, file_header->sm_version);

                if (uncompressed_size > 0) {
                    // PTX payload is LZ4-compressed - decompress it
                    fprintf(stderr, "[cudart_shim] PTX is LZ4-compressed, decompressing %zu -> %zu bytes\n",
                            raw_size, uncompressed_size);

                    char* decompressed = (char*)malloc(uncompressed_size + 1);
                    if (decompressed) {
                        int result = hetgpu_lz4_decompress(
                            (const char*)ptx_payload,
                            decompressed,
                            (int)raw_size,
                            (int)uncompressed_size
                        );

                        if (result > 0) {
                            decompressed[result] = '\0';
                            fprintf(stderr, "[cudart_shim] LZ4 decompression successful: %d bytes\n", result);

                            payload = decompressed;
                            payload_size = (size_t)result;
                            payload_needs_free = 1;
                        } else {
                            fprintf(stderr, "[cudart_shim] LZ4 decompression FAILED (result=%d), trying raw\n", result);
                            free(decompressed);
                            // Fall through to try raw
                            payload = ptx_payload;
                            payload_size = raw_size;
                        }
                    } else {
                        fprintf(stderr, "[cudart_shim] Failed to allocate %zu bytes for decompression\n", uncompressed_size);
                        payload = ptx_payload;
                        payload_size = raw_size;
                    }
                } else {
                    // Uncompressed PTX - use directly
                    fprintf(stderr, "[cudart_shim] PTX payload is uncompressed, using directly\n");
                    payload = ptx_payload;
                    payload_size = raw_size;
                }
                break;  // Prefer PTX, so break immediately
            } else if (file_header->kind == FATBIN_KIND_ELF && payload == NULL) {
                unsigned char* raw_payload = file_ptr + file_header->header_size;

                // Debug: show first 20 bytes starting at different offsets to find ELF magic
                fprintf(stderr, "[cudart_shim] Looking for ELF magic (7f 45 4c 46):\n");
                for (int off = 0; off < 4; off++) {
                    fprintf(stderr, "[cudart_shim]   offset %d: ", off);
                    for (int i = 0; i < 8; i++) {
                        fprintf(stderr, "%02x ", raw_payload[off + i]);
                    }
                    fprintf(stderr, "\n");
                }

                // Check if ELF magic is at offset 1 (skip alignment byte)
                if (raw_payload[1] == 0x7f && raw_payload[2] == 'E' &&
                    raw_payload[3] == 'L' && raw_payload[4] == 'F') {
                    fprintf(stderr, "[cudart_shim] Found ELF magic at offset 1, adjusting payload pointer\n");
                    payload = raw_payload + 1;
                    payload_size = file_header->payload_size - 1;
                } else if (raw_payload[0] == 0x7f && raw_payload[1] == 'E' &&
                           raw_payload[2] == 'L' && raw_payload[3] == 'F') {
                    // ELF magic at expected position
                    payload = raw_payload;
                    payload_size = file_header->payload_size;
                } else {
                    // No ELF magic found, use raw payload
                    fprintf(stderr, "[cudart_shim] WARNING: No ELF magic found, using raw payload\n");
                    payload = raw_payload;
                    payload_size = file_header->payload_size;
                }

                fprintf(stderr, "[cudart_shim] Found ELF/CUBIN payload at offset +%lu, size=%zu\n",
                        (unsigned long)((char*)payload - (char*)fatbin_header_ptr), payload_size);
                // Don't break - keep looking for PTX
            }

            // Move to next file entry
            file_ptr += file_header->padded_payload_size + file_header->header_size;
        }
    } else {
        fprintf(stderr, "[cudart_shim] Invalid or missing FatbinHeader, using data pointer directly\n");
        payload = fatbin_header_ptr;
    }

    // If we didn't find a payload, use the data pointer directly as fallback
    if (payload == NULL) {
        fprintf(stderr, "[cudart_shim] No valid payload found in fat binary, using data pointer as fallback\n");
        payload = fatbin_header_ptr;
    }

    // If we found a CUBIN (not PTX), try to extract PTX using cuobjdump
    // This is necessary because many PyTorch modules only include CUBIN, not PTX
    if (payload != NULL && payload_size > 0) {
        // Check if this is likely a CUBIN (starts with 0x7fELF or looks binary)
        unsigned char* payload_bytes = (unsigned char*)payload;
        int is_binary = (payload_bytes[0] == 0x7f || payload_bytes[0] > 127 ||
                         (payload_bytes[0] < 32 && payload_bytes[0] != '\n'));

        if (is_binary) {
            fprintf(stderr, "[cudart_shim] Detected binary CUBIN, attempting PTX extraction from .so file...\n");
            fprintf(stderr, "[cudart_shim] Payload first 16 bytes: ");
            for (size_t i = 0; i < 16 && i < payload_size; i++) {
                fprintf(stderr, "%02x ", payload_bytes[i]);
            }
            fprintf(stderr, "\n");

            // NEW APPROACH: Find the source .so file from the fatbin pointer address
            // and extract PTX from the full .so using cuobjdump
            char so_path[512] = {0};
            if (find_so_from_address(fatbin_header_ptr, so_path, sizeof(so_path))) {
                fprintf(stderr, "[cudart_shim] Found source .so: %s\n", so_path);

                // Extract PTX from the .so file (or use cached result)
                const char* ptx_dir = extract_ptx_from_so(so_path);
                if (ptx_dir) {
                    // Load a single PTX file (round-robin through available files)
                    // This is MUCH faster than concatenating all 399+ PTX files
                    size_t ptx_size = 0;
                    char* ptx_data = find_matching_ptx(ptx_dir, NULL, &ptx_size);
                    if (ptx_data && ptx_size > 50) {
                        // Verify it looks like PTX
                        if (strstr(ptx_data, ".version") && strstr(ptx_data, ".target")) {
                            fprintf(stderr, "[cudart_shim] Successfully loaded %zu bytes of PTX from .so\n", ptx_size);
                            fprintf(stderr, "[cudart_shim] PTX preview: %.200s...\n", ptx_data);
                            payload = ptx_data;
                            payload_size = ptx_size;
                            payload_needs_free = 1;
                        } else {
                            fprintf(stderr, "[cudart_shim] Extracted PTX doesn't look valid\n");
                            free(ptx_data);
                        }
                    } else if (ptx_data) {
                        fprintf(stderr, "[cudart_shim] PTX too small: %zu bytes\n", ptx_size);
                        free(ptx_data);
                    }
                }
            } else {
                fprintf(stderr, "[cudart_shim] Could not find source .so for address %p\n", fatbin_header_ptr);
                // Fallback: try the old approach with the fatbin directly
                char tmpfile_cubin[256];
                snprintf(tmpfile_cubin, sizeof(tmpfile_cubin), "/tmp/hetgpu_fatbin_%p.fatbin", fatCubin);
                FILE* f = fopen(tmpfile_cubin, "wb");
                if (f) {
                    size_t fatbin_total_size = fatbin_header->header_size + fatbin_header->files_size;
                    fwrite(fatbin_header, 1, fatbin_total_size, f);
                    fclose(f);
                    fprintf(stderr, "[cudart_shim] Wrote fatbin to %s for manual inspection\n", tmpfile_cubin);
                }
            }
        }
    }

    // Try to load the payload as a module
    CUmodule module = NULL;
    CUresult result = cuModuleLoadData(&module, payload);

    // Free decompressed PTX buffer now that cuModuleLoadData has consumed it
    if (payload_needs_free && payload) {
        free(payload);
        payload = NULL;
        payload_needs_free = 0;
    }

    if (result != 0) {
        fprintf(stderr, "[cudart_shim] cuModuleLoadData failed: %d\n", result);
        fprintf(stderr, "[cudart_shim] Module load failed, but continuing with placeholder\n");
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

// Get a dlopen handle to our own .so (libnvcuda.so) so we can resolve
// cu* driver symbols from OUR library, not the system's libcuda.so.1.
static void* get_self_library_handle(void) {
    static void* self_handle = NULL;
    static int tried = 0;
    if (!tried) {
        tried = 1;
        Dl_info info;
        if (dladdr((void*)cudaGetDriverEntryPointByVersion, &info) && info.dli_fname) {
            self_handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW);
            if (self_handle) {
                fprintf(stderr, "[cudart_shim] Self library: %s (handle=%p)\n",
                        info.dli_fname, self_handle);
            }
        }
    }
    return self_handle;
}

cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol,
                                             void** funcPtr,
                                             int driverVersion,
                                             unsigned long long flags) {
    (void)driverVersion; (void)flags;
    if (!funcPtr) return 0;

    // Look up the requested CUDA driver symbol in OUR library (libnvcuda.so).
    // PyTorch 2.9+ uses this to get function pointers for cuDevicePrimaryCtxRetain,
    // cuCtxSetCurrent, etc. at runtime rather than linking directly.
    // We must NOT use RTLD_DEFAULT because that would find the system's
    // /lib/x86_64-linux-gnu/libcuda.so.1 which tries to talk to a real GPU driver.
    void* handle = get_self_library_handle();
    void* ptr = NULL;
    if (handle) {
        ptr = dlsym(handle, symbol);
    }
    if (!ptr) {
        // Fallback to RTLD_DEFAULT if self-lookup fails
        ptr = dlsym(RTLD_DEFAULT, symbol);
    }
    if (ptr) {
        *funcPtr = ptr;
        fprintf(stderr, "[cudart_shim] cudaGetDriverEntryPoint('%s') -> %p\n", symbol, ptr);
    } else {
        *funcPtr = NULL;
        fprintf(stderr, "[cudart_shim] cudaGetDriverEntryPoint('%s') -> NOT FOUND\n", symbol);
    }
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

// Batch memory copy API (CUDA 12.x)
// This is a batched version of cudaMemcpyAsync that copies multiple regions in one call
typedef struct {
    void* dst;
    const void* src;
    size_t count;
} cudaMemcpyBatchOp;

cudaError_t cudaMemcpyBatchAsync(void* opList, size_t numOps, cudaStream_t stream) {
    (void)stream;
    if (!opList || numOps == 0) return 0;

    // Each operation in the batch is a memcpy
    cudaMemcpyBatchOp* ops = (cudaMemcpyBatchOp*)opList;
    for (size_t i = 0; i < numOps; i++) {
        if (ops[i].dst && ops[i].src && ops[i].count > 0) {
            memcpy(ops[i].dst, ops[i].src, ops[i].count);
        }
    }
    return 0;
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

// NVRTC implementation - compiles CUDA source to PTX via nvcc
// nvrtcResult enum values
#define NVRTC_SUCCESS 0
#define NVRTC_ERROR_OUT_OF_MEMORY 1
#define NVRTC_ERROR_PROGRAM_CREATION_FAILURE 2
#define NVRTC_ERROR_INVALID_INPUT 3
#define NVRTC_ERROR_INVALID_PROGRAM 4
#define NVRTC_ERROR_COMPILATION 6

typedef struct nvrtc_program_st {
    char* source;           // CUDA source code
    char* name;             // Program name
    char* ptx;              // Compiled PTX output
    size_t ptx_size;        // Size of PTX (including null terminator)
    char* log;              // Compilation log
    size_t log_size;        // Size of log (including null terminator)
    char** name_expressions; // Added name expressions
    int num_name_expressions;
} nvrtc_program_st;

typedef nvrtc_program_st* nvrtcProgram;

int nvrtcVersion(int* major, int* minor) {
    if (major) *major = 12;
    if (minor) *minor = 8;
    return NVRTC_SUCCESS;
}

int nvrtcCreateProgram(nvrtcProgram* prog, const char* src,
                       const char* name, int numHeaders,
                       const char* const* headers,
                       const char* const* includeNames) {
    if (!prog || !src) return NVRTC_ERROR_INVALID_INPUT;

    nvrtc_program_st* p = (nvrtc_program_st*)calloc(1, sizeof(nvrtc_program_st));
    if (!p) return NVRTC_ERROR_OUT_OF_MEMORY;

    p->source = strdup(src);
    p->name = name ? strdup(name) : strdup("default_program");
    p->ptx = NULL;
    p->ptx_size = 0;
    p->log = strdup("");
    p->log_size = 1;
    p->name_expressions = NULL;
    p->num_name_expressions = 0;

    *prog = p;
    return NVRTC_SUCCESS;
}

int nvrtcDestroyProgram(nvrtcProgram* prog) {
    if (!prog || !*prog) return NVRTC_SUCCESS;
    nvrtc_program_st* p = *prog;
    free(p->source);
    free(p->name);
    free(p->ptx);
    free(p->log);
    if (p->name_expressions) {
        for (int i = 0; i < p->num_name_expressions; i++)
            free(p->name_expressions[i]);
        free(p->name_expressions);
    }
    free(p);
    *prog = NULL;
    return NVRTC_SUCCESS;
}

int nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char* const* options) {
    if (!prog || !prog->source) return NVRTC_ERROR_INVALID_PROGRAM;

    // Write source to temp file
    char src_path[256];
    snprintf(src_path, sizeof(src_path), "/tmp/nvrtc_%d_%p.cu", getpid(), (void*)prog);
    char ptx_path[256];
    snprintf(ptx_path, sizeof(ptx_path), "/tmp/nvrtc_%d_%p.ptx", getpid(), (void*)prog);

    FILE* f = fopen(src_path, "w");
    if (!f) {
        fprintf(stderr, "[nvrtc] Failed to write source to %s\n", src_path);
        return NVRTC_ERROR_COMPILATION;
    }
    // Preprocess source: nvrtc doesn't include system headers but nvcc does,
    // so typedefs like "typedef long long int int64_t;" conflict with <stdint.h>.
    // Wrap them with guards.
    fputs("#include <stdint.h>\n", f);
    const char* src = prog->source;
    const char* line_start = src;
    while (*line_start) {
        const char* line_end = strchr(line_start, '\n');
        if (!line_end) line_end = line_start + strlen(line_start);
        size_t line_len = line_end - line_start;

        // Skip typedef lines that redefine standard integer types
        // (nvrtc doesn't include system headers, but nvcc does)
        int skip = 0;
        const char* trimmed = line_start;
        while (trimmed < line_start + line_len && (*trimmed == ' ' || *trimmed == '\t'))
            trimmed++;
        size_t trimmed_len = line_len - (trimmed - line_start);
        if (trimmed_len > 7 && strncmp(trimmed, "typedef", 7) == 0) {
            char line_buf[512];
            size_t copy_len = line_len < sizeof(line_buf)-1 ? line_len : sizeof(line_buf)-1;
            memcpy(line_buf, line_start, copy_len);
            line_buf[copy_len] = '\0';
            if (strstr(line_buf, "int64_t") ||
                strstr(line_buf, "uint32_t") ||
                strstr(line_buf, "int8_t") ||
                strstr(line_buf, "uint8_t") ||
                strstr(line_buf, "int16_t"))
                skip = 1;
        }

        if (!skip) {
            fwrite(line_start, 1, line_len, f);
            fputc('\n', f);
        } else {
            fputs("// [nvrtc-shim] skipped conflicting typedef\n", f);
        }

        line_start = (*line_end == '\n') ? line_end + 1 : line_end;
    }
    fclose(f);

    // Build nvcc command - translate nvrtc options to nvcc options
    char cmd[4096];
    char arch_flag[64] = "--gpu-architecture=sm_80";
    char extra_opts[2048] = "";
    size_t extra_len = 0;
    int has_device_default = 0;

    for (int i = 0; i < numOptions; i++) {
        if (!options[i]) continue;

        // Architecture flag
        if (strncmp(options[i], "--gpu-architecture", 18) == 0 ||
            strncmp(options[i], "-arch", 5) == 0) {
            snprintf(arch_flag, sizeof(arch_flag), "%s", options[i]);
        }
        // nvrtc's -default-device -> nvcc's --device-as-default-execution-space
        else if (strncmp(options[i], "-default-device", 15) == 0) {
            has_device_default = 1;
        }
        else if (strncmp(options[i], "--device-as-default-execution-space", 35) == 0) {
            has_device_default = 1;
        }
        // Options safe to pass through to nvcc
        else if (strncmp(options[i], "--std=", 6) == 0 ||
                   strncmp(options[i], "-std=", 5) == 0 ||
                   strncmp(options[i], "-D", 2) == 0 ||
                   strncmp(options[i], "--define-macro", 14) == 0 ||
                   strncmp(options[i], "-I", 2) == 0 ||
                   strncmp(options[i], "--include-path", 14) == 0 ||
                   strncmp(options[i], "--pre-include", 13) == 0 ||
                   strncmp(options[i], "--use_fast_math", 15) == 0 ||
                   strncmp(options[i], "-use_fast_math", 14) == 0 ||
                   strncmp(options[i], "--fmad", 6) == 0 ||
                   strncmp(options[i], "--extra-device-vectorization", 28) == 0) {
            int n = snprintf(extra_opts + extra_len, sizeof(extra_opts) - extra_len,
                           " %s", options[i]);
            if (n > 0) extra_len += n;
        }
        // Skip nvrtc-specific options that nvcc doesn't understand
        // e.g., -rdc, --extensible-whole-program, etc.
    }

    // Note: nvrtc's -default-device is not needed for nvcc since
    // the jiterator source has explicit __global__/__device__ annotations
    (void)has_device_default;

    snprintf(cmd, sizeof(cmd),
             "nvcc --ptx %s %s -o %s %s 2>&1",
             arch_flag, extra_opts, ptx_path, src_path);

    fprintf(stderr, "[nvrtc] Compiling: %s\n", cmd);

    // Run nvcc
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        unlink(src_path);
        return NVRTC_ERROR_COMPILATION;
    }

    // Capture log
    char log_buf[8192] = "";
    size_t log_len = 0;
    char line[512];
    while (fgets(line, sizeof(line), pipe)) {
        size_t ll = strlen(line);
        if (log_len + ll < sizeof(log_buf) - 1) {
            memcpy(log_buf + log_len, line, ll);
            log_len += ll;
        }
    }
    log_buf[log_len] = '\0';
    int ret = pclose(pipe);

    // Update log
    free(prog->log);
    prog->log = strdup(log_buf);
    prog->log_size = strlen(log_buf) + 1;

    if (ret != 0) {
        fprintf(stderr, "[nvrtc] Compilation failed (exit %d):\n%s\n", WEXITSTATUS(ret), log_buf);
        unlink(src_path);
        unlink(ptx_path);
        return NVRTC_ERROR_COMPILATION;
    }

    // Read compiled PTX
    FILE* ptx_file = fopen(ptx_path, "r");
    if (!ptx_file) {
        fprintf(stderr, "[nvrtc] Failed to read PTX from %s\n", ptx_path);
        unlink(src_path);
        return NVRTC_ERROR_COMPILATION;
    }

    fseek(ptx_file, 0, SEEK_END);
    long ptx_len = ftell(ptx_file);
    fseek(ptx_file, 0, SEEK_SET);

    free(prog->ptx);
    prog->ptx = (char*)malloc(ptx_len + 1);
    if (!prog->ptx) {
        fclose(ptx_file);
        unlink(src_path);
        unlink(ptx_path);
        return NVRTC_ERROR_OUT_OF_MEMORY;
    }

    size_t read_len = fread(prog->ptx, 1, ptx_len, ptx_file);
    prog->ptx[read_len] = '\0';
    prog->ptx_size = read_len + 1;
    fclose(ptx_file);

    fprintf(stderr, "[nvrtc] Compilation success: %zu bytes PTX\n", read_len);

    // Cleanup temp files
    unlink(src_path);
    unlink(ptx_path);

    return NVRTC_SUCCESS;
}

int nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) {
    if (!prog) return NVRTC_ERROR_INVALID_PROGRAM;
    if (ptxSizeRet) *ptxSizeRet = prog->ptx_size ? prog->ptx_size : 1;
    return NVRTC_SUCCESS;
}

int nvrtcGetPTX(nvrtcProgram prog, char* ptx) {
    if (!prog) return NVRTC_ERROR_INVALID_PROGRAM;
    if (ptx) {
        if (prog->ptx && prog->ptx_size > 0) {
            memcpy(ptx, prog->ptx, prog->ptx_size);
        } else {
            ptx[0] = '\0';
        }
    }
    return NVRTC_SUCCESS;
}

int nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) {
    if (cubinSizeRet) *cubinSizeRet = 0;
    return NVRTC_SUCCESS;
}

int nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) {
    return NVRTC_SUCCESS;
}

int nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) {
    if (!prog) return NVRTC_ERROR_INVALID_PROGRAM;
    if (logSizeRet) *logSizeRet = prog->log_size ? prog->log_size : 1;
    return NVRTC_SUCCESS;
}

int nvrtcGetProgramLog(nvrtcProgram prog, char* log) {
    if (!prog) return NVRTC_ERROR_INVALID_PROGRAM;
    if (log) {
        if (prog->log && prog->log_size > 0) {
            memcpy(log, prog->log, prog->log_size);
        } else {
            log[0] = '\0';
        }
    }
    return NVRTC_SUCCESS;
}

const char* nvrtcGetErrorString(int result) {
    switch (result) {
        case NVRTC_SUCCESS: return "NVRTC_SUCCESS";
        case NVRTC_ERROR_OUT_OF_MEMORY: return "NVRTC_ERROR_OUT_OF_MEMORY";
        case NVRTC_ERROR_PROGRAM_CREATION_FAILURE: return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
        case NVRTC_ERROR_INVALID_INPUT: return "NVRTC_ERROR_INVALID_INPUT";
        case NVRTC_ERROR_INVALID_PROGRAM: return "NVRTC_ERROR_INVALID_PROGRAM";
        case NVRTC_ERROR_COMPILATION: return "NVRTC_ERROR_COMPILATION";
        default: return "NVRTC_ERROR_UNKNOWN";
    }
}

int nvrtcAddNameExpression(nvrtcProgram prog, const char* nameExpression) {
    if (!prog || !nameExpression) return NVRTC_ERROR_INVALID_INPUT;
    prog->name_expressions = (char**)realloc(prog->name_expressions,
        (prog->num_name_expressions + 1) * sizeof(char*));
    if (!prog->name_expressions) return NVRTC_ERROR_OUT_OF_MEMORY;
    prog->name_expressions[prog->num_name_expressions] = strdup(nameExpression);
    prog->num_name_expressions++;
    return NVRTC_SUCCESS;
}

int nvrtcGetLoweredName(nvrtcProgram prog, const char* nameExpression,
                        const char** loweredName) {
    if (!prog || !nameExpression || !loweredName) return NVRTC_ERROR_INVALID_INPUT;
    // For now return the expression as-is (no name mangling)
    *loweredName = nameExpression;
    return NVRTC_SUCCESS;
}

int nvrtcGetNumSupportedArchs(int* numArchs) {
    if (numArchs) *numArchs = 1;
    return NVRTC_SUCCESS;
}

int nvrtcGetSupportedArchs(int* supportedArchs) {
    if (supportedArchs) supportedArchs[0] = 80;
    return NVRTC_SUCCESS;
}

int nvrtcGetLTOIRSize(nvrtcProgram prog, size_t* ltoSizeRet) {
    if (ltoSizeRet) *ltoSizeRet = 0;
    return NVRTC_SUCCESS;
}

int nvrtcGetLTOIR(nvrtcProgram prog, char* lto) {
    return NVRTC_SUCCESS;
}

int nvrtcGetNVVMSize(nvrtcProgram prog, size_t* nvvmSizeRet) {
    if (nvvmSizeRet) *nvvmSizeRet = 0;
    return NVRTC_SUCCESS;
}

int nvrtcGetNVVM(nvrtcProgram prog, char* nvvm) {
    return NVRTC_SUCCESS;
}

int nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t* sizeRet) {
    if (sizeRet) *sizeRet = 0;
    return NVRTC_SUCCESS;
}

int nvrtcGetOptiXIR(nvrtcProgram prog, char* ir) {
    return NVRTC_SUCCESS;
}

int nvrtcSetFlowCallback(nvrtcProgram prog, void* callback, void* payload) {
    return NVRTC_SUCCESS;
}

int nvrtcGetPCHCreateStatus(nvrtcProgram prog) {
    return NVRTC_SUCCESS;
}

int nvrtcGetPCHHeapSize(size_t* size) {
    if (size) *size = 0;
    return NVRTC_SUCCESS;
}

int nvrtcGetPCHHeapSizeRequired(size_t* size) {
    if (size) *size = 0;
    return NVRTC_SUCCESS;
}

int nvrtcSetPCHHeapSize(size_t size) {
    (void)size;
    return NVRTC_SUCCESS;
}
