#define _GNU_SOURCE
#define _XOPEN_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
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

    // Advance past the faulting divide instruction where the platform ABI lets
    // us edit the saved program counter. This keeps the virtual backend
    // permissive during framework probes that can otherwise SIGFPE.
    ucontext_t *uc = (ucontext_t *)ucontext_raw;
#if defined(__x86_64__)
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
#elif defined(__riscv) && defined(REG_PC)
    uintptr_t pc = (uintptr_t)uc->uc_mcontext.__gregs[REG_PC];
    uint16_t insn16 = 0;
    memcpy(&insn16, (const void *)pc, sizeof(insn16));
    uc->uc_mcontext.__gregs[REG_PC] = pc + ((insn16 & 0x3) == 0x3 ? 4 : 2);
#if defined(REG_A0)
    uc->uc_mcontext.__gregs[REG_A0] = 0;
#endif
#else
    (void)uc;
    signal(SIGFPE, SIG_DFL);
#endif
}

__attribute__((constructor))
static void install_sigfpe_handler(void) {
    struct sigaction sa;
    sa.sa_sigaction = sigfpe_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGFPE, &sa, NULL);
    const char* log_sigfpe = getenv("HETGPU_CUDART_LOG_SIGFPE");
    if (log_sigfpe && strcmp(log_sigfpe, "1") == 0) {
        fprintf(stderr, "[hetGPU] SIGFPE handler installed (non-fatal)\n");
    }
}

static int hetgpu_cudart_debug_logs_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("HETGPU_CUDART_DEBUG_LOGS");
        if (!env || !env[0]) {
            env = getenv("HETGPU_DEBUG_LOGS");
        }
        enabled = env && strcmp(env, "1") == 0;
    }
    return enabled;
}

static void hetgpu_cuda_malloc_trace(const char *tag) {
    const char *env = getenv("HETGPU_CUDART_MALLOC_TRACE");
    if (env && env[0] == '1' && tag) {
        write(STDERR_FILENO, tag, strlen(tag));
        write(STDERR_FILENO, "\n", 1);
    }
}

#if defined(HETGPU_DEBUG_LOGS)
#define HETGPU_LOG(...) do { if (hetgpu_cudart_debug_logs_enabled()) fprintf(stderr, __VA_ARGS__); } while (0)
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

enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
};

struct cudaPointerAttributes {
    enum cudaMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
};

#define HETGPU_CUDA_SUCCESS 0
#define HETGPU_CUDA_ERROR_INVALID_VALUE 1
#define HETGPU_CUDA_ERROR_UNKNOWN 999

static cudaError_t g_last_cuda_error = HETGPU_CUDA_SUCCESS;

static int hetgpu_strict_pacc(void) {
    const char* strict = getenv("HETGPU_PACC_STRICT");
    return strict && strcmp(strict, "1") == 0;
}

static int hetgpu_pacc_requires_tracked_allocations(void) {
    const char* shared_mem = getenv("HETGPU_PACC_SHARED_DEVICE_MEM");
    const char* kernel_submit = getenv("HETGPU_PACC_KERNEL_MBOX_SUBMIT");
    const char* control_backend = getenv("HETGPU_PACC_CONTROL_BACKEND");
    return (shared_mem && strcmp(shared_mem, "1") == 0) ||
           (kernel_submit && strcmp(kernel_submit, "1") == 0) ||
           (control_backend && strcmp(control_backend, "shared_ddr") == 0);
}

static int hetgpu_allow_skip_null_registered_kernel(void) {
    const char* allow = getenv("HETGPU_CUDART_ALLOW_SKIP_NULL_FUNCTIONS");
    return allow && strcmp(allow, "1") == 0;
}

static int hetgpu_env_enabled_default(const char *name, int default_value) {
    const char *value = getenv(name);
    if (!value || !*value) {
        return default_value;
    }
    if (strcmp(value, "0") == 0 ||
        strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "no") == 0 ||
        strcasecmp(value, "off") == 0) {
        return 0;
    }
    return 1;
}

static int hetgpu_cudart_fail_open_enabled(void) {
    const char *value = getenv("HETGPU_CUDART_FAIL_OPEN");
    if (!value || !*value) {
        value = getenv("HETGPU_PACC_ASSUME_SUCCESS_ON_WAIT_ERROR");
    }
    if (!value || !*value) {
        return 1;
    }
    if (strcmp(value, "0") == 0 ||
        strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "no") == 0 ||
        strcasecmp(value, "off") == 0) {
        return 0;
    }
    return 1;
}

static int hetgpu_cudart_lazy_ptx_fail_open_enabled(void) {
    return hetgpu_env_enabled_default("HETGPU_CUDART_LAZY_PTX_FAIL_OPEN", 0);
}

static unsigned long long hetgpu_parse_env_ull_default(const char *name, unsigned long long default_value);

static unsigned long long hetgpu_cudart_lazy_ptx_fail_open_log_limit(void) {
    static unsigned long long cached = (unsigned long long)-1;
    if (cached == (unsigned long long)-1) {
        cached = hetgpu_parse_env_ull_default("HETGPU_CUDART_LAZY_PTX_FAIL_OPEN_LOG_LIMIT", 64);
    }
    return cached;
}

static int hetgpu_cudart_kernel_noop_enabled(void) {
    return hetgpu_env_enabled_default("HETGPU_CUDART_KERNEL_NOOP", 0);
}

static int hetgpu_cudart_kernel_noop_log_enabled(void) {
    return hetgpu_env_enabled_default("HETGPU_CUDART_KERNEL_NOOP_LOG", 0);
}

static int hetgpu_cudart_kernel_pacc_noop_enabled(void) {
    return hetgpu_env_enabled_default("HETGPU_CUDART_KERNEL_PACC_NOOP", 0);
}

static unsigned long long hetgpu_parse_env_ull_default(const char *name, unsigned long long default_value) {
    const char *value = getenv(name);
    if (!value || !*value) {
        return default_value;
    }
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 0);
    if (end == value) {
        return default_value;
    }
    return parsed;
}

static unsigned long long hetgpu_cudart_kernel_pacc_noop_every(void) {
    static unsigned long long cached = 0;
    if (cached == 0) {
        cached = hetgpu_parse_env_ull_default("HETGPU_CUDART_KERNEL_PACC_NOOP_EVERY", 0);
        if (cached == 0) {
            cached = hetgpu_parse_env_ull_default("HETGPU_PACC_KERNEL_NOOP_EVERY", 1);
        }
        if (cached == 0) {
            cached = 1;
        }
    }
    return cached;
}

static unsigned long long hetgpu_cudart_kernel_pacc_noop_first(void) {
    static unsigned long long cached = (unsigned long long)-1;
    if (cached == (unsigned long long)-1) {
        cached = hetgpu_parse_env_ull_default("HETGPU_CUDART_KERNEL_PACC_NOOP_FIRST", 4);
    }
    return cached;
}

static int hetgpu_cudart_should_submit_pacc_noop(unsigned long long *launch_index_out) {
    static unsigned long long launch_counter = 0;
    unsigned long long launch_index = __sync_add_and_fetch(&launch_counter, 1);
    if (launch_index_out) {
        *launch_index_out = launch_index;
    }
    unsigned long long first = hetgpu_cudart_kernel_pacc_noop_first();
    if (launch_index <= first) {
        return 1;
    }
    unsigned long long every = hetgpu_cudart_kernel_pacc_noop_every();
    return every <= 1 || (launch_index % every) == 0;
}

static int hetgpu_hide_sync_errors(void) {
    const char* hide = getenv("HETGPU_CUDART_HIDE_SYNC_ERRORS");
    return hide && strcmp(hide, "1") == 0;
}

static cudaError_t hetgpu_set_last_error(cudaError_t error) {
    g_last_cuda_error = error;
    return error;
}

extern CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
extern CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
extern int hetgpu_pacc_is_device_ptr(const void* ptr);

typedef int (*hetgpu_pacc_launch_named_kernel_fn)(
    const char* kernel_name,
    unsigned int grid_dim_x,
    unsigned int grid_dim_y,
    unsigned int grid_dim_z,
    unsigned int block_dim_x,
    unsigned int block_dim_y,
    unsigned int block_dim_z,
    unsigned int shared_mem_bytes,
    void* stream,
    void** kernel_params,
    void** extra);

typedef int (*hetgpu_pacc_launch_kernel_noop_fn)(
    unsigned int device_id,
    const char* kernel_name,
    unsigned int grid_dim_x,
    unsigned int grid_dim_y,
    unsigned int grid_dim_z,
    unsigned int block_dim_x,
    unsigned int block_dim_y,
    unsigned int block_dim_z);

static hetgpu_pacc_launch_named_kernel_fn resolve_hetgpu_pacc_launch_named_kernel(void) {
    static hetgpu_pacc_launch_named_kernel_fn cached = NULL;
    static int tried = 0;
    if (!tried) {
        tried = 1;
        cached = (hetgpu_pacc_launch_named_kernel_fn)dlsym(RTLD_DEFAULT, "hetgpu_pacc_launch_named_kernel");
    }
    return cached;
}

static hetgpu_pacc_launch_kernel_noop_fn resolve_hetgpu_pacc_launch_kernel_noop(void) {
    static hetgpu_pacc_launch_kernel_noop_fn cached = NULL;
    static int tried = 0;
    if (!tried) {
        tried = 1;
        cached = (hetgpu_pacc_launch_kernel_noop_fn)dlsym(RTLD_DEFAULT, "hetgpu_pacc_launch_kernel_noop");
    }
    return cached;
}

static cudaError_t hetgpu_cuda_from_cu(CUresult result) {
    return result == 0 ? HETGPU_CUDA_SUCCESS : HETGPU_CUDA_ERROR_UNKNOWN;
}

static int hetgpu_likely_device_ptr(const void* ptr) {
    if (hetgpu_pacc_is_device_ptr(ptr)) {
        return 1;
    }
    uintptr_t value = (uintptr_t)ptr;
    // Real PACC CUDA pointers are PACC-visible physical addresses. Host
    // userspace pointers are high virtual addresses on the target Linux ABI.
    return value >= 0x1000ULL && value < 0x100000000ULL;
}

static cudaError_t hetgpu_cuda_memcpy_d2d(void* dst, const void* src, size_t count) {
    void* tmp = malloc(count);
    if (!tmp) {
        return HETGPU_CUDA_ERROR_UNKNOWN;
    }

    CUresult result = cuMemcpyDtoH_v2(tmp, (CUdeviceptr)src, count);
    if (result == 0) {
        result = cuMemcpyHtoD_v2((CUdeviceptr)dst, tmp, count);
    }
    free(tmp);
    return hetgpu_cuda_from_cu(result);
}

static int hetgpu_pacc_kernel_has_handle(CUfunction func) {
    if (!func) return 0;
    // PaccKernel is #[repr(C)] with `device` followed by `kernel_ptr`.
    // The C runtime shim only needs to know whether Rust created a real
    // pacc_Kernel handle; it does not dereference the device or Rust String.
    void** words = (void**)func;
    return words[1] != NULL;
}

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
extern int hetgpu_pacc_ipc_get_mem_handle(const void* ptr, void* handle, size_t handle_len);
extern int hetgpu_pacc_ipc_open_mem_handle(void** devPtr, const void* handle, unsigned int flags);
extern int hetgpu_pacc_ipc_close_mem_handle(void* devPtr);
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
typedef struct {
    dim3 gridDim;
    dim3 blockDim;
    size_t dynamicSmemBytes;
    cudaStream_t stream;
    void *attrs;
    unsigned int numAttrs;
} cudaLaunchConfig_t;
typedef struct { size_t width, height, depth; } cudaExtent;
typedef struct { size_t x, y, z; } cudaPos;
typedef struct { void *ptr; size_t pitch; size_t xsize; size_t ysize; } cudaPitchedPtr;
typedef int cudaGraphExecUpdateResult;
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* userData);
typedef void (*cudaHostFn_t)(void* userData);
typedef int cudaMemcpyKind; // use int placeholder

// CUDA runtime current device is per host thread. llama.cpp schedules work for
// multiple visible devices from multiple threads, so a process-global current
// device makes workers race and submit kernels to the wrong PACC context.
static __thread int current_device = 0;

#define HETGPU_STREAM_MAGIC UINT64_C(0x485447505354524d)
typedef struct {
    uint64_t magic;
    int device;
    unsigned int flags;
    int priority;
} HetgpuCudaStream;

static int hetgpu_stream_is_managed(cudaStream_t stream) {
    if (!stream) return 0;
    const HetgpuCudaStream* s = (const HetgpuCudaStream*)stream;
    return s->magic == HETGPU_STREAM_MAGIC;
}

static int hetgpu_stream_device(cudaStream_t stream) {
    if (!hetgpu_stream_is_managed(stream)) return current_device;
    const HetgpuCudaStream* s = (const HetgpuCudaStream*)stream;
    return s->device;
}

static cudaError_t hetgpu_stream_create(cudaStream_t* pStream, unsigned int flags, int priority) {
    if (!pStream) return 1;
    HetgpuCudaStream* s = (HetgpuCudaStream*)calloc(1, sizeof(*s));
    if (!s) return 2;
    s->magic = HETGPU_STREAM_MAGIC;
    s->device = current_device;
    s->flags = flags;
    s->priority = priority;
    *pStream = (cudaStream_t)s;
    return 0;
}

typedef CUresult (*hetgpu_cuModuleLoadData_fn)(CUmodule* module, const void* image);
typedef CUresult (*hetgpu_cuModuleGetFunction_fn)(CUfunction* hfunc, CUmodule hmod, const char* name);

static hetgpu_cuModuleLoadData_fn resolve_cuModuleLoadData(void) {
    static hetgpu_cuModuleLoadData_fn fn = NULL;
    static int attempted = 0;
    if (!attempted) {
        attempted = 1;
        dlerror();
        fn = (hetgpu_cuModuleLoadData_fn)dlsym(RTLD_DEFAULT, "cuModuleLoadData");
        if (!fn) {
            const char* err = dlerror();
            fprintf(stderr, "[cudart_shim] dlsym(cuModuleLoadData) failed: %s\n", err ? err : "(null)");
        }
    }
    return fn;
}

static hetgpu_cuModuleGetFunction_fn resolve_cuModuleGetFunction(void) {
    static hetgpu_cuModuleGetFunction_fn fn = NULL;
    static int attempted = 0;
    if (!attempted) {
        attempted = 1;
        dlerror();
        fn = (hetgpu_cuModuleGetFunction_fn)dlsym(RTLD_DEFAULT, "cuModuleGetFunction");
        if (!fn) {
            const char* err = dlerror();
            fprintf(stderr, "[cudart_shim] dlsym(cuModuleGetFunction) failed: %s\n", err ? err : "(null)");
        }
    }
    return fn;
}

typedef struct {
    const void *srcArray;
    cudaPos srcPos;
    cudaPitchedPtr srcPtr;
    const void *dstArray;
    cudaPos dstPos;
    cudaPitchedPtr dstPtr;
    cudaExtent extent;
    cudaMemcpyKind kind;
} cudaMemcpy3DParms;

typedef struct {
    int srcDevice;
    cudaPos srcPos;
    cudaPitchedPtr srcPtr;
    int dstDevice;
    cudaPos dstPos;
    cudaPitchedPtr dstPtr;
    cudaExtent extent;
} cudaMemcpy3DPeerParms;

enum {
    HETGPU_CUDA_MEMCPY_HOST_TO_HOST = 0,
    HETGPU_CUDA_MEMCPY_HOST_TO_DEVICE = 1,
    HETGPU_CUDA_MEMCPY_DEVICE_TO_HOST = 2,
    HETGPU_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3,
    HETGPU_CUDA_MEMCPY_DEFAULT = 4,
};

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

cudaError_t cudaGraphAddNode_v2(cudaGraphNode_t* pGraphNode,
                                cudaGraph_t graph,
                                const cudaGraphNode_t* pDependencies,
                                size_t numDependencies,
                                void* nodeParams) {
    (void)graph;
    (void)pDependencies;
    (void)numDependencies;
    (void)nodeParams;
    if (pGraphNode) {
        *pGraphNode = (cudaGraphNode_t)0;
    }
    return 0;
}

cudaError_t cudaGraphConditionalHandleCreate(void** pHandle,
                                             cudaGraph_t graph,
                                             unsigned int defaultLaunchValue,
                                             unsigned int flags) {
    (void)graph;
    (void)defaultLaunchValue;
    (void)flags;
    if (pHandle) {
        *pHandle = (void*)0;
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

cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream,
                                          cudaGraph_t graph,
                                          const cudaGraphNode_t* dependencies,
                                          const void* dependencyData,
                                          size_t numDependencies,
                                          cudaStreamCaptureMode mode) {
    (void)stream;
    (void)graph;
    (void)dependencies;
    (void)dependencyData;
    (void)numDependencies;
    (void)mode;
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
    return hetgpu_stream_create(pStream, 0, 0);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
    return hetgpu_stream_create(pStream, flags, 0);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (hetgpu_stream_is_managed(stream)) {
        HetgpuCudaStream* s = (HetgpuCudaStream*)stream;
        s->magic = 0;
        free(s);
    }
    return 0;
}

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
    return hetgpu_stream_create(pStream, flags, priority);
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

static int hetgpu_pacc_normalize_device_ordinal(int device) {
    if (device < 0) {
        return 0;
    }
    return device;
}

static int hetgpu_pacc_pci_bus_id(int device) {
    return 0x02 + hetgpu_pacc_normalize_device_ordinal(device);
}

static int hetgpu_pacc_pci_device_id(int device) {
    (void)device;
    return 0;
}

static int hetgpu_pacc_pci_domain_id(int device) {
    (void)device;
    return 0;
}

cudaError_t cudaGetDeviceCount(int* count) {
    if (!count) return 1; // cudaErrorInvalidValue
    const char* log_count = getenv("HETGPU_CUDART_LOG_DEVICE_COUNT");
    if (log_count && strcmp(log_count, "1") == 0) {
        fprintf(stderr, "[hetGPU] cudaGetDeviceCount called\n");
    }
    cuInit(0);
    int result = cuDeviceGetCount(count);
    if (log_count && strcmp(log_count, "1") == 0) {
        fprintf(stderr, "[hetGPU] cudaGetDeviceCount: %d devices\n", count ? *count : -1);
    }
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
    size_t textureAlignment;         // 368-375
    size_t texturePitchAlignment;    // 376-383
    int    deviceOverlap;            // 384-387
    int    multiProcessorCount;      // 388-391
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
    size_t surfaceAlignment;         // 568-575
    int    concurrentKernels;        // 576-579
    int    ECCEnabled;               // 580-583
    int    pciBusID;                 // 584-587
    int    pciDeviceID;              // 588-591
    int    pciDomainID;              // 592-595
    int    tccDriver;                // 596-599
    int    asyncEngineCount;         // 600-603
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
    snprintf(p.name, sizeof(p.name), "Virtual GPU (hetGPU PACC%d sm_80)", device);

    // Memory properties
    p.totalGlobalMem = 4ULL * 1024 * 1024 * 1024;  // 4GB
    p.sharedMemPerBlock = 48 * 1024;               // 48KB
    p.sharedMemPerMultiprocessor = 64 * 1024;      // 64KB
    p.totalConstMem = 64 * 1024;                   // 64KB
    p.memPitch = 2147483647;
    p.textureAlignment = 512;
    p.texturePitchAlignment = 512;
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

    // PCI info (fake but stable): llama.cpp uses this to de-duplicate devices.
    p.pciBusID = hetgpu_pacc_pci_bus_id(device);
    p.pciDeviceID = hetgpu_pacc_pci_device_id(device);
    p.pciDomainID = hetgpu_pacc_pci_domain_id(device);

    // Copy full struct to caller's buffer
    memcpy(prop, &p, sizeof(p));

    const char* log_props = getenv("HETGPU_CUDART_LOG_DEVICE_PROPS");
    if (log_props && strcmp(log_props, "1") == 0) {
        fprintf(stderr, "[cudart_shim] cudaGetDeviceProperties: name='%s' cc=%d.%d (offset major=%zu, minor=%zu)\n",
                p.name, p.major, p.minor,
                offsetof(cudaDeviceProp_full, major),
                offsetof(cudaDeviceProp_full, minor));
    }

    (void)device;
    return 0;
}

// v2 API variant - just calls the base implementation
cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp_t prop, int device) {
    return cudaGetDeviceProperties(prop, device);
}

cudaError_t cudaSetDevice(int device) {
    HETGPU_LOG("[hetGPU] cudaSetDevice(%d) called\n", device);
    // For virtual device support, be permissive
    if (device < 0) {
        HETGPU_LOG("[hetGPU] cudaSetDevice: invalid device\n");
        return 1; // cudaErrorInvalidDevice
    }

    // Get the CUDA device handle
    CUdevice cu_device;
    CUresult result = cuDeviceGet(&cu_device, device);
    HETGPU_LOG("[hetGPU] cudaSetDevice: cuDeviceGet returned %d, cu_device=%d\n", result, cu_device);
    if (result != 0) {
        // For virtual device, still set current_device and succeed
        HETGPU_LOG("[hetGPU] cudaSetDevice(%d): cuDeviceGet failed (%d), continuing with virtual device\n", device, result);
        current_device = device;
        return 0; // Success for virtual device
    }

    // Retain the primary context for this device
    CUcontext ctx;
    result = cuDevicePrimaryCtxRetain(&ctx, cu_device);
    HETGPU_LOG("[hetGPU] cudaSetDevice: cuDevicePrimaryCtxRetain returned %d, ctx=%p\n", result, ctx);
    if (result != 0) {
        // For virtual device, still set current_device and succeed
        HETGPU_LOG("[hetGPU] cudaSetDevice(%d): cuDevicePrimaryCtxRetain failed (%d), continuing with virtual device\n", device, result);
        current_device = device;
        return 0; // Success for virtual device
    }

    // Set it as the current context
    result = cuCtxSetCurrent(ctx);
    HETGPU_LOG("[hetGPU] cudaSetDevice: cuCtxSetCurrent returned %d\n", result);
    if (result != 0) {
        // For virtual device, still set current_device and succeed
        HETGPU_LOG("[hetGPU] cudaSetDevice(%d): cuCtxSetCurrent failed (%d), continuing with virtual device\n", device, result);
        current_device = device;
        return 0; // Success for virtual device
    }

    current_device = device;
    HETGPU_LOG("[hetGPU] cudaSetDevice: success\n");
    return 0;
}

cudaError_t cudaGetDevice(int* device) {
    HETGPU_LOG("[hetGPU] cudaGetDevice called, returning device %d\n", current_device);
    if (device) *device = current_device;
    return 0;
}

cudaError_t cudaRuntimeGetVersion(int* version) {
    if (version) *version = 12080;
    return 0;
}

cudaError_t cudaDeviceSynchronize(void) {
    return hetgpu_hide_sync_errors() ? HETGPU_CUDA_SUCCESS : g_last_cuda_error;
}
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream;
    return hetgpu_hide_sync_errors() ? HETGPU_CUDA_SUCCESS : g_last_cuda_error;
}
cudaError_t cudaStreamQuery(cudaStream_t stream) { (void)stream; return 0; }
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    (void)stream; (void)event; (void)flags; return 0;
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority) {
    if (priority) {
        *priority = hetgpu_stream_is_managed(stream)
            ? ((HetgpuCudaStream*)stream)->priority
            : 0;
    }
    return 0;
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
            *value = hetgpu_pacc_pci_bus_id(device); break;
        case 35: // cudaDevAttrPciDeviceId
            *value = hetgpu_pacc_pci_device_id(device); break;
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
        case 50: // cudaDevAttrPciDomainId
            *value = hetgpu_pacc_pci_domain_id(device); break;
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

    const char* log_attrs = getenv("HETGPU_CUDART_LOG_DEVICE_ATTRS");
    if (log_attrs && strcmp(log_attrs, "1") == 0) {
        fprintf(stderr, "[cudart_shim] cudaDeviceGetAttribute(attr=%d) = %d\n", attr, *value);
    }
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
    // 256-byte alignment keeps ggml CUDA buffer allocators happy
    if (posix_memalign(&ptr, 256, size) != 0) {
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
    if (!pciBusId || len <= 0) return 1;
    snprintf(pciBusId, (size_t)len, "%04x:%02x:%02x.0",
             hetgpu_pacc_pci_domain_id(device),
             hetgpu_pacc_pci_bus_id(device),
             hetgpu_pacc_pci_device_id(device));
    return 0;
}

cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) {
    if (!device || !pciBusId) return 1;
    unsigned int domain = 0, bus = 0, dev = 0, func = 0;
    if (sscanf(pciBusId, "%x:%x:%x.%x", &domain, &bus, &dev, &func) == 4 &&
        domain == 0 && dev == 0 && func == 0 && bus >= 0x02 && bus < 0x06) {
        *device = (int)(bus - 0x02);
        return 0;
    }
    *device = 0;
    return 0;
}

// Pointer attributes
cudaError_t cudaPointerGetAttributes(void* attributes, const void* ptr) {
    if (!attributes) return HETGPU_CUDA_ERROR_INVALID_VALUE;
    struct cudaPointerAttributes* out = (struct cudaPointerAttributes*)attributes;
    memset(out, 0, sizeof(*out));
    if (ptr && hetgpu_likely_device_ptr(ptr)) {
        out->type = cudaMemoryTypeDevice;
        out->device = 0;
        out->devicePointer = (void*)ptr;
        out->hostPointer = NULL;
    } else {
        out->type = cudaMemoryTypeUnregistered;
        out->device = -1;
        out->devicePointer = NULL;
        out->hostPointer = (void*)ptr;
    }
    return HETGPU_CUDA_SUCCESS;
}

// IPC APIs
cudaError_t cudaIpcGetEventHandle(void* handle, cudaEvent_t event) { (void)handle; (void)event; return 0; }
cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, void* handle) { if (event) *event = (cudaEvent_t)0; (void)handle; return 0; }
cudaError_t cudaIpcGetMemHandle(void* handle, void* devPtr) {
    if (!handle || !devPtr) return 1;
    return hetgpu_pacc_ipc_get_mem_handle(devPtr, handle, 64) == 0 ? 0 : 1;
}
cudaError_t cudaIpcOpenMemHandle(void** devPtr, void* handle, unsigned int flags) {
    if (!devPtr || !handle) return 1;
    return hetgpu_pacc_ipc_open_mem_handle(devPtr, handle, flags) == 0 ? 0 : 1;
}
cudaError_t cudaIpcCloseMemHandle(void* devPtr) {
    return hetgpu_pacc_ipc_close_mem_handle(devPtr) == 0 ? 0 : 1;
}

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
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args) {
    if (!config) return HETGPU_CUDA_ERROR_INVALID_VALUE;
    const char* log_launch_ex = getenv("HETGPU_CUDART_LOG_LAUNCH_EX");
    static int launch_ex_log_count = 0;
    if (log_launch_ex && strcmp(log_launch_ex, "1") == 0 && launch_ex_log_count < 64) {
        fprintf(stderr,
                "[cudart_shim] cudaLaunchKernelExC func=%p grid=(%u,%u,%u) block=(%u,%u,%u) shared=%zu stream=%p args=%p\n",
                func,
                config->gridDim.x, config->gridDim.y, config->gridDim.z,
                config->blockDim.x, config->blockDim.y, config->blockDim.z,
                config->dynamicSmemBytes,
                config->stream,
                args);
        launch_ex_log_count++;
    }
    return __cudaLaunchKernel(func,
                              config->gridDim,
                              config->blockDim,
                              args,
                              config->dynamicSmemBytes,
                              config->stream);
}

typedef struct {
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem;
    cudaStream_t stream;
} hetgpu_call_config_t;

#define HETGPU_CALL_CONFIG_STACK_MAX 64
static __thread hetgpu_call_config_t g_call_config_stack[HETGPU_CALL_CONFIG_STACK_MAX];
static __thread int g_call_config_depth = 0;

// CUDA host stubs generated for triple-chevron launches use push/pop to carry
// launch dimensions into cudaLaunchKernel. Preserve that TLS state so kernels
// do not silently collapse to a single thread.
cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    if (g_call_config_depth >= HETGPU_CALL_CONFIG_STACK_MAX) {
        return HETGPU_CUDA_ERROR_INVALID_VALUE;
    }
    g_call_config_stack[g_call_config_depth++] = (hetgpu_call_config_t){
        .grid_dim = gridDim,
        .block_dim = blockDim,
        .shared_mem = sharedMem,
        .stream = stream,
    };
    return HETGPU_CUDA_SUCCESS;
}

cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, cudaStream_t* stream) {
    hetgpu_call_config_t cfg = {
        .grid_dim = {1, 1, 1},
        .block_dim = {1, 1, 1},
        .shared_mem = 0,
        .stream = (cudaStream_t)0,
    };
    if (g_call_config_depth > 0) {
        cfg = g_call_config_stack[--g_call_config_depth];
    }
    if (gridDim) { *gridDim = cfg.grid_dim; }
    if (blockDim) { *blockDim = cfg.block_dim; }
    if (sharedMem) { *sharedMem = cfg.shared_mem; }
    if (stream) { *stream = cfg.stream; }
    return HETGPU_CUDA_SUCCESS;
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
    void* deviceFun;         // Device/stub function pointer alias when provided
    CUfunction cuFunc;       // Driver API function handle
    char name[256];          // Kernel name for debugging
    CUmodule module;         // Parent module
    CUfunction cuFuncByDevice[4];
    CUmodule moduleByDevice[4];
} RegisteredFunction;

static RegisteredModule g_modules[MAX_MODULES];
static int g_module_count = 0;
static RegisteredFunction g_functions[MAX_FUNCTIONS];
static int g_function_count = 0;
static int g_registry_miss_log_count = 0;
static int g_registry_register_log_count = 0;
static int g_null_function_log_count = 0;

#define MAX_CACHED_PTX_MODULES 128
static struct {
    char ptx_path[256];
    int device;
    CUmodule module;
} g_ptx_module_cache[MAX_CACHED_PTX_MODULES];
static int g_ptx_module_cache_count = 0;
static int g_registry_neighbor_log_count = 0;

static CUfunction lazy_load_registered_function_for_launch(const char* kernel_name, const void* launch_func);

static int hetgpu_current_device_index(void) {
    int dev = current_device;
    if (dev < 0 || dev >= 4) {
        return 0;
    }
    return dev;
}

static CUfunction registered_function_current_cufunc(RegisteredFunction* entry) {
    if (!entry) return NULL;
    int dev = hetgpu_current_device_index();
    if (entry->cuFuncByDevice[dev]) {
        return entry->cuFuncByDevice[dev];
    }
    if (dev == 0) {
        return entry->cuFunc;
    }
    return NULL;
}

static void registered_function_set_current_device(RegisteredFunction* entry, CUfunction func, CUmodule module) {
    if (!entry || !func) return;
    int dev = hetgpu_current_device_index();
    entry->cuFuncByDevice[dev] = func;
    entry->moduleByDevice[dev] = module;
    if (dev == 0) {
        entry->cuFunc = func;
        entry->module = module;
    }
}

static uintptr_t hetgpu_registry_alias_window(const Dl_info *func_info) {
    const char *env = getenv("HETGPU_CUDART_REGISTRY_ALIAS_WINDOW");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long value = strtoull(env, &end, 0);
        if (end && *end == '\0' && value > 0) {
            return (uintptr_t)value;
        }
    }
    if (func_info && func_info->dli_fname && strstr(func_info->dli_fname, "libggml-cuda.so.0") != NULL) {
        return 0x400000;
    }
    return 0x1000;
}

static int hetgpu_allow_skip_unregistered_kernel(void) {
    const char* allow = getenv("HETGPU_CUDART_ALLOW_SKIP_UNREGISTERED_KERNELS");
    return allow && strcmp(allow, "1") == 0;
}

static int hetgpu_requires_named_launch(const char* kernel_name) {
    if (!kernel_name || strcmp(kernel_name, "<unknown>") == 0) {
        return 0;
    }
    return strstr(kernel_name, "rms_norm") != NULL ||
           strstr(kernel_name, "rmsnorm") != NULL;
}

static int hetgpu_same_image(const Dl_info *a, const Dl_info *b) {
    if (!a || !b) {
        return 0;
    }
    if (a->dli_fbase && b->dli_fbase && a->dli_fbase == b->dli_fbase) {
        return 1;
    }
    if (a->dli_fname && b->dli_fname && strcmp(a->dli_fname, b->dli_fname) == 0) {
        return 1;
    }
    return 0;
}

static CUfunction lookup_registered_function(const void *func, const char **func_name) {
    uintptr_t func_normalized = (uintptr_t)func & ~(uintptr_t)0x7;

    for (int i = 0; i < g_function_count; i++) {
        uintptr_t registered_host_normalized = (uintptr_t)g_functions[i].hostFun & ~(uintptr_t)0x7;
        uintptr_t registered_device_normalized = (uintptr_t)g_functions[i].deviceFun & ~(uintptr_t)0x7;
        if (g_functions[i].hostFun == func ||
            g_functions[i].deviceFun == func ||
            registered_host_normalized == func_normalized ||
            registered_device_normalized == func_normalized) {
            if (func_name) {
                *func_name = g_functions[i].name;
            }
            CUfunction current_cufunc = registered_function_current_cufunc(&g_functions[i]);
            const char* log_null_cufunc = getenv("HETGPU_CUDART_LOG_NULL_CUFUNC");
            if (current_cufunc == NULL &&
                    log_null_cufunc && strcmp(log_null_cufunc, "1") == 0 &&
                    g_null_function_log_count < 8) {
                fprintf(stderr,
                        "[cudart_shim] exact registry hit has NULL CUfunction for cuda device %d: func=%p name='%s' host=%p device=%p\n",
                        hetgpu_current_device_index(),
                        func,
                        g_functions[i].name,
                        g_functions[i].hostFun,
                        g_functions[i].deviceFun);
                g_null_function_log_count++;
            }
            return current_cufunc;
        }
    }

    // Fallback: some frameworks launch through a nearby wrapper/thunk in the same
    // shared object rather than the exact host/device registration pointer. If we
    // can place the launch site in the same image, use the nearest registered
    // function pointer from that image as a best-effort alias.
    Dl_info func_info;
    memset(&func_info, 0, sizeof(func_info));
    int have_func_info = dladdr(func, &func_info);

    uintptr_t target = (uintptr_t)func;
    if (have_func_info && func_info.dli_saddr && func_info.dli_saddr != func) {
        uintptr_t symbol_normalized = (uintptr_t)func_info.dli_saddr & ~(uintptr_t)0x7;
        for (int i = 0; i < g_function_count; i++) {
            uintptr_t registered_host_normalized = (uintptr_t)g_functions[i].hostFun & ~(uintptr_t)0x7;
            uintptr_t registered_device_normalized = (uintptr_t)g_functions[i].deviceFun & ~(uintptr_t)0x7;
            if (registered_host_normalized == symbol_normalized ||
                registered_device_normalized == symbol_normalized) {
                if (func_name) {
                    *func_name = g_functions[i].name;
                }
                fprintf(stderr,
                        "[cudart_shim] symbol-base matched function %p (%s+0x%lx) to registered '%s'\n",
                        func,
                        func_info.dli_sname ? func_info.dli_sname : "<unknown>",
                        (unsigned long)((uintptr_t)func - (uintptr_t)func_info.dli_saddr),
                        g_functions[i].name);
                return registered_function_current_cufunc(&g_functions[i]);
            }
        }
    }

    uintptr_t best_distance = ~(uintptr_t)0;
    int best_index = -1;
    void *best_candidate = NULL;
    const char *best_candidate_sym = NULL;
    const char *best_candidate_img = NULL;

    for (int i = 0; i < g_function_count; i++) {
        void *candidates[2] = { g_functions[i].hostFun, g_functions[i].deviceFun };
        for (int j = 0; j < 2; ++j) {
            void *candidate = candidates[j];
            if (!candidate) {
                continue;
            }
            if (!have_func_info) {
                continue;
            }
            Dl_info candidate_info;
            if (!dladdr(candidate, &candidate_info) || !hetgpu_same_image(&candidate_info, &func_info)) {
                continue;
            }
            uintptr_t candidate_addr = (uintptr_t)candidate;
            uintptr_t distance = target > candidate_addr ? (target - candidate_addr) : (candidate_addr - target);
            if (distance < best_distance) {
                best_distance = distance;
                best_index = i;
                best_candidate = candidate;
                best_candidate_sym = candidate_info.dli_sname;
                best_candidate_img = candidate_info.dli_fname;
            }
        }
    }

    uintptr_t fallback_limit = hetgpu_registry_alias_window(have_func_info ? &func_info : NULL);

    if (best_index >= 0 && best_distance <= fallback_limit) {
        if (func_name) {
            *func_name = g_functions[best_index].name;
        }
        fprintf(stderr,
                "[cudart_shim] Fallback-matched function %p to registered '%s' via candidate=%p dist=0x%lx image=%s sym=%s\n",
                func,
                g_functions[best_index].name,
                best_candidate,
                (unsigned long)best_distance,
                best_candidate_img ? best_candidate_img : "<unknown>",
                best_candidate_sym ? best_candidate_sym : "<unknown>");
        return registered_function_current_cufunc(&g_functions[best_index]);
    }

    if (best_index >= 0 && g_registry_neighbor_log_count < 24) {
        fprintf(stderr,
                "[cudart_shim] registry nearest-candidate #%d: func=%p best='%s' candidate=%p dist=0x%lx limit=0x%lx image=%s sym=%s\n",
                g_registry_neighbor_log_count + 1,
                func,
                g_functions[best_index].name,
                best_candidate,
                (unsigned long)best_distance,
                (unsigned long)fallback_limit,
                best_candidate_img ? best_candidate_img : "<unknown>",
                best_candidate_sym ? best_candidate_sym : "<unknown>");
        g_registry_neighbor_log_count++;
    }

    {
        uintptr_t raw_best_distance = ~(uintptr_t)0;
        int raw_best_index = -1;
        void *raw_best_candidate = NULL;

        for (int i = 0; i < g_function_count; i++) {
            void *candidates[2] = { g_functions[i].hostFun, g_functions[i].deviceFun };
            for (int j = 0; j < 2; ++j) {
                void *candidate = candidates[j];
                uintptr_t candidate_addr = (uintptr_t)candidate;
                if (candidate_addr < 0x10000) {
                    continue;
                }
                uintptr_t distance = target > candidate_addr ? (target - candidate_addr) : (candidate_addr - target);
                if (distance < raw_best_distance) {
                    raw_best_distance = distance;
                    raw_best_index = i;
                    raw_best_candidate = candidate;
                }
            }
        }

        uintptr_t raw_fallback_limit = fallback_limit;
        if (raw_best_index >= 0 && raw_best_distance <= raw_fallback_limit) {
            if (func_name) {
                *func_name = g_functions[raw_best_index].name;
            }
            fprintf(stderr,
                    "[cudart_shim] Raw-fallback-matched function %p to registered '%s' via candidate=%p dist=0x%lx\n",
                    func,
                    g_functions[raw_best_index].name,
                    raw_best_candidate,
                    (unsigned long)raw_best_distance);
            return registered_function_current_cufunc(&g_functions[raw_best_index]);
        }

        if (raw_best_index >= 0 && g_registry_neighbor_log_count < 24) {
            fprintf(stderr,
                    "[cudart_shim] registry raw-nearest #%d: func=%p best='%s' candidate=%p dist=0x%lx limit=0x%lx\n",
                    g_registry_neighbor_log_count + 1,
                    func,
                    g_functions[raw_best_index].name,
                    raw_best_candidate,
                    (unsigned long)raw_best_distance,
                    (unsigned long)raw_fallback_limit);
            g_registry_neighbor_log_count++;
        } else if (raw_best_index < 0 && g_registry_neighbor_log_count < 24) {
            fprintf(stderr,
                    "[cudart_shim] registry raw-nearest #%d: func=%p no usable raw candidates registered=%d sample0=(%p,%p,'%s') sample1=(%p,%p,'%s')\n",
                    g_registry_neighbor_log_count + 1,
                    func,
                    g_function_count,
                    g_function_count > 0 ? g_functions[0].hostFun : NULL,
                    g_function_count > 0 ? g_functions[0].deviceFun : NULL,
                    g_function_count > 0 ? g_functions[0].name : "<none>",
                    g_function_count > 1 ? g_functions[1].hostFun : NULL,
                    g_function_count > 1 ? g_functions[1].deviceFun : NULL,
                    g_function_count > 1 ? g_functions[1].name : "<none>");
            g_registry_neighbor_log_count++;
        }
    }

    return NULL;
}

static RegisteredFunction* find_registered_function_by_name(const char* kernel_name) {
    if (!kernel_name || strcmp(kernel_name, "<unknown>") == 0) {
        return NULL;
    }
    for (int i = 0; i < g_function_count; ++i) {
        if (strcmp(g_functions[i].name, kernel_name) == 0) {
            return &g_functions[i];
        }
    }
    return NULL;
}

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
        return hetgpu_set_last_error(HETGPU_CUDA_ERROR_INVALID_VALUE);
    }

    int stream_device = hetgpu_stream_device(stream);
    if (stream_device != current_device) {
        (void)cudaSetDevice(stream_device);
    }

    if (hetgpu_cudart_kernel_noop_enabled()) {
        static int kernel_noop_log_count = 0;
        if (hetgpu_cudart_kernel_noop_log_enabled() && kernel_noop_log_count < 20) {
            fprintf(stderr,
                    "[cudart_shim] KERNEL_NOOP active; treating launch func=%p grid=(%u,%u,%u) block=(%u,%u,%u) as successful\n",
                    func,
                    gridDim.x, gridDim.y, gridDim.z,
                    blockDim.x, blockDim.y, blockDim.z);
            kernel_noop_log_count++;
        }
        return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
    }

    // Look up the function in our registration table.
    const char* funcName = "<unknown>";
    CUfunction cuFunc = lookup_registered_function(func, &funcName);
    if (cuFunc != NULL) {
        HETGPU_LOG("[cudart_shim] Found registered function '%s': %p -> %p\n",
                funcName, func, cuFunc);
    }

    if (hetgpu_cudart_kernel_pacc_noop_enabled()) {
        const char* launch_name = funcName;
        char dladdr_name[512];
        dladdr_name[0] = '\0';
        if (!launch_name || strcmp(launch_name, "<unknown>") == 0) {
            Dl_info info;
            if (dladdr(func, &info) && info.dli_sname && info.dli_sname[0]) {
                snprintf(dladdr_name, sizeof(dladdr_name), "%s", info.dli_sname);
                launch_name = dladdr_name;
            } else {
                launch_name = "<unknown>";
            }
        }

        unsigned long long pacc_noop_launch_index = 0;
        if (!hetgpu_cudart_should_submit_pacc_noop(&pacc_noop_launch_index)) {
            static int pacc_noop_skip_log_count = 0;
            if (pacc_noop_skip_log_count < 5) {
                fprintf(stderr,
                        "[cudart_shim] KERNEL_PACC_NOOP sampled out launch #%llu; reporting success without PACC submit (every=%llu)\n",
                        pacc_noop_launch_index,
                        hetgpu_cudart_kernel_pacc_noop_every());
                pacc_noop_skip_log_count++;
            }
            return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
        }

        hetgpu_pacc_launch_kernel_noop_fn launch_noop = resolve_hetgpu_pacc_launch_kernel_noop();
        if (launch_noop) {
            int rc = launch_noop(
                (unsigned int)stream_device,
                launch_name,
                gridDim.x, gridDim.y, gridDim.z,
                blockDim.x, blockDim.y, blockDim.z);
            if (rc == HETGPU_CUDA_SUCCESS) {
                static int pacc_noop_log_count = 0;
                if (pacc_noop_log_count < 20) {
                    fprintf(stderr,
                            "[cudart_shim] KERNEL_PACC_NOOP submitted '%s' to pacc%d grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                            launch_name, stream_device,
                            gridDim.x, gridDim.y, gridDim.z,
                            blockDim.x, blockDim.y, blockDim.z);
                    pacc_noop_log_count++;
                }
                return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
            }
            if (hetgpu_cudart_fail_open_enabled()) {
                fprintf(stderr,
                        "[cudart_shim] KERNEL_PACC_NOOP submit failed for '%s' on pacc%d rc=%d; fail-open success\n",
                        launch_name, stream_device, rc);
                return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
            }
            return hetgpu_set_last_error(HETGPU_CUDA_ERROR_UNKNOWN);
        }
        if (hetgpu_cudart_fail_open_enabled()) {
            fprintf(stderr,
                    "[cudart_shim] KERNEL_PACC_NOOP requested but hetgpu_pacc_launch_kernel_noop is missing; fail-open success\n");
            return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
        }
        return hetgpu_set_last_error(HETGPU_CUDA_ERROR_UNKNOWN);
    }

    if (cuFunc == NULL) {
        if (funcName && strcmp(funcName, "<unknown>") != 0) {
            hetgpu_pacc_launch_named_kernel_fn launch_named = resolve_hetgpu_pacc_launch_named_kernel();
            if (launch_named) {
                int named_result = launch_named(
                    funcName,
                    gridDim.x, gridDim.y, gridDim.z,
                    blockDim.x, blockDim.y, blockDim.z,
                    (unsigned int)sharedMem,
                    (void*)stream,
                    args,
                    NULL);
                if (named_result == HETGPU_CUDA_SUCCESS) {
                    HETGPU_LOG("[cudart_shim] named launch handled '%s' without CUfunction\n", funcName);
                    return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
                }
                if (named_result == HETGPU_CUDA_ERROR_UNKNOWN) {
                    fprintf(stderr,
                            "[cudart_shim] named launch for '%s' failed; refusing to skip kernel\n",
                            funcName);
                    return hetgpu_set_last_error(HETGPU_CUDA_ERROR_UNKNOWN);
                }
                const char* log_named_miss = getenv("HETGPU_CUDART_LOG_NAMED_MISS");
                if (log_named_miss && strcmp(log_named_miss, "1") == 0 && g_registry_miss_log_count < 12) {
                    fprintf(stderr,
                            "[cudart_shim] named launch did not handle '%s' (result=%d); cuFunc is NULL\n",
                            funcName,
                            named_result);
                    g_registry_miss_log_count++;
                }
            }
            if (hetgpu_requires_named_launch(funcName)) {
                fprintf(stderr,
                        "[cudart_shim] ERROR: named-only PACC kernel '%s' has NULL CUfunction and named launch did not take it; refusing lazy/normal launch\n",
                        funcName);
                return hetgpu_set_last_error(HETGPU_CUDA_ERROR_UNKNOWN);
            }
            if (hetgpu_cudart_lazy_ptx_fail_open_enabled()) {
                static unsigned long long lazy_ptx_fail_open_log_count = 0;
                unsigned long long log_index =
                    __sync_fetch_and_add(&lazy_ptx_fail_open_log_count, 1);
                unsigned long long log_limit =
                    hetgpu_cudart_lazy_ptx_fail_open_log_limit();
                if (log_index < log_limit) {
                    fprintf(stderr,
                            "[cudart_shim] lazy PTX fail-open for '%s'; skipping module load/compile during PACC bring-up\n",
                            funcName);
                } else if (log_limit != 0 && log_index == log_limit) {
                    fprintf(stderr,
                            "[cudart_shim] lazy PTX fail-open log limit reached (%llu); suppressing further messages\n",
                            log_limit);
                }
                return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
            }
            cuFunc = lazy_load_registered_function_for_launch(funcName, func);
            if (cuFunc != NULL) {
                HETGPU_LOG("[cudart_shim] launch-time lazy PTX resolved '%s': %p\n", funcName, cuFunc);
                goto launch_registered_kernel;
            } else if (!hetgpu_allow_skip_null_registered_kernel()) {
                fprintf(stderr,
                        "[cudart_shim] ERROR: registered kernel '%s' has no CUfunction and no named PACC handler; refusing to skip uninitialized output\n",
                        funcName);
                return hetgpu_set_last_error(HETGPU_CUDA_ERROR_UNKNOWN);
            }
        }
        if (g_registry_miss_log_count < 12) {
            Dl_info miss_info;
            if (dladdr(func, &miss_info) && miss_info.dli_fname) {
                fprintf(stderr,
                        "[cudart_shim] registry miss #%d: func=%p image=%s sym=%s registered=%d modules=%d\n",
                        g_registry_miss_log_count + 1,
                        func,
                        miss_info.dli_fname,
                        miss_info.dli_sname ? miss_info.dli_sname : "<unknown>",
                        g_function_count,
                        g_module_count);
            } else {
                fprintf(stderr,
                        "[cudart_shim] registry miss #%d: func=%p image=<unknown> registered=%d modules=%d\n",
                        g_registry_miss_log_count + 1,
                        func,
                        g_function_count,
                        g_module_count);
            }
            g_registry_miss_log_count++;
        }
        if (hetgpu_strict_pacc() || !hetgpu_allow_skip_unregistered_kernel()) {
            fprintf(stderr,
                    "[cudart_shim] ERROR: Function %p not in registry; refusing to skip uninitialized kernel output\n",
                    func);
            return hetgpu_set_last_error(HETGPU_CUDA_ERROR_UNKNOWN);
        }
        fprintf(stderr, "[cudart_shim] WARNING: Function %p not in registry - skipping (output uninitialized!)\n", func);
        fprintf(stderr, "[cudart_shim] This may cause SIGFPE in downstream operations like softmax\n");
        return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
    }

launch_registered_kernel:
    if (hetgpu_strict_pacc() && !hetgpu_pacc_kernel_has_handle(cuFunc)) {
        fprintf(stderr, "[cudart_shim] ERROR: PACC kernel '%s' has no executable handle\n", funcName);
        return hetgpu_set_last_error(HETGPU_CUDA_ERROR_UNKNOWN);
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

    return hetgpu_set_last_error((cudaError_t)result);
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

static int find_mapping_for_address(
    const void* addr,
    unsigned long* mapping_start,
    unsigned long* mapping_end,
    char* path,
    size_t path_size
) {
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
            if (mapping_start) *mapping_start = start;
            if (mapping_end) *mapping_end = end;
            if (path && path_size > 0) {
                path[0] = '\0';
                if (n >= 8 && filepath[0] != '\0' && filepath[0] != '[') {
                    char* fp = filepath;
                    while (*fp == ' ') fp++;
                    strncpy(path, fp, path_size - 1);
                    path[path_size - 1] = '\0';
                }
            }
            fclose(maps);
            return 1;
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

static int refresh_ptx_cache_entry(int cache_idx) {
    char find_cmd[512];
    snprintf(find_cmd, sizeof(find_cmd),
             "find %s -name '*.ptx' -type f 2>/dev/null | sort",
             g_ptx_cache[cache_idx].ptx_dir);

    FILE* find_p = popen(find_cmd, "r");
    g_ptx_cache[cache_idx].ptx_count = 0;
    if (!find_p) {
        return 0;
    }

    char ptx_path[256];
    while (fgets(ptx_path, sizeof(ptx_path), find_p) &&
           g_ptx_cache[cache_idx].ptx_count < MAX_PTX_PER_SO) {
        size_t len = strlen(ptx_path);
        if (len > 0 && ptx_path[len - 1] == '\n') {
            ptx_path[len - 1] = '\0';
        }
        if (ptx_path[0] != '\0') {
            strncpy(g_ptx_cache[cache_idx].ptx_files[g_ptx_cache[cache_idx].ptx_count],
                    ptx_path, 255);
            g_ptx_cache[cache_idx].ptx_files[g_ptx_cache[cache_idx].ptx_count][255] = '\0';
            g_ptx_cache[cache_idx].ptx_count++;
        }
    }
    pclose(find_p);
    return g_ptx_cache[cache_idx].ptx_count;
}

static int compile_ggml_cuda_sources_to_ptx(const char* so_path, const char* ptx_dir) {
    const char* marker = strstr(so_path, "/bin/libggml-cuda.so");
    if (!marker) {
        return 0;
    }

    size_t root_len = (size_t)(marker - so_path);
    if (root_len == 0 || root_len >= 512) {
        return 0;
    }

    char build_root[512];
    memcpy(build_root, so_path, root_len);
    build_root[root_len] = '\0';

    char compile_db[640];
    snprintf(compile_db, sizeof(compile_db), "%s/compile_commands.json", build_root);
    if (access(compile_db, R_OK) != 0) {
        fprintf(stderr, "[cudart_shim] No compile_commands.json at %s\n", compile_db);
        return 0;
    }

    fprintf(stderr, "[cudart_shim] Falling back to source->PTX compilation using %s\n", compile_db);

    const char* script = "/home/ubuntu/Documents/hetGPU_pacc/tools/source_to_ptx.py";
    if (access(script, R_OK) != 0) {
        fprintf(stderr, "[cudart_shim] source->PTX helper not found at %s\n", script);
        return 0;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return 0;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return 0;
    }

    if (pid == 0) {
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);

        unsetenv("LD_PRELOAD");
        unsetenv("HETGPU_PACC_ALLOW_HOST_DEVICE_MEM");
        unsetenv("HETGPU_PACC_LOG_KERNEL_LAUNCHES");
        unsetenv("HETGPU_CUDART_REGISTRY_LOG");
        unsetenv("HETGPU_PTX_EXTRACT_LOG");

        execl("/usr/bin/python3", "python3", script, compile_db, ptx_dir, (char*)NULL);
        _exit(127);
    }

    close(pipefd[1]);
    char line[128] = {0};
    ssize_t nread = read(pipefd[0], line, sizeof(line) - 1);
    close(pipefd[0]);

    int status = 0;
    waitpid(pid, &status, 0);

    int compiled_count = 0;
    if (nread > 0) {
        line[nread] = '\0';
        compiled_count = atoi(line);
    }

    int rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    fprintf(stderr, "[cudart_shim] source->PTX compile returned rc=%d count=%d\n", rc, compiled_count);
    return compiled_count;
}

static int compile_deepep_legacy_sources_to_ptx(const char* so_path, const char* ptx_dir) {
    if (!so_path || !ptx_dir || !strstr(so_path, "DeepEP")) {
        return 0;
    }

    char repo_root[512] = {0};
    const char* override_root = getenv("HETGPU_DEEPEP_ROOT");
    if (override_root && override_root[0] != '\0') {
        strncpy(repo_root, override_root, sizeof(repo_root) - 1);
    } else {
        const char* marker = strstr(so_path, "/build/");
        if (!marker) {
            marker = strstr(so_path, "/deep_ep/_C.");
        }
        if (!marker) {
            return 0;
        }
        size_t root_len = (size_t)(marker - so_path);
        if (root_len == 0 || root_len >= sizeof(repo_root)) {
            return 0;
        }
        memcpy(repo_root, so_path, root_len);
        repo_root[root_len] = '\0';
    }

    char layout_src[640];
    char intranode_src[640];
    char layout_ptx[640];
    char intranode_ptx[640];
    snprintf(layout_src, sizeof(layout_src), "%s/csrc/kernels/legacy/layout.cu", repo_root);
    snprintf(intranode_src, sizeof(intranode_src), "%s/csrc/kernels/legacy/intranode.cu", repo_root);
    snprintf(layout_ptx, sizeof(layout_ptx), "%s/layout.cu.ptx", ptx_dir);
    snprintf(intranode_ptx, sizeof(intranode_ptx), "%s/intranode.cu.ptx", ptx_dir);

    if (access(layout_src, R_OK) != 0 || access(intranode_src, R_OK) != 0) {
        fprintf(stderr, "[cudart_shim] DeepEP legacy CUDA sources not found under %s\n", repo_root);
        return 0;
    }

    const char* common_flags =
        "-DTHRUST_IGNORE_CUB_VERSION_CHECK "
        "-D_CG_LIMIT_INCLUDED_DEPENDENCIES "
        "-D__CUDACC_VER_MAJOR__=12 "
        "-D__CUDACC_VER_MINOR__=9 "
        "-D__CUDACC_VER_BUILD__=0 "
        "-D__CUDACC_VER_BUILD_ID__=0 "
        "--cuda-gpu-arch=sm_80 "
        "--cuda-path=/home/ubuntu/fake_cuda "
        "-I/home/ubuntu/fake_cuda/include "
        "--gcc-install-dir=/usr/lib/gcc/riscv64-linux-gnu/13 "
        "-Wno-unknown-cuda-version "
        "-I/usr/local/cuda/include/cccl "
        "-I/home/ubuntu/pytorch-main/torch/include "
        "-I/home/ubuntu/pytorch-main/torch/include/torch/csrc/api/include "
        "-I/home/ubuntu/fake_cuda/include "
        "-I/usr/include/python3.12 "
        "-D__CUDA_NO_HALF_OPERATORS__ "
        "-D__CUDA_NO_HALF_CONVERSIONS__ "
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__ "
        "-D__CUDA_NO_HALF2_OPERATORS__ "
        "-O3 "
        "-DHETGPU_DEEPEP_LEGACY_ONLY "
        "-DDISABLE_AGGRESSIVE_PTX_INSTRS "
        "-DTORCH_API_INCLUDE_EXTENSION_H "
        "-DTORCH_EXTENSION_NAME=_C "
        "-std=c++20 "
        "--cuda-device-only "
        "-S";

    char cmd[8192];
    int written = snprintf(
        cmd,
        sizeof(cmd),
        "unset LD_PRELOAD HETGPU_PACC_ALLOW_HOST_DEVICE_MEM HETGPU_PACC_LOG_KERNEL_LAUNCHES "
        "HETGPU_CUDART_REGISTRY_LOG HETGPU_PTX_EXTRACT_LOG HETGPU_CUDART_LOG_LAUNCH_EX; "
        "/usr/bin/clang++-20 %s -I\"%s/deep_ep/include\" -I\"%s/third-party/fmt/include\" "
        "\"%s\" -o \"%s\" && "
        "/usr/bin/clang++-20 %s -I\"%s/deep_ep/include\" -I\"%s/third-party/fmt/include\" "
        "\"%s\" -o \"%s\" 2>&1",
        common_flags,
        repo_root,
        repo_root,
        layout_src,
        layout_ptx,
        common_flags,
        repo_root,
        repo_root,
        intranode_src,
        intranode_ptx);
    if (written < 0 || (size_t)written >= sizeof(cmd)) {
        fprintf(stderr, "[cudart_shim] DeepEP source->PTX command was truncated\n");
        return 0;
    }

    fprintf(stderr, "[cudart_shim] Falling back to DeepEP legacy source->PTX compilation under %s\n", repo_root);
    FILE* p = popen(cmd, "r");
    if (!p) {
        return 0;
    }

    const char* log_lazy_ptx = getenv("HETGPU_CUDART_LOG_LAZY_PTX");
    int log_output = log_lazy_ptx && strcmp(log_lazy_ptx, "1") == 0;
    char line[512];
    while (fgets(line, sizeof(line), p)) {
        if (log_output || strstr(line, "error:") || strstr(line, "Error")) {
            fprintf(stderr, "[cudart_shim][deepep-ptx] %s", line);
        }
    }

    int status = pclose(p);
    int rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    int compiled_count = 0;
    if (access(layout_ptx, R_OK) == 0) {
        compiled_count++;
    }
    if (access(intranode_ptx, R_OK) == 0) {
        compiled_count++;
    }

    fprintf(stderr, "[cudart_shim] DeepEP source->PTX compile returned rc=%d count=%d\n", rc, compiled_count);
    return compiled_count;
}

// Extract all PTX from a .so file using cuobjdump
static const char* extract_ptx_from_so(const char* so_path) {
    // Check cache first
    for (int i = 0; i < g_ptx_cache_count; i++) {
        if (strcmp(g_ptx_cache[i].so_path, so_path) == 0) {
            HETGPU_LOG("[cudart_shim] Using cached PTX from %s\n", so_path);
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
             "/tmp/hetgpu_so_ptx_%ld_%d", (long)getpid(), cache_idx);
    mkdir(g_ptx_cache[cache_idx].ptx_dir, 0755);

    HETGPU_LOG("[cudart_shim] Extracting PTX from %s to %s\n",
            so_path, g_ptx_cache[cache_idx].ptx_dir);

    // Run cuobjdump to extract all PTX
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "cd %s && /home/ubuntu/Documents/hetGPU_pacc/tools/cuobjdump --extract-ptx all '%s' 2>&1",
             g_ptx_cache[cache_idx].ptx_dir, so_path);

    FILE* p = popen(cmd, "r");
    if (p) {
        char line[256];
        while (fgets(line, sizeof(line), p)) {
            // Just consume output, maybe log some of it
            if (strstr(line, "Extracting")) {
                HETGPU_LOG("[cudart_shim] %s", line);
            }
        }
        int ret = pclose(p);
        HETGPU_LOG("[cudart_shim] cuobjdump on .so returned: %d\n", WEXITSTATUS(ret));
    }

    refresh_ptx_cache_entry(cache_idx);
    if (g_ptx_cache[cache_idx].ptx_count == 0 && strstr(so_path, "libggml-cuda.so") != NULL) {
        compile_ggml_cuda_sources_to_ptx(so_path, g_ptx_cache[cache_idx].ptx_dir);
        refresh_ptx_cache_entry(cache_idx);
    } else if (g_ptx_cache[cache_idx].ptx_count == 0 && strstr(so_path, "DeepEP") != NULL) {
        compile_deepep_legacy_sources_to_ptx(so_path, g_ptx_cache[cache_idx].ptx_dir);
        refresh_ptx_cache_entry(cache_idx);
    }

    HETGPU_LOG("[cudart_shim] Extracted %d PTX files from %s\n",
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
                        HETGPU_LOG("[cudart_shim] Found matching PTX file: %s\n",
                                g_ptx_cache[i].ptx_files[j]);
                        return load_ptx_file(g_ptx_cache[i].ptx_files[j], out_size);
                    }
                }
            }

            // No pattern match, use round-robin to get next PTX file
            // This distributes modules across different PTX files
            int idx = g_ptx_file_index % g_ptx_cache[i].ptx_count;
            g_ptx_file_index++;

            HETGPU_LOG("[cudart_shim] Using PTX file %d/%d: %s\n",
                    idx + 1, g_ptx_cache[i].ptx_count,
                    g_ptx_cache[i].ptx_files[idx]);
            return load_ptx_file(g_ptx_cache[i].ptx_files[idx], out_size);
        }
    }
    return NULL;
}

static const char* ptx_filename_hint_for_kernel(const char* kernel_name) {
    if (!kernel_name) return NULL;
    if (strstr(kernel_name, "get_dispatch_layout")) return "layout.cu.ptx";
    if (strstr(kernel_name, "deep_ep") &&
        (strstr(kernel_name, "notify_dispatch") ||
         strstr(kernel_name, "cached_notify_dispatch") ||
         strstr(kernel_name, "cached_notify_combine") ||
         strstr(kernel_name, "dispatch") ||
         strstr(kernel_name, "combine") ||
         strstr(kernel_name, "barrier"))) return "intranode.cu.ptx";
    if (strstr(kernel_name, "rms_norm") || strstr(kernel_name, "l2_norm")) return "norm.cu.ptx";
    if (strstr(kernel_name, "rope_")) return "rope.cu.ptx";
    if (strstr(kernel_name, "soft_max")) return "softmax.cu.ptx";
    if (strstr(kernel_name, "quantize")) return "quantize.cu.ptx";
    if (strstr(kernel_name, "mul_mat_vec_q")) return "mmvq.cu.ptx";
    if (strstr(kernel_name, "mul_mat_vec_f")) return "mmvf.cu.ptx";
    if (strstr(kernel_name, "k_get_rows")) return "getrows.cu.ptx";
    if (strstr(kernel_name, "k_set_rows")) return "set-rows.cu.ptx";
    if (strstr(kernel_name, "scale_f32")) return "scale.cu.ptx";
    if (strstr(kernel_name, "concat")) return "concat.cu.ptx";
    if (strstr(kernel_name, "cpy_") || strstr(kernel_name, "cpy_scalar")) return "cpy.cu.ptx";
    if (strstr(kernel_name, "ssm_conv")) return "ssm-conv.cu.ptx";
    if (strstr(kernel_name, "ssm_scan")) return "ssm-scan.cu.ptx";
    if (strstr(kernel_name, "k_bin_bcast")) return "binbcast.cu.ptx";
    if (strstr(kernel_name, "unary_")) return "unary.cu.ptx";
    if (strstr(kernel_name, "gated_delta_net")) return "gated_delta_net.cu.ptx";
    if (strstr(kernel_name, "topk_moe")) return "topk-moe.cu.ptx";
    return NULL;
}

static int ptx_file_contains_kernel(const char* path, const char* kernel_name) {
    if (!path || !kernel_name) return 0;
    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    const size_t needle_len = strlen(kernel_name);
    if (needle_len == 0 || needle_len > 4096) {
        fclose(f);
        return 0;
    }

    char buf[8192 + 4096];
    size_t carry = 0;
    int found = 0;
    while (!found) {
        size_t n = fread(buf + carry, 1, 8192, f);
        size_t total = carry + n;
        if (total > 0) {
            buf[total] = '\0';
            found = strstr(buf, kernel_name) != NULL;
        }
        if (n < 8192) break;
        carry = needle_len > 1 ? needle_len - 1 : 0;
        if (carry > total) carry = total;
        memmove(buf, buf + total - carry, carry);
    }
    fclose(f);
    return found;
}

static int find_ptx_file_for_kernel(const char* ptx_dir, const char* kernel_name, char* out_path, size_t out_size) {
    if (!ptx_dir || !kernel_name || !out_path || out_size == 0) return 0;

    const char* hint = ptx_filename_hint_for_kernel(kernel_name);
    if (hint) {
        char hinted[512];
        snprintf(hinted, sizeof(hinted), "%s/%s", ptx_dir, hint);
        if (access(hinted, R_OK) == 0 && ptx_file_contains_kernel(hinted, kernel_name)) {
            strncpy(out_path, hinted, out_size - 1);
            out_path[out_size - 1] = '\0';
            return 1;
        }
    }

    for (int i = 0; i < g_ptx_cache_count; i++) {
        if (strcmp(g_ptx_cache[i].ptx_dir, ptx_dir) != 0) {
            continue;
        }
        for (int j = 0; j < g_ptx_cache[i].ptx_count; j++) {
            const char* candidate = g_ptx_cache[i].ptx_files[j];
            if (ptx_file_contains_kernel(candidate, kernel_name)) {
                strncpy(out_path, candidate, out_size - 1);
                out_path[out_size - 1] = '\0';
                return 1;
            }
        }
    }
    return 0;
}

static CUmodule load_or_get_ptx_module_for_kernel(const char* so_path, const char* kernel_name) {
    if (!so_path || !kernel_name) return NULL;

    const char* ptx_dir = extract_ptx_from_so(so_path);
    if (!ptx_dir) return NULL;

    char ptx_path[256] = {0};
    if (!find_ptx_file_for_kernel(ptx_dir, kernel_name, ptx_path, sizeof(ptx_path))) {
        fprintf(stderr, "[cudart_shim] No PTX file contains kernel '%s'\n", kernel_name);
        return NULL;
    }

    int device = hetgpu_current_device_index();
    for (int i = 0; i < g_ptx_module_cache_count; i++) {
        if (g_ptx_module_cache[i].device == device &&
            strcmp(g_ptx_module_cache[i].ptx_path, ptx_path) == 0) {
            return g_ptx_module_cache[i].module;
        }
    }

    size_t ptx_size = 0;
    char* ptx_data = load_ptx_file(ptx_path, &ptx_size);
    if (!ptx_data || ptx_size <= 50) {
        if (ptx_data) free(ptx_data);
        return NULL;
    }

    CUmodule module = NULL;
    hetgpu_cuModuleLoadData_fn p_cuModuleLoadData = resolve_cuModuleLoadData();
    (void)cudaSetDevice(device);
    CUresult result = p_cuModuleLoadData ? p_cuModuleLoadData(&module, ptx_data) : 1;
    free(ptx_data);
    if (result != 0 || !module) {
        fprintf(stderr, "[cudart_shim] Lazy PTX module load failed for %s: %d\n", ptx_path, result);
        return NULL;
    }

    if (g_ptx_module_cache_count < MAX_CACHED_PTX_MODULES) {
        strncpy(g_ptx_module_cache[g_ptx_module_cache_count].ptx_path, ptx_path, 255);
        g_ptx_module_cache[g_ptx_module_cache_count].ptx_path[255] = '\0';
        g_ptx_module_cache[g_ptx_module_cache_count].device = device;
        g_ptx_module_cache[g_ptx_module_cache_count].module = module;
        g_ptx_module_cache_count++;
    }
    const char* log_lazy_ptx = getenv("HETGPU_CUDART_LOG_LAZY_PTX");
    if (log_lazy_ptx && strcmp(log_lazy_ptx, "1") == 0) {
        fprintf(stderr, "[cudart_shim] Lazy-loaded PTX module %s for '%s' on cuda device %d\n", ptx_path, kernel_name, device);
    }
    return module;
}

static CUfunction lazy_load_registered_function_for_launch(const char* kernel_name, const void* launch_func) {
    RegisteredFunction* entry = find_registered_function_by_name(kernel_name);
    if (!entry) {
        return NULL;
    }
    CUfunction current_cufunc = registered_function_current_cufunc(entry);
    if (current_cufunc) {
        return current_cufunc;
    }

    void* anchor = entry->hostFun ? entry->hostFun : entry->deviceFun;
    if (!anchor) {
        anchor = (void*)launch_func;
    }

    Dl_info info;
    if (!dladdr(anchor, &info) || !info.dli_fname) {
        return NULL;
    }

    CUmodule module = load_or_get_ptx_module_for_kernel(info.dli_fname, kernel_name);
    if (!module) {
        return NULL;
    }

    CUfunction func = NULL;
    hetgpu_cuModuleGetFunction_fn p_cuModuleGetFunction = resolve_cuModuleGetFunction();
    CUresult result = p_cuModuleGetFunction ? p_cuModuleGetFunction(&func, module, kernel_name) : 1;
    if (result != 0 || !func) {
        HETGPU_LOG("[cudart_shim] launch-time lazy cuModuleGetFunction('%s') failed: %d\n",
                kernel_name,
                result);
        return NULL;
    }

    registered_function_set_current_device(entry, func, module);
    const char* log_lazy_ptx = getenv("HETGPU_CUDART_LOG_LAZY_PTX");
    if (log_lazy_ptx && strcmp(log_lazy_ptx, "1") == 0) {
        fprintf(stderr,
                "[cudart_shim] launch-time lazy resolved '%s' -> %p from %s on cuda device %d\n",
                kernel_name,
                func,
                info.dli_fname,
                hetgpu_current_device_index());
    }
    return func;
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

static int hetgpu_ptx_has_markers(const unsigned char* data, size_t size) {
    if (!data || size < 32) {
        return 0;
    }
    const char* text = (const char*)data;
    return strstr(text, ".version ") != NULL && strstr(text, ".target ") != NULL;
}

static char* hetgpu_dup_ptx_blob(const unsigned char* data, size_t size, size_t* out_size) {
    if (!hetgpu_ptx_has_markers(data, size)) {
        return NULL;
    }
    while (size > 0 && (data[size - 1] == '\0' || data[size - 1] == '\n' || data[size - 1] == '\r')) {
        size--;
    }
    char* out = (char*)malloc(size + 2);
    if (!out) {
        return NULL;
    }
    memcpy(out, data, size);
    out[size] = '\n';
    out[size + 1] = '\0';
    if (out_size) {
        *out_size = size + 1;
    }
    return out;
}

static char* extract_ptx_from_fatbin_memory(const unsigned char* base, size_t size, size_t* out_size) {
    if (!base || size < sizeof(FatbinHeader)) {
        return NULL;
    }

    const FatbinHeader* fatbin_header = (const FatbinHeader*)base;
    if (fatbin_header->magic != FATBIN_MAGIC || fatbin_header->header_size >= size) {
        return NULL;
    }

    const unsigned char* file_ptr = base + fatbin_header->header_size;
    const unsigned char* end_ptr = file_ptr + fatbin_header->files_size;
    if (end_ptr > base + size) {
        end_ptr = base + size;
    }

    while (file_ptr + sizeof(FatbinFileHeader) <= end_ptr) {
        const FatbinFileHeader* file_header = (const FatbinFileHeader*)file_ptr;
        if (file_header->header_size == 0) {
            break;
        }
        if (file_ptr + file_header->header_size > end_ptr) {
            break;
        }
        const unsigned char* payload = file_ptr + file_header->header_size;
        size_t payload_size = file_header->payload_size;
        if (payload + payload_size > end_ptr) {
            break;
        }

        if (file_header->kind == FATBIN_KIND_PTX) {
            if (file_header->uncompressed_payload > 0) {
                char* decompressed = (char*)malloc(file_header->uncompressed_payload + 1);
                if (decompressed) {
                    int result = hetgpu_lz4_decompress(
                        (const char*)payload,
                        decompressed,
                        (int)payload_size,
                        (int)file_header->uncompressed_payload
                    );
                    if (result > 0) {
                        decompressed[result] = '\0';
                        if (hetgpu_ptx_has_markers((const unsigned char*)decompressed, (size_t)result)) {
                            if (out_size) *out_size = (size_t)result;
                            return decompressed;
                        }
                    }
                    free(decompressed);
                }
            }

            char* raw = hetgpu_dup_ptx_blob(payload, payload_size, out_size);
            if (raw) {
                return raw;
            }
        }

        size_t entry_total = file_header->header_size + file_header->padded_payload_size;
        if (entry_total == 0) {
            break;
        }
        file_ptr += entry_total;
    }

    return NULL;
}

static char* extract_ptx_from_mapping_local(const void* anchor, size_t* out_size) {
    unsigned long start = 0;
    unsigned long end = 0;
    char image_path[512] = {0};
    if (!find_mapping_for_address(anchor, &start, &end, image_path, sizeof(image_path)) || end <= start) {
        return NULL;
    }

    const unsigned char* mapping = (const unsigned char*)start;
    size_t mapping_size = (size_t)(end - start);

    for (size_t i = 0; i + 4 <= mapping_size; ++i) {
        unsigned int magic = 0;
        memcpy(&magic, mapping + i, sizeof(magic));
        if (magic == FATBIN_MAGIC) {
            char* ptx = extract_ptx_from_fatbin_memory(mapping + i, mapping_size - i, out_size);
            if (ptx) {
                fprintf(stderr,
                        "[cudart_shim] Local mapping PTX extraction hit FATBIN_MAGIC at +0x%zx in %s\n",
                        i,
                        image_path[0] ? image_path : "<anonymous>");
                return ptx;
            }
        }
    }

    const unsigned char version_marker[] = ".version ";
    for (size_t i = 0; i + sizeof(version_marker) < mapping_size; ++i) {
        if (memcmp(mapping + i, version_marker, sizeof(version_marker) - 1) == 0) {
            size_t j = i;
            while (j < mapping_size) {
                unsigned char c = mapping[j];
                if (c == '\0') {
                    break;
                }
                if (!(c == '\n' || c == '\r' || c == '\t' || (c >= 32 && c <= 126))) {
                    break;
                }
                j++;
            }
            char* ptx = hetgpu_dup_ptx_blob(mapping + i, j - i, out_size);
            if (ptx) {
                fprintf(stderr,
                        "[cudart_shim] Local mapping PTX extraction hit raw PTX at +0x%zx in %s\n",
                        i,
                        image_path[0] ? image_path : "<anonymous>");
                return ptx;
            }
        }
    }

    return NULL;
}

static int pointer_looks_like_string(const void* p) {
    if (!p) return 0;
    const unsigned char* s = (const unsigned char*)p;
    for (size_t i = 0; i < 8; ++i) {
        unsigned char c = s[i];
        if (c == '\0') return i > 0;
        if (!(c == '/' || c == '.' || c == '_' || c == '-' ||
              (c >= '0' && c <= '9') ||
              (c >= 'A' && c <= 'Z') ||
              (c >= 'a' && c <= 'z'))) {
            return 0;
        }
    }
    return 1;
}

void** __cudaRegisterFatBinary(void* fatCubin) {
    HETGPU_LOG("[cudart_shim] __cudaRegisterFatBinary called with %p\n", fatCubin);

    if (!fatCubin) {
        fprintf(stderr, "[cudart_shim] ERROR: NULL fatCubin!\n");
        static void* dummy = NULL;
        return &dummy;
    }

    // Fat binary starts with magic number followed by version
    unsigned int* wrapper_magic = (unsigned int*)fatCubin;
    HETGPU_LOG("[cudart_shim] Fat binary wrapper magic: 0x%08x\n", wrapper_magic[0]);

    // Parse FatbincWrapper:
    // struct FatbincWrapper {
    //     unsigned int magic;      // 0x466243B1 (offset 0, 4 bytes)
    //     unsigned int version;    // 1 or 2    (offset 4, 4 bytes)
    //     void* data;              // pointer to FatbinHeader (offset 8, 8 bytes)
    //     void* filename_or_fatbins;           (offset 16, 8 bytes)
    // }
    // So data pointer is at offset 8
    void* fatbin_header_ptr = NULL;
    void* wrapper_aux_ptr = NULL;
    char wrapper_so_path[512] = {0};

    if (wrapper_magic[0] == 0x466243B1) {
        // Read the data pointer which is at offset 8
        void** data_ptr_location = (void**)((char*)fatCubin + 8);
        fatbin_header_ptr = *data_ptr_location;
        wrapper_aux_ptr = *(void**)((char*)fatCubin + 16);
    } else {
        // Fallback: assume data is offset by 16 bytes
        fatbin_header_ptr = (char*)fatCubin + 16;
    }

    HETGPU_LOG("[cudart_shim] FatbinHeader pointer: %p\n", fatbin_header_ptr);
    HETGPU_LOG("[cudart_shim] Fatbin wrapper aux pointer: %p\n", wrapper_aux_ptr);
    if (pointer_looks_like_string(wrapper_aux_ptr)) {
        HETGPU_LOG("[cudart_shim] Fatbin wrapper aux string: %s\n", (const char*)wrapper_aux_ptr);
    }
    if (find_so_from_address(fatCubin, wrapper_so_path, sizeof(wrapper_so_path))) {
        HETGPU_LOG("[cudart_shim] Fatbin wrapper mapped from .so: %s\n", wrapper_so_path);
    }

    // Parse FatbinHeader to find the actual CUBIN/PTX payload
    FatbinHeader* fatbin_header = (FatbinHeader*)fatbin_header_ptr;
    void* payload = NULL;
    size_t payload_size = 0;
    int payload_needs_free = 0;
    int defer_module_load = 0;

    if (fatbin_header && fatbin_header->magic == FATBIN_MAGIC) {
        HETGPU_LOG("[cudart_shim] Valid FatbinHeader found (magic: 0x%08x)\n", fatbin_header->magic);
        HETGPU_LOG("[cudart_shim] Header size: %u, Files size: %lu\n",
                fatbin_header->header_size, fatbin_header->files_size);

        // Start of file headers
        unsigned char* file_ptr = (unsigned char*)fatbin_header + fatbin_header->header_size;
        unsigned char* end_ptr = file_ptr + fatbin_header->files_size;

        // Iterate through file headers to find PTX or CUBIN
        while (file_ptr < end_ptr) {
            FatbinFileHeader* file_header = (FatbinFileHeader*)file_ptr;

            HETGPU_LOG("[cudart_shim] File entry: kind=0x%04x, header_size=%u, payload_size=%u, padded=%u, sm=%u, uncompressed=%lu\n",
                    file_header->kind, file_header->header_size,
                    file_header->payload_size, file_header->padded_payload_size,
                    file_header->sm_version, file_header->uncompressed_payload);

            // Prefer PTX over CUBIN
            if (file_header->kind == FATBIN_KIND_PTX && payload == NULL) {
                unsigned char* ptx_payload = file_ptr + file_header->header_size;
                size_t raw_size = file_header->payload_size;
                size_t uncompressed_size = file_header->uncompressed_payload;
                HETGPU_LOG("[cudart_shim] Found PTX payload at offset +%lu, compressed_size=%zu, uncompressed_size=%zu, sm=%u\n",
                        (unsigned long)((char*)ptx_payload - (char*)fatbin_header_ptr),
                        raw_size, uncompressed_size, file_header->sm_version);

                if (uncompressed_size > 0) {
                    // PTX payload is LZ4-compressed - decompress it
                    HETGPU_LOG("[cudart_shim] PTX is LZ4-compressed, decompressing %zu -> %zu bytes\n",
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
                            HETGPU_LOG("[cudart_shim] LZ4 decompression successful: %d bytes\n", result);

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
                    HETGPU_LOG("[cudart_shim] PTX payload is uncompressed, using directly\n");
                    payload = ptx_payload;
                    payload_size = raw_size;
                }
                break;  // Prefer PTX, so break immediately
            } else if (file_header->kind == FATBIN_KIND_ELF && payload == NULL) {
                unsigned char* raw_payload = file_ptr + file_header->header_size;

                // Debug: show first 20 bytes starting at different offsets to find ELF magic
                HETGPU_LOG("[cudart_shim] Looking for ELF magic (7f 45 4c 46):\n");
                for (int off = 0; off < 4; off++) {
                    HETGPU_LOG("[cudart_shim]   offset %d: ", off);
                    for (int i = 0; i < 8; i++) {
                        HETGPU_LOG("%02x ", raw_payload[off + i]);
                    }
                    HETGPU_LOG("\n");
                }

                // Check if ELF magic is at offset 1 (skip alignment byte)
                if (raw_payload[1] == 0x7f && raw_payload[2] == 'E' &&
                    raw_payload[3] == 'L' && raw_payload[4] == 'F') {
                    HETGPU_LOG("[cudart_shim] Found ELF magic at offset 1, adjusting payload pointer\n");
                    payload = raw_payload + 1;
                    payload_size = file_header->payload_size - 1;
                } else if (raw_payload[0] == 0x7f && raw_payload[1] == 'E' &&
                           raw_payload[2] == 'L' && raw_payload[3] == 'F') {
                    // ELF magic at expected position
                    payload = raw_payload;
                    payload_size = file_header->payload_size;
                } else {
                    // No ELF magic found, use raw payload
                    HETGPU_LOG("[cudart_shim] WARNING: No ELF magic found, using raw payload\n");
                    payload = raw_payload;
                    payload_size = file_header->payload_size;
                }

                HETGPU_LOG("[cudart_shim] Found ELF/CUBIN payload at offset +%lu, size=%zu\n",
                        (unsigned long)((char*)payload - (char*)fatbin_header_ptr), payload_size);
                // Don't break - keep looking for PTX
            }

            // Move to next file entry
            file_ptr += file_header->padded_payload_size + file_header->header_size;
        }
    } else {
        HETGPU_LOG("[cudart_shim] Invalid or missing FatbinHeader, using data pointer directly\n");
        payload = fatbin_header_ptr;

        const char* eager = getenv("HETGPU_CUDART_EAGER_PTX");
        if (eager && strcmp(eager, "1") == 0) {
            size_t local_ptx_size = 0;
            char* local_ptx = extract_ptx_from_mapping_local(fatCubin, &local_ptx_size);
            if (!local_ptx) {
                local_ptx = extract_ptx_from_mapping_local(fatbin_header_ptr, &local_ptx_size);
            }
            if (local_ptx && local_ptx_size > 50) {
                HETGPU_LOG("[cudart_shim] Invalid-header fallback loaded %zu bytes of PTX from local mapping\n", local_ptx_size);
                payload = local_ptx;
                payload_size = local_ptx_size;
                payload_needs_free = 1;
            }

            char so_path[512] = {0};
            if (!payload_needs_free) {
                if (wrapper_so_path[0] != '\0') {
                    strncpy(so_path, wrapper_so_path, sizeof(so_path) - 1);
                    so_path[sizeof(so_path) - 1] = '\0';
                } else if (find_so_from_address(fatbin_header_ptr, so_path, sizeof(so_path))) {
                    ;
                }
            }
            if (!payload_needs_free && so_path[0] != '\0') {
                HETGPU_LOG("[cudart_shim] Invalid-header fallback source .so: %s\n", so_path);
                const char* ptx_dir = extract_ptx_from_so(so_path);
                if (ptx_dir) {
                    size_t ptx_size = 0;
                    char* ptx_data = find_matching_ptx(ptx_dir, NULL, &ptx_size);
                    if (ptx_data && ptx_size > 50 && strstr(ptx_data, ".version") && strstr(ptx_data, ".target")) {
                        HETGPU_LOG("[cudart_shim] Invalid-header fallback loaded %zu bytes of PTX from .so\n", ptx_size);
                        payload = ptx_data;
                        payload_size = ptx_size;
                        payload_needs_free = 1;
                    } else if (ptx_data) {
                        HETGPU_LOG("[cudart_shim] Invalid-header fallback PTX not usable (%zu bytes)\n", ptx_size);
                        free(ptx_data);
                    }
                }
            }
        } else {
            defer_module_load = 1;
            payload = NULL;
            payload_size = 0;
            if (wrapper_so_path[0] != '\0') {
                HETGPU_LOG("[cudart_shim] Invalid-header fatbin from %s; deferring PTX load until __cudaRegisterFunction\n",
                        wrapper_so_path);
            }
        }
    }

    if (defer_module_load) {
        payload = NULL;
    } else {
        // If we didn't find a payload, use the data pointer directly as fallback
        if (payload == NULL) {
            HETGPU_LOG("[cudart_shim] No valid payload found in fat binary, using data pointer as fallback\n");
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
                HETGPU_LOG("[cudart_shim] Detected binary CUBIN, attempting PTX extraction from .so file...\n");
                HETGPU_LOG("[cudart_shim] Payload first 16 bytes: ");
                for (size_t i = 0; i < 16 && i < payload_size; i++) {
                    HETGPU_LOG("%02x ", payload_bytes[i]);
                }
                HETGPU_LOG("\n");

                size_t local_ptx_size = 0;
                char* local_ptx = extract_ptx_from_mapping_local(fatCubin, &local_ptx_size);
                if (!local_ptx) {
                    local_ptx = extract_ptx_from_mapping_local(fatbin_header_ptr, &local_ptx_size);
                }
                if (local_ptx && local_ptx_size > 50) {
                    HETGPU_LOG("[cudart_shim] Successfully loaded %zu bytes of PTX from local mapping\n", local_ptx_size);
                    payload = local_ptx;
                    payload_size = local_ptx_size;
                    payload_needs_free = 1;
                }

                // NEW APPROACH: Find the source .so file from the fatbin pointer address
                // and extract PTX from the full .so using cuobjdump
                char so_path[512] = {0};
                if (!payload_needs_free) {
                    if (wrapper_so_path[0] != '\0') {
                        strncpy(so_path, wrapper_so_path, sizeof(so_path) - 1);
                        so_path[sizeof(so_path) - 1] = '\0';
                    } else if (find_so_from_address(fatbin_header_ptr, so_path, sizeof(so_path))) {
                        ;
                    }
                }
                if (!payload_needs_free && so_path[0] != '\0') {
                    HETGPU_LOG("[cudart_shim] Found source .so: %s\n", so_path);

                    const char* ptx_dir = extract_ptx_from_so(so_path);
                    if (ptx_dir) {
                        size_t ptx_size = 0;
                        char* ptx_data = find_matching_ptx(ptx_dir, NULL, &ptx_size);
                        if (ptx_data && ptx_size > 50) {
                            if (strstr(ptx_data, ".version") && strstr(ptx_data, ".target")) {
                                HETGPU_LOG("[cudart_shim] Successfully loaded %zu bytes of PTX from .so\n", ptx_size);
                                HETGPU_LOG("[cudart_shim] PTX preview: %.200s...\n", ptx_data);
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
    }

    // Try to load the payload as a module
    CUmodule module = NULL;
    CUresult result = 1;
    if (!defer_module_load && payload != NULL) {
        hetgpu_cuModuleLoadData_fn p_cuModuleLoadData = resolve_cuModuleLoadData();
        result = p_cuModuleLoadData ? p_cuModuleLoadData(&module, payload) : 1;
    }

    // Free decompressed PTX buffer now that cuModuleLoadData has consumed it
    if (payload_needs_free && payload) {
        free(payload);
        payload = NULL;
        payload_needs_free = 0;
    }

    const char* log_module_loads = getenv("HETGPU_CUDART_LOG_MODULE_LOADS");
    int log_module_load = log_module_loads && strcmp(log_module_loads, "1") == 0;
    if (result != 0) {
        if (log_module_load) {
            fprintf(stderr, "[cudart_shim] cuModuleLoadData failed: %d\n", result);
            fprintf(stderr, "[cudart_shim] Module load failed, but continuing with placeholder\n");
        }
    } else if (log_module_load) {
        fprintf(stderr, "[cudart_shim] Successfully loaded module: %p\n", module);
    }

    // Store the module
    if (g_module_count < MAX_MODULES) {
        g_modules[g_module_count].module = module;
        g_modules[g_module_count].fatCubinHandle = fatCubin;
        g_module_count++;

        if (log_module_load) {
            fprintf(stderr, "[cudart_shim] Registered module %d (total: %d)\n",
                    g_module_count - 1, g_module_count);
        }
    }

    // Return the module handle as the fatCubinHandle
    // PyTorch will pass this back to __cudaRegisterFunction
    static void* handle_storage[MAX_MODULES];
    handle_storage[g_module_count - 1] = (void*)module;
    return &handle_storage[g_module_count - 1];
}

void __cudaRegisterFatBinaryEnd(void** fatCubinHandle) {
    HETGPU_LOG("[cudart_shim] __cudaRegisterFatBinaryEnd called\n");
    (void)fatCubinHandle;
}

void __cudaUnregisterFatBinary(void** fatCubinHandle) {
    HETGPU_LOG("[cudart_shim] __cudaUnregisterFatBinary called\n");
    (void)fatCubinHandle;
}

void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun,
                            const char* deviceName, int thread_limit, void* tid, void* bid,
                            void* bDim, void* gDim, void* wSize) {
    (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;

    if (!fatCubinHandle || !hostFun || !deviceName) {
        fprintf(stderr, "[cudart_shim] __cudaRegisterFunction: invalid arguments\n");
        return;
    }

    CUmodule module = (CUmodule)(*fatCubinHandle);
    HETGPU_LOG("[cudart_shim] __cudaRegisterFunction: hostFun=%p deviceFun=%p name='%s', module=%p\n",
            hostFun, deviceFun, deviceName, module);

    // Get the function from the module
    CUfunction func = NULL;
    hetgpu_cuModuleGetFunction_fn p_cuModuleGetFunction = resolve_cuModuleGetFunction();
    CUresult result = p_cuModuleGetFunction ? p_cuModuleGetFunction(&func, module, deviceName) : 1;

    if (result != 0) {
        const char* lazy_register_ptx = getenv("HETGPU_CUDART_LAZY_REGISTER_PTX");
        int try_lazy_register_ptx = lazy_register_ptx && strcmp(lazy_register_ptx, "1") == 0;
        char so_path[512] = {0};
        Dl_info host_info;
        if (try_lazy_register_ptx && dladdr((void*)hostFun, &host_info) && host_info.dli_fname) {
            strncpy(so_path, host_info.dli_fname, sizeof(so_path) - 1);
            so_path[sizeof(so_path) - 1] = '\0';
        }
        if (so_path[0] != '\0') {
            CUmodule lazy_module = load_or_get_ptx_module_for_kernel(so_path, deviceName);
            if (lazy_module) {
                CUfunction lazy_func = NULL;
                CUresult lazy_result = p_cuModuleGetFunction
                    ? p_cuModuleGetFunction(&lazy_func, lazy_module, deviceName)
                    : 1;
                if (lazy_result == 0 && lazy_func) {
                    module = lazy_module;
                    func = lazy_func;
                    result = 0;
                } else {
                    HETGPU_LOG("[cudart_shim] lazy cuModuleGetFunction('%s') failed: %d\n",
                            deviceName,
                            lazy_result);
                }
            }
        }
        if (result != 0) {
            HETGPU_LOG("[cudart_shim] cuModuleGetFunction('%s') failed: %d\n", deviceName, result);
            // Continue anyway - func will be NULL, which we handle in launch
        }
    } else {
        HETGPU_LOG("[cudart_shim] Got function '%s': %p\n", deviceName, func);
    }

    // Store the mapping
    if (g_function_count < MAX_FUNCTIONS) {
        g_functions[g_function_count].hostFun = (void*)hostFun;
        g_functions[g_function_count].deviceFun = (void*)deviceFun;
        g_functions[g_function_count].cuFunc = func;
        g_functions[g_function_count].module = module;
        memset(g_functions[g_function_count].cuFuncByDevice, 0, sizeof(g_functions[g_function_count].cuFuncByDevice));
        memset(g_functions[g_function_count].moduleByDevice, 0, sizeof(g_functions[g_function_count].moduleByDevice));
        registered_function_set_current_device(&g_functions[g_function_count], func, module);
        strncpy(g_functions[g_function_count].name, deviceName, 255);
        g_functions[g_function_count].name[255] = '\0';

        HETGPU_LOG("[cudart_shim] Registered function %d: host=%p device=%p -> %p ('%s')\n",
                g_function_count, hostFun, deviceFun, func, deviceName);
        const char* log_registration = getenv("HETGPU_CUDART_LOG_REGISTRATION");
        if (log_registration && strcmp(log_registration, "1") == 0 &&
                g_registry_register_log_count < 16) {
            Dl_info host_info;
            if (dladdr((void*)hostFun, &host_info) && host_info.dli_fname) {
                fprintf(stderr,
                        "[cudart_shim] register #%d: name='%s' host=%p device=%p cu=%p image=%s sym=%s\n",
                        g_registry_register_log_count + 1,
                        deviceName,
                        hostFun,
                        deviceFun,
                        func,
                        host_info.dli_fname,
                        host_info.dli_sname ? host_info.dli_sname : "<unknown>");
            } else {
                fprintf(stderr,
                        "[cudart_shim] register #%d: name='%s' host=%p device=%p cu=%p image=<unknown>\n",
                        g_registry_register_log_count + 1,
                        deviceName,
                        hostFun,
                        deviceFun,
                        func);
            }
            g_registry_register_log_count++;
        }

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
cudaError_t cudaGetLastError(void) {
    cudaError_t error = g_last_cuda_error;
    g_last_cuda_error = HETGPU_CUDA_SUCCESS;
    return error;
}

cudaError_t cudaPeekAtLastError(void) { return g_last_cuda_error; }

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
    const size_t four_gb = (size_t)4 * 1024 * 1024 * 1024ULL;
    if (free) *free = four_gb;
    if (total) *total = four_gb;
    return 0;
}

// Basic memory/runtime APIs - forward to driver API for proper tracking
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    hetgpu_cuda_malloc_trace("[cudart_malloc] entry");
    if (!devPtr) return 1; // cudaErrorInvalidValue

    // In PACC mode cuMemAlloc_v2 can allocate from the shared-DDR arena without
    // a CUDA context. Calling cudaSetDevice from this low-level allocation path
    // re-enters Rust global_state initialization and can trap on this RISC-V
    // toolchain, so keep the old context creation path opt-in for diagnostics.
    CUcontext cur = NULL;
    hetgpu_cuda_malloc_trace("[cudart_malloc] ctx get before");
    (void)cuCtxGetCurrent(&cur);
    hetgpu_cuda_malloc_trace("[cudart_malloc] ctx get after");
    const char* ensure_context = getenv("HETGPU_CUDART_MALLOC_ENSURE_CONTEXT");
    if (cur == NULL && ensure_context && strcmp(ensure_context, "1") == 0) {
        int dev = 0;
        hetgpu_cuda_malloc_trace("[cudart_malloc] get device before");
        (void)cudaGetDevice(&dev);
        hetgpu_cuda_malloc_trace("[cudart_malloc] get device after");
        hetgpu_cuda_malloc_trace("[cudart_malloc] set device before");
        (void)cudaSetDevice(dev);
        hetgpu_cuda_malloc_trace("[cudart_malloc] set device after");
        hetgpu_cuda_malloc_trace("[cudart_malloc] ctx get2 before");
        (void)cuCtxGetCurrent(&cur);
        hetgpu_cuda_malloc_trace("[cudart_malloc] ctx get2 after");
    }

    CUdeviceptr dptr = 0;
    hetgpu_cuda_malloc_trace("[cudart_malloc] cuMemAlloc before");
    CUresult result = cuMemAlloc_v2(&dptr, size);
    hetgpu_cuda_malloc_trace("[cudart_malloc] cuMemAlloc after");
    if (result != 0) {
        const char* real_mem = getenv("HETGPU_PACC_REAL_DEVICE_MEM");
        if (hetgpu_strict_pacc() ||
            hetgpu_pacc_requires_tracked_allocations() ||
            (real_mem && strcmp(real_mem, "1") == 0)) {
            fprintf(stderr,
                    "[cudart_shim] cudaMalloc(%zu) cuMemAlloc_v2 failed (%d); "
                    "refusing untracked host allocation in PACC mode\n",
                    size, result);
            return hetgpu_set_last_error(2); // cudaErrorMemoryAllocation
        }
        // Fallback: host allocation (zeroed)
        void* ptr = NULL;
        if (size > 0) {
            ptr = aligned_alloc(256, ((size + 255) / 256) * 256);
            if (ptr) memset(ptr, 0, size);
        } else {
            ptr = (void*)0x1; // sentinel
        }
        *devPtr = ptr;
        return ptr ? 0 : 2; // cudaErrorMemoryAllocation if NULL
    }
    hetgpu_cuda_malloc_trace("[cudart_malloc] store before");
    *devPtr = (void*)dptr;
    hetgpu_cuda_malloc_trace("[cudart_malloc] store after");
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

    cudaError_t err = HETGPU_CUDA_SUCCESS;
    switch (kind) {
        case HETGPU_CUDA_MEMCPY_HOST_TO_HOST:
            memcpy(dst, src, count);
            break;
        case HETGPU_CUDA_MEMCPY_HOST_TO_DEVICE:
            err = hetgpu_cuda_from_cu(cuMemcpyHtoD_v2((CUdeviceptr)dst, src, count));
            break;
        case HETGPU_CUDA_MEMCPY_DEVICE_TO_HOST:
            err = hetgpu_cuda_from_cu(cuMemcpyDtoH_v2(dst, (CUdeviceptr)src, count));
            break;
        case HETGPU_CUDA_MEMCPY_DEVICE_TO_DEVICE:
            err = hetgpu_cuda_memcpy_d2d(dst, src, count);
            break;
        case HETGPU_CUDA_MEMCPY_DEFAULT: {
            int dst_dev = hetgpu_likely_device_ptr(dst);
            int src_dev = hetgpu_likely_device_ptr(src);
            if (dst_dev && src_dev) {
                err = hetgpu_cuda_memcpy_d2d(dst, src, count);
            } else if (dst_dev) {
                err = hetgpu_cuda_from_cu(cuMemcpyHtoD_v2((CUdeviceptr)dst, src, count));
            } else if (src_dev) {
                err = hetgpu_cuda_from_cu(cuMemcpyDtoH_v2(dst, (CUdeviceptr)src, count));
            } else {
                memcpy(dst, src, count);
            }
            break;
        }
        default:
            err = HETGPU_CUDA_ERROR_INVALID_VALUE;
            break;
    }
    return hetgpu_set_last_error(err);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    (void)stream; return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch,
                              const void *src, size_t spitch,
                              size_t width, size_t height,
                              cudaMemcpyKind kind, cudaStream_t stream) {
    (void)stream;
    if (!dst || !src) return hetgpu_set_last_error(HETGPU_CUDA_ERROR_INVALID_VALUE);
    for (size_t row = 0; row < height; ++row) {
        const char *src_row = (const char *)src + row * spitch;
        char *dst_row = (char *)dst + row * dpitch;
        cudaError_t err = cudaMemcpy(dst_row, src_row, width, kind);
        if (err != HETGPU_CUDA_SUCCESS) {
            return hetgpu_set_last_error(err);
        }
    }
    return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
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

    cudaMemcpyBatchOp* ops = (cudaMemcpyBatchOp*)opList;
    for (size_t i = 0; i < numOps; i++) {
        if (ops[i].dst && ops[i].src && ops[i].count > 0) {
            cudaError_t err = cudaMemcpy(
                ops[i].dst,
                ops[i].src,
                ops[i].count,
                HETGPU_CUDA_MEMCPY_DEFAULT
            );
            if (err != HETGPU_CUDA_SUCCESS) {
                return err;
            }
        }
    }
    return 0;
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) {
    (void)dstDevice; (void)srcDevice; (void)stream;
    if (!dst || !src || count == 0) return 0;
    return cudaMemcpy(dst, src, count, HETGPU_CUDA_MEMCPY_DEVICE_TO_DEVICE);
}

cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms *p, cudaStream_t stream) {
    (void)stream;
    if (!p) return hetgpu_set_last_error(HETGPU_CUDA_ERROR_INVALID_VALUE);
    size_t row_bytes = p->extent.width;
    for (size_t z = 0; z < p->extent.depth; ++z) {
        for (size_t y = 0; y < p->extent.height; ++y) {
            const char *src_row = (const char *)p->srcPtr.ptr +
                (p->srcPos.z + z) * p->srcPtr.pitch * p->srcPtr.ysize +
                (p->srcPos.y + y) * p->srcPtr.pitch +
                p->srcPos.x;
            char *dst_row = (char *)p->dstPtr.ptr +
                (p->dstPos.z + z) * p->dstPtr.pitch * p->dstPtr.ysize +
                (p->dstPos.y + y) * p->dstPtr.pitch +
                p->dstPos.x;
            cudaError_t err = cudaMemcpy(dst_row, src_row, row_bytes, HETGPU_CUDA_MEMCPY_DEVICE_TO_DEVICE);
            if (err != HETGPU_CUDA_SUCCESS) {
                return hetgpu_set_last_error(err);
            }
        }
    }
    return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
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

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    (void)flags;
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    (void)stream; return cudaFree(devPtr);
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    if (!devPtr || devPtr == (void*)0x1 || count == 0) return 0;
    return hetgpu_set_last_error(
        hetgpu_cuda_from_cu(cuMemsetD8_v2((CUdeviceptr)devPtr, (unsigned char)value, count))
    );
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    (void)stream;
    return cudaMemset(devPtr, value, count);
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    return cudaHostAlloc(ptr, size, 0);
}

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    (void)flags;
    return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
}

cudaError_t cudaLaunchCooperativeKernel(const void *func,
                                        dim3 gridDim,
                                        dim3 blockDim,
                                        void **args,
                                        size_t sharedMem,
                                        cudaStream_t stream) {
    return __cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec,
                                cudaGraph_t hGraph,
                                cudaGraphNode_t *hErrorNode_out,
                                cudaGraphExecUpdateResult *updateResult_out) {
    (void)hGraphExec;
    (void)hGraph;
    if (hErrorNode_out) {
        *hErrorNode_out = NULL;
    }
    if (updateResult_out) {
        *updateResult_out = 0;
    }
    return hetgpu_set_last_error(HETGPU_CUDA_SUCCESS);
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
