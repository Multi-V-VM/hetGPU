#import <Foundation/Foundation.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

int hetgpu_apple_ane_gemm(int transa, int transb,
                          int m, int n, int k,
                          float alpha,
                          const void *A, int Atype, int lda,
                          const void *B, int Btype, int ldb,
                          float beta,
                          void *C, int Ctype, int ldc);

int hetgpu_apple_metal_gemm(int transa, int transb,
                            int m, int n, int k,
                            float alpha,
                            const void *A, int Atype, int lda,
                            const void *B, int Btype, int ldb,
                            float beta,
                            void *C, int Ctype, int ldc);

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUdeviceptr;

enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_NOT_SUPPORTED = 801,
};

typedef CUresult (*HetGpuPtxModuleLoadDataFn)(CUmodule *module, const void *image);
typedef CUresult (*HetGpuPtxModuleGetFunctionFn)(CUfunction *hfunc, CUmodule hmod, const char *name);
typedef CUresult (*HetGpuPtxModuleUnloadFn)(CUmodule module);
typedef CUresult (*HetGpuPtxLaunchKernelFn)(CUfunction f,
                                            unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                            unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                            unsigned int sharedMemBytes,
                                            CUstream hStream,
                                            void **kernelParams,
                                            void **extra);

static void *hetgpu_resolve_optional_symbol(const char *name) {
    return dlsym(RTLD_DEFAULT, name);
}

static int hetgpu_allow_fake_ptx(void) {
    const char *value = getenv("HETGPU_ALLOW_FAKE_PTX");
    return value && strcmp(value, "1") == 0;
}

int hetgpu_ane_gemm(int transa, int transb,
                    int m, int n, int k,
                    float alpha,
                    const void *A, int Atype, int lda,
                    const void *B, int Btype, int ldb,
                    float beta,
                    void *C, int Ctype, int ldc) {
    const char *backend = getenv("HETGPU_APPLE_BACKEND");
    if (backend && strcmp(backend, "metal") == 0) {
        return hetgpu_apple_metal_gemm(transa, transb, m, n, k, alpha,
                                       A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
    }

    int ane_result = hetgpu_apple_ane_gemm(transa, transb, m, n, k, alpha,
                                           A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
    if (ane_result == 0) {
        return 0;
    }
    return hetgpu_apple_metal_gemm(transa, transb, m, n, k, alpha,
                                   A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
}

CUresult cuInit(unsigned int flags) {
    (void)flags;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count) {
    if (!count) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *count = 1;
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *version) {
    if (!version) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *version = 12000;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (!device || ordinal != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    (void)dev;
    if (!name || len <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    snprintf(name, (size_t)len, "hetGPU Apple ANE/Metal");
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    (void)dev;
    if (!bytes) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *bytes = 8ull * 1024ull * 1024ull * 1024ull;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev) {
    (void)attrib;
    (void)dev;
    if (!pi) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pi = 1;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    (void)dev;
    if (!pctx) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pctx = (CUcontext)0x1;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    (void)dev;
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    (void)flags;
    (void)dev;
    if (!pctx) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pctx = (CUcontext)0x1;
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
    (void)ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    (void)ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
    if (!pctx) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pctx = (CUcontext)0x1;
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    if (!dptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *dptr = calloc(1, bytesize ? bytesize : 1);
    return *dptr ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    free(dptr);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t byteCount) {
    if (!dstDevice || (!srcHost && byteCount)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    memcpy(dstDevice, srcHost, byteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t byteCount) {
    if (!dstHost || (!srcDevice && byteCount)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    memcpy(dstHost, srcDevice, byteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount) {
    if (!dstDevice || (!srcDevice && byteCount)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    memmove(dstDevice, srcDevice, byteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t n) {
    if (!dstDevice && n) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    memset(dstDevice, uc, n);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t n) {
    if (!dstDevice && n) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    uint32_t *dst = (uint32_t *)dstDevice;
    for (size_t i = 0; i < n; i++) {
        dst[i] = ui;
    }
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!module || !image) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    HetGpuPtxModuleLoadDataFn load_data =
        (HetGpuPtxModuleLoadDataFn)hetgpu_resolve_optional_symbol("hetgpu_apple_ptx_module_load_data");
    if (load_data) {
        return load_data(module, image);
    }
    if (!hetgpu_allow_fake_ptx()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    *module = (CUmodule)0x2;
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, void *options, void *optionValues) {
    (void)numOptions;
    (void)options;
    (void)optionValues;
    return cuModuleLoadData(module, image);
}

CUresult cuModuleUnload(CUmodule module) {
    HetGpuPtxModuleUnloadFn unload =
        (HetGpuPtxModuleUnloadFn)hetgpu_resolve_optional_symbol("hetgpu_apple_ptx_module_unload");
    if (unload) {
        return unload(module);
    }
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    if (!hfunc || !hmod || !name) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    HetGpuPtxModuleGetFunctionFn get_function =
        (HetGpuPtxModuleGetFunctionFn)hetgpu_resolve_optional_symbol("hetgpu_apple_ptx_module_get_function");
    if (get_function) {
        return get_function(hfunc, hmod, name);
    }
    if (!hetgpu_allow_fake_ptx()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    *hfunc = (CUfunction)0x3;
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void **kernelParams,
                        void **extra) {
    if (!f) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    HetGpuPtxLaunchKernelFn launch =
        (HetGpuPtxLaunchKernelFn)hetgpu_resolve_optional_symbol("hetgpu_apple_ptx_launch_kernel");
    if (launch) {
        return launch(f, gridDimX, gridDimY, gridDimZ,
                      blockDimX, blockDimY, blockDimZ,
                      sharedMemBytes, hStream, kernelParams, extra);
    }
    if (!hetgpu_allow_fake_ptx()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamCreate(CUstream *stream, unsigned int flags) {
    (void)flags;
    if (!stream) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *stream = NULL;
    return CUDA_SUCCESS;
}

CUresult cuStreamSynchronize(CUstream stream) {
    (void)stream;
    return CUDA_SUCCESS;
}

CUresult cuStreamDestroy_v2(CUstream stream) {
    (void)stream;
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char **pStr) {
    if (!pStr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pStr = error == CUDA_SUCCESS ? "CUDA_SUCCESS" : "hetGPU Apple CUDA stub error";
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char **pStr) {
    if (!pStr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pStr = error == CUDA_SUCCESS ? "CUDA_SUCCESS" : "CUDA_ERROR_HETGPU_APPLE_STUB";
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, uint64_t flags) {
    (void)cudaVersion;
    (void)flags;
    if (!symbol || !pfn) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pfn = dlsym(RTLD_DEFAULT, symbol);
    return *pfn ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             uint64_t flags, uint64_t *symbolStatus) {
    CUresult result = cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    if (symbolStatus) {
        *symbolStatus = (result == CUDA_SUCCESS) ? 0 : 1;
    }
    return result;
}
