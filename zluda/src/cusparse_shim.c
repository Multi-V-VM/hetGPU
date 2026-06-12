#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(HETGPU_DEBUG_LOGS)
#define HETGPU_CUSPARSE_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
#define HETGPU_CUSPARSE_LOG(...) ((void)0)
#endif

typedef int cusparseStatus_t;

enum {
    CUSPARSE_STATUS_SUCCESS = 0,
    CUSPARSE_STATUS_INVALID_VALUE = 3,
};

static void *hetgpu_cusparse_alloc_handle(void) {
    void *ptr = calloc(1, 1);
    return ptr ? ptr : (void *)(uintptr_t)0x1;
}

static cusparseStatus_t hetgpu_cusparse_create_handle(void **handle) {
    if (!handle) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    *handle = hetgpu_cusparse_alloc_handle();
    return CUSPARSE_STATUS_SUCCESS;
}

static cusparseStatus_t hetgpu_cusparse_destroy_handle(void *handle) {
    if (handle && handle != (void *)(uintptr_t)0x1) {
        free(handle);
    }
    return CUSPARSE_STATUS_SUCCESS;
}

const char *cusparseGetErrorString(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";
        default:
            return "CUSPARSE_STATUS_UNKNOWN";
    }
}

const char *cusparseGetErrorName(cusparseStatus_t status) {
    return cusparseGetErrorString(status);
}

cusparseStatus_t cusparseCreate(void **handle) {
    HETGPU_CUSPARSE_LOG("[hetGPU cusparse_shim] cusparseCreate\n");
    return hetgpu_cusparse_create_handle(handle);
}

cusparseStatus_t cusparseDestroy(void *handle) {
    HETGPU_CUSPARSE_LOG("[hetGPU cusparse_shim] cusparseDestroy\n");
    return hetgpu_cusparse_destroy_handle(handle);
}

cusparseStatus_t cusparseCreateMatDescr(void **descr) { return hetgpu_cusparse_create_handle(descr); }
cusparseStatus_t cusparseDestroyMatDescr(void *descr) { return hetgpu_cusparse_destroy_handle(descr); }
cusparseStatus_t cusparseCreateBsrsm2Info(void **info) { return hetgpu_cusparse_create_handle(info); }
cusparseStatus_t cusparseDestroyBsrsm2Info(void *info) { return hetgpu_cusparse_destroy_handle(info); }
cusparseStatus_t cusparseCreateBsrsv2Info(void **info) { return hetgpu_cusparse_create_handle(info); }
cusparseStatus_t cusparseDestroyBsrsv2Info(void *info) { return hetgpu_cusparse_destroy_handle(info); }
cusparseStatus_t cusparseSpGEMM_createDescr(void **descr) { return hetgpu_cusparse_create_handle(descr); }
cusparseStatus_t cusparseSpGEMM_destroyDescr(void *descr) { return hetgpu_cusparse_destroy_handle(descr); }
cusparseStatus_t cusparseSpSM_createDescr(void **descr) { return hetgpu_cusparse_create_handle(descr); }
cusparseStatus_t cusparseSpSM_destroyDescr(void *descr) { return hetgpu_cusparse_destroy_handle(descr); }
cusparseStatus_t cusparseSpSV_createDescr(void **descr) { return hetgpu_cusparse_create_handle(descr); }
cusparseStatus_t cusparseSpSV_destroyDescr(void *descr) { return hetgpu_cusparse_destroy_handle(descr); }
cusparseStatus_t cusparseCreateCoo(void **descr) { return hetgpu_cusparse_create_handle(descr); }
cusparseStatus_t cusparseCreateCsr(void **descr) { return hetgpu_cusparse_create_handle(descr); }
cusparseStatus_t cusparseCreateDnMat(void **descr) { return hetgpu_cusparse_create_handle(descr); }
cusparseStatus_t cusparseDestroyDnMat(void *descr) { return hetgpu_cusparse_destroy_handle(descr); }
cusparseStatus_t cusparseCreateDnVec(void **descr) { return hetgpu_cusparse_create_handle(descr); }
cusparseStatus_t cusparseDestroyDnVec(void *descr) { return hetgpu_cusparse_destroy_handle(descr); }
cusparseStatus_t cusparseDestroySpMat(void *descr) { return hetgpu_cusparse_destroy_handle(descr); }

cusparseStatus_t cusparseGetProperty(int type, int *value) {
    (void)type;
    if (value) {
        *value = 0;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetVersion(void *handle, int *version) {
    (void)handle;
    if (version) {
        *version = 12050;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetStream(void *handle, void *stream) {
    (void)handle;
    (void)stream;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetPointerMode(void *handle, int mode) {
    (void)handle;
    (void)mode;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatDiagType(void *descr, int diagType) {
    (void)descr;
    (void)diagType;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatFillMode(void *descr, int fillMode) {
    (void)descr;
    (void)fillMode;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatIndexBase(void *descr, int base) {
    (void)descr;
    (void)base;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatType(void *descr, int type) {
    (void)descr;
    (void)type;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateIdentityPermutation(void *handle, int n, int *p) {
    (void)handle;
    if (p) {
        for (int i = 0; i < n; ++i) {
            p[i] = i;
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCsrSetPointers(void *spMatDescr, void *csrOffsets, void *csrColumns, void *csrValues) {
    (void)spMatDescr;
    (void)csrOffsets;
    (void)csrColumns;
    (void)csrValues;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCsrSetStridedBatch(void *spMatDescr, int batchCount, int64_t offsetsBatchStride, int64_t columnsValuesBatchStride) {
    (void)spMatDescr;
    (void)batchCount;
    (void)offsetsBatchStride;
    (void)columnsValuesBatchStride;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatSetStridedBatch(void *dnMatDescr, int batchCount, int64_t batchStride) {
    (void)dnMatDescr;
    (void)batchCount;
    (void)batchStride;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetSize(void *spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz) {
    (void)spMatDescr;
    if (rows) *rows = 0;
    if (cols) *cols = 0;
    if (nnz) *nnz = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatSetAttribute(void *spMatDescr, int attribute, void *data, size_t dataSize) {
    (void)spMatDescr;
    (void)attribute;
    (void)data;
    (void)dataSize;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXbsrsm2_zeroPivot(void *handle, void *info, int *position) {
    (void)handle;
    (void)info;
    if (position) *position = -1;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXbsrsv2_zeroPivot(void *handle, void *info, int *position) {
    (void)handle;
    (void)info;
    if (position) *position = -1;
    return CUSPARSE_STATUS_SUCCESS;
}

#define HETGPU_CUSPARSE_STATUS_STUB(name) \
    cusparseStatus_t name() { \
        HETGPU_CUSPARSE_LOG("[hetGPU cusparse_shim] " #name "\n"); \
        return CUSPARSE_STATUS_SUCCESS; \
    }

HETGPU_CUSPARSE_STATUS_STUB(cusparseCbsrmm)
HETGPU_CUSPARSE_STATUS_STUB(cusparseCbsrmv)
HETGPU_CUSPARSE_STATUS_STUB(cusparseCbsrsm2_analysis)
HETGPU_CUSPARSE_STATUS_STUB(cusparseCbsrsm2_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseCbsrsm2_solve)
HETGPU_CUSPARSE_STATUS_STUB(cusparseCbsrsv2_analysis)
HETGPU_CUSPARSE_STATUS_STUB(cusparseCbsrsv2_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseCbsrsv2_solve)
HETGPU_CUSPARSE_STATUS_STUB(cusparseCcsrgeam2)
HETGPU_CUSPARSE_STATUS_STUB(cusparseCcsrgeam2_bufferSizeExt)
HETGPU_CUSPARSE_STATUS_STUB(cusparseDbsrmm)
HETGPU_CUSPARSE_STATUS_STUB(cusparseDbsrmv)
HETGPU_CUSPARSE_STATUS_STUB(cusparseDbsrsm2_analysis)
HETGPU_CUSPARSE_STATUS_STUB(cusparseDbsrsm2_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseDbsrsm2_solve)
HETGPU_CUSPARSE_STATUS_STUB(cusparseDbsrsv2_analysis)
HETGPU_CUSPARSE_STATUS_STUB(cusparseDbsrsv2_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseDbsrsv2_solve)
HETGPU_CUSPARSE_STATUS_STUB(cusparseDcsrgeam2)
HETGPU_CUSPARSE_STATUS_STUB(cusparseDcsrgeam2_bufferSizeExt)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSDDMM)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSDDMM_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSDDMM_preprocess)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSbsrmm)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSbsrmv)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSbsrsm2_analysis)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSbsrsm2_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSbsrsm2_solve)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSbsrsv2_analysis)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSbsrsv2_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSbsrsv2_solve)
HETGPU_CUSPARSE_STATUS_STUB(cusparseScsrgeam2)
HETGPU_CUSPARSE_STATUS_STUB(cusparseScsrgeam2_bufferSizeExt)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpGEMM_compute)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpGEMM_copy)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpGEMM_workEstimation)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpMM)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpMM_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpMV)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpMV_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpSM_analysis)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpSM_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpSM_solve)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpSV_analysis)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpSV_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseSpSV_solve)
HETGPU_CUSPARSE_STATUS_STUB(cusparseXcoo2csr)
HETGPU_CUSPARSE_STATUS_STUB(cusparseXcoosortByRow)
HETGPU_CUSPARSE_STATUS_STUB(cusparseXcoosort_bufferSizeExt)
HETGPU_CUSPARSE_STATUS_STUB(cusparseXcsrgeam2Nnz)
HETGPU_CUSPARSE_STATUS_STUB(cusparseXcsrsort)
HETGPU_CUSPARSE_STATUS_STUB(cusparseXcsrsort_bufferSizeExt)
HETGPU_CUSPARSE_STATUS_STUB(cusparseZbsrmm)
HETGPU_CUSPARSE_STATUS_STUB(cusparseZbsrmv)
HETGPU_CUSPARSE_STATUS_STUB(cusparseZbsrsm2_analysis)
HETGPU_CUSPARSE_STATUS_STUB(cusparseZbsrsm2_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseZbsrsm2_solve)
HETGPU_CUSPARSE_STATUS_STUB(cusparseZbsrsv2_analysis)
HETGPU_CUSPARSE_STATUS_STUB(cusparseZbsrsv2_bufferSize)
HETGPU_CUSPARSE_STATUS_STUB(cusparseZbsrsv2_solve)
HETGPU_CUSPARSE_STATUS_STUB(cusparseZcsrgeam2)
HETGPU_CUSPARSE_STATUS_STUB(cusparseZcsrgeam2_bufferSizeExt)
