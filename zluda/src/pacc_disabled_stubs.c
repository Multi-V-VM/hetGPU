#include <stddef.h>

int hetgpu_pacc_is_device_ptr(const void *ptr)
{
    (void)ptr;
    return 0;
}

unsigned long long hetgpu_pacc_resolve_device_addr(const void *ptr)
{
    return (unsigned long long)(size_t)ptr;
}

size_t hetgpu_pacc_allocation_remaining(const void *ptr)
{
    (void)ptr;
    return 0;
}

int hetgpu_pacc_ipc_get_mem_handle(const void *ptr, void *handle, size_t handle_len)
{
    (void)ptr;
    (void)handle;
    (void)handle_len;
    return -1;
}

int hetgpu_pacc_ipc_open_mem_handle(void **devPtr, const void *handle, unsigned int flags)
{
    (void)devPtr;
    (void)handle;
    (void)flags;
    return -1;
}

int hetgpu_pacc_ipc_close_mem_handle(void *devPtr)
{
    (void)devPtr;
    return -1;
}

int hetgpu_pacc_submit_gemm(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType)
{
    (void)transa; (void)transb; (void)m; (void)n; (void)k;
    (void)alpha; (void)A; (void)Atype; (void)lda; (void)strideA;
    (void)B; (void)Btype; (void)ldb; (void)strideB;
    (void)beta; (void)C; (void)Ctype; (void)ldc; (void)strideC;
    (void)batchCount; (void)computeType;
    return -1;
}

int hetgpu_pacc_submit_gemm_staged(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType)
{
    return hetgpu_pacc_submit_gemm(transa, transb, m, n, k, alpha,
        A, Atype, lda, strideA, B, Btype, ldb, strideB, beta,
        C, Ctype, ldc, strideC, batchCount, computeType);
}

int hetgpu_pacc_submit_gemm_staged_on(
    int dev_id, int slot_id,
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType)
{
    (void)dev_id;
    (void)slot_id;
    return hetgpu_pacc_submit_gemm(transa, transb, m, n, k, alpha,
        A, Atype, lda, strideA, B, Btype, ldb, strideB, beta,
        C, Ctype, ldc, strideC, batchCount, computeType);
}

int hetgpu_pacc_submit_gemm_staged_tiled(
    int transa, int transb, int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType,
    int max_m, int max_n, int max_k)
{
    (void)max_m;
    (void)max_n;
    (void)max_k;
    return hetgpu_pacc_submit_gemm(transa, transb, m, n, k, alpha,
        A, Atype, lda, strideA, B, Btype, ldb, strideB, beta,
        C, Ctype, ldc, strideC, batchCount, computeType);
}

int hetgpu_pacc_nccl_all_reduce_f32(
    const float *sendbuff, float *recvbuff, size_t count, int op, int rank, int nranks)
{
    (void)sendbuff;
    (void)recvbuff;
    (void)count;
    (void)op;
    (void)rank;
    (void)nranks;
    return -1;
}

int hetgpu_pacc_nccl_reduce_sum_f32(
    const float *sendbuff, float *recvbuff, size_t count, int root, int rank, int nranks)
{
    (void)sendbuff;
    (void)recvbuff;
    (void)count;
    (void)root;
    (void)rank;
    (void)nranks;
    return -1;
}
