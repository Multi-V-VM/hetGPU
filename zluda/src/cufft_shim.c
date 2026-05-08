#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#if defined(HETGPU_DEBUG_LOGS)
#define HETGPU_CUFFT_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
#define HETGPU_CUFFT_LOG(...) ((void)0)
#endif

typedef int cufftResult;
typedef int cufftHandle;

enum {
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
};

cufftResult cufftCreate(cufftHandle *plan) {
    HETGPU_CUFFT_LOG("[hetGPU cufft_shim] cufftCreate\n");
    if (!plan) {
        return CUFFT_INVALID_PLAN;
    }
    static cufftHandle next_plan = 1;
    *plan = next_plan++;
    return CUFFT_SUCCESS;
}

cufftResult cufftDestroy(cufftHandle plan) {
    (void)plan;
    HETGPU_CUFFT_LOG("[hetGPU cufft_shim] cufftDestroy\n");
    return CUFFT_SUCCESS;
}

cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) {
    (void)plan;
    (void)autoAllocate;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetStream(cufftHandle plan, void *stream) {
    (void)plan;
    (void)stream;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetWorkArea(cufftHandle plan, void *workArea) {
    (void)plan;
    (void)workArea;
    return CUFFT_SUCCESS;
}

cufftResult cufftXtMakePlanMany(cufftHandle plan,
                                int rank,
                                long long int *n,
                                long long int *inembed,
                                long long int istride,
                                long long int idist,
                                int inputtype,
                                long long int *onembed,
                                long long int ostride,
                                long long int odist,
                                int outputtype,
                                long long int batch,
                                size_t *workSize,
                                int executiontype) {
    (void)plan;
    (void)rank;
    (void)n;
    (void)inembed;
    (void)istride;
    (void)idist;
    (void)inputtype;
    (void)onembed;
    (void)ostride;
    (void)odist;
    (void)outputtype;
    (void)batch;
    (void)executiontype;
    if (workSize) {
        *workSize = 0;
    }
    return CUFFT_SUCCESS;
}

cufftResult cufftXtExec(cufftHandle plan, void *input, void *output, int direction) {
    (void)plan;
    (void)input;
    (void)output;
    (void)direction;
    return CUFFT_SUCCESS;
}
