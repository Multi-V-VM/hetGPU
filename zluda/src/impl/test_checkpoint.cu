// Test program for checkpoint functionality
// Compile: nvcc -o test_checkpoint test_checkpoint.cu -lcuda
// Run with hetGPU: LD_PRELOAD=/path/to/libnvcuda.so ./test_checkpoint

#include <cuda.h>
#include <stdio.h>
#include <unistd.h>

// Simple kernel that runs for a while
__global__ void infinite_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate some work
        for (int i = 0; i < 1000000; i++) {
            data[idx] = data[idx] + 1;
            data[idx] = data[idx] - 1;
        }
        data[idx] = idx;
    }
}

int main() {
    printf("=== hetGPU Checkpoint Test ===\n");
    printf("Press Ctrl+C to trigger checkpoint during kernel execution\n\n");

    CUresult result;
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;

    // Initialize CUDA driver API
    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        printf("cuInit failed: %d\n", result);
        return 1;
    }
    printf("[OK] cuInit\n");

    // Get device
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        printf("cuDeviceGet failed: %d\n", result);
        return 1;
    }
    printf("[OK] cuDeviceGet\n");

    // Create context
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        printf("cuCtxCreate failed: %d\n", result);
        return 1;
    }
    printf("[OK] cuCtxCreate\n");

    // Allocate device memory
    CUdeviceptr d_data;
    int n = 1024;
    result = cuMemAlloc(&d_data, n * sizeof(int));
    if (result != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: %d\n", result);
        return 1;
    }
    printf("[OK] cuMemAlloc (%d ints)\n", n);

    // Initialize to zeros
    result = cuMemsetD32(d_data, 0, n);
    if (result != CUDA_SUCCESS) {
        printf("cuMemsetD32 failed: %d\n", result);
        return 1;
    }
    printf("[OK] cuMemsetD32\n");

    // Load PTX module
    const char* ptx_source = R"(
.version 7.5
.target sm_61
.address_size 64

.visible .entry infinite_kernel(
    .param .u64 data,
    .param .s32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [data];
    ld.param.s32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;

    setp.ge.s32 %p1, %r5, %r1;
    @%p1 bra END;

    // Simple loop
    mov.s32 %r6, 0;
LOOP:
    setp.ge.s32 %p1, %r6, 1000;
    @%p1 bra DONE;
    add.s32 %r6, %r6, 1;
    bra LOOP;

DONE:
    // Store result
    mul.wide.s32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.s32 [%rd3], %r5;

END:
    ret;
}
)";

    result = cuModuleLoadData(&module, ptx_source);
    if (result != CUDA_SUCCESS) {
        printf("cuModuleLoadData failed: %d\n", result);
        // Continue anyway for testing
    } else {
        printf("[OK] cuModuleLoadData\n");
    }

    // Get kernel function
    result = cuModuleGetFunction(&kernel, module, "infinite_kernel");
    if (result != CUDA_SUCCESS) {
        printf("cuModuleGetFunction failed: %d\n", result);
        // Continue anyway
    } else {
        printf("[OK] cuModuleGetFunction\n");
    }

    printf("\n=== Starting infinite loop of kernel launches ===\n");
    printf("Press Ctrl+C to trigger checkpoint...\n\n");

    int iteration = 0;
    while (1) {
        iteration++;
        printf("Launching kernel iteration %d...\n", iteration);

        // Set up kernel parameters
        void* args[] = { &d_data, &n };

        // Launch kernel
        result = cuLaunchKernel(
            kernel,
            (n + 255) / 256, 1, 1,  // grid
            256, 1, 1,              // block
            0,                      // shared mem
            NULL,                   // stream
            args,                   // args
            NULL                    // extra
        );

        if (result != CUDA_SUCCESS) {
            printf("cuLaunchKernel failed: %d\n", result);
        }

        // Synchronize
        result = cuCtxSynchronize();
        if (result != CUDA_SUCCESS) {
            printf("cuCtxSynchronize failed: %d\n", result);
        }

        printf("  Iteration %d complete\n", iteration);

        // Small delay between launches
        usleep(100000); // 100ms
    }

    // Cleanup (never reached in infinite loop)
    cuMemFree(d_data);
    cuCtxDestroy(context);

    return 0;
}
