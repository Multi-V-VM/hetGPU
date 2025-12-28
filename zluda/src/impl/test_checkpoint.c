// Simple CUDA driver API test for checkpoint functionality
// Compile: gcc -o test_checkpoint test_checkpoint.c -ldl -lpthread
// Run: LD_PRELOAD=./target/debug/libnvcuda.so ./test_checkpoint

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <stdint.h>

// CUDA types
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef uint64_t CUdeviceptr;

#define CUDA_SUCCESS 0

// Function pointers for CUDA API
typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (*cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemsetD32_t)(CUdeviceptr, unsigned int, size_t);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void*, size_t);
typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int,
                                      unsigned int, CUstream, void**, void**);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuCtxDestroy_t)(CUcontext);

// hetGPU checkpoint FFI structures and functions
typedef struct {
    int error_code;
    int recompiled;
    uint32_t num_memory_mappings;
    uint32_t num_remapped_args;
    uint32_t grid_dim_x, grid_dim_y, grid_dim_z;
    uint32_t block_dim_x, block_dim_y, block_dim_z;
    uint32_t shared_mem_bytes;
    uint32_t resume_ptx_line;
    uint64_t resume_instruction_offset;
} CRestoreResult;

typedef int (*hetgpu_checkpoint_load_t)(const char* path);
typedef int (*hetgpu_checkpoint_restore_t)(CRestoreResult* result);
typedef const char* (*hetgpu_checkpoint_get_ptx_t)(void);
typedef const char* (*hetgpu_checkpoint_get_kernel_name_t)(void);
typedef uint32_t (*hetgpu_checkpoint_get_memory_region_count_t)(void);
typedef int (*hetgpu_checkpoint_get_memory_mapping_t)(uint32_t index, uint64_t* original_addr, uint64_t* size);
typedef int (*hetgpu_checkpoint_get_memory_data_t)(uint32_t index, uint8_t* buffer, uint64_t buffer_size);

// PTX source for a simple kernel
const char* ptx_source =
".version 7.5\n"
".target sm_61\n"
".address_size 64\n"
"\n"
".visible .entry test_kernel(\n"
"    .param .u64 data,\n"
"    .param .s32 n\n"
")\n"
"{\n"
"    .reg .pred %p<2>;\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<5>;\n"
"\n"
"    ld.param.u64 %rd1, [data];\n"
"    ld.param.s32 %r1, [n];\n"
"\n"
"    mov.u32 %r2, %ctaid.x;\n"
"    mov.u32 %r3, %ntid.x;\n"
"    mov.u32 %r4, %tid.x;\n"
"    mad.lo.s32 %r5, %r2, %r3, %r4;\n"
"\n"
"    setp.ge.s32 %p1, %r5, %r1;\n"
"    @%p1 bra END;\n"
"\n"
"    // Store thread index\n"
"    mul.wide.s32 %rd2, %r5, 4;\n"
"    add.s64 %rd3, %rd1, %rd2;\n"
"    st.global.s32 [%rd3], %r5;\n"
"\n"
"END:\n"
"    ret;\n"
"}\n";

// Global for tracking checkpoint file
static char last_checkpoint_path[512] = "";

int main(int argc, char** argv) {
    printf("=== hetGPU Checkpoint Test (C version) ===\n");

    // Check for restore mode
    int restore_mode = 0;
    const char* restore_file = NULL;

    if (argc > 1) {
        if (strcmp(argv[1], "--restore") == 0) {
            restore_mode = 1;
            if (argc > 2) {
                restore_file = argv[2];
            } else {
                // Try to find latest checkpoint
                restore_file = "/tmp/hetgpu_checkpoints/hetgpu_checkpoint_1766889160.json";
            }
            printf("Mode: RESTORE from %s\n\n", restore_file);
        } else if (strcmp(argv[1], "--help") == 0) {
            printf("Usage: %s [--restore [checkpoint_file]]\n", argv[0]);
            printf("  No args: Run kernel loop, press Ctrl+C to checkpoint\n");
            printf("  --restore: Restore from latest checkpoint\n");
            printf("  --restore <file>: Restore from specific checkpoint file\n");
            return 0;
        }
    }

    if (!restore_mode) {
        printf("Mode: CHECKPOINT (press Ctrl+C to trigger)\n\n");
    }

    // Load CUDA library
    // First try to load from LD_PRELOAD or custom path
    const char* lib_path = getenv("HETGPU_CUDA_LIB");
    void* cuda_lib = NULL;

    if (lib_path) {
        cuda_lib = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
        if (cuda_lib) {
            printf("[OK] Loaded CUDA library from: %s\n", lib_path);
        }
    }

    // Try hetGPU library path (release first, then debug)
    if (!cuda_lib) {
        cuda_lib = dlopen("/home/victoryang00/hetGPU/target/release/libnvcuda.so", RTLD_NOW | RTLD_GLOBAL);
        if (cuda_lib) {
            printf("[OK] Loaded hetGPU CUDA library (release)\n");
        }
    }
    if (!cuda_lib) {
        cuda_lib = dlopen("/home/victoryang00/hetGPU/target/debug/libnvcuda.so", RTLD_NOW | RTLD_GLOBAL);
        if (cuda_lib) {
            printf("[OK] Loaded hetGPU CUDA library (debug)\n");
        }
    }

    // Fallback to system CUDA
    if (!cuda_lib) {
        cuda_lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!cuda_lib) {
            cuda_lib = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        }
        if (cuda_lib) {
            printf("[OK] Loaded system CUDA library\n");
        }
    }

    if (!cuda_lib) {
        printf("Failed to load any CUDA library: %s\n", dlerror());
        return 1;
    }

    // Get function pointers
    cuInit_t cuInit = (cuInit_t)dlsym(cuda_lib, "cuInit");
    cuDeviceGet_t cuDeviceGet = (cuDeviceGet_t)dlsym(cuda_lib, "cuDeviceGet");
    cuCtxCreate_t cuCtxCreate = (cuCtxCreate_t)dlsym(cuda_lib, "cuCtxCreate_v2");
    cuMemAlloc_t cuMemAlloc = (cuMemAlloc_t)dlsym(cuda_lib, "cuMemAlloc_v2");
    cuMemsetD32_t cuMemsetD32 = (cuMemsetD32_t)dlsym(cuda_lib, "cuMemsetD32_v2");
    cuModuleLoadData_t cuModuleLoadData = (cuModuleLoadData_t)dlsym(cuda_lib, "cuModuleLoadData");
    cuModuleGetFunction_t cuModuleGetFunction = (cuModuleGetFunction_t)dlsym(cuda_lib, "cuModuleGetFunction");
    cuLaunchKernel_t cuLaunchKernel = (cuLaunchKernel_t)dlsym(cuda_lib, "cuLaunchKernel");
    cuCtxSynchronize_t cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(cuda_lib, "cuCtxSynchronize");
    cuMemFree_t cuMemFree = (cuMemFree_t)dlsym(cuda_lib, "cuMemFree_v2");
    cuCtxDestroy_t cuCtxDestroy = (cuCtxDestroy_t)dlsym(cuda_lib, "cuCtxDestroy_v2");
    cuMemcpyHtoD_t cuMemcpyHtoD = (cuMemcpyHtoD_t)dlsym(cuda_lib, "cuMemcpyHtoD_v2");

    if (!cuInit || !cuDeviceGet || !cuCtxCreate) {
        printf("Failed to get CUDA function pointers\n");
        return 1;
    }

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
    printf("[OK] cuDeviceGet (device=%d)\n", device);

    // Create context
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        printf("cuCtxCreate failed: %d\n", result);
        return 1;
    }
    printf("[OK] cuCtxCreate\n");

    // Allocate device memory
    CUdeviceptr d_data = 0;
    int n = 1024;

    if (cuMemAlloc) {
        result = cuMemAlloc(&d_data, n * sizeof(int));
        if (result != CUDA_SUCCESS) {
            printf("cuMemAlloc failed: %d (continuing anyway)\n", result);
        } else {
            printf("[OK] cuMemAlloc (%d ints at 0x%lx)\n", n, (unsigned long)d_data);
        }
    }

    // Initialize to zeros
    if (cuMemsetD32 && d_data) {
        result = cuMemsetD32(d_data, 0, n);
        if (result != CUDA_SUCCESS) {
            printf("cuMemsetD32 failed: %d\n", result);
        } else {
            printf("[OK] cuMemsetD32\n");
        }
    }

    // Load PTX module
    if (cuModuleLoadData) {
        result = cuModuleLoadData(&module, ptx_source);
        if (result != CUDA_SUCCESS) {
            printf("cuModuleLoadData failed: %d (continuing anyway)\n", result);
            module = NULL;
        } else {
            printf("[OK] cuModuleLoadData\n");
        }
    }

    // Get kernel function
    if (cuModuleGetFunction && module) {
        result = cuModuleGetFunction(&kernel, module, "test_kernel");
        if (result != CUDA_SUCCESS) {
            printf("cuModuleGetFunction failed: %d\n", result);
            kernel = NULL;
        } else {
            printf("[OK] cuModuleGetFunction\n");
        }
    }

    // Handle restore mode
    if (restore_mode && restore_file) {
        printf("\n=== RESTORE MODE ===\n");
        printf("Checkpoint file: %s\n\n", restore_file);

        // Get FFI functions from library
        hetgpu_checkpoint_load_t hetgpu_checkpoint_load =
            (hetgpu_checkpoint_load_t)dlsym(cuda_lib, "hetgpu_checkpoint_load");
        hetgpu_checkpoint_restore_t hetgpu_checkpoint_restore =
            (hetgpu_checkpoint_restore_t)dlsym(cuda_lib, "hetgpu_checkpoint_restore");
        hetgpu_checkpoint_get_ptx_t hetgpu_checkpoint_get_ptx =
            (hetgpu_checkpoint_get_ptx_t)dlsym(cuda_lib, "hetgpu_checkpoint_get_ptx");
        hetgpu_checkpoint_get_kernel_name_t hetgpu_checkpoint_get_kernel_name =
            (hetgpu_checkpoint_get_kernel_name_t)dlsym(cuda_lib, "hetgpu_checkpoint_get_kernel_name");
        hetgpu_checkpoint_get_memory_region_count_t hetgpu_checkpoint_get_memory_region_count =
            (hetgpu_checkpoint_get_memory_region_count_t)dlsym(cuda_lib, "hetgpu_checkpoint_get_memory_region_count");
        hetgpu_checkpoint_get_memory_mapping_t hetgpu_checkpoint_get_memory_mapping =
            (hetgpu_checkpoint_get_memory_mapping_t)dlsym(cuda_lib, "hetgpu_checkpoint_get_memory_mapping");
        hetgpu_checkpoint_get_memory_data_t hetgpu_checkpoint_get_memory_data =
            (hetgpu_checkpoint_get_memory_data_t)dlsym(cuda_lib, "hetgpu_checkpoint_get_memory_data");

        if (!hetgpu_checkpoint_load || !hetgpu_checkpoint_restore) {
            printf("ERROR: Failed to get checkpoint FFI functions\n");
            printf("  hetgpu_checkpoint_load: %p\n", (void*)hetgpu_checkpoint_load);
            printf("  hetgpu_checkpoint_restore: %p\n", (void*)hetgpu_checkpoint_restore);
            return 1;
        }
        printf("[OK] Loaded checkpoint FFI functions\n");

        // Step 1: Load checkpoint file
        printf("\n[Restore] Step 1: Loading checkpoint...\n");
        int load_result = hetgpu_checkpoint_load(restore_file);
        if (load_result != 0) {
            printf("ERROR: Failed to load checkpoint (error %d)\n", load_result);
            printf("\nAvailable checkpoints:\n");
            system("ls -la /tmp/hetgpu_checkpoints/");
            return 1;
        }
        printf("[Restore] Checkpoint loaded successfully\n");

        // Step 2: Get checkpoint info
        printf("\n[Restore] Step 2: Analyzing checkpoint...\n");

        const char* saved_ptx = hetgpu_checkpoint_get_ptx ? hetgpu_checkpoint_get_ptx() : NULL;
        const char* kernel_name = hetgpu_checkpoint_get_kernel_name ? hetgpu_checkpoint_get_kernel_name() : NULL;
        uint32_t memory_region_count = hetgpu_checkpoint_get_memory_region_count ?
            hetgpu_checkpoint_get_memory_region_count() : 0;

        printf("  PTX source: %s\n", saved_ptx ? "available" : "not available");
        printf("  Kernel name: %s\n", kernel_name ? kernel_name : "(none)");
        printf("  Memory regions: %u\n", memory_region_count);

        // Step 3: Show memory regions
        if (memory_region_count > 0 && hetgpu_checkpoint_get_memory_mapping) {
            printf("\n[Restore] Step 3: Memory regions to restore:\n");
            for (uint32_t i = 0; i < memory_region_count && i < 10; i++) {
                uint64_t orig_addr = 0, size = 0;
                if (hetgpu_checkpoint_get_memory_mapping(i, &orig_addr, &size) == 0) {
                    printf("  Region %u: addr=0x%016lx size=%lu bytes\n",
                           i, (unsigned long)orig_addr, (unsigned long)size);
                }
            }
        }

        // Step 4: Perform actual restore
        printf("\n[Restore] Step 4: Performing GPU state restore...\n");
        CRestoreResult restore_result = {0};
        int restore_ret = hetgpu_checkpoint_restore(&restore_result);

        if (restore_ret != 0) {
            printf("ERROR: Restore failed (error %d)\n", restore_ret);
        } else {
            printf("[Restore] Restore completed successfully!\n");
            printf("  Recompiled: %s\n", restore_result.recompiled ? "yes" : "no");
            printf("  Memory mappings: %u\n", restore_result.num_memory_mappings);
            printf("  Remapped args: %u\n", restore_result.num_remapped_args);
            printf("  Grid dim: (%u, %u, %u)\n",
                   restore_result.grid_dim_x, restore_result.grid_dim_y, restore_result.grid_dim_z);
            printf("  Block dim: (%u, %u, %u)\n",
                   restore_result.block_dim_x, restore_result.block_dim_y, restore_result.block_dim_z);
            printf("  Shared mem: %u bytes\n", restore_result.shared_mem_bytes);
            if (restore_result.resume_ptx_line > 0) {
                printf("  Resume at PTX line %u, offset 0x%lx\n",
                       restore_result.resume_ptx_line,
                       (unsigned long)restore_result.resume_instruction_offset);
            }
        }

        // Step 5: If we have PTX, recompile for current backend
        if (saved_ptx) {
            printf("\n[Restore] Step 5: Recompiling PTX for current backend...\n");
            printf("  PTX size: %zu bytes\n", strlen(saved_ptx));

            // Load the PTX module
            if (cuModuleLoadData) {
                CUmodule restored_module;
                result = cuModuleLoadData(&restored_module, saved_ptx);
                if (result == CUDA_SUCCESS) {
                    printf("[Restore] PTX recompiled successfully\n");

                    // Get kernel function if we have the name
                    if (kernel_name && cuModuleGetFunction) {
                        CUfunction restored_kernel;
                        result = cuModuleGetFunction(&restored_kernel, restored_module, kernel_name);
                        if (result == CUDA_SUCCESS) {
                            printf("[Restore] Kernel '%s' loaded\n", kernel_name);
                            // Use restored kernel for verification
                            module = restored_module;
                            kernel = restored_kernel;
                        }
                    }
                } else {
                    printf("[Restore] PTX recompilation failed: %d\n", result);
                }
            }
        }

        // Step 6: Reallocate and restore memory regions
        if (memory_region_count > 0 && cuMemAlloc && cuMemcpyHtoD && hetgpu_checkpoint_get_memory_data) {
            printf("\n[Restore] Step 6: Restoring memory regions...\n");

            for (uint32_t i = 0; i < memory_region_count; i++) {
                uint64_t orig_addr = 0, size = 0;
                if (hetgpu_checkpoint_get_memory_mapping(i, &orig_addr, &size) == 0 && size > 0) {
                    // Allocate new memory
                    CUdeviceptr new_addr = 0;
                    result = cuMemAlloc(&new_addr, size);
                    if (result == CUDA_SUCCESS) {
                        printf("  Region %u: allocated %lu bytes at 0x%lx (was 0x%lx)\n",
                               i, (unsigned long)size, (unsigned long)new_addr, (unsigned long)orig_addr);

                        // Copy data from checkpoint
                        uint8_t* buffer = malloc(size);
                        if (buffer) {
                            int copied = hetgpu_checkpoint_get_memory_data(i, buffer, size);
                            if (copied > 0) {
                                cuMemcpyHtoD(new_addr, buffer, copied);
                                printf("    Restored %d bytes of data\n", copied);
                            }
                            free(buffer);
                        }
                    }
                }
            }
        }

        printf("\n=== Running post-restore verification ===\n");
        printf("Running 3 kernel iterations to verify restored state...\n\n");
    } else {
        printf("\n=== Starting kernel loop (press Ctrl+C to checkpoint) ===\n\n");
    }

    int iteration = 0;
    int max_iterations = restore_mode ? 3 : 0; // 0 = infinite for checkpoint mode

    while (max_iterations == 0 || iteration < max_iterations) {
        iteration++;
        printf("Launching kernel iteration %d...\n", iteration);

        if (cuLaunchKernel && kernel) {
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
                printf("  cuLaunchKernel failed: %d\n", result);
            } else {
                printf("  cuLaunchKernel OK\n");
            }
        }

        // Synchronize
        if (cuCtxSynchronize) {
            result = cuCtxSynchronize();
            if (result != CUDA_SUCCESS) {
                printf("  cuCtxSynchronize failed: %d\n", result);
            } else {
                printf("  cuCtxSynchronize OK\n");
            }
        }

        printf("  Iteration %d complete\n\n", iteration);

        // Small delay between launches (shorter in restore mode)
        if (!restore_mode) {
            usleep(500000); // 500ms
        } else {
            usleep(100000); // 100ms for restore verification
        }
    }

    // Show completion message
    if (restore_mode) {
        printf("=== Restore verification complete ===\n");
        printf("Successfully executed %d kernel iterations after restore.\n", iteration);
    }

    // Cleanup
    printf("\nCleaning up...\n");
    if (cuMemFree && d_data) cuMemFree(d_data);
    if (cuCtxDestroy) cuCtxDestroy(context);
    dlclose(cuda_lib);

    printf("Done.\n");
    return 0;
}
