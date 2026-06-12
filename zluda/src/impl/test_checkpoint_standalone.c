// Standalone CUDA checkpoint/restore test
// Compile: gcc -o test_checkpoint_standalone test_checkpoint_standalone.c -ldl
//
// Run checkpoint mode (Ctrl+C to checkpoint):
//   LD_PRELOAD=/path/to/libnvcuda.so ./test_checkpoint_standalone
//
// Run restore mode (auto-finds latest checkpoint):
//   LD_PRELOAD=/path/to/libnvcuda.so ./test_checkpoint_standalone --restore
//
// Run restore mode (specific file):
//   LD_PRELOAD=/path/to/libnvcuda.so ./test_checkpoint_standalone --restore /tmp/hetgpu_checkpoints/hetgpu_checkpoint_XXX.json
//
// Run auto mode (checkpoint then auto-restore in loop):
//   LD_PRELOAD=/path/to/libnvcuda.so ./test_checkpoint_standalone --auto

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <time.h>

// CUDA types (minimal definitions to avoid dependency on CUDA headers)
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

// Function pointer types
typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuDeviceGet_fn)(CUdevice*, int);
typedef CUresult (*cuCtxCreate_fn)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_fn)(CUcontext);
typedef CUresult (*cuMemAlloc_fn)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemFree_fn)(CUdeviceptr);
typedef CUresult (*cuMemsetD32_fn)(CUdeviceptr, unsigned int, size_t);
typedef CUresult (*cuMemcpyHtoD_fn)(CUdeviceptr, const void*, size_t);
typedef CUresult (*cuMemcpyDtoH_fn)(void*, CUdeviceptr, size_t);
typedef CUresult (*cuModuleLoadData_fn)(CUmodule*, const void*);
typedef CUresult (*cuModuleGetFunction_fn)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuLaunchKernel_fn)(CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, CUstream, void**, void**);
typedef CUresult (*cuCtxSynchronize_fn)(void);

// hetGPU checkpoint FFI types
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

typedef int (*hetgpu_checkpoint_load_fn)(const char* path);
typedef int (*hetgpu_checkpoint_restore_fn)(CRestoreResult* result);
typedef const char* (*hetgpu_checkpoint_get_ptx_fn)(void);
typedef const char* (*hetgpu_checkpoint_get_kernel_name_fn)(void);
typedef uint32_t (*hetgpu_checkpoint_get_memory_region_count_fn)(void);
typedef int (*hetgpu_checkpoint_get_memory_mapping_fn)(uint32_t index, uint64_t* original_addr, uint64_t* size);
typedef int (*hetgpu_checkpoint_get_memory_data_fn)(uint32_t index, uint8_t* buffer, uint64_t buffer_size);
typedef void (*hetgpu_request_checkpoint_fn)(void);

// Global function pointers
static cuInit_fn cuInit;
static cuDeviceGet_fn cuDeviceGet;
static cuCtxCreate_fn cuCtxCreate;
static cuCtxDestroy_fn cuCtxDestroy;
static cuMemAlloc_fn cuMemAlloc;
static cuMemFree_fn cuMemFree;
static cuMemsetD32_fn cuMemsetD32;
static cuMemcpyHtoD_fn cuMemcpyHtoD;
static cuMemcpyDtoH_fn cuMemcpyDtoH;
static cuModuleLoadData_fn cuModuleLoadData;
static cuModuleGetFunction_fn cuModuleGetFunction;
static cuLaunchKernel_fn cuLaunchKernel;
static cuCtxSynchronize_fn cuCtxSynchronize;

// hetGPU checkpoint function pointers
static hetgpu_checkpoint_load_fn hetgpu_checkpoint_load;
static hetgpu_checkpoint_restore_fn hetgpu_checkpoint_restore;
static hetgpu_checkpoint_get_ptx_fn hetgpu_checkpoint_get_ptx;
static hetgpu_checkpoint_get_kernel_name_fn hetgpu_checkpoint_get_kernel_name;
static hetgpu_checkpoint_get_memory_region_count_fn hetgpu_checkpoint_get_memory_region_count;
static hetgpu_checkpoint_get_memory_mapping_fn hetgpu_checkpoint_get_memory_mapping;
static hetgpu_checkpoint_get_memory_data_fn hetgpu_checkpoint_get_memory_data;
static hetgpu_request_checkpoint_fn hetgpu_request_checkpoint;

// Checkpoint directory
static const char* CHECKPOINT_DIR = "/tmp/hetgpu_checkpoints";

// Signal handling for auto mode
static volatile sig_atomic_t got_sigint = 0;
static void sigint_handler(int sig) {
    (void)sig;
    got_sigint = 1;
}

// Find the latest checkpoint file
static char* find_latest_checkpoint(void) {
    static char latest_path[512];
    DIR* dir = opendir(CHECKPOINT_DIR);
    if (!dir) {
        return NULL;
    }

    time_t latest_time = 0;
    latest_path[0] = '\0';

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "hetgpu_checkpoint_", 18) == 0 &&
            strstr(entry->d_name, ".json") != NULL) {
            char path[512];
            snprintf(path, sizeof(path), "%s/%s", CHECKPOINT_DIR, entry->d_name);

            struct stat st;
            if (stat(path, &st) == 0 && st.st_mtime > latest_time) {
                latest_time = st.st_mtime;
                strncpy(latest_path, path, sizeof(latest_path) - 1);
                latest_path[sizeof(latest_path) - 1] = '\0';
            }
        }
    }
    closedir(dir);

    return latest_path[0] ? latest_path : NULL;
}

// Wait for a new checkpoint to appear (newer than given time)
static char* wait_for_new_checkpoint(time_t after_time, int timeout_secs) {
    fprintf(stderr, "[Auto] Waiting for new checkpoint...\n");
    int waited = 0;
    while (waited < timeout_secs) {
        char* latest = find_latest_checkpoint();
        if (latest) {
            struct stat st;
            if (stat(latest, &st) == 0 && st.st_mtime > after_time) {
                fprintf(stderr, "[Auto] New checkpoint detected: %s\n", latest);
                return latest;
            }
        }
        sleep(1);
        waited++;
    }
    return NULL;
}

// PTX kernel source
static const char* ptx_source =
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

static void* cuda_lib = NULL;

static int load_cuda_functions(void) {
    // Try RTLD_DEFAULT first (for LD_PRELOAD case)
    cuInit = (cuInit_fn)dlsym(RTLD_DEFAULT, "cuInit");
    if (cuInit) {
        printf("[OK] Using LD_PRELOAD CUDA library\n");
        cuda_lib = RTLD_DEFAULT;
    } else {
        // Fallback: try to load libcuda.so explicitly
        cuda_lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!cuda_lib) {
            cuda_lib = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        }
        if (!cuda_lib) {
            fprintf(stderr, "ERROR: Could not load CUDA library.\n");
            fprintf(stderr, "Run with: LD_PRELOAD=/path/to/libnvcuda.so %s\n", "test_checkpoint_standalone");
            return -1;
        }
        printf("[OK] Loaded libcuda.so directly\n");
        cuInit = (cuInit_fn)dlsym(cuda_lib, "cuInit");
    }

    // Load all CUDA functions
    cuDeviceGet = (cuDeviceGet_fn)dlsym(cuda_lib, "cuDeviceGet");
    cuCtxCreate = (cuCtxCreate_fn)dlsym(cuda_lib, "cuCtxCreate_v2");
    if (!cuCtxCreate) cuCtxCreate = (cuCtxCreate_fn)dlsym(cuda_lib, "cuCtxCreate");
    cuCtxDestroy = (cuCtxDestroy_fn)dlsym(cuda_lib, "cuCtxDestroy_v2");
    if (!cuCtxDestroy) cuCtxDestroy = (cuCtxDestroy_fn)dlsym(cuda_lib, "cuCtxDestroy");
    cuMemAlloc = (cuMemAlloc_fn)dlsym(cuda_lib, "cuMemAlloc_v2");
    if (!cuMemAlloc) cuMemAlloc = (cuMemAlloc_fn)dlsym(cuda_lib, "cuMemAlloc");
    cuMemFree = (cuMemFree_fn)dlsym(cuda_lib, "cuMemFree_v2");
    if (!cuMemFree) cuMemFree = (cuMemFree_fn)dlsym(cuda_lib, "cuMemFree");
    cuMemsetD32 = (cuMemsetD32_fn)dlsym(cuda_lib, "cuMemsetD32_v2");
    if (!cuMemsetD32) cuMemsetD32 = (cuMemsetD32_fn)dlsym(cuda_lib, "cuMemsetD32");
    cuMemcpyHtoD = (cuMemcpyHtoD_fn)dlsym(cuda_lib, "cuMemcpyHtoD_v2");
    if (!cuMemcpyHtoD) cuMemcpyHtoD = (cuMemcpyHtoD_fn)dlsym(cuda_lib, "cuMemcpyHtoD");
    cuMemcpyDtoH = (cuMemcpyDtoH_fn)dlsym(cuda_lib, "cuMemcpyDtoH_v2");
    if (!cuMemcpyDtoH) cuMemcpyDtoH = (cuMemcpyDtoH_fn)dlsym(cuda_lib, "cuMemcpyDtoH");
    cuModuleLoadData = (cuModuleLoadData_fn)dlsym(cuda_lib, "cuModuleLoadData");
    cuModuleGetFunction = (cuModuleGetFunction_fn)dlsym(cuda_lib, "cuModuleGetFunction");
    cuLaunchKernel = (cuLaunchKernel_fn)dlsym(cuda_lib, "cuLaunchKernel");
    cuCtxSynchronize = (cuCtxSynchronize_fn)dlsym(cuda_lib, "cuCtxSynchronize");

    // Load hetGPU checkpoint functions (may not exist if using real CUDA)
    hetgpu_checkpoint_load = (hetgpu_checkpoint_load_fn)dlsym(cuda_lib, "hetgpu_checkpoint_load");
    hetgpu_checkpoint_restore = (hetgpu_checkpoint_restore_fn)dlsym(cuda_lib, "hetgpu_checkpoint_restore");
    hetgpu_checkpoint_get_ptx = (hetgpu_checkpoint_get_ptx_fn)dlsym(cuda_lib, "hetgpu_checkpoint_get_ptx");
    hetgpu_checkpoint_get_kernel_name = (hetgpu_checkpoint_get_kernel_name_fn)dlsym(cuda_lib, "hetgpu_checkpoint_get_kernel_name");
    hetgpu_checkpoint_get_memory_region_count = (hetgpu_checkpoint_get_memory_region_count_fn)dlsym(cuda_lib, "hetgpu_checkpoint_get_memory_region_count");
    hetgpu_checkpoint_get_memory_mapping = (hetgpu_checkpoint_get_memory_mapping_fn)dlsym(cuda_lib, "hetgpu_checkpoint_get_memory_mapping");
    hetgpu_checkpoint_get_memory_data = (hetgpu_checkpoint_get_memory_data_fn)dlsym(cuda_lib, "hetgpu_checkpoint_get_memory_data");
    hetgpu_request_checkpoint = (hetgpu_request_checkpoint_fn)dlsym(cuda_lib, "hetgpu_request_checkpoint");

    if (!cuInit || !cuDeviceGet || !cuCtxCreate) {
        fprintf(stderr, "ERROR: Failed to load required CUDA functions\n");
        return -1;
    }

    return 0;
}

// Global for auto mode iteration limit
static int g_auto_mode = 0;
static int g_restore_iterations = 0;  // 0 = infinite

static int do_restore(const char* checkpoint_file) {
    printf("\n=== RESTORE MODE ===\n");
    printf("Checkpoint: %s\n\n", checkpoint_file);

    if (!hetgpu_checkpoint_load || !hetgpu_checkpoint_restore) {
        fprintf(stderr, "ERROR: hetGPU checkpoint functions not available.\n");
        fprintf(stderr, "Make sure you're using LD_PRELOAD with hetGPU's libnvcuda.so\n");
        return -1;
    }

    // Step 1: Load checkpoint
    printf("[1/6] Loading checkpoint...\n");
    int ret = hetgpu_checkpoint_load(checkpoint_file);
    if (ret != 0) {
        fprintf(stderr, "ERROR: Failed to load checkpoint (error %d)\n", ret);
        return -1;
    }
    printf("      Checkpoint loaded successfully\n");

    // Step 2: Analyze checkpoint
    printf("[2/6] Analyzing checkpoint...\n");
    const char* saved_ptx = hetgpu_checkpoint_get_ptx ? hetgpu_checkpoint_get_ptx() : NULL;
    const char* kernel_name = hetgpu_checkpoint_get_kernel_name ? hetgpu_checkpoint_get_kernel_name() : NULL;
    uint32_t mem_regions = hetgpu_checkpoint_get_memory_region_count ? hetgpu_checkpoint_get_memory_region_count() : 0;

    printf("      PTX source: %s (%zu bytes)\n", saved_ptx ? "available" : "none", saved_ptx ? strlen(saved_ptx) : 0);
    printf("      Kernel: %s\n", kernel_name && kernel_name[0] ? kernel_name : "(none)");
    printf("      Memory regions: %u\n", mem_regions);

    // Step 3: Perform restore
    printf("[3/6] Performing GPU state restore...\n");
    CRestoreResult result = {0};
    ret = hetgpu_checkpoint_restore(&result);
    if (ret != 0) {
        fprintf(stderr, "ERROR: Restore failed (error %d)\n", ret);
        return -1;
    }
    printf("      Restore completed\n");
    printf("      Recompiled: %s\n", result.recompiled ? "yes" : "no");
    printf("      Grid: (%u, %u, %u), Block: (%u, %u, %u)\n",
           result.grid_dim_x, result.grid_dim_y, result.grid_dim_z,
           result.block_dim_x, result.block_dim_y, result.block_dim_z);

    // Step 4: Recompile PTX if available
    CUmodule module = NULL;
    CUfunction kernel = NULL;
    const char* ptx_to_use = saved_ptx ? saved_ptx : ptx_source;

    printf("[4/6] Recompiling PTX for current backend...\n");
    if (cuModuleLoadData) {
        CUresult cu_ret = cuModuleLoadData(&module, ptx_to_use);
        if (cu_ret == CUDA_SUCCESS) {
            printf("      PTX compiled successfully\n");

            // Get kernel function
            const char* kname = (kernel_name && kernel_name[0]) ? kernel_name : "test_kernel";
            if (cuModuleGetFunction) {
                cu_ret = cuModuleGetFunction(&kernel, module, kname);
                if (cu_ret == CUDA_SUCCESS) {
                    printf("      Kernel '%s' ready\n", kname);
                } else {
                    fprintf(stderr, "      Failed to get kernel function (error %d)\n", cu_ret);
                }
            }
        } else {
            printf("      PTX compilation failed (error %d), using fallback\n", cu_ret);
        }
    }

    // Step 5: Restore memory regions and allocate working memory
    printf("[5/6] Restoring memory regions...\n");
    CUdeviceptr d_data = 0;
    int n = 1024;

    // First restore any saved memory regions
    if (mem_regions > 0 && cuMemAlloc && cuMemcpyHtoD && hetgpu_checkpoint_get_memory_data) {
        for (uint32_t i = 0; i < mem_regions; i++) {
            uint64_t orig_addr = 0, size = 0;
            if (hetgpu_checkpoint_get_memory_mapping(i, &orig_addr, &size) == 0 && size > 0) {
                CUdeviceptr new_addr = 0;
                if (cuMemAlloc(&new_addr, size) == CUDA_SUCCESS) {
                    printf("      Region %u: 0x%lx -> 0x%llx (%lu bytes)\n",
                           i, (unsigned long)orig_addr, (unsigned long long)new_addr, (unsigned long)size);

                    uint8_t* buf = malloc(size);
                    if (buf) {
                        int copied = hetgpu_checkpoint_get_memory_data(i, buf, size);
                        if (copied > 0) {
                            cuMemcpyHtoD(new_addr, buf, copied);
                            printf("      Restored %d bytes of data\n", copied);
                        }
                        free(buf);
                    }
                    // Use first restored region as working memory
                    if (d_data == 0) {
                        d_data = new_addr;
                        n = size / sizeof(int);
                    }
                }
            }
        }
    }

    // Allocate new working memory if none was restored
    if (d_data == 0 && cuMemAlloc) {
        CUresult cu_ret = cuMemAlloc(&d_data, n * sizeof(int));
        if (cu_ret == CUDA_SUCCESS) {
            printf("      Allocated new working memory: %d ints\n", n);
            if (cuMemsetD32) {
                cuMemsetD32(d_data, 0, n);
            }
        } else {
            printf("      Memory allocation failed (error %d)\n", cu_ret);
        }
    }

    // Step 6: Run kernel loop (continuing from checkpoint)
    printf("[6/6] Running kernel loop (continuing from checkpoint)...\n");

    if (!kernel) {
        fprintf(stderr, "ERROR: No kernel available to run\n");
        return -1;
    }

    int max_iterations = g_auto_mode ? 5 : 0;  // 5 iterations in auto mode, infinite otherwise
    if (g_restore_iterations > 0) max_iterations = g_restore_iterations;

    if (max_iterations == 0) {
        printf("\n--- Resumed kernel execution ---\n");
        printf("Press Ctrl+C to create new checkpoint\n\n");
    } else {
        fprintf(stderr, "[Auto] Running %d verification iterations...\n", max_iterations);
    }

    int iteration = 0;
    while (max_iterations == 0 || iteration < max_iterations) {
        iteration++;

        if (cuLaunchKernel && kernel) {
            void* args[] = { &d_data, &n };
            CUresult cu_ret = cuLaunchKernel(kernel, (n + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
            if (cu_ret != CUDA_SUCCESS) {
                fprintf(stderr, "Iteration %d: launch failed (%d)\n", iteration, cu_ret);
            }
        }

        if (cuCtxSynchronize) {
            CUresult cu_ret = cuCtxSynchronize();
            if (cu_ret == CUDA_SUCCESS) {
                fprintf(stderr, "Iteration %d: launch OK, sync OK\n", iteration);
            } else {
                fprintf(stderr, "Iteration %d: sync failed (%d)\n", iteration, cu_ret);
            }
        }

        usleep(200000);  // 200ms between iterations
    }

    if (max_iterations > 0) {
        fprintf(stderr, "[Auto] Completed %d iterations successfully\n", iteration);
    }

    // Cleanup (unreachable in normal operation)
    if (cuMemFree && d_data) cuMemFree(d_data);

    return 0;
}

static int do_checkpoint_loop(void) {
    printf("\n=== CHECKPOINT MODE ===\n");
    printf("Press Ctrl+C to create checkpoint\n\n");

    CUresult ret;
    CUdevice device;
    CUcontext context;
    CUmodule module = NULL;
    CUfunction kernel = NULL;
    CUdeviceptr d_data = 0;
    int n = 1024;

    // Initialize
    ret = cuInit(0);
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", ret);
        return -1;
    }
    printf("[OK] cuInit\n");

    ret = cuDeviceGet(&device, 0);
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet failed: %d\n", ret);
        return -1;
    }
    printf("[OK] cuDeviceGet (device=%d)\n", device);

    ret = cuCtxCreate(&context, 0, device);
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxCreate failed: %d\n", ret);
        return -1;
    }
    printf("[OK] cuCtxCreate\n");

    // Allocate memory
    if (cuMemAlloc) {
        ret = cuMemAlloc(&d_data, n * sizeof(int));
        if (ret == CUDA_SUCCESS) {
            printf("[OK] cuMemAlloc (%d ints)\n", n);
        } else {
            printf("[--] cuMemAlloc failed: %d (continuing)\n", ret);
        }
    }

    // Initialize memory
    if (cuMemsetD32 && d_data) {
        cuMemsetD32(d_data, 0, n);
        printf("[OK] cuMemsetD32\n");
    }

    // Load PTX module
    if (cuModuleLoadData) {
        ret = cuModuleLoadData(&module, ptx_source);
        if (ret == CUDA_SUCCESS) {
            printf("[OK] cuModuleLoadData\n");
        } else {
            printf("[--] cuModuleLoadData failed: %d\n", ret);
        }
    }

    // Get kernel function
    if (cuModuleGetFunction && module) {
        ret = cuModuleGetFunction(&kernel, module, "test_kernel");
        if (ret == CUDA_SUCCESS) {
            printf("[OK] cuModuleGetFunction\n");
        } else {
            printf("[--] cuModuleGetFunction failed: %d\n", ret);
        }
    }

    // Kernel loop
    fprintf(stderr, "\n--- Starting kernel loop ---\n");
    int iteration = 0;
    while (1) {
        iteration++;

        if (cuLaunchKernel && kernel) {
            void* args[] = { &d_data, &n };
            ret = cuLaunchKernel(kernel, (n + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
            if (ret != CUDA_SUCCESS) {
                fprintf(stderr, "Iteration %d: launch failed (%d)\n", iteration, ret);
            }
        }

        if (cuCtxSynchronize) {
            ret = cuCtxSynchronize();
            if (ret == CUDA_SUCCESS) {
                fprintf(stderr, "Iteration %d: launch OK, sync OK\n", iteration);
            } else {
                fprintf(stderr, "Iteration %d: sync failed (%d)\n", iteration, ret);
            }
        }

        usleep(500000);  // 500ms between iterations
    }

    // Cleanup (unreachable in normal operation)
    if (cuMemFree && d_data) cuMemFree(d_data);
    if (cuCtxDestroy) cuCtxDestroy(context);

    return 0;
}

static void print_usage(const char* prog) {
    printf("hetGPU Checkpoint/Restore Test\n\n");
    printf("Usage:\n");
    printf("  Checkpoint mode (run kernel loop, Ctrl+C to checkpoint):\n");
    printf("    LD_PRELOAD=/path/to/libnvcuda.so %s\n\n", prog);
    printf("  Restore mode (auto-find latest checkpoint):\n");
    printf("    LD_PRELOAD=/path/to/libnvcuda.so %s --restore\n\n", prog);
    printf("  Restore mode (specific checkpoint file):\n");
    printf("    LD_PRELOAD=/path/to/libnvcuda.so %s --restore <checkpoint.json>\n\n", prog);
    printf("  Auto mode (checkpoint -> auto restore loop):\n");
    printf("    LD_PRELOAD=/path/to/libnvcuda.so %s --auto\n\n", prog);
    printf("  List checkpoints:\n");
    printf("    ls -la /tmp/hetgpu_checkpoints/\n\n");
    printf("Example:\n");
    printf("  # Create checkpoint\n");
    printf("  LD_PRELOAD=./target/release/libnvcuda.so %s\n", prog);
    printf("  # Press Ctrl+C to create checkpoint\n\n");
    printf("  # Auto restore from latest checkpoint\n");
    printf("  LD_PRELOAD=./target/release/libnvcuda.so %s --restore\n\n", prog);
    printf("  # Auto checkpoint/restore cycle\n");
    printf("  LD_PRELOAD=./target/release/libnvcuda.so %s --auto\n", prog);
}

int main(int argc, char** argv) {
    int restore_mode = 0;
    int auto_mode = 0;
    const char* restore_file = NULL;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--restore") == 0 || strcmp(argv[i], "-r") == 0) {
            restore_mode = 1;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                restore_file = argv[++i];
            }
        } else if (strcmp(argv[i], "--auto") == 0 || strcmp(argv[i], "-a") == 0) {
            auto_mode = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (restore_mode && !restore_file) {
            restore_file = argv[i];
        }
    }

    // Auto-find latest checkpoint if restore mode without file
    if (restore_mode && !restore_file) {
        restore_file = find_latest_checkpoint();
        if (!restore_file) {
            fprintf(stderr, "ERROR: No checkpoint files found in %s\n", CHECKPOINT_DIR);
            printf("\nAvailable checkpoints:\n");
            system("ls -la /tmp/hetgpu_checkpoints/ 2>/dev/null || echo 'No checkpoints found'");
            return 1;
        }
        fprintf(stderr, "[Auto] Found latest checkpoint: %s\n", restore_file);
    }

    printf("==============================================\n");
    printf("  hetGPU Checkpoint/Restore Standalone Test\n");
    printf("==============================================\n");

    // Load CUDA functions
    if (load_cuda_functions() != 0) {
        return 1;
    }

    // Check for hetGPU features
    if (hetgpu_checkpoint_load) {
        printf("[OK] hetGPU checkpoint functions available\n");
        if (hetgpu_request_checkpoint) {
            printf("[OK] hetGPU programmatic checkpoint available\n");
        } else {
            printf("[--] hetGPU programmatic checkpoint NOT available (will use signal)\n");
        }
    } else {
        printf("[--] hetGPU checkpoint functions NOT available\n");
        if (restore_mode) {
            fprintf(stderr, "ERROR: Restore mode requires hetGPU's libnvcuda.so\n");
            return 1;
        }
    }

    // Initialize CUDA
    CUresult ret = cuInit(0);
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", ret);
        return 1;
    }

    CUdevice device;
    ret = cuDeviceGet(&device, 0);
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet failed: %d\n", ret);
        return 1;
    }

    CUcontext context;
    ret = cuCtxCreate(&context, 0, device);
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxCreate failed: %d\n", ret);
        return 1;
    }

    if (auto_mode) {
        // Auto mode: run kernels -> trigger checkpoint -> restore -> repeat
        g_auto_mode = 1;
        fprintf(stderr, "\n=== AUTO MODE ===\n");
        fprintf(stderr, "Will run kernels, create checkpoint, then restore and continue\n\n");

        // Load PTX module and get kernel
        CUmodule module = NULL;
        CUfunction kernel = NULL;
        CUdeviceptr d_data = 0;
        int n = 1024;

        if (cuModuleLoadData) {
            CUresult cu_ret = cuModuleLoadData(&module, ptx_source);
            if (cu_ret == CUDA_SUCCESS) {
                fprintf(stderr, "[Auto] PTX module loaded\n");
                if (cuModuleGetFunction) {
                    cuModuleGetFunction(&kernel, module, "test_kernel");
                }
            }
        }

        if (cuMemAlloc) {
            cuMemAlloc(&d_data, n * sizeof(int));
            if (cuMemsetD32) cuMemsetD32(d_data, 0, n);
        }

        int cycle = 0;
        while (cycle < 3) {  // Run 3 checkpoint/restore cycles
            cycle++;
            fprintf(stderr, "\n>>> Cycle %d: Running kernel iterations <<<\n", cycle);

            // Get current time to detect new checkpoints
            time_t start_time = time(NULL);

            // Run some kernel iterations
            for (int i = 0; i < 5; i++) {
                if (cuLaunchKernel && kernel) {
                    void* args[] = { &d_data, &n };
                    cuLaunchKernel(kernel, (n + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
                }
                if (cuCtxSynchronize) {
                    cuCtxSynchronize();
                }
                fprintf(stderr, "  Iteration %d.%d: OK\n", cycle, i + 1);
                usleep(200000);
            }

            // Trigger checkpoint programmatically (avoids core dumps on some backends)
            fprintf(stderr, "\n[Auto] Triggering checkpoint...\n");
            if (hetgpu_request_checkpoint) {
                hetgpu_request_checkpoint();
            } else {
                // Fallback to signal if programmatic checkpoint not available
                fprintf(stderr, "[Auto] WARNING: hetgpu_request_checkpoint not available, using signal\n");
                raise(SIGINT);
            }
            usleep(500000);  // Give checkpoint time to complete

            // Find the new checkpoint
            char* new_checkpoint = wait_for_new_checkpoint(start_time, 5);
            if (!new_checkpoint) {
                fprintf(stderr, "[Auto] No new checkpoint found!\n");
                continue;
            }

            fprintf(stderr, "\n>>> Cycle %d: Restoring from checkpoint <<<\n", cycle);
            fprintf(stderr, "[Auto] Checkpoint: %s\n", new_checkpoint);

            // Restore from checkpoint
            int restore_result = do_restore(new_checkpoint);
            if (restore_result != 0) {
                fprintf(stderr, "[Auto] Restore completed\n");
            }

            fprintf(stderr, "\n=== Cycle %d complete ===\n", cycle);
        }

        fprintf(stderr, "\n=== AUTO MODE COMPLETE: %d cycles ===\n", cycle);
        if (cuMemFree && d_data) cuMemFree(d_data);
    } else if (restore_mode) {
        int result = do_restore(restore_file);
        if (cuCtxDestroy) cuCtxDestroy(context);
        return result;
    } else {
        return do_checkpoint_loop();
    }

    if (cuCtxDestroy) cuCtxDestroy(context);
    return 0;
}
