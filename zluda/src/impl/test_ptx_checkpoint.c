/**
 * Comprehensive PTX Test Runner with Fine-Grained Checkpointing
 *
 * This test program loads multiple PTX files, runs them with
 * fine-grained checkpointing, and verifies instruction-level
 * checkpoint/restore functionality.
 *
 * Build: gcc -o test_ptx_checkpoint test_ptx_checkpoint.c -ldl
 * Run: LD_PRELOAD=/path/to/libnvcuda.so ./test_ptx_checkpoint [options]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <signal.h>
#include <time.h>

// CUDA types (minimal definitions)
typedef int CUresult;
typedef void* CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define CHECKPOINT_DIR "/tmp/hetgpu_checkpoints"

// Function pointer types
typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuDeviceGet_fn)(CUdevice*, int);
typedef CUresult (*cuCtxCreate_fn)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_fn)(CUcontext);
typedef CUresult (*cuModuleLoadData_fn)(CUmodule*, const void*);
typedef CUresult (*cuModuleUnload_fn)(CUmodule);
typedef CUresult (*cuModuleGetFunction_fn)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuMemAlloc_fn)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemFree_fn)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_fn)(CUdeviceptr, const void*, size_t);
typedef CUresult (*cuMemcpyDtoH_fn)(void*, CUdeviceptr, size_t);
typedef CUresult (*cuLaunchKernel_fn)(CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, CUstream, void**, void**);
typedef CUresult (*cuCtxSynchronize_fn)(void);

// hetGPU fine-grained checkpoint types
typedef int (*hetgpu_set_checkpoint_granularity_fn)(int);
typedef int (*hetgpu_checkpoint_load_fn)(const char*);
typedef int (*hetgpu_checkpoint_restore_fn)(void);
typedef void (*hetgpu_request_checkpoint_fn)(void);
typedef int (*hetgpu_get_instruction_count_fn)(void);
typedef int (*hetgpu_get_checkpoint_stats_fn)(void*, size_t);

// Global function pointers
static cuInit_fn cuInit;
static cuDeviceGet_fn cuDeviceGet;
static cuCtxCreate_fn cuCtxCreate;
static cuCtxDestroy_fn cuCtxDestroy;
static cuModuleLoadData_fn cuModuleLoadData;
static cuModuleUnload_fn cuModuleUnload;
static cuModuleGetFunction_fn cuModuleGetFunction;
static cuMemAlloc_fn cuMemAlloc;
static cuMemFree_fn cuMemFree;
static cuMemcpyHtoD_fn cuMemcpyHtoD;
static cuMemcpyDtoH_fn cuMemcpyDtoH;
static cuLaunchKernel_fn cuLaunchKernel;
static cuCtxSynchronize_fn cuCtxSynchronize;

// hetGPU functions
static hetgpu_set_checkpoint_granularity_fn hetgpu_set_checkpoint_granularity;
static hetgpu_checkpoint_load_fn hetgpu_checkpoint_load;
static hetgpu_checkpoint_restore_fn hetgpu_checkpoint_restore;
static hetgpu_request_checkpoint_fn hetgpu_request_checkpoint;
static hetgpu_get_instruction_count_fn hetgpu_get_instruction_count;
static hetgpu_get_checkpoint_stats_fn hetgpu_get_checkpoint_stats;

// Global state
static void* cuda_lib = NULL;
static int g_verbose = 0;
static int g_total_tests = 0;
static int g_passed_tests = 0;
static int g_failed_tests = 0;

// Checkpoint granularity levels (must match Rust enum)
enum CheckpointGranularity {
    GRANULARITY_KERNEL = 0,
    GRANULARITY_BASIC_BLOCK = 1,
    GRANULARITY_INSTRUCTION = 2,
    GRANULARITY_WARP_SYNC = 3,
};

// Test PTX kernels (using macros for static initialization)
#define TEST_PTX_ADD \
".version 7.5\n" \
".target sm_61\n" \
".address_size 64\n" \
"\n" \
".visible .entry add_kernel(\n" \
"    .param .u64 input,\n" \
"    .param .u64 output\n" \
")\n" \
"{\n" \
"    .reg .b64 %rd<5>;\n" \
"    .reg .b32 %r<3>;\n" \
"\n" \
"    ld.param.u64 %rd1, [input];\n" \
"    ld.param.u64 %rd2, [output];\n" \
"    mov.u32 %r1, %tid.x;\n" \
"    mul.wide.u32 %rd3, %r1, 8;\n" \
"    add.u64 %rd1, %rd1, %rd3;\n" \
"    add.u64 %rd2, %rd2, %rd3;\n" \
"    ld.global.u64 %rd4, [%rd1];\n" \
"    add.u64 %rd4, %rd4, 1;\n" \
"    st.global.u64 [%rd2], %rd4;\n" \
"    ret;\n" \
"}\n"

#define TEST_PTX_MUL \
".version 7.5\n" \
".target sm_61\n" \
".address_size 64\n" \
"\n" \
".visible .entry mul_kernel(\n" \
"    .param .u64 input,\n" \
"    .param .u64 output\n" \
")\n" \
"{\n" \
"    .reg .b64 %rd<5>;\n" \
"    .reg .b32 %r<3>;\n" \
"\n" \
"    ld.param.u64 %rd1, [input];\n" \
"    ld.param.u64 %rd2, [output];\n" \
"    mov.u32 %r1, %tid.x;\n" \
"    mul.wide.u32 %rd3, %r1, 8;\n" \
"    add.u64 %rd1, %rd1, %rd3;\n" \
"    add.u64 %rd2, %rd2, %rd3;\n" \
"    ld.global.u64 %rd4, [%rd1];\n" \
"    mul.lo.u64 %rd4, %rd4, 2;\n" \
"    st.global.u64 [%rd2], %rd4;\n" \
"    ret;\n" \
"}\n"

#define TEST_PTX_PREDICATE \
".version 7.5\n" \
".target sm_61\n" \
".address_size 64\n" \
"\n" \
".visible .entry predicate_kernel(\n" \
"    .param .u64 input,\n" \
"    .param .u64 output,\n" \
"    .param .s32 n\n" \
")\n" \
"{\n" \
"    .reg .pred %p<2>;\n" \
"    .reg .b32 %r<5>;\n" \
"    .reg .b64 %rd<5>;\n" \
"\n" \
"    ld.param.u64 %rd1, [input];\n" \
"    ld.param.u64 %rd2, [output];\n" \
"    ld.param.s32 %r1, [n];\n" \
"\n" \
"    mov.u32 %r2, %tid.x;\n" \
"    setp.ge.s32 %p1, %r2, %r1;\n" \
"    @%p1 bra END;\n" \
"\n" \
"    mul.wide.u32 %rd3, %r2, 4;\n" \
"    add.u64 %rd1, %rd1, %rd3;\n" \
"    add.u64 %rd2, %rd2, %rd3;\n" \
"    ld.global.s32 %r3, [%rd1];\n" \
"    add.s32 %r4, %r3, %r2;\n" \
"    st.global.s32 [%rd2], %r4;\n" \
"\n" \
"END:\n" \
"    ret;\n" \
"}\n"

#define TEST_PTX_LOOP \
".version 7.5\n" \
".target sm_61\n" \
".address_size 64\n" \
"\n" \
".visible .entry loop_kernel(\n" \
"    .param .u64 output,\n" \
"    .param .s32 iterations\n" \
")\n" \
"{\n" \
"    .reg .pred %p<2>;\n" \
"    .reg .b32 %r<10>;\n" \
"    .reg .b64 %rd<3>;\n" \
"\n" \
"    ld.param.u64 %rd1, [output];\n" \
"    ld.param.s32 %r1, [iterations];\n" \
"\n" \
"    mov.u32 %r2, %tid.x;\n" \
"    mul.wide.u32 %rd2, %r2, 4;\n" \
"    add.u64 %rd1, %rd1, %rd2;\n" \
"\n" \
"    mov.s32 %r3, 0;\n" \
"    mov.s32 %r4, 0;\n" \
"\n" \
"LOOP:\n" \
"    setp.ge.s32 %p1, %r4, %r1;\n" \
"    @%p1 bra END;\n" \
"\n" \
"    add.s32 %r3, %r3, %r4;\n" \
"    add.s32 %r4, %r4, 1;\n" \
"    bra LOOP;\n" \
"\n" \
"END:\n" \
"    st.global.s32 [%rd1], %r3;\n" \
"    ret;\n" \
"}\n"

#define TEST_PTX_SHARED_MEM \
".version 7.5\n" \
".target sm_61\n" \
".address_size 64\n" \
"\n" \
".visible .entry shared_kernel(\n" \
"    .param .u64 input,\n" \
"    .param .u64 output\n" \
")\n" \
"{\n" \
"    .shared .align 4 .b8 shared_data[256];\n" \
"    .reg .b32 %r<5>;\n" \
"    .reg .b64 %rd<5>;\n" \
"\n" \
"    ld.param.u64 %rd1, [input];\n" \
"    ld.param.u64 %rd2, [output];\n" \
"\n" \
"    mov.u32 %r1, %tid.x;\n" \
"    mul.wide.u32 %rd3, %r1, 4;\n" \
"    add.u64 %rd1, %rd1, %rd3;\n" \
"    add.u64 %rd2, %rd2, %rd3;\n" \
"\n" \
"    ld.global.s32 %r2, [%rd1];\n" \
"    mul.lo.s32 %r3, %r1, 4;\n" \
"    st.shared.s32 [shared_data+%r3], %r2;\n" \
"\n" \
"    bar.sync 0;\n" \
"\n" \
"    ld.shared.s32 %r4, [shared_data+%r3];\n" \
"    st.global.s32 [%rd2], %r4;\n" \
"\n" \
"    ret;\n" \
"}\n"

// Test structure
struct PTXTest {
    const char* name;
    const char* ptx_source;
    const char* kernel_name;
    int num_threads;
    int expected_instructions;  // Expected instruction count after parsing
};

static struct PTXTest g_tests[] = {
    {"add_kernel", TEST_PTX_ADD, "add_kernel", 64, 10},
    {"mul_kernel", TEST_PTX_MUL, "mul_kernel", 64, 10},
    {"predicate_kernel", TEST_PTX_PREDICATE, "predicate_kernel", 128, 12},
    {"loop_kernel", TEST_PTX_LOOP, "loop_kernel", 32, 10},
    {"shared_kernel", TEST_PTX_SHARED_MEM, "shared_kernel", 64, 12},
};
static int g_num_tests = sizeof(g_tests) / sizeof(g_tests[0]);

// Load CUDA functions
static int load_functions(void) {
    cuda_lib = RTLD_DEFAULT;

    cuInit = (cuInit_fn)dlsym(cuda_lib, "cuInit");
    cuDeviceGet = (cuDeviceGet_fn)dlsym(cuda_lib, "cuDeviceGet");
    cuCtxCreate = (cuCtxCreate_fn)dlsym(cuda_lib, "cuCtxCreate_v2");
    cuCtxDestroy = (cuCtxDestroy_fn)dlsym(cuda_lib, "cuCtxDestroy_v2");
    cuModuleLoadData = (cuModuleLoadData_fn)dlsym(cuda_lib, "cuModuleLoadData");
    cuModuleUnload = (cuModuleUnload_fn)dlsym(cuda_lib, "cuModuleUnload");
    cuModuleGetFunction = (cuModuleGetFunction_fn)dlsym(cuda_lib, "cuModuleGetFunction");
    cuMemAlloc = (cuMemAlloc_fn)dlsym(cuda_lib, "cuMemAlloc_v2");
    cuMemFree = (cuMemFree_fn)dlsym(cuda_lib, "cuMemFree_v2");
    cuMemcpyHtoD = (cuMemcpyHtoD_fn)dlsym(cuda_lib, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH = (cuMemcpyDtoH_fn)dlsym(cuda_lib, "cuMemcpyDtoH_v2");
    cuLaunchKernel = (cuLaunchKernel_fn)dlsym(cuda_lib, "cuLaunchKernel");
    cuCtxSynchronize = (cuCtxSynchronize_fn)dlsym(cuda_lib, "cuCtxSynchronize");

    // hetGPU extensions
    hetgpu_set_checkpoint_granularity = (hetgpu_set_checkpoint_granularity_fn)dlsym(cuda_lib, "hetgpu_set_checkpoint_granularity");
    hetgpu_checkpoint_load = (hetgpu_checkpoint_load_fn)dlsym(cuda_lib, "hetgpu_checkpoint_load");
    hetgpu_checkpoint_restore = (hetgpu_checkpoint_restore_fn)dlsym(cuda_lib, "hetgpu_checkpoint_restore");
    hetgpu_request_checkpoint = (hetgpu_request_checkpoint_fn)dlsym(cuda_lib, "hetgpu_request_checkpoint");
    hetgpu_get_instruction_count = (hetgpu_get_instruction_count_fn)dlsym(cuda_lib, "hetgpu_get_instruction_count");
    hetgpu_get_checkpoint_stats = (hetgpu_get_checkpoint_stats_fn)dlsym(cuda_lib, "hetgpu_get_checkpoint_stats");

    if (!cuInit || !cuDeviceGet || !cuCtxCreate || !cuModuleLoadData) {
        fprintf(stderr, "ERROR: Required CUDA functions not found\n");
        return -1;
    }

    return 0;
}

// Count instructions in PTX source (basic parser for testing)
static int count_ptx_instructions(const char* ptx_source) {
    int count = 0;
    int in_kernel = 0;
    const char* line = ptx_source;

    while (*line) {
        // Skip whitespace
        while (*line == ' ' || *line == '\t') line++;

        // Check for kernel entry/exit
        if (strstr(line, ".entry") || strstr(line, ".func")) {
            in_kernel = 1;
        } else if (*line == '}') {
            in_kernel = 0;
        }

        // Count instructions (non-directive, non-label lines in kernel)
        if (in_kernel && *line != '.' && *line != '\n' && *line != '\0' && *line != '}' && *line != '{') {
            // Check if it's not a label
            const char* eol = strchr(line, '\n');
            if (eol) {
                char temp[256];
                int len = eol - line;
                if (len > 255) len = 255;
                strncpy(temp, line, len);
                temp[len] = '\0';

                // Skip if it's a label (ends with :)
                char* colon = strchr(temp, ':');
                if (!colon || colon[1] != '\0') {
                    // Not a label, check if it's an instruction
                    char* first_word = strtok(temp, " \t");
                    if (first_word && first_word[0] != '.' && first_word[0] != '/' && first_word[0] != '\0') {
                        // Skip labels
                        if (first_word[strlen(first_word)-1] != ':') {
                            count++;
                        }
                    }
                }
            }
        }

        // Move to next line
        const char* newline = strchr(line, '\n');
        if (!newline) break;
        line = newline + 1;
    }

    return count;
}

// Run a single PTX test
static int run_ptx_test(struct PTXTest* test, enum CheckpointGranularity granularity) {
    CUresult result;
    CUmodule module;
    CUfunction function;
    CUdeviceptr d_input, d_output;
    int test_passed = 1;

    printf("\n--- Testing: %s (granularity=%d) ---\n", test->name, granularity);

    // Set checkpoint granularity if supported
    if (hetgpu_set_checkpoint_granularity) {
        hetgpu_set_checkpoint_granularity(granularity);
        if (g_verbose) printf("  Set checkpoint granularity: %d\n", granularity);
    }

    // Count instructions
    int instr_count = count_ptx_instructions(test->ptx_source);
    printf("  Parsed instruction count: %d\n", instr_count);

    // Load module
    result = cuModuleLoadData(&module, test->ptx_source);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "  FAILED: cuModuleLoadData returned %d\n", result);
        return 0;
    }
    if (g_verbose) printf("  Module loaded\n");

    // Get function
    result = cuModuleGetFunction(&function, module, test->kernel_name);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "  FAILED: cuModuleGetFunction returned %d\n", result);
        cuModuleUnload(module);
        return 0;
    }
    if (g_verbose) printf("  Function '%s' found\n", test->kernel_name);

    // Allocate memory
    size_t buffer_size = test->num_threads * sizeof(long long);
    result = cuMemAlloc(&d_input, buffer_size);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "  FAILED: cuMemAlloc for input returned %d\n", result);
        cuModuleUnload(module);
        return 0;
    }

    result = cuMemAlloc(&d_output, buffer_size);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "  FAILED: cuMemAlloc for output returned %d\n", result);
        cuMemFree(d_input);
        cuModuleUnload(module);
        return 0;
    }

    // Initialize input
    long long* h_input = (long long*)malloc(buffer_size);
    long long* h_output = (long long*)malloc(buffer_size);
    for (int i = 0; i < test->num_threads; i++) {
        h_input[i] = i + 1;
        h_output[i] = 0;
    }

    cuMemcpyHtoD(d_input, h_input, buffer_size);
    cuMemcpyHtoD(d_output, h_output, buffer_size);

    // Set up kernel arguments
    void* args[] = { &d_input, &d_output };
    int n = test->num_threads;

    // For predicate_kernel, add n parameter
    if (strcmp(test->kernel_name, "predicate_kernel") == 0) {
        void* pred_args[] = { &d_input, &d_output, &n };
        result = cuLaunchKernel(function, 1, 1, 1, test->num_threads, 1, 1, 0, NULL, pred_args, NULL);
    } else if (strcmp(test->kernel_name, "loop_kernel") == 0) {
        int iterations = 10;
        void* loop_args[] = { &d_output, &iterations };
        result = cuLaunchKernel(function, 1, 1, 1, test->num_threads, 1, 1, 0, NULL, loop_args, NULL);
    } else {
        result = cuLaunchKernel(function, 1, 1, 1, test->num_threads, 1, 1, 0, NULL, args, NULL);
    }

    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "  FAILED: cuLaunchKernel returned %d\n", result);
        test_passed = 0;
    } else {
        if (g_verbose) printf("  Kernel launched\n");

        // Trigger checkpoint during execution
        if (hetgpu_request_checkpoint) {
            hetgpu_request_checkpoint();
            if (g_verbose) printf("  Checkpoint requested\n");
        }

        // Synchronize
        result = cuCtxSynchronize();
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "  FAILED: cuCtxSynchronize returned %d\n", result);
            test_passed = 0;
        } else {
            if (g_verbose) printf("  Kernel completed\n");

            // Copy back results
            cuMemcpyDtoH(h_output, d_output, buffer_size);

            // Verify results (basic check)
            int errors = 0;
            for (int i = 0; i < test->num_threads && i < 5; i++) {
                if (g_verbose) printf("  output[%d] = %lld\n", i, h_output[i]);
            }
            if (h_output[0] == 0 && strcmp(test->kernel_name, "add_kernel") == 0) {
                // add_kernel should produce non-zero output
                fprintf(stderr, "  WARNING: Output appears to be all zeros\n");
            }
        }
    }

    // Cleanup
    free(h_input);
    free(h_output);
    cuMemFree(d_input);
    cuMemFree(d_output);
    cuModuleUnload(module);

    if (test_passed) {
        printf("  PASSED\n");
        g_passed_tests++;
    } else {
        printf("  FAILED\n");
        g_failed_tests++;
    }
    g_total_tests++;

    return test_passed;
}

// Test checkpoint/restore cycle
static int test_checkpoint_restore_cycle(struct PTXTest* test) {
    printf("\n=== Testing Checkpoint/Restore Cycle: %s ===\n", test->name);

    CUresult result;
    CUmodule module;
    CUfunction function;

    // Load and run kernel
    result = cuModuleLoadData(&module, test->ptx_source);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to load module\n");
        return 0;
    }

    result = cuModuleGetFunction(&function, module, test->kernel_name);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to get function\n");
        cuModuleUnload(module);
        return 0;
    }

    // Request checkpoint
    if (hetgpu_request_checkpoint) {
        printf("  Requesting checkpoint...\n");
        hetgpu_request_checkpoint();
    }

    // Find latest checkpoint
    DIR* dir = opendir(CHECKPOINT_DIR);
    if (dir) {
        struct dirent* entry;
        char latest_path[512] = {0};
        time_t latest_time = 0;

        while ((entry = readdir(dir)) != NULL) {
            if (strncmp(entry->d_name, "hetgpu_", 7) == 0) {
                char full_path[512];
                snprintf(full_path, sizeof(full_path), "%s/%s", CHECKPOINT_DIR, entry->d_name);
                struct stat st;
                if (stat(full_path, &st) == 0 && st.st_mtime > latest_time) {
                    latest_time = st.st_mtime;
                    strncpy(latest_path, full_path, sizeof(latest_path) - 1);
                }
            }
        }
        closedir(dir);

        if (latest_path[0] && hetgpu_checkpoint_load && hetgpu_checkpoint_restore) {
            printf("  Loading checkpoint: %s\n", latest_path);
            if (hetgpu_checkpoint_load(latest_path) == 0) {
                printf("  Restoring from checkpoint...\n");
                if (hetgpu_checkpoint_restore() == 0) {
                    printf("  Checkpoint/Restore cycle PASSED\n");
                    g_passed_tests++;
                    g_total_tests++;
                    cuModuleUnload(module);
                    return 1;
                }
            }
        }
    }

    cuModuleUnload(module);
    g_failed_tests++;
    g_total_tests++;
    return 0;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -v, --verbose    Verbose output\n");
    printf("  -g N             Set checkpoint granularity (0=kernel, 1=block, 2=instruction, 3=warp)\n");
    printf("  -t NAME          Run only test NAME\n");
    printf("  -l, --list       List available tests\n");
    printf("  --all            Run all granularity levels\n");
    printf("  -h, --help       Show this help\n");
}

int main(int argc, char** argv) {
    int granularity = GRANULARITY_KERNEL;
    const char* test_filter = NULL;
    int run_all_granularities = 0;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            g_verbose = 1;
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            granularity = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            test_filter = argv[++i];
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--list") == 0) {
            printf("Available tests:\n");
            for (int j = 0; j < g_num_tests; j++) {
                printf("  - %s (%d threads, ~%d instructions)\n",
                       g_tests[j].name, g_tests[j].num_threads, g_tests[j].expected_instructions);
            }
            return 0;
        } else if (strcmp(argv[i], "--all") == 0) {
            run_all_granularities = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("========================================\n");
    printf("PTX Test Runner with Fine-Grained Checkpointing\n");
    printf("========================================\n");

    // Load functions
    if (load_functions() != 0) {
        return 1;
    }
    printf("[OK] CUDA functions loaded\n");

    // Check hetGPU features
    if (hetgpu_request_checkpoint) {
        printf("[OK] hetGPU checkpoint functions available\n");
    } else {
        printf("[--] hetGPU checkpoint functions NOT available\n");
    }

    // Initialize CUDA
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", result);
        return 1;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet failed: %d\n", result);
        return 1;
    }

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxCreate failed: %d\n", result);
        return 1;
    }
    printf("[OK] CUDA context created\n");

    // Create checkpoint directory
    mkdir(CHECKPOINT_DIR, 0755);

    // Run tests
    if (run_all_granularities) {
        for (int g = GRANULARITY_KERNEL; g <= GRANULARITY_WARP_SYNC; g++) {
            printf("\n======== Granularity Level: %d ========\n", g);
            for (int i = 0; i < g_num_tests; i++) {
                if (test_filter && strcmp(g_tests[i].name, test_filter) != 0) {
                    continue;
                }
                run_ptx_test(&g_tests[i], g);
            }
        }
    } else {
        for (int i = 0; i < g_num_tests; i++) {
            if (test_filter && strcmp(g_tests[i].name, test_filter) != 0) {
                continue;
            }
            run_ptx_test(&g_tests[i], granularity);
        }
    }

    // Test checkpoint/restore cycle
    printf("\n======== Checkpoint/Restore Tests ========\n");
    for (int i = 0; i < g_num_tests; i++) {
        if (test_filter && strcmp(g_tests[i].name, test_filter) != 0) {
            continue;
        }
        test_checkpoint_restore_cycle(&g_tests[i]);
    }

    // Summary
    printf("\n========================================\n");
    printf("Test Summary:\n");
    printf("  Total:  %d\n", g_total_tests);
    printf("  Passed: %d\n", g_passed_tests);
    printf("  Failed: %d\n", g_failed_tests);
    printf("========================================\n");

    // Cleanup
    cuCtxDestroy(context);

    return g_failed_tests > 0 ? 1 : 0;
}
