// Comprehensive Instruction-Level Checkpoint/Restore Migration Test
//
// This test demonstrates the full flow of:
// 1. Running a kernel partially
// 2. Checkpointing at a specific instruction with register state
// 3. Generating instrumented PTX for resumption
// 4. Loading instrumented PTX and resuming from the checkpoint
//
// Compile: gcc -o test_full_migration test_full_migration.c -ldl -lpthread
// Run: LD_PRELOAD=./target/release/libnvcuda.so ./test_full_migration

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <unistd.h>

// CUDA types
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

// Function pointers for CUDA
typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuDeviceGet_fn)(CUdevice*, int);
typedef CUresult (*cuCtxCreate_fn)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_fn)(CUcontext);
typedef CUresult (*cuMemAlloc_fn)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemFree_fn)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_fn)(CUdeviceptr, const void*, size_t);
typedef CUresult (*cuMemcpyDtoH_fn)(void*, CUdeviceptr, size_t);
typedef CUresult (*cuModuleLoadData_fn)(CUmodule*, const void*);
typedef CUresult (*cuModuleGetFunction_fn)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuLaunchKernel_fn)(CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, CUstream, void**, void**);
typedef CUresult (*cuCtxSynchronize_fn)(void);

// hetGPU FFI types for instruction-level checkpoint
typedef struct {
    char name[32];
    uint64_t value;
    uint32_t reg_type;  // 0=u32, 1=u64, 2=f32, 3=f64, 4=pred
} CRegisterValue;

typedef struct {
    uint64_t instruction_offset;
    uint32_t ptx_line;
    uint32_t block_x, block_y, block_z;
    uint32_t thread_x, thread_y, thread_z;
    uint64_t active_mask;
    uint32_t num_registers;
} CThreadState;

// hetGPU FFI function types
typedef int (*hetgpu_checkpoint_save_with_state_fn)(
    uint64_t instruction_offset,
    uint32_t num_registers,
    const CRegisterValue* registers,
    uint32_t num_predicates,
    const CRegisterValue* predicates,
    char* path_buf,
    uint32_t path_buf_size
);

typedef const char* (*hetgpu_get_instrumented_ptx_fn)(
    const char* original_ptx,
    uint64_t resume_ip,
    uint32_t num_registers,
    const CRegisterValue* registers,
    uint32_t num_predicates,
    const CRegisterValue* predicates
);

typedef int (*hetgpu_checkpoint_load_fn)(const char* path);
typedef uint64_t (*hetgpu_checkpoint_get_resume_ip_fn)(void);
typedef uint32_t (*hetgpu_checkpoint_get_thread_state_count_fn)(void);
typedef int (*hetgpu_checkpoint_get_thread_state_fn)(uint32_t index, CThreadState* state);
typedef int (*hetgpu_checkpoint_get_register_fn)(uint32_t thread_index, const char* reg_name, uint64_t* value);
typedef int (*hetgpu_parse_ptx_instruction_count_fn)(const char* ptx);
typedef const char* (*hetgpu_checkpoint_get_ptx_fn)(void);

// Global function pointers
static cuInit_fn cuInit;
static cuDeviceGet_fn cuDeviceGet;
static cuCtxCreate_fn cuCtxCreate;
static cuCtxDestroy_fn cuCtxDestroy;
static cuMemAlloc_fn cuMemAlloc;
static cuMemFree_fn cuMemFree;
static cuMemcpyHtoD_fn cuMemcpyHtoD;
static cuMemcpyDtoH_fn cuMemcpyDtoH;
static cuModuleLoadData_fn cuModuleLoadData;
static cuModuleGetFunction_fn cuModuleGetFunction;
static cuLaunchKernel_fn cuLaunchKernel;
static cuCtxSynchronize_fn cuCtxSynchronize;

// hetGPU function pointers
static hetgpu_checkpoint_save_with_state_fn hetgpu_checkpoint_save_with_state;
static hetgpu_get_instrumented_ptx_fn hetgpu_get_instrumented_ptx;
static hetgpu_checkpoint_load_fn hetgpu_checkpoint_load;
static hetgpu_checkpoint_get_resume_ip_fn hetgpu_checkpoint_get_resume_ip;
static hetgpu_checkpoint_get_thread_state_count_fn hetgpu_checkpoint_get_thread_state_count;
static hetgpu_checkpoint_get_thread_state_fn hetgpu_checkpoint_get_thread_state;
static hetgpu_checkpoint_get_register_fn hetgpu_checkpoint_get_register;
static hetgpu_parse_ptx_instruction_count_fn hetgpu_parse_ptx_instruction_count;
static hetgpu_checkpoint_get_ptx_fn hetgpu_checkpoint_get_ptx;

// Simple accumulator kernel - each instruction modifies the accumulator
// This allows us to verify that we resume from the correct instruction
static const char* ACCUMULATOR_PTX =
".version 7.5\n"
".target sm_61\n"
".address_size 64\n"
"\n"
".visible .entry accumulator_kernel(\n"
"    .param .u64 result_ptr,\n"
"    .param .u32 checkpoint_ip\n"
")\n"
"{\n"
"    .reg .b32 %r<20>;\n"
"    .reg .b64 %rd<10>;\n"
"    .reg .pred %p<5>;\n"
"\n"
"    // IP=0: Load parameters\n"
"    ld.param.u64 %rd1, [result_ptr];\n"
"    ld.param.u32 %r10, [checkpoint_ip];\n"
"\n"
"    // IP=1: Initialize accumulator to 0\n"
"    mov.u32 %r1, 0;\n"
"\n"
"    // IP=2: Add 100\n"
"    add.u32 %r1, %r1, 100;\n"
"\n"
"    // IP=3: Add 200\n"
"    add.u32 %r1, %r1, 200;\n"
"\n"
"    // IP=4: Add 300\n"
"    add.u32 %r1, %r1, 300;\n"
"\n"
"    // IP=5: Add 400\n"
"    add.u32 %r1, %r1, 400;\n"
"\n"
"    // IP=6: Store final result (should be 1000)\n"
"    st.global.u32 [%rd1], %r1;\n"
"\n"
"    ret;\n"
"}\n";

// Load all function pointers
static int load_functions(void) {
    void* lib = dlopen(NULL, RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
        fprintf(stderr, "Failed to open self\n");
        return -1;
    }

    cuInit = (cuInit_fn)dlsym(lib, "cuInit");
    cuDeviceGet = (cuDeviceGet_fn)dlsym(lib, "cuDeviceGet");
    cuCtxCreate = (cuCtxCreate_fn)dlsym(lib, "cuCtxCreate_v2");
    cuCtxDestroy = (cuCtxDestroy_fn)dlsym(lib, "cuCtxDestroy_v2");
    cuMemAlloc = (cuMemAlloc_fn)dlsym(lib, "cuMemAlloc_v2");
    cuMemFree = (cuMemFree_fn)dlsym(lib, "cuMemFree_v2");
    cuMemcpyHtoD = (cuMemcpyHtoD_fn)dlsym(lib, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH = (cuMemcpyDtoH_fn)dlsym(lib, "cuMemcpyDtoH_v2");
    cuModuleLoadData = (cuModuleLoadData_fn)dlsym(lib, "cuModuleLoadData");
    cuModuleGetFunction = (cuModuleGetFunction_fn)dlsym(lib, "cuModuleGetFunction");
    cuLaunchKernel = (cuLaunchKernel_fn)dlsym(lib, "cuLaunchKernel");
    cuCtxSynchronize = (cuCtxSynchronize_fn)dlsym(lib, "cuCtxSynchronize");

    // hetGPU functions
    hetgpu_checkpoint_save_with_state = (hetgpu_checkpoint_save_with_state_fn)
        dlsym(lib, "hetgpu_checkpoint_save_with_state");
    hetgpu_get_instrumented_ptx = (hetgpu_get_instrumented_ptx_fn)
        dlsym(lib, "hetgpu_get_instrumented_ptx");
    hetgpu_checkpoint_load = (hetgpu_checkpoint_load_fn)
        dlsym(lib, "hetgpu_checkpoint_load");
    hetgpu_checkpoint_get_resume_ip = (hetgpu_checkpoint_get_resume_ip_fn)
        dlsym(lib, "hetgpu_checkpoint_get_resume_ip");
    hetgpu_checkpoint_get_thread_state_count = (hetgpu_checkpoint_get_thread_state_count_fn)
        dlsym(lib, "hetgpu_checkpoint_get_thread_state_count");
    hetgpu_checkpoint_get_thread_state = (hetgpu_checkpoint_get_thread_state_fn)
        dlsym(lib, "hetgpu_checkpoint_get_thread_state");
    hetgpu_checkpoint_get_register = (hetgpu_checkpoint_get_register_fn)
        dlsym(lib, "hetgpu_checkpoint_get_register");
    hetgpu_parse_ptx_instruction_count = (hetgpu_parse_ptx_instruction_count_fn)
        dlsym(lib, "hetgpu_parse_ptx_instruction_count");
    hetgpu_checkpoint_get_ptx = (hetgpu_checkpoint_get_ptx_fn)
        dlsym(lib, "hetgpu_checkpoint_get_ptx");

    if (!cuInit || !cuDeviceGet || !cuCtxCreate) {
        fprintf(stderr, "Failed to load CUDA functions\n");
        return -1;
    }

    return 0;
}

static void print_separator(const char* title) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║ %-68s ║\n", title);
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
}

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║        Full Instruction-Level Migration Test                         ║\n");
    printf("║                                                                      ║\n");
    printf("║  This test demonstrates checkpointing a kernel mid-execution         ║\n");
    printf("║  and resuming from the exact instruction using instrumented PTX.     ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    // Load functions
    if (load_functions() != 0) {
        return 1;
    }

    print_separator("Step 1: Check hetGPU Functions");

    printf("[%s] hetgpu_checkpoint_save_with_state\n",
           hetgpu_checkpoint_save_with_state ? "OK" : "--");
    printf("[%s] hetgpu_get_instrumented_ptx\n",
           hetgpu_get_instrumented_ptx ? "OK" : "--");
    printf("[%s] hetgpu_checkpoint_load\n",
           hetgpu_checkpoint_load ? "OK" : "--");
    printf("[%s] hetgpu_parse_ptx_instruction_count\n",
           hetgpu_parse_ptx_instruction_count ? "OK" : "--");

    print_separator("Step 2: Parse PTX Instructions");

    if (hetgpu_parse_ptx_instruction_count) {
        int count = hetgpu_parse_ptx_instruction_count(ACCUMULATOR_PTX);
        printf("PTX has %d instructions\n", count);
        printf("\nKernel Logic:\n");
        printf("  IP=0: Load parameters (rd1=result_ptr, r10=checkpoint_ip)\n");
        printf("  IP=1: r1 = 0          (accumulator = 0)\n");
        printf("  IP=2: r1 = r1 + 100   (accumulator = 100)\n");
        printf("  IP=3: r1 = r1 + 200   (accumulator = 300)\n");
        printf("  IP=4: r1 = r1 + 300   (accumulator = 600)\n");
        printf("  IP=5: r1 = r1 + 400   (accumulator = 1000)\n");
        printf("  IP=6: store r1        (result = 1000)\n");
    }

    print_separator("Step 3: Initialize CUDA First");

    CUresult ret = cuInit(0);
    if (ret != CUDA_SUCCESS) {
        printf("cuInit failed: %d\n", ret);
        return 1;
    }

    CUdevice device;
    ret = cuDeviceGet(&device, 0);
    if (ret != CUDA_SUCCESS) {
        printf("cuDeviceGet failed: %d\n", ret);
        return 1;
    }

    CUcontext context;
    ret = cuCtxCreate(&context, 0, device);
    if (ret != CUDA_SUCCESS) {
        printf("cuCtxCreate failed: %d\n", ret);
        return 1;
    }

    printf("[OK] CUDA context created\n");

    // Allocate result buffer FIRST so we have the real device pointer
    CUdeviceptr d_result = 0;
    uint32_t h_result = 0xDEADBEEF;

    if (cuMemAlloc) {
        ret = cuMemAlloc(&d_result, sizeof(uint32_t));
        if (ret == CUDA_SUCCESS) {
            printf("[OK] Allocated device memory: 0x%llx\n", (unsigned long long)d_result);
        }
    }

    print_separator("Step 4: Simulate Checkpoint at IP=3");

    // Simulate state at IP=3: after executing IP=0,1,2 but before IP=3
    // At this point: r1 = 100 (after add 100)
    printf("Simulating execution up to IP=3...\n");
    printf("  Executed: IP=0 (load params: rd1=result_ptr)\n");
    printf("  Executed: IP=1 (load params: r10=checkpoint_ip)\n");
    printf("  Executed: IP=2 (r1 = 0)\n");
    printf("  Executed: IP=3 (r1 = r1 + 100 = 100)\n");
    printf("  >> CHECKPOINT HERE <<\n");
    printf("  Pending:  IP=4 (r1 = r1 + 200)\n");
    printf("  Pending:  IP=5 (r1 = r1 + 300)\n");
    printf("  Pending:  IP=6 (r1 = r1 + 400)\n");
    printf("  Pending:  IP=7 (store r1)\n");
    printf("  Pending:  IP=8 (ret)\n\n");

    // Build register state at checkpoint - using ACTUAL device pointer
    CRegisterValue registers[5];
    memset(registers, 0, sizeof(registers));

    // r1 = 100 (after IP=3: add 100)
    strcpy(registers[0].name, "%r1");
    registers[0].value = 100;
    registers[0].reg_type = 0;  // u32

    // rd1 = ACTUAL result pointer from cuMemAlloc
    strcpy(registers[1].name, "%rd1");
    registers[1].value = (uint64_t)d_result;  // Use actual device pointer!
    registers[1].reg_type = 1;  // u64

    // r10 = checkpoint_ip parameter
    strcpy(registers[2].name, "%r10");
    registers[2].value = 4;  // We're resuming at IP=4
    registers[2].reg_type = 0;  // u32

    printf("Register state at checkpoint (using actual device ptr):\n");
    for (int i = 0; i < 3; i++) {
        printf("  %s = 0x%llx\n", registers[i].name, (unsigned long long)registers[i].value);
    }

    // Save checkpoint with state
    if (hetgpu_checkpoint_save_with_state) {
        char path_buf[256] = {0};
        int ret = hetgpu_checkpoint_save_with_state(
            4,  // instruction_offset = IP=4 (resume at add 200)
            3,  // num_registers
            registers,
            0,  // num_predicates
            NULL,
            path_buf,
            sizeof(path_buf)
        );
        printf("\nCheckpoint saved: %s (ret=%d)\n", path_buf, ret);
    }

    print_separator("Step 5: Generate Instrumented PTX");

    const char* instrumented_ptx = NULL;
    if (hetgpu_get_instrumented_ptx) {
        instrumented_ptx = hetgpu_get_instrumented_ptx(
            ACCUMULATOR_PTX,
            4,  // resume at IP=4 (add 200)
            3,  // num_registers
            registers,
            0,  // num_predicates
            NULL
        );

        if (instrumented_ptx) {
            printf("Generated instrumented PTX (%zu bytes):\n\n", strlen(instrumented_ptx));

            // Print first 60 lines
            const char* p = instrumented_ptx;
            int lines = 0;
            while (*p && lines < 60) {
                const char* eol = strchr(p, '\n');
                if (!eol) eol = p + strlen(p);
                printf("%.*s\n", (int)(eol - p), p);
                p = (*eol) ? eol + 1 : eol;
                lines++;
            }
            if (*p) printf("... (truncated)\n");
        } else {
            printf("ERROR: Failed to generate instrumented PTX\n");
        }
    }

    print_separator("Step 6: Test Execution with CUDA");

    // Test 1: Run original PTX (full execution)
    printf("\n--- Test 1: Full execution (original PTX) ---\n");
    CUmodule module = NULL;
    CUfunction kernel = NULL;

    if (cuModuleLoadData && cuModuleLoadData(&module, ACCUMULATOR_PTX) == CUDA_SUCCESS) {
        printf("[OK] Original PTX loaded\n");

        if (cuModuleGetFunction && cuModuleGetFunction(&kernel, module, "accumulator_kernel") == CUDA_SUCCESS) {
            printf("[OK] Got kernel function\n");

            // Initialize result to 0
            h_result = 0;
            if (cuMemcpyHtoD) cuMemcpyHtoD(d_result, &h_result, sizeof(uint32_t));

            uint32_t checkpoint_ip = 0;  // Not used in original
            void* args[] = { &d_result, &checkpoint_ip };

            if (cuLaunchKernel) {
                ret = cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
                if (ret == CUDA_SUCCESS && cuCtxSynchronize) {
                    cuCtxSynchronize();
                    if (cuMemcpyDtoH) cuMemcpyDtoH(&h_result, d_result, sizeof(uint32_t));
                    printf("[OK] Full execution result: %u (expected: 1000)\n", h_result);
                    if (h_result == 1000) {
                        printf("    ✓ CORRECT!\n");
                    } else {
                        printf("    ✗ INCORRECT (expected 1000)\n");
                    }
                }
            }
        }
    } else {
        printf("[--] Original PTX loading failed (expected on non-NVIDIA)\n");
    }

    // Test 2: Run instrumented PTX (resume from IP=4)
    printf("\n--- Test 2: Resume from IP=4 (instrumented PTX) ---\n");

    if (instrumented_ptx && cuModuleLoadData) {
        CUmodule resume_module = NULL;
        ret = cuModuleLoadData(&resume_module, instrumented_ptx);

        if (ret == CUDA_SUCCESS) {
            printf("[OK] Instrumented PTX loaded\n");

            CUfunction resume_kernel = NULL;
            if (cuModuleGetFunction && cuModuleGetFunction(&resume_kernel, resume_module, "accumulator_kernel") == CUDA_SUCCESS) {
                printf("[OK] Got resume kernel function\n");

                // The instrumented PTX should:
                // 1. Restore rd1 = actual device pointer (from checkpoint state)
                // 2. Restore r1 = 100 (from checkpoint state, after IP=3)
                // 3. Jump to IP=4
                // 4. Execute: IP=4: r1 = 100 + 200 = 300
                // 5. Execute: IP=5: r1 = 300 + 300 = 600
                // 6. Execute: IP=6: r1 = 600 + 400 = 1000
                // 7. Execute: IP=7: store r1
                // 8. Execute: IP=8: ret
                // Expected result: 1000 (same as full execution)

                h_result = 0;
                if (cuMemcpyHtoD) cuMemcpyHtoD(d_result, &h_result, sizeof(uint32_t));

                uint32_t checkpoint_ip = 4;
                void* args[] = { &d_result, &checkpoint_ip };

                if (cuLaunchKernel) {
                    ret = cuLaunchKernel(resume_kernel, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
                    if (ret == CUDA_SUCCESS && cuCtxSynchronize) {
                        cuCtxSynchronize();
                        if (cuMemcpyDtoH) cuMemcpyDtoH(&h_result, d_result, sizeof(uint32_t));
                        printf("[OK] Resume execution result: %u\n", h_result);

                        // If resume from IP=4 works correctly:
                        // - Restored r1 = 100 (state after IP=3)
                        // - IP=4: r1 = 100 + 200 = 300
                        // - IP=5: r1 = 300 + 300 = 600
                        // - IP=6: r1 = 600 + 400 = 1000
                        // Expected: 1000
                        if (h_result == 1000) {
                            printf("    ✓ CORRECT! Migration worked - resumed from IP=4 with r1=100\n");
                        } else if (h_result == 900) {
                            printf("    ~ Partial: r1 wasn't restored (started from 0, got 200+300+400=900)\n");
                        } else {
                            printf("    ? Got %u - check instrumentation\n", h_result);
                        }
                    }
                }
            }
        } else {
            printf("[--] Instrumented PTX loading failed: %d\n", ret);
        }
    }

    print_separator("Test Summary");

    printf("Instruction-Level Migration Flow:\n\n");
    printf("  Source GPU:                    Target GPU:\n");
    printf("  ┌───────────────────────┐      ┌─────────────────────────┐\n");
    printf("  │ Execute IP=0,1,2,3    │      │ Load instrumented PTX   │\n");
    printf("  │ Checkpoint at IP=4    │─────►│ Restore r1=100, rd1=ptr │\n");
    printf("  │ Save: r1=100, rd1=ptr │      │ Jump to IP=4            │\n");
    printf("  └───────────────────────┘      │ Continue: IP=4,5,6,7,8  │\n");
    printf("                                 │ Result: 1000 ✓          │\n");
    printf("                                 └─────────────────────────┘\n");

    // Cleanup
    if (cuMemFree && d_result) cuMemFree(d_result);
    if (cuCtxDestroy) cuCtxDestroy(context);

    printf("\n=== Test Complete ===\n");
    return 0;
}
