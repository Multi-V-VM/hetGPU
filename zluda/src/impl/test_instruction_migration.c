// Test for instruction-level checkpoint/restore migration
// Demonstrates how PTX is instrumented for resuming at specific instructions
//
// Compile: gcc -o test_instruction_migration test_instruction_migration.c -ldl
// Run: LD_PRELOAD=./target/release/libnvcuda.so ./test_instruction_migration

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>

// CUDA types
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;

#define CUDA_SUCCESS 0

// Function pointers
typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuDeviceGet_fn)(CUdevice*, int);
typedef CUresult (*cuCtxCreate_fn)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuModuleLoadData_fn)(CUmodule*, const void*);
typedef CUresult (*cuModuleGetFunction_fn)(CUfunction*, CUmodule, const char*);

// PTX source that will be instrumented for migration
static const char* ORIGINAL_PTX =
".version 7.5\n"
".target sm_61\n"
".address_size 64\n"
"\n"
".visible .entry migrate_kernel(\n"
"    .param .u64 data,\n"
"    .param .s32 n\n"
")\n"
"{\n"
"    .reg .pred %p<2>;\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<5>;\n"
"\n"
"    // IP=0: Load parameters\n"
"    ld.param.u64 %rd1, [data];\n"
"    ld.param.s32 %r1, [n];\n"
"\n"
"    // IP=1: Get thread ID\n"
"    mov.u32 %r2, %ctaid.x;\n"
"    mov.u32 %r3, %ntid.x;\n"
"    mov.u32 %r4, %tid.x;\n"
"    mad.lo.s32 %r5, %r2, %r3, %r4;\n"
"\n"
"    // IP=2: Bounds check\n"
"    setp.ge.s32 %p1, %r5, %r1;\n"
"    @%p1 bra END;\n"
"\n"
"    // IP=3: Compute and store\n"
"    mul.wide.s32 %rd2, %r5, 4;\n"
"    add.s64 %rd3, %rd1, %rd2;\n"
"    st.global.s32 [%rd3], %r5;\n"
"\n"
"END:\n"
"    ret;\n"
"}\n";

// Example of what instrumented PTX looks like for resuming at IP=3
static const char* INSTRUMENTED_PTX_EXAMPLE =
".version 7.5\n"
".target sm_61\n"
".address_size 64\n"
"\n"
".visible .entry migrate_kernel(\n"
"    .param .u64 data,\n"
"    .param .s32 n\n"
")\n"
"{\n"
"    .reg .pred %p<2>;\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<5>;\n"
"\n"
"    // === Migration Restore Dispatcher ===\n"
"    .reg .pred %__restore_active;\n"
"    .reg .u64 %__restore_ip;\n"
"    .reg .u64 %__restore_base;\n"
"    // Check restore mode flag\n"
"    mov.u64 %__restore_ip, 3;  // Resume at IP=3\n"
"    setp.ne.u64 %__restore_active, %__restore_ip, 0;\n"
"    @%__restore_active bra __restore_dispatch;\n"
"    bra __normal_start;\n"
"\n"
"__restore_dispatch:\n"
"    // Restore register state from checkpoint\n"
"    mov.u64 %rd1, 0x7f0000000000;  // Restored data pointer\n"
"    mov.u32 %r1, 1024;              // Restored n value\n"
"    mov.u32 %r5, 42;                // Restored thread index\n"
"    // Jump to correct instruction based on IP\n"
"    setp.eq.u64 %__restore_active, %__restore_ip, 0;\n"
"    @%__restore_active bra __checkpoint_ip_0;\n"
"    setp.eq.u64 %__restore_active, %__restore_ip, 1;\n"
"    @%__restore_active bra __checkpoint_ip_1;\n"
"    setp.eq.u64 %__restore_active, %__restore_ip, 2;\n"
"    @%__restore_active bra __checkpoint_ip_2;\n"
"    setp.eq.u64 %__restore_active, %__restore_ip, 3;\n"
"    @%__restore_active bra __checkpoint_ip_3;\n"
"\n"
"__normal_start:\n"
"    // === Normal execution starts here ===\n"
"\n"
"__checkpoint_ip_0:\n"
"    ld.param.u64 %rd1, [data];\n"
"    ld.param.s32 %r1, [n];\n"
"\n"
"__checkpoint_ip_1:\n"
"    mov.u32 %r2, %ctaid.x;\n"
"    mov.u32 %r3, %ntid.x;\n"
"    mov.u32 %r4, %tid.x;\n"
"    mad.lo.s32 %r5, %r2, %r3, %r4;\n"
"\n"
"__checkpoint_ip_2:\n"
"    setp.ge.s32 %p1, %r5, %r1;\n"
"    @%p1 bra END;\n"
"\n"
"__checkpoint_ip_3:\n"
"    mul.wide.s32 %rd2, %r5, 4;\n"
"    add.s64 %rd3, %rd1, %rd2;\n"
"    st.global.s32 [%rd3], %r5;\n"
"\n"
"END:\n"
"    ret;\n"
"}\n";

void print_migration_explanation() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════╗\n");
    printf("║         Instruction-Level Checkpoint/Restore Migration             ║\n");
    printf("╠════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                    ║\n");
    printf("║  Similar to WebAssembly's migration approach, we instrument PTX    ║\n");
    printf("║  with checkpoint markers at each instruction, allowing execution   ║\n");
    printf("║  to resume from any point.                                         ║\n");
    printf("║                                                                    ║\n");
    printf("║  How it works:                                                     ║\n");
    printf("║  1. Parse original PTX and identify instruction boundaries         ║\n");
    printf("║  2. Add labels at each instruction (__checkpoint_ip_N)             ║\n");
    printf("║  3. Insert restore dispatcher at kernel entry                      ║\n");
    printf("║  4. On restore: load register state, then jump to correct IP       ║\n");
    printf("║                                                                    ║\n");
    printf("║  This enables:                                                     ║\n");
    printf("║  • Migrating execution mid-kernel between different GPUs           ║\n");
    printf("║  • Heterogeneous migration (e.g., NVIDIA -> AMD via PTX)           ║\n");
    printf("║  • Fine-grained fault tolerance with minimal recomputation         ║\n");
    printf("║                                                                    ║\n");
    printf("╚════════════════════════════════════════════════════════════════════╝\n\n");
}

int main(int argc, char** argv) {
    printf("=================================================\n");
    printf("  Instruction-Level PTX Migration Test\n");
    printf("=================================================\n\n");

    print_migration_explanation();

    printf("=== Original PTX (before instrumentation) ===\n");
    printf("%s\n", ORIGINAL_PTX);

    printf("\n=== Instrumented PTX (for resuming at IP=3) ===\n");
    printf("%s\n", INSTRUMENTED_PTX_EXAMPLE);

    printf("\n=== Migration Flow ===\n");
    printf("1. Kernel executes instructions IP=0, IP=1, IP=2 on Source GPU\n");
    printf("2. Checkpoint triggered at IP=2 (saves registers: %%rd1, %%r1, %%r5...)\n");
    printf("3. PTX is instrumented with restore dispatcher\n");
    printf("4. On Target GPU:\n");
    printf("   a. Kernel starts, hits restore dispatcher\n");
    printf("   b. Registers restored: %%rd1=0x7f..., %%r1=1024, %%r5=42\n");
    printf("   c. Jumps to __checkpoint_ip_3\n");
    printf("   d. Execution continues from IP=3\n");
    printf("5. Kernel completes normally\n");

    // Try to load and test with actual CUDA
    void* cuda_lib = dlopen(NULL, RTLD_NOW | RTLD_GLOBAL);
    if (cuda_lib) {
        cuInit_fn cuInit = (cuInit_fn)dlsym(cuda_lib, "cuInit");
        if (cuInit && cuInit(0) == CUDA_SUCCESS) {
            printf("\n=== Live Test with CUDA ===\n");

            cuDeviceGet_fn cuDeviceGet = (cuDeviceGet_fn)dlsym(cuda_lib, "cuDeviceGet");
            cuCtxCreate_fn cuCtxCreate = (cuCtxCreate_fn)dlsym(cuda_lib, "cuCtxCreate_v2");
            cuModuleLoadData_fn cuModuleLoadData = (cuModuleLoadData_fn)dlsym(cuda_lib, "cuModuleLoadData");
            cuModuleGetFunction_fn cuModuleGetFunction = (cuModuleGetFunction_fn)dlsym(cuda_lib, "cuModuleGetFunction");

            if (cuDeviceGet && cuCtxCreate) {
                CUdevice device;
                CUcontext context;
                CUmodule module;

                if (cuDeviceGet(&device, 0) == CUDA_SUCCESS &&
                    cuCtxCreate(&context, 0, device) == CUDA_SUCCESS) {

                    printf("[OK] CUDA context created\n");

                    // Test loading original PTX
                    if (cuModuleLoadData && cuModuleLoadData(&module, ORIGINAL_PTX) == CUDA_SUCCESS) {
                        printf("[OK] Original PTX loaded successfully\n");

                        if (cuModuleGetFunction) {
                            CUfunction func;
                            if (cuModuleGetFunction(&func, module, "migrate_kernel") == CUDA_SUCCESS) {
                                printf("[OK] Kernel function 'migrate_kernel' obtained\n");
                            }
                        }
                    } else {
                        printf("[--] Original PTX loading failed (expected on non-NVIDIA)\n");
                    }

                    // Test loading instrumented PTX
                    if (cuModuleLoadData && cuModuleLoadData(&module, INSTRUMENTED_PTX_EXAMPLE) == CUDA_SUCCESS) {
                        printf("[OK] Instrumented PTX loaded successfully\n");
                    } else {
                        printf("[--] Instrumented PTX loading failed (expected on non-NVIDIA)\n");
                    }
                }
            }
        }
    }

    printf("\n=== Test Complete ===\n");
    return 0;
}
