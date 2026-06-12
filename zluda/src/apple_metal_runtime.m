#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "ane_bridge.h"
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    HETGPU_CUDA_R_32F = 0,
    HETGPU_CUDA_R_16F = 2,
};

typedef struct {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t transa;
    uint32_t transb;
    float alpha;
    float beta;
} HetGpuGemmArgs;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> gemm_f32;
    id<MTLComputePipelineState> gemm_f16;
} HetGpuMetalRuntime;

static NSString *hetgpu_metal_source(void) {
    return @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct GemmArgs { uint m; uint n; uint k; uint lda; uint ldb; uint ldc; uint transa; uint transb; float alpha; float beta; };\n"
           "kernel void hetgpu_gemm_f32(const device float* A [[buffer(0)]], const device float* B [[buffer(1)]], device float* C [[buffer(2)]], constant GemmArgs& args [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {\n"
           "  uint row = gid.x; uint col = gid.y; if (row >= args.m || col >= args.n) return;\n"
           "  float sum = 0.0f;\n"
           "  for (uint p = 0; p < args.k; ++p) {\n"
           "    float av = args.transa ? A[row * args.lda + p] : A[p * args.lda + row];\n"
           "    float bv = args.transb ? B[p * args.ldb + col] : B[col * args.ldb + p];\n"
           "    sum += av * bv;\n"
           "  }\n"
           "  uint ci = col * args.ldc + row; C[ci] = args.alpha * sum + args.beta * C[ci];\n"
           "}\n"
           "kernel void hetgpu_gemm_f16(const device half* A [[buffer(0)]], const device half* B [[buffer(1)]], device half* C [[buffer(2)]], constant GemmArgs& args [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {\n"
           "  uint row = gid.x; uint col = gid.y; if (row >= args.m || col >= args.n) return;\n"
           "  float sum = 0.0f;\n"
           "  for (uint p = 0; p < args.k; ++p) {\n"
           "    float av = args.transa ? float(A[row * args.lda + p]) : float(A[p * args.lda + row]);\n"
           "    float bv = args.transb ? float(B[p * args.ldb + col]) : float(B[col * args.ldb + p]);\n"
           "    sum += av * bv;\n"
           "  }\n"
           "  uint ci = col * args.ldc + row; C[ci] = half(args.alpha * sum + args.beta * float(C[ci]));\n"
           "}\n";
}

static id<MTLComputePipelineState> hetgpu_make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "[hetGPU Metal] missing kernel %s\n", [name UTF8String]);
        return nil;
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "[hetGPU Metal] pipeline compile failed for %s: %s\n", [name UTF8String], [[error localizedDescription] UTF8String]);
    }
    return pipeline;
}

static HetGpuMetalRuntime *hetgpu_runtime(void) {
    static HetGpuMetalRuntime *runtime = NULL;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                fprintf(stderr, "[hetGPU Metal] no Metal device available\n");
                return;
            }

            NSError *error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:hetgpu_metal_source() options:nil error:&error];
            if (!library) {
                fprintf(stderr, "[hetGPU Metal] runtime compilation failed: %s\n", [[error localizedDescription] UTF8String]);
                return;
            }

            HetGpuMetalRuntime *created = (HetGpuMetalRuntime *)calloc(1, sizeof(HetGpuMetalRuntime));
            created->device = device;
            created->queue = [device newCommandQueue];
            created->gemm_f32 = hetgpu_make_pipeline(device, library, @"hetgpu_gemm_f32");
            created->gemm_f16 = hetgpu_make_pipeline(device, library, @"hetgpu_gemm_f16");

            if (!created->queue || !created->gemm_f32 || !created->gemm_f16) {
                free(created);
                return;
            }
            runtime = created;
            fprintf(stderr, "[hetGPU Metal] initialized device: %s\n", [[device name] UTF8String]);
        }
    });
    return runtime;
}

static size_t hetgpu_matrix_elements(int rows, int cols, int leading_dim, int transposed) {
    if (transposed) {
        return (size_t)rows * (size_t)leading_dim;
    }
    return (size_t)cols * (size_t)leading_dim;
}

typedef struct {
    int m;
    int n;
    int k;
    size_t input_bytes;
    size_t output_bytes;
    ANEKernelHandle *kernel;
} HetGpuAneGemmCache;

static pthread_mutex_t g_ane_gemm_lock = PTHREAD_MUTEX_INITIALIZER;
static HetGpuAneGemmCache g_ane_gemm_cache = {0};

static float hetgpu_runtime_f16_to_f32(uint16_t bits) {
    _Float16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
}

static uint16_t hetgpu_runtime_f32_to_f16(float value) {
    _Float16 half = (_Float16)value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static NSString *hetgpu_ane_dyn_matmul_mil(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    int sp = seq + oc;
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        @"{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sp];
    [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, seq];
    [m appendString:@"        tensor<fp16, [1,"];
    [m appendFormat:@"%d", ic];
    [m appendFormat:@",1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", seq];
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", ic, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", seq, ic];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", ic, oc];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", seq, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", oc, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

static ANEKernelHandle *hetgpu_ane_get_gemm_kernel(int m, int n, int k, size_t input_bytes, size_t output_bytes) {
    if (ane_bridge_init() != 0) {
        return NULL;
    }

    if (g_ane_gemm_cache.kernel &&
        g_ane_gemm_cache.m == m &&
        g_ane_gemm_cache.n == n &&
        g_ane_gemm_cache.k == k &&
        g_ane_gemm_cache.input_bytes == input_bytes &&
        g_ane_gemm_cache.output_bytes == output_bytes) {
        return g_ane_gemm_cache.kernel;
    }

    if (g_ane_gemm_cache.kernel) {
        ane_bridge_free(g_ane_gemm_cache.kernel);
        memset(&g_ane_gemm_cache, 0, sizeof(g_ane_gemm_cache));
    }

    NSString *mil = hetgpu_ane_dyn_matmul_mil(k, m, n);
    const char *mil_text = [mil UTF8String];
    size_t input_sizes[1] = { input_bytes };
    size_t output_sizes[1] = { output_bytes };
    ANEKernelHandle *kernel = ane_bridge_compile(
        mil_text, strlen(mil_text), NULL, 0, 1, input_sizes, 1, output_sizes);
    if (!kernel) {
        return NULL;
    }

    g_ane_gemm_cache.m = m;
    g_ane_gemm_cache.n = n;
    g_ane_gemm_cache.k = k;
    g_ane_gemm_cache.input_bytes = input_bytes;
    g_ane_gemm_cache.output_bytes = output_bytes;
    g_ane_gemm_cache.kernel = kernel;
    return kernel;
}

int hetgpu_apple_ane_gemm(
    int transa,
    int transb,
    int m,
    int n,
    int k,
    float alpha,
    const void *A,
    int Atype,
    int lda,
    const void *B,
    int Btype,
    int ldb,
    float beta,
    void *C,
    int Ctype,
    int ldc
) {
    if (!A || !B || !C || m <= 0 || n <= 0 || k <= 0) {
        return -1;
    }
    if (transa || transb || Atype != HETGPU_CUDA_R_16F || Btype != HETGPU_CUDA_R_16F || Ctype != HETGPU_CUDA_R_16F) {
        return -2;
    }
    if (lda < m || ldb < k || ldc < m) {
        return -3;
    }

    @autoreleasepool {
        const size_t input_elems = (size_t)k * (size_t)(n + m);
        const size_t output_elems = (size_t)m * (size_t)n;
        const size_t input_bytes = input_elems * sizeof(uint16_t);
        const size_t output_bytes = output_elems * sizeof(uint16_t);
        uint16_t *packed = (uint16_t *)calloc(input_elems, sizeof(uint16_t));
        uint16_t *out = (uint16_t *)calloc(output_elems, sizeof(uint16_t));
        if (!packed || !out) {
            free(packed);
            free(out);
            return -4;
        }

        const uint16_t *a16 = (const uint16_t *)A;
        const uint16_t *b16 = (const uint16_t *)B;
        uint16_t *c16 = (uint16_t *)C;
        const int sp = n + m;
        for (int p = 0; p < k; ++p) {
            for (int col = 0; col < n; ++col) {
                packed[(size_t)p * (size_t)sp + (size_t)col] = b16[(size_t)col * (size_t)ldb + (size_t)p];
            }
            for (int row = 0; row < m; ++row) {
                packed[(size_t)p * (size_t)sp + (size_t)n + (size_t)row] = a16[(size_t)p * (size_t)lda + (size_t)row];
            }
        }

        pthread_mutex_lock(&g_ane_gemm_lock);
        ANEKernelHandle *kernel = hetgpu_ane_get_gemm_kernel(m, n, k, input_bytes, output_bytes);
        int result = 0;
        if (!kernel) {
            result = -5;
        } else {
            ane_bridge_write_input(kernel, 0, packed, input_bytes);
            if (!ane_bridge_eval(kernel)) {
                result = -6;
            } else {
                ane_bridge_read_output(kernel, 0, out, output_bytes);
            }
        }
        pthread_mutex_unlock(&g_ane_gemm_lock);

        if (result != 0) {
            free(packed);
            free(out);
            return result;
        }

        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                size_t dense_idx = (size_t)row * (size_t)n + (size_t)col;
                size_t c_idx = (size_t)col * (size_t)ldc + (size_t)row;
                float computed = hetgpu_runtime_f16_to_f32(out[dense_idx]);
                float previous = hetgpu_runtime_f16_to_f32(c16[c_idx]);
                c16[c_idx] = hetgpu_runtime_f32_to_f16(alpha * computed + beta * previous);
            }
        }

        free(packed);
        free(out);
    }

    return 0;
}

int hetgpu_apple_metal_gemm(
    int transa,
    int transb,
    int m,
    int n,
    int k,
    float alpha,
    const void *A,
    int Atype,
    int lda,
    const void *B,
    int Btype,
    int ldb,
    float beta,
    void *C,
    int Ctype,
    int ldc
) {
    if (!A || !B || !C || m <= 0 || n <= 0 || k <= 0) {
        return -1;
    }
    if (Atype != Btype || Atype != Ctype || (Atype != HETGPU_CUDA_R_32F && Atype != HETGPU_CUDA_R_16F)) {
        return -2;
    }

    HetGpuMetalRuntime *runtime = hetgpu_runtime();
    if (!runtime) {
        return -3;
    }

    @autoreleasepool {
        const size_t elem_size = (Atype == HETGPU_CUDA_R_16F) ? sizeof(uint16_t) : sizeof(float);
        const size_t a_elems = hetgpu_matrix_elements(m, k, lda, transa);
        const size_t b_elems = hetgpu_matrix_elements(k, n, ldb, transb);
        const size_t c_elems = (size_t)n * (size_t)ldc;
        const size_t a_bytes = a_elems * elem_size;
        const size_t b_bytes = b_elems * elem_size;
        const size_t c_bytes = c_elems * elem_size;

        id<MTLBuffer> a_buffer = [runtime->device newBufferWithBytes:A length:a_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buffer = [runtime->device newBufferWithBytes:B length:b_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> c_buffer = [runtime->device newBufferWithBytes:C length:c_bytes options:MTLResourceStorageModeShared];
        HetGpuGemmArgs args = {
            .m = (uint32_t)m,
            .n = (uint32_t)n,
            .k = (uint32_t)k,
            .lda = (uint32_t)lda,
            .ldb = (uint32_t)ldb,
            .ldc = (uint32_t)ldc,
            .transa = transa ? 1u : 0u,
            .transb = transb ? 1u : 0u,
            .alpha = alpha,
            .beta = beta,
        };
        id<MTLBuffer> args_buffer = [runtime->device newBufferWithBytes:&args length:sizeof(args) options:MTLResourceStorageModeShared];
        if (!a_buffer || !b_buffer || !c_buffer || !args_buffer) {
            return -4;
        }

        id<MTLCommandBuffer> command_buffer = [runtime->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        id<MTLComputePipelineState> pipeline = (Atype == HETGPU_CUDA_R_16F) ? runtime->gemm_f16 : runtime->gemm_f32;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:c_buffer offset:0 atIndex:2];
        [encoder setBuffer:args_buffer offset:0 atIndex:3];

        const NSUInteger width = pipeline.threadExecutionWidth;
        const NSUInteger height = MAX((NSUInteger)1, (NSUInteger)(pipeline.maxTotalThreadsPerThreadgroup / width));
        MTLSize threads_per_group = MTLSizeMake(width, height, 1);
        MTLSize threads = MTLSizeMake((NSUInteger)m, (NSUInteger)n, 1);
        [encoder dispatchThreads:threads threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if ([command_buffer status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "[hetGPU Metal] GEMM command failed: %s\n", [[command_buffer.error localizedDescription] UTF8String]);
            return -5;
        }

        memcpy(C, [c_buffer contents], c_bytes);
    }

    return 0;
}
