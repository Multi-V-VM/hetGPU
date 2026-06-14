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

enum {
    HETGPU_METAL_BUFFER_COPY_IN = 1,
    HETGPU_METAL_BUFFER_COPY_OUT = 2,
};

typedef struct {
    void *host_ptr;
    size_t size;
    uint32_t flags;
} HetGpuMetalBufferBinding;

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

typedef struct {
    void *device;
    void *library;
} HetGpuMetalModule;

typedef struct {
    void *pipeline;
} HetGpuMetalFunction;

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

static char *hetgpu_copy_c_string(NSString *message) {
    if (!message) {
        return NULL;
    }
    const char *utf8 = [message UTF8String];
    if (!utf8) {
        return NULL;
    }
    size_t len = strlen(utf8);
    char *copy = (char *)malloc(len + 1);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, utf8, len + 1);
    return copy;
}

static void hetgpu_set_log(char **out_log, NSString *message) {
    if (out_log) {
        *out_log = hetgpu_copy_c_string(message);
    }
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

void hetgpu_apple_metal_free_string(char *value) {
    free(value);
}

int hetgpu_apple_metal_compile_msl(
    const char *source,
    const char *label,
    void **out_module,
    char **out_log
) {
    if (out_log) {
        *out_log = NULL;
    }
    if (!source || !out_module) {
        hetgpu_set_log(out_log, @"invalid argument");
        return -1;
    }
    *out_module = NULL;

    HetGpuMetalRuntime *runtime = hetgpu_runtime();
    if (!runtime || !runtime->device) {
        hetgpu_set_log(out_log, @"no Metal device available");
        return -2;
    }

    @autoreleasepool {
        NSString *source_string = [NSString stringWithUTF8String:source];
        if (!source_string) {
            hetgpu_set_log(out_log, @"MSL source is not valid UTF-8");
            return -3;
        }

        NSError *error = nil;
        id<MTLLibrary> library = [runtime->device newLibraryWithSource:source_string options:nil error:&error];
        if (!library) {
            hetgpu_set_log(out_log, error ? [error localizedDescription] : @"Metal library compilation failed");
            return -4;
        }
        if (label) {
            NSString *label_string = [NSString stringWithUTF8String:label];
            if (label_string) {
                library.label = label_string;
            }
        }

        HetGpuMetalModule *module = (HetGpuMetalModule *)calloc(1, sizeof(HetGpuMetalModule));
        if (!module) {
            hetgpu_set_log(out_log, @"out of memory");
            return -5;
        }
        module->device = (void *)CFBridgingRetain(runtime->device);
        module->library = (void *)CFBridgingRetain(library);
        *out_module = module;
        return 0;
    }
}

int hetgpu_apple_metal_get_function(
    void *module_ptr,
    const char *name,
    void **out_function,
    char **out_log
) {
    if (out_log) {
        *out_log = NULL;
    }
    if (!module_ptr || !name || !out_function) {
        hetgpu_set_log(out_log, @"invalid argument");
        return -1;
    }
    *out_function = NULL;

    HetGpuMetalModule *module = (HetGpuMetalModule *)module_ptr;
    id<MTLDevice> device = (__bridge id<MTLDevice>)module->device;
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)module->library;
    if (!device || !library) {
        hetgpu_set_log(out_log, @"invalid Metal module");
        return -2;
    }

    @autoreleasepool {
        NSString *function_name = [NSString stringWithUTF8String:name];
        if (!function_name) {
            hetgpu_set_log(out_log, @"function name is not valid UTF-8");
            return -3;
        }

        id<MTLFunction> fn = [library newFunctionWithName:function_name];
        if (!fn) {
            hetgpu_set_log(out_log, [NSString stringWithFormat:@"missing Metal kernel '%@'", function_name]);
            return -4;
        }

        NSError *error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            hetgpu_set_log(out_log, error ? [error localizedDescription] : @"Metal pipeline creation failed");
            return -5;
        }

        HetGpuMetalFunction *function = (HetGpuMetalFunction *)calloc(1, sizeof(HetGpuMetalFunction));
        if (!function) {
            hetgpu_set_log(out_log, @"out of memory");
            return -6;
        }
        function->pipeline = (void *)CFBridgingRetain(pipeline);
        *out_function = function;
        return 0;
    }
}

int hetgpu_apple_metal_launch_raw(
    void *function_ptr,
    const HetGpuMetalBufferBinding *buffers,
    size_t buffer_count,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z,
    uint32_t block_x,
    uint32_t block_y,
    uint32_t block_z,
    char **out_log
) {
    if (out_log) {
        *out_log = NULL;
    }
    if (!function_ptr || (!buffers && buffer_count > 0) || grid_x == 0 || grid_y == 0 || grid_z == 0) {
        hetgpu_set_log(out_log, @"invalid argument");
        return -1;
    }

    HetGpuMetalRuntime *runtime = hetgpu_runtime();
    if (!runtime || !runtime->device || !runtime->queue) {
        hetgpu_set_log(out_log, @"no Metal runtime available");
        return -2;
    }

    HetGpuMetalFunction *function = (HetGpuMetalFunction *)function_ptr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)function->pipeline;
    if (!pipeline) {
        hetgpu_set_log(out_log, @"invalid Metal function");
        return -3;
    }

    @autoreleasepool {
        NSMutableArray<id<MTLBuffer>> *metal_buffers = [NSMutableArray arrayWithCapacity:buffer_count];
        for (size_t i = 0; i < buffer_count; ++i) {
            HetGpuMetalBufferBinding binding = buffers[i];
            if (binding.size == 0 || ((binding.flags & HETGPU_METAL_BUFFER_COPY_OUT) && !binding.host_ptr)) {
                hetgpu_set_log(out_log, @"invalid Metal buffer binding");
                return -4;
            }
            id<MTLBuffer> buffer = nil;
            if ((binding.flags & HETGPU_METAL_BUFFER_COPY_IN) && binding.host_ptr) {
                buffer = [runtime->device newBufferWithBytes:binding.host_ptr length:binding.size options:MTLResourceStorageModeShared];
            } else {
                buffer = [runtime->device newBufferWithLength:binding.size options:MTLResourceStorageModeShared];
            }
            if (!buffer) {
                hetgpu_set_log(out_log, @"failed to allocate Metal buffer");
                return -5;
            }
            [metal_buffers addObject:buffer];
        }

        id<MTLCommandBuffer> command_buffer = [runtime->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!command_buffer || !encoder) {
            hetgpu_set_log(out_log, @"failed to create Metal command encoder");
            return -6;
        }

        [encoder setComputePipelineState:pipeline];
        for (NSUInteger i = 0; i < [metal_buffers count]; ++i) {
            [encoder setBuffer:[metal_buffers objectAtIndex:i] offset:0 atIndex:i];
        }

        if (block_x == 0) {
            block_x = (uint32_t)MAX((NSUInteger)1, pipeline.threadExecutionWidth);
        }
        if (block_y == 0) {
            block_y = 1;
        }
        if (block_z == 0) {
            block_z = 1;
        }

        MTLSize threads = MTLSizeMake((NSUInteger)grid_x, (NSUInteger)grid_y, (NSUInteger)grid_z);
        MTLSize threads_per_group = MTLSizeMake((NSUInteger)block_x, (NSUInteger)block_y, (NSUInteger)block_z);
        [encoder dispatchThreads:threads threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if ([command_buffer status] == MTLCommandBufferStatusError) {
            hetgpu_set_log(out_log, command_buffer.error ? [command_buffer.error localizedDescription] : @"Metal command failed");
            return -7;
        }

        for (size_t i = 0; i < buffer_count; ++i) {
            HetGpuMetalBufferBinding binding = buffers[i];
            if (binding.flags & HETGPU_METAL_BUFFER_COPY_OUT) {
                id<MTLBuffer> buffer = [metal_buffers objectAtIndex:(NSUInteger)i];
                memcpy(binding.host_ptr, [buffer contents], binding.size);
            }
        }
        return 0;
    }
}

int hetgpu_apple_metal_release_module(void *module_ptr) {
    if (!module_ptr) {
        return 0;
    }
    HetGpuMetalModule *module = (HetGpuMetalModule *)module_ptr;
    if (module->library) {
        CFRelease(module->library);
    }
    if (module->device) {
        CFRelease(module->device);
    }
    free(module);
    return 0;
}

int hetgpu_apple_metal_release_function(void *function_ptr) {
    if (!function_ptr) {
        return 0;
    }
    HetGpuMetalFunction *function = (HetGpuMetalFunction *)function_ptr;
    if (function->pipeline) {
        CFRelease(function->pipeline);
    }
    free(function);
    return 0;
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
