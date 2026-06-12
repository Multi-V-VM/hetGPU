#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

int hetgpu_webgpu_init(void) {
#if defined(__EMSCRIPTEN__)
    return EM_ASM_INT({
        if (typeof console !== 'undefined') {
            console.log('[hetGPU WebGPU] backend selected');
        }
        return 0;
    });
#else
    return 0;
#endif
}

int hetgpu_webgpu_device_count(void) {
#if defined(__EMSCRIPTEN__)
    return EM_ASM_INT({
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            return 1;
        }
        return 1;
    });
#else
    return 1;
#endif
}

uint64_t hetgpu_webgpu_module_load(const void* image, uintptr_t image_len) {
    static uint64_t next_module = 1;
    (void)image;
#if defined(__EMSCRIPTEN__)
    EM_ASM({
        if (typeof console !== 'undefined') {
            console.log('[hetGPU WebGPU] module load, bytes=' + $0);
        }
    }, image_len);
#else
    fprintf(stderr, "[hetGPU WebGPU] module load, bytes=%zu\n", (size_t)image_len);
#endif
    return next_module++;
}

uint64_t hetgpu_webgpu_get_function(uint64_t module_id, const char* name) {
    static uint64_t next_kernel = 1;
#if defined(__EMSCRIPTEN__)
    EM_ASM({
        if (typeof console !== 'undefined') {
            var n = UTF8ToString($1);
            console.log('[hetGPU WebGPU] get function module=' + $0 + ' name=' + n);
        }
    }, module_id, name);
#else
    fprintf(stderr,
            "[hetGPU WebGPU] get function module=%llu name=%s\n",
            (unsigned long long)module_id,
            name ? name : "<null>");
#endif
    return next_kernel++;
}

int hetgpu_webgpu_launch_kernel(uint64_t kernel_id,
                                const char* name,
                                uint32_t grid_x,
                                uint32_t grid_y,
                                uint32_t grid_z,
                                uint32_t block_x,
                                uint32_t block_y,
                                uint32_t block_z,
                                uint32_t shared_mem,
                                void** kernel_params) {
    (void)kernel_params;
#if defined(__EMSCRIPTEN__)
    EM_ASM({
        if (typeof console !== 'undefined') {
            var n = UTF8ToString($1);
            console.log('[hetGPU WebGPU] launch kernel=' + $0 +
                        ' name=' + n +
                        ' grid=(' + $2 + ',' + $3 + ',' + $4 + ')' +
                        ' block=(' + $5 + ',' + $6 + ',' + $7 + ')' +
                        ' shared=' + $8);
        }
    }, kernel_id, name, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem);
#else
    fprintf(stderr,
            "[hetGPU WebGPU] launch kernel=%llu name=%s grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u\n",
            (unsigned long long)kernel_id,
            name ? name : "<null>",
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_mem);
#endif
    return 0;
}
