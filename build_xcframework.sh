#!/usr/bin/env bash
set -euo pipefail

echo "Building hetGPU Apple ANE/Metal runtime as XCFramework..."

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${HETGPU_XCFRAMEWORK_BUILD_DIR:-$ROOT_DIR/target/xcframework}"
OUT_DIR="${HETGPU_XCFRAMEWORK_OUT:-$ROOT_DIR/HetGPUAppleRuntime.xcframework}"
HEADERS_DIR="$BUILD_DIR/Headers"
COMPAT_INCLUDE_DIR="$BUILD_DIR/SDKCompat"
IOS_MIN_VERSION="${HETGPU_IOS_MIN_VERSION:-15.0}"
LIB_NAME="libhetgpu_apple_runtime.a"
RUST_TARGET_DIR="${HETGPU_RUST_TARGET_DIR:-$BUILD_DIR/rust-target}"
SKIP_RUST_PTX="${HETGPU_SKIP_RUST_PTX:-0}"

SOURCES=(
    "$ROOT_DIR/zluda/src/apple_cuda_stub.m"
    "$ROOT_DIR/zluda/src/apple_metal_runtime.m"
    "$ROOT_DIR/zluda/src/ane_bridge.m"
    "$ROOT_DIR/zluda/src/cudart_shim.c"
    "$ROOT_DIR/zluda/src/cublas_shim.c"
    "$ROOT_DIR/zluda/src/cublaslt_shim.c"
    "$ROOT_DIR/zluda/src/cusparse_shim.c"
    "$ROOT_DIR/zluda/src/cufft_shim.c"
    "$ROOT_DIR/zluda/src/nccl_shim.c"
    "$ROOT_DIR/zluda/src/torch_abi_shim.c"
    "$ROOT_DIR/zluda/src/pacc_disabled_stubs.c"
)

die() {
    printf 'build_xcframework: %s\n' "$*" >&2
    exit 1
}

require_tool() {
    command -v "$1" >/dev/null || die "$1 not found"
}

find_cargo() {
    local candidate

    if [[ -n "${CARGO:-}" && -x "${CARGO:-}" ]]; then
        printf '%s' "$CARGO"
        return 0
    fi

    if command -v cargo >/dev/null; then
        command -v cargo
        return 0
    fi

    for candidate in \
        /opt/homebrew/bin/cargo \
        /usr/local/bin/cargo \
        /opt/homebrew/opt/rustup/bin/cargo \
        /opt/homebrew/Cellar/rustup/*/bin/cargo \
        /opt/homebrew/Cellar/rustup/*/libexec/bin/cargo
    do
        if [[ -x "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done

    return 1
}

prepare_headers() {
    rm -rf "$HEADERS_DIR"
    mkdir -p "$HEADERS_DIR"
    cp "$ROOT_DIR/zluda/src/ane_bridge.h" "$HEADERS_DIR/ane_bridge.h"
    cat > "$HEADERS_DIR/hetgpu_apple_runtime.h" <<'EOF'
#ifndef HETGPU_APPLE_RUNTIME_H
#define HETGPU_APPLE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include "ane_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

int hetgpu_ane_gemm(int transa, int transb,
                    int m, int n, int k,
                    float alpha,
                    const void *A, int Atype, int lda,
                    const void *B, int Btype, int ldb,
                    float beta,
                    void *C, int Ctype, int ldc);

int hetgpu_apple_ane_gemm(int transa, int transb,
                          int m, int n, int k,
                          float alpha,
                          const void *A, int Atype, int lda,
                          const void *B, int Btype, int ldb,
                          float beta,
                          void *C, int Ctype, int ldc);

int hetgpu_apple_metal_gemm(int transa, int transb,
                            int m, int n, int k,
                            float alpha,
                            const void *A, int Atype, int lda,
                            const void *B, int Btype, int ldb,
                            float beta,
                            void *C, int Ctype, int ldc);

enum {
    HETGPU_METAL_BUFFER_COPY_IN = 1,
    HETGPU_METAL_BUFFER_COPY_OUT = 2
};

typedef struct HetGpuMetalBufferBinding {
    void *host_ptr;
    size_t size;
    uint32_t flags;
} HetGpuMetalBufferBinding;

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUdeviceptr;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;

CUresult cuInit(unsigned int flags);
CUresult cuDriverGetVersion(int *driver_version);
CUresult cuDeviceGetCount(int *count);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetName(char *name, int len, CUdevice device);
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice device);
CUresult cuDeviceGetAttribute(int *value, int attribute, CUdevice device);
CUresult cuCtxCreate_v2(CUcontext *context, unsigned int flags, CUdevice device);
CUresult cuCtxDestroy_v2(CUcontext context);
CUresult cuCtxSetCurrent(CUcontext context);
CUresult cuCtxGetCurrent(CUcontext *context);
CUresult cuCtxSynchronize(void);
CUresult cuMemAlloc_v2(CUdeviceptr *device_ptr, size_t size);
CUresult cuMemFree_v2(CUdeviceptr device_ptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst_device, const void *src_host, size_t size);
CUresult cuMemcpyDtoH_v2(void *dst_host, CUdeviceptr src_device, size_t size);
CUresult cuMemcpyDtoD_v2(CUdeviceptr dst_device, CUdeviceptr src_device, size_t size);
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleLoadDataEx(CUmodule *module,
                            const void *image,
                            unsigned int num_options,
                            void *options,
                            void *option_values);
CUresult cuModuleUnload(CUmodule module);
CUresult cuModuleGetFunction(CUfunction *function, CUmodule module, const char *name);
CUresult cuLaunchKernel(CUfunction function,
                        unsigned int grid_dim_x,
                        unsigned int grid_dim_y,
                        unsigned int grid_dim_z,
                        unsigned int block_dim_x,
                        unsigned int block_dim_y,
                        unsigned int block_dim_z,
                        unsigned int shared_mem_bytes,
                        CUstream stream,
                        void **kernel_params,
                        void **extra);

CUresult hetgpu_apple_ptx_register_allocation(void *ptr, size_t size);
CUresult hetgpu_apple_ptx_unregister_allocation(void *ptr);
CUresult hetgpu_apple_ptx_module_load_data(CUmodule *module, const void *image);
CUresult hetgpu_apple_ptx_module_unload(CUmodule module);
CUresult hetgpu_apple_ptx_module_get_function(CUfunction *function,
                                              CUmodule module,
                                              const char *name);
CUresult hetgpu_apple_ptx_function_release(CUfunction function);
CUresult hetgpu_apple_ptx_launch_kernel(CUfunction function,
                                        unsigned int grid_dim_x,
                                        unsigned int grid_dim_y,
                                        unsigned int grid_dim_z,
                                        unsigned int block_dim_x,
                                        unsigned int block_dim_y,
                                        unsigned int block_dim_z,
                                        unsigned int shared_mem_bytes,
                                        CUstream stream,
                                        void **kernel_params,
                                        void **extra);

int hetgpu_apple_metal_compile_msl(const char *source,
                                   const char *label,
                                   void **out_module,
                                   char **out_log);

int hetgpu_apple_metal_get_function(void *module,
                                    const char *name,
                                    void **out_function,
                                    char **out_log);

int hetgpu_apple_metal_launch_raw(void *function,
                                  const HetGpuMetalBufferBinding *buffers,
                                  size_t buffer_count,
                                  uint32_t grid_x,
                                  uint32_t grid_y,
                                  uint32_t grid_z,
                                  uint32_t block_x,
                                  uint32_t block_y,
                                  uint32_t block_z,
                                  char **out_log);

int hetgpu_apple_metal_release_module(void *module);
int hetgpu_apple_metal_release_function(void *function);
void hetgpu_apple_metal_free_string(char *value);

#ifdef __cplusplus
}
#endif

#endif
EOF
    cat > "$HEADERS_DIR/module.modulemap" <<'EOF'
module HetGPUAppleRuntime {
    umbrella header "hetgpu_apple_runtime.h"
    export *
}
EOF
}

prepare_sdk_compat_headers() {
    rm -rf "$COMPAT_INCLUDE_DIR"
    mkdir -p "$COMPAT_INCLUDE_DIR/IOSurface"
    cat > "$COMPAT_INCLUDE_DIR/IOSurface/IOSurface.h" <<'EOF'
#ifndef HETGPU_IOSURFACE_COMPAT_H
#define HETGPU_IOSURFACE_COMPAT_H

#include <CoreFoundation/CoreFoundation.h>
#include <stdint.h>

typedef struct __IOSurface *IOSurfaceRef;
typedef int kern_return_t;

extern const CFStringRef kIOSurfaceWidth;
extern const CFStringRef kIOSurfaceHeight;
extern const CFStringRef kIOSurfaceBytesPerElement;
extern const CFStringRef kIOSurfaceBytesPerRow;
extern const CFStringRef kIOSurfaceAllocSize;
extern const CFStringRef kIOSurfacePixelFormat;

enum {
    kIOSurfaceLockReadOnly = 0x00000001
};

IOSurfaceRef IOSurfaceCreate(CFDictionaryRef properties);
kern_return_t IOSurfaceLock(IOSurfaceRef buffer, uint32_t options, uint32_t *seed);
kern_return_t IOSurfaceUnlock(IOSurfaceRef buffer, uint32_t options, uint32_t *seed);
void *IOSurfaceGetBaseAddress(IOSurfaceRef buffer);

#endif
EOF
}

compile_source() {
    local clang_path="$1"
    local sdk_path="$2"
    local target="$3"
    local min_flag="$4"
    local source="$5"
    local object="$6"
    local ext

    ext="${source##*.}"
    local flags=(
        -target "$target"
        -isysroot "$sdk_path"
        "$min_flag"
        -I"$COMPAT_INCLUDE_DIR"
        -I"$ROOT_DIR/zluda/src"
        -fPIC
        -DHETGPU_STATIC_CUDART
        -Wno-unused-parameter
        -Wno-unused-function
        -Wno-deprecated-declarations
        -Wno-duplicate-decl-specifier
        -c "$source"
        -o "$object"
    )

    if [[ "$ext" == "m" ]]; then
        flags=(-fobjc-arc "${flags[@]}")
    fi

    "$clang_path" "${flags[@]}"
}

build_rust_ptx_bridge() {
    local cargo_target="$1"
    local slice_dir="$2"
    local output
    local cargo_bin

    if [[ "$SKIP_RUST_PTX" == "1" ]]; then
        printf 'Skipping Rust PTX bridge for %s because HETGPU_SKIP_RUST_PTX=1\n' "$cargo_target" >&2
        return 0
    fi

    cargo_bin="$(find_cargo)" || die "cargo not found; install Rust or set CARGO=/path/to/cargo"
    printf 'Building Rust PTX bridge (%s)...\n' "$cargo_target" >&2
    output="$RUST_TARGET_DIR/$cargo_target/release/libhetgpu_apple_ptx_bridge.a"
    rm -f "$output"
    (
        cd "$ROOT_DIR"
        PATH="$(dirname "$cargo_bin"):$PATH" \
        CARGO_TARGET_DIR="$RUST_TARGET_DIR" \
            "$cargo_bin" build -p apple_ptx_bridge --release --target "$cargo_target" --no-default-features
    ) || die "Rust PTX bridge build failed for $cargo_target"

    [[ -f "$output" ]] || die "missing Rust PTX bridge archive: $output"
    printf '%s' "$output"
}

build_static_lib() {
    local sdk_name="$1"
    local target="$2"
    local name="$3"
    local min_flag="$4"
    local cargo_target="$5"
    local sdk_path
    local clang_path
    local libtool_path
    local slice_dir
    local obj_dir
    local objects=()

    sdk_path="$(xcrun --sdk "$sdk_name" --show-sdk-path)"
    clang_path="$(xcrun --sdk "$sdk_name" --find clang)"
    libtool_path="$(xcrun --sdk "$sdk_name" --find libtool)"
    slice_dir="$BUILD_DIR/$name"
    obj_dir="$slice_dir/objects"

    rm -rf "$slice_dir"
    mkdir -p "$obj_dir"

    printf 'Building %s (%s)...\n' "$name" "$target"
    for source in "${SOURCES[@]}"; do
        [[ -f "$source" ]] || die "missing source: $source"
        local base
        local object
        base="$(basename "$source")"
        object="$obj_dir/${base%.*}.o"
        compile_source "$clang_path" "$sdk_path" "$target" "$min_flag" "$source" "$object"
        objects+=("$object")
    done
    local rust_ptx_lib
    rust_ptx_lib="$(build_rust_ptx_bridge "$cargo_target" "$slice_dir")"
    if [[ -n "$rust_ptx_lib" ]]; then
        objects+=("$rust_ptx_lib")
    fi

    "$libtool_path" -static -o "$slice_dir/$LIB_NAME" "${objects[@]}"
}

require_tool xcrun
require_tool xcodebuild
prepare_headers
prepare_sdk_compat_headers

rm -rf "$OUT_DIR"

build_static_lib iphoneos arm64-apple-ios iphoneos-arm64 "-mios-version-min=$IOS_MIN_VERSION" aarch64-apple-ios
build_static_lib iphonesimulator arm64-apple-ios-simulator iphonesimulator-arm64 "-mios-simulator-version-min=$IOS_MIN_VERSION" aarch64-apple-ios-sim

xcodebuild -create-xcframework \
    -library "$BUILD_DIR/iphoneos-arm64/$LIB_NAME" \
    -headers "$HEADERS_DIR" \
    -library "$BUILD_DIR/iphonesimulator-arm64/$LIB_NAME" \
    -headers "$HEADERS_DIR" \
    -output "$OUT_DIR"

cat <<EOF

Created $OUT_DIR

When linking the XCFramework, also link these Apple frameworks:
  Foundation, Metal, IOSurface

Set HETGPU_APPLE_BACKEND=ane or HETGPU_APPLE_BACKEND=metal at runtime to choose
the preferred GEMM backend. ANE automatically falls back to Metal when needed.
EOF
