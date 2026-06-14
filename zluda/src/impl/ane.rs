use std::ffi::c_void;
use std::os::raw::c_int;

use apple_runtime_sys::{HETGPU_CUDA_R_16F as CUDA_R_16F, HETGPU_CUDA_R_32F as CUDA_R_32F};

/// cuBLAS-compatible GEMM entry used by the C shim for Apple inference.
///
/// `HETGPU_APPLE_BACKEND=ane` tries the maderix/ANE bridge first for supported
/// fp16 inference matmuls, then falls back to the local Metal runtime.
/// `HETGPU_APPLE_BACKEND=metal` skips ANE and runs the Metal path directly.
#[no_mangle]
pub unsafe extern "C" fn hetgpu_ane_gemm(
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: *const c_void,
    atype: c_int,
    lda: c_int,
    b: *const c_void,
    btype: c_int,
    ldb: c_int,
    beta: f32,
    c: *mut c_void,
    ctype: c_int,
    ldc: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return -1;
    }

    if !matches!(
        (atype, btype, ctype),
        (CUDA_R_32F, CUDA_R_32F, CUDA_R_32F) | (CUDA_R_16F, CUDA_R_16F, CUDA_R_16F)
    ) {
        return -2;
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let metal_only = std::env::var("HETGPU_APPLE_BACKEND")
            .map(|backend| backend.eq_ignore_ascii_case("metal"))
            .unwrap_or(false);
        if !metal_only {
            let ane_result = apple_runtime_sys::hetgpu_apple_ane_gemm(
                transa, transb, m, n, k, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc,
            );
            if ane_result == 0 {
                return 0;
            }
        }
        return apple_runtime_sys::hetgpu_apple_metal_gemm(
            transa, transb, m, n, k, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc,
        );
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        let _ = (
            transa, transb, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc,
        );
        -3
    }
}
