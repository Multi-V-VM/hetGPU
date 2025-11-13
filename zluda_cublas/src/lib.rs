#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use libc::{c_double, c_float, c_int, c_void};
use num_complex::Complex;
use num_traits::{Float, One, Zero};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

const CUBLAS_STATUS_SUCCESS: c_int = 0;
const CUBLAS_STATUS_NOT_INITIALIZED: c_int = 1;
const CUBLAS_STATUS_ALLOC_FAILED: c_int = 3;
const CUBLAS_STATUS_INVALID_VALUE: c_int = 7;
const CUBLAS_STATUS_OPERATION_ERROR: c_int = 13;
const CUBLAS_STATUS_NOT_SUPPORTED: c_int = 15;

const CUBLAS_POINTER_MODE_HOST: c_int = 0;
const CUBLAS_POINTER_MODE_DEVICE: c_int = 1;

const CUBLAS_MATH_DEFAULT: c_int = 0;

const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
const CUBLAS_OP_C: c_int = 2;

const CUDA_R_16F: c_int = 2;
const CUDA_R_32F: c_int = 0;
const CUDA_R_64F: c_int = 1;

const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_COMPUTE_64F: c_int = 70;
const CUBLAS_GEMM_DEFAULT: c_int = -1;

type cudaDataType = c_int;

type cublasHandle_t = *mut CublasHandle;

#[repr(C)]
struct CublasHandle {
    id: u64,
    pointer_mode: c_int,
    math_mode: c_int,
    stream: *mut c_void,
    workspace: *mut c_void,
    workspace_size: usize,
}

static HANDLE_COUNTER: AtomicUsize = AtomicUsize::new(1);

#[no_mangle]
pub unsafe extern "C" fn cublasCreate_v2(handle: *mut cublasHandle_t) -> c_int {
    if handle.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let id = HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed) as u64;
    let state = CublasHandle {
        id,
        pointer_mode: CUBLAS_POINTER_MODE_HOST,
        math_mode: CUBLAS_MATH_DEFAULT,
        stream: ptr::null_mut(),
        workspace: ptr::null_mut(),
        workspace_size: 0,
    };
    *handle = Box::into_raw(Box::new(state));
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasDestroy_v2(handle: cublasHandle_t) -> c_int {
    if handle.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    drop(Box::from_raw(handle));
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasSetStream_v2(handle: cublasHandle_t, stream: *mut c_void) -> c_int {
    if handle.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    (*handle).stream = stream;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasSetPointerMode_v2(handle: cublasHandle_t, mode: c_int) -> c_int {
    if handle.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if mode != CUBLAS_POINTER_MODE_HOST {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    (*handle).pointer_mode = mode;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasGetPointerMode_v2(handle: cublasHandle_t, mode: *mut c_int) -> c_int {
    if handle.is_null() || mode.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *mode = (*handle).pointer_mode;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasSetMathMode(handle: cublasHandle_t, mode: c_int) -> c_int {
    if handle.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if mode != CUBLAS_MATH_DEFAULT {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    (*handle).math_mode = mode;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut c_int) -> c_int {
    if handle.is_null() || mode.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *mode = (*handle).math_mode;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasSetWorkspace_v2(
    handle: cublasHandle_t,
    workspace: *mut c_void,
    workspace_size: usize,
) -> c_int {
    if handle.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    (*handle).workspace = workspace;
    (*handle).workspace_size = workspace_size;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasGetVersion_v2(handle: cublasHandle_t, version: *mut c_int) -> c_int {
    if handle.is_null() || version.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *version = 12000;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasSdot_v2(
    handle: cublasHandle_t,
    n: c_int,
    x: *const c_float,
    incx: c_int,
    y: *const c_float,
    incy: c_int,
    result: *mut c_float,
) -> c_int {
    if handle.is_null() || x.is_null() || y.is_null() || result.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (*handle).pointer_mode != CUBLAS_POINTER_MODE_HOST {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    let dot = dot_real_f32(n, x, incx, y, incy);
    match dot {
        Some(value) => {
            *result = value;
            CUBLAS_STATUS_SUCCESS
        }
        None => CUBLAS_STATUS_INVALID_VALUE,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cublasDdot_v2(
    handle: cublasHandle_t,
    n: c_int,
    x: *const c_double,
    incx: c_int,
    y: *const c_double,
    incy: c_int,
    result: *mut c_double,
) -> c_int {
    if handle.is_null() || x.is_null() || y.is_null() || result.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (*handle).pointer_mode != CUBLAS_POINTER_MODE_HOST {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    let dot = dot_real_f64(n, x, incx, y, incy);
    match dot {
        Some(value) => {
            *result = value;
            CUBLAS_STATUS_SUCCESS
        }
        None => CUBLAS_STATUS_INVALID_VALUE,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cublasCdotu_v2(
    handle: cublasHandle_t,
    n: c_int,
    x: *const Complex<c_float>,
    incx: c_int,
    y: *const Complex<c_float>,
    incy: c_int,
    result: *mut Complex<c_float>,
) -> c_int {
    complex_dot(handle, n, x, incx, y, incy, result, false)
}

#[no_mangle]
pub unsafe extern "C" fn cublasCdotc_v2(
    handle: cublasHandle_t,
    n: c_int,
    x: *const Complex<c_float>,
    incx: c_int,
    y: *const Complex<c_float>,
    incy: c_int,
    result: *mut Complex<c_float>,
) -> c_int {
    complex_dot(handle, n, x, incx, y, incy, result, true)
}

#[no_mangle]
pub unsafe extern "C" fn cublasZdotu_v2(
    handle: cublasHandle_t,
    n: c_int,
    x: *const Complex<c_double>,
    incx: c_int,
    y: *const Complex<c_double>,
    incy: c_int,
    result: *mut Complex<c_double>,
) -> c_int {
    complex_dot(handle, n, x, incx, y, incy, result, false)
}

#[no_mangle]
pub unsafe extern "C" fn cublasZdotc_v2(
    handle: cublasHandle_t,
    n: c_int,
    x: *const Complex<c_double>,
    incx: c_int,
    y: *const Complex<c_double>,
    incy: c_int,
    result: *mut Complex<c_double>,
) -> c_int {
    complex_dot(handle, n, x, incx, y, incy, result, true)
}

unsafe fn dot_real_f32(
    n: c_int,
    x: *const c_float,
    incx: c_int,
    y: *const c_float,
    incy: c_int,
) -> Option<c_float> {
    if n < 0 || incx <= 0 || incy <= 0 {
        return None;
    }
    let n = n as usize;
    let mut acc = 0.0f32;
    for i in 0..n {
        let xi = *x.add(i * incx as usize);
        let yi = *y.add(i * incy as usize);
        acc += xi * yi;
    }
    Some(acc)
}

unsafe fn dot_real_f64(
    n: c_int,
    x: *const c_double,
    incx: c_int,
    y: *const c_double,
    incy: c_int,
) -> Option<c_double> {
    if n < 0 || incx <= 0 || incy <= 0 {
        return None;
    }
    let n = n as usize;
    let mut acc = 0.0f64;
    for i in 0..n {
        let xi = *x.add(i * incx as usize);
        let yi = *y.add(i * incy as usize);
        acc += xi * yi;
    }
    Some(acc)
}

unsafe fn complex_dot<T>(
    handle: cublasHandle_t,
    n: c_int,
    x: *const Complex<T>,
    incx: c_int,
    y: *const Complex<T>,
    incy: c_int,
    result: *mut Complex<T>,
    conjugate: bool,
) -> c_int
where
    T: Float,
{
    if handle.is_null() || x.is_null() || y.is_null() || result.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (*handle).pointer_mode != CUBLAS_POINTER_MODE_HOST {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if n < 0 || incx <= 0 || incy <= 0 {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let n = n as usize;
    let mut acc = Complex::<T>::from(T::zero());
    for i in 0..n {
        let mut xi = *x.add(i * incx as usize);
        let yi = *y.add(i * incy as usize);
        if conjugate {
            xi = xi.conj();
        }
        acc += xi * yi;
    }
    *result = acc;
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: c_int,
    b: *const c_float,
    ldb: c_int,
    beta: *const c_float,
    c: *mut c_float,
    ldc: c_int,
) -> c_int {
    if let Err(code) = validate_gemm(handle, m, n, k, alpha, beta, a, b, c, lda, ldb, ldc) {
        return code;
    }
    let alpha = *alpha;
    let beta = *beta;
    gemm_f32(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

#[no_mangle]
pub unsafe extern "C" fn cublasDgemm_v2(
    handle: cublasHandle_t,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: c_int,
    b: *const c_double,
    ldb: c_int,
    beta: *const c_double,
    c: *mut c_double,
    ldc: c_int,
) -> c_int {
    if let Err(code) = validate_gemm(handle, m, n, k, alpha, beta, a, b, c, lda, ldb, ldc) {
        return code;
    }
    let alpha = *alpha;
    let beta = *beta;
    gemm_f64(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

#[no_mangle]
pub unsafe extern "C" fn cublasSgemmStridedBatched(
    handle: cublasHandle_t,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: c_int,
    stride_a: i64,
    b: *const c_float,
    ldb: c_int,
    stride_b: i64,
    beta: *const c_float,
    c: *mut c_float,
    ldc: c_int,
    stride_c: i64,
    batch_count: c_int,
) -> c_int {
    if let Err(code) = validate_gemm(handle, m, n, k, alpha, beta, a, b, c, lda, ldb, ldc) {
        return code;
    }
    if batch_count < 0 {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let alpha = *alpha;
    let beta = *beta;
    for batch in 0..batch_count as isize {
        let a_ptr = a.offset(stride_a as isize * batch);
        let b_ptr = b.offset(stride_b as isize * batch);
        let c_ptr = c.offset(stride_c as isize * batch);
        let status = gemm_f32(transa, transb, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc);
        if status != CUBLAS_STATUS_SUCCESS {
            return status;
        }
    }
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasDgemmStridedBatched(
    handle: cublasHandle_t,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: c_int,
    stride_a: i64,
    b: *const c_double,
    ldb: c_int,
    stride_b: i64,
    beta: *const c_double,
    c: *mut c_double,
    ldc: c_int,
    stride_c: i64,
    batch_count: c_int,
) -> c_int {
    if let Err(code) = validate_gemm(handle, m, n, k, alpha, beta, a, b, c, lda, ldb, ldc) {
        return code;
    }
    if batch_count < 0 {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let alpha = *alpha;
    let beta = *beta;
    for batch in 0..batch_count as isize {
        let a_ptr = a.offset(stride_a as isize * batch);
        let b_ptr = b.offset(stride_b as isize * batch);
        let c_ptr = c.offset(stride_c as isize * batch);
        let status = gemm_f64(transa, transb, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc);
        if status != CUBLAS_STATUS_SUCCESS {
            return status;
        }
    }
    CUBLAS_STATUS_SUCCESS
}

unsafe fn validate_gemm<T>(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const T,
    beta: *const T,
    a: *const T,
    b: *const T,
    c: *mut T,
    lda: c_int,
    ldb: c_int,
    ldc: c_int,
) -> Result<(), c_int> {
    if handle.is_null() || alpha.is_null() || beta.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    if (*handle).pointer_mode != CUBLAS_POINTER_MODE_HOST {
        return Err(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    if m < 0 || n < 0 || k < 0 || lda < std::cmp::max(1, m) || ldb < std::cmp::max(1, k) || ldc < std::cmp::max(1, m) {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    Ok(())
}

unsafe fn gemm_f32(
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: c_float,
    a: *const c_float,
    lda: c_int,
    b: *const c_float,
    ldb: c_int,
    beta: c_float,
    c: *mut c_float,
    ldc: c_int,
) -> c_int {
    gemm_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

unsafe fn gemm_f64(
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: c_double,
    a: *const c_double,
    lda: c_int,
    b: *const c_double,
    ldb: c_int,
    beta: c_double,
    c: *mut c_double,
    ldc: c_int,
) -> c_int {
    gemm_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

unsafe fn gemm_impl<T>(
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: T,
    a: *const T,
    lda: c_int,
    b: *const T,
    ldb: c_int,
    beta: T,
    c: *mut T,
    ldc: c_int,
) -> c_int
where
    T: Copy
        + Zero
        + One
        + std::ops::AddAssign
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::MulAssign,
{
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;
    for col in 0..n {
        for row in 0..m {
            let mut acc = T::zero();
            for inner in 0..k {
                let a_val = read_col_major(a, lda, m, k, transa, row, inner);
                let b_val = read_col_major(b, ldb, k, n, transb, inner, col);
                acc += a_val * b_val;
            }
            let c_index = row + col * ldc;
            let current = *c.add(c_index);
            let mut value = alpha * acc;
            if beta != T::zero() {
                value += beta * current;
            }
            *c.add(c_index) = value;
        }
    }
    CUBLAS_STATUS_SUCCESS
}

unsafe fn read_col_major<T>(
    ptr: *const T,
    ld: usize,
    rows: usize,
    cols: usize,
    trans: c_int,
    row: usize,
    col: usize,
) -> T
where
    T: Copy,
{
    let (orig_row, orig_col) = if trans == CUBLAS_OP_N {
        (row, col)
    } else {
        (col, row)
    };
    debug_assert!(orig_row < rows);
    debug_assert!(orig_col < cols);
    *ptr.add(orig_row + orig_col * ld)
}

#[no_mangle]
pub unsafe extern "C" fn cublasGemmEx(
    handle: cublasHandle_t,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    a_type: cudaDataType,
    lda: c_int,
    b: *const c_void,
    b_type: cudaDataType,
    ldb: c_int,
    beta: *const c_void,
    c: *mut c_void,
    c_type: cudaDataType,
    ldc: c_int,
    compute_type: c_int,
    _algo: c_int,
) -> c_int {
    if handle.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (*handle).pointer_mode != CUBLAS_POINTER_MODE_HOST {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    match (a_type, b_type, c_type, compute_type) {
        (CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F) => {
            let status = validate_gemm(
                handle,
                m,
                n,
                k,
                alpha as *const c_float,
                beta as *const c_float,
                a as *const c_float,
                b as *const c_float,
                c as *mut c_float,
                lda,
                ldb,
                ldc,
            );
            if let Err(code) = status {
                return code;
            }
            gemm_f32(
                transa,
                transb,
                m,
                n,
                k,
                *(alpha as *const c_float),
                a as *const c_float,
                lda,
                b as *const c_float,
                ldb,
                *(beta as *const c_float),
                c as *mut c_float,
                ldc,
            )
        }
        (CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F) => {
            let status = validate_gemm(
                handle,
                m,
                n,
                k,
                alpha as *const c_double,
                beta as *const c_double,
                a as *const c_double,
                b as *const c_double,
                c as *mut c_double,
                lda,
                ldb,
                ldc,
            );
            if let Err(code) = status {
                return code;
            }
            gemm_f64(
                transa,
                transb,
                m,
                n,
                k,
                *(alpha as *const c_double),
                a as *const c_double,
                lda,
                b as *const c_double,
                ldb,
                *(beta as *const c_double),
                c as *mut c_double,
                ldc,
            )
        }
        _ => CUBLAS_STATUS_NOT_SUPPORTED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cublasGemmStridedBatchedEx(
    handle: cublasHandle_t,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    a_type: cudaDataType,
    lda: c_int,
    stride_a: i64,
    b: *const c_void,
    b_type: cudaDataType,
    ldb: c_int,
    stride_b: i64,
    beta: *const c_void,
    c: *mut c_void,
    c_type: cudaDataType,
    ldc: c_int,
    stride_c: i64,
    batch_count: c_int,
    compute_type: c_int,
    _algo: c_int,
) -> c_int {
    if handle.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (*handle).pointer_mode != CUBLAS_POINTER_MODE_HOST {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    match (a_type, b_type, c_type, compute_type) {
        (CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F) => {
            cublasSgemmStridedBatched(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha as *const c_float,
                a as *const c_float,
                lda,
                stride_a,
                b as *const c_float,
                ldb,
                stride_b,
                beta as *const c_float,
                c as *mut c_float,
                ldc,
                stride_c,
                batch_count,
            )
        }
        (CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F) => {
            cublasDgemmStridedBatched(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha as *const c_double,
                a as *const c_double,
                lda,
                stride_a,
                b as *const c_double,
                ldb,
                stride_b,
                beta as *const c_double,
                c as *mut c_double,
                ldc,
                stride_c,
                batch_count,
            )
        }
        _ => CUBLAS_STATUS_NOT_SUPPORTED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cublasSgemmEx(
    handle: cublasHandle_t,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    a_type: cudaDataType,
    lda: c_int,
    b: *const c_void,
    b_type: cudaDataType,
    ldb: c_int,
    beta: *const c_void,
    c: *mut c_void,
    c_type: cudaDataType,
    ldc: c_int,
) -> c_int {
    cublasGemmEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        b,
        b_type,
        ldb,
        beta,
        c,
        c_type,
        ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT,
    )
}

macro_rules! unsupported {
    ($(fn $name:ident($($arg:ident : $ty:ty),*) );* $(;)?) => {
        $(
            #[no_mangle]
            #[allow(unused_variables)]
            pub unsafe extern "C" fn $name($($arg: $ty),*) -> c_int {
                CUBLAS_STATUS_NOT_SUPPORTED
            }
        )*
    };
}

unsupported! {
    fn cublasCgelsBatched(handle: cublasHandle_t, trans: c_int, m: c_int, n: c_int, nrhs: c_int, A: *mut Complex<c_float>, lda: c_int, B: *mut Complex<c_float>, ldb: c_int, info: *mut c_int, batch: c_int);
    fn cublasCgemm_v2(handle: cublasHandle_t, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Complex<c_float>, a: *const Complex<c_float>, lda: c_int, b: *const Complex<c_float>, ldb: c_int, beta: *const Complex<c_float>, c: *mut Complex<c_float>, ldc: c_int);
    fn cublasCgemv_v2(handle: cublasHandle_t, trans: c_int, m: c_int, n: c_int, alpha: *const Complex<c_float>, a: *const Complex<c_float>, lda: c_int, x: *const Complex<c_float>, incx: c_int, beta: *const Complex<c_float>, y: *mut Complex<c_float>, incy: c_int);
    fn cublasCgemmStridedBatched(handle: cublasHandle_t, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Complex<c_float>, a: *const Complex<c_float>, lda: c_int, stride_a: i64, b: *const Complex<c_float>, ldb: c_int, stride_b: i64, beta: *const Complex<c_float>, c: *mut Complex<c_float>, ldc: c_int, stride_c: i64, batch: c_int);
    fn cublasCgeqrfBatched(handle: cublasHandle_t, m: c_int, n: c_int, A: *mut Complex<c_float>, lda: c_int, tau: *mut Complex<c_float>, info: *mut c_int, batch: c_int);
    fn cublasCgetrfBatched(handle: cublasHandle_t, n: c_int, A: *mut Complex<c_float>, lda: c_int, pivots: *mut c_int, info: *mut c_int, batch: c_int);
    fn cublasCgetrsBatched(handle: cublasHandle_t, trans: c_int, n: c_int, nrhs: c_int, A: *const Complex<c_float>, lda: c_int, pivots: *const c_int, B: *mut Complex<c_float>, ldb: c_int, info: *mut c_int, batch: c_int);
    fn cublasCtrsm_v2(handle: cublasHandle_t, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const Complex<c_float>, A: *const Complex<c_float>, lda: c_int, B: *mut Complex<c_float>, ldb: c_int);
    fn cublasCtrsmBatched(handle: cublasHandle_t, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const Complex<c_float>, A: *const Complex<c_float>, lda: c_int, stride_a: i64, B: *mut Complex<c_float>, ldb: c_int, stride_b: i64, batch: c_int);

    fn cublasDgelsBatched(handle: cublasHandle_t, trans: c_int, m: c_int, n: c_int, nrhs: c_int, A: *mut c_double, lda: c_int, B: *mut c_double, ldb: c_int, info: *mut c_int, batch: c_int);
    fn cublasDgemv_v2(handle: cublasHandle_t, trans: c_int, m: c_int, n: c_int, alpha: *const c_double, a: *const c_double, lda: c_int, x: *const c_double, incx: c_int, beta: *const c_double, y: *mut c_double, incy: c_int);
    fn cublasDgeqrfBatched(handle: cublasHandle_t, m: c_int, n: c_int, A: *mut c_double, lda: c_int, tau: *mut c_double, info: *mut c_int, batch: c_int);
    fn cublasDgetrfBatched(handle: cublasHandle_t, n: c_int, A: *mut c_double, lda: c_int, pivots: *mut c_int, info: *mut c_int, batch: c_int);
    fn cublasDgetrsBatched(handle: cublasHandle_t, trans: c_int, n: c_int, nrhs: c_int, A: *const c_double, lda: c_int, pivots: *const c_int, B: *mut c_double, ldb: c_int, info: *mut c_int, batch: c_int);
    fn cublasDtrsm_v2(handle: cublasHandle_t, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c Int, alpha: *const c_double, A: *const c_double, lda: c_int, B: *mut c_double, ldb: c_int);
    fn cublasDtrsmBatched(handle: cublasHandle_t, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const c_double, A: *const c_double, lda: c_int, stride_a: i64, B: *mut c_double, ldb: c_int, stride_b: i64, batch: c_int);

    fn cublasSgelsBatched(handle: cublasHandle_t, trans: c_int, m: c_int, n: c_int, nrhs: c_int, A: *mut c_float, lda: c_int, B: *mut c_float, ldb: c_int, info: *mut c_int, batch: c_int);
    fn cublasSgemv_v2(handle: cublasHandle_t, trans: c_int, m: c_int, n: c_int, alpha: *const c_float, a: *const c_float, lda: c_int, x: *const c_float, incx: c_int, beta: *const c_float, y: *mut c_float, incy: c_int);
    fn cublasSgeqrfBatched(handle: cublasHandle_t, m: c_int, n: c_int, A: *mut c_float, lda: c_int, tau: *mut c_float, info: *mut c_int, batch: c_int);
    fn cublasSgetrfBatched(handle: cublasHandle_t, n: c_int, A: *mut c_float, lda: c_int, pivots: *mut c_int, info: *mut c_int, batch: c_int);
    fn cublasSgetrsBatched(handle: cublasHandle_t, trans: c_int, n: c_int, nrhs: c_int, A: *const c_float, lda: c_int, pivots: *const c_int, B: *mut c_float, ldb: c_int, info: *mut c_int, batch: c_int);
    fn cublasStrsm_v2(handle: cublasHandle_t, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const c_float, A: *const c_float, lda: c_int, B: *mut c_float, ldb: c_int);
    fn cublasStrsmBatched(handle: cublasHandle_t, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const c_float, A: *const c_float, lda: c_int, stride_a: i64, B: *mut c_float, ldb: c_int, stride_b: i64, batch: c_int);

    fn cublasZgelsBatched(handle: cublasHandle_t, trans: c_int, m: c_int, n: c_int, nrhs: c_int, A: *mut Complex<c_double>, lda: c_int, B: *mut Complex<c_double>, ldb: c_int, info: *mut c_int, batch: c_int);
    fn cublasZgemm_v2(handle: cublasHandle_t, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Complex<c_double>, a: *const Complex<c_double>, lda: c_int, b: *const Complex<c_double>, ldb: c_int, beta: *const Complex<c_double>, c: *mut Complex<c_double>, ldc: c_int);
    fn cublasZgemv_v2(handle: cublasHandle_t, trans: c_int, m: c_int, n: c_int, alpha: *const Complex<c_double>, a: *const Complex<c_double>, lda: c_int, x: *const Complex<c_double>, incx: c_int, beta: *const Complex<c_double>, y: *mut Complex<c_double>, incy: c_int);
    fn cublasZgemmStridedBatched(handle: cublasHandle_t, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Complex<c_double>, a: *const Complex<c_double>, lda: c_int, stride_a: i64, b: *const Complex<c_double>, ldb: c_int, stride_b: i64, beta: *const Complex<c_double>, c: *mut Complex<c_double>, ldc: c_int, stride_c: i64, batch: c_int);
    fn cublasZgeqrfBatched(handle: cublasHandle_t, m: c_int, n: c_int, A: *mut Complex<c_double>, lda: c_int, tau: *mut Complex<c_double>, info: *mut c_int, batch: c_int);
    fn cublasZgetrfBatched(handle: cublasHandle_t, n: c_int, A: *mut Complex<c_double>, lda: c_int, pivots: *mut c_int, info: *mut c_int, batch: c_int);
    fn cublasZgetrsBatched(handle: cublasHandle_t, trans: c_int, n: c_int, nrhs: c_int, A: *const Complex<c_double>, lda: c_int, pivots: *const c_int, B: *mut Complex<c_double>, ldb: c_int, info: *mut c_int, batch: c_int);
    fn cublasZtrsm_v2(handle: cublasHandle_t, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const Complex<c_double>, A: *const Complex<c_double>, lda: c_int, B: *mut Complex<c_double>, ldb: c_int);
    fn cublasZtrsmBatched(handle: cublasHandle_t, side: c_int, uplo: c_int, transa: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const Complex<c_double>, A: *const Complex<c_double>, lda: c_int, stride_a: i64, B: *mut Complex<c_double>, ldb: c_int, stride_b: i64, batch: c_int);

    fn cublasDotEx(handle: cublasHandle_t, n: c_int, x: *const c_void, x_type: cudaDataType, incx: c_int, y: *const c_void, y_type: cudaDataType, incy: c_int, result: *mut c_void, result_type: cudaDataType, compute_type: cudaDataType);
}
