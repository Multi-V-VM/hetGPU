#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use libc::{c_int, c_uint, c_void, size_t};
use std::ptr;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

const CUBLAS_STATUS_SUCCESS: c_int = 0;
const CUBLAS_STATUS_INVALID_VALUE: c_int = 7;
const CUBLAS_STATUS_NOT_SUPPORTED: c_int = 15;

const CUDA_R_32F: c_int = 0;

const CUBLASLT_MATRIX_LAYOUT_ORDER: c_int = 1;
const CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: c_int = 5;
const CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: c_int = 6;

const CUBLASLT_MATMUL_DESC_COMPUTE_TYPE: c_int = 0;
const CUBLASLT_MATMUL_DESC_SCALE_TYPE: c_int = 1;
const CUBLASLT_MATMUL_DESC_POINTER_MODE: c_int = 2;
const CUBLASLT_MATMUL_DESC_TRANSA: c_int = 3;
const CUBLASLT_MATMUL_DESC_TRANSB: c_int = 4;

const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: c_int = 1;

const CUBLASLT_ORDER_COL: c_int = 0;
const CUBLASLT_ORDER_ROW: c_int = 1;

const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
const CUBLAS_OP_C: c_int = 2;

type cublasLtHandle_t = *mut LtHandle;
type cublasLtMatrixLayout_t = *mut MatrixLayout;
type cublasLtMatmulDesc_t = *mut MatmulDesc;
type cublasLtMatmulPreference_t = *mut MatmulPreference;
type cublasLtMatmulAlgo_t = MatmulAlgo;
type cublasComputeType_t = c_int;
type cudaDataType_t = c_int;

macro_rules! dyn_array {
    (struct $name:ident, $ty:ty, $len:expr) => {
        #[repr(C)]
        struct $name {
            data: [$ty; $len],
        }
    };
    (struct $name:ident, $ty:ty, $len:expr, $( $field:ident : $fty:ty ),+ ) => {
        #[repr(C)]
        struct $name {
            data: [$ty; $len],
            $( $field : $fty ),+
        }
    };
}

dyn_array! { struct MatmulAlgo, u64, 8 }
dyn_array! { struct cublasLtMatmulHeuristicResult_t, u64, 8, workspace_size: size_t, state: c_int, waves_count: f32, reserved: [c_int; 4] }

#[repr(C)]
struct LtHandle {
    id: u64,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum MatrixOrder {
    ColumnMajor,
    RowMajor,
}

#[repr(C)]
struct MatrixLayout {
    data_type: c_int,
    rows: u64,
    cols: u64,
    ld: i64,
    order: MatrixOrder,
    batch_count: i64,
    batch_stride: i64,
}

#[repr(C)]
struct MatmulDesc {
    compute_type: c_int,
    scale_type: c_int,
    pointer_mode: c_int,
    transa: c_int,
    transb: c_int,
}

#[repr(C)]
struct MatmulPreference {
    max_workspace_bytes: usize,
}

static CPU_DISABLE_MASK: AtomicU32 = AtomicU32::new(0);
static HEURISTICS_CACHE_CAPACITY: AtomicUsize = AtomicUsize::new(0);
static HANDLE_COUNTER: AtomicUsize = AtomicUsize::new(1);

#[no_mangle]
pub unsafe extern "C" fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> c_int {
    if handle.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let id = HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed) as u64;
    let boxed = Box::new(LtHandle { id });
    *handle = Box::into_raw(boxed);
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtDestroy(handle: cublasLtHandle_t) -> c_int {
    if handle.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    drop(Box::from_raw(handle));
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtGetStatusName(status: c_int) -> *const i8 {
    status_name(status)
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtGetStatusString(status: c_int) -> *const i8 {
    status_name(status)
}

fn status_name(status: c_int) -> *const i8 {
    match status {
        CUBLAS_STATUS_SUCCESS => c_str!("CUBLAS_STATUS_SUCCESS"),
        CUBLAS_STATUS_INVALID_VALUE => c_str!("CUBLAS_STATUS_INVALID_VALUE"),
        CUBLAS_STATUS_NOT_SUPPORTED => c_str!("CUBLAS_STATUS_NOT_SUPPORTED"),
        _ => c_str!("CUBLAS_STATUS_OTHER"),
    }
}

macro_rules! c_str {
    ($lit:expr) => {{
        concat!($lit, "\0").as_ptr() as *const i8
    }};
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtGetVersion() -> size_t {
    12000
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtGetCudartVersion() -> size_t {
    12000
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtGetProperty(property: c_int, value: *mut c_int) -> c_int {
    if value.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    match property {
        0 => *value = 12, // MAJOR_VERSION
        1 => *value = 0,  // MINOR_VERSION
        2 => *value = 0,  // PATCH_LEVEL
        _ => return CUBLAS_STATUS_NOT_SUPPORTED,
    }
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtDisableCpuInstructionsSetMask(mask: c_uint) -> c_uint {
    CPU_DISABLE_MASK.swap(mask, Ordering::SeqCst)
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtHeuristicsCacheGetCapacity(capacity: *mut size_t) -> c_int {
    if capacity.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *capacity = HEURISTICS_CACHE_CAPACITY.load(Ordering::Relaxed);
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtHeuristicsCacheSetCapacity(capacity: size_t) -> c_int {
    HEURISTICS_CACHE_CAPACITY.store(capacity, Ordering::Relaxed);
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatrixLayoutCreate(
    layout: *mut cublasLtMatrixLayout_t,
    data_type: cudaDataType_t,
    rows: u64,
    cols: u64,
    ld: i64,
) -> c_int {
    if layout.is_null() || rows == 0 || cols == 0 || ld <= 0 {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let inner = MatrixLayout {
        data_type,
        rows,
        cols,
        ld,
        order: MatrixOrder::ColumnMajor,
        batch_count: 1,
        batch_stride: 0,
    };
    *layout = Box::into_raw(Box::new(inner));
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatrixLayoutDestroy(layout: cublasLtMatrixLayout_t) -> c_int {
    if layout.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    drop(Box::from_raw(layout));
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatrixLayoutSetAttribute(
    layout: cublasLtMatrixLayout_t,
    attr: c_int,
    buf: *const c_void,
    size: size_t,
) -> c_int {
    if layout.is_null() || buf.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let layout = &mut *layout;
    match attr {
        CUBLASLT_MATRIX_LAYOUT_ORDER => {
            if size != std::mem::size_of::<c_int>() {
                return CUBLAS_STATUS_INVALID_VALUE;
            }
            let value = *(buf as *const c_int);
            layout.order = match value {
                CUBLASLT_ORDER_COL => MatrixOrder::ColumnMajor,
                CUBLASLT_ORDER_ROW => MatrixOrder::RowMajor,
                _ => return CUBLAS_STATUS_NOT_SUPPORTED,
            };
        }
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT => {
            if size != std::mem::size_of::<c_int>() {
                return CUBLAS_STATUS_INVALID_VALUE;
            }
            layout.batch_count = *(buf as *const c_int) as i64;
        }
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET => {
            if size != std::mem::size_of::<i64>() {
                return CUBLAS_STATUS_INVALID_VALUE;
            }
            layout.batch_stride = *(buf as *const i64);
        }
        _ => return CUBLAS_STATUS_NOT_SUPPORTED,
    }
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatrixLayoutGetAttribute(
    layout: cublasLtMatrixLayout_t,
    attr: c_int,
    buf: *mut c_void,
    size: size_t,
    size_written: *mut size_t,
) -> c_int {
    if layout.is_null() || buf.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let layout = &*layout;
    macro_rules! write_attr {
        ($ty:ty, $value:expr) => {{
            if size < std::mem::size_of::<$ty>() {
                return CUBLAS_STATUS_INVALID_VALUE;
            }
            *(buf as *mut $ty) = $value;
            if !size_written.is_null() {
                *size_written = std::mem::size_of::<$ty>();
            }
            return CUBLAS_STATUS_SUCCESS;
        }};
    }
    match attr {
        CUBLASLT_MATRIX_LAYOUT_ORDER => write_attr!(c_int, match layout.order {
            MatrixOrder::ColumnMajor => CUBLASLT_ORDER_COL,
            MatrixOrder::RowMajor => CUBLASLT_ORDER_ROW,
        }),
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT => write_attr!(c_int, layout.batch_count as c_int),
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET => write_attr!(i64, layout.batch_stride),
        _ => CUBLAS_STATUS_NOT_SUPPORTED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatmulDescCreate(
    desc: *mut cublasLtMatmulDesc_t,
    compute_type: cublasComputeType_t,
    scale_type: cudaDataType_t,
) -> c_int {
    if desc.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let inner = MatmulDesc {
        compute_type,
        scale_type,
        pointer_mode: 0,
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
    };
    *desc = Box::into_raw(Box::new(inner));
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatmulDescDestroy(desc: cublasLtMatmulDesc_t) -> c_int {
    if desc.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    drop(Box::from_raw(desc));
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatmulDescSetAttribute(
    desc: cublasLtMatmulDesc_t,
    attr: c_int,
    buf: *const c_void,
    size: size_t,
) -> c_int {
    if desc.is_null() || buf.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let desc = &mut *desc;
    match attr {
        CUBLASLT_MATMUL_DESC_COMPUTE_TYPE
        | CUBLASLT_MATMUL_DESC_SCALE_TYPE
        | CUBLASLT_MATMUL_DESC_POINTER_MODE
        | CUBLASLT_MATMUL_DESC_TRANSA
        | CUBLASLT_MATMUL_DESC_TRANSB => {
            if size != std::mem::size_of::<c_int>() {
                return CUBLAS_STATUS_INVALID_VALUE;
            }
            let value = *(buf as *const c_int);
            match attr {
                CUBLASLT_MATMUL_DESC_COMPUTE_TYPE => desc.compute_type = value,
                CUBLASLT_MATMUL_DESC_SCALE_TYPE => desc.scale_type = value,
                CUBLASLT_MATMUL_DESC_POINTER_MODE => desc.pointer_mode = value,
                CUBLASLT_MATMUL_DESC_TRANSA => desc.transa = value,
                CUBLASLT_MATMUL_DESC_TRANSB => desc.transb = value,
                _ => unreachable!(),
            }
        }
        _ => return CUBLAS_STATUS_NOT_SUPPORTED,
    }
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatmulDescGetAttribute(
    desc: cublasLtMatmulDesc_t,
    attr: c_int,
    buf: *mut c_void,
    size: size_t,
    size_written: *mut size_t,
) -> c_int {
    if desc.is_null() || buf.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let desc = &*desc;
    macro_rules! write_attr {
        ($ty:ty, $value:expr) => {{
            if size < std::mem::size_of::<$ty>() {
                return CUBLAS_STATUS_INVALID_VALUE;
            }
            *(buf as *mut $ty) = $value;
            if !size_written.is_null() {
                *size_written = std::mem::size_of::<$ty>();
            }
            return CUBLAS_STATUS_SUCCESS;
        }};
    }
    match attr {
        CUBLASLT_MATMUL_DESC_COMPUTE_TYPE => write_attr!(c_int, desc.compute_type),
        CUBLASLT_MATMUL_DESC_SCALE_TYPE => write_attr!(c_int, desc.scale_type),
        CUBLASLT_MATMUL_DESC_POINTER_MODE => write_attr!(c_int, desc.pointer_mode),
        CUBLASLT_MATMUL_DESC_TRANSA => write_attr!(c_int, desc.transa),
        CUBLASLT_MATMUL_DESC_TRANSB => write_attr!(c_int, desc.transb),
        _ => CUBLAS_STATUS_NOT_SUPPORTED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatmulPreferenceCreate(
    pref: *mut cublasLtMatmulPreference_t,
) -> c_int {
    if pref.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *pref = Box::into_raw(Box::new(MatmulPreference {
        max_workspace_bytes: 0,
    }));
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> c_int {
    if pref.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    drop(Box::from_raw(pref));
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatmulPreferenceSetAttribute(
    pref: cublasLtMatmulPreference_t,
    attr: c_int,
    buf: *const c_void,
    size: size_t,
) -> c_int {
    if pref.is_null() || buf.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let pref = &mut *pref;
    match attr {
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES => {
            if size != std::mem::size_of::<size_t>() {
                return CUBLAS_STATUS_INVALID_VALUE;
            }
            pref.max_workspace_bytes = *(buf as *const size_t);
        }
        _ => return CUBLAS_STATUS_NOT_SUPPORTED,
    }
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatmulPreferenceGetAttribute(
    pref: cublasLtMatmulPreference_t,
    attr: c_int,
    buf: *mut c_void,
    size: size_t,
    size_written: *mut size_t,
) -> c_int {
    if pref.is_null() || buf.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let pref = &*pref;
    match attr {
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES => {
            if size < std::mem::size_of::<size_t>() {
                return CUBLAS_STATUS_INVALID_VALUE;
            }
            *(buf as *mut size_t) = pref.max_workspace_bytes;
            if !size_written.is_null() {
                *size_written = std::mem::size_of::<size_t>();
            }
            CUBLAS_STATUS_SUCCESS
        }
        _ => CUBLAS_STATUS_NOT_SUPPORTED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatmulAlgoGetHeuristic(
    _handle: cublasLtHandle_t,
    _operation_desc: cublasLtMatmulDesc_t,
    _adesc: cublasLtMatrixLayout_t,
    _bdesc: cublasLtMatrixLayout_t,
    _cdesc: cublasLtMatrixLayout_t,
    _ddesc: cublasLtMatrixLayout_t,
    _preference: cublasLtMatmulPreference_t,
    requested_algo_count: c_int,
    heuristic_results: *mut cublasLtMatmulHeuristicResult_t,
    return_algo_count: *mut c_int,
) -> c_int {
    if requested_algo_count <= 0 {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if !return_algo_count.is_null() {
        *return_algo_count = 0;
    }
    if !heuristic_results.is_null() {
        (*heuristic_results).state = CUBLAS_STATUS_NOT_SUPPORTED;
        (*heuristic_results).workspace_size = 0;
        (*heuristic_results).waves_count = 0.0;
        (*heuristic_results).reserved = [0; 4];
    }
    CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cublasLtMatmul(
    _handle: cublasLtHandle_t,
    op_desc: cublasLtMatmulDesc_t,
    alpha: *const c_void,
    a: *const c_void,
    a_desc: cublasLtMatrixLayout_t,
    b: *const c_void,
    b_desc: cublasLtMatrixLayout_t,
    beta: *const c_void,
    c_in: *const c_void,
    c_desc: cublasLtMatrixLayout_t,
    d_out: *mut c_void,
    d_desc: cublasLtMatrixLayout_t,
    _algo: *const cublasLtMatmulAlgo_t,
    _workspace: *mut c_void,
    _workspace_size: size_t,
    _stream: *mut c_void,
) -> c_int {
    if op_desc.is_null() || a.is_null() || b.is_null() || a_desc.is_null() || b_desc.is_null() || c_desc.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let desc = &*op_desc;
    if desc.pointer_mode != 0 {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    let a_layout = &*a_desc;
    let b_layout = &*b_desc;
    let c_layout = &*c_desc;
    let output_layout_ref;
    let output_layout = if !d_desc.is_null() {
        output_layout_ref = &*d_desc;
        output_layout_ref
    } else {
        c_layout
    };

    if a_layout.data_type != CUDA_R_32F
        || b_layout.data_type != CUDA_R_32F
        || c_layout.data_type != CUDA_R_32F
        || output_layout.data_type != CUDA_R_32F
    {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    if a_layout.batch_count > 1
        || b_layout.batch_count > 1
        || c_layout.batch_count > 1
        || output_layout.batch_count > 1
    {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    let alpha = if alpha.is_null() {
        1.0f32
    } else {
        *(alpha as *const f32)
    };
    let beta = if beta.is_null() {
        0.0f32
    } else {
        *(beta as *const f32)
    };

    let a_ptr = a as *const f32;
    let b_ptr = b as *const f32;
    let c_ptr = if c_in.is_null() {
        ptr::null()
    } else {
        c_in as *const f32
    };
    let d_ptr = if !d_out.is_null() {
        d_out as *mut f32
    } else if !c_in.is_null() {
        c_in as *mut f32
    } else {
        return CUBLAS_STATUS_INVALID_VALUE;
    };

    let (m, k_a) = dims_after_transpose(a_layout, desc.transa);
    let (k_b, n) = dims_after_transpose(b_layout, desc.transb);
    let (m_c, n_c) = (c_layout.rows as usize, c_layout.cols as usize);
    let (m_d, n_d) = (output_layout.rows as usize, output_layout.cols as usize);

    if m != m_c || m != m_d || n != n_c || n != n_d || k_a != k_b {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    if let Err(code) = run_matmul_f32(
        alpha,
        a_ptr,
        a_layout,
        desc.transa,
        b_ptr,
        b_layout,
        desc.transb,
        beta,
        c_ptr,
        c_layout,
        d_ptr,
        output_layout,
        m,
        n,
        k_a,
    ) {
        return code;
    }

    CUBLAS_STATUS_SUCCESS
}

fn dims_after_transpose(layout: &MatrixLayout, trans: c_int) -> (usize, usize) {
    if trans == CUBLAS_OP_N {
        (layout.rows as usize, layout.cols as usize)
    } else {
        (layout.cols as usize, layout.rows as usize)
    }
}

unsafe fn run_matmul_f32(
    alpha: f32,
    a_ptr: *const f32,
    a_layout: &MatrixLayout,
    transa: c_int,
    b_ptr: *const f32,
    b_layout: &MatrixLayout,
    transb: c_int,
    beta: f32,
    c_ptr: *const f32,
    c_layout: &MatrixLayout,
    d_ptr: *mut f32,
    d_layout: &MatrixLayout,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), c_int> {
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0f32;
            for inner in 0..k {
                let a = read_elem_f32(a_ptr, a_layout, transa, row, inner)?;
                let b = read_elem_f32(b_ptr, b_layout, transb, inner, col)?;
                acc += a * b;
            }
            let c_val = if beta != 0.0 && !c_ptr.is_null() {
                read_elem_f32(c_ptr, c_layout, CUBLAS_OP_N, row, col)?
            } else {
                0.0
            };
            let value = alpha * acc + beta * c_val;
            write_elem_f32(d_ptr, d_layout, row, col, value)?;
        }
    }
    Ok(())
}

unsafe fn read_elem_f32(
    ptr: *const f32,
    layout: &MatrixLayout,
    trans: c_int,
    row: usize,
    col: usize,
) -> Result<f32, c_int> {
    if ptr.is_null() {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let (orig_row, orig_col) = if trans == CUBLAS_OP_N {
        (row, col)
    } else {
        (col, row)
    };
    if orig_row >= layout.rows as usize || orig_col >= layout.cols as usize {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let idx = match layout.order {
        MatrixOrder::ColumnMajor => orig_row + orig_col * (layout.ld as usize),
        MatrixOrder::RowMajor => orig_row * (layout.ld as usize) + orig_col,
    };
    Ok(*ptr.add(idx))
}

unsafe fn write_elem_f32(
    ptr: *mut f32,
    layout: &MatrixLayout,
    row: usize,
    col: usize,
    value: f32,
) -> Result<(), c_int> {
    if ptr.is_null() {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    if row >= layout.rows as usize || col >= layout.cols as usize {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let idx = match layout.order {
        MatrixOrder::ColumnMajor => row + col * (layout.ld as usize),
        MatrixOrder::RowMajor => row * (layout.ld as usize) + col,
    };
    *ptr.add(idx) = value;
    Ok(())
}
