use std::collections::HashMap;
use std::ffi::{c_char, c_uint, c_void, CStr, CString};
use std::ptr;
use std::sync::{Mutex, OnceLock};

const CUDA_SUCCESS: i32 = 0;
const CUDA_ERROR_INVALID_VALUE: i32 = 1;
const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;
const CUDA_ERROR_INVALID_IMAGE: i32 = 200;
const CUDA_ERROR_NOT_FOUND: i32 = 500;
const CUDA_ERROR_LAUNCH_FAILED: i32 = 719;
const CUDA_ERROR_NOT_SUPPORTED: i32 = 801;

const HETGPU_METAL_BUFFER_COPY_IN: c_uint = 1;
const HETGPU_METAL_BUFFER_COPY_OUT: c_uint = 2;

#[repr(C)]
struct HetGpuMetalBufferBinding {
    host_ptr: *mut c_void,
    size: usize,
    flags: c_uint,
}

extern "C" {
    fn hetgpu_apple_metal_compile_msl(
        source: *const c_char,
        label: *const c_char,
        out_module: *mut *mut c_void,
        out_log: *mut *mut c_char,
    ) -> i32;

    fn hetgpu_apple_metal_get_function(
        module: *mut c_void,
        name: *const c_char,
        out_function: *mut *mut c_void,
        out_log: *mut *mut c_char,
    ) -> i32;

    fn hetgpu_apple_metal_launch_raw(
        function: *mut c_void,
        buffers: *const HetGpuMetalBufferBinding,
        buffer_count: usize,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        out_log: *mut *mut c_char,
    ) -> i32;

    fn hetgpu_apple_metal_release_module(module: *mut c_void) -> i32;
    fn hetgpu_apple_metal_release_function(function: *mut c_void) -> i32;
    fn hetgpu_apple_metal_free_string(value: *mut c_char);
}

struct PtxModule {
    metal_module: *mut c_void,
    kernels: HashMap<String, comgr::AppleKernelMetadata>,
    _msl_source: String,
}

struct PtxFunction {
    metal_function: *mut c_void,
    metadata: comgr::AppleKernelMetadata,
}

unsafe impl Send for PtxModule {}
unsafe impl Send for PtxFunction {}

static ALLOCATIONS: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();

fn allocations() -> &'static Mutex<HashMap<usize, usize>> {
    ALLOCATIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn apple_take_log(log: *mut c_char) -> Option<String> {
    if log.is_null() {
        return None;
    }
    let message = unsafe { CStr::from_ptr(log) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        hetgpu_apple_metal_free_string(log);
    }
    Some(message)
}

fn cuda_launch_dim(grid: c_uint, block: c_uint) -> Result<c_uint, i32> {
    if grid == 0 || block == 0 {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    let total = (grid as u64)
        .checked_mul(block as u64)
        .ok_or(CUDA_ERROR_LAUNCH_FAILED)?;
    c_uint::try_from(total).map_err(|_| CUDA_ERROR_LAUNCH_FAILED)
}

fn lookup_allocation(ptr_value: usize) -> Option<usize> {
    let map = allocations().lock().ok()?;
    for (&base, &size) in map.iter() {
        let end = base.saturating_add(size);
        if ptr_value >= base && ptr_value < end {
            return Some(end - ptr_value);
        }
    }
    None
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_apple_ptx_register_allocation(
    ptr_value: *mut c_void,
    size: usize,
) -> i32 {
    if ptr_value.is_null() || size == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let Ok(mut map) = allocations().lock() else {
        return CUDA_ERROR_OUT_OF_MEMORY;
    };
    map.insert(ptr_value as usize, size);
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_apple_ptx_unregister_allocation(ptr_value: *mut c_void) -> i32 {
    if ptr_value.is_null() {
        return CUDA_SUCCESS;
    }
    let Ok(mut map) = allocations().lock() else {
        return CUDA_ERROR_OUT_OF_MEMORY;
    };
    map.remove(&(ptr_value as usize));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_apple_ptx_module_load_data(
    module: *mut *mut c_void,
    image: *const c_void,
) -> i32 {
    if module.is_null() || image.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let text = match CStr::from_ptr(image.cast::<c_char>()).to_str() {
        Ok(text) => text,
        Err(_) => return CUDA_ERROR_INVALID_VALUE,
    };
    let compiled = match comgr::compile_ptx_to_msl_module(text.as_bytes()) {
        Ok(compiled) => compiled,
        Err(err) => {
            eprintln!("[hetgpu_apple_ptx] PTX-to-MSL failed: {}", err.diagnostics);
            return CUDA_ERROR_INVALID_IMAGE;
        }
    };

    let source = match CString::new(compiled.msl.as_str()) {
        Ok(source) => source,
        Err(_) => return CUDA_ERROR_INVALID_IMAGE,
    };
    let label = CString::new("hetgpu-apple-ptx").expect("static label");
    let mut metal_module = ptr::null_mut();
    let mut log = ptr::null_mut();
    let status = hetgpu_apple_metal_compile_msl(
        source.as_ptr(),
        label.as_ptr(),
        &mut metal_module,
        &mut log,
    );
    if status != 0 || metal_module.is_null() {
        if let Some(message) = apple_take_log(log) {
            eprintln!("[hetgpu_apple_ptx] Metal compile failed: {message}");
        }
        return CUDA_ERROR_INVALID_IMAGE;
    }

    let mut kernels = HashMap::with_capacity(compiled.kernels.len() * 2);
    for kernel in &compiled.kernels {
        kernels.insert(kernel.name.clone(), kernel.clone());
        kernels.insert(kernel.msl_name.clone(), kernel.clone());
    }

    *module = Box::into_raw(Box::new(PtxModule {
        metal_module,
        kernels,
        _msl_source: compiled.msl,
    }))
    .cast::<c_void>();
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_apple_ptx_module_unload(module: *mut c_void) -> i32 {
    if module.is_null() {
        return CUDA_SUCCESS;
    }
    let mut module = Box::from_raw(module.cast::<PtxModule>());
    if !module.metal_module.is_null() {
        let _ = hetgpu_apple_metal_release_module(module.metal_module);
        module.metal_module = ptr::null_mut();
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_apple_ptx_module_get_function(
    function: *mut *mut c_void,
    module: *mut c_void,
    name: *const c_char,
) -> i32 {
    if function.is_null() || module.is_null() || name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let module = &*(module.cast::<PtxModule>());
    let name = match CStr::from_ptr(name).to_str() {
        Ok(name) => name,
        Err(_) => return CUDA_ERROR_INVALID_VALUE,
    };
    let Some(metadata) = module.kernels.get(name).cloned() else {
        return CUDA_ERROR_NOT_FOUND;
    };
    let metal_name = match CString::new(metadata.msl_name.as_str()) {
        Ok(name) => name,
        Err(_) => return CUDA_ERROR_INVALID_VALUE,
    };

    let mut metal_function = ptr::null_mut();
    let mut log = ptr::null_mut();
    let status = hetgpu_apple_metal_get_function(
        module.metal_module,
        metal_name.as_ptr(),
        &mut metal_function,
        &mut log,
    );
    if status != 0 || metal_function.is_null() {
        if let Some(message) = apple_take_log(log) {
            eprintln!(
                "[hetgpu_apple_ptx] Metal function lookup failed for {}: {message}",
                metadata.msl_name
            );
        }
        return CUDA_ERROR_NOT_FOUND;
    }

    *function = Box::into_raw(Box::new(PtxFunction {
        metal_function,
        metadata,
    }))
    .cast::<c_void>();
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_apple_ptx_function_release(function: *mut c_void) -> i32 {
    if function.is_null() {
        return CUDA_SUCCESS;
    }
    let function = Box::from_raw(function.cast::<PtxFunction>());
    if !function.metal_function.is_null() {
        let _ = hetgpu_apple_metal_release_function(function.metal_function);
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_apple_ptx_launch_kernel(
    function: *mut c_void,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    _shared_mem_bytes: c_uint,
    _stream: *mut c_void,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> i32 {
    if function.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if !extra.is_null() {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    let function = &*(function.cast::<PtxFunction>());
    if !function.metadata.params.is_empty() && kernel_params.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let mut scalar_storage: Vec<Vec<u8>> = Vec::new();
    let mut bindings = Vec::with_capacity(function.metadata.params.len());

    for (idx, param) in function.metadata.params.iter().enumerate() {
        let param_slot = *kernel_params.add(idx);
        if param_slot.is_null() {
            return CUDA_ERROR_INVALID_VALUE;
        }

        if param.is_pointer {
            let host_ptr = *(param_slot as *const *mut c_void);
            if host_ptr.is_null() {
                return CUDA_ERROR_INVALID_VALUE;
            }
            let Some(size) = lookup_allocation(host_ptr as usize) else {
                eprintln!(
                    "[hetgpu_apple_ptx] pointer parameter {} ({}) is not a registered allocation",
                    idx, param.ptx_name
                );
                return CUDA_ERROR_INVALID_VALUE;
            };
            bindings.push(HetGpuMetalBufferBinding {
                host_ptr,
                size: size.max(1),
                flags: HETGPU_METAL_BUFFER_COPY_IN | HETGPU_METAL_BUFFER_COPY_OUT,
            });
        } else {
            let size = param.size.max(1);
            let mut bytes = vec![0u8; size];
            ptr::copy_nonoverlapping(param_slot.cast::<u8>(), bytes.as_mut_ptr(), size);
            let host_ptr = bytes.as_mut_ptr().cast::<c_void>();
            scalar_storage.push(bytes);
            bindings.push(HetGpuMetalBufferBinding {
                host_ptr,
                size,
                flags: HETGPU_METAL_BUFFER_COPY_IN,
            });
        }
    }

    let dispatch_x = match cuda_launch_dim(grid_dim_x, block_dim_x) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let dispatch_y = match cuda_launch_dim(grid_dim_y, block_dim_y) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let dispatch_z = match cuda_launch_dim(grid_dim_z, block_dim_z) {
        Ok(value) => value,
        Err(error) => return error,
    };

    let mut log = ptr::null_mut();
    let status = hetgpu_apple_metal_launch_raw(
        function.metal_function,
        bindings.as_ptr(),
        bindings.len(),
        dispatch_x,
        dispatch_y,
        dispatch_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        &mut log,
    );
    drop(scalar_storage);
    if status != 0 {
        if let Some(message) = apple_take_log(log) {
            eprintln!(
                "[hetgpu_apple_ptx] Metal launch failed for {}: {message}",
                function.metadata.name
            );
        }
        return CUDA_ERROR_LAUNCH_FAILED;
    }

    CUDA_SUCCESS
}
