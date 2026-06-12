use std::ffi::CString;
use std::os::raw::{c_char, c_void};

extern "C" {
    fn hetgpu_webgpu_init() -> i32;
    fn hetgpu_webgpu_device_count() -> i32;
    fn hetgpu_webgpu_module_load(image: *const c_void, image_len: usize) -> u64;
    fn hetgpu_webgpu_get_function(module_id: u64, name: *const c_char) -> u64;
    fn hetgpu_webgpu_launch_kernel(
        kernel_id: u64,
        name: *const c_char,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_mem: u32,
        kernel_params: *mut *mut c_void,
    ) -> i32;
}

pub(crate) fn init() -> Result<(), i32> {
    let rc = unsafe { hetgpu_webgpu_init() };
    if rc == 0 {
        Ok(())
    } else {
        Err(rc)
    }
}

pub(crate) fn device_count() -> usize {
    unsafe { hetgpu_webgpu_device_count().max(1) as usize }
}

pub(crate) fn load_module(image: &[u8]) -> Result<u64, i32> {
    let id = unsafe { hetgpu_webgpu_module_load(image.as_ptr() as *const c_void, image.len()) };
    if id != 0 {
        Ok(id)
    } else {
        Err(1)
    }
}

pub(crate) fn get_function(module_id: u64, name: &str) -> Result<u64, i32> {
    let name = CString::new(name).map_err(|_| 1)?;
    let id = unsafe { hetgpu_webgpu_get_function(module_id, name.as_ptr()) };
    if id != 0 {
        Ok(id)
    } else {
        Err(1)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_kernel(
    kernel_id: u64,
    name: &str,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_mem: u32,
    kernel_params: *mut *mut c_void,
) -> Result<(), i32> {
    let name = CString::new(name).map_err(|_| 1)?;
    let rc = unsafe {
        hetgpu_webgpu_launch_kernel(
            kernel_id,
            name.as_ptr(),
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_mem,
            kernel_params,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(rc)
    }
}
