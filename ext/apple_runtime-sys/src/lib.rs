#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::ffi::{c_char, c_int, c_uint, c_void};

pub const HETGPU_CUDA_R_32F: c_int = 0;
pub const HETGPU_CUDA_R_16F: c_int = 2;

pub const HETGPU_METAL_BUFFER_COPY_IN: c_uint = 1;
pub const HETGPU_METAL_BUFFER_COPY_OUT: c_uint = 2;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct hetgpu_apple_metal_buffer_binding {
    pub host_ptr: *mut c_void,
    pub size: usize,
    pub flags: c_uint,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
unsafe extern "C" {
    pub fn hetgpu_apple_ane_gemm(
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
    ) -> c_int;

    pub fn hetgpu_apple_metal_gemm(
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
    ) -> c_int;

    pub fn hetgpu_apple_metal_compile_msl(
        source: *const c_char,
        label: *const c_char,
        out_module: *mut *mut c_void,
        out_log: *mut *mut c_char,
    ) -> c_int;

    pub fn hetgpu_apple_metal_get_function(
        module: *mut c_void,
        name: *const c_char,
        out_function: *mut *mut c_void,
        out_log: *mut *mut c_char,
    ) -> c_int;

    pub fn hetgpu_apple_metal_launch_raw(
        function: *mut c_void,
        buffers: *const hetgpu_apple_metal_buffer_binding,
        buffer_count: usize,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        out_log: *mut *mut c_char,
    ) -> c_int;

    pub fn hetgpu_apple_metal_release_module(module: *mut c_void) -> c_int;
    pub fn hetgpu_apple_metal_release_function(function: *mut c_void) -> c_int;
    pub fn hetgpu_apple_metal_free_string(value: *mut c_char);
}
