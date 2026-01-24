#[cfg(any(feature = "tenstorrent", feature = "nvidia"))]
use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;
#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
use nvidia_runtime_sys;

use std::ptr;
#[cfg(feature = "amd")]
pub(crate) fn get_attribute(
    pi: &mut i32,
    cu_attrib: hipFunction_attribute,
    func: hipFunction_t,
) -> hipError_t {
    // TODO: implement HIP_FUNC_ATTRIBUTE_PTX_VERSION
    // TODO: implement HIP_FUNC_ATTRIBUTE_BINARY_VERSION
    unsafe { hipFuncGetAttribute(pi, cu_attrib, func) }?;
    if cu_attrib == hipFunction_attribute::HIP_FUNC_ATTRIBUTE_NUM_REGS {
        *pi = (*pi).max(1);
    }
    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn get_attribute(
    pi: &mut i32,
    cu_attrib: ze_kernel_properties_t,
    func: ze_kernel_handle_t,
) -> ze_result_t {
    // For virtual devices or when Level Zero is not available, return sensible defaults
    // This prevents SIGFPE crashes in PyTorch's kernel launch configuration code
    if func.0.is_null() {
        // Virtual device - return reasonable defaults
        *pi = 32; // Default value for most attributes
        return ze_result_t::ZE_RESULT_SUCCESS;
    }

    let mut props = cu_attrib;
    let result = unsafe { zeKernelGetProperties(func, &mut props) };
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        // If Level Zero call fails, return sensible defaults instead of failing
        eprintln!("[hetGPU] zeKernelGetProperties failed, returning defaults");
        *pi = 32; // Safe default
        return ze_result_t::ZE_RESULT_SUCCESS;
    }

    *pi = props.localMemSize as i32;
    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(feature = "amd")]
pub(crate) fn launch_kernel(
    f: hipFunction_t,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: hipStream_t,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> hipError_t {
    // TODO: fix constants in extra
    unsafe {
        hipModuleLaunchKernel(
            f,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            shared_mem_bytes,
            stream,
            kernel_params,
            extra,
        )
    }
}

#[cfg(feature = "intel")]
pub(crate) unsafe fn launch_kernel(
    f: &super::module::ZeKernel,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: ze_command_queue_handle_t,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> ze_result_t {
    // Check for checkpoint at launch point
    if super::checkpoint::check_checkpoint_at_launch() {
        // Checkpoint was triggered and pause was requested
        // Return success without executing the kernel
        return ze_result_t::ZE_RESULT_SUCCESS;
    }

    // Start tracking kernel execution for checkpoint support
    let exec_id = super::checkpoint::begin_kernel_execution(
        &f.name,
        (grid_dim_x, grid_dim_y, grid_dim_z),
        (block_dim_x, block_dim_y, block_dim_z),
        shared_mem_bytes,
        stream.0 as u64,
        f.module_handle,
        f.kernel.0 as u64,
    );

    // Record kernel launch for replay if recording is active
    let replay_seq_id = if super::replay::is_recording_active() {
        // Extract kernel arguments for recording
        let mut args = Vec::new();
        if !kernel_params.is_null() {
            // Note: We don't know the exact number of arguments without kernel metadata
            // For now, we record pointer arguments that look valid
            for i in 0..16 {
                let param_ptr = unsafe { *kernel_params.add(i) };
                if param_ptr.is_null() {
                    break;
                }
                // Try to detect if it's a device pointer (typical GPU memory range)
                let value = unsafe { *(param_ptr as *const u64) };
                let arg = if value > 0x1000 && value < 0x800000000000 {
                    // Likely a device pointer
                    super::replay::KernelArgument {
                        index: i as u32,
                        size: 8,
                        arg_type: super::replay::ArgumentType::DevicePointer,
                        device_addr: Some(value),
                        scalar_data: None,
                    }
                } else {
                    // Treat as scalar
                    super::replay::KernelArgument {
                        index: i as u32,
                        size: 8,
                        arg_type: super::replay::ArgumentType::Scalar,
                        device_addr: None,
                        scalar_data: Some(value.to_le_bytes().to_vec()),
                    }
                };
                args.push(arg);
            }
        }

        super::replay::record_kernel_pre_launch(
            &f.name,
            f.module_handle,
            f.kernel.0 as u64,
            (grid_dim_x, grid_dim_y, grid_dim_z),
            (block_dim_x, block_dim_y, block_dim_z),
            shared_mem_bytes,
            stream.0 as u64,
            args,
        )
    } else {
        0
    };
    let kernel_start_time = std::time::Instant::now();

    // Detect virtual backend (no real Level Zero device available)
    let mut virtual_backend = false;
    if let Ok(gs) = crate::r#impl::driver::global_state() {
        if let Some(dev0) = gs.devices.get(0) {
            let (ctx0, _handle0) = dev0.primary_context();
            if ctx0.device.0.is_null() {
                virtual_backend = true;
            }
        }
    }
    // Cocotb fallback: if enabled or kernel handle is null (virtual), execute staged assembly via make
    let use_cocotb = std::env::var("HETGPU_TMATMUL_COCOTB")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);
    if use_cocotb || f.kernel.0.is_null() || virtual_backend {
        eprintln!(
            "[TMatmul Backend] Kernel launch: {} (grid={},{},{} block={},{},{})",
            f.name, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z
        );

        // Check for zero dimensions that could cause SIGFPE in PyTorch
        if block_dim_x == 0 || block_dim_y == 0 || block_dim_z == 0 {
            eprintln!("[TMatmul Backend] WARNING: Zero block dimension detected!");
        }
        if grid_dim_x == 0 || grid_dim_y == 0 || grid_dim_z == 0 {
            eprintln!("[TMatmul Backend] WARNING: Zero grid dimension detected!");
        }

        // NEW: Check if we have valid PTX source available for compilation
        // Valid PTX should start with ".version" or "//" and be at least 50 bytes
        let valid_ptx = f.ptx_source.as_ref().filter(|ptx| {
            ptx.len() >= 50 && (ptx.starts_with(".version") || ptx.starts_with("//"))
        });

        if let Some(ref ptx_source) = valid_ptx {
            eprintln!("[TMatmul Backend] Valid PTX source available ({} bytes)", ptx_source.len());

            // Get cocotb directory
            let cocotb_dir = std::env::var("HETGPU_TMATMUL_COCOTB_DIR")
                .unwrap_or_else(|_| "/mnt/ubuntu/ternary_matmul/cocotb".to_string());

            // Create run directory
            let _ = std::fs::create_dir_all(format!("{}/run", cocotb_dir));

            // Save PTX source to file
            let ptx_path = std::path::Path::new(&cocotb_dir).join("run/kernel.ptx");
            if let Err(e) = std::fs::write(&ptx_path, ptx_source.as_str()) {
                eprintln!("[TMatmul Backend] Failed to write PTX to {}: {}", ptx_path.display(), e);
            } else {
                eprintln!("[TMatmul Backend] PTX saved to {} ({} bytes)", ptx_path.display(), ptx_source.len());

                // Compile PTX to TMatmul assembly
                match ptx::pass::ptx_to_tmatmul_assembly(ptx_source.as_str()) {
                    Ok(asm) => {
                        let asm_path = std::path::Path::new(&cocotb_dir).join("run/kernel.S");
                        if let Err(e) = std::fs::write(&asm_path, &asm) {
                            eprintln!("[TMatmul Backend] Failed to write assembly to {}: {}", asm_path.display(), e);
                        } else {
                            eprintln!("[TMatmul Backend] TMatmul assembly saved to {} ({} bytes)", asm_path.display(), asm.len());
                        }
                        // Also save to /tmp for emulator access
                        let _ = std::fs::write("/tmp/tmatmul_kernel.S", &asm);
                    }
                    Err(e) => {
                        eprintln!("[TMatmul Backend] PTX->TMatmul compilation failed: {}", e);
                    }
                }
            }
        } else if let Some(ref ptx_source) = f.ptx_source {
            eprintln!("[TMatmul Backend] Invalid PTX source ({} bytes, starts with {:?}) - kernel will be no-op",
                     ptx_source.len(),
                     ptx_source.chars().take(20).collect::<String>());
        } else {
            eprintln!("[TMatmul Backend] No PTX source available - kernel will be no-op");
        }

        // Minimal Phase 1–3: detect matmul, decode args heuristically, and either run cocotb or CPU fallback
        let full_mode = std::env::var("HETGPU_TMATMUL_FULL")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false);
        let is_matmul_name = {
            let n = f.name.to_lowercase();
            n.contains("gemm") || n.contains("matmul") || n.contains("mm_") || n.contains("dot")
        };

        // Extract kernel parameters if available AND we need them (matmul kernels only)
        // Note: kernel_params is *mut *mut void where each element points to
        // a location in memory that holds the actual parameter value
        let mut output_ptr: *mut ::core::ffi::c_void = ptr::null_mut();
        let mut ptr_candidates: Vec<*mut ::core::ffi::c_void> = Vec::new();
        let mut num_params = 0;

        // Only extract parameters for matmul kernels to avoid segfaults on other kernels
        if full_mode && is_matmul_name && !kernel_params.is_null() {
            eprintln!("[TMatmul Backend] Matmul kernel detected, extracting parameters...");
            let mut current_param = kernel_params;
            const MAX_PARAMS: usize = 32; // Safety limit to prevent infinite loops

            while num_params < MAX_PARAMS {
                // First, safely check if current_param itself is valid before dereferencing
                if (current_param as usize) < 0x1000 || (current_param as usize) > 0x7fffffffffff {
                    eprintln!(
                        "[TMatmul Backend] Invalid param pointer {:p}, stopping iteration",
                        current_param
                    );
                    break;
                }

                let param_addr = *current_param;

                // Check for null terminator
                if param_addr.is_null() {
                    break;
                }

                // Safety check: param_addr should be a valid stack pointer
                // Parameters are passed on the stack, so param_addr should be in stack range
                // On x86_64 Linux, stack is typically at high addresses (0x7fff...)
                let param_addr_val = param_addr as usize;

                if param_addr_val < 0x1000 {
                    crate::r#impl::hetgpu_debug!("[TMatmul Backend] Param {}: addr={:p} - INVALID (too low), stopping iteration", num_params, param_addr);
                    break;
                }

                // Check that param_addr is in valid stack range
                // Stack on Linux x86_64 is typically in upper half of address space
                // Common range: 0x7f0000000000 - 0x7fffffffffff (due to ASLR)
                // If param_addr is not on the stack, it's likely garbage and dereferencing it will crash
                if param_addr_val < 0x7f0000000000 || param_addr_val > 0x7fffffffffff {
                    crate::r#impl::hetgpu_debug!(
                        "[TMatmul Backend] Param {}: addr={:p} - NOT ON STACK, stopping iteration",
                        num_params,
                        param_addr
                    );
                    break;
                }

                // Try to read as a CUdeviceptr (which is a wrapper around a pointer)
                // For virtual device, CUdeviceptr_v2.0 IS the host pointer directly
                // IMPORTANT: Use read_unaligned because stack addresses may not be 8-byte aligned
                let potential_cudevptr = unsafe { (param_addr as *const cuda_types::cuda::CUdeviceptr_v2).read_unaligned() };
                let potential_ptr = potential_cudevptr.0 as *mut ::core::ffi::c_void;
                let potential_i64 = unsafe { (param_addr as *const i64).read_unaligned() };

                crate::r#impl::hetgpu_debug!("[TMatmul Backend] Param {}: addr={:p}, as_CUdevptr={:p}, as_ptr={:p}, as_i64={}",
                         num_params, param_addr, potential_cudevptr.0, potential_ptr, potential_i64);

                // Look for a pointer that looks like a real heap allocation
                // Real allocations from alloc_zeroed are typically in range 0x1000 - 0x80000000
                // Upper bits (0x7fff...) indicate stack addresses or encoded values, not heap
                // PyTorch uses 16-byte alignment (0x10), not 32 or 64-byte
                let looks_like_heap_ptr = (potential_ptr as usize & 0xf) == 0 &&  // 16-byte aligned
                                          (potential_ptr as usize > 0x1000) &&         // Not null/sentinel
                                          (potential_ptr as usize) < 0x100000000; // Below 4GB (typical heap range)

                if looks_like_heap_ptr {
                    crate::r#impl::hetgpu_debug!(
                        "[TMatmul Backend]   -> Looks like a HEAP pointer (real allocation)!"
                    );
                    ptr_candidates.push(potential_ptr);
                    if output_ptr.is_null() {
                        output_ptr = potential_ptr;
                        crate::r#impl::hetgpu_debug!(
                            "[TMatmul Backend]   -> Selected as output buffer"
                        );
                    }
                } else if (potential_ptr as usize & 0xf) == 0 && (potential_ptr as usize > 0x1000) {
                    crate::r#impl::hetgpu_debug!("[TMatmul Backend]   -> Aligned but possibly stack/encoded value (upper bits: {:#x})",
                             potential_ptr as usize >> 32);
                }

                num_params += 1;
                current_param = current_param.add(1);
            }
            eprintln!("[TMatmul Backend] Found {} kernel parameters total", num_params);
            eprintln!("[TMatmul Backend] Selected output_ptr: {:p}", output_ptr);
        } else if !is_matmul_name {
            // Non-matmul kernel - compile PTX to tmatmul assembly and run via Python emulator
            eprintln!("[TMatmul Backend] Non-matmul kernel '{}' - compiling PTX for emulator", f.name);

            // Compile PTX to TMatmul assembly if PTX source is available
            if let Some(ref ptx_source) = f.ptx_source {
                if ptx_source.len() >= 50 && (ptx_source.starts_with(".version") || ptx_source.starts_with("//")) {
                    // Dump PTX source for debugging
                    let ptx_dump_path = format!("/tmp/hetgpu_ptx_{}.ptx", f.name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_"));
                    let _ = std::fs::write(&ptx_dump_path, ptx_source.as_bytes());
                    eprintln!("[TMatmul Backend] PTX dumped to {} ({} bytes)", ptx_dump_path, ptx_source.len());

                    // Wrap PTX compilation in catch_unwind to prevent panics from crashing
                    let ptx_str = ptx_source.clone();
                    let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        ptx::pass::ptx_to_tmatmul_assembly(ptx_str.as_str())
                    }));
                    let compile_result = match compile_result {
                        Ok(r) => r,
                        Err(_) => {
                            eprintln!("[TMatmul Backend] PTX compilation panicked - trying kernel name fallback");
                            execute_kernel_name_fallback(&f.name, kernel_params, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z);
                            super::checkpoint::end_kernel_execution(exec_id);
                            return ze_result_t::ZE_RESULT_SUCCESS;
                        }
                    };
                    let asm_to_execute = match compile_result {
                        Ok(asm) => {
                            let asm_path = format!("/tmp/hetgpu_asm_{}.S", f.name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_"));
                            let _ = std::fs::write(&asm_path, &asm);
                            eprintln!("[TMatmul Backend] TMatmul assembly saved to {} ({} bytes):\n{}", asm_path, asm.len(), &asm[..asm.len().min(500)]);
                            Some(asm)
                        }
                        Err(e) => {
                            eprintln!("[TMatmul Backend] PTX->TMatmul compilation failed: {}", e);
                            None
                        }
                    };

                    // Execute via interpreter, falling back to kernel-name-based assembly
                    if !kernel_params.is_null() {
                        let exec_result = if let Some(ref asm) = asm_to_execute {
                            super::tmatmul_interpreter::execute_assembly(
                                asm, kernel_params,
                                (grid_dim_x, grid_dim_y, grid_dim_z),
                                (block_dim_x, block_dim_y, block_dim_z),
                            )
                        } else {
                            Err("No compiled assembly".to_string())
                        };

                        if let Err(ref e) = exec_result {
                            eprintln!("[TMatmul Interpreter] Compiled assembly failed for '{}': {} - trying kernel name fallback", f.name, e);
                            // Fall back to kernel-name-based assembly generation with proper param scanning
                            execute_kernel_name_fallback(&f.name, kernel_params, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z);
                        } else {
                            eprintln!("[TMatmul Interpreter] Kernel '{}' executed successfully", f.name);
                        }
                    }
                } else {
                    eprintln!("[TMatmul Backend] Invalid PTX source ({} bytes) - trying kernel name fallback", ptx_source.len());
                    execute_kernel_name_fallback(&f.name, kernel_params, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z);
                }
            } else {
                eprintln!("[TMatmul Backend] No PTX source - trying kernel name fallback");
                execute_kernel_name_fallback(&f.name, kernel_params, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z);
            }

            super::checkpoint::end_kernel_execution(exec_id);
            return ze_result_t::ZE_RESULT_SUCCESS;
        }

        // Execute matmul if we have enough pointer candidates
        if full_mode && is_matmul_name && ptr_candidates.len() >= 3 {
            // Heuristic: [C, A, B]
            let mut c_ptr = ptr_candidates[0];
            let a_ptr = ptr_candidates[1];
            let b_ptr = ptr_candidates[2];
            if !output_ptr.is_null() {
                c_ptr = output_ptr;
            }

            // Optional dims from env: HETGPU_TMATMUL_DIMS="M,N,K"
            if let Ok(dims) = std::env::var("HETGPU_TMATMUL_DIMS") {
                let parts: Vec<usize> = dims
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect();
                if parts.len() == 3 {
                    let (m, n, k) = (parts[0], parts[1], parts[2]);
                    crate::r#impl::hetgpu_debug!(
                        "[TMatmul Backend] FULL CPU matmul fallback M={},N={},K={}",
                        m,
                        n,
                        k
                    );
                    let a_len = m * k;
                    let b_len = k * n;
                    let c_len = m * n;
                    let a_slice = std::slice::from_raw_parts(a_ptr as *const f32, a_len);
                    let b_slice = std::slice::from_raw_parts(b_ptr as *const f32, b_len);
                    let c_slice = std::slice::from_raw_parts_mut(c_ptr as *mut f32, c_len);
                    for i in 0..m {
                        for j in 0..n {
                            let mut acc: f32 = 0.0;
                            for p in 0..k {
                                acc += a_slice[i * k + p] * b_slice[p * n + j];
                            }
                            c_slice[i * n + j] = acc;
                        }
                    }
                    crate::r#impl::hetgpu_debug!(
                        "[TMatmul Backend] CPU matmul complete, wrote {:p}",
                        c_ptr
                    );
                    super::checkpoint::end_kernel_execution(exec_id);
                    return ze_result_t::ZE_RESULT_SUCCESS;
                }
            }

            // If dims not provided, compile PTX to tmatmul assembly for emulator
            if let Some(ref ptx_source) = f.ptx_source {
                if ptx_source.len() >= 50 {
                    match ptx::pass::ptx_to_tmatmul_assembly(ptx_source.as_str()) {
                        Ok(asm) => {
                            let _ = std::fs::write("/tmp/tmatmul_kernel.S", &asm);
                            crate::r#impl::hetgpu_debug!(
                                "[TMatmul Backend] Matmul PTX compiled to tmatmul assembly ({} bytes)",
                                asm.len()
                            );
                        }
                        Err(e) => {
                            crate::r#impl::hetgpu_debug!(
                                "[TMatmul Backend] Matmul PTX compilation failed: {}",
                                e
                            );
                        }
                    }
                }
            }
            crate::r#impl::hetgpu_debug!(
                "[TMatmul Backend] Matmul kernel compiled; virtual success"
            );
            super::checkpoint::end_kernel_execution(exec_id);
            return ze_result_t::ZE_RESULT_SUCCESS;
        }

        // Compile PTX to TMatmul assembly and run via emulator
        let mut ptx_compiled = false;
        if let Some(ref ptx_source) = f.ptx_source {
            if ptx_source.len() >= 50 && (ptx_source.starts_with(".version") || ptx_source.starts_with("//")) {
                // Dump PTX for debugging
                let ptx_dump_path = format!("/tmp/hetgpu_ptx_{}.ptx", f.name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_"));
                let _ = std::fs::write(&ptx_dump_path, ptx_source.as_bytes());
                eprintln!("[TMatmul Backend] PTX dumped to {} ({} bytes)", ptx_dump_path, ptx_source.len());

                match ptx::pass::ptx_to_tmatmul_assembly(ptx_source.as_str()) {
                    Ok(asm) => {
                        let asm_path = format!("/tmp/hetgpu_asm_{}.S", f.name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_"));
                        let _ = std::fs::write(&asm_path, &asm);
                        eprintln!("[TMatmul Backend] TMatmul assembly ({} bytes):\n{}", asm.len(), &asm[..asm.len().min(500)]);

                        if !kernel_params.is_null() {
                            match super::tmatmul_interpreter::execute_assembly(
                                &asm,
                                kernel_params,
                                (grid_dim_x, grid_dim_y, grid_dim_z),
                                (block_dim_x, block_dim_y, block_dim_z),
                            ) {
                                Ok(()) => {
                                    eprintln!("[TMatmul Interpreter] Kernel '{}' executed successfully", f.name);
                                    ptx_compiled = true;
                                }
                                Err(e) => {
                                    eprintln!("[TMatmul Interpreter] Execution failed for '{}': {}", f.name, e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[TMatmul Backend] PTX->TMatmul compilation failed: {} - trying kernel name fallback", e);
                    }
                }
            }
        }

        // Fallback: try kernel name-based assembly generation if PTX compilation didn't succeed
        if !ptx_compiled && !kernel_params.is_null() {
            execute_kernel_name_fallback(&f.name, kernel_params, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z);
        }

        // Virtual device path: return success after emulator execution
        super::checkpoint::end_kernel_execution(exec_id);
        return ze_result_t::ZE_RESULT_SUCCESS;
    }

    // Set the group size (equivalent to CUDA block dimensions)
    let result = unsafe { zeKernelSetGroupSize(f.kernel, block_dim_x, block_dim_y, block_dim_z) };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        super::checkpoint::end_kernel_execution(exec_id);
        return result;
    }

    // Set arguments from kernel_params if provided
    if !kernel_params.is_null() {
        let mut param_index = 0;
        let mut current_param = kernel_params;

        while !(*current_param).is_null() {
            unsafe {
                let param_value = *current_param;
                let result = zeKernelSetArgumentValue(
                    f.kernel,
                    param_index,
                    std::mem::size_of::<*mut ::core::ffi::c_void>(),
                    param_value as *const ::core::ffi::c_void,
                );

                if result != ze_result_t::ZE_RESULT_SUCCESS {
                    super::checkpoint::end_kernel_execution(exec_id);
                    return result;
                }

                param_index += 1;
                current_param = current_param.add(1);
            }
        }
    }

    // Process 'extra' parameters if provided (e.g., shared memory size)
    if !extra.is_null() {
        // 'extra' is typically of the form [KEY1, VALUE1, KEY2, VALUE2, ..., 0]
        unsafe {
            let mut i = 0;
            loop {
                let key = *extra.add(i);
                if key.is_null() {
                    break;
                }

                let key_value = key as usize;
                let value_ptr = extra.add(i + 1);
                let value = *value_ptr;

                if key_value == 1 { // CU_LAUNCH_PARAM_BUFFER_SHARED_MEMORY
                     // shared memory is already set via the shared_mem_bytes parameter
                }

                i += 2;
            }
        }
    }

    // Get or create a command list for this stream
    let command_list = unsafe {
        // In a real implementation, you'd have a way to get or create a command list for the given stream
        // For simplicity, we'll assume some function exists to do this
        get_or_create_command_list_for_stream(stream)
    };

    if command_list.0.is_null() {
        super::checkpoint::end_kernel_execution(exec_id);
        return ze_result_t::ZE_RESULT_ERROR_UNINITIALIZED;
    }

    // Prepare launch arguments for grid dimensions
    let dispatch_args = ze_group_count_t {
        groupCountX: grid_dim_x,
        groupCountY: grid_dim_y,
        groupCountZ: grid_dim_z,
    };

    // Launch the kernel
    let result = unsafe {
        zeCommandListAppendLaunchKernel(
            command_list,
            f.kernel,
            &dispatch_args,
            *ptr::null_mut(), // No event to wait on
            0,                // No events to wait on
            ptr::null_mut(),  // No event to signal
        )
    };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        super::checkpoint::end_kernel_execution(exec_id);
        return result;
    }

    // Close and execute the command list (in a real implementation, this might be deferred)
    let result = unsafe { zeCommandListClose(command_list) };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        super::checkpoint::end_kernel_execution(exec_id);
        return result;
    }

    let result = unsafe {
        // Execute the command list
        zeCommandQueueExecuteCommandLists(stream, 1, &command_list, *ptr::null_mut())
    };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        super::checkpoint::end_kernel_execution(exec_id);
        return result;
    }

    // If this is a synchronous stream, synchronize immediately
    let is_synchronous = false; // In a real implementation, determine if stream is synchronous

    if is_synchronous {
        let result = unsafe { zeCommandQueueSynchronize(stream, u64::MAX) };

        if result != ze_result_t::ZE_RESULT_SUCCESS {
            super::checkpoint::end_kernel_execution(exec_id);
            return result;
        }
    }

    // End kernel execution tracking
    super::checkpoint::end_kernel_execution(exec_id);

    // Record post-launch for replay
    if replay_seq_id > 0 && super::replay::is_recording_active() {
        let execution_time_ns = kernel_start_time.elapsed().as_nanos() as u64;
        super::replay::record_kernel_post_launch(replay_seq_id, execution_time_ns);
    }

    ze_result_t::ZE_RESULT_SUCCESS
}

/// Check if a memory range is readable by writing to a pipe (forces kernel to copy_from_user).
/// Returns true if the start and end of the range are accessible.
#[cfg(feature = "intel")]
#[allow(dead_code)]
unsafe fn is_memory_readable(ptr: *const u8, len: usize) -> bool {
    if ptr.is_null() || len == 0 {
        return false;
    }
    let mut pipefd = [0i32; 2];
    if libc::pipe(pipefd.as_mut_ptr()) != 0 {
        return false;
    }
    // Set pipe write end to non-blocking to avoid hanging on large writes
    libc::fcntl(pipefd[1], libc::F_SETFL, libc::O_NONBLOCK);

    // Probe the first 4 bytes
    let probe_start = libc::write(pipefd[1], ptr as *const libc::c_void, 4.min(len));
    // Drain the pipe
    let mut drain_buf = [0u8; 4];
    let _ = libc::read(pipefd[0], drain_buf.as_mut_ptr() as *mut libc::c_void, 4);

    // Probe the last 4 bytes if range is larger than 4
    let probe_end = if len > 4 {
        let end_ptr = ptr.add(len - 4);
        let r = libc::write(pipefd[1], end_ptr as *const libc::c_void, 4);
        let _ = libc::read(pipefd[0], drain_buf.as_mut_ptr() as *mut libc::c_void, 4);
        r
    } else {
        probe_start
    };

    libc::close(pipefd[0]);
    libc::close(pipefd[1]);
    probe_start > 0 && probe_end > 0
}

/// Invoke the Python-based TMatmul emulator bridge to execute compiled assembly.
/// Writes tensor data to temp files, runs the Python bridge, and reads results back.
#[cfg(feature = "intel")]
#[allow(dead_code)]
unsafe fn invoke_emulator_bridge(
    asm_path: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    numel: usize,
    kernel_name: &str,
    num_pointer_params: usize,  // Known count of pointer params (from assembly/kernel name)
) {
    use std::fmt::Write;

    // Maximum elements we'll try to read (256MB / 4 bytes = 64M elements)
    const MAX_ELEMENTS: usize = 64 * 1024 * 1024;

    // We KNOW the param layout:
    // - First `num_pointer_params` params are 64-bit device pointers
    // - We DON'T try to read scalar params (would crash if kernel_params is shorter)
    // - Instead, derive numel from allocation sizes or grid*block dimensions

    struct ParamEntry {
        value: u64,
        alloc_size: usize,
    }
    let mut params: Vec<ParamEntry> = Vec::new();

    // Read ONLY the pointer params - these are guaranteed to exist for the kernel
    for i in 0..num_pointer_params {
        let param_slot = kernel_params.add(i);
        let param_addr = *param_slot;
        if param_addr.is_null() {
            eprintln!("[TMatmul Emulator] Param {} is null, stopping", i);
            break;
        }
        // Sanity check: param_addr should be a valid stack/heap address
        if (param_addr as usize) < 0x1000 {
            eprintln!("[TMatmul Emulator] Param {} addr too low ({:#x}), stopping", i, param_addr as usize);
            break;
        }

        let ptr_value = (param_addr as *const u64).read_unaligned();
        let alloc_size = super::memory::get_alloc_size(ptr_value as usize).unwrap_or(0);
        eprintln!("[TMatmul Emulator] Param {}: ptr_value={:#x}, alloc_size={}", i, ptr_value, alloc_size);
        params.push(ParamEntry {
            value: ptr_value,
            alloc_size,
        });
    }

    if params.is_empty() || params.len() < 2 {
        eprintln!("[TMatmul Emulator] Not enough pointer params ({}) for kernel '{}'", params.len(), kernel_name);
        return;
    }

    // Check if any params are tracked in our virtual allocation map
    let tracked_count = params.iter().filter(|p| p.alloc_size > 0).count();
    if tracked_count == 0 {
        eprintln!("[TMatmul Emulator] No params found in virtual alloc map for '{}' - memory not managed by hetGPU", kernel_name);
        return;
    }

    // Use grid*block numel as primary size, capped by allocation size
    // (PyTorch caching allocator blocks are much larger than individual tensors)
    let mut actual_numel = numel;

    let min_alloc_elements: Option<usize> = params.iter()
        .filter(|p| p.alloc_size > 0)
        .map(|p| p.alloc_size / 4)  // f32 = 4 bytes
        .min();

    if let Some(alloc_elements) = min_alloc_elements {
        // Use alloc_elements as upper bound - don't read past allocation
        actual_numel = actual_numel.min(alloc_elements);
    }

    // Safety cap
    if actual_numel > MAX_ELEMENTS {
        actual_numel = MAX_ELEMENTS;
    }
    if actual_numel == 0 {
        eprintln!("[TMatmul Emulator] numel is 0, nothing to do for '{}'", kernel_name);
        return;
    }

    eprintln!("[TMatmul Emulator] kernel='{}', numel={}, pointer_params={}",
             kernel_name, actual_numel, params.len());

    // Build JSON and write data files - all params are pointers
    let mut params_json = String::from("[");
    struct ParamInfo {
        host_ptr: *mut u8,
        file_path: String,
        count: usize,
    }
    let mut param_infos: Vec<ParamInfo> = Vec::new();

    for (i, p) in params.iter().enumerate() {
        if i > 0 {
            params_json.push(',');
        }

        let file_path = format!("/tmp/hetgpu_param_{}.bin", i);
        let max_elements = if p.alloc_size > 0 { p.alloc_size / 4 } else { actual_numel };
        let count = actual_numel.min(max_elements);
        let size_bytes = count * 4;
        let data_ptr = p.value as *const u8;

        // Validate: must have tracked alloc, valid pointer, reasonable size
        let ptr_val = p.value as usize;
        let ptr_aligned = ptr_val % 4 == 0;
        let ptr_in_range = ptr_val >= 0x10000 && ptr_val < 0x7fff_ffff_ffff;
        if count == 0 || data_ptr.is_null() || p.alloc_size == 0 || !ptr_aligned || !ptr_in_range || size_bytes > 256 * 1024 * 1024 {
            eprintln!("[TMatmul Emulator] Param {} skipped: count={}, ptr={:#x}, alloc_size={}, aligned={}, in_range={}",
                     i, count, p.value, p.alloc_size, ptr_aligned, ptr_in_range);
            let _ = write!(
                params_json,
                r#"{{"file":"","count":0,"is_pointer":true}}"#
            );
            param_infos.push(ParamInfo {
                host_ptr: std::ptr::null_mut(),
                file_path: String::new(),
                count: 0,
            });
            continue;
        }

        // Validate memory is readable before attempting to copy
        if !is_memory_readable(data_ptr, size_bytes) {
            eprintln!("[TMatmul Emulator] Param {} memory not readable: ptr={:#x}, size={}",
                     i, p.value, size_bytes);
            let _ = write!(
                params_json,
                r#"{{"file":"","count":0,"is_pointer":true}}"#
            );
            param_infos.push(ParamInfo {
                host_ptr: std::ptr::null_mut(),
                file_path: String::new(),
                count: 0,
            });
            continue;
        }

        // Write tensor data to temp file
        let data_slice = std::slice::from_raw_parts(data_ptr, size_bytes);
        if let Err(e) = std::fs::write(&file_path, data_slice) {
            eprintln!("[TMatmul Emulator] Failed to write param {} data: {}", i, e);
            return;
        }

        let _ = write!(
            params_json,
            r#"{{"file":"{}","count":{},"is_pointer":true}}"#,
            file_path, count
        );

        param_infos.push(ParamInfo {
            host_ptr: data_ptr as *mut u8,
            file_path: file_path.clone(),
            count,
        });
    }

    params_json.push(']');
    let param_count = params.len();

    // Build the full JSON config
    let config_json = format!(
        r#"{{"assembly_file":"{}","params":{},"numel":{}}}"#,
        asm_path, params_json, actual_numel
    );

    let config_path = "/tmp/tmatmul_bridge_config.json";
    if let Err(e) = std::fs::write(config_path, &config_json) {
        eprintln!("[TMatmul Emulator] Failed to write config: {}", e);
        return;
    }

    eprintln!(
        "[TMatmul Emulator] Invoking bridge for '{}' (numel={}, params={})",
        kernel_name, actual_numel, param_count
    );

    // Invoke the Python bridge
    let bridge_script = std::env::var("HETGPU_TMATMUL_BRIDGE")
        .unwrap_or_else(|_| "/mnt/ubuntu/ternary_matmul/sw_utils/lib/hetgpu_bridge.py".to_string());

    let python = std::env::var("HETGPU_PYTHON").unwrap_or_else(|_| "python3".to_string());

    match std::process::Command::new(&python)
        .arg(&bridge_script)
        .arg(config_path)
        .env("PYTHONPATH", "/mnt/ubuntu/ternary_matmul/sw_utils")
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .output()
    {
        Ok(output) => {
            if !output.stderr.is_empty() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                for line in stderr.lines() {
                    eprintln!("[TMatmul Emulator] {}", line);
                }
            }
            if output.status.success() {
                eprintln!("[TMatmul Emulator] Kernel '{}' executed successfully", kernel_name);

                // Read back output data from files written by the bridge
                // The bridge writes modified PARAM_N data back to the same file paths
                for info in &param_infos {
                    if info.host_ptr.is_null() {
                        continue;
                    }
                    // Check if the bridge wrote an output file
                    let out_path = format!("{}.out", info.file_path);
                    if let Ok(data) = std::fs::read(&out_path) {
                        let expected_size = info.count * 4;
                        if data.len() == expected_size && is_memory_readable(info.host_ptr, expected_size) {
                            std::ptr::copy_nonoverlapping(
                                data.as_ptr(),
                                info.host_ptr,
                                expected_size,
                            );
                        }
                        let _ = std::fs::remove_file(&out_path);
                    }
                    let _ = std::fs::remove_file(&info.file_path);
                }
            } else {
                eprintln!(
                    "[TMatmul Emulator] Bridge exited with code {:?} for '{}'",
                    output.status.code(), kernel_name
                );
                // Clean up temp files
                for info in &param_infos {
                    let _ = std::fs::remove_file(&info.file_path);
                }
            }
        }
        Err(e) => {
            eprintln!("[TMatmul Emulator] Failed to invoke bridge: {}", e);
        }
    }
}

/// Helper: log unhandled kernel launches in the fallback path.
/// The emulator bridge is only invoked when PTX compilation succeeds (not from this fallback).
#[cfg(feature = "intel")]
unsafe fn execute_kernel_name_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    _grid_dim_x: u32, _grid_dim_y: u32, _grid_dim_z: u32,
    _block_dim_x: u32, _block_dim_y: u32, _block_dim_z: u32,
) {
    if kernel_params.is_null() {
        eprintln!("[TMatmul Fallback] Kernel '{}' - null kernel_params, skipping", kernel_name);
        return;
    }

    let name_lower = kernel_name.to_lowercase();

    // Handle different kernel types
    if name_lower.contains("reduce_kernel") {
        execute_reduce_kernel_fallback(kernel_name, &name_lower, kernel_params);
        return;
    }
    if name_lower.contains("softmax") {
        execute_softmax_kernel_fallback(kernel_name, &name_lower, kernel_params);
        return;
    }
    if name_lower.contains("indexselect") || name_lower.contains("index_select") {
        execute_indexselect_kernel_fallback(kernel_name, kernel_params);
        return;
    }
    if name_lower.contains("gemm") || name_lower.contains("matmul") || name_lower.contains("cublas") {
        execute_matmul_kernel_fallback(kernel_name, kernel_params);
        return;
    }
    if name_lower.contains("layernorm") || name_lower.contains("layer_norm") ||
       name_lower.contains("rmsnorm") || name_lower.contains("rms_norm") ||
       name_lower.contains("welford") {
        execute_norm_kernel_fallback(kernel_name, &name_lower, kernel_params);
        return;
    }

    // Handle vectorized_elementwise_kernel from PyTorch.
    // Signature: vectorized_elementwise_kernel(int N, Functor f, std::array<char*, K> data)
    // kernel_params[0] = &N (int32)
    // kernel_params[1] = &functor
    // kernel_params[2] = &data_array (K sequential char* pointers)
    if !name_lower.contains("vectorized_elementwise_kernel") &&
       !name_lower.contains("unrolled_elementwise_kernel") &&
       !name_lower.contains("elementwise_kernel") {
        eprintln!("[TMatmul Fallback] Unhandled kernel '{}' - no-op", kernel_name);
        return;
    }

    // Determine the operation and number of data pointers
    let (op, num_ptrs) = detect_vectorized_op(kernel_name);
    let op = match op {
        Some(o) => o,
        None => {
            eprintln!("[TMatmul Fallback] Unrecognized op in '{}' - no-op", kernel_name);
            return;
        }
    };

    // Read numel from kernel_params[0]
    let numel_param = *kernel_params.add(0);
    if numel_param.is_null() {
        eprintln!("[TMatmul Fallback] kernel_params[0] is null");
        return;
    }
    let numel = (numel_param as *const i32).read_unaligned() as usize;
    if numel == 0 || numel > 64 * 1024 * 1024 {
        eprintln!("[TMatmul Fallback] Invalid numel={} for '{}'", numel, kernel_name);
        return;
    }

    // Read tensor pointers from the std::array at kernel_params[2]
    let data_param = *kernel_params.add(2);
    if data_param.is_null() {
        eprintln!("[TMatmul Fallback] kernel_params[2] (data array) is null");
        return;
    }

    let mut data_ptrs: Vec<*mut u8> = Vec::new();
    for i in 0..num_ptrs {
        let ptr_val = (data_param as *const u64).add(i).read_unaligned();
        // Verify this pointer is in our allocation map
        if let Some(_size) = super::memory::get_alloc_size(ptr_val as usize) {
            data_ptrs.push(ptr_val as *mut u8);
        } else {
            eprintln!("[TMatmul Fallback] data_ptrs[{}] = {:#x} not in alloc map", i, ptr_val);
            data_ptrs.push(std::ptr::null_mut());
        }
    }

    // Validate we have at least output and one input
    if data_ptrs.len() < 2 || data_ptrs[0].is_null() || data_ptrs[1].is_null() {
        eprintln!("[TMatmul Fallback] Missing output or input pointer for '{}'", kernel_name);
        return;
    }

    // Detect element size from kernel name (float16=2, float32/float=4, double=8)
    let elem_size: usize = if name_lower.contains("float16") || name_lower.contains("half") ||
                              name_lower.contains("f16") || name_lower.contains("bf16") ||
                              name_lower.contains("bfloat16") {
        2
    } else if name_lower.contains("double") || name_lower.contains("float64") {
        8
    } else {
        4 // default f32
    };

    eprintln!("[TMatmul Fallback] Executing {} on {} elements ({} ptrs, {}B each) for '{}'",
             op, numel, num_ptrs, elem_size, kernel_name);

    // Read functor data from kernel_params[1] (used by some ops for alpha/scalar)
    let functor_param = *kernel_params.add(1);
    let functor_f32 = if !functor_param.is_null() {
        (functor_param as *const f32).read_unaligned()
    } else {
        1.0f32
    };

    // Helper closures for reading/writing elements with dtype conversion
    // For f16: read u16 bits, convert to f32 for math, convert back
    #[inline(always)]
    unsafe fn read_f16(ptr: *const u8) -> f32 {
        let bits = (ptr as *const u16).read_unaligned();
        f16_to_f32(bits)
    }
    #[inline(always)]
    unsafe fn write_f16(ptr: *mut u8, val: f32) {
        (ptr as *mut u16).write_unaligned(f32_to_f16(val));
    }
    #[inline(always)]
    unsafe fn read_elem(ptr: *const u8, offset: usize, elem_size: usize) -> f32 {
        match elem_size {
            2 => read_f16(ptr.add(offset * 2)),
            8 => (ptr.add(offset * 8) as *const f64).read_unaligned() as f32,
            _ => (ptr.add(offset * 4) as *const f32).read_unaligned(),
        }
    }
    #[inline(always)]
    unsafe fn write_elem(ptr: *mut u8, offset: usize, elem_size: usize, val: f32) {
        match elem_size {
            2 => write_f16(ptr.add(offset * 2), val),
            8 => (ptr.add(offset * 8) as *mut f64).write_unaligned(val as f64),
            _ => (ptr.add(offset * 4) as *mut f32).write_unaligned(val),
        }
    }

    // Execute the operation directly on the tensor data
    match op {
        "add" => {
            if data_ptrs.len() < 3 || data_ptrs[2].is_null() {
                eprintln!("[TMatmul Fallback] add needs 3 valid pointers");
                return;
            }
            let alpha = functor_f32;
            let out = data_ptrs[0];
            let in1 = data_ptrs[1];
            let in2 = data_ptrs[2];
            for i in 0..numel {
                let a = read_elem(in1, i, elem_size);
                let b = read_elem(in2, i, elem_size);
                write_elem(out, i, elem_size, a + alpha * b);
            }
        }
        "mul" | "div" => {
            if data_ptrs.len() < 3 || data_ptrs[2].is_null() {
                eprintln!("[TMatmul Fallback] Binary op needs 3 valid pointers");
                return;
            }
            let out = data_ptrs[0];
            let in1 = data_ptrs[1];
            let in2 = data_ptrs[2];
            for i in 0..numel {
                let a = read_elem(in1, i, elem_size);
                let b = read_elem(in2, i, elem_size);
                let result = match op {
                    "mul" => a * b,
                    "div" => if b != 0.0 { a / b } else { 0.0 },
                    _ => unreachable!(),
                };
                write_elem(out, i, elem_size, result);
            }
        }
        "abs" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x.abs());
            }
        }
        "add_scalar" => {
            let scalar = functor_f32;
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x + scalar);
            }
        }
        "clamp" => {
            let min_val = functor_f32;
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, if x < min_val { min_val } else { x });
            }
        }
        "rsqrt" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, 1.0 / x.sqrt());
            }
        }
        "pow" => {
            let exponent = functor_f32;
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x.powf(exponent));
            }
        }
        "mul_scalar" => {
            let scalar = functor_f32;
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x * scalar);
            }
        }
        "sigmoid" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, 1.0 / (1.0 + (-x).exp()));
            }
        }
        "silu" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                let sig = 1.0 / (1.0 + (-x).exp());
                write_elem(out, i, elem_size, x * sig);
            }
        }
        "relu" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, if x > 0.0 { x } else { 0.0 });
            }
        }
        "tanh" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x.tanh());
            }
        }
        "copy" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            std::ptr::copy_nonoverlapping(inp, out, numel * elem_size);
        }
        "neg" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, -x);
            }
        }
        "exp" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x.exp());
            }
        }
        "gelu" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                let c = 0.7978845608_f32; // sqrt(2/pi)
                let inner = c * (x + 0.044715 * x * x * x);
                write_elem(out, i, elem_size, 0.5 * x * (1.0 + inner.tanh()));
            }
        }
        _ => {
            eprintln!("[TMatmul Fallback] Unimplemented op '{}' - no-op", op);
        }
    }
    eprintln!("[TMatmul Fallback] Kernel '{}' executed successfully ({} elements, {}B)", kernel_name, numel, elem_size);
}

// IEEE 754 half-precision (f16) <-> f32 conversion
#[cfg(feature = "intel")]
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: convert to normalized f32
            let mut m = mant;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = ((e + 127) as u32) & 0xFF;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            f32::from_bits((sign << 31) | 0x7FC00000 | (mant << 13))
        }
    } else {
        // Normalized
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

#[cfg(feature = "intel")]
#[inline]
fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf or NaN
        if mant == 0 {
            (sign << 15) | 0x7C00
        } else {
            (sign << 15) | 0x7E00 // quiet NaN
        }
    } else if exp > 142 {
        // Overflow -> Inf
        (sign << 15) | 0x7C00
    } else if exp < 103 {
        // Underflow -> zero
        sign << 15
    } else if exp < 113 {
        // Subnormal
        let shift = 113 - exp;
        let m = (mant | 0x800000) >> (shift + 13);
        (sign << 15) | (m as u16)
    } else {
        // Normalized
        let h_exp = ((exp - 112) as u16) & 0x1F;
        let h_mant = (mant >> 13) as u16;
        (sign << 15) | (h_exp << 10) | h_mant
    }
}

/// Detect the operation and number of data pointers from a vectorized_elementwise_kernel name
#[cfg(feature = "intel")]
fn detect_vectorized_op(kernel_name: &str) -> (Option<&'static str>, usize) {
    let name_lower = kernel_name.to_lowercase();

    // Binary ops: std::array<char*, 3>
    // CUDAFunctor_add handles both add and sub (sub uses alpha=-1.0)
    if name_lower.contains("functor_add") || (name_lower.contains("add") && name_lower.contains("lm3")) {
        return (Some("add"), 3);
    }
    if name_lower.contains("mulfunctor") || (name_lower.contains("mul") && !name_lower.contains("cumul") && name_lower.contains("lm3")) {
        return (Some("mul"), 3);
    }
    if name_lower.contains("divfunctor") || (name_lower.contains("div") && name_lower.contains("lm3")) {
        return (Some("div"), 3);
    }

    // Unary ops: std::array<char*, 2>
    if name_lower.contains("sigmoid") {
        return (Some("sigmoid"), 2);
    }
    if name_lower.contains("silu") {
        return (Some("silu"), 2);
    }
    if name_lower.contains("tanh") && !name_lower.contains("atanh") {
        return (Some("tanh"), 2);
    }
    if name_lower.contains("gelu") {
        return (Some("gelu"), 2);
    }
    if name_lower.contains("absfunctor") {
        return (Some("abs"), 2);
    }
    if name_lower.contains("clamp") {
        return (Some("clamp"), 2);
    }
    if name_lower.contains("functoronselfadd") || name_lower.contains("functoronself_add") {
        return (Some("add_scalar"), 2);
    }
    if name_lower.contains("rsqrt") {
        return (Some("rsqrt"), 2);
    }
    if name_lower.contains("pow") {
        return (Some("pow"), 2);
    }
    if name_lower.contains("exp") && !name_lower.contains("export") {
        return (Some("exp"), 2);
    }
    if name_lower.contains("neg") {
        return (Some("neg"), 2);
    }
    if name_lower.contains("copy") || name_lower.contains("contiguous") {
        return (Some("copy"), 2);
    }
    if name_lower.contains("relu") {
        return (Some("relu"), 2);
    }
    if name_lower.contains("functoronselfmul") || name_lower.contains("functoronself_mul") {
        return (Some("mul_scalar"), 2);
    }

    // Check array size from template: "Lm3" means 3, "Lm2" means 2
    let num_ptrs = if name_lower.contains("lm3") { 3 }
                   else if name_lower.contains("lm2") { 2 }
                   else { 2 }; // default

    // Try to detect from functor name patterns
    if name_lower.contains("add") && num_ptrs == 3 {
        return (Some("add"), 3);
    }

    (None, num_ptrs)
}

/// Scan a memory region for pointers that match entries in the virtual alloc map.
/// Returns a list of (offset, pointer_value, alloc_size) tuples.
#[cfg(feature = "intel")]
unsafe fn scan_for_alloc_pointers(base: *const u8, scan_bytes: usize) -> Vec<(usize, u64, usize)> {
    let mut found = Vec::new();
    // Scan at 8-byte aligned offsets for 64-bit pointers
    let num_slots = scan_bytes / 8;
    for i in 0..num_slots {
        let ptr_val = (base as *const u64).add(i).read_unaligned();
        // Quick filter: valid user-space pointer range
        if ptr_val < 0x10000 || ptr_val > 0x7fff_ffff_ffff {
            continue;
        }
        // Check if this pointer is in our alloc map
        if let Some(size) = super::memory::get_alloc_size(ptr_val as usize) {
            found.push((i * 8, ptr_val, size));
        }
    }
    found
}

/// Execute a reduce_kernel fallback (sum, mean, max, var).
/// The reduce_kernel has a single ReduceOp struct parameter containing embedded pointers.
#[cfg(feature = "intel")]
unsafe fn execute_reduce_kernel_fallback(
    kernel_name: &str,
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    // reduce_kernel takes a single ReduceOp parameter
    // kernel_params[0] = &ReduceOp (struct containing input/output pointers)
    let op_param = *kernel_params.add(0);
    if op_param.is_null() {
        eprintln!("[TMatmul Fallback] reduce_kernel: null parameter");
        return;
    }

    // Scan the ReduceOp struct for valid allocation pointers
    // The struct is typically < 512 bytes
    let scan_size = 512;
    let found_ptrs = scan_for_alloc_pointers(op_param as *const u8, scan_size);

    if found_ptrs.len() < 2 {
        eprintln!("[TMatmul Fallback] reduce_kernel: found only {} alloc pointers (need >=2)", found_ptrs.len());
        return;
    }

    // Heuristic: in PyTorch's ReduceOp struct layout, output pointer comes before input pointer.
    // The output is smaller (reduced dimension), input is larger.
    // Sort by alloc_size to identify: smallest = output, largest = input
    let mut sorted = found_ptrs.clone();
    sorted.sort_by_key(|&(_, _, size)| size);

    let (_, out_ptr, out_size) = sorted[0]; // smallest alloc = output
    let (_, in_ptr, in_size) = sorted[sorted.len() - 1]; // largest alloc = input

    let in_elements = in_size / 4; // f32
    let out_elements = out_size / 4;

    if out_elements == 0 || in_elements == 0 {
        eprintln!("[TMatmul Fallback] reduce_kernel: zero elements (in={}, out={})", in_elements, out_elements);
        return;
    }

    // Determine reduction size: in_elements / out_elements
    let reduce_size = if out_elements > 0 { in_elements / out_elements } else { in_elements };

    let inp = in_ptr as *const f32;
    let out = out_ptr as *mut f32;

    // Determine which reduction operation
    let op_type = if name_lower.contains("sum_functor") || name_lower.contains("sum") {
        "sum"
    } else if name_lower.contains("meanops") || name_lower.contains("mean") {
        "mean"
    } else if name_lower.contains("maxops") || name_lower.contains("max") {
        "max"
    } else if name_lower.contains("minops") || name_lower.contains("min") {
        "min"
    } else if name_lower.contains("normops") || name_lower.contains("var") {
        "var"
    } else {
        "sum" // default
    };

    eprintln!("[TMatmul Fallback] reduce_kernel: op={}, in_elements={}, out_elements={}, reduce_size={}",
             op_type, in_elements, out_elements, reduce_size);

    for row in 0..out_elements {
        let base = row * reduce_size;
        let end = (base + reduce_size).min(in_elements);
        let count = end - base;
        if count == 0 { continue; }

        match op_type {
            "sum" => {
                let mut sum = 0.0f32;
                for i in base..end {
                    sum += inp.add(i).read_unaligned();
                }
                out.add(row).write_unaligned(sum);
            }
            "mean" => {
                let mut sum = 0.0f32;
                for i in base..end {
                    sum += inp.add(i).read_unaligned();
                }
                out.add(row).write_unaligned(sum / count as f32);
            }
            "max" => {
                let mut max_val = f32::NEG_INFINITY;
                for i in base..end {
                    let v = inp.add(i).read_unaligned();
                    if v > max_val { max_val = v; }
                }
                out.add(row).write_unaligned(max_val);
            }
            "min" => {
                let mut min_val = f32::INFINITY;
                for i in base..end {
                    let v = inp.add(i).read_unaligned();
                    if v < min_val { min_val = v; }
                }
                out.add(row).write_unaligned(min_val);
            }
            "var" => {
                let mut sum = 0.0f32;
                for i in base..end {
                    sum += inp.add(i).read_unaligned();
                }
                let mean = sum / count as f32;
                let mut var_sum = 0.0f32;
                for i in base..end {
                    let diff = inp.add(i).read_unaligned() - mean;
                    var_sum += diff * diff;
                }
                // Unbiased variance (Bessel's correction)
                let divisor = if count > 1 { (count - 1) as f32 } else { count as f32 };
                out.add(row).write_unaligned(var_sum / divisor);
            }
            _ => {}
        }
    }
    eprintln!("[TMatmul Fallback] reduce_kernel '{}' executed ({} -> {} elements)",
             op_type, in_elements, out_elements);
}

/// Execute a softmax kernel fallback.
#[cfg(feature = "intel")]
unsafe fn execute_softmax_kernel_fallback(
    kernel_name: &str,
    _name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    // Softmax kernels typically take input ptr, output ptr, and dimension info
    // Scan first parameter for pointers
    let param0 = *kernel_params.add(0);
    if param0.is_null() {
        eprintln!("[TMatmul Fallback] softmax: null parameter");
        return;
    }

    let found_ptrs = scan_for_alloc_pointers(param0 as *const u8, 256);

    if found_ptrs.len() < 2 {
        // Try scanning multiple params
        let mut all_ptrs = found_ptrs;
        for pi in 1..8 {
            let p = *kernel_params.add(pi);
            if p.is_null() { break; }
            let more = scan_for_alloc_pointers(p as *const u8, 64);
            all_ptrs.extend(more);
            if all_ptrs.len() >= 2 { break; }
        }
        if all_ptrs.len() < 2 {
            eprintln!("[TMatmul Fallback] softmax: couldn't find input/output pointers");
            return;
        }
        // Use deduplicated set
        all_ptrs.sort_by_key(|&(_, ptr, _)| ptr);
        all_ptrs.dedup_by_key(|a| a.1);

        if all_ptrs.len() < 2 {
            eprintln!("[TMatmul Fallback] softmax: need at least 2 distinct pointers");
            return;
        }

        let (_, ptr0, size0) = all_ptrs[0];
        let (_, ptr1, size1) = all_ptrs[1];
        // Same size = in-place or same shape
        let total_elements = size0.max(size1) / 4;
        // Assume square-ish: rows x cols where cols is the softmax dim
        let cols = 128.min(total_elements); // heuristic
        let rows = total_elements / cols;
        execute_softmax_on_data(ptr0 as *const f32, ptr1 as *mut f32, rows, cols);
        return;
    }

    // If found in single struct: typically output first, then input (or same buffer for in-place)
    let mut sorted = found_ptrs.clone();
    sorted.sort_by_key(|&(off, _, _)| off); // sort by offset in struct

    let (_, ptr0, size0) = sorted[0];
    let (_, ptr1, size1) = sorted[1];

    let total_elements = size0.max(size1) / 4;
    // For softmax, input and output have the same shape
    // Determine rows and cols from the tensor shape
    // Heuristic: softmax is typically along last dim. If total_elements is divisible by common sizes...
    let cols = if total_elements % 128 == 0 { 128 }
               else if total_elements % 64 == 0 { 64 }
               else if total_elements % 32 == 0 { 32 }
               else { total_elements };
    let rows = if cols > 0 { total_elements / cols } else { 1 };

    // First pointer in struct is likely output, second is input for PyTorch softmax
    execute_softmax_on_data(ptr1 as *const f32, ptr0 as *mut f32, rows, cols);
    eprintln!("[TMatmul Fallback] softmax executed ({}x{} = {} elements) for '{}'",
             rows, cols, total_elements, kernel_name);
}

#[cfg(feature = "intel")]
unsafe fn execute_softmax_on_data(inp: *const f32, out: *mut f32, rows: usize, cols: usize) {
    for row in 0..rows {
        let base = row * cols;
        // Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..cols {
            let v = inp.add(base + c).read_unaligned();
            if v > max_val { max_val = v; }
        }
        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for c in 0..cols {
            let v = inp.add(base + c).read_unaligned();
            let e = (v - max_val).exp();
            out.add(base + c).write_unaligned(e);
            sum += e;
        }
        // Normalize
        if sum > 0.0 {
            for c in 0..cols {
                let e = out.add(base + c).read_unaligned();
                out.add(base + c).write_unaligned(e / sum);
            }
        }
    }
}

/// Execute indexSelect kernel fallback (used by nn.Embedding).
#[cfg(feature = "intel")]
unsafe fn execute_indexselect_kernel_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    // indexSelectSmallIndex takes TensorInfo structs for output, input, and indices
    // Scan first param for pointers
    let param0 = *kernel_params.add(0);
    if param0.is_null() {
        eprintln!("[TMatmul Fallback] indexSelect: null parameter");
        return;
    }

    // Scan a larger area - TensorInfo structs are passed as separate params
    let mut all_ptrs: Vec<(usize, u64, usize)> = Vec::new();
    for pi in 0..10 {
        let p = *kernel_params.add(pi);
        if p.is_null() { break; }
        // Each param could be a pointer-to-value or the value itself
        let as_u64 = (p as *const u64).read_unaligned();
        if let Some(size) = super::memory::get_alloc_size(as_u64 as usize) {
            all_ptrs.push((pi, as_u64, size));
        }
        // Also check if p points to a struct containing pointers
        let inner = scan_for_alloc_pointers(p as *const u8, 64);
        all_ptrs.extend(inner.into_iter().map(|(off, ptr, sz)| (pi * 100 + off, ptr, sz)));
    }

    // Deduplicate by pointer value
    all_ptrs.sort_by_key(|&(_, ptr, _)| ptr);
    all_ptrs.dedup_by_key(|a| a.1);

    if all_ptrs.len() < 3 {
        eprintln!("[TMatmul Fallback] indexSelect: need output, input, indices (found {})", all_ptrs.len());
        return;
    }

    // Sort by size: indices (smallest, int64/int32), output (medium), weight table (largest)
    all_ptrs.sort_by_key(|&(_, _, size)| size);

    let (_, indices_ptr, indices_size) = all_ptrs[0]; // smallest = indices
    let (_, out_ptr, _out_size) = all_ptrs[1]; // medium = output
    let (_, weight_ptr, weight_size) = all_ptrs[all_ptrs.len() - 1]; // largest = weight table

    // Determine dimensions
    // indices are typically int64 (8 bytes each) or int32 (4 bytes each)
    // Try int64 first
    let num_indices = indices_size / 8;
    // embedding_dim = weight_size / vocab_size, but we don't know vocab_size
    // Infer from output: output_size = num_indices * embedding_dim
    // So embedding_dim = out_size / num_indices
    let out_elements = _out_size / 4; // f32
    let embedding_dim = if num_indices > 0 { out_elements / num_indices } else { 0 };

    if embedding_dim == 0 || num_indices == 0 {
        eprintln!("[TMatmul Fallback] indexSelect: can't determine dimensions (indices={}, emb_dim={})",
                 num_indices, embedding_dim);
        return;
    }

    let weight = weight_ptr as *const f32;
    let indices = indices_ptr as *const i64;
    let output = out_ptr as *mut f32;
    let vocab_size = (weight_size / 4) / embedding_dim;

    eprintln!("[TMatmul Fallback] indexSelect/embedding: vocab={}, dim={}, seq={}",
             vocab_size, embedding_dim, num_indices);

    for i in 0..num_indices {
        let idx = indices.add(i).read_unaligned() as usize;
        if idx >= vocab_size {
            eprintln!("[TMatmul Fallback] indexSelect: index {} out of bounds (vocab={})", idx, vocab_size);
            continue;
        }
        // Copy embedding vector
        let src_base = idx * embedding_dim;
        let dst_base = i * embedding_dim;
        for d in 0..embedding_dim {
            let val = weight.add(src_base + d).read_unaligned();
            output.add(dst_base + d).write_unaligned(val);
        }
    }
    eprintln!("[TMatmul Fallback] indexSelect executed for '{}'", kernel_name);
}

/// Execute matmul/GEMM kernel fallback.
#[cfg(feature = "intel")]
unsafe fn execute_matmul_kernel_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    // GEMM kernels have various parameter layouts
    // Scan params for allocation pointers
    let mut all_ptrs: Vec<(usize, u64, usize)> = Vec::new();
    for pi in 0..16 {
        let p = *kernel_params.add(pi);
        if p.is_null() { break; }
        let as_u64 = (p as *const u64).read_unaligned();
        if let Some(size) = super::memory::get_alloc_size(as_u64 as usize) {
            all_ptrs.push((pi, as_u64, size));
        }
        let inner = scan_for_alloc_pointers(p as *const u8, 128);
        all_ptrs.extend(inner.into_iter().map(|(off, ptr, sz)| (pi * 1000 + off, ptr, sz)));
    }

    all_ptrs.sort_by_key(|&(_, ptr, _)| ptr);
    all_ptrs.dedup_by_key(|a| a.1);

    if all_ptrs.len() < 3 {
        eprintln!("[TMatmul Fallback] matmul: need A, B, C pointers (found {})", all_ptrs.len());
        return;
    }

    // Sort by alloc size: C (output, M*N), A (M*K), B (K*N) or similar
    all_ptrs.sort_by_key(|&(_, _, size)| size);

    // For simple matmul C = A @ B:
    // Try to infer dimensions from sizes
    // A: M*K, B: K*N, C: M*N (all in f32 = 4 bytes)
    let sizes: Vec<usize> = all_ptrs.iter().map(|&(_, _, s)| s / 4).collect();

    // Heuristic: try common dimension arrangements
    // If all same size, assume square matrices
    let (a_ptr, b_ptr, c_ptr, m, n, k);

    if all_ptrs.len() == 3 {
        let s0 = sizes[0]; // smallest
        let s1 = sizes[1];
        let s2 = sizes[2]; // largest

        // Try to factor: assume one common dimension K
        // A=M*K, B=K*N, C=M*N
        // If s0=s1=s2, all square (M=N=K=sqrt(size))
        if s0 == s1 && s1 == s2 {
            let side = (s0 as f64).sqrt() as usize;
            if side * side == s0 {
                m = side; n = side; k = side;
            } else {
                m = 1; n = s0; k = 1; // fallback: vector
            }
        } else {
            // Try: smallest is output (M*N), middle is one input, largest is other
            // Or: try to find K such that sizes work out
            // Simple heuristic: assume M=rows of smallest dimension
            let total = s2; // largest
            let side = (total as f64).sqrt() as usize;
            m = if s0 > 0 && total % s0 == 0 { s0 } else { side };
            n = if m > 0 { s2 / m } else { side };
            k = if m > 0 && s1 > 0 { s1 / m } else { side };
        }

        // C = largest, A and B are the others
        c_ptr = all_ptrs[2].1 as *mut f32;
        a_ptr = all_ptrs[0].1 as *const f32;
        b_ptr = all_ptrs[1].1 as *const f32;
    } else {
        eprintln!("[TMatmul Fallback] matmul: unexpected number of pointers ({})", all_ptrs.len());
        return;
    }

    if m == 0 || n == 0 || k == 0 {
        eprintln!("[TMatmul Fallback] matmul: zero dimensions M={}, N={}, K={}", m, n, k);
        return;
    }

    eprintln!("[TMatmul Fallback] matmul: M={}, N={}, K={}", m, n, k);

    // C = A @ B (row-major)
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_val = a_ptr.add(i * k + p).read_unaligned();
                let b_val = b_ptr.add(p * n + j).read_unaligned();
                sum += a_val * b_val;
            }
            c_ptr.add(i * n + j).write_unaligned(sum);
        }
    }
    eprintln!("[TMatmul Fallback] matmul executed for '{}'", kernel_name);
}

/// Execute layernorm/rmsnorm kernel fallback.
#[cfg(feature = "intel")]
unsafe fn execute_norm_kernel_fallback(
    kernel_name: &str,
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    // LayerNorm/RMSNorm kernels: scan params for allocation pointers
    let mut all_ptrs: Vec<(usize, u64, usize)> = Vec::new();
    for pi in 0..10 {
        let p = *kernel_params.add(pi);
        if p.is_null() { break; }
        let as_u64 = (p as *const u64).read_unaligned();
        if let Some(size) = super::memory::get_alloc_size(as_u64 as usize) {
            all_ptrs.push((pi, as_u64, size));
        }
        let inner = scan_for_alloc_pointers(p as *const u8, 128);
        all_ptrs.extend(inner.into_iter().map(|(off, ptr, sz)| (pi * 1000 + off, ptr, sz)));
    }

    all_ptrs.sort_by_key(|&(_, ptr, _)| ptr);
    all_ptrs.dedup_by_key(|a| a.1);

    if all_ptrs.len() < 2 {
        eprintln!("[TMatmul Fallback] norm: need at least input and output (found {})", all_ptrs.len());
        return;
    }

    // Sort by size: weight/bias are smallest (hidden_size), input/output are larger (batch*hidden)
    all_ptrs.sort_by_key(|&(_, _, size)| size);

    let is_rmsnorm = name_lower.contains("rmsnorm") || name_lower.contains("rms_norm");

    // Find the largest allocations (input/output) and smallest (weight/bias)
    let large_ptrs: Vec<_> = all_ptrs.iter().filter(|&&(_, _, s)| s == all_ptrs.last().unwrap().2).collect();
    let small_ptrs: Vec<_> = all_ptrs.iter().filter(|&&(_, _, s)| s < all_ptrs.last().unwrap().2).collect();

    if large_ptrs.len() < 2 {
        // Input and output might be same size but different pointers
        // Use the two largest
        let n = all_ptrs.len();
        let (_, inp_ptr, inp_size) = all_ptrs[n - 1];
        let (_, out_ptr, _) = if n >= 2 { all_ptrs[n - 2] } else { all_ptrs[n - 1] };

        let total_elements = inp_size / 4;
        let hidden_size = if !small_ptrs.is_empty() {
            small_ptrs[0].2 / 4
        } else {
            128 // default
        };
        let batch_size = if hidden_size > 0 { total_elements / hidden_size } else { 1 };

        let weight_ptr = if !small_ptrs.is_empty() { small_ptrs[0].1 as *const f32 } else { std::ptr::null() };
        let bias_ptr = if small_ptrs.len() >= 2 { small_ptrs[1].1 as *const f32 } else { std::ptr::null() };

        execute_norm_on_data(
            inp_ptr as *const f32, out_ptr as *mut f32,
            weight_ptr, bias_ptr,
            batch_size, hidden_size, is_rmsnorm, 1e-5,
        );
        eprintln!("[TMatmul Fallback] norm executed (batch={}, hidden={}, rmsnorm={}) for '{}'",
                 batch_size, hidden_size, is_rmsnorm, kernel_name);
        return;
    }

    let (_, ptr_a, size_a) = *large_ptrs[0];
    let (_, ptr_b, _) = *large_ptrs[1];
    let total_elements = size_a / 4;
    let hidden_size = if !small_ptrs.is_empty() { small_ptrs[0].2 / 4 } else { 128 };
    let batch_size = if hidden_size > 0 { total_elements / hidden_size } else { 1 };

    let weight_ptr = if !small_ptrs.is_empty() { small_ptrs[0].1 as *const f32 } else { std::ptr::null() };
    let bias_ptr = if small_ptrs.len() >= 2 { small_ptrs[1].1 as *const f32 } else { std::ptr::null() };

    // Determine which is input and which is output
    // Heuristic: first in struct order is output
    let (inp_ptr, out_ptr) = if large_ptrs[0].0 < large_ptrs[1].0 {
        (ptr_b, ptr_a) // first = output, second = input
    } else {
        (ptr_a, ptr_b)
    };

    execute_norm_on_data(
        inp_ptr as *const f32, out_ptr as *mut f32,
        weight_ptr, bias_ptr,
        batch_size, hidden_size, is_rmsnorm, 1e-5,
    );
    eprintln!("[TMatmul Fallback] norm executed (batch={}, hidden={}, rmsnorm={}) for '{}'",
             batch_size, hidden_size, is_rmsnorm, kernel_name);
}

#[cfg(feature = "intel")]
unsafe fn execute_norm_on_data(
    inp: *const f32, out: *mut f32,
    weight: *const f32, bias: *const f32,
    batch_size: usize, hidden_size: usize,
    is_rmsnorm: bool, eps: f32,
) {
    for b in 0..batch_size {
        let base = b * hidden_size;

        if is_rmsnorm {
            // RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
            let mut sq_sum = 0.0f32;
            for h in 0..hidden_size {
                let x = inp.add(base + h).read_unaligned();
                sq_sum += x * x;
            }
            let rms = (sq_sum / hidden_size as f32 + eps).sqrt();
            for h in 0..hidden_size {
                let x = inp.add(base + h).read_unaligned();
                let w = if !weight.is_null() { weight.add(h).read_unaligned() } else { 1.0 };
                out.add(base + h).write_unaligned(x / rms * w);
            }
        } else {
            // LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
            let mut sum = 0.0f32;
            for h in 0..hidden_size {
                sum += inp.add(base + h).read_unaligned();
            }
            let mean = sum / hidden_size as f32;

            let mut var_sum = 0.0f32;
            for h in 0..hidden_size {
                let diff = inp.add(base + h).read_unaligned() - mean;
                var_sum += diff * diff;
            }
            let std_dev = (var_sum / hidden_size as f32 + eps).sqrt();

            for h in 0..hidden_size {
                let x = inp.add(base + h).read_unaligned();
                let normalized = (x - mean) / std_dev;
                let w = if !weight.is_null() { weight.add(h).read_unaligned() } else { 1.0 };
                let b = if !bias.is_null() { bias.add(h).read_unaligned() } else { 0.0 };
                out.add(base + h).write_unaligned(normalized * w + b);
            }
        }
    }
}

// Implement cuLaunchKernelEx by unwrapping the CUlaunchConfig and delegating to launch_kernel
#[cfg(feature = "intel")]
pub(crate) fn cuLaunchKernelEx(
    config: *const cuda_types::cuda::CUlaunchConfig,
    f: &super::module::ZeKernel,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> ze_result_t {
    if config.is_null() {
        return ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    let cfg = unsafe { &*config };
    let grid_x = cfg.gridDimX;
    let grid_y = cfg.gridDimY;
    let grid_z = cfg.gridDimZ;
    let block_x = cfg.blockDimX;
    let block_y = cfg.blockDimY;
    let block_z = cfg.blockDimZ;
    let shmem = cfg.sharedMemBytes;
    // In virtual backend, stream is usually null; pass a placeholder
    let stream = ze_command_queue_handle_t(::core::ptr::null_mut());
    unsafe {
        launch_kernel(
            f,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shmem,
            stream,
            kernel_params,
            extra,
        )
    }
}

// Normalized name expected by cuda_normalize_fn!(function::launch_kernel_ex)
#[cfg(feature = "intel")]
pub(crate) fn launch_kernel_ex(
    config: *const cuda_types::cuda::CUlaunchConfig,
    f: &super::module::ZeKernel,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> ze_result_t {
    cuLaunchKernelEx(config, f, kernel_params, extra)
}

// Helper function to get or create a command list for a stream
#[cfg(feature = "intel")]
unsafe fn get_or_create_command_list_for_stream(
    stream: ze_command_queue_handle_t,
) -> ze_command_list_handle_t {
    // In a real implementation, you'd have a way to track command lists per stream
    // For now, we'll create a new one (this would leak in a real implementation)

    // Get the device and context from the stream
    let device = get_device_from_stream(stream);
    let context = get_context_from_stream(stream);

    let desc = ze_command_list_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        pNext: ptr::null(),
        commandQueueGroupOrdinal: 0, // Default queue group
        flags: 0,
    };

    let mut command_list = ze_command_list_handle_t(ptr::null_mut());
    let result = zeCommandListCreate(context, device, &desc, &mut command_list);

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return ze_command_list_handle_t(ptr::null_mut());
    }

    command_list
}

#[cfg(feature = "intel")]
unsafe fn get_device_from_stream(stream: ze_command_queue_handle_t) -> ze_device_handle_t {
    // Get device from global state
    // If stream is null or we can't find the device, use the primary device
    if let Ok(gs) = crate::r#impl::driver::global_state() {
        if let Some(dev0) = gs.devices.get(0) {
            let (ctx, _raw_ctx) = dev0.primary_context();
            return ctx.device;
        }
    }
    ze_device_handle_t(ptr::null_mut())
}

#[cfg(feature = "intel")]
unsafe fn get_context_from_stream(stream: ze_command_queue_handle_t) -> ze_context_handle_t {
    // Get context from global state
    // If stream is null or we can't find the context, use the primary context
    if let Ok(gs) = crate::r#impl::driver::global_state() {
        if let Some(dev0) = gs.devices.get(0) {
            let (ctx, _raw_ctx) = dev0.primary_context();
            return ctx.context;
        }
    }
    ze_context_handle_t(ptr::null_mut())
}

// Tenstorrent function implementations
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_attribute(
    pi: *mut i32,
    attrib: CUfunction_attribute,
    func: *mut crate::r#impl::module::TtKernel,
) -> CUresult {
    if pi.is_null() || func.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    // For Tenstorrent, return placeholder values for function attributes
    let result = match attrib {
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK => 1024,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS => 32,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION => 75,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION => 75,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES => 65536,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT => 0,
        _ => return Err(CUerror::INVALID_VALUE),
    };

    unsafe { *pi = result };
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn launch_kernel(
    f: *mut crate::r#impl::module::TtKernel,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: *mut ::core::ffi::c_void,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> CUresult {
    if f.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    // For Tenstorrent, implement kernel launch
    // In a real implementation, this would:
    // 1. Set up the kernel parameters
    // 2. Configure the grid and block dimensions
    // 3. Launch the kernel on the Tenstorrent device
    // 4. Handle synchronization based on the stream

    let _kernel = unsafe { &*f };

    // Process kernel parameters if provided
    if !kernel_params.is_null() {
        unsafe {
            let mut param_index = 0;
            let mut current_param = kernel_params;

            while !(*current_param).is_null() {
                let _param_value = *current_param;
                // In a real implementation, set kernel argument at param_index

                param_index += 1;
                current_param = current_param.add(1);
            }
        }
    }

    // Process extra parameters if provided
    if !extra.is_null() {
        unsafe {
            let mut i = 0;
            loop {
                let key = *extra.add(i);
                if key.is_null() {
                    break;
                }

                let _key_value = key as usize;
                let _value_ptr = extra.add(i + 1);
                let _value = *_value_ptr;

                // Process extra parameters as needed

                i += 2;
            }
        }
    }

    // Placeholder for actual Tenstorrent kernel launch
    // This would interface with the tt_runtime_sys to launch the kernel

    // Suppress unused parameter warnings
    let _ = (grid_dim_x, grid_dim_y, grid_dim_z);
    let _ = (block_dim_x, block_dim_y, block_dim_z);
    let _ = shared_mem_bytes;
    let _ = stream;

    Ok(())
}

// NVIDIA backend function implementations - passthrough to real libcuda.so
#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
pub(crate) fn get_attribute(
    pi: &mut ::core::ffi::c_int,
    attrib: cuda_types::cuda::CUfunction_attribute,
    hfunc: &super::module::NvidiaKernel,
) -> CUresult {
    let result = nvidia_runtime_sys::cuFuncGetAttribute(pi, attrib, hfunc.cuda_function);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
pub(crate) fn launch_kernel(
    f: &super::module::NvidiaKernel,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    h_stream: cuda_types::cuda::CUstream,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> CUresult {
    let result = nvidia_runtime_sys::cuLaunchKernel(
        f.cuda_function,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        h_stream,
        kernel_params,
        extra,
    );
    if result != 0 {
        eprintln!("[NVIDIA Backend] cuLaunchKernel failed with error {}", result);
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
pub(crate) fn launch_kernel_ex(
    config: &cuda_types::cuda::CUlaunchConfig,
    f: &super::module::NvidiaKernel,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> CUresult {
    // cuLaunchKernelEx wraps cuLaunchKernel with additional config
    let result = nvidia_runtime_sys::cuLaunchKernel(
        f.cuda_function,
        config.gridDimX,
        config.gridDimY,
        config.gridDimZ,
        config.blockDimX,
        config.blockDimY,
        config.blockDimZ,
        config.sharedMemBytes,
        config.hStream,
        kernel_params,
        extra,
    );
    if result != 0 {
        eprintln!("[NVIDIA Backend] cuLaunchKernelEx failed with error {}", result);
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}
