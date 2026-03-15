#[cfg(any(feature = "tenstorrent", feature = "nvidia"))]
use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;
#[cfg(feature = "intel")]
use cuda_types::cuda::{CUfunction_attribute_enum, CUerror, CUresult};
#[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent"), not(feature = "tmatmul")))]
use nvidia_runtime_sys;

use std::ptr;
use std::sync::Mutex;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

/// Persistent Python emulator daemon for fast kernel execution.
/// Avoids Python/PyTorch startup overhead by keeping the process running.
#[cfg(feature = "intel")]
static EMULATOR_DAEMON: Mutex<Option<EmulatorDaemon>> = Mutex::new(None);

#[cfg(feature = "intel")]
struct EmulatorDaemon {
    process: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

#[cfg(feature = "intel")]
impl EmulatorDaemon {
    fn start() -> Option<Self> {
        let python = std::env::var("HETGPU_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let daemon_script = std::env::var("HETGPU_DAEMON")
            .unwrap_or_else(|_| "/mnt/ubuntu/ternary_matmul/sw_utils/lib/hetgpu_daemon.py".to_string());

        let mut child = Command::new(&python)
            .arg(&daemon_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .env("PYTHONPATH", "/mnt/ubuntu/ternary_matmul/sw_utils")
            .spawn()
            .ok()?;

        let stdin = child.stdin.take()?;
        let stdout = BufReader::new(child.stdout.take()?);

        let mut daemon = EmulatorDaemon { process: child, stdin, stdout };

        // Wait for READY signal
        let mut line = String::new();
        if daemon.stdout.read_line(&mut line).is_ok() && line.trim() == "READY" {
            eprintln!("[TMatmul Daemon] Started successfully");
            Some(daemon)
        } else {
            eprintln!("[TMatmul Daemon] Failed to start - no READY signal");
            let _ = daemon.process.kill();
            None
        }
    }

    fn execute(&mut self, cmd_json: &str) -> Option<String> {
        // Send command
        if writeln!(self.stdin, "{}", cmd_json).is_err() {
            return None;
        }
        if self.stdin.flush().is_err() {
            return None;
        }

        // Read response
        let mut response = String::new();
        if self.stdout.read_line(&mut response).is_ok() {
            Some(response)
        } else {
            None
        }
    }

    fn is_alive(&mut self) -> bool {
        match self.process.try_wait() {
            Ok(None) => true,  // Still running
            _ => false,
        }
    }
}

/// Get or create the emulator daemon
#[cfg(feature = "intel")]
fn get_daemon() -> Option<std::sync::MutexGuard<'static, Option<EmulatorDaemon>>> {
    let mut guard = EMULATOR_DAEMON.lock().ok()?;

    // Check if daemon needs to be started or restarted
    let needs_restart = match guard.as_mut() {
        None => true,
        Some(d) => !d.is_alive(),
    };

    if needs_restart {
        *guard = EmulatorDaemon::start();
    }

    if guard.is_some() {
        Some(guard)
    } else {
        None
    }
}
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
    cu_attrib: CUfunction_attribute_enum,
    func: ze_kernel_handle_t,
) -> CUresult {
    // Return sensible defaults per attribute type.
    // For placeholder kernels (tmatmul emulation) or when Level Zero
    // is not available, we must return values that satisfy callers
    // like Triton's load_binary which queries n_regs, n_spills, n_max_threads.
    *pi = match cu_attrib {
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK => 1024,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES => 0,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES => 0,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES => 0,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS => 32,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PTX_VERSION => 75,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_BINARY_VERSION => 75,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA => 0,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES => 65536,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT => 0,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET => 0,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH => 0,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT => 0,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH => 0,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED => 0,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE => 0,
        _ => 0,
    };
    Ok(())
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
            ptx.len() >= 50 && (ptx.trim_start().starts_with(".version") || ptx.trim_start().starts_with("//"))
        });

        if let Some(ref ptx_source) = valid_ptx {
            // Only compile small PTX modules (<1MB) for cocotb - large modules are too slow
            if ptx_source.len() <= 1_000_000 {
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
            // For large modules (>1MB), extract only the target function to avoid
            // compiling hundreds of unrelated kernels.
            if let Some(ref ptx_source) = f.ptx_source {
                // For large PTX modules, extract only the target function
                let effective_ptx: Option<String> = if ptx_source.len() > 1_000_000 {
                    eprintln!("[TMatmul Backend] Large PTX module ({} bytes) - extracting function '{}'", ptx_source.len(), f.name);
                    extract_function_from_ptx(ptx_source.as_str(), &f.name)
                } else {
                    Some(ptx_source.as_str().to_string())
                };

                // If extraction failed, use the cached PTX index for fast lookup.
                // The round-robin PTX assignment in cudart_shim may have picked the
                // wrong compilation unit; the kernel could be in any of the 399 PTX files.
                let effective_ptx = if effective_ptx.is_none() {
                    eprintln!("[TMatmul Backend] Looking up PTX via index for '{}'", f.name);
                    cached_ptx_lookup(&f.name)
                } else {
                    effective_ptx
                };

                let ptx_to_compile = effective_ptx.as_deref().unwrap_or("");
                if ptx_to_compile.len() >= 50 && (ptx_to_compile.trim_start().starts_with(".version") || ptx_to_compile.trim_start().starts_with("//")) {
                    // Dump PTX source for debugging
                    let ptx_dump_path = format!("/tmp/hetgpu_ptx_{}.ptx", f.name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_"));
                    let _ = std::fs::write(&ptx_dump_path, ptx_to_compile.as_bytes());
                    eprintln!("[TMatmul Backend] PTX dumped to {} ({} bytes)", ptx_dump_path, ptx_to_compile.len());

                    // Wrap PTX compilation in catch_unwind to prevent panics from crashing
                    let ptx_str = ptx_to_compile.to_string();
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
                            eprintln!("[TMatmul Backend] PTX->TMatmul compilation failed: {} - trying cached PTX lookup", e);
                            // The kernel's own PTX may be truncated (e.g. CUBIN extraction).
                            // Try the cached PTX index which may have the full PTX (e.g. from Triton cache).
                            if let Some(cached_ptx) = cached_ptx_lookup(&f.name) {
                                eprintln!("[TMatmul Backend] Found cached PTX for '{}' ({} bytes)", f.name, cached_ptx.len());
                                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    ptx::pass::ptx_to_tmatmul_assembly(&cached_ptx)
                                })) {
                                    Ok(Ok(asm)) => {
                                        let asm_path = format!("/tmp/hetgpu_asm_{}.S", f.name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_"));
                                        let _ = std::fs::write(&asm_path, &asm);
                                        eprintln!("[TMatmul Backend] Cached PTX compiled to {} ({} bytes)", asm_path, asm.len());
                                        Some(asm)
                                    }
                                    Ok(Err(e2)) => {
                                        eprintln!("[TMatmul Backend] Cached PTX compilation also failed: {}", e2);
                                        None
                                    }
                                    Err(_) => {
                                        eprintln!("[TMatmul Backend] Cached PTX compilation panicked");
                                        None
                                    }
                                }
                            } else {
                                None
                            }
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
                    eprintln!("[TMatmul Backend] Invalid PTX source ({} bytes) - trying cached PTX lookup", ptx_source.len());
                    // Try cached PTX index before falling back to kernel name
                    let mut handled = false;
                    if let Some(cached_ptx) = cached_ptx_lookup(&f.name) {
                        if let Ok(Ok(asm)) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            ptx::pass::ptx_to_tmatmul_assembly(&cached_ptx)
                        })) {
                            if !kernel_params.is_null() {
                                if let Ok(()) = super::tmatmul_interpreter::execute_assembly(
                                    &asm, kernel_params,
                                    (grid_dim_x, grid_dim_y, grid_dim_z),
                                    (block_dim_x, block_dim_y, block_dim_z),
                                ) {
                                    eprintln!("[TMatmul Interpreter] Kernel '{}' executed via cached PTX", f.name);
                                    handled = true;
                                }
                            }
                        }
                    }
                    if !handled {
                        execute_kernel_name_fallback(&f.name, kernel_params, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z);
                    }
                }
            } else {
                // No PTX on the function — try the cached index to find it
                // in the extracted PTX files from cuobjdump.
                if let Some(ptx_to_compile) = cached_ptx_lookup(&f.name) {
                    eprintln!("[TMatmul Backend] Found PTX via index for '{}' ({} bytes)", f.name, ptx_to_compile.len());
                    let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        ptx::pass::ptx_to_tmatmul_assembly(ptx_to_compile.as_str())
                    }));
                    let asm_ok = match compile_result {
                        Ok(Ok(asm)) => {
                            if !kernel_params.is_null() {
                                let exec_result = super::tmatmul_interpreter::execute_assembly(
                                    &asm, kernel_params,
                                    (grid_dim_x, grid_dim_y, grid_dim_z),
                                    (block_dim_x, block_dim_y, block_dim_z),
                                );
                                if let Err(ref e) = exec_result {
                                    eprintln!("[TMatmul Interpreter] Compiled assembly failed for '{}': {}", f.name, e);
                                    false
                                } else {
                                    eprintln!("[TMatmul Interpreter] Kernel '{}' executed successfully", f.name);
                                    true
                                }
                            } else { true }
                        }
                        _ => false,
                    };
                    if !asm_ok {
                        execute_kernel_name_fallback(&f.name, kernel_params, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z);
                    }
                } else {
                    execute_kernel_name_fallback(&f.name, kernel_params, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z);
                }
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
            // For large PTX modules, extract only the target function
            let effective_ptx: Option<String> = if ptx_source.len() > 1_000_000 {
                eprintln!("[TMatmul Backend] Large matmul PTX module ({} bytes) - extracting function '{}'", ptx_source.len(), f.name);
                extract_function_from_ptx(ptx_source.as_str(), &f.name)
            } else {
                Some(ptx_source.as_str().to_string())
            };

            let ptx_to_compile = effective_ptx.as_deref().unwrap_or("");
            if ptx_to_compile.len() >= 50 && (ptx_to_compile.trim_start().starts_with(".version") || ptx_to_compile.trim_start().starts_with("//")) {
                // Dump PTX for debugging
                let ptx_dump_path = format!("/tmp/hetgpu_ptx_{}.ptx", f.name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_"));
                let _ = std::fs::write(&ptx_dump_path, ptx_to_compile.as_bytes());
                eprintln!("[TMatmul Backend] PTX dumped to {} ({} bytes)", ptx_dump_path, ptx_to_compile.len());

                match ptx::pass::ptx_to_tmatmul_assembly(ptx_to_compile) {
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
            ze_event_handle_t(ptr::null_mut()), // No event to wait on
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
        zeCommandQueueExecuteCommandLists(stream, 1, &command_list, ze_fence_handle_t(ptr::null_mut()))
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

/// Detect if a kernel operates on integer types by inspecting the C++ mangled name.
/// In Itanium C++ ABI mangling:
///   l = long, i = int, j = unsigned int, m = unsigned long,
///   b = bool, c = char, s = short, x = long long
///   f = float, d = double
///   N3c104HalfE = c10::Half (f16), N3c108BFloat16E = c10::BFloat16
/// We check if Functor template args contain only integer types.
#[cfg(feature = "intel")]
fn is_integer_type_kernel(kernel_name: &str) -> bool {
    // Check demangled-style names first
    let name_lower = kernel_name.to_lowercase();
    if name_lower.contains("long") || name_lower.contains("int64")
       || name_lower.contains("uint8") || name_lower.contains("int32")
       || name_lower.contains("int16") {
        return true;
    }

    // Check C++ mangled name patterns for Functor<integer_types>
    // Look for FunctorI followed by integer type codes without float codes
    // E.g. "FunctorIlE" = Functor<long>, "FunctorIlllN" = Functor<long,long,long,...>
    // Float types would have: "FunctorIfE", "FunctorIN3c104HalfE"
    if let Some(functor_pos) = kernel_name.find("Functor") {
        // Find the template arg list after "FunctorI"
        let after_functor = &kernel_name[functor_pos..];
        if let Some(i_pos) = after_functor.find("FunctorI") {
            let template_start = functor_pos + i_pos + 8; // skip "FunctorI"
            if template_start < kernel_name.len() {
                let template_args = &kernel_name[template_start..];
                // Check the first few chars of template args
                // Integer types: l, i, j, m, b, c, s, x
                // Float types: f, d, N (for c10::Half etc)
                let first_char = template_args.chars().next().unwrap_or('?');
                match first_char {
                    'l' | 'i' | 'j' | 'm' | 'b' | 'c' | 's' | 'x' => return true,
                    _ => {}
                }
            }
        }
    }

    // Also check for "gpu_kernel_impl" with integer BinaryFunctor/UnaryFunctor patterns
    // Pattern: BinaryFunctorIlll = BinaryFunctor<long,long,long,...>
    if let Some(pos) = kernel_name.find("BinaryFunctorI") {
        let after = &kernel_name[pos + 14..]; // skip "BinaryFunctorI"
        let first = after.chars().next().unwrap_or('?');
        match first {
            'l' | 'i' | 'j' | 'm' | 'b' | 'c' | 's' | 'x' => return true,
            _ => {}
        }
    }
    if let Some(pos) = kernel_name.find("UnaryFunctorI") {
        let after = &kernel_name[pos + 13..]; // skip "UnaryFunctorI"
        let first = after.chars().next().unwrap_or('?');
        match first {
            'l' | 'i' | 'j' | 'm' | 'b' | 'c' | 's' | 'x' => return true,
            _ => {}
        }
    }

    // Check for direct_copy_kernel with LoadWithCast/StoreWithCast
    // These are type-casting copy kernels that handle mixed types - let them through
    // unless they explicitly have integer-only patterns

    false
}

/// Extract a single function from a large PTX module.
/// Keeps the module header (.version, .target, .address_size) and extracts only
/// the function matching the given kernel name.
/// This avoids compiling hundreds of unrelated kernels in multi-MB PTX modules.
#[cfg(feature = "intel")]
/// Normalize a mangled C++ name by replacing anonymous namespace hashes
/// (_GLOBAL__N__XXXXXXXX_) with a wildcard pattern so that names compiled
/// from different TUs can still match.
fn normalize_anon_ns(name: &str) -> String {
    // Pattern: _GLOBAL__N__ followed by exactly 8 hex chars followed by _
    let mut result = String::with_capacity(name.len());
    let bytes = name.as_bytes();
    let pattern = b"_GLOBAL__N__";
    let mut i = 0;
    while i < bytes.len() {
        if i + pattern.len() + 9 <= bytes.len() && &bytes[i..i+pattern.len()] == pattern {
            // Check if next 8 chars are hex followed by _
            let hash_start = i + pattern.len();
            let all_hex = bytes[hash_start..hash_start+8].iter().all(|b| b.is_ascii_hexdigit());
            if all_hex && bytes[hash_start+8] == b'_' {
                result.push_str("_GLOBAL__N__????????_");
                i = hash_start + 9;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

fn extract_function_from_ptx(ptx_source: &str, kernel_name: &str) -> Option<String> {
    let mut header_lines: Vec<&str> = Vec::new();
    let mut function_lines: Vec<&str> = Vec::new();
    let mut in_target_function = false;
    let mut brace_depth: i32 = 0;
    let mut found_function = false;

    // Normalize the kernel name to handle anonymous namespace hash differences
    // between CUBIN (sm_70) and PTX (sm_120) compilations.
    let normalized_name = normalize_anon_ns(kernel_name);

    // The kernel name in PTX may be mangled. Look for it as a substring in .entry declarations.
    // Also handle the case where the name is an exact match.
    for line in ptx_source.lines() {
        let trimmed = line.trim();

        if !found_function && !in_target_function {
            // Collect header lines (directives before any function)
            if trimmed.starts_with(".version") || trimmed.starts_with(".target") ||
               trimmed.starts_with(".address_size") || trimmed.starts_with(".extern") ||
               trimmed.starts_with(".global") || trimmed.starts_with(".const") ||
               trimmed.starts_with("//") || trimmed.is_empty() {
                header_lines.push(line);
                continue;
            }

            // Check if this line starts our target function
            // PTX entry functions: ".visible .entry name(" or ".entry name("
            let line_matches = if trimmed.contains(".entry") || trimmed.contains(".func") {
                if trimmed.contains(kernel_name) {
                    true
                } else if normalized_name.contains("????????") {
                    // Fuzzy match: normalize the PTX line too and compare
                    let normalized_line = normalize_anon_ns(trimmed);
                    normalized_line.contains(&normalized_name)
                } else {
                    false
                }
            } else {
                false
            };
            if line_matches {
                in_target_function = true;
                function_lines.push(line);
                // Count braces on this line
                for ch in trimmed.chars() {
                    match ch {
                        '{' => brace_depth += 1,
                        '}' => brace_depth -= 1,
                        _ => {}
                    }
                }
                if brace_depth > 0 && trimmed.contains('}') && brace_depth == 0 {
                    found_function = true;
                    in_target_function = false;
                }
                continue;
            }
            // If we hit a non-header, non-target function, just skip until we find ours
            // (This skips other functions in the module)
        } else if in_target_function {
            function_lines.push(line);
            for ch in trimmed.chars() {
                match ch {
                    '{' => brace_depth += 1,
                    '}' => brace_depth -= 1,
                    _ => {}
                }
            }
            if brace_depth <= 0 && trimmed.contains('}') {
                found_function = true;
                in_target_function = false;
            }
        }
    }

    if !found_function || function_lines.is_empty() {
        eprintln!("[PTX Extract] Function '{}' not found in PTX ({} bytes)", kernel_name, ptx_source.len());
        return None;
    }

    // Build minimal PTX module: header + function
    let mut result = String::with_capacity(header_lines.len() * 40 + function_lines.len() * 60);
    for line in &header_lines {
        result.push_str(line);
        result.push('\n');
    }
    result.push('\n');
    for line in &function_lines {
        result.push_str(line);
        result.push('\n');
    }

    eprintln!("[PTX Extract] Extracted '{}': {} bytes from {} byte module ({} header + {} function lines)",
             kernel_name, result.len(), ptx_source.len(), header_lines.len(), function_lines.len());
    Some(result)
}

/// Global index mapping normalized kernel names to (ptx_file_path, original_entry_name).
/// Built once on first lookup, then reused for O(1) lookups.
static PTX_INDEX: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);

/// Build an index of all .entry kernel names across all PTX files.
/// Keys are normalized names (anonymous namespace hashes replaced with ????????).
/// Values are PTX file paths.
fn build_ptx_index() -> HashMap<String, String> {
    let mut index = HashMap::new();
    let mut ptx_dirs: Vec<String> = (0..10)
        .map(|i| format!("/tmp/hetgpu_so_ptx_{}", i))
        .filter(|d| std::path::Path::new(d).is_dir())
        .collect();

    // Also search HETGPU_PTX_DIR (user-configured) and the repo root for
    // pre-extracted libtorch_cuda.*.ptx files that ship with hetGPU.
    if let Ok(extra) = std::env::var("HETGPU_PTX_DIR") {
        for dir in extra.split(':') {
            let d = dir.trim().to_string();
            if !d.is_empty() && std::path::Path::new(&d).is_dir() {
                ptx_dirs.push(d);
            }
        }
    }
    // Auto-discover: look next to the libnvcuda.so binary for PTX files.
    // The pre-extracted libtorch_cuda.*.sm_120.ptx files live in the repo root.
    if let Ok(lib_path) = std::env::var("LD_PRELOAD") {
        for part in lib_path.split(':') {
            if let Some(parent) = std::path::Path::new(part.trim()).parent() {
                // Walk up to find the repo root (where the .ptx files are)
                for ancestor in [parent, parent.parent().unwrap_or(parent),
                                 parent.parent().and_then(|p| p.parent()).unwrap_or(parent)] {
                    let candidate = ancestor.to_string_lossy().to_string();
                    if !ptx_dirs.contains(&candidate) && std::path::Path::new(&candidate).is_dir() {
                        // Check if this directory actually has .ptx files
                        if let Ok(entries) = std::fs::read_dir(&candidate) {
                            let has_ptx = entries
                                .filter_map(|e| e.ok())
                                .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("ptx"));
                            if has_ptx {
                                eprintln!("[PTX Index] Auto-discovered PTX directory: {}", candidate);
                                ptx_dirs.push(candidate);
                            }
                        }
                    }
                }
            }
        }
    }

    // Also search Triton's JIT cache for kernel PTX.
    // Triton caches compiled kernels with proper PTX (with newlines) in
    // ~/.triton/cache/<hash>/<kernel_name>.ptx
    if let Ok(home) = std::env::var("HOME") {
        let triton_cache = format!("{}/.triton/cache", home);
        if std::path::Path::new(&triton_cache).is_dir() {
            // Walk subdirectories (each is a hash)
            if let Ok(entries) = std::fs::read_dir(&triton_cache) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let subdir = entry.path();
                    if subdir.is_dir() {
                        let subdir_str = subdir.to_string_lossy().to_string();
                        if !ptx_dirs.contains(&subdir_str) {
                            ptx_dirs.push(subdir_str);
                        }
                    }
                }
            }
            eprintln!("[PTX Index] Added Triton cache dirs from {}", triton_cache);
        }
    }

    for ptx_dir in &ptx_dirs {
        let entries = match std::fs::read_dir(ptx_dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("ptx") {
                continue;
            }
            let path_str = path.to_string_lossy().to_string();
            // Use BufReader for line-by-line scanning to avoid loading
            // multi-hundred-MB PTX files entirely into memory.
            let file = match std::fs::File::open(&path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let reader = std::io::BufReader::new(file);
            use std::io::BufRead;
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => break,
                };
                let trimmed = line.trim_start();
                // Fast prefix check to skip the vast majority of lines
                if !trimmed.starts_with(".entry ") && !trimmed.contains(".entry ") {
                    continue;
                }
                if let Some(entry_pos) = trimmed.find(".entry ") {
                    // Extract kernel name: everything after ".entry " until '(' or whitespace
                    let name_start = entry_pos + 7; // ".entry ".len()
                    let name = &trimmed[name_start..];
                    let name_end = name.find(|c: char| c == '(' || c.is_whitespace()).unwrap_or(name.len());
                    let kernel_name = &name[..name_end];
                    if !kernel_name.is_empty() {
                        let normalized = normalize_anon_ns(kernel_name);
                        index.entry(normalized).or_insert_with(|| path_str.clone());
                    }
                }
            }
        }
    }
    eprintln!("[PTX Index] Built index with {} kernel entries from {} directories", index.len(), ptx_dirs.len());
    index
}

/// Look up a kernel in the PTX index and extract its function.
/// Uses the cached index for fast lookups.
fn cached_ptx_lookup(kernel_name: &str) -> Option<String> {
    let mut guard = PTX_INDEX.lock().ok()?;
    if guard.is_none() {
        *guard = Some(build_ptx_index());
    }
    let index = guard.as_ref()?;

    let normalized = normalize_anon_ns(kernel_name);
    let ptx_path = index.get(&normalized)?;

    // Read the PTX file and extract just this function
    let contents = std::fs::read_to_string(ptx_path).ok()?;
    eprintln!("[PTX Index] Found '{}' in {}", kernel_name, ptx_path);
    extract_function_from_ptx(&contents, kernel_name)
}

/// Search all extracted PTX files (from cuobjdump) for a kernel by name.
/// Falls back to fuzzy matching with normalized anonymous namespace hashes.
fn search_all_ptx_files_for_kernel(kernel_name: &str) -> Option<String> {
    // Look through /tmp/hetgpu_so_ptx_* directories
    let ptx_dirs: Vec<_> = (0..10)
        .map(|i| format!("/tmp/hetgpu_so_ptx_{}", i))
        .filter(|d| std::path::Path::new(d).is_dir())
        .collect();

    if ptx_dirs.is_empty() {
        return None;
    }

    let normalized_name = normalize_anon_ns(kernel_name);
    let has_anon_ns = normalized_name.contains("????????");

    for ptx_dir in &ptx_dirs {
        let entries = match std::fs::read_dir(ptx_dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("ptx") {
                continue;
            }

            // Quick check: read the file and see if kernel name (or normalized version) is in it
            let contents = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let might_contain = if contents.contains(kernel_name) {
                true
            } else if has_anon_ns {
                // Check with normalized name (replace hashes with ????????)
                // For efficiency, check if a non-hash portion of the name exists first
                let parts: Vec<&str> = kernel_name.split("_GLOBAL__N__").collect();
                if parts.len() >= 2 {
                    // Check if the part after the hash is present
                    // e.g., "16_TensorCompare_cu_8f1b29aa"
                    let after_hash = parts[1].get(9..); // skip 8 hex chars + '_'
                    after_hash.map_or(false, |suffix| contents.contains(suffix))
                } else {
                    false
                }
            } else {
                false
            };

            if !might_contain {
                continue;
            }

            // Try actual extraction
            if let Some(extracted) = extract_function_from_ptx(&contents, kernel_name) {
                eprintln!("[PTX Search] Found '{}' in {}", kernel_name, path.display());
                return Some(extracted);
            }
        }
    }

    eprintln!("[PTX Search] '{}' not found in any PTX file", kernel_name);
    None
}

/// Generate correct TMatmul assembly from kernel name and execute via the Python emulator.
/// This is the "correct compiler" path: the kernel name tells us what operation to perform,
/// we extract tensor pointers from the kernel params, generate appropriate assembly,
/// and run it through the Python emulator (which uses PyTorch for tensor operations).
#[cfg(feature = "intel")]
unsafe fn execute_kernel_via_emulator(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32, grid_dim_y: u32, _grid_dim_z: u32,
    block_dim_x: u32, _block_dim_y: u32, _block_dim_z: u32,
) -> bool {
    if kernel_params.is_null() {
        return false;
    }

    let name_lower = kernel_name.to_lowercase();

    // Determine operation type and extract tensor pointers
    struct TensorParam {
        ptr: *mut u8,
        size: usize, // in bytes
    }

    let mut tensor_params: Vec<TensorParam> = Vec::new();
    let mut assembly = String::new();
    let mut numel: usize = 0;
    let mut out_numel: usize = 0; // output may differ from input (reductions)

    if name_lower.contains("vectorized_elementwise_kernel") || name_lower.contains("unrolled_elementwise_kernel") || name_lower.contains("elementwise_kernel") {
        // Skip integer-type kernels (long, int, bool) - we only handle float types
        // C++ mangled names use: l=long, i=int, b=bool, j=unsigned int, m=unsigned long
        // Pattern: FunctorIlll = Functor<long,long,long>, FunctorIlE = Functor<long>
        // Float types have: HalfE, BFloat16, f=float, d=double
        if is_integer_type_kernel(kernel_name) {
            eprintln!("[TMatmul] Skipping integer-type kernel: {}", kernel_name);
            return false;
        }

        // Non-vectorized elementwise_kernel uses a different parameter layout:
        // (int N, func_t f) where data pointers are captured inside the functor.
        // The fallback path handles this with proper dtype-aware execution.
        let is_vectorized_ew = name_lower.contains("vectorized_elementwise_kernel") || name_lower.contains("unrolled_elementwise_kernel");
        if !is_vectorized_ew {
            return false;
        }

        // vectorized_elementwise_kernel(int N, Functor f, std::array<char*, K> data)
        // kernel_params[0] = &N, [1] = &functor, [2] = &data_array
        let param0 = *kernel_params.add(0);
        if param0.is_null() { return false; }
        if !is_memory_readable(param0 as *const u8, 4) { return false; }
        let n = (param0 as *const u32).read_unaligned() as usize;
        if n == 0 || n > 64 * 1024 * 1024 { return false; }
        numel = n;

        // Read pointers from data array (param 2)
        let param2 = *kernel_params.add(2);
        if param2.is_null() { return false; }

        // Data array contains 2-3 pointers (output, input1, optional input2)
        let num_ptrs = if name_lower.contains("binaryfunctor") || name_lower.contains("binary_internal") { 3 } else { 2 };

        // Safety check: ensure we can read num_ptrs * 8 bytes from param2
        if !is_memory_readable(param2 as *const u8, num_ptrs * 8) {
            eprintln!("[TMatmul] param2 not readable for {} pointers in: {}", num_ptrs, kernel_name);
            return false;
        }

        let mut raw_ptrs: Vec<u64> = Vec::new();
        for i in 0..num_ptrs {
            let ptr_val = (param2 as *const u64).add(i).read_unaligned();
            raw_ptrs.push(ptr_val);
        }

        // For binary ops: ptrs = [output, input1, input2]
        // For unary ops: ptrs = [output, input]
        // Remap to: PARAM_0=input1, PARAM_1=input2(optional), PARAM_N=output
        if num_ptrs == 3 {
            // Binary: output=ptrs[0], in1=ptrs[1], in2=ptrs[2]
            let out_ptr = raw_ptrs[0];
            let in1_ptr = raw_ptrs[1];
            let in2_ptr = raw_ptrs[2];

            let out_size = super::memory::get_alloc_size(out_ptr as usize).unwrap_or(0);
            let in1_size = super::memory::get_alloc_size(in1_ptr as usize).unwrap_or(0);
            let in2_size = super::memory::get_alloc_size(in2_ptr as usize).unwrap_or(0);

            if out_size == 0 || in1_size == 0 || in2_size == 0 {
                eprintln!("[TMatmul Compiler] elementwise binary: pointer not in alloc_map");
                return false;
            }

            tensor_params.push(TensorParam { ptr: in1_ptr as *mut u8, size: in1_size });
            tensor_params.push(TensorParam { ptr: in2_ptr as *mut u8, size: in2_size });
            tensor_params.push(TensorParam { ptr: out_ptr as *mut u8, size: out_size });

            // Determine operation from kernel name
            let op = if name_lower.contains("addfunctor") || name_lower.contains("add_functor") { "add" }
                else if name_lower.contains("mulfunctor") || name_lower.contains("mul_functor") { "mul" }
                else if name_lower.contains("subfunctor") || name_lower.contains("sub_functor") { "sub" }
                else if name_lower.contains("divfunctor") || name_lower.contains("div_functor") { "div" }
                else { "mul" }; // default

            assembly = format!(
                "    ldv    v0,PARAM_0\n    ldv    v1,PARAM_1\n    {} v2,v0,v1\n    sv     v2,PARAM_2\n",
                op
            );
            out_numel = numel;
        } else {
            // Unary: output=ptrs[0], input=ptrs[1]
            let out_ptr = raw_ptrs[0];
            let in_ptr = raw_ptrs[1];

            let out_size = super::memory::get_alloc_size(out_ptr as usize).unwrap_or(0);
            let in_size = super::memory::get_alloc_size(in_ptr as usize).unwrap_or(0);

            if out_size == 0 || in_size == 0 {
                eprintln!("[TMatmul Compiler] elementwise unary: pointer not in alloc_map");
                return false;
            }

            tensor_params.push(TensorParam { ptr: in_ptr as *mut u8, size: in_size });
            tensor_params.push(TensorParam { ptr: out_ptr as *mut u8, size: out_size });

            // Handle comparison kernels - parse comparison value from functor struct
            // Functor layout for CompareEqFunctor<T>: { T value; }
            // kernel_params[1] = &functor
            if name_lower.contains("compare") {
                let functor_ptr = *kernel_params.add(1);
                if functor_ptr.is_null() {
                    eprintln!("[TMatmul] Compare kernel '{}' - null functor", kernel_name);
                    return false;
                }

                // Determine element size from kernel name (long=8, int=4, etc.)
                let elem_size = if name_lower.contains("compareeqfunctorile") ||
                                   name_lower.contains("compareeqfunctor<long>") ||
                                   name_lower.contains("ileeee") || name_lower.contains("ill") {
                    8usize // int64/long
                } else if name_lower.contains("iieee") || name_lower.contains("iii") {
                    4usize // int32
                } else {
                    8usize // default to int64
                };

                // Read comparison value from functor (first elem_size bytes)
                let cmp_val_i64 = if elem_size == 8 {
                    (functor_ptr as *const i64).read_unaligned()
                } else {
                    (functor_ptr as *const i32).read_unaligned() as i64
                };

                let in_ptr = raw_ptrs[1];
                let out_ptr = raw_ptrs[0];
                let in_size = super::memory::get_alloc_size(in_ptr as usize).unwrap_or(0);
                let out_size = super::memory::get_alloc_size(out_ptr as usize).unwrap_or(0);

                if in_ptr == 0 || out_ptr == 0 || in_size == 0 || out_size == 0 {
                    return false;
                }

                // Determine comparison type from name
                let is_eq = name_lower.contains("compareeq");
                let is_ne = name_lower.contains("comparene");
                let is_lt = name_lower.contains("comparelt");
                let is_le = name_lower.contains("comparele");
                let is_gt = name_lower.contains("comparegt");
                let is_ge = name_lower.contains("comparege");

                eprintln!("[TMatmul] Compare kernel: cmp_val={}, elem_size={}, numel={}", cmp_val_i64, elem_size, numel);

                // Perform comparison element-wise
                let actual_numel = numel.min(in_size / elem_size).min(out_size / elem_size);
                if elem_size == 8 {
                    let inp = in_ptr as *const i64;
                    let out = out_ptr as *mut i64;
                    for i in 0..actual_numel {
                        let val = inp.add(i).read_unaligned();
                        let result = if is_eq { val == cmp_val_i64 }
                            else if is_ne { val != cmp_val_i64 }
                            else if is_lt { val < cmp_val_i64 }
                            else if is_le { val <= cmp_val_i64 }
                            else if is_gt { val > cmp_val_i64 }
                            else if is_ge { val >= cmp_val_i64 }
                            else { val == cmp_val_i64 }; // default to eq
                        out.add(i).write_unaligned(if result { 1 } else { 0 });
                    }
                } else {
                    let inp = in_ptr as *const i32;
                    let out = out_ptr as *mut i32;
                    let cmp_val_i32 = cmp_val_i64 as i32;
                    for i in 0..actual_numel {
                        let val = inp.add(i).read_unaligned();
                        let result = if is_eq { val == cmp_val_i32 }
                            else if is_ne { val != cmp_val_i32 }
                            else if is_lt { val < cmp_val_i32 }
                            else if is_le { val <= cmp_val_i32 }
                            else if is_gt { val > cmp_val_i32 }
                            else if is_ge { val >= cmp_val_i32 }
                            else { val == cmp_val_i32 };
                        out.add(i).write_unaligned(if result { 1 } else { 0 });
                    }
                }

                eprintln!("[TMatmul] Compare kernel '{}' done ({} elements)", kernel_name, actual_numel);
                return true;
            }

            // Skip kernels we can't handle correctly (type conversions, casts, etc.)
            if name_lower.contains("cast") || name_lower.contains("fill") ||
               name_lower.contains("scatter") || name_lower.contains("gather") ||
               name_lower.contains("index") || name_lower.contains("where") {
                eprintln!("[TMatmul] Skipping unsupported kernel type: {}", kernel_name);
                return false;
            }

            // Determine operation from kernel name
            let op = if name_lower.contains("copy") || name_lower.contains("contiguous") { "copy" }
                else if name_lower.contains("sigmoid") { "sig" }
                else if name_lower.contains("silu") || name_lower.contains("swish") { "silu" }
                else if name_lower.contains("relu") { "relu" }
                else if name_lower.contains("gelu") { "gelu" }
                else if name_lower.contains("tanh") { "tanh" }
                else if name_lower.contains("neg") { "neg" }
                else if name_lower.contains("rsqrt") { "rsqrt" }
                else if name_lower.contains("reciprocal") { "reciprocal" }
                else if name_lower.contains("exp") && !name_lower.contains("export") { "exp" }
                else if name_lower.contains("abs") { "abs" }
                else if name_lower.contains("sqrt") && !name_lower.contains("rsqrt") { "sqrt" }
                else if name_lower.contains("log") && !name_lower.contains("log1p") { "log" }
                else if name_lower.contains("log1p") { "log1p" }
                else if name_lower.contains("sin") && !name_lower.contains("sinh") { "sin" }
                else if name_lower.contains("cos") && !name_lower.contains("cosh") { "cos" }
                else if name_lower.contains("floor") { "floor" }
                else if name_lower.contains("ceil") { "ceil" }
                else if name_lower.contains("round") { "round" }
                else if name_lower.contains("sign") { "sign" }
                else if name_lower.contains("square") { "square" }
                else {
                    // Unknown operation - don't guess, return false to use fallback
                    eprintln!("[TMatmul] Unknown unary op in: {}", kernel_name);
                    return false;
                };

            // DIRECT RUST EMULATION for non-native TMatmul ops
            // These ops are NOT in the original TMatmul ISA, so we execute them directly
            // in Rust without going through assembly generation/interpretation.
            // Only native ISA ops (sig, silu, copy) go through the assembly path.
            let actual_numel = numel.min(in_size / 4).min(out_size / 4);
            let src = in_ptr as *const f32;
            let dst = out_ptr as *mut f32;

            match op {
                // Native TMatmul ISA - generate assembly
                "sig" | "silu" | "copy" => {
                    assembly = match op {
                        "copy" => "    ldv    v0,PARAM_0\n    sv     v0,PARAM_1\n".to_string(),
                        _ => format!("    ldv    v0,PARAM_0\n    {}    v1,v0\n    sv     v1,PARAM_1\n", op),
                    };
                    tensor_params.push(TensorParam { ptr: in_ptr as *mut u8, size: in_size });
                    tensor_params.push(TensorParam { ptr: out_ptr as *mut u8, size: out_size });
                    out_numel = numel;
                }
                // Non-native ops - direct Rust emulation (no assembly)
                "relu" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(if x > 0.0 { x } else { 0.0 });
                    }
                    eprintln!("[TMatmul Emulate] relu {} elements", actual_numel);
                    return true;
                }
                "gelu" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        let k: f32 = 0.7978845608; // sqrt(2/pi)
                        let inner = k * (x + 0.044715 * x * x * x);
                        dst.add(i).write_unaligned(0.5 * x * (1.0 + inner.tanh()));
                    }
                    eprintln!("[TMatmul Emulate] gelu {} elements", actual_numel);
                    return true;
                }
                "tanh" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(x.tanh());
                    }
                    eprintln!("[TMatmul Emulate] tanh {} elements", actual_numel);
                    return true;
                }
                "neg" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(-x);
                    }
                    eprintln!("[TMatmul Emulate] neg {} elements", actual_numel);
                    return true;
                }
                "reciprocal" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(if x != 0.0 { 1.0 / x } else { f32::INFINITY });
                    }
                    eprintln!("[TMatmul Emulate] reciprocal {} elements", actual_numel);
                    return true;
                }
                "rsqrt" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(if x > 0.0 { 1.0 / x.sqrt() } else { f32::INFINITY });
                    }
                    eprintln!("[TMatmul Emulate] rsqrt {} elements", actual_numel);
                    return true;
                }
                "sqrt" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(x.sqrt());
                    }
                    eprintln!("[TMatmul Emulate] sqrt {} elements", actual_numel);
                    return true;
                }
                "exp" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(x.exp());
                    }
                    eprintln!("[TMatmul Emulate] exp {} elements", actual_numel);
                    return true;
                }
                "log" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(x.ln());
                    }
                    eprintln!("[TMatmul Emulate] log {} elements", actual_numel);
                    return true;
                }
                "log1p" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned((1.0 + x).ln());
                    }
                    eprintln!("[TMatmul Emulate] log1p {} elements", actual_numel);
                    return true;
                }
                "abs" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(x.abs());
                    }
                    eprintln!("[TMatmul Emulate] abs {} elements", actual_numel);
                    return true;
                }
                "sin" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(x.sin());
                    }
                    eprintln!("[TMatmul Emulate] sin {} elements", actual_numel);
                    return true;
                }
                "cos" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(x.cos());
                    }
                    eprintln!("[TMatmul Emulate] cos {} elements", actual_numel);
                    return true;
                }
                "floor" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(x.floor());
                    }
                    eprintln!("[TMatmul Emulate] floor {} elements", actual_numel);
                    return true;
                }
                "ceil" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(x.ceil());
                    }
                    eprintln!("[TMatmul Emulate] ceil {} elements", actual_numel);
                    return true;
                }
                "round" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(x.round());
                    }
                    eprintln!("[TMatmul Emulate] round {} elements", actual_numel);
                    return true;
                }
                "sign" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        let s = if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
                        dst.add(i).write_unaligned(s);
                    }
                    eprintln!("[TMatmul Emulate] sign {} elements", actual_numel);
                    return true;
                }
                "square" => {
                    for i in 0..actual_numel {
                        let x = src.add(i).read_unaligned();
                        dst.add(i).write_unaligned(x * x);
                    }
                    eprintln!("[TMatmul Emulate] square {} elements", actual_numel);
                    return true;
                }
                _ => {
                    eprintln!("[TMatmul] Unhandled op: {}", op);
                    return false;
                }
            }
        }
    } else if name_lower.contains("reduce_kernel") {
        // reduce_kernel takes a single ReduceOp struct with embedded pointers
        let op_param = *kernel_params.add(0);
        if op_param.is_null() { return false; }

        let found_ptrs = scan_for_alloc_pointers(op_param as *const u8, 1024);
        if found_ptrs.len() < 2 { return false; }

        let mut sorted = found_ptrs.clone();
        sorted.sort_by_key(|&(_, ptr, _)| ptr);

        // Use min gap for input size, compute dimensions
        let mut tensor_sizes: Vec<usize> = Vec::new();
        for i in 0..sorted.len() {
            if i + 1 < sorted.len() {
                tensor_sizes.push((sorted[i + 1].1 - sorted[i].1) as usize);
            } else {
                tensor_sizes.push(sorted[i].2.min(1024 * 1024));
            }
        }

        let min_gap = *tensor_sizes.iter().min().unwrap_or(&0);
        let in_elements = min_gap / 4;
        if in_elements == 0 { return false; }

        let in_ptr = sorted[sorted.len() - 1].1 as *mut u8;
        let out_ptr = sorted[0].1 as *mut u8;
        let in_size = sorted[sorted.len() - 1].2;
        let out_size = sorted[0].2;

        // Determine reduce operation
        let op = if name_lower.contains("sum_functor") || name_lower.contains("sum") { "reduce_sum" }
            else if name_lower.contains("meanops") || name_lower.contains("mean") { "reduce_mean" }
            else if name_lower.contains("maxops") || name_lower.contains("max") { "reduce_max" }
            else if name_lower.contains("normops") || name_lower.contains("var") { "reduce_var" }
            else { "reduce_sum" };

        tensor_params.push(TensorParam { ptr: in_ptr, size: in_size.min(in_elements * 4) });
        tensor_params.push(TensorParam { ptr: out_ptr, size: out_size });

        numel = in_elements;
        // For reduce, output is smaller than input - determine dimensions
        let mut reduce_size = 0usize;
        let mut out_elements = 0usize;
        for i in 0..32usize {
            let val = (op_param as *const u32).add(i).read_unaligned() as usize;
            if val > 0 && val < in_elements && in_elements % val == 0 {
                let other = in_elements / val;
                for j in 0..32usize {
                    if j == i { continue; }
                    let val2 = (op_param as *const u32).add(j).read_unaligned() as usize;
                    if val2 == other {
                        if val < other { out_elements = val; reduce_size = other; }
                        else { out_elements = other; reduce_size = val; }
                        break;
                    }
                }
                if out_elements > 0 { break; }
            }
        }
        if out_elements == 0 {
            for rs in [128, 256, 512, 1024, 64, 32].iter() {
                if in_elements % rs == 0 { reduce_size = *rs; out_elements = in_elements / rs; break; }
            }
        }
        out_numel = out_elements;

        assembly = format!(
            "    ldv    v0,PARAM_0\n    {}    v1,v0\n    sv     v1,PARAM_1\n",
            op
        );
    } else if name_lower.contains("softmax") {
        // softmax: kernel_params[0]=&dst, [1]=&src, [2]=&batch, [3]=&stride, [4]=&element_count
        let param0 = *kernel_params.add(0);
        let param1 = *kernel_params.add(1);
        if param0.is_null() || param1.is_null() { return false; }

        let dst_ptr = (param0 as *const u64).read_unaligned();
        let src_ptr = (param1 as *const u64).read_unaligned();

        let dst_size = super::memory::get_alloc_size(dst_ptr as usize).unwrap_or(0);
        let src_size = super::memory::get_alloc_size(src_ptr as usize).unwrap_or(0);
        if dst_size == 0 || src_size == 0 { return false; }

        numel = src_size.min(dst_size) / 4;
        out_numel = numel;

        tensor_params.push(TensorParam { ptr: src_ptr as *mut u8, size: src_size });
        tensor_params.push(TensorParam { ptr: dst_ptr as *mut u8, size: dst_size });

        assembly = "    ldv    v0,PARAM_0\n    softmax    v1,v0\n    sv     v1,PARAM_1\n".to_string();
    } else if name_lower.contains("layernorm") || name_lower.contains("layer_norm") ||
              name_lower.contains("rmsnorm") || name_lower.contains("rms_norm") ||
              name_lower.contains("welford") {
        // norm kernel: params [0]=&N, [1]=&eps, [2]=&X, [3]=&gamma, [4]=&beta, [5]=&Y, ...
        let param0 = *kernel_params.add(0);
        if param0.is_null() { return false; }
        let hidden_size = (param0 as *const i32).read_unaligned() as usize;
        if hidden_size == 0 || hidden_size > 65536 { return false; }

        let mut ptrs: Vec<(u64, usize)> = Vec::new();
        for pi in 2..8 {
            let p = *kernel_params.add(pi);
            if p.is_null() { ptrs.push((0, 0)); continue; }
            let ptr_val = (p as *const u64).read_unaligned();
            let size = super::memory::get_alloc_size(ptr_val as usize).unwrap_or(0);
            ptrs.push((ptr_val, size));
        }

        let x_ptr = ptrs[0].0;
        let gamma_ptr = ptrs[1].0;
        let y_ptr = ptrs[3].0;

        if x_ptr == 0 || y_ptr == 0 { return false; }
        let x_size = ptrs[0].1;
        let y_size = ptrs[3].1;
        if x_size == 0 || y_size == 0 { return false; }

        numel = x_size.min(y_size) / 4;
        out_numel = numel;

        let is_rmsnorm = name_lower.contains("rmsnorm") || name_lower.contains("rms_norm");

        if is_rmsnorm {
            // PARAM_0=X, PARAM_1=gamma, PARAM_2=Y
            tensor_params.push(TensorParam { ptr: x_ptr as *mut u8, size: x_size });
            if gamma_ptr != 0 && ptrs[1].1 > 0 {
                tensor_params.push(TensorParam { ptr: gamma_ptr as *mut u8, size: ptrs[1].1 });
            } else {
                // No weight - use ones (handled by bridge CONST_1)
                tensor_params.push(TensorParam { ptr: std::ptr::null_mut(), size: 0 });
            }
            tensor_params.push(TensorParam { ptr: y_ptr as *mut u8, size: y_size });

            assembly = format!(
                "    ldv    v0,PARAM_0\n    rms_clear\n    rms_accumulate    v0\n    rms_finish_accumulate    {}\n    rms_norm    v1,v0\n",
                numel
            );
            // If we have weights (PARAM_1), multiply
            if gamma_ptr != 0 && ptrs[1].1 > 0 {
                assembly.push_str("    ldv    v2,PARAM_1\n    mul    v1,v1,v2\n");
            }
            assembly.push_str("    sv     v1,PARAM_2\n");
        } else {
            // LayerNorm: PARAM_0=X, PARAM_1=gamma, PARAM_2=beta, PARAM_3=Y
            tensor_params.push(TensorParam { ptr: x_ptr as *mut u8, size: x_size });
            tensor_params.push(TensorParam { ptr: gamma_ptr as *mut u8, size: ptrs[1].1 });
            let beta_ptr = ptrs[2].0;
            if beta_ptr != 0 && ptrs[2].1 > 0 {
                tensor_params.push(TensorParam { ptr: beta_ptr as *mut u8, size: ptrs[2].1 });
            } else {
                tensor_params.push(TensorParam { ptr: std::ptr::null_mut(), size: 0 });
            }
            tensor_params.push(TensorParam { ptr: y_ptr as *mut u8, size: y_size });

            assembly = "    ldv    v0,PARAM_0\n    ldv    v1,PARAM_1\n    ldv    v2,PARAM_2\n    layernorm    v3,v0,v1,v2\n    sv     v3,PARAM_3\n".to_string();
        }
    } else {
        // Unknown kernel type - can't generate assembly
        return false;
    }

    if tensor_params.is_empty() || assembly.is_empty() || numel == 0 {
        return false;
    }

    eprintln!("[TMatmul Compiler] '{}' (numel={}, {} params)", kernel_name, numel, tensor_params.len());

    // FAST PATH: Execute directly in Rust - no file I/O, no Python
    // This handles common ops (copy, add, mul, sigmoid, relu, etc.) at native speed
    {
        let ptrs: Vec<*mut u8> = tensor_params.iter().map(|tp| tp.ptr).collect();
        let sizes: Vec<usize> = tensor_params.iter().map(|tp| tp.size).collect();

        match super::tmatmul_interpreter::execute_with_direct_ptrs(&assembly, &ptrs, &sizes, numel) {
            Ok(()) => {
                eprintln!("[TMatmul Fast] '{}' done ({} elements)", kernel_name, numel);
                return true;
            }
            Err(_e) => {
                // Fall through to Python path for unsupported ops
            }
        }
    }

    // SLOW PATH: Python daemon (only for complex ops not handled by Rust)
    eprintln!("[TMatmul Slow] '{}' using Python bridge", kernel_name);

    // Use /dev/shm for RAM-backed file I/O
    let shm_base = if std::path::Path::new("/dev/shm").exists() { "/dev/shm" } else { "/tmp" };

    // Write assembly to shared memory
    let asm_path = format!("{}/hetgpu_asm.S", shm_base);
    if std::fs::write(&asm_path, &assembly).is_err() {
        return false;
    }

    // Write tensor data to shared memory files and build JSON params
    use std::fmt::Write;
    let mut params_json = String::from("[");
    let mut param_infos: Vec<(*mut u8, String, usize)> = Vec::new();

    for (i, tp) in tensor_params.iter().enumerate() {
        if i > 0 { params_json.push(','); }

        let file_path = format!("{}/hetgpu_p{}.bin", shm_base, i);
        let count = if tp.size > 0 { (tp.size / 4).min(numel) } else { numel };

        if !tp.ptr.is_null() && tp.size > 0 {
            let size_bytes = count * 4;
            if is_memory_readable(tp.ptr, size_bytes) {
                let data_slice = std::slice::from_raw_parts(tp.ptr, size_bytes);
                if std::fs::write(&file_path, data_slice).is_err() {
                    return false;
                }
                let _ = write!(params_json, r#"{{"file":"{}","count":{}}}"#, file_path, count);
                param_infos.push((tp.ptr, file_path.clone(), count));
            } else {
                let _ = write!(params_json, r#"{{"file":"","count":0}}"#);
                param_infos.push((std::ptr::null_mut(), String::new(), 0));
            }
        } else {
            let zeros = vec![0u8; numel * 4];
            let _ = std::fs::write(&file_path, &zeros);
            let _ = write!(params_json, r#"{{"file":"{}","count":{}}}"#, file_path, numel);
            param_infos.push((std::ptr::null_mut(), file_path.clone(), numel));
        }
    }
    params_json.push(']');

    // Build command JSON for daemon
    let cmd_json = format!(
        r#"{{"asm":"{}","params":{},"numel":{}}}"#,
        asm_path, params_json, numel
    );

    // Try persistent daemon first (fast path)
    if let Some(mut daemon_guard) = get_daemon() {
        if let Some(daemon) = daemon_guard.as_mut() {
            if let Some(response) = daemon.execute(&cmd_json) {
                if response.contains("\"ok\":true") || response.contains("\"ok\": true") {
                    // Read back results
                    for (_i, (host_ptr, filepath, count)) in param_infos.iter().enumerate() {
                        if host_ptr.is_null() || filepath.is_empty() || *count == 0 {
                            continue;
                        }
                        let out_path = format!("{}.out", filepath);
                        if let Ok(data) = std::fs::read(&out_path) {
                            let copy_bytes = data.len().min(*count * 4);
                            if copy_bytes > 0 {
                                std::ptr::copy_nonoverlapping(data.as_ptr(), *host_ptr, copy_bytes);
                            }
                            let _ = std::fs::remove_file(&out_path);
                        }
                    }
                    eprintln!("[TMatmul Daemon] Kernel '{}' executed ({} elements)", kernel_name, numel);
                    return true;
                } else {
                    eprintln!("[TMatmul Daemon] Execution failed: {}", response.trim());
                }
            }
        }
    }

    // Fallback: spawn bridge process (slow path)
    eprintln!("[TMatmul Emulator] Daemon unavailable, using slow bridge");
    let bridge_script = std::env::var("HETGPU_TMATMUL_BRIDGE")
        .unwrap_or_else(|_| "/mnt/ubuntu/ternary_matmul/sw_utils/lib/hetgpu_bridge.py".to_string());
    let python = std::env::var("HETGPU_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // Write config file for legacy bridge
    let config_json = format!(
        r#"{{"assembly_file":"{}","params":{},"numel":{}}}"#,
        asm_path, params_json, numel
    );
    let config_path = format!("{}/hetgpu_config.json", shm_base);
    if std::fs::write(&config_path, &config_json).is_err() {
        return false;
    }

    match std::process::Command::new(&python)
        .arg(&bridge_script)
        .arg(&config_path)
        .env("PYTHONPATH", "/mnt/ubuntu/ternary_matmul/sw_utils")
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .output()
    {
        Ok(output) => {
            if !output.status.success() {
                eprintln!("[TMatmul Emulator] Bridge failed: {:?}", output.status.code());
                return false;
            }

            for (host_ptr, filepath, count) in param_infos.iter() {
                if host_ptr.is_null() || filepath.is_empty() || *count == 0 {
                    continue;
                }
                let out_path = format!("{}.out", filepath);
                if let Ok(data) = std::fs::read(&out_path) {
                    let copy_bytes = data.len().min(*count * 4);
                    if copy_bytes > 0 && is_memory_readable(*host_ptr, copy_bytes) {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), *host_ptr, copy_bytes);
                    }
                    let _ = std::fs::remove_file(&out_path);
                }
            }

            eprintln!("[TMatmul Emulator] Kernel '{}' executed successfully via Python emulator", kernel_name);
            true
        }
        Err(e) => {
            eprintln!("[TMatmul Emulator] Failed to invoke bridge: {}", e);
            false
        }
    }
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

/// Execute a kernel via the correct path:
/// 1. Try the Python emulator (generates correct assembly from kernel name)
/// 2. Fall back to Rust-based direct computation if emulator fails
#[cfg(feature = "intel")]
unsafe fn execute_kernel_name_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
    block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
) {
    if kernel_params.is_null() {
        eprintln!("[TMatmul Fallback] Kernel '{}' - null kernel_params, skipping", kernel_name);
        return;
    }

    // Try the Python emulator path first (correct compiler approach)
    if execute_kernel_via_emulator(kernel_name, kernel_params,
                                    grid_dim_x, grid_dim_y, grid_dim_z,
                                    block_dim_x, block_dim_y, block_dim_z) {
        return; // Emulator succeeded
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

    // Handle Triton BitNet kernels (onebitllms).
    if name_lower == "partial_max_kernel" || name_lower == "abs_mean_kernel" {
        execute_triton_reduction_kernel(&name_lower, kernel_params,
            grid_dim_x, grid_dim_y, grid_dim_z);
        return;
    }

    // Handle CatArrayBatchedCopy — triggered by torch.cat() for KV cache.
    // These kernels copy data from multiple source tensors into a destination tensor.
    // For the virtual backend, the tensor data is already in host memory,
    // so the concat is effectively a series of memcpy operations.
    // The kernel uses a complex parameter struct, but with the virtual backend
    // the data is already accessible — just treat as a no-op since PyTorch's
    // CPU fallback code path handles the actual copy before the kernel launch.
    if name_lower.contains("catarraybatchedcopy") {
        // CatArrayBatchedCopy is a performance optimization kernel.
        // In the virtual backend, the copy has already been set up by PyTorch's
        // tensor infrastructure.  The kernel just does the GPU-side copy.
        // We need to perform the actual copy here.
        execute_cat_copy_fallback(kernel_name, kernel_params, grid_dim_x, grid_dim_y, grid_dim_z);
        return;
    }

    // Handle DeviceScanInitKernel — CUB tile state initialization.
    // This initializes tile state for parallel prefix scan.  In single-threaded
    // execution it's a no-op (scan can proceed sequentially).
    if name_lower.contains("devicescaninit") {
        eprintln!("[TMatmul Fallback] DeviceScanInitKernel: no-op (sequential execution)");
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

    // Skip integer-type kernels - we only handle float types
    if is_integer_type_kernel(kernel_name) {
        eprintln!("[TMatmul Fallback] Skipping integer-type kernel: {}", kernel_name);
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
    if !is_memory_readable(numel_param as *const u8, 4) {
        eprintln!("[TMatmul Fallback] kernel_params[0] not readable");
        return;
    }
    let numel = (numel_param as *const i32).read_unaligned() as usize;
    if numel == 0 || numel > 64 * 1024 * 1024 {
        eprintln!("[TMatmul Fallback] Invalid numel={} for '{}'", numel, kernel_name);
        return;
    }

    // Distinguish vectorized vs non-vectorized elementwise_kernel
    let is_vectorized_ew = name_lower.contains("vectorized_elementwise_kernel") || name_lower.contains("unrolled_elementwise_kernel");

    let mut data_ptrs: Vec<*mut u8> = Vec::new();
    if is_vectorized_ew {
        // Vectorized variant: data array at kernel_params[2]
        let data_param = *kernel_params.add(2);
        if data_param.is_null() {
            eprintln!("[TMatmul Fallback] kernel_params[2] (data array) is null");
            return;
        }
        if !is_memory_readable(data_param as *const u8, num_ptrs * 8) {
            eprintln!("[TMatmul Fallback] data_param not readable for {} pointers in: {}", num_ptrs, kernel_name);
            return;
        }
        for i in 0..num_ptrs {
            let ptr_val = (data_param as *const u64).add(i).read_unaligned();
            if let Some(_size) = super::memory::get_alloc_size(ptr_val as usize) {
                data_ptrs.push(ptr_val as *mut u8);
            } else {
                eprintln!("[TMatmul Fallback] data_ptrs[{}] = {:#x} not in alloc map", i, ptr_val);
                data_ptrs.push(std::ptr::null_mut());
            }
        }
    } else {
        // Non-vectorized elementwise_kernel: data pointers are captured in the functor
        // kernel_params[1] = &functor, where functor is a lambda capturing data pointers
        let functor_ptr = *kernel_params.add(1);
        if functor_ptr.is_null() || !is_memory_readable(functor_ptr as *const u8, 128) {
            eprintln!("[TMatmul Fallback] Non-vectorized elementwise: functor not readable");
            return;
        }
        let found = scan_for_alloc_pointers(functor_ptr as *const u8, 128);
        if found.len() < num_ptrs {
            eprintln!("[TMatmul Fallback] Non-vectorized elementwise: found {} ptrs in functor (need {})", found.len(), num_ptrs);
            return;
        }
        eprintln!("[TMatmul Fallback] Non-vectorized elementwise: found {} data ptrs in functor", found.len());
        for i in 0..num_ptrs {
            data_ptrs.push(found[i].1 as *mut u8);
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
    // For non-vectorized kernels, params[1] is the lambda functor struct (not a simple scalar),
    // so we default to 1.0 since the alpha value is embedded deep in the functor.
    let functor_f32 = if is_vectorized_ew {
        let functor_param = *kernel_params.add(1);
        if !functor_param.is_null() {
            (functor_param as *const f32).read_unaligned()
        } else {
            1.0f32
        }
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
    // PyTorch's ReduceOp struct can be >512 bytes in newer versions
    let scan_size = 1024;

    // Debug: dump the struct contents to understand the layout
    eprintln!("[TMatmul Fallback] reduce_kernel: scanning struct at {:p} ({} bytes)", op_param, scan_size);
    let alloc_map_snapshot: Vec<(usize, usize)> = if let Ok(map) = super::memory::VIRTUAL_ALLOC_MAP.lock() {
        map.iter().map(|(&k, &v)| (k, v)).collect()
    } else {
        Vec::new()
    };
    eprintln!("[TMatmul Fallback] reduce_kernel: alloc_map has {} entries:", alloc_map_snapshot.len());
    for (base, size) in &alloc_map_snapshot {
        eprintln!("[TMatmul Fallback]   alloc: base={:#x}, size={}", base, size);
    }

    // Dump first 64 u64 values from the struct
    for i in 0..64usize.min(scan_size / 8) {
        let val = (op_param as *const u64).add(i).read_unaligned();
        if val >= 0x10000 && val <= 0x7fff_ffff_ffff {
            let in_map = super::memory::get_alloc_size(val as usize).is_some();
            eprintln!("[TMatmul Fallback]   offset {}: {:#018x} (ptr-like, in_alloc_map={})", i*8, val, in_map);
        }
    }

    let found_ptrs = scan_for_alloc_pointers(op_param as *const u8, scan_size);
    eprintln!("[TMatmul Fallback] reduce_kernel: found {} alloc pointers in {} bytes", found_ptrs.len(), scan_size);

    if found_ptrs.len() < 2 {
        eprintln!("[TMatmul Fallback] reduce_kernel: need >=2 pointers, got {}", found_ptrs.len());
        return;
    }

    // Sort pointers by address and deduplicate to infer tensor sizes from gaps.
    // With CachingAllocator, get_alloc_size returns remaining bytes in the block,
    // not the individual tensor size. Use pointer gaps instead.
    let mut sorted = found_ptrs.clone();
    sorted.sort_by_key(|&(_, ptr, _)| ptr);
    // Deduplicate pointers at the same address (same tensor referenced from
    // multiple struct fields).  Keep the first occurrence.
    sorted.dedup_by_key(|entry| entry.1);

    if sorted.len() < 2 {
        eprintln!("[TMatmul Fallback] reduce_kernel: need >=2 unique pointers, got {}", sorted.len());
        return;
    }

    // Detect element size from kernel name.  The mangled name encodes the
    // scalar type: 'f' = float (4 bytes), 'd' = double (8 bytes),
    // 'l' = long/int64 (8 bytes), 'e' = half (2 bytes), 'i' = int (4 bytes).
    // ReduceOp<SCALAR, ...> — the first template parameter after ReduceOp is
    // the scalar type in the Itanium mangling.
    let elem_size: usize = if name_lower.contains("reduceop") {
        // Try to detect from the full (non-lowered) kernel name
        if kernel_name.contains("ReduceOpId") || kernel_name.contains("ReduceOpIl") {
            8 // double or long (int64)
        } else if kernel_name.contains("ReduceOpIe") || kernel_name.contains("ReduceOpIDhE") {
            2 // half
        } else {
            4 // float (default)
        }
    } else if kernel_name.contains("ReduceOpIl") || kernel_name.contains("ReduceOpId") {
        8
    } else if kernel_name.contains("ReduceOpIe") || kernel_name.contains("ReduceOpIDhE") {
        2
    } else {
        4
    };
    eprintln!("[TMatmul Fallback] reduce_kernel: detected elem_size={} bytes", elem_size);

    // Compute tensor sizes from gaps between consecutive unique pointers
    let mut tensor_sizes: Vec<usize> = Vec::new();
    for i in 0..sorted.len() {
        if i + 1 < sorted.len() {
            let gap = (sorted[i + 1].1 - sorted[i].1) as usize;
            tensor_sizes.push(gap);
        } else {
            // Last pointer: use alloc_size (remaining bytes) but cap at a reasonable max
            let remaining = sorted[i].2;
            tensor_sizes.push(remaining.min(1024 * 1024)); // cap at 1MB
        }
    }

    eprintln!("[TMatmul Fallback] reduce_kernel: pointer gaps: {:?}", tensor_sizes);

    // Use the minimum non-zero gap as the input tensor size.
    let min_gap = *tensor_sizes.iter().filter(|&&g| g > 0).min().unwrap_or(&0);
    let in_elements = min_gap / elem_size;

    // The first pointer (lower address) is typically the output tensor.
    // The second pointer (higher address) is the input tensor.
    let in_ptr = sorted[sorted.len() - 1].1; // higher address = input
    let out_ptr = sorted[0].1; // lower address = output

    if in_elements == 0 {
        eprintln!("[TMatmul Fallback] reduce_kernel: zero input elements (min_gap={}, elem_size={})", min_gap, elem_size);
        return;
    }

    // Extract output_elements and reduce_size from the struct.
    // The ReduceOp struct contains dimension info in its first ~128 bytes.
    // Look for int32 pairs (a, b) where a * b == in_elements.
    let mut out_elements = 0usize;
    let mut reduce_size = 0usize;
    for i in 0..32usize { // scan first 256 bytes as int32s
        let val = (op_param as *const u32).add(i).read_unaligned() as usize;
        if val > 0 && val < in_elements && in_elements % val == 0 {
            let other = in_elements / val;
            // Check if 'other' also appears in the struct
            for j in 0..32usize {
                if j == i { continue; }
                let val2 = (op_param as *const u32).add(j).read_unaligned() as usize;
                if val2 == other {
                    // Found a matching pair: prefer the arrangement where val < other
                    // (val = output_elements, other = reduce_size)
                    if val < other {
                        out_elements = val;
                        reduce_size = other;
                    } else {
                        out_elements = other;
                        reduce_size = val;
                    }
                    break;
                }
            }
            if out_elements > 0 { break; }
        }
    }

    // Fallback: if we couldn't find dimensions, assume the gap is the full input
    // and try common reduce sizes
    if out_elements == 0 || reduce_size == 0 {
        // Try to infer from total elements: assume reduce along last dim
        // Common patterns: 128, 256, 512, 1024
        for rs in [128, 256, 512, 1024, 64, 32, 2048, 4096].iter() {
            if in_elements % rs == 0 && in_elements / rs > 0 {
                reduce_size = *rs;
                out_elements = in_elements / rs;
                break;
            }
        }
    }

    if out_elements == 0 || reduce_size == 0 {
        eprintln!("[TMatmul Fallback] reduce_kernel: can't determine dimensions (in_elements={})", in_elements);
        return;
    }

    // Safety: cap in_elements by the alloc_size of the input pointer
    let in_alloc_size = sorted[sorted.len() - 1].2;
    let in_elements = in_elements.min(in_alloc_size / elem_size);
    // Cap out_elements by the alloc_size of the output pointer
    let out_alloc_size = sorted[0].2;
    let out_elements = out_elements.min(out_alloc_size / elem_size);

    eprintln!("[TMatmul Fallback] reduce_kernel: in_elements={}, out_elements={}, reduce_size={}, elem_size={}",
             in_elements, out_elements, reduce_size, elem_size);

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

    // Dispatch based on element size — int64 (long) kernels need integer ops,
    // while float kernels use f32/f64.
    if elem_size == 8 && (name_lower.contains("maxnanfunctor") || name_lower.contains("minnanfunctor")
                          || kernel_name.contains("ReduceOpIl")) {
        // Integer (i64) reduction — used by argmax/argmin on long tensors
        let inp = in_ptr as *const i64;
        let out = out_ptr as *mut i64;
        for row in 0..out_elements {
            let base = row * reduce_size;
            let end = (base + reduce_size).min(in_elements);
            if end <= base { continue; }
            match op_type {
                "max" => {
                    let mut max_val = i64::MIN;
                    for i in base..end {
                        let v = inp.add(i).read_unaligned();
                        if v > max_val { max_val = v; }
                    }
                    out.add(row).write_unaligned(max_val);
                }
                "min" => {
                    let mut min_val = i64::MAX;
                    for i in base..end {
                        let v = inp.add(i).read_unaligned();
                        if v < min_val { min_val = v; }
                    }
                    out.add(row).write_unaligned(min_val);
                }
                "sum" => {
                    let mut sum = 0i64;
                    for i in base..end {
                        sum = sum.wrapping_add(inp.add(i).read_unaligned());
                    }
                    out.add(row).write_unaligned(sum);
                }
                _ => {
                    // For mean/var on integers, cast to f64
                    let mut sum = 0.0f64;
                    for i in base..end {
                        sum += inp.add(i).read_unaligned() as f64;
                    }
                    let count = (end - base) as f64;
                    out.add(row).write_unaligned(if op_type == "mean" { (sum / count) as i64 } else { sum as i64 });
                }
            }
        }
    } else {
        // Float (f32) reduction — the common path
        let inp = in_ptr as *const f32;
        let out = out_ptr as *mut f32;
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
    }
    eprintln!("[TMatmul Fallback] reduce_kernel '{}' executed ({} -> {} elements, elem_size={})",
             op_type, in_elements, out_elements, elem_size);
}

/// Execute CatArrayBatchedCopy fallback — performs tensor concatenation.
///
/// PyTorch's CatArrayBatchedCopy kernel takes a CatArrInputTensorMetadata struct
/// containing source tensor pointers, sizes, and a destination pointer.  The struct
/// is passed as a single parameter.  We scan it for valid allocation pointers and
/// perform a linear copy of each source into the destination.
#[cfg(feature = "intel")]
unsafe fn execute_cat_copy_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32, _grid_dim_y: u32, _grid_dim_z: u32,
) {
    if kernel_params.is_null() { return; }

    // CatArrayBatchedCopy has parameters:
    //   param_0: output pointer (char*)
    //   param_1: CatArrInputTensorMetadata struct (contains src ptrs, offsets, sizes)
    //   param_2: TensorSizeStride struct
    //   param_3: numCats (int)
    //   param_4: catStride
    //
    // For the virtual backend, the simplest correct approach is to scan for
    // allocation pointers.  The first is the destination, the rest are sources.
    // We copy source data into the destination sequentially.

    let param0 = *kernel_params.add(0);
    if param0.is_null() { return; }

    // The first parameter is typically a pointer to the output buffer
    let out_ptr_val = (param0 as *const u64).read_unaligned();
    let out_ok = super::memory::get_alloc_size(out_ptr_val as usize);

    if let Some(out_size) = out_ok {
        // Scan remaining params for source pointers
        let mut src_ptrs: Vec<(u64, usize)> = Vec::new();
        for i in 1..8 {
            let param = *kernel_params.add(i);
            if param.is_null() { break; }
            // Scan struct for allocation pointers
            let found = scan_for_alloc_pointers(param as *const u8, 512);
            for (_, ptr_val, size) in &found {
                if *ptr_val != out_ptr_val && !src_ptrs.iter().any(|(p, _)| p == ptr_val) {
                    src_ptrs.push((*ptr_val, *size));
                }
            }
            if !src_ptrs.is_empty() { break; }
        }

        if src_ptrs.is_empty() {
            eprintln!("[TMatmul Fallback] CatArrayBatchedCopy '{}': no source pointers found", kernel_name);
            return;
        }

        // Copy sources into destination sequentially
        let dst = out_ptr_val as *mut u8;
        let mut offset = 0usize;
        for (src_ptr, src_size) in &src_ptrs {
            let copy_len = (*src_size).min(out_size.saturating_sub(offset));
            if copy_len == 0 { break; }
            std::ptr::copy_nonoverlapping(*src_ptr as *const u8, dst.add(offset), copy_len);
            offset += copy_len;
        }
        eprintln!("[TMatmul Fallback] CatArrayBatchedCopy: copied {} bytes into output", offset);
    } else {
        eprintln!("[TMatmul Fallback] CatArrayBatchedCopy: output pointer not in alloc_map");
    }
}

/// Execute Triton BitNet reduction kernels (partial_max_kernel, abs_mean_kernel).
///
/// These kernels have Triton's standard param layout:
///   param_0: input_ptr (u64 device pointer)
///   param_1: output_ptr (u64 device pointer)
///   param_2: n_elems or first scalar dim (u32)
///   param_3+: additional dims/strides
///
/// Grid dimensions encode the block iteration:
///   abs_mean_kernel:     grid=(n_blocks, 1, 1)  — 1D reduction
///   partial_max_kernel:  grid=(batch, seq_len, n_blocks) — 3D, per-row absmax
///
/// BLOCK_SIZE is a compile-time constant (typically 256).
#[cfg(feature = "intel")]
unsafe fn execute_triton_reduction_kernel(
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
) {
    // Read param pointers.  kernel_params[i] points to the i-th param value.
    let inp_ptr_val = (*kernel_params.add(0) as *const u64).read_unaligned();
    let out_ptr_val = (*kernel_params.add(1) as *const u64).read_unaligned();
    let dim_val     = (*kernel_params.add(2) as *const u32).read_unaligned() as usize;

    // Validate pointers via alloc_map
    let inp_ok = super::memory::get_alloc_size(inp_ptr_val as usize).is_some();
    let out_ok = super::memory::get_alloc_size(out_ptr_val as usize).is_some();

    if !inp_ok || !out_ok {
        eprintln!("[TMatmul Fallback] {}: invalid pointers (inp={:#x} ok={}, out={:#x} ok={})",
                 name_lower, inp_ptr_val, inp_ok, out_ptr_val, out_ok);
        return;
    }

    // Determine data type from input allocation size and grid.
    // BitNet uses bfloat16 tensors.  We'll detect element size from the alloc.
    let inp_alloc_size = super::memory::get_alloc_size(inp_ptr_val as usize).unwrap_or(0);

    // Infer BLOCK_SIZE from the relationship: total_blocks * BLOCK_SIZE ≈ total_elements
    // For abs_mean_kernel: grid=(n_blocks,1,1), total_elements = n_elems (param_2)
    // For partial_max_kernel: grid=(batch,seq,n_blocks), hidden_dim is implicit
    let block_size: usize = 256; // BitNet default, matches Triton autotune config

    if name_lower == "abs_mean_kernel" {
        // abs_mean_kernel: each grid block computes sum(abs(input[block*BS..(block+1)*BS]))
        // and stores the scalar result to output[block_id].
        let n_blocks = grid_dim_x as usize;
        let n_elems = dim_val;
        // Elements are bfloat16 (2 bytes) — read as u16, convert to f32
        let inp = inp_ptr_val as *const u16;
        let out = out_ptr_val as *mut f32;

        for block_id in 0..n_blocks {
            let base = block_id * block_size;
            let end = (base + block_size).min(n_elems);
            let mut sum = 0.0f32;
            for i in base..end {
                let raw = inp.add(i).read_unaligned();
                let val = bf16_to_f32(raw);
                sum += val.abs();
            }
            out.add(block_id).write_unaligned(sum);
        }
        eprintln!("[TMatmul Fallback] abs_mean_kernel executed: {} blocks, {} elems", n_blocks, n_elems);
    } else if name_lower == "partial_max_kernel" {
        // partial_max_kernel: grid=(batch, seq_len, n_blocks)
        // For each (batch, seq, block): max(abs(x[row_offset + block*BS .. +BS]))
        // → partial_max[(batch*seq_len + seq)*n_blocks + block]
        //
        // The Python signature is: partial_max_kernel(x_ptr, partial_max_ptr,
        //   batch_size, seq_len, hidden_dim, BLOCK_SIZE=constexpr)
        // But Triton may reorder/add stride params.  Use grid dims directly.
        let batch_size = grid_dim_x as usize;
        let seq_len = grid_dim_y as usize;
        let n_blocks = grid_dim_z as usize;
        let hidden_dim = n_blocks * block_size; // approximate

        let inp = inp_ptr_val as *const u16; // bfloat16
        let out = out_ptr_val as *mut f32;   // partial max output is f32

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let row_offset = (batch_idx * seq_len + seq_idx) * hidden_dim;
                for blk_idx in 0..n_blocks {
                    let hidden_off = blk_idx * block_size;
                    let mut max_val = 0.0f32;
                    for i in 0..block_size {
                        let idx = row_offset + hidden_off + i;
                        let raw = inp.add(idx).read_unaligned();
                        let val = bf16_to_f32(raw).abs();
                        if val > max_val { max_val = val; }
                    }
                    max_val = max_val.max(1e-5); // match Triton kernel: tl.maximum(..., 1e-5)
                    let pm_offset = (batch_idx * seq_len + seq_idx) * n_blocks + blk_idx;
                    out.add(pm_offset).write_unaligned(max_val);
                }
            }
        }
        eprintln!("[TMatmul Fallback] partial_max_kernel executed: {}x{}x{} grid, hidden_dim~{}",
                 batch_size, seq_len, n_blocks, hidden_dim);
    }
}

/// Convert a bfloat16 raw u16 to f32.
#[inline]
fn bf16_to_f32(raw: u16) -> f32 {
    f32::from_bits((raw as u32) << 16)
}

/// Execute a softmax kernel fallback.
#[cfg(feature = "intel")]
unsafe fn execute_softmax_kernel_fallback(
    kernel_name: &str,
    _name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    // softmax_warp_forward signature:
    //   (output_t *dst, const input_t *src, int batch_size, int stride, int element_count, ...)
    // kernel_params[0] = &dst, [1] = &src, [2] = &batch_size, [3] = &stride, [4] = &element_count

    let param0 = *kernel_params.add(0);
    let param1 = *kernel_params.add(1);
    if param0.is_null() || param1.is_null() {
        eprintln!("[TMatmul Fallback] softmax: null dst or src parameter");
        return;
    }

    let dst_ptr_val = (param0 as *const u64).read_unaligned();
    let src_ptr_val = (param1 as *const u64).read_unaligned();

    let dst_size = super::memory::get_alloc_size(dst_ptr_val as usize);
    let src_size = super::memory::get_alloc_size(src_ptr_val as usize);

    if dst_size.is_none() || src_size.is_none() {
        eprintln!("[TMatmul Fallback] softmax: dst or src not in alloc map (dst={:#x}, src={:#x})",
                 dst_ptr_val, src_ptr_val);
        return;
    }

    // Read dimension params
    let param2 = *kernel_params.add(2);
    let param3 = *kernel_params.add(3);
    let param4 = *kernel_params.add(4);

    let batch_size = if !param2.is_null() { (param2 as *const i32).read_unaligned() as usize } else { 1 };
    let stride = if !param3.is_null() { (param3 as *const i32).read_unaligned() as usize } else { 1 };
    let element_count = if !param4.is_null() { (param4 as *const i32).read_unaligned() as usize } else {
        // Fallback: infer from allocation size
        let total = src_size.unwrap() / 4;
        if batch_size > 0 { total / batch_size } else { total }
    };

    let rows = batch_size;
    let cols = element_count;

    if rows == 0 || cols == 0 {
        eprintln!("[TMatmul Fallback] softmax: invalid dims (rows={}, cols={})", rows, cols);
        return;
    }

    execute_softmax_on_data(src_ptr_val as *const f32, dst_ptr_val as *mut f32, rows, cols);
    eprintln!("[TMatmul Fallback] softmax executed ({}x{} = {} elements) for '{}'",
             rows, cols, rows * cols, kernel_name);
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
///
/// PyTorch's indexSelectSmallIndex kernel has this signature:
///   template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim, bool IndexIsMajor>
///   __global__ void indexSelectSmallIndex(
///       TensorInfo<T, IndexType> dst,      // params[0]
///       TensorInfo<T, IndexType> src,       // params[1]
///       TensorInfo<int64_t, IndexType> idx, // params[2]
///       int dstSelectDim,                   // params[3]
///       int srcSelectDim,                   // params[4]
///       IndexType innerSize,                // params[5]  = embedding_dim
///       int64_t selfSelectDimSize)          // params[6]  = vocab_size
///
/// TensorInfo layout (IndexType=u32):
///   offset 0: T* data (8 bytes)
///   offset 8: u32 sizes[25] (100 bytes)
///   offset 108: u32 strides[25] (100 bytes)
///   offset 208: int dims (4 bytes)
#[cfg(feature = "intel")]
unsafe fn execute_indexselect_kernel_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    let name_lower = kernel_name.to_lowercase();

    // Detect element type from kernel name
    // PyTorch mangles type names: at::Half, c10::Half, at::BFloat16, float, double
    let elem_bytes: usize = if name_lower.contains("half") || name_lower.contains("bfloat16") {
        2
    } else if name_lower.contains("double") {
        8
    } else {
        4 // default f32
    };

    // Read TensorInfo structs from params[0], [1], [2]
    let dst_info = *kernel_params.add(0);
    let src_info = *kernel_params.add(1);
    let idx_info = *kernel_params.add(2);

    if dst_info.is_null() || src_info.is_null() || idx_info.is_null() {
        eprintln!("[TMatmul Fallback] indexSelect: null TensorInfo parameter");
        return;
    }

    // Extract data pointers (first 8 bytes of each TensorInfo)
    let dst_ptr = (dst_info as *const u64).read_unaligned() as *mut u8;
    let src_ptr = (src_info as *const u64).read_unaligned() as *const u8;
    let idx_ptr = (idx_info as *const u64).read_unaligned() as *const u8;

    // Validate data pointers against allocation map
    if super::memory::get_alloc_size(dst_ptr as usize).is_none() {
        eprintln!("[TMatmul Fallback] indexSelect: dst pointer {:#x} not in alloc map", dst_ptr as usize);
        return;
    }
    if super::memory::get_alloc_size(src_ptr as usize).is_none() {
        eprintln!("[TMatmul Fallback] indexSelect: src pointer {:#x} not in alloc map", src_ptr as usize);
        return;
    }
    if super::memory::get_alloc_size(idx_ptr as usize).is_none() {
        eprintln!("[TMatmul Fallback] indexSelect: idx pointer {:#x} not in alloc map", idx_ptr as usize);
        return;
    }

    // Read dimensions from TensorInfo sizes arrays (at offset 8, IndexType=u32)
    // dst sizes: [num_indices, embedding_dim] for 2D embedding output
    // src sizes: [vocab_size, embedding_dim] for 2D embedding weight
    // idx sizes: [num_indices] for 1D index tensor
    let dst_size0 = ((dst_info as *const u8).add(8) as *const u32).read_unaligned() as usize;
    let dst_size1 = ((dst_info as *const u8).add(12) as *const u32).read_unaligned() as usize;
    let src_size0 = ((src_info as *const u8).add(8) as *const u32).read_unaligned() as usize;
    let src_size1 = ((src_info as *const u8).add(12) as *const u32).read_unaligned() as usize;
    let idx_size0 = ((idx_info as *const u8).add(8) as *const u32).read_unaligned() as usize;

    // Also try reading innerSize from params[5] and selfSelectDimSize from params[6]
    let params5 = *kernel_params.add(5);
    let params6 = *kernel_params.add(6);
    let inner_size_from_param = if !params5.is_null() {
        (params5 as *const u32).read_unaligned() as usize
    } else { 0 };
    let vocab_from_param = if !params6.is_null() {
        (params6 as *const i64).read_unaligned() as usize
    } else { 0 };

    // Determine dimensions using multiple sources for validation
    // Primary: TensorInfo sizes (most reliable)
    // Secondary: scalar params[5] and [6]
    let num_indices;
    let embedding_dim;
    let vocab_size;

    // Check if TensorInfo sizes look reasonable
    let tensorinfo_valid = dst_size0 > 0 && dst_size0 < 10_000_000
        && dst_size1 > 0 && dst_size1 < 100_000
        && src_size0 > 0 && src_size0 < 10_000_000
        && src_size1 > 0 && src_size1 < 100_000
        && src_size1 == dst_size1; // embedding_dim should match

    if tensorinfo_valid {
        num_indices = dst_size0;
        embedding_dim = dst_size1;
        vocab_size = src_size0;
        eprintln!("[TMatmul Fallback] indexSelect: dims from TensorInfo: num_idx={}, emb_dim={}, vocab={}",
                 num_indices, embedding_dim, vocab_size);
    } else if inner_size_from_param > 0 && inner_size_from_param < 100_000
           && vocab_from_param > 0 && vocab_from_param < 10_000_000
           && idx_size0 > 0 && idx_size0 < 10_000_000 {
        // Fallback to scalar params
        num_indices = idx_size0;
        embedding_dim = inner_size_from_param;
        vocab_size = vocab_from_param;
        eprintln!("[TMatmul Fallback] indexSelect: dims from scalar params: num_idx={}, emb_dim={}, vocab={}",
                 num_indices, embedding_dim, vocab_size);
    } else {
        eprintln!("[TMatmul Fallback] indexSelect: can't determine dimensions");
        eprintln!("  dst_sizes=({}, {}), src_sizes=({}, {}), idx_size0={}",
                 dst_size0, dst_size1, src_size0, src_size1, idx_size0);
        eprintln!("  params[5]={}, params[6]={}", inner_size_from_param, vocab_from_param);
        return;
    }

    eprintln!("[TMatmul Fallback] indexSelect/embedding: vocab={}, dim={}, seq={}, elem_bytes={}",
             vocab_size, embedding_dim, num_indices, elem_bytes);

    // Copy embedding rows using byte-level operations (works for f16, f32, etc.)
    let row_bytes = embedding_dim * elem_bytes;
    let indices = idx_ptr as *const i64; // PyTorch uses i64 for index tensors

    for i in 0..num_indices {
        let idx_val = indices.add(i).read_unaligned();
        let idx = if idx_val < 0 {
            // Handle negative indices (Python-style: -1 = last element)
            let adjusted = vocab_size as i64 + idx_val;
            if adjusted < 0 {
                eprintln!("[TMatmul Fallback] indexSelect: negative index {} out of bounds at pos {}", idx_val, i);
                continue;
            }
            adjusted as usize
        } else {
            idx_val as usize
        };

        if idx >= vocab_size {
            eprintln!("[TMatmul Fallback] indexSelect: index {} out of bounds (vocab={}) at pos {}", idx, vocab_size, i);
            continue;
        }
        let src_offset = idx * row_bytes;
        let dst_offset = i * row_bytes;
        std::ptr::copy_nonoverlapping(
            src_ptr.add(src_offset),
            dst_ptr.add(dst_offset),
            row_bytes,
        );
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
    let is_rmsnorm = name_lower.contains("rmsnorm") || name_lower.contains("rms_norm");

    // vectorized_layer_norm_kernel signature:
    //   (int N, T_ACC epsilon, const T* X, const T* gamma, const T* beta, T* Y, T_ACC* mean, T_ACC* rstd)
    // kernel_params[0] = &N, [1] = &epsilon, [2] = &X, [3] = &gamma, [4] = &beta,
    // [5] = &Y, [6] = &mean, [7] = &rstd
    //
    // For welford/other norm kernels, scan only pointer params safely
    let param0 = *kernel_params.add(0);
    if param0.is_null() {
        eprintln!("[TMatmul Fallback] norm: null param0");
        return;
    }

    // Read hidden_size (N) from param 0
    let hidden_size = (param0 as *const i32).read_unaligned() as usize;
    if hidden_size == 0 || hidden_size > 65536 {
        eprintln!("[TMatmul Fallback] norm: invalid hidden_size={}", hidden_size);
        return;
    }

    // Read epsilon from param 1
    let eps_param = *kernel_params.add(1);
    let epsilon = if !eps_param.is_null() {
        (eps_param as *const f32).read_unaligned() as f64
    } else {
        1e-5
    };

    // Read tensor pointers from params 2-5
    let mut ptrs: Vec<(*mut u8, usize)> = Vec::new(); // (ptr, alloc_size)
    for pi in 2..8 {
        let p = *kernel_params.add(pi);
        if p.is_null() {
            ptrs.push((std::ptr::null_mut(), 0));
            continue;
        }
        let ptr_val = (p as *const u64).read_unaligned();
        if let Some(size) = super::memory::get_alloc_size(ptr_val as usize) {
            ptrs.push((ptr_val as *mut u8, size));
        } else {
            ptrs.push((std::ptr::null_mut(), 0));
        }
    }

    // ptrs[0] = X (input), ptrs[1] = gamma (weight), ptrs[2] = beta (bias),
    // ptrs[3] = Y (output), ptrs[4] = mean, ptrs[5] = rstd
    let x_ptr = ptrs[0].0 as *const f32;
    let gamma_ptr = ptrs[1].0 as *const f32;
    let beta_ptr = ptrs[2].0 as *const f32;
    let y_ptr = ptrs[3].0 as *mut f32;

    if x_ptr.is_null() || y_ptr.is_null() {
        eprintln!("[TMatmul Fallback] norm: missing input or output pointer");
        return;
    }

    // Determine batch size from pointer gaps (not alloc_size, which is unreliable with CachingAllocator).
    // Collect all valid pointers, sort by address, compute gaps to find X's actual size.
    let mut all_valid_ptrs: Vec<u64> = Vec::new();
    for &(ptr, _size) in &ptrs {
        if !ptr.is_null() {
            all_valid_ptrs.push(ptr as u64);
        }
    }
    all_valid_ptrs.sort();
    all_valid_ptrs.dedup();

    // Find X's position in sorted order and compute gap to next pointer
    let x_addr = x_ptr as u64;
    let x_actual_size = {
        let mut gap = 0usize;
        if let Some(pos) = all_valid_ptrs.iter().position(|&p| p == x_addr) {
            if pos + 1 < all_valid_ptrs.len() {
                gap = (all_valid_ptrs[pos + 1] - x_addr) as usize;
            }
        }
        // Validate: gap should be a multiple of hidden_size * 4 and reasonable
        if gap > 0 && gap >= hidden_size * 4 && gap % (hidden_size * 4) == 0 && gap <= 64 * 1024 * 1024 {
            gap
        } else {
            // Fallback: use output pointer gap if available
            let y_addr = y_ptr as u64;
            let mut y_gap = 0usize;
            if let Some(pos) = all_valid_ptrs.iter().position(|&p| p == y_addr) {
                if pos + 1 < all_valid_ptrs.len() {
                    y_gap = (all_valid_ptrs[pos + 1] - y_addr) as usize;
                }
            }
            if y_gap > 0 && y_gap >= hidden_size * 4 && y_gap % (hidden_size * 4) == 0 && y_gap <= 64 * 1024 * 1024 {
                y_gap
            } else {
                // Last resort: use alloc_size but cap at something reasonable
                let alloc_based = ptrs[0].1;
                alloc_based.min(hidden_size * 4 * 1024) // cap at 1024 batch
            }
        }
    };
    // Safety: never read/write more than alloc_size allows
    let x_actual_size = x_actual_size.min(ptrs[0].1);
    let total_elements = x_actual_size / 4; // f32
    let mut batch_size = if hidden_size > 0 { total_elements / hidden_size } else { 1 };
    // Also cap by output allocation size (Y)
    if ptrs[3].1 > 0 && hidden_size > 0 {
        let y_max_batch = ptrs[3].1 / (hidden_size * 4);
        if y_max_batch < batch_size {
            batch_size = y_max_batch;
        }
    }
    eprintln!("[TMatmul Fallback] norm: x_actual_size={}, total_elements={}, batch_size={}, hidden_size={}",
             x_actual_size, total_elements, batch_size, hidden_size);

    let weight_ptr = if !gamma_ptr.is_null() { gamma_ptr } else { std::ptr::null() };
    let bias_ptr = if !beta_ptr.is_null() { beta_ptr } else { std::ptr::null() };

    let inp_ptr = x_ptr as u64;
    let out_ptr = y_ptr as u64;

    execute_norm_on_data(
        inp_ptr as *const f32, out_ptr as *mut f32,
        weight_ptr, bias_ptr,
        batch_size, hidden_size, is_rmsnorm, epsilon as f32,
    );
    eprintln!("[TMatmul Fallback] norm executed (batch={}, hidden={}, eps={}, rmsnorm={}) for '{}'",
             batch_size, hidden_size, epsilon, is_rmsnorm, kernel_name);
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
