#[cfg(feature = "intel")]
use super::ze_module;
use super::ZludaObject;
use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
use nvidia_runtime_sys;
#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
use pacc_runtime_sys;
use std::{ffi::CStr, ptr, sync::Arc};
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
use tt_runtime_sys;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;
#[cfg(feature = "amd")]
pub(crate) struct Module {
    base: hipModule_t,
}

#[cfg(feature = "intel")]
pub(crate) struct Module {
    context: ze_context_handle_t,
    device: ze_device_handle_t,
    module: ze_module_handle_t,
    functions: Vec<(String, ze_kernel_handle_t)>,
    // Store PTX source and TMatmul assembly for emulator execution
    // Arc<String> avoids cloning multi-MB PTX for every kernel in the module
    ptx_source: Option<Arc<String>>,
    tmatmul_assembly: Option<String>,
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) struct Module {
    device_id: i32,
    program: Option<tt_runtime_sys::Program>,
    kernels: Vec<(String, tt_runtime_sys::Kernel)>,
}

#[cfg(any(feature = "amd", feature = "intel", feature = "tenstorrent"))]
unsafe impl Send for Module {}
#[cfg(any(feature = "amd", feature = "intel", feature = "tenstorrent"))]
unsafe impl Sync for Module {}
#[cfg(feature = "amd")]
impl ZludaObject for Module {
    const COOKIE: usize = 0xe9138bd040487d4a;

    type CudaHandle = CUmodule;

    fn drop_checked(&mut self) -> CUresult {
        unsafe { hipModuleUnload(self.base).unwrap() };
        Ok(())
    }
}

// CUDA: cuModuleGetLoadingMode
// Report a safe default loading mode so callers don't crash.
#[cfg(any(
    feature = "amd",
    feature = "intel",
    feature = "tenstorrent",
    feature = "tmatmul",
    feature = "nvidia",
    feature = "pacc"
))]
pub(crate) fn get_loading_mode(mode: *mut cuda_types::cuda::CUmoduleLoadingMode) -> CUresult {
    if mode.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    unsafe {
        *mode = cuda_types::cuda::CUmoduleLoadingMode::CU_MODULE_EAGER_LOADING;
    }
    Ok(())
}

#[cfg(feature = "intel")]
impl ZludaObject for Module {
    const COOKIE: usize = 0xe9138bd040487d4a;

    type CudaHandle = CUmodule;

    fn drop_checked(&mut self) -> CUresult {
        // Clean up all kernels first
        for (_, kernel) in &self.functions {
            unsafe {
                if !kernel.0.is_null() {
                    zeKernelDestroy(*kernel);
                }
            }
        }
        self.functions.clear();

        // Destroy the module (skip if null for virtual/cocotb fallback)
        if !self.module.0.is_null() {
            let result = unsafe { zeModuleDestroy(self.module) };
            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return ze_to_cuda_result(result);
            }
        }

        Ok(())
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl ZludaObject for Module {
    const COOKIE: usize = 0xe9138bd040487d4a;

    type CudaHandle = CUmodule;

    fn drop_checked(&mut self) -> CUresult {
        // Clean up kernels (they will be dropped automatically)
        self.kernels.clear();

        // Clean up program (it will be dropped automatically)
        self.program = None;

        Ok(())
    }
}

#[cfg(feature = "amd")]
pub(crate) fn load_data(module: &mut CUmodule, image: *const std::ffi::c_void) -> CUresult {
    let text = unsafe { CStr::from_ptr(image.cast()) }
        .to_str()
        .map_err(|_| CUerror::INVALID_VALUE)?;

    // Register PTX source for checkpoint/restore BEFORE compilation
    // This ensures PTX is available even if compilation fails
    let temp_handle = image as u64;
    crate::r#impl::checkpoint::register_module_ptx(temp_handle, text);
    crate::r#impl::hetgpu_debug!(
        "[AMD Backend] Pre-registered PTX source for checkpointing ({} bytes)",
        text.len()
    );

    // Use the new debug-aware compilation pipeline for SASS to PTX mapping
    crate::r#impl::hetgpu_debug!(
        "ZLUDA DEBUG: Starting PTX to LLVM to PTX compilation for SASS mapping..."
    );
    match ptx::ptx_to_llvm_to_ptx_with_sass_mapping(text) {
        Ok((llvm_module, reconstructed_ptx, sass_mapping)) => {
            // Log the SASS to PTX mapping for debugging
            crate::r#impl::hetgpu_debug!(
                "ZLUDA DEBUG: Generated SASS to PTX mapping with {} entries",
                sass_mapping.len()
            );
            crate::r#impl::hetgpu_debug!(
                "ZLUDA DEBUG: Reconstructed PTX length: {} bytes",
                reconstructed_ptx.len()
            );

            // SASS to PTX mapping registry removed for simplicity

            // Continue with normal compilation
            let mut dev = 0;
            unsafe { hipCtxGetDevice(&mut dev).unwrap() };
            let mut props = unsafe { std::mem::zeroed() };
            unsafe { hipGetDeviceProperties(&mut props, dev).unwrap() };
            let elf_module = comgr::compile_bitcode(
                unsafe { CStr::from_ptr(props.gcnArchName.as_ptr()) },
                &*llvm_module.llvm_ir,
                llvm_module.linked_bitcode(),
            )
            .map_err(|_| CUerror::UNKNOWN)?;
            let mut hip_module = unsafe { std::mem::zeroed() };
            unsafe { hipModuleLoadData(&mut hip_module, elf_module.as_ptr().cast()).unwrap() };
            *module = Module { base: hip_module }.wrap();

            // Re-register with actual module handle
            crate::r#impl::checkpoint::register_module_ptx(module.0 as u64, text);
            Ok(())
        }
        Err(_) => {
            // Fallback to original compilation if debug compilation fails
            let ast =
                ptx_parser::parse_module_checked(text).map_err(|_| CUerror::NO_BINARY_FOR_GPU)?;
            let llvm_module = ptx::to_llvm_module(ast).map_err(|_| CUerror::UNKNOWN)?;
            let mut dev = 0;
            unsafe { hipCtxGetDevice(&mut dev).unwrap() };
            let mut props = unsafe { std::mem::zeroed() };
            unsafe { hipGetDeviceProperties(&mut props, dev).unwrap() };
            let elf_module = comgr::compile_bitcode(
                unsafe { CStr::from_ptr(props.gcnArchName.as_ptr()) },
                &*llvm_module.llvm_ir,
                llvm_module.linked_bitcode(),
            )
            .map_err(|_| CUerror::UNKNOWN)?;
            let mut hip_module = unsafe { std::mem::zeroed() };
            unsafe { hipModuleLoadData(&mut hip_module, elf_module.as_ptr().cast()).unwrap() };
            *module = Module { base: hip_module }.wrap();

            // Re-register with actual module handle
            crate::r#impl::checkpoint::register_module_ptx(module.0 as u64, text);
            Ok(())
        }
    }
}

#[cfg(feature = "intel")]
pub(crate) fn load_data(module: &mut CUmodule, image: *const std::ffi::c_void) -> CUresult {
    crate::r#impl::hetgpu_debug!(
        "[Intel Backend] cuModuleLoadData called from PyTorch/application"
    );

    if image.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    // Detect if this is binary CUBIN or PTX text
    let first_bytes = unsafe { std::slice::from_raw_parts(image as *const u8, 4096.min(4096)) };

    crate::r#impl::hetgpu_debug!(
        "[Intel Backend] First 32 bytes: {:02x?}",
        &first_bytes[..32.min(first_bytes.len())]
    );

    // Check for binary formats (ELF, CUDA fatbin, gzip-compressed, etc.)
    let is_binary = if first_bytes.len() >= 4 {
        // ELF magic: 0x7f 'E' 'L' 'F'
        (first_bytes[0] == 0x7f && first_bytes[1] == b'E' && first_bytes[2] == b'L' && first_bytes[3] == b'F') ||
        // CUDA fatbin magic: 0x50ed55ba
        (first_bytes[0] == 0x50 && first_bytes[1] == 0xed && first_bytes[2] == 0x55 && first_bytes[3] == 0xba) ||
        // Gzip magic (Triton uses this): 0x1f 0x8b
        (first_bytes[0] == 0x1f && first_bytes[1] == 0x8b) ||
        // Check first 16 bytes for non-ASCII/non-printable (excluding valid control chars)
        first_bytes.iter().take(16).filter(|&&b| {
            // Count bytes that are clearly binary (not ASCII printable, not common whitespace/control)
            b > 127 || (b < 32 && b != b'\n' && b != b'\r' && b != b'\t' && b != 0)
        }).count() > 4 // If more than 4 suspicious bytes in first 16, it's binary
    } else {
        false
    };

    if is_binary {
        // This is binary CUBIN - pass it through to Level Zero
        crate::r#impl::hetgpu_debug!(
            "[Intel Backend] Detected binary CUBIN, passing to Level Zero..."
        );

        // For binary modules, we need to use Level Zero's native module loading
        // which can handle pre-compiled binaries
        let (context, device) = get_current_context_and_device().unwrap_or((
            ze_context_handle_t(ptr::null_mut()),
            ze_device_handle_t(ptr::null_mut()),
        ));

        // Get the size of the binary using ELF header info or scanning
        let binary_size = if first_bytes.len() >= 64 && first_bytes[4] == 2 {
            // 64-bit ELF: size = max(e_shoff + e_shnum * e_shentsize, e_phoff + e_phnum * e_phentsize)
            let e_shoff = u64::from_le_bytes([
                first_bytes[40],
                first_bytes[41],
                first_bytes[42],
                first_bytes[43],
                first_bytes[44],
                first_bytes[45],
                first_bytes[46],
                first_bytes[47],
            ]) as usize;
            let e_shentsize = u16::from_le_bytes([first_bytes[58], first_bytes[59]]) as usize;
            let e_shnum = u16::from_le_bytes([first_bytes[60], first_bytes[61]]) as usize;
            let elf_end = e_shoff + e_shnum * e_shentsize;
            if elf_end > 0 && elf_end < 100 * 1024 * 1024 {
                eprintln!(
                    "[Intel Backend] ELF binary size from headers: {} bytes",
                    elf_end
                );
                elf_end
            } else {
                // Fallback: scan for end (less reliable)
                let mut size = 0usize;
                unsafe {
                    let ptr = image as *const u8;
                    while size < 10 * 1024 * 1024 {
                        if *ptr.add(size) == 0 && size > 0 && *ptr.add(size - 1) == 0 {
                            break;
                        }
                        size += 1;
                    }
                }
                size
            }
        } else {
            // Non-ELF binary: scan for double null
            let mut size = 0usize;
            unsafe {
                let ptr = image as *const u8;
                while size < 10 * 1024 * 1024 {
                    if *ptr.add(size) == 0 && size > 0 && *ptr.add(size - 1) == 0 {
                        break;
                    }
                    size += 1;
                }
            }
            size
        };

        // Create module descriptor for binary
        let module_desc = ze_module_desc_t {
            stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MODULE_DESC,
            pNext: ptr::null(),
            format: ze_module_format_t::ZE_MODULE_FORMAT_NATIVE, // Native binary format
            inputSize: binary_size,
            pInputModule: image as *const u8,
            pBuildFlags: ptr::null(),
            pConstants: ptr::null(),
        };

        let mut ze_module = ze_module_handle_t(ptr::null_mut());
        let mut build_log = ptr::null_mut();

        let result = unsafe {
            zeModuleCreate(
                context,
                device,
                &module_desc,
                &mut ze_module,
                &mut build_log,
            )
        };

        if !build_log.is_null() {
            unsafe { zeModuleBuildLogDestroy(build_log) };
        }

        if result != ze_result_t::ZE_RESULT_SUCCESS || context.0.is_null() || device.0.is_null() {
            eprintln!("[Intel Backend] Binary module load failed or virtual device detected ({:?}); attempting PTX extraction (binary_size={})", result, binary_size);

            // Use the full binary for PTX extraction, not just first_bytes
            let full_binary = if binary_size > 0 && binary_size <= 10 * 1024 * 1024 {
                unsafe { std::slice::from_raw_parts(image as *const u8, binary_size) }
            } else {
                // Try ELF-based size detection for 64-bit ELF
                let elf_size = if first_bytes.len() >= 64 && first_bytes[4] == 2 {
                    // 64-bit ELF: section header table end = e_shoff + e_shnum * e_shentsize
                    let e_shoff = u64::from_le_bytes([
                        first_bytes[40],
                        first_bytes[41],
                        first_bytes[42],
                        first_bytes[43],
                        first_bytes[44],
                        first_bytes[45],
                        first_bytes[46],
                        first_bytes[47],
                    ]) as usize;
                    let e_shentsize =
                        u16::from_le_bytes([first_bytes[58], first_bytes[59]]) as usize;
                    let e_shnum = u16::from_le_bytes([first_bytes[60], first_bytes[61]]) as usize;
                    let end = e_shoff + e_shnum * e_shentsize;
                    if end > 0 && end < 100 * 1024 * 1024 {
                        end
                    } else {
                        0
                    }
                } else {
                    0
                };
                if elf_size > 4096 {
                    eprintln!("[Intel Backend] ELF size detected: {} bytes", elf_size);
                    unsafe { std::slice::from_raw_parts(image as *const u8, elf_size) }
                } else {
                    first_bytes
                }
            };

            // Try to extract PTX from CUBIN using full binary
            let ptx_source = try_extract_ptx_from_cubin(full_binary);

            if let Some(ref ptx) = ptx_source {
                eprintln!(
                    "[Intel Backend] Successfully extracted {} bytes of PTX from CUBIN",
                    ptx.len()
                );
                eprintln!(
                    "[Intel Backend] PTX preview: {}...",
                    &ptx[..ptx.len().min(200)]
                );
            } else {
                eprintln!("[Intel Backend] No PTX found in CUBIN - operations will be no-ops");
            }

            // Create placeholder module so downstream symbol lookups and launches are no-ops
            let new_module = Module {
                context,
                device,
                module: ze_module_handle_t(ptr::null_mut()),
                functions: Vec::new(),
                ptx_source: ptx_source.map(Arc::new),
                tmatmul_assembly: None,
            };
            *module = new_module.wrap();
            // Ensure a virtual context exists for subsequent cuCtxSynchronize calls
            ensure_virtual_context(context, device);
            return Ok(());
        }

        // Create and return the Module object
        let new_module = Module {
            context,
            device,
            module: ze_module,
            functions: Vec::new(),
            ptx_source: None,
            tmatmul_assembly: None,
        };
        *module = new_module.wrap();
        ensure_virtual_context(context, device);
        return Ok(());
    }

    // Parse as PTX text
    let text = unsafe { CStr::from_ptr(image.cast()) }
        .to_str()
        .map_err(|_| CUerror::INVALID_VALUE)?;

    // If tmatmul emulation is requested or we are on virtual device, compile PTX for emulator
    let use_cocotb = std::env::var("HETGPU_TMATMUL_COCOTB")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);
    let (ctx_handle, dev_handle) = get_current_context_and_device().unwrap_or((
        ze_context_handle_t(std::ptr::null_mut()),
        ze_device_handle_t(std::ptr::null_mut()),
    ));
    let is_virtual = ctx_handle.0.is_null() || dev_handle.0.is_null();

    if use_cocotb || is_virtual {
        crate::r#impl::hetgpu_debug!(
            "[TMatmul Backend] TMatmul emulation enabled (virtual device or HETGPU_TMATMUL_COCOTB=1)"
        );
        match ptx::pass::ptx_to_tmatmul_assembly(text) {
            Ok(tmatmul_asm) => {
                // Write to /tmp for inspection
                let asm_path = std::env::temp_dir().join("tmatmul_kernel.S");
                if let Err(e) = std::fs::write(&asm_path, &tmatmul_asm) {
                    crate::r#impl::hetgpu_debug!(
                        "[TMatmul Backend] Failed to write /tmp/tmatmul_kernel.S: {}",
                        e
                    );
                } else {
                    crate::r#impl::hetgpu_debug!(
                        "[TMatmul Backend] Assembly saved to: {}",
                        asm_path.display()
                    );
                }

                // Optionally copy into hardware simulator asm dir
                let hw_asm_dir = std::env::var("HETGPU_TMATMUL_ASM_DIR")
                    .unwrap_or_else(|_| "/mnt/ubuntu/ternary_matmul/asm".to_string());
                let hw_asm_out = std::path::Path::new(&hw_asm_dir).join("hetgpu_kernel.S");
                if let Err(e) = (|| -> Result<(), std::io::Error> {
                    std::fs::create_dir_all(&hw_asm_dir)?;
                    std::fs::write(&hw_asm_out, &tmatmul_asm)?;
                    Ok(())
                })() {
                    crate::r#impl::hetgpu_debug!(
                        "[TMatmul Backend] Warning: could not write {}: {}",
                        hw_asm_out.display(),
                        e
                    );
                } else {
                    crate::r#impl::hetgpu_debug!(
                        "[TMatmul Backend] Assembly staged for emulator at: {}",
                        hw_asm_out.display()
                    );
                }

                // Create a placeholder module with null ze handles so later calls succeed gracefully
                // Store PTX and TMatmul assembly for execution during kernel launch
                let (context, device) = (ctx_handle, dev_handle);
                let new_module = Module {
                    context,
                    device,
                    module: ze_module_handle_t(std::ptr::null_mut()),
                    functions: Vec::new(),
                    ptx_source: Some(Arc::new(text.to_string())),
                    tmatmul_assembly: Some(tmatmul_asm),
                };
                *module = new_module.wrap();
                // Register PTX source for checkpoint/restore
                crate::r#impl::checkpoint::register_module_ptx(module.0 as u64, text);
                ensure_virtual_context(ctx_handle, dev_handle);

                crate::r#impl::hetgpu_debug!(
                    "[TMatmul Backend] TMatmul assembly ready for emulator at /tmp/tmatmul_kernel.S"
                );

                return Ok(());
            }
            Err(e) => {
                crate::r#impl::hetgpu_debug!("[TMatmul Backend] Compilation error: {}", e);
                // Try a sanitized PTX pass to strip/normalize unsupported statements for parser
                let sanitized = sanitize_ptx_for_virtual(text);
                match ptx::pass::ptx_to_tmatmul_assembly(&sanitized) {
                    Ok(tmatmul_asm) => {
                        crate::r#impl::hetgpu_debug!(
                            "[TMatmul Backend] Retrying with sanitized PTX succeeded"
                        );
                        let asm_path = std::env::temp_dir().join("tmatmul_kernel.S");
                        let _ = std::fs::write(&asm_path, &tmatmul_asm);
                        let hw_asm_dir = std::env::var("HETGPU_TMATMUL_ASM_DIR")
                            .unwrap_or_else(|_| "/mnt/ubuntu/ternary_matmul/asm".to_string());
                        let hw_asm_out = std::path::Path::new(&hw_asm_dir).join("hetgpu_kernel.S");
                        let _ = (|| -> Result<(), std::io::Error> {
                            std::fs::create_dir_all(&hw_asm_dir)?;
                            std::fs::write(&hw_asm_out, &tmatmul_asm)?;
                            Ok(())
                        })();

                        let (context, device) = (ctx_handle, dev_handle);
                        let new_module = Module {
                            context,
                            device,
                            module: ze_module_handle_t(std::ptr::null_mut()),
                            functions: Vec::new(),
                            ptx_source: Some(Arc::new(sanitized.clone())),
                            tmatmul_assembly: Some(tmatmul_asm),
                        };
                        *module = new_module.wrap();
                        // Register PTX source for checkpoint/restore
                        crate::r#impl::checkpoint::register_module_ptx(module.0 as u64, &sanitized);
                        ensure_virtual_context(ctx_handle, dev_handle);
                        return Ok(());
                    }
                    Err(_) => {
                        crate::r#impl::hetgpu_debug!("[TMatmul Backend] Sanitized PTX still failed; using placeholder module");
                        let (context, device) = (ctx_handle, dev_handle);
                        let new_module = Module {
                            context,
                            device,
                            module: ze_module_handle_t(std::ptr::null_mut()),
                            functions: Vec::new(),
                            ptx_source: Some(Arc::new(text.to_string())),
                            tmatmul_assembly: None,
                        };
                        *module = new_module.wrap();
                        // Register PTX source for checkpoint/restore
                        crate::r#impl::checkpoint::register_module_ptx(module.0 as u64, text);
                        ensure_virtual_context(ctx_handle, dev_handle);
                        return Ok(());
                    }
                }
            }
        }
    }

    // Try the new debug-aware compilation pipeline first
    match ptx::ptx_to_llvm_to_ptx_with_sass_mapping(text) {
        Ok((llvm_module, reconstructed_ptx, sass_mapping)) => {
            // Log the SASS to PTX mapping for debugging
            crate::r#impl::hetgpu_debug!(
                "ZLUDA DEBUG: Intel backend - Generated SASS to PTX mapping with {} entries",
                sass_mapping.len()
            );

            // SASS to PTX mapping registry removed for simplicity

            // Create SPIRV module from the LLVM output
            let spirv_module =
                ze_module::SpirvModule::new(text).map_err(|_| CUerror::NO_BINARY_FOR_GPU)?;
            match load_data_impl(module, spirv_module) {
                Ok(()) => CUresult::SUCCESS,
                Err(e) => Err(e),
            }
        }
        Err(_) => {
            // Fallback to original compilation
            let spirv_module =
                ze_module::SpirvModule::new(text).map_err(|_| CUerror::NO_BINARY_FOR_GPU)?;
            match load_data_impl(module, spirv_module) {
                Ok(()) => CUresult::SUCCESS,
                Err(e) => Err(e),
            }
        }
    }
}

#[cfg(feature = "intel")]
fn ensure_virtual_context(ctx: ze_context_handle_t, dev: ze_device_handle_t) {
    // If we already have a non-null context/device, nothing to do
    if !ctx.0.is_null() && !dev.0.is_null() {
        return;
    }
    // Check whether there is any current context; if not, create a placeholder and push
    let has_ctx = super::context::peek_current().is_some();
    if has_ctx {
        return;
    }
    crate::r#impl::hetgpu_debug!("[Intel Backend] Installing virtual context for synchronization");
    // Create a minimal Level Zero context (handles may still be null in virtual)
    let placeholder = super::context::Context::new(ze_device_handle_t(std::ptr::null_mut()));
    let cu_ctx = placeholder.wrap();
    super::context::push(cu_ctx, dev);
}

// Best-effort sanitizer to make Triton-generated PTX palatable to our parser in virtual mode.
// Rewrites or comments out newer instructions and odd syntax that the simplified parser rejects.
fn sanitize_ptx_for_virtual(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let mut cur = line.to_string();
        {
            let t = cur.trim_start();
            if t.is_empty() {
                out.push_str(&cur);
                out.push('\n');
                continue;
            }
        }
        // Replace unsupported f16x2 conversion with a simple move to keep parser happy
        {
            let t_owned = cur.trim_start().to_string();
            if t_owned.starts_with("cvt.rn.f16x2.f32") {
                // Best-effort: extract dst, src0
                let rest = &t_owned["cvt.rn.f16x2.f32".len()..];
                let toks: Vec<&str> = rest
                    .split(|c| c == ',' || c == ';')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();
                if toks.len() >= 2 {
                    let dst = toks[0];
                    let src0 = toks[1];
                    cur = format!("    mov.b32 {}, {};", dst, src0);
                } else {
                    cur = "    // stripped unsupported cvt.rn.f16x2.f32".to_string();
                }
            }
        }
        // Remove braces around register operands: { %r39 } -> %r39
        if cur.contains('{') && cur.contains('}') {
            cur = cur.replace('{', "").replace('}', "");
        }
        // Simplify "+ 0" in addressing forms
        if cur.contains(" + 0") {
            cur = cur.replace(" + 0", "");
        }
        out.push_str(&cur);
        out.push('\n');
    }
    out
}

#[cfg(feature = "intel")]
pub(crate) fn load_data_impl(
    module: &mut CUmodule,
    spirv_module: ze_module::SpirvModule,
) -> Result<(), CUerror> {
    // Get current context and device
    let (context, device) = get_current_context_and_device()?;

    // Convert PTX to SPIRV - for Intel we need to convert PTX to SPIR-V format
    let spirv_binary = ptx_to_spirv(&spirv_module)?;

    // Create module descriptor
    let module_desc = ze_module_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MODULE_DESC,
        pNext: ptr::null(),
        format: ze_module_format_t::ZE_MODULE_FORMAT_IL_SPIRV,
        inputSize: spirv_binary.len(),
        pInputModule: spirv_binary.as_ptr(),
        pBuildFlags: ptr::null(),
        pConstants: ptr::null(),
    };

    // Create module
    let mut ze_module = ze_module_handle_t(ptr::null_mut());
    let mut build_log = ptr::null_mut();

    let result = unsafe {
        zeModuleCreate(
            context,
            device,
            &module_desc,
            &mut ze_module,
            &mut build_log,
        )
    };

    // Check if build log exists and handle it
    if !build_log.is_null() {
        // In a real implementation, you would process the build log
        unsafe { zeModuleBuildLogDestroy(build_log) };
    }

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return Err(CUerror::UNKNOWN);
    }

    // Create and return the Module object
    // Use ze_module implementation for Intel
    super::ze_module::load_data_impl(module, spirv_module)?;
    Ok(())
}

#[cfg(feature = "intel")]
fn ptx_to_spirv(spirv_module: &ze_module::SpirvModule) -> Result<Vec<u8>, CUerror> {
    // Parse PTX
    let ast = ptx_parser::parse_module_checked(&spirv_module.ptx_text)
        .map_err(|_| CUerror::INVALID_VALUE)?;

    // Convert PTX AST to LLVM IR with default attributes
    let attributes = ptx::Attributes {
        clock_rate: 2124000, // Default clock rate in kHz
        emit_debug_info: false,
    };
    let llvm_module = ptx::to_llvm_module(ast, attributes, |_| {}).map_err(|_| CUerror::UNKNOWN)?;

    // Get LLVM IR string from module
    let llvm_ir = llvm_module.llvm_ir.print_module_to_string();

    // Use the robust SPIRV conversion (stub implementation)
    let spirv_binary = ptx::llvm_to_spirv_robust(llvm_ir.to_str()).map_err(|_| CUerror::UNKNOWN)?;

    Ok(spirv_binary)
}

#[cfg(feature = "intel")]
fn get_current_context_and_device() -> Result<(ze_context_handle_t, ze_device_handle_t), CUerror> {
    // Get the current thread-local context and device
    let current_ctx = super::context::CONTEXT_STACK
        .with(|stack| {
            let stack = stack.borrow();
            stack.last().map(|(ctx, dev)| (*ctx, *dev))
        })
        .ok_or(CUerror::INVALID_CONTEXT)?;

    // Get the ZeContext from the CUcontext
    let context = super::context::get_current_ze()?;

    // Return context and device handles
    Ok((context.context, context.device))
}

#[cfg(any(feature = "amd", feature = "intel"))]
pub(crate) fn unload(hmod: CUmodule) -> CUresult {
    super::drop_checked::<Module>(hmod)
}

#[cfg(feature = "amd")]
pub(crate) fn get_function(
    hfunc: &mut hipFunction_t,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> hipError_t {
    unsafe { hipModuleGetFunction(hfunc, hmod.base, name) }
}

#[cfg(feature = "intel")]
pub(crate) fn get_function(
    hfunc: &mut CUfunction,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> CUresult {
    let name_str = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|_| CUerror::INVALID_VALUE)?;

    // If virtual device or tmatmul emulation mode, return a placeholder kernel
    let use_tmatmul = std::env::var("HETGPU_TMATMUL_COCOTB")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);
    if use_tmatmul || hmod.module.0.is_null() {
        eprintln!(
            "[Intel Backend] Creating placeholder kernel '{}' for tmatmul emulation",
            name_str
        );
        let kernel_wrapper = ZeKernel {
            context: hmod.context,
            device: hmod.device,
            module: hmod.module,
            kernel: ze_kernel_handle_t(std::ptr::null_mut()),
            name: name_str.to_string(),
            ptx_source: hmod.ptx_source.clone(), // Pass PTX to kernel for emulation
            module_handle: hmod.module.0 as u64,
        };
        if let Some(ref ptx) = kernel_wrapper.ptx_source {
            eprintln!(
                "[Intel Backend] Kernel has {} bytes of PTX available",
                ptx.len()
            );
        }
        *hfunc = kernel_wrapper.wrap();
        return CUresult::SUCCESS;
    }

    // Check if kernel already exists
    if let Some((_, kernel)) = hmod.functions.iter().find(|(n, _)| n == name_str) {
        *hfunc = ZeKernel {
            context: hmod.context,
            device: hmod.device,
            module: hmod.module,
            kernel: *kernel,
            name: name_str.to_string(),
            ptx_source: hmod.ptx_source.clone(),
            module_handle: hmod.module.0 as u64,
        }
        .wrap();
        return CUresult::SUCCESS;
    }

    // Create new kernel
    let mut kernel = ze_kernel_handle_t(ptr::null_mut());
    let kernel_desc = ze_kernel_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_KERNEL_DESC,
        pNext: ptr::null(),
        flags: 0,
        pKernelName: name,
    };

    let result = unsafe { zeKernelCreate(hmod.module, &kernel_desc, &mut kernel) };

    match result {
        ze_result_t::ZE_RESULT_SUCCESS => {
            let kernel_wrapper = ZeKernel {
                context: hmod.context,
                device: hmod.device,
                module: hmod.module,
                kernel,
                name: name_str.to_string(),
                ptx_source: hmod.ptx_source.clone(),
                module_handle: hmod.module.0 as u64,
            };

            // Store the kernel in the module's function list
            let module_mut = hmod as *const Module as *mut Module;
            unsafe {
                (*module_mut).functions.push((name_str.to_string(), kernel));
            }

            *hfunc = kernel_wrapper.wrap();
            CUresult::SUCCESS
        }
        ze_result_t::ZE_RESULT_ERROR_INVALID_KERNEL_NAME => CUresult::ERROR_INVALID_IMAGE,
        _ => CUresult::ERROR_INVALID_VALUE,
    }
}

#[cfg(feature = "intel")]
pub(crate) struct ZeKernel {
    pub context: ze_context_handle_t,
    pub device: ze_device_handle_t,
    pub module: ze_module_handle_t,
    pub kernel: ze_kernel_handle_t,
    pub name: String,
    pub ptx_source: Option<Arc<String>>, // Shared PTX - avoids cloning per kernel
    pub module_handle: u64,              // Handle for checkpoint tracking
}
#[cfg(feature = "intel")]
unsafe impl Send for ZeKernel {}
#[cfg(feature = "intel")]
unsafe impl Sync for ZeKernel {}
#[cfg(feature = "intel")]
impl ZludaObject for ZeKernel {
    const COOKIE: usize = 0xad74ceadb9b2d51c;

    type CudaHandle = CUfunction;

    fn drop_checked(&mut self) -> CUresult {
        let result = unsafe { zeKernelDestroy(self.kernel) };
        if result != ze_result_t::ZE_RESULT_SUCCESS {
            return ze_to_cuda_result(result);
        }
        Ok(())
    }
}

#[cfg(feature = "intel")]
fn ze_to_cuda_result(result: ze_result_t) -> CUresult {
    match result {
        ze_result_t::ZE_RESULT_SUCCESS => CUresult::SUCCESS,
        ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
        | ze_result_t::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY => CUresult::ERROR_OUT_OF_MEMORY,
        ze_result_t::ZE_RESULT_ERROR_DEVICE_LOST => CUresult::ERROR_NO_DEVICE,
        ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_HANDLE => CUresult::ERROR_INVALID_HANDLE,
        ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_POINTER => CUresult::ERROR_INVALID_VALUE,
        ze_result_t::ZE_RESULT_ERROR_UNINITIALIZED => CUresult::ERROR_NOT_INITIALIZED,
        _ => CUresult::ERROR_UNKNOWN,
    }
}

// Tenstorrent module implementations
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn load_data(module: &mut CUmodule, image: *const std::ffi::c_void) -> CUresult {
    if image.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    // Try to extract PTX source for checkpointing
    let image_bytes = unsafe { std::slice::from_raw_parts(image as *const u8, 8) };
    let is_ptx = image_bytes.starts_with(b".version") || image_bytes.starts_with(b"//");

    if is_ptx {
        // Register PTX source for checkpoint/restore
        let c_str = unsafe { std::ffi::CStr::from_ptr(image as *const std::ffi::c_char) };
        if let Ok(ptx_text) = c_str.to_str() {
            crate::r#impl::checkpoint::register_module_ptx(image as u64, ptx_text);
        }
    }

    // Create a new Tenstorrent module
    let new_module = Module {
        device_id: 0, // Default device
        program: None,
        kernels: Vec::new(),
    };

    let module_box = Box::new(new_module);
    let module_ptr = Box::into_raw(module_box);
    *module = CUmodule(module_ptr as *mut _);

    // Re-register with actual module handle
    if is_ptx {
        let c_str = unsafe { std::ffi::CStr::from_ptr(image as *const std::ffi::c_char) };
        if let Ok(ptx_text) = c_str.to_str() {
            crate::r#impl::checkpoint::register_module_ptx(module.0 as u64, ptx_text);
        }
    }

    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn unload(hmod: CUmodule) -> CUresult {
    if hmod.0.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    // Convert back to box and drop
    let module_ptr = hmod.0 as *mut Module;
    unsafe {
        let _module_box = Box::from_raw(module_ptr);
        // Module will be dropped and cleaned up automatically
    }

    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_function(
    hfunc: *mut CUfunction,
    hmod: CUmodule,
    name: *const ::core::ffi::c_char,
) -> CUresult {
    if hfunc.is_null() || hmod.0.is_null() || name.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let function_name = unsafe {
        std::ffi::CStr::from_ptr(name)
            .to_str()
            .map_err(|_| CUerror::INVALID_VALUE)?
    };

    // For Tenstorrent, create a placeholder function handle
    // In a real implementation, this would look up the kernel in the program
    let tt_kernel = TtKernel {
        device_id: 0,
        program_id: 0,
        kernel_name: function_name.to_string(),
    };

    let kernel_box = Box::new(tt_kernel);
    let kernel_ptr = Box::into_raw(kernel_box);

    unsafe { *hfunc = CUfunction(kernel_ptr as *mut _) };
    Ok(())
}

// Tenstorrent kernel structure
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) struct TtKernel {
    device_id: i32,
    program_id: usize,
    kernel_name: String,
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
unsafe impl Send for TtKernel {}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
unsafe impl Sync for TtKernel {}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl ZludaObject for TtKernel {
    const COOKIE: usize = 0xad74ceadb9b2d51c;

    type CudaHandle = CUfunction;

    fn drop_checked(&mut self) -> CUresult {
        // Clean up Tenstorrent kernel
        // In a real implementation, this would free kernel resources
        Ok(())
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> super::FromCuda<'a, CUfunction> for &'a TtKernel {
    fn from_cuda(handle: &'a CUfunction) -> Result<Self, CUerror> {
        super::as_ref::<TtKernel>(handle).as_result()
    }
}

// TMatmul module implementations
#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) struct Module {
    assembly_code: String,
    kernels: Vec<(String, String)>, // (kernel_name, assembly_code)
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe impl Send for Module {}
#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe impl Sync for Module {}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl ZludaObject for Module {
    const COOKIE: usize = 0xe9138bd040487d4a;

    type CudaHandle = CUmodule;

    fn drop_checked(&mut self) -> CUresult {
        // Clean up TMatmul module resources
        self.kernels.clear();
        Ok(())
    }
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn load_data(module: &mut CUmodule, image: *const std::ffi::c_void) -> CUresult {
    if image.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    // Parse PTX text
    let text = unsafe { CStr::from_ptr(image.cast()) }
        .to_str()
        .map_err(|_| CUerror::INVALID_VALUE)?;

    // Register PTX source for checkpoint/restore BEFORE compilation
    let temp_handle = image as u64;
    crate::r#impl::checkpoint::register_module_ptx(temp_handle, text);
    crate::r#impl::hetgpu_debug!(
        "[TMatmul Backend] Pre-registered PTX source for checkpointing ({} bytes)",
        text.len()
    );

    crate::r#impl::hetgpu_debug!("[TMatmul Backend] Compiling PTX to TMatmul assembly...");

    // Compile PTX to TMatmul assembly
    let tmatmul_asm = ptx::pass::ptx_to_tmatmul_assembly(text).map_err(|e| {
        crate::r#impl::hetgpu_debug!("[TMatmul Backend] Compilation error: {}", e);
        CUerror::NO_BINARY_FOR_GPU
    })?;

    crate::r#impl::hetgpu_debug!("[TMatmul Backend] Successfully compiled to TMatmul assembly");
    crate::r#impl::hetgpu_debug!("[TMatmul Backend] Assembly:\n{}", tmatmul_asm);

    // Save assembly to file for hardware execution
    let asm_path = std::env::temp_dir().join("tmatmul_kernel.S");
    std::fs::write(&asm_path, &tmatmul_asm).map_err(|e| {
        crate::r#impl::hetgpu_debug!("[TMatmul Backend] Failed to write assembly: {}", e);
        CUerror::UNKNOWN
    })?;

    crate::r#impl::hetgpu_debug!(
        "[TMatmul Backend] Assembly saved to: {}",
        asm_path.display()
    );

    // Create module
    let new_module = Module {
        assembly_code: tmatmul_asm,
        kernels: Vec::new(),
    };

    *module = new_module.wrap();

    // Re-register with actual module handle
    crate::r#impl::checkpoint::register_module_ptx(module.0 as u64, text);
    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn unload(hmod: CUmodule) -> CUresult {
    super::drop_checked::<Module>(hmod)
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_function(
    hfunc: &mut CUfunction,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> CUresult {
    if name.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let function_name = unsafe {
        std::ffi::CStr::from_ptr(name)
            .to_str()
            .map_err(|_| CUerror::INVALID_VALUE)?
    };

    crate::r#impl::hetgpu_debug!("[TMatmul Backend] Getting function: {}", function_name);

    // Create TMatmul kernel handle
    let kernel = TMatmulKernel {
        function_name: function_name.to_string(),
        assembly_code: hmod.assembly_code.clone(),
    };

    *hfunc = kernel.wrap();
    Ok(())
}

// TMatmul kernel structure
#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) struct TMatmulKernel {
    function_name: String,
    assembly_code: String,
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe impl Send for TMatmulKernel {}
#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe impl Sync for TMatmulKernel {}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl ZludaObject for TMatmulKernel {
    const COOKIE: usize = 0xad74ceadb9b2d51c;

    type CudaHandle = CUfunction;

    fn drop_checked(&mut self) -> CUresult {
        crate::r#impl::hetgpu_debug!(
            "[TMatmul Backend] Cleaning up kernel: {}",
            self.function_name
        );
        Ok(())
    }
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> super::FromCuda<'a, CUfunction> for &'a TMatmulKernel {
    fn from_cuda(handle: &'a CUfunction) -> Result<Self, CUerror> {
        super::as_ref::<TMatmulKernel>(handle).as_result()
    }
}

#[cfg(feature = "intel")]
fn try_extract_ptx_from_cubin(binary: &[u8]) -> Option<String> {
    // Check for ELF magic
    if binary.len() < 4
        || !(binary[0] == 0x7f && binary[1] == b'E' && binary[2] == b'L' && binary[3] == b'F')
    {
        eprintln!(
            "[PTX Extract] Not an ELF file (magic: {:02x} {:02x} {:02x} {:02x})",
            binary.get(0).copied().unwrap_or(0),
            binary.get(1).copied().unwrap_or(0),
            binary.get(2).copied().unwrap_or(0),
            binary.get(3).copied().unwrap_or(0)
        );
        return None;
    }

    eprintln!(
        "[PTX Extract] ELF file detected (size: {} bytes), searching for embedded PTX...",
        binary.len()
    );

    // First try to parse ELF properly to find .nv_fatbin section (contains compressed PTX)
    if let Some(ptx) = try_extract_ptx_from_elf_sections(binary) {
        return Some(ptx);
    }

    // Fallback: Search for raw PTX markers
    eprintln!("[PTX Extract] ELF section parsing didn't find PTX, trying raw search...");

    // Search for actual PTX content (starts with ".version X.Y")
    // PTX always starts with .version directive followed by a number
    let version_pattern = b".version ";
    let mut all_version_positions = Vec::new();

    for i in 0..binary.len().saturating_sub(20) {
        if binary[i..].starts_with(version_pattern) {
            // Check if followed by a digit (version number like "7.0")
            let next_char = binary.get(i + version_pattern.len()).copied().unwrap_or(0);
            if next_char.is_ascii_digit() {
                all_version_positions.push(i);
                eprintln!(
                    "[PTX Extract] Found potential PTX at offset {} (next bytes: {:?})",
                    i,
                    &binary[i..binary.len().min(i + 30)]
                        .iter()
                        .map(|&b| if b.is_ascii_graphic() || b == b' ' || b == b'\n' {
                            b as char
                        } else {
                            '.'
                        })
                        .collect::<String>()
                );
            }
        }
    }

    eprintln!(
        "[PTX Extract] Found {} potential PTX start positions",
        all_version_positions.len()
    );

    // Try each position
    for &pos in &all_version_positions {
        if let Some(ptx) = extract_ptx_from_offset_improved(binary, pos) {
            if ptx.len() >= 100 && ptx.contains(".target") {
                eprintln!(
                    "[PTX Extract] Valid PTX found at offset {} ({} bytes)",
                    pos,
                    ptx.len()
                );
                return Some(ptx);
            } else {
                eprintln!("[PTX Extract] Extracted {} bytes from offset {} but doesn't look like valid PTX", ptx.len(), pos);
            }
        }
    }

    eprintln!("[PTX Extract] No valid PTX found in CUBIN");
    eprintln!("[PTX Extract] Note: PyTorch CUBINs often don't contain embedded PTX");
    eprintln!("[PTX Extract] The cuobjdump fallback in cudart_shim.c should handle this");
    None
}

#[cfg(feature = "intel")]
fn try_extract_ptx_from_elf_sections(binary: &[u8]) -> Option<String> {
    // Parse ELF header to find sections
    if binary.len() < 64 {
        return None;
    }

    // Check if 64-bit ELF (class byte at offset 4)
    let is_64bit = binary[4] == 2;
    eprintln!(
        "[PTX Extract] ELF class: {}",
        if is_64bit { "64-bit" } else { "32-bit" }
    );

    if !is_64bit {
        eprintln!("[PTX Extract] 32-bit ELF not supported for PTX extraction");
        return None;
    }

    // For 64-bit ELF:
    // e_shoff (section header offset) is at bytes 40-47
    // e_shentsize (section header entry size) is at bytes 58-59
    // e_shnum (number of section headers) is at bytes 60-61
    // e_shstrndx (section name string table index) is at bytes 62-63

    let e_shoff = u64::from_le_bytes([
        binary[40], binary[41], binary[42], binary[43], binary[44], binary[45], binary[46],
        binary[47],
    ]) as usize;
    let e_shentsize = u16::from_le_bytes([binary[58], binary[59]]) as usize;
    let e_shnum = u16::from_le_bytes([binary[60], binary[61]]) as usize;
    let e_shstrndx = u16::from_le_bytes([binary[62], binary[63]]) as usize;

    eprintln!(
        "[PTX Extract] ELF: shoff={}, shentsize={}, shnum={}, shstrndx={}",
        e_shoff, e_shentsize, e_shnum, e_shstrndx
    );

    if e_shoff == 0 || e_shnum == 0 || e_shoff >= binary.len() {
        eprintln!("[PTX Extract] Invalid section headers");
        return None;
    }

    // Get section name string table
    if e_shstrndx >= e_shnum {
        eprintln!("[PTX Extract] Invalid string table index");
        return None;
    }

    let strtab_offset = e_shoff + e_shstrndx * e_shentsize;
    if strtab_offset + 64 > binary.len() {
        return None;
    }

    // For 64-bit: sh_offset is at bytes 24-31, sh_size is at 32-39
    let strtab_sh_offset = u64::from_le_bytes([
        binary[strtab_offset + 24],
        binary[strtab_offset + 25],
        binary[strtab_offset + 26],
        binary[strtab_offset + 27],
        binary[strtab_offset + 28],
        binary[strtab_offset + 29],
        binary[strtab_offset + 30],
        binary[strtab_offset + 31],
    ]) as usize;

    eprintln!("[PTX Extract] String table at offset {}", strtab_sh_offset);

    // Now search for interesting sections
    let interesting_sections = [".nv_fatbin", ".nv.fatbin", ".nv.module.ptx", ".ptx"];

    for i in 0..e_shnum {
        let sh_offset = e_shoff + i * e_shentsize;
        if sh_offset + 64 > binary.len() {
            continue;
        }

        // sh_name is at bytes 0-3 (offset into string table)
        let sh_name_offset = u32::from_le_bytes([
            binary[sh_offset],
            binary[sh_offset + 1],
            binary[sh_offset + 2],
            binary[sh_offset + 3],
        ]) as usize;

        // Get section name
        let name_start = strtab_sh_offset + sh_name_offset;
        if name_start >= binary.len() {
            continue;
        }

        let name_end = binary[name_start..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| name_start + p)
            .unwrap_or(binary.len().min(name_start + 64));

        let section_name = std::str::from_utf8(&binary[name_start..name_end]).unwrap_or("");

        // sh_offset and sh_size for 64-bit
        let section_offset = u64::from_le_bytes([
            binary[sh_offset + 24],
            binary[sh_offset + 25],
            binary[sh_offset + 26],
            binary[sh_offset + 27],
            binary[sh_offset + 28],
            binary[sh_offset + 29],
            binary[sh_offset + 30],
            binary[sh_offset + 31],
        ]) as usize;
        let section_size = u64::from_le_bytes([
            binary[sh_offset + 32],
            binary[sh_offset + 33],
            binary[sh_offset + 34],
            binary[sh_offset + 35],
            binary[sh_offset + 36],
            binary[sh_offset + 37],
            binary[sh_offset + 38],
            binary[sh_offset + 39],
        ]) as usize;

        if interesting_sections.contains(&section_name)
            || section_name.contains("ptx")
            || section_name.contains("fatbin")
        {
            eprintln!(
                "[PTX Extract] Found interesting section '{}' at offset {}, size {}",
                section_name, section_offset, section_size
            );

            if section_offset > 0
                && section_size > 0
                && section_offset + section_size <= binary.len()
            {
                let section_data = &binary[section_offset..section_offset + section_size];

                // Try to extract PTX from this section
                if let Some(ptx) = try_extract_ptx_from_section(section_data, section_name) {
                    return Some(ptx);
                }
            }
        }
    }

    None
}

#[cfg(feature = "intel")]
fn try_extract_ptx_from_section(data: &[u8], section_name: &str) -> Option<String> {
    eprintln!(
        "[PTX Extract] Analyzing section '{}' ({} bytes)",
        section_name,
        data.len()
    );

    if data.len() < 8 {
        return None;
    }

    // Check for fatbin magic (0xBA55ED50)
    if data.len() >= 4 {
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic == 0xBA55ED50 {
            eprintln!("[PTX Extract] Found fatbin magic in section, parsing fatbin...");
            return try_extract_ptx_from_fatbin(data);
        }
    }

    // Check for compressed data (zlib magic: 0x78)
    if data[0] == 0x78 && (data[1] == 0x01 || data[1] == 0x5e || data[1] == 0x9c || data[1] == 0xda)
    {
        eprintln!("[PTX Extract] Found zlib compressed data, attempting decompression...");
        return try_decompress_zlib(data);
    }

    // Check if it's raw PTX text
    if data.starts_with(b".version ") || data.starts_with(b"//") {
        if let Ok(ptx) = std::str::from_utf8(data) {
            eprintln!("[PTX Extract] Section contains raw PTX text");
            return Some(ptx.trim_end_matches('\0').to_string());
        }
    }

    // Search for PTX within the section
    if let Some(pos) = data.windows(9).position(|w| {
        w.starts_with(b".version ") && w[9..].first().map(|b| b.is_ascii_digit()).unwrap_or(false)
    }) {
        if let Some(ptx) = extract_ptx_from_offset_improved(data, pos) {
            if ptx.len() >= 100 {
                return Some(ptx);
            }
        }
    }

    None
}

#[cfg(feature = "intel")]
fn try_extract_ptx_from_fatbin(data: &[u8]) -> Option<String> {
    // Fatbin structure:
    // Header: magic (4), version (2), header_size (2), fat_size (8)
    // Followed by file entries

    if data.len() < 16 {
        return None;
    }

    let header_size = u16::from_le_bytes([data[6], data[7]]) as usize;

    eprintln!("[PTX Extract] Fatbin header_size: {}", header_size);

    if header_size >= data.len() {
        return None;
    }

    let mut offset = header_size;

    while offset + 24 < data.len() {
        // File entry header
        let kind = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let entry_header_size = u16::from_le_bytes([data[offset + 4], data[offset + 5]]) as usize;
        let payload_size = u64::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
            data[offset + 14],
            data[offset + 15],
        ]) as usize;
        let uncompressed_size = u64::from_le_bytes([
            data[offset + 16],
            data[offset + 17],
            data[offset + 18],
            data[offset + 19],
            data[offset + 20],
            data[offset + 21],
            data[offset + 22],
            data[offset + 23],
        ]) as usize;

        eprintln!(
            "[PTX Extract] Fatbin entry: kind=0x{:04x}, header={}, payload={}, uncompressed={}",
            kind, entry_header_size, payload_size, uncompressed_size
        );

        // kind 0x01 = PTX, 0x02 = CUBIN/ELF
        if kind == 0x01 {
            let payload_start = offset + entry_header_size;
            if payload_start + payload_size <= data.len() {
                let payload = &data[payload_start..payload_start + payload_size];

                // Check if compressed
                if uncompressed_size > payload_size && payload.len() >= 2 {
                    if payload[0] == 0x78 {
                        eprintln!("[PTX Extract] PTX entry is zlib compressed");
                        if let Some(ptx) = try_decompress_zlib(payload) {
                            return Some(ptx);
                        }
                    }
                }

                // Try as raw PTX
                if let Ok(ptx) = std::str::from_utf8(payload) {
                    let ptx = ptx.trim_end_matches('\0');
                    if ptx.len() >= 50 && ptx.contains(".version") {
                        return Some(ptx.to_string());
                    }
                }
            }
        }

        // Move to next entry (aligned)
        let entry_total = entry_header_size + payload_size;
        let aligned = (entry_total + 7) & !7;
        offset += aligned;

        if offset == 0 || aligned == 0 {
            break;
        }
    }

    None
}

#[cfg(feature = "intel")]
fn try_decompress_zlib(data: &[u8]) -> Option<String> {
    use std::io::Read;

    // Try with flate2 if available, otherwise manual inflate
    #[cfg(feature = "flate2")]
    {
        use flate2::read::ZlibDecoder;
        let mut decoder = ZlibDecoder::new(data);
        let mut result = String::new();
        if decoder.read_to_string(&mut result).is_ok() && result.contains(".version") {
            eprintln!(
                "[PTX Extract] Successfully decompressed {} bytes of PTX",
                result.len()
            );
            return Some(result);
        }
    }

    // Fallback: try to use system zlib via C
    eprintln!("[PTX Extract] zlib decompression not available (compile with flate2 feature)");
    eprintln!(
        "[PTX Extract] Compressed data starts with: {:02x} {:02x}",
        data[0], data[1]
    );
    None
}

#[cfg(feature = "intel")]
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(feature = "intel")]
fn find_ptx_start(binary: &[u8], start_offset: usize) -> Option<usize> {
    let version_marker = b".version ";
    for i in start_offset..binary.len().saturating_sub(100) {
        if binary[i..].starts_with(version_marker) {
            eprintln!("[PTX Extract] Found .version directive at offset {}", i);
            return Some(i);
        }
    }
    None
}

// Improved PTX extraction that handles edge cases better
#[cfg(feature = "intel")]
fn extract_ptx_from_offset_improved(binary: &[u8], start: usize) -> Option<String> {
    // PTX should contain certain key elements
    // Look for the end of PTX by finding where valid PTX structure ends

    let mut end = start;
    let mut consecutive_nulls: usize = 0;
    let mut last_valid_end = start;
    let mut found_target = false;
    let mut found_entry = false;
    let mut brace_depth: usize = 0;

    while end < binary.len() && end < start + 500_000 {
        let byte = binary[end];

        // Track null bytes - PTX may have occasional nulls but not many consecutive
        if byte == 0 {
            consecutive_nulls += 1;
            if consecutive_nulls > 3 {
                // More than 3 consecutive nulls - probably end of PTX
                break;
            }
        } else {
            consecutive_nulls = 0;
        }

        // Track braces for function bodies
        if byte == b'{' {
            brace_depth += 1;
        } else if byte == b'}' {
            brace_depth = brace_depth.saturating_sub(1);
            if found_entry && brace_depth == 0 {
                // Found complete function - update valid end
                last_valid_end = end + 1;
            }
        }

        // Check for key PTX markers
        if end + 8 < binary.len() {
            let slice = &binary[end..end + 8];
            if slice.starts_with(b".target ") {
                found_target = true;
            }
            if slice.starts_with(b".entry ") || slice.starts_with(b".func ") {
                found_entry = true;
            }
        }

        // Check for binary data in sliding window
        if end > start + 50 {
            let window_start = end.saturating_sub(50);
            let window = &binary[window_start..end];
            let binary_count = window
                .iter()
                .filter(|&&b| {
                    b > 127 || (b < 32 && b != b'\n' && b != b'\r' && b != b'\t' && b != 0)
                })
                .count();

            // If more than 30% is binary, we've probably left PTX territory
            if binary_count > window.len() * 3 / 10 {
                eprintln!(
                    "[PTX Extract] Binary data detected at offset {}, stopping extraction",
                    end
                );
                break;
            }
        }

        // Update valid end if we're in valid PTX territory
        if byte.is_ascii_graphic()
            || byte == b' '
            || byte == b'\n'
            || byte == b'\r'
            || byte == b'\t'
        {
            if found_target {
                last_valid_end = end + 1;
            }
        }

        end += 1;
    }

    // Use the last valid position if we found PTX structure
    if found_target && last_valid_end > start {
        end = last_valid_end;
    }

    if end <= start {
        eprintln!(
            "[PTX Extract] No valid PTX content found at offset {}",
            start
        );
        return None;
    }

    // Extract and clean up PTX
    let ptx_bytes = &binary[start..end];

    // Filter out null bytes and invalid characters
    let cleaned: Vec<u8> = ptx_bytes
        .iter()
        .copied()
        .filter(|&b| {
            b != 0 && (b.is_ascii_graphic() || b == b' ' || b == b'\n' || b == b'\r' || b == b'\t')
        })
        .collect();

    match String::from_utf8(cleaned) {
        Ok(ptx) => {
            let trimmed = ptx.trim();
            if trimmed.len() < 50 {
                eprintln!(
                    "[PTX Extract] PTX too short ({} bytes) after cleaning",
                    trimmed.len()
                );
                return None;
            }
            eprintln!(
                "[PTX Extract] Extracted {} bytes of PTX (cleaned from {} raw bytes)",
                trimmed.len(),
                end - start
            );
            eprintln!(
                "[PTX Extract] PTX starts with: {}...",
                &trimmed[..trimmed.len().min(100)]
            );
            Some(trimmed.to_string())
        }
        Err(e) => {
            eprintln!("[PTX Extract] Failed to decode PTX as UTF-8: {}", e);
            None
        }
    }
}

// Legacy extraction function (kept for reference)
#[cfg(feature = "intel")]
fn extract_ptx_from_offset(binary: &[u8], start: usize) -> Option<String> {
    let mut end = start;
    while end < binary.len() && end < start + 100_000 {
        let byte = binary[end];
        if byte == 0 {
            break;
        }
        if end > start + 100 {
            let recent = &binary[end.saturating_sub(100)..end];
            let binary_ratio = recent
                .iter()
                .filter(|&&b| b > 127 || (b < 32 && b != b'\n' && b != b'\r' && b != b'\t'))
                .count();
            if binary_ratio > recent.len() / 2 {
                break;
            }
        }
        end += 1;
    }

    let ptx_bytes = &binary[start..end];
    match std::str::from_utf8(ptx_bytes) {
        Ok(ptx) => {
            eprintln!("[PTX Extract] Extracted {} bytes of PTX", ptx.len());
            Some(ptx.to_string())
        }
        Err(e) => {
            eprintln!("[PTX Extract] Failed to decode PTX as UTF-8: {}", e);
            None
        }
    }
}

// NVIDIA backend module implementations - pass through to real libcuda.so
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) struct Module {
    cuda_module: cuda_types::cuda::CUmodule,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
unsafe impl Send for Module {}
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
unsafe impl Sync for Module {}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
impl ZludaObject for Module {
    const COOKIE: usize = 0xe9138bd040487d4a;

    type CudaHandle = CUmodule;

    fn drop_checked(&mut self) -> CUresult {
        let result = nvidia_runtime_sys::cuModuleUnload(self.cuda_module);
        if result != 0 {
            eprintln!(
                "[NVIDIA Backend] cuModuleUnload failed with error {}",
                result
            );
            return Err(CUerror::UNKNOWN);
        }
        Ok(())
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn load_data(module: &mut CUmodule, image: *const std::ffi::c_void) -> CUresult {
    if image.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    eprintln!("[NVIDIA Backend] Loading module data...");

    // Try to extract PTX source for checkpointing
    // Check if image starts with ".version" (indicates PTX text)
    let image_bytes = unsafe { std::slice::from_raw_parts(image as *const u8, 8) };
    let is_ptx = image_bytes.starts_with(b".version") || image_bytes.starts_with(b"//");

    let ptx_source: Option<String> = if is_ptx {
        // It's PTX text, extract the full string
        let c_str = unsafe { std::ffi::CStr::from_ptr(image as *const std::ffi::c_char) };
        match c_str.to_str() {
            Ok(s) => {
                eprintln!("[NVIDIA Backend] Detected PTX source ({} bytes)", s.len());
                Some(s.to_string())
            }
            Err(_) => None,
        }
    } else {
        eprintln!("[NVIDIA Backend] Detected CUBIN/binary module");
        None
    };

    // IMPORTANT: Register PTX source BEFORE loading with NVIDIA driver
    // This ensures PTX is available for checkpointing even if module loading fails
    // (e.g., due to no context or incompatible GPU)
    // We use image address as temporary handle, will update with real handle after loading
    let temp_handle = image as u64;
    if let Some(ref ptx) = ptx_source {
        crate::r#impl::checkpoint::register_module_ptx(temp_handle, ptx);
        eprintln!(
            "[NVIDIA Backend] Pre-registered PTX source for checkpointing (temp handle: 0x{:x})",
            temp_handle
        );
    }

    // Pass through to real CUDA driver
    let mut cuda_module = cuda_types::cuda::CUmodule(ptr::null_mut());
    let result = nvidia_runtime_sys::cuModuleLoadData(&mut cuda_module, image);

    if result != 0 {
        eprintln!(
            "[NVIDIA Backend] cuModuleLoadData failed with error {}",
            result
        );
        // Even though loading failed, PTX is still registered for checkpoint purposes
        // This allows heterogeneous restore to another backend
        eprintln!(
            "[NVIDIA Backend] PTX source is still available for heterogeneous checkpoint/restore"
        );
        return Err(CUerror::NO_BINARY_FOR_GPU);
    }

    eprintln!("[NVIDIA Backend] Module loaded successfully");

    // Create module wrapper
    let new_module = Module { cuda_module };

    *module = new_module.wrap();

    // Re-register PTX source with the actual module handle
    // This updates the checkpoint registry with the correct handle
    if let Some(ptx) = ptx_source {
        crate::r#impl::checkpoint::register_module_ptx(module.0 as u64, &ptx);
        eprintln!(
            "[NVIDIA Backend] Registered PTX source for checkpointing (module: 0x{:x})",
            module.0 as u64
        );
    }

    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn unload(hmod: CUmodule) -> CUresult {
    super::drop_checked::<Module>(hmod)
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn get_function(
    hfunc: &mut CUfunction,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> CUresult {
    if name.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let function_name = unsafe {
        CStr::from_ptr(name)
            .to_str()
            .map_err(|_| CUerror::INVALID_VALUE)?
    };

    eprintln!("[NVIDIA Backend] Getting function: {}", function_name);

    // Pass through to real CUDA driver
    let mut cuda_func = cuda_types::cuda::CUfunction(ptr::null_mut());
    let result = nvidia_runtime_sys::cuModuleGetFunction(&mut cuda_func, hmod.cuda_module, name);

    if result != 0 {
        eprintln!(
            "[NVIDIA Backend] cuModuleGetFunction failed with error {}",
            result
        );
        return Err(CUerror::NOT_FOUND);
    }

    eprintln!("[NVIDIA Backend] Function '{}' found", function_name);

    // Create kernel wrapper
    let kernel = NvidiaKernel {
        cuda_function: cuda_func,
        function_name: function_name.to_string(),
    };

    *hfunc = kernel.wrap();
    Ok(())
}

// NVIDIA kernel structure
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) struct NvidiaKernel {
    pub cuda_function: cuda_types::cuda::CUfunction,
    pub function_name: String,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
unsafe impl Send for NvidiaKernel {}
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
unsafe impl Sync for NvidiaKernel {}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
impl ZludaObject for NvidiaKernel {
    const COOKIE: usize = 0xad74ceadb9b2d51c;

    type CudaHandle = CUfunction;

    fn drop_checked(&mut self) -> CUresult {
        // CUDA functions don't need explicit cleanup - they're cleaned up with the module
        Ok(())
    }
}

// ============================================================================
// PACC backend module implementations (SiFive Intelligence XM / RISC-V IME)
// ============================================================================

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) struct Module {
    device: *mut pacc_runtime_sys::pacc_Device,
    program: Option<*mut pacc_runtime_sys::pacc_Program>,
    kernels: Vec<(String, *mut pacc_runtime_sys::pacc_Kernel)>,
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe impl Send for Module {}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe impl Sync for Module {}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn current_pacc_device_id() -> Result<i32, CUerror> {
    let device_count = super::driver::global_state()?.devices.len() as i32;
    let device_id = match super::context::get_current_pacc() {
        Ok(ctx) => ctx.device_id,
        Err(_) => 0,
    };
    if device_id >= 0 && device_id < device_count {
        Ok(device_id)
    } else {
        Ok(0)
    }
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl ZludaObject for Module {
    const COOKIE: usize = 0xe9138bd040487d4a;

    type CudaHandle = CUmodule;

    fn drop_checked(&mut self) -> CUresult {
        self.kernels.clear();
        self.program = None;
        Ok(())
    }
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn load_data(module: &mut CUmodule, image: *const std::ffi::c_void) -> CUresult {
    if image.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    // Try to extract PTX source
    let image_bytes = unsafe { std::slice::from_raw_parts(image as *const u8, 8) };
    let is_ptx = image_bytes.starts_with(b".version") || image_bytes.starts_with(b"//");

    if is_ptx {
        let c_str = unsafe { std::ffi::CStr::from_ptr(image as *const std::ffi::c_char) };
        if let Ok(ptx_text) = c_str.to_str() {
            crate::r#impl::checkpoint::register_module_ptx(image as u64, ptx_text);
        }
    }

    let device_id = current_pacc_device_id()?;
    // Bind this CUDA module to the current logical PACC device/context.
    let device = unsafe { pacc_runtime_sys::pacc_CreateDevice(device_id as u32) };
    if device.is_null() {
        eprintln!("[PACC Backend] Failed to create PACC device {}", device_id);
        return Err(CUerror::UNKNOWN);
    }
    if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[PACC Backend] cuModuleLoadData binding module to pacc{}",
            device_id
        );
    }

    // Create PACC program
    let program_ptr = unsafe { pacc_runtime_sys::pacc_CreateProgram() };

    // If PTX, ask the PACC runtime to compile PTX -> LLVM -> XM ELF and load it.
    if is_ptx && !program_ptr.is_null() {
        let c_str = unsafe { std::ffi::CStr::from_ptr(image as *const std::ffi::c_char) };
        if let Ok(ptx_text) = c_str.to_str() {
            if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "[PACC Backend] Compiling PTX ({} bytes) through pacc runtime...",
                    ptx_text.len()
                );
            }
            let result = unsafe {
                pacc_runtime_sys::pacc_LoadProgramPtx(
                    program_ptr,
                    std::ptr::null(),
                    c"module.ptx".as_ptr(),
                    ptx_text.as_ptr(),
                    ptx_text.len() as u64,
                    std::ptr::null(),
                    0,
                )
            };
            if result != pacc_runtime_sys::pacc_Result_Success {
                let compile_error = unsafe {
                    program_ptr
                        .as_ref()
                        .and_then(|p| p.compile_error.as_deref())
                        .map(str::to_owned)
                };
                eprintln!(
                    "[PACC Backend] pacc_LoadProgramPtx failed: {}{}",
                    result,
                    compile_error
                        .as_deref()
                        .map(|msg| format!(" ({})", msg))
                        .unwrap_or_default()
                );
            } else if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "[PACC Backend] pacc_LoadProgramPtx succeeded for {} bytes of PTX",
                    ptx_text.len()
                );
            }
        }
    }

    let new_module = Module {
        device,
        program: if program_ptr.is_null() {
            None
        } else {
            Some(program_ptr)
        },
        kernels: Vec::new(),
    };

    *module = new_module.wrap();

    if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        let module_ref = super::as_ref::<Module>(module).as_result()?;
        let elf_len = module_ref
            .program
            .map(|p| unsafe { (*p).elf_bytes.len() })
            .unwrap_or(0);
        eprintln!(
            "[PACC Backend] cuModuleLoadData installed module={:?} program={:?} elf_bytes={}",
            *module, module_ref.program, elf_len
        );
    }

    if is_ptx {
        let c_str = unsafe { std::ffi::CStr::from_ptr(image as *const std::ffi::c_char) };
        if let Ok(ptx_text) = c_str.to_str() {
            crate::r#impl::checkpoint::register_module_ptx(module.0 as u64, ptx_text);
        }
    }

    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn unload(hmod: CUmodule) -> CUresult {
    if hmod.0.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    super::drop_checked::<Module>(hmod)
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_function(
    hfunc: *mut CUfunction,
    hmod: CUmodule,
    name: *const ::core::ffi::c_char,
) -> CUresult {
    if hfunc.is_null() || hmod.0.is_null() || name.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let function_name = unsafe {
        std::ffi::CStr::from_ptr(name)
            .to_str()
            .map_err(|_| CUerror::INVALID_VALUE)?
    };

    crate::r#impl::hetgpu_debug!("[PACC Backend] Getting function: {}", function_name);

    // Get program from the wrapped CUDA module handle. PACC modules use the
    // same LiveCheck wrapper as the other backends; casting CUmodule directly
    // to Module reads the cookie as fields and makes program look like None.
    let module_ref = super::as_ref::<Module>(&hmod).as_result()?;
    let kernel_ptr = if let Some(program) = module_ref.program {
        unsafe { pacc_runtime_sys::pacc_CreateKernelOnDevice(program, module_ref.device, name) }
    } else {
        std::ptr::null_mut()
    };

    if std::env::var("HETGPU_PACC_LOG_KERNEL_HANDLES")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[PACC Backend] Function '{}' module_device={:?} program={:?} elf_bytes={} kernel_ptr={:?}",
            function_name,
            module_ref.device,
            module_ref.program,
            module_ref
                .program
                .map(|p| unsafe { (*p).elf_bytes.len() })
                .unwrap_or(0),
            kernel_ptr
        );
    }

    let pacc_kernel = PaccKernel {
        device: module_ref.device,
        kernel_ptr,
        kernel_name: function_name.to_string(),
    };

    let kernel_box = Box::new(pacc_kernel);
    let kernel_raw = Box::into_raw(kernel_box);
    unsafe { *hfunc = CUfunction(kernel_raw as *mut _) };
    Ok(())
}

// PACC kernel structure
#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[repr(C)]
pub(crate) struct PaccKernel {
    pub device: *mut pacc_runtime_sys::pacc_Device,
    pub kernel_ptr: *mut pacc_runtime_sys::pacc_Kernel,
    pub kernel_name: String,
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe impl Send for PaccKernel {}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe impl Sync for PaccKernel {}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl ZludaObject for PaccKernel {
    const COOKIE: usize = 0xad74ceadb9b2d51c;

    type CudaHandle = CUfunction;

    fn drop_checked(&mut self) -> CUresult {
        eprintln!("[PACC Backend] Cleaning up kernel: {}", self.kernel_name);
        Ok(())
    }
}

// FromCuda<CUfunction> for &PaccKernel is generated by from_cuda_object!(module::PaccKernel)
