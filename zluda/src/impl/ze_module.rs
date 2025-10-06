use super::context;
use super::ZludaObject;
use cuda_types::cuda::*;
use std::os::raw::c_void;
use std::{ffi::CStr, ptr};
use ze_runtime_sys::*;

/// SPIR-V module implementation for Intel
pub(crate) struct SpirvModule {
    pub ptx_text: String,
    pub name: String,
    pub module: ze_module_handle_t,
    pub functions: Vec<(String, ze_kernel_handle_t)>,
}

impl SpirvModule {
    pub fn new(text: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Get module name (using a default if not available)
        let module_name = "anonymous_module".to_string();

        // Create a SPIRV module from the PTX
        let spirv_module = Self::create_spirv_module(text, &module_name)?;

        Ok(spirv_module)
    }

    fn create_spirv_module(
        ptx_text: &str,
        name: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Get the current context
        let ctx = context::get_current_ze().unwrap();

        // Parse PTX
        let ast = ptx_parser::parse_module_checked(ptx_text)
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{:?}", e))) as Box<dyn std::error::Error>)?;

        // Convert PTX to LLVM IR with default attributes
        let attributes = ptx::Attributes {
            clock_rate: 2124000, // Default clock rate in kHz
        };
        let llvm_module = ptx::to_llvm_module(ast, attributes, |_| {})?;

        // Get LLVM IR string from module
        let llvm_ir = llvm_module.llvm_ir.print_module_to_string();

        // For now, return the LLVM IR as bytes (SPIRV conversion would go here)
        // This is a placeholder - actual SPIRV conversion would be needed for full functionality
        let spirv_binary = llvm_ir.to_str().as_bytes().to_vec();

        // Create module build description
        let module_desc = ze_module_desc_t {
            stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MODULE_DESC,
            pNext: ptr::null(),
            format: ze_module_format_t::ZE_MODULE_FORMAT_IL_SPIRV,
            inputSize: spirv_binary.len(),
            pInputModule: spirv_binary.as_ptr(),
            pBuildFlags: ptr::null(),
            pConstants: ptr::null(),
        };

        // Create an empty build log description
        let mut build_log = ptr::null_mut();

        // Create module using the ZE API
        let ze_module: *mut ze_module_handle_t = ptr::null_mut();
        let result = unsafe {
            zeModuleCreate(
                ctx.context,
                ctx.device,
                &module_desc,
                ze_module,
                &mut build_log,
            )
        };

        // Check for build errors
        if result != ze_result_t::ZE_RESULT_SUCCESS {
            unsafe {
                if !build_log.is_null() {
                    zeModuleBuildLogDestroy(build_log);
                }
            }
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create ZE module: {:?}", result),
            )));
        }

        // Create module wrapper
        let module = unsafe { *ze_module };

        // Load module functions
        let functions = Vec::new();
        let mut spirv_module = SpirvModule {
            ptx_text: ptx_text.to_string(),
            name: name.to_string(),
            module,
            functions,
        };

        // Load functions from the module
        spirv_module.load_functions()?;

        Ok(spirv_module)
    }

    fn load_functions(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Get function count
        let mut count = 0u32;
        let result = unsafe { zeModuleGetKernelNames(self.module, &mut count, ptr::null_mut()) };

        if result != ze_result_t::ZE_RESULT_SUCCESS {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to get kernel count: {:?}", result),
            )));
        }

        if count == 0 {
            return Ok(()); // No functions to load
        }

        // Allocate space for function names
        let function_names = Vec::<*mut i8>::with_capacity(count as usize);
        let result = unsafe {
            zeModuleGetKernelNames(self.module, &mut count, function_names.as_ptr() as *mut _)
        };

        if result != ze_result_t::ZE_RESULT_SUCCESS {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to get kernel names: {:?}", result),
            )));
        }

        // Load each function
        for i in 0..count as usize {
            let name_ptr = function_names[i];
            let name_str = unsafe { CStr::from_ptr(name_ptr).to_str()? };

            // Create kernel description
            let kernel_desc = ze_kernel_desc_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_KERNEL_DESC,
                pNext: ptr::null(),
                flags: 0,
                pKernelName: name_ptr,
            };

            // Create the kernel
            let kernel = ptr::null_mut();
            let result = unsafe { zeKernelCreate(self.module, &kernel_desc, unsafe { *kernel }) };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create kernel {}: {:?}", name_str, result),
                )));
            }

            // Store the kernel handle
            self.functions
                .push((name_str.to_string(), unsafe { **kernel }));
        }

        Ok(())
    }

    pub fn get_function(&self, name: &str) -> Option<ze_kernel_handle_t> {
        self.functions
            .iter()
            .find(|(fname, _)| fname == name)
            .map(|(_, kernel)| *kernel)
    }
}

impl Drop for SpirvModule {
    fn drop(&mut self) {
        // Clean up functions
        for (_, kernel) in &self.functions {
            unsafe {
                zeKernelDestroy(*kernel);
            }
        }

        // Clean up module
        unsafe {
            zeModuleDestroy(self.module);
        }
    }
}

pub(crate) struct Module {
    context: ze_context_handle_t,
    device: ze_device_handle_t,
    module: ze_module_handle_t,
    functions: Vec<(String, ze_kernel_handle_t)>,
}
unsafe impl Send for Module {}
unsafe impl Sync for Module {}
impl ZludaObject for Module {
    const COOKIE: usize = 0xe9138bd040487d4a;

    type CudaHandle = CUmodule;

    fn drop_checked(&mut self) -> CUresult {
        // Clean up all kernels first
        for (_, kernel) in &self.functions {
            unsafe {
                zeKernelDestroy(*kernel);
            }
        }
        self.functions.clear();

        // Destroy the module
        let result = unsafe { zeModuleDestroy(self.module) };
        if result != ze_result_t::ZE_RESULT_SUCCESS {
            return ze_to_cuda_result(result);
        }

        Ok(())
    }
}

pub(crate) fn load_data(module: &mut CUmodule, image: *const std::ffi::c_void) -> CUresult {
    // Parse the PTX text
    let text = unsafe { CStr::from_ptr(image.cast()) }
        .to_str()
        .map_err(|_| CUerror::INVALID_VALUE)?;

    let spirv_module = SpirvModule::new(text).map_err(|_| CUerror::NO_BINARY_FOR_GPU)?;
    match load_data_impl(module, spirv_module) {
        Ok(()) => CUresult::SUCCESS,
        Err(e) => Err(e),
    }
}

pub(crate) fn load_data_impl(
    module: &mut CUmodule,
    spirv_module: SpirvModule,
) -> Result<(), CUerror> {
    // Get current context and device
    let (context, device) = get_current_context_and_device()?;

    // Convert PTX to SPIRV
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
    let ze_module = ptr::null_mut();
    let mut build_log = ptr::null_mut();

    let result =
        unsafe { zeModuleCreate(context, device, &module_desc, ze_module, &mut build_log) };

    // Check if build log exists and handle it
    if !build_log.is_null() {
        unsafe { zeModuleBuildLogDestroy(build_log) };
    }

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return Err(CUerror::UNKNOWN);
    }

    // Create an empty list to store kernels
    let functions = Vec::new();

    // Create and return the Module object
    *module = Module {
        context,
        device,
        module: unsafe { *ze_module },
        functions,
    }
    .wrap();

    Ok(())
}

fn ptx_to_spirv(spirv_module: &SpirvModule) -> Result<Vec<u8>, CUerror> {
    // Parse PTX
    let ast = ptx_parser::parse_module_checked(&spirv_module.ptx_text)
        .map_err(|_| CUerror::INVALID_VALUE)?;

    // Convert PTX AST to LLVM IR with default attributes
    let attributes = ptx::Attributes {
        clock_rate: 2124000, // Default clock rate in kHz
    };
    let llvm_module = ptx::to_llvm_module(ast, attributes, |_| {})
        .map_err(|_| CUerror::INVALID_VALUE)?;

    // Get LLVM IR string from module
    let llvm_ir = llvm_module.llvm_ir.print_module_to_string();

    // Placeholder: Return LLVM IR as bytes (SPIRV conversion would be needed)
    let spirv_binary = llvm_ir.to_str().as_bytes().to_vec();

    Ok(spirv_binary)
}

fn get_current_context_and_device() -> Result<(ze_context_handle_t, ze_device_handle_t), CUerror> {
    // Get the current context
    let context = super::context::get_current_ze()?;

    // Return context and device handles
    Ok((context.context, context.device))
}

pub(crate) fn unload(hmod: CUmodule) -> CUresult {
    super::drop_checked::<Module>(hmod)
}

pub(crate) fn get_function(
    hfunc: &mut CUfunction,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> CUresult {
    let name_str = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|_| CUerror::INVALID_VALUE)?;

    // Check if kernel already exists
    if let Some((_, kernel)) = hmod.functions.iter().find(|(n, _)| n == name_str) {
        *hfunc = ZeKernel {
            context: hmod.context,
            device: hmod.device,
            module: hmod.module,
            kernel: *kernel,
        }
        .wrap();
        return CUresult::SUCCESS;
    }

    // Create new kernel
    let kernel: *mut ze_kernel_handle_t = ptr::null_mut();
    let kernel_desc = ze_kernel_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_KERNEL_DESC,
        pNext: ptr::null(),
        flags: 0,
        pKernelName: name,
    };

    let result = unsafe { zeKernelCreate(hmod.module, &kernel_desc, kernel) };

    match result {
        ze_result_t::ZE_RESULT_SUCCESS => {
            let kernel_wrapper = ZeKernel {
                context: hmod.context,
                device: hmod.device,
                module: hmod.module,
                kernel: unsafe { *kernel },
            };

            // Store the kernel in the module's function list
            let module_mut = hmod as *const Module as *mut Module;
            unsafe {
                (*module_mut)
                    .functions
                    .push((name_str.to_string(), unsafe { *kernel }));
            }

            *hfunc = kernel_wrapper.wrap();
            CUresult::SUCCESS
        }
        ze_result_t::ZE_RESULT_ERROR_INVALID_KERNEL_NAME => CUresult::ERROR_INVALID_IMAGE,
        _ => CUresult::ERROR_INVALID_VALUE,
    }
}

pub(crate) struct ZeKernel {
    pub context: ze_context_handle_t,
    pub device: ze_device_handle_t,
    pub module: ze_module_handle_t,
    pub kernel: ze_kernel_handle_t,
}
unsafe impl Send for ZeKernel {}
unsafe impl Sync for ZeKernel {}
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

pub(crate) fn ze_to_cuda_result(result: ze_result_t) -> CUresult {
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

// Memory type conversion
pub fn get_pointer_attribute(
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
    data: *mut c_void,
) -> CUresult {
    // Get the ZE context
    let ze_context = super::context::get_current_ze()?;

    match attribute {
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT => {
            // Get the memory allocation properties
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = unsafe {
                zeMemGetAllocProperties(
                    ze_context.context,
                    ptr.0 as *const c_void,
                    &mut alloc_props,
                    ptr::null_mut(),
                )
            };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return ze_to_cuda_result(result);
            }

            // Store the context handle
            unsafe {
                *(data.cast::<CUcontext>()) = <context::Context as Clone>::clone(&ze_context).wrap()
            };
            CUresult::SUCCESS
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE => {
            // Query memory attributes
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = unsafe {
                zeMemGetAllocProperties(
                    ze_context.context,
                    ptr.0 as *const c_void,
                    &mut alloc_props,
                    ptr::null_mut(),
                )
            };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return ze_to_cuda_result(result);
            }

            // Convert Level Zero memory type to CUDA memory type
            let cuda_type = match alloc_props.type_ {
                ze_memory_type_t::ZE_MEMORY_TYPE_HOST => CUmemorytype::CU_MEMORYTYPE_HOST,
                ze_memory_type_t::ZE_MEMORY_TYPE_DEVICE => CUmemorytype::CU_MEMORYTYPE_DEVICE,
                ze_memory_type_t::ZE_MEMORY_TYPE_SHARED => CUmemorytype::CU_MEMORYTYPE_UNIFIED,
                _ => return CUresult::ERROR_INVALID_VALUE,
            };

            unsafe { *(data.cast()) = cuda_type };
            CUresult::SUCCESS
        }

        // Add implementations for other attributes as needed
        _ => CUresult::ERROR_INVALID_VALUE,
    }
}
