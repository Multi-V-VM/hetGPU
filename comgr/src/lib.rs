#[cfg(feature = "amd")]
use amd_comgr_sys::*;
#[cfg(feature = "gemmini")]
use gemmini_comgr_sys::*;
#[cfg(feature = "intel")]
use intel_comgr_sys::*;
#[cfg(feature = "cutile")]
use cutile_comgr_sys::*;
use std::{
    ffi::{CStr, CString},
    mem, ptr,
};
#[cfg(feature = "tenstorrent")]
use tt_comgr_sys::*;

#[cfg(feature = "amd")]
struct Data(amd_comgr_data_t);
#[cfg(feature = "amd")]
impl Data {
    fn new(
        kind: amd_comgr_data_kind_t,
        name: &CStr,
        content: &[u8],
    ) -> Result<Self, amd_comgr_status_s> {
        let mut data = unsafe { mem::zeroed() };
        unsafe { amd_comgr_create_data(kind, &mut data) }?;
        unsafe { amd_comgr_set_data_name(data, name.as_ptr()) }?;
        unsafe { amd_comgr_set_data(data, content.len(), content.as_ptr().cast()) }?;
        Ok(Self(data))
    }

    fn get(&self) -> amd_comgr_data_t {
        self.0
    }

    fn copy_content(&self) -> Result<Vec<u8>, amd_comgr_status_s> {
        let mut size = unsafe { mem::zeroed() };
        unsafe { amd_comgr_get_data(self.get(), &mut size, ptr::null_mut()) }?;
        let mut result: Vec<u8> = Vec::with_capacity(size);
        unsafe { result.set_len(size) };
        unsafe { amd_comgr_get_data(self.get(), &mut size, result.as_mut_ptr().cast()) }?;
        Ok(result)
    }
}
#[cfg(feature = "amd")]
struct DataSet(amd_comgr_data_set_t);
#[cfg(feature = "amd")]
impl DataSet {
    fn new() -> Result<Self, amd_comgr_status_s> {
        let mut data_set = unsafe { mem::zeroed() };
        unsafe { amd_comgr_create_data_set(&mut data_set) }?;
        Ok(Self(data_set))
    }

    fn add(&self, data: &Data) -> Result<(), amd_comgr_status_s> {
        unsafe { amd_comgr_data_set_add(self.get(), data.get()) }
    }

    fn get(&self) -> amd_comgr_data_set_t {
        self.0
    }

    fn get_data(
        &self,
        kind: amd_comgr_data_kind_t,
        index: usize,
    ) -> Result<Data, amd_comgr_status_s> {
        let mut data = unsafe { mem::zeroed() };
        unsafe { amd_comgr_action_data_get_data(self.get(), kind, index, &mut data) }?;
        Ok(Data(data))
    }
}
#[cfg(feature = "amd")]
impl Drop for DataSet {
    fn drop(&mut self) {
        unsafe { amd_comgr_destroy_data_set(self.get()).ok() };
    }
}
#[cfg(feature = "amd")]
struct ActionInfo(amd_comgr_action_info_t);

#[cfg(feature = "amd")]
impl ActionInfo {
    fn new() -> Result<Self, amd_comgr_status_s> {
        let mut action = unsafe { mem::zeroed() };
        unsafe { amd_comgr_create_action_info(&mut action) }?;
        Ok(Self(action))
    }

    fn set_isa_name(&self, isa: &CStr) -> Result<(), amd_comgr_status_s> {
        let mut full_isa = "amdgcn-amd-amdhsa--".to_string().into_bytes();
        full_isa.extend(isa.to_bytes_with_nul());
        unsafe { amd_comgr_action_info_set_isa_name(self.get(), full_isa.as_ptr().cast()) }
    }

    fn set_language(&self, language: amd_comgr_language_t) -> Result<(), amd_comgr_status_s> {
        unsafe { amd_comgr_action_info_set_language(self.get(), language) }
    }

    fn set_options<'a>(
        &self,
        options: impl Iterator<Item = &'a CStr>,
    ) -> Result<(), amd_comgr_status_s> {
        let options = options.map(|x| x.as_ptr()).collect::<Vec<_>>();
        unsafe {
            amd_comgr_action_info_set_option_list(
                self.get(),
                options.as_ptr().cast_mut(),
                options.len(),
            )
        }
    }

    fn get(&self) -> amd_comgr_action_info_t {
        self.0
    }
}

#[cfg(feature = "amd")]
impl Drop for ActionInfo {
    fn drop(&mut self) {
        unsafe { amd_comgr_destroy_action_info(self.get()).ok() };
    }
}
#[cfg(feature = "amd")]
pub fn compile_bitcode(
    gcn_arch: &CStr,
    main_buffer: &[u8],
    ptx_impl: &[u8],
) -> Result<Vec<u8>, amd_comgr_status_s> {
    let bitcode_data_set = DataSet::new()?;
    let main_bitcode_data = Data::new(
        amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_BC,
        c"zluda.bc",
        main_buffer,
    )?;
    bitcode_data_set.add(&main_bitcode_data)?;
    let stdlib_bitcode_data = Data::new(
        amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_BC,
        c"ptx_impl.bc",
        ptx_impl,
    )?;
    bitcode_data_set.add(&stdlib_bitcode_data)?;
    let linking_info = ActionInfo::new()?;
    let linked_data_set = do_action(
        &bitcode_data_set,
        &linking_info,
        amd_comgr_action_kind_t::AMD_COMGR_ACTION_LINK_BC_TO_BC,
    )?;
    let link_with_device_libs_info = ActionInfo::new()?;
    link_with_device_libs_info.set_isa_name(gcn_arch)?;
    link_with_device_libs_info.set_language(amd_comgr_language_t::AMD_COMGR_LANGUAGE_LLVM_IR)?;
    // This makes no sense, but it makes ockl linking work
    link_with_device_libs_info
        .set_options([c"-Xclang", c"-mno-link-builtin-bitcode-postopt"].into_iter())?;
    let with_device_libs = do_action(
        &linked_data_set,
        &link_with_device_libs_info,
        amd_comgr_action_kind_t::AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC,
    )?;
    let compile_action_info = ActionInfo::new()?;
    compile_action_info.set_isa_name(gcn_arch)?;
    let common_options = [c"-O3", c"-mno-wavefrontsize64", c"-mcumode"].into_iter();
    let opt_options = if cfg!(debug_assertions) {
        [c"-g", c"", c"", c"", c""]
    } else {
        [
            c"-g0",
            // default inlining threshold times 10
            c"-mllvm",
            c"-inline-threshold=2250",
            c"-mllvm",
            c"-inlinehint-threshold=3250",
        ]
    };
    compile_action_info.set_options(common_options.chain(opt_options))?;
    let reloc_data_set = do_action(
        &with_device_libs,
        &compile_action_info,
        amd_comgr_action_kind_t::AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
    )?;
    let exec_data_set = do_action(
        &reloc_data_set,
        &compile_action_info,
        amd_comgr_action_kind_t::AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
    )?;
    let executable =
        exec_data_set.get_data(amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_EXECUTABLE, 0)?;
    executable.copy_content()
}

#[cfg(feature = "amd")]
fn do_action(
    data_set: &DataSet,
    action: &ActionInfo,
    kind: amd_comgr_action_kind_t,
) -> Result<DataSet, amd_comgr_status_s> {
    let result = DataSet::new()?;
    unsafe { amd_comgr_do_action(kind, action.get(), data_set.get(), result.get()) }?;
    Ok(result)
}

#[cfg(feature = "intel")]
pub fn compile_bitcode(
    gcn_arch: &CStr,
    main_buffer: &[u8],
    ptx_impl: &[u8],
) -> Result<Vec<u8>, intel_comgr_status_s> {
    // Optional debug log
    eprintln!("ZLUDA DEBUG: Compiling bitcode for Intel GPU target");
    eprintln!(
        "ZLUDA DEBUG: Main buffer size: {} bytes, PTX impl size: {} bytes",
        main_buffer.len(),
        ptx_impl.len()
    );
    eprintln!(
        "ZLUDA DEBUG: Target architecture: {:?}",
        gcn_arch.to_string_lossy()
    );

    // Directly try to compile - no fallback
    let result = try_compile_bitcode(gcn_arch, main_buffer, ptx_impl);

    match &result {
        Ok(buffer) => {
            eprintln!(
                "ZLUDA DEBUG: Compilation succeeded, generated SPIR-V size: {} bytes",
                buffer.len()
            );
        }
        Err(e) => {
            eprintln!("ZLUDA DEBUG: Compilation failed with error: {:?}", e);
        }
    }

    result
}

#[cfg(feature = "intel")]
fn try_compile_bitcode(
    gcn_arch: &CStr,
    main_buffer: &[u8],
    ptx_impl: &[u8],
) -> Result<Vec<u8>, intel_comgr_status_s> {
    eprintln!(
        "ZLUDA VERBOSE: Creating relocatable with buffer size = {}, ptx_impl size = {}",
        main_buffer.len(),
        ptx_impl.len()
    );

    // Create new DataSet for inputs
    let bitcode_data_set = DataSet::new()?;

    // Create the main bitcode data
    let mut main_data = unsafe { mem::zeroed() };
    match unsafe {
        intel_comgr_create_data(
            intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BC,
            &mut main_data,
        )
    } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Created main bitcode data input"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to create main bitcode data: {:?}", e);
            return Err(e);
        }
    }

    match unsafe {
        intel_comgr_data_set_bytes(
            main_data,
            main_buffer.as_ptr() as *const std::os::raw::c_void,
            main_buffer.len(),
        )
    } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Set main bitcode data content"),
        Err(e) => {
            eprintln!(
                "ZLUDA ERROR: Failed to set main bitcode data content: {:?}",
                e
            );
            return Err(e);
        }
    }

    match unsafe { intel_comgr_data_set_name(main_data, c"combined_module.bc".as_ptr()) } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Set main bitcode data name"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to set main bitcode data name: {:?}", e);
            return Err(e);
        }
    }

    // Add the main bitcode data to the input DataSet
    match unsafe { intel_comgr_data_set_add(bitcode_data_set.0, main_data) } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Added main bitcode data to input DataSet"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to add main bitcode data: {:?}", e);
            return Err(e);
        }
    }

    // Setup compilation options
    let mut compile_info = unsafe { mem::zeroed() };
    match unsafe { intel_comgr_create_action_info(&mut compile_info) } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Created compile action info"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to create compile action info: {:?}", e);
            return Err(e);
        }
    }

    match unsafe {
        intel_comgr_action_info_set_language(
            compile_info,
            intel_comgr_language_s::INTEL_COMGR_LANGUAGE_OPENCL_2_0,
        )
    } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Set compile language to OpenCL 2.0"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to set compile language: {:?}", e);
            return Err(e);
        }
    }

    // Set the target architecture
    let target_cstr =
        CString::new(format!("skl-{}", "64")).expect("failed to create target string");
    match unsafe { intel_comgr_action_info_set_target(compile_info, target_cstr.as_ptr()) } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Set compile target to SKL-64"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to set compile target: {:?}", e);
            return Err(e);
        }
    }

    // Perform the BC to relocatable action
    let action_info = ActionInfo(compile_info);
    let reloc_data_set = do_action(
        &bitcode_data_set,
        &action_info,
        intel_comgr_action_kind_s::INTEL_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
    )?;

    // Get the output relocatable object data
    let mut count = 0;
    match unsafe { intel_comgr_get_data_count(reloc_data_set.0, &mut count) } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Found {} output data objects", count),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to get output data count: {:?}", e);
            return Err(e);
        }
    }

    if count == 0 {
        eprintln!("ZLUDA ERROR: No output data objects found");
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
    }

    // Try each data object until we find one we can read successfully
    for i in 0..count {
        eprintln!(
            "ZLUDA VERBOSE: Attempting to read output object {}/{}",
            i + 1,
            count
        );

        let mut data = unsafe { mem::zeroed() };
        match unsafe { intel_comgr_get_data(reloc_data_set.0, i, &mut data) } {
            Ok(_) => eprintln!("ZLUDA VERBOSE: Successfully got output data #{}", i),
            Err(e) => {
                eprintln!("ZLUDA ERROR: Failed to get output data #{}: {:?}", i, e);
                continue;
            }
        }

        // Get the data kind
        let mut kind = intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_RELOCATABLE;
        match unsafe { intel_comgr_get_data_kind(data, &mut kind) } {
            Ok(_) => {
                let kind_str = match kind.0 {
                    1 => "SOURCE",
                    2 => "INCLUDE",
                    3 => "PRECOMPILED_HEADER",
                    4 => "DIAGNOSTIC",
                    5 => "LOG",
                    6 => "BC",
                    7 => "RELOCATABLE",
                    8 => "EXECUTABLE",
                    9 => "BYTES",
                    _ => "UNKNOWN",
                };
                eprintln!("ZLUDA VERBOSE: Output data kind: {}", kind_str);
            }
            Err(e) => {
                eprintln!("ZLUDA WARNING: Failed to get data kind: {:?}", e);
            }
        }

        // Try to get a handle to any data name
        let mut name_size = 0;
        let mut name_ptr = std::ptr::null_mut();
        match unsafe { intel_comgr_get_data_name(data, &mut name_size, name_ptr) } {
            Ok(_) => {
                if name_size > 0 {
                    let mut name_buffer = vec![0u8; name_size];
                    match unsafe {
                        intel_comgr_get_data_name(
                            data,
                            &mut name_size,
                            name_buffer.as_mut_ptr() as *mut i8,
                        )
                    } {
                        Ok(_) => {
                            if let Ok(name) = std::str::from_utf8(&name_buffer[..name_size - 1]) {
                                eprintln!("ZLUDA VERBOSE: Output data name: {}", name);
                            }
                        }
                        Err(_) => {}
                    }
                }
            }
            Err(_) => {}
        }

        // Get the content of the output data
        let mut buffer = Vec::new();
        let mut size = 0;

        // First try to get the size of the data
        let size_result =
            unsafe { intel_comgr_data_get_bytes(data, std::ptr::null_mut(), &mut size) };

        if let Err(e) = size_result {
            eprintln!(
                "ZLUDA WARNING: Failed to get output data size for object #{}: {:?}",
                i, e
            );
            unsafe { intel_comgr_release_data(data).ok() };
            continue;
        }

        eprintln!("ZLUDA VERBOSE: Output data #{} size: {} bytes", i, size);

        if size > 0 {
            // Allocate buffer and get bytes
            buffer.reserve(size);
            unsafe {
                buffer.set_len(size);
            }
            match unsafe {
                intel_comgr_data_get_bytes(
                    data,
                    buffer.as_mut_ptr() as *mut std::os::raw::c_void,
                    &mut size,
                )
            } {
                Ok(_) => {
                    eprintln!(
                        "ZLUDA VERBOSE: Successfully copied output data #{} of {} bytes",
                        i, size
                    );

                    // Check if the buffer contains our mock marker
                    let marker = b"ZLUDA_MOCK_RELOCATABLE\0";
                    if buffer.len() > marker.len()
                        && buffer.windows(marker.len()).any(|window| window == marker)
                    {
                        eprintln!("ZLUDA VERBOSE: Found mock relocatable marker in output");
                    }

                    // Output some file structure info to help debug
                    if buffer.len() >= 4 && &buffer[0..4] == b"\x7fELF" {
                        eprintln!("ZLUDA VERBOSE: Output has valid ELF header");
                    } else {
                        eprintln!(
                            "ZLUDA WARNING: Output #{} does not have valid ELF header",
                            i
                        );

                        // Try to display the first 32 bytes for debugging
                        if buffer.len() >= 32 {
                            let prefix: Vec<_> =
                                buffer[0..32].iter().map(|b| format!("{:02x}", b)).collect();
                            eprintln!("ZLUDA DEBUG: First 32 bytes: {}", prefix.join(" "));
                        }
                    }

                    // Release resources
                    unsafe { intel_comgr_release_data(data).ok() };

                    // Return successfully
                    eprintln!(
                        "ZLUDA VERBOSE: Compilation completed successfully, output size: {} bytes",
                        buffer.len()
                    );
                    return Ok(buffer);
                }
                Err(e) => {
                    eprintln!("ZLUDA WARNING: Failed to copy output data #{}: {:?}", i, e);
                }
            }
        } else {
            eprintln!("ZLUDA WARNING: Output data #{} size is 0", i);
        }

        // Release resources and try the next data object
        unsafe { intel_comgr_release_data(data).ok() };
    }

    // If we reach here, we've tried all data objects and none worked
    eprintln!(
        "ZLUDA ERROR: Failed to read any valid output data from {} objects",
        count
    );
    Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR)
}

// Helper implementation for DataSet with Intel support
#[cfg(feature = "intel")]
pub struct DataSet(intel_comgr_data_set_t);
#[cfg(feature = "intel")]
impl DataSet {
    fn new() -> Result<Self, intel_comgr_status_s> {
        let mut data_set = unsafe { mem::zeroed() };
        unsafe { intel_comgr_create_data_set(&mut data_set) }?;
        Ok(Self(data_set))
    }

    fn get(&self) -> intel_comgr_data_set_t {
        self.0
    }
}

// Drop implementation for Intel DataSet
#[cfg(feature = "intel")]
impl Drop for DataSet {
    fn drop(&mut self) {
        unsafe { intel_comgr_release_data_set(self.0).ok() };
    }
}
#[cfg(feature = "intel")]
struct ActionInfo(intel_comgr_action_info_t);

// Implementation of ActionInfo for Intel
#[cfg(feature = "intel")]
impl ActionInfo {
    fn new() -> Result<Self, intel_comgr_status_s> {
        let mut action = unsafe { mem::zeroed() };
        unsafe { intel_comgr_create_action_info(&mut action) }?;
        Ok(Self(action))
    }

    fn set_language(&self, language: intel_comgr_language_t) -> Result<(), intel_comgr_status_s> {
        unsafe { intel_comgr_action_info_set_language(self.0, language) }
    }

    fn set_options<'a>(
        &self,
        options: impl Iterator<Item = &'a CStr>,
    ) -> Result<(), intel_comgr_status_s> {
        let options = options.map(|x| x.as_ptr()).collect::<Vec<_>>();
        unsafe { intel_comgr_action_info_set_option_list(self.0, options.as_ptr(), options.len()) }
    }

    fn get(&self) -> intel_comgr_action_info_t {
        self.0
    }
}

// Drop implementation for Intel ActionInfo
#[cfg(feature = "intel")]
impl Drop for ActionInfo {
    fn drop(&mut self) {
        unsafe { intel_comgr_release_action_info(self.0).ok() };
    }
}

#[cfg(feature = "intel")]
fn do_action(
    data_set: &DataSet,
    action: &ActionInfo,
    kind: intel_comgr_action_kind_s,
) -> Result<DataSet, intel_comgr_status_s> {
    let mut result = DataSet::new()?;
    let action_kind_name = match kind.0 {
        0 => "INTEL_COMGR_ACTION_SOURCE_TO_PREPROCESSED",
        1 => "INTEL_COMGR_ACTION_ADD_PRECOMPILED_HEADERS",
        2 => "INTEL_COMGR_ACTION_COMPILE_SOURCE_TO_BC",
        3 => "INTEL_COMGR_ACTION_ADD_DEVICE_LIBRARIES",
        4 => "INTEL_COMGR_ACTION_LINK_BC_TO_BC",
        5 => "INTEL_COMGR_ACTION_OPTIMIZE_BC_TO_BC",
        6 => "INTEL_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE",
        7 => "INTEL_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY",
        8 => "INTEL_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE",
        9 => "INTEL_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN",
        _ => "UNKNOWN_ACTION",
    };

    eprintln!("ZLUDA VERBOSE: Executing action: {}", action_kind_name);

    let status = unsafe { intel_comgr_do_action(kind, action.0, data_set.0, result.0) };

    match status {
        Ok(_) => {
            eprintln!(
                "ZLUDA VERBOSE: Action {} completed successfully",
                action_kind_name
            );

            // Try to get log data if available
            let mut data_count = 0;
            if let Ok(_) = unsafe { intel_comgr_get_data_count(result.0, &mut data_count) } {
                eprintln!("ZLUDA VERBOSE: Action produced {} data objects", data_count);

                // Log retrieval is simplified as Intel doesn't have the data_get_kind function
                for i in 0..data_count {
                    let mut data = unsafe { mem::zeroed() };
                    if let Ok(_) = unsafe { intel_comgr_get_data(result.0, i, &mut data) } {
                        // Try to get data size - if successful, attempt to read it as log
                        let mut size = 0;
                        if let Ok(_) = unsafe {
                            intel_comgr_data_get_bytes(data, std::ptr::null_mut(), &mut size)
                        } {
                            if size > 0 {
                                let mut content = vec![0u8; size];
                                if let Ok(_) = unsafe {
                                    intel_comgr_data_get_bytes(
                                        data,
                                        content.as_mut_ptr() as *mut std::os::raw::c_void,
                                        &mut size,
                                    )
                                } {
                                    if let Ok(text) = String::from_utf8(content) {
                                        if text.contains("error:") || text.contains("warning:") {
                                            eprintln!(
                                                "ZLUDA COMPILER LOG for {}: \n{}",
                                                action_kind_name, text
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        // Release the data
                        unsafe { intel_comgr_release_data(data).ok() };
                    }
                }
            }

            Ok(result)
        }
        Err(e) => {
            eprintln!(
                "ZLUDA ERROR: Action {} failed with error: {:?}",
                action_kind_name, e
            );

            // Check if we can get any logs even in case of failure
            let mut data_count = 0;
            if let Ok(_) = unsafe { intel_comgr_get_data_count(result.0, &mut data_count) } {
                if data_count > 0 {
                    eprintln!("ZLUDA VERBOSE: Found {} logs for failed action", data_count);
                    for i in 0..data_count {
                        let mut data = unsafe { mem::zeroed() };
                        if let Ok(_) = unsafe { intel_comgr_get_data(result.0, i, &mut data) } {
                            let mut size = 0;
                            if let Ok(_) = unsafe {
                                intel_comgr_data_get_bytes(data, std::ptr::null_mut(), &mut size)
                            } {
                                if size > 0 {
                                    let mut content = vec![0u8; size];
                                    if let Ok(_) = unsafe {
                                        intel_comgr_data_get_bytes(
                                            data,
                                            content.as_mut_ptr() as *mut std::os::raw::c_void,
                                            &mut size,
                                        )
                                    } {
                                        if let Ok(text) = String::from_utf8(content) {
                                            eprintln!(
                                                "ZLUDA FAILURE LOG for {}: \n{}",
                                                action_kind_name, text
                                            );
                                        }
                                    }
                                }
                            }
                            unsafe { intel_comgr_release_data(data).ok() };
                        }
                    }
                }
            }

            Err(e)
        }
    }
}

#[cfg(feature = "gemmini")]
pub fn compile_bitcode(
    gcn_arch: &CStr,
    main_buffer: &[u8],
    ptx_impl: &[u8],
) -> Result<Vec<u8>, gemmini_comgr_status_s> {
    eprintln!("ZLUDA DEBUG: Compiling bitcode for Gemmini accelerator");
    eprintln!(
        "ZLUDA DEBUG: Main buffer size: {} bytes, PTX impl size: {} bytes",
        main_buffer.len(),
        ptx_impl.len()
    );
    eprintln!(
        "ZLUDA DEBUG: Target architecture: {:?}",
        gcn_arch.to_string_lossy()
    );

    // Create input data set
    let mut input_data_set = unsafe { mem::zeroed() };
    gemmini_comgr_create_data_set(&mut input_data_set)?;

    // Create main bitcode data
    let mut main_data = unsafe { mem::zeroed() };
    gemmini_comgr_create_data(
        gemmini_comgr_data_kind_s::GEMMINI_COMGR_DATA_KIND_BC,
        &mut main_data,
    )?;
    gemmini_comgr_data_set_bytes(
        main_data,
        main_buffer.as_ptr() as *const std::os::raw::c_void,
        main_buffer.len(),
    )?;
    gemmini_comgr_data_set_name(main_data, c"main.bc".as_ptr())?;
    gemmini_comgr_data_set_add(input_data_set, main_data)?;

    // If PTX impl is provided, add it too
    if !ptx_impl.is_empty() {
        let mut ptx_data = unsafe { mem::zeroed() };
        gemmini_comgr_create_data(
            gemmini_comgr_data_kind_s::GEMMINI_COMGR_DATA_KIND_BC,
            &mut ptx_data,
        )?;
        gemmini_comgr_data_set_bytes(
            ptx_data,
            ptx_impl.as_ptr() as *const std::os::raw::c_void,
            ptx_impl.len(),
        )?;
        gemmini_comgr_data_set_name(ptx_data, c"ptx_impl.bc".as_ptr())?;
        gemmini_comgr_data_set_add(input_data_set, ptx_data)?;
    }

    // Create action info
    let mut action_info = unsafe { mem::zeroed() };
    gemmini_comgr_create_action_info(&mut action_info)?;

    // Set language to LLVM IR
    gemmini_comgr_action_info_set_language(
        action_info,
        gemmini_comgr_language_s::GEMMINI_COMGR_LANGUAGE_LLVM_IR,
    )?;

    // Create output data set
    let mut output_data_set = unsafe { mem::zeroed() };
    gemmini_comgr_create_data_set(&mut output_data_set)?;

    // First link all bitcode together if needed
    let linked_data_set = if !ptx_impl.is_empty() {
        eprintln!("ZLUDA DEBUG: Linking bitcode modules");
        let mut linked_set = unsafe { mem::zeroed() };
        gemmini_comgr_create_data_set(&mut linked_set)?;

        gemmini_comgr_do_action(
            gemmini_comgr_action_kind_s::GEMMINI_COMGR_ACTION_LINK_BC_TO_BC,
            action_info,
            input_data_set,
            linked_set,
        )?;

        linked_set
    } else {
        input_data_set
    };

    // Optimize the bitcode
    eprintln!("ZLUDA DEBUG: Optimizing bitcode");
    let mut optimized_set = unsafe { mem::zeroed() };
    gemmini_comgr_create_data_set(&mut optimized_set)?;

    gemmini_comgr_do_action(
        gemmini_comgr_action_kind_s::GEMMINI_COMGR_ACTION_OPTIMIZE_BC_TO_BC,
        action_info,
        linked_data_set,
        optimized_set,
    )?;

    // Generate executable
    eprintln!("ZLUDA DEBUG: Generating Gemmini executable");
    gemmini_comgr_do_action(
        gemmini_comgr_action_kind_s::GEMMINI_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
        action_info,
        optimized_set,
        output_data_set,
    )?;

    // Get the output data
    let mut count = 0;
    gemmini_comgr_get_data_count(output_data_set, &mut count)?;

    if count == 0 {
        eprintln!("ZLUDA ERROR: No output generated");
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR);
    }

    // Get first output data
    let mut output_data = unsafe { mem::zeroed() };
    gemmini_comgr_get_data(output_data_set, 0, &mut output_data)?;

    // Get size
    let mut size = 0;
    gemmini_comgr_data_get_bytes(output_data, std::ptr::null_mut(), &mut size)?;

    // Read content
    let mut result = vec![0u8; size];
    gemmini_comgr_data_get_bytes(
        output_data,
        result.as_mut_ptr() as *mut std::os::raw::c_void,
        &mut size,
    )?;

    // Cleanup
    gemmini_comgr_release_data(output_data)?;
    gemmini_comgr_release_data_set(output_data_set)?;
    if !ptx_impl.is_empty() {
        gemmini_comgr_release_data_set(linked_data_set)?;
    }
    gemmini_comgr_release_data_set(optimized_set)?;
    gemmini_comgr_release_action_info(action_info)?;

    eprintln!(
        "ZLUDA DEBUG: Gemmini compilation complete, output size: {} bytes",
        result.len()
    );

    Ok(result)
}

/// CuTile bytecode to TOSA transformation and execution pipeline
///
/// This function takes CuTile bytecode, transforms it to TOSA,
/// and lowers it to the specified target backend.
#[cfg(feature = "cutile")]
pub fn compile_cutile_bytecode(
    target: &CStr,
    cutile_bytecode: &[u8],
) -> Result<Vec<u8>, cutile_comgr_status_s> {
    eprintln!("ZLUDA DEBUG: Compiling CuTile bytecode for target: {:?}",
              target.to_string_lossy());
    eprintln!("ZLUDA DEBUG: Bytecode size: {} bytes", cutile_bytecode.len());

    // Determine target backend from target string
    let target_str = target.to_string_lossy();
    let target_backend = if target_str.contains("tenstorrent") || target_str.contains("tt") {
        cutile_comgr_target_s::CUTILE_COMGR_TARGET_TENSTORRENT
    } else if target_str.contains("intel") || target_str.contains("xe") {
        cutile_comgr_target_s::CUTILE_COMGR_TARGET_INTEL
    } else if target_str.contains("amd") || target_str.contains("gfx") {
        cutile_comgr_target_s::CUTILE_COMGR_TARGET_AMD
    } else if target_str.contains("gemmini") || target_str.contains("riscv") {
        cutile_comgr_target_s::CUTILE_COMGR_TARGET_GEMMINI
    } else {
        cutile_comgr_target_s::CUTILE_COMGR_TARGET_CPU
    };

    // Create input data
    let mut input_data = cutile_comgr_data_t { handle: 0 };
    cutile_comgr_create_data(
        cutile_comgr_data_kind_s::CUTILE_COMGR_DATA_KIND_BYTECODE,
        &mut input_data,
    )?;
    cutile_comgr_data_set_bytes(
        input_data,
        cutile_bytecode.as_ptr() as *const std::os::raw::c_void,
        cutile_bytecode.len(),
    )?;
    cutile_comgr_data_set_name(input_data, c"input.ctir".as_ptr())?;

    // Create input data set
    let mut input_set = cutile_comgr_data_set_t { handle: 0 };
    cutile_comgr_create_data_set(&mut input_set)?;
    cutile_comgr_data_set_add(input_set, input_data)?;

    // Create action info with target
    let mut action_info = cutile_comgr_action_info_t { handle: 0 };
    cutile_comgr_create_action_info(&mut action_info)?;
    cutile_comgr_action_info_set_target(action_info, target_backend)?;

    // Create output data set
    let mut output_set = cutile_comgr_data_set_t { handle: 0 };
    cutile_comgr_create_data_set(&mut output_set)?;

    // Execute full pipeline: bytecode -> CuTile MLIR -> TOSA -> target executable
    cutile_comgr_do_action(
        cutile_comgr_action_kind_s::CUTILE_COMGR_ACTION_FULL_PIPELINE,
        action_info,
        input_set,
        output_set,
    )?;

    // Get output data count
    let mut count = 0;
    cutile_comgr_get_data_count(output_set, &mut count)?;

    if count == 0 {
        eprintln!("ZLUDA ERROR: No output generated from CuTile pipeline");
        return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR);
    }

    // Get first output data
    let mut output_data = cutile_comgr_data_t { handle: 0 };
    cutile_comgr_get_data(output_set, 0, &mut output_data)?;

    // Get size
    let mut size = 0;
    cutile_comgr_data_get_bytes(output_data, std::ptr::null_mut(), &mut size)?;

    // Read content
    let mut result = vec![0u8; size];
    cutile_comgr_data_get_bytes(
        output_data,
        result.as_mut_ptr() as *mut std::os::raw::c_void,
        &mut size,
    )?;

    // Cleanup
    cutile_comgr_release_data(output_data)?;
    cutile_comgr_release_data(input_data)?;
    cutile_comgr_release_data_set(output_set)?;
    cutile_comgr_release_data_set(input_set)?;
    cutile_comgr_release_action_info(action_info)?;

    eprintln!(
        "ZLUDA DEBUG: CuTile pipeline complete, output size: {} bytes",
        result.len()
    );

    Ok(result)
}

/// Transform CuTile MLIR text to TOSA MLIR text
#[cfg(feature = "cutile")]
pub fn transform_cutile_to_tosa(
    cutile_mlir: &[u8],
) -> Result<Vec<u8>, cutile_comgr_status_s> {
    eprintln!("ZLUDA DEBUG: Transforming CuTile MLIR to TOSA");

    // Create input data
    let mut input_data = cutile_comgr_data_t { handle: 0 };
    cutile_comgr_create_data(
        cutile_comgr_data_kind_s::CUTILE_COMGR_DATA_KIND_MLIR_CUTILE,
        &mut input_data,
    )?;
    cutile_comgr_data_set_bytes(
        input_data,
        cutile_mlir.as_ptr() as *const std::os::raw::c_void,
        cutile_mlir.len(),
    )?;
    cutile_comgr_data_set_name(input_data, c"input.mlir".as_ptr())?;

    // Create input data set
    let mut input_set = cutile_comgr_data_set_t { handle: 0 };
    cutile_comgr_create_data_set(&mut input_set)?;
    cutile_comgr_data_set_add(input_set, input_data)?;

    // Create action info
    let mut action_info = cutile_comgr_action_info_t { handle: 0 };
    cutile_comgr_create_action_info(&mut action_info)?;

    // Create output data set
    let mut output_set = cutile_comgr_data_set_t { handle: 0 };
    cutile_comgr_create_data_set(&mut output_set)?;

    // Execute CuTile to TOSA transformation
    cutile_comgr_do_action(
        cutile_comgr_action_kind_s::CUTILE_COMGR_ACTION_CUTILE_TO_TOSA,
        action_info,
        input_set,
        output_set,
    )?;

    // Get output
    let mut count = 0;
    cutile_comgr_get_data_count(output_set, &mut count)?;

    if count == 0 {
        return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR);
    }

    let mut output_data = cutile_comgr_data_t { handle: 0 };
    cutile_comgr_get_data(output_set, 0, &mut output_data)?;

    let mut size = 0;
    cutile_comgr_data_get_bytes(output_data, std::ptr::null_mut(), &mut size)?;

    let mut result = vec![0u8; size];
    cutile_comgr_data_get_bytes(
        output_data,
        result.as_mut_ptr() as *mut std::os::raw::c_void,
        &mut size,
    )?;

    // Cleanup
    cutile_comgr_release_data(output_data)?;
    cutile_comgr_release_data(input_data)?;
    cutile_comgr_release_data_set(output_set)?;
    cutile_comgr_release_data_set(input_set)?;
    cutile_comgr_release_action_info(action_info)?;

    Ok(result)
}
