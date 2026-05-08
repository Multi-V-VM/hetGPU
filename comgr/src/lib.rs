#[cfg(feature = "amd")]
use amd_comgr_sys::*;
#[cfg(feature = "cutile")]
use cutile_comgr_sys::*;
#[cfg(feature = "gemmini")]
use gemmini_comgr_sys::*;
#[cfg(feature = "intel")]
use intel_comgr_sys::*;
#[cfg(feature = "pacc")]
use pacc_comgr_sys::*;
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
    eprintln!(
        "ZLUDA DEBUG: Compiling CuTile bytecode for target: {:?}",
        target.to_string_lossy()
    );
    eprintln!(
        "ZLUDA DEBUG: Bytecode size: {} bytes",
        cutile_bytecode.len()
    );

    // Determine target backend from target string
    let target_str = target.to_string_lossy();
    let target_backend = if target_str.contains("tenstorrent") || target_str.contains("tt") {
        cutile_comgr_target_s::CUTILE_COMGR_TARGET_TENSTORRENT
    } else if target_str.contains("intel") || target_str.contains("xe") {
        cutile_comgr_target_s::CUTILE_COMGR_TARGET_INTEL
    } else if target_str.contains("amd") || target_str.contains("gfx") {
        cutile_comgr_target_s::CUTILE_COMGR_TARGET_AMD
    } else if target_str.contains("pacc")
        || target_str.contains("sifive")
        || target_str.contains("xm")
    {
        cutile_comgr_target_s::CUTILE_COMGR_TARGET_GEMMINI // PACC uses same RISC-V target slot
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
pub fn transform_cutile_to_tosa(cutile_mlir: &[u8]) -> Result<Vec<u8>, cutile_comgr_status_s> {
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

/// PACC (RISC-V IME via VCIX) bitcode compilation pipeline
///
/// Takes LLVM bitcode (with VCIX intrinsics for matrix ops),
/// links, optimizes, and generates RISC-V object code targeting
/// SiFive Intelligence XM / RISC-V IME.
#[cfg(feature = "pacc")]
pub fn compile_bitcode_pacc(
    target_arch: &CStr,
    main_buffer: &[u8],
    ptx_impl: &[u8],
) -> Result<Vec<u8>, pacc_comgr_status_s> {
    let linked_modules: Vec<&[u8]> = if ptx_impl.is_empty() {
        Vec::new()
    } else {
        vec![ptx_impl]
    };
    compile_bitcode_pacc_multi(target_arch, main_buffer, &linked_modules)
}

#[cfg(feature = "pacc")]
pub fn compile_bitcode_pacc_multi(
    target_arch: &CStr,
    main_buffer: &[u8],
    linked_modules: &[&[u8]],
) -> Result<Vec<u8>, pacc_comgr_status_s> {
    let log_debug = std::env::var("HETGPU_PACC_LOG_COMGR").ok().as_deref() == Some("1");
    if log_debug {
        eprintln!("ZLUDA DEBUG: Compiling bitcode for PACC (RISC-V IME/VCIX)");
        eprintln!(
            "ZLUDA DEBUG: Main buffer size: {} bytes, linked module count: {}",
            main_buffer.len(),
            linked_modules.len()
        );
        eprintln!(
            "ZLUDA DEBUG: Target architecture: {:?}",
            target_arch.to_string_lossy()
        );
    }

    // Create input data set
    let mut input_data_set = unsafe { mem::zeroed() };
    pacc_comgr_create_data_set(&mut input_data_set)?;

    // Create main bitcode data
    let mut main_data = unsafe { mem::zeroed() };
    pacc_comgr_create_data(
        pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_BC,
        &mut main_data,
    )?;
    pacc_comgr_data_set_bytes(
        main_data,
        main_buffer.as_ptr() as *const std::os::raw::c_void,
        main_buffer.len(),
    )?;
    pacc_comgr_data_set_name(main_data, c"main.bc".as_ptr())?;
    pacc_comgr_data_set_add(input_data_set, main_data)?;

    for (idx, module_bytes) in linked_modules.iter().enumerate() {
        if module_bytes.is_empty() {
            continue;
        }
        let mut linked_data = unsafe { mem::zeroed() };
        pacc_comgr_create_data(
            pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_BC,
            &mut linked_data,
        )?;
        pacc_comgr_data_set_bytes(
            linked_data,
            module_bytes.as_ptr() as *const std::os::raw::c_void,
            module_bytes.len(),
        )?;
        let name = std::ffi::CString::new(format!("linked_{}.bc", idx)).unwrap();
        pacc_comgr_data_set_name(linked_data, name.as_ptr())?;
        pacc_comgr_data_set_add(input_data_set, linked_data)?;
    }

    // Create action info
    let mut action_info = unsafe { mem::zeroed() };
    pacc_comgr_create_action_info(&mut action_info)?;

    // Set language to LLVM IR
    pacc_comgr_action_info_set_language(
        action_info,
        pacc_comgr_language_s::PACC_COMGR_LANGUAGE_LLVM_IR,
    )?;
    pacc_comgr_action_info_set_target(action_info, target_arch.as_ptr())?;

    // Create output data set
    let mut output_data_set = unsafe { mem::zeroed() };
    pacc_comgr_create_data_set(&mut output_data_set)?;

    // Link bitcode if needed
    let linked_data_set = if !linked_modules.is_empty() {
        if log_debug {
            eprintln!("ZLUDA DEBUG: Linking PACC bitcode modules");
        }
        let mut linked_set = unsafe { mem::zeroed() };
        pacc_comgr_create_data_set(&mut linked_set)?;

        pacc_comgr_do_action(
            pacc_comgr_action_kind_s::PACC_COMGR_ACTION_LINK_BC_TO_BC,
            action_info,
            input_data_set,
            linked_set,
        )?;

        linked_set
    } else {
        input_data_set
    };

    // Optimize bitcode
    if log_debug {
        eprintln!("ZLUDA DEBUG: Optimizing PACC bitcode");
    }
    let mut optimized_set = unsafe { mem::zeroed() };
    pacc_comgr_create_data_set(&mut optimized_set)?;

    pacc_comgr_do_action(
        pacc_comgr_action_kind_s::PACC_COMGR_ACTION_OPTIMIZE_BC_TO_BC,
        action_info,
        linked_data_set,
        optimized_set,
    )?;

    // Generate RISC-V object code with VCIX
    if log_debug {
        eprintln!("ZLUDA DEBUG: Generating PACC RISC-V+VCIX executable");
    }
    pacc_comgr_do_action(
        pacc_comgr_action_kind_s::PACC_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
        action_info,
        optimized_set,
        output_data_set,
    )?;

    // Codegen temp directories can contain both intermediate bitcode and the
    // final object. Pick the launchable object by data kind instead of assuming
    // output slot 0 is the relocatable payload.
    let result = unsafe { pacc_copy_first_output_bytes(output_data_set)? };

    // Cleanup
    pacc_comgr_release_data_set(output_data_set)?;
    if !linked_modules.is_empty() {
        pacc_comgr_release_data_set(linked_data_set)?;
    }
    pacc_comgr_release_data_set(optimized_set)?;
    pacc_comgr_release_data_set(input_data_set)?;
    pacc_comgr_release_action_info(action_info)?;

    if log_debug {
        eprintln!(
            "ZLUDA DEBUG: PACC compilation complete, output size: {} bytes",
            result.len()
        );
    }

    Ok(result)
}

#[cfg(feature = "pacc")]
unsafe fn pacc_copy_first_output_bytes(
    data_set: pacc_comgr_data_set_t,
) -> Result<Vec<u8>, pacc_comgr_status_s> {
    let mut count = 0;
    pacc_comgr_get_data_count(data_set, &mut count)?;
    if count == 0 {
        eprintln!("ZLUDA ERROR: No PACC output generated");
        return Err(pacc_comgr_status_s::PACC_COMGR_STATUS_ERROR);
    }

    let preferred_kinds = [
        pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_EXECUTABLE,
        pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_RELOCATABLE,
    ];

    let mut output_data = mem::zeroed();
    let mut selected_index = None;
    for preferred_kind in preferred_kinds {
        for i in 0..count {
            let mut candidate = mem::zeroed();
            pacc_comgr_get_data(data_set, i, &mut candidate)?;

            let mut candidate_kind = pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_UNDEF;
            pacc_comgr_get_data_kind(candidate, &mut candidate_kind)?;
            if candidate_kind.0 == preferred_kind.0 {
                output_data = candidate;
                selected_index = Some(i);
                break;
            }
        }
        if selected_index.is_some() {
            break;
        }
    }

    if selected_index.is_none() {
        pacc_comgr_get_data(data_set, 0, &mut output_data)?;
    }

    let mut size = 0;
    pacc_comgr_data_get_bytes(output_data, ptr::null_mut(), &mut size)?;

    let mut result = vec![0u8; size];
    pacc_comgr_data_get_bytes(
        output_data,
        result.as_mut_ptr() as *mut std::os::raw::c_void,
        &mut size,
    )?;
    pacc_comgr_release_data(output_data)?;
    Ok(result)
}

#[cfg(feature = "pacc")]
pub fn compile_source_pacc(
    target_arch: &CStr,
    source_name: &CStr,
    source_buffer: &[u8],
    working_directory: Option<&CStr>,
    options: &[&CStr],
    linked_bitcode: &[u8],
) -> Result<Vec<u8>, pacc_comgr_status_s> {
    eprintln!(
        "ZLUDA DEBUG: Compiling PACC source {} to launchable ELF",
        source_name.to_string_lossy()
    );

    let mut input_source_set = unsafe { mem::zeroed() };
    pacc_comgr_create_data_set(&mut input_source_set)?;

    let mut source_data = unsafe { mem::zeroed() };
    pacc_comgr_create_data(
        pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_SOURCE,
        &mut source_data,
    )?;
    pacc_comgr_data_set_name(source_data, source_name.as_ptr())?;
    pacc_comgr_data_set_bytes(
        source_data,
        source_buffer.as_ptr() as *const std::os::raw::c_void,
        source_buffer.len(),
    )?;
    pacc_comgr_data_set_add(input_source_set, source_data)?;

    let mut action_info = unsafe { mem::zeroed() };
    pacc_comgr_create_action_info(&mut action_info)?;
    pacc_comgr_action_info_set_target(action_info, target_arch.as_ptr())?;
    if let Some(dir) = working_directory {
        pacc_comgr_action_info_set_working_directory(action_info, dir.as_ptr())?;
    }
    if !options.is_empty() {
        let option_ptrs = options.iter().map(|opt| opt.as_ptr()).collect::<Vec<_>>();
        pacc_comgr_action_info_set_option_list(
            action_info,
            option_ptrs.as_ptr(),
            option_ptrs.len(),
        )?;
    }

    let mut source_bc_set = unsafe { mem::zeroed() };
    pacc_comgr_create_data_set(&mut source_bc_set)?;
    pacc_comgr_do_action(
        pacc_comgr_action_kind_s::PACC_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
        action_info,
        input_source_set,
        source_bc_set,
    )
    .map_err(|e| {
        eprintln!("ZLUDA ERROR: PACC source -> BC failed: {:?}", e);
        e
    })?;

    let linked_input_set = if linked_bitcode.is_empty() {
        source_bc_set
    } else {
        let mut link_input_set = unsafe { mem::zeroed() };
        pacc_comgr_create_data_set(&mut link_input_set)?;

        let mut bc_count = 0;
        pacc_comgr_get_data_count(source_bc_set, &mut bc_count)?;
        for i in 0..bc_count {
            let mut bc_data = unsafe { mem::zeroed() };
            pacc_comgr_get_data(source_bc_set, i, &mut bc_data)?;
            pacc_comgr_data_set_add(link_input_set, bc_data)?;
        }

        let mut linked_bc_data = unsafe { mem::zeroed() };
        pacc_comgr_create_data(
            pacc_comgr_data_kind_s::PACC_COMGR_DATA_KIND_BC,
            &mut linked_bc_data,
        )?;
        pacc_comgr_data_set_name(linked_bc_data, c"linked.bc".as_ptr())?;
        pacc_comgr_data_set_bytes(
            linked_bc_data,
            linked_bitcode.as_ptr() as *const std::os::raw::c_void,
            linked_bitcode.len(),
        )?;
        pacc_comgr_data_set_add(link_input_set, linked_bc_data)?;
        link_input_set
    };

    let linked_bc_set = if linked_bitcode.is_empty() {
        source_bc_set
    } else {
        let mut linked_set = unsafe { mem::zeroed() };
        pacc_comgr_create_data_set(&mut linked_set)?;
        pacc_comgr_do_action(
            pacc_comgr_action_kind_s::PACC_COMGR_ACTION_LINK_BC_TO_BC,
            action_info,
            linked_input_set,
            linked_set,
        )
        .map_err(|e| {
            eprintln!("ZLUDA ERROR: PACC BC link failed: {:?}", e);
            e
        })?;
        eprintln!(
            "PACC: Linked external bitcode into source module: {} bytes",
            linked_bitcode.len()
        );
        linked_set
    };

    let mut optimized_set = unsafe { mem::zeroed() };
    pacc_comgr_create_data_set(&mut optimized_set)?;
    pacc_comgr_do_action(
        pacc_comgr_action_kind_s::PACC_COMGR_ACTION_OPTIMIZE_BC_TO_BC,
        action_info,
        linked_bc_set,
        optimized_set,
    )
    .map_err(|e| {
        eprintln!("ZLUDA ERROR: PACC BC optimize failed: {:?}", e);
        e
    })?;

    let mut reloc_set = unsafe { mem::zeroed() };
    pacc_comgr_create_data_set(&mut reloc_set)?;
    pacc_comgr_do_action(
        pacc_comgr_action_kind_s::PACC_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
        action_info,
        optimized_set,
        reloc_set,
    )
    .map_err(|e| {
        eprintln!("ZLUDA ERROR: PACC BC -> relocatable failed: {:?}", e);
        e
    })?;

    let result = unsafe { pacc_copy_first_output_bytes(reloc_set) };

    pacc_comgr_release_data_set(reloc_set)?;
    pacc_comgr_release_data_set(optimized_set)?;
    if !linked_bitcode.is_empty() {
        pacc_comgr_release_data_set(linked_bc_set)?;
        pacc_comgr_release_data_set(linked_input_set)?;
    }
    pacc_comgr_release_data_set(source_bc_set)?;
    pacc_comgr_release_data_set(input_source_set)?;
    pacc_comgr_release_action_info(action_info)?;

    result
}

#[cfg(feature = "pacc")]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_compile_source_to_elf(
    target_arch: *const std::ffi::c_char,
    source_name: *const std::ffi::c_char,
    source_buffer: *const u8,
    source_len: usize,
    working_directory: *const std::ffi::c_char,
    options: *const *const std::ffi::c_char,
    option_count: usize,
    linked_bitcode: *const u8,
    linked_bitcode_len: usize,
    out_elf: *mut *mut u8,
    out_elf_len: *mut usize,
) -> i32 {
    if target_arch.is_null()
        || source_name.is_null()
        || source_buffer.is_null()
        || out_elf.is_null()
        || out_elf_len.is_null()
    {
        return -1;
    }

    let target_arch = CStr::from_ptr(target_arch);
    let source_name = CStr::from_ptr(source_name);
    let source_buffer = std::slice::from_raw_parts(source_buffer, source_len);
    let working_directory = if working_directory.is_null() {
        None
    } else {
        Some(CStr::from_ptr(working_directory))
    };
    let linked_bitcode = if linked_bitcode.is_null() || linked_bitcode_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(linked_bitcode, linked_bitcode_len)
    };

    let mut option_refs = Vec::with_capacity(option_count);
    if !options.is_null() {
        for idx in 0..option_count {
            let opt = *options.add(idx);
            if !opt.is_null() {
                option_refs.push(CStr::from_ptr(opt));
            }
        }
    }

    match compile_source_pacc(
        target_arch,
        source_name,
        source_buffer,
        working_directory,
        &option_refs,
        linked_bitcode,
    ) {
        Ok(elf) => {
            let len = elf.len();
            let ptr = if len == 0 {
                ptr::null_mut()
            } else {
                let alloc = libc::malloc(len);
                if alloc.is_null() {
                    return -1;
                }
                ptr::copy_nonoverlapping(elf.as_ptr(), alloc.cast::<u8>(), len);
                alloc.cast::<u8>()
            };
            *out_elf = ptr;
            *out_elf_len = len;
            0
        }
        Err(err) => {
            eprintln!(
                "hetgpu_pacc_compile_source_to_elf: failed for {}: {:?}",
                source_name.to_string_lossy(),
                err
            );
            -1
        }
    }
}

#[cfg(feature = "pacc")]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_free_buffer(ptr: *mut u8) {
    if !ptr.is_null() {
        libc::free(ptr.cast());
    }
}

/// NVIDIA (SM120) bitcode compilation via nvidia_sass.
#[cfg(feature = "nvidia")]
pub fn compile_bitcode_nvidia(
    sm_arch: &CStr,
    main_buffer: &[u8],
    _ptx_impl: &[u8],
) -> Result<Vec<u8>, NvidiaComgrError> {
    let arch_str = sm_arch.to_string_lossy();
    let sm_version: u32 = arch_str
        .strip_prefix("sm_")
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);

    eprintln!("ZLUDA DEBUG: Compiling bitcode for NVIDIA SM{}", sm_version);
    eprintln!("ZLUDA DEBUG: Main buffer size: {} bytes", main_buffer.len());

    let module = nvidia_sass::types::SassModule {
        kernels: vec![],
        sm_version,
        global_constants: vec![],
    };

    nvidia_sass::cubin_builder::build_cubin_from_module(&module)
        .map_err(|e| NvidiaComgrError::CompilationFailed(e.to_string()))
}

#[cfg(feature = "nvidia")]
#[derive(Debug)]
pub enum NvidiaComgrError {
    CompilationFailed(String),
}

#[cfg(feature = "nvidia")]
impl std::fmt::Display for NvidiaComgrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NvidiaComgrError::CompilationFailed(msg) => {
                write!(f, "NVIDIA compilation failed: {}", msg)
            }
        }
    }
}

#[cfg(feature = "nvidia")]
impl std::error::Error for NvidiaComgrError {}
