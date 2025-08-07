mod command_wrapper;

use std::ffi::{CStr, CString, c_char, c_int, c_uint};
use std::num::NonZeroU32;
use std::os::raw;
use std::process::Command;

// Version constants similar to AMD's API
pub const GEMMINI_COMGR_INTERFACE_VERSION_MAJOR: u32 = 1;
pub const GEMMINI_COMGR_INTERFACE_VERSION_MINOR: u32 = 0;

// Status types
#[derive(Debug, Clone, Copy)]
pub struct gemmini_comgr_status_s(pub NonZeroU32);
pub type gemmini_comgr_status_t = Result<(), self::gemmini_comgr_status_s>;

// Status constants
impl gemmini_comgr_status_s {
    pub const GEMMINI_COMGR_STATUS_SUCCESS: Result<(), gemmini_comgr_status_s> = Ok(());

    pub const GEMMINI_COMGR_STATUS_ERROR: gemmini_comgr_status_s =
        gemmini_comgr_status_s(unsafe { NonZeroU32::new_unchecked(1) });

    pub const GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT: gemmini_comgr_status_s =
        gemmini_comgr_status_s(unsafe { NonZeroU32::new_unchecked(2) });

    pub const GEMMINI_COMGR_STATUS_ERROR_OUT_OF_RESOURCES: gemmini_comgr_status_s =
        gemmini_comgr_status_s(unsafe { NonZeroU32::new_unchecked(3) });
}

// Language types similar to AMD
#[derive(Default, Clone, Copy)]
pub struct gemmini_comgr_language_s(pub c_uint);
pub type gemmini_comgr_language_t = gemmini_comgr_language_s;

impl gemmini_comgr_language_s {
    pub const GEMMINI_COMGR_LANGUAGE_NONE: gemmini_comgr_language_s = gemmini_comgr_language_s(0);
    pub const GEMMINI_COMGR_LANGUAGE_OPENCL_1_2: gemmini_comgr_language_s =
        gemmini_comgr_language_s(1);
    pub const GEMMINI_COMGR_LANGUAGE_OPENCL_2_0: gemmini_comgr_language_s =
        gemmini_comgr_language_s(2);
    pub const GEMMINI_COMGR_LANGUAGE_SYCL: gemmini_comgr_language_s = gemmini_comgr_language_s(3);
    pub const GEMMINI_COMGR_LANGUAGE_LLVM_IR: gemmini_comgr_language_s =
        gemmini_comgr_language_s(4);
    pub const GEMMINI_COMGR_LANGUAGE_LAST: gemmini_comgr_language_s = gemmini_comgr_language_s(4);
}

// Data kinds similar to AMD
#[derive(Default, Clone, Copy)]
pub struct gemmini_comgr_data_kind_s(pub c_uint);
pub type gemmini_comgr_data_kind_t = gemmini_comgr_data_kind_s;

impl gemmini_comgr_data_kind_s {
    pub const GEMMINI_COMGR_DATA_KIND_UNDEF: gemmini_comgr_data_kind_s =
        gemmini_comgr_data_kind_s(0);
    pub const GEMMINI_COMGR_DATA_KIND_SOURCE: gemmini_comgr_data_kind_s =
        gemmini_comgr_data_kind_s(1);
    pub const GEMMINI_COMGR_DATA_KIND_INCLUDE: gemmini_comgr_data_kind_s =
        gemmini_comgr_data_kind_s(2);
    pub const GEMMINI_COMGR_DATA_KIND_PRECOMPILED_HEADER: gemmini_comgr_data_kind_s =
        gemmini_comgr_data_kind_s(3);
    pub const GEMMINI_COMGR_DATA_KIND_DIAGNOSTIC: gemmini_comgr_data_kind_s =
        gemmini_comgr_data_kind_s(4);
    pub const GEMMINI_COMGR_DATA_KIND_LOG: gemmini_comgr_data_kind_s = gemmini_comgr_data_kind_s(5);
    pub const GEMMINI_COMGR_DATA_KIND_BC: gemmini_comgr_data_kind_s = gemmini_comgr_data_kind_s(6);
    pub const GEMMINI_COMGR_DATA_KIND_RELOCATABLE: gemmini_comgr_data_kind_s =
        gemmini_comgr_data_kind_s(7);
    pub const GEMMINI_COMGR_DATA_KIND_EXECUTABLE: gemmini_comgr_data_kind_s =
        gemmini_comgr_data_kind_s(8);
    pub const GEMMINI_COMGR_DATA_KIND_BYTES: gemmini_comgr_data_kind_s =
        gemmini_comgr_data_kind_s(9);
    pub const GEMMINI_COMGR_DATA_KIND_FATBIN: gemmini_comgr_data_kind_s =
        gemmini_comgr_data_kind_s(16);
    pub const GEMMINI_COMGR_DATA_KIND_LAST: gemmini_comgr_data_kind_s =
        gemmini_comgr_data_kind_s(16);
}

// Data structures similar to AMD
#[derive(Default, Clone, Copy)]
pub struct gemmini_comgr_data_s {
    pub handle: u64,
}
pub type gemmini_comgr_data_t = gemmini_comgr_data_s;

#[derive(Default, Clone, Copy)]
pub struct gemmini_comgr_data_set_s {
    pub handle: u64,
}
pub type gemmini_comgr_data_set_t = gemmini_comgr_data_set_s;
#[derive(Default, Clone, Copy)]
pub struct gemmini_comgr_action_info_s {
    pub handle: u64,
}
pub type gemmini_comgr_action_info_t = gemmini_comgr_action_info_s;

pub struct gemmini_comgr_metadata_node_s {
    pub handle: u64,
}
pub type gemmini_comgr_metadata_node_t = gemmini_comgr_metadata_node_s;

pub struct gemmini_comgr_symbol_s {
    pub handle: u64,
}
pub type gemmini_comgr_symbol_t = gemmini_comgr_symbol_s;

// Define metadata kind constants
pub struct gemmini_comgr_metadata_kind_s(pub c_uint);
pub type gemmini_comgr_metadata_kind_t = gemmini_comgr_metadata_kind_s;

impl gemmini_comgr_metadata_kind_s {
    pub const GEMMINI_COMGR_METADATA_KIND_NULL: gemmini_comgr_metadata_kind_s =
        gemmini_comgr_metadata_kind_s(0);
    pub const GEMMINI_COMGR_METADATA_KIND_STRING: gemmini_comgr_metadata_kind_s =
        gemmini_comgr_metadata_kind_s(1);
    pub const GEMMINI_COMGR_METADATA_KIND_MAP: gemmini_comgr_metadata_kind_s =
        gemmini_comgr_metadata_kind_s(2);
    pub const GEMMINI_COMGR_METADATA_KIND_LIST: gemmini_comgr_metadata_kind_s =
        gemmini_comgr_metadata_kind_s(3);
    pub const GEMMINI_COMGR_METADATA_KIND_LAST: gemmini_comgr_metadata_kind_s =
        gemmini_comgr_metadata_kind_s(3);
}

// Action kinds similar to AMD
#[derive(Default, Clone, Copy)]
pub struct gemmini_comgr_action_kind_s(pub c_uint);
pub type gemmini_comgr_action_kind_t = gemmini_comgr_action_kind_s;

impl gemmini_comgr_action_kind_s {
    pub const GEMMINI_COMGR_ACTION_SOURCE_TO_PREPROCESSOR: gemmini_comgr_action_kind_s =
        gemmini_comgr_action_kind_s(0);
    pub const GEMMINI_COMGR_ACTION_ADD_PRECOMPILED_HEADERS: gemmini_comgr_action_kind_s =
        gemmini_comgr_action_kind_s(1);
    pub const GEMMINI_COMGR_ACTION_COMPILE_SOURCE_TO_BC: gemmini_comgr_action_kind_s =
        gemmini_comgr_action_kind_s(2);
    pub const GEMMINI_COMGR_ACTION_ADD_DEVICE_LIBRARIES: gemmini_comgr_action_kind_s =
        gemmini_comgr_action_kind_s(3);
    pub const GEMMINI_COMGR_ACTION_LINK_BC_TO_BC: gemmini_comgr_action_kind_s =
        gemmini_comgr_action_kind_s(4);
    pub const GEMMINI_COMGR_ACTION_OPTIMIZE_BC_TO_BC: gemmini_comgr_action_kind_s =
        gemmini_comgr_action_kind_s(5);
    pub const GEMMINI_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE: gemmini_comgr_action_kind_s =
        gemmini_comgr_action_kind_s(6);
    pub const GEMMINI_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY: gemmini_comgr_action_kind_s =
        gemmini_comgr_action_kind_s(7);
    pub const GEMMINI_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN: gemmini_comgr_action_kind_s =
        gemmini_comgr_action_kind_s(8);
    pub const GEMMINI_COMGR_ACTION_LAST: gemmini_comgr_action_kind_s =
        gemmini_comgr_action_kind_s(8);
}

// Symbol type and info constants
pub struct gemmini_comgr_symbol_type_s(pub c_int);
pub type gemmini_comgr_symbol_type_t = gemmini_comgr_symbol_type_s;

impl gemmini_comgr_symbol_type_s {
    pub const GEMMINI_COMGR_SYMBOL_TYPE_UNKNOWN: gemmini_comgr_symbol_type_s =
        gemmini_comgr_symbol_type_s(-1);
    pub const GEMMINI_COMGR_SYMBOL_TYPE_NOTYPE: gemmini_comgr_symbol_type_s =
        gemmini_comgr_symbol_type_s(0);
    pub const GEMMINI_COMGR_SYMBOL_TYPE_OBJECT: gemmini_comgr_symbol_type_s =
        gemmini_comgr_symbol_type_s(1);
    pub const GEMMINI_COMGR_SYMBOL_TYPE_FUNC: gemmini_comgr_symbol_type_s =
        gemmini_comgr_symbol_type_s(2);
    pub const GEMMINI_COMGR_SYMBOL_TYPE_SECTION: gemmini_comgr_symbol_type_s =
        gemmini_comgr_symbol_type_s(3);
    pub const GEMMINI_COMGR_SYMBOL_TYPE_FILE: gemmini_comgr_symbol_type_s =
        gemmini_comgr_symbol_type_s(4);
    pub const GEMMINI_COMGR_SYMBOL_TYPE_COMMON: gemmini_comgr_symbol_type_s =
        gemmini_comgr_symbol_type_s(5);
    pub const GEMMINI_COMGR_SYMBOL_TYPE_GEMMINI_SYCL_KERNEL: gemmini_comgr_symbol_type_s =
        gemmini_comgr_symbol_type_s(6);
}

pub struct gemmini_comgr_symbol_info_s(pub c_uint);
pub type gemmini_comgr_symbol_info_t = gemmini_comgr_symbol_info_s;

impl gemmini_comgr_symbol_info_s {
    pub const GEMMINI_COMGR_SYMBOL_INFO_NAME_LENGTH: gemmini_comgr_symbol_info_s =
        gemmini_comgr_symbol_info_s(0);
    pub const GEMMINI_COMGR_SYMBOL_INFO_NAME: gemmini_comgr_symbol_info_s =
        gemmini_comgr_symbol_info_s(1);
    pub const GEMMINI_COMGR_SYMBOL_INFO_TYPE: gemmini_comgr_symbol_info_s =
        gemmini_comgr_symbol_info_s(2);
    pub const GEMMINI_COMGR_SYMBOL_INFO_SIZE: gemmini_comgr_symbol_info_s =
        gemmini_comgr_symbol_info_s(3);
    pub const GEMMINI_COMGR_SYMBOL_INFO_IS_UNDEFINED: gemmini_comgr_symbol_info_s =
        gemmini_comgr_symbol_info_s(4);
    pub const GEMMINI_COMGR_SYMBOL_INFO_VALUE: gemmini_comgr_symbol_info_s =
        gemmini_comgr_symbol_info_s(5);
    pub const GEMMINI_COMGR_SYMBOL_INFO_LAST: gemmini_comgr_symbol_info_s =
        gemmini_comgr_symbol_info_s(5);
}

// Code object info structure
pub struct gemmini_comgr_code_object_info_s {
    pub isa: *const c_char,
    pub size: usize,
    pub offset: u64,
}
pub type gemmini_comgr_code_object_info_t = gemmini_comgr_code_object_info_s;

// Key functions that wrap icpx command

pub fn gemmini_comgr_create_data(
    kind: gemmini_comgr_data_kind_t,
    data: *mut gemmini_comgr_data_t,
) -> gemmini_comgr_status_t {
    // Create a new data object with a next handle
    let mut store = command_wrapper::DATA_STORE.lock().unwrap();
    let handle = command_wrapper::get_next_handle();
    let data_obj = gemmini_comgr_data_t { handle };
    store.insert(
        handle,
        command_wrapper::DataContent {
            kind,
            content: Vec::new(),
            name: None,
        },
    );
    // Set the output
    unsafe {
        *data = data_obj;
    }
    Ok(())
}

pub fn gemmini_comgr_release_data(data: gemmini_comgr_data_t) -> gemmini_comgr_status_t {
    // Remove the data from the store
    let mut data_store = command_wrapper::DATA_STORE.lock().unwrap();
    data_store.remove(&data.handle);

    Ok(())
}

pub fn gemmini_comgr_data_set_bytes(
    data: gemmini_comgr_data_t,
    bytes: *const raw::c_void,
    size: usize,
) -> gemmini_comgr_status_t {
    // Validate params
    if bytes.is_null() && size > 0 {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Copy bytes to data
    let content = if size > 0 {
        let slice = unsafe { std::slice::from_raw_parts(bytes as *const u8, size) };
        slice.to_vec()
    } else {
        Vec::new()
    };

    // Update data in store
    let mut data_store = command_wrapper::DATA_STORE.lock().unwrap();
    if let Some(data_content) = data_store.get_mut(&data.handle) {
        data_content.content = content;
        Ok(())
    } else {
        Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn gemmini_comgr_data_set_name(
    data: gemmini_comgr_data_t,
    name: *const c_char,
) -> gemmini_comgr_status_t {
    // Validate params
    if name.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Convert C string to Rust string
    let name_str = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .to_string();

    // Update data in store
    let mut data_store = command_wrapper::DATA_STORE.lock().unwrap();
    if let Some(data_content) = data_store.get_mut(&data.handle) {
        data_content.name = Some(name_str);
        Ok(())
    } else {
        Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn gemmini_comgr_create_data_set(
    data_set: *mut gemmini_comgr_data_set_t,
) -> gemmini_comgr_status_t {
    // Validate params
    if data_set.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Create a new handle
    let handle = command_wrapper::get_next_handle();

    // Store empty data set
    {
        let mut data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
        data_set_store.insert(handle, Vec::new());
    }

    // Return the handle
    unsafe {
        *data_set = gemmini_comgr_data_set_t { handle };
    }

    Ok(())
}

pub fn gemmini_comgr_release_data_set(
    data_set: gemmini_comgr_data_set_t,
) -> gemmini_comgr_status_t {
    // Remove the data set from the store
    let mut data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    data_set_store.remove(&data_set.handle);

    Ok(())
}

pub fn gemmini_comgr_data_set_add(
    data_set: gemmini_comgr_data_set_t,
    data: gemmini_comgr_data_t,
) -> gemmini_comgr_status_t {
    // Add data to data set
    let mut data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    if let Some(set_handles) = data_set_store.get_mut(&data_set.handle) {
        set_handles.push(data.handle);
        Ok(())
    } else {
        Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn gemmini_comgr_data_set_remove(
    data_set: gemmini_comgr_data_set_t,
    data: gemmini_comgr_data_t,
) -> gemmini_comgr_status_t {
    // Remove data from data set
    let mut data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    if let Some(set_handles) = data_set_store.get_mut(&data_set.handle) {
        set_handles.retain(|handle| *handle != data.handle);
        Ok(())
    } else {
        Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn gemmini_comgr_create_action_info(
    action_info: *mut gemmini_comgr_action_info_t,
) -> gemmini_comgr_status_t {
    // Validate params
    if action_info.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Create a new handle
    let handle = command_wrapper::get_next_handle();

    // Store empty action info
    {
        let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
        action_info_store.insert(handle, command_wrapper::ActionInfo::default());
    }

    // Return the handle
    unsafe {
        *action_info = gemmini_comgr_action_info_t { handle };
    }

    Ok(())
}

pub fn gemmini_comgr_release_action_info(
    action_info: gemmini_comgr_action_info_t,
) -> gemmini_comgr_status_t {
    // Remove the action info from the store
    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    action_info_store.remove(&action_info.handle);

    Ok(())
}

pub fn gemmini_comgr_action_info_set_language(
    action_info: gemmini_comgr_action_info_t,
    language: gemmini_comgr_language_t,
) -> gemmini_comgr_status_t {
    // Update action info in store
    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    if let Some(info) = action_info_store.get_mut(&action_info.handle) {
        info.language = Some(language);
        Ok(())
    } else {
        Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn gemmini_comgr_action_info_set_option_list(
    action_info: gemmini_comgr_action_info_t,
    options: *const *const c_char,
    count: usize,
) -> gemmini_comgr_status_t {
    // Validate params
    if options.is_null() && count > 0 {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Convert C strings to Rust strings
    let mut option_strings = Vec::new();
    for i in 0..count {
        let opt_ptr = unsafe { *options.add(i) };
        if opt_ptr.is_null() {
            continue;
        }

        let opt_str = unsafe { CStr::from_ptr(opt_ptr) }
            .to_string_lossy()
            .to_string();
        option_strings.push(opt_str);
    }

    // Update action info in store
    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    if let Some(info) = action_info_store.get_mut(&action_info.handle) {
        info.options = option_strings;
        Ok(())
    } else {
        Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn gemmini_comgr_action_info_set_working_directory(
    action_info: gemmini_comgr_action_info_t,
    directory: *const c_char,
) -> gemmini_comgr_status_t {
    // Validate params
    if directory.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Convert C string to Rust string
    let dir_str = unsafe { CStr::from_ptr(directory) }
        .to_string_lossy()
        .to_string();

    // Update action info in store
    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    if let Some(info) = action_info_store.get_mut(&action_info.handle) {
        info.working_directory = Some(dir_str);
        Ok(())
    } else {
        Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn gemmini_comgr_action_info_set_target(
    action_info: gemmini_comgr_action_info_t,
    target: *const c_char,
) -> gemmini_comgr_status_t {
    // Validate params
    if target.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Convert C string to Rust string
    let target_str = unsafe { CStr::from_ptr(target) }
        .to_string_lossy()
        .to_string();

    // Update action info in store
    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    if let Some(info) = action_info_store.get_mut(&action_info.handle) {
        info.target = Some(target_str);
        Ok(())
    } else {
        Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn gemmini_comgr_do_action(
    action_kind: gemmini_comgr_action_kind_t,
    action_info: gemmini_comgr_action_info_t,
    input_set: gemmini_comgr_data_set_t,
    output_set: gemmini_comgr_data_set_t,
) -> gemmini_comgr_status_t {
    // Use the command_wrapper module to perform the action
    command_wrapper::perform_action(action_kind, action_info, input_set, output_set)
}

pub fn gemmini_comgr_get_data_kind(
    data: gemmini_comgr_data_t,
    kind: *mut gemmini_comgr_data_kind_t,
) -> gemmini_comgr_status_t {
    let store = command_wrapper::DATA_STORE.lock().unwrap();
    match store.get(&data.handle) {
        Some(data_content) => {
            unsafe {
                *kind = data_content.kind;
            }
            Ok(())
        }
        None => Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    }
}

pub fn gemmini_comgr_get_data(
    data_set: gemmini_comgr_data_set_t,
    index: usize,
    data: *mut gemmini_comgr_data_t,
) -> gemmini_comgr_status_t {
    // Validate params
    if data.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Get data from data set
    let data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    if let Some(set_handles) = data_set_store.get(&data_set.handle) {
        if index >= set_handles.len() {
            return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
        }

        unsafe {
            *data = gemmini_comgr_data_t {
                handle: set_handles[index],
            };
        }
        Ok(())
    } else {
        Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn gemmini_comgr_get_data_count(
    data_set: gemmini_comgr_data_set_t,
    count: *mut usize,
) -> gemmini_comgr_status_t {
    // Validate params
    if count.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Get data count from data set
    let data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    if let Some(set_handles) = data_set_store.get(&data_set.handle) {
        unsafe {
            *count = set_handles.len();
        }
        Ok(())
    } else {
        Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

// Metadata functions
pub fn gemmini_comgr_create_metadata(
    metadata: *mut gemmini_comgr_metadata_node_t,
) -> gemmini_comgr_status_t {
    if metadata.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let handle = command_wrapper::get_next_handle();

    unsafe {
        *metadata = gemmini_comgr_metadata_node_s { handle };
    }

    Ok(())
}

pub fn gemmini_comgr_release_metadata(
    metadata: gemmini_comgr_metadata_node_t,
) -> gemmini_comgr_status_t {
    // In a more complex implementation, we would need to clean up metadata resources
    Ok(())
}

pub fn gemmini_comgr_get_data_metadata(
    data: gemmini_comgr_data_t,
    metadata: *mut gemmini_comgr_metadata_node_t,
) -> gemmini_comgr_status_t {
    if metadata.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Verify data exists
    let data_store = command_wrapper::DATA_STORE.lock().unwrap();
    if !data_store.contains_key(&data.handle) {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Create a new metadata node
    let handle = command_wrapper::get_next_handle();

    unsafe {
        *metadata = gemmini_comgr_metadata_node_s { handle };
    }

    Ok(())
}

// Version information API
pub fn gemmini_comgr_get_version_string(version: *mut *const c_char) -> gemmini_comgr_status_t {
    if version.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    static VERSION_STRING: &str = concat!(
        "Intel COMGR API Version ",
        env!("CARGO_PKG_VERSION"),
        " (ZLUDA Wrapper)"
    );

    // Leak this string - it's okay because it's a static string used for the lifetime of the program
    let c_str = CString::new(VERSION_STRING).unwrap();
    let ptr = c_str.into_raw();

    unsafe {
        *version = ptr;
    }

    Ok(())
}

pub fn gemmini_comgr_get_version(major: *mut u32, minor: *mut u32) -> gemmini_comgr_status_t {
    if major.is_null() || minor.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    unsafe {
        *major = GEMMINI_COMGR_INTERFACE_VERSION_MAJOR;
        *minor = GEMMINI_COMGR_INTERFACE_VERSION_MINOR;
    }

    Ok(())
}

// Add data_get_bytes function to match what's used in comgr implementation
pub fn gemmini_comgr_data_get_bytes(
    data: gemmini_comgr_data_t,
    bytes: *mut raw::c_void,
    size: *mut usize,
) -> gemmini_comgr_status_t {
    // Validate params
    if size.is_null() {
        return Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Get data from store
    let data_store = command_wrapper::DATA_STORE.lock().unwrap();
    if let Some(data_content) = data_store.get(&data.handle) {
        // Set size
        unsafe {
            *size = data_content.content.len();
        }

        // Copy bytes if destination buffer is not null
        if !bytes.is_null() && data_content.content.len() > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data_content.content.as_ptr(),
                    bytes as *mut u8,
                    data_content.content.len(),
                );
            }
        }

        Ok(())
    } else {
        Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn gemmini_comgr_get_data_name(
    data: gemmini_comgr_data_t,
    size: *mut usize,
    name: *mut i8,
) -> gemmini_comgr_status_t {
    let store = command_wrapper::DATA_STORE.lock().unwrap();
    match store.get(&data.handle) {
        Some(data_content) => {
            match &data_content.name {
                Some(name_str) => {
                    let name_bytes = name_str.as_bytes();
                    let name_len = name_bytes.len() + 1; // +1 for null terminator

                    unsafe {
                        *size = name_len;
                    }

                    // If name buffer is provided, copy the name
                    if !name.is_null() {
                        if name_len > 1 {
                            // Copy string contents if we have a name
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    name_bytes.as_ptr() as *const i8,
                                    name,
                                    name_bytes.len(),
                                );
                                // Add null terminator
                                *name.add(name_bytes.len()) = 0;
                            }
                        } else {
                            // Just add null terminator if empty name
                            unsafe {
                                *name = 0;
                            }
                        }
                    }

                    Ok(())
                }
                None => {
                    // No name, return size 1 (just the null terminator)
                    unsafe {
                        *size = 1;
                        if !name.is_null() {
                            *name = 0;
                        }
                    }
                    Ok(())
                }
            }
        }
        None => Err(gemmini_comgr_status_s::GEMMINI_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_release_data() {
        let mut data = gemmini_comgr_data_t { handle: 0 };
        let result = gemmini_comgr_create_data(
            gemmini_comgr_data_kind_s::GEMMINI_COMGR_DATA_KIND_SOURCE,
            &mut data,
        );
        assert!(result.is_ok());

        let result = gemmini_comgr_release_data(data);
        assert!(result.is_ok());
    }

    #[test]
    fn data_set_operations() {
        // Create data set
        let mut data_set = gemmini_comgr_data_set_t { handle: 0 };
        let result = gemmini_comgr_create_data_set(&mut data_set);
        assert!(result.is_ok());

        // Create some data
        let mut data1 = gemmini_comgr_data_t { handle: 0 };
        let result = gemmini_comgr_create_data(
            gemmini_comgr_data_kind_s::GEMMINI_COMGR_DATA_KIND_SOURCE,
            &mut data1,
        );
        assert!(result.is_ok());

        // Set content
        let content = "void main() {}";
        let result =
            gemmini_comgr_data_set_bytes(data1, content.as_ptr() as *const _, content.len());
        assert!(result.is_ok());

        // Add to data set
        let result = gemmini_comgr_data_set_add(data_set, data1);
        assert!(result.is_ok());

        // Check count
        let mut count = 0;
        let result = gemmini_comgr_get_data_count(data_set, &mut count);
        assert!(result.is_ok());
        assert_eq!(count, 1);

        // Clean up
        let result = gemmini_comgr_release_data_set(data_set);
        assert!(result.is_ok());

        let result = gemmini_comgr_release_data(data1);
        assert!(result.is_ok());
    }
}
