mod command_wrapper;

use std::ffi::{CStr, CString, c_char, c_int, c_uint};
use std::num::NonZeroU32;
use std::os::raw;
use std::process::Command;

// Version constants similar to AMD's API
pub const TT_COMGR_INTERFACE_VERSION_MAJOR: u32 = 1;
pub const TT_COMGR_INTERFACE_VERSION_MINOR: u32 = 0;

// Status types
#[derive(Debug, Clone, Copy)]
pub struct tt_comgr_status_s(pub NonZeroU32);
pub type tt_comgr_status_t = Result<(), self::tt_comgr_status_s>;

// Status constants
impl tt_comgr_status_s {
    pub const TT_COMGR_STATUS_SUCCESS: Result<(), tt_comgr_status_s> = Ok(());

    pub const TT_COMGR_STATUS_ERROR: tt_comgr_status_s =
        tt_comgr_status_s(unsafe { NonZeroU32::new_unchecked(1) });

    pub const TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT: tt_comgr_status_s =
        tt_comgr_status_s(unsafe { NonZeroU32::new_unchecked(2) });

    pub const TT_COMGR_STATUS_ERROR_OUT_OF_RESOURCES: tt_comgr_status_s =
        tt_comgr_status_s(unsafe { NonZeroU32::new_unchecked(3) });
}

// Language types similar to AMD
#[derive(Default, Clone, Copy)]
pub struct tt_comgr_language_s(pub c_uint);
pub type tt_comgr_language_t = tt_comgr_language_s;

impl tt_comgr_language_s {
    pub const TT_COMGR_LANGUAGE_NONE: tt_comgr_language_s = tt_comgr_language_s(0);
    pub const TT_COMGR_LANGUAGE_OPENCL_1_2: tt_comgr_language_s = tt_comgr_language_s(1);
    pub const TT_COMGR_LANGUAGE_OPENCL_2_0: tt_comgr_language_s = tt_comgr_language_s(2);
    pub const TT_COMGR_LANGUAGE_SYCL: tt_comgr_language_s = tt_comgr_language_s(3);
    pub const TT_COMGR_LANGUAGE_LLVM_IR: tt_comgr_language_s = tt_comgr_language_s(4);
    pub const TT_COMGR_LANGUAGE_LAST: tt_comgr_language_s = tt_comgr_language_s(4);
}

// Data kinds similar to AMD
#[derive(Default, Clone, Copy)]
pub struct tt_comgr_data_kind_s(pub c_uint);
pub type tt_comgr_data_kind_t = tt_comgr_data_kind_s;

impl tt_comgr_data_kind_s {
    pub const TT_COMGR_DATA_KIND_UNDEF: tt_comgr_data_kind_s = tt_comgr_data_kind_s(0);
    pub const TT_COMGR_DATA_KIND_SOURCE: tt_comgr_data_kind_s = tt_comgr_data_kind_s(1);
    pub const TT_COMGR_DATA_KIND_INCLUDE: tt_comgr_data_kind_s = tt_comgr_data_kind_s(2);
    pub const TT_COMGR_DATA_KIND_PRECOMPILED_HEADER: tt_comgr_data_kind_s = tt_comgr_data_kind_s(3);
    pub const TT_COMGR_DATA_KIND_DIAGNOSTIC: tt_comgr_data_kind_s = tt_comgr_data_kind_s(4);
    pub const TT_COMGR_DATA_KIND_LOG: tt_comgr_data_kind_s = tt_comgr_data_kind_s(5);
    pub const TT_COMGR_DATA_KIND_BC: tt_comgr_data_kind_s = tt_comgr_data_kind_s(6);
    pub const TT_COMGR_DATA_KIND_RELOCATABLE: tt_comgr_data_kind_s = tt_comgr_data_kind_s(7);
    pub const TT_COMGR_DATA_KIND_EXECUTABLE: tt_comgr_data_kind_s = tt_comgr_data_kind_s(8);
    pub const TT_COMGR_DATA_KIND_BYTES: tt_comgr_data_kind_s = tt_comgr_data_kind_s(9);
    pub const TT_COMGR_DATA_KIND_FATBIN: tt_comgr_data_kind_s = tt_comgr_data_kind_s(16);
    pub const TT_COMGR_DATA_KIND_LAST: tt_comgr_data_kind_s = tt_comgr_data_kind_s(16);
}

// Data structures similar to AMD
#[derive(Default, Clone, Copy)]
pub struct tt_comgr_data_s {
    pub handle: u64,
}
pub type tt_comgr_data_t = tt_comgr_data_s;

#[derive(Default, Clone, Copy)]
pub struct tt_comgr_data_set_s {
    pub handle: u64,
}
pub type tt_comgr_data_set_t = tt_comgr_data_set_s;
#[derive(Default, Clone, Copy)]
pub struct tt_comgr_action_info_s {
    pub handle: u64,
}
pub type tt_comgr_action_info_t = tt_comgr_action_info_s;

pub struct tt_comgr_metadata_node_s {
    pub handle: u64,
}
pub type tt_comgr_metadata_node_t = tt_comgr_metadata_node_s;

pub struct tt_comgr_symbol_s {
    pub handle: u64,
}
pub type tt_comgr_symbol_t = tt_comgr_symbol_s;

// Define metadata kind constants
pub struct tt_comgr_metadata_kind_s(pub c_uint);
pub type tt_comgr_metadata_kind_t = tt_comgr_metadata_kind_s;

impl tt_comgr_metadata_kind_s {
    pub const TT_COMGR_METADATA_KIND_NULL: tt_comgr_metadata_kind_s = tt_comgr_metadata_kind_s(0);
    pub const TT_COMGR_METADATA_KIND_STRING: tt_comgr_metadata_kind_s = tt_comgr_metadata_kind_s(1);
    pub const TT_COMGR_METADATA_KIND_MAP: tt_comgr_metadata_kind_s = tt_comgr_metadata_kind_s(2);
    pub const TT_COMGR_METADATA_KIND_LIST: tt_comgr_metadata_kind_s = tt_comgr_metadata_kind_s(3);
    pub const TT_COMGR_METADATA_KIND_LAST: tt_comgr_metadata_kind_s = tt_comgr_metadata_kind_s(3);
}

// Action kinds similar to AMD
#[derive(Default, Clone, Copy)]
pub struct tt_comgr_action_kind_s(pub c_uint);
pub type tt_comgr_action_kind_t = tt_comgr_action_kind_s;

impl tt_comgr_action_kind_s {
    pub const TT_COMGR_ACTION_SOURCE_TO_PREPROCESSOR: tt_comgr_action_kind_s =
        tt_comgr_action_kind_s(0);
    pub const TT_COMGR_ACTION_ADD_PRECOMPILED_HEADERS: tt_comgr_action_kind_s =
        tt_comgr_action_kind_s(1);
    pub const TT_COMGR_ACTION_COMPILE_SOURCE_TO_BC: tt_comgr_action_kind_s =
        tt_comgr_action_kind_s(2);
    pub const TT_COMGR_ACTION_ADD_DEVICE_LIBRARIES: tt_comgr_action_kind_s =
        tt_comgr_action_kind_s(3);
    pub const TT_COMGR_ACTION_LINK_BC_TO_BC: tt_comgr_action_kind_s = tt_comgr_action_kind_s(4);
    pub const TT_COMGR_ACTION_OPTIMIZE_BC_TO_BC: tt_comgr_action_kind_s = tt_comgr_action_kind_s(5);
    pub const TT_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE: tt_comgr_action_kind_s =
        tt_comgr_action_kind_s(6);
    pub const TT_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY: tt_comgr_action_kind_s =
        tt_comgr_action_kind_s(7);
    pub const TT_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN: tt_comgr_action_kind_s =
        tt_comgr_action_kind_s(8);
    pub const TT_COMGR_ACTION_LAST: tt_comgr_action_kind_s = tt_comgr_action_kind_s(8);
}

// Symbol type and info constants
pub struct tt_comgr_symbol_type_s(pub c_int);
pub type tt_comgr_symbol_type_t = tt_comgr_symbol_type_s;

impl tt_comgr_symbol_type_s {
    pub const TT_COMGR_SYMBOL_TYPE_UNKNOWN: tt_comgr_symbol_type_s = tt_comgr_symbol_type_s(-1);
    pub const TT_COMGR_SYMBOL_TYPE_NOTYPE: tt_comgr_symbol_type_s = tt_comgr_symbol_type_s(0);
    pub const TT_COMGR_SYMBOL_TYPE_OBJECT: tt_comgr_symbol_type_s = tt_comgr_symbol_type_s(1);
    pub const TT_COMGR_SYMBOL_TYPE_FUNC: tt_comgr_symbol_type_s = tt_comgr_symbol_type_s(2);
    pub const TT_COMGR_SYMBOL_TYPE_SECTION: tt_comgr_symbol_type_s = tt_comgr_symbol_type_s(3);
    pub const TT_COMGR_SYMBOL_TYPE_FILE: tt_comgr_symbol_type_s = tt_comgr_symbol_type_s(4);
    pub const TT_COMGR_SYMBOL_TYPE_COMMON: tt_comgr_symbol_type_s = tt_comgr_symbol_type_s(5);
    pub const TT_COMGR_SYMBOL_TYPE_TT_SYCL_KERNEL: tt_comgr_symbol_type_s =
        tt_comgr_symbol_type_s(6);
}

pub struct tt_comgr_symbol_info_s(pub c_uint);
pub type tt_comgr_symbol_info_t = tt_comgr_symbol_info_s;

impl tt_comgr_symbol_info_s {
    pub const TT_COMGR_SYMBOL_INFO_NAME_LENGTH: tt_comgr_symbol_info_s = tt_comgr_symbol_info_s(0);
    pub const TT_COMGR_SYMBOL_INFO_NAME: tt_comgr_symbol_info_s = tt_comgr_symbol_info_s(1);
    pub const TT_COMGR_SYMBOL_INFO_TYPE: tt_comgr_symbol_info_s = tt_comgr_symbol_info_s(2);
    pub const TT_COMGR_SYMBOL_INFO_SIZE: tt_comgr_symbol_info_s = tt_comgr_symbol_info_s(3);
    pub const TT_COMGR_SYMBOL_INFO_IS_UNDEFINED: tt_comgr_symbol_info_s = tt_comgr_symbol_info_s(4);
    pub const TT_COMGR_SYMBOL_INFO_VALUE: tt_comgr_symbol_info_s = tt_comgr_symbol_info_s(5);
    pub const TT_COMGR_SYMBOL_INFO_LAST: tt_comgr_symbol_info_s = tt_comgr_symbol_info_s(5);
}

// Code object info structure
pub struct tt_comgr_code_object_info_s {
    pub isa: *const c_char,
    pub size: usize,
    pub offset: u64,
}
pub type tt_comgr_code_object_info_t = tt_comgr_code_object_info_s;

// Key functions that wrap icpx command

pub fn tt_comgr_create_data(
    kind: tt_comgr_data_kind_t,
    data: *mut tt_comgr_data_t,
) -> tt_comgr_status_t {
    // Create a new data object with a next handle
    let mut store = command_wrapper::DATA_STORE.lock().unwrap();
    let handle = command_wrapper::get_next_handle();
    let data_obj = tt_comgr_data_t { handle };
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

pub fn tt_comgr_release_data(data: tt_comgr_data_t) -> tt_comgr_status_t {
    // Remove the data from the store
    let mut data_store = command_wrapper::DATA_STORE.lock().unwrap();
    data_store.remove(&data.handle);

    Ok(())
}

pub fn tt_comgr_data_set_bytes(
    data: tt_comgr_data_t,
    bytes: *const raw::c_void,
    size: usize,
) -> tt_comgr_status_t {
    // Validate params
    if bytes.is_null() && size > 0 {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
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
        Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn tt_comgr_data_set_name(data: tt_comgr_data_t, name: *const c_char) -> tt_comgr_status_t {
    // Validate params
    if name.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
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
        Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn tt_comgr_create_data_set(data_set: *mut tt_comgr_data_set_t) -> tt_comgr_status_t {
    // Validate params
    if data_set.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
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
        *data_set = tt_comgr_data_set_t { handle };
    }

    Ok(())
}

pub fn tt_comgr_release_data_set(data_set: tt_comgr_data_set_t) -> tt_comgr_status_t {
    // Remove the data set from the store
    let mut data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    data_set_store.remove(&data_set.handle);

    Ok(())
}

pub fn tt_comgr_data_set_add(
    data_set: tt_comgr_data_set_t,
    data: tt_comgr_data_t,
) -> tt_comgr_status_t {
    // Add data to data set
    let mut data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    if let Some(set_handles) = data_set_store.get_mut(&data_set.handle) {
        set_handles.push(data.handle);
        Ok(())
    } else {
        Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn tt_comgr_data_set_remove(
    data_set: tt_comgr_data_set_t,
    data: tt_comgr_data_t,
) -> tt_comgr_status_t {
    // Remove data from data set
    let mut data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    if let Some(set_handles) = data_set_store.get_mut(&data_set.handle) {
        set_handles.retain(|handle| *handle != data.handle);
        Ok(())
    } else {
        Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn tt_comgr_create_action_info(action_info: *mut tt_comgr_action_info_t) -> tt_comgr_status_t {
    // Validate params
    if action_info.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
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
        *action_info = tt_comgr_action_info_t { handle };
    }

    Ok(())
}

pub fn tt_comgr_release_action_info(action_info: tt_comgr_action_info_t) -> tt_comgr_status_t {
    // Remove the action info from the store
    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    action_info_store.remove(&action_info.handle);

    Ok(())
}

pub fn tt_comgr_action_info_set_language(
    action_info: tt_comgr_action_info_t,
    language: tt_comgr_language_t,
) -> tt_comgr_status_t {
    // Update action info in store
    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    if let Some(info) = action_info_store.get_mut(&action_info.handle) {
        info.language = Some(language);
        Ok(())
    } else {
        Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn tt_comgr_action_info_set_option_list(
    action_info: tt_comgr_action_info_t,
    options: *const *const c_char,
    count: usize,
) -> tt_comgr_status_t {
    // Validate params
    if options.is_null() && count > 0 {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
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
        Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn tt_comgr_action_info_set_working_directory(
    action_info: tt_comgr_action_info_t,
    directory: *const c_char,
) -> tt_comgr_status_t {
    // Validate params
    if directory.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
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
        Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn tt_comgr_action_info_set_target(
    action_info: tt_comgr_action_info_t,
    target: *const c_char,
) -> tt_comgr_status_t {
    // Validate params
    if target.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
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
        Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn tt_comgr_do_action(
    action_kind: tt_comgr_action_kind_t,
    action_info: tt_comgr_action_info_t,
    input_set: tt_comgr_data_set_t,
    output_set: tt_comgr_data_set_t,
) -> tt_comgr_status_t {
    // Use the command_wrapper module to perform the action
    command_wrapper::perform_action(action_kind, action_info, input_set, output_set)
}

pub fn tt_comgr_get_data_kind(
    data: tt_comgr_data_t,
    kind: *mut tt_comgr_data_kind_t,
) -> tt_comgr_status_t {
    let store = command_wrapper::DATA_STORE.lock().unwrap();
    match store.get(&data.handle) {
        Some(data_content) => {
            unsafe {
                *kind = data_content.kind;
            }
            Ok(())
        }
        None => Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    }
}

pub fn tt_comgr_get_data(
    data_set: tt_comgr_data_set_t,
    index: usize,
    data: *mut tt_comgr_data_t,
) -> tt_comgr_status_t {
    // Validate params
    if data.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Get data from data set
    let data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    if let Some(set_handles) = data_set_store.get(&data_set.handle) {
        if index >= set_handles.len() {
            return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
        }

        unsafe {
            *data = tt_comgr_data_t {
                handle: set_handles[index],
            };
        }
        Ok(())
    } else {
        Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn tt_comgr_get_data_count(
    data_set: tt_comgr_data_set_t,
    count: *mut usize,
) -> tt_comgr_status_t {
    // Validate params
    if count.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Get data count from data set
    let data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    if let Some(set_handles) = data_set_store.get(&data_set.handle) {
        unsafe {
            *count = set_handles.len();
        }
        Ok(())
    } else {
        Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

// Metadata functions
pub fn tt_comgr_create_metadata(metadata: *mut tt_comgr_metadata_node_t) -> tt_comgr_status_t {
    if metadata.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let handle = command_wrapper::get_next_handle();

    unsafe {
        *metadata = tt_comgr_metadata_node_s { handle };
    }

    Ok(())
}

pub fn tt_comgr_release_metadata(metadata: tt_comgr_metadata_node_t) -> tt_comgr_status_t {
    // In a more complex implementation, we would need to clean up metadata resources
    Ok(())
}

pub fn tt_comgr_get_data_metadata(
    data: tt_comgr_data_t,
    metadata: *mut tt_comgr_metadata_node_t,
) -> tt_comgr_status_t {
    if metadata.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Verify data exists
    let data_store = command_wrapper::DATA_STORE.lock().unwrap();
    if !data_store.contains_key(&data.handle) {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // Create a new metadata node
    let handle = command_wrapper::get_next_handle();

    unsafe {
        *metadata = tt_comgr_metadata_node_s { handle };
    }

    Ok(())
}

// Version information API
pub fn tt_comgr_get_version_string(version: *mut *const c_char) -> tt_comgr_status_t {
    if version.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
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

pub fn tt_comgr_get_version(major: *mut u32, minor: *mut u32) -> tt_comgr_status_t {
    if major.is_null() || minor.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    unsafe {
        *major = TT_COMGR_INTERFACE_VERSION_MAJOR;
        *minor = TT_COMGR_INTERFACE_VERSION_MINOR;
    }

    Ok(())
}

// Add data_get_bytes function to match what's used in comgr implementation
pub fn tt_comgr_data_get_bytes(
    data: tt_comgr_data_t,
    bytes: *mut raw::c_void,
    size: *mut usize,
) -> tt_comgr_status_t {
    // Validate params
    if size.is_null() {
        return Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
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
        Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn tt_comgr_get_data_name(
    data: tt_comgr_data_t,
    size: *mut usize,
    name: *mut i8,
) -> tt_comgr_status_t {
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
        None => Err(tt_comgr_status_s::TT_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_release_data() {
        let mut data = tt_comgr_data_t { handle: 0 };
        let result =
            tt_comgr_create_data(tt_comgr_data_kind_s::TT_COMGR_DATA_KIND_SOURCE, &mut data);
        assert!(result.is_ok());

        let result = tt_comgr_release_data(data);
        assert!(result.is_ok());
    }

    #[test]
    fn data_set_operations() {
        // Create data set
        let mut data_set = tt_comgr_data_set_t { handle: 0 };
        let result = tt_comgr_create_data_set(&mut data_set);
        assert!(result.is_ok());

        // Create some data
        let mut data1 = tt_comgr_data_t { handle: 0 };
        let result =
            tt_comgr_create_data(tt_comgr_data_kind_s::TT_COMGR_DATA_KIND_SOURCE, &mut data1);
        assert!(result.is_ok());

        // Set content
        let content = "void main() {}";
        let result = tt_comgr_data_set_bytes(data1, content.as_ptr() as *const _, content.len());
        assert!(result.is_ok());

        // Add to data set
        let result = tt_comgr_data_set_add(data_set, data1);
        assert!(result.is_ok());

        // Check count
        let mut count = 0;
        let result = tt_comgr_get_data_count(data_set, &mut count);
        assert!(result.is_ok());
        assert_eq!(count, 1);

        // Clean up
        let result = tt_comgr_release_data_set(data_set);
        assert!(result.is_ok());

        let result = tt_comgr_release_data(data1);
        assert!(result.is_ok());
    }
}
