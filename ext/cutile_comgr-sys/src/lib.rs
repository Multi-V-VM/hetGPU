//! CuTile to TOSA Transformation Pipeline
//!
//! This crate provides Rust bindings for loading CuTile bytecode,
//! transforming it to TOSA dialect, and executing on non-CUDA backends.

mod bytecode;
mod runtime;
mod tosa_transform;

pub use bytecode::*;
pub use runtime::*;
pub use tosa_transform::*;

use lazy_static::lazy_static;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::num::NonZeroU32;
use std::os::raw;
use std::sync::Mutex;

// Version constants
pub const CUTILE_COMGR_INTERFACE_VERSION_MAJOR: u32 = 1;
pub const CUTILE_COMGR_INTERFACE_VERSION_MINOR: u32 = 0;

// Status types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct cutile_comgr_status_s(pub u32);

pub type cutile_comgr_status_t = Result<(), cutile_comgr_status_s>;

impl cutile_comgr_status_s {
    pub const CUTILE_COMGR_STATUS_SUCCESS: u32 = 0;
    pub const CUTILE_COMGR_STATUS_ERROR: cutile_comgr_status_s = cutile_comgr_status_s(1);
    pub const CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT: cutile_comgr_status_s =
        cutile_comgr_status_s(2);
    pub const CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES: cutile_comgr_status_s =
        cutile_comgr_status_s(3);
    pub const CUTILE_COMGR_STATUS_ERROR_INVALID_BYTECODE: cutile_comgr_status_s =
        cutile_comgr_status_s(4);
    pub const CUTILE_COMGR_STATUS_ERROR_TRANSFORMATION_FAILED: cutile_comgr_status_s =
        cutile_comgr_status_s(5);
    pub const CUTILE_COMGR_STATUS_ERROR_EXECUTION_FAILED: cutile_comgr_status_s =
        cutile_comgr_status_s(6);
}

// Data kinds for CuTile pipeline
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct cutile_comgr_data_kind_s(pub c_uint);

impl cutile_comgr_data_kind_s {
    pub const CUTILE_COMGR_DATA_KIND_UNDEF: cutile_comgr_data_kind_s = cutile_comgr_data_kind_s(0);
    pub const CUTILE_COMGR_DATA_KIND_BYTECODE: cutile_comgr_data_kind_s =
        cutile_comgr_data_kind_s(1);
    pub const CUTILE_COMGR_DATA_KIND_MLIR_CUTILE: cutile_comgr_data_kind_s =
        cutile_comgr_data_kind_s(2);
    pub const CUTILE_COMGR_DATA_KIND_MLIR_TOSA: cutile_comgr_data_kind_s =
        cutile_comgr_data_kind_s(3);
    pub const CUTILE_COMGR_DATA_KIND_EXECUTABLE: cutile_comgr_data_kind_s =
        cutile_comgr_data_kind_s(4);
    pub const CUTILE_COMGR_DATA_KIND_LOG: cutile_comgr_data_kind_s = cutile_comgr_data_kind_s(5);
}

// Action kinds for CuTile pipeline
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct cutile_comgr_action_kind_s(pub c_uint);

impl cutile_comgr_action_kind_s {
    /// Load CuTile bytecode and deserialize to MLIR
    pub const CUTILE_COMGR_ACTION_LOAD_BYTECODE: cutile_comgr_action_kind_s =
        cutile_comgr_action_kind_s(0);
    /// Transform CuTile MLIR to TOSA MLIR
    pub const CUTILE_COMGR_ACTION_CUTILE_TO_TOSA: cutile_comgr_action_kind_s =
        cutile_comgr_action_kind_s(1);
    /// Lower TOSA to target-specific code (e.g., for Tenstorrent, Intel, etc.)
    pub const CUTILE_COMGR_ACTION_TOSA_TO_TARGET: cutile_comgr_action_kind_s =
        cutile_comgr_action_kind_s(2);
    /// Full pipeline: bytecode -> CuTile -> TOSA -> target executable
    pub const CUTILE_COMGR_ACTION_FULL_PIPELINE: cutile_comgr_action_kind_s =
        cutile_comgr_action_kind_s(3);
}

// Target backend types
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct cutile_comgr_target_s(pub c_uint);

impl cutile_comgr_target_s {
    pub const CUTILE_COMGR_TARGET_TENSTORRENT: cutile_comgr_target_s = cutile_comgr_target_s(0);
    pub const CUTILE_COMGR_TARGET_INTEL: cutile_comgr_target_s = cutile_comgr_target_s(1);
    pub const CUTILE_COMGR_TARGET_AMD: cutile_comgr_target_s = cutile_comgr_target_s(2);
    pub const CUTILE_COMGR_TARGET_GEMMINI: cutile_comgr_target_s = cutile_comgr_target_s(3);
    pub const CUTILE_COMGR_TARGET_CPU: cutile_comgr_target_s = cutile_comgr_target_s(4);
}

// Data handle structures
#[derive(Default, Clone, Copy)]
#[repr(C)]
pub struct cutile_comgr_data_s {
    pub handle: u64,
}
pub type cutile_comgr_data_t = cutile_comgr_data_s;

#[derive(Default, Clone, Copy)]
#[repr(C)]
pub struct cutile_comgr_data_set_s {
    pub handle: u64,
}
pub type cutile_comgr_data_set_t = cutile_comgr_data_set_s;

#[derive(Default, Clone, Copy)]
#[repr(C)]
pub struct cutile_comgr_action_info_s {
    pub handle: u64,
}
pub type cutile_comgr_action_info_t = cutile_comgr_action_info_s;

// Internal storage structures
#[derive(Clone)]
pub struct DataContent {
    pub kind: cutile_comgr_data_kind_s,
    pub content: Vec<u8>,
    pub name: Option<String>,
}

#[derive(Clone, Default)]
pub struct ActionInfo {
    pub target: Option<cutile_comgr_target_s>,
    pub options: Vec<String>,
    pub working_directory: Option<String>,
}

lazy_static! {
    static ref HANDLE_COUNTER: Mutex<u64> = Mutex::new(1);
    static ref DATA_STORE: Mutex<HashMap<u64, DataContent>> = Mutex::new(HashMap::new());
    static ref DATA_SET_STORE: Mutex<HashMap<u64, Vec<u64>>> = Mutex::new(HashMap::new());
    static ref ACTION_INFO_STORE: Mutex<HashMap<u64, ActionInfo>> = Mutex::new(HashMap::new());
}

fn get_next_handle() -> u64 {
    let mut counter = HANDLE_COUNTER.lock().unwrap();
    let handle = *counter;
    *counter += 1;
    handle
}

// Core API functions

/// Create a new data object
pub fn cutile_comgr_create_data(
    kind: cutile_comgr_data_kind_s,
    data: *mut cutile_comgr_data_t,
) -> cutile_comgr_status_t {
    if data.is_null() {
        return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let handle = get_next_handle();
    let data_obj = cutile_comgr_data_t { handle };

    let mut store = DATA_STORE.lock().unwrap();
    store.insert(
        handle,
        DataContent {
            kind,
            content: Vec::new(),
            name: None,
        },
    );

    unsafe {
        *data = data_obj;
    }
    Ok(())
}

/// Release a data object
pub fn cutile_comgr_release_data(data: cutile_comgr_data_t) -> cutile_comgr_status_t {
    let mut store = DATA_STORE.lock().unwrap();
    store.remove(&data.handle);
    Ok(())
}

/// Set data content bytes
pub fn cutile_comgr_data_set_bytes(
    data: cutile_comgr_data_t,
    bytes: *const raw::c_void,
    size: usize,
) -> cutile_comgr_status_t {
    if bytes.is_null() && size > 0 {
        return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let content = if size > 0 {
        let slice = unsafe { std::slice::from_raw_parts(bytes as *const u8, size) };
        slice.to_vec()
    } else {
        Vec::new()
    };

    let mut store = DATA_STORE.lock().unwrap();
    if let Some(data_content) = store.get_mut(&data.handle) {
        data_content.content = content;
        Ok(())
    } else {
        Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

/// Set data name
pub fn cutile_comgr_data_set_name(
    data: cutile_comgr_data_t,
    name: *const c_char,
) -> cutile_comgr_status_t {
    if name.is_null() {
        return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let name_str = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .to_string();

    let mut store = DATA_STORE.lock().unwrap();
    if let Some(data_content) = store.get_mut(&data.handle) {
        data_content.name = Some(name_str);
        Ok(())
    } else {
        Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

/// Get data bytes
pub fn cutile_comgr_data_get_bytes(
    data: cutile_comgr_data_t,
    bytes: *mut raw::c_void,
    size: *mut usize,
) -> cutile_comgr_status_t {
    if size.is_null() {
        return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let store = DATA_STORE.lock().unwrap();
    if let Some(data_content) = store.get(&data.handle) {
        unsafe {
            *size = data_content.content.len();
        }

        if !bytes.is_null() && !data_content.content.is_empty() {
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
        Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

/// Create a data set
pub fn cutile_comgr_create_data_set(
    data_set: *mut cutile_comgr_data_set_t,
) -> cutile_comgr_status_t {
    if data_set.is_null() {
        return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let handle = get_next_handle();

    {
        let mut store = DATA_SET_STORE.lock().unwrap();
        store.insert(handle, Vec::new());
    }

    unsafe {
        *data_set = cutile_comgr_data_set_t { handle };
    }
    Ok(())
}

/// Release a data set
pub fn cutile_comgr_release_data_set(data_set: cutile_comgr_data_set_t) -> cutile_comgr_status_t {
    let mut store = DATA_SET_STORE.lock().unwrap();
    store.remove(&data_set.handle);
    Ok(())
}

/// Add data to a data set
pub fn cutile_comgr_data_set_add(
    data_set: cutile_comgr_data_set_t,
    data: cutile_comgr_data_t,
) -> cutile_comgr_status_t {
    let mut store = DATA_SET_STORE.lock().unwrap();
    if let Some(handles) = store.get_mut(&data_set.handle) {
        handles.push(data.handle);
        Ok(())
    } else {
        Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

/// Get data count in a data set
pub fn cutile_comgr_get_data_count(
    data_set: cutile_comgr_data_set_t,
    count: *mut usize,
) -> cutile_comgr_status_t {
    if count.is_null() {
        return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let store = DATA_SET_STORE.lock().unwrap();
    if let Some(handles) = store.get(&data_set.handle) {
        unsafe {
            *count = handles.len();
        }
        Ok(())
    } else {
        Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

/// Get data from a data set by index
pub fn cutile_comgr_get_data(
    data_set: cutile_comgr_data_set_t,
    index: usize,
    data: *mut cutile_comgr_data_t,
) -> cutile_comgr_status_t {
    if data.is_null() {
        return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let store = DATA_SET_STORE.lock().unwrap();
    if let Some(handles) = store.get(&data_set.handle) {
        if index >= handles.len() {
            return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
        }
        unsafe {
            *data = cutile_comgr_data_t {
                handle: handles[index],
            };
        }
        Ok(())
    } else {
        Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

/// Create action info
pub fn cutile_comgr_create_action_info(
    action_info: *mut cutile_comgr_action_info_t,
) -> cutile_comgr_status_t {
    if action_info.is_null() {
        return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let handle = get_next_handle();

    {
        let mut store = ACTION_INFO_STORE.lock().unwrap();
        store.insert(handle, ActionInfo::default());
    }

    unsafe {
        *action_info = cutile_comgr_action_info_t { handle };
    }
    Ok(())
}

/// Release action info
pub fn cutile_comgr_release_action_info(
    action_info: cutile_comgr_action_info_t,
) -> cutile_comgr_status_t {
    let mut store = ACTION_INFO_STORE.lock().unwrap();
    store.remove(&action_info.handle);
    Ok(())
}

/// Set target backend for action
pub fn cutile_comgr_action_info_set_target(
    action_info: cutile_comgr_action_info_t,
    target: cutile_comgr_target_s,
) -> cutile_comgr_status_t {
    let mut store = ACTION_INFO_STORE.lock().unwrap();
    if let Some(info) = store.get_mut(&action_info.handle) {
        info.target = Some(target);
        Ok(())
    } else {
        Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

/// Set options for action
pub fn cutile_comgr_action_info_set_option_list(
    action_info: cutile_comgr_action_info_t,
    options: *const *const c_char,
    count: usize,
) -> cutile_comgr_status_t {
    if options.is_null() && count > 0 {
        return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

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

    let mut store = ACTION_INFO_STORE.lock().unwrap();
    if let Some(info) = store.get_mut(&action_info.handle) {
        info.options = option_strings;
        Ok(())
    } else {
        Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

/// Perform an action on data
pub fn cutile_comgr_do_action(
    action_kind: cutile_comgr_action_kind_s,
    action_info: cutile_comgr_action_info_t,
    input_set: cutile_comgr_data_set_t,
    output_set: cutile_comgr_data_set_t,
) -> cutile_comgr_status_t {
    // Get action info
    let action_info_data = {
        let store = ACTION_INFO_STORE.lock().unwrap();
        store.get(&action_info.handle).cloned()
    };

    let action_info_data = match action_info_data {
        Some(info) => info,
        None => return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    };

    // Get input data
    let input_handles = {
        let store = DATA_SET_STORE.lock().unwrap();
        store.get(&input_set.handle).cloned()
    };

    let input_handles = match input_handles {
        Some(handles) => handles,
        None => return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    };

    match action_kind {
        k if k == cutile_comgr_action_kind_s::CUTILE_COMGR_ACTION_LOAD_BYTECODE => {
            bytecode::load_bytecode(&input_handles, output_set)
        }
        k if k == cutile_comgr_action_kind_s::CUTILE_COMGR_ACTION_CUTILE_TO_TOSA => {
            tosa_transform::transform_to_tosa(&input_handles, output_set)
        }
        k if k == cutile_comgr_action_kind_s::CUTILE_COMGR_ACTION_TOSA_TO_TARGET => {
            let target = action_info_data
                .target
                .unwrap_or(cutile_comgr_target_s::CUTILE_COMGR_TARGET_CPU);
            runtime::lower_tosa_to_target(&input_handles, output_set, target)
        }
        k if k == cutile_comgr_action_kind_s::CUTILE_COMGR_ACTION_FULL_PIPELINE => {
            let target = action_info_data
                .target
                .unwrap_or(cutile_comgr_target_s::CUTILE_COMGR_TARGET_CPU);
            full_pipeline(&input_handles, output_set, target)
        }
        _ => Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    }
}

/// Execute the full pipeline: bytecode -> CuTile -> TOSA -> target
fn full_pipeline(
    input_handles: &[u64],
    output_set: cutile_comgr_data_set_t,
    target: cutile_comgr_target_s,
) -> cutile_comgr_status_t {
    eprintln!("CuTile Pipeline: Starting full transformation pipeline");

    // Step 1: Create intermediate data set for CuTile MLIR
    let mut cutile_mlir_set = cutile_comgr_data_set_t { handle: 0 };
    cutile_comgr_create_data_set(&mut cutile_mlir_set)?;

    // Step 2: Load bytecode to CuTile MLIR
    eprintln!("CuTile Pipeline: Loading bytecode...");
    bytecode::load_bytecode(input_handles, cutile_mlir_set)?;

    // Get handles from intermediate set
    let cutile_handles = {
        let store = DATA_SET_STORE.lock().unwrap();
        store
            .get(&cutile_mlir_set.handle)
            .cloned()
            .unwrap_or_default()
    };

    // Step 3: Create intermediate data set for TOSA MLIR
    let mut tosa_mlir_set = cutile_comgr_data_set_t { handle: 0 };
    cutile_comgr_create_data_set(&mut tosa_mlir_set)?;

    // Step 4: Transform CuTile to TOSA
    eprintln!("CuTile Pipeline: Transforming to TOSA...");
    tosa_transform::transform_to_tosa(&cutile_handles, tosa_mlir_set)?;

    // Get handles from TOSA set
    let tosa_handles = {
        let store = DATA_SET_STORE.lock().unwrap();
        store
            .get(&tosa_mlir_set.handle)
            .cloned()
            .unwrap_or_default()
    };

    // Step 5: Lower TOSA to target
    eprintln!("CuTile Pipeline: Lowering to target: {:?}", target);
    runtime::lower_tosa_to_target(&tosa_handles, output_set, target)?;

    // Clean up intermediate sets
    cutile_comgr_release_data_set(cutile_mlir_set)?;
    cutile_comgr_release_data_set(tosa_mlir_set)?;

    eprintln!("CuTile Pipeline: Transformation complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_release_data() {
        let mut data = cutile_comgr_data_t { handle: 0 };
        let result = cutile_comgr_create_data(
            cutile_comgr_data_kind_s::CUTILE_COMGR_DATA_KIND_BYTECODE,
            &mut data,
        );
        assert!(result.is_ok());
        assert!(data.handle > 0);

        let result = cutile_comgr_release_data(data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_data_set_operations() {
        let mut data_set = cutile_comgr_data_set_t { handle: 0 };
        let result = cutile_comgr_create_data_set(&mut data_set);
        assert!(result.is_ok());

        let mut data = cutile_comgr_data_t { handle: 0 };
        let result = cutile_comgr_create_data(
            cutile_comgr_data_kind_s::CUTILE_COMGR_DATA_KIND_BYTECODE,
            &mut data,
        );
        assert!(result.is_ok());

        let result = cutile_comgr_data_set_add(data_set, data);
        assert!(result.is_ok());

        let mut count = 0;
        let result = cutile_comgr_get_data_count(data_set, &mut count);
        assert!(result.is_ok());
        assert_eq!(count, 1);

        cutile_comgr_release_data_set(data_set).ok();
        cutile_comgr_release_data(data).ok();
    }
}
