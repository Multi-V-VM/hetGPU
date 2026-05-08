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
use std::{ffi::CString, sync::OnceLock};
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
use tt_runtime_sys::*;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

use std::{cell::RefCell, collections::HashMap, ptr::NonNull};

use crate::r#impl::context;

use super::LiveCheck;

#[cfg(unix)]
use libc::{dlsym, RTLD_DEFAULT};

/// Get a handle to our own .so (libnvcuda.so) so we can resolve cu* symbols
/// from OUR library, not the system's /lib/x86_64-linux-gnu/libcuda.so.1.
#[cfg(unix)]
fn get_self_library_handle() -> *mut c_void {
    use std::sync::Once;
    static INIT: Once = Once::new();
    static mut SELF_HANDLE: *mut c_void = std::ptr::null_mut();

    INIT.call_once(|| {
        unsafe {
            // Use dladdr on cuInit (which we know is in our library) to find our .so path
            let mut info: libc::Dl_info = std::mem::zeroed();
            // Use a known exported function as the reference point
            extern "C" {
                fn cuInit(flags: std::os::raw::c_uint) -> std::os::raw::c_int;
            }
            let func_ptr = cuInit as *const c_void;
            if libc::dladdr(func_ptr, &mut info) != 0 && !info.dli_fname.is_null() {
                let handle = libc::dlopen(info.dli_fname, libc::RTLD_NOLOAD | libc::RTLD_NOW);
                if !handle.is_null() {
                    let fname = CStr::from_ptr(info.dli_fname).to_string_lossy();
                    eprintln!(
                        "[hetGPU] cuGetProcAddress: self library = {} (handle={:p})",
                        fname, handle
                    );
                    SELF_HANDLE = handle;
                }
            }
        }
    });

    unsafe { SELF_HANDLE }
}

// CUDA driver entry point lookups
pub(crate) fn get_proc_address(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    _cuda_version: c_int,
    _flags: cuda_types::cuda::cuuint64_t,
) -> Result<(), CUerror> {
    if symbol.is_null() || pfn.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    #[cfg(unix)]
    unsafe {
        let sym_str = std::ffi::CStr::from_ptr(symbol).to_string_lossy();
        // First try our own library to avoid resolving to system's libcuda.so.1
        let self_handle = get_self_library_handle();
        let mut addr = std::ptr::null_mut();
        if !self_handle.is_null() {
            addr = dlsym(self_handle, symbol);
        }
        if addr.is_null() {
            // Fallback to global search
            addr = dlsym(RTLD_DEFAULT, symbol);
        }
        if addr.is_null() {
            eprintln!("[hetGPU] cuGetProcAddress: '{}' NOT FOUND", sym_str);
            *pfn = std::ptr::null_mut();
            return Err(CUerror::NOT_FOUND);
        }
        *pfn = addr as *mut c_void;
    }

    Ok(())
}

pub(crate) fn get_proc_address_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    _cuda_version: c_int,
    _flags: cuda_types::cuda::cuuint64_t,
    symbol_status: *mut cuda_types::cuda::CUdriverProcAddressQueryResult,
) -> Result<(), CUerror> {
    if symbol.is_null() || pfn.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    #[cfg(unix)]
    unsafe {
        // First try our own library to avoid resolving to system's libcuda.so.1
        let self_handle = get_self_library_handle();
        let mut addr = std::ptr::null_mut();
        if !self_handle.is_null() {
            addr = dlsym(self_handle, symbol);
        }
        if addr.is_null() {
            // Fallback to global search
            addr = dlsym(RTLD_DEFAULT, symbol);
        }
        *pfn = addr as *mut c_void;
        if !symbol_status.is_null() {
            (*symbol_status).0 = if addr.is_null() { 1 } else { 0 };
        }
        if addr.is_null() {
            return Err(CUerror::NOT_FOUND);
        }
    }

    Ok(())
}

pub(crate) fn get_export_table(
    pp_export_table: *mut *const c_void,
    _p_export_table_id: *const CUuuid,
) -> Result<(), CUerror> {
    unsafe {
        if !pp_export_table.is_null() {
            *pp_export_table = std::ptr::null();
        }
    }
    Ok(())
}

// Provide safe implementations for cuGetErrorString/cuGetErrorName to avoid NULL deref in callers
pub(crate) fn get_error_string(error: CUresult, pStr: &mut *const c_char) -> Result<(), CUerror> {
    // Stable static messages
    static OK_STR: &[u8] = b"CUDA_SUCCESS\0";
    static ERR_STR: &[u8] = b"CUDA_ERROR_UNKNOWN\0";
    let msg_ptr = match error {
        Ok(()) => OK_STR.as_ptr() as *const c_char,
        Err(_e) => ERR_STR.as_ptr() as *const c_char,
    };
    *pStr = msg_ptr;
    Ok(())
}

pub(crate) fn get_error_name(error: CUresult, pStr: &mut *const c_char) -> Result<(), CUerror> {
    static OK_NAME: &[u8] = b"CUDA_SUCCESS\0";
    static ERR_NAME: &[u8] = b"CUDA_ERROR_UNKNOWN\0";
    let name_ptr = match error {
        Ok(()) => OK_NAME.as_ptr() as *const c_char,
        Err(_e) => ERR_NAME.as_ptr() as *const c_char,
    };
    *pStr = name_ptr;
    Ok(())
}

// Define trait for converting ze_result_t to CUresult
#[cfg(feature = "intel")]
trait ResultExt {
    fn to_cuda_result<T>(self, value: T) -> Result<T, CUerror>;
}

#[cfg(feature = "intel")]
impl ResultExt for ze_result_t {
    fn to_cuda_result<T>(self, value: T) -> Result<T, CUerror> {
        match self {
            ze_result_t::ZE_RESULT_SUCCESS => Ok(value),
            ze_result_t::ZE_RESULT_ERROR_DEVICE_LOST => Err(CUerror::DEVICE_NOT_LICENSED),
            ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY => Err(CUerror::OUT_OF_MEMORY),
            ze_result_t::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY => Err(CUerror::OUT_OF_MEMORY),
            _ => Err(CUerror::UNKNOWN),
        }
    }
}

// Define trait for converting tt_runtime_sys results to CUresult
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
trait TTResultExt {
    fn to_cuda_result<T>(self, value: T) -> Result<T, CUerror>;
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl TTResultExt for Result<(), String> {
    fn to_cuda_result<T>(self, value: T) -> Result<T, CUerror> {
        match self {
            Ok(_) => Ok(value),
            Err(_) => Err(CUerror::UNKNOWN),
        }
    }
}

pub(crate) struct GlobalState {
    pub devices: Vec<Device>,
}
unsafe impl Send for GlobalState {}
unsafe impl Sync for GlobalState {}

pub(crate) struct Device {
    pub(crate) _comgr_isa: CString,
    primary_context: LiveCheck<context::Context>,
}
impl Clone for Device {
    fn clone(&self) -> Self {
        Self {
            _comgr_isa: self._comgr_isa.clone(),
            primary_context: LiveCheck::new(unsafe {
                self.primary_context.data.assume_init_ref().clone()
            }),
        }
    }
}
impl Device {
    pub(crate) fn primary_context<'a>(&'a self) -> (&'a context::Context, CUcontext) {
        unsafe {
            (
                self.primary_context.data.assume_init_ref(),
                self.primary_context.as_handle(),
            )
        }
    }
}
pub(crate) fn device(dev: i32) -> Result<&'static Device, CUerror> {
    global_state()?
        .devices
        .get(dev as usize)
        .ok_or(CUerror::INVALID_DEVICE)
}
#[cfg(feature = "amd")]
pub(crate) fn global_state() -> Result<&'static GlobalState, CUerror> {
    static GLOBAL_STATE: OnceLock<Result<GlobalState, CUerror>> = OnceLock::new();
    fn cast_slice<'a>(bytes: &'a [i8]) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len()) }
    }
    GLOBAL_STATE
        .get_or_init(|| {
            let mut device_count = 0;
            unsafe { hipGetDeviceCount(&mut device_count).unwrap() };
            Ok(GlobalState {
                devices: (0..device_count)
                    .map(|i| {
                        let mut props = unsafe { std::mem::zeroed() };
                        unsafe { hipGetDeviceProperties(&mut props, i).unwrap() };
                        Ok::<_, CUerror>(Device {
                            _comgr_isa: CStr::from_bytes_until_nul(cast_slice(
                                &props.gcnArchName[..],
                            ))
                            .map_err(|_| CUerror::UNKNOWN)?
                            .to_owned(),
                            primary_context: LiveCheck::new(context::Context::new(i)),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            })
        })
        .as_ref()
        .map_err(|e| *e)
}

#[cfg(feature = "intel")]
pub(crate) fn global_state() -> Result<&'static GlobalState, CUerror> {
    static GLOBAL_STATE: OnceLock<Result<GlobalState, CUerror>> = OnceLock::new();

    GLOBAL_STATE
        .get_or_init(|| {
            // Helper function to create virtual device
            fn create_virtual_device(reason: &str) -> Result<GlobalState, CUerror> {
                eprintln!(
                    "[Intel Backend] {} - creating virtual device for TMatmul",
                    reason
                );
                let comgr_isa = CString::new("virtual-gpu").map_err(|_| CUerror::UNKNOWN)?;
                let virtual_device = ze_device_handle_t(std::ptr::null_mut());
                let ctx = context::Context::new(virtual_device);

                let device = Device {
                    _comgr_isa: comgr_isa,
                    primary_context: LiveCheck::new(ctx),
                };

                let device_box = Box::new(device.clone());
                let device_box_clone = Box::new(*device_box.clone());

                DEVICES_ZE.with(|map| {
                    let mut map = map.borrow_mut();
                    let device_ptr =
                        unsafe { NonNull::new_unchecked(Box::into_raw(device_box_clone)) };
                    map.insert(virtual_device, device_ptr);
                });

                // Also store ordinal-to-handle mapping for virtual device (ordinal 0)
                ORDINAL_TO_ZE.with(|map| {
                    let mut map = map.borrow_mut();
                    map.insert(0, virtual_device);
                });

                Ok(GlobalState {
                    devices: vec![*device_box],
                })
            }

            // Initialize Level Zero
            let ze_init_result = unsafe { zeInit(0) };
            if ze_init_result != ze_result_t::ZE_RESULT_SUCCESS {
                return create_virtual_device("Level Zero init failed");
            }

            // Get driver count
            let mut driver_count = 0;
            let driver_result = unsafe { zeDriverGet(&mut driver_count, std::ptr::null_mut()) };
            if driver_result != ze_result_t::ZE_RESULT_SUCCESS || driver_count == 0 {
                return create_virtual_device("No Intel drivers found");
            }

            // Get drivers
            let mut drivers = vec![std::ptr::null_mut(); driver_count as usize];
            let driver_get_result =
                unsafe { zeDriverGet(&mut driver_count, *drivers.as_mut_ptr()) };
            if driver_get_result != ze_result_t::ZE_RESULT_SUCCESS {
                return create_virtual_device("Failed to get drivers");
            }

            // Get device count for the first driver
            let mut device_count = 0;
            let device_result =
                unsafe { zeDeviceGet(*drivers[0], &mut device_count, std::ptr::null_mut()) };
            if device_result != ze_result_t::ZE_RESULT_SUCCESS || device_count == 0 {
                return create_virtual_device("No Intel devices on driver");
            }

            // Real devices found - enumerate them
            let mut devices_vec = Vec::new();

            for i in 0..device_count as i32 {
                let mut devices = vec![std::ptr::null_mut(); device_count as usize];

                // Get the devices
                unsafe {
                    zeDeviceGet(*drivers[0], &mut device_count, *devices.as_mut_ptr())
                        .to_cuda_result(())?;
                }

                let device = if i < devices.len() as i32 {
                    ze_device_handle_t(devices[i as usize] as *mut _)
                } else {
                    return Err(CUerror::INVALID_DEVICE);
                };

                // Get device properties
                let mut props: ze_device_properties_t = unsafe { std::mem::zeroed() };
                props.stype = ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

                unsafe {
                    zeDeviceGetProperties(device, &mut props).to_cuda_result(())?;
                }

                // Create a string from device name - ensure we only include valid characters
                let name_bytes = props
                    .name
                    .iter()
                    .take_while(|&&c| c != 0)
                    .map(|&c| c as u8)
                    .collect::<Vec<_>>();

                let comgr_isa = CString::new(name_bytes).map_err(|_| CUerror::UNKNOWN)?;

                // Create context
                let mut ctx = context::Context::new(device);

                // Initialize the context
                ctx.initialize()?;

                // Create the device and store it in our results
                let device_box = Box::new(Device {
                    _comgr_isa: comgr_isa,
                    primary_context: LiveCheck::new(ctx),
                });

                // 使用克隆，这样原始值不会被移动
                let device_box_clone = Box::new(*device_box.clone());

                // 存储指针到设备映射
                DEVICES_ZE.with(|map| {
                    let mut map = map.borrow_mut();
                    let device_ptr =
                        unsafe { NonNull::new_unchecked(Box::into_raw(device_box_clone)) };
                    map.insert(device, device_ptr);
                });

                // Store ordinal-to-handle mapping for real device
                ORDINAL_TO_ZE.with(|map| {
                    let mut map = map.borrow_mut();
                    map.insert(i, device);
                });

                devices_vec.push(*device_box);
            }

            Ok(GlobalState {
                devices: devices_vec,
            })
        })
        .as_ref()
        .map_err(|e| *e)
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn global_state() -> Result<&'static GlobalState, CUerror> {
    static GLOBAL_STATE: OnceLock<Result<GlobalState, CUerror>> = OnceLock::new();

    GLOBAL_STATE
        .get_or_init(|| {
            // Get device count using tt_runtime_sys
            let device_count = match tt_runtime_sys::get_device_count() {
                Ok(count) => count,
                Err(_) => return Err(CUerror::NO_DEVICE),
            };

            if device_count == 0 {
                return Err(CUerror::NO_DEVICE);
            }

            let mut devices_vec = Vec::new();

            for i in 0..device_count as i32 {
                // Create device
                let tt_device = match tt_runtime_sys::Device::new(i as u32) {
                    Ok(dev) => dev,
                    Err(_) => return Err(CUerror::INVALID_DEVICE),
                };

                // Get device name
                let device_name = match tt_device.get_name() {
                    Ok(name) => name,
                    Err(_) => return Err(CUerror::UNKNOWN),
                };

                let comgr_isa = CString::new(device_name).map_err(|_| CUerror::UNKNOWN)?;

                // Create context
                let ctx = context::Context::new(i);

                // Create the device object for our GlobalState
                let device = Device {
                    _comgr_isa: comgr_isa,
                    primary_context: LiveCheck::new(ctx),
                };

                // Store device in TT_DEVICES map
                let device_box = Box::new(device.clone());
                let device_box_clone = Box::new(*device_box.clone());

                TT_DEVICES.with(|map| {
                    let mut map = map.borrow_mut();
                    let device_ptr =
                        unsafe { NonNull::new_unchecked(Box::into_raw(device_box_clone)) };
                    map.insert(i, device_ptr);
                });

                devices_vec.push(*device_box);
            }

            Ok(GlobalState {
                devices: devices_vec,
            })
        })
        .as_ref()
        .map_err(|e| *e)
}

#[cfg(feature = "amd")]
pub(crate) fn init(flags: ::core::ffi::c_uint) -> CUresult {
    unsafe { hipInit(flags).unwrap() };
    global_state()?;

    // Install checkpoint signal handler if enabled
    if std::env::var("HETGPU_CHECKPOINT_ENABLED").ok().as_deref() != Some("0") {
        if let Err(e) = super::checkpoint::install_signal_handler() {
            eprintln!(
                "[hetGPU] Warning: Failed to install checkpoint handler: {}",
                e
            );
        }
    }

    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn init(flags: ::core::ffi::c_uint) -> CUresult {
    eprintln!("[hetGPU] cuInit called with flags={}", flags);
    unsafe {
        // Initialize Level Zero when available; fall back to virtual device otherwise
        match zeInit(0) {
            ze_result_t::ZE_RESULT_SUCCESS => {
                eprintln!("[hetGPU] cuInit: Level Zero init succeeded");
            }
            err => {
                eprintln!(
                    "[hetGPU] Level Zero init failed with {:?} - continuing with virtual backend",
                    err
                );
            }
        }

        // Ignore CUDA flags as they don't apply to Level Zero
        let _ = flags;
    }

    match global_state() {
        Ok(_) => eprintln!("[hetGPU] cuInit: global_state succeeded"),
        Err(e) => {
            eprintln!("[hetGPU] cuInit: global_state failed with {:?}", e);
            return Err(e);
        }
    }

    // Install checkpoint signal handler if enabled
    if std::env::var("HETGPU_CHECKPOINT_ENABLED").ok().as_deref() != Some("0") {
        if let Err(e) = super::checkpoint::install_signal_handler() {
            eprintln!(
                "[hetGPU] Warning: Failed to install checkpoint handler: {}",
                e
            );
        }
    }

    eprintln!("[hetGPU] cuInit: returning success");
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn init(flags: ::core::ffi::c_uint) -> CUresult {
    // Tenstorrent initialization
    // The tt_runtime_sys doesn't require explicit initialization like CUDA/HIP
    // We just ignore the flags as they don't apply to Tenstorrent
    let _ = flags;

    global_state()?;

    // Install checkpoint signal handler if enabled
    if std::env::var("HETGPU_CHECKPOINT_ENABLED").ok().as_deref() != Some("0") {
        if let Err(e) = super::checkpoint::install_signal_handler() {
            eprintln!(
                "[hetGPU] Warning: Failed to install checkpoint handler: {}",
                e
            );
        }
    }

    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn global_state() -> Result<&'static GlobalState, CUerror> {
    static GLOBAL_STATE: OnceLock<Result<GlobalState, CUerror>> = OnceLock::new();

    GLOBAL_STATE
        .get_or_init(|| {
            // TMatmul is a compiler backend with one virtual device
            let comgr_isa = CString::new("tmatmul-v1").map_err(|_| CUerror::UNKNOWN)?;
            let ctx = context::Context::new(0);

            let device = Device {
                _comgr_isa: comgr_isa,
                primary_context: LiveCheck::new(ctx),
            };

            let device_box = Box::new(device.clone());
            let device_box_clone = Box::new(*device_box.clone());

            TMATMUL_DEVICES.with(|map| {
                let mut map = map.borrow_mut();
                let device_ptr = unsafe { NonNull::new_unchecked(Box::into_raw(device_box_clone)) };
                map.insert(0, device_ptr);
            });

            Ok(GlobalState {
                devices: vec![*device_box],
            })
        })
        .as_ref()
        .map_err(|e| *e)
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn init(flags: ::core::ffi::c_uint) -> CUresult {
    let _ = flags;
    global_state()?;

    // Install checkpoint signal handler if enabled
    if std::env::var("HETGPU_CHECKPOINT_ENABLED").ok().as_deref() != Some("0") {
        if let Err(e) = super::checkpoint::install_signal_handler() {
            eprintln!(
                "[hetGPU] Warning: Failed to install checkpoint handler: {}",
                e
            );
        }
    }

    Ok(())
}

#[cfg(feature = "amd")]
pub(crate) fn get_version(version: &mut ::core::ffi::c_int) -> CUresult {
    *version = std::cmp::max(cuda_types::cuda::CUDA_VERSION as i32, 13000);
    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn get_version(version: &mut ::core::ffi::c_int) -> CUresult {
    eprintln!("[hetGPU] cuDriverGetVersion called");
    // Return the CUDA version same as AMD implementation
    *version = std::cmp::max(cuda_types::cuda::CUDA_VERSION as i32, 13000);
    eprintln!(
        "[hetGPU] cuDriverGetVersion: returning version {}",
        *version
    );
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_version(version: &mut ::core::ffi::c_int) -> CUresult {
    // Return the CUDA version same as other implementations
    *version = std::cmp::max(cuda_types::cuda::CUDA_VERSION as i32, 13000);
    Ok(())
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_version(version: &mut ::core::ffi::c_int) -> CUresult {
    *version = std::cmp::max(cuda_types::cuda::CUDA_VERSION as i32, 13000);
    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn device_ze(dev: ze_device_handle_t) -> Result<&'static Device, CUerror> {
    DEVICES_ZE.with(|map| {
        let map_ref = map.borrow();
        map_ref
            .get(&dev)
            .ok_or(CUerror::INVALID_DEVICE)
            .map(|dev_ptr| unsafe { dev_ptr.as_ref() })
    })
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn device_tt(dev_id: i32) -> Result<&'static Device, CUerror> {
    TT_DEVICES.with(|map| {
        let map_ref = map.borrow();
        map_ref
            .get(&dev_id)
            .ok_or(CUerror::INVALID_DEVICE)
            .map(|dev_ptr| unsafe { dev_ptr.as_ref() })
    })
}

#[cfg(feature = "intel")]
thread_local! {
    static DEVICES_ZE: RefCell<HashMap<ze_device_handle_t, NonNull<Device>>> = RefCell::new(HashMap::new());
    // Ordinal to ze_device_handle_t mapping for device lookup by index
    static ORDINAL_TO_ZE: RefCell<HashMap<i32, ze_device_handle_t>> = RefCell::new(HashMap::new());
}

/// Get ze_device_handle_t by device ordinal (for Intel backend)
#[cfg(feature = "intel")]
pub(crate) fn get_ze_handle_by_ordinal(ordinal: i32) -> Result<ze_device_handle_t, CUerror> {
    // Get device from global state and extract the ze_device_handle_t from its context
    // This avoids thread-local storage issues
    let dev = device(ordinal)?;
    let (ctx, _) = dev.primary_context();
    Ok(ctx.device)
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
thread_local! {
    static TT_DEVICES: RefCell<HashMap<i32, NonNull<Device>>> = RefCell::new(HashMap::new());
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn device_tmatmul(dev_id: i32) -> Result<&'static Device, CUerror> {
    TMATMUL_DEVICES.with(|map| {
        let map_ref = map.borrow();
        map_ref
            .get(&dev_id)
            .ok_or(CUerror::INVALID_DEVICE)
            .map(|dev_ptr| unsafe { dev_ptr.as_ref() })
    })
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
thread_local! {
    static TMATMUL_DEVICES: RefCell<HashMap<i32, NonNull<Device>>> = RefCell::new(HashMap::new());
}

// NVIDIA backend - pass through to real libcuda.so
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn global_state() -> Result<&'static GlobalState, CUerror> {
    static GLOBAL_STATE: OnceLock<Result<GlobalState, CUerror>> = OnceLock::new();

    GLOBAL_STATE
        .get_or_init(|| {
            // Initialize NVIDIA CUDA via nvidia_runtime_sys
            let init_result = nvidia_runtime_sys::cuInit(0);
            if init_result != 0 {
                eprintln!("[NVIDIA Backend] cuInit failed with error {}", init_result);
                return Err(CUerror::NOT_INITIALIZED);
            }

            // Get device count
            let mut device_count: i32 = 0;
            let count_result = nvidia_runtime_sys::cuDeviceGetCount(&mut device_count);
            if count_result != 0 || device_count == 0 {
                eprintln!("[NVIDIA Backend] No NVIDIA devices found");
                return Err(CUerror::NO_DEVICE);
            }

            eprintln!("[NVIDIA Backend] Found {} NVIDIA device(s)", device_count);

            let mut devices_vec = Vec::new();

            for i in 0..device_count {
                // Get device handle
                let mut cuda_device: i32 = 0;
                let dev_result = nvidia_runtime_sys::cuDeviceGet(&mut cuda_device, i);
                if dev_result != 0 {
                    eprintln!("[NVIDIA Backend] Failed to get device {}", i);
                    continue;
                }

                // Get device name
                let mut name_buf = [0i8; 256];
                let name_result =
                    nvidia_runtime_sys::cuDeviceGetName(name_buf.as_mut_ptr(), 256, cuda_device);
                let device_name = if name_result == 0 {
                    let name_cstr = unsafe { CStr::from_ptr(name_buf.as_ptr()) };
                    name_cstr.to_str().unwrap_or("Unknown").to_string()
                } else {
                    "Unknown".to_string()
                };

                eprintln!("[NVIDIA Backend] Device {}: {}", i, device_name);

                let comgr_isa = CString::new(device_name).map_err(|_| CUerror::UNKNOWN)?;

                // Create a primary CUDA context for this device
                let mut cuda_ctx = cuda_types::cuda::CUcontext(std::ptr::null_mut());
                let ctx_result =
                    nvidia_runtime_sys::cuDevicePrimaryCtxRetain(&mut cuda_ctx, cuda_device);
                if ctx_result != 0 {
                    eprintln!(
                        "[NVIDIA Backend] Failed to retain primary context for device {}: error {}",
                        i, ctx_result
                    );
                    continue;
                }

                // Create context for this device
                let ctx = context::Context::new(cuda_device, cuda_ctx);

                let device = Device {
                    _comgr_isa: comgr_isa,
                    primary_context: LiveCheck::new(ctx),
                };

                // Store device in NVIDIA_DEVICES map
                let device_box = Box::new(device.clone());
                let device_box_clone = Box::new(*device_box.clone());

                NVIDIA_DEVICES.with(|map| {
                    let mut map = map.borrow_mut();
                    let device_ptr =
                        unsafe { NonNull::new_unchecked(Box::into_raw(device_box_clone)) };
                    map.insert(i, device_ptr);
                });

                devices_vec.push(*device_box);
            }

            if devices_vec.is_empty() {
                return Err(CUerror::NO_DEVICE);
            }

            Ok(GlobalState {
                devices: devices_vec,
            })
        })
        .as_ref()
        .map_err(|e| *e)
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) fn init(flags: ::core::ffi::c_uint) -> CUresult {
    // Call real cuInit
    let result = nvidia_runtime_sys::cuInit(flags);
    if result != 0 {
        eprintln!("[NVIDIA Backend] cuInit failed with error {}", result);
        return Err(CUerror::NOT_INITIALIZED);
    }

    global_state()?;

    // Install checkpoint signal handler if enabled
    if std::env::var("HETGPU_CHECKPOINT_ENABLED").ok().as_deref() != Some("0") {
        if let Err(e) = super::checkpoint::install_signal_handler() {
            eprintln!(
                "[hetGPU] Warning: Failed to install checkpoint handler: {}",
                e
            );
        }
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
pub(crate) fn get_version(version: &mut ::core::ffi::c_int) -> CUresult {
    // Get version from real CUDA driver
    let result = nvidia_runtime_sys::cuDriverGetVersion(version);
    if result != 0 {
        // Fallback to our version
        *version = std::cmp::max(cuda_types::cuda::CUDA_VERSION as i32, 13000);
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
pub(crate) fn device_nvidia(dev_id: i32) -> Result<&'static Device, CUerror> {
    NVIDIA_DEVICES.with(|map| {
        let map_ref = map.borrow();
        map_ref
            .get(&dev_id)
            .ok_or(CUerror::INVALID_DEVICE)
            .map(|dev_ptr| unsafe { dev_ptr.as_ref() })
    })
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
thread_local! {
    static NVIDIA_DEVICES: RefCell<HashMap<i32, NonNull<Device>>> = RefCell::new(HashMap::new());
}

// ============================================================================
// PACC backend driver implementations (SiFive Intelligence XM / RISC-V IME)
// ============================================================================

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn global_state() -> Result<&'static GlobalState, CUerror> {
    static GLOBAL_STATE: OnceLock<Result<GlobalState, CUerror>> = OnceLock::new();

    GLOBAL_STATE
        .get_or_init(|| {
            let visible_devices = std::env::var("HETGPU_PACC_VISIBLE_DEVICES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4)
                .clamp(1, 4);

            // PACC can expose one logical CUDA device per physical PACC.
            let comgr_isa = CString::new("riscv64-pacc-ime").map_err(|_| CUerror::UNKNOWN)?;
            let mut devices = Vec::with_capacity(visible_devices);

            PACC_DEVICES.with(|map| {
                let mut map = map.borrow_mut();
                for dev_id in 0..visible_devices {
                    let device = Device {
                        _comgr_isa: comgr_isa.clone(),
                        primary_context: LiveCheck::new(context::Context::new(dev_id as i32)),
                    };
                    let device_box = Box::new(device.clone());
                    let device_ptr =
                        unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(device.clone()))) };
                    map.insert(dev_id as i32, device_ptr);
                    devices.push(*device_box);
                }
            });

            eprintln!(
                "[PACC Backend] exposing {} CUDA-visible PACC device(s)",
                visible_devices
            );

            Ok(GlobalState { devices })
        })
        .as_ref()
        .map_err(|e| *e)
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn init(flags: ::core::ffi::c_uint) -> CUresult {
    let _ = flags;
    global_state()?;

    // Install checkpoint signal handler if enabled
    if std::env::var("HETGPU_CHECKPOINT_ENABLED").ok().as_deref() != Some("0") {
        if let Err(e) = super::checkpoint::install_signal_handler() {
            eprintln!(
                "[hetGPU] Warning: Failed to install checkpoint handler: {}",
                e
            );
        }
    }

    eprintln!("[PACC Backend] cuInit: RISC-V IME backend initialized");
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_version(version: &mut ::core::ffi::c_int) -> CUresult {
    *version = std::cmp::max(cuda_types::cuda::CUDA_VERSION as i32, 13000);
    Ok(())
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn device_pacc(dev_id: i32) -> Result<&'static Device, CUerror> {
    if dev_id < 0 {
        return Err(CUerror::INVALID_DEVICE);
    }
    global_state()?
        .devices
        .get(dev_id as usize)
        .ok_or(CUerror::INVALID_DEVICE)
}

#[cfg(all(
    feature = "pacc",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
thread_local! {
    static PACC_DEVICES: RefCell<HashMap<i32, NonNull<Device>>> = RefCell::new(HashMap::new());
}
