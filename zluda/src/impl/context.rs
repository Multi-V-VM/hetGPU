use super::{driver, FromCuda, ZludaObject};
use cuda_types::cuda::*;
use rustc_hash::FxHashSet;
use std::{cell::RefCell, ptr, sync::Mutex};
use std::ffi::c_uint;

// Feature-specific imports
#[cfg(feature = "amd")]
use hip_runtime_sys::*;

#[cfg(feature = "intel")]
use std::os::raw::c_void;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;

#[cfg(feature = "tenstorrent")]
use tt_runtime_sys::*;

// Result conversion traits
#[cfg(feature = "intel")]
trait ResultExt {
    fn to_cuda_result<T>(self, value: T) -> Result<T, CUerror>;
}

#[cfg(feature = "intel")]
impl ResultExt for ze_result_t {
    fn to_cuda_result<T>(self, value: T) -> Result<T, CUerror> {
        match self {
            ze_result_t::ZE_RESULT_SUCCESS => Ok(value),
            ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY => Err(CUerror::OUT_OF_MEMORY),
            ze_result_t::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY => Err(CUerror::OUT_OF_MEMORY),
            _ => Err(CUerror::UNKNOWN),
        }
    }
}

#[cfg(feature = "tenstorrent")]
trait TTResultExt {
    fn to_cuda_result<T>(self, value: T) -> Result<T, CUerror>;
}

#[cfg(feature = "tenstorrent")]
impl<T> TTResultExt for Result<T, String> {
    fn to_cuda_result<U>(self, value: U) -> Result<U, CUerror> {
        match self {
            Ok(_) => Ok(value),
            Err(_) => Err(CUerror::UNKNOWN),
        }
    }
}

// Thread-local context stack - mutually exclusive for each backend
#[cfg(feature = "amd")]
thread_local! {
    pub(crate) static CONTEXT_STACK: RefCell<Vec<(CUcontext, hipDevice_t)>> = RefCell::new(Vec::new());
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
thread_local! {
    pub(crate) static CONTEXT_STACK: RefCell<Vec<(CUcontext, ze_device_handle_t)>> = RefCell::new(Vec::new());
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
thread_local! {
    pub(crate) static CONTEXT_STACK: RefCell<Vec<(CUcontext, i32)>> = RefCell::new(Vec::new());
}

// Context structures - AMD implementation
#[cfg(feature = "amd")]
pub(crate) struct Context {
    pub(crate) device: hipDevice_t,
    pub(crate) mutable: Mutex<OwnedByContext>,
}

#[cfg(feature = "amd")]
impl Clone for Context {
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            mutable: Mutex::new(OwnedByContext {
                ref_count: 0,
                _memory: FxHashSet::default(),
                _streams: FxHashSet::default(),
                _modules: FxHashSet::default(),
            }),
        }
    }
}

#[cfg(feature = "amd")]
pub(crate) struct OwnedByContext {
    pub(crate) ref_count: usize, 
    pub(crate) _memory: FxHashSet<hipDeviceptr_t>,
    pub(crate) _streams: FxHashSet<hipStream_t>,
    pub(crate) _modules: FxHashSet<CUmodule>,
}

#[cfg(feature = "amd")]
impl ZludaObject for Context {
    const COOKIE: usize = 0x1c9a63e0bfb35ca4;
    type CudaHandle = CUcontext;

    fn drop_checked(&mut self) -> CUresult {
        Ok(())
    }
}

#[cfg(feature = "amd")]
impl Context {
    pub(crate) fn new(device: hipDevice_t) -> Self {
        Self {
            device,
            mutable: Mutex::new(OwnedByContext {
                ref_count: 0,
                _memory: FxHashSet::default(),
                _streams: FxHashSet::default(),
                _modules: FxHashSet::default(),
            }),
        }
    }

    pub(crate) fn is_destroyed(&self) -> bool {
        let mutable = self.mutable.lock().unwrap();
        if mutable.ref_count == 0 {
            false
        } else {
            true
        }
    }
}

// Intel Level Zero implementation
#[cfg(all(feature = "intel", not(feature = "amd")))]
pub(crate) struct Context {
    pub(crate) device: ze_device_handle_t,
    pub(crate) context: ze_context_handle_t,
    pub(crate) mutable: Mutex<OwnedByContext>,
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
impl Clone for Context {
    fn clone(&self) -> Self {
        let guard = self.mutable.lock().unwrap();
        Self {
            device: self.device,
            context: self.context,
            mutable: Mutex::new(OwnedByContext {
                ref_count: guard.ref_count,
                _command_queues: guard._command_queues.clone(),
                _command_lists: guard._command_lists.clone(),
                _modules: guard._modules.clone(),
                _allocations: guard._allocations.clone(),
            }),
        }
    }
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
pub(crate) struct OwnedByContext {
    pub(crate) ref_count: usize,
    pub(crate) _command_queues: FxHashSet<ze_command_queue_handle_t>,
    pub(crate) _command_lists: FxHashSet<ze_command_list_handle_t>,
    pub(crate) _modules: FxHashSet<ze_module_handle_t>,
    pub(crate) _allocations: FxHashSet<usize>,
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
impl Context {
    pub(crate) fn new(device: ze_device_handle_t) -> Self {
        // Create Level Zero context
        let mut context_desc = ze_context_desc_t {
            stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_CONTEXT_DESC,
            pNext: ptr::null(),
            flags: 0,
        };

        let mut context_handle = ze_context_handle_t(ptr::null_mut());
        let mut drivers = vec![ze_driver_handle_t(ptr::null_mut()); 1];
        let mut driver_count = 1;
        
        unsafe {
            // This is a simplified initialization - in reality you'd need proper error handling
            let _ = zeInit(0);
            let _ = zeDriverGet(&mut driver_count, drivers.as_mut_ptr());
            let _ = zeContextCreate(drivers[0], &context_desc, &mut context_handle);
        }

        Self {
            device,
            context: context_handle,
            mutable: Mutex::new(OwnedByContext {
                ref_count: 0,
                _command_queues: FxHashSet::default(),
                _command_lists: FxHashSet::default(),
                _modules: FxHashSet::default(),
                _allocations: FxHashSet::default(),
            }),
        }
    }

    pub(crate) fn add_allocation(&self, ptr: *mut c_void) {
        let mut guard = self.mutable.lock().unwrap();
        guard._allocations.insert(ptr as usize);
    }

    pub(crate) fn remove_allocation(&self, ptr: *mut c_void) {
        let mut guard = self.mutable.lock().unwrap();
        guard._allocations.remove(&(ptr as usize));
    }

    pub(crate) fn add_command_queue(&self, queue: ze_command_queue_handle_t) {
        let mut guard = self.mutable.lock().unwrap();
        guard._command_queues.insert(queue);
    }

    pub(crate) fn add_command_list(&self, list: ze_command_list_handle_t) {
        let mut guard = self.mutable.lock().unwrap();
        guard._command_lists.insert(list);
    }

    pub(crate) fn remove_command_queue(&self, queue: ze_command_queue_handle_t) {
        let mut guard = self.mutable.lock().unwrap();
        guard._command_queues.remove(&queue);
    }

    pub(crate) fn remove_command_list(&self, list: ze_command_list_handle_t) {
        let mut guard = self.mutable.lock().unwrap();
        guard._command_lists.remove(&list);
    }

    pub(crate) fn is_destroyed(&self) -> bool {
        let mutable = self.mutable.lock().unwrap();
        mutable.ref_count == 0
    }

    pub(crate) fn initialize(&mut self) -> Result<(), CUerror> {
        // Intel Level Zero context initialization if needed
        // This is mostly a placeholder as the context is already initialized in new()
        Ok(())
    }
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
impl ZludaObject for Context {
    const COOKIE: usize = 0x1c9a63e0bfb35ca4;
    type CudaHandle = CUcontext;

    fn drop_checked(&mut self) -> CUresult {
        Ok(())
    }
}

// Tenstorrent implementation
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) struct Context {
    pub(crate) device_id: i32,
    pub(crate) device: Option<tt_runtime_sys::Device>,
    pub(crate) mutable: Mutex<OwnedByContext>,
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
unsafe impl Send for Context {}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
unsafe impl Sync for Context {}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl Clone for Context {
    fn clone(&self) -> Self {
        let guard = self.mutable.lock().unwrap();
        Self {
            device_id: self.device_id,
            device: None,
            mutable: Mutex::new(OwnedByContext {
                ref_count: guard.ref_count,
                _memory: guard._memory.clone(),
                _streams: guard._streams.clone(),
                _modules: guard._modules.clone(),
            }),
        }
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) struct OwnedByContext {
    pub(crate) ref_count: usize,
    pub(crate) _memory: FxHashSet<usize>,    
    pub(crate) _streams: FxHashSet<usize>,   
    pub(crate) _modules: FxHashSet<usize>,   
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl Context {
    pub(crate) fn new(device_id: i32) -> Self {
        Self {
            device_id,
            device: None,
            mutable: Mutex::new(OwnedByContext {
                ref_count: 0,
                _memory: FxHashSet::default(),
                _streams: FxHashSet::default(),
                _modules: FxHashSet::default(),
            }),
        }
    }

    pub(crate) fn increment_ref_count(&self) {
        let mut guard = self.mutable.lock().unwrap();
        guard.ref_count += 1;
    }

    pub(crate) fn decrement_ref_count(&self) -> usize {
        let mut guard = self.mutable.lock().unwrap();
        if guard.ref_count > 0 {
            guard.ref_count -= 1;
        }
        guard.ref_count
    }

    pub(crate) fn destroy(&self) -> Result<(), CUerror> {
        let mut guard = self.mutable.lock().unwrap();
        guard._memory.clear();
        guard._streams.clear();
        guard._modules.clear();
        Ok(())
    }

    pub(crate) fn is_destroyed(&self) -> bool {
        let mutable = self.mutable.lock().unwrap();
        mutable.ref_count == 0
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl ZludaObject for Context {
    const COOKIE: usize = 0x1c9a63e0bfb35ca4;
    type CudaHandle = CUcontext;

    fn drop_checked(&mut self) -> CUresult {
        Ok(())
    }
}

// Common functions - implemented per backend

// AMD functions
#[cfg(feature = "amd")]
pub(crate) unsafe fn get_limit(pvalue: *mut usize, limit: hipLimit_t) -> hipError_t {
    unsafe { hipDeviceGetLimit(pvalue, limit) }
}

#[cfg(feature = "amd")]
pub(crate) fn set_limit(limit: hipLimit_t, value: usize) -> hipError_t {
    unsafe { hipDeviceSetLimit(limit, value) }
}

#[cfg(feature = "amd")]
pub(crate) fn synchronize() -> hipError_t {
    unsafe { hipDeviceSynchronize() }
}

#[cfg(feature = "amd")]
pub(crate) fn get_primary(hip_dev: hipDevice_t) -> Result<(&'static Context, CUcontext), CUerror> {
    let dev = driver::device(hip_dev)?;
    Ok(dev.primary_context())
}

#[cfg(feature = "amd")]
pub(crate) fn set_current(raw_ctx: CUcontext) -> CUresult {
    let new_device = if raw_ctx.0 == ptr::null_mut() {
        CONTEXT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some((_, old_device)) = stack.pop() {
                Some(old_device)
            } else {
                None
            }
        })
    } else {
        let ctx = FromCuda::from_cuda(&raw_ctx)?;
        let new_device = ctx.device;
        CONTEXT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.push((raw_ctx, new_device));
        });
        Some(new_device)
    };

    if let Some(device) = new_device {
        let mut current = ptr::null_mut();
        unsafe { hipCtxGetCurrent(&mut current) };
        if current != raw_ctx.0 {
            unsafe { hipCtxSetCurrent(raw_ctx.0) }.unwrap();
        }
    }

    Ok(())
}

#[cfg(feature = "amd")]
pub(crate) fn get_current() -> Option<CUcontext> {
    CONTEXT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.pop().map(|(ctx, _)| ctx)
    })
}

#[cfg(feature = "amd")]
pub(crate) fn push(ctx: CUcontext, device: hipDevice_t) {
    CONTEXT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.push((ctx, device));
    });
}

#[cfg(feature = "amd")]
pub(crate) fn get_device_properties(device: hipDevice_t) -> Result<hipDeviceArch_t, CUerror> {
    let mut props = unsafe { std::mem::zeroed() };
    unsafe { hipGetDevicePropertiesR0600(&mut props, device).unwrap() };
    Ok(props.arch)
}

// Intel Level Zero functions
#[cfg(all(feature = "intel", not(feature = "amd")))]
pub(crate) unsafe fn get_limit(_pvalue: *mut usize, _limit: c_uint) -> ze_result_t {
    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
pub(crate) fn set_limit(_limit: c_uint, _value: usize) -> ze_result_t {
    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
pub(crate) fn synchronize() -> ze_result_t {
    let ctx = match get_current() {
        Some(ctx) => ctx,
        None => return ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT,
    };

    let ze_ctx: &Context = match FromCuda::from_cuda(&ctx) {
        Ok(ctx) => ctx,
        Err(_) => return ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT,
    };

    let guard = ze_ctx.mutable.lock().unwrap();
    for &queue in &guard._command_queues {
        unsafe {
            let result = zeCommandQueueSynchronize(queue, u64::MAX);
            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return result;
            }
        }
    }

    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
pub(crate) fn from_cuda_to(ctx: &CUcontext) -> Result<&Context, CUerror> {
    FromCuda::from_cuda(ctx)
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
pub(crate) fn set_current(raw_ctx: CUcontext) -> CUresult {
    let _new_device = if raw_ctx.0 == ptr::null_mut() {
        CONTEXT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some((_, _)) = stack.pop() {
                // Device switching would be handled here
            }
        })
    } else {
        let ctx: &Context = FromCuda::from_cuda(&raw_ctx)?;
        let new_device = ctx.device;
        CONTEXT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.push((raw_ctx, new_device));
        });
    };

    Ok(())
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
pub(crate) fn push(ctx: CUcontext, device: ze_device_handle_t) {
    CONTEXT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.push((ctx, device));
    });
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
pub(crate) fn get_device_properties(
    device: ze_device_handle_t,
) -> Result<ze_device_properties_t, CUerror> {
    let mut props: ze_device_properties_t = unsafe { std::mem::zeroed() };
    props.stype = ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    
    unsafe { zeDeviceGetProperties(device, &mut props).to_cuda_result(props) }
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
pub(crate) fn get_current_ze() -> Result<&'static Context, CUerror> {
    let current_ctx = CONTEXT_STACK
        .with(|stack| stack.borrow().last().map(|(ctx, _)| *ctx))
        .ok_or(CUerror::INVALID_CONTEXT)?;

    let context: &Context = FromCuda::from_cuda(&current_ctx)?;
    Ok(unsafe { std::mem::transmute(context) })
}

#[cfg(all(feature = "intel", not(feature = "amd")))]
pub(crate) fn get_primary_ze(
    device: ze_device_handle_t,
) -> Result<(&'static Context, CUcontext), CUerror> {
    let dev = driver::device_ze(device)?;
    Ok(dev.primary_context())
}

// Tenstorrent functions
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) unsafe fn get_limit(_pvalue: *mut usize, _limit: c_uint) -> Result<(), String> {
    if !_pvalue.is_null() {
        unsafe { *_pvalue = 0 };
    }
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn set_limit(_limit: c_uint, _value: usize) -> Result<(), String> {
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn synchronize() -> Result<(), String> {
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_primary(device_id: i32) -> Result<(&'static Context, CUcontext), CUerror> {
    let dev = driver::device_tt(device_id)?;
    Ok(dev.primary_context())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn set_current(raw_ctx: CUcontext) -> CUresult {
    let _new_device_id = if raw_ctx.0 == ptr::null_mut() {
        CONTEXT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some((_, old_device_id)) = stack.pop() {
                Some(old_device_id)
            } else {
                None
            }
        })
    } else {
        let ctx: &Context = FromCuda::from_cuda(&raw_ctx)?;
        let new_device_id = ctx.device_id;
        CONTEXT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.push((raw_ctx, new_device_id));
        });
        Some(new_device_id)
    };

    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn push(ctx: CUcontext, device_id: i32) {
    CONTEXT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.push((ctx, device_id));
    });
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_device_properties(device_id: i32) -> Result<String, CUerror> {
    let tt_device = tt_runtime_sys::Device::new(device_id as u32)
        .map_err(|_| CUerror::INVALID_DEVICE)?;
    
    let device_name = tt_device.get_name()
        .map_err(|_| CUerror::UNKNOWN)?;
    
    Ok(device_name)
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_current_tt() -> Result<&'static Context, CUerror> {
    let current = get_current().ok_or(CUerror::INVALID_CONTEXT)?;
    let context: &Context = FromCuda::from_cuda(&current)?;
    Ok(unsafe { std::mem::transmute(context) })
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_primary_tt(device_id: i32) -> Result<(&'static Context, CUcontext), CUerror> {
    let dev = driver::device_tt(device_id)?;
    Ok(dev.primary_context())
}

// Common functions that work across all backends
pub(crate) fn get_current() -> Option<CUcontext> {
    #[cfg(feature = "amd")]
    {
        CONTEXT_STACK.with(|stack| {
            let stack = stack.borrow();
            stack.last().map(|(ctx, _)| *ctx)
        })
    }
    #[cfg(all(feature = "intel", not(feature = "amd")))]
    {
        CONTEXT_STACK.with(|stack| {
            let stack = stack.borrow();
            stack.last().map(|(ctx, _)| *ctx)
        })
    }
    #[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
    {
        CONTEXT_STACK.with(|stack| {
            let stack = stack.borrow();
            stack.last().map(|(ctx, _)| *ctx)
        })
    }
}