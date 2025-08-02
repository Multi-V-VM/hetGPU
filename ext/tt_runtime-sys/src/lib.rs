// Generated automatically by zluda_bindgen
// DO NOT EDIT MANUALLY
#![allow(warnings)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::ffi::{c_char, c_int, c_uchar, c_uint, c_void, CString};
use std::ptr;

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct CoreCoord {
    pub x: u32,
    pub y: u32,
}

pub const tt_DataFormat_tt_DataFormat_Invalid: tt_DataFormat = 0;
pub const tt_DataFormat_tt_DataFormat_Bfp8_b: tt_DataFormat = 1;
pub const tt_DataFormat_tt_DataFormat_Bfp4_b: tt_DataFormat = 2;
pub const tt_DataFormat_tt_DataFormat_Float32: tt_DataFormat = 3;
pub const tt_DataFormat_tt_DataFormat_Bfp2_b: tt_DataFormat = 4;
pub const tt_DataFormat_tt_DataFormat_Float16_b: tt_DataFormat = 5;
pub const tt_DataFormat_tt_DataFormat_Bfp8: tt_DataFormat = 6;
pub const tt_DataFormat_tt_DataFormat_Bfp4: tt_DataFormat = 7;
pub const tt_DataFormat_tt_DataFormat_Bfp2: tt_DataFormat = 8;
pub const tt_DataFormat_tt_DataFormat_Float16: tt_DataFormat = 9;
pub const tt_DataFormat_tt_DataFormat_RawUInt8: tt_DataFormat = 10;
pub const tt_DataFormat_tt_DataFormat_RawUInt16: tt_DataFormat = 11;
pub const tt_DataFormat_tt_DataFormat_RawUInt32: tt_DataFormat = 12;
pub const tt_DataFormat_tt_DataFormat_Int8: tt_DataFormat = 13;
pub const tt_DataFormat_tt_DataFormat_UInt8: tt_DataFormat = 14;
pub const tt_DataFormat_tt_DataFormat_UInt16: tt_DataFormat = 15;
pub const tt_DataFormat_tt_DataFormat_Int32: tt_DataFormat = 16;
pub const tt_DataFormat_tt_DataFormat_UInt32: tt_DataFormat = 17;
pub type tt_DataFormat = ::core::ffi::c_uint;

pub const tt_BufferType_tt_BufferType_DRAM: tt_BufferType = 0;
pub const tt_BufferType_tt_BufferType_L1: tt_BufferType = 1;
pub const tt_BufferType_tt_BufferType_L1_SMALL: tt_BufferType = 2;
pub const tt_BufferType_tt_BufferType_TRACE: tt_BufferType = 3;
pub const tt_BufferType_tt_BufferType_SYSTEM_MEMORY: tt_BufferType = 4;
pub type tt_BufferType = ::core::ffi::c_uint;

pub const tt_DataMovementProcessor_tt_DataMovementProcessor_RISCV_0: tt_DataMovementProcessor = 0;
pub const tt_DataMovementProcessor_tt_DataMovementProcessor_RISCV_1: tt_DataMovementProcessor = 1;
pub type tt_DataMovementProcessor = ::core::ffi::c_uint;

pub const tt_NOC_tt_NOC_RISCV_0_default: tt_NOC = 0;
pub const tt_NOC_tt_NOC_RISCV_1_default: tt_NOC = 1;
pub type tt_NOC = ::core::ffi::c_uint;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_Device {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_Program {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_Buffer {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_CircularBuffer {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_Kernel {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_CommandQueue {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_Event {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_Trace {
    _unused: [u8; 0],
}

pub const tt_Result_tt_Result_Success: tt_Result = 0;
pub const tt_Result_tt_Result_Error: tt_Result = 1;
pub type tt_Result = ::core::ffi::c_uint;

pub const tt_EventStatus_tt_EventStatus_Complete: tt_EventStatus = 0;
pub const tt_EventStatus_tt_EventStatus_Running: tt_EventStatus = 1;
pub const tt_EventStatus_tt_EventStatus_Submitted: tt_EventStatus = 2;
pub type tt_EventStatus = ::core::ffi::c_uint;

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct tt_InterleavedBufferConfig {
    pub device: *mut tt_Device,
    pub size: u32,
    pub page_size: u32,
    pub buffer_type: tt_BufferType,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct tt_DataMovementConfig {
    pub processor: tt_DataMovementProcessor,
    pub noc: tt_NOC,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct tt_ComputeConfig {
    pub compile_args: *mut u32,
    pub compile_args_count: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct tt_CircularBufferConfig {
    pub size: u32,
    pub data_format: tt_DataFormat,
    pub page_size: u32,
}

unsafe extern "C" {
    pub fn tt_metal_CreateDevice(device_id: ::core::ffi::c_int) -> *mut tt_Device;
}

unsafe extern "C" {
    pub fn tt_metal_CloseDevice(device: *mut tt_Device) -> ::core::ffi::c_int;
}

unsafe extern "C" {
    pub fn tt_metal_CreateProgram() -> *mut tt_Program;
}

unsafe extern "C" {
    pub fn tt_metal_CreateBuffer(config: *const tt_InterleavedBufferConfig) -> *mut tt_Buffer;
}

unsafe extern "C" {
    pub fn tt_metal_CreateCircularBuffer(
        program: *mut tt_Program,
        core: CoreCoord,
        config: *const tt_CircularBufferConfig,
    ) -> *mut tt_CircularBuffer;
}

unsafe extern "C" {
    pub fn tt_metal_CreateKernel(
        program: *mut tt_Program,
        kernel_file: *const ::core::ffi::c_char,
        core: CoreCoord,
        config: *const tt_DataMovementConfig,
    ) -> *mut tt_Kernel;
}

unsafe extern "C" {
    pub fn tt_metal_CreateComputeKernel(
        program: *mut tt_Program,
        kernel_file: *const ::core::ffi::c_char,
        core: CoreCoord,
        config: *const tt_ComputeConfig,
    ) -> *mut tt_Kernel;
}

unsafe extern "C" {
    pub fn tt_metal_SetRuntimeArgs(
        program: *mut tt_Program,
        kernel_name: *const ::core::ffi::c_char,
        args: *const *const tt_Buffer,
        num_args: i32,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_LaunchProgram(
        device: *mut tt_Device,
        program: *mut tt_Program,
    ) -> ::core::ffi::c_int;
}

unsafe extern "C" {
    pub fn tt_metal_TileSize(data_format: tt_DataFormat) -> u32;
}

unsafe extern "C" {
    pub fn tt_metal_WriteToBuffer(
        buffer: *mut tt_Buffer,
        data: *const core::ffi::c_void,
        size: u64,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_ReadFromBuffer(
        buffer: *mut tt_Buffer,
        data: *mut core::ffi::c_void,
        size: u64,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_buffer_address(buffer: *mut tt_Buffer) -> u64;
}

unsafe extern "C" {
    pub fn tt_metal_create_random_vector_of_bfp8(
        size: u32,
        is_exp_a: ::core::ffi::c_int,
        seed: u32,
        timestamp: u64,
    ) -> *mut u32;
}

unsafe extern "C" {
    pub fn tt_metal_CreateCommandQueue(device: *mut tt_Device) -> *mut tt_CommandQueue;
}

unsafe extern "C" {
    pub fn tt_metal_Synchronize(device: *mut tt_Device) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_BeginTraceCapture(
        device: *mut tt_Device,
        trace_name: *const c_char,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_EndTraceCapture(device: *mut tt_Device, trace: *mut *mut tt_Trace)
        -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_LoadTrace(
        device: *mut tt_Device,
        trace_file: *const c_char,
        trace: *mut *mut tt_Trace,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_ReplayTrace(device: *mut tt_Device, trace: *mut tt_Trace) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_LightMetalBeginCapture(
        device: *mut tt_Device,
        capture_name: *const c_char,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_LightMetalEndCapture(device: *mut tt_Device) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_DumpDeviceProfileResults(
        device: *mut tt_Device,
        output_file: *const c_char,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_LoadFromLLVM(program: *mut tt_Program, llvm_ir: *const c_char) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_WaitForCompletion(program: *mut tt_Program) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_DestroyProgram(program: *mut tt_Program);
}

unsafe extern "C" {
    pub fn tt_metal_WriteToBufferOffset(
        buffer: *mut tt_Buffer,
        data: *const u32,
        offset: u64,
        size: usize,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_ReadFromBufferOffset(
        buffer: *mut tt_Buffer,
        data: *mut u32,
        offset: u64,
        size: usize,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_GetBufferAddress(buffer: *mut tt_Buffer) -> u64;
}

unsafe extern "C" {
    pub fn tt_metal_DestroyBuffer(buffer: *mut tt_Buffer);
}

unsafe extern "C" {
    pub fn tt_metal_EnqueueWriteBuffer(
        queue: *mut tt_CommandQueue,
        buffer: *mut tt_Buffer,
        blocking: bool,
        offset: u64,
        size: u64,
        data: *const c_void,
        event: *mut *mut tt_Event,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_EnqueueReadBuffer(
        queue: *mut tt_CommandQueue,
        buffer: *mut tt_Buffer,
        blocking: bool,
        offset: u64,
        size: u64,
        data: *mut c_void,
        event: *mut *mut tt_Event,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_EnqueueProgram(
        queue: *mut tt_CommandQueue,
        program: *mut tt_Program,
        event: *mut *mut tt_Event,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_EnqueueRecordEvent(
        queue: *mut tt_CommandQueue,
        event: *mut *mut tt_Event,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_EnqueueWaitForEvent(
        queue: *mut tt_CommandQueue,
        event: *mut tt_Event,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_EnqueueTrace(
        queue: *mut tt_CommandQueue,
        trace: *mut tt_Trace,
        event: *mut *mut tt_Event,
    ) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_Finish(queue: *mut tt_CommandQueue) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_DestroyCommandQueue(queue: *mut tt_CommandQueue);
}

unsafe extern "C" {
    pub fn tt_metal_EventQuery(event: *mut tt_Event) -> tt_EventStatus;
}

unsafe extern "C" {
    pub fn tt_metal_EventSynchronize(event: *mut tt_Event) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_DestroyEvent(event: *mut tt_Event);
}

unsafe extern "C" {
    pub fn tt_metal_ReleaseTrace(trace: *mut tt_Trace);
}

unsafe extern "C" {
    pub fn tt_metal_QueryDevices(device_count: *mut u32, device_ids: *mut u32) -> tt_Result;
}

unsafe extern "C" {
    pub fn tt_metal_GetDeviceCount() -> c_int;
}

unsafe impl Send for tt_Device {}
unsafe impl Sync for tt_Device {}
unsafe impl Send for tt_Program {}
unsafe impl Sync for tt_Program {}
unsafe impl Send for tt_Buffer {}
unsafe impl Sync for tt_Buffer {}
unsafe impl Send for tt_CircularBuffer {}
unsafe impl Sync for tt_CircularBuffer {}
unsafe impl Send for tt_Kernel {}
unsafe impl Sync for tt_Kernel {}
unsafe impl Send for tt_CommandQueue {}
unsafe impl Sync for tt_CommandQueue {}
unsafe impl Send for tt_Event {}
unsafe impl Sync for tt_Event {}
unsafe impl Send for tt_Trace {}
unsafe impl Sync for tt_Trace {}

pub struct Device {
    handle: *mut tt_Device,
}

pub struct Program {
    handle: *mut tt_Program,
}

pub struct Buffer {
    handle: *mut tt_Buffer,
}

pub struct Kernel {
    handle: *mut tt_Kernel,
}

pub struct CommandQueue {
    handle: *mut tt_CommandQueue,
}

pub struct Event {
    handle: *mut tt_Event,
}

pub struct Trace {
    handle: *mut tt_Trace,
}

impl Device {
    pub fn new(device_id: u32) -> Result<Self, String> {
        let handle = unsafe { tt_metal_CreateDevice(device_id as c_int) };
        if handle.is_null() {
            return Err(format!("Failed to create device with id {}", device_id));
        }
        Ok(Self { handle })
    }

    pub fn get_name(&self) -> Result<String, String> {
        // Tenstorrent API doesn't provide a direct method to get device name
        // Return a generic name based on the handle pointer value
        Ok(format!("Tenstorrent Device {:p}", self.handle))
    }

    pub fn create_program(&self) -> Result<Program, String> {
        let handle = unsafe { tt_metal_CreateProgram() };
        if handle.is_null() {
            Err("Failed to create program".to_string())
        } else {
            Ok(Program { handle })
        }
    }

    pub fn create_buffer(&self, size: u64) -> Result<Buffer, String> {
        let config = tt_InterleavedBufferConfig {
            device: self.handle,
            size: size as u32,
            page_size: 1024, // Default page size
            buffer_type: 0,  // tt_BufferType_tt_BufferType_DRAM
        };
        let handle = unsafe { tt_metal_CreateBuffer(&config) };
        if handle.is_null() {
            Err("Failed to create buffer".to_string())
        } else {
            Ok(Buffer { handle })
        }
    }

    pub fn create_command_queue(&self) -> Result<CommandQueue, String> {
        let handle = unsafe { tt_metal_CreateCommandQueue(self.handle) };
        if handle.is_null() {
            Err("Failed to create command queue".to_string())
        } else {
            Ok(CommandQueue { handle })
        }
    }

    pub fn synchronize(&self) -> Result<(), String> {
        let result = unsafe { tt_metal_Synchronize(self.handle) };
        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to synchronize device: error code {:?}",
                result
            ))
        }
    }

    pub fn begin_trace_capture(&self, trace_name: &str) -> Result<(), String> {
        let trace_name = CString::new(trace_name).map_err(|e| e.to_string())?;
        let result = unsafe { tt_metal_BeginTraceCapture(self.handle, trace_name.as_ptr()) };
        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to begin trace capture: error code {:?}",
                result
            ))
        }
    }

    pub fn end_trace_capture(&self) -> Result<Trace, String> {
        let mut trace_handle = ptr::null_mut();
        let result = unsafe { tt_metal_EndTraceCapture(self.handle, &mut trace_handle) };
        if result == tt_Result_tt_Result_Success {
            Ok(Trace {
                handle: trace_handle,
            })
        } else {
            Err(format!(
                "Failed to end trace capture: error code {:?}",
                result
            ))
        }
    }

    pub fn load_trace(&self, trace_file: &str) -> Result<Trace, String> {
        let trace_file = CString::new(trace_file).map_err(|e| e.to_string())?;
        let mut trace_handle = ptr::null_mut();
        let result =
            unsafe { tt_metal_LoadTrace(self.handle, trace_file.as_ptr(), &mut trace_handle) };
        if result == tt_Result_tt_Result_Success {
            Ok(Trace {
                handle: trace_handle,
            })
        } else {
            Err(format!("Failed to load trace: error code {:?}", result))
        }
    }

    pub fn replay_trace(&self, trace: &Trace) -> Result<(), String> {
        let result = unsafe { tt_metal_ReplayTrace(self.handle, trace.handle) };
        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!("Failed to replay trace: error code {:?}", result))
        }
    }

    pub fn light_metal_begin_capture(&self, capture_name: &str) -> Result<(), String> {
        let capture_name = CString::new(capture_name).map_err(|e| e.to_string())?;
        let result = unsafe { tt_metal_LightMetalBeginCapture(self.handle, capture_name.as_ptr()) };
        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to begin LightMetal capture: error code {:?}",
                result
            ))
        }
    }

    pub fn light_metal_end_capture(&self) -> Result<(), String> {
        let result = unsafe { tt_metal_LightMetalEndCapture(self.handle) };
        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to end LightMetal capture: error code {:?}",
                result
            ))
        }
    }

    pub fn dump_profile_results(&self, output_file: &str) -> Result<(), String> {
        let output_file = CString::new(output_file).map_err(|e| e.to_string())?;
        let result =
            unsafe { tt_metal_DumpDeviceProfileResults(self.handle, output_file.as_ptr()) };
        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to dump profile results: error code {:?}",
                result
            ))
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            tt_metal_CloseDevice(self.handle);
        }
    }
}

impl Program {

    pub fn create_kernel(&self, kernel_name: &str, core: CoreCoord) -> Result<Kernel, String> {
        eprintln!("ZLUDA DEBUG: Program::create_kernel called with kernel_name={}, core=({},{})", kernel_name, core.x, core.y);
        
        let kernel_name = CString::new(kernel_name).map_err(|e| e.to_string())?;
        let config = tt_DataMovementConfig {
            processor: 0, // tt_DataMovementProcessor_tt_DataMovementProcessor_RISCV_0
            noc: 0,       // tt_NOC_tt_NOC_RISCV_0_default
        };
        
        eprintln!("ZLUDA DEBUG: About to call tt_metal_CreateKernel with program={:p}, kernel_name={:?}, core=({},{}), config=({},{})", 
                  self.handle, kernel_name, core.x, core.y, config.processor, config.noc);
        
        let handle =
            unsafe { tt_metal_CreateKernel(self.handle, kernel_name.as_ptr(), core, &config) };
            
        eprintln!("ZLUDA DEBUG: tt_metal_CreateKernel returned handle={:p}", handle);
        
        if handle.is_null() {
            Err("Failed to create kernel".to_string())
        } else {
            Ok(Kernel { handle })
        }
    }

    pub fn set_runtime_args(
        &self,
        kernel_name: &str,
        buffers: &[&Buffer],
    ) -> Result<(), String> {
        let kernel_name_cstr = std::ffi::CString::new(kernel_name)
            .map_err(|e| format!("Invalid kernel name: {}", e))?;
        
        // Convert buffer references to buffer pointers
        let buffer_ptrs: Vec<*const tt_Buffer> = buffers.iter().map(|b| b.handle as *const tt_Buffer).collect();
        
        let result = unsafe {
            tt_metal_SetRuntimeArgs(
                self.handle,
                kernel_name_cstr.as_ptr(),
                buffer_ptrs.as_ptr(),
                buffer_ptrs.len() as i32,
            )
        };
        
        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!("Failed to set runtime args: error code {:?}", result))
        }
    }

    pub fn launch(&self, device: &Device) -> Result<(), String> {
        let result = unsafe { tt_metal_LaunchProgram(device.handle, self.handle) };
        if result == 0 {
            Ok(())
        } else {
            Err(format!("Failed to launch program: error code {:?}", result))
        }
    }

    pub fn wait_for_completion(&self) -> Result<(), String> {
        let result = unsafe { tt_metal_WaitForCompletion(self.handle) };
        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to wait for program completion: error code {:?}",
                result
            ))
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe {
            tt_metal_DestroyProgram(self.handle);
        }
    }
}

impl Buffer {
    pub fn write(&self, data: &[u8]) -> Result<(), String> {
        let result =
            unsafe { tt_metal_WriteToBuffer(self.handle, data.as_ptr() as *const core::ffi::c_void, data.len() as u64) };

        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to write to buffer: error code {:?}",
                result
            ))
        }
    }

    pub fn read(&self, data: &mut [u8]) -> Result<(), String> {
        let result = unsafe {
            tt_metal_ReadFromBuffer(self.handle, data.as_mut_ptr() as *mut core::ffi::c_void, data.len() as u64)
        };

        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to read from buffer: error code {:?}",
                result
            ))
        }
    }

    pub fn write_offset(&self, data: &[u8], offset: u64) -> Result<(), String> {
        let result = unsafe {
            tt_metal_WriteToBufferOffset(
                self.handle,
                data.as_ptr() as *const u32,
                offset,
                data.len(),
            )
        };

        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to write to buffer at offset: error code {:?}",
                result
            ))
        }
    }

    pub fn read_offset(&self, data: &mut [u8], offset: u64) -> Result<(), String> {
        let result = unsafe {
            tt_metal_ReadFromBufferOffset(
                self.handle,
                data.as_mut_ptr() as *mut u32,
                offset,
                data.len(),
            )
        };

        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to read from buffer at offset: error code {:?}",
                result
            ))
        }
    }

    pub fn get_address(&self) -> u64 {
        // Since tt_metal_GetBufferAddress is not available, we're using a dummy implementation
        // that returns a constant value instead of making the FFI call
        unsafe { tt_metal_GetBufferAddress(self.handle) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            tt_metal_DestroyBuffer(self.handle);
        }
    }
}

impl CommandQueue {
    pub fn enqueue_write_buffer(
        &self,
        buffer: &Buffer,
        blocking: bool,
        offset: u64,
        data: &[u8],
    ) -> Result<Event, String> {
        let mut event_handle = ptr::null_mut();
        let result = unsafe {
            tt_metal_EnqueueWriteBuffer(
                self.handle,
                buffer.handle,
                blocking,
                offset,
                data.len() as u64,
                data.as_ptr() as *const c_void,
                &mut event_handle,
            )
        };

        if result == tt_Result_tt_Result_Success {
            Ok(Event {
                handle: event_handle,
            })
        } else {
            Err(format!(
                "Failed to enqueue write buffer: error code {:?}",
                result
            ))
        }
    }

    pub fn enqueue_read_buffer(
        &self,
        buffer: &Buffer,
        blocking: bool,
        offset: u64,
        data: &mut [u8],
    ) -> Result<Event, String> {
        let mut event_handle = ptr::null_mut();
        let result = unsafe {
            tt_metal_EnqueueReadBuffer(
                self.handle,
                buffer.handle,
                blocking,
                offset,
                data.len() as u64,
                data.as_mut_ptr() as *mut c_void,
                &mut event_handle,
            )
        };

        if result == tt_Result_tt_Result_Success {
            Ok(Event {
                handle: event_handle,
            })
        } else {
            Err(format!(
                "Failed to enqueue read buffer: error code {:?}",
                result
            ))
        }
    }

    pub fn enqueue_program(&self, program: &Program) -> Result<Event, String> {
        let mut event_handle = ptr::null_mut();
        let result =
            unsafe { tt_metal_EnqueueProgram(self.handle, program.handle, &mut event_handle) };

        if result == tt_Result_tt_Result_Success {
            Ok(Event {
                handle: event_handle,
            })
        } else {
            Err(format!(
                "Failed to enqueue program: error code {:?}",
                result
            ))
        }
    }

    pub fn enqueue_record_event(&self) -> Result<Event, String> {
        let mut event_handle = ptr::null_mut();
        let result = unsafe { tt_metal_EnqueueRecordEvent(self.handle, &mut event_handle) };

        if result == tt_Result_tt_Result_Success {
            Ok(Event {
                handle: event_handle,
            })
        } else {
            Err(format!(
                "Failed to enqueue record event: error code {:?}",
                result
            ))
        }
    }

    pub fn enqueue_wait_for_event(&self, event: &Event) -> Result<(), String> {
        let result = unsafe { tt_metal_EnqueueWaitForEvent(self.handle, event.handle) };

        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to enqueue wait for event: error code {:?}",
                result
            ))
        }
    }

    pub fn enqueue_trace(&self, trace: &Trace) -> Result<Event, String> {
        let mut event_handle = ptr::null_mut();
        let result = unsafe { tt_metal_EnqueueTrace(self.handle, trace.handle, &mut event_handle) };

        if result == tt_Result_tt_Result_Success {
            Ok(Event {
                handle: event_handle,
            })
        } else {
            Err(format!("Failed to enqueue trace: error code {:?}", result))
        }
    }

    pub fn finish(&self) -> Result<(), String> {
        let result = unsafe { tt_metal_Finish(self.handle) };
        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to finish command queue: error code {:?}",
                result
            ))
        }
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        unsafe {
            tt_metal_DestroyCommandQueue(self.handle);
        }
    }
}

impl Event {
    pub fn query(&self) -> tt_EventStatus {
        unsafe { tt_metal_EventQuery(self.handle) }
    }

    pub fn synchronize(&self) -> Result<(), String> {
        let result = unsafe { tt_metal_EventSynchronize(self.handle) };
        if result == tt_Result_tt_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to synchronize event: error code {:?}",
                result
            ))
        }
    }

    pub fn is_complete(&self) -> bool {
        self.query() == tt_EventStatus_tt_EventStatus_Complete
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            tt_metal_DestroyEvent(self.handle);
        }
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        unsafe {
            tt_metal_ReleaseTrace(self.handle);
        }
    }
}

pub fn query_devices() -> Result<Vec<u32>, String> {
    let mut device_count = 0;
    let result = unsafe { tt_metal_QueryDevices(&mut device_count, ptr::null_mut()) };

    if result != tt_Result_tt_Result_Success {
        return Err(format!(
            "Failed to query device count: error code {:?}",
            result
        ));
    }

    let mut device_ids = vec![0; device_count as usize];
    let result = unsafe { tt_metal_QueryDevices(&mut device_count, device_ids.as_mut_ptr()) };

    if result == tt_Result_tt_Result_Success {
        Ok(device_ids)
    } else {
        Err(format!(
            "Failed to query device IDs: error code {:?}",
            result
        ))
    }
}

pub fn tile_size(data_format: tt_DataFormat) -> u32 {
    unsafe { tt_metal_TileSize(data_format) }
}

pub fn get_device_count() -> Result<i32, String> {
    let count = unsafe { tt_metal_GetDeviceCount() };
    if count < 0 {
        Err("Failed to get device count".to_string())
    } else {
        Ok(count)
    }
}
