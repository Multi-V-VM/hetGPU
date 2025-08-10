// Gemmini Runtime System using Spike RISC-V ISA Simulator
#![allow(warnings)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::ptr;
use std::sync::Mutex;
use std::time::SystemTime;

// Tensor dimensions - must match those in ptx/src/pass/emit_tosa_mlir.rs
const TENSOR_BATCH_DIM_X: i64 = 32;
const TENSOR_BATCH_DIM_Y: i64 = 32;

// Gemmini configuration constants
pub const GEMMINI_DIM: usize = 16;
pub const GEMMINI_SPAD_ROWS: usize = 256;
pub const GEMMINI_ACC_ROWS: usize = 64;
pub const GEMMINI_BLOCK_SIZE: usize = GEMMINI_DIM;

// Core coordinate for Gemmini (single core in Spike)
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct CoreCoord {
    pub x: u32,
    pub y: u32,
}

// Data format types for Gemmini
pub const gemmini_DataFormat_Invalid: gemmini_DataFormat = 0;
pub const gemmini_DataFormat_Int8: gemmini_DataFormat = 1;
pub const gemmini_DataFormat_Int16: gemmini_DataFormat = 2;
pub const gemmini_DataFormat_Int32: gemmini_DataFormat = 3;
pub const gemmini_DataFormat_Float16: gemmini_DataFormat = 4;
pub const gemmini_DataFormat_Float32: gemmini_DataFormat = 5;
pub const gemmini_DataFormat_Bfloat16: gemmini_DataFormat = 6;
pub type gemmini_DataFormat = ::core::ffi::c_uint;

// Buffer types
pub const gemmini_BufferType_SPAD: gemmini_BufferType = 0;
pub const gemmini_BufferType_ACC: gemmini_BufferType = 1;
pub const gemmini_BufferType_DRAM: gemmini_BufferType = 2;
pub type gemmini_BufferType = ::core::ffi::c_uint;

// Result types
pub const gemmini_Result_Success: gemmini_Result = 0;
pub const gemmini_Result_Error: gemmini_Result = 1;
pub type gemmini_Result = ::core::ffi::c_uint;

// Opaque types for handles
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gemmini_Device {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gemmini_Program {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gemmini_Buffer {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gemmini_Kernel {
    _unused: [u8; 0],
}

// Gemmini configuration structures
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct gemmini_BufferConfig {
    pub device: *mut gemmini_Device,
    pub size: u64,
    pub buffer_type: gemmini_BufferType,
    pub data_format: gemmini_DataFormat,
}

// Internal state management
static SPIKE_STATE: Mutex<Option<SpikeState>> = Mutex::new(None);

struct SpikeState {
    temp_dir: PathBuf,
    kernel_name: String,
    programs: Vec<ProgramData>,
    buffers: Vec<BufferData>,
}

struct ProgramData {
    id: usize,
    kernel_name: Option<String>,
}

struct BufferData {
    id: usize,
    size: u64,
    buffer_type: gemmini_BufferType,
    data: Vec<u8>,
}

// Helper functions for validation
fn validate_program_id(spike_state: &SpikeState, program_id: usize) -> bool {
    program_id < spike_state.programs.len()
}

fn validate_buffer_id(spike_state: &SpikeState, buffer_id: usize) -> bool {
    buffer_id < spike_state.buffers.len()
}

// FFI functions implementation

pub unsafe extern "C" fn gemmini_CreateDevice(
    device_id: ::core::ffi::c_int,
) -> *mut gemmini_Device {
    let mut state = SPIKE_STATE.lock().unwrap();

    if state.is_none() {
        // Initially use a pending directory - will be updated when kernel is created
        let temp_dir = PathBuf::from("/tmp/gemmini_pending");
        *state = Some(SpikeState {
            temp_dir,
            kernel_name: String::new(),
            programs: Vec::new(),
            buffers: Vec::new(),
        });
        eprintln!("Gemmini/Spike: Initialized device {}", device_id);
    }

    // Return a dummy pointer (we only support one device in Spike)
    1 as *mut gemmini_Device
}

pub unsafe extern "C" fn gemmini_CloseDevice(device: *mut gemmini_Device) -> ::core::ffi::c_int {
    if device.is_null() {
        return gemmini_Result_Error as c_int;
    }

    let mut state = SPIKE_STATE.lock().unwrap();
    if state.is_some() {
        *state = None;
    }

    gemmini_Result_Success as c_int
}

pub unsafe extern "C" fn gemmini_CreateProgram() -> *mut gemmini_Program {
    let mut state = SPIKE_STATE.lock().unwrap();

    if let Some(ref mut spike_state) = *state {
        let program_id = spike_state.programs.len();
        spike_state.programs.push(ProgramData { 
            id: program_id,
            kernel_name: None,
        });

        return (program_id + 1) as *mut gemmini_Program;
    }

    ptr::null_mut()
}

pub unsafe extern "C" fn gemmini_CreateBuffer(
    config: *const gemmini_BufferConfig,
) -> *mut gemmini_Buffer {
    if config.is_null() {
        return ptr::null_mut();
    }

    let config = &*config;
    let mut state = SPIKE_STATE.lock().unwrap();

    if let Some(ref mut spike_state) = *state {
        let buffer_id = spike_state.buffers.len();
        spike_state.buffers.push(BufferData {
            id: buffer_id,
            size: config.size,
            buffer_type: config.buffer_type,
            data: vec![0u8; config.size as usize],
        });

        return (buffer_id + 1) as *mut gemmini_Buffer;
    }

    ptr::null_mut()
}

pub unsafe extern "C" fn gemmini_CreateKernel(
    program: *mut gemmini_Program,
    kernel_file: *const ::core::ffi::c_char,
    core: CoreCoord,
    _config: *const c_void,
) -> *mut gemmini_Kernel {
    if program.is_null() || kernel_file.is_null() {
        return ptr::null_mut();
    }

    let kernel_name = CStr::from_ptr(kernel_file).to_string_lossy().to_string();
    let program_id = (program as usize) - 1;

    let mut state = SPIKE_STATE.lock().unwrap();

    if let Some(ref mut spike_state) = *state {
        if !validate_program_id(spike_state, program_id) {
            return ptr::null_mut();
        }

        // Store kernel name for later use when creating directory
        spike_state.kernel_name = kernel_name.clone();
        if program_id < spike_state.programs.len() {
            spike_state.programs[program_id].kernel_name = Some(kernel_name.clone());
        }
        
        eprintln!("Gemmini/Spike: Stored kernel name: {}", kernel_name.clone());

        // Kernels are not actually tracked in this implementation
        return 1 as *mut gemmini_Kernel;
    }

    ptr::null_mut()
}

pub unsafe extern "C" fn gemmini_SetRuntimeArgs(
    program: *mut gemmini_Program,
    kernel_name: *const ::core::ffi::c_char,
    args: *const *const gemmini_Buffer,
    num_args: i32,
) -> gemmini_Result {
    if program.is_null() || kernel_name.is_null() || args.is_null() {
        return gemmini_Result_Error;
    }

    gemmini_Result_Success
}

pub unsafe extern "C" fn gemmini_LaunchProgram(
    device: *mut gemmini_Device,
    program: *mut gemmini_Program,
) -> ::core::ffi::c_int {
    if device.is_null() || program.is_null() {
        return gemmini_Result_Error as c_int;
    }

    let program_id = (program as usize) - 1;
    let mut state = SPIKE_STATE.lock().unwrap();

    if let Some(ref mut spike_state) = *state {
        if !validate_program_id(spike_state, program_id) {
            return gemmini_Result_Error as c_int;
        }

        // Generate and run Gemmini code on Spike
        if let Err(e) = run_on_spike(spike_state, program_id) {
            eprintln!("Gemmini/Spike: Failed to run on Spike: {}", e);
            return gemmini_Result_Error as c_int;
        }

        return gemmini_Result_Success as c_int;
    }

    gemmini_Result_Error as c_int
}

pub unsafe extern "C" fn gemmini_WriteToBuffer(
    buffer: *mut gemmini_Buffer,
    data: *const core::ffi::c_void,
    size: u64,
) -> gemmini_Result {
    if buffer.is_null() || data.is_null() {
        return gemmini_Result_Error;
    }

    let buffer_id = (buffer as usize) - 1;
    let mut state = SPIKE_STATE.lock().unwrap();

    if let Some(ref mut spike_state) = *state {
        if buffer_id >= spike_state.buffers.len() {
            return gemmini_Result_Error;
        }

        let buffer_data = &mut spike_state.buffers[buffer_id];
        let copy_size = std::cmp::min(size as usize, buffer_data.data.len());

        let src = std::slice::from_raw_parts(data as *const u8, copy_size);
        buffer_data.data[..copy_size].copy_from_slice(src);

        return gemmini_Result_Success;
    }

    gemmini_Result_Error
}

pub unsafe extern "C" fn gemmini_ReadFromBuffer(
    buffer: *mut gemmini_Buffer,
    data: *mut core::ffi::c_void,
    size: u64,
) -> gemmini_Result {
    if buffer.is_null() || data.is_null() {
        return gemmini_Result_Error;
    }

    let buffer_id = (buffer as usize) - 1;
    let mut state = SPIKE_STATE.lock().unwrap();

    if let Some(ref mut spike_state) = *state {
        if buffer_id >= spike_state.buffers.len() {
            return gemmini_Result_Error;
        }

        let buffer_data = &spike_state.buffers[buffer_id];
        let copy_size = std::cmp::min(size as usize, buffer_data.data.len());

        let dst = std::slice::from_raw_parts_mut(data as *mut u8, copy_size);
        dst.copy_from_slice(&buffer_data.data[..copy_size]);

        return gemmini_Result_Success;
    }

    gemmini_Result_Error
}

pub unsafe extern "C" fn gemmini_LoadFromMLIR(
    program: *mut gemmini_Program,
    mlir: *const c_char,
) -> gemmini_Result {
    if program.is_null() || mlir.is_null() {
        return gemmini_Result_Error;
    }

    let program_id = (program as usize) - 1;
    let mlir_str = CStr::from_ptr(mlir).to_string_lossy().to_string();

    let mut state = SPIKE_STATE.lock().unwrap();

    if let Some(ref mut spike_state) = *state {
        if program_id >= spike_state.programs.len() {
            eprintln!("Gemmini/Spike: Invalid program ID");
            return gemmini_Result_Error;
        }

        // Create unique directory if we're still using the pending directory
        if spike_state.temp_dir == PathBuf::from("/tmp/gemmini_pending") {
            let kernel_suffix = if !spike_state.kernel_name.is_empty() {
                &spike_state.kernel_name
            } else {
                "unknown"
            };
            
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_micros();
            let unique_dir = format!("/tmp/gemmini_{}_{}", kernel_suffix, timestamp);
            
            spike_state.temp_dir = PathBuf::from(unique_dir);
            eprintln!("Gemmini/Spike: Created unique directory: {}", spike_state.temp_dir.display());
        }
        
        // Write MLIR to file in temp directory
        let temp_dir = &spike_state.temp_dir;
        
        // Ensure the directory exists
        if !temp_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(&temp_dir) {
                eprintln!("Gemmini/Spike: Failed to create directory: {}", e);
                return gemmini_Result_Error;
            }
        }
        
        let mlir_file = temp_dir.join(format!("gemmini_mlir_{}.mlir", program_id));
        if let Err(e) = fs::write(&mlir_file, &mlir_str) {
            eprintln!("Gemmini/Spike: Failed to write MLIR file: {}", e);
            return gemmini_Result_Error;
        }

        // Convert MLIR to executable using Buddy compiler toolchain
        match convert_mlir_to_executable(mlir_file.to_str().unwrap(), spike_state, program_id) {
            Ok(elf_path) => {
                return gemmini_Result_Success;
            }
            Err(e) => {
                eprintln!("Gemmini/Spike: Failed to compile MLIR: {}", e);
                return gemmini_Result_Error;
            }
        }
    }

    gemmini_Result_Error
}

pub unsafe extern "C" fn gemmini_WaitForCompletion(
    program: *mut gemmini_Program,
) -> gemmini_Result {
    if program.is_null() {
        return gemmini_Result_Error;
    }

    gemmini_Result_Success
}

pub unsafe extern "C" fn gemmini_DestroyProgram(program: *mut gemmini_Program) {
    if program.is_null() {
        return;
    }

    let program_id = (program as usize) - 1;
}

pub unsafe extern "C" fn gemmini_DestroyBuffer(buffer: *mut gemmini_Buffer) {
    if buffer.is_null() {
        return;
    }

    let buffer_id = (buffer as usize) - 1;
}

// Safe wrapper types
unsafe impl Send for gemmini_Device {}
unsafe impl Sync for gemmini_Device {}
unsafe impl Send for gemmini_Program {}
unsafe impl Sync for gemmini_Program {}
unsafe impl Send for gemmini_Buffer {}
unsafe impl Sync for gemmini_Buffer {}
unsafe impl Send for gemmini_Kernel {}
unsafe impl Sync for gemmini_Kernel {}

// High-level Rust API
pub struct Device {
    handle: *mut gemmini_Device,
}

pub struct Program {
    handle: *mut gemmini_Program,
}

pub struct Buffer {
    handle: *mut gemmini_Buffer,
}

pub struct Kernel {
    handle: *mut gemmini_Kernel,
}

impl Device {
    pub fn new(device_id: u32) -> Result<Self, String> {
        let handle = unsafe { gemmini_CreateDevice(device_id as c_int) };
        if handle.is_null() {
            return Err(format!("Failed to create Gemmini device {}", device_id));
        }
        Ok(Self { handle })
    }

    pub fn get_name(&self) -> Result<String, String> {
        Ok("Gemmini Accelerator (Spike Simulator)".to_string())
    }

    pub fn create_program(&self) -> Result<Program, String> {
        let handle = unsafe { gemmini_CreateProgram() };
        if handle.is_null() {
            Err("Failed to create program".to_string())
        } else {
            Ok(Program { handle })
        }
    }

    pub fn create_buffer(&self, size: u64) -> Result<Buffer, String> {
        let config = gemmini_BufferConfig {
            device: self.handle,
            size,
            buffer_type: gemmini_BufferType_DRAM,
            data_format: gemmini_DataFormat_Int8,
        };
        let handle = unsafe { gemmini_CreateBuffer(&config) };
        if handle.is_null() {
            Err("Failed to create buffer".to_string())
        } else {
            Ok(Buffer { handle })
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            gemmini_CloseDevice(self.handle);
        }
    }
}

impl Program {
    pub fn load_from_mlir(&self, mlir: &str) -> Result<(), String> {
        let mlir_cstr = CString::new(mlir).map_err(|e| e.to_string())?;
        let result = unsafe { gemmini_LoadFromMLIR(self.handle, mlir_cstr.as_ptr()) };
        if result == gemmini_Result_Success {
            Ok(())
        } else {
            Err(format!("Failed to load MLIR: error code {:?}", result))
        }
    }

    pub fn create_kernel(&self, kernel_name: &str, core: CoreCoord) -> Result<Kernel, String> {
        let kernel_name = CString::new(kernel_name).map_err(|e| e.to_string())?;
        let handle =
            unsafe { gemmini_CreateKernel(self.handle, kernel_name.as_ptr(), core, ptr::null()) };
        if handle.is_null() {
            Err("Failed to create kernel".to_string())
        } else {
            Ok(Kernel { handle })
        }
    }

    pub fn set_runtime_args(&self, kernel_name: &str, buffers: &[&Buffer]) -> Result<(), String> {
        let kernel_name_cstr =
            CString::new(kernel_name).map_err(|e| format!("Invalid kernel name: {}", e))?;

        let buffer_ptrs: Vec<*const gemmini_Buffer> = buffers
            .iter()
            .map(|b| b.handle as *const gemmini_Buffer)
            .collect();

        let result = unsafe {
            gemmini_SetRuntimeArgs(
                self.handle,
                kernel_name_cstr.as_ptr(),
                buffer_ptrs.as_ptr(),
                buffer_ptrs.len() as i32,
            )
        };

        if result == gemmini_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to set runtime args: error code {:?}",
                result
            ))
        }
    }

    pub fn launch(&self, device: &Device) -> Result<(), String> {
        let result = unsafe { gemmini_LaunchProgram(device.handle, self.handle) };
        if result == 0 {
            Ok(())
        } else {
            Err(format!("Failed to launch program: error code {:?}", result))
        }
    }

    pub fn wait_for_completion(&self) -> Result<(), String> {
        let result = unsafe { gemmini_WaitForCompletion(self.handle) };
        if result == gemmini_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to wait for completion: error code {:?}",
                result
            ))
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe {
            gemmini_DestroyProgram(self.handle);
        }
    }
}

impl Buffer {
    pub fn write(&self, data: &[u8]) -> Result<(), String> {
        let result = unsafe {
            gemmini_WriteToBuffer(
                self.handle,
                data.as_ptr() as *const core::ffi::c_void,
                data.len() as u64,
            )
        };

        if result == gemmini_Result_Success {
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
            gemmini_ReadFromBuffer(
                self.handle,
                data.as_mut_ptr() as *mut core::ffi::c_void,
                data.len() as u64,
            )
        };

        if result == gemmini_Result_Success {
            Ok(())
        } else {
            Err(format!(
                "Failed to read from buffer: error code {:?}",
                result
            ))
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            gemmini_DestroyBuffer(self.handle);
        }
    }
}

// Internal helper functions

fn run_on_spike(spike_state: &mut SpikeState, program_id: usize) -> Result<(), String> {
    // Look for the compiled object file
    let temp_dir = &spike_state.temp_dir;
    let obj_file = temp_dir.join(format!("gemmini_program_{}.o", program_id));

    if !obj_file.exists() {
        panic!(
            "Gemmini/Spike: Object file not found: {}",
            obj_file.display()
        );
    }

    // Check if the object file is valid
    let obj_data =
        std::fs::read(&obj_file).map_err(|e| format!("Failed to read object file: {}", e))?;

    if obj_data.len() <= 16 || !obj_data.starts_with(&[0x7f, 0x45, 0x4c, 0x46]) {
        panic!("Gemmini/Spike: Object file is not valid ELF format");
    }

    // Create executable and run with Spike
    let executable = create_executable_from_object(spike_state, obj_file.to_str().unwrap())?;

    // Execute with Spike
    let spike_cmd = format!(
        "LD_LIBRARY_PATH=/repo/riscv-gnu-toolchain/lib spike --extension=gemmini pk {}",
        executable.display()
    );
    eprintln!("Gemmini/Spike: Executing command: {}", spike_cmd);

    let spike_result = Command::new("sh").args(&["-c", &spike_cmd]).output();

    match spike_result {
        Ok(output) => {
            if output.status.success() || output.status.code() == Some(0) {
                read_spike_output_from_memory(spike_state, &output)
            } else {
                let exit_code = output.status.code().unwrap_or(-1);
                eprintln!(
                    "Gemmini/Spike: Execution failed with exit code: {}",
                    exit_code
                );
                eprintln!(
                    "Gemmini/Spike: Full stdout:\n{}",
                    String::from_utf8_lossy(&output.stdout)
                );
                eprintln!(
                    "Gemmini/Spike: Full stderr:\n{}",
                    String::from_utf8_lossy(&output.stderr)
                );

                panic!(
                    "Gemmini/Spike: Execution failed with status: {:?}",
                    output.status
                );
            }
        }
        Err(e) => {
            panic!("Gemmini/Spike: Failed to execute Spike: {}", e);
        }
    }
}

fn create_input_data_assembly(
    path: &std::path::Path,
    spike_state: &SpikeState,
) -> Result<(), String> {
    let mut content = String::new();

    // Create assembly file with input data embedded in data section
    content.push_str(".section .data\n");
    content.push_str(".align 8\n");
    content.push_str(".global __input_data_start\n");
    content.push_str(".global __input_data_end\n");
    content.push_str("__input_data_start:\n");

    // Embed buffer data as bytes
    if !spike_state.buffers.is_empty() {
        // Get all input buffer data (typically first two buffers for binary operations)
        let mut all_data = Vec::new();

        // Add first buffer (input1)
        all_data.extend_from_slice(&spike_state.buffers[0].data);

        // Add second buffer (input2) if it exists
        if spike_state.buffers.len() > 1 {
            all_data.extend_from_slice(&spike_state.buffers[1].data);
        }

        // Write data as .byte directives (8 bytes per line for readability)
        for chunk in all_data.chunks(8) {
            content.push_str("    .byte ");
            for (i, byte) in chunk.iter().enumerate() {
                if i > 0 {
                    content.push_str(", ");
                }
                content.push_str(&format!("0x{:02x}", byte));
            }
            content.push('\n');
        }
    }

    content.push_str("__input_data_end:\n");
    content.push_str("\n");

    // Also export the data size
    content.push_str(".section .rodata\n");
    content.push_str(".align 8\n");
    content.push_str(".global __input_data_size\n");
    content.push_str("__input_data_size:\n");

    let total_size = if spike_state.buffers.is_empty() {
        0
    } else {
        spike_state
            .buffers
            .iter()
            .take(2)
            .map(|b| b.data.len())
            .sum::<usize>()
    };
    content.push_str(&format!("    .quad {}\n", total_size));

    std::fs::write(path, content)
        .map_err(|e| format!("Failed to write input data assembly: {}", e))?;

    Ok(())
}

fn create_linker_script(temp_dir: &std::path::Path) -> Result<String, String> {
    let linker_script = temp_dir.join("gemmini.ld");

    // Read the linker script from the external file
    let script_content = include_str!("gemmini.ld");

    std::fs::write(&linker_script, script_content)
        .map_err(|e| format!("Failed to create linker script: {}", e))?;

    Ok(linker_script.to_string_lossy().to_string())
}

fn create_executable_from_object(
    spike_state: &mut SpikeState,
    obj_file: &str,
) -> Result<std::path::PathBuf, String> {
    let temp_dir = &spike_state.temp_dir;
    let executable = temp_dir.join("gemmini_kernel");

    // Create input data assembly file that embeds the buffer data
    let input_data_s = temp_dir.join("input_data.s");
    create_input_data_assembly(&input_data_s, spike_state)?;

    // Create a simple startup code that calls the kernel function
    let startup_c = temp_dir.join("startup.c");

    // Read the startup C code from the external file
    let startup_content = include_str!("startup.c");

    std::fs::write(&startup_c, startup_content)
        .map_err(|e| format!("Failed to write startup code: {}", e))?;

    // Create constants file with proper section placement
    let constants_s = temp_dir.join("constants.s");

    // Read the constants assembly from the external file
    let constants_content = include_str!("mlir_consts.s");
    std::fs::write(&constants_s, constants_content)
        .map_err(|e| format!("Failed to write constants assembly: {}", e))?;

    // Compile all assembly and C files to object files first
    let startup_o = temp_dir.join("startup.o");
    let input_data_o = temp_dir.join("input_data.o");
    let constants_o = temp_dir.join("constants.o");

    // Compile startup.c with matrix dimensions from TOSA configuration
    let compile_c_cmd = format!(
        "riscv64-unknown-elf-gcc -c -march=rv64gc -mabi=lp64d -mcmodel=medany \
         -DMATRIX_DIM_X={} -DMATRIX_DIM_Y={} {} -o {}",
        TENSOR_BATCH_DIM_X, TENSOR_BATCH_DIM_Y,
        startup_c.display(),
        startup_o.display()
    );
    eprintln!("Gemmini/Spike: Executing command: {}", compile_c_cmd);
    let compile_c_result = Command::new("sh").args(&["-c", &compile_c_cmd]).output();

    match compile_c_result {
        Ok(output) => {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("Failed to compile startup.c: {}", stderr));
            }
        }
        Err(e) => return Err(format!("Failed to run gcc for startup.c: {}", e)),
    }

    // Assemble input_data.s
    let compile_asm_cmd = format!(
        "riscv64-unknown-elf-gcc -c -march=rv64gc -mabi=lp64d {} -o {}",
        input_data_s.display(),
        input_data_o.display()
    );
    eprintln!("Gemmini/Spike: Executing command: {}", compile_asm_cmd);
    let compile_asm_result = Command::new("sh").args(&["-c", &compile_asm_cmd]).output();

    match compile_asm_result {
        Ok(output) => {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("Failed to assemble input_data.s: {}", stderr));
            }
        }
        Err(e) => return Err(format!("Failed to run gcc for input_data.s: {}", e)),
    }

    // Assemble constants.s
    let compile_const_cmd = format!(
        "riscv64-unknown-elf-gcc -c -march=rv64gc -mabi=lp64d {} -o {}",
        constants_s.display(),
        constants_o.display()
    );
    eprintln!("Gemmini/Spike: Executing command: {}", compile_const_cmd);
    let compile_const_result = Command::new("sh")
        .args(&["-c", &compile_const_cmd])
        .output();

    match compile_const_result {
        Ok(output) => {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("Failed to assemble constants.s: {}", stderr));
            }
        }
        Err(e) => return Err(format!("Failed to run gcc for constants.s: {}", e)),
    }

    // Now link all object files together
    let linker_script = create_linker_script(temp_dir)?;

    // First, let's check if we need to patch the object file for missing constants
    // Check for undefined symbols if debugging
    if std::env::var("GEMMINI_DEBUG").is_ok() {
        let nm_cmd = format!("riscv64-unknown-elf-nm -u {}", obj_file);
        eprintln!("Gemmini/Spike: Executing command: {}", nm_cmd);
        if let Ok(output) = Command::new("sh").args(&["-c", &nm_cmd]).output() {
            let undefined = String::from_utf8_lossy(&output.stdout);
            if !undefined.trim().is_empty() {
                eprintln!("Gemmini/Spike: Undefined symbols: {}", undefined.trim());
            }
        }
    }

    let link_cmd = format!("riscv64-unknown-elf-gcc -static -nostartfiles -nostdlib -march=rv64gc -mabi=lp64d -fPIC -mcmodel=medany -mno-relax -Wl,--no-relax -Wl,--gc-sections -T {} {} {} {} {} -o {}",
                           linker_script, constants_o.display(), startup_o.display(), input_data_o.display(), obj_file, executable.display());
    eprintln!("Gemmini/Spike: Executing command: {}", link_cmd);
    let link_result = Command::new("sh").args(&["-c", &link_cmd]).output();

    match link_result {
        Ok(output) => {
            if output.status.success() {
                Ok(executable)
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                eprintln!("Gemmini/Spike: Linking failed!");
                eprintln!("Gemmini/Spike: Exit code: {:?}", output.status.code());
                eprintln!("Gemmini/Spike: stderr: {}", stderr);
                eprintln!("Gemmini/Spike: stdout: {}", stdout);
                panic!("Linking failed: riscv64-none-elf-gcc returned non-zero exit code");
            }
        }
        Err(e) => {
            eprintln!("Gemmini/Spike: Linker not available: {}", e);
            eprintln!("Gemmini/Spike: This usually means riscv64-unknown-elf-gcc is not installed");
            panic!(
                "RISC-V toolchain not available: {}. Please install riscv64-none-elf-gcc",
                e
            );
        }
    }
}

fn read_spike_output_from_memory(
    spike_state: &mut SpikeState,
    spike_output: &std::process::Output,
) -> Result<(), String> {
    // Try to extract output values from stdout/stderr
    let stdout_text = String::from_utf8_lossy(&spike_output.stdout);
    let stderr_text = String::from_utf8_lossy(&spike_output.stderr);
    let combined_output = format!("{}\n{}", stdout_text, stderr_text);
    
    // Debug: Print the full Spike output (commented out for normal use)
    // eprintln!("=== SPIKE FULL OUTPUT ===");
    // eprintln!("{}", combined_output);
    // eprintln!("=== END SPIKE OUTPUT ===");

    // Look for our specific GEMMINI_OUTPUT line with hex bytes
    let mut result_bytes = Vec::new();

    for line in combined_output.lines() {
        if line.contains("GEMMINI_OUTPUT:") {
            // Extract the hex bytes after "GEMMINI_OUTPUT:"
            if let Some(hex_part) = line.split("GEMMINI_OUTPUT:").nth(1) {
                for hex_byte in hex_part.trim().split_whitespace() {
                    if hex_byte.len() == 2 {
                        if let Ok(byte_val) = u8::from_str_radix(hex_byte, 16) {
                            result_bytes.push(byte_val);
                            // eprintln!("Gemmini/Spike: Parsed hex byte: 0x{:02x}", byte_val)
                        }
                    }
                }
            }
            break;
        }
    }

    if !result_bytes.is_empty() {
        eprintln!(
            "Gemmini/Spike: Extracted {} result bytes from Spike output",
            result_bytes.len()
        );
        eprintln!(
            "Gemmini/Spike: First 4 bytes: {:02x} {:02x} {:02x} {:02x}",
            result_bytes.get(0).unwrap_or(&0),
            result_bytes.get(1).unwrap_or(&0),
            result_bytes.get(2).unwrap_or(&0),
            result_bytes.get(3).unwrap_or(&0)
        );

        let output_idx = if spike_state.buffers.len() >= 3 { 2 } else { 1 };
        if output_idx < spike_state.buffers.len() {
            let copy_size = std::cmp::min(
                result_bytes.len(),
                spike_state.buffers[output_idx].data.len(),
            );
            if copy_size > 0 {
                spike_state.buffers[output_idx].data[..copy_size]
                    .copy_from_slice(&result_bytes[..copy_size]);
                eprintln!(
                    "Gemmini/Spike: Stored {} bytes of result data in output buffer",
                    copy_size
                );

                if result_bytes.len() >= 4 {
                    let u32_value = u32::from_le_bytes([
                        result_bytes[0],
                        result_bytes[1],
                        result_bytes[2],
                        result_bytes[3],
                    ]);
                    let f32_value = f32::from_le_bytes([
                        result_bytes[0],
                        result_bytes[1],
                        result_bytes[2],
                        result_bytes[3],
                    ]);
                    eprintln!("Gemmini/Spike: Result interpreted as u32: {}", u32_value);
                    eprintln!("Gemmini/Spike: Result interpreted as f32: {}", f32_value);
                }
                return Ok(());
            }
        }
    }

    // If no valid output was found, check if the program actually ran
    if combined_output.contains("GEMMINI_START") {
        panic!("Gemmini/Spike: Program started but no output found");
    } else {
        panic!("Gemmini/Spike: No program execution detected");
    }
}

fn convert_mlir_to_executable(
    mlir_file: &str,
    spike_state: &mut SpikeState,
    program_id: usize,
) -> Result<PathBuf, String> {
    // Output files
    let temp_dir = &spike_state.temp_dir;
    let base_name = format!("{}/gemmini_program_{}", temp_dir.display(), program_id);
    let tosa_mlir = format!("{}_linalg.mlir", base_name);
    let mut llvm_ir = format!("{}.ll", base_name);
    let obj_file = format!("{}.o", base_name);
    let executable = format!("{}.out", base_name);

    // Step 1: Convert TOSA to Linalg using mlir-opt
    let mlir_opt_cmd = format!(
        "mlir-opt {} --tosa-to-linalg-pipeline -o {}",
        mlir_file, tosa_mlir
    );
    eprintln!("Gemmini/Spike: Executing command: {}", mlir_opt_cmd);
    let mlir_opt_result = Command::new("sh").args(&["-c", &mlir_opt_cmd]).output();

    match mlir_opt_result {
        Ok(output) => {}
        Err(e) => {
            eprintln!("mlir-opt not available: {}", e);
            return Err(format!("TOSA conversion failed"));
        }
    }

    // Step 1.5: Convert tensor operations to memref for buddy-opt compatibility
    let memref_mlir = format!("{}_memref.mlir", base_name);

    let linalg_content = match fs::read_to_string(&tosa_mlir) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Failed to read linalg MLIR file: {}", e);
            return Err(format!("Failed to read linalg MLIR file"));
        }
    };

    match fs::write(&memref_mlir, linalg_content) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Failed to write memref MLIR file: {}", e);
            return Err(format!("Failed to write memref MLIR file"));
        }
    }

    let tosa_mlir = memref_mlir;

    // Step 2: Generate LLVM IR first, then add constants, then compile
    let llvm_ir_file = format!("{}.ll", base_name);
    let mlir_to_llvm_cmd = format!(
        "buddy-opt {} \
            -pass-pipeline='builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))' | \
        buddy-opt \
            -llvm-request-c-wrappers \
            --one-shot-bufferize='bufferize-function-boundaries' \
            -buffer-deallocation-pipeline \
            -convert-bufferization-to-memref \
            -convert-linalg-to-loops \
            -lower-affine \
            -convert-scf-to-cf \
            -convert-vector-to-llvm \
            -finalize-memref-to-llvm \
            -convert-arith-to-llvm \
            -lower-gemmini \
            -convert-func-to-llvm \
            -reconcile-unrealized-casts | \
        buddy-translate -buddy-to-llvmir > {}",
        tosa_mlir, llvm_ir_file
    );

    eprintln!("Gemmini/Spike: Executing command: {}", mlir_to_llvm_cmd);
    let llvm_ir_result = Command::new("sh").args(&["-c", &mlir_to_llvm_cmd]).output();

    match llvm_ir_result {
        Ok(output) => {
            if !output.status.success() {
                eprintln!("Failed to generate LLVM IR!");
                eprintln!("STDERR:\n{}", String::from_utf8_lossy(&output.stderr));
                return Err(format!("Failed to generate LLVM IR"));
            }
        }
        Err(e) => return Err(format!("Failed to run LLVM IR generation: {}", e)),
    }

    // Read the generated LLVM IR and fix constant references
    eprintln!("Gemmini/Spike: Fixing constant references in LLVM IR");
    let llvm_ir =
        fs::read_to_string(&llvm_ir_file).map_err(|e| format!("Failed to read LLVM IR: {}", e))?;

    // Fix missing constant references in LLVM IR
    let mut modified_ir = llvm_ir;
    let needs_constants =
        modified_ir.contains(".L__constant_") && !modified_ir.contains("@.L__constant_");

    if needs_constants {
        eprintln!("Gemmini/Spike: Adding missing .L__constant definitions");
        modified_ir.push_str("\n\n; Constants for undefined symbols\n");

        // Define all required constants
        const CONSTANT_DEFINITIONS: &[(&str, &str)] = &[
            ("@.L__constant_1x1xi32", "@.L__constant_1x1xi32 = internal constant [1 x i32] [i32 1], align 4\n"),
            ("@.L__constant_1x1xf32", "@.L__constant_1x1xf32 = internal constant [1 x float] [float 1.0], align 4\n"),
            ("@.L__constant_2x2xi32", "@.L__constant_2x2xi32 = internal constant [4 x i32] [i32 1, i32 1, i32 1, i32 1], align 4\n"),
            ("@.L__constant_1x1xi64", "@.L__constant_1x1xi64 = internal constant [1 x i64] [i64 1], align 8\n"),
            ("@.L__constant_1x1xf64", "@.L__constant_1x1xf64 = internal constant [1 x double] [double 1.0], align 8\n"),
        ];

        for (name, definition) in CONSTANT_DEFINITIONS {
            if !modified_ir.contains(name) {
                modified_ir.push_str(definition);
            }
        }
    } else {
        eprintln!("Gemmini/Spike: LLVM IR already has proper constant definitions");
    }

    fs::write(&llvm_ir_file, modified_ir)
        .map_err(|e| format!("Failed to write modified LLVM IR: {}", e))?;

    // Now compile the LLVM IR to object file with PIC to avoid relocation issues
    let compile_cmd = format!(
        "buddy-llc -filetype=obj -mtriple=riscv64-unknown-elf \
            -mattr=+m,+a,+f,+d,+c,+buddyext -float-abi=hard \
            -relocation-model=pic -code-model=small \
            {} -o {}",
        llvm_ir_file, obj_file
    );

    eprintln!("Gemmini/Spike: Compiling LLVM IR to object file");
    eprintln!("Gemmini/Spike: Executing command: {}", compile_cmd);

    // First, ensure the output file doesn't exist
    let _ = std::fs::remove_file(&obj_file);

    let compile_result = Command::new("sh").args(&["-c", &compile_cmd]).output();

    match compile_result {
        Ok(output) => {
            if !output.status.success() {
                eprintln!("LLVM compilation failed!");
                eprintln!("STDERR:\n{}", String::from_utf8_lossy(&output.stderr));
                eprintln!("STDOUT:\n{}", String::from_utf8_lossy(&output.stdout));
                eprintln!("Exit status: {:?}", output.status.code());
                return Err(format!("LLVM compilation failed"));
            } else {
                eprintln!("Gemmini/Spike: Pipeline compilation succeeded");
                if let Ok(metadata) = std::fs::metadata(&obj_file) {
                    eprintln!("Gemmini/Spike: Object file size: {} bytes", metadata.len());
                    if metadata.len() == 0 {
                        eprintln!("Gemmini/Spike: WARNING: Object file is empty!");
                        eprintln!(
                            "Gemmini/Spike: Pipeline stderr: {}",
                            String::from_utf8_lossy(&output.stderr)
                        );
                        eprintln!(
                            "Gemmini/Spike: Pipeline stdout: {}",
                            String::from_utf8_lossy(&output.stdout)
                        );
                    }
                } else {
                    eprintln!("Gemmini/Spike: WARNING: Object file not found after compilation!");
                }
            }
        }
        Err(e) => {
            eprintln!("Pipeline execution failed: {}", e);
            eprintln!("Command was: {}", compile_cmd);
            return Err(format!("Pipeline compilation failed"));
        }
    }

    // Step 4: Object file is ready for execution
    eprintln!("Gemmini/Spike: Step 4 - Object file ready: {}", obj_file);
    eprintln!(
        "Gemmini/Spike: Object file size: {} bytes",
        std::fs::metadata(&obj_file).map(|m| m.len()).unwrap_or(0)
    );

    eprintln!(
        "Gemmini/Spike: Successfully compiled MLIR to executable: {}",
        executable
    );
    Ok(PathBuf::from(executable))
}
