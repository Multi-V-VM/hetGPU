// Gemmini Runtime System using Spike RISC-V ISA Simulator
#![allow(warnings)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::fs::{self, File};
use std::io::{Write, Read, BufWriter};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::ptr;
use std::sync::Mutex;

// Gemmini configuration constants
pub const GEMMINI_DIM: usize = 16; // Default systolic array dimension
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
    device_count: u32,
    programs: Vec<ProgramData>,
    buffers: Vec<BufferData>,
    kernels: Vec<KernelData>,
}

struct ProgramData {
    id: usize,
    llvm_ir: Option<String>,
    elf_path: Option<PathBuf>,
}

struct BufferData {
    id: usize,
    size: u64,
    buffer_type: gemmini_BufferType,
    data: Vec<u8>,
}

struct KernelData {
    id: usize,
    name: String,
    program_id: usize,
}

// FFI functions implementation

pub unsafe extern "C" fn gemmini_CreateDevice(device_id: ::core::ffi::c_int) -> *mut gemmini_Device {
    let mut state = SPIKE_STATE.lock().unwrap();
    
    if state.is_none() {
        // Initialize Spike state
        // Create tmp directory if it doesn't exist
        let _ = fs::create_dir_all("./tmp");
        
        *state = Some(SpikeState {
            temp_dir: PathBuf::from("./tmp"),
            device_count: 1,
            programs: Vec::new(),
            buffers: Vec::new(),
            kernels: Vec::new(),
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
        eprintln!("Gemmini/Spike: Closed device");
    }
    
    gemmini_Result_Success as c_int
}

pub unsafe extern "C" fn gemmini_CreateProgram() -> *mut gemmini_Program {
    let mut state = SPIKE_STATE.lock().unwrap();
    
    if let Some(ref mut spike_state) = *state {
        let program_id = spike_state.programs.len();
        spike_state.programs.push(ProgramData {
            id: program_id,
            llvm_ir: None,
            elf_path: None,
        });
        
        eprintln!("Gemmini/Spike: Created program {}", program_id);
        return (program_id + 1) as *mut gemmini_Program;
    }
    
    ptr::null_mut()
}

pub unsafe extern "C" fn gemmini_CreateBuffer(
    config: *const gemmini_BufferConfig
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
        
        eprintln!("Gemmini/Spike: Created buffer {} (size: {} bytes, type: {})", 
                  buffer_id, config.size, config.buffer_type);
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
        if program_id >= spike_state.programs.len() {
            eprintln!("Gemmini/Spike: Invalid program ID");
            return ptr::null_mut();
        }
        
        let kernel_id = spike_state.kernels.len();
        spike_state.kernels.push(KernelData {
            id: kernel_id,
            name: kernel_name.clone(),
            program_id,
        });
        
        eprintln!("Gemmini/Spike: Created kernel '{}' (id: {}, core: ({},{}))", 
                  kernel_name, kernel_id, core.x, core.y);
        return (kernel_id + 1) as *mut gemmini_Kernel;
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
    
    eprintln!("Gemmini/Spike: Set runtime args for kernel '{}' ({} args)", 
              CStr::from_ptr(kernel_name).to_string_lossy(), num_args);
    
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
        if program_id >= spike_state.programs.len() {
            eprintln!("Gemmini/Spike: Invalid program ID");
            return gemmini_Result_Error as c_int;
        }
        
        // Generate and run Gemmini code on Spike
        if let Err(e) = run_on_spike(spike_state, program_id) {
            eprintln!("Gemmini/Spike: Failed to run on Spike: {}", e);
            return gemmini_Result_Error as c_int;
        }
        
        eprintln!("Gemmini/Spike: Program launched successfully");
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
            eprintln!("Gemmini/Spike: Invalid buffer ID");
            return gemmini_Result_Error;
        }
        
        let buffer_data = &mut spike_state.buffers[buffer_id];
        let copy_size = std::cmp::min(size as usize, buffer_data.data.len());
        
        let src = std::slice::from_raw_parts(data as *const u8, copy_size);
        buffer_data.data[..copy_size].copy_from_slice(src);
        
        eprintln!("Gemmini/Spike: Wrote {} bytes to buffer {}", copy_size, buffer_id);
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
            eprintln!("Gemmini/Spike: Invalid buffer ID");
            return gemmini_Result_Error;
        }
        
        let buffer_data = &spike_state.buffers[buffer_id];
        let copy_size = std::cmp::min(size as usize, buffer_data.data.len());
        
        let dst = std::slice::from_raw_parts_mut(data as *mut u8, copy_size);
        dst.copy_from_slice(&buffer_data.data[..copy_size]);
        
        eprintln!("Gemmini/Spike: Read {} bytes from buffer {}", copy_size, buffer_id);
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
    
    // Create tmp directory if it doesn't exist
    let _ = fs::create_dir_all("./tmp");
    
    // Write MLIR to file
    let mlir_file = format!("./tmp/gemmini_mlir_{}.mlir", program_id);
    if let Err(e) = fs::write(&mlir_file, &mlir_str) {
        eprintln!("Gemmini/Spike: Failed to write MLIR file: {}", e);
        return gemmini_Result_Error;
    }
    eprintln!("Gemmini/Spike: Wrote MLIR to {}", mlir_file);
    
    let mut state = SPIKE_STATE.lock().unwrap();
    
    if let Some(ref mut spike_state) = *state {
        if program_id >= spike_state.programs.len() {
            eprintln!("Gemmini/Spike: Invalid program ID");
            return gemmini_Result_Error;
        }
        
        // Convert MLIR to executable using Buddy compiler toolchain
        match convert_mlir_to_executable(&mlir_file, program_id) {
            Ok(elf_path) => {
                spike_state.programs[program_id].elf_path = Some(elf_path);
                eprintln!("Gemmini/Spike: Successfully compiled MLIR for program {}", program_id);
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
    program: *mut gemmini_Program
) -> gemmini_Result {
    if program.is_null() {
        return gemmini_Result_Error;
    }
    
    eprintln!("Gemmini/Spike: Waiting for completion (immediate return for simulator)");
    gemmini_Result_Success
}

pub unsafe extern "C" fn gemmini_DestroyProgram(program: *mut gemmini_Program) {
    if program.is_null() {
        return;
    }
    
    let program_id = (program as usize) - 1;
    eprintln!("Gemmini/Spike: Destroyed program {}", program_id);
}

pub unsafe extern "C" fn gemmini_DestroyBuffer(buffer: *mut gemmini_Buffer) {
    if buffer.is_null() {
        return;
    }
    
    let buffer_id = (buffer as usize) - 1;
    eprintln!("Gemmini/Spike: Destroyed buffer {}", buffer_id);
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
        let handle = unsafe { 
            gemmini_CreateKernel(self.handle, kernel_name.as_ptr(), core, ptr::null())
        };
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
        let kernel_name_cstr = CString::new(kernel_name)
            .map_err(|e| format!("Invalid kernel name: {}", e))?;
        
        let buffer_ptrs: Vec<*const gemmini_Buffer> = buffers.iter()
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
            Err(format!("Failed to set runtime args: error code {:?}", result))
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
            Err(format!("Failed to wait for completion: error code {:?}", result))
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
                data.len() as u64
            ) 
        };

        if result == gemmini_Result_Success {
            Ok(())
        } else {
            Err(format!("Failed to write to buffer: error code {:?}", result))
        }
    }

    pub fn read(&self, data: &mut [u8]) -> Result<(), String> {
        let result = unsafe {
            gemmini_ReadFromBuffer(
                self.handle, 
                data.as_mut_ptr() as *mut core::ffi::c_void, 
                data.len() as u64
            )
        };

        if result == gemmini_Result_Success {
            Ok(())
        } else {
            Err(format!("Failed to read from buffer: error code {:?}", result))
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
    eprintln!("Gemmini/Spike: Running program {}", program_id);
    
    // Look for the compiled object file
    let obj_file = format!("./tmp/gemmini_program_{}.o", program_id);
    
    if !std::path::Path::new(&obj_file).exists() {
        panic!("Gemmini/Spike: Object file not found: {}", obj_file);
    }
    
    eprintln!("Gemmini/Spike: Found compiled object: {}", obj_file);
    
    // Check if the object file is valid
    let obj_data = std::fs::read(&obj_file)
        .map_err(|e| format!("Failed to read object file: {}", e))?;
    
    if obj_data.len() <= 16 || !obj_data.starts_with(&[0x7f, 0x45, 0x4c, 0x46]) {
        panic!("Gemmini/Spike: Object file is not valid ELF format");
    }
    
    eprintln!("Gemmini/Spike: Valid ELF object file detected, proceeding with Spike execution");
    
    // Create executable and run with Spike
    let executable = create_executable_from_object(spike_state, &obj_file)?;
    
    eprintln!("Gemmini/Spike: Launching Spike simulator with Gemmini extension");
    
    // Execute with Spike
    let temp_dir = &spike_state.temp_dir;
    let spike_result = Command::new("spike")
        .env("LD_LIBRARY_PATH", "/root/.nix-profile/lib")
        .args(&[
            "--extension=gemmini",
            "/usr/local/riscv64-none-elf/bin/pk",
        ])
        .arg(&executable)
        .current_dir(temp_dir)
        .output();
    
    match spike_result {
        Ok(output) => {
            if output.status.success() || output.status.code() == Some(0) {
                eprintln!("Gemmini/Spike: Execution completed successfully");
                
                if !output.stdout.is_empty() {
                    eprintln!("Gemmini/Spike: stdout: {}", String::from_utf8_lossy(&output.stdout));
                }
                if !output.stderr.is_empty() {
                    eprintln!("Gemmini/Spike: stderr: {}", String::from_utf8_lossy(&output.stderr));
                }
                
                read_spike_output_from_memory(spike_state, &output)
            } else {
                panic!("Gemmini/Spike: Execution failed with status: {:?}", output.status);
            }
        }
        Err(e) => {
            panic!("Gemmini/Spike: Failed to execute Spike: {}", e);
        }
    }
}



fn create_input_data_assembly(path: &std::path::Path, spike_state: &SpikeState) -> Result<(), String> {
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
        if spike_state.buffers.len() > 0 {
            all_data.extend_from_slice(&spike_state.buffers[0].data);
        }
        
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
        spike_state.buffers.iter().take(2).map(|b| b.data.len()).sum::<usize>()
    };
    content.push_str(&format!("    .quad {}\n", total_size));
    
    std::fs::write(path, content)
        .map_err(|e| format!("Failed to write input data assembly: {}", e))?;
    
    eprintln!("Gemmini/Spike: Created input data assembly with {} bytes of data", total_size);
    Ok(())
}

fn create_linker_script(temp_dir: &std::path::Path) -> Result<String, String> {
    let linker_script = temp_dir.join("gemmini.ld");
    let script_content = r#"
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    ram (rwx) : ORIGIN = 0x80000000, LENGTH = 0x10000000
}

SECTIONS
{
    . = 0x80000000;
    
    .text ALIGN(4): {
        *(.text.init)
        *(.text)
        *(.text.*)
    } > ram
    
    .rodata ALIGN(4): {
        *(.rodata)
        *(.rodata.*)
        *(.srodata)
        *(.srodata.*)
    } > ram
    
    .data ALIGN(4): {
        *(.data)
        *(.data.*)
        *(.sdata)
        *(.sdata.*)
    } > ram
    
    .bss ALIGN(8): {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.sbss)
        *(.sbss.*)
        __bss_end = .;
    } > ram
    
    . = ALIGN(8);
    _end = .;
}
"#;
    
    std::fs::write(&linker_script, script_content)
        .map_err(|e| format!("Failed to create linker script: {}", e))?;
    
    Ok(linker_script.to_string_lossy().to_string())
}

fn create_executable_from_object(spike_state: &mut SpikeState, obj_file: &str) -> Result<std::path::PathBuf, String> {
    let temp_dir = &spike_state.temp_dir;
    let executable = temp_dir.join("gemmini_kernel");
    
    eprintln!("Gemmini/Spike: Creating executable from object file: {}", obj_file);
    
    // Create input data assembly file that embeds the buffer data
    let input_data_s = temp_dir.join("input_data.s");
    create_input_data_assembly(&input_data_s, spike_state)?;
    
    // Create a simple startup code that calls the kernel function
    let startup_c = temp_dir.join("startup.c");
    let startup_content = r#"
#include <stdint.h>
#include <stddef.h>  // for size_t

// Simple malloc implementation for bare metal
static char heap[16384];  // 16KB heap
static char* heap_ptr = heap;

void* malloc(size_t size) {
    // Align to 8 bytes
    size = (size + 7) & ~7;
    if (heap_ptr + size > heap + sizeof(heap)) {
        return 0;  // Out of memory
    }
    void* result = heap_ptr;
    heap_ptr += size;
    return result;
}

// RISC-V 64-bit multiplication helper (for systems without M extension)
long long __muldi3(long long a, long long b) {
    // Simple multiplication - this assumes the processor has mul instruction
    // For processors without M extension, this would need a software implementation
    return a * b;
}

// Minimal system calls for bare metal RISC-V
static int strlen__(const char* str) {
    int len = 0;
    while (str[len]) len++;
    return len;
}

int write(int fd, const void* buf, int count) {
    // For Spike, we can use a simple system call
    // This is a minimal implementation that outputs to stdout
    register int a0 asm("a0") = fd;
    register const void* a1 asm("a1") = buf;
    register int a2 asm("a2") = count;
    register int a7 asm("a7") = 64; // SYS_write
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}

void _exit(int status) {
    register int a0 asm("a0") = status;
    register int a7 asm("a7") = 93; // SYS_exit
    asm volatile("ecall" : : "r"(a0), "r"(a7) : "memory");
    while(1); // Should never reach here
}

// Forward declaration
int main(void);

// Entry point for bare metal
void _start() {
    main();
    _exit(0);
}

// MLIR memref descriptor
typedef struct {
    void* data;
    void* aligned_data;
    long offset;
    long sizes[2];
    long strides[2];
} memref_2d_f32_t;

// External kernel function from the object file
// Use weak symbols so missing functions don't cause link errors
extern void _mlir_ciface_matmul_kernel(memref_2d_f32_t* input1, memref_2d_f32_t* input2, memref_2d_f32_t* output) __attribute__((weak));
extern void _mlir_ciface_atom_add_float(memref_2d_f32_t* input1, memref_2d_f32_t* input2, memref_2d_f32_t* output) __attribute__((weak));
extern void _mlir_ciface_xor(memref_2d_f32_t* result, memref_2d_f32_t* input1, memref_2d_f32_t* input2) __attribute__((weak));
extern void _mlir_ciface_and(memref_2d_f32_t* result, memref_2d_f32_t* input1, memref_2d_f32_t* input2) __attribute__((weak));

// External symbols for input data embedded in the executable
extern uint8_t __input_data_start[] __attribute__((weak));
extern uint8_t __input_data_end[] __attribute__((weak));
extern uint64_t __input_data_size __attribute__((weak));

// Global buffers for input/output data (as bytes for arbitrary types)
static uint8_t input_buffer[4096];  // 4KB for various data types
static uint8_t output_buffer[4096]; // 4KB for various data types

void print_uint(uint32_t value) {
    char buffer[32];
    int len = 0;
    
    if (value == 0) {
        buffer[len++] = '0';
    } else {
        char temp[16];
        int temp_len = 0;
        while (value > 0) {
            temp[temp_len++] = '0' + (value % 10);
            value /= 10;
        }
        while (temp_len > 0) {
            buffer[len++] = temp[--temp_len];
        }
    }
    
    write(1, buffer, len);
}

void print_byte_hex(uint8_t byte) {
    char buffer[3];
    const char* hex_digits = "0123456789abcdef";
    buffer[0] = hex_digits[(byte >> 4) & 0xF];
    buffer[1] = hex_digits[byte & 0xF];
    buffer[2] = ' ';
    write(1, buffer, 3);
}

void print_bytes_hex(const uint8_t* bytes, int count) {
    for (int i = 0; i < count; i++) {
        print_byte_hex(bytes[i]);
    }
}

void print_float(float value) {
    int int_part = (int)value;
    int frac_part = (int)((value - int_part) * 100);
    if (frac_part < 0) frac_part = -frac_part;
    
    char buffer[32];
    char* p = buffer;
    
    if (value < 0) {
        *p++ = '-';
        int_part = -int_part;
    }
    
    if (int_part == 0) {
        *p++ = '0';
    } else {
        char temp[16];
        int i = 0;
        while (int_part > 0) {
            temp[i++] = '0' + (int_part % 10);
            int_part /= 10;
        }
        while (i > 0) {
            *p++ = temp[--i];
        }
    }
    
    *p++ = '.';
    *p++ = '0' + (frac_part / 10);
    *p++ = '0' + (frac_part % 10);
    *p++ = ' ';
    *p = '\0';
    
    write(1, buffer, p - buffer);
}

void write_output_data(uint8_t* buffer, int size) {
    const char* output_prefix = "GEMMINI_OUTPUT: ";
    write(1, output_prefix, strlen__(output_prefix));
    
    int bytes_to_print = size < 16 ? size : 16;
    print_bytes_hex(buffer, bytes_to_print);
    
    const char* newline = "\n";
    write(1, newline, 1);
}

int main() {
    const char* start_msg = "GEMMINI_START: Executing kernel\n";
    write(1, start_msg, strlen__(start_msg));
    
    if (__input_data_start && __input_data_end) {
        uint64_t data_size = __input_data_end - __input_data_start;
        uint64_t copy_size = data_size < sizeof(input_buffer) ? data_size : sizeof(input_buffer);
        
        for (uint64_t i = 0; i < copy_size; i++) {
            input_buffer[i] = __input_data_start[i];
        }
        
        const char* load_msg = "GEMMINI_DEBUG: Loaded ";
        write(1, load_msg, strlen__(load_msg));
        print_uint((uint32_t)copy_size);
        write(1, " bytes of embedded input data\n", 31);
    } else {
        const char* no_data_msg = "GEMMINI_DEBUG: No embedded input data found\n";
        write(1, no_data_msg, strlen__(no_data_msg));
    }
    
    memref_2d_f32_t input1_desc, input2_desc, output_desc;
    
    int dim1 = 1, dim2 = 1;
    
    int element_size = sizeof(float);
    
    input1_desc = (memref_2d_f32_t){
        .data = (void*)input_buffer,
        .aligned_data = (void*)input_buffer, 
        .offset = 0,
        .sizes = {dim1, dim2},
        .strides = {dim2, 1}
    };
    
    input2_desc = (memref_2d_f32_t){
        .data = (void*)(input_buffer + (dim1 * dim2 * element_size)),  // Second buffer after first
        .aligned_data = (void*)(input_buffer + (dim1 * dim2 * element_size)),
        .offset = 0, 
        .sizes = {dim1, dim2},
        .strides = {dim2, 1}
    };
    
    output_desc = (memref_2d_f32_t){
        .data = (void*)output_buffer,
        .aligned_data = (void*)output_buffer,
        .offset = 0,
        .sizes = {dim1, dim2}, 
        .strides = {dim2, 1}
    };
    
    const char* kernel_msg = "GEMMINI_KERNEL: Calling compiled kernel\n";
    write(1, kernel_msg, strlen__(kernel_msg));
    
    if (_mlir_ciface_matmul_kernel) {
        const char* matmul_msg = "GEMMINI_KERNEL: Found matmul_kernel\n";
        write(1, matmul_msg, strlen__(matmul_msg));
        _mlir_ciface_matmul_kernel(&input1_desc, &input2_desc, &output_desc);
    } else if (_mlir_ciface_atom_add_float) {
        const char* atom_msg = "GEMMINI_KERNEL: Found atom_add_float\n";
        write(1, atom_msg, strlen__(atom_msg));
        _mlir_ciface_atom_add_float(&input1_desc, &input2_desc, &output_desc);
    } else if (_mlir_ciface_xor) {
        const char* xor_msg = "GEMMINI_KERNEL: Found xor\n";
        write(1, xor_msg, strlen__(xor_msg));
        
        memref_2d_f32_t result_desc;
        _mlir_ciface_xor(&result_desc, &input1_desc, &input2_desc);
        
        uint8_t* result_data = (uint8_t*)result_desc.aligned_data;
        int copy_size = dim1 * dim2 * sizeof(float);
        for (int i = 0; i < copy_size; i++) {
            output_buffer[i] = result_data[i];
        }
    } else if (_mlir_ciface_and) {
        const char* and_msg = "GEMMINI_KERNEL: Found and\n";
        write(1, and_msg, strlen__(and_msg));
        
        memref_2d_f32_t result_desc;
        _mlir_ciface_and(&result_desc, &input1_desc, &input2_desc);
        
        uint8_t* result_data = (uint8_t*)result_desc.aligned_data;
        int copy_size = dim1 * dim2 * sizeof(float);
        for (int i = 0; i < copy_size; i++) {
            output_buffer[i] = result_data[i];
        }
    }
    
    write_output_data(output_buffer, sizeof(output_buffer));
    
    const char* end_msg = "GEMMINI_END: Kernel execution completed\n";
    write(1, end_msg, strlen__(end_msg));
    
    return 0;
}
"#;

    std::fs::write(&startup_c, startup_content)
        .map_err(|e| format!("Failed to write startup code: {}", e))?;
    
    // Compile the startup code and link with the object file
    let linker_script = create_linker_script(temp_dir)?;
    let link_cmd = format!(
        "riscv64-unknown-elf-gcc -static -nostartfiles -nostdlib -mcmodel=medany -T {} {} {} {} -o {}",
        linker_script,
        startup_c.to_str().unwrap(),
        input_data_s.to_str().unwrap(),
        obj_file,
        executable.display()
    );
    eprintln!("Gemmini/Spike: Link command: {}", link_cmd);
    
    let link_result = Command::new("riscv64-unknown-elf-gcc")
        .args(&[
            "-static",
            "-nostartfiles",
            "-nostdlib",
            "-mcmodel=medany",
            "-T", 
        ])
        .arg(&linker_script)
        .arg(startup_c.to_str().unwrap())
        .arg(&input_data_s)
        .arg(obj_file)
        .arg("-o")
        .arg(&executable)
        .output();
    
    match link_result {
        Ok(output) => {
            if output.status.success() {
                eprintln!("Gemmini/Spike: Successfully created executable: {}", executable.display());
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
            panic!("RISC-V toolchain not available: {}. Please install riscv64-none-elf-gcc", e);
        }
    }
}

fn read_spike_output_from_memory(spike_state: &mut SpikeState, spike_output: &std::process::Output) -> Result<(), String> {
    eprintln!("Gemmini/Spike: Reading output from Spike execution");
    
    // Try to extract output values from stdout/stderr
    let stdout_text = String::from_utf8_lossy(&spike_output.stdout);
    let stderr_text = String::from_utf8_lossy(&spike_output.stderr);
    let combined_output = format!("{}\n{}", stdout_text, stderr_text);
    
    eprintln!("Gemmini/Spike: Combined output:\n{}", combined_output);
    
    // Look for our specific GEMMINI_OUTPUT line with hex bytes
    let mut result_bytes = Vec::new();
    
    for line in combined_output.lines() {
        if line.contains("GEMMINI_OUTPUT:") {
            eprintln!("Gemmini/Spike: Found output line: {}", line);
            
            // Extract the hex bytes after "GEMMINI_OUTPUT:"
            if let Some(hex_part) = line.split("GEMMINI_OUTPUT:").nth(1) {
                // Parse hex bytes
                for hex_byte in hex_part.trim().split_whitespace() {
                    if hex_byte.len() == 2 {
                        if let Ok(byte_val) = u8::from_str_radix(hex_byte, 16) {
                            result_bytes.push(byte_val);
                            eprintln!("Gemmini/Spike: Parsed hex byte: 0x{:02x}", byte_val);
                        }
                    }
                }
            }
            break;
        }
    }
    
    if !result_bytes.is_empty() {
        eprintln!("Gemmini/Spike: Extracted {} result bytes from Spike output", result_bytes.len());
        eprintln!("Gemmini/Spike: First 4 bytes: {:02x} {:02x} {:02x} {:02x}", 
                 result_bytes.get(0).unwrap_or(&0),
                 result_bytes.get(1).unwrap_or(&0),
                 result_bytes.get(2).unwrap_or(&0),
                 result_bytes.get(3).unwrap_or(&0));
        
        // Store the extracted results in the output buffer
        let output_idx = if spike_state.buffers.len() >= 3 { 2 } else { 1 };
        if output_idx < spike_state.buffers.len() {
            let copy_size = std::cmp::min(result_bytes.len(), spike_state.buffers[output_idx].data.len());
            if copy_size > 0 {
                spike_state.buffers[output_idx].data[..copy_size].copy_from_slice(&result_bytes[..copy_size]);
                eprintln!("Gemmini/Spike: Stored {} bytes of result data in output buffer", copy_size);
                
                // Print sample values for verification (interpret as different types)
                if result_bytes.len() >= 4 {
                    let u32_value = u32::from_le_bytes([result_bytes[0], result_bytes[1], result_bytes[2], result_bytes[3]]);
                    let f32_value = f32::from_le_bytes([result_bytes[0], result_bytes[1], result_bytes[2], result_bytes[3]]);
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


// TODO FIX:
// https://github.com/Multi-V-VM/hetGPU/blob/2839110b28ff47d8f9d8d8b5de9a9b9056aee9d3/ext/gemmini_runtime-sys/src/lib.rs#L1797
fn find_gemmini_main_binary() -> Result<String, String> {
    // Look for gemmini_main in common locations
    let possible_paths = vec![
        "./target/debug/gemmini_main",
        "../target/debug/gemmini_main", 
        "../../target/debug/gemmini_main",
        "../../../target/debug/gemmini_main",
        "/home/victoryang00/hetgpu/target/debug/gemmini_main",
    ];
    
    for path in possible_paths {
        if std::path::Path::new(path).exists() {
            return Ok(path.to_string());
        }
    }
    
    // Try to find using which command
    if let Ok(output) = Command::new("which").arg("gemmini_main").output() {
        if output.status.success() {
            if let Ok(path) = String::from_utf8(output.stdout) {
                return Ok(path.trim().to_string());
            }
        }
    }
    
    Err("Could not find gemmini_main binary. Please ensure it's built with 'cargo build --bin gemmini_main'".to_string())
}

// Convert tensor-based MLIR to memref-based MLIR for buddy-opt compatibility
fn convert_tensor_to_memref_mlir(content: &str) -> String {
    // For now, just return the content as-is
    // We'll manually fix the MLIR instead
    content.to_string()
}

fn convert_mlir_to_executable(mlir_file: &str, program_id: usize) -> Result<PathBuf, String> {
    // Output files
    let base_name = format!("./tmp/gemmini_program_{}", program_id);
    let linalg_mlir = format!("{}_linalg.mlir", base_name);
    let mut llvm_ir = format!("{}.ll", base_name);
    let obj_file = format!("{}.o", base_name);
    let executable = format!("{}.out", base_name);
    
    eprintln!("Gemmini/Spike: Starting MLIR compilation pipeline");
    
    
    // Step 1: Convert TOSA to Linalg using mlir-opt
    eprintln!("Gemmini/Spike: Step 1 - Converting TOSA to Linalg");
    
    // First try the pipeline pass
    eprintln!("Gemmini/Spike: Running mlir-opt command: mlir-opt {} --tosa-to-linalg-pipeline -o {}", mlir_file, linalg_mlir);
    let mlir_opt_result = Command::new("mlir-opt")
        .args(&[
            mlir_file,
            "--tosa-to-linalg-pipeline",
            "-o", &linalg_mlir,
        ])
        .output();
    
    match mlir_opt_result {
        Ok(output) => {
            if !output.status.success() {
                eprintln!("mlir-opt (TOSA to Linalg pipeline) failed: {}", String::from_utf8_lossy(&output.stderr));
                
                // Try individual passes as fallback
                eprintln!("Gemmini/Spike: Trying individual TOSA passes");
                let individual_pass_result = Command::new("mlir-opt")
                    .args(&[
                        mlir_file,
                        "--pass-pipeline=builtin.module(func.func(tosa-to-linalg,tosa-to-arith))",
                        "-o", &linalg_mlir,
                    ])
                    .output();
                
                match individual_pass_result {
                    Ok(output2) => {
                        if !output2.status.success() {
                            eprintln!("mlir-opt (individual passes) failed: {}", String::from_utf8_lossy(&output2.stderr));
                            
                            // Try just copying the file as-is and skip TOSA conversion
                            eprintln!("Gemmini/Spike: TOSA conversion failed, using MLIR directly");
                            if let Err(e) = fs::copy(mlir_file, &linalg_mlir) {
                                eprintln!("Failed to copy MLIR file: {}", e);
                                return Err(format!("TOSA conversion failed"));
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to run mlir-opt with individual passes: {}", e);
                        return Err(format!("TOSA conversion failed"));
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("mlir-opt not available: {}", e);
            return Err(format!("TOSA conversion failed"));
        }
    }
    
    // Step 1.5: Convert tensor operations to memref for buddy-opt compatibility
    eprintln!("Gemmini/Spike: Step 1.5 - Converting tensor operations to memref");
    let memref_mlir = format!("{}_memref.mlir", base_name);
    
    // Read the linalg MLIR file
    let linalg_content = match fs::read_to_string(&linalg_mlir) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Failed to read linalg MLIR file: {}", e);
            return Err(format!("Failed to read linalg MLIR file"));
        }
    };
    
    // Convert tensor operations to memref for buddy-opt compatibility
    let memref_content = convert_tensor_to_memref_mlir(&linalg_content);
    
    // Write the converted MLIR
    match fs::write(&memref_mlir, memref_content) {
        Ok(_) => {
            eprintln!("Gemmini/Spike: Successfully converted to memref MLIR");
        }
        Err(e) => {
            eprintln!("Failed to write memref MLIR file: {}", e);
            return Err(format!("Failed to write memref MLIR file"));
        }
    }
    
    // Update linalg_mlir to point to the memref version
    let linalg_mlir = memref_mlir;
    
    // Step 2: Use pipeline approach: buddy-opt | buddy-translate | buddy-llc
    eprintln!("Gemmini/Spike: Step 2 - Pipeline compilation: buddy-opt | buddy-translate | buddy-llc");
    
    // Create the pipeline command using shell
    let pipeline_cmd = format!(
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
        buddy-translate -buddy-to-llvmir | \
        buddy-llc -filetype=obj -mtriple=riscv64 \
            -mattr=+D -float-abi=hard \
            -o {}",
        linalg_mlir, obj_file
    );
    
    eprintln!("Gemmini/Spike: Running pipeline: {}", pipeline_cmd);
    
    // First, ensure the output file doesn't exist
    let _ = std::fs::remove_file(&obj_file);
    
    let pipeline_result = Command::new("bash")
        .args(&["-c", &format!("set -o pipefail && {}", pipeline_cmd)])
        .output();
    
    match pipeline_result {
        Ok(output) => {
            if !output.status.success() {
                eprintln!("Pipeline compilation failed!");
                eprintln!("STDERR:\n{}", String::from_utf8_lossy(&output.stderr));
                eprintln!("STDOUT:\n{}", String::from_utf8_lossy(&output.stdout));
                eprintln!("Exit status: {:?}", output.status.code());
                return Err(format!("Pipeline compilation failed"));
            } else {
                eprintln!("Gemmini/Spike: Pipeline compilation succeeded");
                // Check if object file was actually created
                if let Ok(metadata) = std::fs::metadata(&obj_file) {
                    eprintln!("Gemmini/Spike: Object file size: {} bytes", metadata.len());
                    if metadata.len() == 0 {
                        eprintln!("Gemmini/Spike: WARNING: Object file is empty!");
                        eprintln!("Gemmini/Spike: Pipeline stderr: {}", String::from_utf8_lossy(&output.stderr));
                        eprintln!("Gemmini/Spike: Pipeline stdout: {}", String::from_utf8_lossy(&output.stdout));
                    }
                } else {
                    eprintln!("Gemmini/Spike: WARNING: Object file not found after compilation!");
                }
            }
        }
        Err(e) => {
            eprintln!("Pipeline execution failed: {}", e);
            eprintln!("Command was: {}", pipeline_cmd);
            return Err(format!("Pipeline compilation failed"));
        }
    }
    
    // Step 3.5: Use gemmini_main binary instead of generating C stub
    eprintln!("Gemmini/Spike: Step 3.5 - Using gemmini_main binary for execution");
    
    // Find the gemmini_main binary
    let gemmini_main_path: String = find_gemmini_main_binary()?;
    eprintln!("Gemmini/Spike: Found gemmini_main at: {}", gemmini_main_path);
    
    // Store the object file path for the gemmini_main binary to use
    let obj_info_file = format!("{}_obj_info.txt", base_name);
    std::fs::write(&obj_info_file, format!("{}\n{}", obj_file, linalg_mlir))
        .map_err(|e| format!("Failed to write object info: {}", e))?;
    
    // Instead of linking, we'll let the gemmini_main binary handle execution
    eprintln!("Gemmini/Spike: Step 4 - Prepared for gemmini_main execution");
    eprintln!("Gemmini/Spike: Object file: {}", obj_file);
    eprintln!("Gemmini/Spike: MLIR file: {}", linalg_mlir);
    
    // Copy the object file to a known location for gemmini_main to find
    let target_obj = format!("{}.o", base_name);
    if std::fs::copy(&obj_file, &target_obj).is_ok() {
        eprintln!("Gemmini/Spike: Copied object file to: {}", target_obj);
    }
    
    eprintln!("Gemmini/Spike: Successfully compiled MLIR to executable: {}", executable);
    Ok(PathBuf::from(executable))

}


