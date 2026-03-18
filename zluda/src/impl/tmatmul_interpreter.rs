// TMatmul Assembly Interpreter
// Parses and executes TMatmul assembly instructions directly on host memory.
// Each register holds a single f32 value (scalar mode, d=1).
// The interpreter loops over all elements (total = grid*block) executing
// instructions element-by-element.

use std::collections::HashMap;
use std::sync::Mutex;

// Track the last GEMM output (C pointer, M*N elements, dtype)
// Used by ArgMax to find the logits tensor
lazy_static::lazy_static! {
    pub static ref LAST_GEMM_OUTPUT: Mutex<(usize, usize, i32)> = Mutex::new((0, 0, 0));
}

// Track the last kernel output for dataflow-based parameter resolution.
// (ptr, size_in_bytes, elem_size)
// Updated by elementwise, softmax, LayerNorm, and reduce handlers.
lazy_static::lazy_static! {
    pub static ref LAST_KERNEL_OUTPUT: Mutex<(usize, usize, usize)> = Mutex::new((0, 0, 0));
}

pub fn set_last_output(ptr: usize, size_bytes: usize, elem_size: usize) {
    if let Ok(mut last) = LAST_KERNEL_OUTPUT.lock() {
        *last = (ptr, size_bytes, elem_size);
    }
}

pub fn get_last_output() -> (usize, usize, usize) {
    if let Ok(last) = LAST_KERNEL_OUTPUT.lock() {
        *last
    } else {
        (0, 0, 0)
    }
}

/// Check if a memory page is mapped and readable (prevents SIGSEGV)
unsafe fn is_readable(addr: *const u8) -> bool {
    if addr.is_null() {
        return false;
    }
    let page_size = 4096usize;
    let page_addr = (addr as usize) & !(page_size - 1);
    // msync returns 0 if the page is mapped, -1 with ENOMEM if not
    libc::msync(page_addr as *mut libc::c_void, 1, libc::MS_ASYNC) == 0
}

/// Memory reference in an instruction
#[derive(Debug, Clone)]
pub enum MemRef {
    /// PARAM_N - kernel parameter index, with optional byte offset
    Param(usize),
    /// Spill slot
    Spill(String),
    /// Constant value
    Const(f32),
}

/// Parsed instruction
#[derive(Debug, Clone)]
pub enum Instruction {
    LoadVector { dst: u8, mem: MemRef },
    StoreVector { src: u8, mem: MemRef },
    Add { dst: u8, src1: u8, src2: u8 },
    Sub { dst: u8, src1: u8, src2: u8 },
    Mul { dst: u8, src1: u8, src2: u8 },
    Div { dst: u8, src1: u8, src2: u8 },
    Sigmoid { dst: u8, src: u8 },
    ComplementSigmoid { dst: u8, src: u8 },
    SiLU { dst: u8, src: u8 },
    ReLU { dst: u8, src: u8 },
    GeLU { dst: u8, src: u8 },
    Tanh { dst: u8, src: u8 },
    Neg { dst: u8, src: u8 },
    Reciprocal { dst: u8, src: u8 },
    Rsqrt { dst: u8, src: u8 },
    Sqrt { dst: u8, src: u8 },
    Exp { dst: u8, src: u8 },
    Log { dst: u8, src: u8 },
    Log1p { dst: u8, src: u8 },
    Abs { dst: u8, src: u8 },
    Sin { dst: u8, src: u8 },
    Cos { dst: u8, src: u8 },
    Floor { dst: u8, src: u8 },
    Ceil { dst: u8, src: u8 },
    Round { dst: u8, src: u8 },
    Sign { dst: u8, src: u8 },
    Residual { dst: u8, src1: u8, src2: u8 },
    Softmax { dst: u8, src: u8 },
    ReduceSum { dst: u8, src: u8 },
    ReduceMean { dst: u8, src: u8 },
    ReduceMax { dst: u8, src: u8 },
    ReduceVar { dst: u8, src: u8 },
    LayerNorm { dst: u8, input: u8, weight: u8, bias: Option<u8> },
    RmsClear,
    RmsAccumulate { src: u8 },
    RmsFinishAccumulate { total_length: MemRef },
    RmsNorm { dst: u8, src: u8 },
    Norm { dst: u8, src: u8 },
    TMatmulImport { src: u8 },
    TMatmulGo { weights: MemRef },
    TMatmulExport { dst: u8 },
    Nop,
}

/// Parse a register name like "v0" -> 0, "v7" -> 7
fn parse_register(s: &str) -> Option<u8> {
    let s = s.trim();
    if s.starts_with('v') {
        s[1..].parse::<u8>().ok()
    } else {
        None
    }
}

/// Parse a memory reference
fn parse_memref(s: &str) -> MemRef {
    let s = s.trim();
    if s.starts_with("PARAM_") {
        if let Ok(idx) = s[6..].parse::<usize>() {
            return MemRef::Param(idx);
        }
    }
    if s.starts_with("SPILL_") {
        return MemRef::Spill(s.to_string());
    }
    if s.starts_with("CONST_") {
        let val_str = &s[6..];
        // Handle common constants
        match val_str {
            "0" | "0.0" | "0f00000000" => return MemRef::Const(0.0),
            "1" | "1.0" | "0f3f800000" => return MemRef::Const(1.0),
            _ => {
                if let Ok(v) = val_str.parse::<f32>() {
                    return MemRef::Const(v);
                }
                // Try hex float format (0fXXXXXXXX)
                if val_str.starts_with("0f") || val_str.starts_with("0F") {
                    if let Ok(bits) = u32::from_str_radix(&val_str[2..], 16) {
                        return MemRef::Const(f32::from_bits(bits));
                    }
                }
                return MemRef::Const(0.0);
            }
        }
    }
    // Fallback: try to interpret as PARAM_N if it's a pure number
    if let Ok(idx) = s.parse::<usize>() {
        return MemRef::Param(idx);
    }
    // Unknown memory reference - treat as spill
    MemRef::Spill(s.to_string())
}

/// Parse a single assembly line into an instruction
fn parse_line(line: &str) -> Option<Instruction> {
    let trimmed = line.trim();
    // Skip empty lines and comments
    if trimmed.is_empty() || trimmed.starts_with(';') {
        return None;
    }
    // Skip labels
    if trimmed.ends_with(':') {
        return None;
    }

    // Split into opcode and operands
    // Format: "    ldv    v0,PARAM_0" or "    add    v2,v0,v1"
    let parts: Vec<&str> = trimmed.splitn(2, |c: char| c.is_whitespace()).collect();
    if parts.is_empty() {
        return None;
    }
    let opcode = parts[0].trim();
    let operands_str = if parts.len() > 1 { parts[1].trim() } else { "" };

    // Split operands by comma
    let operands: Vec<&str> = if operands_str.is_empty() {
        vec![]
    } else {
        operands_str.split(',').map(|s| s.trim()).collect()
    };

    match opcode {
        "ldv" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let mem = parse_memref(operands[1]);
                Some(Instruction::LoadVector { dst, mem })
            } else {
                None
            }
        }
        "sv" => {
            if operands.len() >= 2 {
                let src = parse_register(operands[0])?;
                let mem = parse_memref(operands[1]);
                Some(Instruction::StoreVector { src, mem })
            } else {
                None
            }
        }
        "add" => {
            if operands.len() >= 3 {
                let dst = parse_register(operands[0])?;
                let src1 = parse_register(operands[1])?;
                let src2 = parse_register(operands[2])?;
                Some(Instruction::Add { dst, src1, src2 })
            } else {
                None
            }
        }
        "sub" => {
            if operands.len() >= 3 {
                let dst = parse_register(operands[0])?;
                let src1 = parse_register(operands[1])?;
                let src2 = parse_register(operands[2])?;
                Some(Instruction::Sub { dst, src1, src2 })
            } else {
                None
            }
        }
        "mul" => {
            if operands.len() >= 3 {
                let dst = parse_register(operands[0])?;
                let src1 = parse_register(operands[1])?;
                let src2 = parse_register(operands[2])?;
                Some(Instruction::Mul { dst, src1, src2 })
            } else {
                None
            }
        }
        "div" => {
            if operands.len() >= 3 {
                let dst = parse_register(operands[0])?;
                let src1 = parse_register(operands[1])?;
                let src2 = parse_register(operands[2])?;
                Some(Instruction::Div { dst, src1, src2 })
            } else {
                None
            }
        }
        "sig" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Sigmoid { dst, src })
            } else {
                None
            }
        }
        "csig" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::ComplementSigmoid { dst, src })
            } else {
                None
            }
        }
        "silu" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::SiLU { dst, src })
            } else {
                None
            }
        }
        "relu" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::ReLU { dst, src })
            } else {
                None
            }
        }
        "gelu" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::GeLU { dst, src })
            } else {
                None
            }
        }
        "tanh" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Tanh { dst, src })
            } else {
                None
            }
        }
        "neg" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Neg { dst, src })
            } else {
                None
            }
        }
        "reciprocal" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Reciprocal { dst, src })
            } else {
                None
            }
        }
        "rsqrt" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Rsqrt { dst, src })
            } else {
                None
            }
        }
        "sqrt" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Sqrt { dst, src })
            } else {
                None
            }
        }
        "exp" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Exp { dst, src })
            } else {
                None
            }
        }
        "log" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Log { dst, src })
            } else {
                None
            }
        }
        "log1p" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Log1p { dst, src })
            } else {
                None
            }
        }
        "abs" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Abs { dst, src })
            } else {
                None
            }
        }
        "sin" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Sin { dst, src })
            } else {
                None
            }
        }
        "cos" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Cos { dst, src })
            } else {
                None
            }
        }
        "floor" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Floor { dst, src })
            } else {
                None
            }
        }
        "ceil" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Ceil { dst, src })
            } else {
                None
            }
        }
        "round" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Round { dst, src })
            } else {
                None
            }
        }
        "sign" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Sign { dst, src })
            } else {
                None
            }
        }
        "residual" => {
            if operands.len() >= 3 {
                let dst = parse_register(operands[0])?;
                let src1 = parse_register(operands[1])?;
                let src2 = parse_register(operands[2])?;
                Some(Instruction::Residual { dst, src1, src2 })
            } else {
                None
            }
        }
        "softmax" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Softmax { dst, src })
            } else {
                None
            }
        }
        "reduce_sum" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::ReduceSum { dst, src })
            } else {
                None
            }
        }
        "reduce_mean" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::ReduceMean { dst, src })
            } else {
                None
            }
        }
        "reduce_max" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::ReduceMax { dst, src })
            } else {
                None
            }
        }
        "reduce_var" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::ReduceVar { dst, src })
            } else {
                None
            }
        }
        "layernorm" => {
            if operands.len() >= 3 {
                let dst = parse_register(operands[0])?;
                let input = parse_register(operands[1])?;
                let weight = parse_register(operands[2])?;
                let bias = if operands.len() >= 4 {
                    parse_register(operands[3])
                } else {
                    None
                };
                Some(Instruction::LayerNorm { dst, input, weight, bias })
            } else {
                None
            }
        }
        "rms_clear" => {
            Some(Instruction::RmsClear)
        }
        "rms_accumulate" => {
            if operands.len() >= 1 {
                let src = parse_register(operands[0])?;
                Some(Instruction::RmsAccumulate { src })
            } else {
                None
            }
        }
        "rms_finish_accumulate" => {
            if operands.len() >= 1 {
                let total_length = parse_memref(operands[0]);
                Some(Instruction::RmsFinishAccumulate { total_length })
            } else {
                None
            }
        }
        "rms_norm" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::RmsNorm { dst, src })
            } else {
                None
            }
        }
        "norm" => {
            if operands.len() >= 2 {
                let dst = parse_register(operands[0])?;
                let src = parse_register(operands[1])?;
                Some(Instruction::Norm { dst, src })
            } else {
                None
            }
        }
        "tmatmul_import" => {
            if operands.len() >= 1 {
                let src = parse_register(operands[0])?;
                Some(Instruction::TMatmulImport { src })
            } else {
                None
            }
        }
        "tmatmul_go" => {
            if operands.len() >= 1 {
                let weights = parse_memref(operands[0]);
                Some(Instruction::TMatmulGo { weights })
            } else {
                None
            }
        }
        "tmatmul_export" => {
            if operands.len() >= 1 {
                let dst = parse_register(operands[0])?;
                Some(Instruction::TMatmulExport { dst })
            } else {
                None
            }
        }
        "nop" | "stall" => {
            Some(Instruction::Nop)
        }
        _ => None, // Unknown opcode, skip
    }
}

/// Parse assembly text into a list of instructions
pub fn parse_assembly(asm: &str) -> Vec<Instruction> {
    asm.lines().filter_map(parse_line).collect()
}

/// Extract PARAM bindings from assembly comments
/// Returns mapping of PARAM_N -> parameter name (for debugging)
pub fn extract_bindings(asm: &str) -> HashMap<usize, String> {
    let mut bindings = HashMap::new();
    for line in asm.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("; BIND PARAM_") {
            // Format: "; BIND PARAM_0 output_ptr"
            let rest = &trimmed["; BIND PARAM_".len()..];
            let parts: Vec<&str> = rest.splitn(2, ' ').collect();
            if parts.len() == 2 {
                if let Ok(idx) = parts[0].parse::<usize>() {
                    bindings.insert(idx, parts[1].to_string());
                }
            }
        }
    }
    bindings
}

/// Execution mode determined by which global operations are present
#[derive(Debug, PartialEq)]
enum ExecutionMode {
    /// Pure element-wise: single pass
    ElementWise,
    /// Contains norm: two-pass (compute RMS, then normalize)
    Norm,
    /// Contains RMS operations: multi-pass (accumulate, finish, normalize)
    RmsNorm,
    /// Contains reduce: two-pass (accumulate, then store result)
    Reduce,
    /// Contains softmax: three-pass (max, sum_exp, normalize)
    Softmax,
    /// Contains layernorm: three-pass (mean, variance, normalize+scale)
    LayerNorm,
}

/// Detect the required execution mode based on instruction types
fn detect_execution_mode(instructions: &[Instruction]) -> ExecutionMode {
    for inst in instructions {
        match inst {
            Instruction::Softmax { .. } => return ExecutionMode::Softmax,
            Instruction::LayerNorm { .. } => return ExecutionMode::LayerNorm,
            Instruction::RmsNorm { .. } | Instruction::RmsClear |
            Instruction::RmsAccumulate { .. } | Instruction::RmsFinishAccumulate { .. } => {
                return ExecutionMode::RmsNorm;
            }
            Instruction::ReduceSum { .. } | Instruction::ReduceMean { .. } |
            Instruction::ReduceMax { .. } | Instruction::ReduceVar { .. } => {
                return ExecutionMode::Reduce;
            }
            Instruction::Norm { .. } => return ExecutionMode::Norm,
            _ => {}
        }
    }
    ExecutionMode::ElementWise
}

/// Count the number of unique PARAM references to determine how many pointer params to read
pub fn count_param_refs(instructions: &[Instruction]) -> usize {
    let mut max_param = 0usize;
    let mut found_any = false;
    for inst in instructions {
        match inst {
            Instruction::LoadVector { mem, .. } => {
                if let MemRef::Param(idx) = mem {
                    found_any = true;
                    if *idx >= max_param { max_param = *idx + 1; }
                }
            }
            Instruction::StoreVector { mem, .. } => {
                if let MemRef::Param(idx) = mem {
                    found_any = true;
                    if *idx >= max_param { max_param = *idx + 1; }
                }
            }
            Instruction::TMatmulGo { weights } => {
                if let MemRef::Param(idx) = weights {
                    found_any = true;
                    if *idx >= max_param { max_param = *idx + 1; }
                }
            }
            _ => {}
        }
    }
    if found_any { max_param } else { 0 }
}

/// Execute compiled TMatmul assembly on host memory.
///
/// # Arguments
/// * `asm` - The assembly text
/// * `kernel_params` - Pointer to array of pointers to kernel parameter values
/// * `grid_dims` - (grid_x, grid_y, grid_z)
/// * `block_dims` - (block_x, block_y, block_z)
///
/// # Safety
/// Caller must ensure kernel_params points to valid memory with the correct number of parameters.
pub unsafe fn execute_assembly(
    asm: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dims: (u32, u32, u32),
    block_dims: (u32, u32, u32),
) -> Result<(), String> {
    // Parse instructions
    let instructions = parse_assembly(asm);
    if instructions.is_empty() {
        return Err("No executable instructions found".to_string());
    }

    // Calculate total elements
    let total_elements = (grid_dims.0 as u64) * (block_dims.0 as u64)
        * (grid_dims.1 as u64).max(1) * (block_dims.1 as u64).max(1)
        * (grid_dims.2 as u64).max(1) * (block_dims.2 as u64).max(1);

    if total_elements == 0 {
        return Ok(()); // Nothing to do
    }

    // Determine how many params we need (cap at 16 to avoid reading past kernel_params array)
    let num_params_needed = count_param_refs(&instructions).min(16);
    if num_params_needed == 0 {
        return Err("No PARAM references found in assembly".to_string());
    }

    if kernel_params.is_null() {
        return Err("kernel_params is null".to_string());
    }

    // Resolve memory bindings: read kernel_params to get host pointers
    // ONLY use pointers that are verified in VIRTUAL_ALLOC_MAP to prevent SIGSEGV.
    let mut param_ptrs: Vec<*mut u8> = Vec::new();
    let mut param_sizes: Vec<usize> = Vec::new();

    for i in 0..num_params_needed {
        let param_slot = kernel_params.add(i);
        // Check that param_slot points to mapped memory before dereferencing
        if !is_readable(param_slot as *const u8) {
            param_ptrs.push(std::ptr::null_mut());
            param_sizes.push(0);
            continue;
        }
        let param_addr = *param_slot;
        if param_addr.is_null() {
            param_ptrs.push(std::ptr::null_mut());
            param_sizes.push(0);
            continue;
        }
        // Check that param_addr points to mapped memory before reading
        if !is_readable(param_addr as *const u8) {
            param_ptrs.push(std::ptr::null_mut());
            param_sizes.push(0);
            continue;
        }
        // Read the pointer value from the parameter slot
        let ptr_value = (param_addr as *const u64).read_unaligned();

        // Verify this pointer is in our VIRTUAL_ALLOC_MAP.
        // This is the only safe way to know it's a valid tensor pointer vs a scalar.
        let alloc_size = super::memory::get_alloc_size(ptr_value as usize);
        if let Some(size) = alloc_size {
            param_ptrs.push(ptr_value as *mut u8);
            param_sizes.push(size);
        } else {
            // Not a tracked allocation - this is a scalar param (numel, etc.), not a pointer
            param_ptrs.push(std::ptr::null_mut());
            param_sizes.push(0);
        }
    }

    // Check we have at least 2 valid params (need input + output at minimum)
    let valid_params = param_ptrs.iter()
        .filter(|ptr| !ptr.is_null())
        .count();

    if valid_params < 2 {
        return Err(format!("Insufficient valid pointer params: found {} (needed at least 2 for input+output, {} PARAM refs total)", valid_params, num_params_needed));
    }

    // Determine actual element count: min of total_elements and smallest known allocation
    let min_alloc_elements = param_sizes.iter()
        .filter(|s| **s > 0)
        .map(|s| s / 4) // f32 = 4 bytes
        .min()
        .unwrap_or(total_elements as usize);

    let actual_elements = (total_elements as usize).min(min_alloc_elements);
    if actual_elements == 0 {
        return Err("Computed 0 elements to process".to_string());
    }

    // Cap at 64M elements for safety
    let actual_elements = actual_elements.min(64 * 1024 * 1024);

    let mode = detect_execution_mode(&instructions);
    eprintln!("[TMatmul Interpreter] Executing {} instructions on {} elements ({} valid params of {} needed, mode={:?})",
             instructions.len(), actual_elements, valid_params, num_params_needed, mode);

    match mode {
        ExecutionMode::ElementWise => execute_single_pass(&instructions, &param_ptrs, &param_sizes, actual_elements),
        ExecutionMode::Norm => execute_norm_pass(&instructions, &param_ptrs, &param_sizes, actual_elements),
        ExecutionMode::RmsNorm => execute_rms_pass(&instructions, &param_ptrs, &param_sizes, actual_elements),
        ExecutionMode::Reduce => execute_reduce_pass(&instructions, &param_ptrs, &param_sizes, actual_elements),
        ExecutionMode::Softmax => execute_softmax_pass(&instructions, &param_ptrs, &param_sizes, actual_elements),
        ExecutionMode::LayerNorm => execute_layernorm_pass(&instructions, &param_ptrs, &param_sizes, actual_elements),
    }
}

/// Execute TMatmul assembly with pre-resolved tensor pointers.
/// This is the "correct compiler" path: the caller extracts tensor pointers
/// from the complex kernel param structures and passes them directly.
///
/// # Arguments
/// * `asm` - The assembly text (e.g., "ldv v0,PARAM_0\nmul v1,v0,v0\nsv v1,PARAM_1")
/// * `ptrs` - Direct tensor pointers (PARAM_0 → ptrs[0], etc.)
/// * `sizes` - Allocation sizes for each pointer
/// * `num_elements` - Number of f32 elements to process
pub unsafe fn execute_with_direct_ptrs(
    asm: &str,
    ptrs: &[*mut u8],
    sizes: &[usize],
    num_elements: usize,
) -> Result<(), String> {
    let instructions = parse_assembly(asm);
    if instructions.is_empty() {
        return Err("No executable instructions in generated assembly".to_string());
    }

    if num_elements == 0 {
        return Ok(());
    }

    // Cap at 64M elements for safety
    let actual_elements = num_elements.min(64 * 1024 * 1024);

    let mode = detect_execution_mode(&instructions);
    eprintln!("[TMatmul Compiler] Executing {} instructions on {} elements ({} params, mode={:?})",
             instructions.len(), actual_elements, ptrs.len(), mode);

    match mode {
        ExecutionMode::ElementWise => execute_single_pass(&instructions, ptrs, sizes, actual_elements),
        ExecutionMode::Norm => execute_norm_pass(&instructions, ptrs, sizes, actual_elements),
        ExecutionMode::RmsNorm => execute_rms_pass(&instructions, ptrs, sizes, actual_elements),
        ExecutionMode::Reduce => execute_reduce_pass(&instructions, ptrs, sizes, actual_elements),
        ExecutionMode::Softmax => execute_softmax_pass(&instructions, ptrs, sizes, actual_elements),
        ExecutionMode::LayerNorm => execute_layernorm_pass(&instructions, ptrs, sizes, actual_elements),
    }
}

/// Single-pass execution with fast-path for common patterns.
/// Instead of element-by-element instruction dispatch, we detect common patterns
/// and execute them as bulk tensor operations.
unsafe fn execute_single_pass(
    instructions: &[Instruction],
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    num_elements: usize,
) -> Result<(), String> {
    // Try fast path first - pattern match common instruction sequences
    if let Some(result) = try_fast_path(instructions, param_ptrs, param_sizes, num_elements) {
        return result;
    }

    // Fall back to slow element-by-element execution
    eprintln!("[TMatmul Interpreter] Using slow path ({} elements, {} instructions)", num_elements, instructions.len());
    let mut registers = [0.0f32; 8];
    let mut spill_memory: HashMap<String, f32> = HashMap::new();

    for elem in 0..num_elements {
        let byte_offset = elem * 4;
        for inst in instructions {
            execute_instruction(inst, &mut registers, &mut spill_memory,
                              param_ptrs, param_sizes, byte_offset, 0.0);
        }
    }

    Ok(())
}

/// Number of threads for parallel execution
const NUM_THREADS: usize = 8;
/// Minimum elements per thread to justify parallelism
const MIN_ELEMENTS_PER_THREAD: usize = 100_000;

/// Fast parallel unary operation using thread pool
#[inline]
unsafe fn fast_unary_parallel<F: Fn(f32) -> f32 + Sync>(
    src: *const f32,
    dst: *mut f32,
    n: usize,
    op: F,
) {
    if n < MIN_ELEMENTS_PER_THREAD * 2 {
        // Small tensor: single-threaded is faster
        fast_unary_single(src, dst, n, &op);
    } else {
        // Large tensor: use multiple threads
        // Convert pointers to usize for Send (we guarantee memory safety)
        let chunk_size = (n + NUM_THREADS - 1) / NUM_THREADS;
        let src_base = src as usize;
        let dst_base = dst as usize;
        std::thread::scope(|s| {
            for t in 0..NUM_THREADS {
                let start = t * chunk_size;
                let end = (start + chunk_size).min(n);
                if start >= n { break; }

                let src_off = src_base + start * 4;
                let dst_off = dst_base + start * 4;
                let count = end - start;
                let op_ref = &op;
                s.spawn(move || {
                    fast_unary_single(src_off as *const f32, dst_off as *mut f32, count, op_ref);
                });
            }
        });
    }
}

/// Fast single-threaded unary operation with loop unrolling
#[inline]
unsafe fn fast_unary_single<F: Fn(f32) -> f32>(
    src: *const f32,
    dst: *mut f32,
    n: usize,
    op: &F,
) {
    let mut i = 0;
    // Unroll by 4 for better pipelining
    while i + 4 <= n {
        let v0 = *src.add(i);
        let v1 = *src.add(i + 1);
        let v2 = *src.add(i + 2);
        let v3 = *src.add(i + 3);
        *dst.add(i) = op(v0);
        *dst.add(i + 1) = op(v1);
        *dst.add(i + 2) = op(v2);
        *dst.add(i + 3) = op(v3);
        i += 4;
    }
    // Handle remainder
    while i < n {
        *dst.add(i) = op(*src.add(i));
        i += 1;
    }
}

/// Fast parallel binary operation
#[inline]
unsafe fn fast_binary_parallel<F: Fn(f32, f32) -> f32 + Sync>(
    a: *const f32,
    b: *const f32,
    dst: *mut f32,
    n: usize,
    op: F,
) {
    if n < MIN_ELEMENTS_PER_THREAD * 2 {
        fast_binary_single(a, b, dst, n, &op);
    } else {
        let chunk_size = (n + NUM_THREADS - 1) / NUM_THREADS;
        let a_base = a as usize;
        let b_base = b as usize;
        let dst_base = dst as usize;
        std::thread::scope(|s| {
            for t in 0..NUM_THREADS {
                let start = t * chunk_size;
                let end = (start + chunk_size).min(n);
                if start >= n { break; }

                let a_off = a_base + start * 4;
                let b_off = b_base + start * 4;
                let dst_off = dst_base + start * 4;
                let count = end - start;
                let op_ref = &op;
                s.spawn(move || {
                    fast_binary_single(a_off as *const f32, b_off as *const f32, dst_off as *mut f32, count, op_ref);
                });
            }
        });
    }
}

/// Fast single-threaded binary operation with loop unrolling
#[inline]
unsafe fn fast_binary_single<F: Fn(f32, f32) -> f32>(
    a: *const f32,
    b: *const f32,
    dst: *mut f32,
    n: usize,
    op: &F,
) {
    let mut i = 0;
    while i + 4 <= n {
        *dst.add(i) = op(*a.add(i), *b.add(i));
        *dst.add(i + 1) = op(*a.add(i + 1), *b.add(i + 1));
        *dst.add(i + 2) = op(*a.add(i + 2), *b.add(i + 2));
        *dst.add(i + 3) = op(*a.add(i + 3), *b.add(i + 3));
        i += 4;
    }
    while i < n {
        *dst.add(i) = op(*a.add(i), *b.add(i));
        i += 1;
    }
}

/// Fast path for common instruction patterns. Returns None if pattern not recognized.
/// Uses multi-threading for large tensors.
#[inline]
unsafe fn try_fast_path(
    instructions: &[Instruction],
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    num_elements: usize,
) -> Option<Result<(), String>> {
    // Pattern: ldv v0,PARAM_0; sv v0,PARAM_1 (copy)
    if instructions.len() == 2 {
        if let (Instruction::LoadVector { dst: d0, mem: MemRef::Param(src_p) },
                Instruction::StoreVector { src: s1, mem: MemRef::Param(dst_p) }) = (&instructions[0], &instructions[1]) {
            if d0 == s1 && *src_p < param_ptrs.len() && *dst_p < param_ptrs.len() {
                let src = param_ptrs[*src_p];
                let dst = param_ptrs[*dst_p];
                if !src.is_null() && !dst.is_null() {
                    let bytes = num_elements * 4;
                    let src_ok = bytes <= param_sizes[*src_p];
                    let dst_ok = bytes <= param_sizes[*dst_p];
                    if src_ok && dst_ok {
                        std::ptr::copy_nonoverlapping(src, dst, bytes);
                        eprintln!("[TMatmul Fast] copy {} elements", num_elements);
                        return Some(Ok(()));
                    }
                }
            }
        }
    }

    // Pattern: ldv v0,PARAM_0; OP v1,v0; sv v1,PARAM_1 (unary op)
    if instructions.len() == 3 {
        if let (Instruction::LoadVector { dst: d0, mem: MemRef::Param(src_p) },
                Instruction::StoreVector { src: s2, mem: MemRef::Param(dst_p) }) = (&instructions[0], &instructions[2]) {
            if *src_p < param_ptrs.len() && *dst_p < param_ptrs.len() {
                let src_ptr = param_ptrs[*src_p];
                let dst_ptr = param_ptrs[*dst_p];
                if !src_ptr.is_null() && !dst_ptr.is_null() {
                    let bytes = num_elements * 4;
                    if bytes <= param_sizes[*src_p] && bytes <= param_sizes[*dst_p] {
                        // Use raw pointers for speed (no bounds checks)
                        let src = src_ptr as *const f32;
                        let dst = dst_ptr as *mut f32;

                        match &instructions[1] {
                            Instruction::Sigmoid { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| 1.0 / (1.0 + (-x).exp()));
                                eprintln!("[TMatmul Fast] sigmoid {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::SiLU { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| x / (1.0 + (-x).exp()));
                                eprintln!("[TMatmul Fast] silu {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::ReLU { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| if x > 0.0 { x } else { 0.0 });
                                eprintln!("[TMatmul Fast] relu {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::GeLU { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| {
                                    let k: f32 = 0.7978845608;
                                    let inner = k * (x + 0.044715 * x * x * x);
                                    0.5 * x * (1.0 + inner.tanh())
                                });
                                eprintln!("[TMatmul Fast] gelu {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Tanh { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| x.tanh());
                                eprintln!("[TMatmul Fast] tanh {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Neg { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| -x);
                                eprintln!("[TMatmul Fast] neg {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Reciprocal { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| if x != 0.0 { 1.0 / x } else { f32::INFINITY });
                                eprintln!("[TMatmul Fast] reciprocal {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Rsqrt { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| if x > 0.0 { 1.0 / x.sqrt() } else { f32::INFINITY });
                                eprintln!("[TMatmul Fast] rsqrt {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Sqrt { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| x.sqrt());
                                eprintln!("[TMatmul Fast] sqrt {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Exp { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| x.exp());
                                eprintln!("[TMatmul Fast] exp {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Log { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| x.ln());
                                eprintln!("[TMatmul Fast] log {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Log1p { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| (1.0 + x).ln());
                                eprintln!("[TMatmul Fast] log1p {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Abs { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| x.abs());
                                eprintln!("[TMatmul Fast] abs {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Sin { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| x.sin());
                                eprintln!("[TMatmul Fast] sin {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Cos { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| x.cos());
                                eprintln!("[TMatmul Fast] cos {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Floor { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| x.floor());
                                eprintln!("[TMatmul Fast] floor {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Ceil { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| x.ceil());
                                eprintln!("[TMatmul Fast] ceil {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Round { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| x.round());
                                eprintln!("[TMatmul Fast] round {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Sign { dst: d1, src: s1 } if *d0 == *s1 && *d1 == *s2 => {
                                fast_unary_parallel(src, dst, num_elements, |x| {
                                    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
                                });
                                eprintln!("[TMatmul Fast] sign {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    // Pattern: ldv v0,PARAM_0; ldv v1,PARAM_1; OP v2,v0,v1; sv v2,PARAM_2 (binary op)
    if instructions.len() == 4 {
        if let (Instruction::LoadVector { dst: d0, mem: MemRef::Param(p0) },
                Instruction::LoadVector { dst: d1, mem: MemRef::Param(p1) },
                Instruction::StoreVector { src: s3, mem: MemRef::Param(p2) }) = (&instructions[0], &instructions[1], &instructions[3]) {
            if *p0 < param_ptrs.len() && *p1 < param_ptrs.len() && *p2 < param_ptrs.len() {
                let ptr0 = param_ptrs[*p0];
                let ptr1 = param_ptrs[*p1];
                let ptr2 = param_ptrs[*p2];
                if !ptr0.is_null() && !ptr1.is_null() && !ptr2.is_null() {
                    let bytes = num_elements * 4;
                    if bytes <= param_sizes[*p0] && bytes <= param_sizes[*p1] && bytes <= param_sizes[*p2] {
                        let a = ptr0 as *const f32;
                        let b = ptr1 as *const f32;
                        let out = ptr2 as *mut f32;

                        match &instructions[2] {
                            Instruction::Add { dst: d2, src1, src2 } if *d0 == *src1 && *d1 == *src2 && *d2 == *s3 => {
                                fast_binary_parallel(a, b, out, num_elements, |x, y| x + y);
                                eprintln!("[TMatmul Fast] add {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Sub { dst: d2, src1, src2 } if *d0 == *src1 && *d1 == *src2 && *d2 == *s3 => {
                                fast_binary_parallel(a, b, out, num_elements, |x, y| x - y);
                                eprintln!("[TMatmul Fast] sub {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Mul { dst: d2, src1, src2 } if *d0 == *src1 && *d1 == *src2 && *d2 == *s3 => {
                                fast_binary_parallel(a, b, out, num_elements, |x, y| x * y);
                                eprintln!("[TMatmul Fast] mul {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Div { dst: d2, src1, src2 } if *d0 == *src1 && *d1 == *src2 && *d2 == *s3 => {
                                fast_binary_parallel(a, b, out, num_elements, |x, y| if y != 0.0 { x / y } else { 0.0 });
                                eprintln!("[TMatmul Fast] div {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            Instruction::Residual { dst: d2, src1, src2 } if *d0 == *src1 && *d1 == *src2 && *d2 == *s3 => {
                                fast_binary_parallel(a, b, out, num_elements, |x, y| x + y);
                                eprintln!("[TMatmul Fast] residual {} elements", num_elements);
                                return Some(Ok(()));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    None // Pattern not recognized, use slow path
}

/// Two-pass execution for norm: first accumulate RMS, then normalize
unsafe fn execute_norm_pass(
    instructions: &[Instruction],
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    num_elements: usize,
) -> Result<(), String> {
    // Pass 1: Find the norm source and compute RMS
    // We need to identify which PARAM the norm source loads from
    let mut norm_src_param: Option<usize> = None;
    // Trace through instructions to find what the norm's source register was loaded from
    let mut reg_source: [Option<usize>; 8] = [None; 8]; // Which PARAM each register was last loaded from

    for inst in instructions {
        match inst {
            Instruction::LoadVector { dst, mem } => {
                if let MemRef::Param(idx) = mem {
                    reg_source[*dst as usize] = Some(*idx);
                }
            }
            Instruction::Norm { src, .. } => {
                norm_src_param = reg_source[*src as usize];
            }
            _ => {}
        }
    }

    // Compute RMS value over the source tensor
    let rms_value = if let Some(param_idx) = norm_src_param {
        let ptr = param_ptrs[param_idx];
        if ptr.is_null() {
            return Err(format!("Norm source PARAM_{} is null", param_idx));
        }
        let max_elems = param_sizes[param_idx] / 4;
        let count = num_elements.min(max_elems);
        let slice = std::slice::from_raw_parts(ptr as *const f32, count);

        let mut sum_sq: f64 = 0.0;
        for &val in slice {
            sum_sq += (val as f64) * (val as f64);
        }
        let mean_sq = sum_sq / count as f64;
        let rms = (mean_sq + 1e-6_f64).sqrt();
        rms as f32
    } else {
        1.0f32 // Fallback if we can't determine norm source
    };

    eprintln!("[TMatmul Interpreter] Norm RMS value: {}", rms_value);

    // Pass 2: Execute all instructions with the computed RMS value
    let mut registers = [0.0f32; 8];
    let mut spill_memory: HashMap<String, f32> = HashMap::new();

    for elem in 0..num_elements {
        let byte_offset = elem * 4;

        for inst in instructions {
            execute_instruction(
                inst,
                &mut registers,
                &mut spill_memory,
                param_ptrs,
                param_sizes,
                byte_offset,
                rms_value,
            );
        }
    }

    Ok(())
}

/// Multi-pass execution for RMS norm:
/// Pass 1: Accumulate sum of squares over the source tensor
/// Pass 2: Compute RMS and apply normalization element-by-element
unsafe fn execute_rms_pass(
    instructions: &[Instruction],
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    num_elements: usize,
) -> Result<(), String> {
    // Find which PARAM is the source for rms_accumulate or rms_norm
    let mut reg_source: [Option<usize>; 8] = [None; 8];
    let mut rms_src_param: Option<usize> = None;

    for inst in instructions {
        match inst {
            Instruction::LoadVector { dst, mem } => {
                if let MemRef::Param(idx) = mem {
                    reg_source[*dst as usize] = Some(*idx);
                }
            }
            Instruction::RmsAccumulate { src } | Instruction::RmsNorm { src, .. } => {
                if rms_src_param.is_none() {
                    rms_src_param = reg_source[*src as usize];
                }
            }
            _ => {}
        }
    }

    // Compute RMS value
    let rms_value = if let Some(param_idx) = rms_src_param {
        if param_idx >= param_ptrs.len() || param_ptrs[param_idx].is_null() {
            return Err(format!("RMS source PARAM_{} is null or out of range", param_idx));
        }
        let ptr = param_ptrs[param_idx];
        let max_elems = param_sizes[param_idx] / 4;
        let count = num_elements.min(max_elems);
        let slice = std::slice::from_raw_parts(ptr as *const f32, count);

        let mut sum_sq: f64 = 0.0;
        for &val in slice {
            sum_sq += (val as f64) * (val as f64);
        }
        let mean_sq = sum_sq / count as f64;
        (mean_sq + 1e-6_f64).sqrt() as f32
    } else {
        1.0f32
    };

    eprintln!("[TMatmul Interpreter] RMS value: {}", rms_value);

    // Execute all instructions with the RMS value
    let mut registers = [0.0f32; 8];
    let mut spill_memory: HashMap<String, f32> = HashMap::new();

    for elem in 0..num_elements {
        let byte_offset = elem * 4;
        for inst in instructions {
            // Skip RMS control instructions in the execution pass
            match inst {
                Instruction::RmsClear | Instruction::RmsAccumulate { .. } |
                Instruction::RmsFinishAccumulate { .. } => continue,
                _ => {}
            }
            execute_instruction(inst, &mut registers, &mut spill_memory,
                              param_ptrs, param_sizes, byte_offset, rms_value);
        }
    }

    Ok(())
}

/// Multi-pass execution for reduce operations:
/// Pass 1: Read all source elements and accumulate the statistic
/// Pass 2: Write the reduced scalar result to the output
unsafe fn execute_reduce_pass(
    instructions: &[Instruction],
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    num_elements: usize,
) -> Result<(), String> {
    // Trace to find source and destination PARAMs
    let mut reg_source: [Option<usize>; 8] = [None; 8];
    let mut reduce_src_param: Option<usize> = None;
    let mut store_dst_param: Option<usize> = None;
    let mut reduce_type: Option<&str> = None;

    for inst in instructions {
        match inst {
            Instruction::LoadVector { dst, mem } => {
                if let MemRef::Param(idx) = mem {
                    reg_source[*dst as usize] = Some(*idx);
                }
            }
            Instruction::ReduceSum { src, .. } => {
                reduce_src_param = reg_source[*src as usize];
                reduce_type = Some("sum");
            }
            Instruction::ReduceMean { src, .. } => {
                reduce_src_param = reg_source[*src as usize];
                reduce_type = Some("mean");
            }
            Instruction::ReduceMax { src, .. } => {
                reduce_src_param = reg_source[*src as usize];
                reduce_type = Some("max");
            }
            Instruction::ReduceVar { src, .. } => {
                reduce_src_param = reg_source[*src as usize];
                reduce_type = Some("var");
            }
            Instruction::StoreVector { mem, .. } => {
                if let MemRef::Param(idx) = mem {
                    store_dst_param = Some(*idx);
                }
            }
            _ => {}
        }
    }

    let src_param = reduce_src_param.ok_or("Cannot determine reduce source PARAM")?;
    let dst_param = store_dst_param.ok_or("Cannot determine reduce destination PARAM")?;
    let r_type = reduce_type.ok_or("Cannot determine reduce type")?;

    if src_param >= param_ptrs.len() || param_ptrs[src_param].is_null() {
        return Err(format!("Reduce source PARAM_{} is null", src_param));
    }
    if dst_param >= param_ptrs.len() || param_ptrs[dst_param].is_null() {
        return Err(format!("Reduce destination PARAM_{} is null", dst_param));
    }

    let src_ptr = param_ptrs[src_param];
    let dst_ptr = param_ptrs[dst_param];
    let max_src_elems = param_sizes[src_param] / 4;
    let count = num_elements.min(max_src_elems);
    let src_slice = std::slice::from_raw_parts(src_ptr as *const f32, count);

    // Compute the reduction
    let result: f32 = match r_type {
        "sum" => {
            let mut acc: f64 = 0.0;
            for &v in src_slice { acc += v as f64; }
            acc as f32
        }
        "mean" => {
            let mut acc: f64 = 0.0;
            for &v in src_slice { acc += v as f64; }
            (acc / count as f64) as f32
        }
        "max" => {
            let mut m = f32::NEG_INFINITY;
            for &v in src_slice { if v > m { m = v; } }
            m
        }
        "var" => {
            let mut sum: f64 = 0.0;
            let mut sum_sq: f64 = 0.0;
            for &v in src_slice {
                let vd = v as f64;
                sum += vd;
                sum_sq += vd * vd;
            }
            let mean = sum / count as f64;
            let variance = (sum_sq / count as f64) - mean * mean;
            // Bessel's correction for unbiased
            let unbiased = if count > 1 { variance * (count as f64) / (count as f64 - 1.0) } else { variance };
            unbiased as f32
        }
        _ => 0.0,
    };

    // Write the scalar result to the output (first element)
    let dst_max_elems = param_sizes[dst_param] / 4;
    if dst_max_elems >= 1 {
        let dst_f32 = dst_ptr as *mut f32;
        dst_f32.write_unaligned(result);
    }

    eprintln!("[TMatmul Interpreter] Reduce {} over {} elements = {}", r_type, count, result);
    Ok(())
}

/// Multi-pass execution for softmax:
/// Pass 1: Find max over all source elements
/// Pass 2: Compute sum of exp(x - max)
/// Pass 3: Write exp(x - max) / sum for each element
unsafe fn execute_softmax_pass(
    instructions: &[Instruction],
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    num_elements: usize,
) -> Result<(), String> {
    // Trace to find source and destination PARAMs
    let mut reg_source: [Option<usize>; 8] = [None; 8];
    let mut softmax_src_param: Option<usize> = None;
    let mut store_dst_param: Option<usize> = None;

    for inst in instructions {
        match inst {
            Instruction::LoadVector { dst, mem } => {
                if let MemRef::Param(idx) = mem {
                    reg_source[*dst as usize] = Some(*idx);
                }
            }
            Instruction::Softmax { src, .. } => {
                softmax_src_param = reg_source[*src as usize];
            }
            Instruction::StoreVector { mem, .. } => {
                if let MemRef::Param(idx) = mem {
                    store_dst_param = Some(*idx);
                }
            }
            _ => {}
        }
    }

    let src_param = softmax_src_param.ok_or("Cannot determine softmax source PARAM")?;
    let dst_param = store_dst_param.ok_or("Cannot determine softmax destination PARAM")?;

    if src_param >= param_ptrs.len() || param_ptrs[src_param].is_null() {
        return Err(format!("Softmax source PARAM_{} is null", src_param));
    }
    if dst_param >= param_ptrs.len() || param_ptrs[dst_param].is_null() {
        return Err(format!("Softmax destination PARAM_{} is null", dst_param));
    }

    let src_ptr = param_ptrs[src_param];
    let dst_ptr = param_ptrs[dst_param];
    let max_src_elems = param_sizes[src_param] / 4;
    let max_dst_elems = param_sizes[dst_param] / 4;
    let count = num_elements.min(max_src_elems).min(max_dst_elems);
    let src_slice = std::slice::from_raw_parts(src_ptr as *const f32, count);

    // Pass 1: Find max
    let mut max_val = f32::NEG_INFINITY;
    for &v in src_slice {
        if v > max_val { max_val = v; }
    }

    // Pass 2: Compute sum of exp(x - max)
    let mut sum_exp: f64 = 0.0;
    for &v in src_slice {
        sum_exp += ((v - max_val) as f64).exp();
    }

    // Pass 3: Write normalized values
    let dst_slice = std::slice::from_raw_parts_mut(dst_ptr as *mut f32, count);
    for i in 0..count {
        let exp_val = ((src_slice[i] - max_val) as f64).exp();
        dst_slice[i] = (exp_val / sum_exp) as f32;
    }

    eprintln!("[TMatmul Interpreter] Softmax over {} elements (max={}, sum_exp={})", count, max_val, sum_exp);
    Ok(())
}

/// Multi-pass execution for layernorm:
/// Pass 1: Compute mean
/// Pass 2: Compute variance
/// Pass 3: Normalize, scale by weight, add bias
unsafe fn execute_layernorm_pass(
    instructions: &[Instruction],
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    num_elements: usize,
) -> Result<(), String> {
    // Trace to find source, weight, bias, and destination PARAMs
    let mut reg_source: [Option<usize>; 8] = [None; 8];
    let mut input_param: Option<usize> = None;
    let mut weight_param: Option<usize> = None;
    let mut bias_param: Option<usize> = None;
    let mut store_dst_param: Option<usize> = None;

    for inst in instructions {
        match inst {
            Instruction::LoadVector { dst, mem } => {
                if let MemRef::Param(idx) = mem {
                    reg_source[*dst as usize] = Some(*idx);
                }
            }
            Instruction::LayerNorm { input, weight, bias, .. } => {
                input_param = reg_source[*input as usize];
                weight_param = reg_source[*weight as usize];
                if let Some(b) = bias {
                    bias_param = reg_source[*b as usize];
                }
            }
            Instruction::StoreVector { mem, .. } => {
                if let MemRef::Param(idx) = mem {
                    store_dst_param = Some(*idx);
                }
            }
            _ => {}
        }
    }

    let in_param = input_param.ok_or("Cannot determine layernorm input PARAM")?;
    let w_param = weight_param.ok_or("Cannot determine layernorm weight PARAM")?;
    let dst_param = store_dst_param.ok_or("Cannot determine layernorm destination PARAM")?;

    if in_param >= param_ptrs.len() || param_ptrs[in_param].is_null() {
        return Err(format!("LayerNorm input PARAM_{} is null", in_param));
    }
    if w_param >= param_ptrs.len() || param_ptrs[w_param].is_null() {
        return Err(format!("LayerNorm weight PARAM_{} is null", w_param));
    }
    if dst_param >= param_ptrs.len() || param_ptrs[dst_param].is_null() {
        return Err(format!("LayerNorm destination PARAM_{} is null", dst_param));
    }

    let in_ptr = param_ptrs[in_param];
    let w_ptr = param_ptrs[w_param];
    let dst_ptr = param_ptrs[dst_param];

    let max_in_elems = param_sizes[in_param] / 4;
    let max_w_elems = param_sizes[w_param] / 4;
    let max_dst_elems = param_sizes[dst_param] / 4;
    let count = num_elements.min(max_in_elems).min(max_dst_elems);

    let in_slice = std::slice::from_raw_parts(in_ptr as *const f32, count);
    // Weight may be shorter than input (normalized_shape)
    let w_count = count.min(max_w_elems);
    let w_slice = std::slice::from_raw_parts(w_ptr as *const f32, w_count);

    // Pass 1: Compute mean
    let mut sum: f64 = 0.0;
    for &v in in_slice.iter() {
        sum += v as f64;
    }
    let mean = sum / count as f64;

    // Pass 2: Compute variance
    let mut sum_sq_diff: f64 = 0.0;
    for &v in in_slice.iter() {
        let diff = (v as f64) - mean;
        sum_sq_diff += diff * diff;
    }
    let variance = sum_sq_diff / count as f64;
    let std_inv = 1.0 / (variance + 1e-5_f64).sqrt();

    // Bias pointer (optional)
    let bias_slice = if let Some(b_param) = bias_param {
        if b_param < param_ptrs.len() && !param_ptrs[b_param].is_null() {
            let b_ptr = param_ptrs[b_param];
            let b_count = (param_sizes[b_param] / 4).min(w_count);
            Some(std::slice::from_raw_parts(b_ptr as *const f32, b_count))
        } else {
            None
        }
    } else {
        None
    };

    // Pass 3: Normalize, scale, bias
    let dst_slice = std::slice::from_raw_parts_mut(dst_ptr as *mut f32, count);
    for i in 0..count {
        let normalized = ((in_slice[i] as f64) - mean) * std_inv;
        let w_idx = i % w_count; // Cycle weight if input is longer
        let scaled = normalized * (w_slice[w_idx] as f64);
        let result = if let Some(b) = &bias_slice {
            let b_idx = i % b.len();
            scaled + b[b_idx] as f64
        } else {
            scaled
        };
        dst_slice[i] = result as f32;
    }

    eprintln!("[TMatmul Interpreter] LayerNorm over {} elements (mean={:.6}, var={:.6})", count, mean, variance);
    Ok(())
}

/// Execute a single instruction at the given element offset
unsafe fn execute_instruction(
    inst: &Instruction,
    registers: &mut [f32; 8],
    spill_memory: &mut HashMap<String, f32>,
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    byte_offset: usize,
    rms_value: f32,
) {
    match inst {
        Instruction::LoadVector { dst, mem } => {
            let val = read_memory(mem, param_ptrs, param_sizes, byte_offset, spill_memory);
            registers[*dst as usize] = val;
        }
        Instruction::StoreVector { src, mem } => {
            let val = registers[*src as usize];
            write_memory(mem, param_ptrs, param_sizes, byte_offset, val, spill_memory);
        }
        Instruction::Add { dst, src1, src2 } => {
            registers[*dst as usize] = registers[*src1 as usize] + registers[*src2 as usize];
        }
        Instruction::Sub { dst, src1, src2 } => {
            registers[*dst as usize] = registers[*src1 as usize] - registers[*src2 as usize];
        }
        Instruction::Mul { dst, src1, src2 } => {
            registers[*dst as usize] = registers[*src1 as usize] * registers[*src2 as usize];
        }
        Instruction::Div { dst, src1, src2 } => {
            let divisor = registers[*src2 as usize];
            registers[*dst as usize] = if divisor != 0.0 {
                registers[*src1 as usize] / divisor
            } else {
                0.0 // Avoid division by zero
            };
        }
        Instruction::Sigmoid { dst, src } => {
            let x = registers[*src as usize];
            registers[*dst as usize] = 1.0 / (1.0 + (-x).exp());
        }
        Instruction::ComplementSigmoid { dst, src } => {
            let x = registers[*src as usize];
            let sig = 1.0 / (1.0 + (-x).exp());
            registers[*dst as usize] = 1.0 - sig;
        }
        Instruction::SiLU { dst, src } => {
            let x = registers[*src as usize];
            let sig = 1.0 / (1.0 + (-x).exp());
            registers[*dst as usize] = x * sig;
        }
        Instruction::ReLU { dst, src } => {
            let x = registers[*src as usize];
            registers[*dst as usize] = if x > 0.0 { x } else { 0.0 };
        }
        Instruction::GeLU { dst, src } => {
            let x = registers[*src as usize];
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let k: f32 = 0.7978845608; // sqrt(2/pi)
            let inner = k * (x + 0.044715 * x * x * x);
            registers[*dst as usize] = 0.5 * x * (1.0 + inner.tanh());
        }
        Instruction::Tanh { dst, src } => {
            let x = registers[*src as usize];
            registers[*dst as usize] = x.tanh();
        }
        Instruction::Neg { dst, src } => {
            registers[*dst as usize] = -registers[*src as usize];
        }
        Instruction::Reciprocal { dst, src } => {
            let x = registers[*src as usize];
            registers[*dst as usize] = if x != 0.0 { 1.0 / x } else { f32::INFINITY };
        }
        Instruction::Rsqrt { dst, src } => {
            let x = registers[*src as usize];
            registers[*dst as usize] = if x > 0.0 { 1.0 / x.sqrt() } else { f32::INFINITY };
        }
        Instruction::Sqrt { dst, src } => {
            registers[*dst as usize] = registers[*src as usize].sqrt();
        }
        Instruction::Exp { dst, src } => {
            registers[*dst as usize] = registers[*src as usize].exp();
        }
        Instruction::Log { dst, src } => {
            registers[*dst as usize] = registers[*src as usize].ln();
        }
        Instruction::Log1p { dst, src } => {
            let x = registers[*src as usize];
            registers[*dst as usize] = (1.0 + x).ln();
        }
        Instruction::Abs { dst, src } => {
            registers[*dst as usize] = registers[*src as usize].abs();
        }
        Instruction::Sin { dst, src } => {
            registers[*dst as usize] = registers[*src as usize].sin();
        }
        Instruction::Cos { dst, src } => {
            registers[*dst as usize] = registers[*src as usize].cos();
        }
        Instruction::Floor { dst, src } => {
            registers[*dst as usize] = registers[*src as usize].floor();
        }
        Instruction::Ceil { dst, src } => {
            registers[*dst as usize] = registers[*src as usize].ceil();
        }
        Instruction::Round { dst, src } => {
            registers[*dst as usize] = registers[*src as usize].round();
        }
        Instruction::Sign { dst, src } => {
            let x = registers[*src as usize];
            registers[*dst as usize] = if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
        }
        Instruction::Residual { dst, src1, src2 } => {
            // residual connection: output = x + f(x)
            registers[*dst as usize] = registers[*src1 as usize] + registers[*src2 as usize];
        }
        Instruction::Softmax { dst, src } => {
            // In single-pass mode, softmax can't be computed correctly
            // (needs global max and sum). This is handled by execute_softmax_pass.
            // Here we just copy the value as a fallback.
            registers[*dst as usize] = registers[*src as usize];
        }
        Instruction::ReduceSum { dst, .. } |
        Instruction::ReduceMean { dst, .. } |
        Instruction::ReduceMax { dst, .. } |
        Instruction::ReduceVar { dst, .. } => {
            // Reductions are handled by execute_reduce_pass, not per-element.
            // This is a no-op placeholder in per-element mode.
            let _ = dst;
        }
        Instruction::LayerNorm { dst, input, .. } => {
            // LayerNorm is handled by execute_layernorm_pass.
            // Placeholder: just copy input.
            registers[*dst as usize] = registers[*input as usize];
        }
        Instruction::RmsClear | Instruction::RmsAccumulate { .. } |
        Instruction::RmsFinishAccumulate { .. } => {
            // Handled by execute_rms_pass
        }
        Instruction::RmsNorm { dst, src } => {
            // Use pre-computed RMS value (same as Norm in per-element mode)
            let x = registers[*src as usize];
            registers[*dst as usize] = if rms_value != 0.0 { x / rms_value } else { x };
        }
        Instruction::Norm { dst, src } => {
            // Use pre-computed RMS value
            let x = registers[*src as usize];
            registers[*dst as usize] = if rms_value != 0.0 {
                x / rms_value
            } else {
                x
            };
        }
        Instruction::TMatmulImport { .. } |
        Instruction::TMatmulGo { .. } |
        Instruction::TMatmulExport { .. } => {
            // Matrix multiply not supported in scalar interpreter mode
            // These would need a full matrix-vector implementation
        }
        Instruction::Nop => {}
    }
}

/// Read a value from memory
unsafe fn read_memory(
    mem: &MemRef,
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    byte_offset: usize,
    spill_memory: &HashMap<String, f32>,
) -> f32 {
    match mem {
        MemRef::Param(idx) => {
            if *idx >= param_ptrs.len() {
                return 0.0;
            }
            let ptr = param_ptrs[*idx];
            if ptr.is_null() {
                return 0.0;
            }
            let size = param_sizes[*idx];
            if byte_offset + 4 > size {
                return 0.0; // Out of bounds
            }
            let addr = ptr.add(byte_offset) as *const f32;
            addr.read_unaligned()
        }
        MemRef::Spill(name) => {
            spill_memory.get(name).copied().unwrap_or(0.0)
        }
        MemRef::Const(val) => *val,
    }
}

/// Write a value to memory
unsafe fn write_memory(
    mem: &MemRef,
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    byte_offset: usize,
    value: f32,
    spill_memory: &mut HashMap<String, f32>,
) {
    match mem {
        MemRef::Param(idx) => {
            if *idx >= param_ptrs.len() {
                return;
            }
            let ptr = param_ptrs[*idx];
            if ptr.is_null() {
                return;
            }
            let size = param_sizes[*idx];
            if byte_offset + 4 > size {
                return; // Out of bounds
            }
            let addr = ptr.add(byte_offset) as *mut f32;
            addr.write_unaligned(value);
        }
        MemRef::Spill(name) => {
            spill_memory.insert(name.clone(), value);
        }
        MemRef::Const(_) => {
            // Cannot write to constants
        }
    }
}

// ============================================================================
// TMatmul GEMM Implementation
// ============================================================================

/// f16 to f32 conversion
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    let f = if exp == 0 {
        if mant == 0 { sign } else {
            // Denormalized - simplified handling
            sign | ((127 - 14) << 23) | (mant << 13)
        }
    } else if exp == 31 {
        sign | 0x7f800000 | (mant << 13) // inf/nan
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f)
}

/// f32 to f16 conversion
fn f32_to_f16(val: f32) -> u16 {
    let f = val.to_bits();
    let sign = ((f >> 16) & 0x8000) as u16;
    let exp = (((f >> 23) & 0xff) as i32) - 112;
    let mant = f & 0x7fffff;

    if exp <= 0 {
        sign // flush to zero
    } else if exp >= 31 {
        sign | 0x7c00 // inf
    } else {
        sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
    }
}

/// Read element from typed buffer as f32
unsafe fn read_typed_elem(ptr: *const u8, idx: usize, dtype: i32) -> f32 {
    match dtype {
        2 => f16_to_f32((ptr as *const u16).add(idx).read_unaligned()), // f16 (CUDA_R_16F)
        14 => bf16_to_f32((ptr as *const u16).add(idx).read_unaligned()), // bf16 (CUDA_R_16BF)
        0 => (ptr as *const f32).add(idx).read_unaligned(), // f32 (CUDA_R_32F)
        1 => (ptr as *const f64).add(idx).read_unaligned() as f32, // f64 (CUDA_R_64F)
        _ => {
            eprintln!("[TMatmul GEMM] WARNING: unknown dtype {} in read_typed_elem, treating as f32", dtype);
            (ptr as *const f32).add(idx).read_unaligned()
        }
    }
}

/// Convert bf16 (stored as u16) to f32: just shift left by 16 bits
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to bf16: take upper 16 bits (truncation, no rounding)
#[inline]
fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

/// Write f32 element to typed buffer
unsafe fn write_typed_elem(ptr: *mut u8, idx: usize, dtype: i32, val: f32) {
    match dtype {
        2 => (ptr as *mut u16).add(idx).write_unaligned(f32_to_f16(val)), // f16 (CUDA_R_16F)
        14 => (ptr as *mut u16).add(idx).write_unaligned(f32_to_bf16(val)), // bf16 (CUDA_R_16BF)
        0 => (ptr as *mut f32).add(idx).write_unaligned(val), // f32 (CUDA_R_32F)
        1 => (ptr as *mut f64).add(idx).write_unaligned(val as f64), // f64 (CUDA_R_64F)
        _ => {
            eprintln!("[TMatmul GEMM] WARNING: unknown dtype {} in write_typed_elem, treating as f32", dtype);
            (ptr as *mut f32).add(idx).write_unaligned(val);
        }
    }
}

/// Execute GEMM using TMatmul assembly generation and interpretation
/// C = alpha * op(A) * op(B) + beta * C
/// op(A) is m×k, op(B) is k×n, C is m×n (column-major)
pub unsafe fn execute_gemm_tmatmul(
    transa: bool,
    transb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a_ptr: *const u8,
    a_dtype: i32,
    lda: usize,
    b_ptr: *const u8,
    b_dtype: i32,
    ldb: usize,
    beta: f32,
    c_ptr: *mut u8,
    c_dtype: i32,
    ldc: usize,
) -> i32 {
    if a_ptr.is_null() || b_ptr.is_null() || c_ptr.is_null() {
        return -1;
    }
    if m == 0 || n == 0 || k == 0 {
        return 0;
    }

    eprintln!("[TMatmul GEMM] m={}, n={}, k={}, alpha={}, beta={}, transa={}, transb={}",
              m, n, k, alpha, beta, transa, transb);

    // Generate TMatmul assembly for this GEMM
    // For column-major: C[i,j] = sum_p A[i,p] * B[p,j]
    // With transposition handling

    let assembly = format!(r#"
; TMatmul GEMM: C = alpha * op(A) * op(B) + beta * C
; m={m}, n={n}, k={k}, alpha={alpha}, beta={beta}
; PARAM_0 = A, PARAM_1 = B, PARAM_2 = C
    tmatmul_gemm PARAM_0,PARAM_1,PARAM_2,{m},{n},{k},{alpha},{beta},{transa},{transb}
"#, m=m, n=n, k=k, alpha=alpha, beta=beta,
    transa=if transa { 1 } else { 0 },
    transb=if transb { 1 } else { 0 });

    eprintln!("[TMatmul GEMM] Generated assembly:\n{}", assembly);

    // For now, execute directly in Rust (the interpreter would need GEMM instruction support)
    // This is the TMatmul emulation of what the hardware would do

    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0f32;
            for p in 0..k {
                // Column-major indexing with transpose handling
                let a_idx = if transa { i * lda + p } else { p * lda + i };
                let b_idx = if transb { p * ldb + j } else { j * ldb + p };

                let a_val = read_typed_elem(a_ptr, a_idx, a_dtype);
                let b_val = read_typed_elem(b_ptr, b_idx, b_dtype);
                sum += a_val * b_val;
            }

            let c_idx = j * ldc + i;
            let c_old = if beta != 0.0 { read_typed_elem(c_ptr, c_idx, c_dtype) } else { 0.0 };
            let c_new = alpha * sum + beta * c_old;
            write_typed_elem(c_ptr, c_idx, c_dtype, c_new);
        }
    }

    eprintln!("[TMatmul GEMM] Completed: {} multiply-accumulate operations", m * n * k);
    0
}

/// FFI wrapper for GEMM execution - called from cublas_shim.c
#[no_mangle]
pub unsafe extern "C" fn hetgpu_tmatmul_gemm(
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a_ptr: *const std::ffi::c_void,
    a_dtype: i32,
    lda: i32,
    b_ptr: *const std::ffi::c_void,
    b_dtype: i32,
    ldb: i32,
    beta: f32,
    c_ptr: *mut std::ffi::c_void,
    c_dtype: i32,
    ldc: i32,
) -> i32 {
    let result = execute_gemm_tmatmul(
        transa != 0,
        transb != 0,
        m as usize,
        n as usize,
        k as usize,
        alpha,
        a_ptr as *const u8,
        a_dtype,
        lda as usize,
        b_ptr as *const u8,
        b_dtype,
        ldb as usize,
        beta,
        c_ptr as *mut u8,
        c_dtype,
        ldc as usize,
    );
    // Track the last GEMM output for ArgMax and dataflow
    if result == 0 {
        let c_elems = (m as usize) * (n as usize);
        let c_elem_bytes = if c_dtype == 14 { 2 } else { 4 }; // bf16=2, f32=4
        if let Ok(mut last) = LAST_GEMM_OUTPUT.lock() {
            *last = (c_ptr as usize, c_elems, c_dtype);
        }
        set_last_output(c_ptr as usize, c_elems * c_elem_bytes, c_elem_bytes);
    }
    result
}
