// TMatmul Assembly Interpreter
// Parses and executes TMatmul assembly instructions directly on host memory.
// Each register holds a single f32 value (scalar mode, d=1).
// The interpreter loops over all elements (total = grid*block) executing
// instructions element-by-element.

use std::collections::HashMap;

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

/// Check if the instruction list contains a norm operation (requires two-pass)
fn has_norm(instructions: &[Instruction]) -> bool {
    instructions.iter().any(|i| matches!(i, Instruction::Norm { .. }))
}

/// Count the number of unique PARAM references to determine how many pointer params to read
pub fn count_param_refs(instructions: &[Instruction]) -> usize {
    let mut max_param = 0usize;
    let mut found_any = false;
    for inst in instructions {
        let check = |mr: &MemRef| {
            if let MemRef::Param(idx) = mr {
                *idx
            } else {
                0
            }
        };
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

    // Determine how many params we need
    let num_params_needed = count_param_refs(&instructions);
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
        // Validate param_slot pointer before dereferencing
        let slot_addr = param_slot as usize;
        if slot_addr < 0x1000 || slot_addr > 0x7FFF_FFFF_FFFF {
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
        // Validate param_addr before reading from it
        let param_addr_val = param_addr as usize;
        if param_addr_val < 0x1000 || param_addr_val > 0x7FFF_FFFF_FFFF {
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

    // Check we have at least some valid params (non-null pointers)
    let valid_params = param_ptrs.iter()
        .filter(|ptr| !ptr.is_null())
        .count();

    if valid_params == 0 {
        return Err(format!("No valid pointer params found (needed {}, all null/scalar)", num_params_needed));
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

    eprintln!("[TMatmul Interpreter] Executing {} instructions on {} elements ({} valid params of {} needed)",
             instructions.len(), actual_elements, valid_params, num_params_needed);

    // Check if we need two-pass (norm)
    if has_norm(&instructions) {
        execute_two_pass(&instructions, &param_ptrs, &param_sizes, actual_elements)
    } else {
        execute_single_pass(&instructions, &param_ptrs, &param_sizes, actual_elements)
    }
}

/// Single-pass execution: process each element independently
unsafe fn execute_single_pass(
    instructions: &[Instruction],
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    num_elements: usize,
) -> Result<(), String> {
    let mut registers = [0.0f32; 8];
    let mut spill_memory: HashMap<String, f32> = HashMap::new();

    for elem in 0..num_elements {
        let byte_offset = elem * 4; // sizeof(f32)

        // Reset registers for each element? No - PTX semantics have per-thread state
        // but our assembly is designed to process one element at a time with all
        // state in registers.

        for inst in instructions {
            execute_instruction(
                inst,
                &mut registers,
                &mut spill_memory,
                param_ptrs,
                param_sizes,
                byte_offset,
                0.0, // rms_value not used in single-pass
            );
        }
    }

    Ok(())
}

/// Two-pass execution for norm: first accumulate RMS, then normalize
unsafe fn execute_two_pass(
    instructions: &[Instruction],
    param_ptrs: &[*mut u8],
    param_sizes: &[usize],
    num_elements: usize,
) -> Result<(), String> {
    // Pass 1: Find the norm source and compute RMS
    // We need to identify which PARAM the norm source loads from
    let mut norm_src_param: Option<usize> = None;
    let mut norm_src_reg: Option<u8> = None;

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
                norm_src_reg = Some(*src);
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
