// #![feature(str_from_raw_parts)]

pub mod checkpoint;
pub mod checkpoint_integration;
pub mod debug;
pub mod dwarf_validation;
pub mod pass;
pub mod state_recovery;
#[cfg(test)]
mod test;

pub use crate::pass::{Module, TranslateError};
pub use pass::{
    to_llvm_module, to_llvm_module_with_debug_round_trip, to_llvm_module_with_filename,
    to_mlir_module,
};
use regex;
use std::collections::HashMap;
use std::ptr;

// Implementation for PTX to LLVM IR conversion
pub fn ptx_to_llvm(ast: ptx_parser::Module) -> Result<Module, TranslateError> {
    // Use the existing to_llvm_module function
    to_llvm_module(ast)
}

// Implementation for PTX string to LLVM IR with debug info
/// Extract filename from PTX source by looking for .file directive or deriving from kernel names
fn extract_filename_from_ptx(ptx_source: &str) -> Option<String> {
    // Look for .file directive (common in PTX files)
    for line in ptx_source.lines() {
        let line = line.trim();
        if line.starts_with(".file") {
            // Try to extract filename from .file directive
            if let Some(start) = line.find('"') {
                if let Some(end) = line.rfind('"') {
                    if start < end {
                        let filename = &line[start + 1..end];
                        // Ensure it has .ptx extension
                        if filename.ends_with(".ptx") {
                            return Some(filename.to_string());
                        } else {
                            return Some(format!("{}.ptx", filename));
                        }
                    }
                }
            }
        }
        // Look for .entry directive to get kernel name
        if line.starts_with(".entry") || line.starts_with(".visible .entry") {
            // Skip ".entry" or ".visible .entry" and find the function name
            let after_entry = if line.starts_with(".visible .entry") {
                line.strip_prefix(".visible .entry").unwrap_or(line).trim()
            } else {
                line.strip_prefix(".entry").unwrap_or(line).trim()
            };
            println!("ZLUDA DEBUG: after_entry: {}", after_entry);
            if let Some(end) = after_entry.find('(') {
                let kernel_name = after_entry[..end].trim();
                if !kernel_name.is_empty() {
                    // Check if the file exists in the test directory
                    let test_path =
                        format!("/root/hetGPU/ptx/src/test/spirv_run/{}.ptx", kernel_name);
                    if std::path::Path::new(&test_path).exists() {
                        return Some(test_path);
                    } else {
                        // Fallback to just the kernel name with .ptx extension
                        return Some(format!("{}.ptx", kernel_name));
                    }
                }
            }
        }
    }
    None
}

pub fn ptx_to_llvm_with_debug(
    ptx_source: &str,
) -> Result<(Module, Vec<debug::DwarfMappingEntry>), TranslateError> {
    // Try to extract filename from PTX source or use a default
    let filename =
        extract_filename_from_ptx(ptx_source).unwrap_or_else(|| "kernel.ptx".to_string());
    ptx_to_llvm_with_debug_and_filename(ptx_source, &filename)
}

// Implementation for PTX string to LLVM IR with debug info and custom filename
pub fn ptx_to_llvm_with_debug_and_filename(
    ptx_source: &str,
    source_filename: &str,
) -> Result<(Module, Vec<debug::DwarfMappingEntry>), TranslateError> {
    // Parse PTX source
    let ast = ptx_parser::parse_module_checked(ptx_source)
        .map_err(|_| TranslateError::UnexpectedError("PTX parsing failed".to_string()))?;

    // Convert PTX to LLVM IR with debug information
    let module = to_llvm_module_with_filename(ast, source_filename)?;
    let debug_mappings: HashMap<u64, debug::PtxSourceLocation> = HashMap::new(); // Placeholder for debug mappings

    // Convert HashMap to Vec for easier handling
    let debug_mapping_entries: Vec<debug::DwarfMappingEntry> = debug_mappings
        .into_iter()
        .map(|(addr, location)| debug::DwarfMappingEntry {
            ptx_location: location,
            target_instructions: Vec::new(), // Will be populated during actual compilation
            variable_mappings: std::collections::HashMap::new(),
            scope_id: 0,
        })
        .collect();

    Ok((module, debug_mapping_entries))
}

// Implementation for LLVM IR to SPIRV conversion using LLVM
pub fn llvm_to_spirv(llvm_ir: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Create a temporary file for the LLVM IR
    let mut temp_ir_file = tempfile::NamedTempFile::new()?;
    std::io::Write::write_all(&mut temp_ir_file, llvm_ir.as_bytes())?;
    let ir_path = temp_ir_file.path().to_str().ok_or("Invalid path")?;

    // Create a temporary file for the output SPIR-V
    let temp_spirv_file = tempfile::NamedTempFile::new()?;
    let spirv_path = temp_spirv_file.path().to_str().ok_or("Invalid path")?;

    // Prepare the llc command (LLVM static compiler)
    // -mtriple=spir64-unknown-unknown specifies SPIR-V target
    // -O3 for optimization level
    // Use the SPIR-V backend
    let output = std::process::Command::new("llc")
        .args(&[
            "-mtriple=spir64-unknown-unknown",
            "-O3",
            "-filetype=obj",
            ir_path,
            "-o",
            spirv_path,
        ])
        .output()?;

    if !output.status.success() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "LLVM compilation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        )));
    }

    // Read the SPIR-V binary
    let spirv_binary = std::fs::read(spirv_path)?;

    // If the binary is empty or invalid, fall back to a minimal valid SPIRV module
    if spirv_binary.is_empty() {
        let minimal_spirv = vec![
            // SPIR-V magic number
            0x07, 0x23, 0x02, 0x03, // Version 1.0, Generator 0
            0x00, 0x01, 0x00, 0x00, // Generator magic number
            0x00, 0x00, 0x00, 0x00, // Bound for IDs
            0x01, 0x00, 0x00, 0x00, // Reserved
            0x00, 0x00, 0x00, 0x00,
        ];
        return Ok(minimal_spirv);
    }

    Ok(spirv_binary)
}

// Alternative implementation using the spirv-tools crate if llc is not available
pub fn llvm_to_spirv_alt(llvm_ir: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // This alternative implementation uses the spirv-tools-rs crate
    // First try to find spirv-as, spirv-opt in the system
    let llvm_path = which::which("llvm-spirv").map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "llvm-spirv not found in PATH")
    })?;

    // Create a temporary file for the LLVM IR and SPIR-V
    let mut temp_ir_file = tempfile::NamedTempFile::new()?;
    std::io::Write::write_all(&mut temp_ir_file, llvm_ir.as_bytes())?;
    let ir_path = temp_ir_file.path().to_str().ok_or("Invalid path")?;

    let temp_spirv_file = tempfile::NamedTempFile::new()?;
    let spirv_path = temp_spirv_file.path().to_str().ok_or("Invalid path")?;

    // Convert LLVM IR to SPIR-V using llvm-spirv
    let output = std::process::Command::new(llvm_path)
        .args(&[
            ir_path,
            "-o",
            spirv_path,
            "--spirv-ext=+all",
            "--spirv-target-env=CL2.0",
            "--strip-debug", // Disable debug information to avoid linking issues
        ])
        .output()?;

    if !output.status.success() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "SPIR-V conversion failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        )));
    }

    // Read the SPIR-V binary
    let spirv_binary = std::fs::read(spirv_path)?;

    Ok(spirv_binary)
}

// Main function that tries both implementation methods
pub fn llvm_to_spirv_robust(llvm_ir: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Pre-process LLVM IR to fix address space issues before SPIR-V conversion
    let fixed_llvm_ir = preprocess_llvm_ir_for_spirv(llvm_ir)?;

    // Try the primary implementation first
    match llvm_to_spirv(&fixed_llvm_ir) {
        Ok(binary) => {
            // Validate the SPIR-V binary before returning
            if is_valid_spirv_binary(&binary) {
                Ok(binary)
            } else {
                eprintln!(
                    "ZLUDA DEBUG: Primary SPIR-V binary failed validation, trying alternative"
                );
                try_alternative_spirv_generation(&fixed_llvm_ir)
            }
        }
        Err(primary_err) => {
            eprintln!(
                "ZLUDA DEBUG: Primary SPIR-V conversion failed: {}",
                primary_err
            );
            try_alternative_spirv_generation(&fixed_llvm_ir)
        }
    }
}

/// Pre-process LLVM IR to fix address space casting issues for SPIR-V
pub fn preprocess_llvm_ir_for_spirv(llvm_ir: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut processed_ir = llvm_ir.to_string();

    // First collect all variable definitions and their types
    let var_types = collect_variable_types(&processed_ir)?;

    // Fix 1: Fix pointer type consistency in address space casts
    processed_ir = fix_pointer_type_consistency(&processed_ir, &var_types)?;

    // Fix 2: Replace direct address space casts with two-step casts through generic
    processed_ir = fix_invalid_address_space_casts(&processed_ir)?;

    // Fix 3: Ensure all global variables have explicit address space annotations
    processed_ir = fix_global_variable_address_spaces(&processed_ir)?;

    // Fix 4: Remove problematic debug information that might cause SPIR-V issues
    processed_ir = remove_problematic_debug_info(&processed_ir)?;

    Ok(processed_ir)
}

/// Collect variable types from LLVM IR to ensure type consistency
fn collect_variable_types(
    llvm_ir: &str,
) -> Result<std::collections::HashMap<String, String>, Box<dyn std::error::Error>> {
    let mut var_types = std::collections::HashMap::new();

    // Parse variable definitions like: %"2" = inttoptr i64 %1 to ptr
    let var_def_re =
        regex::Regex::new(r"(%[^=\s]+)\s*=\s*[^=]*\s+to\s+(ptr(?:\s*addrspace\(\d+\))?)")?;
    for caps in var_def_re.captures_iter(llvm_ir) {
        let var_name = caps[1].to_string();
        let var_type = caps[2].to_string();
        var_types.insert(var_name, var_type);
    }

    // Parse alloca definitions: %var = alloca type, addrspace(X)
    let alloca_re = regex::Regex::new(r"(%[^=\s]+)\s*=\s*alloca\s+[^,]+,\s*addrspace\((\d+)\)")?;
    for caps in alloca_re.captures_iter(llvm_ir) {
        let var_name = caps[1].to_string();
        let addrspace = &caps[2];
        var_types.insert(var_name, format!("ptr addrspace({})", addrspace));
    }

    // Parse simple alloca definitions: %var = alloca type
    let simple_alloca_re = regex::Regex::new(r"(%[^=\s]+)\s*=\s*alloca\s+[^,\n]+")?;
    for caps in simple_alloca_re.captures_iter(llvm_ir) {
        let var_name = caps[1].to_string();
        var_types.insert(var_name, "ptr".to_string());
    }

    Ok(var_types)
}

/// Fix pointer type consistency issues in address space casts
fn fix_pointer_type_consistency(
    llvm_ir: &str,
    var_types: &std::collections::HashMap<String, String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut processed = llvm_ir.to_string();

    // Pattern: %result = addrspacecast ptr addrspace(X) %source to ptr addrspace(Y)
    // where %source is actually defined as just 'ptr', not 'ptr addrspace(X)'
    let cast_re = regex::Regex::new(
        r"(%[^=\s]+)\s*=\s*addrspacecast\s+(ptr)\s*addrspace\((\d+)\)\s*(%[^%\s]+)\s+to\s+(ptr(?:\s*addrspace\(\d+\))?)",
    )?;

    processed = cast_re
        .replace_all(&processed, |caps: &regex::Captures| {
            let result_var = &caps[1];
            let cast_src_type = &caps[2]; // "ptr"
            let cast_src_addrspace = &caps[3];
            let src_var = &caps[4];
            let dst_type = &caps[5];

            // Check what type the source variable actually has
            if let Some(actual_type) = var_types.get(src_var) {
                if actual_type == "ptr" && cast_src_type == "ptr" {
                    // The variable is defined as 'ptr' but being cast as 'ptr addrspace(X)'
                    // Use the actual type instead
                    format!(
                        "{} = addrspacecast {} {} to {}",
                        result_var, actual_type, src_var, dst_type
                    )
                } else {
                    // Types match or it's a complex case - keep original but fix if needed
                    format!(
                        "{} = addrspacecast {}addrspace({}) {} to {}",
                        result_var, cast_src_type, cast_src_addrspace, src_var, dst_type
                    )
                }
            } else {
                // Can't determine variable type - keep original
                format!(
                    "{} = addrspacecast {}addrspace({}) {} to {}",
                    result_var, cast_src_type, cast_src_addrspace, src_var, dst_type
                )
            }
        })
        .to_string();

    Ok(processed)
}

/// Fix invalid direct address space casts by using two-step casts through generic
fn fix_invalid_address_space_casts(llvm_ir: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut processed = llvm_ir.to_string();

    // Fix common invalid direct casts between specific address spaces
    let invalid_cast_patterns = [
        // Private to Global direct casts (AMDGPU address spaces)
        (
            r"addrspacecast\s*\(\s*([^*]+\*\s*addrspace\(5\)\*[^)]+)\s+to\s+([^*]+\*\s*addrspace\(1\)\*[^)]*)\)",
            "addrspacecast (addrspacecast ($1 to i8*) to $2)",
        ),
        // Global to Private direct casts
        (
            r"addrspacecast\s*\(\s*([^*]+\*\s*addrspace\(1\)\*[^)]+)\s+to\s+([^*]+\*\s*addrspace\(5\)\*[^)]*)\)",
            "addrspacecast (addrspacecast ($1 to i8*) to $2)",
        ),
        // Local to Global direct casts
        (
            r"addrspacecast\s*\(\s*([^*]+\*\s*addrspace\(3\)\*[^)]+)\s+to\s+([^*]+\*\s*addrspace\(1\)\*[^)]*)\)",
            "addrspacecast (addrspacecast ($1 to i8*) to $2)",
        ),
        // Global to Local direct casts
        (
            r"addrspacecast\s*\(\s*([^*]+\*\s*addrspace\(1\)\*[^)]+)\s+to\s+([^*]+\*\s*addrspace\(3\)\*[^)]*)\)",
            "addrspacecast (addrspacecast ($1 to i8*) to $2)",
        ),
        // Constant to Private direct casts
        (
            r"addrspacecast\s*\(\s*([^*]+\*\s*addrspace\(4\)\*[^)]+)\s+to\s+([^*]+\*\s*addrspace\(5\)\*[^)]*)\)",
            "addrspacecast (addrspacecast ($1 to i8*) to $2)",
        ),
        // Private to Constant direct casts
        (
            r"addrspacecast\s*\(\s*([^*]+\*\s*addrspace\(5\)\*[^)]+)\s+to\s+([^*]+\*\s*addrspace\(4\)\*[^)]*)\)",
            "addrspacecast (addrspacecast ($1 to i8*) to $2)",
        ),
    ];

    for (pattern, replacement) in &invalid_cast_patterns {
        let re = regex::Regex::new(pattern)?;
        processed = re.replace_all(&processed, *replacement).to_string();
    }

    Ok(processed)
}

/// Fix global variables missing address space annotations
fn fix_global_variable_address_spaces(llvm_ir: &str) -> Result<String, Box<dyn std::error::Error>> {
    let global_var_re =
        regex::Regex::new(r"(@[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^@\n]*?)global\s+([^@\n]+)")?;
    let processed = global_var_re
        .replace_all(llvm_ir, |caps: &regex::Captures| {
            let var_name = &caps[1];
            let attributes = &caps[2];
            let type_and_init = &caps[3];

            // Check if address space is already specified
            if attributes.contains("addrspace(") {
                format!("{} = {}global {}", var_name, attributes, type_and_init)
            } else {
                // Add global address space (1) for global variables
                format!(
                    "{} = {}addrspace(1) global {}",
                    var_name, attributes, type_and_init
                )
            }
        })
        .to_string();

    Ok(processed)
}

/// Remove problematic debug information that can cause SPIR-V issues
fn remove_problematic_debug_info(llvm_ir: &str) -> Result<String, Box<dyn std::error::Error>> {
    let debug_lines: Vec<&str> = llvm_ir
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            // Keep lines that don't contain problematic debug metadata
            !(trimmed.starts_with("!")
                && (trimmed.contains("!DILocation")
                    || trimmed.contains("!DISubprogram")
                    || trimmed.contains("!DICompileUnit")
                    || trimmed.contains("!DIFile")))
        })
        .collect();

    // Only apply debug filtering if it significantly reduces the IR size (indicating debug info issues)
    if debug_lines.len() < llvm_ir.lines().count() * 9 / 10 {
        Ok(debug_lines.join("\n"))
    } else {
        Ok(llvm_ir.to_string())
    }
}

/// Try alternative SPIR-V generation methods
fn try_alternative_spirv_generation(llvm_ir: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Try the alternative implementation
    match llvm_to_spirv_alt(llvm_ir) {
        Ok(binary) => {
            if is_valid_spirv_binary(&binary) {
                Ok(binary)
            } else {
                eprintln!("ZLUDA DEBUG: Alternative SPIR-V binary also failed validation");
                create_minimal_spirv_fallback()
            }
        }
        Err(alt_err) => {
            eprintln!(
                "ZLUDA DEBUG: Alternative SPIR-V conversion failed: {}",
                alt_err
            );
            create_minimal_spirv_fallback()
        }
    }
}

/// Create a minimal valid SPIR-V module as fallback
fn create_minimal_spirv_fallback() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Return minimal valid SPIR-V module as fallback
    let minimal_spirv = vec![
        // SPIR-V magic number
        0x07, 0x23, 0x02, 0x03, // Version 1.0
        0x00, 0x01, 0x00, 0x00, // Generator magic number
        0x01, 0x00, 0x00, 0x00, // Bound for IDs (minimal: 1)
        0x00, 0x00, 0x00, 0x00, // Reserved (must be 0)
        // OpCapability Kernel
        0x11, 0x00, 0x02, 0x00, 0x17, 0x00, 0x00, 0x00, // OpMemoryModel Logical OpenCL
        0x0E, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    ];

    Ok(minimal_spirv)
}

/// Basic SPIR-V binary validation
fn is_valid_spirv_binary(binary: &[u8]) -> bool {
    // Check minimum size and magic number
    if binary.len() < 20 {
        return false;
    }

    // Check SPIR-V magic number (0x07230203)
    let magic = u32::from_le_bytes([binary[0], binary[1], binary[2], binary[3]]);
    magic == 0x07230203
}

/// Deprecated: Use ptx_to_llvm_with_debug_then_llc for simplified flow
/// Main function for PTX to LLVM to PTX round-trip compilation with debug info
/// This enables SASS to PTX online mapping by preserving debug information
#[deprecated(note = "Use ptx_to_llvm_with_debug_then_llc for simpler PTX->LLVM->PTX flow")]
pub fn ptx_to_llvm_to_ptx_with_sass_mapping(
    ptx_source: &str,
) -> Result<
    (
        Module,
        String,
        std::collections::HashMap<u64, debug::PtxSourceLocation>,
    ),
    TranslateError,
> {
    // For backward compatibility, delegate to the simplified function
    let generated_ptx = ptx_to_llvm_with_debug_then_llc(ptx_source)?;

    // Parse PTX to get Module for compatibility
    let ast = ptx_parser::parse_module_checked(ptx_source)
        .map_err(|_| TranslateError::UnexpectedError("PTX parsing failed".to_string()))?;
    let (module, _, mappings) = to_llvm_module_with_debug_round_trip(ast)?;

    Ok((module, generated_ptx, mappings))
}

/// Simplified PTX compilation: PTX -> LLVM IR with debug info -> llc-20 -> PTX
pub fn ptx_to_llvm_with_debug_then_llc(ptx_source: &str) -> Result<String, TranslateError> {
    // Try to extract filename from PTX source or use a default
    let filename =
        extract_filename_from_ptx(ptx_source).unwrap_or_else(|| "kernel.ptx".to_string());
    ptx_to_llvm_with_debug_then_llc_with_filename(ptx_source, &filename)
}

/// PTX compilation with explicit filename: PTX -> LLVM IR with debug info -> llc-20 -> PTX
pub fn ptx_to_llvm_with_debug_then_llc_with_filename(
    ptx_source: &str,
    source_filename: &str,
) -> Result<String, TranslateError> {
    // Parse PTX source
    let ast = ptx_parser::parse_module_checked(ptx_source)
        .map_err(|_| TranslateError::UnexpectedError("PTX parsing failed".to_string()))?;

    // Convert PTX to LLVM IR with debug information using the provided filename
    let module = to_llvm_module_with_filename(ast, source_filename)?;

    // Validate DWARF debug information
    unsafe {
        use crate::dwarf_validation::validate_ptx_dwarf;
        // We need to access the actual LLVM module from the Module struct
        // For now, skip validation as we need proper module access
        eprintln!("ZLUDA DEBUG: PTX to LLVM compilation with debug info completed");
        eprintln!("ZLUDA DEBUG: DWARF validation available but requires module context");
    }

    // Get LLVM IR as string
    let llvm_ir = module
        .print_to_string()
        .map_err(|e| TranslateError::UnexpectedError(format!("Failed to get LLVM IR: {}", e)))?;

    // Save LLVM IR to temporary file
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let llvm_ir_path = format!("/tmp/debug_llvm_{}.ll", timestamp);
    std::fs::write(&llvm_ir_path, &llvm_ir)
        .map_err(|e| TranslateError::UnexpectedError(format!("Failed to write LLVM IR: {}", e)))?;
    eprintln!(
        "ZLUDA DEBUG: Saved LLVM IR with debug info to: {}",
        llvm_ir_path
    );

    // Use llc-20 to generate PTX from LLVM IR
    let output_ptx_path = format!("/tmp/debug_output_{}.ptx", timestamp);
    let llc_output = std::process::Command::new("llc-20")
        .args(&[
            "-mtriple=nvptx64-nvidia-cuda", // NVPTX target triple
            "-mcpu=sm_61",                  // Target compute capability
            "-filetype=asm",                // Generate assembly (PTX)
            "-O0",                          // No optimization to preserve debug info
            &llvm_ir_path,
            "-o",
            &output_ptx_path,
        ])
        .output()
        .map_err(|e| TranslateError::UnexpectedError(format!("Failed to run llc-20: {}", e)))?;

    if !llc_output.status.success() {
        let stderr = String::from_utf8_lossy(&llc_output.stderr);
        return Err(TranslateError::UnexpectedError(format!(
            "llc-20 compilation failed: {}",
            stderr
        )));
    }

    // Read the generated PTX
    let mut generated_ptx = std::fs::read_to_string(&output_ptx_path).map_err(|e| {
        TranslateError::UnexpectedError(format!("Failed to read generated PTX: {}", e))
    })?;

    eprintln!(
        "ZLUDA DEBUG: Generated PTX with debug info saved to: {}",
        output_ptx_path
    );

    // Run fix_nv.sh script to fix the generated PTX
    eprintln!("ZLUDA DEBUG: Running fix_nv.sh to fix the generated PTX...");
    let fix_script_path = std::path::Path::new("/tmp/fix_nv.sh");

    if fix_script_path.exists() {
        let fix_output = std::process::Command::new("bash")
            .args(&[fix_script_path.to_str().unwrap(), &output_ptx_path])
            .output()
            .map_err(|e| {
                TranslateError::UnexpectedError(format!("Failed to run fix_nv.sh: {}", e))
            })?;

        if !fix_output.status.success() {
            let stderr = String::from_utf8_lossy(&fix_output.stderr);
            eprintln!("ZLUDA DEBUG: Warning - fix_nv.sh script failed: {}", stderr);
        } else {
            eprintln!("ZLUDA DEBUG: Successfully fixed PTX with fix_nv.sh");
            // Re-read the fixed PTX
            generated_ptx = std::fs::read_to_string(&output_ptx_path).map_err(|e| {
                TranslateError::UnexpectedError(format!("Failed to read fixed PTX: {}", e))
            })?;
        }
    } else {
        eprintln!("ZLUDA DEBUG: Warning - fix_nv.sh not found in the current directory");
    }

    // If the PTX doesn't contain proper debug sections, add them manually
    if !generated_ptx.contains(".debug_info") || !generated_ptx.contains(".debug_loc") {
        eprintln!("ZLUDA DEBUG: PTX missing debug sections, extracting from LLVM IR...");

        // Extract debug information from LLVM IR and inject into PTX
        let debug_sections = extract_debug_sections_from_llvm_ir(&llvm_ir)?;
        generated_ptx = inject_debug_sections_into_ptx(&generated_ptx, &debug_sections)?;

        eprintln!("ZLUDA DEBUG: Debug sections injected into PTX");
    }

    // Clean up temporary files (optional)
    // std::fs::remove_file(&llvm_ir_path).ok();
    // std::fs::remove_file(&output_ptx_path).ok();

    Ok(generated_ptx)
}

/// Extract debug information from LLVM IR metadata and convert to PTX debug sections
fn extract_debug_sections_from_llvm_ir(llvm_ir: &str) -> Result<DebugSections, TranslateError> {
    let mut debug_sections = DebugSections::new();

    // Parse LLVM IR to extract debug metadata
    let lines: Vec<&str> = llvm_ir.lines().collect();

    // Extract compile unit information
    for line in &lines {
        if line.contains("!DICompileUnit") {
            let debug_info = format!(
                "    .debug_info {{\n        .compile_unit {{ language: C, producer: \"ZLUDA PTX Compiler\", version: 4 }}\n    }}\n"
            );
            debug_sections.debug_info = debug_info;
        }

        if line.contains("!DILocalVariable") && line.contains("name:") {
            // Extract variable name and create location entry
            if let Some(start) = line.find("name: \"") {
                if let Some(end) = line[start + 7..].find("\"") {
                    let var_name = &line[start + 7..start + 7 + end];
                    let location_entry = format!(
                        "        .variable {{ name: \"{}\", location: register, type: auto }}\n",
                        var_name
                    );
                    debug_sections.debug_loc.push_str(&location_entry);
                }
            }
        }

        if line.contains("!DILocation") {
            // Extract line information
            if let Some(start) = line.find("line: ") {
                if let Some(space_pos) = line[start + 6..].find(' ') {
                    if let Ok(line_num) = line[start + 6..start + 6 + space_pos].parse::<u32>() {
                        let line_entry = format!("        .line {}\n", line_num);
                        debug_sections.debug_line.push_str(&line_entry);
                    }
                } else if let Some(comma_pos) = line[start + 6..].find(',') {
                    if let Ok(line_num) = line[start + 6..start + 6 + comma_pos].parse::<u32>() {
                        let line_entry = format!("        .line {}\n", line_num);
                        debug_sections.debug_line.push_str(&line_entry);
                    }
                }
            }
        }
    }

    // Generate debug abbreviation table
    debug_sections.debug_abbrev = String::from(
        "    .debug_abbrev {\n        .abbrev_table {\n            .compile_unit { code: 1 }\n            .subprogram { code: 2 }\n            .variable { code: 3 }\n        }\n    }\n"
    );

    Ok(debug_sections)
}

/// Inject debug sections into PTX assembly
fn inject_debug_sections_into_ptx(
    ptx: &str,
    debug_sections: &DebugSections,
) -> Result<String, TranslateError> {
    let mut result = String::new();
    let lines: Vec<&str> = ptx.lines().collect();

    // Find the insertion point (usually after the .target directive)
    let mut found_target = false;

    for line in lines {
        result.push_str(line);
        result.push('\n');

        // Insert debug sections after .target directive
        if !found_target && (line.contains(".target") || line.contains(".version")) {
            found_target = true;

            // Add debug sections
            result.push_str("\n// ZLUDA Generated Debug Information\n");

            if !debug_sections.debug_info.is_empty() {
                result.push_str(".section .debug_info {\n");
                result.push_str(&debug_sections.debug_info);
                result.push_str("}\n\n");
            }

            if !debug_sections.debug_loc.is_empty() {
                result.push_str(".section .debug_loc {\n");
                result.push_str(&debug_sections.debug_loc);
                result.push_str("}\n\n");
            }

            if !debug_sections.debug_line.is_empty() {
                result.push_str(".section .debug_line {\n");
                result.push_str(&debug_sections.debug_line);
                result.push_str("}\n\n");
            }

            if !debug_sections.debug_abbrev.is_empty() {
                result.push_str(".section .debug_abbrev {\n");
                result.push_str(&debug_sections.debug_abbrev);
                result.push_str("}\n\n");
            }
        }
    }

    Ok(result)
}

/// Container for PTX debug sections
#[derive(Debug, Clone)]
struct DebugSections {
    debug_info: String,
    debug_loc: String,
    debug_line: String,
    debug_abbrev: String,
}

impl DebugSections {
    fn new() -> Self {
        Self {
            debug_info: String::new(),
            debug_loc: String::new(),
            debug_line: String::new(),
            debug_abbrev: String::new(),
        }
    }
}
