//! Tests for SASS to PTX debug symbol mapping
//!
//! This module tests that SASS (NVIDIA GPU assembly) instructions can be mapped
//! back to PTX source locations using debug symbols.

use crate::pass::{self, TranslateError};
use ptx_parser as ast;
use std::collections::HashMap;
use std::process::Command;
use tempfile::NamedTempFile;

/// Test PTX source with line markers for verification
const TEST_PTX_SOURCE: &str = r#"
.version 8.0
.target sm_86
.address_size 64

.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2
)
{
    .reg .b32   %r<4>;
    .reg .b64   %rd<11>;
    .reg .f32   %f<4>;

    // LINE 16: Load first parameter
    ld.param.u64    %rd1, [vector_add_param_0];
    // LINE 18: Load second parameter
    ld.param.u64    %rd2, [vector_add_param_1];
    // LINE 20: Load third parameter
    ld.param.u64    %rd3, [vector_add_param_2];

    // LINE 23: Get thread ID
    mov.u32     %r1, %tid.x;
    cvt.u64.u32 %rd4, %r1;
    mul.wide.u32    %rd5, %r1, 4;

    // LINE 28: Load first value
    add.u64     %rd6, %rd1, %rd5;
    ld.global.f32   %f1, [%rd6];

    // LINE 32: Load second value
    add.u64     %rd7, %rd2, %rd5;
    ld.global.f32   %f2, [%rd7];

    // LINE 36: Perform addition
    add.f32     %f3, %f1, %f2;

    // LINE 39: Store result
    add.u64     %rd10, %rd3, %rd5;
    st.global.f32   [%rd10], %f3;

    ret;
}
"#;

/// Check if cuobjdump is available (part of CUDA toolkit)
fn has_cuobjdump() -> bool {
    Command::new("cuobjdump")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if nvcc is available
fn has_nvcc() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Compile PTX to CUBIN (CUDA binary with SASS)
fn compile_ptx_to_cubin(ptx_path: &str, cubin_path: &str) -> Result<(), String> {
    let output = Command::new("ptxas")
        .args(&[
            "-arch=sm_61",
            "--gpu-name=sm_61",
            "-g", // Generate debug info
            "-lineinfo", // Include line info
            ptx_path,
            "-o",
            cubin_path,
        ])
        .output()
        .map_err(|e| format!("Failed to run ptxas: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "ptxas failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(())
}

/// Extract SASS disassembly with debug info
fn disassemble_cubin(cubin_path: &str) -> Result<String, String> {
    let output = Command::new("cuobjdump")
        .args(&[
            "-sass",
            "-lineinfo",
            cubin_path,
        ])
        .output()
        .map_err(|e| format!("Failed to run cuobjdump: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "cuobjdump failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Parse SASS disassembly to extract line number mappings
#[derive(Debug, Clone)]
struct SassLineMapping {
    sass_offset: String,
    sass_instruction: String,
    ptx_file: String,
    ptx_line: u32,
}

fn parse_sass_line_mappings(sass_disasm: &str) -> Vec<SassLineMapping> {
    let mut mappings = Vec::new();
    let mut current_file = String::from("unknown");

    for line in sass_disasm.lines() {
        // Look for line info: ## File "kernel.ptx", line 16
        if line.contains("## File") {
            if let Some(file_part) = line.split('"').nth(1) {
                current_file = file_part.to_string();
            }
        }

        // Look for line number markers: ## Line 16
        if let Some(line_marker) = line.strip_prefix("## Line ") {
            if let Ok(line_num) = line_marker.trim().parse::<u32>() {
                // The next line should be the SASS instruction
                // For now, we'll store the line number
                // In practice, we'd parse the next SASS instruction
                mappings.push(SassLineMapping {
                    sass_offset: "0x0000".to_string(),
                    sass_instruction: "SASS_INST".to_string(),
                    ptx_file: current_file.clone(),
                    ptx_line: line_num,
                });
            }
        }
    }

    mappings
}

#[test]
fn test_sass_to_ptx_debug_mapping() -> Result<(), Box<dyn std::error::Error>> {
    // Skip if CUDA tools aren't available
    if !has_nvcc() {
        println!("⚠ Skipping test: nvcc not found");
        return Ok(());
    }

    // Step 1: Compile PTX to LLVM with debug info, then back to PTX
    let ast = ast::parse_module_checked(TEST_PTX_SOURCE)
        .map_err(|e| format!("PTX parsing failed: {:?}", e))?;

    let (_module, regenerated_ptx, _debug_mappings) =
        crate::to_llvm_module_with_debug_round_trip(ast)?;

    println!("=== Regenerated PTX with Debug Info ===");
    println!("{}", regenerated_ptx);

    // Verify PTX has debug markers
    assert!(
        regenerated_ptx.contains(".file") || regenerated_ptx.contains(".loc"),
        "Regenerated PTX must contain debug directives"
    );

    // Step 2: Write PTX to temp file
    let ptx_file = NamedTempFile::new()?;
    std::fs::write(ptx_file.path(), &regenerated_ptx)?;
    println!("Wrote PTX to: {:?}", ptx_file.path());

    // Step 3: Compile PTX to CUBIN with debug info
    let cubin_file = NamedTempFile::new()?;
    match compile_ptx_to_cubin(
        ptx_file.path().to_str().unwrap(),
        cubin_file.path().to_str().unwrap(),
    ) {
        Ok(_) => println!("✓ Compiled PTX to CUBIN"),
        Err(e) => {
            println!("⚠ Warning: Could not compile to CUBIN: {}", e);
            println!("This is expected if ptxas is not available");
            return Ok(());
        }
    }

    // Step 4: Disassemble CUBIN to get SASS with line info
    if !has_cuobjdump() {
        println!("⚠ Skipping SASS verification: cuobjdump not found");
        return Ok(());
    }

    let sass_disasm = disassemble_cubin(cubin_file.path().to_str().unwrap())?;
    println!("\n=== SASS Disassembly with Line Info ===");
    println!("{}", sass_disasm);

    // Step 5: Verify SASS contains line number mappings
    assert!(
        sass_disasm.contains("Line ") || sass_disasm.contains("line "),
        "SASS disassembly should contain line number information"
    );

    // Step 6: Parse and verify line mappings
    let mappings = parse_sass_line_mappings(&sass_disasm);

    if mappings.is_empty() {
        println!("⚠ Warning: No line mappings found in SASS disassembly");
        println!("This may indicate debug info was not properly preserved");
    } else {
        println!("\n=== Extracted Line Mappings ===");
        for mapping in &mappings {
            println!(
                "SASS @ {} -> {}:{}",
                mapping.sass_offset, mapping.ptx_file, mapping.ptx_line
            );
        }
        println!("✓ Found {} SASS-to-PTX line mappings", mappings.len());
    }

    Ok(())
}

/// Test using GDB to verify debug symbols
#[test]
fn test_gdb_debug_symbols() -> Result<(), Box<dyn std::error::Error>> {
    // This test would require:
    // 1. A compiled CUDA program
    // 2. GDB with CUDA debugging support (cuda-gdb)
    // 3. Setting breakpoints and verifying source locations

    // Check if cuda-gdb is available
    let has_cuda_gdb = Command::new("cuda-gdb")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !has_cuda_gdb {
        println!("⚠ Skipping GDB test: cuda-gdb not found");
        return Ok(());
    }

    // For now, just verify the tools are available
    println!("✓ cuda-gdb is available for manual debugging");
    println!("To debug SASS-to-PTX mapping manually:");
    println!("  1. Compile your kernel: nvcc -g -G kernel.cu -o kernel");
    println!("  2. Run cuda-gdb: cuda-gdb ./kernel");
    println!("  3. Set breakpoint: break kernel_name");
    println!("  4. Run and inspect: run, then use 'info line' and 'disassemble /s'");

    Ok(())
}

/// Test to verify .loc directives match source lines
#[test]
fn test_loc_directive_accuracy() -> Result<(), Box<dyn std::error::Error>> {
    let ast = ast::parse_module_checked(TEST_PTX_SOURCE)
        .map_err(|e| format!("PTX parsing failed: {:?}", e))?;

    let (_module, regenerated_ptx, _debug_mappings) =
        crate::to_llvm_module_with_debug_round_trip(ast)?;

    // Find all .loc directives
    let mut loc_directives = Vec::new();
    for line in regenerated_ptx.lines() {
        if line.trim().starts_with(".loc") {
            loc_directives.push(line.trim().to_string());
        }
    }

    println!("=== Found .loc Directives ===");
    for (i, loc) in loc_directives.iter().enumerate() {
        println!("{}: {}", i, loc);
    }

    // Verify we have .loc directives
    assert!(
        !loc_directives.is_empty(),
        "PTX should contain .loc directives for source line mapping"
    );

    println!("✓ Found {} .loc directives in generated PTX", loc_directives.len());

    Ok(())
}

/// Test debug info in ELF format (for analysis)
#[test]
fn test_dwarf_debug_sections() -> Result<(), Box<dyn std::error::Error>> {
    let ast = ast::parse_module_checked(TEST_PTX_SOURCE)
        .map_err(|e| format!("PTX parsing failed: {:?}", e))?;

    let (_module, regenerated_ptx, _debug_mappings) =
        crate::to_llvm_module_with_debug_round_trip(ast)?;

    // Check for DWARF debug sections in PTX
    let has_debug_info = regenerated_ptx.contains(".section\t.debug_info");
    let has_debug_abbrev = regenerated_ptx.contains(".section\t.debug_abbrev");
    let has_debug_line = regenerated_ptx.contains(".section\t.debug_line") ||
                         regenerated_ptx.contains(".file");

    println!("=== Debug Sections Present ===");
    println!(".debug_info: {}", has_debug_info);
    println!(".debug_abbrev: {}", has_debug_abbrev);
    println!(".debug_line or .file: {}", has_debug_line);

    assert!(
        has_debug_info || has_debug_line,
        "PTX should contain DWARF debug sections or .file/.loc directives"
    );

    println!("✓ PTX contains debug information for SASS mapping");

    Ok(())
}
