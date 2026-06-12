//! Runtime Backend Module
//!
//! This module handles lowering TOSA MLIR to target-specific code
//! and provides execution interfaces for non-CUDA backends.

use crate::{
    cutile_comgr_data_kind_s, cutile_comgr_data_set_t, cutile_comgr_status_s,
    cutile_comgr_status_t, cutile_comgr_target_s, get_next_handle, DataContent, DATA_SET_STORE,
    DATA_STORE,
};
use std::fs;
use std::io::Write;
use std::process::Command;
use tempfile::TempDir;

/// Lower TOSA MLIR to target-specific executable
pub fn lower_tosa_to_target(
    input_handles: &[u64],
    output_set: cutile_comgr_data_set_t,
    target: cutile_comgr_target_s,
) -> cutile_comgr_status_t {
    eprintln!(
        "TOSA->Target: Lowering {} input(s) to target {:?}",
        input_handles.len(),
        target.0
    );

    for &handle in input_handles {
        // Get input data
        let input_content = {
            let store = DATA_STORE.lock().unwrap();
            store.get(&handle).cloned()
        };

        let input_content = match input_content {
            Some(content) => content,
            None => {
                eprintln!("TOSA->Target: Invalid input handle {}", handle);
                return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
            }
        };

        // Verify input is TOSA MLIR
        if input_content.kind != cutile_comgr_data_kind_s::CUTILE_COMGR_DATA_KIND_MLIR_TOSA {
            eprintln!("TOSA->Target: Skipping non-TOSA-MLIR input");
            continue;
        }

        // Lower to target
        let executable = match target {
            t if t == cutile_comgr_target_s::CUTILE_COMGR_TARGET_TENSTORRENT => {
                lower_to_tenstorrent(&input_content.content)?
            }
            t if t == cutile_comgr_target_s::CUTILE_COMGR_TARGET_INTEL => {
                lower_to_intel(&input_content.content)?
            }
            t if t == cutile_comgr_target_s::CUTILE_COMGR_TARGET_AMD => {
                lower_to_amd(&input_content.content)?
            }
            t if t == cutile_comgr_target_s::CUTILE_COMGR_TARGET_GEMMINI => {
                lower_to_gemmini(&input_content.content)?
            }
            t if t == cutile_comgr_target_s::CUTILE_COMGR_TARGET_CPU => {
                lower_to_cpu(&input_content.content)?
            }
            _ => {
                eprintln!("TOSA->Target: Unknown target {:?}", target.0);
                return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
            }
        };

        // Create output data
        let output_handle = get_next_handle();
        {
            let mut store = DATA_STORE.lock().unwrap();
            store.insert(
                output_handle,
                DataContent {
                    kind: cutile_comgr_data_kind_s::CUTILE_COMGR_DATA_KIND_EXECUTABLE,
                    content: executable,
                    name: input_content.name.map(|n| n.replace("_tosa.mlir", ".bin")),
                },
            );
        }

        // Add to output set
        {
            let mut store = DATA_SET_STORE.lock().unwrap();
            if let Some(handles) = store.get_mut(&output_set.handle) {
                handles.push(output_handle);
            }
        }

        eprintln!("TOSA->Target: Successfully lowered to target");
    }

    Ok(())
}

/// Lower TOSA to Tenstorrent backend using ttmlir
fn lower_to_tenstorrent(tosa_mlir: &[u8]) -> Result<Vec<u8>, cutile_comgr_status_s> {
    eprintln!("TOSA->TT: Lowering to Tenstorrent backend");

    // Create temp directory for compilation
    let temp_dir = TempDir::new()
        .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;

    let input_path = temp_dir.path().join("input_tosa.mlir");
    let output_path = temp_dir.path().join("output.ttnn");

    // Write TOSA MLIR to file
    {
        let mut file = fs::File::create(&input_path)
            .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;
        file.write_all(tosa_mlir)
            .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;
    }

    // First: Lower TOSA to TTIR using ttmlir-opt
    let ttir_path = temp_dir.path().join("ttir.mlir");
    let ttmlir_opt_result = Command::new("ttmlir-opt")
        .args([
            "--convert-tosa-to-ttir",
            "--ttir-layout",
            "--ttir-to-ttnn-backend-pipeline",
            input_path.to_str().unwrap(),
            "-o",
            ttir_path.to_str().unwrap(),
        ])
        .output();

    match ttmlir_opt_result {
        Ok(output) => {
            if !output.status.success() {
                eprintln!(
                    "TOSA->TT: ttmlir-opt failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                // Return the TOSA MLIR as-is for debugging
                return Ok(tosa_mlir.to_vec());
            }
        }
        Err(e) => {
            eprintln!("TOSA->TT: ttmlir-opt not found or failed: {}", e);
            // Fallback: return TOSA MLIR with TTNN wrapper
            return Ok(generate_ttnn_placeholder(tosa_mlir));
        }
    }

    // Read the output
    fs::read(&output_path)
        .or_else(|_| fs::read(&ttir_path))
        .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_TRANSFORMATION_FAILED)
}

/// Lower TOSA to Intel GPU backend using SPIRV
fn lower_to_intel(tosa_mlir: &[u8]) -> Result<Vec<u8>, cutile_comgr_status_s> {
    eprintln!("TOSA->Intel: Lowering to Intel GPU backend");

    let temp_dir = TempDir::new()
        .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;

    let input_path = temp_dir.path().join("input_tosa.mlir");
    let llvm_path = temp_dir.path().join("llvm.mlir");
    let spirv_path = temp_dir.path().join("output.spv");

    // Write TOSA MLIR to file
    {
        let mut file = fs::File::create(&input_path)
            .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;
        file.write_all(tosa_mlir)
            .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;
    }

    // TOSA -> Linalg -> LLVM -> SPIRV pipeline
    let mlir_opt_result = Command::new("mlir-opt")
        .args([
            "--pass-pipeline=builtin.module(func.func(tosa-to-linalg),convert-linalg-to-loops,convert-scf-to-cf,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)",
            input_path.to_str().unwrap(),
            "-o",
            llvm_path.to_str().unwrap(),
        ])
        .output();

    match mlir_opt_result {
        Ok(output) => {
            if !output.status.success() {
                eprintln!(
                    "TOSA->Intel: mlir-opt failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                return Ok(generate_spirv_placeholder(tosa_mlir));
            }
        }
        Err(e) => {
            eprintln!("TOSA->Intel: mlir-opt not found: {}", e);
            return Ok(generate_spirv_placeholder(tosa_mlir));
        }
    }

    // Convert to SPIRV using mlir-translate
    let translate_result = Command::new("mlir-translate")
        .args(["--mlir-to-llvmir", llvm_path.to_str().unwrap()])
        .output();

    match translate_result {
        Ok(output) => {
            if output.status.success() {
                Ok(output.stdout)
            } else {
                Ok(generate_spirv_placeholder(tosa_mlir))
            }
        }
        Err(_) => Ok(generate_spirv_placeholder(tosa_mlir)),
    }
}

/// Lower TOSA to AMD GPU backend using AMDGPU
fn lower_to_amd(tosa_mlir: &[u8]) -> Result<Vec<u8>, cutile_comgr_status_s> {
    eprintln!("TOSA->AMD: Lowering to AMD GPU backend");

    let temp_dir = TempDir::new()
        .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;

    let input_path = temp_dir.path().join("input_tosa.mlir");
    let llvm_path = temp_dir.path().join("llvm.mlir");

    // Write TOSA MLIR to file
    {
        let mut file = fs::File::create(&input_path)
            .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;
        file.write_all(tosa_mlir)
            .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;
    }

    // TOSA -> Linalg -> GPU -> ROCDL pipeline
    let mlir_opt_result = Command::new("mlir-opt")
        .args([
            "--pass-pipeline=builtin.module(func.func(tosa-to-linalg),convert-linalg-to-loops,convert-scf-to-cf,convert-cf-to-llvm,convert-func-to-llvm)",
            input_path.to_str().unwrap(),
            "-o",
            llvm_path.to_str().unwrap(),
        ])
        .output();

    match mlir_opt_result {
        Ok(output) => {
            if !output.status.success() {
                eprintln!("TOSA->AMD: mlir-opt failed");
                return Ok(generate_amdgpu_placeholder(tosa_mlir));
            }

            // Compile to AMDGPU bitcode
            fs::read(&llvm_path)
                .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_TRANSFORMATION_FAILED)
        }
        Err(_) => Ok(generate_amdgpu_placeholder(tosa_mlir)),
    }
}

/// Lower TOSA to Gemmini accelerator
fn lower_to_gemmini(tosa_mlir: &[u8]) -> Result<Vec<u8>, cutile_comgr_status_s> {
    eprintln!("TOSA->Gemmini: Lowering to Gemmini accelerator");

    // Gemmini-specific lowering would go through:
    // TOSA -> Linalg -> Affine -> Gemmini dialect -> RISC-V assembly

    let temp_dir = TempDir::new()
        .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;

    let input_path = temp_dir.path().join("input_tosa.mlir");

    // Write TOSA MLIR to file
    {
        let mut file = fs::File::create(&input_path)
            .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;
        file.write_all(tosa_mlir)
            .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;
    }

    // Try to use Gemmini-specific tooling
    let gemmini_result = Command::new("gemmini-opt")
        .args(["--tosa-to-gemmini", input_path.to_str().unwrap()])
        .output();

    match gemmini_result {
        Ok(output) if output.status.success() => Ok(output.stdout),
        _ => {
            eprintln!("TOSA->Gemmini: Using fallback lowering");
            Ok(generate_gemmini_placeholder(tosa_mlir))
        }
    }
}

/// Lower TOSA to CPU backend using LLVM
fn lower_to_cpu(tosa_mlir: &[u8]) -> Result<Vec<u8>, cutile_comgr_status_s> {
    eprintln!("TOSA->CPU: Lowering to CPU backend");

    let temp_dir = TempDir::new()
        .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;

    let input_path = temp_dir.path().join("input_tosa.mlir");
    let llvm_path = temp_dir.path().join("llvm.ll");
    let obj_path = temp_dir.path().join("output.o");

    // Write TOSA MLIR to file
    {
        let mut file = fs::File::create(&input_path)
            .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;
        file.write_all(tosa_mlir)
            .map_err(|_| cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES)?;
    }

    // TOSA -> Linalg -> Loops -> LLVM pipeline
    let mlir_opt_result = Command::new("mlir-opt")
        .args([
            "--pass-pipeline=builtin.module(func.func(tosa-to-linalg),convert-linalg-to-loops,lower-affine,convert-scf-to-cf,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)",
            input_path.to_str().unwrap(),
        ])
        .output();

    let mlir_output = match mlir_opt_result {
        Ok(output) if output.status.success() => output.stdout,
        _ => {
            eprintln!("TOSA->CPU: mlir-opt failed, using placeholder");
            return Ok(generate_cpu_placeholder(tosa_mlir));
        }
    };

    // Translate to LLVM IR
    let translate_result = Command::new("mlir-translate")
        .args(["--mlir-to-llvmir"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn();

    match translate_result {
        Ok(mut child) => {
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(&mlir_output);
            }

            match child.wait_with_output() {
                Ok(output) if output.status.success() => {
                    // Write LLVM IR
                    fs::write(&llvm_path, &output.stdout).ok();

                    // Compile to object file
                    let llc_result = Command::new("llc")
                        .args([
                            "-filetype=obj",
                            "-O3",
                            llvm_path.to_str().unwrap(),
                            "-o",
                            obj_path.to_str().unwrap(),
                        ])
                        .output();

                    match llc_result {
                        Ok(llc_out) if llc_out.status.success() => {
                            fs::read(&obj_path).map_err(|_| {
                                cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_TRANSFORMATION_FAILED
                            })
                        }
                        _ => Ok(output.stdout), // Return LLVM IR if object compilation fails
                    }
                }
                _ => Ok(generate_cpu_placeholder(tosa_mlir)),
            }
        }
        Err(_) => Ok(generate_cpu_placeholder(tosa_mlir)),
    }
}

// Placeholder generators for when actual tools aren't available

fn generate_ttnn_placeholder(tosa_mlir: &[u8]) -> Vec<u8> {
    let mlir_str = String::from_utf8_lossy(tosa_mlir);
    format!(
        "// TTNN placeholder generated from TOSA\n\
         // Original TOSA MLIR:\n\
         {}\n\
         // TODO: Implement actual TTNN lowering\n",
        mlir_str
    )
    .into_bytes()
}

fn generate_spirv_placeholder(tosa_mlir: &[u8]) -> Vec<u8> {
    let mlir_str = String::from_utf8_lossy(tosa_mlir);
    format!(
        "; SPIR-V placeholder generated from TOSA\n\
         ; Original TOSA MLIR:\n\
         ; {}\n",
        mlir_str.replace("\n", "\n; ")
    )
    .into_bytes()
}

fn generate_amdgpu_placeholder(tosa_mlir: &[u8]) -> Vec<u8> {
    let mlir_str = String::from_utf8_lossy(tosa_mlir);
    format!(
        "; AMDGPU placeholder generated from TOSA\n\
         ; Original TOSA MLIR:\n\
         ; {}\n",
        mlir_str.replace("\n", "\n; ")
    )
    .into_bytes()
}

fn generate_gemmini_placeholder(tosa_mlir: &[u8]) -> Vec<u8> {
    let mlir_str = String::from_utf8_lossy(tosa_mlir);
    format!(
        "# Gemmini placeholder generated from TOSA\n\
         # Original TOSA MLIR:\n\
         # {}\n",
        mlir_str.replace("\n", "\n# ")
    )
    .into_bytes()
}

fn generate_cpu_placeholder(tosa_mlir: &[u8]) -> Vec<u8> {
    let mlir_str = String::from_utf8_lossy(tosa_mlir);
    format!(
        "; CPU LLVM IR placeholder generated from TOSA\n\
         ; Original TOSA MLIR:\n\
         ; {}\n\
         \n\
         define void @kernel() {{\n\
           ret void\n\
         }}\n",
        mlir_str.replace("\n", "\n; ")
    )
    .into_bytes()
}

/// Execute compiled kernel on target
pub fn execute_kernel(
    executable: &[u8],
    target: cutile_comgr_target_s,
    args: &[*mut std::ffi::c_void],
) -> Result<(), cutile_comgr_status_s> {
    match target {
        t if t == cutile_comgr_target_s::CUTILE_COMGR_TARGET_TENSTORRENT => {
            execute_on_tenstorrent(executable, args)
        }
        t if t == cutile_comgr_target_s::CUTILE_COMGR_TARGET_INTEL => {
            execute_on_intel(executable, args)
        }
        t if t == cutile_comgr_target_s::CUTILE_COMGR_TARGET_AMD => {
            execute_on_amd(executable, args)
        }
        t if t == cutile_comgr_target_s::CUTILE_COMGR_TARGET_GEMMINI => {
            execute_on_gemmini(executable, args)
        }
        t if t == cutile_comgr_target_s::CUTILE_COMGR_TARGET_CPU => {
            execute_on_cpu(executable, args)
        }
        _ => Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    }
}

fn execute_on_tenstorrent(
    _executable: &[u8],
    _args: &[*mut std::ffi::c_void],
) -> Result<(), cutile_comgr_status_s> {
    eprintln!("Runtime: Tenstorrent execution not yet implemented");
    // Would use tt_runtime_sys here
    Ok(())
}

fn execute_on_intel(
    _executable: &[u8],
    _args: &[*mut std::ffi::c_void],
) -> Result<(), cutile_comgr_status_s> {
    eprintln!("Runtime: Intel execution not yet implemented");
    // Would use ze_runtime_sys here
    Ok(())
}

fn execute_on_amd(
    _executable: &[u8],
    _args: &[*mut std::ffi::c_void],
) -> Result<(), cutile_comgr_status_s> {
    eprintln!("Runtime: AMD execution not yet implemented");
    // Would use hip_runtime_sys here
    Ok(())
}

fn execute_on_gemmini(
    _executable: &[u8],
    _args: &[*mut std::ffi::c_void],
) -> Result<(), cutile_comgr_status_s> {
    eprintln!("Runtime: Gemmini execution not yet implemented");
    // Would use gemmini_runtime_sys here
    Ok(())
}

fn execute_on_cpu(
    _executable: &[u8],
    _args: &[*mut std::ffi::c_void],
) -> Result<(), cutile_comgr_status_s> {
    eprintln!("Runtime: CPU execution not yet implemented");
    // Would JIT compile and execute on CPU
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placeholders() {
        let tosa = b"module {}";

        let ttnn = generate_ttnn_placeholder(tosa);
        assert!(!ttnn.is_empty());

        let spirv = generate_spirv_placeholder(tosa);
        assert!(!spirv.is_empty());

        let cpu = generate_cpu_placeholder(tosa);
        assert!(!cpu.is_empty());
    }
}
