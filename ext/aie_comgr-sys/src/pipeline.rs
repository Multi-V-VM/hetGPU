//! Subprocess-driven pipeline from TOSA MLIR to XCLBIN.
//!
//! Stages (each a separate subprocess invocation):
//!   1. aie-opt tosa-to-linalg-and-friends     → lowered.mlir
//!   2. aie-opt aie-objectFifo / lower-broadcast → aie.mlir
//!   3. aie-translate --aie-generate-cdo         → aie_cdo.bin
//!   4. aie-translate --aie-generate-ipu         → ipu_insts.txt
//!   5. aie-translate --aie-generate-xclbin      → kernel.xclbin
//!   6. read kernel.xclbin into Vec<u8>

use std::fs;
use std::path::Path;
use std::process::Command;

use crate::{find_toolchain_binary, AieComgrError, AieCompileConfig};

pub(crate) fn run(tosa_mlir: &str, config: &AieCompileConfig) -> Result<Vec<u8>, AieComgrError> {
    if tosa_mlir.trim().is_empty() {
        return Err(AieComgrError::InvalidInput("empty MLIR input".to_string()));
    }

    let workdir = tempfile::tempdir()?;
    let input_path = workdir.path().join("input.mlir");
    let lowered_path = workdir.path().join("lowered.mlir");
    let aie_path = workdir.path().join("aie.mlir");
    let xclbin_path = workdir.path().join("kernel.xclbin");

    fs::write(&input_path, tosa_mlir)?;

    let aie_opt = find_toolchain_binary("aie-opt")?;
    let aie_translate = find_toolchain_binary("aie-translate")?;

    // Stage 1: tosa → linalg → aievec (high-level)
    run_step(
        "aie-opt (tosa-to-aievec)",
        Command::new(&aie_opt)
            .arg("--pass-pipeline=builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg),linalg-generalize-named-ops,linalg-fuse-elementwise-ops)")
            .args(&config.extra_aie_opt_flags)
            .arg(&input_path)
            .arg("-o")
            .arg(&lowered_path),
    )?;

    // Stage 2: aie dialect transforms (objectFifo + broadcast lowering)
    run_step(
        "aie-opt (aie-transforms)",
        Command::new(&aie_opt)
            .arg("--aie-objectFifo-stateful-transform")
            .arg("--aie-lower-broadcast-packet")
            .arg(&lowered_path)
            .arg("-o")
            .arg(&aie_path),
    )?;

    // Stage 3: generate CDO (configuration data object)
    run_step(
        "aie-translate --aie-generate-cdo",
        Command::new(&aie_translate)
            .arg("--aie-generate-cdo")
            .arg(&aie_path)
            .current_dir(workdir.path()),
    )?;

    // Stage 4: generate IPU instructions
    run_step(
        "aie-translate --aie-generate-ipu",
        Command::new(&aie_translate)
            .arg("--aie-generate-ipu")
            .arg(&aie_path)
            .current_dir(workdir.path()),
    )?;

    // Stage 5: assemble XCLBIN
    run_step(
        "aie-translate --aie-generate-xclbin",
        Command::new(&aie_translate)
            .arg("--aie-generate-xclbin")
            .arg(format!("--xclbin-name={}", xclbin_path.display()))
            .arg(&aie_path)
            .current_dir(workdir.path()),
    )?;

    read_xclbin(&xclbin_path)
}

fn run_step(step: &'static str, cmd: &mut Command) -> Result<(), AieComgrError> {
    let out = cmd.output()?;
    if !out.status.success() {
        let mut diag = String::from_utf8_lossy(&out.stderr).into_owned();
        let stdout = String::from_utf8_lossy(&out.stdout);
        if !stdout.is_empty() {
            if !diag.is_empty() {
                diag.push('\n');
            }
            diag.push_str("--- stdout ---\n");
            diag.push_str(&stdout);
        }
        return Err(AieComgrError::ToolchainFailed {
            step,
            stderr: diag,
            exit_code: out.status.code().unwrap_or(-1),
        });
    }
    Ok(())
}

fn read_xclbin(path: &Path) -> Result<Vec<u8>, AieComgrError> {
    fs::read(path).map_err(AieComgrError::Io)
}
