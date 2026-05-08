//! AMD AIE compilation driver.
//!
//! Takes TOSA-dialect MLIR and invokes the Xilinx/mlir-aie toolchain
//! (`aie-opt`, `aie-translate`) to produce an XCLBIN for Strix NPU.

use std::path::PathBuf;
use thiserror::Error;

mod pipeline;

/// Target AIE device family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AieDevice {
    /// AMD Strix (XDNA2, 4 columns × 5 rows including shim).
    Strix,
}

/// Configuration for an AIE compilation run.
#[derive(Debug, Clone)]
pub struct AieCompileConfig {
    pub device: AieDevice,
    pub num_cols: u32,
    pub num_rows: u32,
    pub extra_aie_opt_flags: Vec<String>,
}

impl Default for AieCompileConfig {
    fn default() -> Self {
        Self::strix()
    }
}

impl AieCompileConfig {
    /// Default configuration for AMD Strix NPU.
    pub fn strix() -> Self {
        Self {
            device: AieDevice::Strix,
            num_cols: 4,
            num_rows: 5,
            extra_aie_opt_flags: Vec::new(),
        }
    }
}

/// Errors produced by the AIE compilation driver.
#[derive(Debug, Error)]
pub enum AieComgrError {
    #[error("mlir-aie toolchain binary not found: {0}. Install from https://github.com/Xilinx/mlir-aie or set AIE_TOOLCHAIN_DIR.")]
    ToolchainNotFound(String),

    #[error("{step} failed (exit {exit_code}):\n{stderr}")]
    ToolchainFailed {
        step: &'static str,
        stderr: String,
        exit_code: i32,
    },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid input MLIR: {0}")]
    InvalidInput(String),
}

/// Compile a TOSA-dialect MLIR string to an XCLBIN byte blob.
pub fn compile_tosa_to_xclbin(
    tosa_mlir: &str,
    config: &AieCompileConfig,
) -> Result<Vec<u8>, AieComgrError> {
    pipeline::run(tosa_mlir, config)
}

/// Locate an mlir-aie toolchain binary: checks `AIE_TOOLCHAIN_DIR` first, then `$PATH`.
pub(crate) fn find_toolchain_binary(name: &str) -> Result<PathBuf, AieComgrError> {
    if let Ok(dir) = std::env::var("AIE_TOOLCHAIN_DIR") {
        let candidate = PathBuf::from(&dir).join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    which::which(name).map_err(|_| AieComgrError::ToolchainNotFound(name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_invalid_input() {
        let config = AieCompileConfig::default();
        let err = compile_tosa_to_xclbin("", &config).unwrap_err();
        assert!(matches!(err, AieComgrError::InvalidInput(_)));
    }

    #[test]
    fn default_config_is_strix() {
        let config = AieCompileConfig::default();
        assert_eq!(config.device, AieDevice::Strix);
        assert_eq!(config.num_cols, 4);
        assert_eq!(config.num_rows, 5);
    }
}
