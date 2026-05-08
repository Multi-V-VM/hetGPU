pub mod control_codes;
pub mod sm120;

use crate::types::*;

pub fn encode(inst: &SassInst, sm_version: u32) -> Result<[u8; 16], NvSassError> {
    match sm_version {
        120 | 121 => sm120::encode(inst),
        _ => Err(NvSassError::UnsupportedSmVersion(sm_version)),
    }
}

pub fn encode_kernel(kernel: &SassKernel, sm_version: u32) -> Result<Vec<[u8; 16]>, NvSassError> {
    kernel
        .instructions
        .iter()
        .map(|inst| encode(inst, sm_version))
        .collect()
}
