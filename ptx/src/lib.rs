// #![feature(str_from_raw_parts)]

pub mod checkpoint;
pub mod checkpoint_integration;
pub mod debug;
pub mod dwarf_validation;
pub mod pass;
pub mod state_recovery;
#[cfg(test)]
mod test;

pub use pass::llvm::bitcode_to_ir;
pub use pass::to_llvm_module;
pub use pass::to_llvm_module_with_debug_round_trip;
pub use pass::to_llvm_module_with_filename;
pub use pass::to_mlir_module;
pub use pass::Attributes;
pub use pass::TranslateError;
pub use pass::Module;

use std::collections::HashMap;

/// Stub function for backward compatibility
/// TODO: Implement proper PTX to LLVM to PTX round-trip with SASS mapping
pub fn ptx_to_llvm_to_ptx_with_sass_mapping(
    _text: &str,
) -> Result<(pass::Module, String, HashMap<u64, u64>), TranslateError> {
    Err(TranslateError::Todo(
        "ptx_to_llvm_to_ptx_with_sass_mapping not implemented".to_string(),
    ))
}

/// Stub function for backward compatibility
/// TODO: Implement proper LLVM to SPIRV conversion
pub fn llvm_to_spirv_robust(_llvm_ir: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Err("llvm_to_spirv_robust not implemented".into())
}
