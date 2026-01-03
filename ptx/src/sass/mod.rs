//! SASS (Streaming ASSembler) parsing and analysis module
//!
//! This module provides native CUBIN/ELF parsing, SASS instruction semantics,
//! and bidirectional SASS ↔ PTX mapping for GPU debugging and profiling.
//!
//! Key components:
//! - `instruction`: Enhanced SASS instruction semantics and PTX templates
//! - `cubin_parser`: Native CUBIN/ELF binary parser
//! - `dwarf_parser`: DWARF debug information extraction
//! - `disassembler`: SASS binary and text disassembly
//! - `llvm_inline`: SASS inlining to LLVM IR for profiling/replay
//! - `ptx_recovery`: PTX source recovery from SASS using debug info

pub mod instruction;
pub mod cubin_parser;
pub mod dwarf_parser;
pub mod disassembler;
pub mod llvm_inline;
pub mod ptx_recovery;

pub use instruction::*;
pub use cubin_parser::*;
pub use dwarf_parser::*;
pub use disassembler::*;
pub use llvm_inline::*;
pub use ptx_recovery::*;
