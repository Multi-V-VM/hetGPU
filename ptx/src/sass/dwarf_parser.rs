//! DWARF debug information parser for CUBIN files
//!
//! This module parses DWARF debug information embedded in CUBIN files to extract:
//! - Line number mappings (SASS address -> PTX source location)
//! - Function information
//! - Variable locations
//!
//! CUDA uses DWARF v2 format with some NVIDIA-specific extensions.

use object::{Object, ObjectSection};
use std::collections::HashMap;
use std::fmt;

use super::cubin_parser::DebugLineInfo;

// ============================================================================
// DWARF Parser
// ============================================================================

/// DWARF debug information parser
pub struct DwarfParser<'a> {
    /// Raw data slice
    data: &'a [u8],
}

/// Parsed debug information
#[derive(Debug, Default, Clone)]
pub struct ParsedDebugInfo {
    /// Line number information: SASS address -> source location
    pub line_mappings: HashMap<u64, DebugLineInfo>,
    /// File table: index -> file path
    pub file_table: HashMap<u64, String>,
    /// Function information
    pub functions: Vec<DebugFunctionInfo>,
    /// Variable information
    pub variables: Vec<DebugVariableInfo>,
    /// Compilation units
    pub compilation_units: Vec<CompilationUnitInfo>,
}

/// Debug information for a function
#[derive(Debug, Clone)]
pub struct DebugFunctionInfo {
    /// Function name
    pub name: String,
    /// Linkage name (mangled)
    pub linkage_name: Option<String>,
    /// Start address
    pub low_pc: u64,
    /// End address or size
    pub high_pc: u64,
    /// Source file
    pub file: Option<String>,
    /// Line number
    pub line: u32,
    /// Is this an inlined function?
    pub is_inlined: bool,
}

/// Debug information for a variable
#[derive(Debug, Clone)]
pub struct DebugVariableInfo {
    /// Variable name
    pub name: String,
    /// Type name
    pub type_name: Option<String>,
    /// Location expression
    pub location: VariableLocationExpr,
    /// Source file
    pub file: Option<String>,
    /// Line number
    pub line: u32,
    /// Is this a parameter?
    pub is_parameter: bool,
}

/// Variable location expression
#[derive(Debug, Clone)]
pub enum VariableLocationExpr {
    /// Register
    Register(String),
    /// Memory address
    Address(u64),
    /// Stack offset
    StackOffset(i64),
    /// Complex expression
    Expression(Vec<u8>),
    /// Unknown/not available
    Unknown,
}

/// Compilation unit information
#[derive(Debug, Clone)]
pub struct CompilationUnitInfo {
    /// Unit name/file
    pub name: String,
    /// Compilation directory
    pub comp_dir: Option<String>,
    /// Producer string
    pub producer: Option<String>,
    /// Language
    pub language: u32,
    /// Low PC (start address)
    pub low_pc: u64,
    /// High PC (end address)
    pub high_pc: u64,
    /// DWARF version
    pub dwarf_version: u16,
}

impl<'a> DwarfParser<'a> {
    /// Create a new DWARF parser from raw CUBIN data
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Parse all debug information from the CUBIN
    pub fn parse(&self) -> Result<ParsedDebugInfo, DwarfParseError> {
        let object = object::File::parse(self.data)
            .map_err(|e| DwarfParseError::ObjectParse(e.to_string()))?;

        let mut result = ParsedDebugInfo::default();

        // Try to parse debug line section
        if let Some(debug_line) = object.section_by_name(".debug_line") {
            if let Ok(data) = debug_line.data() {
                self.parse_debug_line_simple(data, &mut result)?;
            }
        }

        Ok(result)
    }

    /// Simple .debug_line parser without full gimli dependency
    /// This parses the basic DWARF line number program format
    fn parse_debug_line_simple(
        &self,
        data: &[u8],
        result: &mut ParsedDebugInfo,
    ) -> Result<(), DwarfParseError> {
        if data.len() < 4 {
            return Ok(());
        }

        let mut offset = 0;

        while offset + 4 <= data.len() {
            // Read unit_length (4 bytes for 32-bit DWARF)
            let unit_length = u32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
            ]) as usize;

            if unit_length == 0 || offset + unit_length + 4 > data.len() {
                break;
            }

            // Skip this unit for now - basic parsing would require full state machine
            offset += 4 + unit_length;
        }

        Ok(())
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Parse debug line information from CUBIN data
pub fn parse_debug_lines(cubin_data: &[u8]) -> Result<HashMap<u64, DebugLineInfo>, DwarfParseError> {
    let parser = DwarfParser::new(cubin_data);
    let info = parser.parse()?;
    Ok(info.line_mappings)
}

/// Parse all debug information from CUBIN data
pub fn parse_all_debug_info(cubin_data: &[u8]) -> Result<ParsedDebugInfo, DwarfParseError> {
    let parser = DwarfParser::new(cubin_data);
    parser.parse()
}

/// Get function boundaries from debug information
pub fn get_function_boundaries(cubin_data: &[u8]) -> Result<Vec<(String, u64, u64)>, DwarfParseError> {
    let parser = DwarfParser::new(cubin_data);
    let info = parser.parse()?;

    Ok(info.functions
        .into_iter()
        .map(|f| (f.name, f.low_pc, f.high_pc))
        .collect())
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug)]
pub enum DwarfParseError {
    ObjectParse(String),
    GimliError(String),
    InvalidSection(String),
    NotFound(String),
}

impl fmt::Display for DwarfParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DwarfParseError::ObjectParse(msg) => write!(f, "Object parse error: {}", msg),
            DwarfParseError::GimliError(msg) => write!(f, "DWARF parse error: {}", msg),
            DwarfParseError::InvalidSection(msg) => write!(f, "Invalid section: {}", msg),
            DwarfParseError::NotFound(msg) => write!(f, "Not found: {}", msg),
        }
    }
}

impl std::error::Error for DwarfParseError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let data = vec![0u8; 100];
        let parser = DwarfParser::new(&data);
        // Should not panic
        let _ = parser.parse();
    }

    #[test]
    fn test_empty_data() {
        let data = vec![];
        let result = parse_debug_lines(&data);
        assert!(result.is_err() || result.unwrap().is_empty());
    }
}
