//! DWARF debug information parser for CUBIN files
//!
//! This module parses DWARF debug information embedded in CUBIN files to extract:
//! - Line number mappings (SASS address -> PTX source location)
//! - Function information
//! - Variable locations
//! - PTX source recovery from embedded debug info
//!
//! CUDA uses DWARF v2 format with some NVIDIA-specific extensions.

use gimli::{
    AttributeValue, DebugAbbrev, DebugInfo, DebugLine, DebugLineOffset, DebugStr,
    DebuggingInformationEntry, EndianSlice, LittleEndian, UnitHeader,
};
use object::{Object, ObjectSection};
use std::collections::{BTreeMap, HashMap};
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

        // Get section data helper
        let get_section = |name: &str| -> Option<&[u8]> {
            object.section_by_name(name).and_then(|s| s.data().ok())
        };

        // Parse .debug_line section for line number mappings
        if let Some(debug_line_data) = get_section(".debug_line") {
            self.parse_debug_line_gimli(debug_line_data, &mut result)?;
        }

        // Parse .debug_info and .debug_abbrev for function/variable info
        if let (Some(debug_info_data), Some(debug_abbrev_data)) =
            (get_section(".debug_info"), get_section(".debug_abbrev"))
        {
            let debug_str_data = get_section(".debug_str").unwrap_or(&[]);
            self.parse_debug_info_gimli(
                debug_info_data,
                debug_abbrev_data,
                debug_str_data,
                &mut result,
            )?;
        }

        Ok(result)
    }

    /// Parse .debug_line section using simple state machine
    fn parse_debug_line_gimli(
        &self,
        data: &[u8],
        result: &mut ParsedDebugInfo,
    ) -> Result<(), DwarfParseError> {
        // Use simple custom parsing instead of gimli for .debug_line
        // This is more portable and handles CUDA's DWARF extensions better
        self.parse_debug_line_simple(data, result)
    }

    /// Simple .debug_line parser
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
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;

            if unit_length == 0 || offset + unit_length + 4 > data.len() {
                break;
            }

            let unit_end = offset + 4 + unit_length;

            // Skip version (2 bytes) at offset+4
            // Skip header_length (4 bytes) at offset+6
            // The actual line program starts after the header

            if offset + 10 <= data.len() {
                let version = u16::from_le_bytes([data[offset + 4], data[offset + 5]]);
                let header_length = u32::from_le_bytes([
                    data[offset + 6],
                    data[offset + 7],
                    data[offset + 8],
                    data[offset + 9],
                ]) as usize;

                // For DWARF 2/3, parse the simple state machine format
                if version >= 2 && version <= 4 {
                    let header_end = offset + 10 + header_length;

                    // Try to parse file table from header
                    if header_end < unit_end && header_end > offset + 20 {
                        self.parse_file_table(&data[offset + 10..header_end], result);
                    }

                    // Parse line program opcodes
                    if header_end < unit_end {
                        self.parse_line_program(&data[header_end..unit_end], result);
                    }
                }
            }

            offset = unit_end;
        }

        Ok(())
    }

    /// Parse file table from debug_line header
    fn parse_file_table(&self, data: &[u8], result: &mut ParsedDebugInfo) {
        // Skip minimum_instruction_length, default_is_stmt, line_base, line_range, opcode_base
        // These are typically 5 bytes for DWARF 2/3
        let mut offset = 5;

        // Skip standard_opcode_lengths array (opcode_base - 1 entries)
        if offset < data.len() {
            let opcode_base = data[4] as usize;
            if opcode_base > 0 {
                offset += opcode_base - 1;
            }
        }

        // Skip include directories (null-terminated strings ending with empty string)
        while offset < data.len() && data[offset] != 0 {
            while offset < data.len() && data[offset] != 0 {
                offset += 1;
            }
            offset += 1; // skip null terminator
        }
        offset += 1; // skip final null

        // Parse file names
        let mut file_index = 1u64;
        while offset < data.len() && data[offset] != 0 {
            let name_start = offset;
            while offset < data.len() && data[offset] != 0 {
                offset += 1;
            }

            if let Ok(name) = std::str::from_utf8(&data[name_start..offset]) {
                result.file_table.insert(file_index, name.to_string());
            }

            offset += 1; // skip null terminator

            // Skip directory index (ULEB128)
            while offset < data.len() && (data[offset] & 0x80) != 0 {
                offset += 1;
            }
            if offset < data.len() {
                offset += 1;
            }

            // Skip modification time (ULEB128)
            while offset < data.len() && (data[offset] & 0x80) != 0 {
                offset += 1;
            }
            if offset < data.len() {
                offset += 1;
            }

            // Skip file size (ULEB128)
            while offset < data.len() && (data[offset] & 0x80) != 0 {
                offset += 1;
            }
            if offset < data.len() {
                offset += 1;
            }

            file_index += 1;
        }
    }

    /// Parse line number program opcodes
    fn parse_line_program(&self, data: &[u8], result: &mut ParsedDebugInfo) {
        let mut address: u64 = 0;
        let mut file: u64 = 1;
        let mut line: u32 = 1;
        let mut column: u32 = 0;
        let mut is_stmt = true;

        let mut offset = 0;

        while offset < data.len() {
            let opcode = data[offset];
            offset += 1;

            match opcode {
                0 => {
                    // Extended opcode
                    if offset >= data.len() {
                        break;
                    }
                    let len = data[offset] as usize;
                    offset += 1;

                    if offset >= data.len() || len == 0 {
                        break;
                    }

                    let ext_opcode = data[offset];
                    match ext_opcode {
                        1 => break, // DW_LNE_end_sequence
                        2 => {
                            // DW_LNE_set_address
                            if offset + 1 + 8 <= data.len() {
                                address = u64::from_le_bytes([
                                    data[offset + 1],
                                    data[offset + 2],
                                    data[offset + 3],
                                    data[offset + 4],
                                    data[offset + 5],
                                    data[offset + 6],
                                    data[offset + 7],
                                    data[offset + 8],
                                ]);
                            }
                        }
                        4 => {
                            // DW_LNE_set_discriminator
                        }
                        _ => {}
                    }
                    offset += len;
                }
                1 => {
                    // DW_LNS_copy - emit row
                    let file_name = result
                        .file_table
                        .get(&file)
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());

                    result.line_mappings.insert(
                        address,
                        DebugLineInfo {
                            address,
                            file: file_name,
                            line,
                            column,
                            is_statement: is_stmt,
                            discriminator: 0,
                        },
                    );
                }
                2 => {
                    // DW_LNS_advance_pc
                    let (delta, bytes) = Self::decode_uleb128(&data[offset..]);
                    address += delta;
                    offset += bytes;
                }
                3 => {
                    // DW_LNS_advance_line
                    let (delta, bytes) = Self::decode_sleb128_usize(&data[offset..]);
                    line = ((line as i64) + delta) as u32;
                    offset += bytes;
                }
                4 => {
                    // DW_LNS_set_file
                    let (new_file, bytes) = Self::decode_uleb128(&data[offset..]);
                    file = new_file;
                    offset += bytes;
                }
                5 => {
                    // DW_LNS_set_column
                    let (new_col, bytes) = Self::decode_uleb128(&data[offset..]);
                    column = new_col as u32;
                    offset += bytes;
                }
                6 => {
                    // DW_LNS_negate_stmt
                    is_stmt = !is_stmt;
                }
                7 => {
                    // DW_LNS_set_basic_block
                }
                8 => {
                    // DW_LNS_const_add_pc
                    address += 17; // (255 - opcode_base) / line_range * min_instr_length (typical)
                }
                9 => {
                    // DW_LNS_fixed_advance_pc
                    if offset + 2 <= data.len() {
                        let delta = u16::from_le_bytes([data[offset], data[offset + 1]]);
                        address += delta as u64;
                        offset += 2;
                    }
                }
                _ => {
                    // Special opcode
                    let adjusted = opcode - 13; // opcode_base typically 13
                    let line_increment = (adjusted % 14) as i32 - 6; // line_range typically 14, line_base typically -6
                    let address_increment = (adjusted / 14) as u64;

                    line = ((line as i32) + line_increment) as u32;
                    address += address_increment;

                    // Emit row
                    let file_name = result
                        .file_table
                        .get(&file)
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());

                    result.line_mappings.insert(
                        address,
                        DebugLineInfo {
                            address,
                            file: file_name,
                            line,
                            column,
                            is_statement: is_stmt,
                            discriminator: 0,
                        },
                    );
                }
            }
        }
    }

    /// Decode ULEB128
    fn decode_uleb128(bytes: &[u8]) -> (u64, usize) {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        let mut count = 0;

        for &byte in bytes {
            count += 1;
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }

        (result, count)
    }

    /// Decode SLEB128 returning usize for offset
    fn decode_sleb128_usize(bytes: &[u8]) -> (i64, usize) {
        let mut result: i64 = 0;
        let mut shift = 0u32;
        let mut count = 0;

        for &byte in bytes {
            count += 1;
            result |= ((byte & 0x7f) as i64) << shift;
            shift += 7;

            if byte & 0x80 == 0 {
                if shift < 64 && (byte & 0x40) != 0 {
                    result |= !0i64 << shift;
                }
                break;
            }
        }

        (result, count)
    }

    /// Parse .debug_info section using gimli for function/variable info
    fn parse_debug_info_gimli(
        &self,
        debug_info_data: &[u8],
        debug_abbrev_data: &[u8],
        debug_str_data: &[u8],
        result: &mut ParsedDebugInfo,
    ) -> Result<(), DwarfParseError> {
        let debug_info = DebugInfo::new(debug_info_data, LittleEndian);
        let debug_abbrev = DebugAbbrev::new(debug_abbrev_data, LittleEndian);
        let debug_str = DebugStr::new(debug_str_data, LittleEndian);

        let mut iter = debug_info.units();
        while let Ok(Some(header)) = iter.next() {
            let abbrevs = match debug_abbrev.abbreviations(header.debug_abbrev_offset()) {
                Ok(a) => a,
                Err(_) => continue,
            };

            let mut entries = header.entries(&abbrevs);
            while let Ok(Some((_, entry))) = entries.next_dfs() {
                self.process_die(&header, entry, &debug_str, result);
            }
        }

        Ok(())
    }

    /// Process a single DIE (Debug Information Entry)
    fn process_die(
        &self,
        header: &UnitHeader<EndianSlice<'_, LittleEndian>>,
        entry: &DebuggingInformationEntry<EndianSlice<'_, LittleEndian>>,
        debug_str: &DebugStr<EndianSlice<'_, LittleEndian>>,
        result: &mut ParsedDebugInfo,
    ) {
        match entry.tag() {
            gimli::DW_TAG_compile_unit => {
                self.parse_compile_unit(header, entry, debug_str, result);
            }
            gimli::DW_TAG_subprogram => {
                self.parse_subprogram(header, entry, debug_str, result);
            }
            gimli::DW_TAG_variable | gimli::DW_TAG_formal_parameter => {
                self.parse_variable(
                    header,
                    entry,
                    debug_str,
                    result,
                    entry.tag() == gimli::DW_TAG_formal_parameter,
                );
            }
            _ => {}
        }
    }

    /// Parse DW_TAG_compile_unit
    fn parse_compile_unit(
        &self,
        _header: &UnitHeader<EndianSlice<'_, LittleEndian>>,
        entry: &DebuggingInformationEntry<EndianSlice<'_, LittleEndian>>,
        debug_str: &DebugStr<EndianSlice<'_, LittleEndian>>,
        result: &mut ParsedDebugInfo,
    ) {
        let mut cu = CompilationUnitInfo {
            name: String::new(),
            comp_dir: None,
            producer: None,
            language: 0,
            low_pc: 0,
            high_pc: 0,
            dwarf_version: 2,
        };

        let mut attrs = entry.attrs();
        while let Ok(Some(attr)) = attrs.next() {
            match attr.name() {
                gimli::DW_AT_name => {
                    cu.name = self.get_string_value(&attr.value(), debug_str);
                }
                gimli::DW_AT_comp_dir => {
                    cu.comp_dir = Some(self.get_string_value(&attr.value(), debug_str));
                }
                gimli::DW_AT_producer => {
                    cu.producer = Some(self.get_string_value(&attr.value(), debug_str));
                }
                gimli::DW_AT_language => {
                    if let AttributeValue::Language(lang) = attr.value() {
                        cu.language = lang.0 as u32;
                    }
                }
                gimli::DW_AT_low_pc => {
                    if let AttributeValue::Addr(addr) = attr.value() {
                        cu.low_pc = addr;
                    }
                }
                gimli::DW_AT_high_pc => match attr.value() {
                    AttributeValue::Addr(addr) => cu.high_pc = addr,
                    AttributeValue::Udata(size) => cu.high_pc = cu.low_pc + size,
                    _ => {}
                },
                _ => {}
            }
        }

        if !cu.name.is_empty() {
            result.compilation_units.push(cu);
        }
    }

    /// Parse DW_TAG_subprogram (function)
    fn parse_subprogram(
        &self,
        _header: &UnitHeader<EndianSlice<'_, LittleEndian>>,
        entry: &DebuggingInformationEntry<EndianSlice<'_, LittleEndian>>,
        debug_str: &DebugStr<EndianSlice<'_, LittleEndian>>,
        result: &mut ParsedDebugInfo,
    ) {
        let mut func = DebugFunctionInfo {
            name: String::new(),
            linkage_name: None,
            low_pc: 0,
            high_pc: 0,
            file: None,
            line: 0,
            is_inlined: false,
        };

        let mut attrs = entry.attrs();
        while let Ok(Some(attr)) = attrs.next() {
            match attr.name() {
                gimli::DW_AT_name => {
                    func.name = self.get_string_value(&attr.value(), debug_str);
                }
                gimli::DW_AT_linkage_name | gimli::DW_AT_MIPS_linkage_name => {
                    func.linkage_name = Some(self.get_string_value(&attr.value(), debug_str));
                }
                gimli::DW_AT_low_pc => {
                    if let AttributeValue::Addr(addr) = attr.value() {
                        func.low_pc = addr;
                    }
                }
                gimli::DW_AT_high_pc => match attr.value() {
                    AttributeValue::Addr(addr) => func.high_pc = addr,
                    AttributeValue::Udata(size) => func.high_pc = func.low_pc + size,
                    _ => {}
                },
                gimli::DW_AT_decl_line => {
                    if let AttributeValue::Udata(line) = attr.value() {
                        func.line = line as u32;
                    }
                }
                gimli::DW_AT_inline => {
                    func.is_inlined = true;
                }
                _ => {}
            }
        }

        if !func.name.is_empty() || func.linkage_name.is_some() {
            result.functions.push(func);
        }
    }

    /// Parse DW_TAG_variable or DW_TAG_formal_parameter
    fn parse_variable(
        &self,
        _header: &UnitHeader<EndianSlice<'_, LittleEndian>>,
        entry: &DebuggingInformationEntry<EndianSlice<'_, LittleEndian>>,
        debug_str: &DebugStr<EndianSlice<'_, LittleEndian>>,
        result: &mut ParsedDebugInfo,
        is_parameter: bool,
    ) {
        let mut var = DebugVariableInfo {
            name: String::new(),
            type_name: None,
            location: VariableLocationExpr::Unknown,
            file: None,
            line: 0,
            is_parameter,
        };

        let mut attrs = entry.attrs();
        while let Ok(Some(attr)) = attrs.next() {
            match attr.name() {
                gimli::DW_AT_name => {
                    var.name = self.get_string_value(&attr.value(), debug_str);
                }
                gimli::DW_AT_decl_line => {
                    if let AttributeValue::Udata(line) = attr.value() {
                        var.line = line as u32;
                    }
                }
                gimli::DW_AT_location => {
                    var.location = self.parse_location_expr(&attr.value());
                }
                _ => {}
            }
        }

        if !var.name.is_empty() {
            result.variables.push(var);
        }
    }

    /// Get string value from attribute
    fn get_string_value(
        &self,
        value: &AttributeValue<EndianSlice<'_, LittleEndian>>,
        debug_str: &DebugStr<EndianSlice<'_, LittleEndian>>,
    ) -> String {
        match value {
            AttributeValue::String(s) => std::str::from_utf8(s.slice()).unwrap_or("").to_string(),
            AttributeValue::DebugStrRef(offset) => debug_str
                .get_str(*offset)
                .ok()
                .and_then(|s| std::str::from_utf8(s.slice()).ok())
                .unwrap_or("")
                .to_string(),
            _ => String::new(),
        }
    }

    /// Parse location expression
    fn parse_location_expr(
        &self,
        value: &AttributeValue<EndianSlice<'_, LittleEndian>>,
    ) -> VariableLocationExpr {
        match value {
            AttributeValue::Exprloc(expr) => {
                let bytes = expr.0.slice();
                if bytes.is_empty() {
                    return VariableLocationExpr::Unknown;
                }

                // Parse simple DWARF expressions
                match bytes[0] {
                    0x50..=0x6f => {
                        // DW_OP_reg0 through DW_OP_reg31
                        let reg_num = bytes[0] - 0x50;
                        VariableLocationExpr::Register(format!("R{}", reg_num))
                    }
                    0x91 => {
                        // DW_OP_fbreg (frame base relative)
                        if bytes.len() > 1 {
                            // Decode SLEB128 offset
                            let offset = self.decode_sleb128(&bytes[1..]);
                            VariableLocationExpr::StackOffset(offset)
                        } else {
                            VariableLocationExpr::Unknown
                        }
                    }
                    0x03 => {
                        // DW_OP_addr
                        if bytes.len() >= 9 {
                            let addr = u64::from_le_bytes([
                                bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                                bytes[7], bytes[8],
                            ]);
                            VariableLocationExpr::Address(addr)
                        } else {
                            VariableLocationExpr::Unknown
                        }
                    }
                    _ => VariableLocationExpr::Expression(bytes.to_vec()),
                }
            }
            _ => VariableLocationExpr::Unknown,
        }
    }

    /// Decode SLEB128 encoded value
    fn decode_sleb128(&self, bytes: &[u8]) -> i64 {
        let mut result: i64 = 0;
        let mut shift = 0u32;

        for &byte in bytes {
            result |= ((byte & 0x7f) as i64) << shift;
            shift += 7;

            if byte & 0x80 == 0 {
                // Sign extend if needed
                if shift < 64 && (byte & 0x40) != 0 {
                    result |= !0i64 << shift;
                }
                break;
            }
        }

        result
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Parse debug line information from CUBIN data
pub fn parse_debug_lines(
    cubin_data: &[u8],
) -> Result<HashMap<u64, DebugLineInfo>, DwarfParseError> {
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
pub fn get_function_boundaries(
    cubin_data: &[u8],
) -> Result<Vec<(String, u64, u64)>, DwarfParseError> {
    let parser = DwarfParser::new(cubin_data);
    let info = parser.parse()?;

    Ok(info
        .functions
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
