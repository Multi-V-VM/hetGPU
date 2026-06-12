//! Native CUBIN (CUDA Binary) ELF parser
//!
//! This module provides direct parsing of CUBIN files without relying on
//! external tools like cuobjdump. It extracts:
//! - SASS code from .text sections
//! - Debug information from DWARF sections
//! - Symbol table for function boundaries
//! - NVIDIA-specific sections (.nv.info, .nv.constant, etc.)

use object::{
    read::elf::{ElfFile, FileHeader, SectionHeader},
    Endianness, Object, ObjectSection, ObjectSymbol, SectionKind, SymbolKind,
};
use std::collections::HashMap;
use std::fmt;

use super::instruction::*;

// ============================================================================
// CUBIN Structure Constants
// ============================================================================

/// NVIDIA-specific ELF section types
pub mod nv_section_types {
    pub const SHT_CUDA_INFO: u32 = 0x70000000; // .nv.info
    pub const SHT_CUDA_CALLGRAPH: u32 = 0x70000001; // .nv.callgraph
    pub const SHT_CUDA_RELOCINFO: u32 = 0x70000002; // .nv.rel.*
}

/// NVIDIA-specific symbol types
pub mod nv_symbol_types {
    pub const STT_CUDA_FUNC: u32 = 10;
    pub const STT_CUDA_OBJECT: u32 = 11;
    pub const STT_CUDA_TEXTURE: u32 = 12;
    pub const STT_CUDA_SURFACE: u32 = 13;
    pub const STT_CUDA_SAMPLER: u32 = 14;
}

// ============================================================================
// CUBIN Parser Results
// ============================================================================

/// A parsed CUDA kernel function
#[derive(Debug, Clone)]
pub struct CubinKernel {
    /// Kernel name
    pub name: String,
    /// Start address in .text section
    pub address: u64,
    /// Size in bytes
    pub size: u64,
    /// Raw SASS code bytes
    pub code: Vec<u8>,
    /// Number of registers used
    pub num_registers: u32,
    /// Shared memory size
    pub shared_mem_size: u32,
    /// Constant memory size
    pub const_mem_size: u32,
    /// Local memory size per thread
    pub local_mem_size: u32,
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// SM version this kernel was compiled for
    pub sm_version: u32,
}

/// A parsed constant buffer
#[derive(Debug, Clone)]
pub struct CubinConstant {
    /// Constant buffer index (0-17)
    pub bank: u32,
    /// Offset within the bank
    pub offset: u32,
    /// Size in bytes
    pub size: u32,
    /// Name if available
    pub name: Option<String>,
    /// Raw data
    pub data: Vec<u8>,
}

/// Complete parsed CUBIN file
#[derive(Debug)]
pub struct ParsedCubin {
    /// SM architecture version (e.g., 61 for sm_61, 86 for sm_86)
    pub sm_version: u32,
    /// PTX version if embedded
    pub ptx_version: Option<String>,
    /// All kernels in this CUBIN
    pub kernels: Vec<CubinKernel>,
    /// Global constants
    pub constants: Vec<CubinConstant>,
    /// Debug line information (address -> source location)
    pub debug_lines: HashMap<u64, DebugLineInfo>,
    /// Symbol table
    pub symbols: Vec<CubinSymbol>,
    /// Raw section data for advanced analysis
    pub sections: HashMap<String, CubinSection>,
}

/// Debug line information entry
#[derive(Debug, Clone)]
pub struct DebugLineInfo {
    pub address: u64,
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub is_statement: bool,
    pub discriminator: u32,
}

/// Symbol from CUBIN
#[derive(Debug, Clone)]
pub struct CubinSymbol {
    pub name: String,
    pub address: u64,
    pub size: u64,
    pub symbol_type: CubinSymbolType,
    pub section: Option<String>,
    pub is_global: bool,
}

/// Type of CUBIN symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubinSymbolType {
    Function,
    Variable,
    Texture,
    Surface,
    Sampler,
    Section,
    Unknown,
}

/// Raw section data
#[derive(Debug, Clone)]
pub struct CubinSection {
    pub name: String,
    pub address: u64,
    pub size: u64,
    pub data: Vec<u8>,
    pub section_type: u32,
    pub flags: u64,
}

// ============================================================================
// CUBIN Parser
// ============================================================================

/// CUBIN file parser
pub struct CubinParser {
    /// Raw CUBIN data
    data: Vec<u8>,
}

impl CubinParser {
    /// Create a new parser from raw CUBIN data
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Load CUBIN from file
    pub fn from_file(path: &str) -> Result<Self, CubinParseError> {
        let data = std::fs::read(path).map_err(|e| CubinParseError::IoError(e.to_string()))?;
        Ok(Self::new(data))
    }

    /// Parse the CUBIN file
    pub fn parse(&self) -> Result<ParsedCubin, CubinParseError> {
        // Parse as 64-bit ELF (CUBIN is always 64-bit ELF)
        let elf = ElfFile::<object::elf::FileHeader64<Endianness>>::parse(&self.data[..])
            .map_err(|e| CubinParseError::ElfParseError(e.to_string()))?;

        // Extract SM version from ELF flags
        let sm_version = self.extract_sm_version(&elf)?;

        // Parse sections
        let sections = self.parse_sections(&elf)?;

        // Parse symbols
        let symbols = self.parse_symbols(&elf)?;

        // Extract kernels from .text sections
        let kernels = self.extract_kernels(&elf, &sections, &symbols, sm_version)?;

        // Extract constants
        let constants = self.extract_constants(&sections)?;

        // Parse debug info if available
        let debug_lines = self.parse_debug_lines(&elf)?;

        Ok(ParsedCubin {
            sm_version,
            ptx_version: self.find_ptx_version(&sections),
            kernels,
            constants,
            debug_lines,
            symbols,
            sections,
        })
    }

    /// Extract SM version from ELF header flags
    fn extract_sm_version<'a, F: FileHeader>(
        &self,
        elf: &ElfFile<'a, F>,
    ) -> Result<u32, CubinParseError> {
        // NVIDIA encodes SM version in ELF flags
        // The format varies by CUDA version, but generally:
        // flags & 0xFF gives the SM version (e.g., 0x3D = 61 for sm_61)

        // For now, try to detect from section names
        for section in elf.sections() {
            if let Ok(name) = section.name() {
                // .text.kernel_name sections indicate this is a valid CUBIN
                if name.starts_with(".text.") {
                    // Check for .nv.info section which contains SM info
                }
            }
        }

        // Default to sm_61 if we can't determine
        // In a real implementation, we'd parse the .nv.info section
        Ok(61)
    }

    /// Parse all ELF sections
    fn parse_sections<'a, F: FileHeader>(
        &self,
        elf: &ElfFile<'a, F>,
    ) -> Result<HashMap<String, CubinSection>, CubinParseError> {
        let mut sections = HashMap::new();

        for section in elf.sections() {
            let name = section
                .name()
                .map_err(|e| CubinParseError::SectionError(e.to_string()))?
                .to_string();

            if name.is_empty() {
                continue;
            }

            let data = section
                .data()
                .map_err(|e| CubinParseError::SectionError(e.to_string()))?
                .to_vec();

            sections.insert(
                name.clone(),
                CubinSection {
                    name,
                    address: section.address(),
                    size: section.size(),
                    data,
                    section_type: 0, // Would need raw section header access
                    flags: 0,
                },
            );
        }

        Ok(sections)
    }

    /// Parse symbol table
    fn parse_symbols<'a, F: FileHeader>(
        &self,
        elf: &ElfFile<'a, F>,
    ) -> Result<Vec<CubinSymbol>, CubinParseError> {
        let mut symbols = Vec::new();

        for symbol in elf.symbols() {
            let name = match symbol.name() {
                Ok(n) if !n.is_empty() => n.to_string(),
                _ => continue,
            };

            let symbol_type = match symbol.kind() {
                SymbolKind::Text => CubinSymbolType::Function,
                SymbolKind::Data => CubinSymbolType::Variable,
                SymbolKind::Section => CubinSymbolType::Section,
                _ => CubinSymbolType::Unknown,
            };

            let section_name = symbol
                .section_index()
                .and_then(|idx| elf.section_by_index(idx).ok())
                .and_then(|s| s.name().ok().map(|n| n.to_string()));

            symbols.push(CubinSymbol {
                name,
                address: symbol.address(),
                size: symbol.size(),
                symbol_type,
                section: section_name,
                is_global: symbol.is_global(),
            });
        }

        Ok(symbols)
    }

    /// Extract kernel information
    fn extract_kernels<'a, F: FileHeader>(
        &self,
        elf: &ElfFile<'a, F>,
        sections: &HashMap<String, CubinSection>,
        symbols: &[CubinSymbol],
        sm_version: u32,
    ) -> Result<Vec<CubinKernel>, CubinParseError> {
        let mut kernels = Vec::new();

        // Find all .text.* sections (each contains a kernel)
        for (name, section) in sections {
            if name.starts_with(".text.") && section.data.len() > 0 {
                let kernel_name = name.strip_prefix(".text.").unwrap_or(name);

                // Find corresponding symbol for more info
                let kernel_symbol = symbols
                    .iter()
                    .find(|s| s.name == kernel_name || s.section.as_deref() == Some(name));

                // Try to get kernel metadata from .nv.info.{kernel_name}
                let nv_info_name = format!(".nv.info.{}", kernel_name);
                let kernel_info = sections.get(&nv_info_name);

                let (num_registers, shared_mem, const_mem, local_mem, max_threads) =
                    if let Some(info) = kernel_info {
                        self.parse_nv_info(&info.data)
                    } else {
                        (32, 0, 0, 0, 1024) // Defaults
                    };

                kernels.push(CubinKernel {
                    name: kernel_name.to_string(),
                    address: section.address,
                    size: section.size,
                    code: section.data.clone(),
                    num_registers: num_registers,
                    shared_mem_size: shared_mem,
                    const_mem_size: const_mem,
                    local_mem_size: local_mem,
                    max_threads_per_block: max_threads,
                    sm_version,
                });
            }
        }

        Ok(kernels)
    }

    /// Parse .nv.info section to extract kernel metadata
    fn parse_nv_info(&self, data: &[u8]) -> (u32, u32, u32, u32, u32) {
        // .nv.info format is a series of attribute entries:
        // Each entry: attribute_type (2 bytes) + size (2 bytes) + data (size bytes)

        let mut num_regs = 32u32;
        let mut shared_mem = 0u32;
        let mut const_mem = 0u32;
        let mut local_mem = 0u32;
        let mut max_threads = 1024u32;

        let mut offset = 0;
        while offset + 4 <= data.len() {
            let attr_type = u16::from_le_bytes([data[offset], data[offset + 1]]);
            let attr_size = u16::from_le_bytes([data[offset + 2], data[offset + 3]]) as usize;
            offset += 4;

            if offset + attr_size > data.len() {
                break;
            }

            let attr_data = &data[offset..offset + attr_size];
            offset += attr_size;

            // Align to 4 bytes
            offset = (offset + 3) & !3;

            // Parse known attributes
            match attr_type {
                0x0401 => {
                    // EIATTR_REGCOUNT
                    if attr_data.len() >= 4 {
                        num_regs = u32::from_le_bytes([
                            attr_data[0],
                            attr_data[1],
                            attr_data[2],
                            attr_data[3],
                        ]);
                    }
                }
                0x0808 => {
                    // EIATTR_SMEM_SIZE
                    if attr_data.len() >= 4 {
                        shared_mem = u32::from_le_bytes([
                            attr_data[0],
                            attr_data[1],
                            attr_data[2],
                            attr_data[3],
                        ]);
                    }
                }
                0x0A04 => {
                    // EIATTR_LMEM_SIZE
                    if attr_data.len() >= 4 {
                        local_mem = u32::from_le_bytes([
                            attr_data[0],
                            attr_data[1],
                            attr_data[2],
                            attr_data[3],
                        ]);
                    }
                }
                0x0504 => {
                    // EIATTR_MAX_THREADS
                    if attr_data.len() >= 4 {
                        max_threads = u32::from_le_bytes([
                            attr_data[0],
                            attr_data[1],
                            attr_data[2],
                            attr_data[3],
                        ]);
                    }
                }
                _ => {}
            }
        }

        (num_regs, shared_mem, const_mem, local_mem, max_threads)
    }

    /// Extract constant memory sections
    fn extract_constants(
        &self,
        sections: &HashMap<String, CubinSection>,
    ) -> Result<Vec<CubinConstant>, CubinParseError> {
        let mut constants = Vec::new();

        // Look for .nv.constant* sections
        for (name, section) in sections {
            if name.starts_with(".nv.constant") {
                // Parse constant bank from name: .nv.constant0, .nv.constant2, etc.
                let bank = name
                    .strip_prefix(".nv.constant")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);

                constants.push(CubinConstant {
                    bank,
                    offset: 0,
                    size: section.size as u32,
                    name: Some(name.clone()),
                    data: section.data.clone(),
                });
            }
        }

        Ok(constants)
    }

    /// Find embedded PTX version
    fn find_ptx_version(&self, sections: &HashMap<String, CubinSection>) -> Option<String> {
        // PTX may be embedded in .nv.info section or as a separate section
        for (name, section) in sections {
            if name.contains("ptx") {
                // Try to extract version string
                if let Ok(s) = std::str::from_utf8(&section.data) {
                    if let Some(ver) = s.lines().find(|l| l.starts_with(".version")) {
                        return Some(ver.to_string());
                    }
                }
            }
        }
        None
    }

    /// Parse DWARF debug line information
    fn parse_debug_lines<'a, F: FileHeader>(
        &self,
        elf: &ElfFile<'a, F>,
    ) -> Result<HashMap<u64, DebugLineInfo>, CubinParseError> {
        // This will be delegated to the dwarf_parser module
        // For now, return empty map
        Ok(HashMap::new())
    }

    /// Get raw SASS code for a kernel
    pub fn get_kernel_code<'a>(
        &self,
        parsed: &'a ParsedCubin,
        kernel_name: &str,
    ) -> Option<&'a [u8]> {
        parsed
            .kernels
            .iter()
            .find(|k| k.name == kernel_name)
            .map(|k| k.code.as_slice())
    }

    /// Get all kernel names
    pub fn get_kernel_names(parsed: &ParsedCubin) -> Vec<&str> {
        parsed.kernels.iter().map(|k| k.name.as_str()).collect()
    }
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug)]
pub enum CubinParseError {
    IoError(String),
    ElfParseError(String),
    SectionError(String),
    InvalidFormat(String),
    UnsupportedVersion(u32),
}

impl fmt::Display for CubinParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CubinParseError::IoError(msg) => write!(f, "IO error: {}", msg),
            CubinParseError::ElfParseError(msg) => write!(f, "ELF parse error: {}", msg),
            CubinParseError::SectionError(msg) => write!(f, "Section error: {}", msg),
            CubinParseError::InvalidFormat(msg) => write!(f, "Invalid CUBIN format: {}", msg),
            CubinParseError::UnsupportedVersion(ver) => {
                write!(f, "Unsupported SM version: {}", ver)
            }
        }
    }
}

impl std::error::Error for CubinParseError {}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if data appears to be a valid CUBIN file
pub fn is_cubin(data: &[u8]) -> bool {
    // Check ELF magic number
    if data.len() < 16 {
        return false;
    }

    // ELF magic: 0x7F 'E' 'L' 'F'
    if &data[0..4] != b"\x7FELF" {
        return false;
    }

    // Check for 64-bit ELF (CUBIN is always 64-bit)
    if data[4] != 2 {
        return false;
    }

    // Check for little endian (NVIDIA GPUs are little endian)
    if data[5] != 1 {
        return false;
    }

    // Check machine type - NVIDIA uses custom value 190 (0xBE)
    let machine = u16::from_le_bytes([data[18], data[19]]);
    if machine != 190 {
        // Also accept EM_CUDA (0xBE = 190)
        return false;
    }

    true
}

/// Extract SM version from CUBIN filename (e.g., "kernel.sm_86.cubin" -> 86)
pub fn sm_version_from_filename(filename: &str) -> Option<u32> {
    let parts: Vec<&str> = filename.split('.').collect();
    for part in parts {
        if part.starts_with("sm_") {
            return part.strip_prefix("sm_")?.parse().ok();
        }
    }
    None
}

// ============================================================================
// Convenience Types
// ============================================================================

/// Builder for CubinParser with common configurations
pub struct CubinParserBuilder {
    data: Option<Vec<u8>>,
    parse_debug: bool,
    parse_constants: bool,
}

impl CubinParserBuilder {
    pub fn new() -> Self {
        Self {
            data: None,
            parse_debug: true,
            parse_constants: true,
        }
    }

    pub fn data(mut self, data: Vec<u8>) -> Self {
        self.data = Some(data);
        self
    }

    pub fn from_file(mut self, path: &str) -> Result<Self, CubinParseError> {
        let data = std::fs::read(path).map_err(|e| CubinParseError::IoError(e.to_string()))?;
        self.data = Some(data);
        Ok(self)
    }

    pub fn parse_debug(mut self, enable: bool) -> Self {
        self.parse_debug = enable;
        self
    }

    pub fn parse_constants(mut self, enable: bool) -> Self {
        self.parse_constants = enable;
        self
    }

    pub fn build(self) -> Result<CubinParser, CubinParseError> {
        let data = self
            .data
            .ok_or_else(|| CubinParseError::InvalidFormat("No data provided".to_string()))?;

        if !is_cubin(&data) {
            return Err(CubinParseError::InvalidFormat(
                "Not a valid CUBIN file".to_string(),
            ));
        }

        Ok(CubinParser::new(data))
    }
}

impl Default for CubinParserBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_cubin() {
        // Valid ELF header for CUBIN
        let mut cubin_header = vec![0u8; 64];
        cubin_header[0..4].copy_from_slice(b"\x7FELF");
        cubin_header[4] = 2; // 64-bit
        cubin_header[5] = 1; // Little endian
        cubin_header[18] = 190; // EM_CUDA
        cubin_header[19] = 0;

        assert!(is_cubin(&cubin_header));

        // Invalid: wrong magic
        let not_cubin = b"not a cubin file";
        assert!(!is_cubin(not_cubin));

        // Invalid: wrong machine type
        let mut wrong_machine = cubin_header.clone();
        wrong_machine[18] = 0x3E; // x86_64
        assert!(!is_cubin(&wrong_machine));
    }

    #[test]
    fn test_sm_version_from_filename() {
        assert_eq!(sm_version_from_filename("kernel.sm_86.cubin"), Some(86));
        assert_eq!(sm_version_from_filename("test.sm_61.cubin"), Some(61));
        assert_eq!(sm_version_from_filename("kernel.cubin"), None);
    }

    #[test]
    fn test_parse_nv_info() {
        let parser = CubinParser::new(vec![]);

        // Create mock .nv.info data with REGCOUNT attribute
        let mut nv_info = Vec::new();
        // EIATTR_REGCOUNT (0x0401), size 4, value 48
        nv_info.extend_from_slice(&[0x01, 0x04]); // attr_type
        nv_info.extend_from_slice(&[0x04, 0x00]); // size
        nv_info.extend_from_slice(&[48, 0, 0, 0]); // value

        let (regs, _, _, _, _) = parser.parse_nv_info(&nv_info);
        assert_eq!(regs, 48);
    }
}
