//! Enhanced SASS instruction representation with semantic information
//!
//! This module provides a comprehensive SASS instruction model that includes:
//! - Full instruction encoding
//! - Semantic classification (opcode class, memory space, data type)
//! - PTX equivalence mapping
//! - Control code parsing (stall counts, barriers, reuse flags)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ============================================================================
// SASS Instruction Classification
// ============================================================================

/// SASS opcode classification by instruction category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SassOpcodeClass {
    // Arithmetic - Integer
    IntegerArithmetic, // IADD, IADD3, IMAD, IMUL, IABS, INEG, ...
    IntegerComparison, // ISETP, ICMP, ISET, ...
    IntegerLogical,    // LOP3, AND, OR, XOR, NOT, SHL, SHR, ...
    IntegerConversion, // I2I, I2F, F2I, ...

    // Arithmetic - Floating Point
    FloatArithmetic,     // FADD, FMUL, FFMA, FMNMX, FABS, FNEG, ...
    FloatComparison,     // FSETP, FCMP, FSET, ...
    FloatTranscendental, // MUFU (sin, cos, ex2, lg2, rcp, rsq, ...)
    FloatConversion,     // F2F, ...

    // Arithmetic - Double Precision
    DoubleArithmetic, // DADD, DMUL, DFMA, ...

    // Arithmetic - Half Precision
    HalfArithmetic, // HADD2, HMUL2, HFMA2, ...

    // Memory - Global
    GlobalLoad,   // LDG, LDG.E, LDG.U, ...
    GlobalStore,  // STG, STG.E, ...
    GlobalAtomic, // ATOMG, RED, ...

    // Memory - Shared
    SharedLoad,   // LDS, LDS.U, ...
    SharedStore,  // STS, ...
    SharedAtomic, // ATOMS, ...

    // Memory - Local
    LocalLoad,  // LDL, ...
    LocalStore, // STL, ...

    // Memory - Constant
    ConstantLoad, // LDC, ...

    // Memory - Texture/Surface
    TextureFetch,  // TEX, TLD, TLD4, ...
    TextureQuery,  // TXQ, ...
    SurfaceLoad,   // SULD, ...
    SurfaceStore,  // SUST, ...
    SurfaceAtomic, // SUATOM, ...

    // Control Flow
    Branch,            // BRA, JMP, ...
    ConditionalBranch, // @P BRA, BRX, ...
    Call,              // CAL, JCAL, ...
    Return,            // RET, ...
    Exit,              // EXIT, ...

    // Synchronization
    Barrier,       // BAR, DEPBAR, ...
    MemoryBarrier, // MEMBAR, ...

    // Special Registers
    SpecialRegRead,  // S2R (read %tid, %ctaid, %nctaid, %laneid, ...)
    SpecialRegWrite, // not common

    // Predicate Operations
    PredicateSet, // PSETP, PLOP3, ...

    // Data Movement
    Move,    // MOV, MOV32I, PRMT, SHFL, ...
    Shuffle, // SHFL, ...

    // Uniform Operations (SM 7.0+)
    UniformArithmetic, // UIADD, UFLO, etc.
    UniformDataPath,   // ULDC, ULEA, etc.

    // Tensor Core (SM 7.0+)
    TensorCore, // HMMA, IMMA, BMMA, ...

    // Miscellaneous
    NoOperation, // NOP
    ControlCode, // Scheduling/control instructions
    Unknown,
}

impl SassOpcodeClass {
    /// Check if this instruction class accesses memory
    pub fn is_memory_access(&self) -> bool {
        matches!(
            self,
            SassOpcodeClass::GlobalLoad
                | SassOpcodeClass::GlobalStore
                | SassOpcodeClass::GlobalAtomic
                | SassOpcodeClass::SharedLoad
                | SassOpcodeClass::SharedStore
                | SassOpcodeClass::SharedAtomic
                | SassOpcodeClass::LocalLoad
                | SassOpcodeClass::LocalStore
                | SassOpcodeClass::ConstantLoad
                | SassOpcodeClass::TextureFetch
                | SassOpcodeClass::SurfaceLoad
                | SassOpcodeClass::SurfaceStore
                | SassOpcodeClass::SurfaceAtomic
        )
    }

    /// Check if this instruction class is a control flow operation
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            SassOpcodeClass::Branch
                | SassOpcodeClass::ConditionalBranch
                | SassOpcodeClass::Call
                | SassOpcodeClass::Return
                | SassOpcodeClass::Exit
        )
    }

    /// Check if this instruction class is a synchronization operation
    pub fn is_synchronization(&self) -> bool {
        matches!(
            self,
            SassOpcodeClass::Barrier
                | SassOpcodeClass::MemoryBarrier
                | SassOpcodeClass::GlobalAtomic
                | SassOpcodeClass::SharedAtomic
                | SassOpcodeClass::SurfaceAtomic
        )
    }
}

/// Memory space for memory operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SassMemorySpace {
    Global,
    Shared,
    Local,
    Constant,
    Texture,
    Surface,
    Generic, // Address could be any space
}

impl fmt::Display for SassMemorySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SassMemorySpace::Global => write!(f, "global"),
            SassMemorySpace::Shared => write!(f, "shared"),
            SassMemorySpace::Local => write!(f, "local"),
            SassMemorySpace::Constant => write!(f, "const"),
            SassMemorySpace::Texture => write!(f, "tex"),
            SassMemorySpace::Surface => write!(f, "surf"),
            SassMemorySpace::Generic => write!(f, "generic"),
        }
    }
}

/// Data type for instruction operands
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SassDataType {
    // Unsigned integers
    U8,
    U16,
    U32,
    U64,
    U128,

    // Signed integers
    S8,
    S16,
    S32,
    S64,

    // Floating point
    F16,
    F32,
    F64,

    // Special
    BF16,    // Brain Float 16
    TF32,    // TensorFloat 32
    FP8E4M3, // 8-bit float (E4M3)
    FP8E5M2, // 8-bit float (E5M2)

    // Bit types (untyped)
    B8,
    B16,
    B32,
    B64,
    B128,

    // Predicate
    Pred,

    Unknown,
}

impl SassDataType {
    /// Get the size in bits
    pub fn size_bits(&self) -> u32 {
        match self {
            SassDataType::U8
            | SassDataType::S8
            | SassDataType::B8
            | SassDataType::FP8E4M3
            | SassDataType::FP8E5M2 => 8,
            SassDataType::U16
            | SassDataType::S16
            | SassDataType::F16
            | SassDataType::BF16
            | SassDataType::B16 => 16,
            SassDataType::U32
            | SassDataType::S32
            | SassDataType::F32
            | SassDataType::TF32
            | SassDataType::B32 => 32,
            SassDataType::U64 | SassDataType::S64 | SassDataType::F64 | SassDataType::B64 => 64,
            SassDataType::U128 | SassDataType::B128 => 128,
            SassDataType::Pred => 1,
            SassDataType::Unknown => 32,
        }
    }

    /// Parse from SASS modifier string (e.g., ".U32", ".F32", ".S16")
    pub fn from_modifier(modifier: &str) -> Option<Self> {
        let s = modifier.trim_start_matches('.');
        match s.to_uppercase().as_str() {
            "U8" => Some(SassDataType::U8),
            "U16" => Some(SassDataType::U16),
            "U32" => Some(SassDataType::U32),
            "U64" => Some(SassDataType::U64),
            "U128" => Some(SassDataType::U128),
            "S8" => Some(SassDataType::S8),
            "S16" => Some(SassDataType::S16),
            "S32" => Some(SassDataType::S32),
            "S64" => Some(SassDataType::S64),
            "F16" | "H" => Some(SassDataType::F16),
            "F32" | "F" => Some(SassDataType::F32),
            "F64" | "D" => Some(SassDataType::F64),
            "BF16" => Some(SassDataType::BF16),
            "TF32" => Some(SassDataType::TF32),
            "B8" => Some(SassDataType::B8),
            "B16" => Some(SassDataType::B16),
            "B32" => Some(SassDataType::B32),
            "B64" => Some(SassDataType::B64),
            "B128" => Some(SassDataType::B128),
            _ => None,
        }
    }

    /// Convert to PTX type suffix
    pub fn to_ptx_suffix(&self) -> &'static str {
        match self {
            SassDataType::U8 => "u8",
            SassDataType::U16 => "u16",
            SassDataType::U32 => "u32",
            SassDataType::U64 => "u64",
            SassDataType::U128 => "u128",
            SassDataType::S8 => "s8",
            SassDataType::S16 => "s16",
            SassDataType::S32 => "s32",
            SassDataType::S64 => "s64",
            SassDataType::F16 => "f16",
            SassDataType::F32 => "f32",
            SassDataType::F64 => "f64",
            SassDataType::BF16 => "bf16",
            SassDataType::TF32 => "tf32",
            SassDataType::FP8E4M3 => "e4m3",
            SassDataType::FP8E5M2 => "e5m2",
            SassDataType::B8 => "b8",
            SassDataType::B16 => "b16",
            SassDataType::B32 => "b32",
            SassDataType::B64 => "b64",
            SassDataType::B128 => "b128",
            SassDataType::Pred => "pred",
            SassDataType::Unknown => "b32",
        }
    }
}

// ============================================================================
// SASS Operands
// ============================================================================

/// SASS register operand
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SassRegister {
    /// Register type prefix (R, P, UR, UP, SRZ, RZ)
    pub prefix: String,
    /// Register number
    pub number: u32,
    /// Is this a uniform register? (UR0-UR63)
    pub is_uniform: bool,
    /// Is this the zero register? (RZ)
    pub is_zero: bool,
    /// Component selector for vectors (.x, .y, .z, .w, .xy, etc.)
    pub component: Option<String>,
}

impl SassRegister {
    pub fn new(prefix: &str, number: u32) -> Self {
        Self {
            prefix: prefix.to_string(),
            number,
            is_uniform: prefix.starts_with("UR") || prefix.starts_with("UP"),
            is_zero: prefix == "RZ" || prefix == "URZ" || prefix == "PT",
            component: None,
        }
    }

    /// Parse from string like "R0", "R2.X", "UR5", "P0", "RZ"
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();

        // Handle zero registers
        if s == "RZ" {
            return Some(Self {
                prefix: "RZ".to_string(),
                number: 255,
                is_uniform: false,
                is_zero: true,
                component: None,
            });
        }
        if s == "URZ" {
            return Some(Self {
                prefix: "URZ".to_string(),
                number: 63,
                is_uniform: true,
                is_zero: true,
                component: None,
            });
        }
        if s == "PT" {
            return Some(Self {
                prefix: "PT".to_string(),
                number: 7,
                is_uniform: false,
                is_zero: true, // PT is always-true predicate
                component: None,
            });
        }

        // Check for component selector
        let (base, component) = if let Some(dot_pos) = s.find('.') {
            (&s[..dot_pos], Some(s[dot_pos + 1..].to_string()))
        } else {
            (s, None)
        };

        // Parse prefix and number
        let prefix_end = base.chars().position(|c| c.is_ascii_digit())?;
        let prefix = &base[..prefix_end];
        let number: u32 = base[prefix_end..].parse().ok()?;

        Some(Self {
            prefix: prefix.to_string(),
            number,
            is_uniform: prefix.starts_with("UR") || prefix.starts_with("UP"),
            is_zero: false,
            component,
        })
    }

    /// Convert to PTX register name
    pub fn to_ptx_register(&self) -> String {
        if self.is_zero {
            return "%rz".to_string();
        }

        let ptx_prefix = match self.prefix.as_str() {
            "R" => "%r",
            "UR" => "%ur",
            "P" => "%p",
            "UP" => "%up",
            "F" => "%f", // Some SASS uses F for float regs
            _ => "%r",
        };

        format!("{}{}", ptx_prefix, self.number)
    }
}

impl fmt::Display for SassRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref comp) = self.component {
            write!(f, "{}{}.{}", self.prefix, self.number, comp)
        } else {
            write!(f, "{}{}", self.prefix, self.number)
        }
    }
}

/// SASS operand types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SassOperand {
    /// Register operand (R0, P1, UR2, etc.)
    Register(SassRegister),

    /// Immediate value
    Immediate(i64),

    /// Floating-point immediate
    FloatImmediate(f64),

    /// Constant bank reference: c[bank][offset]
    ConstantBank { bank: u32, offset: u32 },

    /// Memory reference: [Rbase + offset]
    Memory {
        base: Option<SassRegister>,
        offset: i64,
        index: Option<SassRegister>,
        scale: u32,
    },

    /// Predicate with optional negation: @P0, @!P1
    Predicate {
        register: SassRegister,
        negated: bool,
    },

    /// Barrier number
    Barrier(u32),

    /// Special register (tid.x, ctaid.y, etc.)
    SpecialRegister(String),

    /// Label/address for branches
    Label(String),

    /// Absolute address
    Address(u64),
}

impl SassOperand {
    /// Try to parse an operand string
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();

        // Empty string
        if s.is_empty() {
            return None;
        }

        // Predicate: @P0, @!P1
        if s.starts_with('@') {
            let negated = s.starts_with("@!");
            let reg_str = if negated { &s[2..] } else { &s[1..] };
            let reg = SassRegister::parse(reg_str)?;
            return Some(SassOperand::Predicate {
                register: reg,
                negated,
            });
        }

        // Constant bank: c[0x0][0x0]
        if s.starts_with("c[") {
            let parts: Vec<&str> = s.split('[').collect();
            if parts.len() >= 3 {
                let bank = parse_hex_or_dec(parts[1].trim_end_matches(']'))?;
                let offset = parse_hex_or_dec(parts[2].trim_end_matches(']'))?;
                return Some(SassOperand::ConstantBank {
                    bank: bank as u32,
                    offset: offset as u32,
                });
            }
        }

        // Memory reference: [R0], [R0+0x10], [R0.64]
        if s.starts_with('[') && s.ends_with(']') {
            let inner = &s[1..s.len() - 1];
            return parse_memory_operand(inner);
        }

        // Immediate: 0x1234, -5, 3.14
        if s.starts_with("0x") || s.starts_with("-0x") {
            if let Some(val) = parse_hex_or_dec(s) {
                return Some(SassOperand::Immediate(val));
            }
        }
        if let Ok(val) = s.parse::<i64>() {
            return Some(SassOperand::Immediate(val));
        }
        if let Ok(val) = s.parse::<f64>() {
            return Some(SassOperand::FloatImmediate(val));
        }

        // Register: R0, P1, UR2, RZ
        if let Some(reg) = SassRegister::parse(s) {
            return Some(SassOperand::Register(reg));
        }

        // Special register: SR_TID.X, SR_CTAID.Y
        if s.starts_with("SR_") {
            return Some(SassOperand::SpecialRegister(s.to_string()));
        }

        // Label or unknown - treat as label
        Some(SassOperand::Label(s.to_string()))
    }

    /// Convert to PTX operand string
    pub fn to_ptx_operand(&self) -> String {
        match self {
            SassOperand::Register(reg) => reg.to_ptx_register(),
            SassOperand::Immediate(val) => format!("{}", val),
            SassOperand::FloatImmediate(val) => format!("{:e}", val),
            SassOperand::ConstantBank { bank, offset } => {
                format!("[const{}+{}]", bank, offset)
            }
            SassOperand::Memory { base, offset, .. } => {
                if let Some(ref b) = base {
                    if *offset != 0 {
                        format!("[{}+{}]", b.to_ptx_register(), offset)
                    } else {
                        format!("[{}]", b.to_ptx_register())
                    }
                } else {
                    format!("[{}]", offset)
                }
            }
            SassOperand::Predicate { register, negated } => {
                if *negated {
                    format!("@!{}", register.to_ptx_register())
                } else {
                    format!("@{}", register.to_ptx_register())
                }
            }
            SassOperand::Barrier(num) => format!("bar{}", num),
            SassOperand::SpecialRegister(name) => {
                // Convert SR_TID.X to %tid.x
                let ptx_name = name.strip_prefix("SR_").unwrap_or(name).to_lowercase();
                format!("%{}", ptx_name)
            }
            SassOperand::Label(name) => name.clone(),
            SassOperand::Address(addr) => format!("0x{:x}", addr),
        }
    }
}

fn parse_hex_or_dec(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.starts_with("0x") || s.starts_with("0X") {
        i64::from_str_radix(&s[2..], 16).ok()
    } else if s.starts_with("-0x") || s.starts_with("-0X") {
        i64::from_str_radix(&s[3..], 16).ok().map(|v| -v)
    } else {
        s.parse().ok()
    }
}

fn parse_memory_operand(inner: &str) -> Option<SassOperand> {
    // Handle various forms:
    // R0
    // R0+0x10
    // R0.64
    // R0+R1
    // R0+0x10.64

    let parts: Vec<&str> = inner.split('+').collect();

    let base = SassRegister::parse(parts[0].split('.').next()?);
    let offset = if parts.len() > 1 {
        parse_hex_or_dec(parts[1].split('.').next()?).unwrap_or(0)
    } else {
        0
    };

    Some(SassOperand::Memory {
        base,
        offset,
        index: None,
        scale: 1,
    })
}

// ============================================================================
// Control Codes
// ============================================================================

/// SASS control codes (scheduling hints, barriers, reuse flags)
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SassControlCodes {
    /// Stall count (0-15): cycles to wait before issuing this instruction
    pub stall_count: u8,
    /// Yield flag: hint to switch to another warp
    pub yield_flag: bool,
    /// Write barrier: wait for previous writes to complete
    pub write_barrier: u8,
    /// Read barrier: wait for previous reads to complete
    pub read_barrier: u8,
    /// Wait barrier mask: which barriers to wait for
    pub wait_barrier_mask: u8,
    /// Reuse flags: which operands to keep in register file cache
    pub reuse_flags: u8,
}

impl SassControlCodes {
    /// Parse from SASS control code string (e.g., "{!5,Y,^1}")
    pub fn parse(s: &str) -> Self {
        let mut codes = SassControlCodes::default();

        let s = s.trim_matches(|c| c == '{' || c == '}' || c == ' ');

        for part in s.split(',') {
            let part = part.trim();

            // Stall count: !5
            if part.starts_with('!') {
                if let Ok(count) = part[1..].parse::<u8>() {
                    codes.stall_count = count;
                }
            }
            // Yield: Y
            else if part == "Y" {
                codes.yield_flag = true;
            }
            // Write barrier: ^1
            else if part.starts_with('^') {
                if let Ok(bar) = part[1..].parse::<u8>() {
                    codes.write_barrier = bar;
                }
            }
            // Read barrier: &2
            else if part.starts_with('&') {
                if let Ok(bar) = part[1..].parse::<u8>() {
                    codes.read_barrier = bar;
                }
            }
            // Wait barrier: W1 or W{1,2}
            else if part.starts_with('W') {
                // Simplified: just set the first barrier number
                if let Ok(bar) = part[1..].parse::<u8>() {
                    codes.wait_barrier_mask = 1 << bar;
                }
            }
            // Reuse: R1, R2, R3 (which source operand to reuse)
            else if part.starts_with('R') && part.len() == 2 {
                if let Ok(reg) = part[1..].parse::<u8>() {
                    codes.reuse_flags |= 1 << reg;
                }
            }
        }

        codes
    }

    /// Check if this instruction should stall
    pub fn should_stall(&self) -> bool {
        self.stall_count > 0 || self.wait_barrier_mask != 0
    }
}

// ============================================================================
// PTX Equivalence
// ============================================================================

/// PTX instruction template for SASS → PTX conversion
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PtxTemplate {
    /// Direct 1:1 mapping
    Direct {
        /// PTX opcode
        opcode: String,
        /// Default type suffix (may be overridden by instruction modifiers)
        default_type: Option<String>,
    },

    /// Composite: one SASS instruction maps to multiple PTX instructions
    Composite {
        /// Sequence of PTX instruction patterns
        patterns: Vec<String>,
    },

    /// No direct PTX equivalent (control instructions, scheduling hints)
    NoEquivalent,

    /// Special handling required
    Special { handler: String },
}

/// Confidence level for PTX mapping
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MappingConfidence {
    /// Exact 1:1 correspondence
    Exact,
    /// High confidence (semantic equivalent)
    High,
    /// Medium confidence (may have slight differences)
    Medium,
    /// Low confidence (approximation)
    Low,
    /// No mapping possible
    None,
}

/// Complete PTX equivalence information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PtxEquivalent {
    /// The PTX instruction string
    pub instruction: String,
    /// Confidence level
    pub confidence: MappingConfidence,
    /// Is this an exact 1:1 match?
    pub is_exact: bool,
    /// Notes about the mapping
    pub notes: Option<String>,
}

// ============================================================================
// Enhanced SASS Instruction
// ============================================================================

/// Enhanced SASS instruction with full semantic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSassInstruction {
    // ========== Basic Information ==========
    /// SASS opcode (e.g., "LDG", "FADD", "IMAD")
    pub opcode: String,
    /// Full instruction text (original disassembly)
    pub instruction_text: String,
    /// Address/offset in the compiled binary
    pub address: u64,
    /// Instruction size in bytes (8 or 16)
    pub size: u8,

    // ========== Encoding ==========
    /// Raw instruction encoding (lower 64 bits)
    pub encoding_lo: u64,
    /// Raw instruction encoding (upper 64 bits, for 128-bit instructions)
    pub encoding_hi: Option<u64>,

    // ========== Semantic Information ==========
    /// Opcode classification
    pub opcode_class: SassOpcodeClass,
    /// Memory space (for memory operations)
    pub memory_space: Option<SassMemorySpace>,
    /// Data type
    pub data_type: Option<SassDataType>,
    /// Vector width (1, 2, 4, or 16 for tensor ops)
    pub vector_width: u8,

    // ========== Operands ==========
    /// Predicate (guard condition)
    pub predicate: Option<SassOperand>,
    /// Destination operands
    pub dest_operands: Vec<SassOperand>,
    /// Source operands
    pub src_operands: Vec<SassOperand>,

    // ========== Modifiers ==========
    /// Instruction modifiers (e.g., ".E", ".STRONG", ".SYS")
    pub modifiers: Vec<String>,
    /// Control codes (stall, yield, barriers)
    pub control_codes: SassControlCodes,

    // ========== PTX Mapping ==========
    /// PTX template for this instruction
    pub ptx_template: PtxTemplate,
    /// Computed PTX equivalent (if available)
    pub ptx_equivalent: Option<PtxEquivalent>,

    // ========== Source Location ==========
    /// PTX source file
    pub ptx_file: Option<String>,
    /// PTX source line
    pub ptx_line: Option<u32>,
    /// PTX source column
    pub ptx_column: Option<u32>,
    /// Function name
    pub function_name: Option<String>,

    // ========== Data Flow ==========
    /// Addresses of instructions this depends on
    pub data_dependencies: Vec<u64>,
    /// Basic block ID (for control flow analysis)
    pub basic_block_id: Option<u32>,
}

impl EnhancedSassInstruction {
    /// Create a new enhanced instruction with minimal information
    pub fn new(opcode: String, address: u64) -> Self {
        Self {
            opcode: opcode.clone(),
            instruction_text: String::new(),
            address,
            size: 16, // Default to 128-bit instructions
            encoding_lo: 0,
            encoding_hi: None,
            opcode_class: classify_opcode(&opcode),
            memory_space: None,
            data_type: None,
            vector_width: 1,
            predicate: None,
            dest_operands: Vec::new(),
            src_operands: Vec::new(),
            modifiers: Vec::new(),
            control_codes: SassControlCodes::default(),
            ptx_template: get_ptx_template(&opcode),
            ptx_equivalent: None,
            ptx_file: None,
            ptx_line: None,
            ptx_column: None,
            function_name: None,
            data_dependencies: Vec::new(),
            basic_block_id: None,
        }
    }

    /// Generate PTX equivalent instruction
    pub fn generate_ptx(&self) -> Option<String> {
        match &self.ptx_template {
            PtxTemplate::Direct {
                opcode,
                default_type,
            } => {
                let type_suffix = self
                    .data_type
                    .map(|dt| dt.to_ptx_suffix())
                    .or(default_type.as_deref())
                    .unwrap_or("b32");

                let pred = self
                    .predicate
                    .as_ref()
                    .map(|p| format!("{} ", p.to_ptx_operand()))
                    .unwrap_or_default();

                let dests: Vec<String> = self
                    .dest_operands
                    .iter()
                    .map(|o| o.to_ptx_operand())
                    .collect();
                let srcs: Vec<String> = self
                    .src_operands
                    .iter()
                    .map(|o| o.to_ptx_operand())
                    .collect();

                let all_operands = if dests.is_empty() {
                    srcs.join(", ")
                } else if srcs.is_empty() {
                    dests.join(", ")
                } else {
                    format!("{}, {}", dests.join(", "), srcs.join(", "))
                };

                Some(format!(
                    "{}{}.{} {};",
                    pred, opcode, type_suffix, all_operands
                ))
            }
            PtxTemplate::Composite { patterns } => {
                // For composite instructions, return the first pattern
                // A more complete implementation would expand all patterns
                patterns.first().cloned()
            }
            PtxTemplate::NoEquivalent => None,
            PtxTemplate::Special { handler } => {
                // Special handling - return placeholder
                Some(format!("// Special: {} ({})", self.opcode, handler))
            }
        }
    }

    /// Check if this instruction writes to the given register
    pub fn writes_to(&self, reg: &SassRegister) -> bool {
        self.dest_operands.iter().any(|op| {
            if let SassOperand::Register(r) = op {
                r.prefix == reg.prefix && r.number == reg.number
            } else {
                false
            }
        })
    }

    /// Check if this instruction reads from the given register
    pub fn reads_from(&self, reg: &SassRegister) -> bool {
        self.src_operands.iter().any(|op| {
            if let SassOperand::Register(r) = op {
                r.prefix == reg.prefix && r.number == reg.number
            } else {
                false
            }
        })
    }

    /// Get the latency estimate for this instruction
    pub fn estimated_latency(&self) -> u32 {
        match self.opcode_class {
            SassOpcodeClass::GlobalLoad => 200, // High latency
            SassOpcodeClass::SharedLoad => 20,  // Medium latency
            SassOpcodeClass::LocalLoad => 50,
            SassOpcodeClass::ConstantLoad => 4, // Cached
            SassOpcodeClass::GlobalStore | SassOpcodeClass::SharedStore => 20,
            SassOpcodeClass::FloatTranscendental => 20, // MUFU
            SassOpcodeClass::DoubleArithmetic => 8,
            SassOpcodeClass::IntegerArithmetic | SassOpcodeClass::FloatArithmetic => 4,
            SassOpcodeClass::Branch | SassOpcodeClass::ConditionalBranch => 8,
            _ => 4, // Default
        }
    }
}

impl fmt::Display for EnhancedSassInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "/*{:04x}*/ ", self.address)?;

        if let Some(ref pred) = self.predicate {
            write!(f, "{} ", pred.to_ptx_operand())?;
        }

        write!(f, "{}", self.opcode)?;

        for modifier in &self.modifiers {
            write!(f, ".{}", modifier)?;
        }

        if !self.dest_operands.is_empty() || !self.src_operands.is_empty() {
            write!(f, " ")?;

            let all_operands: Vec<String> = self
                .dest_operands
                .iter()
                .chain(self.src_operands.iter())
                .map(|o| o.to_ptx_operand())
                .collect();

            write!(f, "{}", all_operands.join(", "))?;
        }

        write!(f, " ;")
    }
}

// ============================================================================
// Opcode Classification and PTX Templates
// ============================================================================

/// Classify a SASS opcode into its category
pub fn classify_opcode(opcode: &str) -> SassOpcodeClass {
    let op = opcode.to_uppercase();

    match op.as_str() {
        // Integer Arithmetic
        "IADD" | "IADD3" | "IADD32I" | "IMAD" | "IMAD32I" | "IMADSP" | "IMUL" | "IMUL32I"
        | "IABS" | "INEG" | "IMNMX" | "LEA" | "LEA.HI" => SassOpcodeClass::IntegerArithmetic,

        // Integer Comparison
        "ISETP" | "ISET" | "ICMP" => SassOpcodeClass::IntegerComparison,

        // Integer Logical
        "LOP" | "LOP3" | "LOP32I" | "AND" | "OR" | "XOR" | "NOT" | "SHL" | "SHR" | "SHF"
        | "BFE" | "BFI" | "FLO" | "POPC" => SassOpcodeClass::IntegerLogical,

        // Integer Conversion
        "I2I" | "I2F" | "F2I" => SassOpcodeClass::IntegerConversion,

        // Floating Point Arithmetic
        "FADD" | "FADD32I" | "FMUL" | "FMUL32I" | "FFMA" | "FMNMX" | "FABS" | "FNEG"
        | "FSWZADD" => SassOpcodeClass::FloatArithmetic,

        // Floating Point Comparison
        "FSETP" | "FSET" | "FCMP" => SassOpcodeClass::FloatComparison,

        // Floating Point Transcendental (MUFU)
        "MUFU" => SassOpcodeClass::FloatTranscendental,

        // Floating Point Conversion
        "F2F" => SassOpcodeClass::FloatConversion,

        // Double Precision
        "DADD" | "DMUL" | "DFMA" | "DSETP" | "DSET" => SassOpcodeClass::DoubleArithmetic,

        // Half Precision
        "HADD2" | "HMUL2" | "HFMA2" | "HSETP2" | "HSET2" => SassOpcodeClass::HalfArithmetic,

        // Global Memory
        "LDG" => SassOpcodeClass::GlobalLoad,
        "STG" => SassOpcodeClass::GlobalStore,
        "ATOMG" | "RED" => SassOpcodeClass::GlobalAtomic,

        // Shared Memory
        "LDS" => SassOpcodeClass::SharedLoad,
        "STS" => SassOpcodeClass::SharedStore,
        "ATOMS" => SassOpcodeClass::SharedAtomic,

        // Local Memory
        "LDL" => SassOpcodeClass::LocalLoad,
        "STL" => SassOpcodeClass::LocalStore,

        // Constant Memory
        "LDC" => SassOpcodeClass::ConstantLoad,

        // Texture
        "TEX" | "TLD" | "TLD4" | "TMML" => SassOpcodeClass::TextureFetch,
        "TXQ" | "TXD" => SassOpcodeClass::TextureQuery,

        // Surface
        "SULD" => SassOpcodeClass::SurfaceLoad,
        "SUST" => SassOpcodeClass::SurfaceStore,
        "SUATOM" | "SURED" => SassOpcodeClass::SurfaceAtomic,

        // Control Flow
        "BRA" | "JMP" | "JMXU" => SassOpcodeClass::Branch,
        "BRX" | "JMX" => SassOpcodeClass::ConditionalBranch,
        "CAL" | "JCAL" => SassOpcodeClass::Call,
        "RET" => SassOpcodeClass::Return,
        "EXIT" => SassOpcodeClass::Exit,

        // Synchronization
        "BAR" | "DEPBAR" | "B2R" | "R2B" => SassOpcodeClass::Barrier,
        "MEMBAR" => SassOpcodeClass::MemoryBarrier,

        // Special Registers
        "S2R" | "CS2R" => SassOpcodeClass::SpecialRegRead,

        // Predicate Operations
        "PSETP" | "PLOP3" | "P2R" | "R2P" => SassOpcodeClass::PredicateSet,

        // Data Movement
        "MOV" | "MOV32I" | "MOVM" | "PRMT" | "SEL" => SassOpcodeClass::Move,
        "SHFL" => SassOpcodeClass::Shuffle,

        // Uniform Operations
        "UIADD3" | "UIMAD" | "UFLO" | "UPOPC" | "ULDC" | "ULEA" | "UMOV" | "UPRMT" | "USEL"
        | "USETP" => SassOpcodeClass::UniformDataPath,

        // Tensor Core
        "HMMA" | "IMMA" | "BMMA" | "DMMA" => SassOpcodeClass::TensorCore,

        // No-op
        "NOP" => SassOpcodeClass::NoOperation,

        _ => SassOpcodeClass::Unknown,
    }
}

/// Get the PTX template for a SASS opcode
pub fn get_ptx_template(opcode: &str) -> PtxTemplate {
    let op = opcode.to_uppercase();

    match op.as_str() {
        // Integer Arithmetic
        "IADD" | "IADD32I" => PtxTemplate::Direct {
            opcode: "add".to_string(),
            default_type: Some("s32".to_string()),
        },
        "IADD3" => PtxTemplate::Special {
            handler: "iadd3_to_add".to_string(),
        },
        "IMAD" | "IMAD32I" => PtxTemplate::Direct {
            opcode: "mad.lo".to_string(),
            default_type: Some("s32".to_string()),
        },
        "IMUL" | "IMUL32I" => PtxTemplate::Direct {
            opcode: "mul.lo".to_string(),
            default_type: Some("s32".to_string()),
        },
        "IABS" => PtxTemplate::Direct {
            opcode: "abs".to_string(),
            default_type: Some("s32".to_string()),
        },
        "INEG" => PtxTemplate::Direct {
            opcode: "neg".to_string(),
            default_type: Some("s32".to_string()),
        },
        "IMNMX" => PtxTemplate::Direct {
            opcode: "min".to_string(),
            default_type: Some("s32".to_string()),
        }, // or max depending on flag

        // Integer Comparison
        "ISETP" => PtxTemplate::Direct {
            opcode: "setp".to_string(),
            default_type: Some("s32".to_string()),
        },
        "ISET" => PtxTemplate::Direct {
            opcode: "set".to_string(),
            default_type: Some("s32".to_string()),
        },

        // Integer Logical
        "LOP" | "LOP3" | "LOP32I" => PtxTemplate::Special {
            handler: "lop3_to_logic".to_string(),
        },
        "SHL" => PtxTemplate::Direct {
            opcode: "shl".to_string(),
            default_type: Some("b32".to_string()),
        },
        "SHR" => PtxTemplate::Direct {
            opcode: "shr".to_string(),
            default_type: Some("b32".to_string()),
        },
        "AND" => PtxTemplate::Direct {
            opcode: "and".to_string(),
            default_type: Some("b32".to_string()),
        },
        "OR" => PtxTemplate::Direct {
            opcode: "or".to_string(),
            default_type: Some("b32".to_string()),
        },
        "XOR" => PtxTemplate::Direct {
            opcode: "xor".to_string(),
            default_type: Some("b32".to_string()),
        },
        "NOT" => PtxTemplate::Direct {
            opcode: "not".to_string(),
            default_type: Some("b32".to_string()),
        },
        "BFE" => PtxTemplate::Direct {
            opcode: "bfe".to_string(),
            default_type: Some("u32".to_string()),
        },
        "BFI" => PtxTemplate::Direct {
            opcode: "bfi".to_string(),
            default_type: Some("b32".to_string()),
        },
        "FLO" => PtxTemplate::Direct {
            opcode: "bfind".to_string(),
            default_type: Some("u32".to_string()),
        },
        "POPC" => PtxTemplate::Direct {
            opcode: "popc".to_string(),
            default_type: Some("b32".to_string()),
        },

        // Floating Point Arithmetic
        "FADD" | "FADD32I" => PtxTemplate::Direct {
            opcode: "add".to_string(),
            default_type: Some("f32".to_string()),
        },
        "FMUL" | "FMUL32I" => PtxTemplate::Direct {
            opcode: "mul".to_string(),
            default_type: Some("f32".to_string()),
        },
        "FFMA" => PtxTemplate::Direct {
            opcode: "fma.rn".to_string(),
            default_type: Some("f32".to_string()),
        },
        "FMNMX" => PtxTemplate::Direct {
            opcode: "min".to_string(),
            default_type: Some("f32".to_string()),
        },
        "FABS" => PtxTemplate::Direct {
            opcode: "abs".to_string(),
            default_type: Some("f32".to_string()),
        },
        "FNEG" => PtxTemplate::Direct {
            opcode: "neg".to_string(),
            default_type: Some("f32".to_string()),
        },

        // Floating Point Comparison
        "FSETP" => PtxTemplate::Direct {
            opcode: "setp".to_string(),
            default_type: Some("f32".to_string()),
        },
        "FSET" => PtxTemplate::Direct {
            opcode: "set".to_string(),
            default_type: Some("f32".to_string()),
        },

        // Transcendental
        "MUFU" => PtxTemplate::Special {
            handler: "mufu_to_transcendental".to_string(),
        },

        // Conversions
        "I2I" => PtxTemplate::Direct {
            opcode: "cvt".to_string(),
            default_type: None,
        },
        "I2F" => PtxTemplate::Direct {
            opcode: "cvt".to_string(),
            default_type: None,
        },
        "F2I" => PtxTemplate::Direct {
            opcode: "cvt".to_string(),
            default_type: None,
        },
        "F2F" => PtxTemplate::Direct {
            opcode: "cvt".to_string(),
            default_type: None,
        },

        // Double Precision
        "DADD" => PtxTemplate::Direct {
            opcode: "add".to_string(),
            default_type: Some("f64".to_string()),
        },
        "DMUL" => PtxTemplate::Direct {
            opcode: "mul".to_string(),
            default_type: Some("f64".to_string()),
        },
        "DFMA" => PtxTemplate::Direct {
            opcode: "fma.rn".to_string(),
            default_type: Some("f64".to_string()),
        },

        // Memory Operations
        "LDG" => PtxTemplate::Direct {
            opcode: "ld.global".to_string(),
            default_type: None,
        },
        "STG" => PtxTemplate::Direct {
            opcode: "st.global".to_string(),
            default_type: None,
        },
        "LDS" => PtxTemplate::Direct {
            opcode: "ld.shared".to_string(),
            default_type: None,
        },
        "STS" => PtxTemplate::Direct {
            opcode: "st.shared".to_string(),
            default_type: None,
        },
        "LDL" => PtxTemplate::Direct {
            opcode: "ld.local".to_string(),
            default_type: None,
        },
        "STL" => PtxTemplate::Direct {
            opcode: "st.local".to_string(),
            default_type: None,
        },
        "LDC" => PtxTemplate::Direct {
            opcode: "ld.const".to_string(),
            default_type: None,
        },

        // Atomics
        "ATOMG" => PtxTemplate::Direct {
            opcode: "atom.global".to_string(),
            default_type: None,
        },
        "ATOMS" => PtxTemplate::Direct {
            opcode: "atom.shared".to_string(),
            default_type: None,
        },
        "RED" => PtxTemplate::Direct {
            opcode: "red.global".to_string(),
            default_type: None,
        },

        // Control Flow
        "BRA" | "JMP" => PtxTemplate::Direct {
            opcode: "bra".to_string(),
            default_type: None,
        },
        "CAL" | "JCAL" => PtxTemplate::Direct {
            opcode: "call".to_string(),
            default_type: None,
        },
        "RET" => PtxTemplate::Direct {
            opcode: "ret".to_string(),
            default_type: None,
        },
        "EXIT" => PtxTemplate::Direct {
            opcode: "exit".to_string(),
            default_type: None,
        },

        // Synchronization
        "BAR" => PtxTemplate::Direct {
            opcode: "bar.sync".to_string(),
            default_type: None,
        },
        "MEMBAR" => PtxTemplate::Direct {
            opcode: "membar".to_string(),
            default_type: None,
        },

        // Special Registers
        "S2R" => PtxTemplate::Direct {
            opcode: "mov".to_string(),
            default_type: Some("u32".to_string()),
        },

        // Data Movement
        "MOV" | "MOV32I" => PtxTemplate::Direct {
            opcode: "mov".to_string(),
            default_type: Some("b32".to_string()),
        },
        "SEL" => PtxTemplate::Direct {
            opcode: "selp".to_string(),
            default_type: Some("b32".to_string()),
        },
        "PRMT" => PtxTemplate::Direct {
            opcode: "prmt".to_string(),
            default_type: Some("b32".to_string()),
        },
        "SHFL" => PtxTemplate::Direct {
            opcode: "shfl".to_string(),
            default_type: None,
        },

        // Predicates
        "PSETP" => PtxTemplate::Direct {
            opcode: "setp".to_string(),
            default_type: Some("pred".to_string()),
        },

        // Tensor Core
        "HMMA" | "IMMA" | "BMMA" | "DMMA" => PtxTemplate::Special {
            handler: "mma_to_wmma".to_string(),
        },

        // No-op
        "NOP" => PtxTemplate::NoEquivalent,

        // Control codes have no PTX equivalent
        _ => PtxTemplate::NoEquivalent,
    }
}

/// Get memory space from SASS opcode
pub fn get_memory_space(opcode: &str) -> Option<SassMemorySpace> {
    let op = opcode.to_uppercase();

    match op.as_str() {
        "LDG" | "STG" | "ATOMG" | "RED" => Some(SassMemorySpace::Global),
        "LDS" | "STS" | "ATOMS" => Some(SassMemorySpace::Shared),
        "LDL" | "STL" => Some(SassMemorySpace::Local),
        "LDC" => Some(SassMemorySpace::Constant),
        "TEX" | "TLD" | "TLD4" | "TXQ" => Some(SassMemorySpace::Texture),
        "SULD" | "SUST" | "SUATOM" => Some(SassMemorySpace::Surface),
        _ => None,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_parse() {
        let r0 = SassRegister::parse("R0").unwrap();
        assert_eq!(r0.prefix, "R");
        assert_eq!(r0.number, 0);
        assert!(!r0.is_uniform);
        assert!(!r0.is_zero);

        let rz = SassRegister::parse("RZ").unwrap();
        assert!(rz.is_zero);

        let ur5 = SassRegister::parse("UR5").unwrap();
        assert!(ur5.is_uniform);
        assert_eq!(ur5.number, 5);

        let p3 = SassRegister::parse("P3").unwrap();
        assert_eq!(p3.prefix, "P");
        assert_eq!(p3.number, 3);
    }

    #[test]
    fn test_operand_parse() {
        // Register
        let op = SassOperand::parse("R0").unwrap();
        if let SassOperand::Register(reg) = op {
            assert_eq!(reg.number, 0);
        } else {
            panic!("Expected register");
        }

        // Immediate
        let op = SassOperand::parse("0x10").unwrap();
        if let SassOperand::Immediate(val) = op {
            assert_eq!(val, 16);
        } else {
            panic!("Expected immediate");
        }

        // Constant bank
        let op = SassOperand::parse("c[0x0][0x160]").unwrap();
        if let SassOperand::ConstantBank { bank, offset } = op {
            assert_eq!(bank, 0);
            assert_eq!(offset, 0x160);
        } else {
            panic!("Expected constant bank");
        }

        // Memory
        let op = SassOperand::parse("[R2]").unwrap();
        if let SassOperand::Memory { base, offset, .. } = op {
            assert!(base.is_some());
            assert_eq!(offset, 0);
        } else {
            panic!("Expected memory");
        }
    }

    #[test]
    fn test_opcode_classification() {
        assert_eq!(classify_opcode("FADD"), SassOpcodeClass::FloatArithmetic);
        assert_eq!(classify_opcode("LDG"), SassOpcodeClass::GlobalLoad);
        assert_eq!(classify_opcode("STG"), SassOpcodeClass::GlobalStore);
        assert_eq!(classify_opcode("BRA"), SassOpcodeClass::Branch);
        assert_eq!(classify_opcode("IMAD"), SassOpcodeClass::IntegerArithmetic);
        assert_eq!(classify_opcode("BAR"), SassOpcodeClass::Barrier);
    }

    #[test]
    fn test_data_type_parse() {
        assert_eq!(SassDataType::from_modifier(".F32"), Some(SassDataType::F32));
        assert_eq!(SassDataType::from_modifier(".U64"), Some(SassDataType::U64));
        assert_eq!(SassDataType::from_modifier(".S16"), Some(SassDataType::S16));
        assert_eq!(
            SassDataType::from_modifier("BF16"),
            Some(SassDataType::BF16)
        );
    }

    #[test]
    fn test_control_codes_parse() {
        let codes = SassControlCodes::parse("{!5,Y,^1}");
        assert_eq!(codes.stall_count, 5);
        assert!(codes.yield_flag);
        assert_eq!(codes.write_barrier, 1);
    }

    #[test]
    fn test_enhanced_instruction() {
        let mut inst = EnhancedSassInstruction::new("FADD".to_string(), 0x100);
        inst.data_type = Some(SassDataType::F32);
        inst.dest_operands
            .push(SassOperand::Register(SassRegister::new("R", 0)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 1)));
        inst.src_operands
            .push(SassOperand::Register(SassRegister::new("R", 2)));

        let ptx = inst.generate_ptx();
        assert!(ptx.is_some());
        let ptx = ptx.unwrap();
        assert!(ptx.contains("add.f32"));
    }
}
