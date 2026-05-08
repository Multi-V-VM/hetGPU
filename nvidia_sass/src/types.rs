/// A SASS instruction ready for encoding.
#[derive(Debug, Clone, PartialEq)]
pub struct SassInst {
    pub opcode: Opcode,
    pub dst: Option<Reg>,
    pub srcs: Vec<Operand>,
    pub pred: Option<Predicate>,
    pub modifiers: Vec<Modifier>,
    pub control: ControlCodes,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Opcode {
    pub mnemonic: &'static str,
    pub class: OpcodeClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpcodeClass {
    Alu3,
    Alu2,
    Fma,
    Load,
    Store,
    Branch,
    Comparison,
    Sync,
    Special,
    Nop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reg {
    R(u8),  // General purpose R0-R255
    RZ,     // Zero register (255)
    P(u8),  // Predicate P0-P6
    PT,     // True predicate (7)
    UR(u8), // Uniform register
    UP(u8), // Uniform predicate
}

impl Reg {
    pub fn encode_gpr(&self) -> u8 {
        match self {
            Reg::R(n) => *n,
            Reg::RZ => 255,
            _ => panic!("not a GPR: {:?}", self),
        }
    }

    pub fn encode_pred(&self) -> u8 {
        match self {
            Reg::P(n) => *n,
            Reg::PT => 7,
            _ => panic!("not a predicate: {:?}", self),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Reg(Reg),
    Imm20(i32),
    Imm32(i32),
    ConstBank { bank: u8, offset: u16 },
    Memory { base: Reg, offset: i32 },
    BranchTarget(u32),
    SReg(SpecialReg),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Predicate {
    pub reg: Reg,
    pub negated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Modifier {
    DataType(DataType),
    MemScope(MemScope),
    CacheOp(CacheOp),
    CmpOp(CmpOp),
    BoolOp(u8),
    MufuOp(MufuOp),
    Neg(u8),
    Abs(u8),
    Wide,
    Hi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    U8,
    U16,
    U32,
    U64,
    U128,
    S8,
    S16,
    S32,
    S64,
    F16,
    F32,
    F64,
    BF16,
    TF32,
    FP8E4M3,
    FP8E5M2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemScope {
    E,
    Strong,
    Cta,
    Gpu,
    Sys,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheOp {
    Ef,
    El,
    Lu,
    Eu,
    Na,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    Ltu,
    Equ,
    Leu,
    Gtu,
    Neu,
    Geu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MufuOp {
    Rcp,
    Rsq,
    Sin,
    Cos,
    Ex2,
    Lg2,
    Rcp64h,
    Rsq64h,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialReg {
    TidX,
    TidY,
    TidZ,
    CtaidX,
    CtaidY,
    CtaidZ,
    NctaidX,
    NctaidY,
    NctaidZ,
    NtidX,
    NtidY,
    NtidZ,
    LaneId,
    WarpId,
    SmId,
    ClockLo,
    ClockHi,
    GlobalTimerLo,
    GlobalTimerHi,
}

impl SpecialReg {
    pub fn encode(&self) -> u8 {
        match self {
            SpecialReg::LaneId => 0x00,
            SpecialReg::WarpId => 0x02,
            SpecialReg::SmId => 0x06,
            SpecialReg::TidX => 0x21,
            SpecialReg::TidY => 0x22,
            SpecialReg::TidZ => 0x23,
            SpecialReg::CtaidX => 0x25,
            SpecialReg::CtaidY => 0x26,
            SpecialReg::CtaidZ => 0x27,
            SpecialReg::NtidX => 0x29,
            SpecialReg::NtidY => 0x2a,
            SpecialReg::NtidZ => 0x2b,
            SpecialReg::NctaidX => 0x2d,
            SpecialReg::NctaidY => 0x2e,
            SpecialReg::NctaidZ => 0x2f,
            SpecialReg::ClockLo => 0x50,
            SpecialReg::ClockHi => 0x51,
            SpecialReg::GlobalTimerLo => 0x58,
            SpecialReg::GlobalTimerHi => 0x59,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlCodes {
    pub stall: u8,
    pub yield_flag: bool,
    pub write_barrier: u8,
    pub read_barrier: u8,
    pub wait_mask: u8,
    pub reuse: u8,
}

impl Default for ControlCodes {
    fn default() -> Self {
        ControlCodes {
            stall: 15,
            yield_flag: false,
            write_barrier: 7,
            read_barrier: 7,
            wait_mask: 0,
            reuse: 0,
        }
    }
}

#[derive(Debug)]
pub struct SassKernel {
    pub name: String,
    pub instructions: Vec<SassInst>,
    pub num_registers: u32,
    pub shared_mem_bytes: u32,
    pub const_mem_bytes: u32,
    pub local_mem_bytes: u32,
    pub max_threads: u32,
    pub params: Vec<(String, u32, u32)>,
}

#[derive(Debug)]
pub struct SassModule {
    pub kernels: Vec<SassKernel>,
    pub sm_version: u32,
    pub global_constants: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum NvSassError {
    #[error("unsupported SM version: {0}")]
    UnsupportedSmVersion(u32),
    #[error("encoding error for {opcode}: {msg}")]
    EncodingError { opcode: String, msg: String },
    #[error("register allocation failed: {0}")]
    RegAllocError(String),
    #[error("instruction selection failed: {0}")]
    ISelError(String),
    #[error("CUBIN generation failed: {0}")]
    CubinError(String),
    #[error("ELF write error: {0}")]
    ElfError(String),
}
