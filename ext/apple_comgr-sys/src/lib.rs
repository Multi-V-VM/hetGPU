#![allow(non_camel_case_types)]

use ptx_parser::{
    AtomSemantics, AtomicOp, CvtMode, Directive, Function, FunnelShiftMode, ImmediateValue,
    Instruction, Module, Mul24Control, MulDetails, MulIntControl, MultiVariable, ParsedOperand,
    PredAt, RegOrImmediate, RightShiftKind, ScalarType, SetpBoolPostOp, SetpCompareFloat,
    SetpCompareInt, SetpCompareOp, ShiftDirection, StateSpace, Statement, Type,
};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_uint, c_void, CStr};
use std::ptr;
use std::sync::{Mutex, OnceLock};

pub const APPLE_COMGR_INTERFACE_VERSION_MAJOR: u32 = 1;
pub const APPLE_COMGR_INTERFACE_VERSION_MINOR: u32 = 0;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct apple_comgr_status_s(pub u32);

pub type apple_comgr_status_t = Result<(), apple_comgr_status_s>;

impl apple_comgr_status_s {
    pub const APPLE_COMGR_STATUS_SUCCESS: Result<(), apple_comgr_status_s> = Ok(());
    pub const APPLE_COMGR_STATUS_ERROR: apple_comgr_status_s = apple_comgr_status_s(1);
    pub const APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT: apple_comgr_status_s =
        apple_comgr_status_s(2);
    pub const APPLE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES: apple_comgr_status_s =
        apple_comgr_status_s(3);
    pub const APPLE_COMGR_STATUS_ERROR_PARSE_FAILED: apple_comgr_status_s = apple_comgr_status_s(4);
    pub const APPLE_COMGR_STATUS_ERROR_UNSUPPORTED_PTX: apple_comgr_status_s =
        apple_comgr_status_s(5);
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct apple_comgr_data_kind_s(pub c_uint);

pub type apple_comgr_data_kind_t = apple_comgr_data_kind_s;

impl apple_comgr_data_kind_s {
    pub const APPLE_COMGR_DATA_KIND_UNDEF: apple_comgr_data_kind_s = apple_comgr_data_kind_s(0);
    pub const APPLE_COMGR_DATA_KIND_SOURCE: apple_comgr_data_kind_s = apple_comgr_data_kind_s(1);
    pub const APPLE_COMGR_DATA_KIND_PTX: apple_comgr_data_kind_s = apple_comgr_data_kind_s(2);
    pub const APPLE_COMGR_DATA_KIND_MSL: apple_comgr_data_kind_s = apple_comgr_data_kind_s(3);
    pub const APPLE_COMGR_DATA_KIND_EXECUTABLE: apple_comgr_data_kind_s =
        apple_comgr_data_kind_s(4);
    pub const APPLE_COMGR_DATA_KIND_LOG: apple_comgr_data_kind_s = apple_comgr_data_kind_s(5);
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct apple_comgr_action_kind_s(pub c_uint);

pub type apple_comgr_action_kind_t = apple_comgr_action_kind_s;

impl apple_comgr_action_kind_s {
    pub const APPLE_COMGR_ACTION_COMPILE_PTX_TO_MSL: apple_comgr_action_kind_s =
        apple_comgr_action_kind_s(1);
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct apple_comgr_data_s {
    pub handle: u64,
}

pub type apple_comgr_data_t = apple_comgr_data_s;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct apple_comgr_data_set_s {
    pub handle: u64,
}

pub type apple_comgr_data_set_t = apple_comgr_data_set_s;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct apple_comgr_action_info_s {
    pub handle: u64,
}

pub type apple_comgr_action_info_t = apple_comgr_action_info_s;

#[derive(Debug, Clone)]
pub struct AppleComgrCompileError {
    pub status: apple_comgr_status_s,
    pub diagnostics: String,
}

#[derive(Debug, Clone)]
pub struct AppleCompiledModule {
    pub msl: String,
    pub kernels: Vec<AppleKernelMetadata>,
}

#[derive(Debug, Clone)]
pub struct AppleKernelMetadata {
    pub name: String,
    pub msl_name: String,
    pub params: Vec<AppleKernelParamMetadata>,
}

#[derive(Debug, Clone)]
pub struct AppleKernelParamMetadata {
    pub ptx_name: String,
    pub msl_name: String,
    pub size: usize,
    pub is_pointer: bool,
}

const PTX_MSL_HELPERS: &str = r#"static inline uint zluda_bfe_u32(uint value, uint offset, uint width) {
    if (offset >= 32u || width == 0u) {
        return uint(0);
    }
    uint w = min(width, 32u - offset);
    uint mask = (w >= 32u) ? ~uint(0) : ((uint(1) << w) - uint(1));
    return (value >> offset) & mask;
}

static inline int zluda_bfe_s32(int value, uint offset, uint width) {
    uint bits = zluda_bfe_u32(uint(value), offset, width);
    if (width == 0u) {
        return int(0);
    }
    uint w = min(width, 32u);
    if (w >= 32u) {
        return int(bits);
    }
    uint sign = uint(1) << (w - 1u);
    return int((bits ^ sign) - sign);
}

static inline ulong zluda_bfe_u64(ulong value, uint offset, uint width) {
    if (offset >= 64u || width == 0u) {
        return ulong(0);
    }
    uint w = min(width, 64u - offset);
    ulong mask = (w >= 64u) ? ~ulong(0) : ((ulong(1) << w) - ulong(1));
    return (value >> offset) & mask;
}

static inline long zluda_bfe_s64(long value, uint offset, uint width) {
    ulong bits = zluda_bfe_u64(ulong(value), offset, width);
    if (width == 0u) {
        return long(0);
    }
    uint w = min(width, 64u);
    if (w >= 64u) {
        return long(bits);
    }
    ulong sign = ulong(1) << (w - 1u);
    return long((bits ^ sign) - sign);
}

static inline uint zluda_bfi_b32(uint insert, uint base, uint offset, uint width) {
    if (offset >= 32u || width == 0u) {
        return base;
    }
    uint w = min(width, 32u - offset);
    uint low_mask = (w >= 32u) ? ~uint(0) : ((uint(1) << w) - uint(1));
    uint mask = low_mask << offset;
    return (base & ~mask) | ((insert << offset) & mask);
}

static inline ulong zluda_bfi_b64(ulong insert, ulong base, uint offset, uint width) {
    if (offset >= 64u || width == 0u) {
        return base;
    }
    uint w = min(width, 64u - offset);
    ulong low_mask = (w >= 64u) ? ~ulong(0) : ((ulong(1) << w) - ulong(1));
    ulong mask = low_mask << offset;
    return (base & ~mask) | ((insert << offset) & mask);
}

"#;

#[derive(Clone)]
struct DataContent {
    kind: apple_comgr_data_kind_s,
    content: Vec<u8>,
    name: Option<String>,
}

#[derive(Clone, Default)]
struct ActionInfo {
    options: Vec<String>,
}

static HANDLE_COUNTER: OnceLock<Mutex<u64>> = OnceLock::new();
static DATA_STORE: OnceLock<Mutex<HashMap<u64, DataContent>>> = OnceLock::new();
static DATA_SET_STORE: OnceLock<Mutex<HashMap<u64, Vec<u64>>>> = OnceLock::new();
static ACTION_INFO_STORE: OnceLock<Mutex<HashMap<u64, ActionInfo>>> = OnceLock::new();

fn handle_counter() -> &'static Mutex<u64> {
    HANDLE_COUNTER.get_or_init(|| Mutex::new(1))
}

fn data_store() -> &'static Mutex<HashMap<u64, DataContent>> {
    DATA_STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn data_set_store() -> &'static Mutex<HashMap<u64, Vec<u64>>> {
    DATA_SET_STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn action_info_store() -> &'static Mutex<HashMap<u64, ActionInfo>> {
    ACTION_INFO_STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_handle() -> u64 {
    let mut counter = handle_counter().lock().unwrap();
    let handle = *counter;
    *counter += 1;
    handle
}

pub fn apple_comgr_create_data(
    kind: apple_comgr_data_kind_t,
    data: *mut apple_comgr_data_t,
) -> apple_comgr_status_t {
    if data.is_null() {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let handle = next_handle();
    data_store().lock().unwrap().insert(
        handle,
        DataContent {
            kind,
            content: Vec::new(),
            name: None,
        },
    );
    unsafe {
        *data = apple_comgr_data_t { handle };
    }
    Ok(())
}

pub fn apple_comgr_release_data(data: apple_comgr_data_t) -> apple_comgr_status_t {
    data_store().lock().unwrap().remove(&data.handle);
    Ok(())
}

pub fn apple_comgr_data_set_bytes(
    data: apple_comgr_data_t,
    bytes: *const c_void,
    size: usize,
) -> apple_comgr_status_t {
    if bytes.is_null() && size > 0 {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    let content = if size == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(bytes.cast::<u8>(), size) }.to_vec()
    };
    let mut store = data_store().lock().unwrap();
    let Some(data_content) = store.get_mut(&data.handle) else {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    };
    data_content.content = content;
    Ok(())
}

pub fn apple_comgr_data_get_bytes(
    data: apple_comgr_data_t,
    bytes: *mut c_void,
    size: *mut usize,
) -> apple_comgr_status_t {
    if size.is_null() {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    let store = data_store().lock().unwrap();
    let Some(data_content) = store.get(&data.handle) else {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    };
    unsafe {
        *size = data_content.content.len();
        if !bytes.is_null() && !data_content.content.is_empty() {
            ptr::copy_nonoverlapping(
                data_content.content.as_ptr(),
                bytes.cast::<u8>(),
                data_content.content.len(),
            );
        }
    }
    Ok(())
}

pub fn apple_comgr_data_set_name(
    data: apple_comgr_data_t,
    name: *const c_char,
) -> apple_comgr_status_t {
    if name.is_null() {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    let mut store = data_store().lock().unwrap();
    let Some(data_content) = store.get_mut(&data.handle) else {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    };
    data_content.name = Some(
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned(),
    );
    Ok(())
}

pub fn apple_comgr_create_data_set(data_set: *mut apple_comgr_data_set_t) -> apple_comgr_status_t {
    if data_set.is_null() {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    let handle = next_handle();
    data_set_store().lock().unwrap().insert(handle, Vec::new());
    unsafe {
        *data_set = apple_comgr_data_set_t { handle };
    }
    Ok(())
}

pub fn apple_comgr_release_data_set(data_set: apple_comgr_data_set_t) -> apple_comgr_status_t {
    data_set_store().lock().unwrap().remove(&data_set.handle);
    Ok(())
}

pub fn apple_comgr_data_set_add(
    data_set: apple_comgr_data_set_t,
    data: apple_comgr_data_t,
) -> apple_comgr_status_t {
    if !data_store().lock().unwrap().contains_key(&data.handle) {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    let mut sets = data_set_store().lock().unwrap();
    let Some(items) = sets.get_mut(&data_set.handle) else {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    };
    items.push(data.handle);
    Ok(())
}

pub fn apple_comgr_get_data_count(
    data_set: apple_comgr_data_set_t,
    count: *mut usize,
) -> apple_comgr_status_t {
    if count.is_null() {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    let sets = data_set_store().lock().unwrap();
    let Some(items) = sets.get(&data_set.handle) else {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    };
    unsafe {
        *count = items.len();
    }
    Ok(())
}

pub fn apple_comgr_get_data(
    data_set: apple_comgr_data_set_t,
    index: usize,
    data: *mut apple_comgr_data_t,
) -> apple_comgr_status_t {
    if data.is_null() {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    let sets = data_set_store().lock().unwrap();
    let Some(items) = sets.get(&data_set.handle) else {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    };
    let Some(handle) = items.get(index).copied() else {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    };
    unsafe {
        *data = apple_comgr_data_t { handle };
    }
    Ok(())
}

pub fn apple_comgr_create_action_info(
    action_info: *mut apple_comgr_action_info_t,
) -> apple_comgr_status_t {
    if action_info.is_null() {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    let handle = next_handle();
    action_info_store()
        .lock()
        .unwrap()
        .insert(handle, ActionInfo::default());
    unsafe {
        *action_info = apple_comgr_action_info_t { handle };
    }
    Ok(())
}

pub fn apple_comgr_release_action_info(
    action_info: apple_comgr_action_info_t,
) -> apple_comgr_status_t {
    action_info_store()
        .lock()
        .unwrap()
        .remove(&action_info.handle);
    Ok(())
}

pub fn apple_comgr_action_info_set_option_list(
    action_info: apple_comgr_action_info_t,
    options: *mut *const c_char,
    count: usize,
) -> apple_comgr_status_t {
    if options.is_null() && count > 0 {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    let mut parsed = Vec::with_capacity(count);
    for idx in 0..count {
        let ptr = unsafe { *options.add(idx) };
        if ptr.is_null() {
            return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
        }
        parsed.push(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        );
    }

    let mut store = action_info_store().lock().unwrap();
    let Some(info) = store.get_mut(&action_info.handle) else {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    };
    info.options = parsed;
    Ok(())
}

pub fn apple_comgr_do_action(
    kind: apple_comgr_action_kind_t,
    action_info: apple_comgr_action_info_t,
    input: apple_comgr_data_set_t,
    output: apple_comgr_data_set_t,
) -> apple_comgr_status_t {
    if kind != apple_comgr_action_kind_s::APPLE_COMGR_ACTION_COMPILE_PTX_TO_MSL {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    if !action_info_store()
        .lock()
        .unwrap()
        .contains_key(&action_info.handle)
    {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let input_handles = {
        let sets = data_set_store().lock().unwrap();
        let Some(handles) = sets.get(&input.handle) else {
            return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
        };
        handles.clone()
    };
    if !data_set_store()
        .lock()
        .unwrap()
        .contains_key(&output.handle)
    {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let input_ptx = {
        let store = data_store().lock().unwrap();
        input_handles
            .iter()
            .filter_map(|handle| store.get(handle))
            .find(|content| {
                matches!(
                    content.kind,
                    apple_comgr_data_kind_s::APPLE_COMGR_DATA_KIND_PTX
                        | apple_comgr_data_kind_s::APPLE_COMGR_DATA_KIND_SOURCE
                )
            })
            .map(|content| content.content.clone())
    };
    let Some(input_ptx) = input_ptx else {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    };

    match apple_comgr_compile_ptx_to_msl_with_diagnostics(&input_ptx) {
        Ok(msl) => add_output_data(
            output,
            apple_comgr_data_kind_s::APPLE_COMGR_DATA_KIND_MSL,
            "module.metal",
            msl,
        ),
        Err(err) => {
            let _ = add_output_data(
                output,
                apple_comgr_data_kind_s::APPLE_COMGR_DATA_KIND_LOG,
                "compile.log",
                err.diagnostics.into_bytes(),
            );
            Err(err.status)
        }
    }
}

fn add_output_data(
    data_set: apple_comgr_data_set_t,
    kind: apple_comgr_data_kind_s,
    name: &str,
    content: Vec<u8>,
) -> apple_comgr_status_t {
    let handle = next_handle();
    data_store().lock().unwrap().insert(
        handle,
        DataContent {
            kind,
            content,
            name: Some(name.to_string()),
        },
    );
    let mut sets = data_set_store().lock().unwrap();
    let Some(items) = sets.get_mut(&data_set.handle) else {
        return Err(apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    };
    items.push(handle);
    Ok(())
}

pub fn apple_comgr_compile_ptx_to_msl(ptx: &[u8]) -> Result<Vec<u8>, apple_comgr_status_s> {
    apple_comgr_compile_ptx_to_msl_with_diagnostics(ptx).map_err(|err| err.status)
}

pub fn apple_comgr_compile_ptx_to_msl_with_diagnostics(
    ptx: &[u8],
) -> Result<Vec<u8>, AppleComgrCompileError> {
    apple_comgr_compile_ptx_to_msl_module(ptx).map(|module| module.msl.into_bytes())
}

pub fn apple_comgr_compile_ptx_to_msl_module(
    ptx: &[u8],
) -> Result<AppleCompiledModule, AppleComgrCompileError> {
    let text = std::str::from_utf8(ptx).map_err(|err| AppleComgrCompileError {
        status: apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_PARSE_FAILED,
        diagnostics: format!("PTX is not UTF-8: {err}"),
    })?;
    compile_ptx_to_msl_module_text(text)
}

pub fn compile_ptx_to_msl_text(ptx: &str) -> Result<String, AppleComgrCompileError> {
    compile_ptx_to_msl_module_text(ptx).map(|module| module.msl)
}

pub fn compile_ptx_to_msl_module_text(
    ptx: &str,
) -> Result<AppleCompiledModule, AppleComgrCompileError> {
    let mut diagnostics = Vec::new();
    let module = match ptx_parser::parse_module_checked(ptx) {
        Ok(module) => module,
        Err(errors) => {
            diagnostics.extend(errors.iter().map(|err| format!("{err:?}")));
            ptx_parser::parse_module_unchecked(ptx)
        }
    };

    compile_module_to_msl(&module, diagnostics)
}

fn compile_module_to_msl<'input>(
    module: &Module<'input>,
    mut diagnostics: Vec<String>,
) -> Result<AppleCompiledModule, AppleComgrCompileError> {
    let mut kernels = Vec::new();
    for directive in &module.directives {
        if let Directive::Method(_, function) = directive {
            if function.func_directive.name.is_kernel() {
                kernels.push(function);
            }
        }
    }
    if kernels.is_empty() {
        diagnostics.push("PTX module contains no .entry kernels".to_string());
        return Err(AppleComgrCompileError {
            status: apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_PARSE_FAILED,
            diagnostics: diagnostics.join("\n"),
        });
    }

    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\n");
    out.push_str("using namespace metal;\n\n");
    out.push_str(PTX_MSL_HELPERS);

    if module.invalid_directives > 0 {
        out.push_str("// PTX parser skipped unsupported non-kernel directives.\n");
    }

    let mut lowering_diagnostics = Vec::new();
    let mut metadata = Vec::new();
    for kernel in kernels {
        let mut lowering = KernelLowering::new(kernel);
        match lowering.emit() {
            Ok(msl) => {
                out.push_str(&msl);
                out.push('\n');
                metadata.push(lowering.metadata());
            }
            Err(mut kernel_diagnostics) => lowering_diagnostics.append(&mut kernel_diagnostics),
        }
    }

    if !lowering_diagnostics.is_empty() {
        diagnostics.append(&mut lowering_diagnostics);
        Err(AppleComgrCompileError {
            status: apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_UNSUPPORTED_PTX,
            diagnostics: diagnostics.join("\n"),
        })
    } else if out.contains("kernel void") {
        Ok(AppleCompiledModule {
            msl: out,
            kernels: metadata,
        })
    } else {
        Err(AppleComgrCompileError {
            status: apple_comgr_status_s::APPLE_COMGR_STATUS_ERROR_PARSE_FAILED,
            diagnostics: diagnostics.join("\n"),
        })
    }
}

type PtxOperand<'a> = ParsedOperand<&'a str>;
type PtxStatement<'a> = Statement<PtxOperand<'a>>;

#[derive(Clone, Debug)]
struct KernelParam {
    ptx_name: String,
    msl_name: String,
    ty: Type,
    is_pointer: bool,
}

#[derive(Clone, Debug)]
enum Def {
    LdParam(String),
    Mov(String),
    Cvta(String, MslAddressSpace),
    Add(String, String),
    Sub(String, String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MslAddressSpace {
    Device,
    Threadgroup,
    Thread,
    Constant,
}

impl MslAddressSpace {
    fn keyword(self) -> &'static str {
        match self {
            MslAddressSpace::Device => "device",
            MslAddressSpace::Threadgroup => "threadgroup",
            MslAddressSpace::Thread => "thread",
            MslAddressSpace::Constant => "constant",
        }
    }
}

struct KernelLowering<'module, 'input> {
    kernel: &'module Function<'input, &'input str, PtxStatement<'input>>,
    params: Vec<KernelParam>,
    var_types: HashMap<String, Type>,
    var_spaces: HashMap<String, StateSpace>,
    defs: HashMap<String, Def>,
    pointer_regs: HashSet<String>,
    pointer_addr_spaces: HashMap<String, MslAddressSpace>,
    pointer_params: HashSet<String>,
    terminal_return_labels: HashSet<String>,
    scalar_aliases: HashMap<String, String>,
    next_label: Option<String>,
    diagnostics: Vec<String>,
    indent: usize,
    temp_counter: usize,
}

impl<'module, 'input> KernelLowering<'module, 'input> {
    fn new(kernel: &'module Function<'input, &'input str, PtxStatement<'input>>) -> Self {
        let params = kernel
            .func_directive
            .input_arguments
            .iter()
            .map(|param| KernelParam {
                ptx_name: param.name.to_string(),
                msl_name: sanitize_ident(param.name),
                ty: param.info.v_type.clone(),
                is_pointer: false,
            })
            .collect();

        let mut lowering = Self {
            kernel,
            params,
            var_types: HashMap::new(),
            var_spaces: HashMap::new(),
            defs: HashMap::new(),
            pointer_regs: HashSet::new(),
            pointer_addr_spaces: HashMap::new(),
            pointer_params: HashSet::new(),
            terminal_return_labels: HashSet::new(),
            scalar_aliases: HashMap::new(),
            next_label: None,
            diagnostics: Vec::new(),
            indent: 1,
            temp_counter: 0,
        };
        if let Some(body) = &kernel.body {
            lowering.collect_declarations(body);
            lowering.collect_defs_and_memory_uses(body);
            lowering.infer_pointers();
            lowering.terminal_return_labels = collect_terminal_return_labels(body);
        }
        lowering
    }

    fn emit(&mut self) -> Result<String, Vec<String>> {
        let Some(body) = &self.kernel.body else {
            return Err(vec![format!(
                "kernel {} has no body",
                self.kernel.func_directive.name()
            )]);
        };

        for param in &mut self.params {
            if self.pointer_params.contains(&param.ptx_name) {
                param.is_pointer = true;
            }
        }

        let mut out = String::new();
        let kernel_name = sanitize_ident(self.kernel.func_directive.name());
        out.push_str(&format!("kernel void {kernel_name}(\n"));
        let mut args = Vec::new();
        for (idx, param) in self.params.iter().enumerate() {
            let arg = if param.is_pointer {
                format!("    device uchar* {} [[buffer({idx})]]", param.msl_name)
            } else {
                format!(
                    "    constant {}& {} [[buffer({idx})]]",
                    msl_type(&param.ty),
                    param.msl_name
                )
            };
            args.push(arg);
        }
        args.push("    uint3 tid [[thread_position_in_threadgroup]]".to_string());
        args.push("    uint3 ctaid [[threadgroup_position_in_grid]]".to_string());
        args.push("    uint3 ntid [[threads_per_threadgroup]]".to_string());
        args.push("    uint3 nctaid [[threadgroups_per_grid]]".to_string());
        out.push_str(&args.join(",\n"));
        out.push_str("\n) {\n");

        self.emit_local_declarations(&mut out);
        self.emit_statements(body, &mut out);
        out.push_str("}\n");

        if self.diagnostics.is_empty() {
            Ok(out)
        } else {
            Err(std::mem::take(&mut self.diagnostics))
        }
    }

    fn metadata(&self) -> AppleKernelMetadata {
        AppleKernelMetadata {
            name: self.kernel.func_directive.name().to_string(),
            msl_name: sanitize_ident(self.kernel.func_directive.name()),
            params: self
                .params
                .iter()
                .map(|param| AppleKernelParamMetadata {
                    ptx_name: param.ptx_name.clone(),
                    msl_name: param.msl_name.clone(),
                    size: if param.is_pointer {
                        8
                    } else {
                        type_size(&param.ty)
                    },
                    is_pointer: param.is_pointer,
                })
                .collect(),
        }
    }

    fn collect_declarations(&mut self, body: &[PtxStatement<'input>]) {
        for statement in body {
            match statement {
                Statement::Variable(variable) => match variable {
                    MultiVariable::Parameterized { info, name, count } => {
                        let prefix = name.trim_end_matches('>');
                        for idx in 0..*count {
                            let raw_name = if prefix.ends_with('<') {
                                format!("{}{}", prefix.trim_end_matches('<'), idx)
                            } else {
                                format!("{name}{idx}")
                            };
                            self.var_types.insert(raw_name.clone(), info.v_type.clone());
                            self.var_spaces.insert(raw_name, info.state_space);
                        }
                    }
                    MultiVariable::Names { info, names } => {
                        for name in names {
                            self.var_types
                                .insert((*name).to_string(), info.v_type.clone());
                            self.var_spaces
                                .insert((*name).to_string(), info.state_space);
                        }
                    }
                },
                Statement::Block(inner) => self.collect_declarations(inner),
                _ => {}
            }
        }
    }

    fn collect_defs_and_memory_uses(&mut self, body: &[PtxStatement<'input>]) {
        for statement in body {
            match statement {
                Statement::Instruction(_, instruction) => self.record_instruction(instruction),
                Statement::Block(inner) => self.collect_defs_and_memory_uses(inner),
                _ => {}
            }
        }
    }

    fn record_instruction(&mut self, instruction: &Instruction<PtxOperand<'input>>) {
        match instruction {
            Instruction::Ld { data, arguments } => {
                if is_param_state_space(data.state_space) {
                    if let (Some(dst), Some(param)) =
                        (operand_reg(&arguments.dst), operand_reg(&arguments.src))
                    {
                        self.defs.insert(dst, Def::LdParam(param));
                    }
                } else if matches!(
                    data.state_space,
                    StateSpace::Global
                        | StateSpace::Generic
                        | StateSpace::Const
                        | StateSpace::Shared
                        | StateSpace::SharedCta
                        | StateSpace::SharedCluster
                        | StateSpace::Local
                ) {
                    if let Some(addr) = operand_reg(&arguments.src) {
                        self.trace_pointer_from(
                            addr,
                            state_space_to_msl_or_device(data.state_space),
                        );
                    }
                }
            }
            Instruction::St { data, arguments } => {
                if !is_param_state_space(data.state_space) {
                    if let Some(addr) = operand_reg(&arguments.src1) {
                        self.trace_pointer_from(
                            addr,
                            state_space_to_msl_or_device(data.state_space),
                        );
                    }
                }
            }
            Instruction::Atom { data, arguments } => {
                if let Some(addr) = operand_reg(&arguments.src1) {
                    self.trace_pointer_from(addr, state_space_to_msl_or_device(data.space));
                }
            }
            Instruction::AtomCas { data, arguments } => {
                if let Some(addr) = operand_reg(&arguments.src1) {
                    self.trace_pointer_from(addr, state_space_to_msl_or_device(data.space));
                }
            }
            Instruction::CpAsync { data, arguments } => {
                if let Some(dst) = operand_reg(&arguments.src_to) {
                    self.trace_pointer_from(dst, state_space_to_msl_or_device(data.space));
                }
                if let Some(src) = operand_reg(&arguments.src_from) {
                    self.trace_pointer_from(src, MslAddressSpace::Device);
                }
            }
            Instruction::Mov { arguments, .. } => {
                if let (Some(dst), Some(src)) =
                    (operand_reg(&arguments.dst), operand_reg(&arguments.src))
                {
                    self.defs.insert(dst, Def::Mov(src));
                }
            }
            Instruction::Cvta { data, arguments } => {
                if let (Some(dst), Some(src)) =
                    (operand_reg(&arguments.dst), operand_reg(&arguments.src))
                {
                    self.defs.insert(
                        dst,
                        Def::Cvta(src, state_space_to_msl_or_device(data.state_space)),
                    );
                }
            }
            Instruction::Add { arguments, .. } => {
                if let (Some(dst), Some(src1), Some(src2)) = (
                    operand_reg(&arguments.dst),
                    operand_reg(&arguments.src1),
                    operand_reg(&arguments.src2),
                ) {
                    self.defs.insert(dst, Def::Add(src1, src2));
                }
            }
            Instruction::Sub { arguments, .. } => {
                if let (Some(dst), Some(src1), Some(src2)) = (
                    operand_reg(&arguments.dst),
                    operand_reg(&arguments.src1),
                    operand_reg(&arguments.src2),
                ) {
                    self.defs.insert(dst, Def::Sub(src1, src2));
                }
            }
            _ => {}
        }
    }

    fn trace_pointer_from(&mut self, reg: String, space: MslAddressSpace) {
        let mut seen = HashSet::new();
        self.trace_pointer_reg(&reg, space, &mut seen);
    }

    fn trace_pointer_reg(&mut self, reg: &str, space: MslAddressSpace, seen: &mut HashSet<String>) {
        if !seen.insert(reg.to_string()) {
            return;
        }
        if self
            .var_spaces
            .get(reg)
            .is_some_and(|state_space| *state_space != StateSpace::Reg)
        {
            return;
        }
        self.pointer_regs.insert(reg.to_string());
        self.pointer_addr_spaces
            .entry(reg.to_string())
            .or_insert(space);
        let Some(def) = self.defs.get(reg).cloned() else {
            return;
        };
        match def {
            Def::LdParam(param) => {
                self.pointer_params.insert(param);
            }
            Def::Mov(src) => self.trace_pointer_reg(&src, space, seen),
            Def::Cvta(src, cvta_space) => self.trace_pointer_reg(&src, cvta_space, seen),
            Def::Add(src1, src2) | Def::Sub(src1, src2) => {
                let src1_reaches = self.reaches_param(&src1, &mut HashSet::new());
                let src2_reaches = self.reaches_param(&src2, &mut HashSet::new());
                match (src1_reaches, src2_reaches) {
                    (true, false) => self.trace_pointer_reg(&src1, space, seen),
                    (false, true) => self.trace_pointer_reg(&src2, space, seen),
                    (true, true) => {
                        if self.prefers_pointer_base(&src1, &src2) {
                            self.trace_pointer_reg(&src1, space, seen);
                        } else {
                            self.trace_pointer_reg(&src2, space, seen);
                        }
                    }
                    (false, false) => self.trace_pointer_reg(&src1, space, seen),
                }
            }
        }
    }

    fn reaches_param(&self, reg: &str, seen: &mut HashSet<String>) -> bool {
        if !seen.insert(reg.to_string()) {
            return false;
        }
        match self.defs.get(reg) {
            Some(Def::LdParam(_)) => true,
            Some(Def::Mov(src)) | Some(Def::Cvta(src, _)) => self.reaches_param(src, seen),
            Some(Def::Add(src1, src2)) | Some(Def::Sub(src1, src2)) => {
                self.reaches_param(src1, seen) || self.reaches_param(src2, seen)
            }
            None => false,
        }
    }

    fn prefers_pointer_base(&self, left: &str, right: &str) -> bool {
        let left_size = self.var_types.get(left).map(type_size);
        let right_size = self.var_types.get(right).map(type_size);
        match (left_size, right_size) {
            (Some(8), Some(size)) if size != 8 => true,
            (Some(size), Some(8)) if size != 8 => false,
            _ => true,
        }
    }

    fn infer_pointers(&mut self) {
        loop {
            let mut changed = false;
            for (dst, def) in &self.defs {
                let becomes_pointer = match def {
                    Def::LdParam(param) => self.pointer_params.contains(param),
                    Def::Mov(src) | Def::Cvta(src, _) => self.pointer_regs.contains(src),
                    Def::Add(src1, src2) | Def::Sub(src1, src2) => {
                        self.pointer_regs.contains(src1) || self.pointer_regs.contains(src2)
                    }
                };
                if becomes_pointer && self.pointer_regs.insert(dst.clone()) {
                    let space = match def {
                        Def::LdParam(_) => MslAddressSpace::Device,
                        Def::Mov(src) => self
                            .pointer_addr_spaces
                            .get(src)
                            .copied()
                            .unwrap_or(MslAddressSpace::Device),
                        Def::Cvta(_, space) => *space,
                        Def::Add(src1, src2) | Def::Sub(src1, src2) => self
                            .pointer_addr_spaces
                            .get(src1)
                            .or_else(|| self.pointer_addr_spaces.get(src2))
                            .copied()
                            .unwrap_or(MslAddressSpace::Device),
                    };
                    self.pointer_addr_spaces.insert(dst.clone(), space);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn emit_local_declarations(&self, out: &mut String) {
        let mut names = self.var_types.keys().cloned().collect::<Vec<_>>();
        names.sort();
        for name in names {
            let ty = &self.var_types[&name];
            let local = sanitize_ident(&name);
            if self.pointer_regs.contains(&name) {
                let space = self
                    .pointer_addr_spaces
                    .get(&name)
                    .copied()
                    .unwrap_or(MslAddressSpace::Device);
                self.line(
                    out,
                    &format!("{} uchar* {local} = nullptr;", space.keyword()),
                );
            } else if let Some(space) = self.var_spaces.get(&name) {
                match state_space_to_msl(*space) {
                    Some(MslAddressSpace::Threadgroup) => {
                        self.line(out, &format!("threadgroup {};", msl_var_decl(ty, &local)));
                    }
                    Some(MslAddressSpace::Thread) => {
                        self.line(out, &format!("{};", msl_var_decl(ty, &local)));
                    }
                    Some(MslAddressSpace::Device | MslAddressSpace::Constant) => {
                        self.line(
                            out,
                            &format!(
                                "/* unsupported {} declaration {} */",
                                space,
                                sanitize_ident(&name)
                            ),
                        );
                    }
                    None => {
                        self.line(
                            out,
                            &format!("{} {local} = {};", msl_type(ty), zero_value(ty)),
                        );
                    }
                }
            } else {
                self.line(
                    out,
                    &format!("{} {local} = {};", msl_type(ty), zero_value(ty)),
                );
            }
        }
        if !self.var_types.is_empty() {
            out.push('\n');
        }
    }

    fn emit_statement(&mut self, statement: &PtxStatement<'input>, out: &mut String) {
        match statement {
            Statement::Label(label) => {
                self.line(out, &format!("/* PTX label {} */", sanitize_label(label)));
            }
            Statement::Instruction(pred, instruction) => {
                self.emit_instruction(pred, instruction, out)
            }
            Statement::Block(inner) => {
                self.emit_statements(inner, out);
            }
            Statement::Variable(_) => {}
        }
    }

    fn emit_statements(&mut self, body: &[PtxStatement<'input>], out: &mut String) {
        let saved_next_label = self.next_label.clone();
        for (idx, statement) in body.iter().enumerate() {
            self.next_label = next_label_after(&body[idx + 1..]);
            self.emit_statement(statement, out);
        }
        self.next_label = saved_next_label;
    }

    fn emit_instruction(
        &mut self,
        pred: &Option<PredAt<&'input str>>,
        instruction: &Instruction<PtxOperand<'input>>,
        out: &mut String,
    ) {
        if let Instruction::Bra { arguments } = instruction {
            self.emit_branch(pred, arguments.src, out);
            return;
        }

        if let Some(pred) = pred {
            let pred_expr = if pred.not {
                format!("!{}", sanitize_ident(pred.label))
            } else {
                sanitize_ident(pred.label)
            };
            self.line(out, &format!("if ({pred_expr}) {{"));
            self.indent += 1;
            self.emit_unpredicated_instruction(instruction, out);
            self.indent -= 1;
            self.line(out, "}");
        } else {
            self.emit_unpredicated_instruction(instruction, out);
        }
    }

    fn emit_unpredicated_instruction(
        &mut self,
        instruction: &Instruction<PtxOperand<'input>>,
        out: &mut String,
    ) {
        match instruction {
            Instruction::Ld { data, arguments } => {
                let dst = self.expr(&arguments.dst);
                if is_param_state_space(data.state_space) {
                    let src = self.expr(&arguments.src);
                    self.line(out, &format!("{dst} = {src};"));
                } else {
                    let src = self.memory_addr_expr(&arguments.src, data.state_space);
                    let space = self
                        .address_space_for_operand(&arguments.src, data.state_space)
                        .keyword();
                    self.line(
                        out,
                        &format!(
                            "{dst} = *reinterpret_cast<{space} {}*>({src});",
                            msl_type(&data.typ)
                        ),
                    );
                }
            }
            Instruction::St { data, arguments } => {
                let dst = self.memory_addr_expr(&arguments.src1, data.state_space);
                let src = self.expr(&arguments.src2);
                let space = self
                    .address_space_for_operand(&arguments.src1, data.state_space)
                    .keyword();
                self.line(
                    out,
                    &format!(
                        "*reinterpret_cast<{space} {}*>({dst}) = {src};",
                        msl_type(&data.typ)
                    ),
                );
            }
            Instruction::Atom { data, arguments } => {
                self.emit_atomic(out, data, &arguments.dst, &arguments.src1, &arguments.src2);
            }
            Instruction::AtomCas { data, arguments } => {
                self.emit_atomic_cas(
                    out,
                    data,
                    &arguments.dst,
                    &arguments.src1,
                    &arguments.src2,
                    &arguments.src3,
                );
            }
            Instruction::Mov { arguments, .. } => {
                self.emit_move_like(out, &arguments.dst, &arguments.src);
            }
            Instruction::Cvta { arguments, .. } => {
                self.emit_move_like(out, &arguments.dst, &arguments.src);
            }
            Instruction::Cvt { data, arguments } => {
                let src = self.scalar_expr(&arguments.src);
                let to = scalar_msl_type(data.to);
                let expr = match data.mode {
                    CvtMode::Bitcast if data.to.size_of() == data.from.size_of() => {
                        format!("as_type<{to}>({src})")
                    }
                    _ => format!("{to}({src})"),
                };
                self.emit_scalar_result(out, &arguments.dst, expr);
            }
            Instruction::Add { arguments, .. } => {
                self.emit_binary(out, &arguments.dst, &arguments.src1, &arguments.src2, "+");
            }
            Instruction::Sub { arguments, .. } => {
                self.emit_binary(out, &arguments.dst, &arguments.src1, &arguments.src2, "-");
            }
            Instruction::Mul { data, arguments } => {
                let src1 = self.scalar_expr(&arguments.src1);
                let src2 = self.scalar_expr(&arguments.src2);
                let expr = match data {
                    MulDetails::Integer {
                        control: MulIntControl::High,
                        ..
                    } => format!("(({src1}) * ({src2})) >> 32"),
                    _ => format!("({src1}) * ({src2})"),
                };
                self.emit_scalar_result(out, &arguments.dst, expr);
            }
            Instruction::Mad { data, arguments } => {
                let src1 = self.scalar_expr(&arguments.src1);
                let src2 = self.scalar_expr(&arguments.src2);
                let src3 = self.scalar_expr(&arguments.src3);
                let expr = match data {
                    ptx_parser::MadDetails::Integer {
                        control: MulIntControl::High,
                        ..
                    } => format!("((({src1}) * ({src2})) >> 32) + ({src3})"),
                    _ => format!("(({src1}) * ({src2})) + ({src3})"),
                };
                self.emit_scalar_result(out, &arguments.dst, expr);
            }
            Instruction::Fma { arguments, .. } => {
                let dst = self.expr(&arguments.dst);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                let src3 = self.expr(&arguments.src3);
                self.line(out, &format!("{dst} = fma({src1}, {src2}, {src3});"));
            }
            Instruction::Div { arguments, .. } => {
                self.emit_binary(out, &arguments.dst, &arguments.src1, &arguments.src2, "/");
            }
            Instruction::Rem { arguments, .. } => {
                self.emit_binary(out, &arguments.dst, &arguments.src1, &arguments.src2, "%");
            }
            Instruction::And { arguments, .. } => {
                self.emit_binary(out, &arguments.dst, &arguments.src1, &arguments.src2, "&");
            }
            Instruction::Or { arguments, .. } => {
                self.emit_binary(out, &arguments.dst, &arguments.src1, &arguments.src2, "|");
            }
            Instruction::Xor { arguments, .. } => {
                self.emit_binary(out, &arguments.dst, &arguments.src1, &arguments.src2, "^");
            }
            Instruction::Shl { arguments, .. } => {
                self.emit_binary(out, &arguments.dst, &arguments.src1, &arguments.src2, "<<");
            }
            Instruction::Shr { data, arguments } => {
                let dst = self.expr(&arguments.dst);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                let expr = match data.kind {
                    RightShiftKind::Arithmetic | RightShiftKind::Logical => {
                        format!("({src1}) >> ({src2})")
                    }
                };
                self.line(out, &format!("{dst} = {expr};"));
            }
            Instruction::Not { data, arguments } => {
                let dst = self.expr(&arguments.dst);
                let src = self.expr(&arguments.src);
                let op = if *data == ScalarType::Pred { "!" } else { "~" };
                self.line(out, &format!("{dst} = {op}{src};"));
            }
            Instruction::Neg { arguments, .. } => {
                let dst = self.expr(&arguments.dst);
                let src = self.expr(&arguments.src);
                self.line(out, &format!("{dst} = -{src};"));
            }
            Instruction::Abs { arguments, .. } => {
                let dst = self.expr(&arguments.dst);
                let src = self.expr(&arguments.src);
                self.line(out, &format!("{dst} = abs({src});"));
            }
            Instruction::Min { arguments, .. } => {
                let dst = self.expr(&arguments.dst);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                self.line(out, &format!("{dst} = min({src1}, {src2});"));
            }
            Instruction::Max { arguments, .. } => {
                let dst = self.expr(&arguments.dst);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                self.line(out, &format!("{dst} = max({src1}, {src2});"));
            }
            Instruction::Set { data, arguments } => {
                let dst = self.expr(&arguments.dst);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                let cmp = compare_expr(data.base.cmp_op, &src1, &src2);
                let value = set_result_expr(data.dtype, &cmp);
                self.line(out, &format!("{dst} = {value};"));
            }
            Instruction::SetBool { data, arguments } => {
                let dst = self.expr(&arguments.dst);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                let src3 = self.expr(&arguments.src3);
                let cmp = compare_expr(data.base.base.cmp_op, &src1, &src2);
                let pred = bool_post_expr(data.base.bool_op, &cmp, &src3, data.base.negate_src3);
                let value = set_result_expr(data.dtype, &pred);
                self.line(out, &format!("{dst} = {value};"));
            }
            Instruction::Setp { data, arguments } => {
                let dst1 = self.expr(&arguments.dst1);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                let cmp = compare_expr(data.cmp_op, &src1, &src2);
                self.line(out, &format!("{dst1} = {cmp};"));
                if let Some(dst2) = &arguments.dst2 {
                    let dst2 = self.expr(dst2);
                    self.line(out, &format!("{dst2} = !({dst1});"));
                }
            }
            Instruction::SetpBool { data, arguments } => {
                let dst1 = self.expr(&arguments.dst1);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                let src3 = self.expr(&arguments.src3);
                let cmp = compare_expr(data.base.cmp_op, &src1, &src2);
                let pred = bool_post_expr(data.bool_op, &cmp, &src3, data.negate_src3);
                self.line(out, &format!("{dst1} = {pred};"));
                if let Some(dst2) = &arguments.dst2 {
                    let dst2 = self.expr(dst2);
                    self.line(out, &format!("{dst2} = !({dst1});"));
                }
            }
            Instruction::Selp { arguments, .. } => {
                let dst = self.expr(&arguments.dst);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                let pred = self.expr(&arguments.src3);
                self.line(out, &format!("{dst} = {pred} ? {src1} : {src2};"));
            }
            Instruction::Bra { arguments } => {
                self.emit_branch(&None, arguments.src, out);
            }
            Instruction::Ret { .. } => self.line(out, "return;"),
            Instruction::Sin { arguments, .. } => {
                self.emit_unary_call(out, &arguments.dst, &arguments.src, "sin")
            }
            Instruction::Cos { arguments, .. } => {
                self.emit_unary_call(out, &arguments.dst, &arguments.src, "cos")
            }
            Instruction::Lg2 { arguments, .. } => {
                self.emit_unary_call(out, &arguments.dst, &arguments.src, "log2")
            }
            Instruction::Ex2 { arguments, .. } => {
                self.emit_unary_call(out, &arguments.dst, &arguments.src, "exp2")
            }
            Instruction::Copysign { arguments, .. } => {
                let dst = self.expr(&arguments.dst);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                self.line(out, &format!("{dst} = copysign({src1}, {src2});"));
            }
            Instruction::Tanh { arguments, .. } => {
                self.emit_unary_call(out, &arguments.dst, &arguments.src, "tanh")
            }
            Instruction::Clz { arguments, .. } => {
                self.emit_unary_call(out, &arguments.dst, &arguments.src, "clz")
            }
            Instruction::Brev { arguments, .. } => {
                self.emit_unary_call(out, &arguments.dst, &arguments.src, "reverse_bits")
            }
            Instruction::Popc { arguments, .. } => {
                self.emit_unary_call(out, &arguments.dst, &arguments.src, "popcount")
            }
            Instruction::Bfe { data, arguments } => {
                self.emit_bfe(
                    out,
                    *data,
                    &arguments.dst,
                    &arguments.src1,
                    &arguments.src2,
                    &arguments.src3,
                );
            }
            Instruction::Bfi { data, arguments } => {
                self.emit_bfi(
                    out,
                    *data,
                    &arguments.dst,
                    &arguments.src1,
                    &arguments.src2,
                    &arguments.src3,
                    &arguments.src4,
                );
            }
            Instruction::Mul24 { data, arguments } => {
                self.emit_mul24(out, data, &arguments.dst, &arguments.src1, &arguments.src2);
            }
            Instruction::Shf { data, arguments } => {
                self.emit_shf(
                    out,
                    data,
                    &arguments.dst,
                    &arguments.src_a,
                    &arguments.src_b,
                    &arguments.src_c,
                );
            }
            Instruction::Activemask { arguments } => {
                let dst = self.expr(&arguments.dst);
                self.line(out, &format!("{dst} = ~uint(0);"));
            }
            Instruction::Nanosleep { .. }
            | Instruction::CpAsyncCommitGroup {}
            | Instruction::CpAsyncWaitGroup { .. }
            | Instruction::CpAsyncWaitAll {} => {
                self.line(out, "/* PTX async/wait hint lowered to no-op on Metal. */");
            }
            Instruction::CpAsync { data, arguments } => {
                self.emit_cp_async(out, data, &arguments.src_to, &arguments.src_from);
            }
            Instruction::Trap {} => self.line(out, "return;"),
            Instruction::Sqrt { arguments, .. } => {
                self.emit_unary_call(out, &arguments.dst, &arguments.src, "sqrt")
            }
            Instruction::Rsqrt { arguments, .. } => {
                self.emit_unary_call(out, &arguments.dst, &arguments.src, "rsqrt")
            }
            Instruction::Rcp { arguments, .. } => {
                let dst = self.expr(&arguments.dst);
                let src = self.expr(&arguments.src);
                self.line(out, &format!("{dst} = 1.0 / ({src});"));
            }
            Instruction::Membar { .. } => {
                self.line(out, "threadgroup_barrier(mem_flags::mem_device);")
            }
            Instruction::Bar { .. } | Instruction::BarWarp { .. } | Instruction::BarRed { .. } => {
                self.line(out, "threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            _ => self.unsupported("instruction"),
        }
    }

    fn emit_atomic(
        &mut self,
        out: &mut String,
        data: &ptx_parser::AtomDetails,
        dst: &PtxOperand<'input>,
        addr: &PtxOperand<'input>,
        value: &PtxOperand<'input>,
    ) {
        let Some(atomic_ty) = atomic_msl_type(&data.type_) else {
            self.unsupported("atomic type");
            return;
        };
        let Some(op) = atomic_fetch_op(data.op) else {
            self.unsupported("atomic op");
            return;
        };

        let dst = self.expr(dst);
        let addr_expr = self.memory_addr_expr(addr, data.space);
        let value = self.expr(value);
        let space = self.address_space_for_operand(addr, data.space);
        let order = atomic_memory_order(data.semantics);
        self.line(
            out,
            &format!(
                "{dst} = {op}(reinterpret_cast<{} {atomic_ty}*>({addr_expr}), {value}, {order});",
                space.keyword()
            ),
        );
    }

    fn emit_atomic_cas(
        &mut self,
        out: &mut String,
        data: &ptx_parser::AtomCasDetails,
        dst: &PtxOperand<'input>,
        addr: &PtxOperand<'input>,
        compare: &PtxOperand<'input>,
        value: &PtxOperand<'input>,
    ) {
        let Some(atomic_ty) = atomic_msl_scalar_type(data.type_) else {
            self.unsupported("atomic cas type");
            return;
        };
        let scalar_ty = scalar_msl_type(data.type_);
        let dst_expr = self.expr(dst);
        let addr_expr = self.memory_addr_expr(addr, data.space);
        let compare_expr = self.expr(compare);
        let value_expr = self.expr(value);
        let space = self.address_space_for_operand(addr, data.space);
        let order = atomic_memory_order(data.semantics);
        let temp = self.next_temp("cas_expected");

        self.line(out, "{");
        self.indent += 1;
        self.line(out, &format!("{scalar_ty} {temp} = {compare_expr};"));
        self.line(
            out,
            &format!(
                "atomic_compare_exchange_weak_explicit(reinterpret_cast<{} {atomic_ty}*>({addr_expr}), &{temp}, {value_expr}, {order}, {order});",
                space.keyword()
            ),
        );
        self.line(out, &format!("{dst_expr} = {temp};"));
        self.indent -= 1;
        self.line(out, "}");
    }

    fn emit_bfe(
        &self,
        out: &mut String,
        data: ScalarType,
        dst: &PtxOperand<'input>,
        src1: &PtxOperand<'input>,
        src2: &PtxOperand<'input>,
        src3: &PtxOperand<'input>,
    ) {
        let dst = self.expr(dst);
        let src1 = self.expr(src1);
        let src2 = self.expr(src2);
        let src3 = self.expr(src3);
        let helper = match data {
            ScalarType::U32 => "zluda_bfe_u32",
            ScalarType::S32 => "zluda_bfe_s32",
            ScalarType::U64 => "zluda_bfe_u64",
            ScalarType::S64 => "zluda_bfe_s64",
            _ => {
                return;
            }
        };
        self.line(
            out,
            &format!("{dst} = {helper}({src1}, uint({src2}), uint({src3}));"),
        );
    }

    fn emit_bfi(
        &self,
        out: &mut String,
        data: ScalarType,
        dst: &PtxOperand<'input>,
        src1: &PtxOperand<'input>,
        src2: &PtxOperand<'input>,
        src3: &PtxOperand<'input>,
        src4: &PtxOperand<'input>,
    ) {
        let dst = self.expr(dst);
        let src1 = self.expr(src1);
        let src2 = self.expr(src2);
        let src3 = self.expr(src3);
        let src4 = self.expr(src4);
        let helper = match data {
            ScalarType::B32 => "zluda_bfi_b32",
            ScalarType::B64 => "zluda_bfi_b64",
            _ => {
                return;
            }
        };
        self.line(
            out,
            &format!("{dst} = {helper}({src1}, {src2}, uint({src3}), uint({src4}));"),
        );
    }

    fn emit_mul24(
        &self,
        out: &mut String,
        data: &ptx_parser::Mul24Details,
        dst: &PtxOperand<'input>,
        src1: &PtxOperand<'input>,
        src2: &PtxOperand<'input>,
    ) {
        let dst = self.expr(dst);
        let src1 = self.expr(src1);
        let src2 = self.expr(src2);
        let product = match data.type_ {
            ScalarType::U32 => {
                format!("(ulong(({src1}) & 0x00ffffffu) * ulong(({src2}) & 0x00ffffffu))")
            }
            ScalarType::S32 => {
                let lhs = format!("(long(int(({src1}) << 8)) >> 8)");
                let rhs = format!("(long(int(({src2}) << 8)) >> 8)");
                format!("({lhs} * {rhs})")
            }
            _ => {
                return;
            }
        };
        let expr = match (data.type_, data.control) {
            (ScalarType::U32, Mul24Control::Lo) => format!("uint({product})"),
            (ScalarType::U32, Mul24Control::Hi) => format!("uint(({product}) >> 16)"),
            (ScalarType::S32, Mul24Control::Lo) => format!("int({product})"),
            (ScalarType::S32, Mul24Control::Hi) => format!("int(({product}) >> 16)"),
            _ => return,
        };
        self.line(out, &format!("{dst} = {expr};"));
    }

    fn emit_shf(
        &mut self,
        out: &mut String,
        data: &ptx_parser::ShfDetails,
        dst: &PtxOperand<'input>,
        src_a: &PtxOperand<'input>,
        src_b: &PtxOperand<'input>,
        src_c: &PtxOperand<'input>,
    ) {
        let dst = self.expr(dst);
        let src_a = self.expr(src_a);
        let src_b = self.expr(src_b);
        let src_c = self.expr(src_c);
        let a = self.next_temp("shf_a");
        let b = self.next_temp("shf_b");
        let shift = self.next_temp("shf_shift");
        let masked = self.next_temp("shf_masked");
        let inverse = self.next_temp("shf_inverse");
        let shifted = self.next_temp("shf_shifted");

        self.line(out, "{");
        self.indent += 1;
        self.line(out, &format!("uint {a} = uint({src_a});"));
        self.line(out, &format!("uint {b} = uint({src_b});"));
        self.line(out, &format!("uint {shift} = uint({src_c});"));
        self.line(out, &format!("uint {masked} = {shift} & 31u;"));
        self.line(out, &format!("uint {inverse} = (32u - {masked}) & 31u;"));
        let merge = match data.direction {
            ShiftDirection::L => format!("(({b} << {masked}) | ({a} >> {inverse}))"),
            ShiftDirection::R => format!("(({a} >> {masked}) | ({b} << {inverse}))"),
        };
        self.line(out, &format!("uint {shifted} = {merge};"));
        let expr = if data.mode == FunnelShiftMode::Clamp {
            let clamp_value = match data.direction {
                ShiftDirection::L => &a,
                ShiftDirection::R => &b,
            };
            format!("({shift} >= 32u) ? {clamp_value} : {shifted}")
        } else {
            shifted
        };
        self.line(out, &format!("{dst} = {expr};"));
        self.indent -= 1;
        self.line(out, "}");
    }

    fn emit_cp_async(
        &mut self,
        out: &mut String,
        data: &ptx_parser::CpAsyncDetails,
        dst: &PtxOperand<'input>,
        src: &PtxOperand<'input>,
    ) {
        let dst_addr = self.memory_addr_expr(dst, data.space);
        let src_addr = self.memory_addr_expr(src, StateSpace::Global);
        let copy_size = data.cp_size.as_u64();
        let valid_size = data.src_size.unwrap_or(copy_size).min(copy_size);

        self.line(out, "{");
        self.indent += 1;
        for idx in 0..copy_size {
            if idx < valid_size {
                self.line(out, &format!("{dst_addr}[{idx}] = {src_addr}[{idx}];"));
            } else {
                self.line(out, &format!("{dst_addr}[{idx}] = uchar(0);"));
            }
        }
        self.indent -= 1;
        self.line(out, "}");
    }

    fn emit_binary(
        &mut self,
        out: &mut String,
        dst: &PtxOperand<'input>,
        src1: &PtxOperand<'input>,
        src2: &PtxOperand<'input>,
        op: &str,
    ) {
        if self.is_pointer_destination(dst) {
            let src1_is_pointer = self.is_pointer_operand(src1);
            let src2_is_pointer = self.is_pointer_operand(src2);
            match (op, src1_is_pointer, src2_is_pointer) {
                ("+", true, false) | ("-", true, false) => {
                    let dst_expr = self.expr(dst);
                    let src1_expr = self.expr(src1);
                    let src2_expr = self.scalar_expr(src2);
                    if let Some(reg) = operand_reg(dst) {
                        self.scalar_aliases.remove(&reg);
                    }
                    self.line(
                        out,
                        &format!("{dst_expr} = ({src1_expr}) {op} ({src2_expr});"),
                    );
                    return;
                }
                ("+", false, true) => {
                    let dst_expr = self.expr(dst);
                    let src1_expr = self.scalar_expr(src1);
                    let src2_expr = self.expr(src2);
                    if let Some(reg) = operand_reg(dst) {
                        self.scalar_aliases.remove(&reg);
                    }
                    self.line(out, &format!("{dst_expr} = ({src2_expr}) + ({src1_expr});"));
                    return;
                }
                ("+" | "-" | "&" | "|" | "^" | "<<" | ">>" | "*" | "/" | "%", false, false) => {
                    let expr = format!(
                        "({}) {op} ({})",
                        self.scalar_expr(src1),
                        self.scalar_expr(src2)
                    );
                    self.emit_scalar_result(out, dst, expr);
                    return;
                }
                _ => {
                    self.diagnostics.push(format!(
                        "kernel {} contains unsupported PTX pointer arithmetic",
                        self.kernel.func_directive.name()
                    ));
                    self.line(out, "/* unsupported PTX pointer arithmetic */");
                    return;
                }
            }
        }

        let dst_expr = self.expr(dst);
        let src1_expr = self.scalar_expr(src1);
        let src2_expr = self.scalar_expr(src2);
        if let Some(reg) = operand_reg(dst) {
            self.scalar_aliases.remove(&reg);
        }
        self.line(
            out,
            &format!("{dst_expr} = ({src1_expr}) {op} ({src2_expr});"),
        );
    }

    fn emit_move_like(
        &mut self,
        out: &mut String,
        dst: &PtxOperand<'input>,
        src: &PtxOperand<'input>,
    ) {
        if self.is_pointer_destination(dst) && !self.is_pointer_operand(src) {
            self.emit_scalar_result(out, dst, self.scalar_expr(src));
            return;
        }

        if let Some(reg) = operand_reg(dst) {
            self.scalar_aliases.remove(&reg);
        }
        let dst = self.expr(dst);
        let src = self.expr(src);
        self.line(out, &format!("{dst} = {src};"));
    }

    fn emit_scalar_result(&mut self, out: &mut String, dst: &PtxOperand<'input>, expr: String) {
        if self.is_pointer_destination(dst) {
            if let Some(reg) = operand_reg(dst) {
                self.scalar_aliases.insert(reg, expr);
            }
            return;
        }

        if let Some(reg) = operand_reg(dst) {
            self.scalar_aliases.remove(&reg);
        }
        let dst = self.expr(dst);
        self.line(out, &format!("{dst} = {expr};"));
    }

    fn emit_branch(
        &mut self,
        pred: &Option<PredAt<&'input str>>,
        target: &'input str,
        out: &mut String,
    ) {
        if self.next_label.as_deref() == Some(target) {
            self.line(
                out,
                &format!(
                    "/* PTX branch to next label {} lowered to fallthrough. */",
                    sanitize_label(target)
                ),
            );
            return;
        }

        if !self.terminal_return_labels.contains(target) {
            self.diagnostics.push(format!(
                "kernel {} contains unsupported PTX branch to non-terminal label {}",
                self.kernel.func_directive.name(),
                sanitize_label(target)
            ));
            self.line(
                out,
                &format!("/* unsupported PTX branch to {} */", sanitize_label(target)),
            );
            return;
        }

        if let Some(pred) = pred {
            let pred_expr = if pred.not {
                format!("!{}", sanitize_ident(pred.label))
            } else {
                sanitize_ident(pred.label)
            };
            self.line(out, &format!("if ({pred_expr}) {{"));
            self.indent += 1;
            self.line(out, "return;");
            self.indent -= 1;
            self.line(out, "}");
        } else {
            self.line(out, "return;");
        }
    }

    fn is_pointer_operand(&self, operand: &PtxOperand<'input>) -> bool {
        match operand {
            ParsedOperand::Reg(id) | ParsedOperand::RegOffset(id, _) => {
                self.pointer_regs.contains(*id) && !self.scalar_aliases.contains_key(*id)
            }
            _ => false,
        }
    }

    fn is_pointer_destination(&self, operand: &PtxOperand<'input>) -> bool {
        match operand {
            ParsedOperand::Reg(id) | ParsedOperand::RegOffset(id, _) => {
                self.pointer_regs.contains(*id)
            }
            _ => false,
        }
    }

    fn emit_unary_call(
        &self,
        out: &mut String,
        dst: &PtxOperand<'input>,
        src: &PtxOperand<'input>,
        name: &str,
    ) {
        let dst = self.expr(dst);
        let src = self.expr(src);
        self.line(out, &format!("{dst} = {name}({src});"));
    }

    fn next_temp(&mut self, prefix: &str) -> String {
        let current = self.temp_counter;
        self.temp_counter += 1;
        format!("{prefix}_{current}")
    }

    fn expr(&self, operand: &PtxOperand<'input>) -> String {
        match operand {
            ParsedOperand::Reg(id) => special_or_ident(id).unwrap_or_else(|| sanitize_ident(id)),
            ParsedOperand::RegOffset(id, offset) => {
                let base = special_or_ident(id).unwrap_or_else(|| sanitize_ident(id));
                if *offset >= 0 {
                    format!("({base} + {offset})")
                } else {
                    format!("({base} - {})", offset.unsigned_abs())
                }
            }
            ParsedOperand::Imm(value) => immediate_expr(*value),
            ParsedOperand::VecMember(id, idx) => special_member_or_ident(id, *idx)
                .unwrap_or_else(|| format!("{}.{}", sanitize_ident(id), vector_suffix(*idx))),
            ParsedOperand::VecPack(items) => {
                let values = items
                    .iter()
                    .map(|item| match item {
                        RegOrImmediate::Reg(id) => sanitize_ident(id),
                        RegOrImmediate::Imm(value) => immediate_expr(*value),
                        RegOrImmediate::Discard => "0".to_string(),
                    })
                    .collect::<Vec<_>>();
                format!("{{{}}}", values.join(", "))
            }
        }
    }

    fn scalar_expr(&self, operand: &PtxOperand<'input>) -> String {
        match operand {
            ParsedOperand::Reg(id) => self
                .scalar_aliases
                .get(*id)
                .cloned()
                .unwrap_or_else(|| self.expr(operand)),
            ParsedOperand::RegOffset(id, offset) => {
                let base = self
                    .scalar_aliases
                    .get(*id)
                    .cloned()
                    .unwrap_or_else(|| self.expr(&ParsedOperand::Reg(*id)));
                if *offset >= 0 {
                    format!("({base} + {offset})")
                } else {
                    format!("({base} - {})", offset.unsigned_abs())
                }
            }
            _ => self.expr(operand),
        }
    }

    fn memory_addr_expr(&self, operand: &PtxOperand<'input>, state_space: StateSpace) -> String {
        match operand {
            ParsedOperand::Reg(id) => self.memory_reg_addr_expr(id, state_space),
            ParsedOperand::RegOffset(id, offset) => {
                let base = self.memory_reg_addr_expr(id, state_space);
                if *offset >= 0 {
                    format!("({base} + {offset})")
                } else {
                    format!("({base} - {})", offset.unsigned_abs())
                }
            }
            _ => self.expr(operand),
        }
    }

    fn memory_reg_addr_expr(&self, id: &str, state_space: StateSpace) -> String {
        let name = sanitize_ident(id);
        let space = self.address_space_for_reg(id, state_space).keyword();
        if self
            .var_spaces
            .get(id)
            .is_some_and(|decl_space| *decl_space != StateSpace::Reg)
        {
            let needs_address = self
                .var_types
                .get(id)
                .is_some_and(|ty| !matches!(ty, Type::Array(_, _, _)));
            if needs_address {
                format!("reinterpret_cast<{space} uchar*>(&{name})")
            } else {
                format!("reinterpret_cast<{space} uchar*>({name})")
            }
        } else {
            special_or_ident(id).unwrap_or_else(|| sanitize_ident(id))
        }
    }

    fn address_space_for_operand(
        &self,
        operand: &PtxOperand<'input>,
        state_space: StateSpace,
    ) -> MslAddressSpace {
        match operand {
            ParsedOperand::Reg(id) | ParsedOperand::RegOffset(id, _) => {
                self.address_space_for_reg(id, state_space)
            }
            _ => state_space_to_msl_or_device(state_space),
        }
    }

    fn address_space_for_reg(&self, id: &str, state_space: StateSpace) -> MslAddressSpace {
        if let Some(space) = self.pointer_addr_spaces.get(id).copied() {
            return space;
        }
        if let Some(space) = self
            .var_spaces
            .get(id)
            .and_then(|decl_space| state_space_to_msl(*decl_space))
        {
            return space;
        }
        state_space_to_msl_or_device(state_space)
    }

    fn line(&self, out: &mut String, text: &str) {
        for _ in 0..self.indent {
            out.push_str("    ");
        }
        out.push_str(text);
        out.push('\n');
    }

    fn unsupported(&mut self, name: &str) {
        self.diagnostics.push(format!(
            "kernel {} contains unsupported PTX {name}",
            self.kernel.func_directive.name()
        ));
    }
}

fn collect_terminal_return_labels<'input>(body: &[PtxStatement<'input>]) -> HashSet<String> {
    let mut labels = HashSet::new();
    collect_terminal_return_labels_in_block(body, &mut labels);
    labels
}

fn collect_terminal_return_labels_in_block<'input>(
    body: &[PtxStatement<'input>],
    labels: &mut HashSet<String>,
) {
    for (idx, statement) in body.iter().enumerate() {
        match statement {
            Statement::Label(label) => {
                if next_statement_is_unconditional_return(&body[idx + 1..]) {
                    labels.insert((*label).to_string());
                }
            }
            Statement::Block(inner) => collect_terminal_return_labels_in_block(inner, labels),
            _ => {}
        }
    }
}

fn next_statement_is_unconditional_return<'input>(body: &[PtxStatement<'input>]) -> bool {
    for statement in body {
        match statement {
            Statement::Variable(_) => continue,
            Statement::Instruction(None, Instruction::Ret { .. }) => return true,
            Statement::Block(inner) => return next_statement_is_unconditional_return(inner),
            _ => return false,
        }
    }
    false
}

fn next_label_after<'input>(body: &[PtxStatement<'input>]) -> Option<String> {
    for statement in body {
        match statement {
            Statement::Variable(_) => continue,
            Statement::Label(label) => return Some((*label).to_string()),
            _ => return None,
        }
    }
    None
}

fn state_space_to_msl(state_space: StateSpace) -> Option<MslAddressSpace> {
    match state_space {
        StateSpace::Global | StateSpace::Generic => Some(MslAddressSpace::Device),
        StateSpace::Const => Some(MslAddressSpace::Constant),
        StateSpace::Shared | StateSpace::SharedCta | StateSpace::SharedCluster => {
            Some(MslAddressSpace::Threadgroup)
        }
        StateSpace::Local => Some(MslAddressSpace::Thread),
        StateSpace::Reg | StateSpace::Param | StateSpace::ParamFunc | StateSpace::ParamEntry => {
            None
        }
    }
}

fn state_space_to_msl_or_device(state_space: StateSpace) -> MslAddressSpace {
    state_space_to_msl(state_space).unwrap_or(MslAddressSpace::Device)
}

fn is_param_state_space(state_space: StateSpace) -> bool {
    matches!(
        state_space,
        StateSpace::Param | StateSpace::ParamFunc | StateSpace::ParamEntry
    )
}

fn msl_var_decl(ty: &Type, name: &str) -> String {
    match ty {
        Type::Scalar(scalar) => format!("{} {name}", scalar_msl_type(*scalar)),
        Type::Vector(count, scalar) => format!("{}{} {name}", scalar_msl_type(*scalar), count),
        Type::Array(vector, scalar, dims) => {
            let elem_ty = match vector {
                Some(count) => format!("{}{}", scalar_msl_type(*scalar), count),
                None => scalar_msl_type(*scalar).to_string(),
            };
            let len = dims.iter().copied().reduce(std::ops::Mul::mul).unwrap_or(0);
            format!("{elem_ty} {name}[{len}]")
        }
        Type::Pointer(scalar, space) => {
            let msl_space = state_space_to_msl_or_device(*space).keyword();
            format!("{msl_space} {}* {name}", scalar_msl_type(*scalar))
        }
    }
}

fn atomic_msl_type(ty: &Type) -> Option<&'static str> {
    match ty {
        Type::Scalar(scalar) => atomic_msl_scalar_type(*scalar),
        _ => None,
    }
}

fn atomic_msl_scalar_type(scalar: ScalarType) -> Option<&'static str> {
    match scalar {
        ScalarType::B32 | ScalarType::U32 => Some("atomic_uint"),
        ScalarType::S32 => Some("atomic_int"),
        ScalarType::B64 | ScalarType::U64 => Some("atomic_ulong"),
        ScalarType::S64 => Some("atomic_long"),
        _ => None,
    }
}

fn atomic_fetch_op(op: AtomicOp) -> Option<&'static str> {
    match op {
        AtomicOp::And => Some("atomic_fetch_and_explicit"),
        AtomicOp::Or => Some("atomic_fetch_or_explicit"),
        AtomicOp::Xor => Some("atomic_fetch_xor_explicit"),
        AtomicOp::Exchange => Some("atomic_exchange_explicit"),
        AtomicOp::Add => Some("atomic_fetch_add_explicit"),
        AtomicOp::SignedMin | AtomicOp::UnsignedMin => Some("atomic_fetch_min_explicit"),
        AtomicOp::SignedMax | AtomicOp::UnsignedMax => Some("atomic_fetch_max_explicit"),
        AtomicOp::IncrementWrap
        | AtomicOp::DecrementWrap
        | AtomicOp::FloatAdd
        | AtomicOp::FloatMin
        | AtomicOp::FloatMax => None,
    }
}

fn atomic_memory_order(semantics: AtomSemantics) -> &'static str {
    match semantics {
        AtomSemantics::Relaxed => "memory_order_relaxed",
        AtomSemantics::Acquire => "memory_order_acquire",
        AtomSemantics::Release => "memory_order_release",
        AtomSemantics::AcqRel => "memory_order_acq_rel",
    }
}

fn operand_reg(operand: &PtxOperand<'_>) -> Option<String> {
    match operand {
        ParsedOperand::Reg(id)
        | ParsedOperand::RegOffset(id, _)
        | ParsedOperand::VecMember(id, _) => Some((*id).to_string()),
        _ => None,
    }
}

fn sanitize_ident(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.trim_start_matches('%').chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('_');
    }
    if out.as_bytes()[0].is_ascii_digit() {
        out.insert(0, '_');
    }
    out
}

fn sanitize_label(raw: &str) -> String {
    sanitize_ident(raw)
}

fn special_or_ident(raw: &str) -> Option<String> {
    match raw {
        "%warpsize" => Some("uint(32)".to_string()),
        "%laneid" => Some("uint(0)".to_string()),
        _ => None,
    }
}

fn special_member_or_ident(raw: &str, idx: u8) -> Option<String> {
    let suffix = vector_suffix(idx);
    match raw {
        "%tid" => Some(format!("tid.{suffix}")),
        "%ntid" => Some(format!("ntid.{suffix}")),
        "%ctaid" => Some(format!("ctaid.{suffix}")),
        "%nctaid" => Some(format!("nctaid.{suffix}")),
        _ => None,
    }
}

fn vector_suffix(idx: u8) -> &'static str {
    match idx {
        0 => "x",
        1 => "y",
        2 => "z",
        3 => "w",
        _ => "x",
    }
}

fn immediate_expr(value: ImmediateValue) -> String {
    match value {
        ImmediateValue::U64(value) => value.to_string(),
        ImmediateValue::S64(value) => value.to_string(),
        ImmediateValue::F32(value) => {
            if value.is_finite() {
                let text = value.to_string();
                if text.contains('.') {
                    format!("{text}f")
                } else {
                    format!("{text}.0f")
                }
            } else if value.is_nan() {
                "NAN".to_string()
            } else if value.is_sign_positive() {
                "INFINITY".to_string()
            } else {
                "-INFINITY".to_string()
            }
        }
        ImmediateValue::F64(value) => value.to_string(),
    }
}

fn type_size(ty: &Type) -> usize {
    ty.layout().size()
}

fn msl_type(ty: &Type) -> String {
    match ty {
        Type::Scalar(scalar) => scalar_msl_type(*scalar).to_string(),
        Type::Vector(count, scalar) => format!("{}{}", scalar_msl_type(*scalar), count),
        Type::Array(_, scalar, _) => scalar_msl_type(*scalar).to_string(),
        Type::Pointer(scalar, _) => format!("device {}*", scalar_msl_type(*scalar)),
    }
}

fn scalar_msl_type(scalar: ScalarType) -> &'static str {
    match scalar {
        ScalarType::Pred => "bool",
        ScalarType::U8 | ScalarType::B8 => "uchar",
        ScalarType::S8 => "char",
        ScalarType::U16 | ScalarType::B16 | ScalarType::U16x2 => "ushort",
        ScalarType::S16 | ScalarType::S16x2 => "short",
        ScalarType::U32 | ScalarType::B32 => "uint",
        ScalarType::S32 => "int",
        ScalarType::U64 | ScalarType::B64 => "ulong",
        ScalarType::S64 => "long",
        ScalarType::F16 | ScalarType::BF16 | ScalarType::E4m3x2 | ScalarType::E5m2x2 => "half",
        ScalarType::F16x2 | ScalarType::BF16x2 => "half2",
        ScalarType::F32 => "float",
        ScalarType::F64 => "double",
        ScalarType::B128 => "uint4",
    }
}

fn zero_value(ty: &Type) -> String {
    match ty {
        Type::Scalar(ScalarType::Pred) => "false".to_string(),
        Type::Scalar(scalar) => format!("{}(0)", scalar_msl_type(*scalar)),
        Type::Vector(count, scalar) => format!("{}{}(0)", scalar_msl_type(*scalar), count),
        Type::Array(_, scalar, _) => format!("{}(0)", scalar_msl_type(*scalar)),
        Type::Pointer(_, _) => "nullptr".to_string(),
    }
}

fn set_result_expr(dtype: ScalarType, pred: &str) -> String {
    let ty = scalar_msl_type(dtype);
    let true_value = if dtype.kind() == ptx_parser::ScalarKind::Float {
        format!("{ty}(1.0)")
    } else {
        format!("{ty}(-1)")
    };
    format!("({pred}) ? {true_value} : {ty}(0)")
}

fn bool_post_expr(op: SetpBoolPostOp, lhs: &str, rhs: &str, negate_rhs: bool) -> String {
    let rhs = if negate_rhs {
        format!("!({rhs})")
    } else {
        format!("({rhs})")
    };
    let op = match op {
        SetpBoolPostOp::And => "&&",
        SetpBoolPostOp::Or => "||",
        SetpBoolPostOp::Xor => "!=",
    };
    format!("(({lhs}) {op} {rhs})")
}

fn compare_expr(cmp: SetpCompareOp, src1: &str, src2: &str) -> String {
    match cmp {
        SetpCompareOp::Integer(op) => match op {
            SetpCompareInt::Eq => format!("({src1}) == ({src2})"),
            SetpCompareInt::NotEq => format!("({src1}) != ({src2})"),
            SetpCompareInt::UnsignedLess | SetpCompareInt::SignedLess => {
                format!("({src1}) < ({src2})")
            }
            SetpCompareInt::UnsignedLessOrEq | SetpCompareInt::SignedLessOrEq => {
                format!("({src1}) <= ({src2})")
            }
            SetpCompareInt::UnsignedGreater | SetpCompareInt::SignedGreater => {
                format!("({src1}) > ({src2})")
            }
            SetpCompareInt::UnsignedGreaterOrEq | SetpCompareInt::SignedGreaterOrEq => {
                format!("({src1}) >= ({src2})")
            }
        },
        SetpCompareOp::Float(op) => match op {
            SetpCompareFloat::Eq | SetpCompareFloat::NanEq => format!("({src1}) == ({src2})"),
            SetpCompareFloat::NotEq | SetpCompareFloat::NanNotEq => {
                format!("({src1}) != ({src2})")
            }
            SetpCompareFloat::Less | SetpCompareFloat::NanLess => format!("({src1}) < ({src2})"),
            SetpCompareFloat::LessOrEq | SetpCompareFloat::NanLessOrEq => {
                format!("({src1}) <= ({src2})")
            }
            SetpCompareFloat::Greater | SetpCompareFloat::NanGreater => {
                format!("({src1}) > ({src2})")
            }
            SetpCompareFloat::GreaterOrEq | SetpCompareFloat::NanGreaterOrEq => {
                format!("({src1}) >= ({src2})")
            }
            SetpCompareFloat::IsNotNan => format!("!isnan({src1}) && !isnan({src2})"),
            SetpCompareFloat::IsAnyNan => format!("isnan({src1}) || isnan({src2})"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metal_compiler_candidates() -> Vec<std::path::PathBuf> {
        let mut candidates = Vec::new();
        if let Ok(path) = std::env::var("HETGPU_METALC") {
            candidates.push(std::path::PathBuf::from(path));
        }
        if let Ok(output) = std::process::Command::new("xcrun")
            .args(["-sdk", "iphoneos", "--find", "metal"])
            .output()
        {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    candidates.push(std::path::PathBuf::from(path));
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir("/private/var/run/com.apple.security.cryptexd/mnt") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if !name
                    .to_string_lossy()
                    .starts_with("com.apple.MobileAsset.MetalToolchain-")
                {
                    continue;
                }
                candidates.push(entry.path().join("Metal.xctoolchain/usr/bin/metal"));
            }
        }
        candidates.retain(|path| path.is_file());
        candidates.sort();
        candidates.dedup();
        candidates
    }

    fn compile_msl_with_metal(test_name: &str, source: &str) -> Result<(), String> {
        let candidates = metal_compiler_candidates();
        if candidates.is_empty() {
            return Err("no Metal compiler found".to_string());
        }

        let temp_dir = std::env::temp_dir().join(format!(
            "hetgpu_apple_msl_{}_{}",
            std::process::id(),
            test_name
        ));
        std::fs::create_dir_all(&temp_dir).map_err(|err| err.to_string())?;
        let source_path = temp_dir.join("module.metal");
        let air_path = temp_dir.join("module.air");
        let module_cache = temp_dir.join("ModuleCache");
        std::fs::create_dir_all(&module_cache).map_err(|err| err.to_string())?;
        std::fs::write(&source_path, source).map_err(|err| err.to_string())?;

        let mut failures = Vec::new();
        for compiler in candidates {
            let output = std::process::Command::new(&compiler)
                .arg(format!(
                    "-fmodules-cache-path={}",
                    module_cache.to_string_lossy()
                ))
                .arg("-std=ios-metal2.4")
                .arg("-c")
                .arg(&source_path)
                .arg("-o")
                .arg(&air_path)
                .output()
                .map_err(|err| format!("failed to run {}: {err}", compiler.display()))?;
            if output.status.success() {
                let _ = std::fs::remove_dir_all(&temp_dir);
                return Ok(());
            }
            failures.push(format!(
                "{} failed:\n{}{}",
                compiler.display(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let _ = std::fs::remove_dir_all(&temp_dir);
        Err(failures.join("\n"))
    }

    #[test]
    fn compiles_vector_add_to_msl() {
        let ptx = include_str!("../../../ptx/src/test/vectorAdd_kernel64.ptx");
        let module =
            compile_ptx_to_msl_module_text(ptx).expect("vector add PTX should lower to MSL");
        assert!(module.msl.contains("kernel void VecAdd_kernel"));
        assert!(module.msl.contains("device uchar* VecAdd_kernel_param_0"));
        assert!(module.msl.contains("reinterpret_cast<device float*>"));
        assert!(!module.msl.contains("goto "));
        assert!(module.msl.contains("if (p2)"));
        assert!(module.msl.contains("return;"));
        assert_eq!(module.kernels.len(), 1);
        assert_eq!(module.kernels[0].name, "VecAdd_kernel");
        assert_eq!(module.kernels[0].params.len(), 4);
        assert!(module.kernels[0].params[0].is_pointer);
        assert_eq!(module.kernels[0].params[3].size, 4);
    }

    #[test]
    fn lowers_shared_memory_to_threadgroup() {
        let ptx = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry shared_roundtrip(
    .param .u64 out
)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<3>;
    .shared .align 4 .b32 scratch[1];

    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    st.shared.u32 [scratch], %r1;
    bar.sync 0;
    ld.shared.u32 %r2, [scratch];
    st.global.u32 [%rd1], %r2;
    ret;
}
"#;
        let module = compile_ptx_to_msl_module_text(ptx)
            .expect("shared memory PTX should lower to threadgroup MSL");
        assert!(module.msl.contains("threadgroup uint scratch[1];"));
        assert!(module
            .msl
            .contains("reinterpret_cast<threadgroup uchar*>(scratch)"));
        assert!(module.msl.contains("reinterpret_cast<threadgroup uint*>"));
        assert!(module
            .msl
            .contains("threadgroup_barrier(mem_flags::mem_threadgroup);"));
    }

    #[test]
    fn lowers_local_memory_to_thread_storage() {
        let ptx = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry local_roundtrip(
    .param .u64 out
)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<3>;
    .local .align 4 .b32 slot[1];

    ld.param.u64 %rd1, [out];
    mov.u32 %r1, 7;
    st.local.u32 [slot], %r1;
    ld.local.u32 %r2, [slot];
    st.global.u32 [%rd1], %r2;
    ret;
}
"#;
        let module =
            compile_ptx_to_msl_module_text(ptx).expect("local memory PTX should lower to MSL");
        assert!(module.msl.contains("uint slot[1];"));
        assert!(module.msl.contains("reinterpret_cast<thread uchar*>(slot)"));
        assert!(module.msl.contains("reinterpret_cast<thread uint*>"));
    }

    #[test]
    fn lowers_integer_atomics_to_msl_atomics() {
        let ptx = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry atomic_ops(
    .param .u64 out
)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<5>;
    .shared .align 4 .b32 scratch[1];

    ld.param.u64 %rd1, [out];
    mov.u32 %r1, 1;
    mov.u32 %r2, 2;
    atom.global.add.u32 %r3, [%rd1], %r1;
    atom.shared.add.u32 %r4, [scratch], %r1;
    atom.global.cas.b32 %r1, [%rd1], %r1, %r2;
    ret;
}
"#;
        let module =
            compile_ptx_to_msl_module_text(ptx).expect("integer atomics should lower to MSL");
        assert!(module.msl.contains("atomic_fetch_add_explicit"));
        assert!(module.msl.contains("reinterpret_cast<device atomic_uint*>"));
        assert!(module
            .msl
            .contains("reinterpret_cast<threadgroup atomic_uint*>"));
        assert!(module.msl.contains("atomic_compare_exchange_weak_explicit"));
    }

    #[test]
    fn lowers_scalar_predicate_and_bit_ops_to_msl() {
        let ptx = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry scalar_ops(
    .param .u64 out
)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<18>;
    .reg .pred %p<4>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [out];
    mov.u32 %r1, 3;
    mov.u32 %r2, 5;
    setp.lt.u32 %p1, %r1, %r2;
    setp.gt.and.u32 %p2, %r2, %r1, %p1;
    set.lt.u32.u32 %r3, %r1, %r2;
    set.lt.and.u32.u32 %r4, %r1, %r2, %p1;
    clz.b32 %r5, %r1;
    popc.b32 %r6, %r2;
    brev.b32 %r7, %r1;
    bfe.u32 %r8, %r7, 8, 8;
    bfi.b32 %r9, %r8, %r6, 4, 8;
    mul24.lo.u32 %r10, %r1, %r2;
    mul24.hi.u32 %r11, %r1, %r2;
    shf.l.wrap.b32 %r12, %r9, %r10, 5;
    activemask.b32 %r13;
    nanosleep.u32 1;
    mov.f32 %f1, 0f3f800000;
    mov.f32 %f2, 0fc0000000;
    copysign.f32 %f3, %f1, %f2;
    tanh.approx.f32 %f1, %f3;
    st.global.u32 [%rd1], %r12;
    ret;
}
"#;
        let module =
            compile_ptx_to_msl_module_text(ptx).expect("scalar PTX ops should lower to MSL");
        assert!(module.msl.contains("p2 = (((r2) > (r1)) && (p1));"));
        assert!(module
            .msl
            .contains("r3 = ((r1) < (r2)) ? uint(-1) : uint(0);"));
        assert!(module.msl.contains("r5 = clz(r1);"));
        assert!(module.msl.contains("r6 = popcount(r2);"));
        assert!(module.msl.contains("r7 = reverse_bits(r1);"));
        assert!(module
            .msl
            .contains("r8 = zluda_bfe_u32(r7, uint(8), uint(8));"));
        assert!(module
            .msl
            .contains("r9 = zluda_bfi_b32(r8, r6, uint(4), uint(8));"));
        assert!(module.msl.contains("r10 = uint((ulong((r1) & 0x00ffffffu)"));
        assert!(module.msl.contains("shf_shifted"));
        assert!(module.msl.contains("r13 = ~uint(0);"));
        assert!(module.msl.contains("f3 = copysign(f1, f2);"));
        assert!(module.msl.contains("f1 = tanh(f3);"));
    }

    #[test]
    fn lowers_cp_async_to_threadgroup_copy() {
        let ptx = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry cp_async_kernel(
    .param .u64 out,
    .param .u64 input
)
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<2>;
    .shared .align 16 .b8 scratch[16];

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [input];
    cp.async.ca.shared.global [scratch], [%rd2], 16;
    cp.async.commit_group;
    cp.async.wait_all;
    ld.shared.u32 %r1, [scratch];
    st.global.u32 [%rd1], %r1;
    ret;
}
"#;
        let module = compile_ptx_to_msl_module_text(ptx).expect("cp.async PTX should lower to MSL");
        assert!(module.msl.contains("threadgroup uchar scratch[16];"));
        assert!(module
            .msl
            .contains("reinterpret_cast<threadgroup uchar*>(scratch)[0]"));
        assert!(module.msl.contains("rd2[0]"));
        assert!(module
            .msl
            .contains("PTX async/wait hint lowered to no-op on Metal"));
        assert!(module.kernels[0].params[1].is_pointer);
    }

    #[test]
    #[ignore = "requires Apple Metal Toolchain; run with --ignored on macOS/Xcode hosts"]
    fn metal_compiler_accepts_generated_msl() {
        let ptx = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry scalar_ops(
    .param .u64 out
)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<18>;
    .reg .pred %p<4>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [out];
    mov.u32 %r1, 3;
    mov.u32 %r2, 5;
    setp.lt.u32 %p1, %r1, %r2;
    setp.gt.and.u32 %p2, %r2, %r1, %p1;
    set.lt.u32.u32 %r3, %r1, %r2;
    set.lt.and.u32.u32 %r4, %r1, %r2, %p1;
    clz.b32 %r5, %r1;
    popc.b32 %r6, %r2;
    brev.b32 %r7, %r1;
    bfe.u32 %r8, %r7, 8, 8;
    bfi.b32 %r9, %r8, %r6, 4, 8;
    mul24.lo.u32 %r10, %r1, %r2;
    mul24.hi.u32 %r11, %r1, %r2;
    shf.l.wrap.b32 %r12, %r9, %r10, 5;
    activemask.b32 %r13;
    nanosleep.u32 1;
    mov.f32 %f1, 0f3f800000;
    mov.f32 %f2, 0fc0000000;
    copysign.f32 %f3, %f1, %f2;
    tanh.approx.f32 %f1, %f3;
    st.global.u32 [%rd1], %r12;
    ret;
}

.visible .entry cp_async_kernel(
    .param .u64 out,
    .param .u64 input
)
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<2>;
    .shared .align 16 .b8 scratch[16];

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [input];
    cp.async.ca.shared.global [scratch], [%rd2], 16;
    cp.async.commit_group;
    cp.async.wait_all;
    ld.shared.u32 %r1, [scratch];
    st.global.u32 [%rd1], %r1;
    ret;
}
"#;
        let module = compile_ptx_to_msl_module_text(ptx).expect("PTX fixture should lower to MSL");
        compile_msl_with_metal("generated", &module.msl)
            .expect("generated MSL should compile with Apple Metal");
    }

    #[test]
    fn comgr_action_emits_msl_data() {
        let ptx = include_str!("../../../ptx/src/test/vectorAdd_kernel64.ptx");
        let mut input = apple_comgr_data_t::default();
        apple_comgr_create_data(
            apple_comgr_data_kind_s::APPLE_COMGR_DATA_KIND_PTX,
            &mut input,
        )
        .unwrap();
        apple_comgr_data_set_bytes(input, ptx.as_ptr().cast(), ptx.len()).unwrap();

        let mut input_set = apple_comgr_data_set_t::default();
        let mut output_set = apple_comgr_data_set_t::default();
        apple_comgr_create_data_set(&mut input_set).unwrap();
        apple_comgr_create_data_set(&mut output_set).unwrap();
        apple_comgr_data_set_add(input_set, input).unwrap();

        let mut info = apple_comgr_action_info_t::default();
        apple_comgr_create_action_info(&mut info).unwrap();
        apple_comgr_do_action(
            apple_comgr_action_kind_s::APPLE_COMGR_ACTION_COMPILE_PTX_TO_MSL,
            info,
            input_set,
            output_set,
        )
        .unwrap();

        let mut count = 0;
        apple_comgr_get_data_count(output_set, &mut count).unwrap();
        assert_eq!(count, 1);
        let mut output = apple_comgr_data_t::default();
        apple_comgr_get_data(output_set, 0, &mut output).unwrap();
        let mut size = 0;
        apple_comgr_data_get_bytes(output, ptr::null_mut(), &mut size).unwrap();
        assert!(size > 0);
    }
}
