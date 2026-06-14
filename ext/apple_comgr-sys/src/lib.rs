#![allow(non_camel_case_types)]

use ptx_parser::{
    CvtMode, Directive, Function, ImmediateValue, Instruction, Module, MulDetails, MulIntControl,
    MultiVariable, ParsedOperand, PredAt, RegOrImmediate, RightShiftKind, ScalarType,
    SetpCompareFloat, SetpCompareInt, SetpCompareOp, StateSpace, Statement, Type,
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

pub fn compile_ptx_to_msl_module_text(ptx: &str) -> Result<AppleCompiledModule, AppleComgrCompileError> {
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
    Cvta(String),
    Add(String, String),
    Sub(String, String),
}

struct KernelLowering<'module, 'input> {
    kernel: &'module Function<'input, &'input str, PtxStatement<'input>>,
    params: Vec<KernelParam>,
    var_types: HashMap<String, Type>,
    defs: HashMap<String, Def>,
    pointer_regs: HashSet<String>,
    pointer_params: HashSet<String>,
    diagnostics: Vec<String>,
    indent: usize,
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
            defs: HashMap::new(),
            pointer_regs: HashSet::new(),
            pointer_params: HashSet::new(),
            diagnostics: Vec::new(),
            indent: 1,
        };
        if let Some(body) = &kernel.body {
            lowering.collect_declarations(body);
            lowering.collect_defs_and_memory_uses(body);
            lowering.infer_pointers();
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
        for statement in body {
            self.emit_statement(statement, &mut out);
        }
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
                    size: if param.is_pointer { 8 } else { type_size(&param.ty) },
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
                            self.var_types.insert(raw_name, info.v_type.clone());
                        }
                    }
                    MultiVariable::Names { info, names } => {
                        for name in names {
                            self.var_types
                                .insert((*name).to_string(), info.v_type.clone());
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
                if data.state_space == StateSpace::Param {
                    if let (Some(dst), Some(param)) =
                        (operand_reg(&arguments.dst), operand_reg(&arguments.src))
                    {
                        self.defs.insert(dst, Def::LdParam(param));
                    }
                } else if matches!(
                    data.state_space,
                    StateSpace::Global | StateSpace::Generic | StateSpace::Const
                ) {
                    if let Some(addr) = operand_reg(&arguments.src) {
                        self.trace_pointer_from(addr);
                    }
                }
            }
            Instruction::St { data, arguments } => {
                if data.state_space != StateSpace::Param {
                    if let Some(addr) = operand_reg(&arguments.src1) {
                        self.trace_pointer_from(addr);
                    }
                }
            }
            Instruction::Mov { arguments, .. } => {
                if let (Some(dst), Some(src)) =
                    (operand_reg(&arguments.dst), operand_reg(&arguments.src))
                {
                    self.defs.insert(dst, Def::Mov(src));
                }
            }
            Instruction::Cvta { arguments, .. } => {
                if let (Some(dst), Some(src)) =
                    (operand_reg(&arguments.dst), operand_reg(&arguments.src))
                {
                    self.defs.insert(dst, Def::Cvta(src));
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

    fn trace_pointer_from(&mut self, reg: String) {
        let mut seen = HashSet::new();
        self.trace_pointer_reg(&reg, &mut seen);
    }

    fn trace_pointer_reg(&mut self, reg: &str, seen: &mut HashSet<String>) {
        if !seen.insert(reg.to_string()) {
            return;
        }
        self.pointer_regs.insert(reg.to_string());
        let Some(def) = self.defs.get(reg).cloned() else {
            return;
        };
        match def {
            Def::LdParam(param) => {
                self.pointer_params.insert(param);
            }
            Def::Mov(src) | Def::Cvta(src) => self.trace_pointer_reg(&src, seen),
            Def::Add(src1, src2) | Def::Sub(src1, src2) => {
                let src1_reaches = self.reaches_param(&src1, &mut HashSet::new());
                let src2_reaches = self.reaches_param(&src2, &mut HashSet::new());
                match (src1_reaches, src2_reaches) {
                    (true, false) => self.trace_pointer_reg(&src1, seen),
                    (false, true) => self.trace_pointer_reg(&src2, seen),
                    (true, true) => {
                        if self.prefers_pointer_base(&src1, &src2) {
                            self.trace_pointer_reg(&src1, seen);
                        } else {
                            self.trace_pointer_reg(&src2, seen);
                        }
                    }
                    (false, false) => self.trace_pointer_reg(&src1, seen),
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
            Some(Def::Mov(src)) | Some(Def::Cvta(src)) => self.reaches_param(src, seen),
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
                    Def::Mov(src) | Def::Cvta(src) => self.pointer_regs.contains(src),
                    Def::Add(src1, src2) | Def::Sub(src1, src2) => {
                        self.pointer_regs.contains(src1) || self.pointer_regs.contains(src2)
                    }
                };
                if becomes_pointer && self.pointer_regs.insert(dst.clone()) {
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
                self.line(out, &format!("device uchar* {local} = nullptr;"));
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
                out.push_str(&format!("{}:\n", sanitize_label(label)));
            }
            Statement::Instruction(pred, instruction) => {
                self.emit_instruction(pred, instruction, out)
            }
            Statement::Block(inner) => {
                for statement in inner {
                    self.emit_statement(statement, out);
                }
            }
            Statement::Variable(_) => {}
        }
    }

    fn emit_instruction(
        &mut self,
        pred: &Option<PredAt<&'input str>>,
        instruction: &Instruction<PtxOperand<'input>>,
        out: &mut String,
    ) {
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
                let src = self.expr(&arguments.src);
                if data.state_space == StateSpace::Param {
                    self.line(out, &format!("{dst} = {src};"));
                } else {
                    self.line(
                        out,
                        &format!(
                            "{dst} = *reinterpret_cast<device {}*>({src});",
                            msl_type(&data.typ)
                        ),
                    );
                }
            }
            Instruction::St { data, arguments } => {
                let dst = self.expr(&arguments.src1);
                let src = self.expr(&arguments.src2);
                self.line(
                    out,
                    &format!(
                        "*reinterpret_cast<device {}*>({dst}) = {src};",
                        msl_type(&data.typ)
                    ),
                );
            }
            Instruction::Mov { arguments, .. } => {
                let dst = self.expr(&arguments.dst);
                let src = self.expr(&arguments.src);
                self.line(out, &format!("{dst} = {src};"));
            }
            Instruction::Cvta { arguments, .. } => {
                let dst = self.expr(&arguments.dst);
                let src = self.expr(&arguments.src);
                self.line(out, &format!("{dst} = {src};"));
            }
            Instruction::Cvt { data, arguments } => {
                let dst = self.expr(&arguments.dst);
                let src = self.expr(&arguments.src);
                let to = scalar_msl_type(data.to);
                let expr = match data.mode {
                    CvtMode::Bitcast if data.to.size_of() == data.from.size_of() => {
                        format!("as_type<{to}>({src})")
                    }
                    _ => format!("{to}({src})"),
                };
                self.line(out, &format!("{dst} = {expr};"));
            }
            Instruction::Add { arguments, .. } => {
                self.emit_binary(out, &arguments.dst, &arguments.src1, &arguments.src2, "+");
            }
            Instruction::Sub { arguments, .. } => {
                self.emit_binary(out, &arguments.dst, &arguments.src1, &arguments.src2, "-");
            }
            Instruction::Mul { data, arguments } => {
                let dst = self.expr(&arguments.dst);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                let expr = match data {
                    MulDetails::Integer {
                        control: MulIntControl::High,
                        ..
                    } => format!("(({src1}) * ({src2})) >> 32"),
                    _ => format!("({src1}) * ({src2})"),
                };
                self.line(out, &format!("{dst} = {expr};"));
            }
            Instruction::Mad { data, arguments } => {
                let dst = self.expr(&arguments.dst);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                let src3 = self.expr(&arguments.src3);
                let expr = match data {
                    ptx_parser::MadDetails::Integer {
                        control: MulIntControl::High,
                        ..
                    } => format!("((({src1}) * ({src2})) >> 32) + ({src3})"),
                    _ => format!("(({src1}) * ({src2})) + ({src3})"),
                };
                self.line(out, &format!("{dst} = {expr};"));
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
            Instruction::Selp { arguments, .. } => {
                let dst = self.expr(&arguments.dst);
                let src1 = self.expr(&arguments.src1);
                let src2 = self.expr(&arguments.src2);
                let pred = self.expr(&arguments.src3);
                self.line(out, &format!("{dst} = {pred} ? {src1} : {src2};"));
            }
            Instruction::Bra { arguments } => {
                self.line(out, &format!("goto {};", sanitize_label(arguments.src)));
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

    fn emit_binary(
        &self,
        out: &mut String,
        dst: &PtxOperand<'input>,
        src1: &PtxOperand<'input>,
        src2: &PtxOperand<'input>,
        op: &str,
    ) {
        let dst_expr = self.expr(dst);
        let src1_expr = self.expr(src1);
        let src2_expr = self.expr(src2);
        self.line(
            out,
            &format!("{dst_expr} = ({src1_expr}) {op} ({src2_expr});"),
        );
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

    #[test]
    fn compiles_vector_add_to_msl() {
        let ptx = include_str!("../../../ptx/src/test/vectorAdd_kernel64.ptx");
        let module = compile_ptx_to_msl_module_text(ptx).expect("vector add PTX should lower to MSL");
        assert!(module.msl.contains("kernel void VecAdd_kernel"));
        assert!(module.msl.contains("device uchar* VecAdd_kernel_param_0"));
        assert!(module.msl.contains("reinterpret_cast<device float*>"));
        assert!(module.msl.contains("goto BB0_2"));
        assert_eq!(module.kernels.len(), 1);
        assert_eq!(module.kernels[0].name, "VecAdd_kernel");
        assert_eq!(module.kernels[0].params.len(), 4);
        assert!(module.kernels[0].params[0].is_pointer);
        assert_eq!(module.kernels[0].params[3].size, 4);
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
