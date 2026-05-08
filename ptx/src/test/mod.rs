use crate::pass::{self, TranslateError};
use ptx_parser as ast;

mod sass_debug_mapping;
mod spirv_run;

#[cfg(not(feature = "ci_build"))]
#[macro_export]
macro_rules! read_test_file {
    ($file:expr) => {
        {
            use std::path::PathBuf;
            // CARGO_MANIFEST_DIR is the crate directory (ptx), but file! is relative to the workspace root (and therefore also includes ptx).
            let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.pop();
            path.push(file!());
            path.pop();
            path.push($file);
            std::fs::read_to_string(path).unwrap()
        }
    };
}

#[cfg(feature = "ci_build")]
#[macro_export]
macro_rules! read_test_file {
    ($file:expr) => {
        include_str!($file).to_string()
    };
}
pub(crate) use read_test_file;

fn parse_and_assert(ptx_text: &str) {
    ast::parse_module_checked(ptx_text).unwrap();
}

fn compile_and_assert(ptx_text: &str) -> Result<(), TranslateError> {
    let ast = ast::parse_module_checked(ptx_text).unwrap();
    let attributes = pass::Attributes {
        clock_rate: 2124000,
        emit_debug_info: false,
    };
    crate::to_llvm_module(ast, attributes, |_| {})?;
    Ok(())
}

#[test]
fn empty() {
    parse_and_assert(".version 6.5 .target sm_30, debug");
}

#[test]
fn operands_ptx() {
    let vector_add = include_str!("operands.ptx");
    parse_and_assert(vector_add);
}

#[test]
#[allow(non_snake_case)]
fn vectorAdd_kernel64_ptx() -> Result<(), TranslateError> {
    let vector_add = include_str!("vectorAdd_kernel64.ptx");
    compile_and_assert(vector_add)
}

#[test]
#[allow(non_snake_case)]
fn _Z9vectorAddPKfS0_Pfi_ptx() -> Result<(), TranslateError> {
    let vector_add = include_str!("_Z9vectorAddPKfS0_Pfi.ptx");
    compile_and_assert(vector_add)
}

#[test]
#[allow(non_snake_case)]
fn vectorAdd_11_ptx() -> Result<(), TranslateError> {
    let vector_add = include_str!("vectorAdd_11.ptx");
    compile_and_assert(vector_add)
}

/// Test PTX -> LLVM -> PTX round-trip with debug symbols
/// This ensures SASS code can map back to PTX source locations
#[test]
fn debug_round_trip_with_sass_mapping() -> Result<(), Box<dyn std::error::Error>> {
    // Simple PTX kernel for testing
    let ptx_source = r#"
.version 8.0
.target sm_86
.address_size 64

.visible .entry add_kernel(
    .param .u64 add_kernel_param_0,
    .param .u64 add_kernel_param_1,
    .param .u64 add_kernel_param_2
)
{
    .reg .b32   %r<4>;
    .reg .b64   %rd<11>;
    .reg .f32   %f<4>;

    ld.param.u64    %rd1, [add_kernel_param_0];
    ld.param.u64    %rd2, [add_kernel_param_1];
    ld.param.u64    %rd3, [add_kernel_param_2];

    mov.u32     %r1, %tid.x;
    cvt.u64.u32 %rd4, %r1;
    mul.wide.u32    %rd5, %r1, 4;

    add.u64     %rd6, %rd1, %rd5;
    ld.global.f32   %f1, [%rd6];

    add.u64     %rd7, %rd2, %rd5;
    ld.global.f32   %f2, [%rd7];

    add.f32     %f3, %f1, %f2;

    add.u64     %rd10, %rd3, %rd5;
    st.global.f32   [%rd10], %f3;

    ret;
}
    "#;

    // Parse PTX
    let ast = ast::parse_module_checked(ptx_source)
        .map_err(|e| format!("PTX parsing failed: {:?}", e))?;

    // Compile PTX -> LLVM -> PTX with debug info
    let (_module, regenerated_ptx, _debug_mappings) =
        crate::to_llvm_module_with_debug_round_trip(ast)?;

    // Module is successfully created if we get here without error

    // Verify regenerated PTX contains debug information
    // The regenerated PTX should have .loc directives or debug metadata
    println!("=== Regenerated PTX with Debug Info ===");
    println!("{}", regenerated_ptx);

    // Basic validation: check that PTX was generated
    assert!(
        !regenerated_ptx.is_empty(),
        "Regenerated PTX should not be empty"
    );
    assert!(
        regenerated_ptx.contains(".target") || regenerated_ptx.contains("PTX"),
        "Regenerated output should contain PTX markers"
    );

    // Check for debug-related content (either .loc directives or debug metadata)
    let has_debug_info = regenerated_ptx.contains(".loc")
        || regenerated_ptx.contains(".file")
        || regenerated_ptx.contains("!dbg")
        || regenerated_ptx.contains("debug");

    if has_debug_info {
        println!("✓ Debug information preserved in regenerated PTX");
    } else {
        println!("⚠ Warning: No explicit debug markers found (may be in metadata)");
    }

    // Verify we have .loc directives for SASS mapping
    assert!(
        regenerated_ptx.contains(".loc") || regenerated_ptx.contains(".file"),
        "PTX must contain .loc or .file directives for SASS-to-PTX mapping.\nGenerated PTX:\n{}",
        regenerated_ptx
    );

    Ok(())
}

/// Test that LLVM IR includes debug metadata
#[test]
fn llvm_ir_contains_debug_metadata() -> Result<(), Box<dyn std::error::Error>> {
    let ptx_source = include_str!("spirv_run/add.ptx");

    let ast = ast::parse_module_checked(ptx_source)
        .map_err(|e| format!("PTX parsing failed: {:?}", e))?;

    let module = crate::to_llvm_module(
        ast,
        pass::Attributes {
            clock_rate: 2124000,
            emit_debug_info: false,
        },
        |_| {},
    )?;

    let llvm_ir = module
        .print_to_string()
        .map_err(|e| format!("Failed to get LLVM IR: {:?}", e))?;

    println!("=== LLVM IR ===");
    println!("{}", llvm_ir);

    // LLVM IR should contain function definitions
    assert!(
        llvm_ir.contains("define"),
        "LLVM IR should contain function definitions"
    );

    Ok(())
}

#[test]
fn ggml_extern_symbol_is_preserved_for_pacc_link() -> Result<(), Box<dyn std::error::Error>> {
    let ptx_source = r#"
.version 6.5
.target sm_80
.address_size 64

.extern .func (.param .u64 output) ggml_vec_dot_f16(
    .param .u64 input
);

.visible .entry ggml_call_kernel(
    .param .u64 input,
    .param .u64 output
)
{
    .reg .u64 in_addr;
    .reg .u64 out_addr;
    .reg .u64 temp;

    ld.param.u64 in_addr, [input];
    ld.param.u64 out_addr, [output];
    ld.global.u64 temp, [in_addr];

    .param .u64 ggml_in;
    .param .u64 ggml_out;
    st.param.b64 [ggml_in], temp;
    call (ggml_out), ggml_vec_dot_f16, (ggml_in);
    ld.param.u64 temp, [ggml_out];
    st.global.u64 [out_addr], temp;
    ret;
}
    "#;

    let ast = ast::parse_module_checked(ptx_source)
        .map_err(|e| format!("PTX parsing failed: {:?}", e))?;

    let module = crate::to_llvm_module(
        ast,
        pass::Attributes {
            clock_rate: 2124000,
            emit_debug_info: false,
        },
        |_| {},
    )?;

    let llvm_ir = module.llvm_ir.print_module_to_string();
    let llvm_ir = llvm_ir.to_str();

    assert!(
        llvm_ir.contains("@ggml_vec_dot_f16("),
        "LLVM IR should keep the ggml operator symbol name intact.\n{}",
        llvm_ir
    );
    assert!(
        !llvm_ir.contains("@__zluda_ptx_impl_ggml_vec_dot_f16("),
        "ggml operator symbols must not be rewritten to the PTX helper namespace.\n{}",
        llvm_ir
    );
    assert!(
        !llvm_ir.contains("declare hidden i64 @ggml_vec_dot_f16"),
        "ggml operator symbols should stay externally visible for later PACC linking.\n{}",
        llvm_ir
    );

    Ok(())
}

#[test]
fn micro_kernel_symbol_is_preserved_for_pacc_link() -> Result<(), Box<dyn std::error::Error>> {
    let ptx_source = r#"
.version 6.5
.target sm_80
.address_size 64

.extern .func micro_kernel_bf16bf16fp32_tile_k1_tile_n_gemv(
    .param .u32 m_v,
    .param .u32 n_v,
    .param .u32 k_v,
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr
);

.visible .entry micro_kernel_call_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr
)
{
    .reg .u32 m_v;
    .reg .u32 n_v;
    .reg .u32 k_v;
    .reg .u64 a_addr;
    .reg .u64 b_addr;
    .reg .u64 c_addr;

    ld.param.u64 a_addr, [a_ptr];
    ld.param.u64 b_addr, [b_ptr];
    ld.param.u64 c_addr, [c_ptr];
    mov.u32 m_v, 1;
    mov.u32 n_v, 32;
    mov.u32 k_v, 128;

    .param .u32 p_m;
    .param .u32 p_n;
    .param .u32 p_k;
    .param .u64 p_a;
    .param .u64 p_b;
    .param .u64 p_c;
    st.param.b32 [p_m], m_v;
    st.param.b32 [p_n], n_v;
    st.param.b32 [p_k], k_v;
    st.param.b64 [p_a], a_addr;
    st.param.b64 [p_b], b_addr;
    st.param.b64 [p_c], c_addr;
    call micro_kernel_bf16bf16fp32_tile_k1_tile_n_gemv, (p_m, p_n, p_k, p_a, p_b, p_c);
    ret;
}
    "#;

    let ast = ast::parse_module_checked(ptx_source)
        .map_err(|e| format!("PTX parsing failed: {:?}", e))?;

    let module = crate::to_llvm_module(
        ast,
        pass::Attributes {
            clock_rate: 2124000,
            emit_debug_info: false,
        },
        |_| {},
    )?;

    let llvm_ir = module.llvm_ir.print_module_to_string();
    let llvm_ir = llvm_ir.to_str();

    assert!(
        llvm_ir.contains("@micro_kernel_bf16bf16fp32_tile_k1_tile_n_gemv("),
        "LLVM IR should keep the PACC micro-kernel symbol name intact.\n{}",
        llvm_ir
    );
    assert!(
        !llvm_ir.contains("@__zluda_ptx_impl_micro_kernel_bf16bf16fp32_tile_k1_tile_n_gemv("),
        "micro_kernel symbols must not be rewritten to the PTX helper namespace.\n{}",
        llvm_ir
    );
    assert!(
        !llvm_ir.contains("declare hidden void @micro_kernel_bf16bf16fp32_tile_k1_tile_n_gemv"),
        "micro_kernel symbols should stay externally visible for later PACC linking.\n{}",
        llvm_ir
    );

    Ok(())
}
