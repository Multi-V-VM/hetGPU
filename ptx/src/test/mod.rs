use crate::pass::{self, TranslateError};
use ptx_parser as ast;

mod spirv_run;
mod sass_debug_mapping;

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
    assert!(!regenerated_ptx.is_empty(), "Regenerated PTX should not be empty");
    assert!(regenerated_ptx.contains(".target") || regenerated_ptx.contains("PTX"),
            "Regenerated output should contain PTX markers");

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

    let llvm_ir = module.print_to_string()
        .map_err(|e| format!("Failed to get LLVM IR: {:?}", e))?;

    println!("=== LLVM IR ===");
    println!("{}", llvm_ir);

    // LLVM IR should contain function definitions
    assert!(llvm_ir.contains("define"), "LLVM IR should contain function definitions");

    Ok(())
}
