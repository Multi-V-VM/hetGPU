//! Integration test: compile a trivial hand-written TOSA module to XCLBIN.
//! Requires mlir-aie on PATH. Gated as #[ignore] so CI can skip it.

use aie_comgr_sys::{compile_tosa_to_xclbin, AieCompileConfig};

#[test]
#[ignore = "requires mlir-aie toolchain on PATH"]
fn trivial_tosa_compiles() {
    let mlir = r#"
func.func @kernel(%arg0: tensor<1x4xi32>) -> tensor<1x4xi32> {
  return %arg0 : tensor<1x4xi32>
}
"#;
    let config = AieCompileConfig::strix();
    let xclbin = compile_tosa_to_xclbin(mlir, &config).expect("compilation failed");
    assert!(xclbin.len() > 64, "XCLBIN should be nontrivial");
    assert_eq!(&xclbin[0..7], b"xclbin2");
}
