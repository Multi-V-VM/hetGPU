use std::env;
use std::path::PathBuf;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if !matches!(target_os.as_str(), "macos" | "ios") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let zluda_src = manifest_dir.join("../../zluda/src");
    let metal_runtime = zluda_src.join("apple_metal_runtime.m");
    let ane_bridge = zluda_src.join("ane_bridge.m");
    let ane_header = zluda_src.join("ane_bridge.h");

    println!("cargo:rerun-if-changed={}", metal_runtime.display());
    println!("cargo:rerun-if-changed={}", ane_bridge.display());
    println!("cargo:rerun-if-changed={}", ane_header.display());

    cc::Build::new()
        .file(&metal_runtime)
        .file(&ane_bridge)
        .include(&zluda_src)
        .flag("-fobjc-arc")
        .compile("hetgpu_apple_runtime");

    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=IOSurface");
}
