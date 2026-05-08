//! Build script: locate XRT via pkg-config (falling back to /opt/xilinx/xrt),
//! then run bindgen against wrapper.h.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=XRT_PATH");

    // 1. Locate XRT headers/libs.
    let include_dirs: Vec<String> = match pkg_config::Config::new()
        .atleast_version("2.15")
        .probe("xrt_coreutil")
    {
        Ok(lib) => lib
            .include_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect(),
        Err(_) => {
            // Fallback: default install location.
            let xrt_root = env::var("XRT_PATH").unwrap_or_else(|_| "/opt/xilinx/xrt".to_string());
            println!("cargo:rustc-link-search=native={}/lib", xrt_root);
            println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
            vec![format!("{}/include", xrt_root)]
        }
    };

    // 2. Generate bindings.
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_function("xrt.*")
        .allowlist_type("xrt.*")
        .allowlist_var("XRT_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    for inc in &include_dirs {
        builder = builder.clang_arg(format!("-I{}", inc));
    }

    let bindings = builder.generate().expect("bindgen failed for XRT");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("failed to write bindings.rs");
}
