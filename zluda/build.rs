fn main() {
    // Only build and link embedded shims when the feature is enabled.
    // We ship cudart/cublas/cublasLt shims so PyTorch can start even if the
    // system libraries are absent (e.g. missing cublasGetMathMode).
    let embed = std::env::var("CARGO_FEATURE_EMBED_CUDART").is_ok();
    if embed {
        println!("cargo:rerun-if-changed=src/cudart_shim.c");
        println!("cargo:rerun-if-changed=src/cublas_shim.c");
        println!("cargo:rerun-if-changed=src/cublaslt_shim.c");

        let cargo_manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let tools_dir = std::path::Path::new(&cargo_manifest_dir).parent().unwrap();

        println!(
            "cargo:rerun-if-changed={}",
            tools_dir
                .join("tools/cudart_shim/cudart_shim.map")
                .display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            tools_dir
                .join("tools/cublas_shim/cublas_shim.map")
                .display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            tools_dir
                .join("tools/cublaslt_shim/cublaslt_shim.map")
                .display()
        );

        // Add version scripts for all shims
        let version_scripts = [
            tools_dir.join("tools/cudart_shim/cudart_shim.map"),
            tools_dir.join("tools/cublas_shim/cublas_shim.map"),
            tools_dir.join("tools/cublaslt_shim/cublaslt_shim.map"),
        ];

        for version_script in &version_scripts {
            if version_script.exists() {
                println!(
                    "cargo:rustc-link-arg=-Wl,--version-script={}",
                    version_script.display()
                );
            } else {
                eprintln!(
                    "Warning: Version script not found at {}",
                    version_script.display()
                );
            }
        }

        // Force the linker to keep all symbols from the shim archives
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg=-lcudart_shim");
        println!("cargo:rustc-link-arg=-lcublas_shim");
        println!("cargo:rustc-link-arg=-lcublaslt_shim");
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

        let enable_logs = std::env::var("HETGPU_DEBUG_LOGS")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on"))
            .unwrap_or(false);

        // Build cudart_shim
        let mut cudart_build = cc::Build::new();
        cudart_build.file("src/cudart_shim.c");
        cudart_build.flag("-fPIC");
        cudart_build.flag("-Wno-unused-parameter");
        if enable_logs {
            cudart_build.define("HETGPU_DEBUG_LOGS", None);
        }
        cudart_build.compile("cudart_shim");

        // Build cublas_shim
        let mut cublas_build = cc::Build::new();
        cublas_build.file("src/cublas_shim.c");
        cublas_build.flag("-fPIC");
        cublas_build.flag("-Wno-unused-parameter");
        if enable_logs {
            cublas_build.define("HETGPU_DEBUG_LOGS", None);
        }
        cublas_build.compile("cublas_shim");

        // Build cublaslt_shim
        let mut cublaslt_build = cc::Build::new();
        cublaslt_build.file("src/cublaslt_shim.c");
        cublaslt_build.flag("-fPIC");
        cublaslt_build.flag("-Wno-unused-parameter");
        if enable_logs {
            cublaslt_build.define("HETGPU_DEBUG_LOGS", None);
        }
        cublaslt_build.compile("cublaslt_shim");
    }
}
