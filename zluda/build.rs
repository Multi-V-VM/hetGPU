fn main() {
    let apple_backend = std::env::var("CARGO_FEATURE_ANE").is_ok();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if apple_backend && matches!(target_os.as_str(), "macos" | "ios") {
        println!("cargo:rerun-if-changed=src/apple_metal_runtime.m");
        println!("cargo:rerun-if-changed=src/apple_cuda_stub.m");
        println!("cargo:rerun-if-changed=src/ane_bridge.h");
        println!("cargo:rerun-if-changed=src/ane_bridge.m");
        let mut metal_build = cc::Build::new();
        metal_build.file("src/apple_metal_runtime.m");
        metal_build.file("src/ane_bridge.m");
        metal_build.flag("-fobjc-arc");
        metal_build.compile("hetgpu_apple_metal_runtime");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=IOSurface");
    }

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

        // Set SONAME so the real libcudart.so.12 (which uses dlopen("libcuda.so.1"))
        // finds our library when it's already loaded in the process.
        println!("cargo:rustc-link-arg=-Wl,-soname,libcuda.so.1");

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

        // Also build standalone shared libraries (libcudart.so.12, libcublas.so.12,
        // libcublasLt.so.12) so they can replace the real system libraries.
        // Without these, PyTorch resolves cudart symbols from the real libcudart.so.12
        // which internally computes occupancy using stale/invalid state, causing SIGFPE.
        build_standalone_shim_libraries(tools_dir, enable_logs);
    }
}

/// Build standalone .so shim libraries that can be deployed alongside libcuda.so.1.
/// When LD_LIBRARY_PATH includes the output directory, PyTorch finds these shims
/// instead of the real CUDA libraries.
fn build_standalone_shim_libraries(tools_dir: &std::path::Path, enable_logs: bool) {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = std::path::Path::new(&out_dir);
    // Place standalone shims next to the final library output.
    // Cargo builds into target/{profile}/deps or target/{profile}/build/...
    // We put them in OUT_DIR and also copy to the target directory.
    let target_dir = out_path
        .ancestors()
        .find(|p| p.file_name().map_or(false, |n| n == "debug" || n == "release"))
        .map(|p| p.to_path_buf());

    // Only build standalone libcudart.so.12. The cublas/cublasLt shims are
    // intentionally NOT built as standalone libraries — the real cublas libraries
    // are functional and our shims lack the 77+ internal symbols that
    // libcublas.so.12 imports from libcublasLt.so.12.
    let shims: &[(&str, &str, &str, &str)] = &[
        (
            "src/cudart_shim.c",
            "tools/cudart_shim/cudart_shim.map",
            "libcudart.so.12",
            "cudart",
        ),
    ];

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    for (src, map, soname, label) in shims {
        let src_path = std::path::Path::new(&manifest_dir).join(src);
        let map_path = tools_dir.join(map);

        if !src_path.exists() {
            eprintln!("Warning: shim source {} not found, skipping standalone {}", src, soname);
            continue;
        }

        let obj_path = out_path.join(format!("{}_standalone.o", label));
        let so_path = out_path.join(soname);

        // Compile to object file
        let mut compile = std::process::Command::new("cc");
        compile
            .arg("-fPIC")
            .arg("-shared")
            .arg("-Wno-unused-parameter")
            .arg("-o")
            .arg(&so_path)
            .arg(&src_path);

        if enable_logs {
            compile.arg("-DHETGPU_DEBUG_LOGS");
        }

        // Add version script if available
        if map_path.exists() {
            compile.arg(format!("-Wl,--version-script={}", map_path.display()));
        }

        // Set SONAME so the dynamic linker treats this as the real library
        compile.arg(format!("-Wl,-soname,{}", soname));

        // The standalone cudart shim uses extern cu* driver API functions.
        // Create a symlink to libcuda.so.1 so we can link against it and
        // generate a DT_NEEDED entry. At runtime, LD_PRELOAD provides the
        // real libcuda.so.1 (which is our hetGPU driver).
        if let Some(ref tdir) = target_dir {
            let cuda_link = out_path.join("libcuda.so");
            let cuda_target = tdir.join("libcuda.so.1");
            if cuda_target.exists() {
                let _ = std::fs::remove_file(&cuda_link);
                let _ = std::os::unix::fs::symlink(&cuda_target, &cuda_link);
                compile.arg(format!("-L{}", out_path.display()));
                compile.arg("-lcuda");
            }
        }

        compile.arg("-Wl,--no-as-needed");

        let status = compile.status();
        match status {
            Ok(s) if s.success() => {
                eprintln!("Built standalone {}", soname);
                // Copy to target directory if found
                if let Some(ref tdir) = target_dir {
                    let dest = tdir.join(soname);
                    // Remove existing file/symlink first to avoid
                    // following symlinks during copy
                    let _ = std::fs::remove_file(&dest);
                    match std::fs::copy(&so_path, &dest) {
                        Ok(_) => eprintln!("  -> copied to {}", dest.display()),
                        Err(e) => eprintln!("  -> copy to {} failed: {}", dest.display(), e),
                    }
                }
            }
            Ok(s) => {
                eprintln!("Warning: building standalone {} failed (exit {})", soname, s);
            }
            Err(e) => {
                eprintln!("Warning: could not run cc to build standalone {}: {}", soname, e);
            }
        }
    }
}
