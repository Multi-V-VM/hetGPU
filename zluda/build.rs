fn main() {
    // Only build and link embedded shims when the feature is enabled.
    // We ship cudart/cublas/cublasLt shims so PyTorch can start even if the
    // system libraries are absent (e.g. missing cublasGetMathMode).
    let embed = std::env::var("CARGO_FEATURE_EMBED_CUDART").is_ok();
    if embed {
        println!("cargo:rerun-if-changed=src/cudart_shim.c");
        println!("cargo:rerun-if-changed=src/cublas_shim.c");
        println!("cargo:rerun-if-changed=src/cublaslt_shim.c");
        println!("cargo:rerun-if-changed=src/cusparse_shim.c");
        println!("cargo:rerun-if-changed=src/cufft_shim.c");
        println!("cargo:rerun-if-changed=src/nccl_shim.c");
        println!("cargo:rerun-if-changed=src/torch_abi_shim.c");
        println!("cargo:rerun-if-changed=src/pacc_disabled_stubs.c");

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

        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
        let static_cudart = target_os == "emscripten"
            || target_arch == "wasm32"
            || target_env == "emscripten"
            || std::env::var("HETGPU_EMBED_CUDART_STATIC").is_ok();
        let enable_logs = std::env::var("HETGPU_DEBUG_LOGS")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on"))
            .unwrap_or(false);

        if static_cudart {
            let mut build = cc::Build::new();
            build
                .file("src/cudart_shim.c")
                .file("src/cublas_shim.c")
                .file("src/cublaslt_shim.c")
                .file("src/cusparse_shim.c")
                .file("src/cufft_shim.c")
                .file("src/nccl_shim.c")
                .file("src/torch_abi_shim.c")
                .file("src/pacc_disabled_stubs.c")
                .flag_if_supported("-Wno-unused-parameter")
                .define("_GNU_SOURCE", None)
                .define("_GLIBCXX_USE_CXX11_ABI", Some("0"))
                .define("HETGPU_STATIC_CUDART", None);
            if target_os == "emscripten" || target_arch == "wasm32" {
                build.define("HETGPU_WASM", None);
            }
            if enable_logs {
                build.define("HETGPU_DEBUG_LOGS", None);
            }
            build.compile("hetgpu_cuda_shim");
            return;
        }

        let out_dir = std::env::var("OUT_DIR").unwrap();
        let profile_dir = std::path::Path::new(&out_dir)
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("OUT_DIR should be under target/<profile>/build/<pkg>/out");

        let is_macos = target_os == "macos";

        let shim_so = profile_dir.join(if is_macos {
            "libhetgpu_cuda_shim.dylib"
        } else {
            "libhetgpu_cuda_shim.so"
        });
        let compiler = cc::Build::new().get_compiler();
        let mut shim_build = compiler.to_command();
        if is_macos {
            shim_build.arg("-dynamiclib");
            shim_build.arg("-undefined");
            shim_build.arg("dynamic_lookup");
        } else {
            shim_build.arg("-shared");
        }
        shim_build.arg("-fPIC");
        shim_build.arg("-Wno-unused-parameter");
        shim_build.arg("-D_GLIBCXX_USE_CXX11_ABI=0");
        if enable_logs {
            shim_build.arg("-DHETGPU_DEBUG_LOGS");
        }
        shim_build.arg("-o");
        shim_build.arg(&shim_so);
        shim_build.arg("src/cudart_shim.c");
        shim_build.arg("src/cublas_shim.c");
        shim_build.arg("src/cublaslt_shim.c");
        shim_build.arg("src/cusparse_shim.c");
        shim_build.arg("src/cufft_shim.c");
        shim_build.arg("src/nccl_shim.c");
        shim_build.arg("src/torch_abi_shim.c");
        if !is_macos {
            shim_build.arg("-Wl,-soname,libhetgpu_cuda_shim.so");
            shim_build.arg("-ldl");
        }
        shim_build.arg("-lz");
        let status = shim_build.status().unwrap();
        assert!(status.success(), "failed to build embedded CUDA shim");

        let nccl_so = profile_dir.join(if is_macos {
            "libnccl.2.dylib"
        } else {
            "libnccl.so.2"
        });
        let mut nccl_build = compiler.to_command();
        if is_macos {
            nccl_build.arg("-dynamiclib");
            nccl_build.arg("-undefined");
            nccl_build.arg("dynamic_lookup");
        } else {
            nccl_build.arg("-shared");
        }
        nccl_build.arg("-fPIC");
        nccl_build.arg("-Wno-unused-parameter");
        if enable_logs {
            nccl_build.arg("-DHETGPU_DEBUG_LOGS");
        }
        nccl_build.arg("-o");
        nccl_build.arg(&nccl_so);
        nccl_build.arg("src/nccl_shim.c");
        if !is_macos {
            nccl_build.arg("-Wl,-soname,libnccl.so.2");
        }
        let status = nccl_build.status().unwrap();
        assert!(status.success(), "failed to build embedded NCCL shim");

        let nccl_link = profile_dir.join(if is_macos {
            "libnccl.dylib"
        } else {
            "libnccl.so"
        });
        let _ = std::fs::remove_file(&nccl_link);
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            if is_macos {
                "libnccl.2.dylib"
            } else {
                "libnccl.so.2"
            },
            &nccl_link,
        )
        .unwrap();

        println!("cargo:rustc-link-search=native={}", profile_dir.display());
        if is_macos {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
            println!("cargo:rustc-link-lib=dylib=hetgpu_cuda_shim");
        } else {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-lhetgpu_cuda_shim");
            println!("cargo:rustc-link-arg=-Wl,--as-needed");
        }
    }
}
