use std::env::VarError;
use std::{env, path::PathBuf};

fn main() -> Result<(), VarError> {
    let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("src");

    cc::Build::new()
        .file(src_dir.join("runner/ze_runner.c"))
        .include(src_dir.clone())
        .compile("ze_runner");

    if cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=ze_loader_1");
        let env_val = env::var("CARGO_CFG_TARGET_ENV")?;
        if env_val == "msvc" {
            let mut path = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
            path.push("lib");
            println!("cargo:rustc-link-search=native={}", path.display());
        } else {
            println!("cargo:rustc-link-search=native=C:\\Windows\\System32");
        }
    } else {
        // Only link ze_loader if it is actually present on this system.
        // On non-Intel platforms (RISC-V, ARM, etc.) ze_loader is not installed,
        // and building pacc/gemmini/tenstorrent features should not fail because of it.
        let search_dirs = [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/local/lib",
            "/usr/lib",
        ];
        let found = search_dirs.iter().any(|d| {
            std::path::Path::new(d).join("libze_loader.so").exists()
                || std::path::Path::new(d).join("libze_loader.so.1").exists()
        });
        if found {
            println!("cargo:rustc-link-lib=dylib=ze_loader");
            for d in &search_dirs {
                println!("cargo:rustc-link-search=native={}", d);
            }
        } else {
            cc::Build::new()
                .file(src_dir.join("runner/ze_stub.c"))
                .include(src_dir.clone())
                .compile("ze_loader_stub");

            // Provide local failing stubs so cdylibs can be loaded on systems
            // without Level Zero. Consumers that use another backend should
            // not be blocked by unresolved ze_* symbols.
            println!("cargo:warning=ze_loader not found — Intel Level Zero runtime disabled");
        }
    }

    println!("cargo:rerun-if-changed=src/runner/ze_runner.c");
    println!("cargo:rerun-if-changed=src/runner/ze_runner.h");
    println!("cargo:rerun-if-changed=src/runner/ze_stub.c");
    println!("cargo:rerun-if-changed=src/level-zero/ze_api.h");

    Ok(())
}
