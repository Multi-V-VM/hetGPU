// Build script for gemmini_comgr-sys
// This sets up the environment for using Buddy Compiler for MLIR processing

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Check for Buddy Compiler installation
    if let Ok(buddy_path) = std::env::var("BUDDY_COMPILER_PATH") {
        println!("cargo:rustc-env=BUDDY_COMPILER_PATH={}", buddy_path);
    } else {
        // Try to find buddy-opt in PATH
        if let Ok(output) = std::process::Command::new("which")
            .arg("buddy-opt")
            .output()
        {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let buddy_dir = std::path::Path::new(&path)
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "/usr/local/bin".to_string());
                println!("cargo:rustc-env=BUDDY_COMPILER_PATH={}", buddy_dir);
            }
        }
    }

    // Check for individual Buddy tools
    if std::process::Command::new("buddy-opt")
        .arg("--help")
        .output()
        .is_ok()
    {
        println!("cargo:rustc-cfg=has_buddy_opt");
        println!("cargo:warning=Found buddy-opt optimizer");
    } else {
        println!("cargo:warning=buddy-opt not found, will use fallback");
    }

    if std::process::Command::new("buddy-translate")
        .arg("--help")
        .output()
        .is_ok()
    {
        println!("cargo:rustc-cfg=has_buddy_translate");
        println!("cargo:warning=Found buddy-translate");
    } else {
        println!("cargo:warning=buddy-translate not found, will use fallback");
    }

    // Check for MLIR tools
    if std::process::Command::new("mlir-opt")
        .arg("--version")
        .output()
        .is_ok()
    {
        println!("cargo:rustc-cfg=has_mlir_opt");
        println!("cargo:warning=Found MLIR optimizer");
    }

    if std::process::Command::new("mlir-translate")
        .arg("--version")
        .output()
        .is_ok()
    {
        println!("cargo:rustc-cfg=has_mlir_translate");
        println!("cargo:warning=Found MLIR translator");
    }

    // Check for LLVM tools
    if std::process::Command::new("llc")
        .arg("--version")
        .output()
        .is_ok()
    {
        println!("cargo:rustc-cfg=has_llc");
        println!("cargo:warning=Found LLVM compiler (llc)");
    }

    // Set compile-time configuration
    println!("cargo:rustc-env=GEMMINI_TARGET=riscv64-unknown-elf");
    println!("cargo:rustc-env=MLIR_DIALECT_VERSION=1.0");
}
