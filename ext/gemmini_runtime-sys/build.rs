// Build script for gemmini_runtime-sys
// This sets up the environment for using Spike RISC-V ISA simulator with Gemmini extensions

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Check for Spike installation
    if let Ok(spike_path) = std::env::var("SPIKE_PATH") {
        println!("cargo:rustc-env=SPIKE_PATH={}", spike_path);
    } else {
        // Try to find spike in PATH
        if let Ok(output) = std::process::Command::new("which").arg("spike").output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("cargo:rustc-env=SPIKE_PATH={}", path);
            }
        }
    }

    // Check for RISC-V toolchain
    if let Ok(riscv_path) = std::env::var("RISCV") {
        println!("cargo:rustc-env=RISCV={}", riscv_path);
    }

    // Check for Gemmini library
    if let Ok(gemmini_lib) = std::env::var("GEMMINI_LIB_PATH") {
        println!("cargo:rustc-link-search=native={}", gemmini_lib);
        println!("cargo:rustc-link-lib=dylib=gemmini");
    }

    // Set features based on available tools
    if std::process::Command::new("spike")
        .arg("--help")
        .output()
        .is_ok()
    {
        println!("cargo:rustc-cfg=has_spike");
        println!("cargo:warning=Found Spike RISC-V ISA simulator");
    } else {
        println!("cargo:warning=Spike not found, will use mock execution");
    }

    if std::process::Command::new("riscv64-unknown-elf-gcc")
        .arg("--version")
        .output()
        .is_ok()
    {
        println!("cargo:rustc-cfg=has_riscv_toolchain");
        println!("cargo:warning=Found RISC-V toolchain");
    } else {
        println!("cargo:warning=RISC-V toolchain not found, will use fallback");
    }

    // Check for LLVM tools (llc for compilation)
    if std::process::Command::new("llc")
        .arg("--version")
        .output()
        .is_ok()
    {
        println!("cargo:rustc-cfg=has_llc");
        println!("cargo:warning=Found LLVM compiler (llc)");
    }

    // Compile-time configuration for Gemmini parameters
    println!("cargo:rustc-env=GEMMINI_DIM=16");
    println!("cargo:rustc-env=GEMMINI_SPAD_ROWS=256");
    println!("cargo:rustc-env=GEMMINI_ACC_ROWS=64");
}
