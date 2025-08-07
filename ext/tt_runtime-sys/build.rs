use std::env::VarError;
use std::path::Path;

fn main() -> Result<(), VarError> {
    // For now, compile with stub headers to get the wrapper building
    // Full TT Metal integration will be done in a separate phase
    println!("cargo:warning=Starting C++ compilation");

    // Use direct g++ compilation instead of cc crate for now
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let obj_file = format!("{}/tt_metal_wrapper.o", out_dir);
    let lib_file = format!("{}/libtt_metal_wrapper.a", out_dir);

    // Compile the C++ file
    let compile_status = std::process::Command::new("g++")
        .args(&[
            "-std=c++20",
            "-c",
            "src/tt_metal_wrapper.cpp",
            "-o",
            &obj_file,
        ])
        .status()
        .expect("Failed to run g++");

    if !compile_status.success() {
        panic!("C++ compilation failed");
    }

    // Create static library
    let ar_status = std::process::Command::new("ar")
        .args(&["rcs", &lib_file, &obj_file])
        .status()
        .expect("Failed to run ar");

    if !ar_status.success() {
        panic!("Static library creation failed");
    }

    println!("cargo:warning=C++ compilation completed successfully");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let lib_path = Path::new(&out_dir).join("libtt_metal_wrapper.a");

    // If the library doesn't exist, warn but don't fail
    if !lib_path.exists() {
        println!(
            "cargo:warning=libtt_metal_wrapper.a not found in OUT_DIR: {}",
            out_dir
        );
    } else {
        println!(
            "cargo:warning=Found libtt_metal_wrapper.a in OUT_DIR: {}",
            out_dir
        );
    }

    // Add the lib directory to the library search path
    println!("cargo:rustc-link-search=native=lib");
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-search=native=/repo/tt-metal/build_Release/lib");

    // Link to the TT Metal libraries
    println!("cargo:rustc-link-lib=static=tt_metal_wrapper"); // Our wrapper
    println!("cargo:rustc-link-lib=dylib=tt_metal"); // TT Metal library
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=atomic"); // Atomic library for 64-bit operations
    Ok(())
}
