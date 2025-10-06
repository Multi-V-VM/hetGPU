use cmake::Config;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

const COMPONENTS: &[&'static str] = &[
    "LLVMCore",
    "LLVMBitWriter",
    #[cfg(debug_assertions)]
    "LLVMAnalysis", // for module verify
    #[cfg(debug_assertions)]
    "LLVMBitReader",
];

fn main() {
    let mut cmake = Config::new(r"../ext/llvm-project/llvm");
    try_use_ninja(&mut cmake);
    cmake
        // It's not like we can do anything about the warnings
        .define("LLVM_ENABLE_WARNINGS", "OFF")
        .define("LLVM_ENABLE_TERMINFO", "OFF")
        .define("LLVM_ENABLE_LIBXML2", "OFF")
        .define("LLVM_ENABLE_LIBEDIT", "OFF")
        .define("LLVM_ENABLE_LIBPFM", "OFF")
        .define("LLVM_ENABLE_ZLIB", "OFF")
        .define("LLVM_ENABLE_ZSTD", "OFF")
        .define("LLVM_INCLUDE_BENCHMARKS", "OFF")
        .define("LLVM_INCLUDE_EXAMPLES", "OFF")
        .define("LLVM_INCLUDE_TESTS", "OFF")
        .define("LLVM_BUILD_TOOLS", "ON")
        // Build X86 for linking and NVPTX for PTX generation with debug info
        .define("LLVM_TARGETS_TO_BUILD", "X86;NVPTX")
        .define("LLVM_ENABLE_PROJECTS", "");
    
    // For some reason Rust always links to release MSVCRT
    #[cfg(windows)]
    cmake.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
    
    // Override problematic Windows-specific C++ flags on non-Windows platforms
    #[cfg(not(windows))]
    cmake.define("CMAKE_CXX_FLAGS", "-ffunction-sections -fdata-sections -fPIC -m64");
    
    cmake.build_target("llvm-config");
    let llvm_dir = cmake.build();

    // Build llc and llvm-dis tools for debug round-trip
    cmake.build_target("llc");
    cmake.build();
    cmake.build_target("llvm-dis");
    cmake.build();

    for c in COMPONENTS {
        cmake.build_target(c);
        cmake.build();
    }
    let cmake_profile = cmake.get_profile();
    let (cxxflags, ldflags, libdir, lib_names, system_libs) =
        llvm_config(&llvm_dir, &["build", "bin", "llvm-config"])
            .or_else(|_| llvm_config(&llvm_dir, &["build", cmake_profile, "bin", "llvm-config"]))
            .unwrap();
    println!("cargo:rustc-link-arg={ldflags}");
    println!("cargo:rustc-link-search=native={libdir}");
    println!(
        "cargo:rustc-link-search=native={libdir}/../../../../../../../ext/llvm-project/build/lib"
    );
    for lib in system_libs.split_ascii_whitespace() {
        println!("cargo:rustc-link-arg={lib}");
    }
    link_llvm_components(lib_names);
    compile_cxx_lib(cxxflags);

    // Export LLVM tool paths for debug round-trip testing
    // Try multiple possible locations for the built tools
    let tool_paths = [
        llvm_dir.join("build").join("bin"),
        llvm_dir.join("build").join(cmake_profile).join("bin"),
        llvm_dir.join("bin"),
    ];

    for tool_path in &tool_paths {
        let llc_path = tool_path.join("llc");
        let llvm_dis_path = tool_path.join("llvm-dis");

        if llc_path.exists() && llvm_dis_path.exists() {
            println!("cargo:rustc-env=LLC_PATH={}", llc_path.display());
            println!("cargo:rustc-env=LLVM_DIS_PATH={}", llvm_dis_path.display());
            println!("cargo:warning=Found LLVM tools at: {}", tool_path.display());
            break;
        }
    }
}

fn try_use_ninja(cmake: &mut Config) {
    let mut cmd = Command::new("ninja");
    cmd.arg("--version");
    if let Ok(status) = cmd.status() {
        if status.success() {
            cmake.generator("Ninja");
        }
    }
}

fn llvm_config(
    llvm_build_dir: &PathBuf,
    path_to_llvm_config: &[&str],
) -> io::Result<(String, String, String, String, String)> {
    let mut llvm_build_path = llvm_build_dir.clone();
    llvm_build_path.extend(path_to_llvm_config);
    let mut cmd = Command::new(llvm_build_path);
    cmd.args([
        "--link-static",
        "--cxxflags",
        "--ldflags",
        "--libdir",
        "--libnames",
        "--system-libs",
    ]);
    for c in COMPONENTS {
        cmd.arg(c[4..].to_lowercase());
    }
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(io::Error::from(io::ErrorKind::Other));
    }
    let output = unsafe { String::from_utf8_unchecked(output.stdout) };
    let mut lines = output.lines();
    let cxxflags = lines.next().unwrap();
    let ldflags = lines.next().unwrap();
    let libdir = lines.next().unwrap();
    let lib_names = lines.next().unwrap();
    let system_libs = lines.next().unwrap();
    Ok((
        cxxflags.to_string(),
        ldflags.to_string(),
        libdir.to_string(),
        lib_names.to_string(),
        system_libs.to_string(),
    ))
}

fn compile_cxx_lib(cxxflags: String) {
    println!("cargo:warning=Compiling C++ library with CXXFLAGS: {}", cxxflags);
    let mut cc = cc::Build::new();
    
    // Force use of GCC toolchain on non-Windows platforms
    #[cfg(not(windows))]
    {
        let gpp_available = Command::new("g++")
            .arg("--version")
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        let clang_available = Command::new("clang++")
            .arg("--version")
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        let gpp_accepts_msvc_flags = if gpp_available {
            Command::new("g++")
                .arg("-?")
                .status()
                .map(|status| status.success())
                .unwrap_or(false)
        } else {
            false
        };

        let compiler = if gpp_available && !gpp_accepts_msvc_flags {
            "g++"
        } else if gpp_accepts_msvc_flags && Path::new("/usr/bin/c++.bak").exists() {
            "/usr/bin/c++.bak"
        } else if clang_available {
            "clang++"
        } else if gpp_available {
            "g++"
        } else {
            // Fall back to g++ so build failures surface with a clear error
            "g++"
        };

        cc.compiler(compiler);
        cc.archiver("ar");
    }
    
    for flag in cxxflags.split_whitespace() {
        cc.flag(flag);
    }
    cc.cpp(true).file("src/lib.cpp");
    
    // Add required includes for LLVM
    cc.include("../ext/llvm-project/llvm/include");
    
    println!("cargo:warning=About to compile lib.cpp");
    cc.compile("llvm_zluda_cpp");
    println!("cargo:warning=Finished compiling lib.cpp");
    
    println!("cargo:rerun-if-changed=src/lib.cpp");
    println!("cargo:rerun-if-changed=src/lib.rs");
}

fn link_llvm_components(components: String) {
    for component in components.split_whitespace() {
        let component = if let Some(component) = component
            .strip_prefix("lib")
            .and_then(|component| component.strip_suffix(".a"))
        {
            // Unix (Linux/Mac)
            // libLLVMfoo.a
            component
        } else if let Some(component) = component.strip_suffix(".lib") {
            // Windows
            // LLVMfoo.lib
            component
        } else {
            panic!("'{}' does not look like a static library name", component)
        };
        println!("cargo:rustc-link-lib={component}");
    }
    println!("cargo:rustc-link-lib=LLVMTarget");
}
