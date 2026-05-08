use cmake::Config;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

const COMPONENTS: &[&'static str] = &[
    "LLVMCore",
    "LLVMBitWriter",
    "LLVMAnalysis", // for module verify
    "LLVMBitReader",
    // X86 target support
    "LLVMX86CodeGen",
    "LLVMX86AsmParser",
    "LLVMX86Desc",
    "LLVMX86Disassembler",
    "LLVMX86Info",
    // NVPTX target support
    "LLVMNVPTXCodeGen",
    "LLVMNVPTXDesc",
    "LLVMNVPTXInfo",
    // RISC-V target support for PACC codegen.
    "LLVMRISCVCodeGen",
    "LLVMRISCVAsmParser",
    "LLVMRISCVDesc",
    "LLVMRISCVDisassembler",
    "LLVMRISCVInfo",
];

fn main() {
    println!("cargo:rerun-if-env-changed=LLVM_ZLUDA_PREBUILT");
    println!("cargo:rerun-if-env-changed=LLVM_ZLUDA_CMAKE_PROFILE");
    println!("cargo:rerun-if-env-changed=LLVM_ZLUDA_COMPILER_LAUNCHER");
    println!("cargo:rerun-if-env-changed=CMAKE_C_COMPILER_LAUNCHER");
    println!("cargo:rerun-if-env-changed=CMAKE_CXX_COMPILER_LAUNCHER");
    println!("cargo:rerun-if-env-changed=CMAKE_ASM_COMPILER_LAUNCHER");

    // Allow using a pre-built LLVM installation via environment variable.
    // Set LLVM_ZLUDA_PREBUILT to the LLVM prefix directory (e.g. /usr/lib/llvm-18).
    if let Ok(prebuilt) = std::env::var("LLVM_ZLUDA_PREBUILT") {
        let llvm_prefix = PathBuf::from(&prebuilt);
        let llvm_config_path = llvm_prefix.join("bin").join("llvm-config");
        println!("cargo:warning=Using pre-built LLVM from: {}", prebuilt);

        let (cxxflags, ldflags, libdir, lib_names, system_libs) =
            llvm_config_from_path(&llvm_config_path).expect("Failed to run llvm-config");

        compile_cxx_lib_with_include(
            cxxflags,
            llvm_prefix.join("include").to_str().unwrap().to_string(),
        );
        println!("cargo:rustc-link-arg={ldflags}");
        println!("cargo:rustc-link-search=native={libdir}");
        link_llvm_components(lib_names);
        for lib in system_libs.split_ascii_whitespace() {
            println!("cargo:rustc-link-arg={lib}");
        }

        let llc_path = llvm_prefix.join("bin").join("llc");
        let llvm_dis_path = llvm_prefix.join("bin").join("llvm-dis");
        if llc_path.exists() {
            println!("cargo:rustc-env=LLC_PATH={}", llc_path.display());
        }
        if llvm_dis_path.exists() {
            println!("cargo:rustc-env=LLVM_DIS_PATH={}", llvm_dis_path.display());
        }
        return;
    }

    let mut cmake = Config::new(r"../ext/llvm-project/llvm");
    try_use_ninja(&mut cmake);
    configure_compiler_launcher(&mut cmake);
    let cmake_profile =
        std::env::var("LLVM_ZLUDA_CMAKE_PROFILE").unwrap_or_else(|_| "Release".to_string());
    cmake.profile(&cmake_profile);
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
        // Build X86 for host-side helpers, NVPTX for PTX/debug flows, and
        // RISCV for PACC object generation. Clang is built from the same
        // LLVM tree so PACC never has to fall back to system clang.
        .define("LLVM_TARGETS_TO_BUILD", "X86;NVPTX;RISCV")
        .define("LLVM_ENABLE_PROJECTS", "clang");

    // For some reason Rust always links to release MSVCRT
    #[cfg(windows)]
    cmake.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");

    // Override problematic Windows-specific C++ flags on non-Windows platforms.
    #[cfg(not(windows))]
    {
        let mut cxx_flags = "-ffunction-sections -fdata-sections -fPIC".to_string();
        if matches!(std::env::consts::ARCH, "x86" | "x86_64") {
            cxx_flags.push_str(" -m64");
        }
        cmake.define("CMAKE_CXX_FLAGS", cxx_flags);
    }

    cmake.build_target("llvm-config");
    let llvm_dir = cmake.build();

    // Build the tools PACC uses from this LLVM tree. Keeping llvm-link/opt/
    // clang in lockstep with llvm-sys avoids mixing system tools with
    // the llvm_zluda LLVM 21 libraries.
    for tool in ["llc", "llvm-dis", "llvm-link", "opt", "clang"] {
        cmake.build_target(tool);
        cmake.build();
    }

    for c in COMPONENTS {
        cmake.build_target(c);
        cmake.build();
    }
    let cmake_profile = cmake.get_profile();
    let (cxxflags, ldflags, libdir, lib_names, system_libs) =
        llvm_config(&llvm_dir, &["build", "bin", "llvm-config"])
            .or_else(|_| llvm_config(&llvm_dir, &["build", cmake_profile, "bin", "llvm-config"]))
            .unwrap();
    compile_cxx_lib(cxxflags);
    println!("cargo:rustc-link-arg={ldflags}");
    println!("cargo:rustc-link-search=native={libdir}");
    println!(
        "cargo:rustc-link-search=native={libdir}/../../../../../../../ext/llvm-project/build/lib"
    );
    link_llvm_components(lib_names);
    for lib in system_libs.split_ascii_whitespace() {
        println!("cargo:rustc-link-arg={lib}");
    }

    // Export LLVM tool paths for debug round-trip testing
    // Try multiple possible locations for the built tools
    let tool_paths = [
        llvm_dir.join("build").join("bin"),
        llvm_dir.join("build").join("tools"),
        llvm_dir.join("build").join(cmake_profile).join("bin"),
        llvm_dir.join("build").join(cmake_profile).join("tools"),
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

fn configure_compiler_launcher(cmake: &mut Config) {
    let launcher = std::env::var("LLVM_ZLUDA_COMPILER_LAUNCHER")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(find_default_compiler_launcher);

    if let Some(launcher) = launcher {
        println!("cargo:warning=Using LLVM compiler launcher: {launcher}");
        cmake.define("CMAKE_C_COMPILER_LAUNCHER", &launcher);
        cmake.define("CMAKE_CXX_COMPILER_LAUNCHER", &launcher);
        cmake.define("CMAKE_ASM_COMPILER_LAUNCHER", &launcher);
    }
}

fn find_default_compiler_launcher() -> Option<String> {
    for candidate in ["sccache", "ccache"] {
        if command_available(candidate) {
            return Some(candidate.to_string());
        }
    }
    None
}

fn command_available(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn llvm_config(
    llvm_build_dir: &PathBuf,
    path_to_llvm_config: &[&str],
) -> io::Result<(String, String, String, String, String)> {
    let mut llvm_build_path = llvm_build_dir.clone();
    llvm_build_path.extend(path_to_llvm_config);
    llvm_config_from_path(&llvm_build_path)
}

fn llvm_config_from_path(
    llvm_config_path: &PathBuf,
) -> io::Result<(String, String, String, String, String)> {
    let mut cmd = Command::new(llvm_config_path);
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
    compile_cxx_lib_with_include(cxxflags, "../ext/llvm-project/llvm/include".to_string());
}

fn compile_cxx_lib_with_include(cxxflags: String, include_path: String) {
    println!(
        "cargo:warning=Compiling C++ library with CXXFLAGS: {}",
        cxxflags
    );
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

    let forced_include = format!("-I{include_path}");
    cc.flag(&forced_include);
    for flag in cxxflags.split_whitespace() {
        if flag == forced_include {
            continue;
        }
        cc.flag(flag);
    }
    cc.cpp(true).file("src/lib.cpp");

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
