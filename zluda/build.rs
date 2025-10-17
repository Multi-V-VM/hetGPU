fn main() {
    // Only build and link embedded cudart shim when the feature is enabled.
    let embed = std::env::var("CARGO_FEATURE_EMBED_CUDART").is_ok();
    if embed {
        println!("cargo:rerun-if-changed=src/cudart_shim.c");

        // Use the version script from tools/cudart_shim
        let cargo_manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let version_script = std::path::Path::new(&cargo_manifest_dir)
            .parent()
            .unwrap()
            .join("tools/cudart_shim/cudart_shim.map");

        if version_script.exists() {
            println!("cargo:rustc-link-arg=-Wl,--version-script={}", version_script.display());
        } else {
            eprintln!("Warning: Version script not found at {}", version_script.display());
        }

        // Force the linker to keep all symbols from the cudart_shim archive
        // using --whole-archive to prevent stripping unreferenced symbols
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg=-lcudart_shim");
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

        let mut build = cc::Build::new();
        build.file("src/cudart_shim.c");
        build.flag("-fPIC");
        build.flag("-Wno-unused-parameter");
        build.compile("cudart_shim");
    }
}
