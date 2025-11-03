use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let version_script = manifest_dir
        .parent()
        .unwrap()
        .join("tools/cublas_shim/libcublasLt.map");
    if version_script.exists() {
        println!(
            "cargo:rustc-link-arg=-Wl,--version-script={}",
            version_script.display()
        );
    }
    println!("cargo:rustc-link-arg=-Wl,-soname,libcublasLt.so.12");

    #[cfg(unix)]
    {
        use std::fs;
        use std::os::unix::fs::symlink;

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        if let Some(profile_dir) = out_dir.ancestors().nth(3) {
            let symlink_path = profile_dir.join("libcublasLt.so.12");
            // Remove stale symlink/file if present
            if symlink_path.exists() {
                let _ = fs::remove_file(&symlink_path);
            }
            // Create symlink pointing to the compiled cdylib in the same directory
            let _ = symlink("libcublasLt.so", &symlink_path);
        }
    }
}
