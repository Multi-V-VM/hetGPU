use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let version_script = manifest_dir
        .parent()
        .unwrap()
        .join("tools/cublas_shim/libcublas.map");
    if version_script.exists() {
        println!(
            "cargo:rustc-link-arg=-Wl,--version-script={}",
            version_script.display()
        );
    }
    println!("cargo:rustc-link-arg=-Wl,-soname,libcublas.so.12");

    #[cfg(unix)]
    {
        use std::fs;
        use std::os::unix::fs::symlink;

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        if let Some(profile_dir) = out_dir.ancestors().nth(3) {
            let symlink_path = profile_dir.join("libcublas.so.12");
            if symlink_path.exists() {
                let _ = fs::remove_file(&symlink_path);
            }
            let _ = symlink("libcublas.so", &symlink_path);
        }
    }
}
