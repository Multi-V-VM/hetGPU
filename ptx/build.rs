use std::path::PathBuf;

fn main() {
    // Find the built LLVM tools from llvm_zluda
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut base_path = PathBuf::from(&manifest_dir);
    base_path.pop(); // Go to workspace root

    // Look for the llvm_zluda build output
    let target_dir = base_path.join("target");

    // Try to find llvm_zluda build directory
    if let Ok(entries) = std::fs::read_dir(target_dir.join("debug").join("build")) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("llvm_zluda-"))
                .unwrap_or(false)
            {
                // Try multiple possible tool locations
                let tool_paths = [
                    path.join("out").join("build").join("bin"),
                    path.join("out").join("build").join("Debug").join("bin"),
                    path.join("out").join("build").join("Release").join("bin"),
                    path.join("out").join("bin"),
                ];

                for tool_path in &tool_paths {
                    let llc_path = tool_path.join("llc");
                    let llvm_dis_path = tool_path.join("llvm-dis");

                    if llc_path.exists() && llvm_dis_path.exists() {
                        println!("cargo:rustc-env=LLC_PATH={}", llc_path.display());
                        println!("cargo:rustc-env=LLVM_DIS_PATH={}", llvm_dis_path.display());
                        println!("cargo:warning=Found LLVM tools at: {}", tool_path.display());
                        return;
                    }
                }
            }
        }
    }

    println!("cargo:warning=Could not find built LLVM tools, will fall back to system tools");
}
