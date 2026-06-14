use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if !matches!(target_os.as_str(), "macos" | "ios") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let zluda_src = manifest_dir.join("../../zluda/src");
    let metal_runtime = zluda_src.join("apple_metal_runtime.m");
    let ane_bridge = zluda_src.join("ane_bridge.m");
    let ane_header = zluda_src.join("ane_bridge.h");
    let compat_include_dir = out_dir.join("SDKCompat");

    println!("cargo:rerun-if-changed={}", metal_runtime.display());
    println!("cargo:rerun-if-changed={}", ane_bridge.display());
    println!("cargo:rerun-if-changed={}", ane_header.display());

    prepare_sdk_compat_headers(&compat_include_dir)
        .expect("failed to prepare Apple SDK compatibility headers");

    let mut build = cc::Build::new();
    build
        .file(&metal_runtime)
        .file(&ane_bridge)
        .include(&compat_include_dir)
        .include(&zluda_src)
        .flag("-fobjc-arc");

    if target_os == "ios" {
        let target = env::var("TARGET").unwrap_or_default();
        if target.ends_with("-sim")
            || target.contains("ios-sim")
            || target.contains("ios-simulator")
        {
            build.flag("-mios-simulator-version-min=15.0");
        } else {
            build.flag("-mios-version-min=15.0");
        }
    }

    build.compile("hetgpu_apple_runtime");

    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=IOSurface");
}

fn prepare_sdk_compat_headers(include_dir: &PathBuf) -> io::Result<()> {
    let iosurface_dir = include_dir.join("IOSurface");
    fs::create_dir_all(&iosurface_dir)?;
    fs::write(
        iosurface_dir.join("IOSurface.h"),
        r#"#ifndef HETGPU_IOSURFACE_COMPAT_H
#define HETGPU_IOSURFACE_COMPAT_H

#include <CoreFoundation/CoreFoundation.h>
#include <stdint.h>

typedef struct __IOSurface *IOSurfaceRef;
typedef int kern_return_t;

extern const CFStringRef kIOSurfaceWidth;
extern const CFStringRef kIOSurfaceHeight;
extern const CFStringRef kIOSurfaceBytesPerElement;
extern const CFStringRef kIOSurfaceBytesPerRow;
extern const CFStringRef kIOSurfaceAllocSize;
extern const CFStringRef kIOSurfacePixelFormat;

enum {
    kIOSurfaceLockReadOnly = 0x00000001
};

IOSurfaceRef IOSurfaceCreate(CFDictionaryRef properties);
kern_return_t IOSurfaceLock(IOSurfaceRef buffer, uint32_t options, uint32_t *seed);
kern_return_t IOSurfaceUnlock(IOSurfaceRef buffer, uint32_t options, uint32_t *seed);
void *IOSurfaceGetBaseAddress(IOSurfaceRef buffer);

#endif
"#,
    )
}
