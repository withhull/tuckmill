//! Compiles the TuckCore–Metal C ABI (ADR-007) on macOS: one Objective-C
//! translation unit, linked against Metal and Foundation. The MSL kernels
//! are NOT compiled here — they ship as source (`native/metal/kernels.metal`)
//! and are compiled by the OS Metal compiler at context creation, so no
//! Xcode toolchain is required to build the repository.

fn main() {
    println!("cargo:rerun-if-changed=../../native/metal/tm_metal.m");
    println!("cargo:rerun-if-changed=../../native/metal/tm_metal.h");
    println!("cargo:rerun-if-changed=../../native/metal/kernels.metal");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    cc::Build::new()
        .file("../../native/metal/tm_metal.m")
        .flag("-fno-objc-arc")
        .flag("-fblocks")
        .warnings(true)
        .compile("tm_metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
