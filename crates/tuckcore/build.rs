//! Emits the source revision for build identity (records must name the
//! source that produced them). Best-effort: "unknown" when git metadata is
//! absent (tarball builds).

fn main() {
    let rev = std::process::Command::new("git")
        .args(["-C", "../..", "rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=TUCKMILL_SOURCE_REV={rev}");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
