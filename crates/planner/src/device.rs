//! Hardware profiler.
//!
//! Reports what was measured, with the method. Signals that this build cannot
//! read (Metal working set, thermal state on unsupported paths, battery
//! status) are recorded as unavailable or unknown — never guessed.

use std::path::Path;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ids::{hash_of, now_unix_ms, Id};
use tuckmill_manifest::records::{
    CapabilityStatus, DeviceProfile, GpuProfile, PowerProfile, StorageProfile,
};

/// Sample the device. `data_path` selects the storage volume the report
/// measures; it does not have to exist yet (its nearest parent is used).
pub fn detect(data_path: &Path) -> Result<DeviceProfile> {
    #[cfg(target_os = "macos")]
    {
        detect_macos(data_path)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = data_path;
        Err(TuckmillError::new(
            ErrorCode::Unavailable,
            "device detection is implemented for macOS only in this build",
        ))
    }
}

#[cfg(target_os = "macos")]
fn detect_macos(data_path: &Path) -> Result<DeviceProfile> {
    let mut capabilities = std::collections::BTreeMap::new();

    let chip = sysctl_string("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".to_string());
    let memory_bytes = sysctl_u64("hw.memsize")
        .ok_or_else(|| TuckmillError::new(ErrorCode::Unavailable, "hw.memsize is unavailable"))?;
    let cpu_cores_total = sysctl_u64("hw.ncpu").unwrap_or(1) as u32;
    let cpu_cores_performance = sysctl_u64("hw.perflevel0.physicalcpu").map(|v| v as u32);
    let cpu_cores_efficiency = sysctl_u64("hw.perflevel1.physicalcpu").map(|v| v as u32);

    let os_version = sysctl_string("kern.osproductversion").unwrap_or_else(|| "unknown".into());
    let os_build = sysctl_string("kern.osversion").unwrap_or_else(|| "unknown".into());

    let (family_estimate, family_source) = map_chip_to_gpu_family(&chip);
    capabilities.insert(
        "gpu_family".to_string(),
        CapabilityStatus::unknown(format!(
            "{family_source}; runtime Metal capability query is not yet implemented (native bridge)"
        )),
    );
    capabilities.insert(
        "recommended_max_working_set".to_string(),
        CapabilityStatus::unavailable(
            "requires the Metal API bridge, which is not implemented in this build; the value is advisory for GPU allocation performance and is not available system memory",
        ),
    );

    let (storage, storage_capability) = storage_profile(data_path);
    capabilities.insert("storage".to_string(), storage_capability);

    let (thermal_start, thermal_status) = thermal_state();
    capabilities.insert("thermal_state".to_string(), thermal_status);

    let (pressure, pressure_status) = memory_pressure();
    capabilities.insert("memory_pressure".to_string(), pressure_status);

    let (low_power, power_status) = low_power_mode();
    capabilities.insert("low_power_mode".to_string(), power_status);
    capabilities.insert(
        "battery_status".to_string(),
        CapabilityStatus::unavailable(
            "battery/AC detection requires the IOKit bridge; not implemented in this build",
        ),
    );
    capabilities.insert(
        "compressor_stats".to_string(),
        CapabilityStatus::available("via host_statistics64 (used by the governor)"),
    );

    // Device identity is stable across samples but changes when a material
    // input changes (OS update, storage volume move), which is exactly what
    // must invalidate cached plans (PRD §5).
    #[derive(serde::Serialize)]
    struct Identity<'a> {
        chip: &'a str,
        cpu_cores_total: u32,
        cpu_cores_performance: Option<u32>,
        cpu_cores_efficiency: Option<u32>,
        memory_bytes: u64,
        os_version: &'a str,
        os_build: &'a str,
        volume_identity: &'a Id,
        path_class: &'a str,
    }
    let device_id = hash_of(&Identity {
        chip: &chip,
        cpu_cores_total,
        cpu_cores_performance,
        cpu_cores_efficiency,
        memory_bytes,
        os_version: &os_version,
        os_build: &os_build,
        volume_identity: &storage.volume_identity,
        path_class: &storage.path_class,
    })?;

    Ok(DeviceProfile {
        schema_version: 1,
        device_id,
        sampled_unix_ms: now_unix_ms(),
        chip,
        cpu_cores_total,
        cpu_cores_performance,
        cpu_cores_efficiency,
        memory_bytes,
        os_name: "macOS".into(),
        os_version,
        os_build,
        gpu: GpuProfile {
            family_estimate,
            family_source,
            recommended_working_set_bytes: None,
        },
        storage,
        power: PowerProfile {
            on_battery: None,
            low_power_mode: low_power,
        },
        thermal_start,
        pressure,
        capabilities,
    })
}

/// Static chip-to-GPU-family mapping (Apple GPU families follow chip
/// generations: S26). The plan must treat this as an estimate until the
/// native bridge queries `MTLDevice.supportsFamily`.
fn map_chip_to_gpu_family(chip: &str) -> (String, String) {
    const SOURCE: &str =
        "static chip mapping; verify via MTLDevice.supportsFamily in the native bridge";
    let family = if chip.contains("M1") {
        "apple7"
    } else if chip.contains("M2") {
        "apple8"
    } else if chip.contains("M3") {
        "apple9"
    } else if chip.contains("M4") {
        "apple10"
    } else {
        return (
            "unknown".to_string(),
            format!("{SOURCE}; chip not in mapping table"),
        );
    };
    (family.to_string(), SOURCE.to_string())
}

#[cfg(target_os = "macos")]
fn storage_profile(data_path: &Path) -> (StorageProfile, CapabilityStatus) {
    let mut probe = data_path.to_path_buf();
    while !probe.exists() {
        match probe.parent() {
            Some(parent) => probe = parent.to_path_buf(),
            None => break,
        }
    }
    match statfs_of(&probe) {
        Some((free, total, volume_identity)) => {
            let path_class = if probe.to_string_lossy().starts_with("/Volumes/") {
                "external-volume"
            } else {
                "internal-ssd"
            };
            (
                StorageProfile {
                    path_class: path_class.to_string(),
                    volume_identity,
                    free_bytes: free,
                    total_bytes: total,
                },
                CapabilityStatus::available(format!("statfs on {}", probe.display())),
            )
        }
        None => (
            StorageProfile {
                path_class: "unknown".into(),
                volume_identity: Id::new("0".repeat(64)).expect("zero id"),
                free_bytes: 0,
                total_bytes: 0,
            },
            CapabilityStatus::unknown("statfs failed for the data path".to_string()),
        ),
    }
}

#[cfg(target_os = "macos")]
fn statfs_of(path: &Path) -> Option<(u64, u64, Id)> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c = CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: valid C string pointer; statfs writes into a properly sized
    // zeroed buffer owned by this scope.
    unsafe {
        let mut buf: libc::statfs = std::mem::zeroed();
        if libc::statfs(c.as_ptr(), &mut buf) != 0 {
            return None;
        }
        let bsize = buf.f_bsize as u64;
        let total = (buf.f_blocks as u64).saturating_mul(bsize);
        let free = (buf.f_bavail as u64).saturating_mul(bsize);
        // SAFETY: fsid_t is a plain 8-byte pair of integers in the statfs
        // result; reading its raw bytes for a volume identity is well-defined.
        let fsid_bytes = std::slice::from_raw_parts(
            &buf.f_fsid as *const libc::fsid_t as *const u8,
            std::mem::size_of::<libc::fsid_t>(),
        );
        let identity = Id::new(tuckmill_manifest::ids::sha256_hex(
            &[fsid_bytes, &buf.f_fssubtype.to_le_bytes()].concat(),
        ))
        .ok()?;
        Some((free, total, identity))
    }
}

#[cfg(target_os = "macos")]
fn thermal_state() -> (String, CapabilityStatus) {
    use objc2_foundation::{NSProcessInfo, NSProcessInfoThermalState};
    let info = NSProcessInfo::processInfo();
    let state = info.thermalState();
    let (label, ok) = match state {
        NSProcessInfoThermalState::Nominal => ("nominal", true),
        NSProcessInfoThermalState::Fair => ("fair", true),
        NSProcessInfoThermalState::Serious => ("serious", true),
        NSProcessInfoThermalState::Critical => ("critical", true),
        _ => ("unknown", false),
    };
    let status = if ok {
        CapabilityStatus::available("NSProcessInfo.thermalState")
    } else {
        CapabilityStatus::unknown("unrecognized thermal state value")
    };
    (label.to_string(), status)
}

#[cfg(target_os = "macos")]
fn low_power_mode() -> (Option<bool>, CapabilityStatus) {
    use objc2_foundation::NSProcessInfo;
    let info = NSProcessInfo::processInfo();
    let enabled = info.isLowPowerModeEnabled();
    (
        Some(enabled),
        CapabilityStatus::available("NSProcessInfo.isLowPowerModeEnabled"),
    )
}

/// Map `kern.memorystatus_vm_pressure_level` (1 = normal, 2 = warning,
/// 4 = critical per XNU memorystatus) into a pressure label. Unknown levels
/// are reported with their raw value instead of being coerced.
#[cfg(target_os = "macos")]
fn memory_pressure() -> (String, CapabilityStatus) {
    match sysctl_u64("kern.memorystatus_vm_pressure_level") {
        Some(1) => (
            "normal".to_string(),
            CapabilityStatus::available("kern.memorystatus_vm_pressure_level=1 (normal)"),
        ),
        Some(2) => (
            "warning".to_string(),
            CapabilityStatus::available("kern.memorystatus_vm_pressure_level=2 (warning)"),
        ),
        Some(4) => (
            "critical".to_string(),
            CapabilityStatus::available("kern.memorystatus_vm_pressure_level=4 (critical)"),
        ),
        Some(other) => (
            format!("unknown(level {other})"),
            CapabilityStatus::unknown("unrecognized pressure level value"),
        ),
        None => (
            "unavailable".to_string(),
            CapabilityStatus::unavailable("kern.memorystatus_vm_pressure_level is not readable"),
        ),
    }
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    use std::ffi::CString;
    let c = CString::new(name).ok()?;
    let mut size: libc::size_t = 0;
    // SAFETY: standard sysctl size probe with a null buffer.
    unsafe {
        if libc::sysctlbyname(
            c.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }
    match size {
        4 => {
            let mut v: u32 = 0;
            let mut s = 4usize;
            // SAFETY: buffer and size match the probed width.
            unsafe {
                if libc::sysctlbyname(
                    c.as_ptr(),
                    &mut v as *mut u32 as *mut libc::c_void,
                    &mut s,
                    std::ptr::null_mut(),
                    0,
                ) != 0
                {
                    return None;
                }
            }
            Some(v as u64)
        }
        8 => {
            let mut v: u64 = 0;
            let mut s = 8usize;
            // SAFETY: buffer and size match the probed width.
            unsafe {
                if libc::sysctlbyname(
                    c.as_ptr(),
                    &mut v as *mut u64 as *mut libc::c_void,
                    &mut s,
                    std::ptr::null_mut(),
                    0,
                ) != 0
                {
                    return None;
                }
            }
            Some(v)
        }
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn sysctl_string(name: &str) -> Option<String> {
    use std::ffi::CString;
    let c = CString::new(name).ok()?;
    let mut size: libc::size_t = 0;
    // SAFETY: standard sysctl size probe with a null buffer.
    unsafe {
        if libc::sysctlbyname(
            c.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
            || size == 0
        {
            return None;
        }
    }
    let mut buf = vec![0u8; size];
    // SAFETY: buffer sized from the probe.
    unsafe {
        if libc::sysctlbyname(
            c.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }
    buf.truncate(size);
    while buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8(buf).ok()
}

/// Physical footprint of a process in bytes via `proc_pid_rusage`
/// (RUSAGE_INFO_V4 `ri_phys_footprint`). Returns None when unavailable.
#[cfg(target_os = "macos")]
pub fn physical_footprint_of(pid: i32) -> Option<u64> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY (call): proc_pid_rusage writes the flavor's struct into the
    // buffer we pass; libc models the C `(rusage_info_t *)&buffer` cast in
    // its signature, so the buffer address is passed with that pointer type.
    let rc = unsafe {
        libc::proc_pid_rusage(
            pid,
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr() as *mut libc::rusage_info_t,
        )
    };
    if rc != 0 {
        return None;
    }
    // SAFETY: initialized by the successful call above.
    let info = unsafe { info.assume_init() };
    Some(info.ri_phys_footprint)
}

#[cfg(not(target_os = "macos"))]
pub fn physical_footprint_of(_pid: i32) -> Option<u64> {
    None
}

/// Compressor resident bytes via host_statistics64 when available.
#[cfg(target_os = "macos")]
#[allow(deprecated)] // mach_host_self deprecation notice suggests mach2; libc path kept to avoid a heavier dependency for one call.
pub fn compressor_bytes() -> Option<u64> {
    // SAFETY: host_statistics64 with the HOST_VM_INFO64 flavor into a sized
    // struct; page size read from the vm_page_size static.
    unsafe {
        let mut stats: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        if libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            &mut stats as *mut libc::vm_statistics64 as *mut libc::integer_t,
            &mut count,
        ) != 0
        {
            return None;
        }
        let page_size = libc::vm_page_size as u64;
        Some(stats.compressor_page_count as u64 * page_size)
    }
}

#[cfg(not(target_os = "macos"))]
pub fn compressor_bytes() -> Option<u64> {
    None
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn detects_real_device() {
        let dir = tempfile::tempdir().unwrap();
        let profile = detect(dir.path()).unwrap();
        assert!(!profile.chip.is_empty());
        assert!(profile.memory_bytes > 0);
        assert!(!profile.os_version.is_empty());
        assert_eq!(
            profile.capabilities["storage"].status,
            tuckmill_manifest::records::CapabilityStatusKind::Available
        );
        // Device id is stable across samples.
        let again = detect(dir.path()).unwrap();
        assert_eq!(profile.device_id, again.device_id);
        // Thermal state is a real reading on macOS.
        assert!(["nominal", "fair", "serious", "critical", "unknown"]
            .contains(&profile.thermal_start.as_str()));
    }

    #[test]
    fn footprint_of_self_is_available() {
        let fp = physical_footprint_of(std::process::id() as i32);
        assert!(fp.is_some(), "proc_pid_rusage should work for self");
        assert!(fp.unwrap() > 0);
    }

    #[test]
    fn compressor_is_available() {
        assert!(compressor_bytes().is_some());
    }

    #[test]
    fn chip_mapping_is_conservative() {
        let (f, s) = map_chip_to_gpu_family("Apple M3");
        assert_eq!(f, "apple9");
        assert!(s.contains("static"));
        let (f, _) = map_chip_to_gpu_family("Apple M9 Ultra");
        assert_eq!(f, "unknown");
    }
}
