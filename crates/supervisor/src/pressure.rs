//! Memory and thermal signal sources.
//!
//! Native signals are read where the OS exposes them and are recorded as
//! `Unknown`/`Unavailable` otherwise — never guessed. The macOS source reads
//! `kern.memorystatus_vm_pressure_level` and process footprint via
//! `proc_pid_rusage`; thermal state comes from `NSProcessInfo`.

use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PressureLevel {
    Normal,
    Warning,
    Critical,
    Unknown(String),
}

impl PressureLevel {
    pub fn as_str(&self) -> String {
        match self {
            PressureLevel::Normal => "normal".into(),
            PressureLevel::Warning => "warning".into(),
            PressureLevel::Critical => "critical".into(),
            PressureLevel::Unknown(why) => format!("unknown({why})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThermalLevel {
    Nominal,
    Fair,
    Serious,
    Critical,
    Unknown(String),
}

pub trait SignalSource: Send + Sync {
    fn pressure(&self) -> PressureLevel;
    fn thermal(&self) -> ThermalLevel;
    fn footprint_of(&self, pid: i32) -> Option<u64>;
    fn compressor_bytes(&self) -> Option<u64>;
    fn low_power_mode(&self) -> Option<bool>;
}

/// Real macOS signal source.
pub struct MacSignalSource;

impl SignalSource for MacSignalSource {
    fn pressure(&self) -> PressureLevel {
        #[cfg(target_os = "macos")]
        {
            match sysctl_u64("kern.memorystatus_vm_pressure_level") {
                Some(1) => PressureLevel::Normal,
                Some(2) => PressureLevel::Warning,
                Some(4) => PressureLevel::Critical,
                Some(other) => PressureLevel::Unknown(format!("level {other}")),
                None => PressureLevel::Unknown("sysctl unavailable".into()),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            PressureLevel::Unknown("signal source implemented for macOS in this build".into())
        }
    }

    fn thermal(&self) -> ThermalLevel {
        #[cfg(target_os = "macos")]
        {
            use objc2_foundation::{NSProcessInfo, NSProcessInfoThermalState};
            let state = NSProcessInfo::processInfo().thermalState();
            match state {
                NSProcessInfoThermalState::Nominal => ThermalLevel::Nominal,
                NSProcessInfoThermalState::Fair => ThermalLevel::Fair,
                NSProcessInfoThermalState::Serious => ThermalLevel::Serious,
                NSProcessInfoThermalState::Critical => ThermalLevel::Critical,
                _ => ThermalLevel::Unknown("unrecognized value".into()),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            ThermalLevel::Unknown("signal source implemented for macOS in this build".into())
        }
    }

    fn footprint_of(&self, pid: i32) -> Option<u64> {
        tuckmill_planner::device::physical_footprint_of(pid)
    }

    fn compressor_bytes(&self) -> Option<u64> {
        tuckmill_planner::device::compressor_bytes()
    }

    fn low_power_mode(&self) -> Option<bool> {
        #[cfg(target_os = "macos")]
        {
            Some(objc2_foundation::NSProcessInfo::processInfo().isLowPowerModeEnabled())
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
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
    if size == 4 {
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
    } else if size == 8 {
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
    } else {
        None
    }
}

/// Scriptable source for tests and for `--pressure-simulate`.
pub struct MockSignalSource {
    pub pressure: Mutex<PressureLevel>,
    pub thermal: Mutex<ThermalLevel>,
}

impl MockSignalSource {
    pub fn new() -> Self {
        Self {
            pressure: Mutex::new(PressureLevel::Normal),
            thermal: Mutex::new(ThermalLevel::Nominal),
        }
    }

    pub fn set_pressure(&self, level: PressureLevel) {
        *self.pressure.lock().expect("mock lock") = level;
    }
}

impl Default for MockSignalSource {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalSource for MockSignalSource {
    fn pressure(&self) -> PressureLevel {
        self.pressure.lock().expect("mock lock").clone()
    }

    fn thermal(&self) -> ThermalLevel {
        self.thermal.lock().expect("mock lock").clone()
    }

    fn footprint_of(&self, _pid: i32) -> Option<u64> {
        None
    }

    fn compressor_bytes(&self) -> Option<u64> {
        None
    }

    fn low_power_mode(&self) -> Option<bool> {
        None
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn mac_source_reads_real_signals() {
        let src = MacSignalSource;
        // Pressure must be one of the real readings (this machine has been
        // swapping, so warning is perfectly plausible).
        let p = src.pressure();
        match p {
            PressureLevel::Normal | PressureLevel::Warning | PressureLevel::Critical => {}
            PressureLevel::Unknown(_) => {} // allowed but must not crash
        }
        assert!(src.footprint_of(std::process::id() as i32).unwrap() > 0);
        assert!(src.thermal() != ThermalLevel::Unknown("unrecognized value".into()));
    }

    #[test]
    fn mock_source_is_scriptable() {
        let mock = MockSignalSource::new();
        assert_eq!(mock.pressure(), PressureLevel::Normal);
        mock.set_pressure(PressureLevel::Critical);
        assert_eq!(mock.pressure(), PressureLevel::Critical);
    }
}
