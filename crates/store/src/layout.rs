//! Store layout: directories, permissions, disk checks, volume identity.
//!
//! Canonical model data is under `~/Library/Application Support/Tuckmill`
//! unless `TUCKMILL_DATA_DIR` selects another root (for tests, or an
//! explicitly chosen external volume).

use std::path::{Path, PathBuf};

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ids::Id;
use tuckmill_manifest::units::Bytes;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskStatus {
    pub free_bytes: u64,
    pub total_bytes: u64,
    pub volume_identity: Id,
}

#[derive(Debug, Clone)]
pub struct StoreLayout {
    root: PathBuf,
}

impl StoreLayout {
    /// Resolve the data root: `TUCKMILL_DATA_DIR`, else Application Support.
    pub fn resolve() -> Result<Self> {
        if let Ok(dir) = std::env::var("TUCKMILL_DATA_DIR") {
            if !dir.trim().is_empty() {
                return Ok(Self {
                    root: PathBuf::from(dir),
                });
            }
        }
        let home = std::env::var("HOME").map_err(|_| {
            TuckmillError::new(
                ErrorCode::InvalidConfiguration,
                "HOME is not set; cannot locate the Application Support directory",
            )
        })?;
        Ok(Self {
            root: PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("Tuckmill"),
        })
    }

    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn artifacts_dir(&self) -> PathBuf {
        self.root.join("artifacts")
    }

    pub fn tmp_dir(&self) -> PathBuf {
        self.root.join("tmp")
    }

    pub fn downloads_dir(&self) -> PathBuf {
        self.root.join("downloads")
    }

    pub fn records_dir(&self) -> PathBuf {
        self.root.join("records")
    }

    pub fn plans_dir(&self) -> PathBuf {
        self.records_dir().join("plans")
    }

    pub fn sessions_dir(&self) -> PathBuf {
        self.records_dir().join("sessions")
    }

    pub fn bench_dir(&self) -> PathBuf {
        self.records_dir().join("bench")
    }

    pub fn control_dir(&self) -> PathBuf {
        self.root.join("control")
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }

    pub fn aliases_path(&self) -> PathBuf {
        self.root.join("aliases.json")
    }

    pub fn calibration_path(&self) -> PathBuf {
        self.root.join("calibration.json")
    }

    /// Artifact directory for a content id.
    pub fn artifact_dir(&self, id: &Id) -> PathBuf {
        self.artifacts_dir().join(&id.as_str()[..16])
    }

    pub fn artifact_payload_path(&self, id: &Id) -> PathBuf {
        self.artifact_dir(id).join("model.bin")
    }

    pub fn artifact_manifest_path(&self, id: &Id) -> PathBuf {
        self.artifact_dir(id).join("manifest.json")
    }

    pub fn artifact_pack_dir(&self, id: &Id) -> PathBuf {
        self.artifact_dir(id).join("pack")
    }

    /// Create the directory skeleton with user-only permissions where it
    /// matters. Existing directories are reused.
    pub fn ensure(&self) -> Result<()> {
        for dir in [
            self.root.clone(),
            self.artifacts_dir(),
            self.tmp_dir(),
            self.downloads_dir(),
            self.records_dir(),
            self.plans_dir(),
            self.sessions_dir(),
            self.bench_dir(),
            self.control_dir(),
            self.logs_dir(),
        ] {
            std::fs::create_dir_all(&dir).map_err(TuckmillError::from)?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o700);
            for dir in [self.root.clone(), self.control_dir(), self.tmp_dir()] {
                let _ = std::fs::set_permissions(&dir, perms.clone());
            }
        }
        Ok(())
    }

    /// Free/total bytes and a stable volume identity for a path inside the
    /// store. A mount path reused by a different drive yields a different
    /// identity, so it must not inherit previous trust (PRD §10).
    pub fn disk_status(&self, path: &Path) -> Result<DiskStatus> {
        let mut probe = path.to_path_buf();
        while !probe.exists() {
            match probe.parent() {
                Some(p) => probe = p.to_path_buf(),
                None => break,
            }
        }
        statfs(&probe).ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::StorageLost,
                format!("cannot read filesystem status for {}", probe.display()),
            )
        })
    }

    /// Preflight disk space. `need` is the peak requirement; the caller says
    /// what peaks coexist (source, destination, temporary, journal).
    pub fn ensure_space_for(&self, need: Bytes, path: &Path) -> Result<DiskStatus> {
        let status = self.disk_status(path)?;
        if status.free_bytes < need.bytes() {
            return Err(TuckmillError::new(
                ErrorCode::InsufficientDisk,
                format!(
                    "insufficient disk: need {} but only {} is free",
                    need,
                    Bytes(status.free_bytes)
                ),
            )
            .with_action(
                "Free space or choose a smaller profile; the source file was not modified.",
            ));
        }
        Ok(status)
    }
}

/// statfs wrapper returning (free, total, volume identity).
#[cfg(unix)]
fn statfs(path: &Path) -> Option<DiskStatus> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c = CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: valid C string; statfs writes into a zeroed, correctly-sized
    // struct owned by this scope.
    unsafe {
        let mut buf: libc::statfs = std::mem::zeroed();
        if libc::statfs(c.as_ptr(), &mut buf) != 0 {
            return None;
        }
        let bsize = buf.f_bsize as u64;
        let total = (buf.f_blocks as u64).saturating_mul(bsize);
        let free = (buf.f_bavail as u64).saturating_mul(bsize);
        // SAFETY: fsid is a plain 8-byte integer pair in statfs output.
        let fsid_bytes = std::slice::from_raw_parts(
            &buf.f_fsid as *const libc::fsid_t as *const u8,
            std::mem::size_of::<libc::fsid_t>(),
        );
        let identity = Id::new(tuckmill_manifest::ids::sha256_hex(fsid_bytes)).ok()?;
        Some(DiskStatus {
            free_bytes: free,
            total_bytes: total,
            volume_identity: identity,
        })
    }
}

#[cfg(not(unix))]
fn statfs(_path: &Path) -> Option<DiskStatus> {
    None
}

/// Best-effort fsync of a directory entry so a rename is durable.
pub fn fsync_dir(path: &Path) {
    if let Ok(dir) = std::fs::File::open(path) {
        let _ = dir.sync_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_root_is_application_support() {
        let layout = StoreLayout::resolve().unwrap();
        let root = layout.root().to_string_lossy().to_string();
        if std::env::var("TUCKMILL_DATA_DIR").is_err() {
            assert!(root.contains("Application Support"), "got {root}");
            assert!(root.ends_with("Tuckmill"));
        }
    }

    #[test]
    fn ensure_creates_skeleton() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        layout.ensure().unwrap();
        assert!(layout.artifacts_dir().is_dir());
        assert!(layout.tmp_dir().is_dir());
        assert!(layout.control_dir().is_dir());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(layout.root())
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o700, "store root must be user-only");
        }
    }

    #[test]
    fn disk_status_reads_real_volume() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path());
        let status = layout.disk_status(dir.path()).unwrap();
        assert!(status.total_bytes > 0);
        assert!(status.free_bytes <= status.total_bytes);
        // Identity is stable.
        let again = layout.disk_status(dir.path()).unwrap();
        assert_eq!(status.volume_identity, again.volume_identity);
    }

    #[test]
    fn ensure_space_rejects_when_short() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path());
        let err = layout
            .ensure_space_for(Bytes(u64::MAX / 2), dir.path())
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InsufficientDisk);
    }
}
