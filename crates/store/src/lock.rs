//! Advisory file locks so concurrent imports cannot corrupt each other
//! (fault-injection requirement: concurrent import).

use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::path::Path;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};

pub struct FileLock {
    _file: File,
}

impl FileLock {
    /// Take an exclusive lock, waiting for contending holders.
    pub fn exclusive(path: &Path) -> Result<Self> {
        Self::with_flags(path, libc::LOCK_EX)
    }

    /// Non-blocking variant: returns None when another process holds it.
    pub fn try_exclusive(path: &Path) -> Result<Option<Self>> {
        match Self::with_flags(path, libc::LOCK_EX | libc::LOCK_NB) {
            Ok(lock) => Ok(Some(lock)),
            Err(e) if e.code() == ErrorCode::QueueFull => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn with_flags(path: &Path, flags: libc::c_int) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(TuckmillError::from)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
            .map_err(TuckmillError::from)?;
        // SAFETY: valid fd owned by `file`; flock is advisory and released on
        // close or process exit.
        let rc = unsafe { libc::flock(file.as_raw_fd(), flags) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                return Err(TuckmillError::new(
                    ErrorCode::QueueFull,
                    format!("lock {} is held by another process", path.display()),
                ));
            }
            return Err(TuckmillError::from(err));
        }
        Ok(Self { _file: file })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusive_lock_is_reentrant_safe_across_handles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("l.lock");
        let lock = FileLock::exclusive(&path).unwrap();
        // flock is per-open-file-description: a same-process second open can
        // still be blocked, which is what we want for child processes.
        let second = FileLock::try_exclusive(&path).unwrap();
        assert!(second.is_none(), "second lock attempt must not succeed");
        drop(lock);
        let third = FileLock::try_exclusive(&path).unwrap();
        assert!(third.is_some(), "lock releases on drop");
    }
}
