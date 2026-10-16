//! Privacy-preserving local logs (PRD §16).
//!
//! Default logs contain timings, error codes, artifact identities, and
//! resource measurements — never prompt text, generated text, credentials, or
//! document contents. Rotation: 50MiB total across kept files, 7-day
//! retention. External sharing is out of scope; nothing here transmits.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use serde_json::json;

pub const MAX_TOTAL_BYTES: u64 = 50 * 1024 * 1024;
pub const RETENTION: Duration = Duration::from_secs(7 * 24 * 3600);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

impl Level {
    fn as_str(self) -> &'static str {
        match self {
            Level::Info => "info",
            Level::Warn => "warn",
            Level::Error => "error",
        }
    }
}

pub struct Logger {
    dir: PathBuf,
    inner: Mutex<LoggerInner>,
}

struct LoggerInner {
    file: Option<File>,
    current_bytes: u64,
}

impl Logger {
    pub fn new(dir: &Path) -> Self {
        let logger = Self {
            dir: dir.to_path_buf(),
            inner: Mutex::new(LoggerInner {
                file: None,
                current_bytes: 0,
            }),
        };
        logger.rotate_and_cleanup();
        logger
    }

    fn log_path(&self) -> PathBuf {
        self.dir.join("tuckmill.log")
    }

    fn rotate_and_cleanup(&self) {
        let _ = std::fs::create_dir_all(&self.dir);
        // 7-day retention and the 50MiB total cap.
        let now = SystemTime::now();
        let mut entries: Vec<(PathBuf, u64, SystemTime)> = Vec::new();
        if let Ok(read) = std::fs::read_dir(&self.dir) {
            for e in read.flatten() {
                let path = e.path();
                if let Ok(meta) = e.metadata() {
                    let mtime = meta.modified().unwrap_or(now);
                    if now.duration_since(mtime).unwrap_or_default() > RETENTION {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                    entries.push((path, meta.len(), mtime));
                }
            }
        }
        // Oldest first.
        entries.sort_by_key(|(_, _, m)| *m);
        let mut total: u64 = entries.iter().map(|(_, len, _)| *len).sum();
        for (path, len, _) in entries {
            if total <= MAX_TOTAL_BYTES {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(len);
            }
        }
    }

    /// Append one JSON line. Fields must never contain prompt/completion
    /// content; this module does not accept free-form message strings for
    /// that reason beyond the structured `code` label.
    pub fn event(&self, level: Level, code: &str, fields: BTreeMap<String, serde_json::Value>) {
        let mut inner = self.inner.lock().expect("logger lock");
        if inner.file.is_none() {
            let _ = std::fs::create_dir_all(&self.dir);
            match OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.log_path())
            {
                Ok(f) => {
                    inner.current_bytes = f.metadata().map(|m| m.len()).unwrap_or(0);
                    inner.file = Some(f);
                }
                Err(_) => return, // Logging never fails a request.
            }
        }
        let mut obj = serde_json::Map::new();
        obj.insert("ts_ms".into(), json!(tuckmill_manifest::ids::now_unix_ms()));
        obj.insert("level".into(), json!(level.as_str()));
        obj.insert("code".into(), json!(code));
        for (k, v) in fields {
            obj.insert(k, v);
        }
        let line = match serde_json::to_string(&obj) {
            Ok(s) => s,
            Err(_) => return,
        };
        if inner.current_bytes + line.len() as u64 + 1 > MAX_TOTAL_BYTES {
            // Rotate: rename current, start fresh.
            drop(inner.file.take());
            let rotated = self.dir.join(format!(
                "tuckmill-{}.log",
                tuckmill_manifest::ids::now_unix_ms()
            ));
            let _ = std::fs::rename(self.log_path(), rotated);
            if let Ok(f) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.log_path())
            {
                inner.file = Some(f);
                inner.current_bytes = 0;
            }
            self.rotate_and_cleanup();
        }
        if let Some(file) = inner.file.as_mut() {
            let _ = writeln!(file, "{line}");
            inner.current_bytes += line.len() as u64 + 1;
        }
    }

    pub fn info(&self, code: &str, fields: BTreeMap<String, serde_json::Value>) {
        self.event(Level::Info, code, fields);
    }

    pub fn warn(&self, code: &str, fields: BTreeMap<String, serde_json::Value>) {
        self.event(Level::Warn, code, fields);
    }

    pub fn error(&self, code: &str, fields: BTreeMap<String, serde_json::Value>) {
        self.event(Level::Error, code, fields);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_json_lines_without_content_fields() {
        let dir = tempfile::tempdir().unwrap();
        let logger = Logger::new(dir.path());
        let mut fields = BTreeMap::new();
        fields.insert("plan_id".into(), json!("abc"));
        fields.insert("committed".into(), json!(12));
        logger.info("REQUEST_DONE", fields);
        let content = std::fs::read_to_string(dir.path().join("tuckmill.log")).unwrap();
        assert!(content.contains("\"code\":\"REQUEST_DONE\""));
        assert!(content.contains("\"level\":\"info\""));
        assert!(!content.contains("prompt"));
    }

    #[test]
    fn retention_removes_old_files() {
        let dir = tempfile::tempdir().unwrap();
        let old = dir.path().join("tuckmill-old.log");
        std::fs::write(&old, b"old").unwrap();
        // Backdate beyond retention.
        let backdate = SystemTime::now() - Duration::from_secs(8 * 24 * 3600);
        let _ = std::process::Command::new("/usr/bin/touch")
            .arg("-t")
            .arg("202001010000.00")
            .arg(&old)
            .status();
        let _ = backdate;
        let _logger = Logger::new(dir.path());
        assert!(!old.exists(), "expired logs must be removed");
    }
}
