//! Cache inventory and garbage collection.
//!
//! GC shows reclaimable files and removes only derived, inactive data:
//! partial imports without a live lock, orphaned journals, stale download
//! scratch files, and abandoned pack temp directories. Active artifacts and
//! user-owned source files are never removed (PRD §10). `prune --dry-run`
//! reports without deleting.

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use serde::Serialize;

use tuckmill_manifest::error::Result;

use crate::aliases;
use crate::layout::StoreLayout;
use crate::lock::FileLock;

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactEntry {
    pub artifact_id: String,
    pub file_name: String,
    pub quant_label: Option<String>,
    pub arch_id: Option<String>,
    pub bytes: u64,
    pub alias: Option<String>,
    pub created_unix_ms: u64,
    pub has_pack: bool,
    pub pack_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Inventory {
    pub artifacts: Vec<ArtifactEntry>,
    pub artifact_bytes: u64,
    pub pack_bytes: u64,
    pub tmp_bytes: u64,
    pub download_bytes: u64,
    pub reclaimable_bytes: u64,
}

pub fn inventory(layout: &StoreLayout) -> Result<Inventory> {
    let manifests = aliases::list_manifests(layout)?;
    let alias_registry = aliases::AliasRegistry::load(layout).unwrap_or_default();
    let reverse_alias: std::collections::BTreeMap<&tuckmill_manifest::Id, &String> = alias_registry
        .aliases
        .iter()
        .map(|(alias, id)| (id, alias))
        .collect();

    let mut artifacts = Vec::new();
    let mut artifact_bytes = 0u64;
    let mut pack_bytes = 0u64;
    for a in &manifests {
        let payload = layout.artifact_payload_path(&a.artifact_id);
        let bytes = std::fs::metadata(&payload).map(|m| m.len()).unwrap_or(0);
        let pack_dir = layout.artifact_pack_dir(&a.artifact_id);
        let (has_pack, pb) = dir_size(&pack_dir);
        artifact_bytes += bytes;
        pack_bytes += pb;
        artifacts.push(ArtifactEntry {
            artifact_id: a.artifact_id.as_str().to_string(),
            file_name: a.file_name.clone(),
            quant_label: a.quant_label.clone(),
            arch_id: a.arch_id.clone(),
            bytes,
            alias: reverse_alias.get(&a.artifact_id).map(|s| (*s).clone()),
            created_unix_ms: a.created_unix_ms,
            has_pack,
            pack_bytes: pb,
        });
    }

    let tmp_bytes = dir_size(&layout.tmp_dir()).1;
    let download_bytes = dir_size(&layout.downloads_dir()).1;
    let reclaimable = reclaimable_bytes(layout);

    Ok(Inventory {
        artifacts,
        artifact_bytes,
        pack_bytes,
        tmp_bytes,
        download_bytes,
        reclaimable_bytes: reclaimable,
    })
}

fn dir_size(dir: &std::path::Path) -> (bool, u64) {
    let mut total = 0u64;
    let mut any = false;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            any = true;
            let path = entry.path();
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                } else if meta.is_dir() {
                    total += dir_size(&path).1;
                }
            }
        }
    }
    (any, total)
}

/// Sum of bytes that `prune` would remove under the given age thresholds.
fn reclaimable_bytes(layout: &StoreLayout) -> u64 {
    let mut total = 0u64;
    for path in prune_candidates(layout, Duration::from_secs(3600)) {
        if let Ok(meta) = std::fs::metadata(&path) {
            if meta.is_file() {
                total += meta.len();
            } else {
                total += dir_size(&path).1;
            }
        }
    }
    total
}

/// Candidate paths for pruning: partial imports older than `min_age` whose
/// lock is not held, journals whose partial is gone, stale downloads, and
/// abandoned pack temp dirs.
fn prune_candidates(layout: &StoreLayout, min_age: Duration) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let now = SystemTime::now();
    let old_enough = |p: &std::path::Path| -> bool {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| now.duration_since(t).ok())
            .map(|age| age >= min_age)
            .unwrap_or(false)
    };

    if let Ok(entries) = std::fs::read_dir(layout.tmp_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".partial") && old_enough(&path) {
                // Lock name derived the same way import derives it would need
                // the source path; instead check for any lock file matching
                // the same stem.
                let stem = name.trim_end_matches(".partial");
                let lock = layout.tmp_dir().join(format!("{stem}.lock"));
                let held = FileLock::try_exclusive(&lock)
                    .map(|maybe| maybe.is_none())
                    .unwrap_or(false);
                if !held {
                    out.push(path);
                }
            } else if name.ends_with(".journal.json") {
                let stem = name.trim_end_matches(".journal.json");
                let partial = layout.tmp_dir().join(format!("{stem}.partial"));
                if !partial.exists() {
                    out.push(path);
                }
            } else if (name.starts_with("pack-") && name.ends_with("-partial")
                || name.ends_with(".tmp"))
                && old_enough(&path)
            {
                out.push(path);
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir(layout.downloads_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if old_enough(&path) {
                out.push(path);
            }
        }
    }
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct PruneReport {
    pub dry_run: bool,
    pub removed: Vec<PrunedEntry>,
    pub bytes_reclaimed: u64,
    pub skipped_active: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrunedEntry {
    pub path: String,
    pub bytes: u64,
}

/// Prune derived, inactive data. Never touches artifact directories, their
/// manifests, aliases, or anything outside the store.
pub fn prune(layout: &StoreLayout, dry_run: bool, min_age: Duration) -> Result<PruneReport> {
    let candidates = prune_candidates(layout, min_age);
    let mut removed = Vec::new();
    let mut bytes_reclaimed = 0u64;
    let mut skipped_active = 0usize;
    for path in candidates {
        // Final safety check: never prune inside an artifact directory.
        if path.starts_with(layout.artifacts_dir()) {
            skipped_active += 1;
            continue;
        }
        let size = std::fs::metadata(&path)
            .map(|m| {
                if m.is_file() {
                    m.len()
                } else {
                    dir_size(&path).1
                }
            })
            .unwrap_or(0);
        if !dry_run {
            let result = if path.is_dir() {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
            if result.is_err() {
                skipped_active += 1;
                continue;
            }
        }
        bytes_reclaimed += size;
        removed.push(PrunedEntry {
            path: path.display().to_string(),
            bytes: size,
        });
    }
    Ok(PruneReport {
        dry_run,
        removed,
        bytes_reclaimed,
        skipped_active,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::{import_local, ImportOptions};
    use tuckmill_manifest::adapter::AdapterRegistry;
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    #[test]
    fn inventory_lists_artifacts_and_packs() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        let src = dir.path().join("m.gguf");
        std::fs::write(&src, synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3())).unwrap();
        let registry = AdapterRegistry::builtin().unwrap();
        let outcome = import_local(
            &layout,
            &registry,
            &src,
            &ImportOptions {
                alias: Some("m".into()),
                ..Default::default()
            },
        )
        .unwrap();
        crate::pack::pack_artifact(&layout, &outcome.artifact, &Default::default()).unwrap();
        let inv = inventory(&layout).unwrap();
        assert_eq!(inv.artifacts.len(), 1);
        assert_eq!(inv.artifacts[0].alias.as_deref(), Some("m"));
        assert!(inv.artifacts[0].has_pack);
        assert!(inv.artifact_bytes > 0);
        assert!(inv.pack_bytes > 0);
    }

    #[test]
    fn prune_dry_run_reports_without_deleting() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        layout.ensure().unwrap();
        let partial = layout.tmp_dir().join("import-abc.partial");
        std::fs::write(&partial, vec![0u8; 4096]).unwrap();
        // Backdate the file so it passes the age gate.
        let old = SystemTime::now() - Duration::from_secs(7200);
        set_mtime(&partial, old);

        let report = prune(&layout, true, Duration::from_secs(3600)).unwrap();
        assert!(partial.exists(), "dry-run must not delete");
        assert_eq!(report.removed.len(), 1);
        assert!(report.bytes_reclaimed >= 4096);

        let report = prune(&layout, false, Duration::from_secs(3600)).unwrap();
        assert!(!partial.exists());
        assert!(report.bytes_reclaimed >= 4096);
    }

    #[test]
    fn prune_never_touches_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        let src = dir.path().join("m.gguf");
        std::fs::write(&src, synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3())).unwrap();
        let registry = AdapterRegistry::builtin().unwrap();
        let outcome = import_local(&layout, &registry, &src, &ImportOptions::default()).unwrap();
        let report = prune(&layout, false, Duration::from_secs(0)).unwrap();
        assert!(
            outcome.installed_path.exists(),
            "artifact payload must survive prune"
        );
        assert_eq!(report.skipped_active, 0);
        assert!(report.removed.iter().all(|e| !e.path.contains("artifacts")));
    }

    fn set_mtime(path: &std::path::Path, t: SystemTime) {
        // Use `touch -t` equivalent via filetime-less approach: rewrite then
        // fudge with `touch`? Keeping dependency-free: std cannot set mtime.
        // Use /usr/bin/touch via Command (test-only).
        let secs = t.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
        let _ = std::process::Command::new("/usr/bin/touch")
            .arg("-t")
            .arg(epoch_to_touch(secs))
            .arg(path)
            .status();
    }

    fn epoch_to_touch(secs: u64) -> String {
        // YYYYMMDDhhmm.ss from epoch seconds without a calendar crate.
        let days = secs / 86400;
        let rem = secs % 86400;
        let (y, m, d) = civil_from_days(days as i64);
        format!(
            "{y:04}{m:02}{d:02}{:02}{:02}.{:02}",
            rem / 3600,
            (rem % 3600) / 60,
            rem % 60
        )
    }

    fn civil_from_days(z: i64) -> (i64, u32, u32) {
        let z = z + 719_468;
        let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        (if m <= 2 { y + 1 } else { y }, m, d)
    }
}
