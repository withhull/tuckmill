//! Calibration store: measured evidence that upgrades confidence labels.
//!
//! A projection becomes "locally calibrated" only when a recorded measurement
//! matches the exact device, artifact, adapter, engine, ABI versions, and a
//! context at least as large as the requested one. Reference-test-engine
//! measurements are stored for diagnostics but never promote confidence, and
//! no entry can claim what it did not measure.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tuckmill_manifest::error::Result;
use tuckmill_manifest::ids::{now_unix_ms, Id};
use tuckmill_manifest::records::CacheState;

pub const CALIBRATION_SCHEMA_VERSION: u32 = 1;

/// The only engine whose measurements can promote a plan's confidence.
pub const REAL_ENGINE_NAMES: &[&str] = &["reference", "native"];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationEntry {
    pub device_id: Id,
    pub artifact_id: Id,
    pub adapter_id: String,
    pub engine: String,
    pub backend_abi_version: u32,
    pub worker_protocol_version: u16,
    pub workload_suite: String,
    pub context_total: u64,
    pub max_output_tokens: u64,
    pub cache_state: CacheState,
    pub tps_p10: f64,
    pub tps_median: f64,
    pub tps_p90: f64,
    pub ttft_p10_ms: f64,
    pub ttft_median_ms: f64,
    pub ttft_p90_ms: f64,
    pub gap_p95_ms: f64,
    pub footprint_peak_bytes: u64,
    pub samples: u32,
    pub recorded_unix_ms: u64,
    /// False for reference-test-engine measurements; such entries are
    /// retained for pipeline diagnostics only.
    pub counts_toward_confidence: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStore {
    pub schema_version: u32,
    pub entries: Vec<CalibrationEntry>,
}

impl Default for CalibrationStore {
    fn default() -> Self {
        Self {
            schema_version: CALIBRATION_SCHEMA_VERSION,
            entries: Vec::new(),
        }
    }
}

impl CalibrationStore {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = std::fs::read(path).map_err(tuckmill_manifest::error::TuckmillError::from)?;
        let store: CalibrationStore = serde_json::from_slice(&bytes).map_err(|e| {
            tuckmill_manifest::error::json_error(
                e,
                tuckmill_manifest::ErrorCode::InvalidConfiguration,
                "calibration store",
            )
        })?;
        if store.schema_version != CALIBRATION_SCHEMA_VERSION {
            return Ok(Self::default()); // Format change: start fresh, never misread.
        }
        Ok(store)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(tuckmill_manifest::error::TuckmillError::from)?;
        }
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| {
            tuckmill_manifest::error::json_error(
                e,
                tuckmill_manifest::ErrorCode::Internal,
                "calibration serialization",
            )
        })?;
        std::fs::write(&tmp, bytes).map_err(tuckmill_manifest::error::TuckmillError::from)?;
        std::fs::rename(&tmp, path).map_err(tuckmill_manifest::error::TuckmillError::from)?;
        Ok(())
    }

    /// Insert or replace the entry with the same identity key.
    pub fn record(&mut self, mut entry: CalibrationEntry) {
        entry.recorded_unix_ms = now_unix_ms();
        entry.counts_toward_confidence = REAL_ENGINE_NAMES.contains(&entry.engine.as_str());
        self.entries.retain(|e| !same_key(e, &entry));
        self.entries.push(entry);
    }

    /// Find a matching measurement for a plan. Requires exact identity and a
    /// measured context at least as large as requested (smaller contexts do
    /// not transfer upward).
    #[allow(clippy::too_many_arguments)]
    pub fn lookup(
        &self,
        device_id: &Id,
        artifact_id: &Id,
        adapter_id: &str,
        engine: &str,
        backend_abi_version: u32,
        worker_protocol_version: u16,
        workload_suite: &str,
        context_total: u64,
    ) -> Option<&CalibrationEntry> {
        self.entries.iter().find(|e| {
            e.counts_toward_confidence
                && &e.device_id == device_id
                && &e.artifact_id == artifact_id
                && e.adapter_id == adapter_id
                && e.engine == engine
                && e.backend_abi_version == backend_abi_version
                && e.worker_protocol_version == worker_protocol_version
                && e.workload_suite == workload_suite
                && e.context_total >= context_total
                && e.samples > 0
        })
    }
}

fn same_key(a: &CalibrationEntry, b: &CalibrationEntry) -> bool {
    a.device_id == b.device_id
        && a.artifact_id == b.artifact_id
        && a.adapter_id == b.adapter_id
        && a.engine == b.engine
        && a.backend_abi_version == b.backend_abi_version
        && a.worker_protocol_version == b.worker_protocol_version
        && a.workload_suite == b.workload_suite
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(engine: &str, ctx: u64) -> CalibrationEntry {
        CalibrationEntry {
            device_id: Id::new("d".repeat(64)).unwrap(),
            artifact_id: Id::new("a".repeat(64)).unwrap(),
            adapter_id: "qwen3".into(),
            engine: engine.into(),
            backend_abi_version: tuckmill_manifest::BACKEND_ABI_VERSION,
            worker_protocol_version: tuckmill_manifest::WORKER_PROTOCOL_VERSION,
            workload_suite: "standard-v1".into(),
            context_total: ctx,
            max_output_tokens: 256,
            cache_state: CacheState::Warm,
            tps_p10: 4.0,
            tps_median: 5.0,
            tps_p90: 6.0,
            ttft_p10_ms: 900.0,
            ttft_median_ms: 1100.0,
            ttft_p90_ms: 1400.0,
            gap_p95_ms: 400.0,
            footprint_peak_bytes: 1_500_000_000,
            samples: 5,
            recorded_unix_ms: 0,
            counts_toward_confidence: false,
        }
    }

    #[test]
    fn reference_test_entries_never_promote_confidence() {
        let mut store = CalibrationStore::default();
        store.record(entry("reference-test", 2048));
        assert!(!store.entries[0].counts_toward_confidence);
        let hit = store.lookup(
            &Id::new("d".repeat(64)).unwrap(),
            &Id::new("a".repeat(64)).unwrap(),
            "qwen3",
            "reference-test",
            tuckmill_manifest::BACKEND_ABI_VERSION,
            tuckmill_manifest::WORKER_PROTOCOL_VERSION,
            "standard-v1",
            2048,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn real_engine_entries_match_exactly() {
        let mut store = CalibrationStore::default();
        let mut e = entry("reference", 2048);
        e.counts_toward_confidence = false; // record() will set it correctly
        store.record(e);
        let hit = store.lookup(
            &Id::new("d".repeat(64)).unwrap(),
            &Id::new("a".repeat(64)).unwrap(),
            "qwen3",
            "reference",
            tuckmill_manifest::BACKEND_ABI_VERSION,
            tuckmill_manifest::WORKER_PROTOCOL_VERSION,
            "standard-v1",
            1024,
        );
        assert!(hit.is_some(), "measured at 2048 covers a 1024 request");

        let miss = store.lookup(
            &Id::new("d".repeat(64)).unwrap(),
            &Id::new("a".repeat(64)).unwrap(),
            "qwen3",
            "reference",
            tuckmill_manifest::BACKEND_ABI_VERSION,
            tuckmill_manifest::WORKER_PROTOCOL_VERSION,
            "standard-v1",
            4096,
        );
        assert!(
            miss.is_none(),
            "smaller measured context must not transfer upward"
        );
    }

    #[test]
    fn abi_change_invalidates_entry() {
        let mut store = CalibrationStore::default();
        store.record(entry("reference", 4096));
        let miss = store.lookup(
            &Id::new("d".repeat(64)).unwrap(),
            &Id::new("a".repeat(64)).unwrap(),
            "qwen3",
            "reference",
            tuckmill_manifest::BACKEND_ABI_VERSION + 1,
            tuckmill_manifest::WORKER_PROTOCOL_VERSION,
            "standard-v1",
            1024,
        );
        assert!(miss.is_none());
    }

    #[test]
    fn record_replaces_same_key() {
        let mut store = CalibrationStore::default();
        store.record(entry("reference", 2048));
        let mut second = entry("reference", 2048);
        second.tps_median = 9.0;
        store.record(second);
        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].tps_median, 9.0);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("calibration.json");
        let mut store = CalibrationStore::default();
        store.record(entry("reference", 2048));
        store.save(&path).unwrap();
        let loaded = CalibrationStore::load(&path).unwrap();
        assert_eq!(loaded.entries.len(), 1);
    }
}
