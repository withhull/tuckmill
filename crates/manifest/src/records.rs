//! Persistent records: ModelArtifact, DeviceProfile, Session, BenchmarkRun.
//!
//! Every record carries a schema version and references immutable IDs. All
//! measurement fields are explicitly labeled; a projected value must never be
//! serialized into a measured field.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ids::Id;
use crate::units::Bytes;

/// How much trust a displayed number has earned (PRD §5): every
/// configuration is *estimated*, *locally calibrated*, or *certified*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    Estimated,
    LocallyCalibrated,
    Certified,
}

impl Confidence {
    pub fn as_str(self) -> &'static str {
        match self {
            Confidence::Estimated => "estimated",
            Confidence::LocallyCalibrated => "locally-calibrated",
            Confidence::Certified => "certified",
        }
    }
}

/// Performance class from the PRD §5 table. These are classification targets,
/// not measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatencyClass {
    Interactive,
    PatientLocal,
    BatchLocal,
    Infeasible,
}

impl LatencyClass {
    pub fn as_str(self) -> &'static str {
        match self {
            LatencyClass::Interactive => "interactive",
            LatencyClass::PatientLocal => "patient-local",
            LatencyClass::BatchLocal => "batch-local",
            LatencyClass::Infeasible => "infeasible",
        }
    }
}

/// Byte accounting for a model's tensor payload.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TensorStats {
    pub total_tensors: usize,
    pub total_bytes: u64,
    /// Bytes by tensor role (token_embedding, attention, ...).
    pub by_class: BTreeMap<String, u64>,
    /// Bytes by storage type name (F16, Q4_K, ...).
    pub by_type: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    Gguf,
    Safetensors,
    Packed { format_version: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactSource {
    /// Where the file came from: a local path or a pinned HTTPS origin.
    pub uri: String,
    /// Pinned revision identifier for remote sources.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// License reference recorded at import time. Importing is not a license
    /// review (PRD §4.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
}

/// The immutable artifact record (PRD §18 model artifact contract).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub schema_version: u32,
    pub artifact_id: Id,
    pub created_unix_ms: u64,
    pub format: ModelFormat,
    pub source: ArtifactSource,
    pub file_name: String,
    pub file_bytes: u64,
    /// Content hash of the whole file (chunked; resumable).
    pub content_hash: Id,
    /// Hash of the tensor index only.
    pub tensor_index_hash: Id,
    /// Hash of the metadata section.
    pub metadata_hash: Id,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_hash: Option<Id>,
    /// Architecture id from the artifact (may be unknown for safetensors).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch_id: Option<String>,
    /// Model display name from the artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Context length advertised by the artifact (not an admission promise).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advertised_context_length: Option<u64>,
    /// Resolved geometry for planning; absent when no adapter matched yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<ModelGeometry>,
    pub tensor_stats: TensorStats,
    /// Dominant quantization profile label derived from the tensor types,
    /// for display only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_label: Option<String>,
    /// Path of the packed artifact if one exists for this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed: Option<PackedInfo>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackedInfo {
    pub format_version: u32,
    pub content_hash: Id,
    pub file_bytes: u64,
    pub segments: usize,
}

/// Resolved architecture geometry used by the planner's memory math. Filled
/// at inspect/import time via the adapter; absent when no adapter resolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelGeometry {
    pub layers: u64,
    pub head_count: u64,
    pub kv_head_count: u64,
    pub key_length: u64,
    pub value_length: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_used_per_token: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityStatusKind {
    Available,
    Unavailable,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityStatus {
    pub status: CapabilityStatusKind,
    pub detail: String,
}

impl CapabilityStatus {
    pub fn available(detail: impl Into<String>) -> Self {
        Self {
            status: CapabilityStatusKind::Available,
            detail: detail.into(),
        }
    }
    pub fn unavailable(detail: impl Into<String>) -> Self {
        Self {
            status: CapabilityStatusKind::Unavailable,
            detail: detail.into(),
        }
    }
    pub fn unknown(detail: impl Into<String>) -> Self {
        Self {
            status: CapabilityStatusKind::Unknown,
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuProfile {
    /// GPU family estimate (e.g. "apple9"). Static mapping unless the bridge
    /// verified it through a Metal capability query.
    pub family_estimate: String,
    /// How the estimate was obtained. Static values are explicitly labeled.
    pub family_source: String,
    /// Recommended max working set, when the Metal bridge is available.
    /// Advisory for GPU allocation performance; not available system memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_working_set_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageProfile {
    /// Coarse class: "internal-ssd", "external-ssd", "unknown".
    pub path_class: String,
    /// Robust volume identity; a reused mount path must not inherit trust.
    pub volume_identity: Id,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_battery: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low_power_mode: Option<bool>,
}

/// Hardware profile sampled by the doctor/profiler (PRD §18.3). Unavailable
/// signals are recorded as unavailable, never guessed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceProfile {
    pub schema_version: u32,
    pub device_id: Id,
    pub sampled_unix_ms: u64,
    pub chip: String,
    pub cpu_cores_total: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_cores_performance: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_cores_efficiency: Option<u32>,
    pub memory_bytes: u64,
    pub os_name: String,
    pub os_version: String,
    pub os_build: String,
    pub gpu: GpuProfile,
    pub storage: StorageProfile,
    pub power: PowerProfile,
    /// Thermal state at profile time ("nominal", "fair", "serious",
    /// "critical", or "unknown").
    pub thermal_start: String,
    /// Memory pressure at profile time ("normal", "warning", "critical", or
    /// "unknown").
    pub pressure: String,
    pub capabilities: BTreeMap<String, CapabilityStatus>,
}

/// Session record (minimum persistent records, PRD §15). Sessions never
/// persist speculative state as committed state (PRD §14).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionRecord {
    pub schema_version: u32,
    pub session_id: Id,
    pub created_unix_ms: u64,
    pub updated_unix_ms: u64,
    pub artifact_id: Id,
    pub plan_id: Id,
    /// Committed tokens only.
    pub committed_tokens: u64,
    /// Last acknowledged committed position for crash recovery. Clients may
    /// restart from here; the system never fabricates continuation state.
    pub last_committed_position: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineIdentity {
    pub engine: String,
    /// Whether this engine is certified for the named artifact/profile.
    pub certified: bool,
    pub backend_abi_version: u32,
    pub worker_protocol_version: u16,
    /// Kernel ids actually selected (empty for the reference path).
    #[serde(default)]
    pub kernels: Vec<String>,
    /// Build identity of the binary that produced the run.
    pub build_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkloadSpec {
    pub suite: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub repetitions: u32,
    pub prompt_set: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheState {
    ColdDeclared,
    Warm,
    PrefixHit,
    CacheUncontrolled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptSample {
    pub prompt_id: String,
    pub ttft_ms: f64,
    pub committed_tokens: u32,
    pub total_ms: f64,
    pub gap_p95_ms: f64,
    pub gap_max_ms: f64,
    pub footprint_peak_bytes: u64,
    pub logical_read_bytes: u64,
    /// Decode-only throughput: committed tokens after the first, divided by
    /// the time from the first committed token to the last. `None` when the
    /// run produced fewer than two tokens. Never compare this with another
    /// engine's end-to-end rate without saying which one is meant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub throughput_decode_tps: Option<f64>,
    /// Worker-side (backend-measured) request durations. The gap to the
    /// supervisor-side values is queueing and transport, never compute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_total_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_prefill_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_decode_ms: Option<f64>,
    /// Resolved kernel implementation actually used, when the backend
    /// selects kernels ("scalar"/"neon").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_impl: Option<String>,
    /// Resolved weight-storage mode actually used ("mapped"/"owned").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_mode: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummaryStats {
    pub ttft_median_ms: f64,
    pub ttft_p95_ms: f64,
    /// End-to-end output throughput (committed tokens / total request time,
    /// including the wait before the first token).
    pub throughput_median_tps: f64,
    /// Median decode-only throughput over samples that produced one. See
    /// `PromptSample::throughput_decode_tps`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub throughput_decode_median_tps: Option<f64>,
    pub gap_p95_ms: f64,
    pub completion_rate: f64,
    pub footprint_peak_bytes: u64,
}

/// A reproducible benchmark report (FR-14). Claims use these records, not
/// screenshots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub schema_version: u32,
    pub run_id: Id,
    pub started_unix_ms: u64,
    pub device_id: Id,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<Id>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<Id>,
    pub engine: EngineIdentity,
    pub workload: WorkloadSpec,
    pub cache_state: CacheState,
    pub samples: Vec<PromptSample>,
    pub summary: SummaryStats,
    pub performance_class: LatencyClass,
    pub confidence: Confidence,
    /// Explicit slow-mode acceptance, required for batch-local labels.
    pub slow_mode_accepted: bool,
    /// Honest caveats; a report without its caveats is incomplete evidence.
    pub caveats: Vec<String>,
}

/// Helpers shared by supervisor and CLI for footprint reporting.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FootprintSample {
    pub sampled_unix_ms: u64,
    /// Process physical footprint (bytes), measured, not estimated.
    pub physical_footprint_bytes: u64,
    /// Managed live reservations (bytes) as tracked by the ledger.
    pub managed_live_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compressor_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swap_delta_bytes: Option<i64>,
}

/// Reservation categories for the admission ledger (PRD §9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReserveClass {
    Weights,
    KvState,
    Workspaces,
    IoBuffers,
    ExpertCache,
    DraftState,
    PackingTemps,
    Other,
}

impl ReserveClass {
    pub fn as_str(self) -> &'static str {
        match self {
            ReserveClass::Weights => "weights",
            ReserveClass::KvState => "kv_state",
            ReserveClass::Workspaces => "workspaces",
            ReserveClass::IoBuffers => "io_buffers",
            ReserveClass::ExpertCache => "expert_cache",
            ReserveClass::DraftState => "draft_state",
            ReserveClass::PackingTemps => "packing_temps",
            ReserveClass::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReservationBreakdown {
    pub weights: Bytes,
    pub kv_state: Bytes,
    pub workspaces: Bytes,
    pub io_buffers: Bytes,
    pub expert_cache: Bytes,
    pub draft_state: Bytes,
    pub other: Bytes,
}

impl ReservationBreakdown {
    pub fn total(&self) -> Bytes {
        Bytes(
            self.weights
                .bytes()
                .saturating_add(self.kv_state.bytes())
                .saturating_add(self.workspaces.bytes())
                .saturating_add(self.io_buffers.bytes())
                .saturating_add(self.expert_cache.bytes())
                .saturating_add(self.draft_state.bytes())
                .saturating_add(self.other.bytes()),
        )
    }

    pub fn add(&mut self, class: ReserveClass, bytes: Bytes) {
        let slot = match class {
            ReserveClass::Weights => &mut self.weights,
            ReserveClass::KvState => &mut self.kv_state,
            ReserveClass::Workspaces => &mut self.workspaces,
            ReserveClass::IoBuffers => &mut self.io_buffers,
            ReserveClass::ExpertCache => &mut self.expert_cache,
            ReserveClass::DraftState => &mut self.draft_state,
            ReserveClass::PackingTemps | ReserveClass::Other => &mut self.other,
        };
        *slot = Bytes(slot.bytes().saturating_add(bytes.bytes()));
    }

    pub fn as_map(&self) -> BTreeMap<String, u64> {
        let mut m = BTreeMap::new();
        m.insert(
            ReserveClass::Weights.as_str().to_string(),
            self.weights.bytes(),
        );
        m.insert(
            ReserveClass::KvState.as_str().to_string(),
            self.kv_state.bytes(),
        );
        m.insert(
            ReserveClass::Workspaces.as_str().to_string(),
            self.workspaces.bytes(),
        );
        m.insert(
            ReserveClass::IoBuffers.as_str().to_string(),
            self.io_buffers.bytes(),
        );
        m.insert(
            ReserveClass::ExpertCache.as_str().to_string(),
            self.expert_cache.bytes(),
        );
        m.insert(
            ReserveClass::DraftState.as_str().to_string(),
            self.draft_state.bytes(),
        );
        m.insert(ReserveClass::Other.as_str().to_string(), self.other.bytes());
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservation_breakdown_totals() {
        let mut b = ReservationBreakdown::default();
        b.add(ReserveClass::Weights, Bytes(1000));
        b.add(ReserveClass::KvState, Bytes(500));
        b.add(ReserveClass::Other, Bytes(250));
        assert_eq!(b.total().bytes(), 1750);
        assert_eq!(b.weights.bytes(), 1000);
        assert_eq!(b.as_map()["weights"], 1000);
    }

    #[test]
    fn confidence_labels_serialize_snake_case() {
        let s = serde_json::to_string(&Confidence::LocallyCalibrated).unwrap();
        assert_eq!(s, "\"locally_calibrated\"");
        let s = serde_json::to_string(&LatencyClass::PatientLocal).unwrap();
        assert_eq!(s, "\"patient_local\"");
    }
}
