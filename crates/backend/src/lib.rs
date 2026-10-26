//! Tuckmill inference-backend interface (ADR-005).
//!
//! A narrow, versioned contract between the supervisor and execution
//! backends. Backends own weights, kernels, samplers, and execution buffers;
//! they never own admission, queues, cancellation policy, or lifecycle.
//!
//! Compatibility contract for every backend (PRD §4.1, §12):
//! same model artifact, same token IDs, same declared numerical profile. A
//! backend may change scheduling and cache placement, never the model.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::plan::ExecutionPlan;
use tuckmill_manifest::records::EngineIdentity;

pub mod output;
pub mod sampling;

/// Version of this interface. Bumped only with a migration note; plans and
/// worker sessions record it so incompatible combinations fail closed.
pub const BACKEND_API_VERSION: u32 = 1;

/// Stable backend identifiers (wire + display).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendId {
    /// Deterministic synthetic engine for pipeline validation. Executes no
    /// model weights; never certified.
    #[serde(rename = "reference-test")]
    ReferenceTest,
    /// Experimental custom native executor (ADR-005). Real weights, scalar
    /// reference kernels at Stage 0.
    #[serde(rename = "tuckcore")]
    TuckCore,
    /// Pinned llama.cpp via the checked C shim: the compatibility backend and
    /// numerical oracle (ADR-005/ADR-006).
    #[serde(rename = "llama.cpp")]
    LlamaCpp,
}

impl BackendId {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendId::ReferenceTest => "reference-test",
            BackendId::TuckCore => "tuckcore",
            BackendId::LlamaCpp => "llama.cpp",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "reference-test" => Some(BackendId::ReferenceTest),
            "tuckcore" => Some(BackendId::TuckCore),
            "llama.cpp" | "llama" => Some(BackendId::LlamaCpp),
            _ => None,
        }
    }
}

/// What a backend can execute. Checked before load; mismatches fail with
/// UNSUPPORTED_ARCHITECTURE plus a safe action, never a substitution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Artifact architecture ids this backend executes (lowercase).
    pub architectures: Vec<String>,
    pub moe: bool,
    pub hybrid: bool,
    pub ternary: bool,
    /// True when the backend requires a GPU bridge. Stage 0 TuckCore is
    /// scalar CPU; the flag exists so the supervisor can say so honestly.
    pub requires_gpu: bool,
    pub notes: Vec<String>,
}

impl BackendCapabilities {
    pub fn supports_arch(&self, arch: &str) -> bool {
        let arch = arch.to_ascii_lowercase();
        self.architectures.contains(&arch)
    }
}

/// Identity of the artifact a backend will execute. The backend must verify
/// these hashes against the file on disk before executing; a mismatch is
/// INTEGRITY_FAILURE, never "close enough".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub path: String,
    pub file_bytes: u64,
    pub metadata_hash: String,
    pub tensor_index_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_hash: Option<String>,
    /// Resolved architecture, when known; used for capability gating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch_id: Option<String>,
    /// Canonical chunked content digest of the payload, when the store
    /// recorded one. Engines may verify it before executing (bounded
    /// full-file verification; see backend load options).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

/// Scheduling states surfaced to users (PRD §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Warming,
    Prefilling,
    Generating,
    Throttling,
    Paused,
}

impl RunState {
    pub fn as_str(self) -> &'static str {
        match self {
            RunState::Warming => "warming",
            RunState::Prefilling => "prefilling",
            RunState::Generating => "generating",
            RunState::Throttling => "throttling",
            RunState::Paused => "paused",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Eos,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "cancelled",
            FinishReason::Eos => "eos",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub max_output_tokens: u64,
    pub stop: Vec<String>,
    pub seed: u64,
    /// Requested sampler settings. Linear/logit-bias knobs this interface
    /// does not support must be rejected by the supervisor, never dropped
    /// silently here.
    pub sampling: SamplingPolicy,
}

/// Typed sampler settings carried by a generation request (ADR-005).
///
/// `temperature <= 0.0` means deterministic greedy selection, the default.
/// A positive temperature applies softmax over the non-NaN logits scaled by
/// the temperature, then the optional nucleus (`top_p`) transform, then a
/// deterministic draw seeded by the request seed. Backends that cannot
/// implement a requested transform must refuse the request rather than
/// silently generating greedily.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SamplingPolicy {
    /// 0.0 = greedy (default).
    #[serde(default)]
    pub temperature: f32,
    /// Nucleus threshold in (0.0, 1.0]; 1.0 or absent disables top-p.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

impl Default for SamplingPolicy {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
        }
    }
}

impl SamplingPolicy {
    pub fn greedy() -> Self {
        Self::default()
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Validate the settings; unsupported values are typed rejections.
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!(
                    "temperature {} is outside the supported range 0.0..=2.0",
                    self.temperature
                ),
            ));
        }
        if let Some(p) = self.top_p {
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!("top_p {p} is outside the supported range 0.0..=1.0"),
                ));
            }
        }
        Ok(())
    }
}

/// Cooperative controls and callbacks passed through a generation.
pub struct GenControl<'a> {
    /// Checked between committed tokens.
    pub cancel: Arc<AtomicBool>,
    /// Safe-boundary pause; cleared by the supervisor after the stability
    /// window. A backend must pause after the current committed token.
    pub paused: Arc<AtomicBool>,
    /// Cache-trim epoch: the governor increments this to request that
    /// expendable caches be released. Engines that own caches check it at
    /// safe points (never releasing in-flight buffers) and apply the trim.
    pub trim_epoch: Arc<std::sync::atomic::AtomicU64>,
    /// Chance to service external input (control frames, heartbeats) while
    /// computing; backends call it during long single-step work so
    /// cancellation and liveness do not depend on the next token callback.
    pub poll: Option<&'a dyn Fn()>,
    /// Called for each committed token; returning false requests a stop.
    pub on_token: &'a mut dyn FnMut(u64, &str) -> bool,
    /// State transitions worth showing to the user.
    pub on_state: &'a mut dyn FnMut(RunState),
}

impl<'a> GenControl<'a> {
    /// Standard construction; `trim_epoch` starts fresh and can be replaced
    /// with a shared handle where a governor needs to reach it.
    pub fn new(
        cancel: Arc<AtomicBool>,
        paused: Arc<AtomicBool>,
        on_token: &'a mut dyn FnMut(u64, &str) -> bool,
        on_state: &'a mut dyn FnMut(RunState),
    ) -> Self {
        Self {
            cancel,
            paused,
            trim_epoch: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            poll: None,
            on_token,
            on_state,
        }
    }

    /// Service external input if a poll hook is installed.
    pub fn poll(&self) {
        if let Some(poll) = self.poll {
            poll();
        }
    }
}

/// Worker-side timings for one generation request, measured inside the
/// backend (not by the supervisor). Supervisor wall time must be >= these:
/// the difference is queueing and transport, which are deliberately kept
/// separate from compute.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RequestTiming {
    /// Request entry to the final outcome.
    pub total_ms: f64,
    /// Request entry to prefill completion; `None` when the request ended
    /// during prefill (for example, cancellation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_ms: Option<f64>,
    /// Request entry to the first committed token; `None` when no token was
    /// produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_token_ms: Option<f64>,
    /// First committed token to the last; `None` with fewer than two tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<f64>,
}

/// Why a generation finished, with committed-token accounting and optional
/// worker-side timing.
///
/// `committed` counts tokens with at least one byte of user-visible text;
/// tokens dropped inside a completed stop sequence, or contributing no
/// bytes, are not counted. `generated` counts every sampled token id,
/// including the ones a stop sequence or EOS removed from the visible
/// stream — the two numbers are tracked separately so a report can never
/// conflate "tokens computed" with "tokens delivered".
#[derive(Debug, Clone)]
pub struct GenOutcome {
    pub finish_reason: FinishReason,
    pub committed: u64,
    pub generated: u64,
    /// Worker-side timing of the request, when the backend measures it.
    pub timing: Option<RequestTiming>,
}

impl GenOutcome {
    pub fn new(finish_reason: FinishReason, committed: u64) -> Self {
        Self {
            finish_reason,
            committed,
            generated: committed,
            timing: None,
        }
    }

    /// Record the raw sampled-token count when it differs from `committed`.
    pub fn with_generated(mut self, generated: u64) -> Self {
        self.generated = generated;
        self
    }

    pub fn with_timing(mut self, timing: RequestTiming) -> Self {
        self.timing = Some(timing);
        self
    }
}

/// Storage accounting reported by backends that manage weight storage.
/// These numbers overlap nothing: `owned_cache_bytes` and
/// `owned_in_flight_bytes` are heap buffers, `mapped_vm_bytes` is address
/// space, `mapped_resident_bytes` is a measured page-residency sample, and
/// the process footprint is sampled separately (`BackendStats::
/// footprint_bytes`). They must never be summed into one figure.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StorageStats {
    /// Resolved mode actually in use ("mapped" = zero-copy views of one
    /// read-only artifact mapping; "owned" = copied into bounded buffers).
    pub mode: String,
    /// Owned (copied) buffer bytes currently retained in the cache.
    pub owned_cache_bytes: u64,
    /// Owned (copied) buffer bytes leased to in-flight work.
    pub owned_in_flight_bytes: u64,
    /// Address space covered by the artifact mapping (0 in owned mode).
    /// Virtual address space — not resident memory.
    pub mapped_vm_bytes: u64,
    /// Measured resident bytes within the mapping (`mincore`), when
    /// measurable; `None` in owned mode or when the call fails.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapped_resident_bytes: Option<u64>,
    /// True when the residency scan was sampled (bounded windows) rather
    /// than covering the whole mapping.
    #[serde(default)]
    pub mapped_resident_sampled: bool,
    /// Page size used for the residency scan.
    #[serde(default)]
    pub page_size_bytes: u64,
    /// Bytes handed to `madvise(MADV_DONTNEED)` on cache-entry eviction
    /// (advisory; issued advice, not freed memory).
    #[serde(default)]
    pub page_advice_bytes: u64,
}

/// Measured backend statistics. Reservations are never reported here;
/// these are observed values or explicit zeros with no claim.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BackendStats {
    /// Logical bytes read from storage for weights (not physical SSD traffic).
    pub logical_read_bytes: u64,
    pub blocks_loaded: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    /// Cache-entry evictions: entries dropped from the application cache.
    /// Not evidence that mapped pages left RAM — see `storage`.
    pub evictions: u64,
    /// Weight blocks currently leased to in-flight work (completion fences
    /// not yet signaled). Must be 0 after a drained shutdown.
    pub in_flight_leases: u64,
    /// Completed completion fences (buffer release events) since load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub buffer_releases: Option<u64>,
    /// Bytes released through completed fences since load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub released_bytes: Option<u64>,
    /// Resolved storage mode and its accounting, when the backend manages
    /// weight storage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<StorageStats>,
    /// Process footprint attributed to the backend, when measurable. This is
    /// the charged footprint metric; it is sampled independently of the
    /// storage object above and never summed with it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint_bytes: Option<u64>,
    /// Actual KV cache precision in use ("f16"/"f32"), reported by the
    /// backend — comparisons must rely on the backend's report, not on the
    /// supervisor's request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_precision: Option<String>,
    /// KV-state bytes the backend has prepared or will allocate for the
    /// admitted context, in the backend's actual storage representation.
    /// The supervisor compares this against the plan's KV reservation after
    /// every load; a worker that would allocate more than the plan reserved
    /// fails closed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_state_bytes: Option<u64>,
    /// KV bytes allocated for the most recent request's own admitted need
    /// (prompt + output allowance), which is what a warm-reused worker
    /// actually allocates — never a previously prepared larger capacity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_kv_bytes: Option<u64>,
    /// Largest weight-cache allocation this load can retain between
    /// requests, in bytes (cache budget clamped to the artifact payload,
    /// plus one block of transient overhead). The supervisor compares this
    /// against every later plan's weights + I/O reservations before reusing
    /// the worker: model identity alone is not resource admission.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retained_weight_bytes: Option<u64>,
    /// Cache trims applied at safe checkpoints since load (governor
    /// `trim-caches` commands that reached the engine and were acted on).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trims_applied: Option<u64>,
    /// Live GPU bytes held by the engine's Metal context (resident weights,
    /// workspaces); absent for CPU-only paths. Evidence that GPU
    /// allocations match the admitted plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_allocated_bytes: Option<u64>,
    /// Persistent GPU workspace bytes prepared at load (batched prefill
    /// buffers). Reuse of a warm worker must re-check this against the
    /// current plan's workspaces reservation, exactly like retained weight
    /// bytes; absent when the engine retains no GPU workspaces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retained_workspace_bytes: Option<u64>,
    /// Session state (KV planes, recurrent state) allocated once at load
    /// for the prepared capacity and held across requests. Engines that
    /// allocate KV per request report none. When present, reuse of a warm
    /// worker re-checks it against the current plan's KV reservation: a
    /// worker prepared for a larger context is never inherited by a plan
    /// that reserved less state than the worker holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retained_state_bytes: Option<u64>,
    /// Effective Metal prefill micro-batch prepared at load. Recorded so
    /// benchmarks can report requested versus effective batching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_batch: Option<u32>,
    /// Thread count in use for compute, when the backend is threaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    /// Resolved packed-kernel implementation actually selected at load
    /// ("scalar"/"neon"), when the backend selects kernels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_impl: Option<String>,
    /// Declared numerical profile in effect, when the backend has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numerical_profile: Option<String>,
    /// Optional backend profile counters (phase attribution); present only
    /// when profiling was explicitly enabled for the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<serde_json::Value>,
}

/// The versioned backend contract.
pub trait InferenceBackend: Send {
    fn backend_id(&self) -> BackendId;

    fn api_version(&self) -> u32 {
        BACKEND_API_VERSION
    }

    fn capabilities(&self) -> BackendCapabilities;

    /// Engine identity for records and responses. `certified` is false unless
    /// named tests passed for exactly this artifact/profile/device.
    fn identity(&self) -> EngineIdentity;

    /// Verify the artifact, load weights (bounded), and report a note.
    /// The note must disclose synthetic engines and experimental kernels.
    fn load(
        &mut self,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        options: Option<&serde_json::Value>,
    ) -> Result<String>;

    /// Generate committed tokens. Honors `ctl.cancel` between committed
    /// tokens and `ctl.paused` at committed-token boundaries. Never streams
    /// draft text as committed output.
    fn generate(&mut self, req: &GenerateRequest, ctl: &mut GenControl<'_>) -> Result<GenOutcome>;

    /// Teacher-forced scoring: next-token logits for `tokens[position]` under
    /// the causal prefix `tokens[..=position]`. The sink receives
    /// `(position, logits)`; returning false stops early. Returns the number
    /// of scored positions. Implementations must bound the vocabulary-logit
    /// matrix per step (no full long-context matrix in RAM).
    ///
    /// `cancelled` must be consulted between positions (and, where possible,
    /// inside long per-position compute) so a stalled or cancelled scoring
    /// request can always terminate; implementations return `CANCELLED`
    /// rather than a partial ScoreDone when it fires.
    fn score(
        &mut self,
        tokens: &[u32],
        positions: &[u64],
        cancelled: &dyn Fn() -> bool,
        sink: &mut dyn FnMut(u64, &[f32]) -> bool,
    ) -> Result<u64>;

    fn stats(&self) -> BackendStats;

    /// Shared cache-trim epoch handle. The governor increments it; engines
    /// that own expendable caches observe the change at safe checkpoints and
    /// release what can be released (never in-flight buffers). Engines
    /// without managed caches return a fresh handle and ignore it.
    fn trim_epoch(&self) -> Arc<std::sync::atomic::AtomicU64> {
        Arc::new(std::sync::atomic::AtomicU64::new(0))
    }

    /// Release buffers after completion fences. Idempotent.
    fn shutdown(&mut self);
}

impl ArtifactRef {
    /// Build a reference from a stored artifact record. The path is the
    /// store's canonical payload path; hashes are the recorded identities the
    /// backend must re-verify before executing.
    pub fn from_record(
        artifact: &tuckmill_manifest::records::ModelArtifact,
        payload_path: &std::path::Path,
    ) -> Self {
        Self {
            path: payload_path.display().to_string(),
            file_bytes: artifact.file_bytes,
            metadata_hash: artifact.metadata_hash.as_str().to_string(),
            tensor_index_hash: artifact.tensor_index_hash.as_str().to_string(),
            tokenizer_hash: artifact
                .tokenizer_hash
                .as_ref()
                .map(|h| h.as_str().to_string()),
            arch_id: artifact.arch_id.clone(),
            content_hash: Some(artifact.content_hash.as_str().to_string()),
        }
    }
}

/// Backend and placement selection for the supervisor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    ReferenceTest,
    TuckCore,
    LlamaCpp,
}

impl EngineKind {
    pub fn backend_id(self) -> BackendId {
        match self {
            EngineKind::ReferenceTest => BackendId::ReferenceTest,
            EngineKind::TuckCore => BackendId::TuckCore,
            EngineKind::LlamaCpp => BackendId::LlamaCpp,
        }
    }
}

/// True when this engine's output is synthetic and must be labeled as such.
pub fn is_synthetic_engine(engine: &str) -> bool {
    engine.starts_with("reference-test")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_ids_roundtrip_on_the_wire() {
        for id in [
            BackendId::ReferenceTest,
            BackendId::TuckCore,
            BackendId::LlamaCpp,
        ] {
            let s = serde_json::to_string(&id).unwrap();
            assert_eq!(s, format!("\"{}\"", id.as_str()));
            let back: BackendId = serde_json::from_str(&s).unwrap();
            assert_eq!(back, id);
            assert_eq!(BackendId::parse(id.as_str()), Some(id));
        }
        assert_eq!(BackendId::parse("llama"), Some(BackendId::LlamaCpp));
    }

    #[test]
    fn capabilities_match_case_insensitively() {
        let caps = BackendCapabilities {
            architectures: vec!["qwen3".into()],
            moe: false,
            hybrid: false,
            ternary: false,
            requires_gpu: false,
            notes: vec![],
        };
        assert!(caps.supports_arch("QWEN3"));
        assert!(!caps.supports_arch("qwen3moe"));
    }

    #[test]
    fn synthetic_flag_is_identity_based() {
        assert!(is_synthetic_engine("reference-test"));
        assert!(is_synthetic_engine(
            "reference-test (kernels not implemented: K01)"
        ));
        assert!(!is_synthetic_engine("tuckcore"));
        assert!(!is_synthetic_engine("native"));
    }
}
