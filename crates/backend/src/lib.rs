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

use tuckmill_manifest::error::Result;
use tuckmill_manifest::plan::ExecutionPlan;
use tuckmill_manifest::records::EngineIdentity;

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
}

impl BackendId {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendId::ReferenceTest => "reference-test",
            BackendId::TuckCore => "tuckcore",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "reference-test" => Some(BackendId::ReferenceTest),
            "tuckcore" => Some(BackendId::TuckCore),
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
}

/// Cooperative controls and callbacks passed through a generation.
pub struct GenControl<'a> {
    /// Checked between committed tokens.
    pub cancel: Arc<AtomicBool>,
    /// Safe-boundary pause; cleared by the supervisor after the stability
    /// window. A backend must pause after the current committed token.
    pub paused: Arc<AtomicBool>,
    /// Called for each committed token; returning false requests a stop.
    pub on_token: &'a mut dyn FnMut(u64, &str) -> bool,
    /// State transitions worth showing to the user.
    pub on_state: &'a mut dyn FnMut(RunState),
}

#[derive(Debug, Clone)]
pub struct GenOutcome {
    pub finish_reason: FinishReason,
    pub committed: u64,
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
    pub evictions: u64,
    /// Weight blocks currently leased to in-flight work (completion fences
    /// not yet signaled). Must be 0 after a drained shutdown.
    pub in_flight_leases: u64,
    /// Process footprint attributed to the backend, when measurable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint_bytes: Option<u64>,
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
    fn score(
        &mut self,
        tokens: &[u32],
        positions: &[u64],
        sink: &mut dyn FnMut(u64, &[f32]) -> bool,
    ) -> Result<u64>;

    fn stats(&self) -> BackendStats;

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
        }
    }
}

/// Backend and placement selection for the supervisor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    ReferenceTest,
    TuckCore,
}

impl EngineKind {
    pub fn backend_id(self) -> BackendId {
        match self {
            EngineKind::ReferenceTest => BackendId::ReferenceTest,
            EngineKind::TuckCore => BackendId::TuckCore,
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
        for id in [BackendId::ReferenceTest, BackendId::TuckCore] {
            let s = serde_json::to_string(&id).unwrap();
            assert_eq!(s, format!("\"{}\"", id.as_str()));
            let back: BackendId = serde_json::from_str(&s).unwrap();
            assert_eq!(back, id);
            assert_eq!(BackendId::parse(id.as_str()), Some(id));
        }
        assert_eq!(BackendId::parse("llama.cpp"), None);
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
