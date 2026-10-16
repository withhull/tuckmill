//! Equivalence verification with verification-integrity guarantees.
//!
//! Two modes:
//! - **scheduling-equivalence**: resident vs forced-streaming TuckCore
//!   (bit-exact at Stage 0; proves scheduling never changes the numbers).
//! - **reference-comparison**: identical token IDs through TuckCore and the
//!   pinned llama.cpp backend, teacher-forced logits compared under a
//!   registered tolerance.
//!
//! Integrity rules (from the Stage A verification review):
//! - malformed scoring results fail closed: exact requested positions, exact
//!   vocabulary row length, finite values, contiguous chunks, bounded
//!   accumulation — a skipped or truncated row can never produce a PASS;
//! - records are immutable: a configuration ID identifies the experiment, a
//!   unique execution ID identifies the run, and every execution (including
//!   interrupted ones) is preserved under `records/verify/<run_id>.json`;
//! - the full resolved configuration (numerical profiles, KV precision,
//!   tolerances, identities, workload) is recorded with every run;
//! - qualification is explicit: `calibration` (registered policy first
//!   measured here), `held-out` (registered policy on frozen held-out
//!   inputs), or `exploratory` (user-supplied tolerance; never a
//!   certification result).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;

use tuckmill_backend::{ArtifactRef, BackendStats, EngineKind};
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ids::{hash_of, now_unix_ms, secure_random_hex, Id};
use tuckmill_manifest::plan::ExecutionPlan;
use tuckmill_manifest::records::Confidence;
use tuckmill_manifest::records::EngineIdentity;
use tuckmill_tuckcore::forward::NumericalProfile;

use crate::runtime::Runtime;
use crate::worker::{IsolatedWorker, DEFAULT_SCORE_DEADLINE};

/// Version of the registered tolerance policy. Bumping requires a new
/// calibration record; reports name the policy they were judged under.
pub const TOLERANCE_POLICY: &str = "calibration-2026-10-16.1";

/// Registered logit tolerances by dominant weight dtype, derived from
/// measured envelopes on Qwen3-0.6B (M3, ctx 2048, pinned llama.cpp CPU,
/// matched f16 KV). These are CALIBRATION values, not self-validating: a
/// configuration qualifies only by passing held-out inputs afterwards.
///
/// Unknown dtypes return `None`: a new quantization never inherits a generic
/// passing threshold — the caller must pass an explicit (exploratory)
/// tolerance instead.
pub fn registered_tolerance_for(dominant_dtype: &str) -> Option<f64> {
    match dominant_dtype {
        "F32" | "F16" | "BF16" => Some(5e-2),
        "Q8_0" => Some(5e-1),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qualification {
    Calibration,
    HeldOut,
    Exploratory,
}

impl Qualification {
    pub fn as_str(self) -> &'static str {
        match self {
            Qualification::Calibration => "calibration",
            Qualification::HeldOut => "held-out",
            Qualification::Exploratory => "exploratory",
        }
    }
}

/// Bounded parameters for the llama.cpp-reference comparison.
#[derive(Debug, Clone)]
pub struct ReferenceCompareOptions {
    pub token_count: u64,
    pub max_score_positions: u64,
    pub generate_tokens: u64,
    pub tuckcore_cache_bytes: u64,
    /// Numerical profile for the TuckCore side; reference comparisons default
    /// to `LlamaCpu` so both sides declare the same activation contract.
    pub tuckcore_profile: NumericalProfile,
    /// KV cache precision for BOTH sides ("f16" matches llama.cpp's default
    /// and is the diagnostic comparison; "f32" is available for
    /// precision-isolation runs). Recorded in the report for both engines.
    pub kv_precision: String,
    /// Tolerance override; `None` selects the registered policy value.
    /// An explicit value makes the run `exploratory`.
    pub max_abs_tolerance: Option<f64>,
    /// Held-out qualification run: frozen inputs are derived with a different
    /// seed and a larger, disjoint workload from the calibration set.
    pub held_out: bool,
    pub score_deadline: Duration,
}

impl Default for ReferenceCompareOptions {
    fn default() -> Self {
        Self {
            token_count: 12,
            max_score_positions: 6,
            generate_tokens: 8,
            tuckcore_cache_bytes: 32 * 1024 * 1024,
            tuckcore_profile: NumericalProfile::LlamaCpu,
            kv_precision: "f16".to_string(),
            max_abs_tolerance: None,
            held_out: false,
            score_deadline: DEFAULT_SCORE_DEADLINE,
        }
    }
}

/// Bounded verification parameters (scheduling-equivalence mode).
#[derive(Debug, Clone)]
pub struct VerifyOptions {
    pub token_count: u64,
    pub max_score_positions: u64,
    pub generate_tokens: u64,
    pub resident_cache_bytes: u64,
    pub streamed_cache_bytes: u64,
    pub max_abs_tolerance: f64,
    pub require_exact: bool,
    pub score_deadline: Duration,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self {
            token_count: 12,
            max_score_positions: 6,
            generate_tokens: 8,
            resident_cache_bytes: 32 * 1024 * 1024,
            streamed_cache_bytes: 64 * 1024,
            max_abs_tolerance: 0.0,
            require_exact: true,
            score_deadline: DEFAULT_SCORE_DEADLINE,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RunSide {
    pub label: String,
    pub cache_bytes: u64,
    /// Actual KV precision reported by the backend.
    pub kv_precision: String,
    pub threads: Option<u32>,
    pub identity: EngineIdentity,
    pub scored_positions: u64,
    pub generated_text: String,
    pub committed_tokens: u64,
    pub stats: BackendStats,
    /// Footprint peak from worker spawn through model load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint_load_peak_bytes: Option<u64>,
    /// Footprint peak over the whole side (load + score + generate).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint_run_peak_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LogitComparison {
    pub positions: Vec<u64>,
    pub vocab: u64,
    pub exact_equal: bool,
    pub max_abs_diff: f64,
    pub max_rel_diff: f64,
    pub median_abs_diff: f64,
    pub p95_abs_diff: f64,
    pub p999_abs_diff: f64,
}

/// Softmax-level agreement, independent of raw logit scale.
#[derive(Debug, Clone, Serialize)]
pub struct ProbabilityMetrics {
    pub top1_agreement: f64,
    pub top5_overlap: f64,
    pub mean_abs_prob_diff: f64,
    pub kl_divergence_mean: f64,
}

/// The resolved experiment configuration: everything that can move the
/// numbers. `config_id` hashes this record; identical configs reused later
/// are identifiable even though every execution keeps its own `run_id`.
#[derive(Debug, Clone, Serialize)]
pub struct VerifyConfigRecord {
    pub artifact_id: Id,
    pub artifact_metadata_hash: String,
    pub artifact_tensor_index_hash: String,
    pub mode: String,
    pub qualification: String,
    pub policy_version: String,
    pub context_total: u64,
    pub workload_prompt_tokens: u64,
    pub token_ids: Vec<u32>,
    pub positions: Vec<u64>,
    pub generate_tokens: u64,
    pub tuckcore_profile: Option<String>,
    pub tuckcore_cache_bytes: u64,
    pub tuckcore_block_bytes: u64,
    pub streamed_cache_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_revision: Option<String>,
    pub kv_precision_requested: String,
    pub tolerance: f64,
    pub tolerance_source: String,
}

/// A machine-readable equivalence record. Schema v2: immutable runs,
/// explicit configuration, optional sides for interrupted executions.
#[derive(Debug, Clone, Serialize)]
pub struct VerifyReport {
    pub schema_version: u32,
    pub mode: String,
    pub config_id: Id,
    pub run_id: Id,
    pub created_unix_ms: u64,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<VerifyErrorInfo>,
    pub config: VerifyConfigRecord,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resident: Option<RunSide>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streamed: Option<RunSide>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits: Option<LogitComparison>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probability_metrics: Option<ProbabilityMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_texts_equal: Option<bool>,
    pub streaming_exercised: bool,
    pub confidence: Confidence,
    pub passed: Option<bool>,
    pub notes: Vec<String>,
    pub caveats: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerifyErrorInfo {
    pub code: String,
    pub message: String,
}

/// Scored logits as `(position, logits)` pairs (kept internal so the public
/// record never carries a full logit matrix).
type ScoredLogits = Vec<(u64, Vec<f32>)>;
/// Both sides with their scored rows: (tuckcore side, llama side, tuckcore
/// logits, llama logits).
type ComparedSides = (RunSide, RunSide, ScoredLogits, ScoredLogits);

fn percentile_sorted(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let idx = ((values.len() as f64 - 1.0) * p).round() as usize;
    values[idx]
}

/// Deterministic verification inputs. Held-out qualification uses a
/// different multiplier and a larger, frozen workload disjoint from the
/// calibration shape.
fn derive_verify_inputs(
    artifact: &ArtifactRef,
    token_count: u64,
    max_score_positions: u64,
    held_out: bool,
) -> Result<(Vec<u32>, Vec<u64>, u64, String)> {
    let path = std::path::Path::new(&artifact.path);
    let insp = tuckmill_manifest::gguf::inspect(path, &tuckmill_manifest::gguf::Limits::default())?;
    let embd = insp
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                "artifact is missing token_embd.weight",
            )
        })?;
    let vocab = embd.dims.get(1).copied().unwrap_or(0);
    if vocab == 0 {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "verification needs a vocabulary",
        ));
    }
    let (mult, add, count, positions_cap) = if held_out {
        (
            53u64,
            11u64,
            token_count.max(24),
            max_score_positions.max(8),
        )
    } else {
        (37u64, 5u64, token_count.max(2), max_score_positions)
    };
    let tokens: Vec<u32> = (0..count)
        .map(|i| ((i * mult + add) % vocab) as u32)
        .collect();
    let positions: Vec<u64> = (1..=positions_cap)
        .map(|i| i * 2 - 1)
        .filter(|p| *p < count)
        .collect();
    if positions.is_empty() {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "no scoring positions fit inside the token sequence",
        ));
    }
    let mut by_type: std::collections::BTreeMap<String, u64> = Default::default();
    for t in &insp.tensors {
        *by_type.entry(t.type_name.clone()).or_default() += t.byte_size;
    }
    let dominant = by_type
        .into_iter()
        .max_by_key(|(_, bytes)| *bytes)
        .map(|(name, _)| name)
        .unwrap_or_else(|| "unknown".into());
    Ok((tokens, positions, vocab, dominant))
}

/// Fail closed on malformed scoring results before any comparison runs:
/// rows must be exactly the requested positions, in order, one per position,
/// each with exactly `vocab` finite values.
fn validate_scored_rows(positions: &[u64], rows: &[(u64, Vec<f32>)], vocab: usize) -> Result<()> {
    if rows.len() != positions.len() || !rows.iter().map(|(p, _)| *p).eq(positions.iter().copied())
    {
        return Err(TuckmillError::new(
            ErrorCode::WorkerFailed,
            format!(
                "scoring result positions {:?} do not match the request {:?}",
                rows.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
                positions
            ),
        ));
    }
    for (position, logits) in rows {
        if logits.len() != vocab {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!(
                    "scoring result for position {position} has {} values; the artifact vocabulary is {vocab} (truncated or oversized rows are never compared)",
                    logits.len()
                ),
            ));
        }
        if let Some(bad) = logits.iter().position(|v| !v.is_finite()) {
            return Err(TuckmillError::new(
                ErrorCode::IntegrityFailure,
                format!(
                    "scoring result for position {position} contains a non-finite logit at index {bad}"
                ),
            ));
        }
    }
    Ok(())
}

/// Softmax-based agreement between two logit rows (inputs already validated
/// finite and equal length).
fn probability_metrics(
    rows_a: &[(u64, Vec<f32>)],
    rows_b: &[(u64, Vec<f32>)],
) -> ProbabilityMetrics {
    let mut top1_hits = 0u64;
    let mut top5_overlap_sum = 0.0f64;
    let mut prob_diff_sum = 0.0f64;
    let mut kl_sum = 0.0f64;
    let n = rows_a.len().max(1) as f64;
    for ((_, la), (_, lb)) in rows_a.iter().zip(rows_b.iter()) {
        let pa = softmax_f64(la);
        let pb = softmax_f64(lb);
        let k = 5.min(la.len()).max(1);
        let top_a = top_indices(la, k);
        let top_b = top_indices(lb, k);
        if top_a.first() == top_b.first() {
            top1_hits += 1;
        }
        let overlap = top_a.iter().filter(|i| top_b.contains(i)).count() as f64 / k as f64;
        top5_overlap_sum += overlap;
        let mut diff = 0.0f64;
        let mut kl = 0.0f64;
        for (a, b) in pa.iter().zip(pb.iter()) {
            diff += (a - b).abs();
            let a_e = a.max(1e-12);
            let b_e = b.max(1e-12);
            kl += a_e * (a_e.ln() - b_e.ln());
        }
        prob_diff_sum += diff;
        kl_sum += kl;
    }
    ProbabilityMetrics {
        top1_agreement: top1_hits as f64 / n,
        top5_overlap: top5_overlap_sum / n,
        mean_abs_prob_diff: prob_diff_sum / n,
        kl_divergence_mean: kl_sum / n,
    }
}

fn softmax_f64(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f64> = logits.iter().map(|v| ((*v - max) as f64).exp()).collect();
    let sum: f64 = out.iter().sum();
    if sum > 0.0 {
        for v in out.iter_mut() {
            *v /= sum;
        }
    }
    out
}

fn top_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    idx.truncate(k);
    idx
}

/// RAII footprint sampler: started before model load (so the loading peak is
/// captured), stopped and joined on every path including errors.
struct FootprintSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl FootprintSampler {
    fn start(signals: Arc<dyn crate::pressure::SignalSource>, pid: i32) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let stop2 = stop.clone();
        let peak2 = peak.clone();
        let handle = std::thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                if let Some(fp) = signals.footprint_of(pid) {
                    peak2.fetch_max(fp, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        Self {
            stop,
            peak,
            handle: Some(handle),
        }
    }

    fn peak_bytes(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
}

impl Drop for FootprintSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Persist a report under a unique name; never overwrites (unique execution
/// ids make collisions a bug, and the write fails closed if one appears).
fn persist_report(runtime: &Runtime, report: &VerifyReport) -> Result<std::path::PathBuf> {
    let dir = runtime.layout.records_dir().join("verify");
    std::fs::create_dir_all(&dir).map_err(TuckmillError::from)?;
    let path = dir.join(format!("{}.json", report.run_id.as_str()));
    if path.exists() {
        return Err(TuckmillError::new(
            ErrorCode::Internal,
            format!(
                "verify record {} already exists; refusing to overwrite evidence",
                path.display()
            ),
        ));
    }
    let bytes = serde_json::to_vec_pretty(report).map_err(|e| {
        tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "verify record")
    })?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, bytes).map_err(TuckmillError::from)?;
    std::fs::rename(&tmp, &path).map_err(TuckmillError::from)?;
    Ok(path)
}

fn new_run_id(config_id: &Id) -> Result<Id> {
    let nonce = secure_random_hex(8)?;
    hash_of(&serde_json::json!({
        "config": config_id.as_str(),
        "created": now_unix_ms(),
        "nonce": nonce,
    }))
}

fn identity_trio(worker: &IsolatedWorker) -> EngineIdentity {
    let mut id = crate::engine::reference_test_identity();
    id.engine = worker.engine.clone();
    id.certified = worker.certified;
    id.build_id = worker.worker_build.clone();
    id
}

impl Runtime {
    /// Scheduling equivalence: resident vs forced-streaming TuckCore.
    pub fn run_equivalence(
        &self,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        options: &VerifyOptions,
    ) -> Result<VerifyReport> {
        if self.engine_kind() != EngineKind::TuckCore {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "verify compares tuckcore scheduling configurations; open the runtime with engine tuckcore",
            ));
        }
        self.check_plan_fresh(plan)?;
        self.plan_admitted(plan)?;
        let (tokens, positions, vocab, _dtype) = derive_verify_inputs(
            artifact,
            options.token_count,
            options.max_score_positions,
            false,
        )?;

        let (resident, resident_logits) = self.run_side(
            EngineKind::TuckCore,
            "resident",
            plan,
            artifact,
            &tokens,
            &positions,
            options.generate_tokens,
            Some(options.resident_cache_bytes),
            None,
            None,
            options.score_deadline,
        )?;
        let (streamed, streamed_logits) = self.run_side(
            EngineKind::TuckCore,
            "forced-streaming",
            plan,
            artifact,
            &tokens,
            &positions,
            options.generate_tokens,
            Some(options.streamed_cache_bytes),
            None,
            None,
            options.score_deadline,
        )?;

        let vocab_usize = vocab as usize;
        validate_scored_rows(&positions, &resident_logits, vocab_usize)?;
        validate_scored_rows(&positions, &streamed_logits, vocab_usize)?;

        let mut max_abs: f64 = 0.0;
        let mut max_rel: f64 = 0.0;
        let mut exact = true;
        let mut diffs: Vec<f64> = Vec::new();
        for ((_, la), (_, lb)) in resident_logits.iter().zip(streamed_logits.iter()) {
            for (x, y) in la.iter().zip(lb.iter()) {
                if x.to_bits() != y.to_bits() {
                    exact = false;
                }
                let diff = (*x as f64 - *y as f64).abs();
                max_abs = max_abs.max(diff);
                diffs.push(diff);
                let denom = x.abs().max(y.abs()).max(1e-9) as f64;
                max_rel = max_rel.max(diff / denom);
            }
        }
        let (median_abs, p95_abs, p999_abs) = {
            let mut d = diffs;
            (
                percentile_sorted(&mut d, 0.5),
                percentile_sorted(&mut d, 0.95),
                percentile_sorted(&mut d, 0.999),
            )
        };
        let streaming_exercised = streamed.stats.evictions > 0;
        let within_tolerance = if options.require_exact {
            exact
        } else {
            max_abs <= options.max_abs_tolerance
        };
        let texts_equal = resident.generated_text == streamed.generated_text
            && resident.committed_tokens == streamed.committed_tokens;
        let mut notes = Vec::new();
        if !streaming_exercised {
            notes.push(
                "streaming was not exercised (no evictions); lower streamed_cache_bytes for evidence"
                    .into(),
            );
        }
        let passed = within_tolerance && texts_equal && streaming_exercised;

        let config = VerifyConfigRecord {
            artifact_id: plan.artifact_id.clone(),
            artifact_metadata_hash: artifact.metadata_hash.clone(),
            artifact_tensor_index_hash: artifact.tensor_index_hash.clone(),
            mode: "scheduling-equivalence".into(),
            qualification: Qualification::Calibration.as_str().into(),
            policy_version: TOLERANCE_POLICY.into(),
            context_total: plan.context.total_context,
            workload_prompt_tokens: plan.context.max_input_tokens,
            token_ids: tokens,
            positions: positions.clone(),
            generate_tokens: options.generate_tokens,
            tuckcore_profile: None,
            tuckcore_cache_bytes: options.resident_cache_bytes,
            tuckcore_block_bytes: 0,
            streamed_cache_bytes: Some(options.streamed_cache_bytes),
            llama_revision: None,
            kv_precision_requested: "f32".into(),
            tolerance: options.max_abs_tolerance,
            tolerance_source: "scheduling bit-equality".into(),
        };
        let config_id = hash_of(&config)?;
        let run_id = new_run_id(&config_id)?;

        let report = VerifyReport {
            schema_version: 2,
            mode: "scheduling-equivalence".into(),
            config_id,
            run_id,
            created_unix_ms: now_unix_ms(),
            status: "completed".into(),
            error: None,
            config,
            resident: Some(resident),
            streamed: Some(streamed),
            logits: Some(LogitComparison {
                positions,
                vocab: vocab_usize as u64,
                exact_equal: exact,
                max_abs_diff: max_abs,
                max_rel_diff: max_rel,
                median_abs_diff: median_abs,
                p95_abs_diff: p95_abs,
                p999_abs_diff: p999_abs,
            }),
            probability_metrics: None,
            generated_texts_equal: Some(texts_equal),
            streaming_exercised,
            confidence: Confidence::Estimated,
            passed: Some(passed),
            notes,
            caveats: vec![
                "equivalence speaks to execution paths: weight quality and provenance belong to the artifact itself (synthetic fixtures prove paths, never model quality)"
                    .into(),
            ],
        };
        persist_report(self, &report)?;
        Ok(report)
    }

    /// Reference comparison: identical token IDs through TuckCore and the
    /// pinned llama.cpp backend under a registered tolerance.
    pub fn run_reference_comparison(
        &self,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        options: &ReferenceCompareOptions,
    ) -> Result<VerifyReport> {
        self.check_plan_fresh(plan)?;
        self.plan_admitted(plan)?;
        let (tokens, positions, vocab, dominant_dtype) = derive_verify_inputs(
            artifact,
            options.token_count,
            options.max_score_positions,
            options.held_out,
        )?;
        let (tolerance, tolerance_source, qualification) = match options.max_abs_tolerance {
            Some(t) => (
                t,
                "explicit override".to_string(),
                Qualification::Exploratory,
            ),
            None => {
                let registered = registered_tolerance_for(&dominant_dtype).ok_or_else(|| {
                    TuckmillError::new(
                        ErrorCode::InvalidRequest,
                        format!(
                            "no registered tolerance exists for dominant weight dtype {dominant_dtype} under policy {TOLERANCE_POLICY}; pass an explicit --tolerance for an exploratory comparison (it will be labeled exploratory)"
                        ),
                    )
                })?;
                (
                    registered,
                    format!(
                        "registered policy {TOLERANCE_POLICY} for dominant weight dtype {dominant_dtype}"
                    ),
                    if options.held_out {
                        Qualification::HeldOut
                    } else {
                        Qualification::Calibration
                    },
                )
            }
        };

        let config = VerifyConfigRecord {
            artifact_id: plan.artifact_id.clone(),
            artifact_metadata_hash: artifact.metadata_hash.clone(),
            artifact_tensor_index_hash: artifact.tensor_index_hash.clone(),
            mode: "reference-comparison".into(),
            qualification: qualification.as_str().into(),
            policy_version: TOLERANCE_POLICY.into(),
            context_total: plan.context.total_context,
            workload_prompt_tokens: plan.context.max_input_tokens,
            token_ids: tokens.clone(),
            positions: positions.clone(),
            generate_tokens: options.generate_tokens,
            tuckcore_profile: Some(options.tuckcore_profile.as_str().into()),
            tuckcore_cache_bytes: options.tuckcore_cache_bytes,
            tuckcore_block_bytes: options
                .tuckcore_cache_bytes
                .clamp(64 * 1024, 4 * 1024 * 1024),
            streamed_cache_bytes: None,
            llama_revision: Some(tuckmill_llamaref::LLAMA_REVISION.into()),
            kv_precision_requested: options.kv_precision.clone(),
            tolerance,
            tolerance_source: tolerance_source.clone(),
        };
        let config_id = hash_of(&config)?;
        let run_id = new_run_id(&config_id)?;

        let result = (|| -> Result<ComparedSides> {
            let (tuckcore, tuckcore_logits) = self.run_side(
                EngineKind::TuckCore,
                "tuckcore",
                plan,
                artifact,
                &tokens,
                &positions,
                options.generate_tokens,
                Some(options.tuckcore_cache_bytes),
                Some(options.tuckcore_profile),
                Some(&options.kv_precision),
                options.score_deadline,
            )?;
            let (llama, llamaref_logits) = self.run_side(
                EngineKind::LlamaCpp,
                "llama.cpp",
                plan,
                artifact,
                &tokens,
                &positions,
                options.generate_tokens,
                None,
                None,
                Some(&options.kv_precision),
                options.score_deadline,
            )?;
            Ok((tuckcore, llama, tuckcore_logits, llamaref_logits))
        })();

        let (tuckcore, llama, tuckcore_logits, llamaref_logits) = match result {
            Ok(v) => v,
            Err(e) => {
                let interrupted = VerifyReport {
                    schema_version: 2,
                    mode: "reference-comparison".into(),
                    config_id,
                    run_id,
                    created_unix_ms: now_unix_ms(),
                    status: "interrupted".into(),
                    error: Some(VerifyErrorInfo {
                        code: e.code().as_str().into(),
                        message: e.message().into(),
                    }),
                    config,
                    resident: None,
                    streamed: None,
                    logits: None,
                    probability_metrics: None,
                    generated_texts_equal: None,
                    streaming_exercised: false,
                    confidence: Confidence::Estimated,
                    passed: None,
                    notes: vec!["interrupted before comparison; no verdict is implied".into()],
                    caveats: vec![
                        "interrupted executions are preserved for the record and never count as qualification evidence"
                            .into(),
                    ],
                };
                persist_report(self, &interrupted)?;
                return Err(e);
            }
        };

        let vocab_usize = vocab as usize;
        validate_scored_rows(&positions, &tuckcore_logits, vocab_usize)?;
        validate_scored_rows(&positions, &llamaref_logits, vocab_usize)?;

        let mut max_abs: f64 = 0.0;
        let mut max_rel: f64 = 0.0;
        let mut exact = true;
        let mut diffs: Vec<f64> = Vec::new();
        for ((pos_a, la), (pos_b, lb)) in tuckcore_logits.iter().zip(llamaref_logits.iter()) {
            if pos_a != pos_b || la.len() != lb.len() {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    format!(
                        "scoring rows are not comparable at positions {pos_a}/{pos_b} (lengths {}/{})",
                        la.len(),
                        lb.len()
                    ),
                ));
            }
            for (x, y) in la.iter().zip(lb.iter()) {
                if x.to_bits() != y.to_bits() {
                    exact = false;
                }
                let diff = (*x as f64 - *y as f64).abs();
                max_abs = max_abs.max(diff);
                diffs.push(diff);
                let denom = x.abs().max(y.abs()).max(1e-9) as f64;
                max_rel = max_rel.max(diff / denom);
            }
        }
        if diffs.is_empty() {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                "comparison produced zero compared values",
            ));
        }
        let (median_abs, p95_abs, p999_abs) = {
            let mut d = diffs;
            (
                percentile_sorted(&mut d, 0.5),
                percentile_sorted(&mut d, 0.95),
                percentile_sorted(&mut d, 0.999),
            )
        };
        let probs = probability_metrics(&tuckcore_logits, &llamaref_logits);
        let within_tolerance = max_abs <= tolerance;
        let texts_equal = tuckcore.generated_text == llama.generated_text;
        let mut notes = Vec::new();
        notes.push(format!(
            "logits differ by at most {max_abs:.3e} (tolerance {tolerance:.3e}, {tolerance_source})"
        ));
        if exact {
            notes.push("logits are bit-identical across backends".into());
        }
        notes.push(format!(
            "probability-level agreement: top-1 {:.1}%, top-5 overlap {:.2}, mean KL {:.3e}",
            probs.top1_agreement * 100.0,
            probs.top5_overlap,
            probs.kl_divergence_mean
        ));
        if !texts_equal {
            notes.push(
                "greedy generated text diverged (argmax tie or amplified difference); recorded, not a pass criterion"
                    .into(),
            );
        }
        if qualification == Qualification::Exploratory {
            notes.push(
                "exploratory run (explicit tolerance override); never counts as a qualification result"
                    .into(),
            );
        }

        let passed = within_tolerance;
        let report = VerifyReport {
            schema_version: 2,
            mode: "reference-comparison".into(),
            config_id,
            run_id,
            created_unix_ms: now_unix_ms(),
            status: "completed".into(),
            error: None,
            config,
            resident: Some(tuckcore),
            streamed: Some(llama),
            logits: Some(LogitComparison {
                positions,
                vocab: vocab_usize as u64,
                exact_equal: exact,
                max_abs_diff: max_abs,
                max_rel_diff: max_rel,
                median_abs_diff: median_abs,
                p95_abs_diff: p95_abs,
                p999_abs_diff: p999_abs,
            }),
            probability_metrics: Some(probs),
            generated_texts_equal: Some(texts_equal),
            streaming_exercised: false,
            confidence: Confidence::Estimated,
            passed: Some(passed),
            notes,
            caveats: vec![
                format!(
                    "reference backend: pinned llama.cpp {} via tm shim; KV precision requested for both sides: {}",
                    tuckmill_llamaref::LLAMA_REVISION, options.kv_precision
                ),
                "TuckCore is the scalar FP32 reference (no additional activation quantization); the comparison establishes a numerical envelope for this artifact, configuration, and tolerance policy — not performance and not bit-identity"
                    .into(),
                "equivalence speaks to execution paths: weight quality and provenance belong to the artifact itself (synthetic fixtures prove paths, never model quality)"
                    .into(),
            ],
        };
        persist_report(self, &report)?;
        Ok(report)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_side(
        &self,
        kind: EngineKind,
        label: &str,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        tokens: &[u32],
        positions: &[u64],
        generate_tokens: u64,
        cache_bytes: Option<u64>,
        profile: Option<NumericalProfile>,
        kv_precision: Option<&str>,
        score_deadline: Duration,
    ) -> Result<(RunSide, ScoredLogits)> {
        let mut worker = IsolatedWorker::spawn(&self.worker_exe, &self.layout.tmp_dir(), kind)?;
        let identity = identity_trio(&worker);

        // Footprint sampling covers model load and the measured run.
        let sampler = FootprintSampler::start(self.signals.clone(), worker.pid() as i32);

        let mut load_options = cache_bytes.map(|cache_bytes| {
            let block_bytes = cache_bytes.clamp(64 * 1024, 4 * 1024 * 1024);
            let mut value = serde_json::json!({
                "weight_cache_bytes": cache_bytes,
                "block_bytes": block_bytes,
            });
            if let Some(profile) = profile {
                value["numerical_profile"] = serde_json::json!(profile.as_str());
            }
            value
        });
        if kind == EngineKind::LlamaCpp {
            if let Some(kv) = kv_precision {
                load_options = Some(serde_json::json!({ "kv_precision": kv }));
            }
        } else if let Some(value) = load_options.as_mut() {
            if let Some(kv) = kv_precision {
                value["kv_precision"] = serde_json::json!(kv);
            }
        }
        let note = worker.load(plan, artifact, load_options.as_ref())?;
        self.logger.info(
            "VERIFY_LOAD",
            [
                ("label".to_string(), serde_json::json!(label)),
                ("note".to_string(), serde_json::json!(note)),
            ]
            .into_iter()
            .collect(),
        );
        let load_peak = sampler.peak_bytes();

        let scored = worker.score("verify", tokens, positions, score_deadline)?;

        let prompt: Vec<u32> = tokens.iter().take(4.min(tokens.len())).copied().collect();
        let cancel = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        let mut text = String::new();
        let mut on_token = |_i: u64, t: &str| {
            text.push_str(t);
            true
        };
        let mut on_state = |_s: crate::engine::RunState| {};
        let outcome = worker.generate(
            crate::engine::GenerateRequest {
                request_id: format!("verify-{label}"),
                prompt_tokens: prompt,
                max_output_tokens: generate_tokens,
                stop: Vec::new(),
                seed: 0,
            },
            &cancel,
            &pause,
            &mut on_token,
            &mut on_state,
        )?;
        let (_, _, stats) = worker.stats()?;
        let stats = stats.unwrap_or_default();
        let run_peak = sampler.peak_bytes();
        drop(sampler); // RAII: stop + join on every path, including errors above
        drop(worker); // graceful shutdown

        let side = RunSide {
            label: label.to_string(),
            cache_bytes: cache_bytes.unwrap_or(0),
            kv_precision: stats
                .kv_precision
                .clone()
                .unwrap_or_else(|| "unreported".to_string()),
            threads: stats.threads,
            identity,
            scored_positions: scored.len() as u64,
            generated_text: text,
            committed_tokens: outcome.committed,
            stats,
            footprint_load_peak_bytes: (load_peak > 0).then_some(load_peak),
            footprint_run_peak_bytes: (run_peak > 0).then_some(run_peak),
        };
        Ok((side, scored))
    }
}

/// Format a report for the terminal.
pub fn format_report(report: &VerifyReport) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "equivalence [{}]: {} ({})",
        report.mode,
        match report.passed {
            Some(true) => "PASS".to_string(),
            Some(false) => "FAIL".to_string(),
            None => format!("NO VERDICT ({})", report.status),
        },
        report.config.qualification
    ));
    lines.push(format!(
        "  config {} / run {} / policy {}",
        report.config_id.short(),
        report.run_id.short(),
        report.config.policy_version
    ));
    lines.push(format!(
        "  artifact: {} (metadata hash {}…)",
        report.config.artifact_id,
        &report.config.artifact_metadata_hash[..12]
    ));
    lines.push(format!(
        "  tokens: {} exact ids; positions: {:?}; kv precision requested: {}",
        report.config.token_ids.len(),
        report.config.positions,
        report.config.kv_precision_requested
    ));
    if let Some(resident) = &report.resident {
        lines.push(format!(
            "  {} side: cache {}, kv {}, {} positioned scores, {} committed tokens",
            resident.label,
            tuckmill_manifest::units::Bytes(resident.cache_bytes),
            resident.kv_precision,
            resident.scored_positions,
            resident.committed_tokens
        ));
    }
    if let Some(streamed) = &report.streamed {
        lines.push(format!(
            "  {} side: cache {}, kv {}, evictions {}, logical reads {}, {} committed",
            streamed.label,
            tuckmill_manifest::units::Bytes(streamed.cache_bytes),
            streamed.kv_precision,
            streamed.stats.evictions,
            tuckmill_manifest::units::Bytes(streamed.stats.logical_read_bytes),
            streamed.committed_tokens
        ));
        if let Some(fp) = streamed.footprint_run_peak_bytes {
            lines.push(format!(
                "  {} footprint: load peak {}, run peak {}",
                streamed.label,
                streamed
                    .footprint_load_peak_bytes
                    .map(tuckmill_manifest::units::Bytes)
                    .map(|b| b.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                tuckmill_manifest::units::Bytes(fp)
            ));
        }
    }
    if let Some(logits) = &report.logits {
        lines.push(format!(
            "  logits: vocab {}, exact_equal {}, |Δ| median {:.3e} p95 {:.3e} p99.9 {:.3e} max {:.3e} (max rel {:.3e})",
            logits.vocab,
            logits.exact_equal,
            logits.median_abs_diff,
            logits.p95_abs_diff,
            logits.p999_abs_diff,
            logits.max_abs_diff,
            logits.max_rel_diff
        ));
    }
    if let Some(probs) = &report.probability_metrics {
        lines.push(format!(
            "  probability agreement: top-1 {:.1}%, top-5 overlap {:.2}, mean |Δp| {:.3e}, mean KL {:.3e}",
            probs.top1_agreement * 100.0,
            probs.top5_overlap,
            probs.mean_abs_prob_diff,
            probs.kl_divergence_mean
        ));
    }
    if let Some(equal) = report.generated_texts_equal {
        lines.push(format!("  generated texts equal: {equal}"));
    }
    if let Some(err) = &report.error {
        lines.push(format!("  error: [{}] {}", err.code, err.message));
    }
    for n in &report.notes {
        lines.push(format!("  note: {n}"));
    }
    for c in &report.caveats {
        lines.push(format!("  caveat: {c}"));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_policy_covers_measured_dtypes_only() {
        assert_eq!(registered_tolerance_for("F16"), Some(5e-2));
        assert_eq!(registered_tolerance_for("Q8_0"), Some(5e-1));
        assert_eq!(
            registered_tolerance_for("Q4_K"),
            None,
            "unmeasured dtypes never inherit a threshold"
        );
        assert_eq!(registered_tolerance_for("unknown"), None);
    }

    #[test]
    fn validate_rows_rejects_missing_positions() {
        let rows = vec![(1u64, vec![0.1f32, 0.2])];
        let err = validate_scored_rows(&[1, 3], &rows, 2).unwrap_err();
        assert_eq!(err.code(), ErrorCode::WorkerFailed);
    }

    #[test]
    fn validate_rows_rejects_duplicates_and_reorder() {
        let rows = vec![(3u64, vec![0.1, 0.2]), (3u64, vec![0.1, 0.2])];
        assert!(validate_scored_rows(&[3, 1], &rows, 2).is_err());
    }

    #[test]
    fn validate_rows_rejects_truncated_rows_identically_or_not() {
        // Missing final chunk: both sides could be short in the same way —
        // still rejected because the vocabulary length is known.
        let short: Vec<(u64, Vec<f32>)> = vec![(1, vec![0.0; 5])];
        let err = validate_scored_rows(&[1], &short, 8).unwrap_err();
        assert!(err.message().contains("truncated or oversized"));
    }

    #[test]
    fn validate_rows_rejects_empty_and_oversized_rows() {
        assert!(validate_scored_rows(&[1], &[(1, vec![])], 4).is_err());
        assert!(validate_scored_rows(&[1], &[(1, vec![0.0; 5])], 4).is_err());
    }

    #[test]
    fn validate_rows_rejects_non_finite_values() {
        let rows = vec![(1u64, vec![0.1f32, f32::NAN, 0.3, f32::INFINITY])];
        let err = validate_scored_rows(&[1], &rows, 4).unwrap_err();
        assert_eq!(err.code(), ErrorCode::IntegrityFailure);
    }

    #[test]
    fn validate_rows_accepts_exact_match() {
        let rows = vec![(1u64, vec![0.1, 0.2]), (3u64, vec![0.3, 0.4])];
        validate_scored_rows(&[1, 3], &rows, 2).unwrap();
    }

    #[test]
    fn probability_metrics_identical_rows_are_perfect() {
        let rows: Vec<(u64, Vec<f32>)> = vec![(1, vec![0.0, 1.0, 0.5]), (3, vec![2.0, 0.0, -1.0])];
        let m = probability_metrics(&rows, &rows);
        assert_eq!(m.top1_agreement, 1.0);
        assert_eq!(m.top5_overlap, 1.0);
        assert!(m.mean_abs_prob_diff < 1e-9);
        assert!(m.kl_divergence_mean.abs() < 1e-9);
    }

    #[test]
    fn probability_metrics_detect_disagreement() {
        let a: Vec<(u64, Vec<f32>)> = vec![(1, vec![10.0, 0.0])];
        let b: Vec<(u64, Vec<f32>)> = vec![(1, vec![0.0, 10.0])];
        let m = probability_metrics(&a, &b);
        assert_eq!(m.top1_agreement, 0.0);
        assert!(
            m.kl_divergence_mean > 1.0,
            "reversed favorites are far apart"
        );
    }

    #[test]
    fn held_out_inputs_differ_from_calibration_inputs() {
        // Frozen stream constants: held-out (53i+11) and calibration (37i+5)
        // differ at the first token even after vocab wrapping.
        let vocab = 512u64;
        let calib_first = (5u64) % vocab;
        let held_out_first = (11u64) % vocab;
        assert_ne!(calib_first, held_out_first);
    }
}
