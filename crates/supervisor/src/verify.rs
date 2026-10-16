//! Equivalence verification: identical token IDs, identical artifact, two
//! scheduling configurations, compared teacher-forced logits.
//!
//! This is the interim form of the reference-versus-custom comparison the
//! amendment requires: until the pinned llama.cpp reference backend is
//! integrated, the harness proves that TuckCore's forced-streaming
//! configuration (tiny cache, constant eviction) produces bit-identical
//! logits and generation to its resident configuration. The same harness
//! compares against llama.cpp once that backend lands (same artifact, same
//! token IDs, registered tolerances).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;

use tuckmill_backend::{ArtifactRef, BackendStats, EngineKind};
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ids::{hash_of, now_unix_ms, Id};
use tuckmill_manifest::plan::ExecutionPlan;
use tuckmill_manifest::records::Confidence;
use tuckmill_tuckcore::forward::NumericalProfile;

use crate::runtime::Runtime;
use crate::worker::IsolatedWorker;

/// Scored logits as `(position, logits)` pairs; kept internal so the public
/// record never carries a full logit matrix.
type ScoredLogits = Vec<(u64, Vec<f32>)>;

/// Bounded parameters for the llama.cpp-reference comparison (ADR-006):
/// identical token IDs through TuckCore and the pinned llama.cpp backend,
/// teacher-forced logits compared under a registered tolerance.
#[derive(Debug, Clone)]
pub struct ReferenceCompareOptions {
    pub token_count: u64,
    pub max_score_positions: u64,
    pub generate_tokens: u64,
    pub tuckcore_cache_bytes: u64,
    /// Numerical profile for the TuckCore side. Reference comparisons default
    /// to `LlamaCpu` so both sides declare the same execution contract
    /// (activations quantized per weight dtype); `OracleF32` measures the
    /// drift envelope between the oracle and the llama.cpp CPU kernels.
    pub tuckcore_profile: NumericalProfile,
    /// Registered tolerance for LlamaRef-vs-TuckCore logits. `None` selects
    /// the registered envelope for the artifact's dominant weight dtype.
    /// Tolerances are set from measurements and tightened only with new
    /// evidence — never to make a run pass.
    pub max_abs_tolerance: Option<f64>,
}

impl Default for ReferenceCompareOptions {
    fn default() -> Self {
        Self {
            token_count: 12,
            max_score_positions: 6,
            generate_tokens: 8,
            tuckcore_cache_bytes: 32 * 1024 * 1024,
            tuckcore_profile: NumericalProfile::LlamaCpu,
            max_abs_tolerance: None,
        }
    }
}

/// Registered logit tolerances by dominant weight dtype, derived from
/// measured envelopes on Qwen3-0.6B (M3, ctx 2048, pinned llama.cpp CPU):
/// F16 max 1.95e-2 (smooth accumulation noise), Q8_0 max 3.5e-1 (dominated
/// by llama.cpp CPU 8-bit activation quantization, whose architecture-
/// specific kernel dispatch is not reproducible from reference semantics).
/// Values carry ~1.4-2.5x margin over measurement.
pub fn registered_tolerance_for(dominant_dtype: &str) -> f64 {
    match dominant_dtype {
        "F32" | "F16" | "BF16" => 5e-2,
        "Q8_0" => 5e-1,
        _ => 1e-1,
    }
}

/// Bounded verification parameters.
#[derive(Debug, Clone)]
pub struct VerifyOptions {
    pub token_count: u64,
    pub max_score_positions: u64,
    pub generate_tokens: u64,
    pub resident_cache_bytes: u64,
    pub streamed_cache_bytes: u64,
    /// Registered tolerance. Stage 0 requires exact bit equality; kernel
    /// paths get their tolerances registered before any tuning (PRD §13).
    pub max_abs_tolerance: f64,
    pub require_exact: bool,
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
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RunSide {
    pub label: String,
    pub cache_bytes: u64,
    pub scored_positions: u64,
    pub generated_text: String,
    pub committed_tokens: u64,
    pub stats: BackendStats,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint_peak_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LogitComparison {
    pub positions: Vec<u64>,
    pub vocab: u64,
    pub exact_equal: bool,
    pub max_abs_diff: f64,
    pub max_rel_diff: f64,
    /// Distribution beyond the maxima: a large tail at cancellation points
    /// and a tight body is a different finding from broad divergence.
    pub median_abs_diff: f64,
    pub p95_abs_diff: f64,
    pub p999_abs_diff: f64,
}

fn percentile_sorted(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let idx = ((values.len() as f64 - 1.0) * p).round() as usize;
    values[idx]
}

/// A machine-readable equivalence record (FR-21/24 evidence).
#[derive(Debug, Clone, Serialize)]
pub struct VerifyReport {
    pub schema_version: u32,
    /// "scheduling-equivalence" (resident vs forced-streaming TuckCore) or
    /// "reference-comparison" (llama.cpp vs TuckCore over identical ids).
    pub mode: String,
    pub run_id: Id,
    pub created_unix_ms: u64,
    pub artifact_id: Id,
    pub artifact_metadata_hash: String,
    pub engine: String,
    pub token_ids: Vec<u32>,
    pub resident: RunSide,
    pub streamed: RunSide,
    pub logits: LogitComparison,
    pub generated_texts_equal: bool,
    pub streaming_exercised: bool,
    pub confidence: Confidence,
    pub passed: bool,
    pub notes: Vec<String>,
    pub caveats: Vec<String>,
}

impl Runtime {
    /// Run the resident-vs-streamed equivalence check on this runtime's
    /// selected backend (TuckCore).
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
        let (tokens, positions, _vocab, _dtype) =
            derive_verify_inputs(artifact, options.token_count, options.max_score_positions)?;

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
            false,
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
            true,
        )?;

        // Compare logits elementwise.
        if resident.scored_positions as usize != positions.len()
            || streamed.scored_positions as usize != positions.len()
        {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!(
                    "scoring incomplete: resident {} / streamed {} of {} positions",
                    resident.scored_positions,
                    streamed.scored_positions,
                    positions.len()
                ),
            ));
        }
        let mut max_abs: f64 = 0.0;
        let mut max_rel: f64 = 0.0;
        let mut exact = true;
        let mut vocab_seen = 0u64;
        let mut diffs: Vec<f64> = Vec::new();
        for (a, b) in resident_logits.iter().zip(streamed_logits.iter()) {
            let (pos_a, la) = a;
            let (pos_b, lb) = b;
            if pos_a != pos_b || la.len() != lb.len() {
                exact = false;
                continue;
            }
            vocab_seen = la.len() as u64;
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
            let mut d = diffs.clone();
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
        if !exact && within_tolerance {
            notes.push(format!(
                "logits differ by at most {max_abs:.3e} (within the registered tolerance {:.3e})",
                options.max_abs_tolerance
            ));
        }
        let passed = within_tolerance && texts_equal && streaming_exercised;

        let caveats = vec![
            "the pinned llama.cpp reference backend is not yet integrated; this record proves scheduling/streaming equivalence (resident vs constant-eviction) of tuckcore over identical token IDs, and is the harness that will perform reference-vs-custom logit comparison once that backend lands"
                .into(),
            "fixture artifacts carry synthetic (non-trained) weights: equivalence speaks to execution paths, never model quality"
                .into(),
            format!(
                "tolerance registered: exact bit equality = {}; max_abs = {:.3e}",
                options.require_exact, options.max_abs_tolerance
            ),
        ];

        let run_id = hash_of(&serde_json::json!({
            "artifact": artifact.metadata_hash,
            "tokens": tokens,
            "positions": positions,
            "resident_cache": options.resident_cache_bytes,
            "streamed_cache": options.streamed_cache_bytes,
        }))?;
        let report = VerifyReport {
            schema_version: 1,
            mode: "scheduling-equivalence".to_string(),
            run_id: run_id.clone(),
            created_unix_ms: now_unix_ms(),
            artifact_id: plan.artifact_id.clone(),
            artifact_metadata_hash: artifact.metadata_hash.clone(),
            engine: self.engine_identity().engine,
            token_ids: tokens,
            resident,
            streamed,
            logits: LogitComparison {
                positions,
                vocab: vocab_seen,
                exact_equal: exact,
                max_abs_diff: max_abs,
                max_rel_diff: max_rel,
                median_abs_diff: median_abs,
                p95_abs_diff: p95_abs,
                p999_abs_diff: p999_abs,
            },
            generated_texts_equal: texts_equal,
            streaming_exercised,
            confidence: Confidence::Estimated,
            passed,
            notes,
            caveats,
        };

        // Persist the record.
        let dir = self.layout.records_dir().join("verify");
        std::fs::create_dir_all(&dir).map_err(TuckmillError::from)?;
        let path = dir.join(format!("{}.json", run_id.as_str()));
        let bytes = serde_json::to_vec_pretty(&report).map_err(|e| {
            tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "verify record")
        })?;
        std::fs::write(&path, bytes).map_err(TuckmillError::from)?;

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
        sample_footprint: bool,
    ) -> Result<(RunSide, ScoredLogits)> {
        let mut worker = IsolatedWorker::spawn(&self.worker_exe, &self.layout.tmp_dir(), kind)?;
        // Only TuckCore takes streaming options; llama.cpp manages its own
        // memory (mmap) and rejects unknown option keys nowhere else.
        let load_options = cache_bytes.map(|cache_bytes| {
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

        // Footprint sampling during the whole side.
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let sampler = if sample_footprint {
            let signals = self.signals.clone();
            let pid = worker.pid() as i32;
            let peak2 = peak.clone();
            let stop2 = stop.clone();
            Some(std::thread::spawn(move || {
                while !stop2.load(Ordering::Relaxed) {
                    if let Some(fp) = signals.footprint_of(pid) {
                        peak2.fetch_max(fp, Ordering::Relaxed);
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
            }))
        } else {
            None
        };

        let scored = worker.score("verify", tokens, positions)?;
        let mut logits = Vec::with_capacity(scored.len());
        for (pos, values) in &scored {
            logits.push((*pos, values.clone()));
        }

        // Generate from a bounded prefix for text equality.
        let prompt: Vec<u32> = tokens.iter().take(4.min(tokens.len())).copied().collect();
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pause = Arc::new(std::sync::atomic::AtomicBool::new(false));
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
        let footprint_peak = peak.load(Ordering::Relaxed);
        if let Some(handle) = sampler {
            stop.store(true, Ordering::Relaxed);
            let _ = handle.join();
        }
        drop(worker); // graceful shutdown

        // The full logit matrix stays out of the public record: the report
        // carries digests and maxima only.
        let side = RunSide {
            label: label.to_string(),
            cache_bytes: cache_bytes.unwrap_or(0),
            scored_positions: scored.len() as u64,
            generated_text: text,
            committed_tokens: outcome.committed,
            stats,
            footprint_peak_bytes: if sample_footprint && footprint_peak > 0 {
                Some(footprint_peak)
            } else {
                None
            },
        };
        Ok((side, logits))
    }
}

/// Deterministic verification inputs: token ids within the artifact's
/// vocabulary and odd scoring positions inside the sequence.
fn derive_verify_inputs(
    artifact: &ArtifactRef,
    token_count: u64,
    max_score_positions: u64,
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
    if vocab == 0 || token_count < 2 {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "verification needs a vocabulary and at least two tokens",
        ));
    }
    let tokens: Vec<u32> = (0..token_count)
        .map(|i| ((i * 37 + 5) % vocab) as u32)
        .collect();
    let positions: Vec<u64> = (1..=max_score_positions)
        .map(|i| i * 2 - 1)
        .filter(|p| *p < token_count)
        .collect();
    if positions.is_empty() {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "no scoring positions fit inside the token sequence",
        ));
    }
    // Dominant weight dtype by payload bytes (chooses the registered
    // tolerance when the caller did not name one).
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

impl Runtime {
    /// Reference comparison (ADR-006): identical token IDs through TuckCore
    /// and the pinned llama.cpp backend; teacher-forced logits compared under
    /// a registered tolerance. Generated-text equality is recorded but is not
    /// a pass criterion: greedy argmax can diverge on exact ties, and the
    /// numerical claim is about logits, not tie-breaking luck.
    pub fn run_reference_comparison(
        &self,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        options: &ReferenceCompareOptions,
    ) -> Result<VerifyReport> {
        self.check_plan_fresh(plan)?;
        self.plan_admitted(plan)?;
        let (tokens, positions, _vocab, dominant_dtype) =
            derive_verify_inputs(artifact, options.token_count, options.max_score_positions)?;
        let (tolerance, tolerance_source) = match options.max_abs_tolerance {
            Some(t) => (t, "explicit".to_string()),
            None => (
                registered_tolerance_for(&dominant_dtype),
                format!("registered for dominant weight dtype {dominant_dtype}"),
            ),
        };

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
            false,
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
            true,
        )?;

        if tuckcore.scored_positions as usize != positions.len()
            || llama.scored_positions as usize != positions.len()
        {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!(
                    "scoring incomplete: tuckcore {} / llama.cpp {} of {} positions",
                    tuckcore.scored_positions,
                    llama.scored_positions,
                    positions.len()
                ),
            ));
        }

        let mut max_abs: f64 = 0.0;
        let mut max_rel: f64 = 0.0;
        let mut exact = true;
        let mut vocab_seen = 0u64;
        let mut diffs: Vec<f64> = Vec::new();
        for (a, b) in tuckcore_logits.iter().zip(llamaref_logits.iter()) {
            let (pos_a, la) = a;
            let (pos_b, lb) = b;
            if pos_a != pos_b || la.len() != lb.len() {
                exact = false;
                continue;
            }
            vocab_seen = la.len() as u64;
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
        let within_tolerance = max_abs <= tolerance;
        let texts_equal = tuckcore.generated_text == llama.generated_text;
        let mut notes = Vec::new();
        if exact {
            notes.push("logits are bit-identical across backends".into());
        } else {
            notes.push(format!(
                "logits differ by at most {max_abs:.3e} (tolerance {tolerance:.3e}, {tolerance_source}): drift sources are kernel/accumulation ordering and llama.cpp CPU activation quantization",
            ));
        }
        if !texts_equal {
            notes.push(
                "greedy generated text diverged (argmax tie or amplified difference); recorded, not a pass criterion"
                    .into(),
            );
        }
        let passed = within_tolerance;

        let caveats = vec![
            format!(
                "reference backend: pinned llama.cpp {} via tm shim, CPU execution (n_gpu_layers=0)",
                tuckmill_llamaref::LLAMA_REVISION
            ),
            "TuckCore is the scalar f32 oracle: comparison establishes the numerical profile envelope for this artifact and context, not performance"
                .into(),
            "equivalence speaks to execution paths: weight quality and provenance belong to the artifact itself (synthetic fixtures prove paths, never model quality)"
                .into(),
            format!("tolerance {tolerance:.3e} selected ({tolerance_source}); registered before tuning, tightened only with new evidence"),
        ];

        let run_id = hash_of(&serde_json::json!({
            "artifact": artifact.metadata_hash,
            "tokens": tokens,
            "positions": positions,
            "mode": "reference-comparison",
            "llama_revision": tuckmill_llamaref::LLAMA_REVISION,
        }))?;
        let report = VerifyReport {
            schema_version: 1,
            mode: "reference-comparison".to_string(),
            run_id: run_id.clone(),
            created_unix_ms: now_unix_ms(),
            artifact_id: plan.artifact_id.clone(),
            artifact_metadata_hash: artifact.metadata_hash.clone(),
            engine: "tuckcore vs llama.cpp".to_string(),
            token_ids: tokens,
            resident: tuckcore,
            streamed: llama,
            logits: LogitComparison {
                positions,
                vocab: vocab_seen,
                exact_equal: exact,
                max_abs_diff: max_abs,
                max_rel_diff: max_rel,
                median_abs_diff: median_abs,
                p95_abs_diff: p95_abs,
                p999_abs_diff: p999_abs,
            },
            generated_texts_equal: texts_equal,
            streaming_exercised: false,
            confidence: Confidence::Estimated,
            passed,
            notes,
            caveats,
        };

        let dir = self.layout.records_dir().join("verify");
        std::fs::create_dir_all(&dir).map_err(TuckmillError::from)?;
        let path = dir.join(format!("{}.json", run_id.as_str()));
        let bytes = serde_json::to_vec_pretty(&report).map_err(|e| {
            tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "verify record")
        })?;
        std::fs::write(&path, bytes).map_err(TuckmillError::from)?;
        Ok(report)
    }
}

/// Format a report for the terminal.
pub fn format_report(report: &VerifyReport) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "equivalence [{}]: {} ({})",
        report.mode,
        if report.passed { "PASS" } else { "FAIL" },
        report.engine
    ));
    lines.push(format!(
        "  artifact: {} (metadata hash {}…)",
        report.artifact_id,
        &report.artifact_metadata_hash[..12]
    ));
    lines.push(format!(
        "  tokens: {} exact ids; positions: {:?}",
        report.token_ids.len(),
        report.logits.positions
    ));
    lines.push(format!(
        "  {} side: cache {}, {} positioned scores, {} committed tokens",
        report.resident.label,
        tuckmill_manifest::units::Bytes(report.resident.cache_bytes),
        report.resident.scored_positions,
        report.resident.committed_tokens
    ));
    lines.push(format!(
        "  {} side: cache {}, evictions {}, logical reads {}, {} committed",
        report.streamed.label,
        tuckmill_manifest::units::Bytes(report.streamed.cache_bytes),
        report.streamed.stats.evictions,
        tuckmill_manifest::units::Bytes(report.streamed.stats.logical_read_bytes),
        report.streamed.committed_tokens
    ));
    if let Some(fp) = report.streamed.footprint_peak_bytes {
        lines.push(format!(
            "  streamed footprint peak: {}",
            tuckmill_manifest::units::Bytes(fp)
        ));
    }
    lines.push(format!(
        "  logits: vocab {}, exact_equal {}, |Δ| median {:.3e} p95 {:.3e} p99.9 {:.3e} max {:.3e} (max rel {:.3e})",
        report.logits.vocab,
        report.logits.exact_equal,
        report.logits.median_abs_diff,
        report.logits.p95_abs_diff,
        report.logits.p999_abs_diff,
        report.logits.max_abs_diff,
        report.logits.max_rel_diff
    ));
    lines.push(format!(
        "  generated texts equal: {}",
        report.generated_texts_equal
    ));
    for n in &report.notes {
        lines.push(format!("  note: {n}"));
    }
    for c in &report.caveats {
        lines.push(format!("  caveat: {c}"));
    }
    lines.join("\n")
}
