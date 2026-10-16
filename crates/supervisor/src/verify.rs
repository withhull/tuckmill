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

use crate::runtime::Runtime;
use crate::worker::IsolatedWorker;

/// Scored logits as `(position, logits)` pairs; kept internal so the public
/// record never carries a full logit matrix.
type ScoredLogits = Vec<(u64, Vec<f32>)>;

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
}

/// A machine-readable equivalence record (FR-21/24 evidence).
#[derive(Debug, Clone, Serialize)]
pub struct VerifyReport {
    pub schema_version: u32,
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

        // Deterministic token IDs within the artifact's vocabulary.
        let path = std::path::Path::new(&artifact.path);
        let insp =
            tuckmill_manifest::gguf::inspect(path, &tuckmill_manifest::gguf::Limits::default())?;
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
        if vocab == 0 || options.token_count < 2 {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "verification needs a vocabulary and at least two tokens",
            ));
        }
        let tokens: Vec<u32> = (0..options.token_count)
            .map(|i| ((i * 37 + 5) % vocab) as u32)
            .collect();
        let positions: Vec<u64> = (1..=options.max_score_positions)
            .map(|i| i * 2 - 1)
            .filter(|p| *p < options.token_count)
            .collect();
        if positions.is_empty() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "no scoring positions fit inside the token sequence",
            ));
        }

        let (resident, resident_logits) = self.run_side(
            "resident",
            plan,
            artifact,
            &tokens,
            &positions,
            options,
            options.resident_cache_bytes,
            false,
        )?;
        let (streamed, streamed_logits) = self.run_side(
            "forced-streaming",
            plan,
            artifact,
            &tokens,
            &positions,
            options,
            options.streamed_cache_bytes,
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
                let denom = x.abs().max(y.abs()).max(1e-9) as f64;
                max_rel = max_rel.max(diff / denom);
            }
        }

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
        label: &str,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        tokens: &[u32],
        positions: &[u64],
        options: &VerifyOptions,
        cache_bytes: u64,
        sample_footprint: bool,
    ) -> Result<(RunSide, ScoredLogits)> {
        let mut worker = IsolatedWorker::spawn(
            &self.worker_exe,
            &self.layout.tmp_dir(),
            EngineKind::TuckCore,
        )?;
        let block_bytes = cache_bytes.clamp(64 * 1024, 4 * 1024 * 1024);
        let load_options = serde_json::json!({
            "weight_cache_bytes": cache_bytes,
            "block_bytes": block_bytes,
        });
        let note = worker.load(plan, artifact, Some(&load_options))?;
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
                max_output_tokens: options.generate_tokens,
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
            cache_bytes,
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

/// Format a report for the terminal.
pub fn format_report(report: &VerifyReport) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "equivalence: {} ({})",
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
        "  resident side: cache {}, {} positioned scores, {} committed tokens",
        tuckmill_manifest::units::Bytes(report.resident.cache_bytes),
        report.resident.scored_positions,
        report.resident.committed_tokens
    ));
    lines.push(format!(
        "  streamed side: cache {}, evictions {}, logical reads {}, {} committed",
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
        "  logits: vocab {}, exact_equal {}, max|Δ| {:.3e}, max rel {:.3e}",
        report.logits.vocab,
        report.logits.exact_equal,
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
