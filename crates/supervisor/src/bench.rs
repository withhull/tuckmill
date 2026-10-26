//! Benchmark harness: standard-v1 workload, machine-readable evidence
//! (FR-14, PRD §20).
//!
//! Labels are explicit: cache state, engine identity, certification, and
//! caveats. Draft tokens are never reported as final throughput; the
//! reference engine's synthetic output is disclosed in the record.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ids::{hash_of, now_unix_ms, Id};
use tuckmill_manifest::plan::ExecutionPlan;
use tuckmill_manifest::records::{
    BenchmarkRun, CacheState, Confidence, LatencyClass, PromptSample, SummaryStats, WorkloadSpec,
};
use tuckmill_planner::calibration::CalibrationEntry;

use crate::runtime::{RunEvent, Runtime};

pub const STANDARD_SUITE: &str = "standard-v1";

/// Frozen workload definition, embedded from benchmarks/standard-v1.json.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteDefinition {
    pub schema_version: u32,
    pub suite: String,
    pub notes: Vec<String>,
    /// The standard measurement shape: 512 input / 256 output.
    pub default_input_tokens: u32,
    pub default_output_tokens: u32,
    pub default_repetitions: u32,
    pub input_lengths_short: Vec<u32>,
    pub output_lengths_short: Vec<u32>,
    /// Calibration prompts are separate from held-out qualification prompts.
    #[serde(default)]
    pub calibration_prompt_ids: Vec<String>,
    #[serde(default)]
    pub qualification_prompt_set: Option<String>,
    #[serde(default)]
    pub qualification_prompt_topics: Vec<String>,
}

pub fn load_suite() -> Result<SuiteDefinition> {
    let src = include_str!("../../../benchmarks/standard-v1.json");
    serde_json::from_str(src).map_err(|e| {
        tuckmill_manifest::error::json_error(
            e,
            ErrorCode::InvalidConfiguration,
            "standard-v1 suite definition",
        )
    })
}

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub repetitions: u32,
    pub cache_state: CacheState,
    pub suite: String,
    /// Generate every requested output token on every engine: the model's
    /// EOS does not end generation (decode-throughput workloads for
    /// instruct models that end early on the frozen prompts).
    pub ignore_eos: bool,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            input_tokens: 512,
            output_tokens: 256,
            repetitions: 3,
            cache_state: CacheState::CacheUncontrolled,
            suite: STANDARD_SUITE.to_string(),
            ignore_eos: false,
        }
    }
}

/// Deterministic frozen prompt set: exact token IDs derived from the prompt
/// identifier, stable across runs and machines (no private prompts are
/// collected; PRD §20). When `vocab` is known (real backends), ids are
/// bounded into the artifact's vocabulary so the same frozen prompts are
/// valid for every backend.
fn prompt_tokens_for(prompt_id: &str, input_tokens: u32, vocab: Option<u64>) -> Vec<u32> {
    let seed = tuckmill_manifest::ids::sha256_hex(prompt_id.as_bytes());
    let mut out = Vec::with_capacity(input_tokens as usize);
    let mut acc: u64 = u64::from_str_radix(&seed[..16], 16).unwrap_or(0x9e37);
    for _ in 0..input_tokens {
        // xorshift64*: deterministic, dependency-free.
        acc ^= acc >> 12;
        acc ^= acc << 25;
        acc ^= acc >> 27;
        let raw = (acc.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32;
        out.push(match vocab {
            Some(v) if v > 0 => raw % v as u32,
            _ => raw,
        });
    }
    out
}

/// Vocabulary length of the artifact, when it exposes one (bounded header
/// read; used to keep frozen prompts valid for every backend).
fn artifact_vocab(artifact: &tuckmill_backend::ArtifactRef) -> Option<u64> {
    let insp = tuckmill_manifest::gguf::inspect(
        std::path::Path::new(&artifact.path),
        &tuckmill_manifest::gguf::Limits::default(),
    )
    .ok()?;
    insp.vocabulary_len()
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let idx = ((values.len() as f64 - 1.0) * p).round() as usize;
    values[idx]
}

/// Execute the standard workload against an admitted plan and produce a
/// machine-readable BenchmarkRun plus the calibration entry.
pub fn run_standard_bench(
    runtime: &Arc<Runtime>,
    plan: &ExecutionPlan,
    artifact: &tuckmill_backend::ArtifactRef,
    options: &BenchOptions,
) -> Result<(BenchmarkRun, CalibrationEntry)> {
    runtime.check_plan_fresh(plan)?;
    runtime.plan_admitted(plan)?;
    let suite = load_suite()?;

    let started = now_unix_ms();
    let vocab = artifact_vocab(artifact);
    let mut samples: Vec<PromptSample> = Vec::new();
    let mut ttfts: Vec<f64> = Vec::new();
    let mut tps_values: Vec<f64> = Vec::new();
    let mut decode_values: Vec<f64> = Vec::new();
    let mut all_gap_p95: Vec<f64> = Vec::new();
    let mut footprint_peak: u64 = 0;
    let mut resolved_identity: Option<tuckmill_manifest::records::EngineIdentity> = None;
    let mut resolved_batch: Option<u32> = None;

    for rep in 0..options.repetitions.max(1) {
        let prompt_id = format!(
            "std-{}-{}-r{rep}",
            options.input_tokens, options.output_tokens
        );
        let prompt = prompt_tokens_for(&prompt_id, options.input_tokens, vocab);

        // Baseline counters are captured while the worker is idle (before
        // dispatch). The measurement starts before the request is submitted,
        // and events are consumed immediately: nothing that can block on the
        // busy worker ever sits between dispatch and the timer or the drain.
        let logical_reads_before = runtime
            .backend_stats()
            .map(|s| s.logical_read_bytes)
            .unwrap_or(0);

        let ticket = runtime.submit()?;
        let t0 = Instant::now();
        let run = runtime.start_run(
            &ticket,
            plan.clone(),
            artifact.clone(),
            crate::runtime::RunRequest {
                prompt_tokens: prompt,
                max_output_tokens: options.output_tokens as u64,
                stop: Vec::new(),
                seed: rep as u64,
                sampling: tuckmill_backend::SamplingPolicy::default(),
                ignore_eos: options.ignore_eos,
            },
        )?;

        // Visible-stream cadence (user experience): first/last visible
        // chunk and the gaps between them.
        let mut first_visible_at: Option<Instant> = None;
        let mut last_visible_at: Option<Instant> = None;
        let mut visible_chunks: u64 = 0;
        let mut gaps: Vec<f64> = Vec::new();
        // Authoritative model-token counts from the worker's Done frame.
        let mut committed: u64 = 0;
        let mut generated: u64 = 0;
        let mut failure: Option<String> = None;
        let mut worker_timing: Option<tuckmill_backend::RequestTiming> = None;

        // Footprint sampling alongside event consumption.
        let sample_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let sampler = {
            let runtime = runtime.clone();
            let stop = sample_stop.clone();
            let peak = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let peak2 = peak.clone();
            let handle = std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if let Some(pid) = runtime.worker_pid() {
                        if let Some(fp) = runtime.signals.footprint_of(pid as i32) {
                            peak2.fetch_max(fp, Ordering::Relaxed);
                        }
                    } else if let Some(fp) = runtime.signals.footprint_of(std::process::id() as i32)
                    {
                        peak2.fetch_max(fp, Ordering::Relaxed);
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            });
            (handle, peak)
        };

        loop {
            match run.events.recv() {
                Ok(RunEvent::State(_)) => {}
                Ok(RunEvent::Token { text, .. }) => {
                    let now = Instant::now();
                    if first_visible_at.is_none() {
                        first_visible_at = Some(now);
                    } else if let Some(last) = last_visible_at {
                        gaps.push((now - last).as_secs_f64() * 1000.0);
                    }
                    last_visible_at = Some(now);
                    visible_chunks += 1;
                    let _ = text;
                    // Text chunks are the visible stream; token counts below
                    // come from the worker's Done frame, never from chunk
                    // events (a chunk is not a token: buffering, Unicode
                    // assembly, and stop handling can all change the count).
                }
                Ok(RunEvent::Done {
                    committed: c,
                    generated: g,
                    timing,
                    ..
                }) => {
                    committed = c;
                    generated = if g == 0 { c } else { g };
                    worker_timing = timing;
                    break;
                }
                Ok(RunEvent::Error { code, message, .. }) => {
                    failure = Some(format!("{code}: {message}"));
                    break;
                }
                Err(_) => {
                    failure = Some("event stream closed".into());
                    break;
                }
            }
        }
        if let Some(id) = run.identity.get() {
            resolved_identity = Some(id.clone());
        }
        if let Some(batch) = runtime.backend_stats().and_then(|s| s.prefill_batch) {
            resolved_batch = Some(batch);
        }
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let final_stats = runtime.backend_stats();
        let logical_read_bytes = final_stats
            .as_ref()
            .map(|s| s.logical_read_bytes.saturating_sub(logical_reads_before))
            .unwrap_or(0);
        let kernel_impl = final_stats.as_ref().and_then(|s| s.kernel_impl.clone());
        let storage_mode = final_stats
            .as_ref()
            .and_then(|s| s.storage.as_ref().map(|st| st.mode.clone()));
        sample_stop.store(true, Ordering::Relaxed);
        let (sampler_handle, peak) = sampler;
        let _ = sampler_handle.join();
        let run_peak = peak.load(Ordering::Relaxed);
        footprint_peak = footprint_peak.max(run_peak);

        // TTFT as the user experiences it: time to the first visible chunk.
        let ttft = first_visible_at
            .map(|t| t.duration_since(t0).as_secs_f64() * 1000.0)
            .unwrap_or(total_ms);
        // End-to-end throughput over model tokens (the worker's own
        // generated-token count), including the wait before the first token.
        let tps = if total_ms > 0.0 {
            generated as f64 / (total_ms / 1000.0)
        } else {
            0.0
        };
        // Decode-only throughput over model-token production: post-first
        // generated tokens over the worker-measured first-to-last sampled
        // span. Buffering and visibility never enter this number. Absent
        // when the backend reports no production timing.
        let decode_tps = match worker_timing.as_ref().and_then(|t| t.decode_ms) {
            Some(ms) if ms > 0.0 && generated >= 2 => Some((generated - 1) as f64 / (ms / 1000.0)),
            _ => None,
        };
        let mut gaps_for_p95 = gaps.clone();
        let gap_p95 = percentile(&mut gaps_for_p95, 0.95);
        let gap_max = gaps_for_p95.last().copied().unwrap_or(0.0);

        if failure.is_none() {
            ttfts.push(ttft);
            tps_values.push(tps);
            if let Some(d) = decode_tps {
                decode_values.push(d);
            }
            all_gap_p95.push(gap_p95);
        }
        let mut notes = Vec::new();
        if let Some(f) = &failure {
            notes.push(format!("failed: {f}"));
        }
        if runtime.engine_mode_label_str() == "in-process-reference" {
            notes.push("in-process engine mode (benchmark aid)".into());
        }
        samples.push(PromptSample {
            prompt_id,
            ttft_ms: ttft,
            committed_tokens: committed as u32,
            generated_tokens: generated as u32,
            visible_chunks: visible_chunks as u32,
            total_ms,
            gap_p95_ms: gap_p95,
            gap_max_ms: gap_max,
            footprint_peak_bytes: run_peak,
            logical_read_bytes,
            throughput_decode_tps: decode_tps,
            worker_total_ms: worker_timing.as_ref().map(|t| t.total_ms),
            worker_prefill_ms: worker_timing.as_ref().and_then(|t| t.prefill_ms),
            worker_decode_ms: worker_timing.as_ref().and_then(|t| t.decode_ms),
            kernel_impl,
            storage_mode,
            notes,
        });
    }

    let completed = ttfts.len() as f64;
    let completion_rate = completed / samples.len().max(1) as f64;
    let ttft_median = percentile(&mut ttfts.clone(), 0.5);
    let ttft_p95 = percentile(&mut ttfts.clone(), 0.95);
    let tps_median = percentile(&mut tps_values.clone(), 0.5);
    let decode_median = if decode_values.is_empty() {
        None
    } else {
        Some(percentile(&mut decode_values.clone(), 0.5))
    };
    let gap_p95 = percentile(&mut all_gap_p95.clone(), 0.95);

    let performance_class = if completed == 0.0 {
        LatencyClass::Infeasible
    } else if ttft_median <= 15_000.0 && tps_median >= 5.0 && gap_p95 <= 2_000.0 {
        LatencyClass::Interactive
    } else if ttft_median <= 90_000.0 && tps_median >= 1.0 && gap_p95 <= 10_000.0 {
        LatencyClass::PatientLocal
    } else {
        LatencyClass::BatchLocal
    };

    // Lock-free: the resolved identity comes from the last run's published
    // snapshot (set by the executor after load); falls back to the declared
    // identity, never waits on a busy worker.
    let engine = resolved_identity.unwrap_or_else(|| runtime.declared_identity());
    let summary = SummaryStats {
        ttft_median_ms: ttft_median,
        ttft_p95_ms: ttft_p95,
        throughput_median_tps: tps_median,
        throughput_decode_median_tps: decode_median,
        gap_p95_ms: gap_p95,
        completion_rate,
        footprint_peak_bytes: footprint_peak,
    };
    let workload = WorkloadSpec {
        suite: options.suite.clone(),
        input_tokens: options.input_tokens,
        output_tokens: options.output_tokens,
        repetitions: options.repetitions,
        prompt_set: "frozen-synthetic-ids".into(),
        ignore_eos: options.ignore_eos,
    };
    let synthetic = tuckmill_backend::is_synthetic_engine(&engine.engine);
    let mut caveats = vec![
        format!("suite {} frozen: {}", suite.suite, suite.notes.join("; ")),
        "all throughput figures count model tokens from the worker's Done frame: throughput_median_tps is end-to-end (generated tokens / total request time, including TTFT) and throughput_decode_median_tps is (generated - 1) over the worker's first-to-last sampled-token span; ttft_ms, visible_chunks, and the gap percentiles describe the visible stream (user experience), not token production".into(),
        if synthetic {
            "reference test engine: synthetic output; model weights were not executed".into()
        } else {
            format!(
                "experimental backend {}: uncertified kernels; real weights executed, performance and quality unmeasured",
                engine.engine
            )
        },
        if synthetic {
            "logical read traffic is not physical SSD traffic; the reference engine reads no weights"
                .into()
        } else {
            "logical read traffic is measured weight bytes read via positioned reads; not physical SSD counters"
                .into()
        },
        "footprint sampled from proc_pid_rusage at ~50ms cadence".into(),
        "cold-start and warm-cache are separate labels; this run's cache state is declared explicitly".into(),
    ];
    if options.cache_state == CacheState::CacheUncontrolled {
        caveats
            .push("cache state was not controlled or established; treated as uncontrolled".into());
    }
    if performance_class == LatencyClass::BatchLocal {
        caveats.push("batch-local classification requires explicit slow-mode acceptance".into());
    }
    if options.ignore_eos {
        caveats.push(
            "ignore-EOS workload: the model's end-of-sequence token did not end generation, so every engine generated the full output count; decode throughput is comparable, output text past EOS is not meaningful"
                .into(),
        );
    }

    let run = BenchmarkRun {
        // 2: decode-only throughput fields. 3: worker-side component timings
        //    (worker_total_ms / worker_prefill_ms / worker_decode_ms); all
        //    optional, older records parse. 4: token counts are the worker's
        //    authoritative Done counts (committed = visible tokens,
        //    generated = sampled tokens) and decode throughput uses
        //    production timing; text-chunk counts are reported separately as
        //    visible_chunks. Values from v3-and-earlier records counted
        //    streamed chunks, which match the v4 figures on plain ASCII
        //    output but not with buffering, Unicode assembly, or stops.
        //    5: prefill_batch_requested / prefill_batch_effective recorded
        //    separately (requested is the option; effective is what the
        //    worker acknowledged after plan-bounded batch selection).
        schema_version: 5,
        run_id: Id::new("0".repeat(64)).expect("zero id"),
        started_unix_ms: started,
        device_id: plan.device_id.clone(),
        artifact_id: Some(plan.artifact_id.clone()),
        plan_id: Some(plan.plan_id.clone()),
        engine,
        workload,
        cache_state: options.cache_state,
        samples,
        summary: summary.clone(),
        performance_class,
        confidence: Confidence::Estimated,
        // Prefill chunking is fixed by the engine (ADR-011); nothing is
        // requested, the effective chunk is recorded.
        prefill_batch_requested: None,
        prefill_batch_effective: resolved_batch,
        slow_mode_accepted: false,
        caveats,
    };
    let mut run = run;
    run.run_id = hash_of(&serde_json::json!({
        "started": run.started_unix_ms,
        "plan": run.plan_id.as_ref().map(|i| i.as_str()),
        "suite": run.workload.suite,
    }))?;

    // Persist the record.
    let dir = runtime.layout.bench_dir();
    std::fs::create_dir_all(&dir).map_err(TuckmillError::from)?;
    let path = dir.join(format!("{}.json", run.run_id.as_str()));
    let bytes = serde_json::to_vec_pretty(&run).map_err(|e| {
        tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "benchmark record")
    })?;
    std::fs::write(&path, bytes).map_err(TuckmillError::from)?;

    let entry = CalibrationEntry {
        device_id: plan.device_id.clone(),
        artifact_id: plan.artifact_id.clone(),
        adapter_id: plan.adapter_id.clone(),
        engine: if run.engine.engine.starts_with("reference-test") {
            "reference-test".to_string()
        } else {
            run.engine.engine.clone()
        },
        backend_abi_version: tuckmill_manifest::BACKEND_ABI_VERSION,
        worker_protocol_version: tuckmill_manifest::WORKER_PROTOCOL_VERSION,
        workload_suite: options.suite.clone(),
        context_total: plan.context.total_context,
        max_output_tokens: plan.context.max_output_tokens,
        cache_state: options.cache_state,
        tps_p10: percentile(&mut tps_values.clone(), 0.1),
        tps_median,
        tps_p90: percentile(&mut tps_values.clone(), 0.9),
        ttft_p10_ms: percentile(&mut ttfts.clone(), 0.1),
        ttft_median_ms: ttft_median,
        ttft_p90_ms: percentile(&mut ttfts.clone(), 0.9),
        gap_p95_ms: gap_p95,
        footprint_peak_bytes: footprint_peak,
        samples: options.repetitions,
        recorded_unix_ms: now_unix_ms(),
        counts_toward_confidence: false,
    };

    Ok((run, entry))
}

/// Convenience wrapper: record the calibration entry too (it will not
/// promote confidence; see CalibrationStore::record).
pub fn run_and_record(
    runtime: &Arc<Runtime>,
    plan: &ExecutionPlan,
    artifact: &tuckmill_backend::ArtifactRef,
    options: &BenchOptions,
) -> Result<BenchmarkRun> {
    let (run, entry) = run_standard_bench(runtime, plan, artifact, options)?;
    {
        let mut store = runtime.calibration_store();
        store.record(entry);
        let path = runtime.layout.calibration_path();
        store.save(&path)?;
    }
    Ok(run)
}

/// Summary pretty-print used by the CLI.
pub fn format_summary(run: &BenchmarkRun) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "suite: {} ({} input / {} output tokens, {} repetitions, prompts: {})",
        run.workload.suite,
        run.workload.input_tokens,
        run.workload.output_tokens,
        run.workload.repetitions,
        run.workload.prompt_set
    ));
    lines.push(format!(
        "engine: {} (certified: {}, abi {})",
        run.engine.engine, run.engine.certified, run.engine.backend_abi_version
    ));
    lines.push(format!(
        "ttft: median {:.0}ms p95 {:.0}ms | throughput: median {:.2} tok/s end-to-end{} | gap p95 {:.0}ms",
        run.summary.ttft_median_ms,
        run.summary.ttft_p95_ms,
        run.summary.throughput_median_tps,
        match run.summary.throughput_decode_median_tps {
            Some(d) => format!(" ({d:.2} tok/s decode-only)"),
            None => String::new(),
        },
        run.summary.gap_p95_ms
    ));
    lines.push(format!(
        "completion rate: {:.0}% | footprint peak: {}",
        run.summary.completion_rate * 100.0,
        tuckmill_manifest::units::Bytes(run.summary.footprint_peak_bytes)
    ));
    lines.push(format!(
        "class: {} ({}), slow-mode accepted: {}",
        run.performance_class.as_str(),
        run.confidence.as_str(),
        run.slow_mode_accepted
    ));
    for c in &run.caveats {
        lines.push(format!("caveat: {c}"));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuckmill_manifest::units::Bytes;

    #[test]
    fn suite_definition_is_consistent() {
        let suite = load_suite().unwrap();
        assert_eq!(suite.default_input_tokens, 512);
        assert_eq!(suite.default_output_tokens, 256);
        assert!(suite.notes.iter().any(|n| n.contains("frozen")));
    }

    #[test]
    fn prompt_ids_are_deterministic_and_parameterized() {
        let a = prompt_tokens_for("std-512-256-r0", 512, None);
        let b = prompt_tokens_for("std-512-256-r0", 512, None);
        assert_eq!(a, b);
        assert_eq!(a.len(), 512);
        let c = prompt_tokens_for("std-512-256-r1", 512, None);
        assert_ne!(a, c, "different repetitions use different fixed prompts");
        // Vocab-bounded ids stay deterministic and inside the vocabulary.
        let v = prompt_tokens_for("std-512-256-r0", 512, Some(512));
        assert_eq!(v, prompt_tokens_for("std-512-256-r0", 512, Some(512)));
        assert!(v.iter().all(|id| *id < 512));
    }

    #[test]
    fn percentile_math() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&mut v, 0.5), 3.0);
        assert_eq!(percentile(&mut v, 1.0), 5.0);
        assert_eq!(percentile(&mut [], 0.5), 0.0);
    }

    fn bench_setup() -> (tempfile::TempDir, Arc<Runtime>) {
        use crate::runtime::{EngineMode, Runtime, RuntimeOptions};
        use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};
        use tuckmill_manifest::units::Bytes;
        use tuckmill_store::StoreLayout;

        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        let src = dir.path().join("m.gguf");
        write_synthetic(&src, &SyntheticConfig::tiny_qwen3()).unwrap();
        let registry = tuckmill_manifest::adapter::AdapterRegistry::builtin().unwrap();
        tuckmill_store::import::import_local(
            &layout,
            &registry,
            &src,
            &tuckmill_store::import::ImportOptions {
                alias: Some("local".into()),
                ..Default::default()
            },
        )
        .unwrap();
        let signals = Arc::new(crate::pressure::MockSignalSource::new());
        let runtime = Runtime::open(RuntimeOptions {
            data_dir: Some(layout.root().to_path_buf()),
            engine_mode: EngineMode::InProcess,
            engine_kind: tuckmill_backend::EngineKind::ReferenceTest,
            worker_exe: None,
            queue_capacity: 2,
            execution: crate::options::ExecutionOptions::default(),
            default_budget: Some(Bytes::parse("2GiB").unwrap()),
            default_context: 2048,
            default_max_output: 256,
            policy: tuckmill_planner::PlannerPolicy::default(),
            signal_source: Some(signals),
            governor_poll: Duration::from_millis(30),
            governor_config: crate::governor::GovernorConfig {
                stability_interval: Duration::from_millis(150),
                min_dwell: Duration::from_millis(20),
            },
            monitor: true,
            allow_test_engine: true,
        })
        .unwrap();
        (dir, runtime)
    }

    /// The delayed test engine's per-token delay must appear in the measured
    /// elapsed time, regardless of thread scheduling: the timer starts before
    /// dispatch and events are drained immediately.
    #[test]
    fn delayed_engine_delay_is_included_in_the_measurement() {
        let (_dir, runtime) = bench_setup();
        // In-process placement pins a 2 ms per-token delay (runtime.rs).
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();
        let plan = runtime
            .plan_for(&artifact, &adapter, Bytes::parse("2GiB").unwrap(), 2048, 64)
            .unwrap();
        let artifact_ref = tuckmill_backend::ArtifactRef::from_record(
            &artifact,
            &runtime.layout.artifact_payload_path(&artifact.artifact_id),
        );
        let options = BenchOptions {
            input_tokens: 8,
            output_tokens: 24,
            repetitions: 2,
            ..Default::default()
        };
        let (run, _) = run_standard_bench(&runtime, &plan, &artifact_ref, &options).unwrap();
        assert_eq!(run.samples.len(), 2, "warm repeated runs complete");
        for sample in &run.samples {
            assert_eq!(sample.committed_tokens, 24);
            // 24 tokens at the 2 ms/token engine delay: the measurement must
            // include at least the engine-side production time.
            assert!(
                sample.total_ms >= 24.0,
                "measured total {} ms must include the engine delay",
                sample.total_ms
            );
            assert!(
                sample.ttft_ms >= 0.0 && sample.ttft_ms <= sample.total_ms,
                "ttft {} within total {}",
                sample.ttft_ms,
                sample.total_ms
            );
            // Worker-side timing is reported by this engine and can never
            // exceed the supervisor-side measurement beyond clock granularity.
            let wt = sample
                .worker_total_ms
                .expect("reference engine reports worker timing");
            assert!(
                wt <= sample.total_ms + 1.0,
                "worker total {wt} ms must not exceed supervisor total {} ms",
                sample.total_ms
            );
            assert!(
                wt >= 24.0,
                "worker-side timing {wt} ms must include the delay"
            );
        }
        runtime.shutdown();
    }

    /// A generation producing far more events than the 256-capacity channel
    /// must complete: the consumer drains continuously, so the executor is
    /// never blocked behind a statement holding the worker lock.
    #[test]
    fn long_generation_drains_events_without_stalling() {
        let (_dir, runtime) = bench_setup();
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();
        let plan = runtime
            .plan_for(
                &artifact,
                &adapter,
                Bytes::parse("2GiB").unwrap(),
                2048,
                600,
            )
            .unwrap();
        let artifact_ref = tuckmill_backend::ArtifactRef::from_record(
            &artifact,
            &runtime.layout.artifact_payload_path(&artifact.artifact_id),
        );
        let options = BenchOptions {
            input_tokens: 8,
            output_tokens: 600,
            repetitions: 1,
            ..Default::default()
        };
        let (run, _) = run_standard_bench(&runtime, &plan, &artifact_ref, &options).unwrap();
        assert_eq!(run.samples[0].committed_tokens, 600);
        runtime.shutdown();
    }
}
