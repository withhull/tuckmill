//! Reference test engine: the deterministic synthetic backend.
//!
//! It validates the full pipeline (admission, scheduling, streaming,
//! cancellation, governor integration) with *synthetic output*. It executes
//! no model weights and is never certified. Its streams are labeled with
//! `[ref:i]` tokens so no consumer can mistake them for model output, and
//! `EngineIdentity.certified` stays false.
//!
//! Shared backend types (RunState, FinishReason, GenControl, ...) live in
//! `tuckmill-backend` (ADR-005) and are re-exported here for compatibility.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::plan::ExecutionPlan;
use tuckmill_manifest::records::EngineIdentity;
use tuckmill_manifest::{BACKEND_ABI_VERSION, WORKER_PROTOCOL_VERSION};

pub use tuckmill_backend::{
    is_synthetic_engine, BackendCapabilities, BackendId, BackendStats, FinishReason, GenControl,
    GenOutcome, GenerateRequest, RunState, BACKEND_API_VERSION,
};

/// Compatibility alias: the backend interface originally named this type
/// `EngineRequest`.
pub type EngineRequest = GenerateRequest;

pub const REFERENCE_TEST_ENGINE: &str = "reference-test";

pub fn reference_test_identity() -> EngineIdentity {
    EngineIdentity {
        engine: REFERENCE_TEST_ENGINE.to_string(),
        certified: false,
        backend_abi_version: BACKEND_ABI_VERSION,
        worker_protocol_version: WORKER_PROTOCOL_VERSION,
        kernels: Vec::new(),
        build_id: build_id(),
    }
}

pub fn build_id() -> String {
    format!(
        "tuckmill {} ({})",
        env!("CARGO_PKG_VERSION"),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    )
}

/// Deterministic prompt-to-token conversion for the reference engine.
/// No vocabulary or tokenizer executes in this build; exact token IDs from a
/// real tokenizer arrive with a certified executor. Bounded to `max_tokens`.
pub fn synthetic_prompt_tokens(text: &str, max_tokens: usize) -> Vec<u32> {
    let mut h = tuckmill_manifest::ids::sha256_hex(text.as_bytes());
    let mut ids = Vec::new();
    while h.len() >= 8 && ids.len() < max_tokens {
        if let Ok(v) = u32::from_str_radix(&h[..8], 16) {
            ids.push(v);
        }
        h.drain(..8);
    }
    if ids.is_empty() {
        ids.push(1);
    }
    ids
}

/// The deterministic reference test engine.
#[derive(Debug)]
pub struct ReferenceTestEngine {
    loaded_plan: Option<Box<ExecutionPlan>>,
    /// Simulated per-token work (honest pacing so benches measure something
    /// real about the pipeline, not the nonexistent model).
    pub per_token_delay: Duration,
    /// Bytes the synthetic engine claims to have read (always 0; the engine
    /// touches no weights).
    pub logical_read_bytes: u64,
}

impl Default for ReferenceTestEngine {
    fn default() -> Self {
        Self {
            loaded_plan: None,
            per_token_delay: Duration::from_millis(2),
            logical_read_bytes: 0,
        }
    }
}

impl ReferenceTestEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn identity(&self) -> EngineIdentity {
        reference_test_identity()
    }

    /// Load a plan (inherent form used by the in-process path and tests).
    pub fn load(&mut self, plan: &ExecutionPlan) -> Result<String> {
        plan.validate()?;
        if plan.mode == tuckmill_manifest::plan::ExecutionMode::Rejected {
            return Err(TuckmillError::new(
                ErrorCode::InvalidPlan,
                "refusing to load a rejected plan",
            ));
        }
        let note = format!(
            "reference test engine loaded plan {} (mode {}, engine {}); synthetic output only — no model weights will be executed",
            plan.plan_id.short(),
            plan.mode.as_str(),
            plan.engine.engine
        );
        self.loaded_plan = Some(Box::new(plan.clone()));
        Ok(note)
    }

    pub fn loaded_plan(&self) -> Option<&ExecutionPlan> {
        self.loaded_plan.as_deref()
    }

    /// Generate. Honors the admitted context: prompt + max output must fit.
    pub fn generate(
        &mut self,
        req: &EngineRequest,
        ctl: &mut GenControl<'_>,
    ) -> Result<GenOutcome> {
        let plan = self.loaded_plan.as_ref().ok_or_else(|| {
            TuckmillError::new(ErrorCode::WorkerFailed, "engine has no loaded plan")
        })?;
        let prompt_len = req.prompt_tokens.len() as u64;
        if prompt_len == 0 {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "prompt must contain at least one token",
            ));
        }
        let total = prompt_len.saturating_add(req.max_output_tokens);
        if total > plan.context.total_context {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!(
                    "prompt {prompt_len} + max output {} exceeds admitted context {}",
                    req.max_output_tokens, plan.context.total_context
                ),
            ));
        }
        if req.max_output_tokens == 0 {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "max_output_tokens must be positive",
            ));
        }

        let t0 = Instant::now();
        let mut first_committed_at: Option<Instant> = None;
        let mut last_committed_at: Option<Instant> = None;
        let finish_with_timing = |outcome: GenOutcome,
                                  t0: Instant,
                                  prefill_ms: Option<f64>,
                                  first: Option<Instant>,
                                  last: Option<Instant>| {
            let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1000.0;
            outcome.with_timing(tuckmill_backend::RequestTiming {
                total_ms: ms(t0, Instant::now()),
                prefill_ms,
                first_token_ms: first.map(|f| ms(t0, f)),
                decode_ms: match (first, last) {
                    (Some(f), Some(l)) if l > f => Some(ms(f, l)),
                    _ => None,
                },
            })
        };

        (ctl.on_state)(RunState::Warming);
        // Simulated warm-up; visibly short and deterministic.
        pause_loop(ctl, Duration::from_millis(5))?;

        (ctl.on_state)(RunState::Prefilling);
        // Prefill pacing: fixed 1000 prompt tokens/second simulation.
        let prefill = Duration::from_millis(prompt_len.max(1));
        pause_loop(ctl, prefill.min(Duration::from_secs(5)))?;
        if ctl.cancel.load(Ordering::Relaxed) {
            return Ok(finish_with_timing(
                GenOutcome::new(FinishReason::Cancelled, 0),
                t0,
                None,
                None,
                None,
            ));
        }
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        (ctl.on_state)(RunState::Generating);
        let mut committed: u64 = 0;
        let mut generated = String::new();
        for i in 0..req.max_output_tokens {
            if ctl.cancel.load(Ordering::Relaxed) {
                return Ok(finish_with_timing(
                    GenOutcome::new(FinishReason::Cancelled, committed),
                    t0,
                    Some(prefill_ms),
                    first_committed_at,
                    last_committed_at,
                ));
            }
            // Safe-boundary pause between committed tokens.
            pause_loop(ctl, self.per_token_delay)?;

            let text = format!("[ref:{i}]");
            let now = Instant::now();
            if first_committed_at.is_none() {
                first_committed_at = Some(now);
            }
            last_committed_at = Some(now);
            generated.push_str(&text);
            committed += 1;
            if !(ctl.on_token)(i, &text) {
                return Ok(finish_with_timing(
                    GenOutcome::new(FinishReason::Stop, committed),
                    t0,
                    Some(prefill_ms),
                    first_committed_at,
                    last_committed_at,
                ));
            }
            // Stop sequences are checked on the committed text stream.
            if req
                .stop
                .iter()
                .any(|s| !s.is_empty() && generated.ends_with(s.as_str()))
            {
                return Ok(finish_with_timing(
                    GenOutcome::new(FinishReason::Stop, committed),
                    t0,
                    Some(prefill_ms),
                    first_committed_at,
                    last_committed_at,
                ));
            }
        }
        Ok(finish_with_timing(
            GenOutcome::new(FinishReason::Length, committed),
            t0,
            Some(prefill_ms),
            first_committed_at,
            last_committed_at,
        ))
    }
}

impl tuckmill_backend::InferenceBackend for ReferenceTestEngine {
    fn backend_id(&self) -> BackendId {
        BackendId::ReferenceTest
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            // Empty architecture list: this backend accepts any plan because
            // it validates the pipeline, not any architecture's execution.
            architectures: Vec::new(),
            moe: false,
            hybrid: false,
            ternary: false,
            requires_gpu: false,
            notes: vec![
                "pipeline validation with synthetic output; model weights are never executed"
                    .to_string(),
            ],
        }
    }

    fn identity(&self) -> EngineIdentity {
        reference_test_identity()
    }

    fn load(
        &mut self,
        plan: &ExecutionPlan,
        _artifact: &tuckmill_backend::ArtifactRef,
        options: Option<&serde_json::Value>,
    ) -> Result<String> {
        // Benchmark/test aid only: an explicit per-token production delay so
        // measured latency can be proven to include worker-side pacing.
        if let Some(value) = options.and_then(|o| o.get("per_token_delay_ms")) {
            let ms = value.as_u64().ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    "per_token_delay_ms must be a non-negative integer",
                )
            })?;
            self.per_token_delay = Duration::from_millis(ms);
        }
        self.load(plan)
    }

    fn generate(&mut self, req: &GenerateRequest, ctl: &mut GenControl<'_>) -> Result<GenOutcome> {
        self.generate(req, ctl)
    }

    fn score(
        &mut self,
        _tokens: &[u32],
        _positions: &[u64],
        _cancelled: &dyn Fn() -> bool,
        _sink: &mut dyn FnMut(u64, &[f32]) -> bool,
    ) -> Result<u64> {
        Err(TuckmillError::new(
            ErrorCode::Unavailable,
            "the reference test engine produces no logits; teacher-forced scoring requires a real backend (for example tuckcore)",
        ))
    }

    fn stats(&self) -> BackendStats {
        BackendStats {
            logical_read_bytes: self.logical_read_bytes,
            ..Default::default()
        }
    }

    fn shutdown(&mut self) {}
}

/// Sleep in small slices while honoring pause and cancel.
fn pause_loop(ctl: &mut GenControl<'_>, mut remaining: Duration) -> Result<()> {
    let slice = Duration::from_millis(2);
    let mut paused_reported = false;
    while remaining > Duration::ZERO {
        ctl.poll();
        if ctl.cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        if ctl.paused.load(Ordering::Relaxed) {
            if !paused_reported {
                (ctl.on_state)(RunState::Paused);
                paused_reported = true;
            }
            std::thread::sleep(slice);
            continue;
        }
        if paused_reported {
            (ctl.on_state)(RunState::Generating);
            paused_reported = false;
        }
        let step = remaining.min(slice);
        std::thread::sleep(step);
        remaining -= step;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use tuckmill_manifest::ids::Id;
    use tuckmill_manifest::plan::{ExecutionMode, ExecutionPlan};

    fn admitted_plan() -> ExecutionPlan {
        let mut p = ExecutionPlan::new_skeleton(
            Id::new("a".repeat(64)).unwrap(),
            Id::new("b".repeat(64)).unwrap(),
            "qwen3".into(),
        );
        p.mode = ExecutionMode::Resident;
        p.context = tuckmill_manifest::plan::PlanContext::new(256, 64);
        p.plan_id = p.compute_id().unwrap();
        p
    }

    fn request(max_output: u64) -> EngineRequest {
        EngineRequest {
            request_id: "req_test".into(),
            prompt_tokens: (0..8).collect(),
            max_output_tokens: max_output,
            stop: Vec::new(),
            seed: 7,
        }
    }

    fn run(
        engine: &mut ReferenceTestEngine,
        req: &EngineRequest,
    ) -> (GenOutcome, String, Vec<RunState>) {
        let cancel = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let mut text = String::new();
        let mut states = Vec::new();
        let mut on_token = |_i: u64, t: &str| {
            text.push_str(t);
            true
        };
        let mut on_state = |s: RunState| states.push(s);
        let mut ctl = GenControl::new(cancel.clone(), paused.clone(), &mut on_token, &mut on_state);
        let outcome = engine.generate(req, &mut ctl).unwrap();
        (outcome, text, states)
    }

    #[test]
    fn generates_exact_committed_count() {
        let mut engine = ReferenceTestEngine::new();
        engine.per_token_delay = Duration::ZERO;
        engine.load(&admitted_plan()).unwrap();
        let (outcome, text, states) = run(&mut engine, &request(10));
        assert_eq!(outcome.finish_reason, FinishReason::Length);
        assert_eq!(outcome.committed, 10);
        assert_eq!(
            text,
            "[ref:0][ref:1][ref:2][ref:3][ref:4][ref:5][ref:6][ref:7][ref:8][ref:9]"
        );
        assert_eq!(
            states,
            vec![
                RunState::Warming,
                RunState::Prefilling,
                RunState::Generating
            ]
        );
    }

    #[test]
    fn context_overflow_is_rejected_before_generation() {
        let mut engine = ReferenceTestEngine::new();
        engine.load(&admitted_plan()).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let mut on_token = |_i: u64, _t: &str| true;
        let mut on_state = |_s: RunState| {};
        let mut ctl = GenControl::new(cancel.clone(), paused.clone(), &mut on_token, &mut on_state);
        let err = engine.generate(&request(1000), &mut ctl).unwrap_err();
        assert_eq!(err.code(), ErrorCode::ContextExceedsBudget);
    }

    #[test]
    fn output_stops_on_stop_sequence() {
        let mut engine = ReferenceTestEngine::new();
        engine.per_token_delay = Duration::ZERO;
        engine.load(&admitted_plan()).unwrap();
        let mut req = request(30);
        req.stop = vec!["[ref:3]".into()];
        let (outcome, text, _) = run(&mut engine, &req);
        assert_eq!(outcome.finish_reason, FinishReason::Stop);
        assert_eq!(outcome.committed, 4);
        assert!(text.ends_with("[ref:3]"));
    }

    #[test]
    fn cancellation_is_honored_between_commits() {
        let mut engine = ReferenceTestEngine::new();
        engine.per_token_delay = Duration::from_millis(1);
        engine.load(&admitted_plan()).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let cancel2 = cancel.clone();
        let mut text = String::new();
        let mut count = 0u64;
        let mut on_token = |_i: u64, t: &str| {
            text.push_str(t);
            count += 1;
            if count == 5 {
                cancel2.store(true, Ordering::Relaxed);
            }
            true
        };
        let mut on_state = |_s: RunState| {};
        let mut ctl = GenControl::new(cancel.clone(), paused.clone(), &mut on_token, &mut on_state);
        let outcome = engine.generate(&request(100), &mut ctl).unwrap();
        assert_eq!(outcome.finish_reason, FinishReason::Cancelled);
        assert!(outcome.committed < 100);
        assert_eq!(outcome.committed, count);
    }

    #[test]
    fn pause_holds_at_token_boundary_then_resumes() {
        let mut engine = ReferenceTestEngine::new();
        engine.per_token_delay = Duration::from_millis(1);
        engine.load(&admitted_plan()).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(true)); // start paused
        let mut states = Vec::new();
        let mut on_token = |_i: u64, _t: &str| true;
        let mut on_state = |s: RunState| states.push(s);
        let paused2 = paused.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(60));
            paused2.store(false, Ordering::Relaxed);
        });
        let mut ctl = GenControl::new(cancel.clone(), paused.clone(), &mut on_token, &mut on_state);
        let outcome = engine.generate(&request(4), &mut ctl).unwrap();
        assert_eq!(outcome.committed, 4);
        assert!(
            states.contains(&RunState::Paused),
            "pause must be visible: {states:?}"
        );
    }

    #[test]
    fn rejected_plan_refuses_to_load() {
        let mut engine = ReferenceTestEngine::new();
        let mut plan = admitted_plan();
        plan.mode = ExecutionMode::Rejected;
        plan.rejection = Some(tuckmill_manifest::plan::PlanRejection {
            code: "CONTEXT_EXCEEDS_BUDGET".into(),
            message: "nope".into(),
        });
        plan.plan_id = plan.compute_id().unwrap();
        let err = engine.load(&plan).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidPlan);
    }

    #[test]
    fn identity_is_never_certified() {
        let id = reference_test_identity();
        assert!(!id.certified);
        assert_eq!(id.engine, REFERENCE_TEST_ENGINE);
        assert!(is_synthetic_engine(&id.engine));
    }

    #[test]
    fn trait_impl_accepts_pipeline_plans_for_any_arch() {
        use tuckmill_backend::InferenceBackend;
        let mut engine = ReferenceTestEngine::new();
        let caps = engine.capabilities();
        assert!(caps.architectures.is_empty(), "empty list = pipeline-only");
        let artifact = tuckmill_backend::ArtifactRef {
            path: "/nonexistent".into(),
            file_bytes: 1,
            metadata_hash: "0".repeat(64),
            tensor_index_hash: "0".repeat(64),
            tokenizer_hash: None,
            arch_id: None,
            content_hash: None,
        };
        let note = tuckmill_backend::InferenceBackend::load(
            &mut engine,
            &admitted_plan(),
            &artifact,
            None,
        )
        .expect("synthetic load ignores the artifact");
        assert!(note.contains("synthetic"));
        assert!(engine
            .score(&[1, 2], &[1], &|| false, &mut |_, _| true)
            .is_err());
    }
}
