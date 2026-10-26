//! Supervisor-side worker process: spawn, handshake, load, generate with
//! cancellation, liveness detection, and teardown.
//!
//! The supervisor stays responsive when the worker fails: a crash surfaces as
//! `WORKER_FAILED` with the last acknowledged output position retained, and
//! never fabricated continuation state (PRD §7.4).

use std::collections::VecDeque;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tuckmill_backend::{ArtifactRef, BackendStats, EngineKind};
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::plan::ExecutionPlan;

use crate::engine::{FinishReason, GenOutcome, RunState};
use crate::proto::{hello, validate_handshake, WorkerFrame};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const FRAME_TIMEOUT: Duration = Duration::from_secs(30);
const CANCEL_GRACE: Duration = Duration::from_secs(3);
/// Absolute wall-clock budget for one scoring request: heartbeats prove the
/// worker is alive, but must never let a stalled request run forever.
pub const DEFAULT_SCORE_DEADLINE: Duration = Duration::from_secs(900);

/// A governor/administrative command scoped to the request that was active
/// when it was decided. Scope mismatch at flush time means the command's
/// request already ended: it is dropped, never applied to a later request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlCommand {
    pub scope: String,
    pub action: String,
}

pub struct IsolatedWorker {
    child: Child,
    stream: UnixStream,
    socket_path: PathBuf,
    pub worker_build: String,
    pub engine: String,
    pub certified: bool,
    /// Kernel ids resolved by the backend at load (empty until loaded or
    /// for engines without kernel selection).
    pub resolved_kernels: Vec<String>,
    last_frame: Instant,
    /// Governor commands queued by other threads; the generate/score loops
    /// flush them to the worker as Control frames at read checkpoints.
    pending_controls: Arc<Mutex<VecDeque<ControlCommand>>>,
    /// Commands drained at a request boundary whose scope belonged to an
    /// earlier request: dropped, never applied to the wrong request.
    stale_controls_dropped: u64,
}

impl IsolatedWorker {
    /// Spawn a worker process and complete the handshake. `exe` is the
    /// tuckmill binary (its hidden `__worker-reference` subcommand, which
    /// selects the backend via `--engine`); child stdio is discarded so
    /// nothing leaks to the user's terminal.
    pub fn spawn(exe: &Path, work_dir: &Path, kind: EngineKind) -> Result<Self> {
        std::fs::create_dir_all(work_dir).map_err(TuckmillError::from)?;
        let socket_path = work_dir.join(format!(
            "worker-{}.sock",
            &tuckmill_manifest::ids::secure_random_hex(6)?
        ));
        let child = Command::new(exe)
            .arg("__worker-reference")
            .arg("--socket")
            .arg(&socket_path)
            .arg("--engine")
            .arg(kind.backend_id().as_str())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    format!("cannot spawn worker {}: {e}", exe.display()),
                )
                .with_source(e)
            })?;

        let deadline = Instant::now() + CONNECT_TIMEOUT;
        let stream = loop {
            match UnixStream::connect(&socket_path) {
                Ok(s) => break s,
                Err(_) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(25))
                }
                Err(e) => {
                    let mut child = child;
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!("worker socket never became connectable: {e}"),
                    ));
                }
            }
        };
        stream
            .set_read_timeout(Some(FRAME_TIMEOUT))
            .map_err(TuckmillError::from)?;

        let mut worker = Self {
            child,
            stream,
            socket_path,
            worker_build: String::new(),
            engine: String::new(),
            certified: false,
            resolved_kernels: Vec::new(),
            last_frame: Instant::now(),
            pending_controls: Arc::new(Mutex::new(VecDeque::new())),
            stale_controls_dropped: 0,
        };
        worker.handshake()?;
        Ok(worker)
    }

    fn handshake(&mut self) -> Result<()> {
        hello(&crate::engine::build_id()).write_to(&mut self.stream)?;
        match self.read_frame_skipping_heartbeats(FRAME_TIMEOUT)? {
            Some(WorkerFrame::HelloAck {
                protocol,
                abi,
                worker_build,
                engine,
                certified,
            }) => {
                validate_handshake(protocol, abi)?;
                self.worker_build = worker_build;
                self.engine = engine;
                self.certified = certified;
                Ok(())
            }
            Some(other) => Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!("expected HelloAck, got {other:?}"),
            )),
            None => Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                "worker closed during handshake",
            )),
        }
    }

    pub fn load(
        &mut self,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        options: Option<&serde_json::Value>,
    ) -> Result<String> {
        WorkerFrame::Load {
            plan: Box::new(plan.clone()),
            artifact: artifact.clone(),
            options: options.cloned(),
        }
        .write_to(&mut self.stream)?;
        match self.read_frame_skipping_heartbeats(FRAME_TIMEOUT)? {
            Some(WorkerFrame::Loaded {
                note,
                resolved_kernels,
                ..
            }) => {
                self.resolved_kernels = resolved_kernels;
                Ok(note)
            }
            Some(WorkerFrame::Error {
                code,
                message,
                action,
            }) => Err(map_worker_error(&code, &message, &action)),
            Some(other) => Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!("expected Loaded, got {other:?}"),
            )),
            None => Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                "worker closed during load",
            )),
        }
    }

    /// Generate with streaming callbacks. `pause` is consulted between
    /// tokens: when set, the supervisor stops reading (backpressure pause at a
    /// committed-token boundary) and reports Paused; reads resume when it is
    /// cleared. Cancellation is always honored: the cancel frame is sent and
    /// the stream is drained.
    pub fn generate(
        &mut self,
        spec: crate::engine::EngineRequest,
        cancel: &Arc<AtomicBool>,
        pause: &Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u64, &str) -> bool,
        on_state: &mut dyn FnMut(RunState),
    ) -> Result<GenOutcome> {
        let request_id = spec.request_id.clone();
        WorkerFrame::Generate {
            request_id: request_id.clone(),
            prompt_tokens: spec.prompt_tokens,
            max_output_tokens: spec.max_output_tokens,
            stop: spec.stop,
            seed: spec.seed,
            sampling: spec.sampling,
            ignore_eos: spec.ignore_eos,
        }
        .write_to(&mut self.stream)?;

        let mut committed: u64 = 0;
        let mut cancel_sent_at: Option<Instant> = None;
        let mut paused_reported = false;
        loop {
            self.flush_controls(&request_id)?;
            if cancel.load(Ordering::Relaxed) && cancel_sent_at.is_none() {
                let _ = WorkerFrame::Cancel {
                    request_id: request_id.clone(),
                }
                .write_to(&mut self.stream);
                cancel_sent_at = Some(Instant::now());
                // A pause must lift so the worker can observe the cancel.
                pause.store(false, Ordering::Relaxed);
            }
            if pause.load(Ordering::Relaxed) {
                if !paused_reported {
                    on_state(RunState::Paused);
                    paused_reported = true;
                }
                std::thread::sleep(Duration::from_millis(20));
                continue;
            }
            if paused_reported {
                on_state(RunState::Generating);
                paused_reported = false;
            }
            let timeout = if cancel_sent_at.is_some() {
                CANCEL_GRACE
            } else {
                FRAME_TIMEOUT
            };
            match self.read_frame_timed(timeout)? {
                Some(WorkerFrame::Token { index, text }) => {
                    committed = committed.max(index + 1);
                    if !on_token(index, &text) && cancel_sent_at.is_none() {
                        let _ = WorkerFrame::Cancel {
                            request_id: request_id.clone(),
                        }
                        .write_to(&mut self.stream);
                        cancel_sent_at = Some(Instant::now());
                    }
                }
                Some(WorkerFrame::State { state }) => {
                    let s = match state.as_str() {
                        "warming" => RunState::Warming,
                        "prefilling" => RunState::Prefilling,
                        "generating" => RunState::Generating,
                        "throttling" => RunState::Throttling,
                        "paused" => RunState::Paused,
                        _ => continue,
                    };
                    on_state(s);
                }
                Some(WorkerFrame::Heartbeat { .. }) => {}
                // The governor's Control frames are acknowledged at the next
                // safe checkpoint; that ACK legitimately lands mid-generation
                // (the scoring loop already tolerates it). Never a violation.
                Some(WorkerFrame::ControlAck { .. }) => {}
                Some(WorkerFrame::Done {
                    reason,
                    committed: c,
                    generated,
                    timing,
                }) => {
                    committed = c;
                    let finish = match reason.as_str() {
                        "stop" => FinishReason::Stop,
                        "length" => FinishReason::Length,
                        "cancelled" => FinishReason::Cancelled,
                        "eos" => FinishReason::Eos,
                        other => {
                            return Err(TuckmillError::new(
                                ErrorCode::WorkerFailed,
                                format!("worker reported unknown finish reason {other:?}"),
                            ))
                        }
                    };
                    let generated = if generated == 0 { c } else { generated };
                    let mut outcome =
                        GenOutcome::new(finish, committed).with_generated(generated.max(c));
                    outcome.timing = timing;
                    return Ok(outcome);
                }
                Some(WorkerFrame::Error {
                    code,
                    message,
                    action,
                }) => {
                    return Err(map_worker_error(&code, &message, &action));
                }
                Some(other) => {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!("unexpected frame during generation: {other:?}"),
                    ));
                }
                None => {
                    // EOF: worker crashed. The supervisor keeps the last
                    // acknowledged committed position and reports failure.
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!(
                            "worker exited mid-generation after {committed} committed tokens; restart from the last acknowledged position"
                        ),
                    ));
                }
            }
            if cancel_sent_at
                .map(|t| t.elapsed() > CANCEL_GRACE * 4)
                .unwrap_or(false)
            {
                // The worker did not acknowledge the cancel in time; the
                // supervisor escalates by killing the process rather than
                // pretending the request was cleanly cancelled.
                let _ = self.child.kill();
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "worker did not honor cancellation within its grace period and was terminated",
                ));
            }
        }
    }

    /// Query worker statistics.
    pub fn stats(&mut self) -> Result<(u64, u64, Option<BackendStats>)> {
        WorkerFrame::Stats.write_to(&mut self.stream)?;
        let deadline = Instant::now() + FRAME_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match self.read_frame_skipping_heartbeats(remaining)? {
                Some(WorkerFrame::StatsAck {
                    footprint_bytes,
                    uptime_ms,
                    backend,
                }) => return Ok((footprint_bytes, uptime_ms, backend)),
                // A governor ControlAck queued during a run may arrive here;
                // it is expected traffic, never a protocol violation.
                Some(WorkerFrame::ControlAck { .. }) => continue,
                Some(other) => {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!("expected StatsAck, got {other:?}"),
                    ))
                }
                None => {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        "worker closed during stats query",
                    ))
                }
            }
        }
    }

    /// Teacher-forced scoring: requests bounded logits for the listed
    /// positions and returns them in request order. Bounded by the protocol
    /// (<= 64 positions, <= 16384 tokens per request) and by the caller.
    /// Teacher-forced scoring: requests bounded logits for the listed
    /// positions and returns them in request order. Bounded by the protocol
    /// (<= 64 positions, <= 16384 tokens per request), by the accumulation
    /// caps in `scoring`, and by `deadline` — heartbeats keep liveness
    /// evidence flowing but never extend the absolute budget; expiry cancels
    /// at a position boundary.
    pub fn score(
        &mut self,
        request_id: &str,
        tokens: &[u32],
        positions: &[u64],
        deadline: Duration,
    ) -> Result<Vec<(u64, Vec<f32>)>> {
        WorkerFrame::Score {
            request_id: request_id.to_string(),
            tokens: tokens.to_vec(),
            positions: positions.to_vec(),
        }
        .write_to(&mut self.stream)?;
        let mut acc = crate::scoring::ScoreAccumulator::new();
        let deadline = Instant::now() + deadline;
        let mut cancel_sent = false;
        loop {
            self.flush_controls(request_id)?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                if !cancel_sent {
                    let _ = WorkerFrame::Cancel {
                        request_id: request_id.to_string(),
                    }
                    .write_to(&mut self.stream);
                    cancel_sent = true;
                }
                // Grace: accept a clean acknowledgement, never a partial row.
                match self.read_frame_timed(CANCEL_GRACE)? {
                    Some(WorkerFrame::Error { code, .. }) if code == "CANCELLED" => {
                        return Err(TuckmillError::new(
                            ErrorCode::WorkerFailed,
                            "scoring exceeded its deadline and was cancelled at a position boundary",
                        ));
                    }
                    Some(WorkerFrame::Error {
                        code,
                        message,
                        action,
                    }) => {
                        return Err(map_worker_error(&code, &message, &action));
                    }
                    Some(_) => {}
                    None => {
                        let _ = self.child.kill();
                        return Err(TuckmillError::new(
                            ErrorCode::WorkerFailed,
                            "scoring exceeded its deadline; the worker did not acknowledge cancellation and was terminated",
                        ));
                    }
                }
                continue;
            }
            match self.read_frame_timed(remaining.min(FRAME_TIMEOUT))? {
                Some(WorkerFrame::ScoreChunk {
                    position,
                    offset,
                    logits,
                }) => {
                    acc.push_chunk(position, offset, &logits)?;
                }
                Some(WorkerFrame::Heartbeat { .. }) => {}
                Some(WorkerFrame::ControlAck { .. }) => {}
                Some(WorkerFrame::ScoreDone { scored }) => {
                    return acc.finish(scored, positions);
                }
                Some(WorkerFrame::Error {
                    code,
                    message,
                    action,
                }) => return Err(map_worker_error(&code, &message, &action)),
                Some(other) => {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!("unexpected frame during scoring: {other:?}"),
                    ))
                }
                None => {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        "worker exited during scoring",
                    ))
                }
            }
        }
    }

    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Handle for other threads (governor) to enqueue backend commands; the
    /// running loop flushes them within one read slice. Commands are scoped
    /// to the request that was active when the governor decided them, so a
    /// command can never silently execute during a different request.
    pub fn controls(&self) -> Arc<Mutex<VecDeque<ControlCommand>>> {
        self.pending_controls.clone()
    }

    /// Commands dropped at request boundaries because their scope belonged
    /// to an earlier request.
    pub fn stale_controls_dropped(&self) -> u64 {
        self.stale_controls_dropped
    }

    /// Flush queued Control frames for this request's scope; called from the
    /// generate/score read loops so mid-run commands reach the worker
    /// without contending for its lock. Commands scoped to another (older)
    /// request are dropped and counted, never sent late.
    fn flush_controls(&mut self, scope: &str) -> Result<()> {
        let (actions, stale): (Vec<String>, usize) = {
            let mut queue = self.pending_controls.lock().map_err(|_| {
                TuckmillError::new(ErrorCode::WorkerFailed, "control queue poisoned")
            })?;
            let drained: Vec<ControlCommand> = queue.drain(..).collect();
            let mut actions = Vec::new();
            let mut stale = 0usize;
            for cmd in drained {
                if cmd.scope == scope {
                    actions.push(cmd.action);
                } else {
                    stale += 1;
                }
            }
            (actions, stale)
        };
        self.stale_controls_dropped += stale as u64;
        for action in actions {
            WorkerFrame::Control { action }.write_to(&mut self.stream)?;
        }
        Ok(())
    }

    /// Read the next meaningful frame, skipping heartbeats (they are
    /// liveness evidence, never protocol payload).
    fn read_frame_skipping_heartbeats(&mut self, timeout: Duration) -> Result<Option<WorkerFrame>> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "worker produced no meaningful frame within the timeout",
                ));
            }
            match self.read_frame_timed(remaining)? {
                Some(WorkerFrame::Heartbeat { .. }) => continue,
                other => return Ok(other),
            }
        }
    }

    fn read_frame_timed(&mut self, timeout: Duration) -> Result<Option<WorkerFrame>> {
        use std::os::unix::io::AsRawFd;
        let deadline = Instant::now() + timeout;
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    format!(
                        "worker produced no frame for {:?} (last frame {:?} ago)",
                        timeout,
                        self.last_frame.elapsed()
                    ),
                ));
            }
            let wait = (deadline - now).min(Duration::from_millis(250));
            let mut pfd = libc::pollfd {
                fd: self.stream.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: poll with one initialized pollfd.
            let rc = unsafe { libc::poll(&mut pfd, 1, wait.as_millis() as i32) };
            if rc < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    format!("worker socket poll failed: {err}"),
                ));
            }
            if rc == 0 {
                continue;
            }
            // The socket is readable or dead. A read on a dead socket may
            // surface as EOF (0 bytes) or as an OS error; both mean the
            // worker connection is gone, which is a WORKER_FAILED condition —
            // never a generic IO error. The read timeout was set once at
            // spawn; do not touch socket options here (some platforms reject
            // set_read_timeout on a disconnected socket).
            return match WorkerFrame::read_from(&mut self.stream) {
                Ok(frame) => {
                    if frame.is_some() {
                        self.last_frame = Instant::now();
                    }
                    Ok(frame)
                }
                Err(e) if e.code() == ErrorCode::Io => Ok(None),
                Err(e) => Err(e),
            };
        }
    }

    /// Graceful shutdown; falls back to kill after a short grace period.
    pub fn shutdown(&mut self) {
        let _ = WorkerFrame::Shutdown.write_to(&mut self.stream);
        let deadline = Instant::now() + Duration::from_millis(700);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(20))
                }
                _ => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    break;
                }
            }
        }
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

impl Drop for IsolatedWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn map_worker_error(code: &str, message: &str, action: &str) -> TuckmillError {
    let parsed = serde_json::from_value::<tuckmill_manifest::ErrorCode>(serde_json::Value::String(
        code.to_string(),
    ))
    .unwrap_or(tuckmill_manifest::ErrorCode::WorkerFailed);
    TuckmillError::new(parsed, message.to_string()).with_action(action.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full isolated-worker round trip. Requires a tuckmill binary path in
    /// TUCKMILL_TEST_WORKER_BIN (the CLI e2e suite sets it); without it the
    /// test is skipped rather than simulated.
    #[test]
    fn isolated_worker_round_trip_when_binary_available() {
        let Some(bin) = crate::test_worker_bin() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut worker =
            IsolatedWorker::spawn(Path::new(&bin), dir.path(), EngineKind::ReferenceTest).unwrap();
        assert_eq!(worker.engine, crate::engine::REFERENCE_TEST_ENGINE);
        assert!(!worker.certified);
        assert!(worker.is_alive());

        let mut plan = ExecutionPlan::new_skeleton(
            tuckmill_manifest::Id::new("a".repeat(64)).unwrap(),
            tuckmill_manifest::Id::new("b".repeat(64)).unwrap(),
            "qwen3".into(),
        );
        plan.mode = tuckmill_manifest::plan::ExecutionMode::Resident;
        plan.context = tuckmill_manifest::plan::PlanContext::new(128, 16);
        plan.plan_id = plan.compute_id().unwrap();
        let artifact = ArtifactRef {
            path: dir.path().join("fixture.bin").display().to_string(),
            file_bytes: 0,
            metadata_hash: "0".repeat(64),
            tensor_index_hash: "0".repeat(64),
            tokenizer_hash: None,
            arch_id: Some("qwen3".into()),
            content_hash: None,
        };
        let note = worker.load(&plan, &artifact, None).unwrap();
        assert!(note.contains("synthetic"));

        // Regression (observed live on the real 0.6B under memory pressure):
        // a governor `trim-caches` queued before/during generation produces a
        // ControlAck that lands mid-stream. The read loops must absorb it —
        // it used to abort the run as "unexpected frame".
        worker.controls().lock().unwrap().push_back(ControlCommand {
            scope: "r1".into(),
            action: "trim-caches".into(),
        });

        let cancel = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        let mut text = String::new();
        let mut on_token = |_i: u64, t: &str| {
            text.push_str(t);
            true
        };
        let mut on_state = |_s: RunState| {};
        let outcome = worker
            .generate(
                crate::engine::EngineRequest {
                    request_id: "r1".into(),
                    prompt_tokens: vec![1, 2, 3, 4],
                    max_output_tokens: 12,
                    stop: vec![],
                    seed: 7,
                    sampling: tuckmill_backend::SamplingPolicy::default(),
                    ignore_eos: false,
                },
                &cancel,
                &pause,
                &mut on_token,
                &mut on_state,
            )
            .unwrap();
        assert_eq!(outcome.finish_reason, FinishReason::Length);
        assert_eq!(outcome.committed, 12);
        assert!(text.starts_with("[ref:0]"));

        let (footprint, uptime, backend) = worker.stats().unwrap();
        assert!(footprint > 0);
        assert!(uptime > 0);
        let backend = backend.expect("backend stats present");
        assert_eq!(
            backend.logical_read_bytes, 0,
            "synthetic engine reads nothing"
        );

        // Cancel path.
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel2 = cancel.clone();
        let mut count = 0u64;
        let mut on_token = |_i: u64, _t: &str| {
            count += 1;
            if count >= 2 {
                cancel2.store(true, Ordering::Relaxed);
            }
            true
        };
        let mut on_state = |_s: RunState| {};
        let outcome = worker
            .generate(
                crate::engine::EngineRequest {
                    request_id: "r2".into(),
                    prompt_tokens: vec![1, 2, 3, 4],
                    max_output_tokens: 100,
                    stop: vec![],
                    seed: 7,
                    sampling: tuckmill_backend::SamplingPolicy::default(),
                    ignore_eos: false,
                },
                &cancel,
                &pause,
                &mut on_token,
                &mut on_state,
            )
            .unwrap();
        assert_eq!(outcome.finish_reason, FinishReason::Cancelled);
        assert!(outcome.committed < 100);
    }

    #[test]
    fn worker_error_mapping_preserves_codes() {
        let err = map_worker_error("CONTEXT_EXCEEDS_BUDGET", "too long", "shorten it");
        assert_eq!(err.code(), ErrorCode::ContextExceedsBudget);
        assert_eq!(err.action(), "shorten it");
        let err = map_worker_error("NOT_A_CODE", "x", "y");
        assert_eq!(err.code(), ErrorCode::WorkerFailed);
    }

    /// A deliberately delayed worker: the per-token delay must appear in
    /// measured wall time and in the worker-side timing — a regression guard
    /// that latencies are not measured after production already happened.
    #[test]
    fn isolated_worker_delay_option_is_measured() {
        let Some(bin) = crate::test_worker_bin() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut worker =
            IsolatedWorker::spawn(Path::new(&bin), dir.path(), EngineKind::ReferenceTest).unwrap();

        let mut plan = ExecutionPlan::new_skeleton(
            tuckmill_manifest::Id::new("a".repeat(64)).unwrap(),
            tuckmill_manifest::Id::new("b".repeat(64)).unwrap(),
            "qwen3".into(),
        );
        plan.mode = tuckmill_manifest::plan::ExecutionMode::Resident;
        plan.context = tuckmill_manifest::plan::PlanContext::new(128, 16);
        plan.plan_id = plan.compute_id().unwrap();
        let artifact = ArtifactRef {
            path: dir.path().join("fixture.bin").display().to_string(),
            file_bytes: 0,
            metadata_hash: "0".repeat(64),
            tensor_index_hash: "0".repeat(64),
            tokenizer_hash: None,
            arch_id: Some("qwen3".into()),
            content_hash: None,
        };
        let options = serde_json::json!({ "per_token_delay_ms": 50 });
        let note = worker.load(&plan, &artifact, Some(&options)).unwrap();
        assert!(note.contains("synthetic"));

        let cancel = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        let mut on_token = |_i: u64, _t: &str| true;
        let mut on_state = |_s: RunState| {};
        let t0 = Instant::now();
        let outcome = worker
            .generate(
                crate::engine::EngineRequest {
                    request_id: "delayed".into(),
                    prompt_tokens: vec![1, 2, 3, 4],
                    max_output_tokens: 12,
                    stop: vec![],
                    seed: 7,
                    sampling: tuckmill_backend::SamplingPolicy::default(),
                    ignore_eos: false,
                },
                &cancel,
                &pause,
                &mut on_token,
                &mut on_state,
            )
            .unwrap();
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(outcome.committed, 12);
        let timing = outcome.timing.expect("worker reports request timing");
        assert!(
            wall_ms >= 500.0,
            "12 tokens at 50 ms/token must take >= 600 ms of wall time, got {wall_ms}"
        );
        assert!(
            timing.total_ms >= 500.0,
            "worker-side total {} ms must include the delay",
            timing.total_ms
        );
        assert!(timing.first_token_ms.unwrap_or(0.0) >= 40.0);
        assert!(timing.decode_ms.unwrap_or(0.0) >= 500.0);
        assert!(
            wall_ms + 5.0 >= timing.total_ms,
            "supervisor wall {wall_ms} ms must not be below worker total {} ms",
            timing.total_ms
        );
    }

    /// Commands are scoped to the request that was active when the governor
    /// decided them: a command whose request already ended is dropped at the
    /// next flush (counted), never applied during a different request.
    #[test]
    fn stale_control_commands_never_cross_request_scopes() {
        let Some(bin) = crate::test_worker_bin() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut worker =
            IsolatedWorker::spawn(&bin, dir.path(), EngineKind::ReferenceTest).unwrap();
        let mut plan = ExecutionPlan::new_skeleton(
            tuckmill_manifest::Id::new("a".repeat(64)).unwrap(),
            tuckmill_manifest::Id::new("b".repeat(64)).unwrap(),
            "qwen3".into(),
        );
        plan.mode = tuckmill_manifest::plan::ExecutionMode::Resident;
        plan.context = tuckmill_manifest::plan::PlanContext::new(128, 16);
        plan.plan_id = plan.compute_id().unwrap();
        let artifact = ArtifactRef {
            path: dir.path().join("fixture.bin").display().to_string(),
            file_bytes: 0,
            metadata_hash: "0".repeat(64),
            tensor_index_hash: "0".repeat(64),
            tokenizer_hash: None,
            arch_id: Some("qwen3".into()),
            content_hash: None,
        };
        worker.load(&plan, &artifact, None).unwrap();

        {
            let controls = worker.controls();
            let mut queue = controls.lock().unwrap();
            queue.push_back(ControlCommand {
                scope: "previous-request".into(),
                action: "trim-caches".into(),
            });
            queue.push_back(ControlCommand {
                scope: "r1".into(),
                action: "set-prefetch=off".into(),
            });
        }

        let run = |worker: &mut IsolatedWorker, request_id: &str| {
            let cancel = Arc::new(AtomicBool::new(false));
            let pause = Arc::new(AtomicBool::new(false));
            let mut on_token = |_i: u64, _t: &str| true;
            let mut on_state = |_s: RunState| {};
            worker
                .generate(
                    crate::engine::EngineRequest {
                        request_id: request_id.into(),
                        prompt_tokens: vec![1, 2, 3, 4],
                        max_output_tokens: 4,
                        stop: vec![],
                        seed: 0,
                        sampling: tuckmill_backend::SamplingPolicy::default(),
                        ignore_eos: false,
                    },
                    &cancel,
                    &pause,
                    &mut on_token,
                    &mut on_state,
                )
                .unwrap()
        };

        // The stale-scope command is dropped; the r1-scope command is sent
        // and acknowledged during r1 without failing the run.
        let outcome = run(&mut worker, "r1");
        assert_eq!(outcome.finish_reason, FinishReason::Length);
        assert_eq!(
            worker.stale_controls_dropped(),
            1,
            "the previous-request-scoped command must be dropped, not late-applied"
        );
        // Nothing left to drop for the next request.
        let outcome = run(&mut worker, "r2");
        assert_eq!(outcome.finish_reason, FinishReason::Length);
        assert_eq!(worker.stale_controls_dropped(), 1);
    }
}
