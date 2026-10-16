//! Worker host: serves a selected inference backend over the framed protocol
//! on a private local socket. This runs inside the dedicated worker process;
//! the supervisor never computes tensors or holds model state.
//!
//! Control/liveness design (Stage A review hardening):
//! - all outbound frames go through one mutex-guarded writer, and a heartbeat
//!   thread emits frames every 500 ms while the engine computes — a long
//!   single position can no longer look like a stalled worker;
//! - the engine receives a poll hook that drains inbound control frames
//!   during long single-step work, so cancellation and governor commands are
//!   consumed promptly instead of waiting for the next token callback;
//! - `Control` frames apply locally (`trim-caches` bumps a shared epoch the
//!   engine checks at safe points; in-flight buffers are never released) and
//!   are acknowledged with `ControlAck`.

use std::cell::{Cell, RefCell};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tuckmill_backend::{BackendId, EngineKind, InferenceBackend};
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};

use crate::engine::{GenControl, ReferenceTestEngine, RunState};
use crate::proto::{hello_ack, validate_handshake, WorkerFrame};

const POLL_SLICE: Duration = Duration::from_millis(250);
const HEARTBEAT_INTERVAL: Duration = Duration::from_millis(500);
/// Bounds for scoring requests (PRD §15: bounded evaluation interface).
const MAX_SCORE_TOKENS: usize = 16384;
const MAX_SCORE_POSITIONS: usize = 64;
/// Logits per ScoreChunk frame. JSON floats worst-case ~24 bytes each, so
/// 16k floats stay far under the 1 MiB cap while a 150k+ vocabulary streams
/// in a handful of frames.
const SCORE_CHUNK_FLOATS: usize = 16_384;

fn make_backend(kind: EngineKind) -> Box<dyn InferenceBackend> {
    match kind {
        EngineKind::ReferenceTest => Box::new(ReferenceTestEngine::new()),
        EngineKind::TuckCore => Box::new(tuckmill_tuckcore::TuckCore::new()),
        EngineKind::LlamaCpp => Box::new(tuckmill_llamaref::LlamaRef::new()),
    }
}

/// Shared outbound writer: the main thread and the heartbeat thread both
/// write whole frames under this lock, so frames can never interleave.
type Writer = Arc<Mutex<UnixStream>>;

fn write_frame(writer: &Writer, frame: &WorkerFrame) -> Result<()> {
    let mut stream = writer
        .lock()
        .map_err(|_| TuckmillError::new(ErrorCode::WorkerFailed, "worker writer lock poisoned"))?;
    frame.write_to(&mut *stream)
}

fn write_frame_best_effort(writer: &Writer, frame: &WorkerFrame) {
    let _ = write_frame(writer, frame);
}

/// Serve one supervisor connection until Shutdown or EOF. `socket_path` must
/// be a private path (the store's tmp directory).
pub fn run_worker(socket_path: &Path, engine: EngineKind) -> Result<()> {
    if socket_path.exists() {
        std::fs::remove_file(socket_path).map_err(TuckmillError::from)?;
    }
    let listener = UnixListener::bind(socket_path).map_err(|e| {
        TuckmillError::new(
            ErrorCode::WorkerFailed,
            format!("cannot bind worker socket {}: {e}", socket_path.display()),
        )
        .with_source(e)
    })?;
    let (stream, _) = listener.accept().map_err(|e| {
        TuckmillError::new(ErrorCode::WorkerFailed, format!("accept failed: {e}")).with_source(e)
    })?;
    let result = serve(stream, engine);
    let _ = std::fs::remove_file(socket_path);
    result
}

/// Backwards-compatible helper for the original single-engine worker.
pub fn run_reference_worker(socket_path: &Path) -> Result<()> {
    run_worker(socket_path, EngineKind::ReferenceTest)
}

struct HeartbeatScope {
    stop: Arc<AtomicBool>,
}

impl Drop for HeartbeatScope {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Heartbeats every 500 ms from a helper thread while a long engine call
/// runs on the main thread.
fn heartbeat_thread(writer: &Writer, started: Instant) -> HeartbeatScope {
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    let writer2 = writer.clone();
    std::thread::spawn(move || {
        while !stop2.load(Ordering::Relaxed) {
            std::thread::sleep(HEARTBEAT_INTERVAL);
            if stop2.load(Ordering::Relaxed) {
                break;
            }
            write_frame_best_effort(
                &writer2,
                &WorkerFrame::Heartbeat {
                    uptime_ms: started.elapsed().as_millis() as u64,
                },
            );
        }
    });
    HeartbeatScope { stop }
}

fn serve(stream: UnixStream, kind: EngineKind) -> Result<()> {
    let started = Instant::now();
    let mut engine = make_backend(kind);
    let mut loaded = false;

    // Reader (main thread only) and shared writer (main + heartbeat threads).
    let reader = RefCell::new(stream.try_clone().map_err(TuckmillError::from)?);
    let writer: Writer = Arc::new(Mutex::new(stream));
    let cancel = Arc::new(AtomicBool::new(false));
    let paused = Arc::new(AtomicBool::new(false));
    // Local control state applied by engines at safe checkpoints. The trim
    // epoch is the engine's own shared handle so Control drains and the
    // engine's checkpoint checks act on the same counter.
    let trim_epoch = engine.trim_epoch();
    let prefetch_enabled = Arc::new(AtomicBool::new(true));

    // Handshake: the first frame must be Hello.
    let first = read_frame_blocking(&reader)?;
    match first {
        Some(WorkerFrame::Hello {
            protocol,
            abi,
            client_build: _,
        }) => {
            validate_handshake(protocol, abi)?;
            let identity = engine.identity();
            write_frame(
                &writer,
                &hello_ack(
                    &crate::engine::build_id(),
                    &identity.engine,
                    identity.certified,
                ),
            )?;
        }
        Some(_) => {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                "first frame was not Hello",
            ))
        }
        None => return Ok(()), // supervisor went away before handshake
    }

    let mut last_heartbeat = Instant::now();
    loop {
        // Poll for the next control frame; emit heartbeats while idle.
        match poll_frame(&reader, POLL_SLICE)? {
            PollResult::Timeout => {
                if last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
                    write_frame(
                        &writer,
                        &WorkerFrame::Heartbeat {
                            uptime_ms: started.elapsed().as_millis() as u64,
                        },
                    )?;
                    last_heartbeat = Instant::now();
                }
                continue;
            }
            PollResult::Eof => {
                engine.shutdown();
                return Ok(());
            }
            PollResult::Frame(frame) => match *frame {
                WorkerFrame::Load {
                    plan,
                    artifact,
                    options,
                } => {
                    let result = (|| {
                        // Capability gate: a backend with a declared
                        // architecture list refuses anything outside it.
                        let caps = engine.capabilities();
                        if !caps.architectures.is_empty() {
                            if let Some(arch) = artifact.arch_id.as_deref() {
                                if !caps.supports_arch(arch) {
                                    return Err(TuckmillError::new(
                                        ErrorCode::UnsupportedArchitecture,
                                        format!(
                                            "backend {} does not execute architecture {arch:?} in this build (supported: {})",
                                            engine.backend_id().as_str(),
                                            caps.architectures.join(", ")
                                        ),
                                    )
                                    .with_action(
                                        "Use the reference test engine for pipeline validation or wait for this architecture's adapter work.",
                                    ));
                                }
                            }
                        }
                        let _heartbeat = heartbeat_thread(&writer, started);
                        engine.load(&plan, &artifact, options.as_ref())
                    })();
                    match result {
                        Ok(note) => {
                            loaded = true;
                            write_frame(
                                &writer,
                                &WorkerFrame::Loaded {
                                    note,
                                    footprint_bytes: crate::host_footprint_bytes(),
                                    resolved_kernels: engine.identity().kernels,
                                },
                            )?;
                        }
                        Err(e) => {
                            write_frame(&writer, &error_frame(&e))?;
                        }
                    }
                }
                WorkerFrame::Generate {
                    request_id,
                    prompt_tokens,
                    max_output_tokens,
                    stop,
                    seed,
                } => {
                    if !loaded {
                        write_frame(
                            &writer,
                            &WorkerFrame::Error {
                                code: ErrorCode::WorkerFailed.as_str().into(),
                                message: "Generate before Load".into(),
                                action: "Load a plan first".into(),
                            },
                        )?;
                        continue;
                    }
                    cancel.store(false, Ordering::Relaxed);
                    paused.store(false, Ordering::Relaxed);
                    last_heartbeat = Instant::now();
                    let _heartbeat = heartbeat_thread(&writer, started);
                    let outcome = generate_streaming(
                        engine.as_mut(),
                        &reader,
                        &writer,
                        &cancel,
                        &paused,
                        &trim_epoch,
                        &prefetch_enabled,
                        started,
                        crate::engine::GenerateRequest {
                            request_id,
                            prompt_tokens,
                            max_output_tokens,
                            stop,
                            seed,
                        },
                    );
                    cancel.store(false, Ordering::Relaxed);
                    paused.store(false, Ordering::Relaxed);
                    match outcome {
                        Ok(outcome) => {
                            write_frame(
                                &writer,
                                &WorkerFrame::Done {
                                    reason: outcome.finish_reason.as_str().to_string(),
                                    committed: outcome.committed,
                                    timing: outcome.timing,
                                },
                            )?;
                        }
                        Err(e) => {
                            write_frame(&writer, &error_frame(&e))?;
                        }
                    }
                }
                WorkerFrame::Score {
                    request_id: _,
                    tokens,
                    positions,
                } => {
                    if !loaded {
                        write_frame(
                            &writer,
                            &WorkerFrame::Error {
                                code: ErrorCode::WorkerFailed.as_str().into(),
                                message: "Score before Load".into(),
                                action: "Load a plan first".into(),
                            },
                        )?;
                        continue;
                    }
                    let bounded = tokens.len() <= MAX_SCORE_TOKENS
                        && positions.len() <= MAX_SCORE_POSITIONS
                        && positions.iter().all(|p| (*p as usize) < tokens.len());
                    if !bounded {
                        write_frame(
                            &writer,
                            &WorkerFrame::Error {
                                code: ErrorCode::InvalidRequest.as_str().into(),
                                message: format!(
                                    "scoring request outside bounds (max {MAX_SCORE_TOKENS} tokens, {MAX_SCORE_POSITIONS} positions, positions within the token sequence)"
                                ),
                                action: "Reduce the scoring request; the evaluation interface is deliberately bounded.".into(),
                            },
                        )?;
                        continue;
                    }
                    let _heartbeat = heartbeat_thread(&writer, started);
                    let mut write_failed = false;
                    let cancelled = || {
                        // Drains inbound control frames (Cancel, Shutdown,
                        // Control) between chunks of long per-position work.
                        if drain_control(
                            &reader,
                            &cancel,
                            &paused,
                            &trim_epoch,
                            &prefetch_enabled,
                            &writer,
                        )
                        .is_err()
                        {
                            cancel.store(true, Ordering::Relaxed);
                        }
                        cancel.load(Ordering::Relaxed)
                    };
                    let result =
                        engine.score(&tokens, &positions, &cancelled, &mut |position, logits| {
                            for (i, chunk) in logits.chunks(SCORE_CHUNK_FLOATS).enumerate() {
                                let frame = WorkerFrame::ScoreChunk {
                                    position,
                                    offset: (i * SCORE_CHUNK_FLOATS) as u32,
                                    logits: chunk.to_vec(),
                                };
                                if write_frame(&writer, &frame).is_err() {
                                    write_failed = true;
                                    return false;
                                }
                            }
                            true
                        });
                    let _ = write_failed;
                    match result {
                        Ok(scored) => {
                            write_frame(&writer, &WorkerFrame::ScoreDone { scored })?;
                        }
                        Err(e) => {
                            write_frame(&writer, &error_frame(&e))?;
                        }
                    }
                }
                WorkerFrame::Control { action } => {
                    let result = apply_control(&action, &trim_epoch, &prefetch_enabled);
                    write_frame(&writer, &WorkerFrame::ControlAck { action, result })?;
                }
                WorkerFrame::Cancel { .. } => {
                    cancel.store(true, Ordering::Relaxed);
                }
                WorkerFrame::Stats => {
                    write_frame(
                        &writer,
                        &WorkerFrame::StatsAck {
                            footprint_bytes: crate::host_footprint_bytes(),
                            uptime_ms: started.elapsed().as_millis() as u64,
                            backend: Some(engine.stats()),
                        },
                    )?;
                }
                WorkerFrame::Shutdown => {
                    engine.shutdown();
                    write_frame(&writer, &WorkerFrame::Bye)?;
                    return Ok(());
                }
                other => {
                    // Unexpected but harmless control frames (e.g. heartbeat
                    // from a confused client) are a protocol violation.
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!("unexpected control frame in worker: {other:?}"),
                    ));
                }
            },
        }
    }
}

fn error_frame(e: &TuckmillError) -> WorkerFrame {
    WorkerFrame::Error {
        code: e.code().as_str().into(),
        message: e.message().to_string(),
        action: e.action().to_string(),
    }
}

/// Apply a control command to the local engine-facing flags. Honest about
/// what this build supports: there is no speculative prefetch yet, so
/// prefetch commands configure the flag for engines that add one and ack
/// what actually happened.
fn apply_control(
    action: &str,
    trim_epoch: &Arc<AtomicU64>,
    prefetch_enabled: &Arc<AtomicBool>,
) -> String {
    match action {
        "trim-caches" => {
            trim_epoch.fetch_add(1, Ordering::Relaxed);
            "queued: engine applies at its next safe check (in-flight buffers are never released)".into()
        }
        "set-prefetch=off" => {
            prefetch_enabled.store(false, Ordering::Relaxed);
            "recorded: no speculative prefetch exists in this build".into()
        }
        "set-prefetch=on" => {
            prefetch_enabled.store(true, Ordering::Relaxed);
            "recorded: no speculative prefetch exists in this build".into()
        }
        other => format!("unsupported control action {other:?} (this build applies: trim-caches, set-prefetch=on/off)"),
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_streaming(
    engine: &mut dyn InferenceBackend,
    reader: &RefCell<UnixStream>,
    writer: &Writer,
    cancel: &Arc<AtomicBool>,
    paused: &Arc<AtomicBool>,
    trim_epoch: &Arc<AtomicU64>,
    prefetch_enabled: &Arc<AtomicBool>,
    started: Instant,
    req: crate::engine::GenerateRequest,
) -> Result<crate::engine::GenOutcome> {
    // The poll hook lets engines service inbound control frames (cancel,
    // shutdown, governor commands) inside long single-step compute, instead
    // of waiting for the next token callback.
    let poll_hook = || {
        if drain_control(reader, cancel, paused, trim_epoch, prefetch_enabled, writer).is_err() {
            cancel.store(true, Ordering::Relaxed);
        }
    };
    let pending_state: Cell<Option<RunState>> = Cell::new(None);
    let mut outcome_slot: Option<TuckmillError> = None;
    let result = {
        let mut on_token = |index: u64, text: &str| -> bool {
            let _ = drain_control(reader, cancel, paused, trim_epoch, prefetch_enabled, writer);
            if let Some(state) = pending_state.take() {
                write_frame_best_effort(
                    writer,
                    &WorkerFrame::State {
                        state: state.as_str().to_string(),
                    },
                );
            }
            if write_frame(
                writer,
                &WorkerFrame::Token {
                    index,
                    text: text.to_string(),
                },
            )
            .is_err()
            {
                outcome_slot = Some(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "supervisor connection lost while streaming",
                ));
                return false;
            }
            true
        };
        let mut on_state = |state: RunState| {
            pending_state.set(Some(state));
        };
        let mut ctl = GenControl::new(cancel.clone(), paused.clone(), &mut on_token, &mut on_state);
        ctl.trim_epoch = trim_epoch.clone();
        ctl.poll = Some(&poll_hook);
        engine.generate(&req, &mut ctl)
    };
    let _ = started;
    if let Some(err) = outcome_slot {
        return Err(err);
    }
    result
}

/// Non-blocking drain of inbound control frames: Cancel/Shutdown set flags;
/// Control applies locally and is acknowledged; everything else is ignored
/// mid-run.
fn drain_control(
    reader: &RefCell<UnixStream>,
    cancel: &Arc<AtomicBool>,
    _paused: &Arc<AtomicBool>,
    trim_epoch: &Arc<AtomicU64>,
    prefetch_enabled: &Arc<AtomicBool>,
    writer: &Writer,
) -> Result<()> {
    loop {
        match poll_frame(reader, Duration::ZERO)? {
            PollResult::Frame(frame) => match *frame {
                WorkerFrame::Cancel { .. } | WorkerFrame::Shutdown => {
                    cancel.store(true, Ordering::Relaxed);
                }
                WorkerFrame::Control { action } => {
                    let result = apply_control(&action, trim_epoch, prefetch_enabled);
                    write_frame_best_effort(writer, &WorkerFrame::ControlAck { action, result });
                }
                _ => { /* ignore other frames mid-run */ }
            },
            PollResult::Timeout => return Ok(()),
            PollResult::Eof => {
                cancel.store(true, Ordering::Relaxed);
                return Ok(());
            }
        }
    }
}

enum PollResult {
    // Boxed: control frames got large with record-bearing payloads (stats,
    // loaded identities) and the timeout path must stay tiny.
    Frame(Box<WorkerFrame>),
    Timeout,
    Eof,
}

fn poll_frame(reader: &RefCell<UnixStream>, timeout: Duration) -> Result<PollResult> {
    use std::os::unix::io::AsRawFd;
    let stream = reader.borrow_mut();
    let mut pfd = libc::pollfd {
        fd: stream.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let ms = timeout.as_millis().min(i32::MAX as u128) as i32;
    // SAFETY: poll with one initialized pollfd; the fd is valid while the
    // RefCell borrow is held.
    let rc = unsafe { libc::poll(&mut pfd, 1, ms) };
    if rc < 0 {
        let err = std::io::Error::last_os_error();
        if err.kind() == std::io::ErrorKind::Interrupted {
            return Ok(PollResult::Timeout);
        }
        return Err(TuckmillError::from(err));
    }
    if rc == 0 {
        return Ok(PollResult::Timeout);
    }
    if pfd.revents & (libc::POLLHUP | libc::POLLERR) != 0 && pfd.revents & libc::POLLIN == 0 {
        return Ok(PollResult::Eof);
    }
    // Readable: a full frame is waiting; a short blocking read is safe.
    drop(stream);
    let mut stream = reader.borrow_mut();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(TuckmillError::from)?;
    match WorkerFrame::read_from(&mut *stream)? {
        Some(frame) => Ok(PollResult::Frame(Box::new(frame))),
        None => Ok(PollResult::Eof),
    }
}

fn read_frame_blocking(reader: &RefCell<UnixStream>) -> Result<Option<WorkerFrame>> {
    let mut stream = reader.borrow_mut();
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(TuckmillError::from)?;
    WorkerFrame::read_from(&mut *stream)
}

/// Parse the `--engine <name>` worker argument. Absent means reference-test.
pub fn engine_from_args(args: &[String]) -> Result<EngineKind> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--engine" {
            let value = iter.next().ok_or_else(|| {
                TuckmillError::new(ErrorCode::InvalidRequest, "--engine needs a value")
            })?;
            return match BackendId::parse(value) {
                Some(BackendId::ReferenceTest) => Ok(EngineKind::ReferenceTest),
                Some(BackendId::TuckCore) => Ok(EngineKind::TuckCore),
                Some(BackendId::LlamaCpp) => Ok(EngineKind::LlamaCpp),
                None => Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!("unknown worker engine {value:?}"),
                )),
            };
        }
    }
    Ok(EngineKind::ReferenceTest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{hello, WorkerFrame};
    use tuckmill_backend::ArtifactRef;
    use tuckmill_manifest::ids::Id;
    use tuckmill_manifest::plan::{ExecutionMode, ExecutionPlan, PlanContext};

    fn admitted_plan() -> ExecutionPlan {
        let mut p = ExecutionPlan::new_skeleton(
            Id::new("a".repeat(64)).unwrap(),
            Id::new("b".repeat(64)).unwrap(),
            "qwen3".into(),
        );
        p.mode = ExecutionMode::Resident;
        p.context = PlanContext::new(256, 64);
        p.plan_id = p.compute_id().unwrap();
        p
    }

    fn artifact() -> ArtifactRef {
        ArtifactRef {
            path: "/nonexistent/synthetic".into(),
            file_bytes: 0,
            metadata_hash: "0".repeat(64),
            tensor_index_hash: "0".repeat(64),
            tokenizer_hash: None,
            arch_id: Some("qwen3".into()),
            content_hash: None,
        }
    }

    fn connect(dir: &Path, kind: EngineKind) -> (UnixStream, std::thread::JoinHandle<Result<()>>) {
        let socket = dir.join("w.sock");
        let socket2 = socket.clone();
        let handle = std::thread::spawn(move || run_worker(&socket2, kind));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if let Ok(s) = UnixStream::connect(&socket) {
                break (s, handle);
            }
            assert!(std::time::Instant::now() < deadline, "worker never bound");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn engine_arg_parsing_is_strict() {
        assert_eq!(engine_from_args(&[]).unwrap(), EngineKind::ReferenceTest);
        assert_eq!(
            engine_from_args(&["--engine".into(), "tuckcore".into()]).unwrap(),
            EngineKind::TuckCore
        );
        assert_eq!(
            engine_from_args(&["--engine".into(), "llama".into()]).unwrap(),
            EngineKind::LlamaCpp
        );
        assert!(engine_from_args(&["--engine".into()]).is_err());
        assert!(engine_from_args(&["--engine".into(), "nope".into()]).is_err());
    }

    #[test]
    fn worker_serves_handshake_load_generate_and_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let (mut stream, handle) = connect(dir.path(), EngineKind::ReferenceTest);

        hello("test").write_to(&mut stream).unwrap();
        match read_frame_blocking_client(&mut stream) {
            WorkerFrame::HelloAck {
                engine, certified, ..
            } => {
                assert_eq!(engine, crate::engine::REFERENCE_TEST_ENGINE);
                assert!(!certified);
            }
            other => panic!("expected HelloAck, got {other:?}"),
        }

        WorkerFrame::Load {
            plan: Box::new(admitted_plan()),
            artifact: artifact(),
            options: None,
        }
        .write_to(&mut stream)
        .unwrap();
        match read_frame_blocking_client(&mut stream) {
            WorkerFrame::Loaded { note, .. } => assert!(note.contains("synthetic")),
            other => panic!("expected Loaded, got {other:?}"),
        }

        // Generate 5 tokens, cancel in the middle of the run.
        WorkerFrame::Generate {
            request_id: "r1".into(),
            prompt_tokens: vec![1, 2, 3, 4],
            max_output_tokens: 50,
            stop: vec![],
            seed: 1,
        }
        .write_to(&mut stream)
        .unwrap();
        let mut tokens = 0u64;
        let mut sent_cancel = false;
        loop {
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            match WorkerFrame::read_from(&mut stream).unwrap().unwrap() {
                WorkerFrame::Token { .. } => {
                    tokens += 1;
                    if tokens >= 3 && !sent_cancel {
                        WorkerFrame::Cancel {
                            request_id: "r1".into(),
                        }
                        .write_to(&mut stream)
                        .unwrap();
                        sent_cancel = true;
                    }
                }
                WorkerFrame::Done {
                    reason, committed, ..
                } => {
                    assert_eq!(reason, "cancelled");
                    assert_eq!(committed, tokens);
                    assert!(tokens < 50, "cancel must stop before the full run");
                    break;
                }
                WorkerFrame::State { .. } | WorkerFrame::Heartbeat { .. } => {}
                other => panic!("unexpected frame {other:?}"),
            }
        }

        // Stats carry backend statistics.
        WorkerFrame::Stats.write_to(&mut stream).unwrap();
        match read_frame_blocking_client(&mut stream) {
            WorkerFrame::StatsAck { backend, .. } => {
                let backend = backend.expect("stats include backend data");
                assert_eq!(
                    backend.logical_read_bytes, 0,
                    "synthetic engine reads nothing"
                );
            }
            other => panic!("expected StatsAck, got {other:?}"),
        }

        // Control frames are applied locally and acknowledged honestly.
        WorkerFrame::Control {
            action: "trim-caches".into(),
        }
        .write_to(&mut stream)
        .unwrap();
        match read_frame_blocking_client(&mut stream) {
            WorkerFrame::ControlAck { action, result } => {
                assert_eq!(action, "trim-caches");
                assert!(result.contains("queued"), "{result}");
            }
            other => panic!("expected ControlAck, got {other:?}"),
        }

        // Scoring is refused by the synthetic engine with a typed error.
        WorkerFrame::Score {
            request_id: "s1".into(),
            tokens: vec![1, 2, 3, 4],
            positions: vec![3],
        }
        .write_to(&mut stream)
        .unwrap();
        match read_frame_blocking_client(&mut stream) {
            WorkerFrame::Error { code, .. } => assert_eq!(code, "UNAVAILABLE"),
            other => panic!("expected Error, got {other:?}"),
        }

        // Chunked score frames: a 152k-logit row must serialize in bounded
        // frames (the Stage A vocabulary bug this split fixes).
        let big: Vec<f32> = (0..152_000).map(|i| i as f32 * 1e-3).collect();
        let chunks: Vec<WorkerFrame> = big
            .chunks(SCORE_CHUNK_FLOATS)
            .enumerate()
            .map(|(i, c)| WorkerFrame::ScoreChunk {
                position: 7,
                offset: (i * SCORE_CHUNK_FLOATS) as u32,
                logits: c.to_vec(),
            })
            .collect();
        assert!(chunks.len() >= 9, "152k floats split into >=9 frames");
        for frame in &chunks {
            let encoded = frame.encode().unwrap();
            assert!(
                encoded.len() as u32 <= crate::proto::MAX_FRAME_BYTES,
                "frame of {} bytes must stay under the cap",
                encoded.len()
            );
        }
        let total: usize = chunks
            .iter()
            .map(|f| match f {
                WorkerFrame::ScoreChunk { logits, .. } => logits.len(),
                _ => 0,
            })
            .sum();
        assert_eq!(total, big.len(), "no logits lost in chunking");

        WorkerFrame::Shutdown.write_to(&mut stream).unwrap();
        match WorkerFrame::read_from(&mut stream).unwrap() {
            Some(WorkerFrame::Bye) | None => {}
            other => panic!("expected Bye, got {other:?}"),
        }
        handle.join().unwrap().unwrap();
    }

    fn read_frame_blocking_client(stream: &mut UnixStream) -> WorkerFrame {
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        loop {
            match WorkerFrame::read_from(stream).unwrap() {
                Some(WorkerFrame::Heartbeat { .. }) => continue,
                Some(frame) => return frame,
                None => panic!("worker closed unexpectedly"),
            }
        }
    }

    #[test]
    fn control_commands_reach_the_engine_during_generation() {
        let dir = tempfile::tempdir().unwrap();
        let (mut stream, handle) = connect(dir.path(), EngineKind::ReferenceTest);
        hello("test").write_to(&mut stream).unwrap();
        let _ = read_frame_blocking_client(&mut stream);
        WorkerFrame::Load {
            plan: Box::new(admitted_plan()),
            artifact: artifact(),
            options: None,
        }
        .write_to(&mut stream)
        .unwrap();
        let _ = read_frame_blocking_client(&mut stream);

        // Long synthetic run; send a control command mid-stream and expect a
        // ControlAck delivered while generation is still in flight.
        WorkerFrame::Generate {
            request_id: "r2".into(),
            prompt_tokens: vec![1, 2],
            max_output_tokens: 60,
            stop: vec![],
            seed: 0,
        }
        .write_to(&mut stream)
        .unwrap();
        let mut tokens = 0u64;
        let mut acked = false;
        WorkerFrame::Control {
            action: "trim-caches".into(),
        }
        .write_to(&mut stream)
        .unwrap();
        loop {
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            match WorkerFrame::read_from(&mut stream).unwrap().unwrap() {
                WorkerFrame::Token { .. } => tokens += 1,
                WorkerFrame::ControlAck { action, result } => {
                    assert_eq!(action, "trim-caches");
                    assert!(result.contains("queued"));
                    acked = true;
                }
                WorkerFrame::Done { reason, .. } => {
                    assert_eq!(reason, "length");
                    assert!(acked, "control ack must arrive during generation");
                    assert!(tokens >= 60, "acks do not disturb token flow");
                    break;
                }
                WorkerFrame::State { .. } | WorkerFrame::Heartbeat { .. } => {}
                other => panic!("unexpected frame {other:?}"),
            }
        }
        WorkerFrame::Shutdown.write_to(&mut stream).unwrap();
        let _ = WorkerFrame::read_from(&mut stream);
        handle.join().unwrap().unwrap();
    }

    #[test]
    fn worker_requires_hello_first() {
        let dir = tempfile::tempdir().unwrap();
        let (mut stream, handle) = connect(dir.path(), EngineKind::ReferenceTest);
        WorkerFrame::Stats.write_to(&mut stream).unwrap();
        // The worker must close/refuse rather than continue.
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let result = WorkerFrame::read_from(&mut stream);
        assert!(result.is_err() || result.unwrap().is_none());
        let _ = handle.join();
    }
}
