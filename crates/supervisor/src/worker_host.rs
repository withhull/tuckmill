//! Worker host: serves a selected inference backend over the framed protocol
//! on a private local socket. This runs inside the dedicated worker process;
//! the supervisor never computes tensors or holds model state.
//!
//! The control channel doubles as the cancellation path: the worker polls the
//! socket between committed tokens, so a cancel is honored at a token
//! boundary rather than at process teardown.

use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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

fn serve(mut stream: UnixStream, kind: EngineKind) -> Result<()> {
    let started = Instant::now();
    let mut engine = make_backend(kind);
    let mut loaded = false;

    // Handshake: the first frame must be Hello.
    let first = read_frame_blocking(&mut stream)?;
    match first {
        Some(WorkerFrame::Hello {
            protocol,
            abi,
            client_build: _,
        }) => {
            validate_handshake(protocol, abi)?;
            let identity = engine.identity();
            hello_ack(
                &crate::engine::build_id(),
                &identity.engine,
                identity.certified,
            )
            .write_to(&mut stream)?;
        }
        Some(_) => {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                "first frame was not Hello",
            ))
        }
        None => return Ok(()), // supervisor went away before handshake
    }

    let cancel = Arc::new(AtomicBool::new(false));
    let paused = Arc::new(AtomicBool::new(false));
    let mut last_heartbeat = Instant::now();

    loop {
        // Poll for the next control frame; emit heartbeats while idle.
        match poll_frame(&mut stream, POLL_SLICE)? {
            PollResult::Timeout => {
                if last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
                    WorkerFrame::Heartbeat {
                        uptime_ms: started.elapsed().as_millis() as u64,
                    }
                    .write_to(&mut stream)?;
                    last_heartbeat = Instant::now();
                }
                continue;
            }
            PollResult::Eof => {
                engine.shutdown();
                return Ok(());
            }
            PollResult::Frame(frame) => match frame {
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
                        engine.load(&plan, &artifact, options.as_ref())
                    })();
                    match result {
                        Ok(note) => {
                            loaded = true;
                            WorkerFrame::Loaded {
                                note,
                                footprint_bytes: crate::host_footprint_bytes(),
                            }
                            .write_to(&mut stream)?;
                        }
                        Err(e) => {
                            WorkerFrame::Error {
                                code: e.code().as_str().into(),
                                message: e.message().to_string(),
                                action: e.action().to_string(),
                            }
                            .write_to(&mut stream)?;
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
                        WorkerFrame::Error {
                            code: ErrorCode::WorkerFailed.as_str().into(),
                            message: "Generate before Load".into(),
                            action: "Load a plan first".into(),
                        }
                        .write_to(&mut stream)?;
                        continue;
                    }
                    cancel.store(false, Ordering::Relaxed);
                    paused.store(false, Ordering::Relaxed);
                    last_heartbeat = Instant::now();
                    let outcome = generate_streaming(
                        engine.as_mut(),
                        &mut stream,
                        &cancel,
                        &paused,
                        &started,
                        &mut last_heartbeat,
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
                            WorkerFrame::Done {
                                reason: outcome.finish_reason.as_str().to_string(),
                                committed: outcome.committed,
                            }
                            .write_to(&mut stream)?;
                        }
                        Err(e) => {
                            WorkerFrame::Error {
                                code: e.code().as_str().into(),
                                message: e.message().to_string(),
                                action: e.action().to_string(),
                            }
                            .write_to(&mut stream)?;
                        }
                    }
                }
                WorkerFrame::Score {
                    request_id: _,
                    tokens,
                    positions,
                } => {
                    // Scoring can hold one position in compute for a long
                    // time (a real model through the scalar oracle); emit a
                    // heartbeat before each frame so supervisor liveness
                    // tracking stays honest without loosening the timeout.
                    if !loaded {
                        WorkerFrame::Error {
                            code: ErrorCode::WorkerFailed.as_str().into(),
                            message: "Score before Load".into(),
                            action: "Load a plan first".into(),
                        }
                        .write_to(&mut stream)?;
                        continue;
                    }
                    let bounded = tokens.len() <= MAX_SCORE_TOKENS
                        && positions.len() <= MAX_SCORE_POSITIONS
                        && positions.iter().all(|p| (*p as usize) < tokens.len());
                    if !bounded {
                        WorkerFrame::Error {
                            code: ErrorCode::InvalidRequest.as_str().into(),
                            message: format!(
                                "scoring request outside bounds (max {MAX_SCORE_TOKENS} tokens, {MAX_SCORE_POSITIONS} positions, positions within the token sequence)"
                            ),
                            action: "Reduce the scoring request; the evaluation interface is deliberately bounded.".into(),
                        }
                        .write_to(&mut stream)?;
                        continue;
                    }
                    let mut write_failed = false;
                    let result = engine.score(&tokens, &positions, &mut |position, logits| {
                        let heartbeat = WorkerFrame::Heartbeat {
                            uptime_ms: started.elapsed().as_millis() as u64,
                        };
                        if heartbeat.write_to(&mut stream).is_err() {
                            write_failed = true;
                            return false;
                        }
                        for (i, chunk) in logits.chunks(SCORE_CHUNK_FLOATS).enumerate() {
                            let frame = WorkerFrame::ScoreChunk {
                                position,
                                offset: (i * SCORE_CHUNK_FLOATS) as u32,
                                logits: chunk.to_vec(),
                            };
                            if frame.write_to(&mut stream).is_err() {
                                write_failed = true;
                                return false;
                            }
                        }
                        true
                    });
                    match result {
                        Ok(scored) => {
                            WorkerFrame::ScoreDone { scored }.write_to(&mut stream)?;
                        }
                        Err(e) => {
                            WorkerFrame::Error {
                                code: e.code().as_str().into(),
                                message: e.message().to_string(),
                                action: e.action().to_string(),
                            }
                            .write_to(&mut stream)?;
                        }
                    }
                    let _ = write_failed;
                }
                WorkerFrame::Cancel { .. } => {
                    cancel.store(true, Ordering::Relaxed);
                }
                WorkerFrame::Stats => {
                    WorkerFrame::StatsAck {
                        footprint_bytes: crate::host_footprint_bytes(),
                        uptime_ms: started.elapsed().as_millis() as u64,
                        backend: Some(engine.stats()),
                    }
                    .write_to(&mut stream)?;
                }
                WorkerFrame::Shutdown => {
                    engine.shutdown();
                    WorkerFrame::Bye.write_to(&mut stream)?;
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

#[allow(clippy::too_many_arguments)]
fn generate_streaming(
    engine: &mut dyn InferenceBackend,
    stream: &mut UnixStream,
    cancel: &Arc<AtomicBool>,
    paused: &Arc<AtomicBool>,
    started: &Instant,
    last_heartbeat: &mut Instant,
    req: crate::engine::GenerateRequest,
) -> Result<crate::engine::GenOutcome> {
    // Control frames may arrive mid-generation; the backend only sees the
    // shared flags, so we interleave a lightweight poll before each token.
    // State transitions are coalesced through a cell and flushed by the token
    // path (a single closure owns the socket, keeping the borrow graph simple).
    let mut outcome_slot: Option<TuckmillError> = None;
    let pending_state: std::cell::RefCell<Option<RunState>> = std::cell::RefCell::new(None);
    let result = {
        let mut on_token = |index: u64, text: &str| -> bool {
            // Interleave: check inbound control frames, heartbeat, emit.
            if let Err(e) = drain_control(stream, cancel, paused) {
                outcome_slot = Some(e);
                return false;
            }
            if let Some(state) = pending_state.borrow_mut().take() {
                let _ = WorkerFrame::State {
                    state: state.as_str().to_string(),
                }
                .write_to(stream);
            }
            if last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
                let heartbeat = WorkerFrame::Heartbeat {
                    uptime_ms: started.elapsed().as_millis() as u64,
                };
                if heartbeat.write_to(stream).is_err() {
                    outcome_slot = Some(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        "supervisor connection lost while streaming",
                    ));
                    return false;
                }
                *last_heartbeat = Instant::now();
            }
            if (WorkerFrame::Token {
                index,
                text: text.to_string(),
            })
            .write_to(stream)
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
            *pending_state.borrow_mut() = Some(state);
        };
        let mut ctl = GenControl {
            cancel: cancel.clone(),
            paused: paused.clone(),
            on_token: &mut on_token,
            on_state: &mut on_state,
        };
        engine.generate(&req, &mut ctl)
    };
    if let Some(err) = outcome_slot {
        return Err(err);
    }
    result
}

/// Non-blocking check for Cancel/Shutdown between tokens.
fn drain_control(
    stream: &mut UnixStream,
    cancel: &Arc<AtomicBool>,
    _paused: &Arc<AtomicBool>,
) -> Result<()> {
    loop {
        match poll_frame(stream, Duration::ZERO)? {
            PollResult::Frame(WorkerFrame::Cancel { .. }) => {
                cancel.store(true, Ordering::Relaxed);
            }
            PollResult::Frame(WorkerFrame::Shutdown) => {
                cancel.store(true, Ordering::Relaxed);
            }
            PollResult::Frame(_) => { /* ignore other frames mid-run */ }
            PollResult::Timeout => return Ok(()),
            PollResult::Eof => {
                cancel.store(true, Ordering::Relaxed);
                return Ok(());
            }
        }
    }
}

enum PollResult {
    Frame(WorkerFrame),
    Timeout,
    Eof,
}

fn poll_frame(stream: &mut UnixStream, timeout: Duration) -> Result<PollResult> {
    use std::os::unix::io::AsRawFd;
    let mut pfd = libc::pollfd {
        fd: stream.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let ms = timeout.as_millis().min(i32::MAX as u128) as i32;
    // SAFETY: poll with one initialized pollfd.
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
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(TuckmillError::from)?;
    match WorkerFrame::read_from(stream)? {
        Some(frame) => Ok(PollResult::Frame(frame)),
        None => Ok(PollResult::Eof),
    }
}

fn read_frame_blocking(stream: &mut UnixStream) -> Result<Option<WorkerFrame>> {
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(TuckmillError::from)?;
    WorkerFrame::read_from(stream)
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
        }
    }

    #[test]
    fn engine_arg_parsing_is_strict() {
        assert_eq!(engine_from_args(&[]).unwrap(), EngineKind::ReferenceTest);
        assert_eq!(
            engine_from_args(&["--engine".into(), "tuckcore".into()]).unwrap(),
            EngineKind::TuckCore
        );
        assert!(engine_from_args(&["--engine".into()]).is_err());
        assert!(engine_from_args(&["--engine".into(), "nope".into()]).is_err());
    }

    #[test]
    fn worker_serves_handshake_load_generate_and_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("w.sock");
        let socket2 = socket.clone();
        let handle = std::thread::spawn(move || run_worker(&socket2, EngineKind::ReferenceTest));

        // Connect with retry while the worker binds.
        let mut stream = {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            loop {
                if let Ok(s) = UnixStream::connect(&socket) {
                    break s;
                }
                assert!(std::time::Instant::now() < deadline, "worker never bound");
                std::thread::sleep(Duration::from_millis(20));
            }
        };

        hello("test").write_to(&mut stream).unwrap();
        match read_frame_blocking(&mut stream).unwrap().unwrap() {
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
        match read_frame_blocking(&mut stream).unwrap().unwrap() {
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
                WorkerFrame::Done { reason, committed } => {
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
        match read_frame_blocking(&mut stream).unwrap().unwrap() {
            WorkerFrame::StatsAck { backend, .. } => {
                let backend = backend.expect("stats include backend data");
                assert_eq!(
                    backend.logical_read_bytes, 0,
                    "synthetic engine reads nothing"
                );
            }
            other => panic!("expected StatsAck, got {other:?}"),
        }

        // Scoring is refused by the synthetic engine with a typed error.
        WorkerFrame::Score {
            request_id: "s1".into(),
            tokens: vec![1, 2, 3, 4],
            positions: vec![3],
        }
        .write_to(&mut stream)
        .unwrap();
        match read_frame_blocking(&mut stream).unwrap().unwrap() {
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

    #[test]
    fn worker_requires_hello_first() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("w.sock");
        let socket2 = socket.clone();
        let handle = std::thread::spawn(move || run_worker(&socket2, EngineKind::ReferenceTest));
        let mut stream = {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            loop {
                if let Ok(s) = UnixStream::connect(&socket) {
                    break s;
                }
                assert!(std::time::Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(20));
            }
        };
        WorkerFrame::Stats.write_to(&mut stream).unwrap();
        // The worker must close/refuse rather than continue.
        let result = read_frame_blocking(&mut stream);
        assert!(result.is_err() || result.unwrap().is_none());
        let _ = handle.join();
    }
}
