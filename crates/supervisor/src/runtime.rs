//! Runtime: admission, queues, worker lifecycle, governor monitoring, and
//! the request path shared by CLI, API, and benchmarks.
//!
//! One active low-memory generation at a time; additional requests queue with
//! a bounded depth. The supervisor never computes tensors: all execution
//! happens inside the worker (isolated process by default).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use tuckmill_backend::{ArtifactRef, BackendStats, EngineKind};
use tuckmill_manifest::adapter::{AdapterRegistry, AdapterSpec};
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ids::{now_unix_ms, secure_random_hex, Id};
use tuckmill_manifest::plan::{ExecutionMode, ExecutionPlan};
use tuckmill_manifest::records::{DeviceProfile, ModelArtifact, SessionRecord};
use tuckmill_manifest::units::Bytes;
use tuckmill_planner::calibration::CalibrationStore;
use tuckmill_planner::kernel_registry::KernelRegistry;
use tuckmill_planner::{build_plan, PlanRequest, PlannerPolicy};
use tuckmill_store::aliases;
use tuckmill_store::layout::StoreLayout;

use crate::admission::{LedgerStats, LiveLedger};

/// Governor command queue shared with the live worker (queued by monitor
/// threads, flushed by the generate/score loops).
pub type ControlQueue = Arc<Mutex<Option<Arc<Mutex<std::collections::VecDeque<String>>>>>>;
use crate::engine::{FinishReason, GenOutcome, ReferenceTestEngine, RunState};
use crate::governor::{GovState, Governor, GovernorAction, GovernorConfig, GovernorSignal};
use crate::log::Logger;
use crate::pressure::{MacSignalSource, SignalSource};
use crate::queue::{QueueTicket, RequestQueue};
use crate::worker::IsolatedWorker;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineMode {
    /// Worker process isolation (default; the real architecture).
    Isolated,
    /// In-process reference engine (tests and debugging only).
    InProcess,
}

pub struct RuntimeOptions {
    pub data_dir: Option<PathBuf>,
    pub engine_mode: EngineMode,
    /// Which backend executes requests (ADR-005).
    pub engine_kind: EngineKind,
    /// Path to the tuckmill binary for worker spawns (defaults to the
    /// current executable).
    pub worker_exe: Option<PathBuf>,
    pub queue_capacity: usize,
    /// Default request budget; None resolves to the hardware-adaptive
    /// suggestion (ADR-006: memory/4, clamped to 2..24 GiB).
    pub default_budget: Option<Bytes>,
    pub default_context: u64,
    pub default_max_output: u64,
    pub policy: PlannerPolicy,
    pub signal_source: Option<Arc<dyn SignalSource>>,
    /// Governor poll interval (2s in normal operation; tests shorten it).
    pub governor_poll: Duration,
    /// Governor thresholds (stability window, dwell).
    pub governor_config: GovernorConfig,
    /// Whether the runtime started a monitoring thread.
    pub monitor: bool,
    /// Explicit acceptance of the reference test engine (synthetic output).
    /// Never enabled implicitly: without it, non-certified profiles are
    /// refused with a typed error (PRD claim policy).
    pub allow_test_engine: bool,
    /// Typed execution options, resolved once by the caller and validated
    /// against the engine at open; the worker's acknowledged settings are
    /// compared against this request after every load, and a substitution
    /// fails closed.
    pub execution: crate::options::ExecutionOptions,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            data_dir: None,
            engine_mode: EngineMode::Isolated,
            engine_kind: EngineKind::ReferenceTest,
            worker_exe: None,
            queue_capacity: 4,
            default_budget: None,
            default_context: 2048,
            default_max_output: 256,
            policy: PlannerPolicy::default(),
            signal_source: None,
            governor_poll: Duration::from_secs(2),
            governor_config: GovernorConfig::default(),
            monitor: true,
            allow_test_engine: false,
            execution: crate::options::ExecutionOptions::default(),
        }
    }
}

/// Events streamed to the client while a request runs.
#[derive(Debug, Clone)]
pub enum RunEvent {
    State(String),
    Token {
        index: u64,
        text: String,
    },
    Done {
        reason: String,
        committed: u64,
        /// Worker-side timing of the request, when the backend reports it.
        timing: Option<tuckmill_backend::RequestTiming>,
    },
    Error {
        code: String,
        message: String,
        action: String,
    },
}

struct ActiveRequest {
    request_id: String,
    cancel: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
    committed: Arc<AtomicU64>,
    state: Arc<Mutex<String>>,
    /// Governor command queue of the live worker, registered after spawn.
    controls: ControlQueue,
}

/// Handle to a started run.
pub struct ActiveRun {
    pub request_id: String,
    pub session_id: Id,
    pub cancel: Arc<AtomicBool>,
    pub pause: Arc<AtomicBool>,
    pub committed: Arc<AtomicU64>,
    pub state: Arc<Mutex<String>>,
    pub events: Receiver<RunEvent>,
}

impl ActiveRun {
    /// Convenience: mark this run cancelled.
    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// What a run was started with; enough to reconstruct the session record.
#[derive(Debug, Clone)]
pub struct RunSpec {
    pub plan: ExecutionPlan,
    pub prompt_tokens: u64,
    pub max_output_tokens: u64,
    pub stop: Vec<String>,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStatus {
    pub governor: String,
    pub queue_depth: usize,
    pub queue_capacity: usize,
    pub active_request: Option<String>,
    pub active_state: Option<String>,
    pub active_committed: u64,
    pub ledger: LedgerStats,
    pub engine_mode: String,
    pub pressure: String,
    pub thermal: String,
    pub device: DeviceProfile,
}

pub struct Runtime {
    pub layout: StoreLayout,
    pub device: DeviceProfile,
    pub registry: AdapterRegistry,
    pub kernels: KernelRegistry,
    pub logger: Logger,
    pub signals: Arc<dyn SignalSource>,
    pub queue: RequestQueue,
    governor: Arc<Mutex<Governor>>,
    calibration: Mutex<CalibrationStore>,
    active: Mutex<Option<ActiveRequest>>,
    /// Live ledger of the current or most recent run.
    ledger: Mutex<Arc<LiveLedger>>,
    engine_mode: EngineMode,
    engine_kind: EngineKind,
    pub(crate) worker_exe: PathBuf,
    in_process: Mutex<ReferenceTestEngine>,
    worker: Mutex<Option<WorkerSlot>>,
    pub policy: PlannerPolicy,
    pub default_budget: Bytes,
    pub default_context: u64,
    pub default_max_output: u64,
    monitor_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    monitor_stop: Arc<AtomicBool>,
    allow_test_engine: bool,
    execution: crate::options::ExecutionOptions,
    /// Worker PID for lock-free monitoring (0 = no worker).
    worker_pid: AtomicU32,
    /// Consecutive footprint-overrun samples (enforcement escalates at 3).
    overrun_streak: AtomicU64,
}

struct WorkerSlot {
    worker: IsolatedWorker,
    plan_id: Id,
    kind: EngineKind,
}

impl Runtime {
    pub fn open(options: RuntimeOptions) -> Result<Arc<Self>> {
        if options.engine_kind != EngineKind::ReferenceTest
            && options.engine_mode == EngineMode::InProcess
        {
            return Err(TuckmillError::new(
                ErrorCode::InvalidConfiguration,
                format!(
                    "{} runs in the isolated worker process; in-process placement is reserved for the reference test engine",
                    options.engine_kind.backend_id().as_str()
                ),
            ));
        }
        options.execution.validate(options.engine_kind)?;
        let layout = match options.data_dir {
            Some(dir) => StoreLayout::open(dir),
            None => StoreLayout::resolve()?,
        };
        layout.ensure()?;
        let signals: Arc<dyn SignalSource> = options
            .signal_source
            .clone()
            .unwrap_or_else(|| Arc::new(MacSignalSource));
        let device = tuckmill_planner::device::detect(layout.root())?;
        let registry = match std::env::var("TUCKMILL_ADAPTERS_DIR") {
            Ok(dir) if !dir.trim().is_empty() => {
                AdapterRegistry::load_dir(std::path::Path::new(&dir))?
            }
            _ => AdapterRegistry::builtin()?,
        };
        let kernels = KernelRegistry::builtin()?;
        let logger = Logger::new(&layout.logs_dir());
        let calibration = CalibrationStore::load(&layout.calibration_path())?;

        let default_budget = options.default_budget.unwrap_or_else(|| {
            tuckmill_planner::budget::suggested_process_budget(device.memory_bytes)
        });
        let runtime = Arc::new(Self {
            layout,
            device,
            registry,
            kernels,
            logger,
            signals,
            queue: RequestQueue::new(options.queue_capacity),
            governor: Arc::new(Mutex::new(Governor::new(options.governor_config.clone()))),
            calibration: Mutex::new(calibration),
            active: Mutex::new(None),
            ledger: Mutex::new(Arc::new(LiveLedger::new(
                default_budget,
                Bytes(options.policy.overhead_estimate_bytes),
                Bytes(options.policy.safety_reserve(default_budget.bytes())),
            ))),
            engine_mode: options.engine_mode,
            engine_kind: options.engine_kind,
            worker_exe: options.worker_exe.unwrap_or_else(|| {
                std::env::current_exe().unwrap_or_else(|_| PathBuf::from("tuckmill"))
            }),
            in_process: Mutex::new(ReferenceTestEngine::new()),
            worker: Mutex::new(None),
            policy: options.policy,
            default_budget,
            default_context: options.default_context,
            default_max_output: options.default_max_output,
            monitor_thread: Mutex::new(None),
            monitor_stop: Arc::new(AtomicBool::new(false)),
            allow_test_engine: options.allow_test_engine,
            execution: options.execution,
            worker_pid: AtomicU32::new(0),
            overrun_streak: AtomicU64::new(0),
        });

        if options.monitor {
            let rt = runtime.clone();
            let stop = runtime.monitor_stop.clone();
            let poll = options.governor_poll;
            let handle = std::thread::Builder::new()
                .name("tuckmill-governor".into())
                .spawn(move || rt.monitor_loop(stop, poll))
                .map_err(TuckmillError::from)?;
            *runtime.monitor_thread.lock().expect("monitor lock") = Some(handle);
        }
        Ok(runtime)
    }

    // ---- discovery / planning ----

    pub fn resolve_artifact(&self, reference: &str) -> Result<(ModelArtifact, AdapterSpec)> {
        let id = aliases::resolve_reference(&self.layout, reference)?;
        let artifact = aliases::load_manifest(&self.layout, &id)?;
        let arch = artifact.arch_id.as_deref().ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                "artifact has no resolved architecture",
            )
        })?;
        let adapter = self
            .registry
            .find(arch)
            .ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::UnsupportedArchitecture,
                    format!("no adapter is registered for architecture {arch:?}"),
                )
            })?
            .clone();
        Ok((artifact, adapter))
    }

    pub fn plan_for(
        &self,
        artifact: &ModelArtifact,
        adapter: &AdapterSpec,
        budget: Bytes,
        context_total: u64,
        max_output: u64,
    ) -> Result<ExecutionPlan> {
        let calibration = self.calibration.lock().expect("calibration lock");
        build_plan(
            &self.kernels,
            artifact,
            adapter,
            &self.device,
            &PlanRequest {
                context: tuckmill_manifest::plan::PlanContext::new(context_total, max_output),
                requested_budget: budget,
                policy: self.policy.clone(),
            },
            &calibration,
        )
    }

    /// Any capacity rejection becomes a typed error for run paths.
    pub fn plan_admitted(&self, plan: &ExecutionPlan) -> Result<()> {
        match plan.mode {
            ExecutionMode::Rejected => {
                let rejection = plan.rejection.as_ref();
                let code = rejection
                    .and_then(|r| {
                        serde_json::from_value::<ErrorCode>(serde_json::Value::String(
                            r.code.clone(),
                        ))
                        .ok()
                    })
                    .unwrap_or(ErrorCode::InvalidPlan);
                Err(TuckmillError::new(
                    code,
                    rejection
                        .map(|r| r.message.clone())
                        .unwrap_or_else(|| "plan was rejected".into()),
                ))
            }
            _ => Ok(()),
        }
    }

    pub fn check_plan_fresh(&self, plan: &ExecutionPlan) -> Result<()> {
        plan.validate()?;
        if plan.device_id != self.device.device_id {
            return Err(TuckmillError::new(
                ErrorCode::PlanStale,
                "plan was built for a different device profile (OS, memory, or storage changed)",
            ));
        }
        Ok(())
    }

    pub fn calibration_store(&self) -> std::sync::MutexGuard<'_, CalibrationStore> {
        self.calibration.lock().expect("calibration lock")
    }

    pub fn governor_state(&self) -> GovState {
        self.governor.lock().expect("governor lock").state()
    }

    pub fn status(&self) -> RuntimeStatus {
        let active = self.active.lock().expect("active lock");
        let ledger = self.ledger.lock().expect("ledger lock").clone();
        RuntimeStatus {
            governor: self.governor_state().as_str().to_string(),
            queue_depth: self.queue.depth(),
            queue_capacity: self.queue.capacity(),
            active_request: active.as_ref().map(|a| a.request_id.clone()),
            active_state: active
                .as_ref()
                .map(|a| a.state.lock().expect("state lock").clone()),
            active_committed: active
                .as_ref()
                .map(|a| a.committed.load(Ordering::Relaxed))
                .unwrap_or(0),
            ledger: ledger.stats(),
            engine_mode: crate::runtime::engine_mode_label(self.engine_mode).into(),
            pressure: self.signals.pressure().as_str(),
            thermal: format!("{:?}", self.signals.thermal()).to_lowercase(),
            device: self.device.clone(),
        }
    }

    /// Gate execution per selected backend:
    /// - reference-test requires explicit `--allow-test-engine` because its
    ///   output is synthetic;
    /// - tuckcore is a real executor (weights are executed) and the explicit
    ///   `--engine tuckcore` selection is the acceptance for its
    ///   uncertified, experimental kernels; every response labels it;
    /// - a certified engine needs no gate.
    pub fn admit_engine(&self, plan: &ExecutionPlan) -> Result<()> {
        use tuckmill_manifest::plan::EngineAvailability;
        match self.engine_kind {
            EngineKind::TuckCore | EngineKind::LlamaCpp => Ok(()),
            EngineKind::ReferenceTest => match plan.engine.availability {
                EngineAvailability::Certified if self.engine_kind == EngineKind::ReferenceTest => {
                    Ok(())
                }
                _ if self.allow_test_engine => Ok(()),
                _ => Err(TuckmillError::new(
                    ErrorCode::Unavailable,
                    format!(
                        "no certified execution engine exists for this profile in this build (engine would be {:?}); refusing to present synthetic output as model output",
                        plan.engine.engine
                    ),
                )
                .with_action(
                    "Re-run with --allow-test-engine for pipeline validation, or --engine tuckcore to execute the artifact with the experimental scalar executor.",
                )),
            },
        }
    }

    pub fn engine_kind(&self) -> EngineKind {
        self.engine_kind
    }

    /// Measured backend statistics from the live worker, when one exists.
    pub fn backend_stats(&self) -> Option<BackendStats> {
        let mut slot = self.worker.lock().ok()?;
        let slot = slot.as_mut()?;
        match slot.worker.stats() {
            Ok((_, _, backend)) => backend,
            Err(_) => None,
        }
    }

    pub fn allows_test_engine(&self) -> bool {
        self.allow_test_engine
    }

    pub fn engine_mode_label_str(&self) -> &'static str {
        engine_mode_label(self.engine_mode)
    }

    /// Identity of the engine that would serve the next request. Falls back
    /// to the selected backend's declared identity before any worker exists.
    pub fn engine_identity(&self) -> tuckmill_manifest::records::EngineIdentity {
        if let Ok(slot) = self.worker.lock() {
            if let Some(s) = slot.as_ref() {
                let mut id = crate::engine::reference_test_identity();
                id.engine = s.worker.engine.clone();
                id.certified = s.worker.certified;
                id.build_id = s.worker.worker_build.clone();
                if !s.worker.resolved_kernels.is_empty() {
                    id.kernels = s.worker.resolved_kernels.clone();
                }
                return id;
            }
        }
        match self.engine_kind {
            EngineKind::ReferenceTest => crate::engine::reference_test_identity(),
            EngineKind::TuckCore => tuckmill_tuckcore::TuckCore::new().identity(),
            EngineKind::LlamaCpp => tuckmill_llamaref::identity(),
        }
    }

    /// The selected backend id (pre-spawn safe).
    pub fn selected_backend(&self) -> tuckmill_backend::BackendId {
        self.engine_kind.backend_id()
    }

    /// PID of the live worker, when one is running. Lock-free so the
    /// governor monitor never waits on in-flight generation.
    pub fn worker_pid(&self) -> Option<u32> {
        let pid = self.worker_pid.load(Ordering::Relaxed);
        (pid != 0).then_some(pid)
    }

    /// Cancel the currently active run, if any. Returns whether one was found.
    pub fn cancel_active(&self) -> bool {
        let flags = {
            let active = self.active.lock().expect("active lock");
            active.as_ref().map(|a| (a.cancel.clone(), a.pause.clone()))
        };
        match flags {
            Some((cancel, pause)) => {
                cancel.store(true, Ordering::Relaxed);
                pause.store(false, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }

    // ---- run path ----

    pub fn submit(&self) -> Result<QueueTicket> {
        self.queue.submit()
    }

    /// Start a run: waits for the queue turn, reserves the ledger, records a
    /// session, and returns a streaming handle. The caller consumes events
    /// until Done/Error or drops the handle (which cancels via backpressure).
    #[allow(clippy::too_many_arguments)]
    pub fn start_run(
        self: &Arc<Self>,
        ticket: &QueueTicket,
        plan: ExecutionPlan,
        artifact: ArtifactRef,
        prompt_tokens: Vec<u32>,
        max_output_tokens: u64,
        stop: Vec<String>,
        seed: u64,
    ) -> Result<ActiveRun> {
        self.check_plan_fresh(&plan)?;
        self.plan_admitted(&plan)?;
        if self.governor_state() == GovState::Failed {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                "governor is in the failed state; restart the service",
            ));
        }

        // Wait until our ticket reaches the front. The ticket stays in the
        // queue until the active slot is free, so one active generation is
        // enforced by the queue itself.
        if !self
            .queue
            .wait_until_front(ticket, Duration::from_secs(120))?
        {
            return Err(TuckmillError::new(
                ErrorCode::Cancelled,
                "service is shutting down",
            ));
        }

        let request_id = format!("req_{}", secure_random_hex(6)?);
        let session_id = Id::new(tuckmill_manifest::ids::sha256_hex(request_id.as_bytes()))?;

        // Ledger: reserve the plan's categories. Strict R+U+H<=B.
        let ledger = Arc::new(LiveLedger::new(
            plan.budget.requested_budget,
            plan.budget.overhead_estimate,
            plan.budget.safety_reserve,
        ));
        let rb = &plan.budget.reservation_breakdown;
        let mut handles = Vec::new();
        for (class, bytes) in [
            (
                tuckmill_manifest::records::ReserveClass::Weights,
                rb.weights.bytes(),
            ),
            (
                tuckmill_manifest::records::ReserveClass::KvState,
                rb.kv_state.bytes(),
            ),
            (
                tuckmill_manifest::records::ReserveClass::Workspaces,
                rb.workspaces.bytes(),
            ),
            (
                tuckmill_manifest::records::ReserveClass::IoBuffers,
                rb.io_buffers.bytes(),
            ),
            (
                tuckmill_manifest::records::ReserveClass::ExpertCache,
                rb.expert_cache.bytes(),
            ),
        ] {
            if bytes > 0 {
                handles.push(ledger.try_reserve(class, Bytes(bytes), class.as_str())?);
            }
        }
        *self.ledger.lock().expect("ledger lock") = ledger.clone();

        let cancel = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        let committed = Arc::new(AtomicU64::new(0));
        let state = Arc::new(Mutex::new("admitted".to_string()));
        let controls: ControlQueue = Arc::new(Mutex::new(None));

        // Session record (committed state only; no speculative state).
        let session = SessionRecord {
            schema_version: 1,
            session_id: session_id.clone(),
            created_unix_ms: now_unix_ms(),
            updated_unix_ms: now_unix_ms(),
            artifact_id: plan.artifact_id.clone(),
            plan_id: plan.plan_id.clone(),
            committed_tokens: 0,
            last_committed_position: 0,
            finished_reason: None,
        };
        self.save_session(&session)?;

        // Acquire the active slot and claim the front position atomically
        // with respect to other start_run callers. Lock order is always
        // active -> queue.
        {
            let claim_deadline = std::time::Instant::now() + Duration::from_secs(120);
            loop {
                let mut active = self.active.lock().expect("active lock");
                if active.is_none() && self.queue.claim_front(ticket)? {
                    *active = Some(ActiveRequest {
                        request_id: request_id.clone(),
                        cancel: cancel.clone(),
                        pause: pause.clone(),
                        committed: committed.clone(),
                        state: state.clone(),
                        controls: controls.clone(),
                    });
                    break;
                }
                if std::time::Instant::now() >= claim_deadline {
                    self.queue.abandon(ticket);
                    return Err(TuckmillError::new(
                        ErrorCode::QueueFull,
                        "timed out waiting for the active generation slot",
                    ));
                }
                drop(active);
                // The executor thread pokes the queue on completion; this
                // short sleep bounds the handoff latency in any case.
                std::thread::sleep(Duration::from_millis(10));
                self.queue.notify_waiters();
            }
        }

        let (tx, rx) = sync_channel::<RunEvent>(256);
        let runtime = self.clone();
        let plan_for_thread = plan.clone();
        let state_evt = state.clone();
        let committed_evt = committed.clone();
        let cancel_evt = cancel.clone();
        let pause_evt = pause.clone();
        let request_id_evt = request_id.clone();
        let session_id_evt = session_id.clone();
        let artifact_evt = artifact;
        let controls_evt = controls.clone();
        let thread_name = format!("tuckmill-run-{request_id}");
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                let result = runtime.execute_run(
                    &plan_for_thread,
                    &artifact_evt,
                    &request_id_evt,
                    &session_id_evt,
                    &controls_evt,
                    prompt_tokens,
                    max_output_tokens,
                    stop,
                    seed,
                    &cancel_evt,
                    &pause_evt,
                    &committed_evt,
                    &state_evt,
                    &tx,
                    handles,
                );
                if let Err(e) = result {
                    let _ = tx.send(RunEvent::Error {
                        code: e.code().as_str().to_string(),
                        message: e.message().to_string(),
                        action: e.action().to_string(),
                    });
                }
                // Clear active slot and hand off to the next queued ticket.
                {
                    let mut active = runtime.active.lock().expect("active lock");
                    if active
                        .as_ref()
                        .map(|a| a.request_id == request_id_evt)
                        .unwrap_or(false)
                    {
                        *active = None;
                    }
                }
                runtime.queue.notify_waiters();
            })
            .map_err(TuckmillError::from)?;

        Ok(ActiveRun {
            request_id,
            session_id,
            cancel,
            pause,
            committed,
            state,
            events: rx,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_run(
        self: &Arc<Self>,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        request_id: &str,
        session_id: &Id,
        controls: &ControlQueue,
        prompt_tokens: Vec<u32>,
        max_output_tokens: u64,
        stop: Vec<String>,
        seed: u64,
        cancel: &Arc<AtomicBool>,
        pause: &Arc<AtomicBool>,
        committed: &Arc<AtomicU64>,
        state: &Arc<Mutex<String>>,
        tx: &SyncSender<RunEvent>,
        _handles: Vec<crate::admission::ReservationHandle>,
    ) -> Result<GenOutcome> {
        let started = Instant::now();
        let set_state = |s: &str| {
            *state.lock().expect("state lock") = s.to_string();
            let _ = tx.try_send(RunEvent::State(s.to_string()));
        };

        self.logger.info(
            "REQUEST_START",
            [
                ("request_id".to_string(), serde_json::json!(request_id)),
                (
                    "plan_id".to_string(),
                    serde_json::json!(plan.plan_id.as_str()),
                ),
                ("mode".to_string(), serde_json::json!(plan.mode.as_str())),
            ]
            .into_iter()
            .collect(),
        );

        let prompt_len = prompt_tokens.len() as u64;
        match self.engine_mode {
            EngineMode::Isolated => {
                let mut slot = self.worker.lock().expect("worker lock");
                let need_spawn = match slot.as_ref() {
                    Some(w) => w.plan_id != plan.plan_id || w.kind != self.engine_kind,
                    None => true,
                };
                if need_spawn {
                    if let Some(old) = slot.take() {
                        drop(old);
                    }
                    set_state("warming");
                    let mut worker = IsolatedWorker::spawn(
                        &self.worker_exe,
                        &self.layout.tmp_dir(),
                        self.engine_kind,
                    )?;
                    let load_options = self.execution.load_json(self.engine_kind);
                    let note = worker.load(plan, artifact, load_options.as_ref())?;
                    // Resolved settings, acknowledged by the worker right
                    // after load (the worker is idle here): an explicit
                    // request must match exactly or the run is refused.
                    let (_, _, ack_stats) = worker.stats()?;
                    let acknowledged = self.execution.acknowledged(
                        self.engine_kind,
                        &worker.resolved_kernels,
                        ack_stats.as_ref().unwrap_or(&BackendStats::default()),
                    )?;
                    self.logger.info(
                        "WORKER_RESOLVED",
                        [
                            (
                                "kernels".to_string(),
                                serde_json::json!(acknowledged.kernel_impl),
                            ),
                            (
                                "storage".to_string(),
                                serde_json::json!(acknowledged.storage_mode),
                            ),
                            (
                                "numerical_profile".to_string(),
                                serde_json::json!(acknowledged.numerical_profile),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    );
                    self.logger.info(
                        "WORKER_LOADED",
                        [("note".to_string(), serde_json::json!(note))]
                            .into_iter()
                            .collect(),
                    );
                    self.worker_pid.store(worker.pid(), Ordering::Relaxed);
                    *slot = Some(WorkerSlot {
                        plan_id: plan.plan_id.clone(),
                        kind: self.engine_kind,
                        worker,
                    });
                    if let Ok(mut slot_controls) = controls.lock() {
                        *slot_controls =
                            Some(slot.as_ref().expect("slot set above").worker.controls());
                    }
                }
                let WorkerSlot { worker, .. } = slot.as_mut().expect("worker slot");
                let mut token_tx = |index: u64, text: &str| -> bool {
                    committed.store(index + 1, Ordering::Relaxed);
                    *state.lock().expect("state lock") = "generating".to_string();
                    tx.send(RunEvent::Token {
                        index,
                        text: text.to_string(),
                    })
                    .is_ok()
                };
                let mut on_state = |s: RunState| {
                    set_state(s.as_str());
                };
                let outcome = worker.generate(
                    crate::engine::EngineRequest {
                        request_id: request_id.to_string(),
                        prompt_tokens,
                        max_output_tokens,
                        stop,
                        seed,
                    },
                    cancel,
                    pause,
                    &mut token_tx,
                    &mut on_state,
                )?;
                committed.store(outcome.committed, Ordering::Relaxed);
                self.finish_run(
                    request_id, session_id, plan, prompt_len, &outcome, started, tx,
                )?;
                Ok(outcome)
            }
            EngineMode::InProcess => {
                // Placement is validated at open(): in-process is reserved
                // for the reference test engine.
                let mut engine = self.in_process.lock().expect("engine lock");
                engine.per_token_delay = Duration::from_millis(2);
                engine.load(plan)?;
                let req = crate::engine::EngineRequest {
                    request_id: request_id.to_string(),
                    prompt_tokens,
                    max_output_tokens,
                    stop,
                    seed,
                };
                let mut token_tx = |index: u64, text: &str| -> bool {
                    committed.store(index + 1, Ordering::Relaxed);
                    tx.send(RunEvent::Token {
                        index,
                        text: text.to_string(),
                    })
                    .is_ok()
                };
                let mut on_state = |s: RunState| {
                    set_state(s.as_str());
                };
                let mut ctl = crate::engine::GenControl::new(
                    cancel.clone(),
                    pause.clone(),
                    &mut token_tx,
                    &mut on_state,
                );
                let outcome = engine.generate(&req, &mut ctl)?;
                self.finish_run(
                    request_id, session_id, plan, prompt_len, &outcome, started, tx,
                )?;
                Ok(outcome)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_run(
        &self,
        request_id: &str,
        session_id: &Id,
        plan: &ExecutionPlan,
        prompt_len: u64,
        outcome: &GenOutcome,
        started: Instant,
        tx: &SyncSender<RunEvent>,
    ) -> Result<()> {
        // Session persistence: committed position only.
        let session = SessionRecord {
            schema_version: 1,
            session_id: session_id.clone(),
            created_unix_ms: now_unix_ms(),
            updated_unix_ms: now_unix_ms(),
            artifact_id: plan.artifact_id.clone(),
            plan_id: plan.plan_id.clone(),
            committed_tokens: outcome.committed,
            last_committed_position: prompt_len + outcome.committed,
            finished_reason: Some(outcome.finish_reason.as_str().to_string()),
        };
        self.save_session(&session)?;
        self.logger.info(
            "REQUEST_DONE",
            [
                ("request_id".to_string(), serde_json::json!(request_id)),
                (
                    "reason".to_string(),
                    serde_json::json!(outcome.finish_reason.as_str()),
                ),
                (
                    "committed".to_string(),
                    serde_json::json!(outcome.committed),
                ),
                (
                    "elapsed_ms".to_string(),
                    serde_json::json!(started.elapsed().as_millis() as u64),
                ),
            ]
            .into_iter()
            .collect(),
        );
        let _ = tx.send(RunEvent::Done {
            reason: outcome.finish_reason.as_str().to_string(),
            committed: outcome.committed,
            timing: outcome.timing,
        });
        Ok(())
    }

    fn save_session(&self, session: &SessionRecord) -> Result<()> {
        let dir = self.layout.sessions_dir();
        std::fs::create_dir_all(&dir).map_err(TuckmillError::from)?;
        let path = dir.join(format!("{}.json", session.session_id.as_str()));
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(session).map_err(|e| {
            tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "session serialization")
        })?;
        std::fs::write(&tmp, bytes).map_err(TuckmillError::from)?;
        std::fs::rename(&tmp, &path).map_err(TuckmillError::from)?;
        Ok(())
    }

    // ---- governor monitor ----

    fn monitor_loop(self: Arc<Self>, stop: Arc<AtomicBool>, poll: Duration) {
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(poll);
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let now = Instant::now();
            let pressure = self.signals.pressure();
            let thermal = self.signals.thermal();
            let mut actions: Vec<GovernorAction> = Vec::new();
            {
                let mut gov = self.governor.lock().expect("governor lock");
                let prev = gov.state();
                actions.extend(gov.on_signal(GovernorSignal::Pressure(pressure), now));
                if !matches!(gov.state(), GovState::Paused | GovState::Failed) {
                    actions.extend(gov.on_signal(GovernorSignal::Thermal(thermal), now));
                    if matches!(
                        self.signals.pressure(),
                        crate::pressure::PressureLevel::Normal
                    ) && !matches!(gov.state(), GovState::Failed)
                    {
                        actions.extend(gov.on_signal(GovernorSignal::ConditionsImproved, now));
                    }
                }
                if gov.state() != prev {
                    let transitions = gov.transitions().to_vec();
                    if let Some((_, state, why)) = transitions.last() {
                        self.logger.warn(
                            "GOVERNOR_TRANSITION",
                            [
                                ("state".to_string(), serde_json::json!(state.as_str())),
                                ("why".to_string(), serde_json::json!(why)),
                            ]
                            .into_iter()
                            .collect(),
                        );
                    }
                }
            }

            // Short guard: clone the handles we need, then act without the
            // active lock so cancellation and status never queue behind the
            // monitor (Stage A review: no nested active->worker locking).
            let snapshot = {
                let active = self.active.lock().expect("active lock");
                active.as_ref().map(|a| {
                    (
                        a.cancel.clone(),
                        a.pause.clone(),
                        a.committed.clone(),
                        a.state.clone(),
                        a.controls.clone(),
                    )
                })
            };
            if let Some((cancel, pause, committed, state, controls)) = snapshot {
                let push_control = |action: &str| {
                    if let Ok(guard) = controls.lock() {
                        if let Some(queue) = guard.as_ref() {
                            if let Ok(mut queue) = queue.lock() {
                                queue.push_back(action.to_string());
                            }
                        }
                    }
                };
                for action in actions {
                    match action {
                        GovernorAction::PauseComputeAndIo => {
                            pause.store(true, Ordering::Relaxed);
                            *state.lock().expect("state lock") = "paused".into();
                        }
                        GovernorAction::ResumeReads => {
                            pause.store(false, Ordering::Relaxed);
                            *state.lock().expect("state lock") = "generating".into();
                        }
                        GovernorAction::TrimCaches => {
                            push_control("trim-caches");
                            *state.lock().expect("state lock") = "throttling".into();
                        }
                        GovernorAction::StopPrefetchGrowth
                        | GovernorAction::ReduceSubmissionAndSpeculation => {
                            push_control("set-prefetch=off");
                            *state.lock().expect("state lock") = "throttling".into();
                        }
                        GovernorAction::DrainThenRelease | GovernorAction::FailRequest => {
                            cancel.store(true, Ordering::Relaxed);
                            pause.store(false, Ordering::Relaxed);
                        }
                    }
                }
                // Footprint sampling: measured, never conflated with
                // reservations. Enforcement: a sustained overrun trims and
                // then pauses rather than only logging.
                if let Some(pid) = self.worker_pid() {
                    if let Some(fp) = self.signals.footprint_of(pid as i32) {
                        let ledger = self.ledger.lock().expect("ledger lock").clone();
                        if ledger.observe_footprint(fp) {
                            self.logger.warn(
                                "FOOTPRINT_OVERRUN",
                                [
                                    ("measured".to_string(), serde_json::json!(fp)),
                                    (
                                        "committed".to_string(),
                                        serde_json::json!(committed.load(Ordering::Relaxed)),
                                    ),
                                ]
                                .into_iter()
                                .collect(),
                            );
                            push_control("trim-caches");
                            *state.lock().expect("state lock") = "throttling".into();
                            let streak = self.overrun_streak.fetch_add(1, Ordering::Relaxed) + 1;
                            if streak >= 3 {
                                pause.store(true, Ordering::Relaxed);
                                self.logger.warn(
                                    "FOOTPRINT_OVERRUN_ENFORCED",
                                    [("streak".to_string(), serde_json::json!(streak))]
                                        .into_iter()
                                        .collect(),
                                );
                            }
                        } else {
                            self.overrun_streak.store(0, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    }

    /// Stop the monitor thread (used by tests and shutdown).
    pub fn shutdown(&self) {
        self.monitor_stop.store(true, Ordering::Relaxed);
        self.queue.close();
        if let Some(handle) = self.monitor_thread.lock().expect("monitor lock").take() {
            let _ = handle.join();
        }
        if let Some(slot) = self.worker.lock().expect("worker lock").take() {
            drop(slot);
        }
        self.worker_pid.store(0, Ordering::Relaxed);
    }
}

/// Engine completion derived from a run's event stream (used by bench).
pub fn reason_from_str(s: &str) -> FinishReason {
    match s {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "cancelled" => FinishReason::Cancelled,
        _ => FinishReason::Eos,
    }
}

pub fn engine_mode_label(mode: EngineMode) -> &'static str {
    match mode {
        EngineMode::Isolated => "isolated-worker",
        EngineMode::InProcess => "in-process-reference",
    }
}

/// Collect a run synchronously (CLI convenience; API streams instead).
pub fn collect_run(run: &ActiveRun, mut on_token: impl FnMut(u64, &str)) -> (FinishReason, u64) {
    loop {
        match run.events.recv() {
            Ok(RunEvent::State(_)) => {}
            Ok(RunEvent::Token { index, text }) => on_token(index, &text),
            Ok(RunEvent::Done {
                reason,
                committed,
                timing: _,
            }) => return (reason_from_str(&reason), committed),
            Ok(RunEvent::Error { .. }) | Err(_) => {
                return (
                    FinishReason::Cancelled,
                    run.committed.load(Ordering::Relaxed),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pressure::{MockSignalSource, PressureLevel};
    use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

    fn test_runtime(mode: EngineMode) -> (tempfile::TempDir, Arc<Runtime>, Arc<MockSignalSource>) {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        let src = dir.path().join("m.gguf");
        write_synthetic(&src, &SyntheticConfig::tiny_qwen3()).unwrap();
        let registry = AdapterRegistry::builtin().unwrap();
        let import = tuckmill_store::import::import_local(
            &layout,
            &registry,
            &src,
            &tuckmill_store::import::ImportOptions {
                alias: Some("local".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(import.artifact.geometry.is_some());

        let signals = Arc::new(MockSignalSource::new());
        let runtime = Runtime::open(RuntimeOptions {
            data_dir: Some(layout.root().to_path_buf()),
            engine_mode: mode,
            engine_kind: EngineKind::ReferenceTest,
            worker_exe: None,
            queue_capacity: 2,
            execution: crate::options::ExecutionOptions::default(),
            default_budget: Some(Bytes::parse("2GiB").unwrap()),
            default_context: 2048,
            default_max_output: 256,
            policy: PlannerPolicy::default(),
            signal_source: Some(signals.clone()),
            governor_poll: Duration::from_millis(30),
            governor_config: GovernorConfig {
                stability_interval: Duration::from_millis(150),
                min_dwell: Duration::from_millis(20),
            },
            monitor: true,
            allow_test_engine: true,
        })
        .unwrap();
        (dir, runtime, signals)
    }

    /// Tests read the artifact record back from the store each time; keeping
    /// it out of the Runtime avoids widening the production API for tests.
    fn artifact_ref_for(runtime: &Runtime) -> tuckmill_backend::ArtifactRef {
        let id = aliases::resolve_reference(&runtime.layout, "local").unwrap();
        let artifact = aliases::load_manifest(&runtime.layout, &id).unwrap();
        tuckmill_backend::ArtifactRef::from_record(
            &artifact,
            &runtime.layout.artifact_payload_path(&artifact.artifact_id),
        )
    }

    fn run_to_completion(
        runtime: &Arc<Runtime>,
        prompt_len: usize,
        max_out: u64,
    ) -> (Vec<u8>, String, u64) {
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();
        let plan = runtime
            .plan_for(
                &artifact,
                &adapter,
                Bytes::parse("2GiB").unwrap(),
                2048,
                256,
            )
            .unwrap();
        runtime.plan_admitted(&plan).unwrap();
        let artifact = artifact_ref_for(runtime);
        let ticket = runtime.submit().unwrap();
        let run = runtime
            .start_run(
                &ticket,
                plan,
                artifact,
                (0..prompt_len as u32).collect(),
                max_out,
                vec![],
                42,
            )
            .unwrap();
        let mut text = Vec::new();
        let (reason, committed) = collect_run(&run, |_i, t| text.extend_from_slice(t.as_bytes()));
        (text, reason.as_str().to_string(), committed)
    }

    #[test]
    fn end_to_end_run_in_process() {
        let (_dir, runtime, _signals) = test_runtime(EngineMode::InProcess);
        let (text, reason, committed) = run_to_completion(&runtime, 8, 5);
        assert_eq!(reason, "length");
        assert_eq!(committed, 5);
        let s = String::from_utf8(text).unwrap();
        assert!(s.starts_with("[ref:0]"));
        // Session saved with committed position.
        let sessions: Vec<_> = std::fs::read_dir(runtime.layout.sessions_dir())
            .unwrap()
            .flatten()
            .collect();
        assert_eq!(sessions.len(), 1);
        runtime.shutdown();
    }

    #[test]
    fn queued_second_request_gets_full_run() {
        let (_dir, runtime, _signals) = test_runtime(EngineMode::InProcess);
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();
        let plan = runtime
            .plan_for(&artifact, &adapter, Bytes::parse("2GiB").unwrap(), 2048, 64)
            .unwrap();
        let ticket_a = runtime.submit().unwrap();
        let ticket_b = runtime.submit().unwrap();
        let artifact_a = artifact_ref_for(&runtime);
        let artifact_b = artifact_a.clone();
        let run_a = runtime
            .start_run(
                &ticket_a,
                plan.clone(),
                artifact_a,
                vec![1, 2],
                8,
                vec![],
                1,
            )
            .unwrap();
        let runtime2 = runtime.clone();
        let handle = std::thread::spawn(move || {
            runtime2.start_run(&ticket_b, plan, artifact_b, vec![1, 2], 8, vec![], 2)
        });
        let (reason_a, committed_a) = collect_run(&run_a, |_, _| {});
        assert_eq!(reason_a, FinishReason::Length);
        assert_eq!(committed_a, 8);
        let run_b = handle.join().unwrap().unwrap();
        let (reason_b, committed_b) = collect_run(&run_b, |_, _| {});
        assert_eq!(reason_b, FinishReason::Length);
        assert_eq!(committed_b, 8);
        runtime.shutdown();
    }

    #[test]
    fn governor_pauses_under_critical_pressure_and_resumes_after_stability() {
        let (_dir, runtime, signals) = test_runtime(EngineMode::InProcess);
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();
        let plan = runtime
            .plan_for(
                &artifact,
                &adapter,
                Bytes::parse("2GiB").unwrap(),
                2048,
                300,
            )
            .unwrap();
        let ticket = runtime.submit().unwrap();
        let artifact = artifact_ref_for(&runtime);
        // 300 tokens at ~2ms = ~0.6s of work; plenty of time to drive the
        // governor across a pause and a resume.
        let run = runtime
            .start_run(&ticket, plan, artifact, vec![1, 2, 3], 300, vec![], 0)
            .unwrap();

        let mut saw_paused = false;
        let mut saw_resumed = false;
        let mut tokens_seen = 0u64;
        let mut drove_critical = false;
        loop {
            let ev = run.events.recv_timeout(Duration::from_secs(10)).unwrap();
            match ev {
                RunEvent::Token { .. } => {
                    tokens_seen += 1;
                    if tokens_seen == 5 && !drove_critical {
                        drove_critical = true;
                        signals.set_pressure(PressureLevel::Critical);
                    }
                }
                RunEvent::State(s) if s == "paused" => {
                    saw_paused = true;
                    // Improve conditions; the governor resumes only after the
                    // configured stability window.
                    signals.set_pressure(PressureLevel::Normal);
                }
                RunEvent::State(_) => {
                    if saw_paused && tokens_seen > 5 {
                        saw_resumed = true;
                    }
                }
                RunEvent::Done { committed, .. } => {
                    assert_eq!(committed, 300);
                    break;
                }
                RunEvent::Error { message, .. } => panic!("run failed: {message}"),
            }
        }
        assert!(saw_paused, "governor must pause under critical pressure");
        assert!(
            saw_resumed,
            "governor must resume after the stability window"
        );
        assert_eq!(runtime.governor_state(), GovState::Normal);
        runtime.shutdown();
    }

    #[test]
    fn control_paths_never_wait_on_the_worker_lock() {
        // Stage A review regression: the governor monitor, status(), and
        // cancellation must all make progress while a generation holds the
        // worker lock for its whole duration.
        let (_dir, runtime, _signals) = test_runtime(EngineMode::InProcess);
        let runtime2 = runtime.clone();
        let held = std::thread::spawn(move || {
            let _guard = runtime2.worker.lock().expect("hold worker lock");
            std::thread::sleep(Duration::from_millis(600));
        });
        std::thread::sleep(Duration::from_millis(50));
        let t0 = std::time::Instant::now();
        let _ = runtime.cancel_active();
        let _ = runtime.status();
        let _ = runtime.worker_pid();
        assert!(
            t0.elapsed() < Duration::from_millis(150),
            "control paths must not block on the worker lock (took {:?})",
            t0.elapsed()
        );
        held.join().expect("holder thread");
        runtime.shutdown();
    }

    #[test]
    fn governor_throttling_reaches_the_active_run_state() {
        let (_dir, runtime, signals) = test_runtime(EngineMode::InProcess);
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();
        let plan = runtime
            .plan_for(
                &artifact,
                &adapter,
                Bytes::parse("2GiB").unwrap(),
                2048,
                300,
            )
            .unwrap();
        let ticket = runtime.submit().unwrap();
        let artifact_ref = artifact_ref_for(&runtime);
        let run = runtime
            .start_run(&ticket, plan, artifact_ref, vec![1, 2, 3], 300, vec![], 0)
            .unwrap();
        signals.set_pressure(crate::pressure::PressureLevel::Warning);
        let mut saw_throttling = false;
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if runtime.status().active_state.as_deref() == Some("throttling") {
                saw_throttling = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            saw_throttling,
            "warning pressure must throttle the active run"
        );
        let _ = run;
        runtime.shutdown();
    }

    #[test]
    fn generic_api_usage_is_typed() {
        let (_dir, runtime, _signals) = test_runtime(EngineMode::InProcess);
        let err = runtime.resolve_artifact("missing").unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidRequest);
        runtime.shutdown();
    }

    #[test]
    fn stale_plan_is_rejected() {
        let (_dir, runtime, _signals) = test_runtime(EngineMode::InProcess);
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();
        let mut plan = runtime
            .plan_for(
                &artifact,
                &adapter,
                Bytes::parse("2GiB").unwrap(),
                2048,
                256,
            )
            .unwrap();
        plan.device_id = Id::new("9".repeat(64)).unwrap();
        plan.plan_id = plan.compute_id().unwrap();
        let err = runtime.check_plan_fresh(&plan).unwrap_err();
        assert_eq!(err.code(), ErrorCode::PlanStale);
        runtime.shutdown();
    }

    #[test]
    fn pressure_level_display() {
        let m = MockSignalSource::new();
        m.set_pressure(PressureLevel::Critical);
        assert_eq!(m.pressure().as_str(), "critical");
    }
}
