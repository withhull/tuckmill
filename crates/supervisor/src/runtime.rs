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
pub type ControlQueue =
    Arc<Mutex<Option<Arc<Mutex<std::collections::VecDeque<crate::worker::ControlCommand>>>>>>;
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
        /// Tokens with visible bytes (the honest completion-token count).
        committed: u64,
        /// Raw sampled tokens, including stop/EOS-trimmed tokens.
        generated: u64,
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
    /// Resolved engine identity, published once by the executor thread
    /// after the load completes. Streaming consumers read this snapshot;
    /// they must never query the worker directly (it is busy generating
    /// and holding its lock — querying it from the event consumer is the
    /// Gate B review's SSE deadlock).
    pub identity: Arc<std::sync::OnceLock<tuckmill_manifest::records::EngineIdentity>>,
}

impl ActiveRun {
    /// Convenience: mark this run cancelled.
    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Typed request passed to `start_run`: prompt, output allowance, stopping
/// and sampling policy. Chat versus raw prompt construction happens before
/// this point (supervisor::bundle::RequestSpec); the runtime and every
/// backend only see token ids plus this policy.
#[derive(Debug, Clone)]
pub struct RunRequest {
    pub prompt_tokens: Vec<u32>,
    pub max_output_tokens: u64,
    pub stop: Vec<String>,
    pub seed: u64,
    pub sampling: tuckmill_backend::SamplingPolicy,
}

impl RunRequest {
    /// Greedy request (tests, benchmarks, and pipeline paths).
    pub fn greedy(prompt_tokens: Vec<u32>, max_output_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            max_output_tokens,
            stop: Vec::new(),
            seed: 0,
            sampling: tuckmill_backend::SamplingPolicy::default(),
        }
    }
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
    /// Live worker process id (None when none is running).
    pub worker_pid: Option<u32>,
    /// Worker spawns (model loads). Equivalent consecutive requests keep
    /// this flat; see the loaded-model key (B4).
    pub worker_generation: u64,
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
    /// Count of worker processes spawned (loads). Warm reuse keeps this
    /// flat across equivalent requests; it is the observable used to prove
    /// no unnecessary reloads.
    worker_generations: AtomicU64,
    /// Consecutive footprint-overrun samples (enforcement escalates at 3).
    overrun_streak: AtomicU64,
}

/// Validate a run request against the plan it will actually execute under.
/// This is the shared dispatch boundary (CLI, bench, API, future SDKs and
/// the Metal path all pass through `start_run`): callers must not rely on
/// the worker's previously prepared context, because a warm worker can hold
/// a larger preparation than the current plan admits. Checked arithmetic;
/// typed rejections.
fn validate_request_against_plan(plan: &ExecutionPlan, request: &RunRequest) -> Result<()> {
    let prompt = request.prompt_tokens.len() as u64;
    if prompt == 0 {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "prompt must contain at least one token",
        ));
    }
    if request.max_output_tokens == 0 {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "max_output_tokens must be positive",
        ));
    }
    if request.max_output_tokens > plan.context.max_output_tokens {
        return Err(TuckmillError::new(
            ErrorCode::ContextExceedsBudget,
            format!(
                "max_output_tokens {} exceeds the plan's admitted output allowance of {}",
                request.max_output_tokens, plan.context.max_output_tokens
            ),
        )
        .with_action("Lower max_output_tokens or admit a plan with a larger output allowance."));
    }
    let total = prompt
        .checked_add(request.max_output_tokens)
        .ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                "prompt length plus max_output_tokens overflows; refusing to dispatch",
            )
        })?;
    if total > plan.context.total_context {
        return Err(TuckmillError::new(
            ErrorCode::ContextExceedsBudget,
            format!(
                "prompt {prompt} + max output {} exceeds the plan's admitted context {}",
                request.max_output_tokens, plan.context.total_context
            ),
        )
        .with_action("Lower max_output_tokens, shorten the prompt, or admit a larger context."));
    }
    request.sampling.validate()?;
    Ok(())
}

/// Reuse-compatibility gate: the worker's retained weight allocation (as
/// acknowledged at load) must fit the *current* plan's weights + I/O
/// reservations. A worker prepared under a generous plan must not be
/// silently inherited by a plan that did not reserve what the worker can
/// retain. KV needs no such check: sessions are allocated per request for
/// the request's own admitted need, which the request validation above
/// bounds to the current plan.
fn reuse_fits_retained(plan: &ExecutionPlan, retained_weight_bytes: u64) -> bool {
    let allowance = plan
        .budget
        .reservation_breakdown
        .weights
        .bytes()
        .saturating_add(plan.budget.reservation_breakdown.io_buffers.bytes());
    retained_weight_bytes <= allowance
}

/// Resolved identity of a loaded worker (caller holds the worker lock or is
/// inside the executor thread after load).
fn worker_identity(worker: &IsolatedWorker) -> tuckmill_manifest::records::EngineIdentity {
    let mut id = crate::engine::reference_test_identity();
    id.engine = worker.engine.clone();
    id.certified = worker.certified;
    id.build_id = worker.worker_build.clone();
    if !worker.resolved_kernels.is_empty() {
        id.kernels = worker.resolved_kernels.clone();
    }
    id
}

struct WorkerSlot {
    worker: IsolatedWorker,
    /// Stable identity + prepared capacities of the loaded model. A new
    /// request reuses this worker only when its own key is satisfied by
    /// this one; request timestamps, seeds, prompts, budgets, and plan ids
    /// deliberately do not participate.
    key: LoadedModelKey,
    /// Largest weight-cache allocation this load can retain (acknowledged
    /// by the worker at load, `None` when the backend retains none). Reuse
    /// additionally requires it to fit the current plan's weights + I/O
    /// reservations: model identity alone is not resource admission.
    retained_weight_bytes: u64,
}

/// Stable identity of a loaded model in a worker (B4). Built from immutable
/// model identity, the resolved load options, and the prepared context
/// capacity — never from request time or plan creation time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedModelKey {
    /// Backend that executed the load.
    pub engine: EngineKind,
    pub artifact_id: Id,
    pub file_bytes: u64,
    pub metadata_hash: String,
    pub tensor_index_hash: String,
    pub content_hash: Option<String>,
    /// KV capacity the load prepared. A request with a smaller admitted
    /// context is served by the same prepared model; a larger one requires
    /// a reload.
    pub prepared_context_total: u64,
    /// Resolved backend load options (kernel/storage/numerical/precision
    /// selections). Any change to these is a different loaded model.
    pub load_options: Option<serde_json::Value>,
}

impl LoadedModelKey {
    pub fn new(
        engine: EngineKind,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        load_options: Option<&serde_json::Value>,
    ) -> Self {
        Self {
            engine,
            artifact_id: plan.artifact_id.clone(),
            file_bytes: artifact.file_bytes,
            metadata_hash: artifact.metadata_hash.clone(),
            tensor_index_hash: artifact.tensor_index_hash.clone(),
            content_hash: artifact.content_hash.clone(),
            prepared_context_total: plan.context.total_context,
            load_options: load_options.cloned(),
        }
    }

    /// Whether the already-prepared model behind `self` satisfies the
    /// request keyed by `request`: same immutable identity and load options,
    /// and a prepared context at least as large as requested.
    pub fn satisfies(&self, request: &LoadedModelKey) -> bool {
        self.engine == request.engine
            && self.artifact_id == request.artifact_id
            && self.file_bytes == request.file_bytes
            && self.metadata_hash == request.metadata_hash
            && self.tensor_index_hash == request.tensor_index_hash
            && self.content_hash == request.content_hash
            && self.load_options == request.load_options
            && self.prepared_context_total >= request.prepared_context_total
    }
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
            worker_generations: AtomicU64::new(0),
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
                kv_bytes_per_element: self
                    .execution
                    .kv_bytes_per_element(self.engine_kind, self.policy.kv_bytes_per_element),
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
            worker_pid: self.worker_pid(),
            worker_generation: self.worker_generation(),
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
    /// May wait for the worker lock: streaming paths must use the run-scoped
    /// snapshot (`ActiveRun::identity`) instead, never this.
    pub fn engine_identity(&self) -> tuckmill_manifest::records::EngineIdentity {
        if let Ok(slot) = self.worker.lock() {
            if let Some(s) = slot.as_ref() {
                return worker_identity(&s.worker);
            }
        }
        self.declared_identity()
    }

    /// Declared identity of the selected backend, lock-free. Used as the
    /// explicit fallback for a request whose worker has not finished
    /// loading (never a substitute for a resolved worker identity).
    pub fn declared_identity(&self) -> tuckmill_manifest::records::EngineIdentity {
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

    /// Number of worker spawns (model loads) this runtime has performed.
    /// Equivalent consecutive requests must not increase it (B4).
    pub fn worker_generation(&self) -> u64 {
        self.worker_generations.load(Ordering::Relaxed)
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
    pub fn start_run(
        self: &Arc<Self>,
        ticket: &QueueTicket,
        plan: ExecutionPlan,
        artifact: ArtifactRef,
        request: RunRequest,
    ) -> Result<ActiveRun> {
        // Early admission checks run as one group so a rejected request
        // leaves no ticket behind in the bounded queue. The request is
        // validated here against the plan it will run under, before any
        // queue wait or allocation: a warm worker's previously prepared
        // (possibly larger) context is never an admission authority.
        if let Err(e) = self
            .check_plan_fresh(&plan)
            .and_then(|_| self.plan_admitted(&plan))
            .and_then(|_| validate_request_against_plan(&plan, &request))
        {
            self.queue.abandon(ticket);
            return Err(e);
        }
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
        let identity = Arc::new(std::sync::OnceLock::new());

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
        let identity_evt = identity.clone();
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
                    &identity_evt,
                    request,
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
            identity,
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
        identity: &Arc<std::sync::OnceLock<tuckmill_manifest::records::EngineIdentity>>,
        request: RunRequest,
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
                (
                    "temperature".to_string(),
                    serde_json::json!(request.sampling.temperature),
                ),
            ]
            .into_iter()
            .collect(),
        );

        let prompt_len = request.prompt_tokens.len() as u64;
        match self.engine_mode {
            EngineMode::Isolated => {
                let mut slot = self.worker.lock().expect("worker lock");
                let load_options = self.execution.load_json(self.engine_kind);
                let requested =
                    LoadedModelKey::new(self.engine_kind, plan, artifact, load_options.as_ref());
                let need_spawn = match slot.as_ref() {
                    Some(w) => {
                        let identity_ok = w.key.satisfies(&requested);
                        let retention_ok = reuse_fits_retained(plan, w.retained_weight_bytes);
                        if identity_ok && !retention_ok {
                            self.logger.warn(
                                "REUSE_DECLINED_RETENTION",
                                [
                                    (
                                        "retained_weight_bytes".to_string(),
                                        serde_json::json!(w.retained_weight_bytes),
                                    ),
                                    (
                                        "plan_allowance_bytes".to_string(),
                                        serde_json::json!(plan
                                            .budget
                                            .reservation_breakdown
                                            .weights
                                            .bytes()
                                            .saturating_add(
                                                plan.budget
                                                    .reservation_breakdown
                                                    .io_buffers
                                                    .bytes()
                                            )),
                                    ),
                                ]
                                .into_iter()
                                .collect(),
                            );
                        }
                        !identity_ok || !retention_ok
                    }
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
                    self.worker_generations.fetch_add(1, Ordering::Relaxed);
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
                    // Plan acknowledgement (B3): the worker's prepared KV
                    // state must fit what the admitted plan reserved. A
                    // worker that would allocate more than the plan priced
                    // fails closed before any request runs on it.
                    let kv_prepared = ack_stats
                        .as_ref()
                        .and_then(|s| s.kv_state_bytes)
                        .unwrap_or(0);
                    let retained_weight = ack_stats
                        .as_ref()
                        .and_then(|s| s.retained_weight_bytes)
                        .unwrap_or(0);
                    let kv_reserved = plan.budget.reservation_breakdown.kv_state.bytes();
                    if kv_prepared > kv_reserved {
                        return Err(TuckmillError::new(
                            ErrorCode::InvalidPlan,
                            format!(
                                "the worker prepared {} of KV state for context {} but the plan reserved only {}; refusing to run on an understated plan",
                                tuckmill_manifest::units::Bytes(kv_prepared),
                                plan.context.total_context,
                                tuckmill_manifest::units::Bytes(kv_reserved)
                            ),
                        )
                        .with_action(
                            "Rebuild the plan from the worker's resolved KV representation.",
                        ));
                    }
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
                            ("kv_state_bytes".to_string(), serde_json::json!(kv_prepared)),
                            (
                                "kv_reserved_bytes".to_string(),
                                serde_json::json!(kv_reserved),
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
                        key: requested,
                        retained_weight_bytes: retained_weight,
                        worker,
                    });
                } else {
                    // Warm reuse: the loaded model identity and prepared
                    // capacities satisfy this request; no reload, no new
                    // session state (KV state is allocated per request).
                    self.logger.info(
                        "WORKER_REUSED",
                        [
                            ("request_id".to_string(), serde_json::json!(request_id)),
                            (
                                "prepared_context".to_string(),
                                serde_json::json!(requested.prepared_context_total),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    );
                }
                // Every request — fresh load or warm reuse — attaches the
                // live worker's control queue to its own scoped holder, so a
                // governor command reaches the backend even when the worker
                // was reused (Gate B review: the reuse branch used to leave
                // it None and throttling was labelled without a command).
                if let Ok(mut slot_controls) = controls.lock() {
                    *slot_controls = Some(slot.as_ref().expect("worker slot").worker.controls());
                }
                // Publish the resolved identity for streaming consumers.
                let _ = identity.set(worker_identity(&slot.as_ref().expect("worker slot").worker));
                let WorkerSlot { worker, .. } = slot.as_mut().expect("worker slot");
                let mut token_tx = |index: u64, text: &str| -> bool {
                    // Interim status only (release ordinal, not a token
                    // count); the authoritative visible-token count arrives
                    // with Done and is stored below.
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
                        prompt_tokens: request.prompt_tokens,
                        max_output_tokens: request.max_output_tokens,
                        stop: request.stop,
                        seed: request.seed,
                        sampling: request.sampling,
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
                let _ = identity.set(engine.identity());
                let req = crate::engine::EngineRequest {
                    request_id: request_id.to_string(),
                    prompt_tokens: request.prompt_tokens,
                    max_output_tokens: request.max_output_tokens,
                    stop: request.stop,
                    seed: request.seed,
                    sampling: request.sampling,
                };
                let mut token_tx = |index: u64, text: &str| -> bool {
                    // Interim status only (release ordinal, not a token
                    // count); the authoritative visible-token count arrives
                    // with Done and is stored below.
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
            generated: outcome.generated,
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
                        a.request_id.clone(),
                        a.cancel.clone(),
                        a.pause.clone(),
                        a.committed.clone(),
                        a.state.clone(),
                        a.controls.clone(),
                    )
                })
            };
            if let Some((request_id, cancel, pause, committed, state, controls)) = snapshot {
                // Commands are scoped to the request that was active when
                // the governor decided them; the worker drops late commands
                // whose request already ended instead of applying them to a
                // different request.
                let push_control = |action: &str| {
                    if let Ok(guard) = controls.lock() {
                        if let Some(queue) = guard.as_ref() {
                            if let Ok(mut queue) = queue.lock() {
                                queue.push_back(crate::worker::ControlCommand {
                                    scope: request_id.clone(),
                                    action: action.to_string(),
                                });
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
                generated: _,
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
        test_runtime_with(mode, None)
    }

    fn test_runtime_with(
        mode: EngineMode,
        worker_exe: Option<PathBuf>,
    ) -> (tempfile::TempDir, Arc<Runtime>, Arc<MockSignalSource>) {
        test_runtime_kind(mode, worker_exe, EngineKind::ReferenceTest)
    }

    fn test_runtime_kind(
        mode: EngineMode,
        worker_exe: Option<PathBuf>,
        kind: EngineKind,
    ) -> (tempfile::TempDir, Arc<Runtime>, Arc<MockSignalSource>) {
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
            engine_kind: kind,
            worker_exe,
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
            allow_test_engine: kind == EngineKind::ReferenceTest,
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
                RunRequest {
                    prompt_tokens: (0..prompt_len as u32).collect(),
                    max_output_tokens: max_out,
                    stop: vec![],
                    seed: 42,
                    sampling: tuckmill_backend::SamplingPolicy::default(),
                },
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
                RunRequest::greedy(vec![1, 2], 8),
            )
            .unwrap();
        let runtime2 = runtime.clone();
        let handle = std::thread::spawn(move || {
            runtime2.start_run(
                &ticket_b,
                plan,
                artifact_b,
                RunRequest::greedy(vec![1, 2], 8),
            )
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
    fn loaded_model_key_ignores_time_and_request_content() {
        let plan = |context: u64, created: u64| {
            let mut p = ExecutionPlan::new_skeleton(
                Id::new("a".repeat(64)).unwrap(),
                Id::new("b".repeat(64)).unwrap(),
                "qwen3".into(),
            );
            p.mode = ExecutionMode::Resident;
            p.context = tuckmill_manifest::plan::PlanContext::new(context, 64);
            p.created_unix_ms = created;
            p.plan_id = p.compute_id().unwrap();
            p
        };
        let artifact = tuckmill_backend::ArtifactRef {
            path: "/tmp/m.gguf".into(),
            file_bytes: 11,
            metadata_hash: "1".repeat(64),
            tensor_index_hash: "2".repeat(64),
            tokenizer_hash: None,
            arch_id: Some("qwen3".into()),
            content_hash: Some("3".repeat(64)),
        };
        let opts = Some(serde_json::json!({"kv_precision": "f16"}));
        let earlier = LoadedModelKey::new(
            EngineKind::TuckCore,
            &plan(2048, 1),
            &artifact,
            opts.as_ref(),
        );
        let later = LoadedModelKey::new(
            EngineKind::TuckCore,
            &plan(2048, 9_999),
            &artifact,
            opts.as_ref(),
        );
        assert_eq!(
            earlier, later,
            "plan creation time must not enter the loaded-model key"
        );
        assert!(earlier.satisfies(&later));

        let smaller = LoadedModelKey::new(
            EngineKind::TuckCore,
            &plan(1024, 2),
            &artifact,
            opts.as_ref(),
        );
        assert!(
            earlier.satisfies(&smaller),
            "a prepared 2048 context serves a 1024 request"
        );
        assert!(
            !smaller.satisfies(&earlier),
            "a prepared 1024 context cannot serve a 2048 request"
        );

        let other_opts = Some(serde_json::json!({"kv_precision": "f32"}));
        let reloaded = LoadedModelKey::new(
            EngineKind::TuckCore,
            &plan(2048, 1),
            &artifact,
            other_opts.as_ref(),
        );
        assert!(
            !earlier.satisfies(&reloaded),
            "a load-defining option change must force a reload"
        );

        let mut edited = artifact.clone();
        edited.content_hash = Some("4".repeat(64));
        let tampered =
            LoadedModelKey::new(EngineKind::TuckCore, &plan(2048, 1), &edited, opts.as_ref());
        assert!(
            !earlier.satisfies(&tampered),
            "artifact identity change must force a reload"
        );
    }

    #[test]
    fn sequential_requests_reuse_the_loaded_worker_without_state_leaks() {
        let Some(worker_exe) = crate::test_worker_bin() else {
            return;
        };
        let (_dir, runtime, _signals) = test_runtime_with(EngineMode::Isolated, Some(worker_exe));
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();
        let run_once = |context: u64| -> (String, String, u64, Id) {
            let plan = runtime
                .plan_for(
                    &artifact,
                    &adapter,
                    Bytes::parse("2GiB").unwrap(),
                    context,
                    64,
                )
                .unwrap();
            runtime.plan_admitted(&plan).unwrap();
            let artifact_ref = artifact_ref_for(&runtime);
            let ticket = runtime.submit().unwrap();
            let run = runtime
                .start_run(
                    &ticket,
                    plan,
                    artifact_ref,
                    RunRequest::greedy(vec![5, 6], 4),
                )
                .unwrap();
            let mut text = String::new();
            let (reason, committed) = collect_run(&run, |_i, t| text.push_str(t));
            (text, reason.as_str().to_string(), committed, run.session_id)
        };

        // Ten equivalent sequential requests: one load, ten isolated runs.
        let mut first: Option<(String, Id)> = None;
        for i in 0..10 {
            let (text, reason, committed, session) = run_once(2048);
            assert_eq!(reason, "length");
            assert_eq!(committed, 4);
            let fresh_session = first.as_ref().map(|(_, s)| s != &session).unwrap_or(true);
            match &first {
                None => first = Some((text, session)),
                Some((first_text, _)) => assert_eq!(
                    &text, first_text,
                    "request {i}: visible output must not inherit session state"
                ),
            }
            assert!(fresh_session, "each request must get a fresh session id");
            assert_eq!(
                runtime.worker_generation(),
                1,
                "request {i} must reuse the loaded model"
            );
        }

        // A load-defining change (larger prepared context) reloads once.
        let (_text, _reason, _committed, _session) = run_once(4096);
        assert_eq!(runtime.worker_generation(), 2);
        // A smaller request fits the prepared model: reuse again.
        let (_text, _reason, _committed, _session) = run_once(1024);
        assert_eq!(runtime.worker_generation(), 2);
        runtime.shutdown();
    }

    #[test]
    fn warm_reuse_allocates_kv_for_the_current_request_need() {
        // Gate B review finding 2: reusing a worker prepared for a larger
        // context must not allocate the old capacity against the current,
        // smaller admitted plan. TuckCore allocates per-request KV for the
        // request's own prompt+output need; assert the actual bytes the
        // worker reports, not just that it did not reload.
        let Some(worker_exe) = crate::test_worker_bin() else {
            return;
        };
        let (_dir, runtime, _signals) =
            test_runtime_kind(EngineMode::Isolated, Some(worker_exe), EngineKind::TuckCore);
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();
        let run_once = |context: u64| -> (u64, u64) {
            let plan = runtime
                .plan_for(
                    &artifact,
                    &adapter,
                    Bytes::parse("2GiB").unwrap(),
                    context,
                    16,
                )
                .unwrap();
            runtime.plan_admitted(&plan).unwrap();
            runtime.admit_engine(&plan).unwrap();
            let ticket = runtime.submit().unwrap();
            let run = runtime
                .start_run(
                    &ticket,
                    plan,
                    artifact_ref_for(&runtime),
                    RunRequest::greedy(vec![5, 6, 7, 8], 16),
                )
                .unwrap();
            let (reason, _committed) = collect_run(&run, |_, _| {});
            assert_eq!(reason, FinishReason::Length);
            let stats = runtime.backend_stats().expect("backend stats");
            (
                stats.session_kv_bytes.expect("session KV bytes reported"),
                stats.kv_state_bytes.expect("prepared KV bytes reported"),
            )
        };

        // Load once with a large prepared context, then reuse it for a much
        // smaller admitted context: the second request allocates exactly the
        // same small session (prompt+output need), never the prepared 4k.
        let (session_large_prepared, prepared_large) = run_once(4096);
        assert_eq!(runtime.worker_generation(), 1);
        assert!(
            session_large_prepared < prepared_large,
            "a request allocates its own need ({session_large_prepared} B), not the prepared capacity ({prepared_large} B)"
        );
        let (session_small_plan, prepared_after_small) = run_once(1024);
        assert_eq!(runtime.worker_generation(), 1, "still one load");
        assert_eq!(
            session_small_plan, session_large_prepared,
            "identical requests allocate identical session KV regardless of prepared capacity"
        );
        assert_eq!(
            prepared_after_small, prepared_large,
            "the reused worker still reports its one loaded (4096) preparation"
        );
        assert!(
            session_small_plan < prepared_large,
            "the reused request's KV allocation ({session_small_plan} B) is its own need, not the prepared 4096-token capacity ({prepared_large} B)"
        );
        runtime.shutdown();
    }

    /// Small handcrafted plan for boundary tests (reservations as given).
    /// Uses the runtime's own device id so plan freshness holds.
    fn boundary_plan(
        device_id: &Id,
        context: u64,
        max_out: u64,
        budget: &str,
        weights: u64,
        kv: u64,
        io: u64,
    ) -> ExecutionPlan {
        let mut p = ExecutionPlan::new_skeleton(
            Id::new("a".repeat(64)).unwrap(),
            device_id.clone(),
            "qwen3".into(),
        );
        p.mode = tuckmill_manifest::plan::ExecutionMode::Resident;
        p.context = tuckmill_manifest::plan::PlanContext::new(context, max_out);
        p.budget.requested_budget = Bytes::parse(budget).unwrap();
        p.budget.add_reservation(
            tuckmill_manifest::records::ReserveClass::Weights,
            Bytes(weights),
        );
        if kv > 0 {
            p.budget
                .add_reservation(tuckmill_manifest::records::ReserveClass::KvState, Bytes(kv));
        }
        p.budget.add_reservation(
            tuckmill_manifest::records::ReserveClass::IoBuffers,
            Bytes(io),
        );
        p.budget.reservation = p.budget.reservation_breakdown.total();
        p.plan_id = p.compute_id().unwrap();
        p
    }

    #[test]
    fn request_plan_validation_is_typed_and_uses_the_plan_allowance() {
        // The validation is a pure function of (plan, request), so a cold
        // and a warm worker necessarily reject identically; this test pins
        // the rules themselves.
        let device = Id::new("d".repeat(64)).unwrap();
        let plan = boundary_plan(&device, 1024, 128, "1GiB", 1 << 20, 0, 1 << 20);
        let request = |prompt: usize, max: u64| RunRequest {
            prompt_tokens: vec![0; prompt],
            max_output_tokens: max,
            stop: vec![],
            seed: 0,
            sampling: tuckmill_backend::SamplingPolicy::default(),
        };
        assert!(validate_request_against_plan(&plan, &request(2, 100)).is_ok());
        assert_eq!(
            validate_request_against_plan(&plan, &request(0, 10))
                .unwrap_err()
                .code(),
            ErrorCode::InvalidRequest
        );
        assert_eq!(
            validate_request_against_plan(&plan, &request(2, 0))
                .unwrap_err()
                .code(),
            ErrorCode::InvalidRequest
        );
        // Output allowance: 2000 > the plan's admitted 128, regardless of
        // how much context the plan (or a warm worker) could hold.
        assert_eq!(
            validate_request_against_plan(&plan, &request(1012, 2000))
                .unwrap_err()
                .code(),
            ErrorCode::ContextExceedsBudget
        );
        // Context fit: prompt + output must be within the plan's context.
        assert_eq!(
            validate_request_against_plan(&plan, &request(1000, 100))
                .unwrap_err()
                .code(),
            ErrorCode::ContextExceedsBudget
        );
        // Unsupported sampling is refused at the same boundary.
        let mut bad_sampling = request(2, 100);
        bad_sampling.sampling.temperature = 3.0;
        assert_eq!(
            validate_request_against_plan(&plan, &bad_sampling)
                .unwrap_err()
                .code(),
            ErrorCode::InvalidRequest
        );
    }

    #[test]
    fn reuse_gate_requires_retained_allocations_to_fit_the_current_plan() {
        let device = Id::new("d".repeat(64)).unwrap();
        let plan = boundary_plan(&device, 2048, 64, "1GiB", 1 << 20, 0, 1 << 20);
        let allowance = (1u64 << 20) + (1u64 << 20);
        assert!(reuse_fits_retained(&plan, 0));
        assert!(reuse_fits_retained(&plan, allowance));
        assert!(
            !reuse_fits_retained(&plan, allowance + 1),
            "retained allocations beyond the plan's weights + I/O reservations must decline reuse"
        );
    }

    #[test]
    fn invalid_requests_are_rejected_identically_on_cold_and_warm_workers() {
        // Gate B review finding: the shared runtime boundary must not rely
        // on the worker's previously prepared context. Cache state must
        // never change admission correctness.
        let Some(worker_exe) = crate::test_worker_bin() else {
            return;
        };
        let (_dir, runtime, _signals) =
            test_runtime_kind(EngineMode::Isolated, Some(worker_exe), EngineKind::TuckCore);
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();
        let plan = runtime
            .plan_for(&artifact, &adapter, Bytes::parse("2GiB").unwrap(), 2048, 64)
            .unwrap();
        runtime.plan_admitted(&plan).unwrap();
        runtime.admit_engine(&plan).unwrap();
        let start = |request: RunRequest| {
            let ticket = runtime.submit().unwrap();
            runtime.start_run(&ticket, plan.clone(), artifact_ref_for(&runtime), request)
        };
        let expect_err = |r: Result<ActiveRun>| match r {
            Ok(_) => panic!("request must be rejected"),
            Err(e) => e,
        };

        // Cold worker: max_output 2000 exceeds the plan's admitted 64.
        let cold = expect_err(start(RunRequest::greedy(vec![1, 2], 2000)));
        assert_eq!(cold.code(), ErrorCode::ContextExceedsBudget);
        assert_eq!(
            runtime.worker_generation(),
            0,
            "rejected before any worker was dispatched"
        );

        // Warm worker: a successful run first, then the identical request.
        let run = start(RunRequest::greedy(vec![1, 2], 16)).unwrap();
        let (reason, committed) = collect_run(&run, |_, _| {});
        assert_eq!(reason, FinishReason::Length);
        assert_eq!(committed, 16);
        assert_eq!(runtime.worker_generation(), 1);
        let warm = expect_err(start(RunRequest::greedy(vec![1, 2], 2000)));
        assert_eq!(warm.code(), cold.code());
        assert_eq!(warm.message(), cold.message());
        assert_eq!(
            runtime.worker_generation(),
            1,
            "a rejected request must not reload or disturb the worker"
        );

        // Prompt + output beyond the plan's context: rejected warm as well.
        let warm_ctx = expect_err(start(RunRequest::greedy(vec![0; 2000], 100)));
        assert_eq!(warm_ctx.code(), ErrorCode::ContextExceedsBudget);
        assert_eq!(runtime.worker_generation(), 1);
        runtime.shutdown();
    }

    #[test]
    fn reuse_declines_when_retained_allocations_exceed_the_current_plan() {
        // A lower-budget plan must not silently inherit the warm worker's
        // retained allocations: reuse declines, and the reload under the
        // new plan fails closed (the existing load-time arena check).
        let Some(worker_exe) = crate::test_worker_bin() else {
            return;
        };
        let (_dir, runtime, _signals) =
            test_runtime_kind(EngineMode::Isolated, Some(worker_exe), EngineKind::TuckCore);
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();

        // Warm worker under a generous plan.
        let plan_a = runtime
            .plan_for(&artifact, &adapter, Bytes::parse("2GiB").unwrap(), 2048, 64)
            .unwrap();
        runtime.plan_admitted(&plan_a).unwrap();
        runtime.admit_engine(&plan_a).unwrap();
        let ticket = runtime.submit().unwrap();
        let run = runtime
            .start_run(
                &ticket,
                plan_a,
                artifact_ref_for(&runtime),
                RunRequest::greedy(vec![1, 2], 16),
            )
            .unwrap();
        let (reason, _) = collect_run(&run, |_, _| {});
        assert_eq!(reason, FinishReason::Length);
        assert_eq!(runtime.worker_generation(), 1);
        let retained = runtime
            .backend_stats()
            .and_then(|s| s.retained_weight_bytes)
            .expect("tuckcore reports retained weight bytes");
        assert!(retained > 0);

        // A tiny plan whose weights + I/O reservations cannot cover the
        // retained allocation: reuse must decline. The reload then fails
        // closed at the worker's arena check (explicit rejection).
        let starved = boundary_plan(
            &runtime.device.device_id,
            512,
            64,
            "64MiB",
            2 * 1024,
            4 * 1024 * 1024,
            2 * 1024,
        );
        runtime.plan_admitted(&starved).unwrap();
        runtime.admit_engine(&starved).unwrap();
        let ticket = runtime.submit().unwrap();
        let run = runtime
            .start_run(
                &ticket,
                starved,
                artifact_ref_for(&runtime),
                RunRequest::greedy(vec![1, 2, 3, 4], 16),
            )
            .unwrap();
        let terminal = loop {
            match run.events.recv() {
                Ok(RunEvent::Error {
                    code,
                    message,
                    action,
                }) => break Some((code, message, action)),
                Ok(RunEvent::Done { .. }) => break None,
                Ok(_) => {}
                Err(_) => break None,
            }
        };
        let (code, message, _action) = terminal.expect("the reload must fail closed");
        assert_eq!(code, "INVALID_PLAN", "{message}");
        assert!(
            message.contains("weight-cache"),
            "the rejection must name the uncovered allocation: {message}"
        );
        assert_eq!(
            runtime.worker_generation(),
            2,
            "the declined reuse reloaded under the new plan"
        );

        // And a plan that does cover the retained allocation reuses again
        // after a fresh load.
        let generous = boundary_plan(
            &runtime.device.device_id,
            512,
            64,
            "64MiB",
            retained + (1 << 20),
            4 * 1024 * 1024,
            16 * 1024 * 1024,
        );
        runtime.plan_admitted(&generous).unwrap();
        runtime.admit_engine(&generous).unwrap();
        let ticket = runtime.submit().unwrap();
        let run = runtime
            .start_run(
                &ticket,
                generous,
                artifact_ref_for(&runtime),
                RunRequest::greedy(vec![1, 2, 3, 4], 16),
            )
            .unwrap();
        let (reason, committed) = collect_run(&run, |_, _| {});
        assert_eq!(reason, FinishReason::Length);
        assert_eq!(committed, 16);
        assert_eq!(
            runtime.worker_generation(),
            3,
            "a covering plan reloads once and runs"
        );
        runtime.shutdown();
    }

    #[test]
    fn pressure_during_a_reused_request_reaches_the_backend() {
        // Gate B review finding 3: the warm-reuse branch used to leave the
        // per-request control holder empty, so a governor Throttling state
        // was labelled without any command reaching the worker. Complete a
        // request, reuse the worker for a second one under warning pressure,
        // and verify the backend actually acted (its loader cache was
        // trimmed), not just that the display said throttling.
        let Some(worker_exe) = crate::test_worker_bin() else {
            return;
        };
        let (_dir, runtime, signals) =
            test_runtime_kind(EngineMode::Isolated, Some(worker_exe), EngineKind::TuckCore);
        let (artifact, adapter) = runtime.resolve_artifact("local").unwrap();
        let run_once = |tokens: u64| {
            let plan = runtime
                .plan_for(
                    &artifact,
                    &adapter,
                    Bytes::parse("2GiB").unwrap(),
                    2048,
                    tokens,
                )
                .unwrap();
            runtime.plan_admitted(&plan).unwrap();
            runtime.admit_engine(&plan).unwrap();
            let ticket = runtime.submit().unwrap();
            let run = runtime
                .start_run(
                    &ticket,
                    plan,
                    artifact_ref_for(&runtime),
                    RunRequest::greedy(vec![1, 2, 3, 4], tokens),
                )
                .unwrap();
            let (reason, committed) = collect_run(&run, |_, _| {});
            assert_eq!(reason, FinishReason::Length);
            assert_eq!(committed, tokens);
        };

        // First request loads the worker.
        run_once(4);
        assert_eq!(runtime.worker_generation(), 1);
        let trims_before = runtime
            .backend_stats()
            .and_then(|s| s.trims_applied)
            .unwrap_or(0);

        // Warning pressure while reused requests run. The observable is the
        // engine applying the trim at a safe checkpoint (`trims_applied`),
        // not an eviction count: evicting an empty expendable set is
        // legitimately zero and must not decide this regression (CI run
        // 39021538597 exposed exactly that flake). The monitor thread is
        // independently scheduled, so a single short run can straddle a
        // starved poll under load; the assertion retries a bounded number
        // of reused requests and requires the trim to land in at least one.
        signals.set_pressure(crate::pressure::PressureLevel::Warning);
        let mut trims_after = trims_before;
        for attempt in 1..=3 {
            run_once(48);
            trims_after = runtime
                .backend_stats()
                .and_then(|s| s.trims_applied)
                .unwrap_or(0);
            if trims_after > trims_before {
                break;
            }
            eprintln!("attempt {attempt}: trim not applied yet; retrying under sustained pressure");
        }
        signals.set_pressure(crate::pressure::PressureLevel::Normal);

        assert_eq!(
            runtime.worker_generation(),
            1,
            "the reused requests kept the one loaded worker"
        );
        assert!(
            trims_after > trims_before,
            "the trim command must reach the reused worker's backend and be applied: trims {trims_before} -> {trims_after}"
        );
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
            .start_run(
                &ticket,
                plan,
                artifact,
                RunRequest::greedy(vec![1, 2, 3], 300),
            )
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
            .start_run(
                &ticket,
                plan,
                artifact_ref,
                RunRequest::greedy(vec![1, 2, 3], 300),
            )
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
