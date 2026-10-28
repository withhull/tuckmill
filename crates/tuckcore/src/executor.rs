//! TuckCore executor: the `InferenceBackend` implementation.
//!
//! Stage 0 executes dense Qwen3 GGUF artifacts with scalar f32 reference
//! kernels and bounded weight streaming. It is uncertified by construction:
//! `EngineIdentity.certified` is false, and its note says exactly what it is.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;

use tuckmill_backend::output::VisibleOutputBuffer;
use tuckmill_backend::{
    ArtifactRef, BackendCapabilities, BackendId, BackendStats, FinishReason, GenControl,
    GenOutcome, GenerateRequest, InferenceBackend, RequestTiming, RunState, StorageStats,
};
use tuckmill_manifest::adapter::AdapterRegistry;
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::plan::{ExecutionMode, ExecutionPlan};
use tuckmill_manifest::records::EngineIdentity;
use tuckmill_manifest::units::Bytes;
use tuckmill_manifest::{BACKEND_ABI_VERSION, WORKER_PROTOCOL_VERSION};

use crate::completion::CompletionTracker;
use crate::engine::metal::{
    self as engine_metal, EngineModel, EngineOptions, GreedyInput, KvCache, PendingToken,
};
use crate::forward::{
    self, Forward, KvPrecision, KvState, ModelWeights, NumericalProfile, ProfileCounters,
};
use crate::kernels::{KernelSelection, ResolvedKernels};
use crate::loader::{BlockLoader, BlockPolicy, StorageSelection};
use crate::model;
use crate::qwen35_forward::{ArtifactTensorSource, TensorSource};
use crate::qwen35_gpu::{GpuAllocPlan, Qwen35Gpu};
use crate::sampling::{self, Rng, SamplingConfig};
use crate::tensor;
use crate::tokenizer::{DecodeVocab, Utf8StreamDecoder};

/// Defaults; overridable per run through the versioned Load options.
pub const DEFAULT_CACHE_BYTES: u64 = 512 * 1024 * 1024;
pub const DEFAULT_BLOCK_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct TuckCoreOptions {
    weight_cache_bytes: Option<u64>,
    block_bytes: Option<u64>,
    /// "oracle-f32" (default) or "llama-cpu" (mirrors pinned llama.cpp CPU
    /// kernels; used for reference comparisons).
    numerical_profile: Option<String>,
    /// KV cache precision: "f32" (default) or "f16". Stored entries are
    /// f16-representable f32 values when "f16"; the actual setting is
    /// reported in stats so comparisons never trust the request alone.
    kv_precision: Option<String>,
    /// Verify the artifact's canonical content digest before executing
    /// (default true; a bounded single streaming pass). Disabling is an
    /// explicit, recorded choice.
    verify_content_hash: Option<bool>,
    /// Enable phase profiling counters (also via TUCKMILL_TUCKORE_PROFILE=1).
    profile: Option<bool>,
    /// Kernel implementation: "auto" (default; NEON where the target has
    /// it), "scalar" (force the portable reference kernels), or "neon"
    /// (require NEON; refused where unavailable).
    kernels: Option<KernelSelection>,
    /// Weight-block storage: "auto" (default; map when possible), "mapped"
    /// (require the zero-copy mapping), or "owned" (always copy into
    /// bounded buffers).
    storage: Option<StorageSelection>,
    /// Metal prefill micro-batch (default 8, bounded 1..=32; the effective
    /// size is the largest candidate whose exact workspace geometry fits
    /// the admitted workspaces reservation). Applies to `--kernels metal`
    /// only; refused elsewhere rather than ignored.
    prefill_batch: Option<u32>,
    /// Engine prefix cache (ADR-012 step 2); default on. Benchmarks turn it
    /// off so repeated prompts never become hidden cache hits.
    #[serde(default)]
    prefix_cache: Option<bool>,
    /// Speculative greedy decoding with the artifact's MTP head (ADR-012
    /// step 3); default off. Greedy output is token-identical to plain
    /// decoding; refused for an artifact without an MTP block.
    #[serde(default)]
    speculative: Option<bool>,
}

struct Loaded {
    loader: Arc<BlockLoader>,
    tracker: Arc<CompletionTracker>,
    weights: Arc<ModelWeights>,
    tokenizer: DecodeVocab,
    context_total: u64,
    profile: NumericalProfile,
    kv_precision: KvPrecision,
    counters: ProfileCounters,
    kernels: ResolvedKernels,
    /// KV bytes allocated for the most recent request (the request's actual
    /// prompt+output need), set at generate time. The prepared capacity is
    /// `context_total`; per-request sessions are allocated for their own
    /// admitted need, never for a previously prepared larger capacity.
    session_kv_bytes: Arc<AtomicU64>,
    /// Largest weight-cache allocation this load can retain between
    /// requests: the cache budget clamped to the payload, plus one block of
    /// transient overhead. This is exactly the quantity the load-time
    /// arena check validated against the plan, reported so the supervisor
    /// can re-check it against every later plan before reusing the worker.
    retained_weight_bytes: u64,
}

/// A loaded artifact served by the unified engine on the resident Metal
/// path (ADR-011): `qwen35`, and dense `qwen3` with `--kernels metal`.
/// Weights, KV planes and any recurrent/convolution state live on one GPU
/// owner for the prepared capacity; every request starts from a reset
/// session, so state is never shared across requests.
struct EngineLoaded {
    /// The graph executed by the unified engine.
    model: EngineModel,
    /// Declared KV cache precision of the attention layers.
    kv: KvCache,
    /// Kernel identity reported in records.
    kernel_id: String,
    tokenizer: DecodeVocab,
    eos_token_id: Option<u32>,
    context_total: u64,
    /// Exact preflighted GPU allocation (equal to the device ledger after
    /// load); each class was checked against its plan reservation.
    alloc: GpuAllocPlan,
    /// Session state bytes used by the most recent request (the prepared
    /// state; 0 before the first request).
    session_kv_bytes: Arc<AtomicU64>,
    /// Prompt positions the most recent request resumed from the prefix
    /// cache (0: cold).
    last_prefix_hit: AtomicU64,
    /// Whether the engine was loaded with the prefix cache.
    prefix_cache: bool,
    counters: ProfileCounters,
}

/// The experimental custom native executor.
pub struct TuckCore {
    loaded: Option<Loaded>,
    /// A model served by the unified engine on Metal (qwen35, and dense
    /// qwen3 with `--kernels metal`).
    engine: Option<EngineLoaded>,
    /// Optional artificial per-token pacing (tests only; real forward work is
    /// its own pacing).
    pub per_token_delay: Duration,
    /// Shared cache-trim epoch (governor -> engine). Incremented via Control
    /// frames; applied by the forward loops at safe checkpoints.
    trim_epoch: Arc<std::sync::atomic::AtomicU64>,
    applied_trim_epoch: Arc<std::sync::atomic::AtomicU64>,
}

impl Default for TuckCore {
    fn default() -> Self {
        Self::new()
    }
}

impl TuckCore {
    pub fn new() -> Self {
        Self {
            loaded: None,
            engine: None,
            per_token_delay: Duration::ZERO,
            trim_epoch: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            applied_trim_epoch: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.is_some() || self.engine.is_some()
    }

    /// Applies a pending cache trim if the governor asked for one; returns
    /// the number of evicted blocks (0 when nothing was pending). Pinned,
    /// in-flight blocks are never evicted.
    fn apply_pending_trim(&self) -> usize {
        let requested = self.trim_epoch.load(Ordering::Relaxed);
        let applied = self.applied_trim_epoch.load(Ordering::Relaxed);
        if requested == applied {
            return 0;
        }
        let Some(loaded) = self.loaded.as_ref() else {
            self.applied_trim_epoch.store(requested, Ordering::Relaxed);
            return 0;
        };
        let evicted = loaded.loader.evict_all();
        loaded
            .counters
            .trims_applied
            .fetch_add(1, Ordering::Relaxed);
        self.applied_trim_epoch.store(requested, Ordering::Relaxed);
        evicted
    }

    /// Engine identity (inherent form so the supervisor can query the
    /// backend's declared identity before any worker or load exists). After
    /// a load, `kernels` names the kernel package and the *resolved*
    /// implementation actually selected; before a load it declares the
    /// package without claiming an implementation.
    pub fn identity(&self) -> EngineIdentity {
        EngineIdentity {
            engine: BackendId::TuckCore.as_str().to_string(),
            certified: false,
            backend_abi_version: BACKEND_ABI_VERSION,
            worker_protocol_version: WORKER_PROTOCOL_VERSION,
            kernels: if let Some(engine) = &self.engine {
                vec![engine.kernel_id.clone()]
            } else {
                kernel_ids(self.loaded.as_ref().map(|l| l.kernels))
            },
            build_id: crate::format_build_id(),
        }
    }
}

impl InferenceBackend for TuckCore {
    fn backend_id(&self) -> BackendId {
        BackendId::TuckCore
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            architectures: vec!["qwen3".to_string(), "qwen35".to_string()],
            moe: false,
            hybrid: true,
            ternary: false,
            requires_gpu: false,
            notes: vec![
                "Stage 0: dense Qwen3 GGUF, scalar f32 reference kernels, bounded weight streaming (ADR-005)"
                    .to_string(),
                "Metal (--kernels metal): the unified engine (ADR-011) serves dense qwen3 (f32 or f16 KV) and qwen35 (packed f16 KV), resident, batched prefill"
                    .to_string(),
                "MoE and ternary profiles remain separately gated; routing is never substituted"
                    .to_string(),
            ],
        }
    }

    fn identity(&self) -> EngineIdentity {
        self.identity()
    }

    fn load(
        &mut self,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        options: Option<&serde_json::Value>,
    ) -> Result<String> {
        plan.validate()?;
        if plan.mode == ExecutionMode::Rejected {
            return Err(TuckmillError::new(
                ErrorCode::InvalidPlan,
                "refusing to load a rejected plan",
            ));
        }
        let opts: TuckCoreOptions = match options {
            Some(v) => serde_json::from_value(v.clone()).map_err(|e| {
                TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!("invalid tuckcore options: {e}"),
                )
            })?,
            None => TuckCoreOptions::default(),
        };
        let policy = BlockPolicy {
            block_bytes: opts.block_bytes.unwrap_or(DEFAULT_BLOCK_BYTES),
            cache_bytes: opts.weight_cache_bytes.unwrap_or(DEFAULT_CACHE_BYTES),
        };
        policy.validate()?;
        let profile = match opts.numerical_profile.as_deref() {
            None | Some("oracle-f32") => NumericalProfile::OracleF32,
            Some("llama-cpu") => NumericalProfile::LlamaCpu,
            Some(other) => {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!(
                        "unknown numerical profile {other:?}; use \"oracle-f32\" or \"llama-cpu\""
                    ),
                ))
            }
        };
        let verify_content_hash = opts.verify_content_hash.unwrap_or(true);
        let profile_enabled = opts.profile.unwrap_or(false)
            || std::env::var("TUCKMILL_TUCKORE_PROFILE")
                .map(|v| v == "1")
                .unwrap_or(false);
        let counters = ProfileCounters {
            enabled: profile_enabled,
            ..Default::default()
        };
        let kernels = opts.kernels.unwrap_or(KernelSelection::Auto).resolve()?;
        let metal_mode = kernels == ResolvedKernels::Metal;
        let storage_selection = opts.storage.unwrap_or(StorageSelection::Auto);
        if !metal_mode && opts.prefill_batch.is_some() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "prefill_batch applies only to the Metal kernel path; the resolved CPU kernels execute single-token steps and must not silently ignore it",
            ));
        }
        if metal_mode && storage_selection != StorageSelection::Auto {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "storage selection applies to CPU weight streaming; the Metal path keeps weights resident on the GPU",
            ));
        }
        if metal_mode && profile != NumericalProfile::OracleF32 {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                "the Metal path implements the oracle-f32 numerical profile only; the llama-cpu activation-quantization profile is CPU-only",
            )
            .with_action("Select --numerical-profile oracle-f32 (the default) for --kernels metal."));
        }
        let load_start = Instant::now();
        let kv_precision = match opts.kv_precision.as_deref() {
            None | Some("f32") => KvPrecision::F32,
            Some("f16") => KvPrecision::F16,
            Some(other) => {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!("unknown kv_precision {other:?}; use \"f32\" or \"f16\""),
                ))
            }
        };

        // ---- identity verification before anything executes ----
        let path = std::path::Path::new(&artifact.path);
        let insp =
            tuckmill_manifest::gguf::inspect(path, &tuckmill_manifest::gguf::Limits::default())?;
        if insp.file_len != artifact.file_bytes {
            return Err(TuckmillError::new(
                ErrorCode::IntegrityFailure,
                format!(
                    "artifact file is {} bytes but the recorded identity says {}",
                    insp.file_len, artifact.file_bytes
                ),
            ));
        }
        let metadata_hash = insp.metadata_hash()?;
        let tensor_index_hash = insp.tensor_index_hash()?;
        if metadata_hash.as_str() != artifact.metadata_hash
            || tensor_index_hash.as_str() != artifact.tensor_index_hash
        {
            return Err(TuckmillError::new(
                ErrorCode::IntegrityFailure,
                "artifact hashes do not match the recorded identity; refusing to execute",
            )
            .with_action("Re-import the artifact; never execute drifted weights."));
        }
        if let (Some(expected), Ok(Some(actual))) =
            (&artifact.tokenizer_hash, insp.tokenizer_hash())
        {
            if actual.as_str() != expected.as_str() {
                return Err(TuckmillError::new(
                    ErrorCode::IntegrityFailure,
                    "tokenizer hash does not match the recorded identity",
                ));
            }
        }
        // Open the artifact once. Content verification, the length check,
        // and the weight mapping all use this exact handle — never a re-open
        // by path — so the file identity that was verified is the file
        // identity that executes. Store replacement is atomic (rename or
        // unlink; see docs), and the length is re-checked after the
        // verification pass to catch truncation during the window.
        let artifact_file = std::fs::File::open(path).map_err(TuckmillError::from)?;
        let opened_len = artifact_file.metadata().map_err(TuckmillError::from)?.len();
        if opened_len != artifact.file_bytes {
            return Err(TuckmillError::new(
                ErrorCode::IntegrityFailure,
                format!(
                    "artifact file opened at {opened_len} bytes but the recorded identity says {}; refusing to execute",
                    artifact.file_bytes
                ),
            )
            .with_action("Re-import the artifact; never execute drifted weights."));
        }
        // Full-payload identity: the store's canonical chunked digest of
        // the open file (independent chunks hashed in parallel with
        // positioned reads, bounded buffers), compared against the recorded
        // one. Drifted bytes must never execute; disabling is explicit and
        // recorded.
        let verify_start = Instant::now();
        let content_verified = if verify_content_hash {
            match &artifact.content_hash {
                Some(expected) => {
                    let (actual, _) = tuckmill_manifest::digest::digest_file(
                        &artifact_file,
                        opened_len,
                        tuckmill_manifest::digest::DEFAULT_CHUNK_BYTES,
                    )
                    .map_err(TuckmillError::from)?;
                    if &actual != expected {
                        return Err(TuckmillError::new(
                            ErrorCode::IntegrityFailure,
                            "artifact content digest does not match the recorded identity; refusing to execute drifted weights",
                        ));
                    }
                    true
                }
                None => false,
            }
        } else {
            false
        };
        let post_len = artifact_file.metadata().map_err(TuckmillError::from)?.len();
        if post_len != artifact.file_bytes {
            return Err(TuckmillError::new(
                ErrorCode::IntegrityFailure,
                format!(
                    "artifact changed size during verification ({post_len} bytes now); refusing to execute"
                ),
            ));
        }
        ProfileCounters::record(&counters.content_verify_nanos, verify_start);

        // ---- configuration and type checks ----
        for t in &insp.tensors {
            if !tensor::is_supported(t.type_id) {
                return Err(TuckmillError::new(
                    ErrorCode::UnsupportedQuantization,
                    format!(
                        "tensor {:?} uses {}, which tuckcore Stage 0 does not execute (supported: {})",
                        t.name,
                        t.type_name,
                        tensor::supported_type_names()
                    ),
                )
                .with_action("Use a supported type set; packing never re-quantizes."));
            }
        }
        let payload: u64 = insp.tensors.iter().map(|t| t.byte_size).sum();
        if insp.arch_id().as_deref() == Some("qwen35") {
            let note = self.load_qwen35(
                plan,
                artifact,
                &opts,
                kernels,
                &insp,
                &artifact_file,
                EngineLoadContext {
                    payload,
                    content_verified,
                    counters,
                    load_start,
                },
            )?;
            return Ok(note);
        }
        let registry = AdapterRegistry::builtin()?;
        let (cfg, _adapter) = model::resolve(&insp, &registry)?;
        if metal_mode {
            return self.load_qwen3_engine(
                plan,
                artifact,
                &opts,
                &cfg,
                kv_precision,
                &insp,
                &artifact_file,
                EngineLoadContext {
                    payload,
                    content_verified,
                    counters,
                    load_start,
                },
            );
        }
        let kv_bound = KvState::bytes_for(&cfg, plan.context.total_context);
        if kv_bound > KvState::MAX_KV_BYTES {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!(
                    "KV state for context {} needs {} bytes, beyond the stage-0 executor bound",
                    plan.context.total_context,
                    tuckmill_manifest::units::Bytes(kv_bound)
                ),
            ));
        }
        // ---- plan arena enforcement (B3) ----
        // Every live arena this load will own must fit the reservation the
        // supervisor already admitted against the process budget. An
        // understated or foreign plan is refused before anything allocates.
        let kv_reserved = plan.budget.reservation_breakdown.kv_state.bytes();
        if kv_bound > kv_reserved {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!(
                    "the plan reserves {} for KV state but this executor allocates {} for the admitted context {} in its f32 store; refusing to allocate beyond the acknowledged plan",
                    Bytes(kv_reserved),
                    Bytes(kv_bound),
                    plan.context.total_context
                ),
            )
            .with_action(
                "Rebuild the plan from the executor's resolved KV representation (4 bytes per element); understated reservations are never loaded.",
            ));
        }
        let weight_allowance = plan
            .budget
            .reservation_breakdown
            .weights
            .bytes()
            .saturating_add(plan.budget.reservation_breakdown.io_buffers.bytes());
        // A cache larger than the payload is never useful; clamp before
        // charging it so the check prices the allocation that can happen.
        let effective_cache = policy.cache_bytes.min(payload);
        let cache_wanted = effective_cache.saturating_add(policy.block_bytes);
        if cache_wanted > weight_allowance {
            return Err(TuckmillError::new(
                ErrorCode::InvalidPlan,
                format!(
                    "the plan reserves {} for weights + I/O but this load requests up to {} (cache {} clamped to payload {} plus one block of {}); refusing an unacknowledged weight-cache allocation",
                    Bytes(weight_allowance),
                    Bytes(cache_wanted),
                    Bytes(policy.cache_bytes),
                    Bytes(payload),
                    Bytes(policy.block_bytes)
                ),
            )
            .with_action(
                "Raise --memory-budget so the plan reserves resident weights, lower --context, or configure a smaller weight cache — this plan's mode does not cover the executor's live arenas.",
            ));
        }
        let mut names = model::TensorNames::for_layers(cfg.layers);
        names.resolve(&insp)?;
        // The worker only decodes (requests carry token ids): keep the
        // compact decode table, not the encoder.
        let tokenizer = DecodeVocab::from_gguf(&insp)?;

        let entries = forward::tensor_entries(&insp, policy.block_bytes);
        let loader = Arc::new(BlockLoader::open_file(
            artifact_file,
            insp.data_start,
            entries,
            policy,
            storage_selection,
            Some(artifact.file_bytes),
        )?);
        let tracker = Arc::new(CompletionTracker::new());
        let weights = forward::prepare(&loader, cfg.clone(), names)?;

        let mut note = format!(
            "tuckcore (experimental, uncertified): dense Qwen3, {} layers, vocab {}, byte-level BPE tokenizer, kernels {} [{}] (K01 scope: Q8_0/F16 CPU fused dot; Q4 and Metal unbuilt), weights {}; kv {}; real weights from {} ({} payload), weight cache {}, block {}",
            cfg.layers,
            cfg.vocab,
            kernels.as_str(),
            profile.as_str(),
            loader.storage_mode().as_str(),
            kv_precision.as_str(),
            artifact.path,
            tuckmill_manifest::units::Bytes(payload),
            tuckmill_manifest::units::Bytes(policy.cache_bytes),
            tuckmill_manifest::units::Bytes(policy.block_bytes)
        );
        note.push_str(&format!(
            "; payload content hash {}",
            if content_verified {
                "verified"
            } else {
                "not verified (no digest recorded or verification disabled by option)"
            }
        ));
        if let Some(extra) = tokenizer.pretokenizer_note() {
            note.push_str("; ");
            note.push_str(&extra);
        }
        ProfileCounters::record(&counters.load_nanos, load_start);
        self.engine = None;
        self.loaded = Some(Loaded {
            loader,
            tracker,
            weights,
            tokenizer,
            context_total: plan.context.total_context,
            profile,
            kv_precision,
            counters,
            kernels,
            session_kv_bytes: Arc::new(AtomicU64::new(0)),
            retained_weight_bytes: cache_wanted,
        });
        crate::release_freed_heap();
        Ok(note)
    }

    fn generate(&mut self, req: &GenerateRequest, ctl: &mut GenControl<'_>) -> Result<GenOutcome> {
        let context_total = match (&self.loaded, &self.engine) {
            (Some(l), _) => l.context_total,
            (None, Some(h)) => h.context_total,
            (None, None) => {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "tuckcore has no loaded artifact",
                ))
            }
        };
        let prompt_len = req.prompt_tokens.len() as u64;
        if prompt_len == 0 {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "prompt must contain at least one token",
            ));
        }
        if req.max_output_tokens == 0 {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "max_output_tokens must be positive",
            ));
        }
        if prompt_len + req.max_output_tokens > context_total {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!(
                    "prompt {prompt_len} + max output {} exceeds admitted context {context_total}",
                    req.max_output_tokens
                ),
            ));
        }

        if let Some(hybrid) = &self.engine {
            // Every request starts from a reset session, except that with
            // the prefix cache a prompt extending the snapshotted prefix
            // resumes from the snapshot (state that is exactly what a cold
            // prefill of those tokens produces; ADR-012 step 2).
            let start = hybrid.model.begin_request(&req.prompt_tokens)?;
            hybrid
                .last_prefix_hit
                .store(start as u64, Ordering::Relaxed);
            hybrid
                .session_kv_bytes
                .store(hybrid.alloc.state, Ordering::Relaxed);
            let mut engine = StepEngine::Engine {
                model: &hybrid.model,
            };
            let spec = SessionSpec {
                tokenizer: &hybrid.tokenizer,
                eos_token_id: hybrid.eos_token_id,
                prefill_batch: engine_metal::PREFILL_BATCH,
                start,
                per_token_delay: self.per_token_delay,
            };
            return run_session(&mut engine, &spec, req, ctl);
        }
        let Some(loaded) = self.loaded.as_ref() else {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                "tuckcore has no loaded artifact",
            ));
        };
        // Control closure for the forward loops: services external input
        // (poll) and applies governor-ordered cache trims at safe points.
        // `poll` is copied out so the closure never borrows `ctl` mutably.
        let poll_hook = ctl.poll;
        let control = || {
            if let Some(poll) = poll_hook {
                poll();
            }
            let _ = self.apply_pending_trim();
        };
        // Per-request KV session: allocated for this request's actual
        // admitted need (prompt + output), bounded by the prepared capacity.
        // A warm-reused worker never allocates for an older, larger
        // prepared context than the one this request was priced against.
        let session_context = prompt_len + req.max_output_tokens;
        let session_kv_bytes = KvState::bytes_for(&loaded.weights.cfg, session_context);
        loaded
            .session_kv_bytes
            .store(session_kv_bytes, Ordering::Relaxed);
        let kv = KvState::new(&loaded.weights.cfg, session_context, loaded.kv_precision)?;
        let mut engine = StepEngine::Cpu {
            fwd: Box::new(Forward {
                loader: &loaded.loader,
                tracker: &loaded.tracker,
                weights: &loaded.weights,
                profile: loaded.profile,
                kernels: loaded.kernels,
                kv_precision: loaded.kv_precision,
                control: &control,
                counters: if loaded.counters.enabled {
                    Some(&loaded.counters)
                } else {
                    None
                },
            }),
            kv,
        };

        let spec = SessionSpec {
            tokenizer: &loaded.tokenizer,
            eos_token_id: loaded.weights.cfg.eos_token_id,
            prefill_batch: 1,
            start: 0,
            per_token_delay: self.per_token_delay,
        };
        run_session(&mut engine, &spec, req, ctl)
    }

    fn score(
        &mut self,
        tokens: &[u32],
        positions: &[u64],
        cancelled: &dyn Fn() -> bool,
        sink: &mut dyn FnMut(u64, &[f32]) -> bool,
    ) -> Result<u64> {
        if let Some(hybrid) = &self.engine {
            return Self::score_engine(hybrid, tokens, positions, cancelled, sink);
        }
        let loaded = self.loaded.as_ref().ok_or_else(|| {
            TuckmillError::new(ErrorCode::WorkerFailed, "tuckcore has no loaded artifact")
        })?;
        let control = || {
            let _ = self.apply_pending_trim();
        };
        let fwd = Forward {
            loader: &loaded.loader,
            tracker: &loaded.tracker,
            weights: &loaded.weights,
            profile: loaded.profile,
            kernels: loaded.kernels,
            kv_precision: loaded.kv_precision,
            control: &control,
            counters: if loaded.counters.enabled {
                Some(&loaded.counters)
            } else {
                None
            },
        };
        let mut scored = 0u64;
        for position in positions {
            if cancelled() {
                return Err(TuckmillError::new(
                    ErrorCode::Cancelled,
                    "scoring cancelled at a position boundary",
                ));
            }
            let pos = *position as usize;
            let logits = fwd.logits_at(tokens, pos, cancelled)?;
            scored += 1;
            if !sink(*position, &logits) {
                break;
            }
        }
        Ok(scored)
    }

    fn trim_epoch(&self) -> Arc<std::sync::atomic::AtomicU64> {
        self.trim_epoch.clone()
    }

    fn stats(&self) -> BackendStats {
        if let Some(hybrid) = &self.engine {
            return Self::engine_stats(hybrid);
        }
        match &self.loaded {
            Some(loaded) => {
                let s = loaded.loader.stats();
                let residency = loaded.loader.residency();
                let storage = Some(StorageStats {
                    mode: s.storage_mode.clone(),
                    owned_cache_bytes: s.owned_cache_bytes,
                    owned_in_flight_bytes: loaded.tracker.in_flight_owned_bytes(),
                    mapped_vm_bytes: s.mapped_vm_bytes,
                    mapped_resident_bytes: residency.as_ref().map(|r| r.resident_bytes),
                    mapped_resident_sampled: residency.as_ref().map(|r| r.sampled).unwrap_or(false),
                    page_size_bytes: residency.as_ref().map(|r| r.page_size_bytes).unwrap_or(0),
                    page_advice_bytes: s.page_advice_bytes,
                });
                BackendStats {
                    logical_read_bytes: s.logical_read_bytes,
                    blocks_loaded: s.blocks_loaded,
                    cache_hits: s.cache_hits,
                    cache_misses: s.cache_misses,
                    evictions: s.cache_entry_evictions,
                    in_flight_leases: loaded.tracker.in_flight() as u64,
                    buffer_releases: Some(loaded.tracker.completed()),
                    released_bytes: Some(loaded.tracker.released_bytes()),
                    storage,
                    footprint_bytes: None,
                    kv_precision: Some(loaded.kv_precision.as_str().to_string()),
                    kv_state_bytes: Some(KvState::bytes_for(
                        &loaded.weights.cfg,
                        loaded.context_total,
                    )),
                    session_kv_bytes: match loaded.session_kv_bytes.load(Ordering::Relaxed) {
                        0 => None,
                        bytes => Some(bytes),
                    },
                    retained_weight_bytes: Some(loaded.retained_weight_bytes),
                    // Dense KV is allocated per request; nothing is held.
                    retained_state_bytes: None,
                    trims_applied: Some(loaded.counters.trims_applied.load(Ordering::Relaxed)),
                    gpu_allocated_bytes: None,
                    retained_workspace_bytes: None,
                    prefill_batch: None,
                    threads: Some(1), // scalar executor is single-threaded
                    kernel_impl: Some(loaded.kernels.as_str().to_string()),
                    numerical_profile: Some(loaded.profile.as_str().to_string()),
                    profile: if loaded.counters.enabled {
                        let c = &loaded.counters;
                        let ns = |a: &std::sync::atomic::AtomicU64| {
                            a.load(Ordering::Relaxed) as f64 / 1e9
                        };
                        Some(serde_json::json!({
                            "enabled": true,
                            "load_seconds": ns(&c.load_nanos),
                            "content_verify_seconds": ns(&c.content_verify_nanos),
                            "forward_seconds": ns(&c.forward_nanos),
                            "matvec_seconds": ns(&c.matvec_nanos),
                            "matvec_fetch_seconds": ns(&c.matvec_fetch_nanos),
                            "matvec_compute_seconds": ns(&c.matvec_compute_nanos),
                            "embed_seconds": ns(&c.embed_nanos),
                            "attention_seconds": ns(&c.attention_nanos),
                            "norm_rope_seconds": ns(&c.norm_rope_nanos),
                            "ffn_seconds": ns(&c.ffn_nanos),
                            "tokens_processed": c.tokens.load(Ordering::Relaxed),
                            "profile": loaded.profile.as_str(),
                            "kv_precision": loaded.kv_precision.as_str(),
                        }))
                    } else {
                        None
                    },
                }
            }
            None => BackendStats::default(),
        }
    }

    fn shutdown(&mut self) {
        self.engine = None;
        if let Some(loaded) = self.loaded.take() {
            loaded.loader.evict_all();
            // Leases must have drained; anything left would be a bug.
            debug_assert_eq!(loaded.tracker.in_flight(), 0);
        }
    }
}

/// What the shared decode loop needs from a loaded model family.
struct SessionSpec<'a> {
    tokenizer: &'a DecodeVocab,
    eos_token_id: Option<u32>,
    prefill_batch: usize,
    /// Prompt positions already in the session (a prefix-cache hit);
    /// prefill starts here.
    start: usize,
    per_token_delay: Duration,
}

/// The request loop shared by every model family: chunked prefill, typed
/// sampling, stop/EOS-correct visible output, and production-based timing.
/// The engine decides the arithmetic; this loop decides nothing about it.
fn run_session(
    engine: &mut StepEngine<'_>,
    spec: &SessionSpec<'_>,
    req: &GenerateRequest,
    ctl: &mut GenControl<'_>,
) -> Result<GenOutcome> {
    let prompt_len = req.prompt_tokens.len() as u64;
    let cancelled = {
        let cancel = ctl.cancel.clone();
        move || cancel.load(Ordering::Relaxed)
    };
    // Worker-side timing: measured inside the request, so supervisor
    // wall time minus this is queueing/transport, never compute.
    let t0 = Instant::now();
    let empty_timing = |t0: Instant| RequestTiming {
        total_ms: t0.elapsed().as_secs_f64() * 1000.0,
        prefill_ms: None,
        first_token_ms: None,
        decode_ms: None,
    };

    (ctl.on_state)(RunState::Warming);
    pause_loop(ctl, Duration::from_millis(1))?;
    if ctl.cancel.load(Ordering::Relaxed) {
        return Ok(GenOutcome::new(FinishReason::Cancelled, 0).with_timing(empty_timing(t0)));
    }

    (ctl.on_state)(RunState::Prefilling);
    // Chunked prefill: the Metal path processes up to its prepared
    // micro-batch per dispatch group; the CPU path keeps single-token
    // steps. Vocabulary logits are computed for the final chunk only.
    let prefill_batch = spec.prefill_batch;
    let mut logits: Vec<f32> = Vec::new();
    // Greedy requests on an engine that can choose on the device never
    // download logits: the step returns the token the CPU greedy rule
    // would pick (`sampling::argmax`, verified equal).
    let device_greedy = engine.device_greedy() && req.sampling.temperature <= 0.0;
    let mut next: Option<Option<u32>> = None;
    // The pipelined step already queued (it computes the token after the
    // one being collected).
    let mut in_flight: Option<PendingToken> = None;
    // A token a speculative cycle already verified (its accepted draft).
    let mut verified: Option<u32> = None;
    let prompt_tokens = &req.prompt_tokens;
    let mut pos = spec.start;
    while pos < prompt_tokens.len() {
        let take = prefill_batch.min(prompt_tokens.len() - pos);
        let want = pos + take == prompt_tokens.len();
        pause_loop(ctl, Duration::ZERO)?;
        let chunk = &prompt_tokens[pos..pos + take];
        let result = if device_greedy {
            engine
                .prefill_chunk_greedy(chunk, pos, want, &cancelled)
                .map(|choice| {
                    if choice.is_some() {
                        next = choice;
                    }
                    None
                })
        } else {
            engine.prefill_chunk(chunk, pos, want, &cancelled)
        };
        match result {
            Ok(Some(l)) => logits = l,
            Ok(None) => {}
            Err(e) if e.code() == ErrorCode::Cancelled => {
                return Ok(GenOutcome::new(FinishReason::Cancelled, 0).with_timing(empty_timing(t0)))
            }
            Err(e) => return Err(e),
        }
        pos += take;
    }
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

    (ctl.on_state)(RunState::Generating);
    // Typed sampling policy from the request: greedy when temperature is
    // zero; the declared transforms and seed otherwise. Unsupported
    // values are refused before any compute, never ignored.
    req.sampling.validate()?;
    let sampling_cfg = SamplingConfig {
        temperature: req.sampling.temperature,
        top_k: None,
        top_p: req.sampling.top_p,
    };
    let mut rng = Rng::new(req.seed);
    let mut generated_ids: u64 = 0;
    let mut release_index: u64 = 0;
    // Model-token production timing (the honest basis for decode
    // throughput); visible-content timing stays a supervisor-side,
    // user-experience concern and is never mixed into it.
    let mut first_generated_at: Option<Instant> = None;
    let mut last_generated_at: Option<Instant> = None;
    // Incremental UTF-8 assembly with byte-level token provenance:
    // multibyte characters split across tokens are held until complete,
    // and every token that contributed a byte to a visible character is
    // counted, never just the token that happened to complete it.
    let mut decoder = Utf8StreamDecoder::new();
    // Bounded pending-output buffering: a stop sequence spanning several
    // tokens (or completing inside a piece, with trailing text after it)
    // is held until it is found, so no stop marker can leak into the
    // visible stream.
    let mut buffer = VisibleOutputBuffer::new(&req.stop);
    // Emit a release: deliver the chunk and report whether the client
    // asked to stop.
    let emit = |text: &str, release_index: &mut u64, ctl: &mut GenControl<'_>| -> bool {
        if text.is_empty() {
            return true;
        }
        let keep = (ctl.on_token)(*release_index, text);
        *release_index += 1;
        keep
    };
    let outcome = 'decode: {
        for i in 0..req.max_output_tokens {
            if ctl.cancel.load(Ordering::Relaxed) {
                let r = buffer.flush();
                let _ = emit(&r.text, &mut release_index, ctl);
                break 'decode GenOutcome::new(FinishReason::Cancelled, buffer.visible_tokens())
                    .with_generated(generated_ids);
            }
            pause_loop(ctl, spec.per_token_delay)?;

            let choice = if device_greedy {
                next.take().flatten()
            } else {
                sampling::sample(&logits, &sampling_cfg, &mut rng)
            };
            let Some(token) = choice else {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "sampling produced no token: every logit was non-finite",
                ));
            };
            generated_ids = i + 1;
            let produced_at = Instant::now();
            if first_generated_at.is_none() {
                first_generated_at = Some(produced_at);
            }
            last_generated_at = Some(produced_at);
            let is_eos = Some(token) == spec.eos_token_id;
            if is_eos && !req.ignore_eos {
                // The model-defined EOS token is never visible: its
                // bytes are not pushed, and anything held is flushed
                // because no stop sequence can complete after EOS.
                let r = buffer.flush();
                let _ = emit(&r.text, &mut release_index, ctl);
                break 'decode GenOutcome::new(FinishReason::Eos, buffer.visible_tokens())
                    .with_generated(generated_ids);
            }
            // Benchmark workloads (`ignore_eos`): EOS stays invisible but
            // does not end generation; it is fed back like any token.
            let piece = match (is_eos, spec.tokenizer.contains(token)) {
                (true, _) => Vec::new(),
                (false, true) => decoder.push_token(spec.tokenizer.token_bytes(token), i),
                (false, false) => {
                    let placeholder = format!("<tok {token}>");
                    decoder.push_token(placeholder.as_bytes(), i)
                }
            };
            let mut client_stop = false;
            let mut stop = false;
            for run in piece {
                let release = buffer.push(&run.owners, &run.text);
                if !emit(&release.text, &mut release_index, ctl) {
                    client_stop = true;
                    break;
                }
                if release.stop {
                    stop = true;
                    break;
                }
            }
            if client_stop {
                let r = buffer.flush();
                let _ = emit(&r.text, &mut release_index, ctl);
                break 'decode GenOutcome::new(FinishReason::Stop, buffer.visible_tokens())
                    .with_generated(generated_ids);
            }
            if stop {
                break 'decode GenOutcome::new(FinishReason::Stop, buffer.visible_tokens())
                    .with_generated(generated_ids);
            }
            if i + 1 < req.max_output_tokens {
                // The generated token i occupies position prompt_len + i.
                let pos = (prompt_len + i) as usize;
                let result = match (device_greedy, engine.engine_model()) {
                    // Pipelined: queue step i+1 (reading step i's choice on
                    // the device) before waiting for step i, so the host's
                    // encoding overlaps the GPU's work.
                    (true, Some(model)) => (|| -> Result<()> {
                        if cancelled() {
                            return Err(TuckmillError::new(
                                ErrorCode::Cancelled,
                                "generation cancelled at a token boundary",
                            ));
                        }
                        // Speculative greedy (ADR-012 step 3): one cycle
                        // yields the next token and, when the draft was
                        // right, the one after it, both exactly the tokens
                        // plain greedy decoding chooses.
                        if let Some(t) = verified.take() {
                            next = Some(Some(t));
                            return Ok(());
                        }
                        if in_flight.is_none() && model.can_speculate() {
                            let step = model.speculate_greedy(token)?;
                            next = Some(step.first);
                            verified = step.second;
                            return Ok(());
                        }
                        let current = match in_flight.take() {
                            Some(p) => p,
                            None => model.submit_greedy(GreedyInput::Token(token))?,
                        };
                        if i + 2 < req.max_output_tokens {
                            in_flight = Some(model.submit_greedy(GreedyInput::Previous(current))?);
                        }
                        next = Some(model.collect_greedy(current)?);
                        Ok(())
                    })(),
                    _ => engine.step(token, pos, &cancelled, &mut logits),
                };
                match result {
                    Ok(()) => {}
                    Err(e) if e.code() == ErrorCode::Cancelled => {
                        let r = buffer.flush();
                        let _ = emit(&r.text, &mut release_index, ctl);
                        break 'decode GenOutcome::new(
                            FinishReason::Cancelled,
                            buffer.visible_tokens(),
                        )
                        .with_generated(generated_ids);
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        let r = buffer.flush();
        let _ = emit(&r.text, &mut release_index, ctl);
        GenOutcome::new(FinishReason::Length, buffer.visible_tokens()).with_generated(generated_ids)
    };
    let ms = |from: Instant, to: Instant| (to - from).as_secs_f64() * 1000.0;
    let timing = RequestTiming {
        total_ms: ms(t0, Instant::now()),
        prefill_ms: Some(prefill_ms),
        // Production-based: time to the first sampled token, and the
        // span between first and last sampled token. Buffering and
        // visibility decisions cannot inflate model throughput.
        first_token_ms: first_generated_at.map(|f| ms(t0, f)),
        decode_ms: match (first_generated_at, last_generated_at) {
            (Some(f), Some(l)) if l > f => Some(ms(f, l)),
            _ => None,
        },
    };
    Ok(outcome.with_timing(timing))
}

/// One request's execution session: the CPU forward with its KV, or the
/// unified engine on Metal. Selection happened at load (`--kernels`); there
/// is no per-step fallback.
enum StepEngine<'a> {
    Cpu {
        fwd: Box<Forward<'a>>,
        kv: KvState,
    },
    /// The unified engine on the GPU: the model owns its (reset) session
    /// state and its position counter; tokens are processed strictly in
    /// order.
    Engine {
        model: &'a EngineModel,
    },
}

impl<'a> StepEngine<'a> {
    /// One decode step into `logits` (reused across steps: the hybrid
    /// engine fills it in place instead of allocating per token).
    fn step(
        &mut self,
        token: u32,
        pos: usize,
        cancelled: &dyn Fn() -> bool,
        logits: &mut Vec<f32>,
    ) -> Result<()> {
        match self {
            StepEngine::Cpu { fwd, kv } => {
                *logits = fwd.forward_token(kv, token, pos, cancelled)?;
            }
            StepEngine::Engine { model } => engine_step(model, token, pos, cancelled, logits)?,
        }
        Ok(())
    }

    /// Whether greedy choices can be made on the device (only the 4-byte
    /// token crosses to the host); identical to the CPU greedy rule.
    fn device_greedy(&self) -> bool {
        matches!(self, StepEngine::Engine { .. })
    }

    /// Device-greedy prefill chunk: `Some(choice)` for the final chunk
    /// (`choice` is `None` when every logit was NaN), `None` otherwise.
    fn prefill_chunk_greedy(
        &mut self,
        tokens: &[u32],
        start: usize,
        last: bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Option<u32>>> {
        let StepEngine::Engine { model } = self else {
            return Err(device_greedy_unsupported());
        };
        engine_chunk_preamble(model, start, cancelled)?;
        if last {
            Ok(Some(model.prefill_chunk_greedy(tokens)?))
        } else {
            model.prefill_chunk(tokens, false, &mut Vec::new())?;
            Ok(None)
        }
    }

    /// The engine model, when this session runs on the unified engine
    /// (pipelined greedy decoding talks to it directly).
    fn engine_model(&self) -> Option<&'a EngineModel> {
        match self {
            StepEngine::Engine { model } => Some(*model),
            _ => None,
        }
    }

    /// Process a prefix chunk; logits are produced only when requested
    /// (the last chunk of a prefill). The CPU arm keeps its single-token
    /// semantics; the engine arm batches the whole chunk.
    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        start: usize,
        want_logits: bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>> {
        match self {
            StepEngine::Cpu { fwd, kv } => {
                let mut last = None;
                for (i, token) in tokens.iter().enumerate() {
                    last = Some(fwd.forward_token(kv, *token, start + i, cancelled)?);
                }
                Ok(if want_logits { last } else { None })
            }
            // Batched prefill (Q35-2D): one command buffer per chunk of at
            // most PREFILL_BATCH tokens; cancellation is honored between
            // chunks.
            StepEngine::Engine { model } => {
                engine_chunk_preamble(model, start, cancelled)?;
                let mut logits = Vec::new();
                model.prefill_chunk(tokens, want_logits, &mut logits)?;
                Ok(if want_logits { Some(logits) } else { None })
            }
        }
    }
}

/// One in-order hybrid step. Cancellation is honored at every token
/// boundary; a position that disagrees with the model's own counter is an
/// internal fault, never silently re-aligned.
fn device_greedy_unsupported() -> TuckmillError {
    TuckmillError::new(
        ErrorCode::Internal,
        "device greedy selection requested on an engine without it",
    )
}

/// Chunk-boundary checks shared by the hybrid prefill paths.
fn engine_chunk_preamble(
    model: &EngineModel,
    start: usize,
    cancelled: &dyn Fn() -> bool,
) -> Result<()> {
    if cancelled() {
        return Err(TuckmillError::new(
            ErrorCode::Cancelled,
            "prefill cancelled at a chunk boundary",
        ));
    }
    if model.positions() != start {
        return Err(TuckmillError::new(
            ErrorCode::Internal,
            format!(
                "engine prefill at position {start} but the session holds {} positions",
                model.positions()
            ),
        ));
    }
    Ok(())
}

/// Token-boundary checks shared by the hybrid decode paths.
fn engine_preamble(model: &EngineModel, pos: usize, cancelled: &dyn Fn() -> bool) -> Result<()> {
    if cancelled() {
        return Err(TuckmillError::new(
            ErrorCode::Cancelled,
            "generation cancelled at a token boundary",
        ));
    }
    if model.positions() != pos {
        return Err(TuckmillError::new(
            ErrorCode::Internal,
            format!(
                "engine step at position {pos} but the session holds {} positions",
                model.positions()
            ),
        ));
    }
    Ok(())
}

fn engine_step(
    model: &EngineModel,
    token: u32,
    pos: usize,
    cancelled: &dyn Fn() -> bool,
    logits: &mut Vec<f32>,
) -> Result<()> {
    engine_preamble(model, pos, cancelled)?;
    model.forward_token(token, logits)
}

/// Identity of the hybrid device path: the Q35-2C GPU model over raw
/// Q8_0/F16 weights, resolved implementation "metal".
const HYBRID_KERNEL_ID: &str = "Q35-2C:qwen35-q8_0+f16:metal";

/// Identity of dense Qwen3 on the unified engine (ADR-011 step 2), over raw
/// Q8_0/F16 weights, resolved implementation "metal".
const QWEN3_ENGINE_KERNEL_ID: &str = "ADR011:qwen3-q8_0+f16:metal";

/// Load-time facts the shared verification prologue hands to the hybrid
/// loader.
struct EngineLoadContext {
    payload: u64,
    content_verified: bool,
    counters: ProfileCounters,
    load_start: Instant,
}

impl TuckCore {
    /// Q35-1E.2: load a verified `qwen35` artifact onto the resident Metal
    /// path. Every option that does not apply is refused rather than
    /// ignored, and every GPU allocation class is checked against its plan
    /// reservation from the exact preflight before any buffer exists.
    #[allow(clippy::too_many_arguments)]
    fn load_qwen35(
        &mut self,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        opts: &TuckCoreOptions,
        kernels: ResolvedKernels,
        insp: &tuckmill_manifest::gguf::GgufInspection,
        artifact_file: &std::fs::File,
        ctx: EngineLoadContext,
    ) -> Result<String> {
        if kernels != ResolvedKernels::Metal {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "qwen35 executes on the resident Metal path only in this build (resolved kernels: {}); the CPU forward is the test oracle, not a served path",
                    kernels.as_str()
                ),
            )
            .with_action("Select --kernels metal."));
        }
        if plan.mode != ExecutionMode::Resident {
            return Err(TuckmillError::new(
                ErrorCode::InvalidPlan,
                format!(
                    "the qwen35 Metal path is resident-only, but the plan mode is {}; streaming is not implemented for this architecture",
                    plan.mode.as_str()
                ),
            )
            .with_action("Raise --memory-budget so the plan is resident, or lower --context."));
        }
        if opts.prefill_batch.is_some() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "qwen35 prefills in fixed chunks of 16 tokens in this build; prefill_batch must not be set",
            ));
        }
        if opts.weight_cache_bytes.is_some() || opts.block_bytes.is_some() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "weight-cache and block options apply to CPU weight streaming; the qwen35 Metal path keeps weights resident",
            ));
        }
        match opts.kv_precision.as_deref() {
            None | Some("f16") => {}
            Some("f32") => {
                return Err(TuckmillError::new(
                    ErrorCode::UnsupportedArchitecture,
                    "the qwen35 GPU KV cache stores packed f16 only; f32 KV is not implemented on this path and is never substituted",
                )
                .with_action("Use the default (f16) KV precision for qwen35."))
            }
            Some(other) => {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!("unknown kv_precision {other:?}; use \"f32\" or \"f16\""),
                ))
            }
        }

        // The tensor source reads through the verified handle.
        // Reuses the identity-checked inspection (hashes vs the record, full
        // content digest of this handle) instead of parsing it again.
        let source = ArtifactTensorSource::with_inspection(artifact_file, insp)?;
        let graph = crate::engine::adapters::qwen35(source.config(), source.layout());
        self.load_engine(
            plan,
            artifact,
            insp,
            graph,
            &source,
            KvCache::F16,
            opts.prefix_cache.unwrap_or(true),
            opts.speculative.unwrap_or(false),
            HYBRID_KERNEL_ID,
            ctx,
        )
    }

    /// ADR-011 step 2: load a verified dense `qwen3` artifact onto the
    /// unified engine (`--kernels metal`). Options that do not apply are
    /// refused; the declared KV precision (f32 by default, as on the CPU
    /// path) is honored exactly.
    #[allow(clippy::too_many_arguments)]
    fn load_qwen3_engine(
        &mut self,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        opts: &TuckCoreOptions,
        cfg: &model::Qwen3Config,
        kv_precision: KvPrecision,
        insp: &tuckmill_manifest::gguf::GgufInspection,
        artifact_file: &std::fs::File,
        ctx: EngineLoadContext,
    ) -> Result<String> {
        if plan.mode != ExecutionMode::Resident {
            return Err(TuckmillError::new(
                ErrorCode::InvalidPlan,
                format!(
                    "the Metal path is resident-only, but the plan mode is {}; GPU streaming is not implemented (the CPU kernels stream)",
                    plan.mode.as_str()
                ),
            )
            .with_action("Raise --memory-budget so the plan is resident, lower --context, or use the CPU kernels."));
        }
        if opts.prefill_batch.is_some() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!(
                    "the unified engine prefills in fixed chunks of {} tokens; prefill_batch must not be set",
                    engine_metal::PREFILL_BATCH
                ),
            ));
        }
        if opts.weight_cache_bytes.is_some() || opts.block_bytes.is_some() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "weight-cache and block options apply to CPU weight streaming; the Metal path keeps weights resident",
            ));
        }
        let mut names = model::TensorNames::for_layers(cfg.layers);
        names.resolve(insp)?;
        let graph = crate::engine::adapters::qwen3(cfg, &names, insp)?;
        let source = crate::engine::GgufWeights::with_inspection(artifact_file, insp)?;
        let kv = match kv_precision {
            KvPrecision::F32 => KvCache::F32,
            KvPrecision::F16 => KvCache::F16,
        };
        self.load_engine(
            plan,
            artifact,
            insp,
            graph,
            &source,
            kv,
            opts.prefix_cache.unwrap_or(true),
            opts.speculative.unwrap_or(false),
            QWEN3_ENGINE_KERNEL_ID,
            ctx,
        )
    }

    /// The load shared by every engine-served family: per-class plan
    /// checks before any GPU allocation, the exact preflighted upload, and
    /// the ledger == plan check after it.
    #[allow(clippy::too_many_arguments)]
    fn load_engine(
        &mut self,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        insp: &tuckmill_manifest::gguf::GgufInspection,
        graph: crate::engine::ModelGraph,
        source: &dyn crate::engine::WeightSource,
        kv: KvCache,
        prefix_cache: bool,
        speculative: bool,
        kernel_id: &str,
        ctx: EngineLoadContext,
    ) -> Result<String> {
        let capacity = usize::try_from(plan.context.total_context).map_err(|_| {
            TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                "admitted context does not fit this platform's address space",
            )
        })?;
        let arch = graph.architecture.clone();
        let engine_opts = EngineOptions {
            kv,
            prefix_cache,
            mtp: speculative,
        };
        let alloc = engine_metal::alloc_plan_with(&graph, source, capacity, engine_opts)?;
        let reserved = &plan.budget.reservation_breakdown;
        for (class, need, have) in [
            ("weights", alloc.weights, reserved.weights.bytes()),
            ("kv_state", alloc.state, reserved.kv_state.bytes()),
            ("workspaces", alloc.scratch, reserved.workspaces.bytes()),
        ] {
            if need > have {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidPlan,
                    format!(
                        "the plan reserves {} for {class} but the {arch} GPU model needs {} at context {}; refusing before any GPU allocation",
                        Bytes(have),
                        Bytes(need),
                        plan.context.total_context
                    ),
                )
                .with_action("Rebuild the plan for this executor; understated reservations are never loaded."));
            }
        }
        let model = EngineModel::load_with(
            Qwen35Gpu::open()?,
            graph,
            source,
            capacity,
            alloc.total(),
            engine_opts,
        )?;
        if model.allocated_bytes() != alloc.total() {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!(
                    "{arch} GPU ledger holds {} but the preflight priced {}; refusing an unaccounted allocation",
                    model.allocated_bytes(),
                    alloc.total()
                ),
            ));
        }
        let tokenizer = DecodeVocab::from_gguf(insp)?;
        let kv_note = match kv {
            KvCache::F16 => "kv f16 packed",
            KvCache::F32 => "kv f32",
        };
        let mut note = format!(
            "tuckcore (experimental, uncertified): unified engine (ADR-011), graph {} with {} layers, vocab {}, kernels metal [oracle-f32], {kv_note}, batched prefill x{}; real weights from {} ({} payload), GPU ledger {} = weights {} + state {} + scratch {}",
            model.graph().architecture,
            model.graph().layers.len(),
            model.graph().vocab,
            engine_metal::PREFILL_BATCH,
            artifact.path,
            Bytes(ctx.payload),
            Bytes(alloc.total()),
            Bytes(alloc.weights),
            Bytes(alloc.state),
            Bytes(alloc.scratch)
        );
        note.push_str(&format!(
            "; payload content hash {}",
            if ctx.content_verified {
                "verified"
            } else {
                "not verified (no digest recorded or verification disabled by option)"
            }
        ));
        if speculative {
            note.push_str(
                "; speculative greedy decoding with the artifact's MTP head (drafts verified with the decode arithmetic; output token-identical to plain greedy decoding)",
            );
        }
        if let Some(extra) = tokenizer.pretokenizer_note() {
            note.push_str("; ");
            note.push_str(&extra);
        }
        ProfileCounters::record(&ctx.counters.load_nanos, ctx.load_start);
        self.loaded = None;
        self.engine = Some(EngineLoaded {
            eos_token_id: model.graph().eos_token_id,
            model,
            kv,
            kernel_id: kernel_id.to_string(),
            tokenizer,
            context_total: plan.context.total_context,
            alloc,
            session_kv_bytes: Arc::new(AtomicU64::new(0)),
            last_prefix_hit: AtomicU64::new(0),
            prefix_cache,
            counters: ctx.counters,
        });
        crate::release_freed_heap();
        Ok(note)
    }

    /// Teacher-forced scoring on the hybrid path: the session is replayed
    /// in order from a reset state, so logits at a position always come
    /// from exactly the tokens before and at it.
    fn score_engine(
        hybrid: &EngineLoaded,
        tokens: &[u32],
        positions: &[u64],
        cancelled: &dyn Fn() -> bool,
        sink: &mut dyn FnMut(u64, &[f32]) -> bool,
    ) -> Result<u64> {
        let model = &hybrid.model;
        model.reset()?;
        let mut logits = Vec::new();
        let mut scored = 0u64;
        for position in positions {
            let pos = usize::try_from(*position).unwrap_or(usize::MAX);
            if tokens.is_empty() || pos >= tokens.len() {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    "scoring position is outside the token sequence",
                ));
            }
            if model.positions() > pos {
                model.reset()?;
            }
            while model.positions() <= pos {
                if cancelled() {
                    return Err(TuckmillError::new(
                        ErrorCode::Cancelled,
                        "scoring cancelled at a position boundary",
                    ));
                }
                model.forward_token(tokens[model.positions()], &mut logits)?;
            }
            scored += 1;
            if !sink(*position, &logits) {
                break;
            }
        }
        Ok(scored)
    }

    fn engine_stats(hybrid: &EngineLoaded) -> BackendStats {
        BackendStats {
            kv_precision: Some(hybrid.kv.as_str().to_string()),
            kv_state_bytes: Some(hybrid.alloc.state),
            session_kv_bytes: match hybrid.session_kv_bytes.load(Ordering::Relaxed) {
                0 => None,
                bytes => Some(bytes),
            },
            retained_weight_bytes: Some(hybrid.alloc.weights),
            retained_workspace_bytes: Some(hybrid.alloc.scratch),
            retained_state_bytes: Some(hybrid.alloc.state),
            trims_applied: Some(hybrid.counters.trims_applied.load(Ordering::Relaxed)),
            gpu_allocated_bytes: Some(hybrid.model.allocated_bytes()),
            prefill_batch: Some(engine_metal::PREFILL_BATCH as u32),
            kernel_impl: Some(ResolvedKernels::Metal.as_str().to_string()),
            numerical_profile: Some(NumericalProfile::OracleF32.as_str().to_string()),
            profile: Some(serde_json::json!({
                "prefix_cache": hybrid.prefix_cache,
                "last_prefix_hit_tokens": hybrid.last_prefix_hit.load(Ordering::Relaxed),
            })),
            ..BackendStats::default()
        }
    }
}

/// Kernel-package identifiers for identity records: K01 covers Q8_0/F16
/// CPU fused dot only (Q4 acceleration and Metal are not implemented); the
/// implementation suffix is the resolved selection, "auto" before a load.
fn kernel_ids(impl_: Option<ResolvedKernels>) -> Vec<String> {
    match impl_ {
        Some(k) => vec![format!("K01:q8_0+f16:{}", k.as_str())],
        None => vec!["K01:q8_0+f16:auto".to_string()],
    }
}

/// Same safe-boundary pause semantics as every backend.
fn pause_loop(ctl: &mut GenControl<'_>, mut remaining: Duration) -> Result<()> {
    let slice = Duration::from_millis(2);
    let mut paused_reported = false;
    loop {
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
        if remaining <= Duration::ZERO {
            return Ok(());
        }
        let step = remaining.min(slice);
        std::thread::sleep(step);
        remaining -= step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use tuckmill_backend::SamplingPolicy;
    use tuckmill_manifest::ids::Id;
    use tuckmill_manifest::plan::PlanContext;
    use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

    // Local alias to avoid importing the full plan module in every test.
    use tuckmill_manifest::plan::ExecutionPlan as Plan;

    fn fixture() -> (tempfile::TempDir, ArtifactRef, Plan) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let artifact = ArtifactRef {
            path: path.display().to_string(),
            file_bytes: insp.file_len,
            metadata_hash: insp.metadata_hash().unwrap().as_str().to_string(),
            tensor_index_hash: insp.tensor_index_hash().unwrap().as_str().to_string(),
            tokenizer_hash: insp
                .tokenizer_hash()
                .unwrap()
                .map(|h| h.as_str().to_string()),
            arch_id: Some("qwen3".into()),
            content_hash: None,
        };
        let mut plan = Plan::new_skeleton(
            Id::new("a".repeat(64)).unwrap(),
            Id::new("b".repeat(64)).unwrap(),
            "qwen3".into(),
        );
        plan.mode = ExecutionMode::Resident;
        plan.context = PlanContext::new(256, 64);
        // Arena reservations the executor acknowledges at load: KV priced at
        // the actual f32 representation, plus generous weights/IO for the
        // tiny fixture. Understated plans are exercised separately.
        plan.budget.requested_budget = tuckmill_manifest::units::Bytes::parse("2GiB").unwrap();
        plan.budget.add_reservation(
            tuckmill_manifest::records::ReserveClass::KvState,
            Bytes(64 * 1024 * 1024),
        );
        plan.budget.add_reservation(
            tuckmill_manifest::records::ReserveClass::Weights,
            Bytes(64 * 1024 * 1024),
        );
        plan.budget.add_reservation(
            tuckmill_manifest::records::ReserveClass::IoBuffers,
            Bytes(16 * 1024 * 1024),
        );
        plan.budget.add_reservation(
            tuckmill_manifest::records::ReserveClass::Workspaces,
            Bytes(64 * 1024 * 1024),
        );
        plan.plan_id = plan.compute_id().unwrap();
        (dir, artifact, plan)
    }

    fn run_generate(
        engine: &mut TuckCore,
        prompt: Vec<u32>,
        max_out: u64,
    ) -> (GenOutcome, Vec<String>) {
        run_generate_with(
            engine,
            prompt,
            max_out,
            vec![],
            7,
            SamplingPolicy::default(),
        )
    }

    fn run_generate_with(
        engine: &mut TuckCore,
        prompt: Vec<u32>,
        max_out: u64,
        stop: Vec<String>,
        seed: u64,
        sampling: SamplingPolicy,
    ) -> (GenOutcome, Vec<String>) {
        let cancel = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let mut texts = Vec::new();
        let mut on_token = |_i: u64, t: &str| {
            texts.push(t.to_string());
            true
        };
        let mut on_state = |_s: RunState| {};
        let mut ctl = GenControl::new(cancel.clone(), paused.clone(), &mut on_token, &mut on_state);
        let req = GenerateRequest {
            request_id: "t1".into(),
            prompt_tokens: prompt,
            max_output_tokens: max_out,
            stop,
            seed,
            sampling,
            ignore_eos: false,
        };
        let outcome = engine.generate(&req, &mut ctl).unwrap();
        (outcome, texts)
    }

    #[test]
    fn load_verifies_identity_and_reports_honestly() {
        let (_dir, artifact, plan) = fixture();
        let mut engine = TuckCore::new();
        let note = engine
            .load(
                &plan,
                &artifact,
                Some(&serde_json::json!({"weight_cache_bytes": 262144, "block_bytes": 65536})),
            )
            .unwrap();
        assert!(note.contains("uncertified"));
        assert!(
            note.contains("kernels neon [oracle-f32]")
                || note.contains("kernels scalar [oracle-f32]")
        );
        assert!(note.contains("K01 scope: Q8_0/F16 CPU fused dot"));
        let identity = engine.identity();
        assert_eq!(identity.engine, "tuckcore");
        assert!(!identity.certified);
        assert!(engine.is_loaded());

        // Wrong hash: refuse to execute.
        let mut bad = artifact.clone();
        bad.metadata_hash = "0".repeat(64);
        let err = engine.load(&plan, &bad, None).unwrap_err();
        assert_eq!(err.code(), ErrorCode::IntegrityFailure);
    }

    #[test]
    fn unknown_options_are_rejected_not_ignored() {
        let (_dir, artifact, plan) = fixture();
        let mut engine = TuckCore::new();
        let err = engine
            .load(
                &plan,
                &artifact,
                Some(&serde_json::json!({"cache_bytes": 1})),
            )
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidRequest);
    }

    #[test]
    fn generate_is_deterministic_and_bounded() {
        let (_dir, artifact, plan) = fixture();
        let mut engine = TuckCore::new();
        engine.load(&plan, &artifact, None).unwrap();
        let (a, ta) = run_generate(&mut engine, vec![1, 2, 3], 6);
        let (_b, tb) = run_generate(&mut engine, vec![1, 2, 3], 6);
        assert_eq!(a.finish_reason, FinishReason::Length);
        assert_eq!(a.committed, 6);
        assert_eq!(ta, tb, "greedy generation is deterministic");
        assert!(ta.iter().all(|t| !t.is_empty()));
        // Context overflow rejected before compute.
        let err = {
            let cancel = Arc::new(AtomicBool::new(false));
            let paused = Arc::new(AtomicBool::new(false));
            let mut on_token = |_i: u64, _t: &str| true;
            let mut on_state = |_s: RunState| {};
            let mut ctl =
                GenControl::new(cancel.clone(), paused.clone(), &mut on_token, &mut on_state);
            let req = GenerateRequest {
                request_id: "t2".into(),
                prompt_tokens: vec![1; 200],
                max_output_tokens: 200,
                stop: vec![],
                seed: 0,
                sampling: tuckmill_backend::SamplingPolicy::default(),
                ignore_eos: false,
            };
            engine.generate(&req, &mut ctl).unwrap_err()
        };
        assert_eq!(err.code(), ErrorCode::ContextExceedsBudget);
    }

    #[test]
    fn understated_plan_reservations_are_refused_before_allocation() {
        let (_dir, artifact, plan) = fixture();
        let mut engine = TuckCore::new();

        // KV understated (the pre-Gate-B "2 bytes per element" mistake).
        let mut understated = plan.clone();
        understated.budget.reservation_breakdown.kv_state = Bytes::ZERO;
        understated.budget.reservation = understated.budget.reservation_breakdown.total();
        understated.plan_id = understated.compute_id().unwrap();
        let err = engine.load(&understated, &artifact, None).unwrap_err();
        assert_eq!(err.code(), ErrorCode::ContextExceedsBudget);
        assert!(err.message().contains("acknowledged plan"));

        // Weight cache not covered by the weights + I/O reservation.
        let mut starved = plan.clone();
        starved.budget.reservation_breakdown.weights = Bytes(0);
        starved.budget.reservation_breakdown.io_buffers = Bytes(0);
        starved.budget.reservation = starved.budget.reservation_breakdown.total();
        starved.plan_id = starved.compute_id().unwrap();
        let err = engine.load(&starved, &artifact, None).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidPlan);

        // The fixture plan remains loadable and acknowledges its KV bytes
        // and the weight-cache allocation it can retain between requests.
        engine.load(&plan, &artifact, None).unwrap();
        let stats = engine.stats();
        assert_eq!(
            stats.kv_state_bytes,
            Some(KvState::bytes_for(
                &engine.loaded.as_ref().unwrap().weights.cfg,
                256
            ))
        );
        let payload: u64 = tuckmill_manifest::gguf::inspect(
            std::path::Path::new(&artifact.path),
            &tuckmill_manifest::gguf::Limits::default(),
        )
        .unwrap()
        .tensors
        .iter()
        .map(|t| t.byte_size)
        .sum();
        assert_eq!(
            stats.retained_weight_bytes,
            Some(payload.min(DEFAULT_CACHE_BYTES) + DEFAULT_BLOCK_BYTES),
            "the retained weight allocation is the cache clamped to the payload plus one block"
        );
    }

    /// Gate C2 end-to-end on the fixture: the Metal resident path must
    /// produce teacher-forced logits and greedy output matching the CPU
    /// scalar reference exactly where required (top-1, finish reason) and
    /// within the registered f32 reduction-order tolerance where GPU
    /// summation order legitimately differs. CI runs this because the
    /// runner has a GPU; a dev machine without one skips loudly.
    /// Hybrid (qwen35) fixture artifact with the same generous arena
    /// reservations as the dense fixture.
    fn hybrid_fixture() -> (tempfile::TempDir, ArtifactRef, Plan) {
        let (dir, mut artifact, plan) = fixture();
        let path = dir.path().join("q35.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen35()).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        artifact.path = path.display().to_string();
        artifact.file_bytes = insp.file_len;
        artifact.metadata_hash = insp.metadata_hash().unwrap().as_str().to_string();
        artifact.tensor_index_hash = insp.tensor_index_hash().unwrap().as_str().to_string();
        artifact.tokenizer_hash = insp
            .tokenizer_hash()
            .unwrap()
            .map(|h| h.as_str().to_string());
        artifact.arch_id = Some("qwen35".into());
        (dir, artifact, plan)
    }

    fn metal_or_skip() -> bool {
        if tuckmill_metal::probe().available {
            return true;
        }
        assert!(
            std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
            "TUCKMILL_REQUIRE_METAL is set but no Metal device is available"
        );
        eprintln!("skipping: no Metal device");
        false
    }

    #[test]
    fn hybrid_load_refuses_every_option_it_cannot_honor() {
        let (_dir, artifact, plan) = hybrid_fixture();
        assert!(TuckCore::new().capabilities().supports_arch("qwen35"));
        for (opts, code) in [
            (
                serde_json::json!({"kernels": "scalar"}),
                ErrorCode::UnsupportedArchitecture,
            ),
            (
                serde_json::json!({"kernels": "metal", "kv_precision": "f32"}),
                ErrorCode::UnsupportedArchitecture,
            ),
            (
                serde_json::json!({"kernels": "metal", "prefill_batch": 4}),
                ErrorCode::InvalidRequest,
            ),
            (
                serde_json::json!({"kernels": "metal", "weight_cache_bytes": 65536}),
                ErrorCode::InvalidRequest,
            ),
        ] {
            let mut engine = TuckCore::new();
            let err = engine.load(&plan, &artifact, Some(&opts)).unwrap_err();
            assert_eq!(err.code(), code, "{opts}: {err}");
            assert!(!engine.is_loaded(), "{opts}: nothing is loaded on refusal");
        }
    }

    #[test]
    fn hybrid_load_refuses_understated_reservations_before_allocating() {
        if !metal_or_skip() {
            return;
        }
        let (_dir, artifact, mut plan) = hybrid_fixture();
        plan.budget.reservation_breakdown.kv_state = Bytes(1);
        plan.plan_id = plan.compute_id().unwrap();
        let mut engine = TuckCore::new();
        let err = engine
            .load(
                &plan,
                &artifact,
                Some(&serde_json::json!({"kernels": "metal"})),
            )
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidPlan, "{err}");
        assert!(err.to_string().contains("kv_state"), "{err}");
        assert!(!engine.is_loaded());
    }

    #[test]
    fn hybrid_metal_session_matches_the_cpu_oracle_and_resets_per_request() {
        if !metal_or_skip() {
            return;
        }
        let (_dir, artifact, plan) = hybrid_fixture();
        let mut engine = TuckCore::new();
        let note = engine
            .load(
                &plan,
                &artifact,
                Some(&serde_json::json!({"kernels": "metal"})),
            )
            .unwrap();
        assert!(
            note.contains("unified engine") && note.contains("graph qwen35"),
            "{note}"
        );
        assert_eq!(
            engine.identity().kernels,
            vec![HYBRID_KERNEL_ID.to_string()]
        );

        // The device ledger equals the preflight, and every class is
        // reported for the supervisor's plan and reuse checks.
        let stats = engine.stats();
        let state = stats.retained_state_bytes.expect("held session state");
        let weights = stats.retained_weight_bytes.unwrap();
        let scratch = stats.retained_workspace_bytes.unwrap();
        assert_eq!(stats.gpu_allocated_bytes, Some(weights + state + scratch));
        assert_eq!(stats.kv_state_bytes, Some(state));
        assert_eq!(stats.kernel_impl.as_deref(), Some("metal"));
        assert_eq!(stats.kv_precision.as_deref(), Some("f16"));

        // Teacher-forced logits vs the CPU oracle (registered 1e-4 on the
        // fixture, ADR-008 Addendum 10), positions out of order included.
        let tokens: Vec<u32> = vec![1, 9, 4, 17, 5, 23];
        let positions = [0u64, 3, 5, 2];
        let mut got = Vec::new();
        {
            let mut sink = |pos: u64, logits: &[f32]| {
                got.push((pos, logits.to_vec()));
                true
            };
            let scored = engine
                .score(&tokens, &positions, &|| false, &mut sink)
                .unwrap();
            assert_eq!(scored, positions.len() as u64);
        }
        let path = std::path::Path::new(&artifact.path);
        for (pos, logits) in &got {
            let mut oracle = crate::qwen35_forward::Qwen35Model::open(
                path,
                tokens.len(),
                crate::qwen35_kv::KvPrecision::F16,
            )
            .unwrap();
            let mut want = Vec::new();
            oracle
                .forward_tokens(&tokens[..=*pos as usize], &mut want)
                .unwrap();
            let worst = logits
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(worst <= 1e-4, "position {pos}: |diff| {worst}");
        }

        // Two identical requests produce identical output: recurrent state
        // never leaks from one request into the next.
        let (first, text_a) = run_generate(&mut engine, vec![1, 2, 3], 6);
        let (second, text_b) = run_generate(&mut engine, vec![1, 2, 3], 6);
        assert_eq!(first.generated, second.generated);
        assert_eq!(text_a, text_b);
        assert_eq!(engine.stats().session_kv_bytes, Some(state));
    }

    #[test]
    fn metal_forward_matches_scalar_on_fixture() {
        if !tuckmill_metal::probe().available {
            assert!(
                std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
                "TUCKMILL_REQUIRE_METAL is set but no Metal device is available"
            );
            eprintln!("skipping: no Metal device");
            return;
        }
        let (_dir, artifact, plan) = fixture();
        let mut cpu = TuckCore::new();
        cpu.load(
            &plan,
            &artifact,
            Some(&serde_json::json!({"kernels": "scalar"})),
        )
        .unwrap();
        let mut gpu = TuckCore::new();
        let note = gpu
            .load(
                &plan,
                &artifact,
                Some(&serde_json::json!({"kernels": "metal"})),
            )
            .unwrap();
        assert!(
            note.contains("unified engine")
                && note.contains("graph qwen3 ")
                && note.contains("kv f32"),
            "{note}"
        );

        // Teacher-forced logits on a fixed sequence (held positions).
        let tokens: Vec<u32> = vec![1, 9, 4, 17, 5, 23];
        let positions = [0u64, 2, 4];
        let collect = |engine: &mut TuckCore| -> Vec<(u64, Vec<f32>)> {
            let mut out = Vec::new();
            {
                let mut sink = |pos: u64, logits: &[f32]| {
                    out.push((pos, logits.to_vec()));
                    true
                };
                engine
                    .score(&tokens, &positions, &|| false, &mut sink)
                    .unwrap();
            }
            out
        };
        let cpu_logits = collect(&mut cpu);
        let gpu_logits = collect(&mut gpu);
        assert_eq!(cpu_logits.len(), 3);
        assert_eq!(gpu_logits.len(), 3);
        for ((pos, a), (_, b)) in cpu_logits.iter().zip(gpu_logits.iter()) {
            assert_eq!(a.len(), b.len());
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
            let max_diff = a
                .iter()
                .zip(b.iter())
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            // Registered fixture policy: same f32 arithmetic, different
            // summation order; ~1e-4 relative is expected, 1e-2 is margin.
            assert!(
                max_diff <= 1e-2 * scale,
                "position {pos}: max |diff| {max_diff} over scale {scale}"
            );
            assert_eq!(
                sampling::argmax(a),
                sampling::argmax(b),
                "position {pos}: top-1 must agree"
            );
        }

        // Greedy generation path: same tokens and finish through both.
        let (cpu_outcome, cpu_text) = run_generate_with(
            &mut cpu,
            tokens.clone(),
            6,
            vec![],
            0,
            SamplingPolicy::default(),
        );
        let (gpu_outcome, gpu_text) =
            run_generate_with(&mut gpu, tokens, 6, vec![], 0, SamplingPolicy::default());
        assert_eq!(cpu_outcome.finish_reason, gpu_outcome.finish_reason);
        assert_eq!(
            cpu_text, gpu_text,
            "greedy visible text must match the scalar reference"
        );

        // Honest stats: the GPU path reports itself and its live bytes.
        let st = gpu.stats();
        assert_eq!(st.kernel_impl.as_deref(), Some("metal"));
        assert!(st.gpu_allocated_bytes.unwrap_or(0) > 0);
        assert!(
            st.storage.is_none(),
            "GPU resident path has no CPU storage mode"
        );
        assert!(st.retained_weight_bytes.unwrap_or(0) > 0);
    }

    /// ADR-011 step 2: dense Qwen3 on the engine with declared f16 KV
    /// matches the CPU oracle with the same declared precision (packed f16
    /// holds exactly the values the CPU stores as f16-rounded f32).
    #[test]
    fn metal_engine_f16_kv_matches_the_cpu_oracle_on_fixture() {
        if !metal_or_skip() {
            return;
        }
        let (_dir, artifact, plan) = fixture();
        let mut cpu = TuckCore::new();
        cpu.load(
            &plan,
            &artifact,
            Some(&serde_json::json!({"kernels": "scalar", "kv_precision": "f16"})),
        )
        .unwrap();
        let mut gpu = TuckCore::new();
        let note = gpu
            .load(
                &plan,
                &artifact,
                Some(&serde_json::json!({"kernels": "metal", "kv_precision": "f16"})),
            )
            .unwrap();
        assert!(note.contains("kv f16 packed"), "{note}");
        assert_eq!(gpu.stats().kv_precision.as_deref(), Some("f16"));
        let tokens: Vec<u32> = vec![1, 9, 4, 17, 5, 23];
        let positions = [1u64, 3, 5];
        let collect = |engine: &mut TuckCore| -> Vec<Vec<f32>> {
            let mut out = Vec::new();
            {
                let mut sink = |_: u64, logits: &[f32]| {
                    out.push(logits.to_vec());
                    true
                };
                engine
                    .score(&tokens, &positions, &|| false, &mut sink)
                    .unwrap();
            }
            out
        };
        for (a, b) in collect(&mut cpu).iter().zip(collect(&mut gpu).iter()) {
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
            let max_diff = a
                .iter()
                .zip(b.iter())
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            assert!(
                max_diff <= 1e-2 * scale,
                "max |diff| {max_diff} over {scale}"
            );
            assert_eq!(sampling::argmax(a), sampling::argmax(b), "top-1");
        }
        let (_, cpu_text) = run_generate_with(
            &mut cpu,
            tokens.clone(),
            6,
            vec![],
            0,
            SamplingPolicy::default(),
        );
        let (_, gpu_text) =
            run_generate_with(&mut gpu, tokens, 6, vec![], 0, SamplingPolicy::default());
        assert_eq!(cpu_text, gpu_text, "greedy text with f16 KV");
    }

    #[test]
    fn metal_engine_refuses_options_it_cannot_honor_for_qwen3() {
        let (_dir, artifact, plan) = fixture();
        for opts in [
            serde_json::json!({"kernels": "metal", "prefill_batch": 4}),
            serde_json::json!({"kernels": "metal", "weight_cache_bytes": 65536}),
        ] {
            let mut engine = TuckCore::new();
            let err = engine.load(&plan, &artifact, Some(&opts)).unwrap_err();
            assert_eq!(err.code(), ErrorCode::InvalidRequest, "{opts}: {err}");
            assert!(!engine.is_loaded(), "{opts}: nothing is loaded on refusal");
        }
    }

    /// Gate C2 milestone (release-gated, real weights): the Metal resident
    /// path must reproduce the CPU scalar oracle on the pinned Qwen3-0.6B
    /// Q8_0 artifact — teacher-forced logits at held positions and greedy
    /// generation — within a registered f32 reduction-order policy.
    /// Debug builds skip it (the CPU oracle is ~10x slower there); run with
    /// `cargo test --release -p tuckmill-tuckcore real_model_metal` and
    /// TUCKMILL_REAL_GGUF pointing at the pinned artifact.
    #[cfg(not(debug_assertions))]
    #[test]
    fn real_model_metal_matches_the_cpu_oracle() {
        let Ok(gguf) = std::env::var("TUCKMILL_REAL_GGUF") else {
            eprintln!("skipping: TUCKMILL_REAL_GGUF not set (real-weight label)");
            return;
        };
        if !tuckmill_metal::probe().available {
            assert!(
                std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
                "TUCKMILL_REQUIRE_METAL is set but no Metal device is available"
            );
            eprintln!("skipping: no Metal device");
            return;
        }
        let path = std::path::PathBuf::from(&gguf);
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let registry = AdapterRegistry::builtin().unwrap();
        let (cfg, _) = crate::model::resolve(&insp, &registry).unwrap();
        // Fixed in-vocabulary token sequence: the comparison is about
        // arithmetic, not tokenization (which reference-first encoding
        // covers separately in the supervisor).
        let tokens: Vec<u32> = vec![9707, 11, 1879, 0, 374, 264, 1879];
        for t in &tokens {
            assert!((*t as u64) < cfg.vocab, "token {t} in vocabulary");
        }

        let artifact = {
            let meta = insp.metadata_hash().unwrap();
            let index = insp.tensor_index_hash().unwrap();
            tuckmill_backend::ArtifactRef {
                path: path.display().to_string(),
                file_bytes: insp.file_len,
                metadata_hash: meta.as_str().to_string(),
                tensor_index_hash: index.as_str().to_string(),
                tokenizer_hash: None,
                arch_id: Some("qwen3".into()),
                content_hash: None,
            }
        };
        let payload: u64 = insp.tensors.iter().map(|t| t.byte_size).sum();
        let context = 2048u64;
        let plan = {
            let mut p = tuckmill_manifest::plan::ExecutionPlan::new_skeleton(
                artifact_artifact_id(),
                tuckmill_manifest::ids::Id::new("d".repeat(64)).unwrap(),
                "qwen3".into(),
            );
            p.mode = tuckmill_manifest::plan::ExecutionMode::Resident;
            p.context = tuckmill_manifest::plan::PlanContext::new(context, 64);
            p.budget.requested_budget = tuckmill_manifest::units::Bytes::parse("2GiB").unwrap();
            p.budget.add_reservation(
                tuckmill_manifest::records::ReserveClass::Weights,
                tuckmill_manifest::units::Bytes(payload),
            );
            p.budget.add_reservation(
                tuckmill_manifest::records::ReserveClass::KvState,
                tuckmill_manifest::units::Bytes(KvState::bytes_for(&cfg, context)),
            );
            p.budget.add_reservation(
                tuckmill_manifest::records::ReserveClass::Workspaces,
                tuckmill_manifest::units::Bytes(64 * 1024 * 1024),
            );
            p.budget.add_reservation(
                tuckmill_manifest::records::ReserveClass::IoBuffers,
                tuckmill_manifest::units::Bytes(16 * 1024 * 1024),
            );
            p.budget.reservation = p.budget.reservation_breakdown.total();
            p.plan_id = p.compute_id().unwrap();
            p
        };

        let mut cpu = TuckCore::new();
        cpu.load(
            &plan,
            &artifact,
            Some(&serde_json::json!({"kernels": "scalar", "verify_content_hash": false})),
        )
        .unwrap();
        let mut gpu = TuckCore::new();
        gpu.load(
            &plan,
            &artifact,
            Some(&serde_json::json!({"kernels": "metal", "verify_content_hash": false})),
        )
        .unwrap();

        // Teacher-forced logits at held positions.
        let positions: Vec<u64> = vec![(tokens.len() - 1) as u64];
        let collect = |engine: &mut TuckCore| -> Vec<(u64, Vec<f32>)> {
            let mut out = Vec::new();
            {
                let mut sink = |pos: u64, logits: &[f32]| {
                    out.push((pos, logits.to_vec()));
                    true
                };
                engine
                    .score(&tokens, &positions, &|| false, &mut sink)
                    .unwrap();
            }
            out
        };
        let cpu_logits = collect(&mut cpu);
        let gpu_logits = collect(&mut gpu);
        for ((pos, a), (_, b)) in cpu_logits.iter().zip(gpu_logits.iter()) {
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
            let max_diff = a
                .iter()
                .zip(b.iter())
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            let top1_a = sampling::argmax(a).unwrap();
            let top1_b = sampling::argmax(b).unwrap();
            println!(
                "position {pos}: max |diff| {max_diff:.4e} over scale {scale:.4e}; top1 {top1_a} vs {top1_b}"
            );
            assert_eq!(top1_a, top1_b, "position {pos}: top-1 must agree");
            assert!(
                max_diff <= 5e-2 * scale,
                "position {pos}: max |diff| {max_diff} over scale {scale}"
            );
        }

        // Greedy generation through both paths.
        let (cpu_outcome, cpu_text) = run_generate_with(
            &mut cpu,
            tokens.clone(),
            6,
            vec![],
            0,
            SamplingPolicy::default(),
        );
        let (gpu_outcome, gpu_text) =
            run_generate_with(&mut gpu, tokens, 6, vec![], 0, SamplingPolicy::default());
        println!("cpu text: {cpu_text:?}");
        println!("gpu text: {gpu_text:?}");
        assert_eq!(cpu_outcome.finish_reason, gpu_outcome.finish_reason);
        assert_eq!(cpu_text, gpu_text, "greedy output must match the oracle");
        let st = gpu.stats();
        println!(
            "gpu stats: kernel_impl {:?}, gpu_allocated_bytes {:?}, retained {:?}",
            st.kernel_impl, st.gpu_allocated_bytes, st.retained_weight_bytes
        );
        assert_eq!(st.kernel_impl.as_deref(), Some("metal"));
    }

    #[cfg(not(debug_assertions))]
    fn artifact_artifact_id() -> tuckmill_manifest::ids::Id {
        tuckmill_manifest::ids::Id::new("a".repeat(64)).unwrap()
    }

    #[test]
    fn warm_style_smaller_request_allocates_its_own_session_kv() {
        // Gate B review finding 2, at the byte level: with a large capacity
        // already prepared, a smaller request's KV session is allocated for
        // its own prompt+output need — never for the prepared capacity.
        let (_dir, artifact, plan) = fixture();
        let mut engine = TuckCore::new();
        engine.load(&plan, &artifact, None).unwrap();
        // Snapshot the geometry so the closure does not hold the engine.
        let cfg = engine.loaded.as_ref().unwrap().weights.cfg.clone();
        let cfg_bytes = move |cap: u64| KvState::bytes_for(&cfg, cap);

        let prepared = engine.stats().kv_state_bytes.unwrap();
        assert_eq!(prepared, cfg_bytes(256), "prepared capacity is the plan's");

        // Prompt 8 + max 16 = 24 admitted tokens.
        let (_o, _t) = run_generate_with(
            &mut engine,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            16,
            vec![],
            0,
            SamplingPolicy::default(),
        );
        let session = engine.stats().session_kv_bytes.expect("session bytes");
        assert_eq!(
            session,
            cfg_bytes(24),
            "the session allocates the request's own admitted need"
        );
        assert!(
            session < prepared,
            "a reused worker must not allocate the prepared capacity for a smaller request"
        );
    }

    #[test]
    fn stop_markers_never_reach_the_visible_stream() {
        let (_dir, artifact, plan) = fixture();
        let mut engine = TuckCore::new();
        engine.load(&plan, &artifact, None).unwrap();

        // Learn the deterministic greedy continuation, then use its own
        // suffix as the stop sequence: the stopped run is the same greedy
        // stream, so the stop must fire and must never be visible.
        let (_o, texts) = run_generate(&mut engine, vec![1, 2, 3], 12);
        let plain: String = texts.concat();
        assert!(
            plain.chars().count() >= 4,
            "fixture text too short: {plain:?}"
        );
        let stop: String = plain
            .chars()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let (outcome, stopped) = run_generate_with(
            &mut engine,
            vec![1, 2, 3],
            12,
            vec![stop.clone()],
            7,
            SamplingPolicy::default(),
        );
        let visible: String = stopped.concat();
        assert_eq!(outcome.finish_reason, FinishReason::Stop);
        assert!(
            !visible.ends_with(stop.as_str()),
            "stop marker leaked: {visible:?}"
        );
        assert!(
            plain.starts_with(&visible),
            "visible output must be an unmodified prefix"
        );
        assert!(visible.len() < plain.len());
        assert!(outcome.generated >= outcome.committed);
        assert_eq!(outcome.generated, 12.min(outcome.generated)); // bounded by max
    }

    #[test]
    fn sampling_policy_from_the_request_is_honored() {
        let (_dir, artifact, plan) = fixture();
        let mut engine = TuckCore::new();
        engine.load(&plan, &artifact, None).unwrap();

        // Greedy stays deterministic regardless of seed.
        let (g1, t1) = run_generate_with(
            &mut engine,
            vec![9, 8, 7],
            8,
            vec![],
            1,
            SamplingPolicy::default(),
        );
        let (_g2, t2) = run_generate_with(
            &mut engine,
            vec![9, 8, 7],
            8,
            vec![],
            2,
            SamplingPolicy::default(),
        );
        assert_eq!(g1.finish_reason, FinishReason::Length);
        assert_eq!(t1, t2, "greedy ignores the seed");

        // Nonzero temperature with a seed is reproducible...
        let warm = SamplingPolicy {
            temperature: 1.2,
            top_p: Some(0.9),
        };
        let (a, ta) = run_generate_with(&mut engine, vec![9, 8, 7], 12, vec![], 11, warm);
        let (_b, tb) = run_generate_with(&mut engine, vec![9, 8, 7], 12, vec![], 11, warm);
        assert_eq!(ta, tb, "seeded sampling is reproducible");
        assert_eq!(a.finish_reason, FinishReason::Length);
        // ...and a different seed diverges.
        let (_c, tc) = run_generate_with(&mut engine, vec![9, 8, 7], 12, vec![], 12, warm);
        assert_ne!(ta, tc, "different seeds must produce different samples");

        // Unsupported values are refused before compute, never ignored.
        let bad = SamplingPolicy {
            temperature: 3.0,
            top_p: None,
        };
        let err = {
            let cancel = Arc::new(AtomicBool::new(false));
            let paused = Arc::new(AtomicBool::new(false));
            let mut on_token = |_i: u64, _t: &str| true;
            let mut on_state = |_s: RunState| {};
            let mut ctl =
                GenControl::new(cancel.clone(), paused.clone(), &mut on_token, &mut on_state);
            let req = GenerateRequest {
                request_id: "bad-sampling".into(),
                prompt_tokens: vec![1, 2],
                max_output_tokens: 4,
                stop: vec![],
                seed: 0,
                sampling: bad,
                ignore_eos: false,
            };
            engine.generate(&req, &mut ctl).unwrap_err()
        };
        assert_eq!(err.code(), ErrorCode::InvalidRequest);
    }

    #[test]
    fn generation_cancels_at_a_boundary() {
        let (_dir, artifact, plan) = fixture();
        let mut engine = TuckCore::new();
        engine.load(&plan, &artifact, None).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let cancel2 = cancel.clone();
        let mut count = 0u64;
        let mut on_token = |_i: u64, _t: &str| {
            count += 1;
            if count >= 3 {
                cancel2.store(true, Ordering::Relaxed);
            }
            true
        };
        let mut on_state = |_s: RunState| {};
        let mut ctl = GenControl::new(cancel.clone(), paused.clone(), &mut on_token, &mut on_state);
        let req = GenerateRequest {
            request_id: "t3".into(),
            prompt_tokens: vec![4, 5],
            max_output_tokens: 200,
            stop: vec![],
            seed: 0,
            sampling: tuckmill_backend::SamplingPolicy::default(),
            ignore_eos: false,
        };
        let outcome = engine.generate(&req, &mut ctl).unwrap();
        assert_eq!(outcome.finish_reason, FinishReason::Cancelled);
        assert!(outcome.committed >= 3);
        assert!(outcome.committed < 200);
    }

    #[test]
    fn scoring_returns_vocab_logits_per_position() {
        let (_dir, artifact, plan) = fixture();
        let mut engine = TuckCore::new();
        engine.load(&plan, &artifact, None).unwrap();
        let tokens = [2u32, 4, 6, 8];
        let mut seen = Vec::new();
        let scored = engine
            .score(&tokens, &[1, 3], &|| false, &mut |pos, logits| {
                seen.push((pos, logits.len(), logits[0]));
                true
            })
            .unwrap();
        assert_eq!(scored, 2);
        assert_eq!(seen.len(), 2);
        assert!(seen.iter().all(|(_, len, _)| *len == 512));
        // Logits at the same position are deterministic across runs.
        let mut a = Vec::new();
        engine
            .score(&tokens, &[3], &|| false, &mut |_, l| {
                a.extend_from_slice(l);
                true
            })
            .unwrap();
        let mut b = Vec::new();
        engine
            .score(&tokens, &[3], &|| false, &mut |_, l| {
                b.extend_from_slice(l);
                true
            })
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn streaming_config_changes_evidence_not_numbers() {
        let (_dir, artifact, plan) = fixture();
        let tokens = [3u32, 1, 4, 1, 5];
        let mut resident = TuckCore::new();
        resident
            .load(
                &plan,
                &artifact,
                Some(&serde_json::json!({"weight_cache_bytes": 33554432, "block_bytes": 1048576})),
            )
            .unwrap();
        let mut streamed = TuckCore::new();
        streamed
            .load(
                &plan,
                &artifact,
                Some(&serde_json::json!({"weight_cache_bytes": 65536, "block_bytes": 65536})),
            )
            .unwrap();
        let mut ra = Vec::new();
        resident
            .score(&tokens, &[4], &|| false, &mut |_, l| {
                ra.extend_from_slice(l);
                true
            })
            .unwrap();
        let mut sb = Vec::new();
        streamed
            .score(&tokens, &[4], &|| false, &mut |_, l| {
                sb.extend_from_slice(l);
                true
            })
            .unwrap();
        assert_eq!(ra, sb, "streaming must not change the numbers");
        let stats = streamed.stats();
        assert!(stats.evictions > 0, "forced streaming exercised: {stats:?}");
        assert_eq!(stats.in_flight_leases, 0, "leases drain after scoring");
    }

    #[test]
    fn governor_trim_epoch_evicts_caches_during_generation() {
        let (_dir, artifact, plan) = fixture();
        let mut engine = TuckCore::new();
        engine
            .load(
                &plan,
                &artifact,
                Some(&serde_json::json!({"weight_cache_bytes": 65536, "block_bytes": 65536})),
            )
            .unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let trim = engine.trim_epoch();
        // Bump the epoch from "the governor" mid-run via the token callback.
        let trim2 = trim.clone();
        let mut bumped = false;
        let mut on_token = |_i: u64, _t: &str| {
            if !bumped {
                bumped = true;
                trim2.fetch_add(1, Ordering::Relaxed);
            }
            true
        };
        let mut on_state = |_s: RunState| {};
        let mut ctl = GenControl::new(cancel, paused, &mut on_token, &mut on_state);
        ctl.trim_epoch = trim.clone();
        let req = GenerateRequest {
            request_id: "trim".into(),
            prompt_tokens: vec![1, 2],
            max_output_tokens: 8,
            stop: vec![],
            seed: 0,
            sampling: tuckmill_backend::SamplingPolicy::default(),
            ignore_eos: false,
        };
        let before = engine.stats().evictions;
        let outcome = engine.generate(&req, &mut ctl).unwrap();
        assert_eq!(outcome.committed, 8);
        let after = engine.stats().evictions;
        assert!(
            after > before,
            "trim request must evict expendable caches at a safe point (before {before}, after {after})"
        );
        assert_eq!(
            engine.stats().in_flight_leases,
            0,
            "trim never touches in-flight buffers"
        );
    }

    #[test]
    fn shutdown_releases_everything() {
        let (_dir, artifact, plan) = fixture();
        let mut engine = TuckCore::new();
        engine.load(&plan, &artifact, None).unwrap();
        let _ = run_generate(&mut engine, vec![1], 2);
        engine.shutdown();
        assert!(!engine.is_loaded());
        assert_eq!(engine.stats(), BackendStats::default());
    }
}
