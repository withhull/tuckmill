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
use crate::forward::{
    self, Forward, KvPrecision, KvState, ModelWeights, NumericalProfile, ProfileCounters,
};
use crate::kernels::{KernelSelection, ResolvedKernels};
use crate::loader::{BlockLoader, BlockPolicy, StorageSelection};
use crate::model;
use crate::sampling::{self, Rng, SamplingConfig};
use crate::tensor;
use crate::tokenizer::{Tokenizer, Utf8StreamDecoder};

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
}

struct Loaded {
    loader: Arc<BlockLoader>,
    tracker: Arc<CompletionTracker>,
    weights: Arc<ModelWeights>,
    tokenizer: Tokenizer,
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

/// The experimental custom native executor.
pub struct TuckCore {
    loaded: Option<Loaded>,
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
            per_token_delay: Duration::ZERO,
            trim_epoch: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            applied_trim_epoch: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.is_some()
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
            kernels: kernel_ids(self.loaded.as_ref().map(|l| l.kernels)),
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
            architectures: vec!["qwen3".to_string()],
            moe: false,
            hybrid: false,
            ternary: false,
            requires_gpu: false,
            notes: vec![
                "Stage 0: dense Qwen3 GGUF, scalar f32 reference kernels, bounded weight streaming (ADR-005)"
                    .to_string(),
                "MoE, hybrid, and ternary profiles remain separately gated; routing is never substituted"
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
        let storage_selection = opts.storage.unwrap_or(StorageSelection::Auto);
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
        // Full-payload identity: one bounded streaming pass over the open
        // file, compared against the store's recorded canonical digest.
        // Drifted bytes must never execute; disabling is explicit and
        // recorded.
        let verify_start = Instant::now();
        let content_verified = if verify_content_hash {
            match &artifact.content_hash {
                Some(expected) => {
                    let mut reader = std::io::BufReader::with_capacity(1 << 20, &artifact_file);
                    let (actual, _) = tuckmill_manifest::digest::digest_reader(
                        &mut reader,
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
        let registry = AdapterRegistry::builtin()?;
        let (cfg, _adapter) = model::resolve(&insp, &registry)?;
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
        let payload: u64 = insp.tensors.iter().map(|t| t.byte_size).sum();
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
        let tokenizer = Tokenizer::from_gguf(&insp)?;

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
        Ok(note)
    }

    fn generate(&mut self, req: &GenerateRequest, ctl: &mut GenControl<'_>) -> Result<GenOutcome> {
        let loaded = self.loaded.as_ref().ok_or_else(|| {
            TuckmillError::new(ErrorCode::WorkerFailed, "tuckcore has no loaded artifact")
        })?;
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
        if prompt_len + req.max_output_tokens > loaded.context_total {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!(
                    "prompt {prompt_len} + max output {} exceeds admitted context {}",
                    req.max_output_tokens, loaded.context_total
                ),
            ));
        }

        let cancelled = {
            let cancel = ctl.cancel.clone();
            move || cancel.load(Ordering::Relaxed)
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
        let mut kv = KvState::new(&loaded.weights.cfg, session_context, loaded.kv_precision)?;
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
        let mut logits: Vec<f32> = Vec::new();
        for (pos, token) in req.prompt_tokens.iter().enumerate() {
            pause_loop(ctl, Duration::ZERO)?;
            match fwd.forward_token(&mut kv, *token, pos, &cancelled) {
                Ok(l) => logits = l,
                Err(e) if e.code() == ErrorCode::Cancelled => {
                    return Ok(
                        GenOutcome::new(FinishReason::Cancelled, 0).with_timing(empty_timing(t0))
                    )
                }
                Err(e) => return Err(e),
            }
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
                    break 'decode GenOutcome::new(
                        FinishReason::Cancelled,
                        buffer.visible_tokens(),
                    )
                    .with_generated(generated_ids);
                }
                pause_loop(ctl, self.per_token_delay)?;

                let Some(token) = sampling::sample(&logits, &sampling_cfg, &mut rng) else {
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
                if Some(token) == loaded.weights.cfg.eos_token_id {
                    // The model-defined EOS token is never visible: its
                    // bytes are not pushed, and anything held is flushed
                    // because no stop sequence can complete after EOS.
                    let r = buffer.flush();
                    let _ = emit(&r.text, &mut release_index, ctl);
                    break 'decode GenOutcome::new(FinishReason::Eos, buffer.visible_tokens())
                        .with_generated(generated_ids);
                }
                let piece = match loaded.tokenizer.token_text(token) {
                    Some(_) => decoder.push_token(&loaded.tokenizer.token_bytes(token), i),
                    None => {
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
                    match fwd.forward_token(&mut kv, token, pos, &cancelled) {
                        Ok(l) => logits = l,
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
            GenOutcome::new(FinishReason::Length, buffer.visible_tokens())
                .with_generated(generated_ids)
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

    fn score(
        &mut self,
        tokens: &[u32],
        positions: &[u64],
        cancelled: &dyn Fn() -> bool,
        sink: &mut dyn FnMut(u64, &[f32]) -> bool,
    ) -> Result<u64> {
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
        match &self.loaded {
            Some(loaded) => {
                let s = loaded.loader.stats();
                let residency = loaded.loader.residency();
                BackendStats {
                    logical_read_bytes: s.logical_read_bytes,
                    blocks_loaded: s.blocks_loaded,
                    cache_hits: s.cache_hits,
                    cache_misses: s.cache_misses,
                    evictions: s.cache_entry_evictions,
                    in_flight_leases: loaded.tracker.in_flight() as u64,
                    buffer_releases: Some(loaded.tracker.completed()),
                    released_bytes: Some(loaded.tracker.released_bytes()),
                    storage: Some(StorageStats {
                        mode: s.storage_mode.clone(),
                        owned_cache_bytes: s.owned_cache_bytes,
                        owned_in_flight_bytes: loaded.tracker.in_flight_owned_bytes(),
                        mapped_vm_bytes: s.mapped_vm_bytes,
                        mapped_resident_bytes: residency.as_ref().map(|r| r.resident_bytes),
                        mapped_resident_sampled: residency
                            .as_ref()
                            .map(|r| r.sampled)
                            .unwrap_or(false),
                        page_size_bytes: residency.as_ref().map(|r| r.page_size_bytes).unwrap_or(0),
                        page_advice_bytes: s.page_advice_bytes,
                    }),
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
                    trims_applied: Some(loaded.counters.trims_applied.load(Ordering::Relaxed)),
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
        if let Some(loaded) = self.loaded.take() {
            loaded.loader.evict_all();
            // Leases must have drained; anything left would be a bug.
            debug_assert_eq!(loaded.tracker.in_flight(), 0);
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
