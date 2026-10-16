//! Command implementations.

use std::io::{Read, Write};
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::CacheCmd;
use crate::Cmd;
use crate::INTERRUPTED;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::gguf::{self, GgufValue};
use tuckmill_manifest::records::CapabilityStatusKind;
use tuckmill_manifest::units::Bytes;
use tuckmill_planner::device;
use tuckmill_store::aliases;
use tuckmill_store::cache;
use tuckmill_store::import::{import_local, ImportOptions};
use tuckmill_store::layout::StoreLayout;
use tuckmill_store::pack::{pack_artifact, PackOptions};
use tuckmill_supervisor::bench::{format_summary, run_and_record, BenchOptions};
use tuckmill_supervisor::runtime::{EngineMode, RunEvent, Runtime, RuntimeOptions};

pub const DEFAULT_PRUNE_AGE_SECS: u64 = 3600;

pub fn dispatch(cmd: &Cmd, json: bool, data_dir: Option<&Path>) -> Result<ExitCode> {
    match cmd {
        Cmd::Doctor { probe_storage } => doctor(*probe_storage, json, data_dir),
        Cmd::Inspect { path } => inspect(path, json),
        Cmd::Import {
            path,
            alias,
            license,
            revision,
        } => import(
            path,
            alias.clone(),
            license.clone(),
            revision.clone(),
            json,
            data_dir,
        ),
        Cmd::Pack { model } => pack(model, json, data_dir),
        Cmd::Plan {
            model,
            memory_budget,
            context,
            max_output,
        } => plan(
            model,
            memory_budget.as_deref(),
            *context,
            *max_output,
            json,
            data_dir,
        ),
        Cmd::Run {
            model,
            prompt,
            memory_budget,
            context,
            max_output,
            allow_test_engine,
            engine,
            quiet,
        } => run(
            model,
            prompt.clone(),
            memory_budget.as_deref(),
            *context,
            *max_output,
            *allow_test_engine,
            engine,
            *quiet,
            json,
            data_dir,
        ),
        Cmd::Serve {
            port,
            allow_test_engine,
            engine,
            probe_only,
        } => serve(
            *port,
            *allow_test_engine,
            engine,
            *probe_only,
            json,
            data_dir,
        ),
        Cmd::Verify {
            model,
            memory_budget,
            context,
            positions,
            generate,
            reference,
            tolerance,
        } => verify(
            model,
            memory_budget.as_deref(),
            *context,
            *positions,
            *generate,
            *reference,
            *tolerance,
            json,
            data_dir,
        ),
        Cmd::Bench {
            model,
            suite,
            input,
            output,
            repetitions,
            memory_budget,
            context,
            allow_test_engine,
            engine,
        } => bench(
            model,
            suite,
            *input,
            *output,
            *repetitions,
            memory_budget.as_deref(),
            *context,
            *allow_test_engine,
            engine,
            json,
            data_dir,
        ),
        Cmd::Cache { cmd } => match cmd {
            CacheCmd::List => cache_list(json, data_dir),
            CacheCmd::Prune { dry_run } => cache_prune(*dry_run, json, data_dir),
        },
        Cmd::WorkerReference { socket, engine } => {
            let kind = match tuckmill_backend::BackendId::parse(engine) {
                Some(tuckmill_backend::BackendId::ReferenceTest) => {
                    tuckmill_backend::EngineKind::ReferenceTest
                }
                Some(tuckmill_backend::BackendId::TuckCore) => {
                    tuckmill_backend::EngineKind::TuckCore
                }
                Some(tuckmill_backend::BackendId::LlamaCpp) => {
                    tuckmill_backend::EngineKind::LlamaCpp
                }
                None => {
                    return Err(TuckmillError::new(
                        ErrorCode::InvalidRequest,
                        format!("unknown worker engine {engine:?}"),
                    ))
                }
            };
            tuckmill_supervisor::worker_host::run_worker(socket, kind)?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Resolve the requested budget: explicit argument, else the
/// hardware-adaptive default (ADR-006).
fn resolve_budget(runtime: &Runtime, requested: Option<&str>) -> Result<Bytes> {
    match requested {
        Some(text) => Bytes::parse(text),
        None => Ok(runtime.default_budget),
    }
}

fn open_layout(data_dir: Option<&Path>) -> Result<StoreLayout> {
    let layout = match data_dir {
        Some(dir) => StoreLayout::open(dir),
        None => StoreLayout::resolve()?,
    };
    layout.ensure()?;
    Ok(layout)
}

/// Parse the `--engine` selection into (backend kind, placement).
/// - "isolated" / "reference-test": reference test engine in a worker process
/// - "in-process": reference test engine in the CLI process (debugging)
/// - "tuckcore": experimental custom executor (isolated worker)
/// - "llama" / "llama.cpp": pinned llama.cpp reference (isolated worker)
fn parse_engine(engine: &str) -> Result<(tuckmill_backend::EngineKind, EngineMode)> {
    use tuckmill_backend::EngineKind;
    match engine {
        "isolated" | "reference-test" => Ok((EngineKind::ReferenceTest, EngineMode::Isolated)),
        "in-process" => Ok((EngineKind::ReferenceTest, EngineMode::InProcess)),
        "tuckcore" => Ok((EngineKind::TuckCore, EngineMode::Isolated)),
        "llama" | "llama.cpp" => Ok((EngineKind::LlamaCpp, EngineMode::Isolated)),
        other => Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            format!(
                "unknown engine {other:?}; use 'reference-test' (alias 'isolated'), 'in-process', 'tuckcore', or 'llama'"
            ),
        )),
    }
}

fn open_runtime(
    data_dir: Option<&Path>,
    engine: &str,
    allow_test_engine: bool,
) -> Result<Arc<Runtime>> {
    let (engine_kind, engine_mode) = parse_engine(engine)?;
    Runtime::open(RuntimeOptions {
        data_dir: data_dir.map(|d| d.to_path_buf()),
        engine_mode,
        engine_kind,
        worker_exe: std::env::current_exe().ok(),
        queue_capacity: 4,
        allow_test_engine,
        ..Default::default()
    })
}

// ---------------- doctor ----------------

fn doctor(probe_storage: bool, json: bool, data_dir: Option<&Path>) -> Result<ExitCode> {
    let layout = open_layout(data_dir)?;
    let profile = device::detect(layout.root())?;
    let kernels = tuckmill_planner::kernel_registry::KernelRegistry::builtin()?;
    let adapters = tuckmill_manifest::adapter::AdapterRegistry::builtin()?;
    let curl = tuckmill_store::download::CurlFetcher::discover()
        .map(|f| f.path().display().to_string())
        .unwrap_or_else(|_| "unavailable".to_string());

    let mut storage_probe = None;
    if probe_storage {
        storage_probe = Some(run_storage_probe(&layout)?);
    }

    if json {
        let mut value = serde_json::to_value(&profile)
            .map_err(|e| tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "device"))?;
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "kernels".into(),
                serde_json::json!(kernels
                    .kernels
                    .iter()
                    .map(|k| serde_json::json!({"id": k.id, "status": k.status}))
                    .collect::<Vec<_>>()),
            );
            obj.insert(
                "adapters".into(),
                serde_json::json!(adapters
                    .all()
                    .iter()
                    .map(|a| serde_json::json!({
                        "id": a.id,
                        "architecture": a.architecture,
                        "status": a.status,
                        "certification": a.certification.state,
                    }))
                    .collect::<Vec<_>>()),
            );
            obj.insert("curl".into(), serde_json::json!(curl));
            obj.insert(
                "data_dir".into(),
                serde_json::json!(layout.root().display().to_string()),
            );
            obj.insert(
                "storage_probe".into(),
                serde_json::to_value(&storage_probe).unwrap_or(serde_json::Value::Null),
            );
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&value).map_err(|e| {
                tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "json")
            })?
        );
        return Ok(ExitCode::SUCCESS);
    }

    println!(
        "Tuckmill doctor — {} ({} cores, {} RAM), {} {} ({})",
        profile.chip,
        profile.cpu_cores_total,
        Bytes(profile.memory_bytes),
        profile.os_name,
        profile.os_version,
        profile.os_build
    );
    println!("data directory: {}", layout.root().display());
    println!(
        "storage: {} — {} free of {} ({})",
        profile.storage.path_class,
        Bytes(profile.storage.free_bytes),
        Bytes(profile.storage.total_bytes),
        profile.capabilities["storage"].detail
    );
    println!("pressure: {}", profile.pressure);
    println!("thermal: {}", profile.thermal_start);
    println!(
        "gpu family (estimate): {} — {}",
        profile.gpu.family_estimate, profile.gpu.family_source
    );
    println!("curl: {curl}");
    println!("adapters:");
    for a in adapters.all() {
        println!(
            "  {} ({}): {} — certification {}",
            a.id, a.architecture, a.status, a.certification.state
        );
    }
    let unavailable: Vec<String> = profile
        .capabilities
        .iter()
        .filter(|(_, c)| c.status == CapabilityStatusKind::Unavailable)
        .map(|(k, c)| format!("{k} ({})", c.detail))
        .collect();
    if !unavailable.is_empty() {
        println!("unavailable signals (recorded honestly, never guessed):");
        for u in unavailable {
            println!("  {u}");
        }
    }
    if let Some(p) = &storage_probe {
        println!(
            "storage probe (bounded, opt-in): write {:.0} MB/s, read {:.0} MB/s (logical file I/O, not physical SSD counters)",
            p.write_mb_s, p.read_mb_s
        );
    }
    println!(
        "kernel work packages: {} specified, {} implemented",
        kernels.kernels.len(),
        kernels.kernels.iter().filter(|k| k.is_available()).count()
    );
    println!(
        "status: no certified execution engine in this build; the reference test engine (synthetic output) is available with --allow-test-engine."
    );
    Ok(ExitCode::SUCCESS)
}

#[derive(serde::Serialize)]
struct StorageProbe {
    bytes: u64,
    write_mb_s: f64,
    read_mb_s: f64,
}

fn run_storage_probe(layout: &StoreLayout) -> Result<StorageProbe> {
    const PROBE_BYTES: u64 = 64 * 1024 * 1024;
    layout.ensure_space_for(Bytes(PROBE_BYTES), &layout.tmp_dir())?;
    let path = layout.tmp_dir().join("storage-probe.bin");
    // Bounded and explicit: a temporary file inside the store's own tmp area.
    let payload = vec![0xA5u8; 1024 * 1024];
    let t0 = Instant::now();
    {
        let mut f = std::fs::File::create(&path).map_err(TuckmillError::from)?;
        let mut written = 0u64;
        while written < PROBE_BYTES {
            f.write_all(&payload).map_err(TuckmillError::from)?;
            written += payload.len() as u64;
        }
        f.sync_all().map_err(TuckmillError::from)?;
    }
    let write_s = t0.elapsed().as_secs_f64().max(1e-9);
    let t1 = Instant::now();
    {
        let mut f = std::fs::File::open(&path).map_err(TuckmillError::from)?;
        let mut buf = vec![0u8; payload.len()];
        let mut read = 0u64;
        while read < PROBE_BYTES {
            f.read_exact(&mut buf).map_err(TuckmillError::from)?;
            read += buf.len() as u64;
        }
    }
    let read_s = t1.elapsed().as_secs_f64().max(1e-9);
    let _ = std::fs::remove_file(&path);
    Ok(StorageProbe {
        bytes: PROBE_BYTES,
        write_mb_s: PROBE_BYTES as f64 / 1e6 / write_s,
        read_mb_s: PROBE_BYTES as f64 / 1e6 / read_s,
    })
}

// ---------------- inspect ----------------

fn kv_summary(v: &GgufValue) -> serde_json::Value {
    match v {
        GgufValue::Array(items) => serde_json::json!({"array_len": items.len()}),
        GgufValue::String(s) => {
            let mut t = s.clone();
            t.truncate(120);
            serde_json::json!(t)
        }
        other => serde_json::to_value(other).unwrap_or(serde_json::Value::Null),
    }
}

fn inspect(path: &Path, json: bool) -> Result<ExitCode> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        TuckmillError::new(
            ErrorCode::InvalidRequest,
            format!("cannot open {}: {e}", path.display()),
        )
        .with_source(e)
    })?;
    let mut magic = [0u8; 4];
    let read = file.read(&mut magic).map_err(TuckmillError::from)?;
    if read != 4 {
        return Err(TuckmillError::new(
            ErrorCode::InvalidArtifact,
            "file is smaller than any known model header",
        ));
    }

    if &magic == gguf::GGUF_MAGIC {
        let insp = gguf::inspect(path, &gguf::Limits::default())?;
        let stats = insp.tensor_stats();
        if json {
            let meta: Vec<serde_json::Value> = insp
                .metadata
                .iter()
                .map(|kv| {
                    serde_json::json!({
                        "key": kv.key,
                        "type": kv.value.type_name(),
                        "value": kv_summary(&kv.value),
                    })
                })
                .collect();
            let tensors: Vec<serde_json::Value> = insp
                .tensors
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "dims": t.dims,
                        "type": t.type_name,
                        "offset": t.offset,
                        "bytes": t.byte_size,
                    })
                })
                .collect();
            let out = serde_json::json!({
                "format": "gguf",
                "version": insp.version,
                "architecture": insp.arch_id(),
                "name": insp.model_name(),
                "context_length": insp.advertised_context_length(),
                "alignment": insp.alignment,
                "header_bytes": insp.header_bytes,
                "data_start": insp.data_start,
                "file_len": insp.file_len,
                "identity_hash": insp.identity_hash()?.as_str(),
                "tensor_index_hash": insp.tensor_index_hash()?.as_str(),
                "metadata_hash": insp.metadata_hash()?.as_str(),
                "tokenizer_hash": insp.tokenizer_hash()?.map(|i| i.as_str().to_string()),
                "tensor_stats": stats,
                "layout_warnings": insp.layout_warnings,
                "tensors": tensors,
                "metadata": meta,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&out).map_err(|e| {
                    tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "json")
                })?
            );
        } else {
            println!("format: GGUF v{}", insp.version);
            println!(
                "architecture: {}",
                insp.arch_id().unwrap_or_else(|| "unknown".into())
            );
            if let Some(name) = insp.model_name() {
                println!("name: {name}");
            }
            if let Some(ctx) = insp.advertised_context_length() {
                println!("advertised context: {ctx}");
            }
            println!(
                "tensors: {} ({} payload)",
                stats.total_tensors,
                Bytes(stats.total_bytes)
            );
            println!("tensor types:");
            for (t, bytes) in &stats.by_type {
                println!("  {t}: {}", Bytes(*bytes));
            }
            println!("tensor roles:");
            for (c, bytes) in &stats.by_class {
                println!("  {c}: {}", Bytes(*bytes));
            }
            println!("identity: {}", insp.identity_hash()?);
            for w in &insp.layout_warnings {
                println!("warning: {w}");
            }
        }
        Ok(ExitCode::SUCCESS)
    } else {
        let insp = tuckmill_manifest::safetensors::inspect(path)?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&insp).map_err(|e| {
                    tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "json")
                })?
            );
        } else {
            println!("format: safetensors");
            println!("tensors: {}", insp.tensors.len());
            println!("payload: {}", Bytes(insp.tensor_stats().total_bytes));
            if let Some(arch) = insp.arch_id() {
                println!("architecture (from metadata): {arch}");
            } else {
                println!(
                    "architecture: unknown (safetensors carries no standard architecture field; an adapter is required for planning)"
                );
            }
        }
        Ok(ExitCode::SUCCESS)
    }
}

// ---------------- import / pack ----------------

fn import(
    path: &Path,
    alias: Option<String>,
    license: Option<String>,
    revision: Option<String>,
    json: bool,
    data_dir: Option<&Path>,
) -> Result<ExitCode> {
    let layout = open_layout(data_dir)?;
    let registry = tuckmill_manifest::adapter::AdapterRegistry::builtin()?;
    let outcome = import_local(
        &layout,
        &registry,
        path,
        &ImportOptions {
            alias,
            license,
            revision,
            ..Default::default()
        },
    )?;
    if json {
        let out = serde_json::json!({
            "artifact": outcome.artifact,
            "installed_path": outcome.installed_path.display().to_string(),
            "already_present": outcome.already_present,
            "resumed": outcome.resumed,
            "warnings": outcome.warnings,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&out).map_err(
                |e| tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "json")
            )?
        );
    } else {
        println!(
            "{} {}",
            if outcome.already_present {
                "already imported:"
            } else {
                "imported:"
            },
            outcome.artifact.artifact_id
        );
        println!("  file: {}", outcome.installed_path.display());
        println!(
            "  architecture: {}",
            outcome
                .artifact
                .arch_id
                .clone()
                .unwrap_or_else(|| "unknown".into())
        );
        if let Some(g) = &outcome.artifact.geometry {
            println!(
                "  geometry: {} layers, {} kv heads, head dim {}, experts {:?}",
                g.layers, g.kv_head_count, g.key_length, g.expert_count
            );
        }
        if outcome.resumed {
            println!("  resumed from an interrupted import");
        }
        for w in &outcome.warnings {
            println!("  warning: {w}");
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn pack(model: &str, json: bool, data_dir: Option<&Path>) -> Result<ExitCode> {
    let layout = open_layout(data_dir)?;
    let id = aliases::resolve_reference(&layout, model)?;
    let artifact = aliases::load_manifest(&layout, &id)?;
    let (updated, outcome) = pack_artifact(&layout, &artifact, &PackOptions::default())?;
    let info = updated.packed.clone().expect("pack recorded");
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "artifact_id": id.as_str(),
                "packed": info,
                "already_packed": outcome.already_packed,
                "resumed": outcome.resumed,
            }))
            .map_err(|e| tuckmill_manifest::error::json_error(
                e,
                ErrorCode::Internal,
                "json"
            ))?
        );
    } else {
        println!(
            "{} {} ({} segments, {} total)",
            if outcome.already_packed {
                "already packed:"
            } else {
                "packed:"
            },
            id,
            info.segments,
            Bytes(info.file_bytes)
        );
        if outcome.resumed {
            println!("  resumed from an interrupted pack");
        }
    }
    Ok(ExitCode::SUCCESS)
}

// ---------------- plan ----------------

fn plan(
    model: &str,
    memory_budget: Option<&str>,
    context: u64,
    max_output: u64,
    json: bool,
    data_dir: Option<&Path>,
) -> Result<ExitCode> {
    let runtime = open_runtime(data_dir, "in-process", true)?;
    let budget = resolve_budget(&runtime, memory_budget)?;
    let (artifact, adapter) = runtime.resolve_artifact(model)?;
    let plan = runtime.plan_for(&artifact, &adapter, budget, context, max_output)?;

    // Persist the plan record (inspectable, immutable).
    let dir = runtime.layout.plans_dir();
    std::fs::create_dir_all(&dir).map_err(TuckmillError::from)?;
    let path = dir.join(format!("{}.json", plan.plan_id.as_str()));
    let bytes = serde_json::to_vec_pretty(&plan).map_err(|e| {
        tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "plan serialization")
    })?;
    std::fs::write(&path, bytes).map_err(TuckmillError::from)?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&plan).map_err(|e| {
                tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "json")
            })?
        );
        runtime.shutdown();
        return Ok(ExitCode::SUCCESS);
    }

    println!("plan {} ({})", plan.plan_id, plan.confidence.as_str());
    println!(
        "  model: {} ({}, {})",
        artifact
            .model_name
            .clone()
            .unwrap_or_else(|| artifact.file_name.clone()),
        plan.adapter_id,
        artifact
            .quant_label
            .clone()
            .unwrap_or_else(|| "unknown".into())
    );
    println!(
        "  mode: {} | class: {} | engine: {}",
        plan.mode.as_str(),
        plan.performance_class.as_str(),
        plan.engine.engine
    );
    println!(
        "  context: {} total (input {} + output {})",
        plan.context.total_context, plan.context.max_input_tokens, plan.context.max_output_tokens
    );
    println!(
        "  budget: {} = reservation {} + overhead {} + safety reserve {} (headroom {})",
        plan.budget.requested_budget,
        plan.budget.reservation,
        plan.budget.overhead_estimate,
        plan.budget.safety_reserve,
        plan.budget.headroom()
    );
    for (k, v) in plan.budget.reservation_breakdown.as_map() {
        if v > 0 {
            println!("    {k}: {}", Bytes(v));
        }
    }
    println!(
        "  cache policy: {} (prefetch {}, in-flight protected {})",
        plan.cache_policy.policy, plan.cache_policy.prefetch, plan.cache_policy.protect_in_flight
    );
    if !plan.engine.required_kernels.is_empty() {
        println!(
            "  required kernels: {} — implemented in this build: none",
            plan.engine.required_kernels.join(", ")
        );
    }
    if !plan.engine.required_transforms.is_empty() {
        println!(
            "  required transforms (part of the execution contract): {}",
            plan.engine.required_transforms.join(", ")
        );
    }
    println!("  reason: {}", plan.reason);
    if let Some(r) = &plan.rejection {
        println!("  rejection[{}]: {}", r.code, r.message);
    }
    println!("  assumptions:");
    for a in &plan.assumptions {
        println!("    - {a}");
    }
    println!("  invalidated by:");
    for c in &plan.invalidation_conditions {
        println!("    - {c}");
    }
    println!("  saved: {}", path.display());
    runtime.shutdown();
    Ok(ExitCode::SUCCESS)
}

// ---------------- run ----------------

#[allow(clippy::too_many_arguments)]
fn run(
    model: &str,
    prompt: Option<String>,
    memory_budget: Option<&str>,
    context: u64,
    max_output: u64,
    allow_test_engine: bool,
    engine: &str,
    quiet: bool,
    json: bool,
    data_dir: Option<&Path>,
) -> Result<ExitCode> {
    let runtime = open_runtime(data_dir, engine, allow_test_engine)?;
    let budget = resolve_budget(&runtime, memory_budget)?;
    let (artifact, adapter) = runtime.resolve_artifact(model)?;
    let plan = runtime.plan_for(&artifact, &adapter, budget, context, max_output)?;
    runtime.plan_admitted(&plan)?;
    runtime.admit_engine(&plan)?;

    let prompt = match prompt {
        Some(p) => p,
        None => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .map_err(TuckmillError::from)?;
            if buf.trim().is_empty() {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    "no prompt: pass --prompt or pipe text on stdin",
                ));
            }
            buf
        }
    };
    let identity = runtime.engine_identity();
    let synthetic = tuckmill_backend::is_synthetic_engine(&identity.engine);
    let prompt_tokens = if synthetic {
        tuckmill_supervisor::engine::synthetic_prompt_tokens(&prompt, 4096)
    } else {
        // Real tokenization from the artifact's own tokenizer tables.
        let tokenizer = artifact_tokenizer(&runtime, &artifact)?;
        if let Some(note) = tokenizer.pretokenizer_note() {
            if !quiet {
                eprintln!("note: {note}");
            }
        }
        tokenizer.encode(&prompt, false)?
    };

    if !quiet {
        if synthetic {
            eprintln!(
                "reference test engine — synthetic output; no model weights are executed (plan {}, mode {})",
                plan.plan_id.short(),
                plan.mode.as_str()
            );
        } else {
            eprintln!(
                "backend {} [{}] — real weights executed; uncertified kernels and quality (plan {}, mode {})",
                identity.engine,
                identity.kernels.join(","),
                plan.plan_id.short(),
                plan.mode.as_str()
            );
        }
    }

    let artifact_ref = tuckmill_backend::ArtifactRef::from_record(
        &artifact,
        &runtime.layout.artifact_payload_path(&artifact.artifact_id),
    );
    let ticket = runtime.submit()?;
    let run = runtime.start_run(
        &ticket,
        plan.clone(),
        artifact_ref,
        prompt_tokens,
        max_output,
        Vec::new(),
        0,
    )?;

    let mut last_state = String::new();
    let mut stderr = std::io::stderr();
    let mut stdout = std::io::stdout();
    let mut committed: u64 = 0;
    let mut collected_text = String::new();
    let started = Instant::now();
    let outcome_reason;

    loop {
        if INTERRUPTED.load(Ordering::SeqCst) {
            run.request_cancel();
        }
        match run.events.recv_timeout(Duration::from_millis(100)) {
            Ok(RunEvent::State(s)) => {
                if s != last_state && !quiet {
                    let _ = writeln!(stderr, "· {s}");
                }
                last_state = s;
            }
            Ok(RunEvent::Token { text, .. }) => {
                committed += 1;
                if json {
                    // Machine mode: the full JSON record is the only stdout
                    // output; tokens are collected into it.
                    collected_text.push_str(&text);
                } else {
                    let _ = stdout.write_all(text.as_bytes());
                    let _ = stdout.flush();
                }
            }
            Ok(RunEvent::Done {
                reason,
                committed: c,
            }) => {
                committed = c;
                outcome_reason = reason;
                break;
            }
            Ok(RunEvent::Error {
                code,
                message,
                action,
            }) => {
                if !json {
                    let _ = writeln!(stdout);
                }
                runtime.shutdown();
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "request_id": run.request_id,
                            "committed": run.committed.load(Ordering::Relaxed),
                            "error": {"code": code, "message": message, "action": action},
                        }))
                        .map_err(|e| {
                            tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "json")
                        })?
                    );
                }
                return Err(TuckmillError::new(
                    serde_json::from_value::<ErrorCode>(serde_json::Value::String(code))
                        .unwrap_or(ErrorCode::WorkerFailed),
                    message,
                )
                .with_action(action));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                outcome_reason = "error".to_string();
                break;
            }
        }
    }
    if !json {
        let _ = writeln!(stdout);
    }
    let elapsed = started.elapsed();

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "request_id": run.request_id,
                "session_id": run.session_id.as_str(),
                "plan_id": plan.plan_id.as_str(),
                "reason": outcome_reason,
                "committed_tokens": committed,
                "text": collected_text,
                "elapsed_ms": elapsed.as_millis() as u64,
                "backend": identity.engine,
                "certified": identity.certified,
                "synthetic_output": synthetic,
            }))
            .map_err(|e| tuckmill_manifest::error::json_error(
                e,
                ErrorCode::Internal,
                "json"
            ))?
        );
    } else if !quiet {
        let tps = committed as f64 / elapsed.as_secs_f64().max(1e-9);
        eprintln!(
            "done: {outcome_reason}, {committed} tokens committed in {:.2}s ({tps:.1} tok/s, {} back end{})",
            elapsed.as_secs_f64(),
            identity.engine,
            if synthetic { ", synthetic output" } else { ", uncertified" }
        );
    }
    runtime.shutdown();

    if INTERRUPTED.load(Ordering::SeqCst) {
        return Ok(ExitCode::from(130));
    }
    Ok(ExitCode::SUCCESS)
}

// ---------------- serve ----------------

static SERVE_SHUTDOWN: OnceLock<Arc<AtomicBool>> = OnceLock::new();

fn serve(
    port: u16,
    allow_test_engine: bool,
    engine: &str,
    probe_only: bool,
    json: bool,
    data_dir: Option<&Path>,
) -> Result<ExitCode> {
    let runtime = open_runtime(data_dir, engine, allow_test_engine)?;
    let server = tuckmill_supervisor::api::ApiServer::bind(runtime.clone(), port, None)?;
    let addr = server.local_addr();
    let token_file = tuckmill_supervisor::api::token_file_path(&runtime);

    let identity = runtime.engine_identity();
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "listening": addr.to_string(),
                "token_file": token_file.display().to_string(),
                "allow_test_engine": allow_test_engine,
                "engine_mode": runtime.engine_mode_label_str(),
                "backend": identity.engine,
                "certified": identity.certified,
                "default_budget_bytes": runtime.default_budget.bytes(),
            }))
            .map_err(|e| tuckmill_manifest::error::json_error(
                e,
                ErrorCode::Internal,
                "json"
            ))?
        );
    } else {
        println!("tuckmill serve: listening on http://{addr}");
        println!(
            "token file: {} (Authorization: Bearer <token>)",
            token_file.display()
        );
        println!("routes: GET /healthz, GET /v1/models, POST /v1/chat/completions, GET /control/status, POST /control/cancel, GET /control/bench/latest");
        println!(
            "engine: {} ({}) | test-engine accepted: {}",
            identity.engine,
            runtime.engine_mode_label_str(),
            if allow_test_engine {
                "yes (synthetic output)"
            } else {
                "no"
            }
        );
        if !tuckmill_backend::is_synthetic_engine(&identity.engine) {
            println!(
                "note: experimental backend — real weights, scalar reference kernels, uncertified"
            );
        }
        println!(
            "default request budget: {} (hardware-adaptive; override per request)",
            runtime.default_budget
        );
        println!("press ctrl-c to stop");
    }

    if probe_only {
        runtime.shutdown();
        return Ok(ExitCode::SUCCESS);
    }

    let shutdown = Arc::new(AtomicBool::new(false));
    let _ = SERVE_SHUTDOWN.set(shutdown.clone());
    let result = server.serve(shutdown);
    runtime.shutdown();
    result?;
    Ok(ExitCode::SUCCESS)
}

/// Called from the SIGINT handler path via the global flag.
pub fn serve_shutdown_flag() -> Option<&'static Arc<AtomicBool>> {
    SERVE_SHUTDOWN.get()
}

// ---------------- bench ----------------

#[allow(clippy::too_many_arguments)]
fn bench(
    model: &str,
    suite: &str,
    input: u32,
    output: u32,
    repetitions: u32,
    memory_budget: Option<&str>,
    context: u64,
    allow_test_engine: bool,
    engine: &str,
    json: bool,
    data_dir: Option<&Path>,
) -> Result<ExitCode> {
    if suite != "standard-v1" {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            format!("unknown suite {suite:?}; this build provides standard-v1"),
        ));
    }
    let runtime = open_runtime(data_dir, engine, allow_test_engine)?;
    let budget = resolve_budget(&runtime, memory_budget)?;
    let (artifact, adapter) = runtime.resolve_artifact(model)?;
    let plan = runtime.plan_for(&artifact, &adapter, budget, context, output as u64)?;
    runtime.plan_admitted(&plan)?;
    runtime.admit_engine(&plan)?;

    let options = BenchOptions {
        input_tokens: input,
        output_tokens: output,
        repetitions,
        suite: suite.to_string(),
        ..Default::default()
    };
    let artifact_ref = tuckmill_backend::ArtifactRef::from_record(
        &artifact,
        &runtime.layout.artifact_payload_path(&artifact.artifact_id),
    );
    let record = run_and_record(&runtime, &plan, &artifact_ref, &options)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&record).map_err(|e| {
                tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "json")
            })?
        );
    } else {
        println!("{}", format_summary(&record));
        println!(
            "record: {}",
            runtime
                .layout
                .bench_dir()
                .join(format!("{}.json", record.run_id.as_str()))
                .display()
        );
    }
    runtime.shutdown();
    Ok(ExitCode::SUCCESS)
}

/// Load the artifact's own tokenizer (bounded header read, no weights).
fn artifact_tokenizer(
    runtime: &Arc<Runtime>,
    artifact: &tuckmill_manifest::records::ModelArtifact,
) -> Result<tuckmill_tuckcore::tokenizer::Tokenizer> {
    let path = runtime.layout.artifact_payload_path(&artifact.artifact_id);
    let insp = gguf::inspect(&path, &gguf::Limits::default())?;
    tuckmill_tuckcore::tokenizer::Tokenizer::from_gguf(&insp)
}

// ---------------- verify ----------------

#[allow(clippy::too_many_arguments)]
fn verify(
    model: &str,
    memory_budget: Option<&str>,
    context: u64,
    positions: u64,
    generate: u64,
    reference: bool,
    tolerance: f64,
    json: bool,
    data_dir: Option<&Path>,
) -> Result<ExitCode> {
    let runtime = open_runtime(data_dir, "tuckcore", false)?;
    let budget = resolve_budget(&runtime, memory_budget)?;
    let (artifact, adapter) = runtime.resolve_artifact(model)?;
    let plan = runtime.plan_for(&artifact, &adapter, budget, context, generate.max(1))?;
    runtime.plan_admitted(&plan)?;
    let artifact_ref = tuckmill_backend::ArtifactRef::from_record(
        &artifact,
        &runtime.layout.artifact_payload_path(&artifact.artifact_id),
    );
    let report = if reference {
        let options = tuckmill_supervisor::verify::ReferenceCompareOptions {
            max_score_positions: positions,
            generate_tokens: generate,
            max_abs_tolerance: tolerance,
            ..Default::default()
        };
        runtime.run_reference_comparison(&plan, &artifact_ref, &options)?
    } else {
        let options = tuckmill_supervisor::verify::VerifyOptions {
            max_score_positions: positions,
            generate_tokens: generate,
            ..Default::default()
        };
        runtime.run_equivalence(&plan, &artifact_ref, &options)?
    };
    let passed = report.passed;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| {
                tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "json")
            })?
        );
    } else {
        println!("{}", tuckmill_supervisor::verify::format_report(&report));
        println!(
            "record: {}",
            runtime
                .layout
                .records_dir()
                .join("verify")
                .join(format!("{}.json", report.run_id.as_str()))
                .display()
        );
    }
    runtime.shutdown();
    if !passed {
        return Ok(ExitCode::from(1));
    }
    Ok(ExitCode::SUCCESS)
}

// ---------------- cache ----------------

fn cache_list(json: bool, data_dir: Option<&Path>) -> Result<ExitCode> {
    let layout = open_layout(data_dir)?;
    let inv = cache::inventory(&layout)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&inv).map_err(
                |e| tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "json")
            )?
        );
        return Ok(ExitCode::SUCCESS);
    }
    if inv.artifacts.is_empty() {
        println!("no installed artifacts");
    } else {
        println!("installed artifacts:");
        for a in &inv.artifacts {
            println!(
                "  {}  {}  {}  {}{}",
                &a.artifact_id[..12],
                a.alias.clone().unwrap_or_else(|| "-".into()),
                a.quant_label.clone().unwrap_or_else(|| "?".into()),
                Bytes(a.bytes),
                if a.has_pack {
                    format!(" (+pack {})", Bytes(a.pack_bytes))
                } else {
                    String::new()
                }
            );
        }
    }
    println!(
        "totals: artifacts {} | packs {} | tmp {} | downloads {} | reclaimable ~{}",
        Bytes(inv.artifact_bytes),
        Bytes(inv.pack_bytes),
        Bytes(inv.tmp_bytes),
        Bytes(inv.download_bytes),
        Bytes(inv.reclaimable_bytes)
    );
    Ok(ExitCode::SUCCESS)
}

fn cache_prune(dry_run: bool, json: bool, data_dir: Option<&Path>) -> Result<ExitCode> {
    let layout = open_layout(data_dir)?;
    let report = cache::prune(
        &layout,
        dry_run,
        Duration::from_secs(DEFAULT_PRUNE_AGE_SECS),
    )?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| {
                tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "json")
            })?
        );
        return Ok(ExitCode::SUCCESS);
    }
    println!(
        "{}: {} entries, {} {}",
        if dry_run { "dry-run" } else { "pruned" },
        report.removed.len(),
        Bytes(report.bytes_reclaimed),
        if dry_run {
            "would be reclaimed"
        } else {
            "reclaimed"
        }
    );
    for e in &report.removed {
        println!("  {} ({})", e.path, Bytes(e.bytes));
    }
    if report.skipped_active > 0 {
        println!(
            "{} active or protected entries were skipped (artifacts and source files are never removed)",
            report.skipped_active
        );
    }
    Ok(ExitCode::SUCCESS)
}
