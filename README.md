# Tuckmill

**Big models. Small footprint.**

Tuckmill is a local-first model inference framework for Apple Silicon Macs.
It inspects a model and the host, produces a feasible execution plan, and
runs through a resident or storage-backed engine — with explicit quality,
memory, and performance controls. The runtime changes scheduling and cache
allocation, **never the requested model's identity**: no dropped experts, no
silent truncation, no precision substitutions, no remote fallback.

Engineered against the Tuckmill PRD v0.2 (October 16, 2026).

> **Honest build status (read this first).** The orchestration stack is
> implemented and tested — inspection, planning, admission, storage, packing,
> worker isolation, the local API, and the benchmark harness. Two backends
> exist (ADR-005): the *reference test engine* (synthetic output, explicitly
> accepted with `--allow-test-engine`) and **TuckCore**, an experimental
> custom executor with a versioned backend interface that executes **real
> weights** for dense Qwen3 GGUF via scalar f32 reference kernels, bounded
> block streaming, explicit buffer ownership, and completion-fenced release.
> TuckCore is **uncertified**: no Metal kernels (K01–K09 are specified, not
> implemented), no performance claim, no tokenizer yet (prompts map
> deterministically into the artifact's vocabulary and the limitation is
> labeled in output). `tuckmill verify` proves bit-exact equivalence between
> resident and forced-streaming execution over identical token IDs. See
> [`docs/certification-matrix.md`](docs/certification-matrix.md) and
> [`docs/claims.md`](docs/claims.md).

## Quickstart

```bash
cargo build --release          # or: scripts/build.sh
./target/release/tuckmill doctor               # chip, memory, OS, signals, honesty report

# Try the whole pipeline on a generated fixture (no real model needed):
scripts/demo.sh

# With a real GGUF file (inspection/planning works today; execution requires
# a certified executor or --allow-test-engine):
tuckmill inspect ./model.gguf
tuckmill import ./model.gguf --alias my-model
tuckmill plan my-model --memory-budget 2GiB --context 2048
tuckmill run  my-model --prompt "explain RoPE" --allow-test-engine      # synthetic pipeline run
tuckmill run  my-model --prompt "explain RoPE" --engine tuckcore        # real weights, scalar kernels
tuckmill verify my-model                                                # bit-exact scheduling equivalence
tuckmill bench my-model --allow-test-engine
tuckmill serve --allow-test-engine          # loopback API on 127.0.0.1:8378
tuckmill cache list && tuckmill cache prune --dry-run
```

## What works today

| Layer | State |
| --- | --- |
| `doctor` — real device profile via sysctl/statfs/NSProcessInfo; unavailable signals labeled, never guessed | ✅ |
| `inspect` — GGUF v2/v3 and Safetensors validation with checked arithmetic; rejects malformed sizes/offsets before allocation | ✅ |
| `import` / `pack` — resumable 256 MiB-chunk digesting, atomic publish, bounded lossless packer with per-tile checksums and verify | ✅ |
| `plan` — `R + U + H <= B` admission, KV math, mode selection (resident / streamed-MoE / streamed-dense), typed infeasibility, deterministic cost model with explicit assumptions, estimated vs locally-calibrated labels | ✅ |
| `run` / `serve` — isolated worker process over a versioned framed protocol, streaming, cancellation, pause/resume, queueing (`QUEUE_FULL`), typed errors, loopback HTTP with bearer auth, SSE | ✅ |
| `run --engine tuckcore` — real-weight execution for dense Qwen3: bounded block loader, LRU byte budget, pinned/leased buffers with completion fences, scalar f32 forward (RMSNorm/NeoX-RoPE/GQA+SwiGLU), artifact-identity verification before execution, in-vocab deterministic prompt mapping | ✅ (experimental, uncertified) |
| `verify` — resident vs forced-streaming equivalence: identical token IDs, teacher-forced logits compared (bit-exact at Stage 0), eviction and footprint evidence recorded | ✅ |
| Governor — Normal/Constrained/Paused/Draining/Failed with 10 s stability resume and oscillation guard, driven by real memory-pressure/thermal signals | ✅ |
| `bench` — standard-v1 machine-readable records with per-prompt data, cache-state labels, and caveats | ✅ |
| llama.cpp/ggml reference backend | ⛔ not yet integrated (planned as the compatibility backend via the same interface) |
| Custom Metal kernels (K01–K09) | ⛔ specified only (`kernels/registry.json`); TuckCore Stage 0 is the scalar oracle they will be validated against |

## Architecture

```text
CLI / local API client
        |
Rust supervisor (admission, queue, governor, cancellation, non-content logs)
        |
model inspector + hardware profiler + planner  ->  versioned ExecutionPlan
        |
isolated native inference worker (framed protocol, private socket)
        |
resident | MoE streaming | dense streaming  ->  bounded tensor store + Metal (future)
```

- **Backend interface** (`crates/backend`) is a narrow, versioned contract
  (ADR-005): `load(plan, ArtifactRef, options)`, `generate`, `score`,
  `stats`. Backends: `reference-test` (synthetic), `tuckcore` (experimental
  real executor), and `reference`/llama.cpp when it lands.
- **Supervisor** (`crates/supervisor`) never computes tensors; it stays
  responsive when a worker fails (`WORKER_FAILED`, last acknowledged position
  retained, no fabricated continuation state).
- **Planner** (`crates/planner`) prices memory and latency from artifact
  metadata, adapter capabilities, device measurements, and calibration
  evidence; projections are labeled `estimated` until real measurements with
  matching identity upgrade them.
- **Store** (`crates/store`) keeps canonical data under
  `~/Library/Application Support/Tuckmill` (override: `TUCKMILL_DATA_DIR`),
  installs atomically, resumes interrupted work only when the source identity
  still matches, and never deletes sources.
- **Manifest** (`crates/manifest`) owns every schema, the typed error
  vocabulary (PRD §15 codes + internal support codes), exact byte units
  (GiB ≠ GB), and pinned GGML/GGUF constants.

## Repository layout

```text
crates/manifest     schemas, GGUF/Safetensors inspection, adapters, fixtures
crates/backend      versioned inference-backend interface (ADR-005)
crates/planner      device profiler, KV/admission math, cost model, calibration
crates/store        imports, downloads, packing, cache/GC
crates/tuckcore     experimental executor: block loader, completion fences, scalar Qwen3 forward
crates/supervisor   queue, ledger, governor, worker protocol/process, API, bench, verify
crates/cli          the tuckmill binary (incl. hidden __worker-reference)
native/executor     C ABI header + protocol spec for the future real executor
kernels/            K01-K09 work-package registry (all "specified")
adapters/           qwen3, qwen3moe, qwen3.5-hybrid, bonsai2-hybrid specs
benchmarks/         standard-v1 frozen workload definition
docs/adr/           ADR-001..005 (product contract, build, engine boundary, memory, backend interface + TuckCore)
docs/               certification matrix, claim ledger
scripts/            build.sh, test.sh, demo.sh
```

## Testing

```bash
scripts/test.sh    # fmt + clippy -D warnings + full test suite (184 tests)
```

Tests include: GGUF/Safetensors adversarial fixtures (truncation, overflow,
misalignment, unknown types), planner math against a PRD worked example,
governor escalation/oscillation semantics, queue handoff, worker protocol
framing, HTTP auth/host/origin/SSE behavior, import resume/restart/concurrent
imports, packer corruption and resume, and a full CLI end-to-end suite that
runs the real binary including the isolated worker path and a live API
session on a temporary data directory.

## License

Provisional: open-source core (Apache-2.0) per the PRD recommendation; the
final licensing decision — and naming/trademark/distribution clearance — are
outstanding launch dependencies. Importing a model artifact is **not** a
license review; model weights carry their own licenses.
