# Tuckmill

**Big models. Small footprint.**

Tuckmill is a Mac-native, high-performance inference framework engineered to
run large models as efficiently as possible across Apple Silicon — from 8 GB
MacBook Airs to high-memory MacBook Pros and Mac Studios. The goal is not
simply to run models that do not fit in RAM; it is to make the best possible
use of every Mac's compute, memory, and storage while preserving model
quality.

> **Build status (read this first).** The orchestration stack is implemented
> and tested: inspection, planning, admission, storage, packing, worker
> isolation, the local API, and the benchmark harness. Two backends exist
> (ADR-005): the *reference test engine* (synthetic output, `--allow-test-engine`)
> and **TuckCore**, an experimental executor with a versioned backend
> interface that runs **real weights** for dense Qwen3 GGUF through scalar f32
> reference kernels — bounded block streaming, explicit buffer ownership,
> completion-fenced release, artifact identity verified before execution.
> TuckCore is **uncertified**: no Metal kernels (K01–K09 are specified), no
> performance claim, no text tokenizer yet (prompts map deterministically into
> the artifact's vocabulary; the limitation is labeled everywhere it applies).
> Nothing in this repository is certified; see
> [docs/certification-matrix.md](docs/certification-matrix.md) and
> [docs/claims.md](docs/claims.md).

## Quickstart

```bash
scripts/build.sh                # cargo build + native ABI stub check
scripts/test.sh                 # fmt + clippy -D warnings + full test suite
scripts/demo.sh                 # end-to-end on a generated fixture

tuckmill doctor                                  # device profile, honest labels
tuckmill inspect model.gguf                      # validate without loading weights
tuckmill import model.gguf --alias my-model      # atomic, resumable, verified
tuckmill plan my-model                           # R+U+H<=B admission, mode, assumptions
tuckmill run my-model --prompt "explain RoPE" --engine tuckcore   # real weights
tuckmill verify my-model                         # bit-exact residency/streaming equivalence
tuckmill run my-model --allow-test-engine        # synthetic pipeline run
tuckmill bench my-model --engine tuckcore        # machine-readable record
tuckmill serve --allow-test-engine               # loopback API (127.0.0.1:8378)
tuckmill cache list && tuckmill cache prune --dry-run
```

The default process budget is hardware-adaptive: a quarter of unified memory,
floored at 2 GiB and capped at 24 GiB. `--memory-budget` overrides it.

## Execution strategy by memory tier

One execution engine, different optimized strategies — never separate
products. Strategies are chosen per request from measured device state; they
are strategies, not guarantees that every model fits a tier.

| Unified memory | Strategy |
| --- | --- |
| 8 GB | Compact weights, strict memory budgeting, selective streaming, foreground protection |
| 16–24 GB | Hybrid resident/streamed execution, optimized quantized kernels, adaptive caching |
| 32–48 GB | Larger resident models, specialized Metal kernels, efficient prefill, speculative decoding where beneficial |
| 64–128 GB+ | Maximum resident execution, advanced batching, low-latency decoding, high-throughput inference |

**Resident-first:** when weights fit, they load once and stay resident.
Streaming protects constrained machines — it is a strategy, not the
architecture.

## Engineering goals

1. **Maximum performance** — prefill, token generation, first-token latency,
   sustained throughput.
2. **Maximum memory efficiency** — minimal allocations, weight movement, and
   context-state overhead.
3. **Broad model compatibility** — dense, MoE, hybrid, and compressed
   architectures through validated adapters.
4. **Hardware-adaptive execution** — resident, partially streamed, or fully
   storage-backed, driven by actual device capabilities.
5. **Preserved quality** — the selected model's computation and numerical
   profile; compressed models are evaluated separately from runtime changes.

## What works today

| Component | State |
| --- | --- |
| `doctor` — real device profile (sysctl/statfs/NSProcessInfo); unavailable signals labeled, never guessed | ✅ |
| `inspect` — GGUF v2/v3 and Safetensors with checked-arithmetic validation; malformed sizes/offsets rejected before allocation | ✅ |
| `import` / `pack` — resumable chunked-digest imports, atomic publish, bounded lossless packer with per-tile checksums and verify | ✅ |
| `plan` — `R + U + H ≤ B` admission, KV math, resident / streamed-MoE / streamed-dense selection, typed infeasibility, deterministic cost model with recorded assumptions | ✅ |
| `run` / `serve` — isolated worker process, versioned framed protocol, streaming, cancellation, pause/resume, bounded queue, typed errors, loopback HTTP with bearer auth + SSE | ✅ |
| `run --engine tuckcore` — real-weight dense-Qwen3 execution: bounded block loader (LRU byte budget, pinned/leased buffers), scalar f32 forward (RMSNorm, NeoX RoPE, GQA, SwiGLU), F32/F16/BF16/Q8_0/Q4_0/Q4_1 | ✅ experimental, uncertified |
| `verify` — resident vs forced-streaming with identical token IDs: teacher-forced logits (bit-exact at Stage 0), eviction and footprint evidence, machine-readable record | ✅ |
| Governor — Normal/Constrained/Paused/Draining/Failed with stability-window resume; driven by real memory-pressure and thermal signals | ✅ |
| `bench` — standard-v1 records with per-prompt data, cache-state labels, caveats | ✅ |
| llama.cpp/ggml reference backend | ⛔ designated (ADR-005/ADR-006), not yet integrated |
| Custom Metal kernels (K01–K09) | ⛔ specified only; TuckCore Stage 0 is the scalar oracle they will be validated against |
| MoE streaming, hybrid/compressed (Bonsai) execution | ⛔ gated milestones |

## Architecture

```text
CLI / local API client
        |
Rust supervisor — admission · queue · governor · cancellation · non-content logs
        |
planner  ->  versioned ExecutionPlan (mode, budget, assumptions, confidence)
        |
InferenceBackend interface (versioned, ADR-005)
   ├── reference-test   synthetic pipeline engine (disclosed)
   ├── tuckcore         experimental custom executor, scalar f32 reference kernels
   └── reference        pinned llama.cpp/ggml (planned: compatibility + oracle)
        |
resident | partially streamed | storage-backed execution, chosen by the planner
```

- **Supervisor** (`crates/supervisor`) never computes tensors; it stays
  responsive when a worker fails, keeping the last acknowledged position and
  fabricating nothing.
- **Planner** (`crates/planner`) prices memory and latency from artifact
  metadata, adapter capabilities, device measurements, and calibration
  evidence; projections stay labeled `estimated` until matching measurements
  exist.
- **Store** (`crates/store`) keeps canonical data under
  `~/Library/Application Support/Tuckmill` (override: `TUCKMILL_DATA_DIR`),
  installs atomically, resumes interrupted work only on matching source
  identity, and never deletes sources.

## Repository layout

```text
crates/manifest     schemas, GGUF/Safetensors inspection, adapters, fixtures
crates/backend      versioned inference-backend interface
crates/planner      device profiler, admission/KV math, cost model, calibration, kernel registry
crates/store        imports, downloads, packing, cache/GC
crates/tuckcore     experimental executor: block loader, completion fences, scalar Qwen3 forward
crates/supervisor   queue, ledger, governor, worker protocol/process, API, bench, verify
crates/cli          the tuckmill binary (incl. hidden __worker-reference)
native/executor     C ABI header + protocol spec (PROTOCOL.md, v2)
kernels/            K01–K09 work-package registry (all "specified")
adapters/           qwen3, qwen3moe, qwen3.5-hybrid, bonsai2-hybrid specs
benchmarks/         standard-v1 frozen workload; comparator program (llama.cpp, MLX, Splash)
docs/adr/           ADR-001..006
docs/               certification matrix, claim ledger
scripts/            build.sh, test.sh, demo.sh
```

## Verification and benchmarks

`scripts/test.sh` runs formatting, `clippy -D warnings`, and 236 tests,
including adversarial GGUF/Safetensors fixtures, planner math, governor
semantics, import/pack fault injection, worker-protocol framing, HTTP
auth/SSE, CLI end-to-end runs (isolated worker, crash injection, SIGINT
cancel), and TuckCore equivalence. `scripts/demo.sh` exercises the full flow
on a generated fixture.

Performance and quality claims require measuring against llama.cpp, MLX, and
Splash across memory tiers, with identical artifacts, token IDs, and declared
numerical profiles (ADR-006). No comparative claim exists until the
measurement is in the record.

## License

Provisional: open-source core (Apache-2.0); the final licensing decision and
naming/trademark clearance are outstanding launch dependencies. Importing a
model artifact is **not** a license review; model weights carry their own
licenses.
