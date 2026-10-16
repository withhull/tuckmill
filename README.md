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
> isolation, the local API, and the benchmark harness. Three backends exist
> (ADR-005): the *reference test engine* (synthetic output,
> `--allow-test-engine`), **TuckCore** (experimental executor for dense Qwen3:
> bounded block streaming, explicit buffer ownership, completion-fenced
> release, artifact identity verified before execution, real byte-level BPE
> tokenization), and the **pinned llama.cpp reference** (`--engine llama`,
> runtime-loaded, never vendored). Stage A is underway: Qwen3-0.6B executes
> end-to-end on real weights, `verify --reference` records teacher-forced
> logits under registered per-dtype tolerances, and the measured gap to
> llama.cpp named the kernel work: K01's first CPU increment (NEON fused
> packed dot, zero-copy mapped weight blocks) is implemented and re-validated
> — scheduling stays bit-exact, the registered envelopes are unchanged under
> both forced selections, and the matched smoke matrix separates
> vectorization from copy-avoidance effects. Kernel implementation and
> weight storage are explicit, recorded load choices (`--kernels`,
> `--storage`); eviction counts are never presented as RAM release.
> Nothing is certified; no Metal kernel exists (K02–K09 specified); no
> performance comparative claim is made. Stage A verification is hardened:
> malformed scoring fails closed, verification records are immutable with
> held-out qualification, the governor drives real backend commands
> (protocol v7), tokenization is reference-first (llama.cpp when configured;
> non-GPT-2 pre-tokenizers are refused, never approximated), and executors
> verify the artifact's full content digest before running. Chat requests
> execute with the artifact's own template and typed sampling through the
> worker: prompts are rendered by a byte-exact recognized Qwen3 template and
> tokenized reference-first; completed stop sequences and EOS markers never
> reach visible output — including a stop completing inside a token's text
> with trailing bytes — with visible (`committed`) and sampled (`generated`)
> token counts tracked separately and byte provenance preserved for UTF-8
> characters split across tokens; the worker refuses to load when its live
> arenas exceed the plan's reservations (KV is priced at the executed
> representation, 4 bytes per element in TuckCore); and equivalent requests
> reuse one loaded model with isolated per-request state — a reused request
> allocates KV for its own admitted need, never for a larger prepared
> capacity. HTTP streaming reads a request-scoped identity snapshot and can
> never block on the generating worker; governor commands are scoped to the
> request that was active when they were decided; benchmark throughput
> counts model tokens from the worker's Done frame (record schema v4), with
> visible-chunk cadence reported separately. See
> [docs/certification-matrix.md](docs/certification-matrix.md) and
> [docs/claims.md](docs/claims.md).

## Quickstart

```bash
scripts/build.sh                # cargo build + native ABI stub check
scripts/test.sh                 # fmt + clippy -D warnings + full test suite
scripts/demo.sh                 # end-to-end on a generated fixture
scripts/fetch-llama.sh          # pinned llama.cpp reference build + tm shim (into .deps/)
                                # powers --engine llama, verify --reference, and reference-first tokenization

tuckmill doctor                                  # device profile, honest labels
tuckmill inspect model.gguf                      # validate without loading weights
tuckmill import model.gguf --alias my-model      # atomic, resumable, verified
tuckmill plan my-model                           # R+U+H<=B admission, mode, assumptions
tuckmill run my-model --prompt "explain RoPE" --engine tuckcore   # real weights
tuckmill run my-model --engine tuckcore --kernels scalar --storage owned   # forced reference variants
tuckmill verify my-model                         # residency/streaming equivalence
tuckmill verify my-model --reference             # llama.cpp vs TuckCore logits (needs TUCKMILL_LLAMA_LIB);
                                                 # registered per-dtype tolerances
tuckmill verify my-model --reference --kernels scalar --storage owned   # forced, recorded, fail-closed selection
tuckmill run my-model --allow-test-engine        # synthetic pipeline run
tuckmill bench my-model --engine tuckcore        # record: e2e + decode-only throughput, resolved selections
tuckmill bench my-model --engine llama --gpu-layers 99   # Metal baseline (performance only)
tuckmill stats my-model --engine tuckcore        # resolved kernel/storage, storage accounting, phase counters
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
| `run --engine tuckcore` — real-weight dense-Qwen3 execution: bounded block loader (LRU byte budget, pinned/leased buffers, selectable mapped/owned storage over one verified file handle), scalar f32 forward (RMSNorm, NeoX RoPE, GQA, SwiGLU) plus the K01 fused Q8_0/F16 dot with selectable scalar/NEON implementation and a checked shape boundary, F32/F16/BF16/Q8_0/Q4_0/Q4_1 | ✅ experimental, uncertified |
| `verify` — resident vs forced-streaming with identical token IDs: teacher-forced logits (bit-exact at Stage 0), eviction and footprint evidence, machine-readable record | ✅ |
| Governor — Normal/Constrained/Paused/Draining/Failed with stability-window resume; driven by real memory-pressure and thermal signals | ✅ |
| `bench` — standard-v1 records with per-prompt data, cache-state labels, caveats | ✅ |
| llama.cpp/ggml reference backend (`--engine llama`) — pinned revision via a header-checked C shim, loaded at runtime (never vendored); `verify --reference` compares it against TuckCore on identical token IDs | ✅ integrated (wrapper uncertified) |
| Custom CPU kernels | 🟡 K01's first increment — aarch64 NEON fused Q8_0/F16 dot with zero-copy mapped blocks — implemented and numerically validated; K02–K09 specified |
| Custom Metal kernels | ⛔ none implemented; the scalar CPU path remains the arithmetic oracle |
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
   ├── tuckcore         experimental custom executor: scalar f32 reference kernels + first NEON increment (K01)
   └── reference        pinned llama.cpp/ggml (compatibility + oracle; wrapper uncertified)
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
crates/tuckcore     experimental executor: zero-copy block loader, completion fences, Qwen3 forward (scalar oracle + K01 NEON kernels), tokenizer
crates/llamaref     pinned llama.cpp reference backend (dlopen of the checked tm shim)
crates/supervisor   queue, ledger, governor, worker protocol/process, API, bench, verify
crates/cli          the tuckmill binary (incl. hidden __worker-reference)
native/executor     C ABI header + protocol spec (PROTOCOL.md, v7)
native/llamaref     tm_llama_shim.c: narrow C surface over pinned llama.cpp
kernels/            K01–K09 work-package registry (K01 implemented increment; K02–K09 specified)
adapters/           qwen3, qwen3moe, qwen3.5-hybrid, bonsai2-hybrid specs
benchmarks/         standard-v1 frozen workload; comparator program (llama.cpp, MLX, Splash)
docs/adr/           ADR-001..006
docs/               certification matrix, claim ledger
scripts/            build.sh, test.sh, demo.sh
```

## Verification and benchmarks

`scripts/test.sh` runs formatting, `clippy -D warnings`, and the full test
suite (351 tests),
including adversarial GGUF/Safetensors fixtures, planner math, governor
semantics, import/pack fault injection, worker-protocol framing, HTTP
auth/SSE (including long streams through the real endpoint), CLI end-to-end
runs (isolated worker, crash injection, SIGINT cancel, truncated/replaced
payloads refused), and TuckCore equivalence. Test labels are explicit:
portable tests always run; isolated-worker tests spawn the built binary and
are mandatory in the gate (`TUCKMILL_TEST_WORKER_BIN`, with
`TUCKMILL_REQUIRE_ISOLATED_WORKER=1` making a missing binary a failure, not
a skip); real-weight reference tests (pinned llama.cpp library,
`TUCKMILL_REAL_GGUF`) and physical-Mac qualification are separate,
machine-local gates.
`rust-toolchain.toml` pins Rust 1.96.0 and CI runs that same pinned
toolchain — a floating stable once failed the remote gate on lints the local
toolchain could not reproduce (run 38910541208). `scripts/demo.sh` exercises
the full flow on a generated fixture.

Performance and quality claims require measuring against llama.cpp, MLX, and
Splash across memory tiers, with identical artifacts, token IDs, and declared
numerical profiles (ADR-006). No comparative claim exists until the
measurement is in the record.

## License

Provisional: open-source core (Apache-2.0); the final licensing decision and
naming/trademark clearance are outstanding launch dependencies. Importing a
model artifact is **not** a license review; model weights carry their own
licenses.
