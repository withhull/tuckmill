# Tuckmill

Tuckmill is a Mac-native local inference framework for Apple Silicon, from
8 GB MacBook Airs to high-memory Macs. It runs the model you chose, with its
own computation and numerical profile, as fast as each Mac allows, and
labels every number it shows (`estimated`, `locally-calibrated` or
`certified`).

Positioning and goals: [ADR-006](docs/adr/ADR-006-positioning-and-memory-tiers.md).
Product contract: [ADR-001](docs/adr/ADR-001-product-contract.md).

## Status

Engineering build; nothing is certified. Evidence lives in
[docs/certification-matrix.md](docs/certification-matrix.md), and allowed
statements in [docs/claims.md](docs/claims.md).

- **Orchestration**: implemented and tested. This covers inspect, import,
  pack, plan (`R + U + H <= B` admission), an isolated worker, an
  OpenAI-style `/v1/chat/completions` endpoint with SSE, `verify
  --reference` against pinned llama.cpp, and labeled bench records.
- **TuckCore executor**:
  - **Unified engine (ADR-011):** an adapter describes a model as a typed
    graph, and one Metal engine runs it from shared blocks. Qwen3.5 0.8B
    and 2B (Q8_0) are served this way, with batched prefill and pipelined
    greedy decoding on the GPU.
  - **Dense Qwen3:** runs on its older CPU and Metal paths until it
    migrates to the engine.
- **Reference backend**: pinned llama.cpp b11695 (`--engine llama`). It is
  loaded at runtime, never vendored, and is both the comparison oracle and
  the speed baseline.
- **Not built yet**: GPU partial residency and streaming, MoE execution,
  speculative decoding, prefix caching, K-quant kernels.
- **Hardware exercised**: one MacBook Air M3 8 GB. Other tiers are pending.

### Measured against llama.cpp (Qwen3.5-0.8B Q8_0, M3 8 GB)

Identity-matched harness (`scripts/bench-baseline.sh`): same artifact, same
prompt tokens, 512 tokens in and 256 out, ABBA rounds. Idle run at
`d95ecc1`, locally-calibrated:

| | TuckCore | llama.cpp Metal | Ratio |
| --- | --- | --- | --- |
| Time to first token | 0.272 s | 0.270 s (0.269-0.287) | 99% of llama.cpp |
| Decode | 84.3 tok/s | 75.9 tok/s (67.9-84.7) | 1.11× (0.99-1.24 paired) |

- **Phase 1 bar** (≥ 90% of llama.cpp for decode and prefill): met in this
  run on this model. It is one run, so a repeat comes before
  certification.
- **Decode** is at parity or ahead. llama.cpp's own decode in this harness
  varies between runs (about 66-85 tok/s), so this is not stated as a
  lead.
- **Remaining prefill difference:** f16 vs f32 operand precision. TuckCore
  keeps f32 (ADR-011).
- **Not measured:** other models, contexts and memory tiers. Records are in
  [benchmarks/results/](benchmarks/results/).

## Quickstart

```bash
scripts/build.sh         # cargo build + native ABI stub check
scripts/test.sh          # fmt + clippy -D warnings + full test suite
scripts/demo.sh          # end-to-end on a generated fixture
scripts/fetch-llama.sh   # pinned llama.cpp reference + tm shim into .deps/
```

```bash
tuckmill doctor                                    # device profile, honest labels
tuckmill inspect model.gguf                        # validate without loading weights
tuckmill import model.gguf --alias my-model        # atomic, resumable, verified
tuckmill plan my-model                             # admission, mode, assumptions
tuckmill run my-model --prompt "..." --engine tuckcore --kernels metal
tuckmill verify my-model --reference               # TuckCore vs llama.cpp logits
tuckmill bench my-model --engine tuckcore          # labeled benchmark record
tuckmill serve --engine tuckcore --kernels metal   # loopback API (127.0.0.1:8378)
scripts/bench-baseline.sh model.gguf               # ABBA: TuckCore vs llama.cpp
```

The default process budget is a quarter of unified memory, at least 2 GiB
and at most 24 GiB. `--memory-budget` overrides it.

## Execution strategy by memory tier

There is one engine, and the planner chooses a strategy per model and per
Mac by admission (`R + U + H <= B`). Execution is resident-first: when the
weights fit, they load once and stay resident. Streaming is a strategy for
constrained machines, not the architecture.

| Unified memory | Strategy | State |
| --- | --- | --- |
| 8 GB | Compact weights, strict budgeting, selective streaming | budgeting and admission ✅; resident Metal ✅ (≤ 2B); CPU block streaming ✅ (dense Qwen3); GPU streaming ⛔ Phase 2 |
| 16–24 GB | Hybrid resident and streamed execution, adaptive caching | ⛔ Phases 2–3 |
| 32–48 GB | Larger resident models, fast prefill, speculative decoding | resident Metal ✅ (small models); speculation ⛔ Phase 4 |
| 64–128 GB+ | Maximum resident execution, batching, low-latency decoding | ⛔ Phases 1, 4 |

## What works today

| Component | State |
| --- | --- |
| `doctor`: device profile from the system; unavailable signals are labeled, never guessed | ✅ |
| `inspect`: GGUF v2/v3 and Safetensors, validated with checked arithmetic before any allocation | ✅ |
| `import` / `pack`: resumable chunked-digest imports, atomic publish, lossless packer with per-tile checksums | ✅ |
| `plan`: `R + U + H ≤ B` admission, KV math, mode selection, typed infeasibility, recorded assumptions | ✅ |
| `run` / `serve`: isolated worker, framed protocol, streaming, cancellation, bounded queue, loopback HTTP with bearer auth and SSE | ✅ |
| `verify`: resident vs streamed, and TuckCore vs llama.cpp, on identical token IDs with registered tolerances | ✅ |
| `bench`: standard-v1 records with cache-state labels and caveats | ✅ |
| Governor: Normal / Constrained / Paused / Draining / Failed, driven by real memory-pressure and thermal signals | ✅ |
| Qwen3.5 hybrid on Metal (unified engine): Gated DeltaNet and gated attention, packed f16 KV, batched prefill, device greedy sampling | ✅ experimental; raw completion only, chat ⛔ (template not recognized) |
| Dense Qwen3: scalar f32 oracle and NEON Q8_0/F16 kernel on CPU, resident Metal path | ✅ experimental; engine migration pending (ADR-011 step 2) |
| Formats executed | CPU: F32, F16, BF16, Q8_0, Q4_0, Q4_1. Metal: Q8_0, F16. K-quants and IQ formats are refused |
| GPU partial residency and streaming, MoE execution, compressed (Bonsai) execution | ⛔ Phases 2–3 |

## Architecture

```text
CLI / local API client
        |
Rust supervisor: admission, queue, governor, cancellation, non-content logs
        |
planner -> versioned ExecutionPlan (mode, budget, assumptions, confidence)
        |
InferenceBackend interface (ADR-005)
   ├── tuckcore         custom executor: unified Metal engine (ADR-011), CPU oracle
   ├── reference        pinned llama.cpp/ggml (compatibility and oracle)
   └── reference-test   synthetic pipeline engine (disclosed)
```

- **Supervisor** (`crates/supervisor`): never computes tensors. It stays
  responsive when a worker fails and fabricates nothing.
- **Planner** (`crates/planner`): prices memory and latency from artifact
  metadata and device measurements. Projections stay `estimated` until they
  are measured.
- **Store** (`crates/store`): keeps data under
  `~/Library/Application Support/Tuckmill` (override with
  `TUCKMILL_DATA_DIR`), installs atomically and never deletes sources.

## Repository layout

```text
crates/manifest     schemas, GGUF/Safetensors inspection, adapters, fixtures
crates/backend      versioned inference-backend interface
crates/planner      device profiler, admission and KV math, cost model
crates/store        imports, packing, cache/GC
crates/tuckcore     custom executor: unified engine, CPU oracle, loader, tokenizer
crates/metal        typed Rust handles over the Metal C ABI
crates/llamaref     pinned llama.cpp reference backend
crates/supervisor   queue, governor, worker protocol, API, bench, verify
crates/cli          the tuckmill binary
native/executor     executor C ABI header and worker protocol spec
native/metal        Metal C ABI (tm_metal.h) and MSL kernels (ADR-007)
native/llamaref     narrow C shim over pinned llama.cpp
kernels/            kernel work-package registry
adapters/           model-family adapter specs
benchmarks/         standard-v1 workload and per-tier results
docs/adr/           architecture decisions (ADR-001..011)
scripts/            build, test, demo, fetch-llama, bench-baseline
```

## Verification

`scripts/test.sh` is the gate (492 tests at this revision). It runs
formatting, `clippy -D warnings` and the full suite:
- adversarial GGUF and Safetensors fixtures;
- planner math, governor semantics and import fault injection;
- worker protocol and HTTP/SSE;
- CLI end-to-end runs with an isolated worker, crash injection and
  cancellation;
- Metal kernels and engine blocks qualified against the CPU oracle.

GPU and isolated-worker tests are mandatory: a missing device or worker
binary fails the gate instead of skipping. Real-weight comparisons against
pinned llama.cpp are machine-local gates. `rust-toolchain.toml` pins the
toolchain that CI also uses.

Comparisons follow the four-way identity rules in
[benchmarks/README.md](benchmarks/README.md): identical artifacts, token
IDs, numerical profiles and declared cache state. Nothing is claimed until
the measurement is in the record.

## License

Provisional: open-source core (Apache-2.0). The final licensing decision and
naming clearance are outstanding. Importing a model is not a license review;
model weights carry their own licenses.
