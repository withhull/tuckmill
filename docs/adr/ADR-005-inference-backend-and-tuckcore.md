# ADR-005: Inference-backend interface and TuckCore

Status: **committed** (2026-10-16). Runtime + Systems. Amends ADR-003 (the
"one engine exists" statement) and prepares the llama.cpp integration planned
by ADR-002.

**Update (2026-10-16):** the llama.cpp reference backend is integrated as
decision 2 anticipated: pinned revision `b11695`, runtime-loaded via the
header-checked C shim `native/llamaref/tm_llama_shim.c`
(`scripts/fetch-llama.sh`), never vendored. `verify --reference` records
cross-backend teacher-forced logits under a registered tolerance; the first
profile on the fixture measured max|Δ| = 1.98e-4 against a 1e-2 registration.
The wrapper remains uncertified, and llama.cpp execution is CPU-only here
(`n_gpu_layers = 0`) — GPU offload is a separately measured milestone.

**Update 2 (Stage A, 2026-10-16):** worker protocol v3 chunks score frames so
real vocabularies (150k+ logits) fit the 1 MiB cap, and workers heartbeat
during long scoring so liveness checks stay honest without loosening
timeouts. TuckCore declares a **numerical profile** (`oracle-f32` exact
scalar, default; `llama-cpu` mirroring ggml's activation quantization);
reference comparisons default to the matched profile and select a **registered
per-dtype tolerance** (F16/F32 5e-2, Q8_0 5e-1) from measured envelopes on
Qwen3-0.6B — F16 max 1.95e-2 (smooth accumulation noise), Q8_0 max 4.6e-1
(dominated by llama.cpp CPU 8-bit activation quantization; its
architecture-specific kernel dispatch proved not reproducible from reference
semantics, so the envelope — not bit-identity — is the contract). Greedy text
divergence on near-ties is recorded, never required. The scalar oracle sits
~20x behind llama.cpp CPU on the real 0.6B — measured, per ADR-006, before
any kernel work.

**Update 3 (verification hardening, 2026-10-16):** an external review of
commit `064836a` found (and local inspection confirmed) gaps that this update
closes. (1) Verification fails closed: exact requested positions, exact
vocabulary row length, finite values, contiguous bounded chunk accumulation —
a skipped or truncated row can never produce a PASS (regression tests at the
accumulator and validator layers). (2) Evidence is immutable: a configuration
ID (full resolved settings: profiles, KV precision, tolerances, identities,
workload) is separate from a unique execution ID; interrupted executions are
persisted with `status: "interrupted"` and no verdict; files are never
overwritten. (3) KV cache precision is explicit and matched on both sides
(shim sets `type_k/type_v`; TuckCore stores f16-representable values when
asked). (4) Tolerances are a versioned per-dtype policy; unmeasured dtypes
are refused by default, explicit overrides label the run `exploratory`, and
`--held-out` qualifies on frozen disjoint inputs (Qwen3-0.6B: Q8_0 max
2.68e-1 and F16 max 1.13e-2, both PASS, top-1 agreement 100%).
(5) The governor has a real backend contract: protocol v4
`Control`/`ControlAck` frames, engines apply `trim-caches` at safe
checkpoints via a shared epoch (tested mid-generation), continuous heartbeat
threads during long compute, and cancellation is serviced by engine poll
hooks during single-step work instead of waiting for the next callback. The
monitor, `status()`, and cancellation are lock-free with respect to the
in-flight generation lock (regression-tested). (6) Misleading wording fixed:
`oracle-f32` is a **scalar FP32 reference with no additional activation
quantization**, never "exact arithmetic"; "exact" is reserved for tested
statements. (7) Tokenization is reference-first: llama.cpp (vocab-only load,
CPU-pinned, log-filtered) is the authoritative encoder; the built-in BPE is
used only where it is exact (gpt2-style, golden-tested) and non-GPT-2
pre-tokenizers are refused rather than approximated; streaming committed text
uses an incremental UTF-8 decoder. (8) TuckCore verifies the artifact's full
canonical content digest before executing (drifted payloads are refused
before any compute; disabling is an explicit recorded option).

**Update 4 (K01 + zero-copy, 2026-10-16):** the measured ~20x gap (update 2)
named the first optimization, and profiling named the order. K01's first
increment: an aarch64 NEON fused packed-dot kernel for Q8_0/F16 (scalar
fallback retained per the kernel-registry gate), plus **zero-copy read-only
`mmap` weight-block views** (the owned-read path stays as fallback). On the
real 0.6B, TuckCore moved 0.3 → 1.5 tok/s with fusion, → 2.4 tok/s (16/8) and
5.7 tok/s (64/64) with zero-copy; footprint halved to 512 MiB; measured
attribution after the work: weight fetch 0.49 s → ~0 ms (mmap removed the
copy) and matvec compute 3.43 s → 0.39 s, leaving scalar decode compute
(~78 ms/token) and unbatched prefill as the named next bottlenecks.
Re-validation: scheduling stays **bit-exact**
and the registered Q8/F16 envelopes are unchanged — smoke numbers only
(`benchmarks/README.md`), no claim. The shim gains `gpu_layers` (shim v4):
the default remains CPU (`gpu_layers = 0`) for arithmetic diagnostics, while
llama.cpp **Metal** offload is available as a labeled performance baseline
(`run`/`bench --engine llama --gpu-layers N`) and is never used for
numerical comparison. This supersedes update 1's "GPU offload is a
separately measured milestone": the Metal baseline is now measurable; the
next targets named by the measurements are K02 (microbatched prefill) and
further K01 lane work or a Metal kernel, gated per the kernel registry.
`tuckmill stats` exposes the phase counters and loader statistics behind
these numbers.

**Update 5 (K01 hardening + matched baseline, 2026-10-16):** the update-4
increment landed, and this update closes its boundary, control, and
measurement gaps. (1) **Checked kernel boundary:** the public entry points
validate packed-row alignment, activation length, block count, and scale
length per row; malformed shapes are refused with typed errors (the unsafe
NEON implementations are private behind that boundary; regression tests
include a valid block paired with an empty activation). (2) **Explicit,
recorded selection:** `--kernels auto|scalar|neon` (NEON refused off
aarch64) and `--storage auto|mapped|owned` (mapped refused when the file
cannot be mapped) resolve at load; the resolved choices are reported in the
load note, `BackendStats` (`kernel_impl`, `numerical_profile`, `storage`),
the engine identity (`K01:q8_0+f16:<impl>` — Q4 acceleration and Metal are
not implemented), and per-sample benchmark records. Forced-scalar tests
prove the selection changes the implementation. (3) **Accounting:** owned
buffers, mapped address space, `mincore`-measured page residency, issued
`madvise(MADV_DONTNEED)` advice on mapped eviction, and charged process
footprint are separate numbers, never summed; eviction counts are
cache-entry evidence, not RAM release. (4) **File identity:** the executor
opens the artifact once — content verification, the length check, and the
weight mapping use the same file object; replacement and truncation are
tested (loader unit tests and isolated-worker e2e). (5) **Records:**
benchmarks report end-to-end and decode-only throughput separately (record
schema v2) and name the resolved kernel/storage per sample; protocol v5
carries resolved kernels on `Loaded`. (6) **Matched smoke matrix** (real
Qwen3-0.6B Q8_0, 64/64, single-run, cache-uncontrolled; e2e/decode tok/s):
scalar+owned 1.24/2.65, scalar+mapped 1.57/3.24, neon+owned 2.69/5.70,
neon+mapped 4.19/9.18; footprint 1.10 GiB → 512 MiB with mapped storage;
llama.cpp CPU 35.2/89.6 and Metal 61.4/88.4 speed baselines on the same
artifact. Both selections PASS `verify --reference` with identical
statistics (max |Δ| 4.624e-1 @ registered 5e-1, top-1 100%) — the
`llama-cpu` int32 block dot is bit-identical across scalar and NEON. (7)
**Live bug fixed:** a governor `ControlAck` landing mid-generation aborted
the run as an "unexpected frame" under memory pressure; the generate and
stats read loops now absorb it (regression-tested in the isolated-worker
round trip). (8) **Toolchain:** `rust-toolchain.toml` pins 1.96.0 and CI
runs the same pin (floating stable failed run 38910541208 on lints not
reproducible locally); constant-size chunk iteration uses `as_chunks`.

## Context and review

The v0.2 engineering build finished the orchestration stack with exactly one
executable engine: the deterministic **reference test engine** (synthetic
output, no weights). Direction from product review: keep llama.cpp/ggml as the
*reference and compatibility backend*, and build a custom, narrowly-scoped
streaming executor — **TuckCore** — behind an experimental backend switch,
rather than adopting either Flash-MoE's or Splash's architecture wholesale
(both demonstrate on hardware far above the 8 GB cohort; Flash-MoE's
demonstration also changes routing and reports quality regressions in its
2-bit path, which a same-profile runtime must not inherit).

Inspection of the current implementation before restructuring:

| Completed component | Coupling to execution |
| --- | --- |
| `crates/manifest` — schemas, inspection, adapters, errors, identities | none (pure data) |
| `crates/planner` — admission, KV math, cost model, calibration | none (produces `ExecutionPlan` only) |
| `crates/store` — imports, packing (per-tile index + checksums), cache | none |
| `crates/supervisor` — queue, ledger, governor, API, bench, worker protocol | **one dispatch point**: `runtime.rs` `match self.engine_mode`, and `worker_host.rs` hardcodes `ReferenceTestEngine` |
| `crates/cli` | none (drives the supervisor) |

Single coupling point confirmed: the engine dispatch is two match arms and a
hardcoded struct. The worker protocol has no artifact reference (workers could
not read weights even if an executor existed) and no scoring frames.

## Decision

1. **Narrow, versioned backend interface.** New crate `tuckmill-backend`
   defines `InferenceBackend` (API version 1): `identity`, `capabilities`,
   `load(plan, ArtifactRef, options)`, `generate`, `score` (teacher-forced
   bounded logits), `stats`, `shutdown`, plus the shared `RunState`,
   `FinishReason`, `GenControl`, `GenOutcome` types. `tuckmill-manifest`
   remains dependency-free of execution concerns; supervisor and backends
   depend on `tuckmill-backend`.
2. **Backends.** `ReferenceTest` (existing synthetic engine, unchanged
   semantics) and `TuckCore` (new crate `tuckmill-tuckcore`). llama.cpp will
   arrive as a third backend (`Reference`, `EngineAvailability::Certified`
   candidate) through the same interface — not through new coupling.
3. **TuckCore starts narrow and honest.** Dense Qwen3 only (moe/hybrid
   architectures are refused with `UNSUPPORTED_ARCHITECTURE` and an action).
   Stage 0 implements, with bounded allocations:
   - a block loader reading tensor row-chunks via positioned reads, with an
     LRU byte-budget cache, pinning, and forced-eviction support;
   - explicit buffer ownership (blocks are `Arc`-owned; eviction only drops
     cache references) plus fence-based release through a completion tracker
     (leases survive eviction until the fence completes);
   - a scalar f32 forward pass (RMSNorm, NeoX RoPE, GQA attention with Qwen3
     per-head q/k norms, SwiGLU; dequantizers for F32/F16/BF16/Q8_0/Q4_0/Q4_1)
     that is the numerical oracle for later Metal kernels;
   - greedy/knobbed sampling with tokenizer strings from the artifact.
   Custom Metal kernels are implemented only after profiling identifies
   measurable gaps (no unmeasured rewrites).
4. **Protocol v2.** `WORKER_PROTOCOL_VERSION` becomes 2: `Load` carries an
   `ArtifactRef` (path + identity hashes the worker must verify before
   executing) and optional backend options; new `Score`/`ScoreChunk`/
   `ScoreDone` frames provide bounded teacher-forced logits; `StatsAck`
   carries backend statistics. Plans recorded against v1 are `PLAN_STALE`.
5. **Explicit experimental switch.** `--engine tuckcore` (run/serve/bench) is
   the acceptance for real-but-uncertified execution; `reference-test` keeps
   requiring `--allow-test-engine`. Responses mark
   `certified:false, backend:"tuckcore", synthetic_output:false`; the
   synthetic flag is computed from the backend identity, never hardcoded.
6. **Verification before promotion.** `tuckmill verify` runs identical token
   IDs through resident and forced-streaming TuckCore configurations with the
   same artifact (identity-verified), compares teacher-forced logits under a
   registered tolerance, and records eviction/footprint evidence, plus
   generated-token equality. Cancellation and forced eviction are unit-tested
   inside TuckCore. Promotion criteria remain the PRD gates (G1 numerics,
   G2-small), and FR-06/native routing stay mandatory for any MoE work.
7. **Deviations recorded.** (a) The supervisor keeps its std-only bounded
   HTTP subset; adopting tokio/axum now would be an unmeasured rewrite of a
   tested surface and is deferred until it earns its place (consistent with
   this repository's own principles). (b) No second general-purpose engine is
   created: TuckCore must earn its place by capability (8 GB streaming) or a
   measured advantage, or be retired.

## Consequences

- The dispatch point stays two match arms; every future engine slots behind
  the same trait and the same frames.
- Same-profile guarantees bind TuckCore exactly as they bind every engine:
  identical artifact, identical token IDs, no routing/precision/context
  changes; numerical equivalence is demonstrated, not assumed.
- `EngineAvailability` continues to describe kernel-gate state; backend
  choice is a runtime concern and is labeled per response.
