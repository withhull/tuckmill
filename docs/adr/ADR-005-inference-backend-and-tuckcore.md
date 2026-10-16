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
