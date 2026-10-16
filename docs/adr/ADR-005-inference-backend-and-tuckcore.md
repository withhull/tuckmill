# ADR-005: Inference-backend interface and TuckCore

Status: **committed** (2026-10-16). Runtime + Systems. Amends ADR-003 (the
"one engine exists" statement) and prepares the llama.cpp integration planned
by ADR-002.

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
