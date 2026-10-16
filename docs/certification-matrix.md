# Certification matrix

**Nothing in this build is certified.** This file records the honest state of
every claim-bearing capability. Certification attaches to a specific artifact,
adapter, precision profile, context range, backend, OS, and hardware cohort —
a parameter count alone certifies nothing.

Status vocabulary:
- **specified** — designed and recorded; no validation performed.
- **implemented** — code paths exist and pass the repository tests; no
  physical-device qualification.
- **unmeasured** — no evidence exists; no public claim may be made.

| Area | Status | Evidence in this build | What remains for certification (PRD gate) |
| --- | --- | --- | --- |
| Orchestration stack (CLI, planner, store, supervisor, API) | implemented | 236 repository tests; live M3 8 GiB (minimum tier) smoke run in `scripts/demo.sh` | G4 reliability campaign, G5 distribution |
| GGUF inspection & validation | implemented | synthetic + adversarial fixtures; checked overflow/bounds rejects | Fuzzing campaign against real artifact corpus |
| Safetensors inspection | implemented (header-level) | synthetic fixtures; offset/dtype validation | Real-corpus validation; adapter mapping |
| Planner admission math | implemented | unit tests incl. PRD worked example (2.5 GiB KV for 80×8×128 @8k) | Calibration at G0 (U, bandwidths) on physical devices |
| Governor state machine | implemented | unit tests (escalation, stability window, oscillation guard) | Pressure-injection soak on physical devices (G4) |
| Worker isolation & protocol | implemented | protocol unit tests; CLI e2e runs the isolated worker | Crash/fault-injection campaign (G4) |
| Bounded packer | implemented | lossless-sum tests, checksum corruption test, resume test | Real-artifact pack/verify at 100B storage scale |
| Resumable import | implemented | resume/restart/concurrent tests; atomic publish tests | External-volume loss and disk-full fault injection |
| Reference test engine | implemented | pipeline e2e (streaming, cancel, pause) | **Never certifiable as inference**; replaced by a real executor |
| TuckCore Stage 0 (dense Qwen3, scalar f32, bounded streaming) | implemented (experimental) | forward/dtype/loader/lease unit tests; state equivalence bit-exact (max\|Δ\|=0) with 856 forced evictions on M3 8 GiB; identity-verified loads; cancellation at boundaries | G1 numerics vs the pinned llama.cpp reference; real-weight qualification; tokenizer milestone |
| Equivalence harness (`tuckmill verify`) | implemented | resident vs forced-streaming comparison record per run (`records/verify/`) | extended to reference-vs-custom logits once llama.cpp backend lands; per-op tolerances registered before any kernel tuning |
| llama.cpp/ggml reference backend | not integrated | designated by ADR-005 as the compatibility backend behind `InferenceBackend` | integration, build pinning, ABI/versioning, G1 |
| TuckCore performance (tokens/s, sustained thermals) | unmeasured | one live smoke number on the fixture (29.7 tok/s, tiny model, debug build) — not a claim | G3 with real models; scalar path is an oracle, not an optimized executor |
| Custom kernels K01–K09 | specified | `kernels/registry.json`; plans list required kernels | Numerical gates per PRD §11.3, then G1 |
| Qwen3 dense execution | unmeasured | adapter specified; no executor | Stages A–C: 0.6B → 1.7B → 4B on physical 8 GB (G1/G2-small) |
| Qwen3 MoE streaming | unmeasured | adapter + expert geometry resolution implemented | Stage G; FR-06; native routing preserved |
| Qwen3.5 hybrid adapter | unmeasured | adapter specified; metadata namespace unverified against a real artifact | FR-29; K07 state tests |
| Bonsai 2 (PQ2_0/PTQ1_0) | unmeasured | adapter specified with transform contract + certification blockers | FR-29/30; K04–K07; vendor-compatible reference first |
| M1 / M2 minimum-tier cohorts | unmeasured | profiler reads them; nothing ran on them | Physical M1/M2 qualification (FR-27) |
| M3 minimum-tier cohort (this machine) | partially exercised | CLI/planner/store/supervisor smoke flows ran on macOS 27, 8 GiB | G2 with a real executor; foreground workload tests |
| Performance claims (any) | unmeasured | benchmark harness produces labeled records; comparator program defined (`benchmarks/README.md`) | G3 controlled experiments across memory tiers vs llama.cpp / MLX / Splash under four-way identity |

## Benchmark labels in use

`estimated` (cost model), `locally calibrated` (recorded measurements with
matching identity), `certified` (named tests on a named configuration).
The reference test engine's measurements are stored for pipeline
diagnostics only and can never promote a plan's confidence.

## Device tier record

Qualification is tier-based (ADR-006); 8 GiB is the first qualifiable tier,
never the ceiling of ambition.

| Tier | Physically exercised | Notes |
| --- | --- | --- |
| 8 GiB (floor): MacBook Air M3, macOS 27.0.1 (26A434) | yes (this build's development machine) | minimum-tier smoke flows; pressure: warning observed during development (host swapping); no large-model run performed |
| 8 GiB: MacBook Air M1/M2 | no | unsupported claim until FR-27 passes |
| 16–24 GiB | no | strategy designed (ADR-006); unmeasured |
| 32–48 GiB | no | Metal kernels + larger resident models pending |
| 64–128 GiB+ | no | maximum-resident and batching work pending |
| Intel MacBook Air | excluded | not a target |
