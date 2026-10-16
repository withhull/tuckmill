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
| Orchestration stack (CLI, planner, store, supervisor, API) | implemented | 336 repository tests; live M3 8 GiB (minimum tier) smoke run in `scripts/demo.sh` | G4 reliability campaign, G5 distribution |
| GGUF inspection & validation | implemented | synthetic + adversarial fixtures; checked overflow/bounds rejects | Fuzzing campaign against real artifact corpus |
| Safetensors inspection | implemented (header-level) | synthetic fixtures; offset/dtype validation | Real-corpus validation; adapter mapping |
| Planner admission math | implemented | unit tests incl. PRD worked example (2.5 GiB KV for 80×8×128 @8k) | Calibration at G0 (U, bandwidths) on physical devices |
| Governor state machine | implemented | unit tests (escalation, stability window, oscillation guard) | Pressure-injection soak on physical devices (G4) |
| Worker isolation & protocol | implemented | protocol unit tests; CLI e2e runs the isolated worker; v7 frames (typed sampling, visible/generated counts, KV acknowledgement) regression-tested | Crash/fault-injection campaign (G4) |
| Chat input contract (template + tokenizer) | implemented | `ModelBundle` recognizes the pinned Qwen3 template byte-exactly and refuses unknown templates; the renderer is byte-verified against 12 recorded reference conversations (multi-turn, Unicode, reasoning-split, newline-trimming, tool-marker, literal-special cases); token IDs reproduced exactly against the unchanged Qwen3-0.6B artifact via the pinned reference tokenizer (gated golden test); llama.cpp bundled-jinja divergence recorded, not copied | broader model family; tool-call subset deferred by design |
| Output correctness and usage accounting | implemented | bounded stop buffering (multi-token/overlapping/Unicode/partial-token unit tests); stop/EOS markers excluded from visible output; `committed` vs `generated` tracked separately; API tests assert streaming vs non-streaming equality and non-zero prompt tokens in usage; isolated-worker e2e asserts stop trimming and finish reasons | streaming-disconnect soak under load (G4) |
| Plan enforcement (KV representation and arenas) | implemented | planner prices KV at the resolved representation (TuckCore 4 B/element; 4-byte pricing flips admission where 2-byte pricing admitted, regression-tested); worker refuses to load understated KV or weight-cache reservations; supervisor verifies acknowledged `kv_state_bytes` vs the plan and fails closed | calibrated U/workspace derivation at G0; pressure soak (G4) |
| Warm model reuse | implemented | `LoadedModelKey` (identity + load options + prepared capacity) replaces plan-id comparison; unit tests prove time/seed/prompt insensitivity and capacity semantics; isolated-worker test: ten equivalent requests reuse one load with per-request isolation, larger context reloads, smaller reuses | multi-user/queueing policy beyond the single-worker model |
| Bounded packer | implemented | lossless-sum tests, checksum corruption test, resume test | Real-artifact pack/verify at 100B storage scale |
| Resumable import | implemented | resume/restart/concurrent tests; atomic publish tests | External-volume loss and disk-full fault injection |
| Reference test engine | implemented | pipeline e2e (streaming, cancel, pause) | **Never certifiable as inference**; replaced by a real executor |
| TuckCore Stage 0 (dense Qwen3, scalar reference + K01 CPU increment, selectable mapped/owned storage, bounded streaming) | implemented (experimental) | unit tests + fixture scheduling equivalence (bit-exact, 856 evictions) + Stage A: real Qwen3-0.6B execution, cross-decoder check against llama-quantize, cross-backend logits within registered envelopes, protocol v4 scoring + v5 resolved-kernel identity + v6 worker-side timing + v7 typed sampling / visible-vs-generated accounting / KV arena acknowledgement, heartbeat liveness; single-open artifact handle (verify + map + read), checked kernel boundary (malformed shapes refused), typed execution options shared by run/bench/stats/verify with post-load acknowledgement and fail-closed substitution checks; generation and scoring share one KV precision contract; load refuses understated plan reservations | instruct-quality evaluation; 1.7B/4B; remaining kernel work gated on measured gaps to llama.cpp |
| Equivalence harness (`tuckmill verify`) | implemented | resident vs forced-streaming comparison record per run (`records/verify/`) | extended to reference-vs-custom logits once llama.cpp backend lands; per-op tolerances registered before any kernel tuning |
| llama.cpp/ggml reference backend | integrated (wrapper uncertified) | pinned b11695, runtime-loaded via header-checked C shim; loads and generates on the fixture; CPU diagnostic path pins `n_gpu_layers = 0`; Metal offload (`--gpu-layers`, shim v4) is a labeled performance baseline, never used for arithmetic diagnostics | broader artifact corpus; Metal baseline qualification; wrapper certification |
| Tokenizer (byte-level BPE) vs llama.cpp | verified, reference-first | when the shim is configured, encoding IS llama.cpp (vocab-only load, CPU-pinned, golden-tested on Qwen3-0.6B incl. digits/contractions/unicode with exact byte roundtrip); built-in BPE exact on byte-level fixture goldens, refused for non-GPT-2 pre-tokenizers; incremental UTF-8 streaming decoder shared by both engines | frozen public corpus goldens; `qwen2` pre-tokenizer equivalence for the built-in path |
| TuckCore vs llama.cpp numerical profile | measured + held-out qualified (Qwen3-0.6B, M3 8 GiB, ctx 2048) | under policy `calibration-2026-10-16.1`: forced neon+mapped f16 held-out max 2.7259e-1; forced scalar+owned f16 held-out max 2.7259e-1 (bit-identical: exact int32 q8_0×q8_0 dot); f32-KV held-out max 2.6768e-1; scalar+owned calibration max 3.3386e-1 — all PASS at 5e-1 with top-1 100%. Scoring now executes the declared KV precision (schema-v3 records separate requested from resolved execution and fail closed on substitution; earlier f16-labeled schema-v2 records predate that unification and are historical only) | per-quantization policies for new formats require their own calibration; task-quality suites distinct; re-derive the envelope table under G3 controlled conditions |
| Verification integrity | implemented + adversarial-tested | malformed/truncated/duplicate/oversized/non-finite scoring is rejected before comparison; reports immutable with unique run IDs (reruns proven distinct in e2e); interrupted runs recorded without verdict; content-digest drift refused before execution (e2e) | fault-injected worker campaigns in CI |
| Governor–backend control contract | implemented + tested | protocol v4 Control/ControlAck; mid-generation trim evicts expendable caches without touching in-flight leases; heartbeats continuous during long compute; engine poll hooks service cancellation in single-step work; control paths lock-free vs generation (regression tests) | deliberate-pressure soak on physical hardware (G4) |
| TuckCore performance (tokens/s, sustained thermals) | smoke baselines measured (not claims) | repeated 64/64 matrix (median of 3, one process) + historical single runs, real Qwen3-0.6B Q8_0, release, cache/thermal-uncontrolled: neon+mapped reproduced within ~6% across sessions (≈4 e2e / 8-9 decode tok/s, ttft ≈8 s); owned-storage rows degraded 2-6× under sustained load and low free memory; footprint 513 MiB (mapped) vs 1.1-1.26 GiB (owned); worker-side prefill/decode split recorded per sample; llama.cpp CPU ≈29-35 / ≈72-90 and Metal ≈56-61 / ≈82-88 baselines | G3 controlled experiments (ABBA, warm/cold, foreground, thermal soak); exact rates are provisional; scalar path stays the oracle, not the optimized executor |
| Custom kernels K01–K09 | K01 first increment implemented; K02–K09 specified | K01: aarch64 NEON fused Q8_0/F16 packed dot (scalar fallback, explicitly selectable and recorded), validated against scalar references, malformed-shape refusal tests, and the llama.cpp envelopes under both selections (identical statistics); Q4_0/Q4_1 stay on the scalar dequant path; statuses in `kernels/registry.json` | Q4_0/Q4_1 and Metal paths; numerical gates per PRD §11.3 for each remaining package, then G1 |
| Weight storage modes and accounting | implemented + selectable | `owned` (copied bounded buffers) and `mapped` (zero-copy views of one artifact map) selectable and recorded; accounting separates owned bytes, mapped VM, `mincore` residency, and issued `madvise` advice — never summed; metric definitions in stats records | bounded-memory interpretation on oversized models is gated on the streaming qualification (G3); pressure soak for mapped-mode host interference |
| Qwen3 dense execution (Stage A) | partially exercised (real 0.6B) | Qwen3-0.6B (Q8_0, plus F16 twin) executes end-to-end through TuckCore; real-model completion correct; `verify --reference` records persisted | Stage A completion: 1.7B/4B, instruct-quality evaluation, G2 on other tiers |
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
