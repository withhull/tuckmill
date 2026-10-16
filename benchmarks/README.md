# Benchmark protocol (standard-v1)

Frozen workload definition: [`standard-v1.json`](standard-v1.json).

## Rules

- **Distinct labels.** `cold-declared`, `warm`, `prefix-hit`, or
  `cache-uncontrolled`. A first run is not automatically a cold-storage run;
  when coldness cannot be established it is disclosed, and privileged
  cache-purging commands are never required in normal operation.
- **Frozen prompts.** This build derives prompt token IDs deterministically
  from frozen prompt identifiers (`frozen-synthetic-ids`). No private user
  prompts are collected. Calibration prompts are separate from held-out
  qualification prompts.
- **Draft honesty.** Speculative draft tokens are never reported as final
  throughput; only committed tokens count. (This build has no speculation.)
- **Recorded identity.** Records carry model/tokenizer hashes, quantization,
  backend and build versions, OS, chip, storage class, power state, context,
  sampler, lengths, and cache state.
- **Caveats travel with the report.** Every `BenchmarkRun` lists its engine
  identity, certification state, and honest caveats; a report without its
  caveats is incomplete evidence.

## This build's output

`tuckmill bench <model> --allow-test-engine` measures the *pipeline* through
the reference test engine (synthetic output). Records show
`engine: reference-test`, `certified: false`, and a caveat stating that no
model weights were executed. These records calibrate nothing about model
quality or speed and can never promote a plan's confidence label.

Real measurements require a certified executor and the four-way comparison
of PRD §20.1 (R0 source baseline, R1 compressed reference, T0 Tuckmill
reference kernels, T1 Tuckmill custom kernels) with randomized/ABBA ordering
and matched no-inference foreground controls.

## Comparator program (ADR-006)

Tuckmill is measured against **llama.cpp**, **MLX**, and **Splash** across
memory tiers — not only validated on the 8 GiB floor. A comparison counts as
evidence only when:

- the same artifact and the same token IDs are used (identity-verified),
- declared numerical profiles and cache states match,
- tolerances are registered before the run (`tuckmill verify --reference` is
  the harness; the pinned llama.cpp backend is integrated and the first
  numerical profile is recorded),
- ordering is randomized/ABBA with matched foreground controls, and
- the record states the tier, chip, OS, and build for every engine.

No comparative performance or quality claim exists until its measurement is
in `docs/claims.md`.

## Fixture smoke baselines (not claims)

Recorded on the development machine (MacBook Air M3, 8 GiB, macOS 27.0.1),
release build, standard-v1 shape 64 in / 32 out, 3 repetitions, synthetic
fixture weights, context 2048:

| Engine | ttft p50 | throughput (median) | footprint peak |
| --- | --- | --- | --- |
| TuckCore (scalar f32 oracle) | 72 ms | 314 tok/s | 7.2 MiB |
| llama.cpp b11695 (CPU via tm shim) | 1.6 ms | 6412 tok/s | 86.8 MiB |

## Real-model Stage A measurements (not claims)

Qwen3-0.6B Q8_0, M3 8 GiB, 64 in / 64 out, warm, release build, single run;
cache state uncontrolled; NOT a controlled comparison:

| Engine | tps | ttft | footprint peak | gap p95 |
| --- | --- | --- | --- | --- |
| TuckCore (K01 NEON + zero-copy, scalar oracle retained) | 5.7 (decode-only ~12.6) | 6.4 s | 512 MiB | 79 ms |
| llama.cpp b11695 CPU | 37.0 | 1.0 s | 970 MiB | 12 ms |
| llama.cpp b11695 Metal (gpu_layers 99) | 52.3 | 0.55 s | 369 MiB | 12 ms |

TuckCore optimization trail on the same shapes: pre-K01 0.3 tps → K01 fused
kernels 1.5 tps → + zero-copy mapped blocks 2.4 tps (16/8) and 5.7 tps
(64/64) — an 8-19× improvement, footprint 1113 → 512 MiB. Measured
attribution after the work: weight fetch 0.49 s → ~0 ms (mmap), matvec
compute 3.43 s → 0.39 s; the remaining decode cost is scalar compute (~78 ms/
token) and prefill is unbatched. Those two named bottlenecks — K02
(microbatched prefill) and further K01 lane work or a Metal kernel — are the
measured next targets. Metal is the performance reference; CPU remains the
arithmetic diagnostic oracle.

Logical read totals in records count every cache-miss block load, including
zero-copy mapped views (the pre-K01 run reported ~13.9 GiB of logical reads
for the 604 MiB model with a 32 MiB cache: one full weight sweep per forward
pass). That is logical read traffic, not physical SSD traffic — the page
cache serves most of it; raising the cache allowance reduces sweeps at the
cost of resident memory.

These exercise the harness and measured the kernel gap that named K01's first
increment; they are single-run smoke numbers, not controlled comparisons (no
ABBA, no warm/cold controls, no foreground matching). Fixture numbers say
nothing about real-model behavior (untrained tiny weights). Repetition 1
includes worker spawn and model load, which shows up in the p95 values;
`serve` keeps workers warm. Further kernel work — K02 onward — is accepted
only against measured gaps on real models and the numerical gates, never
against these numbers.
