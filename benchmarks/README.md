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

Qwen3-0.6B Q8_0, M3 8 GiB, 64 in / 64 out, release build, single run per row,
cache-uncontrolled; NOT a controlled comparison. Both throughput definitions
are reported: **end-to-end** (committed tokens / total request time, including
TTFT) and **decode-only** (tokens after the first / first-to-last token span).
Never compare a number across the two definitions.

| Engine / selection | e2e tok/s | decode-only tok/s | ttft | footprint peak | gap p95 |
| --- | --- | --- | --- | --- | --- |
| TuckCore, scalar kernels + owned storage | 1.24 | 2.65 | 27.6 s | 1.10 GiB | 397 ms |
| TuckCore, scalar kernels + mapped storage | 1.57 | 3.24 | 21.2 s | 513 MiB | 324 ms |
| TuckCore, neon kernels + owned storage | 2.69 | 5.70 | 12.7 s | 1.11 GiB | 183 ms |
| TuckCore, neon kernels + mapped storage | 4.19 | 9.18 | 8.4 s | 512 MiB | 114 ms |
| llama.cpp b11695 CPU (speed baseline) | 35.22 | 89.56 | 1.1 s | 969 MiB | 12 ms |
| llama.cpp b11695 Metal (gpu_layers 99, speed baseline) | 61.44 | 88.37 | 0.33 s | 370 MiB | 13 ms |

Matched readings (same artifact and workload; all TuckCore rows at the same
declared numerical profile, `oracle-f32`):

- **Vectorization** (scalar → neon), mapped storage: 1.57 → 4.19 e2e (≈2.7×),
  decode-only 3.24 → 9.18 (≈2.8×), ttft 21.2 → 8.4 s.
- **Copy avoidance** (owned → mapped): scalar 1.24 → 1.57 (≈1.3×), neon
  2.69 → 4.19 (≈1.6×); footprint 1.10 GiB → 513 MiB.
- The llama.cpp rows are speed baselines on the identical artifact; the
  *numerical* contract is exercised by `verify --reference`, never inferred
  from these rows.

Both selections pass the registered numerical envelope on this artifact:
`verify --reference --kernels scalar` and `--kernels neon` each PASS at
max |Δ| 4.624e-1 under the registered 5e-1 Q8_0 policy (median 3.372e-2,
top-1 agreement 100%, mean KL ~1.4e-3) with **identical statistics** — the
`llama-cpu` profile uses the exact int32 q8_0×q8_0 block dot, which the
scalar and NEON implementations compute identically.

Measurement corrections this increment made explicit (see `docs/claims.md`):

- **Throughput definitions.** The previous single-run row mixed end-to-end
  and decode-only figures ("5.7 tok/s (decode-only ~12.6)"); the harness now
  records and labels both per sample. That row is superseded by the matched
  matrix above; single-run values vary with cache state and system pressure
  (uncontrolled here).
- **Optimization trail across shapes.** The earlier trail (0.3 → 1.5 → 2.4
  at 16/8 → 5.7 at 64/64) mixed two workload shapes; the matrix above reruns
  every variant on the same 64/64 shape. Do not quote a single precise
  speedup across shapes.
- **Fetch time in mapped mode.** `matvec_fetch_seconds` in mapped mode
  measures view acquisition (metadata), not page servicing; first-touch
  page-fault work is attributed to `matvec_compute_seconds`. "Fetch ≈ 0 ms"
  in mapped mode does not mean weight access is free.
- **Eviction is not RAM release.** `cache_entry_evictions` counts dropped
  cache entries. Mapped-mode eviction issues `madvise(MADV_DONTNEED)`
  (`page_advice_bytes`; advisory on macOS) and observed residency is measured
  separately with `mincore` (`mapped_resident_bytes` in stats). The
  "512 MiB" figure is the charged process footprint (proc_pid_rusage):
  clean file-backed pages can be resident outside that figure, so it does
  not establish "the model uses ≤512 MiB of total RAM".

Logical read totals in records count every cache-miss block load, including
zero-copy mapped views (the pre-K01 run reported ~13.9 GiB of logical reads
for the 604 MiB model with a 32 MiB cache: one full weight sweep per forward
pass). That is logical read traffic, not physical SSD traffic — the page
cache serves most of it; raising the cache allowance reduces sweeps at the
cost of resident memory.

K01 coverage is precise: Q8_0/F16 CPU fused dot implemented (scalar and
NEON, selectable and recorded); Q4_0/Q4_1 stay on the scalar dequant path;
Metal kernels are unbuilt. These exercise the harness and measured the
kernel gap that named K01's first increment; they are single-run smoke
numbers, not controlled comparisons (no ABBA, no warm/cold controls, no
foreground matching). Fixture numbers say nothing about real-model behavior
(untrained tiny weights). Repetition 1 includes worker spawn and model load,
which shows up in the p95 values; `serve` keeps workers warm. Further kernel
work — K02 onward — is accepted only against measured gaps on real models
and the numerical gates, never against these numbers.
