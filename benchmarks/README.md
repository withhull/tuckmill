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

`tuckmill bench <model>` writes a standard-v1 record for the selected
backend: `--allow-test-engine` measures the *pipeline* only (synthetic
output, `engine: reference-test`, a caveat stating that no weights ran —
never evidence of model speed); `--engine tuckcore` and `--engine llama`
execute real weights and count the worker's model tokens (schema v4).
Every record carries `certified: false` and its caveats.

## Phase 0 baselines (`scripts/bench-baseline.sh`)

One command per artifact and machine:

```bash
scripts/bench-baseline.sh path/to/model.gguf
```

It runs TuckCore (`--kernels metal`) and the pinned llama.cpp
(`--gpu-layers 99`) through the same harness on the same frozen token ids,
alternating their order every round (ABBA), then the pinned `llama-bench`
tool on the same file (its own tokens; a vendor-tool row, never
identity-matched), then an offline MLX probe that only accepts the
identical artifact. Before every model process it checks reclaimable
memory and that no other model process runs; a run that grows swap beyond
`SWAP_DISCARD_MIB` is listed as discarded and excluded. The first
repetition of each invocation (worker spawn + model load) is reported as
cold-load and kept out of warm medians. Results land in
[`results/<tier>/`](results/) with the raw records; read them with
[`results/TEMPLATE.md`](results/TEMPLATE.md).

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

Qwen3-0.6B Q8_0, M3 8 GiB, 64 in / 64 out, release build, cache-uncontrolled;
NOT a controlled comparison. Both throughput definitions are reported:
**end-to-end** (committed tokens / total request time, including TTFT) and
**decode-only** (tokens after the first / first-to-last token span). Never
compare a number across the two definitions. Worker-side component timings
(`worker_total_ms` / `worker_prefill_ms` / `worker_decode_ms`) are recorded
per sample; supervisor time minus worker time is queueing and transport,
never compute.

Repeated session — one process, three warm repetitions per configuration,
same artifact and workload (medians; timing values are **provisional**, see
variance note below). Historical single-run values from the previous session
are kept alongside; neither set is a controlled comparison:

| Configuration | e2e tok/s (median of 3) | decode tok/s | ttft | footprint peak | historical single run (e2e/decode, ttft) |
| --- | --- | --- | --- | --- | --- |
| TuckCore, scalar + owned | 0.25 | 0.75 | 166.4 s | 1.24 GiB | 1.24 / 2.65, 27.6 s |
| TuckCore, scalar + mapped | 1.17 | 2.38 | 28.1 s | 513 MiB | 1.57 / 3.24, 21.2 s |
| TuckCore, neon + owned | 1.36 | 2.79 | 27.1 s | 1.26 GiB | 2.69 / 5.70, 12.7 s |
| TuckCore, neon + mapped | 3.96 | 7.87 | 8.2 s | 513 MiB | 4.19 / 9.18, 8.4 s |
| llama.cpp b11695 CPU (speed baseline) | 28.71 | 71.83 | 226 ms | 928 MiB | 35.22 / 89.56, 1.1 s |
| llama.cpp b11695 Metal (gpu_layers 99, speed baseline) | 55.97 | 82.47 | 52 ms | 370 MiB | 61.44 / 88.37, 0.33 s |

**Variance note (measured, not assumed).** The repeated session ran after
~30 minutes of sustained load on a fanless 8 GiB M3 Air with little free
memory (≈70 MiB free pages plus reclaimable inactive pages); the historical
single runs started fresher. Owned-storage rows degraded 2-6× between the
sessions; **mapped-mode neon reproduced within ~6%** (16.1-16.7 s totals,
8.9-11.0 s ttft across sessions and a later control run). That asymmetry is
itself the useful observation: heap-copied weights are sensitive to system
memory/thermal state, clean file-backed views are not. Do not certify exact
rates from either session; G3 defines the controlled conditions (ABBA,
warm/cold controls, foreground matching, thermal soak).

Matched readings (same artifact and workload; all TuckCore rows at the same
declared numerical profile, `oracle-f32`):

- **Vectorization** (scalar → neon), mapped storage: ≈2.7-3.4× e2e and
  decode across sessions; ttft 21.2 → 8.4 s (historical) and 28.1 → 8.2 s
  (repeated).
- **Copy avoidance** (owned → mapped): neon 2.69 → 4.19 (historical) and
  1.36 → 3.96 (repeated); footprint 1.10-1.26 GiB → 513 MiB.
- The llama.cpp rows are speed baselines on the identical artifact; the
  *numerical* contract is exercised by `verify --reference`, never inferred
  from these rows.

Numerical qualification after the Gate A corrections (schema v3 records:
requested and resolved execution are recorded separately, and a substitution
fails closed). All four forced-selection reference comparisons on this
artifact PASS under the registered 5e-1 Q8_0 policy:

- `--kernels neon --storage mapped`, f16 KV, held-out: max |Δ| 2.7259e-1
  (top-1 100%).
- `--kernels scalar --storage owned`, f16 KV, held-out: max |Δ| 2.7259e-1 —
  **bit-identical to the neon run**: the `llama-cpu` profile uses the exact
  int32 q8_0×q8_0 block dot, computed identically by both implementations
  (and the runs now demonstrably executed the requested selections).
- `--kv-precision f32`, held-out: max |Δ| 2.6768e-1 — f16 and f32 scoring
  are separately qualified; scoring uses the declared KV precision (the
  earlier f16-labeled records were produced before that path was unified and
  are preserved as historical, schema-v2 evidence only).
- Calibration run, scalar + owned, f16: max |Δ| 3.3386e-1.

Measurement corrections this line of work made explicit (see `docs/claims.md`):

- **Throughput definitions.** Earlier rows mixed end-to-end and decode-only
  figures ("5.7 tok/s (decode-only ~12.6)"); the harness records and labels
  both per sample.
- **Optimization trail across shapes.** The earlier trail (0.3 → 1.5 → 2.4
  at 16/8 → 5.7 at 64/64) mixed two workload shapes; do not quote a single
  precise speedup across shapes.
- **Fetch time in mapped mode.** `matvec_fetch_seconds` in mapped mode
  measures view acquisition (metadata), not page servicing; first-touch
  page-fault work is attributed to `matvec_compute_seconds`. "Fetch ≈ 0 ms"
  in mapped mode does not mean weight access is free.
- **Eviction is not RAM release.** `cache_entry_evictions` counts dropped
  cache entries. Mapped-mode eviction issues `madvise(MADV_DONTNEED)`
  (`page_advice_bytes`; advisory on macOS) and observed residency is measured
  separately with `mincore` (`mapped_resident_bytes` in stats). The
  "513 MiB" figure is the charged process footprint (proc_pid_rusage):
  clean file-backed pages can be resident outside that figure, so it does
  not establish "the model uses ≤513 MiB of total RAM".

Logical read totals in records count every cache-miss block load, including
zero-copy mapped views (the pre-K01 run reported ~13.9 GiB of logical reads
for the 604 MiB model with a 32 MiB cache: one full weight sweep per forward
pass). That is logical read traffic, not physical SSD traffic — the page
cache serves most of it; raising the cache allowance reduces sweeps at the
cost of resident memory.

K01 coverage is precise: Q8_0/F16 CPU fused dot implemented (scalar and
NEON, selectable and recorded); Q4_0/Q4_1 stay on the scalar dequant path;
the Metal kernels came later (Gate C, Q35-2) and are not covered by this
CPU table. These exercise the harness and measured the
kernel gap that named K01's first increment; they are single-run smoke
numbers, not controlled comparisons (no ABBA, no warm/cold controls, no
foreground matching). Fixture numbers say nothing about real-model behavior
(untrained tiny weights). Repetition 1 includes worker spawn and model load,
which shows up in the p95 values; `serve` keeps workers warm. Further kernel
work — K02 onward — is accepted only against measured gaps on real models
and the numerical gates, never against these numbers.
