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
- tolerances are registered before the run (`tuckmill verify` is the harness;
  reference-vs-custom logits arrive when the pinned llama.cpp backend lands),
- ordering is randomized/ABBA with matched foreground controls, and
- the record states the tier, chip, OS, and build for every engine.

No comparative performance or quality claim exists until its measurement is
in `docs/claims.md`.
