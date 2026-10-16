# AGENTS.md — working instructions for this repository

## What this project is

Tuckmill is a Mac-native, high-performance local inference framework for
Apple Silicon, from 8 GB upward. Canonical positioning and the five
engineering goals live in `docs/adr/ADR-006-positioning-and-memory-tiers.md`;
the product contract (local-only, same-profile execution, honest labels) is
`docs/adr/ADR-001-product-contract.md`. 8 GB is the **minimum tier**, never
the framing of the product.

## Commands

```bash
scripts/build.sh     # cargo build --workspace + native ABI stub check
scripts/test.sh      # fmt check + clippy -D warnings + full test suite
scripts/demo.sh      # end-to-end demo on a generated fixture (temp data dir)

cargo test -p tuckmill-tuckcore          # single crate
TUCKMILL_DATA_DIR=$(mktemp -d) target/debug/tuckmill doctor
cargo run -p tuckmill-manifest --example gen_fixture -- /tmp/fixture.gguf
```

Always run `scripts/test.sh` before calling work done. It is the same gate
CI would run.

## Non-negotiable rules

1. **Honesty first.** Every displayed number is `estimated`,
   `locally-calibrated`, or `certified`; never present a projection as a
   measurement, synthetic output as model output, or an unrun benchmark as
   evidence. New claims go in `docs/claims.md`; capability states go in
   `docs/certification-matrix.md`.
2. **Same-profile execution.** Never change the model: no dropped experts,
   pruned layers, truncated context, precision substitution, requantization,
   or remote fallback — ever.
3. **Resident-first, tier-adaptive.** Execution mode comes from admission
   (`R + U + H <= B`); streaming is a strategy for constrained machines, not
   the architecture (ADR-006).
4. **Bounded everything.** Untrusted input (GGUF metadata, requests, frames)
   is validated with checked arithmetic before allocation. Buffers are
   leased through completion fences; never reclaim in-flight memory to fake
   fast cancellation.
5. **Backends behind the interface.** New execution work implements
   `tuckmill_backend::InferenceBackend` (ADR-005) or stays outside the
   runtime. Do not couple new engines to the supervisor directly.
6. **Comparisons are measured, not asserted.** Benchmark against
   llama.cpp / MLX / Splash only under the four-way identity rules
   (identical artifacts, token IDs, numerical profiles, declared cache
   state) with registered tolerances (`benchmarks/README.md`).
7. **No unmeasured rewrites.** Profile first; a custom kernel or replacement
   subsystem must demonstrate a gap or a capability the reference path lacks.
8. **Architecture decisions get an ADR** in `docs/adr/` before restructuring
   code. Keep ADRs short and decisive; supersede rather than rewrite.

## Repository conventions

- Workspace crates: `manifest` (schemas/inspection), `backend` (engine
  interface), `planner` (admission/cost), `store` (artifacts/packing),
  `tuckcore` (custom executor), `supervisor` (runtime/API/bench/verify),
  `cli`. Dependencies point one way; never create cycles.
- Typed errors only: `tuckmill_manifest::ErrorCode` (`crates/manifest/src/error.rs`),
  each with a safe next action. No `anyhow` in library crates.
- Units are exact: `Bytes` (GiB != GB). JSON/serde for all records; schema
  versions bump with migration notes.
- Upstream constants that matter are pinned by revision and re-derived, not
  hand-edited (see `crates/manifest/src/ggml.rs`).
- Tests: fixture-based (`tuckmill_manifest::testing` writes valid tiny GGUF
  files with finite values); adversarial cases belong next to the code they
  guard. Isolated-worker e2e coverage lives in `crates/cli/tests/e2e.rs`.
- Keep the README clean and current: short sections, real states, no
  aspirational marketing. Unfinished work is labeled ⛔ with its gate.

## Data and secrets

- Runtime data defaults to `~/Library/Application Support/Tuckmill`
  (`TUCKMILL_DATA_DIR` overrides; tests always use temp dirs).
- API tokens live in `control/api-token` (0600). Never log prompts,
  completions, or tokens; logs are structured, rotated, and capped.
- Never commit secrets, model weights, or generated store data.
