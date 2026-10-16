# ADR-002: Workspace, repository layout, and build

Status: **committed** (2026-10-16). Platform.

## Context

The production runtime must not require Python, a cloud account, or Docker.
The PRD proposes a Rust orchestration layer, a C++/Objective-C++ execution
layer behind a narrow ABI, and custom Metal kernels.

## Decision

- A single Cargo workspace with five crates:
  - `crates/manifest` — validated schemas, typed errors, units, identities,
    GGUF/Safetensors inspection, adapter specs, synthetic fixtures.
  - `crates/planner` — device profiler, KV math, admission/budget math,
    deterministic cost model, calibration store, kernel registry.
  - `crates/store` — artifact layout, resumable chunked-digest imports,
    controlled downloads, bounded packer, cache/GC.
  - `crates/supervisor` — queue, admission ledger, governor, worker protocol
    and process lifecycle, local HTTP API, benchmark harness, privacy logs.
  - `crates/cli` — the `tuckmill` binary; `serve` and `run` use the isolated
    worker process by default.
- Dependencies are pinned to exact minor lines in `Cargo.toml`
  (`[workspace.dependencies]`) and locked by `Cargo.lock`, which is committed.
  Upstream constants that matter for correctness are pinned by revision:
  the GGML tensor-type table in `crates/manifest/src/ggml.rs` is transcribed
  from `ggml-org/llama.cpp@ceefd16daa2093730bf3340315fa2b84b286ddbc` and
  re-derived (not hand-edited) on update.
- `scripts/build.sh`, `scripts/test.sh`, `scripts/demo.sh` are the one-entry
  points. CI on Linux may validate portable logic only; Metal correctness and
  laptop usability require physical-Mac qualification and are *not* established
  by any automated test here.
- Data lives under `~/Library/Application Support/Tuckmill` (override with
  `TUCKMILL_DATA_DIR`). Installation is atomic: temporary work never appears
  as a valid artifact; the source file is never deleted automatically.

## Consequences

- `cargo build --workspace && cargo test --workspace` is the reproducible
  baseline; `scripts/test.sh` runs it with `clippy -D warnings` and `fmt`.
- The worker protocol is versioned (`WORKER_PROTOCOL_VERSION`) and the plan
  records it; incompatible caches are rejected as `PLAN_STALE`.
