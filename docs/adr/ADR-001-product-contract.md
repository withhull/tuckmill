# ADR-001: Product contract

Status: **committed** (2026-10-16). Product + Runtime.

## Context

Tuckmill is a local-first inference framework for Apple Silicon Macs. The PRD
(v0.2) fixes a product contract that every layer must enforce, not merely
advertise. This ADR is the binding statement of that contract for this
repository.

## Decision

1. **Local only.** Inference never uses a remote provider. There is no remote
   fallback anywhere in the codebase. `serve` binds loopback only; downloads
   use approved HTTPS origins, and downloads are a separate permission from
   inference.
2. **Same-profile execution.** The runtime may change scheduling and cache
   placement — never the model: no silently dropped experts, pruned layers,
   truncated context, changed precision, substituted model, or requantization.
   A supported request either runs within a feasible plan, enters an explicitly
   disclosed slower mode, pauses safely, or fails with a typed reason.
3. **Admission rule `R + U + H <= B`.** A requested process budget `B`
   (`--memory-budget`, binary units) admits a request only when the controlled
   live-allocation reservation `R` plus estimated overhead `U` plus safety
   reserve `H` fits. This is a managed allocation policy, not an OS guarantee;
   measured process footprint is reported separately from reservations.
4. **Honest labels.** Every displayed configuration is **estimated**,
   **locally calibrated**, or **certified**. A projection is never presented
   as a measurement. Confidence upgrades only from recorded measurements that
   match device, artifact, adapter, engine, and ABI versions exactly.
5. **No synthetic output presented as model output.** This build ships a
   deterministic *reference test engine* that exercises the full pipeline with
   clearly-labeled synthetic tokens (`[ref:i]`). Using it requires the
   explicit `--allow-test-engine` acceptance; without it, non-certified
   profiles are refused with a typed error.
6. **Bounded artifacts and inputs.** Model metadata, templates, and requests
   are untrusted input: sizes and offsets are validated with checked
   arithmetic before allocation; frames and request bodies are capped.

## Consequences

- The CLI and API surface expose refusals, not workarounds: an unsupported
  operation is a compatibility failure, never permission to substitute a
  different architecture.
- Benchmarks and plans carry caveats; evidence is machine-readable.
- Any change weakening 1-6 requires a new ADR superseding this one.
