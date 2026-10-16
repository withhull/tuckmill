# ADR-006: Positioning, memory tiers, and resident-first execution

Status: **committed** (2026-10-16). Product + Runtime. Amends the framing of
ADR-001/ADR-004 (8 GB was treated as *the* target; it is the *floor*).

## Context

The v0.2 engineering context revolved around an 8 GB machine (the PRD's
development cohort, and this repository's first physical test device). That
context leaked into framing: several documents read as if Tuckmill were an
inference engine specifically for 8 GB Macs. It is not — 8 GB is the minimum
supported target.

## Decision

1. **Positioning (canonical).**
   > llama.cpp is a general-purpose inference engine. Tuckmill is a Mac-native,
   > high-performance inference framework engineered to run large models as
   > efficiently as possible across Apple Silicon hardware, from 8 GB MacBook
   > Airs to high-memory MacBook Pros and Mac Studios.

   The goal is not simply to run models that do not fit in RAM. It is to make
   the best possible use of every Mac's compute, memory, and storage while
   preserving model quality.

2. **One execution engine, per-tier strategies — never separate products.**

   | Unified memory | Strategy |
   | --- | --- |
   | 8 GB | Compact weights, strict memory budgeting, selective streaming, foreground protection |
   | 16–24 GB | Hybrid resident/streamed execution, optimized quantized kernels, adaptive caching |
   | 32–48 GB | Larger resident models, specialized Metal kernels, efficient prefill, speculative decoding where beneficial |
   | 64–128 GB+ | Maximum resident execution, advanced batching, low-latency decoding, high-throughput inference |

   Strategies are chosen per request from measured device state; they are
   strategies, not guarantees that every model fits a tier.

3. **Resident-first. Streaming is a strategy, not the architecture.** When
   weights fit, they load once and stay resident. When memory is constrained,
   the planner decides which weights stay resident and which stream. The
   hardware-adaptive default process budget follows this: a quarter of unified
   memory, floored at 2 GiB and capped at 24 GiB, always overridable with
   `--memory-budget`.

4. **Five core engineering goals** (binding for every work package):
   1. Maximum performance — prefill, token generation, first-token latency,
      sustained throughput.
   2. Maximum memory efficiency — minimal allocations, weight movement, and
      context-state overhead.
   3. Broad model compatibility — dense, MoE, hybrid, compressed; through
      validated adapters.
   4. Hardware-adaptive execution — resident / partially streamed / fully
      storage-backed, chosen from actual device capabilities.
   5. Preserved quality — the selected model's computation and numerical
      profile; compressed models are evaluated separately from runtime
      changes.

5. **Benchmark mandate.** Tuckmill is measured against llama.cpp, MLX, and
   Splash across all memory tiers — not only "it works on 8 GB". Comparisons
   are valid only under the four-way identity rules (identical artifacts,
   token IDs, numerical profiles, declared cache state) with registered
   tolerances; no comparative performance or quality claim exists until the
   measurement is in the record (docs/claims.md).

6. **Research target.** Compatibility and performance certified per
   architecture, quantization, and hardware configuration, with a long-term
   target of models up to ~100B parameters. 8 GB remains the floor at which
   qualification begins, not the ceiling of ambition.

## Consequences

- README and docs present tiers, not a single cohort; the certification
  matrix is organized by memory tier with 8 GiB as the first qualifiable tier.
- The default process budget is device-derived (point 3) instead of a fixed
  2 GiB, so high-memory machines are not artificially capped.
- MoE/hybrid/compressed families remain gated milestones (ADR-005); nothing
  here weakens the same-profile rules.
