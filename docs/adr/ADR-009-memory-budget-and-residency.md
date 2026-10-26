# ADR-009: Measured memory budget and residency ranking

Status: **accepted** (2026-10-26). Supersedes ADR-006 decision 3's default
budget rule (a quarter of unified memory, floored at 2 GiB, capped at
24 GiB). Everything else in ADR-006 stands.

## Context

The quarter-of-RAM rule leaves 75% of a 192 GB Mac idle and still
over-commits a loaded 8 GB Mac: on the development M3 Air, ~2.4 GiB sit in
the compressor for other apps while the rule offers 2 GiB regardless.
Separately, `split_weights` is all-or-nothing (resident, or stream every
tile), so a model 10% over budget streams 100% of its layers. Phase 2
streaming needs a placement model that keeps the most valuable weights
resident and grows speed with RAM.

## Decision

1. **Default budget is measured, not proportional.**
   `B_default = memsize − wired − compressor_occupied − R_fg`, read from
   `host_statistics64` at plan time, where `R_fg = max(1 GiB, memsize / 16)`
   is the foreground reserve. No 1/4 rule, no 24 GiB cap. `--memory-budget`
   still overrides, and admission (`R + U + H ≤ B`, ADR-004) is unchanged.
   When a signal is unavailable, the plan says so and falls back to the
   ADR-006 rule, labeled `estimated`. It never guesses.
2. **Insufficient headroom is a typed refusal, never a silent floor.**
   If even the minimum plan (one layer or one expert set, plus KV and
   scratch) does not fit `B`, planning fails with the measured numbers and
   the next action. There is no "2 GiB anyway".
3. **The governor resizes between requests only.** It recomputes `B` from
   the same signals at request boundaries and may shrink or grow the
   resident set then. In-flight buffers are leased until their completion
   fence (AGENTS rule 4); nothing in flight is reclaimed to look fast.
4. **Residency is ranked by speed gained per resident byte.** The planner
   splits weights into units and fills `B` in rank order:
   - per-token full reads (output head, norms, every dense layer): equal
     value per byte. Dense layers rank equally, so the resident set is a
     contiguous prefix (the cheapest to schedule);
   - the token embedding is read one row per token: its value per byte is
     tiny, so it is row-streamed whenever anything has to stream;
   - MoE experts rank by measured routing frequency (Phase 3), shared
     tensors above all experts.
   `split_weights` becomes `rank_units`, which yields a resident set plus a
   streamed remainder. The all-or-nothing split is removed.
5. **Placement never changes arithmetic.** The same kernels and precision
   run on resident and streamed weights. Every budget in a sweep must give
   identical greedy output, proven with `tuckmill verify` (ADR-001
   same-profile rule).

## Consequences

- Phase 1 lands decision 1–3 (planner + governor + doctor output) with
  unit tests over recorded `host_statistics64` snapshots. Decisions 4–5
  land with GPU streaming in Phase 2, where their gate (monotonic tok/s
  across a budget sweep, identical greedy output) is measured.
- Plans and `doctor` print each budget input (memsize, wired, compressor,
  reserve) so a user can see why a model was admitted or refused.
- Tests of the old rule are replaced by the new formula; the ADR-006 rule
  survives only as the labeled fallback.
