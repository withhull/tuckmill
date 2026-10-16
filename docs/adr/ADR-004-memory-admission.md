# ADR-004: Memory admission and the governor

Status: **committed** (2026-10-16). Runtime + Mac systems.

## Context

Apple Silicon CPU and GPU share system memory; a process budget is not a hard
cap on all system memory attributable to inference. Framework allocations,
driver state, compression, file-backed pages, and the filesystem cache need
observation and headroom.

## Decision

1. **Admission rule.** `R + U + H <= B` with:
   - `R` = controlled live-allocation reservation, itemized by class
     (weights, kv_state, workspaces, io_buffers, expert_cache, draft_state);
   - `U` = estimated unavoidable overhead (policy default 192 MiB; an
     assumption to be calibrated at G0, recorded in every plan);
   - `H` = safety reserve (12.5% of budget, floor 192 MiB);
   - `B` = requested process budget.
2. **Reservations before async work; release after fences.** The live ledger
   (`crates/supervisor/src/admission.rs`) hands out RAII guards; guards are
   dropped only after the worker confirms completion. Buffers are never
   reclaimed to fake fast cancellation.
3. **Measured footprint is separate.** `proc_pid_rusage` physical footprint is
   sampled (supervisor and worker) and compared against reservation + overhead
   with a 25% overrun tolerance; overruns are logged as `FOOTPRINT_OVERRUN`.
   Reservations are never reported as measured RAM.
4. **Governor states.** Normal → Constrained → Paused → Draining → Failed.
   Warning pressure trims caches and stops prefetch growth; critical pressure
   pauses new compute/IO at the next safe boundary; resume requires the full
   stability interval (default 10 s) of improved conditions. Thermal
   `serious`/`critical` reduce submission or pause. Escalations are immediate;
   de-escalations respect a minimum dwell to prevent oscillation.
5. **Signals are read, not guessed.** `kern.memorystatus_vm_pressure_level`
   and `NSProcessInfo` thermal state are native; unavailable signals (Metal
   working set, battery) are recorded as `unavailable` in the device profile.

## Consequences

- Infeasible configurations are rejected before loading with exact numbers in
  the error (`INVALID_PLAN` / `CONTEXT_EXCEEDS_BUDGET`) and a safe next action.
- A 2 GiB starter budget on the 8 GB cohort follows the PRD's initial
  experiment; 3 GiB experiments require measured host headroom.
