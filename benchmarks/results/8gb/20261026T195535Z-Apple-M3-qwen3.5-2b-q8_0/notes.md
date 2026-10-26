# Qwen3.5-2B Q8_0 at d95ecc1 (first official 2B run)

- Command: `MEMORY_BUDGET=3GiB scripts/bench-baseline.sh artifacts-local/Qwen3.5-2B-Q8_0.gguf`
  (ROUNDS=3), foreground, idle Mac (no thermal warning, no swap growth).
  3 GiB is the smallest budget at which the planner admits the 2B
  resident (reservation 2.11 GiB, headroom 336 MiB). At the default
  2 GiB it plans streamed-dense, which the GPU path does not implement
  yet (Phase 2). Both engines ran under the same budget.
- MacBook Air M3 8 GB; artifact sha256 1b04acba…21f2c1 (unsloth/Qwen3.5-2B-GGUF
  at f6d5376); 512 in / 256 out, 3 reps, 3 ABBA rounds; all 6 runs valid
  on the first attempt.
- Locally-calibrated results (not claims):
  - TuckCore: warm decode 39.28 tok/s (38.65-39.34), warm TTFT 0.625 s
    (0.623-0.627 s).
  - llama.cpp b11695 Metal: 40.52 tok/s (37.96-40.53), TTFT 0.587 s
    (0.587-0.814 s; round 3 was slow); llama-bench pp512 880.2, tg256
    29.7 ± 2.8 (its own tokens, from an empty context; not comparable).
  - Paired decode 0.971 (0.971, 0.954, 1.035); paired TTFT 1.06x
    (1.061, 1.065, 0.770).
- Phase 1 bar (>= 90% of llama.cpp decode and prefill): met in this run
  by the paired medians (prefill 94%, decode 97%). In the two clean pairs
  prefill is 94% and decode 95-97%. One run; a repeat comes before
  certification.
- Memory: TuckCore footprint peak 2123 MiB. llama.cpp's 248 MiB
  footprint excludes its memory-mapped weights (llama-bench RSS 2.1 GB),
  so RAM is about equal.
- Cold-load TTFT 5.8-6.3 s vs llama.cpp 1.0-2.0 s (spawn + load + first
  prefill): unprofiled, as on the 0.8B.
- Prefill on the 2B is relatively further behind than on the 0.8B (94%
  vs ~99-106%). The 2B has wider matrices (hidden 2048), so the GEMM's
  remaining f16-vs-f32 gap (ADR-011) weighs more; not yet profiled on
  this model.
