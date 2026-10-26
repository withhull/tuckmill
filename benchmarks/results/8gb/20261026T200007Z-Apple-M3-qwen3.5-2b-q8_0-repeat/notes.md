# Repeat: Qwen3.5-2B Q8_0 at d95ecc1 (idle machine) — result confirmed

- Command: `MEMORY_BUDGET=3GiB scripts/bench-baseline.sh artifacts-local/Qwen3.5-2B-Q8_0.gguf`
  (ROUNDS=3), foreground, idle Mac (no thermal warning, no swap growth).
  Same engine code and budget as the first 2B run.
- MacBook Air M3 8 GB; artifact sha256 1b04acba…21f2c1; 512 in / 256
  out, 3 reps, 3 ABBA rounds; all 6 runs valid on the first attempt.
- Locally-calibrated results (not claims):
  - TuckCore: warm decode 39.39 tok/s (39.35-39.39), warm TTFT 0.623 s
    (0.622-0.624 s).
  - llama.cpp b11695 Metal: 38.25 tok/s (34.79-40.55), TTFT 0.588 s
    (0.587-0.851 s; round 3 slow again, as in the first run).
  - Paired decode 1.030 (1.030, 0.970, 1.132); paired TTFT 1.06x
    (1.061, 1.061, 0.731).
  - The llama-bench row was degraded this time (pp512 741 ± 83, tg256
    15.1 ± 2.0, against 880 / 29.7 in the first run). It is the vendor
    tool's own row (its own tokens, not identity-matched), not part of
    the paired comparison.
- Across the two 2B runs at d95ecc1:
  - TuckCore reproduced: TTFT 0.622-0.627 s, decode 38.65-39.39 tok/s.
  - In the clean pairs, prefill is 94% of llama.cpp and decode 95-103%.
  - Paired medians: prefill 94% in both runs; decode 0.971 and 1.030.
- Phase 1 bar (>= 90% decode and prefill): met in both runs. Recorded as
  prefill 94% and decode at parity, not a lead.
- llama.cpp's third-round slowdown appeared in both runs (TTFT
  0.81-0.85 s). It falls on the round after TuckCore's worker held
  ~2.1 GiB; the cause is not investigated, and the paired medians do not
  depend on it.
