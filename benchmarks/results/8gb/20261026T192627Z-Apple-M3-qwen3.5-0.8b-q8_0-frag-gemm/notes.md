# Qwen3.5-0.8B Q8_0 at b5899a1 (fragment-layout matrix-unit GEMM)

- Command: `scripts/bench-baseline.sh artifacts-local/Qwen3.5-0.8B-Q8_0.gguf`
  (ROUNDS=3), foreground, idle Mac (Claude and WindowServer only; no
  thermal warning; swap 1.42 GiB pre-existing, no growth).
- MacBook Air M3 8 GB; 512 in / 256 out, 3 reps, 3 ABBA rounds; all 6
  runs valid on the first attempt.
- Locally-calibrated results (not claims):
  - TuckCore: warm decode 84.30 tok/s (84.13-84.33), warm TTFT 0.300 s
    (0.299-0.300 s).
  - llama.cpp b11695 Metal: 80.26 tok/s (67.13-84.38), TTFT 0.270 s
    (0.269-0.289 s); llama-bench pp512 1986.9, tg256 86.9.
  - Paired decode 1.048 (1.048, 0.999, 1.256); paired TTFT 1.11x
    (1.115, 1.107, 1.038).
- Vs the two runs at 171e53c: TuckCore TTFT 0.311 -> 0.300 s, which
  matches the in-process A/B (311 -> 297 ms). Prefill vs llama.cpp is 90%
  by the paired median (0.270 / 0.300), against 88-91% before. It is still
  at the Phase 1 bar, not clearly over it: llama.cpp's own TTFT spread
  (0.269-0.289 s) is as large as the margin.
- Decode stays at parity or ahead. llama.cpp's harness decode was in its
  fast mode for two of three rounds this time (80.3, 84.4 tok/s), which is
  where the pairs come out even (1.05, 1.00).
