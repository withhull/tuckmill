# Repeat at d95ecc1 (idle machine): Phase 1 result confirmed on the ~1B row

- Command: `scripts/bench-baseline.sh artifacts-local/Qwen3.5-0.8B-Q8_0.gguf`
  (ROUNDS=3), foreground. Started after a macOS background daemon
  (`duetexpertd`, 99% CPU) had finished; then idle (no thermal warning, no
  swap growth). Same engine code as the first run (d95ecc1; the tree adds
  records and docs only).
- MacBook Air M3 8 GB; 512 in / 256 out, 3 reps, 3 ABBA rounds; all 6
  runs valid on the first attempt.
- Locally-calibrated results (not claims):
  - TuckCore: warm decode 84.24 tok/s (83.29-84.31), warm TTFT 0.272 s
    (0.272-0.273 s).
  - llama.cpp b11695 Metal: 67.39 tok/s (66.75-68.10), TTFT 0.288 s
    (0.285-0.300 s); llama-bench pp512 1985.8, tg256 86.8.
  - Paired decode 1.250 (1.223, 1.263, 1.250); paired TTFT 0.94x
    (0.910, 0.954, 0.944).
- Across the two runs at d95ecc1:
  - TuckCore's TTFT reproduced exactly (0.272 s; 0.272-0.273 in all six
    rounds).
  - Prefill is 94-106% of llama.cpp per pair; the paired medians are 99%
    and 106%. The spread is llama.cpp's own run-to-run variation
    (0.269-0.300 s).
  - Decode paired medians 1.111 and 1.250 (0.99-1.26 per pair).
- Phase 1 bar (>= 90% of llama.cpp decode and prefill): met in both runs,
  in every pair. Recorded as prefill parity and decode at parity or
  ahead. Not stated as a prefill lead, because llama.cpp's spread covers
  the difference.
