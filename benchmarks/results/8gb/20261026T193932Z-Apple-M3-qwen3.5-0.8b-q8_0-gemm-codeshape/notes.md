# Qwen3.5-0.8B Q8_0 at d95ecc1 (GEMM code shape after llama.cpp, f32)

- Command: `scripts/bench-baseline.sh artifacts-local/Qwen3.5-0.8B-Q8_0.gguf`
  (ROUNDS=3), foreground, idle Mac (Claude and WindowServer only; no
  thermal warning; swap 1.61 GiB pre-existing, no growth).
- MacBook Air M3 8 GB; 512 in / 256 out, 3 reps, 3 ABBA rounds; all 6
  runs valid on the first attempt.
- Locally-calibrated results (not claims):
  - TuckCore: warm decode 84.30 tok/s (84.12-84.39), warm TTFT 0.272 s
    (0.272 s in every round).
  - llama.cpp b11695 Metal: 75.87 tok/s (67.90-84.69), TTFT 0.270 s
    (0.269-0.287 s); llama-bench pp512 1986.6, tg256 86.7.
  - Paired decode 1.111 (0.993, 1.111, 1.243); paired TTFT 1.01x
    (1.011, 1.007, 0.948).
- Vs b5899a1 (`20261026T192627Z-...`): TuckCore TTFT 0.300 -> 0.272 s,
  which matches the in-process A/B (319-321 -> 286-292 ms). Prefill is
  99% of llama.cpp by the paired median, and >= 98.9% in every pair.
- Phase 1 bar (>= 90% of llama.cpp decode and prefill): met in this run
  with margin on both. Decode was at parity in the pair where llama.cpp
  ran in its fast mode (0.993), and ahead in the others.
- The prefill difference that remains is the f16-vs-f32 operand precision
  documented in ADR-011; TuckCore keeps the oracle-f32 profile.
- Footprint peak 972 MiB vs 241 MiB is not like-for-like (llama.cpp's
  memory-mapped weights are excluded from its footprint); RAM is about
  equal by RSS + footprint.
- Cold-load TTFT 2.4-2.8 s vs 0.6-0.7 s: still unprofiled.
