# Qwen3.5-0.8B Q8_0 at 45b8386 (32-token matrix-unit GEMM tile)

- Command: `scripts/bench-baseline.sh artifacts-local/Qwen3.5-0.8B-Q8_0.gguf`
  (ROUNDS=3), foreground, idle Mac (only Claude and WindowServer active,
  no thermal warning, swap 1.23 GiB pre-existing, no growth in any run).
- MacBook Air M3 8 GB; artifact sha256 in `environment.json`; 512 in /
  256 out, 3 reps, 3 ABBA rounds; all 6 runs valid on the first attempt.
- Locally-calibrated results (not claims):
  - TuckCore: warm decode 68.48 tok/s (68.09-68.49), warm TTFT 0.490 s
    (all three rounds).
  - llama.cpp b11695 Metal: 66.62 tok/s (66.26-67.97), TTFT 0.288 s;
    llama-bench pp512 1988.0, tg256 86.8.
  - Paired decode 1.028 (1.002, 1.028, 1.033); paired TTFT 1.70x.
- Vs the previous idle run at 6b6a014 (`20261026T165640Z-...`): TTFT
  0.588 -> 0.490 s (ratio 2.12x -> 1.70x); decode unchanged (68.47 ->
  68.48). Matches the in-process A/B (512-token prefill 625 -> 517 ms).
- Decode is recorded as "about parity" (llama.cpp's own decode varied
  66-78 tok/s across sessions), never as a lead. Prefill still misses the
  Phase 1 bar (needs <= ~0.32 s at 90%).
- Cold-load TTFT 2.6 s vs llama.cpp 0.6 s (worker spawn + load +
  first prefill): a separate gap, not yet profiled.
