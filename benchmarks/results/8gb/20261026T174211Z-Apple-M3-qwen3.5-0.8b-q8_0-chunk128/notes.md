# Qwen3.5-0.8B Q8_0 at 537419d (one GEMM launch per 128-token prefill chunk)

- Command: `scripts/bench-baseline.sh artifacts-local/Qwen3.5-0.8B-Q8_0.gguf`
  (ROUNDS=3), foreground, idle Mac (Claude, WindowServer and Spotlight
  only; no thermal warning; swap 1.27 GiB pre-existing, no growth).
- MacBook Air M3 8 GB; artifact sha256 in `environment.json`; 512 in /
  256 out, 3 reps, 3 ABBA rounds; all 6 runs valid on the first attempt.
- Locally-calibrated results (not claims):
  - TuckCore: warm decode 68.45 tok/s (68.19-68.49), warm TTFT 0.378 s
    (0.378-0.379 s in every round).
  - llama.cpp b11695 Metal: 68.31 tok/s (67.58-84.71), TTFT 0.283 s;
    llama-bench pp512 1986.3, tg256 86.8.
  - Paired decode 1.002 (0.805, 1.002, 1.013); paired TTFT 1.34x.
- Vs the idle run at 45b8386 (`20261026T171206Z-...`): TTFT 0.490 ->
  0.378 s (ratio 1.70x -> 1.34x); decode unchanged. Footprint peak 965 ->
  986 MiB: the larger prefill scratch (planned +17 MiB at capacity 640,
  more at this run's 2048 context).
- Round 1's llama.cpp decode (84.7 tok/s) is an outlier against its other
  rounds (68.3, 67.6) and every earlier session; it is kept as recorded,
  and the median is used. Decode is recorded as "about parity", never a
  lead.
- Prefill still misses the Phase 1 90% bar (needs <= ~0.31 s).
- Cold-load TTFT 2.5-2.7 s vs llama.cpp 0.6-0.7 s: still unprofiled.
