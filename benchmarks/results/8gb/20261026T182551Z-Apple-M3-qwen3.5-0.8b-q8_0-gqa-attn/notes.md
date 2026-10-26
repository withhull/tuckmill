# Qwen3.5-0.8B Q8_0 at 171e53c (lane-split DeltaNet recurrence + grouped-query attention)

- Command: `scripts/bench-baseline.sh artifacts-local/Qwen3.5-0.8B-Q8_0.gguf`
  (ROUNDS=3), foreground, idle Mac (Claude and WindowServer only; no
  thermal warning; swap 1.04 GiB pre-existing, no growth).
- MacBook Air M3 8 GB; artifact sha256 in `environment.json`; 512 in /
  256 out, 3 reps, 3 ABBA rounds; all 6 runs valid on the first attempt.
- Locally-calibrated results (not claims):
  - TuckCore: warm decode 84.28 tok/s (84.05-84.31), warm TTFT 0.311 s
    (every round).
  - llama.cpp b11695 Metal: 66.93 tok/s (65.80-67.23), TTFT 0.283 s;
    llama-bench pp512 1985.5, tg256 86.8.
  - Paired decode 1.259 (1.250, 1.281, 1.259); paired TTFT 1.10x.
- Vs the idle run at 537419d (`20261026T174211Z-...`): TTFT 0.378 ->
  0.311 s (1.34x -> 1.10x); decode 68.45 -> 84.28 tok/s. The decode gain
  is attention: this shape decodes at 512-767 positions, where the old
  per-head kernel walked the whole prefix serially per thread (0.34 ms
  per token at a short context and growing with it).
- Phase 1 bar (>= 90% of llama.cpp): prefill is at 91% in this run
  (0.283 / 0.311), borderline; decode is above it.
- Decode lead scope: measured only in this harness (same artifact, same
  frozen prompt tokens, 512-token context). llama-bench's tg256 (86.8)
  decodes from an empty context with its own tokens and is not
  comparable. Not generalized to other models, contexts or tiers.
- Footprint peak 972 MiB (vs 242 MiB for llama.cpp, whose memory-mapped
  weights the metric excludes; RAM is about equal by RSS + footprint).
- Cold-load TTFT 2.4-2.6 s vs llama.cpp 0.6-0.7 s: still unprofiled.
