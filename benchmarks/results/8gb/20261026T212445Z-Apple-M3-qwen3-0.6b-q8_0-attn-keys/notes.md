# Qwen3-0.6B Q8_0 at 4b7d2ae (attention scores without reductions)

- Command: `KV_PRECISION=f16 scripts/bench-baseline.sh artifacts-local/Qwen3-0.6B-Q8_0.gguf`
  (ROUNDS=3), foreground, idle Mac (no thermal warning, no swap growth).
  f16 KV declared for both engines; pinned artifact sha256 9465e63a…bb031;
  default 2 GiB budget (resident); 3 reps, 3 ABBA rounds; all 6 runs
  valid on the first attempt.
- **Valid: time to first token** (identical 512-token prompt):
  - TuckCore 0.254 s (0.252-0.255);
  - llama.cpp b11695 Metal 0.212 s (0.207-0.220);
  - paired 1.20x (1.159, 1.198, 1.217), i.e. prefill at 83% of llama.cpp.
- Vs the first run at 6486c7d (`20261026T204326Z-...`): TuckCore TTFT
  0.287 -> 0.254 s, matching the in-process A/B (299 -> 261 ms); the
  paired ratio went from 1.38x (72%) to 1.20x (83%).
- **Not valid: decode.** As before, the frozen standard-v1 tokens end in
  EOS after 1-2 tokens on both engines (generated_tokens 1-2 in every
  sample). The harness decode column rests on one interval per sample;
  llama.cpp's own value swung 34-100 tok/s across rounds for that reason.
  It is not recorded as a measurement.
- llama-bench row (own tokens, ignores EOS): pp512 2576.4, tg256 117.9.
- The Phase 1 prefill bar (>= 90%) is not met on dense Qwen3. The
  remaining difference is mostly GEMM operand precision (llama.cpp f16,
  TuckCore f32; ADR-011) plus the remaining attention cost.
