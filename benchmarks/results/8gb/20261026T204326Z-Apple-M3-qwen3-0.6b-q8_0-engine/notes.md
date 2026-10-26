# Qwen3-0.6B Q8_0 at 6486c7d (dense Qwen3 on the unified engine) — first run

- Command: `KV_PRECISION=f16 scripts/bench-baseline.sh artifacts-local/Qwen3-0.6B-Q8_0.gguf`
  (ROUNDS=3), foreground, idle Mac (no thermal warning, no swap growth).
  Both engines ran with declared f16 KV, the llama.cpp default. TuckCore's
  dense default is f32, so the precision is set explicitly for identity;
  the supervisor's acknowledgement check enforces it.
- MacBook Air M3 8 GB; pinned artifact Qwen/Qwen3-0.6B-GGUF, sha256
  9465e63a…bb031; default 2 GiB budget (resident); 512 in / 256 out
  requested, 3 reps, 3 ABBA rounds; all 6 runs valid on the first attempt.
- **Valid: time to first token** (identical 512-token prompt, both
  engines):
  - TuckCore 0.287 s (0.286-0.290);
  - llama.cpp b11695 Metal 0.208 s (0.207-0.209);
  - paired 1.38x (1.382, 1.388, 1.380), i.e. prefill at 72% of llama.cpp.
- **Not valid: decode.** On the frozen standard-v1 token ids, this
  instruct model emits end-of-sequence after 1-2 tokens on BOTH engines
  (generated_tokens 1-2, committed 0-1 per sample; the two engines agree).
  The harness "decode tok/s" (TuckCore 89.5, llama.cpp 99.9, paired 0.896)
  therefore rests on one inter-token interval per sample and is not a
  measurement; it is not recorded as one. A decode comparison needs a
  workload that generates the requested 256 tokens on both engines.
- llama-bench row (its own tokens, not identity-matched; it ignores EOS):
  pp512 2576.9, tg256 118.4.
- Footprint 969 MiB vs 373 MiB is not like-for-like (llama.cpp's
  memory-mapped weights are excluded from its footprint).
- Cold-load TTFT 1.8-1.9 s vs 0.45-0.52 s.
- Reading: dense Qwen3 prefill on the engine is at 72%, behind the 0.8B
  hybrid (94-106%). Every layer of this model is full attention (28
  layers, head_dim 128, group 2, f16 KV); the hybrid has 6 of 24. Not yet
  profiled on this model.
