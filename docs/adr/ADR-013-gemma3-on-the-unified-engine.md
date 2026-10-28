# ADR-013: Gemma 3 on the unified engine

Status: accepted (2026-10-28)

## Context

The unified engine (ADR-011) served only the Qwen families. Gemma 3 1B
(pinned `ggml-org/gemma-3-1b-it-GGUF`, `gemma-3-1b-it-Q8_0.gguf`, SHA-256
`b205840c…10d5`) is a current small family whose blocks recur in other
models. It is the first non-Qwen family on the engine.

## Decision

Extend the shared graph, not a per-family forward pass:

- `AttentionSpec.window`: sliding-window attention (a query attends the
  last `window` positions). One `window` argument on the chunk attention
  kernels, which decode and prefill share; 0 is full causal attention.
  The loop bounds are the only change, so Qwen results are unchanged.
- `LayerSpec.mixer_out_norm` / `ffn_out_norm`: RMS norms on each block's
  output before the residual add (Gemma's post-attention and post-FFN
  norms).
- `FfnSpec::GeGlu`: `down(gelu_tanh(gate x) * up x)` with ggml's tanh
  GELU (`tm_gelu_tanh_mul_f32`).
- `ModelGraph.embedding_scale`: Gemma's `sqrt(hidden)` on the embedded rows.
- `adapters::gemma3`: a global layer every sixth, otherwise a 512-token
  window; RoPE base 10,000 on window layers and the artifact's base on
  global ones; tied head. It refuses RoPE scaling, logit soft-capping and
  other layer patterns rather than approximating them.
- SentencePiece decoding in `DecodeVocab` (llama.cpp's piece rules).
  Encoding uses the pinned llama.cpp tokenizer.

Every layer keeps full-context KV rows; window layers read only their
window. Capping window-layer KV is a later memory optimization.

## Consequences

- Reference for correctness: pinned llama.cpp on Metal. Its CPU path
  quantizes activations to Q8_0, which Gemma's large activations amplify:
  llama.cpp's CPU and Metal paths differ by up to 0.82 in logits on the
  same tokens, so the CPU comparison is recorded, not gated.
- No Gemma-specific tuning (one engine; per-family optimizations are out
  of scope).

## Record (2026-10-28)

- **Fixture gate** (`gemma3_tests`, tiny Gemma 3, F16, window 8): graph
  shape; ledger == plan; batched prefill bit-identical to token-by-token
  decode and chunk cores to per-token cores at 1-132 tokens across the
  window; the window changes nothing below its size and changes longer
  prompts; the fixture responds to both an old and a recent token.
- **Pinned 1B vs llama.cpp Metal** (`gemma3_engine_logits_match_reference_when_available`,
  real text with BOS, f16 KV, teacher-forced): top-1 equal and max |Δ|
  0.006-0.022 (mean ≤ 0.0055) at 2, 9, 40, 200, 511, 513 and 688 tokens.
  Decoding across the window from 500 to 688 gives max 0.014, mean 0.0024.
  Against llama.cpp CPU: max 0.31-0.82, top-1 equal except at 2 tokens.
- **Speed** (`scripts/bench-baseline.sh`, 512/256, ignore-EOS, f16 KV,
  three ABBA rounds, `benchmarks/results/8gb/20261028T121538Z-Apple-M3-gemma3-1b`):
  decode 57.06 vs 63.68 tok/s (paired 0.896), warm TTFT 368 vs 331 ms
  (paired 1.11×, prefill ~90%). Locally-calibrated, one Mac.
