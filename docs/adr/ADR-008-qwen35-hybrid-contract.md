# ADR-008: Qwen3.5 (`qwen35`) hybrid contract and admission

Status: accepted (Q35-0)

## Context

Gate C4.0 (`e73196d`) is accepted; TuckCore is still a dense-Qwen3 executor.
The next target family is Qwen3.5-0.8B: a 3:1 hybrid of Gated DeltaNet and
full-attention layers. Its GGUF architecture id is `qwen35` (not `qwen3.5`),
and the adapter that shipped earlier was speculative — wrong namespace, an
assumed MoE expert layout, and conventional attention for every layer. It was
never accepted by any real artifact.

## Decision

1. **Target and pin.** The pinned artifact is ggml-org/Qwen3.5-0.8B-GGUF
   `Qwen3.5-0.8B-Q8_0.gguf`, 833,592,096 bytes, sha256
   `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
   fetched 2026-10-17 and verified against the published hash before use. The
   served text stack is `block_count` minus `nextn_predict_layers`
   (24 of 25 blocks); the MTP block is excluded until its own gate.
2. **Contract only for now.** `inspect`, `import` and `plan` work on the real
   artifact; execution remains refused with a precise message until the
   Q35-1 CPU numerical oracle and Q35-2 Metal gates land. Qwen3-0.6B stays
   frozen as the regression fixture for the kernel and admission machinery.
3. **Hybrid layering is declared, not guessed.** Adapter schema v2 adds
   `attention` (full-attention layer selection: an interval key, an optional
   trailing-block exclusion, and an optional llama.cpp-style per-layer
   `attention.recurrent_layers` boolean array that wins over the interval)
   and `recurrent` (delta-rule state plus convolution history geometry).
   Geometry resolution computes the attention layer count and a constant
   per-session `state_bytes`.
4. **Hybrid admission.** Attention KV is priced only on the attention layers;
   recurrent state is context-independent and reserved once. The formulas
   (re-derived from llama.cpp b11695 `src/models/qwen35.cpp` and registered
   here by revision rather than copied):
   - delta-rule state per recurrent layer: `groups x state_size x
     (inner_size / groups)` elements at 4 bytes (`mamba_ssm_dtype`
     float32);
   - convolution history per recurrent layer: `(conv_kernel - 1)` samples of
     `inner_size + 2 x groups x state_size` channels (the declared
     `ssm_conv1d` width) at 4 bytes.
   On the pinned 0.8B: 18 recurrent layers x 1,122,304 B = 20,201,472 B;
   attention KV = 6 layers x 2 KV heads x (256 + 256) x ctx x 2 B.
5. **Text path only.** The artifact is multimodal; vision/video and MTP stay
   out of scope. The built-in tokenizer does not recognize the `qwen35`
   pre-tokenizer yet and the multimodal chat template is unrecognized by
   design (byte-exact recognition only) — chat requests stay refused, and
   raw completion stops at the tokenizer contract, both with typed errors.

## Consequences

- The adapter is no longer speculative: `adapters/qwen3.5-hybrid.json`
  declares the verified `qwen35` namespace; portable contract tests use a
  synthetic hybrid fixture; the real-artifact gate is a machine-local e2e
  test (`TUCKMILL_QWEN35_GGUF`) that pins the store identity and size.
- Adapter schema v2 is a compatible extension of v1's field set, but the
  version check is strict: every builtin adapter now declares v2 (migration
  note: bump `schema_version`; no field changes for non-hybrid adapters).
- Execution types (a `Qwen35Config`, explicit linear/full attention layer
  types, recurrent state containers) land with the Q35-1 oracle instead of
  existing as unreachable code.
