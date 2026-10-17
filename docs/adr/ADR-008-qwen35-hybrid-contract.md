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

## Addendum (Q35-1A)

- **Attention-layer selection follows llama.cpp b11695 exactly.** A served
  layer is full attention iff `(i + 1) % full_attention_interval == 0` —
  full groups only, so the count is floor(served / interval); a trailing
  partial group adds no attention layer. An interval or per-layer array
  that leaves zero attention layers, an array shorter than the served
  stack, or an array with non-boolean entries refuses the artifact — no
  silent attention-layer split.
- **KV precision stays the declared-precision contract.** The Q35-1
  oracle stores attention KV at the declared width (packed f16 default,
  priced at 2 B/element; f32 priced at 4 B/element) and must prove by
  test that the admitted plan covers what it allocates. Recurrent and
  convolution state are f32 as declared by `mamba_ssm_dtype`.
- **Typed contract (Q35-1A).** `crates/tuckcore/src/qwen35.rs` resolves
  `Qwen35Config`, the exact layer map, the tensor map (every served
  tensor consumed exactly once with dimensions verified; unknown tensors
  inside the served stack refuse; MTP blocks excluded) and the state
  layout. The adapter's priced geometry and the typed layout are
  cross-checked against each other at resolve time. The loader does not
  call it yet: execution stays refused until the Q35-1 checklist passes.

## Addendum 2 (Q35-1A follow-up)

- **Per-layer arrays are authoritative everywhere.** The adapter already
  preferred a declared `attention.recurrent_layers` array over the
  interval; the typed resolver now does the same, so non-periodic maps
  are honored position-for-position and a valid array-only artifact
  without `full_attention_interval` resolves. `Qwen35Config` records the
  interval as optional and stores the resolved kind map that every
  consumer reads — there is no code path that silently re-derives kinds
  from the interval.
- **Value-head geometry derives from `ssm.time_step_rank`.** n_v_heads =
  time_step_rank, head_v = inner / time_step_rank (validated divisible);
  upstream `n_embd_s()` is `ssm_d_state * ssm_d_inner`, i.e.
  time_step_rank x head_v x state_size. The adapter's recurrent-state
  sizing is now stated with the same derivation (numerically identical to
  the previous group-based product, which only held because the pinned
  model has groups == time_step_rank). `ssm_norm` resolves against
  head_v, not inner/groups.

## Addendum 3 (Q35-1B)

- Scalar reference operators for the Gated DeltaNet layers live in
  `crates/tuckcore/src/qwen35_ops.rs`, with semantics taken from the
  pinned build: `ggml_ssm_conv` (windowed per-channel dot, no flip,
  `(kernel - 1)`-row history), `build_gdn_l2_norm` (`x / sqrt(sum(x^2) +
  eps)`, exactly `rms_norm(x, eps/n) / sqrt(n)`), the autoregressive
  delta rule (decay `exp(g)` applied to the state before the delta
  correction; `d = beta (v - s~^T k)`; `s = s~ + k d^T`; `o = s^T
  (q / sqrt(S_k))`), and ggml's `silu`/`softplus` (threshold at 20)
  scalars.
- Evidence: hand-computed first/second-token and 2-D cases; a transposed
  f64 algebraic cross-check over multiple steps; conv chunked-vs-single
  pass equivalence (state handoff exact); determinism; beta-0 /
  decay-1 identities; state sizes tie exactly to the contract's pricing
  (18 x (1,048,576 + 73,728) = 20,201,472 B). Still not wired into the
  loader.

## Addendum 4 (Q35-1C)

- Full-attention and FFN operators join the scalar reference library
  (`crates/tuckcore/src/qwen35_ops.rs`): weighted per-head RMS norm via
  the shared `rmsnorm_in_place` primitive; partial split-half RoPE that
  rotates only the declared span (`rope.dimension_count` = 64 of 256),
  with the frequency table spaced over the rotated dims exactly as
  `ggml_mrope_cache_init` does — for text positions every IMRoPE axis
  carries the same position, which reduces to this standard pairing;
  numerically stable softmax; grouped causal decode attention with
  consecutive-head KV mapping (`h / (heads_q / heads_kv)`); the joint
  Q-projection sigmoid gate applied before the output projection; SwiGLU
  and the residual adds.
- Evidence: hand-computed partial-RoPE rotation (rotated pairs and
  untouched tail), softmax limits, GQA hand case plus head-mapping case,
  gate/FFN/residual hand values, and the weighted head norm. Operator
  contracts are architecture-neutral where the mathematics is (norm,
  RoPE, softmax, attention, SwiGLU, residual), per the operator-first
  direction in the Q35 roadmap review.

## Addendum 5 (Q35-1D state contract)

- **Packed KV storage is the declared precision.** `crates/tuckcore/src/
  qwen35_kv.rs` stores per-attention-layer K/V planes at the declared
  element width (packed f16 default, f32 selectable) via the shared
  round-to-nearest-even converters. A cache allocates exactly
  `attention_layers x kv_heads x capacity x head_dim x element_bytes`
  per side; a test proves allocation equality against
  `Qwen35StateLayout::bytes_for` (fixture geometry, f16 and f32) and
  against the pinned artifact's published numbers (25,165,824 B of
  attention KV at ctx 2048 plus 20,201,472 B of recurrent/convolution
  state = 45,367,296 B). Writing at or beyond the admitted capacity
  refuses with a plan-typed error.
- **Attention over packed storage agrees with the f32 path** within
  half-precision tolerance (integration test through the cache); the
  f16 roundtrip pins the converter precision and f32 stores exactly.
- **Softmax is fail-closed** before it joins normal execution: empty
  rows are a no-op; a row without a finite maximum, a non-finite
  exponential, or a non-positive normalizer refuses with a typed error,
  and attention writes no output on refusal.

## Addendum 6 (Q35-1D execution and reference qualification)

- **Complete CPU reference forward pass.** `crates/tuckcore/src/
  qwen35_forward.rs` executes the served 24-layer hybrid stack (18 GDN +
  6 full attention; MTP excluded) over explicit token IDs: tied token
  embedding; per layer the pre-attention norm, the kind-specific
  attention path (GDN: QKV projection, GGUF-order causal convolution
  with the `(kernel - 1)`-row state, SiLU, per-head L2, sigmoid beta,
  `ssm_a * softplus(alpha + dt_bias)` decay, delta rule, gated
  normalization with `ssm_norm` and `silu(z)`, `ssm_out`; full
  attention: joint Q+gate projection, per-head QK norm, partial RoPE,
  packed-cache GQA, sigmoid output gate, `attn_output`), the
  post-attention norm, SwiGLU FFN and both residuals; final norm and
  tied-head logits. Weights are read from the artifact rows on demand;
  norms and other small vectors are cached at open. One layout fix rode
  along: `ssm_conv1d` is consumed in GGUF memory order (taps fastest,
  `weights[channel * kernel + tap]`), matching `ggml_ssm_conv`.
- **Evidence (machine-local, pinned artifact, release build).**
  Fixture: finite logits; chunked-versus-stepwise bitwise-equal logits
  and state checksum; reset clears every stateful byte; determinism.
  Pinned 0.8B: an 8-token teacher-forced pass with all 248,320 logits
  finite; against the pinned llama.cpp reference under identical token
  IDs and f16 KV — top-1 identical, max|diff| 0.325, mean|diff| 0.048
  at reference logit scale 20.6 (registered tolerances 0.5 / 0.1). The
  profile difference is declared: the oracle keeps f32 activations while
  llama.cpp quantizes activations to Q8_0 on the Q8_0 weight path.

## Addendum 7 (Q35-1E, session contract)

- **Verified-handle open.** `Qwen35Model::open_with_handle` builds the
  reference from the executor's already-digested file handle and
  inspection; the path-based `open` now inspects through the opened file
  itself. Weight reads and inspection always come from the exact bytes
  that were (or will be) verified — never a pathname reopen.
- **Session invalidation.** Any step that fails part-way marks the
  session invalid; subsequent forwards refuse until `reset()`. Partial
  recurrent/KV state is never reused (tested with a capacity-exhaustion
  interruption).
- **Cancellation polling.** `forward_tokens_with_poll` checks the
  cancellation callback before every served layer; a cancelled forward
  reports `Cancelled` and discards the session. Reset restores
  bitwise-identical output (tested).
- **KV storage contract hardened.** Writes are strictly sequential per
  attention layer with typed refusals for invalid layer indices and
  out-of-order tokens; reads refuse positions that were never written
  (allocated capacity and initialized state are distinct facts;
  checksum includes per-layer counts).
- **Preflight budget and exact pricing.** `required_bytes` computes the
  weights payload, packed KV, recurrent/conv state and a conservative
  scratch bound from inspected geometry before anything allocates; the
  model's actual allocations equal it (fixture and pinned numbers:
  25,165,824 B KV at ctx 2048 f16, 20,201,472 B state). The supervisor's
  KV width is now resolved by backend AND model family: dense TuckCore
  stays 4 B/element (f32 store), hybrid TuckCore prices the declared
  packed precision (2 default, 4 for f32), llama.cpp unchanged.

## Addendum 8 (Q35-2A: Gated DeltaNet Metal operators)

- Six isolated Metal kernels join the ABI and the kernel library:
  causal convolution step plus history advance (GGUF channel-major
  order preserved), per-head L2 norm, sigmoid-beta / softplus-decay
  gates, the f32 delta-rule recurrence with device-resident state, and
  the gated output normalization. Host dispatches follow the existing
  frame/group discipline; the Rust wrapper set mirrors the CPU operator
  signatures in `crates/tuckcore/src/qwen35_gpu.rs`.
- The recurrence is one thread per head (correct-first): per-head
  arithmetic order matches the scalar rule, and prefill batching waits
  until decode and state transitions are qualified.
- Registered tolerances (Metal vs scalar, fixture and pinned shapes,
  3-6 token sequences with state compared every step): 1e-4 absolute /
  2e-4 relative; observed conv 2.4e-7, L2 1.5e-8, gated norm 1.2e-7,
  delta rule 1.2e-7 (fixture) and 2.4e-4 output / 9.2e-5 state (pinned
  16x128). Device state stays f32.
