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

## Addendum 9 (Q35-2B.0-2B.2: full-attention kernels, shared owner, packed f16 KV)

- **Shared GPU owner.** `Qwen35Gpu` owns one Metal context for the
  whole model and allocates every buffer through it (single allocation
  ledger); its `*_dev` methods are buffer-to-buffer, so model
  activations stay on the device. `GdnGpu`'s slice methods are now
  explicitly test adapters over those device ops. All buffers are
  zeroed at allocation: no result may depend on unspecified initial GPU
  memory (regression included).
- **Full-attention kernels.** Joint Q+gate split, partial split-half
  RoPE over the declared span only (`n_rot` of `head_dim`; IMRoPE with
  text positions reduces to it), sigmoid attention gating, and causal
  GQA over a packed f16 cache; per-head Q/K RMS norm reuses the
  existing weighted-rmsnorm kernel. All qualified against the CPU
  oracle on fixture and pinned shapes.
- **Packed f16 GPU KV.** `tm_qwen35_kv_store_f16` stores true two-byte
  entries; a test proves the stored bytes are bit-identical to the CPU
  RNE packer, and the f16 attention path matches the CPU packed-cache
  reference token by token. Recurrent DeltaNet state stays f32. One
  dispatch bug was caught by the pinned shapes and fixed: a 128-thread
  group per head only covered half of the 256-wide rows.

## Addendum 10 (Q35-2B.3: complete device layers)

- **Complete served layers on the device.** `LayerGpu` assembles both
  layer kinds from the artifact: F16 matrices are uploaded raw and
  projected with the existing kernels, F32 vectors feed the norm/conv
  kernels, and every activation stays on the GPU between layers (one
  Metal context, one allocation ledger). New small kernels: silu in
  place and the conv-output q/k/v split. All layer state (packed f16
  KV, f32 delta-rule and convolution state) is device-resident.
- **Fixture full-model qualification.** A three-token run of the
  complete fixture model, verified layer by layer against the CPU
  oracle's activation trace (embedding row and every served layer's
  hidden state) plus the tied-head logits: worst hidden |diff| 5.4e-7,
  worst logits |diff| 1.4e-6; registered tolerances 1e-5 / 1e-4.
- The fixture's distinct GDN state size (8) and attention head dim (16)
  caught a dimension conflation in the first assembly — exactly the
  class of bug the pinned-vs-fixture shape pairs exist for.
- Scope: F16 matrices only; the Q8_0 matrix path and the real-artifact
  upload/preflight land with Q35-2C.

## Addendum 11 (Q35-2C: the complete 0.8B on the GPU)

- **Real-artifact device execution.** `Qwen35GpuModel` assembles the
  whole served stack on one shared owner: Q8_0 and F16 matrices are
  uploaded raw (no re-quantization) and projected with the existing
  kernels, the embedding lookup dequantizes one stored row directly on
  the device, and the tied head projects to full-vocabulary logits. All
  weights, activations and state stay on the GPU across the 24 layers.
- **Ledger honesty.** The device allocation is reported as a single
  number and asserted against the accounted budget (weights payload +
  packed KV + recurrent state + per-layer scratch). On the pinned
  artifact: 825,411,072 B held against 849,597,184 B accounted
  (weights 822,629,632 + state 20,398,080 + 234,624 B scratch per
  layer).
- **Qualification.** Pinned artifact, four tokens: GPU vs the CPU
  oracle — top-1 identical, worst logits |diff| 4.88e-4 (registered
  2e-3). Independently against the pinned llama.cpp reference in the
  same run: max|diff| 0.325343 with identical top-1 — statistically
  the same declared profile gap the CPU oracle shows (0.325393),
  confirming the GPU introduces no additional divergence. Fixture
  layer-by-layer trace comparison remains green (5.4e-7 / 1.4e-6).
- Scope: sequential per-token decode; prefill batching (Q35-2D) stays
  gated behind this qualification, and text/chat enablement still
  waits on Q35-1E.2 plus tokenizer/template support.

## Addendum 12 (Q35-2C.1: GPU integration hardening)

- **Preflight admission, before any allocation.** `gpu_alloc_plan`
  computes the exact device allocation (raw served weights + derived
  RoPE tables, packed KV + recurrent/convolution state, per-layer
  scratch + model activations) from the tensor source, and
  `Qwen35GpuModel::load` refuses an admission below the plan total
  with the component breakdown — no model buffer exists when the check
  fails (regression: the shared ledger does not move on refusal). After
  load, the plan total equals the ledger exactly (tested on the
  fixture; on the pinned artifact 825,411,072 B == weights 800,882,688
  + state 20,398,080 + scratch 4,130,304); the earlier 49,152-byte
  formula drift (one missing `heads x head_dim` scratch buffer per
  attention layer) was caught by exactly this equality.
- **Fail-closed GPU sessions.** Any failure inside a traced forward —
  including a verification-callback failure — invalidates the session;
  further forwards refuse typed until `reset()`, which restores
  bitwise-identical output (tests mirror the CPU oracle's lifecycle).
- **No large host staging.** Buffers are zeroed in place (shared
  storage) or via a GPU fill for private storage (new ABI
  `tm_metal_buffer_zero`); weights stream to the device in bounded
  4 MiB chunks from a lightweight `ArtifactTensorSource` that holds
  only the file handle, inspection and resolved contract — device
  clients no longer need a CPU inference session (tested).
- **Real-artifact trace qualification.** Beyond logits, the pinned
  model's hidden states now compare at layers [0, 3, 11, 23] and every
  position: worst |diff| 6.4e-5 (registered 5e-4); logits 4.88e-4.

## Addendum 13 (Q35-1E.2: supervised worker wiring)

- **Served path.** `TuckCore` declares `qwen35` and routes it, after the
  shared identity prologue (metadata/tensor-index hashes, full content
  digest through the one opened handle), to `Qwen35GpuModel` on the
  resident Metal path. The CPU forward stays the oracle: any other
  resolved kernel set is a typed refusal naming `--kernels metal`.
- **Options are honored or refused, never ignored.** f32 KV (the GPU
  stores packed f16 only), `prefill_batch` (batched prefill is Q35-2D),
  CPU weight-cache/block options, and non-resident plan modes are typed
  refusals before any GPU allocation.
- **Plan enforcement per class.** `gpu_alloc_plan` is checked against the
  plan's weights, kv_state and workspaces reservations separately; after
  load the device ledger must equal the preflight total. On the pinned
  artifact at context 2048 the plan's kv_state reservation (45,367,296 B)
  equals the preflighted state exactly.
- **Session state is held, so reuse re-checks it.** Unlike dense TuckCore
  (KV per request), the hybrid model allocates its KV and recurrent state
  once at load. The worker reports it as `retained_state_bytes`, and the
  supervisor's reuse gate now requires it to fit the current plan's KV
  reservation. Every request resets the session first: recurrent state
  never crosses requests (fixture test: identical repeated requests give
  identical output).
- **Scoring** replays from a reset session in order, so logits at a
  position always come from exactly the prefix (positions out of order
  included); fixture logits match the CPU oracle within the registered
  1e-4.
- **Text.** Raw completion runs with the pinned reference tokenizer (the
  `qwen35` pre-tokenizer is never approximated). An unrecognized declared
  chat template now refuses chat only; it previously refused raw
  completion too, contradicting its own next action.
- **Evidence** (pinned artifact, isolated worker, release): greedy
  output for "The capital of France is" equals the CPU oracle's greedy
  continuation exactly (" Paris.\nThe capital of France is", 8 tokens);
  the pinned llama.cpp Metal run produces the same 8 tokens.
- **Fixed on the way:** `ArtifactTensorSource::from_open_file` inspected
  through the handle's shared cursor, which the content-digest pass
  leaves at EOF; it now rewinds explicitly (regression test).
- Scope: sequential prefill (one token per step) and per-layer dispatch
  as qualified in Q35-2C. The measured baseline is recorded in
  `benchmarks/results/8gb/`; performance work is Q35-2D/Phase 1.
