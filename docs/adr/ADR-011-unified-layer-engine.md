# ADR-011: One unified layer engine for every model family

Status: **accepted** (2026-10-26). Extends ADR-005 (TuckCore behind
`InferenceBackend`) and ADR-007 (Metal ABI). Supersedes the per-family
forward passes (`forward.rs`/`gpu.rs` for dense Qwen3, `qwen35_forward.rs`
/`qwen35_gpu.rs` for Qwen3.5) once their migration gates pass.

## Context

Each model family has its own hand-written forward pass, on CPU and on
Metal, about 8,000 lines across the two families. Every cross-cutting
improvement (one command buffer per step, batched prefill, attention
rewrites, matrix-unit GEMM) had to be built per family. They drifted: the
Qwen3.5 attention and prefill work never reached dense Qwen3. The target
list (Llama 3.x, Gemma, Mistral, DeepSeek, gpt-oss, MoE) makes that cost
multiply. The product owner decided: one unified engine only.

## Decision

1. **Adapters describe, one engine executes.** An adapter resolves an
   artifact's metadata and tensor index into a typed `ModelGraph`:
   - the embedding, a list of layers, the final norm and the output head;
   - per layer: a pre-norm, a token **mixer** and an **FFN**, with residual
     wiring;
   - each tensor binding checked once for dtype and shape.
   An unknown architecture or block is a typed refusal that names the
   missing block. It is never approximated.
2. **Blocks are the only place model math lives.** Each block is
   implemented once, on the CPU oracle and on Metal:
   - mixers: `Attention` (GQA; optional Q/K norm; full or partial RoPE,
     NeoX or standard; optional sigmoid output gate) and `GatedDeltaNet`;
   - FFNs: `SwiGLU` now, `GeGLU` and `MoE` (routed + shared experts) as
     families need them;
   - norms: `RMSNorm` with the weight variants families use.
   Each block offers `decode` (one token) and `prefill` (a chunk) and
   declares its session state (KV planes, recurrent/convolution state)
   and scratch for the planner. A new family adds an adapter, and a new
   block only for genuinely new math.
3. **The engine owns everything cross-cutting, once:**
   - one command buffer per decode step and per prefill chunk;
   - chunked prefill, with logits for the last token only;
   - device-side greedy sampling;
   - session reset and cancellation at chunk/token boundaries;
   - preflight allocation from the graph (one plan function, ledger ==
     plan), checked per class against the admitted plan;
   - per-operator profiling;
   - weight access through one `WeightStore`: resident now, and the place
     Phase 2 streaming plugs in.
4. **One oracle per block, plus the old oracles as references.** The CPU
   engine executes the same `ModelGraph` with scalar f32 blocks. The
   existing per-family CPU oracles stay as independent references until
   the graph oracle reproduces them bit-for-bit, then retire.
5. **Same-profile execution is unchanged.** The graph *is* the artifact's
   computation: no block is substituted, dropped or approximated, and
   precision profiles stay declared and recorded (ADR-010).

## Migration (each step gated, nothing deleted before its gate)

1. Engine core plus the blocks Qwen3.5 needs. Qwen3.5 is served through
   the engine. Gate: identical greedy output and teacher-forced logits
   matching the current Qwen3.5 GPU path (same kernels, so bit-identical
   decode is expected) and the CPU oracle within registered tolerances.
   Then retire `qwen35_gpu.rs`'s model code.
2. Dense Qwen3 as a graph on the same blocks. Gate: its CPU oracle and
   registered llama.cpp envelopes, plus the recorded batch-size claims
   re-proven or explicitly superseded in `docs/claims.md`. Then retire
   `gpu.rs`/`forward.rs` model code.
3. New families are adapters (+ blocks): Llama 3.x, Gemma, Mistral, then
   MoE.

Performance work from now on (chunk-wide DeltaNet, batched attention,
fusions, GPU sampling, streaming) lands in engine blocks or the engine,
never in a family-specific path.

## Consequences

- One place to optimize: every gain reaches every family.
- The migration costs some performance work now (rebuilding today's
  Qwen3.5 speed on blocks). The bit-identical gate makes that safe.
- The CLI, worker protocol and planner contracts do not change; TuckCore
  still implements `InferenceBackend`.

## Step 1 record (2026-10-26): Qwen3.5 served through the engine

- `crate::engine`: `ModelGraph` (`graph.rs`), the `WeightSource` seam
  (`mod.rs`), the Qwen3.5 adapter (`adapters.rs`), and on Metal the
  blocks `Matrix`, `SwiGluBlock`, `GdnBlock` and `AttentionBlock` plus
  `EngineModel` (`metal.rs`): one plan function, one command buffer per
  step and per chunk, sessions and profiling. Each block's `decode` and
  `prefill` share one `core`.
- **Gate passed before anything was deleted.** Run side by side, one model
  resident at a time, on the fixture, the pinned 0.8B and the 2B (sha256
  `1b04acba…21f2c1`): equal session state and plan total, equal device
  ledger, bit-identical per-layer hidden states and logits at every decode
  step, bit-identical batched-prefill logits and the decode step after it.
- The executor serves `qwen35` through `EngineModel`; the load note names
  the graph. Worker greedy output still equals the CPU oracle.
- **Retired:** the Qwen3.5 model code in `qwen35_gpu.rs` (layers, model,
  plan; 1,207 lines). Only the device operators remain there. Its
  qualification suite moved to `engine::metal::oracle_tests` unchanged,
  except that it targets the engine.
- **Found on the way:**
  - The engine makes the head explicit (`output` if present, else the
    tied embedding). The retired model always used the embedding; the
    CPU oracle still does and is fixed when the graph oracle lands.
  - The derived RoPE tables are now priced as scratch, not weights. The
    planner reserves exactly the artifact's bytes, and on the 2B (no MTP
    block, so no slack) the old classing was rightly refused by 768 B.
- **2B evidence:** `tuckmill run` (unified engine) and pinned llama.cpp
  Metal produce the same 16 greedy tokens for "The capital of France
  is".
- Next: step 2 (dense Qwen3 on the same blocks), and the graph-driven CPU
  oracle.

## Engine note (2026-10-26): greedy decoding on the device, pipelined

- `tm_metal_argmax_f32` chooses the greedy token on the GPU with exactly
  the CPU rule (`sampling::argmax`: NaN skipped, ties to the lowest index;
  the order is associative, so the two-pass reduction gives the
  sequential answer). It writes one of two alternating token slots.
  `tm_metal_row_indirect_f32` embeds a token id read from such a slot; the
  all-NaN marker writes zeros and the host fails the request when it
  reads the marker.
- `EngineModel::submit_greedy` / `collect_greedy`: the executor queues step
  i+1 (reading step i's choice on the device) before waiting for step i,
  and reads each token with `download_after`, which waits only for that
  step's command buffer. The worker's per-token work (output frames,
  stop/EOS handling) now overlaps the GPU. Temperature > 0 keeps the CPU
  sampler. `reset()` drains first, so a step left in flight at EOS or a
  stop sequence never races the next request's state reset.
- Evidence (pinned 0.8B, M3 8 GB): device greedy and pipelined greedy
  produce the CPU rule's tokens at every step (fixture and pinned, 40
  steps). In-process the step is GPU-bound (14.2–15.0 ms for all three
  paths: the logits download was not the cost). Served decode via
  `tuckmill bench` (128 in / 256 out, ABAB): 56.5 → 64.0 and 51.0 → 59.0
  tok/s (+13–16%), locally-calibrated.

## Engine note (2026-10-26): chunk-wide mixer cores in prefill

- Prefill used to run each layer's sequential core once per token of a
  16-token chunk (about 14 launches plus 5 slice copies per token per
  layer). It now runs both mixer cores chunk-wide:
  - **Gated DeltaNet:** convolution over the chunk with the carried
    history (`conv_chunk` + history refresh), row-batched SiLU, split,
    L2 norms, beta/decay and gated norm, and one recurrence launch that
    walks the chunk's tokens inside each (head, column) thread.
  - **Gated attention:** row-batched gate split and Q/K norms, then RoPE,
    packed-f16 KV store and causal attention for all chunk queries.
    Attention uses one shared inline body (`tm_q35_attend`) for the
    per-token and chunk kernels.
- Every chunk kernel keeps its per-token counterpart's per-element
  arithmetic. Gate: with `set_chunk_core(false)` (the per-token cores) as
  reference, last-token logits and the next decode step are
  **bit-identical** on the fixture (prompts 1/5/16/37) and the pinned 0.8B
  (1/5/16/37/512). The per-token path stays as that reference.
- Effect (pinned 0.8B, M3 8 GB, in-process): 512-token prefill
  1.73 s → 1.29 s (DeltaNet core) → 0.84 s (plus attention). Profile now:
  the matrix-unit GEMM 58%, the DeltaNet chunk recurrence 27%.

## Engine note (2026-10-26): register-resident DeltaNet recurrence; fast-math caveat

- The chunk recurrence was memory-traffic bound: it re-read each 128-float
  state column from device memory twice per token and wrote it once
  (~48 MB per 16-token launch, ~27 GB per 512-token prompt).
  `tm_qwen35_gdn_chunk_k128` (a template instance for key dim 128) keeps
  the column in registers for the whole chunk: read once, written once.
  Other key dims keep the generic kernel.
- **Fast-math caveat.** The kernel library is compiled with Metal's
  default options, i.e. fast math, which may reassociate sums per
  kernel. Source-identical arithmetic in two kernels is therefore not
  guaranteed to round identically. The register kernel differs from the
  generic one by 5.0e-4 / 7.2e-4 on 512-token logits / next step (within
  the registered 2e-3 GPU bound). Earlier "bit-identical" results held
  because those kernels compiled identically, not by guarantee. Rule from
  here on: paths that must agree bit-for-bit share one compiled kernel.
  Decode's single-token step now runs the same chunk kernel with n = 1,
  so chunked prefill and decode stay bit-identical to each other (gated).
- Qualification against the CPU oracle (registered tolerances unchanged):
  pinned delta operator 4.27e-4 out / 1.37e-4 state, real artifact hidden
  7.0e-5 / logits 4.7e-4, fixture model 6.6e-7 / 1.4e-6, batched vs
  sequential 4.6e-4 / 4.0e-4 with identical top-1; worker greedy output
  equals the CPU oracle.
- Effect (pinned 0.8B, M3 8 GB, in-process): 512-token prefill 879 →
  646 ms; recurrence per decode token 1.25 → 0.47 ms; greedy decode
  ~73–77 tok/s in-process.
