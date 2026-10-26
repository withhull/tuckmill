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

## Engine note (2026-10-26): 32-token matrix-unit GEMM tile

- `tm_q8_0_gemm_mma` now owns a 32-row x 32-token tile (4 accumulators per
  SIMD group instead of 2), so each dequantized weight tile serves twice as
  many tokens; prefill chunks are 32 tokens (`PREFILL_BATCH`). Per-token
  summation order over K is unchanged.
- Gate: GEMM tolerance test (adds batch 32 and a 27-token partial tile;
  batch 33 refused); engine and chunk-core tests; 512-token batched vs
  sequential logits 4.55e-4 / next step 3.98e-4, identical to the 16-token
  tile, same top-1.
- Effect (pinned 0.8B, M3 8 GB, in-process, alternating A/B x3 from two
  prebuilt binaries): 512-token prefill 624-625 ms -> 516-518 ms (-17%),
  locally-calibrated.

## Engine note (2026-10-26): one GEMM launch per 128-token prefill chunk

- Profile at `45b8386` (512 tokens, ungrouped): matrix-unit GEMM 69%,
  DeltaNet recurrence 10%, attention 9%. A 64-row tile (16 rows x 32
  tokens per SIMD group) was slower (520 -> 568 ms, A/B x3) and was not
  kept: the GEMM was short of threadgroups, not of math per tile. With a
  32-token chunk a 1,024-row matrix gets 32 threadgroups, and each chunk
  re-reads every weight.
- `tm_q8_0_gemm_mma` now covers a whole batch in one launch: a 1-D grid
  of 32-row x 32-token tiles, with neighbouring threadgroups sharing a
  weight tile (batch up to `TM_METAL_GEMM_MMA_MAX_BATCH` = 512). Per-tile
  arithmetic is unchanged. `Matrix::gemm` slices F16 and batch-1 GEMMs
  into <= 32-token pieces through the existing offsets. Prefill chunks
  are 128 tokens.
- Chunk size measured, not assumed (pinned 0.8B, M3 8 GB, in-process,
  alternating prebuilt binaries, two passes; planned scratch at capacity
  640):

  | chunk | 512-token prefill | scratch |
  | --- | --- | --- |
  | 32 | 493-522 ms | 9.7 MiB |
  | 64 | 413-438 ms | 15.4 MiB |
  | 128 | 393-414 ms | 26.8 MiB |
  | 256 | 401-409 ms | 49.6 MiB |

  128 is kept: the gains flatten after it while scratch keeps doubling.
  The attention-score scratch is chunk x heads x context, so long
  admitted contexts pay more. The planner prices it, so admission stays
  exact; a tiled attention without the score buffer is the follow-up.
- Gate: GEMM tolerance at batches 64, 77, 512 (513 refused); fixture
  batched prefill still bit-identical to sequential decode (1, 3, 128,
  132 tokens); 512-token logits 4.55e-4 / next step 3.98e-4, the same as
  before, top-1 equal. Profile after: GEMM 338 ms (was 422 ms ungrouped),
  still 68%.

## Engine note (2026-10-26): GEMM kernel variants measured, none kept

Isolated timing (`q8_0_gemm_batch_timing` with `TUCKMILL_PROFILE=1
TUCKMILL_GEMM_MMA=1`, now batches 32 and 128 after a warm-up, 100
iterations; M3 8 GB, alternating prebuilt binaries, two passes). Baseline
at 128 tokens: 3584x1024 470-500 us, 6144x1024 787-830 us, 1024x3584
500-556 us, about 2 TFLOP/s f32. Variants, all tolerance-passing:

| variant | 3584x1024 | 6144x1024 | 1024x3584 | verdict |
| --- | --- | --- | --- | --- |
| 2x2 SIMD-group layout, vector loads, 32x32 | 461-467 | 756-780 | 480-521 | noise |
| same, 64x32 / 32x64 | 441-483 | 745-803 | 543-595 | no gain |
| same, 64x64 (16 accumulators) | 2294-2416 | 3728-3746 | 2377-2484 | register spill |
| double-buffered stages, one barrier per block | 496-598 | 832-990 | 524-744 | slower |
| K = 64 per stage (two Q8_0 blocks) | 521-599 | 876-991 | 540-687 | slower |
| f16 operand tiles (prototype, not a profile change) | 476-492 | 784-822 | 534-553 | no gain |

- The pinned llama.cpp on this M3 does not use the Metal tensor API
  ("disabled for pre-M5"). Its `kernel_mul_mm` stages dequantized
  weights and activations as half. The f16 prototype shows that operand
  precision is not what separates the two engines, so the oracle-f32
  profile stays.
- From the isolated rates, the GEMMs are about 265 ms of the ~400 ms
  512-token prefill; the rest (DeltaNet recurrence, attention, norms and
  small operators) is about 135 ms. llama.cpp's whole pp512 is about
  258 ms, so closing the gap needs both less non-GEMM time and a faster
  GEMM. Why llama.cpp's GEMM is faster is not yet identified: it reads
  the same 34-byte Q8_0 blocks, a 64x32 tile and K = 32 stages, all of
  which were tried above. It is not timed in isolation here (the pinned
  tools have no matmul benchmark).

## Engine note (2026-10-26): DeltaNet recurrence split across SIMD lanes

- Non-GEMM profile at `00fd250` (512 tokens, ungrouped): DeltaNet
  recurrence 61 ms, attention 39 ms, RMS norm 16 ms, then small
  operators. The recurrence ran one thread per (head, state column):
  16 x 128 = 2,048 threads walking every token with two 128-long dot
  products, so it was latency-bound.
- `tm_qwen35_gdn_chunk_k128` now splits each column across P = 8 lanes
  of one SIMD group (16 rows each, in registers). Lane partial dot
  products are combined with XOR shuffles, so every lane of a column sees
  the same total. A SIMD group covers 4 adjacent columns, and lane l works
  on column l % 4, so state loads and stores stay coalesced. Decode's
  single-token step runs the same kernel (n = 1), so chunked prefill and
  decode remain bit-identical to each other.
- Measured (recurrence GPU time, profile, two passes each; decode is the
  per-token recurrence in `profile_pinned_decode`):

  | layout | prefill recurrence | decode recurrence |
  | --- | --- | --- |
  | one lane per column (before) | 61-62 ms | 0.445-0.468 ms |
  | 4 lanes, blocked | 65 ms | - |
  | 8 lanes, blocked | 35-41 ms | - |
  | 16 lanes, blocked | 26-28 ms | 0.617-0.619 ms (slower) |
  | 32 lanes, blocked | 44-49 ms | - |
  | 16 lanes, interleaved | 25 ms | 0.609-0.638 ms (slower) |
  | **8 lanes, interleaved (kept)** | **19.5 ms** | **0.404-0.424 ms** |

- Accuracy (registered tolerances unchanged): real-artifact GPU vs CPU
  oracle, hidden 7.0e-5 -> 6.5e-5 and logits 4.7e-4 -> 2.8e-4; 512-token
  batched vs sequential 4.27e-4 / 3.52e-4, top-1 equal; lane-split vs
  generic recurrence within 2e-3.
- Whole 512-token prefill, in-process, five alternating passes
  (the Claude app's renderer was using up to 65% CPU, so passes are noisy):
  best 407 -> 368 ms, median 422 -> 399 ms. Locally-calibrated.

## Engine note (2026-10-26): grouped-query attention kernel

- The attention kernel ran one threadgroup per (query head, row). The
  four query heads sharing a KV head each re-read the K/V prefix, each
  thread walked every key serially for its two output dims, and both
  softmax reductions used threadgroup barriers.
- `tm_qwen35_attention_gqa<HD, G>` runs one threadgroup per (KV head,
  row) for all G heads:
  - each K and V row is read once;
  - lanes hold contiguous head dims of every head's query in registers;
  - SIMD group h normalizes head h with SIMD-only reductions, using the
    same per-element softmax expression;
  - the V pass splits keys across SIMD groups and sums the four partials
    in a fixed order.
- Instantiated for (256, 4), which covers the 0.8B and the 2B, and for
  (16, 2), the fixture, so its bit-identity gate exercises it. Other
  geometries keep the old kernel. Decode's single-token call routes
  through the same kernel (n = 1), so prefill and decode stay
  bit-identical.
- Measured (pinned 0.8B, M3 8 GB, alternating prebuilt binaries):
  - attention GPU time, 512-token prefill: 39.5-40.4 -> 8.8-9.0 ms;
  - decode, per token at a short context: 0.333-0.340 -> 0.113-0.117 ms
    (this grows with context);
  - whole 512-token prefill, five passes each: 362-365 -> 329-333 ms.
- Accuracy (registered tolerances unchanged):
  - fixture prefill is still bit-identical to sequential decode;
  - operator tests vs the CPU packed-cache reference pass (4x2x16 and
    8x2x256);
  - real-artifact GPU vs CPU oracle: hidden 6.5e-5, logits 2.78e-4;
  - 512-token batched vs sequential: 3.36e-4 / 3.36e-4, top-1 equal.
