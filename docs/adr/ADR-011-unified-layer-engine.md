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
