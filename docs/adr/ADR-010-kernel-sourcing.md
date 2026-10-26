# ADR-010: Kernel sourcing, precision profiles, and kernel acceptance

Status: **accepted** (2026-10-26). Extends ADR-007 (Metal ABI) and ADR-005
(TuckCore). Does not change AGENTS rule 6: the llama.cpp reference backend
stays pinned and runtime-loaded.

## Context

TuckCore's Metal path is correct-first. It uses one simdgroup per row,
byte loads, GEMM that re-reads activations per row, f32 activations, and
f32 dense KV. It executes only Q8_0/F16; K-quants and IQ formats are
refused. The Phase 0 baseline measures the gap against pinned llama.cpp
Metal. ggml-metal's quantized kernels are MIT-licensed, mature, and
written for exactly these block formats.

## Decision

1. **Port, attribute, pin.** Quantized kernels for Q4_0, Q8_0, Q4_K, Q5_K,
   Q6_K and IQ4_XS are ported from ggml-metal at the pinned llama.cpp
   revision (b11695) into `native/metal/kernels_ggml.metal`, adapted to the
   TuckCore ABI. The file header names the source path, the revision and
   the upstream file's SHA-256. `THIRD_PARTY_NOTICES` carries the MIT
   license text. This is porting source, not vendoring the reference
   backend, which keeps loading at runtime and never becomes the TuckCore
   execution path.
2. **Block layouts are derived, never hand-written.** Block sizes, scale
   packing and grid tables come from the pinned `ggml.rs` constants and the
   matching CPU dequantizers. The CPU dequantizer for each format lands
   first and is checked bit-exactly against llama.cpp's own dequantization
   of real artifact tensors.
3. **The scalar CPU path stays the oracle.** No kernel is selectable
   before it passes three registered checks in `kernels/registry.json`:
   - dequantization: bit-exact against the CPU reference;
   - GEMV/GEMM: within a registered f32 reduction-order tolerance, on
     boundary data and real artifact bytes;
   - whole model: teacher-forced logits within the model's registered
     envelope.
   The fallback decision stays recorded for every selection.
4. **Precision is a declared, recorded profile, never a substitution.**
   - The model's stored weights are never requantized.
   - Activation precision is a numerical profile: `oracle-f32` (current
     default) and `metal-f16-act`, the latter qualified separately against
     the oracle and the reference.
   - KV precision is declared per request: `f16` and `q8_0`, each
     qualified separately; q8_0 KV is opt-in only.
   - The resolved profile appears in identity, stats and every record. A
     default changes only by an explicit, recorded decision after its
     qualification passes.
5. **Kernels are accepted by measurement (AGENTS rule 7).** Each kernel
   lands with a before/after run of `scripts/bench-baseline.sh` on the
   same artifact and machine. It is kept only if complete requests get
   faster, and the record is committed with it. Planned shapes:
   - multi-row GEMV with vector loads;
   - `simdgroup_matrix` GEMM for prefill;
   - fused RMSNorm + projection, SwiGLU and RoPE.
   Ported kernels are measured the same way; provenance is not evidence.

## Consequences

- `tensor.rs` gains CPU dequantizers for every listed format before any
  Metal kernel for that format exists. Formats outside the list stay typed
  refusals.
- `kernels/registry.json` gets one entry per ported kernel: source,
  revision, tolerance and qualification record.
- The certification matrix tracks each format × kernel × profile
  separately. "Q4_K works" is never one row.

## Staging (2026-10-26)

Phase 1 is proven on small models first: the gate's ~1B row runs on the
pinned Qwen3.5-0.8B Q8_0 (no new download). The ~8B and ~30B rows, and
the K-quant formats that need new artifacts, stay pending until the
small-model results hold. That is a scheduling decision, not a change
to the gate.
