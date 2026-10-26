# Kernel work packages (K01–K09)

Statuses live in [`registry.json`](registry.json) and are enforced by
`tuckmill_planner::kernel_registry`: a kernel must never be selected before
its correctness gate passes, and every selection keeps a validated reference
fallback decision. **K01 has implemented increments** on the CPU (aarch64
NEON fused Q8_0/F16 packed dot with zero-copy mapped blocks; Q4_0/Q4_1 stay
on the scalar path) and on Metal (Q8_0 GEMV, Gate C1/C2); K02 has its Metal
batched-GEMM prefill increment (C3); the Qwen3.5 Metal operators (Q35-2)
are qualified in ADR-008 and not yet registered as packages. The remaining
packages are specified only — see `registry.json` for exact statuses; every
plan lists the kernels its profile requires so the gap is explicit rather
than implied.

| Package | Function | Gate summary |
| --- | --- | --- |
| K01 | Q8/Q4 packed matrix-vector decode | exact format/scale decoding; bounded tiles; per-operation error checks |
| K02 | Prompt matrix multiplication (shape-specific) | same projection semantics; recorded microbatch; bounded workspace |
| K03 | RMSNorm/activation/residual fusion | epsilon, dependency and rounding profile preserved |
| K04 | Ternary PQ2_0/PTQ1_0 projections | native codes/scales; transform metadata; malformed trits rejected |
| K05 | Hadamard/sign transforms | normalization, sign vector, block size, direction, rounding matched |
| K06 | Capability-compatible arithmetic | stored bits and rounding preserved; software BF16 where needed |
| K07 | Hybrid linear-attention/state kernels | recurrent precision and update order; prefill/chunk/decode consistency |
| K08 | Attention/KV and MoE dispatch | positions, masks, native expert set preserved |
| K09 | Verified candidate-block evaluation | correct sampling/rollback; committed tokens only |

Selection inputs (all must match): operation, shape, tensor format,
activation/state precision, GPU family, OS/driver, workspace budget, and a
validated build ID. Unsupported combinations fall back to the reference path
or are rejected before allocation.

Implementation rules: pinned reference executor first; one bottleneck at a
time; per-kernel feature flags; conservative floating-point settings in the
reference; reassociation/fast-math as explicitly tracked variants with new
numerical evidence; bounded work units with resources retained until fences
complete.
