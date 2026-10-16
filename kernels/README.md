# Kernel work packages (K01–K09)

Statuses live in [`registry.json`](registry.json) and are enforced by
`tuckmill_planner::kernel_registry`: a kernel must never be selected before
its correctness gate passes, and every selection keeps a validated reference
fallback decision. **This build implements none of them** — every plan lists
the kernels its profile requires so the gap is explicit rather than implied.

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
