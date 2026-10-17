# ADR-007: TuckCore Metal path and the TuckCore–Metal ABI

Status: accepted (Gate C, C0/C1)

## Context

Gate B is accepted: the request, worker, and CPU foundation is verified on the
unchanged Qwen3-0.6B Q8_0 artifact (sha256 `9465e63a…bb031`). The next
milestone is real Qwen inference executed end-to-end by TuckCore on the Apple
GPU, without delegating execution to llama.cpp. The existing
`native/executor/tuckmill_worker_abi.h` is a *worker-level* scaffold; forcing
Metal operator calls through it would couple the supervisor protocol to GPU
resource management.

## Decision

1. **Dedicated TuckCore–Metal ABI.** `native/metal/tm_metal.h` is a narrow C
   ABI for resource handles (buffers), compiled pipelines, dispatch,
   synchronization, and measurement. The supervisor/worker protocol
   (PROTOCOL.md v7) is untouched; only TuckCore executes Metal calls.
2. **One owner.** A `tm_metal_ctx` owns exactly one `MTLDevice`, one
   `MTLCommandQueue`, and the compiled pipeline states. All Metal objects are
   created, used, and released through it.
3. **Shaders ship in-repo and compile at runtime.** MSL lives in
   `native/metal/kernels.metal` (versioned with the code), is embedded into
   the crate at build time, and is compiled by the OS Metal compiler at
   context creation. No Xcode toolchain is required at build time, and the
   shader revision is exactly the repository revision — nothing is fetched.
4. **Explicit ownership and accounting.** Every GPU buffer is a handle owned
   by the Rust wrapper with RAII release; the context reports live allocated
   bytes (`allocated_bytes`) and committed/completed dispatch counts, so the
   planner/executor can acknowledge actual Metal allocations before
   execution instead of assuming them. GPU/CPU copies are budgeted, never
   duplicated silently: tests assert live allocation returns to zero after
   teardown.
5. **Correctness before speed.** The first Metal increment implements the
   Q8_0 packed matrix-vector decode plus F16/F32 utility ops, validated
   against the CPU scalar reference on real tensor shapes (including tails
   and quantization boundaries) and on real artifact tensor bytes. Kernels
   beyond that are added per the C3/C4 plan only when profiling shows the
   gap; the registry keeps reference fallbacks for every selection.
6. **Test gating.** GPU tests run whenever a Metal device is present and are
   skipped with a loud note otherwise; `TUCKMILL_REQUIRE_METAL=1` (the
   repository gate on owned hardware) turns a missing device into a
   failure. Numerical assertions are tolerance-registered, never exact-bit,
   because GPU reduction order differs from the CPU sequence.

## Consequences

- `crates/metal` (`tuckmill-metal`) wraps the ABI into typed Rust handles
  (`MetalContext`, `MetalBuffer`) with `Result` errors; no `unsafe` outside
  the wrapper, each block carrying a SAFETY note.
- The C2 stage plugs these buffers and ops into the TuckCore executor behind
  the existing `InferenceBackend` interface; planner admission gains a Metal
  reservation class before any resident weights move to the GPU.
- The llama.cpp Metal baseline remains the comparator, not the execution
  path; identical artifacts, token IDs, and declared numerical profiles
  still govern every comparison (ADR-006).
