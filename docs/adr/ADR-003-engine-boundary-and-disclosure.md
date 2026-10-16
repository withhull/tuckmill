# ADR-003: Engine boundary and disclosure

Status: **committed** (2026-10-16). Runtime.
**Amended by ADR-005**: more than one backend now exists behind a versioned
`InferenceBackend` interface (reference-test, TuckCore, later llama.cpp); the
disclosure rules below apply to every backend.

## Context

The supervisor must stay responsive when a worker fails, hold no model state,
and never compute tensors. The PRD requires a narrow C-compatible ABI between
the Rust supervision layer and the C++/Objective-C++ execution layer, and it
requires that no draft text or synthetic output is ever presented as final
model output.

## Decision

1. **Worker isolation is the default.** `tuckmill run`, `tuckmill bench`, and
   `tuckmill serve` spawn a dedicated worker process (`tuckmill
   __worker-reference`) that owns the engine. The supervisor communicates
   over a versioned, length-framed JSON protocol on a private Unix socket
   (`native/executor/PROTOCOL.md`, 1 MiB frame cap, NUL-free message schema).
2. **One engine exists in this build: the reference test engine.** It
   validates pipeline behavior (admission, scheduling, streaming, cancellation,
   pause/resume, process isolation) with deterministic synthetic tokens
   (`[ref:i]`). It executes no model weights, holds no vocabulary, and is
   never certified. `EngineIdentity.certified` is `false` and plans record
   `reference_test_only`.
3. **Explicit acceptance gate.** Any profile without a certified executor is
   refused (`UNAVAILABLE`) unless the caller passes `--allow-test-engine` /
   starts `serve` with it. SSE/JSON responses carry
   `"synthetic_output": true` in the `tuckmill` extension object.
4. **Failure semantics.** Worker crash ⇒ `WORKER_FAILED` with the last
   acknowledged committed position retained; the supervisor never fabricates
   continuation state. Cancellation is cooperative at committed-token
   boundaries with a bounded grace period; if the worker does not honor it,
   the supervisor kills the process and reports the escalation instead of
   pretending the cancel was clean.
5. **Pause is backpressure.** Governor pauses stop the supervisor from reading
   further token frames at a committed-token boundary; buffers are never
   reclaimed mid-flight to fake fast cancellation.

## Consequences

- The C ABI header (`native/executor/tuckmill_worker_abi.h`) defines the
  surface a real executor must implement; the protocol document is the wire
  contract. Replacing the reference engine means implementing that ABI, not
  changing the supervisor.
- Kernel work packages K01–K09 remain unimplemented; every plan lists the
  kernels its profile would require, and certification stays `unmeasured`
  until physical-Mac gates (G1/G2) pass.
