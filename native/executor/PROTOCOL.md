# Worker control protocol (v2)

Wire contract between the Tuckmill supervisor and an execution worker. The
transport is a private Unix domain socket under the store's `tmp/` directory.
The current worker implements this protocol in Rust; the future C++/
Objective-C++ executor implements the same frames behind
`tuckmill_worker_abi.h`.

## Framing

```
frame := u32_le length | payload[length]
payload := UTF-8 JSON, one message object
```

- `length` must be `1..=1_048_576` bytes (1 MiB). Frames outside the range
  fail as `WORKER_FAILED`; they are never buffered or truncated.
- v2 (ADR-005) adds: `ArtifactRef` + backend options on `Load`; the
  `Score`/`ScoreChunk`/`ScoreDone` teacher-forced evaluation messages; backend
  statistics on `StatsAck`. v1 plans are `PLAN_STALE`.
- Clean EOF before the first byte of a frame closes the session. EOF mid-frame
  is `WORKER_FAILED`.
- All fields are validated before use; unknown message tags are protocol
  violations, not ignored.

## Message envelope

Every message is a JSON object with `"t"` as the tag:

```jsonc
{ "t": "Hello", "protocol": 1, "abi": 1, "client_build": "tuckmill 0.2.0 (debug)" }
```

## Message catalogue

| Tag | Direction | Payload | Notes |
| --- | --- | --- | --- |
| `Hello` | C→W | `protocol`, `abi`, `client_build` | Must be the first frame. |
| `HelloAck` | W→C | `protocol`, `abi`, `worker_build`, `engine`, `certified` | Protocol/ABI mismatch fails closed. |
| `Load` | C→W | `plan` (full ExecutionPlan JSON), `artifact` (`ArtifactRef`: path + identity hashes), optional `options` | The backend must re-verify the artifact hashes before executing; options are validated and unknown keys rejected. |
| `Loaded` | W→C | `note`, `footprint_bytes` | `note` must disclose synthetic engines and experimental kernels. |
| `Generate` | C→W | `request_id`, `prompt_tokens` (exact IDs), `max_output_tokens`, `stop`, `seed` | Prompt+output must fit the admitted context. |
| `State` | W→C | `state` ∈ warming/prefilling/generating/throttling/paused | Advisory progress. |
| `Token` | W→C | `index`, `text` | Committed tokens only. Draft text is forbidden. |
| `Heartbeat` | W→C | `uptime_ms` | Emitted at ≈500 ms while idle; liveness evidence. |
| `Done` | W→C | `reason` ∈ stop/length/cancelled/eos, `committed` | Terminal for the current request. |
| `Error` | W→C | `code` (public error code), `message`, `action` | Terminal; follows the typed error vocabulary. |
| `Cancel` | C→W | `request_id` | Honored at the next committed-token boundary. |
| `Score` | C→W | `request_id`, `tokens` (exact ids), `positions` | Bounded: ≤16384 tokens, ≤64 positions, positions within the sequence. |
| `ScoreChunk` | W→C | `position`, `logits` (vocabulary order, f32) | Teacher-forced logits for one position. |
| `ScoreDone` | W→C | `scored` | Terminal for the scoring request. |
| `Stats` | C→W | — | Request for `StatsAck`. |
| `StatsAck` | W→C | `footprint_bytes`, `uptime_ms`, optional backend stats | Backend stats: logical reads, cache hits/misses, evictions, in-flight leases. Measured or 0-with-no-claim; never guessed. |
| `Shutdown` | C→W | — | Worker replies `Bye` and exits. |
| `Bye` | W→C | — | Clean teardown. |

## Semantics

- **Cancellation:** the supervisor sends `Cancel`; the worker checks it between
  committed tokens and returns `Done{reason:"cancelled"}`. If the worker does
  not acknowledge within the supervisor's grace period, the supervisor kills
  the process and reports the escalation (`WORKER_FAILED`) instead of faking
  a clean cancel.
- **Pause:** pause is implemented as supervisor-side backpressure (the
  supervisor stops reading token frames at a committed-token boundary). The
  worker may block on socket writes; this is intentional and bounded by
  kernel socket buffers.
- **Crash:** supervisor-side EOF mid-generation ⇒ `WORKER_FAILED` with the
  last acknowledged committed index retained. The supervisor never fabricates
  continuation state.
- **Liveness:** during generation, silence longer than 30 s (or 3 s in the
  cancel grace window) is a failure; heartbeats prevent false positives.
