# Worker control protocol (v7)

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
- v2 (ADR-005): `ArtifactRef` + backend options on `Load`;
  `Score`/`ScoreChunk`/`ScoreDone` teacher-forced evaluation; backend
  statistics on `StatsAck`.
- v3: `ScoreChunk` carries an `offset`; one position's logits stream in
  multiple frames sized for the 1 MiB cap (150k+ vocabularies).
- v4: `Control`/`ControlAck` frames give the governor a real command
  contract — `trim-caches` (applied by engines at safe checkpoints; in-flight
  buffers are never released) and `set-prefetch=on/off` (recorded; no
  speculative prefetch exists in this build). Workers emit heartbeats from a
  helper thread during long single-step compute, and supervisors skip
  heartbeats when awaiting meaningful frames. Older plans are `PLAN_STALE`.
- v5: `Loaded` carries `resolved_kernels` — the kernel ids the backend
  actually resolved at load (e.g. `K01:q8_0+f16:neon`). Supervisors record
  them into engine identities so benchmark and verification records name the
  implementation that ran, not a build-time guess. Older workers omit the
  field (defaults to empty).
- v6: `Done` carries optional worker-side `timing` (`total_ms`, `prefill_ms`,
  `first_token_ms`, `decode_ms`) measured inside the backend. Supervisors
  record it separately from their own wall time so queueing and transport
  are never confused with compute; older workers omit the field.
- v7: `Generate` carries the typed `sampling` policy (`temperature`, `top_p`;
  absent means greedy). The worker validates it and refuses unsupported
  values before any compute — never clamps or ignores them. `Done` carries
  `generated` alongside `committed`: `committed` counts tokens with visible
  bytes (stop/EOS-trimmed tokens are excluded), `generated` counts every
  sampled token id. `StatsAck` backend stats include `kv_state_bytes`, the
  prepared KV-state bytes in the backend's actual representation, plus
  `session_kv_bytes`, the KV bytes allocated for the most recent request's
  own admitted need, and `retained_weight_bytes`, the largest weight-cache
  allocation the load can retain between requests (the supervisor re-checks
  it against every later plan's reservations before reusing the worker); the supervisor checks the prepared number against the
  plan's KV reservation after every load and fails closed on an understated
  plan. `Control` frames are request-scoped on the supervisor side: a
  command whose request already ended is dropped (counted), never replayed
  into a later request.
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
| `Loaded` | W→C | `note`, `footprint_bytes`, `resolved_kernels` (v5) | `note` must disclose synthetic engines and experimental kernels; `resolved_kernels` names the implementation actually selected. |
| `Generate` | C→W | `request_id`, `prompt_tokens` (exact IDs), `max_output_tokens`, `stop`, `seed`, `sampling` (v7) | Prompt+output must fit the admitted context. `sampling.temperature <= 0` means greedy; unsupported values are a typed rejection. |
| `State` | W→C | `state` ∈ warming/prefilling/generating/throttling/paused | Advisory progress. |
| `Token` | W→C | `index`, `text` | Visible tokens only, stop/EOS-trimmed text excluded. Draft text is forbidden. |
| `Heartbeat` | W→C | `uptime_ms` | Emitted at ≈500 ms while idle; liveness evidence. |
| `Done` | W→C | `reason` ∈ stop/length/cancelled/eos, `committed`, `generated` (v7) | Terminal for the current request. |
| `Error` | W→C | `code` (public error code), `message`, `action` | Terminal; follows the typed error vocabulary. |
| `Cancel` | C→W | `request_id` | Honored at the next committed-token boundary. |
| `Score` | C→W | `request_id`, `tokens` (exact ids), `positions` | Bounded: ≤16384 tokens, ≤64 positions, positions within the sequence. |
| `ScoreChunk` | W→C | `position`, `logits` (vocabulary order, f32) | Teacher-forced logits for one position. |
| `ScoreDone` | W→C | `scored` | Terminal for the scoring request. |
| `Control` | C→W | `action` | Governor/administrative command (`trim-caches`, `set-prefetch=on/off`). Applied locally; engines act at safe checkpoints. |
| `ControlAck` | W→C | `action`, `result` | Honest result string ("queued", "recorded", "unsupported ..."). |
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
