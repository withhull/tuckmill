# Review notes: after Phase 1B (`dafeadd`), Qwen3.5-0.8B Q8_0, M3 8 GB

All numbers are **locally-calibrated**. This run was **heavily disturbed**,
so read it as a direction, not a measurement.

- 4 of 6 invocations were discarded for swap growth (+136 to +462 MiB);
  one valid run per engine, so there is no paired round.
- The pinned llama-bench tool itself dropped to tg256 46.5 ± 4.1 tok/s,
  against 69.0 in the Phase 0 run on the same machine. The machine was
  slower overall.
- Valid TuckCore run (round 2): warm decode 34.6 tok/s, warm TTFT 17.7 s
  at 512 tokens. Phase 0 recorded 5.55 tok/s and 88.9 s.
- Valid llama.cpp run (round 1): 71.3 tok/s, 0.28 s.
- The in-process profiler measured 52 tok/s for the same commit. The gap
  to 34.6 is host-side work the profiler excludes (sampling over 248,320
  logits, the request loop) plus this run's memory pressure. Phase 1C
  closes it.

Follow-up: the script now retries a swap-discarded invocation once, so a
noisy machine still yields pairs.
