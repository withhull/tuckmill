# Review notes: register-resident recurrence (`6b6a014`), cool machine, Qwen3.5-0.8B Q8_0, M3 8 GB

Follows [../../TEMPLATE.md](../../TEMPLATE.md). All numbers are
**locally-calibrated**. Rerun of `20261026T162728Z-…` (throttled) after
~15 minutes idle: reclaimable 3.4 GiB and compressor ~0.26 GiB at the
start. The first TuckCore attempt was discarded (swap +2,045 MiB, most
likely another app waking) and retried cleanly; all 6 final invocations
are valid. llama.cpp ran at its normal speed (llama-bench tg256 75.7).

## Results

| engine | rounds | warm TTFT (512 tok) | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 0.630 s | 54.44 | 62.65 (62.10–62.72) | 961 MiB |
| llama-metal | 3 | 0.299 s | 65.21 | 70.31 (63.77–71.76) | 241 MiB |

- Paired decode ratio: **0.891** (0.974, 0.891, 0.874). Round 1's 0.974 is
  partly llama.cpp's slower first round; rounds 2–3 give 0.87–0.89.
- Paired TTFT ratio: **2.11×** (0.630 vs 0.299 s).
- llama-bench vendor row: pp512 1807.4 ± 38.1, tg256 75.7 ± 6.1.

## Progression (official runs, same machine)

| run | commit | TuckCore decode | TuckCore TTFT | decode vs llama.cpp | TTFT vs llama.cpp |
| --- | --- | --- | --- | --- | --- |
| Phase 0 | `5c6e889` | 5.55 | 88.9 s | 0.08 | 300× |
| after 1C | `2619850` | 55.73 | 7.97 s | 0.74 | 29.5× |
| unified engine | `8fe1536` | 54.12 | 1.71 s | 0.82 | 5.7× |
| chunk cores | `6ba9823` | 59.54 | 0.835 s | 0.84 | 2.8× |
| **register recurrence** | `6b6a014` | **62.65** | **0.630 s** | **0.89** | **2.1×** |

## Reading

- **Decode** is just under the Phase 1 gate's 90% (paired median 0.891).
  TuckCore's own rate is up from 59.5 to 62.7 tok/s.
- **Prefill** is 2.1× behind, ~47% of llama.cpp's speed. The remaining
  cost is mainly the matrix-unit weight GEMM.
- Phase 1 gate (≥90% decode and prefill) on the ~1B row: **not yet met**.
