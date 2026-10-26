# Review notes: unified engine (`8fe1536`), Qwen3.5-0.8B Q8_0, M3 8 GB

Follows [../../TEMPLATE.md](../../TEMPLATE.md). All numbers are
**locally-calibrated**. This is the first official run with batched
prefill, the matrix-unit prefill GEMM and the unified engine (ADR-011
step 1).

## Conditions

- Chrome and other heavy apps closed; the "Hull" app (~1 GiB) and the
  Claude app were running. Reclaimable memory before each run was
  1.9–3.2 GiB.
- Two attempts were discarded for swap growth and retried once (new
  retry rule): llama-metal round 2 (+197 MiB) and tuckcore-metal round 2
  (+1,356 MiB). All 6 final invocations are valid.
- llama.cpp ran slower than in the previous clean run (`20261026T104858Z`:
  72.7 tok/s; here 65.6; llama-bench tg256 60.6 ± 15.0 vs 86.8). Paired
  ratios compare engines within the same round, which removes most of
  that drift.

## Results

| engine | rounds | warm TTFT (512 tok) | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 1.71 s | 39.70 | 54.12 (52.61–56.60) | 1,109 MiB |
| llama-metal | 3 | 0.29 s | 61.08 | 65.64 (64.89–66.68) | 241 MiB |

- Paired decode ratio TuckCore / llama.cpp: **0.824** (0.872, 0.824,
  0.789). Previous clean run: 0.741.
- Paired TTFT ratio: **5.69×** (previous: 29.5×).
- Against Phase 0: decode 5.55 → 54.12 tok/s (9.8×); TTFT 88.9 → 1.71 s
  (52×); e2e 1.93 → 39.70 tok/s (20.6×).
- Against the previous official run (`2619850`): TTFT 7.97 → 1.71 s
  (4.7×) from batched prefill. Decode is unchanged within noise (55.7 vs
  54.1): no decode work landed in between.

## Reading

- The decode ratio rose partly because llama.cpp ran slower in this
  session. TuckCore's own decode did not change. The remaining decode gap
  is host-side per-token work (logits download + CPU sampling over
  248,320 candidates).
- **Footprint regressed: 928 → 1,109 MiB.** The device ledger barely moved
  (826.9 vs 825.4 MB), so the growth is outside the GPU plan, on the host
  side of the worker. Unexplained; it goes first in the memory work.
- Prefill is 5.7× behind llama.cpp. The rest is the per-token DeltaNet,
  attention and copies inside each chunk (see ADR-008 Add. 15/16).
