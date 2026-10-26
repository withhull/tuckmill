# Review notes: chunk-wide cores + pipelined decode (`6ba9823`), Qwen3.5-0.8B Q8_0, M3 8 GB

Follows [../../TEMPLATE.md](../../TEMPLATE.md). All numbers are
**locally-calibrated**. Clean run: heavy apps closed, reclaimable
2.3–3.2 GiB, no swap growth, all 6 invocations valid on the first
attempt.

## Results

| engine | rounds | warm TTFT (512 tok) | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 0.835 s | 50.02 | 59.54 (57.80–60.77) | 948 MiB |
| llama-metal | 3 | 0.294 s | 64.65 | 69.59 (67.84–70.97) | 241 MiB |

- Paired decode ratio TuckCore / llama.cpp: **0.839** (0.839, 0.896,
  0.831).
- Paired TTFT ratio: **2.84×** (previous official run: 5.69×).
- llama-bench vendor row: pp512 1847.6 ± 14.5, tg256 77.0 ± 2.1 tok/s.
- Footprint is not like-for-like across engines (llama.cpp's mmap'd
  weights are excluded from its footprint; see the correction in the
  `20261026T131919Z` notes).

## Progression on this machine (official runs)

| run | commit | TuckCore decode | TuckCore TTFT | decode vs llama.cpp | TTFT vs llama.cpp |
| --- | --- | --- | --- | --- | --- |
| Phase 0 | `5c6e889` | 5.55 | 88.9 s | 0.08 | 300× slower |
| after 1C | `2619850` | 55.73 | 7.97 s | 0.74 | 29.5× |
| unified engine | `8fe1536` | 54.12 | 1.71 s | 0.82 | 5.7× |
| **this run** | `6ba9823` | **59.54** | **0.835 s** | **0.84** | **2.8×** |

## Reading

- **Decode** is at 84% of llama.cpp paired. Pipelined greedy decoding
  (`4d59c17`) lifted TuckCore from ~54 to ~60 tok/s.
- **Prefill** is 2.8× behind. The matrix-unit GEMM (~58%) and the DeltaNet
  chunk recurrence (~27%) dominate; a 64-wide-stage GEMM variant showed no
  reliable gain under thermal drift and was not kept.
- **Not established:** thermals beyond 3 rounds; other prompts or tiers;
  the 2B.
