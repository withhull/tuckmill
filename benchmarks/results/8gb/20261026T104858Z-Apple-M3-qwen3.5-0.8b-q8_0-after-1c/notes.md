# Review notes: after Phase 1C (`2619850`), Qwen3.5-0.8B Q8_0, M3 8 GB

Follows [../../TEMPLATE.md](../../TEMPLATE.md). All numbers are
**locally-calibrated**. This is a clean run: heavy apps were closed
(compressor ~1.1 GiB, reclaimable 2.2–3.6 GiB), and all 6 invocations
were valid with no swap growth.

## Results

| engine | rounds | warm TTFT (512 tok) | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 7.97 s | 20.40 | 55.73 (49.59–56.77) | 928 MiB |
| llama-metal | 3 | 0.28 s | 67.30 | 72.68 (70.13–76.59) | 242 MiB |

- Paired decode ratio TuckCore / llama.cpp: **0.741** (0.707, 0.767,
  0.741).
- Paired TTFT ratio: **29.5×**.
- llama-bench vendor row: pp512 1982.5 ± 6.9, tg256 86.77 ± 0.03 tok/s.
- Against Phase 0 (`20261026T092923Z-…`): decode 5.55 → 55.73 tok/s
  (10.0×), TTFT 88.9 → 7.97 s (11.2×), footprint 953 → 928 MiB.

## Reading

- **Decode.** The in-process profiler measures ~14.4 ms of GPU step per
  token (~69 tok/s). The served request takes ~17.9 ms. The ~3.5 ms
  difference is host-side per-token work: a 1 MB logits download and CPU
  sampling over 248,320 candidates. llama.cpp pays a similar cost (its
  harness 72.7 vs llama-bench 86.8 tok/s) but less of it. That is the
  next decode target.
- **Prefill** is the largest gap: TuckCore's Qwen3.5 path still prefills
  one token per step (Q35-2D).
- **Footprint** is still ~3.8× llama.cpp's charged footprint. Not yet
  explained.
- **Not established:** thermals beyond these 3 rounds; other prompts or
  tiers.
