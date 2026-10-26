# Review notes: repeat at `6b6a014`, idle machine, Qwen3.5-0.8B Q8_0, M3 8 GB

Follows [../../TEMPLATE.md](../../TEMPLATE.md). All numbers are
**locally-calibrated**. Repeat of `20261026T164551Z-…` on the same
commit, run in the foreground when the user confirmed the Mac idle: 1-min
load 1.95, reclaimable 3.8 GiB, compressor 0.43 GiB, no swap growth in any
invocation, all 6 valid on the first attempt. llama.cpp was at full speed
(llama-bench tg256 86.88 ± 0.09, the highest reading of the day).

## Results

| engine | rounds | warm TTFT (512 tok) | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 0.588 s | 59.36 | 68.47 (68.36–68.52) | 961 MiB |
| llama-metal | 3 | 0.278 s | 63.66 | 68.30 (66.25–77.62) | 242 MiB |

- Paired decode ratio: median **1.001** (1.001, 0.883, 1.034). TuckCore is
  very steady (±0.1 tok/s); llama.cpp varied 66.3–77.6 across rounds.
  Against llama.cpp's best round, TuckCore is at 0.88.
- Paired TTFT ratio: **2.12×** (0.588 vs 0.278 s).
- llama-bench (its own tokens, no harness overhead): pp512 1985.9,
  tg256 86.9.

## Reproducibility (same commit, two runs)

| run | TuckCore decode | llama.cpp decode | paired decode | TuckCore TTFT | paired TTFT |
| --- | --- | --- | --- | --- | --- |
| `20261026T164551Z` (cool) | 62.65 | 70.31 | 0.891 | 0.630 s | 2.11× |
| `20261026T165640Z` (idle) | 68.47 | 68.30 | 1.001 | 0.588 s | 2.12× |

The prefill ratio reproduces (2.1×). The decode ratio moved 0.89 → 1.00
with machine state: TuckCore itself rose 62.7 → 68.5 on the same code.

## Reading

- **Decode** is at llama.cpp's level in this harness (0.88–1.03 paired
  across rounds, median 1.00), and it reproduced at ≥ 0.87 in both runs.
  Treat it as "about 90–100% depending on machine state", not as a lead.
- **Prefill** is 2.1× behind (~47% of llama.cpp), reproduced. It is the
  remaining Phase 1 gap on the ~1B row.
