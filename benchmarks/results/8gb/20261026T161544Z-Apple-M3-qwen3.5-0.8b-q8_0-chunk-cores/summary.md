# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `6ba9823c3d89` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2320 MiB | 0 MiB | 835 ms | 50.02 | 59.54 | 3290 ms | 948 MiB |
| 1 | B | llama-metal | ok | 2840 MiB | -32 MiB | 294 ms | 65.84 | 70.97 | 850 ms | 240 MiB |
| 2 | A | llama-metal | ok | 2384 MiB | -27 MiB | 295 ms | 63.13 | 67.84 | 821 ms | 242 MiB |
| 2 | B | tuckcore-metal | ok | 2598 MiB | 0 MiB | 823 ms | 50.99 | 60.77 | 3199 ms | 947 MiB |
| 3 | A | tuckcore-metal | ok | 2788 MiB | -8 MiB | 850 ms | 48.64 | 57.80 | 3164 ms | 948 MiB |
| 3 | B | llama-metal | ok | 3158 MiB | -8 MiB | 294 ms | 64.65 | 69.59 | 1169 ms | 241 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 294 ms | 64.65 | 69.59 | 241 MiB |
| tuckcore-metal | 3 | 835 ms | 50.02 | 59.54 | 948 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.839 over 3 paired rounds (0.839, 0.896, 0.831)
- warm TTFT ratio (higher = slower prefill): median 2.84 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1847.59 ± 14.54 | MTL,BLAS |
| tg256 | 77.02 ± 2.09 | MTL,BLAS |
- `/usr/bin/time -l`: 947453952  maximum resident set size
- `/usr/bin/time -l`: 238274096  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
