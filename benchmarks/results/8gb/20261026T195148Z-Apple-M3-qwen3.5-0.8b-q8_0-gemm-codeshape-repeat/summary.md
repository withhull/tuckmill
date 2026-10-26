# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `6392d662c6ff` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2793 MiB | 0 MiB | 273 ms | 76.75 | 83.29 | 2406 ms | 972 MiB |
| 1 | B | llama-metal | ok | 3267 MiB | 0 MiB | 300 ms | 63.26 | 68.10 | 715 ms | 241 MiB |
| 2 | A | llama-metal | ok | 3159 MiB | -8 MiB | 285 ms | 62.34 | 66.75 | 620 ms | 241 MiB |
| 2 | B | tuckcore-metal | ok | 3133 MiB | 0 MiB | 272 ms | 77.65 | 84.31 | 2407 ms | 972 MiB |
| 3 | A | tuckcore-metal | ok | 3452 MiB | 0 MiB | 272 ms | 77.59 | 84.24 | 2395 ms | 972 MiB |
| 3 | B | llama-metal | ok | 3388 MiB | -8 MiB | 288 ms | 62.86 | 67.39 | 644 ms | 241 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 288 ms | 62.86 | 67.39 | 241 MiB |
| tuckcore-metal | 3 | 272 ms | 77.59 | 84.24 | 972 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.250 over 3 paired rounds (1.223, 1.263, 1.250)
- warm TTFT ratio (higher = slower prefill): median 0.94 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1985.75 ± 2.89 | MTL,BLAS |
| tg256 | 86.78 ± 0.06 | MTL,BLAS |
- `/usr/bin/time -l`: 1025507328  maximum resident set size
- `/usr/bin/time -l`: 238667360  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
