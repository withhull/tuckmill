# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- tuckmill `b5899a143da6` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 3111 MiB | 0 MiB | 300 ms | 76.85 | 84.13 | 2881 ms | 972 MiB |
| 1 | B | llama-metal | ok | 3481 MiB | 0 MiB | 269 ms | 74.24 | 80.26 | 691 ms | 239 MiB |
| 2 | A | llama-metal | ok | 3456 MiB | 0 MiB | 270 ms | 77.75 | 84.38 | 600 ms | 243 MiB |
| 2 | B | tuckcore-metal | ok | 3462 MiB | 0 MiB | 299 ms | 77.00 | 84.30 | 2418 ms | 972 MiB |
| 3 | A | tuckcore-metal | ok | 3906 MiB | 0 MiB | 300 ms | 77.02 | 84.33 | 2423 ms | 972 MiB |
| 3 | B | llama-metal | ok | 3904 MiB | 0 MiB | 289 ms | 62.60 | 67.13 | 610 ms | 242 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 270 ms | 74.24 | 80.26 | 242 MiB |
| tuckcore-metal | 3 | 300 ms | 77.00 | 84.30 | 972 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.048 over 3 paired rounds (1.048, 0.999, 1.256)
- warm TTFT ratio (higher = slower prefill): median 1.11 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1986.92 ± 1.14 | MTL,BLAS |
| tg256 | 86.87 ± 0.05 | MTL,BLAS |
- `/usr/bin/time -l`: 1032929280  maximum resident set size
- `/usr/bin/time -l`: 239846984  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
