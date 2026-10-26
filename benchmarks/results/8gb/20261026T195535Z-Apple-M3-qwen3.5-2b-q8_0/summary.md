# Baseline: Qwen3.5-2B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `2e9335a973b1` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-2B-Q8_0.gguf`, 2012012800 B, sha256 `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2704 MiB | 0 MiB | 623 ms | 36.03 | 39.34 | 5823 ms | 2123 MiB |
| 1 | B | llama-metal | ok | 3668 MiB | 0 MiB | 587 ms | 37.20 | 40.52 | 1767 ms | 248 MiB |
| 2 | A | llama-metal | ok | 3510 MiB | 0 MiB | 587 ms | 37.21 | 40.53 | 977 ms | 248 MiB |
| 2 | B | tuckcore-metal | ok | 3595 MiB | 0 MiB | 625 ms | 35.44 | 38.65 | 5830 ms | 2123 MiB |
| 3 | A | tuckcore-metal | ok | 3357 MiB | -16 MiB | 627 ms | 35.96 | 39.28 | 6338 ms | 2123 MiB |
| 3 | B | llama-metal | ok | 3814 MiB | 0 MiB | 814 ms | 33.98 | 37.96 | 1984 ms | 249 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 587 ms | 37.20 | 40.52 | 248 MiB |
| tuckcore-metal | 3 | 625 ms | 35.96 | 39.28 | 2123 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.971 over 3 paired rounds (0.971, 0.954, 1.035)
- warm TTFT ratio (higher = slower prefill): median 1.06 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 880.16 ± 0.82 | MTL,BLAS |
| tg256 | 29.74 ± 2.82 | MTL,BLAS |
- `/usr/bin/time -l`: 2105704448  maximum resident set size
- `/usr/bin/time -l`: 251398544  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-2B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
