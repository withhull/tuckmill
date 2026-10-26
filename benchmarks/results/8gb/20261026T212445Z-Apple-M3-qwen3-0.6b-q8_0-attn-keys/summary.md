# Baseline: Qwen3-0.6B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `4b7d2ae2702e` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3-0.6B-Q8_0.gguf`, 639446688 B, sha256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2958 MiB | 0 MiB | 255 ms | 5.72 | 98.85 | 1933 ms | 968 MiB |
| 1 | B | llama-metal | ok | 3305 MiB | 0 MiB | 220 ms | 6.51 | 34.37 | 541 ms | 375 MiB |
| 2 | A | llama-metal | ok | 3246 MiB | 0 MiB | 212 ms | 6.77 | 70.94 | 442 ms | 374 MiB |
| 2 | B | tuckcore-metal | ok | 3140 MiB | 0 MiB | 254 ms | 5.74 | 99.65 | 1790 ms | 968 MiB |
| 3 | A | tuckcore-metal | ok | 3404 MiB | 0 MiB | 252 ms | 5.80 | 99.17 | 1790 ms | 968 MiB |
| 3 | B | llama-metal | ok | 3399 MiB | 0 MiB | 207 ms | 7.00 | 99.74 | 449 ms | 373 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 212 ms | 6.77 | 70.94 | 374 MiB |
| tuckcore-metal | 3 | 254 ms | 5.74 | 99.17 | 968 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.405 over 3 paired rounds (2.876, 1.405, 0.994)
- warm TTFT ratio (higher = slower prefill): median 1.20 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 2576.40 ± 0.50 | MTL,BLAS |
| tg256 | 117.94 ± 0.23 | MTL,BLAS |
- `/usr/bin/time -l`: 847953920  maximum resident set size
- `/usr/bin/time -l`: 221168960  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3-0.6B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
