# Baseline: Qwen3-0.6B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `6486c7dc0b3b` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3-0.6B-Q8_0.gguf`, 639446688 B, sha256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 3662 MiB | 0 MiB | 286 ms | 5.11 | 89.72 | 1862 ms | 969 MiB |
| 1 | B | llama-metal | ok | 3425 MiB | 0 MiB | 207 ms | 7.01 | 98.94 | 517 ms | 373 MiB |
| 2 | A | llama-metal | ok | 3282 MiB | 0 MiB | 209 ms | 6.97 | 99.88 | 450 ms | 374 MiB |
| 2 | B | tuckcore-metal | ok | 3396 MiB | 0 MiB | 290 ms | 5.04 | 89.46 | 1838 ms | 968 MiB |
| 3 | A | tuckcore-metal | ok | 3673 MiB | 0 MiB | 287 ms | 5.09 | 89.48 | 1825 ms | 969 MiB |
| 3 | B | llama-metal | ok | 3669 MiB | 0 MiB | 208 ms | 6.99 | 100.35 | 449 ms | 373 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 208 ms | 6.99 | 99.88 | 373 MiB |
| tuckcore-metal | 3 | 287 ms | 5.09 | 89.48 | 969 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.896 over 3 paired rounds (0.907, 0.896, 0.892)
- warm TTFT ratio (higher = slower prefill): median 1.38 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 2576.94 ± 0.36 | MTL,BLAS |
| tg256 | 118.36 ± 0.28 | MTL,BLAS |
- `/usr/bin/time -l`: 850395136  maximum resident set size
- `/usr/bin/time -l`: 221840728  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3-0.6B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
