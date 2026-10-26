# Baseline: Qwen3.5-2B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `e09f393dd43b` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-2B-Q8_0.gguf`, 2012012800 B, sha256 `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2647 MiB | 0 MiB | 624 ms | 36.07 | 39.39 | 6056 ms | 2123 MiB |
| 1 | B | llama-metal | ok | 3637 MiB | 0 MiB | 588 ms | 35.28 | 38.25 | 1742 ms | 247 MiB |
| 2 | A | llama-metal | ok | 3476 MiB | 0 MiB | 587 ms | 37.23 | 40.55 | 987 ms | 249 MiB |
| 2 | B | tuckcore-metal | ok | 3591 MiB | 0 MiB | 623 ms | 36.03 | 39.35 | 5790 ms | 2123 MiB |
| 3 | A | tuckcore-metal | ok | 3597 MiB | 0 MiB | 622 ms | 36.07 | 39.39 | 5775 ms | 2123 MiB |
| 3 | B | llama-metal | ok | 3594 MiB | -99 MiB | 851 ms | 31.32 | 34.79 | 1743 ms | 246 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 588 ms | 35.28 | 38.25 | 247 MiB |
| tuckcore-metal | 3 | 623 ms | 36.07 | 39.39 | 2123 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.030 over 3 paired rounds (1.030, 0.970, 1.132)
- warm TTFT ratio (higher = slower prefill): median 1.06 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 741.29 ± 82.96 | MTL,BLAS |
| tg256 | 15.05 ± 2.04 | MTL,BLAS |
- `/usr/bin/time -l`: 2103279616  maximum resident set size
- `/usr/bin/time -l`: 251890040  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-2B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
