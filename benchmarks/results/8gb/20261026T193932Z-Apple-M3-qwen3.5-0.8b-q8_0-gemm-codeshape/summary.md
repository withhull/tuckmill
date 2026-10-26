# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `d95ecc16b360` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 3446 MiB | 0 MiB | 272 ms | 77.49 | 84.12 | 2754 ms | 973 MiB |
| 1 | B | llama-metal | ok | 3543 MiB | -8 MiB | 269 ms | 78.04 | 84.69 | 706 ms | 241 MiB |
| 2 | A | llama-metal | ok | 3508 MiB | 0 MiB | 270 ms | 70.42 | 75.87 | 605 ms | 241 MiB |
| 2 | B | tuckcore-metal | ok | 3298 MiB | 0 MiB | 272 ms | 77.64 | 84.30 | 2399 ms | 972 MiB |
| 3 | A | tuckcore-metal | ok | 3902 MiB | 0 MiB | 272 ms | 77.72 | 84.39 | 2389 ms | 972 MiB |
| 3 | B | llama-metal | ok | 3897 MiB | 0 MiB | 287 ms | 63.30 | 67.90 | 609 ms | 241 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 270 ms | 70.42 | 75.87 | 241 MiB |
| tuckcore-metal | 3 | 272 ms | 77.64 | 84.30 | 972 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.111 over 3 paired rounds (0.993, 1.111, 1.243)
- warm TTFT ratio (higher = slower prefill): median 1.01 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1986.61 ± 0.86 | MTL,BLAS |
| tg256 | 86.73 ± 0.07 | MTL,BLAS |
- `/usr/bin/time -l`: 1035829248  maximum resident set size
- `/usr/bin/time -l`: 238388784  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
