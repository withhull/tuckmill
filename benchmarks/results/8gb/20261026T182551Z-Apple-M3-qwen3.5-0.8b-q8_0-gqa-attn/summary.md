# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `171e53c273bf` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2945 MiB | 0 MiB | 311 ms | 76.53 | 84.05 | 2617 ms | 973 MiB |
| 1 | B | llama-metal | ok | 3420 MiB | 0 MiB | 285 ms | 62.75 | 67.23 | 729 ms | 241 MiB |
| 2 | A | llama-metal | ok | 3222 MiB | 0 MiB | 283 ms | 61.53 | 65.80 | 608 ms | 242 MiB |
| 2 | B | tuckcore-metal | ok | 3276 MiB | 0 MiB | 311 ms | 76.75 | 84.31 | 2450 ms | 972 MiB |
| 3 | A | tuckcore-metal | ok | 3605 MiB | -16 MiB | 311 ms | 76.73 | 84.28 | 2439 ms | 972 MiB |
| 3 | B | llama-metal | ok | 3495 MiB | 0 MiB | 279 ms | 62.57 | 66.93 | 616 ms | 242 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 283 ms | 62.57 | 66.93 | 242 MiB |
| tuckcore-metal | 3 | 311 ms | 76.73 | 84.28 | 972 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.259 over 3 paired rounds (1.250, 1.281, 1.259)
- warm TTFT ratio (higher = slower prefill): median 1.10 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1985.45 ± 2.64 | MTL,BLAS |
| tg256 | 86.77 ± 0.01 | MTL,BLAS |
- `/usr/bin/time -l`: 1028718592  maximum resident set size
- `/usr/bin/time -l`: 239175240  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
