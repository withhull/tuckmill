# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `537419dfe594` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 3108 MiB | -8 MiB | 378 ms | 62.16 | 68.19 | 2702 ms | 986 MiB |
| 1 | B | llama-metal | ok | 3560 MiB | 0 MiB | 269 ms | 78.05 | 84.71 | 730 ms | 240 MiB |
| 2 | A | llama-metal | ok | 3487 MiB | 0 MiB | 290 ms | 63.61 | 68.31 | 604 ms | 242 MiB |
| 2 | B | tuckcore-metal | ok | 3440 MiB | -24 MiB | 378 ms | 62.37 | 68.45 | 2522 ms | 986 MiB |
| 3 | A | tuckcore-metal | ok | 3581 MiB | 0 MiB | 379 ms | 62.40 | 68.49 | 2514 ms | 986 MiB |
| 3 | B | llama-metal | ok | 3684 MiB | 0 MiB | 283 ms | 63.09 | 67.58 | 607 ms | 242 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 283 ms | 63.61 | 68.31 | 242 MiB |
| tuckcore-metal | 3 | 378 ms | 62.37 | 68.45 | 986 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.002 over 3 paired rounds (0.805, 1.002, 1.013)
- warm TTFT ratio (higher = slower prefill): median 1.34 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1986.33 ± 3.28 | MTL,BLAS |
| tg256 | 86.80 ± 0.05 | MTL,BLAS |
- `/usr/bin/time -l`: 1030422528  maximum resident set size
- `/usr/bin/time -l`: 239748632  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
