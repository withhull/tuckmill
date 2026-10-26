# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `5c6e88962ba7` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2519 MiB | -377 MiB | 88933 ms | 1.93 | 5.80 | 101963 ms | 953 MiB |
| 1 | B | llama-metal | discarded-swap | 2027 MiB | 418 MiB | n/a | n/a | n/a | n/a | n/a |
| 2 | A | llama-metal | ok | 1922 MiB | 28 MiB | 289 ms | 66.76 | 71.94 | 858 ms | 242 MiB |
| 2 | B | tuckcore-metal | ok | 2101 MiB | -64 MiB | 89001 ms | 1.33 | 3.32 | 170331 ms | 951 MiB |
| 3 | A | tuckcore-metal | ok | 1903 MiB | -405 MiB | 86393 ms | 1.93 | 5.55 | 123162 ms | 990 MiB |
| 3 | B | llama-metal | discarded-swap | 2395 MiB | 458 MiB | n/a | n/a | n/a | n/a | n/a |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 1 | 289 ms | 66.76 | 71.94 | 242 MiB |
| tuckcore-metal | 3 | 88933 ms | 1.93 | 5.55 | 953 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.046 over 1 paired rounds (0.046)
- warm TTFT ratio (higher = slower prefill): median 307.72 over 1 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1854.90 ± 2.75 | MTL,BLAS |
| tg256 | 69.01 ± 3.43 | MTL,BLAS |
- `/usr/bin/time -l`: 927596544  maximum resident set size
- `/usr/bin/time -l`: 240584240  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
