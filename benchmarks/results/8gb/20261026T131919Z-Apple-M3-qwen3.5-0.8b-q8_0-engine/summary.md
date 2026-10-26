# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- tuckmill `8fe1536b84a9` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2298 MiB | -88 MiB | 1633 ms | 41.71 | 56.60 | 4862 ms | 1109 MiB |
| 1 | B | llama-metal | ok | 3227 MiB | -24 MiB | 292 ms | 60.56 | 64.89 | 1149 ms | 242 MiB |
| 2 | A | llama-metal | discarded-swap | 2059 MiB | 197 MiB | n/a | n/a | n/a | n/a | n/a |
| 2 | A | llama-metal | ok | 1924 MiB | 0 MiB | 305 ms | 61.08 | 65.64 | 1340 ms | 240 MiB |
| 2 | B | tuckcore-metal | discarded-swap | 2135 MiB | 1356 MiB | n/a | n/a | n/a | n/a | n/a |
| 2 | B | tuckcore-metal | ok | 2621 MiB | -296 MiB | 1734 ms | 39.70 | 54.12 | 4698 ms | 1032 MiB |
| 3 | A | tuckcore-metal | ok | 3068 MiB | -40 MiB | 1710 ms | 39.04 | 52.61 | 4669 ms | 1110 MiB |
| 3 | B | llama-metal | ok | 3032 MiB | -16 MiB | 291 ms | 62.18 | 66.68 | 1173 ms | 241 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 292 ms | 61.08 | 65.64 | 241 MiB |
| tuckcore-metal | 3 | 1710 ms | 39.70 | 54.12 | 1109 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.824 over 3 paired rounds (0.872, 0.824, 0.789)
- warm TTFT ratio (higher = slower prefill): median 5.69 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1836.52 ± 12.98 | MTL,BLAS |
| tg256 | 60.58 ± 15.00 | MTL,BLAS |
- `/usr/bin/time -l`: 981614592  maximum resident set size
- `/usr/bin/time -l`: 240191000  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
