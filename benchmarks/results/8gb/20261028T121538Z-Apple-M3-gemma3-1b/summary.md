# Baseline: gemma-3-1b-it-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- tuckmill `0590455f3a25` (dirty tree: True), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q8_0.gguf`, 1069306368 B, sha256 `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2759 MiB | -64 MiB | 367 ms | 53.17 | 57.34 | 1368 ms | 1227 MiB |
| 1 | B | llama-metal | discarded-swap | 2741 MiB | 353 MiB | n/a | n/a | n/a | n/a | n/a |
| 1 | B | llama-metal | ok | 2273 MiB | -36 MiB | 328 ms | 59.63 | 64.32 | 967 ms | 198 MiB |
| 2 | A | llama-metal | ok | 2568 MiB | -80 MiB | 331 ms | 59.04 | 63.68 | 682 ms | 199 MiB |
| 2 | B | tuckcore-metal | ok | 2668 MiB | -24 MiB | 368 ms | 52.92 | 57.06 | 1100 ms | 1226 MiB |
| 3 | A | tuckcore-metal | ok | 2966 MiB | -88 MiB | 374 ms | 51.87 | 55.92 | 1152 ms | 1227 MiB |
| 3 | B | llama-metal | ok | 2996 MiB | -16 MiB | 337 ms | 57.68 | 62.20 | 1345 ms | 198 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 331 ms | 59.04 | 63.68 | 198 MiB |
| tuckcore-metal | 3 | 368 ms | 52.92 | 57.06 | 1227 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.896 over 3 paired rounds (0.891, 0.896, 0.899)
- warm TTFT ratio (higher = slower prefill): median 1.11 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
