# Baseline: gemma-3-1b-it-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `ab44d6869098` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q8_0.gguf`, 1069306368 B, sha256 `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2680 MiB | -296 MiB | 698 ms | 43.30 | 48.91 | 1137 ms | 1227 MiB |
| 1 | B | llama-metal | ok | 3021 MiB | -40 MiB | 550 ms | 46.86 | 51.89 | 1552 ms | 199 MiB |
| 2 | A | llama-metal | ok | 2723 MiB | 0 MiB | 537 ms | 46.01 | 50.71 | 871 ms | 200 MiB |
| 2 | B | tuckcore-metal | ok | 2740 MiB | -24 MiB | 874 ms | 39.31 | 45.24 | 1188 ms | 1227 MiB |
| 3 | A | tuckcore-metal | discarded-swap | 3241 MiB | 265 MiB | n/a | n/a | n/a | n/a | n/a |
| 3 | A | tuckcore-metal | discarded-swap | 1706 MiB | 568 MiB | n/a | n/a | n/a | n/a | n/a |
| 3 | B | llama-metal | discarded-swap | 2255 MiB | 903 MiB | n/a | n/a | n/a | n/a | n/a |
| 3 | B | llama-metal | discarded-swap | 1965 MiB | 185 MiB | n/a | n/a | n/a | n/a | n/a |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 2 | 544 ms | 46.43 | 51.30 | 199 MiB |
| tuckcore-metal | 2 | 786 ms | 41.30 | 47.07 | 1227 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.917 over 2 paired rounds (0.943, 0.892)
- warm TTFT ratio (higher = slower prefill): median 1.45 over 2 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
