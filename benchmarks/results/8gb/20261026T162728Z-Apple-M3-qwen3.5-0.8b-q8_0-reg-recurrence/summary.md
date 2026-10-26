# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `6b6a0146caaf` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2023 MiB | -40 MiB | 683 ms | 50.92 | 58.71 | 3837 ms | 962 MiB |
| 1 | B | llama-metal | ok | 3228 MiB | -16 MiB | 367 ms | 63.39 | 69.47 | 1212 ms | 239 MiB |
| 2 | A | llama-metal | ok | 2924 MiB | 0 MiB | 461 ms | 57.83 | 64.33 | 704 ms | 243 MiB |
| 2 | B | tuckcore-metal | ok | 3256 MiB | -144 MiB | 721 ms | 50.29 | 58.25 | 2839 ms | 961 MiB |
| 3 | A | tuckcore-metal | ok | 3253 MiB | -8 MiB | 1215 ms | 43.04 | 53.71 | 2929 ms | 961 MiB |
| 3 | B | llama-metal | ok | 3197 MiB | 0 MiB | 434 ms | 46.35 | 50.11 | 752 ms | 241 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 434 ms | 57.83 | 64.33 | 241 MiB |
| tuckcore-metal | 3 | 721 ms | 50.29 | 58.25 | 961 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.905 over 3 paired rounds (0.845, 0.905, 1.072)
- warm TTFT ratio (higher = slower prefill): median 1.86 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1781.33 ± 16.64 | MTL,BLAS |
| tg256 | 52.96 ± 2.48 | MTL,BLAS |
- `/usr/bin/time -l`: 1026080768  maximum resident set size
- `/usr/bin/time -l`: 238700104  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
