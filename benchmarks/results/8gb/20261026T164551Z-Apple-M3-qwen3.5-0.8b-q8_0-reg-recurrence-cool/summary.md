# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `64f60c7b4e08` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | discarded-swap | 2857 MiB | 2045 MiB | n/a | n/a | n/a | n/a | n/a |
| 1 | A | tuckcore-metal | ok | 4368 MiB | -304 MiB | 628 ms | 54.07 | 62.10 | 2957 ms | 961 MiB |
| 1 | B | llama-metal | ok | 3424 MiB | -56 MiB | 309 ms | 59.42 | 63.77 | 877 ms | 243 MiB |
| 2 | A | llama-metal | ok | 2487 MiB | -183 MiB | 298 ms | 65.21 | 70.31 | 790 ms | 240 MiB |
| 2 | B | tuckcore-metal | ok | 2365 MiB | -24 MiB | 632 ms | 54.44 | 62.65 | 2900 ms | 961 MiB |
| 3 | A | tuckcore-metal | ok | 2900 MiB | 0 MiB | 630 ms | 54.52 | 62.72 | 2847 ms | 961 MiB |
| 3 | B | llama-metal | ok | 3021 MiB | -16 MiB | 299 ms | 66.44 | 71.76 | 908 ms | 241 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 299 ms | 65.21 | 70.31 | 241 MiB |
| tuckcore-metal | 3 | 630 ms | 54.44 | 62.65 | 961 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.891 over 3 paired rounds (0.974, 0.891, 0.874)
- warm TTFT ratio (higher = slower prefill): median 2.11 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1807.40 ± 38.11 | MTL,BLAS |
| tg256 | 75.72 ± 6.10 | MTL,BLAS |
- `/usr/bin/time -l`: 997638144  maximum resident set size
- `/usr/bin/time -l`: 238863944  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
