# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `08eafb7d614c` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 3010 MiB | 0 MiB | 311 ms | 76.36 | 83.84 | 2356 ms | 972 MiB |
| 1 | B | llama-metal | ok | 3323 MiB | 0 MiB | 270 ms | 77.85 | 84.49 | 689 ms | 240 MiB |
| 2 | A | llama-metal | ok | 3304 MiB | 0 MiB | 286 ms | 61.95 | 66.34 | 602 ms | 242 MiB |
| 2 | B | tuckcore-metal | ok | 3293 MiB | 0 MiB | 311 ms | 76.74 | 84.30 | 2434 ms | 972 MiB |
| 3 | A | tuckcore-metal | ok | 3654 MiB | 0 MiB | 330 ms | 71.23 | 78.10 | 2435 ms | 972 MiB |
| 3 | B | llama-metal | ok | 3001 MiB | 0 MiB | 275 ms | 63.75 | 68.19 | 752 ms | 242 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 275 ms | 63.75 | 68.19 | 242 MiB |
| tuckcore-metal | 3 | 311 ms | 76.36 | 83.84 | 972 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.145 over 3 paired rounds (0.992, 1.271, 1.145)
- warm TTFT ratio (higher = slower prefill): median 1.15 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1986.00 ± 3.82 | MTL,BLAS |
| tg256 | 86.76 ± 0.07 | MTL,BLAS |
- `/usr/bin/time -l`: 1025228800  maximum resident set size
- `/usr/bin/time -l`: 238110256  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
