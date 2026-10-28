# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `ab44d6869098` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2374 MiB | 0 MiB | 290 ms | 72.31 | 78.48 | 928 ms | 997 MiB |
| 1 | B | llama-metal | ok | 3063 MiB | -32 MiB | 291 ms | 64.24 | 69.06 | 844 ms | 240 MiB |
| 2 | A | llama-metal | ok | 3108 MiB | -24 MiB | 287 ms | 62.59 | 67.08 | 608 ms | 243 MiB |
| 2 | B | tuckcore-metal | ok | 2337 MiB | -16 MiB | 279 ms | 75.83 | 82.35 | 1139 ms | 997 MiB |
| 3 | A | tuckcore-metal | ok | 2637 MiB | -8 MiB | 274 ms | 76.18 | 82.62 | 852 ms | 997 MiB |
| 3 | B | llama-metal | ok | 2723 MiB | 0 MiB | 277 ms | 62.97 | 67.35 | 1098 ms | 242 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 287 ms | 62.97 | 67.35 | 242 MiB |
| tuckcore-metal | 3 | 279 ms | 75.83 | 82.35 | 997 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.227 over 3 paired rounds (1.136, 1.228, 1.227)
- warm TTFT ratio (higher = slower prefill): median 0.99 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
