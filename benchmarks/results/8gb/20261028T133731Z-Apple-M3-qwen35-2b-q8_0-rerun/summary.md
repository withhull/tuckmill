# Baseline: Qwen3.5-2B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `ab44d6869098` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-2B-Q8_0.gguf`, 2012012800 B, sha256 `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | discarded-swap | 2000 MiB | 73 MiB | n/a | n/a | n/a | n/a | n/a |
| 1 | A | tuckcore-metal | ok | 3585 MiB | 0 MiB | 623 ms | 36.40 | 39.79 | 2125 ms | 2148 MiB |
| 1 | B | llama-metal | ok | 3539 MiB | 0 MiB | 597 ms | 35.07 | 38.05 | 2095 ms | 247 MiB |
| 2 | A | llama-metal | discarded-swap | 3299 MiB | 129 MiB | n/a | n/a | n/a | n/a | n/a |
| 2 | A | llama-metal | discarded-swap | 2745 MiB | 150 MiB | n/a | n/a | n/a | n/a | n/a |
| 2 | B | tuckcore-metal | discarded-swap | 3050 MiB | 184 MiB | n/a | n/a | n/a | n/a | n/a |
| 2 | B | tuckcore-metal | ok | 2889 MiB | -46 MiB | 1262 ms | 20.43 | 22.51 | 2500 ms | 2148 MiB |
| 3 | A | tuckcore-metal | ok | 3399 MiB | -8 MiB | 1578 ms | 20.31 | 23.15 | 2773 ms | 2148 MiB |
| 3 | B | llama-metal | discarded-swap | 3174 MiB | 215 MiB | n/a | n/a | n/a | n/a | n/a |
| 3 | B | llama-metal | discarded-swap | 2635 MiB | 84 MiB | n/a | n/a | n/a | n/a | n/a |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 1 | 597 ms | 35.07 | 38.05 | 247 MiB |
| tuckcore-metal | 3 | 1262 ms | 20.43 | 23.15 | 2148 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.046 over 1 paired rounds (1.046)
- warm TTFT ratio (higher = slower prefill): median 1.04 over 1 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-2B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
