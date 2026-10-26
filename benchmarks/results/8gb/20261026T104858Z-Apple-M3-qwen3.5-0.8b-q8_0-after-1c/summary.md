# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `261985054a9c` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2174 MiB | -909 MiB | 8447 ms | 18.84 | 49.59 | 12016 ms | 927 MiB |
| 1 | B | llama-metal | ok | 3010 MiB | -64 MiB | 284 ms | 65.30 | 70.13 | 1119 ms | 243 MiB |
| 2 | A | llama-metal | ok | 2675 MiB | -16 MiB | 294 ms | 67.30 | 72.68 | 695 ms | 242 MiB |
| 2 | B | tuckcore-metal | ok | 2635 MiB | -16 MiB | 7972 ms | 20.40 | 55.73 | 10405 ms | 928 MiB |
| 3 | A | tuckcore-metal | ok | 3030 MiB | -48 MiB | 7951 ms | 20.58 | 56.77 | 11027 ms | 1004 MiB |
| 3 | B | llama-metal | ok | 3557 MiB | 0 MiB | 269 ms | 71.06 | 76.59 | 1025 ms | 239 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 284 ms | 67.30 | 72.68 | 242 MiB |
| tuckcore-metal | 3 | 7972 ms | 20.40 | 55.73 | 928 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.741 over 3 paired rounds (0.707, 0.767, 0.741)
- warm TTFT ratio (higher = slower prefill): median 29.53 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1982.49 ± 6.88 | MTL,BLAS |
| tg256 | 86.77 ± 0.03 | MTL,BLAS |
- `/usr/bin/time -l`: 1025998848  maximum resident set size
- `/usr/bin/time -l`: 238569008  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
