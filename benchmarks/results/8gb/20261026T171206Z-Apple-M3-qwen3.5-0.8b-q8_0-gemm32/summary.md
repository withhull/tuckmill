# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `45b838633bae` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 3080 MiB | 0 MiB | 490 ms | 60.44 | 68.09 | 2670 ms | 965 MiB |
| 1 | B | llama-metal | ok | 3422 MiB | -8 MiB | 288 ms | 63.35 | 67.97 | 697 ms | 239 MiB |
| 2 | A | llama-metal | ok | 3320 MiB | 0 MiB | 290 ms | 62.14 | 66.62 | 610 ms | 241 MiB |
| 2 | B | tuckcore-metal | ok | 3226 MiB | 0 MiB | 490 ms | 60.75 | 68.49 | 2612 ms | 965 MiB |
| 3 | A | tuckcore-metal | ok | 3530 MiB | 0 MiB | 490 ms | 60.74 | 68.48 | 2621 ms | 965 MiB |
| 3 | B | llama-metal | ok | 3523 MiB | 0 MiB | 280 ms | 61.99 | 66.26 | 608 ms | 241 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 288 ms | 62.14 | 66.62 | 241 MiB |
| tuckcore-metal | 3 | 490 ms | 60.74 | 68.48 | 965 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.028 over 3 paired rounds (1.002, 1.028, 1.033)
- warm TTFT ratio (higher = slower prefill): median 1.70 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1987.97 ± 2.24 | MTL,BLAS |
| tg256 | 86.76 ± 0.08 | MTL,BLAS |
- `/usr/bin/time -l`: 1029668864  maximum resident set size
- `/usr/bin/time -l`: 238323296  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
