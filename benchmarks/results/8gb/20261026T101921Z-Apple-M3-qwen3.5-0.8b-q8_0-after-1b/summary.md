# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `dafeadd03f16` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | discarded-swap | 1804 MiB | 136 MiB | n/a | n/a | n/a | n/a | n/a |
| 1 | B | llama-metal | ok | 2710 MiB | -192 MiB | 282 ms | 66.30 | 71.27 | 1237 ms | 241 MiB |
| 2 | A | llama-metal | discarded-swap | 2092 MiB | 251 MiB | n/a | n/a | n/a | n/a | n/a |
| 2 | B | tuckcore-metal | ok | 2126 MiB | -23 MiB | 17677 ms | 10.56 | 34.59 | 20177 ms | 953 MiB |
| 3 | A | tuckcore-metal | discarded-swap | 2033 MiB | 462 MiB | n/a | n/a | n/a | n/a | n/a |
| 3 | B | llama-metal | discarded-swap | 2415 MiB | 303 MiB | n/a | n/a | n/a | n/a | n/a |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 1 | 282 ms | 66.30 | 71.27 | 241 MiB |
| tuckcore-metal | 1 | 17677 ms | 10.56 | 34.59 | 953 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median n/a over 0 paired rounds (none)
- warm TTFT ratio (higher = slower prefill): median n/a over 0 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1937.84 ± 15.21 | MTL,BLAS |
| tg256 | 46.49 ± 4.08 | MTL,BLAS |
- `/usr/bin/time -l`: 906739712  maximum resident set size
- `/usr/bin/time -l`: 237291056  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
