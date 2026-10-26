# Baseline: Qwen3.5-0.8B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `6bb7d9955f7c` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q8_0.gguf`, 833592096 B, sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 3812 MiB | 0 MiB | 588 ms | 59.29 | 68.36 | 2647 ms | 961 MiB |
| 1 | B | llama-metal | ok | 3417 MiB | 0 MiB | 286 ms | 63.66 | 68.30 | 697 ms | 240 MiB |
| 2 | A | llama-metal | ok | 3445 MiB | 0 MiB | 270 ms | 71.95 | 77.62 | 600 ms | 242 MiB |
| 2 | B | tuckcore-metal | ok | 3416 MiB | -8 MiB | 588 ms | 59.40 | 68.52 | 2721 ms | 961 MiB |
| 3 | A | tuckcore-metal | ok | 3781 MiB | 0 MiB | 589 ms | 59.36 | 68.47 | 2711 ms | 961 MiB |
| 3 | B | llama-metal | ok | 3775 MiB | -8 MiB | 278 ms | 62.00 | 66.25 | 616 ms | 242 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 278 ms | 63.66 | 68.30 | 242 MiB |
| tuckcore-metal | 3 | 588 ms | 59.36 | 68.47 | 961 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.001 over 3 paired rounds (1.001, 0.883, 1.034)
- warm TTFT ratio (higher = slower prefill): median 2.12 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1985.93 ± 0.84 | MTL,BLAS |
| tg256 | 86.88 ± 0.09 | MTL,BLAS |
- `/usr/bin/time -l`: 1037303808  maximum resident set size
- `/usr/bin/time -l`: 238323272  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
