# Review notes: Qwen3.5-0.8B Q8_0, MacBook Air M3 8 GB (2026-10-26)

Follows [../../TEMPLATE.md](../../TEMPLATE.md). Numbers come from this
directory's `summary.md`, generated from the unmodified records. All are
**locally-calibrated**.

## Machine

- 8 GB tier, Apple M3 (MacBook Air, fanless), macOS 27.0.1 (26A434), AC
  power. Other apps were open: reclaimable memory before each run was
  1.9–2.5 GiB, and about 2.4 GiB of other apps' memory sat in the
  compressor at start.
- tuckmill `5c6e889` (clean tree), llama.cpp b11695 (pinned).

## Artifact

- `Qwen3.5-0.8B-Q8_0.gguf`, 833,592,096 B, sha256
  `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
  (pinned, ADR-008). Architecture qwen35, Q8_0, context 2048.

## Protocol

- 512 in / 256 out, 3 repetitions per invocation, 3 ABBA rounds,
  identical frozen token ids for both harness engines.
- **Discarded:** llama-metal rounds 1 and 3 grew swap by 418 / 458 MiB
  (limit 64). The asymmetry is itself a finding: on this full 8 GB
  machine, llama.cpp's load pushes other apps' pages into swap, while
  TuckCore's runs released swap (−377 / −64 / −405 MiB). So the swap rule
  removes llama.cpp rounds more often than TuckCore's. One valid paired
  round (round 2) remains. The pinned llama-bench row (3 repetitions)
  agrees with it: tg256 69.0 ± 3.4 vs 71.9 decode tok/s in the harness.
- Warm (repetitions 2–3) and cold-load (repetition 1) are kept separate.
- **Label caveat:** the llama.cpp records carry the identity
  `llama.cpp-cpu`. That was a hard-coded label: the backend reported a CPU
  identity even with `--gpu-layers 99`. The run executed on Metal (the
  harness decode rate matches the Metal llama-bench row; llama.cpp's CPU
  path on this machine measured ≈29–35 e2e tok/s, on Qwen3-0.6B). It is fixed in
  the commit that adds these records (`llama.cpp:ngl=99:metal`). The raw
  records are left unedited.

## Results (locally-calibrated)

| engine | valid rounds | warm TTFT (512 tok) | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 88.9 s | 1.93 | 5.55 (3.32–5.80) | 953 MiB |
| llama-metal | 1 | 0.29 s | 66.76 | 71.94 | 242 MiB |

- Paired decode ratio TuckCore / llama.cpp: **0.046** (round 2; that
  round's TuckCore decode, 3.32, was its slowest, so TuckCore's median of
  5.55 gives ≈0.077).
- Paired TTFT ratio: **≈308×** slower prefill.
- llama-bench vendor row: pp512 1854.9 ± 2.8 tok/s, tg256 69.0 ± 3.4
  tok/s; peak memory footprint 240,584,240 B.
- MLX: mlx 0.32.4 / mlx-lm 0.32.2 installed, but it cannot load the
  identical GGUF, so there is no MLX row (four-way identity).

## Reading

- **Decode** is ~13× slower than llama.cpp Metal (5.55 vs 71.9 tok/s).
  **Prefill** is ~300× slower: TuckCore's Qwen3.5 path prefills one token
  per full forward pass (~174 ms per token; Q35-2D not built), while
  llama.cpp batches the prompt.
- **Footprint** is ~4× llama.cpp's charged footprint (953 vs 242 MiB).
  The two are measured differently: TuckCore's worker footprint includes
  GPU buffers created from host allocations, and llama.cpp maps the file.
  Phase 1 must explain this gap before it is compared.
- **Where Phase 1 starts:** profile the decode step (per-op command
  buffers and waits, the full-vocabulary logits download each token, the
  one-thread-per-head recurrence) and build batched prefill. Kernel ports
  come after the profile says kernels are the bottleneck.
- **Not established:** thermals (fanless, uncontrolled), other prompts,
  any other tier, cold-storage behavior.
