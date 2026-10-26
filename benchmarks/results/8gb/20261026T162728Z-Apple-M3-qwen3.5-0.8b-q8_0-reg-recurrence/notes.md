# Review notes: register-resident recurrence (`6b6a014`), Qwen3.5-0.8B Q8_0, M3 8 GB

Follows [../../TEMPLATE.md](../../TEMPLATE.md). All numbers are
**locally-calibrated**. All 6 invocations are valid (no swap growth), but
**the machine throttled during the run**: llama.cpp's own decode fell
69.47 → 64.33 → 50.11 tok/s across rounds 1–3, its TTFT rose from the
usual ~0.29 s to 0.37–0.46 s, and llama-bench tg256 read 52.96 ± 2.48
(77.02 in the previous run). Memory was also tighter at the start
(compressor ~2.3 GiB).

## Results

| engine | rounds | warm TTFT (512 tok) | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 0.721 s | 50.29 | 58.25 (53.71–58.71) | 961 MiB |
| llama-metal | 3 | 0.434 s | 57.83 | 64.33 (50.11–69.47) | 241 MiB |

- Paired decode ratio: median 0.905 (0.845, 0.905, 1.072). Round 3's
  1.072 reflects llama.cpp's throttled 50.11 tok/s, not a TuckCore lead.
- Paired TTFT ratio: median 1.86 (previous run 2.84).

## Reading

- **Best-conditioned pair (round 1):** decode 58.71 vs 69.47 tok/s
  (0.845), TTFT 0.683 vs 0.367 s (1.86×). Read decode as ~0.85, unchanged
  from the previous run within noise. The register recurrence mainly
  speeds prefill; its decode effect (~0.8 ms/token in-process) is within
  this run's noise.
- **Prefill:** TuckCore warm TTFT 0.835 → 0.683–0.721 s in rounds 1–2,
  consistent with the in-process 879 → 646 ms. The ratio to llama.cpp is
  ~1.9× in the best pair, but llama.cpp's TTFT was itself inflated this
  run, so ~2× is the honest reading.
- **Not established:** a cool-machine comparison for this commit. Rerun
  after a cool-down before quoting a ratio.
