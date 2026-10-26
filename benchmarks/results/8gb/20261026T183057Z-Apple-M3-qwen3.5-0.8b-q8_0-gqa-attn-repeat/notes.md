# Repeat at 171e53c (idle machine): prefill result not confirmed

- Command: `scripts/bench-baseline.sh artifacts-local/Qwen3.5-0.8B-Q8_0.gguf`
  (ROUNDS=3), foreground, idle Mac (Claude and WindowServer only; no
  thermal warning; swap 0.97 GiB pre-existing, no growth); same code as
  `20261026T182551Z-...` (171e53c; this tree adds records and docs only).
- MacBook Air M3 8 GB; 512 in / 256 out, 3 reps, 3 ABBA rounds; all 6
  runs valid on the first attempt.
- Locally-calibrated results (not claims):
  - TuckCore: warm decode 83.84 tok/s (78.10-84.30), warm TTFT 0.311 s
    (0.311 / 0.311 / 0.330 s).
  - llama.cpp b11695 Metal: 68.19 tok/s (66.34-84.49), TTFT 0.275 s
    (0.270-0.286 s); llama-bench pp512 1986.0, tg256 86.8.
  - Paired decode 1.145 (0.992, 1.271, 1.145); paired TTFT 1.15x.
- Prefill: 88% of llama.cpp here (0.275 / 0.311) against 91% in the
  first run. TuckCore's TTFT reproduced (0.311 s); the ratio moved with
  llama.cpp's own run-to-run spread (0.270-0.286 s). The Phase 1 prefill
  bar (>= 90%) is therefore **not confirmed**: 88-91% across two runs.
- Decode: llama.cpp's harness decode is bimodal across rounds and runs
  (66-68 tok/s usually, 84.5 tok/s in round 1 here, 84.7 once before).
  When it runs fast the pair is at parity (0.992). Recorded as "at
  parity or ahead (0.99-1.28 paired)", not as a robust lead.
