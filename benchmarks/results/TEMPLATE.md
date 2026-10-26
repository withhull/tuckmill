# Per-tier baseline template

Copy the headings below into a tier's notes when a run is reviewed. Fill
every field from the run directory; write "n/a" with the reason instead of
leaving a field out.

## Machine

- Tier / chip / unified memory:
- macOS build, power source, ambient notes (lid, fan, other apps):
- tuckmill commit (clean tree?), llama.cpp revision:

## Artifact

- File, bytes, SHA-256 (must match the pinned identity):
- Architecture / quantization / context admitted:

## Protocol

- Shape (in / out / repetitions), ABBA rounds:
- Runs discarded (swap growth) or failed, with reason:
- Warm vs cold-load separation confirmed (first repetition excluded):

## Results (locally-calibrated)

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | | | | | |
| llama-metal | | | | | |

- Paired decode ratio TuckCore / llama.cpp (median, per-round values):
- Paired TTFT ratio:
- llama-bench vendor row (pp / tg):
- MLX: available on the identical artifact? reason:

## Reading

- What the numbers say about the gap, and which phase gate it feeds:
- What they do not establish (thermals, other prompts, other tiers):
