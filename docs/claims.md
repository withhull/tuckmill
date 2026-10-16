# Claim and falsification ledger

Every externally visible statement about this repository must map to a row in
this table. Statements not in the "allowed" column are not made anywhere in
the code, output, or documentation.

| Claim or hypothesis | Current evidence | Allowed statement in this build |
| --- | --- | --- |
| Custom kernels are in the development scope | K01–K09 specified in `kernels/registry.json` | "Specified" — never "implemented" or "fast" |
| A compact 9B can be useful on 8 GB | Published weight file sizes (publisher-reported) | "Candidate awaiting physical-Mac qualification" |
| A 27B model retains nearly all quality | Publisher's aggregates with per-task losses | Report the named suite and limitations, never universal equivalence |
| Tuckmill is better than any other engine | No matched benchmark exists | Unproven; do not compare |
| M1/M2 support follows from M3 support | None | Requires separate kernels, builds, and device tests |
| Any 100B model runs interactively | None | Long-term compatibility/research target only |
| The reference test engine "runs" a model | It executes no weights | Always: "synthetic output; no model weights are executed" |
| Budgets protect the host absolutely | Managed admission policy only | Always: "not an OS guarantee; footprint measured separately" |
| Importing = redistributable | Import records a license note only | Always: importing is not a license review |
| Tuckmill is an engine "for 8 GB Macs" | 8 GiB is the minimum target tier; strategies scale to 512 GiB+ (ADR-006) | "Mac-native high-performance inference for Apple Silicon, from 8 GB upward" |
| Tuckmill beats llama.cpp / MLX / Splash | No matched measurement exists | Unproven; comparisons run only under the four-way identity rules with registered tolerances (`benchmarks/README.md`) |
| TuckCore matches llama.cpp numerically | Qwen3-0.6B on M3 8 GiB, ctx 2048, pinned llama.cpp CPU: F16 twin max abs logit diff 1.95e-2 (median 2.2e-3) at registered 5e-2; Q8_0 max 4.6e-1 (median 3.9e-2) at registered 5e-1 — Q8 drift dominated by llama.cpp's 8-bit activation quantization, whose architecture-specific kernel dispatch is not reproducible from public semantics; greedy text may diverge on near-ties (recorded, not a pass criterion) | "Within the registered per-dtype envelope on this artifact and configuration" — never "matches" without the record; bit-identity is not claimed |
| Our tokenizer matches llama.cpp | Reference-first policy: when the pinned llama.cpp shim is configured, encoding IS llama.cpp (vocab-only load, CPU-pinned, golden-tested on Qwen3-0.6B incl. digits/contractions/unicode with exact byte roundtrip). The built-in BPE is exact only for GPT-2-style pre-tokenizers (fixture goldens) and is refused otherwise | Real-corpus breadth pending; the built-in BPE is never used for non-GPT-2 pre-tokenizers |
| Verification cannot false-pass | Malformed scoring fails closed (exact positions, exact vocabulary length, finite values, contiguous bounded chunks — regression-tested); reports are immutable (config ID vs unique run ID; interrupted runs preserved); tolerances are a versioned policy with held-out qualification (Q8_0/F16 PASS on frozen disjoint inputs) | Exploratory (explicit-tolerance) runs never count as qualification; task-quality suites remain a separate milestone |
| Governor actions reach the engine | Protocol v4 Control frames + shared trim epoch; mid-generation `trim-caches` evicts expendable caches (tested, in-flight blocks untouched); monitor/cancel/status are lock-free vs the generation lock; footprint overruns trim then pause after 3 sustained samples | Pressure soak on physical hardware (deliberate memory pressure during real generation) is the remaining evidence |
| Benchmarks measure model performance | This build benches the pipeline | Records carry `engine: reference-test`, `certified: false` and caveats |
| TuckCore "runs" a model | Real Qwen3-0.6B Q8_0 executes end-to-end: correct completions on real weights (e.g. "The capital of France is" → " Paris. The capital of France"), token IDs within registered llama.cpp envelopes, records persisted | "Experimental, uncertified executor; real-model execution demonstrated; quality and performance unmeasured" |
| TuckCore has a tokenizer | It does not; prompts map deterministically into the artifact vocabulary and the limitation is labeled in run output and API extensions | Always: "no text tokenizer in this build" |
| TuckCore is fast | One smoke number on a 0.8 MiB fixture (debug build) is not evidence | No performance statement; scalar kernels are the numeric oracle |
| Streaming equivalence in fixture tests proves real-model behavior | Fixtures prove execution paths, not model behavior | Equivalence speaks to scheduling/streaming fidelity only |

## Build status statement (canonical)

> Tuckmill v0.2 engineering build (Stage A in progress): a Mac-native, high-performance local
> inference framework for Apple Silicon, engineered from the 8 GiB minimum
> tier upward (ADR-006). The orchestration stack (inspection, planning,
> admission, storage, packing, worker isolation, local API, benchmark
> harness) is implemented and tested. TuckCore, the experimental custom
> executor, runs real dense-Qwen3 weights through scalar f32 reference
> kernels with bit-exact residency/streaming equivalence demonstrated; no
> Metal kernel is implemented; no configuration is certified; no performance
> comparison against llama.cpp / MLX / Splash is claimed before it is
> measured.
