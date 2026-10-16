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
| Benchmarks measure model performance | This build benches the pipeline | Records carry `engine: reference-test`, `certified: false` and caveats |
| TuckCore "runs" a model | It executes the artifact's real tensor bytes through scalar f32 kernels; fixture weights are synthetic (untrained), so outputs are meaningless outside equivalence tests | "Experimental, uncertified executor; bit-exact streaming equivalence demonstrated; quality and performance unmeasured" |
| TuckCore has a tokenizer | It does not; prompts map deterministically into the artifact vocabulary and the limitation is labeled in run output and API extensions | Always: "no text tokenizer in this build" |
| TuckCore is fast | One smoke number on a 0.8 MiB fixture (debug build) is not evidence | No performance statement; scalar kernels are the numeric oracle |
| Streaming equivalence in fixture tests proves real-model behavior | Fixtures prove execution paths, not model behavior | Equivalence speaks to scheduling/streaming fidelity only |

## Build status statement (canonical)

> Tuckmill v0.2 engineering build: the orchestration stack (inspection,
> planning, admission, storage, packing, worker isolation, local API,
> benchmark harness) is implemented and tested. No model-execution kernel is
> implemented; no configuration is certified; no performance claim is made.
> Running a model requires a certified executor that does not yet exist; the
> reference test engine exercises the pipeline with clearly-labeled synthetic
> output only.
