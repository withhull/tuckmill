# ADR-012: Serving features adopted from the Ollama reference

Status: **accepted** (2026-10-27). Builds on ADR-001 (product contract),
ADR-006 (tiers), ADR-008 (Qwen3.5 contract) and ADR-011 (unified engine).

## Context

A review of Ollama (`github.com/ollama/ollama`) showed five features that
make a local engine usable day to day and that Tuckmill does not have:

- chat for every supported family;
- a prefix cache;
- speculative decoding through the model's own MTP head (Ollama's MLX
  runner does this for Qwen3.5);
- model lifecycle (keep-alive, several resident models, idle eviction);
- distribution (pull and verify, Anthropic-compatible API).

Ollama also makes choices Tuckmill's contract forbids. On out-of-memory it
shrinks the context; it admits models by an 80%-of-free-memory heuristic;
and it writes one forward pass per family. Those are explicitly **not**
adopted.

## Decision

Adopt the five features in this order. Each is a gated step of its own,
and each keeps the contract: same-profile execution, exact admission,
honest labels, and the unified engine.

1. **Chat for Qwen3.5.** Pin the declared templates byte-exactly (the 0.8B
   and 2B revisions differ only inside the tool-call branch, which is
   outside the request subset). Implement their semantics in the
   restricted renderer. Gate: goldens rendered with Jinja2 configured as
   Hugging Face applies it, and token ids from pinned llama.cpp. The
   template's default (thinking disabled) is the default.
2. **Prefix cache.** Reuse an exact token prefix's session state across
   requests: KV for attention layers, plus the recurrent and convolution
   state at the prefix boundary for Gated DeltaNet layers. It is priced in
   admission. Gate: logits after a cache hit are bit-identical to a cold
   prefill of the same tokens. Hits are recorded as the `prefix-hit` cache
   state, never mixed with cold numbers.
3. **MTP speculative decoding** (Qwen3.5 artifacts that carry the MTP
   block). Draft with the MTP head and verify with the full model in one
   batched step. Gate: greedy output token-identical to non-speculative
   decoding. Sampling ships only with exact rejection sampling. Speedups
   are measured, never assumed.
4. **Model lifecycle.** Keep-alive and idle eviction, and several resident
   models only when every load is admitted exactly. A load that does not
   fit is refused or evicts by policy; context is never truncated.
5. **Distribution.**
   - Pull from a declared source with the content hash verified before
     import.
   - An Anthropic-compatible endpoint next to the OpenAI-style one.
   - Structured output and tool calls only through templates that define
     them, never by prompt injection.

## Consequences

- Each step lands with its evidence in `docs/claims.md` and
  `docs/certification-matrix.md`. A step that cannot pass its gate stays
  ⛔ in the README.
- Steps 2-3 are engine work on the unified blocks and benefit every
  family. Steps 4-5 are supervisor and store work, outside the engine.

## Step 1 record (2026-10-27): chat for Qwen3.5

- **Pinned templates.** The two pinned templates are stored verbatim
  (`crates/manifest/src/templates/qwen35-{0.8b,2b}.jinja`; SHA-256
  `273d8e0e…` and `7f0e5290…`) and recognized by exact match.
- **Renderer.** `ChatTemplate::Qwen35` in `crates/supervisor/src/bundle.rs`
  implements the template's semantics for the text subset:
  - trimmed content (Jinja2 `trim` = Python `str.strip()`);
  - a system message first only;
  - a real user query required;
  - explicit thinking blocks after the last query;
  - the default generation prompt with thinking off.
  The template's own refusals become typed `InvalidRequest` errors, so
  `ChatTemplate::render` is now fallible.
- **Goldens.** `scripts/qwen35_chat_goldens.py` renders 15 conversations
  with Jinja2 3.1.6, configured as Hugging Face applies chat templates,
  from both artifacts' template bytes; both revisions agree on every case.
  Token ids come from pinned llama.cpp `llama-tokenize`, not from
  Tuckmill's encoder.
- **Verified:**
  - the renderer reproduces all 12 prompts byte for byte and all 3
    refusals;
  - the reference-first encoder reproduces the golden token ids on the
    real 0.8B and 2B;
  - `/v1/chat/completions` on the real 0.8B (unified engine, Metal)
    answers "The capital of France is Paris." and finishes at EOS.
- **Not included (refused, never approximated):** tools and tool calls,
  image or video content, and `enable_thinking`.
- **Prioritization note** (same day): the product owner asked not to
  spend effort on dense Qwen3, an older family. Steps 2-3 target Qwen3.5
  and newer families.
