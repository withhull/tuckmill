/*
 * tm_llama_shim.c — narrow C shim over the pinned llama.cpp C API.
 *
 * Why a shim: llama.cpp's public API passes structs by value and grows fields
 * across revisions. Transcribing those layouts into Rust would be ABI
 * guessing. This shim is compiled against the *pinned revision's own headers*
 * (scripts/fetch-llama.sh), so struct usage is compile-time checked; the Rust
 * side only ever sees the stable pointer/int/float ABI below.
 *
 * Rules (ADR-001/ADR-005):
 *  - No remote access, no network, no substitutions.
 *  - CPU execution (n_gpu_layers = 0) in this build: deterministic across
 *    GPU families and free of shader-library coupling. GPU offload is a
 *    separately measured milestone.
 *  - Errors are returned as status ints plus a bounded message; nothing aborts.
 *
 * Shrink-wrapped surface for the Tuckmill reference backend:
 *   load/free, tokenize, token->piece, decode (append / reset), last-token
 *   logits, greedy sample, model description/size, meta values.
 */

#include "llama.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define TM_SHIM_VERSION 1

typedef struct tm_llama {
    struct llama_model   *model;
    struct llama_context *ctx;
    const struct llama_vocab *vocab;
    struct llama_sampler *sampler;
    uint32_t n_ctx;
    int32_t  pos; /* next decode position in the (single) sequence */
    int      backend_initialized;
} tm_llama;

static void tm_set_err(char *err, int cap, const char *msg) {
    if (err == NULL || cap <= 0) {
        return;
    }
    snprintf(err, (size_t) cap, "%s", msg);
}

int tm_shim_version(void) {
    return TM_SHIM_VERSION;
}

const char *tm_llama_build_info(void) {
    return "llama.cpp pinned revision (see scripts/fetch-llama.sh)";
}

tm_llama *tm_load(const char *path, int n_ctx, int n_threads, char *err, int err_cap) {
    if (path == NULL) {
        tm_set_err(err, err_cap, "null model path");
        return NULL;
    }
    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; /* CPU reference path (see header comment) */

    struct llama_model *model = llama_model_load_from_file(path, mparams);
    if (model == NULL) {
        tm_set_err(err, err_cap, "llama_model_load_from_file failed");
        return NULL;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const uint32_t n_ctx_train = llama_model_n_ctx_train(model);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (n_ctx > 0) ? (uint32_t) n_ctx
                               : (n_ctx_train > 0 ? n_ctx_train : 512u);
    cparams.n_threads = (n_threads > 0) ? n_threads : 4;
    cparams.n_threads_batch = cparams.n_threads;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        llama_model_free(model);
        tm_set_err(err, err_cap, "llama_init_from_model failed");
        return NULL;
    }

    struct llama_sampler *sampler =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    tm_llama *h = (tm_llama *) calloc(1, sizeof(tm_llama));
    if (h == NULL) {
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        tm_set_err(err, err_cap, "out of memory");
        return NULL;
    }
    h->model = model;
    h->ctx = ctx;
    h->vocab = vocab;
    h->sampler = sampler;
    h->n_ctx = llama_n_ctx(ctx);
    h->pos = 0;
    h->backend_initialized = 1;
    return h;
}

void tm_free(tm_llama *h) {
    if (h == NULL) {
        return;
    }
    if (h->sampler) llama_sampler_free(h->sampler);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    if (h->backend_initialized) llama_backend_free();
    free(h);
}

int32_t tm_vocab_size(const tm_llama *h) {
    return h && h->vocab ? llama_vocab_n_tokens(h->vocab) : -1;
}

int32_t tm_bos(const tm_llama *h) {
    return h && h->vocab ? llama_vocab_bos(h->vocab) : -1;
}

int32_t tm_eos(const tm_llama *h) {
    return h && h->vocab ? llama_vocab_eos(h->vocab) : -1;
}

uint32_t tm_n_ctx(const tm_llama *h) {
    return h ? h->n_ctx : 0;
}

/* Tokenize text; returns token count (>=0) or a negative required capacity. */
int32_t tm_tokenize(const tm_llama *h, const char *text, int32_t text_len,
                    int add_special, int32_t *out, int32_t out_cap) {
    if (h == NULL || text == NULL || out == NULL || text_len < 0) {
        return -1;
    }
    return llama_tokenize(h->vocab, text, text_len, out, out_cap,
                          add_special != 0, /*parse_special=*/true);
}

/* Piece bytes for a token id; returns byte count written (no NUL). */
int32_t tm_token_to_piece(const tm_llama *h, int32_t id, char *buf, int32_t cap) {
    if (h == NULL || buf == NULL) {
        return -1;
    }
    return llama_token_to_piece(h->vocab, id, buf, cap, /*lstrip=*/0, /*special=*/false);
}

/* Append tokens at the current position. Returns 0 on success. */
int32_t tm_decode(tm_llama *h, const int32_t *ids, int32_t n) {
    if (h == NULL || ids == NULL || n <= 0) {
        return -1;
    }
    if ((uint32_t) (h->pos + n) > h->n_ctx) {
        return -2; /* context overflow, refused before decode */
    }
    struct llama_batch batch = llama_batch_init(n, 0, 1);
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i] = (llama_token) ids[i];
        batch.pos[i] = h->pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n - 1) ? 1 : 0;
    }
    batch.n_tokens = n;
    const int32_t rc = llama_decode(h->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return -3; /* decode failed */
    }
    h->pos += n;
    return 0;
}

/* Forget all KV state; the next decode starts at position 0. */
int32_t tm_reset(tm_llama *h) {
    if (h == NULL) {
        return -1;
    }
    llama_memory_clear(llama_get_memory(h->ctx), true);
    llama_sampler_reset(h->sampler);
    h->pos = 0;
    return 0;
}

/* Copy the logits of the last decoded token into out; returns vocab size. */
int32_t tm_logits(tm_llama *h, float *out, int32_t cap) {
    if (h == NULL || out == NULL) {
        return -1;
    }
    const int32_t n = llama_vocab_n_tokens(h->vocab);
    if (cap < n) {
        return -n;
    }
    const float *logits = llama_get_logits_ith(h->ctx, -1);
    if (logits == NULL) {
        return -2;
    }
    memcpy(out, logits, (size_t) n * sizeof(float));
    return n;
}

/* Greedy sample from the last decoded token's logits. */
int32_t tm_sample_greedy(tm_llama *h, int32_t *out_token) {
    if (h == NULL || out_token == NULL) {
        return -1;
    }
    const llama_token id = llama_sampler_sample(h->sampler, h->ctx, -1);
    llama_sampler_accept(h->sampler, id);
    *out_token = (int32_t) id;
    return 0;
}

int32_t tm_model_desc(const tm_llama *h, char *buf, int32_t cap) {
    if (h == NULL || buf == NULL) {
        return -1;
    }
    return llama_model_desc(h->model, buf, (size_t) cap);
}

int32_t tm_meta(const tm_llama *h, const char *key, char *buf, int32_t cap) {
    if (h == NULL || key == NULL || buf == NULL) {
        return -1;
    }
    return llama_model_meta_val_str(h->model, key, buf, (size_t) cap);
}

uint64_t tm_model_size(const tm_llama *h) {
    return h ? llama_model_size(h->model) : 0;
}
