/*
 * tm_metal.h — TuckCore <-> Metal narrow C ABI (ADR-007).
 *
 * One context owns one MTLDevice, one MTLCommandQueue, and the compiled
 * compute pipelines for the operations declared here. Buffers are opaque
 * resource handles with explicit lifetime; dispatch is asynchronous and
 * completion-fenced through tm_metal_fence(). Measurements (committed and
 * completed dispatch counts, last GPU time, live allocated bytes) are
 * exposed so the caller can account for real GPU resources.
 *
 * All functions are thread-safe in the sense that a context may be used
 * from one thread at a time; completion counters are atomic and may be
 * observed from any thread. Return 0 on success, a TM_METAL_ERR_* code on
 * failure, with a human-readable message in `err` when provided.
 *
 * This is NOT the worker ABI (native/executor/tuckmill_worker_abi.h); only
 * TuckCore uses this interface.
 */
#ifndef TM_METAL_H
#define TM_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tm_metal_ctx tm_metal_ctx;
typedef struct tm_metal_buffer tm_metal_buffer;

enum {
    TM_METAL_OK = 0,
    TM_METAL_ERR_NO_DEVICE = -1,   /* no Metal device is available */
    TM_METAL_ERR_SOURCE = -2,      /* MSL compilation failed */
    TM_METAL_ERR_ALLOC = -3,       /* buffer allocation failed */
    TM_METAL_ERR_BOUNDS = -4,      /* size/range mismatch, zero length */
    TM_METAL_ERR_UNSUPPORTED = -5, /* invalid arguments / mode */
    TM_METAL_ERR_INTERNAL = -6     /* Metal/Objective-C runtime failure */
};

/* 1 when a Metal device exists (cheap; no context required). */
int tm_metal_available(void);

/* One-line runtime description (device name, unified memory size). Always
 * returns a valid static string. */
const char *tm_metal_runtime_info(void);

/* Create a context: device, queue, library compiled from `msl_source`, and
 * the pipeline states for every operation below. On failure returns NULL
 * and fills `err`. */
tm_metal_ctx *tm_metal_create(const char *msl_source, char *err, size_t err_cap);

/* Destroy the context. All buffers created from it must have been freed
 * first; the caller (Rust RAII handles) guarantees this. */
void tm_metal_destroy(tm_metal_ctx *ctx);

/* Allocate a buffer of `bytes` (> 0). `shared` buffers support CPU access
 * and host<->device copies; private buffers are GPU-only. Returns NULL on
 * failure. */
tm_metal_buffer *tm_metal_buffer_new(tm_metal_ctx *ctx, size_t bytes, int shared,
                                     char *err, size_t err_cap);

/* Release a buffer. Passing a buffer not owned by the context is a
 * programming error and reports TM_METAL_ERR_UNSUPPORTED. */
void tm_metal_buffer_free(tm_metal_ctx *ctx, tm_metal_buffer *buffer);

/* CPU pointer into a shared buffer (NULL for private buffers or if the
 * context does not own the buffer). Valid until the buffer is freed. */
void *tm_metal_buffer_contents(tm_metal_ctx *ctx, tm_metal_buffer *buffer);

/* Length of the buffer in bytes (0 when the handle is invalid). */
size_t tm_metal_buffer_len(const tm_metal_ctx *ctx, const tm_metal_buffer *buffer);

/* Host -> device and device -> host copies through shared storage. */
int tm_metal_upload(tm_metal_ctx *ctx, tm_metal_buffer *dst, size_t off,
                    const void *src, size_t len, char *err, size_t err_cap);
int tm_metal_download(tm_metal_ctx *ctx, tm_metal_buffer *src, size_t off,
                      void *dst, size_t len, char *err, size_t err_cap);

/* y[i] = x[i] * factor, i < n. n may be any length (tail-safe). */
int tm_metal_f32_scale(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                       float factor, uint32_t n, char *err, size_t err_cap);

/* y[i] = y[i] + alpha * a[i], i < n. */
int tm_metal_f32_axpy(tm_metal_ctx *ctx, tm_metal_buffer *a, tm_metal_buffer *y,
                      float alpha, uint32_t n, char *err, size_t err_cap);

/* y[i] = (float) x[i], i < n, where x holds IEEE binary16 values. */
int tm_metal_cast_f16_f32(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                          uint32_t n, char *err, size_t err_cap);

/* y[i] = (half) x[i], i < n, round-to-nearest-even. */
int tm_metal_cast_f32_f16(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                          uint32_t n, char *err, size_t err_cap);

/*
 * Grouped encoding (C2): operations between begin_group/end_group encode
 * into ONE command buffer committed at end_group, instead of one commit per
 * operator. abort_group discards the open group's work without committing
 * (cancellation at a step boundary; no partial GPU work is submitted).
 * Copies and fence require no open group.
 */
int tm_metal_begin_group(tm_metal_ctx *ctx, char *err, size_t err_cap);
int tm_metal_end_group(tm_metal_ctx *ctx, char *err, size_t err_cap);
int tm_metal_abort_group(tm_metal_ctx *ctx, char *err, size_t err_cap);

/* Dequantize row `row` of a packed Q8_0 matrix to `cols` f32 values. */
int tm_metal_q8_0_row_f32(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *out,
                          uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap);

/* F16 matvec: y[r] = dot(f32(w_row_r), x); `w` holds rows*cols f16 values. */
int tm_metal_f16_gemv(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *x,
                      tm_metal_buffer *y, uint32_t rows, uint32_t cols,
                      char *err, size_t err_cap);

/* y[row] = rmsnorm(x[row], weight, eps) for `rows` rows of `cols`. */
int tm_metal_rmsnorm_f32(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *weight,
                         tm_metal_buffer *y, uint32_t rows, uint32_t cols, float eps,
                         char *err, size_t err_cap);

/* NeoX RoPE in place over n_q_heads of q and n_kv_heads of k at `pos`,
 * using host-precomputed inv_freq[head_dim/2]. */
int tm_metal_rope_f32(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k,
                      tm_metal_buffer *inv_freq, uint32_t pos, uint32_t n_q_heads,
                      uint32_t n_kv_heads, uint32_t head_dim, char *err, size_t err_cap);

/* k_cache[h][pos][:] = k_src[h][:], same for v; `round_f16` stores
 * f16-representable f32 values (the declared f16 cache contract). */
int tm_metal_kv_store_f32(tm_metal_ctx *ctx, tm_metal_buffer *k_src, tm_metal_buffer *v_src,
                          tm_metal_buffer *k_cache, tm_metal_buffer *v_cache,
                          uint32_t kv_heads, uint32_t pos, uint32_t cap, uint32_t head_dim,
                          uint32_t round_f16, char *err, size_t err_cap);

/* Causal GQA attention at `pos`; `scratch` holds heads*cap f32. */
int tm_metal_attention_f32(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k_cache,
                           tm_metal_buffer *v_cache, tm_metal_buffer *out,
                           tm_metal_buffer *scratch, uint32_t heads, uint32_t pos,
                           uint32_t cap, uint32_t head_dim, uint32_t group, float scale,
                           char *err, size_t err_cap);

/* Dequantize row `row` of an F16 matrix to `cols` f32 values. */
int tm_metal_f16_row_f32(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *out,
                         uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap);

/* RMSNorm with an F16-weight table. */
int tm_metal_rmsnorm_f16w(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *weight,
                          tm_metal_buffer *y, uint32_t rows, uint32_t cols, float eps,
                          char *err, size_t err_cap);

/* ---------------- C3: batched prefill operations ---------------- */

/* Y[b, :] = W . X[b, :] for b < batch (1..=32); X is [batch, cols] at f32
 * element offset `x_off`, Y is [batch, rows] at `y_off`. Per-token
 * arithmetic order matches the single-token GEMV, so batch=1 is
 * bit-identical to it. */
int tm_metal_q8_0_gemm(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *x,
                       tm_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                       uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int tm_metal_f16_gemm(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *x,
                      tm_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                      uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);

/* Batched RoPE: token t rotates at start_pos + t (q [n_tokens, n_q_heads,
 * dim], k [n_tokens, n_kv_heads, dim]). */
int tm_metal_rope_batch_f32(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k,
                            tm_metal_buffer *inv_freq, uint32_t start_pos, uint32_t n_tokens,
                            uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
                            char *err, size_t err_cap);

/* Batched KV writes: row (t, head) lands at position start_pos + t. */
int tm_metal_kv_store_batch_f32(tm_metal_ctx *ctx, tm_metal_buffer *k_src, tm_metal_buffer *v_src,
                                tm_metal_buffer *k_cache, tm_metal_buffer *v_cache,
                                uint32_t n_tokens, uint32_t start_pos, uint32_t cap,
                                uint32_t head_dim, uint32_t kv_heads, uint32_t round_f16,
                                char *err, size_t err_cap);

/* Batched causal attention: one threadgroup per (token, query head); query
 * token t attends keys 0..=(start_pos + t). Scratch is
 * n_tokens * heads * cap f32. */
int tm_metal_attention_batch_f32(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k_cache,
                                 tm_metal_buffer *v_cache, tm_metal_buffer *out,
                                 tm_metal_buffer *scratch, uint32_t heads, uint32_t n_tokens,
                                 uint32_t start_pos, uint32_t cap, uint32_t head_dim,
                                 uint32_t group, float scale, char *err, size_t err_cap);

/* out[i] = silu(gate[i]) * up[i]. */
int tm_metal_silu_mul_f32(tm_metal_ctx *ctx, tm_metal_buffer *gate, tm_metal_buffer *up,
                          tm_metal_buffer *out, uint32_t n, char *err, size_t err_cap);

/* Packed Q8_0 matrix-vector decode: y[r] = dot(dequant_q8_0(w_row_r), x).
 * `w` holds `rows * cols / 32` blocks of 34 bytes ([f16 d][i8 qs; 32]).
 * `cols` must be a positive multiple of 32; `x` is f32 of length `cols`;
 * `y` is f32 of length `rows`. */
int tm_metal_q8_0_gemv(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *x,
                       tm_metal_buffer *y, uint32_t rows, uint32_t cols,
                       char *err, size_t err_cap);

/* Block until every committed command buffer has completed. Safe to call
 * with no pending work. */
int tm_metal_fence(tm_metal_ctx *ctx, char *err, size_t err_cap);

/* Measurement surface. */
uint64_t tm_metal_committed_dispatches(const tm_metal_ctx *ctx);
uint64_t tm_metal_completed_dispatches(const tm_metal_ctx *ctx);
/* GPU time of the most recently completed command buffer, milliseconds
 * (0.0 before the first completion). */
double tm_metal_last_gpu_ms(const tm_metal_ctx *ctx);
/* Live bytes currently allocated in buffers from this context. */
uint64_t tm_metal_allocated_bytes(const tm_metal_ctx *ctx);

/* Bytes leased by pending command buffers (physically alive until their
 * completion, even after the owning handles were freed). gpu_allocated +
 * in_flight is a conservative upper bound on live GPU bytes: a buffer that
 * is both handle-live and awaiting completion is counted twice, so the
 * bound never understates. */
uint64_t tm_metal_in_flight_bytes(const tm_metal_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* TM_METAL_H */
