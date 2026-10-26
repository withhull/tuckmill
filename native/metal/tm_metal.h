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

/* Zero `len` bytes of a buffer at byte offset `off`, without a host
 * staging allocation: shared storage is zeroed in place, private storage
 * uses a GPU fill. */
int tm_metal_buffer_zero(tm_metal_ctx *ctx, tm_metal_buffer *buffer, size_t off, size_t len,
                         char *err, size_t err_cap);

/* Host -> device and device -> host copies through shared storage. */
int tm_metal_upload(tm_metal_ctx *ctx, tm_metal_buffer *dst, size_t off,
                    const void *src, size_t len, char *err, size_t err_cap);
int tm_metal_download(tm_metal_ctx *ctx, tm_metal_buffer *src, size_t off,
                      void *dst, size_t len, char *err, size_t err_cap);

/* y[i] = x[i] * factor, i < n. n may be any length (tail-safe). */
int tm_metal_f32_scale(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                       float factor, uint32_t n, char *err, size_t err_cap);

/* y[i] = y[i] + alpha * a[i], i < n. */
/* Greedy argmax of x[0..n) into out[0] (u32), exactly the CPU rule: NaN
 * skipped, largest value wins, ties to the lowest index; 0xFFFFFFFF when
 * every entry is NaN, written to out[out_index]. `partial` is 512 bytes of
 * scratch. Encodable inside a dispatch group. */
int tm_metal_argmax_f32(tm_metal_ctx *ctx, tm_metal_buffer *x, uint32_t n,
                        tm_metal_buffer *partial, tm_metal_buffer *out, uint32_t out_index,
                        char *err, size_t err_cap);

/* out[i] = row ids[slot] of matrix `w` (type 8 = Q8_0, 1 = F16; `rows` x
 * `cols`), dequantized to f32: an embedding lookup whose token id was
 * produced on the device. An id >= rows (the argmax all-NaN marker) writes
 * zeros. Encodable inside a dispatch group. */
int tm_metal_row_indirect_f32(tm_metal_ctx *ctx, tm_metal_buffer *w, uint32_t type_id,
                              uint32_t rows, uint32_t cols, tm_metal_buffer *ids, uint32_t slot,
                              tm_metal_buffer *out, char *err, size_t err_cap);

/* dst[dst_off + i] = src[src_off + i] for i < n (f32 elements, bit-exact),
 * encodable inside a dispatch group. Offsets and n are checked against
 * both buffers. */
int tm_metal_copy_f32(tm_metal_ctx *ctx, tm_metal_buffer *src, uint32_t src_off,
                      tm_metal_buffer *dst, uint32_t dst_off, uint32_t n, char *err,
                      size_t err_cap);

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
/* Largest batch tm_metal_q8_0_gemm_mma accepts (one launch, a 2-D grid of
 * 32-row x 32-token tiles). */
#define TM_METAL_GEMM_MMA_MAX_BATCH 512u

/* Same contract as tm_metal_q8_0_gemm for batch 1..=TM_METAL_GEMM_MMA_MAX_BATCH,
 * on the SIMD-group matrix units (f32 tiles). Summation order differs from the GEMV, so it
 * is tolerance-qualified, not bit-identical; dense paths that claim
 * batch-size bit-identity keep tm_metal_q8_0_gemm. */
int tm_metal_q8_0_gemm_mma(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *x,
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

/* ---------------- Q35-2A: Gated DeltaNet operators ---------------- */

/* One decode step of the depthwise causal convolution: `hist` holds the
 * previous (kernel - 1) rows (channel-major, oldest first), `x` the new
 * row, `w` the ssm_conv1d weights in GGUF memory order (channel-major,
 * taps fastest), `y` the output row. */
int tm_metal_qwen35_conv_step(tm_metal_ctx *ctx, tm_metal_buffer *hist, tm_metal_buffer *x,
                              tm_metal_buffer *w, tm_metal_buffer *y, uint32_t channels,
                              uint32_t kernel, char *err, size_t err_cap);

/* Advance the convolution history after tm_metal_qwen35_conv_step: shift
 * down one row and append the new input row (same group). */
int tm_metal_qwen35_conv_shift(tm_metal_ctx *ctx, tm_metal_buffer *hist, tm_metal_buffer *x,
                               uint32_t channels, uint32_t kernel, char *err, size_t err_cap);

/* Per-head L2 norm with the epsilon inside the root:
 * y[r][i] = x[r][i] / sqrt(sum_i x^2 + eps). */
int tm_metal_qwen35_l2norm_heads(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                                 uint32_t rows, uint32_t cols, float eps, char *err,
                                 size_t err_cap);

/* Per-head gates: beta_out = sigmoid(raw_beta);
 * gate_out = ssm_a * softplus(alpha + dt_bias) with ggml's thresholded
 * softplus (x > 20 ? x : log(1 + e^x)). */
int tm_metal_qwen35_beta_decay(tm_metal_ctx *ctx, tm_metal_buffer *raw_beta, tm_metal_buffer *alpha,
                               tm_metal_buffer *dt_bias, tm_metal_buffer *ssm_a,
                               tm_metal_buffer *beta_out, tm_metal_buffer *gate_out, uint32_t n,
                               char *err, size_t err_cap);

/* One token of the autoregressive delta rule for every value head; the
 * f32 state buffer [heads, k_dim, v_dim] (i-major) is updated in place. */
int tm_metal_qwen35_gdn_step(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k,
                             tm_metal_buffer *v, tm_metal_buffer *gate, tm_metal_buffer *beta,
                             tm_metal_buffer *state, tm_metal_buffer *out, uint32_t heads,
                             uint32_t k_dim, uint32_t v_dim, float q_scale, char *err,
                             size_t err_cap);

/* Gated output normalization: per row, RMS norm over `cols` with the
 * ssm_norm weights, times SiLU of the z row. */
int tm_metal_qwen35_gated_norm(tm_metal_ctx *ctx, tm_metal_buffer *o, tm_metal_buffer *z,
                               tm_metal_buffer *weight, tm_metal_buffer *out, uint32_t rows,
                               uint32_t cols, float eps, char *err, size_t err_cap);

/* ---------------- Q35-2B: full-attention operators ---------------- */

/* Split the joint Q projection [heads, 2*head_dim] (query then gate per
 * head) into q and gate, both [heads, head_dim]. */
int tm_metal_qwen35_q_gate_split(tm_metal_ctx *ctx, tm_metal_buffer *qg, tm_metal_buffer *q_out,
                                 tm_metal_buffer *gate_out, uint32_t heads, uint32_t head_dim,
                                 char *err, size_t err_cap);

/* Partial split-half RoPE: rotate the first n_rot dims of every q/k head
 * row, pairs (i, i + n_rot/2), at `pos`; inv_freq holds n_rot/2 entries. */
int tm_metal_qwen35_rope_partial(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k,
                                 tm_metal_buffer *inv_freq, uint32_t pos, uint32_t n_q_heads,
                                 uint32_t n_kv_heads, uint32_t head_dim, uint32_t n_rot,
                                 char *err, size_t err_cap);

/* Packed f16 KV store: k_cache[h][pos][d] = (half) k_src[h][d], two-byte
 * entries (same for v). */
int tm_metal_qwen35_kv_store_f16(tm_metal_ctx *ctx, tm_metal_buffer *k_src, tm_metal_buffer *v_src,
                                 tm_metal_buffer *k_cache, tm_metal_buffer *v_cache, uint32_t pos,
                                 uint32_t cap, uint32_t head_dim, uint32_t kv_heads, char *err,
                                 size_t err_cap);

/* Causal GQA attention over a packed f16 cache; scratch is heads*cap f32. */
int tm_metal_qwen35_attention_f16(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k_cache,
                                  tm_metal_buffer *v_cache, tm_metal_buffer *out,
                                  tm_metal_buffer *scratch, uint32_t heads, uint32_t pos,
                                  uint32_t cap, uint32_t head_dim, uint32_t group, float scale,
                                  char *err, size_t err_cap);

/* x[i] = silu(x[i]) in place (conv output activation). */
int tm_metal_qwen35_silu(tm_metal_ctx *ctx, tm_metal_buffer *x, uint32_t n, char *err,
                         size_t err_cap);

/* Split the conv output [3*width] into q, k, v [width] each. */
int tm_metal_qwen35_split_qkv(tm_metal_ctx *ctx, tm_metal_buffer *y, tm_metal_buffer *q,
                              tm_metal_buffer *k, tm_metal_buffer *v, uint32_t width, char *err,
                              size_t err_cap);

/* attn[i] *= sigmoid(gate[i]). */
int tm_metal_qwen35_attn_gate(tm_metal_ctx *ctx, tm_metal_buffer *attn, tm_metal_buffer *gate,
                              uint32_t n, char *err, size_t err_cap);

/* Block until every committed command buffer has completed. Safe to call
 * with no pending work. */
int tm_metal_fence(tm_metal_ctx *ctx, char *err, size_t err_cap);

/* Measurement surface. */
uint64_t tm_metal_committed_dispatches(const tm_metal_ctx *ctx);
/* Copy a shared buffer range once the first `completed_target` committed
 * command buffers have completed, without waiting for later work (which
 * must not touch the range). Used to read a pipelined decode step's token
 * while the next step runs. */
int tm_metal_download_after(tm_metal_ctx *ctx, tm_metal_buffer *src, size_t off, void *dst,
                            size_t len, uint64_t completed_target, char *err, size_t err_cap);
uint64_t tm_metal_completed_dispatches(const tm_metal_ctx *ctx);
/* GPU time of the most recently completed command buffer, milliseconds
 * (0.0 before the first completion). */
double tm_metal_last_gpu_ms(const tm_metal_ctx *ctx);
/* Cumulative GPU nanoseconds over every completed command buffer. Wall
 * time minus this, over the same span, is host/dispatch time. */
uint64_t tm_metal_gpu_nanos_total(const tm_metal_ctx *ctx);
/* Opt-in per-operator GPU time attribution (profiling only). When
 * enabled, each committed command buffer carries its operator label and
 * its GPU time accumulates per label; buffers of an open dispatch group
 * are attributed to "group". Setting either value clears the table. */
int tm_metal_profile_set(tm_metal_ctx *ctx, int enabled);
/* Reads table entry `index`: returns 1 and fills name/nanos/count while
 * index is below the number of labels seen, 0 past the end. */
int tm_metal_profile_read(tm_metal_ctx *ctx, uint32_t index, char *name, size_t name_cap,
                          uint64_t *nanos, uint64_t *count);
/* Live bytes currently allocated in buffers from this context. */
uint64_t tm_metal_allocated_bytes(const tm_metal_ctx *ctx);

/* Bytes leased by pending command buffers (physically alive until their
 * completion, even after the owning handles were freed). gpu_allocated +
 * in_flight is a conservative upper bound on live GPU bytes: a buffer that
 * is both handle-live and awaiting completion is counted twice, so the
 * bound never understates. */
uint64_t tm_metal_in_flight_bytes(const tm_metal_ctx *ctx);

/* Chunk-wide Gated DeltaNet core (engine prefill): each processes `n`
 * tokens with exactly the per-element arithmetic of its per-token
 * counterpart (tm_metal_qwen35_conv_step + conv_shift, split_qkv,
 * beta_decay, gdn_step). Rows are token-major. */
int tm_metal_qwen35_conv_chunk(tm_metal_ctx *ctx, tm_metal_buffer *hist, tm_metal_buffer *x,
                               tm_metal_buffer *w, tm_metal_buffer *y, uint32_t channels,
                               uint32_t kern, uint32_t n, char *err, size_t err_cap);
int tm_metal_qwen35_split_qkv_chunk(tm_metal_ctx *ctx, tm_metal_buffer *y, tm_metal_buffer *q,
                                    tm_metal_buffer *k, tm_metal_buffer *v, uint32_t width,
                                    uint32_t n, char *err, size_t err_cap);
int tm_metal_qwen35_beta_decay_chunk(tm_metal_ctx *ctx, tm_metal_buffer *raw_beta,
                                     tm_metal_buffer *alpha, tm_metal_buffer *dt_bias,
                                     tm_metal_buffer *ssm_a, tm_metal_buffer *beta_out,
                                     tm_metal_buffer *gate_out, uint32_t per, uint32_t n,
                                     char *err, size_t err_cap);
int tm_metal_qwen35_gdn_chunk(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k,
                              tm_metal_buffer *v, tm_metal_buffer *gate, tm_metal_buffer *beta,
                              tm_metal_buffer *state, tm_metal_buffer *out, uint32_t heads,
                              uint32_t k_dim, uint32_t v_dim, float q_scale, uint32_t n,
                              uint32_t qk_stride, uint32_t v_stride, uint32_t g_stride,
                              char *err, size_t err_cap);

/* Chunk-wide gated attention core (engine prefill): RoPE, packed f16 KV
 * store and causal attention for `n` tokens at positions pos0..pos0+n-1,
 * each with exactly its per-token counterpart's arithmetic. Rows are
 * token-major; attention scratch is [n, heads, cap]. */
int tm_metal_qwen35_rope_partial_chunk(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k,
                                       tm_metal_buffer *inv_freq, uint32_t pos0,
                                       uint32_t n_q_heads, uint32_t n_kv_heads,
                                       uint32_t head_dim, uint32_t n_rot, uint32_t n, char *err,
                                       size_t err_cap);
int tm_metal_qwen35_kv_store_f16_chunk(tm_metal_ctx *ctx, tm_metal_buffer *k_src,
                                       tm_metal_buffer *v_src, tm_metal_buffer *k_cache,
                                       tm_metal_buffer *v_cache, uint32_t pos0, uint32_t cap,
                                       uint32_t head_dim, uint32_t kv_heads, uint32_t n, char *err,
                                       size_t err_cap);
int tm_metal_qwen35_attention_f16_chunk(tm_metal_ctx *ctx, tm_metal_buffer *q,
                                        tm_metal_buffer *k_cache, tm_metal_buffer *v_cache,
                                        tm_metal_buffer *out, tm_metal_buffer *scratch,
                                        uint32_t heads, uint32_t pos0, uint32_t cap,
                                        uint32_t head_dim, uint32_t group, float scale, uint32_t n,
                                        char *err, size_t err_cap);

/* Testing switch: use the generic device-memory recurrence kernel instead
 * of the register-resident specialization (results are bit-identical). */
int tm_metal_set_gdn_chunk_generic(tm_metal_ctx *ctx, int generic);

#ifdef __cplusplus
}
#endif

#endif /* TM_METAL_H */
