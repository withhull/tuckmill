/*
 * TuckCore Metal kernels (ADR-007), compiled at runtime from this exact
 * repository revision. Correctness-first: every kernel is validated against
 * the CPU scalar reference before any tuning; see crates/metal tests.
 *
 * Numerics: f32 accumulation with the same dequantization the CPU reference
 * uses (Q8_0 dequant = d * q, f16 scale). GPU reduction order differs from
 * the CPU sequence, so comparisons are tolerance-registered, never exact.
 */
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/* y[i] = x[i] * factor; tail-safe for any n. */
kernel void tm_scale_f32(device const float *x [[buffer(0)]],
                         device float *y [[buffer(1)]],
                         constant float &factor [[buffer(2)]],
                         constant uint &n [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    y[gid] = x[gid] * factor;
}

/* y[i] = y[i] + alpha * a[i]. */
kernel void tm_axpy_f32(device const float *a [[buffer(0)]],
                        device float *y [[buffer(1)]],
                        constant float &alpha [[buffer(2)]],
                        constant uint &n [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    y[gid] += alpha * a[gid];
}

/* y[i] = (float) x[i]; x holds IEEE binary16. */
kernel void tm_cast_f16_f32(device const half *x [[buffer(0)]],
                            device float *y [[buffer(1)]],
                            constant uint &n [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    y[gid] = (float)x[gid];
}

/* y[i] = (half) x[i]; round-to-nearest-even, overflow to infinity. */
kernel void tm_cast_f32_f16(device const float *x [[buffer(0)]],
                            device half *y [[buffer(1)]],
                            constant uint &n [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    y[gid] = (half)x[gid];
}

/*
 * Packed Q8_0 matrix-vector decode: y[r] = dot(dequant_q8_0(w_row_r), x).
 *
 * Q8_0 blocks are 34 bytes: [f16 d][i8 qs; 32]; dequant(w) = d * q.
 * Multi-row form (Phase 1C): each SIMD group computes TM_Q8_ROWS rows, and
 * a threadgroup holds TM_Q8_SG SIMD groups. Lane L handles quants
 * (L % 4) * 8 .. +7 of block (L / 4), so one SIMD-group step covers 8
 * blocks; the 8 matching x values are loaded once into registers and
 * reused for every row. Per-lane partials reduce with simd_sum. Reduction
 * order differs from the CPU sequence (tolerance-registered, ADR-007).
 * `cols` must be a positive multiple of 32 (validated C-side).
 */
#define TM_Q8_ROWS 4u
#define TM_Q8_SG 2u

kernel void tm_q8_0_gemv(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 34u;
    const uint first = (tg * TM_Q8_SG + sg) * TM_Q8_ROWS;
    if (first >= rows) {
        return;
    }
    const uint sub = lane % 4u;   /* which 8-quant slice of the block */
    const uint lane_blk = lane / 4u; /* which of the 8 blocks this step */

    float acc[TM_Q8_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint b = lane_blk; b < blocks; b += 8u) {
        const device packed_float4 *xv =
            (const device packed_float4 *)(x + (size_t)b * 32u + sub * 8u);
        const float4 x0 = float4(xv[0]);
        const float4 x1 = float4(xv[1]);
        for (uint r = 0; r < TM_Q8_ROWS; ++r) {
            const uint row = first + r;
            if (row >= rows) {
                break;
            }
            const device uchar *blk = w + (size_t)row * row_bytes + (size_t)b * 34u;
            const float d = (float)(*(const device half *)blk);
            const device packed_char4 *qv = (const device packed_char4 *)(blk + 2u + sub * 8u);
            const float4 q0 = float4(char4(qv[0]));
            const float4 q1 = float4(char4(qv[1]));
            acc[r] += d * (dot(q0, x0) + dot(q1, x1));
        }
    }
    for (uint r = 0; r < TM_Q8_ROWS; ++r) {
        const float total = simd_sum(acc[r]);
        const uint row = first + r;
        if (lane == 0u && row < rows) {
            y[row] = total;
        }
    }
}

/* dst[dst_off + i] = src[src_off + i] for i < n: a bit-exact f32 copy
 * between buffers, encodable inside a compute dispatch group (batched
 * prefill moves per-token slices with it). */
kernel void tm_copy_f32(device const float *src [[buffer(0)]],
                        device float *dst [[buffer(1)]],
                        constant uint &src_off [[buffer(2)]],
                        constant uint &dst_off [[buffer(3)]],
                        constant uint &n [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    dst[dst_off + gid] = src[src_off + gid];
}

/* ---------------- C2: transformer operation kernels ---------------- */

/* Dequantize one Q8_0 matrix row (or embedding row) to f32:
 * out[d] = d_scale(d/32) * q(d). One thread per output element. */
kernel void tm_q8_0_row_f32(device const uchar *w [[buffer(0)]],
                            device float *out [[buffer(1)]],
                            constant uint &row [[buffer(2)]],
                            constant uint &cols [[buffer(3)]],
                            constant uint &out_off [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 34u;
    const uint b = gid / 32u;
    const uint lane = gid % 32u;
    const device uchar *blk = wrow + (size_t)b * 34u;
    const float d = (float)(*(const device half *)blk);
    const float q = (float)((const device char *)(blk + 2))[lane];
    out[out_off + gid] = d * q;
}

/* F16 matvec: y[r] = dot(f32(w_row_r), x). Lane-strided within one SIMD
 * group per row; used for F16-weight artifacts (fixtures, F16 twins). */
kernel void tm_f16_gemv(device const half *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant uint &rows [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows) {
        return;
    }
    const device half *wrow = w + (size_t)row * (size_t)cols;
    float acc = 0.0f;
    for (uint i = lane; i < cols; i += 32u) {
        acc += (float)wrow[i] * x[i];
    }
    const float total = simd_sum(acc);
    if (lane == 0u) {
        y[row] = total;
    }
}

/* Threadgroup-wide reduction of `value` over 128 threads: 4 SIMD partials
 * are combined through shared memory. Returns the sum to every lane. */
inline float tm_reduce_sum_128(float value, threadgroup float *scratch,
                               uint simd_lane, uint simd_group) {
    value = simd_sum(value);
    if (simd_lane == 0u) {
        scratch[simd_group] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (simd_group == 0u) {
        float v = simd_lane < 4u ? scratch[simd_lane] : 0.0f;
        total = simd_sum(v);
        if (simd_lane == 0u) {
            scratch[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

/* RMSNorm over one row of `cols` f32 values with a per-column weight and
 * epsilon. One threadgroup of 128 threads per row; in-place (x == y) is
 * safe because each thread only reads its own elements after the barrier. */
kernel void tm_rmsnorm_f32(device const float *x [[buffer(0)]],
                           device const float *weight [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant uint &cols [[buffer(3)]],
                           constant float &eps [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[4];
    const device float *xrow = x + (size_t)row * (size_t)cols;
    device float *yrow = y + (size_t)row * (size_t)cols;
    float acc = 0.0f;
    for (uint i = tid; i < cols; i += 128u) {
        acc += xrow[i] * xrow[i];
    }
    const float sumsq = tm_reduce_sum_128(acc, scratch, lane, simd_group);
    const float rms = 1.0f / sqrt(sumsq / (float)cols + eps);
    for (uint i = tid; i < cols; i += 128u) {
        yrow[i] = xrow[i] * rms * weight[i];
    }
}

/*
 * NeoX-style RoPE applied in place to q (n_q heads) and k (n_kv heads) at
 * position `pos`, with inv_freq[i] = theta^(-2i/head_dim) for i < head_dim/2
 * precomputed on the host (matching the scalar executor's f32 table).
 * Layout per head: [.., dim] rotations pair (i, i + head_dim/2). One
 * threadgroup per head of head_dim/2 lanes.
 */
kernel void tm_rope_f32(device float *q [[buffer(0)]],
                        device float *k [[buffer(1)]],
                        device const float *inv_freq [[buffer(2)]],
                        constant uint &pos [[buffer(3)]],
                        constant uint &n_q_heads [[buffer(4)]],
                        constant uint &n_kv_heads [[buffer(5)]],
                        constant uint &head_dim [[buffer(6)]],
                        uint head [[threadgroup_position_in_grid]],
                        uint lane [[thread_position_in_threadgroup]]) {
    const uint half_dim = head_dim / 2u;
    if (lane >= half_dim) {
        return;
    }
    const bool is_q = head < n_q_heads;
    const uint local_head = is_q ? head : head - n_q_heads;
    device float *base = is_q ? q : k;
    const device float *row = base + (size_t)local_head * (size_t)head_dim;
    device float *mutable_row = base + (size_t)local_head * (size_t)head_dim;
    const float angle = (float)pos * inv_freq[lane];
    const float c = cos(angle);
    const float s = sin(angle);
    const float a = row[lane];
    const float b = row[lane + half_dim];
    mutable_row[lane] = a * c - b * s;
    mutable_row[lane + half_dim] = a * s + b * c;
}

/* k_cache[h][pos][d] = k_src[h][d] (same for v). One threadgroup per kv
 * head over head_dim lanes. */
kernel void tm_kv_store_f32(device const float *k_src [[buffer(0)]],
                            device const float *v_src [[buffer(1)]],
                            device float *k_cache [[buffer(2)]],
                            device float *v_cache [[buffer(3)]],
                            constant uint &pos [[buffer(4)]],
                            constant uint &cap [[buffer(5)]],
                            constant uint &head_dim [[buffer(6)]],
                            constant uint &round_f16 [[buffer(7)]],
                            uint head [[threadgroup_position_in_grid]],
                            uint lane [[thread_position_in_threadgroup]]) {
    if (lane >= head_dim) {
        return;
    }
    const size_t src = (size_t)head * (size_t)head_dim + lane;
    const size_t dst = ((size_t)head * (size_t)cap + (size_t)pos) * (size_t)head_dim + lane;
    if (round_f16 != 0u) {
        /* Declared f16 cache precision: entries stay f32 values that are
         * exactly f16-representable, mirroring the CPU executor's contract. */
        k_cache[dst] = (float)(half)k_src[src];
        v_cache[dst] = (float)(half)v_src[src];
    } else {
        k_cache[dst] = k_src[src];
        v_cache[dst] = v_src[src];
    }
}

/*
 * Causal GQA attention for one query position. One threadgroup per query
 * head; 128 threads. Scores for keys 0..=pos land in scratch (per-head
 * region of `heads * cap` floats), softmax in registers/shared, weighted
 * value accumulation parallel over head_dim lanes. `scale` is
 * 1/sqrt(head_dim). No mask beyond causality (Qwen3 dense here has no
 * sliding window).
 */
kernel void tm_attention_f32(device const float *q [[buffer(0)]],
                             device const float *k_cache [[buffer(1)]],
                             device const float *v_cache [[buffer(2)]],
                             device float *out [[buffer(3)]],
                             device float *scratch [[buffer(4)]],
                             constant uint &pos [[buffer(5)]],
                             constant uint &cap [[buffer(6)]],
                             constant uint &head_dim [[buffer(7)]],
                             constant uint &group [[buffer(8)]],
                             constant float &scale [[buffer(9)]],
                             uint head [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float reduce_scratch[4];
    threadgroup float row_stats[2]; /* [0] = running max, [1] = denominator */
    const uint kv_head = head / group;
    const device float *qh = q + (size_t)head * (size_t)head_dim;
    const device float *kc = k_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim;
    const device float *vc = v_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim;
    device float *head_scratch = scratch + (size_t)head * (size_t)cap;
    const uint seq = pos + 1u;

    /* scores */
    for (uint j = 0; j < seq; ++j) {
        float part = 0.0f;
        for (uint d = tid; d < head_dim; d += 128u) {
            part += qh[d] * kc[(size_t)j * (size_t)head_dim + d];
        }
        const float s = tm_reduce_sum_128(part, reduce_scratch, lane, simd_group) * scale;
        if (tid == 0u) {
            head_scratch[j] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* row max */
    float local_max = -INFINITY;
    for (uint j = tid; j < seq; j += 128u) {
        local_max = max(local_max, head_scratch[j]);
    }
    local_max = simd_max(local_max);
    if (lane == 0u) {
        reduce_scratch[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float m = reduce_scratch[0];
        for (uint g = 1u; g < 4u; ++g) {
            m = max(m, reduce_scratch[g]);
        }
        row_stats[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_max = row_stats[0];

    /* denominator */
    float local_sum = 0.0f;
    for (uint j = tid; j < seq; j += 128u) {
        local_sum += exp(head_scratch[j] - row_max);
    }
    const float denom = tm_reduce_sum_128(local_sum, reduce_scratch, lane, simd_group);
    if (tid == 0u) {
        row_stats[1] = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *oh = out + (size_t)head * (size_t)head_dim;

    /* weighted values; each lane owns head_dim/128 = 1 dim when head_dim
     * is 128, but the loop is written for threadgroup 128 generally. */
    for (uint d = tid; d < head_dim; d += 128u) {
        float accv = 0.0f;
        for (uint j = 0; j < seq; ++j) {
            const float w = exp(head_scratch[j] - row_max) / denom;
            accv += w * vc[(size_t)j * (size_t)head_dim + d];
        }
        oh[d] = accv;
    }
}

/* out[i] = silu(gate[i]) * up[i], i < n. */
kernel void tm_silu_mul_f32(device const float *gate [[buffer(0)]],
                            device const float *up [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            constant uint &n [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    const float g = gate[gid];
    const float s = g / (1.0f + exp(-g));
    out[gid] = s * up[gid];
}

/* Dequantize one F16 matrix row to f32 (embedding rows for F16 artifacts). */
kernel void tm_f16_row_f32(device const half *w [[buffer(0)]],
                           device float *out [[buffer(1)]],
                           constant uint &row [[buffer(2)]],
                           constant uint &cols [[buffer(3)]],
                           constant uint &out_off [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    out[out_off + gid] = (float)w[(size_t)row * (size_t)cols + gid];
}

/* RMSNorm with an F16-weight table (fixture/F16-twin artifacts). */
kernel void tm_rmsnorm_f16w(device const float *x [[buffer(0)]],
                            device const half *weight [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant uint &cols [[buffer(3)]],
                            constant float &eps [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[4];
    const device float *xrow = x + (size_t)row * (size_t)cols;
    device float *yrow = y + (size_t)row * (size_t)cols;
    float acc = 0.0f;
    for (uint i = tid; i < cols; i += 128u) {
        acc += xrow[i] * xrow[i];
    }
    const float sumsq = tm_reduce_sum_128(acc, scratch, lane, simd_group);
    const float rms = 1.0f / sqrt(sumsq / (float)cols + eps);
    for (uint i = tid; i < cols; i += 128u) {
        yrow[i] = xrow[i] * rms * (float)weight[i];
    }
}

/* ---------------- C3: batched prefill kernels ---------------- */

/*
 * Batched packed Q8_0 GEMM: Y[b, :] = W · X[b, :] for b < batch. One SIMD
 * group per output row; each weight slice is loaded once and applied to
 * the whole batch, so weight traffic does not multiply with batch size.
 * Per-(row, b) arithmetic is exactly the single-token GEMV's: the same
 * lane-to-quant mapping (lane L: block L/4 of each 8-block step, quants
 * (L%4)*8..+7), the same block order, the same per-block expression and
 * the same simd_sum, so batch=1 is bit-identical to tm_q8_0_gemv.
 * X is [batch, cols] at element offset `x_off`; Y is [batch, rows] at
 * `y_off`; batch must be 1..=32 (validated C-side).
 */
kernel void tm_q8_0_gemm(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         constant uint &batch [[buffer(5)]],
                         constant uint &x_off [[buffer(6)]],
                         constant uint &y_off [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows || batch == 0u || batch > 32u) {
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 34u;
    const uint sub = lane % 4u;
    const uint lane_blk = lane / 4u;
    float acc[32];
    for (uint b = 0; b < batch; ++b) {
        acc[b] = 0.0f;
    }
    for (uint blk = lane_blk; blk < blocks; blk += 8u) {
        const device uchar *block = wrow + (size_t)blk * 34u;
        const float d = (float)(*(const device half *)block);
        const device packed_char4 *qv = (const device packed_char4 *)(block + 2u + sub * 8u);
        const float4 q0 = float4(char4(qv[0]));
        const float4 q1 = float4(char4(qv[1]));
        const uint col = blk * 32u + sub * 8u;
        for (uint b = 0; b < batch; ++b) {
            const device packed_float4 *xv =
                (const device packed_float4 *)(x + x_off + b * cols + col);
            const float4 x0 = float4(xv[0]);
            const float4 x1 = float4(xv[1]);
            acc[b] += d * (dot(q0, x0) + dot(q1, x1));
        }
    }
    for (uint b = 0; b < batch; ++b) {
        const float total = simd_sum(acc[b]);
        if (lane == 0u) {
            y[y_off + b * rows + row] = total;
        }
    }
}

/*
 * Batched Q8_0 GEMM on the SIMD-group matrix units (Phase 1C prefill):
 * Y[b, :] = W · X[b, :] for b < batch <= 16. One threadgroup (4 SIMD
 * groups) owns a 32-row x 16-token output tile. Per 32-wide K block it
 * dequantizes the W tile (d * q, exactly the CPU dequant) and loads the X
 * tile into threadgroup memory as f32; each SIMD group then accumulates
 * its 8 rows x 16 tokens with 8x8x8 f32 multiply-accumulates. Arithmetic
 * is f32 throughout (oracle-f32 profile); summation order differs from the
 * GEMV, so results are tolerance-registered, not bit-identical. Rows and
 * tokens beyond the bounds are zero-padded in the tile and never stored.
 */
#define TM_MMA_ROWS 32u
#define TM_MMA_TOK 16u

kernel void tm_q8_0_gemm_mma(device const uchar *w [[buffer(0)]],
                             device const float *x [[buffer(1)]],
                             device float *y [[buffer(2)]],
                             constant uint &rows [[buffer(3)]],
                             constant uint &cols [[buffer(4)]],
                             constant uint &batch [[buffer(5)]],
                             constant uint &x_off [[buffer(6)]],
                             constant uint &y_off [[buffer(7)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float wt[TM_MMA_ROWS * 32u];
    threadgroup float xt[TM_MMA_TOK * 32u];
    if (batch == 0u || batch > TM_MMA_TOK) {
        return;
    }
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 34u;
    const uint row0 = tg * TM_MMA_ROWS;

    simdgroup_float8x8 c0 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 c1 = simdgroup_float8x8(0.0f);

    /* W tile: thread t dequantizes 8 quants of row t/4. */
    const uint w_r = tid / 4u;
    const uint w_k = (tid % 4u) * 8u;
    /* X tile: thread t loads 4 values of token t/8. */
    const uint x_b = tid / 8u;
    const uint x_k = (tid % 8u) * 4u;

    for (uint blk = 0; blk < blocks; ++blk) {
        const uint row = row0 + w_r;
        if (row < rows) {
            const device uchar *block = w + (size_t)row * row_bytes + (size_t)blk * 34u;
            const float d = (float)(*(const device half *)block);
            const device char *qs = (const device char *)(block + 2u);
            for (uint i = 0; i < 8u; ++i) {
                wt[w_r * 32u + w_k + i] = d * (float)qs[w_k + i];
            }
        } else {
            for (uint i = 0; i < 8u; ++i) {
                wt[w_r * 32u + w_k + i] = 0.0f;
            }
        }
        if (x_b < batch) {
            const device float *xr = x + x_off + (size_t)x_b * cols + blk * 32u + x_k;
            for (uint i = 0; i < 4u; ++i) {
                xt[x_b * 32u + x_k + i] = xr[i];
            }
        } else {
            for (uint i = 0; i < 4u; ++i) {
                xt[x_b * 32u + x_k + i] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_float8x8 a;
            simdgroup_float8x8 b0;
            simdgroup_float8x8 b1;
            simdgroup_load(a, wt + sg * 8u * 32u + kk, 32);
            simdgroup_load(b0, xt + 0u * 32u + kk, 32, ulong2(0, 0), true);
            simdgroup_load(b1, xt + 8u * 32u + kk, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(c0, a, b0, c0);
            simdgroup_multiply_accumulate(c1, a, b1, c1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Stage the [32 rows x 16 tokens] tile, then store in bounds only. */
    threadgroup float *ot = wt;
    simdgroup_store(c0, ot + sg * 8u * TM_MMA_TOK + 0u, TM_MMA_TOK);
    simdgroup_store(c1, ot + sg * 8u * TM_MMA_TOK + 8u, TM_MMA_TOK);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < TM_MMA_ROWS * TM_MMA_TOK; idx += 128u) {
        const uint r = idx / TM_MMA_TOK;
        const uint b = idx % TM_MMA_TOK;
        if (row0 + r < rows && b < batch) {
            y[y_off + b * rows + row0 + r] = ot[r * TM_MMA_TOK + b];
        }
    }
}

/* Batched F16 GEMM with the same contract as tm_q8_0_gemm. */
kernel void tm_f16_gemm(device const half *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant uint &rows [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        constant uint &batch [[buffer(5)]],
                        constant uint &x_off [[buffer(6)]],
                        constant uint &y_off [[buffer(7)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows || batch == 0u || batch > 32u) {
        return;
    }
    const device half *wrow = w + (size_t)row * (size_t)cols;
    float acc[32];
    for (uint b = 0; b < batch; ++b) {
        acc[b] = 0.0f;
    }
    for (uint i = lane; i < cols; i += 32u) {
        const float wv = (float)wrow[i];
        for (uint b = 0; b < batch; ++b) {
            acc[b] += wv * x[x_off + b * cols + i];
        }
    }
    for (uint b = 0; b < batch; ++b) {
        const float total = simd_sum(acc[b]);
        if (lane == 0u) {
            y[y_off + b * rows + row] = total;
        }
    }
}

/*
 * Batched NeoX RoPE: rows = n_tokens * (n_q_heads + n_kv_heads); the first
 * n_tokens*n_q_heads rows rotate q, the rest rotate k. Token t is rotated
 * at position start_pos + t, so a prefilled chunk matches single-token
 * semantics exactly (same angles, same pair order).
 */
kernel void tm_rope_batch_f32(device float *q [[buffer(0)]],
                              device float *k [[buffer(1)]],
                              device const float *inv_freq [[buffer(2)]],
                              constant uint &start_pos [[buffer(3)]],
                              constant uint &n_tokens [[buffer(4)]],
                              constant uint &n_q_heads [[buffer(5)]],
                              constant uint &n_kv_heads [[buffer(6)]],
                              constant uint &head_dim [[buffer(7)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint lane [[thread_position_in_threadgroup]]) {
    const uint half_dim = head_dim / 2u;
    if (lane >= half_dim) {
        return;
    }
    const uint q_rows = n_tokens * n_q_heads;
    const uint total_rows = q_rows + n_tokens * n_kv_heads;
    if (row >= total_rows) {
        return;
    }
    const bool is_q = row < q_rows;
    const uint t = is_q ? row / n_q_heads : (row - q_rows) / n_kv_heads;
    const uint head = is_q ? row % n_q_heads : (row - q_rows) % n_kv_heads;
    const uint heads = is_q ? n_q_heads : n_kv_heads;
    device float *base = is_q ? q : k;
    device float *r = base + ((size_t)t * heads + head) * (size_t)head_dim;
    const float angle = (float)(start_pos + t) * inv_freq[lane];
    const float c = cos(angle);
    const float s = sin(angle);
    const float a = r[lane];
    const float bb = r[lane + half_dim];
    r[lane] = a * c - bb * s;
    r[lane + half_dim] = a * s + bb * c;
}

/* Batched KV writes: rows = n_tokens * kv_heads; token t lands at
 * start_pos + t. Same f16 rounding contract as the single-token kernel. */
kernel void tm_kv_store_batch_f32(device const float *k_src [[buffer(0)]],
                                  device const float *v_src [[buffer(1)]],
                                  device float *k_cache [[buffer(2)]],
                                  device float *v_cache [[buffer(3)]],
                                  constant uint &n_tokens [[buffer(4)]],
                                  constant uint &start_pos [[buffer(5)]],
                                  constant uint &cap [[buffer(6)]],
                                  constant uint &head_dim [[buffer(7)]],
                                  constant uint &kv_heads [[buffer(8)]],
                                  constant uint &round_f16 [[buffer(9)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint lane [[thread_position_in_threadgroup]]) {
    if (lane >= head_dim) {
        return;
    }
    const uint t = row / kv_heads;
    const uint head = row % kv_heads;
    if (t >= n_tokens) {
        return;
    }
    const uint pos = start_pos + t;
    const size_t src = ((size_t)t * kv_heads + head) * (size_t)head_dim + lane;
    const size_t dst = ((size_t)head * (size_t)cap + (size_t)pos) * (size_t)head_dim + lane;
    if (round_f16 != 0u) {
        k_cache[dst] = (float)(half)k_src[src];
        v_cache[dst] = (float)(half)v_src[src];
    } else {
        k_cache[dst] = k_src[src];
        v_cache[dst] = v_src[src];
    }
}

/*
 * Batched causal GQA attention: one threadgroup per (token, query head);
 * query token t attends keys 0..=(start_pos + t) — exactly the masking and
 * arithmetic of the single-token kernel per (t, head). Scratch region is
 * (t * heads + head) * cap floats.
 */
kernel void tm_attention_batch_f32(device const float *q [[buffer(0)]],
                                   device const float *k_cache [[buffer(1)]],
                                   device const float *v_cache [[buffer(2)]],
                                   device float *out [[buffer(3)]],
                                   device float *scratch [[buffer(4)]],
                                   constant uint &heads [[buffer(5)]],
                                   constant uint &n_tokens [[buffer(6)]],
                                   constant uint &start_pos [[buffer(7)]],
                                   constant uint &cap [[buffer(8)]],
                                   constant uint &head_dim [[buffer(9)]],
                                   constant uint &group [[buffer(10)]],
                                   constant float &scale [[buffer(11)]],
                                   uint gid [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float reduce_scratch[4];
    threadgroup float row_stats[2];
    const uint t = gid / heads;
    const uint head = gid % heads;
    if (t >= n_tokens) {
        return;
    }
    const uint pos = start_pos + t;
    const uint seq = pos + 1u;
    const uint kv_head = head / group;
    const device float *qh = q + ((size_t)t * heads + head) * (size_t)head_dim;
    const device float *kc = k_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim;
    const device float *vc = v_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim;
    device float *oh = out + ((size_t)t * heads + head) * (size_t)head_dim;
    device float *head_scratch = scratch + ((size_t)t * heads + head) * (size_t)cap;

    for (uint j = 0; j < seq; ++j) {
        float part = 0.0f;
        for (uint d = tid; d < head_dim; d += 128u) {
            part += qh[d] * kc[(size_t)j * (size_t)head_dim + d];
        }
        const float s = tm_reduce_sum_128(part, reduce_scratch, lane, simd_group) * scale;
        if (tid == 0u) {
            head_scratch[j] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint j = tid; j < seq; j += 128u) {
        local_max = max(local_max, head_scratch[j]);
    }
    local_max = simd_max(local_max);
    if (lane == 0u) {
        reduce_scratch[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float m = reduce_scratch[0];
        for (uint g = 1u; g < 4u; ++g) {
            m = max(m, reduce_scratch[g]);
        }
        row_stats[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_max = row_stats[0];

    float local_sum = 0.0f;
    for (uint j = tid; j < seq; j += 128u) {
        local_sum += exp(head_scratch[j] - row_max);
    }
    const float denom = tm_reduce_sum_128(local_sum, reduce_scratch, lane, simd_group);
    if (tid == 0u) {
        row_stats[1] = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < head_dim; d += 128u) {
        float accv = 0.0f;
        for (uint j = 0; j < seq; ++j) {
            const float w = exp(head_scratch[j] - row_max) / denom;
            accv += w * vc[(size_t)j * (size_t)head_dim + d];
        }
        oh[d] = accv;
    }
}

/* ---------------- Q35-2A: Gated DeltaNet operators ---------------- */

/*
 * One decode step of the depthwise causal convolution. `hist` holds the
 * previous (kernel - 1) rows of the QKV stream (channel-major, oldest
 * first), `x` the new row, `w` the `ssm_conv1d` weights in GGUF memory
 * order (channel-major, taps fastest: w[c * kernel + tap]). The window is
 * hist followed by x, so the output row is the graph's `ggml_ssm_conv`
 * result for this token. One thread per channel.
 */
kernel void tm_qwen35_conv_step(device const float *hist [[buffer(0)]],
                                device const float *x [[buffer(1)]],
                                device const float *w [[buffer(2)]],
                                device float *y [[buffer(3)]],
                                constant uint &channels [[buffer(4)]],
                                constant uint &kern [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= channels) {
        return;
    }
    const uint k = kern;
    float acc = 0.0f;
    for (uint j = 0; j + 1u < k; ++j) {
        acc += w[gid * k + j] * hist[j * channels + gid];
    }
    acc += w[gid * k + (k - 1u)] * x[gid];
    y[gid] = acc;
}

/*
 * Advance the convolution history after a conv_step: shift down by one
 * row and append the new input row. Runs after tm_qwen35_conv_step inside
 * the same group, so the next token sees exactly the last (kernel - 1)
 * rows of the window.
 */
kernel void tm_qwen35_conv_shift(device float *hist [[buffer(0)]],
                                 device const float *x [[buffer(1)]],
                                 constant uint &channels [[buffer(2)]],
                                 constant uint &kern [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= channels || kern < 2u) {
        return;
    }
    for (uint j = 0; j + 2u < kern; ++j) {
        hist[j * channels + gid] = hist[(j + 1u) * channels + gid];
    }
    hist[(kern - 2u) * channels + gid] = x[gid];
}

/*
 * Per-head L2 normalization with the tolerance epsilon inside the root:
 * x / sqrt(sum(x^2) + eps). One threadgroup of 128 threads per row
 * (head); shared with the full-attention Q/K head norms.
 */
kernel void tm_qwen35_l2norm_heads(device const float *x [[buffer(0)]],
                                   device float *y [[buffer(1)]],
                                   constant uint &cols [[buffer(2)]],
                                   constant float &eps [[buffer(3)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[4];
    const device float *xrow = x + (size_t)row * (size_t)cols;
    device float *yrow = y + (size_t)row * (size_t)cols;
    float acc = 0.0f;
    for (uint i = tid; i < cols; i += 128u) {
        acc += xrow[i] * xrow[i];
    }
    const float sumsq = tm_reduce_sum_128(acc, scratch, lane, simd_group);
    const float inv = 1.0f / sqrt(sumsq + eps);
    for (uint i = tid; i < cols; i += 128u) {
        yrow[i] = xrow[i] * inv;
    }
}

/*
 * Per-head gate preparation: beta = sigmoid(raw_beta) and
 * gate = ssm_a * softplus(alpha + dt_bias), with ggml's thresholded
 * softplus (x > 20 ? x : log(1 + exp(x))). One thread per head.
 */
kernel void tm_qwen35_beta_decay(device const float *raw_beta [[buffer(0)]],
                                 device const float *alpha [[buffer(1)]],
                                 device const float *dt_bias [[buffer(2)]],
                                 device const float *ssm_a [[buffer(3)]],
                                 device float *beta_out [[buffer(4)]],
                                 device float *gate_out [[buffer(5)]],
                                 constant uint &n [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    const float b = raw_beta[gid];
    beta_out[gid] = 1.0f / (1.0f + exp(-b));
    const float t = alpha[gid] + dt_bias[gid];
    const float sp = t > 20.0f ? t : log(1.0f + exp(t));
    gate_out[gid] = ssm_a[gid] * sp;
}

/*
 * One token of the autoregressive delta rule, matching
 * `build_delta_net_autoregressive` in per-element arithmetic:
 *   decay = exp(gate); s~ = s * decay; sk = s~^T k; d = beta (v - sk);
 *   s = s~ + k d^T; o = s^T (q * scale).
 * State layout: [head][i][j] with i over the key dim (contraction) and j
 * over the value dim, i-major f32.
 *
 * Every value column j is independent: its decay, its sk_j, d_j, column
 * update and o_j touch only column j. One thread per (head, column), so
 * the recurrence stays sequential where it must (over i, ascending, as in
 * the CPU rule) and parallel where it can. Adjacent threads read adjacent
 * j at the same i (coalesced). s~ is recomputed in the second pass
 * instead of stored: the same product, so the same rounding.
 * (Phase 1B: the earlier one-thread-per-head form was 90% of decode time.)
 */
kernel void tm_qwen35_gdn_step(device const float *q [[buffer(0)]],
                               device const float *k [[buffer(1)]],
                               device const float *v [[buffer(2)]],
                               device const float *gate [[buffer(3)]],
                               device const float *beta [[buffer(4)]],
                               device float *state [[buffer(5)]],
                               device float *out [[buffer(6)]],
                               constant uint &k_dim [[buffer(7)]],
                               constant uint &v_dim [[buffer(8)]],
                               constant float &q_scale [[buffer(9)]],
                               uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint head = gid.y;
    const uint sk = k_dim;
    const uint sv = v_dim;
    if (j >= sv) {
        return;
    }
    device float *s = state + (size_t)head * (size_t)sk * (size_t)sv;
    const device float *kk = k + (size_t)head * (size_t)sk;
    const device float *qq = q + (size_t)head * (size_t)sk;
    const float decay = exp(gate[head]);
    const float b = beta[head];

    float sk_acc = 0.0f;
    for (uint i = 0; i < sk; ++i) {
        sk_acc += (s[i * sv + j] * decay) * kk[i];
    }
    const float d = b * (v[(size_t)head * (size_t)sv + j] - sk_acc);
    float acc = 0.0f;
    for (uint i = 0; i < sk; ++i) {
        const float updated = s[i * sv + j] * decay + kk[i] * d;
        s[i * sv + j] = updated;
        acc += updated * (qq[i] * q_scale);
    }
    out[(size_t)head * (size_t)sv + j] = acc;
}

/*
 * Gated output normalization: per value head, RMS norm over the head's
 * value dim with the `ssm_norm` weights, times SiLU of the gate row.
 * One thread per head, same order as the CPU operator.
 */
kernel void tm_qwen35_gated_norm(device const float *o [[buffer(0)]],
                                 device const float *z [[buffer(1)]],
                                 device const float *weight [[buffer(2)]],
                                 device float *out [[buffer(3)]],
                                 constant uint &cols [[buffer(4)]],
                                 constant float &eps [[buffer(5)]],
                                 uint head [[thread_position_in_grid]]) {
    const device float *row = o + (size_t)head * (size_t)cols;
    const device float *zrow = z + (size_t)head * (size_t)cols;
    device float *orow = out + (size_t)head * (size_t)cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) {
        acc += row[i] * row[i];
    }
    const float inv = 1.0f / sqrt(acc / (float)cols + eps);
    for (uint i = 0; i < cols; ++i) {
        const float zz = zrow[i];
        const float silu = zz / (1.0f + exp(-zz));
        orow[i] = row[i] * inv * weight[i] * silu;
    }
}

/* ---------------- Q35-2B: full-attention operators ---------------- */

/*
 * Split the joint Q projection `qg` [heads, 2*head_dim] (query first,
 * gate second, per head) into q and gate, both [heads, head_dim].
 */
kernel void tm_qwen35_q_gate_split(device const float *qg [[buffer(0)]],
                                   device float *q_out [[buffer(1)]],
                                   device float *gate_out [[buffer(2)]],
                                   constant uint &head_dim [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    const uint hd = head_dim;
    const uint head = gid / hd;
    const uint i = gid % hd;
    q_out[gid] = qg[(size_t)head * (size_t)(2u * hd) + i];
    gate_out[gid] = qg[(size_t)head * (size_t)(2u * hd) + hd + i];
}

/*
 * Partial split-half RoPE (IMRoPE reduced to text positions): rotate only
 * the first `n_rot` dims of each head row, pairs (i, i + n_rot/2), angle
 * pos * inv_freq[i] (inv_freq holds n_rot/2 entries); dims at or beyond
 * n_rot pass through untouched. One threadgroup per head over q then k
 * (like tm_rope_f32).
 */
kernel void tm_qwen35_rope_partial_f32(device float *q [[buffer(0)]],
                                       device float *k [[buffer(1)]],
                                       device const float *inv_freq [[buffer(2)]],
                                       constant uint &pos [[buffer(3)]],
                                       constant uint &n_q_heads [[buffer(4)]],
                                       constant uint &n_kv_heads [[buffer(5)]],
                                       constant uint &head_dim [[buffer(6)]],
                                       constant uint &n_rot [[buffer(7)]],
                                       uint head [[threadgroup_position_in_grid]],
                                       uint lane [[thread_position_in_threadgroup]]) {
    const uint half_rot = n_rot / 2u;
    if (lane >= half_rot) {
        return;
    }
    const bool is_q = head < n_q_heads;
    const uint local_head = is_q ? head : head - n_q_heads;
    device float *base = is_q ? q : k;
    device float *row = base + (size_t)local_head * (size_t)head_dim;
    const float angle = (float)pos * inv_freq[lane];
    const float c = cos(angle);
    const float s = sin(angle);
    const float a = row[lane];
    const float b = row[lane + half_rot];
    row[lane] = a * c - b * s;
    row[lane + half_rot] = a * s + b * c;
}

/* Packed f16 KV store: k_cache[h][pos][d] = (half) k_src[h][d], true
 * two-byte entries (the declared packed cache contract), same for v. */
kernel void tm_qwen35_kv_store_f16(device const float *k_src [[buffer(0)]],
                                   device const float *v_src [[buffer(1)]],
                                   device half *k_cache [[buffer(2)]],
                                   device half *v_cache [[buffer(3)]],
                                   constant uint &pos [[buffer(4)]],
                                   constant uint &cap [[buffer(5)]],
                                   constant uint &head_dim [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    const size_t head = (size_t)gid / (size_t)head_dim;
    const size_t lane = (size_t)gid % (size_t)head_dim;
    const size_t src = head * (size_t)head_dim + lane;
    const size_t dst = (head * (size_t)cap + (size_t)pos) * (size_t)head_dim + lane;
    k_cache[dst] = (half)k_src[src];
    v_cache[dst] = (half)v_src[src];
}

/*
 * Causal GQA attention over a packed f16 cache (values decode to f32 for
 * the arithmetic, mirroring the CPU reference's packed-cache path).
 * One threadgroup (4 SIMD groups) per head. Scores: each SIMD group takes
 * positions sg, sg+4, ... and reduces its dot product with simd_sum (no
 * threadgroup barrier per position). Softmax weights are normalized once
 * per position into the scratch row, then every value dim accumulates
 * them (Phase 1C: the per-position barrier and the per-dim recomputation
 * of exp() dominated prefill attention).
 */
kernel void tm_qwen35_attention_f16(device const float *q [[buffer(0)]],
                                    device const half *k_cache [[buffer(1)]],
                                    device const half *v_cache [[buffer(2)]],
                                    device float *out [[buffer(3)]],
                                    device float *scratch [[buffer(4)]],
                                    constant uint &pos [[buffer(5)]],
                                    constant uint &cap [[buffer(6)]],
                                    constant uint &head_dim [[buffer(7)]],
                                    constant uint &group [[buffer(8)]],
                                    constant float &scale [[buffer(9)]],
                                    uint head [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float reduce_scratch[4];
    threadgroup float row_stats[2];
    const uint kv_head = head / group;
    const device float *qh = q + (size_t)head * (size_t)head_dim;
    const device half *kc = k_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim;
    const device half *vc = v_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim;
    device float *head_scratch = scratch + (size_t)head * (size_t)cap;
    const uint seq = pos + 1u;

    for (uint j = simd_group; j < seq; j += 4u) {
        const device half *kr = kc + (size_t)j * (size_t)head_dim;
        float part = 0.0f;
        for (uint d = lane; d < head_dim; d += 32u) {
            part += qh[d] * (float)kr[d];
        }
        const float s = simd_sum(part) * scale;
        if (lane == 0u) {
            head_scratch[j] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint j = tid; j < seq; j += 128u) {
        local_max = max(local_max, head_scratch[j]);
    }
    local_max = simd_max(local_max);
    if (lane == 0u) {
        reduce_scratch[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float m = reduce_scratch[0];
        for (uint g = 1u; g < 4u; ++g) {
            m = max(m, reduce_scratch[g]);
        }
        row_stats[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_max = row_stats[0];

    float local_sum = 0.0f;
    for (uint j = tid; j < seq; j += 128u) {
        local_sum += exp(head_scratch[j] - row_max);
    }
    const float denom = tm_reduce_sum_128(local_sum, reduce_scratch, lane, simd_group);
    if (tid == 0u) {
        row_stats[1] = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float norm = row_stats[1];
    for (uint j = tid; j < seq; j += 128u) {
        head_scratch[j] = exp(head_scratch[j] - row_max) / norm;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    device float *oh = out + (size_t)head * (size_t)head_dim;

    for (uint d = tid; d < head_dim; d += 128u) {
        float accv = 0.0f;
        for (uint j = 0; j < seq; ++j) {
            accv += head_scratch[j] * (float)vc[(size_t)j * (size_t)head_dim + d];
        }
        oh[d] = accv;
    }
}

/* attn[i] *= sigmoid(gate[i]) — the joint Q-projection output gate. */
kernel void tm_qwen35_attn_gate(device float *attn [[buffer(0)]],
                                device const float *gate [[buffer(1)]],
                                constant uint &n [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    attn[gid] *= 1.0f / (1.0f + exp(-gate[gid]));
}

/* x[i] = silu(x[i]) in place (conv output activation, graph order). */
kernel void tm_qwen35_silu(device float *x [[buffer(0)]],
                           constant uint &n [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    const float v = x[gid];
    x[gid] = v / (1.0f + exp(-v));
}

/* Split the conv output [3*width] into q, k, v [width] each (graph order:
 * q first, then k, then v). */
kernel void tm_qwen35_split_qkv(device const float *y [[buffer(0)]],
                                device float *q [[buffer(1)]],
                                device float *k [[buffer(2)]],
                                device float *v [[buffer(3)]],
                                constant uint &width [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= width) {
        return;
    }
    q[gid] = y[gid];
    k[gid] = y[width + gid];
    v[gid] = y[2u * width + gid];
}
