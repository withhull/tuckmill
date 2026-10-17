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
 * Q8_0 blocks are 34 bytes: [f16 d][i8 qs; 32]; dequant(w) = d * q. One
 * threadgroup (one SIMD group of 32 lanes) per output row; lane L owns
 * element L of every block, so loads within a row are coalesced and the
 * per-block partial reductions use simd_sum. `cols` must be a positive
 * multiple of 32 (validated C-side).
 */
kernel void tm_q8_0_gemv(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows) {
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 34u;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        const device uchar *blk = wrow + (size_t)b * 34u;
        const float d = (float)(*(const device half *)blk);
        const float q = (float)((const device char *)(blk + 2))[lane];
        acc += d * q * x[b * 32u + lane];
    }
    const float total = simd_sum(acc);
    if (lane == 0u) {
        y[row] = total;
    }
}

/* ---------------- C2: transformer operation kernels ---------------- */

/* Dequantize one Q8_0 matrix row (or embedding row) to f32:
 * out[d] = d_scale(d/32) * q(d). One thread per output element. */
kernel void tm_q8_0_row_f32(device const uchar *w [[buffer(0)]],
                            device float *out [[buffer(1)]],
                            constant uint &row [[buffer(2)]],
                            constant uint &cols [[buffer(3)]],
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
    out[gid] = d * q;
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
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    out[gid] = (float)w[(size_t)row * (size_t)cols + gid];
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
