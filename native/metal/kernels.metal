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

/*
 * Greedy argmax over f32 logits, exactly the CPU rule
 * (`tuckmill_backend::sampling::argmax`): NaN entries are skipped, the
 * largest value wins, and ties go to the lowest index. That order is
 * associative, so the two-pass parallel reduction returns the sequential
 * answer. TM_ARGMAX_NONE means every entry was NaN.
 */
#define TM_ARGMAX_NONE 0xFFFFFFFFu
#define TM_ARGMAX_GROUPS 64u
#define TM_ARGMAX_THREADS 256u

inline bool tm_argmax_better(float av, uint ai, float bv, uint bi) {
    if (ai == TM_ARGMAX_NONE) {
        return false;
    }
    if (bi == TM_ARGMAX_NONE) {
        return true;
    }
    return av > bv || (av == bv && ai < bi);
}

inline void tm_argmax_simd(thread float &v, thread uint &i) {
    for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(v, off);
        const uint oi = simd_shuffle_down(i, off);
        if (tm_argmax_better(ov, oi, v, i)) {
            v = ov;
            i = oi;
        }
    }
}

inline void tm_argmax_threadgroup(thread float &v, thread uint &i,
                                  threadgroup float *tv, threadgroup uint *ti,
                                  uint lane, uint sg) {
    tm_argmax_simd(v, i);
    if (lane == 0u) {
        tv[sg] = v;
        ti[sg] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        v = lane < TM_ARGMAX_THREADS / 32u ? tv[lane] : -INFINITY;
        i = lane < TM_ARGMAX_THREADS / 32u ? ti[lane] : TM_ARGMAX_NONE;
        tm_argmax_simd(v, i);
    }
}

/* Pass 1: each of TM_ARGMAX_GROUPS threadgroups reduces a strided slice. */
kernel void tm_argmax_partial(device const float *x [[buffer(0)]],
                              device float *part_v [[buffer(1)]],
                              device uint *part_i [[buffer(2)]],
                              constant uint &n [[buffer(3)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tv[TM_ARGMAX_THREADS / 32u];
    threadgroup uint ti[TM_ARGMAX_THREADS / 32u];
    float v = -INFINITY;
    uint i = TM_ARGMAX_NONE;
    for (uint k = tg * TM_ARGMAX_THREADS + tid; k < n; k += TM_ARGMAX_GROUPS * TM_ARGMAX_THREADS) {
        const float xv = x[k];
        if (!isnan(xv) && tm_argmax_better(xv, k, v, i)) {
            v = xv;
            i = k;
        }
    }
    tm_argmax_threadgroup(v, i, tv, ti, lane, sg);
    if (tid == 0u) {
        part_v[tg] = v;
        part_i[tg] = i;
    }
}

/* Pass 2: one threadgroup reduces the partials into out[out_index]. */
kernel void tm_argmax_final(device const float *part_v [[buffer(0)]],
                            device const uint *part_i [[buffer(1)]],
                            device uint *out [[buffer(2)]],
                            constant uint &out_index [[buffer(3)]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tv[TM_ARGMAX_THREADS / 32u];
    threadgroup uint ti[TM_ARGMAX_THREADS / 32u];
    float v = tid < TM_ARGMAX_GROUPS ? part_v[tid] : -INFINITY;
    uint i = tid < TM_ARGMAX_GROUPS ? part_i[tid] : TM_ARGMAX_NONE;
    tm_argmax_threadgroup(v, i, tv, ti, lane, sg);
    if (tid == 0u) {
        out[out_index] = i;
    }
}

/*
 * Embedding lookup whose row is a token id produced on the device
 * (ids[slot], written by tm_argmax_final), so the host can encode the next
 * decode step before it has read the token. An id outside the table (the
 * all-NaN marker) writes zeros; the host fails that request when it reads
 * the marker, so the row is never used.
 */
kernel void tm_q8_0_row_indirect_f32(device const uchar *w [[buffer(0)]],
                                     device float *out [[buffer(1)]],
                                     device const uint *ids [[buffer(2)]],
                                     constant uint &slot [[buffer(3)]],
                                     constant uint &rows [[buffer(4)]],
                                     constant uint &cols [[buffer(5)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint row = ids[slot];
    if (row >= rows) {
        out[gid] = 0.0f;
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *blk = w + ((size_t)row * (size_t)blocks + gid / 32u) * 34u;
    const float d = (float)(*(const device half *)blk);
    const float q = (float)((const device char *)(blk + 2))[gid % 32u];
    out[gid] = d * q;
}

kernel void tm_f16_row_indirect_f32(device const half *w [[buffer(0)]],
                                    device float *out [[buffer(1)]],
                                    device const uint *ids [[buffer(2)]],
                                    constant uint &slot [[buffer(3)]],
                                    constant uint &rows [[buffer(4)]],
                                    constant uint &cols [[buffer(5)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint row = ids[slot];
    out[gid] = row < rows ? (float)w[(size_t)row * (size_t)cols + gid] : 0.0f;
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
/* k_cache[h][pos][d] = k_src[h][d] (same for v). One threadgroup per kv
 * head over head_dim lanes. */
/*
 * Causal GQA attention for one query position. One threadgroup per query
 * head; 128 threads. Scores for keys 0..=pos land in scratch (per-head
 * region of `heads * cap` floats), softmax in registers/shared, weighted
 * value accumulation parallel over head_dim lanes. `scale` is
 * 1/sqrt(head_dim). No mask beyond causality (Qwen3 dense here has no
 * sliding window).
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
 * Batched Q8_0 GEMM on the SIMD-group matrix units (prefill):
 * Y[b, :] = W · X[b, :] for b < batch, as C (tokens x rows) = X · W^T.
 * One threadgroup (4 SIMD groups) owns a 64-row x 32-token output tile;
 * threadgroup tg covers row tile tg / token_tiles and token tile
 * tg % token_tiles, so neighbours share a weight tile and one launch covers
 * a whole prefill chunk. Per 32-wide K block (one Q8_0 block per row):
 * - every thread first loads its share of the next W block (dequantized,
 *   d * q, exactly the CPU dequant) and X block into registers, so device
 *   latency overlaps other SIMD groups' multiplies;
 * - it then stores them into threadgroup memory as contiguous 8x8
 *   fragments (stride-8, conflict-free simdgroup loads, no transpose):
 *   X fragments [token][k], W^T fragments [k][row];
 * - SIMD group sg computes rows (sg % 2) * 32 .. +32 and tokens
 *   (sg / 2) * 16 .. +16 with 8 f32 accumulators.
 * Full tiles store straight from the accumulators to device memory;
 * edge tiles stage through threadgroup memory and store in bounds only.
 * Arithmetic is f32 throughout (oracle-f32 profile); summation order
 * differs from the GEMV, so results are tolerance-registered, not
 * bit-identical. Layout and code shape follow the pinned llama.cpp
 * kernel_mul_mm, kept in f32: out-of-range rows and tokens load a clamped
 * in-range row instead of branching, loops unroll fully, block pointers
 * advance by increment, fragment indices are 16-bit, and SIMD barriers
 * separate the fragment loads from the multiplies (ADR-011).
 */
#define TM_MMA_ROWS 64u
#define TM_MMA_TOK 32u
#define TM_UNROLL _Pragma("clang loop unroll(full)")

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
    /* sa: 32 W^T fragments (4 k-blocks x 8 row-blocks), sb: 16 X fragments
     * (4 k-blocks x 4 token-blocks); 64 floats each. The edge-tile output
     * stage (64 x 32) reuses sa. */
    threadgroup float sa[TM_MMA_ROWS * 32u];
    threadgroup float sb[TM_MMA_TOK * 32u];

    const uint token_tiles = (batch + TM_MMA_TOK - 1u) / TM_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * TM_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, TM_MMA_TOK);
    const uint row0 = (tg / token_tiles) * TM_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 34u;

    /* W loader: thread t owns 16 quants (half a Q8_0 block) of row t / 2. */
    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    /* X loader: thread t owns 8 values of token t / 4. */
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    /* Fragment slots this thread writes. */
    threadgroup float *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup float *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u; /* first row-block of this SIMD group */
    const uint sg_t = (sg / 2u) * 2u; /* first token-block */
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    /* Out-of-range rows and tokens load a clamped in-range row; their
     * results land only in output positions that are never stored. */
    const device uchar *block = wrow;
    const device float *xblk = xrow;
    /* This SIMD group's fragment bases; each k-block of 8 advances them. */
    const threadgroup float *lsa = sa + 64u * sg_r;
    const threadgroup float *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, block += 34, xblk += 32) {
        float wv[16];
        const float d = (float)(*(const device half *)block);
        const device packed_char4 *qv = (const device packed_char4 *)(block + 2u + w_k);
        TM_UNROLL for (uint i = 0; i < 4u; ++i) {
            const float4 q = float4(char4(qv[i]));
            wv[4u * i + 0u] = d * q.x;
            wv[4u * i + 1u] = d * q.y;
            wv[4u * i + 2u] = d * q.z;
            wv[4u * i + 3u] = d * q.w;
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* W^T fragment element (k % 8, row % 8); the 16 k values span two
         * k-blocks of 8. */
        TM_UNROLL for (ushort i = 0; i < 16; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = wv[i];
        }
        /* X fragment row (token % 8), 8 contiguous k values. */
        *(threadgroup float4 *)(sb_dst) = xv0;
        *(threadgroup float4 *)(sb_dst + 4u) = xv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        TM_UNROLL for (ushort kb = 0; kb < 4; ++kb) {
            simdgroup_float8x8 ma[4];
            simdgroup_float8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            TM_UNROLL for (ushort i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TM_UNROLL for (ushort j = 0; j < 2; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TM_UNROLL for (ushort j = 0; j < 2; ++j) {
                TM_UNROLL for (ushort i = 0; i < 4; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + TM_MMA_ROWS <= rows && tile_batch == TM_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    /* Edge tile: stage [32 tokens x 64 rows], store in bounds only. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *ot = sa;
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            simdgroup_store(mc[j][i], ot + (sg_t + j) * 8u * TM_MMA_ROWS + (sg_r + i) * 8u,
                            TM_MMA_ROWS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < TM_MMA_ROWS * TM_MMA_TOK; idx += 128u) {
        const uint r = idx % TM_MMA_ROWS;
        const uint b = idx / TM_MMA_ROWS;
        if (row0 + r < rows && b < tile_batch) {
            y[y_off + (size_t)(tok0 + b) * rows + row0 + r] = ot[b * TM_MMA_ROWS + r];
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
/* Batched KV writes: rows = n_tokens * kv_heads; token t lands at
 * start_pos + t. Same f16 rounding contract as the single-token kernel. */
/*
 * Batched causal GQA attention: one threadgroup per (token, query head);
 * query token t attends keys 0..=(start_pos + t) — exactly the masking and
 * arithmetic of the single-token kernel per (t, head). Scratch region is
 * (t * heads + head) * cap floats.
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
 * n_rot pass through untouched; with n_rot == head_dim it is full NeoX
 * RoPE (dense Qwen3). One threadgroup per head over q then k.
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
/* One head's causal attention over a packed f16 (or f32) cache, shared by
 * the per-token and chunk kernels, so both run exactly this code. */
template <typename C>
inline void tm_q35_attend(const device float *qh, const device C *kc, const device C *vc,
                          device float *oh, device float *head_scratch, uint seq,
                          uint head_dim, float scale, uint tid, uint lane, uint simd_group,
                          threadgroup float *reduce_scratch, threadgroup float *row_stats) {
    for (uint j = simd_group; j < seq; j += 4u) {
        const device C *kr = kc + (size_t)j * (size_t)head_dim;
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

    for (uint d = tid; d < head_dim; d += 128u) {
        float accv = 0.0f;
        for (uint j = 0; j < seq; ++j) {
            accv += head_scratch[j] * (float)vc[(size_t)j * (size_t)head_dim + d];
        }
        oh[d] = accv;
    }
}

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
    tm_q35_attend(q + (size_t)head * (size_t)head_dim,
                  k_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim,
                  v_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim,
                  out + (size_t)head * (size_t)head_dim, scratch + (size_t)head * (size_t)cap,
                  pos + 1u, head_dim, scale, tid, lane, simd_group, reduce_scratch, row_stats);
}

/* Chunk-wide causal attention: threadgroup (head, t) attends query row t of
 * a chunk starting at `pos0` over positions 0..pos0+t, with the same code as
 * the per-token kernel. Every chunk token's K/V must already be stored.
 * Scratch rows are [n, heads, cap]. */
template <typename C>
kernel void tm_qwen35_attention_chunk(device const float *q [[buffer(0)]],
                                      device const C *k_cache [[buffer(1)]],
                                      device const C *v_cache [[buffer(2)]],
                                          device float *out [[buffer(3)]],
                                          device float *scratch [[buffer(4)]],
                                          constant uint &pos0 [[buffer(5)]],
                                          constant uint &cap [[buffer(6)]],
                                          constant uint &head_dim [[buffer(7)]],
                                          constant uint &group [[buffer(8)]],
                                          constant float &scale [[buffer(9)]],
                                          constant uint &heads [[buffer(10)]],
                                          uint2 tg [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]],
                                          uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float reduce_scratch[4];
    threadgroup float row_stats[2];
    const uint head = tg.x;
    const uint t = tg.y;
    const uint kv_head = head / group;
    const size_t row = (size_t)t * heads + head;
    tm_q35_attend(q + row * (size_t)head_dim,
                  k_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim,
                  v_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim,
                  out + row * (size_t)head_dim, scratch + row * (size_t)cap, pos0 + t + 1u,
                  head_dim, scale, tid, lane, simd_group, reduce_scratch, row_stats);
}

#define TM_ATTN_CHUNK_ARGS(C)                                                               \
    device const float *q [[buffer(0)]], device const C *k_cache [[buffer(1)]],             \
        device const C *v_cache [[buffer(2)]], device float *out [[buffer(3)]],             \
        device float *scratch [[buffer(4)]], constant uint &pos0 [[buffer(5)]],             \
        constant uint &cap [[buffer(6)]], constant uint &head_dim [[buffer(7)]],            \
        constant uint &group [[buffer(8)]], constant float &scale [[buffer(9)]],            \
        constant uint &heads [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],     \
        uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],   \
        uint simd_group [[simdgroup_index_in_threadgroup]]

template [[host_name("tm_qwen35_attention_f16_chunk")]] kernel void
tm_qwen35_attention_chunk<half>(TM_ATTN_CHUNK_ARGS(half));
template [[host_name("tm_qwen35_attention_f32_chunk")]] kernel void
tm_qwen35_attention_chunk<float>(TM_ATTN_CHUNK_ARGS(float));

/* Grouped-query causal attention for one KV head and one query row: the G
 * query heads that share KV head `tg.x` attend query row `tg.y` of a chunk
 * starting at `pos0` (decode is the n = 1 case, so prefill and decode run
 * one compiled kernel). Each K and V row is read once for all G heads
 * instead of once per head.
 * - Scores: SIMD groups take keys in turn; lane l holds head dims
 *   [l * EPL, (l + 1) * EPL) of every head's query in registers.
 * - Softmax: SIMD group h normalizes head h with SIMD-only reductions,
 *   with the per-element expression of tm_q35_attend.
 * - Values: SIMD groups split the keys (j = sg mod 4), lane l accumulates
 *   its dims for all G heads, and the four partials are summed in a fixed
 *   order.
 * Scratch rows are [n, heads, cap], as for tm_qwen35_attention_f16_chunk.
 * Summation order differs from tm_q35_attend (tolerance-qualified).
 * Instantiated for the (head_dim, group) pairs in use. */
template <uint HD, uint G, typename C>
kernel void tm_qwen35_attention_gqa(device const float *q [[buffer(0)]],
                                    device const C *k_cache [[buffer(1)]],
                                    device const C *v_cache [[buffer(2)]],
                                    device float *out [[buffer(3)]],
                                    device float *scratch [[buffer(4)]],
                                    constant uint &pos0 [[buffer(5)]],
                                    constant uint &cap [[buffer(6)]],
                                    constant uint &head_dim [[buffer(7)]],
                                    constant uint &group [[buffer(8)]],
                                    constant float &scale [[buffer(9)]],
                                    constant uint &heads [[buffer(10)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint EPL = (HD + 31u) / 32u;
    threadgroup float partial[4u * G * HD];
    if (head_dim != HD || group != G) {
        return;
    }
    const uint kv_head = tg.x;
    const uint t = tg.y;
    const uint seq = pos0 + t + 1u;
    const uint d0 = lane * EPL;
    const bool active = d0 < HD;
    const size_t row0 = (size_t)t * heads + (size_t)kv_head * G;
    const device C *kc = k_cache + (size_t)kv_head * (size_t)cap * HD;
    const device C *vc = v_cache + (size_t)kv_head * (size_t)cap * HD;

    float qv[G][EPL];
    for (uint h = 0; h < G; ++h) {
        for (uint e = 0; e < EPL; ++e) {
            qv[h][e] = active ? q[(row0 + h) * HD + d0 + e] : 0.0f;
        }
    }
    for (uint j = sg; j < seq; j += 4u) {
        float kv[EPL];
        for (uint e = 0; e < EPL; ++e) {
            kv[e] = active ? (float)kc[(size_t)j * HD + d0 + e] : 0.0f;
        }
        for (uint h = 0; h < G; ++h) {
            float part = 0.0f;
            for (uint e = 0; e < EPL; ++e) {
                part += qv[h][e] * kv[e];
            }
            const float s = simd_sum(part) * scale;
            if (lane == 0u) {
                scratch[(row0 + h) * cap + j] = s;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint h = sg; h < G; h += 4u) {
        device float *sh = scratch + (row0 + h) * cap;
        float m = -INFINITY;
        for (uint j = lane; j < seq; j += 32u) {
            m = max(m, sh[j]);
        }
        m = simd_max(m);
        float sum = 0.0f;
        for (uint j = lane; j < seq; j += 32u) {
            sum += exp(sh[j] - m);
        }
        const float norm = simd_sum(sum);
        for (uint j = lane; j < seq; j += 32u) {
            sh[j] = exp(sh[j] - m) / norm;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    float acc[G][EPL];
    for (uint h = 0; h < G; ++h) {
        for (uint e = 0; e < EPL; ++e) {
            acc[h][e] = 0.0f;
        }
    }
    for (uint j = sg; j < seq; j += 4u) {
        float vv[EPL];
        for (uint e = 0; e < EPL; ++e) {
            vv[e] = active ? (float)vc[(size_t)j * HD + d0 + e] : 0.0f;
        }
        for (uint h = 0; h < G; ++h) {
            const float p = scratch[(row0 + h) * cap + j];
            for (uint e = 0; e < EPL; ++e) {
                acc[h][e] += p * vv[e];
            }
        }
    }
    if (active) {
        for (uint h = 0; h < G; ++h) {
            for (uint e = 0; e < EPL; ++e) {
                partial[(sg * G + h) * HD + d0 + e] = acc[h][e];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < G * HD; idx += 128u) {
        const float total = ((partial[idx] + partial[G * HD + idx]) + partial[2u * G * HD + idx]) +
                            partial[3u * G * HD + idx];
        out[row0 * HD + idx] = total;
    }
}

#define TM_ATTN_GQA_ARGS(C)                                                               \
    device const float *q [[buffer(0)]], device const C *k_cache [[buffer(1)]],             \
        device const C *v_cache [[buffer(2)]], device float *out [[buffer(3)]],             \
        device float *scratch [[buffer(4)]], constant uint &pos0 [[buffer(5)]],              \
        constant uint &cap [[buffer(6)]], constant uint &head_dim [[buffer(7)]],             \
        constant uint &group [[buffer(8)]], constant float &scale [[buffer(9)]],             \
        constant uint &heads [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],      \
        uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],    \
        uint sg [[simdgroup_index_in_threadgroup]]

template [[host_name("tm_qwen35_attention_gqa_256x4")]] kernel void
tm_qwen35_attention_gqa<256u, 4u, half>(TM_ATTN_GQA_ARGS(half));
template [[host_name("tm_qwen35_attention_gqa_16x2")]] kernel void
tm_qwen35_attention_gqa<16u, 2u, half>(TM_ATTN_GQA_ARGS(half));
template [[host_name("tm_qwen35_attention_gqa_128x2")]] kernel void
tm_qwen35_attention_gqa<128u, 2u, half>(TM_ATTN_GQA_ARGS(half));
template [[host_name("tm_qwen35_attention_gqa_128x4")]] kernel void
tm_qwen35_attention_gqa<128u, 4u, half>(TM_ATTN_GQA_ARGS(half));
template [[host_name("tm_qwen35_attention_gqa_f32_16x2")]] kernel void
tm_qwen35_attention_gqa<16u, 2u, float>(TM_ATTN_GQA_ARGS(float));
template [[host_name("tm_qwen35_attention_gqa_f32_128x2")]] kernel void
tm_qwen35_attention_gqa<128u, 2u, float>(TM_ATTN_GQA_ARGS(float));
template [[host_name("tm_qwen35_attention_gqa_f32_128x4")]] kernel void
tm_qwen35_attention_gqa<128u, 4u, float>(TM_ATTN_GQA_ARGS(float));

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

/* ---------------- Chunk-wide Gated DeltaNet core (engine prefill) ----------
 * Each kernel processes all `n` tokens of a prefill chunk in one launch with
 * exactly the per-element arithmetic of its per-token counterpart above,
 * so chunked prefill equals running the per-token core n times. */

/* Causal convolution over a chunk. Token t's tap j (j < k-1) reads the
 * input at time t-(k-1)+j: from the chunk when that is >= 0, else from the
 * carried history (oldest first), exactly as tm_qwen35_conv_step sees it
 * after t shifts. One thread per (channel, token). */
kernel void tm_qwen35_conv_chunk(device const float *hist [[buffer(0)]],
                                 device const float *x [[buffer(1)]],
                                 device const float *w [[buffer(2)]],
                                 device float *y [[buffer(3)]],
                                 constant uint &channels [[buffer(4)]],
                                 constant uint &kern [[buffer(5)]],
                                 constant uint &n [[buffer(6)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= channels || t >= n) {
        return;
    }
    const uint k = kern;
    float acc = 0.0f;
    for (uint j = 0; j + 1u < k; ++j) {
        const int src = (int)t + (int)j - (int)(k - 1u);
        const float v = src >= 0 ? x[(size_t)src * channels + c]
                                 : hist[(size_t)(t + j) * channels + c];
        acc += w[c * k + j] * v;
    }
    acc += w[c * k + (k - 1u)] * x[(size_t)t * channels + c];
    y[(size_t)t * channels + c] = acc;
}

/* History after a chunk: the last k-1 inputs (oldest first), in place.
 * Reads index n+j > j, so ascending j never reads an overwritten slot. */
kernel void tm_qwen35_conv_chunk_shift(device float *hist [[buffer(0)]],
                                       device const float *x [[buffer(1)]],
                                       constant uint &channels [[buffer(2)]],
                                       constant uint &kern [[buffer(3)]],
                                       constant uint &n [[buffer(4)]],
                                       uint c [[thread_position_in_grid]]) {
    if (c >= channels || kern < 2u) {
        return;
    }
    const uint h = kern - 1u;
    for (uint j = 0; j < h; ++j) {
        const uint src = n + j;
        hist[(size_t)j * channels + c] =
            src < h ? hist[(size_t)src * channels + c] : x[(size_t)(src - h) * channels + c];
    }
}

/* Per-token q/k/v split of conv rows [n, 3*width] into [n, width] each. */
kernel void tm_qwen35_split_qkv_chunk(device const float *y [[buffer(0)]],
                                      device float *q [[buffer(1)]],
                                      device float *k [[buffer(2)]],
                                      device float *v [[buffer(3)]],
                                      constant uint &width [[buffer(4)]],
                                      constant uint &n [[buffer(5)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint t = gid.y;
    if (i >= width || t >= n) {
        return;
    }
    const size_t row = (size_t)t * 3u * width;
    q[(size_t)t * width + i] = y[row + i];
    k[(size_t)t * width + i] = y[row + width + i];
    v[(size_t)t * width + i] = y[row + 2u * width + i];
}

/* tm_qwen35_beta_decay over [n, per] rows; the per-head biases index by
 * gid % per. */
kernel void tm_qwen35_beta_decay_chunk(device const float *raw_beta [[buffer(0)]],
                                       device const float *alpha [[buffer(1)]],
                                       device const float *dt_bias [[buffer(2)]],
                                       device const float *ssm_a [[buffer(3)]],
                                       device float *beta_out [[buffer(4)]],
                                       device float *gate_out [[buffer(5)]],
                                       constant uint &total [[buffer(6)]],
                                       constant uint &per [[buffer(7)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= total) {
        return;
    }
    const uint h = gid % per;
    const float b = raw_beta[gid];
    beta_out[gid] = 1.0f / (1.0f + exp(-b));
    const float t = alpha[gid] + dt_bias[h];
    const float sp = t > 20.0f ? t : log(1.0f + exp(t));
    gate_out[gid] = ssm_a[h] * sp;
}

/* The delta rule over a chunk: thread (column j, head) walks tokens
 * t = 0..n-1 in order with exactly tm_qwen35_gdn_step's per-element
 * arithmetic, so the state after the chunk equals n single steps. Per
 * token, rows are `qk_stride` (q, k), `v_stride` (v, out) and `g_stride`
 * (gate, beta) wide; head h reads offset h*k_dim / h*v_dim / h. */
kernel void tm_qwen35_gdn_chunk(device const float *q [[buffer(0)]],
                                device const float *k [[buffer(1)]],
                                device const float *v [[buffer(2)]],
                                device const float *gate [[buffer(3)]],
                                device const float *beta [[buffer(4)]],
                                device float *state [[buffer(5)]],
                                device float *out [[buffer(6)]],
                                constant uint &k_dim [[buffer(7)]],
                                constant uint &v_dim [[buffer(8)]],
                                constant float &q_scale [[buffer(9)]],
                                constant uint &heads [[buffer(10)]],
                                constant uint &n [[buffer(11)]],
                                constant uint &qk_stride [[buffer(12)]],
                                constant uint &v_stride [[buffer(13)]],
                                constant uint &g_stride [[buffer(14)]],
                                uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint head = gid.y;
    const uint sk = k_dim;
    const uint sv = v_dim;
    if (j >= sv || head >= heads) {
        return;
    }
    device float *s = state + (size_t)head * (size_t)sk * (size_t)sv;
    for (uint t = 0; t < n; ++t) {
        const size_t qk_row = (size_t)t * qk_stride + (size_t)head * sk;
        const size_t v_row = (size_t)t * v_stride + (size_t)head * sv;
        const device float *kk = k + qk_row;
        const device float *qq = q + qk_row;
        const float decay = exp(gate[(size_t)t * g_stride + head]);
        const float b = beta[(size_t)t * g_stride + head];
        float sk_acc = 0.0f;
        for (uint i = 0; i < sk; ++i) {
            sk_acc += (s[i * sv + j] * decay) * kk[i];
        }
        const float d = b * (v[v_row + j] - sk_acc);
        float acc = 0.0f;
        for (uint i = 0; i < sk; ++i) {
            const float updated = s[i * sv + j] * decay + kk[i] * d;
            s[i * sv + j] = updated;
            acc += updated * (qq[i] * q_scale);
        }
        out[v_row + j] = acc;
    }
}



/* Partial split-half RoPE over a chunk: threadgroup (head, t) rotates q or
 * k row t at position pos0+t, exactly tm_qwen35_rope_partial_f32's math.
 * q rows are [n, n_q_heads, head_dim], k rows [n, n_kv_heads, head_dim]. */
kernel void tm_qwen35_rope_partial_chunk(device float *q [[buffer(0)]],
                                         device float *k [[buffer(1)]],
                                         device const float *inv_freq [[buffer(2)]],
                                         constant uint &pos0 [[buffer(3)]],
                                         constant uint &n_q_heads [[buffer(4)]],
                                         constant uint &n_kv_heads [[buffer(5)]],
                                         constant uint &head_dim [[buffer(6)]],
                                         constant uint &n_rot [[buffer(7)]],
                                         uint2 tg [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_threadgroup]]) {
    const uint half_rot = n_rot / 2u;
    if (lane >= half_rot) {
        return;
    }
    const uint head = tg.x;
    const uint t = tg.y;
    const bool is_q = head < n_q_heads;
    const uint local_head = is_q ? head : head - n_q_heads;
    device float *row = is_q
        ? q + ((size_t)t * n_q_heads + local_head) * (size_t)head_dim
        : k + ((size_t)t * n_kv_heads + local_head) * (size_t)head_dim;
    const float angle = (float)(pos0 + t) * inv_freq[lane];
    const float c = cos(angle);
    const float s = sin(angle);
    const float a = row[lane];
    const float b = row[lane + half_rot];
    row[lane] = a * c - b * s;
    row[lane + half_rot] = a * s + b * c;
}

/* KV store of a chunk's rows [n, kv_heads, head_dim] at positions
 * pos0+t: packed f16 (same rounding as tm_qwen35_kv_store_f16) or f32
 * (an exact copy). */
template <typename C>
kernel void tm_qwen35_kv_store_chunk(device const float *k_src [[buffer(0)]],
                                     device const float *v_src [[buffer(1)]],
                                     device C *k_cache [[buffer(2)]],
                                     device C *v_cache [[buffer(3)]],
                                         constant uint &pos0 [[buffer(4)]],
                                         constant uint &cap [[buffer(5)]],
                                         constant uint &head_dim [[buffer(6)]],
                                         constant uint &kv_heads [[buffer(7)]],
                                         constant uint &n [[buffer(8)]],
                                         uint gid [[thread_position_in_grid]]) {
    const uint per = kv_heads * head_dim;
    if (gid >= per * n) {
        return;
    }
    const size_t t = gid / per;
    const size_t rem = gid % per;
    const size_t head = rem / head_dim;
    const size_t lane = rem % head_dim;
    const size_t dst = (head * (size_t)cap + (size_t)pos0 + t) * (size_t)head_dim + lane;
    k_cache[dst] = (C)k_src[gid];
    v_cache[dst] = (C)v_src[gid];
}

#define TM_KV_CHUNK_ARGS(C)                                                                  \
    device const float *k_src [[buffer(0)]], device const float *v_src [[buffer(1)]],        \
        device C *k_cache [[buffer(2)]], device C *v_cache [[buffer(3)]],                     \
        constant uint &pos0 [[buffer(4)]], constant uint &cap [[buffer(5)]],                  \
        constant uint &head_dim [[buffer(6)]], constant uint &kv_heads [[buffer(7)]],         \
        constant uint &n [[buffer(8)]], uint gid [[thread_position_in_grid]]

template [[host_name("tm_qwen35_kv_store_f16_chunk")]] kernel void
tm_qwen35_kv_store_chunk<half>(TM_KV_CHUNK_ARGS(half));
template [[host_name("tm_qwen35_kv_store_f32_chunk")]] kernel void
tm_qwen35_kv_store_chunk<float>(TM_KV_CHUNK_ARGS(float));

/* tm_qwen35_gdn_chunk with the state column held in registers for the
 * whole chunk (read once, written once) instead of re-read from device
 * memory twice per token. Each column is split across P lanes of one
 * SIMD group, each holding SK / P rows; the two per-token dot products
 * are lane partial sums combined with XOR shuffles, so every lane of a
 * column sees the same total. A SIMD group covers C = 32 / P adjacent
 * columns and lane l works on column l % C, so neighbouring lanes touch
 * neighbouring state columns (coalesced state loads and stores). Same per-element expressions as the generic
 * kernel; the dot-product summation order differs (tolerance-qualified).
 * Decode's single-token step runs this kernel with n = 1, so chunked
 * prefill and decode stay bit-identical to each other. Instantiated for
 * the key dimensions in use (the array size must be a compile-time
 * constant); the grid is P * v_dim lanes per head in threadgroups of 32,
 * and v_dim must be a multiple of C. */
template <uint SK, uint P>
kernel void tm_qwen35_gdn_chunk_reg(device const float *q [[buffer(0)]],
                                    device const float *k [[buffer(1)]],
                                    device const float *v [[buffer(2)]],
                                    device const float *gate [[buffer(3)]],
                                    device const float *beta [[buffer(4)]],
                                    device float *state [[buffer(5)]],
                                    device float *out [[buffer(6)]],
                                    constant uint &k_dim [[buffer(7)]],
                                    constant uint &v_dim [[buffer(8)]],
                                    constant float &q_scale [[buffer(9)]],
                                    constant uint &heads [[buffer(10)]],
                                    constant uint &n [[buffer(11)]],
                                    constant uint &qk_stride [[buffer(12)]],
                                    constant uint &v_stride [[buffer(13)]],
                                    constant uint &g_stride [[buffer(14)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    constexpr uint RPL = SK / P;
    constexpr uint C = 32u / P;
    const uint lane = gid.x % 32u;
    const uint j = (gid.x / 32u) * C + lane % C;
    const uint part = lane / C;
    const uint head = gid.y;
    const uint sv = v_dim;
    /* Whole columns only: a column's P lanes all pass or all return, so
     * the shuffles below always see their partners. */
    if (j >= sv || head >= heads || k_dim != SK) {
        return;
    }
    const uint r0 = part * RPL;
    device float *s = state + (size_t)head * (size_t)SK * (size_t)sv;
    float col[RPL];
    for (uint i = 0; i < RPL; ++i) {
        col[i] = s[(r0 + i) * sv + j];
    }
    for (uint t = 0; t < n; ++t) {
        const size_t qk_row = (size_t)t * qk_stride + (size_t)head * SK + r0;
        const size_t v_row = (size_t)t * v_stride + (size_t)head * sv;
        const device float *kk = k + qk_row;
        const device float *qq = q + qk_row;
        const float decay = exp(gate[(size_t)t * g_stride + head]);
        const float b = beta[(size_t)t * g_stride + head];
        float sk_acc = 0.0f;
        for (uint i = 0; i < RPL; ++i) {
            sk_acc += (col[i] * decay) * kk[i];
        }
        for (uint m = C; m < 32u; m <<= 1) {
            sk_acc += simd_shuffle_xor(sk_acc, (ushort)m);
        }
        const float d = b * (v[v_row + j] - sk_acc);
        float acc = 0.0f;
        for (uint i = 0; i < RPL; ++i) {
            const float updated = col[i] * decay + kk[i] * d;
            col[i] = updated;
            acc += updated * (qq[i] * q_scale);
        }
        for (uint m = C; m < 32u; m <<= 1) {
            acc += simd_shuffle_xor(acc, (ushort)m);
        }
        if (part == 0u) {
            out[v_row + j] = acc;
        }
    }
    for (uint i = 0; i < RPL; ++i) {
        s[(r0 + i) * sv + j] = col[i];
    }
}

/* Lanes per state column (must match the host's TM_GDN_LANES). */
#define TM_GDN_LANES 8u

#define TM_GDN_REG_ARGS                                                          \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],      \
        device const float *v [[buffer(2)]], device const float *gate [[buffer(3)]], \
        device const float *beta [[buffer(4)]], device float *state [[buffer(5)]],  \
        device float *out [[buffer(6)]], constant uint &k_dim [[buffer(7)]],        \
        constant uint &v_dim [[buffer(8)]], constant float &q_scale [[buffer(9)]],  \
        constant uint &heads [[buffer(10)]], constant uint &n [[buffer(11)]],       \
        constant uint &qk_stride [[buffer(12)]], constant uint &v_stride [[buffer(13)]], \
        constant uint &g_stride [[buffer(14)]], uint2 gid [[thread_position_in_grid]]

template [[host_name("tm_qwen35_gdn_chunk_k128")]] kernel void
tm_qwen35_gdn_chunk_reg<128u, TM_GDN_LANES>(TM_GDN_REG_ARGS);

