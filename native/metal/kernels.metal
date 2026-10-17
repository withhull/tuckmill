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
