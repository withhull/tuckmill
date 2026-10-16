//! K01 first increment: fused, packed matrix-vector dot kernels.
//!
//! Scope (honest): aarch64 NEON kernels for the types that dominate real
//! usage — Q8_0 weights × f32 activations (the scalar oracle path), Q8_0 ×
//! q8_0 quantized activations (the llama-cpu diagnostic profile), and F16
//! weights. They fuse dequantization into the dot product (no f32 scratch
//! materialization) and use integer/convert SIMD lanes. Everything else
//! keeps the scalar reference path, and every kernel is validated against a
//! scalar reference on randomized data (tests below) before being trusted.
//!
//! Numerical contract: the integer dot for Q8_0 × Q8_0 is exact (int32);
//! f32 accumulation order differs from the scalar reference by design and is
//! covered by the registered verification tolerance. Q8_0 × f32 converts
//! int8 lanes to f32 (exact for |q| <= 127) and multiplies by the f16 scale
//! per block, mirroring the scalar path's arithmetic.

use crate::tensor::{f16_to_f32, f32_to_f16_bits, quantize_row_q8_0};

/// Scalar reference kernels (also the non-aarch64 implementation and the
/// test oracle). Kept intentionally simple.
pub mod scalar {
    use super::*;

    pub fn dot_q8_0_x_f32(row: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        let mut xi = 0usize;
        for block in row.chunks_exact(34) {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let mut block_sum = 0.0f32;
            for q in 0..32 {
                let w = (block[2 + q] as i8) as f32;
                block_sum += w * x[xi + q];
            }
            acc += d * block_sum;
            xi += 32;
        }
        acc
    }

    pub fn dot_q8_0_q8_0(row: &[u8], x_qs: &[[i8; 32]], x_d: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (blk, block) in row.chunks_exact(34).enumerate() {
            let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qx = &x_qs[blk];
            let mut sumi: i32 = 0;
            for q in 0..32 {
                sumi += ((block[2 + q] as i8) as i32) * (qx[q] as i32);
            }
            acc += (d_w * x_d[blk]) * (sumi as f32);
        }
        acc
    }

    pub fn dot_f16_x_f32(row: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (i, pair) in row.chunks_exact(2).enumerate() {
            let w = f16_to_f32(u16::from_le_bytes([pair[0], pair[1]]));
            acc += w * x[i];
        }
        acc
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// Q8_0 weight row × f32 activation, fused (no scratch).
    pub fn dot_q8_0_x_f32(row: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        let mut xi = 0usize;
        for block in row.chunks_exact(34) {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qs = &block[2..34];
            // SAFETY (whole kernel): qs has 32 elements, x has at least 32
            // remaining elements (caller contract: bytes_per_row/34*32 ==
            // elems_per_row), all loads are in-bounds and aligned via
            // unaligned load intrinsics.
            unsafe {
                let q_lo = vld1q_s8(qs.as_ptr() as *const i8);
                let q_hi = vld1q_s8(qs.as_ptr().add(16) as *const i8);
                let x_lo = vld1q_f32(x.as_ptr().add(xi));
                let x_hi = vld1q_f32(x.as_ptr().add(xi + 4));
                let x_lo2 = vld1q_f32(x.as_ptr().add(xi + 8));
                let x_hi2 = vld1q_f32(x.as_ptr().add(xi + 12));
                let x_b1 = vld1q_f32(x.as_ptr().add(xi + 16));
                let x_b2 = vld1q_f32(x.as_ptr().add(xi + 20));
                let x_b3 = vld1q_f32(x.as_ptr().add(xi + 24));
                let x_b4 = vld1q_f32(x.as_ptr().add(xi + 28));

                let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(q_lo)))));
                let w1 = vcvtq_f32_s32(vmovl_high_s16(vmovl_s8(vget_low_s8(q_lo))));
                let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(q_lo)))));
                let w3 = vcvtq_f32_s32(vmovl_high_s16(vmovl_s8(vget_high_s8(q_lo))));
                let w4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(q_hi)))));
                let w5 = vcvtq_f32_s32(vmovl_high_s16(vmovl_s8(vget_low_s8(q_hi))));
                let w6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(q_hi)))));
                let w7 = vcvtq_f32_s32(vmovl_high_s16(vmovl_s8(vget_high_s8(q_hi))));

                let mut s0 = vmulq_f32(w0, x_lo);
                s0 = vfmaq_f32(s0, w1, x_hi);
                s0 = vfmaq_f32(s0, w2, x_lo2);
                s0 = vfmaq_f32(s0, w3, x_hi2);
                let mut s1 = vmulq_f32(w4, x_b1);
                s1 = vfmaq_f32(s1, w5, x_b2);
                s1 = vfmaq_f32(s1, w6, x_b3);
                s1 = vfmaq_f32(s1, w7, x_b4);
                let block_sum = vaddvq_f32(vaddq_f32(s0, s1));
                acc += d * block_sum;
            }
            xi += 32;
        }
        acc
    }

    /// Q8_0 weight row × q8_0-quantized activation (exact int32 block dot).
    pub fn dot_q8_0_q8_0(row: &[u8], x_qs: &[[i8; 32]], x_d: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (blk, block) in row.chunks_exact(34).enumerate() {
            let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qs = &block[2..34];
            let qx = &x_qs[blk];
            // SAFETY: qs and qx both have 32 elements; int8 loads only.
            let sumi: i32 = unsafe {
                let w_lo = vld1q_s8(qs.as_ptr() as *const i8);
                let w_hi = vld1q_s8(qs.as_ptr().add(16) as *const i8);
                let x_lo = vld1q_s8(qx.as_ptr());
                let x_hi = vld1q_s8(qx.as_ptr().add(16));
                let p0 = vpaddlq_s16(vmull_s8(vget_low_s8(w_lo), vget_low_s8(x_lo)));
                let p1 = vpaddlq_s16(vmull_s8(vget_high_s8(w_lo), vget_high_s8(x_lo)));
                let p2 = vpaddlq_s16(vmull_s8(vget_low_s8(w_hi), vget_low_s8(x_hi)));
                let p3 = vpaddlq_s16(vmull_s8(vget_high_s8(w_hi), vget_high_s8(x_hi)));
                vaddvq_s32(vaddq_s32(vaddq_s32(p0, p1), vaddq_s32(p2, p3)))
            };
            acc += (d_w * x_d[blk]) * (sumi as f32);
        }
        acc
    }

    /// F16 weight row × f32 activation, fused via f16→f32 conversion.
    pub fn dot_f16_x_f32(row: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        let mut xi = 0usize;
        let mut rest = row;
        // 16 weights per iteration (two 4-wide f32 accumulators, four batches).
        while rest.len() >= 32 {
            // SAFETY: rest has >= 32 bytes -> 16 halves; x has >= 16 remaining.
            unsafe {
                let h0 = vld1_u16(rest.as_ptr() as *const u16);
                let h1 = vld1_u16(rest.as_ptr().add(8) as *const u16);
                let h2 = vld1_u16(rest.as_ptr().add(16) as *const u16);
                let h3 = vld1_u16(rest.as_ptr().add(24) as *const u16);
                let w0 = vcvt_f32_f16(vreinterpret_f16_u16(h0));
                let w1 = vcvt_f32_f16(vreinterpret_f16_u16(h1));
                let w2 = vcvt_f32_f16(vreinterpret_f16_u16(h2));
                let w3 = vcvt_f32_f16(vreinterpret_f16_u16(h3));
                let x0 = vld1q_f32(x.as_ptr().add(xi));
                let x1 = vld1q_f32(x.as_ptr().add(xi + 4));
                let x2 = vld1q_f32(x.as_ptr().add(xi + 8));
                let x3 = vld1q_f32(x.as_ptr().add(xi + 12));
                let mut s = vmulq_f32(w0, x0);
                s = vfmaq_f32(s, w1, x1);
                s = vfmaq_f32(s, w2, x2);
                s = vfmaq_f32(s, w3, x3);
                acc += vaddvq_f32(s);
            }
            rest = &rest[32..];
            xi += 16;
        }
        // Tail (rows are even-sized for all supported geometries; F16 has no
        // block size, so a scalar tail is correct and cheap).
        for (i, pair) in rest.chunks_exact(2).enumerate() {
            let w = f16_to_f32(u16::from_le_bytes([pair[0], pair[1]]));
            acc += w * x[xi + i];
        }
        acc
    }
}

/// Fused Q8_0 weight row × f32 activations.
pub fn dot_q8_0_x_f32(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_q8_0_x_f32(row, x)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::dot_q8_0_x_f32(row, x)
    }
}

/// Q8_0 × q8_0 (both int8): exact int32 block dot with f16 scales.
pub fn dot_q8_0_q8_0(row: &[u8], x_qs: &[[i8; 32]], x_d: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_q8_0_q8_0(row, x_qs, x_d)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::dot_q8_0_q8_0(row, x_qs, x_d)
    }
}

/// Fused F16 weight row × f32 activations.
pub fn dot_f16_x_f32(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_f16_x_f32(row, x)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::dot_f16_x_f32(row, x)
    }
}

/// Convenience for callers that need q8 availability (tests, tooling).
pub fn quantize_for_dot(x: &[f32]) -> (Vec<[i8; 32]>, Vec<f32>) {
    let mut qs = Vec::new();
    let mut ds = Vec::new();
    quantize_row_q8_0(x, &mut qs, &mut ds);
    (qs, ds)
}

/// Encode f32 -> f16 bits (used by tests and callers).
pub fn f16_bits(v: f32) -> u16 {
    f32_to_f16_bits(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift for reproducible randomized cases.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self, scale: f32) -> f32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let u = (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32;
            ((u / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
        }
    }

    fn random_q8_0_row(rng: &mut Rng, blocks: usize) -> Vec<u8> {
        let mut row = Vec::with_capacity(blocks * 34);
        for _ in 0..blocks {
            let d = rng.next_f32(0.05).abs() + 1e-4;
            row.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
            for _ in 0..32 {
                let q = (rng.next_f32(127.0)).round().clamp(-127.0, 127.0) as i8;
                row.push(q as u8);
            }
        }
        row
    }

    fn random_f16_row(rng: &mut Rng, elems: usize) -> Vec<u8> {
        let mut row = Vec::with_capacity(elems * 2);
        for _ in 0..elems {
            row.extend_from_slice(&f32_to_f16_bits(rng.next_f32(0.25)).to_le_bytes());
        }
        row
    }

    #[test]
    fn q8_0_x_f32_matches_scalar_on_random_data() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        for blocks in [1usize, 3, 32] {
            let row = random_q8_0_row(&mut rng, blocks);
            let x: Vec<f32> = (0..blocks * 32).map(|_| rng.next_f32(2.0)).collect();
            let kernel = dot_q8_0_x_f32(&row, &x);
            let reference = scalar::dot_q8_0_x_f32(&row, &x);
            let denom = reference.abs().max(1.0);
            assert!(
                (kernel - reference).abs() / denom < 2e-4,
                "blocks {blocks}: kernel {kernel} vs scalar {reference}"
            );
        }
    }

    #[test]
    fn q8_0_q8_0_int_dot_matches_scalar_exactly_ish() {
        let mut rng = Rng(0xfeed_beef_cafe_1234);
        for blocks in [1usize, 8, 40] {
            let row = random_q8_0_row(&mut rng, blocks);
            let x: Vec<f32> = (0..blocks * 32).map(|_| rng.next_f32(1.5)).collect();
            let (x_qs, x_d) = quantize_for_dot(&x);
            let kernel = dot_q8_0_q8_0(&row, &x_qs, &x_d);
            let reference = scalar::dot_q8_0_q8_0(&row, &x_qs, &x_d);
            let denom = reference.abs().max(1.0);
            assert!(
                (kernel - reference).abs() / denom < 1e-5,
                "blocks {blocks}: kernel {kernel} vs scalar {reference}"
            );
        }
    }

    #[test]
    fn f16_x_f32_matches_scalar_on_random_data() {
        let mut rng = Rng(0x0bad_c0de_dead_beef);
        for elems in [32usize, 34, 128, 1024 + 16] {
            let row = random_f16_row(&mut rng, elems);
            let x: Vec<f32> = (0..elems).map(|_| rng.next_f32(3.0)).collect();
            let kernel = dot_f16_x_f32(&row, &x);
            let reference = scalar::dot_f16_x_f32(&row, &x);
            let denom = reference.abs().max(1.0);
            assert!(
                (kernel - reference).abs() / denom < 2e-4,
                "elems {elems}: kernel {kernel} vs scalar {reference} (tail handling)"
            );
        }
    }

    #[test]
    fn zero_rows_and_zero_x_are_zero() {
        let row = vec![0u8; 34];
        let x = vec![0.0f32; 32];
        assert_eq!(dot_q8_0_x_f32(&row, &x), 0.0);
        let (qs, ds) = quantize_for_dot(&x);
        assert_eq!(dot_q8_0_q8_0(&row, &qs, &ds), 0.0);
        let f16_row = vec![0u8; 64];
        assert_eq!(dot_f16_x_f32(&f16_row, &x), 0.0);
    }
}
