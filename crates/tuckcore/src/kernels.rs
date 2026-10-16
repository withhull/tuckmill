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
//! **Boundary contract:** only the checked entry points of this module are
//! public API. They validate packed-row alignment, activation length,
//! quantization-block count, and scale-array length once per row before any
//! load; the raw implementations (scalar and NEON) are private and must
//! never be called with unvalidated shapes. Malformed shapes are rejected
//! with typed errors — never executed.
//!
//! Numerical contract: the integer dot for Q8_0 × Q8_0 is exact (int32);
//! f32 accumulation order differs between the scalar and NEON
//! implementations by design and is covered by the registered verification
//! tolerance. Q8_0 × f32 converts int8 lanes to f32 (exact for |q| <= 127)
//! and multiplies by the f16 scale per block, mirroring the scalar path's
//! arithmetic.

use serde::{Deserialize, Serialize};

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};

use crate::tensor::{f16_to_f32, f32_to_f16_bits, quantize_row_q8_0};

/// Kernel implementation selection (a load-time option, recorded in stats).
/// `Auto` resolves to NEON where the target architecture has it and to the
/// scalar reference everywhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KernelSelection {
    Auto,
    Scalar,
    Neon,
}

impl KernelSelection {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "scalar" => Some(Self::Scalar),
            "neon" => Some(Self::Neon),
            _ => None,
        }
    }

    /// Resolve to a concrete implementation. `Neon` on a non-aarch64 target
    /// is refused — never silently downgraded.
    pub fn resolve(self) -> Result<ResolvedKernels> {
        match self {
            Self::Scalar => Ok(ResolvedKernels::Scalar),
            Self::Auto => Ok(ResolvedKernels::default_for_target()),
            Self::Neon => {
                if cfg!(target_arch = "aarch64") {
                    Ok(ResolvedKernels::Neon)
                } else {
                    Err(TuckmillError::new(
                        ErrorCode::InvalidRequest,
                        "the NEON kernel implementation requires an aarch64 target",
                    )
                    .with_action("Select --kernels auto or --kernels scalar on this machine."))
                }
            }
        }
    }
}

/// A concrete kernel implementation actually used at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedKernels {
    Scalar,
    Neon,
}

impl ResolvedKernels {
    pub fn default_for_target() -> Self {
        if cfg!(target_arch = "aarch64") {
            Self::Neon
        } else {
            Self::Scalar
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
        }
    }
}

// ---------------------------------------------------------------------------
// Shape validation (the checked boundary)
// ---------------------------------------------------------------------------

fn packed_row_error(row: &[u8], block_bytes: usize, what: &str) -> TuckmillError {
    TuckmillError::new(
        ErrorCode::InvalidArtifact,
        format!(
            "packed {what} row length {} is not a positive multiple of {block_bytes} bytes",
            row.len()
        ),
    )
    .with_action("Re-import the artifact; block geometry is validated at inspection.")
}

fn activation_error(expected: usize, got: usize, what: &str) -> TuckmillError {
    TuckmillError::new(
        ErrorCode::Internal,
        format!("{what} dot expects {expected} activation elements, got {got}"),
    )
}

/// Validate a packed Q8_0 row and its activation slice. Returns the block
/// count.
fn validate_q8_0_row(row: &[u8], x_len: usize) -> Result<usize> {
    if row.is_empty() || !row.len().is_multiple_of(34) {
        return Err(packed_row_error(row, 34, "Q8_0"));
    }
    let blocks = row.len() / 34;
    let expected = blocks * 32;
    if x_len != expected {
        return Err(activation_error(expected, x_len, "Q8_0 x f32"));
    }
    Ok(blocks)
}

/// Validate a packed Q8_0 row, its quantized activation blocks, and the
/// per-block scale array.
fn validate_q8_0_q8_0(row: &[u8], x_qs_len: usize, x_d_len: usize) -> Result<usize> {
    if row.is_empty() || !row.len().is_multiple_of(34) {
        return Err(packed_row_error(row, 34, "Q8_0"));
    }
    let blocks = row.len() / 34;
    if x_qs_len != blocks {
        return Err(activation_error(blocks, x_qs_len, "Q8_0 x q8_0 block"));
    }
    if x_d_len != blocks {
        return Err(activation_error(blocks, x_d_len, "Q8_0 x q8_0 scale"));
    }
    Ok(blocks)
}

/// Validate an F16 row and its activation slice. Returns the element count.
fn validate_f16_row(row: &[u8], x_len: usize) -> Result<usize> {
    if row.is_empty() || !row.len().is_multiple_of(2) {
        return Err(packed_row_error(row, 2, "F16"));
    }
    let elems = row.len() / 2;
    if x_len != elems {
        return Err(activation_error(elems, x_len, "F16 x f32"));
    }
    Ok(elems)
}

// ---------------------------------------------------------------------------
// Raw implementations (private; call only through the checked entry points)
// ---------------------------------------------------------------------------

mod scalar {
    use super::*;

    pub fn dot_q8_0_x_f32(row: &[u8], x: &[f32]) -> f32 {
        let (wblocks, w_rest) = row.as_chunks::<34>();
        let (xblocks, x_rest) = x.as_chunks::<32>();
        debug_assert!(
            w_rest.is_empty() && x_rest.is_empty(),
            "validated at the checked boundary"
        );
        let mut acc = 0.0f32;
        for (block, xb) in wblocks.iter().zip(xblocks.iter()) {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let mut block_sum = 0.0f32;
            for q in 0..32 {
                let w = (block[2 + q] as i8) as f32;
                block_sum += w * xb[q];
            }
            acc += d * block_sum;
        }
        acc
    }

    pub fn dot_q8_0_q8_0(row: &[u8], x_qs: &[[i8; 32]], x_d: &[f32]) -> f32 {
        let (wblocks, w_rest) = row.as_chunks::<34>();
        debug_assert!(w_rest.is_empty(), "validated at the checked boundary");
        debug_assert_eq!(wblocks.len(), x_qs.len());
        debug_assert_eq!(wblocks.len(), x_d.len());
        let mut acc = 0.0f32;
        for (blk, (block, qx)) in wblocks.iter().zip(x_qs.iter()).enumerate() {
            let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let mut sumi: i32 = 0;
            for q in 0..32 {
                sumi += ((block[2 + q] as i8) as i32) * (qx[q] as i32);
            }
            acc += (d_w * x_d[blk]) * (sumi as f32);
        }
        acc
    }

    pub fn dot_f16_x_f32(row: &[u8], x: &[f32]) -> f32 {
        let (pairs, p_rest) = row.as_chunks::<2>();
        debug_assert!(p_rest.is_empty(), "validated at the checked boundary");
        let mut acc = 0.0f32;
        for (i, pair) in pairs.iter().enumerate() {
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
        let (wblocks, w_rest) = row.as_chunks::<34>();
        let (xblocks, x_rest) = x.as_chunks::<32>();
        debug_assert!(
            w_rest.is_empty() && x_rest.is_empty(),
            "validated at the checked boundary"
        );
        let mut acc = 0.0f32;
        for (block, xb) in wblocks.iter().zip(xblocks.iter()) {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qs = &block[2..];
            // SAFETY: qs has exactly 32 elements (block is [u8; 34]) and xb
            // has exactly 32 elements ([f32; 32] from as_chunks); all loads
            // are unaligned-capable NEON loads fully in bounds.
            unsafe {
                let q_lo = vld1q_s8(qs.as_ptr() as *const i8);
                let q_hi = vld1q_s8(qs.as_ptr().add(16) as *const i8);
                let x_lo = vld1q_f32(xb.as_ptr());
                let x_hi = vld1q_f32(xb.as_ptr().add(4));
                let x_lo2 = vld1q_f32(xb.as_ptr().add(8));
                let x_hi2 = vld1q_f32(xb.as_ptr().add(12));
                let x_b1 = vld1q_f32(xb.as_ptr().add(16));
                let x_b2 = vld1q_f32(xb.as_ptr().add(20));
                let x_b3 = vld1q_f32(xb.as_ptr().add(24));
                let x_b4 = vld1q_f32(xb.as_ptr().add(28));

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
        }
        acc
    }

    /// Q8_0 weight row × q8_0-quantized activation (exact int32 block dot).
    pub fn dot_q8_0_q8_0(row: &[u8], x_qs: &[[i8; 32]], x_d: &[f32]) -> f32 {
        let (wblocks, w_rest) = row.as_chunks::<34>();
        debug_assert!(w_rest.is_empty(), "validated at the checked boundary");
        debug_assert_eq!(wblocks.len(), x_qs.len());
        let mut acc = 0.0f32;
        for (blk, (block, qx)) in wblocks.iter().zip(x_qs.iter()).enumerate() {
            let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qs = &block[2..];
            // SAFETY: qs and qx both have exactly 32 elements.
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
        let (pairs, p_rest) = row.as_chunks::<2>();
        debug_assert!(p_rest.is_empty(), "validated at the checked boundary");
        let mut acc = 0.0f32;
        let mut xi = 0usize;
        let mut rest = pairs;
        // 16 weights per iteration (four 4-wide f32 groups).
        while rest.len() >= 16 {
            let chunk = &rest[..16];
            // SAFETY: chunk is a contiguous [[u8; 2]; 16] — 32 bytes read as
            // 16 u16 halves; x has at least 16 elements remaining (validated
            // length, xi + 16 <= pairs.len().
            unsafe {
                let h0 = vld1_u16(chunk.as_ptr() as *const u16);
                let h1 = vld1_u16((chunk.as_ptr() as *const u16).add(4));
                let h2 = vld1_u16((chunk.as_ptr() as *const u16).add(8));
                let h3 = vld1_u16((chunk.as_ptr() as *const u16).add(12));
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
            rest = &rest[16..];
            xi += 16;
        }
        // Tail (rows are even-sized for all supported geometries; F16 has no
        // block size, so a scalar tail is correct and cheap).
        for (i, pair) in rest.iter().enumerate() {
            let w = f16_to_f32(u16::from_le_bytes([pair[0], pair[1]]));
            acc += w * x[xi + i];
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// Checked entry points
// ---------------------------------------------------------------------------

/// Fused Q8_0 weight row × f32 activations, using the resolved
/// implementation. Validates the packed row and activation width first.
pub fn dot_q8_0_x_f32(impl_: ResolvedKernels, row: &[u8], x: &[f32]) -> Result<f32> {
    validate_q8_0_row(row, x.len())?;
    Ok(match impl_ {
        ResolvedKernels::Scalar => scalar::dot_q8_0_x_f32(row, x),
        #[cfg(target_arch = "aarch64")]
        ResolvedKernels::Neon => neon::dot_q8_0_x_f32(row, x),
        #[cfg(not(target_arch = "aarch64"))]
        ResolvedKernels::Neon => unreachable!("resolution refuses NEON on this target"),
    })
}

/// Q8_0 × q8_0 (both int8): exact int32 block dot with f16 scales.
pub fn dot_q8_0_q8_0(
    impl_: ResolvedKernels,
    row: &[u8],
    x_qs: &[[i8; 32]],
    x_d: &[f32],
) -> Result<f32> {
    validate_q8_0_q8_0(row, x_qs.len(), x_d.len())?;
    Ok(match impl_ {
        ResolvedKernels::Scalar => scalar::dot_q8_0_q8_0(row, x_qs, x_d),
        #[cfg(target_arch = "aarch64")]
        ResolvedKernels::Neon => neon::dot_q8_0_q8_0(row, x_qs, x_d),
        #[cfg(not(target_arch = "aarch64"))]
        ResolvedKernels::Neon => unreachable!("resolution refuses NEON on this target"),
    })
}

/// Fused F16 weight row × f32 activations.
pub fn dot_f16_x_f32(impl_: ResolvedKernels, row: &[u8], x: &[f32]) -> Result<f32> {
    validate_f16_row(row, x.len())?;
    Ok(match impl_ {
        ResolvedKernels::Scalar => scalar::dot_f16_x_f32(row, x),
        #[cfg(target_arch = "aarch64")]
        ResolvedKernels::Neon => neon::dot_f16_x_f32(row, x),
        #[cfg(not(target_arch = "aarch64"))]
        ResolvedKernels::Neon => unreachable!("resolution refuses NEON on this target"),
    })
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
    fn selection_resolves_and_parses() {
        assert_eq!(KernelSelection::parse("auto"), Some(KernelSelection::Auto));
        assert_eq!(
            KernelSelection::parse("scalar"),
            Some(KernelSelection::Scalar)
        );
        assert_eq!(KernelSelection::parse("neon"), Some(KernelSelection::Neon));
        assert_eq!(KernelSelection::parse("simd"), None);
        assert_eq!(
            KernelSelection::Scalar.resolve().unwrap(),
            ResolvedKernels::Scalar
        );
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(
                KernelSelection::Auto.resolve().unwrap(),
                ResolvedKernels::Neon
            );
            assert_eq!(
                KernelSelection::Neon.resolve().unwrap(),
                ResolvedKernels::Neon
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            assert_eq!(
                KernelSelection::Auto.resolve().unwrap(),
                ResolvedKernels::Scalar
            );
            assert!(KernelSelection::Neon.resolve().is_err());
        }
    }

    #[test]
    fn q8_0_x_f32_matches_scalar_on_random_data() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        for blocks in [1usize, 3, 32] {
            let row = random_q8_0_row(&mut rng, blocks);
            let x: Vec<f32> = (0..blocks * 32).map(|_| rng.next_f32(2.0)).collect();
            let kernel = dot_q8_0_x_f32(ResolvedKernels::default_for_target(), &row, &x).unwrap();
            let reference = scalar::dot_q8_0_x_f32(&row, &x);
            let denom = reference.abs().max(1.0);
            assert!(
                (kernel - reference).abs() / denom < 2e-4,
                "blocks {blocks}: kernel {kernel} vs scalar {reference}"
            );
        }
    }

    #[test]
    fn forced_scalar_selection_is_the_scalar_reference() {
        let mut rng = Rng(0x5511_2233_4455_6677);
        let row = random_q8_0_row(&mut rng, 4);
        let x: Vec<f32> = (0..128).map(|_| rng.next_f32(1.0)).collect();
        let forced = dot_q8_0_x_f32(ResolvedKernels::Scalar, &row, &x).unwrap();
        assert_eq!(forced, scalar::dot_q8_0_x_f32(&row, &x));
    }

    #[test]
    fn q8_0_q8_0_int_dot_matches_scalar_exactly_ish() {
        let mut rng = Rng(0xfeed_beef_cafe_1234);
        for blocks in [1usize, 8, 40] {
            let row = random_q8_0_row(&mut rng, blocks);
            let x: Vec<f32> = (0..blocks * 32).map(|_| rng.next_f32(1.5)).collect();
            let (x_qs, x_d) = quantize_for_dot(&x);
            let kernel =
                dot_q8_0_q8_0(ResolvedKernels::default_for_target(), &row, &x_qs, &x_d).unwrap();
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
            let kernel = dot_f16_x_f32(ResolvedKernels::default_for_target(), &row, &x).unwrap();
            let reference = scalar::dot_f16_x_f32(&row, &x);
            let denom = reference.abs().max(1.0);
            assert!(
                (kernel - reference).abs() / denom < 2e-4,
                "elems {elems}: kernel {kernel} vs scalar {reference} (tail handling)"
            );
        }
    }

    #[test]
    fn malformed_shapes_are_rejected_at_the_boundary() {
        let impl_ = ResolvedKernels::default_for_target();
        // The reviewer's exact case: one valid 34-byte block + empty x.
        let row34 = vec![0u8; 34];
        let err = dot_q8_0_x_f32(impl_, &row34, &[]).unwrap_err();
        assert_eq!(err.code(), ErrorCode::Internal);
        // x one short.
        assert!(dot_q8_0_x_f32(impl_, &row34, &[0.0f32; 31]).is_err());
        // Two blocks but only one block of activations.
        let row68 = vec![0u8; 68];
        assert!(dot_q8_0_x_f32(impl_, &row68, &[0.0f32; 32]).is_err());
        // Row not block aligned (33 bytes) with otherwise plausible x.
        assert!(dot_q8_0_x_f32(impl_, &[0u8; 33], &[0.0f32; 32]).is_err());
        // Empty row.
        assert!(dot_q8_0_x_f32(impl_, &[], &[]).is_err());

        // q8_0 × q8_0: block-count and scale-length mismatches.
        let (x_qs, x_d) = quantize_for_dot(&vec![0.0f32; 64]);
        assert_eq!(x_qs.len(), 2);
        assert!(dot_q8_0_q8_0(impl_, &row68, &x_qs[..1], &x_d).is_err());
        assert!(dot_q8_0_q8_0(impl_, &row68, &x_qs, &x_d[..1]).is_err());
        assert!(dot_q8_0_q8_0(impl_, &row34, &x_qs, &x_d).is_err());

        // F16: odd byte length and short/over-long activations.
        assert!(dot_f16_x_f32(impl_, &[0u8; 5], &[0.0f32; 2]).is_err());
        assert!(dot_f16_x_f32(impl_, &[0u8; 4], &[0.0f32; 1]).is_err());
        assert!(dot_f16_x_f32(impl_, &[0u8; 4], &[0.0f32; 3]).is_err());
    }

    #[test]
    fn zero_rows_and_zero_x_are_zero() {
        let impl_ = ResolvedKernels::default_for_target();
        let row = vec![0u8; 34];
        let x = vec![0.0f32; 32];
        assert_eq!(dot_q8_0_x_f32(impl_, &row, &x).unwrap(), 0.0);
        let (qs, ds) = quantize_for_dot(&x);
        assert_eq!(dot_q8_0_q8_0(impl_, &row, &qs, &ds).unwrap(), 0.0);
        let f16_row = vec![0u8; 64];
        assert_eq!(dot_f16_x_f32(impl_, &f16_row, &x).unwrap(), 0.0);
    }
}
