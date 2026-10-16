//! Tensor element formats: decoding for the type set TuckCore executes.
//!
//! Supported in Stage 0: F32, F16, BF16, Q8_0, Q4_0, Q4_1. Everything else
//! is refused with `UNSUPPORTED_QUANTIZATION` listing what is supported —
//! never substituted. Block sizes come from the pinned GGML table in
//! `tuckmill-manifest`.

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ggml;

/// Conversion from IEEE 754 binary16 bits.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let value = match (exp, mant) {
        (0, 0) => 0.0,
        (0, m) => (m as f32) * 2.0f32.powi(-24),
        (31, 0) => f32::INFINITY,
        (31, _) => f32::NAN,
        (e, m) => (1.0 + m as f32 / 1024.0) * 2.0f32.powi(e as i32 - 15),
    };
    if sign == 1 {
        -value
    } else {
        value
    }
}

/// Conversion from bfloat16 bits (low 16 bits of the f32 pattern).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

/// Types Stage 0 executes.
pub fn is_supported(type_id: u32) -> bool {
    matches!(type_id, 0 | 1 | 2 | 3 | 8 | 30)
}

pub fn supported_type_names() -> &'static str {
    "F32, F16, BF16, Q8_0, Q4_0, Q4_1"
}

/// Decode `bytes` (a whole number of blocks) of `type_id` into `out`
/// (extended, not cleared). Returns the element count written. The caller
/// guarantees `bytes` is block aligned, which artifact inspection already
/// validated.
pub fn dequant_into(type_id: u32, bytes: &[u8], out: &mut Vec<f32>) -> Result<usize> {
    let info = ggml::lookup(type_id).ok_or_else(|| unsupported(type_id))?;
    if !is_supported(type_id) {
        return Err(unsupported(type_id));
    }
    let blocks = bytes.len() as u64 / info.block_bytes;
    let elems = (blocks * info.block_elems) as usize;
    out.reserve(elems);
    match type_id {
        0 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        1 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        30 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        8 => {
            // Q8_0: [f16 d][i8 qs; 32]
            for block in bytes.chunks_exact(34) {
                let d = f16_to_f32(read_u16(block, 0));
                for q in &block[2..34] {
                    out.push(d * (*q as i8) as f32);
                }
            }
        }
        2 => {
            // Q4_0: [f16 d][u8 qs; 16]; low nibble = element j,
            // high nibble = element j+16 (llama.cpp layout).
            for block in bytes.chunks_exact(18) {
                let d = f16_to_f32(read_u16(block, 0));
                let qs = &block[2..18];
                for &b in qs {
                    out.push(d * ((b & 0x0f) as i32 - 8) as f32);
                }
                for &b in qs {
                    out.push(d * ((b >> 4) as i32 - 8) as f32);
                }
            }
        }
        3 => {
            // Q4_1: [f16 d][f16 m][u8 qs; 16]; x = d*q + m.
            for block in bytes.chunks_exact(20) {
                let d = f16_to_f32(read_u16(block, 0));
                let m = f16_to_f32(read_u16(block, 2));
                let qs = &block[4..20];
                for &b in qs {
                    out.push(d * (b & 0x0f) as f32 + m);
                }
                for &b in qs {
                    out.push(d * (b >> 4) as f32 + m);
                }
            }
        }
        _ => unreachable!("guarded by is_supported"),
    }
    Ok(elems)
}

/// IEEE binary16 encoding with round-to-nearest-even (mirrors ggml's
/// `GGML_CPU_FP32_TO_FP16` conversion used for activation scales and the
/// llama.cpp-CPU numerical profile).
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp >= 0x1f {
        // Overflow to inf (f16 range is exceeded): saturate like hardware.
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflow to zero
        }
        let m = mant | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half = (m >> shift) & 0x3ff;
        // Round-to-nearest-even on the dropped bits.
        let rem = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let round_up = rem > halfway || (rem == halfway && (half & 1) == 1);
        let half = half + round_up as u32;
        return sign | half as u16;
    }
    // Normal range: round-to-nearest-even on the 13 dropped mantissa bits.
    let rounded = mant + 0x0fff + ((mant >> 13) & 1);
    if rounded & 0x80_0000 != 0 {
        let exp = exp + 1;
        if exp >= 0x1f {
            return sign | 0x7c00;
        }
        return sign | ((exp as u16) << 10);
    }
    sign | ((exp as u16) << 10) | ((rounded >> 13) as u16)
}

/// Quantize a row of activations to Q8_0 exactly as ggml's
/// `quantize_row_q8_0` does: per 32-element block, `d = amax/127`, the
/// stored scale is the f16 rounding of `d`, and values are
/// round-to-nearest-even of `x * (1/d)`.
pub fn quantize_row_q8_0(x: &[f32], blocks_qs: &mut Vec<[i8; 32]>, blocks_d: &mut Vec<f32>) {
    blocks_qs.clear();
    blocks_d.clear();
    for chunk in x.chunks(32) {
        let mut amax = 0.0f32;
        for v in chunk {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = f16_to_f32(f32_to_f16_bits(d));
        let mut qs = [0i8; 32];
        for (i, v) in chunk.iter().enumerate() {
            let q = (v * id).round_ties_even().clamp(-127.0, 127.0) as i8;
            qs[i] = q;
        }
        // Blocks shorter than 32 cannot occur for supported geometries (rows
        // are validated block-aligned); if they did, callers would see the
        // zero tail rather than corrupted data.
        blocks_qs.push(qs);
        blocks_d.push(d_f16);
    }
}

fn unsupported(type_id: u32) -> TuckmillError {
    TuckmillError::new(
        ErrorCode::UnsupportedQuantization,
        format!(
            "tuckcore does not execute tensor type {} in this build (supported: {})",
            ggml::type_name(type_id),
            supported_type_names()
        ),
    )
    .with_action(
        "Choose an artifact within the supported type set, or wait for its kernel work package (K01/K04).",
    )
}

/// Row geometry of a 2D weight: `dims[0]` elements per row, `dims[1]` rows
/// (GGUF stores `dims[0]` fastest). Quantized rows must be block aligned to
/// be splittable; otherwise the whole tensor is one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowGeometry {
    pub elems_per_row: u64,
    pub bytes_per_row: u64,
    pub rows: u64,
    /// Rows per `block_bytes` chunk (>= 1).
    pub rows_per_chunk: u64,
}

pub fn row_geometry(type_id: u32, dims: &[u64], block_bytes: u64) -> Option<RowGeometry> {
    if dims.len() != 2 {
        return None;
    }
    let info = ggml::lookup(type_id)?;
    let (elems_per_row, rows) = (dims[0], dims[1]);
    if elems_per_row == 0 || elems_per_row % info.block_elems != 0 {
        return None;
    }
    let blocks_per_row = elems_per_row / info.block_elems;
    let bytes_per_row = blocks_per_row.checked_mul(info.block_bytes)?;
    let rows_per_chunk = (block_bytes / bytes_per_row.max(1)).max(1);
    Some(RowGeometry {
        elems_per_row,
        bytes_per_row,
        rows,
        rows_per_chunk,
    })
}

/// Row geometry for 1D tensors (norm weights): one row.
pub fn vector_geometry(type_id: u32, elems: u64, block_bytes: u64) -> Option<RowGeometry> {
    row_geometry(type_id, &[elems, 1], block_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C01).is_nan());
        // Smallest positive subnormal: 2^-24.
        assert!((f16_to_f32(0x0001) - 2.0f32.powi(-24)).abs() < 1e-12);
    }

    #[test]
    fn bf16_known_values() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
    }

    #[test]
    fn f32_and_f16_dequant() {
        let mut out = Vec::new();
        let bytes: Vec<u8> = [1.0f32, -0.5, 2.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let n = dequant_into(0, &bytes, &mut out).unwrap();
        assert_eq!(n, 3);
        assert_eq!(out, vec![1.0, -0.5, 2.25]);

        let mut out = Vec::new();
        let bytes: Vec<u8> = [0x3C00u16, 0xC000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let n = dequant_into(1, &bytes, &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out, vec![1.0, -2.0]);
    }

    #[test]
    fn q8_0_layout_and_scaling() {
        // d = 0.5 (f16 0x3800), qs = [-1, 2, 0, ...].
        let mut block = Vec::new();
        block.extend_from_slice(&0x3800u16.to_le_bytes());
        block.extend_from_slice(&[-1i8, 2].iter().map(|v| *v as u8).collect::<Vec<_>>());
        block.extend_from_slice(&[0u8; 30]);
        let mut out = Vec::new();
        let n = dequant_into(8, &block, &mut out).unwrap();
        assert_eq!(n, 32);
        assert_eq!(out[0], -0.5);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn q4_0_nibble_order_matches_llama_cpp() {
        // d = 1.0 (0x3C00); qs[0] = 0x21 -> element0 = 1-8 = -7, element16 =
        // 2-8 = -6. All other nibbles 0x88 -> 0.
        let mut block = Vec::new();
        block.extend_from_slice(&0x3C00u16.to_le_bytes());
        let mut qs = [0x88u8; 16];
        qs[0] = 0x21;
        block.extend_from_slice(&qs);
        let mut out = Vec::new();
        let n = dequant_into(2, &block, &mut out).unwrap();
        assert_eq!(n, 32);
        assert_eq!(out[0], -7.0);
        assert_eq!(out[16], -6.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[17], 0.0);
    }

    #[test]
    fn q4_1_affine_layout() {
        // d = 1.0, m = 2.0; low nibble 3 -> 3+2 = 5; high nibble 4 -> 6.
        let mut block = Vec::new();
        block.extend_from_slice(&0x3C00u16.to_le_bytes());
        block.extend_from_slice(&0x4000u16.to_le_bytes());
        let mut qs = [0u8; 16];
        qs[0] = 0x43;
        block.extend_from_slice(&qs);
        let mut out = Vec::new();
        let n = dequant_into(3, &block, &mut out).unwrap();
        assert_eq!(n, 32);
        assert_eq!(out[0], 5.0);
        assert_eq!(out[16], 6.0);
    }

    #[test]
    fn unsupported_types_are_refused_with_the_supported_list() {
        let err = dequant_into(12, &[0u8; 144], &mut Vec::new()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::UnsupportedQuantization);
        assert!(err.message().contains("F16"), "{}", err.message());
    }

    #[test]
    fn f32_to_f16_matches_known_values() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(100000.0), 0x7c00, "saturates to inf");
        assert_eq!(f32_to_f16_bits(5.96e-8), 0x0001, "smallest subnormal");
        // Round-to-nearest-even: 1 + 2^-11 exactly halfway -> even mantissa.
        let halfway = 1.0 + 2.0f32.powi(-11);
        assert_eq!(f32_to_f16_bits(halfway), 0x3c00);
        let above = 1.0 + 2.0f32.powi(-11) + 2.0f32.powi(-13);
        assert_eq!(f32_to_f16_bits(above), 0x3c01);
    }

    #[test]
    fn q8_0_activation_quantization_matches_ggml_rules() {
        let mut x = vec![0.0f32; 32];
        x[0] = 127.0;
        x[1] = -63.5;
        x[2] = 1.0;
        let mut qs = Vec::new();
        let mut ds = Vec::new();
        quantize_row_q8_0(&x, &mut qs, &mut ds);
        assert_eq!(qs.len(), 1);
        assert_eq!(ds, vec![1.0f32], "d = amax/127 = 1.0");
        assert_eq!(qs[0][0], 127);
        assert_eq!(qs[0][1], -64, "(-63.5).round_ties_even() = -64");
        assert_eq!(qs[0][2], 1);
        assert_eq!(qs[0][3], 0);
        // Two blocks quantize independently.
        let x2: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        quantize_row_q8_0(&x2, &mut qs, &mut ds);
        assert_eq!(qs.len(), 2);
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn row_geometry_requires_block_alignment() {
        // F16 128-elem rows are always splittable.
        let g = row_geometry(1, &[128, 512], 4 << 20).unwrap();
        assert_eq!(g.elems_per_row, 128);
        assert_eq!(g.bytes_per_row, 256);
        assert_eq!(g.rows, 512);
        assert_eq!(g.rows_per_chunk, (4 << 20) / 256);

        // Q4_0 rows of 100 elements are not block aligned (100 % 32 != 0).
        assert!(row_geometry(2, &[100, 8], 4 << 20).is_none());

        // Q4_0 rows of 256 elements: 8 blocks x 18 bytes = 144 bytes/row.
        let g = row_geometry(2, &[256, 16], 1024).unwrap();
        assert_eq!(g.bytes_per_row, 144);
        assert_eq!(g.rows_per_chunk, 7);
    }
}
