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
