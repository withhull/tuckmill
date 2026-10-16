//! GGML tensor type table.
//!
//! Block sizes are transcribed from a pinned llama.cpp revision:
//! `ggml-org/llama.cpp@ceefd16daa2093730bf3340315fa2b84b286ddbc`
//! (`ggml/include/ggml.h` enum `ggml_type`, `ggml/src/ggml-common.h` struct
//! sizes, verified by compiling `sizeof()` against that header on 2026-10-16).
//! Do not hand-edit sizes: re-derive them from a new pinned revision and note
//! the change.

use crate::error::{ErrorCode, Result, TuckmillError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorTypeInfo {
    pub id: u32,
    pub name: &'static str,
    /// Elements per block (1 for non-quantized types).
    pub block_elems: u64,
    /// Stored bytes per block (element size for non-quantized types).
    pub block_bytes: u64,
}

impl TensorTypeInfo {
    pub fn bits_per_weight(&self) -> f64 {
        (self.block_bytes as f64 * 8.0) / self.block_elems as f64
    }
}

macro_rules! ty {
    ($id:expr, $name:expr, $elems:expr, $bytes:expr) => {
        TensorTypeInfo {
            id: $id,
            name: $name,
            block_elems: $elems,
            block_bytes: $bytes,
        }
    };
}

/// The full GGUF-capable type table for the pinned revision.
pub const TYPES: &[TensorTypeInfo] = &[
    ty!(0, "F32", 1, 4),
    ty!(1, "F16", 1, 2),
    ty!(2, "Q4_0", 32, 18),
    ty!(3, "Q4_1", 32, 20),
    ty!(6, "Q5_0", 32, 22),
    ty!(7, "Q5_1", 32, 24),
    ty!(8, "Q8_0", 32, 34),
    ty!(9, "Q8_1", 32, 36),
    ty!(10, "Q2_K", 256, 84),
    ty!(11, "Q3_K", 256, 110),
    ty!(12, "Q4_K", 256, 144),
    ty!(13, "Q5_K", 256, 176),
    ty!(14, "Q6_K", 256, 210),
    ty!(15, "Q8_K", 256, 292),
    ty!(16, "IQ2_XXS", 256, 66),
    ty!(17, "IQ2_XS", 256, 74),
    ty!(18, "IQ3_XXS", 256, 98),
    ty!(19, "IQ1_S", 256, 50),
    ty!(20, "IQ4_NL", 32, 18),
    ty!(21, "IQ3_S", 256, 110),
    ty!(22, "IQ2_S", 256, 82),
    ty!(23, "IQ4_XS", 256, 136),
    ty!(24, "I8", 1, 1),
    ty!(25, "I16", 1, 2),
    ty!(26, "I32", 1, 4),
    ty!(27, "I64", 1, 8),
    ty!(28, "F64", 1, 8),
    ty!(29, "IQ1_M", 256, 56),
    ty!(30, "BF16", 1, 2),
    ty!(34, "TQ1_0", 256, 54),
    ty!(35, "TQ2_0", 256, 66),
    ty!(39, "MXFP4", 32, 17),
    ty!(40, "NVFP4", 64, 36),
    ty!(41, "Q1_0", 128, 18),
    ty!(42, "Q2_0", 64, 18),
];

pub fn lookup(type_id: u32) -> Option<&'static TensorTypeInfo> {
    TYPES.iter().find(|t| t.id == type_id)
}

pub fn type_name(type_id: u32) -> String {
    lookup(type_id)
        .map(|t| t.name.to_string())
        .unwrap_or_else(|| format!("UNKNOWN({type_id})"))
}

/// Stored byte size of a tensor with `elems` elements of `type_id`.
///
/// Element count must be a whole multiple of the block size; a tensor that is
/// not block-aligned is a malformed artifact, not something to round up.
pub fn tensor_bytes(type_id: u32, elems: u64) -> Result<u64> {
    let info = lookup(type_id).ok_or_else(|| {
        TuckmillError::new(
            ErrorCode::UnsupportedQuantization,
            format!("unknown GGML tensor type id {type_id}"),
        )
    })?;
    if !elems.is_multiple_of(info.block_elems) {
        return Err(TuckmillError::new(
            ErrorCode::InvalidArtifact,
            format!(
                "tensor element count {elems} is not a multiple of {} block size {} for type {}",
                info.name, info.block_elems, info.name
            ),
        ));
    }
    let blocks = elems / info.block_elems;
    blocks.checked_mul(info.block_bytes).ok_or_else(|| {
        TuckmillError::new(
            ErrorCode::InvalidArtifact,
            format!(
                "tensor byte size overflows for {} x {elems} elements",
                info.name
            ),
        )
    })
}

/// Types with at least one element per byte (used for byte-size tables).
pub fn is_quantized(type_id: u32) -> bool {
    lookup(type_id).map(|t| t.block_elems > 1).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_sorted_and_unique() {
        let mut ids: Vec<u32> = TYPES.iter().map(|t| t.id).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "duplicate type ids");
        assert!(
            TYPES.windows(2).all(|w| w[0].id < w[1].id),
            "table must be sorted by id"
        );
    }

    #[test]
    fn known_sizes_match_pinned_revision() {
        assert_eq!(tensor_bytes(0, 10).unwrap(), 40); // F32
        assert_eq!(tensor_bytes(1, 10).unwrap(), 20); // F16
        assert_eq!(tensor_bytes(30, 10).unwrap(), 20); // BF16
        assert_eq!(tensor_bytes(2, 64).unwrap(), 36, "Q4_0: 2 blocks x 18B");
        assert_eq!(tensor_bytes(8, 64).unwrap(), 68, "Q8_0: 2 blocks x 34B");
        assert_eq!(tensor_bytes(12, 256).unwrap(), 144, "Q4_K single block");
        assert_eq!(tensor_bytes(14, 512).unwrap(), 420, "Q6_K two blocks");
        assert_eq!(tensor_bytes(35, 256).unwrap(), 66, "TQ2_0 single block");
        assert_eq!(tensor_bytes(42, 128).unwrap(), 36, "Q2_0 two blocks");
    }

    #[test]
    fn misaligned_tensor_is_rejected() {
        let err = tensor_bytes(2, 33).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArtifact);
        let err = tensor_bytes(1, 3).unwrap();
        assert_eq!(err, 6, "non-quantized types accept any count");
    }

    #[test]
    fn unknown_type_is_unsupported_quantization() {
        let err = tensor_bytes(99, 32).unwrap_err();
        assert_eq!(err.code(), ErrorCode::UnsupportedQuantization);
        assert_eq!(type_name(99), "UNKNOWN(99)");
    }

    #[test]
    fn bits_per_weight_sanity() {
        assert_eq!(lookup(1).unwrap().bits_per_weight(), 16.0);
        let q4 = lookup(2).unwrap().bits_per_weight();
        assert!(
            (q4 - 4.5).abs() < 1e-9,
            "Q4_0 uses 4.5 bits/weight, got {q4}"
        );
    }
}
