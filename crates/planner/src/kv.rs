//! KV cache estimation (PRD §14).
//!
//! For grouped-query attention the illustrative estimate is
//! `2 x layers x seq x kv_heads x head_dim x bytes_per_element`.
//! This module computes the exact analog from the resolved geometry:
//! `layers x seq x kv_heads x (key_length + value_length) x bytes`.

use tuckmill_manifest::adapter::KvGeometry;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvEstimate {
    pub total_bytes: u64,
    /// Bytes per sequence position; used by streaming cost models.
    pub per_token_bytes: u64,
}

/// Estimate KV payload for `total_tokens` positions. Rejects overflow rather
/// than wrapping.
pub fn estimate_kv_bytes(
    kv: &KvGeometry,
    total_tokens: u64,
    bytes_per_element: u32,
) -> Option<KvEstimate> {
    let per_token = kv
        .layers
        .checked_mul(kv.kv_head_count)?
        .checked_mul(kv.key_length.checked_add(kv.value_length)?)?
        .checked_mul(bytes_per_element as u64)?;
    let total = per_token.checked_mul(total_tokens)?;
    Some(KvEstimate {
        total_bytes: total,
        per_token_bytes: per_token,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_prd_worked_example() {
        // 80 layers, 8 KV heads, head dim 128, 16-bit, 8192 tokens = 2.5GiB.
        let kv = KvGeometry {
            layers: 80,
            attention_layers: 80,
            state_bytes: 0,
            head_count: 64,
            kv_head_count: 8,
            head_dim: 128,
            key_length: 128,
            value_length: 128,
        };
        let est = estimate_kv_bytes(&kv, 8192, 2).unwrap();
        let gib = est.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!((gib - 2.5).abs() < 0.01, "expected ~2.5GiB, got {gib}");
    }

    #[test]
    fn scales_linearly_with_sequence() {
        let kv = KvGeometry {
            layers: 28,
            attention_layers: 28,
            state_bytes: 0,
            head_count: 16,
            kv_head_count: 8,
            head_dim: 128,
            key_length: 128,
            value_length: 128,
        };
        let a = estimate_kv_bytes(&kv, 1024, 2).unwrap();
        let b = estimate_kv_bytes(&kv, 2048, 2).unwrap();
        assert_eq!(a.total_bytes * 2, b.total_bytes);
        assert_eq!(a.per_token_bytes, 28 * 8 * 256 * 2);
    }

    #[test]
    fn asymmetric_key_value_lengths_are_supported() {
        let kv = KvGeometry {
            layers: 1,
            attention_layers: 1,
            state_bytes: 0,
            head_count: 1,
            kv_head_count: 1,
            head_dim: 128,
            key_length: 128,
            value_length: 64,
        };
        let est = estimate_kv_bytes(&kv, 10, 2).unwrap();
        assert_eq!(est.per_token_bytes, (128 + 64) * 2);
    }

    #[test]
    fn overflow_is_rejected() {
        let kv = KvGeometry {
            layers: u64::MAX,
            attention_layers: u64::MAX,
            state_bytes: 0,
            head_count: 1,
            kv_head_count: 2,
            head_dim: 1,
            key_length: 1,
            value_length: 1,
        };
        assert!(estimate_kv_bytes(&kv, 2, 2).is_none());
    }
}
