//! Packed attention KV storage for the Qwen3.5 reference executor
//! (Q35-1D state contract, ADR-008).
//!
//! The declared KV precision *is* the contract: f16 by default (packed,
//! priced at 2 B/element by the adapter geometry and the request), f32
//! selectable (priced at 4). A cache allocates exactly
//! `attention_layers x kv_heads x capacity x head_dim x element_bytes`
//! per side (K and V); tests prove allocation equality against
//! `Qwen35StateLayout::bytes_for` and against the pinned artifact's
//! published plan numbers (25,165,824 B of attention KV at ctx 2048).
//!
//! Reads decode into caller-provided scratch; nothing here allocates
//! per-read except the exact scratch the caller owns.

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};

use crate::tensor::{f16_to_f32, f32_to_f16_bits};

/// Declared KV element precision (the request's declared profile).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    /// Packed half precision (default; 2 B per element).
    F16,
    /// Full f32 (selectable; 4 B per element).
    F32,
}

impl KvPrecision {
    pub fn element_bytes(self) -> u32 {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

/// One packed plane (`kv_heads x capacity x head_dim` elements), either
/// half-precision bits or f32, in layer order.
#[derive(Debug, Clone)]
enum Plane {
    F16(Vec<u16>),
    F32(Vec<f32>),
}

impl Plane {
    fn zeros(elements: usize, precision: KvPrecision) -> Self {
        match precision {
            KvPrecision::F16 => Plane::F16(vec![0; elements]),
            KvPrecision::F32 => Plane::F32(vec![0.0; elements]),
        }
    }

    fn write(&mut self, index: usize, value: f32) {
        match self {
            Plane::F16(bits) => bits[index] = f32_to_f16_bits(value),
            Plane::F32(values) => values[index] = value,
        }
    }

    fn read(&self, index: usize) -> f32 {
        match self {
            Plane::F16(bits) => f16_to_f32(bits[index]),
            Plane::F32(values) => values[index],
        }
    }

    fn bytes(&self) -> u64 {
        match self {
            Plane::F16(bits) => (bits.len() * 2) as u64,
            Plane::F32(values) => (values.len() * 4) as u64,
        }
    }
}

/// Packed K/V planes for every attention layer of one sequence.
#[derive(Debug, Clone)]
pub struct Qwen35KvCache {
    precision: KvPrecision,
    kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    keys: Vec<Plane>,
    values: Vec<Plane>,
}

impl Qwen35KvCache {
    /// Allocate packed planes for `attention_layers` attention layers.
    /// The element count is checked before allocation (bounded everything).
    pub fn new(
        attention_layers: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        precision: KvPrecision,
    ) -> Result<Self> {
        let overflow = || {
            TuckmillError::new(
                ErrorCode::InvalidPlan,
                "KV cache element count overflows for this plan",
            )
        };
        let per_layer = kv_heads
            .checked_mul(capacity)
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(overflow)?;
        let mut keys = Vec::with_capacity(attention_layers);
        let mut values = Vec::with_capacity(attention_layers);
        for _ in 0..attention_layers {
            keys.push(Plane::zeros(per_layer, precision));
            values.push(Plane::zeros(per_layer, precision));
        }
        Ok(Self {
            precision,
            kv_heads,
            head_dim,
            capacity,
            keys,
            values,
        })
    }

    pub fn precision(&self) -> KvPrecision {
        self.precision
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn attention_layers(&self) -> usize {
        self.keys.len()
    }

    /// Bytes allocated by this cache — the number the admitted plan
    /// reserved under the KV state class.
    pub fn bytes(&self) -> u64 {
        let k: u64 = self.keys.iter().map(Plane::bytes).sum();
        let v: u64 = self.values.iter().map(Plane::bytes).sum();
        k + v
    }

    /// Store one token's K/V rows (`kv_heads x head_dim` each) for one
    /// attention layer at `token`. Writing at or beyond the admitted
    /// capacity refuses: the plan promised `capacity` positions.
    pub fn write(
        &mut self,
        layer: usize,
        token: usize,
        k_rows: &[f32],
        v_rows: &[f32],
    ) -> Result<()> {
        let (stride, plane_len) = self.shape()?;
        if token >= self.capacity {
            return Err(TuckmillError::new(
                ErrorCode::InvalidPlan,
                format!(
                    "KV write at token {token} exceeds the {}-position plan",
                    self.capacity
                ),
            )
            .with_action(
                "Ask for a larger context or a shorter batch; the admitted plan bounds the KV state.",
            ));
        }
        if k_rows.len() != stride || v_rows.len() != stride {
            return Err(shape_error(stride, k_rows.len(), v_rows.len()));
        }
        let base = token * stride;
        if base + stride > plane_len {
            return Err(shape_error(stride, k_rows.len(), v_rows.len()));
        }
        for (offset, (k, v)) in k_rows.iter().zip(v_rows.iter()).enumerate() {
            self.keys[layer].write(base + offset, *k);
            self.values[layer].write(base + offset, *v);
        }
        Ok(())
    }

    /// Decode one K row (`head_dim` f32 values) into `out`.
    pub fn key_row(&self, layer: usize, head: usize, token: usize, out: &mut [f32]) -> Result<()> {
        self.read_row(&self.keys, layer, head, token, out)
    }

    /// Decode one V row (`head_dim` f32 values) into `out`.
    pub fn value_row(
        &self,
        layer: usize,
        head: usize,
        token: usize,
        out: &mut [f32],
    ) -> Result<()> {
        self.read_row(&self.values, layer, head, token, out)
    }

    fn shape(&self) -> Result<(usize, usize)> {
        let stride = self.kv_heads.checked_mul(self.head_dim).ok_or_else(|| {
            TuckmillError::new(ErrorCode::Internal, "KV stride geometry overflows")
        })?;
        let plane_len = self.capacity.checked_mul(stride).ok_or_else(|| {
            TuckmillError::new(ErrorCode::Internal, "KV plane geometry overflows")
        })?;
        Ok((stride, plane_len))
    }

    fn read_row(
        &self,
        planes: &[Plane],
        layer: usize,
        head: usize,
        token: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let (stride, plane_len) = self.shape()?;
        if layer >= planes.len() || head >= self.kv_heads || token >= self.capacity {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!("KV read out of range (layer {layer}, head {head}, token {token})"),
            ));
        }
        if out.len() != self.head_dim {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!(
                    "KV row read wants {} values, got {}",
                    self.head_dim,
                    out.len()
                ),
            ));
        }
        let base = token * stride + head * self.head_dim;
        debug_assert!(base + self.head_dim <= plane_len);
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = planes[layer].read(base + i);
        }
        Ok(())
    }
}

fn shape_error(stride: usize, k: usize, v: usize) -> TuckmillError {
    TuckmillError::new(
        ErrorCode::Internal,
        format!("KV write wants head rows of {stride} values each, got k={k} v={v}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::{resolve, Qwen35Config, Qwen35StateLayout};
    use crate::qwen35_ops::{gqa_cache_attention, ConvState, GdnState, GqaGeometry};
    use tuckmill_manifest::adapter::AdapterRegistry;
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    fn fixture() -> (Qwen35Config, Qwen35StateLayout) {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen35());
        let mut cursor = std::io::Cursor::new(bytes.clone());
        let insp = tuckmill_manifest::gguf::inspect_reader(
            &mut cursor,
            bytes.len() as u64,
            &tuckmill_manifest::gguf::Limits::default(),
            None,
        )
        .unwrap();
        let registry = AdapterRegistry::builtin().unwrap();
        let (cfg, layout) = resolve(&insp, &registry).unwrap();
        (cfg, layout.state)
    }

    fn fixture_cache(ctx: usize, precision: KvPrecision) -> Qwen35KvCache {
        let (cfg, _) = fixture();
        let layout = cfg.state_layout();
        Qwen35KvCache::new(
            layout.attention_layers as usize,
            cfg.kv_heads as usize,
            cfg.head_dim as usize,
            ctx,
            precision,
        )
        .unwrap()
    }

    /// The allocation equals what the planner priced: attention KV from
    /// the cache plus the recurrent/convolution state from the operators
    /// is exactly `Qwen35StateLayout::bytes_for`.
    #[test]
    fn cache_and_state_allocations_equal_the_planned_reservation() {
        let (cfg, layout) = fixture();
        let ctx = 1024usize;
        let cache = fixture_cache(ctx, KvPrecision::F16);
        let mut state_bytes = 0u64;
        for _ in 0..layout.recurrent_layers {
            state_bytes += GdnState::new(
                cfg.gdn.dt_rank as usize,
                cfg.gdn.state_size as usize,
                cfg.gdn.head_v as usize,
            )
            .state_bytes();
            state_bytes +=
                ConvState::new(cfg.gdn.conv_kernel as usize, cfg.conv_channels() as usize)
                    .state_bytes();
        }
        assert_eq!(
            cache.bytes() + state_bytes,
            layout
                .bytes_for(&cfg, ctx as u64, KvPrecision::F16.element_bytes())
                .unwrap()
        );
        // f32 pricing is 4 bytes per element and changes only the
        // attention term.
        let cache32 = fixture_cache(ctx, KvPrecision::F32);
        assert_eq!(cache32.bytes(), cache.bytes() * 2);
        assert_eq!(
            cache32.bytes() + state_bytes,
            layout
                .bytes_for(&cfg, ctx as u64, KvPrecision::F32.element_bytes())
                .unwrap()
        );
    }

    /// Pinned-model numbers: 6 attention layers x 2 KV heads x 512
    /// (key + value) x 2048 x 2 B = 25,165,824 B, and the operator state
    /// containers hold the published 20,201,472 B.
    #[test]
    fn pinned_allocations_match_the_published_plan_numbers() {
        let cache = Qwen35KvCache::new(6, 2, 256, 2048, KvPrecision::F16).unwrap();
        assert_eq!(cache.bytes(), 25_165_824);
        let mut state = 0u64;
        for _ in 0..18 {
            state += GdnState::new(16, 128, 128).state_bytes();
            state += ConvState::new(4, 6144).state_bytes();
        }
        assert_eq!(state, 20_201_472);
        assert_eq!(cache.bytes() + state, 45_367_296);
    }

    #[test]
    fn f16_roundtrip_is_within_half_precision() {
        let mut cache = Qwen35KvCache::new(1, 1, 4, 8, KvPrecision::F16).unwrap();
        let k = [1.5f32, -2.25, 1.2345, 0.0078125];
        let v = [0.0f32, 1.0, -1.0, 100.0];
        cache.write(0, 0, &k, &v).unwrap();
        let mut row = [0.0f32; 4];
        cache.key_row(0, 0, 0, &mut row).unwrap();
        for (got, want) in row.iter().zip(k.iter()) {
            let tol = want.abs() * 1e-3 + 1e-6;
            assert!((got - want).abs() <= tol, "{got} vs {want}");
        }
        cache.value_row(0, 0, 0, &mut row).unwrap();
        for (got, want) in row.iter().zip(v.iter()) {
            let tol = want.abs() * 1e-3 + 1e-6;
            assert!((got - want).abs() <= tol, "{got} vs {want}");
        }
    }

    #[test]
    fn f32_precision_roundtrips_exactly() {
        let mut cache = Qwen35KvCache::new(1, 1, 2, 4, KvPrecision::F32).unwrap();
        let k = [0.1f32, -12345.678];
        cache.write(0, 0, &k, &k).unwrap();
        let mut row = [0.0f32; 2];
        cache.key_row(0, 0, 0, &mut row).unwrap();
        assert_eq!(row, k);
    }

    #[test]
    fn writes_beyond_the_admitted_capacity_refuse() {
        let mut cache = Qwen35KvCache::new(1, 1, 2, 2, KvPrecision::F16).unwrap();
        let rows = [1.0f32, 2.0];
        cache.write(0, 0, &rows, &rows).unwrap();
        cache.write(0, 1, &rows, &rows).unwrap();
        let err = cache.write(0, 2, &rows, &rows).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidPlan);
        assert!(err.message().contains("exceeds"), "{}", err.message());
        assert!(err.action().contains("larger context"), "{err}");
    }

    /// Attention through the packed cache equals attention on f32
    /// storage within half-precision tolerance — the storage precision
    /// and the attention operator agree.
    #[test]
    fn attention_through_the_packed_cache_matches_f32() {
        let (cfg, _) = fixture();
        let kv_heads = cfg.kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let heads_q = cfg.heads as usize;
        let seq = 3usize;
        let geometry = GqaGeometry {
            heads_q,
            heads_kv: kv_heads,
            head_dim,
        };
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut seed = 0x5EEDu64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let q: Vec<f32> = (0..heads_q * head_dim).map(|_| next()).collect();

        let mut cache = fixture_cache(seq, KvPrecision::F16);
        let mut f32_k = vec![0.0f32; kv_heads * seq * head_dim];
        let mut f32_v = vec![0.0f32; kv_heads * seq * head_dim];
        for t in 0..seq {
            let k: Vec<f32> = (0..kv_heads * head_dim).map(|_| next()).collect();
            let v: Vec<f32> = (0..kv_heads * head_dim).map(|_| next()).collect();
            cache.write(0, t, &k, &v).unwrap();
            // The attention operator's cache layout is head-major:
            // [(kv_head * seq + token) * head_dim].
            for h in 0..kv_heads {
                let src = h * head_dim;
                let dst = (h * seq + t) * head_dim;
                f32_k[dst..dst + head_dim].copy_from_slice(&k[src..src + head_dim]);
                f32_v[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
            }
        }

        let mut packed_k = vec![0.0f32; kv_heads * seq * head_dim];
        let mut packed_v = vec![0.0f32; kv_heads * seq * head_dim];
        let mut row = vec![0.0f32; head_dim];
        for t in 0..seq {
            for h in 0..kv_heads {
                let base = (h * seq + t) * head_dim;
                cache.key_row(0, h, t, &mut row).unwrap();
                packed_k[base..base + head_dim].copy_from_slice(&row);
                cache.value_row(0, h, t, &mut row).unwrap();
                packed_v[base..base + head_dim].copy_from_slice(&row);
            }
        }

        let mut out_f32 = vec![0.0f32; heads_q * head_dim];
        let mut out_packed = vec![0.0f32; heads_q * head_dim];
        gqa_cache_attention(&q, &f32_k, &f32_v, geometry, seq, scale, &mut out_f32).unwrap();
        gqa_cache_attention(
            &q,
            &packed_k,
            &packed_v,
            geometry,
            seq,
            scale,
            &mut out_packed,
        )
        .unwrap();
        for (a, b) in out_f32.iter().zip(out_packed.iter()) {
            assert!((a - b).abs() <= 1e-3, "{a} vs {b}");
        }
    }
}
