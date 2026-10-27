//! Memory model and reservation math (PRD §9).
//!
//! Admit execution only when `R + U + H <= B`, counting resident weights,
//! attention state, I/O buffers, workspaces, and caches. This is a managed
//! admission and allocation policy, not an operating-system guarantee; the
//! plan says so.

use tuckmill_manifest::plan::{PlanBudget, PlanContext};
use tuckmill_manifest::records::{ReservationBreakdown, ReserveClass, TensorStats};
use tuckmill_manifest::units::{Bytes, GIB};

use crate::kv::estimate_kv_bytes;
use crate::PlannerPolicy;

/// Hardware-adaptive default process budget (ADR-006): a quarter of unified
/// memory, floored at 2 GiB (the PRD's conservative starter on the 8 GB
/// minimum tier) and capped at 24 GiB so the admission reserve stays
/// meaningful on very large machines. Always overridable per request.
pub fn suggested_process_budget(memory_bytes: u64) -> Bytes {
    Bytes((memory_bytes / 4).clamp(2 * GIB, 24 * GIB))
}

/// Byte split of the tensor payload used for mode selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightSplit {
    pub total: u64,
    /// Native-routing MoE expert bytes (streamed in MoE mode).
    pub moe_expert: u64,
    /// Everything that is not an expert tensor.
    pub non_expert: u64,
    /// Portion expected to stay resident even in dense-streaming mode
    /// (embeddings, output head, norms).
    pub resident_core: u64,
    /// Portion processed as streamed tiles in dense-streaming mode.
    pub streamed_tiles: u64,
}

pub fn split_weights(stats: &TensorStats) -> WeightSplit {
    let get = |k: &str| stats.by_class.get(k).copied().unwrap_or(0);
    let moe_expert = get("moe_expert");
    let resident_core = get("token_embedding")
        .saturating_add(get("output_head"))
        .saturating_add(get("norm"))
        .saturating_add(get("attn_norm"));
    let total = stats.total_bytes;
    let non_expert = total.saturating_sub(moe_expert);
    let streamed_tiles = total
        .saturating_sub(resident_core)
        .saturating_sub(moe_expert);
    WeightSplit {
        total,
        moe_expert,
        non_expert,
        resident_core,
        streamed_tiles,
    }
}

/// Inputs needed to price a concrete mode.
#[derive(Debug, Clone, Copy)]
pub struct BudgetInputs {
    pub requested_budget: Bytes,
    pub policy_overhead: Bytes,
    pub policy_safety_reserve: Bytes,
    pub active_params: u64,
}

/// Compute the reservations for a mode. `expert_cache` is only meaningful in
/// MoE streaming; `kv_tokens` should be the admitted total context.
pub struct ModeReservation {
    pub weights: u64,
    pub kv: u64,
    pub workspaces: u64,
    pub io_buffers: u64,
    pub expert_cache: u64,
}

pub fn reserve_resident(
    split: &WeightSplit,
    kv_bytes: u64,
    policy: &PlannerPolicy,
) -> ModeReservation {
    ModeReservation {
        weights: split.total,
        kv: kv_bytes,
        workspaces: policy.workspaces_resident_bytes,
        io_buffers: policy.io_buffers_resident_bytes,
        expert_cache: 0,
    }
}

pub fn reserve_streamed_moe(
    split: &WeightSplit,
    kv_bytes: u64,
    expert_cache_allowance: u64,
    policy: &PlannerPolicy,
) -> ModeReservation {
    ModeReservation {
        weights: split.non_expert,
        kv: kv_bytes,
        workspaces: policy.workspaces_streamed_bytes,
        io_buffers: policy.io_buffers_streamed_bytes,
        expert_cache: expert_cache_allowance.min(split.moe_expert),
    }
}

pub fn reserve_streamed_dense(
    split: &WeightSplit,
    kv_bytes: u64,
    policy: &PlannerPolicy,
) -> ModeReservation {
    ModeReservation {
        weights: split.resident_core,
        kv: kv_bytes,
        workspaces: policy.workspaces_streamed_bytes,
        io_buffers: policy.io_buffers_streamed_bytes,
        expert_cache: 0,
    }
}

/// Total reserved bytes for a reservation.
pub fn reservation_total(r: &ModeReservation) -> u64 {
    r.weights
        .saturating_add(r.kv)
        .saturating_add(r.workspaces)
        .saturating_add(r.io_buffers)
        .saturating_add(r.expert_cache)
}

/// Build a PlanBudget from a reservation. Admission rule: R + U + H <= B.
pub fn make_plan_budget(
    requested_budget: Bytes,
    inputs_overhead: Bytes,
    safety_reserve: Bytes,
    reservation: &ModeReservation,
) -> PlanBudget {
    let mut breakdown = ReservationBreakdown::default();
    breakdown.add(ReserveClass::Weights, Bytes(reservation.weights));
    breakdown.add(ReserveClass::KvState, Bytes(reservation.kv));
    breakdown.add(ReserveClass::Workspaces, Bytes(reservation.workspaces));
    breakdown.add(ReserveClass::IoBuffers, Bytes(reservation.io_buffers));
    breakdown.add(ReserveClass::ExpertCache, Bytes(reservation.expert_cache));
    PlanBudget {
        requested_budget,
        reservation: breakdown.total(),
        overhead_estimate: inputs_overhead,
        safety_reserve,
        reservation_breakdown: breakdown,
    }
}

/// KV payload for an admitted context: attention KV on the layers that
/// carry it, plus the constant recurrent state of hybrid stacks
/// (context-independent), reserved twice: the live session state and the
/// engine's prefix-cache snapshot of it (ADR-012 step 2). Attention KV
/// needs no extra reservation for the prefix cache: reused rows are the
/// session's own.
pub fn kv_bytes_for(
    geometry: Option<&tuckmill_manifest::records::ModelGeometry>,
    context: &PlanContext,
    bytes_per_element: u32,
) -> Option<u64> {
    let g = geometry?;
    let attention_layers = g.attention_layers.unwrap_or(g.layers);
    let kv_geometry = tuckmill_manifest::adapter::KvGeometry {
        layers: attention_layers,
        attention_layers,
        state_bytes: 0,
        head_count: g.head_count,
        kv_head_count: g.kv_head_count,
        head_dim: g.key_length,
        key_length: g.key_length,
        value_length: g.value_length,
    };
    let attention = estimate_kv_bytes(&kv_geometry, context.total_context, bytes_per_element)?;
    let state = g.state_bytes.unwrap_or(0).checked_mul(2)?;
    attention.total_bytes.checked_add(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tuckmill_manifest::records::TensorStats;

    fn stats() -> TensorStats {
        let mut by_class = BTreeMap::new();
        by_class.insert("token_embedding".to_string(), 100);
        by_class.insert("output_head".to_string(), 100);
        by_class.insert("norm".to_string(), 20);
        by_class.insert("attn_norm".to_string(), 20);
        by_class.insert("attention".to_string(), 300);
        by_class.insert("ffn".to_string(), 200);
        by_class.insert("moe_expert".to_string(), 4_000);
        TensorStats {
            total_tensors: 7,
            total_bytes: 4_740,
            by_class,
            by_type: BTreeMap::new(),
        }
    }

    #[test]
    fn weight_split_accounts_everything() {
        let s = split_weights(&stats());
        assert_eq!(s.total, 4_740);
        assert_eq!(s.moe_expert, 4_000);
        assert_eq!(s.non_expert, 740);
        assert_eq!(s.resident_core, 240);
        assert_eq!(s.streamed_tiles, 500);
    }

    #[test]
    fn admission_rule_is_strict_r_plus_u_plus_h() {
        let policy = PlannerPolicy::default();
        let s = split_weights(&stats());
        let res = reserve_resident(&s, 1_000, &policy);
        let budget = make_plan_budget(
            Bytes(10_000),
            Bytes(policy.overhead_estimate_bytes),
            Bytes(policy.safety_reserve_floor_bytes),
            &res,
        );
        assert!(!budget.fits(), "overhead+safety far exceed 10_000 bytes");
        let big = make_plan_budget(
            Bytes(10_000_000_000),
            Bytes(192 * 1024 * 1024),
            Bytes(192 * 1024 * 1024),
            &res,
        );
        assert!(big.fits());
        assert_eq!(
            big.reservation.bytes(),
            4_740 + 1_000 + policy.workspaces_resident_bytes + policy.io_buffers_resident_bytes
        );
    }

    #[test]
    fn expert_cache_is_clamped_to_expert_bytes() {
        let policy = PlannerPolicy::default();
        let s = split_weights(&stats());
        let res = reserve_streamed_moe(&s, 0, u64::MAX, &policy);
        assert_eq!(
            res.expert_cache, 4_000,
            "cannot reserve more cache than weights exist"
        );
        let res2 = reserve_streamed_moe(&s, 0, 1_000, &policy);
        assert_eq!(res2.expert_cache, 1_000);
    }

    #[test]
    fn suggested_budget_scales_with_unified_memory() {
        assert_eq!(suggested_process_budget(8 * GIB).bytes(), 2 * GIB);
        assert_eq!(suggested_process_budget(16 * GIB).bytes(), 4 * GIB);
        assert_eq!(suggested_process_budget(24 * GIB).bytes(), 6 * GIB);
        assert_eq!(suggested_process_budget(32 * GIB).bytes(), 8 * GIB);
        assert_eq!(suggested_process_budget(64 * GIB).bytes(), 16 * GIB);
        assert_eq!(suggested_process_budget(128 * GIB).bytes(), 24 * GIB);
        assert_eq!(suggested_process_budget(512 * GIB).bytes(), 24 * GIB);
        // Floor holds below the minimum tier too (never below 2 GiB).
        assert_eq!(suggested_process_budget(4 * GIB).bytes(), 2 * GIB);
    }

    #[test]
    fn kv_bytes_missing_geometry_is_none() {
        let ctx = PlanContext::new(2048, 256);
        assert!(kv_bytes_for(None, &ctx, 2).is_none());
    }

    /// Hybrid geometry prices attention KV on the attention layers only and
    /// reserves the constant recurrent state exactly once (ADR-008).
    #[test]
    fn hybrid_kv_prices_attention_layers_plus_state_and_its_prefix_snapshot() {
        use tuckmill_manifest::records::ModelGeometry;
        let ctx = PlanContext::new(2048, 256);
        let hybrid = ModelGeometry {
            layers: 24,
            head_count: 8,
            kv_head_count: 2,
            key_length: 256,
            value_length: 256,
            attention_layers: Some(6),
            state_bytes: Some(20_201_472),
            expert_count: None,
            expert_used_per_token: None,
        };
        let attention = 6 * 2 * (256 + 256) * 2048 * 2;
        // Live recurrent state plus the prefix-cache snapshot of it.
        assert_eq!(
            kv_bytes_for(Some(&hybrid), &ctx, 2),
            Some(attention + 2 * 20_201_472)
        );

        // A dense geometry without hybrid fields is unchanged: every layer
        // carries KV and there is no constant state.
        let dense = ModelGeometry {
            attention_layers: None,
            state_bytes: None,
            ..hybrid.clone()
        };
        assert_eq!(
            kv_bytes_for(Some(&dense), &ctx, 2),
            Some(24 * 2 * (256 + 256) * 2048 * 2)
        );
    }
}
