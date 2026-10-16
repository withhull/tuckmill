//! Deterministic cost model (PRD §13).
//!
//! Estimates committed tokens per second and time-to-first-token for each
//! candidate mode from bytes moved and a documented bandwidth/compute
//! assumption set. The denominator includes critical-path I/O, compute, and
//! synchronization; we do not subtract I/O time merely because two queues
//! exist. Every output is a projection carrying the assumptions used.

use serde::{Deserialize, Serialize};
use tuckmill_manifest::plan::ExecutionMode;

use crate::budget::WeightSplit;
use crate::PlannerPolicy;

/// A low/mid/high assumption band for one quantity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Band {
    pub low: f64,
    pub mid: f64,
    pub high: f64,
}

impl Band {
    pub fn new(low: f64, mid: f64, high: f64) -> Self {
        Self { low, mid, high }
    }
}

/// Estimated behavior of one mode for the declared workload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScheduleEstimate {
    pub mode: ExecutionMode,
    pub feasible: bool,
    /// Human-readable reason the mode is or is not feasible.
    pub reason: String,
    /// Logical bytes read per committed decode token (no speculation).
    pub bytes_per_token: u64,
    /// Logical bytes read during prefill for the standard prompt.
    pub prefill_read_bytes: u64,
    /// Committed tokens/second.
    pub tps: Band,
    /// Time to first committed token, milliseconds.
    pub ttft_ms: Band,
}

/// Inputs to the estimate for one mode.
#[derive(Debug, Clone, Copy)]
pub struct EstimateInputs {
    pub split: WeightSplit,
    /// Active weight bytes touched per decode token (MoE sparsity applied).
    pub active_weight_bytes_per_token: u64,
    /// Expert bytes crossing storage per decode token (streamed MoE).
    pub streamed_expert_bytes_per_token: u64,
    /// Expert bytes read during one standard-prompt prefill (union model).
    pub prefill_expert_read_bytes: u64,
    /// KV bytes per sequence position.
    pub kv_bytes_per_token: u64,
    /// Standard prompt length used for the TTFT projection.
    pub prompt_tokens: u64,
}

/// `active_params` estimate: compute is modeled as F16-equivalent arithmetic
/// (`assumed_bytes_per_param`), a documented assumption, not a hardware fact.
fn compute_time_seconds(
    active_weight_bytes: u64,
    tokens: u64,
    flops_per_s: f64,
    assumed_bytes_per_param: f64,
) -> f64 {
    let params = active_weight_bytes as f64 / assumed_bytes_per_param;
    2.0 * params * tokens as f64 / flops_per_s
}

fn tps_from(
    bytes_per_token: u64,
    active_weight_bytes_per_token: u64,
    bandwidth: f64,
    flops_per_s: f64,
    policy: &PlannerPolicy,
) -> f64 {
    if bytes_per_token == 0 {
        return f64::INFINITY;
    }
    let io_s = bytes_per_token as f64 / bandwidth;
    let compute_s = compute_time_seconds(active_weight_bytes_per_token, 1, flops_per_s, 2.0)
        / policy.bandwidth_efficiency.max(0.05);
    let per_token_s = io_s + compute_s;
    if per_token_s <= 0.0 {
        return f64::INFINITY;
    }
    1.0 / per_token_s
}

fn ttft_ms_from(
    prefill_read_bytes: u64,
    active_weight_bytes: u64,
    prompt_tokens: u64,
    bandwidth: f64,
    flops_per_s: f64,
    policy: &PlannerPolicy,
) -> f64 {
    let io_s = if bandwidth > 0.0 {
        prefill_read_bytes as f64 / bandwidth
    } else {
        0.0
    };
    let compute_s = compute_time_seconds(active_weight_bytes, prompt_tokens, flops_per_s, 2.0)
        / policy.bandwidth_efficiency.max(0.05);
    (io_s + compute_s) * 1000.0
}

fn band3<F: Fn(f64, f64) -> f64>(policy: &PlannerPolicy, f: F) -> Band {
    let low = f(
        policy
            .memory_bandwidth_low
            .min(policy.ssd_read_bandwidth_low),
        policy.prefill_flops_low,
    );
    let mid = f(
        policy
            .memory_bandwidth_mid
            .min(policy.ssd_read_bandwidth_mid),
        policy.prefill_flops_mid,
    );
    let high = f(
        policy
            .memory_bandwidth_high
            .min(policy.ssd_read_bandwidth_high),
        policy.prefill_flops_high,
    );
    Band::new(low, mid, high)
}

/// Estimate resident execution.
pub fn estimate_resident(
    inputs: &EstimateInputs,
    feasible: bool,
    reason: impl Into<String>,
    policy: &PlannerPolicy,
    memory_budget_ok: bool,
) -> ScheduleEstimate {
    let bytes_per_token = inputs
        .active_weight_bytes_per_token
        .saturating_add(inputs.kv_bytes_per_token);
    let tps = Band::new(
        tps_from(
            bytes_per_token,
            inputs.active_weight_bytes_per_token,
            policy.memory_bandwidth_low * policy.bandwidth_efficiency,
            policy.prefill_flops_low,
            policy,
        ),
        tps_from(
            bytes_per_token,
            inputs.active_weight_bytes_per_token,
            policy.memory_bandwidth_mid * policy.bandwidth_efficiency,
            policy.prefill_flops_mid,
            policy,
        ),
        tps_from(
            bytes_per_token,
            inputs.active_weight_bytes_per_token,
            policy.memory_bandwidth_high * policy.bandwidth_efficiency,
            policy.prefill_flops_high,
            policy,
        ),
    );
    // TTFT band convention: `low` is the optimistic (fastest assumptions)
    // time, `high` the pessimistic one.
    let ttft = Band::new(
        ttft_ms_from(
            0,
            inputs.active_weight_bytes_per_token,
            inputs.prompt_tokens,
            policy.memory_bandwidth_high,
            policy.prefill_flops_high,
            policy,
        ),
        ttft_ms_from(
            0,
            inputs.active_weight_bytes_per_token,
            inputs.prompt_tokens,
            policy.memory_bandwidth_mid,
            policy.prefill_flops_mid,
            policy,
        ),
        ttft_ms_from(
            0,
            inputs.active_weight_bytes_per_token,
            inputs.prompt_tokens,
            policy.memory_bandwidth_low,
            policy.prefill_flops_low,
            policy,
        ),
    );
    let _ = memory_budget_ok;
    ScheduleEstimate {
        mode: ExecutionMode::Resident,
        feasible,
        reason: reason.into(),
        bytes_per_token,
        prefill_read_bytes: 0,
        tps,
        ttft_ms: ttft,
    }
}

/// Estimate native-routing MoE streaming: non-expert weights stay resident,
/// required expert blocks stream from storage.
pub fn estimate_streamed_moe(
    inputs: &EstimateInputs,
    feasible: bool,
    reason: impl Into<String>,
    policy: &PlannerPolicy,
) -> ScheduleEstimate {
    let bytes_per_token = inputs
        .streamed_expert_bytes_per_token
        .saturating_add(inputs.kv_bytes_per_token);
    let per_band = |bw: f64, flops: f64| {
        tps_from(
            bytes_per_token,
            inputs.active_weight_bytes_per_token,
            bw,
            flops,
            policy,
        )
    };
    let tps = band3(policy, |bw, flops| {
        per_band(bw.min(policy.ssd_read_bandwidth_mid), flops)
    });
    let ttft = Band::new(
        ttft_ms_from(
            inputs.prefill_expert_read_bytes,
            inputs.active_weight_bytes_per_token,
            inputs.prompt_tokens,
            policy.ssd_read_bandwidth_high,
            policy.prefill_flops_high,
            policy,
        ),
        ttft_ms_from(
            inputs.prefill_expert_read_bytes,
            inputs.active_weight_bytes_per_token,
            inputs.prompt_tokens,
            policy.ssd_read_bandwidth_mid,
            policy.prefill_flops_mid,
            policy,
        ),
        ttft_ms_from(
            inputs.prefill_expert_read_bytes,
            inputs.active_weight_bytes_per_token,
            inputs.prompt_tokens,
            policy.ssd_read_bandwidth_low,
            policy.prefill_flops_low,
            policy,
        ),
    );
    ScheduleEstimate {
        mode: ExecutionMode::StreamedMoe,
        feasible,
        reason: reason.into(),
        bytes_per_token,
        prefill_read_bytes: inputs.prefill_expert_read_bytes,
        tps,
        ttft_ms: ttft,
    }
}

/// Estimate dense streaming: resident core stays, tiles stream.
pub fn estimate_streamed_dense(
    inputs: &EstimateInputs,
    feasible: bool,
    reason: impl Into<String>,
    policy: &PlannerPolicy,
) -> ScheduleEstimate {
    let streamed = inputs.split.streamed_tiles;
    let bytes_per_token = streamed.saturating_add(inputs.kv_bytes_per_token);
    let tps = Band::new(
        tps_from(
            bytes_per_token,
            streamed,
            policy.ssd_read_bandwidth_low,
            policy.prefill_flops_low,
            policy,
        ),
        tps_from(
            bytes_per_token,
            streamed,
            policy.ssd_read_bandwidth_mid,
            policy.prefill_flops_mid,
            policy,
        ),
        tps_from(
            bytes_per_token,
            streamed,
            policy.ssd_read_bandwidth_high,
            policy.prefill_flops_high,
            policy,
        ),
    );
    let ttft = Band::new(
        ttft_ms_from(
            streamed,
            streamed,
            inputs.prompt_tokens,
            policy.ssd_read_bandwidth_high,
            policy.prefill_flops_high,
            policy,
        ),
        ttft_ms_from(
            streamed,
            streamed,
            inputs.prompt_tokens,
            policy.ssd_read_bandwidth_mid,
            policy.prefill_flops_mid,
            policy,
        ),
        ttft_ms_from(
            streamed,
            streamed,
            inputs.prompt_tokens,
            policy.ssd_read_bandwidth_low,
            policy.prefill_flops_low,
            policy,
        ),
    );
    ScheduleEstimate {
        mode: ExecutionMode::StreamedDense,
        feasible,
        reason: reason.into(),
        bytes_per_token,
        prefill_read_bytes: streamed,
        tps,
        ttft_ms: ttft,
    }
}

/// Choose a schedule: a candidate must beat the baseline's projected mid
/// throughput by at least `margin` (proposed default 15%) before it is
/// preferred. The baseline is the first feasible estimate in given order
/// (resident, streamed MoE, streamed dense). Online cooldown between changes
/// is a runtime concern (PRD §13).
pub fn choose_schedule(estimates: &[ScheduleEstimate], margin: f64) -> Option<&ScheduleEstimate> {
    let baseline = estimates.iter().find(|e| e.feasible)?;
    let best = estimates
        .iter()
        .filter(|e| e.feasible)
        .max_by(|a, b| a.tps.mid.total_cmp(&b.tps.mid))?;
    if best.mode == baseline.mode {
        return Some(best);
    }
    if best.tps.mid >= baseline.tps.mid * (1.0 + margin) {
        Some(best)
    } else {
        Some(baseline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> EstimateInputs {
        EstimateInputs {
            split: WeightSplit {
                total: 3_000_000_000,
                moe_expert: 2_500_000_000,
                non_expert: 500_000_000,
                resident_core: 300_000_000,
                streamed_tiles: 200_000_000,
            },
            active_weight_bytes_per_token: 400_000_000,
            streamed_expert_bytes_per_token: 160_000_000,
            prefill_expert_read_bytes: 2_500_000_000,
            kv_bytes_per_token: 200_000,
            prompt_tokens: 512,
        }
    }

    #[test]
    fn resident_estimate_is_positive_and_ordered() {
        let policy = PlannerPolicy::default();
        let est = estimate_resident(&inputs(), true, "fits", &policy, true);
        assert!(est.tps.low <= est.tps.mid);
        assert!(est.tps.mid <= est.tps.high);
        assert!(est.ttft_ms.low <= est.ttft_ms.mid);
        assert!(est.ttft_ms.mid <= est.ttft_ms.high);
        assert!(est.tps.mid > 0.0);
    }

    #[test]
    fn streamed_moe_reads_expert_bytes_per_token() {
        let policy = PlannerPolicy::default();
        let est = estimate_streamed_moe(&inputs(), true, "fits", &policy);
        assert_eq!(
            est.bytes_per_token,
            inputs().streamed_expert_bytes_per_token + inputs().kv_bytes_per_token
        );
        assert!(est.tps.mid > 0.0);
    }

    #[test]
    fn fifteen_percent_margin_rule_keeps_baseline() {
        let policy = PlannerPolicy::default();
        let resident = estimate_resident(&inputs(), true, "baseline", &policy, true);
        // Only 5% better: below the margin, baseline must win.
        let mut alt = resident.clone();
        alt.mode = ExecutionMode::StreamedMoe;
        alt.tps.mid = resident.tps.mid * 1.05;
        let candidates = [resident.clone(), alt];
        let chosen = choose_schedule(&candidates, 0.15).unwrap();
        assert_eq!(chosen.mode, ExecutionMode::Resident);

        // 20% better: the alternative wins.
        let mut alt = resident.clone();
        alt.mode = ExecutionMode::StreamedMoe;
        alt.tps.mid = resident.tps.mid * 1.20;
        let candidates = [resident, alt];
        let chosen = choose_schedule(&candidates, 0.15).unwrap();
        assert_eq!(chosen.mode, ExecutionMode::StreamedMoe);
    }

    #[test]
    fn infeasible_candidates_are_never_chosen() {
        let policy = PlannerPolicy::default();
        let resident = estimate_resident(&inputs(), false, "too big", &policy, false);
        let moe = estimate_streamed_moe(&inputs(), true, "fits", &policy);
        let candidates = [resident, moe.clone()];
        let chosen = choose_schedule(&candidates, 0.15).unwrap();
        assert_eq!(chosen.mode, ExecutionMode::StreamedMoe);
        let only = [moe];
        let chosen = choose_schedule(&only, 0.15).unwrap();
        assert_eq!(chosen.mode, ExecutionMode::StreamedMoe);
    }

    #[test]
    fn empty_or_all_infeasible_yields_none() {
        assert!(choose_schedule(&[], 0.15).is_none());
        let policy = PlannerPolicy::default();
        let r = estimate_resident(&inputs(), false, "no", &policy, false);
        assert!(choose_schedule(&[r], 0.15).is_none());
    }
}
