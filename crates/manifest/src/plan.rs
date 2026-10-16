//! Execution plan schema.
//!
//! A plan is produced before model loading and is inspectable and exportable
//! (FR-03). It records its reason, assumptions, cache policy, reservations,
//! expected latency interval, mode, measurement confidence, invalidation
//! conditions, and safe degradation actions. It never claims that a projected
//! value was measured.

use serde::{Deserialize, Serialize};

use crate::ids::Id;
use crate::records::{Confidence, LatencyClass, ReservationBreakdown, ReserveClass};
use crate::units::Bytes;
use crate::{BACKEND_ABI_VERSION, SCHEMA_VERSION, WORKER_PROTOCOL_VERSION};

/// Execution modes (PRD §6). The runtime changes scheduling and cache
/// allocation, never the requested model's identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Weights, admitted context, and workspaces fit with headroom.
    Resident,
    /// Native-routing MoE streaming (P1; requires FR-06).
    StreamedMoe,
    /// Tiled dense streaming (P1; requires FR-15).
    StreamedDense,
    /// Rejected before loading.
    Rejected,
}

impl ExecutionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ExecutionMode::Resident => "resident",
            ExecutionMode::StreamedMoe => "streamed-moe",
            ExecutionMode::StreamedDense => "streamed-dense",
            ExecutionMode::Rejected => "rejected",
        }
    }
}

/// Whether an execution engine exists for the plan's artifact and profile.
/// Capacity feasibility and execution availability are separate statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineAvailability {
    /// No engine in this build can execute this profile correctly.
    None,
    /// The deterministic reference test engine can run the pipeline with
    /// synthetic output. It is not model output and never certified.
    ReferenceTestOnly,
    /// A certified engine for this exact artifact/profile/device.
    Certified,
}

/// Cache policy attached to the plan (PRD §11.2 E01).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachePolicy {
    /// "none" (resident path), "bounded-explicit", "os-managed",
    /// "mixed".
    pub policy: String,
    pub expert_cache_bytes: Bytes,
    /// Protected in-flight blocks are never evicted mid-read.
    pub protect_in_flight: bool,
    pub prefetch: bool,
}

/// Budget and reservation accounting (PRD §9):
/// admit only when R + U + H <= B.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanBudget {
    /// Requested process budget B.
    pub requested_budget: Bytes,
    /// Controlled live-allocation reservation R.
    pub reservation: Bytes,
    /// Estimated unavoidable runtime overhead U.
    pub overhead_estimate: Bytes,
    /// Safety reserve H.
    pub safety_reserve: Bytes,
    /// Full per-category breakdown of R.
    pub reservation_breakdown: ReservationBreakdown,
}

impl PlanBudget {
    pub fn admitted_total(&self) -> Bytes {
        Bytes(
            self.reservation
                .bytes()
                .saturating_add(self.overhead_estimate.bytes())
                .saturating_add(self.safety_reserve.bytes()),
        )
    }

    pub fn fits(&self) -> bool {
        self.admitted_total().bytes() <= self.requested_budget.bytes()
    }

    pub fn headroom(&self) -> Bytes {
        Bytes(
            self.requested_budget
                .bytes()
                .saturating_sub(self.admitted_total().bytes()),
        )
    }

    pub fn add_reservation(&mut self, class: ReserveClass, bytes: Bytes) {
        self.reservation_breakdown.add(class, bytes);
        self.reservation = self.reservation_breakdown.total();
    }
}

/// Admitted request geometry. `--context` is the total sequence allowance
/// including output (PRD §15); admission includes input + maximum output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanContext {
    pub total_context: u64,
    pub max_output_tokens: u64,
    /// Prompt budget after reserving output: total - max_output.
    pub max_input_tokens: u64,
}

impl PlanContext {
    pub fn new(total_context: u64, max_output_tokens: u64) -> Self {
        Self {
            total_context,
            max_output_tokens,
            max_input_tokens: total_context.saturating_sub(max_output_tokens),
        }
    }
}

/// Expected latency interval. All bounds are projections unless the
/// confidence label says otherwise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedLatency {
    pub ttft_ms_low: f64,
    pub ttft_ms_high: f64,
    pub tps_low: f64,
    pub tps_high: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnginePlan {
    pub availability: EngineAvailability,
    pub engine: String,
    /// Kernel packages this profile would require when a certified executor
    /// exists (e.g. K01..K03). Recorded so kernel gates are explicit.
    pub required_kernels: Vec<String>,
    pub required_transforms: Vec<String>,
}

/// Typed infeasibility explanation attached to a rejected plan (FR-03).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanRejection {
    /// A public error code string, e.g. "CONTEXT_EXCEEDS_BUDGET".
    pub code: String,
    pub message: String,
}

/// The versioned, inspectable execution plan (FR-03, PRD §13).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub schema_version: u32,
    pub plan_id: Id,
    pub created_unix_ms: u64,
    pub artifact_id: Id,
    pub device_id: Id,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_hash: Option<Id>,
    pub adapter_id: String,
    pub mode: ExecutionMode,
    pub context: PlanContext,
    pub budget: PlanBudget,
    pub cache_policy: CachePolicy,
    pub engine: EnginePlan,
    pub performance_class: LatencyClass,
    pub confidence: Confidence,
    /// Human-readable reason for the chosen mode, including rejections.
    pub reason: String,
    pub assumptions: Vec<String>,
    /// Conditions that make this plan stale (PRD §5).
    pub invalidation_conditions: Vec<String>,
    /// Safe degradation actions in order of preference.
    pub degradation_actions: Vec<String>,
    /// Present when the plan rejects the configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejection: Option<PlanRejection>,
    pub backend_abi_version: u32,
    pub worker_protocol_version: u16,
}

impl ExecutionPlan {
    /// Recompute the plan ID from the plan body. Plans are immutable; the ID
    /// is the hash of everything except the ID field itself.
    pub fn compute_id(&self) -> crate::Result<Id> {
        #[derive(Serialize)]
        struct Body<'a> {
            schema_version: u32,
            created_unix_ms: u64,
            artifact_id: &'a Id,
            device_id: &'a Id,
            tokenizer_hash: &'a Option<Id>,
            adapter_id: &'a str,
            mode: ExecutionMode,
            context: PlanContext,
            budget: &'a PlanBudget,
            cache_policy: &'a CachePolicy,
            engine: &'a EnginePlan,
            performance_class: LatencyClass,
            confidence: Confidence,
            reason: &'a str,
            assumptions: &'a [String],
            invalidation_conditions: &'a [String],
            degradation_actions: &'a [String],
            rejection: &'a Option<PlanRejection>,
            backend_abi_version: u32,
            worker_protocol_version: u16,
        }
        let body = Body {
            schema_version: self.schema_version,
            created_unix_ms: self.created_unix_ms,
            artifact_id: &self.artifact_id,
            device_id: &self.device_id,
            tokenizer_hash: &self.tokenizer_hash,
            adapter_id: &self.adapter_id,
            mode: self.mode,
            context: self.context,
            budget: &self.budget,
            cache_policy: &self.cache_policy,
            engine: &self.engine,
            performance_class: self.performance_class,
            confidence: self.confidence,
            reason: &self.reason,
            assumptions: &self.assumptions,
            invalidation_conditions: &self.invalidation_conditions,
            degradation_actions: &self.degradation_actions,
            rejection: &self.rejection,
            backend_abi_version: self.backend_abi_version,
            worker_protocol_version: self.worker_protocol_version,
        };
        crate::ids::hash_of(&body)
    }

    pub fn new_skeleton(artifact_id: Id, device_id: Id, adapter_id: String) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            plan_id: Id::new("0".repeat(64)).expect("zero id is valid"),
            created_unix_ms: crate::ids::now_unix_ms(),
            artifact_id,
            device_id,
            tokenizer_hash: None,
            adapter_id,
            mode: ExecutionMode::Rejected,
            context: PlanContext::new(0, 0),
            budget: PlanBudget {
                requested_budget: Bytes::ZERO,
                reservation: Bytes::ZERO,
                overhead_estimate: Bytes::ZERO,
                safety_reserve: Bytes::ZERO,
                reservation_breakdown: ReservationBreakdown::default(),
            },
            cache_policy: CachePolicy {
                policy: "none".into(),
                expert_cache_bytes: Bytes::ZERO,
                protect_in_flight: true,
                prefetch: false,
            },
            engine: EnginePlan {
                availability: EngineAvailability::None,
                engine: "none".into(),
                required_kernels: Vec::new(),
                required_transforms: Vec::new(),
            },
            performance_class: LatencyClass::Infeasible,
            confidence: Confidence::Estimated,
            reason: String::new(),
            assumptions: Vec::new(),
            invalidation_conditions: vec![
                "backend or worker ABI version change".into(),
                "artifact content change".into(),
                "device profile change (memory, OS, storage)".into(),
            ],
            degradation_actions: vec![
                "trim expendable caches and reduce prefetch depth".into(),
                "pause at the next safe token boundary under critical pressure".into(),
                "fail the request with a retriable error if essential state cannot be retained"
                    .into(),
            ],
            rejection: None,
            backend_abi_version: BACKEND_ABI_VERSION,
            worker_protocol_version: WORKER_PROTOCOL_VERSION,
        }
    }

    /// Validate internal consistency; used when loading plan documents.
    pub fn validate(&self) -> crate::Result<()> {
        use crate::error::{ErrorCode, TuckmillError};
        let err = |msg: String| TuckmillError::new(ErrorCode::InvalidPlan, msg);
        if self.schema_version != SCHEMA_VERSION {
            return Err(err(format!(
                "plan schema version {} is not supported (want {})",
                self.schema_version, SCHEMA_VERSION
            )));
        }
        if self.backend_abi_version != BACKEND_ABI_VERSION {
            return Err(TuckmillError::new(
                ErrorCode::PlanStale,
                format!(
                    "plan was built for backend ABI {} but this build is {}",
                    self.backend_abi_version, BACKEND_ABI_VERSION
                ),
            ));
        }
        if self.worker_protocol_version != WORKER_PROTOCOL_VERSION {
            return Err(TuckmillError::new(
                ErrorCode::PlanStale,
                format!(
                    "plan was built for worker protocol {} but this build is {}",
                    self.worker_protocol_version, WORKER_PROTOCOL_VERSION
                ),
            ));
        }
        if self.context.max_output_tokens > self.context.total_context {
            return Err(err("max_output_tokens exceeds total context".into()));
        }
        let recomputed = self.compute_id()?;
        if recomputed != self.plan_id {
            return Err(err(format!(
                "plan id mismatch: recorded {}, computed {}",
                self.plan_id, recomputed
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_plan() -> ExecutionPlan {
        let mut p = ExecutionPlan::new_skeleton(
            Id::new("a".repeat(64)).unwrap(),
            Id::new("b".repeat(64)).unwrap(),
            "qwen3".into(),
        );
        p.mode = ExecutionMode::Resident;
        p.context = PlanContext::new(2048, 256);
        p.budget.requested_budget = Bytes::parse("2GiB").unwrap();
        p.budget
            .add_reservation(ReserveClass::Weights, Bytes::parse("700MiB").unwrap());
        p.budget
            .add_reservation(ReserveClass::KvState, Bytes::parse("200MiB").unwrap());
        p.budget.overhead_estimate = Bytes::parse("400MiB").unwrap();
        p.budget.safety_reserve = Bytes::parse("300MiB").unwrap();
        p.plan_id = p.compute_id().unwrap();
        p
    }

    #[test]
    fn budget_admission_rule_r_plus_u_plus_h() {
        let p = sample_plan();
        assert_eq!(
            p.budget.reservation.bytes(),
            700 * 1024 * 1024 + 200 * 1024 * 1024
        );
        assert_eq!(
            p.budget.admitted_total().bytes(),
            900 * 1024 * 1024 + 400 * 1024 * 1024 + 300 * 1024 * 1024
        );
        assert!(p.budget.fits(), "1.6GiB must fit in 2GiB");
        assert_eq!(
            p.budget.headroom().bytes(),
            2 * 1024 * 1024 * 1024 - 1600 * 1024 * 1024
        );
    }

    #[test]
    fn plan_id_detects_tampering() {
        let mut p = sample_plan();
        p.validate().unwrap();
        p.budget.safety_reserve = Bytes::parse("900MiB").unwrap();
        let err = p.validate().unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidPlan);
    }

    #[test]
    fn context_reserves_output_inside_total() {
        let c = PlanContext::new(2048, 256);
        assert_eq!(c.max_input_tokens, 1792);
        let mut p = sample_plan();
        p.context = PlanContext::new(256, 512);
        p.plan_id = p.compute_id().unwrap();
        let err = p.validate().unwrap_err();
        assert!(err.message().contains("exceeds total context"));
    }

    #[test]
    fn abi_mismatch_is_plan_stale() {
        let mut p = sample_plan();
        p.backend_abi_version = BACKEND_ABI_VERSION + 1;
        p.plan_id = p.compute_id().unwrap();
        let err = p.validate().unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::PlanStale);
    }
}
