//! Plan construction: combine artifact, adapter, device, budget, and
//! calibration evidence into an inspectable ExecutionPlan (FR-03, PRD §13).
//!
//! Capacity feasibility and execution availability are separate statements.
//! A plan can be capacity-feasible while no certified engine exists; that is
//! recorded honestly instead of being conflated.

use tuckmill_manifest::adapter::AdapterSpec;
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::plan::{
    CachePolicy, EngineAvailability, EnginePlan, ExecutionMode, ExecutionPlan, PlanBudget,
    PlanContext, PlanRejection,
};
use tuckmill_manifest::records::{Confidence, DeviceProfile, LatencyClass, ModelArtifact};
use tuckmill_manifest::units::Bytes;

use crate::budget::{
    kv_bytes_for, make_plan_budget, reservation_total, reserve_resident, reserve_streamed_dense,
    reserve_streamed_moe, split_weights, WeightSplit,
};
use crate::calibration::CalibrationStore;
use crate::cost::{
    choose_schedule, estimate_resident, estimate_streamed_dense, estimate_streamed_moe,
    EstimateInputs, ScheduleEstimate,
};
use crate::kernel_registry::KernelRegistry;
use crate::PlannerPolicy;

/// The workload the plan's latency projection describes (PRD §20.2 standard
/// workload shape). User-experience runs report their own measurements.
pub const PLAN_WORKLOAD_PROMPT_TOKENS: u64 = 512;
pub const PLAN_WORKLOAD_OUTPUT_TOKENS: u64 = 256;
pub const PLAN_WORKLOAD_SUITE: &str = "standard-v1";

/// A planning request.
#[derive(Debug, Clone)]
pub struct PlanRequest {
    pub context: PlanContext,
    pub requested_budget: Bytes,
    pub policy: PlannerPolicy,
    /// Bytes per KV element in the *actual* storage representation of the
    /// backend that will execute (for example 4 for the current TuckCore
    /// f32 KV store, 2 for an f16 llama.cpp cache). The runtime resolves
    /// this from the selected engine and its acknowledged execution
    /// options; the planner never assumes a representation on its own.
    pub kv_bytes_per_element: u32,
}

/// Active weight bytes touched per decode token for a given split.
fn active_weight_bytes(split: &WeightSplit, artifact: &ModelArtifact) -> u64 {
    let g = artifact.geometry.as_ref();
    match g.and_then(|g| Some((g.expert_count?, g.expert_used_per_token?))) {
        Some((count, used)) if count > 0 && split.moe_expert > 0 => {
            let per_expert = split.moe_expert / count.max(1);
            split
                .non_expert
                .saturating_add(per_expert.saturating_mul(used.min(count)))
        }
        _ => split.total,
    }
}

fn expert_union_fraction(artifact: &ModelArtifact, prompt_tokens: u64) -> f64 {
    let Some(g) = &artifact.geometry else {
        return 1.0;
    };
    let (Some(count), Some(used)) = (g.expert_count, g.expert_used_per_token) else {
        return 1.0;
    };
    if count == 0 {
        return 1.0;
    }
    ((used as f64) * (prompt_tokens as f64) / (count as f64)).min(1.0)
}

fn required_kernel_plan(adapter: &AdapterSpec, kernels: &KernelRegistry) -> EnginePlan {
    let missing = kernels.missing(&adapter.required_kernels);
    let all_kernels = kernels.all_available(&adapter.required_kernels);
    let certified = all_kernels && adapter.certification.state == "certified";
    let availability = if certified {
        EngineAvailability::Certified
    } else {
        // No certified executor in this build: the deterministic reference
        // test engine can exercise the pipeline with synthetic output only.
        EngineAvailability::ReferenceTestOnly
    };
    let engine = if certified {
        "native".to_string()
    } else {
        let missing_list = if missing.is_empty() {
            "adapter certification pending".to_string()
        } else {
            format!("kernels not implemented: {}", missing.join(", "))
        };
        format!("reference-test ({missing_list})")
    };
    EnginePlan {
        availability,
        engine,
        required_kernels: adapter.required_kernels.clone(),
        required_transforms: adapter.required_transforms.clone(),
    }
}

fn latency_class(ttft_ms: f64, tps: f64) -> LatencyClass {
    if ttft_ms <= 15_000.0 && tps >= 5.0 {
        LatencyClass::Interactive
    } else if ttft_ms <= 90_000.0 && tps >= 1.0 {
        LatencyClass::PatientLocal
    } else {
        LatencyClass::BatchLocal
    }
}

/// Build a plan. Returns Err only for malformed requests or adapter/artifact
/// mismatches; capacity problems produce a Rejected plan document with a
/// typed rejection attached (FR-03).
pub fn build_plan(
    kernels: &KernelRegistry,
    artifact: &ModelArtifact,
    adapter: &AdapterSpec,
    device: &DeviceProfile,
    request: &PlanRequest,
    calibration: &CalibrationStore,
) -> Result<ExecutionPlan> {
    let policy = &request.policy;

    // ---- identity and request validation ----
    let artifact_arch = artifact.arch_id.as_deref().ok_or_else(|| {
        TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            "artifact has no resolved architecture; re-inspect it with a matching adapter",
        )
    })?;
    if artifact_arch != adapter.architecture {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "adapter {} targets architecture {:?} but the artifact is {:?}",
                adapter.id, adapter.architecture, artifact_arch
            ),
        ));
    }
    if request.requested_budget.bytes() > device.memory_bytes {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            format!(
                "requested budget {} exceeds the device's physical memory {}",
                request.requested_budget,
                Bytes(device.memory_bytes)
            ),
        ));
    }
    if request.context.total_context == 0 || request.context.max_output_tokens == 0 {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "context and max output tokens must be positive",
        ));
    }
    if request.context.max_output_tokens > request.context.total_context {
        return Err(TuckmillError::new(
            ErrorCode::ContextExceedsBudget,
            "max output tokens exceed the total context allowance",
        ));
    }
    if let Some(advertised) = artifact.advertised_context_length {
        if request.context.total_context > advertised {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!(
                    "requested context {} exceeds the artifact's advertised context {}",
                    request.context.total_context, advertised
                ),
            ));
        }
    }

    let mut plan = ExecutionPlan::new_skeleton(
        artifact.artifact_id.clone(),
        device.device_id.clone(),
        adapter.id.clone(),
    );
    plan.tokenizer_hash = artifact.tokenizer_hash.clone();
    plan.context = request.context;

    let safety_reserve = Bytes(policy.safety_reserve(request.requested_budget.bytes()));
    let overhead = Bytes(policy.overhead_estimate_bytes);

    // ---- capacity math ----
    let geometry = artifact.geometry.as_ref();
    let kv_bytes = geometry
        .and_then(|_| kv_bytes_for(geometry, &request.context, request.kv_bytes_per_element));
    let geometry_missing = kv_bytes.is_none();

    let split = split_weights(&artifact.tensor_stats);
    let active = active_weight_bytes(&split, artifact);
    let expert_per_token = if split.moe_expert > 0 {
        match geometry.and_then(|g| Some((g.expert_count?, g.expert_used_per_token?))) {
            Some((count, used)) if count > 0 => (split.moe_expert / count).saturating_mul(used),
            _ => split.moe_expert,
        }
    } else {
        0
    };
    let union = expert_union_fraction(artifact, PLAN_WORKLOAD_PROMPT_TOKENS);
    let prefill_expert_reads = (split.moe_expert as f64 * union) as u64;

    // Without geometry we cannot price state; reject with context guidance.
    if geometry_missing {
        plan.mode = ExecutionMode::Rejected;
        plan.performance_class = LatencyClass::Infeasible;
        plan.engine = required_kernel_plan(adapter, kernels);
        plan.reason = format!(
            "artifact geometry (layers/heads/context) was not resolved, so KV state cannot be priced; adapter {} requires geometry resolution at inspect time",
            adapter.id
        );
        plan.rejection = Some(PlanRejection {
            code: ErrorCode::UnsupportedArchitecture.as_str().to_string(),
            message: plan.reason.clone(),
        });
        plan.plan_id = plan.compute_id()?;
        plan.validate()?;
        return Ok(plan);
    }
    let kv_bytes = kv_bytes.expect("checked above");

    // Mode candidates with feasibility by the strict R + U + H <= B rule.
    let estimate_inputs = EstimateInputs {
        split,
        active_weight_bytes_per_token: active,
        streamed_expert_bytes_per_token: expert_per_token,
        prefill_expert_read_bytes: prefill_expert_reads,
        kv_bytes_per_token: kv_bytes
            .checked_div(request.context.total_context)
            .unwrap_or(0),
        prompt_tokens: PLAN_WORKLOAD_PROMPT_TOKENS,
    };

    let mut estimates: Vec<ScheduleEstimate> = Vec::new();
    let mut budgets: Vec<(ExecutionMode, PlanBudget)> = Vec::new();

    // Resident.
    {
        let res = reserve_resident(&split, kv_bytes, policy);
        let budget = make_plan_budget(request.requested_budget, overhead, safety_reserve, &res);
        let fits = budget.fits();
        let reason = if fits {
            format!(
                "resident: weights {} + state {} + workspaces fit within budget {} with reserve",
                Bytes(split.total),
                Bytes(kv_bytes),
                request.requested_budget
            )
        } else {
            format!(
                "does not fit: needs {} against budget {} (over by {})",
                budget.admitted_total(),
                request.requested_budget,
                Bytes(
                    reservation_total(&res)
                        .saturating_add(policy.overhead_estimate_bytes)
                        .saturating_add(safety_reserve.bytes())
                        .saturating_sub(request.requested_budget.bytes())
                )
            )
        };
        estimates.push(estimate_resident(
            &estimate_inputs,
            fits,
            reason,
            policy,
            fits,
        ));
        budgets.push((ExecutionMode::Resident, budget));
    }

    // Streamed MoE (native routing; requires the adapter and expert geometry).
    if adapter.traits.moe && split.moe_expert > 0 {
        let mut chosen_cache = policy.expert_cache_default_bytes.min(split.moe_expert);
        let mut res = reserve_streamed_moe(&split, kv_bytes, chosen_cache, policy);
        let mut budget = make_plan_budget(request.requested_budget, overhead, safety_reserve, &res);
        if !budget.fits() {
            // Degrade the cache down to the floor before rejecting.
            chosen_cache = policy.expert_cache_floor_bytes.min(split.moe_expert);
            res = reserve_streamed_moe(&split, kv_bytes, chosen_cache, policy);
            budget = make_plan_budget(request.requested_budget, overhead, safety_reserve, &res);
        }
        let fits = budget.fits();
        let reason = if fits {
            format!(
                "streamed MoE: non-expert weights {} resident + bounded expert cache {} while required experts stream",
                Bytes(split.non_expert),
                Bytes(chosen_cache)
            )
        } else {
            format!(
                "does not fit even at the expert-cache floor: needs {} against budget {}",
                budget.admitted_total(),
                request.requested_budget
            )
        };
        estimates.push(estimate_streamed_moe(
            &estimate_inputs,
            fits,
            reason,
            policy,
        ));
        budgets.push((ExecutionMode::StreamedMoe, budget));
    } else if adapter.traits.moe {
        // MoE adapter but no expert tensors detected: adapter/artifact mismatch.
        plan.reason = format!(
            "adapter {} declares MoE routing but the artifact exposes no expert tensors",
            adapter.id
        );
        plan.mode = ExecutionMode::Rejected;
        plan.performance_class = LatencyClass::Infeasible;
        plan.engine = required_kernel_plan(adapter, kernels);
        plan.rejection = Some(PlanRejection {
            code: ErrorCode::InvalidArtifact.as_str().to_string(),
            message: plan.reason.clone(),
        });
        plan.plan_id = plan.compute_id()?;
        plan.validate()?;
        return Ok(plan);
    }

    // Streamed dense (P1 capacity path).
    {
        let res = reserve_streamed_dense(&split, kv_bytes, policy);
        let budget = make_plan_budget(request.requested_budget, overhead, safety_reserve, &res);
        let fits = budget.fits();
        let reason = if fits {
            format!(
                "streamed dense: resident core {} with tiled streaming of {}",
                Bytes(split.resident_core),
                Bytes(split.streamed_tiles)
            )
        } else {
            format!(
                "does not fit: needs {} against budget {}",
                budget.admitted_total(),
                request.requested_budget
            )
        };
        estimates.push(estimate_streamed_dense(
            &estimate_inputs,
            fits,
            reason,
            policy,
        ));
        budgets.push((ExecutionMode::StreamedDense, budget));
    }

    let chosen = choose_schedule(&estimates, 0.15).cloned();
    let engine_plan = required_kernel_plan(adapter, kernels);

    match chosen {
        Some(est) if est.feasible => {
            let (_, budget) = budgets
                .iter()
                .find(|(m, _)| *m == est.mode)
                .expect("chosen mode has a budget");
            plan.mode = est.mode;
            plan.budget = budget.clone();
            plan.cache_policy = match est.mode {
                ExecutionMode::Resident => CachePolicy {
                    policy: "none".into(),
                    expert_cache_bytes: Bytes::ZERO,
                    protect_in_flight: true,
                    prefetch: false,
                },
                ExecutionMode::StreamedMoe => CachePolicy {
                    policy: "bounded-explicit".into(),
                    expert_cache_bytes: Bytes(budget.reservation_breakdown.expert_cache.bytes()),
                    protect_in_flight: true,
                    prefetch: false,
                },
                _ => CachePolicy {
                    policy: "mixed".into(),
                    expert_cache_bytes: Bytes::ZERO,
                    protect_in_flight: true,
                    prefetch: false,
                },
            };
            plan.engine = engine_plan;
            plan.reason = est.reason.clone();

            // Confidence: upgraded only by exact matching measured evidence.
            let workload = workload_assumption_string();
            plan.assumptions.push(workload);
            match calibration.lookup(
                &device.device_id,
                &artifact.artifact_id,
                &adapter.id,
                "reference",
                tuckmill_manifest::BACKEND_ABI_VERSION,
                tuckmill_manifest::WORKER_PROTOCOL_VERSION,
                PLAN_WORKLOAD_SUITE,
                request.context.total_context,
            ) {
                Some(entry) => {
                    plan.confidence = Confidence::LocallyCalibrated;
                    plan.performance_class = latency_class(entry.ttft_median_ms, entry.tps_median);
                    plan.assumptions.push(format!(
                        "latency from {}-sample local calibration recorded {} (device/artifact matched)",
                        entry.samples,
                        tuckmill_manifest::ids::iso8601(entry.recorded_unix_ms)
                    ));
                }
                None => {
                    plan.confidence = Confidence::Estimated;
                    plan.performance_class = latency_class(est.ttft_ms.mid, est.tps.mid);
                    plan.assumptions.push(
                        "no matching local calibration; latency class derived from the deterministic cost model"
                            .to_string(),
                    );
                }
            }
            push_standard_assumptions(
                &mut plan,
                policy,
                &est,
                artifact,
                device,
                request.kv_bytes_per_element,
            );
        }
        _ => {
            // Nothing fits. Reject with the most informative typed reason.
            plan.mode = ExecutionMode::Rejected;
            plan.performance_class = LatencyClass::Infeasible;
            plan.engine = engine_plan;
            let tiny = split.resident_core
                + kv_bytes
                + policy.workspaces_resident_bytes
                + policy.io_buffers_resident_bytes
                + policy.overhead_estimate_bytes
                + safety_reserve.bytes();
            let message = format!(
                "minimum live state does not fit: even the smallest resident core plus state and overhead needs {} against budget {}",
                Bytes(tiny),
                request.requested_budget
            );
            let code = if kv_bytes.saturating_add(policy.overhead_estimate_bytes)
                > request.requested_budget.bytes()
            {
                ErrorCode::ContextExceedsBudget
            } else {
                ErrorCode::InvalidPlan
            };
            plan.reason = message.clone();
            plan.rejection = Some(PlanRejection {
                code: code.as_str().to_string(),
                message: format!(
                    "{message}. Lower --context or raise --memory-budget; alternatives are presented separately."
                ),
            });
            plan.assumptions.push(workload_assumption_string());
        }
    }

    // Adapter registry changes invalidate cached plans.
    let adapter_note = format!(
        "adapter registry or kernel gates changed (adapter {} required kernels {})",
        adapter.id,
        adapter.required_kernels.join(",")
    );
    plan.invalidation_conditions.push(adapter_note);
    if plan.engine.availability == EngineAvailability::ReferenceTestOnly {
        plan.assumptions.push(
            "no certified executor for this profile in this build; reference test engine produces synthetic output only and executes no model weights"
                .to_string(),
        );
    }

    plan.plan_id = plan.compute_id()?;
    plan.validate()?;
    Ok(plan)
}

fn workload_assumption_string() -> String {
    format!(
        "latency projection describes the standard workload shape ({} input / {} output tokens, suite {PLAN_WORKLOAD_SUITE}); cold-start reported separately",
        PLAN_WORKLOAD_PROMPT_TOKENS, PLAN_WORKLOAD_OUTPUT_TOKENS
    )
}

fn push_standard_assumptions(
    plan: &mut ExecutionPlan,
    policy: &PlannerPolicy,
    est: &ScheduleEstimate,
    artifact: &ModelArtifact,
    device: &DeviceProfile,
    kv_bytes_per_element: u32,
) {
    plan.assumptions.push(format!(
        "budget arithmetic: reservation {} + overhead estimate {} + safety reserve {} <= {}",
        plan.budget.reservation,
        plan.budget.overhead_estimate,
        plan.budget.safety_reserve,
        plan.budget.requested_budget
    ));
    plan.assumptions.push(format!(
        "bandwidth bands: memory {:.0}-{:.0} GB/s, storage {:.1}-{:.1} GB/s, efficiency {:.2} (assumptions, not measurements)",
        policy.memory_bandwidth_low / 1e9,
        policy.memory_bandwidth_high / 1e9,
        policy.ssd_read_bandwidth_low / 1e9,
        policy.ssd_read_bandwidth_high / 1e9,
        policy.bandwidth_efficiency
    ));
    plan.assumptions.push(format!(
        "projection: ttft {:.1}–{:.1} ms, throughput {:.1}–{:.1} tokens/s (mid {:.1}); logical reads only, not physical SSD traffic",
        est.ttft_ms.low, est.ttft_ms.high, est.tps.low, est.tps.high, est.tps.mid
    ));
    plan.assumptions.push(format!(
        "KV state priced at {} bytes per element (the resolved backend storage representation, not a nominal precision) for {} total tokens",
        kv_bytes_per_element, plan.context.total_context
    ));
    plan.assumptions.push(
        "no speculation and no mandatory draft; a draft is enabled only after measured benefit"
            .to_string(),
    );
    plan.assumptions.push(format!(
        "GPU family estimate {} sourced from: {}",
        device.gpu.family_estimate, device.gpu.family_source
    ));
    plan.assumptions.push(format!(
        "artifact {} ({:?}), quantization {}",
        artifact.file_name,
        artifact.format,
        artifact
            .quant_label
            .clone()
            .unwrap_or_else(|| "unknown".into())
    ));
    plan.assumptions
        .push("managed admission policy, not an OS guarantee; process footprint is reported separately from reservations".to_string());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{mock_artifact, mock_device, mock_geometry};
    use tuckmill_manifest::adapter::AdapterRegistry;
    use tuckmill_manifest::plan::EngineAvailability;

    fn registry() -> AdapterRegistry {
        AdapterRegistry::builtin().unwrap()
    }
    fn kernels() -> KernelRegistry {
        KernelRegistry::builtin().unwrap()
    }

    fn request(budget: &str, context: u64, output: u64) -> PlanRequest {
        PlanRequest {
            context: PlanContext::new(context, output),
            requested_budget: Bytes::parse(budget).unwrap(),
            policy: PlannerPolicy::default(),
            kv_bytes_per_element: 2,
        }
    }

    #[test]
    fn small_dense_fits_resident_on_2gib() {
        // 0.6B-class Q4: ~400MB weights.
        let artifact = mock_artifact("qwen3", 400 * 1024 * 1024, 0, Some(mock_geometry(true)));
        let adapter = registry().find("qwen3").unwrap().clone();
        let device = mock_device(8 * 1024 * 1024 * 1024);
        let plan = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device,
            &request("2GiB", 2048, 256),
            &CalibrationStore::default(),
        )
        .unwrap();
        assert_eq!(plan.mode, ExecutionMode::Resident);
        assert!(plan.budget.fits());
        assert_eq!(plan.confidence, Confidence::Estimated);
        assert_eq!(
            plan.engine.availability,
            EngineAvailability::ReferenceTestOnly
        );
        assert!(!plan.engine.required_kernels.is_empty());
        plan.validate().unwrap();
    }

    #[test]
    fn giant_dense_rejects_at_2gib_and_succeeds_at_larger_budget() {
        // 27B-class Q4: ~15GiB weights, cannot fit any 2GiB plan.
        let artifact = mock_artifact(
            "qwen3",
            15 * 1024 * 1024 * 1024,
            0,
            Some(mock_geometry(true)),
        );
        let adapter = registry().find("qwen3").unwrap().clone();
        let device = mock_device(8 * 1024 * 1024 * 1024);
        let rejected = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device,
            &request("2GiB", 1024, 128),
            &CalibrationStore::default(),
        )
        .unwrap();
        assert_eq!(rejected.mode, ExecutionMode::Rejected);
        assert_eq!(rejected.performance_class, LatencyClass::Infeasible);
        assert!(rejected.rejection.is_some());

        // Streamed dense fits when resident core + state + overhead fit.
        let streamed = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device,
            &request("2GiB", 1024, 128),
            &CalibrationStore::default(),
        )
        .unwrap();
        // The mock's resident core is 45% of weights -> still too large at 2GiB.
        assert_eq!(streamed.mode, ExecutionMode::Rejected);

        let device_big = mock_device(64 * 1024 * 1024 * 1024);
        let resident_big = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device_big,
            &request("24GiB", 2048, 256),
            &CalibrationStore::default(),
        )
        .unwrap();
        assert_eq!(resident_big.mode, ExecutionMode::Resident);
    }

    #[test]
    fn moe_streams_experts_under_small_budget() {
        // MoE with 2.5GiB experts and a small non-expert core.
        let artifact = mock_artifact(
            "qwen3moe",
            3 * 1024 * 1024 * 1024,
            2_560 * 1024 * 1024,
            Some(mock_geometry(false)),
        );
        let adapter = registry().find("qwen3moe").unwrap().clone();
        let device = mock_device(8 * 1024 * 1024 * 1024);
        let plan = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device,
            &request("2GiB", 2048, 256),
            &CalibrationStore::default(),
        )
        .unwrap();
        assert_eq!(plan.mode, ExecutionMode::StreamedMoe);
        assert!(plan.budget.fits());
        assert!(plan.budget.reservation_breakdown.expert_cache.bytes() > 0);
        assert_eq!(plan.cache_policy.policy, "bounded-explicit");
        assert!(plan.cache_policy.protect_in_flight);
    }

    #[test]
    fn context_beyond_advertised_is_rejected() {
        let artifact = mock_artifact("qwen3", 400 * 1024 * 1024, 0, Some(mock_geometry(true)));
        let adapter = registry().find("qwen3").unwrap().clone();
        let device = mock_device(8 * 1024 * 1024 * 1024);
        let err = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device,
            &request("2GiB", 8192, 256),
            &CalibrationStore::default(),
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::ContextExceedsBudget);
    }

    #[test]
    fn kv_pricing_follows_the_resolved_representation() {
        // The same request must be priced from the backend's actual KV
        // representation: 4 bytes per element (TuckCore's f32 store) must
        // reserve twice the bytes of a 2-byte (f16) assumption and can flip
        // admission. This is the corrected Gate B memory model.
        let artifact = mock_artifact("qwen3", 400 * 1024 * 1024, 0, Some(mock_geometry(true)));
        let adapter = registry().find("qwen3").unwrap().clone();
        let device = mock_device(8 * 1024 * 1024 * 1024);
        let mut two = request("1400MiB", 4096, 256);
        two.kv_bytes_per_element = 2;
        let plan_two = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device,
            &two,
            &CalibrationStore::default(),
        )
        .unwrap();
        let mut four = two.clone();
        four.kv_bytes_per_element = 4;
        let plan_four = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device,
            &four,
            &CalibrationStore::default(),
        )
        .unwrap();

        // 28 layers x 8 kv heads x 256 (k+v) x 4096 tokens.
        let per_token = 28u64 * 8 * 256;
        assert_ne!(plan_two.mode, ExecutionMode::Rejected);
        assert_eq!(
            plan_two.budget.reservation_breakdown.kv_state.bytes(),
            per_token * 4096 * 2
        );
        assert_eq!(
            plan_four.mode,
            ExecutionMode::Rejected,
            "4-byte KV pricing must reject what 2-byte pricing admitted"
        );
        assert!(plan_four.rejection.is_some());
        assert!(plan_two
            .assumptions
            .iter()
            .any(|a| a.contains("resolved backend storage representation")));
    }

    #[test]
    fn budget_beyond_physical_memory_is_invalid() {
        let artifact = mock_artifact("qwen3", 400 * 1024 * 1024, 0, Some(mock_geometry(true)));
        let adapter = registry().find("qwen3").unwrap().clone();
        let device = mock_device(8 * 1024 * 1024 * 1024);
        let err = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device,
            &request("16GiB", 1024, 128),
            &CalibrationStore::default(),
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidRequest);
    }

    #[test]
    fn arch_mismatch_is_unsupported() {
        let artifact = mock_artifact("qwen3", 400 * 1024 * 1024, 0, Some(mock_geometry(true)));
        let adapter = registry().find("qwen3moe").unwrap().clone();
        let device = mock_device(8 * 1024 * 1024 * 1024);
        let err = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device,
            &request("2GiB", 1024, 128),
            &CalibrationStore::default(),
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::UnsupportedArchitecture);
    }

    #[test]
    fn missing_geometry_rejects_with_guidance() {
        let artifact = mock_artifact("qwen3", 400 * 1024 * 1024, 0, None);
        let adapter = registry().find("qwen3").unwrap().clone();
        let device = mock_device(8 * 1024 * 1024 * 1024);
        let plan = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device,
            &request("2GiB", 1024, 128),
            &CalibrationStore::default(),
        )
        .unwrap();
        assert_eq!(plan.mode, ExecutionMode::Rejected);
        assert!(plan.rejection.unwrap().message.contains("geometry"));
    }

    #[test]
    fn plan_records_assumptions_and_invalidation() {
        let artifact = mock_artifact("qwen3", 400 * 1024 * 1024, 0, Some(mock_geometry(true)));
        let adapter = registry().find("qwen3").unwrap().clone();
        let device = mock_device(8 * 1024 * 1024 * 1024);
        let plan = build_plan(
            &kernels(),
            &artifact,
            &adapter,
            &device,
            &request("2GiB", 2048, 256),
            &CalibrationStore::default(),
        )
        .unwrap();
        assert!(plan
            .assumptions
            .iter()
            .any(|a| a.contains("logical reads only")));
        assert!(plan
            .invalidation_conditions
            .iter()
            .any(|c| c.contains("backend or worker ABI")));
        assert!(plan
            .assumptions
            .iter()
            .any(|a| a.contains("synthetic output only")));
    }
}
