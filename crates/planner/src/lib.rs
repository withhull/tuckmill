//! Tuckmill planner: feasibility, memory admission, and the deterministic
//! cost model.
//!
//! The planner combines artifact metadata, adapter capabilities, device
//! measurements, context requirements, and resource limits into a versioned,
//! inspectable execution plan. It begins with static feasibility and a simple
//! deterministic cost model; measured calibration comes only from recorded
//! evidence and upgrades a plan's confidence label, never its honesty
//! (PRD §13, §5).

pub mod budget;
pub mod calibration;
pub mod cost;
pub mod device;
pub mod kernel_registry;
pub mod kv;
pub mod plan;

pub use plan::{build_plan, PlanRequest};

/// Documented deterministic assumptions behind every estimate. Changing a
/// value here changes plan output; the plan records which values were used so
/// a projection is never presented as a measurement.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlannerPolicy {
    /// Safety reserve H as a fraction of the requested budget B.
    pub safety_reserve_ratio: f64,
    /// Floor for H so tiny budgets still get headroom.
    pub safety_reserve_floor_bytes: u64,
    /// Estimated unavoidable runtime overhead U (supervisor + worker +
    /// framework state). An assumption to be calibrated (G0).
    pub overhead_estimate_bytes: u64,
    /// Workspace allowance for resident execution.
    pub workspaces_resident_bytes: u64,
    /// Workspace allowance for streamed execution.
    pub workspaces_streamed_bytes: u64,
    /// Small I/O allowance for resident execution.
    pub io_buffers_resident_bytes: u64,
    /// Bounded read buffers for streamed execution.
    pub io_buffers_streamed_bytes: u64,
    /// Default bounded expert cache allowance for MoE streaming.
    pub expert_cache_default_bytes: u64,
    /// Minimum expert cache allowance before rejecting a streamed plan.
    pub expert_cache_floor_bytes: u64,
    /// Candidate read sizes (bytes) for streamed reads (E02 experiment).
    pub read_buffer_candidates_bytes: [u64; 3],
    /// Effective SSD read bandwidth band (bytes/second). Scenario
    /// assumptions, not measurements.
    pub ssd_read_bandwidth_low: f64,
    pub ssd_read_bandwidth_mid: f64,
    pub ssd_read_bandwidth_high: f64,
    /// Unified memory bandwidth band (bytes/second) for resident decode.
    pub memory_bandwidth_low: f64,
    pub memory_bandwidth_mid: f64,
    pub memory_bandwidth_high: f64,
    /// Fraction of peak bandwidth convertible into token throughput.
    pub bandwidth_efficiency: f64,
    /// Effective prefill compute band (FLOPs/second).
    pub prefill_flops_low: f64,
    pub prefill_flops_mid: f64,
    pub prefill_flops_high: f64,
    /// Fallback KV element size in bytes for engines whose resolved storage
    /// representation the runtime cannot determine (the reference test
    /// engine allocates no KV state at all). Real backends are priced from
    /// their actual resolved representation via
    /// `PlanRequest::kv_bytes_per_element` — never from a nominal precision.
    pub kv_bytes_per_element: u32,
    /// Speculation is off by default; no mandatory draft (PRD §11.3, §12).
    pub speculation_enabled: bool,
    /// Unsupported combination guard: refuse speculation with hybrid state
    /// until rollback tests pass.
    pub speculation_allowed_for_hybrid: bool,
}

impl Default for PlannerPolicy {
    fn default() -> Self {
        Self {
            safety_reserve_ratio: 0.125,
            safety_reserve_floor_bytes: 192 * 1024 * 1024,
            overhead_estimate_bytes: 192 * 1024 * 1024,
            workspaces_resident_bytes: 192 * 1024 * 1024,
            workspaces_streamed_bytes: 128 * 1024 * 1024,
            io_buffers_resident_bytes: 16 * 1024 * 1024,
            io_buffers_streamed_bytes: 64 * 1024 * 1024,
            expert_cache_default_bytes: 512 * 1024 * 1024,
            expert_cache_floor_bytes: 128 * 1024 * 1024,
            read_buffer_candidates_bytes: [1 << 20, 4 << 20, 16 << 20],
            ssd_read_bandwidth_low: 1.5e9,
            ssd_read_bandwidth_mid: 2.5e9,
            ssd_read_bandwidth_high: 4.0e9,
            memory_bandwidth_low: 60e9,
            memory_bandwidth_mid: 80e9,
            memory_bandwidth_high: 100e9,
            bandwidth_efficiency: 0.6,
            prefill_flops_low: 1.0e12,
            prefill_flops_mid: 2.0e12,
            prefill_flops_high: 3.5e12,
            kv_bytes_per_element: 2,
            speculation_enabled: false,
            speculation_allowed_for_hybrid: false,
        }
    }
}

impl PlannerPolicy {
    pub fn safety_reserve(&self, budget: u64) -> u64 {
        let by_ratio = (budget as f64 * self.safety_reserve_ratio).round() as u64;
        by_ratio.max(self.safety_reserve_floor_bytes)
    }
}

/// Test-only construction helpers shared by other crates' test suites.
pub mod testing {
    use tuckmill_manifest::records::*;
    use tuckmill_manifest::Id;

    /// Deterministic mock device for unit tests.
    pub fn mock_device(memory_bytes: u64) -> DeviceProfile {
        let mut capabilities = std::collections::BTreeMap::new();
        capabilities.insert(
            "metal".to_string(),
            CapabilityStatus::unknown("mock device"),
        );
        DeviceProfile {
            schema_version: 1,
            device_id: Id::new("d".repeat(64)).unwrap(),
            sampled_unix_ms: 0,
            chip: "Apple M-test".into(),
            cpu_cores_total: 8,
            cpu_cores_performance: Some(4),
            cpu_cores_efficiency: Some(4),
            memory_bytes,
            os_name: "macOS".into(),
            os_version: "27.0.1".into(),
            os_build: "TEST".into(),
            gpu: GpuProfile {
                family_estimate: "apple9".into(),
                family_source: "mock".into(),
                recommended_working_set_bytes: None,
            },
            storage: StorageProfile {
                path_class: "internal-ssd".into(),
                volume_identity: Id::new("e".repeat(64)).unwrap(),
                free_bytes: 100 * 1024 * 1024 * 1024,
                total_bytes: 460 * 1024 * 1024 * 1024,
            },
            power: PowerProfile {
                on_battery: Some(false),
                low_power_mode: Some(false),
            },
            thermal_start: "nominal".into(),
            pressure: "normal".into(),
            capabilities,
        }
    }

    /// Deterministic mock artifact with a compressed-style byte split.
    pub fn mock_artifact(
        arch: &str,
        total_bytes: u64,
        expert_bytes: u64,
        geometry: Option<ModelGeometry>,
    ) -> ModelArtifact {
        use std::collections::BTreeMap;
        let non_expert = total_bytes.saturating_sub(expert_bytes);
        let mut by_class = BTreeMap::new();
        let embedding = non_expert / 5;
        let attention = non_expert / 3;
        let norms = non_expert / 20;
        let output = non_expert / 5;
        by_class.insert("token_embedding".to_string(), embedding);
        by_class.insert("attention".to_string(), attention);
        by_class.insert("norm".to_string(), norms);
        by_class.insert("attn_norm".to_string(), norms);
        by_class.insert("output_head".to_string(), output);
        let ffn_like = non_expert.saturating_sub(embedding + attention + norms * 2 + output);
        by_class.insert("ffn".to_string(), ffn_like);
        if expert_bytes > 0 {
            by_class.insert("moe_expert".to_string(), expert_bytes);
        }
        let artifact_id = Id::new(tuckmill_manifest::ids::sha256_hex(
            format!("{arch}:{total_bytes}:{expert_bytes}").as_bytes(),
        ))
        .unwrap();
        ModelArtifact {
            schema_version: 1,
            artifact_id,
            created_unix_ms: 0,
            format: ModelFormat::Gguf,
            source: ArtifactSource {
                uri: "test://mock".into(),
                revision: None,
                license: None,
            },
            file_name: format!("{arch}-mock.gguf"),
            file_bytes: total_bytes.saturating_add(1 << 20),
            content_hash: Id::new("c".repeat(64)).unwrap(),
            tensor_index_hash: Id::new("f".repeat(64)).unwrap(),
            metadata_hash: Id::new("1".repeat(64)).unwrap(),
            tokenizer_hash: None,
            arch_id: Some(arch.to_string()),
            model_name: Some(format!("{arch} mock")),
            advertised_context_length: Some(4096),
            geometry,
            tensor_stats: TensorStats {
                total_tensors: 20,
                total_bytes,
                by_class,
                by_type: BTreeMap::new(),
            },
            quant_label: Some("Q4_K_M".into()),
            packed: None,
        }
    }

    pub fn mock_geometry(dense: bool) -> ModelGeometry {
        ModelGeometry {
            layers: 28,
            head_count: 16,
            kv_head_count: 8,
            key_length: 128,
            value_length: 128,
            expert_count: if dense { None } else { Some(128) },
            expert_used_per_token: if dense { None } else { Some(8) },
        }
    }
}
