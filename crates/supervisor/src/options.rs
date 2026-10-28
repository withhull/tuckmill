//! Typed execution options: one resolver shared by `run`, `bench`, `stats`,
//! and `verify`. Requested values are resolved once and used to build every
//! worker load request; the worker's *acknowledged* resolved settings come
//! back through the load identity and stats, are compared against the
//! request, and any unexpected substitution fails closed — a run must never
//! be recorded as a configuration it did not execute.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use tuckmill_backend::{BackendStats, EngineKind};
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_tuckcore::{KernelSelection, StorageSelection};

/// Requested execution options for one backend worker. `None` means "engine
/// default"; `Auto` means "engine default resolution, recorded afterwards".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernels: Option<KernelSelection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<StorageSelection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_cache_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numerical_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_precision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<bool>,
    /// Test aid: a per-token delay for the reference test engine, used by
    /// pipeline/lifecycle tests that need a slow producer (long streaming,
    /// governor commands mid-run). Refused for real engines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_token_delay_ms: Option<u64>,
    /// TuckCore engine prefix cache (ADR-012 step 2; engine default on).
    /// Benchmarks turn it off so repeated prompts never become hidden
    /// cache hits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_cache: Option<bool>,
    /// TuckCore speculative greedy decoding with the artifact's MTP head
    /// (ADR-012 step 3; engine default off). Greedy output is
    /// token-identical to plain decoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative: Option<bool>,
}

impl ExecutionOptions {
    /// Reject options that do not apply to the selected engine — never
    /// silently drop them.
    pub fn validate(&self, kind: EngineKind) -> Result<()> {
        match kind {
            EngineKind::TuckCore => {
                if self.gpu_layers.is_some() {
                    return Err(TuckmillError::new(
                        ErrorCode::InvalidRequest,
                        "--gpu-layers applies only to --engine llama (the Metal performance baseline); arithmetic diagnostics stay CPU",
                    ));
                }
                if self.per_token_delay_ms.is_some() {
                    return Err(TuckmillError::new(
                        ErrorCode::InvalidRequest,
                        "per-token delay is a reference-test-engine test aid; real engines never simulate slowness",
                    ));
                }
                Ok(())
            }
            EngineKind::LlamaCpp => {
                if self.kernels.is_some() || self.storage.is_some() {
                    return Err(TuckmillError::new(
                        ErrorCode::InvalidRequest,
                        "--kernels/--storage apply to the tuckcore engine; llama.cpp selects its own kernels",
                    ));
                }
                if self.numerical_profile.is_some() {
                    return Err(TuckmillError::new(
                        ErrorCode::InvalidRequest,
                        "--tuckcore-profile applies to the tuckcore engine",
                    ));
                }
                if self.profile.is_some() {
                    return Err(TuckmillError::new(
                        ErrorCode::InvalidRequest,
                        "phase profile counters apply to the tuckcore engine",
                    ));
                }
                if self.prefix_cache.is_some() {
                    return Err(TuckmillError::new(
                        ErrorCode::InvalidRequest,
                        "the prefix cache option applies to the tuckcore engine",
                    ));
                }
                if self.speculative.is_some() {
                    return Err(TuckmillError::new(
                        ErrorCode::InvalidRequest,
                        "--speculative applies to the tuckcore engine (MTP speculative decoding)",
                    ));
                }
                if let Some(n) = self.gpu_layers {
                    if n < 0 {
                        return Err(TuckmillError::new(
                            ErrorCode::InvalidRequest,
                            "--gpu-layers must be >= 0",
                        ));
                    }
                }
                Ok(())
            }
            EngineKind::ReferenceTest => {
                let mut without_test_aid = self.clone();
                without_test_aid.per_token_delay_ms = None;
                if without_test_aid != Self::default() {
                    return Err(TuckmillError::new(
                        ErrorCode::InvalidRequest,
                        "the reference test engine executes synthetic work and accepts no execution options",
                    ));
                }
                Ok(())
            }
        }
    }

    /// Backend load-options JSON for a normal load (run/bench/stats/serve).
    /// `validate` must have accepted the pair first; the shaping here drops
    /// nothing silently — inapplicable fields were already rejected.
    pub fn load_json(&self, kind: EngineKind) -> Option<Value> {
        let mut obj = Map::new();
        match kind {
            EngineKind::TuckCore => {
                if let Some(v) = self.weight_cache_bytes {
                    obj.insert("weight_cache_bytes".into(), Value::from(v));
                }
                if let Some(v) = self.block_bytes {
                    obj.insert("block_bytes".into(), Value::from(v));
                }
                if let Some(v) = &self.numerical_profile {
                    obj.insert("numerical_profile".into(), Value::from(v.clone()));
                }
                if let Some(v) = &self.kv_precision {
                    obj.insert("kv_precision".into(), Value::from(v.clone()));
                }
                if let Some(v) = self.kernels {
                    obj.insert("kernels".into(), serde_json::to_value(v).ok()?);
                }
                if let Some(v) = self.storage {
                    obj.insert("storage".into(), serde_json::to_value(v).ok()?);
                }
                if let Some(v) = self.profile {
                    obj.insert("profile".into(), Value::from(v));
                }
                if let Some(v) = self.prefix_cache {
                    obj.insert("prefix_cache".into(), Value::from(v));
                }
                if let Some(v) = self.speculative {
                    obj.insert("speculative".into(), Value::from(v));
                }
            }
            EngineKind::LlamaCpp => {
                if let Some(v) = self.gpu_layers {
                    obj.insert("gpu_layers".into(), Value::from(v));
                }
                if let Some(v) = &self.kv_precision {
                    obj.insert("kv_precision".into(), Value::from(v.clone()));
                }
            }
            EngineKind::ReferenceTest => {
                if let Some(v) = self.per_token_delay_ms {
                    obj.insert("per_token_delay_ms".into(), Value::from(v));
                }
            }
        }
        if obj.is_empty() {
            None
        } else {
            Some(Value::Object(obj))
        }
    }

    /// Bytes per KV element in the engine's *actual* storage representation.
    /// This is the number the planner must price; a nominal precision is
    /// never used. The width is resolved by backend AND model family:
    ///
    /// - TuckCore + dense attention keeps f32 values even in its f16-rounded
    ///   mode, so it is always 4 bytes;
    /// - TuckCore + hybrid linear attention (qwen35) stores packed KV at the
    ///   declared precision — f16 by default, f32 when selected — so the
    ///   planner prices 2 or 4 exactly as the executor allocates;
    /// - llama.cpp honors its configured cache type (f16 default);
    /// - the reference test engine allocates no KV state and takes the
    ///   declared fallback assumption.
    pub fn kv_bytes_per_element(
        &self,
        kind: EngineKind,
        hybrid_linear_attention: bool,
        fallback: u32,
    ) -> u32 {
        match kind {
            EngineKind::TuckCore => {
                if hybrid_linear_attention {
                    match self.kv_precision.as_deref() {
                        Some("f32") => 4,
                        _ => 2,
                    }
                } else {
                    4
                }
            }
            EngineKind::LlamaCpp => match self.kv_precision.as_deref() {
                Some("f32") => 4,
                _ => 2,
            },
            EngineKind::ReferenceTest => fallback,
        }
    }

    /// Per-side options for verification. `cache_override` supplies the
    /// cache geometry for the side (the streamed side is deliberately tiny);
    /// the block size follows the cache size as before.
    pub fn side_load_json(&self, kind: EngineKind, cache_override: Option<u64>) -> Option<Value> {
        let mut opts = self.clone();
        // Cache geometry shapes CPU weight streaming only; the Metal path
        // keeps weights resident and refuses streaming options.
        let streams = !matches!(opts.kernels, Some(KernelSelection::Metal));
        if let (Some(cache), true) = (cache_override, streams) {
            opts.weight_cache_bytes = Some(cache);
            opts.block_bytes = Some(cache.clamp(64 * 1024, 4 * 1024 * 1024));
        }
        // Verification never offloads to Metal: the llama side is the CPU
        // arithmetic reference by construction.
        if kind == EngineKind::LlamaCpp {
            opts.gpu_layers = None;
        }
        opts.load_json(kind)
    }

    /// Compare the request against what the worker actually resolved.
    /// Explicit selections must match exactly; `Auto`/`None` is recorded,
    /// not constrained. Returns the acknowledged settings for the record.
    pub fn acknowledged(
        &self,
        kind: EngineKind,
        resolved_kernels: &[String],
        stats: &BackendStats,
    ) -> Result<AcknowledgedExecution> {
        let kernel_impl = stats.kernel_impl.clone().or_else(|| {
            resolved_kernels
                .iter()
                .filter_map(|k| k.rsplit(':').next().map(str::to_string))
                .next_back()
        });
        let ack = AcknowledgedExecution {
            kernel_impl,
            storage_mode: stats.storage.as_ref().map(|s| s.mode.clone()),
            numerical_profile: stats.numerical_profile.clone(),
            kv_precision: stats.kv_precision.clone(),
        };
        let mismatch = |what: &str, requested: &str, resolved: Option<&str>| {
            TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!(
                    "requested {what} {requested:?} but the worker resolved {}; refusing to record this run as the requested configuration",
                    resolved.unwrap_or("nothing")
                ),
            )
            .with_action(
                "Do not use this run as evidence for the requested configuration; report the discrepancy.",
            )
        };
        if let Some(sel) = self.kernels {
            match sel {
                KernelSelection::Auto => {}
                explicit => {
                    let want = match explicit {
                        KernelSelection::Scalar => "scalar",
                        KernelSelection::Neon => "neon",
                        KernelSelection::Metal => "metal",
                        KernelSelection::Auto => unreachable!(),
                    };
                    if ack.kernel_impl.as_deref() != Some(want) {
                        return Err(mismatch("kernels", want, ack.kernel_impl.as_deref()));
                    }
                }
            }
        }
        if let Some(sel) = self.storage {
            match sel {
                StorageSelection::Auto => {}
                explicit => {
                    let want = match explicit {
                        StorageSelection::Mapped => "mapped",
                        StorageSelection::Owned => "owned",
                        StorageSelection::Auto => unreachable!(),
                    };
                    if ack.storage_mode.as_deref() != Some(want) {
                        return Err(mismatch("storage", want, ack.storage_mode.as_deref()));
                    }
                }
            }
        }
        if let Some(profile) = &self.numerical_profile {
            if ack.numerical_profile.as_deref() != Some(profile.as_str()) {
                return Err(mismatch(
                    "numerical profile",
                    profile,
                    ack.numerical_profile.as_deref(),
                ));
            }
        }
        if let Some(kv) = &self.kv_precision {
            if kind != EngineKind::ReferenceTest && ack.kv_precision.as_deref() != Some(kv.as_str())
            {
                return Err(mismatch("kv precision", kv, ack.kv_precision.as_deref()));
            }
        }
        Ok(ack)
    }
}

/// What the worker acknowledged at load. Recorded separately from the
/// requested configuration so a report can never blur the two.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcknowledgedExecution {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_impl: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numerical_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_precision: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine prefills in fixed chunks (ADR-011); a requested prefill
    /// batch is an unknown option, refused rather than ignored.
    #[test]
    fn a_requested_prefill_batch_is_refused_as_unknown() {
        let parsed: std::result::Result<ExecutionOptions, _> =
            serde_json::from_value(serde_json::json!({"kernels": "metal", "prefill_batch": 4}));
        assert!(parsed.is_err());
    }
}

#[cfg(test)]
mod kv_width_tests {
    use super::*;

    #[test]
    fn kv_width_resolves_by_backend_and_model_family() {
        // Dense TuckCore keeps f32 values regardless of the nominal profile.
        let dense = ExecutionOptions::default();
        assert_eq!(
            dense.kv_bytes_per_element(EngineKind::TuckCore, false, 2),
            4
        );

        // Hybrid (qwen35) packed KV prices the declared precision.
        let default_hybrid = ExecutionOptions::default();
        assert_eq!(
            default_hybrid.kv_bytes_per_element(EngineKind::TuckCore, true, 2),
            2,
            "packed f16 is the default"
        );
        let f32_kv = ExecutionOptions {
            kv_precision: Some("f32".to_string()),
            ..Default::default()
        };
        assert_eq!(
            f32_kv.kv_bytes_per_element(EngineKind::TuckCore, true, 2),
            4
        );
        assert_eq!(
            f32_kv.kv_bytes_per_element(EngineKind::LlamaCpp, false, 2),
            4
        );

        // llama.cpp honors its cache type; the reference test engine takes
        // the declared fallback.
        let llama = ExecutionOptions::default();
        assert_eq!(
            llama.kv_bytes_per_element(EngineKind::LlamaCpp, false, 4),
            2
        );
        assert_eq!(
            llama.kv_bytes_per_element(EngineKind::ReferenceTest, true, 3),
            3
        );
    }
}
