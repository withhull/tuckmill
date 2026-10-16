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
                if self != &Self::default() {
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
            }
            EngineKind::LlamaCpp => {
                if let Some(v) = self.gpu_layers {
                    obj.insert("gpu_layers".into(), Value::from(v));
                }
                if let Some(v) = &self.kv_precision {
                    obj.insert("kv_precision".into(), Value::from(v.clone()));
                }
            }
            EngineKind::ReferenceTest => {}
        }
        if obj.is_empty() {
            None
        } else {
            Some(Value::Object(obj))
        }
    }

    /// Per-side options for verification. `cache_override` supplies the
    /// cache geometry for the side (the streamed side is deliberately tiny);
    /// the block size follows the cache size as before.
    pub fn side_load_json(&self, kind: EngineKind, cache_override: Option<u64>) -> Option<Value> {
        let mut opts = self.clone();
        if let Some(cache) = cache_override {
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
