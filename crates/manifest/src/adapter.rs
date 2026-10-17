//! Architecture adapter specifications.
//!
//! Adapters declare what an architecture needs: required metadata keys,
//! KV geometry sources, expert routing metadata, required kernel packages,
//! and required activation transforms. They never enable execution by
//! themselves; kernel gates and certifications decide availability.
//!
//! Built-in adapters are embedded from the repository `adapters/` directory
//! at compile time; `TUCKMILL_ADAPTERS_DIR` can supply experimental
//! overrides at runtime.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{ErrorCode, Result, TuckmillError};
use crate::gguf::{GgufInspection, GgufValue};
use crate::ids::hash_of;

/// Adapter schema version.
///
/// 2: KV geometry can express hybrid stacks — `attention` selects the
///    full-attention layers of a mixed stack and `recurrent` declares the
///    per-session state of the linear-attention layers (ADR-008). Files
///    that declare version 1 remain readable; only the new optional
///    fields change.
pub const ADAPTER_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterTraits {
    pub dense: bool,
    pub moe: bool,
    pub hybrid_linear_attention: bool,
    pub ternary_quantized: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationRecord {
    /// "unmeasured" until named tests pass on a named configuration.
    pub state: String,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetadataRequirement {
    pub key: String,
    /// One of: "u32", "u64", "f32", "string", "bool", "array", "any".
    pub kind: String,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvGeometrySpec {
    pub layers_key: String,
    pub head_count_key: String,
    pub kv_head_count_key: String,
    pub embedding_key: String,
    #[serde(default)]
    pub key_length_key: Option<String>,
    #[serde(default)]
    pub value_length_key: Option<String>,
    /// Full-attention layer selection for hybrid stacks. Absent = every
    /// served layer carries attention KV (dense attention).
    #[serde(default)]
    pub attention: Option<AttentionLayerSpec>,
    /// Per-session recurrent state of linear-attention layers. Absent =
    /// the stack carries no recurrent state.
    #[serde(default)]
    pub recurrent: Option<RecurrentStateSpec>,
}

/// Full-attention layer selection in a hybrid stack (ADR-008). Attention
/// KV is carried only by the selected layers; every other served layer
/// carries recurrent state instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionLayerSpec {
    /// Metadata key holding the 1-based interval: every Nth served layer
    /// is a full-attention layer.
    pub interval_key: String,
    /// Optional metadata key holding the number of trailing blocks that
    /// are not part of the served stack (for example multi-token
    /// prediction / `nextn` layers). The served stack is
    /// `layers_key - excluded_tail_key`.
    #[serde(default)]
    pub excluded_tail_key: Option<String>,
    /// Optional metadata key holding a per-layer boolean array with
    /// llama.cpp `attention.recurrent_layers` semantics (true = recurrent
    /// / linear-attention layer). When the artifact carries a full-length
    /// array it takes precedence over the interval because it states the
    /// same fact per layer.
    #[serde(default)]
    pub layers_key: Option<String>,
}

/// Per-session recurrent state of the linear-attention layers, sized from
/// the pinned ggml-org Qwen3.5-0.8B GGUF contract (`qwen35.ssm.*` keys;
/// shapes re-derived from llama.cpp b11695 `src/models/qwen35.cpp` and
/// registered in ADR-008):
///
/// - delta-rule state per layer: `groups x state_size x (inner_size /
///   groups)` elements (k-head x v-head matrix per value head);
/// - convolution history per layer: `(conv_kernel - 1)` samples of
///   `inner_size + 2 * groups * state_size` channels — the width of the
///   declared `ssm_conv1d` tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecurrentStateSpec {
    pub groups_key: String,
    pub state_size_key: String,
    pub inner_size_key: String,
    pub conv_kernel_key: String,
    /// Stored bytes per state element (Qwen3.5 declares `mamba_ssm_dtype`
    /// float32, so 4).
    pub element_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertMetaSpec {
    pub count_key: String,
    pub used_key: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterSpec {
    pub schema_version: u32,
    pub id: String,
    pub display_name: String,
    pub family: String,
    pub architecture: String,
    pub traits: AdapterTraits,
    pub status: String,
    pub certification: CertificationRecord,
    pub required_kernels: Vec<String>,
    /// Required activation transforms that are part of the execution contract
    /// (skipping one is corruption, not an optimization).
    #[serde(default)]
    pub required_transforms: Vec<String>,
    pub metadata_requirements: Vec<MetadataRequirement>,
    pub kv: KvGeometrySpec,
    #[serde(default)]
    pub expert_meta: Option<ExpertMetaSpec>,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Resolved KV geometry for a concrete artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvGeometry {
    /// Served stack layers: `layers_key` minus any trailing non-served
    /// blocks (for example MTP/`nextn` prediction layers).
    pub layers: u64,
    /// Served layers that carry per-request attention KV. Equal to
    /// `layers` for dense attention; smaller for hybrid stacks.
    pub attention_layers: u64,
    /// Constant per-session recurrent state (delta-rule state plus
    /// convolution history of the linear-attention layers). Zero for
    /// pure-attention stacks; independent of context length.
    pub state_bytes: u64,
    pub head_count: u64,
    pub kv_head_count: u64,
    /// Per-head dimension of K/V.
    pub head_dim: u64,
    pub key_length: u64,
    pub value_length: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertGeometry {
    pub count: u64,
    pub used_per_token: u64,
}

impl AdapterSpec {
    pub fn validate(&self) -> Result<()> {
        let err = |msg: String| TuckmillError::new(ErrorCode::InvalidConfiguration, msg);
        if self.schema_version != ADAPTER_SCHEMA_VERSION {
            return Err(err(format!(
                "adapter {} has schema version {} (want {})",
                self.id, self.schema_version, ADAPTER_SCHEMA_VERSION
            )));
        }
        if self.id.is_empty() || self.architecture.is_empty() {
            return Err(err("adapter id and architecture must be non-empty".into()));
        }
        for k in &self.required_kernels {
            if !is_kernel_id(k) {
                return Err(err(format!(
                    "adapter {} lists invalid kernel id {k:?} (expected K01-style)",
                    self.id
                )));
            }
        }
        for req in &self.metadata_requirements {
            if !matches!(
                req.kind.as_str(),
                "u32" | "u64" | "f32" | "string" | "bool" | "array" | "any"
            ) {
                return Err(err(format!(
                    "adapter {} requirement {} has unknown kind {:?}",
                    self.id, req.key, req.kind
                )));
            }
        }
        if let Some(a) = &self.kv.attention {
            if a.interval_key.is_empty() {
                return Err(err(format!(
                    "adapter {}: attention layer interval key must be non-empty",
                    self.id
                )));
            }
        }
        if let Some(r) = &self.kv.recurrent {
            if !matches!(r.element_bytes, 1 | 2 | 4) {
                return Err(err(format!(
                    "adapter {}: recurrent state element width {} is not supported",
                    self.id, r.element_bytes
                )));
            }
        }
        Ok(())
    }

    pub fn certification_state(&self) -> &str {
        &self.certification.state
    }

    /// Check required metadata presence and kinds against a real artifact.
    /// Missing required keys are an architecture-compatibility failure
    /// (FR-26); all problems are aggregated so the user sees one action.
    pub fn validate_metadata(&self, insp: &GgufInspection) -> Result<()> {
        let mut problems: Vec<String> = Vec::new();
        for req in &self.metadata_requirements {
            match insp.get(&req.key) {
                None => {
                    if req.required {
                        problems.push(format!("missing required key {:?}", req.key));
                    }
                }
                Some(value) => {
                    if !value_matches_kind(value, &req.kind) {
                        problems.push(format!(
                            "key {:?} has type {} but adapter requires {}",
                            req.key,
                            value.type_name(),
                            req.kind
                        ));
                    }
                }
            }
        }
        if problems.is_empty() {
            return Ok(());
        }
        Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "artifact does not satisfy adapter {}: {}",
                self.id,
                problems.join("; ")
            ),
        )
        .with_action(
            "Confirm the artifact family with 'tuckmill inspect'; if it is a different architecture, install or update its adapter.",
        ))
    }

    fn get_u64_required(&self, insp: &GgufInspection, key: &str) -> Result<u64> {
        insp.get_u64(key).filter(|v| *v > 0).ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "adapter {} requires positive integer metadata {key:?}",
                    self.id
                ),
            )
        })
    }

    pub fn kv_geometry(&self, insp: &GgufInspection) -> Result<KvGeometry> {
        let total_layers = self.get_u64_required(insp, &self.kv.layers_key)?;
        let excluded_tail = match self
            .kv
            .attention
            .as_ref()
            .and_then(|a| a.excluded_tail_key.as_deref())
        {
            Some(key) => insp.get_u64(key).unwrap_or(0),
            None => 0,
        };
        if excluded_tail >= total_layers {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "adapter {}: {excluded_tail} trailing non-served blocks leave no served stack out of {total_layers} blocks",
                    self.id
                ),
            ));
        }
        let layers = total_layers - excluded_tail;
        let attention_layers = match &self.kv.attention {
            None => layers,
            Some(spec) => {
                // A full-length per-layer boolean array, when the artifact
                // carries it, states the fact per layer and wins over the
                // interval. llama.cpp semantics: true = recurrent layer,
                // so attention layers are the remaining served entries.
                let recurrent_from_array =
                    spec.layers_key
                        .as_deref()
                        .and_then(|key| match insp.get(key) {
                            Some(GgufValue::Array(items)) if items.len() as u64 >= layers => Some(
                                items
                                    .iter()
                                    .take(layers as usize)
                                    .filter(|v| matches!(v, GgufValue::Bool(true)))
                                    .count() as u64,
                            ),
                            _ => None,
                        });
                match recurrent_from_array {
                    Some(recurrent) if recurrent < layers => layers - recurrent,
                    Some(recurrent) => {
                        return Err(TuckmillError::new(
                            ErrorCode::UnsupportedArchitecture,
                            format!(
                                "adapter {}: per-layer array marks all {recurrent} served layers as recurrent (no attention layer remains)",
                                self.id
                            ),
                        ));
                    }
                    None => {
                        let interval = self.get_u64_required(insp, &spec.interval_key)?;
                        layers / interval + u64::from(layers % interval != 0)
                    }
                }
            }
        };
        let head_count = self.get_u64_required(insp, &self.kv.head_count_key)?;
        let kv_head_count = self.get_u64_required(insp, &self.kv.kv_head_count_key)?;
        let embedding = self.get_u64_required(insp, &self.kv.embedding_key)?;
        if head_count == 0 || embedding % head_count != 0 {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "adapter {}: embedding length {embedding} is not divisible by head count {head_count}",
                    self.id
                ),
            ));
        }
        let default_head_dim = embedding / head_count;
        let key_length = match &self.kv.key_length_key {
            Some(k) => insp.get_u64(k).unwrap_or(default_head_dim),
            None => default_head_dim,
        };
        let value_length = match &self.kv.value_length_key {
            Some(k) => insp.get_u64(k).unwrap_or(default_head_dim),
            None => key_length,
        };
        if key_length == 0 || value_length == 0 {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "adapter {}: key/value length metadata resolves to zero",
                    self.id
                ),
            ));
        }
        let state_bytes = self.recurrent_state_bytes(insp, layers, attention_layers)?;
        Ok(KvGeometry {
            layers,
            attention_layers,
            state_bytes,
            head_count,
            kv_head_count,
            head_dim: key_length,
            key_length,
            value_length,
        })
    }

    /// Per-session bytes of the linear-attention state, from the declared
    /// `ssm.*` geometry. The two classes are the delta-rule state and the
    /// convolution history; the formulas and their derivation are
    /// registered in ADR-008.
    fn recurrent_state_bytes(
        &self,
        insp: &GgufInspection,
        layers: u64,
        attention_layers: u64,
    ) -> Result<u64> {
        let Some(spec) = &self.kv.recurrent else {
            return Ok(0);
        };
        let overflow = || {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!("adapter {}: recurrent state geometry overflows", self.id),
            )
        };
        let groups = self.get_u64_required(insp, &spec.groups_key)?;
        let state_size = self.get_u64_required(insp, &spec.state_size_key)?;
        let inner = self.get_u64_required(insp, &spec.inner_size_key)?;
        let conv = self.get_u64_required(insp, &spec.conv_kernel_key)?;
        if inner % groups != 0 {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "adapter {}: ssm inner size {inner} is not divisible by group count {groups}",
                    self.id
                ),
            ));
        }
        let elem = spec.element_bytes as u64;
        let per_layer_state = groups
            .checked_mul(state_size)
            .and_then(|v| v.checked_mul(inner / groups))
            .and_then(|v| v.checked_mul(elem))
            .ok_or_else(overflow)?;
        let conv_channels = groups
            .checked_mul(state_size)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_add(inner))
            .ok_or_else(overflow)?;
        let per_layer_conv = conv
            .saturating_sub(1)
            .checked_mul(conv_channels)
            .and_then(|v| v.checked_mul(elem))
            .ok_or_else(overflow)?;
        let recurrent_layers = layers - attention_layers;
        per_layer_state
            .checked_add(per_layer_conv)
            .and_then(|v| v.checked_mul(recurrent_layers))
            .ok_or_else(overflow)
    }

    pub fn expert_geometry(&self, insp: &GgufInspection) -> Result<Option<ExpertGeometry>> {
        let Some(meta) = &self.expert_meta else {
            return Ok(None);
        };
        let count = self.get_u64_required(insp, &meta.count_key)?;
        let used_per_token = self.get_u64_required(insp, &meta.used_key)?;
        if used_per_token > count {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "adapter {}: expert_used_count {used_per_token} exceeds expert_count {count}",
                    self.id
                ),
            ));
        }
        Ok(Some(ExpertGeometry {
            count,
            used_per_token,
        }))
    }
}

fn is_kernel_id(k: &str) -> bool {
    let bytes = k.as_bytes();
    bytes.len() == 3 && bytes[0] == b'K' && bytes[1].is_ascii_digit() && bytes[2].is_ascii_digit()
}

fn value_matches_kind(value: &GgufValue, kind: &str) -> bool {
    match kind {
        "u32" => matches!(
            value,
            GgufValue::U8(_) | GgufValue::U16(_) | GgufValue::U32(_) | GgufValue::U64(_)
        ),
        "u64" => matches!(
            value,
            GgufValue::U8(_) | GgufValue::U16(_) | GgufValue::U32(_) | GgufValue::U64(_)
        ),
        "f32" => matches!(value, GgufValue::F32(_) | GgufValue::F64(_)),
        "string" => matches!(value, GgufValue::String(_)),
        "bool" => matches!(value, GgufValue::Bool(_)),
        "array" => matches!(value, GgufValue::Array(_)),
        "any" => true,
        _ => false,
    }
}

/// Registry of adapter specifications.
#[derive(Debug, Clone)]
pub struct AdapterRegistry {
    adapters: Vec<AdapterSpec>,
}

impl AdapterRegistry {
    /// Adapters embedded from the repository `adapters/` directory.
    pub fn builtin() -> Result<Self> {
        let sources: [&str; 4] = [
            include_str!("../../../adapters/qwen3.json"),
            include_str!("../../../adapters/qwen3moe.json"),
            include_str!("../../../adapters/qwen3.5-hybrid.json"),
            include_str!("../../../adapters/bonsai2-hybrid.json"),
        ];
        let mut adapters = Vec::with_capacity(sources.len());
        for (i, src) in sources.iter().enumerate() {
            let spec: AdapterSpec = serde_json::from_str(src).map_err(|e| {
                crate::error::json_error(
                    e,
                    ErrorCode::InvalidConfiguration,
                    &format!("builtin adapter #{i} is invalid"),
                )
            })?;
            spec.validate()?;
            adapters.push(spec);
        }
        Ok(Self { adapters })
    }

    /// Load experimental adapters over the builtins. Duplicate ids replace
    /// the builtin; a replacement must still validate.
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let mut registry = Self::builtin()?;
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| {
                TuckmillError::new(
                    ErrorCode::InvalidConfiguration,
                    format!("cannot read adapters directory {}: {e}", dir.display()),
                )
            })?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        entries.sort();
        for path in entries {
            let src = std::fs::read_to_string(&path).map_err(TuckmillError::from)?;
            let spec: AdapterSpec = serde_json::from_str(&src).map_err(|e| {
                crate::error::json_error(
                    e,
                    ErrorCode::InvalidConfiguration,
                    &format!("adapter file {} is invalid", path.display()),
                )
            })?;
            spec.validate()?;
            if let Some(existing) = registry.adapters.iter_mut().find(|a| a.id == spec.id) {
                *existing = spec;
            } else {
                registry.adapters.push(spec);
            }
        }
        Ok(registry)
    }

    pub fn all(&self) -> &[AdapterSpec] {
        &self.adapters
    }

    pub fn find(&self, architecture: &str) -> Option<&AdapterSpec> {
        let arch = architecture.to_ascii_lowercase();
        self.adapters.iter().find(|a| a.architecture == arch)
    }

    pub fn find_by_id(&self, id: &str) -> Option<&AdapterSpec> {
        self.adapters.iter().find(|a| a.id == id)
    }

    /// Identity of the adapter set; plans record it so cached plans cannot be
    /// reused after an adapter change.
    pub fn registry_hash(&self) -> Result<crate::ids::Id> {
        #[derive(Serialize)]
        struct Row<'a> {
            id: &'a str,
            architecture: &'a str,
            required_kernels: &'a [String],
            required_transforms: &'a [String],
            certification: &'a CertificationRecord,
        }
        let rows: Vec<Row<'_>> = self
            .adapters
            .iter()
            .map(|a| Row {
                id: &a.id,
                architecture: &a.architecture,
                required_kernels: &a.required_kernels,
                required_transforms: &a.required_transforms,
                certification: &a.certification,
            })
            .collect();
        hash_of(&rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    fn registry() -> AdapterRegistry {
        AdapterRegistry::builtin().unwrap()
    }

    #[test]
    fn builtin_adapters_load_and_validate() {
        let r = registry();
        assert_eq!(r.all().len(), 4);
        for a in r.all() {
            assert_eq!(a.schema_version, ADAPTER_SCHEMA_VERSION);
            assert!(!a.required_kernels.is_empty(), "{} has no kernels", a.id);
            assert_eq!(a.certification.state, "unmeasured");
        }
    }

    #[test]
    fn finds_by_architecture() {
        let r = registry();
        assert!(r.find("qwen3").is_some());
        assert!(r.find("QWEN3MOE").is_some());
        assert!(r.find("qwen35").is_some());
        assert!(r.find("nope").is_none());
    }

    #[test]
    fn validates_metadata_on_fixture() {
        let r = registry();
        let adapter = r.find("qwen3").unwrap();
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        let mut cursor = std::io::Cursor::new(bytes.clone());
        let insp = crate::gguf::inspect_reader(
            &mut cursor,
            bytes.len() as u64,
            &crate::gguf::Limits::default(),
            None,
        )
        .unwrap();
        adapter.validate_metadata(&insp).unwrap();
        let kv = adapter.kv_geometry(&insp).unwrap();
        assert_eq!(kv.layers, 2);
        assert_eq!(kv.kv_head_count, 2);
        assert_eq!(kv.head_dim, 32);
        assert!(adapter.expert_geometry(&insp).unwrap().is_none());
    }

    #[test]
    fn moe_fixture_yields_expert_geometry() {
        let r = registry();
        let adapter = r.find("qwen3moe").unwrap();
        let cfg = SyntheticConfig::tiny_qwen3moe();
        let bytes = synthetic_qwen3_bytes(&cfg);
        let mut cursor = std::io::Cursor::new(bytes.clone());
        let insp = crate::gguf::inspect_reader(
            &mut cursor,
            bytes.len() as u64,
            &crate::gguf::Limits::default(),
            None,
        )
        .unwrap();
        adapter.validate_metadata(&insp).unwrap();
        let eg = adapter.expert_geometry(&insp).unwrap().unwrap();
        assert_eq!(eg.count, 8);
        assert_eq!(eg.used_per_token, 2);
    }

    #[test]
    fn missing_required_metadata_is_unsupported_architecture() {
        let r = registry();
        let adapter = r.find("qwen35").unwrap(); // requires hybrid keys the qwen3 fixture lacks
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        let mut cursor = std::io::Cursor::new(bytes.clone());
        let insp = crate::gguf::inspect_reader(
            &mut cursor,
            bytes.len() as u64,
            &crate::gguf::Limits::default(),
            None,
        )
        .unwrap();
        let err = adapter.validate_metadata(&insp).unwrap_err();
        assert_eq!(err.code(), ErrorCode::UnsupportedArchitecture);
        assert!(err.message().contains("missing required key"));
    }

    fn inspect_fixture(cfg: &SyntheticConfig) -> crate::gguf::GgufInspection {
        let bytes = synthetic_qwen3_bytes(cfg);
        let mut cursor = std::io::Cursor::new(bytes.clone());
        crate::gguf::inspect_reader(
            &mut cursor,
            bytes.len() as u64,
            &crate::gguf::Limits::default(),
            None,
        )
        .unwrap()
    }

    /// The qwen35 fixture mirrors the pinned 3:1 layout: the MTP block is
    /// excluded from the served stack, attention KV exists on the full
    /// attention layers only, and the recurrent state prices the declared
    /// `ssm.*` geometry.
    #[test]
    fn qwen35_fixture_resolves_hybrid_geometry() {
        let r = registry();
        let adapter = r.find("qwen35").unwrap();
        let insp = inspect_fixture(&SyntheticConfig::tiny_qwen35());
        adapter.validate_metadata(&insp).unwrap();
        let kv = adapter.kv_geometry(&insp).unwrap();
        assert_eq!(kv.layers, 4, "MTP block excluded from the served stack");
        assert_eq!(kv.attention_layers, 1);
        assert_eq!(kv.head_count, 4);
        assert_eq!(kv.kv_head_count, 2);
        assert_eq!(kv.head_dim, 16);
        assert!(adapter.expert_geometry(&insp).unwrap().is_none());
        // groups 2 x state 8 x head_v 8 x 4 B = 512 plus conv history
        // (4 - 1) x (16 + 2*2*8) x 4 B = 576 per recurrent layer, on 3
        // recurrent layers (4 served - 1 attention).
        assert_eq!(kv.state_bytes, (512 + 576) * 3);
    }

    /// A full-length `attention.recurrent_layers` array (true = recurrent,
    /// llama.cpp semantics) states the same fact per layer and wins over
    /// the interval.
    #[test]
    fn qwen35_recurrent_array_states_the_same_fact_per_layer() {
        let r = registry();
        let adapter = r.find("qwen35").unwrap();
        let insp = inspect_fixture(&SyntheticConfig::tiny_qwen35_with_recurrent_array());
        adapter.validate_metadata(&insp).unwrap();
        let kv = adapter.kv_geometry(&insp).unwrap();
        assert_eq!(kv.layers, 4);
        assert_eq!(kv.attention_layers, 1);
        assert_eq!(kv.state_bytes, (512 + 576) * 3);
    }

    /// An array that marks every served layer recurrent leaves no
    /// attention layer; that is not this architecture and must refuse.
    #[test]
    fn qwen35_all_recurrent_array_is_refused() {
        let r = registry();
        let adapter = r.find("qwen35").unwrap();
        let mut cfg = SyntheticConfig::tiny_qwen35();
        cfg.hybrid.as_mut().unwrap().recurrent_layers_array = Some(vec![true; 4]);
        let insp = inspect_fixture(&cfg);
        let err = adapter.kv_geometry(&insp).unwrap_err();
        assert_eq!(err.code(), ErrorCode::UnsupportedArchitecture);
        assert!(err.message().contains("no attention layer remains"));
    }

    #[test]
    fn kernel_ids_are_strictly_formed() {
        assert!(is_kernel_id("K01"));
        assert!(is_kernel_id("K99"));
        assert!(!is_kernel_id("K1"));
        assert!(!is_kernel_id("k01"));
        assert!(!is_kernel_id("K011"));
    }
}
