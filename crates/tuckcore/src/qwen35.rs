//! Qwen3.5 (`qwen35`) typed contract: configuration, exact layer map,
//! tensor resolution and state layout (Q35-1A, ADR-008).
//!
//! This module is the contract every Q35-1 oracle increment validates
//! against. The loader deliberately does not call it yet: `qwen35`
//! execution stays refused until the full Q35-1 checklist passes, so these
//! types cannot silently become an execution path. Everything here is
//! derived from the pinned artifact contract, never hand-edited:
//! layer kinds follow llama.cpp b11695 `qwen35.cpp`
//! (`(i + 1) % interval == 0`), tensor shapes follow the pinned
//! ggml-org Qwen3.5-0.8B GGUF and are verified against it by a
//! machine-local gated test.

use std::collections::BTreeMap;

use tuckmill_manifest::adapter::AdapterRegistry;
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::gguf::{GgufInspection, GgufValue};

/// Attention kind of one served layer, from the pinned interval rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35LayerKind {
    /// Gated DeltaNet layer: delta-rule recurrent state plus convolution
    /// history; QKV projection, output gate, per-head decay/beta gates.
    LinearAttention,
    /// Full attention: KV cache, gated output projection, per-head QK
    /// norm and partial RoPE.
    FullAttention,
}

/// Gated DeltaNet geometry of the linear-attention layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnConfig {
    /// Value/key head groups (the delta rule's head count).
    pub groups: u64,
    /// Per-head key and value state dimension.
    pub state_size: u64,
    /// Total value width; `inner / groups` is the per-head value dim.
    pub inner: u64,
    /// Causal convolution kernel length.
    pub conv_kernel: u64,
    /// Time-step rank (per-head decay parameters).
    pub dt_rank: u64,
}

/// Resolved Qwen3.5 configuration of the served stack.
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    /// Served layers (`block_count` minus trailing MTP/`nextn` blocks).
    pub layers: u64,
    /// Blocks in the artifact including trailing MTP/`nextn` blocks.
    pub block_count: u64,
    pub hidden: u64,
    /// Full-attention query heads.
    pub heads: u64,
    /// Full-attention KV heads.
    pub kv_heads: u64,
    /// Full-attention K/V head dimension (`attention.key_length`).
    pub head_dim: u64,
    pub ffn: u64,
    pub vocab: u64,
    pub rope_base: f32,
    /// Rotated dimensions per head (partial rotary factor applied).
    pub rope_dims: u64,
    /// Declared mrope sections (text positions use one axis, kept for the
    /// multimodal extension; recorded, never approximated).
    pub rope_sections: Vec<u64>,
    pub rms_eps: f32,
    /// Full-attention interval over served layers.
    pub interval: u64,
    pub gdn: GdnConfig,
    pub eos_token_id: Option<u32>,
}

impl Qwen35Config {
    /// Per-layer kind map: a served layer is full attention iff
    /// `(i + 1) % interval == 0` (llama.cpp b11695 `qwen35.cpp`).
    pub fn layer_kinds(&self) -> Vec<Qwen35LayerKind> {
        (0..self.layers)
            .map(|i| {
                if (i + 1) % self.interval == 0 {
                    Qwen35LayerKind::FullAttention
                } else {
                    Qwen35LayerKind::LinearAttention
                }
            })
            .collect()
    }

    /// Width of the `ssm_conv1d` channel dimension.
    pub fn conv_channels(&self) -> u64 {
        self.gdn.inner + 2 * self.gdn.groups * self.gdn.state_size
    }

    /// Rows of the GDN QKV projection (q + k + v, all at `inner` width).
    pub fn qkv_rows(&self) -> u64 {
        3 * self.gdn.inner
    }

    /// Constant per-session state layout (context-independent part).
    pub fn state_layout(&self) -> Qwen35StateLayout {
        let kinds = self.layer_kinds();
        let attention_layers = kinds
            .iter()
            .filter(|k| **k == Qwen35LayerKind::FullAttention)
            .count() as u64;
        let recurrent_layers = self.layers - attention_layers;
        let head_v = self.gdn.inner / self.gdn.groups;
        Qwen35StateLayout {
            recurrent_layers,
            attention_layers,
            bytes_per_recurrent_layer: self.gdn.groups * self.gdn.state_size * head_v * 4,
            bytes_per_conv_layer: (self.gdn.conv_kernel.saturating_sub(1))
                * self.conv_channels()
                * 4,
        }
    }
}

/// Per-session state sizing. Element widths follow the declared contract:
/// the delta-rule and convolution state are f32 (`mamba_ssm_dtype`), the
/// attention KV element width is the declared KV precision (f16 default,
/// f32 selectable) and must match what the executor actually allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35StateLayout {
    pub recurrent_layers: u64,
    pub attention_layers: u64,
    pub bytes_per_recurrent_layer: u64,
    pub bytes_per_conv_layer: u64,
}

impl Qwen35StateLayout {
    /// State bytes for an admitted context. Overflow is refused, never
    /// wrapped.
    pub fn bytes_for(
        &self,
        cfg: &Qwen35Config,
        context: u64,
        kv_element_bytes: u32,
    ) -> Result<u64> {
        let overflow = || {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                "qwen35 state layout overflows u64",
            )
        };
        let per_token = self
            .attention_layers
            .checked_mul(cfg.kv_heads)
            .and_then(|v| v.checked_mul(cfg.head_dim))
            .and_then(|v| v.checked_mul(2)) // key + value
            .and_then(|v| v.checked_mul(kv_element_bytes as u64))
            .ok_or_else(overflow)?;
        let attention = per_token.checked_mul(context).ok_or_else(overflow)?;
        let recurrent = self
            .bytes_per_recurrent_layer
            .checked_add(self.bytes_per_conv_layer)
            .and_then(|v| v.checked_mul(self.recurrent_layers))
            .ok_or_else(overflow)?;
        attention.checked_add(recurrent).ok_or_else(overflow)
    }
}

/// Resolved tensor slot: name plus its index in the inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRef {
    pub name: String,
    pub index: usize,
}

/// The two attention tensor sets; the variant is the layer kind, so an
/// invalid combination cannot be represented.
#[derive(Debug, Clone)]
pub enum Qwen35AttentionTensors {
    Linear {
        qkv: TensorRef,
        gate: TensorRef,
        a: TensorRef,
        dt: TensorRef,
        alpha: TensorRef,
        beta: TensorRef,
        conv1d: TensorRef,
        norm: TensorRef,
        out: TensorRef,
    },
    Full {
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        output: TensorRef,
        q_norm: TensorRef,
        k_norm: TensorRef,
    },
}

/// One resolved served layer.
#[derive(Debug, Clone)]
pub struct Qwen35Layer {
    pub attention: Qwen35AttentionTensors,
    pub attn_norm: TensorRef,
    pub ffn_gate: TensorRef,
    pub ffn_up: TensorRef,
    pub ffn_down: TensorRef,
    pub post_attn_norm: TensorRef,
}

impl Qwen35Layer {
    pub fn kind(&self) -> Qwen35LayerKind {
        match self.attention {
            Qwen35AttentionTensors::Linear { .. } => Qwen35LayerKind::LinearAttention,
            Qwen35AttentionTensors::Full { .. } => Qwen35LayerKind::FullAttention,
        }
    }
}

/// Full resolved layout: globals, every served layer, and the state
/// sizing. MTP/`nextn` blocks are not part of the laid-out stack.
#[derive(Debug, Clone)]
pub struct Qwen35Layout {
    pub token_embd: TensorRef,
    pub output_norm: TensorRef,
    /// Separate head; `None` means tied to the token embedding by the
    /// format's convention (the pinned artifact is tied).
    pub output: Option<TensorRef>,
    pub layers: Vec<Qwen35Layer>,
    pub state: Qwen35StateLayout,
}

impl Qwen35Layout {
    /// Number of resolved tensors (globals plus served layers).
    pub fn tensor_count(&self) -> usize {
        let mut n = 2 + usize::from(self.output.is_some());
        for l in &self.layers {
            n += 5;
            match &l.attention {
                Qwen35AttentionTensors::Linear { .. } => n += 9,
                Qwen35AttentionTensors::Full { .. } => n += 6,
            }
        }
        n
    }
}

/// Resolve the typed Qwen3.5 contract against an inspected artifact.
///
/// Fails closed on: wrong architecture, missing hybrid adapter, missing or
/// mistyped metadata, missing tensors, tensors whose dimensions disagree
/// with the resolved configuration, and unknown tensors inside the served
/// stack (an unrecognized tensor is a contract violation, not something to
/// skip). Trailing non-served blocks (MTP/`nextn`) are excluded by the
/// served-stack definition and may carry any tensors.
pub fn resolve(
    inspection: &GgufInspection,
    registry: &AdapterRegistry,
) -> Result<(Qwen35Config, Qwen35Layout)> {
    let arch = inspection.arch_id().ok_or_else(|| {
        TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            "artifact declares no architecture",
        )
    })?;
    if arch != "qwen35" {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!("the typed Qwen35 contract resolves only qwen35 artifacts; this artifact is {arch:?}"),
        ));
    }
    let adapter = registry
        .find(&arch)
        .ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                "no adapter is registered for architecture \"qwen35\"",
            )
        })?
        .clone();
    if !adapter.traits.hybrid_linear_attention {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "adapter {} does not declare a hybrid linear-attention stack",
                adapter.id
            ),
        ));
    }
    adapter.validate_metadata(inspection)?;
    let kv = adapter.kv_geometry(inspection)?;

    let req = |key: &str| -> Result<u64> {
        let full = format!("qwen35.{key}");
        inspection.get_u64(&full).filter(|v| *v > 0).ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!("{full} is required and must be positive"),
            )
        })
    };

    let block_count = req("block_count")?;
    let excluded = inspection
        .get_u64("qwen35.nextn_predict_layers")
        .unwrap_or(0);
    let layers = block_count.checked_sub(excluded).filter(|v| *v > 0).ok_or_else(|| {
        TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "qwen35: block_count {block_count} minus nextn_predict_layers {excluded} leaves no served stack"
            ),
        )
    })?;

    let mut index = Index::new(inspection);
    let token_embd_entry = index.lookup("token_embd.weight")?;
    let embd_dims = &inspection.tensors[token_embd_entry].dims;
    let hidden = req("embedding_length")?;
    if embd_dims.len() != 2 || embd_dims[0] != hidden {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!("token_embd.weight has dims {embd_dims:?} but embedding_length is {hidden}"),
        ));
    }
    let vocab = embd_dims[1];
    index.consumed[token_embd_entry] = true;

    let interval = req("full_attention_interval")?;
    let sections = match inspection.get("qwen35.rope.dimension_sections") {
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|v| match v {
                GgufValue::U8(x) => Ok(*x as u64),
                GgufValue::U16(x) => Ok(*x as u64),
                GgufValue::U32(x) => Ok(*x as u64),
                GgufValue::U64(x) => Ok(*x),
                GgufValue::I8(x) if *x >= 0 => Ok(*x as u64),
                GgufValue::I16(x) if *x >= 0 => Ok(*x as u64),
                GgufValue::I32(x) if *x >= 0 => Ok(*x as u64),
                GgufValue::I64(x) if *x >= 0 => Ok(*x as u64),
                other => Err(TuckmillError::new(
                    ErrorCode::UnsupportedArchitecture,
                    format!(
                        "qwen35.rope.dimension_sections entry has type {} (want a non-negative integer)",
                        other.type_name()
                    ),
                )),
            })
            .collect::<Result<Vec<u64>>>()?,
        Some(other) => {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "qwen35.rope.dimension_sections has type {} (want an array)",
                    other.type_name()
                ),
            ));
        }
        None => Vec::new(),
    };

    let cfg = Qwen35Config {
        layers,
        block_count,
        hidden,
        heads: req("attention.head_count")?,
        kv_heads: req("attention.head_count_kv")?,
        head_dim: req("attention.key_length")?,
        ffn: req("feed_forward_length")?,
        vocab,
        rope_base: inspection
            .get_f32("qwen35.rope.freq_base")
            .unwrap_or(10_000.0),
        rope_dims: req("rope.dimension_count")?,
        rope_sections: sections,
        rms_eps: inspection
            .get_f32("qwen35.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-6),
        interval,
        gdn: GdnConfig {
            groups: req("ssm.group_count")?,
            state_size: req("ssm.state_size")?,
            inner: req("ssm.inner_size")?,
            conv_kernel: req("ssm.conv_kernel")?,
            dt_rank: req("ssm.time_step_rank")?,
        },
        eos_token_id: inspection
            .get_u64("tokenizer.ggml.eos_token_id")
            .map(|v| v as u32),
    };
    if !cfg.gdn.inner.is_multiple_of(cfg.gdn.groups) {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "qwen35: ssm inner size {} is not divisible by group count {}",
                cfg.gdn.inner, cfg.gdn.groups
            ),
        ));
    }

    let kinds = cfg.layer_kinds();
    let state = cfg.state_layout();
    let attention_layers = state.attention_layers;
    let recurrent_layers = state.recurrent_layers;
    if attention_layers == 0 || recurrent_layers == 0 {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "qwen35: the interval {interval} splits {layers} served layers into {attention_layers} attention and {recurrent_layers} recurrent layers; a hybrid stack needs both"
            ),
        ));
    }

    // The adapter's priced geometry and this layout are two independent
    // derivations of the same fact; disagreement is a contract bug.
    if kv.layers != cfg.layers || kv.attention_layers != attention_layers {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "qwen35: adapter geometry ({} served, {} attention) disagrees with the typed layout ({}, {})",
                kv.layers, kv.attention_layers, cfg.layers, attention_layers
            ),
        ));
    }
    let constant_state = state.bytes_for(&cfg, 0, 0)?;
    if kv.state_bytes != constant_state {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "qwen35: adapter state pricing {constant_state} B disagrees with the typed layout {kv_state} B",
                kv_state = kv.state_bytes
            ),
        ));
    }

    // --- tensor map ---
    let mut layers_out = Vec::with_capacity(layers as usize);
    for (l, kind) in kinds.iter().enumerate() {
        let p = |s: &str| format!("blk.{l}.{s}");
        let attention = match kind {
            Qwen35LayerKind::LinearAttention => Qwen35AttentionTensors::Linear {
                qkv: index.take(&p("attn_qkv.weight"), &[hidden, cfg.qkv_rows()], "GDN qkv")?,
                gate: index.take(&p("attn_gate.weight"), &[hidden, cfg.gdn.inner], "GDN gate")?,
                a: index.take(&p("ssm_a"), &[cfg.gdn.dt_rank], "GDN ssm_a")?,
                dt: index.take(&p("ssm_dt.bias"), &[cfg.gdn.dt_rank], "GDN ssm_dt")?,
                alpha: index.take(
                    &p("ssm_alpha.weight"),
                    &[hidden, cfg.gdn.dt_rank],
                    "GDN alpha",
                )?,
                beta: index.take(
                    &p("ssm_beta.weight"),
                    &[hidden, cfg.gdn.dt_rank],
                    "GDN beta",
                )?,
                conv1d: index.take(
                    &p("ssm_conv1d.weight"),
                    &[cfg.gdn.conv_kernel, cfg.conv_channels()],
                    "GDN conv1d",
                )?,
                norm: index.take(
                    &p("ssm_norm.weight"),
                    &[cfg.gdn.inner / cfg.gdn.groups],
                    "GDN norm",
                )?,
                out: index.take(&p("ssm_out.weight"), &[cfg.gdn.inner, hidden], "GDN out")?,
            },
            Qwen35LayerKind::FullAttention => Qwen35AttentionTensors::Full {
                q: index.take(
                    &p("attn_q.weight"),
                    &[hidden, cfg.heads * cfg.head_dim * 2],
                    "attention q (+gate)",
                )?,
                k: index.take(
                    &p("attn_k.weight"),
                    &[hidden, cfg.kv_heads * cfg.head_dim],
                    "attention k",
                )?,
                v: index.take(
                    &p("attn_v.weight"),
                    &[hidden, cfg.kv_heads * cfg.head_dim],
                    "attention v",
                )?,
                output: index.take(
                    &p("attn_output.weight"),
                    &[cfg.heads * cfg.head_dim, hidden],
                    "attention output",
                )?,
                q_norm: index.take(
                    &p("attn_q_norm.weight"),
                    &[cfg.head_dim],
                    "attention q norm",
                )?,
                k_norm: index.take(
                    &p("attn_k_norm.weight"),
                    &[cfg.head_dim],
                    "attention k norm",
                )?,
            },
        };
        layers_out.push(Qwen35Layer {
            attention,
            attn_norm: index.take(&p("attn_norm.weight"), &[hidden], "attn norm")?,
            ffn_gate: index.take(&p("ffn_gate.weight"), &[hidden, cfg.ffn], "ffn gate")?,
            ffn_up: index.take(&p("ffn_up.weight"), &[hidden, cfg.ffn], "ffn up")?,
            ffn_down: index.take(&p("ffn_down.weight"), &[cfg.ffn, hidden], "ffn down")?,
            post_attn_norm: index.take(
                &p("post_attention_norm.weight"),
                &[hidden],
                "post attention norm",
            )?,
        });
    }

    let output_norm = index.take("output_norm.weight", &[hidden], "output norm")?;
    let output = if index.has("output.weight") {
        Some(index.take("output.weight", &[hidden, vocab], "output head")?)
    } else {
        None
    };
    index.require_no_leftovers(layers)?;

    let layout = Qwen35Layout {
        token_embd: TensorRef {
            name: "token_embd.weight".into(),
            index: token_embd_entry,
        },
        output_norm,
        output,
        layers: layers_out,
        state,
    };
    Ok((cfg, layout))
}

/// Name-indexed resolver over an inspection: each served tensor is taken
/// exactly once, with its dimensions verified against the resolved
/// configuration.
struct Index<'a> {
    inspection: &'a GgufInspection,
    by_name: BTreeMap<&'a str, usize>,
    consumed: Vec<bool>,
}

impl<'a> Index<'a> {
    fn new(inspection: &'a GgufInspection) -> Self {
        let by_name = inspection
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.as_str(), i))
            .collect();
        Self {
            consumed: vec![false; inspection.tensors.len()],
            by_name,
            inspection,
        }
    }

    fn lookup(&self, name: &str) -> Result<usize> {
        self.by_name.get(name).copied().ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!("artifact is missing required qwen35 tensor {name:?}"),
            )
        })
    }

    fn has(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    fn take(&mut self, name: &str, dims: &[u64], what: &str) -> Result<TensorRef> {
        let index = self.lookup(name)?;
        let t = &self.inspection.tensors[index];
        if t.dims != dims {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "tensor {name:?} ({what}) has dims {:?} but the resolved contract requires {dims:?}",
                    t.dims
                ),
            ));
        }
        self.consumed[index] = true;
        Ok(TensorRef {
            name: name.to_string(),
            index,
        })
    }

    /// Any tensor inside the served stack that was not consumed is a
    /// contract violation; trailing blocks at or beyond `layers` are the
    /// non-served MTP/`nextn` region and may carry anything.
    fn require_no_leftovers(&self, layers: u64) -> Result<()> {
        for (i, t) in self.inspection.tensors.iter().enumerate() {
            if self.consumed[i] {
                continue;
            }
            let in_trailing = t
                .name
                .strip_prefix("blk.")
                .and_then(|rest| rest.split_once('.'))
                .and_then(|(n, _)| n.parse::<u64>().ok())
                .is_some_and(|n| n >= layers);
            if !in_trailing {
                return Err(TuckmillError::new(
                    ErrorCode::UnsupportedArchitecture,
                    format!(
                        "tensor {:?} is not part of the resolved qwen35 served contract",
                        t.name
                    ),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    fn fixture(cfg: &SyntheticConfig) -> GgufInspection {
        let bytes = synthetic_qwen3_bytes(cfg);
        let mut cursor = std::io::Cursor::new(bytes.clone());
        tuckmill_manifest::gguf::inspect_reader(
            &mut cursor,
            bytes.len() as u64,
            &tuckmill_manifest::gguf::Limits::default(),
            None,
        )
        .unwrap()
    }

    fn registry() -> AdapterRegistry {
        AdapterRegistry::builtin().unwrap()
    }

    #[test]
    fn fixture_resolves_the_typed_contract() {
        let insp = fixture(&SyntheticConfig::tiny_qwen35());
        let (cfg, layout) = resolve(&insp, &registry()).unwrap();
        assert_eq!(cfg.layers, 4);
        assert_eq!(cfg.block_count, 5, "one MTP block is not served");
        assert_eq!(cfg.interval, 4);
        let kinds = cfg.layer_kinds();
        assert_eq!(kinds[0], Qwen35LayerKind::LinearAttention);
        assert_eq!(kinds[1], Qwen35LayerKind::LinearAttention);
        assert_eq!(kinds[2], Qwen35LayerKind::LinearAttention);
        assert_eq!(kinds[3], Qwen35LayerKind::FullAttention);
        assert_eq!(layout.layers.len(), 4);
        assert_eq!(layout.layers[3].kind(), Qwen35LayerKind::FullAttention);
        assert!(
            layout.output.is_none(),
            "the hybrid fixture mirrors the pinned artifact's tied embeddings"
        );
        assert_eq!(layout.state.recurrent_layers, 3);
        assert_eq!(layout.state.attention_layers, 1);
        assert_eq!(layout.state.bytes_per_recurrent_layer, 2 * 8 * 8 * 4);
        assert_eq!(layout.state.bytes_per_conv_layer, 3 * 48 * 4);
        // 1 attention layer x 2 kv heads x 16 head_dim x 2 (k+v) x 2 B
        // per token, plus the constant recurrent state.
        assert_eq!(
            layout.state.bytes_for(&cfg, 1024, 2).unwrap(),
            1024 * 128 + 3 * (512 + 576)
        );
        // 2 globals (tied output) + 3 GDN x 14 + 1 full attention x 11.
        assert_eq!(layout.tensor_count(), 2 + 3 * 14 + 11);
        // No MTP tensor leaks into the layout.
        for l in &layout.layers {
            assert!(!l.attn_norm.name.starts_with("blk.4."));
        }
    }

    #[test]
    fn non_divisible_stack_keeps_full_groups_only() {
        let insp = fixture(&SyntheticConfig::tiny_qwen35_non_divisible());
        let (cfg, layout) = resolve(&insp, &registry()).unwrap();
        assert_eq!(cfg.layers, 5);
        let kinds = cfg.layer_kinds();
        assert_eq!(
            kinds,
            vec![
                Qwen35LayerKind::LinearAttention,
                Qwen35LayerKind::LinearAttention,
                Qwen35LayerKind::LinearAttention,
                Qwen35LayerKind::FullAttention,
                Qwen35LayerKind::LinearAttention,
            ],
            "the trailing partial group adds no attention layer"
        );
        assert_eq!(layout.state.attention_layers, 1);
        assert_eq!(layout.state.recurrent_layers, 4);
    }

    #[test]
    fn non_qwen35_artifacts_are_refused() {
        let insp = fixture(&SyntheticConfig::tiny_qwen3());
        let err = resolve(&insp, &registry()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::UnsupportedArchitecture);
        assert!(err.message().contains("only qwen35"), "{}", err.message());
    }

    fn real_artifact_path() -> Option<std::path::PathBuf> {
        let p = std::env::var("TUCKMILL_QWEN35_GGUF").ok()?;
        let p = std::path::PathBuf::from(p);
        p.exists().then_some(p)
    }

    /// Machine-local Q35-1A gate: the typed contract must resolve the
    /// pinned artifact exactly — every served tensor, every dimension,
    /// MTP excluded, state sizing equal to the adapter's priced geometry.
    #[test]
    fn real_artifact_layout_when_available() {
        let Some(path) = real_artifact_path() else {
            eprintln!("skipping: TUCKMILL_QWEN35_GGUF not set (pinned Qwen3.5-0.8B Q8_0)");
            return;
        };
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let (cfg, layout) = resolve(&insp, &registry()).unwrap();
        assert_eq!(cfg.layers, 24);
        assert_eq!(cfg.block_count, 25);
        assert_eq!(cfg.hidden, 1024);
        assert_eq!(cfg.heads, 8);
        assert_eq!(cfg.kv_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.ffn, 3584);
        assert_eq!(cfg.vocab, 248320);
        assert_eq!(cfg.rope_dims, 64);
        assert_eq!(cfg.rope_sections, vec![11, 11, 10, 0]);
        assert_eq!(cfg.interval, 4);
        assert_eq!(cfg.gdn.groups, 16);
        assert_eq!(cfg.gdn.state_size, 128);
        assert_eq!(cfg.gdn.inner, 2048);
        assert_eq!(cfg.gdn.conv_kernel, 4);
        assert_eq!(cfg.gdn.dt_rank, 16);
        assert_eq!(cfg.eos_token_id, Some(248046));
        assert!(
            layout.output.is_none(),
            "the pinned artifact ties its output to the token embedding"
        );

        let kinds = cfg.layer_kinds();
        assert_eq!(
            kinds
                .iter()
                .filter(|k| **k == Qwen35LayerKind::FullAttention)
                .count(),
            6
        );
        assert_eq!(
            kinds
                .iter()
                .filter(|k| **k == Qwen35LayerKind::LinearAttention)
                .count(),
            18
        );
        for (i, k) in kinds.iter().enumerate() {
            let want = if (i + 1) % 4 == 0 {
                Qwen35LayerKind::FullAttention
            } else {
                Qwen35LayerKind::LinearAttention
            };
            assert_eq!(*k, want, "layer {i}");
        }

        assert_eq!(layout.state.attention_layers, 6);
        assert_eq!(layout.state.recurrent_layers, 18);
        assert_eq!(layout.state.bytes_per_recurrent_layer, 16 * 128 * 128 * 4);
        assert_eq!(
            layout.state.bytes_per_conv_layer,
            3 * (2048 + 2 * 16 * 128) * 4
        );
        assert_eq!(
            layout.state.bytes_for(&cfg, 2048, 2).unwrap(),
            2048 * 12288 + 18 * 1_122_304,
            "attention KV (f16) at ctx 2048 plus the constant recurrent state"
        );
        // 335 tensors in the artifact: 320 served (consumed by resolve,
        // which refuses leftovers) and 15 in the MTP block.
        assert_eq!(layout.tensor_count(), 320);
        assert_eq!(insp.tensors.len(), 335);
    }
}
