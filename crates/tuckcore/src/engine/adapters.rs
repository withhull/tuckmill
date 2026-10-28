//! Adapters: resolved family contracts → [`ModelGraph`] (ADR-011). An
//! adapter only describes; it never executes.

use super::graph::{
    AttentionSpec, FfnSpec, GatedDeltaNetSpec, LayerSpec, MixerSpec, ModelGraph, MtpSpec, Tensor,
};
use crate::model::{Qwen3Config, TensorNames};
use crate::qwen35::{Qwen35AttentionTensors, Qwen35Config, Qwen35Layout};
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::gguf::GgufInspection;

fn t(r: &crate::qwen35::TensorRef) -> Tensor {
    Tensor { index: r.index }
}

/// Qwen3.5 hybrid (`qwen35`): Gated DeltaNet and gated full-attention
/// layers, SwiGLU FFNs, tied head unless the artifact carries `output`.
pub fn qwen35(cfg: &Qwen35Config, layout: &Qwen35Layout) -> ModelGraph {
    let layers = layout
        .layers
        .iter()
        .map(|layer| qwen35_layer(cfg, layer))
        .collect();
    let mtp = layout.mtp.as_ref().map(|m| MtpSpec {
        enorm: t(&m.enorm),
        hnorm: t(&m.hnorm),
        eh_proj: t(&m.eh_proj),
        layer: qwen35_layer(cfg, &m.layer),
        final_norm: t(&m.shared_head_norm),
    });
    qwen35_graph(cfg, layout, layers, mtp)
}

fn qwen35_layer(cfg: &Qwen35Config, layer: &crate::qwen35::Qwen35Layer) -> LayerSpec {
    let mixer = match &layer.attention {
        Qwen35AttentionTensors::Linear {
            qkv,
            gate,
            a,
            dt,
            alpha,
            beta,
            conv1d,
            norm,
            out,
        } => MixerSpec::GatedDeltaNet(GatedDeltaNetSpec {
            qkv: t(qkv),
            z: t(gate),
            alpha: t(alpha),
            beta: t(beta),
            out: t(out),
            ssm_a: t(a),
            ssm_dt: t(dt),
            ssm_norm: t(norm),
            conv1d: t(conv1d),
            channels: cfg.conv_channels() as usize,
            groups: cfg.gdn.groups as usize,
            state_size: cfg.gdn.state_size as usize,
            head_v: cfg.gdn.head_v as usize,
            dt_rank: cfg.gdn.dt_rank as usize,
            conv_kernel: cfg.gdn.conv_kernel as usize,
        }),
        Qwen35AttentionTensors::Full {
            q,
            k,
            v,
            output,
            q_norm,
            k_norm,
        } => MixerSpec::Attention(AttentionSpec {
            q: t(q),
            k: t(k),
            v: t(v),
            o: t(output),
            q_norm: Some(t(q_norm)),
            k_norm: Some(t(k_norm)),
            heads: cfg.heads as usize,
            kv_heads: cfg.kv_heads as usize,
            head_dim: cfg.head_dim as usize,
            rope_dims: cfg.rope_dims as usize,
            rope_base: cfg.rope_base,
            output_gate: true,
            window: None,
        }),
    };
    LayerSpec {
        pre_norm: t(&layer.attn_norm),
        mixer,
        mixer_out_norm: None,
        post_norm: t(&layer.post_attn_norm),
        ffn: FfnSpec::SwiGlu {
            gate: t(&layer.ffn_gate),
            up: t(&layer.ffn_up),
            down: t(&layer.ffn_down),
            width: cfg.ffn as usize,
        },
        ffn_out_norm: None,
    }
}

fn qwen35_graph(
    cfg: &Qwen35Config,
    layout: &Qwen35Layout,
    layers: Vec<LayerSpec>,
    mtp: Option<MtpSpec>,
) -> ModelGraph {
    ModelGraph {
        architecture: "qwen35".to_string(),
        hidden: cfg.hidden as usize,
        vocab: cfg.vocab as usize,
        eps: cfg.rms_eps,
        embedding: t(&layout.token_embd),
        embedding_scale: None,
        layers,
        final_norm: t(&layout.output_norm),
        head: layout
            .output
            .as_ref()
            .map(t)
            .unwrap_or(t(&layout.token_embd)),
        mtp,
        eos_token_id: cfg.eos_token_id,
    }
}

/// Dense Qwen3 (`qwen3`): grouped-query attention with per-head Q/K RMS
/// norms and full NeoX RoPE, no output gate, SwiGLU FFNs; tied head unless
/// the artifact carries `output`. `names` must already be resolved against
/// `insp` (every tensor present).
pub fn qwen3(cfg: &Qwen3Config, names: &TensorNames, insp: &GgufInspection) -> Result<ModelGraph> {
    if cfg.key_length != cfg.value_length {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "dense Qwen3 with key length {} != value length {} has no engine block; attention heads are square",
                cfg.key_length, cfg.value_length
            ),
        )
        .with_action("Use an artifact with equal key and value head sizes; shapes are never approximated."));
    }
    let index = |name: &str| -> Result<Tensor> {
        insp.tensors
            .iter()
            .position(|t| t.name == name)
            .map(|index| Tensor { index })
            .ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::UnsupportedArchitecture,
                    format!("artifact is missing required tensor {name:?} for dense Qwen3"),
                )
            })
    };
    let mut layers = Vec::with_capacity(cfg.layers as usize);
    for l in 0..cfg.layers as usize {
        layers.push(LayerSpec {
            pre_norm: index(&names.attn_norm[l])?,
            mixer: MixerSpec::Attention(AttentionSpec {
                q: index(&names.attn_q[l])?,
                k: index(&names.attn_k[l])?,
                v: index(&names.attn_v[l])?,
                o: index(&names.attn_output[l])?,
                q_norm: Some(index(&names.attn_q_norm[l])?),
                k_norm: Some(index(&names.attn_k_norm[l])?),
                heads: cfg.heads as usize,
                kv_heads: cfg.kv_heads as usize,
                head_dim: cfg.key_length as usize,
                rope_dims: cfg.key_length as usize,
                rope_base: cfg.rope_base,
                output_gate: false,
                window: None,
            }),
            mixer_out_norm: None,
            post_norm: index(&names.ffn_norm[l])?,
            ffn: FfnSpec::SwiGlu {
                gate: index(&names.ffn_gate[l])?,
                up: index(&names.ffn_up[l])?,
                down: index(&names.ffn_down[l])?,
                width: cfg.ffn as usize,
            },
            ffn_out_norm: None,
        });
    }
    let embedding = index(&names.token_embd)?;
    Ok(ModelGraph {
        architecture: "qwen3".to_string(),
        hidden: cfg.hidden as usize,
        vocab: cfg.vocab as usize,
        eps: cfg.rms_eps,
        embedding,
        embedding_scale: None,
        layers,
        final_norm: index(&names.output_norm)?,
        head: match &names.output {
            Some(output) => index(output)?,
            None => embedding,
        },
        mtp: None,
        eos_token_id: cfg.eos_token_id,
    })
}

/// Every `GEMMA3_SWA_PATTERN`-th layer (1-based) is a global-attention
/// layer; the others use the sliding window (llama.cpp's gemma3 pattern).
const GEMMA3_SWA_PATTERN: usize = 6;
/// RoPE base of the sliding-window layers (fixed by the architecture; the
/// artifact's `rope.freq_base` is the global layers').
const GEMMA3_SWA_ROPE_BASE: f32 = 10_000.0;

/// Gemma 3 text (`gemma3`, ADR-013): per-head Q/K RMS norms and full NeoX
/// RoPE, sliding-window layers with a global layer every sixth, sandwich
/// norms around attention and the FFN, GeGLU FFNs, embeddings scaled by
/// `sqrt(hidden)`, tied head. Variants this graph does not express exactly
/// (RoPE scaling, logit soft-capping, another layer pattern) are refused.
pub fn gemma3(insp: &GgufInspection) -> Result<ModelGraph> {
    let refuse = |what: String| {
        TuckmillError::new(ErrorCode::UnsupportedArchitecture, what)
            .with_action("Use a Gemma 3 text artifact this graph expresses exactly; variants are never approximated.")
    };
    let num = |key: &str| -> Result<usize> {
        insp.get_u64(&format!("gemma3.{key}"))
            .map(|v| v as usize)
            .ok_or_else(|| refuse(format!("gemma3 artifact lacks gemma3.{key}")))
    };
    for key in [
        "gemma3.rope.scaling.factor",
        "gemma3.final_logit_softcapping",
        "gemma3.attn_logit_softcapping",
    ] {
        if insp.get(key).is_some() {
            return Err(refuse(format!(
                "gemma3 artifact declares {key}, which this graph does not implement"
            )));
        }
    }
    if let Some(p) = insp.get("gemma3.attention.sliding_window_pattern") {
        if p.as_u64() != Some(GEMMA3_SWA_PATTERN as u64) {
            return Err(refuse(format!(
                "gemma3 sliding-window pattern {p:?} is not every {GEMMA3_SWA_PATTERN}th layer"
            )));
        }
    }
    let (layers_n, hidden, ffn) = (
        num("block_count")?,
        num("embedding_length")?,
        num("feed_forward_length")?,
    );
    let (heads, kv_heads) = (
        num("attention.head_count")?,
        num("attention.head_count_kv")?,
    );
    let (key_len, value_len) = (num("attention.key_length")?, num("attention.value_length")?);
    let window = num("attention.sliding_window")?;
    if key_len != value_len || heads == 0 || kv_heads == 0 || heads % kv_heads != 0 || window == 0 {
        return Err(refuse(format!(
            "gemma3 attention geometry (heads {heads}, kv heads {kv_heads}, key {key_len}, value {value_len}, window {window}) has no engine block"
        )));
    }
    let eps = insp
        .get_f32("gemma3.attention.layer_norm_rms_epsilon")
        .ok_or_else(|| refuse("gemma3 artifact lacks its RMS norm epsilon".into()))?;
    let global_base = insp
        .get_f32("gemma3.rope.freq_base")
        .ok_or_else(|| refuse("gemma3 artifact lacks gemma3.rope.freq_base".into()))?;
    let index = |name: &str| -> Result<Tensor> {
        insp.tensors
            .iter()
            .position(|t| t.name == name)
            .map(|index| Tensor { index })
            .ok_or_else(|| {
                refuse(format!(
                    "artifact is missing required tensor {name:?} for gemma3"
                ))
            })
    };
    if insp.tensors.iter().any(|t| t.name == "output.weight") {
        return Err(refuse(
            "gemma3 artifact carries an untied output head; this graph ties it".into(),
        ));
    }
    let embedding = index("token_embd.weight")?;
    let vocab = insp.tensors[embedding.index]
        .dims
        .get(1)
        .copied()
        .ok_or_else(|| refuse("token embedding is not 2-D".into()))? as usize;
    let mut layers = Vec::with_capacity(layers_n);
    for l in 0..layers_n {
        let p = |s: &str| format!("blk.{l}.{s}.weight");
        let global = (l + 1) % GEMMA3_SWA_PATTERN == 0;
        layers.push(LayerSpec {
            pre_norm: index(&p("attn_norm"))?,
            mixer: MixerSpec::Attention(AttentionSpec {
                q: index(&p("attn_q"))?,
                k: index(&p("attn_k"))?,
                v: index(&p("attn_v"))?,
                o: index(&p("attn_output"))?,
                q_norm: Some(index(&p("attn_q_norm"))?),
                k_norm: Some(index(&p("attn_k_norm"))?),
                heads,
                kv_heads,
                head_dim: key_len,
                rope_dims: key_len,
                rope_base: if global {
                    global_base
                } else {
                    GEMMA3_SWA_ROPE_BASE
                },
                output_gate: false,
                window: if global { None } else { Some(window) },
            }),
            mixer_out_norm: Some(index(&p("post_attention_norm"))?),
            post_norm: index(&p("ffn_norm"))?,
            ffn: FfnSpec::GeGlu {
                gate: index(&p("ffn_gate"))?,
                up: index(&p("ffn_up"))?,
                down: index(&p("ffn_down"))?,
                width: ffn,
            },
            ffn_out_norm: Some(index(&p("post_ffw_norm"))?),
        });
    }
    Ok(ModelGraph {
        architecture: "gemma3".to_string(),
        hidden,
        vocab,
        eps,
        embedding,
        embedding_scale: Some((hidden as f32).sqrt()),
        layers,
        final_norm: index("output_norm.weight")?,
        head: embedding,
        mtp: None,
        eos_token_id: insp
            .get_u64("tokenizer.ggml.eos_token_id")
            .map(|v| v as u32),
    })
}
