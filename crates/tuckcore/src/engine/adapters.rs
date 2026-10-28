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
        }),
    };
    LayerSpec {
        pre_norm: t(&layer.attn_norm),
        mixer,
        post_norm: t(&layer.post_attn_norm),
        ffn: FfnSpec::SwiGlu {
            gate: t(&layer.ffn_gate),
            up: t(&layer.ffn_up),
            down: t(&layer.ffn_down),
            width: cfg.ffn as usize,
        },
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
            }),
            post_norm: index(&names.ffn_norm[l])?,
            ffn: FfnSpec::SwiGlu {
                gate: index(&names.ffn_gate[l])?,
                up: index(&names.ffn_up[l])?,
                down: index(&names.ffn_down[l])?,
                width: cfg.ffn as usize,
            },
        });
    }
    let embedding = index(&names.token_embd)?;
    Ok(ModelGraph {
        architecture: "qwen3".to_string(),
        hidden: cfg.hidden as usize,
        vocab: cfg.vocab as usize,
        eps: cfg.rms_eps,
        embedding,
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
