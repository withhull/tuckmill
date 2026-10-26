//! Adapters: resolved family contracts → [`ModelGraph`] (ADR-011). An
//! adapter only describes; it never executes.

use super::graph::{
    AttentionSpec, FfnSpec, GatedDeltaNetSpec, LayerSpec, MixerSpec, ModelGraph, Tensor,
};
use crate::qwen35::{Qwen35AttentionTensors, Qwen35Config, Qwen35Layout};

fn t(r: &crate::qwen35::TensorRef) -> Tensor {
    Tensor { index: r.index }
}

/// Qwen3.5 hybrid (`qwen35`): Gated DeltaNet and gated full-attention
/// layers, SwiGLU FFNs, tied head unless the artifact carries `output`.
pub fn qwen35(cfg: &Qwen35Config, layout: &Qwen35Layout) -> ModelGraph {
    let layers = layout
        .layers
        .iter()
        .map(|layer| {
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
        })
        .collect();
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
        eos_token_id: cfg.eos_token_id,
    }
}
