//! The typed model graph every adapter produces and the engine executes
//! (ADR-011). A graph is the artifact's computation, exactly: blocks are
//! never substituted, dropped or approximated. Every tensor binding is an
//! inspection index whose dtype and shape the engine checks at load.

/// A bound artifact tensor (its index in the inspection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tensor {
    pub index: usize,
}

/// Grouped-query attention.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionSpec {
    /// Query projection; with `output_gate` it jointly produces the query
    /// and a sigmoid output gate (2 x heads x head_dim rows).
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub o: Tensor,
    /// Per-head RMS norms of Q and K (weighted, before RoPE).
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Rotated dimensions per head (split-half RoPE over the leading
    /// `rope_dims`; equal to `head_dim` for full rotation).
    pub rope_dims: usize,
    pub rope_base: f32,
    /// Sigmoid gate on the attention output from the joint Q projection.
    pub output_gate: bool,
    /// Sliding-window attention: each query attends only the last `window`
    /// positions up to and including its own; `None` is full causal
    /// attention.
    pub window: Option<usize>,
}

/// Gated DeltaNet linear attention (Qwen3.5 recurrent layers).
#[derive(Debug, Clone, PartialEq)]
pub struct GatedDeltaNetSpec {
    pub qkv: Tensor,
    pub z: Tensor,
    pub alpha: Tensor,
    pub beta: Tensor,
    pub out: Tensor,
    pub ssm_a: Tensor,
    pub ssm_dt: Tensor,
    pub ssm_norm: Tensor,
    pub conv1d: Tensor,
    /// Convolution channel width (q + k + v segments).
    pub channels: usize,
    /// Key heads (delta-rule groups).
    pub groups: usize,
    /// Key/query state size per group.
    pub state_size: usize,
    /// Value-head dimension.
    pub head_v: usize,
    /// Value heads (time-step rank).
    pub dt_rank: usize,
    pub conv_kernel: usize,
}

/// The token mixer of a layer.
#[derive(Debug, Clone, PartialEq)]
pub enum MixerSpec {
    Attention(AttentionSpec),
    GatedDeltaNet(GatedDeltaNetSpec),
}

/// The feed-forward block of a layer.
#[derive(Debug, Clone, PartialEq)]
pub enum FfnSpec {
    /// `down(silu(gate x) * up x)`.
    SwiGlu {
        gate: Tensor,
        up: Tensor,
        down: Tensor,
        width: usize,
    },
    /// `down(gelu_tanh(gate x) * up x)` (GeGLU, ggml's tanh GELU).
    GeGlu {
        gate: Tensor,
        up: Tensor,
        down: Tensor,
        width: usize,
    },
}

/// Gate activation of a gated FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GluActivation {
    Silu,
    GeluTanh,
}

impl FfnSpec {
    /// `(gate, up, down, width, activation)`.
    pub fn parts(&self) -> (Tensor, Tensor, Tensor, usize, GluActivation) {
        match self {
            Self::SwiGlu {
                gate,
                up,
                down,
                width,
            } => (*gate, *up, *down, *width, GluActivation::Silu),
            Self::GeGlu {
                gate,
                up,
                down,
                width,
            } => (*gate, *up, *down, *width, GluActivation::GeluTanh),
        }
    }
}

/// One layer: `x += out_norm(mixer(norm(x))); x += out_norm(ffn(norm(x)))`,
/// where each output norm is present only when the artifact has it (Gemma's
/// post-attention and post-FFN norms).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSpec {
    pub pre_norm: Tensor,
    pub mixer: MixerSpec,
    /// RMS norm on the mixer output before its residual add.
    pub mixer_out_norm: Option<Tensor>,
    pub post_norm: Tensor,
    pub ffn: FfnSpec,
    /// RMS norm on the FFN output before its residual add.
    pub ffn_out_norm: Option<Tensor>,
}

/// A multi-token-prediction head (ADR-012 step 3): from the main model's
/// hidden state at position i and the embedding of token i+1, it drafts
/// token i+2. Drafts are only proposals; the main model verifies every one,
/// so the served output never depends on this head.
#[derive(Debug, Clone, PartialEq)]
pub struct MtpSpec {
    /// Norm of the next token's embedding.
    pub enorm: Tensor,
    /// Norm of the main model's hidden state.
    pub hnorm: Tensor,
    /// `[enorm(e); hnorm(h)]` (2 x hidden) -> hidden.
    pub eh_proj: Tensor,
    /// One transformer layer with its own KV.
    pub layer: LayerSpec,
    /// Norm before the shared output head.
    pub final_norm: Tensor,
}

/// The complete served computation of a model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelGraph {
    /// Architecture id the adapter resolved (records and refusals).
    pub architecture: String,
    pub hidden: usize,
    pub vocab: usize,
    /// RMS norm epsilon shared by every norm in the graph.
    pub eps: f32,
    pub embedding: Tensor,
    /// Multiplier on embedding rows before the first layer (Gemma:
    /// `sqrt(hidden)`); `None` is 1.
    pub embedding_scale: Option<f32>,
    pub layers: Vec<LayerSpec>,
    pub final_norm: Tensor,
    /// Output projection; the embedding itself when the artifact ties them.
    pub head: Tensor,
    /// Draft head for speculative decoding, when the artifact carries one.
    pub mtp: Option<MtpSpec>,
    pub eos_token_id: Option<u32>,
}
