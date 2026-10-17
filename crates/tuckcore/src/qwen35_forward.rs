//! Qwen3.5 CPU reference forward pass (Q35-1D composition + execution).
//!
//! Executes the served 24-layer hybrid stack (18 Gated DeltaNet + 6 full
//! attention; MTP/`nextn` blocks excluded) over explicit token IDs using
//! the operator library ([`crate::qwen35_ops`]), the packed KV storage
//! ([`crate::qwen35_kv`]) and the typed contract ([`crate::qwen35`]).
//! Weight matrices are read from the artifact rows on demand (`pread` +
//! dequantize); norms and other small vectors are cached at open.
//!
//! This is the numerical reference: CPU, f32 activations, declared KV
//! precision, f32 state. It is deliberately not wired into the loader —
//! the CLI/worker path keeps refusing Qwen3.5 execution until the full
//! Q35-1 checklist passes.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use tuckmill_manifest::adapter::AdapterRegistry;
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::gguf::{self, GgufTensor, Limits};

use crate::qwen35::{resolve, Qwen35AttentionTensors, Qwen35Config, Qwen35Layout};
use crate::qwen35_kv::{KvPrecision, Qwen35KvCache};
use crate::qwen35_ops::{
    apply_sigmoid_gate, gqa_cache_attention, l2_normalize_head, partial_rope_in_place,
    residual_add_in_place, rms_norm_weighted_in_place, rope_inv_freq_partial, sigmoid, silu,
    softplus, swiglu_in_place, ConvState, GdnState, GqaGeometry,
};
use crate::tensor;

/// Row geometry of one artifact tensor, plus its absolute file offset.
#[derive(Debug, Clone, Copy)]
struct TensorMeta {
    offset: u64,
    type_id: u32,
    in_dim: u64,
    out_dim: u64,
    row_bytes: u64,
    byte_size: u64,
}

impl TensorMeta {
    fn from_tensor(t: &GgufTensor, data_start: u64, file_len: u64) -> Result<Self> {
        let (in_dim, out_dim) = match t.dims.len() {
            1 => (t.dims[0], 1),
            2 => (t.dims[0], t.dims[1]),
            n => {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!(
                        "tensor {:?} has {n} dims; the qwen35 reference reads 1-D and 2-D tensors only",
                        t.name
                    ),
                ));
            }
        };
        if in_dim == 0 || out_dim == 0 || t.byte_size == 0 {
            return Err(TuckmillError::new(
                ErrorCode::InvalidArtifact,
                format!("tensor {:?} is empty", t.name),
            ));
        }
        if !t.byte_size.is_multiple_of(out_dim) {
            return Err(TuckmillError::new(
                ErrorCode::InvalidArtifact,
                format!(
                    "tensor {:?} bytes {} are not a whole number of rows of {out_dim}",
                    t.name, t.byte_size
                ),
            ));
        }
        let offset = data_start
            .checked_add(t.offset)
            .filter(|start| {
                start
                    .checked_add(t.byte_size)
                    .is_some_and(|e| e <= file_len)
            })
            .ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!("tensor {:?} lies outside the artifact", t.name),
                )
            })?;
        Ok(Self {
            offset,
            type_id: t.type_id,
            in_dim,
            out_dim,
            row_bytes: t.byte_size / out_dim,
            byte_size: t.byte_size,
        })
    }
}

/// Small per-layer tensors cached at open (all 1-D weights are tiny).
#[derive(Debug, Clone)]
struct LayerVectors {
    attn_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    /// Full-attention layers only.
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// Gated DeltaNet layers only.
    ssm_norm: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_dt: Vec<f32>,
    /// `ssm_conv1d` in GGUF memory order (`[kernel, channels]`, taps
    /// fastest), 1-D per channel.
    conv_w: Vec<f32>,
}

/// Per-served-layer execution state (only the relevant kind is present).
enum LayerState {
    Linear { gdn: GdnState, conv: ConvState },
    Full { kv_slot: usize },
}

/// The reference model: config, layout, file access, state and scratch.
pub struct Qwen35Model {
    cfg: Qwen35Config,
    layout: Qwen35Layout,
    file: File,
    meta: Vec<TensorMeta>,
    vectors: Vec<LayerVectors>,
    output_norm: Vec<f32>,
    rope_inv: Vec<f32>,
    cache: Qwen35KvCache,
    state: Vec<LayerState>,
    positions: usize,
    scratch_bound: u64,
    /// A step that failed part-way may have advanced some layer state and
    /// not others; such a session is invalid until reset. Interrupted
    /// state is never reused (Q35-1E).
    session_valid: bool,
}

/// Preflight byte budget of the reference executor for one admitted
/// context: the weights payload (streamed row-on-demand, never resident),
/// the packed KV state, the recurrent/convolution state and a
/// conservative bound of the f32 scratch the forward pass allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35Budget {
    pub weights_payload: u64,
    pub kv_state: u64,
    pub recurrent_state: u64,
    pub scratch: u64,
}

impl Qwen35Budget {
    pub fn state_total(&self) -> u64 {
        self.kv_state.saturating_add(self.recurrent_state)
    }
}

/// Conservative upper bound of f32 scratch the per-token forward pass
/// allocates (elements x 4 bytes), including the capacity-sized attention
/// slabs that decode the packed cache per layer.
fn scratch_bound(cfg: &Qwen35Config, capacity: u64) -> Result<u64> {
    let hidden = cfg.hidden;
    let ffn = cfg.ffn;
    let inner = cfg.gdn.inner;
    let dt = cfg.gdn.dt_rank;
    let hd = cfg.head_dim;
    let hq = cfg.heads;
    let kvh = cfg.kv_heads;
    let qkv_rows = cfg.qkv_rows();
    let elems = hidden // x
        .saturating_add(hidden) // h
        .saturating_add(hidden) // h2
        .saturating_add(hidden) // proj
        .saturating_add(qkv_rows) // projection
        .saturating_add(qkv_rows) // conv_out
        .saturating_add(inner) // z
        .saturating_add(inner) // o
        .saturating_add(2 * dt)
        .saturating_add(ffn * 2)
        .saturating_add(hidden) // ffn down
        .saturating_add(2 * hq * hd) // qg
        .saturating_add(hq * hd) // q
        .saturating_add(hq * hd) // gate
        .saturating_add(2 * kvh * hd) // k + v
        .saturating_add(hq * hd) // attn_out
        .saturating_add(hd) // probe row
        .saturating_add(2 * kvh * capacity * hd) // decoded slabs (K and V)
        .saturating_add(hidden * 2); // dequant row buffers (generous)
    elems
        .checked_mul(4)
        .ok_or_else(|| TuckmillError::new(ErrorCode::InvalidPlan, "qwen35 scratch bound overflows"))
}

/// Preflight the byte budget for a context before anything allocates.
pub fn required_bytes(
    insp: &tuckmill_manifest::gguf::GgufInspection,
    capacity: u64,
    precision: KvPrecision,
) -> Result<Qwen35Budget> {
    let registry = AdapterRegistry::builtin()?;
    let (cfg, layout) = resolve(insp, &registry)?;
    let per_token = layout
        .state
        .attention_layers
        .checked_mul(cfg.kv_heads)
        .and_then(|v| v.checked_mul(cfg.head_dim))
        .and_then(|v| v.checked_mul(2)) // key + value
        .and_then(|v| v.checked_mul(precision.element_bytes() as u64))
        .ok_or_else(|| TuckmillError::new(ErrorCode::InvalidPlan, "qwen35 KV budget overflows"))?;
    let kv_state = per_token
        .checked_mul(capacity)
        .ok_or_else(|| TuckmillError::new(ErrorCode::InvalidPlan, "qwen35 KV budget overflows"))?;
    let recurrent_state = layout.state.bytes_for(&cfg, 0, 0)?;
    let weights_payload = insp.tensors.iter().map(|t| t.byte_size).sum();
    Ok(Qwen35Budget {
        weights_payload,
        kv_state,
        recurrent_state,
        scratch: scratch_bound(&cfg, capacity)?,
    })
}

impl Qwen35Model {
    /// Open an artifact, resolve the typed contract, and allocate the
    /// admitted state: KV planes for `capacity` positions at the declared
    /// precision, plus the recurrent/convolution state of every GDN layer.
    pub fn open(path: &Path, capacity: usize, precision: KvPrecision) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            TuckmillError::new(
                ErrorCode::StorageLost,
                format!("cannot open artifact {}: {e}", path.display()),
            )
        })?;
        let len = file
            .metadata()
            .map_err(|e| {
                TuckmillError::new(
                    ErrorCode::StorageLost,
                    format!("cannot stat artifact {}: {e}", path.display()),
                )
            })?
            .len();
        let mut reader = &file;
        let insp = gguf::inspect_reader(&mut reader, len, &Limits::default(), None)?;
        Self::open_with_handle(&file, &insp, capacity, precision)
    }

    /// Build from an already-verified file handle and inspection: the
    /// executor's load path digests this exact handle, so the bytes that
    /// were verified are the bytes every weight read decodes (never a
    /// pathname reopen).
    pub fn open_with_handle(
        file: &File,
        insp: &tuckmill_manifest::gguf::GgufInspection,
        capacity: usize,
        precision: KvPrecision,
    ) -> Result<Self> {
        let registry = AdapterRegistry::builtin()?;
        let (cfg, layout) = resolve(insp, &registry)?;

        let mut meta = Vec::with_capacity(insp.tensors.len());
        for t in &insp.tensors {
            meta.push(TensorMeta::from_tensor(t, insp.data_start, insp.file_len)?);
        }

        // The QKV split requires groups x state_size == inner_size (the
        // pinned model and the fixture satisfy this; a variant that does
        // not needs an explicit split rule before it can execute).
        if cfg.gdn.groups * cfg.gdn.state_size != cfg.gdn.inner {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "qwen35: QKV split geometry (groups {} x state_size {}) does not equal inner_size {}",
                    cfg.gdn.groups, cfg.gdn.state_size, cfg.gdn.inner
                ),
            ));
        }
        let mut model = Self {
            rope_inv: rope_inv_freq_partial(cfg.rope_dims as usize, cfg.rope_base),
            cache: Qwen35KvCache::new(
                layout.state.attention_layers as usize,
                cfg.kv_heads as usize,
                cfg.head_dim as usize,
                capacity,
                precision,
            )?,
            scratch_bound: scratch_bound(&cfg, capacity as u64)?,
            session_valid: true,
            cfg,
            layout,
            file: file.try_clone().map_err(|e| {
                TuckmillError::new(
                    ErrorCode::StorageLost,
                    format!("cannot clone the verified artifact handle: {e}"),
                )
            })?,
            meta,
            vectors: Vec::new(),
            output_norm: Vec::new(),
            state: Vec::new(),
            positions: 0,
        };
        model.load_vectors()?;
        Ok(model)
    }

    /// Packed KV bytes currently allocated.
    pub fn kv_bytes(&self) -> u64 {
        self.cache.bytes()
    }

    /// Delta-rule plus convolution state bytes currently allocated.
    pub fn recurrent_bytes(&self) -> u64 {
        self.state
            .iter()
            .map(|state| match state {
                LayerState::Linear { gdn, conv } => gdn.state_bytes() + conv.state_bytes(),
                LayerState::Full { .. } => 0,
            })
            .sum()
    }

    /// All session state bytes (KV + recurrent), the number the plan's
    /// KV-state reservation must cover.
    pub fn state_bytes(&self) -> u64 {
        self.kv_bytes() + self.recurrent_bytes()
    }

    /// Conservative scratch bound of this instance (matches
    /// [`required_bytes`] for the same geometry).
    pub fn scratch_bound_bytes(&self) -> u64 {
        self.scratch_bound
    }

    fn ensure_valid(&self) -> Result<()> {
        if !self.session_valid {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "qwen35 session was interrupted and its partial state was discarded",
            )
            .with_action(
                "Reset the session before continuing; interrupted recurrent state is never reused.",
            ));
        }
        Ok(())
    }

    fn finish(&mut self, result: Result<()>) -> Result<()> {
        if result.is_err() {
            self.session_valid = false;
        }
        result
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    /// Resolved layout (tensor indices per served layer).
    pub fn layout(&self) -> &Qwen35Layout {
        &self.layout
    }

    /// Raw stored bytes of one tensor, exactly as they are on disk (the
    /// device client uploads these verbatim; no re-quantization).
    pub fn raw_tensor_bytes(&self, tensor_index: usize) -> Result<Vec<u8>> {
        let m = *self.meta_of(tensor_index)?;
        let mut bytes = Vec::new();
        self.read_bytes(m.offset, m.byte_size as usize, &mut bytes)?;
        Ok(bytes)
    }

    /// Stored type id of one tensor.
    pub fn tensor_type_id(&self, tensor_index: usize) -> Result<u32> {
        Ok(self.meta_of(tensor_index)?.type_id)
    }

    /// `(in_dim, out_dim)` of one tensor (1-D tensors read as `(n, 1)`).
    pub fn tensor_dims(&self, tensor_index: usize) -> Result<(u64, u64)> {
        let m = self.meta_of(tensor_index)?;
        Ok((m.in_dim, m.out_dim))
    }

    /// Positions consumed so far (the next token's position).
    pub fn positions(&self) -> usize {
        self.positions
    }

    /// Clear all session state: packed KV planes, delta-rule and
    /// convolution state. A request reset must never leak recurrent state
    /// into the next request (tested).
    pub fn reset(&mut self) {
        self.cache.reset();
        for state in &mut self.state {
            match state {
                LayerState::Linear { gdn, conv } => {
                    gdn.reset();
                    conv.reset();
                }
                LayerState::Full { .. } => {}
            }
        }
        self.positions = 0;
        self.session_valid = true;
    }

    /// FNV-1a over every stateful byte (KV planes, delta-rule state,
    /// convolution history): the state equivalence digest used by the
    /// chunked-versus-stepwise and reset tests.
    pub fn state_checksum(&self) -> u64 {
        let mix = |hash: &mut u64, bits: u64| {
            for b in bits.to_le_bytes() {
                *hash ^= b as u64;
                *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        let mut hash = self.cache.checksum();
        for state in &self.state {
            if let LayerState::Linear { gdn, conv } = state {
                for v in gdn.snapshot() {
                    mix(&mut hash, v.to_bits() as u64);
                }
                for v in conv.history() {
                    mix(&mut hash, v.to_bits() as u64);
                }
            }
        }
        hash
    }

    /// Teacher-forced prefill over explicit token IDs. Only the last
    /// token's logits (full vocab, tied head) are produced; call it once
    /// per chunk or once per token — both paths run the identical
    /// per-token arithmetic and must agree bitwise (tested).
    pub fn forward_tokens(&mut self, tokens: &[u32], logits: &mut Vec<f32>) -> Result<()> {
        self.forward_tokens_with_poll(tokens, logits, &|| false)
    }

    /// Teacher-forced prefill with a cancellation poll: `cancel` is
    /// checked before every served layer, so an interrupted step discards
    /// the session (partial recurrent state is never reused) and reports
    /// [`ErrorCode::Cancelled`].
    pub fn forward_tokens_with_poll(
        &mut self,
        tokens: &[u32],
        logits: &mut Vec<f32>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<()> {
        self.ensure_valid()?;
        if tokens.is_empty() {
            return self.finish(Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "forward needs at least one token id",
            )));
        }
        let result = (|| {
            for (i, token) in tokens.iter().enumerate() {
                if cancel() {
                    return Err(cancelled_error());
                }
                let last = i + 1 == tokens.len();
                if last {
                    self.forward_one(*token, Some(logits), Some(cancel), None)?;
                } else {
                    self.forward_one(*token, None, Some(cancel), None)?;
                }
            }
            Ok(())
        })();
        self.finish(result)
    }

    /// Single-token step returning that token's logits.
    pub fn forward_token(&mut self, token: u32, logits: &mut Vec<f32>) -> Result<()> {
        self.ensure_valid()?;
        let result = self.forward_one(token, Some(logits), None, None);
        self.finish(result)
    }

    /// Single-token step with an activation trace for verification:
    /// `trace[0]` is the embedding row, `trace[1 + l]` the hidden state
    /// after served layer `l`. Used to qualify the device layers against
    /// the oracle layer by layer (Q35-2B.3 / Q35-2C evidence).
    pub fn forward_token_traced(
        &mut self,
        token: u32,
        logits: &mut Vec<f32>,
        trace: &mut Vec<Vec<f32>>,
    ) -> Result<()> {
        self.ensure_valid()?;
        trace.clear();
        let result = self.forward_one(token, Some(logits), None, Some(trace));
        self.finish(result)
    }

    fn forward_one(
        &mut self,
        token: u32,
        logits: Option<&mut Vec<f32>>,
        cancel: Option<&dyn Fn() -> bool>,
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<()> {
        let hidden = self.cfg.hidden as usize;
        let vocab = self.cfg.vocab as usize;
        if token as u64 >= self.cfg.vocab {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!(
                    "token id {token} is outside the {}-entry vocabulary",
                    self.cfg.vocab
                ),
            ));
        }
        let pos = self.positions;
        let mut x = vec![0.0f32; hidden];
        self.embedding_row(token, &mut x)?;
        if let Some(t) = trace.as_mut() {
            t.push(x.clone());
        }

        for l in 0..self.layout.layers.len() {
            if let Some(cancel) = cancel {
                if cancel() {
                    return Err(cancelled_error());
                }
            }
            self.layer(l, &mut x, pos)?;
            if let Some(t) = trace.as_mut() {
                t.push(x.clone());
            }
        }

        let norm = self.output_norm.clone();
        rms_norm_weighted_in_place(&mut x, &norm, self.cfg.rms_eps);

        if let Some(logits) = logits {
            logits.clear();
            logits.resize(vocab, 0.0);
            let tied = self.layout.token_embd.index;
            self.matvec(tied, &x, logits)?;
        }
        self.positions += 1;
        Ok(())
    }

    // --- weight access ---

    fn read_bytes(&self, offset: u64, len: usize, out: &mut Vec<u8>) -> Result<()> {
        out.clear();
        out.resize(len, 0);
        let mut done = 0usize;
        while done < len {
            let n = self
                .file
                .read_at(&mut out[done..], offset + done as u64)
                .map_err(|e| {
                    TuckmillError::new(
                        ErrorCode::StorageLost,
                        format!("artifact read failed at offset {offset}: {e}"),
                    )
                })?;
            if n == 0 {
                return Err(TuckmillError::new(
                    ErrorCode::StorageLost,
                    format!("artifact ended early at offset {}", offset + done as u64),
                ));
            }
            done += n;
        }
        Ok(())
    }

    fn meta_of(&self, tensor_index: usize) -> Result<&TensorMeta> {
        self.meta.get(tensor_index).ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::Internal,
                format!("tensor index {tensor_index} is outside the inspected range"),
            )
        })
    }

    /// Row `row` of a 2-D tensor, dequantized into `out`.
    fn read_row(&self, tensor_index: usize, row: usize, out: &mut Vec<f32>) -> Result<()> {
        let m = *self.meta_of(tensor_index)?;
        if row as u64 >= m.out_dim {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!("row {row} is outside the {}-row tensor", m.out_dim),
            ));
        }
        let mut bytes = Vec::new();
        self.read_bytes(
            m.offset + row as u64 * m.row_bytes,
            m.row_bytes as usize,
            &mut bytes,
        )?;
        out.clear();
        tensor::dequant_into(m.type_id, &bytes, out)?;
        if out.len() as u64 != m.in_dim {
            return Err(TuckmillError::new(
                ErrorCode::InvalidArtifact,
                format!(
                    "row {row} dequantized to {} values, expected {}",
                    out.len(),
                    m.in_dim
                ),
            ));
        }
        Ok(())
    }

    /// `out = W x` reading every row of the weight on demand.
    fn matvec(&self, tensor_index: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        let m = *self.meta_of(tensor_index)?;
        if x.len() as u64 != m.in_dim || out.len() as u64 != m.out_dim {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!(
                    "matvec shape mismatch: x {} (want {}), out {} (want {})",
                    x.len(),
                    m.in_dim,
                    out.len(),
                    m.out_dim
                ),
            ));
        }
        let mut bytes = Vec::new();
        let mut row = Vec::new();
        for (o, slot) in out.iter_mut().enumerate() {
            self.read_bytes(
                m.offset + o as u64 * m.row_bytes,
                m.row_bytes as usize,
                &mut bytes,
            )?;
            row.clear();
            tensor::dequant_into(m.type_id, &bytes, &mut row)?;
            let mut acc = 0.0f32;
            for (w, v) in row.iter().zip(x.iter()) {
                acc += w * v;
            }
            *slot = acc;
        }
        Ok(())
    }

    /// Whole 1-D tensor (or small vector), dequantized.
    /// Whole 1-D tensor (or small vector), dequantized (device clients
    /// upload the result for norm/conv/gate kernels).
    pub fn read_vector(&self, tensor_index: usize) -> Result<Vec<f32>> {
        let m = *self.meta_of(tensor_index)?;
        let mut bytes = Vec::new();
        self.read_bytes(m.offset, m.byte_size as usize, &mut bytes)?;
        let mut out = Vec::new();
        tensor::dequant_into(m.type_id, &bytes, &mut out)?;
        Ok(out)
    }

    fn embedding_row(&self, token: u32, out: &mut [f32]) -> Result<()> {
        let mut row = Vec::new();
        self.read_row(self.layout.token_embd.index, token as usize, &mut row)?;
        if row.len() != out.len() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidArtifact,
                format!(
                    "embedding row has {} values, hidden is {}",
                    row.len(),
                    out.len()
                ),
            ));
        }
        out.copy_from_slice(&row);
        Ok(())
    }

    fn load_vectors(&mut self) -> Result<()> {
        let mut vectors = Vec::with_capacity(self.layout.layers.len());
        let mut state = Vec::with_capacity(self.layout.layers.len());
        let mut kv_slot = 0usize;
        for layer in &self.layout.layers {
            let mut v = LayerVectors {
                attn_norm: self.read_vector(layer.attn_norm.index)?,
                post_attn_norm: self.read_vector(layer.post_attn_norm.index)?,
                q_norm: Vec::new(),
                k_norm: Vec::new(),
                ssm_norm: Vec::new(),
                ssm_a: Vec::new(),
                ssm_dt: Vec::new(),
                conv_w: Vec::new(),
            };
            let state_entry = match &layer.attention {
                Qwen35AttentionTensors::Linear {
                    a,
                    dt,
                    conv1d,
                    norm,
                    ..
                } => {
                    v.ssm_norm = self.read_vector(norm.index)?;
                    v.ssm_a = self.read_vector(a.index)?;
                    v.ssm_dt = self.read_vector(dt.index)?;
                    v.conv_w = self.read_vector(conv1d.index)?;
                    LayerState::Linear {
                        gdn: GdnState::new(
                            self.cfg.gdn.dt_rank as usize,
                            self.cfg.gdn.state_size as usize,
                            self.cfg.gdn.head_v as usize,
                        ),
                        conv: ConvState::new(
                            self.cfg.gdn.conv_kernel as usize,
                            self.cfg.conv_channels() as usize,
                        ),
                    }
                }
                Qwen35AttentionTensors::Full { q_norm, k_norm, .. } => {
                    v.q_norm = self.read_vector(q_norm.index)?;
                    v.k_norm = self.read_vector(k_norm.index)?;
                    let slot = kv_slot;
                    kv_slot += 1;
                    LayerState::Full { kv_slot: slot }
                }
            };
            vectors.push(v);
            state.push(state_entry);
        }
        self.output_norm = self.read_vector(self.layout.output_norm.index)?;
        self.vectors = vectors;
        self.state = state;
        Ok(())
    }

    // --- layers ---

    fn layer(&mut self, l: usize, x: &mut [f32], pos: usize) -> Result<()> {
        let hidden = self.cfg.hidden as usize;
        // 1. Pre-attention norm.
        let attn_norm = self.vectors[l].attn_norm.clone();
        let mut h = vec![0.0f32; hidden];
        h.copy_from_slice(x);
        rms_norm_weighted_in_place(&mut h, &attn_norm, self.cfg.rms_eps);

        // 2. Attention path (kind-specific), writing the projected
        //    output back into `x` with a residual add.
        let layer = self.layout.layers[l].clone();
        match &layer.attention {
            Qwen35AttentionTensors::Linear {
                qkv,
                gate,
                alpha,
                beta,
                out,
                ..
            } => {
                let t = GdnTensors {
                    qkv: qkv.index,
                    z: gate.index,
                    alpha: alpha.index,
                    beta: beta.index,
                    out: out.index,
                };
                self.gdn_attention(l, &h, t, x)?;
            }
            Qwen35AttentionTensors::Full {
                q, k, v, output, ..
            } => {
                let t = FullTensors {
                    q: q.index,
                    k: k.index,
                    v: v.index,
                    output: output.index,
                };
                self.full_attention(l, &h, t, pos, x)?;
            }
        }

        // 3. Post-attention norm + FFN + residual.
        let post_norm = self.vectors[l].post_attn_norm.clone();
        let mut h2 = vec![0.0f32; hidden];
        h2.copy_from_slice(x);
        rms_norm_weighted_in_place(&mut h2, &post_norm, self.cfg.rms_eps);

        let ffn = self.cfg.ffn as usize;
        let mut gate = vec![0.0f32; ffn];
        let mut up = vec![0.0f32; ffn];
        self.matvec(layer.ffn_gate.index, &h2, &mut gate)?;
        self.matvec(layer.ffn_up.index, &h2, &mut up)?;
        swiglu_in_place(&mut gate, &up);
        let mut down = vec![0.0f32; hidden];
        self.matvec(layer.ffn_down.index, &gate, &mut down)?;
        residual_add_in_place(x, &down);
        Ok(())
    }

    fn gdn_attention(&mut self, l: usize, h: &[f32], t: GdnTensors, x: &mut [f32]) -> Result<()> {
        let inner = self.cfg.gdn.inner as usize;
        let groups = self.cfg.gdn.groups as usize;
        let head_k = self.cfg.gdn.state_size as usize;
        let dt_rank = self.cfg.gdn.dt_rank as usize;

        // Projections.
        let mut qkv_mixed = vec![0.0f32; self.cfg.qkv_rows() as usize];
        self.matvec(t.qkv, h, &mut qkv_mixed)?;
        let mut z = vec![0.0f32; inner];
        self.matvec(t.z, h, &mut z)?;
        let mut alpha = vec![0.0f32; dt_rank];
        self.matvec(t.alpha, h, &mut alpha)?;
        let mut beta = vec![0.0f32; dt_rank];
        self.matvec(t.beta, h, &mut beta)?;

        // Causal convolution over the QKV stream, then SiLU.
        let mut conv_out = vec![0.0f32; qkv_mixed.len()];
        {
            let LayerState::Linear { conv, .. } = &mut self.state[l] else {
                return Err(TuckmillError::new(
                    ErrorCode::Internal,
                    "layer state kind disagrees with the resolved layer kind",
                ));
            };
            let weights = self.vectors[l].conv_w.clone();
            conv.step_sequence(&weights, &qkv_mixed, &mut conv_out);
        }
        for v in conv_out.iter_mut() {
            *v = silu(*v);
        }

        // Split Q / K / V; L2-normalize Q and K per head.
        let q_len = groups * head_k;
        let (q, rest) = conv_out.split_at_mut(q_len);
        let (k, v) = rest.split_at_mut(q_len);
        for head in 0..groups {
            l2_normalize_head(&mut q[head * head_k..(head + 1) * head_k], self.cfg.rms_eps);
            l2_normalize_head(&mut k[head * head_k..(head + 1) * head_k], self.cfg.rms_eps);
        }

        // Decay gate: exp(ssm_a * softplus(alpha + dt_bias)), per head.
        let mut sigmoid_beta = vec![0.0f32; dt_rank];
        let mut gate = vec![0.0f32; dt_rank];
        {
            let vecs = &self.vectors[l];
            for hd in 0..dt_rank {
                sigmoid_beta[hd] = sigmoid(beta[hd]);
                gate[hd] = vecs.ssm_a[hd] * softplus(alpha[hd] + vecs.ssm_dt[hd]);
            }
        }

        // Delta rule.
        let mut o = vec![0.0f32; inner];
        {
            let LayerState::Linear { gdn, .. } = &mut self.state[l] else {
                return Err(TuckmillError::new(
                    ErrorCode::Internal,
                    "layer state kind disagrees with the resolved layer kind",
                ));
            };
            gdn.step(q, k, v, &gate, &sigmoid_beta, &mut o);
        }

        // Gated normalization: norm per value head with the ssm_norm
        // weights, times SiLU(z).
        let head_v = self.cfg.gdn.head_v as usize;
        {
            let vecs = &self.vectors[l];
            for hv in 0..self.cfg.gdn.dt_rank as usize {
                let row = &mut o[hv * head_v..(hv + 1) * head_v];
                rms_norm_weighted_in_place(row, &vecs.ssm_norm, self.cfg.rms_eps);
                for (i, val) in row.iter_mut().enumerate() {
                    *val *= silu(z[hv * head_v + i]);
                }
            }
        }

        // Output projection + residual.
        let mut proj = vec![0.0f32; self.cfg.hidden as usize];
        self.matvec(t.out, &o, &mut proj)?;
        residual_add_in_place(x, &proj);
        Ok(())
    }

    fn full_attention(
        &mut self,
        l: usize,
        h: &[f32],
        t: FullTensors,
        pos: usize,
        x: &mut [f32],
    ) -> Result<()> {
        let heads_q = self.cfg.heads as usize;
        let kv_heads = self.cfg.kv_heads as usize;
        let head_dim = self.cfg.head_dim as usize;

        // Joint Q (+ gate) projection, K and V projections.
        let mut qg = vec![0.0f32; heads_q * head_dim * 2];
        self.matvec(t.q, h, &mut qg)?;
        let mut k = vec![0.0f32; kv_heads * head_dim];
        self.matvec(t.k, h, &mut k)?;
        let mut v = vec![0.0f32; kv_heads * head_dim];
        self.matvec(t.v, h, &mut v)?;

        // Per-head Q/K norm + partial RoPE.
        let vecs = self.vectors[l].clone();
        let mut q = vec![0.0f32; heads_q * head_dim];
        let mut gate = vec![0.0f32; heads_q * head_dim];
        for head in 0..heads_q {
            let src = &qg[head * head_dim * 2..(head + 1) * head_dim * 2];
            let qrow = &mut q[head * head_dim..(head + 1) * head_dim];
            qrow.copy_from_slice(&src[..head_dim]);
            gate[head * head_dim..(head + 1) * head_dim].copy_from_slice(&src[head_dim..]);
            rms_norm_weighted_in_place(qrow, &vecs.q_norm, self.cfg.rms_eps);
            partial_rope_in_place(qrow, pos, &self.rope_inv);
        }
        for head in 0..kv_heads {
            let krow = &mut k[head * head_dim..(head + 1) * head_dim];
            rms_norm_weighted_in_place(krow, &vecs.k_norm, self.cfg.rms_eps);
            partial_rope_in_place(krow, pos, &self.rope_inv);
        }

        // Store the new K/V rows, then decode the plane for attention.
        let kv_slot = match &self.state[l] {
            LayerState::Full { kv_slot } => *kv_slot,
            LayerState::Linear { .. } => {
                return Err(TuckmillError::new(
                    ErrorCode::Internal,
                    "layer state kind disagrees with the resolved layer kind",
                ));
            }
        };
        self.cache.write(kv_slot, pos, &k, &v)?;

        let seq = pos + 1;
        let geometry = GqaGeometry {
            heads_q,
            heads_kv: kv_heads,
            head_dim,
        };
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut k_slab = vec![0.0f32; kv_heads * seq * head_dim];
        let mut v_slab = vec![0.0f32; kv_heads * seq * head_dim];
        let mut row = vec![0.0f32; head_dim];
        for head in 0..kv_heads {
            for t in 0..seq {
                let base = (head * seq + t) * head_dim;
                self.cache.key_row(kv_slot, head, t, &mut row)?;
                k_slab[base..base + head_dim].copy_from_slice(&row);
                self.cache.value_row(kv_slot, head, t, &mut row)?;
                v_slab[base..base + head_dim].copy_from_slice(&row);
            }
        }
        let mut attn_out = vec![0.0f32; heads_q * head_dim];
        gqa_cache_attention(&q, &k_slab, &v_slab, geometry, seq, scale, &mut attn_out)?;

        // Output gate, projection, residual.
        apply_sigmoid_gate(&mut attn_out, &gate);
        let mut proj = vec![0.0f32; self.cfg.hidden as usize];
        self.matvec(t.output, &attn_out, &mut proj)?;
        residual_add_in_place(x, &proj);
        Ok(())
    }
}

fn cancelled_error() -> TuckmillError {
    TuckmillError::new(
        ErrorCode::Cancelled,
        "qwen35 forward cancelled; the session was discarded",
    )
    .with_action("Partial recurrent state is never reused; reset the session before continuing.")
}

/// Tensor indices of one Gated DeltaNet layer (into the inspection).
#[derive(Debug, Clone, Copy)]
struct GdnTensors {
    qkv: usize,
    z: usize,
    alpha: usize,
    beta: usize,
    out: usize,
}

/// Tensor indices of one full-attention layer (into the inspection).
#[derive(Debug, Clone, Copy)]
struct FullTensors {
    q: usize,
    k: usize,
    v: usize,
    output: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    fn fixture_model(tokens_hint: usize) -> (tempfile::TempDir, Qwen35Model) {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen35());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q35.gguf");
        std::fs::write(&path, &bytes).unwrap();
        let model = Qwen35Model::open(&path, tokens_hint, KvPrecision::F16).unwrap();
        (dir, model)
    }

    #[test]
    fn fixture_forward_produces_finite_logits() {
        let (_dir, mut model) = fixture_model(16);
        let mut logits = Vec::new();
        model.forward_tokens(&[1, 2, 3], &mut logits).unwrap();
        assert_eq!(logits.len(), 512, "vocab-sized logits");
        assert!(logits.iter().all(|v| v.is_finite()));
        assert!(logits.iter().any(|v| *v != 0.0), "logits are not all zero");
        assert_eq!(model.positions(), 3);
    }

    #[test]
    fn chunked_and_stepwise_execution_match_bitwise() {
        let (_dir_a, mut chunked) = fixture_model(16);
        let (_dir_b, mut stepwise) = fixture_model(16);
        let tokens = [7u32, 11, 13, 17];

        let mut chunk_logits = Vec::new();
        chunked.forward_tokens(&tokens, &mut chunk_logits).unwrap();

        let mut step_logits = Vec::new();
        for t in &tokens {
            stepwise.forward_token(*t, &mut step_logits).unwrap();
        }

        assert_eq!(chunk_logits, step_logits, "logits agree bitwise");
        assert_eq!(
            chunked.state_checksum(),
            stepwise.state_checksum(),
            "recurrent and KV state agree bitwise"
        );
    }

    #[test]
    fn reset_clears_every_stateful_byte() {
        let (_dir, mut model) = fixture_model(16);
        let mut logits_a = Vec::new();
        model.forward_tokens(&[3, 1, 4], &mut logits_a).unwrap();
        let checksum_after_first = model.state_checksum();

        model.reset();
        assert_eq!(model.positions(), 0);
        let (_dir_fresh, fresh) = fixture_model(16);
        assert_eq!(
            model.state_checksum(),
            fresh.state_checksum(),
            "a reset model equals a fresh one"
        );
        let mut logits_b = Vec::new();
        model.forward_tokens(&[3, 1, 4], &mut logits_b).unwrap();
        assert_eq!(logits_a, logits_b, "no state leaks across reset");
        assert_eq!(model.state_checksum(), checksum_after_first);
    }

    #[test]
    fn forward_is_deterministic() {
        let (_dir_a, mut a) = fixture_model(8);
        let (_dir_b, mut b) = fixture_model(8);
        let mut la = Vec::new();
        let mut lb = Vec::new();
        a.forward_tokens(&[5, 9], &mut la).unwrap();
        b.forward_tokens(&[5, 9], &mut lb).unwrap();
        assert_eq!(la, lb);
    }

    /// The preflight budget equals what the model actually allocates for
    /// state and scratch (planner-versus-allocation equality at the model
    /// level).
    #[test]
    fn required_bytes_match_actual_allocations() {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen35());
        let (_dir, model) = fixture_model(12);
        let mut cursor = std::io::Cursor::new(bytes.clone());
        let insp = tuckmill_manifest::gguf::inspect_reader(
            &mut cursor,
            bytes.len() as u64,
            &Limits::default(),
            None,
        )
        .unwrap();
        let budget = required_bytes(&insp, 12, KvPrecision::F16).unwrap();
        assert_eq!(budget.kv_state, model.kv_bytes());
        assert_eq!(budget.recurrent_state, model.recurrent_bytes());
        assert_eq!(budget.scratch, model.scratch_bound_bytes());
        assert_eq!(budget.state_total(), model.state_bytes());
        let expected_payload: u64 = insp.tensors.iter().map(|t| t.byte_size).sum();
        assert_eq!(budget.weights_payload, expected_payload);
    }

    /// A step that fails part-way invalidates the session; nothing of the
    /// partial state is reused until reset.
    #[test]
    fn interrupted_steps_invalidate_the_session() {
        let (_dir, mut model) = fixture_model(2);
        let mut logits = Vec::new();
        let err = model.forward_tokens(&[1, 2, 3], &mut logits).unwrap_err();
        assert_eq!(
            err.code(),
            ErrorCode::InvalidPlan,
            "capacity exhausted mid-step"
        );

        let err = model.forward_tokens(&[1], &mut logits).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidRequest);
        assert!(err.message().contains("interrupted"), "{}", err.message());
        assert!(err.action().contains("Reset"), "{err}");

        model.reset();
        model.forward_tokens(&[1, 2], &mut logits).unwrap();
        assert_eq!(model.positions(), 2);
    }

    /// Cancellation is polled before every served layer; a cancelled run
    /// discards the session and a reset restores bitwise-identical output.
    #[test]
    fn cancellation_invalidates_and_reset_restores() {
        let (_dir_a, mut baseline) = fixture_model(8);
        let mut expected = Vec::new();
        baseline.forward_tokens(&[5, 6, 7], &mut expected).unwrap();

        let (_dir_b, mut model) = fixture_model(8);
        let polls = std::cell::Cell::new(0usize);
        let cancel = || {
            polls.set(polls.get() + 1);
            polls.get() > 3
        };
        let mut logits = Vec::new();
        let err = model
            .forward_tokens_with_poll(&[5, 6, 7], &mut logits, &cancel)
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::Cancelled);

        let err = model.forward_token(5, &mut logits).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidRequest, "session discarded");

        model.reset();
        let mut restored = Vec::new();
        model.forward_tokens(&[5, 6, 7], &mut restored).unwrap();
        assert_eq!(
            restored, expected,
            "reset restores bitwise-identical output"
        );
    }

    fn real_artifact_path() -> Option<std::path::PathBuf> {
        let p = std::env::var("TUCKMILL_QWEN35_GGUF").ok()?;
        let p = std::path::PathBuf::from(p);
        p.exists().then_some(p)
    }

    /// Machine-local Q35-1D gate: the complete served stack executes on
    /// the pinned artifact and the last-token logits are finite. Run
    /// with --release for speed; debug builds are slow (scalar rows).
    #[test]
    fn real_artifact_forward_when_available() {
        let Some(path) = real_artifact_path() else {
            eprintln!("skipping: TUCKMILL_QWEN35_GGUF not set (pinned Qwen3.5-0.8B Q8_0)");
            return;
        };
        // Static budget on the pinned geometry, before any allocation.
        let file = std::fs::File::open(&path).unwrap();
        let len = file.metadata().unwrap().len();
        let mut reader = &file;
        let insp =
            tuckmill_manifest::gguf::inspect_reader(&mut reader, len, &Limits::default(), None)
                .unwrap();
        let budget = required_bytes(&insp, 2048, KvPrecision::F16).unwrap();
        assert_eq!(budget.kv_state, 25_165_824);
        assert_eq!(budget.recurrent_state, 20_201_472);
        assert_eq!(budget.state_total(), 45_367_296);

        let mut model = Qwen35Model::open(&path, 16, KvPrecision::F16).unwrap();
        assert_eq!(model.config().layers, 24);
        assert_eq!(model.kv_bytes(), 6 * 2 * 256 * 2 * 2 * 16);
        assert_eq!(model.recurrent_bytes(), 20_201_472);
        let tokens = [9707u32, 11, 1879, 1000, 358, 1079, 13, 1000];
        let mut logits = Vec::new();
        model.forward_tokens(&tokens, &mut logits).unwrap();
        assert_eq!(logits.len(), 248320);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits finite on the real stack"
        );
        assert!(
            logits.iter().any(|v| v.is_finite() && *v != 0.0),
            "logits carry signal"
        );
    }
}
