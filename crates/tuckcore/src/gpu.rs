//! TuckCore's Metal forward path (Gate C, C2): the complete dense-Qwen3
//! decode executed on the GPU through `tuckmill-metal`, with the CPU scalar
//! executor as the registered numerical reference.
//!
//! Scope and honesty: weights upload once into resident GPU buffers in
//! their artifact-packed form (Q8_0 or F16 bytes; norms kept at their
//! stored precision), so GPU weight bytes equal the artifact payload and
//! every allocation can be checked against the admitted plan before it is
//! accepted. One token step encodes the whole transformer into a single
//! grouped command buffer (one completion), then downloads the vocab logits
//! for the existing CPU sampler — sampling and output decoding are
//! deliberately unchanged (neither is the bottleneck, per the C2 plan).
//!
//! Numerical contract: f32 activations with exact dequantization (the
//! `oracle-f32` profile). KV entries are f32, or f16-representable f32 when
//! the declared cache precision asks for it — the same contract the CPU
//! executor honors. Comparisons against `llama-cpu` arithmetic remain a
//! separate qualification, never automatic bit-equivalence.

use std::os::unix::fs::FileExt;
use std::path::Path;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::gguf::{GgufInspection, GgufTensor};
use tuckmill_metal::{MetalBuffer, MetalContext};

use crate::forward::KvPrecision;
use crate::model::Qwen3Config;
use crate::model::TensorNames;

const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q8_0: u32 = 8;

fn gpu_err(message: String) -> TuckmillError {
    TuckmillError::new(ErrorCode::InvalidPlan, message).with_action(
        "Fix the artifact, execution options, or admitted plan; the GPU path never falls back silently.",
    )
}

fn cancelled_error() -> TuckmillError {
    TuckmillError::new(
        ErrorCode::Cancelled,
        "generation cancelled at a safe boundary",
    )
}

/// A weight matrix resident on the GPU, in its artifact-packed form.
enum GpuMatrix {
    Q8_0 {
        buffer: MetalBuffer,
        rows: u64,
        cols: u64,
    },
    F16 {
        buffer: MetalBuffer,
        rows: u64,
        cols: u64,
    },
}

impl GpuMatrix {
    fn shape(&self) -> (u64, u64) {
        match self {
            Self::Q8_0 { rows, cols, .. } | Self::F16 { rows, cols, .. } => (*rows, *cols),
        }
    }
}

/// A norm weight table (length = cols) at its stored precision.
enum GpuNorm {
    F32(MetalBuffer),
    F16(MetalBuffer),
}

struct GpuLayer {
    attn_norm: GpuNorm,
    q_norm: GpuNorm,
    k_norm: GpuNorm,
    ffn_norm: GpuNorm,
    attn_q: GpuMatrix,
    attn_k: GpuMatrix,
    attn_v: GpuMatrix,
    attn_output: GpuMatrix,
    ffn_gate: GpuMatrix,
    ffn_up: GpuMatrix,
    ffn_down: GpuMatrix,
}

struct Workspace {
    x: MetalBuffer,
    normed: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attn_out: MetalBuffer,
    oproj: MetalBuffer,
    gate: MetalBuffer,
    up: MetalBuffer,
    silu: MetalBuffer,
    down: MetalBuffer,
    logits: MetalBuffer,
    attn_scratch: MetalBuffer,
}

/// Resident GPU model: all weights plus reusable step workspaces for the
/// admitted context. One per loaded worker.
pub struct MetalForward {
    ctx: MetalContext,
    cfg: Qwen3Config,
    precision: KvPrecision,
    token_embd: GpuMatrix,
    /// LM head: a separate `output.weight` when present, else the tied
    /// embedding matrix.
    lm_head: LmHead,
    output_norm: GpuNorm,
    inv_freq: MetalBuffer,
    layers: Vec<GpuLayer>,
    ws: Workspace,
    weights_bytes: u64,
    workspace_bytes: u64,
}

enum LmHead {
    Tied,
    Separate(GpuMatrix),
}

impl MetalForward {
    /// Resident GPU weight bytes (the payload) — planner-checkable before
    /// the step path allocates anything else.
    pub fn weights_bytes(&self) -> u64 {
        self.weights_bytes
    }

    /// Reusable workspace bytes (staging included).
    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    /// KV bytes for a capacity, exactly matching `KvState::bytes_for`
    /// (f32 entries; f16 precision stores f16-representable f32 values).
    pub fn kv_bytes_for(cfg: &Qwen3Config, cap: u64) -> u64 {
        crate::forward::KvState::bytes_for(cfg, cap)
    }

    pub fn ctx(&self) -> &MetalContext {
        &self.ctx
    }

    /// Build the resident GPU model from the artifact payload. All shapes
    /// and types are validated first; anything unsupported is refused with
    /// a typed error, never approximated on the GPU.
    pub fn open(
        artifact_path: &Path,
        inspection: &GgufInspection,
        names: &TensorNames,
        cfg: &Qwen3Config,
        precision: KvPrecision,
        context_cap: u64,
    ) -> Result<Self> {
        let probe = tuckmill_metal::probe();
        if !probe.available {
            return Err(TuckmillError::new(
                ErrorCode::Unavailable,
                format!(
                    "Metal execution was selected but no Metal device is available ({})",
                    probe.info
                ),
            )
            .with_action("Run on a Metal-capable Mac or select --kernels neon/scalar."));
        }
        let ctx = MetalContext::open()?;
        let file = std::fs::File::open(artifact_path).map_err(|e| {
            TuckmillError::new(
                ErrorCode::Io,
                format!(
                    "cannot open artifact payload {}: {e}",
                    artifact_path.display()
                ),
            )
        })?;

        let read = |tensor: &GgufTensor| -> Result<Vec<u8>> {
            let start = inspection
                .data_start
                .checked_add(tensor.offset)
                .ok_or_else(|| gpu_err("tensor offset overflows the artifact".into()))?;
            let size = usize::try_from(tensor.byte_size)
                .map_err(|_| gpu_err("tensor byte size does not fit memory".into()))?;
            let mut bytes = vec![0u8; size];
            file.read_exact_at(&mut bytes, start).map_err(|e| {
                TuckmillError::new(
                    ErrorCode::Io,
                    format!("reading tensor {} at {start}: {e}", tensor.name),
                )
            })?;
            Ok(bytes)
        };
        let find = |name: &str| -> Result<&GgufTensor> {
            inspection
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| {
                    gpu_err(format!(
                        "artifact has no tensor {name}; the Metal path refuses incomplete models"
                    ))
                })
        };
        let upload_matrix = |name: &str| -> Result<GpuMatrix> {
            let tensor = find(name)?;
            // GGUF dims are [in, out]; a matvec maps in -> out.
            if tensor.dims.len() != 2 {
                return Err(gpu_err(format!(
                    "tensor {name} is {}-dimensional; matrices are required",
                    tensor.dims.len()
                )));
            }
            let cols = tensor.dims[0];
            let rows = tensor.dims[1];
            let bytes = read(tensor)?;
            let expected = match tensor.type_id {
                GGML_TYPE_Q8_0 => rows * (cols / 32) * 34,
                GGML_TYPE_F16 => rows * cols * 2,
                other => {
                    return Err(TuckmillError::new(
                        ErrorCode::UnsupportedQuantization,
                        format!(
                            "tensor {name} has type {other}; the Metal path implements Q8_0 and F16 weights"
                        ),
                    )
                    .with_action(
                        "Use a Q8_0 or F16 artifact for the GPU path, or run the CPU kernels.",
                    ))
                }
            };
            if (cols % 32 != 0 && tensor.type_id == GGML_TYPE_Q8_0)
                || bytes.len() as u64 != expected
            {
                return Err(gpu_err(format!(
                    "tensor {name} shape {}x{} does not match its {} payload bytes",
                    rows,
                    cols,
                    bytes.len()
                )));
            }
            let buffer = ctx.buffer(bytes.len(), true)?;
            buffer.upload(0, &bytes)?;
            Ok(match tensor.type_id {
                GGML_TYPE_Q8_0 => GpuMatrix::Q8_0 { buffer, rows, cols },
                _ => GpuMatrix::F16 { buffer, rows, cols },
            })
        };
        let upload_norm = |name: &str, cols: u64| -> Result<GpuNorm> {
            let tensor = find(name)?;
            let bytes = read(tensor)?;
            let expected = match tensor.type_id {
                0 => cols * 4, // F32
                GGML_TYPE_F16 => cols * 2,
                other => {
                    return Err(gpu_err(format!(
                        "norm {name} has type {other}; F32 or F16 required"
                    )))
                }
            };
            if bytes.len() as u64 != expected {
                return Err(gpu_err(format!(
                    "norm {name} holds {} bytes, expected {expected}",
                    bytes.len()
                )));
            }
            let buffer = ctx.buffer(bytes.len(), true)?;
            buffer.upload(0, &bytes)?;
            Ok(match tensor.type_id {
                0 => GpuNorm::F32(buffer),
                _ => GpuNorm::F16(buffer),
            })
        };

        // ---- weights ----
        let token_embd = upload_matrix(&names.token_embd)?;
        let lm_head = match &names.output {
            Some(name) => LmHead::Separate(upload_matrix(name)?),
            None => LmHead::Tied,
        };
        let output_norm = upload_norm(&names.output_norm, cfg.hidden)?;
        let mut layers = Vec::with_capacity(names.attn_q.len());
        for i in 0..names.attn_q.len() {
            layers.push(GpuLayer {
                attn_norm: upload_norm(&names.attn_norm[i], cfg.hidden)?,
                q_norm: upload_norm(&names.attn_q_norm[i], cfg.key_length)?,
                k_norm: upload_norm(&names.attn_k_norm[i], cfg.key_length)?,
                ffn_norm: upload_norm(&names.ffn_norm[i], cfg.hidden)?,
                attn_q: upload_matrix(&names.attn_q[i])?,
                attn_k: upload_matrix(&names.attn_k[i])?,
                attn_v: upload_matrix(&names.attn_v[i])?,
                attn_output: upload_matrix(&names.attn_output[i])?,
                ffn_gate: upload_matrix(&names.ffn_gate[i])?,
                ffn_up: upload_matrix(&names.ffn_up[i])?,
                ffn_down: upload_matrix(&names.ffn_down[i])?,
            });
        }
        if layers.len() as u64 != cfg.layers {
            return Err(gpu_err(format!(
                "resolved {} layers but the config declares {}",
                layers.len(),
                cfg.layers
            )));
        }

        // Shape audit against the config: the GPU path never estimates.
        let expect = |what: &str, got: (u64, u64), want: (u64, u64)| -> Result<()> {
            if got != want {
                return Err(gpu_err(format!(
                    "{what} is {}x{} but the config requires {}x{}",
                    got.0, got.1, want.0, want.1
                )));
            }
            Ok(())
        };
        expect("token_embd", token_embd.shape(), (cfg.vocab, cfg.hidden))?;
        if let LmHead::Separate(m) = &lm_head {
            expect("output", m.shape(), (cfg.vocab, cfg.hidden))?;
        }
        for (l, layer) in layers.iter().enumerate() {
            let q_width = cfg.heads * cfg.key_length;
            let kv_width = cfg.kv_heads * cfg.key_length;
            expect(
                &format!("blk.{l}.attn_q"),
                layer.attn_q.shape(),
                (q_width, cfg.hidden),
            )?;
            expect(
                &format!("blk.{l}.attn_k"),
                layer.attn_k.shape(),
                (kv_width, cfg.hidden),
            )?;
            expect(
                &format!("blk.{l}.attn_v"),
                layer.attn_v.shape(),
                (kv_width, cfg.hidden),
            )?;
            expect(
                &format!("blk.{l}.attn_output"),
                layer.attn_output.shape(),
                (cfg.hidden, q_width),
            )?;
            expect(
                &format!("blk.{l}.ffn_gate"),
                layer.ffn_gate.shape(),
                (cfg.ffn, cfg.hidden),
            )?;
            expect(
                &format!("blk.{l}.ffn_up"),
                layer.ffn_up.shape(),
                (cfg.ffn, cfg.hidden),
            )?;
            expect(
                &format!("blk.{l}.ffn_down"),
                layer.ffn_down.shape(),
                (cfg.hidden, cfg.ffn),
            )?;
        }
        if cfg.key_length != cfg.value_length {
            return Err(gpu_err(
                "asymmetric key/value lengths are not implemented on the GPU path".into(),
            ));
        }

        // Exact resident weight bytes: the model tensors, byte-for-byte the
        // artifact payload (norms included at their stored precision).
        let weights_bytes = ctx.allocated_bytes();
        // The rope frequency table is a small constant of the step path
        // (not a weight): it is charged to the workspace class.
        let inv_freq_bytes: Vec<u8> = crate::forward::rope_inv_freq(cfg)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let inv_freq = ctx.buffer(inv_freq_bytes.len(), true)?;
        inv_freq.upload(0, &inv_freq_bytes)?;

        // ---- workspaces (reused every step) ----
        let hidden = cfg.hidden as usize;
        let q_width = (cfg.heads * cfg.key_length) as usize;
        let kv_width = (cfg.kv_heads * cfg.key_length) as usize;
        let ffn = cfg.ffn as usize;
        let cap = context_cap as usize;
        let f32b = |n: usize| n * 4;
        let ws = Workspace {
            x: ctx.buffer(f32b(hidden), true)?,
            normed: ctx.buffer(f32b(hidden), true)?,
            q: ctx.buffer(f32b(q_width), true)?,
            k: ctx.buffer(f32b(kv_width), true)?,
            v: ctx.buffer(f32b(kv_width), true)?,
            attn_out: ctx.buffer(f32b(q_width), true)?,
            oproj: ctx.buffer(f32b(hidden), true)?,
            gate: ctx.buffer(f32b(ffn), true)?,
            up: ctx.buffer(f32b(ffn), true)?,
            silu: ctx.buffer(f32b(ffn), true)?,
            down: ctx.buffer(f32b(hidden), true)?,
            logits: ctx.buffer(f32b(cfg.vocab as usize), true)?,
            attn_scratch: ctx.buffer(f32b(cfg.heads as usize * cap), true)?,
        };
        let workspace_bytes = ctx.allocated_bytes() - weights_bytes;

        Ok(Self {
            ctx,
            cfg: cfg.clone(),
            precision,
            token_embd,
            lm_head,
            output_norm,
            inv_freq,
            layers,
            ws,
            weights_bytes,
            workspace_bytes,
        })
    }

    /// Total live GPU bytes (diagnostics/plan acknowledgement).
    pub fn allocated_bytes(&self) -> u64 {
        self.ctx.allocated_bytes()
    }

    /// Per-session KV: one k and v buffer per layer at the requested
    /// capacity. Fresh per request — sessions never share state.
    pub fn new_kv(&self, cap: u64) -> Result<MetalKv> {
        let per_layer = (self.cfg.kv_heads * cap * self.cfg.key_length * 4) as usize;
        let mut k = Vec::with_capacity(self.layers.len());
        let mut v = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            k.push(self.ctx.buffer(per_layer, true)?);
            v.push(self.ctx.buffer(per_layer, true)?);
        }
        Ok(MetalKv {
            k,
            v,
            cap,
            len: 0,
            per_layer_bytes: per_layer as u64,
        })
    }

    fn norm(
        &self,
        x: &MetalBuffer,
        weight: &GpuNorm,
        y: &MetalBuffer,
        rows: u64,
        cols: u64,
    ) -> Result<()> {
        match weight {
            GpuNorm::F32(w) => {
                self.ctx
                    .rmsnorm(x, w, y, rows as u32, cols as u32, self.cfg.rms_eps)
            }
            GpuNorm::F16(w) => {
                self.ctx
                    .rmsnorm_f16w(x, w, y, rows as u32, cols as u32, self.cfg.rms_eps)
            }
        }
    }

    fn gemv(&self, w: &GpuMatrix, x: &MetalBuffer, y: &MetalBuffer) -> Result<()> {
        let (rows, cols) = w.shape();
        match w {
            GpuMatrix::Q8_0 { buffer, .. } => {
                self.ctx.q8_0_gemv(buffer, x, y, rows as u32, cols as u32)
            }
            GpuMatrix::F16 { buffer, .. } => {
                self.ctx.f16_gemv(buffer, x, y, rows as u32, cols as u32)
            }
        }
    }

    fn embed_row(&self, token: u32, out: &MetalBuffer) -> Result<()> {
        match &self.token_embd {
            GpuMatrix::Q8_0 { buffer, cols, .. } => {
                self.ctx.q8_0_row(buffer, out, token, *cols as u32)
            }
            GpuMatrix::F16 { buffer, cols, .. } => {
                self.ctx.f16_row(buffer, out, token, *cols as u32)
            }
        }
    }

    /// One full transformer step at `pos`, appending to `kv`, producing
    /// vocab logits (downloaded; the sampler stays on the CPU by design).
    pub fn forward_token(
        &self,
        kv: &mut MetalKv,
        token: u32,
        pos: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        if (token as u64) >= cfg.vocab {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!(
                    "token id {token} is outside the vocabulary (0..{})",
                    cfg.vocab
                ),
            ));
        }
        if pos as u64 >= kv.cap {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!("position {pos} exceeds KV capacity {}", kv.cap),
            ));
        }
        if cancelled() {
            return Err(cancelled_error());
        }
        let ctx = &self.ctx;
        let hidden = cfg.hidden;
        let key_len = cfg.key_length;
        let group = (cfg.heads / cfg.kv_heads) as u32;
        let scale = 1.0f32 / (key_len as f32).sqrt();
        let round_f16 = self.precision == KvPrecision::F16;

        // One grouped command buffer for the whole step; cancellation before
        // any dispatch can abort the group without submitting partial work.
        if cancelled() {
            return Err(cancelled_error());
        }
        ctx.begin_group()?;
        let step = || -> Result<()> {
            self.embed_row(token, &self.ws.x)?;
            for (l, layer) in self.layers.iter().enumerate() {
                if cancelled() {
                    return Err(cancelled_error());
                }
                // ---- attention block ----
                self.norm(&self.ws.x, &layer.attn_norm, &self.ws.normed, 1, hidden)?;
                self.gemv(&layer.attn_q, &self.ws.normed, &self.ws.q)?;
                self.gemv(&layer.attn_k, &self.ws.normed, &self.ws.k)?;
                self.gemv(&layer.attn_v, &self.ws.normed, &self.ws.v)?;
                self.norm(&self.ws.q, &layer.q_norm, &self.ws.q, cfg.heads, key_len)?;
                self.norm(&self.ws.k, &layer.k_norm, &self.ws.k, cfg.kv_heads, key_len)?;
                ctx.rope(
                    &self.ws.q,
                    &self.ws.k,
                    &self.inv_freq,
                    pos as u32,
                    cfg.heads as u32,
                    cfg.kv_heads as u32,
                    key_len as u32,
                )?;
                ctx.kv_store(
                    &self.ws.k,
                    &self.ws.v,
                    &kv.k[l],
                    &kv.v[l],
                    cfg.kv_heads as u32,
                    pos as u32,
                    kv.cap as u32,
                    key_len as u32,
                    round_f16,
                )?;
                ctx.attention(
                    &self.ws.q,
                    &kv.k[l],
                    &kv.v[l],
                    &self.ws.attn_out,
                    &self.ws.attn_scratch,
                    cfg.heads as u32,
                    pos as u32,
                    kv.cap as u32,
                    key_len as u32,
                    group,
                    scale,
                )?;
                self.gemv(&layer.attn_output, &self.ws.attn_out, &self.ws.oproj)?;
                ctx.f32_axpy(&self.ws.oproj, &self.ws.x, 1.0, hidden as u32)?;
                // ---- FFN block ----
                self.norm(&self.ws.x, &layer.ffn_norm, &self.ws.normed, 1, hidden)?;
                self.gemv(&layer.ffn_gate, &self.ws.normed, &self.ws.gate)?;
                self.gemv(&layer.ffn_up, &self.ws.normed, &self.ws.up)?;
                ctx.silu_mul(&self.ws.gate, &self.ws.up, &self.ws.silu, cfg.ffn as u32)?;
                self.gemv(&layer.ffn_down, &self.ws.silu, &self.ws.down)?;
                ctx.f32_axpy(&self.ws.down, &self.ws.x, 1.0, hidden as u32)?;
            }
            self.norm(&self.ws.x, &self.output_norm, &self.ws.normed, 1, hidden)?;
            match &self.lm_head {
                LmHead::Tied => self.gemv(&self.token_embd, &self.ws.normed, &self.ws.logits),
                LmHead::Separate(m) => self.gemv(m, &self.ws.normed, &self.ws.logits),
            }
        };
        match step() {
            Ok(()) => {
                ctx.end_group()?;
            }
            Err(e) => {
                // Discard the open group; no partial GPU work is submitted.
                let _ = ctx.abort_group();
                return Err(e);
            }
        }
        kv.len = kv.len.max(pos + 1);

        // Logits come back to the CPU sampler; the copy contract fences the
        // committed step first (one completion per token step).
        let mut bytes = vec![0u8; cfg.vocab as usize * 4];
        self.ws.logits.download(0, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4-byte chunk")))
            .collect())
    }

    /// Teacher-forced logits at `position` over `tokens[..=position]`,
    /// mirroring `Forward::logits_at` (fresh session per call).
    pub fn logits_at(
        &self,
        tokens: &[u32],
        position: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() || position >= tokens.len() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "scoring position is outside the token sequence",
            ));
        }
        let mut kv = self.new_kv((position + 1) as u64)?;
        let mut logits = Vec::new();
        for (pos, token) in tokens[..=position].iter().enumerate() {
            logits = self.forward_token(&mut kv, *token, pos, cancelled)?;
        }
        Ok(logits)
    }
}

/// Per-request KV: fresh per session, one k/v buffer per layer. The
/// reported size is derived from the buffers' own construction lengths, so
/// it can never drift from the allocation.
pub struct MetalKv {
    k: Vec<MetalBuffer>,
    v: Vec<MetalBuffer>,
    cap: u64,
    len: usize,
    per_layer_bytes: u64,
}

impl MetalKv {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes held by this session's KV (k + v across layers).
    pub fn allocated_bytes(&self) -> u64 {
        self.per_layer_bytes * 2 * self.k.len() as u64
    }
}
