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
    #[cfg(test)]
    fn buffer(&self) -> &MetalBuffer {
        match self {
            Self::Q8_0 { buffer, .. } | Self::F16 { buffer, .. } => buffer,
        }
    }

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
/// chosen prefill batch and the admitted context. One per loaded worker.
/// Decode (batch 1) reuses the same buffers at offset 0, so a chunked
/// prefill and the following decode share one set of arenas.
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
    /// Chosen prefill micro-batch (bounded workspace); decode always uses
    /// batch 1 from the same buffers.
    batch: u32,
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

    /// Chosen prefill micro-batch (decode always uses batch 1).
    pub fn prefill_batch(&self) -> u32 {
        self.batch
    }

    #[cfg(test)]
    fn token_embd_bytes_for_test(&self, offset: usize, out: &mut [u8]) {
        self.token_embd
            .buffer()
            .download(offset, out)
            .expect("download token_embd");
    }

    /// KV bytes for a capacity, exactly matching `KvState::bytes_for`
    /// (f32 entries; f16 precision stores f16-representable f32 values).
    pub fn kv_bytes_for(cfg: &Qwen3Config, cap: u64) -> u64 {
        crate::forward::KvState::bytes_for(cfg, cap)
    }

    pub fn ctx(&self) -> &MetalContext {
        &self.ctx
    }

    /// Pure geometry preflight: exact bytes this resident model will hold
    /// for the given batch/context — computed from the inspected tensor
    /// list and the config, before anything is allocated. The executor
    /// checks these against the admitted plan first; `open` then verifies
    /// the measured allocations against the same numbers (second check).
    pub fn required_bytes(
        inspection: &GgufInspection,
        cfg: &Qwen3Config,
        context_cap: u64,
        batch: u32,
    ) -> Result<(u64, u64)> {
        if batch == 0 || batch > 32 {
            return Err(gpu_err(format!(
                "prefill batch {batch} is outside the supported 1..=32 range"
            )));
        }
        let weights: u64 = inspection.tensors.iter().map(|t| t.byte_size).sum();
        let b = batch as u64;
        let hidden = cfg.hidden;
        let q_width = cfg.heads * cfg.key_length;
        let kv_width = cfg.kv_heads * cfg.key_length;
        let ffn = cfg.ffn;
        let per_batch =
            (4 * hidden + 2 * q_width + 2 * kv_width + 3 * ffn + cfg.heads * context_cap)
                .saturating_mul(b)
                .saturating_mul(4);
        let workspace = (cfg.key_length / 2)
            .saturating_mul(4)
            .saturating_add(cfg.vocab.saturating_mul(4))
            .saturating_add(per_batch);
        Ok((weights, workspace))
    }

    /// Build the resident GPU model from the artifact payload, reading
    /// through the caller's already-verified file handle (no pathname
    /// reopen: the bytes that were verified are the bytes that upload). All
    /// shapes and types are validated first; anything unsupported is
    /// refused with a typed error, never approximated on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        file: &std::fs::File,
        inspection: &GgufInspection,
        names: &TensorNames,
        cfg: &Qwen3Config,
        precision: KvPrecision,
        context_cap: u64,
        batch: u32,
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
        // Preflight from inspected geometry: the executor has already
        // checked these against the admitted plan, and the measured
        // allocations are re-verified against the same numbers after
        // construction (second check) — an estimate is never accepted.
        let (weights_required, workspace_required) =
            Self::required_bytes(inspection, cfg, context_cap, batch)?;

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
        if weights_bytes != weights_required {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!(
                    "Metal weight upload allocated {weights_bytes} bytes but the preflight computed {weights_required}; refusing a model that does not match its inspected geometry"
                ),
            ));
        }
        // The rope frequency table is a small constant of the step path
        // (not a weight): it is charged to the workspace class.
        let inv_freq_bytes: Vec<u8> = crate::forward::rope_inv_freq(cfg)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let inv_freq = ctx.buffer(inv_freq_bytes.len(), true)?;
        inv_freq.upload(0, &inv_freq_bytes)?;

        // ---- workspaces (reused every step; sized for the chosen prefill
        // batch, and batch 1 reuses offset 0 of the same buffers) ----
        let hidden = cfg.hidden as usize;
        let q_width = (cfg.heads * cfg.key_length) as usize;
        let kv_width = (cfg.kv_heads * cfg.key_length) as usize;
        let ffn = cfg.ffn as usize;
        let cap = context_cap as usize;
        let b = batch as usize;
        let f32b = |n: usize| n * 4;
        let ws = Workspace {
            x: ctx.buffer(f32b(hidden * b), true)?,
            normed: ctx.buffer(f32b(hidden * b), true)?,
            q: ctx.buffer(f32b(q_width * b), true)?,
            k: ctx.buffer(f32b(kv_width * b), true)?,
            v: ctx.buffer(f32b(kv_width * b), true)?,
            attn_out: ctx.buffer(f32b(q_width * b), true)?,
            oproj: ctx.buffer(f32b(hidden * b), true)?,
            gate: ctx.buffer(f32b(ffn * b), true)?,
            up: ctx.buffer(f32b(ffn * b), true)?,
            silu: ctx.buffer(f32b(ffn * b), true)?,
            down: ctx.buffer(f32b(hidden * b), true)?,
            logits: ctx.buffer(f32b(cfg.vocab as usize), true)?,
            attn_scratch: ctx.buffer(f32b(cfg.heads as usize * cap * b), true)?,
        };
        let workspace_bytes = ctx.allocated_bytes() - weights_bytes;
        if workspace_bytes != workspace_required {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!(
                    "Metal workspaces allocated {workspace_bytes} bytes but the preflight computed {workspace_required}"
                ),
            ));
        }

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
            batch,
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

    fn embed_row(&self, token: u32, out: &MetalBuffer, out_off: u32) -> Result<()> {
        match &self.token_embd {
            GpuMatrix::Q8_0 { buffer, cols, .. } => {
                self.ctx.q8_0_row(buffer, out, token, *cols as u32, out_off)
            }
            GpuMatrix::F16 { buffer, cols, .. } => {
                self.ctx.f16_row(buffer, out, token, *cols as u32, out_off)
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
            self.embed_row(token, &self.ws.x, 0)?;
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

    /// Chunked causal prefill: process `tokens` (a micro-batch at positions
    /// `start_pos..start_pos+b`) in ONE grouped command buffer — batched
    /// projections (packed GEMM), row-wise norms, batched RoPE, batched KV
    /// writes and masked GQA attention — then, only when `want_logits`, the
    /// final norm and vocabulary projection for the last token (C3.1: no
    /// intermediate logits, no per-position round trips). Per-token
    /// arithmetic order matches the single-token path, so results are
    /// independent of the batch size.
    pub fn prefill_chunk(
        &self,
        kv: &mut MetalKv,
        tokens: &[u32],
        start_pos: usize,
        want_logits: bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>> {
        let cfg = &self.cfg;
        let b = tokens.len();
        if b == 0 {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "prefill chunk must contain at least one token",
            ));
        }
        if b as u32 > self.batch || b > 32 {
            return Err(gpu_err(format!(
                "prefill chunk of {b} tokens exceeds the prepared batch {}",
                self.batch
            )));
        }
        if start_pos as u64 + b as u64 > kv.cap {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!(
                    "chunk {}..{} exceeds KV capacity {}",
                    start_pos,
                    start_pos + b,
                    kv.cap
                ),
            ));
        }
        for token in tokens {
            if (*token as u64) >= cfg.vocab {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!(
                        "token id {token} is outside the vocabulary (0..{})",
                        cfg.vocab
                    ),
                ));
            }
        }
        if cancelled() {
            return Err(cancelled_error());
        }
        let hidden = cfg.hidden;
        let key_len = cfg.key_length;
        let group = (cfg.heads / cfg.kv_heads) as u32;
        let scale = 1.0f32 / (key_len as f32).sqrt();
        let round_f16 = self.precision == KvPrecision::F16;
        let bu = b as u32;
        let ctx = &self.ctx;

        ctx.begin_group()?;
        let step = || -> Result<()> {
            for (t, token) in tokens.iter().enumerate() {
                self.embed_row(*token, &self.ws.x, (t * hidden as usize) as u32)?;
            }
            for (l, layer) in self.layers.iter().enumerate() {
                if cancelled() {
                    return Err(cancelled_error());
                }
                self.norm(
                    &self.ws.x,
                    &layer.attn_norm,
                    &self.ws.normed,
                    b as u64,
                    hidden,
                )?;
                self.gemm(&layer.attn_q, &self.ws.normed, &self.ws.q, bu)?;
                self.gemm(&layer.attn_k, &self.ws.normed, &self.ws.k, bu)?;
                self.gemm(&layer.attn_v, &self.ws.normed, &self.ws.v, bu)?;
                self.norm(
                    &self.ws.q,
                    &layer.q_norm,
                    &self.ws.q,
                    b as u64 * cfg.heads,
                    key_len,
                )?;
                self.norm(
                    &self.ws.k,
                    &layer.k_norm,
                    &self.ws.k,
                    b as u64 * cfg.kv_heads,
                    key_len,
                )?;
                ctx.rope_batch(
                    &self.ws.q,
                    &self.ws.k,
                    &self.inv_freq,
                    start_pos as u32,
                    bu,
                    cfg.heads as u32,
                    cfg.kv_heads as u32,
                    key_len as u32,
                )?;
                ctx.kv_store_batch(
                    &self.ws.k,
                    &self.ws.v,
                    &kv.k[l],
                    &kv.v[l],
                    bu,
                    start_pos as u32,
                    kv.cap as u32,
                    key_len as u32,
                    cfg.kv_heads as u32,
                    round_f16,
                )?;
                ctx.attention_batch(
                    &self.ws.q,
                    &kv.k[l],
                    &kv.v[l],
                    &self.ws.attn_out,
                    &self.ws.attn_scratch,
                    cfg.heads as u32,
                    bu,
                    start_pos as u32,
                    kv.cap as u32,
                    key_len as u32,
                    group,
                    scale,
                )?;
                self.gemm(&layer.attn_output, &self.ws.attn_out, &self.ws.oproj, bu)?;
                ctx.f32_axpy(
                    &self.ws.oproj,
                    &self.ws.x,
                    1.0,
                    (b * cfg.hidden as usize) as u32,
                )?;
                self.norm(
                    &self.ws.x,
                    &layer.ffn_norm,
                    &self.ws.normed,
                    b as u64,
                    hidden,
                )?;
                self.gemm(&layer.ffn_gate, &self.ws.normed, &self.ws.gate, bu)?;
                self.gemm(&layer.ffn_up, &self.ws.normed, &self.ws.up, bu)?;
                ctx.silu_mul(
                    &self.ws.gate,
                    &self.ws.up,
                    &self.ws.silu,
                    (b * cfg.ffn as usize) as u32,
                )?;
                self.gemm(&layer.ffn_down, &self.ws.silu, &self.ws.down, bu)?;
                ctx.f32_axpy(
                    &self.ws.down,
                    &self.ws.x,
                    1.0,
                    (b * cfg.hidden as usize) as u32,
                )?;
            }
            if want_logits {
                self.norm(
                    &self.ws.x,
                    &self.output_norm,
                    &self.ws.normed,
                    b as u64,
                    hidden,
                )?;
                // Vocabulary projection for the LAST token only (C3.1).
                let x_off = ((b - 1) * cfg.hidden as usize) as u32;
                match &self.lm_head {
                    LmHead::Tied => {
                        self.gemm_at(&self.token_embd, &self.ws.normed, &self.ws.logits, 1, x_off)?
                    }
                    LmHead::Separate(m) => {
                        self.gemm_at(m, &self.ws.normed, &self.ws.logits, 1, x_off)?
                    }
                }
            }
            Ok(())
        };
        match step() {
            Ok(()) => ctx.end_group()?,
            Err(e) => {
                let _ = ctx.abort_group();
                return Err(e);
            }
        }
        kv.len = kv.len.max(start_pos + b);
        if want_logits {
            let mut bytes = vec![0u8; cfg.vocab as usize * 4];
            self.ws.logits.download(0, &mut bytes)?;
            Ok(Some(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("4-byte chunk")))
                    .collect(),
            ))
        } else {
            Ok(None)
        }
    }

    /// Batched promotion on one workspace (y is [batch, rows|cols]).
    fn gemm(&self, w: &GpuMatrix, x: &MetalBuffer, y: &MetalBuffer, batch: u32) -> Result<()> {
        self.gemm_at(w, x, y, batch, 0)
    }

    /// Batched promotion with an explicit input row offset (used for the
    /// last-token vocabulary projection).
    fn gemm_at(
        &self,
        w: &GpuMatrix,
        x: &MetalBuffer,
        y: &MetalBuffer,
        batch: u32,
        x_off: u32,
    ) -> Result<()> {
        let (rows, cols) = w.shape();
        match w {
            GpuMatrix::Q8_0 { buffer, .. } => {
                self.ctx
                    .q8_0_gemm(buffer, x, y, rows as u32, cols as u32, batch, x_off, 0)
            }
            GpuMatrix::F16 { buffer, .. } => {
                self.ctx
                    .f16_gemm(buffer, x, y, rows as u32, cols as u32, batch, x_off, 0)
            }
        }
    }

    /// Teacher-forced logits at `position` over `tokens[..=position]`,
    /// mirroring `Forward::logits_at` (fresh session per call). Prefill is
    /// chunked; only the final chunk computes the vocabulary projection.
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
        let end = position + 1;
        let mut pos = 0usize;
        let mut logits: Option<Vec<f32>> = None;
        while pos < end {
            let take = (self.batch as usize).min(end - pos);
            let want = pos + take == end;
            let out =
                self.prefill_chunk(&mut kv, &tokens[pos..pos + take], pos, want, cancelled)?;
            if let Some(l) = out {
                logits = Some(l);
            }
            pos += take;
        }
        logits.ok_or_else(|| TuckmillError::new(ErrorCode::Internal, "scoring produced no logits"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

    fn metal_or_skip() -> Option<tuckmill_metal::Probe> {
        let probe = tuckmill_metal::probe();
        if !probe.available {
            assert!(
                std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
                "TUCKMILL_REQUIRE_METAL is set but no Metal device is available"
            );
            eprintln!("skipping: no Metal device");
            return None;
        }
        Some(probe)
    }

    fn fixture(path: &std::path::Path) {
        write_synthetic(path, &SyntheticConfig::tiny_qwen3()).unwrap();
    }

    fn open_forward(
        path: &std::path::Path,
        batch: u32,
    ) -> (MetalForward, std::fs::File, GgufInspection) {
        let file = std::fs::File::open(path).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let registry = tuckmill_manifest::adapter::AdapterRegistry::builtin().unwrap();
        let (cfg, _) = crate::model::resolve(&insp, &registry).unwrap();
        let mut names = TensorNames::for_layers(cfg.layers);
        names.resolve(&insp).unwrap();
        let forward =
            MetalForward::open(&file, &insp, &names, &cfg, KvPrecision::F32, 256, batch).unwrap();
        (forward, file, insp)
    }

    /// C3.0: Metal weights come from the verified, already-open file handle.
    /// Replacing the pathname by rename after verification must not change
    /// a single weight byte the GPU uploads.
    #[test]
    fn metal_weights_come_from_the_verified_handle_not_the_pathname() {
        if metal_or_skip().is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        fixture(&path);
        let original = std::fs::read(&path).unwrap();

        // Verified-handle scenario: open + verify A, then swap the path.
        let file = std::fs::File::open(&path).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let registry = tuckmill_manifest::adapter::AdapterRegistry::builtin().unwrap();
        let (cfg, _) = crate::model::resolve(&insp, &registry).unwrap();
        let mut names = TensorNames::for_layers(cfg.layers);
        names.resolve(&insp).unwrap();

        // File B: same geometry, one payload byte flipped.
        let replacement = dir.path().join("replacement.gguf");
        let mut tampered = original.clone();
        let idx = tampered.len() - 1024; // inside the tensor payload
        tampered[idx] ^= 0xFF;
        std::fs::write(&replacement, &tampered).unwrap();
        std::fs::rename(&replacement, &path).unwrap();

        let forward =
            MetalForward::open(&file, &insp, &names, &cfg, KvPrecision::F32, 256, 4).unwrap();
        // Download the first tensor buffer and compare with A's bytes.
        let first = insp.tensors.first().unwrap();
        let mut got = vec![0u8; first.byte_size as usize];
        forward.token_embd_bytes_for_test(0, &mut got);
        let start = insp.data_start as usize + first.offset as usize;
        assert_eq!(
            got,
            original[start..start + got.len()],
            "GPU weights must equal the verified file's bytes, not the replaced pathname's"
        );
    }

    /// C3.4: the effective batch is the largest candidate whose exact
    /// workspace geometry fits the plan, and results are independent of it.
    #[test]
    fn batch_geometry_preflight_and_batch_independence() {
        if metal_or_skip().is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        fixture(&path);

        // Preflight is pure geometry and matches the measured allocation.
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let registry = tuckmill_manifest::adapter::AdapterRegistry::builtin().unwrap();
        let (cfg, _) = crate::model::resolve(&insp, &registry).unwrap();
        let (w_req, ws1) = MetalForward::required_bytes(&insp, &cfg, 256, 1).unwrap();
        let (_, ws8) = MetalForward::required_bytes(&insp, &cfg, 256, 8).unwrap();
        assert!(ws8 > ws1, "batch 8 needs more workspace than batch 1");
        let (forward, _file, _insp) = open_forward(&path, 8);
        assert_eq!(
            forward.weights_bytes(),
            w_req,
            "preflight == measured weights"
        );
        assert_eq!(
            forward.workspace_bytes(),
            ws8,
            "preflight == measured workspaces"
        );
        assert_eq!(forward.prefill_batch(), 8);

        // Bit-identical teacher-forced logits across batch sizes.
        let tokens: Vec<u32> = vec![1, 9, 4, 17, 5, 23, 7];
        let mut baseline: Option<Vec<f32>> = None;
        for batch in [1u32, 2, 3, 4, 8] {
            let (forward, _f, _i) = open_forward(&path, batch);
            let logits = forward
                .logits_at(&tokens, tokens.len() - 1, &|| false)
                .unwrap();
            match &baseline {
                None => baseline = Some(logits),
                Some(base) => {
                    assert_eq!(base.len(), logits.len());
                    for (i, (a, b)) in base.iter().zip(logits.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "batch {batch}: logit {i} differs from the batch-1 baseline"
                        );
                    }
                }
            }
        }
    }
}
