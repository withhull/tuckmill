//! The unified engine on Metal (ADR-011): shared blocks plus one model
//! that executes any [`ModelGraph`].
//!
//! Every block is built from the qualified device operators
//! ([`Qwen35Gpu`]'s `*_dev` ops and the `tuckmill-metal` kernels). Each
//! block offers a per-token `decode` and a chunked `prefill`, and both run
//! the block's one sequential `core`, so the two paths cannot diverge. The
//! engine owns one command buffer per decode step and per prefill chunk,
//! session reset, the exact preflight plan (ledger == plan) and
//! per-operator profiling.

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_metal::{
    MetalBuffer, MetalContext, ARGMAX_NONE, ARGMAX_PARTIAL_BYTES, GEMM_MAX_BATCH,
    GEMM_MMA_MAX_BATCH,
};

use super::graph::{AttentionSpec, FfnSpec, GatedDeltaNetSpec, MixerSpec, ModelGraph, Tensor};
use super::WeightSource;
use crate::qwen35_gpu::{f32_bytes, GpuAllocPlan, Qwen35Gpu, STAGING_CHUNK_BYTES};

/// Prompt tokens per batched prefill chunk (one command buffer each).
pub const PREFILL_BATCH: usize = 128;

/// Prefill chunks allowed in flight on the GPU at once. Each chunk is one
/// command buffer of thousands of dispatches, and Metal backs its encoding
/// with driver memory (~3 MB on the pinned 0.8B) that it then keeps
/// pooled. Committing a long prompt's chunks unbounded held ~100 MB of it
/// at once; two keeps the next chunk queued behind the running one.
const MAX_PREFILL_IN_FLIGHT: usize = 2;

/// A decode step committed without waiting (pipelined greedy decoding):
/// its chosen token is in `slot` of the token buffer once the first
/// `commit` command buffers have completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingToken {
    commit: u64,
    slot: u32,
}

/// The input token of a pipelined greedy step.
#[derive(Debug, Clone, Copy)]
pub enum GreedyInput {
    /// A token the host already knows.
    Token(u32),
    /// The token a previous step chose, read on the device.
    Previous(PendingToken),
}

/// Token-buffer slot a step at `pos` writes; consecutive steps alternate,
/// so a step never overwrites the slot of the step before it.
fn token_slot(pos: usize) -> u32 {
    (pos % 2) as u32
}

/// What a step or prefill chunk returns for its last position.
enum Out<'a> {
    /// Nothing (a non-final prefill chunk).
    None,
    /// Full-vocabulary logits, downloaded.
    Logits(&'a mut Vec<f32>),
    /// The greedy token chosen on the device (`None`: every logit NaN);
    /// only 4 bytes cross to the host.
    Greedy(&'a mut Option<u32>),
}

/// Stage reported to a traced decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Embedding,
    Layer(usize),
}

/// How a chunk of rows runs through the blocks.
#[derive(Debug, Clone, Copy)]
struct ChunkMode {
    /// Mixer cores chunk-wide (default) or per token on the decode kernels.
    chunk_core: bool,
    /// Every projection on the per-row GEMMs, bit-identical to the decode
    /// GEMV (a speculative verify), instead of the matrix-unit GEMM.
    exact: bool,
    /// Capture each Gated DeltaNet layer's state after the chunk's first
    /// token (the rollback point of a speculative verify).
    capture: bool,
}

impl ChunkMode {
    fn prefill(chunk_core: bool) -> Self {
        Self {
            chunk_core,
            exact: false,
            capture: false,
        }
    }
}

// ---------------------------------------------------------------- tensors --

fn vector_len(src: &dyn WeightSource, t: Tensor) -> Result<usize> {
    let (in_dim, out_dim) = src.tensor_dims(t.index)?;
    Ok((in_dim * out_dim.max(1)) as usize)
}

/// A device matrix uploaded raw (F16 halves or Q8_0 blocks).
struct Matrix {
    buf: MetalBuffer,
    type_id: u32,
    in_dim: u64,
    out_dim: u64,
}

impl Matrix {
    fn plan(src: &dyn WeightSource, t: Tensor) -> Result<u64> {
        src.tensor_bytes(t.index)
    }

    fn load(gpu: &Qwen35Gpu, src: &dyn WeightSource, t: Tensor) -> Result<Self> {
        let total = src.tensor_bytes(t.index)? as usize;
        let type_id = src.tensor_type_id(t.index)?;
        let (in_dim, out_dim) = src.tensor_dims(t.index)?;
        if out_dim <= 1 {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!("tensor {} is 1-D; vectors use the f32 path", t.index),
            ));
        }
        if type_id != 1 && type_id != 8 {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedQuantization,
                format!(
                    "engine matrices must be F16 or Q8_0; tensor {} is type {type_id}",
                    t.index
                ),
            ));
        }
        // Fully overwritten: no zero pass; bounded host staging.
        let buf = gpu.buffer_uninit(total)?;
        let mut stage = vec![0u8; STAGING_CHUNK_BYTES.min(total).max(1)];
        let mut off = 0usize;
        while off < total {
            let n = STAGING_CHUNK_BYTES.min(total - off);
            src.read_tensor_into(t.index, off as u64, &mut stage[..n])?;
            buf.upload(off, &stage[..n])?;
            off += n;
        }
        Ok(Self {
            buf,
            type_id,
            in_dim,
            out_dim,
        })
    }

    /// `y = W x` with the projection kernel for the stored type.
    fn gemv(&self, gpu: &Qwen35Gpu, x: &MetalBuffer, y: &MetalBuffer) -> Result<()> {
        let ctx = gpu.context();
        let (rows, cols) = (self.out_dim as u32, self.in_dim as u32);
        match self.type_id {
            1 => ctx.f16_gemv(&self.buf, x, y, rows, cols),
            8 => ctx.q8_0_gemv(&self.buf, x, y, rows, cols),
            other => Err(unsupported_type(other)),
        }
    }

    /// `Y[b] = W X[b]` for `b < batch`. F16, batch 1 and `exact` keep the
    /// GEMV's per-row arithmetic (in slices of at most `GEMM_MAX_BATCH`
    /// tokens); other Q8_0 batches of 2 or more use the SIMD-group matrix
    /// units in one launch (tolerance-qualified, ADR-008 Addendum 16).
    fn gemm(
        &self,
        gpu: &Qwen35Gpu,
        x: &MetalBuffer,
        y: &MetalBuffer,
        batch: u32,
        exact: bool,
    ) -> Result<()> {
        let ctx = gpu.context();
        let (rows, cols) = (self.out_dim as u32, self.in_dim as u32);
        if !exact && self.type_id == 8 && (2..=GEMM_MMA_MAX_BATCH).contains(&batch) {
            return ctx.q8_0_gemm_mma(&self.buf, x, y, rows, cols, batch, 0, 0);
        }
        let mut t0 = 0u32;
        while t0 < batch {
            let n = (batch - t0).min(GEMM_MAX_BATCH);
            let (x_off, y_off) = (t0 * cols, t0 * rows);
            match self.type_id {
                1 => ctx.f16_gemm(&self.buf, x, y, rows, cols, n, x_off, y_off)?,
                8 => ctx.q8_0_gemm(&self.buf, x, y, rows, cols, n, x_off, y_off)?,
                other => return Err(unsupported_type(other)),
            }
            t0 += n;
        }
        Ok(())
    }

    /// Embedding lookup of a token id chosen on the device (`ids[slot]`).
    fn row_indirect(
        &self,
        gpu: &Qwen35Gpu,
        ids: &MetalBuffer,
        slot: u32,
        out: &MetalBuffer,
    ) -> Result<()> {
        gpu.context().row_indirect_f32(
            &self.buf,
            self.type_id,
            self.out_dim as u32,
            self.in_dim as u32,
            ids,
            slot,
            out,
        )
    }

    /// Dequantize one row into `out` at `out_off` (embedding lookup).
    fn row_into(&self, gpu: &Qwen35Gpu, row: u32, out: &MetalBuffer, out_off: u32) -> Result<()> {
        let ctx = gpu.context();
        match self.type_id {
            1 => ctx.f16_row(&self.buf, out, row, self.in_dim as u32, out_off),
            8 => ctx.q8_0_row(&self.buf, out, row, self.in_dim as u32, out_off),
            other => Err(unsupported_type(other)),
        }
    }
}

fn unsupported_type(type_id: u32) -> TuckmillError {
    TuckmillError::new(
        ErrorCode::UnsupportedQuantization,
        format!("no device projection for stored type {type_id}"),
    )
}

fn vector(gpu: &Qwen35Gpu, src: &dyn WeightSource, t: Tensor) -> Result<MetalBuffer> {
    floats(gpu, &src.read_vector(t.index)?)
}

fn floats(gpu: &Qwen35Gpu, values: &[f32]) -> Result<MetalBuffer> {
    let buf = gpu.buffer(values.len().max(1) * 4)?;
    buf.upload(0, &f32_bytes(values))?;
    Ok(buf)
}

fn zeros(gpu: &Qwen35Gpu, elems: usize) -> Result<MetalBuffer> {
    gpu.buffer(elems.max(1) * 4)
}

/// Bytes of a buffer created by [`zeros`] / [`floats`].
fn f32_buffer_bytes(elems: usize) -> u64 {
    4 * elems.max(1) as u64
}

/// Allocation accumulator mirroring every buffer a block creates.
#[derive(Default)]
struct Plan {
    weights: u64,
    state: u64,
    scratch: u64,
}

impl Plan {
    fn weights_f32(&mut self, elems: usize) {
        self.weights += f32_buffer_bytes(elems);
    }
    fn state_f32(&mut self, elems: usize) {
        self.state += f32_buffer_bytes(elems);
    }
    fn scratch_f32(&mut self, elems: usize) {
        self.scratch += f32_buffer_bytes(elems);
    }
}

// --------------------------------------------------------- prefill scratch --

/// Per-token widths of the shared batched-prefill activations: the maximum
/// over every layer of the graph.
#[derive(Debug, Clone, Copy, Default)]
struct PrefillWidths {
    hidden: usize,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    e: usize,
    ffn: usize,
    /// Chunk-wide Gated DeltaNet core: convolution output, q/k (key
    /// width), v/out (value width), beta/gate (value heads).
    conv: usize,
    gk: usize,
    gv: usize,
    gh: usize,
    /// Chunk-wide attention core: q / gate rows and score rows.
    aq: usize,
    ascr: usize,
}

impl PrefillWidths {
    fn of(graph: &ModelGraph, capacity: usize) -> Self {
        let mut w = PrefillWidths {
            hidden: graph.hidden,
            ..Default::default()
        };
        for layer in &graph.layers {
            match &layer.mixer {
                MixerSpec::GatedDeltaNet(g) => {
                    let v = g.head_v * g.dt_rank;
                    w.a = w.a.max(g.channels);
                    w.b = w.b.max(v);
                    w.c = w.c.max(g.dt_rank);
                    w.d = w.d.max(g.dt_rank);
                    w.e = w.e.max(v);
                    w.conv = w.conv.max(g.channels);
                    w.gk = w.gk.max(g.groups * g.state_size);
                    w.gv = w.gv.max(v);
                    w.gh = w.gh.max(g.dt_rank);
                }
                MixerSpec::Attention(a) => {
                    let kv = a.kv_heads * a.head_dim;
                    let qw = a.heads * a.head_dim * if a.output_gate { 2 } else { 1 };
                    w.a = w.a.max(qw);
                    w.b = w.b.max(kv);
                    w.c = w.c.max(kv);
                    w.e = w.e.max(a.heads * a.head_dim);
                    w.aq = w.aq.max(a.heads * a.head_dim);
                    w.ascr = w.ascr.max(a.heads * capacity);
                }
            }
            match &layer.ffn {
                FfnSpec::SwiGlu { width, .. } => w.ffn = w.ffn.max(*width),
            }
        }
        w
    }

    fn sizes(&self) -> [usize; 22] {
        let n = PREFILL_BATCH;
        [
            n * self.hidden, // x
            n * self.hidden, // h
            n * self.hidden, // h2
            n * self.a,
            n * self.b,
            n * self.c,
            n * self.d,
            n * self.e,
            n * self.hidden, // proj
            n * self.hidden, // d2
            n * self.ffn,    // g
            n * self.ffn,    // u
            n * self.conv,   // gdn conv output
            n * self.gk,     // gdn q
            n * self.gk,     // gdn k
            n * self.gv,     // gdn v
            n * self.gh,     // gdn beta
            n * self.gh,     // gdn gate
            n * self.gv,     // gdn recurrence output
            n * self.aq,     // attention q
            n * self.aq,     // attention gate
            n * self.ascr,   // attention scores
        ]
    }
}

/// Batched-prefill activations shared by all layers (`[PREFILL_BATCH,
/// width]` row-major f32). Layers run in order, so one set suffices.
struct PrefillScratch {
    x: MetalBuffer,
    h: MetalBuffer,
    h2: MetalBuffer,
    a: MetalBuffer,
    b: MetalBuffer,
    c: MetalBuffer,
    d: MetalBuffer,
    e: MetalBuffer,
    proj: MetalBuffer,
    d2: MetalBuffer,
    g: MetalBuffer,
    u: MetalBuffer,
    gconv: MetalBuffer,
    gq: MetalBuffer,
    gk: MetalBuffer,
    gv: MetalBuffer,
    gbeta: MetalBuffer,
    ggate: MetalBuffer,
    gout: MetalBuffer,
    aq: MetalBuffer,
    ag: MetalBuffer,
    ascr: MetalBuffer,
}

impl PrefillScratch {
    fn plan(graph: &ModelGraph, capacity: usize, p: &mut Plan) {
        for elems in PrefillWidths::of(graph, capacity).sizes() {
            p.scratch_f32(elems);
        }
    }

    fn load(gpu: &Qwen35Gpu, graph: &ModelGraph, capacity: usize) -> Result<Self> {
        let [x, h, h2, a, b, c, d, e, proj, d2, g, u, gconv, gq, gk, gv, gbeta, ggate, gout, aq, ag, ascr] =
            PrefillWidths::of(graph, capacity).sizes();
        Ok(Self {
            x: zeros(gpu, x)?,
            h: zeros(gpu, h)?,
            h2: zeros(gpu, h2)?,
            a: zeros(gpu, a)?,
            b: zeros(gpu, b)?,
            c: zeros(gpu, c)?,
            d: zeros(gpu, d)?,
            e: zeros(gpu, e)?,
            proj: zeros(gpu, proj)?,
            d2: zeros(gpu, d2)?,
            g: zeros(gpu, g)?,
            u: zeros(gpu, u)?,
            gconv: zeros(gpu, gconv)?,
            gq: zeros(gpu, gq)?,
            gk: zeros(gpu, gk)?,
            gv: zeros(gpu, gv)?,
            gbeta: zeros(gpu, gbeta)?,
            ggate: zeros(gpu, ggate)?,
            gout: zeros(gpu, gout)?,
            aq: zeros(gpu, aq)?,
            ag: zeros(gpu, ag)?,
            ascr: zeros(gpu, ascr)?,
        })
    }
}

// ------------------------------------------------------------------ blocks --

/// SwiGLU FFN: `down(silu(gate x) * up x)`.
struct SwiGluBlock {
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    g: MetalBuffer,
    u: MetalBuffer,
    out: MetalBuffer,
    width: usize,
}

impl SwiGluBlock {
    fn plan(spec: &FfnSpec, hidden: usize, src: &dyn WeightSource, p: &mut Plan) -> Result<()> {
        let FfnSpec::SwiGlu {
            gate,
            up,
            down,
            width,
        } = spec;
        p.weights +=
            Matrix::plan(src, *gate)? + Matrix::plan(src, *up)? + Matrix::plan(src, *down)?;
        p.scratch_f32(*width);
        p.scratch_f32(*width);
        p.scratch_f32(hidden);
        Ok(())
    }

    fn load(
        gpu: &Qwen35Gpu,
        src: &dyn WeightSource,
        spec: &FfnSpec,
        hidden: usize,
    ) -> Result<Self> {
        let FfnSpec::SwiGlu {
            gate,
            up,
            down,
            width,
        } = spec;
        Ok(Self {
            gate: Matrix::load(gpu, src, *gate)?,
            up: Matrix::load(gpu, src, *up)?,
            down: Matrix::load(gpu, src, *down)?,
            g: zeros(gpu, *width)?,
            u: zeros(gpu, *width)?,
            out: zeros(gpu, hidden)?,
            width: *width,
        })
    }

    /// `x += ffn(h)` for one token.
    fn decode(
        &self,
        gpu: &Qwen35Gpu,
        h: &MetalBuffer,
        x: &MetalBuffer,
        hidden: usize,
    ) -> Result<()> {
        let ctx = gpu.context();
        self.gate.gemv(gpu, h, &self.g)?;
        self.up.gemv(gpu, h, &self.u)?;
        // In place: `out` aliases `gate`; each element is read first.
        ctx.silu_mul(&self.g, &self.u, &self.g, self.width as u32)?;
        self.down.gemv(gpu, &self.g, &self.out)?;
        ctx.f32_axpy(&self.out, x, 1.0, hidden as u32)
    }

    /// `x += ffn(h)` for the `n` chunk rows of `s.h2` / `s.x`.
    fn prefill(
        &self,
        gpu: &Qwen35Gpu,
        s: &PrefillScratch,
        n: usize,
        hidden: usize,
        exact: bool,
    ) -> Result<()> {
        let ctx = gpu.context();
        let nn = n as u32;
        self.gate.gemm(gpu, &s.h2, &s.g, nn, exact)?;
        self.up.gemm(gpu, &s.h2, &s.u, nn, exact)?;
        ctx.silu_mul(&s.g, &s.u, &s.g, (n * self.width) as u32)?;
        self.down.gemm(gpu, &s.g, &s.d2, nn, exact)?;
        ctx.f32_axpy(&s.d2, &s.x, 1.0, (n * hidden) as u32)
    }
}

/// Gated DeltaNet mixer: projections, causal convolution, L2-normed Q/K,
/// gates, the delta-rule recurrence and the gated output norm.
struct GdnBlock {
    spec: GatedDeltaNetSpec,
    eps: f32,
    qkv: Matrix,
    z: Matrix,
    alpha: Matrix,
    beta: Matrix,
    out: Matrix,
    ssm_a: MetalBuffer,
    ssm_dt: MetalBuffer,
    ssm_norm: MetalBuffer,
    conv_w: MetalBuffer,
    state: MetalBuffer,
    hist: MetalBuffer,
    /// Prefix-cache snapshot of `state` / `hist` at a chunk boundary
    /// (ADR-012 step 2); absent when the prefix cache is disabled.
    snapshot: Option<(MetalBuffer, MetalBuffer)>,
    /// `state` / `hist` after the first token of a speculative verify,
    /// restored when its draft is rejected; absent without speculation.
    rollback: Option<(MetalBuffer, MetalBuffer)>,
    qkv_out: MetalBuffer,
    conv_y: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    z_out: MetalBuffer,
    alpha_out: MetalBuffer,
    beta_raw: MetalBuffer,
    beta_out: MetalBuffer,
    gate: MetalBuffer,
    o: MetalBuffer,
    gated: MetalBuffer,
}

impl GdnBlock {
    fn state_elems(s: &GatedDeltaNetSpec) -> (usize, usize) {
        (
            s.groups * s.state_size * s.head_v,
            s.conv_kernel.saturating_sub(1).max(1) * s.channels,
        )
    }

    fn scratch_elems(s: &GatedDeltaNetSpec) -> [usize; 12] {
        let gs = s.groups * s.state_size;
        let v = s.head_v * s.dt_rank;
        [
            s.channels, s.channels, gs, gs, v, v, s.dt_rank, s.dt_rank, s.dt_rank, s.dt_rank, v, v,
        ]
    }

    /// The q/k/v split (per token and chunk-wide) takes equal key and
    /// value widths; anything else needs its own split before it can run.
    fn check_supported(s: &GatedDeltaNetSpec) -> Result<()> {
        let key = s.groups * s.state_size;
        if s.head_v * s.dt_rank != key || s.channels != 3 * key || s.groups != s.dt_rank {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "Gated DeltaNet geometry (key {key}, value {}, channels {}, groups {}, value heads {}) has no engine block yet",
                    s.head_v * s.dt_rank,
                    s.channels,
                    s.groups,
                    s.dt_rank
                ),
            )
            .with_action("Wait for the matching DeltaNet block; geometry is never approximated."));
        }
        Ok(())
    }

    fn plan(
        s: &GatedDeltaNetSpec,
        src: &dyn WeightSource,
        opts: EngineOptions,
        p: &mut Plan,
    ) -> Result<()> {
        Self::check_supported(s)?;
        for m in [s.qkv, s.z, s.alpha, s.beta, s.out] {
            p.weights += Matrix::plan(src, m)?;
        }
        for v in [s.ssm_a, s.ssm_dt, s.ssm_norm, s.conv1d] {
            p.weights_f32(vector_len(src, v)?);
        }
        let (state, hist) = Self::state_elems(s);
        let copies = if opts.prefix_cache { 2 } else { 1 };
        for _ in 0..copies {
            p.state_f32(state);
            p.state_f32(hist);
        }
        if opts.mtp {
            // Rollback buffers: rewritten by every verify, so scratch.
            p.scratch_f32(state);
            p.scratch_f32(hist);
        }
        for elems in Self::scratch_elems(s) {
            p.scratch_f32(elems);
        }
        Ok(())
    }

    fn load(
        gpu: &Qwen35Gpu,
        src: &dyn WeightSource,
        s: &GatedDeltaNetSpec,
        eps: f32,
        opts: EngineOptions,
    ) -> Result<Self> {
        Self::check_supported(s)?;
        let (state, hist) = Self::state_elems(s);
        let pair = |on: bool| -> Result<Option<(MetalBuffer, MetalBuffer)>> {
            Ok(if on {
                Some((zeros(gpu, state)?, zeros(gpu, hist)?))
            } else {
                None
            })
        };
        let snapshot = pair(opts.prefix_cache)?;
        let rollback = pair(opts.mtp)?;
        let [qkv_out, conv_y, q, k, v, z_out, alpha_out, beta_raw, beta_out, gate, o, gated] =
            Self::scratch_elems(s);
        Ok(Self {
            qkv: Matrix::load(gpu, src, s.qkv)?,
            z: Matrix::load(gpu, src, s.z)?,
            alpha: Matrix::load(gpu, src, s.alpha)?,
            beta: Matrix::load(gpu, src, s.beta)?,
            out: Matrix::load(gpu, src, s.out)?,
            ssm_a: vector(gpu, src, s.ssm_a)?,
            ssm_dt: vector(gpu, src, s.ssm_dt)?,
            ssm_norm: vector(gpu, src, s.ssm_norm)?,
            conv_w: vector(gpu, src, s.conv1d)?,
            state: zeros(gpu, state)?,
            hist: zeros(gpu, hist)?,
            snapshot,
            rollback,
            qkv_out: zeros(gpu, qkv_out)?,
            conv_y: zeros(gpu, conv_y)?,
            q: zeros(gpu, q)?,
            k: zeros(gpu, k)?,
            v: zeros(gpu, v)?,
            z_out: zeros(gpu, z_out)?,
            alpha_out: zeros(gpu, alpha_out)?,
            beta_raw: zeros(gpu, beta_raw)?,
            beta_out: zeros(gpu, beta_out)?,
            gate: zeros(gpu, gate)?,
            o: zeros(gpu, o)?,
            gated: zeros(gpu, gated)?,
            spec: s.clone(),
            eps,
        })
    }

    fn reset(&self) -> Result<()> {
        zero_fill(&self.state)?;
        zero_fill(&self.hist)
    }

    /// Copy the live state into the snapshot (`save`) or back (restore),
    /// on the device, in queue order with the surrounding chunks.
    fn copy_snapshot(&self, gpu: &Qwen35Gpu, save: bool) -> Result<()> {
        let Some((snap_state, snap_hist)) = &self.snapshot else {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                "prefix snapshot requested on a block loaded without the prefix cache",
            ));
        };
        let ctx = gpu.context();
        for (live, snap) in [(&self.state, snap_state), (&self.hist, snap_hist)] {
            let elems = (live.len() / 4) as u32;
            if save {
                ctx.copy_f32(live, 0, snap, 0, elems)?;
            } else {
                ctx.copy_f32(snap, 0, live, 0, elems)?;
            }
        }
        Ok(())
    }

    /// Restore the state captured after the first token of the last
    /// speculative verify (its draft was rejected).
    fn roll_back(&self, gpu: &Qwen35Gpu) -> Result<()> {
        let Some((rb_state, rb_hist)) = &self.rollback else {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                "rollback requested on a block loaded without speculation",
            ));
        };
        let ctx = gpu.context();
        for (rb, live) in [(rb_state, &self.state), (rb_hist, &self.hist)] {
            ctx.copy_f32(rb, 0, live, 0, (live.len() / 4) as u32)?;
        }
        Ok(())
    }

    /// One token: `proj = mixer(h)`.
    fn decode(&self, gpu: &Qwen35Gpu, h: &MetalBuffer, proj: &MetalBuffer) -> Result<()> {
        self.qkv.gemv(gpu, h, &self.qkv_out)?;
        self.z.gemv(gpu, h, &self.z_out)?;
        self.alpha.gemv(gpu, h, &self.alpha_out)?;
        self.beta.gemv(gpu, h, &self.beta_raw)?;
        self.core(gpu)?;
        self.out.gemv(gpu, &self.gated, proj)
    }

    /// `n` chunk rows: `s.proj = mixer(s.h)`. Projections are batched
    /// GEMMs; the core runs chunk-wide (`chunk_core`) or per token on the
    /// decode kernels; both give identical results.
    fn prefill(
        &self,
        gpu: &Qwen35Gpu,
        s: &PrefillScratch,
        n: usize,
        mode: ChunkMode,
    ) -> Result<()> {
        let ctx = gpu.context();
        let spec = &self.spec;
        let nn = n as u32;
        let exact = mode.exact;
        self.qkv.gemm(gpu, &s.h, &s.a, nn, exact)?;
        self.z.gemm(gpu, &s.h, &s.b, nn, exact)?;
        self.alpha.gemm(gpu, &s.h, &s.c, nn, exact)?;
        self.beta.gemm(gpu, &s.h, &s.d, nn, exact)?;
        if mode.chunk_core {
            self.core_chunk(gpu, s, n, mode.capture)?;
            return self.out.gemm(gpu, &s.e, &s.proj, nn, exact);
        }
        if mode.capture {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                "rollback capture needs the chunk-wide core",
            ));
        }
        let ch = spec.channels;
        let vz = spec.head_v * spec.dt_rank;
        let dr = spec.dt_rank;
        for i in 0..n {
            ctx.copy_f32(&s.a, (i * ch) as u32, &self.qkv_out, 0, ch as u32)?;
            ctx.copy_f32(&s.b, (i * vz) as u32, &self.z_out, 0, vz as u32)?;
            ctx.copy_f32(&s.c, (i * dr) as u32, &self.alpha_out, 0, dr as u32)?;
            ctx.copy_f32(&s.d, (i * dr) as u32, &self.beta_raw, 0, dr as u32)?;
            self.core(gpu)?;
            ctx.copy_f32(&self.gated, 0, &s.e, (i * vz) as u32, vz as u32)?;
        }
        self.out.gemm(gpu, &s.e, &s.proj, nn, exact)
    }

    /// The core over all `n` chunk tokens at once, from the projected rows
    /// `s.a` (qkv), `s.b` (z), `s.c` (alpha), `s.d` (raw beta) to `s.e`
    /// (gated), with exactly [`Self::core`]'s per-element arithmetic. With
    /// `capture`, the recurrent state and convolution history after the
    /// first token are also written to the rollback buffers.
    fn core_chunk(
        &self,
        gpu: &Qwen35Gpu,
        s: &PrefillScratch,
        n: usize,
        capture: bool,
    ) -> Result<()> {
        let ctx = gpu.context();
        let spec = &self.spec;
        let nn = n as u32;
        let key = (spec.groups * spec.state_size) as u32;
        let value = (spec.head_v * spec.dt_rank) as u32;
        let rollback = if capture {
            let Some(rb) = &self.rollback else {
                return Err(TuckmillError::new(
                    ErrorCode::Internal,
                    "rollback capture on a block loaded without speculation",
                ));
            };
            // History after token 0 = the history before the chunk shifted
            // by token 0's input row.
            ctx.copy_f32(&self.hist, 0, &rb.1, 0, (self.hist.len() / 4) as u32)?;
            ctx.qwen35_conv_chunk_shift(
                &rb.1,
                &s.a,
                spec.channels as u32,
                spec.conv_kernel as u32,
                1,
            )?;
            Some(rb)
        } else {
            None
        };
        ctx.qwen35_conv_chunk(
            &self.hist,
            &s.a,
            &self.conv_w,
            &s.gconv,
            spec.channels as u32,
            spec.conv_kernel as u32,
            nn,
        )?;
        ctx.qwen35_silu(&s.gconv, nn * spec.channels as u32)?;
        ctx.qwen35_split_qkv_chunk(&s.gconv, &s.gq, &s.gk, &s.gv, key, nn)?;
        let rows = nn * spec.groups as u32;
        gpu.l2norm_dev(&s.gq, &s.gq, rows, spec.state_size as u32, self.eps)?;
        gpu.l2norm_dev(&s.gk, &s.gk, rows, spec.state_size as u32, self.eps)?;
        ctx.qwen35_beta_decay_chunk(
            &s.d,
            &s.c,
            &self.ssm_dt,
            &self.ssm_a,
            &s.gbeta,
            &s.ggate,
            spec.dt_rank as u32,
            nn,
        )?;
        let q_scale = 1.0 / (spec.state_size as f32).sqrt();
        let strides = (key, value, spec.dt_rank as u32);
        match rollback {
            Some(rb) => ctx.qwen35_gdn_chunk_snap(
                &s.gq,
                &s.gk,
                &s.gv,
                &s.ggate,
                &s.gbeta,
                &self.state,
                &s.gout,
                (&rb.0, 0),
                spec.groups as u32,
                spec.state_size as u32,
                spec.head_v as u32,
                q_scale,
                nn,
                strides,
            )?,
            None => ctx.qwen35_gdn_chunk(
                &s.gq,
                &s.gk,
                &s.gv,
                &s.ggate,
                &s.gbeta,
                &self.state,
                &s.gout,
                spec.groups as u32,
                spec.state_size as u32,
                spec.head_v as u32,
                q_scale,
                nn,
                strides,
            )?,
        }
        gpu.gated_norm_dev(
            &s.gout,
            &s.b,
            &self.ssm_norm,
            &s.e,
            nn * spec.dt_rank as u32,
            spec.head_v as u32,
            self.eps,
        )
    }

    /// The sequential per-token core, from `qkv_out`/`z_out`/`alpha_out`/
    /// `beta_raw` to `gated`.
    fn core(&self, gpu: &Qwen35Gpu) -> Result<()> {
        let ctx = gpu.context();
        let s = &self.spec;
        gpu.conv_step_dev(
            &self.hist,
            &self.qkv_out,
            &self.conv_w,
            &self.conv_y,
            s.channels as u32,
            s.conv_kernel as u32,
        )?;
        ctx.qwen35_silu(&self.conv_y, s.channels as u32)?;
        ctx.qwen35_split_qkv(
            &self.conv_y,
            &self.q,
            &self.k,
            &self.v,
            (s.groups * s.state_size) as u32,
        )?;
        gpu.l2norm_dev(
            &self.q,
            &self.q,
            s.groups as u32,
            s.state_size as u32,
            self.eps,
        )?;
        gpu.l2norm_dev(
            &self.k,
            &self.k,
            s.groups as u32,
            s.state_size as u32,
            self.eps,
        )?;
        gpu.beta_decay_dev(
            &self.beta_raw,
            &self.alpha_out,
            &self.ssm_dt,
            &self.ssm_a,
            &self.beta_out,
            &self.gate,
            s.dt_rank as u32,
        )?;
        gpu.gdn_step_dev(
            &self.q,
            &self.k,
            &self.v,
            &self.gate,
            &self.beta_out,
            &self.state,
            &self.o,
            s.groups as u32,
            s.state_size as u32,
            s.head_v as u32,
            1.0 / (s.state_size as f32).sqrt(),
        )?;
        gpu.gated_norm_dev(
            &self.o,
            &self.z_out,
            &self.ssm_norm,
            &self.gated,
            s.dt_rank as u32,
            s.head_v as u32,
            self.eps,
        )
    }
}

/// Grouped-query attention with Q/K norms, partial split-half RoPE,
/// packed f16 KV and a sigmoid output gate. Other variants are refused at
/// load until their block lands (never approximated).
/// Declared KV cache precision of attention layers (part of the numerical
/// profile; never substituted). `F16` stores packed f16 entries (the values
/// an f16-rounding f32 cache would hold); `F32` stores exact f32 entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCache {
    F16,
    F32,
}

impl KvCache {
    fn entry_bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }
}

struct AttentionBlock {
    spec: AttentionSpec,
    kv: KvCache,
    eps: f32,
    q_w: Matrix,
    k_w: Matrix,
    v_w: Matrix,
    o_w: Matrix,
    q_norm: MetalBuffer,
    k_norm: MetalBuffer,
    inv_freq: MetalBuffer,
    k_cache: MetalBuffer,
    v_cache: MetalBuffer,
    /// Joint Q + gate projection output (gated attention only).
    qg: Option<MetalBuffer>,
    q: MetalBuffer,
    /// Sigmoid output gate (gated attention only).
    gate: Option<MetalBuffer>,
    k: MetalBuffer,
    v: MetalBuffer,
    attn: MetalBuffer,
    kv_scratch: MetalBuffer,
    cache_cap: usize,
}

impl AttentionBlock {
    fn check_supported(s: &AttentionSpec) -> Result<(Tensor, Tensor)> {
        match (s.q_norm, s.k_norm) {
            (Some(q_norm), Some(k_norm)) => Ok((q_norm, k_norm)),
            _ => Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "attention variant (output gate {}, q_norm {}, k_norm {}) has no engine block yet; only attention with Q/K norms is implemented",
                    s.output_gate,
                    s.q_norm.is_some(),
                    s.k_norm.is_some()
                ),
            )
            .with_action("Wait for the matching attention block; variants are never approximated.")),
        }
    }

    fn kv_plane_bytes(s: &AttentionSpec, capacity: usize, kv: KvCache) -> u64 {
        (s.kv_heads * capacity * s.head_dim * kv.entry_bytes()) as u64
    }

    /// `[qg, q, gate, k, v, attn, kv_scratch]` in f32 elements; `qg` and
    /// `gate` are 0 (not allocated) without an output gate.
    fn scratch_elems(s: &AttentionSpec, capacity: usize) -> [usize; 7] {
        let hd = s.heads * s.head_dim;
        let kv = s.kv_heads * s.head_dim;
        let gated = if s.output_gate { hd } else { 0 };
        [gated * 2, hd, gated, kv, kv, hd, s.heads * capacity]
    }

    fn plan(
        s: &AttentionSpec,
        src: &dyn WeightSource,
        capacity: usize,
        kv: KvCache,
        p: &mut Plan,
    ) -> Result<()> {
        let (q_norm, k_norm) = Self::check_supported(s)?;
        for m in [s.q, s.k, s.v, s.o] {
            p.weights += Matrix::plan(src, m)?;
        }
        p.weights_f32(vector_len(src, q_norm)?);
        p.weights_f32(vector_len(src, k_norm)?);
        // The RoPE frequency table is derived by the engine, not artifact
        // weight bytes, so it is priced as scratch (the planner's weights
        // reservation is exactly the artifact's tensors).
        p.scratch_f32(s.rope_dims / 2);
        p.state += 2 * Self::kv_plane_bytes(s, capacity, kv);
        for elems in Self::scratch_elems(s, capacity) {
            if elems > 0 {
                p.scratch_f32(elems);
            }
        }
        Ok(())
    }

    fn load(
        gpu: &Qwen35Gpu,
        src: &dyn WeightSource,
        s: &AttentionSpec,
        eps: f32,
        capacity: usize,
        kv: KvCache,
    ) -> Result<Self> {
        let (q_norm, k_norm) = Self::check_supported(s)?;
        let [qg, q, gate, k, v, attn, kv_scratch] = Self::scratch_elems(s, capacity);
        let plane = Self::kv_plane_bytes(s, capacity, kv) as usize;
        let optional = |elems: usize| -> Result<Option<MetalBuffer>> {
            if elems == 0 {
                Ok(None)
            } else {
                zeros(gpu, elems).map(Some)
            }
        };
        Ok(Self {
            q_w: Matrix::load(gpu, src, s.q)?,
            k_w: Matrix::load(gpu, src, s.k)?,
            v_w: Matrix::load(gpu, src, s.v)?,
            o_w: Matrix::load(gpu, src, s.o)?,
            q_norm: vector(gpu, src, q_norm)?,
            k_norm: vector(gpu, src, k_norm)?,
            inv_freq: floats(
                gpu,
                &crate::qwen35_ops::rope_inv_freq_partial(s.rope_dims, s.rope_base),
            )?,
            k_cache: gpu.buffer(plane)?,
            v_cache: gpu.buffer(plane)?,
            qg: optional(qg)?,
            q: zeros(gpu, q)?,
            gate: optional(gate)?,
            k: zeros(gpu, k)?,
            v: zeros(gpu, v)?,
            attn: zeros(gpu, attn)?,
            kv_scratch: zeros(gpu, kv_scratch)?,
            cache_cap: capacity,
            spec: s.clone(),
            kv,
            eps,
        })
    }

    fn reset(&self) -> Result<()> {
        zero_fill(&self.k_cache)?;
        zero_fill(&self.v_cache)
    }

    fn decode(&self, gpu: &Qwen35Gpu, h: &MetalBuffer, proj: &MetalBuffer, pos: u32) -> Result<()> {
        self.q_w.gemv(gpu, h, self.qg.as_ref().unwrap_or(&self.q))?;
        self.k_w.gemv(gpu, h, &self.k)?;
        self.v_w.gemv(gpu, h, &self.v)?;
        self.core(gpu, pos)?;
        self.o_w.gemv(gpu, &self.attn, proj)
    }

    fn prefill(
        &self,
        gpu: &Qwen35Gpu,
        s: &PrefillScratch,
        n: usize,
        pos0: u32,
        mode: ChunkMode,
    ) -> Result<()> {
        let ctx = gpu.context();
        let spec = &self.spec;
        let nn = n as u32;
        let exact = mode.exact;
        self.q_w.gemm(gpu, &s.h, &s.a, nn, exact)?;
        self.k_w.gemm(gpu, &s.h, &s.b, nn, exact)?;
        self.v_w.gemm(gpu, &s.h, &s.c, nn, exact)?;
        if mode.chunk_core {
            self.core_chunk(gpu, s, n, pos0)?;
            return self.o_w.gemm(gpu, &s.e, &s.proj, nn, exact);
        }
        let aw = spec.heads * spec.head_dim;
        let qgw = if spec.output_gate { aw * 2 } else { aw };
        let kvw = spec.kv_heads * spec.head_dim;
        let q_dst = self.qg.as_ref().unwrap_or(&self.q);
        for i in 0..n {
            ctx.copy_f32(&s.a, (i * qgw) as u32, q_dst, 0, qgw as u32)?;
            ctx.copy_f32(&s.b, (i * kvw) as u32, &self.k, 0, kvw as u32)?;
            ctx.copy_f32(&s.c, (i * kvw) as u32, &self.v, 0, kvw as u32)?;
            self.core(gpu, pos0 + i as u32)?;
            ctx.copy_f32(&self.attn, 0, &s.e, (i * aw) as u32, aw as u32)?;
        }
        self.o_w.gemm(gpu, &s.e, &s.proj, nn, exact)
    }

    /// The core over all `n` chunk queries at positions `pos0..`, from the
    /// projected rows `s.a` (q+gate), `s.b` (k), `s.c` (v) to `s.e`, with
    /// exactly [`Self::core`]'s per-element arithmetic (every chunk K/V is
    /// stored before any chunk query attends).
    fn core_chunk(&self, gpu: &Qwen35Gpu, s: &PrefillScratch, n: usize, pos0: u32) -> Result<()> {
        let ctx = gpu.context();
        let spec = &self.spec;
        let nn = n as u32;
        let (heads, kv, hd) = (
            spec.heads as u32,
            spec.kv_heads as u32,
            spec.head_dim as u32,
        );
        // Query rows: split from the joint projection when gated, else the
        // projection output itself.
        let q = if spec.output_gate {
            gpu.q_gate_split_dev(&s.a, &s.aq, &s.ag, nn * heads, hd)?;
            &s.aq
        } else {
            &s.a
        };
        ctx.rmsnorm(q, &self.q_norm, q, nn * heads, hd, self.eps)?;
        ctx.rmsnorm(&s.b, &self.k_norm, &s.b, nn * kv, hd, self.eps)?;
        ctx.qwen35_rope_partial_chunk(
            q,
            &s.b,
            &self.inv_freq,
            pos0,
            heads,
            kv,
            hd,
            spec.rope_dims as u32,
            nn,
        )?;
        let f32_cache = self.kv == KvCache::F32;
        ctx.qwen35_kv_store_chunk(
            &s.b,
            &s.c,
            &self.k_cache,
            &self.v_cache,
            pos0,
            self.cache_cap as u32,
            hd,
            kv,
            nn,
            f32_cache,
        )?;
        ctx.qwen35_attention_chunk(
            q,
            &self.k_cache,
            &self.v_cache,
            &s.e,
            &s.ascr,
            heads,
            pos0,
            self.cache_cap as u32,
            hd,
            heads / kv,
            1.0 / (spec.head_dim as f32).sqrt(),
            nn,
            f32_cache,
        )?;
        if spec.output_gate {
            gpu.attn_gate_dev(&s.e, &s.ag, nn * heads * hd)?;
        }
        Ok(())
    }

    /// The sequential per-token core, from `qg`/`k`/`v` to `attn`.
    fn core(&self, gpu: &Qwen35Gpu, pos: u32) -> Result<()> {
        let ctx = gpu.context();
        let s = &self.spec;
        if let (Some(qg), Some(gate)) = (&self.qg, &self.gate) {
            gpu.q_gate_split_dev(qg, &self.q, gate, s.heads as u32, s.head_dim as u32)?;
        }
        ctx.rmsnorm(
            &self.q,
            &self.q_norm,
            &self.q,
            s.heads as u32,
            s.head_dim as u32,
            self.eps,
        )?;
        ctx.rmsnorm(
            &self.k,
            &self.k_norm,
            &self.k,
            s.kv_heads as u32,
            s.head_dim as u32,
            self.eps,
        )?;
        gpu.rope_partial_dev(
            &self.q,
            &self.k,
            &self.inv_freq,
            pos,
            s.heads as u32,
            s.kv_heads as u32,
            s.head_dim as u32,
            s.rope_dims as u32,
        )?;
        match self.kv {
            KvCache::F16 => {
                gpu.kv_store_f16_dev(
                    &self.k,
                    &self.v,
                    &self.k_cache,
                    &self.v_cache,
                    pos,
                    self.cache_cap as u32,
                    s.head_dim as u32,
                    s.kv_heads as u32,
                )?;
                gpu.attention_f16_dev(
                    &self.q,
                    &self.k_cache,
                    &self.v_cache,
                    &self.attn,
                    &self.kv_scratch,
                    s.heads as u32,
                    pos,
                    self.cache_cap as u32,
                    s.head_dim as u32,
                    (s.heads / s.kv_heads) as u32,
                    1.0 / (s.head_dim as f32).sqrt(),
                )?;
            }
            // f32 KV runs the chunk kernels with n = 1, so decode and
            // chunked prefill share one compiled kernel (bit-identical).
            KvCache::F32 => {
                ctx.qwen35_kv_store_chunk(
                    &self.k,
                    &self.v,
                    &self.k_cache,
                    &self.v_cache,
                    pos,
                    self.cache_cap as u32,
                    s.head_dim as u32,
                    s.kv_heads as u32,
                    1,
                    true,
                )?;
                ctx.qwen35_attention_chunk(
                    &self.q,
                    &self.k_cache,
                    &self.v_cache,
                    &self.attn,
                    &self.kv_scratch,
                    s.heads as u32,
                    pos,
                    self.cache_cap as u32,
                    s.head_dim as u32,
                    (s.heads / s.kv_heads) as u32,
                    1.0 / (s.head_dim as f32).sqrt(),
                    1,
                    true,
                )?;
            }
        }
        if let Some(gate) = &self.gate {
            gpu.attn_gate_dev(&self.attn, gate, (s.heads * s.head_dim) as u32)?;
        }
        Ok(())
    }
}

/// Zero a session-state buffer on the device side (no host staging: the
/// earlier host zero vectors, up to ~2 MB each, were allocated on every
/// request reset).
fn zero_fill(buf: &MetalBuffer) -> Result<()> {
    buf.zero(0, buf.len())
}

enum Mixer {
    Gdn(Box<GdnBlock>),
    Attention(Box<AttentionBlock>),
}

/// One layer: `x += mixer(norm(x)); x += ffn(norm(x))`.
struct Layer {
    pre_norm: MetalBuffer,
    post_norm: MetalBuffer,
    h: MetalBuffer,
    h2: MetalBuffer,
    proj: MetalBuffer,
    mixer: Mixer,
    ffn: SwiGluBlock,
}

impl Layer {
    fn plan(
        spec: &super::graph::LayerSpec,
        graph: &ModelGraph,
        src: &dyn WeightSource,
        capacity: usize,
        opts: EngineOptions,
        p: &mut Plan,
    ) -> Result<()> {
        p.weights_f32(vector_len(src, spec.pre_norm)?);
        p.weights_f32(vector_len(src, spec.post_norm)?);
        p.scratch_f32(graph.hidden); // h
        p.scratch_f32(graph.hidden); // h2
        p.scratch_f32(graph.hidden); // proj
        match &spec.mixer {
            MixerSpec::GatedDeltaNet(g) => GdnBlock::plan(g, src, opts, p)?,
            MixerSpec::Attention(a) => AttentionBlock::plan(a, src, capacity, opts.kv, p)?,
        }
        SwiGluBlock::plan(&spec.ffn, graph.hidden, src, p)
    }

    fn load(
        gpu: &Qwen35Gpu,
        src: &dyn WeightSource,
        spec: &super::graph::LayerSpec,
        graph: &ModelGraph,
        capacity: usize,
        opts: EngineOptions,
    ) -> Result<Self> {
        Ok(Self {
            pre_norm: vector(gpu, src, spec.pre_norm)?,
            post_norm: vector(gpu, src, spec.post_norm)?,
            h: zeros(gpu, graph.hidden)?,
            h2: zeros(gpu, graph.hidden)?,
            proj: zeros(gpu, graph.hidden)?,
            mixer: match &spec.mixer {
                MixerSpec::GatedDeltaNet(g) => {
                    Mixer::Gdn(Box::new(GdnBlock::load(gpu, src, g, graph.eps, opts)?))
                }
                MixerSpec::Attention(a) => Mixer::Attention(Box::new(AttentionBlock::load(
                    gpu, src, a, graph.eps, capacity, opts.kv,
                )?)),
            },
            ffn: SwiGluBlock::load(gpu, src, &spec.ffn, graph.hidden)?,
        })
    }

    fn reset(&self) -> Result<()> {
        match &self.mixer {
            Mixer::Gdn(b) => b.reset(),
            Mixer::Attention(b) => b.reset(),
        }
    }

    /// Save or restore this layer's prefix snapshot. Attention layers hold
    /// nothing extra: their KV rows below the prefix boundary stay in place
    /// and are never overwritten by a request that extends the prefix.
    fn copy_snapshot(&self, gpu: &Qwen35Gpu, save: bool) -> Result<()> {
        match &self.mixer {
            Mixer::Gdn(b) => b.copy_snapshot(gpu, save),
            Mixer::Attention(_) => Ok(()),
        }
    }

    /// Undo the second token of a rejected speculative verify (recurrent
    /// layers; attention rows past the position are simply rewritten).
    fn roll_back(&self, gpu: &Qwen35Gpu) -> Result<()> {
        match &self.mixer {
            Mixer::Gdn(b) => b.roll_back(gpu),
            Mixer::Attention(_) => Ok(()),
        }
    }

    fn decode(
        &self,
        gpu: &Qwen35Gpu,
        x: &MetalBuffer,
        pos: u32,
        hidden: usize,
        eps: f32,
    ) -> Result<()> {
        let ctx = gpu.context();
        ctx.rmsnorm(x, &self.pre_norm, &self.h, 1, hidden as u32, eps)?;
        match &self.mixer {
            Mixer::Gdn(b) => b.decode(gpu, &self.h, &self.proj)?,
            Mixer::Attention(b) => b.decode(gpu, &self.h, &self.proj, pos)?,
        }
        ctx.f32_axpy(&self.proj, x, 1.0, hidden as u32)?;
        ctx.rmsnorm(x, &self.post_norm, &self.h2, 1, hidden as u32, eps)?;
        self.ffn.decode(gpu, &self.h2, x, hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        gpu: &Qwen35Gpu,
        s: &PrefillScratch,
        n: usize,
        pos0: u32,
        hidden: usize,
        eps: f32,
        mode: ChunkMode,
    ) -> Result<()> {
        let ctx = gpu.context();
        let nn = n as u32;
        ctx.rmsnorm(&s.x, &self.pre_norm, &s.h, nn, hidden as u32, eps)?;
        match &self.mixer {
            Mixer::Gdn(b) => b.prefill(gpu, s, n, mode)?,
            Mixer::Attention(b) => b.prefill(gpu, s, n, pos0, mode)?,
        }
        ctx.f32_axpy(&s.proj, &s.x, 1.0, (n * hidden) as u32)?;
        ctx.rmsnorm(&s.x, &self.post_norm, &s.h2, nn, hidden as u32, eps)?;
        self.ffn.prefill(gpu, s, n, hidden, mode.exact)
    }
}

// --------------------------------------------------------------------- MTP --

/// How the MTP head reads its inputs. The artifact does not record these
/// conventions; they only change the draft (never the served output), so
/// they are chosen by measured acceptance (ADR-012 step 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpConvention {
    /// `[enorm(e); hnorm(h)]` (true) or `[hnorm(h); enorm(e)]`.
    pub embedding_first: bool,
    /// The main hidden state after the model's final norm (true) or the
    /// raw residual stream.
    pub hidden_post_norm: bool,
}

impl Default for MtpConvention {
    fn default() -> Self {
        Self {
            embedding_first: true,
            hidden_post_norm: true,
        }
    }
}

/// Draft token slot and the two verify slots of [`MtpHead::ids`].
const SLOT_DRAFT: u32 = 0;
const SLOT_VERIFY: u32 = 1;

/// The multi-token-prediction head and the buffers of speculative greedy
/// decoding (ADR-012 step 3).
struct MtpHead {
    enorm: MetalBuffer,
    hnorm: MetalBuffer,
    final_norm: MetalBuffer,
    eh_proj: Matrix,
    /// One attention layer with its own KV rows (MTP positions).
    layer: Layer,
    /// `[PREFILL_BATCH, hidden]` next-token embeddings of the MTP rows.
    e: MetalBuffer,
    /// `[PREFILL_BATCH, hidden]` main hidden states of the MTP rows.
    hin: MetalBuffer,
    /// `[PREFILL_BATCH, 2 * hidden]` concatenated projection input.
    cat: MetalBuffer,
    /// `[2, hidden]` main hidden states of the rows the next draft runs.
    hid: MetalBuffer,
    /// Prefix-cache snapshot of `hid` row 0 (with the prefix cache).
    hid_snap: Option<MetalBuffer>,
    /// `[2, vocab]` verify logits.
    logits: MetalBuffer,
    /// Draft and verify token ids.
    ids: MetalBuffer,
}

impl MtpHead {
    fn scratch_elems(graph: &ModelGraph) -> [usize; 6] {
        let (h, b) = (graph.hidden, PREFILL_BATCH);
        [b * h, b * h, b * 2 * h, 2 * h, 2 * graph.vocab, 3]
    }

    fn plan(
        spec: &super::graph::MtpSpec,
        graph: &ModelGraph,
        src: &dyn WeightSource,
        capacity: usize,
        opts: EngineOptions,
        p: &mut Plan,
    ) -> Result<()> {
        for v in [spec.enorm, spec.hnorm, spec.final_norm] {
            p.weights_f32(vector_len(src, v)?);
        }
        p.weights += Matrix::plan(src, spec.eh_proj)?;
        // Everything else is an optional speculation buffer outside the
        // planner's served-stack KV, so it is priced as workspace (the
        // head's KV rows included).
        let mut layer = Plan::default();
        Layer::plan(&spec.layer, graph, src, capacity, opts, &mut layer)?;
        p.weights += layer.weights;
        p.scratch += layer.state + layer.scratch;
        for elems in Self::scratch_elems(graph) {
            p.scratch_f32(elems);
        }
        if opts.prefix_cache {
            p.scratch_f32(graph.hidden);
        }
        Ok(())
    }

    fn load(
        gpu: &Qwen35Gpu,
        src: &dyn WeightSource,
        spec: &super::graph::MtpSpec,
        graph: &ModelGraph,
        capacity: usize,
        opts: EngineOptions,
    ) -> Result<Self> {
        let [e, hin, cat, hid, logits, ids] = Self::scratch_elems(graph);
        Ok(Self {
            enorm: vector(gpu, src, spec.enorm)?,
            hnorm: vector(gpu, src, spec.hnorm)?,
            final_norm: vector(gpu, src, spec.final_norm)?,
            eh_proj: Matrix::load(gpu, src, spec.eh_proj)?,
            layer: Layer::load(gpu, src, &spec.layer, graph, capacity, opts)?,
            e: zeros(gpu, e)?,
            hin: zeros(gpu, hin)?,
            cat: zeros(gpu, cat)?,
            hid: zeros(gpu, hid)?,
            hid_snap: if opts.prefix_cache {
                Some(zeros(gpu, graph.hidden)?)
            } else {
                None
            },
            logits: zeros(gpu, logits)?,
            ids: zeros(gpu, ids)?,
        })
    }

    /// Run `m` MTP rows at positions `pos0..` from `e` / `hin` (rows
    /// `0..m`); the outputs are left in `s.x` rows `0..m`.
    #[allow(clippy::too_many_arguments)]
    fn rows(
        &self,
        gpu: &Qwen35Gpu,
        s: &PrefillScratch,
        m: usize,
        pos0: u32,
        main_final_norm: &MetalBuffer,
        conv: MtpConvention,
        hidden: usize,
        eps: f32,
    ) -> Result<()> {
        let ctx = gpu.context();
        let (mm, hh) = (m as u32, hidden as u32);
        ctx.rmsnorm(&self.e, &self.enorm, &self.e, mm, hh, eps)?;
        if conv.hidden_post_norm {
            ctx.rmsnorm(&self.hin, main_final_norm, &self.hin, mm, hh, eps)?;
        }
        ctx.rmsnorm(&self.hin, &self.hnorm, &self.hin, mm, hh, eps)?;
        let (e_off, h_off) = if conv.embedding_first {
            (0, hh)
        } else {
            (hh, 0)
        };
        for r in 0..mm {
            ctx.copy_f32(&self.e, r * hh, &self.cat, 2 * r * hh + e_off, hh)?;
            ctx.copy_f32(&self.hin, r * hh, &self.cat, 2 * r * hh + h_off, hh)?;
        }
        self.eh_proj.gemm(gpu, &self.cat, &s.x, mm, false)?;
        self.layer
            .prefill(gpu, s, m, pos0, hidden, eps, ChunkMode::prefill(true))
    }
}

/// One speculative greedy cycle's result: `first` is the main model's
/// token after the input token (always committed); `second` follows it
/// when the draft matched `first` (`accepted`). `None` tokens mean every
/// logit was NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecStep {
    pub first: Option<u32>,
    pub second: Option<u32>,
    pub accepted: bool,
}

// ------------------------------------------------------------------ engine --

/// Exact device allocation for a graph at `capacity` positions, computed
/// before anything allocates; after [`EngineModel::load`] the ledger
/// equals its total.
pub fn alloc_plan(
    graph: &ModelGraph,
    src: &dyn WeightSource,
    capacity: usize,
) -> Result<GpuAllocPlan> {
    alloc_plan_with(graph, src, capacity, EngineOptions::default())
}

/// Load-time engine options (part of the exact allocation plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineOptions {
    /// Declared KV cache precision of attention layers.
    pub kv: KvCache,
    /// Keep one prefix snapshot (ADR-012 step 2): the recurrent and
    /// convolution state of every Gated DeltaNet layer at a chunk
    /// boundary, so a request that extends the snapshotted prefix resumes
    /// there instead of prefilling it again.
    pub prefix_cache: bool,
    /// Load the model's multi-token-prediction block and the buffers of
    /// speculative greedy decoding (ADR-012 step 3). Refused for a graph
    /// without an MTP block.
    pub mtp: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            kv: KvCache::F16,
            prefix_cache: false,
            mtp: false,
        }
    }
}

/// [`alloc_plan`] for explicit engine options.
pub fn alloc_plan_with(
    graph: &ModelGraph,
    src: &dyn WeightSource,
    capacity: usize,
    opts: EngineOptions,
) -> Result<GpuAllocPlan> {
    let mut p = Plan::default();
    for layer in &graph.layers {
        Layer::plan(layer, graph, src, capacity, opts, &mut p)?;
    }
    p.weights += Matrix::plan(src, graph.embedding)?;
    if graph.head != graph.embedding {
        p.weights += Matrix::plan(src, graph.head)?;
    }
    p.weights_f32(vector_len(src, graph.final_norm)?);
    p.scratch_f32(graph.hidden); // x
    p.scratch_f32(graph.hidden); // h
    p.scratch_f32(graph.vocab); // logits
    p.scratch_f32(ARGMAX_PARTIAL_BYTES / 4); // argmax partials
    p.scratch_f32(2); // token slots (pipelined greedy)
    PrefillScratch::plan(graph, capacity, &mut p);
    if opts.mtp {
        MtpHead::plan(mtp_spec(graph)?, graph, src, capacity, opts, &mut p)?;
    }
    Ok(GpuAllocPlan {
        weights: p.weights,
        state: p.state,
        scratch: p.scratch,
    })
}

fn mtp_spec(graph: &ModelGraph) -> Result<&super::graph::MtpSpec> {
    graph.mtp.as_ref().ok_or_else(|| {
        TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "{} artifact has no multi-token-prediction block; speculative decoding needs one",
                graph.architecture
            ),
        )
        .with_action("Serve without speculation; the draft head is never substituted.")
    })
}

/// Any [`ModelGraph`] resident on the GPU, with one session.
pub struct EngineModel {
    gpu: Qwen35Gpu,
    graph: ModelGraph,
    layers: Vec<Layer>,
    embedding: Matrix,
    /// `None` when the head is tied to the embedding.
    head: Option<Matrix>,
    final_norm: MetalBuffer,
    x: MetalBuffer,
    h: MetalBuffer,
    logits_buf: MetalBuffer,
    argmax_partial: MetalBuffer,
    /// Greedy token ids, two alternating slots (see [`token_slot`]).
    token_ids: MetalBuffer,
    prefill: PrefillScratch,
    capacity: usize,
    positions: std::cell::Cell<usize>,
    session_valid: std::cell::Cell<bool>,
    grouped: std::cell::Cell<bool>,
    /// Commit counts of the prefill chunks still possibly in flight.
    prefill_in_flight: std::cell::RefCell<std::collections::VecDeque<u64>>,
    /// Reused host staging for the logits download (one block for the
    /// session instead of one allocation per generated token).
    logits_staging: std::cell::RefCell<Vec<u8>>,
    /// Prefill runs mixer cores chunk-wide (default) instead of per token.
    chunk_core: std::cell::Cell<bool>,
    /// Whether the prefix cache was allocated (see [`EngineOptions`]).
    prefix_cache: bool,
    prefix: std::cell::RefCell<PrefixState>,
    /// The draft head (with [`EngineOptions::mtp`]).
    mtp: Option<MtpHead>,
    /// Tokens of the MTP rows the next draft runs first (their hidden
    /// states are `MtpHead::hid` rows); the last is the next input token.
    /// Empty when speculation cannot continue the session.
    mtp_pending: std::cell::RefCell<Vec<u32>>,
    mtp_convention: std::cell::Cell<MtpConvention>,
}

/// The prefix cache's host-side bookkeeping. The snapshot is valid only
/// while the session's KV rows below its boundary still hold exactly
/// `tokens`; every [`EngineModel::reset`] invalidates it.
#[derive(Default)]
struct PrefixState {
    /// Tokens of the snapshotted prefix (its length is the boundary, a
    /// multiple of [`PREFILL_BATCH`]); valid only when `valid`.
    tokens: Vec<u32>,
    valid: bool,
    /// The boundary this request's prefill snapshots at, with its tokens.
    planned: Option<Vec<u32>>,
    /// Positions of `planned` already confirmed written by this request's
    /// prefill chunks (starting at the restored position). A chunk that
    /// does not continue the plan exactly drops it.
    matched: usize,
}

impl EngineModel {
    /// Preflight, admission check, then upload. No buffer of this model
    /// exists before `admitted_gpu_bytes` covers the exact plan.
    pub fn load(
        gpu: Qwen35Gpu,
        graph: ModelGraph,
        src: &dyn WeightSource,
        capacity: usize,
        admitted_gpu_bytes: u64,
    ) -> Result<Self> {
        Self::load_with(
            gpu,
            graph,
            src,
            capacity,
            admitted_gpu_bytes,
            EngineOptions::default(),
        )
    }

    /// [`Self::load`] with explicit engine options.
    pub fn load_with(
        gpu: Qwen35Gpu,
        graph: ModelGraph,
        src: &dyn WeightSource,
        capacity: usize,
        admitted_gpu_bytes: u64,
        opts: EngineOptions,
    ) -> Result<Self> {
        let plan = alloc_plan_with(&graph, src, capacity, opts)?;
        let total = plan.total();
        if total > admitted_gpu_bytes {
            return Err(TuckmillError::new(
                ErrorCode::InvalidPlan,
                format!(
                    "GPU admission of {admitted_gpu_bytes} B does not cover the exact plan of {total} B (weights {} + state {} + scratch {}); nothing was allocated",
                    plan.weights, plan.state, plan.scratch
                ),
            )
            .with_action(
                "Raise the GPU admission (memory budget or plan reservations) so it covers the preflighted model allocation.",
            ));
        }
        let layers = graph
            .layers
            .iter()
            .map(|spec| Layer::load(&gpu, src, spec, &graph, capacity, opts))
            .collect::<Result<Vec<_>>>()?;
        let embedding = Matrix::load(&gpu, src, graph.embedding)?;
        let head = if graph.head != graph.embedding {
            Some(Matrix::load(&gpu, src, graph.head)?)
        } else {
            None
        };
        let final_norm = vector(&gpu, src, graph.final_norm)?;
        let x = zeros(&gpu, graph.hidden)?;
        let h = zeros(&gpu, graph.hidden)?;
        let logits_buf = zeros(&gpu, graph.vocab)?;
        let argmax_partial = zeros(&gpu, ARGMAX_PARTIAL_BYTES / 4)?;
        let token_ids = zeros(&gpu, 2)?;
        let prefill = PrefillScratch::load(&gpu, &graph, capacity)?;
        let mtp = if opts.mtp {
            Some(MtpHead::load(
                &gpu,
                src,
                mtp_spec(&graph)?,
                &graph,
                capacity,
                opts,
            )?)
        } else {
            None
        };
        Ok(Self {
            gpu,
            layers,
            embedding,
            head,
            final_norm,
            x,
            h,
            logits_buf,
            argmax_partial,
            token_ids,
            prefill,
            capacity,
            graph,
            positions: std::cell::Cell::new(0),
            session_valid: std::cell::Cell::new(true),
            grouped: std::cell::Cell::new(true),
            prefill_in_flight: Default::default(),
            logits_staging: Default::default(),
            chunk_core: std::cell::Cell::new(true),
            prefix_cache: opts.prefix_cache,
            prefix: Default::default(),
            mtp,
            mtp_pending: Default::default(),
            mtp_convention: Default::default(),
        })
    }

    /// Block until fewer than [`MAX_PREFILL_IN_FLIGHT`] prefill chunks are
    /// in flight. Command buffers complete in queue order, so waiting for a
    /// chunk's commit count also covers everything before it.
    fn bound_prefill_in_flight(&self) {
        let ctx = self.gpu.context();
        let mut in_flight = self.prefill_in_flight.borrow_mut();
        while in_flight.len() >= MAX_PREFILL_IN_FLIGHT {
            let Some(target) = in_flight.pop_front() else {
                break;
            };
            while ctx.completed_dispatches() < target {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        }
    }

    pub fn graph(&self) -> &ModelGraph {
        &self.graph
    }

    pub fn positions(&self) -> usize {
        self.positions.get()
    }

    /// Total bytes held on the device (weights, state, scratch).
    pub fn allocated_bytes(&self) -> u64 {
        self.gpu.context().allocated_bytes()
    }

    /// The model's Metal context (dispatch counts, GPU time, profiling).
    pub fn context(&self) -> &MetalContext {
        self.gpu.context()
    }

    /// One command buffer per step/chunk (`true`, default) or one per
    /// operator (profiling attribution). Arithmetic is identical.
    pub fn set_grouped(&self, grouped: bool) {
        self.grouped.set(grouped);
    }

    /// Run prefill mixer cores chunk-wide (`true`, default) or per token on
    /// the decode kernels (`false`, the reference the chunk path is gated
    /// against). Results are identical.
    pub fn set_chunk_core(&self, chunk: bool) {
        self.chunk_core.set(chunk);
    }

    /// Zero every session state and the position counter. Waits for any
    /// in-flight step first (a pipelined step may still be running when a
    /// request ends at EOS or a stop sequence).
    pub fn reset(&self) -> Result<()> {
        self.drain()?;
        self.prefill_in_flight.borrow_mut().clear();
        // Zeroing the KV planes invalidates the prefix snapshot.
        *self.prefix.borrow_mut() = PrefixState::default();
        self.mtp_pending.borrow_mut().clear();
        for layer in &self.layers {
            layer.reset()?;
        }
        if let Some(mtp) = &self.mtp {
            mtp.layer.reset()?;
        }
        self.positions.set(0);
        self.session_valid.set(true);
        Ok(())
    }

    /// Start a request for `prompt` (ADR-012 step 2). With the prefix cache,
    /// a prompt that strictly extends the snapshotted prefix resumes from
    /// the snapshot: the Gated DeltaNet state is restored on the device,
    /// attention KV rows below the boundary are kept, and the returned
    /// count of positions is already in the session. Otherwise the session
    /// is reset and 0 is returned. Either way this prompt's prefill will
    /// snapshot at its last chunk boundary below its length, so prefill
    /// chunks from the returned position are exactly those a cold prefill
    /// would run (bit-identical results by construction).
    pub fn begin_request(&self, prompt: &[u32]) -> Result<usize> {
        if !self.prefix_cache || prompt.is_empty() {
            self.reset()?;
            return Ok(0);
        }
        let hit = {
            let p = self.prefix.borrow();
            if p.valid
                && self.session_valid.get()
                && prompt.len() > p.tokens.len()
                && prompt.starts_with(&p.tokens)
            {
                p.tokens.len()
            } else {
                0
            }
        };
        if hit == 0 {
            self.reset()?;
        } else {
            self.drain()?;
            self.prefill_in_flight.borrow_mut().clear();
            self.mtp_pending.borrow_mut().clear();
            let restore = || -> Result<()> {
                for layer in &self.layers {
                    layer.copy_snapshot(&self.gpu, false)?;
                }
                self.mtp_snapshot(false)
            };
            if let Err(e) = self.gpu.grouped(restore) {
                self.session_valid.set(false);
                *self.prefix.borrow_mut() = PrefixState::default();
                return Err(e);
            }
            self.positions.set(hit);
        }
        let boundary = (prompt.len() - 1) / PREFILL_BATCH * PREFILL_BATCH;
        let mut p = self.prefix.borrow_mut();
        p.planned = if boundary > hit {
            Some(prompt[..boundary].to_vec())
        } else {
            None
        };
        p.matched = hit;
        Ok(hit)
    }

    fn head(&self) -> &Matrix {
        self.head.as_ref().unwrap_or(&self.embedding)
    }

    /// Save or restore the MTP head's prefix state: the main hidden state
    /// at the boundary's last position (its KV rows below stay in place).
    fn mtp_snapshot(&self, save: bool) -> Result<()> {
        let Some(mtp) = &self.mtp else {
            return Ok(());
        };
        let Some(snap) = &mtp.hid_snap else {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                "MTP prefix snapshot requested without the prefix cache",
            ));
        };
        let ctx = self.gpu.context();
        let hh = self.graph.hidden as u32;
        if save {
            ctx.copy_f32(&mtp.hid, 0, snap, 0, hh)
        } else {
            ctx.copy_f32(snap, 0, &mtp.hid, 0, hh)
        }
    }

    /// Whether the session can continue with [`Self::speculate_greedy`].
    pub fn can_speculate(&self) -> bool {
        self.mtp.is_some()
            && !self.mtp_pending.borrow().is_empty()
            && self.positions.get() + 2 <= self.capacity
    }

    /// Choose the MTP input conventions (acceptance experiments). Drafts
    /// change; served tokens never do.
    pub fn set_mtp_convention(&self, conv: MtpConvention) {
        self.mtp_convention.set(conv);
    }

    /// One speculative greedy cycle (ADR-012 step 3) after `token`, the
    /// last token this session produced: the MTP head drafts the token
    /// after next, then the main model verifies `[token, draft]` in one
    /// chunk with the decode arithmetic (per-row GEMMs, the decode cores),
    /// so `first` is bit-for-bit the token a decode step would choose, and
    /// `second` (only when the draft equals `first`) the token the decode
    /// step after it would choose. A rejected draft rolls the recurrent
    /// state back to after `token`.
    pub fn speculate_greedy(&self, token: u32) -> Result<SpecStep> {
        self.speculate(token, None)
    }

    /// [`Self::speculate_greedy`] with a host-chosen draft instead of the
    /// MTP head's (verification of accept and reject paths).
    pub fn speculate_with_draft(&self, token: u32, draft: u32) -> Result<SpecStep> {
        self.speculate(token, Some(draft))
    }

    fn speculate(&self, token: u32, forced: Option<u32>) -> Result<SpecStep> {
        self.ensure_valid()?;
        let Some(mtp) = &self.mtp else {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "speculation needs an engine loaded with the MTP head",
            ));
        };
        let pending = self.mtp_pending.borrow().clone();
        if pending.last() != Some(&token) {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "speculation must continue the session's last greedy token",
            )
            .with_action("Start speculation right after a greedy prefill or speculative step."));
        }
        let result = (|| {
            let draft_check = forced.unwrap_or(0);
            let p = self.check_tokens(&[token, draft_check])?;
            let m = pending.len();
            let (hidden, eps) = (self.graph.hidden, self.graph.eps);
            let hh = hidden as u32;
            let conv = self.mtp_convention.get();
            let s = &self.prefill;
            let ctx = self.gpu.context();
            let encode = || -> Result<()> {
                // Draft: the pending MTP rows end at position p - 1.
                for (r, t) in pending.iter().enumerate() {
                    self.embedding
                        .row_into(&self.gpu, *t, &mtp.e, (r * hidden) as u32)?;
                }
                ctx.copy_f32(&mtp.hid, 0, &mtp.hin, 0, m as u32 * hh)?;
                mtp.rows(
                    &self.gpu,
                    s,
                    m,
                    (p - m) as u32,
                    &self.final_norm,
                    conv,
                    hidden,
                    eps,
                )?;
                ctx.copy_f32(&s.x, (m as u32 - 1) * hh, &self.x, 0, hh)?;
                ctx.rmsnorm(&self.x, &mtp.final_norm, &self.h, 1, hh, eps)?;
                self.head().gemv(&self.gpu, &self.h, &self.logits_buf)?;
                ctx.argmax_f32(
                    &self.logits_buf,
                    self.graph.vocab as u32,
                    &self.argmax_partial,
                    &mtp.ids,
                    SLOT_DRAFT,
                )?;
                // Verify [token, draft] with the decode arithmetic.
                self.embedding.row_into(&self.gpu, token, &s.x, 0)?;
                match forced {
                    Some(d) => self.embedding.row_into(&self.gpu, d, &s.x, hh)?,
                    None => {
                        self.embedding
                            .row_indirect(&self.gpu, &mtp.ids, SLOT_DRAFT, &self.x)?;
                        ctx.copy_f32(&self.x, 0, &s.x, hh, hh)?;
                    }
                }
                let mode = ChunkMode {
                    chunk_core: true,
                    exact: true,
                    capture: true,
                };
                for layer in &self.layers {
                    layer.prefill(&self.gpu, s, 2, p as u32, hidden, eps, mode)?;
                }
                ctx.copy_f32(&s.x, 0, &mtp.hid, 0, 2 * hh)?;
                ctx.rmsnorm(&s.x, &self.final_norm, &s.h, 2, hh, eps)?;
                self.head().gemm(&self.gpu, &s.h, &mtp.logits, 2, true)?;
                let vocab = self.graph.vocab as u32;
                for r in 0..2u32 {
                    ctx.copy_f32(&mtp.logits, r * vocab, &self.logits_buf, 0, vocab)?;
                    ctx.argmax_f32(
                        &self.logits_buf,
                        vocab,
                        &self.argmax_partial,
                        &mtp.ids,
                        SLOT_VERIFY + r,
                    )?;
                }
                Ok(())
            };
            if self.grouped.get() {
                self.gpu.grouped(encode)?;
            } else {
                encode()?;
            }
            let mut b = [0u8; 12];
            mtp.ids.download(0, &mut b)?;
            let id = |i: usize| {
                let v = u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
                (v != ARGMAX_NONE).then_some(v)
            };
            let draft = forced.or(id(SLOT_DRAFT as usize));
            let (first, second) = (id(SLOT_VERIFY as usize), id(SLOT_VERIFY as usize + 1));
            let accepted = first.is_some() && first == draft;
            let mut next = self.mtp_pending.borrow_mut();
            next.clear();
            if accepted {
                self.positions.set(p + 2);
                if let (Some(a0), Some(a1)) = (first, second) {
                    next.extend([a0, a1]);
                }
            } else {
                // The queue orders the rollback before any later work.
                self.gpu.grouped(|| {
                    for layer in &self.layers {
                        layer.roll_back(&self.gpu)?;
                    }
                    Ok(())
                })?;
                self.positions.set(p + 1);
                next.extend(first);
            }
            Ok(SpecStep {
                first,
                second: if accepted { second } else { None },
                accepted,
            })
        })();
        if result.is_err() {
            self.session_valid.set(false);
            self.mtp_pending.borrow_mut().clear();
        }
        result
    }

    fn ensure_valid(&self) -> Result<()> {
        if !self.session_valid.get() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "engine session was interrupted and its partial state was discarded",
            )
            .with_action(
                "Reset the session before continuing; interrupted state is never reused.",
            ));
        }
        Ok(())
    }

    fn finish(&self, result: Result<()>) -> Result<()> {
        if result.is_err() {
            self.session_valid.set(false);
        }
        result
    }

    fn check_tokens(&self, tokens: &[u32]) -> Result<usize> {
        let vocab = self.graph.vocab;
        if let Some(bad) = tokens.iter().find(|t| **t as usize >= vocab) {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!("token id {bad} is outside the {vocab}-entry vocabulary"),
            ));
        }
        let pos = self.positions.get();
        if pos + tokens.len() > self.capacity {
            return Err(TuckmillError::new(
                ErrorCode::InvalidPlan,
                format!(
                    "tokens {pos}..{} exceed the admitted {}-position context",
                    pos + tokens.len(),
                    self.capacity
                ),
            ));
        }
        Ok(pos)
    }

    fn download_logits(&self, logits: &mut Vec<f32>) -> Result<()> {
        let vocab = self.graph.vocab;
        let mut bytes = self.logits_staging.borrow_mut();
        bytes.resize(vocab * 4, 0);
        self.logits_buf.download(0, &mut bytes)?;
        logits.clear();
        logits.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
        Ok(())
    }

    /// One decode step; full-vocabulary logits for the caller.
    pub fn forward_token(&self, token: u32, logits: &mut Vec<f32>) -> Result<()> {
        self.step(
            token,
            Out::Logits(logits),
            self.grouped.get(),
            |_, _| Ok(()),
        )
    }

    /// One decode step whose next token is chosen on the device with the
    /// CPU greedy rule (`sampling::argmax`); 4 bytes return instead of the
    /// logits. `None` means every logit was NaN.
    pub fn forward_token_greedy(&self, token: u32) -> Result<Option<u32>> {
        let mut next = None;
        self.step(token, Out::Greedy(&mut next), self.grouped.get(), |_, _| {
            Ok(())
        })?;
        Ok(next)
    }

    /// A decode step with a per-stage callback (verification). Callbacks
    /// read activations mid-step, so a traced step is never grouped.
    pub fn forward_token_traced(
        &self,
        token: u32,
        logits: &mut Vec<f32>,
        on_stage: impl FnMut(Stage, &MetalBuffer) -> Result<()>,
    ) -> Result<()> {
        self.step(token, Out::Logits(logits), false, on_stage)
    }

    /// Encode the greedy choice over `logits_buf` into token `slot`
    /// (inside the open group).
    fn encode_argmax(&self, slot: u32) -> Result<()> {
        self.gpu.context().argmax_f32(
            &self.logits_buf,
            self.graph.vocab as u32,
            &self.argmax_partial,
            &self.token_ids,
            slot,
        )
    }

    fn read_token(&self, slot: u32, completed_target: Option<u64>) -> Result<Option<u32>> {
        let mut b = [0u8; 4];
        match completed_target {
            Some(target) => self
                .token_ids
                .download_after(slot as usize * 4, &mut b, target)?,
            None => self.token_ids.download(slot as usize * 4, &mut b)?,
        }
        let id = u32::from_le_bytes(b);
        Ok((id != ARGMAX_NONE).then_some(id))
    }

    /// Deliver the output of the step whose last position is `pos`, after
    /// its command buffer was committed.
    fn deliver(&self, out: Out<'_>, pos: usize) -> Result<()> {
        match out {
            Out::None => Ok(()),
            Out::Logits(logits) => self.download_logits(logits),
            Out::Greedy(next) => {
                *next = self.read_token(token_slot(pos), None)?;
                Ok(())
            }
        }
    }

    /// Commit one greedy decode step without waiting for it, so the host
    /// can encode the next step while this one runs. With
    /// [`GreedyInput::Previous`] the step embeds the token the previous
    /// step chose on the device; the host learns it later through
    /// [`Self::collect_greedy`]. Arithmetic is exactly that of
    /// [`Self::forward_token_greedy`].
    pub fn submit_greedy(&self, input: GreedyInput) -> Result<PendingToken> {
        self.ensure_valid()?;
        self.mtp_pending.borrow_mut().clear();
        let result = (|| {
            let pos = match input {
                GreedyInput::Token(t) => self.check_tokens(&[t])?,
                GreedyInput::Previous(prev) => {
                    let pos = self.check_tokens(&[])?;
                    if pos == 0 || prev.slot != token_slot(pos - 1) {
                        return Err(TuckmillError::new(
                            ErrorCode::Internal,
                            "pipelined step does not follow the step it reads",
                        ));
                    }
                    if pos + 1 > self.capacity {
                        return Err(TuckmillError::new(
                            ErrorCode::InvalidPlan,
                            format!(
                                "token {pos} exceeds the admitted {}-position context",
                                self.capacity
                            ),
                        ));
                    }
                    pos
                }
            };
            let (hidden, eps) = (self.graph.hidden, self.graph.eps);
            let slot = token_slot(pos);
            let encode = || -> Result<()> {
                match input {
                    GreedyInput::Token(t) => self.embedding.row_into(&self.gpu, t, &self.x, 0)?,
                    GreedyInput::Previous(prev) => self.embedding.row_indirect(
                        &self.gpu,
                        &self.token_ids,
                        prev.slot,
                        &self.x,
                    )?,
                }
                for layer in &self.layers {
                    layer.decode(&self.gpu, &self.x, pos as u32, hidden, eps)?;
                }
                self.gpu.context().rmsnorm(
                    &self.x,
                    &self.final_norm,
                    &self.h,
                    1,
                    hidden as u32,
                    eps,
                )?;
                self.head().gemv(&self.gpu, &self.h, &self.logits_buf)?;
                self.encode_argmax(slot)
            };
            if self.grouped.get() {
                self.gpu.grouped(encode)?;
            } else {
                encode()?;
            }
            self.positions.set(pos + 1);
            Ok(PendingToken {
                commit: self.gpu.context().committed_dispatches(),
                slot,
            })
        })();
        if result.is_err() {
            self.session_valid.set(false);
        }
        result
    }

    /// Wait for a submitted step only (not for later ones) and read its
    /// greedy token; `None` means every logit was NaN.
    pub fn collect_greedy(&self, pending: PendingToken) -> Result<Option<u32>> {
        self.read_token(pending.slot, Some(pending.commit))
    }

    /// Wait for all submitted work (before a reset or the next session).
    pub fn drain(&self) -> Result<()> {
        self.gpu.context().fence()
    }

    fn step(
        &self,
        token: u32,
        out: Out<'_>,
        grouped: bool,
        mut on_stage: impl FnMut(Stage, &MetalBuffer) -> Result<()>,
    ) -> Result<()> {
        self.ensure_valid()?;
        self.mtp_pending.borrow_mut().clear();
        let result = (|| {
            let pos = self.check_tokens(&[token])?;
            let (hidden, eps) = (self.graph.hidden, self.graph.eps);
            let greedy = matches!(out, Out::Greedy(_));
            let mut encode = || -> Result<()> {
                self.embedding.row_into(&self.gpu, token, &self.x, 0)?;
                on_stage(Stage::Embedding, &self.x)?;
                for (l, layer) in self.layers.iter().enumerate() {
                    layer.decode(&self.gpu, &self.x, pos as u32, hidden, eps)?;
                    on_stage(Stage::Layer(l), &self.x)?;
                }
                self.gpu.context().rmsnorm(
                    &self.x,
                    &self.final_norm,
                    &self.h,
                    1,
                    hidden as u32,
                    eps,
                )?;
                self.head().gemv(&self.gpu, &self.h, &self.logits_buf)?;
                if greedy {
                    self.encode_argmax(token_slot(pos))?;
                }
                Ok(())
            };
            if grouped {
                self.gpu.grouped(encode)?;
            } else {
                encode()?;
            }
            self.deliver(out, pos)?;
            self.positions.set(pos + 1);
            Ok(())
        })();
        self.finish(result)
    }

    /// Batched prefill of up to [`PREFILL_BATCH`] consecutive prompt tokens
    /// as one command buffer. Logits (last token) only when `want_logits`.
    pub fn prefill_chunk(
        &self,
        tokens: &[u32],
        want_logits: bool,
        logits: &mut Vec<f32>,
    ) -> Result<()> {
        let out = if want_logits {
            Out::Logits(logits)
        } else {
            Out::None
        };
        self.prefill(tokens, out)
    }

    /// The final prefill chunk with the first generated token chosen on
    /// the device (the CPU greedy rule); `None` means every logit was NaN.
    pub fn prefill_chunk_greedy(&self, tokens: &[u32]) -> Result<Option<u32>> {
        let mut next = None;
        self.prefill(tokens, Out::Greedy(&mut next))?;
        Ok(next)
    }

    fn prefill(&self, tokens: &[u32], out: Out<'_>) -> Result<()> {
        self.ensure_valid()?;
        let result = (|| {
            let n = tokens.len();
            if n == 0 || n > PREFILL_BATCH {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!("prefill chunk of {n} tokens is outside 1..={PREFILL_BATCH}"),
                ));
            }
            let pos = self.check_tokens(tokens)?;
            let (hidden, eps) = (self.graph.hidden, self.graph.eps);
            let want_logits = !matches!(out, Out::None);
            let greedy = matches!(out, Out::Greedy(_));
            let mode = ChunkMode::prefill(self.chunk_core.get());
            // Snapshot when this chunk ends exactly at the planned boundary
            // and the session holds the planned tokens up to it.
            let snapshot_here = {
                let mut p = self.prefix.borrow_mut();
                let continues = matches!(
                    &p.planned,
                    Some(t) if p.matched == pos && pos + n <= t.len() && t[pos..pos + n] == *tokens
                );
                if continues {
                    p.matched = pos + n;
                } else {
                    p.planned = None;
                }
                matches!(&p.planned, Some(t) if t.len() == p.matched)
            };
            self.bound_prefill_in_flight();
            let s = &self.prefill;
            let encode = || -> Result<()> {
                for (i, token) in tokens.iter().enumerate() {
                    self.embedding
                        .row_into(&self.gpu, *token, &s.x, (i * hidden) as u32)?;
                }
                for layer in &self.layers {
                    layer.prefill(&self.gpu, s, n, pos as u32, hidden, eps, mode)?;
                }
                let ctx = self.gpu.context();
                let hh = hidden as u32;
                // MTP rows of this chunk (they reuse the prefill scratch, so
                // their inputs are gathered first).
                let mtp_rows = match &self.mtp {
                    Some(mtp) => {
                        // Row j pairs h(pos + j - 1) with token j; the first
                        // prompt token has no hidden state before it.
                        let first = usize::from(pos == 0);
                        for (r, t) in tokens[first..].iter().enumerate() {
                            self.embedding
                                .row_into(&self.gpu, *t, &mtp.e, (r * hidden) as u32)?;
                        }
                        if first == 0 {
                            ctx.copy_f32(&mtp.hid, 0, &mtp.hin, 0, hh)?;
                        }
                        if n > 1 {
                            ctx.copy_f32(
                                &s.x,
                                0,
                                &mtp.hin,
                                (1 - first) as u32 * hh,
                                (n - 1) as u32 * hh,
                            )?;
                        }
                        ctx.copy_f32(&s.x, (n - 1) as u32 * hh, &mtp.hid, 0, hh)?;
                        Some((mtp, n - first, pos + first - 1))
                    }
                    None => None,
                };
                if snapshot_here {
                    for layer in &self.layers {
                        layer.copy_snapshot(&self.gpu, true)?;
                    }
                    self.mtp_snapshot(true)?;
                }
                if want_logits {
                    ctx.copy_f32(&s.x, ((n - 1) * hidden) as u32, &self.x, 0, hh)?;
                    ctx.rmsnorm(&self.x, &self.final_norm, &self.h, 1, hh, eps)?;
                    self.head().gemv(&self.gpu, &self.h, &self.logits_buf)?;
                    if greedy {
                        self.encode_argmax(token_slot(pos + n - 1))?;
                    }
                }
                if let Some((mtp, m, mpos)) = mtp_rows {
                    if m > 0 {
                        mtp.rows(
                            &self.gpu,
                            s,
                            m,
                            mpos as u32,
                            &self.final_norm,
                            self.mtp_convention.get(),
                            hidden,
                            eps,
                        )?;
                    }
                }
                Ok(())
            };
            if self.grouped.get() {
                self.gpu.grouped(encode)?;
            } else {
                encode()?;
            }
            // Speculation continues only after a greedy final chunk.
            let mut spec_from = None;
            if want_logits {
                // The download waits for all committed work.
                let greedy_out = match out {
                    Out::Greedy(next) => {
                        self.deliver(Out::Greedy(&mut *next), pos + n - 1)?;
                        spec_from = *next;
                        Ok(())
                    }
                    other => self.deliver(other, pos + n - 1),
                };
                greedy_out?;
                self.prefill_in_flight.borrow_mut().clear();
            } else {
                // Without logits nothing waits here: the queue orders the
                // next chunk after this one, bounded by the in-flight cap.
                self.prefill_in_flight
                    .borrow_mut()
                    .push_back(self.gpu.context().committed_dispatches());
            }
            self.positions.set(pos + n);
            {
                let mut pending = self.mtp_pending.borrow_mut();
                pending.clear();
                if self.mtp.is_some() {
                    pending.extend(spec_from);
                }
            }
            if snapshot_here {
                let mut p = self.prefix.borrow_mut();
                p.tokens = p.planned.take().unwrap_or_default();
                p.valid = true;
            }
            Ok(())
        })();
        self.finish(result)
    }
}

#[cfg(test)]
mod oracle_tests {
    //! The Qwen3.5 GPU qualification suite, retargeted from the retired
    //! per-family model to the unified engine executing the qwen35 graph.
    use super::*;
    use crate::qwen35_forward::TensorSource;
    use crate::qwen35_gpu::bytes_f32;
    use tuckmill_metal::probe;

    fn qwen35_graph(src: &impl TensorSource) -> ModelGraph {
        crate::engine::adapters::qwen35(src.config(), src.layout())
    }
    use crate::qwen35_forward::Qwen35Model;
    use crate::qwen35_kv::KvPrecision;
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    /// Registered device-vs-oracle tolerances for the complete fixture
    /// model (observed worst: hidden 5.4e-7, logits 1.4e-6 over 3 tokens).
    const HIDDEN_TOL: f32 = 1e-5;
    const LOGITS_TOL: f32 = 1e-4;

    fn gpu_or_skip() -> Option<()> {
        let probe = probe();
        if !probe.available {
            assert!(
                std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
                "TUCKMILL_REQUIRE_METAL is set but no Metal device is available"
            );
            eprintln!("skipping: no Metal device");
            return None;
        }
        Some(())
    }

    fn assert_close(got: f32, want: f32, abs: f32, rel: f32, what: &str) -> f32 {
        let tol = abs + rel * want.abs();
        assert!(
            (got - want).abs() <= tol,
            "{what}: {got} vs {want} (tol {tol})"
        );
        (got - want).abs()
    }

    fn device_f32(buf: &MetalBuffer, n: usize) -> Vec<f32> {
        let mut b = vec![0u8; n * 4];
        buf.download(0, &mut b).unwrap();
        bytes_f32(&b)
    }

    fn top1(logits: &[f32]) -> usize {
        let mut best = 0usize;
        for (i, v) in logits.iter().enumerate() {
            if *v > logits[best] {
                best = i;
            }
        }
        best
    }

    /// Complete fixture-model verification: every served layer executes on
    /// the device through the single assembled model (weights and
    /// activations stay on the GPU between layers) and each hidden state
    /// matches the CPU oracle's activation trace; the tied logits agree.
    #[test]
    fn fixture_device_layers_match_the_cpu_oracle_layer_by_layer() {
        if gpu_or_skip().is_none() {
            return;
        }
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen35());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q35.gguf");
        std::fs::write(&path, &bytes).unwrap();
        let capacity = 8usize;
        let mut cpu = Qwen35Model::open(&path, capacity, KvPrecision::F16).unwrap();
        let hidden = cpu.config().hidden as usize;
        let plan = alloc_plan(&qwen35_graph(&cpu), &cpu, capacity).unwrap();
        let model = EngineModel::load(
            Qwen35Gpu::open().unwrap(),
            qwen35_graph(&cpu),
            &cpu,
            capacity,
            plan.total(),
        )
        .unwrap();

        let tokens = [3u32, 11, 29];
        let mut worst_hidden = 0.0f32;
        let mut worst_logits = 0.0f32;
        for token in tokens {
            let mut cpu_logits = Vec::new();
            let mut trace = Vec::new();
            cpu.forward_token_traced(token, &mut cpu_logits, &mut trace)
                .unwrap();

            let mut gpu_logits = Vec::new();
            model
                .forward_token_traced(token, &mut gpu_logits, |stage, act| {
                    let idx = match stage {
                        Stage::Embedding => 0,
                        Stage::Layer(l) => l + 1,
                    };
                    let gx = device_f32(act, hidden);
                    for (g, c) in gx.iter().zip(trace[idx].iter()) {
                        worst_hidden = worst_hidden.max(assert_close(
                            *g,
                            *c,
                            HIDDEN_TOL,
                            HIDDEN_TOL,
                            &format!("{stage:?}"),
                        ));
                    }
                    Ok(())
                })
                .unwrap();

            for (g, c) in gpu_logits.iter().zip(cpu_logits.iter()) {
                worst_logits = worst_logits.max((g - c).abs());
            }
            assert_eq!(
                top1(&gpu_logits),
                top1(&cpu_logits),
                "token {token}: top-1 matches the oracle"
            );
        }
        eprintln!(
            "fixture device model: worst hidden|diff| {worst_hidden:.3e}, worst logits|diff| {worst_logits:.3e} over {tokens:?}"
        );
        assert!(worst_hidden <= HIDDEN_TOL, "registered hidden tolerance");
        assert!(worst_logits <= LOGITS_TOL, "registered logits tolerance");
    }

    /// Batched prefill is bit-identical to token-by-token decode: same
    /// logits for the last prompt token and the same next decode step
    /// (so recurrent state, convolution history and KV agree), for single,
    /// partial, full and multi-chunk prompts.
    #[test]
    fn batched_prefill_is_bit_identical_to_sequential_decode() {
        if gpu_or_skip().is_none() {
            return;
        }
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen35());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q35.gguf");
        std::fs::write(&path, &bytes).unwrap();
        let cpu = Qwen35Model::open(&path, 64, KvPrecision::F16).unwrap();
        // Room for the longest prompt below (two chunks) plus its next step.
        let capacity = PREFILL_BATCH + 8;
        let plan = alloc_plan(&qwen35_graph(&cpu), &cpu, capacity).unwrap();
        let model = EngineModel::load(
            Qwen35Gpu::open().unwrap(),
            qwen35_graph(&cpu),
            &cpu,
            capacity,
            plan.total(),
        )
        .unwrap();
        assert_eq!(
            model.allocated_bytes(),
            plan.total(),
            "ledger == plan incl. prefill scratch"
        );

        for len in [1usize, 3, PREFILL_BATCH, PREFILL_BATCH + 4] {
            let prompt: Vec<u32> = (0..len as u32).map(|i| (i * 37 + 5) % 500).collect();
            let next = 77u32;

            model.reset().unwrap();
            let mut seq = Vec::new();
            for t in &prompt {
                model.forward_token(*t, &mut seq).unwrap();
            }
            let mut seq_next = Vec::new();
            model.forward_token(next, &mut seq_next).unwrap();

            model.reset().unwrap();
            let mut batched = Vec::new();
            let chunks: Vec<&[u32]> = prompt.chunks(PREFILL_BATCH).collect();
            for (c, chunk) in chunks.iter().enumerate() {
                model
                    .prefill_chunk(chunk, c + 1 == chunks.len(), &mut batched)
                    .unwrap();
            }
            assert_eq!(model.positions(), len);
            let mut batched_next = Vec::new();
            model.forward_token(next, &mut batched_next).unwrap();

            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&batched),
                bits(&seq),
                "prompt {len}: last-token logits"
            );
            assert_eq!(
                bits(&batched_next),
                bits(&seq_next),
                "prompt {len}: next decode step"
            );
        }

        // Out-of-range chunks are refused and leave the session usable
        // after reset.
        model.reset().unwrap();
        let mut l = Vec::new();
        assert!(model.prefill_chunk(&[], true, &mut l).is_err());
        let too_many = vec![1u32; PREFILL_BATCH + 1];
        assert!(model.prefill_chunk(&too_many, true, &mut l).is_err());
    }

    /// Machine-local (TUCKMILL_QWEN35_GGUF): on the pinned Q8_0 artifact,
    /// batched prefill gives bit-identical last-token logits and next
    /// decode step to sequential decode; prints 512-token prefill time for
    /// both (measurement, not asserted).
    #[test]
    fn real_artifact_batched_prefill_matches_sequential_when_available() {
        let Some(path) = std::env::var("TUCKMILL_QWEN35_GGUF")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists())
        else {
            eprintln!("skipping: TUCKMILL_QWEN35_GGUF not set");
            return;
        };
        if gpu_or_skip().is_none() {
            return;
        }
        let source = crate::qwen35_forward::ArtifactTensorSource::open(&path).unwrap();
        let capacity = 640usize;
        let plan = alloc_plan(&qwen35_graph(&source), &source, capacity).unwrap();
        let model = EngineModel::load(
            Qwen35Gpu::open().unwrap(),
            qwen35_graph(&source),
            &source,
            capacity,
            plan.total(),
        )
        .unwrap();
        assert_eq!(model.allocated_bytes(), plan.total());
        let prompt: Vec<u32> = (0..512u32).map(|i| 1000 + (i * 7919) % 90_000).collect();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

        model.reset().unwrap();
        let t0 = std::time::Instant::now();
        let mut seq = Vec::new();
        for t in &prompt {
            model.forward_token(*t, &mut seq).unwrap();
        }
        let seq_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut seq_next = Vec::new();
        model.forward_token(9707, &mut seq_next).unwrap();

        model.reset().unwrap();
        let t1 = std::time::Instant::now();
        let mut batched = Vec::new();
        let chunks: Vec<&[u32]> = prompt.chunks(PREFILL_BATCH).collect();
        for (c, chunk) in chunks.iter().enumerate() {
            model
                .prefill_chunk(chunk, c + 1 == chunks.len(), &mut batched)
                .unwrap();
        }
        let batched_ms = t1.elapsed().as_secs_f64() * 1e3;
        let mut batched_next = Vec::new();
        model.forward_token(9707, &mut batched_next).unwrap();

        eprintln!(
            "512-token prefill: sequential {seq_ms:.0} ms ({:.0} tok/s), batched x{PREFILL_BATCH} {batched_ms:.0} ms ({:.0} tok/s); scratch {:.1} MiB at capacity {capacity}",
            512e3 / seq_ms,
            512e3 / batched_ms,
            plan.scratch as f64 / 1048576.0
        );
        let drift = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        let top1 = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |m, (i, x)| {
                    if *x > m.1 {
                        (i, *x)
                    } else {
                        m
                    }
                })
                .0
        };
        let scale = seq.iter().fold(0f32, |m, v| m.max(v.abs()));
        eprintln!(
            "batched vs sequential: last-token logits max|diff| {:.3e}, next step {:.3e} (logit scale {scale:.2}); top-1 {} vs {}",
            drift(&batched, &seq),
            drift(&batched_next, &seq_next),
            top1(&batched),
            top1(&seq)
        );
        let _ = bits;
        // Registered: the GPU-vs-oracle logits bound (2e-3, Q35-2C);
        // observed 3.2e-4 / 3.6e-4 at logit scale 11.1 (ADR-008 Add. 16).
        assert!(
            drift(&batched, &seq) <= 2e-3,
            "last-token logits within 2e-3"
        );
        assert!(
            drift(&batched_next, &seq_next) <= 2e-3,
            "next step within 2e-3"
        );
        assert_eq!(top1(&batched), top1(&seq), "last-token top-1");
        assert_eq!(top1(&batched_next), top1(&seq_next), "next-step top-1");

        if std::env::var("TUCKMILL_PROFILE").ok().as_deref() == Some("1") {
            let ctx = model.context();
            model.reset().unwrap();
            model.set_grouped(false);
            ctx.set_profiling(true);
            let mut l = Vec::new();
            for (c, chunk) in chunks.iter().enumerate() {
                model
                    .prefill_chunk(chunk, c + 1 == chunks.len(), &mut l)
                    .unwrap();
            }
            ctx.fence().unwrap();
            let mut table = ctx.profile_snapshot();
            ctx.set_profiling(false);
            model.set_grouped(true);
            table.sort_by_key(|row| std::cmp::Reverse(row.1));
            let total: u64 = table.iter().map(|r| r.1).sum();
            eprintln!("ungrouped prefill GPU time per operator (512 tokens):");
            for (label, nanos, count) in table.iter().take(14) {
                eprintln!(
                    "  {label:<28} {:>8.1} ms  {:>5.1}%  {:>6} buffers",
                    *nanos as f64 / 1e6,
                    100.0 * *nanos as f64 / total.max(1) as f64,
                    count
                );
            }
        }
    }

    /// Phase 1B measurement (machine-local, opt-in): per-token wall time,
    /// GPU busy time, command buffers, and GPU time per operator for the
    /// pinned model's decode. Prints a table; asserts only that the
    /// accounting is consistent. Run with
    /// `TUCKMILL_QWEN35_GGUF=... TUCKMILL_PROFILE=1 cargo test --release
    /// -p tuckmill-tuckcore profile_pinned_decode -- --nocapture`.
    #[test]
    fn profile_pinned_decode_when_requested() {
        let Some(path) = std::env::var("TUCKMILL_QWEN35_GGUF")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists())
        else {
            eprintln!("skipping: TUCKMILL_QWEN35_GGUF not set");
            return;
        };
        if std::env::var("TUCKMILL_PROFILE").ok().as_deref() != Some("1") {
            eprintln!("skipping: TUCKMILL_PROFILE=1 not set (measurement only)");
            return;
        }
        if gpu_or_skip().is_none() {
            return;
        }
        let source = crate::qwen35_forward::ArtifactTensorSource::open(&path).unwrap();
        let capacity = 128usize;
        let plan = alloc_plan(&qwen35_graph(&source), &source, capacity).unwrap();
        let model = EngineModel::load(
            Qwen35Gpu::open().unwrap(),
            qwen35_graph(&source),
            &source,
            capacity,
            plan.total(),
        )
        .unwrap();
        let ctx = model.context();
        // Deterministic token ids; content does not change the cost of a
        // decode step (same shapes for every token).
        let tokens: Vec<u32> = (0..56u32).map(|i| 1000 + i * 37).collect();
        let mut logits = Vec::new();
        for t in &tokens[..8] {
            model.forward_token(*t, &mut logits).unwrap();
        }

        let measured = &tokens[8..40];
        let n = measured.len() as f64;
        let (c0, g0) = (ctx.committed_dispatches(), ctx.gpu_nanos_total());
        let t0 = std::time::Instant::now();
        for t in measured {
            model.forward_token(*t, &mut logits).unwrap();
        }
        ctx.fence().unwrap();
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3 / n;
        let gpu_ms = (ctx.gpu_nanos_total() - g0) as f64 / 1e6 / n;
        let cbs = (ctx.committed_dispatches() - c0) as f64 / n;
        eprintln!(
            "decode per token: wall {wall_ms:.2} ms ({:.2} tok/s), GPU busy {gpu_ms:.2} ms ({:.0}%), host/dispatch {:.2} ms, {cbs:.0} command buffers",
            1e3 / wall_ms,
            100.0 * gpu_ms / wall_ms,
            wall_ms - gpu_ms
        );
        assert!(gpu_ms <= wall_ms * 1.05, "GPU time cannot exceed wall time");

        model.set_grouped(false);
        let (u0, t1) = (ctx.gpu_nanos_total(), std::time::Instant::now());
        for t in &tokens[40..48] {
            model.forward_token(*t, &mut logits).unwrap();
        }
        ctx.fence().unwrap();
        eprintln!(
            "ungrouped (one command buffer per operator): wall {:.2} ms, GPU busy {:.2} ms per token",
            t1.elapsed().as_secs_f64() * 1e3 / 8.0,
            (ctx.gpu_nanos_total() - u0) as f64 / 1e6 / 8.0
        );
        ctx.set_profiling(true);
        let profiled = &tokens[48..56];
        for t in profiled {
            model.forward_token(*t, &mut logits).unwrap();
        }
        ctx.fence().unwrap();
        let mut table = ctx.profile_snapshot();
        ctx.set_profiling(false);
        table.sort_by_key(|row| std::cmp::Reverse(row.1));
        let total: u64 = table.iter().map(|r| r.1).sum();
        let per = profiled.len() as f64;
        eprintln!(
            "GPU time per operator (per token, {} tokens):",
            profiled.len()
        );
        for (label, nanos, count) in &table {
            eprintln!(
                "  {label:<28} {:>8.3} ms  {:>5.1}%  {:>6.1} buffers",
                *nanos as f64 / 1e6 / per,
                100.0 * *nanos as f64 / total.max(1) as f64,
                *count as f64 / per
            );
        }
    }

    /// Q35-2C/2C.1 machine-local gate: the complete pinned 0.8B executes
    /// on the GPU from a lightweight artifact tensor source (no CPU
    /// session), the exact preflight plan equals the device ledger, and
    /// logits plus selected-layer hidden states match the CPU oracle.
    /// Run with --release for speed.
    #[test]
    fn real_artifact_gpu_model_matches_cpu_when_available() {
        let Some(path) = std::env::var("TUCKMILL_QWEN35_GGUF")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists())
        else {
            eprintln!("skipping: TUCKMILL_QWEN35_GGUF not set (pinned Qwen3.5-0.8B Q8_0)");
            return;
        };
        if gpu_or_skip().is_none() {
            return;
        }
        let capacity = 16usize;
        let mut cpu = Qwen35Model::open(&path, capacity, KvPrecision::F16).unwrap();
        let source = crate::qwen35_forward::ArtifactTensorSource::open(&path).unwrap();

        // Preflight: nothing allocates before the admission check passes.
        let plan = alloc_plan(&qwen35_graph(&source), &source, capacity).unwrap();
        let model = EngineModel::load(
            Qwen35Gpu::open().unwrap(),
            qwen35_graph(&source),
            &source,
            capacity,
            plan.total(),
        )
        .unwrap();
        assert_eq!(
            model.allocated_bytes(),
            plan.total(),
            "exact preflight plan equals the device ledger"
        );

        finish_real_gpu_checks(&mut cpu, &model, plan, &tokens_for_real_gpu())
    }

    fn tokens_for_real_gpu() -> Vec<u32> {
        vec![9707u32, 11, 1879, 1000]
    }

    fn finish_real_gpu_checks(
        cpu: &mut Qwen35Model,
        model: &EngineModel,
        plan: GpuAllocPlan,
        tokens: &[u32],
    ) {
        let hidden = cpu.config().hidden as usize;
        let selected: [usize; 4] = [0, 3, 11, 23];
        let mut worst = 0.0f32;
        let mut worst_hidden = 0.0f32;
        for (step, token) in tokens.iter().enumerate() {
            let mut cpu_logits = Vec::new();
            let mut trace = Vec::new();
            cpu.forward_token_traced(*token, &mut cpu_logits, &mut trace)
                .unwrap();
            let mut gpu_logits = Vec::new();
            model
                .forward_token_traced(*token, &mut gpu_logits, |stage, act| {
                    if let Stage::Layer(l) = stage {
                        if selected.contains(&l) {
                            let gx = device_f32(act, hidden);
                            for (g, c) in gx.iter().zip(trace[l + 1].iter()) {
                                worst_hidden = worst_hidden.max(assert_close(
                                    *g,
                                    *c,
                                    REAL_HIDDEN_TOL,
                                    REAL_HIDDEN_TOL,
                                    &format!("token {token} layer {l}"),
                                ));
                            }
                        }
                    }
                    Ok(())
                })
                .unwrap();
            for (g, c) in gpu_logits.iter().zip(cpu_logits.iter()) {
                worst = worst.max((g - c).abs());
            }
            assert_eq!(
                top1(&gpu_logits),
                top1(&cpu_logits),
                "step {step} token {token}: GPU top-1 matches the CPU oracle"
            );
        }
        eprintln!(
            "real artifact GPU model: ledger {} B == exact plan (weights {} + state {} + scratch {}); worst hidden|diff| {worst_hidden:.6} at layers {selected:?}, worst logits|diff| {worst:.6} over {tokens:?}",
            model.allocated_bytes(),
            plan.weights,
            plan.state,
            plan.scratch
        );
        assert!(
            worst_hidden <= REAL_HIDDEN_TOL,
            "registered GPU-vs-oracle hidden tolerance (observed {worst_hidden:.6})"
        );
        assert!(worst <= 2e-3, "registered GPU-vs-oracle logits tolerance");
    }

    /// Registered hidden-state tolerance for the real artifact (observed
    /// worst 6.4e-5 at layers [0, 3, 11, 23] over 4 tokens; recorded in
    /// the matrix with the commit).
    const REAL_HIDDEN_TOL: f32 = 5e-4;

    /// An insufficient admission is refused BEFORE any GPU buffer of the
    /// model is created: the shared ledger does not move.
    #[test]
    fn preflight_refuses_insufficient_admission_before_allocating() {
        if gpu_or_skip().is_none() {
            return;
        }
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen35());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q35.gguf");
        std::fs::write(&path, &bytes).unwrap();
        let source = crate::qwen35_forward::ArtifactTensorSource::open(&path).unwrap();
        let gpu = Qwen35Gpu::open().unwrap();
        let plan = alloc_plan(&qwen35_graph(&source), &source, 4).unwrap();
        let before = gpu.context().allocated_bytes();
        let err = match EngineModel::load(
            gpu.clone(),
            qwen35_graph(&source),
            &source,
            4,
            plan.total() - 1,
        ) {
            Ok(_) => panic!("an under-admitted load must refuse"),
            Err(e) => e,
        };
        assert_eq!(err.code(), ErrorCode::InvalidPlan);
        assert!(
            err.message().contains("nothing was allocated"),
            "{}",
            err.message()
        );
        assert_eq!(
            gpu.context().allocated_bytes(),
            before,
            "refusal happens before any model buffer exists"
        );
    }

    /// Fail-closed session lifecycle: a step that fails part-way
    /// invalidates the session; reset restores bitwise-identical output.
    #[test]
    fn gpu_session_invalidates_and_reset_restores() {
        if gpu_or_skip().is_none() {
            return;
        }
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen35());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q35.gguf");
        std::fs::write(&path, &bytes).unwrap();
        let source = crate::qwen35_forward::ArtifactTensorSource::open(&path).unwrap();

        let baseline = {
            let plan = alloc_plan(&qwen35_graph(&source), &source, 4).unwrap();
            let model = EngineModel::load(
                Qwen35Gpu::open().unwrap(),
                qwen35_graph(&source),
                &source,
                2,
                plan.total(),
            )
            .unwrap();
            let mut logits = Vec::new();
            model.forward_token(3, &mut logits).unwrap();
            model.forward_token(11, &mut logits).unwrap();
            logits
        };
        let _ = &baseline;

        let plan = alloc_plan(&qwen35_graph(&source), &source, 4).unwrap();
        let model = EngineModel::load(
            Qwen35Gpu::open().unwrap(),
            qwen35_graph(&source),
            &source,
            2,
            plan.total(),
        )
        .unwrap();
        let mut logits = Vec::new();
        model.forward_token(3, &mut logits).unwrap();
        model.forward_token(11, &mut logits).unwrap();
        let err = model.forward_token(29, &mut logits).unwrap_err();
        assert_eq!(
            err.code(),
            ErrorCode::InvalidPlan,
            "capacity exhausted mid-step"
        );

        let err = model.forward_token(3, &mut logits).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidRequest);
        assert!(err.message().contains("interrupted"), "{}", err.message());

        model.reset().unwrap();
        let mut restored = Vec::new();
        model.forward_token(3, &mut restored).unwrap();
        model.forward_token(11, &mut restored).unwrap();
        assert_eq!(
            restored, baseline,
            "reset restores bitwise-identical output"
        );
    }

    /// The lightweight artifact source loads and runs without any CPU
    /// inference session, and produces the same logits as the oracle.
    #[test]
    fn artifact_source_loads_without_a_cpu_model() {
        if gpu_or_skip().is_none() {
            return;
        }
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen35());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q35.gguf");
        std::fs::write(&path, &bytes).unwrap();

        let source = crate::qwen35_forward::ArtifactTensorSource::open(&path).unwrap();
        let plan = alloc_plan(&qwen35_graph(&source), &source, 4).unwrap();
        let model = EngineModel::load(
            Qwen35Gpu::open().unwrap(),
            qwen35_graph(&source),
            &source,
            4,
            plan.total(),
        )
        .unwrap();

        let mut cpu = Qwen35Model::open(&path, 4, KvPrecision::F16).unwrap();
        let tokens = [5u32, 17];
        let mut worst = 0.0f32;
        for token in tokens {
            let mut cpu_logits = Vec::new();
            let mut trace = Vec::new();
            cpu.forward_token_traced(token, &mut cpu_logits, &mut trace)
                .unwrap();
            let mut gpu_logits = Vec::new();
            model.forward_token(token, &mut gpu_logits).unwrap();
            for (g, c) in gpu_logits.iter().zip(cpu_logits.iter()) {
                worst = worst.max((g - c).abs());
            }
            assert_eq!(top1(&gpu_logits), top1(&cpu_logits));
        }
        assert!(
            worst <= LOGITS_TOL,
            "registered logits tolerance ({worst:.3e})"
        );
    }
}

#[cfg(test)]
mod greedy_tests {
    use super::*;
    use crate::qwen35_forward::{ArtifactTensorSource, TensorSource};
    use crate::sampling::argmax;
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    fn gpu_or_skip() -> bool {
        if tuckmill_metal::probe().available {
            return true;
        }
        assert!(
            std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
            "TUCKMILL_REQUIRE_METAL is set but no Metal device is available"
        );
        eprintln!("skipping: no Metal device");
        false
    }

    /// Greedy decoding on the device picks exactly what the CPU greedy
    /// rule picks from the downloaded logits, at the prompt's last chunk
    /// and at every decode step (each run feeds back its own choice).
    /// Returns ms/token for (logits path, device greedy, pipelined).
    fn device_greedy_equals_cpu_greedy(
        src: &(impl TensorSource + WeightSource),
        steps: usize,
    ) -> (f64, f64, f64) {
        let graph = crate::engine::adapters::qwen35(src.config(), src.layout());
        let capacity = 64usize;
        let plan = alloc_plan(&graph, src, capacity).unwrap();
        let model = EngineModel::load(
            Qwen35Gpu::open().unwrap(),
            graph,
            src,
            capacity,
            plan.total(),
        )
        .unwrap();
        assert_eq!(
            model.allocated_bytes(),
            plan.total(),
            "ledger == plan incl. argmax buffers"
        );
        let prompt: Vec<u32> = (0..20u32).map(|i| (i * 37 + 5) % 400).collect();

        // CPU rule over downloaded logits.
        let mut cpu_tokens = Vec::new();
        let mut logits = Vec::new();
        for (c, chunk) in prompt.chunks(PREFILL_BATCH).enumerate() {
            let last = (c + 1) * PREFILL_BATCH >= prompt.len();
            model.prefill_chunk(chunk, last, &mut logits).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            let next = argmax(&logits).expect("finite logits");
            cpu_tokens.push(next);
            model.forward_token(next, &mut logits).unwrap();
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;

        // Device choice.
        model.reset().unwrap();
        let mut dev_tokens = Vec::new();
        let mut first = None;
        for (c, chunk) in prompt.chunks(PREFILL_BATCH).enumerate() {
            let last = (c + 1) * PREFILL_BATCH >= prompt.len();
            if last {
                first = model.prefill_chunk_greedy(chunk).unwrap();
            } else {
                model.prefill_chunk(chunk, false, &mut Vec::new()).unwrap();
            }
        }
        let mut next = first.expect("finite logits");
        let t1 = std::time::Instant::now();
        for _ in 0..steps {
            dev_tokens.push(next);
            next = model
                .forward_token_greedy(next)
                .unwrap()
                .expect("finite logits");
        }
        let dev_ms = t1.elapsed().as_secs_f64() * 1e3 / steps as f64;
        assert_eq!(
            dev_tokens, cpu_tokens,
            "device greedy == CPU greedy, token for token"
        );

        // Pipelined: step k+1 is queued (reading step k's choice on the
        // device) before step k is collected.
        model.reset().unwrap();
        let mut first = None;
        for (c, chunk) in prompt.chunks(PREFILL_BATCH).enumerate() {
            let last = (c + 1) * PREFILL_BATCH >= prompt.len();
            if last {
                first = model.prefill_chunk_greedy(chunk).unwrap();
            } else {
                model.prefill_chunk(chunk, false, &mut Vec::new()).unwrap();
            }
        }
        let mut pipe_tokens = vec![first.expect("finite logits")];
        let t2 = std::time::Instant::now();
        let mut current = model
            .submit_greedy(GreedyInput::Token(pipe_tokens[0]))
            .unwrap();
        while pipe_tokens.len() < steps {
            let following = if pipe_tokens.len() + 1 < steps {
                Some(model.submit_greedy(GreedyInput::Previous(current)).unwrap())
            } else {
                None
            };
            pipe_tokens.push(model.collect_greedy(current).unwrap().expect("finite"));
            match following {
                Some(f) => current = f,
                None => break,
            }
        }
        let pipe_ms = t2.elapsed().as_secs_f64() * 1e3 / (steps - 1) as f64;
        assert_eq!(
            pipe_tokens, cpu_tokens,
            "pipelined greedy == CPU greedy, token for token"
        );
        model.reset().unwrap();
        (cpu_ms, dev_ms, pipe_ms)
    }

    #[test]
    fn device_greedy_equals_cpu_greedy_on_the_fixture() {
        if !gpu_or_skip() {
            return;
        }
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen35());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q35.gguf");
        std::fs::write(&path, &bytes).unwrap();
        device_greedy_equals_cpu_greedy(&ArtifactTensorSource::open(&path).unwrap(), 24);
    }

    /// Machine-local (TUCKMILL_QWEN35_GGUF): the same equality on the
    /// pinned artifact, with per-token time for both paths.
    #[test]
    fn device_greedy_equals_cpu_greedy_on_the_pinned_artifact() {
        let Some(path) = std::env::var("TUCKMILL_QWEN35_GGUF")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists())
        else {
            eprintln!("skipping: TUCKMILL_QWEN35_GGUF not set");
            return;
        };
        if !gpu_or_skip() {
            return;
        }
        let (cpu_ms, dev_ms, pipe_ms) =
            device_greedy_equals_cpu_greedy(&ArtifactTensorSource::open(&path).unwrap(), 40);
        eprintln!(
            "greedy decode: logits + CPU argmax {cpu_ms:.2} ms/token ({:.1} tok/s), device argmax {dev_ms:.2} ({:.1} tok/s), pipelined {pipe_ms:.2} ({:.1} tok/s)",
            1e3 / cpu_ms,
            1e3 / dev_ms,
            1e3 / pipe_ms
        );
    }
}

#[cfg(test)]
mod chunk_core_tests {
    use super::*;
    use crate::qwen35_forward::{ArtifactTensorSource, TensorSource};
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    fn gpu_or_skip() -> bool {
        if tuckmill_metal::probe().available {
            return true;
        }
        assert!(
            std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
            "TUCKMILL_REQUIRE_METAL is set but no Metal device is available"
        );
        eprintln!("skipping: no Metal device");
        false
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Prefill `prompt` then one decode step; returns (last-token logits,
    /// next-step logits, prefill ms).
    fn run(model: &EngineModel, prompt: &[u32]) -> (Vec<f32>, Vec<f32>, f64) {
        model.reset().unwrap();
        let t0 = std::time::Instant::now();
        let mut logits = Vec::new();
        let chunks: Vec<&[u32]> = prompt.chunks(PREFILL_BATCH).collect();
        for (c, chunk) in chunks.iter().enumerate() {
            model
                .prefill_chunk(chunk, c + 1 == chunks.len(), &mut logits)
                .unwrap();
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut next = Vec::new();
        model.forward_token(prompt[0], &mut next).unwrap();
        (logits, next, ms)
    }

    /// The chunk-wide mixer cores are bit-identical to running the
    /// per-token decode cores over the chunk (last-token logits and the
    /// next decode step, so recurrent state, convolution history and KV
    /// agree), for partial, single, exact and multi-chunk prompts.
    fn chunk_core_matches_per_token(src: &(impl TensorSource + WeightSource), lens: &[usize]) {
        let graph = crate::engine::adapters::qwen35(src.config(), src.layout());
        let capacity = lens.iter().max().unwrap() + 8;
        let plan = alloc_plan(&graph, src, capacity).unwrap();
        let model = EngineModel::load(
            Qwen35Gpu::open().unwrap(),
            graph,
            src,
            capacity,
            plan.total(),
        )
        .unwrap();
        assert_eq!(model.allocated_bytes(), plan.total(), "ledger == plan");
        for &len in lens {
            let prompt: Vec<u32> = (0..len as u32).map(|i| (i * 7919 + 13) % 500).collect();
            model.set_chunk_core(false);
            let (a_logits, a_next, a_ms) = run(&model, &prompt);
            model.set_chunk_core(true);
            let (b_logits, b_next, b_ms) = run(&model, &prompt);
            assert_eq!(
                bits(&b_logits),
                bits(&a_logits),
                "len {len}: prefill logits"
            );
            assert_eq!(bits(&b_next), bits(&a_next), "len {len}: next decode step");
            // The register-resident recurrence vs the generic device-memory
            // kernel: same source arithmetic, but separately compiled under
            // fast math, so equal within the registered GPU logits bound
            // (2e-3), not bitwise.
            model.context().set_gdn_chunk_generic(true);
            let (c_logits, c_next, c_ms) = run(&model, &prompt);
            model.context().set_gdn_chunk_generic(false);
            let drift = |x: &[f32], y: &[f32]| {
                x.iter()
                    .zip(y)
                    .map(|(p, q)| (p - q).abs())
                    .fold(0f32, f32::max)
            };
            let (dl, dn) = (drift(&b_logits, &c_logits), drift(&b_next, &c_next));
            assert!(
                dl <= 2e-3 && dn <= 2e-3,
                "len {len}: register vs generic {dl:e} / {dn:e}"
            );
            if len >= 256 {
                eprintln!(
                    "{len}-token prefill: per-token cores {a_ms:.0} ms, chunk cores (generic recurrence) {c_ms:.0} ms, chunk cores (register recurrence) {b_ms:.0} ms; register vs generic max|diff| {dl:.2e} / {dn:.2e}"
                );
            }
        }
    }

    #[test]
    fn chunk_cores_are_bit_identical_to_per_token_cores_on_the_fixture() {
        if !gpu_or_skip() {
            return;
        }
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen35());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q35.gguf");
        std::fs::write(&path, &bytes).unwrap();
        chunk_core_matches_per_token(&ArtifactTensorSource::open(&path).unwrap(), &[1, 5, 16, 37]);
    }

    /// Machine-local (TUCKMILL_QWEN35_GGUF), with 512-token timing.
    #[test]
    fn chunk_cores_are_bit_identical_on_the_pinned_artifact() {
        let Some(path) = std::env::var("TUCKMILL_QWEN35_GGUF")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists())
        else {
            eprintln!("skipping: TUCKMILL_QWEN35_GGUF not set");
            return;
        };
        if !gpu_or_skip() {
            return;
        }
        chunk_core_matches_per_token(
            &ArtifactTensorSource::open(&path).unwrap(),
            &[1, 5, 16, 37, 512],
        );
    }
}

#[cfg(test)]
mod qwen3_tests {
    use super::*;
    use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

    fn gpu_or_skip() -> bool {
        if tuckmill_metal::probe().available {
            return true;
        }
        assert!(
            std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
            "TUCKMILL_REQUIRE_METAL is set but no Metal device is available"
        );
        eprintln!("skipping: no Metal device");
        false
    }

    /// Dense Qwen3 on the engine (ADR-011 step 2), for both declared KV
    /// precisions: ledger == plan with KV planes at the declared entry size,
    /// and batched prefill bit-identical to token-by-token decode (single,
    /// partial, full and multi-chunk prompts).
    #[test]
    fn qwen3_graph_prefill_is_bit_identical_to_decode_for_both_kv_precisions() {
        if !gpu_or_skip() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qwen3.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let registry = tuckmill_manifest::adapter::AdapterRegistry::builtin().unwrap();
        let (cfg, _) = crate::model::resolve(&insp, &registry).unwrap();
        let mut names = crate::model::TensorNames::for_layers(cfg.layers);
        names.resolve(&insp).unwrap();
        let graph = super::super::adapters::qwen3(&cfg, &names, &insp).unwrap();
        assert!(graph.layers.iter().all(|l| matches!(
            &l.mixer,
            MixerSpec::Attention(a) if !a.output_gate && a.rope_dims == a.head_dim
        )));
        let source = super::super::GgufWeights::with_inspection(&file, &insp).unwrap();
        let capacity = PREFILL_BATCH + 8;
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

        for kv in [KvCache::F32, KvCache::F16] {
            let plan = alloc_plan_with(
                &graph,
                &source,
                capacity,
                EngineOptions {
                    kv,
                    prefix_cache: false,
                    mtp: false,
                },
            )
            .unwrap();
            let expected_state = cfg.layers as usize
                * 2
                * cfg.kv_heads as usize
                * capacity
                * cfg.key_length as usize
                * kv.entry_bytes();
            assert_eq!(plan.state as usize, expected_state, "{kv:?}: KV planes");
            let model = EngineModel::load_with(
                Qwen35Gpu::open().unwrap(),
                graph.clone(),
                &source,
                capacity,
                plan.total(),
                EngineOptions {
                    kv,
                    prefix_cache: false,
                    mtp: false,
                },
            )
            .unwrap();
            assert_eq!(
                model.allocated_bytes(),
                plan.total(),
                "{kv:?}: ledger == plan"
            );

            for len in [1usize, 3, PREFILL_BATCH, PREFILL_BATCH + 4] {
                let prompt: Vec<u32> = (0..len as u32)
                    .map(|i| (i * 37 + 5) % cfg.vocab as u32)
                    .collect();
                model.reset().unwrap();
                let mut seq = Vec::new();
                for t in &prompt {
                    model.forward_token(*t, &mut seq).unwrap();
                }
                let mut seq_next = Vec::new();
                model.forward_token(7, &mut seq_next).unwrap();

                model.reset().unwrap();
                let mut batched = Vec::new();
                let chunks: Vec<&[u32]> = prompt.chunks(PREFILL_BATCH).collect();
                for (c, chunk) in chunks.iter().enumerate() {
                    model
                        .prefill_chunk(chunk, c + 1 == chunks.len(), &mut batched)
                        .unwrap();
                }
                let mut batched_next = Vec::new();
                model.forward_token(7, &mut batched_next).unwrap();
                assert_eq!(bits(&batched), bits(&seq), "{kv:?} prompt {len}: logits");
                assert_eq!(
                    bits(&batched_next),
                    bits(&seq_next),
                    "{kv:?} prompt {len}: next decode step"
                );
            }
        }
    }
}

#[cfg(test)]
mod qwen3_real_tests {
    use super::*;

    /// Measurement (opt-in): 512-token prefill of the pinned dense Qwen3
    /// (`TUCKMILL_REAL_GGUF`) on the engine with declared f16 KV, batched vs
    /// sequential drift, and with `TUCKMILL_PROFILE=1` the ungrouped GPU time
    /// per operator.
    #[test]
    fn real_qwen3_prefill_timing_and_profile_when_available() {
        let Some(path) = std::env::var("TUCKMILL_REAL_GGUF")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists())
        else {
            eprintln!("skipping: TUCKMILL_REAL_GGUF not set");
            return;
        };
        if !tuckmill_metal::probe().available {
            eprintln!("skipping: no Metal device");
            return;
        }
        let file = std::fs::File::open(&path).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let registry = tuckmill_manifest::adapter::AdapterRegistry::builtin().unwrap();
        let Ok((cfg, _)) = crate::model::resolve(&insp, &registry) else {
            eprintln!("skipping: TUCKMILL_REAL_GGUF is not a dense Qwen3 artifact");
            return;
        };
        let mut names = crate::model::TensorNames::for_layers(cfg.layers);
        names.resolve(&insp).unwrap();
        let graph = super::super::adapters::qwen3(&cfg, &names, &insp).unwrap();
        let source = super::super::GgufWeights::with_inspection(&file, &insp).unwrap();
        let capacity = 640usize;
        let kv = KvCache::F16;
        let plan = alloc_plan_with(
            &graph,
            &source,
            capacity,
            EngineOptions {
                kv,
                prefix_cache: false,
                mtp: false,
            },
        )
        .unwrap();
        let model = EngineModel::load_with(
            Qwen35Gpu::open().unwrap(),
            graph,
            &source,
            capacity,
            plan.total(),
            EngineOptions {
                kv,
                prefix_cache: false,
                mtp: false,
            },
        )
        .unwrap();
        let prompt: Vec<u32> = (0..512u32).map(|i| 1000 + (i * 7919) % 90_000).collect();
        let prefill = |out: &mut Vec<f32>| {
            let chunks: Vec<&[u32]> = prompt.chunks(PREFILL_BATCH).collect();
            for (c, chunk) in chunks.iter().enumerate() {
                model
                    .prefill_chunk(chunk, c + 1 == chunks.len(), out)
                    .unwrap();
            }
        };
        model.reset().unwrap();
        let mut warm = Vec::new();
        prefill(&mut warm);
        model.reset().unwrap();
        let t0 = std::time::Instant::now();
        let mut batched = Vec::new();
        prefill(&mut batched);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        model.reset().unwrap();
        let mut seq = Vec::new();
        for t in &prompt {
            model.forward_token(*t, &mut seq).unwrap();
        }
        let drift = batched
            .iter()
            .zip(&seq)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = seq.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
        eprintln!(
            "qwen3 512-token prefill x{PREFILL_BATCH}: {ms:.0} ms ({:.0} tok/s); batched vs sequential max|diff| {drift:.3e} (logit scale {scale:.2})",
            512e3 / ms
        );
        // The registered dense-Qwen3 GPU policy (executor tests): 1e-2 of
        // the logit scale, and the same top-1.
        assert!(drift <= 1e-2 * scale, "batched vs sequential within policy");
        assert_eq!(
            crate::sampling::argmax(&batched),
            crate::sampling::argmax(&seq),
            "top-1"
        );

        if std::env::var("TUCKMILL_PROFILE").ok().as_deref() == Some("1") {
            let ctx = model.context();
            model.reset().unwrap();
            model.set_grouped(false);
            ctx.set_profiling(true);
            let mut l = Vec::new();
            prefill(&mut l);
            ctx.fence().unwrap();
            let mut table = ctx.profile_snapshot();
            ctx.set_profiling(false);
            model.set_grouped(true);
            table.sort_by_key(|row| std::cmp::Reverse(row.1));
            let total: u64 = table.iter().map(|r| r.1).sum();
            eprintln!("ungrouped qwen3 prefill GPU time per operator (512 tokens):");
            for (label, nanos, count) in table.iter().take(16) {
                eprintln!(
                    "  {label:<28} {:>8.1} ms  {:>5.1}%  {:>6} buffers",
                    *nanos as f64 / 1e6,
                    100.0 * *nanos as f64 / total.max(1) as f64,
                    count
                );
            }
        }
    }
}

#[cfg(test)]
mod prefix_cache_tests {
    use super::*;
    use crate::qwen35_forward::{ArtifactTensorSource, TensorSource};
    use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

    fn gpu_or_skip() -> bool {
        if tuckmill_metal::probe().available {
            return true;
        }
        assert!(
            std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
            "TUCKMILL_REQUIRE_METAL is set but no Metal device is available"
        );
        eprintln!("skipping: no Metal device");
        false
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Prefill `prompt[start..]` in aligned chunks (as the executor does),
    /// then `steps` greedy decode steps fed with the argmax of the previous
    /// logits. Returns the last-prompt-token logits and each step's logits.
    fn run_from(model: &EngineModel, prompt: &[u32], start: usize, steps: usize) -> Vec<Vec<f32>> {
        assert_eq!(model.positions(), start);
        let mut logits = Vec::new();
        let rest = &prompt[start..];
        let chunks: Vec<&[u32]> = rest.chunks(PREFILL_BATCH).collect();
        for (c, chunk) in chunks.iter().enumerate() {
            model
                .prefill_chunk(chunk, c + 1 == chunks.len(), &mut logits)
                .unwrap();
        }
        let mut out = vec![logits.clone()];
        for _ in 0..steps {
            let next = crate::sampling::argmax(&logits).unwrap_or(0) as u32;
            model.forward_token(next, &mut logits).unwrap();
            out.push(logits.clone());
        }
        out
    }

    /// ADR-012 step 2 gate: a request resumed from the prefix snapshot
    /// produces bit-identical logits (last prompt token and the following
    /// decode steps) to a cold prefill of the same tokens; misses, exact
    /// matches and resets resume nothing; ledger == plan with the snapshot.
    fn check(source: &ArtifactTensorSource, vocab_mod: u32, base: u32) {
        let graph = crate::engine::adapters::qwen35(source.config(), source.layout());
        let pb = PREFILL_BATCH;
        let capacity = 3 * pb + 64;
        let on = EngineOptions {
            kv: KvCache::F16,
            prefix_cache: true,
            mtp: false,
        };
        let plan = alloc_plan_with(&graph, source, capacity, on).unwrap();
        let off_plan = alloc_plan_with(&graph, source, capacity, EngineOptions::default()).unwrap();
        let gdn_state: u64 = graph
            .layers
            .iter()
            .filter_map(|l| match &l.mixer {
                MixerSpec::GatedDeltaNet(g) => {
                    let (state, hist) = GdnBlock::state_elems(g);
                    Some(((state + hist) * 4) as u64)
                }
                _ => None,
            })
            .sum();
        assert_eq!(
            plan.state,
            off_plan.state + gdn_state,
            "the snapshot costs one copy of the DeltaNet state"
        );
        let model = EngineModel::load_with(
            Qwen35Gpu::open().unwrap(),
            graph,
            source,
            capacity,
            plan.total(),
            on,
        )
        .unwrap();
        assert_eq!(model.allocated_bytes(), plan.total(), "ledger == plan");

        let tok = |i: u32, salt: u32| base + (i * 7919 + salt) % vocab_mod;
        let prompt_a: Vec<u32> = (0..(2 * pb + 10) as u32).map(|i| tok(i, 13)).collect();
        // B shares A's first 2*pb tokens (A's snapshot boundary), then
        // diverges and runs past A's length.
        let mut prompt_b: Vec<u32> = prompt_a[..2 * pb].to_vec();
        prompt_b.extend((0..40u32).map(|i| tok(i, 101)));

        assert_eq!(
            model.begin_request(&prompt_a).unwrap(),
            0,
            "first request is cold"
        );
        let _ = run_from(&model, &prompt_a, 0, 4); // leaves stale KV rows past the boundary
        let hit = model.begin_request(&prompt_b).unwrap();
        assert_eq!(hit, 2 * pb, "B resumes at A's snapshot boundary");
        let resumed = run_from(&model, &prompt_b, hit, 6);

        model.reset().unwrap();
        let cold = run_from(&model, &prompt_b, 0, 6);
        for (step, (a, b)) in resumed.iter().zip(&cold).enumerate() {
            assert_eq!(
                bits(a),
                bits(b),
                "step {step}: resumed == cold, bit for bit"
            );
        }

        // reset() invalidated the snapshot; the cold run above (not started
        // through begin_request) planned none, so B is a miss now.
        assert_eq!(
            model.begin_request(&prompt_b).unwrap(),
            0,
            "reset invalidates"
        );
        let _ = run_from(&model, &prompt_b, 0, 0);
        // B snapshotted at its own boundary (2*pb): a prompt diverging
        // inside the prefix misses, and so does one equal to the prefix.
        let mut diverged = prompt_b.clone();
        diverged[pb] ^= 1;
        assert_eq!(
            model.begin_request(&diverged).unwrap(),
            0,
            "divergence misses"
        );
        let _ = run_from(&model, &diverged, 0, 0);
        assert_eq!(
            model.begin_request(&diverged[..2 * pb]).unwrap(),
            0,
            "a prompt equal to the prefix has nothing to prefill: miss"
        );
    }

    #[test]
    fn prefix_hit_is_bit_identical_to_a_cold_prefill_on_the_fixture() {
        if !gpu_or_skip() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q35.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen35()).unwrap();
        let source = ArtifactTensorSource::open(&path).unwrap();
        let vocab = source.config().vocab as u32;
        check(&source, vocab - 8, 0);
    }

    #[test]
    fn prefix_hit_is_bit_identical_to_a_cold_prefill_on_the_pinned_artifact() {
        let Some(path) = std::env::var("TUCKMILL_QWEN35_GGUF")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists())
        else {
            eprintln!("skipping: TUCKMILL_QWEN35_GGUF not set");
            return;
        };
        if !gpu_or_skip() {
            return;
        }
        let source = ArtifactTensorSource::open(&path).unwrap();
        check(&source, 90_000, 1000);
    }
}

#[cfg(test)]
mod mtp_tests {
    use super::*;
    use crate::qwen35_forward::{ArtifactTensorSource, TensorSource};
    use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

    fn gpu_or_skip() -> bool {
        if tuckmill_metal::probe().available {
            return true;
        }
        assert!(
            std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
            "TUCKMILL_REQUIRE_METAL is set but no Metal device is available"
        );
        eprintln!("skipping: no Metal device");
        false
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn load(source: &ArtifactTensorSource, capacity: usize) -> EngineModel {
        let graph = crate::engine::adapters::qwen35(source.config(), source.layout());
        let opts = EngineOptions {
            mtp: true,
            ..EngineOptions::default()
        };
        let plan = alloc_plan_with(&graph, source, capacity, opts).unwrap();
        let off = alloc_plan_with(&graph, source, capacity, EngineOptions::default()).unwrap();
        assert_eq!(plan.state, off.state, "speculation adds no planner KV");
        assert!(plan.weights > off.weights && plan.scratch > off.scratch);
        let model = EngineModel::load_with(
            Qwen35Gpu::open().unwrap(),
            graph,
            source,
            capacity,
            plan.total(),
            opts,
        )
        .unwrap();
        assert_eq!(model.allocated_bytes(), plan.total(), "ledger == plan");
        model
    }

    /// Greedy prefill of `prompt` (aligned chunks); the first generated token.
    fn prefill(model: &EngineModel, prompt: &[u32]) -> u32 {
        model.reset().unwrap();
        let chunks: Vec<&[u32]> = prompt.chunks(PREFILL_BATCH).collect();
        let mut first = None;
        for (c, chunk) in chunks.iter().enumerate() {
            if c + 1 == chunks.len() {
                first = model.prefill_chunk_greedy(chunk).unwrap();
            } else {
                model.prefill_chunk(chunk, false, &mut Vec::new()).unwrap();
            }
        }
        first.expect("finite logits")
    }

    /// Plain greedy decoding: `n` tokens after the prompt.
    fn plain(model: &EngineModel, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut out = vec![prefill(model, prompt)];
        while out.len() < n {
            let t = *out.last().unwrap();
            out.push(model.forward_token_greedy(t).unwrap().expect("finite"));
        }
        out
    }

    /// How a speculative run chooses its drafts.
    #[derive(Clone, Copy, Debug)]
    enum Drafts {
        /// The MTP head's own.
        Head,
        /// The true next tokens (every draft accepted).
        Oracle,
        /// Always wrong (every draft rejected).
        Wrong,
        /// Right on even cycles, wrong on odd ones.
        Alternating,
    }

    /// Speculative greedy decoding of at least `n` tokens; returns the
    /// tokens and the number of accepted drafts over the cycles run.
    fn speculative(
        model: &EngineModel,
        prompt: &[u32],
        n: usize,
        drafts: Drafts,
        reference: &[u32],
    ) -> (Vec<u32>, usize, usize) {
        let mut out = vec![prefill(model, prompt)];
        let (mut accepted, mut cycles) = (0, 0);
        let vocab = model.graph().vocab as u32;
        while out.len() < n {
            let t = *out.last().unwrap();
            let truth = reference[out.len()];
            let step = match drafts {
                Drafts::Head => model.speculate_greedy(t),
                Drafts::Oracle => model.speculate_with_draft(t, truth),
                Drafts::Wrong => model.speculate_with_draft(t, (truth + 1) % vocab),
                Drafts::Alternating if cycles % 2 == 0 => model.speculate_with_draft(t, truth),
                Drafts::Alternating => model.speculate_with_draft(t, (truth + 1) % vocab),
            }
            .unwrap();
            cycles += 1;
            out.push(step.first.expect("finite"));
            if step.accepted {
                accepted += 1;
                out.push(step.second.expect("finite"));
            } else {
                assert_eq!(step.second, None, "a rejected cycle commits one token");
            }
            assert_eq!(
                model.positions(),
                prompt.len() + out.len() - 1,
                "every committed token but the last is in the session"
            );
        }
        (out, accepted, cycles)
    }

    /// ADR-012 step 3 gate: speculative greedy output is token-identical to
    /// plain greedy decoding whatever the drafts, and the session state it
    /// leaves (recurrent state after rollbacks, KV rows) gives bit-identical
    /// logits on the next decode step.
    fn check(source: &ArtifactTensorSource, prompts: &[Vec<u32>], n: usize, policies: &[Drafts]) {
        let capacity = prompts.iter().map(Vec::len).max().unwrap() + n + 8;
        let model = load(source, capacity);
        for prompt in prompts {
            let reference = plain(&model, prompt, n + 2);
            for &drafts in policies {
                let (out, accepted, cycles) = speculative(&model, prompt, n, drafts, &reference);
                assert_eq!(
                    out[..],
                    reference[..out.len()],
                    "prompt {}, {drafts:?}: tokens identical to plain greedy",
                    prompt.len()
                );
                match drafts {
                    Drafts::Oracle => assert_eq!(accepted, cycles),
                    Drafts::Wrong => assert_eq!(accepted, 0),
                    _ => {}
                }
                eprintln!(
                    "prompt {}, {drafts:?}: {accepted}/{cycles} drafts accepted, {} tokens",
                    prompt.len(),
                    out.len()
                );
                // Next decode step from the speculative session vs a plain
                // session that consumed the same tokens.
                let last = *out.last().unwrap();
                let mut spec_logits = Vec::new();
                model.forward_token(last, &mut spec_logits).unwrap();
                let _ = prefill(&model, prompt);
                let mut plain_logits = Vec::new();
                for t in &out {
                    model.forward_token(*t, &mut plain_logits).unwrap();
                }
                assert_eq!(
                    bits(&spec_logits),
                    bits(&plain_logits),
                    "prompt {}, {drafts:?}: state after speculation, bit for bit",
                    prompt.len()
                );
            }
        }
        // The input conventions change only the drafts (ADR-012 step 3):
        // every one stays token-identical; the acceptance rates printed
        // here are what chose the default.
        let reference = plain(&model, &prompts[0], n + 2);
        for embedding_first in [true, false] {
            for hidden_post_norm in [true, false] {
                let conv = MtpConvention {
                    embedding_first,
                    hidden_post_norm,
                };
                model.set_mtp_convention(conv);
                let (out, accepted, cycles) =
                    speculative(&model, &prompts[0], n, Drafts::Head, &reference);
                assert_eq!(
                    out[..],
                    reference[..out.len()],
                    "{conv:?}: identical tokens"
                );
                eprintln!("{conv:?}: {accepted}/{cycles} drafts accepted");
            }
        }
        model.set_mtp_convention(MtpConvention::default());
        // Speculation continues only its own session.
        let t = prefill(&model, &prompts[0]);
        assert!(
            model.speculate_greedy(t + 1).is_err(),
            "wrong token refused"
        );
        let t = prefill(&model, &prompts[0]);
        model.forward_token_greedy(t).unwrap();
        assert!(!model.can_speculate(), "a plain step ends speculation");
    }

    #[test]
    fn speculative_greedy_is_token_identical_on_the_fixture() {
        if !gpu_or_skip() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q35.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen35()).unwrap();
        let source = ArtifactTensorSource::open(&path).unwrap();
        let vocab = source.config().vocab as u32;
        let prompts: Vec<Vec<u32>> = [1usize, 5, PREFILL_BATCH + 3]
            .iter()
            .map(|&len| {
                (0..len as u32)
                    .map(|i| (i * 37 + 5) % (vocab - 8))
                    .collect()
            })
            .collect();
        check(
            &source,
            &prompts,
            24,
            &[
                Drafts::Head,
                Drafts::Oracle,
                Drafts::Wrong,
                Drafts::Alternating,
            ],
        );
    }

    #[test]
    fn speculative_greedy_is_token_identical_on_the_pinned_artifact() {
        let Some(path) = std::env::var("TUCKMILL_QWEN35_GGUF")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists())
        else {
            eprintln!("skipping: TUCKMILL_QWEN35_GGUF not set");
            return;
        };
        if !gpu_or_skip() {
            return;
        }
        let source = ArtifactTensorSource::open(&path).unwrap();
        let prompts: Vec<Vec<u32>> = [7usize, PREFILL_BATCH + 40]
            .iter()
            .map(|&len| {
                (0..len as u32)
                    .map(|i| 1000 + (i * 7919 + 13) % 90_000)
                    .collect()
            })
            .collect();
        check(
            &source,
            &prompts,
            64,
            &[
                Drafts::Head,
                Drafts::Oracle,
                Drafts::Wrong,
                Drafts::Alternating,
            ],
        );
    }
}
