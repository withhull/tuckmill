//! Scalar f32 forward pass for dense Qwen3 (Stage 0 numerical oracle).
//!
//! Weights stream through the bounded block loader; every block used by a
//! matvec chunk is pinned and leased through the completion tracker until the
//! chunk's work is finished, then released. The profile is explicit: f32
//! storage and accumulation, RMSNorm/RoPE/SwiGLU/GQA exactly as declared by
//! the artifact's adapter, no approximations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::gguf::GgufInspection;

use crate::completion::CompletionTracker;
use crate::kernels::ResolvedKernels;
use crate::loader::{BlockLoader, TensorEntry};
use crate::model::{Qwen3Config, TensorNames};
use crate::tensor::{self, f16_to_f32, f32_to_f16_bits, quantize_row_q8_0};

/// Small tensors dequantized once at load: norms and the RoPE table.
#[derive(Debug)]
pub struct Norms {
    pub output_norm: Vec<f32>,
    pub attn_norm: Vec<Vec<f32>>,
    pub ffn_norm: Vec<Vec<f32>>,
    pub q_norm: Vec<Vec<f32>>,
    pub k_norm: Vec<Vec<f32>>,
}

/// Resolved loader indices for streamed tensors.
#[derive(Debug)]
pub struct TensorIds {
    pub token_embd: usize,
    pub output: usize,
    pub attn_q: Vec<usize>,
    pub attn_k: Vec<usize>,
    pub attn_v: Vec<usize>,
    pub attn_output: Vec<usize>,
    pub ffn_gate: Vec<usize>,
    pub ffn_up: Vec<usize>,
    pub ffn_down: Vec<usize>,
}

#[derive(Debug)]
pub struct ModelWeights {
    pub cfg: Qwen3Config,
    pub names: TensorNames,
    pub norms: Norms,
    pub rope_inv_freq: Vec<f32>,
    pub ids: TensorIds,
}

impl ModelWeights {
    /// Dequantize the small tensors and resolve streamed-tensor indices
    /// against the loader. Everything here is bounded: norms are `hidden` and
    /// `head_dim` sized; the RoPE table is `head_dim/2`.
    pub fn prepare(loader: &BlockLoader, cfg: Qwen3Config, names: TensorNames) -> Result<Self> {
        let need = |name: &str| -> Result<usize> {
            loader.tensor_index(name).ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::UnsupportedArchitecture,
                    format!("artifact is missing required tensor {name:?}"),
                )
            })
        };
        let deq = |name: &str| loader.dequant_tensor(name);
        let layers = cfg.layers as usize;

        let output_norm = deq(&names.output_norm)?;
        let mut attn_norm = Vec::with_capacity(layers);
        let mut ffn_norm = Vec::with_capacity(layers);
        let mut q_norm = Vec::with_capacity(layers);
        let mut k_norm = Vec::with_capacity(layers);
        for l in 0..layers {
            attn_norm.push(deq(&names.attn_norm[l])?);
            ffn_norm.push(deq(&names.ffn_norm[l])?);
            q_norm.push(deq(&names.attn_q_norm[l])?);
            k_norm.push(deq(&names.attn_k_norm[l])?);
        }
        for (v, expect) in [
            (&output_norm, cfg.hidden),
            (q_norm.first().unwrap_or(&vec![]), cfg.key_length),
            (k_norm.first().unwrap_or(&vec![]), cfg.key_length),
        ] {
            if v.len() as u64 != expect {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!(
                        "norm tensor has {} elements but geometry declares {expect}",
                        v.len()
                    ),
                ));
            }
        }

        let mut rope_inv_freq = Vec::with_capacity((cfg.key_length / 2) as usize);
        for i in 0..cfg.key_length / 2 {
            let exponent = 2.0 * i as f64 / cfg.key_length as f64;
            rope_inv_freq.push((1.0 / (cfg.rope_base as f64).powf(exponent)) as f32);
        }

        let output_name = names.output.clone();
        let ids = TensorIds {
            token_embd: need(&names.token_embd)?,
            output: match &output_name {
                Some(name) => need(name)?,
                None => need(&names.token_embd)?, // tied embeddings
            },
            attn_q: names
                .attn_q
                .iter()
                .map(|n| need(n))
                .collect::<Result<_>>()?,
            attn_k: names
                .attn_k
                .iter()
                .map(|n| need(n))
                .collect::<Result<_>>()?,
            attn_v: names
                .attn_v
                .iter()
                .map(|n| need(n))
                .collect::<Result<_>>()?,
            attn_output: names
                .attn_output
                .iter()
                .map(|n| need(n))
                .collect::<Result<_>>()?,
            ffn_gate: names
                .ffn_gate
                .iter()
                .map(|n| need(n))
                .collect::<Result<_>>()?,
            ffn_up: names
                .ffn_up
                .iter()
                .map(|n| need(n))
                .collect::<Result<_>>()?,
            ffn_down: names
                .ffn_down
                .iter()
                .map(|n| need(n))
                .collect::<Result<_>>()?,
        };
        Ok(Self {
            cfg,
            names,
            norms: Norms {
                output_norm,
                attn_norm,
                ffn_norm,
                q_norm,
                k_norm,
            },
            rope_inv_freq,
            ids,
        })
    }
}

/// KV cache precision. Mirrors llama.cpp's explicit `type_k`/`type_v`
/// selection so diagnostic comparisons can match both sides; entries stored
/// "f16" are f16-representable f32 values, never a different cache
/// implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvPrecision {
    F32,
    F16,
}

impl KvPrecision {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }

    fn round(self, v: f32) -> f32 {
        match self {
            Self::F32 => v,
            Self::F16 => f16_to_f32(f32_to_f16_bits(v)),
        }
    }
}

/// Bounded KV state for one sequence. Capacity is fixed at admission time.
#[derive(Debug)]
pub struct KvState {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub len: usize,
    cap: usize,
    key_len: u64,
    value_len: u64,
    precision: KvPrecision,
}

impl KvState {
    /// Bytes of KV state for a capacity, without allocating.
    pub fn bytes_for(cfg: &Qwen3Config, cap: u64) -> u64 {
        cfg.layers
            .saturating_mul(cfg.kv_heads)
            .saturating_mul(cap)
            .saturating_mul(cfg.key_length + cfg.value_length)
            .saturating_mul(4)
    }

    pub fn new(cfg: &Qwen3Config, cap: u64, precision: KvPrecision) -> Result<Self> {
        let total = Self::bytes_for(cfg, cap);
        if total > Self::MAX_KV_BYTES {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!(
                    "stage-0 KV state for {cap} tokens across {} layers needs {} bytes, beyond the executor's {} bound",
                    cfg.layers,
                    total,
                    Self::MAX_KV_BYTES
                ),
            ));
        }
        let k_layer = vec![0.0f32; (cfg.kv_heads * cap * cfg.key_length) as usize];
        let v_layer = vec![0.0f32; (cfg.kv_heads * cap * cfg.value_length) as usize];
        Ok(Self {
            k: vec![k_layer; cfg.layers as usize],
            v: vec![v_layer; cfg.layers as usize],
            len: 0,
            cap: cap as usize,
            key_len: cfg.key_length,
            value_len: cfg.value_length,
            precision,
        })
    }

    /// Honest stage-0 bound for a bounded-context executor (not a product
    /// promise; admission remains the supervisor's job).
    pub const MAX_KV_BYTES: u64 = 6 * 1024 * 1024 * 1024;

    fn store(&mut self, layer: usize, head: u64, pos: usize, k_row: &[f32], v_row: &[f32]) {
        let k_off = ((head as usize) * self.cap + pos) * self.key_len as usize;
        let k_slot = &mut self.k[layer][k_off..k_off + self.key_len as usize];
        for (slot, value) in k_slot.iter_mut().zip(k_row.iter()) {
            *slot = self.precision.round(*value);
        }
        let v_off = ((head as usize) * self.cap + pos) * self.value_len as usize;
        let v_slot = &mut self.v[layer][v_off..v_off + self.value_len as usize];
        for (slot, value) in v_slot.iter_mut().zip(v_row.iter()) {
            *slot = self.precision.round(*value);
        }
    }

    fn k_row(&self, layer: usize, head: u64, pos: usize) -> &[f32] {
        let off = ((head as usize) * self.cap + pos) * self.key_len as usize;
        &self.k[layer][off..off + self.key_len as usize]
    }

    fn v_row(&self, layer: usize, head: u64, pos: usize) -> &[f32] {
        let off = ((head as usize) * self.cap + pos) * self.value_len as usize;
        &self.v[layer][off..off + self.value_len as usize]
    }
}

/// Declared numerical profile for weight×activation products.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NumericalProfile {
    /// Exact f32 scalar arithmetic: activations stay f32, weights are
    /// dequantized exactly. The most precise path; the oracle.
    OracleF32,
    /// Mirrors llama.cpp CPU kernels (pinned b11695): activations are
    /// quantized per weight dtype — q8_0 (amax/127, f16 scale, nearest-even)
    /// for Q8_0 weights, f16 rounding for F16 weights — and dots accumulate
    /// per block. This is the profile under which TuckCore and llama.cpp can
    /// be compared as the same declared numerical contract.
    LlamaCpu,
}

impl NumericalProfile {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "oracle-f32" => Some(Self::OracleF32),
            "llama-cpu" => Some(Self::LlamaCpu),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::OracleF32 => "oracle-f32",
            Self::LlamaCpu => "llama-cpu",
        }
    }
}

/// Phase counters for attributing wall-clock time (explicitly enabled per
/// run; disabled counters are never touched by the hot path except one
/// `if let Some` branch).
#[derive(Debug, Default)]
pub struct ProfileCounters {
    pub enabled: bool,
    pub forward_nanos: AtomicU64,
    pub matvec_nanos: AtomicU64,
    pub matvec_fetch_nanos: AtomicU64,
    pub matvec_compute_nanos: AtomicU64,
    pub embed_nanos: AtomicU64,
    pub attention_nanos: AtomicU64,
    pub norm_rope_nanos: AtomicU64,
    pub ffn_nanos: AtomicU64,
    pub load_nanos: AtomicU64,
    pub content_verify_nanos: AtomicU64,
    pub tokens: AtomicU64,
}

impl ProfileCounters {
    fn add(counter: &AtomicU64, start: Instant) {
        counter.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Public single-timer helper (executor phases).
    pub fn record(counter: &AtomicU64, start: Instant) {
        Self::add(counter, start);
    }
}

/// The forward executor: streams weights, computes one token at a time.
pub struct Forward<'a> {
    pub loader: &'a BlockLoader,
    pub tracker: &'a CompletionTracker,
    pub weights: &'a ModelWeights,
    pub profile: NumericalProfile,
    /// Resolved packed-kernel implementation (K01). Reported in stats and
    /// records; `Scalar` forces the portable reference kernels.
    pub kernels: ResolvedKernels,
    /// Called between chunks/layers: services external control (poll hook)
    /// and applies pending cache trims at a safe point. Never releases
    /// in-flight buffers (pinned chunks are protected by the loader).
    pub control: &'a dyn Fn(),
    /// Optional phase counters (None = not profiling; zero overhead beyond
    /// the branch).
    pub counters: Option<&'a ProfileCounters>,
}

/// Cancellation hook checked between layers and chunks.
pub type CancelledFn<'a> = &'a dyn Fn() -> bool;

impl<'a> Forward<'a> {
    /// Process one token at `pos` (0-based), appending to `kv`, producing
    /// logits for the next token. Returns the logits vector (vocab sized).
    pub fn forward_token(
        &self,
        kv: &mut KvState,
        token: u32,
        pos: usize,
        cancelled: CancelledFn<'_>,
    ) -> Result<Vec<f32>> {
        let _total = self.counters.map(|_| Instant::now());
        let result = self.forward_token_inner(kv, token, pos, cancelled);
        if let (Some(counters), Some(start)) = (self.counters, _total) {
            ProfileCounters::add(&counters.forward_nanos, start);
            counters.tokens.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn forward_token_inner(
        &self,
        kv: &mut KvState,
        token: u32,
        pos: usize,
        cancelled: CancelledFn<'_>,
    ) -> Result<Vec<f32>> {
        let cfg = &self.weights.cfg;
        if (token as u64) >= cfg.vocab {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!(
                    "token id {token} is outside the vocabulary (0..{})",
                    cfg.vocab
                ),
            ));
        }
        if pos >= kv.cap {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!("position {pos} exceeds KV capacity {}", kv.cap),
            ));
        }
        let hidden = cfg.hidden as usize;
        let heads = cfg.heads as usize;
        let kv_heads = cfg.kv_heads as usize;
        let key_len = cfg.key_length as usize;
        let head_dim = key_len;
        let group = heads / kv_heads;

        let embed_start = self.counters.map(|_| Instant::now());
        let mut x = self.embed_row(self.weights.ids.token_embd, token)?;
        if let (Some(counters), Some(start)) = (self.counters, embed_start) {
            ProfileCounters::add(&counters.embed_nanos, start);
        }
        debug_assert_eq!(x.len(), hidden);

        let mut logits: Vec<f32> = Vec::new();
        for layer in 0..cfg.layers as usize {
            if cancelled() {
                return Err(cancelled_error());
            }
            (self.control)();
            // ---- attention block ----
            let attn_start = self.counters.map(|_| Instant::now());
            let normed = rmsnorm(&x, &self.weights.norms.attn_norm[layer], cfg.rms_eps);
            let mut q = vec![0.0f32; heads * head_dim];
            self.matvec(self.weights.ids.attn_q[layer], &normed, &mut q, cancelled)?;
            let mut k = vec![0.0f32; kv_heads * key_len];
            self.matvec(self.weights.ids.attn_k[layer], &normed, &mut k, cancelled)?;
            let mut v = vec![0.0f32; kv_heads * cfg.value_length as usize];
            self.matvec(self.weights.ids.attn_v[layer], &normed, &mut v, cancelled)?;

            let nr_start = self.counters.map(|_| Instant::now());
            for h in 0..heads {
                let qh = &mut q[h * head_dim..(h + 1) * head_dim];
                rmsnorm_in_place(qh, &self.weights.norms.q_norm[layer], cfg.rms_eps);
                rope_apply(qh, pos, &self.weights.rope_inv_freq);
            }
            for h in 0..kv_heads {
                let kh = &mut k[h * key_len..(h + 1) * key_len];
                rmsnorm_in_place(kh, &self.weights.norms.k_norm[layer], cfg.rms_eps);
                rope_apply(kh, pos, &self.weights.rope_inv_freq);
            }
            if let (Some(counters), Some(start)) = (self.counters, nr_start) {
                ProfileCounters::add(&counters.norm_rope_nanos, start);
            }

            for h in 0..kv_heads {
                let kv_row = &k[h * key_len..(h + 1) * key_len];
                let v_row = &v[h * cfg.value_length as usize..(h + 1) * cfg.value_length as usize];
                kv.store(layer, h as u64, pos, kv_row, v_row);
            }
            let seq = pos + 1;
            let scale = 1.0f32 / (key_len as f32).sqrt();
            let mut attn_out = vec![0.0f32; heads * head_dim];
            for h in 0..heads {
                let kv_head = (h / group) as u64;
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let mut scores = vec![0.0f32; seq];
                let mut max = f32::NEG_INFINITY;
                for (j, score) in scores.iter_mut().enumerate() {
                    let kj = kv.k_row(layer, kv_head, j);
                    let mut dot = 0.0f32;
                    for i in 0..key_len {
                        dot += qh[i] * kj[i];
                    }
                    *score = dot * scale;
                    if *score > max {
                        max = *score;
                    }
                }
                let mut sum = 0.0f32;
                for score in scores.iter_mut() {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                let out_h = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                if sum > 0.0 {
                    for (j, score) in scores.iter().enumerate() {
                        let w = score / sum;
                        let vj = kv.v_row(layer, kv_head, j);
                        for i in 0..out_h.len() {
                            out_h[i] += w * vj[i];
                        }
                    }
                }
            }
            let mut proj = vec![0.0f32; hidden];
            self.matvec(
                self.weights.ids.attn_output[layer],
                &attn_out,
                &mut proj,
                cancelled,
            )?;
            for i in 0..hidden {
                x[i] += proj[i];
            }
            if let (Some(counters), Some(start)) = (self.counters, attn_start) {
                ProfileCounters::add(&counters.attention_nanos, start);
            }

            // ---- FFN block (SwiGLU) ----
            let ffn_start = self.counters.map(|_| Instant::now());
            let normed = rmsnorm(&x, &self.weights.norms.ffn_norm[layer], cfg.rms_eps);
            let mut gate = vec![0.0f32; cfg.ffn as usize];
            self.matvec(
                self.weights.ids.ffn_gate[layer],
                &normed,
                &mut gate,
                cancelled,
            )?;
            let mut up = vec![0.0f32; cfg.ffn as usize];
            self.matvec(self.weights.ids.ffn_up[layer], &normed, &mut up, cancelled)?;
            for i in 0..gate.len() {
                gate[i] = silu(gate[i]) * up[i];
            }
            let mut down = vec![0.0f32; hidden];
            self.matvec(
                self.weights.ids.ffn_down[layer],
                &gate,
                &mut down,
                cancelled,
            )?;
            for i in 0..hidden {
                x[i] += down[i];
            }
            if let (Some(counters), Some(start)) = (self.counters, ffn_start) {
                ProfileCounters::add(&counters.ffn_nanos, start);
            }
        }
        if cancelled() {
            return Err(cancelled_error());
        }
        let normed = rmsnorm(&x, &self.weights.norms.output_norm, cfg.rms_eps);
        self.matvec(self.weights.ids.output, &normed, &mut logits, cancelled)?;
        kv.len = seq_len(kv.len, pos);
        Ok(logits)
    }

    /// Embedding row lookup (direct positioned read; small and uncached).
    fn embed_row(&self, tensor: usize, token: u32) -> Result<Vec<f32>> {
        let entry = self.loader.entry(tensor);
        let bytes = self.loader.read_rows_direct(tensor, token as u64, 1)?;
        let mut out = Vec::with_capacity(entry.dims[0] as usize);
        tensor::dequant_into(entry.type_id, &bytes, &mut out)?;
        Ok(out)
    }

    /// Streaming, block-bounded matrix-vector product. Each chunk is pinned
    /// and leased until its work completes.
    pub fn matvec(
        &self,
        tensor: usize,
        x: &[f32],
        out: &mut Vec<f32>,
        cancelled: CancelledFn<'_>,
    ) -> Result<()> {
        let total_start = self.counters.map(|_| Instant::now());
        let result = self.matvec_inner(tensor, x, out, cancelled);
        if let (Some(counters), Some(start)) = (self.counters, total_start) {
            ProfileCounters::add(&counters.matvec_nanos, start);
        }
        result
    }

    fn matvec_inner(
        &self,
        tensor: usize,
        x: &[f32],
        out: &mut Vec<f32>,
        cancelled: CancelledFn<'_>,
    ) -> Result<()> {
        let entry: &TensorEntry = self.loader.entry(tensor);
        let row = entry.row.as_ref().ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedQuantization,
                format!(
                    "tensor {:?} cannot be streamed row-wise in this build (rows are not block aligned)",
                    entry.name
                ),
            )
        })?;
        if row.elems_per_row as usize != x.len() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidArtifact,
                format!(
                    "tensor {:?} expects input width {} but the forward pass produced {}",
                    entry.name,
                    row.elems_per_row,
                    x.len()
                ),
            ));
        }
        out.resize(row.rows as usize, 0.0);
        if self.profile == NumericalProfile::LlamaCpu {
            match entry.type_id {
                8 => {
                    return self.matvec_q8_0_llama_cpu(
                        tensor,
                        x,
                        out,
                        row.elems_per_row as usize,
                        row.bytes_per_row as usize,
                        row.rows_per_chunk,
                        cancelled,
                    )
                }
                1 => {
                    return self.matvec_f16_llama_cpu(
                        tensor,
                        x,
                        out,
                        row.elems_per_row as usize,
                        row.bytes_per_row as usize,
                        row.rows_per_chunk,
                        cancelled,
                    )
                }
                _ => {}
            }
        }
        // K01 fused kernels (oracle profile): no f32 scratch materialization
        // for the dominant types; everything else keeps dequant+dot.
        if matches!(entry.type_id, 8 | 1) {
            return self.matvec_fused(
                tensor,
                x,
                out,
                row.elems_per_row as usize,
                row.bytes_per_row as usize,
                row.rows_per_chunk,
                cancelled,
            );
        }
        let mut scratch: Vec<f32> = Vec::new();
        for chunk in 0..entry.chunks {
            if cancelled() {
                return Err(cancelled_error());
            }
            (self.control)();
            let fetch_start = self.counters.map(|_| Instant::now());
            let block = self.loader.block(tensor, chunk as u32)?;
            self.loader.pin(tensor, chunk as u32);
            let fence = self.tracker.submit(vec![block.clone()]);
            if let (Some(counters), Some(start)) = (self.counters, fetch_start) {
                ProfileCounters::add(&counters.matvec_fetch_nanos, start);
            }
            let compute_start = self.counters.map(|_| Instant::now());
            let result = (|| -> Result<()> {
                scratch.clear();
                tensor::dequant_into(entry.type_id, block.bytes(), &mut scratch)?;
                let elems = row.elems_per_row as usize;
                let rows_in_block = scratch.len() / elems;
                let first_row = (chunk * row.rows_per_chunk) as usize;
                for r in 0..rows_in_block {
                    let row_slice = &scratch[r * elems..(r + 1) * elems];
                    let mut acc = 0.0f32;
                    for i in 0..elems {
                        acc += row_slice[i] * x[i];
                    }
                    out[first_row + r] = acc;
                }
                Ok(())
            })();
            if let (Some(counters), Some(start)) = (self.counters, compute_start) {
                ProfileCounters::add(&counters.matvec_compute_nanos, start);
            }
            // Completion fence: release the lease (and unpin) only now.
            let released = self.tracker.complete(fence);
            debug_assert!(released.is_some());
            self.loader.unpin(tensor, chunk as u32);
            result?;
        }
        Ok(())
    }

    /// Fused packed dot path: Q8_0 rows use int8→f32 lanes; F16 rows use
    /// conversion + fma. Blocks are pinned and leased per chunk exactly like
    /// every other path.
    #[allow(clippy::too_many_arguments)]
    fn matvec_fused(
        &self,
        tensor: usize,
        x: &[f32],
        out: &mut [f32],
        elems_per_row: usize,
        bytes_per_row: usize,
        rows_per_chunk: u64,
        cancelled: CancelledFn<'_>,
    ) -> Result<()> {
        let entry = self.loader.entry(tensor);
        for chunk in 0..entry.chunks {
            if cancelled() {
                return Err(cancelled_error());
            }
            (self.control)();
            let fetch_start = self.counters.map(|_| Instant::now());
            let block = self.loader.block(tensor, chunk as u32)?;
            self.loader.pin(tensor, chunk as u32);
            let fence = self.tracker.submit(vec![block.clone()]);
            if let (Some(counters), Some(start)) = (self.counters, fetch_start) {
                ProfileCounters::add(&counters.matvec_fetch_nanos, start);
            }
            let compute_start = self.counters.map(|_| Instant::now());
            let bytes = block.bytes();
            let rows_in_block = bytes.len() / bytes_per_row;
            let first_row = (chunk * rows_per_chunk) as usize;
            for r in 0..rows_in_block {
                let row_bytes = &bytes[r * bytes_per_row..(r + 1) * bytes_per_row];
                let acc = match entry.type_id {
                    8 => crate::kernels::dot_q8_0_x_f32(self.kernels, row_bytes, x)?,
                    1 => crate::kernels::dot_f16_x_f32(self.kernels, row_bytes, x)?,
                    _ => unreachable!("matvec_fused is entered only for Q8_0/F16"),
                };
                out[first_row + r] = acc;
            }
            let _ = elems_per_row;
            if let (Some(counters), Some(start)) = (self.counters, compute_start) {
                ProfileCounters::add(&counters.matvec_compute_nanos, start);
            }
            let released = self.tracker.complete(fence);
            debug_assert!(released.is_some());
            self.loader.unpin(tensor, chunk as u32);
        }
        Ok(())
    }

    /// Per-block Q8_0 dot body (kept as a method: borrow discipline around
    /// the completion fence is easier to audit without a closure).
    #[allow(clippy::too_many_arguments)]
    fn q8_block_dot(
        &self,
        bytes: &[u8],
        bytes_per_row: usize,
        rows_per_chunk: u64,
        chunk: u64,
        x_qs: &[[i8; 32]],
        x_d: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let rows_in_block = bytes.len() / bytes_per_row;
        let first_row = (chunk * rows_per_chunk) as usize;
        for r in 0..rows_in_block {
            let row_bytes = &bytes[r * bytes_per_row..(r + 1) * bytes_per_row];
            out[first_row + r] = crate::kernels::dot_q8_0_q8_0(self.kernels, row_bytes, x_qs, x_d)?;
        }
        Ok(())
    }

    /// Per-block F16 dot body (f16-rounded activations, f32 accumulation).
    fn f16_block_dot(
        &self,
        bytes: &[u8],
        bytes_per_row: usize,
        rows_per_chunk: u64,
        chunk: u64,
        x16: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let rows_in_block = bytes.len() / bytes_per_row;
        let first_row = (chunk * rows_per_chunk) as usize;
        for r in 0..rows_in_block {
            let row_bytes = &bytes[r * bytes_per_row..(r + 1) * bytes_per_row];
            out[first_row + r] = crate::kernels::dot_f16_x_f32(self.kernels, row_bytes, x16)?;
        }
        Ok(())
    }

    /// Q8_0 weights × q8_0-quantized activations, mirroring ggml's
    /// `quantize_row_q8_0` + int32 block dot with an f16-rounded scale.
    #[allow(clippy::too_many_arguments)]
    fn matvec_q8_0_llama_cpu(
        &self,
        tensor: usize,
        x: &[f32],
        out: &mut [f32],
        elems_per_row: usize,
        bytes_per_row: usize,
        rows_per_chunk: u64,
        cancelled: CancelledFn<'_>,
    ) -> Result<()> {
        let mut x_qs: Vec<[i8; 32]> = Vec::new();
        let mut x_d: Vec<f32> = Vec::new();
        quantize_row_q8_0(x, &mut x_qs, &mut x_d);
        let entry = self.loader.entry(tensor);
        for chunk in 0..entry.chunks {
            if cancelled() {
                return Err(cancelled_error());
            }
            (self.control)();
            let block = self.loader.block(tensor, chunk as u32)?;
            self.loader.pin(tensor, chunk as u32);
            let fence = self.tracker.submit(vec![block.clone()]);
            let result = self.q8_block_dot(
                block.bytes(),
                bytes_per_row,
                rows_per_chunk,
                chunk,
                &x_qs,
                &x_d,
                out,
            );
            let released = self.tracker.complete(fence);
            debug_assert!(released.is_some());
            self.loader.unpin(tensor, chunk as u32);
            result?;
        }
        let _ = elems_per_row;
        Ok(())
    }

    /// F16 weights × f16-rounded activations with f32 accumulation, mirroring
    /// ggml's F16 CPU dot (activations are converted to the weight's
    /// `vec_dot_type` before the dot).
    #[allow(clippy::too_many_arguments)]
    fn matvec_f16_llama_cpu(
        &self,
        tensor: usize,
        x: &[f32],
        out: &mut [f32],
        elems_per_row: usize,
        bytes_per_row: usize,
        rows_per_chunk: u64,
        cancelled: CancelledFn<'_>,
    ) -> Result<()> {
        let x16: Vec<f32> = x.iter().map(|v| f16_to_f32(f32_to_f16_bits(*v))).collect();
        let entry = self.loader.entry(tensor);
        for chunk in 0..entry.chunks {
            if cancelled() {
                return Err(cancelled_error());
            }
            (self.control)();
            let fetch_start = self.counters.map(|_| Instant::now());
            let block = self.loader.block(tensor, chunk as u32)?;
            self.loader.pin(tensor, chunk as u32);
            let fence = self.tracker.submit(vec![block.clone()]);
            if let (Some(counters), Some(start)) = (self.counters, fetch_start) {
                ProfileCounters::add(&counters.matvec_fetch_nanos, start);
            }
            let compute_start = self.counters.map(|_| Instant::now());
            let result = self.f16_block_dot(
                block.bytes(),
                bytes_per_row,
                rows_per_chunk,
                chunk,
                &x16,
                out,
            );
            if let (Some(counters), Some(start)) = (self.counters, compute_start) {
                ProfileCounters::add(&counters.matvec_compute_nanos, start);
            }
            let released = self.tracker.complete(fence);
            debug_assert!(released.is_some());
            self.loader.unpin(tensor, chunk as u32);
            result?;
        }
        let _ = elems_per_row;
        Ok(())
    }

    /// Teacher-forced logits at `position` for a token sequence, using a
    /// fresh KV state bounded by the sequence length.
    pub fn logits_at(
        &self,
        tokens: &[u32],
        position: usize,
        cancelled: CancelledFn<'_>,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() || position >= tokens.len() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "scoring position is outside the token sequence",
            ));
        }
        let mut kv = KvState::new(&self.weights.cfg, (position + 1) as u64, KvPrecision::F32)?;
        let mut logits = Vec::new();
        for (pos, token) in tokens[..=position].iter().enumerate() {
            logits = self.forward_token(&mut kv, *token, pos, cancelled)?;
        }
        Ok(logits)
    }
}

fn seq_len(old_len: usize, pos: usize) -> usize {
    old_len.max(pos + 1)
}

fn cancelled_error() -> TuckmillError {
    TuckmillError::new(
        ErrorCode::Cancelled,
        "generation cancelled at a safe boundary",
    )
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RMSNorm: y = x / sqrt(mean(x^2) + eps) * w (f32 accumulation).
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    rmsnorm_in_place(&mut out, w, eps);
    out
}

pub fn rmsnorm_in_place(x: &mut [f32], w: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), w.len());
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0f32 / (ss / x.len() as f32 + eps).sqrt();
    for (xi, wi) in x.iter_mut().zip(w.iter()) {
        *xi = *xi * inv * *wi;
    }
}

/// NeoX-style RoPE over the split-half convention used by Qwen:
/// pairs (x[i], x[i + d/2]) rotate by pos * inv_freq[i].
pub fn rope_apply(x: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        let theta = pos as f32 * inv_freq[i];
        let (s, c) = theta.sin_cos();
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

/// Prepare the streamed tensors list from a GGUF inspection (bounded rows).
pub fn tensor_entries(inspection: &GgufInspection, block_bytes: u64) -> Vec<TensorEntry> {
    inspection
        .tensors
        .iter()
        .map(|t| {
            // 1D tensors (norms) and non-matrix shapes are handled whole.
            let row = tensor::row_geometry(t.type_id, &t.dims, block_bytes);
            let chunks = match &row {
                Some(r) => r.rows.div_ceil(r.rows_per_chunk),
                None => t.byte_size.div_ceil(block_bytes).max(1),
            };
            TensorEntry {
                name: t.name.clone(),
                type_id: t.type_id,
                dims: t.dims.clone(),
                rel_offset: t.offset,
                byte_size: t.byte_size,
                row,
                chunks,
            }
        })
        .collect()
}

/// Convenience constructor used by the executor and tests.
pub fn prepare(
    loader: &BlockLoader,
    cfg: Qwen3Config,
    names: TensorNames,
) -> Result<Arc<ModelWeights>> {
    Ok(Arc::new(ModelWeights::prepare(loader, cfg, names)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::BlockPolicy;
    use crate::model;
    use tuckmill_manifest::adapter::AdapterRegistry;
    use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

    struct Fixture {
        _dir: tempfile::TempDir,
        loader: Arc<BlockLoader>,
        tracker: Arc<CompletionTracker>,
        weights: Arc<ModelWeights>,
    }

    fn fixture(policy: BlockPolicy) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let registry = AdapterRegistry::builtin().unwrap();
        let (cfg, _adapter) = model::resolve(&insp, &registry).unwrap();
        let entries = tensor_entries(&insp, policy.block_bytes);
        let loader =
            Arc::new(BlockLoader::open(path.as_path(), insp.data_start, entries, policy).unwrap());
        let mut names = TensorNames::for_layers(cfg.layers);
        names.resolve(&insp).unwrap();
        let weights = prepare(&loader, cfg, names).unwrap();
        Fixture {
            _dir: dir,
            loader,
            tracker: Arc::new(CompletionTracker::new()),
            weights,
        }
    }

    #[test]
    fn rmsnorm_matches_manual_computation() {
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let y = rmsnorm(&x, &w, 0.0);
        let rms = ((9.0f32 + 16.0) / 2.0).sqrt();
        assert!((y[0] - 3.0 / rms).abs() < 1e-6);
        assert!((y[1] - 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn rope_rotates_pairs_preserving_norm() {
        let mut x = [1.0f32, 0.0, 0.5, 1.5];
        let before: f32 = x.iter().map(|v| v * v).sum();
        let freq = [1.0f32, 0.5];
        rope_apply(&mut x, 3, &freq);
        let after: f32 = x.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-5, "rotation preserves norm");
        // theta = 3 for pair 0: (1.0, 0.5) -> (c - 0.5 s, s + 0.5 c).
        let (s3, c3) = 3.0f32.sin_cos();
        assert!((x[0] - (c3 - 0.5 * s3)).abs() < 1e-6);
        assert!((x[2] - (s3 + 0.5 * c3)).abs() < 1e-6);
        // theta = 1.5 for pair 1: (0.0, 1.5) -> (-1.5 sin1.5, 1.5 cos1.5).
        let (s15, c15) = 1.5f32.sin_cos();
        assert!((x[1] - (-1.5 * s15)).abs() < 1e-6);
        assert!((x[3] - (1.5 * c15)).abs() < 1e-6);
    }

    #[test]
    fn silu_is_bounded_and_monotone_ish() {
        assert!((silu(0.0) - 0.0).abs() < 1e-8);
        assert!(silu(10.0) > 9.9);
        assert!(silu(-10.0) > -0.001 && silu(-10.0) < 0.0);
    }

    #[test]
    fn forward_produces_vocab_sized_logits_deterministically() {
        let f = fixture(BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 4 << 20,
        });
        let fwd = Forward {
            loader: &f.loader,
            tracker: &f.tracker,
            weights: &f.weights,
            profile: NumericalProfile::OracleF32,
            kernels: ResolvedKernels::default_for_target(),
            control: &|| {},
            counters: None,
        };
        let tokens = [1u32, 2, 3];
        let a = fwd.logits_at(&tokens, 2, &|| false).unwrap();
        let b = fwd.logits_at(&tokens, 2, &|| false).unwrap();
        assert_eq!(a.len(), f.weights.cfg.vocab as usize);
        assert_eq!(a, b, "deterministic forward pass");
        assert!(a.iter().all(|v| v.is_finite()));
        // KV state was populated and leases were released.
        assert_eq!(f.tracker.in_flight(), 0);
        assert_eq!(f.loader.stats().pinned_blocks, 0);
    }

    #[test]
    fn streaming_and_resident_paths_produce_identical_logits() {
        let resident = fixture(BlockPolicy {
            block_bytes: 1 << 20,
            cache_bytes: 32 << 20,
        });
        let streamed = fixture(BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 64 * 1024, // one block: constant eviction
        });
        let tokens = [7u32, 11, 13, 17];
        let a = Forward {
            loader: &resident.loader,
            tracker: &resident.tracker,
            weights: &resident.weights,
            profile: NumericalProfile::OracleF32,
            kernels: ResolvedKernels::default_for_target(),
            control: &|| {},
            counters: None,
        }
        .logits_at(&tokens, 3, &|| false)
        .unwrap();
        let b = Forward {
            loader: &streamed.loader,
            tracker: &streamed.tracker,
            weights: &streamed.weights,
            profile: NumericalProfile::OracleF32,
            kernels: ResolvedKernels::default_for_target(),
            control: &|| {},
            counters: None,
        }
        .logits_at(&tokens, 3, &|| false)
        .unwrap();
        assert_eq!(a, b, "scheduling must not change the numbers");
        let stats = streamed.loader.stats();
        assert!(
            stats.cache_entry_evictions > 0,
            "streamed run must exercise eviction: {stats:?}"
        );
        assert!(
            resident.loader.stats().cache_entry_evictions == 0
                || resident.loader.stats().blocks_loaded > 0
        );
        assert_eq!(streamed.tracker.in_flight(), 0);
    }

    #[test]
    fn cancellation_is_honored_inside_the_forward_pass() {
        let f = fixture(BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 1 << 20,
        });
        let fwd = Forward {
            loader: &f.loader,
            tracker: &f.tracker,
            weights: &f.weights,
            profile: NumericalProfile::OracleF32,
            kernels: ResolvedKernels::default_for_target(),
            control: &|| {},
            counters: None,
        };
        let err = fwd.logits_at(&[1, 2, 3], 2, &|| true).unwrap_err();
        assert_eq!(err.code(), ErrorCode::Cancelled);
        assert_eq!(f.tracker.in_flight(), 0, "no leases leak on cancellation");
    }

    #[test]
    fn llama_cpu_profile_stays_close_to_the_oracle_on_the_fixture() {
        let f = fixture(BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 4 << 20,
        });
        let tokens = [5u32, 9, 13];
        let oracle = Forward {
            loader: &f.loader,
            tracker: &f.tracker,
            weights: &f.weights,
            profile: NumericalProfile::OracleF32,
            kernels: ResolvedKernels::default_for_target(),
            control: &|| {},
            counters: None,
        }
        .logits_at(&tokens, 2, &|| false)
        .unwrap();
        let llama = Forward {
            loader: &f.loader,
            tracker: &f.tracker,
            weights: &f.weights,
            profile: NumericalProfile::LlamaCpu,
            kernels: ResolvedKernels::default_for_target(),
            control: &|| {},
            counters: None,
        }
        .logits_at(&tokens, 2, &|| false)
        .unwrap();
        assert_eq!(oracle.len(), llama.len());
        let max_diff = oracle
            .iter()
            .zip(llama.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.05,
            "profiles must agree on the fixture: {max_diff}"
        );
        assert_eq!(f.tracker.in_flight(), 0);
    }

    #[test]
    fn out_of_vocab_tokens_are_rejected() {
        let f = fixture(BlockPolicy::default());
        let fwd = Forward {
            loader: &f.loader,
            tracker: &f.tracker,
            weights: &f.weights,
            profile: NumericalProfile::OracleF32,
            kernels: ResolvedKernels::default_for_target(),
            control: &|| {},
            counters: None,
        };
        let err = fwd.logits_at(&[999_999], 0, &|| false).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidRequest);
    }

    #[test]
    fn scoring_position_bounds_are_checked() {
        let f = fixture(BlockPolicy::default());
        let fwd = Forward {
            loader: &f.loader,
            tracker: &f.tracker,
            weights: &f.weights,
            profile: NumericalProfile::OracleF32,
            kernels: ResolvedKernels::default_for_target(),
            control: &|| {},
            counters: None,
        };
        assert!(fwd.logits_at(&[1, 2], 5, &|| false).is_err());
        assert!(fwd.logits_at(&[], 0, &|| false).is_err());
    }
}
