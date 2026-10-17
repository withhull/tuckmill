//! GPU Gated DeltaNet operators (Q35-2A, ADR-008).
//!
//! Isolated Metal kernels for the Qwen3.5 linear-attention operators —
//! causal convolution (GGUF order, `(kernel - 1)`-row history), per-head
//! L2 norm, beta/decay gates, the f32 delta-rule recurrence and the gated
//! output normalization — exercised against the scalar reference in
//! [`crate::qwen35_ops`] across fixture and pinned shapes/sequences. The
//! operator signatures mirror the CPU ones, so later milestones can swap
//! execution paths without changing the contract.
//!
//! The recurrence is intentionally one thread per head for now
//! (correct-first; per-head arithmetic order matches the CPU rule) and is
//! numerically qualified before any prefill batching work.

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_metal::{probe, MetalBuffer, MetalContext};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Shared GPU owner for the Qwen3.5 execution paths: ONE Metal context
/// (device/queue) for the whole model, with every operator buffer
/// allocated through it — a single authoritative allocation ledger
/// (`allocated_bytes`). All `*_dev` methods are buffer-to-buffer: model
/// activations stay on the device; the slice-based adapters on
/// [`GdnGpu`] exist only for operator-level tests.
#[derive(Clone)]
pub struct Qwen35Gpu {
    ctx: MetalContext,
}

impl Qwen35Gpu {
    pub fn open() -> Result<Self> {
        let probe = probe();
        if !probe.available {
            return Err(TuckmillError::new(
                ErrorCode::Unavailable,
                format!("no Metal device is available ({})", probe.info),
            ));
        }
        Ok(Self {
            ctx: MetalContext::open()?,
        })
    }

    pub fn context(&self) -> &MetalContext {
        &self.ctx
    }

    /// Allocate a shared-storage buffer of `bytes` (zeroed explicitly:
    /// no inference result may depend on unspecified initial contents).
    pub fn buffer(&self, bytes: usize) -> Result<MetalBuffer> {
        let buffer = self.ctx.buffer(bytes, true)?;
        buffer.upload(0, &vec![0u8; bytes])?;
        Ok(buffer)
    }

    /// Buffer-to-buffer GDN operators (GdnGpu's methods delegate here;
    /// the model paths call these directly).
    pub fn conv_step_dev(
        &self,
        hist: &MetalBuffer,
        x: &MetalBuffer,
        w: &MetalBuffer,
        y: &MetalBuffer,
        channels: u32,
        kernel: u32,
    ) -> Result<()> {
        self.ctx.begin_group()?;
        let result = (|| {
            self.ctx.qwen35_conv_step(hist, x, w, y, channels, kernel)?;
            self.ctx.qwen35_conv_shift(hist, x, channels, kernel)
        })();
        match result {
            Ok(()) => self.ctx.end_group(),
            Err(e) => {
                let _ = self.ctx.abort_group();
                Err(e)
            }
        }
    }

    pub fn l2norm_dev(
        &self,
        x: &MetalBuffer,
        y: &MetalBuffer,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> Result<()> {
        self.ctx.qwen35_l2norm_heads(x, y, rows, cols, eps)
    }

    #[allow(clippy::too_many_arguments)] // mirrors the kernel arity
    pub fn beta_decay_dev(
        &self,
        raw_beta: &MetalBuffer,
        alpha: &MetalBuffer,
        dt_bias: &MetalBuffer,
        ssm_a: &MetalBuffer,
        beta_out: &MetalBuffer,
        gate_out: &MetalBuffer,
        n: u32,
    ) -> Result<()> {
        self.ctx
            .qwen35_beta_decay(raw_beta, alpha, dt_bias, ssm_a, beta_out, gate_out, n)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_step_dev(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        gate: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        out: &MetalBuffer,
        heads: u32,
        k_dim: u32,
        v_dim: u32,
        q_scale: f32,
    ) -> Result<()> {
        self.ctx.qwen35_gdn_step(
            q, k, v, gate, beta, state, out, heads, k_dim, v_dim, q_scale,
        )
    }

    #[allow(clippy::too_many_arguments)] // mirrors the kernel arity
    pub fn gated_norm_dev(
        &self,
        o: &MetalBuffer,
        z: &MetalBuffer,
        weight: &MetalBuffer,
        out: &MetalBuffer,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> Result<()> {
        self.ctx
            .qwen35_gated_norm(o, z, weight, out, rows, cols, eps)
    }

    // --- Q35-2B full-attention operators ---

    #[allow(clippy::too_many_arguments)]
    pub fn q_gate_split_dev(
        &self,
        qg: &MetalBuffer,
        q_out: &MetalBuffer,
        gate_out: &MetalBuffer,
        heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        self.ctx
            .qwen35_q_gate_split(qg, q_out, gate_out, heads, head_dim)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_partial_dev(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        inv_freq: &MetalBuffer,
        pos: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> Result<()> {
        self.ctx
            .qwen35_rope_partial(q, k, inv_freq, pos, n_q_heads, n_kv_heads, head_dim, n_rot)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_store_f16_dev(
        &self,
        k_src: &MetalBuffer,
        v_src: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        pos: u32,
        cap: u32,
        head_dim: u32,
        kv_heads: u32,
    ) -> Result<()> {
        self.ctx
            .qwen35_kv_store_f16(k_src, v_src, k_cache, v_cache, pos, cap, head_dim, kv_heads)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_f16_dev(
        &self,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        out: &MetalBuffer,
        scratch: &MetalBuffer,
        heads: u32,
        pos: u32,
        cap: u32,
        head_dim: u32,
        group: u32,
        scale: f32,
    ) -> Result<()> {
        self.ctx.qwen35_attention_f16(
            q, k_cache, v_cache, out, scratch, heads, pos, cap, head_dim, group, scale,
        )
    }

    pub fn attn_gate_dev(&self, attn: &MetalBuffer, gate: &MetalBuffer, n: u32) -> Result<()> {
        self.ctx.qwen35_attn_gate(attn, gate, n)
    }
}

/// Device-resident GDN operator set for one layer, with slice-based
/// convenience methods used by the operator-level tests. Interior state
/// buffers are zeroed at construction (no result may depend on
/// unspecified initial GPU contents).
pub struct GdnGpu {
    gpu: Qwen35Gpu,
    heads: usize,
    k_dim: usize,
    v_dim: usize,
    conv_kernel: usize,
    conv_channels: usize,
    scratch_len: usize,
    x_in: MetalBuffer,
    y_out: MetalBuffer,
    q_in: MetalBuffer,
    k_in: MetalBuffer,
    v_in: MetalBuffer,
    beta_in: MetalBuffer,
    alpha_in: MetalBuffer,
    dt_in: MetalBuffer,
    a_in: MetalBuffer,
    norm_w: MetalBuffer,
    beta_gpu: MetalBuffer,
    gate_gpu: MetalBuffer,
    w_conv: MetalBuffer,
    state: MetalBuffer,
    hist: MetalBuffer,
}

impl GdnGpu {
    pub fn new(
        heads: usize,
        k_dim: usize,
        v_dim: usize,
        conv_kernel: usize,
        conv_channels: usize,
    ) -> Result<Self> {
        Self::new_with(
            &Qwen35Gpu::open()?,
            heads,
            k_dim,
            v_dim,
            conv_kernel,
            conv_channels,
        )
    }

    /// Build on a shared model-wide GPU owner (one device/queue for every
    /// layer; buffers join the same allocation ledger).
    pub fn new_with(
        gpu: &Qwen35Gpu,
        heads: usize,
        k_dim: usize,
        v_dim: usize,
        conv_kernel: usize,
        conv_channels: usize,
    ) -> Result<Self> {
        let buf = |elems: usize| -> Result<MetalBuffer> { gpu.buffer(elems * 4) };
        let scratch_len = conv_channels.max(heads * k_dim).max(heads * v_dim);
        Ok(Self {
            x_in: buf(scratch_len)?,
            y_out: buf(scratch_len)?,
            q_in: buf(heads * k_dim)?,
            k_in: buf(heads * k_dim)?,
            v_in: buf(heads * v_dim)?,
            beta_in: buf(heads)?,
            alpha_in: buf(heads)?,
            dt_in: buf(heads)?,
            a_in: buf(heads)?,
            norm_w: buf(v_dim)?,
            beta_gpu: buf(heads)?,
            gate_gpu: buf(heads)?,
            w_conv: buf(conv_kernel * conv_channels)?,
            state: buf(heads * k_dim * v_dim)?,
            hist: buf((conv_kernel - 1).max(1) * conv_channels)?,
            gpu: gpu.clone(),
            heads,
            k_dim,
            v_dim,
            conv_kernel,
            conv_channels,
            scratch_len,
        })
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    /// The delta rule's internal query scale (`1 / sqrt(k_dim)`).
    pub fn q_scale(&self) -> f32 {
        1.0 / (self.k_dim as f32).sqrt()
    }

    /// Conv weights in GGUF memory order (`ssm_conv1d`, taps fastest).
    pub fn upload_conv_weights(&self, w: &[f32]) -> Result<()> {
        if w.len() != self.conv_kernel * self.conv_channels {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!(
                    "conv weights have {} values, expected {}",
                    w.len(),
                    self.conv_kernel * self.conv_channels
                ),
            ));
        }
        self.w_conv.upload(0, &f32_bytes(w))
    }

    /// One decode step of the causal convolution plus history advance.
    pub fn conv_step(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != self.conv_channels || out.len() != self.conv_channels {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "conv_step row size must equal the channel count",
            ));
        }
        self.x_in.upload(0, &f32_bytes(x))?;
        self.gpu.conv_step_dev(
            &self.hist,
            &self.x_in,
            &self.w_conv,
            &self.y_out,
            self.conv_channels as u32,
            self.conv_kernel as u32,
        )?;
        let mut bytes = vec![0u8; out.len() * 4];
        self.y_out.download(0, &mut bytes)?;
        out.copy_from_slice(&bytes_f32(&bytes));
        Ok(())
    }

    /// Per-head L2 norm over `rows x cols` (cols = head dim).
    pub fn l2norm_heads(&self, x: &[f32], cols: usize, out: &mut [f32], eps: f32) -> Result<()> {
        if cols == 0 || x.len() != out.len() || !x.len().is_multiple_of(cols) {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "l2norm_heads operands must be whole rows",
            ));
        }
        let bytes = f32_bytes(x);
        self.x_in.upload(0, &bytes)?;
        self.gpu.l2norm_dev(
            &self.x_in,
            &self.y_out,
            (x.len() / cols) as u32,
            cols as u32,
            eps,
        )?;
        let mut out_bytes = vec![0u8; out.len() * 4];
        self.y_out.download(0, &mut out_bytes)?;
        out.copy_from_slice(&bytes_f32(&out_bytes));
        Ok(())
    }

    /// Per-head gates: `beta = sigmoid(raw_beta)`;
    /// `gate = ssm_a * softplus(alpha + dt_bias)`.
    pub fn beta_decay(
        &self,
        raw_beta: &[f32],
        alpha: &[f32],
        dt_bias: &[f32],
        ssm_a: &[f32],
        beta_out: &mut [f32],
        gate_out: &mut [f32],
    ) -> Result<()> {
        let n = raw_beta.len();
        if alpha.len() != n || dt_bias.len() != n || ssm_a.len() != n {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "beta_decay operands must share the head count",
            ));
        }
        if beta_out.len() != n || gate_out.len() != n {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "beta_decay outputs must match the head count",
            ));
        }
        self.beta_in.upload(0, &f32_bytes(raw_beta))?;
        self.alpha_in.upload(0, &f32_bytes(alpha))?;
        self.dt_in.upload(0, &f32_bytes(dt_bias))?;
        self.a_in.upload(0, &f32_bytes(ssm_a))?;
        self.gpu.beta_decay_dev(
            &self.beta_in,
            &self.alpha_in,
            &self.dt_in,
            &self.a_in,
            &self.beta_gpu,
            &self.gate_gpu,
            n as u32,
        )?;
        let mut b = vec![0u8; n * 4];
        self.beta_gpu.download(0, &mut b)?;
        beta_out.copy_from_slice(&bytes_f32(&b));
        let mut g = vec![0u8; n * 4];
        self.gate_gpu.download(0, &mut g)?;
        gate_out.copy_from_slice(&bytes_f32(&g));
        Ok(())
    }

    /// One token of the delta rule; the device state is updated in place.
    pub fn gdn_step(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if q.len() != self.heads * self.k_dim
            || k.len() != self.heads * self.k_dim
            || v.len() != self.heads * self.v_dim
            || out.len() != self.heads * self.v_dim
        {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "gdn_step operand geometry must match the operator",
            ));
        }
        if gate.len() != self.heads || beta.len() != self.heads {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "gdn_step gates must be per head",
            ));
        }
        self.q_in.upload(0, &f32_bytes(q))?;
        self.k_in.upload(0, &f32_bytes(k))?;
        self.v_in.upload(0, &f32_bytes(v))?;
        self.beta_in.upload(0, &f32_bytes(beta))?;
        self.a_in.upload(0, &f32_bytes(gate))?;
        self.gpu.gdn_step_dev(
            &self.q_in,
            &self.k_in,
            &self.v_in,
            &self.a_in,
            &self.beta_in,
            &self.state,
            &self.y_out,
            self.heads as u32,
            self.k_dim as u32,
            self.v_dim as u32,
            self.q_scale(),
        )?;
        let mut bytes = vec![0u8; out.len() * 4];
        self.y_out.download(0, &mut bytes)?;
        out.copy_from_slice(&bytes_f32(&bytes));
        Ok(())
    }

    /// Gated output normalization: per head, RMS norm over `cols` with
    /// `weight`, times SiLU of the z row.
    pub fn gated_norm(
        &self,
        o: &[f32],
        z: &[f32],
        weight: &[f32],
        cols: usize,
        out: &mut [f32],
        eps: f32,
    ) -> Result<()> {
        if cols == 0
            || o.len() != z.len()
            || o.len() != out.len()
            || weight.len() != cols
            || !o.len().is_multiple_of(cols)
        {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "gated_norm operands must be whole rows with a full weight row",
            ));
        }
        let rows = o.len() / cols;
        self.x_in.upload(0, &f32_bytes(o))?;
        self.v_in.upload(0, &f32_bytes(z))?;
        self.norm_w.upload(0, &f32_bytes(weight))?;
        self.gpu.gated_norm_dev(
            &self.x_in,
            &self.v_in,
            &self.norm_w,
            &self.y_out,
            rows as u32,
            cols as u32,
            eps,
        )?;
        let mut bytes = vec![0u8; out.len() * 4];
        self.y_out.download(0, &mut bytes)?;
        out.copy_from_slice(&bytes_f32(&bytes));
        Ok(())
    }

    /// Zero the delta-rule state and the convolution history.
    pub fn reset(&self) -> Result<()> {
        self.state
            .upload(0, &vec![0u8; self.heads * self.k_dim * self.v_dim * 4])?;
        self.hist.upload(
            0,
            &vec![0u8; (self.conv_kernel - 1).max(1) * self.conv_channels * 4],
        )
    }

    /// Device delta-rule state snapshot (`heads x k_dim x v_dim`, i-major).
    pub fn state_snapshot(&self) -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; self.heads * self.k_dim * self.v_dim * 4];
        self.state.download(0, &mut bytes)?;
        Ok(bytes_f32(&bytes))
    }

    /// Device convolution history snapshot (`(kernel - 1) x channels`).
    pub fn conv_history(&self) -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; (self.conv_kernel - 1).max(1) * self.conv_channels * 4];
        self.hist.download(0, &mut bytes)?;
        let _ = self.scratch_len;
        Ok(bytes_f32(&bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35_ops::{
        l2_normalize_head, rms_norm_weighted_in_place, sigmoid, silu, softplus, ConvState, GdnState,
    };

    /// Deterministic pseudo-random values in [-1, 1).
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn assert_close(got: f32, want: f32, abs: f32, rel: f32, what: &str) {
        let tol = abs + rel * want.abs();
        assert!(
            (got - want).abs() <= tol,
            "{what}: {got} vs {want} (tol {tol})"
        );
    }

    /// Registered operator tolerance (Metal vs scalar). Observed max
    /// deviations on the fixture and pinned shapes (6-token conv
    /// sequences, 3-5-token delta sequences): conv 2.4e-7, L2 1.5e-8,
    /// gated norm 1.2e-7, delta fixture 1.2e-7 / pinned 16x128 2.4e-4 in
    /// the output (9.2e-5 in state). The bounds below carry headroom and
    /// are registered in the certification matrix.
    const ABS_TOL: f32 = 1e-4;
    const REL_TOL: f32 = 2e-4;

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

    /// (heads, k_dim, v_dim, conv_kernel, conv_channels): the fixture
    /// shape and the pinned 0.8B shape.
    const SHAPES: [(usize, usize, usize, usize, usize); 2] =
        [(2, 8, 8, 4, 48), (16, 128, 128, 4, 6144)];

    #[test]
    fn conv_matches_cpu_across_sequences() {
        if gpu_or_skip().is_none() {
            return;
        }
        let mut seed = 0xC0FFEEu64;
        for (_, _, _, kernel, channels) in SHAPES {
            let gpu = GdnGpu::new(2, 8, 8, kernel, channels).unwrap();
            let w: Vec<f32> = (0..kernel * channels).map(|_| lcg(&mut seed)).collect();
            gpu.upload_conv_weights(&w).unwrap();
            let mut cpu = ConvState::new(kernel, channels);
            let mut max_diff = 0.0f32;
            for _ in 0..6 {
                let x: Vec<f32> = (0..channels).map(|_| lcg(&mut seed)).collect();
                let mut cpu_out = vec![0.0f32; channels];
                cpu.step_sequence(&w, &x, &mut cpu_out);
                let mut gpu_out = vec![0.0f32; channels];
                gpu.conv_step(&x, &mut gpu_out).unwrap();
                for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
                    max_diff = max_diff.max((g - c).abs());
                    assert_close(*g, *c, ABS_TOL, REL_TOL, "conv");
                }
            }
            let history = gpu.conv_history().unwrap();
            for (g, c) in history.iter().zip(cpu.history().iter()) {
                assert_close(*g, *c, ABS_TOL, REL_TOL, "conv history");
            }
            eprintln!("conv k={kernel} ch={channels}: max|diff| {max_diff:.3e}");
        }
    }

    #[test]
    fn l2norm_matches_cpu_across_shapes() {
        if gpu_or_skip().is_none() {
            return;
        }
        let mut seed = 0x5EEDu64;
        for (heads, k_dim, v_dim, kernel, channels) in SHAPES {
            let gpu = GdnGpu::new(heads, k_dim, v_dim, kernel, channels).unwrap();
            let x: Vec<f32> = (0..heads * k_dim).map(|_| lcg(&mut seed)).collect();
            let mut cpu = x.clone();
            for head in 0..heads {
                l2_normalize_head(&mut cpu[head * k_dim..(head + 1) * k_dim], 1e-6);
            }
            let mut gpu_out = vec![0.0f32; x.len()];
            gpu.l2norm_heads(&x, k_dim, &mut gpu_out, 1e-6).unwrap();
            let mut max_diff = 0.0f32;
            for (g, c) in gpu_out.iter().zip(cpu.iter()) {
                max_diff = max_diff.max((g - c).abs());
                assert_close(*g, *c, ABS_TOL, REL_TOL, "l2norm");
            }
            eprintln!("l2norm heads={heads} k={k_dim}: max|diff| {max_diff:.3e}");
        }
    }

    #[test]
    fn beta_decay_matches_cpu() {
        if gpu_or_skip().is_none() {
            return;
        }
        let mut seed = 0xBEEFu64;
        for (heads, k_dim, v_dim, kernel, channels) in SHAPES {
            let gpu = GdnGpu::new(heads, k_dim, v_dim, kernel, channels).unwrap();
            let raw_beta: Vec<f32> = (0..heads).map(|_| 2.0 * lcg(&mut seed)).collect();
            let alpha: Vec<f32> = (0..heads).map(|_| 2.0 * lcg(&mut seed)).collect();
            let dt: Vec<f32> = (0..heads).map(|_| lcg(&mut seed)).collect();
            let a: Vec<f32> = (0..heads).map(|_| -0.5 - lcg(&mut seed).abs()).collect();
            let mut beta = vec![0.0f32; heads];
            let mut gate = vec![0.0f32; heads];
            gpu.beta_decay(&raw_beta, &alpha, &dt, &a, &mut beta, &mut gate)
                .unwrap();
            for h in 0..heads {
                assert_close(beta[h], sigmoid(raw_beta[h]), 1e-6, 1e-6, "sigmoid");
                let want = a[h] * softplus(alpha[h] + dt[h]);
                assert_close(gate[h], want, ABS_TOL, REL_TOL, "gate");
            }
        }
    }

    #[test]
    fn delta_rule_matches_cpu_across_sequences() {
        if gpu_or_skip().is_none() {
            return;
        }
        let mut seed = 0xD00Du64;
        for (heads, k_dim, v_dim, kernel, channels) in SHAPES {
            let gpu = GdnGpu::new(heads, k_dim, v_dim, kernel, channels).unwrap();
            gpu.reset().unwrap();
            let mut cpu = GdnState::new(heads, k_dim, v_dim);
            let tokens = if k_dim > 32 { 3 } else { 5 };
            let mut max_out = 0.0f32;
            let mut max_state = 0.0f32;
            for _ in 0..tokens {
                let q: Vec<f32> = (0..heads * k_dim).map(|_| lcg(&mut seed)).collect();
                let k: Vec<f32> = (0..heads * k_dim).map(|_| lcg(&mut seed)).collect();
                let v: Vec<f32> = (0..heads * v_dim).map(|_| lcg(&mut seed)).collect();
                let gate: Vec<f32> = (0..heads).map(|_| -0.2 - lcg(&mut seed).abs()).collect();
                let beta: Vec<f32> = (0..heads)
                    .map(|_| 0.5 + 0.5 * lcg(&mut seed).abs())
                    .collect();

                let mut cpu_out = vec![0.0f32; heads * v_dim];
                cpu.step(&q, &k, &v, &gate, &beta, &mut cpu_out);
                let mut gpu_out = vec![0.0f32; heads * v_dim];
                gpu.gdn_step(&q, &k, &v, &gate, &beta, &mut gpu_out)
                    .unwrap();

                for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
                    max_out = max_out.max((g - c).abs());
                    assert_close(*g, *c, ABS_TOL, REL_TOL, "delta out");
                }
                let gpu_state = gpu.state_snapshot().unwrap();
                for (g, c) in gpu_state.iter().zip(cpu.snapshot().iter()) {
                    max_state = max_state.max((g - c).abs());
                    assert_close(*g, *c, ABS_TOL, REL_TOL, "delta state");
                }
            }
            eprintln!(
                "delta heads={heads} k={k_dim}: max|out diff| {max_out:.3e}, max|state diff| {max_state:.3e}"
            );
        }
    }

    #[test]
    fn gated_norm_matches_cpu_across_shapes() {
        if gpu_or_skip().is_none() {
            return;
        }
        let mut seed = 0xF00Du64;
        for (heads, _, v_dim, kernel, channels) in SHAPES {
            let gpu = GdnGpu::new(heads, 8, v_dim, kernel, channels).unwrap();
            let o: Vec<f32> = (0..heads * v_dim).map(|_| lcg(&mut seed)).collect();
            let z: Vec<f32> = (0..heads * v_dim).map(|_| 2.0 * lcg(&mut seed)).collect();
            let weight: Vec<f32> = (0..v_dim).map(|_| 1.0 + 0.5 * lcg(&mut seed)).collect();

            let mut cpu = vec![0.0f32; o.len()];
            for head in 0..heads {
                cpu[head * v_dim..(head + 1) * v_dim]
                    .copy_from_slice(&o[head * v_dim..(head + 1) * v_dim]);
                rms_norm_weighted_in_place(
                    &mut cpu[head * v_dim..(head + 1) * v_dim],
                    &weight,
                    1e-6,
                );
                for i in 0..v_dim {
                    cpu[head * v_dim + i] *= silu(z[head * v_dim + i]);
                }
            }
            let mut gpu_out = vec![0.0f32; o.len()];
            gpu.gated_norm(&o, &z, &weight, v_dim, &mut gpu_out, 1e-6)
                .unwrap();
            let mut max_diff = 0.0f32;
            for (g, c) in gpu_out.iter().zip(cpu.iter()) {
                max_diff = max_diff.max((g - c).abs());
                assert_close(*g, *c, ABS_TOL, REL_TOL, "gated_norm");
            }
            eprintln!("gated_norm heads={heads} v={v_dim}: max|diff| {max_diff:.3e}");
        }
    }

    #[test]
    fn reset_clears_device_state() {
        if gpu_or_skip().is_none() {
            return;
        }
        let gpu = GdnGpu::new(2, 8, 8, 4, 48).unwrap();
        let mut seed = 0x1234u64;
        let q: Vec<f32> = (0..2 * 8).map(|_| lcg(&mut seed)).collect();
        let v: Vec<f32> = (0..2 * 8).map(|_| lcg(&mut seed)).collect();
        let gate = [-0.5f32, -0.1];
        let beta = [0.5f32, 0.5];
        let mut out = vec![0.0f32; 2 * 8];
        gpu.gdn_step(&q, &q, &v, &gate, &beta, &mut out).unwrap();
        assert!(gpu.state_snapshot().unwrap().iter().any(|s| *s != 0.0));
        gpu.reset().unwrap();
        assert!(gpu.state_snapshot().unwrap().iter().all(|s| *s == 0.0));
        assert!(gpu.conv_history().unwrap().iter().all(|s| *s == 0.0));
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use crate::qwen35_kv::{KvPrecision, Qwen35KvCache};
    use crate::qwen35_ops::{
        apply_sigmoid_gate, gqa_cache_attention, partial_rope_in_place, rope_inv_freq_partial,
        sigmoid, GqaGeometry,
    };
    use crate::tensor::f32_to_f16_bits;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn assert_close(got: f32, want: f32, abs: f32, rel: f32, what: &str) {
        let tol = abs + rel * want.abs();
        assert!(
            (got - want).abs() <= tol,
            "{what}: {got} vs {want} (tol {tol})"
        );
    }

    const ABS_TOL: f32 = 1e-4;
    const REL_TOL: f32 = 2e-4;

    fn gpu_or_skip_simple() -> Option<()> {
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

    fn upload_f32(gpu: &Qwen35Gpu, data: &[f32]) -> MetalBuffer {
        let buffer = gpu.buffer(data.len() * 4).unwrap();
        buffer.upload(0, &f32_bytes(data)).unwrap();
        buffer
    }

    fn download_f32(buffer: &MetalBuffer, n: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; n * 4];
        buffer.download(0, &mut bytes).unwrap();
        bytes_f32(&bytes)
    }

    /// Fresh construction is explicitly zeroed: state never depends on
    /// unspecified initial GPU contents, even after prior allocations.
    #[test]
    fn fresh_construction_is_zeroed_and_deterministic() {
        if gpu_or_skip_simple().is_none() {
            return;
        }
        let gpu = Qwen35Gpu::open().unwrap();
        // Prior allocation churn in the same context.
        let _noise = gpu.buffer(1 << 16).unwrap();
        let _noise2 = gpu.buffer(1 << 16).unwrap();

        let layer_a = GdnGpu::new_with(&gpu, 2, 8, 8, 4, 48).unwrap();
        let layer_b = GdnGpu::new_with(&gpu, 2, 8, 8, 4, 48).unwrap();
        assert!(layer_a.state_snapshot().unwrap().iter().all(|v| *v == 0.0));
        assert!(layer_a.conv_history().unwrap().iter().all(|v| *v == 0.0));

        let mut seed = 0xABCDu64;
        let q: Vec<f32> = (0..16).map(|_| lcg(&mut seed)).collect();
        let v: Vec<f32> = (0..16).map(|_| lcg(&mut seed)).collect();
        let gate = [-0.3f32, -0.7];
        let beta = [0.6f32, 0.4];
        let mut out_a = vec![0.0f32; 16];
        let mut out_b = vec![0.0f32; 16];
        layer_a
            .gdn_step(&q, &q, &v, &gate, &beta, &mut out_a)
            .unwrap();
        layer_b
            .gdn_step(&q, &q, &v, &gate, &beta, &mut out_b)
            .unwrap();
        assert_eq!(out_a, out_b, "deterministic from explicitly zeroed state");
        assert_eq!(
            layer_a.state_snapshot().unwrap(),
            layer_b.state_snapshot().unwrap()
        );
    }

    #[test]
    fn q_gate_split_matches_cpu() {
        if gpu_or_skip_simple().is_none() {
            return;
        }
        let gpu = Qwen35Gpu::open().unwrap();
        let mut seed = 0x6D5u64;
        for (heads, hd) in [(4usize, 16usize), (8, 256)] {
            let qg: Vec<f32> = (0..heads * 2 * hd).map(|_| lcg(&mut seed)).collect();
            let qg_buf = upload_f32(&gpu, &qg);
            let (q_buf, gate_buf) = (
                gpu.buffer(heads * hd * 4).unwrap(),
                gpu.buffer(heads * hd * 4).unwrap(),
            );
            gpu.q_gate_split_dev(&qg_buf, &q_buf, &gate_buf, heads as u32, hd as u32)
                .unwrap();
            let q = download_f32(&q_buf, heads * hd);
            let gate = download_f32(&gate_buf, heads * hd);
            for h in 0..heads {
                for i in 0..hd {
                    assert_eq!(q[h * hd + i], qg[h * 2 * hd + i], "q element");
                    assert_eq!(gate[h * hd + i], qg[h * 2 * hd + i + hd], "gate element");
                }
            }
        }
    }

    #[test]
    fn partial_rope_matches_cpu() {
        if gpu_or_skip_simple().is_none() {
            return;
        }
        let gpu = Qwen35Gpu::open().unwrap();
        let mut seed = 0x20CEu64;
        // (q_heads, kv_heads, head_dim, n_rot, pos)
        for (hq, hkv, hd, n_rot, pos) in
            [(4usize, 2usize, 16usize, 16usize, 3u32), (8, 2, 256, 64, 5)]
        {
            let inv = rope_inv_freq_partial(n_rot, 1e7);
            let q: Vec<f32> = (0..hq * hd).map(|_| lcg(&mut seed)).collect();
            let k: Vec<f32> = (0..hkv * hd).map(|_| lcg(&mut seed)).collect();
            let mut cpu_q = q.clone();
            let mut cpu_k = k.clone();
            for h in 0..hq {
                partial_rope_in_place(&mut cpu_q[h * hd..(h + 1) * hd], pos as usize, &inv);
            }
            for h in 0..hkv {
                partial_rope_in_place(&mut cpu_k[h * hd..(h + 1) * hd], pos as usize, &inv);
            }

            let q_buf = upload_f32(&gpu, &q);
            let k_buf = upload_f32(&gpu, &k);
            let f_buf = upload_f32(&gpu, &inv);
            gpu.rope_partial_dev(
                &q_buf,
                &k_buf,
                &f_buf,
                pos,
                hq as u32,
                hkv as u32,
                hd as u32,
                n_rot as u32,
            )
            .unwrap();
            let gq = download_f32(&q_buf, hq * hd);
            let gk = download_f32(&k_buf, hkv * hd);
            for (g, c) in gq.iter().zip(cpu_q.iter()) {
                assert_close(*g, *c, ABS_TOL, REL_TOL, "rope q");
            }
            for (g, c) in gk.iter().zip(cpu_k.iter()) {
                assert_close(*g, *c, ABS_TOL, REL_TOL, "rope k");
            }
        }
    }

    #[test]
    fn attn_gate_matches_cpu() {
        if gpu_or_skip_simple().is_none() {
            return;
        }
        let gpu = Qwen35Gpu::open().unwrap();
        let mut seed = 0x9A7Eu64;
        let n = 8 * 256;
        let attn: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
        let gate: Vec<f32> = (0..n).map(|_| 2.0 * lcg(&mut seed)).collect();
        let mut cpu = attn.clone();
        apply_sigmoid_gate(&mut cpu, &gate);

        let attn_buf = upload_f32(&gpu, &attn);
        let gate_buf = upload_f32(&gpu, &gate);
        gpu.attn_gate_dev(&attn_buf, &gate_buf, n as u32).unwrap();
        let got = download_f32(&attn_buf, n);
        for (g, c) in got.iter().zip(cpu.iter()) {
            assert_close(*g, *c, ABS_TOL, REL_TOL, "attn gate");
        }
    }

    /// The GPU stores true two-byte f16 entries; the packed cache bytes
    /// are bit-identical to the CPU packer's round-to-nearest-even bits.
    #[test]
    fn packed_kv_store_matches_cpu_bitwise() {
        if gpu_or_skip_simple().is_none() {
            return;
        }
        let gpu = Qwen35Gpu::open().unwrap();
        let kv_heads = 2usize;
        let hd = 16usize;
        let cap = 4usize;
        let mut seed = 0xF16u64;
        let k: Vec<f32> = (0..kv_heads * hd).map(|_| lcg(&mut seed)).collect();
        let v: Vec<f32> = (0..kv_heads * hd).map(|_| lcg(&mut seed)).collect();

        let k_src = upload_f32(&gpu, &k);
        let v_src = upload_f32(&gpu, &v);
        let k_cache = gpu.buffer(kv_heads * cap * hd * 2).unwrap();
        let v_cache = gpu.buffer(kv_heads * cap * hd * 2).unwrap();
        gpu.kv_store_f16_dev(
            &k_src,
            &v_src,
            &k_cache,
            &v_cache,
            0,
            cap as u32,
            hd as u32,
            kv_heads as u32,
        )
        .unwrap();

        let mut gpu_k_bits = vec![0u8; kv_heads * cap * hd * 2];
        k_cache.download(0, &mut gpu_k_bits).unwrap();
        let mut cpu_k_bits = vec![0u8; kv_heads * cap * hd * 2];
        for h in 0..kv_heads {
            for i in 0..hd {
                let src = h * hd + i;
                let dst = h * cap * hd + i; // pos = 0
                let bits = f32_to_f16_bits(k[src]).to_le_bytes();
                cpu_k_bits[dst * 2] = bits[0];
                cpu_k_bits[dst * 2 + 1] = bits[1];
            }
        }
        assert_eq!(gpu_k_bits, cpu_k_bits, "packed f16 bytes are bit-identical");
    }

    /// Full attention pipeline over the packed cache matches the CPU
    /// reference (packed cache + decoded rows + GQA), token by token.
    #[test]
    fn attention_f16_matches_cpu_across_tokens() {
        if gpu_or_skip_simple().is_none() {
            return;
        }
        let gpu = Qwen35Gpu::open().unwrap();
        let mut seed = 0xA77u64;
        // (heads_q, kv_heads, head_dim, cap, tokens)
        for (hq, hkv, hd, cap, tokens) in
            [(4usize, 2usize, 16usize, 4usize, 3usize), (8, 2, 256, 3, 2)]
        {
            let scale = 1.0 / (hd as f32).sqrt();
            let geometry = GqaGeometry {
                heads_q: hq,
                heads_kv: hkv,
                head_dim: hd,
            };
            let mut cpu_cache = Qwen35KvCache::new(1, hkv, hd, cap, KvPrecision::F16).unwrap();
            let k_cache = gpu.buffer(hkv * cap * hd * 2).unwrap();
            let v_cache = gpu.buffer(hkv * cap * hd * 2).unwrap();
            let scratch = gpu.buffer(hq * cap * 4).unwrap();
            let out_buf = gpu.buffer(hq * hd * 4).unwrap();

            for t in 0..tokens {
                let k: Vec<f32> = (0..hkv * hd).map(|_| lcg(&mut seed)).collect();
                let v: Vec<f32> = (0..hkv * hd).map(|_| lcg(&mut seed)).collect();
                let q: Vec<f32> = (0..hq * hd).map(|_| lcg(&mut seed)).collect();

                cpu_cache.write(0, t, &k, &v).unwrap();
                let k_src = upload_f32(&gpu, &k);
                let v_src = upload_f32(&gpu, &v);
                gpu.kv_store_f16_dev(
                    &k_src, &v_src, &k_cache, &v_cache, t as u32, cap as u32, hd as u32, hkv as u32,
                )
                .unwrap();

                let q_buf = upload_f32(&gpu, &q);
                gpu.attention_f16_dev(
                    &q_buf,
                    &k_cache,
                    &v_cache,
                    &out_buf,
                    &scratch,
                    hq as u32,
                    t as u32,
                    cap as u32,
                    hd as u32,
                    (hq / hkv) as u32,
                    scale,
                )
                .unwrap();
                let gpu_out = download_f32(&out_buf, hq * hd);

                // CPU reference: decode the packed cache and run GQA.
                let seq = t + 1;
                let mut k_slab = vec![0.0f32; hkv * seq * hd];
                let mut v_slab = vec![0.0f32; hkv * seq * hd];
                let mut row = vec![0.0f32; hd];
                for h in 0..hkv {
                    for tt in 0..seq {
                        let base = (h * seq + tt) * hd;
                        cpu_cache.key_row(0, h, tt, &mut row).unwrap();
                        k_slab[base..base + hd].copy_from_slice(&row);
                        cpu_cache.value_row(0, h, tt, &mut row).unwrap();
                        v_slab[base..base + hd].copy_from_slice(&row);
                    }
                }
                let mut cpu_out = vec![0.0f32; hq * hd];
                gqa_cache_attention(&q, &k_slab, &v_slab, geometry, seq, scale, &mut cpu_out)
                    .unwrap();

                for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
                    assert_close(*g, *c, ABS_TOL, REL_TOL, "attention f16");
                }
            }
        }
    }

    /// One GPU owner serves several layer/lens objects: the whole model
    /// shares a device/queue and a single allocation ledger.
    #[test]
    fn shared_context_serves_multiple_layers() {
        if gpu_or_skip_simple().is_none() {
            return;
        }
        let gpu = Qwen35Gpu::open().unwrap();
        let before = gpu.context().allocated_bytes();
        let layer_a = GdnGpu::new_with(&gpu, 2, 8, 8, 4, 48).unwrap();
        let layer_b = GdnGpu::new_with(&gpu, 16, 128, 128, 4, 6144).unwrap();
        let after = gpu.context().allocated_bytes();
        assert!(
            after > before,
            "layer buffers join the shared allocation ledger"
        );

        let mut seed = 0x5A5Eu64;
        let q: Vec<f32> = (0..2 * 8).map(|_| lcg(&mut seed)).collect();
        let v: Vec<f32> = (0..2 * 8).map(|_| lcg(&mut seed)).collect();
        let gate = [-0.5f32, -0.2];
        let beta = [0.5f32, 0.5];
        let mut out_a = vec![0.0f32; 16];
        layer_a
            .gdn_step(&q, &q, &v, &gate, &beta, &mut out_a)
            .unwrap();
        // The second layer is untouched by the first.
        assert!(layer_b.state_snapshot().unwrap().iter().all(|x| *x == 0.0));
    }

    /// Sanity: the sigmoid helper used in the CPU comparisons matches the
    /// operator gate formula (guards the reference, not the kernel).
    #[test]
    fn cpu_sigmoid_reference_matches_operator_gate() {
        assert_close(sigmoid(0.0), 0.5, 1e-7, 1e-7, "sigmoid");
    }
}

// ---------------------------------------------------------------------------
// Q35-2B.3: complete device layers (fixture scale: F16 matrices + F32
// vectors) verified layer-by-layer against the CPU oracle. Q8_0 matrices
// and the real-artifact upload path land with Q35-2C.
// ---------------------------------------------------------------------------

use crate::qwen35::Qwen35Layer;
use crate::qwen35_forward::Qwen35Model;

/// Geometry the device layer dispatch needs, captured from the resolved
/// config at load time (one source of truth for both layer kinds).
#[derive(Debug, Clone, Copy)]
struct LayerGeometry {
    hidden: usize,
    ffn: usize,
    eps: f32,
    channels: usize,
    groups: usize,
    /// GDN delta-rule key/value state size (not the attention head dim).
    state_size: usize,
    head_v: usize,
    dt_rank: usize,
    conv_kernel: usize,
    heads: usize,
    kv_heads: usize,
    /// Full-attention per-head K/V dimension.
    head_dim: usize,
    rope_dims: usize,
}

impl LayerGeometry {
    fn from_model(model: &Qwen35Model) -> Self {
        let cfg = model.config();
        Self {
            hidden: cfg.hidden as usize,
            ffn: cfg.ffn as usize,
            eps: cfg.rms_eps,
            channels: cfg.conv_channels() as usize,
            groups: cfg.gdn.groups as usize,
            state_size: cfg.gdn.state_size as usize,
            head_v: cfg.gdn.head_v as usize,
            dt_rank: cfg.gdn.dt_rank as usize,
            conv_kernel: cfg.gdn.conv_kernel as usize,
            heads: cfg.heads as usize,
            kv_heads: cfg.kv_heads as usize,
            head_dim: cfg.head_dim as usize,
            rope_dims: cfg.rope_dims as usize,
        }
    }
}

/// One uploaded weight tensor; matrices stay raw (F16 halves are
/// uploaded verbatim) and 1-D vectors are raw f32.
pub struct DeviceTensor {
    buf: MetalBuffer,
    in_dim: u64,
    out_dim: u64,
}

impl DeviceTensor {
    fn load(gpu: &Qwen35Gpu, model: &Qwen35Model, index: usize) -> Result<Self> {
        let bytes = model.raw_tensor_bytes(index)?;
        let type_id = model.tensor_type_id(index)?;
        let (in_dim, out_dim) = model.tensor_dims(index)?;
        if out_dim <= 1 {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!("tensor {index} is 1-D; vectors use the f32 path"),
            ));
        }
        if type_id != 1 {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedQuantization,
                format!(
                    "device layer assembly currently wires F16 matrices only; tensor {index} is type {type_id} (Q8_0 lands with Q35-2C)"
                ),
            ));
        }
        let buf = gpu.buffer(bytes.len())?;
        buf.upload(0, &bytes)?;
        Ok(Self {
            buf,
            in_dim,
            out_dim,
        })
    }

    fn gemv(&self, gpu: &Qwen35Gpu, x: &MetalBuffer, y: &MetalBuffer) -> Result<()> {
        gpu.context()
            .f16_gemv(&self.buf, x, y, self.out_dim as u32, self.in_dim as u32)
    }
}

fn vec_buf(gpu: &Qwen35Gpu, values: &[f32]) -> Result<MetalBuffer> {
    let buf = gpu.buffer(values.len().max(1) * 4)?;
    buf.upload(0, &f32_bytes(values))?;
    Ok(buf)
}

fn zero_buf(gpu: &Qwen35Gpu, elems: usize) -> Result<MetalBuffer> {
    gpu.buffer(elems.max(1) * 4)
}

/// Norms, FFN and residual stage shared by both layer kinds.
struct LayerCommon {
    geom: LayerGeometry,
    attn_norm: MetalBuffer,
    post_norm: MetalBuffer,
    h: MetalBuffer,
    h2: MetalBuffer,
    ffn_gate: DeviceTensor,
    ffn_up: DeviceTensor,
    ffn_down: DeviceTensor,
    ffn_g: MetalBuffer,
    ffn_u: MetalBuffer,
    ffn_d: MetalBuffer,
    proj: MetalBuffer,
}

impl LayerCommon {
    fn load(gpu: &Qwen35Gpu, model: &Qwen35Model, layer: &Qwen35Layer) -> Result<Self> {
        let geom = LayerGeometry::from_model(model);
        Ok(Self {
            attn_norm: vec_buf(gpu, &model.read_vector(layer.attn_norm.index)?)?,
            post_norm: vec_buf(gpu, &model.read_vector(layer.post_attn_norm.index)?)?,
            h: zero_buf(gpu, geom.hidden)?,
            h2: zero_buf(gpu, geom.hidden)?,
            ffn_gate: DeviceTensor::load(gpu, model, layer.ffn_gate.index)?,
            ffn_up: DeviceTensor::load(gpu, model, layer.ffn_up.index)?,
            ffn_down: DeviceTensor::load(gpu, model, layer.ffn_down.index)?,
            ffn_g: zero_buf(gpu, geom.ffn)?,
            ffn_u: zero_buf(gpu, geom.ffn)?,
            ffn_d: zero_buf(gpu, geom.hidden)?,
            proj: zero_buf(gpu, geom.hidden)?,
            geom,
        })
    }

    /// Post-attention norm, SwiGLU FFN, residual add on `x`.
    fn ffn_residual(&self, gpu: &Qwen35Gpu, x: &MetalBuffer) -> Result<()> {
        let ctx = gpu.context();
        ctx.rmsnorm(
            x,
            &self.post_norm,
            &self.h2,
            1,
            self.geom.hidden as u32,
            self.geom.eps,
        )?;
        self.ffn_gate.gemv(gpu, &self.h2, &self.ffn_g)?;
        self.ffn_up.gemv(gpu, &self.h2, &self.ffn_u)?;
        // SwiGLU in place: `out` aliases `gate`; the kernel reads each
        // element before writing it.
        ctx.silu_mul(&self.ffn_g, &self.ffn_u, &self.ffn_g, self.geom.ffn as u32)?;
        self.ffn_down.gemv(gpu, &self.ffn_g, &self.ffn_d)?;
        ctx.f32_axpy(&self.ffn_d, x, 1.0, self.geom.hidden as u32)
    }
}

pub struct GdnLayerGpu {
    common: LayerCommon,
    qkv: DeviceTensor,
    z: DeviceTensor,
    alpha: DeviceTensor,
    beta: DeviceTensor,
    out_proj: DeviceTensor,
    ssm_a: MetalBuffer,
    ssm_dt: MetalBuffer,
    ssm_norm: MetalBuffer,
    conv_w: MetalBuffer,
    state: MetalBuffer,
    hist: MetalBuffer,
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

pub struct AttentionLayerGpu {
    common: LayerCommon,
    q_w: DeviceTensor,
    k_w: DeviceTensor,
    v_w: DeviceTensor,
    o_w: DeviceTensor,
    q_norm: MetalBuffer,
    k_norm: MetalBuffer,
    inv_freq: MetalBuffer,
    k_cache: MetalBuffer,
    v_cache: MetalBuffer,
    qg: MetalBuffer,
    q: MetalBuffer,
    gate: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attn: MetalBuffer,
    kv_scratch: MetalBuffer,
    cache_cap: usize,
}

/// A complete served layer executed entirely on the device.
pub enum LayerGpu {
    Gdn(Box<GdnLayerGpu>),
    Attention(Box<AttentionLayerGpu>),
}

impl LayerGpu {
    /// Upload one served layer's weights and allocate its state and
    /// scratch (fixture scale: F16 matrices, F32 vectors).
    pub fn load(gpu: &Qwen35Gpu, model: &Qwen35Model, l: usize, capacity: usize) -> Result<Self> {
        let layer = &model.layout().layers[l];
        let common = LayerCommon::load(gpu, model, layer)?;
        let geom = common.geom;
        match &layer.attention {
            crate::qwen35::Qwen35AttentionTensors::Linear {
                qkv,
                gate,
                a,
                dt,
                alpha,
                beta,
                conv1d,
                norm,
                out,
            } => Ok(LayerGpu::Gdn(Box::new(GdnLayerGpu {
                qkv: DeviceTensor::load(gpu, model, qkv.index)?,
                z: DeviceTensor::load(gpu, model, gate.index)?,
                alpha: DeviceTensor::load(gpu, model, alpha.index)?,
                beta: DeviceTensor::load(gpu, model, beta.index)?,
                out_proj: DeviceTensor::load(gpu, model, out.index)?,
                ssm_a: vec_buf(gpu, &model.read_vector(a.index)?)?,
                ssm_dt: vec_buf(gpu, &model.read_vector(dt.index)?)?,
                ssm_norm: vec_buf(gpu, &model.read_vector(norm.index)?)?,
                conv_w: vec_buf(gpu, &model.read_vector(conv1d.index)?)?,
                state: zero_buf(gpu, geom.groups * geom.state_size * geom.head_v)?,
                hist: zero_buf(
                    gpu,
                    geom.conv_kernel.saturating_sub(1).max(1) * geom.channels,
                )?,
                qkv_out: zero_buf(gpu, geom.channels)?,
                conv_y: zero_buf(gpu, geom.channels)?,
                q: zero_buf(gpu, geom.groups * geom.state_size)?,
                k: zero_buf(gpu, geom.groups * geom.state_size)?,
                v: zero_buf(gpu, geom.head_v * geom.dt_rank)?,
                z_out: zero_buf(gpu, geom.head_v * geom.dt_rank)?,
                alpha_out: zero_buf(gpu, geom.dt_rank)?,
                beta_raw: zero_buf(gpu, geom.dt_rank)?,
                beta_out: zero_buf(gpu, geom.dt_rank)?,
                gate: zero_buf(gpu, geom.dt_rank)?,
                o: zero_buf(gpu, geom.head_v * geom.dt_rank)?,
                gated: zero_buf(gpu, geom.head_v * geom.dt_rank)?,
                common,
            }))),
            crate::qwen35::Qwen35AttentionTensors::Full {
                q,
                k,
                v,
                output,
                q_norm,
                k_norm,
            } => Ok(LayerGpu::Attention(Box::new(AttentionLayerGpu {
                q_w: DeviceTensor::load(gpu, model, q.index)?,
                k_w: DeviceTensor::load(gpu, model, k.index)?,
                v_w: DeviceTensor::load(gpu, model, v.index)?,
                o_w: DeviceTensor::load(gpu, model, output.index)?,
                q_norm: vec_buf(gpu, &model.read_vector(q_norm.index)?)?,
                k_norm: vec_buf(gpu, &model.read_vector(k_norm.index)?)?,
                inv_freq: vec_buf(
                    gpu,
                    &crate::qwen35_ops::rope_inv_freq_partial(
                        geom.rope_dims,
                        model.config().rope_base,
                    ),
                )?,
                k_cache: gpu.buffer(geom.kv_heads * capacity * geom.head_dim * 2)?,
                v_cache: gpu.buffer(geom.kv_heads * capacity * geom.head_dim * 2)?,
                qg: zero_buf(gpu, geom.heads * geom.head_dim * 2)?,
                q: zero_buf(gpu, geom.heads * geom.head_dim)?,
                gate: zero_buf(gpu, geom.heads * geom.head_dim)?,
                k: zero_buf(gpu, geom.kv_heads * geom.head_dim)?,
                v: zero_buf(gpu, geom.kv_heads * geom.head_dim)?,
                attn: zero_buf(gpu, geom.heads * geom.head_dim)?,
                kv_scratch: zero_buf(gpu, geom.heads * capacity)?,
                cache_cap: capacity,
                common,
            }))),
        }
    }

    /// One decode step for this layer, in place on `x`.
    pub fn forward(&self, gpu: &Qwen35Gpu, x: &MetalBuffer, pos: u32) -> Result<()> {
        match self {
            LayerGpu::Gdn(l) => l.forward(gpu, x),
            LayerGpu::Attention(l) => l.forward(gpu, x, pos),
        }
    }
}

impl GdnLayerGpu {
    fn forward(&self, gpu: &Qwen35Gpu, x: &MetalBuffer) -> Result<()> {
        let ctx = gpu.context();
        let geom = self.common.geom;
        ctx.rmsnorm(
            x,
            &self.common.attn_norm,
            &self.common.h,
            1,
            geom.hidden as u32,
            geom.eps,
        )?;
        self.qkv.gemv(gpu, &self.common.h, &self.qkv_out)?;
        gpu.conv_step_dev(
            &self.hist,
            &self.qkv_out,
            &self.conv_w,
            &self.conv_y,
            geom.channels as u32,
            geom.conv_kernel as u32,
        )?;
        ctx.qwen35_silu(&self.conv_y, geom.channels as u32)?;
        ctx.qwen35_split_qkv(
            &self.conv_y,
            &self.q,
            &self.k,
            &self.v,
            (geom.groups * geom.state_size) as u32,
        )?;
        gpu.l2norm_dev(
            &self.q,
            &self.q,
            geom.groups as u32,
            geom.state_size as u32,
            geom.eps,
        )?;
        gpu.l2norm_dev(
            &self.k,
            &self.k,
            geom.groups as u32,
            geom.state_size as u32,
            geom.eps,
        )?;
        self.z.gemv(gpu, &self.common.h, &self.z_out)?;
        self.alpha.gemv(gpu, &self.common.h, &self.alpha_out)?;
        self.beta.gemv(gpu, &self.common.h, &self.beta_raw)?;
        gpu.beta_decay_dev(
            &self.beta_raw,
            &self.alpha_out,
            &self.ssm_dt,
            &self.ssm_a,
            &self.beta_out,
            &self.gate,
            geom.dt_rank as u32,
        )?;
        gpu.gdn_step_dev(
            &self.q,
            &self.k,
            &self.v,
            &self.gate,
            &self.beta_out,
            &self.state,
            &self.o,
            geom.groups as u32,
            geom.state_size as u32,
            geom.head_v as u32,
            1.0 / (geom.state_size as f32).sqrt(),
        )?;
        gpu.gated_norm_dev(
            &self.o,
            &self.z_out,
            &self.ssm_norm,
            &self.gated,
            geom.dt_rank as u32,
            geom.head_v as u32,
            geom.eps,
        )?;
        self.out_proj.gemv(gpu, &self.gated, &self.common.proj)?;
        ctx.f32_axpy(&self.common.proj, x, 1.0, geom.hidden as u32)?;
        self.common.ffn_residual(gpu, x)
    }
}

impl AttentionLayerGpu {
    fn forward(&self, gpu: &Qwen35Gpu, x: &MetalBuffer, pos: u32) -> Result<()> {
        let ctx = gpu.context();
        let geom = self.common.geom;
        ctx.rmsnorm(
            x,
            &self.common.attn_norm,
            &self.common.h,
            1,
            geom.hidden as u32,
            geom.eps,
        )?;
        self.q_w.gemv(gpu, &self.common.h, &self.qg)?;
        gpu.q_gate_split_dev(
            &self.qg,
            &self.q,
            &self.gate,
            geom.heads as u32,
            geom.head_dim as u32,
        )?;
        self.k_w.gemv(gpu, &self.common.h, &self.k)?;
        self.v_w.gemv(gpu, &self.common.h, &self.v)?;
        ctx.rmsnorm(
            &self.q,
            &self.q_norm,
            &self.q,
            geom.heads as u32,
            geom.head_dim as u32,
            geom.eps,
        )?;
        ctx.rmsnorm(
            &self.k,
            &self.k_norm,
            &self.k,
            geom.kv_heads as u32,
            geom.head_dim as u32,
            geom.eps,
        )?;
        gpu.rope_partial_dev(
            &self.q,
            &self.k,
            &self.inv_freq,
            pos,
            geom.heads as u32,
            geom.kv_heads as u32,
            geom.head_dim as u32,
            geom.rope_dims as u32,
        )?;
        gpu.kv_store_f16_dev(
            &self.k,
            &self.v,
            &self.k_cache,
            &self.v_cache,
            pos,
            self.cache_cap as u32,
            geom.head_dim as u32,
            geom.kv_heads as u32,
        )?;
        gpu.attention_f16_dev(
            &self.q,
            &self.k_cache,
            &self.v_cache,
            &self.attn,
            &self.kv_scratch,
            geom.heads as u32,
            pos,
            self.cache_cap as u32,
            geom.head_dim as u32,
            (geom.heads / geom.kv_heads) as u32,
            1.0 / (geom.head_dim as f32).sqrt(),
        )?;
        gpu.attn_gate_dev(&self.attn, &self.gate, (geom.heads * geom.head_dim) as u32)?;
        self.o_w.gemv(gpu, &self.attn, &self.common.proj)?;
        ctx.f32_axpy(&self.common.proj, x, 1.0, geom.hidden as u32)?;
        self.common.ffn_residual(gpu, x)
    }
}

#[cfg(test)]
mod layer_tests {
    use super::*;
    use crate::qwen35_kv::KvPrecision;
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

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

    /// Registered device-vs-oracle tolerances for the complete fixture
    /// model (observed worst: hidden 5.4e-7, logits 1.4e-6 over 3 tokens).
    const HIDDEN_TOL: f32 = 1e-5;
    const LOGITS_TOL: f32 = 1e-4;

    fn device_f32(buf: &MetalBuffer, n: usize) -> Vec<f32> {
        let mut b = vec![0u8; n * 4];
        buf.download(0, &mut b).unwrap();
        bytes_f32(&b)
    }

    /// Complete fixture-model verification: every served layer executes on
    /// the device (weights and activations stay on the GPU between
    /// layers) and each hidden state matches the CPU oracle bit-close;
    /// the final tied logits agree with the oracle's.
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
        let gpu = Qwen35Gpu::open().unwrap();

        let hidden = cpu.config().hidden as usize;
        let vocab = cpu.config().vocab as usize;
        let eps = cpu.config().rms_eps;

        let layers: Vec<LayerGpu> = (0..cpu.layout().layers.len())
            .map(|l| LayerGpu::load(&gpu, &cpu, l, capacity).unwrap())
            .collect();
        let output_norm = vec_buf(
            &gpu,
            &cpu.read_vector(cpu.layout().output_norm.index).unwrap(),
        )
        .unwrap();
        let tied = DeviceTensor::load(&gpu, &cpu, cpu.layout().token_embd.index).unwrap();
        let x = zero_buf(&gpu, hidden).unwrap();
        let h = zero_buf(&gpu, hidden).unwrap();
        let logits_buf = zero_buf(&gpu, vocab).unwrap();
        let embed_buf = gpu.buffer(hidden * 2).unwrap();
        let embed_bytes = cpu.raw_tensor_bytes(cpu.layout().token_embd.index).unwrap();

        let tokens = [3u32, 11, 29];
        let mut worst_hidden = 0.0f32;
        let mut worst_logits = 0.0f32;
        for (pos, token) in tokens.iter().enumerate() {
            let mut cpu_logits = Vec::new();
            let mut trace = Vec::new();
            cpu.forward_token_traced(*token, &mut cpu_logits, &mut trace)
                .unwrap();

            // Embedding row: the fixture's token_embd is F16, uploaded raw.
            let start = *token as usize * hidden * 2;
            embed_buf
                .upload(0, &embed_bytes[start..start + hidden * 2])
                .unwrap();
            gpu.context()
                .cast_f16_to_f32(&embed_buf, &x, hidden as u32)
                .unwrap();
            let gx = device_f32(&x, hidden);
            for (g, c) in gx.iter().zip(trace[0].iter()) {
                worst_hidden =
                    worst_hidden.max(assert_close(*g, *c, HIDDEN_TOL, HIDDEN_TOL, "embedding"));
            }

            for (l, layer) in layers.iter().enumerate() {
                layer.forward(&gpu, &x, pos as u32).unwrap();
                let gx = device_f32(&x, hidden);
                for (g, c) in gx.iter().zip(trace[l + 1].iter()) {
                    worst_hidden = worst_hidden.max(assert_close(
                        *g,
                        *c,
                        HIDDEN_TOL,
                        HIDDEN_TOL,
                        &format!("layer {l}"),
                    ));
                }
            }

            gpu.context()
                .rmsnorm(&x, &output_norm, &h, 1, hidden as u32, eps)
                .unwrap();
            tied.gemv(&gpu, &h, &logits_buf).unwrap();
            let gpu_logits = device_f32(&logits_buf, vocab);
            let mut top_gpu = 0usize;
            let mut top_cpu = 0usize;
            for i in 0..vocab {
                if gpu_logits[i] > gpu_logits[top_gpu] {
                    top_gpu = i;
                }
                if cpu_logits[i] > cpu_logits[top_cpu] {
                    top_cpu = i;
                }
                worst_logits = worst_logits.max((gpu_logits[i] - cpu_logits[i]).abs());
            }
            assert_eq!(top_gpu, top_cpu, "token {token}: top-1 matches the oracle");
        }
        eprintln!(
            "fixture device model: worst hidden|diff| {worst_hidden:.3e}, worst logits|diff| {worst_logits:.3e} over {tokens:?}"
        );
        assert!(worst_hidden <= HIDDEN_TOL, "registered hidden tolerance");
        assert!(worst_logits <= LOGITS_TOL, "registered logits tolerance");
    }
}
