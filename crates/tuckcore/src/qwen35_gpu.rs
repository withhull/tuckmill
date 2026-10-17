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

/// Device-resident GDN operator set for one layer.
pub struct GdnGpu {
    ctx: MetalContext,
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
        let probe = probe();
        if !probe.available {
            return Err(TuckmillError::new(
                ErrorCode::Unavailable,
                format!("no Metal device is available ({})", probe.info),
            ));
        }
        let ctx = MetalContext::open()?;
        let buf = |elems: usize| -> Result<MetalBuffer> { ctx.buffer(elems * 4, true) };
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
            ctx,
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
        self.ctx.begin_group()?;
        let result = (|| {
            self.ctx.qwen35_conv_step(
                &self.hist,
                &self.x_in,
                &self.w_conv,
                &self.y_out,
                self.conv_channels as u32,
                self.conv_kernel as u32,
            )?;
            self.ctx.qwen35_conv_shift(
                &self.hist,
                &self.x_in,
                self.conv_channels as u32,
                self.conv_kernel as u32,
            )
        })();
        match result {
            Ok(()) => self.ctx.end_group()?,
            Err(e) => {
                let _ = self.ctx.abort_group();
                return Err(e);
            }
        }
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
        self.ctx.qwen35_l2norm_heads(
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
        self.ctx.qwen35_beta_decay(
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
        self.ctx.qwen35_gdn_step(
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
        self.ctx.qwen35_gated_norm(
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
