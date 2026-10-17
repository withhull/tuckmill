//! Scalar reference operators for the Qwen3.5 hybrid layers
//! (Q35-1B/1C, ADR-008).
//!
//! Semantics re-derived from llama.cpp b11695 and exercised by
//! deterministic operator and recurrent-state tests:
//!
//! - convolution: `ggml_ssm_conv` (CPU) — `out[t][c] = sum_j
//!   w[j][c] * x[t+j][c]`, no bias, no flip; the state is the previous
//!   `kernel - 1` input rows (`build_conv_state`);
//! - L2 norm (`build_gdn_l2_norm`): `x / sqrt(sum(x^2) + eps)` per head,
//!   which is exactly `scale(rms_norm(x, eps/n), 1/sqrt(n))`;
//! - delta rule (`build_delta_net_autoregressive`): per value head,
//!   `s~ = s * exp(g)`, `sk = s~^T k`, `d = beta * (v - sk)`,
//!   `s = s~ + k d^T`, `o = s^T (q / sqrt(S_k))`;
//! - scalars match ggml: `silu(x) = x * sigmoid(x)`, `softplus(x) =
//!   x > 20 ? x : ln(1 + e^x)`, `sigmoid(x) = 1 / (1 + e^-x)`;
//! - full attention (Q35-1C): weighted per-head RMS norm for Q/K
//!   (`rmsnorm_in_place`, the shared Qwen3 primitive), partial split-half
//!   RoPE over the rotated span only (`n_rot` = `rope.dimension_count`;
//!   IMRoPE with text positions reduces to this), grouped causal
//!   decode attention with consecutive-head KV mapping, and the joint
//!   Q-projection sigmoid gate applied before the output projection;
//! - FFN (Q35-1C): SwiGLU (`silu(gate) * up`) and the residual adds.
//!
//! The operators are portable, f32, and deterministic; they are the
//! reference the Q35-1C assembly and later Metal kernels are validated
//! against. Nothing here is wired into the loader: execution stays
//! refused until the Q35-1 checklist passes.

/// `sigmoid(x) = 1 / (1 + e^-x)` (ggml's f32 sigmoid).
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `silu(x) = x * sigmoid(x)` (ggml's f32 silu).
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// ggml's numerically safe softplus (`ggml_compute_softplus_f32`):
/// `x > 20 ? x : ln(1 + e^x)`.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Per-head L2 normalization in place: `x / sqrt(sum(x^2) + eps)`.
/// `row` holds exactly one head's vector; the caller iterates heads.
pub fn l2_normalize_head(row: &mut [f32], eps: f32) {
    let mut sum = 0.0f32;
    for x in row.iter() {
        sum += x * x;
    }
    let inv = 1.0 / (sum + eps).sqrt();
    for x in row.iter_mut() {
        *x *= inv;
    }
}

/// Per-layer convolution state: the previous `kernel - 1` input rows of
/// the (silu-applied-later) QKV stream, `channels` wide each.
#[derive(Debug, Clone)]
pub struct ConvState {
    kernel: usize,
    channels: usize,
    /// Oldest first; `kernel - 1` rows. Zero rows before the first token.
    history: Vec<f32>,
}

impl ConvState {
    pub fn new(kernel: usize, channels: usize) -> Self {
        assert!(kernel >= 1, "conv kernel must be at least 1");
        Self {
            kernel,
            channels,
            history: vec![0.0; (kernel - 1) * channels],
        }
    }

    /// Bytes held by this state (mirrors the admitted plan's pricing).
    pub fn state_bytes(&self) -> u64 {
        (self.history.len() * std::mem::size_of::<f32>()) as u64
    }

    pub fn reset(&mut self) {
        self.history.fill(0.0);
    }

    /// Process `tokens` input rows (`tokens.len() == n * channels`)
    /// through the depthwise causal convolution, writing `n * channels`
    /// outputs. `weights` is the `ssm_conv1d` tensor `[kernel,
    /// channels]` (tap-major). Advancing the state across chunked calls
    /// is exactly equivalent to one full pass (tested).
    pub fn step_sequence(&mut self, weights: &[f32], tokens: &[f32], out: &mut [f32]) {
        let k = self.kernel;
        let c = self.channels;
        assert_eq!(
            weights.len(),
            k * c,
            "conv weights must be [kernel, channels]"
        );
        assert_eq!(tokens.len() % c, 0, "tokens must be whole rows");
        let n = tokens.len() / c;
        assert_eq!(out.len(), n * c, "output must match token count");

        // Working window: history rows followed by the input rows.
        let mut window = Vec::with_capacity(((k - 1) + n) * c);
        window.extend_from_slice(&self.history);
        window.extend_from_slice(tokens);

        for t in 0..n {
            for ch in 0..c {
                let mut acc = 0.0f32;
                for j in 0..k {
                    acc += weights[j * c + ch] * window[(t + j) * c + ch];
                }
                out[t * c + ch] = acc;
            }
        }

        // The new state is the last `kernel - 1` rows of the window.
        if k > 1 {
            let total_rows = (k - 1) + n;
            let keep_from = total_rows - (k - 1);
            self.history
                .copy_from_slice(&window[keep_from * c..total_rows * c]);
        }
    }
}

/// Per-layer recurrent (delta-rule) state of one linear-attention layer.
///
/// Layout: `heads` matrices `s[h][i, j]`, `i` over the key dimension
/// (the contraction axis) and `j` over the value dimension (the output
/// axis), stored `i`-major. This matches `ggml_ssm_conv`-era ggml:
/// `sk[j] = sum_i s~[i, j] * k[i]` and `o[j] = sum_i s[i, j] * q[i]`.
#[derive(Debug, Clone)]
pub struct GdnState {
    heads: usize,
    k_dim: usize,
    v_dim: usize,
    s: Vec<f32>,
}

impl GdnState {
    pub fn new(heads: usize, k_dim: usize, v_dim: usize) -> Self {
        Self {
            heads,
            k_dim,
            v_dim,
            s: vec![0.0; heads * k_dim * v_dim],
        }
    }

    /// Bytes held by this state (mirrors the admitted plan's pricing:
    /// `heads x k_dim x v_dim x 4`).
    pub fn state_bytes(&self) -> u64 {
        (self.s.len() * std::mem::size_of::<f32>()) as u64
    }

    pub fn reset(&mut self) {
        self.s.fill(0.0);
    }

    /// Read-only view of the raw state (`heads x k_dim x v_dim`,
    /// `i`-major).
    pub fn snapshot(&self) -> &[f32] {
        &self.s
    }

    /// One token through the delta rule for every head.
    ///
    /// - `q`, `k`: `heads x k_dim` (already L2-normalized); `q` is
    ///   scaled by `1 / sqrt(k_dim)` here, exactly as the graph does;
    /// - `v`: `heads x v_dim`;
    /// - `gate`: raw per-head gate values (`ssm_a * softplus(alpha +
    ///   dt_bias)`); the decay is `exp(gate)`, computed here so the
    ///   decay-before-delta ordering cannot be misapplied by callers;
    /// - `beta`: per-head sigmoid of the beta projection;
    /// - `out`: `heads x v_dim`.
    pub fn step(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        out: &mut [f32],
    ) {
        let h = self.heads;
        let sk_dim = self.k_dim;
        let sv_dim = self.v_dim;
        assert_eq!(q.len(), h * sk_dim, "q must be [heads, k_dim]");
        assert_eq!(k.len(), h * sk_dim, "k must be [heads, k_dim]");
        assert_eq!(v.len(), h * sv_dim, "v must be [heads, v_dim]");
        assert_eq!(gate.len(), h, "gate must be per head");
        assert_eq!(beta.len(), h, "beta must be per head");
        assert_eq!(out.len(), h * sv_dim, "out must be [heads, v_dim]");

        let q_scale = 1.0 / (sk_dim as f32).sqrt();

        let mut sk = vec![0.0f32; sv_dim];
        let mut d = vec![0.0f32; sv_dim];
        for head in 0..h {
            let decay = gate[head].exp();
            let b = beta[head];
            let s = &mut self.s[head * sk_dim * sv_dim..(head + 1) * sk_dim * sv_dim];
            let kk = &k[head * sk_dim..(head + 1) * sk_dim];
            let vv = &v[head * sv_dim..(head + 1) * sv_dim];

            // s~ = s * exp(g); sk = s~^T k; d = beta * (v - sk).
            for j in 0..sv_dim {
                let mut acc = 0.0f32;
                for i in 0..sk_dim {
                    let elem = &mut s[i * sv_dim + j];
                    *elem *= decay;
                    acc += *elem * kk[i];
                }
                sk[j] = acc;
                d[j] = b * (vv[j] - acc);
            }

            // s = s~ + k d^T; o = s^T (q * scale).
            for j in 0..sv_dim {
                out[head * sv_dim + j] = 0.0;
            }
            for i in 0..sk_dim {
                let ki = kk[i];
                let qs = q[head * sk_dim + i] * q_scale;
                for j in 0..sv_dim {
                    let elem = &mut s[i * sv_dim + j];
                    *elem += ki * d[j];
                    out[head * sv_dim + j] += *elem * qs;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full-attention, RoPE and FFN operators (Q35-1C)
// ---------------------------------------------------------------------------

use crate::forward::{rmsnorm_in_place, rope_apply};

/// RoPE inverse-frequency table for a partially-rotated head.
///
/// `rope_dims` is the number of rotated dims (the pinned model rotates 64
/// of 256 full-attention dims). ggml spaces the frequencies over the
/// *rotated* dim count, not the head width: `theta^(-2i/rope_dims)`,
/// which matches `ggml_mrope_cache_init` for text positions (identical
/// positions on every IMRoPE axis reduce to the standard table).
pub fn rope_inv_freq_partial(rope_dims: usize, base: f32) -> Vec<f32> {
    (0..rope_dims / 2)
        .map(|i| {
            let exponent = 2.0 * i as f64 / rope_dims as f64;
            (1.0 / (base as f64).powf(exponent)) as f32
        })
        .collect()
}

/// Split-half (NeoX / IMRoPE) RoPE for one head vector, in place: pairs
/// `(i, i + rope_dims/2)` rotate by `pos * inv_freq[i]`; dims at or beyond
/// `2 * inv_freq.len()` pass through untouched. This is ggml's
/// `rotate_pairs` with `n_offset = n_dims / 2`.
pub fn partial_rope_in_place(head: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let rope_dims = inv_freq.len() * 2;
    assert!(
        head.len() >= rope_dims,
        "head shorter than the rotated span"
    );
    rope_apply(&mut head[..rope_dims], pos, inv_freq);
}

/// RMS norm with an explicit weight vector (ggml `LLM_NORM_RMS`): the
/// attention path's Q/K head norms and the layer norms all share this.
pub fn rms_norm_weighted_in_place(x: &mut [f32], weight: &[f32], eps: f32) {
    rmsnorm_in_place(x, weight, eps);
}

/// Numerically stable softmax over one score row (subtract the max,
/// mirroring `ggml_compute_forward_softmax_f32`).
pub fn softmax_in_place(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for v in row.iter() {
        max = max.max(*v);
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// GQA mapping: consecutive query heads share a KV head,
/// `h / (heads_q / heads_kv)` (llama.cpp grouped attention).
pub fn kv_head_for(head: usize, heads_q: usize, heads_kv: usize) -> usize {
    assert!(heads_kv > 0 && heads_q.is_multiple_of(heads_kv));
    head / (heads_q / heads_kv)
}

/// Query/KV geometry of one full-attention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GqaGeometry {
    pub heads_q: usize,
    pub heads_kv: usize,
    pub head_dim: usize,
}

/// One decode step of grouped causal attention.
///
/// `q` is `heads_q x head_dim` for the newest token; `k_cache`/`v_cache`
/// are `heads_kv x seq x head_dim` and already include that token, so
/// every position in the cache is causally visible. `scale` is the
/// graph's query scale (`1 / sqrt(head_dim)` for the pinned model).
pub fn gqa_cache_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    geometry: GqaGeometry,
    seq: usize,
    scale: f32,
    out: &mut [f32],
) {
    let GqaGeometry {
        heads_q,
        heads_kv,
        head_dim,
    } = geometry;
    assert!(seq > 0, "attention needs at least one cached position");
    assert_eq!(q.len(), heads_q * head_dim, "q must be [heads_q, head_dim]");
    assert_eq!(k_cache.len(), heads_kv * seq * head_dim);
    assert_eq!(v_cache.len(), k_cache.len());
    assert_eq!(out.len(), heads_q * head_dim);

    let mut scores = vec![0.0f32; seq];
    for h in 0..heads_q {
        let kv = kv_head_for(h, heads_q, heads_kv);
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        for t in 0..seq {
            let kt = &k_cache[(kv * seq + t) * head_dim..][..head_dim];
            let mut acc = 0.0f32;
            for d in 0..head_dim {
                acc += qh[d] * kt[d];
            }
            scores[t] = acc * scale;
        }
        softmax_in_place(&mut scores);
        let o = &mut out[h * head_dim..(h + 1) * head_dim];
        o.fill(0.0);
        for t in 0..seq {
            let vt = &v_cache[(kv * seq + t) * head_dim..][..head_dim];
            let w = scores[t];
            for d in 0..head_dim {
                o[d] += w * vt[d];
            }
        }
    }
}

/// Qwen3.5's gated attention output: `attn[i] *= sigmoid(gate[i])`. The
/// gate is the second half of the joint Q projection and is applied
/// before the output projection.
pub fn apply_sigmoid_gate(attn: &mut [f32], gate: &[f32]) {
    assert_eq!(
        attn.len(),
        gate.len(),
        "gate must match the attention output"
    );
    for (a, g) in attn.iter_mut().zip(gate.iter()) {
        *a *= sigmoid(*g);
    }
}

/// SwiGLU activation: `gate[i] = silu(gate[i]) * up[i]` (the FFN's
/// elementwise stage between the gate/up projections and the down
/// projection).
pub fn swiglu_in_place(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    for (g, u) in gate.iter_mut().zip(up.iter()) {
        *g = silu(*g) * *u;
    }
}

/// Residual accumulation: `dst[i] += src[i]` (the layer's two residual
/// adds).
pub fn residual_add_in_place(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random values in [-1, 1).
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    #[test]
    fn scalar_transforms_match_the_ggml_reference() {
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(1.0) - 0.7310586).abs() < 1e-6, "silu(1)");
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!((sigmoid(3.0) - 1.0 / (1.0 + (-3.0f32).exp())).abs() < 1e-7);
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert!(
            (softplus(20.0) - (1.0 + 20.0f32.exp()).ln()).abs() < 1e-6,
            "at the threshold the formula still applies"
        );
        assert_eq!(softplus(25.0), 25.0, "x > 20 returns x unchanged");
    }

    #[test]
    fn l2_norm_adds_epsilon_inside_the_root() {
        let mut v = [3.0f32, 4.0];
        l2_normalize_head(&mut v, 1e-6);
        let inv = 1.0 / (25.0f32 + 1e-6).sqrt();
        assert!((v[0] - 3.0 * inv).abs() < 1e-7);
        assert!((v[1] - 4.0 * inv).abs() < 1e-7);

        let mut zero = [0.0f32, 0.0];
        l2_normalize_head(&mut zero, 1e-6);
        assert_eq!(zero, [0.0, 0.0], "all-zero rows stay finite at zero");
    }

    #[test]
    fn conv_matches_the_hand_computed_window() {
        // k = 2, channels = 2, weights tap-major.
        let weights = [1.0f32, 10.0, 100.0, 1000.0];
        let tokens = [1.0f32, 2.0, 3.0, 4.0];
        let mut state = ConvState::new(2, 2);
        let mut out = [0.0f32; 4];
        state.step_sequence(&weights, &tokens, &mut out);
        // window = [0,0 | 1,2 | 3,4]; out[t] = w0*window[t] + w1*window[t+1].
        assert_eq!(out, [100.0, 2000.0, 301.0, 4020.0]);
        // The state keeps the last kernel-1 = 1 window row.
        assert_eq!(state.history, [3.0, 4.0]);
    }

    #[test]
    fn conv_chunked_calls_match_one_full_pass() {
        let kernel = 3;
        let channels = 4;
        let mut seed = 0xC0FFEEu64;
        let weights: Vec<f32> = (0..kernel * channels).map(|_| lcg(&mut seed)).collect();
        let n = 7;
        let tokens: Vec<f32> = (0..n * channels).map(|_| lcg(&mut seed)).collect();

        let mut one = ConvState::new(kernel, channels);
        let mut out_one = vec![0.0f32; n * channels];
        one.step_sequence(&weights, &tokens, &mut out_one);

        let mut chunked = ConvState::new(kernel, channels);
        let mut out_chunked = vec![0.0f32; n * channels];
        chunked.step_sequence(
            &weights,
            &tokens[..2 * channels],
            &mut out_chunked[..2 * channels],
        );
        chunked.step_sequence(
            &weights,
            &tokens[2 * channels..5 * channels],
            &mut out_chunked[2 * channels..5 * channels],
        );
        chunked.step_sequence(
            &weights,
            &tokens[5 * channels..],
            &mut out_chunked[5 * channels..],
        );

        assert_eq!(out_one, out_chunked, "conv state handoff is exact");
        assert_eq!(one.history, chunked.history);
    }

    #[test]
    fn delta_rule_matches_hand_computed_first_token() {
        let mut state = GdnState::new(1, 1, 1);
        let mut out = [0.0f32; 1];
        // Zero state: d = beta * v; s = k * d; o[j] = s[j] * q * scale.
        state.step(&[1.0], &[2.0], &[3.0], &[-1.0], &[0.5], &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6, "{}", out[0]);
        assert!((state.snapshot()[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn delta_rule_matches_hand_computed_second_token() {
        let mut state = GdnState::new(1, 1, 1);
        let mut out = [0.0f32; 1];
        state.step(&[1.0], &[2.0], &[3.0], &[-1.0], &[0.5], &mut out);
        // Decay 0.5 with beta 0: s~ = 1.5, d = 0, s stays 1.5, o = 1.5.
        let gate = 0.5f32.ln();
        state.step(&[1.0], &[1.0], &[9.0], &[gate], &[0.0], &mut out);
        assert!((state.snapshot()[0] - 1.5).abs() < 1e-6);
        assert!((out[0] - 1.5).abs() < 1e-6, "{}", out[0]);
    }

    #[test]
    fn delta_rule_two_dim_hand_case() {
        let mut state = GdnState::new(1, 2, 2);
        // s = [[1, 0], [0, 2]] (i-major).
        state.s.copy_from_slice(&[1.0, 0.0, 0.0, 2.0]);
        let mut out = [0.0f32; 2];
        let gate = 0.5f32.ln(); // decay 0.5
        state.step(
            &[1.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[gate],
            &[1.0],
            &mut out,
        );
        // s~ = [[0.5, 0], [0, 1]]; sk = [0.5, 1]; d = [0.5, 0];
        // s = [[1, 0], [0.5, 1]]; o = s^T (q / sqrt(2)) = [1/sqrt(2), 0].
        assert_eq!(state.snapshot(), [1.0, 0.0, 0.5, 1.0].as_slice());
        let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();
        assert!((out[0] - inv_sqrt2).abs() < 1e-6, "{}", out[0]);
        assert_eq!(out[1], 0.0);
    }

    /// Independent algebraic check with the transposed arrangement:
    /// `St = St~ + d k^T` with `d = beta * (v - St~ k)` and
    /// `o = (q * scale) St^T`, computed in f64 over several steps.
    #[test]
    fn delta_rule_matches_the_transposed_algebraic_form() {
        let heads = 2;
        let sk = 3;
        let sv = 3;
        let mut seed = 0xBEEFCAFEu64;
        let mut module_state = GdnState::new(heads, sk, sv);

        // f64 reference: st[head] is [v][k] (transposed), row j = output.
        let mut st = vec![vec![0.0f64; sk * sv]; heads];
        let q_scale = 1.0 / (sk as f64).sqrt();

        for _step in 0..4 {
            let q: Vec<f32> = (0..heads * sk).map(|_| lcg(&mut seed)).collect();
            let k: Vec<f32> = (0..heads * sk).map(|_| lcg(&mut seed)).collect();
            let v: Vec<f32> = (0..heads * sv).map(|_| lcg(&mut seed)).collect();
            let gate: Vec<f32> = (0..heads).map(|_| lcg(&mut seed)).collect();
            let beta: Vec<f32> = (0..heads)
                .map(|_| 0.5 + 0.5 * lcg(&mut seed).abs())
                .collect();

            let mut out = vec![0.0f32; heads * sv];
            module_state.step(&q, &k, &v, &gate, &beta, &mut out);

            for h in 0..heads {
                let decay = (gate[h] as f64).exp();
                let b = beta[h] as f64;
                let kk: &[f32] = &k[h * sk..(h + 1) * sk];
                let vv: &[f32] = &v[h * sv..(h + 1) * sv];
                // s~ then d = beta (v - s~ k).
                for x in st[h].iter_mut() {
                    *x *= decay;
                }
                let mut sk_vec = vec![0.0f64; sv];
                for j in 0..sv {
                    for i in 0..sk {
                        sk_vec[j] += st[h][j * sk + i] * kk[i] as f64;
                    }
                }
                let d: Vec<f64> = (0..sv).map(|j| b * (vv[j] as f64 - sk_vec[j])).collect();
                for j in 0..sv {
                    for i in 0..sk {
                        st[h][j * sk + i] += d[j] * kk[i] as f64;
                    }
                }
                for j in 0..sv {
                    let mut o = 0.0f64;
                    for i in 0..sk {
                        o += q[h * sk + i] as f64 * q_scale * st[h][j * sk + i];
                    }
                    assert!(
                        (o - out[h * sv + j] as f64).abs() < 1e-5,
                        "step mismatch: {o} vs {}",
                        out[h * sv + j]
                    );
                }
            }
        }
    }

    #[test]
    fn zero_beta_is_pure_decay_and_unity_decay_is_the_identity() {
        let mut state = GdnState::new(1, 2, 2);
        state.s.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let mut out = [0.0f32; 2];
        // decay 1, beta 0: state unchanged, o = s^T (q / sqrt(2)).
        // q = [sqrt(2), 0] cancels the internal scale exactly.
        state.step(
            &[2.0f32.sqrt(), 0.0],
            &[9.0, 9.0],
            &[9.0, 9.0],
            &[0.0],
            &[0.0],
            &mut out,
        );
        assert_eq!(state.snapshot(), [1.0, 2.0, 3.0, 4.0].as_slice());
        assert!((out[0] - 1.0).abs() < 1e-6, "{}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-6, "{}", out[1]);
    }

    #[test]
    fn operators_are_deterministic() {
        let mut seed = 0x12345678u64;
        let weights: Vec<f32> = (0..3 * 4).map(|_| lcg(&mut seed)).collect();
        let tokens: Vec<f32> = (0..5 * 4).map(|_| lcg(&mut seed)).collect();
        let q: Vec<f32> = (0..2 * 3).map(|_| lcg(&mut seed)).collect();
        let k: Vec<f32> = (0..2 * 3).map(|_| lcg(&mut seed)).collect();
        let v: Vec<f32> = (0..2 * 3).map(|_| lcg(&mut seed)).collect();
        let gate = [0.1f32, -0.2];
        let beta = [0.7f32, 0.3];

        let mut a = ConvState::new(3, 4);
        let mut b = ConvState::new(3, 4);
        let mut out_a = vec![0.0f32; tokens.len()];
        let mut out_b = vec![0.0f32; tokens.len()];
        a.step_sequence(&weights, &tokens, &mut out_a);
        b.step_sequence(&weights, &tokens, &mut out_b);
        assert_eq!(out_a, out_b);

        let mut sa = GdnState::new(2, 3, 3);
        let mut sb = GdnState::new(2, 3, 3);
        let mut oa = vec![0.0f32; 6];
        let mut ob = vec![0.0f32; 6];
        sa.step(&q, &k, &v, &gate, &beta, &mut oa);
        sb.step(&q, &k, &v, &gate, &beta, &mut ob);
        assert_eq!(oa, ob);
        assert_eq!(sa.snapshot(), sb.snapshot());
    }

    /// The operator state sizes tie back to the contract's pricing:
    /// 18 recurrent layers x (1,048,576 + 73,728) B on the pinned model.
    #[test]
    fn state_sizes_match_the_pinned_geometry() {
        let gdn = GdnState::new(16, 128, 128);
        assert_eq!(gdn.state_bytes(), 1_048_576);
        let mut conv = ConvState::new(4, 6144);
        // One sample of history is consumed per step; the byte cost is
        // the declared (kernel - 1) rows.
        conv.reset();
        assert_eq!(conv.state_bytes(), 73_728);
        assert_eq!(18 * (gdn.state_bytes() + conv.state_bytes()), 20_201_472);
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;

    #[test]
    fn partial_rope_frequencies_use_the_rotated_span() {
        let inv = rope_inv_freq_partial(4, 1e7);
        assert_eq!(inv.len(), 2);
        assert_eq!(inv[0], 1.0);
        let expect = (1e7f64).powf(-2.0 / 4.0) as f32;
        assert!((inv[1] - expect).abs() < 1e-12, "{}", inv[1]);
    }

    #[test]
    fn partial_rope_rotates_only_the_declared_dims() {
        let inv = rope_inv_freq_partial(4, 1e7); // 2 pairs within 4 dims
        let mut head = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // pos 0 is the identity.
        let mut at_zero = head;
        partial_rope_in_place(&mut at_zero, 0, &inv);
        assert_eq!(at_zero, head);

        // pos 1: pairs (0, 2) and (1, 3) rotate; dims 4..8 pass through.
        partial_rope_in_place(&mut head, 1, &inv);
        let (s0, c0) = 1.0f32.sin_cos();
        let (s1, c1) = inv[1].sin_cos();
        let expected = [
            1.0 * c0 - 3.0 * s0,
            2.0 * c1 - 4.0 * s1,
            1.0 * s0 + 3.0 * c0,
            2.0 * s1 + 4.0 * c1,
            5.0,
            6.0,
            7.0,
            8.0,
        ];
        for (got, want) in head.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn softmax_matches_manual_computation_and_stays_finite() {
        let mut row = [1.0f32, 2.0, 3.0];
        softmax_in_place(&mut row);
        let e = |x: f32| x.exp();
        let sum = e(1.0) + e(2.0) + e(3.0);
        assert!((row[0] - e(1.0) / sum).abs() < 1e-7);
        assert!((row[1] - e(2.0) / sum).abs() < 1e-7);
        assert!((row[2] - e(3.0) / sum).abs() < 1e-7);

        let mut large = [1000.0f32, 1001.0];
        softmax_in_place(&mut large);
        assert!(large.iter().all(|v| v.is_finite()));
        assert!(
            large[1] > large[0],
            "the larger score keeps the larger weight"
        );
        assert!((large[0] + large[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gqa_decode_attention_matches_the_hand_computed_case() {
        // 1 head, seq 2, head_dim 2, scale 1.
        let q = [1.0f32, 0.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 2];
        gqa_cache_attention(
            &q,
            &k,
            &v,
            GqaGeometry {
                heads_q: 1,
                heads_kv: 1,
                head_dim: 2,
            },
            2,
            1.0,
            &mut out,
        );
        let w0 = 1.0f32.exp() / (1.0f32.exp() + 1.0);
        let w1 = 1.0 - w0;
        assert!((out[0] - (w0 * 1.0 + w1 * 3.0)).abs() < 1e-6, "{}", out[0]);
        assert!((out[1] - (w0 * 2.0 + w1 * 4.0)).abs() < 1e-6, "{}", out[1]);
    }

    #[test]
    fn gqa_maps_consecutive_query_heads_to_one_kv_head() {
        assert_eq!(kv_head_for(0, 8, 2), 0);
        assert_eq!(kv_head_for(3, 8, 2), 0);
        assert_eq!(kv_head_for(4, 8, 2), 1);
        assert_eq!(kv_head_for(7, 8, 2), 1);

        // heads_q = 2, heads_kv = 1: both heads see kv head 0.
        let q = [1.0f32, 0.0, 0.0, 1.0];
        let k = [1.0f32, 0.0, 0.0, 1.0]; // two cached positions
        let v = [1.0f32, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 4];
        gqa_cache_attention(
            &q,
            &k,
            &v,
            GqaGeometry {
                heads_q: 2,
                heads_kv: 1,
                head_dim: 2,
            },
            2,
            1.0,
            &mut out,
        );
        // Head 0 scores [1, 0] and head 1 scores [0, 1]; with identity v
        // the outputs equal the softmax weights themselves.
        let w0 = 1.0f32.exp() / (1.0f32.exp() + 1.0);
        let w1 = 1.0 - w0;
        assert!((out[0] - w0).abs() < 1e-6);
        assert!((out[1] - w1).abs() < 1e-6);
        assert!((out[2] - w1).abs() < 1e-6);
        assert!((out[3] - w0).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_gate_matches_the_hand_computed_case() {
        let mut attn = [1.0f32, 2.0];
        let gate = [0.0f32, 3.0f32.ln()]; // sigmoid -> 0.5, 0.75
        apply_sigmoid_gate(&mut attn, &gate);
        assert!((attn[0] - 0.5).abs() < 1e-7);
        assert!((attn[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn swiglu_and_residual_match_the_hand_computed_cases() {
        let mut gate = [1.0f32, 2.0];
        let up = [3.0f32, 4.0];
        swiglu_in_place(&mut gate, &up);
        assert!((gate[0] - 0.7310586 * 3.0).abs() < 1e-6, "{}", gate[0]);
        let silu2 = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((gate[1] - silu2 * 4.0).abs() < 1e-6, "{}", gate[1]);

        let mut residual = [1.0f32, 2.0];
        residual_add_in_place(&mut residual, &[3.0, 4.0]);
        assert_eq!(residual, [4.0, 6.0]);
    }

    /// Q/K head norms use the stored weights as-is (the converter already
    /// applied the +1 shift); spot-check the weighted RMS math on a head.
    #[test]
    fn weighted_head_rms_norm_matches_manual() {
        let mut head = [3.0f32, 4.0];
        let weight = [1.0f32, 2.0];
        rms_norm_weighted_in_place(&mut head, &weight, 0.0);
        let mean_sq = (9.0f32 + 16.0) / 2.0;
        let inv = 1.0 / mean_sq.sqrt();
        assert!((head[0] - 3.0 * inv).abs() < 1e-6);
        assert!((head[1] - 4.0 * inv * 2.0).abs() < 1e-6);
    }
}
