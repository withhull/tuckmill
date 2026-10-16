//! Deterministic sampling shared by every backend (ADR-005).
//!
//! Greedy is the default and the only behavior when `temperature <= 0`.
//! Nonzero temperature applies the documented transforms (softmax over
//! non-NaN logits, optional top-k, optional nucleus top-p) deterministically
//! before a seeded draw, so two backends given the same logits, policy, and
//! seed produce the same token. The RNG is xorshift64*: reproducible across
//! platforms and runs.
//!
//! This module moved here from TuckCore when the supervisor's sampling
//! policy became part of the backend contract: the same transforms must
//! apply no matter which engine executes the artifact.

/// xorshift64* deterministic RNG: reproducible across platforms and runs.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Zero state would lock xorshift; mix it.
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sampling knobs. Defaults are greedy; nonzero temperature applies the
/// optional transforms before selection, deterministically.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0, // greedy
            top_k: None,
            top_p: None,
        }
    }
}

/// Greedy argmax with deterministic tie-breaking (lowest index). NaN logits
/// are skipped; +infinity legitimately wins (it is the maximum), while an
/// all-NaN row yields None so the caller fails loudly instead of guessing.
pub fn argmax(logits: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (i, v) in logits.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            None => best = Some((i, *v)),
            Some((_, bv)) if *v > bv => best = Some((i, *v)),
            _ => {}
        }
    }
    best.map(|(i, _)| i as u32)
}

/// Sample one token from logits under the knobs.
pub fn sample(logits: &[f32], cfg: &SamplingConfig, rng: &mut Rng) -> Option<u32> {
    if cfg.temperature <= 0.0 {
        return argmax(logits);
    }
    let temp = cfg.temperature as f64;
    // Softmax over non-NaN logits (max-shifted for stability).
    let max = logits
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return None; // every logit was NaN
    }
    if max == f32::INFINITY {
        // An infinite logit wins deterministically; exp() cannot rank it.
        return argmax(logits);
    }
    let mut probs: Vec<(u32, f64)> = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for (i, v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        let p = (((*v - max) as f64) / temp).exp();
        probs.push((i as u32, p));
        sum += p;
    }
    if sum <= 0.0 {
        return None;
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }
    // Top-k: keep the k largest.
    if let Some(k) = cfg.top_k {
        let k = k.max(1).min(probs.len());
        probs.sort_by(|a, b| b.1.total_cmp(&a.1));
        probs.truncate(k);
        let s: f64 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= s;
        }
    }
    // Top-p (nucleus): keep the smallest set whose cumulative mass >= p.
    if let Some(p) = cfg.top_p {
        let p = p.clamp(0.0, 1.0) as f64;
        if p < 1.0 {
            probs.sort_by(|a, b| b.1.total_cmp(&a.1));
            let mut acc = 0.0f64;
            let mut keep = 0usize;
            for (_, pr) in &probs {
                acc += pr;
                keep += 1;
                if acc >= p {
                    break;
                }
            }
            probs.truncate(keep.max(1));
            let s: f64 = probs.iter().map(|(_, pr)| pr).sum();
            for (_, pr) in probs.iter_mut() {
                *pr /= s;
            }
        }
    }
    // Deterministic draw.
    let r = rng.next_f64();
    let mut acc = 0.0f64;
    let mut last = probs[0].0;
    for (token, p) in &probs {
        last = *token;
        acc += *p;
        if r < acc {
            return Some(*token);
        }
    }
    Some(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_argmax_and_tie_breaks_low() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), Some(1));
        assert_eq!(argmax(&[3.0, 3.0]), Some(0));
        assert_eq!(argmax(&[f32::NAN, 1.0]), Some(1));
        assert_eq!(argmax(&[f32::NAN, f32::INFINITY]), Some(1));
        assert_eq!(argmax(&[]), None);
    }

    #[test]
    fn temperature_zero_uses_greedy() {
        let cfg = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = Rng::new(1);
        assert_eq!(sample(&[0.1, 0.9], &cfg, &mut rng), Some(1));
    }

    #[test]
    fn sampling_is_deterministic_for_a_seed() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: None,
        };
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        let logits = [0.5f32, 1.0, -0.5, 2.0];
        let sa: Vec<u32> = (0..32)
            .filter_map(|_| sample(&logits, &cfg, &mut a))
            .collect();
        let sb: Vec<u32> = (0..32)
            .filter_map(|_| sample(&logits, &cfg, &mut b))
            .collect();
        assert_eq!(sa, sb);
    }

    #[test]
    fn top_k_restricts_support() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: Some(1),
            top_p: None,
        };
        let mut rng = Rng::new(7);
        for _ in 0..16 {
            assert_eq!(sample(&[0.0, 5.0, 1.0], &cfg, &mut rng), Some(1));
        }
    }

    #[test]
    fn top_p_keeps_the_nucleus() {
        // One dominant probability: nucleus of 0.9 contains only it.
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.9),
        };
        let mut rng = Rng::new(3);
        let logits = [0.0f32, 10.0, 0.0];
        for _ in 0..16 {
            assert_eq!(sample(&logits, &cfg, &mut rng), Some(1));
        }
    }

    #[test]
    fn all_nan_is_none() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: None,
        };
        let mut rng = Rng::new(0);
        assert_eq!(sample(&[f32::NAN, f32::NAN], &cfg, &mut rng), None);
    }
}
