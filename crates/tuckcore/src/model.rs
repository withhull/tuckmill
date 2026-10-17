//! Qwen3 model configuration resolved from the artifact and its adapter.

use tuckmill_manifest::adapter::{AdapterRegistry, AdapterSpec};
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::gguf::GgufInspection;

/// Dense Qwen3 geometry plus the tensor names Stage 0 executes.
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub layers: u64,
    pub hidden: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub key_length: u64,
    pub value_length: u64,
    pub ffn: u64,
    pub vocab: u64,
    pub rope_base: f32,
    pub rms_eps: f32,
    pub eos_token_id: Option<u32>,
}

/// Resolved tensor names for one layer and the globals.
#[derive(Debug, Clone)]
pub struct TensorNames {
    pub token_embd: String,
    pub output: Option<String>,
    pub output_norm: String,
    pub attn_norm: Vec<String>,
    pub attn_q: Vec<String>,
    pub attn_k: Vec<String>,
    pub attn_v: Vec<String>,
    pub attn_output: Vec<String>,
    pub attn_q_norm: Vec<String>,
    pub attn_k_norm: Vec<String>,
    pub ffn_norm: Vec<String>,
    pub ffn_gate: Vec<String>,
    pub ffn_up: Vec<String>,
    pub ffn_down: Vec<String>,
}

impl TensorNames {
    pub fn for_layers(layers: u64) -> Self {
        let p = |l: u64, s: &str| format!("blk.{l}.{s}");
        let mut names = Self {
            token_embd: "token_embd.weight".into(),
            output: None,
            output_norm: "output_norm.weight".into(),
            attn_norm: vec![],
            attn_q: vec![],
            attn_k: vec![],
            attn_v: vec![],
            attn_output: vec![],
            attn_q_norm: vec![],
            attn_k_norm: vec![],
            ffn_norm: vec![],
            ffn_gate: vec![],
            ffn_up: vec![],
            ffn_down: vec![],
        };
        for l in 0..layers {
            names.attn_norm.push(p(l, "attn_norm.weight"));
            names.attn_q.push(p(l, "attn_q.weight"));
            names.attn_k.push(p(l, "attn_k.weight"));
            names.attn_v.push(p(l, "attn_v.weight"));
            names.attn_output.push(p(l, "attn_output.weight"));
            names.attn_q_norm.push(p(l, "attn_q_norm.weight"));
            names.attn_k_norm.push(p(l, "attn_k_norm.weight"));
            names.ffn_norm.push(p(l, "ffn_norm.weight"));
            names.ffn_gate.push(p(l, "ffn_gate.weight"));
            names.ffn_up.push(p(l, "ffn_up.weight"));
            names.ffn_down.push(p(l, "ffn_down.weight"));
        }
        names
    }

    /// Resolve against what the artifact actually contains; `output` falls
    /// back to tied embeddings when absent (documented behavior, not a guess:
    /// absence of output.weight means tied by the format's convention).
    pub fn resolve(&mut self, inspection: &GgufInspection) -> Result<()> {
        let has = |name: &str| inspection.tensors.iter().any(|t| t.name == name);
        if !has(&self.token_embd) {
            return Err(missing(&self.token_embd));
        }
        if has("output.weight") {
            self.output = Some("output.weight".into());
        }
        let check_all = |names: &[String], what: &str| -> Result<()> {
            for n in names {
                if !has(n) {
                    return Err(missing(n));
                }
            }
            let _ = what;
            Ok(())
        };
        check_all(std::slice::from_ref(&self.output_norm), "output norm")?;
        check_all(&self.attn_norm, "attn norm")?;
        check_all(&self.attn_q, "attn q")?;
        check_all(&self.attn_k, "attn k")?;
        check_all(&self.attn_v, "attn v")?;
        check_all(&self.attn_output, "attn output")?;
        check_all(&self.attn_q_norm, "attn q norm")?;
        check_all(&self.attn_k_norm, "attn k norm")?;
        check_all(&self.ffn_norm, "ffn norm")?;
        check_all(&self.ffn_gate, "ffn gate")?;
        check_all(&self.ffn_up, "ffn up")?;
        check_all(&self.ffn_down, "ffn down")?;
        Ok(())
    }
}

fn missing(name: &str) -> TuckmillError {
    TuckmillError::new(
        ErrorCode::UnsupportedArchitecture,
        format!("artifact is missing required tensor {name:?} for dense Qwen3"),
    )
    .with_action(
        "Confirm the artifact family with 'tuckmill inspect'; tuckcore Stage 0 executes dense Qwen3 GGUF only.",
    )
}

/// Resolve the model configuration. Refuses anything that is not dense
/// Qwen3 — routing and architecture are never substituted.
pub fn resolve(
    inspection: &GgufInspection,
    registry: &AdapterRegistry,
) -> Result<(Qwen3Config, AdapterSpec)> {
    let arch = inspection.arch_id().ok_or_else(|| {
        TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            "artifact declares no architecture",
        )
    })?;
    if arch == "qwen35" {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            "qwen35 is a hybrid Gated-DeltaNet + full-attention architecture; tuckcore does not implement its operators yet",
        )
        .with_action(
            "The artifact contract (inspect/import/plan) is recognized; execution waits for the Q35-1 CPU oracle and Q35-2 Metal gates (ADR-008).",
        ));
    }
    if arch != "qwen3" {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "tuckcore Stage 0 executes dense qwen3 artifacts; this artifact is {arch:?}"
            ),
        )
        .with_action(
            "Use a dense Qwen3 GGUF, or the reference test engine for pipeline validation of other architectures.",
        ));
    }
    let adapter = registry
        .find(&arch)
        .ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!("no adapter is registered for architecture {arch:?}"),
            )
        })?
        .clone();
    if adapter.traits.moe || adapter.traits.hybrid_linear_attention {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "tuckcore Stage 0 does not execute MoE or hybrid adapters ({})",
                adapter.id
            ),
        )
        .with_action("Dense Qwen3 first; MoE streaming remains its own gated milestone (FR-06)."));
    }
    adapter.validate_metadata(inspection)?;
    let kv = adapter.kv_geometry(inspection)?;

    let hidden = inspection
        .get_u64("qwen3.embedding_length")
        .filter(|v| *v > 0)
        .ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                "qwen3.embedding_length is required",
            )
        })?;
    let ffn = inspection
        .get_u64("qwen3.feed_forward_length")
        .filter(|v| *v > 0)
        .ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                "qwen3.feed_forward_length is required",
            )
        })?;
    let rope_base = inspection
        .get_f32("qwen3.rope.freq_base")
        .unwrap_or(10_000.0);
    let rms_eps = inspection
        .get_f32("qwen3.attention.layer_norm_rms_epsilon")
        .unwrap_or(1e-6);
    let eos_token_id = inspection
        .get_u64("tokenizer.ggml.eos_token_id")
        .map(|v| v as u32);

    let token_embd = inspection
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .ok_or_else(|| missing("token_embd.weight"))?;
    if token_embd.dims.len() != 2 || token_embd.dims[0] != hidden {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "token_embd.weight has dims {:?} but embedding_length is {hidden}",
                token_embd.dims
            ),
        ));
    }
    let vocab = token_embd.dims[1];

    Ok((
        Qwen3Config {
            layers: kv.layers,
            hidden,
            heads: kv.head_count,
            kv_heads: kv.kv_head_count,
            key_length: kv.key_length,
            value_length: kv.value_length,
            ffn,
            vocab,
            rope_base,
            rms_eps,
            eos_token_id,
        },
        adapter,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    fn inspection() -> GgufInspection {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        let mut cursor = std::io::Cursor::new(bytes.clone());
        tuckmill_manifest::gguf::inspect_reader(
            &mut cursor,
            bytes.len() as u64,
            &tuckmill_manifest::gguf::Limits::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn resolves_dense_qwen3_config() {
        let insp = inspection();
        let registry = AdapterRegistry::builtin().unwrap();
        let (cfg, adapter) = resolve(&insp, &registry).unwrap();
        assert_eq!(cfg.layers, 2);
        assert_eq!(cfg.hidden, 128);
        assert_eq!(cfg.heads, 4);
        assert_eq!(cfg.kv_heads, 2);
        assert_eq!(cfg.key_length, 32);
        assert_eq!(cfg.ffn, 256);
        assert_eq!(cfg.vocab, 512);
        assert!((cfg.rope_base - 1_000_000.0).abs() < 1.0);
        assert!(cfg.eos_token_id.is_some());
        assert_eq!(adapter.id, "qwen3");

        let mut names = TensorNames::for_layers(cfg.layers);
        names.resolve(&insp).unwrap();
        assert!(names.output.is_some());
    }

    #[test]
    fn moe_artifacts_are_refused_honestly() {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3moe());
        let mut cursor = std::io::Cursor::new(bytes.clone());
        let insp = tuckmill_manifest::gguf::inspect_reader(
            &mut cursor,
            bytes.len() as u64,
            &tuckmill_manifest::gguf::Limits::default(),
            None,
        )
        .unwrap();
        let registry = AdapterRegistry::builtin().unwrap();
        let err = resolve(&insp, &registry).unwrap_err();
        assert_eq!(err.code(), ErrorCode::UnsupportedArchitecture);
        assert!(err.message().contains("qwen3moe"));
    }

    /// qwen35 artifacts are recognized at the contract level but the
    /// hybrid operators are unimplemented; the refusal names the gate.
    #[test]
    fn qwen35_artifacts_are_refused_with_the_gate_reference() {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen35());
        let mut cursor = std::io::Cursor::new(bytes.clone());
        let insp = tuckmill_manifest::gguf::inspect_reader(
            &mut cursor,
            bytes.len() as u64,
            &tuckmill_manifest::gguf::Limits::default(),
            None,
        )
        .unwrap();
        let registry = AdapterRegistry::builtin().unwrap();
        let err = resolve(&insp, &registry).unwrap_err();
        assert_eq!(err.code(), ErrorCode::UnsupportedArchitecture);
        assert!(err.message().contains("hybrid"));
        assert!(err.message().contains("does not implement its operators"));
        assert!(err.action().contains("Q35-1 CPU oracle"));
    }
}
