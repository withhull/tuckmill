//! Synthetic GGUF fixtures.
//!
//! These builders produce *valid* GGUF files with tiny but real shapes. They
//! prove parser, planner, packer, and pipeline code paths. They are never
//! evidence that a real oversized model will behave the same way (PRD §18.1);
//! tests that use them must say so.

use std::path::Path;

use crate::error::Result;

/// A tiny Qwen3-shaped (or Qwen3-MoE-shaped) fixture configuration.
#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    pub arch: String,
    pub name: String,
    pub layers: u32,
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub ffn: u32,
    pub vocab: u32,
    pub context: u64,
    pub expert_count: Option<u32>,
    pub expert_used: Option<u32>,
    /// Store expert tensors as Q4_0 to exercise quantized size math.
    pub quantize_experts: bool,
}

impl SyntheticConfig {
    pub fn tiny_qwen3() -> Self {
        Self {
            arch: "qwen3".into(),
            name: "tuckmill-fixture-qwen3-tiny".into(),
            layers: 2,
            hidden: 128,
            heads: 4,
            kv_heads: 2,
            head_dim: 32,
            ffn: 256,
            vocab: 512,
            context: 4096,
            expert_count: None,
            expert_used: None,
            quantize_experts: false,
        }
    }

    pub fn tiny_qwen3moe() -> Self {
        Self {
            arch: "qwen3moe".into(),
            name: "tuckmill-fixture-qwen3moe-tiny".into(),
            layers: 2,
            hidden: 128,
            heads: 4,
            kv_heads: 2,
            head_dim: 32,
            ffn: 64,
            vocab: 512,
            context: 4096,
            expert_count: Some(8),
            expert_used: Some(2),
            quantize_experts: true,
        }
    }
}

const ALIGNMENT: u64 = 32;

struct Tensor {
    name: String,
    dims: Vec<u64>,
    type_id: u32,
    bytes: u64,
}

fn tensor(name: &str, dims: &[u64], type_id: u32) -> Result<Tensor> {
    let elems = dims
        .iter()
        .try_fold(1u64, |a, d| a.checked_mul(*d))
        .unwrap_or(0);
    Ok(Tensor {
        name: name.to_string(),
        dims: dims.to_vec(),
        type_id,
        bytes: crate::ggml::tensor_bytes(type_id, elems)?,
    })
}

/// Build a valid synthetic GGUF as bytes.
pub fn synthetic_qwen3_bytes(cfg: &SyntheticConfig) -> Vec<u8> {
    build(cfg).expect("synthetic fixture configuration is internally valid")
}

/// Write a synthetic GGUF to a path; returns the file length.
pub fn write_synthetic(path: &Path, cfg: &SyntheticConfig) -> Result<u64> {
    let bytes = build(cfg)?;
    std::fs::write(path, &bytes).map_err(crate::error::TuckmillError::from)?;
    Ok(bytes.len() as u64)
}

fn build(cfg: &SyntheticConfig) -> Result<Vec<u8>> {
    // --- tensor list ---
    let mut tensors: Vec<Tensor> = Vec::new();
    let is_moe = cfg.arch == "qwen3moe" || cfg.expert_count.is_some();
    tensors.push(tensor(
        "token_embd.weight",
        &[cfg.hidden as u64, cfg.vocab as u64],
        1,
    )?);
    tensors.push(tensor("output_norm.weight", &[cfg.hidden as u64], 0)?);
    tensors.push(tensor(
        "output.weight",
        &[cfg.hidden as u64, cfg.vocab as u64],
        1,
    )?);
    let q_out = (cfg.heads * cfg.head_dim) as u64;
    let kv_out = (cfg.kv_heads * cfg.head_dim) as u64;
    for l in 0..cfg.layers {
        let p = |s: &str| format!("blk.{l}.{s}");
        tensors.push(tensor(&p("attn_norm.weight"), &[cfg.hidden as u64], 0)?);
        // GGUF stores dims fastest-first: for a linear (out x in) the dims
        // are [in, out] — matching llama.cpp's create_tensor convention.
        tensors.push(tensor(&p("attn_q.weight"), &[cfg.hidden as u64, q_out], 1)?);
        tensors.push(tensor(
            &p("attn_k.weight"),
            &[cfg.hidden as u64, kv_out],
            1,
        )?);
        tensors.push(tensor(
            &p("attn_v.weight"),
            &[cfg.hidden as u64, kv_out],
            1,
        )?);
        tensors.push(tensor(
            &p("attn_output.weight"),
            &[q_out, cfg.hidden as u64],
            1,
        )?);
        tensors.push(tensor(&p("attn_q_norm.weight"), &[cfg.head_dim as u64], 0)?);
        tensors.push(tensor(&p("attn_k_norm.weight"), &[cfg.head_dim as u64], 0)?);
        tensors.push(tensor(&p("ffn_norm.weight"), &[cfg.hidden as u64], 0)?);
        if is_moe {
            let e = cfg.expert_count.unwrap_or(8) as u64;
            let f = cfg.ffn as u64;
            let h = cfg.hidden as u64;
            let expert_ty = if cfg.quantize_experts { 2 } else { 1 };
            tensors.push(tensor(&p("ffn_gate_inp.weight"), &[h, e], 0)?);
            tensors.push(tensor(&p("ffn_gate_exps.weight"), &[h, f, e], expert_ty)?);
            tensors.push(tensor(&p("ffn_up_exps.weight"), &[h, f, e], expert_ty)?);
            tensors.push(tensor(&p("ffn_down_exps.weight"), &[f, h, e], expert_ty)?);
            tensors.push(tensor(&p("ffn_gate_shexp.weight"), &[h, f], 1)?);
        } else {
            let f = cfg.ffn as u64;
            let h = cfg.hidden as u64;
            tensors.push(tensor(&p("ffn_gate.weight"), &[h, f], 1)?);
            tensors.push(tensor(&p("ffn_up.weight"), &[h, f], 1)?);
            tensors.push(tensor(&p("ffn_down.weight"), &[f, h], 1)?);
        }
    }

    // --- data offsets (32-byte aligned, relative to data start) ---
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cursor: u64 = 0;
    for t in &tensors {
        let aligned = (cursor + ALIGNMENT - 1) & !(ALIGNMENT - 1);
        offsets.push(aligned);
        cursor = aligned + t.bytes;
    }
    let data_len = cursor;

    // --- header ---
    let mut out: Vec<u8> = Vec::with_capacity((cursor + 64 * 1024) as usize);
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());

    let mut kvs: Vec<(String, KV)> = Vec::new();
    let arch = &cfg.arch;
    kvs.push(("general.architecture".into(), KV::Str(arch.clone())));
    kvs.push(("general.name".into(), KV::Str(cfg.name.clone())));
    kvs.push(("general.alignment".into(), KV::U32(ALIGNMENT as u32)));
    kvs.push((
        "general.file_type".into(),
        KV::U32(if is_moe { 2 } else { 1 }),
    ));
    kvs.push(("general.quantization_version".into(), KV::U32(2)));
    kvs.push((format!("{arch}.block_count"), KV::U32(cfg.layers)));
    kvs.push((
        format!("{arch}.context_length"),
        KV::U32(cfg.context as u32),
    ));
    kvs.push((format!("{arch}.embedding_length"), KV::U32(cfg.hidden)));
    kvs.push((format!("{arch}.attention.head_count"), KV::U32(cfg.heads)));
    kvs.push((
        format!("{arch}.attention.head_count_kv"),
        KV::U32(cfg.kv_heads),
    ));
    kvs.push((
        format!("{arch}.attention.key_length"),
        KV::U32(cfg.head_dim),
    ));
    kvs.push((
        format!("{arch}.attention.value_length"),
        KV::U32(cfg.head_dim),
    ));
    kvs.push((format!("{arch}.feed_forward_length"), KV::U32(cfg.ffn)));
    kvs.push((format!("{arch}.rope.freq_base"), KV::F32(1_000_000.0)));
    kvs.push((
        format!("{arch}.attention.layer_norm_rms_epsilon"),
        KV::F32(1e-6),
    ));
    if let (Some(ec), Some(eu)) = (cfg.expert_count, cfg.expert_used) {
        kvs.push((format!("{arch}.expert_count"), KV::U32(ec)));
        kvs.push((format!("{arch}.expert_used_count"), KV::U32(eu)));
    }
    kvs.push(("tokenizer.ggml.model".into(), KV::Str("gpt2".to_string())));

    // Byte-level vocabulary: 256 byte pieces (GPT-2 alphabet, type 6), a
    // small deterministic merge table so BPE merging is exercised, filler
    // unused pieces, and control tokens. This makes the fixture tokenizer
    // *real*: text round-trips exactly and llama.cpp-style artifacts behave
    // the same way.
    let alphabet = crate::bytes_alphabet::bytes_to_unicode();
    let mut tokens: Vec<KV> = alphabet.iter().map(|c| KV::Str(c.to_string())).collect();
    let mut token_types: Vec<KV> = (0..256).map(|_| KV::I32(6)).collect();

    // Merge table (rank order matters): enough to compress "hello world"
    // into two committed pieces and to test multi-step merges.
    let space = alphabet[0x20];
    let merge_table: [(&str, &str, String); 10] = [
        ("h", "e", "he".to_string()),
        ("l", "l", "ll".to_string()),
        ("he", "ll", "hell".to_string()),
        ("hell", "o", "hello".to_string()),
        (&space.to_string(), "w", format!("{space}w")),
        ("w", "o", "wo".to_string()),
        ("wo", "r", "wor".to_string()),
        ("wor", "l", "worl".to_string()),
        ("worl", "d", "world".to_string()),
        (&space.to_string(), "world", format!("{space}world")),
    ];
    let mut merges: Vec<KV> = Vec::new();
    for (left, right, result) in merge_table {
        merges.push(KV::Str(format!("{left} {right}")));
        tokens.push(KV::Str(result));
        token_types.push(KV::I32(1));
    }
    while tokens.len() < cfg.vocab as usize - 3 {
        let i = tokens.len();
        tokens.push(KV::Str(format!("unused-{i}")));
        token_types.push(KV::I32(5));
    }
    tokens.push(KV::Str("<|bos|>".into()));
    token_types.push(KV::I32(3));
    tokens.push(KV::Str("<|eos|>".into()));
    token_types.push(KV::I32(3));
    tokens.push(KV::Str("<|pad|>".into()));
    token_types.push(KV::I32(3));
    let bos_id = (cfg.vocab - 3) as u32;
    let eos_id = (cfg.vocab - 2) as u32;
    let pad_id = (cfg.vocab - 1) as u32;

    kvs.push(("tokenizer.ggml.tokens".into(), KV::Array(8, tokens)));
    kvs.push((
        "tokenizer.ggml.token_type".into(),
        KV::Array(5, token_types),
    ));
    kvs.push(("tokenizer.ggml.merges".into(), KV::Array(8, merges)));
    kvs.push(("tokenizer.ggml.eos_token_id".into(), KV::U32(eos_id)));
    kvs.push(("tokenizer.ggml.bos_token_id".into(), KV::U32(bos_id)));
    kvs.push(("tokenizer.ggml.padding_token_id".into(), KV::U32(pad_id)));

    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (key, value) in &kvs {
        write_string(&mut out, key);
        write_kv_value(&mut out, value);
    }

    // --- tensor infos ---
    for (t, offset) in tensors.iter().zip(&offsets) {
        write_string(&mut out, &t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.type_id.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }

    // --- data section: deterministic, FINITE values per dtype ---
    //
    // The fixture proves code paths and numeric *equivalence* between
    // scheduling configurations. Raw pseudo-random bytes would decode to
    // NaN/inf half-floats and make numeric tests meaningless, so values are
    // encoded per dtype into a small, well-behaved range.
    pad_to_alignment(&mut out, ALIGNMENT);
    let base = out.len() as u64;
    out.resize((base + data_len) as usize, 0);
    for (t, offset) in tensors.iter().zip(&offsets) {
        write_tensor_values(&mut out, base + offset, t);
    }
    Ok(out)
}

/// Deterministic fixture value in [-0.5, 0.5): finite, non-degenerate.
fn fixture_value(elem: u64) -> f32 {
    (((elem * 31 + 7) % 64) as f32 - 32.0) / 64.0
}

/// IEEE binary16 encoding (round-to-nearest-even) for fixture data.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mut mant = bits & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow to inf (not reachable for fixture values)
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        mant |= 0x80_0000;
        let shift = (14 - exp) as u32;
        let half_mant = mant >> shift;
        return sign | half_mant as u16;
    }
    // Round-to-nearest-even on the 13 dropped bits.
    let round = (mant >> 13) & 1;
    let sticky = mant & 0x1fff;
    if round == 1 && (sticky != 0 || ((mant >> 13) & 1) == 1) {
        mant += 1 << 13;
        if mant & 0x80_0000 != 0 {
            mant = 0;
            exp += 1;
        }
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn write_tensor_values(out: &mut [u8], start: u64, t: &Tensor) {
    let elems: u64 = t.dims.iter().product();
    let start = start as usize;
    match t.type_id {
        0 => {
            for e in 0..elems {
                let v = fixture_value(e);
                out[start + (e as usize) * 4..start + (e as usize) * 4 + 4]
                    .copy_from_slice(&v.to_le_bytes());
            }
        }
        1 => {
            for e in 0..elems {
                let half = f32_to_f16_bits(fixture_value(e));
                out[start + (e as usize) * 2..start + (e as usize) * 2 + 2]
                    .copy_from_slice(&half.to_le_bytes());
            }
        }
        2 => {
            // Q4_0: [f16 d][16 nibbles]; d = 1/32 keeps values in [-0.5, 0.5).
            let d = f32_to_f16_bits(0.03125);
            let blocks = elems / 32;
            for b in 0..blocks {
                let bo = start + (b as usize) * 18;
                out[bo..bo + 2].copy_from_slice(&d.to_le_bytes());
                for j in 0..16u64 {
                    let lo = (((b * 32 + j) * 7 + 3) % 16) as u8;
                    let hi = (((b * 32 + j + 16) * 7 + 3) % 16) as u8;
                    out[bo + 2 + j as usize] = lo | (hi << 4);
                }
            }
        }
        8 => {
            // Q8_0: [f16 d][32 i8]; d = 1/128 keeps values in [-0.5, 0.5).
            let d = f32_to_f16_bits(0.0078125);
            let blocks = elems / 32;
            for b in 0..blocks {
                let bo = start + (b as usize) * 34;
                out[bo..bo + 2].copy_from_slice(&d.to_le_bytes());
                for j in 0..32u64 {
                    let q = (((b * 32 + j) * 11 + 5) % 128) as i32 - 64;
                    out[bo + 2 + j as usize] = (q as i8) as u8;
                }
            }
        }
        _ => {} // other types are not used by fixtures; region stays zeroed
    }
}

enum KV {
    U32(u32),
    I32(i32),
    F32(f32),
    Str(String),
    Array(u32, Vec<KV>),
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_kv_value(out: &mut Vec<u8>, v: &KV) {
    match v {
        KV::U32(x) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        KV::I32(x) => {
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        KV::F32(x) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        KV::Str(s) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            write_string(out, s);
        }
        KV::Array(item_type, items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&item_type.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            // Array items carry no type tag; the item type was declared once.
            for item in items {
                write_value_payload(out, item);
            }
        }
    }
}

/// Write a value payload with no type tag (used for array items).
fn write_value_payload(out: &mut Vec<u8>, v: &KV) {
    match v {
        KV::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        KV::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        KV::F32(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
        KV::Str(s) => write_string(out, s),
        KV::Array(..) => unreachable!("nested arrays are not used in fixtures"),
    }
}

fn pad_to_alignment(out: &mut Vec<u8>, alignment: u64) {
    let rem = out.len() as u64 % alignment;
    if rem != 0 {
        out.resize(out.len() + (alignment - rem) as usize, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{inspect_reader, Limits};

    #[test]
    fn fixture_parses_and_validates() {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        let mut cur = std::io::Cursor::new(bytes.clone());
        let insp = inspect_reader(&mut cur, bytes.len() as u64, &Limits::default(), None).unwrap();
        assert_eq!(insp.arch_id().as_deref(), Some("qwen3"));
        assert!(insp.layout_warnings.is_empty());
        assert!(insp
            .tensors
            .iter()
            .any(|t| t.name == "blk.1.ffn_down.weight"));
        assert!(!insp.tensors.iter().any(|t| t.name.contains("_exps")));
    }

    #[test]
    fn moe_fixture_has_experts() {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3moe());
        let mut cur = std::io::Cursor::new(bytes.clone());
        let insp = inspect_reader(&mut cur, bytes.len() as u64, &Limits::default(), None).unwrap();
        assert_eq!(insp.arch_id().as_deref(), Some("qwen3moe"));
        assert_eq!(insp.get_u64("qwen3moe.expert_count"), Some(8));
        let stats = insp.tensor_stats();
        assert!(stats.by_class.contains_key("moe_expert"));
        assert!(
            stats.by_type.contains_key("Q4_0"),
            "experts quantized in fixture"
        );
    }

    #[test]
    fn fixture_bytes_are_deterministic() {
        let a = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        let b = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        assert_eq!(a, b);
    }

    #[test]
    fn write_synthetic_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture.gguf");
        let len = write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), len);
    }
}
