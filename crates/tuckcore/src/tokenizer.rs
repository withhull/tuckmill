//! Byte-level BPE tokenizer for GGUF artifacts (`tokenizer.ggml.model` =
//! "gpt2"), as used by GPT-2, Qwen, and most modern GGUF releases.
//!
//! Encoding: GPT-2 pre-tokenization split, byte→unicode alphabet, merge-rank
//! BPE, byte-coverage fallback. Decoding: exact reverse over non-control
//! tokens, lossy UTF-8 assembly.
//!
//! Honest scope (claims ledger): this is tuckmill's own implementation,
//! validated by round-trip and structural tests plus fixture goldens. Full
//! equivalence with llama.cpp's tokenizer (including architecture-specific
//! pre-tokenizers such as `tokenizer.ggml.pre = "qwen2"`) is a separately
//! gated milestone; when an artifact declares a non-GPT-2 pre-tokenizer this
//! type records it and callers surface the limitation instead of hiding it.

use std::collections::HashMap;

use tuckmill_manifest::bytes_alphabet::bytes_to_unicode;
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::gguf::{GgufInspection, GgufValue};

/// GGUF `tokenizer.ggml.token_type` values (llama.cpp convention).
pub const TOKEN_TYPE_NORMAL: u8 = 1;
pub const TOKEN_TYPE_UNKNOWN: u8 = 2;
pub const TOKEN_TYPE_CONTROL: u8 = 3;
pub const TOKEN_TYPE_USER_DEFINED: u8 = 4;
pub const TOKEN_TYPE_UNUSED: u8 = 5;
pub const TOKEN_TYPE_BYTE: u8 = 6;

const MAX_VOCAB: usize = 1_048_576;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
    pub unk: Option<u32>,
}

#[derive(Debug)]
pub struct Tokenizer {
    tokens: Vec<String>,
    token_types: Vec<u8>,
    /// (left, right) -> rank; lower rank merges first.
    merges: HashMap<(String, String), u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub special: SpecialTokens,
    /// `tokenizer.ggml.pre` as declared ("gpt2" when absent).
    pub pretokenizer: String,
    /// True when all 256 byte pieces exist (encoding can never fail).
    pub byte_coverage: bool,
}

impl Tokenizer {
    /// Build from a GGUF inspection. Only `model = "gpt2"` (byte-level BPE)
    /// is supported in this build; anything else is a typed refusal.
    pub fn from_gguf(inspection: &GgufInspection) -> Result<Self> {
        let model = inspection
            .get_str("tokenizer.ggml.model")
            .unwrap_or("")
            .to_string();
        if model != "gpt2" {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "tokenizer model {model:?} is not supported in this build (byte-level BPE \"gpt2\" only); no text was tokenized"
                ),
            )
            .with_action("Use an artifact with a gpt2-style tokenizer, or the synthetic engine for pipeline runs."));
        }

        let tokens_value = inspection.get("tokenizer.ggml.tokens").ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::InvalidArtifact,
                "artifact embeds no tokenizer.ggml.tokens",
            )
        })?;
        let items = tokens_value.as_array().ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::InvalidArtifact,
                "tokenizer.ggml.tokens is not an array",
            )
        })?;
        if items.is_empty() || items.len() > MAX_VOCAB {
            return Err(TuckmillError::new(
                ErrorCode::InvalidArtifact,
                format!("vocabulary size {} is outside 1..={MAX_VOCAB}", items.len()),
            ));
        }
        let mut tokens = Vec::with_capacity(items.len());
        for item in items {
            let s = item.as_str().ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    "tokenizer.ggml.tokens contains a non-string entry",
                )
            })?;
            tokens.push(s.to_string());
        }

        let mut token_types = vec![TOKEN_TYPE_NORMAL; tokens.len()];
        if let Some(types) = inspection
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
        {
            if types.len() != tokens.len() {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!(
                        "tokenizer.ggml.token_type has {} entries for {} tokens",
                        types.len(),
                        tokens.len()
                    ),
                ));
            }
            for (i, t) in types.iter().enumerate() {
                if let Some(v) = t.as_u64() {
                    token_types[i] = v as u8;
                }
            }
        }

        let mut merges = HashMap::new();
        if let Some(list) = inspection
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
        {
            for (rank, item) in list.iter().enumerate() {
                let Some(s) = item.as_str() else {
                    return Err(TuckmillError::new(
                        ErrorCode::InvalidArtifact,
                        "tokenizer.ggml.merges contains a non-string entry",
                    ));
                };
                if let Some((left, right)) = s.split_once(' ') {
                    merges
                        .entry((left.to_string(), right.to_string()))
                        .or_insert(rank as u32);
                }
            }
        }

        let byte_to_char = bytes_to_unicode();
        let char_to_byte: HashMap<char, u8> = byte_to_char
            .iter()
            .enumerate()
            .map(|(b, c)| (*c, b as u8))
            .collect();
        let byte_coverage = byte_to_char
            .iter()
            .all(|c| tokens.iter().any(|t| t == &c.to_string()));

        let special = SpecialTokens {
            bos: meta_u32(inspection, "tokenizer.ggml.bos_token_id"),
            eos: meta_u32(inspection, "tokenizer.ggml.eos_token_id"),
            pad: meta_u32(inspection, "tokenizer.ggml.padding_token_id"),
            unk: meta_u32(inspection, "tokenizer.ggml.unknown_token_id"),
        };
        let pretokenizer = inspection
            .get_str("tokenizer.ggml.pre")
            .unwrap_or("gpt2")
            .to_string();

        Ok(Self {
            tokens,
            token_types,
            merges,
            byte_to_char,
            char_to_byte,
            special,
            pretokenizer,
            byte_coverage,
        })
    }

    pub fn vocab_len(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_type(&self, id: u32) -> Option<u8> {
        self.token_types.get(id as usize).copied()
    }

    pub fn is_control(&self, id: u32) -> bool {
        self.token_type(id) == Some(TOKEN_TYPE_CONTROL)
    }

    /// True when this tokenizer's own encoding is exact for the artifact's
    /// declared pre-tokenizer (GPT-2-style splits only).
    pub fn encode_is_exact(&self) -> bool {
        matches!(self.pretokenizer.as_str(), "gpt2" | "gpt-2")
    }

    /// Disclosed limitation when encoding would not be exact: callers must
    /// use the llama.cpp reference tokenizer (TUCKMILL_LLAMA_LIB) instead.
    pub fn pretokenizer_note(&self) -> Option<String> {
        if self.encode_is_exact() {
            None
        } else {
            Some(format!(
                "artifact declares pre-tokenizer {:?}; exact encoding requires the llama.cpp reference tokenizer (TUCKMILL_LLAMA_LIB), which this build will use in preference",
                self.pretokenizer
            ))
        }
    }

    /// Encode text. `add_bos` prepends the BOS piece when the tokenizer
    /// declares one. Encoding is exact: with byte coverage every input maps,
    /// and the byte-coverage requirement is checked up front rather than
    /// silently substituting unknowns.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        if !self.encode_is_exact() {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!(
                    "artifact declares pre-tokenizer {:?}; tuckmill's built-in tokenizer only reproduces GPT-2-style splits exactly and will not approximate this one (token IDs must match the model's intended tokenizer)",
                    self.pretokenizer
                ),
            )
            .with_action(
                "Set TUCKMILL_LLAMA_LIB (scripts/fetch-llama.sh) so encoding uses the pinned llama.cpp reference tokenizer.",
            ));
        }
        if !self.byte_coverage {
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                "artifact tokenizer lacks full byte coverage; encoding would require substitutions, which are never silent",
            )
            .with_action("Use an artifact with a complete byte-level vocabulary."));
        }
        let mut piece_ids: HashMap<&str, u32> = HashMap::with_capacity(self.tokens.len());
        for (i, t) in self.tokens.iter().enumerate() {
            piece_ids.entry(t.as_str()).or_insert(i as u32);
        }
        let mut ids = Vec::new();
        if add_bos {
            if let Some(bos) = self.special.bos {
                ids.push(bos);
            }
        }
        for segment in pretokenize(text) {
            // Byte-level mapping then BPE merges.
            let mapped: String = segment
                .bytes()
                .map(|b| self.byte_to_char[b as usize])
                .collect();
            for piece in self.bpe(mapped) {
                let id = piece_ids.get(piece.as_str()).ok_or_else(|| {
                    TuckmillError::new(
                        ErrorCode::IntegrityFailure,
                        format!(
                            "BPE produced piece {piece:?} that is absent from the vocabulary; the tokenizer tables are inconsistent"
                        ),
                    )
                })?;
                ids.push(*id);
            }
        }
        Ok(ids)
    }

    /// Greedy merge-rank BPE over one pre-tokenized segment (standard
    /// algorithm: repeatedly merge the lowest-rank adjacent pair).
    fn bpe(&self, piece: String) -> Vec<String> {
        if self.merges.is_empty() {
            return piece.chars().map(|c| c.to_string()).collect();
        }
        // Represent as a list of owned symbols; naive rescan is fine for
        // pre-tokenizer-scale segments.
        let mut symbols: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(rank) = self
                    .merges
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    match best {
                        Some((_, r)) if *rank >= r => {}
                        _ => best = Some((i, *rank)),
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..i + 2, [merged]);
        }
        symbols
    }

    /// Decode committed tokens to text. Control tokens are skipped; bytes are
    /// re-assembled exactly and invalid UTF-8 boundaries become U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 2);
        for id in ids {
            let Some(token) = self.tokens.get(*id as usize) else {
                continue;
            };
            if self.is_control(*id) {
                continue;
            }
            for c in token.chars() {
                if let Some(b) = self.char_to_byte.get(&c) {
                    bytes.push(*b);
                } else {
                    // Not a byte-alphabet char (special/user token string):
                    // keep its UTF-8 bytes verbatim.
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a single committed token (streaming path).
    pub fn decode_token(&self, id: u32) -> String {
        self.decode(&[id])
    }

    /// Exact bytes for one token id (control tokens yield empty bytes); feed
    /// to [`Utf8StreamDecoder`] for streaming text assembly.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        let Some(token) = self.tokens.get(id as usize) else {
            return bytes;
        };
        if self.is_control(id) {
            return bytes;
        }
        for c in token.chars() {
            if let Some(b) = self.char_to_byte.get(&c) {
                bytes.push(*b);
            } else {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
        bytes
    }

    /// Raw stored string for a token id (diagnostics, "unused" checks).
    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }
}

fn meta_u32(inspection: &GgufInspection, key: &str) -> Option<u32> {
    match inspection.get(key) {
        Some(GgufValue::U32(v)) => Some(*v),
        Some(v) => v.as_u64().map(|v| v as u32),
        None => None,
    }
}

/// Stateful incremental UTF-8 decoder (shared implementation).
pub use tuckmill_manifest::utf8::Utf8StreamDecoder;

/// GPT-2 pre-tokenization. Implemented manually (no regex dependency) to the
/// canonical pattern:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
/// with case-insensitive contractions. The whitespace rule is implemented
/// exactly: a whitespace run followed by a non-space yields run-minus-last as
/// one segment, with the final space attaching to the next segment.
pub fn pretokenize(text: &str) -> Vec<String> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        // Contractions (case-insensitive).
        if c == '\'' {
            let rest: String = chars[i..].iter().take(3).collect::<String>().to_lowercase();
            let mut matched = None;
            for cont in CONTRACTIONS.iter().rev() {
                if rest.starts_with(cont) {
                    matched = Some(cont.chars().count());
                    break;
                }
            }
            if let Some(len) = matched {
                out.push(chars[i..i + len].iter().collect());
                i += len;
                continue;
            }
        }
        let classify = |c: char| -> u8 {
            if c.is_alphabetic() {
                0
            } else if c.is_numeric() {
                1
            } else if !c.is_whitespace() {
                2
            } else {
                255
            }
        };
        if c.is_whitespace() {
            // `\s+(?!\S)|\s+` with the ` ?` prefix of the following class:
            // a run followed by a non-space yields run-minus-last as one
            // segment, and the final space then attaches to the next segment.
            let mut end = i;
            while end < chars.len() && chars[end].is_whitespace() {
                end += 1;
            }
            if end == chars.len() {
                out.push(chars[i..end].iter().collect());
                i = end;
            } else if end - i >= 2 {
                out.push(chars[i..end - 1].iter().collect());
                i = end - 1; // last space is handled by the ` ?` case below
            } else {
                // Exactly one space before a non-space: ` ?<class>+`.
                let class = classify(chars[end]);
                let mut run_end = end;
                if class != 255 {
                    while run_end < chars.len() && classify(chars[run_end]) == class {
                        run_end += 1;
                    }
                } else {
                    run_end = end + 1;
                }
                out.push(chars[i..run_end].iter().collect());
                i = run_end;
            }
            continue;
        }
        // Letters / numbers / other runs.
        let class = classify(c);
        if class != 255 {
            let mut end = i;
            while end < chars.len() && classify(chars[end]) == class {
                end += 1;
            }
            out.push(chars[i..end].iter().collect());
            i = end;
            continue;
        }
        // Fallback: single character.
        out.push(chars[i..i + 1].iter().collect());
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    fn fixture_tokenizer() -> Tokenizer {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        let mut cursor = std::io::Cursor::new(bytes.clone());
        let insp = tuckmill_manifest::gguf::inspect_reader(
            &mut cursor,
            bytes.len() as u64,
            &tuckmill_manifest::gguf::Limits::default(),
            None,
        )
        .unwrap();
        Tokenizer::from_gguf(&insp).unwrap()
    }

    #[test]
    fn fixture_vocab_has_byte_coverage() {
        let t = fixture_tokenizer();
        assert!(t.byte_coverage);
        assert_eq!(t.vocab_len(), 512);
        assert!(t.pretokenizer_note().is_none());
        assert_eq!(t.special.eos, Some(510));
        assert_eq!(t.token_type(0), Some(TOKEN_TYPE_NORMAL));
        assert_eq!(t.token_type(510), Some(TOKEN_TYPE_CONTROL));
    }

    #[test]
    fn encode_decode_roundtrip_is_exact() {
        let t = fixture_tokenizer();
        for text in [
            "hello world",
            "Hello, World! 123",
            "don't stop",
            "multi   spaces\nand\tnewlines",
            "unicode: héllo wörld 日本語",
        ] {
            let ids = t.encode(text, false).unwrap();
            assert!(!ids.is_empty(), "{text:?}");
            let back = t.decode(&ids);
            assert_eq!(back, text, "roundtrip for {text:?}");
        }
    }

    #[test]
    fn merges_are_applied_when_present() {
        let t = fixture_tokenizer();
        let merged = t.encode("hello", false).unwrap();
        let split: usize = "hello".chars().count();
        assert!(
            merged.len() < split,
            "fixture merges should shorten 'hello': {merged:?}"
        );
        assert_eq!(t.decode(&merged), "hello");
    }

    #[test]
    fn pretokenizer_splits_follow_the_gpt2_rules() {
        assert_eq!(pretokenize("hello world"), vec!["hello", " world"]);
        assert_eq!(pretokenize("don't"), vec!["don", "'t"]);
        assert_eq!(pretokenize("a  b"), vec!["a", " ", " b"]);
        assert_eq!(pretokenize("123 456"), vec!["123", " 456"]);
        assert_eq!(pretokenize("!!!"), vec!["!!!"]);
        assert_eq!(pretokenize("trailing  "), vec!["trailing", "  "]);
        assert_eq!(pretokenize("Don't"), vec!["Don", "'t"]);
    }

    #[test]
    fn bos_is_prepended_on_request() {
        let t = fixture_tokenizer();
        let ids = t.encode("hi", true).unwrap();
        assert_eq!(ids[0], t.special.bos.unwrap());
        let ids = t.encode("hi", false).unwrap();
        assert_ne!(ids[0], t.special.bos.unwrap());
    }

    #[test]
    fn control_tokens_are_skipped_in_decode() {
        let t = fixture_tokenizer();
        let text_id = t.encode("x", false).unwrap()[0];
        let decoded = t.decode(&[t.special.bos.unwrap(), text_id, t.special.eos.unwrap()]);
        assert_eq!(decoded, "x");
    }

    #[test]
    fn token_bytes_match_decode_for_byte_tokens() {
        let t = fixture_tokenizer();
        let ids = t.encode("héllo", false).unwrap();
        let mut bytes = Vec::new();
        for id in &ids {
            bytes.extend_from_slice(&t.token_bytes(*id));
        }
        assert_eq!(String::from_utf8(bytes).unwrap(), "héllo");
    }

    #[test]
    fn non_gpt2_tokenizer_models_are_refused() {
        // Handcraft an inspection-like check via the fixture then mutate is
        // not possible; instead verify the error path with a fresh header.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        let kv = |out: &mut Vec<u8>, key: &str, value: &str| {
            out.extend_from_slice(&(key.len() as u64).to_le_bytes());
            out.extend_from_slice(key.as_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(value.len() as u64).to_le_bytes());
            out.extend_from_slice(value.as_bytes());
        };
        kv(&mut bytes, "tokenizer.ggml.model", "sentencepiece");
        kv(&mut bytes, "general.architecture", "llama");
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        let len = bytes.len() as u64;
        let mut cursor = std::io::Cursor::new(bytes);
        let insp = tuckmill_manifest::gguf::inspect_reader(
            &mut cursor,
            len,
            &tuckmill_manifest::gguf::Limits::default(),
            None,
        )
        .unwrap();
        let err = Tokenizer::from_gguf(&insp).unwrap_err();
        assert_eq!(err.code(), ErrorCode::UnsupportedArchitecture);
    }
}
