//! Artifact tokenization with a reference-first policy.
//!
//! Token IDs must match the selected model's intended tokenizer. llama.cpp is
//! the reference implementation; when its pinned shim is configured
//! (`TUCKMILL_LLAMA_LIB`), encoding always goes through it (vocab-only load,
//! cheap). Without it, tuckmill's built-in byte-level BPE is used only when it
//! is exact (GPT-2-style pre-tokenizers, golden-tested); any other declared
//! pre-tokenizer is refused rather than approximated.

use std::path::Path;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_tuckcore::tokenizer::Tokenizer;

pub enum ArtifactEncoder {
    Reference(Box<ReferenceArtifactEncoder>),
    Native(Box<NativeArtifactEncoder>),
}

pub struct ReferenceArtifactEncoder {
    tokenizer: tuckmill_llamaref::ReferenceTokenizer,
    note: String,
}

pub struct NativeArtifactEncoder {
    tokenizer: Tokenizer,
    note: String,
}

impl ArtifactEncoder {
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        match self {
            ArtifactEncoder::Reference(inner) => inner.tokenizer.tokenize(text, add_bos),
            ArtifactEncoder::Native(inner) => inner.tokenizer.encode(text, add_bos),
        }
    }

    pub fn note(&self) -> &str {
        match self {
            ArtifactEncoder::Reference(inner) => &inner.note,
            ArtifactEncoder::Native(inner) => &inner.note,
        }
    }

    /// Short machine-readable label for API extensions/records.
    pub fn label(&self) -> &'static str {
        match self {
            ArtifactEncoder::Reference(_) => "llama.cpp reference tokenizer (pinned revision)",
            ArtifactEncoder::Native(_) => "tuckmill byte-level BPE (gpt2-style, golden-tested)",
        }
    }

    pub fn vocab_len(&self) -> Result<usize> {
        match self {
            ArtifactEncoder::Reference(inner) => inner.tokenizer.vocab_len(),
            ArtifactEncoder::Native(inner) => Ok(inner.tokenizer.vocab_len()),
        }
    }
}

/// Open the best available encoder for an artifact.
pub fn open_encoder(model_path: &Path, lib_path: Option<&str>) -> Result<ArtifactEncoder> {
    // Reference-first when the pinned library is configured and loadable.
    if let Ok(resolved) = tuckmill_llamaref::resolve_lib_path(lib_path) {
        if resolved.exists() {
            match tuckmill_llamaref::ReferenceTokenizer::open(lib_path, model_path) {
                Ok(tokenizer) => {
                    let vocab = tokenizer.vocab_len().unwrap_or(0);
                    return Ok(ArtifactEncoder::Reference(Box::new(
                        ReferenceArtifactEncoder {
                            tokenizer,
                            note: format!(
                                "encoding via the llama.cpp reference tokenizer ({}), vocab {}",
                                tuckmill_llamaref::LLAMA_REVISION,
                                vocab
                            ),
                        },
                    )));
                }
                Err(e) => {
                    // Fall through to the native path only when it can be
                    // exact; otherwise surface the reference failure.
                    let insp = tuckmill_manifest::gguf::inspect(
                        model_path,
                        &tuckmill_manifest::gguf::Limits::default(),
                    )?;
                    let native = Tokenizer::from_gguf(&insp)?;
                    if native.encode_is_exact() {
                        let note = format!(
                            "encoding via tuckmill byte-level BPE (reference tokenizer unavailable: {})",
                            e.message()
                        );
                        return Ok(ArtifactEncoder::Native(Box::new(NativeArtifactEncoder {
                            tokenizer: native,
                            note,
                        })));
                    }
                    return Err(TuckmillError::new(
                        e.code(),
                        format!(
                            "exact tokenization requires the llama.cpp reference tokenizer, which failed to load: {}",
                            e.message()
                        ),
                    )
                    .with_action(
                        "Run scripts/fetch-llama.sh and set TUCKMILL_LLAMA_LIB; this build will not approximate a non-GPT-2 pre-tokenizer.",
                    ));
                }
            }
        }
    }
    let insp =
        tuckmill_manifest::gguf::inspect(model_path, &tuckmill_manifest::gguf::Limits::default())?;
    let tokenizer = Tokenizer::from_gguf(&insp)?;
    if !tokenizer.encode_is_exact() {
        return Err(TuckmillError::new(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "artifact declares pre-tokenizer {:?}; tuckmill's built-in tokenizer is exact only for GPT-2-style splits",
                tokenizer.pretokenizer
            ),
        )
        .with_action(
            "Run scripts/fetch-llama.sh and set TUCKMILL_LLAMA_LIB so encoding uses the pinned llama.cpp reference tokenizer.",
        ));
    }
    let note = format!(
        "encoding via tuckmill byte-level BPE (gpt2-style; validated against llama.cpp goldens), vocab {}",
        insp.vocabulary_len().unwrap_or(0)
    );
    Ok(ArtifactEncoder::Native(Box::new(NativeArtifactEncoder {
        tokenizer,
        note,
    })))
}
