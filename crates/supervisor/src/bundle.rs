//! Model bundles and typed request specifications (Gate B, B1).
//!
//! A `ModelBundle` is the qualified input contract for one artifact: the
//! artifact's tokenizer (reference-first, exact-or-refuse) plus its chat
//! template, rendered by a *restricted* renderer that implements the
//! recognized template's canonical semantics and refuses every unrecognized
//! template rather than approximating it. A `RequestSpec` is the typed
//! result of turning a request into token ids under that contract: chat
//! prompts are rendered with the model's template, raw prompts are not
//! (chat and raw completion stay distinct), and both are tokenized with
//! special-token spellings mapped to their real ids exactly as llama.cpp
//! does with `parse_special`.
//!
//! The Qwen3 renderer implements the semantics of the template embedded in
//! the pinned artifact, under the standard Jinja2 rules the template was
//! authored for (Hugging Face applies it with Jinja2). llama.cpp's bundled
//! jinja deviates on the trailing-assistant reasoning path; that divergence
//! is recorded in the goldens file and deliberately not reproduced — the
//! model's intended rendering is the template's own semantics.

use std::path::Path;

use tuckmill_backend::SamplingPolicy;
use tuckmill_manifest::chat::{
    QWEN35_CHAT_TEMPLATE_0_8B, QWEN35_CHAT_TEMPLATE_2B, QWEN35_THINK_END, QWEN35_THINK_START,
    QWEN3_CHAT_TEMPLATE, QWEN3_IM_END, QWEN3_IM_START,
};
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ids::Id;
use tuckmill_manifest::records::ModelArtifact;

use crate::tokenization::{open_encoder, ArtifactEncoder};

/// A chat message in the supported subset: one of the three Qwen3 roles
/// with string content. Tool calls, tool responses and separate reasoning
/// fields are not part of the tuckmill request contract and are refused
/// upstream, never synthesized here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessageSpec {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn parse(role: &str) -> Option<Self> {
        match role {
            "system" => Some(ChatRole::System),
            "user" => Some(ChatRole::User),
            "assistant" => Some(ChatRole::Assistant),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}

/// Recognized chat templates. Recognition is byte-exact against the pinned
/// template revision; an unknown template is refused, never approximated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    Qwen3,
    /// Qwen3.5 (both pinned revisions: 0.8B and 2B).
    Qwen35,
}

impl ChatTemplate {
    /// Recognize a declared `tokenizer.chat_template` value.
    pub fn recognize(declared: &str) -> Option<Self> {
        if declared == QWEN3_CHAT_TEMPLATE {
            Some(ChatTemplate::Qwen3)
        } else if declared == QWEN35_CHAT_TEMPLATE_0_8B || declared == QWEN35_CHAT_TEMPLATE_2B {
            Some(ChatTemplate::Qwen35)
        } else {
            None
        }
    }

    pub fn id(&self) -> &'static str {
        match self {
            ChatTemplate::Qwen3 => "qwen3-chatml",
            ChatTemplate::Qwen35 => "qwen35-chatml",
        }
    }

    /// Render the conversation (generation prompt included). Messages use
    /// only the supported roles; a conversation the template itself rejects
    /// (it raises) is refused with the template's own message.
    pub fn render(&self, messages: &[ChatMessageSpec]) -> Result<String> {
        match self {
            ChatTemplate::Qwen3 => Ok(render_qwen3(messages)),
            ChatTemplate::Qwen35 => render_qwen35(messages),
        }
    }
}

/// Render the pinned Qwen3 template with its canonical Jinja2 semantics for
/// the supported message subset (tools absent; roles system/user/assistant;
/// no tool_calls). Golden-tested against the recorded reference renderings.
fn render_qwen3(messages: &[ChatMessageSpec]) -> String {
    let n = messages.len();
    let mut out = String::new();
    // With no `tools`, the template's first branch reduces to: a leading
    // system message is emitted once, directly.
    if let Some(first) = messages.first() {
        if first.role == ChatRole::System {
            out.push_str(QWEN3_IM_START);
            out.push_str("system\n");
            out.push_str(&first.content);
            out.push_str(QWEN3_IM_END);
            out.push('\n');
        }
    }
    // The template scans backwards for the newest user turn that is not a
    // tool-response carrier; the scan stops at the first qualifying user.
    let mut last_query_index = n - 1;
    for index in (0..n).rev() {
        let m = &messages[index];
        if m.role == ChatRole::User
            && !(m.content.contains("<tool_response>") && m.content.contains("</tool_response>"))
        {
            last_query_index = index;
            break;
        }
    }
    for (index, m) in messages.iter().enumerate() {
        match m.role {
            ChatRole::System if index == 0 => {}
            ChatRole::System | ChatRole::User => {
                out.push_str(QWEN3_IM_START);
                out.push_str(m.role.as_str());
                out.push('\n');
                out.push_str(&m.content);
                out.push_str(QWEN3_IM_END);
                out.push('\n');
            }
            ChatRole::Assistant => {
                let (reasoning, content) = split_reasoning(&m.content);
                out.push_str(QWEN3_IM_START);
                out.push_str("assistant\n");
                let after_last_query = index > last_query_index;
                // loop.last or (not loop.last and reasoning_content)
                let think_branch = after_last_query && (index == n - 1 || !reasoning.is_empty());
                if think_branch {
                    out.push_str("<think>\n");
                    out.push_str(reasoning.trim_matches('\n'));
                    out.push_str("\n</think>\n\n");
                    out.push_str(content.trim_start_matches('\n'));
                } else {
                    out.push_str(&content);
                }
                // tool_calls are outside the supported subset.
                out.push_str(QWEN3_IM_END);
                out.push('\n');
            }
        }
    }
    // add_generation_prompt: append the assistant header. The template's
    // `enable_thinking` kwarg is not part of the request contract, so the
    // undefined branch applies (no extra marker).
    out.push_str(QWEN3_IM_START);
    out.push_str("assistant\n");
    out
}

/// Jinja2's `trim` filter: Python `str.strip()`, whose whitespace set is
/// Rust's `char::is_whitespace` plus the separators U+001C..U+001F.
fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn template_refusal(message: &str) -> TuckmillError {
    TuckmillError::new(
        ErrorCode::InvalidRequest,
        format!("the model's chat template rejects this conversation: {message}"),
    )
    .with_action("Fix the message list (roles and order) to match the model's chat template.")
}

/// Render the pinned Qwen3.5 templates with their canonical Jinja2
/// semantics for the supported subset (tools absent; string content; roles
/// system/user/assistant; no tool_calls). Every message's content is
/// trimmed; a system message may only come first; at least one user turn
/// must not be a bare tool response; an assistant turn after the last user
/// query carries an explicit (possibly empty) thinking block; the
/// generation prompt uses the template's default, thinking off
/// (`enable_thinking` is not part of the request contract). Golden-tested
/// against Jinja2 renderings of both pinned revisions.
fn render_qwen35(messages: &[ChatMessageSpec]) -> Result<String> {
    let n = messages.len();
    if n == 0 {
        return Err(template_refusal("No messages provided."));
    }
    let mut out = String::new();
    if messages[0].role == ChatRole::System {
        out.push_str(QWEN3_IM_START);
        out.push_str("system\n");
        out.push_str(py_strip(&messages[0].content));
        out.push_str(QWEN3_IM_END);
        out.push('\n');
    }
    // The newest user turn whose trimmed content is not a bare
    // `<tool_response>...</tool_response>` carrier.
    let last_query_index = (0..n).rev().find(|&i| {
        let m = &messages[i];
        let c = py_strip(&m.content);
        m.role == ChatRole::User
            && !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>"))
    });
    let Some(last_query_index) = last_query_index else {
        return Err(template_refusal("No user query found in messages."));
    };
    for (index, m) in messages.iter().enumerate() {
        let content = py_strip(&m.content);
        match m.role {
            ChatRole::System => {
                if index != 0 {
                    return Err(template_refusal("System message must be at the beginning."));
                }
            }
            ChatRole::User => {
                out.push_str(QWEN3_IM_START);
                out.push_str("user\n");
                out.push_str(content);
                out.push_str(QWEN3_IM_END);
                out.push('\n');
            }
            ChatRole::Assistant => {
                let (reasoning, content) = split_reasoning(content);
                out.push_str(QWEN3_IM_START);
                out.push_str("assistant\n");
                if index > last_query_index {
                    out.push_str(QWEN35_THINK_START);
                    out.push('\n');
                    out.push_str(py_strip(&reasoning));
                    out.push('\n');
                    out.push_str(QWEN35_THINK_END);
                    out.push_str("\n\n");
                }
                out.push_str(&content);
                out.push_str(QWEN3_IM_END);
                out.push('\n');
            }
        }
    }
    out.push_str(QWEN3_IM_START);
    out.push_str("assistant\n");
    out.push_str(QWEN35_THINK_START);
    out.push_str("\n\n");
    out.push_str(QWEN35_THINK_END);
    out.push_str("\n\n");
    Ok(out)
}

/// Split assistant content at `</think>` exactly as the template does:
/// `reasoning_content` is the text after the last `<think>` before the
/// final `</think>` (newline-trimmed), `content` is what follows it. If the
/// thought marker does not contain `</think>`, the reasoning is empty and
/// the content is unchanged.
fn split_reasoning(content: &str) -> (String, String) {
    if !content.contains("</think>") {
        return (String::new(), content.to_string());
    }
    let before = content.split("</think>").next().unwrap_or("");
    let after = content.split("</think>").last().unwrap_or("");
    let reasoning = before
        .trim_end_matches('\n')
        .rsplit("<think>")
        .next()
        .unwrap_or("")
        .trim_start_matches('\n')
        .to_string();
    let content = after.trim_start_matches('\n').to_string();
    (reasoning, content)
}

/// A model artifact's qualified input contract.
pub struct ModelBundle {
    pub artifact_id: Id,
    pub encoder: ArtifactEncoder,
    /// The recognized chat template, when the artifact declares one.
    pub template: Option<ChatTemplate>,
    /// The declared but unrecognized template, kept verbatim for the error.
    declared_template: Option<String>,
    pub eos_token_id: Option<u32>,
    /// Non-content description for records/logs.
    pub note: String,
}

impl ModelBundle {
    /// Open the bundle for an installed artifact.
    pub fn open(layout: &tuckmill_store::StoreLayout, artifact: &ModelArtifact) -> Result<Self> {
        let path = layout.artifact_payload_path(&artifact.artifact_id);
        Self::open_path(&path, &artifact.artifact_id)
    }

    /// Open the bundle for an artifact payload path.
    pub fn open_path(path: &Path, artifact_id: &Id) -> Result<Self> {
        let insp =
            tuckmill_manifest::gguf::inspect(path, &tuckmill_manifest::gguf::Limits::default())?;
        let encoder = open_encoder(path, None)?;
        let declared = insp
            .get_str("tokenizer.chat_template")
            .map(|s| s.to_string());
        // An unrecognized declared template disables chat only: raw
        // completion never applies a template, so it stays available, and
        // `render_chat` refuses with the declared template's size. The
        // template is never approximated.
        let template = declared.as_deref().and_then(ChatTemplate::recognize);
        let eos_token_id = insp
            .get_u64("tokenizer.ggml.eos_token_id")
            .map(|v| v as u32);
        let template_note = match (template, declared.as_deref()) {
            (Some(t), _) => format!("{} (canonical semantics)", t.id()),
            (None, Some(_)) => "unrecognized (chat refused; raw completion only)".to_string(),
            (None, None) => "none".to_string(),
        };
        let note = format!(
            "{}; chat template {template_note}; eos {}",
            encoder.note(),
            eos_token_id
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unset".into()),
        );
        Ok(Self {
            artifact_id: artifact_id.clone(),
            encoder,
            template,
            declared_template: declared,
            eos_token_id,
            note,
        })
    }

    /// Render a chat conversation. Refuses when the artifact declares no
    /// chat template (raw completion is the correct mode, and stays
    /// distinct).
    pub fn render_chat(&self, messages: &[ChatMessageSpec]) -> Result<String> {
        let Some(template) = self.template else {
            let detail = match self.declared_template.as_deref() {
                Some(t) => format!(
                    "the declared template ({} bytes) is not one of the recognized revisions",
                    t.len()
                ),
                None => "the artifact declares no tokenizer.chat_template".to_string(),
            };
            return Err(TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!("chat completion is unavailable: {detail}"),
            )
            .with_action(
                "Use raw completion for this artifact, or install an artifact with a recognized chat template.",
            ));
        };
        template.render(messages)
    }

    /// Tokenize text through the qualified encoder (special-token spellings
    /// map to their real vocabulary ids; `add_bos` is never set for chat).
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        self.encoder.encode(text, add_special)
    }
}

/// How the prompt tokens were produced; kept distinct so records can never
/// conflate raw completion with chat completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestMode {
    Chat {
        messages: Vec<ChatMessageSpec>,
        template: &'static str,
        /// The exact rendered prompt (evidence for records and tests).
        rendered: String,
    },
    Raw {
        text: String,
    },
}

/// The typed request: prompt tokens plus the resolved sampling, stopping
/// and output allowance. Constructed only after rendering, tokenization and
/// the context/output fit check, so every later stage operates on a request
/// that is known to be executable.
#[derive(Debug, Clone)]
pub struct RequestSpec {
    pub mode: RequestMode,
    pub prompt_tokens: Vec<u32>,
    pub max_output_tokens: u64,
    pub stop: Vec<String>,
    pub seed: u64,
    pub sampling: SamplingPolicy,
    pub stream: bool,
    pub tokenizer_label: &'static str,
}

impl RequestSpec {
    /// Build a chat-completion request: render with the model's template,
    /// tokenize, and validate the context and output allowance.
    #[allow(clippy::too_many_arguments)]
    pub fn chat(
        bundle: &ModelBundle,
        messages: Vec<ChatMessageSpec>,
        max_output_tokens: u64,
        stop: Vec<String>,
        seed: u64,
        sampling: SamplingPolicy,
        stream: bool,
        context_total: u64,
    ) -> Result<Self> {
        if messages.is_empty() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "messages must not be empty",
            ));
        }
        let rendered = bundle.render_chat(&messages)?;
        let prompt_tokens = bundle.encode(&rendered, false)?;
        let spec = Self {
            mode: RequestMode::Chat {
                messages,
                template: bundle.template.map(|t| t.id()).unwrap_or("none"),
                rendered,
            },
            prompt_tokens,
            max_output_tokens,
            stop,
            seed,
            sampling,
            stream,
            tokenizer_label: bundle.encoder.label(),
        };
        spec.validate_allowance(context_total)?;
        Ok(spec)
    }

    /// Build a raw-completion request: the prompt is tokenized as-is; no
    /// chat template is ever applied.
    #[allow(clippy::too_many_arguments)]
    pub fn raw(
        bundle: &ModelBundle,
        text: String,
        max_output_tokens: u64,
        stop: Vec<String>,
        seed: u64,
        sampling: SamplingPolicy,
        stream: bool,
        context_total: u64,
    ) -> Result<Self> {
        let prompt_tokens = bundle.encode(&text, false)?;
        let spec = Self {
            mode: RequestMode::Raw { text },
            prompt_tokens,
            max_output_tokens,
            stop,
            seed,
            sampling,
            stream,
            tokenizer_label: bundle.encoder.label(),
        };
        spec.validate_allowance(context_total)?;
        Ok(spec)
    }

    /// Context and output allowance: the prompt is counted in actual
    /// tokens, and prompt + output must fit the admitted context.
    fn validate_allowance(&self, context_total: u64) -> Result<()> {
        if self.max_output_tokens == 0 {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "max_tokens must be positive",
            ));
        }
        self.sampling.validate()?;
        check_allowance(
            self.prompt_tokens.len() as u64,
            self.max_output_tokens,
            context_total,
        )
    }

    /// True when the mode is a chat completion.
    pub fn is_chat(&self) -> bool {
        matches!(self.mode, RequestMode::Chat { .. })
    }
}

/// Shared context/output allowance check for call sites that assemble
/// prompt tokens outside a bundle (the synthetic test-engine path).
pub fn check_allowance(
    prompt_tokens: u64,
    max_output_tokens: u64,
    context_total: u64,
) -> Result<()> {
    if prompt_tokens == 0 {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "the prompt tokenizes to zero tokens",
        ));
    }
    if prompt_tokens + max_output_tokens > context_total {
        return Err(TuckmillError::new(
            ErrorCode::ContextExceedsBudget,
            format!(
                "prompt ({prompt_tokens} tokens) plus max_tokens ({max_output_tokens}) exceeds the admitted context of {context_total} tokens"
            ),
        )
        .with_action("Lower max_tokens or raise the served context size."));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(role: ChatRole, content: &str) -> ChatMessageSpec {
        ChatMessageSpec {
            role,
            content: content.to_string(),
        }
    }

    fn bundle_from_fixture(dir: &tempfile::TempDir) -> ModelBundle {
        let path = dir.path().join("f.gguf");
        tuckmill_manifest::testing::write_synthetic(
            &path,
            &tuckmill_manifest::testing::SyntheticConfig::tiny_qwen3(),
        )
        .unwrap();
        ModelBundle::open_path(&path, &Id::new("a".repeat(64)).unwrap()).unwrap()
    }

    #[test]
    fn an_unrecognized_template_refuses_chat_but_keeps_raw_completion() {
        // Same-length single-byte edit of the embedded template: the GGUF
        // stays valid, recognition (byte-exact) must fail.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        tuckmill_manifest::testing::write_synthetic(
            &path,
            &tuckmill_manifest::testing::SyntheticConfig::tiny_qwen3(),
        )
        .unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let template = tuckmill_manifest::chat::QWEN3_CHAT_TEMPLATE.as_bytes();
        let at = bytes
            .windows(template.len())
            .position(|w| w == template)
            .expect("fixture embeds the template");
        let last = at + template.len() - 1;
        bytes[last] = if bytes[last] == b' ' { b'\t' } else { b' ' };
        std::fs::write(&path, &bytes).unwrap();

        let bundle = ModelBundle::open_path(&path, &Id::new("a".repeat(64)).unwrap()).unwrap();
        assert!(bundle.template.is_none());
        assert!(bundle.note.contains("unrecognized"), "{}", bundle.note);
        let err = bundle
            .render_chat(&[spec(ChatRole::User, "hi")])
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::UnsupportedArchitecture);
        let raw = RequestSpec::raw(
            &bundle,
            "Hello, world!".to_string(),
            16,
            vec![],
            7,
            SamplingPolicy::default(),
            false,
            2048,
        )
        .unwrap();
        assert!(!raw.is_chat());
        assert!(!raw.prompt_tokens.is_empty());
    }

    #[test]
    fn renderer_matches_the_recorded_pinned_reference() {
        // Goldens recorded from the unchanged Qwen3-0.6B artifact: prompts
        // rendered with canonical Jinja2 (the template's semantics, as HF
        // applies it) and token ids from pinned llama.cpp b11695. The
        // renderer is compared byte-for-byte on every conversation.
        let goldens: serde_json::Value =
            serde_json::from_str(include_str!("../tests/data/qwen3_chat_goldens.json")).unwrap();
        let conversations = goldens["conversations"].as_array().expect("conversations");
        assert!(conversations.len() >= 10);
        for conv in conversations {
            let id = conv["id"].as_str().unwrap();
            let messages: Vec<ChatMessageSpec> = conv["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    spec(
                        ChatRole::parse(m["role"].as_str().unwrap()).unwrap(),
                        m["content"].as_str().unwrap(),
                    )
                })
                .collect();
            let rendered = ChatTemplate::Qwen3.render(&messages).unwrap();
            assert_eq!(
                rendered,
                conv["prompt"].as_str().unwrap(),
                "conversation {id:?} must render exactly like the pinned reference"
            );
        }
    }

    fn qwen35_goldens() -> Vec<(String, Vec<ChatMessageSpec>, serde_json::Value)> {
        let goldens: serde_json::Value =
            serde_json::from_str(include_str!("../tests/data/qwen35_chat_goldens.json")).unwrap();
        goldens["conversations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|conv| {
                let messages = conv["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|m| {
                        spec(
                            ChatRole::parse(m["role"].as_str().unwrap()).unwrap(),
                            m["content"].as_str().unwrap(),
                        )
                    })
                    .collect();
                (
                    conv["id"].as_str().unwrap().to_string(),
                    messages,
                    conv.clone(),
                )
            })
            .collect()
    }

    /// ADR-012 step 1: both pinned Qwen3.5 revisions are recognized
    /// byte-exactly, and the renderer reproduces the Jinja2 goldens (prompts
    /// and the template's own refusals) for every conversation.
    #[test]
    fn qwen35_renderer_matches_the_jinja2_goldens() {
        for (template, sha) in [
            (
                QWEN35_CHAT_TEMPLATE_0_8B,
                "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
            ),
            (
                QWEN35_CHAT_TEMPLATE_2B,
                "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67",
            ),
        ] {
            assert_eq!(
                tuckmill_manifest::ids::sha256_hex(template.as_bytes()),
                sha,
                "pinned template bytes"
            );
            assert_eq!(
                ChatTemplate::recognize(template),
                Some(ChatTemplate::Qwen35)
            );
        }
        let goldens = qwen35_goldens();
        assert!(goldens.len() >= 15);
        for (id, messages, conv) in goldens {
            let rendered = ChatTemplate::Qwen35.render(&messages);
            match conv["error"].as_str() {
                Some(expected) => {
                    let err = rendered.expect_err(&id);
                    assert_eq!(err.code(), ErrorCode::InvalidRequest, "{id}");
                    assert!(err.to_string().contains(expected), "{id}: {err}");
                }
                None => assert_eq!(
                    rendered.unwrap(),
                    conv["prompt"].as_str().unwrap(),
                    "conversation {id:?} must render exactly like Jinja2"
                ),
            }
        }
    }

    #[test]
    fn qwen35_reference_token_ids_are_reproduced_when_the_pinned_library_is_configured() {
        let (Ok(_lib), Ok(gguf)) = (
            std::env::var("TUCKMILL_LLAMA_LIB"),
            std::env::var("TUCKMILL_QWEN35_GGUF"),
        ) else {
            eprintln!("skipping: set TUCKMILL_LLAMA_LIB and TUCKMILL_QWEN35_GGUF");
            return;
        };
        let path = std::path::PathBuf::from(gguf);
        let bundle = ModelBundle::open_path(&path, &Id::new("c".repeat(64)).unwrap()).unwrap();
        assert_eq!(bundle.template.map(|t| t.id()), Some("qwen35-chatml"));
        for (id, messages, conv) in qwen35_goldens() {
            let Some(expected) = conv["tokens"].as_array() else {
                continue;
            };
            let rendered = bundle.render_chat(&messages).unwrap();
            let tokens = bundle.encode(&rendered, false).unwrap();
            let expected: Vec<u32> = expected
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(tokens, expected, "conversation {id:?} token ids");
        }
    }

    #[test]
    fn reference_token_ids_are_reproduced_when_the_pinned_library_is_configured() {
        let (Ok(_lib), Ok(gguf)) = (
            std::env::var("TUCKMILL_LLAMA_LIB"),
            std::env::var("TUCKMILL_REAL_GGUF"),
        ) else {
            eprintln!("skipping: set TUCKMILL_LLAMA_LIB and TUCKMILL_REAL_GGUF");
            return;
        };
        let path = std::path::PathBuf::from(gguf);
        let bundle = ModelBundle::open_path(&path, &Id::new("b".repeat(64)).unwrap()).unwrap();
        assert_eq!(
            bundle.template.map(|t| t.id()),
            Some("qwen3-chatml"),
            "the real artifact's template must be recognized"
        );
        let goldens: serde_json::Value =
            serde_json::from_str(include_str!("../tests/data/qwen3_chat_goldens.json")).unwrap();
        for conv in goldens["conversations"].as_array().unwrap() {
            let id = conv["id"].as_str().unwrap();
            let messages: Vec<ChatMessageSpec> = conv["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    spec(
                        ChatRole::parse(m["role"].as_str().unwrap()).unwrap(),
                        m["content"].as_str().unwrap(),
                    )
                })
                .collect();
            let rendered = bundle.render_chat(&messages).unwrap();
            let tokens = bundle.encode(&rendered, false).unwrap();
            let expected: Vec<u32> = conv["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(tokens, expected, "conversation {id:?} token ids");
        }
    }

    #[test]
    fn chat_requires_a_recognized_template_and_stays_distinct_from_raw() {
        let dir = tempfile::tempdir().unwrap();
        let bundle = bundle_from_fixture(&dir);
        assert_eq!(bundle.template.map(|t| t.id()), Some("qwen3-chatml"));

        let chat = RequestSpec::chat(
            &bundle,
            vec![spec(ChatRole::User, "Hello, world!")],
            16,
            vec![],
            7,
            SamplingPolicy::default(),
            false,
            2048,
        )
        .unwrap();
        assert!(chat.is_chat());
        assert!(!chat.prompt_tokens.is_empty());

        let raw = RequestSpec::raw(
            &bundle,
            "Hello, world!".to_string(),
            16,
            vec![],
            7,
            SamplingPolicy::default(),
            false,
            2048,
        )
        .unwrap();
        assert!(!raw.is_chat());
        // Raw mode must not apply the template.
        assert_ne!(raw.prompt_tokens, chat.prompt_tokens);
        assert!(raw.prompt_tokens.len() < chat.prompt_tokens.len());
    }

    #[test]
    fn context_and_output_allowance_is_validated_on_actual_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let bundle = bundle_from_fixture(&dir);
        let err = RequestSpec::chat(
            &bundle,
            vec![spec(ChatRole::User, "long enough")],
            4096,
            vec![],
            0,
            SamplingPolicy::default(),
            false,
            256,
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::ContextExceedsBudget);

        let err = RequestSpec::chat(
            &bundle,
            vec![spec(ChatRole::User, "hi")],
            0,
            vec![],
            0,
            SamplingPolicy::default(),
            false,
            256,
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidRequest);

        // A rejected sampling policy is refused at spec construction.
        let err = RequestSpec::chat(
            &bundle,
            vec![spec(ChatRole::User, "hi")],
            8,
            vec![],
            0,
            SamplingPolicy {
                temperature: 5.0,
                top_p: None,
            },
            false,
            256,
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidRequest);
    }
}
