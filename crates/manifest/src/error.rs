//! Typed error vocabulary.
//!
//! The PRD (§15) fixes the public error codes. Every code carries a safe next
//! action that never leaks private paths or tokens. Internal codes beyond the
//! PRD list are marked and must not be presented as user-facing product
//! semantics without a spec change.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Public typed error codes (PRD §15) plus internal supporting codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    // ---- PRD §15 public codes ----
    UnsupportedArchitecture,
    UnsupportedQuantization,
    InsufficientDisk,
    ContextExceedsBudget,
    PlanStale,
    MemoryPressurePause,
    StorageLost,
    IntegrityFailure,
    WorkerFailed,
    QueueFull,
    Cancelled,
    // ---- Internal supporting codes ----
    /// Malformed or unreadable artifact (header, offsets, or index invalid).
    InvalidArtifact,
    /// Caller request is malformed or asks for an unsupported field.
    InvalidRequest,
    /// Plan document failed validation or is internally inconsistent.
    InvalidPlan,
    /// Local configuration is missing or invalid.
    InvalidConfiguration,
    /// I/O failure that is not better described by a more specific code.
    Io,
    /// Feature or capability is not available in this build/device.
    Unavailable,
    /// Invariant violation; indicates a bug, never a normal outcome.
    Internal,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::UnsupportedArchitecture => "UNSUPPORTED_ARCHITECTURE",
            ErrorCode::UnsupportedQuantization => "UNSUPPORTED_QUANTIZATION",
            ErrorCode::InsufficientDisk => "INSUFFICIENT_DISK",
            ErrorCode::ContextExceedsBudget => "CONTEXT_EXCEEDS_BUDGET",
            ErrorCode::PlanStale => "PLAN_STALE",
            ErrorCode::MemoryPressurePause => "MEMORY_PRESSURE_PAUSE",
            ErrorCode::StorageLost => "STORAGE_LOST",
            ErrorCode::IntegrityFailure => "INTEGRITY_FAILURE",
            ErrorCode::WorkerFailed => "WORKER_FAILED",
            ErrorCode::QueueFull => "QUEUE_FULL",
            ErrorCode::Cancelled => "CANCELLED",
            ErrorCode::InvalidArtifact => "INVALID_ARTIFACT",
            ErrorCode::InvalidRequest => "INVALID_REQUEST",
            ErrorCode::InvalidPlan => "INVALID_PLAN",
            ErrorCode::InvalidConfiguration => "INVALID_CONFIGURATION",
            ErrorCode::Io => "IO",
            ErrorCode::Unavailable => "UNAVAILABLE",
            ErrorCode::Internal => "INTERNAL",
        }
    }

    /// Safe next action for the user (PRD §15: "Include a safe next action").
    pub fn next_action(self) -> &'static str {
        match self {
            ErrorCode::UnsupportedArchitecture => {
                "Run 'tuckmill inspect' to confirm the architecture, then install a matching adapter or choose a supported model."
            }
            ErrorCode::UnsupportedQuantization => {
                "Choose an artifact whose quantization profile is listed as supported, or convert it with an approved tool."
            }
            ErrorCode::InsufficientDisk => {
                "Free space on the target volume or choose a smaller profile; no source file was modified."
            }
            ErrorCode::ContextExceedsBudget => {
                "Lower --context or raise --memory-budget; the request was rejected before loading."
            }
            ErrorCode::PlanStale => {
                "Re-run 'tuckmill plan' to produce a fresh plan for the current artifact and device."
            }
            ErrorCode::MemoryPressurePause => {
                "Close memory-heavy applications and retry; the request paused safely and retained its committed position."
            }
            ErrorCode::StorageLost => {
                "Reconnect the external volume that holds the artifact, then retry."
            }
            ErrorCode::IntegrityFailure => {
                "Re-import the artifact from a trusted source; the existing copy is not published as valid."
            }
            ErrorCode::WorkerFailed => {
                "Restart the request; the last acknowledged output position was retained and no continuation state was fabricated."
            }
            ErrorCode::QueueFull => {
                "Wait for the active request to finish or cancel it, then retry."
            }
            ErrorCode::Cancelled => "No action required; the request was cancelled.",
            ErrorCode::InvalidArtifact => {
                "Point at a complete .gguf or .safetensors file; run 'tuckmill inspect' for details."
            }
            ErrorCode::InvalidRequest => {
                "Fix the request fields; only the documented API subset is accepted."
            }
            ErrorCode::InvalidPlan => {
                "Re-run 'tuckmill plan'; if the problem persists, report the plan id."
            }
            ErrorCode::InvalidConfiguration => {
                "Run 'tuckmill doctor' and correct the reported configuration issue."
            }
            ErrorCode::Io => "Check file permissions and available space, then retry.",
            ErrorCode::Unavailable => {
                "This capability is not available in this build or on this device; see 'tuckmill doctor'."
            }
            ErrorCode::Internal => "This indicates a bug; please report it with the displayed context.",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The single error type crossing crate boundaries.
#[derive(Debug)]
pub struct TuckmillError {
    code: ErrorCode,
    message: String,
    action: String,
    source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl TuckmillError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            action: code.next_action().to_string(),
            source: None,
        }
    }

    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.action = action.into();
        self
    }

    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    pub fn code(&self) -> ErrorCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn action(&self) -> &str {
        &self.action
    }

    /// Machine-readable form used by the CLI (--json) and the local API.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "code": self.code.as_str(),
                "message": self.message,
                "action": self.action,
            }
        })
    }
}

impl fmt::Display for TuckmillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} — next: {}",
            self.code, self.message, self.action
        )
    }
}

impl std::error::Error for TuckmillError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

impl From<std::io::Error> for TuckmillError {
    fn from(err: std::io::Error) -> Self {
        TuckmillError::new(ErrorCode::Io, err.to_string())
    }
}

/// Convenience constructor: `bail!(IntegrityFailure, "bad header: {}", why)`.
#[macro_export]
macro_rules! bail {
    ($code:ident, $($arg:tt)*) => {
        return Err($crate::error::TuckmillError::new(
            $crate::error::ErrorCode::$code,
            format!($($arg)*),
        ))
    };
}

pub type Result<T> = std::result::Result<T, TuckmillError>;

/// Attach a context code to a serde_json failure, which otherwise would be an
/// opaque string.
pub fn json_error(err: serde_json::Error, code: ErrorCode, what: &str) -> TuckmillError {
    TuckmillError::new(code, format!("{what}: {err}")).with_source(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable_strings() {
        assert_eq!(
            ErrorCode::UnsupportedArchitecture.as_str(),
            "UNSUPPORTED_ARCHITECTURE"
        );
        assert_eq!(
            ErrorCode::ContextExceedsBudget.as_str(),
            "CONTEXT_EXCEEDS_BUDGET"
        );
        assert_eq!(ErrorCode::QueueFull.as_str(), "QUEUE_FULL");
    }

    #[test]
    fn serde_roundtrip_is_screaming_snake() {
        let code = ErrorCode::MemoryPressurePause;
        let s = serde_json::to_string(&code).unwrap();
        assert_eq!(s, "\"MEMORY_PRESSURE_PAUSE\"");
        let back: ErrorCode = serde_json::from_str(&s).unwrap();
        assert_eq!(back, code);
    }

    #[test]
    fn every_code_has_nonempty_action() {
        for code in [
            ErrorCode::UnsupportedArchitecture,
            ErrorCode::UnsupportedQuantization,
            ErrorCode::InsufficientDisk,
            ErrorCode::ContextExceedsBudget,
            ErrorCode::PlanStale,
            ErrorCode::MemoryPressurePause,
            ErrorCode::StorageLost,
            ErrorCode::IntegrityFailure,
            ErrorCode::WorkerFailed,
            ErrorCode::QueueFull,
            ErrorCode::Cancelled,
            ErrorCode::InvalidArtifact,
            ErrorCode::InvalidRequest,
            ErrorCode::InvalidPlan,
            ErrorCode::InvalidConfiguration,
            ErrorCode::Io,
            ErrorCode::Unavailable,
            ErrorCode::Internal,
        ] {
            assert!(!code.next_action().is_empty(), "{code}");
        }
    }

    #[test]
    fn json_form_has_error_object() {
        let err = TuckmillError::new(ErrorCode::QueueFull, "depth 8 reached");
        let v = err.to_json();
        assert_eq!(v["error"]["code"], "QUEUE_FULL");
        assert!(v["error"]["action"].as_str().unwrap().len() > 10);
    }
}
