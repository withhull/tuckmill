//! Versioned, length-framed worker control protocol (PRD §15, ADR-005).
//!
//! Frames are `u32` little-endian length prefixes followed by UTF-8 JSON.
//! Message lengths are capped, and every field is validated before it is
//! used. The protocol carries control messages, tokens, scores, errors, and
//! statistics — it is deliberately not a general tensor RPC service.
//!
//! Protocol v2 (current): `Load` carries an `ArtifactRef` plus optional
//! backend options; `Score`/`ScoreChunk`/`ScoreDone` provide bounded
//! teacher-forced logits; `StatsAck` carries backend statistics.

use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use tuckmill_backend::{ArtifactRef, BackendStats};
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::plan::ExecutionPlan;
use tuckmill_manifest::{BACKEND_ABI_VERSION, WORKER_PROTOCOL_VERSION};

/// Maximum accepted frame payload (1 MiB). Larger frames are a protocol
/// violation, not something to buffer.
pub const MAX_FRAME_BYTES: u32 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t")]
pub enum WorkerFrame {
    Hello {
        protocol: u16,
        abi: u32,
        client_build: String,
    },
    HelloAck {
        protocol: u16,
        abi: u32,
        worker_build: String,
        engine: String,
        certified: bool,
    },
    Load {
        plan: Box<ExecutionPlan>,
        artifact: ArtifactRef,
        /// Backend-specific options (for example a bounded weight-cache
        /// size). Validated by the backend; unknown keys are ignored only if
        /// the backend says so — never silently.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        options: Option<serde_json::Value>,
    },
    Loaded {
        note: String,
        footprint_bytes: u64,
    },
    Generate {
        request_id: String,
        prompt_tokens: Vec<u32>,
        max_output_tokens: u64,
        stop: Vec<String>,
        seed: u64,
    },
    State {
        state: String,
    },
    Token {
        index: u64,
        text: String,
    },
    Done {
        reason: String,
        committed: u64,
    },
    Error {
        code: String,
        message: String,
        action: String,
    },
    /// Teacher-forced scoring request: next-token logits at the listed
    /// positions under the causal prefix.
    Score {
        request_id: String,
        tokens: Vec<u32>,
        positions: Vec<u64>,
    },
    /// Logits for one scored position (vocabulary-order, f32). A position's
    /// logits may arrive as several frames: `offset` is the element index of
    /// the first value in this frame, so real vocabularies (150k+) stay under
    /// the 1 MiB frame cap.
    ScoreChunk {
        position: u64,
        offset: u32,
        logits: Vec<f32>,
    },
    ScoreDone {
        scored: u64,
    },
    Heartbeat {
        uptime_ms: u64,
    },
    Cancel {
        request_id: String,
    },
    Stats,
    StatsAck {
        footprint_bytes: u64,
        uptime_ms: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        backend: Option<BackendStats>,
    },
    Shutdown,
    Bye,
}

impl WorkerFrame {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let payload = serde_json::to_vec(self).map_err(|e| {
            tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "frame encoding")
        })?;
        if payload.len() as u64 > MAX_FRAME_BYTES as u64 {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!("frame payload {} bytes exceeds the cap", payload.len()),
            ));
        }
        let mut out = Vec::with_capacity(4 + payload.len());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        let bytes = self.encode()?;
        writer.write_all(&bytes).map_err(TuckmillError::from)?;
        writer.flush().map_err(TuckmillError::from)
    }

    /// Read one frame. Returns Ok(None) on clean EOF before any byte of a
    /// new frame. Enforces the size cap before allocating.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Option<WorkerFrame>> {
        let mut len_buf = [0u8; 4];
        if !read_exact_or_eof(reader, &mut len_buf)? {
            return Ok(None);
        }
        let len = u32::from_le_bytes(len_buf);
        if len == 0 || len > MAX_FRAME_BYTES {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!("worker frame length {len} is outside 1..={MAX_FRAME_BYTES}"),
            ));
        }
        let mut payload = vec![0u8; len as usize];
        reader.read_exact(&mut payload).map_err(|e| {
            TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!("worker frame truncated: {e}"),
            )
        })?;
        let frame: WorkerFrame = serde_json::from_slice(&payload).map_err(|e| {
            TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!("worker frame is not a valid message: {e}"),
            )
        })?;
        Ok(Some(frame))
    }
}

fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<bool> {
    let mut filled = 0usize;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => {
                if filled == 0 {
                    return Ok(false);
                }
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "worker disconnected mid-frame",
                ));
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(TuckmillError::from(e)),
        }
    }
    Ok(true)
}

/// Validate a handshake pair. Both sides must agree on protocol and ABI.
pub fn validate_handshake(protocol: u16, abi: u32) -> Result<()> {
    if protocol != WORKER_PROTOCOL_VERSION {
        return Err(TuckmillError::new(
            ErrorCode::WorkerFailed,
            format!(
                "worker protocol mismatch: worker {protocol}, supervisor {WORKER_PROTOCOL_VERSION}"
            ),
        ));
    }
    if abi != BACKEND_ABI_VERSION {
        return Err(TuckmillError::new(
            ErrorCode::WorkerFailed,
            format!("backend ABI mismatch: worker {abi}, supervisor {BACKEND_ABI_VERSION}"),
        ));
    }
    Ok(())
}

pub fn hello(client_build: &str) -> WorkerFrame {
    WorkerFrame::Hello {
        protocol: WORKER_PROTOCOL_VERSION,
        abi: BACKEND_ABI_VERSION,
        client_build: client_build.to_string(),
    }
}

pub fn hello_ack(worker_build: &str, engine: &str, certified: bool) -> WorkerFrame {
    WorkerFrame::HelloAck {
        protocol: WORKER_PROTOCOL_VERSION,
        abi: BACKEND_ABI_VERSION,
        worker_build: worker_build.to_string(),
        engine: engine.to_string(),
        certified,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn artifact() -> ArtifactRef {
        ArtifactRef {
            path: "/tmp/model.bin".into(),
            file_bytes: 1234,
            metadata_hash: "a".repeat(64),
            tensor_index_hash: "b".repeat(64),
            tokenizer_hash: Some("c".repeat(64)),
            arch_id: Some("qwen3".into()),
        }
    }

    fn frames() -> Vec<WorkerFrame> {
        vec![
            hello("tuckmill-test"),
            hello_ack("tuckmill-test", "reference-test", false),
            WorkerFrame::Generate {
                request_id: "req_1".into(),
                prompt_tokens: vec![1, 2, 3],
                max_output_tokens: 16,
                stop: vec!["</s>".into()],
                seed: 42,
            },
            WorkerFrame::Score {
                request_id: "req_2".into(),
                tokens: vec![4, 5, 6, 7],
                positions: vec![1, 3],
            },
            WorkerFrame::ScoreChunk {
                position: 1,
                offset: 0,
                logits: vec![0.5, -1.0, 2.0],
            },
            WorkerFrame::ScoreDone { scored: 2 },
            WorkerFrame::Token {
                index: 0,
                text: "t0".into(),
            },
            WorkerFrame::Done {
                reason: "stop".into(),
                committed: 5,
            },
            WorkerFrame::Error {
                code: "WORKER_FAILED".into(),
                message: "boom".into(),
                action: "retry".into(),
            },
            WorkerFrame::Cancel {
                request_id: "req_1".into(),
            },
            WorkerFrame::Heartbeat { uptime_ms: 1200 },
            WorkerFrame::Stats,
            WorkerFrame::StatsAck {
                footprint_bytes: 1234,
                uptime_ms: 5000,
                backend: Some(BackendStats {
                    logical_read_bytes: 4096,
                    blocks_loaded: 3,
                    cache_hits: 7,
                    cache_misses: 3,
                    evictions: 1,
                    in_flight_leases: 0,
                    footprint_bytes: Some(999),
                }),
            },
            WorkerFrame::Shutdown,
            WorkerFrame::Bye,
        ]
    }

    #[test]
    fn frames_roundtrip() {
        for frame in frames() {
            let mut encoded = Vec::new();
            frame.write_to(&mut encoded).unwrap();
            let mut cursor = Cursor::new(encoded);
            let decoded = WorkerFrame::read_from(&mut cursor).unwrap().unwrap();
            assert_eq!(decoded, frame);
            assert!(WorkerFrame::read_from(&mut cursor).unwrap().is_none());
        }
    }

    #[test]
    fn load_frame_carries_artifact_and_options() {
        let mut plan = ExecutionPlan::new_skeleton(
            tuckmill_manifest::Id::new("a".repeat(64)).unwrap(),
            tuckmill_manifest::Id::new("b".repeat(64)).unwrap(),
            "qwen3".into(),
        );
        plan.mode = tuckmill_manifest::plan::ExecutionMode::Resident;
        plan.plan_id = plan.compute_id().unwrap();
        let frame = WorkerFrame::Load {
            plan: Box::new(plan),
            artifact: artifact(),
            options: Some(serde_json::json!({"weight_cache_bytes": 67108864u64})),
        };
        let mut encoded = Vec::new();
        frame.write_to(&mut encoded).unwrap();
        let decoded = WorkerFrame::read_from(&mut Cursor::new(encoded))
            .unwrap()
            .unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn oversized_frame_is_rejected_before_allocation() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(MAX_FRAME_BYTES + 1).to_le_bytes());
        buf.extend_from_slice(&[0u8; 64]);
        let err = WorkerFrame::read_from(&mut Cursor::new(buf)).unwrap_err();
        assert_eq!(err.code(), ErrorCode::WorkerFailed);
        assert!(err.message().contains("length"));
    }

    #[test]
    fn zero_length_frame_is_rejected() {
        let buf = 0u32.to_le_bytes().to_vec();
        assert!(WorkerFrame::read_from(&mut Cursor::new(buf)).is_err());
    }

    #[test]
    fn invalid_json_is_a_worker_failure() {
        let payload = b"{\"t\":\"Nope\"}";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(payload);
        let err = WorkerFrame::read_from(&mut Cursor::new(buf)).unwrap_err();
        assert_eq!(err.code(), ErrorCode::WorkerFailed);
    }

    #[test]
    fn truncated_stream_is_detected() {
        let frame = WorkerFrame::Heartbeat { uptime_ms: 7 };
        let mut encoded = Vec::new();
        frame.write_to(&mut encoded).unwrap();
        encoded.truncate(encoded.len() - 2);
        let err = WorkerFrame::read_from(&mut Cursor::new(encoded)).unwrap_err();
        assert_eq!(err.code(), ErrorCode::WorkerFailed);
    }

    #[test]
    fn handshake_mismatches_fail_closed() {
        assert!(validate_handshake(WORKER_PROTOCOL_VERSION, BACKEND_ABI_VERSION).is_ok());
        assert!(validate_handshake(WORKER_PROTOCOL_VERSION + 1, BACKEND_ABI_VERSION).is_err());
        assert!(validate_handshake(WORKER_PROTOCOL_VERSION, BACKEND_ABI_VERSION + 1).is_err());
    }
}
