//! Tuckmill manifest: validated model and plan schemas.
//!
//! This crate owns the immutable records described in the Tuckmill PRD
//! (ModelArtifact, DeviceProfile, ExecutionPlan, Session, BenchmarkRun), the
//! typed error vocabulary, exact byte units, content identities, model format
//! inspection (GGUF, Safetensors), architecture adapter specifications, and
//! the synthetic fixtures used by tests.
//!
//! Nothing in this crate loads model weights. Inspection reads headers and
//! indexes only; every size and offset is validated with checked arithmetic
//! before any allocation is implied.

pub mod adapter;
pub mod bytes_alphabet;
pub mod chat;
pub mod digest;
pub mod error;
pub mod ggml;
pub mod gguf;
pub mod ids;
pub mod plan;
pub mod profiles;
pub mod records;
pub mod safetensors;
pub mod testing;
pub mod units;
pub mod utf8;

pub use error::{ErrorCode, Result, TuckmillError};
pub use ids::Id;
pub use units::Bytes;

/// Schema version for persisted records. Bump only with a migration note in
/// docs/adr and a compatibility decision.
pub const SCHEMA_VERSION: u32 = 1;

/// Version of the native worker ABI (native/executor/tuckmill_worker_abi.h).
/// Plans record this so cached plans cannot be reused across ABI changes.
pub const BACKEND_ABI_VERSION: u32 = 1;

/// Version of the worker control protocol frames (native/executor/PROTOCOL.md).
/// v2: Load carries an ArtifactRef + backend options; Score/ScoreChunk/
/// ScoreDone provide bounded teacher-forced logits; StatsAck carries backend
/// statistics (ADR-005).
/// v3: ScoreChunk carries an `offset` and one position's logits stream in
/// multiple frames sized for the 1 MiB cap — real vocabularies (150k+ logits)
/// cannot fit a single JSON frame.
/// v4: Control/ControlAck frames give the governor a real command contract
/// (cache trim, prefetch control) applied by engines at safe checkpoints.
/// v5: Loaded carries resolved kernel ids; Load options are typed per backend.
/// v6: Done carries worker-side request timing; StatsAck carries the actual
/// KV precision.
/// v7: Generate carries the typed sampling policy (temperature/top-p); Done
/// carries the raw generated-token count alongside the visible-token count,
/// and StatsAck carries the prepared KV-state bytes so the plan's KV
/// reservation can be acknowledged and enforced. Plans recorded against
/// older versions are PLAN_STALE.
pub const WORKER_PROTOCOL_VERSION: u16 = 7;
