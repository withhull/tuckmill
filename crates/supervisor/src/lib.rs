//! Tuckmill supervisor: admission, API, worker lifecycle.
//!
//! The supervisor owns configuration, admission, request queues, local
//! authentication, cancellation, worker lifecycle, and non-content telemetry.
//! It remains responsive when a worker fails. Tensors are never computed
//! here; that is the worker's job.

pub mod admission;
pub mod api;
pub mod bench;
pub mod engine;
pub mod governor;
pub mod log;
pub mod pressure;
pub mod proto;
pub mod queue;
pub mod runtime;
pub mod scoring;
pub mod tokenization;
pub mod verify;
pub mod worker;
pub mod worker_host;

use tuckmill_manifest::ids::now_unix_ms;

/// Physical footprint of this host process (supervisor side).
pub fn host_footprint_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        tuckmill_planner::device::physical_footprint_of(std::process::id() as i32).unwrap_or(0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Simple monotonic helper used across worker frames.
pub fn uptime_ms(started: std::time::Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

/// Timestamp helper for tests that need a stable clock origin.
pub fn now_ms() -> u64 {
    now_unix_ms()
}
