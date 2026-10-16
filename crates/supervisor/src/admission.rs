//! Live allocation ledger (TM-006).
//!
//! Every controlled allocation is reserved before asynchronous work is
//! submitted and released only after completion fences. Measured process
//! footprint is tracked separately from reservations and never summed into a
//! fictional "total RAM" figure (PRD §9).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::records::ReserveClass;
use tuckmill_manifest::units::Bytes;

#[derive(Debug)]
struct Reservation {
    id: u64,
    class: ReserveClass,
    bytes: u64,
    label: String,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LedgerStats {
    pub budget: u64,
    pub overhead_estimate: u64,
    pub safety_reserve: u64,
    pub reserved: u64,
    pub reserved_by_class: BTreeMap<String, u64>,
    pub headroom: u64,
    pub measured_peak_bytes: u64,
    pub active_reservations: usize,
    /// Labels of currently held reservations (for status reporting).
    pub active_labels: Vec<String>,
}

#[derive(Debug)]
pub struct LiveLedger {
    budget: u64,
    overhead_estimate: u64,
    safety_reserve: u64,
    inner: Mutex<Vec<Reservation>>,
    next_id: AtomicU64,
    measured_peak: AtomicU64,
}

impl LiveLedger {
    pub fn new(budget: Bytes, overhead_estimate: Bytes, safety_reserve: Bytes) -> Self {
        Self {
            budget: budget.bytes(),
            overhead_estimate: overhead_estimate.bytes(),
            safety_reserve: safety_reserve.bytes(),
            inner: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
            measured_peak: AtomicU64::new(0),
        }
    }

    fn reserved_total(&self, reservations: &[Reservation]) -> u64 {
        reservations.iter().map(|r| r.bytes).sum()
    }

    /// Reserve `bytes` for `class`. Admission is strict:
    /// `R + U + H <= B`. On failure the error includes exact numbers and a
    /// safe next action.
    pub fn try_reserve(
        self: &std::sync::Arc<Self>,
        class: ReserveClass,
        bytes: Bytes,
        label: impl Into<String>,
    ) -> Result<ReservationHandle> {
        let mut reservations = self.inner.lock().expect("ledger lock");
        let current = self.reserved_total(&reservations);
        let new_total = current.saturating_add(bytes.bytes());
        let admitted = new_total
            .saturating_add(self.overhead_estimate)
            .saturating_add(self.safety_reserve);
        if admitted > self.budget {
            return Err(TuckmillError::new(
                ErrorCode::InvalidPlan,
                format!(
                    "reservation '{}' of {} would need {} (R {} + U {} + H {}) against budget {}",
                    label.into(),
                    bytes,
                    Bytes(admitted),
                    Bytes(new_total),
                    Bytes(self.overhead_estimate),
                    Bytes(self.safety_reserve),
                    Bytes(self.budget),
                ),
            )
            .with_action(
                "Lower the reservation, free resources, or raise --memory-budget; nothing was allocated.",
            ));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        reservations.push(Reservation {
            id,
            class,
            bytes: bytes.bytes(),
            label: label.into(),
        });
        Ok(ReservationHandle {
            ledger: self.clone(),
            id,
        })
    }

    fn release(&self, id: u64) {
        let mut reservations = self.inner.lock().expect("ledger lock");
        let before = reservations.len();
        reservations.retain(|r| r.id != id);
        debug_assert_eq!(before, reservations.len() + 1, "double release");
    }

    /// Record an observed physical footprint sample. Returns true when the
    /// measurement exceeds reservation + overhead beyond the allowed
    /// tolerance (default 25%), which the caller must surface as an overrun.
    pub fn observe_footprint(&self, measured_bytes: u64) -> bool {
        let prev = self
            .measured_peak
            .fetch_max(measured_bytes, Ordering::Relaxed);
        let _ = prev;
        let reservations = self.inner.lock().expect("ledger lock");
        let reserved = self.reserved_total(&reservations);
        let expected = reserved
            .saturating_add(self.overhead_estimate)
            .saturating_add(self.safety_reserve);
        measured_bytes > expected + expected / 4
    }

    pub fn stats(&self) -> LedgerStats {
        let reservations = self.inner.lock().expect("ledger lock");
        let reserved = self.reserved_total(&reservations);
        let mut by_class: BTreeMap<String, u64> = BTreeMap::new();
        for r in reservations.iter() {
            *by_class.entry(r.class.as_str().to_string()).or_default() += r.bytes;
        }
        LedgerStats {
            budget: self.budget,
            overhead_estimate: self.overhead_estimate,
            safety_reserve: self.safety_reserve,
            reserved,
            reserved_by_class: by_class,
            headroom: self.budget.saturating_sub(
                reserved
                    .saturating_add(self.overhead_estimate)
                    .saturating_add(self.safety_reserve),
            ),
            measured_peak_bytes: self.measured_peak.load(Ordering::Relaxed),
            active_reservations: reservations.len(),
            active_labels: reservations.iter().map(|r| r.label.clone()).collect(),
        }
    }
}

/// RAII reservation. The guard is dropped only after the completion fence:
/// callers must keep it alive until the worker confirms the operation is
/// fully finished (never to fake fast cancellation by dropping early).
#[derive(Debug)]
pub struct ReservationHandle {
    ledger: std::sync::Arc<LiveLedger>,
    id: u64,
}

impl ReservationHandle {
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Explicit completion fence; equivalent to dropping.
    pub fn fence(self) {}
}

impl Drop for ReservationHandle {
    fn drop(&mut self) {
        self.ledger.release(self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn ledger(budget: u64, overhead: u64, safety: u64) -> Arc<LiveLedger> {
        Arc::new(LiveLedger::new(
            Bytes(budget),
            Bytes(overhead),
            Bytes(safety),
        ))
    }

    #[test]
    fn admission_rule_is_enforced_exactly() {
        // Budget 1000, U=100, H=100 => max reserved = 800.
        let l = ledger(1000, 100, 100);
        let _a = l
            .try_reserve(ReserveClass::Weights, Bytes(500), "w")
            .unwrap();
        let _b = l
            .try_reserve(ReserveClass::KvState, Bytes(300), "kv")
            .unwrap();
        let err = l
            .try_reserve(ReserveClass::Workspaces, Bytes(1), "ws")
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidPlan);
        assert!(err.message().contains("1000"), "{}", err.message());
    }

    #[test]
    fn release_returns_headroom() {
        let l = ledger(1000, 100, 100);
        {
            let _a = l
                .try_reserve(ReserveClass::Weights, Bytes(800), "w")
                .unwrap();
            assert_eq!(l.stats().headroom, 0);
        }
        assert_eq!(l.stats().headroom, 800);
        assert_eq!(l.stats().active_reservations, 0);
    }

    #[test]
    fn breakdown_is_by_class() {
        let l = ledger(10_000, 0, 0);
        let _a = l
            .try_reserve(ReserveClass::Weights, Bytes(1000), "w")
            .unwrap();
        let _b = l
            .try_reserve(ReserveClass::ExpertCache, Bytes(2000), "c")
            .unwrap();
        let stats = l.stats();
        assert_eq!(stats.reserved, 3000);
        assert_eq!(stats.reserved_by_class["weights"], 1000);
        assert_eq!(stats.reserved_by_class["expert_cache"], 2000);
    }

    #[test]
    fn overrun_detection_flags_measured_excess() {
        let l = ledger(10_000_000, 1_000_000, 1_000_000);
        let _a = l
            .try_reserve(ReserveClass::Weights, Bytes(2_000_000), "w")
            .unwrap();
        // Expected = 2M + 1M + 1M = 4M; 25% tolerance => 5M is fine, 6M is not.
        assert!(!l.observe_footprint(4_500_000));
        assert!(l.observe_footprint(6_000_000));
        assert_eq!(l.stats().measured_peak_bytes, 6_000_000);
    }

    #[test]
    fn fence_releases_on_drop() {
        let l = ledger(1000, 0, 0);
        let h = l
            .try_reserve(ReserveClass::Weights, Bytes(1000), "w")
            .unwrap();
        assert_eq!(l.stats().reserved, 1000);
        h.fence();
        assert_eq!(l.stats().reserved, 0);
    }
}
