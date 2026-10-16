//! Bounded request queue: one active low-memory generation, additional
//! requests queued with a hard depth limit (PRD §4.2, FR-10).
//!
//! No throughput-oriented multi-user batching exists in this release.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};

#[derive(Debug)]
pub struct QueueTicket {
    pub id: u64,
    pub position: usize,
}

struct Inner {
    next_id: u64,
    queue: VecDeque<u64>,
    closed: bool,
}

pub struct RequestQueue {
    capacity: usize,
    inner: Mutex<Inner>,
    available: Condvar,
}

impl RequestQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            inner: Mutex::new(Inner {
                next_id: 1,
                queue: VecDeque::new(),
                closed: false,
            }),
            available: Condvar::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn depth(&self) -> usize {
        self.inner.lock().expect("queue lock").queue.len()
    }

    /// Enqueue a request. Fails with QUEUE_FULL when the depth limit is hit.
    pub fn submit(&self) -> Result<QueueTicket> {
        let mut inner = self.inner.lock().expect("queue lock");
        if inner.closed {
            return Err(TuckmillError::new(
                ErrorCode::Cancelled,
                "the request queue is closed; the service is shutting down",
            ));
        }
        if inner.queue.len() >= self.capacity {
            return Err(TuckmillError::new(
                ErrorCode::QueueFull,
                format!(
                    "queue depth {} has reached its limit of {}",
                    inner.queue.len(),
                    self.capacity
                ),
            ));
        }
        let id = inner.next_id;
        inner.next_id += 1;
        inner.queue.push_back(id);
        let position = inner.queue.len() - 1;
        self.available.notify_all();
        Ok(QueueTicket { id, position })
    }

    /// Wait until `ticket` reaches the front of the queue *without*
    /// claiming it. Returns false when the queue closes while waiting.
    /// An expiration removes the ticket and reports a cancellation.
    pub fn wait_until_front(&self, ticket: &QueueTicket, timeout: Duration) -> Result<bool> {
        let deadline = std::time::Instant::now() + timeout;
        let mut inner = self.inner.lock().expect("queue lock");
        loop {
            if inner.closed {
                inner.queue.retain(|id| *id != ticket.id);
                return Ok(false);
            }
            if !inner.queue.contains(&ticket.id) {
                // Someone else removed us (close/abandon path).
                return Err(TuckmillError::new(
                    ErrorCode::Cancelled,
                    "the request left the queue before its turn",
                ));
            }
            if inner.queue.front() == Some(&ticket.id) {
                return Ok(true);
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                // Remove ourselves from the queue: the client gave up.
                inner.queue.retain(|id| *id != ticket.id);
                return Err(TuckmillError::new(
                    ErrorCode::Cancelled,
                    "gave up waiting for a turn in the request queue",
                ));
            }
            let wait = (deadline - now).min(Duration::from_millis(100));
            let (guard, _) = self
                .available
                .wait_timeout(inner, wait)
                .expect("queue wait");
            inner = guard;
        }
    }

    /// Claim the front position. Returns true when `ticket` was the front
    /// entry and has been removed; false when it is no longer front (or was
    /// already removed).
    pub fn claim_front(&self, ticket: &QueueTicket) -> Result<bool> {
        let mut inner = self.inner.lock().expect("queue lock");
        if inner.queue.front() == Some(&ticket.id) {
            inner.queue.pop_front();
            self.available.notify_all();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Wake waiters (used when the active slot frees to hand off promptly).
    pub fn notify_waiters(&self) {
        self.available.notify_all();
    }

    /// Leave the queue without claiming (client disappeared while queued).
    pub fn abandon(&self, ticket: &QueueTicket) {
        let mut inner = self.inner.lock().expect("queue lock");
        inner.queue.retain(|id| *id != ticket.id);
    }

    pub fn close(&self) {
        let mut inner = self.inner.lock().expect("queue lock");
        inner.closed = true;
        self.available.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn depth_limit_produces_queue_full() {
        let q = RequestQueue::new(2);
        let _a = q.submit().unwrap();
        let _b = q.submit().unwrap();
        let err = q.submit().unwrap_err();
        assert_eq!(err.code(), ErrorCode::QueueFull);
        assert_eq!(q.depth(), 2);
    }

    #[test]
    fn tickets_grant_in_order() {
        let q = Arc::new(RequestQueue::new(4));
        let a = q.submit().unwrap();
        let b = q.submit().unwrap();
        assert!(q.wait_until_front(&a, Duration::from_secs(1)).unwrap());
        assert!(q.claim_front(&a).unwrap());
        assert!(q.wait_until_front(&b, Duration::from_secs(1)).unwrap());
        assert!(q.claim_front(&b).unwrap());
    }

    #[test]
    fn abandoning_removes_without_blocking_others() {
        let q = Arc::new(RequestQueue::new(4));
        let a = q.submit().unwrap();
        let b = q.submit().unwrap();
        q.abandon(&a);
        assert!(q.wait_until_front(&b, Duration::from_secs(1)).unwrap());
        assert!(q.claim_front(&b).unwrap());
    }

    #[test]
    fn close_wakes_waiters() {
        let q = Arc::new(RequestQueue::new(4));
        let _blocker = q.submit().unwrap(); // never claims
        let ticket = q.submit().unwrap();
        let q2 = q.clone();
        let handle =
            std::thread::spawn(move || q2.wait_until_front(&ticket, Duration::from_secs(5)));
        std::thread::sleep(Duration::from_millis(50));
        q.close();
        assert!(!handle.join().unwrap().unwrap());
    }

    #[test]
    fn give_up_timeout_is_a_cancelled_error() {
        let q = Arc::new(RequestQueue::new(4));
        let a = q.submit().unwrap();
        let b_ticket = q.submit().unwrap();
        // `a` never claims; b gives up quickly.
        let start = std::time::Instant::now();
        let err = q
            .wait_until_front(&b_ticket, Duration::from_millis(200))
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::Cancelled);
        assert!(start.elapsed() < Duration::from_secs(2));
        // a can still claim afterwards.
        assert!(q.wait_until_front(&a, Duration::from_secs(1)).unwrap());
        assert!(q.claim_front(&a).unwrap());
    }
}
