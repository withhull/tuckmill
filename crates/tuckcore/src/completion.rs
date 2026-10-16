//! GPU-style completion tracking.
//!
//! Buffers submitted to asynchronous work are leased until their fence
//! completes; the ledger makes "release only after completion" a testable
//! property rather than a comment. On the Metal path this structure will be
//! driven by `MTLCommandBuffer` completion handlers; Stage 0 drives it
//! around scalar compute so the lifecycle semantics are already enforced.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::loader::WeightBlock;

/// A pending work item holding blocks it still needs.
#[derive(Debug, Default)]
pub struct Lease {
    pub fence: u64,
    pub blocks: Vec<Arc<WeightBlock>>,
    pub keys: Vec<(usize, u32)>,
}

impl Lease {
    pub fn bytes(&self) -> u64 {
        self.blocks.iter().map(|b| b.bytes().len() as u64).sum()
    }
}

#[derive(Debug, Default)]
pub struct CompletionTracker {
    next_fence: AtomicU64,
    in_flight: Mutex<BTreeMap<u64, Lease>>,
    completed: AtomicU64,
}

impl CompletionTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register in-flight work holding `blocks`. Returns the fence id.
    pub fn submit(&self, blocks: Vec<Arc<WeightBlock>>) -> u64 {
        let fence = self.next_fence.fetch_add(1, Ordering::Relaxed) + 1;
        let keys = blocks.iter().map(|b| (b.tensor, b.chunk)).collect();
        let mut in_flight = self.in_flight.lock().expect("tracker lock");
        in_flight.insert(
            fence,
            Lease {
                fence,
                blocks,
                keys,
            },
        );
        fence
    }

    /// Signal fence completion; returns the released lease. Releases happen
    /// only here — never on eviction, cancellation, or shutdown shortcuts.
    pub fn complete(&self, fence: u64) -> Option<Lease> {
        let lease = self.in_flight.lock().expect("tracker lock").remove(&fence);
        if lease.is_some() {
            self.completed.fetch_add(1, Ordering::Relaxed);
        }
        lease
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.lock().expect("tracker lock").len()
    }

    pub fn in_flight_bytes(&self) -> u64 {
        self.in_flight
            .lock()
            .expect("tracker lock")
            .values()
            .map(|l| l.bytes())
            .sum()
    }

    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{BlockLoader, BlockPolicy, TensorEntry};
    use crate::tensor;
    use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

    fn loader_with_block() -> (tempfile::TempDir, BlockLoader) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let policy = BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 128 * 1024,
        };
        let tensors: Vec<TensorEntry> = insp
            .tensors
            .iter()
            .map(|t| {
                let row = tensor::row_geometry(t.type_id, &t.dims, policy.block_bytes);
                let chunks = match &row {
                    Some(r) => r.rows.div_ceil(r.rows_per_chunk),
                    None => t.byte_size.div_ceil(policy.block_bytes),
                };
                TensorEntry {
                    name: t.name.clone(),
                    type_id: t.type_id,
                    dims: t.dims.clone(),
                    rel_offset: t.offset,
                    byte_size: t.byte_size,
                    row,
                    chunks,
                }
            })
            .collect();
        let loader = BlockLoader::open(path.as_path(), insp.data_start, tensors, policy).unwrap();
        (dir, loader)
    }

    #[test]
    fn leases_protect_blocks_until_the_fence_completes() {
        let (_dir, loader) = loader_with_block();
        let tracker = CompletionTracker::new();
        let idx = loader.tensor_index("token_embd.weight").unwrap();
        let block = loader.block(idx, 0).unwrap();
        let original_generation = block.generation;
        loader.pin(block.tensor, block.chunk);
        let fence = tracker.submit(vec![block.clone()]);
        assert_eq!(tracker.in_flight(), 1);
        assert_eq!(tracker.in_flight_bytes(), block.bytes().len() as u64);

        // Forced eviction while leased: the pinned/leased block survives.
        assert_eq!(loader.evict_all(), 0);

        // Fence completes: release, then eviction succeeds.
        let lease = tracker.complete(fence).expect("lease released");
        assert_eq!(lease.bytes(), block.bytes().len() as u64);
        for (t, c) in &lease.keys {
            loader.unpin(*t, *c);
        }
        assert_eq!(tracker.in_flight(), 0);
        assert_eq!(tracker.completed(), 1);
        assert!(loader.evict_all() >= 1);
        // The Arc still owns valid bytes after cache eviction and release.
        assert_eq!(block.generation, original_generation);
        assert!(!block.bytes().is_empty());
    }

    #[test]
    fn unknown_fences_are_noops() {
        let tracker = CompletionTracker::new();
        assert!(tracker.complete(42).is_none());
        assert_eq!(tracker.completed(), 0);
    }

    #[test]
    fn multiple_fences_are_tracked_independently() {
        let (_dir, loader) = loader_with_block();
        let tracker = CompletionTracker::new();
        let idx = loader.tensor_index("token_embd.weight").unwrap();
        let a = loader.block(idx, 0).unwrap();
        let b = loader.block(idx, 1).unwrap();
        let fa = tracker.submit(vec![a]);
        let fb = tracker.submit(vec![b]);
        assert_eq!(tracker.in_flight(), 2);
        assert!(tracker.complete(fa).is_some());
        assert_eq!(tracker.in_flight(), 1);
        assert!(tracker.complete(fb).is_some());
        assert_eq!(tracker.in_flight(), 0);
    }
}
