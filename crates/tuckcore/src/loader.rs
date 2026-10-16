//! Bounded weight-block loader.
//!
//! Reads tensor row-chunks with positioned reads (`pread`), caches them under
//! an explicit byte budget, and supports pinning so eviction can never drop a
//! block that in-flight work still references. Measured evidence (logical
//! reads, cache hits/misses, evictions) is tracked separately from
//! reservations (PRD §9, §13).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};

use crate::tensor::{self, RowGeometry};

#[derive(Debug, Clone, Copy)]
pub struct BlockPolicy {
    /// Target block size; bounds every single read.
    pub block_bytes: u64,
    /// Cache byte budget. Bounded reads, not bounded dreams: a single block
    /// may exceed the budget when the budget is smaller than one block.
    pub cache_bytes: u64,
}

impl Default for BlockPolicy {
    fn default() -> Self {
        Self {
            block_bytes: 4 * 1024 * 1024,
            cache_bytes: 512 * 1024 * 1024,
        }
    }
}

impl BlockPolicy {
    pub const MIN_BLOCK_BYTES: u64 = 64 * 1024;
    pub const MAX_BLOCK_BYTES: u64 = 64 * 1024 * 1024;
    pub const MIN_CACHE_BYTES: u64 = 64 * 1024;
    pub const MAX_CACHE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

    pub fn validate(&self) -> Result<()> {
        let bounds = |v: u64, lo: u64, hi: u64, what: &str| -> Result<()> {
            if v < lo || v > hi {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!("{what} {v} is outside {lo}..={hi} bytes"),
                ));
            }
            Ok(())
        };
        bounds(
            self.block_bytes,
            Self::MIN_BLOCK_BYTES,
            Self::MAX_BLOCK_BYTES,
            "block_bytes",
        )?;
        bounds(
            self.cache_bytes,
            Self::MIN_CACHE_BYTES,
            Self::MAX_CACHE_BYTES,
            "cache_bytes",
        )?;
        Ok(())
    }
}

/// One tensor in the artifact, with the geometry needed to stream it.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    pub type_id: u32,
    pub dims: Vec<u64>,
    pub rel_offset: u64,
    pub byte_size: u64,
    /// Present when the tensor is row-splittable; otherwise it loads whole.
    pub row: Option<RowGeometry>,
    /// Number of chunks this tensor is served in.
    pub chunks: u64,
}

/// A loaded, owned weight block. Ownership is explicit: eviction only drops
/// cache references; anything holding an `Arc` keeps its bytes alive.
#[derive(Debug)]
pub struct WeightBlock {
    pub tensor: usize,
    pub chunk: u32,
    /// Byte range relative to the tensor start.
    pub rel_start: u64,
    pub bytes: Vec<u8>,
    /// Monotonic load generation; lets tests prove reloads happened.
    pub generation: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LoaderSnapshot {
    pub logical_read_bytes: u64,
    pub blocks_loaded: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub cache_bytes_used: u64,
    pub pinned_blocks: u64,
}

#[derive(Default)]
struct Stats {
    logical_read_bytes: AtomicU64,
    blocks_loaded: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    evictions: AtomicU64,
}

#[derive(Default)]
struct Cache {
    map: HashMap<(usize, u32), Arc<WeightBlock>>,
    order: VecDeque<(usize, u32)>,
    pinned: HashSet<(usize, u32)>,
    used_bytes: u64,
}

/// The bounded loader. One process-wide instance per loaded artifact.
pub struct BlockLoader {
    file: File,
    data_start: u64,
    tensors: Vec<TensorEntry>,
    by_name: HashMap<String, usize>,
    cache: Mutex<Cache>,
    stats: Stats,
    generation: AtomicU64,
    pub policy: BlockPolicy,
}

impl BlockLoader {
    pub fn open(
        path: &Path,
        data_start: u64,
        tensors: Vec<TensorEntry>,
        policy: BlockPolicy,
    ) -> Result<Self> {
        policy.validate()?;
        let file = File::open(path).map_err(|e| {
            TuckmillError::new(
                ErrorCode::StorageLost,
                format!("cannot open artifact {}: {e}", path.display()),
            )
            .with_source(e)
        })?;
        let by_name = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        Ok(Self {
            file,
            data_start,
            tensors,
            by_name,
            cache: Mutex::new(Cache::default()),
            stats: Stats::default(),
            generation: AtomicU64::new(1),
            policy,
        })
    }

    pub fn tensor_index(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    pub fn entry(&self, index: usize) -> &TensorEntry {
        &self.tensors[index]
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Byte range covered by one chunk of a tensor.
    fn chunk_range(&self, entry: &TensorEntry, chunk: u32) -> (u64, u64) {
        match &entry.row {
            Some(row) => {
                let rows_per_chunk = row.rows_per_chunk;
                let first_row = (chunk as u64) * rows_per_chunk;
                let last_row = (first_row + rows_per_chunk).min(row.rows);
                (first_row * row.bytes_per_row, last_row * row.bytes_per_row)
            }
            None => {
                let start = (chunk as u64) * self.policy.block_bytes;
                let end = (start + self.policy.block_bytes).min(entry.byte_size);
                (start, end)
            }
        }
    }

    /// Fetch a block, reading from storage on a miss. Blocks are pinned only
    /// through [`Self::pin`]; the cache itself may evict at any time.
    pub fn block(&self, tensor: usize, chunk: u32) -> Result<Arc<WeightBlock>> {
        if tensor >= self.tensors.len() {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!("tensor index {tensor} out of range"),
            ));
        }
        let entry = &self.tensors[tensor];
        if chunk as u64 >= entry.chunks {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!("chunk {chunk} out of range for tensor {:?}", entry.name),
            ));
        }
        let key = (tensor, chunk);
        {
            let mut cache = self.cache.lock().expect("loader cache lock");
            if let Some(block) = cache.map.get(&key).cloned() {
                // Refresh recency.
                cache.order.retain(|k| *k != key);
                cache.order.push_back(key);
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(block);
            }
        }
        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);

        let (start, end) = self.chunk_range(entry, chunk);
        let len = (end - start) as usize;
        let mut bytes = vec![0u8; len];
        self.file
            .read_exact_at(&mut bytes, self.data_start + entry.rel_offset + start)
            .map_err(|e| {
                TuckmillError::new(
                    ErrorCode::StorageLost,
                    format!("artifact read failed for tensor {:?}: {e}", entry.name),
                )
                .with_source(e)
            })?;
        self.stats
            .logical_read_bytes
            .fetch_add(len as u64, Ordering::Relaxed);
        self.stats.blocks_loaded.fetch_add(1, Ordering::Relaxed);
        let generation = self.generation.fetch_add(1, Ordering::Relaxed);
        let block = Arc::new(WeightBlock {
            tensor,
            chunk,
            rel_start: start,
            bytes,
            generation,
        });

        let mut cache = self.cache.lock().expect("loader cache lock");
        cache.map.insert(key, block.clone());
        cache.order.retain(|k| *k != key);
        cache.order.push_back(key);
        cache.used_bytes += len as u64;
        self.evict_locked(&mut cache);
        Ok(block)
    }

    /// Read tensor rows directly without caching; used for embedding rows and
    /// other small random access where caching whole chunks would be waste.
    pub fn read_rows_direct(
        &self,
        tensor: usize,
        first_row: u64,
        row_count: u64,
    ) -> Result<Vec<u8>> {
        let entry = &self.tensors[tensor];
        let row = entry.row.as_ref().ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedQuantization,
                format!(
                    "tensor {:?} has no splittable row geometry; cannot random-access rows",
                    entry.name
                ),
            )
        })?;
        if first_row + row_count > row.rows {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!("row range outside tensor {:?}", entry.name),
            ));
        }
        let start = first_row * row.bytes_per_row;
        let len = (row_count * row.bytes_per_row) as usize;
        let mut bytes = vec![0u8; len];
        self.file
            .read_exact_at(&mut bytes, self.data_start + entry.rel_offset + start)
            .map_err(|e| {
                TuckmillError::new(
                    ErrorCode::StorageLost,
                    format!("artifact read failed for tensor {:?}: {e}", entry.name),
                )
                .with_source(e)
            })?;
        self.stats
            .logical_read_bytes
            .fetch_add(len as u64, Ordering::Relaxed);
        Ok(bytes)
    }

    /// Pin a block so `evict_all` and normal eviction skip it while in-flight
    /// work references it (released through the completion fence).
    pub fn pin(&self, tensor: usize, chunk: u32) {
        let mut cache = self.cache.lock().expect("loader cache lock");
        cache.pinned.insert((tensor, chunk));
    }

    pub fn unpin(&self, tensor: usize, chunk: u32) {
        let mut cache = self.cache.lock().expect("loader cache lock");
        cache.pinned.remove(&(tensor, chunk));
    }

    /// Forced eviction (tests, governor cache trimming, shutdown). Pinned
    /// blocks are never evicted. Returns the number of evicted blocks.
    pub fn evict_all(&self) -> usize {
        let mut cache = self.cache.lock().expect("loader cache lock");
        let before = cache.map.len();
        let pinned = cache.pinned.clone();
        let keys: Vec<(usize, u32)> = cache
            .map
            .keys()
            .copied()
            .filter(|k| !pinned.contains(k))
            .collect();
        for key in keys {
            if let Some(block) = cache.map.remove(&key) {
                cache.used_bytes = cache.used_bytes.saturating_sub(block.bytes.len() as u64);
                cache.order.retain(|k| *k != key);
            }
        }
        let evicted = before - cache.map.len();
        self.stats
            .evictions
            .fetch_add(evicted as u64, Ordering::Relaxed);
        evicted
    }

    fn evict_locked(&self, cache: &mut Cache) {
        while cache.used_bytes > self.policy.cache_bytes {
            // Find the oldest unpinned entry.
            let victim = cache
                .order
                .iter()
                .find(|k| !cache.pinned.contains(k))
                .copied();
            match victim {
                Some(key) => {
                    if let Some(block) = cache.map.remove(&key) {
                        cache.used_bytes =
                            cache.used_bytes.saturating_sub(block.bytes.len() as u64);
                    }
                    cache.order.retain(|k| *k != key);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
                None => break, // everything pinned: stay over budget briefly
            }
        }
    }

    pub fn stats(&self) -> LoaderSnapshot {
        let cache = self.cache.lock().expect("loader cache lock");
        LoaderSnapshot {
            logical_read_bytes: self.stats.logical_read_bytes.load(Ordering::Relaxed),
            blocks_loaded: self.stats.blocks_loaded.load(Ordering::Relaxed),
            cache_hits: self.stats.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.stats.cache_misses.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            cache_bytes_used: cache.used_bytes,
            pinned_blocks: cache.pinned.len() as u64,
        }
    }

    /// Dequantize a whole (small) tensor, streaming through the cache so the
    /// memory ceiling holds even for large 1D weights.
    pub fn dequant_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let index = self.tensor_index(name).ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::UnsupportedArchitecture,
                format!("artifact is missing required tensor {name:?}"),
            )
        })?;
        let entry = self.entry(index);
        let elems: u64 = entry.dims.iter().product();
        let mut out = Vec::with_capacity(elems as usize);
        for chunk in 0..entry.chunks {
            let block = self.block(index, chunk as u32)?;
            tensor::dequant_into(entry.type_id, &block.bytes, &mut out)?;
        }
        if out.len() as u64 != elems {
            return Err(TuckmillError::new(
                ErrorCode::IntegrityFailure,
                format!(
                    "tensor {name:?} decoded {} elements but its index declares {elems}",
                    out.len()
                ),
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

    fn open_fixture(policy: BlockPolicy) -> (tempfile::TempDir, BlockLoader) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
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
    fn reads_are_block_bounded_and_cached() {
        let policy = BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 256 * 1024,
        };
        let (_dir, loader) = open_fixture(policy);
        let idx = loader.tensor_index("token_embd.weight").unwrap();
        let entry = loader.entry(idx);
        assert!(entry.chunks >= 2, "128KiB embedding in 64KiB blocks");
        let b0 = loader.block(idx, 0).unwrap();
        let b1 = loader.block(idx, 0).unwrap();
        assert_eq!(b0.generation, b1.generation, "second read is a cache hit");
        let stats = loader.stats();
        assert_eq!(stats.blocks_loaded, 1);
        assert_eq!(stats.cache_hits, 1);
        assert!(b0.bytes.len() <= policy.block_bytes as usize);
    }

    #[test]
    fn small_cache_evicts_and_reloads_without_error() {
        let policy = BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 64 * 1024,
        };
        let (_dir, loader) = open_fixture(policy);
        let idx = loader.tensor_index("token_embd.weight").unwrap();
        let chunks = loader.entry(idx).chunks;
        // Walk all chunks twice; with a one-block cache the second pass must
        // reload everything (evictions > 0, same bytes).
        let first: Vec<Vec<u8>> = (0..chunks)
            .map(|c| loader.block(idx, c as u32).unwrap().bytes.clone())
            .collect();
        let second: Vec<Vec<u8>> = (0..chunks)
            .map(|c| loader.block(idx, c as u32).unwrap().bytes.clone())
            .collect();
        assert_eq!(first, second, "reloads must return identical bytes");
        let stats = loader.stats();
        assert!(stats.evictions > 0, "{stats:?}");
        assert_eq!(stats.blocks_loaded, chunks * 2);
    }

    #[test]
    fn pinned_blocks_survive_forced_eviction() {
        let policy = BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 256 * 1024,
        };
        let (_dir, loader) = open_fixture(policy);
        let idx = loader.tensor_index("token_embd.weight").unwrap();
        let b = loader.block(idx, 0).unwrap();
        loader.pin(idx, 0);
        let evicted = loader.evict_all();
        assert_eq!(evicted, 0, "pinned block must not be evicted");
        assert_eq!(loader.stats().pinned_blocks, 1);
        loader.unpin(idx, 0);
        assert!(loader.evict_all() >= 1, "unpinned block evicts");
        // The Arc still owns valid bytes after eviction.
        assert!(b.generation >= 1);
    }

    #[test]
    fn direct_row_reads_are_exact_slices_of_blocks() {
        let policy = BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 256 * 1024,
        };
        let (_dir, loader) = open_fixture(policy);
        let idx = loader.tensor_index("token_embd.weight").unwrap();
        let entry = loader.entry(idx);
        let row_bytes = entry.row.as_ref().unwrap().bytes_per_row as usize;
        let rows = loader.read_rows_direct(idx, 3, 2).unwrap();
        assert_eq!(rows.len(), row_bytes * 2);
        // Compare against the equivalent cached chunk (same tensor bytes).
        let chunk_containing = 3 / entry.row.as_ref().unwrap().rows_per_chunk;
        let block = loader.block(idx, chunk_containing as u32).unwrap();
        let rel = 3 * row_bytes as u64 - block.rel_start;
        assert_eq!(
            &block.bytes[rel as usize..rel as usize + rows.len()],
            &rows[..]
        );
    }

    #[test]
    fn bounds_are_validated() {
        let policy = BlockPolicy {
            block_bytes: 1,
            cache_bytes: 1,
        };
        assert!(policy.validate().is_err());
        assert!(BlockPolicy::default().validate().is_ok());
    }

    #[test]
    fn out_of_range_requests_are_internal_errors() {
        let (_dir, loader) = open_fixture(BlockPolicy::default());
        assert!(loader.block(10_000, 0).is_err());
        let idx = loader.tensor_index("output_norm.weight").unwrap();
        assert!(loader.block(idx, 9999).is_err());
    }
}
