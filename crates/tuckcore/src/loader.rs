//! Bounded weight-block loader.
//!
//! Reads tensor row-chunks with positioned reads (`pread`), caches them under
//! an explicit byte budget, and supports pinning so eviction can never drop a
//! block that in-flight work still references. Measured evidence (logical
//! reads, cache hits/misses, evictions) is tracked separately from
//! reservations (PRD §9, §13).
//!
//! ## Storage modes and what eviction proves
//!
//! Two block storages exist and are explicitly selectable: `owned` (every
//! block is copied into a bounded heap buffer) and `mapped` (blocks are
//! read-only views of one `mmap` of the artifact). Cache-entry eviction drops
//! the loader's *reference*; it is **not** proof that mapped pages left RAM.
//! Mapped-mode eviction additionally issues `madvise(MADV_DONTNEED)` on the
//! evicted range (advisory on macOS — recorded as `page_advice_bytes`), and
//! observed page residency is measured separately with `mincore`
//! ([`BlockLoader::residency`]). Owned and mapped byte counts are reported
//! separately and are never summed into one "memory used" number; charged
//! process footprint is measured by the supervisor (`proc_pid_rusage`) and is
//! a third, independently sampled metric.
//!
//! ## File identity and immutability
//!
//! [`BlockLoader::open_file`] takes one open `File` handle: positioned reads
//! and the mapping come from the *same* object, and its size is checked
//! against the expected artifact length before anything is mapped. Store
//! installs publish atomically (write to temp, rename), replacement unlinks
//! or renames rather than writing in place, and delete-while-open keeps the
//! inode alive for an open handle — so a loaded artifact cannot be mutated
//! from under execution by Tuckmill's own components. External in-place
//! modification of the payload is outside the contract: the content-digest
//! verification in the executor detects it before execution begins.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};

use crate::tensor::{self, RowGeometry};

/// Storage selection for weight blocks (a load-time option, recorded in
/// stats). `Auto` maps when the platform and file allow it and falls back to
/// owned reads; `Mapped` requires the zero-copy mapping; `Owned` never maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StorageSelection {
    Auto,
    Mapped,
    Owned,
}

impl StorageSelection {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "mapped" => Some(Self::Mapped),
            "owned" => Some(Self::Owned),
            _ => None,
        }
    }
}

/// The storage mode actually resolved for a loaded artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResolvedStorage {
    Mapped,
    Owned,
}

impl ResolvedStorage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mapped => "mapped",
            Self::Owned => "owned",
        }
    }
}

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

/// Read-only memory map of the artifact file. Opens once per loader; blocks
/// are zero-copy slices of it (the file was content-verified before any
/// block is served). Dropping all clones unmaps.
pub struct MappedFile {
    ptr: *const u8,
    len: usize,
}

// SAFETY: the mapping is read-only and immutable for its lifetime; the file
// is store-managed (imported atomically, content-verified at load) and no
// code writes through this map.
unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
    /// Map an already-open file. The mapping is created from this exact
    /// object; callers must have validated the file identity (and, where
    /// required, content) through the same handle.
    pub fn from_file(file: &File) -> std::io::Result<Arc<Self>> {
        use std::os::unix::io::AsRawFd;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "empty file cannot be mapped",
            ));
        }
        // SAFETY: standard read-only mmap of an open file; the pointer is
        // checked for MAP_FAILED and the mapping is unmapped exactly once in
        // Drop.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Arc::new(MappedFile {
            ptr: ptr as *const u8,
            len,
        }))
    }

    pub fn open(path: &Path) -> std::io::Result<Arc<Self>> {
        let file = File::open(path)?;
        Self::from_file(&file)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrowed slice; None when the requested range is out of bounds.
    pub fn slice(&self, offset: u64, len: usize) -> Option<&[u8]> {
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(len)?;
        if end > self.len {
            return None;
        }
        // SAFETY: range checked against the mapping length; read-only access.
        Some(unsafe { std::slice::from_raw_parts(self.ptr.add(start), len) })
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        // SAFETY: unmapping the exact region created in open().
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

impl std::fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MappedFile {{ len: {} }}", self.len)
    }
}

enum BlockStorage {
    Owned(Vec<u8>),
    Mapped {
        map: Arc<MappedFile>,
        offset: usize,
        len: usize,
    },
}

impl std::fmt::Debug for BlockStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockStorage::Owned(bytes) => write!(f, "Owned({})", bytes.len()),
            BlockStorage::Mapped { offset, len, .. } => {
                write!(f, "Mapped {{ offset: {offset}, len: {len} }}")
            }
        }
    }
}

/// A loaded weight block. Ownership is explicit: eviction only drops cache
/// references; anything holding an `Arc` keeps its bytes alive. With zero-copy
/// mapping enabled the bytes are a read-only slice of the artifact mmap (the
/// loader copies only when mapping is unavailable).
#[derive(Debug)]
pub struct WeightBlock {
    pub tensor: usize,
    pub chunk: u32,
    /// Byte range relative to the tensor start.
    pub rel_start: u64,
    storage: BlockStorage,
    /// Monotonic load generation; lets tests prove reloads happened.
    pub generation: u64,
}

impl WeightBlock {
    pub fn bytes(&self) -> &[u8] {
        match &self.storage {
            BlockStorage::Owned(bytes) => bytes,
            BlockStorage::Mapped { map, offset, len } => map
                .slice(*offset as u64, *len)
                .expect("block range validated when the block was created"),
        }
    }

    pub fn len(&self) -> usize {
        self.bytes().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True when this block is a zero-copy view of the artifact mapping
    /// (no owned heap buffer backs it).
    pub fn is_mapped(&self) -> bool {
        matches!(self.storage, BlockStorage::Mapped { .. })
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LoaderSnapshot {
    /// Logical weight bytes loaded (cache misses); not physical SSD traffic.
    pub logical_read_bytes: u64,
    pub blocks_loaded: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    /// Cache-entry evictions: entries dropped from the application cache.
    /// For mapped blocks this is **not** evidence that pages left RAM — see
    /// [`BlockLoader::residency`] and `page_advice_bytes`.
    pub cache_entry_evictions: u64,
    /// All bytes currently retained in the cache (owned + mapped views).
    pub cache_bytes_used: u64,
    /// Subset of `cache_bytes_used` that is owned heap buffers (copied).
    pub owned_cache_bytes: u64,
    pub pinned_blocks: u64,
    /// The storage mode actually resolved at load ("mapped"/"owned").
    pub storage_mode: String,
    /// Address space covered by the artifact mapping (0 in owned mode). This
    /// is virtual address space, not resident memory.
    pub mapped_vm_bytes: u64,
    /// Bytes handed to `madvise(MADV_DONTNEED)` on cache-entry eviction
    /// (advisory on macOS; recorded as issued advice, not as freed memory).
    pub page_advice_bytes: u64,
}

/// A point-in-time measurement of mapped-page residency (via `mincore`).
/// Distinct from address space and from charged process footprint; never
/// summed with either.
#[derive(Debug, Clone, Serialize)]
pub struct ResidencySample {
    pub page_size_bytes: u64,
    pub mapped_bytes: u64,
    pub scanned_bytes: u64,
    pub resident_bytes: u64,
    /// True when the mapping exceeded the scan cap and windows were sampled.
    pub sampled: bool,
}

#[derive(Default)]
struct Stats {
    logical_read_bytes: AtomicU64,
    blocks_loaded: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    cache_entry_evictions: AtomicU64,
    page_advice_bytes: AtomicU64,
}

#[derive(Default)]
struct Cache {
    map: HashMap<(usize, u32), Arc<WeightBlock>>,
    order: VecDeque<(usize, u32)>,
    pinned: HashSet<(usize, u32)>,
    used_bytes: u64,
    owned_bytes: u64,
}

/// The bounded loader. One process-wide instance per loaded artifact.
pub struct BlockLoader {
    file: File,
    mapped: Option<Arc<MappedFile>>,
    storage: ResolvedStorage,
    data_start: u64,
    tensors: Vec<TensorEntry>,
    by_name: HashMap<String, usize>,
    cache: Mutex<Cache>,
    stats: Stats,
    generation: AtomicU64,
    pub policy: BlockPolicy,
}

impl BlockLoader {
    /// Open by path with an explicit storage selection and no length check
    /// (convenience for tests/tooling; the executor uses
    /// [`Self::open_file`] after content verification).
    pub fn open_with_storage(
        path: &Path,
        data_start: u64,
        tensors: Vec<TensorEntry>,
        policy: BlockPolicy,
        storage: StorageSelection,
    ) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            TuckmillError::new(
                ErrorCode::StorageLost,
                format!("cannot open artifact {}: {e}", path.display()),
            )
            .with_source(e)
        })?;
        Self::open_file(file, data_start, tensors, policy, storage, None)
    }

    pub fn open(
        path: &Path,
        data_start: u64,
        tensors: Vec<TensorEntry>,
        policy: BlockPolicy,
    ) -> Result<Self> {
        Self::open_with_storage(path, data_start, tensors, policy, StorageSelection::Auto)
    }

    /// Build the loader around one already-open artifact handle. Positioned
    /// reads and the mapping use this exact object (no re-open by path), so
    /// the file identity verified by the caller is the file identity
    /// executed. When `expected_len` is set, the current size must match it —
    /// a truncated or drifted file is refused before anything is mapped.
    pub fn open_file(
        file: File,
        data_start: u64,
        tensors: Vec<TensorEntry>,
        policy: BlockPolicy,
        storage: StorageSelection,
        expected_len: Option<u64>,
    ) -> Result<Self> {
        policy.validate()?;
        let len = file.metadata().map_err(TuckmillError::from)?.len();
        if len == 0 {
            return Err(TuckmillError::new(
                ErrorCode::StorageLost,
                "artifact file is empty",
            ));
        }
        if let Some(expected) = expected_len {
            if len != expected {
                return Err(TuckmillError::new(
                    ErrorCode::IntegrityFailure,
                    format!(
                        "artifact file is {len} bytes but the recorded identity says {expected}; refusing to map or execute"
                    ),
                )
                .with_action("Re-import the artifact; never execute drifted weights."));
            }
        }
        let by_name = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        let (mapped, resolved) = match storage {
            StorageSelection::Owned => (None, ResolvedStorage::Owned),
            StorageSelection::Mapped => match MappedFile::from_file(&file) {
                Ok(map) => (Some(map), ResolvedStorage::Mapped),
                Err(e) => {
                    return Err(TuckmillError::new(
                        ErrorCode::StorageLost,
                        format!("mapped weight storage was requested but mmap failed: {e}"),
                    )
                    .with_source(e)
                    .with_action("Use --storage auto (map when possible) or --storage owned."))
                }
            },
            StorageSelection::Auto => match MappedFile::from_file(&file) {
                Ok(map) => (Some(map), ResolvedStorage::Mapped),
                Err(_) => (None, ResolvedStorage::Owned),
            },
        };
        Ok(Self {
            file,
            mapped,
            storage: resolved,
            data_start,
            tensors,
            by_name,
            cache: Mutex::new(Cache::default()),
            stats: Stats::default(),
            generation: AtomicU64::new(1),
            policy,
        })
    }

    /// The resolved storage mode for this loader.
    pub fn storage_mode(&self) -> ResolvedStorage {
        self.storage
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
        let file_offset = self.data_start + entry.rel_offset + start;
        let storage = match &self.mapped {
            Some(map) => BlockStorage::Mapped {
                map: map.clone(),
                offset: usize::try_from(file_offset).map_err(|_| {
                    TuckmillError::new(ErrorCode::Internal, "block offset does not fit memory")
                })?,
                len,
            },
            None => {
                let mut bytes = vec![0u8; len];
                self.file
                    .read_exact_at(&mut bytes, file_offset)
                    .map_err(|e| {
                        TuckmillError::new(
                            ErrorCode::StorageLost,
                            format!("artifact read failed for tensor {:?}: {e}", entry.name),
                        )
                        .with_source(e)
                    })?;
                BlockStorage::Owned(bytes)
            }
        };
        self.stats
            .logical_read_bytes
            .fetch_add(len as u64, Ordering::Relaxed);
        self.stats.blocks_loaded.fetch_add(1, Ordering::Relaxed);
        let generation = self.generation.fetch_add(1, Ordering::Relaxed);
        let block = Arc::new(WeightBlock {
            tensor,
            chunk,
            rel_start: start,
            storage,
            generation,
        });

        let mut cache = self.cache.lock().expect("loader cache lock");
        if !block.is_mapped() {
            cache.owned_bytes += len as u64;
        }
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

    fn drop_entry(&self, cache: &mut Cache, block: &WeightBlock) {
        cache.used_bytes = cache.used_bytes.saturating_sub(block.len() as u64);
        if !block.is_mapped() {
            cache.owned_bytes = cache.owned_bytes.saturating_sub(block.len() as u64);
        } else {
            self.advise_release(block);
        }
    }

    /// Advise the kernel that a mapped block's pages are reclaimable. This is
    /// advice, not a guaranteed release; the bytes are recorded as
    /// `page_advice_bytes` and observed residency is measured separately by
    /// [`Self::residency`].
    fn advise_release(&self, block: &WeightBlock) {
        let BlockStorage::Mapped { map, offset, len } = &block.storage else {
            return;
        };
        let page = page_size_bytes();
        let start = (*offset as u64 / page) * page;
        let end = ((*offset as u64 + *len as u64).div_ceil(page) * page).min(map.len() as u64);
        if end <= start {
            return;
        }
        // SAFETY: the range is inside the read-only mapping (clamped above);
        // MADV_DONTNEED is a read-only lifetime hint and cannot fault.
        unsafe {
            libc::madvise(
                map.ptr.add(start as usize) as *mut libc::c_void,
                (end - start) as usize,
                libc::MADV_DONTNEED,
            );
        }
        self.stats
            .page_advice_bytes
            .fetch_add(end - start, Ordering::Relaxed);
    }

    /// Forced eviction (tests, governor cache trimming, shutdown). Pinned
    /// blocks are never evicted. Returns the number of evicted blocks.
    /// Eviction drops cache references (and, for mapped blocks, issues page
    /// advice); it is not proof that pages left RAM.
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
                self.drop_entry(&mut cache, &block);
                cache.order.retain(|k| *k != key);
            }
        }
        let evicted = before - cache.map.len();
        self.stats
            .cache_entry_evictions
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
                        self.drop_entry(cache, &block);
                    }
                    cache.order.retain(|k| *k != key);
                    self.stats
                        .cache_entry_evictions
                        .fetch_add(1, Ordering::Relaxed);
                }
                None => break, // everything pinned: stay over budget briefly
            }
        }
    }

    /// True when blocks are zero-copy views over the artifact mapping.
    pub fn has_zero_copy(&self) -> bool {
        self.mapped.is_some()
    }

    pub fn stats(&self) -> LoaderSnapshot {
        let cache = self.cache.lock().expect("loader cache lock");
        LoaderSnapshot {
            logical_read_bytes: self.stats.logical_read_bytes.load(Ordering::Relaxed),
            blocks_loaded: self.stats.blocks_loaded.load(Ordering::Relaxed),
            cache_hits: self.stats.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.stats.cache_misses.load(Ordering::Relaxed),
            cache_entry_evictions: self.stats.cache_entry_evictions.load(Ordering::Relaxed),
            cache_bytes_used: cache.used_bytes,
            owned_cache_bytes: cache.owned_bytes,
            pinned_blocks: cache.pinned.len() as u64,
            storage_mode: self.storage.as_str().to_string(),
            mapped_vm_bytes: self.mapped.as_ref().map(|m| m.len() as u64).unwrap_or(0),
            page_advice_bytes: self.stats.page_advice_bytes.load(Ordering::Relaxed),
        }
    }

    /// Measure mapped-page residency via `mincore`. Returns `None` in owned
    /// mode or when the platform call fails. The scan is bounded: mappings
    /// above [`Self::MAX_FULL_RESIDENCY_SCAN`] are sampled in evenly spaced
    /// windows and reported as `sampled` — never extrapolated into a total.
    pub fn residency(&self) -> Option<ResidencySample> {
        let map = self.mapped.as_ref()?;
        let page = page_size_bytes();
        let mapped = map.len() as u64;
        let (scanned, resident, sampled) = if mapped <= Self::MAX_FULL_RESIDENCY_SCAN {
            let (resident, scanned) = mincore_resident_bytes(map.ptr, 0, mapped as usize, page)?;
            (scanned, resident, false)
        } else {
            let windows = 512usize;
            let window_bytes = 8 * 1024 * 1024usize;
            let span = mapped.saturating_sub(window_bytes as u64) / windows as u64;
            let mut resident = 0u64;
            let mut scanned = 0u64;
            for w in 0..windows {
                let offset = (w as u64 * span) as usize;
                let len = window_bytes.min(map.len().saturating_sub(offset));
                if len == 0 {
                    continue;
                }
                let (r, s) = mincore_resident_bytes(map.ptr, offset, len, page)?;
                resident += r;
                scanned += s;
            }
            (scanned, resident, true)
        };
        Some(ResidencySample {
            page_size_bytes: page,
            mapped_bytes: mapped,
            scanned_bytes: scanned,
            // mincore counts whole pages: never report more resident bytes
            // than the mapping covers.
            resident_bytes: resident.min(mapped),
            sampled,
        })
    }

    /// Mappings at or below this size are scanned fully by `residency()`.
    pub const MAX_FULL_RESIDENCY_SCAN: u64 = 8 * 1024 * 1024 * 1024;

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
            tensor::dequant_into(entry.type_id, block.bytes(), &mut out)?;
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

fn page_size_bytes() -> u64 {
    // SAFETY: sysconf is thread-safe and takes no pointers.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v <= 0 {
        4096
    } else {
        v as u64
    }
}

/// Count resident bytes in `[offset, offset+len)` of the mapping at `ptr`
/// using `mincore`. Returns `(resident_bytes, scanned_bytes)` page-aligned to
/// the containing pages; `None` when the call fails.
fn mincore_resident_bytes(
    ptr: *const u8,
    offset: usize,
    len: usize,
    page: u64,
) -> Option<(u64, u64)> {
    if len == 0 {
        return Some((0, 0));
    }
    let start = (offset as u64) / page * page;
    let end = (offset as u64 + len as u64).div_ceil(page) * page;
    let span = (end - start) as usize;
    let pages = span / page as usize;
    if pages == 0 {
        return Some((0, 0));
    }
    let mut vec = vec![0i8; pages];
    // SAFETY: the range lies inside the mapping (caller clamps), the output
    // buffer has one byte per page as required, and mincore only reads.
    let rc = unsafe {
        libc::mincore(
            ptr.add(start as usize) as *mut libc::c_void,
            span,
            vec.as_mut_ptr() as *mut libc::c_char,
        )
    };
    if rc != 0 {
        return None;
    }
    let resident_pages = vec.iter().filter(|&&b| b & 1 != 0).count() as u64;
    Some((resident_pages * page, span as u64))
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
        assert!(b0.bytes().len() <= policy.block_bytes as usize);
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
            .map(|c| loader.block(idx, c as u32).unwrap().bytes().to_vec())
            .collect();
        let second: Vec<Vec<u8>> = (0..chunks)
            .map(|c| loader.block(idx, c as u32).unwrap().bytes().to_vec())
            .collect();
        assert_eq!(first, second, "reloads must return identical bytes");
        let stats = loader.stats();
        assert!(stats.cache_entry_evictions > 0, "{stats:?}");
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
            &block.bytes()[rel as usize..rel as usize + rows.len()],
            &rows[..]
        );
    }

    #[test]
    fn q8_row_decodes_identically_to_the_llama_quantized_f16_twin() {
        // Cross-decoder probe: llama-quantize produced the F16 file by
        // decoding the Q8_0 file with llama.cpp's own reader. If our Q8_0
        // decode matches the F16 twin row-for-row (within f16 rounding),
        // our interpretation of real Q8_0 blocks is correct.
        let (Ok(q8_path), Ok(f16_path)) = (
            std::env::var("TUCKMILL_Q8_GGUF"),
            std::env::var("TUCKMILL_F16_GGUF"),
        ) else {
            eprintln!("skipping: set TUCKMILL_Q8_GGUF and TUCKMILL_F16_GGUF");
            return;
        };
        let open = |path: &str| -> BlockLoader {
            let insp = tuckmill_manifest::gguf::inspect(
                std::path::Path::new(path),
                &tuckmill_manifest::gguf::Limits::default(),
            )
            .unwrap();
            let policy = BlockPolicy::default();
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
            BlockLoader::open(std::path::Path::new(path), insp.data_start, tensors, policy).unwrap()
        };
        let q8 = open(&q8_path);
        let f16 = open(&f16_path);
        let q8_idx = q8.tensor_index("token_embd.weight").unwrap();
        let f16_idx = f16.tensor_index("token_embd.weight").unwrap();
        assert_eq!(q8.entry(q8_idx).type_id, 8, "expected Q8_0 source");
        assert_eq!(f16.entry(f16_idx).type_id, 1, "expected F16 twin");

        let mut worst = 0.0f32;
        for row in [0u64, 1, 1000, 12345, 151935] {
            let raw = q8.read_rows_direct(q8_idx, row, 1).unwrap();
            let mut q8_vals = Vec::new();
            tensor::dequant_into(8, &raw, &mut q8_vals).unwrap();
            let raw16 = f16.read_rows_direct(f16_idx, row, 1).unwrap();
            let mut f16_vals = Vec::new();
            tensor::dequant_into(1, &raw16, &mut f16_vals).unwrap();
            assert_eq!(q8_vals.len(), f16_vals.len());
            for (a, b) in q8_vals.iter().zip(f16_vals.iter()) {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(
            worst < 1e-2,
            "Q8_0 decode diverges from llama.cpp's own decode: worst |Δ| = {worst}"
        );
    }

    #[test]
    fn zero_copy_blocks_match_the_file_bytes_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        let raw = std::fs::read(&path).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let policy = BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 256 * 1024,
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
        assert!(loader.has_zero_copy(), "mmap should be available on macOS");
        let idx = loader.tensor_index("token_embd.weight").unwrap();
        let entry = loader.entry(idx);
        let block = loader.block(idx, 0).unwrap();
        let file_start = (insp.data_start + entry.rel_offset) as usize;
        assert_eq!(
            block.bytes(),
            &raw[file_start..file_start + block.bytes().len()],
            "mapped block must be byte-identical to the file"
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

    fn entries_for(
        insp: &tuckmill_manifest::gguf::GgufInspection,
        block_bytes: u64,
    ) -> Vec<TensorEntry> {
        insp.tensors
            .iter()
            .map(|t| {
                let row = tensor::row_geometry(t.type_id, &t.dims, block_bytes);
                let chunks = match &row {
                    Some(r) => r.rows.div_ceil(r.rows_per_chunk),
                    None => t.byte_size.div_ceil(block_bytes),
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
            .collect()
    }

    #[test]
    fn owned_storage_is_selectable_and_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        let raw = std::fs::read(&path).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let policy = BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 256 * 1024,
        };
        let loader = BlockLoader::open_with_storage(
            &path,
            insp.data_start,
            entries_for(&insp, policy.block_bytes),
            policy,
            StorageSelection::Owned,
        )
        .unwrap();
        assert_eq!(loader.storage_mode(), ResolvedStorage::Owned);
        assert!(!loader.has_zero_copy());
        assert!(loader.residency().is_none(), "no mapping, no residency");

        let idx = loader.tensor_index("token_embd.weight").unwrap();
        let entry = loader.entry(idx);
        let block = loader.block(idx, 0).unwrap();
        assert!(
            !block.is_mapped(),
            "owned storage never returns mapped views"
        );
        let file_start = (insp.data_start + entry.rel_offset) as usize;
        assert_eq!(
            block.bytes(),
            &raw[file_start..file_start + block.bytes().len()]
        );

        let stats = loader.stats();
        assert_eq!(stats.storage_mode, "owned");
        assert_eq!(stats.mapped_vm_bytes, 0);
        assert!(stats.owned_cache_bytes > 0, "{stats:?}");
        assert_eq!(stats.owned_cache_bytes, stats.cache_bytes_used);
    }

    #[test]
    fn mapped_storage_reports_vm_separately_from_measured_residency() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let policy = BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 256 * 1024,
        };
        let loader = BlockLoader::open_with_storage(
            &path,
            insp.data_start,
            entries_for(&insp, policy.block_bytes),
            policy,
            StorageSelection::Mapped,
        )
        .unwrap();
        assert_eq!(loader.storage_mode(), ResolvedStorage::Mapped);
        assert!(loader.has_zero_copy());

        // Touch a block through the mapping, then measure pages.
        let idx = loader.tensor_index("token_embd.weight").unwrap();
        let block = loader.block(idx, 0).unwrap();
        assert!(block.is_mapped());
        assert!(!block.bytes().is_empty());
        let sample = loader.residency().expect("mincore on a mapping");
        let file_len = std::fs::metadata(&path).unwrap().len();
        assert_eq!(sample.mapped_bytes, file_len);
        assert!(
            sample.resident_bytes > 0,
            "touched pages must be observed resident: {sample:?}"
        );
        assert!(sample.resident_bytes <= sample.scanned_bytes);

        let stats = loader.stats();
        assert_eq!(stats.storage_mode, "mapped");
        assert_eq!(stats.mapped_vm_bytes, sample.mapped_bytes);
        assert_eq!(stats.owned_cache_bytes, 0, "mapped blocks own no buffers");
    }

    #[test]
    fn auto_falls_back_to_owned_when_mapping_fails_and_mapped_is_strict() {
        // Directories open as Files but cannot be mapped: Auto must resolve
        // to owned without error, Mapped must refuse explicitly.
        let dir = tempfile::tempdir().unwrap();
        let policy = BlockPolicy::default();
        let auto = BlockLoader::open_with_storage(
            dir.path(),
            0,
            Vec::new(),
            policy,
            StorageSelection::Auto,
        )
        .unwrap();
        assert_eq!(auto.storage_mode(), ResolvedStorage::Owned);
        let mapped = BlockLoader::open_with_storage(
            dir.path(),
            0,
            Vec::new(),
            policy,
            StorageSelection::Mapped,
        );
        let err = mapped
            .err()
            .expect("mapped storage must refuse unmappable files");
        assert_eq!(err.code(), ErrorCode::StorageLost);
        assert!(err.message().contains("mmap failed"), "{}", err.message());
    }

    #[test]
    fn replacement_by_rename_does_not_change_the_open_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        let raw_before = std::fs::read(&path).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let policy = BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 256 * 1024,
        };
        // The loader holds the original handle; mapping and reads use it.
        let file = File::open(&path).unwrap();
        let loader = BlockLoader::open_file(
            file,
            insp.data_start,
            entries_for(&insp, policy.block_bytes),
            policy,
            StorageSelection::Auto,
            Some(raw_before.len() as u64),
        )
        .unwrap();

        // Atomically replace the path with a different (truncated-shape) byte
        // stream: the open handle must keep serving the original inode.
        let other = dir.path().join("other.gguf");
        std::fs::write(&other, vec![0xAAu8; raw_before.len()]).unwrap();
        std::fs::rename(&other, &path).unwrap();
        let raw_after = std::fs::read(&path).unwrap();
        assert_ne!(raw_before, raw_after, "path now names different bytes");

        let idx = loader.tensor_index("token_embd.weight").unwrap();
        let entry = loader.entry(idx);
        let block = loader.block(idx, 0).unwrap();
        let file_start = (insp.data_start + entry.rel_offset) as usize;
        assert_eq!(
            block.bytes(),
            &raw_before[file_start..file_start + block.bytes().len()],
            "execution follows the verified open file, not the path"
        );
    }

    #[test]
    fn truncation_is_refused_when_the_expected_length_is_known() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        let original_len = std::fs::metadata(&path).unwrap().len();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let policy = BlockPolicy::default();
        let tensors = entries_for(&insp, policy.block_bytes);

        // Truncate the payload before opening: the recorded identity says
        // original_len, so the loader must refuse before mapping anything.
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(original_len - 1).unwrap();
        let file = File::open(&path).unwrap();
        let err = BlockLoader::open_file(
            file,
            insp.data_start,
            tensors.clone(),
            policy,
            StorageSelection::Auto,
            Some(original_len),
        )
        .err()
        .expect("truncated artifact must be refused");
        assert_eq!(err.code(), ErrorCode::IntegrityFailure);
        assert!(
            err.message().contains("refusing to map or execute"),
            "{}",
            err.message()
        );

        // Without the recorded length the file still opens (the executor's
        // digest pass is what refuses drifted content); this proves the
        // length check is the second line of defense, not the only one.
        let file = File::open(&path).unwrap();
        assert!(BlockLoader::open_file(
            file,
            insp.data_start,
            tensors,
            policy,
            StorageSelection::Auto,
            None
        )
        .is_ok());
    }

    #[test]
    fn eviction_evidence_distinguishes_entries_advice_and_residency() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        let insp =
            tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let policy = BlockPolicy {
            block_bytes: 64 * 1024,
            cache_bytes: 256 * 1024,
        };
        let loader = BlockLoader::open_with_storage(
            &path,
            insp.data_start,
            entries_for(&insp, policy.block_bytes),
            policy,
            StorageSelection::Mapped,
        )
        .unwrap();
        let idx = loader.tensor_index("token_embd.weight").unwrap();
        for chunk in [0u32, 1] {
            let block = loader.block(idx, chunk).unwrap();
            std::hint::black_box(block.bytes()[0]);
        }
        let before = loader.residency().unwrap();
        assert!(before.resident_bytes > 0, "{before:?}");

        let evicted = loader.evict_all();
        assert!(evicted >= 1, "unpinned entries evict");
        let stats = loader.stats();
        assert_eq!(stats.cache_entry_evictions, evicted as u64);
        assert!(
            stats.page_advice_bytes > 0,
            "mapped eviction issues page advice: {stats:?}"
        );
        // Residency is re-measured, not assumed: the sample stays bounded by
        // the mapping and never exceeds it.
        let after = loader.residency().unwrap();
        assert!(after.resident_bytes <= after.scanned_bytes);
        assert!(after.resident_bytes <= before.resident_bytes.max(after.scanned_bytes));
    }
}
