//! Bounded packer: lossless, layout-preserving packing of a validated GGUF.
//!
//! Repacking changes layout, not quantized values (PRD §10). The packed
//! format v1 stores the original header region plus immutable data segments
//! with 64-bit offsets; every tile carries a checksum and the decoding kernel
//! family it will need. The packer streams with a 4MiB copy buffer — far
//! below the proposed 512MiB controlled-memory ceiling — and never
//! instantiates the full model or a full-precision intermediate.
//!
//! Interrupted work resumes from a journal only when the source identity
//! still matches, and a completed pack is verified (checksums re-read) before
//! it is published atomically.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::Digest;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ids::{now_unix_ms, Id};
use tuckmill_manifest::records::{ModelArtifact, PackedInfo};

use crate::aliases;
use crate::layout::StoreLayout;
use crate::lock::FileLock;

pub const PACK_FORMAT_VERSION: u32 = 1;
pub const DEFAULT_SEGMENT_BYTES: u64 = 1024 * 1024 * 1024;
const COPY_BUFFER: usize = 4 * 1024 * 1024;
const TILE_ALIGNMENT: u64 = 64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PackedTile {
    pub name: String,
    pub dims: Vec<u64>,
    pub type_id: u32,
    pub type_name: String,
    /// Offset in the logical packed data region (sum of segment sizes).
    pub offset: u64,
    pub byte_size: u64,
    pub sha256: String,
    /// Required decoding kernel family ("K01", "K04", "K06", ...) or None for
    /// pass-through types.
    pub decoding_kernel: Option<String>,
    /// Index of the owning tile when this tile aliases the same bytes
    /// (tied/shared tensors); None when this tile owns its storage.
    pub shared_with: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentRecord {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackManifest {
    pub schema_version: u32,
    pub format: String,
    pub artifact_id: Id,
    pub source_content_hash: Id,
    pub source_file_bytes: u64,
    pub source_data_start: u64,
    pub alignment: u64,
    pub header_bytes: u64,
    pub header_sha256: String,
    pub tiles: Vec<PackedTile>,
    pub segments: Vec<SegmentRecord>,
    pub created_unix_ms: u64,
}

#[derive(Debug, Clone)]
pub struct PackOptions {
    pub segment_bytes: u64,
    /// Fault injection for tests: stop cleanly after N unique tiles, leaving
    /// a resumable journal behind. Never set in normal operation.
    pub fault_inject_stop_after_tiles: Option<usize>,
}

impl Default for PackOptions {
    fn default() -> Self {
        Self {
            segment_bytes: DEFAULT_SEGMENT_BYTES,
            fault_inject_stop_after_tiles: None,
        }
    }
}

#[derive(Debug)]
pub struct PackOutcome {
    pub packed: PackedInfo,
    pub resumed: bool,
    pub already_packed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PackJournal {
    schema_version: u32,
    source_content_hash: Id,
    source_data_start: u64,
    header_written: bool,
    tiles: Vec<PackedTile>,
    segments: Vec<SegmentRecord>,
    next_source_offset: u64,
}

/// Pack a validated artifact already present in the store. Returns the
/// updated artifact record (with `packed` filled in) after publishing.
pub fn pack_artifact(
    layout: &StoreLayout,
    artifact: &ModelArtifact,
    options: &PackOptions,
) -> Result<(ModelArtifact, PackOutcome)> {
    let id = &artifact.artifact_id;
    let payload = layout.artifact_payload_path(id);
    if !payload.exists() {
        return Err(TuckmillError::new(
            ErrorCode::StorageLost,
            format!("artifact payload for {id} is missing"),
        ));
    }
    let nbytes = std::fs::metadata(&payload)
        .map_err(TuckmillError::from)?
        .len();
    layout.ensure_space_for(
        tuckmill_manifest::units::Bytes(nbytes.saturating_add(16 * 1024 * 1024)),
        &layout.artifacts_dir(),
    )?;

    let lock_path = layout
        .tmp_dir()
        .join(format!("pack-{}.lock", &id.as_str()[..16]));
    let _lock = FileLock::exclusive(&lock_path)?;

    let pack_dir = layout.artifact_pack_dir(id);
    if pack_dir.join("pack.json").exists() {
        match verify_pack(layout, id) {
            Ok(()) => {
                let packed = packed_info_from(&pack_dir)?;
                return Ok((
                    artifact.clone(),
                    PackOutcome {
                        packed,
                        resumed: false,
                        already_packed: true,
                    },
                ));
            }
            Err(_) => {
                // A failed prior pack is derived data; replace it.
                std::fs::remove_dir_all(&pack_dir).map_err(TuckmillError::from)?;
            }
        }
    }

    let tmp_dir = layout
        .tmp_dir()
        .join(format!("pack-{}-partial", &id.as_str()[..16]));
    std::fs::create_dir_all(&tmp_dir).map_err(TuckmillError::from)?;

    // Validate the source again: packing must not operate on a drifted file.
    let mut file = std::fs::File::open(&payload).map_err(TuckmillError::from)?;
    let inspection = tuckmill_manifest::gguf::inspect_reader(
        &mut std::io::BufReader::with_capacity(1 << 20, &mut file),
        nbytes,
        &tuckmill_manifest::gguf::Limits::default(),
        None,
    )?;
    // The artifact's metadata/tensor hashes must match the payload on disk.
    if inspection.metadata_hash()? != artifact.metadata_hash
        || inspection.tensor_index_hash()? != artifact.tensor_index_hash
    {
        return Err(TuckmillError::new(
            ErrorCode::IntegrityFailure,
            "artifact payload does not match its manifest hashes; refusing to pack",
        ));
    }

    let journal_path = tmp_dir.join("pack.journal.json");
    let mut journal = match PackJournal::load(&journal_path) {
        Ok(j) if j.source_content_hash == artifact.content_hash => j,
        _ => {
            // Fresh start: reset the partial directory.
            for entry in std::fs::read_dir(&tmp_dir)
                .map_err(TuckmillError::from)?
                .flatten()
            {
                let _ = std::fs::remove_file(entry.path());
            }
            PackJournal {
                schema_version: 1,
                source_content_hash: artifact.content_hash.clone(),
                source_data_start: inspection.data_start,
                header_written: false,
                tiles: Vec::new(),
                segments: Vec::new(),
                next_source_offset: 0,
            }
        }
    };
    let resumed = !journal.tiles.is_empty() || journal.header_written;

    // ---- header region ----
    if !journal.header_written {
        let mut header = vec![0u8; inspection.data_start as usize];
        file.seek(SeekFrom::Start(0)).map_err(TuckmillError::from)?;
        file.read_exact(&mut header).map_err(TuckmillError::from)?;
        std::fs::write(tmp_dir.join("header.bin"), &header).map_err(TuckmillError::from)?;
        journal.header_written = true;
        journal.save(&journal_path)?;
    }

    // ---- deduplicate aliased extents ----
    // (source_offset, byte_size) -> owner tile index
    let mut extents: BTreeMap<(u64, u64), usize> = BTreeMap::new();
    let mut ordered: Vec<(usize, &tuckmill_manifest::gguf::GgufTensor, Option<usize>)> = Vec::new();
    for (idx, t) in inspection.tensors.iter().enumerate() {
        let owner = if t.byte_size == 0 {
            None
        } else {
            match extents.get(&(t.offset, t.byte_size)) {
                Some(&owner) => Some(owner),
                None => {
                    extents.insert((t.offset, t.byte_size), idx);
                    None
                }
            }
        };
        ordered.push((idx, t, owner));
    }

    // ---- copy tiles (resumable at tile granularity) ----
    let to_do: Vec<_> = ordered
        .iter()
        .filter(|(idx, _, owner)| {
            owner.is_none()
                && !journal
                    .tiles
                    .iter()
                    .any(|t| t.name == inspection.tensors[*idx].name)
        })
        .collect();
    for (done, (idx, t, _)) in to_do.iter().enumerate() {
        if let Some(limit) = options.fault_inject_stop_after_tiles {
            if done >= limit {
                journal.save(&journal_path)?;
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "fault injection: packing interrupted after requested tiles",
                ));
            }
        }
        let (segment_name, segment_file, offset_in_segment) =
            segment_for_tile(&tmp_dir, &journal, options.segment_bytes, t.byte_size)?;
        let align_pad = (TILE_ALIGNMENT - (offset_in_segment % TILE_ALIGNMENT)) % TILE_ALIGNMENT;
        let mut writer = std::io::BufWriter::with_capacity(COPY_BUFFER, segment_file);
        if align_pad > 0 {
            let pad = vec![0u8; align_pad as usize];
            writer.write_all(&pad).map_err(TuckmillError::from)?;
        }
        file.seek(SeekFrom::Start(inspection.data_start + t.offset))
            .map_err(TuckmillError::from)?;
        let mut hasher = sha2::Sha256::new();
        let mut remaining = t.byte_size;
        let mut buf = vec![0u8; COPY_BUFFER];
        while remaining > 0 {
            let take = remaining.min(buf.len() as u64) as usize;
            file.read_exact(&mut buf[..take])
                .map_err(TuckmillError::from)?;
            {
                use sha2::Digest;
                hasher.update(&buf[..take]);
            }
            writer
                .write_all(&buf[..take])
                .map_err(TuckmillError::from)?;
            remaining -= take as u64;
        }
        writer.flush().map_err(TuckmillError::from)?;
        let segment_meta = writer
            .into_inner()
            .map_err(|e| TuckmillError::new(ErrorCode::Io, format!("segment flush failed: {e}")))?;
        segment_meta.sync_all().map_err(TuckmillError::from)?;

        // Logical offset of this tile across the concatenated segments.
        let segment_base: u64 = journal
            .segments
            .iter()
            .filter(|s| s.name != segment_name)
            .map(|s| s.bytes)
            .sum();
        let tile = PackedTile {
            name: t.name.clone(),
            dims: t.dims.clone(),
            type_id: t.type_id,
            type_name: t.type_name.clone(),
            offset: segment_base + offset_in_segment + align_pad,
            byte_size: t.byte_size,
            sha256: {
                use sha2::Digest;
                tuckmill_manifest::ids::hex(&hasher.finalize())
            },
            decoding_kernel: decoding_kernel_for(t.type_id),
            shared_with: None,
        };
        match journal.segments.iter_mut().find(|s| s.name == segment_name) {
            Some(s) => s.bytes = offset_in_segment + align_pad + t.byte_size,
            None => {
                journal.segments.push(SegmentRecord {
                    name: segment_name,
                    bytes: offset_in_segment + align_pad + t.byte_size,
                    sha256: String::new(), // filled during finalize
                });
            }
        }
        journal.tiles.push(tile);
        journal.next_source_offset = t.offset + t.byte_size;
        journal.save(&journal_path)?;
        let _ = idx;
    }

    // ---- store aliased tiles with shared_with pointers ----
    for (_idx, t, owner) in &ordered {
        if let Some(owner_idx) = owner {
            let owner_name = &inspection.tensors[*owner_idx].name;
            let owner_pos = journal
                .tiles
                .iter()
                .position(|pt| &pt.name == owner_name)
                .ok_or_else(|| {
                    TuckmillError::new(
                        ErrorCode::Internal,
                        format!("aliased tile {tname:?} has no owner", tname = t.name),
                    )
                })?;
            let owner_tile = &journal.tiles[owner_pos];
            let alias_tile = PackedTile {
                name: t.name.clone(),
                dims: t.dims.clone(),
                type_id: t.type_id,
                type_name: t.type_name.clone(),
                offset: owner_tile.offset,
                byte_size: owner_tile.byte_size,
                sha256: owner_tile.sha256.clone(),
                decoding_kernel: decoding_kernel_for(t.type_id),
                shared_with: Some(owner_pos),
            };
            if !journal.tiles.iter().any(|pt| pt.name == alias_tile.name) {
                journal.tiles.push(alias_tile);
            }
        }
    }

    // ---- finalize: segment hashes, header hash, manifest ----
    let mut segments = Vec::new();
    for seg in &journal.segments {
        let path = tmp_dir.join(&seg.name);
        let (hash, len) = crate::digester::digest_file(&path, 64 * 1024 * 1024).map_err(|e| {
            TuckmillError::new(
                ErrorCode::Io,
                format!("cannot re-read segment {}: {}", seg.name, e.message()),
            )
        })?;
        segments.push(SegmentRecord {
            name: seg.name.clone(),
            bytes: len,
            sha256: hash.as_str().to_string(),
        });
    }
    let header_path = tmp_dir.join("header.bin");
    let header_sha = tuckmill_manifest::ids::sha256_of_reader(&mut std::io::BufReader::new(
        std::fs::File::open(&header_path).map_err(TuckmillError::from)?,
    ))
    .map_err(TuckmillError::from)?;
    let header_bytes = std::fs::metadata(&header_path)
        .map_err(TuckmillError::from)?
        .len();

    let manifest = PackManifest {
        schema_version: PACK_FORMAT_VERSION,
        format: "tuckmill-packed-v1".into(),
        artifact_id: id.clone(),
        source_content_hash: artifact.content_hash.clone(),
        source_file_bytes: nbytes,
        source_data_start: inspection.data_start,
        alignment: TILE_ALIGNMENT,
        header_bytes,
        header_sha256: header_sha,
        tiles: journal.tiles.clone(),
        segments: segments.clone(),
        created_unix_ms: now_unix_ms(),
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| {
        tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "pack manifest")
    })?;
    std::fs::write(tmp_dir.join("pack.json"), &manifest_bytes).map_err(TuckmillError::from)?;

    // ---- publish atomically ----
    std::fs::rename(&tmp_dir, &pack_dir).map_err(TuckmillError::from)?;
    crate::layout::fsync_dir(&layout.artifact_dir(id));

    // ---- verify the published pack before recording it ----
    verify_pack(layout, id)?;

    let packed = packed_info_from(&pack_dir)?;
    let mut updated = artifact.clone();
    updated.packed = Some(packed.clone());
    aliases::save_manifest(layout, &updated)?;

    Ok((
        updated,
        PackOutcome {
            packed,
            resumed,
            already_packed: false,
        },
    ))
}

impl PackJournal {
    fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(TuckmillError::from)?;
        serde_json::from_slice(&bytes).map_err(|e| {
            tuckmill_manifest::error::json_error(e, ErrorCode::IntegrityFailure, "pack journal")
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec(self).map_err(|e| {
            tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "pack journal write")
        })?;
        std::fs::write(&tmp, bytes).map_err(TuckmillError::from)?;
        std::fs::rename(&tmp, path).map_err(TuckmillError::from)?;
        Ok(())
    }
}

/// Select (or open) the segment that receives the next tile. A segment
/// rotates when the incoming tile (plus alignment) no longer fits.
fn segment_for_tile(
    tmp_dir: &Path,
    journal: &PackJournal,
    segment_bytes: u64,
    tile_bytes: u64,
) -> Result<(String, std::fs::File, u64)> {
    let (name, offset, fresh) = match journal.segments.last() {
        Some(last) if last.bytes + TILE_ALIGNMENT + tile_bytes <= segment_bytes => {
            (last.name.clone(), last.bytes, false)
        }
        _ => (format!("data.{}", journal.segments.len()), 0, true),
    };
    let path = tmp_dir.join(&name);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .map_err(TuckmillError::from)?;
    if fresh {
        file.set_len(0).map_err(TuckmillError::from)?;
    } else {
        // Resume: drop any bytes past the recorded append point.
        file.set_len(offset).map_err(TuckmillError::from)?;
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(TuckmillError::from)?;
    Ok((name, file, offset))
}

/// Required decoding kernel family for a stored type (PRD §11.3 K01/K04/K06).
fn decoding_kernel_for(type_id: u32) -> Option<String> {
    match type_id {
        34 | 35 => Some("K04".to_string()), // TQ1_0 / TQ2_0 ternary
        2 | 3 | 6 | 7 | 8 | 9 | 10..=23 | 39 | 40 | 41 | 42 => Some("K01".to_string()),
        0 | 1 | 30 => Some("K06".to_string()), // F32 / F16 / BF16 arithmetic
        _ => None,
    }
}

fn read_manifest(pack_dir: &Path) -> Result<PackManifest> {
    let bytes = std::fs::read(pack_dir.join("pack.json")).map_err(TuckmillError::from)?;
    serde_json::from_slice(&bytes).map_err(|e| {
        tuckmill_manifest::error::json_error(e, ErrorCode::IntegrityFailure, "pack manifest")
    })
}

fn packed_info_from(pack_dir: &Path) -> Result<PackedInfo> {
    let manifest = read_manifest(pack_dir)?;
    let total: u64 = manifest.segments.iter().map(|s| s.bytes).sum();
    let (manifest_hash, _) =
        crate::digester::digest_file(&pack_dir.join("pack.json"), 64 * 1024 * 1024)?;
    Ok(PackedInfo {
        format_version: PACK_FORMAT_VERSION,
        content_hash: manifest_hash,
        file_bytes: total + manifest.header_bytes,
        segments: manifest.segments.len(),
    })
}

/// Verify a published pack: header hash, segment sizes, and every unique
/// tile's checksum. Aliased tiles reuse their owner's stored bytes.
pub fn verify_pack(layout: &StoreLayout, id: &Id) -> Result<()> {
    let pack_dir = layout.artifact_pack_dir(id);
    let manifest = read_manifest(&pack_dir)?;
    if manifest.schema_version != PACK_FORMAT_VERSION {
        return Err(TuckmillError::new(
            ErrorCode::IntegrityFailure,
            format!("unsupported pack schema {}", manifest.schema_version),
        ));
    }
    // Header.
    let header_path = pack_dir.join("header.bin");
    let header_len = std::fs::metadata(&header_path)
        .map_err(|_| TuckmillError::new(ErrorCode::IntegrityFailure, "pack header.bin is missing"))?
        .len();
    if header_len != manifest.header_bytes {
        return Err(TuckmillError::new(
            ErrorCode::IntegrityFailure,
            "pack header length does not match the manifest",
        ));
    }
    let header_sha = tuckmill_manifest::ids::sha256_of_reader(&mut std::io::BufReader::new(
        std::fs::File::open(&header_path).map_err(TuckmillError::from)?,
    ))
    .map_err(TuckmillError::from)?;
    if header_sha != manifest.header_sha256 {
        return Err(TuckmillError::new(
            ErrorCode::IntegrityFailure,
            "pack header checksum mismatch",
        ));
    }
    // Segments: sizes plus tile checksums read back from the stored bytes.
    let mut segment_files: BTreeMap<&str, (u64, std::fs::File)> = BTreeMap::new();
    for seg in &manifest.segments {
        let path = pack_dir.join(&seg.name);
        let meta = std::fs::metadata(&path).map_err(|_| {
            TuckmillError::new(
                ErrorCode::IntegrityFailure,
                format!("pack segment {} is missing", seg.name),
            )
        })?;
        if meta.len() != seg.bytes {
            return Err(TuckmillError::new(
                ErrorCode::IntegrityFailure,
                format!(
                    "pack segment {} is {} bytes but the manifest records {}",
                    seg.name,
                    meta.len(),
                    seg.bytes
                ),
            ));
        }
        let file = std::fs::File::open(&path).map_err(TuckmillError::from)?;
        segment_files.insert(seg.name.as_str(), (seg.bytes, file));
    }
    // Build logical offset -> (segment, offset) map.
    let mut bounds = Vec::new();
    let mut base = 0u64;
    for seg in &manifest.segments {
        bounds.push((base, base + seg.bytes, seg.name.as_str()));
        base += seg.bytes;
    }
    let locate = |offset: u64| -> Option<(&str, u64)> {
        for (lo, hi, name) in &bounds {
            if offset >= *lo && offset < *hi {
                return Some((name, offset - lo));
            }
        }
        None
    };
    for tile in &manifest.tiles {
        if tile.shared_with.is_some() {
            continue;
        }
        let (name, off) = locate(tile.offset).ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::IntegrityFailure,
                format!("tile {} points outside all segments", tile.name),
            )
        })?;
        let (_, file) = segment_files.get_mut(name).expect("segment exists");
        file.seek(SeekFrom::Start(off))
            .map_err(TuckmillError::from)?;
        let mut hasher = sha2::Sha256::new();
        let mut remaining = tile.byte_size;
        let mut buf = vec![0u8; COPY_BUFFER];
        while remaining > 0 {
            let take = remaining.min(buf.len() as u64) as usize;
            file.read_exact(&mut buf[..take]).map_err(|_| {
                TuckmillError::new(
                    ErrorCode::IntegrityFailure,
                    format!("tile {} extends beyond its segment", tile.name),
                )
            })?;
            {
                use sha2::Digest;
                hasher.update(&buf[..take]);
            }
            remaining -= take as u64;
        }
        use sha2::Digest;
        let digest = tuckmill_manifest::ids::hex(&hasher.finalize());
        if digest != tile.sha256 {
            return Err(TuckmillError::new(
                ErrorCode::IntegrityFailure,
                format!("tile {} checksum mismatch", tile.name),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::{import_local, ImportOptions};
    use tuckmill_manifest::adapter::AdapterRegistry;
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    fn setup() -> (
        tempfile::TempDir,
        StoreLayout,
        ModelArtifact,
        AdapterRegistry,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        let src = dir.path().join("model.gguf");
        std::fs::write(&src, synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3())).unwrap();
        let registry = AdapterRegistry::builtin().unwrap();
        let outcome = import_local(&layout, &registry, &src, &ImportOptions::default()).unwrap();
        (dir, layout, outcome.artifact, registry)
    }

    #[test]
    fn packs_lossless_with_checksums_and_publishes_atomically() {
        let (_dir, layout, artifact, _registry) = setup();
        let (updated, outcome) =
            pack_artifact(&layout, &artifact, &PackOptions::default()).unwrap();
        assert!(!outcome.already_packed);
        assert!(updated.packed.is_some());
        let pack_dir = layout.artifact_pack_dir(&artifact.artifact_id);
        assert!(pack_dir.join("pack.json").exists());
        verify_pack(&layout, &artifact.artifact_id).unwrap();

        // Every source tensor is represented exactly once as an owner or alias.
        let manifest = read_manifest(&pack_dir).unwrap();
        let insp_path = layout.artifact_payload_path(&artifact.artifact_id);
        let insp = tuckmill_manifest::gguf::inspect(
            &insp_path,
            &tuckmill_manifest::gguf::Limits::default(),
        )
        .unwrap();
        assert_eq!(manifest.tiles.len(), insp.tensors.len());
        // Lossless: tile bytes sum to the source tensor byte total.
        let owners: u64 = manifest
            .tiles
            .iter()
            .filter(|t| t.shared_with.is_none())
            .map(|t| t.byte_size)
            .sum();
        let src_bytes: u64 = insp.tensors.iter().map(|t| t.byte_size).sum();
        assert_eq!(
            owners, src_bytes,
            "no tensor bytes may be lost or duplicated"
        );
        // Manifest updated on disk.
        let reloaded = aliases::load_manifest(&layout, &artifact.artifact_id).unwrap();
        assert!(reloaded.packed.is_some());
    }

    #[test]
    fn packing_is_idempotent_and_detects_corruption() {
        let (_dir, layout, artifact, _registry) = setup();
        let (_, first) = pack_artifact(&layout, &artifact, &PackOptions::default()).unwrap();
        assert!(!first.already_packed);
        let (_, second) = pack_artifact(&layout, &artifact, &PackOptions::default()).unwrap();
        assert!(second.already_packed);

        // Corrupt one byte in a segment: verification must fail and a repack
        // must rebuild from the validated source.
        let pack_dir = layout.artifact_pack_dir(&artifact.artifact_id);
        let manifest = read_manifest(&pack_dir).unwrap();
        let seg = pack_dir.join(&manifest.segments[0].name);
        let mut bytes = std::fs::read(&seg).unwrap();
        bytes[100] ^= 0xff;
        std::fs::write(&seg, &bytes).unwrap();
        let err = verify_pack(&layout, &artifact.artifact_id).unwrap_err();
        assert_eq!(err.code(), ErrorCode::IntegrityFailure);

        let (_, rebuilt) = pack_artifact(&layout, &artifact, &PackOptions::default()).unwrap();
        assert!(!rebuilt.already_packed, "corrupt pack must be rebuilt");
        verify_pack(&layout, &artifact.artifact_id).unwrap();
    }

    #[test]
    fn interrupted_pack_resumes_and_completes() {
        let (_dir, layout, artifact, _registry) = setup();
        let options = PackOptions {
            segment_bytes: 256 * 1024,
            fault_inject_stop_after_tiles: Some(5),
        };
        let err = pack_artifact(&layout, &artifact, &options).unwrap_err();
        assert_eq!(
            err.code(),
            ErrorCode::WorkerFailed,
            "fault injection stops the pack"
        );
        // No published pack may exist after an interrupted run.
        assert!(!layout
            .artifact_pack_dir(&artifact.artifact_id)
            .join("pack.json")
            .exists());

        // Resume with the same options minus the fault.
        let options2 = PackOptions {
            segment_bytes: 256 * 1024,
            fault_inject_stop_after_tiles: None,
        };
        let (_, outcome) = pack_artifact(&layout, &artifact, &options2).unwrap();
        assert!(outcome.resumed, "second run must resume the journal");
        verify_pack(&layout, &artifact.artifact_id).unwrap();
    }

    #[test]
    fn decoding_kernel_mapping_follows_kernel_program() {
        assert_eq!(decoding_kernel_for(2).as_deref(), Some("K01")); // Q4_0
        assert_eq!(decoding_kernel_for(35).as_deref(), Some("K04")); // TQ2_0
        assert_eq!(decoding_kernel_for(1).as_deref(), Some("K06")); // F16
        assert_eq!(decoding_kernel_for(24), None); // I8: pass-through
    }
}
