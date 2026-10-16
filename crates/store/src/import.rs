//! Local-file import: validate, copy with a bounded memory ceiling, hash,
//! and install atomically.
//!
//! Rules (PRD §10, §16):
//! - metadata is validated before the artifact is published;
//! - interrupted work resumes only when the source identity still matches;
//! - interrupted output never appears as a valid artifact;
//! - the source file is never deleted automatically.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use tuckmill_manifest::adapter::AdapterRegistry;
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ids::{now_unix_ms, Id};
use tuckmill_manifest::records::{ArtifactSource, ModelArtifact, ModelFormat, ModelGeometry};
use tuckmill_manifest::units::Bytes;
use tuckmill_manifest::SCHEMA_VERSION;

use crate::aliases::{self, AliasRegistry};
use crate::digester::{prefix_hash, ChunkedDigester, DigestJournal};
use crate::layout::StoreLayout;
use crate::lock::FileLock;

/// Copy buffer; the whole import path stays far below the 512MiB controlled
/// memory ceiling.
const COPY_BUFFER: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct ImportOptions {
    pub digest_chunk_bytes: u64,
    pub alias: Option<String>,
    /// Treat the source as moveable? Never: sources are always preserved.
    pub license: Option<String>,
    pub revision: Option<String>,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            digest_chunk_bytes: crate::digester::DEFAULT_CHUNK_BYTES,
            alias: None,
            license: None,
            revision: None,
        }
    }
}

#[derive(Debug)]
pub struct ImportOutcome {
    pub artifact: ModelArtifact,
    pub installed_path: PathBuf,
    pub already_present: bool,
    pub resumed: bool,
    pub warnings: Vec<String>,
}

struct SourceIdentity {
    len: u64,
    prefix_sha256: String,
}

fn source_identity(path: &Path) -> Result<SourceIdentity> {
    let meta = std::fs::metadata(path).map_err(TuckmillError::from)?;
    Ok(SourceIdentity {
        len: meta.len(),
        prefix_sha256: prefix_hash(path, 1024 * 1024)?,
    })
}

/// Import a local file into the store.
pub fn import_local(
    layout: &StoreLayout,
    adapters: &AdapterRegistry,
    source: &Path,
    options: &ImportOptions,
) -> Result<ImportOutcome> {
    layout.ensure()?;
    let meta = std::fs::metadata(source).map_err(|e| {
        TuckmillError::new(
            ErrorCode::InvalidRequest,
            format!("cannot open source {}: {e}", source.display()),
        )
        .with_source(e)
    })?;
    if !meta.is_file() {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            format!("source {} is not a regular file", source.display()),
        ));
    }
    if meta.len() == 0 {
        return Err(TuckmillError::new(
            ErrorCode::InvalidArtifact,
            "source file is empty",
        ));
    }

    // Lock per source path so concurrent imports of the same file serialize.
    let lock_name = tuckmill_manifest::ids::sha256_hex(source.to_string_lossy().as_bytes());
    let lock_path = layout
        .tmp_dir()
        .join(format!("import-{}.lock", &lock_name[..16]));
    let _lock = FileLock::exclusive(&lock_path)?;

    // Disk preflight: destination + journal, with a modest margin for the
    // manifest and temporary rename metadata.
    layout.ensure_space_for(
        Bytes(meta.len().saturating_add(16 * 1024 * 1024)),
        &layout.artifacts_dir(),
    )?;

    let identity = source_identity(source)?;
    let partial = layout
        .tmp_dir()
        .join(format!("import-{}.partial", &lock_name[..16]));
    let journal_path = layout
        .tmp_dir()
        .join(format!("import-{}.journal.json", &lock_name[..16]));

    // Resume decision: partial exists, journal matches, source unchanged.
    let mut resumed = false;
    let mut resume_from = 0u64;
    let mut digester = ChunkedDigester::new(options.digest_chunk_bytes, Some(journal_path.clone()));
    if partial.exists() {
        match DigestJournal::load(&journal_path) {
            Ok(journal) if journal.chunk_bytes == options.digest_chunk_bytes => {
                let matches = journal
                    .prefix_sha256
                    .as_deref()
                    .map(|p| p == identity.prefix_sha256)
                    .unwrap_or(false);
                if matches && journal.recorded_bytes() <= identity.len {
                    let partial_len = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
                    if partial_len >= journal.recorded_bytes() {
                        // Truncate any mid-chunk tail and resume at the
                        // journal boundary.
                        let f = std::fs::OpenOptions::new()
                            .write(true)
                            .open(&partial)
                            .map_err(TuckmillError::from)?;
                        f.set_len(journal.recorded_bytes())
                            .map_err(TuckmillError::from)?;
                        digester.journal = journal;
                        resume_from = digester.journal.recorded_bytes();
                        resumed = true;
                    }
                }
            }
            _ => {}
        }
        if !resumed {
            // Source changed or journal unusable: restart from scratch.
            let _ = std::fs::remove_file(&partial);
            let _ = std::fs::remove_file(&journal_path);
            digester = ChunkedDigester::new(options.digest_chunk_bytes, Some(journal_path.clone()));
        }
    }
    if !resumed {
        digester.journal.prefix_sha256 = Some(identity.prefix_sha256.clone());
        digester.journal.save(&journal_path)?;
    }

    // Copy + hash.
    let mut src = std::fs::File::open(source).map_err(TuckmillError::from)?;
    let mut dst = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&partial)
        .map_err(TuckmillError::from)?;
    if resume_from > 0 {
        src.seek(SeekFrom::Start(resume_from))
            .map_err(TuckmillError::from)?;
        dst.seek(SeekFrom::Start(resume_from))
            .map_err(TuckmillError::from)?;
    }
    let content_hash = copy_and_hash(&mut src, &mut dst, &mut digester)?;
    dst.sync_all().map_err(TuckmillError::from)?;
    drop(dst);

    // Duplicate detection: identical content already installed.
    let artifact_id = content_hash.clone();
    if layout.artifact_manifest_path(&artifact_id).exists() {
        let artifact = aliases::load_manifest(layout, &artifact_id)?;
        if let Some(alias) = &options.alias {
            register_alias(layout, alias, &artifact_id)?;
        }
        let _ = std::fs::remove_file(&partial);
        let _ = std::fs::remove_file(&journal_path);
        return Ok(ImportOutcome {
            artifact,
            installed_path: layout.artifact_payload_path(&artifact_id),
            already_present: true,
            resumed,
            warnings: Vec::new(),
        });
    }

    // Validate the completed file (header, index, offsets) before publish.
    let mut warnings = Vec::new();
    let inspection = validate_payload(&partial)?;
    let mut artifact = ModelArtifact {
        schema_version: SCHEMA_VERSION,
        artifact_id: artifact_id.clone(),
        created_unix_ms: now_unix_ms(),
        format: inspection.format,
        source: ArtifactSource {
            uri: source.display().to_string(),
            revision: options.revision.clone(),
            license: options.license.clone(),
        },
        file_name: source
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into()),
        file_bytes: identity.len,
        content_hash: content_hash.clone(),
        tensor_index_hash: inspection.tensor_index_hash,
        metadata_hash: inspection.metadata_hash,
        tokenizer_hash: inspection.tokenizer_hash,
        arch_id: inspection.arch_id,
        model_name: inspection.model_name,
        advertised_context_length: inspection.advertised_context_length,
        geometry: None,
        tensor_stats: inspection.tensor_stats,
        quant_label: inspection.quant_label,
        packed: None,
    };

    // Best-effort adapter resolution: record geometry when the artifact
    // satisfies an adapter; warn otherwise (planning will reject honestly).
    if let Some(arch) = artifact.arch_id.clone() {
        match adapters.find(&arch) {
            Some(adapter) => match &inspection.gguf {
                Some(insp) => match adapter.validate_metadata(insp) {
                    Ok(()) => {
                        if let Ok(kv) = adapter.kv_geometry(insp) {
                            let experts = adapter.expert_geometry(insp).ok().flatten();
                            artifact.geometry = Some(ModelGeometry {
                                layers: kv.layers,
                                head_count: kv.head_count,
                                kv_head_count: kv.kv_head_count,
                                key_length: kv.key_length,
                                value_length: kv.value_length,
                                expert_count: experts.map(|e| e.count),
                                expert_used_per_token: experts.map(|e| e.used_per_token),
                            });
                        }
                    }
                    Err(e) => warnings.push(format!(
                        "adapter {} did not accept the artifact metadata: {}",
                        adapter.id,
                        e.message()
                    )),
                },
                None => warnings.push(
                    "adapter geometry resolution requires a GGUF inspection; none was available"
                        .into(),
                ),
            },
            None => warnings.push(format!(
                "no adapter is registered for architecture {arch:?}; planning will reject this artifact until an adapter exists"
            )),
        }
    } else {
        warnings.push("artifact declares no architecture; planning will require an adapter".into());
    }

    // Atomic publish: directory first (payload), then manifest.
    let dir = layout.artifact_dir(&artifact_id);
    std::fs::create_dir_all(&dir).map_err(TuckmillError::from)?;
    let payload_path = layout.artifact_payload_path(&artifact_id);
    // rename within the same filesystem is atomic; both live under the store
    // root, and ensure() created tmp/ and artifacts/ under it.
    std::fs::rename(&partial, &payload_path).map_err(TuckmillError::from)?;
    crate::layout::fsync_dir(&dir);
    let _ = std::fs::remove_file(&journal_path);
    aliases::save_manifest(layout, &artifact)?;

    if let Some(alias) = &options.alias {
        register_alias(layout, alias, &artifact_id)?;
    }

    Ok(ImportOutcome {
        artifact,
        installed_path: payload_path,
        already_present: false,
        resumed,
        warnings,
    })
}

fn register_alias(layout: &StoreLayout, alias: &str, id: &Id) -> Result<()> {
    let mut registry = AliasRegistry::load(layout)?;
    aliases::ensure_alias_free(&registry, alias, id)?;
    registry.set(alias, id.clone())?;
    registry.save(layout)
}

fn copy_and_hash(
    src: &mut std::fs::File,
    dst: &mut std::fs::File,
    digester: &mut ChunkedDigester,
) -> Result<Id> {
    let chunk = digester.journal.chunk_bytes.max(1);
    let recorded = digester.journal.recorded_bytes();
    // `recorded` is always a whole multiple of the chunk size: the journal
    // only records complete chunks while a job is interrupted.
    debug_assert_eq!(recorded % chunk, 0);
    let mut acc = ChunkAccumulator::new(chunk);
    let mut buf = vec![0u8; COPY_BUFFER];
    let mut copied: u64 = 0;
    loop {
        let n = src.read(&mut buf).map_err(TuckmillError::from)?;
        if n == 0 {
            break;
        }
        dst.write_all(&buf[..n]).map_err(TuckmillError::from)?;
        acc.update(
            &buf[..n],
            &mut digester.journal,
            digester.journal_path.as_deref(),
        )?;
        copied += n as u64;
    }
    dst.sync_all().map_err(TuckmillError::from)?;
    acc.finish(&mut digester.journal, digester.journal_path.as_deref())?;
    digester.journal.total_bytes = recorded + copied;
    if let Some(path) = &digester.journal_path {
        digester.journal.save(path)?;
    }
    Ok(digester.journal.final_digest())
}

/// Incremental SHA-256 accumulator that checkpoints whole chunks into the
/// journal as they complete.
struct ChunkAccumulator {
    hasher: sha2::Sha256,
    len: u64,
    chunk: u64,
}

impl ChunkAccumulator {
    fn new(chunk: u64) -> Self {
        use sha2::Digest;
        Self {
            hasher: sha2::Sha256::new(),
            len: 0,
            chunk,
        }
    }

    fn update(
        &mut self,
        data: &[u8],
        journal: &mut DigestJournal,
        journal_path: Option<&Path>,
    ) -> Result<()> {
        use sha2::Digest;
        let mut offset = 0usize;
        while offset < data.len() {
            let space = (self.chunk - self.len) as usize;
            let take = space.min(data.len() - offset);
            self.hasher.update(&data[offset..offset + take]);
            self.len += take as u64;
            offset += take;
            if self.len == self.chunk {
                let digest = self.hasher.finalize_reset();
                journal.chunks.push(crate::digester::ChunkRecord {
                    len: self.len,
                    sha256: tuckmill_manifest::ids::hex(&digest),
                });
                if let Some(path) = journal_path {
                    journal.save(path)?;
                }
                self.len = 0;
            }
        }
        Ok(())
    }

    fn finish(&mut self, journal: &mut DigestJournal, journal_path: Option<&Path>) -> Result<()> {
        use sha2::Digest;
        if self.len > 0 {
            let digest = self.hasher.finalize_reset();
            journal.chunks.push(crate::digester::ChunkRecord {
                len: self.len,
                sha256: tuckmill_manifest::ids::hex(&digest),
            });
            self.len = 0;
            if let Some(path) = journal_path {
                journal.save(path)?;
            }
        }
        Ok(())
    }
}

/// Validate a completed import payload.
pub struct InspectionSummary {
    pub format: ModelFormat,
    pub tensor_index_hash: Id,
    pub metadata_hash: Id,
    pub tokenizer_hash: Option<Id>,
    pub arch_id: Option<String>,
    pub model_name: Option<String>,
    pub advertised_context_length: Option<u64>,
    pub tensor_stats: tuckmill_manifest::records::TensorStats,
    pub quant_label: Option<String>,
    pub gguf: Option<tuckmill_manifest::gguf::GgufInspection>,
}

fn validate_payload(path: &Path) -> Result<InspectionSummary> {
    let mut file = std::fs::File::open(path).map_err(TuckmillError::from)?;
    let len = file.metadata().map_err(TuckmillError::from)?.len();
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).map_err(|_| {
        TuckmillError::new(
            ErrorCode::InvalidArtifact,
            "payload is smaller than a header",
        )
    })?;
    file.seek(SeekFrom::Start(0)).map_err(TuckmillError::from)?;
    if &magic == tuckmill_manifest::gguf::GGUF_MAGIC {
        let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
        let insp = tuckmill_manifest::gguf::inspect_reader(
            &mut reader,
            len,
            &tuckmill_manifest::gguf::Limits::default(),
            None,
        )?;
        Ok(InspectionSummary {
            format: ModelFormat::Gguf,
            tensor_index_hash: insp.tensor_index_hash()?,
            metadata_hash: insp.metadata_hash()?,
            tokenizer_hash: insp.tokenizer_hash()?,
            arch_id: insp.arch_id(),
            model_name: insp.model_name().map(|s| s.to_string()),
            advertised_context_length: insp.advertised_context_length(),
            tensor_stats: insp.tensor_stats(),
            quant_label: dominant_quant_label(&insp.tensor_stats()),
            gguf: Some(insp),
        })
    } else {
        // Safetensors (or rejected): read fully is too heavy for huge files;
        // only the 8-byte prefix + header is needed.
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix).map_err(|_| {
            TuckmillError::new(
                ErrorCode::InvalidArtifact,
                "payload is smaller than a header",
            )
        })?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 || header_len > 64 * 1024 * 1024 || 8 + header_len > len {
            return Err(TuckmillError::new(
                ErrorCode::InvalidArtifact,
                "file is neither GGUF nor a valid safetensors container",
            ));
        }
        file.seek(SeekFrom::Start(0)).map_err(TuckmillError::from)?;
        let mut head = vec![0u8; (8 + header_len) as usize];
        file.read_exact(&mut head).map_err(TuckmillError::from)?;
        // inspect_bytes needs the tail for bounds checks; reconstruct with a
        // synthetic full buffer only for the header portion by validating the
        // offsets against the true length directly.
        let header_json: serde_json::Value = serde_json::from_slice(&head[8..]).map_err(|e| {
            tuckmill_manifest::error::json_error(
                e,
                ErrorCode::InvalidArtifact,
                "safetensors header",
            )
        })?;
        let obj = header_json.as_object().ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::InvalidArtifact,
                "safetensors header is not an object",
            )
        })?;
        let mut data_end: u64 = 0;
        let mut tensor_count = 0usize;
        let mut by_type: std::collections::BTreeMap<String, u64> =
            std::collections::BTreeMap::new();
        for (name, value) in obj {
            if name == "__metadata__" {
                continue;
            }
            tensor_count += 1;
            let t = value.as_object().ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!("tensor {name:?} entry is not an object"),
                )
            })?;
            let dtype = t.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!("tensor {name:?} has no dtype"),
                )
            })?;
            let elem = tuckmill_manifest::safetensors::dtype_size(dtype).ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!("tensor {name:?} has unknown dtype {dtype:?}"),
                )
            })?;
            let shape = t.get("shape").and_then(|v| v.as_array()).ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!("tensor {name:?} has no shape"),
                )
            })?;
            let mut elems: u64 = 1;
            for d in shape {
                elems = elems.checked_mul(d.as_u64().unwrap_or(0)).ok_or_else(|| {
                    TuckmillError::new(
                        ErrorCode::InvalidArtifact,
                        format!("tensor {name:?} element count overflows"),
                    )
                })?;
            }
            let offsets = t
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .filter(|a| a.len() == 2)
                .ok_or_else(|| {
                    TuckmillError::new(
                        ErrorCode::InvalidArtifact,
                        format!("tensor {name:?} has no valid data_offsets"),
                    )
                })?;
            let start = offsets[0].as_u64().unwrap_or(u64::MAX);
            let end = offsets[1].as_u64().unwrap_or(u64::MAX);
            let bytes = end.checked_sub(start).ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!("tensor {name:?} offsets are invalid"),
                )
            })?;
            let expect = elems.checked_mul(elem).ok_or_else(|| {
                TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!("tensor {name:?} size overflows"),
                )
            })?;
            if bytes != expect {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!("tensor {name:?} claims {bytes} bytes but {dtype} x {shape:?} requires {expect}"),
                ));
            }
            let data_len = len - (8 + header_len);
            if end > data_len {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidArtifact,
                    format!("tensor {name:?} extends beyond the data section"),
                ));
            }
            data_end = data_end.max(end);
            *by_type.entry(dtype.to_string()).or_default() += bytes;
        }
        let stats = tuckmill_manifest::records::TensorStats {
            total_tensors: tensor_count,
            total_bytes: data_end,
            by_class: Default::default(),
            by_type,
        };
        let header_hash = tuckmill_manifest::ids::sha256_hex(&head[8..]);
        let metadata_hash = Id::new(header_hash).expect("sha256 hex");
        Ok(InspectionSummary {
            format: ModelFormat::Safetensors,
            tensor_index_hash: metadata_hash.clone(),
            metadata_hash,
            tokenizer_hash: None,
            arch_id: None,
            model_name: None,
            advertised_context_length: None,
            tensor_stats: stats,
            quant_label: None,
            gguf: None,
        })
    }
}

fn dominant_quant_label(stats: &tuckmill_manifest::records::TensorStats) -> Option<String> {
    stats
        .by_type
        .iter()
        .max_by_key(|(_, bytes)| **bytes)
        .map(|(name, _)| name.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuckmill_manifest::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    fn registry() -> AdapterRegistry {
        AdapterRegistry::builtin().unwrap()
    }

    fn write_fixture(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3())).unwrap();
        path
    }

    #[test]
    fn imports_valid_gguf_with_geometry_and_alias() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        let src = write_fixture(dir.path(), "model.gguf");
        let options = ImportOptions {
            alias: Some("local-model".into()),
            ..Default::default()
        };
        let outcome = import_local(&layout, &registry(), &src, &options).unwrap();
        assert!(!outcome.already_present);
        assert!(outcome.installed_path.exists());
        assert!(src.exists(), "source must be preserved");
        let artifact = &outcome.artifact;
        assert_eq!(artifact.arch_id.as_deref(), Some("qwen3"));
        let geometry = artifact.geometry.as_ref().expect("geometry resolved");
        assert_eq!(geometry.layers, 2);
        assert_eq!(geometry.head_count, 4);
        assert_eq!(geometry.kv_head_count, 2);
        assert_eq!(geometry.key_length, 32);
        assert!(outcome.warnings.is_empty(), "{:?}", outcome.warnings);

        let aliases = AliasRegistry::load(&layout).unwrap();
        assert_eq!(aliases.resolve("local-model"), Some(&artifact.artifact_id));
    }

    #[test]
    fn reimport_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        let src = write_fixture(dir.path(), "model.gguf");
        let first = import_local(&layout, &registry(), &src, &ImportOptions::default()).unwrap();
        let second = import_local(&layout, &registry(), &src, &ImportOptions::default()).unwrap();
        assert!(!first.already_present);
        assert!(second.already_present);
        assert_eq!(first.artifact.artifact_id, second.artifact.artifact_id);
    }

    #[test]
    fn corrupt_source_is_rejected_before_publish() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        let path = dir.path().join("bad.gguf");
        let mut bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        bytes[0] = b'X';
        std::fs::write(&path, &bytes).unwrap();
        let err = import_local(&layout, &registry(), &path, &ImportOptions::default()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArtifact);
        // Nothing may appear as a valid artifact.
        let manifests = aliases::list_manifests(&layout).unwrap();
        assert!(manifests.is_empty());
    }

    #[test]
    fn interrupted_import_resumes_at_chunk_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        layout.ensure().unwrap();
        let src = write_fixture(dir.path(), "model.gguf");
        let bytes = std::fs::read(&src).unwrap();

        // Simulate an interrupted import: partial file holding the first two
        // chunks (chunk = 4096) and a matching journal.
        let chunk = 4096u64;
        let lock_name = tuckmill_manifest::ids::sha256_hex(src.to_string_lossy().as_bytes());
        let partial = layout
            .tmp_dir()
            .join(format!("import-{}.partial", &lock_name[..16]));
        let journal_path = layout
            .tmp_dir()
            .join(format!("import-{}.journal.json", &lock_name[..16]));
        std::fs::write(&partial, &bytes[..(chunk * 2) as usize]).unwrap();
        let mut dig = ChunkedDigester::new(chunk, None);
        let mut prefix_dig = ChunkedDigester::new(chunk, Some(journal_path.clone()));
        prefix_dig.journal.prefix_sha256 = Some(prefix_hash(&src, 1024 * 1024).unwrap());
        let _ = dig.consume(&mut std::io::Cursor::new(&bytes[..(chunk * 2) as usize]));
        prefix_dig.journal.chunks = dig.journal.chunks.clone();
        prefix_dig.journal.total_bytes = dig.journal.total_bytes;
        prefix_dig.journal.save(&journal_path).unwrap();

        let options = ImportOptions {
            digest_chunk_bytes: chunk,
            ..Default::default()
        };
        let outcome = import_local(&layout, &registry(), &src, &options).unwrap();
        assert!(outcome.resumed, "must resume from the journal");
        // Content identity equals a fresh full digest with same chunk size.
        let (fresh, _) = crate::digester::digest_file(&src, chunk).unwrap();
        assert_eq!(outcome.artifact.content_hash, fresh);
    }

    #[test]
    fn changed_source_restarts_instead_of_resuming() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        layout.ensure().unwrap();
        let src = write_fixture(dir.path(), "model.gguf");

        // Journal claims a different prefix than the current source.
        let lock_name = tuckmill_manifest::ids::sha256_hex(src.to_string_lossy().as_bytes());
        let journal_path = layout
            .tmp_dir()
            .join(format!("import-{}.journal.json", &lock_name[..16]));
        let partial = layout
            .tmp_dir()
            .join(format!("import-{}.partial", &lock_name[..16]));
        std::fs::write(&partial, b"stale").unwrap();
        let mut stale = ChunkedDigester::new(4096, Some(journal_path.clone()));
        stale.journal.prefix_sha256 = Some("0".repeat(64));
        stale.journal.save(&journal_path).unwrap();

        let outcome = import_local(
            &layout,
            &registry(),
            &src,
            &ImportOptions {
                digest_chunk_bytes: 4096,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!outcome.resumed);
        assert!(outcome.installed_path.exists());
    }

    #[test]
    fn concurrent_imports_serialize_and_both_succeed() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        let src = write_fixture(dir.path(), "model.gguf");
        let layout2 = layout.clone();
        let src2 = src.clone();
        let handle = std::thread::spawn(move || {
            import_local(&layout2, &registry(), &src2, &ImportOptions::default())
        });
        let a = import_local(&layout, &registry(), &src, &ImportOptions::default()).unwrap();
        let b = handle.join().unwrap().unwrap();
        assert_eq!(a.artifact.artifact_id, b.artifact.artifact_id);
    }
}
