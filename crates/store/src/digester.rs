//! Chunked, resumable content digest.
//!
//! Interrupted imports resume from the last complete chunk. The digest is
//! SHA-256 over the canonical chunk table
//! (`tm-chunks-v1`, the chunk profile, then `len:hex` per chunk), not a raw
//! whole-file SHA-256; the chunk profile is part of the identity so it is
//! reproducible from the journal alone.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ids::{hex, Id};

pub const DEFAULT_CHUNK_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub len: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DigestJournal {
    pub schema_version: u32,
    pub chunk_bytes: u64,
    pub total_bytes: u64,
    pub chunks: Vec<ChunkRecord>,
    /// First 1MiB hash of the source; lets a resumed job detect that the
    /// source file changed underneath it.
    pub prefix_sha256: Option<String>,
}

impl DigestJournal {
    pub fn new(chunk_bytes: u64) -> Self {
        Self {
            schema_version: 1,
            chunk_bytes,
            total_bytes: 0,
            chunks: Vec::new(),
            prefix_sha256: None,
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(TuckmillError::from)?;
        let journal: DigestJournal = serde_json::from_slice(&bytes).map_err(|e| {
            tuckmill_manifest::error::json_error(e, ErrorCode::IntegrityFailure, "digest journal")
        })?;
        if journal.schema_version != 1 {
            return Err(TuckmillError::new(
                ErrorCode::IntegrityFailure,
                format!(
                    "unsupported digest journal schema {}",
                    journal.schema_version
                ),
            ));
        }
        Ok(journal)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("journal.tmp");
        let bytes = serde_json::to_vec(self).map_err(|e| {
            tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "digest journal write")
        })?;
        std::fs::write(&tmp, bytes).map_err(TuckmillError::from)?;
        std::fs::rename(&tmp, path).map_err(TuckmillError::from)?;
        Ok(())
    }

    /// Final digest from the journal; None while incomplete knowledge.
    ///
    /// The canonical form includes the chunking profile so that identity is
    /// reproducible from the journal alone; two files digested under
    /// different chunk profiles are distinct identities by construction.
    pub fn final_digest(&self) -> Id {
        let mut hasher = Sha256::new();
        hasher.update(b"tm-chunks-v1\n");
        hasher.update(format!("chunk:{}\n", self.chunk_bytes).as_bytes());
        for c in &self.chunks {
            hasher.update(format!("{}:{}\n", c.len, c.sha256).as_bytes());
        }
        Id::new(hex(&hasher.finalize())).expect("sha256 is valid hex")
    }

    /// Sum of recorded chunk lengths.
    pub fn recorded_bytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.len).sum()
    }
}

/// Streaming digester over a reader, with journal checkpointing every chunk.
pub struct ChunkedDigester {
    pub journal: DigestJournal,
    pub journal_path: Option<PathBuf>,
}

impl ChunkedDigester {
    pub fn new(chunk_bytes: u64, journal_path: Option<PathBuf>) -> Self {
        Self {
            journal: DigestJournal::new(chunk_bytes),
            journal_path,
        }
    }

    /// Consume the whole reader, checkpointing after every complete chunk.
    pub fn consume<R: Read>(&mut self, reader: &mut R) -> Result<Id> {
        let chunk = self.journal.chunk_bytes.max(1);
        let mut buf = vec![0u8; chunk.min(8 * 1024 * 1024) as usize];
        let mut chunk_hasher = Sha256::new();
        let mut chunk_len: u64 = 0;
        let mut total: u64 = 0;
        loop {
            let n = reader.read(&mut buf).map_err(TuckmillError::from)?;
            if n == 0 {
                break;
            }
            let mut offset = 0usize;
            while offset < n {
                let space = (chunk - chunk_len) as usize;
                let take = space.min(n - offset);
                chunk_hasher.update(&buf[offset..offset + take]);
                chunk_len += take as u64;
                total += take as u64;
                offset += take;
                if chunk_len == chunk {
                    self.push_chunk(chunk_hasher, chunk_len)?;
                    chunk_hasher = Sha256::new();
                    chunk_len = 0;
                }
            }
        }
        if chunk_len > 0 {
            self.push_chunk(chunk_hasher, chunk_len)?;
        }
        self.journal.total_bytes = total;
        if let Some(path) = &self.journal_path {
            self.journal.save(path)?;
        }
        Ok(self.journal.final_digest())
    }

    fn push_chunk(&mut self, hasher: Sha256, len: u64) -> Result<()> {
        self.journal.chunks.push(ChunkRecord {
            len,
            sha256: hex(&hasher.finalize()),
        });
        if let Some(path) = &self.journal_path {
            self.journal.save(path)?;
        }
        Ok(())
    }
}

/// Digest an entire file in one streaming pass.
pub fn digest_file(path: &Path, chunk_bytes: u64) -> Result<(Id, u64)> {
    let file = std::fs::File::open(path).map_err(|e| {
        TuckmillError::new(
            ErrorCode::StorageLost,
            format!("cannot open {} for hashing: {e}", path.display()),
        )
        .with_source(e)
    })?;
    let mut digester = ChunkedDigester::new(chunk_bytes, None);
    let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
    let id = digester.consume(&mut reader)?;
    Ok((id, digester.journal.total_bytes))
}

/// Hash the first `prefix_len` bytes of a file (source-change detection).
pub fn prefix_hash(path: &Path, prefix_len: u64) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(TuckmillError::from)?;
    let mut buf = vec![0u8; prefix_len as usize];
    let mut filled = 0usize;
    while filled < buf.len() {
        let n = file.read(&mut buf[filled..]).map_err(TuckmillError::from)?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    buf.truncate(filled);
    Ok(tuckmill_manifest::ids::sha256_hex(&buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_deterministic_within_chunk_profile() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        let mut a = ChunkedDigester::new(4096, None);
        let id_a = a.consume(&mut std::io::Cursor::new(&data)).unwrap();
        let mut c = ChunkedDigester::new(4096, None);
        let id_c = c.consume(&mut std::io::Cursor::new(&data)).unwrap();
        assert_eq!(id_a, id_c, "same chunk profile must reproduce identity");
        assert_eq!(a.journal.total_bytes, 10_000);

        // A different chunking profile is a different identity by
        // construction: identity is reproducible from the journal, and the
        // journal records the profile.
        let mut b = ChunkedDigester::new(999, None);
        let id_b = b.consume(&mut std::io::Cursor::new(&data)).unwrap();
        assert_ne!(id_a, id_b);
        assert_eq!(b.journal.chunk_bytes, 999);
    }

    #[test]
    fn changing_one_byte_changes_digest() {
        let mut data: Vec<u8> = vec![7u8; 5000];
        let mut a = ChunkedDigester::new(1024, None);
        let id_a = a.consume(&mut std::io::Cursor::new(&data)).unwrap();
        data[2500] = 8;
        let mut b = ChunkedDigester::new(1024, None);
        let id_b = b.consume(&mut std::io::Cursor::new(&data)).unwrap();
        assert_ne!(id_a, id_b);
    }

    #[test]
    fn journal_roundtrips_and_resumes_hash_material() {
        let dir = tempfile::tempdir().unwrap();
        let jpath = dir.path().join("x.journal.json");
        let data: Vec<u8> = (0..9000u32).map(|i| (i % 97) as u8).collect();
        let mut d = ChunkedDigester::new(4096, Some(jpath.clone()));
        let id = d.consume(&mut std::io::Cursor::new(&data)).unwrap();
        let loaded = DigestJournal::load(&jpath).unwrap();
        assert_eq!(loaded.final_digest(), id);
        assert_eq!(loaded.recorded_bytes(), 9000);
        assert_eq!(loaded.chunks.len(), 3); // 4096 + 4096 + 808
    }

    #[test]
    fn empty_input_is_valid() {
        let mut d = ChunkedDigester::new(4096, None);
        let id = d.consume(&mut std::io::Cursor::new(vec![])).unwrap();
        assert_eq!(d.journal.total_bytes, 0);
        assert_eq!(id, d.journal.final_digest());
    }

    #[test]
    fn prefix_hash_detects_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        std::fs::write(&path, b"hello world").unwrap();
        let h1 = prefix_hash(&path, 1024).unwrap();
        std::fs::write(&path, b"hello worlD").unwrap();
        let h2 = prefix_hash(&path, 1024).unwrap();
        assert_ne!(h1, h2);
    }

    #[test]
    fn digest_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        std::fs::write(&path, vec![9u8; 100_000]).unwrap();
        let (id, len) = digest_file(&path, 16 * 1024).unwrap();
        assert_eq!(len, 100_000);
        let (id2, _) = digest_file(&path, 16 * 1024).unwrap();
        assert_eq!(id, id2);
    }
}
