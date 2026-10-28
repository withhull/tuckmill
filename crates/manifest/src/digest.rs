//! Canonical chunked content digest shared by the store (imports, resumable
//! journals) and executors (payload identity verification at load).
//!
//! Canonical form: SHA-256 over `tm-chunks-v1`, the chunk profile line, then
//! one `len:sha256` line per chunk, in order. The chunk profile is part of
//! the identity, so a digest is reproducible from its journal alone.

use std::io::Read;

use crate::ids::{hex, sha256_hex};

pub const DEFAULT_CHUNK_BYTES: u64 = 256 * 1024 * 1024;

/// Read buffer per hashing stream (bounded, independent of chunk size).
const READ_BUFFER_BYTES: u64 = 8 * 1024 * 1024;

/// Most chunks hashed at once by [`digest_file`] (one read buffer each).
const MAX_HASH_THREADS: usize = 4;

/// SHA-256 of one chunk. On macOS this is CommonCrypto, which uses the
/// CPU's SHA-256 instructions (about 5x the portable `sha2` software path
/// on Apple Silicon); elsewhere it is `sha2`. Both compute standard
/// SHA-256, so digests are identical across platforms.
pub struct ChunkHasher {
    #[cfg(target_os = "macos")]
    ctx: Box<commoncrypto::Ctx>,
    #[cfg(not(target_os = "macos"))]
    inner: sha2::Sha256,
}

#[cfg(target_os = "macos")]
mod commoncrypto {
    /// `CC_SHA256_CTX`: count[2], hash[8], wbuf[16] as `CC_LONG` (u32).
    #[repr(C)]
    pub struct Ctx {
        pub words: [u32; 26],
    }
    extern "C" {
        // libSystem (CommonCrypto); each returns 1.
        pub fn CC_SHA256_Init(c: *mut Ctx) -> i32;
        pub fn CC_SHA256_Update(c: *mut Ctx, data: *const u8, len: u32) -> i32;
        pub fn CC_SHA256_Final(md: *mut u8, c: *mut Ctx) -> i32;
    }
}

impl Default for ChunkHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkHasher {
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        {
            let mut ctx = Box::new(commoncrypto::Ctx { words: [0; 26] });
            // SAFETY: `ctx` is a live, correctly sized CC_SHA256_CTX.
            unsafe { commoncrypto::CC_SHA256_Init(&mut *ctx) };
            Self { ctx }
        }
        #[cfg(not(target_os = "macos"))]
        {
            use sha2::Digest;
            Self {
                inner: sha2::Sha256::new(),
            }
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        #[cfg(target_os = "macos")]
        for part in data.chunks(u32::MAX as usize) {
            // SAFETY: `part` is a valid slice of `part.len()` <= u32::MAX
            // bytes and `ctx` was initialized by CC_SHA256_Init.
            unsafe {
                commoncrypto::CC_SHA256_Update(&mut *self.ctx, part.as_ptr(), part.len() as u32)
            };
        }
        #[cfg(not(target_os = "macos"))]
        {
            use sha2::Digest;
            self.inner.update(data);
        }
    }

    /// Lowercase hex digest; the hasher is consumed.
    pub fn finalize_hex(mut self) -> String {
        let mut md = [0u8; 32];
        #[cfg(target_os = "macos")]
        // SAFETY: `md` holds the 32-byte digest; `ctx` is initialized.
        unsafe {
            commoncrypto::CC_SHA256_Final(md.as_mut_ptr(), &mut *self.ctx)
        };
        #[cfg(not(target_os = "macos"))]
        {
            use sha2::Digest;
            md.copy_from_slice(&std::mem::take(&mut self.inner).finalize());
        }
        hex(&md)
    }
}

/// Digest a whole reader in one streaming pass; returns (hex digest, bytes).
pub fn digest_reader<R: Read>(reader: &mut R, chunk_bytes: u64) -> std::io::Result<(String, u64)> {
    let chunk_bytes = chunk_bytes.max(1);
    let mut chunks: Vec<(u64, String)> = Vec::new();
    let mut total: u64 = 0;
    let mut buf = vec![0u8; chunk_bytes.min(READ_BUFFER_BYTES) as usize];
    let mut chunk_hasher = ChunkHasher::new();
    let mut chunk_len: u64 = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let mut offset = 0usize;
        while offset < n {
            let space = (chunk_bytes - chunk_len) as usize;
            let take = space.min(n - offset);
            chunk_hasher.update(&buf[offset..offset + take]);
            chunk_len += take as u64;
            total += take as u64;
            offset += take;
            if chunk_len == chunk_bytes {
                let done = std::mem::take(&mut chunk_hasher);
                chunks.push((chunk_len, done.finalize_hex()));
                chunk_len = 0;
            }
        }
    }
    if chunk_len > 0 {
        chunks.push((chunk_len, chunk_hasher.finalize_hex()));
    }
    Ok((final_digest(chunk_bytes, &chunks), total))
}

/// [`digest_reader`] over an open file of `len` bytes, hashing its
/// independent chunks in parallel with positioned reads (at most
/// [`MAX_HASH_THREADS`] at once, one bounded read buffer each). The digest
/// is exactly `digest_reader`'s. A file shorter than `len` is an error.
#[cfg(unix)]
pub fn digest_file(
    file: &std::fs::File,
    len: u64,
    chunk_bytes: u64,
) -> std::io::Result<(String, u64)> {
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let chunk_bytes = chunk_bytes.max(1);
    let count = usize::try_from(len.div_ceil(chunk_bytes))
        .map_err(|_| std::io::Error::other("file has more chunks than this platform can index"))?;
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(MAX_HASH_THREADS)
        .min(count.max(1));
    let next = AtomicUsize::new(0);
    let hash_chunk = |index: usize, buf: &mut [u8]| -> std::io::Result<(u64, String)> {
        let start = index as u64 * chunk_bytes;
        let end = (start + chunk_bytes).min(len);
        let mut hasher = ChunkHasher::new();
        let mut at = start;
        while at < end {
            let n = (end - at).min(buf.len() as u64) as usize;
            file.read_exact_at(&mut buf[..n], at)?;
            hasher.update(&buf[..n]);
            at += n as u64;
        }
        Ok((end - start, hasher.finalize_hex()))
    };
    // Per worker: the (chunk index, (length, sha256)) rows it hashed.
    type Rows = Vec<(usize, (u64, String))>;
    let results: Vec<std::io::Result<Rows>> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut buf = vec![0u8; chunk_bytes.clamp(1, READ_BUFFER_BYTES) as usize];
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= count {
                            return Ok(done);
                        }
                        done.push((index, hash_chunk(index, &mut buf)?));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| {
                w.join()
                    .unwrap_or_else(|_| Err(std::io::Error::other("digest worker panicked")))
            })
            .collect()
    });
    let mut chunks: Vec<Option<(u64, String)>> = vec![None; count];
    for worker in results {
        for (index, row) in worker? {
            chunks[index] = Some(row);
        }
    }
    let chunks = chunks
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| std::io::Error::other("a digest chunk was not hashed"))?;
    Ok((final_digest(chunk_bytes, &chunks), len))
}

/// Canonical digest from a chunk table (used by resumable jobs).
pub fn final_digest(chunk_bytes: u64, chunks: &[(u64, String)]) -> String {
    let mut line = String::with_capacity(64 + chunks.len() * 90);
    line.push_str("tm-chunks-v1\n");
    line.push_str(&format!("chunk:{chunk_bytes}\n"));
    for (len, sha) in chunks {
        line.push_str(&format!("{len}:{sha}\n"));
    }
    sha256_hex(line.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_stable_and_sensitive() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        let (a, n) = digest_reader(&mut std::io::Cursor::new(&data), 4096).unwrap();
        assert_eq!(n, 10_000);
        let (b, _) = digest_reader(&mut std::io::Cursor::new(&data), 4096).unwrap();
        assert_eq!(a, b);
        let mut changed = data.clone();
        changed[5000] ^= 1;
        let (c, _) = digest_reader(&mut std::io::Cursor::new(&changed), 4096).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn chunk_profile_participates_in_identity() {
        let data = vec![7u8; 9000];
        let (a, _) = digest_reader(&mut std::io::Cursor::new(&data), 4096).unwrap();
        let (b, _) = digest_reader(&mut std::io::Cursor::new(&data), 1000).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn journal_table_matches_streaming_result() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 97) as u8).collect();
        let (streamed, _) = digest_reader(&mut std::io::Cursor::new(&data), 1024).unwrap();
        // Rebuild the table manually the way the store journal does.
        let mut rows = Vec::new();
        for chunk in data.chunks(1024) {
            rows.push((chunk.len() as u64, sha256_hex(chunk)));
        }
        assert_eq!(streamed, final_digest(1024, &rows));
    }

    /// The hardware-backed hasher is standard SHA-256 (FIPS 180-2 vectors
    /// and the portable implementation), and the parallel file digest is
    /// exactly the streaming one.
    #[test]
    fn chunk_hasher_is_sha256_and_file_digest_matches_streaming() {
        let mut h = ChunkHasher::new();
        h.update(b"abc");
        assert_eq!(
            h.finalize_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            ChunkHasher::new().finalize_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let data: Vec<u8> = (0..300_000u32).map(|i| (i * 31 % 251) as u8).collect();
        let mut split = ChunkHasher::new();
        for part in data.chunks(777) {
            split.update(part);
        }
        assert_eq!(split.finalize_hex(), sha256_hex(&data));

        let dir = tempfile::tempdir().unwrap();
        for (len, chunk) in [
            (0usize, 1024u64),
            (1, 1024),
            (300_000, 4096),
            (300_000, 1 << 20),
            (8192, 4096),
        ] {
            let path = dir.path().join(format!("f{len}-{chunk}"));
            std::fs::write(&path, &data[..len]).unwrap();
            let file = std::fs::File::open(&path).unwrap();
            let streamed = digest_reader(&mut std::io::Cursor::new(&data[..len]), chunk).unwrap();
            assert_eq!(
                digest_file(&file, len as u64, chunk).unwrap(),
                streamed,
                "{len}/{chunk}"
            );
        }
        // A file shorter than the declared length is refused.
        let path = dir.path().join("short");
        std::fs::write(&path, &data[..1000]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        assert!(digest_file(&file, 5000, 1024).is_err());
    }
}
