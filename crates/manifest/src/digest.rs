//! Canonical chunked content digest shared by the store (imports, resumable
//! journals) and executors (payload identity verification at load).
//!
//! Canonical form: SHA-256 over `tm-chunks-v1`, the chunk profile line, then
//! one `len:sha256` line per chunk, in order. The chunk profile is part of
//! the identity, so a digest is reproducible from its journal alone.

use std::io::Read;

use crate::ids::{hex, sha256_hex};

pub const DEFAULT_CHUNK_BYTES: u64 = 256 * 1024 * 1024;

/// Digest a whole reader in one streaming pass; returns (hex digest, bytes).
pub fn digest_reader<R: Read>(reader: &mut R, chunk_bytes: u64) -> std::io::Result<(String, u64)> {
    let chunk_bytes = chunk_bytes.max(1);
    use sha2::{Digest, Sha256};
    let mut chunks: Vec<(u64, String)> = Vec::new();
    let mut total: u64 = 0;
    let mut buf = vec![0u8; chunk_bytes.min(8 * 1024 * 1024) as usize];
    let mut chunk_hasher = Sha256::new();
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
                chunks.push((chunk_len, hex(&chunk_hasher.finalize_reset())));
                chunk_len = 0;
            }
        }
    }
    if chunk_len > 0 {
        chunks.push((chunk_len, hex(&chunk_hasher.finalize_reset())));
    }
    Ok((final_digest(chunk_bytes, &chunks), total))
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
}
