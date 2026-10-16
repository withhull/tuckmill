//! Content identities and small deterministic helpers.
//!
//! Records reference immutable revisions. IDs are SHA-256 hex strings so that
//! identity is verifiable from bytes, not from mutable metadata.

use std::fmt;
use std::io::Read;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{ErrorCode, Result, TuckmillError};

/// A 64-character lowercase hex SHA-256 identity.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Id(String);

impl Id {
    pub fn new(s: impl Into<String>) -> Result<Self> {
        let s = s.into();
        if s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            Ok(Id(s))
        } else {
            Err(TuckmillError::new(
                ErrorCode::Internal,
                format!(
                    "invalid id: expected 64 lowercase hex chars, got {:?}",
                    truncate(&s, 24)
                ),
            ))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn short(&self) -> &str {
        &self.0[..12]
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Id({})", self.short())
    }
}

fn truncate(s: &str, n: usize) -> &str {
    if s.len() <= n {
        s
    } else {
        &s[..n]
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex(&hasher.finalize())
}

pub fn sha256_of_reader<R: Read>(reader: &mut R) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

pub fn hex(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(TABLE[(b >> 4) as usize] as char);
        out.push(TABLE[(b & 0x0f) as usize] as char);
    }
    out
}

/// Stable hash of a serializable record. Struct definition order is the
/// canonical order; callers must not reorder fields after a record has
/// shipped without bumping its schema version.
pub fn hash_of<T: Serialize>(value: &T) -> Result<Id> {
    let bytes = serde_json::to_vec(value)
        .map_err(|e| crate::error::json_error(e, ErrorCode::Internal, "canonical serialization"))?;
    Id::new(sha256_hex(&bytes))
}

pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Format a UNIX epoch millisecond timestamp as RFC 3339 / ISO 8601 UTC.
/// Kept dependency-free via the civil-from-days algorithm.
pub fn iso8601(unix_ms: u64) -> String {
    let secs = (unix_ms / 1000) as i64;
    let millis = unix_ms % 1000;
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z",
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60,
        millis
    )
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 → (y, m, d).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Cryptographically secure random hex of `n` bytes (2n hex chars), read from
/// the OS. There is deliberately no weak fallback: if the OS source is
/// unavailable this fails loudly.
pub fn secure_random_hex(n: usize) -> Result<String> {
    let mut buf = vec![0u8; n];
    let mut f = std::fs::File::open("/dev/urandom").map_err(|e| {
        TuckmillError::new(
            ErrorCode::Internal,
            "secure randomness source unavailable (/dev/urandom)".to_string(),
        )
        .with_source(e)
    })?;
    f.read_exact(&mut buf).map_err(|e| {
        TuckmillError::new(
            ErrorCode::Internal,
            "failed reading /dev/urandom".to_string(),
        )
        .with_source(e)
    })?;
    Ok(hex(&buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_validates_shape() {
        assert!(Id::new("a".repeat(64)).is_ok());
        assert!(Id::new("A".repeat(64)).is_err(), "uppercase rejected");
        assert!(Id::new("a".repeat(63)).is_err());
        assert!(Id::new("z".repeat(64)).is_err());
    }

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn iso8601_epoch_and_known_dates() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00.000Z");
        // 2026-10-16T00:00:00Z (20742 days after epoch)
        assert_eq!(iso8601(1_792_108_800_000), "2026-10-16T00:00:00.000Z");
        assert_eq!(iso8601(1_661_990_400_000), "2022-09-01T00:00:00.000Z");
    }

    #[test]
    fn secure_random_is_hex_and_unique() {
        let a = secure_random_hex(32).unwrap();
        let b = secure_random_hex(32).unwrap();
        assert_eq!(a.len(), 64);
        assert_ne!(a, b);
        assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn hash_of_is_deterministic() {
        #[derive(serde::Serialize)]
        struct R {
            a: u32,
            b: String,
        }
        let r = R {
            a: 1,
            b: "x".into(),
        };
        assert_eq!(hash_of(&r).unwrap(), hash_of(&r).unwrap());
        let r2 = R {
            a: 1,
            b: "y".into(),
        };
        assert_ne!(hash_of(&r).unwrap(), hash_of(&r2).unwrap());
    }
}
