//! Exact byte quantities.
//!
//! The PRD (Appendix A) requires that GB and GiB are never interchanged in the
//! interface. `Bytes` stores exact integers; parsing preserves the unit
//! system the user wrote, and formatting is explicit about which system it
//! chose.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{ErrorCode, Result, TuckmillError};

pub const KIB: u64 = 1024;
pub const MIB: u64 = 1024 * KIB;
pub const GIB: u64 = 1024 * MIB;
pub const TIB: u64 = 1024 * GIB;
pub const KB: u64 = 1000;
pub const MB: u64 = 1000 * KB;
pub const GB: u64 = 1000 * MB;
pub const TB: u64 = 1000 * GB;

/// An exact byte count. Serialized as a plain number of bytes.
#[derive(
    Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Bytes(pub u64);

impl Bytes {
    pub const ZERO: Bytes = Bytes(0);

    pub fn bytes(self) -> u64 {
        self.0
    }

    pub fn from_gib(v: f64) -> Self {
        Bytes((v * GIB as f64).round() as u64)
    }

    pub fn from_mib(v: f64) -> Self {
        Bytes((v * MIB as f64).round() as u64)
    }

    /// Decimal GB (10^9). Never mix with GiB.
    pub fn from_gb(v: f64) -> Self {
        Bytes((v * GB as f64).round() as u64)
    }

    pub fn from_mb(v: f64) -> Self {
        Bytes((v * MB as f64).round() as u64)
    }

    /// Binary GiB quantity.
    pub fn gib(self) -> f64 {
        self.0 as f64 / GIB as f64
    }

    /// Decimal GB quantity.
    pub fn gb(self) -> f64 {
        self.0 as f64 / GB as f64
    }

    pub fn is_multiple_of(self, unit: u64) -> bool {
        unit != 0 && self.0.is_multiple_of(unit)
    }

    /// Parse a user budget string. Accepts `"3GiB"`, `"2gib"`, `"1.5GiB"`,
    /// `"4GB"` (decimal), `"512MiB"`, `"3000MB"`, `"1024"` (exact bytes), and
    /// `"123B"`. The suffix decides the unit system; a bare number is bytes.
    pub fn parse(input: &str) -> Result<Bytes> {
        let s = input.trim();
        if s.is_empty() {
            return Err(bad(input, "empty quantity"));
        }
        let split = s
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '_'))
            .unwrap_or(s.len());
        let (num_part, unit_part) = s.split_at(split);
        let num_part = num_part.replace('_', "");
        if num_part.is_empty() {
            return Err(bad(input, "missing numeric part"));
        }
        let value: f64 = num_part
            .parse()
            .map_err(|_| bad(input, "numeric part is not a number"))?;
        if !value.is_finite() || value < 0.0 {
            return Err(bad(input, "quantity must be a finite non-negative number"));
        }
        let unit = unit_part.trim().to_ascii_lowercase();
        let multiplier: u64 = match unit.as_str() {
            "b" | "" => 1,
            "kib" => KIB,
            "mib" => MIB,
            "gib" => GIB,
            "tib" => TIB,
            "kb" => KB,
            "mb" => MB,
            "gb" => GB,
            "tb" => TB,
            _ => {
                return Err(bad(
                    input,
                    "unknown unit; use B, KiB/MiB/GiB/TiB (binary) or KB/MB/GB/TB (decimal)",
                ))
            }
        };
        let total = value * multiplier as f64;
        if total > u64::MAX as f64 {
            return Err(bad(input, "quantity overflows 64-bit bytes"));
        }
        Ok(Bytes(total.round() as u64))
    }

    /// Binary-system display, e.g. "2.00GiB". Binary is the default because
    /// model sizes and macOS memory are reported in binary units.
    pub fn display_binary(self) -> String {
        let v = self.0 as f64;
        if v >= TIB as f64 {
            format!("{:.2}TiB", v / TIB as f64)
        } else if v >= GIB as f64 {
            format!("{:.2}GiB", v / GIB as f64)
        } else if v >= MIB as f64 {
            format!("{:.1}MiB", v / MIB as f64)
        } else if v >= KIB as f64 {
            format!("{:.1}KiB", v / KIB as f64)
        } else {
            format!("{}B", self.0)
        }
    }

    /// Decimal-system display, e.g. "2.15GB". Used only when a source figure
    /// was decimal (publisher-listed file sizes, for example).
    pub fn display_decimal(self) -> String {
        let v = self.0 as f64;
        if v >= TB as f64 {
            format!("{:.2}TB", v / TB as f64)
        } else if v >= GB as f64 {
            format!("{:.2}GB", v / GB as f64)
        } else if v >= MB as f64 {
            format!("{:.1}MB", v / MB as f64)
        } else if v >= KB as f64 {
            format!("{:.1}KB", v / KB as f64)
        } else {
            format!("{}B", self.0)
        }
    }
}

fn bad(input: &str, why: &str) -> TuckmillError {
    TuckmillError::new(
        ErrorCode::InvalidRequest,
        format!("could not parse quantity {input:?}: {why}"),
    )
}

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display_binary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_and_decimal_are_distinct() {
        assert_eq!(Bytes::parse("1GiB").unwrap().bytes(), 1_073_741_824);
        assert_eq!(Bytes::parse("1GB").unwrap().bytes(), 1_000_000_000);
        assert_ne!(Bytes::parse("1GiB").unwrap(), Bytes::parse("1GB").unwrap());
    }

    #[test]
    fn fractional_quantities_round_to_bytes() {
        assert_eq!(Bytes::parse("1.5GiB").unwrap().bytes(), 1_610_612_736);
        assert_eq!(Bytes::parse("0.5MiB").unwrap().bytes(), 524_288);
    }

    #[test]
    fn case_insensitive_suffixes() {
        assert_eq!(Bytes::parse("3gib").unwrap(), Bytes::parse("3GiB").unwrap());
        assert_eq!(Bytes::parse("512MIB").unwrap().bytes(), 512 * MIB);
    }

    #[test]
    fn bare_numbers_are_bytes() {
        assert_eq!(Bytes::parse("1024").unwrap().bytes(), 1024);
        assert_eq!(Bytes::parse("7B").unwrap().bytes(), 7);
    }

    #[test]
    fn rejects_bad_input() {
        for s in ["", "GiB", "-1GiB", "1.2.3GiB", "3XB", "infGiB", "NaN"] {
            assert!(Bytes::parse(s).is_err(), "{s} should fail");
        }
    }

    #[test]
    fn rejects_overflow() {
        assert!(Bytes::parse("20000000TiB").is_err());
    }

    #[test]
    fn display_roundtrips_semantics() {
        let b = Bytes::parse("2GiB").unwrap();
        assert_eq!(b.display_binary(), "2.00GiB");
        let d = Bytes::parse("2GB").unwrap();
        assert_eq!(d.display_decimal(), "2.00GB");
        // 2GB is not 2GiB and must never render as the same string.
        assert_ne!(b.display_binary(), d.display_decimal());
    }
}
