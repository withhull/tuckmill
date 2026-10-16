//! Safetensors header inspection (metadata-only).
//!
//! The container does not implement a model architecture and does not
//! guarantee a streaming-compatible quantization scheme (PRD §10, S13).
//! Inspection validates the JSON header, dtype table, and byte offsets with
//! checked arithmetic, and reports an unknown architecture unless the header
//! states one. Execution requires an adapter regardless.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{ErrorCode, Result, TuckmillError};
use crate::ids::{now_unix_ms, sha256_hex};
use crate::records::TensorStats;

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub start: u64,
    pub end: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsInspection {
    pub schema_version: u32,
    pub header_bytes: u64,
    pub data_start: u64,
    pub file_len: u64,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<SafetensorInfo>,
    pub header_hash: String,
    pub inspected_at_unix_ms: u64,
}

impl SafetensorsInspection {
    /// Architecture stated in the header metadata, if any. There is no
    /// standardized field; absence is normal.
    pub fn arch_id(&self) -> Option<String> {
        self.metadata
            .get("architecture")
            .map(|s| s.to_ascii_lowercase())
    }

    pub fn tensor_stats(&self) -> TensorStats {
        let mut by_type: BTreeMap<String, u64> = BTreeMap::new();
        let mut total_bytes: u64 = 0;
        for t in &self.tensors {
            *by_type.entry(t.dtype.clone()).or_default() += t.bytes;
            total_bytes = total_bytes.saturating_add(t.bytes);
        }
        TensorStats {
            total_tensors: self.tensors.len(),
            total_bytes,
            by_class: BTreeMap::new(),
            by_type,
        }
    }
}

pub fn dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        _ => return None,
    })
}

pub fn inspect(path: &std::path::Path) -> Result<SafetensorsInspection> {
    let bytes = std::fs::read(path).map_err(|e| {
        TuckmillError::new(
            ErrorCode::InvalidArtifact,
            format!("cannot read {}: {e}", path.display()),
        )
        .with_source(e)
    })?;
    inspect_bytes(&bytes)
}

pub fn inspect_bytes(file: &[u8]) -> Result<SafetensorsInspection> {
    let file_len = file.len() as u64;
    if file_len < 8 {
        return Err(bad("file is smaller than the 8-byte length prefix"));
    }
    let header_len = u64::from_le_bytes(file[..8].try_into().expect("8 bytes"));
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return Err(bad(format!(
            "header length {header_len} is outside 1..={MAX_HEADER_BYTES}"
        )));
    }
    let data_start = 8u64
        .checked_add(header_len)
        .ok_or_else(|| bad("header length overflows"))?;
    if data_start > file_len {
        return Err(bad(format!(
            "file is {file_len} bytes but the header claims {header_len}"
        )));
    }
    let header = &file[8..data_start as usize];
    let header_json: serde_json::Value = serde_json::from_slice(header).map_err(|e| {
        crate::error::json_error(e, ErrorCode::InvalidArtifact, "safetensors header")
    })?;
    let obj = header_json
        .as_object()
        .ok_or_else(|| bad("header JSON is not an object"))?;

    let mut metadata = BTreeMap::new();
    let mut tensors: Vec<SafetensorInfo> = Vec::new();
    let data_len = file_len - data_start;

    for (name, value) in obj {
        if name == "__metadata__" {
            let meta = value
                .as_object()
                .ok_or_else(|| bad("__metadata__ must be an object of strings"))?;
            for (k, v) in meta {
                let s = v
                    .as_str()
                    .ok_or_else(|| bad(format!("__metadata__.{k} is not a string")))?;
                metadata.insert(k.clone(), s.to_string());
            }
            continue;
        }
        let t = value
            .as_object()
            .ok_or_else(|| bad(format!("tensor {name:?} entry is not an object")))?;
        let dtype = t
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| bad(format!("tensor {name:?} has no dtype")))?
            .to_string();
        let elem = dtype_size(&dtype)
            .ok_or_else(|| bad(format!("tensor {name:?} has unknown dtype {dtype:?}")))?;
        let shape: Vec<u64> = t
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| bad(format!("tensor {name:?} has no shape")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| bad(format!("tensor {name:?} shape contains a non-integer")))
            })
            .collect::<Result<_>>()?;
        let mut elems: u64 = 1;
        for d in &shape {
            elems = elems
                .checked_mul(*d)
                .ok_or_else(|| bad(format!("tensor {name:?} element count overflows")))?;
        }
        let expect_bytes = elems
            .checked_mul(elem)
            .ok_or_else(|| bad(format!("tensor {name:?} byte size overflows")))?;
        let offsets = t
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| bad(format!("tensor {name:?} has no data_offsets")))?;
        if offsets.len() != 2 {
            return Err(bad(format!(
                "tensor {name:?} data_offsets must have two entries"
            )));
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| bad(format!("tensor {name:?} start is not an integer")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| bad(format!("tensor {name:?} end is not an integer")))?;
        if end < start {
            return Err(bad(format!(
                "tensor {name:?} end {end} precedes start {start}"
            )));
        }
        let bytes = end - start;
        if bytes != expect_bytes {
            return Err(bad(format!(
                "tensor {name:?} claims {bytes} bytes but {dtype} x {shape:?} requires {expect_bytes}"
            )));
        }
        if end > data_len {
            return Err(bad(format!(
                "tensor {name:?} extent ends at {end} beyond data length {data_len}"
            )));
        }
        tensors.push(SafetensorInfo {
            name: name.clone(),
            dtype,
            shape,
            start,
            end,
            bytes,
        });
    }

    // The spec requires byte-order sorting, contiguity, and no holes.
    tensors.sort_by_key(|t| t.start);
    let mut cursor = 0u64;
    for t in &tensors {
        if t.start != cursor {
            return Err(bad(format!(
                "tensor order is invalid: {:?} starts at {} but the contiguous position is {cursor}",
                t.name, t.start
            )));
        }
        cursor = t.end;
    }

    Ok(SafetensorsInspection {
        schema_version: 1,
        header_bytes: header_len,
        data_start,
        file_len,
        metadata,
        tensors,
        header_hash: sha256_hex(header),
        inspected_at_unix_ms: now_unix_ms(),
    })
}

fn bad(msg: impl Into<String>) -> TuckmillError {
    TuckmillError::new(ErrorCode::InvalidArtifact, msg)
}

/// Build a tiny, valid safetensors file for tests.
pub fn synthetic_safetensors_bytes() -> Vec<u8> {
    let payload: Vec<u8> = (0..64u8).collect();
    let header = serde_json::json!({
        "__metadata__": { "format": "tuckmill-test", "architecture": "qwen3" },
        "model.embed_tokens.weight": { "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32] },
        "model.norm.weight": { "dtype": "F32", "shape": [4], "data_offsets": [32, 48] },
        "model.layers.0.self_attn.q_proj.weight": { "dtype": "F16", "shape": [4, 2], "data_offsets": [48, 64] }
    });
    let header_bytes = serde_json::to_vec(&header).expect("json");
    // JSON object key order from serde_json is sorted (BTreeMap); offsets above
    // match that payload layout regardless of key order.
    let mut out = Vec::new();
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&payload);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspects_synthetic_file() {
        let bytes = synthetic_safetensors_bytes();
        let insp = inspect_bytes(&bytes).unwrap();
        assert_eq!(insp.tensors.len(), 3);
        assert_eq!(insp.metadata["format"], "tuckmill-test");
        assert_eq!(insp.arch_id().as_deref(), Some("qwen3"));
        // Contiguous: 0..32, 32..48, 48..64
        let starts: Vec<u64> = insp.tensors.iter().map(|t| t.start).collect();
        assert_eq!(starts, vec![0, 32, 48]);
        let stats = insp.tensor_stats();
        assert_eq!(stats.total_bytes, 64);
    }

    #[test]
    fn rejects_size_mismatch() {
        let header = serde_json::json!({
            "a": { "dtype": "F32", "shape": [4], "data_offsets": [0, 15] }
        });
        let hb = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        out.extend_from_slice(&hb);
        out.extend_from_slice(&[0u8; 16]);
        let err = inspect_bytes(&out).unwrap_err();
        assert!(err.message().contains("requires 16"));
    }

    #[test]
    fn rejects_gap_in_byte_order() {
        let header = serde_json::json!({
            "a": { "dtype": "F16", "shape": [4], "data_offsets": [0, 8] },
            "b": { "dtype": "F16", "shape": [4], "data_offsets": [12, 20] }
        });
        let hb = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        out.extend_from_slice(&hb);
        out.extend_from_slice(&[0u8; 24]);
        let err = inspect_bytes(&out).unwrap_err();
        assert!(err.message().contains("contiguous"));
    }

    #[test]
    fn rejects_unknown_dtype() {
        let header = serde_json::json!({
            "a": { "dtype": "FLOAT7", "shape": [4], "data_offsets": [0, 4] }
        });
        let hb = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        out.extend_from_slice(&hb);
        out.extend_from_slice(&[0u8; 8]);
        let err = inspect_bytes(&out).unwrap_err();
        assert!(err.message().contains("unknown dtype"));
    }

    #[test]
    fn rejects_truncated() {
        let bytes = synthetic_safetensors_bytes();
        let err = inspect_bytes(&bytes[..bytes.len() - 4]).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArtifact);
    }
}
