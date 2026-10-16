//! GGUF inspection: validated header, metadata, and tensor index.
//!
//! Inspection never reads tensor data. Every offset and size is validated
//! with checked arithmetic against the real file length before anything is
//! computed from it (FR-01). Malformed sizes and offsets are rejected up
//! front, not discovered during execution.

use std::collections::BTreeMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{ErrorCode, Result, TuckmillError};
use crate::ggml;
use crate::ids::{hash_of, now_unix_ms, sha256_hex, Id};
use crate::records::TensorStats;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Bounded-read limits for untrusted files (PRD §16).
#[derive(Debug, Clone)]
pub struct Limits {
    pub max_metadata_bytes: u64,
    pub max_string_len: u64,
    pub max_array_items: u64,
    pub max_tensors: u64,
    pub max_dims: usize,
    pub max_alignment: u64,
    pub max_array_depth: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_metadata_bytes: 512 * 1024 * 1024,
            max_string_len: 64 * 1024 * 1024,
            max_array_items: 100_000_000,
            max_tensors: 1_000_000,
            max_dims: 4,
            max_alignment: 16 * 1024 * 1024,
            max_array_depth: 2,
        }
    }
}

/// A metadata value. Kept in an externally tagged form so serialization to
/// JSON is stable and hashable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I16(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::F32(v) => Some(*v as f64),
            GgufValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            GgufValue::U8(_) => "u8",
            GgufValue::I8(_) => "i8",
            GgufValue::U16(_) => "u16",
            GgufValue::I16(_) => "i16",
            GgufValue::U32(_) => "u32",
            GgufValue::I32(_) => "i32",
            GgufValue::F32(_) => "f32",
            GgufValue::Bool(_) => "bool",
            GgufValue::String(_) => "string",
            GgufValue::Array(_) => "array",
            GgufValue::U64(_) => "u64",
            GgufValue::I64(_) => "i64",
            GgufValue::F64(_) => "f64",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufKv {
    pub key: String,
    pub value: GgufValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub type_id: u32,
    pub type_name: String,
    pub offset: u64,
    pub byte_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufInspection {
    pub schema_version: u32,
    pub version: u32,
    pub alignment: u64,
    pub header_bytes: u64,
    pub data_start: u64,
    pub file_len: u64,
    pub metadata: Vec<GgufKv>,
    pub tensors: Vec<GgufTensor>,
    /// Non-fatal layout observations (for example aliased or overlapping
    /// tensor extents). Fatal problems are errors instead.
    pub layout_warnings: Vec<String>,
    pub inspected_at_unix_ms: u64,
}

impl GgufInspection {
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata
            .iter()
            .find(|kv| kv.key == key)
            .map(|kv| &kv.value)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| v.as_u64())
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(|v| v.as_f32())
    }

    /// `general.architecture`, lowercased.
    pub fn arch_id(&self) -> Option<String> {
        self.get_str("general.architecture")
            .map(|s| s.to_ascii_lowercase())
    }

    pub fn model_name(&self) -> Option<&str> {
        self.get_str("general.name")
    }

    /// Vocabulary length: from token_embd.weight's shape when present,
    /// otherwise the embedded tokenizer array length.
    pub fn vocabulary_len(&self) -> Option<u64> {
        if let Some(t) = self.tensors.iter().find(|t| t.name == "token_embd.weight") {
            if t.dims.len() == 2 {
                return Some(t.dims[1]);
            }
        }
        self.get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len() as u64)
    }

    /// Context length advertised by the artifact, if any of the usual keys
    /// exist. Advertised does not mean admissible under a budget.
    pub fn advertised_context_length(&self) -> Option<u64> {
        let arch = self.arch_id()?;
        self.get_u64(&format!("{arch}.context_length"))
    }

    pub fn tensor_stats(&self) -> TensorStats {
        let mut by_class: BTreeMap<String, u64> = BTreeMap::new();
        let mut by_type: BTreeMap<String, u64> = BTreeMap::new();
        let mut total_bytes: u64 = 0;
        for t in &self.tensors {
            *by_class
                .entry(class_name(classify_tensor(&t.name)).to_string())
                .or_default() += t.byte_size;
            *by_type.entry(t.type_name.clone()).or_default() += t.byte_size;
            total_bytes = total_bytes.saturating_add(t.byte_size);
        }
        TensorStats {
            total_tensors: self.tensors.len(),
            total_bytes,
            by_class,
            by_type,
        }
    }

    /// Canonical hash of the tensor index (names, shapes, types, offsets).
    pub fn tensor_index_hash(&self) -> Result<Id> {
        #[derive(Serialize)]
        struct Row<'a> {
            name: &'a str,
            dims: &'a [u64],
            type_id: u32,
            offset: u64,
            byte_size: u64,
        }
        let rows: Vec<Row<'_>> = self
            .tensors
            .iter()
            .map(|t| Row {
                name: &t.name,
                dims: &t.dims,
                type_id: t.type_id,
                offset: t.offset,
                byte_size: t.byte_size,
            })
            .collect();
        hash_of(&rows)
    }

    /// Canonical hash of the metadata section.
    pub fn metadata_hash(&self) -> Result<Id> {
        hash_of(&self.metadata)
    }

    /// Hash of the tokenizer components when the artifact embeds them.
    /// Returns None when the artifact does not carry an embedded tokenizer.
    pub fn tokenizer_hash(&self) -> Result<Option<Id>> {
        let mut parts: Vec<(&str, &GgufValue)> = Vec::new();
        for key in [
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.model",
            "tokenizer.ggml.pre",
        ] {
            if let Some(v) = self.get(key) {
                parts.push((key, v));
            }
        }
        if parts.is_empty() {
            return Ok(None);
        }
        Ok(Some(hash_of(&parts)?))
    }

    /// Combined identity of metadata plus tensor index. This is the immutable
    /// artifact identity used by plans and session records.
    pub fn identity_hash(&self) -> Result<Id> {
        Id::new(sha256_hex(
            format!(
                "{}\n{}",
                self.metadata_hash()?.as_str(),
                self.tensor_index_hash()?.as_str()
            )
            .as_bytes(),
        ))
    }
}

/// Tensor role classification from GGUF naming conventions. Classification is
/// used for accounting and adapter checks, never to skip tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorClass {
    TokenEmbedding,
    OutputHead,
    Attention,
    AttnNorm,
    Ffn,
    MoeRouter,
    MoeExpert,
    MoeSharedExpert,
    Norm,
    Other,
}

pub fn class_name(c: TensorClass) -> &'static str {
    match c {
        TensorClass::TokenEmbedding => "token_embedding",
        TensorClass::OutputHead => "output_head",
        TensorClass::Attention => "attention",
        TensorClass::AttnNorm => "attn_norm",
        TensorClass::Ffn => "ffn",
        TensorClass::MoeRouter => "moe_router",
        TensorClass::MoeExpert => "moe_expert",
        TensorClass::MoeSharedExpert => "moe_shared_expert",
        TensorClass::Norm => "norm",
        TensorClass::Other => "other",
    }
}

pub fn classify_tensor(name: &str) -> TensorClass {
    let n = name;
    if n == "token_embd.weight" || n == "token_embd" {
        return TensorClass::TokenEmbedding;
    }
    if n == "output.weight" {
        return TensorClass::OutputHead;
    }
    if n.contains("attn_q_norm") || n.contains("attn_k_norm") {
        return TensorClass::AttnNorm;
    }
    if n.contains("ffn_gate_inp") {
        return TensorClass::MoeRouter;
    }
    if n.ends_with("_exps.weight") {
        if n.contains("_shexp") {
            return TensorClass::MoeSharedExpert;
        }
        return TensorClass::MoeExpert;
    }
    if n.contains("_shexp") {
        return TensorClass::MoeSharedExpert;
    }
    if n.contains("attn_q")
        || n.contains("attn_k")
        || n.contains("attn_v")
        || n.contains("attn_qkv")
        || n.contains("attn_output")
    {
        return TensorClass::Attention;
    }
    if n.contains("ffn_gate") || n.contains("ffn_up") || n.contains("ffn_down") {
        return TensorClass::Ffn;
    }
    if n.contains("norm") {
        return TensorClass::Norm;
    }
    TensorClass::Other
}

/// Parse and validate a GGUF file without reading tensor data.
pub fn inspect(path: &Path, limits: &Limits) -> Result<GgufInspection> {
    let file = std::fs::File::open(path).map_err(|e| {
        TuckmillError::new(
            ErrorCode::InvalidArtifact,
            format!("cannot open {}: {e}", path.display()),
        )
        .with_source(e)
    })?;
    let file_len = file.metadata().map_err(TuckmillError::from)?.len();
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let inspection = inspect_reader(&mut reader, file_len, limits, None)?;
    Ok(inspection)
}

/// Parse and validate GGUF from an arbitrary reader (used by import so the
/// file is validated while it is being copied).
pub fn inspect_reader<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    limits: &Limits,
    max_read: Option<u64>,
) -> Result<GgufInspection> {
    let mut r = CountingReader::new(reader, max_read);
    let mut magic = [0u8; 4];
    r.read_exact_bytes(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(TuckmillError::new(
            ErrorCode::InvalidArtifact,
            "file does not start with the GGUF magic",
        ));
    }
    let version = r.read_u32()?;
    if version != 2 && version != 3 {
        return Err(TuckmillError::new(
            ErrorCode::InvalidArtifact,
            format!("unsupported GGUF version {version} (expected 2 or 3)"),
        ));
    }
    let tensor_count = r.read_u64()?;
    let kv_count = r.read_u64()?;
    if tensor_count > limits.max_tensors {
        return Err(TuckmillError::new(
            ErrorCode::InvalidArtifact,
            format!(
                "tensor count {tensor_count} exceeds limit {}",
                limits.max_tensors
            ),
        ));
    }

    let mut metadata = Vec::new();
    let mut metadata_bytes: u64 = 0;
    for i in 0..kv_count {
        let key_len = r.read_u64()?;
        if key_len > limits.max_string_len {
            return Err(bad(format!(
                "metadata key {i} length {key_len} exceeds limit"
            )));
        }
        let key = r.read_string(key_len)?;
        let value = read_value(&mut r, limits, 0)?;
        metadata_bytes = metadata_bytes
            .checked_add(key_len)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| bad("metadata byte count overflow"))?;
        if metadata_bytes > limits.max_metadata_bytes {
            return Err(bad(format!(
                "metadata section exceeds {} bytes",
                limits.max_metadata_bytes
            )));
        }
        metadata.push(GgufKv { key, value });
    }

    // Alignment must be read before tensors so size validation below can use
    // the real data start.
    let alignment = metadata
        .iter()
        .find(|kv| kv.key == "general.alignment")
        .map(|kv| {
            kv.value
                .as_u64()
                .ok_or_else(|| bad("general.alignment must be an integer"))
        })
        .transpose()?
        .unwrap_or(32);
    if alignment == 0 || !alignment.is_power_of_two() || alignment > limits.max_alignment {
        return Err(bad(format!(
            "general.alignment {alignment} is not a power of two within 1..={}",
            limits.max_alignment
        )));
    }

    let mut tensors = Vec::with_capacity(tensor_count.min(4096) as usize);
    for i in 0..tensor_count {
        let name_len = r.read_u64()?;
        if name_len == 0 || name_len > limits.max_string_len {
            return Err(bad(format!(
                "tensor {i} name length {name_len} out of range"
            )));
        }
        let name = r.read_string(name_len)?;
        let n_dims = r.read_u32()? as usize;
        if n_dims == 0 || n_dims > limits.max_dims {
            return Err(bad(format!(
                "tensor {name:?} has {n_dims} dimensions (allowed 1..={})",
                limits.max_dims
            )));
        }
        let mut dims = Vec::with_capacity(n_dims);
        let mut elems: u64 = 1;
        for d in 0..n_dims {
            let dim = r.read_u64()?;
            if dim == 0 {
                return Err(bad(format!("tensor {name:?} dimension {d} is zero")));
            }
            elems = elems
                .checked_mul(dim)
                .ok_or_else(|| bad(format!("tensor {name:?} element count overflows")))?;
            dims.push(dim);
        }
        let type_id = r.read_u32()?;
        let offset = r.read_u64()?;
        let byte_size = ggml::tensor_bytes(type_id, elems).map_err(|e| {
            TuckmillError::new(e.code(), format!("tensor {name:?}: {}", e.message()))
        })?;
        tensors.push(GgufTensor {
            name,
            dims,
            type_id,
            type_name: ggml::type_name(type_id),
            offset,
            byte_size,
        });
    }

    let header_end = r.count;
    let data_start =
        align_up(header_end, alignment).ok_or_else(|| bad("data start offset overflow"))?;
    if file_len < data_start {
        return Err(bad(format!(
            "file is {file_len} bytes but the tensor data section starts at {data_start}"
        )));
    }
    let data_len = file_len - data_start;
    for t in &tensors {
        let end = t
            .offset
            .checked_add(t.byte_size)
            .ok_or_else(|| bad(format!("tensor {:?} extent overflows", t.name)))?;
        if end > data_len {
            return Err(bad(format!(
                "tensor {:?} extent [{}..{end}) exceeds data section of {data_len} bytes",
                t.name, t.offset
            )));
        }
    }

    // Overlap detection (warning only; identical aliasing included).
    let mut ranges: Vec<(u64, u64, &str)> = tensors
        .iter()
        .map(|t| (t.offset, t.offset + t.byte_size, t.name.as_str()))
        .collect();
    ranges.sort_unstable();
    let mut layout_warnings = Vec::new();
    for w in ranges.windows(2) {
        let (_, end_a, name_a) = w[0];
        let (start_b, _, name_b) = w[1];
        if start_b < end_a {
            layout_warnings.push(format!(
                "tensor extents overlap: {name_a:?} ends at {end_a}, {name_b:?} starts at {start_b}"
            ));
        }
    }

    if let Some(cap) = max_read {
        if cap < header_end {
            return Err(bad(format!(
                "header needs {header_end} bytes but the read budget is {cap}"
            )));
        }
    }

    Ok(GgufInspection {
        schema_version: 1,
        version,
        alignment,
        header_bytes: header_end,
        data_start,
        file_len,
        metadata,
        tensors,
        layout_warnings,
        inspected_at_unix_ms: now_unix_ms(),
    })
}

fn read_value<R: Read>(
    r: &mut CountingReader<'_, R>,
    limits: &Limits,
    depth: usize,
) -> Result<GgufValue> {
    if depth > limits.max_array_depth {
        return Err(bad("metadata arrays are nested too deeply"));
    }
    let ty = r.read_u32()?;
    Ok(match ty {
        0 => GgufValue::U8(r.read_u8()?),
        1 => GgufValue::I8(r.read_u8()? as i8),
        2 => GgufValue::U16(r.read_u16()?),
        3 => GgufValue::I16(r.read_u16()? as i16),
        4 => GgufValue::U32(r.read_u32()?),
        5 => GgufValue::I32(r.read_u32()? as i32),
        6 => GgufValue::F32(f32::from_bits(r.read_u32()?)),
        7 => GgufValue::Bool(match r.read_u8()? {
            0 => false,
            1 => true,
            other => return Err(bad(format!("bool value {other} is neither 0 nor 1"))),
        }),
        8 => {
            let len = r.read_u64()?;
            if len > limits.max_string_len {
                return Err(bad(format!("string length {len} exceeds limit")));
            }
            GgufValue::String(r.read_string(len)?)
        }
        9 => {
            let item_ty = r.read_u32()?;
            let count = r.read_u64()?;
            if count > limits.max_array_items {
                return Err(bad(format!("array count {count} exceeds limit")));
            }
            let mut items = Vec::with_capacity(count.min(1 << 20) as usize);
            // 9 = array-of-array is pathological; bounded by max_array_depth.
            for _ in 0..count {
                items.push(read_value_of_type(r, limits, depth + 1, item_ty)?);
            }
            GgufValue::Array(items)
        }
        10 => GgufValue::U64(r.read_u64()?),
        11 => GgufValue::I64(r.read_u64()? as i64),
        12 => GgufValue::F64(f64::from_bits(r.read_u64()?)),
        other => return Err(bad(format!("unknown metadata value type {other}"))),
    })
}

fn read_value_of_type<R: Read>(
    r: &mut CountingReader<'_, R>,
    limits: &Limits,
    depth: usize,
    ty: u32,
) -> Result<GgufValue> {
    // Re-dispatch on the declared element type without re-reading it.
    if ty == 9 {
        if depth > limits.max_array_depth {
            return Err(bad("metadata arrays are nested too deeply"));
        }
        let item_ty = r.read_u32()?;
        let count = r.read_u64()?;
        if count > limits.max_array_items {
            return Err(bad(format!("array count {count} exceeds limit")));
        }
        let mut items = Vec::with_capacity(count.min(1 << 20) as usize);
        for _ in 0..count {
            items.push(read_value_of_type(r, limits, depth + 1, item_ty)?);
        }
        return Ok(GgufValue::Array(items));
    }
    Ok(match ty {
        0 => GgufValue::U8(r.read_u8()?),
        1 => GgufValue::I8(r.read_u8()? as i8),
        2 => GgufValue::U16(r.read_u16()?),
        3 => GgufValue::I16(r.read_u16()? as i16),
        4 => GgufValue::U32(r.read_u32()?),
        5 => GgufValue::I32(r.read_u32()? as i32),
        6 => GgufValue::F32(f32::from_bits(r.read_u32()?)),
        7 => GgufValue::Bool(match r.read_u8()? {
            0 => false,
            1 => true,
            other => return Err(bad(format!("bool value {other} is neither 0 nor 1"))),
        }),
        8 => {
            let len = r.read_u64()?;
            if len > limits.max_string_len {
                return Err(bad(format!("string length {len} exceeds limit")));
            }
            GgufValue::String(r.read_string(len)?)
        }
        10 => GgufValue::U64(r.read_u64()?),
        11 => GgufValue::I64(r.read_u64()? as i64),
        12 => GgufValue::F64(f64::from_bits(r.read_u64()?)),
        other => return Err(bad(format!("unknown array element type {other}"))),
    })
}

fn bad(msg: impl Into<String>) -> TuckmillError {
    TuckmillError::new(ErrorCode::InvalidArtifact, msg)
}

fn align_up(value: u64, align: u64) -> Option<u64> {
    debug_assert!(align.is_power_of_two());
    let mask = align - 1;
    value.checked_add(mask).map(|v| v & !mask)
}

/// Reader wrapper tracking exact byte position and an optional read budget.
struct CountingReader<'a, R: Read> {
    inner: &'a mut R,
    count: u64,
    budget: Option<u64>,
}

impl<'a, R: Read> CountingReader<'a, R> {
    fn new(inner: &'a mut R, budget: Option<u64>) -> Self {
        Self {
            inner,
            count: 0,
            budget,
        }
    }

    fn read_exact_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        if let Some(budget) = self.budget {
            if self.count + buf.len() as u64 > budget {
                return Err(bad("read budget exceeded while reading header"));
            }
        }
        self.inner
            .read_exact(buf)
            .map_err(|e| bad(format!("unexpected end of file while reading header: {e}")))?;
        self.count += buf.len() as u64;
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.read_exact_bytes(&mut b)?;
        Ok(b[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        self.read_exact_bytes(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.read_exact_bytes(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.read_exact_bytes(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn read_string(&mut self, len: u64) -> Result<String> {
        let len_usize = usize::try_from(len)
            .map_err(|_| bad(format!("string length {len} does not fit memory")))?;
        let chunk = 1 << 20;
        let mut buf = vec![0u8; len_usize.min(chunk)];
        let mut remaining = len_usize;
        let mut out = Vec::with_capacity(len_usize.min(chunk));
        while remaining > 0 {
            let take = remaining.min(buf.len());
            self.read_exact_bytes(&mut buf[..take])?;
            out.extend_from_slice(&buf[..take]);
            remaining -= take;
        }
        String::from_utf8(out).map_err(|_| bad("string is not valid UTF-8"))
    }
}

/// Seek helper used by callers that want the exact data region.
pub fn seek_to_data<R: Seek>(reader: &mut R, inspection: &GgufInspection) -> Result<()> {
    reader
        .seek(SeekFrom::Start(inspection.data_start))
        .map_err(TuckmillError::from)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{synthetic_qwen3_bytes, SyntheticConfig};

    #[test]
    fn inspects_synthetic_qwen3() {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        let mut reader = std::io::Cursor::new(bytes.clone());
        let insp = inspect_reader(&mut reader, bytes.len() as u64, &Limits::default(), None)
            .expect("valid fixture");
        assert_eq!(insp.version, 3);
        assert_eq!(insp.alignment, 32);
        assert_eq!(insp.arch_id().as_deref(), Some("qwen3"));
        assert_eq!(insp.get_u64("qwen3.block_count"), Some(2));
        assert!(insp.tensor_stats().total_bytes > 0);
        assert!(
            insp.layout_warnings.is_empty(),
            "{:?}",
            insp.layout_warnings
        );
        assert_eq!(insp.file_len, bytes.len() as u64);
        // Data region must be accounted exactly: header + pad + tensors <= file
        let stats = &insp.tensor_stats();
        assert!(insp.data_start + stats.total_bytes <= insp.file_len);
        assert!(insp.tokenizer_hash().unwrap().is_some());
    }

    #[test]
    fn tensor_index_hash_is_stable_and_sensitive() {
        let cfg = SyntheticConfig::tiny_qwen3();
        let a = inspect_cfg(&cfg);
        let b = inspect_cfg(&cfg);
        assert_eq!(
            a.tensor_index_hash().unwrap(),
            b.tensor_index_hash().unwrap()
        );
        assert_eq!(a.identity_hash().unwrap(), b.identity_hash().unwrap());

        let mut cfg2 = SyntheticConfig::tiny_qwen3();
        cfg2.layers = 3;
        let c = inspect_cfg(&cfg2);
        assert_ne!(
            a.tensor_index_hash().unwrap(),
            c.tensor_index_hash().unwrap()
        );
    }

    fn inspect_cfg(cfg: &SyntheticConfig) -> GgufInspection {
        let bytes = synthetic_qwen3_bytes(cfg);
        let mut reader = std::io::Cursor::new(bytes.clone());
        inspect_reader(&mut reader, bytes.len() as u64, &Limits::default(), None).unwrap()
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        bytes[0] = b'X';
        let mut reader = std::io::Cursor::new(bytes.clone());
        let err =
            inspect_reader(&mut reader, bytes.len() as u64, &Limits::default(), None).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArtifact);
        assert!(err.message().contains("magic"));
    }

    #[test]
    fn rejects_truncated_file() {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        let cut = bytes.len() / 2;
        let mut reader = std::io::Cursor::new(bytes[..cut].to_vec());
        let err = inspect_reader(&mut reader, cut as u64, &Limits::default(), None).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArtifact);
    }

    #[test]
    fn rejects_out_of_bounds_tensor_extent() {
        // Shrink the declared file length so the last tensor runs past the end.
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        let mut reader = std::io::Cursor::new(bytes.clone());
        let err = inspect_reader(
            &mut reader,
            bytes.len() as u64 - 1,
            &Limits::default(),
            None,
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArtifact);
        assert!(err.message().contains("exceeds data section"));
    }

    #[test]
    fn rejects_unknown_tensor_type() {
        // Handcraft a tiny header whose single tensor uses type id 250.
        let handcrafted = handcraft_single_tensor_with_type(250);
        let len = handcrafted.len() as u64;
        let err = inspect_reader(
            &mut std::io::Cursor::new(handcrafted),
            len,
            &Limits::default(),
            None,
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::UnsupportedQuantization);
    }

    fn handcraft_single_tensor_with_type(type_id: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes()); // tensors
        out.extend_from_slice(&0u64.to_le_bytes()); // kv count
        let name = b"blk.0.attn_q.weight";
        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        out.extend_from_slice(&128u64.to_le_bytes());
        out.extend_from_slice(&128u64.to_le_bytes());
        out.extend_from_slice(&type_id.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // offset
                                                    // pad to alignment 32
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend_from_slice(&[0u8; 128]);
        out
    }

    #[test]
    fn rejects_misaligned_quantized_tensor() {
        // Q4_0 with 100 elements is not block aligned (100 % 32 != 0).
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        let name = b"blk.0.ffn_gate.weight";
        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&100u64.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes()); // Q4_0
        out.extend_from_slice(&0u64.to_le_bytes());
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend_from_slice(&[0u8; 64]);
        let len = out.len() as u64;
        let err = inspect_reader(
            &mut std::io::Cursor::new(out),
            len,
            &Limits::default(),
            None,
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArtifact);
        assert!(err.message().contains("block size"));
    }

    #[test]
    fn classifies_tensor_names() {
        assert_eq!(
            classify_tensor("token_embd.weight"),
            TensorClass::TokenEmbedding
        );
        assert_eq!(classify_tensor("output.weight"), TensorClass::OutputHead);
        assert_eq!(
            classify_tensor("blk.3.attn_q.weight"),
            TensorClass::Attention
        );
        assert_eq!(
            classify_tensor("blk.3.attn_q_norm.weight"),
            TensorClass::AttnNorm
        );
        assert_eq!(
            classify_tensor("blk.3.ffn_gate_inp.weight"),
            TensorClass::MoeRouter
        );
        assert_eq!(
            classify_tensor("blk.3.ffn_gate_exps.weight"),
            TensorClass::MoeExpert
        );
        assert_eq!(
            classify_tensor("blk.3.ffn_gate_shexp.weight"),
            TensorClass::MoeSharedExpert
        );
        assert_eq!(classify_tensor("blk.3.ffn_gate.weight"), TensorClass::Ffn);
        assert_eq!(classify_tensor("blk.3.ffn_norm.weight"), TensorClass::Norm);
        assert_eq!(classify_tensor("rope_freqs.weight"), TensorClass::Other);
    }

    #[test]
    fn boolean_values_must_be_0_or_1() {
        // Handcraft a metadata section with an invalid bool.
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        let key = b"flag";
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(&7u32.to_le_bytes()); // bool
        out.push(2);
        let len = out.len() as u64;
        let err = inspect_reader(
            &mut std::io::Cursor::new(out),
            len,
            &Limits::default(),
            None,
        )
        .unwrap_err();
        assert!(err.message().contains("neither 0 nor 1"));
    }

    #[test]
    fn import_style_read_budget_is_enforced() {
        let bytes = synthetic_qwen3_bytes(&SyntheticConfig::tiny_qwen3());
        let len = bytes.len() as u64;
        let mut reader = std::io::Cursor::new(bytes);
        let err = inspect_reader(&mut reader, len, &Limits::default(), Some(16)).unwrap_err();
        assert!(err.message().contains("budget"));
    }

    #[test]
    fn unicode_and_hash_helpers() {
        assert_eq!(crate::ids::hex(&[0x00, 0xff]), "00ff");
    }
}
