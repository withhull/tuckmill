//! The unified layer engine (ADR-011).
//!
//! Adapters resolve an artifact into a typed [`ModelGraph`]; the engine
//! executes any graph from shared blocks (attention variants, Gated
//! DeltaNet, SwiGLU, RMSNorm). Model math lives only in those blocks; the
//! engine owns command buffers, chunked prefill, sessions, preflight
//! planning and profiling. Weights are reached only through
//! [`WeightSource`], the seam where Phase 2 streaming plugs in.

pub mod adapters;
pub mod graph;
pub mod metal;

pub use graph::{
    AttentionSpec, FfnSpec, GatedDeltaNetSpec, LayerSpec, MixerSpec, ModelGraph, MtpSpec, Tensor,
};

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};

/// Read access to an artifact's tensors by inspection index. Blocks see
/// weights only through this trait.
pub trait WeightSource {
    fn tensor_type_id(&self, tensor_index: usize) -> Result<u32>;
    /// `(in_dim, out_dim)` in GGUF order (`out_dim` 1 for vectors).
    fn tensor_dims(&self, tensor_index: usize) -> Result<(u64, u64)>;
    fn tensor_bytes(&self, tensor_index: usize) -> Result<u64>;
    /// Fill `dst` with `dst.len()` bytes of the tensor from `offset`
    /// (bounded staging; the caller sizes `dst`).
    fn read_tensor_into(&self, tensor_index: usize, offset: u64, dst: &mut [u8]) -> Result<()>;
    /// A whole small vector, dequantized to f32 (norms, gates, kernels).
    fn read_vector(&self, tensor_index: usize) -> Result<Vec<f32>>;
}

/// Any GGUF artifact's tensors over an opened, verified file handle:
/// geometry from the inspection plus bounded positional reads. Family
/// independent (adapters bind tensors by index).
pub struct GgufWeights {
    file: std::fs::File,
    meta: Vec<crate::qwen35_forward::TensorMeta>,
}

impl GgufWeights {
    /// Build over an opened handle from an inspection of exactly these
    /// bytes (the executor checks the inspection's hashes against the
    /// import record and verifies the handle's content digest first).
    pub fn with_inspection(
        file: &std::fs::File,
        insp: &tuckmill_manifest::gguf::GgufInspection,
    ) -> Result<Self> {
        Ok(Self {
            file: file.try_clone().map_err(|e| {
                TuckmillError::new(
                    ErrorCode::StorageLost,
                    format!("cannot clone the verified artifact handle: {e}"),
                )
            })?,
            meta: crate::qwen35_forward::build_tensor_meta(insp)?,
        })
    }

    fn meta_of(&self, tensor_index: usize) -> Result<&crate::qwen35_forward::TensorMeta> {
        self.meta.get(tensor_index).ok_or_else(|| {
            TuckmillError::new(
                ErrorCode::Internal,
                format!("tensor index {tensor_index} is outside the inspected range"),
            )
        })
    }
}

impl WeightSource for GgufWeights {
    fn tensor_type_id(&self, tensor_index: usize) -> Result<u32> {
        Ok(self.meta_of(tensor_index)?.type_id)
    }

    fn tensor_dims(&self, tensor_index: usize) -> Result<(u64, u64)> {
        let m = self.meta_of(tensor_index)?;
        Ok((m.in_dim, m.out_dim))
    }

    fn tensor_bytes(&self, tensor_index: usize) -> Result<u64> {
        Ok(self.meta_of(tensor_index)?.byte_size)
    }

    fn read_tensor_into(&self, tensor_index: usize, offset: u64, dst: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let m = *self.meta_of(tensor_index)?;
        if offset + dst.len() as u64 > m.byte_size {
            return Err(TuckmillError::new(
                ErrorCode::Internal,
                format!("tensor {tensor_index} read past its {} bytes", m.byte_size),
            ));
        }
        let mut done = 0usize;
        while done < dst.len() {
            let n = self
                .file
                .read_at(&mut dst[done..], m.offset + offset + done as u64)
                .map_err(|e| {
                    TuckmillError::new(ErrorCode::StorageLost, format!("artifact read failed: {e}"))
                })?;
            if n == 0 {
                return Err(TuckmillError::new(
                    ErrorCode::StorageLost,
                    "artifact ended early while reading a tensor",
                ));
            }
            done += n;
        }
        Ok(())
    }

    fn read_vector(&self, tensor_index: usize) -> Result<Vec<f32>> {
        let m = *self.meta_of(tensor_index)?;
        let mut bytes = vec![0u8; m.byte_size as usize];
        self.read_tensor_into(tensor_index, 0, &mut bytes)?;
        let mut out = Vec::new();
        crate::tensor::dequant_into(m.type_id, &bytes, &mut out)?;
        Ok(out)
    }
}

impl<T: crate::qwen35_forward::TensorSource + ?Sized> WeightSource for T {
    fn tensor_type_id(&self, tensor_index: usize) -> Result<u32> {
        crate::qwen35_forward::TensorSource::tensor_type_id(self, tensor_index)
    }

    fn tensor_dims(&self, tensor_index: usize) -> Result<(u64, u64)> {
        crate::qwen35_forward::TensorSource::tensor_dims(self, tensor_index)
    }

    fn tensor_bytes(&self, tensor_index: usize) -> Result<u64> {
        crate::qwen35_forward::TensorSource::tensor_bytes(self, tensor_index)
    }

    fn read_tensor_into(&self, tensor_index: usize, offset: u64, dst: &mut [u8]) -> Result<()> {
        crate::qwen35_forward::TensorSource::read_tensor_into(self, tensor_index, offset, dst)
    }

    fn read_vector(&self, tensor_index: usize) -> Result<Vec<f32>> {
        crate::qwen35_forward::TensorSource::read_vector(self, tensor_index)
    }
}
