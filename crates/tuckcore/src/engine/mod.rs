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
    AttentionSpec, FfnSpec, GatedDeltaNetSpec, LayerSpec, MixerSpec, ModelGraph, Tensor,
};

use tuckmill_manifest::error::Result;

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
