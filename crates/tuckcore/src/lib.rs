//! TuckCore: experimental custom native executor (ADR-005).
//!
//! Stage 0 is deliberately narrow and honest: dense Qwen3 GGUF artifacts,
//! scalar f32 reference kernels, bounded weight streaming through a block
//! loader with explicit buffer ownership and fence-based release. It is the
//! numerical oracle later Metal kernels are validated against, and the
//! vehicle for proving the 8 GB streaming architecture before optimizing it.
//!
//! TuckCore never changes the model: same artifact (identity-verified),
//! same token IDs, same declared numerical profile. Custom Metal kernels are
//! added only after profiling identifies measurable gaps.

pub mod completion;
pub mod forward;
pub mod loader;
pub mod model;
pub mod sampling;
pub mod tensor;
pub mod tokenizer;

pub use completion::{CompletionTracker, Lease};
pub use executor::TuckCore;
pub use loader::{BlockLoader, BlockPolicy, LoaderSnapshot, TensorEntry, WeightBlock};

mod executor;

/// Build identity for engine records.
pub fn format_build_id() -> String {
    format!(
        "tuckmill-tuckcore {} ({})",
        env!("CARGO_PKG_VERSION"),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    )
}
