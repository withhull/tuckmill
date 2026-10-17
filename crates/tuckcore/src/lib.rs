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
pub mod gpu;
pub mod kernels;
pub mod loader;
pub mod model;
pub mod qwen35;
pub mod qwen35_ops;
pub mod tensor;
pub mod tokenizer;

/// Deterministic sampling transforms and the RNG live in the backend
/// contract (`tuckmill_backend::sampling`) so every engine selects tokens
/// under the identical declared policy; re-exported here for existing paths.
pub use tuckmill_backend::sampling;

pub use completion::{CompletionTracker, Lease};
pub use executor::TuckCore;
pub use kernels::{KernelSelection, ResolvedKernels};
pub use loader::{
    BlockLoader, BlockPolicy, LoaderSnapshot, ResidencySample, ResolvedStorage, StorageSelection,
    TensorEntry, WeightBlock,
};

mod executor;

/// Build identity for engine records: package version, build profile,
/// target architecture, and source revision. Kernel implementation and
/// storage choices are runtime-resolved and recorded per load
/// (stats/identity) — a build id alone never identifies them.
pub fn format_build_id() -> String {
    format!(
        "tuckmill-tuckcore {} ({}, {}, rev {})",
        env!("CARGO_PKG_VERSION"),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        std::env::consts::ARCH,
        env!("TUCKMILL_SOURCE_REV")
    )
}
