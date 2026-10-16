//! Tuckmill store: downloads, packing, cache metadata, and the artifact
//! lifecycle (FR-07, FR-08, TM-007).
//!
//! Canonical model data lives under the user's Application Support directory
//! or an explicitly selected external volume. Installation is atomic:
//! interrupted work never appears as a valid artifact, and the source file is
//! never deleted to make space.

pub mod aliases;
pub mod cache;
pub mod digester;
pub mod download;
pub mod import;
pub mod layout;
pub mod lock;
pub mod pack;

pub use layout::StoreLayout;
