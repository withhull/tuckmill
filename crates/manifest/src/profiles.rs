//! Separate representation from execution (PRD §11.4).
//!
//! Four immutable records with explicit change policies:
//! SourceModel → QualityProfile → PackingProfile → ExecutionProfile.
//! A packing change must be a lossless layout change; a quality change
//! requires explicit approval, never an automatic substitution.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{ErrorCode, Result, TuckmillError};
use crate::ids::Id;

/// The parent model identity: weights, tokenizer, architecture, license.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceModel {
    pub id: Id,
    pub parent: Option<Id>,
    pub weights_ref: String,
    pub tokenizer_hash: Id,
    pub arch_id: String,
    pub license: String,
}

/// Selected trained/quantized weights plus required transforms and
/// evaluation provenance. May not change automatically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityProfile {
    pub id: Id,
    pub source_model: Id,
    /// Human label, e.g. "Q4_K_M", "PQ2_0".
    pub weights_select: String,
    /// Required activation transforms (for example a rotation) that are part
    /// of the execution contract, not an optimization.
    pub required_transforms: Vec<String>,
    pub precision_map: BTreeMap<String, String>,
    /// Where evaluation evidence for this profile lives (may be empty).
    pub evaluation_provenance: Vec<String>,
}

/// Lossless code layout for the same quality profile. May change only to a
/// qualified equivalent layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackingProfile {
    pub id: Id,
    pub quality_profile: Id,
    pub layout_version: u32,
    pub alignment: u64,
    /// The packing this one is qualified-equivalent to, if any.
    pub equivalent_to: Option<Id>,
}

/// Kernels, accumulation, state precision, cache and sampler semantics.
/// May change only among certified same-profile paths.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionProfile {
    pub id: Id,
    pub quality_profile: Id,
    pub packing: Option<Id>,
    pub kernels: Vec<String>,
    pub accumulation: String,
    pub state_precision: String,
    pub cache_policy: String,
    pub sampler_semantics: String,
}

/// What kind of change is being proposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileChange {
    Quality,
    Packing { lossless_layout_only: bool },
    Execution,
}

/// Check whether a change may proceed automatically under the record's
/// policy. Returns the user-facing error for disallowed automatic changes.
pub fn check_change(change: ProfileChange) -> Result<()> {
    match change {
        ProfileChange::Quality => Err(TuckmillError::new(
            ErrorCode::InvalidConfiguration,
            "quality profile changes (weights, transforms, precision) may not happen automatically",
        )
        .with_action(
            "Re-run with an explicitly selected quality profile; importing or packing never re-quantizes silently.",
        )),
        ProfileChange::Packing {
            lossless_layout_only: true,
        } => Ok(()),
        ProfileChange::Packing {
            lossless_layout_only: false,
        } => Err(TuckmillError::new(
            ErrorCode::InvalidConfiguration,
            "a packing change that alters quantization values is a new quality profile",
        )
        .with_action("Create a new quality-profile identity and approve it explicitly.")),
        ProfileChange::Execution => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_changes_are_never_automatic() {
        let err = check_change(ProfileChange::Quality).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidConfiguration);
    }

    #[test]
    fn only_lossless_packing_is_automatic() {
        assert!(check_change(ProfileChange::Packing {
            lossless_layout_only: true
        })
        .is_ok());
        assert!(check_change(ProfileChange::Packing {
            lossless_layout_only: false
        })
        .is_err());
    }

    #[test]
    fn execution_changes_are_allowed_among_certified_paths() {
        assert!(check_change(ProfileChange::Execution).is_ok());
    }
}
