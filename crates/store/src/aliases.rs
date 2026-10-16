//! Alias registry: user-facing names that resolve to immutable artifact ids.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::ids::Id;

use crate::layout::StoreLayout;

pub const ALIASES_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasRegistry {
    pub schema_version: u32,
    /// alias -> artifact id
    pub aliases: BTreeMap<String, Id>,
}

impl Default for AliasRegistry {
    fn default() -> Self {
        Self {
            schema_version: ALIASES_SCHEMA_VERSION,
            aliases: BTreeMap::new(),
        }
    }
}

pub fn valid_alias(alias: &str) -> bool {
    !alias.is_empty()
        && alias.len() <= 64
        && alias
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
}

impl AliasRegistry {
    pub fn load(layout: &StoreLayout) -> Result<Self> {
        let path = layout.aliases_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = std::fs::read(&path).map_err(TuckmillError::from)?;
        let registry: AliasRegistry = serde_json::from_slice(&bytes).map_err(|e| {
            tuckmill_manifest::error::json_error(
                e,
                ErrorCode::InvalidConfiguration,
                "alias registry",
            )
        })?;
        if registry.schema_version != ALIASES_SCHEMA_VERSION {
            return Ok(Self::default());
        }
        Ok(registry)
    }

    pub fn save(&self, layout: &StoreLayout) -> Result<()> {
        let path = layout.aliases_path();
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| {
            tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "alias serialization")
        })?;
        std::fs::write(&tmp, bytes).map_err(TuckmillError::from)?;
        std::fs::rename(&tmp, &path).map_err(TuckmillError::from)?;
        crate::layout::fsync_dir(layout.root());
        Ok(())
    }

    pub fn set(&mut self, alias: &str, id: Id) -> Result<()> {
        if !valid_alias(alias) {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!(
                    "alias {alias:?} is invalid: use 1-64 characters of ASCII letters, digits, '-', '_' or '.'"
                ),
            ));
        }
        self.aliases.insert(alias.to_string(), id);
        Ok(())
    }

    pub fn resolve(&self, alias: &str) -> Option<&Id> {
        self.aliases.get(alias)
    }

    pub fn remove(&mut self, alias: &str) -> Option<Id> {
        self.aliases.remove(alias)
    }
}

/// Resolve a user string that is either an alias or a full artifact id.
pub fn resolve_reference(layout: &StoreLayout, reference: &str) -> Result<Id> {
    if reference.len() == 64 {
        if let Ok(id) = Id::new(reference.to_string()) {
            if layout.artifact_manifest_path(&id).exists() {
                return Ok(id);
            }
        }
    }
    let aliases = AliasRegistry::load(layout)?;
    if let Some(id) = aliases.resolve(reference) {
        if layout.artifact_manifest_path(id).exists() {
            return Ok(id.clone());
        }
        return Err(TuckmillError::new(
            ErrorCode::StorageLost,
            format!("alias {reference:?} points to artifact {id} which is not installed"),
        ));
    }
    Err(TuckmillError::new(
        ErrorCode::InvalidRequest,
        format!("no installed artifact or alias matches {reference:?}"),
    )
    .with_action("Run 'tuckmill cache list' to see installed models; import one first."))
}

/// Read a model artifact manifest from the store.
pub fn load_manifest(
    layout: &StoreLayout,
    id: &Id,
) -> Result<tuckmill_manifest::records::ModelArtifact> {
    let path = layout.artifact_manifest_path(id);
    let bytes = std::fs::read(&path).map_err(|e| {
        TuckmillError::new(
            ErrorCode::StorageLost,
            format!("cannot read manifest for artifact {id}: {e}"),
        )
        .with_source(e)
    })?;
    let artifact: tuckmill_manifest::records::ModelArtifact = serde_json::from_slice(&bytes)
        .map_err(|e| {
            tuckmill_manifest::error::json_error(
                e,
                ErrorCode::IntegrityFailure,
                "artifact manifest",
            )
        })?;
    Ok(artifact)
}

/// Atomically write a model artifact manifest.
pub fn save_manifest(
    layout: &StoreLayout,
    artifact: &tuckmill_manifest::records::ModelArtifact,
) -> Result<()> {
    let path = layout.artifact_manifest_path(&artifact.artifact_id);
    let dir = layout.artifact_dir(&artifact.artifact_id);
    std::fs::create_dir_all(&dir).map_err(TuckmillError::from)?;
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(artifact).map_err(|e| {
        tuckmill_manifest::error::json_error(e, ErrorCode::Internal, "artifact serialization")
    })?;
    std::fs::write(&tmp, bytes).map_err(TuckmillError::from)?;
    std::fs::rename(&tmp, &path).map_err(TuckmillError::from)?;
    crate::layout::fsync_dir(&dir);
    Ok(())
}

/// List every installed artifact manifest.
pub fn list_manifests(
    layout: &StoreLayout,
) -> Result<Vec<tuckmill_manifest::records::ModelArtifact>> {
    let dir = layout.artifacts_dir();
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Ok(out),
    };
    for entry in entries.flatten() {
        let manifest = entry.path().join("manifest.json");
        if manifest.exists() {
            if let Ok(bytes) = std::fs::read(&manifest) {
                if let Ok(artifact) =
                    serde_json::from_slice::<tuckmill_manifest::records::ModelArtifact>(&bytes)
                {
                    out.push(artifact);
                }
            }
        }
    }
    out.sort_by(|a, b| a.artifact_id.cmp(&b.artifact_id));
    Ok(out)
}

pub fn manifest_path_exists(layout: &StoreLayout, id: &Id) -> bool {
    layout.artifact_manifest_path(id).exists()
}

pub fn ensure_alias_free(aliases: &AliasRegistry, alias: &str, id: &Id) -> Result<()> {
    if let Some(existing) = aliases.resolve(alias) {
        if existing != id {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!("alias {alias:?} already refers to artifact {existing}"),
            )
            .with_action("Choose another alias or remove the existing one first."));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_validation() {
        assert!(valid_alias("qwen3-0.6b"));
        assert!(valid_alias("model.v2_test"));
        assert!(!valid_alias(""));
        assert!(!valid_alias("has space"));
        assert!(!valid_alias("slash/name"));
        assert!(!valid_alias(&"a".repeat(65)));
    }

    #[test]
    fn registry_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path());
        layout.ensure().unwrap();
        let mut reg = AliasRegistry::default();
        let id = Id::new("a".repeat(64)).unwrap();
        reg.set("local-model", id.clone()).unwrap();
        reg.save(&layout).unwrap();
        let loaded = AliasRegistry::load(&layout).unwrap();
        assert_eq!(loaded.resolve("local-model"), Some(&id));
    }

    #[test]
    fn unresolved_reference_is_actionable() {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path());
        layout.ensure().unwrap();
        let err = resolve_reference(&layout, "nope").unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidRequest);
    }
}
