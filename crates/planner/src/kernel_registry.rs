//! Kernel registry (PRD §11.3 work packages K01-K09).
//!
//! A kernel is selected by operation, shape, tensor format, activation/state
//! precision, GPU family, OS/driver, workspace budget and validated build ID.
//! Until a kernel's gate passes, every requiring profile falls back to the
//! reference path or is rejected before allocation.

use serde::Deserialize;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelSpec {
    pub id: String,
    pub name: String,
    /// "specified" | "implemented" | "certified" | "disabled".
    pub status: String,
    #[serde(default)]
    pub reference_fallback: bool,
    #[serde(default)]
    pub gates: Vec<String>,
}

impl KernelSpec {
    pub fn is_available(&self) -> bool {
        self.status == "implemented" || self.status == "certified"
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelRegistry {
    pub schema_version: u32,
    #[serde(default)]
    pub revision_note: String,
    pub kernels: Vec<KernelSpec>,
}

impl KernelRegistry {
    pub fn builtin() -> Result<Self> {
        let src = include_str!("../../../kernels/registry.json");
        let registry: KernelRegistry = serde_json::from_str(src).map_err(|e| {
            tuckmill_manifest::error::json_error(
                e,
                ErrorCode::InvalidConfiguration,
                "builtin kernel registry",
            )
        })?;
        for k in &registry.kernels {
            if !(k.id.len() == 3
                && k.id.starts_with('K')
                && k.id[1..].chars().all(|c| c.is_ascii_digit()))
            {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidConfiguration,
                    format!("kernel registry contains invalid id {:?}", k.id),
                ));
            }
        }
        Ok(registry)
    }

    pub fn find(&self, id: &str) -> Option<&KernelSpec> {
        self.kernels.iter().find(|k| k.id == id)
    }

    /// True when every listed kernel is implemented or certified in this
    /// build. Required ids that are not in the registry at all count as
    /// unavailable (fail closed).
    pub fn all_available(&self, ids: &[String]) -> bool {
        ids.iter()
            .all(|id| self.find(id).map(|k| k.is_available()).unwrap_or(false))
    }

    /// Ids from `ids` that are not available in this build.
    pub fn missing(&self, ids: &[String]) -> Vec<String> {
        ids.iter()
            .filter(|id| self.find(id).map(|k| !k.is_available()).unwrap_or(true))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_registry_loads_all_work_packages() {
        let r = KernelRegistry::builtin().unwrap();
        assert_eq!(r.kernels.len(), 9);
        for k in &r.kernels {
            assert!(
                k.status == "specified" || k.status == "implemented",
                "{}: unexpected status {:?}",
                k.id,
                k.status
            );
            assert!(
                k.reference_fallback,
                "memory: every kernel keeps a reference fallback"
            );
            assert!(!k.gates.is_empty());
        }
        // K01's first increment has landed; the remaining packages are
        // still specification-only.
        assert_eq!(r.find("K01").unwrap().status, "implemented");
    }

    #[test]
    fn availability_tracks_registry_status() {
        let r = KernelRegistry::builtin().unwrap();
        // K01's first increment is available; K03 is not implemented yet,
        // so a profile requiring it still falls back to the reference path.
        let ids = vec!["K01".to_string(), "K03".to_string()];
        assert!(r.find("K01").unwrap().is_available());
        assert!(!r.all_available(&ids), "K03 is not implemented");
        assert_eq!(r.missing(&ids), vec!["K03".to_string()]);
        assert!(!r.all_available(&["K02".to_string()]));
        assert_eq!(r.missing(&["K02".to_string()]), vec!["K02".to_string()]);
    }

    #[test]
    fn unknown_kernel_ids_fail_closed() {
        let r = KernelRegistry::builtin().unwrap();
        assert!(!r.all_available(&["K99".to_string()]));
        assert_eq!(r.missing(&["K99".to_string()]), vec!["K99".to_string()]);
    }
}
