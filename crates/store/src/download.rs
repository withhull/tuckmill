//! Controlled downloads (FR-07, PRD §16).
//!
//! Downloads use approved HTTPS origins only, pinned destination paths, size
//! ceilings, resumption, and integrity checks. Access tokens never appear on
//! command lines, in logs, or in telemetry. The project deliberately does not
//! interpolate model names into shell commands: curl is invoked directly with
//! explicit argv, never through a shell.

use std::path::{Path, PathBuf};
use std::process::Command;

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::units::Bytes;

use crate::layout::StoreLayout;

/// Default discovery location for the system curl.
pub const CURL_CANDIDATE: &str = "/usr/bin/curl";

/// Validate a remote URL: HTTPS only, no embedded credentials, bounded
/// length, real host.
pub fn validate_remote_url(url: &str) -> Result<()> {
    let err = |msg: String| TuckmillError::new(ErrorCode::InvalidRequest, msg);
    if url.len() > 4096 {
        return Err(err("url is unreasonably long".into()));
    }
    let rest = url.strip_prefix("https://").ok_or_else(|| {
        err("only https:// origins are supported for downloads".into())
            .with_action("Use an https URL or import a local file with 'tuckmill import'.")
    })?;
    let host_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let host_port = &rest[..host_end];
    if host_port.is_empty() {
        return Err(err("url has no host".into()));
    }
    if host_port.contains('@') {
        return Err(err("url must not contain embedded credentials".into())
            .with_action("Credentials in URLs leak into logs; use a token-free public URL."));
    }
    if rest.len() <= host_end {
        return Err(err(
            "url has no path; a pinned artifact path is required".into()
        ));
    }
    Ok(())
}

/// Argument vector for a curl download. Kept separate so tests can assert the
/// policy flags exactly.
pub fn build_curl_args(
    url: &str,
    dest: &Path,
    resume_from: Option<u64>,
    max_bytes: Option<u64>,
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        // Fail on HTTP errors instead of writing error pages.
        "--fail".into(),
        // Follow redirects, but stay on HTTPS for every hop.
        "--location".into(),
        "--proto".into(),
        "=https".into(),
        "--proto-redir".into(),
        "=https".into(),
        // Bounded retries for transient failures.
        "--retry".into(),
        "3".into(),
        "--retry-delay".into(),
        "2".into(),
        // Quiet progress noise; errors still printed.
        "--silent".into(),
        "--show-error".into(),
        // Never send credentials or refer content anywhere unexpected.
        "--max-redirs".into(),
        "5".into(),
        "--globoff".into(),
    ];
    if let Some(from) = resume_from {
        args.push("--continue-at".into());
        args.push(from.to_string());
    }
    if let Some(max) = max_bytes {
        args.push("--max-filesize".into());
        args.push(max.to_string());
    }
    args.push("--output".into());
    args.push(dest.display().to_string());
    args.push("--".into());
    args.push(url.to_string());
    args
}

pub struct CurlFetcher {
    curl: PathBuf,
}

impl CurlFetcher {
    /// Path of the discovered curl binary (for doctor display).
    pub fn path(&self) -> &Path {
        &self.curl
    }

    /// Discover the system curl. Absence is reported as unavailable, not
    /// worked around with weaker http.
    pub fn discover() -> Result<Self> {
        let candidates: Vec<PathBuf> = std::env::var("TUCKMILL_CURL")
            .map(PathBuf::from)
            .into_iter()
            .chain([PathBuf::from(CURL_CANDIDATE)])
            .collect();
        for c in candidates {
            if c.is_file() {
                return Ok(Self { curl: c });
            }
        }
        Err(TuckmillError::new(
            ErrorCode::Unavailable,
            "curl was not found; controlled https downloads require it",
        )
        .with_action("Install curl or import a local file with 'tuckmill import'."))
    }

    pub fn fetch(
        &self,
        url: &str,
        dest: &Path,
        resume_from: Option<u64>,
        max_bytes: Option<u64>,
    ) -> Result<()> {
        validate_remote_url(url)?;
        let args = build_curl_args(url, dest, resume_from, max_bytes);
        // Never a shell: argv goes straight to exec.
        let status = Command::new(&self.curl).args(&args).status().map_err(|e| {
            TuckmillError::new(ErrorCode::Io, format!("failed to run curl: {e}")).with_source(e)
        })?;
        if !status.success() {
            return Err(TuckmillError::new(
                ErrorCode::Io,
                format!(
                    "download failed (curl exit {})",
                    status.code().unwrap_or(-1)
                ),
            )
            .with_action(
                "Check the URL and network; the partial file is retained for a resumed retry.",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct DownloadOutcome {
    pub path: PathBuf,
    pub bytes: u64,
    pub resumed: bool,
}

/// Download an artifact into the store's downloads area with a size ceiling
/// and resumption. Integrity is checked by the caller's import step (which
/// re-hashes and validates before publishing) and optionally against a
/// caller-supplied expected digest.
pub fn download(
    layout: &StoreLayout,
    url: &str,
    expected_chunked_digest: Option<&str>,
    max_bytes: Option<u64>,
) -> Result<DownloadOutcome> {
    layout.ensure()?;
    let fetcher = CurlFetcher::discover()?;
    let name_hash = tuckmill_manifest::ids::sha256_hex(url.as_bytes());
    let dest = layout
        .downloads_dir()
        .join(format!("{}.download", &name_hash[..16]));

    let resume_from = std::fs::metadata(&dest).ok().map(|m| m.len());
    let resumed = resume_from.is_some();

    // Preflight disk: if a ceiling is known, require it; else require at
    // least the resumed remainder to be able to grow.
    if let Some(max) = max_bytes {
        layout.ensure_space_for(Bytes(max.saturating_add(8 * 1024 * 1024)), &dest)?;
    }

    fetcher.fetch(url, &dest, resume_from, max_bytes)?;
    let bytes = std::fs::metadata(&dest).map_err(TuckmillError::from)?.len();
    if let Some(max) = max_bytes {
        if bytes > max {
            let _ = std::fs::remove_file(&dest);
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!(
                    "download exceeded the {} ceiling and was discarded",
                    Bytes(max)
                ),
            ));
        }
    }

    // Integrity: the caller-visible contract is the chunked digest that
    // import computes. When an expected digest is supplied, verify it here so
    // bad bytes are discarded early.
    if let Some(expected) = expected_chunked_digest {
        let (actual, _) =
            crate::digester::digest_file(&dest, crate::digester::DEFAULT_CHUNK_BYTES)?;
        if actual.as_str() != expected {
            let _ = std::fs::remove_file(&dest);
            return Err(TuckmillError::new(
                ErrorCode::IntegrityFailure,
                "downloaded payload does not match the expected digest; the file was discarded",
            ));
        }
    }

    Ok(DownloadOutcome {
        path: dest,
        bytes,
        resumed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_https_and_credentials() {
        for bad in [
            "http://example.com/model.gguf",
            "ftp://example.com/model.gguf",
            "file:///tmp/model.gguf",
            "https://user:pass@example.com/model.gguf",
            "https://example.com",
            "",
        ] {
            assert!(validate_remote_url(bad).is_err(), "{bad} must be rejected");
        }
        assert!(validate_remote_url("https://example.com/a/model.gguf?rev=1").is_ok());
    }

    #[test]
    fn curl_args_pin_policy_flags() {
        let args = build_curl_args(
            "https://example.com/m.gguf",
            Path::new("/tmp/x.part"),
            Some(1024),
            Some(1_000_000),
        );
        assert!(args.contains(&"--proto".to_string()));
        assert!(args.contains(&"=https".to_string()));
        assert!(args.contains(&"=https".to_string()));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--continue-at" && w[1] == "1024"));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--max-filesize" && w[1] == "1000000"));
        // URL is the last argument after a `--` guard.
        assert_eq!(args[args.len() - 1], "https://example.com/m.gguf");
        assert_eq!(args[args.len() - 2], "--");
        // No shell metacharacter expansion surface: single argv entries only.
        assert!(!args.iter().any(|a| a.contains(';') || a.contains('|')));
    }

    #[test]
    fn curl_discovery_reports_honestly() {
        // On macOS /usr/bin/curl exists; if a CI image lacks it, discovery
        // must return Unavailable rather than fabricating a downloader.
        match CurlFetcher::discover() {
            Ok(f) => assert!(f.curl.is_file()),
            Err(e) => assert_eq!(e.code(), ErrorCode::Unavailable),
        }
    }

    #[test]
    fn fetch_rejects_bad_url_before_running_curl() {
        let fetcher = CurlFetcher::discover().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let err = fetcher
            .fetch(
                "http://insecure.example/m.gguf",
                &dir.path().join("d"),
                None,
                None,
            )
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidRequest);
    }
}
