//! End-to-end CLI tests: the full path from doctor through serve.
//!
//! These tests run the real `tuckmill` binary (including the isolated worker
//! process path) against a synthetic fixture and a temporary data directory.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use tuckmill_manifest::adapter::AdapterRegistry;
use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_tuckmill")
}

struct TestEnv {
    _dir: tempfile::TempDir,
    data_dir: PathBuf,
    fixture: PathBuf,
}

impl TestEnv {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("store");
        let fixture = dir.path().join("tiny-qwen3.gguf");
        write_synthetic(&fixture, &SyntheticConfig::tiny_qwen3()).unwrap();
        Self {
            _dir: dir,
            data_dir,
            fixture,
        }
    }

    fn run(&self, args: &[&str]) -> (i32, String, String) {
        let output = Command::new(bin())
            .args(args)
            .env("TUCKMILL_DATA_DIR", &self.data_dir)
            .output()
            .expect("spawn tuckmill");
        (
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stdout).to_string(),
            String::from_utf8_lossy(&output.stderr).to_string(),
        )
    }

    fn run_ok(&self, args: &[&str]) -> String {
        let (code, stdout, stderr) = self.run(args);
        assert_eq!(
            code, 0,
            "command {args:?} failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        stdout
    }
}

#[test]
fn doctor_reports_the_real_machine_and_never_guesses() {
    let env = TestEnv::new();
    let out = env.run_ok(&["doctor", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("doctor json");
    assert!(v["chip"].as_str().unwrap().contains("Apple"), "chip: {v}");
    assert!(v["memory_bytes"].as_u64().unwrap() > 0);
    assert_eq!(v["kernels"].as_array().unwrap().len(), 9);
    let statuses: Vec<&str> = v["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k["status"].as_str().unwrap())
        .collect();
    assert!(
        statuses
            .iter()
            .all(|s| *s == "specified" || *s == "implemented"),
        "kernel statuses are specified/implemented only"
    );
    let k01 = v["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["id"] == "K01")
        .expect("K01 present in the registry");
    assert_eq!(
        k01["status"], "implemented",
        "K01's first increment has landed"
    );
    // Human output too.
    let out = env.run_ok(&["doctor"]);
    assert!(out.contains("no certified execution engine"), "{out}");
}

#[test]
fn inspect_import_plan_run_bench_flow() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();

    // inspect --json carries identity hashes and tensor stats.
    let out = env.run_ok(&["inspect", &fixture, "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("inspect json");
    assert_eq!(v["format"], "gguf");
    assert_eq!(v["architecture"], "qwen3");
    assert_eq!(v["tensor_stats"]["total_tensors"], 25);
    assert!(v["identity_hash"].as_str().unwrap().len() == 64);

    // import with alias.
    let out = env.run_ok(&["import", &fixture, "--alias", "local-model", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("import json");
    let artifact_id = v["artifact"]["artifact_id"].as_str().unwrap().to_string();
    assert_eq!(artifact_id.len(), 64);

    // Re-import is idempotent.
    let out = env.run_ok(&["import", &fixture, "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("import json");
    assert_eq!(v["already_present"], true);

    // cache list shows it.
    let out = env.run_ok(&["cache", "list"]);
    assert!(out.contains("local-model"), "{out}");
    assert!(out.contains(&artifact_id[..12]), "{out}");

    // plan --json: resident, estimated, honest engine availability.
    let out = env.run_ok(&["plan", "local-model", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("plan json");
    assert_eq!(v["mode"], "resident");
    assert_eq!(v["confidence"], "estimated");
    assert_eq!(v["engine"]["availability"], "reference_test_only");
    assert!(v["assumptions"].as_array().unwrap().len() >= 5);

    // run without the gate must refuse with a typed error.
    let (code, _, stderr) = env.run(&["run", "local-model", "--prompt", "hi", "--max-output", "4"]);
    assert_eq!(code, 1);
    assert!(stderr.contains("UNAVAILABLE"), "{stderr}");
    assert!(stderr.contains("--allow-test-engine"), "{stderr}");

    // run with the gate: isolated worker path, synthetic tokens stream out.
    let out = env.run_ok(&[
        "run",
        "local-model",
        "--prompt",
        "hi",
        "--max-output",
        "5",
        "--allow-test-engine",
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("run json");
    assert_eq!(v["reason"], "length");
    assert_eq!(v["committed_tokens"], 5);
    assert_eq!(v["synthetic_output"], true);

    // pack is idempotent and verified.
    let out = env.run_ok(&["pack", "local-model", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("pack json");
    assert!(v["packed"]["segments"].as_u64().unwrap() >= 1);
    let out = env.run_ok(&["pack", "local-model", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("pack json");
    assert_eq!(v["already_packed"], true);

    // bench writes a machine-readable record with honest caveats.
    let out = env.run_ok(&[
        "bench",
        "local-model",
        "--input",
        "32",
        "--output",
        "8",
        "--repetitions",
        "1",
        "--allow-test-engine",
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("bench json");
    assert_eq!(v["engine"]["certified"], false);
    assert_eq!(v["performance_class"], "interactive");
    let caveats = v["caveats"].as_array().unwrap();
    assert!(caveats
        .iter()
        .any(|c| c.as_str().unwrap().contains("synthetic")));
    assert!(v["samples"].as_array().unwrap().len() == 1);
}

#[test]
fn rejected_plan_surfaces_as_typed_error() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "m"]);
    // Budget far below minimum live state: typed rejection with next action.
    let (code, stdout, stderr) = env.run(&[
        "run",
        "m",
        "--prompt",
        "x",
        "--memory-budget",
        "300MiB",
        "--allow-test-engine",
    ]);
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("CONTEXT_EXCEEDS_BUDGET") || stderr.contains("INVALID_PLAN"),
        "{stderr}"
    );
    // The plan command still shows the rejected plan document rather than failing.
    let out = env.run_ok(&["plan", "m", "--memory-budget", "300MiB", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["mode"], "rejected");
    assert!(v["rejection"]["code"].as_str().is_some());
}

#[test]
fn context_beyond_advertised_is_rejected() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "m"]);
    let (code, _, stderr) = env.run(&[
        "run",
        "m",
        "--prompt",
        "x",
        "--context",
        "8192",
        "--allow-test-engine",
    ]);
    assert_eq!(code, 1);
    assert!(stderr.contains("CONTEXT_EXCEEDS_BUDGET"), "{stderr}");
}

#[test]
fn cache_prune_dry_run_reports_without_deleting() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "m"]);
    let out = env.run_ok(&["cache", "prune", "--dry-run", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["dry_run"], true);
    // Artifact payload must still exist afterwards.
    let out = env.run_ok(&["cache", "list"]);
    assert!(out.contains("m "));
}

#[test]
fn serve_end_to_end_with_auth_and_streaming() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "local-model"]);

    let mut child = Command::new(bin())
        .args(["serve", "--port", "0", "--allow-test-engine"])
        .env("TUCKMILL_DATA_DIR", &env.data_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn serve");

    // Read the startup line to learn the bound port.
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);
    let mut addr_line = String::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut addr = None;
    while std::time::Instant::now() < deadline {
        addr_line.clear();
        if reader.read_line(&mut addr_line).unwrap_or(0) == 0 {
            break;
        }
        if let Some(rest) = addr_line
            .trim()
            .strip_prefix("tuckmill serve: listening on http://")
        {
            addr = Some(rest.to_string());
            break;
        }
    }
    let addr = addr.expect("serve should print its bound address");
    // Drain remaining stdout lines in a thread so the pipe never blocks.
    std::thread::spawn(move || {
        let mut sink = String::new();
        let _ = reader.read_to_string(&mut sink);
    });

    let token = std::fs::read_to_string(env.data_dir.join("control/api-token")).unwrap();
    let token = token.trim();

    // healthz needs no auth.
    let resp = http_request(
        &addr,
        "GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
        None,
    );
    assert!(resp.contains("200"), "{resp}");

    // Unauthorized.
    let resp = http_request(
        &addr,
        "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
        None,
    );
    assert!(resp.contains("401"), "{resp}");

    // Models list.
    let req = format!(
        "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n"
    );
    let resp = http_request(&addr, &req, None);
    assert!(resp.contains("local-model"), "{resp}");

    // Non-streaming completion.
    let body = serde_json::json!({
        "model": "local-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4
    })
    .to_string();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let resp = http_request(&addr, &req, None);
    assert!(resp.contains("200"), "{resp}");
    assert!(resp.contains("[ref:"), "{resp}");

    // Streaming completion.
    let body = serde_json::json!({
        "model": "local-model",
        "messages": [{"role": "user", "content": "stream"}],
        "max_tokens": 3,
        "stream": true
    })
    .to_string();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let resp = http_request(&addr, &req, None);
    assert!(resp.contains("text/event-stream"), "{resp}");
    assert!(resp.contains("data: [DONE]"), "{resp}");

    // Control status.
    let req = format!(
        "GET /control/status HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n"
    );
    let resp = http_request(&addr, &req, None);
    assert!(resp.contains("\"governor\""), "{resp}");
    assert!(resp.contains("\"queue_capacity\""), "{resp}");

    // Stop the server.
    let _ = child.kill();
    let _ = child.wait();
}

fn http_request(addr: &str, request: &str, _unused: Option<()>) -> String {
    let mut stream = TcpStream::connect(addr).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut out = String::new();
    let _ = stream.read_to_string(&mut out);
    out
}

/// Gate B end-to-end through the real API and the isolated TuckCore worker:
/// identical requests reuse one loaded model, sampling settings reach the
/// worker, stop sequences never leak, and usage accounts for the actual
/// prompt tokens.
#[test]
fn gate_b_chat_semantics_through_the_real_worker() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "local-model"]);

    let mut child = Command::new(bin())
        .args(["serve", "--port", "0", "--engine", "tuckcore"])
        .env("TUCKMILL_DATA_DIR", &env.data_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn serve");

    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);
    let mut addr_line = String::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let mut addr = None;
    while std::time::Instant::now() < deadline {
        addr_line.clear();
        if reader.read_line(&mut addr_line).unwrap_or(0) == 0 {
            break;
        }
        if let Some(rest) = addr_line
            .trim()
            .strip_prefix("tuckmill serve: listening on http://")
        {
            addr = Some(rest.to_string());
            break;
        }
    }
    let addr = addr.expect("serve should print its bound address");
    std::thread::spawn(move || {
        let mut sink = String::new();
        let _ = reader.read_to_string(&mut sink);
    });

    let token = std::fs::read_to_string(env.data_dir.join("control/api-token")).unwrap();
    let token = token.trim().to_string();

    let chat = |body: &serde_json::Value| -> serde_json::Value {
        let body = body.to_string();
        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let resp = http_request(&addr, &req, None);
        let payload = resp
            .split("\r\n\r\n")
            .nth(1)
            .unwrap_or_else(|| panic!("no body: {resp}"));
        serde_json::from_str(payload).unwrap_or_else(|e| panic!("not JSON ({e}): {resp}"))
    };
    let status = || -> serde_json::Value {
        let req = format!(
            "GET /control/status HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n"
        );
        let resp = http_request(&addr, &req, None);
        let payload = resp.split("\r\n\r\n").nth(1).expect("status body");
        serde_json::from_str(payload).expect("status JSON")
    };

    let messages = serde_json::json!([
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "say something"}
    ]);

    // Greedy requests are deterministic; ten equivalent sequential
    // requests must reuse one loaded model (worker_generation stays 1) and
    // must not inherit session state (identical visible output).
    let greedy_body = serde_json::json!({
        "model": "local-model",
        "messages": messages,
        "max_tokens": 6,
    });
    let first = chat(&greedy_body);
    assert!(
        !first["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .is_empty(),
        "{first}"
    );
    assert_eq!(first["choices"][0]["finish_reason"], "length");
    let prompt_tokens = first["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(
        prompt_tokens > 0,
        "usage must count the real prompt: {first}"
    );
    assert_eq!(first["usage"]["completion_tokens"], 6);
    assert_eq!(
        first["usage"]["total_tokens"].as_u64().unwrap(),
        prompt_tokens + 6
    );
    let first_text = first["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    for i in 0..9 {
        let again = chat(&greedy_body);
        assert_eq!(
            again["choices"][0]["message"]["content"].as_str().unwrap(),
            first_text,
            "request {i} must not inherit session state"
        );
    }
    let s = status();
    assert_eq!(
        s["worker_generation"], 1,
        "equivalent requests must reuse the loaded model: {s}"
    );
    let pid = s["worker_pid"].as_u64().expect("worker pid");
    assert!(pid > 0, "{s}");

    // Sampling: the requested transforms and seed reach the worker.
    let sampled = |seed: u64, top_p: Option<f64>| -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": "local-model",
            "messages": messages,
            "max_tokens": 12,
            "temperature": 1.0,
            "seed": seed,
        });
        if let Some(p) = top_p {
            body["top_p"] = serde_json::json!(p);
        }
        chat(&body)
    };
    let a = sampled(11, Some(0.9));
    let b = sampled(11, Some(0.9));
    assert_eq!(
        a["choices"][0]["message"]["content"],
        b["choices"][0]["message"]["content"]
    );
    assert_eq!(a["tuckmill"]["sampling"]["greedy"], false);
    let c = sampled(12, None);
    assert_ne!(
        a["choices"][0]["message"]["content"], c["choices"][0]["message"]["content"],
        "different seeds must sample different continuations"
    );

    // Stop handling: a stop sequence that completes mid-stream is trimmed
    // from visible output and reported as finish_reason "stop".
    let long = chat(&serde_json::json!({
        "model": "local-model",
        "messages": messages,
        "max_tokens": 8,
    }));
    let base = long["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        base.chars().count() >= 3,
        "fixture text too short: {base:?}"
    );
    let stop: String = base
        .chars()
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let stopped = chat(&serde_json::json!({
        "model": "local-model",
        "messages": messages,
        "max_tokens": 8,
        "stop": [stop.clone()],
    }));
    assert_eq!(stopped["choices"][0]["finish_reason"], "stop");
    let visible = stopped["choices"][0]["message"]["content"]
        .as_str()
        .unwrap();
    assert!(base.starts_with(visible), "{visible:?} vs {base:?}");
    assert!(!visible.contains(&stop), "stop marker leaked: {visible:?}");
    let completion_tokens = stopped["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(completion_tokens <= 8, "{stopped}");
    assert!(
        stopped["tuckmill"]["generated_tokens"].as_u64().unwrap() >= completion_tokens,
        "{stopped}"
    );

    // The worker persisted across every request above.
    let s = status();
    assert_eq!(s["worker_generation"], 1, "{s}");
    assert_eq!(s["worker_pid"].as_u64().unwrap(), pid);

    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn import_of_missing_file_is_a_clean_typed_error() {
    let env = TestEnv::new();
    let missing = env._dir.path().join("nope.gguf");
    let (code, _, stderr) = env.run(&["import", missing.to_str().unwrap()]);
    assert_eq!(code, 1);
    assert!(stderr.contains("INVALID_REQUEST"), "{stderr}");
    assert!(stderr.contains("next:"), "{stderr}");
}

#[test]
fn corrupt_artifact_is_rejected_before_publish() {
    let env = TestEnv::new();
    let path = env._dir.path().join("corrupt.gguf");
    let mut bytes = std::fs::read(&env.fixture).unwrap();
    bytes[100] ^= 0xff; // metadata corruption
    std::fs::write(&path, &bytes).unwrap();
    let (code, _, stderr) = env.run(&["import", path.to_str().unwrap()]);
    assert_eq!(code, 1);
    assert!(
        stderr.contains("INVALID_ARTIFACT") || stderr.contains("INTEGRITY_FAILURE"),
        "{stderr}"
    );
    // Nothing published.
    let out = env.run_ok(&["cache", "list", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["artifacts"].as_array().unwrap().len(), 0);
}

#[test]
fn corrupted_store_payload_is_refused_before_execution() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    let out = env.run_ok(&["import", &fixture, "--alias", "m", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    let id = v["artifact"]["artifact_id"].as_str().unwrap().to_string();
    let payload = env
        .data_dir
        .join("artifacts")
        .join(&id[..16])
        .join("model.bin");
    // Flip one byte deep inside the weight payload (past header + index).
    let mut bytes = std::fs::read(&payload).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&payload, &bytes).unwrap();

    let (code, _, stderr) = env.run(&[
        "run",
        "m",
        "--prompt",
        "x",
        "--engine",
        "tuckcore",
        "--max-output",
        "2",
    ]);
    assert_eq!(code, 1, "drifted payload must not execute");
    assert!(
        stderr.contains("INTEGRITY_FAILURE") && stderr.contains("content digest"),
        "expected content-digest refusal: {stderr}"
    );
}

#[test]
fn truncated_payload_is_refused_by_the_isolated_worker() {
    // Truncation must be caught by the size check on the very file object
    // the worker opens for execution — never mapped or executed.
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    let out = env.run_ok(&["import", &fixture, "--alias", "m", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    let id = v["artifact"]["artifact_id"].as_str().unwrap().to_string();
    let payload = env
        .data_dir
        .join("artifacts")
        .join(&id[..16])
        .join("model.bin");
    let len = std::fs::metadata(&payload).unwrap().len();
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(&payload)
        .unwrap();
    f.set_len(len - 1).unwrap();

    let (code, _, stderr) = env.run(&[
        "run",
        "m",
        "--prompt",
        "x",
        "--engine",
        "tuckcore",
        "--max-output",
        "2",
    ]);
    assert_eq!(code, 1, "truncated payload must not execute");
    // Refused by the typed gates (artifact inspection or the identity/digest
    // check), with a next action — never executed and never a panic.
    assert!(
        stderr.contains("INTEGRITY_FAILURE") || stderr.contains("INVALID_ARTIFACT"),
        "expected a typed integrity refusal for the truncated file: {stderr}"
    );
    assert!(
        stderr.contains("next:"),
        "typed errors carry an action: {stderr}"
    );
}

#[test]
fn replaced_payload_is_refused_by_the_isolated_worker() {
    // Replacing the payload path with different bytes (rename) must be
    // refused: the recorded identity no longer matches what opens.
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    let out = env.run_ok(&["import", &fixture, "--alias", "m", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    let id = v["artifact"]["artifact_id"].as_str().unwrap().to_string();
    let payload = env
        .data_dir
        .join("artifacts")
        .join(&id[..16])
        .join("model.bin");
    let len = std::fs::metadata(&payload).unwrap().len();
    let impostor = env._dir.path().join("impostor.bin");
    std::fs::write(&impostor, vec![0u8; len as usize]).unwrap();
    std::fs::rename(&impostor, &payload).unwrap();

    let (code, _, stderr) = env.run(&[
        "run",
        "m",
        "--prompt",
        "x",
        "--engine",
        "tuckcore",
        "--max-output",
        "2",
    ]);
    assert_eq!(code, 1, "replaced payload must not execute");
    assert!(
        stderr.contains("INTEGRITY_FAILURE") || stderr.contains("INVALID_ARTIFACT"),
        "expected a typed integrity refusal for the replaced file: {stderr}"
    );
    assert!(
        stderr.contains("next:"),
        "typed errors carry an action: {stderr}"
    );
}

#[test]
fn adversarial_metadata_is_bounded() {
    // A GGUF with an absurd tensor count must be rejected without allocation.
    let env = TestEnv::new();
    let path = env._dir.path().join("absurd.gguf");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // tensor count
    bytes.extend_from_slice(&0u64.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let (code, _, stderr) = env.run(&["inspect", path.to_str().unwrap()]);
    assert_eq!(code, 1);
    assert!(stderr.contains("INVALID_ARTIFACT"), "{stderr}");
}

#[test]
fn worker_crash_surfaces_as_worker_failed_and_supervisor_survives() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "m"]);

    let child = Command::new(bin())
        .args([
            "run",
            "m",
            "--prompt",
            "crash me",
            "--context",
            "4096",
            "--max-output",
            "3000",
            "--allow-test-engine",
        ])
        .env("TUCKMILL_DATA_DIR", &env.data_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn run");

    // Wait for the isolated worker to appear as a child of the run process.
    let run_pid = child.id();
    let mut worker_pid: Option<i32> = None;
    for _ in 0..200 {
        if let Ok(out) = Command::new("pgrep")
            .args(["-P", &run_pid.to_string(), "-f", "__worker-reference"])
            .output()
        {
            if let Some(pid) = String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .and_then(|l| l.trim().parse::<i32>().ok())
            {
                worker_pid = Some(pid);
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let worker_pid = worker_pid.expect("worker process should be running");
    std::thread::sleep(Duration::from_millis(200)); // let some tokens commit

    // Kill -9: no cleanup, no farewell.
    // SAFETY: test-only signal to a child process we spawned.
    assert_eq!(
        unsafe { libc::kill(worker_pid, libc::SIGKILL) },
        0,
        "kill worker"
    );

    let output = child.wait_with_output().expect("wait run");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "stderr: {stderr}");
    assert!(
        stderr.contains("WORKER_FAILED"),
        "crash must surface as WORKER_FAILED: {stderr}"
    );
    // The supervisor itself must still be usable afterwards.
    let out = env.run_ok(&["cache", "list"]);
    assert!(out.contains("m "), "{out}");
}

#[test]
fn sigint_cancels_a_run_at_a_committed_token_boundary() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "m"]);

    let child = Command::new(bin())
        .args([
            "run",
            "m",
            "--prompt",
            "interrupt me",
            "--context",
            "4096",
            "--max-output",
            "3000",
            "--allow-test-engine",
            "--json",
            "--quiet",
        ])
        .env("TUCKMILL_DATA_DIR", &env.data_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn run");

    // Let a few tokens commit, then interrupt.
    std::thread::sleep(Duration::from_millis(500));
    // SAFETY: test-only signal to the child process we spawned.
    assert_eq!(
        unsafe { libc::kill(child.id() as i32, libc::SIGINT) },
        0,
        "send SIGINT"
    );
    let output = child.wait_with_output().expect("wait run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(130),
        "interrupted runs exit 130\nstdout: {stdout}\nstderr: {stderr}"
    );
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("json record");
    assert_eq!(v["reason"], "cancelled");
    let committed = v["committed_tokens"].as_u64().unwrap();
    assert!(
        committed > 0,
        "some tokens should have committed before SIGINT"
    );
    assert!(committed < 3000, "cancellation must stop the run");
}

#[test]
fn tuckcore_executes_real_weights_with_honest_labels() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "local-model"]);

    let out = env.run_ok(&[
        "run",
        "local-model",
        "--prompt",
        "hello core",
        "--max-output",
        "5",
        "--engine",
        "tuckcore",
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("run json");
    assert_eq!(v["backend"], "tuckcore");
    assert_eq!(v["certified"], false);
    assert_eq!(v["synthetic_output"], false, "real weights were executed");
    assert_eq!(v["committed_tokens"], 5);
    assert_eq!(v["reason"], "length");
    let text = v["text"].as_str().unwrap();
    assert!(!text.is_empty());
    assert!(!text.contains("[ref:"), "not synthetic output: {text}");
}

#[test]
fn verify_proves_streaming_equivalence_bit_exact() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "m"]);

    let out = env.run_ok(&["verify", "m", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("verify json");
    assert_eq!(v["passed"], true, "verify must pass: {v}");
    assert_eq!(v["logits"]["exact_equal"], true);
    assert_eq!(v["logits"]["max_abs_diff"], 0.0);
    assert_eq!(v["generated_texts_equal"], true);
    assert_eq!(v["streaming_exercised"], true);
    let evictions = v["streamed"]["stats"]["evictions"].as_u64().unwrap();
    assert!(evictions > 0, "forced streaming must evict: {evictions}");
    let reads = v["streamed"]["stats"]["logical_read_bytes"]
        .as_u64()
        .unwrap();
    assert!(reads > 0, "streaming must read weights from storage");
    assert_eq!(v["resident"]["stats"]["in_flight_leases"], 0);
    assert_eq!(v["schema_version"], 3);
    assert_eq!(v["config"]["qualification"], "calibration");
    assert_eq!(v["config"]["policy_version"], "calibration-2026-10-16.1");
    assert!(v["config_id"].as_str().unwrap().len() == 64);
    assert_ne!(
        v["config_id"], v["run_id"],
        "config and run identities are distinct"
    );

    // The record is persisted under records/verify/.
    let verify_dir = env.data_dir.join("records/verify");
    let entries: Vec<_> = std::fs::read_dir(&verify_dir).unwrap().flatten().collect();
    assert_eq!(entries.len(), 1);

    // Immutable evidence: a rerun keeps the same configuration identity but
    // a distinct execution identity, and never overwrites the first record.
    let out2 = env.run_ok(&["verify", "m", "--json"]);
    let v2: serde_json::Value = serde_json::from_str(&out2).unwrap();
    assert_eq!(v2["config_id"], v["config_id"], "same experiment config");
    assert_ne!(v2["run_id"], v["run_id"], "distinct execution id");
    let entries: Vec<_> = std::fs::read_dir(&verify_dir).unwrap().flatten().collect();
    assert_eq!(entries.len(), 2, "every execution is preserved");
}

#[test]
fn llama_reference_comparison_when_available() {
    let Some(lib) = std::env::var("TUCKMILL_LLAMA_LIB")
        .ok()
        .filter(|p| std::path::Path::new(p).exists())
    else {
        eprintln!("skipping: TUCKMILL_LLAMA_LIB not set (run scripts/fetch-llama.sh)");
        return;
    };
    let _ = lib;
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "m"]);

    // Run through the pinned llama.cpp reference backend (real weights).
    let out = env.run_ok(&[
        "run",
        "m",
        "--prompt",
        "hello world",
        "--max-output",
        "3",
        "--engine",
        "llama",
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("run json");
    assert_eq!(v["backend"], "llama.cpp");
    assert_eq!(v["synthetic_output"], false);
    assert_eq!(v["committed_tokens"], 3);

    // Cross-backend numerical comparison: identical ids, registered tolerance.
    let out = env.run_ok(&[
        "verify",
        "m",
        "--reference",
        "--tolerance",
        "0.01",
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("verify json");
    assert_eq!(v["mode"], "reference-comparison");
    assert_eq!(v["passed"], true, "comparison must pass: {v}");
    assert_eq!(v["resident"]["label"], "tuckcore");
    assert_eq!(v["streamed"]["label"], "llama.cpp");
    let max_abs = v["logits"]["max_abs_diff"].as_f64().unwrap();
    assert!(max_abs <= 0.01, "within registered tolerance: {max_abs}");
    assert!(v["logits"]["vocab"].as_u64().unwrap() == 512);
}

#[test]
fn worker_and_adapter_consistency() {
    // The fixture must resolve the builtin qwen3 adapter metadata fully; this
    // guards against fixture/adapter drift that would shadow real bugs.
    let env = TestEnv::new();
    let registry = AdapterRegistry::builtin().unwrap();
    let adapter = registry.find("qwen3").unwrap();
    let insp = tuckmill_manifest::gguf::inspect(
        Path::new(&env.fixture),
        &tuckmill_manifest::gguf::Limits::default(),
    )
    .unwrap();
    adapter.validate_metadata(&insp).unwrap();
    adapter.kv_geometry(&insp).unwrap();
}

#[test]
fn verify_records_requested_and_resolved_selection() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "m"]);

    // Forced scalar + owned must actually execute scalar/owned on both sides
    // of the equivalence run, and the report must say so.
    let out = env.run_ok(&[
        "verify",
        "m",
        "--kernels",
        "scalar",
        "--storage",
        "owned",
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("verify json");
    assert_eq!(v["passed"], true, "verify must pass: {v}");
    assert_eq!(v["config"]["tuckcore_kernels_requested"], "scalar");
    assert_eq!(v["config"]["tuckcore_storage_requested"], "owned");
    let resolved = &v["config"]["resolved_tuckcore"];
    assert_eq!(resolved["kernel_impl"], "scalar");
    assert_eq!(resolved["storage_mode"], "owned");
    assert_eq!(v["resident"]["stats"]["kernel_impl"], "scalar");
    assert_eq!(v["resident"]["stats"]["storage"]["mode"], "owned");
    assert_eq!(v["streamed"]["stats"]["kernel_impl"], "scalar");
    assert_eq!(v["streamed"]["stats"]["storage"]["mode"], "owned");

    // Configuration-equivalent rerun: same configuration identity, distinct
    // execution identity.
    let out2 = env.run_ok(&[
        "verify",
        "m",
        "--kernels",
        "scalar",
        "--storage",
        "owned",
        "--json",
    ]);
    let v2: serde_json::Value = serde_json::from_str(&out2).unwrap();
    assert_eq!(
        v2["config_id"], v["config_id"],
        "same requested+resolved config"
    );
    assert_ne!(v2["run_id"], v["run_id"], "distinct execution id");

    // A different forced selection is a different configuration and must
    // resolve to the different implementation.
    let out3 = env.run_ok(&[
        "verify",
        "m",
        "--kernels",
        "neon",
        "--storage",
        "mapped",
        "--json",
    ]);
    let v3: serde_json::Value = serde_json::from_str(&out3).unwrap();
    assert_eq!(v3["passed"], true, "verify must pass: {v3}");
    assert_ne!(v3["config_id"], v["config_id"]);
    assert_eq!(v3["config"]["resolved_tuckcore"]["kernel_impl"], "neon");
    assert_eq!(v3["config"]["resolved_tuckcore"]["storage_mode"], "mapped");
    assert_eq!(v3["resident"]["stats"]["storage"]["mode"], "mapped");
}

#[test]
fn bench_warm_repetitions_and_long_output_drain() {
    let env = TestEnv::new();
    let fixture = env.fixture.display().to_string();
    env.run_ok(&["import", &fixture, "--alias", "m"]);

    // 600 tokens is far beyond the 256-event channel capacity of an isolated
    // worker run: it must complete (continuous draining), warm reps included.
    let out = env.run_ok(&[
        "bench",
        "m",
        "--input",
        "8",
        "--output",
        "600",
        "--repetitions",
        "3",
        "--allow-test-engine",
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("bench json");
    let samples = v["samples"].as_array().unwrap();
    assert_eq!(samples.len(), 3);
    for s in samples {
        assert_eq!(s["committed_tokens"], 600);
        let total = s["total_ms"].as_f64().unwrap();
        assert!(total > 0.0);
        if let Some(worker_total) = s["worker_total_ms"].as_f64() {
            assert!(
                worker_total <= total + 1.0,
                "worker-side total {worker_total} must not exceed supervisor total {total}"
            );
        }
    }
    assert_eq!(v["schema_version"], 3);
}
