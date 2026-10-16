//! Local control and inference API (FR-13).
//!
//! Loopback-only HTTP with a generated bearer credential (0600 file),
//! Host/Origin validation, strict request schemas that reject unsupported
//! fields explicitly, SSE streaming, cancellation via client disconnect or
//! the control endpoint, and typed error bodies. Administrative operations
//! live on the authenticated control surface; installing models or reading
//! arbitrary paths is not callable through the inference schema.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;

use tuckmill_manifest::error::{ErrorCode, TuckmillError};
use tuckmill_manifest::ids::{now_unix_ms, secure_random_hex};

use crate::runtime::{RunEvent, Runtime};

pub const DEFAULT_PORT: u16 = 8378;
const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 1024 * 1024;
const MAX_CONNECTIONS: usize = 16;

// ---- request schemas (strict: unknown fields are rejected explicitly) ----

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
}

fn validate_chat_request(req: &ChatCompletionRequest) -> Result<(), TuckmillError> {
    if req.messages.is_empty() {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "messages must contain at least one message",
        ));
    }
    for m in &req.messages {
        match m.role.as_str() {
            "system" | "user" | "assistant" => {}
            other => {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!(
                        "role {other:?} is not supported; supported roles: system, user, assistant"
                    ),
                ))
            }
        }
    }
    if let Some(t) = req.temperature {
        if !(0.0..=2.0).contains(&t) {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "temperature must be within 0.0..=2.0",
            ));
        }
    }
    if let Some(p) = req.top_p {
        if !(0.0..=1.0).contains(&p) {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "top_p must be within 0.0..=1.0",
            ));
        }
    }
    if let Some(stop) = &req.stop {
        if stop.len() > 4 {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "at most 4 stop sequences are supported",
            ));
        }
    }
    if let Some(0) = req.max_tokens {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "max_tokens must be positive",
        ));
    }
    Ok(())
}

// ---- HTTP plumbing ----

struct HttpRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

fn read_request(stream: &mut BufReader<TcpStream>) -> Result<Option<HttpRequest>, TuckmillError> {
    let mut request_line = String::new();
    match stream.read_line(&mut request_line) {
        Ok(0) => return Ok(None),
        Ok(_) => {}
        Err(e) => return Err(TuckmillError::from(e)),
    }
    let request_line = request_line.trim_end();
    let mut parts = request_line.split(' ');
    let (method, path) = match (parts.next(), parts.next(), parts.next()) {
        (Some(m), Some(p), Some(v)) if v.starts_with("HTTP/1.") => (m.to_string(), p.to_string()),
        _ => {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "malformed request line",
            ))
        }
    };
    if path.len() > 2048 {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "request path is too long",
        ));
    }
    let mut headers = BTreeMap::new();
    let mut header_bytes = request_line.len();
    loop {
        let mut line = String::new();
        stream.read_line(&mut line).map_err(TuckmillError::from)?;
        header_bytes += line.len();
        if header_bytes > MAX_HEADER_BYTES {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "request headers exceed the size cap",
            ));
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    // Body: Content-Length only; chunked transfer is not part of the subset.
    let mut body = Vec::new();
    if let Some(te) = headers.get("transfer-encoding") {
        return Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            format!("transfer-encoding {te:?} is not supported; use content-length"),
        ));
    }
    if let Some(len) = headers.get("content-length") {
        let len: usize = len
            .parse()
            .map_err(|_| TuckmillError::new(ErrorCode::InvalidRequest, "invalid content-length"))?;
        if len > MAX_BODY_BYTES {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!("request body exceeds {MAX_BODY_BYTES} bytes"),
            ));
        }
        body.resize(len, 0);
        stream.read_exact(&mut body).map_err(TuckmillError::from)?;
    }
    Ok(Some(HttpRequest {
        method,
        path,
        headers,
        body,
    }))
}

fn respond_json(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    value: &serde_json::Value,
) -> std::io::Result<()> {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(&body)?;
    stream.flush()
}

fn respond_error(stream: &mut TcpStream, err: &TuckmillError) {
    let status = match err.code() {
        ErrorCode::InvalidRequest => 400,
        ErrorCode::QueueFull => 429,
        ErrorCode::Cancelled => 499,
        ErrorCode::UnsupportedArchitecture
        | ErrorCode::UnsupportedQuantization
        | ErrorCode::ContextExceedsBudget => 422,
        ErrorCode::WorkerFailed | ErrorCode::Internal => 500,
        ErrorCode::MemoryPressurePause => 503,
        _ => 500,
    };
    let _ = respond_json(stream, status, "error", &err.to_json());
}

// ---- authentication ----

/// Load or create the local API credential. The file lives in the store's
/// control directory with user-only permissions and is never logged.
pub fn load_or_create_token(control_dir: &std::path::Path) -> Result<String, TuckmillError> {
    std::fs::create_dir_all(control_dir).map_err(TuckmillError::from)?;
    let path = control_dir.join("api-token");
    if path.exists() {
        let token = std::fs::read_to_string(&path).map_err(TuckmillError::from)?;
        let token = token.trim().to_string();
        if !token.is_empty() {
            return Ok(token);
        }
    }
    let token = secure_random_hex(32)?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &token).map_err(TuckmillError::from)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))
            .map_err(TuckmillError::from)?;
    }
    std::fs::rename(&tmp, &path).map_err(TuckmillError::from)?;
    Ok(token)
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ---- server ----

pub struct ApiServer {
    runtime: Arc<Runtime>,
    token: String,
    listener: TcpListener,
    conn_count: Arc<AtomicU64>,
}

impl ApiServer {
    pub fn bind(
        runtime: Arc<Runtime>,
        port: u16,
        token: Option<String>,
    ) -> Result<Self, TuckmillError> {
        let token = match token {
            Some(t) => t,
            None => load_or_create_token(&runtime.layout.control_dir())?,
        };
        // Loopback only, by construction (PRD §16).
        let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| {
            TuckmillError::new(
                ErrorCode::InvalidConfiguration,
                format!("cannot bind 127.0.0.1:{port}: {e}"),
            )
            .with_source(e)
        })?;
        Ok(Self {
            runtime,
            token,
            listener,
            conn_count: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn local_addr(&self) -> std::net::SocketAddr {
        self.listener
            .local_addr()
            .unwrap_or_else(|_| "127.0.0.1:0".parse().expect("literal"))
    }

    /// Serve until `shutdown` is set. One accept loop plus bounded
    /// per-connection threads.
    pub fn serve(self, shutdown: Arc<AtomicBool>) -> Result<(), TuckmillError> {
        self.listener
            .set_nonblocking(true)
            .map_err(TuckmillError::from)?;
        while !shutdown.load(Ordering::Relaxed) {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    if self.conn_count.load(Ordering::Relaxed) >= MAX_CONNECTIONS as u64 {
                        let mut s = stream;
                        let err = TuckmillError::new(
                            ErrorCode::QueueFull,
                            "too many concurrent connections",
                        );
                        respond_error(&mut s, &err);
                        continue;
                    }
                    let runtime = self.runtime.clone();
                    let token = self.token.clone();
                    let count = self.conn_count.clone();
                    count.fetch_add(1, Ordering::Relaxed);
                    std::thread::Builder::new()
                        .name("tuckmill-http".into())
                        .spawn(move || {
                            let _guard = ConnGuard(count);
                            let _ = handle_connection(runtime, token, stream);
                        })
                        .map_err(TuckmillError::from)?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => return Err(TuckmillError::from(e)),
            }
        }
        Ok(())
    }
}

struct ConnGuard(Arc<AtomicU64>);

impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

fn handle_connection(
    runtime: Arc<Runtime>,
    token: String,
    stream: TcpStream,
) -> Result<(), TuckmillError> {
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(TuckmillError::from)?;
    let mut reader = BufReader::new(stream.try_clone().map_err(TuckmillError::from)?);
    let mut stream = stream;
    let request = match read_request(&mut reader) {
        Ok(Some(r)) => r,
        Ok(None) => return Ok(()),
        Err(e) => {
            respond_error(&mut stream, &e);
            return Ok(());
        }
    };

    // Host validation.
    if let Some(host) = request.headers.get("host") {
        let host_ok = host == "127.0.0.1"
            || host.starts_with("127.0.0.1:")
            || host == "localhost"
            || host.starts_with("localhost:");
        if !host_ok {
            respond_error(
                &mut stream,
                &TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!("Host {host:?} is not accepted; bind is loopback-only"),
                ),
            );
            return Ok(());
        }
    } else {
        respond_error(
            &mut stream,
            &TuckmillError::new(ErrorCode::InvalidRequest, "Host header is required"),
        );
        return Ok(());
    }
    // Origin: CORS is off.
    if let Some(origin) = request.headers.get("origin") {
        if origin != "null" {
            respond_error(
                &mut stream,
                &TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    "cross-origin requests are not enabled",
                ),
            );
            return Ok(());
        }
    }

    let path = request.path.split('?').next().unwrap_or("").to_string();

    // Public health probe (no auth, no data).
    if request.method == "GET" && path == "/healthz" {
        let _ = respond_json(&mut stream, 200, "OK", &serde_json::json!({"status":"ok"}));
        return Ok(());
    }

    // Everything else requires the bearer token.
    let authorized = request
        .headers
        .get("authorization")
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|presented| constant_time_eq(presented.as_bytes(), token.as_bytes()))
        .unwrap_or(false);
    if !authorized {
        let _ = respond_json(
            &mut stream,
            401,
            "Unauthorized",
            &serde_json::json!({
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "missing or invalid bearer token",
                    "action": "Read the token from the control directory api-token file."
                }
            }),
        );
        return Ok(());
    }

    match (request.method.as_str(), path.as_str()) {
        ("GET", "/v1/models") => {
            let models = list_models(&runtime);
            let _ = respond_json(&mut stream, 200, "OK", &models);
            Ok(())
        }
        ("POST", "/v1/chat/completions") => {
            handle_chat_completions(runtime, stream, &request.body);
            Ok(())
        }
        ("GET", "/control/status") => {
            let status = runtime.status();
            let _ = respond_json(
                &mut stream,
                200,
                "OK",
                &serde_json::to_value(status).unwrap_or_else(|_| serde_json::json!({})),
            );
            Ok(())
        }
        ("POST", "/control/cancel") => {
            let cancelled = runtime.cancel_active();
            let _ = respond_json(
                &mut stream,
                200,
                "OK",
                &serde_json::json!({"cancelled": cancelled}),
            );
            Ok(())
        }
        ("GET", "/control/bench/latest") => {
            match latest_bench(&runtime) {
                Ok(value) => {
                    let _ = respond_json(&mut stream, 200, "OK", &value);
                }
                Err(e) => respond_error(&mut stream, &e),
            }
            Ok(())
        }
        _ => {
            respond_error(
                &mut stream,
                &TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!("no route for {} {path}", request.method),
                ),
            );
            Ok(())
        }
    }
}

fn list_models(runtime: &Runtime) -> serde_json::Value {
    let mut data = Vec::new();
    let aliases = tuckmill_store::aliases::AliasRegistry::load(&runtime.layout).unwrap_or_default();
    let reverse: BTreeMap<&str, &String> = aliases
        .aliases
        .iter()
        .map(|(a, id)| (id.as_str(), a))
        .collect();
    if let Ok(manifests) = tuckmill_store::aliases::list_manifests(&runtime.layout) {
        for m in manifests {
            let id = m.artifact_id.as_str();
            data.push(serde_json::json!({
                "id": id,
                "object": "model",
                "created": m.created_unix_ms / 1000,
                "owned_by": "tuckmill",
                "tuckmill": {
                    "alias": reverse.get(id).map(|a| a.as_str()),
                    "arch_id": m.arch_id,
                    "quant_label": m.quant_label,
                    "file_bytes": m.file_bytes,
                    "context_length": m.advertised_context_length,
                    "adapter_certification": runtime.registry.find(
                        m.arch_id.as_deref().unwrap_or("")).map(|a| a.certification_state()),
                }
            }));
        }
    }
    serde_json::json!({"object": "list", "data": data})
}

fn latest_bench(runtime: &Runtime) -> Result<serde_json::Value, TuckmillError> {
    let dir = runtime.layout.bench_dir();
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(meta) = entry.metadata() {
                    let t = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                    if newest.as_ref().map(|(bt, _)| t > *bt).unwrap_or(true) {
                        newest = Some((t, path));
                    }
                }
            }
        }
    }
    match newest {
        Some((_, path)) => {
            let bytes = std::fs::read(&path).map_err(TuckmillError::from)?;
            serde_json::from_slice(&bytes).map_err(|e| {
                tuckmill_manifest::error::json_error(
                    e,
                    ErrorCode::IntegrityFailure,
                    "benchmark record",
                )
            })
        }
        None => Err(TuckmillError::new(
            ErrorCode::InvalidRequest,
            "no benchmark runs recorded yet",
        )),
    }
}

fn handle_chat_completions(runtime: Arc<Runtime>, mut stream: TcpStream, body: &[u8]) {
    let request: ChatCompletionRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            // Strict schemas: unknown/unsupported fields are rejected
            // explicitly, with the supported subset stated.
            let err = TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!("request does not match the supported chat-completions subset: {e}"),
            )
            .with_action(
                "Supported fields: model, messages[{role,content}], stream, max_tokens, temperature, top_p, stop, seed.",
            );
            respond_error(&mut stream, &err);
            return;
        }
    };
    if let Err(e) = validate_chat_request(&request) {
        respond_error(&mut stream, &e);
        return;
    }

    // Resolve the model and plan.
    let (artifact, adapter) = match runtime.resolve_artifact(&request.model) {
        Ok(v) => v,
        Err(e) => {
            respond_error(&mut stream, &e);
            return;
        }
    };
    let max_tokens = request.max_tokens.unwrap_or(runtime.default_max_output);
    let plan = match runtime.plan_for(
        &artifact,
        &adapter,
        runtime.default_budget,
        runtime.default_context,
        max_tokens,
    ) {
        Ok(p) => p,
        Err(e) => {
            respond_error(&mut stream, &e);
            return;
        }
    };
    if let Err(e) = runtime.plan_admitted(&plan) {
        respond_error(&mut stream, &e);
        return;
    }
    if let Err(e) = runtime.admit_engine(&plan) {
        respond_error(&mut stream, &e);
        return;
    }

    // Prompt tokenization: the reference test engine accepts arbitrary ids
    // (synthetic); a real backend tokenizes with the artifact's own byte-level
    // BPE tables.
    let identity = runtime.engine_identity();
    let synthetic = tuckmill_backend::is_synthetic_engine(&identity.engine);
    let prompt_tokens = if synthetic {
        prompt_to_tokens(&request.messages)
    } else {
        let encoder = match crate::tokenization::open_encoder(
            &runtime.layout.artifact_payload_path(&artifact.artifact_id),
            None,
        ) {
            Ok(t) => t,
            Err(e) => {
                respond_error(&mut stream, &e);
                return;
            }
        };
        match encoder.encode(&messages_text(&request.messages), false) {
            Ok(ids) => ids,
            Err(e) => {
                respond_error(&mut stream, &e);
                return;
            }
        }
    };
    let stop = request.stop.clone().unwrap_or_default();
    let seed = request.seed.unwrap_or_else(now_unix_ms);

    let ticket = match runtime.submit() {
        Ok(t) => t,
        Err(e) => {
            respond_error(&mut stream, &e);
            return;
        }
    };
    let artifact_ref = tuckmill_backend::ArtifactRef::from_record(
        &artifact,
        &runtime.layout.artifact_payload_path(&artifact.artifact_id),
    );
    let run = match runtime.start_run(
        &ticket,
        plan.clone(),
        artifact_ref,
        prompt_tokens,
        max_tokens,
        stop,
        seed,
    ) {
        Ok(r) => r,
        Err(e) => {
            respond_error(&mut stream, &e);
            return;
        }
    };

    let completion_id = format!("chatcmpl-{}", &run.request_id[4..]);
    let created = now_unix_ms() / 1000;

    if request.stream {
        stream_sse(
            &mut stream,
            &runtime,
            &run,
            &completion_id,
            created,
            &request.model,
            &plan,
        );
    } else {
        collect_json(
            &mut stream,
            &runtime,
            &run,
            &completion_id,
            created,
            &request.model,
            &plan,
        );
    }
}

/// Deterministic prompt-to-token conversion for the reference engine. No
/// vocabulary or tokenizer is executed in this build; the exact token IDs a
/// real tokenizer would produce arrive with the certified executor.
fn prompt_to_tokens(messages: &[ChatMessage]) -> Vec<u32> {
    crate::engine::synthetic_prompt_tokens(&messages_text(messages), 4096)
}

fn messages_text(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| format!("{}:{}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn tuckmill_extension(
    runtime: &Runtime,
    plan: &tuckmill_manifest::plan::ExecutionPlan,
    request_id: &str,
) -> serde_json::Value {
    let identity = runtime.engine_identity();
    serde_json::json!({
        "request_id": request_id,
        "plan_id": plan.plan_id.as_str(),
        "mode": plan.mode.as_str(),
        "planned_engine": plan.engine.engine,
        "engine_availability": plan.engine.availability,
        "backend": identity.engine,
        "certified": identity.certified,
        "confidence": plan.confidence,
        "performance_class": plan.performance_class,
        "synthetic_output": tuckmill_backend::is_synthetic_engine(&identity.engine),
        "tokenizer": if tuckmill_backend::is_synthetic_engine(&identity.engine) {
            "none (synthetic engine)"
        } else {
            "reference-first (llama.cpp pinned; gpt2-style BPE fallback)"
        },
    })
}

fn collect_json(
    stream: &mut TcpStream,
    runtime: &Runtime,
    run: &crate::runtime::ActiveRun,
    completion_id: &str,
    created: u64,
    model: &str,
    plan: &tuckmill_manifest::plan::ExecutionPlan,
) {
    let mut text = String::new();
    let mut committed: u64 = 0;
    loop {
        match run.events.recv() {
            Ok(RunEvent::Token { text: t, .. }) => {
                text.push_str(&t);
                committed += 1;
            }
            Ok(RunEvent::State(_)) => {}
            Ok(RunEvent::Done { reason, .. }) => {
                let body = serde_json::json!({
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": normalize_finish(&reason),
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": committed,
                        "total_tokens": committed,
                    },
                    "tuckmill": tuckmill_extension(runtime, plan, &run.request_id),
                });
                let _ = runtime.status(); // keeps runtime alive for logging hooks
                let _ = respond_json(stream, 200, "OK", &body);
                return;
            }
            Ok(RunEvent::Error {
                code,
                message,
                action,
            }) => {
                let err = TuckmillError::new(
                    serde_json::from_value::<ErrorCode>(serde_json::Value::String(code))
                        .unwrap_or(ErrorCode::WorkerFailed),
                    message,
                )
                .with_action(action);
                respond_error(stream, &err);
                return;
            }
            Err(_) => {
                let err = TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "run event stream closed unexpectedly",
                );
                respond_error(stream, &err);
                return;
            }
        }
    }
}

fn normalize_finish(reason: &str) -> serde_json::Value {
    match reason {
        "length" => serde_json::json!("length"),
        "cancelled" => serde_json::json!("cancelled"),
        _ => serde_json::json!("stop"),
    }
}

fn stream_sse(
    stream: &mut TcpStream,
    runtime: &Runtime,
    run: &crate::runtime::ActiveRun,
    completion_id: &str,
    created: u64,
    model: &str,
    plan: &tuckmill_manifest::plan::ExecutionPlan,
) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\nConnection: close\r\n\r\n";
    if stream.write_all(head.as_bytes()).is_err() {
        run.request_cancel();
        return;
    }
    let _ = stream.flush();

    let mut first = true;
    let mut role_sent = false;
    loop {
        match run.events.recv() {
            Ok(RunEvent::State(_state)) => {
                // State transitions are reflected in the extension object of
                // the next chunk rather than as separate SSE events to stay
                // within the OpenAI subset.
            }
            Ok(RunEvent::Token { index: _, text }) => {
                if first {
                    // First content chunk also carries the role delta.
                    let _ = first;
                    first = false;
                }
                let identity = runtime.engine_identity();
                let chunk = serde_json::json!({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": if !role_sent {
                            role_sent = true;
                            serde_json::json!({"role": "assistant", "content": text})
                        } else {
                            serde_json::json!({"content": text})
                        },
                        "finish_reason": serde_json::Value::Null,
                    }],
                    "tuckmill": {
                        "request_id": run.request_id,
                        "backend": identity.engine,
                        "synthetic_output": tuckmill_backend::is_synthetic_engine(&identity.engine),
                    },
                });
                if write_sse(stream, &chunk.to_string()).is_err() {
                    // Client disconnected: prompt cancellation, never fake it.
                    run.request_cancel();
                    return;
                }
            }
            Ok(RunEvent::Done { reason, committed }) => {
                let chunk = serde_json::json!({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": normalize_finish(&reason),
                    }],
                    "tuckmill": {
                        "request_id": run.request_id,
                        "plan_id": plan.plan_id.as_str(),
                        "committed_tokens": committed,
                    },
                });
                let _ = write_sse(stream, &chunk.to_string());
                let _ = stream.write_all(b"data: [DONE]\n\n");
                let _ = stream.flush();
                runtime.logger.info(
                    "SSE_DONE",
                    [("request_id".to_string(), serde_json::json!(run.request_id))]
                        .into_iter()
                        .collect(),
                );
                return;
            }
            Ok(RunEvent::Error {
                code,
                message,
                action,
            }) => {
                let err = TuckmillError::new(
                    serde_json::from_value::<ErrorCode>(serde_json::Value::String(code))
                        .unwrap_or(ErrorCode::WorkerFailed),
                    message,
                )
                .with_action(action);
                let _ = write_sse(stream, &err.to_json().to_string());
                let _ = stream.write_all(b"data: [DONE]\n\n");
                let _ = stream.flush();
                return;
            }
            Err(_) => return,
        }
    }
}

fn write_sse(stream: &mut TcpStream, json: &str) -> std::io::Result<()> {
    stream.write_all(b"data: ")?;
    stream.write_all(json.as_bytes())?;
    stream.write_all(b"\n\n")?;
    stream.flush()
}

/// Format the token file path for display; the token value itself is never
/// printed.
pub fn token_file_path(runtime: &Runtime) -> PathBuf {
    runtime.layout.control_dir().join("api-token")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::RuntimeOptions;
    use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};
    use tuckmill_manifest::units::Bytes;
    use tuckmill_store::layout::StoreLayout;

    fn test_server() -> (tempfile::TempDir, Arc<Runtime>, ApiServer, String) {
        let dir = tempfile::tempdir().unwrap();
        let layout = StoreLayout::open(dir.path().join("store"));
        let src = dir.path().join("m.gguf");
        write_synthetic(&src, &SyntheticConfig::tiny_qwen3()).unwrap();
        let registry = tuckmill_manifest::adapter::AdapterRegistry::builtin().unwrap();
        tuckmill_store::import::import_local(
            &layout,
            &registry,
            &src,
            &tuckmill_store::import::ImportOptions {
                alias: Some("local".into()),
                ..Default::default()
            },
        )
        .unwrap();
        let runtime = Runtime::open(RuntimeOptions {
            data_dir: Some(layout.root().to_path_buf()),
            engine_mode: crate::runtime::EngineMode::InProcess,
            queue_capacity: 4,
            default_budget: Some(Bytes::parse("2GiB").unwrap()),
            default_context: 2048,
            default_max_output: 64,
            governor_poll: Duration::from_millis(200),
            monitor: true,
            allow_test_engine: true,
            ..Default::default()
        })
        .unwrap();
        let server = ApiServer::bind(runtime.clone(), 0, None).unwrap();
        let token = server.token.clone();
        (dir, runtime, server, token)
    }

    fn raw_request(addr: std::net::SocketAddr, req: &str) -> String {
        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        stream.write_all(req.as_bytes()).unwrap();
        let mut out = String::new();
        let _ = stream.read_to_string(&mut out);
        out
    }

    #[test]
    fn auth_host_and_origin_are_enforced() {
        let (_dir, runtime, server, token) = test_server();
        let addr = server.local_addr();
        let handle = {
            let shutdown = Arc::new(AtomicBool::new(false));
            let s2 = shutdown.clone();
            let h = std::thread::spawn(move || server.serve(s2));
            (h, shutdown)
        };

        // No token: 401.
        let resp = raw_request(
            addr,
            "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
        );
        assert!(resp.contains("401"), "{resp}");

        // Bad host: rejected.
        let resp = raw_request(
            addr,
            &format!(
                "GET /v1/models HTTP/1.1\r\nHost: evil.example\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n"
            ),
        );
        assert!(resp.contains("400"), "{resp}");

        // Origin set: rejected.
        let resp = raw_request(
            addr,
            &format!(
                "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://example.com\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n"
            ),
        );
        assert!(resp.contains("400"), "{resp}");

        // Authorized: lists the model.
        let resp = raw_request(
            addr,
            &format!(
                "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n"
            ),
        );
        assert!(resp.contains("200"), "{resp}");
        assert!(resp.contains("\"object\":\"list\""), "{resp}");
        assert!(resp.contains("qwen3"), "{resp}");

        handle.1.store(true, Ordering::Relaxed);
        let _ = handle.0.join();
        runtime.shutdown();
    }

    #[test]
    fn chat_completions_streams_and_finishes() {
        let (_dir, runtime, server, token) = test_server();
        let addr = server.local_addr();
        let shutdown = Arc::new(AtomicBool::new(false));
        let s2 = shutdown.clone();
        let handle = std::thread::spawn(move || server.serve(s2));

        let body = serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
            "stream": false
        })
        .to_string();
        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let resp = raw_request(addr, &req);
        assert!(resp.contains("200"), "{resp}");
        assert!(resp.contains("[ref:0]"), "{resp}");
        assert!(resp.contains("\"synthetic_output\":true"), "{resp}");

        // Unsupported field: explicit rejection.
        let body = serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function"}]
        })
        .to_string();
        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let resp = raw_request(addr, &req);
        assert!(resp.contains("400"), "{resp}");
        assert!(resp.contains("INVALID_REQUEST"), "{resp}");

        shutdown.store(true, Ordering::Relaxed);
        let _ = handle.join();
        runtime.shutdown();
    }

    #[test]
    fn sse_streams_chunks_until_done() {
        let (_dir, runtime, server, token) = test_server();
        let addr = server.local_addr();
        let shutdown = Arc::new(AtomicBool::new(false));
        let s2 = shutdown.clone();
        let handle = std::thread::spawn(move || server.serve(s2));

        let body = serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "stream please"}],
            "max_tokens": 4,
            "stream": true
        })
        .to_string();
        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let resp = raw_request(addr, &req);
        assert!(resp.contains("text/event-stream"), "{resp}");
        assert!(resp.contains("chat.completion.chunk"), "{resp}");
        assert!(resp.contains("data: [DONE]"), "{resp}");
        assert!(resp.contains("\"finish_reason\":\"length\""), "{resp}");

        shutdown.store(true, Ordering::Relaxed);
        let _ = handle.join();
        runtime.shutdown();
    }

    #[test]
    fn token_file_is_created_with_user_only_permissions() {
        let dir = tempfile::tempdir().unwrap();
        let token = load_or_create_token(dir.path()).unwrap();
        assert_eq!(token.len(), 64);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.path().join("api-token"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let again = load_or_create_token(dir.path()).unwrap();
        assert_eq!(token, again);
    }

    #[test]
    fn prompt_tokens_are_deterministic_and_bounded() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hello world".into(),
        }];
        let a = prompt_to_tokens(&messages);
        let b = prompt_to_tokens(&messages);
        assert_eq!(a, b);
        assert!(!a.is_empty());
        assert!(a.len() <= 4096);
    }

    #[test]
    fn validation_rejects_bad_fields() {
        let base = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "x".into(),
            }],
            stream: false,
            max_tokens: None,
            temperature: Some(1.0),
            top_p: None,
            stop: None,
            seed: None,
        };
        assert!(validate_chat_request(&base).is_ok());
        let bad_role = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: "tool".into(),
                content: "x".into(),
            }],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            seed: None,
        };
        assert!(validate_chat_request(&bad_role).is_err());
        let bad_temp = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "x".into(),
            }],
            stream: false,
            max_tokens: None,
            temperature: Some(3.0),
            top_p: None,
            stop: None,
            seed: None,
        };
        assert!(validate_chat_request(&bad_temp).is_err());
    }
}
