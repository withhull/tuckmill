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

use crate::bundle::{ChatMessageSpec, ChatRole, ModelBundle, RequestMode, RequestSpec};
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
        if stop.iter().any(|s| s.is_empty()) {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "stop sequences must be non-empty",
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

    // B1 order: resolve the model bundle (tokenizer + template), render the
    // model-specific chat prompt, tokenize it, count the actual prompt
    // tokens, validate the context/output allowance — and only then admit a
    // plan and execute under the typed request.
    let (artifact, adapter) = match runtime.resolve_artifact(&request.model) {
        Ok(v) => v,
        Err(e) => {
            respond_error(&mut stream, &e);
            return;
        }
    };
    let bundle = match ModelBundle::open(&runtime.layout, &artifact) {
        Ok(b) => b,
        Err(e) => {
            respond_error(&mut stream, &e);
            return;
        }
    };
    let max_tokens = request.max_tokens.unwrap_or(runtime.default_max_output);
    let sampling = tuckmill_backend::SamplingPolicy {
        temperature: request.temperature.unwrap_or(0.0) as f32,
        top_p: request.top_p.map(|p| p as f32),
    };
    let stop = request.stop.clone().unwrap_or_default();
    let seed = request.seed.unwrap_or_else(now_unix_ms);
    let messages: Vec<ChatMessageSpec> = request
        .messages
        .iter()
        .map(|m| ChatMessageSpec {
            role: ChatRole::parse(&m.role).expect("validated role"),
            content: m.content.clone(),
        })
        .collect();
    let spec = match RequestSpec::chat(
        &bundle,
        messages,
        max_tokens,
        stop,
        seed,
        sampling,
        request.stream,
        runtime.default_context,
    ) {
        Ok(s) => s,
        Err(e) => {
            respond_error(&mut stream, &e);
            return;
        }
    };

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
        crate::runtime::RunRequest {
            prompt_tokens: spec.prompt_tokens.clone(),
            max_output_tokens: spec.max_output_tokens,
            stop: spec.stop.clone(),
            seed: spec.seed,
            sampling: spec.sampling,
        },
    ) {
        Ok(r) => r,
        Err(e) => {
            respond_error(&mut stream, &e);
            return;
        }
    };

    let meta = CompletionMeta {
        completion_id: format!("chatcmpl-{}", &run.request_id[4..]),
        created: now_unix_ms() / 1000,
        model: request.model.clone(),
        plan: plan.clone(),
        spec,
    };

    if meta.spec.stream {
        stream_sse(&mut stream, &runtime, &run, &meta);
    } else {
        collect_json(&mut stream, &runtime, &run, &meta);
    }
}

/// Per-request metadata shared by the JSON and SSE serializers.
struct CompletionMeta {
    completion_id: String,
    created: u64,
    model: String,
    plan: tuckmill_manifest::plan::ExecutionPlan,
    spec: RequestSpec,
}

fn tuckmill_extension(
    runtime: &Runtime,
    meta: &CompletionMeta,
    run: &crate::runtime::ActiveRun,
) -> serde_json::Value {
    let identity = runtime.engine_identity();
    let mode = match &meta.spec.mode {
        RequestMode::Chat { template, .. } => serde_json::json!({
            "kind": "chat",
            "chat_template": template,
        }),
        RequestMode::Raw { .. } => serde_json::json!({"kind": "raw"}),
    };
    serde_json::json!({
        "request_id": run.request_id,
        "plan_id": meta.plan.plan_id.as_str(),
        "mode": meta.plan.mode.as_str(),
        "planned_engine": meta.plan.engine.engine,
        "engine_availability": meta.plan.engine.availability,
        "backend": identity.engine,
        "certified": identity.certified,
        "confidence": meta.plan.confidence,
        "performance_class": meta.plan.performance_class,
        "synthetic_output": tuckmill_backend::is_synthetic_engine(&identity.engine),
        "tokenizer": meta.spec.tokenizer_label,
        "prompt": mode,
        "prompt_tokens": meta.spec.prompt_tokens.len(),
        "sampling": {
            "temperature": meta.spec.sampling.temperature,
            "top_p": meta.spec.sampling.top_p,
            "greedy": meta.spec.sampling.is_greedy(),
            "seed": meta.spec.seed,
        },
    })
}

fn collect_json(
    stream: &mut TcpStream,
    runtime: &Runtime,
    run: &crate::runtime::ActiveRun,
    meta: &CompletionMeta,
) {
    let mut text = String::new();
    loop {
        match run.events.recv() {
            Ok(RunEvent::Token { text: t, .. }) => {
                text.push_str(&t);
            }
            Ok(RunEvent::State(_)) => {}
            Ok(RunEvent::Done {
                reason,
                committed,
                generated,
                timing: _,
            }) => {
                let prompt_tokens = meta.spec.prompt_tokens.len() as u64;
                let body = serde_json::json!({
                    "id": meta.completion_id,
                    "object": "chat.completion",
                    "created": meta.created,
                    "model": meta.model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": normalize_finish(&reason),
                    }],
                    "usage": {
                        // Actual tokenized prompt; visible-token completion
                        // count from the worker's own accounting.
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": committed,
                        "total_tokens": prompt_tokens + committed,
                    },
                    "tuckmill": tuckmill_extension(runtime, meta, run),
                });
                let mut body = body;
                body["tuckmill"]["generated_tokens"] = serde_json::json!(generated);
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
    meta: &CompletionMeta,
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
                    "id": meta.completion_id,
                    "object": "chat.completion.chunk",
                    "created": meta.created,
                    "model": meta.model,
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
            Ok(RunEvent::Done {
                reason,
                committed,
                generated,
                timing: _,
            }) => {
                let prompt_tokens = meta.spec.prompt_tokens.len() as u64;
                let chunk = serde_json::json!({
                    "id": meta.completion_id,
                    "object": "chat.completion.chunk",
                    "created": meta.created,
                    "model": meta.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": normalize_finish(&reason),
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": committed,
                        "total_tokens": prompt_tokens + committed,
                    },
                    "tuckmill": {
                        "request_id": run.request_id,
                        "plan_id": meta.plan.plan_id.as_str(),
                        "committed_tokens": committed,
                        "generated_tokens": generated,
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

    /// Issue a chat-completions request and return the raw HTTP response.
    fn chat_request(addr: std::net::SocketAddr, token: &str, body: &serde_json::Value) -> String {
        let body = body.to_string();
        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        raw_request(addr, &req)
    }

    fn response_json(resp: &str) -> serde_json::Value {
        let body = resp.split("\r\n\r\n").nth(1).expect("response body");
        serde_json::from_str(body).unwrap_or_else(|e| panic!("not JSON ({e}): {resp}"))
    }

    /// Collect the visible text from an SSE response body.
    fn sse_text(resp: &str) -> String {
        let mut text = String::new();
        for line in resp.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload == "[DONE]" {
                continue;
            }
            if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(payload) {
                if let Some(content) = chunk["choices"][0]["delta"]["content"].as_str() {
                    text.push_str(content);
                }
            }
        }
        text
    }

    #[test]
    fn chat_prompts_are_rendered_and_tokenized_with_the_model_contract() {
        let (_dir, runtime, server, token) = test_server();
        let addr = server.local_addr();
        let shutdown = Arc::new(AtomicBool::new(false));
        let s2 = shutdown.clone();
        let handle = std::thread::spawn(move || server.serve(s2));

        let resp = chat_request(
            addr,
            &token,
            &serde_json::json!({
                "model": "local",
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "hello"}
                ],
                "max_tokens": 3,
                "stream": false
            }),
        );
        assert!(resp.contains("200"), "{resp}");
        let body = response_json(&resp);
        // Usage accounts for the actual rendered+tokenized prompt.
        let prompt_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap();
        assert!(prompt_tokens > 0, "{body}");
        assert_eq!(body["usage"]["completion_tokens"], 3);
        assert_eq!(
            body["usage"]["total_tokens"].as_u64().unwrap(),
            prompt_tokens + 3
        );
        // The request is reported as a chat with the recognized template,
        // and greedy by default.
        assert_eq!(body["tuckmill"]["prompt"]["kind"], "chat");
        assert_eq!(body["tuckmill"]["prompt"]["chat_template"], "qwen3-chatml");
        assert_eq!(body["tuckmill"]["sampling"]["greedy"], true);
        assert_eq!(body["tuckmill"]["generated_tokens"], 3);

        // Nonzero temperature is refused by the synthetic engine instead of
        // silently generating greedily.
        let resp = chat_request(
            addr,
            &token,
            &serde_json::json!({
                "model": "local",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 3,
                "temperature": 1.0
            }),
        );
        assert!(resp.contains("UNAVAILABLE"), "{resp}");
        assert!(!resp.contains("200 OK"), "{resp}");

        shutdown.store(true, Ordering::Relaxed);
        let _ = handle.join();
        runtime.shutdown();
    }

    #[test]
    fn streaming_and_non_streaming_agree_and_unicode_roundtrips() {
        let (_dir, runtime, server, token) = test_server();
        let addr = server.local_addr();
        let shutdown = Arc::new(AtomicBool::new(false));
        let s2 = shutdown.clone();
        let handle = std::thread::spawn(move || server.serve(s2));

        let request = serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "héllo 🐎 日本語"}],
            "max_tokens": 4,
        });
        let mut non_stream = request.clone();
        non_stream["stream"] = serde_json::json!(false);
        let mut stream = request.clone();
        stream["stream"] = serde_json::json!(true);

        let resp_a = chat_request(addr, &token, &non_stream);
        let body_a = response_json(&resp_a);
        assert_eq!(body_a["choices"][0]["finish_reason"], "length");
        let text_a = body_a["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();

        let resp_b = chat_request(addr, &token, &stream);
        assert!(resp_b.contains("text/event-stream"), "{resp_b}");
        let text_b = sse_text(&resp_b);

        assert_eq!(
            text_a, text_b,
            "visible output must be identical across transports"
        );
        // Unicode content tokenized and rendered without loss.
        assert!(
            body_a["usage"]["prompt_tokens"].as_u64().unwrap() > 0,
            "{body_a}"
        );
        assert!(resp_b.contains("\"completion_tokens\":4"), "{resp_b}");

        shutdown.store(true, Ordering::Relaxed);
        let _ = handle.join();
        runtime.shutdown();
    }

    #[test]
    fn stop_sequences_are_trimmed_from_visible_output() {
        let (_dir, runtime, server, token) = test_server();
        let addr = server.local_addr();
        let shutdown = Arc::new(AtomicBool::new(false));
        let s2 = shutdown.clone();
        let handle = std::thread::spawn(move || server.serve(s2));

        let resp = chat_request(
            addr,
            &token,
            &serde_json::json!({
                "model": "local",
                "messages": [{"role": "user", "content": "stop early"}],
                "max_tokens": 8,
                "stop": ["[ref:2]"]
            }),
        );
        assert!(resp.contains("200"), "{resp}");
        let body = response_json(&resp);
        let text = body["choices"][0]["message"]["content"].as_str().unwrap();
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(text, "[ref:0][ref:1]");
        assert!(!text.contains("[ref:2]"), "stop marker leaked: {text:?}");
        assert_eq!(body["usage"]["completion_tokens"], 2);
        // The stop token was generated (and billed honestly as generated),
        // but only visible tokens count as completion tokens.
        assert!(body["tuckmill"]["generated_tokens"].as_u64().unwrap() >= 2);

        shutdown.store(true, Ordering::Relaxed);
        let _ = handle.join();
        runtime.shutdown();
    }

    #[test]
    fn max_tokens_beyond_context_is_rejected_before_admission() {
        let (_dir, runtime, server, token) = test_server();
        let addr = server.local_addr();
        let shutdown = Arc::new(AtomicBool::new(false));
        let s2 = shutdown.clone();
        let handle = std::thread::spawn(move || server.serve(s2));

        let resp = chat_request(
            addr,
            &token,
            &serde_json::json!({
                "model": "local",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4096
            }),
        );
        assert!(resp.contains("CONTEXT_EXCEEDS_BUDGET"), "{resp}");
        assert!(!resp.contains("200 OK"), "{resp}");

        shutdown.store(true, Ordering::Relaxed);
        let _ = handle.join();
        runtime.shutdown();
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
        let empty_stop = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "x".into(),
            }],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: Some(vec!["".into()]),
            seed: None,
        };
        assert!(validate_chat_request(&empty_stop).is_err());
    }
}
