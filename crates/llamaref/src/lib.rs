//! Tuckmill reference backend: pinned llama.cpp behind the checked C shim
//! (ADR-005/ADR-006).
//!
//! llama.cpp is the compatibility backend and the numerical oracle. It is
//! executed through `native/llamaref/tm_llama_shim.c`, compiled against the
//! pinned revision's own headers by `scripts/fetch-llama.sh` — so struct
//! layouts are compile-time checked and the Rust side only speaks a stable
//! pointer/integer ABI. The library is discovered at runtime
//! (`TUCKMILL_LLAMA_LIB` or the `lib_path` load option): the repository never
//! vendors llama.cpp.
//!
//! Execution is CPU-only here (`n_gpu_layers = 0`): deterministic across GPU
//! families and free of shader-library coupling. GPU offload is a separately
//! measured milestone. `certified` stays false: the shim wrapper itself is
//! uncertified until reference-vs-custom comparisons are recorded.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::time::Instant;

use tuckmill_backend::{
    ArtifactRef, BackendCapabilities, BackendId, BackendStats, FinishReason, GenControl,
    GenOutcome, GenerateRequest, InferenceBackend, RunState,
};
use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};
use tuckmill_manifest::plan::{ExecutionMode, ExecutionPlan};
use tuckmill_manifest::records::EngineIdentity;
use tuckmill_manifest::{BACKEND_ABI_VERSION, WORKER_PROTOCOL_VERSION};

/// llama.cpp revision this shim is compiled against (scripts/fetch-llama.sh).
pub const LLAMA_REVISION: &str = "b11695";
pub const SHIM_VERSION: c_int = 4;
const LIB_FILE_NAME: &str = "libtm_llama.dylib";

#[repr(C)]
struct TmHandle {
    _private: [u8; 0],
}

type FnShimVersion = unsafe extern "C" fn() -> c_int;
type FnLoad = unsafe extern "C" fn(
    *const c_char,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *mut c_char,
    c_int,
) -> *mut TmHandle;
type FnProbe = unsafe extern "C" fn(*const TmHandle) -> i32;
type FnFree = unsafe extern "C" fn(*mut TmHandle);
type FnVocabSize = unsafe extern "C" fn(*const TmHandle) -> i32;
type FnBosEos = unsafe extern "C" fn(*const TmHandle) -> i32;
type FnNCtx = unsafe extern "C" fn(*const TmHandle) -> u32;
type FnTokenize =
    unsafe extern "C" fn(*const TmHandle, *const c_char, c_int, c_int, *mut i32, i32) -> i32;
type FnTokenToPiece = unsafe extern "C" fn(*const TmHandle, i32, *mut c_char, i32) -> i32;
type FnDecode = unsafe extern "C" fn(*mut TmHandle, *const i32, i32) -> i32;
type FnReset = unsafe extern "C" fn(*mut TmHandle) -> i32;
type FnLogits = unsafe extern "C" fn(*mut TmHandle, *mut f32, i32) -> i32;
type FnSampleGreedy = unsafe extern "C" fn(*mut TmHandle, *mut i32) -> i32;
type FnModelDesc = unsafe extern "C" fn(*const TmHandle, *mut c_char, i32) -> i32;
type FnMeta = unsafe extern "C" fn(*const TmHandle, *const c_char, *mut c_char, i32) -> i32;
type FnModelSize = unsafe extern "C" fn(*const TmHandle) -> u64;

/// Resolved shim symbol table. The `Dl` keeps the library mapped.
struct Shim {
    _dl: Dl,
    version: FnShimVersion,
    load: FnLoad,
    free: FnFree,
    vocab_size: FnVocabSize,
    bos: FnBosEos,
    eos: FnBosEos,
    n_ctx: FnNCtx,
    tokenize: FnTokenize,
    token_to_piece: FnTokenToPiece,
    decode: FnDecode,
    reset: FnReset,
    kv_type: FnProbe,
    n_threads: FnProbe,
    logits: FnLogits,
    sample_greedy: FnSampleGreedy,
    model_desc: FnModelDesc,
    meta: FnMeta,
    model_size: FnModelSize,
}

struct Dl {
    handle: *mut c_void,
}

impl Drop for Dl {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return; // placeholder instance was never opened
        }
        // SAFETY: handle came from a successful dlopen and is closed once.
        unsafe {
            libc::dlclose(self.handle);
        }
    }
}

impl Shim {
    fn open(lib_path: &Path) -> Result<Shim> {
        let c_path = CString::new(lib_path.as_os_str().to_string_lossy().as_bytes().to_vec())
            .map_err(|_| {
                TuckmillError::new(ErrorCode::InvalidConfiguration, "library path contains NUL")
            })?;
        // SAFETY: dlopen on a caller-provided path; RTLD_NOW resolves all
        // symbols eagerly so a mismatched library fails at open, not mid-run.
        let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(TuckmillError::new(
                ErrorCode::Unavailable,
                format!("cannot load {}: {}", lib_path.display(), dlerror_string()),
            )
            .with_action("Run scripts/fetch-llama.sh and set TUCKMILL_LLAMA_LIB."));
        }
        let dl = Dl { handle };
        let sym = |name: &str| -> Result<*mut c_void> {
            let c_name = CString::new(name).expect("static symbol name");
            // SAFETY: symbol lookup on an open handle.
            let ptr = unsafe { libc::dlsym(dl.handle, c_name.as_ptr()) };
            if ptr.is_null() {
                return Err(TuckmillError::new(
                    ErrorCode::Unavailable,
                    format!(
                        "llama shim is missing symbol {name}; rebuild it with scripts/fetch-llama.sh (expected shim v{SHIM_VERSION})"
                    ),
                ));
            }
            Ok(ptr)
        };
        macro_rules! get {
            ($name:literal, $ty:ty) => {{
                let raw = sym($name)?;
                // SAFETY: symbol was resolved from the shim with this exact
                // declaration (kept in sync with tm_llama_shim.c).
                unsafe { std::mem::transmute::<*mut c_void, $ty>(raw) }
            }};
        }
        let shim = Shim {
            version: get!("tm_shim_version", FnShimVersion),
            load: get!("tm_load", FnLoad),
            free: get!("tm_free", FnFree),
            vocab_size: get!("tm_vocab_size", FnVocabSize),
            bos: get!("tm_bos", FnBosEos),
            eos: get!("tm_eos", FnBosEos),
            n_ctx: get!("tm_n_ctx", FnNCtx),
            tokenize: get!("tm_tokenize", FnTokenize),
            token_to_piece: get!("tm_token_to_piece", FnTokenToPiece),
            decode: get!("tm_decode", FnDecode),
            reset: get!("tm_reset", FnReset),
            kv_type: get!("tm_kv_type", FnProbe),
            n_threads: get!("tm_n_threads", FnProbe),
            logits: get!("tm_logits", FnLogits),
            sample_greedy: get!("tm_sample_greedy", FnSampleGreedy),
            model_desc: get!("tm_model_desc", FnModelDesc),
            meta: get!("tm_meta", FnMeta),
            model_size: get!("tm_model_size", FnModelSize),
            _dl: dl,
        };
        // SAFETY: shim version probe is side-effect free.
        let version = unsafe { (shim.version)() };
        if version != SHIM_VERSION {
            return Err(TuckmillError::new(
                ErrorCode::Unavailable,
                format!("llama shim version {version} is not supported (want {SHIM_VERSION})"),
            ));
        }
        Ok(shim)
    }
}

fn dlerror_string() -> String {
    // SAFETY: dlerror returns either NULL or a valid NUL-terminated string
    // owned by the loader.
    unsafe {
        let ptr = libc::dlerror();
        if ptr.is_null() {
            "unknown dlopen error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Resolve the shim library path: the `lib_path` load option, else
/// `TUCKMILL_LLAMA_LIB` (file or directory).
pub fn resolve_lib_path(option: Option<&str>) -> Result<PathBuf> {
    if let Some(value) = option {
        let path = PathBuf::from(value);
        return if path.is_dir() {
            Ok(path.join(LIB_FILE_NAME))
        } else {
            Ok(path)
        };
    }
    if let Ok(value) = std::env::var("TUCKMILL_LLAMA_LIB") {
        if !value.trim().is_empty() {
            let path = PathBuf::from(value);
            return if path.is_dir() {
                Ok(path.join(LIB_FILE_NAME))
            } else {
                Ok(path)
            };
        }
    }
    Err(TuckmillError::new(
        ErrorCode::Unavailable,
        "the llama.cpp reference backend is not configured (no lib_path option, TUCKMILL_LLAMA_LIB unset)",
    )
    .with_action(
        "Run scripts/fetch-llama.sh, then export TUCKMILL_LLAMA_LIB to the printed libtm_llama.dylib path.",
    ))
}

/// The llama.cpp reference tokenizer: a vocab-only handle over the same
/// pinned shim. Used as the authoritative encoder for artifacts whose
/// pre-tokenizer this build does not reproduce exactly (everything except
/// GPT-2-style splits), so token IDs always match the selected model's
/// intended tokenizer rather than an approximation.
pub struct ReferenceTokenizer {
    shim: Shim,
    handle: *mut TmHandle,
}

// SAFETY: same single-owner discipline as LlamaRef.
unsafe impl Send for ReferenceTokenizer {}

impl ReferenceTokenizer {
    /// Open a vocab-only handle for `model_path`. The library resolves from
    /// the `lib_path` option or `TUCKMILL_LLAMA_LIB`.
    pub fn open(lib_path: Option<&str>, model_path: &std::path::Path) -> Result<Self> {
        let lib = resolve_lib_path(lib_path)?;
        let shim = Shim::open(&lib)?;
        let c_path = CString::new(model_path.as_os_str().to_string_lossy().as_bytes().to_vec())
            .map_err(|_| {
                TuckmillError::new(ErrorCode::InvalidConfiguration, "path contains NUL")
            })?;
        let mut err_buf = vec![0u8; 512];
        // SAFETY: shim version-checked; path and error buffer valid;
        // vocab_only=1 skips context creation entirely.
        let handle = unsafe {
            (shim.load)(
                c_path.as_ptr(),
                0,
                0,
                0,
                0,
                /*vocab_only=*/ 1,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as c_int,
            )
        };
        if handle.is_null() {
            let len = err_buf
                .iter()
                .position(|b| *b == 0)
                .unwrap_or(err_buf.len());
            let msg = String::from_utf8_lossy(&err_buf[..len]).into_owned();
            return Err(TuckmillError::new(
                ErrorCode::InvalidArtifact,
                format!(
                    "llama.cpp could not load the tokenizer of {}: {msg}",
                    model_path.display()
                ),
            ));
        }
        Ok(Self { shim, handle })
    }

    pub fn vocab_len(&self) -> Result<usize> {
        // SAFETY: handle valid.
        let n = unsafe { (self.shim.vocab_size)(self.handle) };
        if n <= 0 {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                "llama.cpp reported no vocabulary",
            ));
        }
        Ok(n as usize)
    }

    pub fn eos(&self) -> i32 {
        // SAFETY: handle valid.
        unsafe { (self.shim.eos)(self.handle) }
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let c_text = CString::new(text)
            .map_err(|_| TuckmillError::new(ErrorCode::InvalidRequest, "text contains NUL"))?;
        let mut out = vec![0i32; 4096];
        // SAFETY: pointers valid for the calls; capacities match buffers.
        let n = unsafe {
            (self.shim.tokenize)(
                self.handle,
                c_text.as_ptr(),
                text.len() as c_int,
                add_special as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        };
        if n < 0 {
            out = vec![0i32; (-n) as usize];
            let n2 = unsafe {
                (self.shim.tokenize)(
                    self.handle,
                    c_text.as_ptr(),
                    text.len() as c_int,
                    add_special as c_int,
                    out.as_mut_ptr(),
                    out.len() as i32,
                )
            };
            if n2 < 0 {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "llama_tokenize failed twice",
                ));
            }
            out.truncate(n2 as usize);
        } else {
            out.truncate(n as usize);
        }
        Ok(out.into_iter().map(|v| v as u32).collect())
    }

    /// Exact piece bytes for a token id.
    pub fn piece_bytes(&self, id: u32) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; 1024];
        // SAFETY: buffer/capacity pair; returns required size when small.
        let n = unsafe {
            (self.shim.token_to_piece)(
                self.handle,
                id as i32,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
            )
        };
        if n < 0 {
            buf = vec![0u8; (-n) as usize];
            let n2 = unsafe {
                (self.shim.token_to_piece)(
                    self.handle,
                    id as i32,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                )
            };
            if n2 < 0 {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "llama_token_to_piece failed twice",
                ));
            }
            buf.truncate(n2 as usize);
        } else {
            buf.truncate(n as usize);
        }
        Ok(buf)
    }
}

impl Drop for ReferenceTokenizer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle came from tm_load and is freed once.
            unsafe { (self.shim.free)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Declared identity, available without loading the library.
pub fn identity() -> EngineIdentity {
    EngineIdentity {
        engine: BackendId::LlamaCpp.as_str().to_string(),
        certified: false,
        backend_abi_version: BACKEND_ABI_VERSION,
        worker_protocol_version: WORKER_PROTOCOL_VERSION,
        kernels: vec!["llama.cpp-cpu".to_string()],
        build_id: format!("tm-shim/{SHIM_VERSION} llama.cpp/{LLAMA_REVISION}"),
    }
}

/// The reference backend instance.
#[derive(Debug, Clone, serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct LlamaRefOptions {
    lib_path: Option<String>,
    /// "f16" (llama.cpp default) or "f32"; explicit so comparisons never
    /// leave cache precision uncontrolled.
    kv_precision: Option<String>,
    /// GPU offload layers for the Metal baseline (0 = CPU reference, the
    /// default; greater values are for performance comparison only, never
    /// for arithmetic diagnostics).
    gpu_layers: Option<i32>,
}

pub struct LlamaRef {
    shim: Shim,
    handle: *mut TmHandle,
    vocab: i32,
    eos: i32,
    ctx_tokens: u32,
    kv_precision: String,
    threads: u32,
    /// GPU offload layers in effect for the loaded model (0 = CPU only);
    /// reported in the resolved identity so a Metal baseline is never
    /// recorded as a CPU run.
    gpu_layers: i32,
    note: String,
}

// SAFETY: the handle is used by exactly one worker thread at a time (the
// supervisor serializes generations and the worker host is single-threaded
// per connection); the shim itself holds no hidden global mutable state
// beyond llama.cpp's internally synchronized backend registry.
unsafe impl Send for LlamaRef {}

impl LlamaRef {
    pub fn new() -> Self {
        Self {
            shim: Shim {
                _dl: Dl {
                    handle: std::ptr::null_mut(),
                },
                version: unreachable_fn,
                load: unreachable_load,
                free: unreachable_free,
                vocab_size: unreachable_vocab,
                bos: unreachable_bos,
                eos: unreachable_bos,
                n_ctx: unreachable_nctx,
                tokenize: unreachable_tokenize,
                token_to_piece: unreachable_piece,
                decode: unreachable_decode,
                reset: unreachable_reset,
                kv_type: unreachable_probe,
                n_threads: unreachable_probe,
                logits: unreachable_logits,
                sample_greedy: unreachable_sample,
                model_desc: unreachable_desc,
                meta: unreachable_meta,
                model_size: unreachable_size,
            },
            handle: std::ptr::null_mut(),
            vocab: 0,
            eos: -1,
            ctx_tokens: 0,
            kv_precision: "f16".to_string(),
            threads: 0,
            gpu_layers: 0,
            note: String::new(),
        }
    }

    /// Open a model directly for reference comparisons (teacher-forced
    /// logits during the Qwen3.5 Q35-1D verification). Mirrors the backend
    /// load with default options: pinned library, CPU, f16 or f32 KV.
    pub fn open_reference_model(
        &mut self,
        model_path: &std::path::Path,
        kv_f16: bool,
    ) -> Result<()> {
        if !self.handle.is_null() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "reference model is already loaded",
            ));
        }
        let lib_path = resolve_lib_path(None)?;
        let shim = Shim::open(&lib_path)?;
        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1) as c_int)
            .unwrap_or(4);
        let path_str = model_path.to_string_lossy().into_owned();
        let c_path = CString::new(path_str.clone().into_bytes()).map_err(|_| {
            TuckmillError::new(ErrorCode::InvalidConfiguration, "path contains NUL")
        })?;
        let mut err_buf = vec![0u8; 512];
        // SAFETY: shim was version-checked at open; path and buffers are valid.
        let handle = unsafe {
            (shim.load)(
                c_path.as_ptr(),
                4096,
                threads,
                0, // the reference comparison runs on the CPU
                kv_f16 as c_int,
                0,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as c_int,
            )
        };
        if handle.is_null() {
            let len = err_buf
                .iter()
                .position(|b| *b == 0)
                .unwrap_or(err_buf.len());
            let msg = String::from_utf8_lossy(&err_buf[..len]).into_owned();
            return Err(TuckmillError::new(
                ErrorCode::InvalidArtifact,
                format!("llama.cpp failed to load {path_str}: {msg}"),
            ));
        }
        // SAFETY: `handle` is checked non-null and owned by this instance
        // until Drop frees it; shim calls below only query or decode.
        let vocab = unsafe { (shim.vocab_size)(handle) };
        let eos = unsafe { (shim.eos)(handle) };
        let ctx_tokens = unsafe { (shim.n_ctx)(handle) };
        let actual_kv = unsafe { (shim.kv_type)(handle) };
        let actual_threads = unsafe { (shim.n_threads)(handle) };
        self.kv_precision = if actual_kv == 1 {
            "f16".to_string()
        } else if actual_kv == 0 {
            "f32".to_string()
        } else {
            "unreported".to_string()
        };
        self.threads = actual_threads.max(1) as u32;
        self.handle = handle;
        self.vocab = vocab;
        self.eos = eos;
        self.ctx_tokens = if ctx_tokens > 0 { ctx_tokens } else { 4096 };
        self.shim = shim;
        self.note = format!(
            "llama.cpp reference model for teacher-forced comparisons ({LLAMA_REVISION}, kv {})",
            self.kv_precision
        );
        Ok(())
    }

    /// Teacher-forced logits: reset the context, decode the exact token
    /// ids, and copy the last position's logits (full vocabulary). Used
    /// by the Qwen3.5 reference verification under identical token IDs
    /// and the declared precision.
    pub fn teacher_forced_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "teacher forcing needs at least one token id",
            ));
        }
        if tokens.len() > self.ctx_tokens as usize {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!(
                    "{} tokens exceed the reference context {}",
                    tokens.len(),
                    self.ctx_tokens
                ),
            ));
        }
        let h = self.h()?;
        // SAFETY: handle valid.
        let rc = unsafe { (self.shim.reset)(h) };
        if rc != 0 {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!("llama context reset failed (rc {rc})"),
            ));
        }
        self.decode_chunked(tokens)?;
        let mut logits = vec![0f32; self.vocab.max(0) as usize];
        // SAFETY: handle valid; buffer sized to the vocabulary.
        let n = unsafe { (self.shim.logits)(h, logits.as_mut_ptr(), logits.len() as i32) };
        if n < 0 {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!("llama logits unavailable (rc {n})"),
            ));
        }
        logits.truncate(n as usize);
        Ok(logits)
    }

    /// Beginning-of-sequence token id from llama.cpp (surface for callers).
    pub fn bos(&self) -> i32 {
        if self.handle.is_null() {
            return -1;
        }
        // SAFETY: handle valid.
        unsafe { (self.shim.bos)(self.handle) }
    }

    fn h(&self) -> Result<*mut TmHandle> {
        if self.handle.is_null() {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                "llama.cpp reference backend has no loaded model",
            ));
        }
        Ok(self.handle)
    }

    /// Tokenize text with llama.cpp itself (used for tokenizer goldens).
    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let h = self.h()?;
        let c_text = CString::new(text)
            .map_err(|_| TuckmillError::new(ErrorCode::InvalidRequest, "text contains NUL"))?;
        let mut out = vec![0i32; 4096];
        // SAFETY: pointers are valid for the calls; capacities match buffers.
        let n = unsafe {
            (self.shim.tokenize)(
                h,
                c_text.as_ptr(),
                text.len() as c_int,
                add_special as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        };
        if n < 0 {
            let needed = (-n) as usize;
            out = vec![0i32; needed];
            // SAFETY: retry with a buffer of the size llama.cpp requested.
            let n2 = unsafe {
                (self.shim.tokenize)(
                    h,
                    c_text.as_ptr(),
                    text.len() as c_int,
                    add_special as c_int,
                    out.as_mut_ptr(),
                    out.len() as i32,
                )
            };
            if n2 < 0 {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "llama_tokenize failed twice",
                ));
            }
            out.truncate(n2 as usize);
        } else {
            out.truncate(n as usize);
        }
        Ok(out.into_iter().map(|v| v as u32).collect())
    }

    /// Exact piece bytes for a token id. Multi-byte UTF-8 characters can be
    /// split across tokens, so exact text assembly must concatenate these
    /// bytes before UTF-8 validation (see `piece` for the lossy streaming
    /// convenience).
    pub fn piece_bytes(&self, id: u32) -> Result<Vec<u8>> {
        let h = self.h()?;
        let mut buf = vec![0u8; 1024];
        // SAFETY: buffer/capacity pair; returns required size when small.
        let n = unsafe {
            (self.shim.token_to_piece)(
                h,
                id as i32,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
            )
        };
        if n < 0 {
            buf = vec![0u8; (-n) as usize];
            // SAFETY: retry with a buffer of the size llama.cpp requested.
            let n2 = unsafe {
                (self.shim.token_to_piece)(
                    h,
                    id as i32,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                )
            };
            if n2 < 0 {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "llama_token_to_piece failed twice",
                ));
            }
            buf.truncate(n2 as usize);
        } else {
            buf.truncate(n as usize);
        }
        Ok(buf)
    }

    /// Lossy per-piece text (streaming convenience; a piece may end mid
    /// character, which renders as U+FFFD for that callback only).
    pub fn piece(&self, id: u32) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.piece_bytes(id)?).into_owned())
    }

    fn decode_chunked(&mut self, ids: &[u32]) -> Result<()> {
        self.decode_chunked_with_poll(ids, None)
    }

    /// Decode with an optional poll hook between chunks, so cancellation is
    /// consumed during long prefills instead of only at the next callback.
    fn decode_chunked_with_poll(&mut self, ids: &[u32], poll: Option<&dyn Fn()>) -> Result<()> {
        let h = self.h()?;
        for chunk in ids.chunks(256) {
            if let Some(poll) = poll {
                poll();
            }
            let c_chunk: Vec<i32> = chunk.iter().map(|v| *v as i32).collect();
            // SAFETY: ids buffer is valid for chunk length.
            let rc = unsafe { (self.shim.decode)(h, c_chunk.as_ptr(), c_chunk.len() as i32) };
            if rc != 0 {
                return Err(TuckmillError::new(
                    if rc == -2 {
                        ErrorCode::ContextExceedsBudget
                    } else {
                        ErrorCode::WorkerFailed
                    },
                    format!("llama_decode failed (rc {rc})"),
                ));
            }
        }
        Ok(())
    }
}

impl Default for LlamaRef {
    fn default() -> Self {
        Self::new()
    }
}

// Placeholder symbol table for a not-yet-opened instance; every function is
// replaced by `Shim::open` before any call (the null-handle guards in `h()`
// make calling them unreachable).
unsafe extern "C" fn unreachable_fn() -> c_int {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_load(
    _: *const c_char,
    _: c_int,
    _: c_int,
    _: c_int,
    _: c_int,
    _: c_int,
    _: *mut c_char,
    _: c_int,
) -> *mut TmHandle {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_free(_: *mut TmHandle) {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_vocab(_: *const TmHandle) -> i32 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_bos(_: *const TmHandle) -> i32 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_nctx(_: *const TmHandle) -> u32 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_tokenize(
    _: *const TmHandle,
    _: *const c_char,
    _: c_int,
    _: c_int,
    _: *mut i32,
    _: i32,
) -> i32 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_piece(_: *const TmHandle, _: i32, _: *mut c_char, _: i32) -> i32 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_decode(_: *mut TmHandle, _: *const i32, _: i32) -> i32 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_reset(_: *mut TmHandle) -> i32 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_logits(_: *mut TmHandle, _: *mut f32, _: i32) -> i32 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_sample(_: *mut TmHandle, _: *mut i32) -> i32 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_desc(_: *const TmHandle, _: *mut c_char, _: i32) -> i32 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_meta(
    _: *const TmHandle,
    _: *const c_char,
    _: *mut c_char,
    _: i32,
) -> i32 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_size(_: *const TmHandle) -> u64 {
    unreachable!("shim not open")
}
unsafe extern "C" fn unreachable_probe(_: *const TmHandle) -> i32 {
    unreachable!("shim not open")
}

/// Production-based timing for a generate request: time to the first
/// sampled token and the first-to-last sampled span (model-token
/// production, never visible-chunk release).
fn production_timing(
    t0: std::time::Instant,
    first: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
) -> tuckmill_backend::RequestTiming {
    let ms = |from: std::time::Instant, to: std::time::Instant| (to - from).as_secs_f64() * 1000.0;
    tuckmill_backend::RequestTiming {
        total_ms: ms(t0, std::time::Instant::now()),
        prefill_ms: None,
        first_token_ms: first.map(|f| ms(t0, f)),
        decode_ms: match (first, last) {
            (Some(f), Some(l)) if l > f => Some(ms(f, l)),
            _ => None,
        },
    }
}

impl InferenceBackend for LlamaRef {
    fn backend_id(&self) -> BackendId {
        BackendId::LlamaCpp
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            // Expanded as adapters are validated: the Qwen3 dense family,
            // and Qwen3.5 hybrid (qwen35), whose adapter was qualified
            // against this pinned revision (teacher-forced logits, Q35-1D).
            architectures: vec!["qwen3".to_string(), "qwen35".to_string()],
            moe: false,
            hybrid: true,
            ternary: false,
            requires_gpu: false,
            notes: vec![
                format!("pinned llama.cpp {LLAMA_REVISION} via tm shim v{SHIM_VERSION}; CPU execution by default (n_gpu_layers=0); --gpu-layers enables Metal as a performance baseline only"),
                "compatibility backend and numerical oracle; wrapper uncertified until comparisons are recorded".to_string(),
            ],
        }
    }

    fn identity(&self) -> EngineIdentity {
        let mut id = identity();
        if self.gpu_layers > 0 {
            id.kernels = vec![format!("llama.cpp:ngl={}:metal", self.gpu_layers)];
        }
        id
    }

    fn load(
        &mut self,
        plan: &ExecutionPlan,
        artifact: &ArtifactRef,
        options: Option<&serde_json::Value>,
    ) -> Result<String> {
        plan.validate()?;
        if plan.mode == ExecutionMode::Rejected {
            return Err(TuckmillError::new(
                ErrorCode::InvalidPlan,
                "refusing to load a rejected plan",
            ));
        }
        let opts: LlamaRefOptions = match options {
            Some(v) => serde_json::from_value(v.clone()).map_err(|e| {
                TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!("invalid llama backend options: {e}"),
                )
            })?,
            None => LlamaRefOptions::default(),
        };
        let kv_f16 = match opts.kv_precision.as_deref() {
            None | Some("f16") => true,
            Some("f32") => false,
            Some(other) => {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    format!("unknown kv_precision {other:?}; use \"f16\" or \"f32\""),
                ))
            }
        };
        let lib_path = resolve_lib_path(opts.lib_path.as_deref())?;
        let shim = Shim::open(&lib_path)?;

        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1) as c_int)
            .unwrap_or(4);
        let c_path = CString::new(artifact.path.as_bytes().to_vec()).map_err(|_| {
            TuckmillError::new(ErrorCode::InvalidConfiguration, "path contains NUL")
        })?;
        let mut err_buf = vec![0u8; 512];
        // SAFETY: shim was version-checked; path and error buffer are valid.
        let gpu_layers = opts.gpu_layers.unwrap_or(0).max(0);
        let handle = unsafe {
            (shim.load)(
                c_path.as_ptr(),
                plan.context.total_context as c_int,
                threads,
                gpu_layers,
                kv_f16 as c_int,
                0, // full context
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as c_int,
            )
        };
        if handle.is_null() {
            let len = err_buf
                .iter()
                .position(|b| *b == 0)
                .unwrap_or(err_buf.len());
            let msg = String::from_utf8_lossy(&err_buf[..len]).into_owned();
            return Err(TuckmillError::new(
                ErrorCode::InvalidArtifact,
                format!("llama.cpp failed to load {}: {msg}", artifact.path),
            ));
        }
        // SAFETY (all shim calls below): `handle` was checked non-null and is
        // owned by this instance; the shim was version-verified at open; every
        // buffer/capacity pair is valid for the call.
        let vocab = unsafe { (shim.vocab_size)(handle) };
        let eos = unsafe { (shim.eos)(handle) };
        let ctx_tokens = unsafe { (shim.n_ctx)(handle) };
        let mut desc_buf = vec![0u8; 256];
        // SAFETY: handle valid; buffer sized 256 with cap passed as 256.
        let desc_len =
            unsafe { (shim.model_desc)(handle, desc_buf.as_mut_ptr() as *mut c_char, 256) };
        let desc = if desc_len > 0 {
            String::from_utf8_lossy(&desc_buf[..(desc_len as usize).min(255)]).into_owned()
        } else {
            "unknown".into()
        };
        // SAFETY: handle valid; read-only metadata query.
        let model_bytes = unsafe { (shim.model_size)(handle) };
        let mut arch_buf = vec![0u8; 64];
        let arch_key = CString::new("general.architecture").expect("static");
        // SAFETY: handle valid; key is a stored CString; buffer sized/capped.
        let arch_len = unsafe {
            (shim.meta)(
                handle,
                arch_key.as_ptr(),
                arch_buf.as_mut_ptr() as *mut c_char,
                63,
            )
        };
        let arch = if arch_len > 0 {
            String::from_utf8_lossy(&arch_buf[..(arch_len as usize).min(63)]).into_owned()
        } else {
            "unknown".into()
        };

        let actual_kv = unsafe { (shim.kv_type)(handle) };
        let actual_threads = unsafe { (shim.n_threads)(handle) };
        self.kv_precision = if actual_kv == 1 {
            "f16".to_string()
        } else if actual_kv == 0 {
            "f32".to_string()
        } else {
            "unreported".to_string()
        };
        self.gpu_layers = gpu_layers;
        self.threads = actual_threads.max(1) as u32;
        self.handle = handle;
        self.vocab = vocab;
        self.eos = eos;
        self.ctx_tokens = if ctx_tokens > 0 {
            ctx_tokens
        } else {
            plan.context.total_context as u32
        };
        self.shim = shim;
        self.note = format!(
            "llama.cpp reference backend ({LLAMA_REVISION}, gpu_layers {gpu_layers}): {desc} [arch {arch}], {vocab} vocab, {} context, kv {}, {} threads, {} model bytes; wrapper uncertified",
            self.ctx_tokens,
            self.kv_precision,
            self.threads,
            tuckmill_manifest::units::Bytes(model_bytes)
        );
        Ok(self.note.clone())
    }

    fn generate(&mut self, req: &GenerateRequest, ctl: &mut GenControl<'_>) -> Result<GenOutcome> {
        req.sampling.validate()?;
        let prompt_len = req.prompt_tokens.len() as u64;
        if prompt_len == 0 {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "prompt must contain at least one token",
            ));
        }
        if req.max_output_tokens == 0 {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                "max_output_tokens must be positive",
            ));
        }
        if prompt_len + req.max_output_tokens > self.ctx_tokens as u64 {
            return Err(TuckmillError::new(
                ErrorCode::ContextExceedsBudget,
                format!(
                    "prompt {prompt_len} + max output {} exceeds the context {}",
                    req.max_output_tokens, self.ctx_tokens
                ),
            ));
        }
        (ctl.on_state)(RunState::Warming);
        // A fresh sequence per request: the loaded model is reused, session
        // state never is. Without this, a second request would append to the
        // first request's positions (leaked context).
        {
            let h = self.h()?;
            // SAFETY: handle valid; clears KV memory and samples state.
            let rc = unsafe { (self.shim.reset)(h) };
            if rc != 0 {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "llama context reset failed before prefill",
                ));
            }
        }
        (ctl.on_state)(RunState::Prefilling);
        self.decode_chunked_with_poll(&req.prompt_tokens, ctl.poll)?;
        if ctl.cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(GenOutcome::new(FinishReason::Cancelled, 0));
        }

        (ctl.on_state)(RunState::Generating);
        let t0 = Instant::now();
        let mut first_generated_at: Option<Instant> = None;
        let mut last_generated_at: Option<Instant> = None;
        let mut rng = tuckmill_backend::sampling::Rng::new(req.seed);
        let sampling_cfg = tuckmill_backend::sampling::SamplingConfig {
            temperature: req.sampling.temperature,
            top_k: None,
            top_p: req.sampling.top_p,
        };
        let mut buffer = tuckmill_backend::output::VisibleOutputBuffer::new(&req.stop);
        let mut release_index: u64 = 0;
        let mut generated_ids: u64 = 0;
        // Incremental UTF-8 assembly with byte-level token provenance.
        let mut decoder = tuckmill_manifest::utf8::Utf8StreamDecoder::new();
        let emit = |text: &str, release_index: &mut u64, ctl: &mut GenControl<'_>| -> bool {
            if text.is_empty() {
                return true;
            }
            let keep = (ctl.on_token)(*release_index, text);
            *release_index += 1;
            keep
        };
        for i in 0..req.max_output_tokens {
            if ctl.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                let r = buffer.flush();
                let _ = emit(&r.text, &mut release_index, ctl);
                return Ok(
                    GenOutcome::new(FinishReason::Cancelled, buffer.visible_tokens())
                        .with_generated(generated_ids)
                        .with_timing(production_timing(t0, first_generated_at, last_generated_at)),
                );
            }
            let h = self.h()?;
            let token: u32 = if req.sampling.is_greedy() {
                let mut sampled: i32 = -1;
                // SAFETY: handle valid; out pointer valid.
                let rc = unsafe { (self.shim.sample_greedy)(h, &mut sampled) };
                if rc != 0 || sampled < 0 {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!("llama greedy sampling failed (rc {rc})"),
                    ));
                }
                sampled as u32
            } else {
                // Nonzero temperature: the shared sampler applies the
                // declared transforms to llama.cpp's own logits.
                let mut logits = vec![0f32; self.vocab.max(0) as usize];
                // SAFETY: logits buffer sized to the vocabulary.
                let n = unsafe { (self.shim.logits)(h, logits.as_mut_ptr(), logits.len() as i32) };
                if n < 0 {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!("llama logits unavailable (rc {n})"),
                    ));
                }
                logits.truncate(n as usize);
                let Some(t) = tuckmill_backend::sampling::sample(&logits, &sampling_cfg, &mut rng)
                else {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        "sampling produced no token: every logit was non-finite",
                    ));
                };
                t
            };
            generated_ids = i + 1;
            let produced_at = Instant::now();
            if first_generated_at.is_none() {
                first_generated_at = Some(produced_at);
            }
            last_generated_at = Some(produced_at);
            if token == self.eos.max(0) as u32 {
                // The model-defined EOS marker is never visible.
                let r = buffer.flush();
                let _ = emit(&r.text, &mut release_index, ctl);
                return Ok(GenOutcome::new(FinishReason::Eos, buffer.visible_tokens())
                    .with_generated(generated_ids)
                    .with_timing(production_timing(t0, first_generated_at, last_generated_at)));
            }
            let piece = decoder.push_token(&self.piece_bytes(token)?, i);
            let mut client_stop = false;
            let mut stop = false;
            for run in piece {
                let release = buffer.push(&run.owners, &run.text);
                if !emit(&release.text, &mut release_index, ctl) {
                    client_stop = true;
                    break;
                }
                if release.stop {
                    stop = true;
                    break;
                }
            }
            if client_stop || stop {
                let r = buffer.flush();
                let _ = emit(&r.text, &mut release_index, ctl);
                return Ok(GenOutcome::new(FinishReason::Stop, buffer.visible_tokens())
                    .with_generated(generated_ids)
                    .with_timing(production_timing(t0, first_generated_at, last_generated_at)));
            }
            if i + 1 < req.max_output_tokens {
                self.decode_chunked_with_poll(&[token], ctl.poll)?;
            }
        }
        let r = buffer.flush();
        let _ = emit(&r.text, &mut release_index, ctl);
        Ok(
            GenOutcome::new(FinishReason::Length, buffer.visible_tokens())
                .with_generated(generated_ids)
                .with_timing(production_timing(t0, first_generated_at, last_generated_at)),
        )
    }

    fn score(
        &mut self,
        tokens: &[u32],
        positions: &[u64],
        cancelled: &dyn Fn() -> bool,
        sink: &mut dyn FnMut(u64, &[f32]) -> bool,
    ) -> Result<u64> {
        let mut scored = 0u64;
        for position in positions {
            if cancelled() {
                return Err(TuckmillError::new(
                    ErrorCode::Cancelled,
                    "scoring cancelled at a position boundary",
                ));
            }
            let pos = *position as usize;
            if pos >= tokens.len() {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    "scoring position outside the token sequence",
                ));
            }
            let h = self.h()?;
            // SAFETY: handle valid.
            let rc = unsafe { (self.shim.reset)(h) };
            if rc != 0 {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    "llama_reset failed",
                ));
            }
            self.decode_chunked(&tokens[..=pos])?;
            let mut logits = vec![0f32; self.vocab.max(0) as usize];
            // SAFETY: logits buffer sized to the vocabulary.
            let n = unsafe { (self.shim.logits)(h, logits.as_mut_ptr(), logits.len() as i32) };
            if n < 0 {
                return Err(TuckmillError::new(
                    ErrorCode::WorkerFailed,
                    format!("llama logits unavailable (rc {n})"),
                ));
            }
            logits.truncate(n as usize);
            scored += 1;
            if !sink(*position, &logits) {
                break;
            }
        }
        Ok(scored)
    }

    fn stats(&self) -> BackendStats {
        // Weight reads go through mmap: physical reads are unobservable here
        // (recorded as 0 with no claim). KV precision and threads are actual.
        BackendStats {
            kv_precision: Some(self.kv_precision.clone()),
            threads: Some(self.threads),
            ..Default::default()
        }
    }

    fn shutdown(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle came from tm_load and is freed exactly once.
            unsafe { (self.shim.free)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
        self.gpu_layers = 0;
    }
}

impl Drop for LlamaRef {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

    fn lib_available() -> Option<PathBuf> {
        resolve_lib_path(None).ok().filter(|p| p.exists())
    }

    fn fixture() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.gguf");
        write_synthetic(&path, &SyntheticConfig::tiny_qwen3()).unwrap();
        (dir, path)
    }

    fn plan_for(ctx: u64) -> ExecutionPlan {
        let mut p = ExecutionPlan::new_skeleton(
            tuckmill_manifest::Id::new("a".repeat(64)).unwrap(),
            tuckmill_manifest::Id::new("b".repeat(64)).unwrap(),
            "qwen3".into(),
        );
        p.mode = ExecutionMode::Resident;
        p.context = tuckmill_manifest::plan::PlanContext::new(ctx, 64);
        p.plan_id = p.compute_id().unwrap();
        p
    }

    #[test]
    fn lib_resolution_reports_honestly_when_unset() {
        // Only meaningful when the environment genuinely lacks the library.
        if std::env::var("TUCKMILL_LLAMA_LIB").is_ok() {
            return;
        }
        let err = resolve_lib_path(None).unwrap_err();
        assert_eq!(err.code(), ErrorCode::Unavailable);
        assert!(err.action().contains("fetch-llama.sh"));
    }

    #[test]
    fn shim_missing_symbols_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("libtm_llama.dylib");
        std::fs::write(&path, b"not a dylib").unwrap();
        let err = match Shim::open(&path) {
            Err(e) => e,
            Ok(_) => panic!("opening a non-library must fail"),
        };
        assert_eq!(err.code(), ErrorCode::Unavailable);
    }

    #[test]
    fn tokenizer_golden_matches_llama_cpp_when_available() {
        let Some(_lib) = lib_available() else {
            eprintln!("skipping: TUCKMILL_LLAMA_LIB not set (run scripts/fetch-llama.sh)");
            return;
        };
        let (_dir, fixture) = fixture();
        let insp =
            tuckmill_manifest::gguf::inspect(&fixture, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let ours = tuckmill_tuckcore::tokenizer::Tokenizer::from_gguf(&insp).unwrap();

        let mut backend = LlamaRef::new();
        let artifact = ArtifactRef {
            path: fixture.display().to_string(),
            file_bytes: insp.file_len,
            metadata_hash: insp.metadata_hash().unwrap().as_str().to_string(),
            tensor_index_hash: insp.tensor_index_hash().unwrap().as_str().to_string(),
            tokenizer_hash: None,
            arch_id: Some("qwen3".into()),
            content_hash: None,
        };
        backend.load(&plan_for(512), &artifact, None).unwrap();

        for text in [
            "hello world",
            "Hello, World! 123",
            "don't stop",
            "multi   spaces",
            "unicode: héllo wörld",
        ] {
            let theirs = backend.tokenize(text, false).unwrap();
            let ours_ids = ours.encode(text, false).unwrap();
            assert_eq!(
                ours_ids, theirs,
                "tokenizer divergence for {text:?}: ours {ours_ids:?} vs llama.cpp {theirs:?}"
            );
            // And full decode roundtrip through the reference backend
            // (bytes assembled exactly, validated once as UTF-8).
            let mut bytes = Vec::new();
            for id in &theirs {
                bytes.extend_from_slice(&backend.piece_bytes(*id).unwrap());
            }
            assert_eq!(
                String::from_utf8(bytes).unwrap(),
                text,
                "llama.cpp piece decode roundtrip for {text:?}"
            );
        }
        backend.shutdown();
    }

    #[test]
    fn real_model_reference_tokenizer_goldens_when_available() {
        let (Ok(lib), Ok(real)) = (
            std::env::var("TUCKMILL_LLAMA_LIB"),
            std::env::var("TUCKMILL_REAL_GGUF"),
        ) else {
            eprintln!("skipping: set TUCKMILL_LLAMA_LIB and TUCKMILL_REAL_GGUF");
            return;
        };
        if !std::path::Path::new(&real).exists() {
            eprintln!("skipping: TUCKMILL_REAL_GGUF does not exist");
            return;
        }
        let _ = lib;
        let tokenizer =
            ReferenceTokenizer::open(None, std::path::Path::new(&real)).expect("vocab-only load");
        assert!(
            tokenizer.vocab_len().unwrap() > 100_000,
            "real-quant vocabulary"
        );
        for text in [
            "The capital of France is",
            "123 456 7890",
            "don't stop believing",
            "multi   spaces\tand\nnewlines",
            "unicode: héllo wörld 日本語 😀",
        ] {
            let ids = tokenizer.tokenize(text, false).unwrap();
            assert!(!ids.is_empty(), "{text:?}");
            // Exact byte assembly round-trips through the same reference.
            let mut bytes = Vec::new();
            for id in &ids {
                bytes.extend_from_slice(&tokenizer.piece_bytes(*id).unwrap());
            }
            assert_eq!(
                String::from_utf8(bytes).unwrap(),
                text,
                "reference piece roundtrip for {text:?}"
            );
            // Deterministic re-encode.
            assert_eq!(tokenizer.tokenize(text, false).unwrap(), ids);
        }
        // And the built-in tokenizer refuses non-GPT-2 pre-tokenizers rather
        // than approximating them (same-profile honesty).
        let insp = tuckmill_manifest::gguf::inspect(
            std::path::Path::new(&real),
            &tuckmill_manifest::gguf::Limits::default(),
        )
        .unwrap();
        let native = tuckmill_tuckcore::tokenizer::Tokenizer::from_gguf(&insp).unwrap();
        if !native.encode_is_exact() {
            let err = native.encode("test", false).unwrap_err();
            assert_eq!(
                err.code(),
                tuckmill_manifest::ErrorCode::UnsupportedArchitecture
            );
            assert!(err.action().contains("TUCKMILL_LLAMA_LIB"));
        }
    }

    #[test]
    fn reference_backend_scores_and_generates_when_available() {
        let Some(_lib) = lib_available() else {
            eprintln!("skipping: TUCKMILL_LLAMA_LIB not set (run scripts/fetch-llama.sh)");
            return;
        };
        let (_dir, fixture) = fixture();
        let insp =
            tuckmill_manifest::gguf::inspect(&fixture, &tuckmill_manifest::gguf::Limits::default())
                .unwrap();
        let artifact = ArtifactRef {
            path: fixture.display().to_string(),
            file_bytes: insp.file_len,
            metadata_hash: insp.metadata_hash().unwrap().as_str().to_string(),
            tensor_index_hash: insp.tensor_index_hash().unwrap().as_str().to_string(),
            tokenizer_hash: None,
            arch_id: Some("qwen3".into()),
            content_hash: None,
        };
        let mut backend = LlamaRef::new();
        let note = backend.load(&plan_for(512), &artifact, None).unwrap();
        assert!(note.contains("llama.cpp"));
        assert!(note.contains("uncertified"));

        // Scoring is deterministic and vocab-sized.
        let tokens = [259u32, 260, 111];
        let mut a = Vec::new();
        backend
            .score(&tokens, &[2], &|| false, &mut |_, l| {
                a.extend_from_slice(l);
                true
            })
            .unwrap();
        assert_eq!(a.len(), 512);
        let mut b = Vec::new();
        backend
            .score(&tokens, &[2], &|| false, &mut |_, l| {
                b.extend_from_slice(l);
                true
            })
            .unwrap();
        assert_eq!(a, b, "llama.cpp scoring repeats exactly");

        // Greedy generation is deterministic.
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pause = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut texts = Vec::new();
        let mut on_token = |_i: u64, t: &str| {
            texts.push(t.to_string());
            true
        };
        let mut on_state = |_s: RunState| {};
        let mut ctl = GenControl::new(cancel.clone(), pause.clone(), &mut on_token, &mut on_state);
        let req = GenerateRequest {
            request_id: "ref-1".into(),
            prompt_tokens: tokens.to_vec(),
            max_output_tokens: 4,
            stop: vec![],
            seed: 0,
            sampling: tuckmill_backend::SamplingPolicy::default(),
        };
        let outcome = backend.generate(&req, &mut ctl).unwrap();
        assert_eq!(outcome.finish_reason, FinishReason::Length);
        assert_eq!(outcome.committed, 4);
        assert!(!texts.is_empty());

        // A second equivalent request reuses the loaded model but must not
        // inherit any session state: the reset makes greedy output identical
        // to the first run instead of appending to it.
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pause = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut texts2 = Vec::new();
        let mut on_token2 = |_i: u64, t: &str| {
            texts2.push(t.to_string());
            true
        };
        let mut on_state2 = |_s: RunState| {};
        let mut ctl2 = GenControl::new(
            cancel.clone(),
            pause.clone(),
            &mut on_token2,
            &mut on_state2,
        );
        let outcome2 = backend.generate(&req, &mut ctl2).unwrap();
        assert_eq!(outcome2.finish_reason, FinishReason::Length);
        assert_eq!(texts2, texts, "no session state may leak between requests");

        // Nonzero temperature: the shared sampler applies the declared
        // transforms to llama.cpp's own logits, reproducibly per seed.
        let mut sampled = |seed: u64| -> Vec<String> {
            let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let pause = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let mut out = Vec::new();
            let mut on_token = |_i: u64, t: &str| {
                out.push(t.to_string());
                true
            };
            let mut on_state = |_s: RunState| {};
            let mut ctl =
                GenControl::new(cancel.clone(), pause.clone(), &mut on_token, &mut on_state);
            let mut req = req.clone();
            req.max_output_tokens = 8;
            req.seed = seed;
            req.sampling = tuckmill_backend::SamplingPolicy {
                temperature: 1.0,
                top_p: Some(0.9),
            };
            let outcome = backend.generate(&req, &mut ctl).unwrap();
            assert_eq!(outcome.finish_reason, FinishReason::Length);
            assert!(
                outcome.committed <= outcome.generated,
                "visible tokens can never exceed sampled tokens"
            );
            out
        };
        let s1 = sampled(7);
        let s2 = sampled(7);
        assert_eq!(s1, s2, "seeded sampling through llama.cpp is reproducible");
        let s3 = sampled(8);
        assert_ne!(s1, s3, "different seeds must diverge (8 sampled tokens)");
        backend.shutdown();
    }

    #[test]
    fn identity_is_never_certified() {
        let id = identity();
        assert!(!id.certified);
        assert_eq!(id.engine, "llama.cpp");
    }

    #[test]
    fn resolved_identity_names_gpu_offload() {
        // Regression: Metal baselines (gpu_layers > 0) were recorded with
        // the CPU identity.
        let mut backend = LlamaRef::new();
        assert_eq!(
            InferenceBackend::identity(&backend).kernels,
            vec!["llama.cpp-cpu".to_string()]
        );
        backend.gpu_layers = 99;
        assert_eq!(
            InferenceBackend::identity(&backend).kernels,
            vec!["llama.cpp:ngl=99:metal".to_string()]
        );
        backend.shutdown();
        assert_eq!(
            InferenceBackend::identity(&backend).kernels,
            vec!["llama.cpp-cpu".to_string()]
        );
    }
}
