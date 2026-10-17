//! TuckCore Metal path: typed Rust handles over the narrow TuckCore–Metal C
//! ABI (ADR-007).
//!
//! One `MetalContext` owns one device, one command queue, and the compiled
//! pipeline states for the operations implemented so far (Q8_0 packed
//! matrix-vector decode plus F16/F32 utility operations). Buffers are RAII
//! handles: they hold the context alive, release through the ABI on drop,
//! and the context reports live allocated bytes and dispatch completion
//! counts so callers account for real GPU resources instead of assuming
//! them.
//!
//! The MSL kernels are compiled from `native/metal/kernels.metal` at this
//! exact repository revision; correctness is validated against the CPU
//! scalar reference (tolerance-registered — GPU reduction order differs
//! from the CPU sequence) before any tuning.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Arc, Mutex};

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};

/// The MSL source shipped with this revision, compiled at context creation.
pub const KERNELS_MSL: &str = include_str!("../../../native/metal/kernels.metal");

const ERR_CAP: usize = 512;

#[cfg(target_os = "macos")]
mod imp {
    use super::*;

    #[repr(C)]
    pub struct TmMetalCtx {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct TmMetalBuffer {
        _private: [u8; 0],
    }

    pub const TM_METAL_ERR_NO_DEVICE: c_int = -1;
    pub const TM_METAL_ERR_SOURCE: c_int = -2;
    pub const TM_METAL_ERR_ALLOC: c_int = -3;
    pub const TM_METAL_ERR_BOUNDS: c_int = -4;
    pub const TM_METAL_ERR_UNSUPPORTED: c_int = -5;
    pub const TM_METAL_ERR_INTERNAL: c_int = -6;

    extern "C" {
        pub fn tm_metal_available() -> c_int;
        pub fn tm_metal_runtime_info() -> *const c_char;
        pub fn tm_metal_create(
            msl_source: *const c_char,
            err: *mut c_char,
            err_cap: usize,
        ) -> *mut TmMetalCtx;
        pub fn tm_metal_destroy(ctx: *mut TmMetalCtx);
        pub fn tm_metal_buffer_new(
            ctx: *mut TmMetalCtx,
            bytes: usize,
            shared: c_int,
            err: *mut c_char,
            err_cap: usize,
        ) -> *mut TmMetalBuffer;
        pub fn tm_metal_buffer_free(ctx: *mut TmMetalCtx, buffer: *mut TmMetalBuffer);
        pub fn tm_metal_buffer_len(ctx: *const TmMetalCtx, buffer: *const TmMetalBuffer) -> usize;
        pub fn tm_metal_upload(
            ctx: *mut TmMetalCtx,
            dst: *mut TmMetalBuffer,
            off: usize,
            src: *const c_void,
            len: usize,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_download(
            ctx: *mut TmMetalCtx,
            src: *mut TmMetalBuffer,
            off: usize,
            dst: *mut c_void,
            len: usize,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_f32_scale(
            ctx: *mut TmMetalCtx,
            x: *mut TmMetalBuffer,
            y: *mut TmMetalBuffer,
            factor: f32,
            n: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_f32_axpy(
            ctx: *mut TmMetalCtx,
            a: *mut TmMetalBuffer,
            y: *mut TmMetalBuffer,
            alpha: f32,
            n: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_cast_f16_f32(
            ctx: *mut TmMetalCtx,
            x: *mut TmMetalBuffer,
            y: *mut TmMetalBuffer,
            n: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_cast_f32_f16(
            ctx: *mut TmMetalCtx,
            x: *mut TmMetalBuffer,
            y: *mut TmMetalBuffer,
            n: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_q8_0_gemv(
            ctx: *mut TmMetalCtx,
            w: *mut TmMetalBuffer,
            x: *mut TmMetalBuffer,
            y: *mut TmMetalBuffer,
            rows: u32,
            cols: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_fence(ctx: *mut TmMetalCtx, err: *mut c_char, err_cap: usize) -> c_int;
        pub fn tm_metal_begin_group(
            ctx: *mut TmMetalCtx,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_end_group(ctx: *mut TmMetalCtx, err: *mut c_char, err_cap: usize) -> c_int;
        pub fn tm_metal_abort_group(
            ctx: *mut TmMetalCtx,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_q8_0_row_f32(
            ctx: *mut TmMetalCtx,
            w: *mut TmMetalBuffer,
            out: *mut TmMetalBuffer,
            row: u32,
            cols: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_f16_gemv(
            ctx: *mut TmMetalCtx,
            w: *mut TmMetalBuffer,
            x: *mut TmMetalBuffer,
            y: *mut TmMetalBuffer,
            rows: u32,
            cols: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_rmsnorm_f32(
            ctx: *mut TmMetalCtx,
            x: *mut TmMetalBuffer,
            weight: *mut TmMetalBuffer,
            y: *mut TmMetalBuffer,
            rows: u32,
            cols: u32,
            eps: f32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_rope_f32(
            ctx: *mut TmMetalCtx,
            q: *mut TmMetalBuffer,
            k: *mut TmMetalBuffer,
            inv_freq: *mut TmMetalBuffer,
            pos: u32,
            n_q_heads: u32,
            n_kv_heads: u32,
            head_dim: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_kv_store_f32(
            ctx: *mut TmMetalCtx,
            k_src: *mut TmMetalBuffer,
            v_src: *mut TmMetalBuffer,
            k_cache: *mut TmMetalBuffer,
            v_cache: *mut TmMetalBuffer,
            kv_heads: u32,
            pos: u32,
            cap: u32,
            head_dim: u32,
            round_f16: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_attention_f32(
            ctx: *mut TmMetalCtx,
            q: *mut TmMetalBuffer,
            k_cache: *mut TmMetalBuffer,
            v_cache: *mut TmMetalBuffer,
            out: *mut TmMetalBuffer,
            scratch: *mut TmMetalBuffer,
            heads: u32,
            pos: u32,
            cap: u32,
            head_dim: u32,
            group: u32,
            scale: f32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_f16_row_f32(
            ctx: *mut TmMetalCtx,
            w: *mut TmMetalBuffer,
            out: *mut TmMetalBuffer,
            row: u32,
            cols: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_rmsnorm_f16w(
            ctx: *mut TmMetalCtx,
            x: *mut TmMetalBuffer,
            weight: *mut TmMetalBuffer,
            y: *mut TmMetalBuffer,
            rows: u32,
            cols: u32,
            eps: f32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_silu_mul_f32(
            ctx: *mut TmMetalCtx,
            gate: *mut TmMetalBuffer,
            up: *mut TmMetalBuffer,
            out: *mut TmMetalBuffer,
            n: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn tm_metal_committed_dispatches(ctx: *const TmMetalCtx) -> u64;
        pub fn tm_metal_completed_dispatches(ctx: *const TmMetalCtx) -> u64;
        pub fn tm_metal_last_gpu_ms(ctx: *const TmMetalCtx) -> f64;
        pub fn tm_metal_allocated_bytes(ctx: *const TmMetalCtx) -> u64;
    }
}

/// Cheap availability probe: is a Metal device present on this machine?
pub fn probe() -> Probe {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: both calls are side-effect free; the info pointer is a
        // static string owned by the C layer.
        let available = unsafe { imp::tm_metal_available() } == 1;
        let info = unsafe { CStr::from_ptr(imp::tm_metal_runtime_info()) }
            .to_string_lossy()
            .into_owned();
        Probe { available, info }
    }
    #[cfg(not(target_os = "macos"))]
    {
        Probe {
            available: false,
            info: "Metal is only available on macOS".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Probe {
    pub available: bool,
    pub info: String,
}

struct ContextInner {
    #[cfg(target_os = "macos")]
    raw: *mut imp::TmMetalCtx,
    /// Serializes dispatches: the ABI's `pending` bookkeeping is not
    /// concurrent; the Metal objects themselves are thread-safe.
    lock: Mutex<()>,
}

// SAFETY: the raw context is only touched under `lock` (buffer creation and
// drops take the same lock), and the C layer's completion counters are
// atomics. Ownership can move between threads; concurrent dispatch is
// serialized by the mutex rather than relying on queue thread-safety alone.
unsafe impl Send for ContextInner {}
unsafe impl Sync for ContextInner {}

impl Drop for ContextInner {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        unsafe {
            // SAFETY: created by tm_metal_create; all buffers hold an Arc to
            // this inner and have been dropped before the last reference.
            imp::tm_metal_destroy(self.raw);
        }
    }
}

fn err_text(buf: &[c_char]) -> String {
    let text = unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    if text.is_empty() {
        "unknown error".to_string()
    } else {
        text
    }
}

fn metal_error(code: c_int, text: String, context: &str) -> TuckmillError {
    #[cfg(target_os = "macos")]
    let mapped = match code {
        imp::TM_METAL_ERR_NO_DEVICE => ErrorCode::Unavailable,
        imp::TM_METAL_ERR_ALLOC => ErrorCode::Unavailable,
        imp::TM_METAL_ERR_BOUNDS | imp::TM_METAL_ERR_UNSUPPORTED => ErrorCode::InvalidRequest,
        imp::TM_METAL_ERR_SOURCE => ErrorCode::Internal,
        imp::TM_METAL_ERR_INTERNAL => ErrorCode::Internal,
        _ => ErrorCode::Internal,
    };
    #[cfg(not(target_os = "macos"))]
    let (mapped, _) = (ErrorCode::Unavailable, code);
    TuckmillError::new(mapped, format!("Metal {context}: {text}"))
}

/// One device, one command queue, and the compiled pipeline states.
#[derive(Clone)]
pub struct MetalContext {
    inner: Arc<ContextInner>,
}

impl MetalContext {
    /// Open a context with the repository's MSL kernels.
    pub fn open() -> Result<Self> {
        Self::open_with_source(KERNELS_MSL)
    }

    /// Open a context compiling exactly `msl_source` (tests use this to
    /// exercise the failure path; production always uses [`KERNELS_MSL`]).
    pub fn open_with_source(msl_source: &str) -> Result<Self> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = msl_source;
            return Err(TuckmillError::new(
                ErrorCode::Unavailable,
                "Metal contexts are only available on macOS",
            ));
        }
        #[cfg(target_os = "macos")]
        {
            let source = CString::new(msl_source).map_err(|_| {
                TuckmillError::new(ErrorCode::InvalidRequest, "MSL source contains a NUL byte")
            })?;
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: source is a valid NUL-terminated string; err points to
            // a writable buffer of ERR_CAP bytes.
            let raw = unsafe { imp::tm_metal_create(source.as_ptr(), err.as_mut_ptr(), ERR_CAP) };
            if raw.is_null() {
                let mut text = err_text(&err);
                if !probe().available {
                    text = format!("no Metal device available ({text})");
                }
                return Err(
                    metal_error(imp::TM_METAL_ERR_NO_DEVICE, text, "create").with_action(
                        "Ensure the machine has a Metal-capable GPU and the process may use it.",
                    ),
                );
            }
            Ok(Self {
                inner: Arc::new(ContextInner {
                    raw,
                    lock: Mutex::new(()),
                }),
            })
        }
    }

    fn guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.inner.lock.lock().expect("metal context lock")
    }

    /// Live bytes currently allocated in buffers from this context.
    pub fn allocated_bytes(&self) -> u64 {
        #[cfg(target_os = "macos")]
        unsafe {
            // SAFETY: raw is valid while the inner lives.
            imp::tm_metal_allocated_bytes(self.inner.raw)
        }
        #[cfg(not(target_os = "macos"))]
        {
            0
        }
    }

    pub fn committed_dispatches(&self) -> u64 {
        #[cfg(target_os = "macos")]
        unsafe {
            imp::tm_metal_committed_dispatches(self.inner.raw)
        }
        #[cfg(not(target_os = "macos"))]
        {
            0
        }
    }

    pub fn completed_dispatches(&self) -> u64 {
        #[cfg(target_os = "macos")]
        unsafe {
            imp::tm_metal_completed_dispatches(self.inner.raw)
        }
        #[cfg(not(target_os = "macos"))]
        {
            0
        }
    }

    /// GPU milliseconds of the most recently completed dispatch.
    pub fn last_gpu_ms(&self) -> f64 {
        #[cfg(target_os = "macos")]
        unsafe {
            imp::tm_metal_last_gpu_ms(self.inner.raw)
        }
        #[cfg(not(target_os = "macos"))]
        {
            0.0
        }
    }

    /// Allocate a GPU buffer. `shared` buffers are CPU-accessible (unified
    /// memory on Apple Silicon); private buffers are GPU-only.
    pub fn buffer(&self, bytes: usize, shared: bool) -> Result<MetalBuffer> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (bytes, shared);
            return Err(TuckmillError::new(
                ErrorCode::Unavailable,
                "Metal buffers are only available on macOS",
            ));
        }
        #[cfg(target_os = "macos")]
        {
            if bytes == 0 {
                return Err(TuckmillError::new(
                    ErrorCode::InvalidRequest,
                    "zero-length Metal buffers are refused (allocation must be explicit)",
                ));
            }
            let _guard = self.guard();
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw is valid under the lock; err is a writable buffer.
            let raw = unsafe {
                imp::tm_metal_buffer_new(
                    self.inner.raw,
                    bytes,
                    if shared { 1 } else { 0 },
                    err.as_mut_ptr(),
                    ERR_CAP,
                )
            };
            if raw.is_null() {
                return Err(metal_error(
                    imp::TM_METAL_ERR_ALLOC,
                    err_text(&err),
                    "buffer",
                ));
            }
            Ok(MetalBuffer {
                ctx: self.inner.clone(),
                raw,
            })
        }
    }

    fn operate(
        &self,
        name: &str,
        op: impl FnOnce(*mut imp::TmMetalCtx) -> (c_int, [c_char; ERR_CAP]),
    ) -> Result<()> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = op;
            return Err(TuckmillError::new(
                ErrorCode::Unavailable,
                format!("Metal {name} is only available on macOS"),
            ));
        }
        #[cfg(target_os = "macos")]
        {
            let _guard = self.guard();
            let (code, err) = op(self.inner.raw);
            if code != 0 {
                return Err(metal_error(code, err_text(&err), name));
            }
            Ok(())
        }
    }

    /// y = x * factor over `n` elements.
    pub fn f32_scale(&self, x: &MetalBuffer, y: &MetalBuffer, factor: f32, n: u32) -> Result<()> {
        self.check_pair(x, y)?;
        let xc = x.raw;
        let yc = y.raw;
        self.operate("f32_scale", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code = unsafe {
                imp::tm_metal_f32_scale(ctx, xc, yc, factor, n, err.as_mut_ptr(), ERR_CAP)
            };
            (code, err)
        })
    }

    /// y = y + alpha * a over `n` elements.
    pub fn f32_axpy(&self, a: &MetalBuffer, y: &MetalBuffer, alpha: f32, n: u32) -> Result<()> {
        self.check_pair(a, y)?;
        let ac = a.raw;
        let yc = y.raw;
        self.operate("f32_axpy", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code =
                unsafe { imp::tm_metal_f32_axpy(ctx, ac, yc, alpha, n, err.as_mut_ptr(), ERR_CAP) };
            (code, err)
        })
    }

    /// y = (f32) x over `n` binary16 elements.
    pub fn cast_f16_to_f32(&self, x: &MetalBuffer, y: &MetalBuffer, n: u32) -> Result<()> {
        self.check_pair(x, y)?;
        let xc = x.raw;
        let yc = y.raw;
        self.operate("cast_f16_f32", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code =
                unsafe { imp::tm_metal_cast_f16_f32(ctx, xc, yc, n, err.as_mut_ptr(), ERR_CAP) };
            (code, err)
        })
    }

    /// y = (half) x over `n` f32 elements (round-to-nearest-even).
    pub fn cast_f32_to_f16(&self, x: &MetalBuffer, y: &MetalBuffer, n: u32) -> Result<()> {
        self.check_pair(x, y)?;
        let xc = x.raw;
        let yc = y.raw;
        self.operate("cast_f32_f16", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code =
                unsafe { imp::tm_metal_cast_f32_f16(ctx, xc, yc, n, err.as_mut_ptr(), ERR_CAP) };
            (code, err)
        })
    }

    /// Packed Q8_0 matrix-vector decode: y[r] = dot(dequant(w_row_r), x).
    /// `w` holds `rows * cols / 32` blocks of 34 bytes; `cols` is a positive
    /// multiple of 32.
    pub fn q8_0_gemv(
        &self,
        w: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        for (buffer, name) in [(w, "weights"), (x, "x"), (y, "y")] {
            self.check_owned(buffer, name)?;
        }
        let (wc, xc, yc) = (w.raw, x.raw, y.raw);
        self.operate("q8_0_gemv", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code = unsafe {
                imp::tm_metal_q8_0_gemv(ctx, wc, xc, yc, rows, cols, err.as_mut_ptr(), ERR_CAP)
            };
            (code, err)
        })
    }

    /// Open a grouped-encoding scope: subsequent operations encode into one
    /// command buffer, committed (and completion-tracked) by `end_group`
    /// instead of one command buffer per operator.
    pub fn begin_group(&self) -> Result<()> {
        self.operate("begin_group", |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: caller holds the lock; ctx is valid.
            let code = unsafe { imp::tm_metal_begin_group(ctx, err.as_mut_ptr(), ERR_CAP) };
            (code, err)
        })
    }

    /// Commit the open group: one command buffer, one completion.
    pub fn end_group(&self) -> Result<()> {
        self.operate("end_group", |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: caller holds the lock; ctx is valid.
            let code = unsafe { imp::tm_metal_end_group(ctx, err.as_mut_ptr(), ERR_CAP) };
            (code, err)
        })
    }

    /// Discard the open group's work without committing it (cancellation at
    /// a step boundary; no partial GPU work is ever submitted).
    pub fn abort_group(&self) -> Result<()> {
        self.operate("abort_group", |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: caller holds the lock; ctx is valid.
            let code = unsafe { imp::tm_metal_abort_group(ctx, err.as_mut_ptr(), ERR_CAP) };
            (code, err)
        })
    }

    /// Dequantize one packed Q8_0 matrix row to f32.
    pub fn q8_0_row(&self, w: &MetalBuffer, out: &MetalBuffer, row: u32, cols: u32) -> Result<()> {
        self.check_pair(w, out)?;
        let (wc, oc) = (w.raw, out.raw);
        self.operate("q8_0_row", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code = unsafe {
                imp::tm_metal_q8_0_row_f32(ctx, wc, oc, row, cols, err.as_mut_ptr(), ERR_CAP)
            };
            (code, err)
        })
    }

    /// F16 matvec (fixture/F16-twin artifacts).
    pub fn f16_gemv(
        &self,
        w: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        for (buffer, name) in [(w, "weights"), (x, "x"), (y, "y")] {
            self.check_owned(buffer, name)?;
        }
        let (wc, xc, yc) = (w.raw, x.raw, y.raw);
        self.operate("f16_gemv", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code = unsafe {
                imp::tm_metal_f16_gemv(ctx, wc, xc, yc, rows, cols, err.as_mut_ptr(), ERR_CAP)
            };
            (code, err)
        })
    }

    /// Dequantize one F16 matrix row to f32.
    pub fn f16_row(&self, w: &MetalBuffer, out: &MetalBuffer, row: u32, cols: u32) -> Result<()> {
        self.check_pair(w, out)?;
        let (wc, oc) = (w.raw, out.raw);
        self.operate("f16_row", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code = unsafe {
                imp::tm_metal_f16_row_f32(ctx, wc, oc, row, cols, err.as_mut_ptr(), ERR_CAP)
            };
            (code, err)
        })
    }

    /// Row-wise RMSNorm with an F16-weight table.
    pub fn rmsnorm_f16w(
        &self,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        y: &MetalBuffer,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> Result<()> {
        for (buffer, name) in [(x, "x"), (weight, "weight"), (y, "y")] {
            self.check_owned(buffer, name)?;
        }
        let (xc, wc, yc) = (x.raw, weight.raw, y.raw);
        self.operate("rmsnorm_f16w", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code = unsafe {
                imp::tm_metal_rmsnorm_f16w(
                    ctx,
                    xc,
                    wc,
                    yc,
                    rows,
                    cols,
                    eps,
                    err.as_mut_ptr(),
                    ERR_CAP,
                )
            };
            (code, err)
        })
    }

    /// Row-wise RMSNorm: y[row] = rmsnorm(x[row], weight, eps).
    pub fn rmsnorm(
        &self,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        y: &MetalBuffer,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> Result<()> {
        for (buffer, name) in [(x, "x"), (weight, "weight"), (y, "y")] {
            self.check_owned(buffer, name)?;
        }
        let (xc, wc, yc) = (x.raw, weight.raw, y.raw);
        self.operate("rmsnorm", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code = unsafe {
                imp::tm_metal_rmsnorm_f32(
                    ctx,
                    xc,
                    wc,
                    yc,
                    rows,
                    cols,
                    eps,
                    err.as_mut_ptr(),
                    ERR_CAP,
                )
            };
            (code, err)
        })
    }

    /// NeoX RoPE in place over q (n_q_heads) and k (n_kv_heads) at `pos`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        inv_freq: &MetalBuffer,
        pos: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        for (buffer, name) in [(q, "q"), (k, "k"), (inv_freq, "inv_freq")] {
            self.check_owned(buffer, name)?;
        }
        let (qc, kc, fc) = (q.raw, k.raw, inv_freq.raw);
        self.operate("rope", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code = unsafe {
                imp::tm_metal_rope_f32(
                    ctx,
                    qc,
                    kc,
                    fc,
                    pos,
                    n_q_heads,
                    n_kv_heads,
                    head_dim,
                    err.as_mut_ptr(),
                    ERR_CAP,
                )
            };
            (code, err)
        })
    }

    /// Append k/v rows into the KV caches at `pos`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_store(
        &self,
        k_src: &MetalBuffer,
        v_src: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        kv_heads: u32,
        pos: u32,
        cap: u32,
        head_dim: u32,
        round_f16: bool,
    ) -> Result<()> {
        for (buffer, name) in [
            (k_src, "k_src"),
            (v_src, "v_src"),
            (k_cache, "k_cache"),
            (v_cache, "v_cache"),
        ] {
            self.check_owned(buffer, name)?;
        }
        let (ks, vs, kcat, vcat) = (k_src.raw, v_src.raw, k_cache.raw, v_cache.raw);
        self.operate("kv_store", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code = unsafe {
                imp::tm_metal_kv_store_f32(
                    ctx,
                    ks,
                    vs,
                    kcat,
                    vcat,
                    kv_heads,
                    pos,
                    cap,
                    head_dim,
                    u32::from(round_f16),
                    err.as_mut_ptr(),
                    ERR_CAP,
                )
            };
            (code, err)
        })
    }

    /// Causal GQA attention for one position; `scratch` holds heads*cap f32.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        out: &MetalBuffer,
        scratch: &MetalBuffer,
        heads: u32,
        pos: u32,
        cap: u32,
        head_dim: u32,
        group: u32,
        scale: f32,
    ) -> Result<()> {
        for (buffer, name) in [
            (q, "q"),
            (k_cache, "k_cache"),
            (v_cache, "v_cache"),
            (out, "out"),
            (scratch, "scratch"),
        ] {
            self.check_owned(buffer, name)?;
        }
        let (qc, kc, vc, oc, sc) = (q.raw, k_cache.raw, v_cache.raw, out.raw, scratch.raw);
        self.operate("attention", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code = unsafe {
                imp::tm_metal_attention_f32(
                    ctx,
                    qc,
                    kc,
                    vc,
                    oc,
                    sc,
                    heads,
                    pos,
                    cap,
                    head_dim,
                    group,
                    scale,
                    err.as_mut_ptr(),
                    ERR_CAP,
                )
            };
            (code, err)
        })
    }

    /// out[i] = silu(gate[i]) * up[i].
    pub fn silu_mul(
        &self,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        out: &MetalBuffer,
        n: u32,
    ) -> Result<()> {
        for (buffer, name) in [(gate, "gate"), (up, "up"), (out, "out")] {
            self.check_owned(buffer, name)?;
        }
        let (gc, uc, oc) = (gate.raw, up.raw, out.raw);
        self.operate("silu_mul", move |ctx| {
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: raw handles were validated against this context.
            let code = unsafe {
                imp::tm_metal_silu_mul_f32(ctx, gc, uc, oc, n, err.as_mut_ptr(), ERR_CAP)
            };
            (code, err)
        })
    }

    /// Block until every committed dispatch has completed.
    pub fn fence(&self) -> Result<()> {
        self.operate("fence", |_ctx| {
            #[cfg(target_os = "macos")]
            {
                let mut err = [0 as c_char; ERR_CAP];
                // SAFETY: caller holds the lock; raw is valid.
                let code = unsafe { imp::tm_metal_fence(_ctx, err.as_mut_ptr(), ERR_CAP) };
                (code, err)
            }
            #[cfg(not(target_os = "macos"))]
            {
                (0, [0 as c_char; ERR_CAP])
            }
        })
    }

    fn check_owned(&self, buffer: &MetalBuffer, name: &str) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &buffer.ctx) {
            return Err(TuckmillError::new(
                ErrorCode::InvalidRequest,
                format!("Metal {name} buffer belongs to a different context"),
            ));
        }
        Ok(())
    }

    fn check_pair(&self, x: &MetalBuffer, y: &MetalBuffer) -> Result<()> {
        self.check_owned(x, "input")?;
        self.check_owned(y, "output")
    }
}

/// RAII GPU buffer handle. Holds its context alive; releases the device
/// allocation on drop.
pub struct MetalBuffer {
    ctx: Arc<ContextInner>,
    #[cfg(target_os = "macos")]
    raw: *mut imp::TmMetalBuffer,
}

// SAFETY: same discipline as ContextInner — every ABI call goes through the
// context lock, and the buffer's own handle is only used inside those calls
// or while the holder has exclusive access.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    pub fn len(&self) -> usize {
        #[cfg(target_os = "macos")]
        unsafe {
            imp::tm_metal_buffer_len(self.ctx.raw, self.raw)
        }
        #[cfg(not(target_os = "macos"))]
        {
            0
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Host -> device copy (shared buffers only).
    pub fn upload(&self, offset: usize, data: &[u8]) -> Result<()> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (offset, data);
            return Err(TuckmillError::new(ErrorCode::Unavailable, "no Metal"));
        }
        #[cfg(target_os = "macos")]
        {
            let _guard = self.ctx.lock.lock().expect("metal lock");
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: handles valid; data slice pointer/len valid for read.
            let code = unsafe {
                imp::tm_metal_upload(
                    self.ctx.raw,
                    self.raw,
                    offset,
                    data.as_ptr() as *const c_void,
                    data.len(),
                    err.as_mut_ptr(),
                    ERR_CAP,
                )
            };
            if code != 0 {
                return Err(metal_error(code, err_text(&err), "upload"));
            }
            Ok(())
        }
    }

    /// Device -> host copy (shared buffers only).
    pub fn download(&self, offset: usize, dst: &mut [u8]) -> Result<()> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (offset, dst);
            return Err(TuckmillError::new(ErrorCode::Unavailable, "no Metal"));
        }
        #[cfg(target_os = "macos")]
        {
            let _guard = self.ctx.lock.lock().expect("metal lock");
            let mut err = [0 as c_char; ERR_CAP];
            // SAFETY: handles valid; dst slice pointer/len valid for write.
            let code = unsafe {
                imp::tm_metal_download(
                    self.ctx.raw,
                    self.raw,
                    offset,
                    dst.as_mut_ptr() as *mut c_void,
                    dst.len(),
                    err.as_mut_ptr(),
                    ERR_CAP,
                )
            };
            if code != 0 {
                return Err(metal_error(code, err_text(&err), "download"));
            }
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for MetalBuffer {
    fn drop(&mut self) {
        let _guard = self.ctx.lock.lock().expect("metal lock");
        // SAFETY: created by tm_metal_buffer_new against this context; freed
        // exactly once because Drop runs once.
        unsafe { imp::tm_metal_buffer_free(self.ctx.raw, self.raw) };
    }
}

#[cfg(test)]
mod tests;
