/*
 * tm_metal.h — TuckCore <-> Metal narrow C ABI (ADR-007).
 *
 * One context owns one MTLDevice, one MTLCommandQueue, and the compiled
 * compute pipelines for the operations declared here. Buffers are opaque
 * resource handles with explicit lifetime; dispatch is asynchronous and
 * completion-fenced through tm_metal_fence(). Measurements (committed and
 * completed dispatch counts, last GPU time, live allocated bytes) are
 * exposed so the caller can account for real GPU resources.
 *
 * All functions are thread-safe in the sense that a context may be used
 * from one thread at a time; completion counters are atomic and may be
 * observed from any thread. Return 0 on success, a TM_METAL_ERR_* code on
 * failure, with a human-readable message in `err` when provided.
 *
 * This is NOT the worker ABI (native/executor/tuckmill_worker_abi.h); only
 * TuckCore uses this interface.
 */
#ifndef TM_METAL_H
#define TM_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tm_metal_ctx tm_metal_ctx;
typedef struct tm_metal_buffer tm_metal_buffer;

enum {
    TM_METAL_OK = 0,
    TM_METAL_ERR_NO_DEVICE = -1,   /* no Metal device is available */
    TM_METAL_ERR_SOURCE = -2,      /* MSL compilation failed */
    TM_METAL_ERR_ALLOC = -3,       /* buffer allocation failed */
    TM_METAL_ERR_BOUNDS = -4,      /* size/range mismatch, zero length */
    TM_METAL_ERR_UNSUPPORTED = -5, /* invalid arguments / mode */
    TM_METAL_ERR_INTERNAL = -6     /* Metal/Objective-C runtime failure */
};

/* 1 when a Metal device exists (cheap; no context required). */
int tm_metal_available(void);

/* One-line runtime description (device name, unified memory size). Always
 * returns a valid static string. */
const char *tm_metal_runtime_info(void);

/* Create a context: device, queue, library compiled from `msl_source`, and
 * the pipeline states for every operation below. On failure returns NULL
 * and fills `err`. */
tm_metal_ctx *tm_metal_create(const char *msl_source, char *err, size_t err_cap);

/* Destroy the context. All buffers created from it must have been freed
 * first; the caller (Rust RAII handles) guarantees this. */
void tm_metal_destroy(tm_metal_ctx *ctx);

/* Allocate a buffer of `bytes` (> 0). `shared` buffers support CPU access
 * and host<->device copies; private buffers are GPU-only. Returns NULL on
 * failure. */
tm_metal_buffer *tm_metal_buffer_new(tm_metal_ctx *ctx, size_t bytes, int shared,
                                     char *err, size_t err_cap);

/* Release a buffer. Passing a buffer not owned by the context is a
 * programming error and reports TM_METAL_ERR_UNSUPPORTED. */
void tm_metal_buffer_free(tm_metal_ctx *ctx, tm_metal_buffer *buffer);

/* CPU pointer into a shared buffer (NULL for private buffers or if the
 * context does not own the buffer). Valid until the buffer is freed. */
void *tm_metal_buffer_contents(tm_metal_ctx *ctx, tm_metal_buffer *buffer);

/* Length of the buffer in bytes (0 when the handle is invalid). */
size_t tm_metal_buffer_len(const tm_metal_ctx *ctx, const tm_metal_buffer *buffer);

/* Host -> device and device -> host copies through shared storage. */
int tm_metal_upload(tm_metal_ctx *ctx, tm_metal_buffer *dst, size_t off,
                    const void *src, size_t len, char *err, size_t err_cap);
int tm_metal_download(tm_metal_ctx *ctx, tm_metal_buffer *src, size_t off,
                      void *dst, size_t len, char *err, size_t err_cap);

/* y[i] = x[i] * factor, i < n. n may be any length (tail-safe). */
int tm_metal_f32_scale(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                       float factor, uint32_t n, char *err, size_t err_cap);

/* y[i] = y[i] + alpha * a[i], i < n. */
int tm_metal_f32_axpy(tm_metal_ctx *ctx, tm_metal_buffer *a, tm_metal_buffer *y,
                      float alpha, uint32_t n, char *err, size_t err_cap);

/* y[i] = (float) x[i], i < n, where x holds IEEE binary16 values. */
int tm_metal_cast_f16_f32(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                          uint32_t n, char *err, size_t err_cap);

/* y[i] = (half) x[i], i < n, round-to-nearest-even. */
int tm_metal_cast_f32_f16(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                          uint32_t n, char *err, size_t err_cap);

/* Packed Q8_0 matrix-vector decode: y[r] = dot(dequant_q8_0(w_row_r), x).
 * `w` holds `rows * cols / 32` blocks of 34 bytes ([f16 d][i8 qs; 32]).
 * `cols` must be a positive multiple of 32; `x` is f32 of length `cols`;
 * `y` is f32 of length `rows`. */
int tm_metal_q8_0_gemv(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *x,
                       tm_metal_buffer *y, uint32_t rows, uint32_t cols,
                       char *err, size_t err_cap);

/* Block until every committed command buffer has completed. Safe to call
 * with no pending work. */
int tm_metal_fence(tm_metal_ctx *ctx, char *err, size_t err_cap);

/* Measurement surface. */
uint64_t tm_metal_committed_dispatches(const tm_metal_ctx *ctx);
uint64_t tm_metal_completed_dispatches(const tm_metal_ctx *ctx);
/* GPU time of the most recently completed command buffer, milliseconds
 * (0.0 before the first completion). */
double tm_metal_last_gpu_ms(const tm_metal_ctx *ctx);
/* Live bytes currently allocated in buffers from this context. */
uint64_t tm_metal_allocated_bytes(const tm_metal_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* TM_METAL_H */
