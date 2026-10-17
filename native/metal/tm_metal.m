/*
 * tm_metal.m — Objective-C implementation of the TuckCore <-> Metal ABI
 * (ADR-007). Compiled without ARC: this is a C-style FFI layer with
 * explicit retain/release ownership; completion counters and the live
 * allocation figure are C11 atomics so completion handlers can update them
 * from Metal's thread without locks.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tm_metal.h"

struct tm_metal_ctx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pso_scale;
    id<MTLComputePipelineState> pso_axpy;
    id<MTLComputePipelineState> pso_f16f32;
    id<MTLComputePipelineState> pso_f32f16;
    id<MTLComputePipelineState> pso_q8gemv;
    /* Last committed command buffer (retained); the queue executes in
     * order, so waiting on it fences everything committed before it. */
    id<MTLCommandBuffer> pending;
    _Atomic(uint64_t) committed;
    _Atomic(uint64_t) completed;
    _Atomic(uint64_t) last_gpu_nanos;
    _Atomic(uint64_t) allocated;
};

struct tm_metal_buffer {
    id<MTLBuffer> buffer; /* retained */
    size_t len;
    int shared;
    tm_metal_ctx *owner; /* not retained; RAII handles order lifetime */
};

static void tm_set_err(char *err, size_t cap, NSString *msg) {
    if (err == NULL || cap == 0) {
        return;
    }
    const char *utf8 = [msg UTF8String];
    snprintf(err, cap, "%s", utf8 != NULL ? utf8 : "unknown metal error");
}

static int tm_fail(char *err, size_t cap, int code, NSString *msg) {
    tm_set_err(err, cap, msg);
    return code;
}

static int tm_check_ctx(tm_metal_ctx *ctx, char *err, size_t cap, const char *what) {
    if (ctx == NULL) {
        return tm_fail(err, cap, TM_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: null context", what]);
    }
    return TM_METAL_OK;
}

/* Validates a buffer handle: owned by this context, non-null, long enough. */
static int tm_check_buffer(tm_metal_ctx *ctx, tm_metal_buffer *b, size_t need,
                           char *err, size_t cap, const char *name) {
    if (b == NULL || b->owner != ctx) {
        return tm_fail(err, cap, TM_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: buffer does not belong to this context", name]);
    }
    if (need == 0 || b->len < need) {
        return tm_fail(err, cap, TM_METAL_ERR_BOUNDS,
                       [NSString stringWithFormat:@"%s: buffer holds %zu bytes but %zu are required",
                                                  name, b->len, need]);
    }
    return TM_METAL_OK;
}

static int tm_commit(tm_metal_ctx *ctx, id<MTLCommandBuffer> cb, char *err, size_t cap) {
    (void)err;
    (void)cap;
    atomic_fetch_add_explicit(&ctx->committed, 1, memory_order_relaxed);
    if (ctx->pending != nil) {
        [ctx->pending release];
    }
    ctx->pending = [cb retain];
    tm_metal_ctx *c = ctx;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        double nanos = (buffer.GPUEndTime - buffer.GPUStartTime) * 1e9;
        if (nanos < 0.0) {
            nanos = 0.0;
        }
        atomic_store_explicit(&c->last_gpu_nanos, (uint64_t)nanos, memory_order_relaxed);
        atomic_fetch_add_explicit(&c->completed, 1, memory_order_relaxed);
    }];
    [cb commit];
    return TM_METAL_OK;
}

int tm_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil ? 1 : 0;
    }
}

const char *tm_metal_runtime_info(void) {
    static char info[256];
    static int filled = 0;
    if (!filled) {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                snprintf(info, sizeof(info), "no Metal device available");
            } else {
                snprintf(info, sizeof(info), "%s (unified memory %llu MiB)",
                         [[device name] UTF8String],
                         (unsigned long long)([device hasUnifiedMemory]
                                                  ? [device recommendedMaxWorkingSetSize] / (1024ull * 1024ull)
                                                  : 0ull));
            }
            filled = 1;
        }
    }
    return info;
}

static id<MTLComputePipelineState> tm_pipeline(tm_metal_ctx *ctx, NSString *name,
                                               NSError **out_error) {
    id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
    if (fn == nil) {
        if (out_error != NULL) {
            *out_error = [NSError errorWithDomain:@"tm_metal"
                                             code:TM_METAL_ERR_SOURCE
                                         userInfo:@{NSLocalizedDescriptionKey:
                                                        [NSString stringWithFormat:@"kernel %@ not found in the MSL source", name]}];
        }
        return nil;
    }
    id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:out_error];
    [fn release];
    return pso;
}

tm_metal_ctx *tm_metal_create(const char *msl_source, char *err, size_t err_cap) {
    if (msl_source == NULL) {
        tm_set_err(err, err_cap, @"null MSL source");
        return NULL;
    }
    @autoreleasepool {
        tm_metal_ctx *ctx = calloc(1, sizeof(tm_metal_ctx));
        if (ctx == NULL) {
            tm_set_err(err, err_cap, @"context allocation failed");
            return NULL;
        }
        /* Create-rule: MTLCreateSystemDefaultDevice returns +1. */
        ctx->device = MTLCreateSystemDefaultDevice();
        if (ctx->device == nil) {
            tm_set_err(err, err_cap, @"no Metal device available");
            free(ctx);
            return NULL;
        }
        /* New-rule: +1. */
        ctx->queue = [ctx->device newCommandQueue];
        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:msl_source];
        /* New-rule: +1. */
        ctx->library = [ctx->device newLibraryWithSource:source options:nil error:&error];
        if (ctx->library == nil) {
            tm_set_err(err, err_cap,
                       [NSString stringWithFormat:@"MSL compilation failed: %@",
                                                  error != nil ? [error localizedDescription] : @"unknown"]);
            tm_metal_destroy(ctx);
            return NULL;
        }
        /* tm_pipeline returns +1 (New-rule inside). */
        ctx->pso_scale = tm_pipeline(ctx, @"tm_scale_f32", &error);
        ctx->pso_axpy = tm_pipeline(ctx, @"tm_axpy_f32", &error);
        ctx->pso_f16f32 = tm_pipeline(ctx, @"tm_cast_f16_f32", &error);
        ctx->pso_f32f16 = tm_pipeline(ctx, @"tm_cast_f32_f16", &error);
        ctx->pso_q8gemv = tm_pipeline(ctx, @"tm_q8_0_gemv", &error);
        if (ctx->pso_scale == nil || ctx->pso_axpy == nil || ctx->pso_f16f32 == nil ||
            ctx->pso_f32f16 == nil || ctx->pso_q8gemv == nil) {
            tm_set_err(err, err_cap,
                       [NSString stringWithFormat:@"pipeline creation failed: %@",
                                                  error != nil ? [error localizedDescription] : @"unknown"]);
            tm_metal_destroy(ctx);
            return NULL;
        }
        atomic_init(&ctx->committed, 0);
        atomic_init(&ctx->completed, 0);
        atomic_init(&ctx->last_gpu_nanos, 0);
        atomic_init(&ctx->allocated, 0);
        return ctx;
    }
}

void tm_metal_destroy(tm_metal_ctx *ctx) {
    if (ctx == NULL) {
        return;
    }
    @autoreleasepool {
        if (ctx->pending != nil) {
            [ctx->pending waitUntilCompleted];
            [ctx->pending release];
            ctx->pending = nil;
        }
        uint64_t still_allocated = atomic_load_explicit(&ctx->allocated, memory_order_relaxed);
        if (still_allocated != 0) {
            fprintf(stderr, "tm_metal: %llu bytes still allocated at destroy; buffer lifetime bug\n",
                    (unsigned long long)still_allocated);
        }
        [ctx->pso_q8gemv release];
        [ctx->pso_f32f16 release];
        [ctx->pso_f16f32 release];
        [ctx->pso_axpy release];
        [ctx->pso_scale release];
        [ctx->library release];
        [ctx->queue release];
        [ctx->device release];
        free(ctx);
    }
}

tm_metal_buffer *tm_metal_buffer_new(tm_metal_ctx *ctx, size_t bytes, int shared,
                                     char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "buffer_new") != TM_METAL_OK) {
        return NULL;
    }
    if (bytes == 0) {
        tm_set_err(err, err_cap, @"buffer_new: zero-length allocation");
        return NULL;
    }
    @autoreleasepool {
        MTLResourceOptions options = shared ? MTLResourceStorageModeShared
                                            : MTLResourceStorageModePrivate;
        id<MTLBuffer> buffer = [ctx->device newBufferWithLength:bytes options:options];
        if (buffer == nil) {
            tm_set_err(err, err_cap,
                       [NSString stringWithFormat:@"buffer_new: allocation of %zu bytes failed", bytes]);
            return NULL;
        }
        tm_metal_buffer *handle = calloc(1, sizeof(tm_metal_buffer));
        if (handle == NULL) {
            [buffer release];
            tm_set_err(err, err_cap, @"buffer_new: handle allocation failed");
            return NULL;
        }
        handle->buffer = buffer; /* +1 from newBufferWithLength */
        handle->len = bytes;
        handle->shared = shared ? 1 : 0;
        handle->owner = ctx;
        atomic_fetch_add_explicit(&ctx->allocated, bytes, memory_order_relaxed);
        return handle;
    }
}

void tm_metal_buffer_free(tm_metal_ctx *ctx, tm_metal_buffer *buffer) {
    if (ctx == NULL || buffer == NULL) {
        return;
    }
    if (buffer->owner != ctx) {
        fprintf(stderr, "tm_metal: buffer_free from a different context refused\n");
        return;
    }
    @autoreleasepool {
        atomic_fetch_sub_explicit(&ctx->allocated, buffer->len, memory_order_relaxed);
        [buffer->buffer release];
        buffer->buffer = nil;
        free(buffer);
    }
}

void *tm_metal_buffer_contents(tm_metal_ctx *ctx, tm_metal_buffer *buffer) {
    if (ctx == NULL || buffer == NULL || buffer->owner != ctx || !buffer->shared) {
        return NULL;
    }
    return [buffer->buffer contents];
}

size_t tm_metal_buffer_len(const tm_metal_ctx *ctx, const tm_metal_buffer *buffer) {
    if (ctx == NULL || buffer == NULL || buffer->owner != (tm_metal_ctx *)ctx) {
        return 0;
    }
    return buffer->len;
}

static int tm_copy(tm_metal_ctx *ctx, tm_metal_buffer *buffer, size_t off,
                   void *host, size_t len, int to_device, char *err, size_t err_cap,
                   const char *what) {
    if (tm_check_ctx(ctx, err, err_cap, what) != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (len == 0 || off > buffer->len || len > buffer->len - off) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS,
                       [NSString stringWithFormat:@"%s: range %zu..%zu outside buffer", what, off, off + len]);
    }
    if (tm_check_buffer(ctx, buffer, off + len, err, err_cap, what) != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    if (!buffer->shared) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: private buffers have no host copy", what]);
    }
    void *contents = [buffer->buffer contents];
    if (to_device) {
        memcpy((uint8_t *)contents + off, host, len);
    } else {
        memcpy(host, (const uint8_t *)contents + off, len);
    }
    return TM_METAL_OK;
}

int tm_metal_upload(tm_metal_ctx *ctx, tm_metal_buffer *dst, size_t off,
                    const void *src, size_t len, char *err, size_t err_cap) {
    return tm_copy(ctx, dst, off, (void *)src, len, 1, err, err_cap, "upload");
}

int tm_metal_download(tm_metal_ctx *ctx, tm_metal_buffer *src, size_t off,
                      void *dst, size_t len, char *err, size_t err_cap) {
    return tm_copy(ctx, src, off, dst, len, 0, err, err_cap, "download");
}

/* ---------------------------------------------------------------- ops -- */

static int tm_dispatch_1d(tm_metal_ctx *ctx, id<MTLComputePipelineState> pso,
                          tm_metal_buffer *a, tm_metal_buffer *b, float scalar,
                          int has_scalar, uint32_t n, int cast_f16_in, int cast_f16_out,
                          char *err, size_t err_cap, const char *what) {
    if (tm_check_ctx(ctx, err, err_cap, what) != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS,
                       [NSString stringWithFormat:@"%s: zero length", what]);
    }
    size_t a_need = (size_t)n * (cast_f16_in ? 2 : 4);
    size_t b_need = (size_t)n * (cast_f16_out ? 2 : 4);
    if (tm_check_buffer(ctx, a, a_need, err, err_cap, what) != TM_METAL_OK ||
        tm_check_buffer(ctx, b, b_need, err, err_cap, what) != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                           [NSString stringWithFormat:@"%s: command encoder unavailable", what]);
        }
        [enc setComputePipelineState:pso];
        [enc setBuffer:a->buffer offset:0 atIndex:0];
        [enc setBuffer:b->buffer offset:0 atIndex:1];
        uint32_t index = 2;
        if (has_scalar) {
            [enc setBytes:&scalar length:sizeof(scalar) atIndex:index++];
        }
        [enc setBytes:&n length:sizeof(n) atIndex:index];
        NSUInteger width = pso.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width : 32;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        return tm_commit(ctx, cb, err, err_cap);
    }
}

int tm_metal_f32_scale(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                       float factor, uint32_t n, char *err, size_t err_cap) {
    return tm_dispatch_1d(ctx, ctx->pso_scale, x, y, factor, 1, n, 0, 0, err, err_cap,
                          "f32_scale");
}

int tm_metal_f32_axpy(tm_metal_ctx *ctx, tm_metal_buffer *a, tm_metal_buffer *y,
                      float alpha, uint32_t n, char *err, size_t err_cap) {
    return tm_dispatch_1d(ctx, ctx->pso_axpy, a, y, alpha, 1, n, 0, 0, err, err_cap,
                          "f32_axpy");
}

int tm_metal_cast_f16_f32(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                          uint32_t n, char *err, size_t err_cap) {
    return tm_dispatch_1d(ctx, ctx->pso_f16f32, x, y, 0.0f, 0, n, 1, 0, err, err_cap,
                          "cast_f16_f32");
}

int tm_metal_cast_f32_f16(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                          uint32_t n, char *err, size_t err_cap) {
    return tm_dispatch_1d(ctx, ctx->pso_f32f16, x, y, 0.0f, 0, n, 0, 1, err, err_cap,
                          "cast_f32_f16");
}

int tm_metal_q8_0_gemv(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *x,
                       tm_metal_buffer *y, uint32_t rows, uint32_t cols,
                       char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "q8_0_gemv") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"q8_0_gemv: rows %u, cols %u (cols must be a positive multiple of 32)",
                                                  rows, cols]);
    }
    size_t w_need = (size_t)rows * ((size_t)cols / 32u) * 34u;
    size_t x_need = (size_t)cols * 4u;
    size_t y_need = (size_t)rows * 4u;
    if (tm_check_buffer(ctx, w, w_need, err, err_cap, "q8_0_gemv.weights") != TM_METAL_OK ||
        tm_check_buffer(ctx, x, x_need, err, err_cap, "q8_0_gemv.x") != TM_METAL_OK ||
        tm_check_buffer(ctx, y, y_need, err, err_cap, "q8_0_gemv.y") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                           @"q8_0_gemv: command encoder unavailable");
        }
        [enc setComputePipelineState:ctx->pso_q8gemv];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [enc setBuffer:y->buffer offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        NSUInteger lanes = ctx->pso_q8gemv.threadExecutionWidth;
        if (lanes != 32) {
            [enc endEncoding];
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                           [NSString stringWithFormat:@"q8_0_gemv: device simd width %lu != 32", (unsigned long)lanes]);
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        return tm_commit(ctx, cb, err, err_cap);
    }
}

int tm_metal_fence(tm_metal_ctx *ctx, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "fence") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (ctx->pending == nil) {
        return TM_METAL_OK;
    }
    [ctx->pending waitUntilCompleted];
    if (ctx->pending.status == MTLCommandBufferStatusError) {
        return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                       [NSString stringWithFormat:@"fence: command buffer failed: %@",
                                                  ctx->pending.error != nil
                                                      ? [ctx->pending.error localizedDescription]
                                                      : @"unknown"]);
    }
    return TM_METAL_OK;
}

uint64_t tm_metal_committed_dispatches(const tm_metal_ctx *ctx) {
    return ctx == NULL ? 0 : atomic_load_explicit(&ctx->committed, memory_order_relaxed);
}

uint64_t tm_metal_completed_dispatches(const tm_metal_ctx *ctx) {
    return ctx == NULL ? 0 : atomic_load_explicit(&ctx->completed, memory_order_relaxed);
}

double tm_metal_last_gpu_ms(const tm_metal_ctx *ctx) {
    if (ctx == NULL) {
        return 0.0;
    }
    return (double)atomic_load_explicit(&ctx->last_gpu_nanos, memory_order_relaxed) / 1e6;
}

uint64_t tm_metal_allocated_bytes(const tm_metal_ctx *ctx) {
    return ctx == NULL ? 0 : atomic_load_explicit(&ctx->allocated, memory_order_relaxed);
}
