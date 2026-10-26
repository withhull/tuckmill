/*
 * tm_metal.m — Objective-C implementation of the TuckCore <-> Metal ABI
 * (ADR-007). Compiled without ARC: this is a C-style FFI layer with
 * explicit retain/release ownership; completion counters and the live
 * allocation figure are C11 atomics so completion handlers can update them
 * from Metal's thread without locks.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <os/lock.h>
#include <stdatomic.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tm_metal.h"

#define TM_PROFILE_SLOTS 64
#define TM_PROFILE_NAME 40

/* One operator label's accumulated GPU time (profiling only). */
typedef struct {
    char name[TM_PROFILE_NAME];
    uint64_t nanos;
    uint64_t count;
} tm_profile_slot;

/* A label copied by value into completion handlers, so no pointer has to
 * outlive the dispatch call that produced it. */
typedef struct {
    char s[TM_PROFILE_NAME];
} tm_label;

struct tm_metal_ctx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pso_scale;
    id<MTLComputePipelineState> pso_axpy;
    id<MTLComputePipelineState> pso_f16f32;
    id<MTLComputePipelineState> pso_f32f16;
    id<MTLComputePipelineState> pso_q8gemv;
    id<MTLComputePipelineState> pso_f16gemv;
    id<MTLComputePipelineState> pso_q8row;
    id<MTLComputePipelineState> pso_rmsnorm;
    id<MTLComputePipelineState> pso_rope;
    id<MTLComputePipelineState> pso_kvstore;
    id<MTLComputePipelineState> pso_attention;
    id<MTLComputePipelineState> pso_silu;
    id<MTLComputePipelineState> pso_f16row;
    id<MTLComputePipelineState> pso_rmsnorm_f16w;
    id<MTLComputePipelineState> pso_q8gemm;
    id<MTLComputePipelineState> pso_f16gemm;
    id<MTLComputePipelineState> pso_rope_batch;
    id<MTLComputePipelineState> pso_kvstore_batch;
    id<MTLComputePipelineState> pso_attention_batch;
    id<MTLComputePipelineState> pso_q35_conv;
    id<MTLComputePipelineState> pso_q35_conv_shift;
    id<MTLComputePipelineState> pso_q35_l2norm;
    id<MTLComputePipelineState> pso_q35_beta_decay;
    id<MTLComputePipelineState> pso_q35_gdn_step;
    id<MTLComputePipelineState> pso_q35_gated_norm;
    id<MTLComputePipelineState> pso_q35_qg_split;
    id<MTLComputePipelineState> pso_q35_rope_partial;
    id<MTLComputePipelineState> pso_q35_kv_store_f16;
    id<MTLComputePipelineState> pso_q35_attention_f16;
    id<MTLComputePipelineState> pso_q35_attn_gate;
    id<MTLComputePipelineState> pso_q35_silu;
    id<MTLComputePipelineState> pso_q35_split_qkv;
    /* Last committed command buffer (retained); the queue executes in
     * order, so waiting on it fences everything committed before it. */
    id<MTLCommandBuffer> pending;
    /* Open dispatch group (nil when none): operations encode into one
     * command buffer instead of committing per operator. */
    id<MTLCommandBuffer> group_cb;
    id<MTLComputeCommandEncoder> group_enc;
    NSMutableArray *group_keep; /* in-flight resource leases for the group */
    int group_active;
    _Atomic(uint64_t) committed;
    _Atomic(uint64_t) completed;
    _Atomic(uint64_t) last_gpu_nanos;
    /* Cumulative GPU time over every completed command buffer. */
    _Atomic(uint64_t) gpu_nanos_total;
    /* Opt-in per-operator attribution (profiling only; see the header). */
    _Atomic(int) profiling;
    os_unfair_lock profile_lock;
    uint32_t profile_used;
    tm_profile_slot profile[TM_PROFILE_SLOTS];
    _Atomic(uint64_t) allocated;
    /* Bytes leased by pending command buffers: physically alive until the
     * completion handler runs, even if their handles were already freed.
     * Combined with `allocated` this bounds live GPU bytes from above (a
     * buffer that is both handle-live and in flight is counted twice; the
     * bound is conservative, never understated). */
    _Atomic(uint64_t) leased;
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

/* Commits a command buffer, retaining `in_flight` (the buffers its work
 * reads or writes) until the command buffer completes. A buffer handle may
 * therefore be freed while its GPU work is still pending: the device
 * resource stays leased until the completion handler runs. Without this,
 * dropping a buffer before an explicit fence could free storage the GPU is
 * still using. */
static int tm_commit(tm_metal_ctx *ctx, id<MTLCommandBuffer> cb, NSArray *in_flight,
                     const char *what, char *err, size_t cap) {
    (void)err;
    (void)cap;
    tm_label label = {{0}};
    if (atomic_load_explicit(&ctx->profiling, memory_order_relaxed) && what != NULL) {
        strlcpy(label.s, what, sizeof label.s);
    }
    uint64_t leased = 0;
    for (id<MTLBuffer> buffer in in_flight) {
        leased += (uint64_t)buffer.length;
    }
    atomic_fetch_add_explicit(&ctx->leased, leased, memory_order_relaxed);
    atomic_fetch_add_explicit(&ctx->committed, 1, memory_order_relaxed);
    if (ctx->pending != nil) {
        [ctx->pending release];
    }
    ctx->pending = [cb retain];
    tm_metal_ctx *c = ctx;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        /* Touch the lease array so the block capture keeps it (and every
         * leased MTLBuffer) alive until completion; the array is released
         * with the handler afterwards. */
        (void)in_flight.count;
        atomic_fetch_sub_explicit(&c->leased, leased, memory_order_relaxed);
        double nanos = (buffer.GPUEndTime - buffer.GPUStartTime) * 1e9;
        if (nanos < 0.0) {
            nanos = 0.0;
        }
        atomic_store_explicit(&c->last_gpu_nanos, (uint64_t)nanos, memory_order_relaxed);
        atomic_fetch_add_explicit(&c->gpu_nanos_total, (uint64_t)nanos, memory_order_relaxed);
        if (label.s[0] != '\0') {
            os_unfair_lock_lock(&c->profile_lock);
            uint32_t slot = 0;
            while (slot < c->profile_used && strcmp(c->profile[slot].name, label.s) != 0) {
                slot++;
            }
            if (slot == c->profile_used && slot < TM_PROFILE_SLOTS) {
                strlcpy(c->profile[slot].name, label.s, TM_PROFILE_NAME);
                c->profile_used++;
            }
            if (slot < TM_PROFILE_SLOTS) {
                c->profile[slot].nanos += (uint64_t)nanos;
                c->profile[slot].count += 1;
            }
            os_unfair_lock_unlock(&c->profile_lock);
        }
        atomic_fetch_add_explicit(&c->completed, 1, memory_order_relaxed);
    }];
    [cb commit];
    return TM_METAL_OK;
}

/* Waits for every committed command buffer. Shared by fence and the copy
 * contract: host copies are ordered after all committed GPU work. */
static int tm_wait_pending(tm_metal_ctx *ctx, char *err, size_t cap) {
    if (ctx->pending == nil) {
        return TM_METAL_OK;
    }
    [ctx->pending waitUntilCompleted];
    if (ctx->pending.status == MTLCommandBufferStatusError) {
        return tm_fail(err, cap, TM_METAL_ERR_INTERNAL,
                       [NSString stringWithFormat:@"command buffer failed: %@",
                                                  ctx->pending.error != nil
                                                      ? [ctx->pending.error localizedDescription]
                                                      : @"unknown"]);
    }
    /* waitUntilCompleted signals execution completion; addCompletedHandler
     * blocks are scheduled separately and may still be pending. Settlement
     * of the counter/lease accounting is part of the fence contract (the
     * in-flight byte figure must be zero when the fence returns), so wait
     * for the handlers to run. Execution is already complete, so the wait
     * only covers scheduling latency (2 s budget). If it still does not
     * settle, report a typed failure: claiming success here would weaken
     * the acknowledged-fence invariant. */
    for (int spin = 0; spin < 40000; ++spin) {
        if (atomic_load_explicit(&ctx->completed, memory_order_relaxed) >=
            atomic_load_explicit(&ctx->committed, memory_order_relaxed)) {
            return TM_METAL_OK;
        }
        usleep(50);
    }
    return tm_fail(err, cap, TM_METAL_ERR_INTERNAL,
                   @"fence: command buffers completed but completion-handler accounting did not settle");
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
                snprintf(info, sizeof(info), "%s (recommended GPU working set %llu MiB)",
                         [[device name] UTF8String],
                         (unsigned long long)([device recommendedMaxWorkingSetSize] / (1024ull * 1024ull)));
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
        ctx->pso_f16gemv = tm_pipeline(ctx, @"tm_f16_gemv", &error);
        ctx->pso_q8row = tm_pipeline(ctx, @"tm_q8_0_row_f32", &error);
        ctx->pso_rmsnorm = tm_pipeline(ctx, @"tm_rmsnorm_f32", &error);
        ctx->pso_rope = tm_pipeline(ctx, @"tm_rope_f32", &error);
        ctx->pso_kvstore = tm_pipeline(ctx, @"tm_kv_store_f32", &error);
        ctx->pso_attention = tm_pipeline(ctx, @"tm_attention_f32", &error);
        ctx->pso_silu = tm_pipeline(ctx, @"tm_silu_mul_f32", &error);
        ctx->pso_f16row = tm_pipeline(ctx, @"tm_f16_row_f32", &error);
        ctx->pso_rmsnorm_f16w = tm_pipeline(ctx, @"tm_rmsnorm_f16w", &error);
        ctx->pso_q8gemm = tm_pipeline(ctx, @"tm_q8_0_gemm", &error);
        ctx->pso_f16gemm = tm_pipeline(ctx, @"tm_f16_gemm", &error);
        ctx->pso_rope_batch = tm_pipeline(ctx, @"tm_rope_batch_f32", &error);
        ctx->pso_kvstore_batch = tm_pipeline(ctx, @"tm_kv_store_batch_f32", &error);
        ctx->pso_attention_batch = tm_pipeline(ctx, @"tm_attention_batch_f32", &error);
        ctx->pso_q35_conv = tm_pipeline(ctx, @"tm_qwen35_conv_step", &error);
        ctx->pso_q35_conv_shift = tm_pipeline(ctx, @"tm_qwen35_conv_shift", &error);
        ctx->pso_q35_l2norm = tm_pipeline(ctx, @"tm_qwen35_l2norm_heads", &error);
        ctx->pso_q35_beta_decay = tm_pipeline(ctx, @"tm_qwen35_beta_decay", &error);
        ctx->pso_q35_gdn_step = tm_pipeline(ctx, @"tm_qwen35_gdn_step", &error);
        ctx->pso_q35_gated_norm = tm_pipeline(ctx, @"tm_qwen35_gated_norm", &error);
        ctx->pso_q35_qg_split = tm_pipeline(ctx, @"tm_qwen35_q_gate_split", &error);
        ctx->pso_q35_rope_partial = tm_pipeline(ctx, @"tm_qwen35_rope_partial_f32", &error);
        ctx->pso_q35_kv_store_f16 = tm_pipeline(ctx, @"tm_qwen35_kv_store_f16", &error);
        ctx->pso_q35_attention_f16 = tm_pipeline(ctx, @"tm_qwen35_attention_f16", &error);
        ctx->pso_q35_attn_gate = tm_pipeline(ctx, @"tm_qwen35_attn_gate", &error);
        ctx->pso_q35_silu = tm_pipeline(ctx, @"tm_qwen35_silu", &error);
        ctx->pso_q35_split_qkv = tm_pipeline(ctx, @"tm_qwen35_split_qkv", &error);
        if (ctx->pso_scale == nil || ctx->pso_axpy == nil || ctx->pso_f16f32 == nil ||
            ctx->pso_f32f16 == nil || ctx->pso_q8gemv == nil || ctx->pso_f16gemv == nil ||
            ctx->pso_q8row == nil || ctx->pso_rmsnorm == nil || ctx->pso_rope == nil ||
            ctx->pso_kvstore == nil || ctx->pso_attention == nil || ctx->pso_silu == nil ||
            ctx->pso_f16row == nil || ctx->pso_rmsnorm_f16w == nil ||
            ctx->pso_q8gemm == nil || ctx->pso_f16gemm == nil ||
            ctx->pso_rope_batch == nil || ctx->pso_kvstore_batch == nil ||
            ctx->pso_attention_batch == nil ||
            ctx->pso_q35_conv == nil || ctx->pso_q35_conv_shift == nil ||
            ctx->pso_q35_l2norm == nil || ctx->pso_q35_beta_decay == nil ||
            ctx->pso_q35_gdn_step == nil || ctx->pso_q35_gated_norm == nil ||
            ctx->pso_q35_qg_split == nil || ctx->pso_q35_rope_partial == nil ||
            ctx->pso_q35_kv_store_f16 == nil || ctx->pso_q35_attention_f16 == nil ||
            ctx->pso_q35_attn_gate == nil || ctx->pso_q35_silu == nil ||
            ctx->pso_q35_split_qkv == nil) {
            tm_set_err(err, err_cap,
                       [NSString stringWithFormat:@"pipeline creation failed: %@",
                                                  error != nil ? [error localizedDescription] : @"unknown"]);
            tm_metal_destroy(ctx);
            return NULL;
        }
        atomic_init(&ctx->committed, 0);
        atomic_init(&ctx->completed, 0);
        atomic_init(&ctx->last_gpu_nanos, 0);
        atomic_init(&ctx->gpu_nanos_total, 0);
        atomic_init(&ctx->profiling, 0);
        ctx->profile_lock = OS_UNFAIR_LOCK_INIT;
        ctx->profile_used = 0;
        atomic_init(&ctx->allocated, 0);
        atomic_init(&ctx->leased, 0);
        return ctx;
    }
}

void tm_metal_destroy(tm_metal_ctx *ctx) {
    if (ctx == NULL) {
        return;
    }
    @autoreleasepool {
        if (ctx->group_active) {
            /* Uncommitted group: discard its work, never abandon silently. */
            [ctx->group_enc endEncoding];
            [ctx->group_enc release];
            [ctx->group_cb release];
            [ctx->group_keep release];
            ctx->group_enc = nil;
            ctx->group_cb = nil;
            ctx->group_keep = nil;
            ctx->group_active = 0;
        }
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
        [ctx->pso_attention_batch release];
        [ctx->pso_kvstore_batch release];
        [ctx->pso_rope_batch release];
        [ctx->pso_f16gemm release];
        [ctx->pso_q8gemm release];
        [ctx->pso_rmsnorm_f16w release];
        [ctx->pso_f16row release];
        [ctx->pso_silu release];
        [ctx->pso_attention release];
        [ctx->pso_kvstore release];
        [ctx->pso_rope release];
        [ctx->pso_rmsnorm release];
        [ctx->pso_q8row release];
        [ctx->pso_f16gemv release];
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

/*
 * Copy contract: host<->device copies on shared buffers are ordered after
 * every committed GPU command buffer (the call waits for completion first).
 * Reads therefore observe finished compute, and writes cannot race
 * in-flight work. Explicit tm_metal_fence() checkpoints remain available
 * for callers that want to pay the wait once instead of per copy.
 */
static int tm_copy(tm_metal_ctx *ctx, tm_metal_buffer *buffer, size_t off,
                   void *host, size_t len, int to_device, char *err, size_t err_cap,
                   const char *what) {
    if (tm_check_ctx(ctx, err, err_cap, what) != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    /* Validate the handle (null, ownership) before touching its fields. */
    if (buffer == NULL || buffer->owner != ctx) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: buffer does not belong to this context", what]);
    }
    if (len == 0 || off > buffer->len || len > buffer->len - off) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS,
                       [NSString stringWithFormat:@"%s: range %zu..%zu outside buffer", what, off, off + len]);
    }
    if (!buffer->shared) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: private buffers have no host copy", what]);
    }
    if (ctx->group_active) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: close the open dispatch group first", what]);
    }
    if (tm_wait_pending(ctx, err, err_cap) != TM_METAL_OK) {
        return TM_METAL_ERR_INTERNAL;
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

/*
 * Dispatch frames. With no open group, an operation creates and commits its
 * own command buffer (correctness-first; C0 behavior). With an open group
 * (tm_metal_begin_group), operations append to that one command buffer, so
 * a whole transformer step becomes one committed unit with one completion —
 * no per-operator commit overhead.
 */
typedef struct {
    id<MTLCommandBuffer> cb;          /* nil inside an open group */
    id<MTLComputeCommandEncoder> enc;
    NSMutableArray *keep;             /* in-flight resource leases */
    int implicit;
} tm_frame;

static id<MTLComputeCommandEncoder> tm_frame_open(tm_metal_ctx *ctx, tm_frame *f) {
    if (ctx->group_active) {
        f->implicit = 0;
        f->cb = nil;
        f->enc = ctx->group_enc;
        f->keep = ctx->group_keep;
        return f->enc;
    }
    f->implicit = 1;
    f->cb = [ctx->queue commandBuffer];
    f->enc = [f->cb computeCommandEncoder];
    f->keep = [NSMutableArray array];
    return f->enc;
}

static int tm_frame_commit(tm_metal_ctx *ctx, tm_frame *f, char *err, size_t cap,
                           const char *what) {
    if (f->enc == nil) {
        return tm_fail(err, cap, TM_METAL_ERR_INTERNAL,
                       [NSString stringWithFormat:@"%s: command encoder unavailable", what]);
    }
    if (!f->implicit) {
        return TM_METAL_OK; /* the open group commits once at end_group */
    }
    [f->enc endEncoding];
    return tm_commit(ctx, f->cb, f->keep, what, err, cap);
}

int tm_metal_begin_group(tm_metal_ctx *ctx, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "begin_group") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (ctx->group_active) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       @"begin_group: a dispatch group is already open");
    }
    @autoreleasepool {
        ctx->group_cb = [[ctx->queue commandBuffer] retain];
        ctx->group_enc = [[ctx->group_cb computeCommandEncoder] retain];
        ctx->group_keep = [[NSMutableArray alloc] init];
        if (ctx->group_cb == nil || ctx->group_enc == nil || ctx->group_keep == nil) {
            if (ctx->group_enc != nil) { [ctx->group_enc endEncoding]; }
            [ctx->group_enc release];
            [ctx->group_cb release];
            [ctx->group_keep release];
            ctx->group_enc = nil;
            ctx->group_cb = nil;
            ctx->group_keep = nil;
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                           @"begin_group: command buffering unavailable");
        }
        ctx->group_active = 1;
        return TM_METAL_OK;
    }
}

int tm_metal_end_group(tm_metal_ctx *ctx, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "end_group") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (!ctx->group_active) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       @"end_group: no dispatch group is open");
    }
    @autoreleasepool {
        [ctx->group_enc endEncoding];
        id<MTLCommandBuffer> cb = ctx->group_cb;
        NSMutableArray *keep = ctx->group_keep;
        id<MTLComputeCommandEncoder> enc = ctx->group_enc;
        ctx->group_cb = nil;
        ctx->group_enc = nil;
        ctx->group_keep = nil;
        ctx->group_active = 0;
        int rc = tm_commit(ctx, cb, keep, "group", err, err_cap);
        [enc release];
        [cb release];
        [keep release];
        return rc;
    }
}

/* Discards the open group's work without committing it (cancellation at a
 * step boundary: no partial GPU work is ever submitted). */
int tm_metal_abort_group(tm_metal_ctx *ctx, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "abort_group") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (!ctx->group_active) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       @"abort_group: no dispatch group is open");
    }
    [ctx->group_enc endEncoding];
    [ctx->group_enc release];
    [ctx->group_cb release];
    [ctx->group_keep release];
    ctx->group_enc = nil;
    ctx->group_cb = nil;
    ctx->group_keep = nil;
    ctx->group_active = 0;
    return TM_METAL_OK;
}

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
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                           [NSString stringWithFormat:@"%s: command encoder unavailable", what]);
        }
        [enc setComputePipelineState:pso];
        [enc setBuffer:a->buffer offset:0 atIndex:0];
        [enc setBuffer:b->buffer offset:0 atIndex:1];
        [f.keep addObject:a->buffer];
        [f.keep addObject:b->buffer];
        uint32_t index = 2;
        if (has_scalar) {
            [enc setBytes:&scalar length:sizeof(scalar) atIndex:index++];
        }
        [enc setBytes:&n length:sizeof(n) atIndex:index];
        NSUInteger width = pso.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width : 32;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, what);
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
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                           @"q8_0_gemv: command encoder unavailable");
        }
        [enc setComputePipelineState:ctx->pso_q8gemv];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [enc setBuffer:y->buffer offset:0 atIndex:2];
        [f.keep addObject:w->buffer];
        [f.keep addObject:x->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        NSUInteger lanes = ctx->pso_q8gemv.threadExecutionWidth;
        if (lanes != 32) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                           [NSString stringWithFormat:@"q8_0_gemv: device simd width %lu != 32", (unsigned long)lanes]);
        }
        /* Multi-row kernel: 2 SIMD groups x 4 rows per threadgroup (the
         * TM_Q8_SG / TM_Q8_ROWS constants in kernels.metal). */
        const NSUInteger rows_per_tg = 8;
        NSUInteger groups = ((NSUInteger)rows + rows_per_tg - 1) / rows_per_tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        /* Profiling attributes GEMV time per matrix shape. */
        char label[TM_PROFILE_NAME];
        if (atomic_load_explicit(&ctx->profiling, memory_order_relaxed)) {
            snprintf(label, sizeof label, "q8_0_gemv:%ux%u", rows, cols);
        } else {
            strlcpy(label, "q8_0_gemv", sizeof label);
        }
        return tm_frame_commit(ctx, &f, err, err_cap, label);
    }
}

int tm_metal_f16_gemv(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *x,
                      tm_metal_buffer *y, uint32_t rows, uint32_t cols,
                      char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "f16_gemv") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED, @"f16_gemv: zero shape");
    }
    size_t w_need = (size_t)rows * (size_t)cols * 2u;
    size_t x_need = (size_t)cols * 4u;
    size_t y_need = (size_t)rows * 4u;
    if (tm_check_buffer(ctx, w, w_need, err, err_cap, "f16_gemv.weights") != TM_METAL_OK ||
        tm_check_buffer(ctx, x, x_need, err, err_cap, "f16_gemv.x") != TM_METAL_OK ||
        tm_check_buffer(ctx, y, y_need, err, err_cap, "f16_gemv.y") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                           @"f16_gemv: command encoder unavailable");
        }
        [enc setComputePipelineState:ctx->pso_f16gemv];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [enc setBuffer:y->buffer offset:0 atIndex:2];
        [f.keep addObject:w->buffer];
        [f.keep addObject:x->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "f16_gemv");
    }
}

int tm_metal_q8_0_row_f32(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *out,
                          uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "q8_0_row") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (cols == 0 || (cols % 32u) != 0u) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       @"q8_0_row: cols must be a positive multiple of 32");
    }
    size_t w_need = ((size_t)row + 1u) * ((size_t)cols / 32u) * 34u;
    if (tm_check_buffer(ctx, w, w_need, err, err_cap, "q8_0_row.weights") != TM_METAL_OK ||
        tm_check_buffer(ctx, out, ((size_t)out_off + cols) * 4u, err, err_cap, "q8_0_row.out") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                           @"q8_0_row: command encoder unavailable");
        }
        [enc setComputePipelineState:ctx->pso_q8row];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:out->buffer offset:0 atIndex:1];
        [f.keep addObject:w->buffer];
        [f.keep addObject:out->buffer];
        [enc setBytes:&row length:sizeof(row) atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&out_off length:sizeof(out_off) atIndex:4];
        NSUInteger tg = 256;
        NSUInteger width = ctx->pso_q8row.threadExecutionWidth;
        if (width > 0 && width < tg) { tg = width * 8; }
        [enc dispatchThreads:MTLSizeMake((NSUInteger)cols, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "q8_0_row");
    }
}

static int tm_rmsnorm_dispatch(tm_metal_ctx *ctx, id<MTLComputePipelineState> pso,
                               tm_metal_buffer *x, tm_metal_buffer *weight,
                               tm_metal_buffer *y, uint32_t rows, uint32_t cols, float eps,
                               size_t weight_window, char *err, size_t err_cap,
                               const char *what) {
    if (tm_check_ctx(ctx, err, err_cap, what) != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: zero shape", what]);
    }
    size_t x_need = (size_t)rows * (size_t)cols * 4u;
    if (tm_check_buffer(ctx, x, x_need, err, err_cap, "rmsnorm.x") != TM_METAL_OK ||
        tm_check_buffer(ctx, weight, (size_t)cols * weight_window, err, err_cap, "rmsnorm.weight") != TM_METAL_OK ||
        tm_check_buffer(ctx, y, x_need, err, err_cap, "rmsnorm.y") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                           [NSString stringWithFormat:@"%s: no encoder", what]);
        }
        [enc setComputePipelineState:pso];
        [enc setBuffer:x->buffer offset:0 atIndex:0];
        [enc setBuffer:weight->buffer offset:0 atIndex:1];
        [enc setBuffer:y->buffer offset:0 atIndex:2];
        [f.keep addObject:x->buffer];
        [f.keep addObject:weight->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&eps length:sizeof(eps) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, what);
    }
}

int tm_metal_rmsnorm_f32(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *weight,
                         tm_metal_buffer *y, uint32_t rows, uint32_t cols, float eps,
                         char *err, size_t err_cap) {
    return tm_rmsnorm_dispatch(ctx, ctx->pso_rmsnorm, x, weight, y, rows, cols, eps, 4,
                               err, err_cap, "rmsnorm");
}

int tm_metal_rope_f32(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k,
                      tm_metal_buffer *inv_freq, uint32_t pos, uint32_t n_q_heads,
                      uint32_t n_kv_heads, uint32_t head_dim, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "rope") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (head_dim == 0 || (head_dim % 2u) != 0u || n_q_heads == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED, @"rope: shape");
    }
    size_t q_need = (size_t)n_q_heads * (size_t)head_dim * 4u;
    size_t k_need = (size_t)n_kv_heads * (size_t)head_dim * 4u;
    if (tm_check_buffer(ctx, q, q_need, err, err_cap, "rope.q") != TM_METAL_OK ||
        tm_check_buffer(ctx, k, k_need, err, err_cap, "rope.k") != TM_METAL_OK ||
        tm_check_buffer(ctx, inv_freq, (size_t)(head_dim / 2u) * 4u, err, err_cap, "rope.inv_freq") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"rope: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_rope];
        [enc setBuffer:q->buffer offset:0 atIndex:0];
        [enc setBuffer:k->buffer offset:0 atIndex:1];
        [enc setBuffer:inv_freq->buffer offset:0 atIndex:2];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k->buffer];
        [f.keep addObject:inv_freq->buffer];
        [enc setBytes:&pos length:sizeof(pos) atIndex:3];
        [enc setBytes:&n_q_heads length:sizeof(n_q_heads) atIndex:4];
        [enc setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:5];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(n_q_heads + n_kv_heads), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(head_dim / 2u, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "rope");
    }
}

int tm_metal_kv_store_f32(tm_metal_ctx *ctx, tm_metal_buffer *k_src, tm_metal_buffer *v_src,
                          tm_metal_buffer *k_cache, tm_metal_buffer *v_cache,
                          uint32_t kv_heads, uint32_t pos, uint32_t cap, uint32_t head_dim,
                          uint32_t round_f16, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "kv_store") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (pos >= cap || head_dim == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS,
                       [NSString stringWithFormat:@"kv_store: pos %u, cap %u", pos, cap]);
    }
    size_t src_need = (size_t)kv_heads * (size_t)head_dim * 4u;
    size_t cache_need = (size_t)kv_heads * (size_t)cap * (size_t)head_dim * 4u;
    if (tm_check_buffer(ctx, k_src, src_need, err, err_cap, "kv_store.k_src") != TM_METAL_OK ||
        tm_check_buffer(ctx, v_src, src_need, err, err_cap, "kv_store.v_src") != TM_METAL_OK ||
        tm_check_buffer(ctx, k_cache, cache_need, err, err_cap, "kv_store.k_cache") != TM_METAL_OK ||
        tm_check_buffer(ctx, v_cache, cache_need, err, err_cap, "kv_store.v_cache") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"kv_store: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_kvstore];
        [enc setBuffer:k_src->buffer offset:0 atIndex:0];
        [enc setBuffer:v_src->buffer offset:0 atIndex:1];
        [enc setBuffer:k_cache->buffer offset:0 atIndex:2];
        [enc setBuffer:v_cache->buffer offset:0 atIndex:3];
        [f.keep addObject:k_src->buffer];
        [f.keep addObject:v_src->buffer];
        [f.keep addObject:k_cache->buffer];
        [f.keep addObject:v_cache->buffer];
        [enc setBytes:&pos length:sizeof(pos) atIndex:4];
        [enc setBytes:&cap length:sizeof(cap) atIndex:5];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
        [enc setBytes:&round_f16 length:sizeof(round_f16) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)kv_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)head_dim, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "kv_store");
    }
}

int tm_metal_attention_f32(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k_cache,
                           tm_metal_buffer *v_cache, tm_metal_buffer *out,
                           tm_metal_buffer *scratch, uint32_t heads, uint32_t pos,
                           uint32_t cap, uint32_t head_dim, uint32_t group, float scale,
                           char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "attention") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (heads == 0 || group == 0 || pos >= cap || head_dim == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS,
                       [NSString stringWithFormat:@"attention: heads %u, group %u, pos %u, cap %u",
                                                  heads, group, pos, cap]);
    }
    uint32_t kv_heads = heads / group;
    if (kv_heads * group != heads) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED, @"attention: heads not divisible by group");
    }
    size_t q_need = (size_t)heads * (size_t)head_dim * 4u;
    size_t cache_need = (size_t)kv_heads * (size_t)cap * (size_t)head_dim * 4u;
    size_t scratch_need = (size_t)heads * (size_t)cap * 4u;
    if (tm_check_buffer(ctx, q, q_need, err, err_cap, "attention.q") != TM_METAL_OK ||
        tm_check_buffer(ctx, k_cache, cache_need, err, err_cap, "attention.k_cache") != TM_METAL_OK ||
        tm_check_buffer(ctx, v_cache, cache_need, err, err_cap, "attention.v_cache") != TM_METAL_OK ||
        tm_check_buffer(ctx, out, q_need, err, err_cap, "attention.out") != TM_METAL_OK ||
        tm_check_buffer(ctx, scratch, scratch_need, err, err_cap, "attention.scratch") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"attention: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_attention];
        [enc setBuffer:q->buffer offset:0 atIndex:0];
        [enc setBuffer:k_cache->buffer offset:0 atIndex:1];
        [enc setBuffer:v_cache->buffer offset:0 atIndex:2];
        [enc setBuffer:out->buffer offset:0 atIndex:3];
        [enc setBuffer:scratch->buffer offset:0 atIndex:4];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k_cache->buffer];
        [f.keep addObject:v_cache->buffer];
        [f.keep addObject:out->buffer];
        [f.keep addObject:scratch->buffer];
        [enc setBytes:&pos length:sizeof(pos) atIndex:5];
        [enc setBytes:&cap length:sizeof(cap) atIndex:6];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
        [enc setBytes:&group length:sizeof(group) atIndex:8];
        [enc setBytes:&scale length:sizeof(scale) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "attention");
    }
}

int tm_metal_silu_mul_f32(tm_metal_ctx *ctx, tm_metal_buffer *gate, tm_metal_buffer *up,
                          tm_metal_buffer *out, uint32_t n, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "silu_mul") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"silu_mul: zero length");
    }
    size_t need = (size_t)n * 4u;
    if (tm_check_buffer(ctx, gate, need, err, err_cap, "silu_mul.gate") != TM_METAL_OK ||
        tm_check_buffer(ctx, up, need, err, err_cap, "silu_mul.up") != TM_METAL_OK ||
        tm_check_buffer(ctx, out, need, err, err_cap, "silu_mul.out") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"silu_mul: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_silu];
        [enc setBuffer:gate->buffer offset:0 atIndex:0];
        [enc setBuffer:up->buffer offset:0 atIndex:1];
        [enc setBuffer:out->buffer offset:0 atIndex:2];
        [f.keep addObject:gate->buffer];
        [f.keep addObject:up->buffer];
        [f.keep addObject:out->buffer];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        NSUInteger width = ctx->pso_silu.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "silu_mul");
    }
}

int tm_metal_rmsnorm_f16w(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *weight,
                          tm_metal_buffer *y, uint32_t rows, uint32_t cols, float eps,
                          char *err, size_t err_cap) {
    return tm_rmsnorm_dispatch(ctx, ctx->pso_rmsnorm_f16w, x, weight, y, rows, cols, eps, 2,
                               err, err_cap, "rmsnorm_f16w");
}

int tm_metal_f16_row_f32(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *out,
                         uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "f16_row") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (cols == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED, @"f16_row: zero cols");
    }
    size_t w_need = ((size_t)row + 1u) * (size_t)cols * 2u;
    if (tm_check_buffer(ctx, w, w_need, err, err_cap, "f16_row.weights") != TM_METAL_OK ||
        tm_check_buffer(ctx, out, ((size_t)out_off + cols) * 4u, err, err_cap, "f16_row.out") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"f16_row: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_f16row];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:out->buffer offset:0 atIndex:1];
        [f.keep addObject:w->buffer];
        [f.keep addObject:out->buffer];
        [enc setBytes:&row length:sizeof(row) atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&out_off length:sizeof(out_off) atIndex:4];
        NSUInteger tg = 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)cols, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "f16_row");
    }
}

static int tm_gemm_dispatch(tm_metal_ctx *ctx, id<MTLComputePipelineState> pso,
                            tm_metal_buffer *w, size_t w_need, tm_metal_buffer *x,
                            tm_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                            uint32_t x_off, uint32_t y_off, char *err, size_t err_cap,
                            const char *what) {
    if (tm_check_ctx(ctx, err, err_cap, what) != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0 || batch == 0 || batch > 32) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: rows %u, cols %u, batch %u (1..=32)",
                                                  what, rows, cols, batch]);
    }
    size_t x_need = ((size_t)x_off + (size_t)batch * (size_t)cols) * 4u;
    size_t y_need = ((size_t)y_off + (size_t)batch * (size_t)rows) * 4u;
    if (tm_check_buffer(ctx, w, w_need, err, err_cap, "gemm.weights") != TM_METAL_OK ||
        tm_check_buffer(ctx, x, x_need, err, err_cap, "gemm.x") != TM_METAL_OK ||
        tm_check_buffer(ctx, y, y_need, err, err_cap, "gemm.y") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL,
                           [NSString stringWithFormat:@"%s: no encoder", what]);
        }
        [enc setComputePipelineState:pso];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [enc setBuffer:y->buffer offset:0 atIndex:2];
        [f.keep addObject:w->buffer];
        [f.keep addObject:x->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&batch length:sizeof(batch) atIndex:5];
        [enc setBytes:&x_off length:sizeof(x_off) atIndex:6];
        [enc setBytes:&y_off length:sizeof(y_off) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, what);
    }
}

int tm_metal_q8_0_gemm(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *x,
                       tm_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                       uint32_t x_off, uint32_t y_off, char *err, size_t err_cap) {
    if (cols == 0 || (cols % 32u) != 0u) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       @"q8_0_gemm: cols must be a positive multiple of 32");
    }
    size_t w_need = (size_t)rows * ((size_t)cols / 32u) * 34u;
    return tm_gemm_dispatch(ctx, ctx->pso_q8gemm, w, w_need, x, y, rows, cols, batch, x_off,
                            y_off, err, err_cap, "q8_0_gemm");
}

int tm_metal_f16_gemm(tm_metal_ctx *ctx, tm_metal_buffer *w, tm_metal_buffer *x,
                      tm_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                      uint32_t x_off, uint32_t y_off, char *err, size_t err_cap) {
    size_t w_need = (size_t)rows * (size_t)cols * 2u;
    return tm_gemm_dispatch(ctx, ctx->pso_f16gemm, w, w_need, x, y, rows, cols, batch, x_off,
                            y_off, err, err_cap, "f16_gemm");
}

int tm_metal_rope_batch_f32(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k,
                            tm_metal_buffer *inv_freq, uint32_t start_pos, uint32_t n_tokens,
                            uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
                            char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "rope_batch") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (head_dim == 0 || (head_dim % 2u) != 0u || n_tokens == 0 || n_q_heads == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED, @"rope_batch: shape");
    }
    size_t q_need = (size_t)n_tokens * n_q_heads * head_dim * 4u;
    size_t k_need = (size_t)n_tokens * n_kv_heads * head_dim * 4u;
    if (tm_check_buffer(ctx, q, q_need, err, err_cap, "rope_batch.q") != TM_METAL_OK ||
        tm_check_buffer(ctx, k, k_need, err, err_cap, "rope_batch.k") != TM_METAL_OK ||
        tm_check_buffer(ctx, inv_freq, (size_t)(head_dim / 2u) * 4u, err, err_cap, "rope_batch.inv_freq") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"rope_batch: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_rope_batch];
        [enc setBuffer:q->buffer offset:0 atIndex:0];
        [enc setBuffer:k->buffer offset:0 atIndex:1];
        [enc setBuffer:inv_freq->buffer offset:0 atIndex:2];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k->buffer];
        [f.keep addObject:inv_freq->buffer];
        [enc setBytes:&start_pos length:sizeof(start_pos) atIndex:3];
        [enc setBytes:&n_tokens length:sizeof(n_tokens) atIndex:4];
        [enc setBytes:&n_q_heads length:sizeof(n_q_heads) atIndex:5];
        [enc setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:6];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens * (n_q_heads + n_kv_heads), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(head_dim / 2u, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "rope_batch");
    }
}

int tm_metal_kv_store_batch_f32(tm_metal_ctx *ctx, tm_metal_buffer *k_src, tm_metal_buffer *v_src,
                                tm_metal_buffer *k_cache, tm_metal_buffer *v_cache,
                                uint32_t n_tokens, uint32_t start_pos, uint32_t cap,
                                uint32_t head_dim, uint32_t kv_heads, uint32_t round_f16,
                                char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "kv_store_batch") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (n_tokens == 0 || start_pos + n_tokens > cap || head_dim == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"kv_store_batch: range");
    }
    size_t src_need = (size_t)n_tokens * kv_heads * head_dim * 4u;
    size_t cache_need = (size_t)kv_heads * cap * head_dim * 4u;
    if (tm_check_buffer(ctx, k_src, src_need, err, err_cap, "kv_store_batch.k_src") != TM_METAL_OK ||
        tm_check_buffer(ctx, v_src, src_need, err, err_cap, "kv_store_batch.v_src") != TM_METAL_OK ||
        tm_check_buffer(ctx, k_cache, cache_need, err, err_cap, "kv_store_batch.k_cache") != TM_METAL_OK ||
        tm_check_buffer(ctx, v_cache, cache_need, err, err_cap, "kv_store_batch.v_cache") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"kv_store_batch: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_kvstore_batch];
        [enc setBuffer:k_src->buffer offset:0 atIndex:0];
        [enc setBuffer:v_src->buffer offset:0 atIndex:1];
        [enc setBuffer:k_cache->buffer offset:0 atIndex:2];
        [enc setBuffer:v_cache->buffer offset:0 atIndex:3];
        [f.keep addObject:k_src->buffer];
        [f.keep addObject:v_src->buffer];
        [f.keep addObject:k_cache->buffer];
        [f.keep addObject:v_cache->buffer];
        [enc setBytes:&n_tokens length:sizeof(n_tokens) atIndex:4];
        [enc setBytes:&start_pos length:sizeof(start_pos) atIndex:5];
        [enc setBytes:&cap length:sizeof(cap) atIndex:6];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
        [enc setBytes:&kv_heads length:sizeof(kv_heads) atIndex:8];
        [enc setBytes:&round_f16 length:sizeof(round_f16) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens * kv_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)head_dim, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "kv_store_batch");
    }
}

int tm_metal_attention_batch_f32(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k_cache,
                                 tm_metal_buffer *v_cache, tm_metal_buffer *out,
                                 tm_metal_buffer *scratch, uint32_t heads, uint32_t n_tokens,
                                 uint32_t start_pos, uint32_t cap, uint32_t head_dim,
                                 uint32_t group, float scale, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "attention_batch") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (heads == 0 || group == 0 || n_tokens == 0 || head_dim == 0 ||
        start_pos + n_tokens > cap) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"attention_batch: shape");
    }
    uint32_t kv_heads = heads / group;
    if (kv_heads * group != heads) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED, @"attention_batch: heads/group");
    }
    size_t q_need = (size_t)n_tokens * heads * head_dim * 4u;
    size_t cache_need = (size_t)kv_heads * cap * head_dim * 4u;
    size_t scratch_need = (size_t)n_tokens * heads * cap * 4u;
    if (tm_check_buffer(ctx, q, q_need, err, err_cap, "attention_batch.q") != TM_METAL_OK ||
        tm_check_buffer(ctx, k_cache, cache_need, err, err_cap, "attention_batch.k_cache") != TM_METAL_OK ||
        tm_check_buffer(ctx, v_cache, cache_need, err, err_cap, "attention_batch.v_cache") != TM_METAL_OK ||
        tm_check_buffer(ctx, out, q_need, err, err_cap, "attention_batch.out") != TM_METAL_OK ||
        tm_check_buffer(ctx, scratch, scratch_need, err, err_cap, "attention_batch.scratch") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"attention_batch: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_attention_batch];
        [enc setBuffer:q->buffer offset:0 atIndex:0];
        [enc setBuffer:k_cache->buffer offset:0 atIndex:1];
        [enc setBuffer:v_cache->buffer offset:0 atIndex:2];
        [enc setBuffer:out->buffer offset:0 atIndex:3];
        [enc setBuffer:scratch->buffer offset:0 atIndex:4];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k_cache->buffer];
        [f.keep addObject:v_cache->buffer];
        [f.keep addObject:out->buffer];
        [f.keep addObject:scratch->buffer];
        [enc setBytes:&heads length:sizeof(heads) atIndex:5];
        [enc setBytes:&n_tokens length:sizeof(n_tokens) atIndex:6];
        [enc setBytes:&start_pos length:sizeof(start_pos) atIndex:7];
        [enc setBytes:&cap length:sizeof(cap) atIndex:8];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:9];
        [enc setBytes:&group length:sizeof(group) atIndex:10];
        [enc setBytes:&scale length:sizeof(scale) atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens * heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "attention_batch");
    }
}

int tm_metal_fence(tm_metal_ctx *ctx, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "fence") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (ctx->group_active) {
        return tm_fail(err, err_cap, TM_METAL_ERR_UNSUPPORTED,
                       @"fence: a dispatch group is open; end or abort it first");
    }
    return tm_wait_pending(ctx, err, err_cap);
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

uint64_t tm_metal_gpu_nanos_total(const tm_metal_ctx *ctx) {
    return ctx == NULL ? 0 : atomic_load_explicit(&ctx->gpu_nanos_total, memory_order_relaxed);
}

int tm_metal_profile_set(tm_metal_ctx *ctx, int enabled) {
    if (ctx == NULL) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    os_unfair_lock_lock(&ctx->profile_lock);
    ctx->profile_used = 0;
    memset(ctx->profile, 0, sizeof ctx->profile);
    os_unfair_lock_unlock(&ctx->profile_lock);
    atomic_store_explicit(&ctx->profiling, enabled ? 1 : 0, memory_order_relaxed);
    return TM_METAL_OK;
}

int tm_metal_profile_read(tm_metal_ctx *ctx, uint32_t index, char *name, size_t name_cap,
                          uint64_t *nanos, uint64_t *count) {
    if (ctx == NULL || name == NULL || name_cap == 0 || nanos == NULL || count == NULL) {
        return 0;
    }
    int found = 0;
    os_unfair_lock_lock(&ctx->profile_lock);
    if (index < ctx->profile_used) {
        strlcpy(name, ctx->profile[index].name, name_cap);
        *nanos = ctx->profile[index].nanos;
        *count = ctx->profile[index].count;
        found = 1;
    }
    os_unfair_lock_unlock(&ctx->profile_lock);
    return found;
}

uint64_t tm_metal_allocated_bytes(const tm_metal_ctx *ctx) {
    return ctx == NULL ? 0 : atomic_load_explicit(&ctx->allocated, memory_order_relaxed);
}

uint64_t tm_metal_in_flight_bytes(const tm_metal_ctx *ctx) {
    return ctx == NULL ? 0 : atomic_load_explicit(&ctx->leased, memory_order_relaxed);
}

/* ---------------- Q35-2A: Gated DeltaNet operator dispatches ---------------- */

int tm_metal_qwen35_conv_step(tm_metal_ctx *ctx, tm_metal_buffer *hist, tm_metal_buffer *x,
                              tm_metal_buffer *w, tm_metal_buffer *y, uint32_t channels,
                              uint32_t kernel, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "qwen35_conv") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (channels == 0 || kernel == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"qwen35_conv: zero geometry");
    }
    if (tm_check_buffer(ctx, hist, (size_t)(kernel - 1) * channels * 4u, err, err_cap,
                        "qwen35_conv.hist") != TM_METAL_OK ||
        tm_check_buffer(ctx, x, (size_t)channels * 4u, err, err_cap, "qwen35_conv.x") != TM_METAL_OK ||
        tm_check_buffer(ctx, w, (size_t)kernel * channels * 4u, err, err_cap,
                        "qwen35_conv.w") != TM_METAL_OK ||
        tm_check_buffer(ctx, y, (size_t)channels * 4u, err, err_cap, "qwen35_conv.y") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"qwen35_conv: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_conv];
        [enc setBuffer:hist->buffer offset:0 atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [enc setBuffer:w->buffer offset:0 atIndex:2];
        [enc setBuffer:y->buffer offset:0 atIndex:3];
        [f.keep addObject:hist->buffer];
        [f.keep addObject:x->buffer];
        [f.keep addObject:w->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&channels length:sizeof(channels) atIndex:4];
        [enc setBytes:&kernel length:sizeof(kernel) atIndex:5];
        NSUInteger width = ctx->pso_q35_conv.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)channels, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_conv");
    }
}

int tm_metal_qwen35_conv_shift(tm_metal_ctx *ctx, tm_metal_buffer *hist, tm_metal_buffer *x,
                               uint32_t channels, uint32_t kernel, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "qwen35_conv_shift") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (channels == 0 || kernel == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"qwen35_conv_shift: zero geometry");
    }
    if (tm_check_buffer(ctx, hist, (size_t)(kernel - 1) * channels * 4u, err, err_cap,
                        "qwen35_conv_shift.hist") != TM_METAL_OK ||
        tm_check_buffer(ctx, x, (size_t)channels * 4u, err, err_cap,
                        "qwen35_conv_shift.x") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"qwen35_conv_shift: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_conv_shift];
        [enc setBuffer:hist->buffer offset:0 atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [f.keep addObject:hist->buffer];
        [f.keep addObject:x->buffer];
        [enc setBytes:&channels length:sizeof(channels) atIndex:2];
        [enc setBytes:&kernel length:sizeof(kernel) atIndex:3];
        NSUInteger width = ctx->pso_q35_conv_shift.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)channels, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_conv_shift");
    }
}

int tm_metal_qwen35_l2norm_heads(tm_metal_ctx *ctx, tm_metal_buffer *x, tm_metal_buffer *y,
                                 uint32_t rows, uint32_t cols, float eps, char *err,
                                 size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "qwen35_l2norm") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"qwen35_l2norm: zero geometry");
    }
    if (tm_check_buffer(ctx, x, (size_t)rows * cols * 4u, err, err_cap, "qwen35_l2norm.x") != TM_METAL_OK ||
        tm_check_buffer(ctx, y, (size_t)rows * cols * 4u, err, err_cap, "qwen35_l2norm.y") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"qwen35_l2norm: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_l2norm];
        [enc setBuffer:x->buffer offset:0 atIndex:0];
        [enc setBuffer:y->buffer offset:0 atIndex:1];
        [f.keep addObject:x->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&cols length:sizeof(cols) atIndex:2];
        [enc setBytes:&eps length:sizeof(eps) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_l2norm");
    }
}

int tm_metal_qwen35_beta_decay(tm_metal_ctx *ctx, tm_metal_buffer *raw_beta, tm_metal_buffer *alpha,
                               tm_metal_buffer *dt_bias, tm_metal_buffer *ssm_a,
                               tm_metal_buffer *beta_out, tm_metal_buffer *gate_out, uint32_t n,
                               char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "qwen35_beta_decay") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"qwen35_beta_decay: zero length");
    }
    size_t need = (size_t)n * 4u;
    if (tm_check_buffer(ctx, raw_beta, need, err, err_cap, "qwen35_beta_decay.beta") != TM_METAL_OK ||
        tm_check_buffer(ctx, alpha, need, err, err_cap, "qwen35_beta_decay.alpha") != TM_METAL_OK ||
        tm_check_buffer(ctx, dt_bias, need, err, err_cap, "qwen35_beta_decay.dt") != TM_METAL_OK ||
        tm_check_buffer(ctx, ssm_a, need, err, err_cap, "qwen35_beta_decay.a") != TM_METAL_OK ||
        tm_check_buffer(ctx, beta_out, need, err, err_cap, "qwen35_beta_decay.beta_out") != TM_METAL_OK ||
        tm_check_buffer(ctx, gate_out, need, err, err_cap, "qwen35_beta_decay.gate_out") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"qwen35_beta_decay: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_beta_decay];
        [enc setBuffer:raw_beta->buffer offset:0 atIndex:0];
        [enc setBuffer:alpha->buffer offset:0 atIndex:1];
        [enc setBuffer:dt_bias->buffer offset:0 atIndex:2];
        [enc setBuffer:ssm_a->buffer offset:0 atIndex:3];
        [enc setBuffer:beta_out->buffer offset:0 atIndex:4];
        [enc setBuffer:gate_out->buffer offset:0 atIndex:5];
        [f.keep addObject:raw_beta->buffer];
        [f.keep addObject:alpha->buffer];
        [f.keep addObject:dt_bias->buffer];
        [f.keep addObject:ssm_a->buffer];
        [f.keep addObject:beta_out->buffer];
        [f.keep addObject:gate_out->buffer];
        [enc setBytes:&n length:sizeof(n) atIndex:6];
        NSUInteger width = ctx->pso_q35_beta_decay.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_beta_decay");
    }
}

int tm_metal_qwen35_gdn_step(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k,
                             tm_metal_buffer *v, tm_metal_buffer *gate, tm_metal_buffer *beta,
                             tm_metal_buffer *state, tm_metal_buffer *out, uint32_t heads,
                             uint32_t k_dim, uint32_t v_dim, float q_scale, char *err,
                             size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "qwen35_gdn_step") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (heads == 0 || k_dim == 0 || v_dim == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"qwen35_gdn_step: zero geometry");
    }
    size_t qk = (size_t)heads * k_dim * 4u;
    size_t vv = (size_t)heads * v_dim * 4u;
    if (tm_check_buffer(ctx, q, qk, err, err_cap, "gdn_step.q") != TM_METAL_OK ||
        tm_check_buffer(ctx, k, qk, err, err_cap, "gdn_step.k") != TM_METAL_OK ||
        tm_check_buffer(ctx, v, vv, err, err_cap, "gdn_step.v") != TM_METAL_OK ||
        tm_check_buffer(ctx, gate, (size_t)heads * 4u, err, err_cap, "gdn_step.gate") != TM_METAL_OK ||
        tm_check_buffer(ctx, beta, (size_t)heads * 4u, err, err_cap, "gdn_step.beta") != TM_METAL_OK ||
        tm_check_buffer(ctx, state, (size_t)heads * k_dim * v_dim * 4u, err, err_cap,
                        "gdn_step.state") != TM_METAL_OK ||
        tm_check_buffer(ctx, out, vv, err, err_cap, "gdn_step.out") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"gdn_step: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_gdn_step];
        [enc setBuffer:q->buffer offset:0 atIndex:0];
        [enc setBuffer:k->buffer offset:0 atIndex:1];
        [enc setBuffer:v->buffer offset:0 atIndex:2];
        [enc setBuffer:gate->buffer offset:0 atIndex:3];
        [enc setBuffer:beta->buffer offset:0 atIndex:4];
        [enc setBuffer:state->buffer offset:0 atIndex:5];
        [enc setBuffer:out->buffer offset:0 atIndex:6];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k->buffer];
        [f.keep addObject:v->buffer];
        [f.keep addObject:gate->buffer];
        [f.keep addObject:beta->buffer];
        [f.keep addObject:state->buffer];
        [f.keep addObject:out->buffer];
        uint32_t kd = k_dim;
        uint32_t vd = v_dim;
        [enc setBytes:&kd length:sizeof(kd) atIndex:7];
        [enc setBytes:&vd length:sizeof(vd) atIndex:8];
        [enc setBytes:&q_scale length:sizeof(q_scale) atIndex:9];
        /* One thread per (value column, head); columns of one head share
         * a threadgroup row so a SIMD group reads contiguous state. */
        NSUInteger width = ctx->pso_q35_gdn_step.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width : 32;
        if (tg > (NSUInteger)v_dim) {
            tg = (NSUInteger)v_dim;
        }
        [enc dispatchThreads:MTLSizeMake((NSUInteger)v_dim, (NSUInteger)heads, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_gdn_step");
    }
}

int tm_metal_qwen35_gated_norm(tm_metal_ctx *ctx, tm_metal_buffer *o, tm_metal_buffer *z,
                               tm_metal_buffer *weight, tm_metal_buffer *out, uint32_t rows,
                               uint32_t cols, float eps, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "qwen35_gated_norm") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"qwen35_gated_norm: zero geometry");
    }
    size_t need = (size_t)rows * cols * 4u;
    if (tm_check_buffer(ctx, o, need, err, err_cap, "gated_norm.o") != TM_METAL_OK ||
        tm_check_buffer(ctx, z, need, err, err_cap, "gated_norm.z") != TM_METAL_OK ||
        tm_check_buffer(ctx, weight, (size_t)cols * 4u, err, err_cap, "gated_norm.w") != TM_METAL_OK ||
        tm_check_buffer(ctx, out, need, err, err_cap, "gated_norm.out") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"gated_norm: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_gated_norm];
        [enc setBuffer:o->buffer offset:0 atIndex:0];
        [enc setBuffer:z->buffer offset:0 atIndex:1];
        [enc setBuffer:weight->buffer offset:0 atIndex:2];
        [enc setBuffer:out->buffer offset:0 atIndex:3];
        [f.keep addObject:o->buffer];
        [f.keep addObject:z->buffer];
        [f.keep addObject:weight->buffer];
        [f.keep addObject:out->buffer];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&eps length:sizeof(eps) atIndex:5];
        NSUInteger width = ctx->pso_q35_gated_norm.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width : 32;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_gated_norm");
    }
}

/* ---------------- Q35-2B: full-attention operator dispatches ---------------- */

int tm_metal_qwen35_q_gate_split(tm_metal_ctx *ctx, tm_metal_buffer *qg, tm_metal_buffer *q_out,
                                 tm_metal_buffer *gate_out, uint32_t heads, uint32_t head_dim,
                                 char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "qg_split") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (heads == 0 || head_dim == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"qg_split: zero geometry");
    }
    size_t row = (size_t)heads * head_dim * 4u;
    if (tm_check_buffer(ctx, qg, row * 2u, err, err_cap, "qg_split.qg") != TM_METAL_OK ||
        tm_check_buffer(ctx, q_out, row, err, err_cap, "qg_split.q") != TM_METAL_OK ||
        tm_check_buffer(ctx, gate_out, row, err, err_cap, "qg_split.gate") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"qg_split: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_qg_split];
        [enc setBuffer:qg->buffer offset:0 atIndex:0];
        [enc setBuffer:q_out->buffer offset:0 atIndex:1];
        [enc setBuffer:gate_out->buffer offset:0 atIndex:2];
        [f.keep addObject:qg->buffer];
        [f.keep addObject:q_out->buffer];
        [f.keep addObject:gate_out->buffer];
        uint32_t hd = head_dim;
        [enc setBytes:&hd length:sizeof(hd) atIndex:3];
        NSUInteger width = ctx->pso_q35_qg_split.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        NSUInteger total = (NSUInteger)heads * head_dim;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_qg_split");
    }
}

int tm_metal_qwen35_rope_partial(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k,
                                 tm_metal_buffer *inv_freq, uint32_t pos, uint32_t n_q_heads,
                                 uint32_t n_kv_heads, uint32_t head_dim, uint32_t n_rot,
                                 char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "q35_rope_partial") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (head_dim == 0 || n_rot == 0 || n_rot > head_dim || n_rot % 2u != 0u) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS,
                       @"q35_rope_partial: n_rot must be an even prefix of head_dim");
    }
    if (tm_check_buffer(ctx, q, (size_t)n_q_heads * head_dim * 4u, err, err_cap,
                        "rope.q") != TM_METAL_OK ||
        tm_check_buffer(ctx, k, (size_t)n_kv_heads * head_dim * 4u, err, err_cap,
                        "rope.k") != TM_METAL_OK ||
        tm_check_buffer(ctx, inv_freq, (size_t)(n_rot / 2u) * 4u, err, err_cap,
                        "rope.inv_freq") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"q35_rope_partial: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_rope_partial];
        [enc setBuffer:q->buffer offset:0 atIndex:0];
        [enc setBuffer:k->buffer offset:0 atIndex:1];
        [enc setBuffer:inv_freq->buffer offset:0 atIndex:2];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k->buffer];
        [f.keep addObject:inv_freq->buffer];
        [enc setBytes:&pos length:sizeof(pos) atIndex:3];
        [enc setBytes:&n_q_heads length:sizeof(n_q_heads) atIndex:4];
        [enc setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:5];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
        [enc setBytes:&n_rot length:sizeof(n_rot) atIndex:7];
        NSUInteger total = (NSUInteger)(n_q_heads + n_kv_heads);
        [enc dispatchThreadgroups:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_rope_partial");
    }
}

int tm_metal_qwen35_kv_store_f16(tm_metal_ctx *ctx, tm_metal_buffer *k_src, tm_metal_buffer *v_src,
                                 tm_metal_buffer *k_cache, tm_metal_buffer *v_cache, uint32_t pos,
                                 uint32_t cap, uint32_t head_dim, uint32_t kv_heads, char *err,
                                 size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "kv_store_f16") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (head_dim == 0 || kv_heads == 0 || pos >= cap) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"kv_store_f16: geometry out of range");
    }
    size_t src = (size_t)kv_heads * head_dim * 4u;
    size_t dst = (size_t)kv_heads * cap * head_dim * 2u;
    if (tm_check_buffer(ctx, k_src, src, err, err_cap, "kv_store_f16.k_src") != TM_METAL_OK ||
        tm_check_buffer(ctx, v_src, src, err, err_cap, "kv_store_f16.v_src") != TM_METAL_OK ||
        tm_check_buffer(ctx, k_cache, dst, err, err_cap, "kv_store_f16.k_cache") != TM_METAL_OK ||
        tm_check_buffer(ctx, v_cache, dst, err, err_cap, "kv_store_f16.v_cache") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"kv_store_f16: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_kv_store_f16];
        [enc setBuffer:k_src->buffer offset:0 atIndex:0];
        [enc setBuffer:v_src->buffer offset:0 atIndex:1];
        [enc setBuffer:k_cache->buffer offset:0 atIndex:2];
        [enc setBuffer:v_cache->buffer offset:0 atIndex:3];
        [f.keep addObject:k_src->buffer];
        [f.keep addObject:v_src->buffer];
        [f.keep addObject:k_cache->buffer];
        [f.keep addObject:v_cache->buffer];
        [enc setBytes:&pos length:sizeof(pos) atIndex:4];
        [enc setBytes:&cap length:sizeof(cap) atIndex:5];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
        /* A full row per head: head_dim can exceed the 128-thread group. */
        [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_heads * head_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_kv_store_f16");
    }
}

int tm_metal_qwen35_attention_f16(tm_metal_ctx *ctx, tm_metal_buffer *q, tm_metal_buffer *k_cache,
                                  tm_metal_buffer *v_cache, tm_metal_buffer *out,
                                  tm_metal_buffer *scratch, uint32_t heads, uint32_t pos,
                                  uint32_t cap, uint32_t head_dim, uint32_t group, float scale,
                                  char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "attn_f16") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (heads == 0 || head_dim == 0 || group == 0 || pos >= cap || heads % group != 0u) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"attn_f16: geometry out of range");
    }
    size_t qb = (size_t)heads * head_dim * 4u;
    size_t cache = (size_t)(heads / group) * cap * head_dim * 2u;
    if (tm_check_buffer(ctx, q, qb, err, err_cap, "attn_f16.q") != TM_METAL_OK ||
        tm_check_buffer(ctx, k_cache, cache, err, err_cap, "attn_f16.k") != TM_METAL_OK ||
        tm_check_buffer(ctx, v_cache, cache, err, err_cap, "attn_f16.v") != TM_METAL_OK ||
        tm_check_buffer(ctx, out, qb, err, err_cap, "attn_f16.out") != TM_METAL_OK ||
        tm_check_buffer(ctx, scratch, (size_t)heads * cap * 4u, err, err_cap,
                        "attn_f16.scratch") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"attn_f16: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_attention_f16];
        [enc setBuffer:q->buffer offset:0 atIndex:0];
        [enc setBuffer:k_cache->buffer offset:0 atIndex:1];
        [enc setBuffer:v_cache->buffer offset:0 atIndex:2];
        [enc setBuffer:out->buffer offset:0 atIndex:3];
        [enc setBuffer:scratch->buffer offset:0 atIndex:4];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k_cache->buffer];
        [f.keep addObject:v_cache->buffer];
        [f.keep addObject:out->buffer];
        [f.keep addObject:scratch->buffer];
        uint32_t capv = cap;
        uint32_t hdv = head_dim;
        uint32_t gv = group;
        [enc setBytes:&pos length:sizeof(pos) atIndex:5];
        [enc setBytes:&capv length:sizeof(capv) atIndex:6];
        [enc setBytes:&hdv length:sizeof(hdv) atIndex:7];
        [enc setBytes:&gv length:sizeof(gv) atIndex:8];
        [enc setBytes:&scale length:sizeof(scale) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_attention_f16");
    }
}

int tm_metal_qwen35_attn_gate(tm_metal_ctx *ctx, tm_metal_buffer *attn, tm_metal_buffer *gate,
                              uint32_t n, char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "attn_gate") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"attn_gate: zero length");
    }
    size_t need = (size_t)n * 4u;
    if (tm_check_buffer(ctx, attn, need, err, err_cap, "attn_gate.attn") != TM_METAL_OK ||
        tm_check_buffer(ctx, gate, need, err, err_cap, "attn_gate.gate") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"attn_gate: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_attn_gate];
        [enc setBuffer:attn->buffer offset:0 atIndex:0];
        [enc setBuffer:gate->buffer offset:0 atIndex:1];
        [f.keep addObject:attn->buffer];
        [f.keep addObject:gate->buffer];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        NSUInteger width = ctx->pso_q35_attn_gate.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_attn_gate");
    }
}

int tm_metal_qwen35_silu(tm_metal_ctx *ctx, tm_metal_buffer *x, uint32_t n, char *err,
                         size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "q35_silu") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"q35_silu: zero length");
    }
    if (tm_check_buffer(ctx, x, (size_t)n * 4u, err, err_cap, "q35_silu.x") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"q35_silu: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_silu];
        [enc setBuffer:x->buffer offset:0 atIndex:0];
        [f.keep addObject:x->buffer];
        [enc setBytes:&n length:sizeof(n) atIndex:1];
        NSUInteger width = ctx->pso_q35_silu.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_silu");
    }
}

int tm_metal_qwen35_split_qkv(tm_metal_ctx *ctx, tm_metal_buffer *y, tm_metal_buffer *q,
                              tm_metal_buffer *k, tm_metal_buffer *v, uint32_t width, char *err,
                              size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "q35_split_qkv") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (width == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"q35_split_qkv: zero width");
    }
    size_t qk = (size_t)width * 4u;
    if (tm_check_buffer(ctx, y, qk * 3u, err, err_cap, "split_qkv.y") != TM_METAL_OK ||
        tm_check_buffer(ctx, q, qk, err, err_cap, "split_qkv.q") != TM_METAL_OK ||
        tm_check_buffer(ctx, k, qk, err, err_cap, "split_qkv.k") != TM_METAL_OK ||
        tm_check_buffer(ctx, v, qk, err, err_cap, "split_qkv.v") != TM_METAL_OK) {
        return TM_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        tm_frame f;
        id<MTLComputeCommandEncoder> enc = tm_frame_open(ctx, &f);
        if (enc == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"q35_split_qkv: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_split_qkv];
        [enc setBuffer:y->buffer offset:0 atIndex:0];
        [enc setBuffer:q->buffer offset:0 atIndex:1];
        [enc setBuffer:k->buffer offset:0 atIndex:2];
        [enc setBuffer:v->buffer offset:0 atIndex:3];
        [f.keep addObject:y->buffer];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k->buffer];
        [f.keep addObject:v->buffer];
        [enc setBytes:&width length:sizeof(width) atIndex:4];
        NSUInteger tw = ctx->pso_q35_split_qkv.threadExecutionWidth;
        NSUInteger tg = tw > 0 ? tw * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)width, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return tm_frame_commit(ctx, &f, err, err_cap, "qwen35_split_qkv");
    }
}

int tm_metal_buffer_zero(tm_metal_ctx *ctx, tm_metal_buffer *buffer, size_t off, size_t len,
                         char *err, size_t err_cap) {
    if (tm_check_ctx(ctx, err, err_cap, "buffer_zero") != TM_METAL_OK) {
        return TM_METAL_ERR_UNSUPPORTED;
    }
    if (buffer == NULL || buffer->owner != ctx) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"buffer_zero: foreign buffer");
    }
    if (len == 0) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"buffer_zero: zero length");
    }
    if (off + len > buffer->len) {
        return tm_fail(err, err_cap, TM_METAL_ERR_BOUNDS, @"buffer_zero: range outside the buffer");
    }
    @autoreleasepool {
        if (buffer->shared) {
            /* Shared storage is host-coherent: zero in place, no staging. */
            memset((char *)buffer->buffer.contents + off, 0, len);
            return TM_METAL_OK;
        }
        /* Private storage: GPU-side fill (cold path, synchronous). */
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"buffer_zero: no command buffer");
        }
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit fillBuffer:buffer->buffer range:NSMakeRange(off, len) value:0];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            return tm_fail(err, err_cap, TM_METAL_ERR_INTERNAL, @"buffer_zero: GPU fill failed");
        }
        return TM_METAL_OK;
    }
}
