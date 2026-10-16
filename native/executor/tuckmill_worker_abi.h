/*
 * tuckmill_worker_abi.h — narrow C ABI for the Tuckmill execution worker.
 *
 * Version: ABI 1 (must match tuckmill_manifest::BACKEND_ABI_VERSION and the
 * `abi` field of the Hello/HelloAck protocol frames).
 *
 * Design rules (PRD sections 6, 11.3, 15):
 *  - The supervisor owns admission, queues, cancellation policy, and
 *    lifecycle; this ABI only exposes execution.
 *  - No model state or tensor computation crosses the ABI boundary: callbacks
 *    deliver committed tokens and state transitions, nothing else.
 *  - Every function returns a tuckmill_status_t; TM_OK is zero. No function
 *    aborts, and no function may retain pointers after it returns unless the
 *    contract below says so explicitly.
 *  - The worker must never stream draft/speculative text as committed output.
 *
 * The current build's worker implements this ABI in Rust behind the framed
 * protocol (see PROTOCOL.md); native/executor/reference_worker_stub.cpp is a
 * compilable reference of the coupling a real C++/Objective-C++ executor
 * must provide.
 */

#ifndef TUCKMILL_WORKER_ABI_H
#define TUCKMILL_WORKER_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TUCKMILL_ABI_VERSION 1u

typedef enum tuckmill_status {
    TM_OK = 0,
    /* The requested operation is outside the admitted plan. */
    TM_ERR_PLAN = 1,
    /* Context or output overflow; nothing was executed. */
    TM_ERR_CONTEXT = 2,
    /* Designated execution kernels are not available in this build. */
    TM_ERR_KERNELS = 3,
    /* A required transform (e.g. activation rotation) is unimplemented. */
    TM_ERR_TRANSFORM = 4,
    /* Artifact integrity failure discovered during load. */
    TM_ERR_INTEGRITY = 5,
    /* Memory admission failed inside the worker. */
    TM_ERR_MEMORY = 6,
    /* Storage backing the artifact was lost. */
    TM_ERR_STORAGE = 7,
    /* Cancellation was requested and honored at a safe boundary. */
    TM_ERR_CANCELLED = 8,
    /* Internal invariant violation. */
    TM_ERR_INTERNAL = 9,
} tuckmill_status_t;

typedef enum tuckmill_run_state {
    TM_STATE_WARMING = 0,
    TM_STATE_PREFILLING = 1,
    TM_STATE_GENERATING = 2,
    TM_STATE_THROTTLING = 3,
    TM_STATE_PAUSED = 4,
} tuckmill_run_state_t;

/*
 * Engine descriptor. `certified` must be 0 unless the named tests passed for
 * exactly this artifact/profile/device configuration; a worker presenting
 * non-model output as model output is a release-blocking defect.
 */
typedef struct tuckmill_engine_info {
    const char *engine_name;      /* e.g. "reference-test" or "native"      */
    const char *build_id;         /* version + build identity               */
    uint32_t abi_version;         /* must equal TUCKMILL_ABI_VERSION        */
    uint32_t certified;           /* 0/1                                    */
} tuckmill_engine_info_t;

/*
 * Execution callbacks. All callbacks are invoked from the generation thread.
 * Returning TM_OK from on_token acknowledges the committed token; any other
 * value requests a stop at that boundary (the worker reports TM_ERR_CANCELLED
 * if stop_requested is the reason).
 */
typedef tuckmill_status_t (*tuckmill_on_token_fn)(void *ctx,
                                                  uint64_t index,
                                                  const char *utf8_text,
                                                  size_t utf8_len);

typedef void (*tuckmill_on_state_fn)(void *ctx, tuckmill_run_state_t state);

typedef struct tuckmill_callbacks {
    void *ctx;
    tuckmill_on_token_fn on_token;
    tuckmill_on_state_fn on_state;
} tuckmill_callbacks_t;

typedef struct tuckmill_generate_params {
    const uint32_t *prompt_tokens; /* exact token IDs, not text              */
    size_t prompt_len;
    uint64_t max_output_tokens;
    const char *const *stop;       /* NULL-terminated array, may be NULL     */
    uint64_t seed;
} tuckmill_generate_params_t;

/* Opaque worker handle. */
typedef struct tuckmill_worker tuckmill_worker_t;

/* ---- lifecycle ---- */

/* Must be the first call. Returns NULL with *status set on failure. */
tuckmill_worker_t *tuckmill_worker_create(const char *plan_json,
                                          size_t plan_json_len,
                                          tuckmill_status_t *status);

/*
 * Load the plan's artifact. Executors that stream from storage must verify
 * artifact identity hashes here (tensor index + metadata) before executing;
 * a mismatch is TM_ERR_INTEGRITY. `reason_out` receives a static string.
 */
tuckmill_status_t tuckmill_worker_load(tuckmill_worker_t *w,
                                       const char **reason_out);

/* ---- generation ---- */

/*
 * Generate. Honors cancellation (tuckmill_worker_cancel) at committed-token
 * boundaries and reports state transitions through callbacks. Committed
 * tokens only: never draft text.
 */
tuckmill_status_t tuckmill_worker_generate(tuckmill_worker_t *w,
                                           const tuckmill_generate_params_t *params,
                                           const tuckmill_callbacks_t *callbacks,
                                           uint64_t *committed_out);

/* Cooperative cancellation; safe to call from another thread. */
void tuckmill_worker_cancel(tuckmill_worker_t *w);

/* Safe-boundary pause control (governor). Safe to call from another thread. */
void tuckmill_worker_set_paused(tuckmill_worker_t *w, uint32_t paused);

/* ---- introspection / teardown ---- */

tuckmill_status_t tuckmill_worker_info(const tuckmill_worker_t *w,
                                       tuckmill_engine_info_t *info_out);

/* Process footprint attributable to the worker, when measurable. */
tuckmill_status_t tuckmill_worker_footprint(const tuckmill_worker_t *w,
                                            uint64_t *bytes_out);

/*
 * Destroy the worker. Blocks until in-flight work has drained or was safely
 * cancelled; buffers are released only after completion fences. Never returns
 * with work still executing in another thread.
 */
void tuckmill_worker_destroy(tuckmill_worker_t *w);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TUCKMILL_WORKER_ABI_H */
