// reference_worker_stub.cpp — compilable reference of the worker ABI.
//
// This stub does NOT execute any model. It exists so the ABI stays honest:
// it compiles against tuckmill_worker_abi.h, refuses profiles whose kernels
// are unimplemented, and emits clearly-labeled synthetic tokens, exactly like
// the Rust reference worker used by this build.
//
// Build:  clang++ -std=c++17 -Wall -Wextra -c reference_worker_stub.cpp
// (see scripts/build.sh for the invocation used in CI-less local checks)

#include "tuckmill_worker_abi.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <thread>

namespace {

struct WorkerState {
    std::string plan_json;
    std::atomic<bool> cancel{false};
    std::atomic<bool> paused{false};
    bool loaded = false;
};

int parse_plan_mode(const std::string &plan_json) {
    // The stub only needs to reject "rejected" plans; a real executor parses
    // the full ExecutionPlan schema (see PROTOCOL.md for the JSON contract).
    return plan_json.find("\"mode\":\"rejected\"") == std::string::npos ? 1 : 0;
}

} // namespace

extern "C" {

tuckmill_worker_t *tuckmill_worker_create(const char *plan_json,
                                          size_t plan_json_len,
                                          tuckmill_status_t *status) {
    if (plan_json == nullptr || status == nullptr) {
        if (status) *status = TM_ERR_PLAN;
        return nullptr;
    }
    auto *state = new (std::nothrow) WorkerState();
    if (state == nullptr) {
        *status = TM_ERR_MEMORY;
        return nullptr;
    }
    state->plan_json.assign(plan_json, plan_json_len);
    if (!parse_plan_mode(state->plan_json)) {
        delete state;
        *status = TM_ERR_PLAN;
        return nullptr;
    }
    *status = TM_OK;
    return reinterpret_cast<tuckmill_worker_t *>(state);
}

tuckmill_status_t tuckmill_worker_load(tuckmill_worker_t *w, const char **reason_out) {
    auto *state = reinterpret_cast<WorkerState *>(w);
    if (state == nullptr) return TM_ERR_INTERNAL;
    state->loaded = true;
    if (reason_out) {
        *reason_out = "reference stub: synthetic output only; no model weights executed";
    }
    return TM_OK;
}

tuckmill_status_t tuckmill_worker_generate(tuckmill_worker_t *w,
                                           const tuckmill_generate_params_t *params,
                                           const tuckmill_callbacks_t *callbacks,
                                           uint64_t *committed_out) {
    auto *state = reinterpret_cast<WorkerState *>(w);
    if (state == nullptr || params == nullptr || committed_out == nullptr) {
        return TM_ERR_INTERNAL;
    }
    if (!state->loaded) return TM_ERR_PLAN;
    *committed_out = 0;

    if (callbacks && callbacks->on_state) {
        callbacks->on_state(callbacks->ctx, TM_STATE_WARMING);
        callbacks->on_state(callbacks->ctx, TM_STATE_PREFILLING);
    }

    // IMPORTANT (ABI contract): a real executor must return TM_ERR_TRANSFORM
    // or TM_ERR_KERNELS here when a required kernel/transform is unavailable.
    // This reference stub emulates the Rust reference worker: it emits
    // clearly-labeled synthetic tokens so the pipeline can be exercised. It
    // must never be presented as model output (the Load reason string says so).

    if (callbacks && callbacks->on_state) {
        callbacks->on_state(callbacks->ctx, TM_STATE_GENERATING);
    }
    for (uint64_t i = 0; i < params->max_output_tokens; ++i) {
        while (state->paused.load() && !state->cancel.load()) {
            if (callbacks && callbacks->on_state) {
                callbacks->on_state(callbacks->ctx, TM_STATE_PAUSED);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (state->cancel.load()) {
            return TM_ERR_CANCELLED;
        }
        if (callbacks && callbacks->on_state) {
            callbacks->on_state(callbacks->ctx, TM_STATE_GENERATING);
        }
        char text[32];
        int n = std::snprintf(text, sizeof(text), "[ref:%llu]",
                              static_cast<unsigned long long>(i));
        if (callbacks && callbacks->on_token) {
            tuckmill_status_t rc =
                callbacks->on_token(callbacks->ctx, i, text, static_cast<size_t>(n));
            if (rc != TM_OK) {
                return TM_ERR_CANCELLED;
            }
        }
        *committed_out += 1;
    }
    return TM_OK;
}

void tuckmill_worker_cancel(tuckmill_worker_t *w) {
    auto *state = reinterpret_cast<WorkerState *>(w);
    if (state) state->cancel.store(true);
}

void tuckmill_worker_set_paused(tuckmill_worker_t *w, uint32_t paused) {
    auto *state = reinterpret_cast<WorkerState *>(w);
    if (state) state->paused.store(paused != 0);
}

tuckmill_status_t tuckmill_worker_info(const tuckmill_worker_t *w,
                                       tuckmill_engine_info_t *info_out) {
    if (w == nullptr || info_out == nullptr) return TM_ERR_INTERNAL;
    info_out->engine_name = "reference-test";
    info_out->build_id = "c++-stub";
    info_out->abi_version = TUCKMILL_ABI_VERSION;
    info_out->certified = 0;
    return TM_OK;
}

tuckmill_status_t tuckmill_worker_footprint(const tuckmill_worker_t *w,
                                            uint64_t *bytes_out) {
    if (w == nullptr || bytes_out == nullptr) return TM_ERR_INTERNAL;
    *bytes_out = 0; // Unknown: report 0, never guess.
    return TM_OK;
}

void tuckmill_worker_destroy(tuckmill_worker_t *w) {
    auto *state = reinterpret_cast<WorkerState *>(w);
    if (state == nullptr) return;
    state->cancel.store(true);
    delete state;
}

} // extern "C"
