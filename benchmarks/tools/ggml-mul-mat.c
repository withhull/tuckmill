// Times the pinned llama.cpp/ggml Metal mul_mat (Q8_0 weights x f32
// activations) in isolation, for comparison with TuckCore's matrix-unit
// GEMM (`q8_0_gemm_mma_grouped_wall_timing_when_requested` in
// crates/metal/src/tests.rs, which uses the same method: 16 serial
// dispatches, wall time per dispatch). Measurement only; never a gate.
//
// Build against the pinned reference (scripts/fetch-llama.sh):
//   L=.deps/llama
//   clang -O2 -I$L/include benchmarks/tools/ggml-mul-mat.c -L$L -lggml -lggml-base \
//         -Wl,-rpath,$L -o /tmp/ggml-mul-mat
// Usage: ggml-mul-mat <rows> <cols> <tokens> [nodes=16] [iters=20] [chain]
// `chain` (rows == cols only) feeds each node's output to the next node,
// so nodes are dependent as in a model graph; otherwise nodes are
// independent and ggml may overlap them.
// Run (serial nodes, like a dependent model graph):
//   GGML_METAL_CONCURRENCY_DISABLE=1 /tmp/ggml-mul-mat 3584 1024 128
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: mm rows cols tokens [nodes] [iters]\n"); return 2; }
    const int rows = atoi(argv[1]), cols = atoi(argv[2]), toks = atoi(argv[3]);
    const int nodes = argc > 4 ? atoi(argv[4]) : 16;
    const int iters = argc > 5 ? atoi(argv[5]) : 20;
    const int chain = argc > 6 && strcmp(argv[6], "chain") == 0;
    if (chain && rows != cols) { fprintf(stderr, "chain needs rows == cols\n"); return 2; }
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
    if (!be) { fprintf(stderr, "no GPU backend\n"); return 1; }

    struct ggml_init_params ip = { ggml_tensor_overhead() * (nodes + 8) + ggml_graph_overhead(), NULL, true };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, cols, rows);
    struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, toks);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    struct ggml_tensor *in = x;
    for (int i = 0; i < nodes; ++i) {
        struct ggml_tensor *out = ggml_mul_mat(ctx, w, in);
        ggml_build_forward_expand(gf, out);
        if (chain) in = out;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

    size_t wbytes = ggml_nbytes(w);
    unsigned char *wb = malloc(wbytes);
    srand(7);
    for (size_t b = 0; b < wbytes / 34; ++b) {
        ggml_fp16_t d = ggml_fp32_to_fp16(0.001f + (rand() % 100) * 1e-5f);
        memcpy(wb + b * 34, &d, 2);
        for (int k = 0; k < 32; ++k) wb[b * 34 + 2 + k] = (unsigned char)(rand() % 255 - 127);
    }
    ggml_backend_tensor_set(w, wb, 0, wbytes);
    float *xb = malloc(ggml_nbytes(x));
    for (int i = 0; i < cols * toks; ++i) xb[i] = (rand() % 2000 - 1000) * 1e-3f;
    ggml_backend_tensor_set(x, xb, 0, ggml_nbytes(x));

    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(be, gf);
    double t0 = now_s();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, gf);
    double per = (now_s() - t0) / ((double)iters * nodes);
    double gflops = 2.0 * rows * cols * (double)toks / per / 1e9;
    printf("ggml mul_mat %5dx%-5d tokens %3d%s: %8.1f us  %6.0f GFLOP/s\n", rows, cols, toks,
           chain ? " chain" : "", per * 1e6, gflops);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
