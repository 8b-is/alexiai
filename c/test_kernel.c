/*
 * test_kernel.c — the C99 lane's test harness. No framework, no dependencies:
 * a handful of assertions and a non-zero exit on the first failure. The trit
 * dot test cross-checks the assembly path against the portable C reference,
 * so the platform asm is proven correct on whatever machine this runs.
 */

#include "kernel.h"
#ifdef GMQ_CUDA
#include "kernel_cuda.h"
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static double c_trit_dot(const int8_t *pos, const int8_t *neg, const double *x, size_t n) {
    double acc = 0.0;
    size_t i;
    for (i = 0; i < n; i++) acc += (double)(pos[i] - neg[i]) * x[i];
    return acc;
}

static void test_round_trip(void) {
    size_t n;
    for (n = 0; n <= 32; n++) {
        int8_t trits[33];
        uint8_t packed[(33 + 4) / 5];
        int8_t out[33];
        size_t i;
        for (i = 0; i < n; i++) trits[i] = (i % 2 == 0) ? 1 : -1;
        memset(packed, 0, sizeof(packed));
        gmq_pack_ternary(trits, n, packed);
        gmq_unpack_ternary(packed, n, out);
        for (i = 0; i < n; i++) {
            if (out[i] != trits[i]) {
                CHECK(0, "round trip mismatch");
                return;
            }
        }
    }
    CHECK(1, "round trips hold");
}

static void test_byte_242(void) {
    uint8_t packed = 242;
    int8_t out[5];
    gmq_unpack_ternary(&packed, 5, out);
    CHECK(out[0] == -1 && out[1] == -1 && out[2] == -1 && out[3] == -1 && out[4] == -1,
          "byte 242 must unpack to five -1 trits");
}

static void test_asm_dot_matches_c(void) {
    size_t n;
    for (n = 1; n <= 40; n++) {
        int8_t pos[40], neg[40];
        double x[40];
        size_t i;
        for (i = 0; i < n; i++) {
            pos[i] = (int8_t)(i % 3 == 0 ? 1 : 0);
            neg[i] = (int8_t)(i % 5 == 0 ? 1 : 0);
            x[i] = (double)(i % 7) * 0.25;
        }
        {
            double a = gmq_trit_dot(pos, neg, x, n);
            double b = c_trit_dot(pos, neg, x, n);
            if (fabs(a - b) > 1e-12) {
                CHECK(0, "asm dot disagrees with C reference");
                return;
            }
        }
    }
    CHECK(1, "asm dot matches C reference");
}

static void test_kernel_matches_dense(void) {
    enum { ROWS = 3, COLS = 5 };
    int8_t trits[ROWS * COLS] = {
        1, -1, 0, 1, -1,
        -1, 1, 1, 0, 0,
        0, 0, -1, 1, 1,
    };
    double x[ROWS] = {2.0, -1.0, 3.0};
    uint8_t packed[(ROWS * COLS + 4) / 5];
    double ternary[COLS], dense[COLS];
    double w[ROWS * COLS];
    int i;
    memset(packed, 0, sizeof(packed));
    gmq_pack_ternary(trits, ROWS * COLS, packed);
    gmq_ternary_matmul(packed, ROWS, COLS, x, NULL, 1.0, ternary);
    for (i = 0; i < ROWS * COLS; i++) w[i] = (double)trits[i];
    gmq_dense_matmul(w, ROWS, COLS, x, dense);
    for (i = 0; i < COLS; i++) {
        if (ternary[i] != dense[i]) {
            CHECK(0, "kernel disagrees with dense reference");
            return;
        }
    }
    CHECK(1, "kernel matches dense");
}

static void test_non_aligned_rows(void) {
    enum { ROWS = 4, COLS = 3 };
    int8_t trits[ROWS * COLS];
    double x[ROWS] = {1.0, 2.0, 3.0, 4.0};
    uint8_t packed[(ROWS * COLS + 4) / 5];
    double out[COLS], ref[COLS];
    double w[ROWS * COLS];
    int i;
    for (i = 0; i < ROWS * COLS; i++) trits[i] = (int8_t)(-1 + (i % 3));
    memset(packed, 0, sizeof(packed));
    gmq_pack_ternary(trits, ROWS * COLS, packed);
    gmq_ternary_matmul(packed, ROWS, COLS, x, NULL, 1.0, out);
    for (i = 0; i < ROWS * COLS; i++) w[i] = (double)trits[i];
    gmq_dense_matmul(w, ROWS, COLS, x, ref);
    for (i = 0; i < COLS; i++) {
        if (out[i] != ref[i]) {
            CHECK(0, "non-aligned rows disagree with dense");
            return;
        }
    }
    CHECK(1, "non-aligned rows hold");
}

static void test_relative_error(void) {
    double a[3] = {1, 2, 3};
    CHECK(gmq_relative_error(a, a, 3) == 0.0, "identical vectors must have zero error");
}

static void test_fold_spectrum(void) {
    const char *prompt = "Analyze Shinjuku 2.4GHz Ch 6 saturation, BLE proximity & AirPlay RAOP surface";
    size_t len = strlen(prompt);
    gmq_fold_t low = gmq_fold_spectrum(prompt, len, 0.5);
    gmq_fold_t high = gmq_fold_spectrum(prompt, len, 4.8);
    CHECK(low.seed == high.seed, "fold_spectrum seed must be deterministic");
    CHECK(low.weather == high.weather, "fold_spectrum weather must be deterministic");
    CHECK(high.entropy > low.entropy, "fold_spectrum high RF entropy must lift Chaos entropy");
    CHECK(high.coherence >= 0.0 && high.coherence <= 1.0, "fold_spectrum coherence in [0, 1]");
}

static void test_pq_kernels(void) {
    enum { N = 135, ROWS = 16, COLS = 25 };
    int8_t trits_dot[N];
    double x_dot[N];
    int32_t px_dot[N];
    int64_t ref_dot = 0, got_dot;
    int8_t trits_m[ROWS * COLS];
    double x_m[ROWS], scales_m[ROWS];
    int32_t px_m[ROWS];
    uint8_t packed_m[(ROWS * COLS + 4) / 5];
    int64_t acc_m[COLS];
    double out_float[COLS], out_pq[COLS];
    int i;

    for (i = 0; i < N; i++) {
        trits_dot[i] = (int8_t)((i % 3) - 1);
        x_dot[i] = ((double)((i * 11 + 3) % 19) / 9.0) - 1.0;
    }
    gmq_quantize_pq(x_dot, N, GMQ_PQ_DEFAULT_Q, px_dot);
    for (i = 0; i < N; i++) {
        ref_dot += (int64_t)trits_dot[i] * (int64_t)px_dot[i];
    }
    got_dot = gmq_trit_dot_pq(trits_dot, px_dot, N);
    CHECK(got_dot == ref_dot, "gmq_trit_dot_pq SSE4.1/C99 matches exact integer numerator");

    for (i = 0; i < ROWS * COLS; i++) trits_m[i] = (int8_t)(((i * 7 + 2) % 3) - 1);
    for (i = 0; i < ROWS; i++) {
        x_m[i] = ((double)((i * 13 + 5) % 17) / 8.0) - 1.0;
        scales_m[i] = 0.35 + (double)(i % 5) * 0.12;
    }
    memset(packed_m, 0, sizeof(packed_m));
    gmq_pack_ternary(trits_m, ROWS * COLS, packed_m);
    gmq_quantize_pq(x_m, ROWS, GMQ_PQ_DEFAULT_Q, px_m);

    gmq_ternary_matmul(packed_m, ROWS, COLS, x_m, scales_m, 1.0, out_float);
    gmq_ternary_matmul_pq(packed_m, ROWS, COLS, px_m, GMQ_PQ_DEFAULT_Q, scales_m, 1.0, acc_m, out_pq);
    CHECK(gmq_relative_error(out_pq, out_float, COLS) < 1e-5,
          "gmq_ternary_matmul_pq matches float kernel within 1e-5");
}

#ifdef GMQ_CUDA
static void test_cuda_matches_cpu(void) {
    char gpu_name[128] = {0};
    int sm_maj = 0, sm_min = 0;
    enum { ROWS = 17, COLS = 23 };
    int8_t trits[ROWS * COLS];
    double x[ROWS], scales[ROWS], w[ROWS * COLS];
    uint8_t packed[(ROWS * COLS + 4) / 5];
    double cpu_ternary[COLS], gpu_ternary[COLS];
    double cpu_dense[COLS], gpu_dense[COLS];
    int i, r, c;

    if (!gmq_cuda_device_info(gpu_name, sizeof(gpu_name), &sm_maj, &sm_min)) {
        printf("  (no CUDA device visible — skipping GPU parity check)\n");
        return;
    }
    printf("  cuda device: %s (sm_%d%d)\n", gpu_name, sm_maj, sm_min);

    for (r = 0; r < ROWS; r++) {
        x[r] = (double)(r - 8) * 0.375;
        scales[r] = 0.25 + (double)(r % 5) * 0.15;
        for (c = 0; c < COLS; c++) {
            int idx = r * COLS + c;
            trits[idx] = (int8_t)((idx % 3) - 1);
            w[idx] = (double)trits[idx] * scales[r];
        }
    }
    memset(packed, 0, sizeof(packed));
    gmq_pack_ternary(trits, ROWS * COLS, packed);

    gmq_ternary_matmul(packed, ROWS, COLS, x, scales, 1.618, cpu_ternary);
    CHECK(gmq_cuda_ternary_matmul(packed, ROWS, COLS, x, scales, 1.618, gpu_ternary) == 0,
          "gmq_cuda_ternary_matmul succeeded");

    gmq_dense_matmul(w, ROWS, COLS, x, cpu_dense);
    CHECK(gmq_cuda_dense_matmul(w, ROWS, COLS, x, gpu_dense) == 0,
          "gmq_cuda_dense_matmul succeeded");

    for (i = 0; i < COLS; i++) {
        if (fabs(cpu_ternary[i] - gpu_ternary[i]) > 1e-12) {
            CHECK(0, "CUDA ternary kernel disagrees with C99 reference");
            return;
        }
        if (fabs(cpu_dense[i] - gpu_dense[i]) > 1e-12) {
            CHECK(0, "CUDA dense kernel disagrees with C99 reference");
            return;
        }
    }
    CHECK(1, "CUDA GPU ternary & dense kernels match C99 reference");
}
#endif

int main(void) {
    printf("gmq C99 test harness — asm path: %s\n", gmq_uses_asm() ? "compiled in" : "not compiled");
    test_round_trip();
    test_byte_242();
    test_asm_dot_matches_c();
    test_kernel_matches_dense();
    test_non_aligned_rows();
    test_relative_error();
    test_fold_spectrum();
    test_pq_kernels();
#ifdef GMQ_CUDA
    test_cuda_matches_cpu();
#endif
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
