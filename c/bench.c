/*
 * bench.c — honest C99 (+ optional CUDA GPU) benchmark for GAIA-MLX-QUANT.
 *
 * Uses the exact same deterministic synthetic weights and per-row absmean
 * quantization as the Rust bench (`rust/gaia-mlx-quant/examples/bench.rs`),
 * reporting measured timing and relative L2 reconstruction error.
 */

#define _POSIX_C_SOURCE 199309L
#include "kernel.h"
#ifdef GMQ_CUDA
#include "kernel_cuda.h"
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void fill_weights(double *out, size_t n, uint32_t seed) {
    uint32_t s = seed;
    size_t i;
    for (i = 0; i < n; i++) {
        uint32_t t;
        double u;
        s += 0x6d2b79f5u;
        t = s;
        t = (t ^ (t >> 15)) * (t | 1u);
        t ^= t + ((t ^ (t >> 7)) * (t | 61u));
        u = (double)((t ^ (t >> 14)) >> 8) / 16777216.0;
        out[i] = u + u - 1.0;
    }
}

static void quantize_per_row(const double *w, size_t rows, size_t cols,
                             int8_t *trits, double *scales) {
    size_t r, c;
    for (r = 0; r < rows; r++) {
        const double *row = &w[r * cols];
        double sum = 0.0, scale;
        for (c = 0; c < cols; c++) sum += fabs(row[c]);
        scale = (sum == 0.0) ? 1.0 : (sum / (double)cols);
        scales[r] = scale;
        for (c = 0; c < cols; c++) {
            double q = floor((row[c] / scale) + 0.5);
            trits[r * cols + c] = (q > 0.0) ? 1 : (q < 0.0) ? -1 : 0;
        }
    }
}

int main(int argc, char **argv) {
    size_t dim = (argc > 1 && atoi(argv[1]) > 0) ? (size_t)atoi(argv[1]) : 256;
    size_t rows = dim, cols = dim;
    size_t packed_len = (rows * cols + GMQ_TRITS_PER_BYTE - 1) / GMQ_TRITS_PER_BYTE;

    size_t words_per_col = (rows + 63) / 64;
    double *w = (double *)calloc(rows * cols, sizeof(double));
    double *x = (double *)calloc(rows, sizeof(double));
    int8_t *trits = (int8_t *)calloc(rows * cols, sizeof(int8_t));
    int8_t *trits_i = (int8_t *)calloc(rows * cols, sizeof(int8_t));
    int8_t *m_x_bfp = (int8_t *)calloc(rows, sizeof(int8_t));
    uint64_t *w_pos_u64 = (uint64_t *)calloc(cols * words_per_col, sizeof(uint64_t));
    uint64_t *w_neg_u64 = (uint64_t *)calloc(cols * words_per_col, sizeof(uint64_t));
    uint64_t *x_pos_u64 = (uint64_t *)calloc(words_per_col, sizeof(uint64_t));
    uint64_t *x_neg_u64 = (uint64_t *)calloc(words_per_col, sizeof(uint64_t));
    double *scales = (double *)calloc(rows, sizeof(double));
    int32_t *p_x_scaled = (int32_t *)calloc(rows, sizeof(int32_t));
    uint8_t *packed = (uint8_t *)calloc(packed_len, sizeof(uint8_t));
    double *dense_out = (double *)calloc(cols, sizeof(double));
    double *ternary_out = (double *)calloc(cols, sizeof(double));
    int64_t *pq_packed_out = (int64_t *)calloc(cols, sizeof(int64_t));
    int64_t *pq_simd_out = (int64_t *)calloc(cols, sizeof(int64_t));
    int64_t *pq_bfp_out = (int64_t *)calloc(cols, sizeof(int64_t));
    int64_t *pq_quantum_out = (int64_t *)calloc(cols, sizeof(int64_t));

    int iters = 500;
    int k;
    size_t r, c;
    int32_t scale_qx;
    uint64_t t0, dense_ns, ternary_ns, pq_packed_ns, pq_simd_ns, pq_bfp_ns, pq_quantum_ns;

    fill_weights(w, rows * cols, 0x5eedu);
    fill_weights(x, rows, 0xc0ffeeu);
    quantize_per_row(w, rows, cols, trits, scales);
    gmq_pack_ternary(trits, rows * cols, packed);

    for (r = 0; r < rows; r++) {
        double v = x[r] * scales[r] * (double)GMQ_PQ_DEFAULT_Q;
        p_x_scaled[r] = (int32_t)(v >= 0.0 ? v + 0.5 : v - 0.5);
        m_x_bfp[r] = (int8_t)(x[r] >= 0.0 ? (x[r] * 127.0 + 0.5) : (x[r] * 127.0 - 0.5));
        for (c = 0; c < cols; c++) {
            trits_i[c * rows + r] = trits[r * cols + c];
        }
    }
    gmq_pack_quantum_bitplanes_i_major(trits, rows, cols, w_pos_u64, w_neg_u64);
    scale_qx = gmq_quantize_quantum_act_pq(p_x_scaled, rows, x_pos_u64, x_neg_u64);

    /* Warm-up */
    gmq_dense_matmul(w, rows, cols, x, dense_out);
    gmq_ternary_matmul(packed, rows, cols, x, scales, 1.0, ternary_out);
    gmq_ternary_matmul_pq(packed, rows, cols, p_x_scaled, NULL, GMQ_PQ_DEFAULT_Q, pq_packed_out);
    gmq_simd_matmul_pq(trits_i, rows, cols, p_x_scaled, pq_simd_out);
    gmq_omni_bfp_i8_matmul_pq(trits_i, rows, cols, m_x_bfp, pq_bfp_out);
    gmq_quantum_bitplane_matmul_pq(w_pos_u64, w_neg_u64, rows, cols, x_pos_u64, x_neg_u64, scale_qx, pq_quantum_out);

    t0 = now_ns();
    for (k = 0; k < iters; k++) {
        gmq_dense_matmul(w, rows, cols, x, dense_out);
    }
    dense_ns = (now_ns() - t0) / (uint64_t)iters;

    t0 = now_ns();
    for (k = 0; k < iters; k++) {
        gmq_quantum_bitplane_matmul_pq(w_pos_u64, w_neg_u64, rows, cols, x_pos_u64, x_neg_u64, scale_qx, pq_quantum_out);
    }
    pq_quantum_ns = (now_ns() - t0) / (uint64_t)iters;

    t0 = now_ns();
    for (k = 0; k < iters; k++) {
        gmq_omni_bfp_i8_matmul_pq(trits_i, rows, cols, m_x_bfp, pq_bfp_out);
    }
    pq_bfp_ns = (now_ns() - t0) / (uint64_t)iters;

    t0 = now_ns();
    for (k = 0; k < iters; k++) {
        gmq_simd_matmul_pq(trits_i, rows, cols, p_x_scaled, pq_simd_out);
    }
    pq_simd_ns = (now_ns() - t0) / (uint64_t)iters;

    t0 = now_ns();
    for (k = 0; k < iters; k++) {
        gmq_ternary_matmul_pq(packed, rows, cols, p_x_scaled, NULL, GMQ_PQ_DEFAULT_Q, pq_packed_out);
    }
    pq_packed_ns = (now_ns() - t0) / (uint64_t)iters;

    t0 = now_ns();
    for (k = 0; k < iters; k++) {
        gmq_ternary_matmul(packed, rows, cols, x, scales, 1.0, ternary_out);
    }
    ternary_ns = (now_ns() - t0) / (uint64_t)iters;

    printf("GAIA-MLX-QUANT C99 v0.3.0 bench — ULTRA-SUPER-OMNI-Quantum Ternary & p/q numbers\n");
    printf("  asm             %s\n", gmq_uses_asm() ? "SSE4.1 / NEON compiled in" : "portable C99");
    printf("  shape           %zux%zu\n", rows, cols);
    printf("  packing         1.5850 bits/weight · %zu bytes vs %zu f64 (%.2fx)\n",
           packed_len, rows * cols * sizeof(double),
           (double)(rows * cols * sizeof(double)) / (double)packed_len);
    printf("  cpu dense       %llu ns/iter  (f64 reference)\n", (unsigned long long)dense_ns);
    printf("  cpu quantum-u64 %llu ns/iter  (64-trit POPCNT 8-col 512 MACs/step, 0 floats, %.2fx vs dense)\n",
           (unsigned long long)pq_quantum_ns,
           pq_quantum_ns > 0 ? (double)dense_ns / (double)pq_quantum_ns : 0.0);
    printf("  cpu omni-bfp-i8 %llu ns/iter  (NEON/SSE4.1 16-lane smlal i8×i8->i16, 0 floats, %.2fx vs dense)\n",
           (unsigned long long)pq_bfp_ns,
           pq_bfp_ns > 0 ? (double)dense_ns / (double)pq_bfp_ns : 0.0);
    printf("  cpu pq-simd     %llu ns/iter  (NEON/SSE4.1 8-lane i8×i32->i32, 0 floats, %.2fx vs dense)\n",
           (unsigned long long)pq_simd_ns,
           pq_simd_ns > 0 ? (double)dense_ns / (double)pq_simd_ns : 0.0);
    printf("  cpu pq-b158     %llu ns/iter  (5-trits/byte p/q integer, 0 floats)\n",
           (unsigned long long)pq_packed_ns);
    printf("  cpu packed      %llu ns/iter  (f64 boundary, unguarded interior bytes)\n",
           (unsigned long long)ternary_ns);
    printf("  error           %.4f relative L2 vs dense (random weights — the worst case)\n",
           gmq_relative_error(ternary_out, dense_out, cols));

#ifdef GMQ_CUDA
    {
        char gpu_name[128] = {0};
        int sm_maj = 0, sm_min = 0;
        if (gmq_cuda_device_info(gpu_name, sizeof(gpu_name), &sm_maj, &sm_min)) {
            double *gpu_out = (double *)calloc(cols, sizeof(double));
            double gpu_ternary_ns = 0.0, gpu_dense_ns = 0.0;
            int rc = gmq_cuda_bench_resident(
                packed, w, rows, cols, x, scales, 1.0, iters,
                gpu_out, &gpu_ternary_ns, &gpu_dense_ns);
            if (rc == 0) {
                double parity_err = gmq_relative_error(gpu_out, ternary_out, cols);
                printf("  gpu device %s (sm_%d%d)\n", gpu_name, sm_maj, sm_min);
                printf("  gpu dense  %.0f ns/iter (resident kernel)\n", gpu_dense_ns);
                printf("  gpu packed %.0f ns/iter (resident kernel)\n", gpu_ternary_ns);
                printf("  gpu parity %.2e relative diff vs CPU ternary\n", parity_err);
            }
            free(gpu_out);
        }
    }
#endif

    free(w);
    free(x);
    free(trits);
    free(trits_i);
    free(scales);
    free(p_x_scaled);
    free(packed);
    free(dense_out);
    free(ternary_out);
    free(pq_packed_out);
    free(pq_simd_out);
    free(pq_bfp_out);
    free(pq_quantum_out);
    free(m_x_bfp);
    free(w_pos_u64);
    free(w_neg_u64);
    free(x_pos_u64);
    free(x_neg_u64);
    return 0;
}
