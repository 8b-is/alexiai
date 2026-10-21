/*
 * kernel.h — the C99 lane of the GAIA-MLX-QUANT substrate.
 *
 * Ternary b1.58 packing (5 trits/byte), the masked kernel, and a trit-dot
 * primitive that compiles to inline NEON (aarch64) or SSE4.1 (x86_64)
 * assembly and falls back to portable C99 everywhere else. No dependencies,
 * no allocation in the hot paths, OS-agnostic by construction — the same
 * math as the Rust core and the Go lane.
 */
#ifndef GMQ_KERNEL_H
#define GMQ_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GMQ_TRITS_PER_BYTE 5
#define GMQ_PQ_DEFAULT_Q 1048576

/* Pack trits into a caller-owned, zero-initialized byte buffer. */
void gmq_pack_ternary(const int8_t *trits, size_t n, uint8_t *out);

/* Unpack bytes back into trits (out must hold `count` trits). */
void gmq_unpack_ternary(const uint8_t *bytes, size_t count, int8_t *out);

/* sum_j (pos[j] - neg[j]) * x[j] — the primitive the kernels are built on. */
double gmq_trit_dot(const int8_t *pos, const int8_t *neg, const double *x, size_t n);

/* Pure-integer p/q trit dot product: sum_j trits[j] * p_x[j] with NEON/SSE4.1 integer SIMD (0 floats). */
int64_t gmq_trit_dot_pq(const int8_t *trits, const int32_t *p_x, size_t n);

/* Pure-integer per-row absmean ternary quantization in p/q (0 floats, 0 per-weight divisions). */
void gmq_quantize_per_row_pq(const int32_t *w_p, size_t rows, size_t cols,
                             int32_t q, int8_t *trits_out, int32_t *scales_p_out);

/* Pure-integer j-major 5-trits/byte matmul in p/q (0 floats). */
void gmq_ternary_matmul_pq(const uint8_t *packed, size_t rows, size_t cols,
                           const int32_t *p_x, const int32_t *p_scales,
                           int32_t q_scale, int64_t *p_out);

/* Pure-integer i-major NEON/SSE4.1 SIMD matmul in p/q (0 floats). */
void gmq_simd_matmul_pq(const int8_t *trits_i_major, size_t rows, size_t cols,
                        const int32_t *p_x, int64_t *p_out);

/* The masked kernel, j-major: out[i] = gamma * sum_j trit[j][i] * scale[j] * x[j]. */
void gmq_ternary_matmul(const uint8_t *packed, size_t rows, size_t cols,
                        const double *x, const double *scales, double gamma,
                        double *out);

/* Dense float64 reference. */
void gmq_dense_matmul(const double *w, size_t rows, size_t cols,
                      const double *x, double *out);

/* Relative L2 error of `a` against `b`. */
double gmq_relative_error(const double *a, const double *b, size_t n);

/* 1 when the platform assembly path is compiled in, 0 otherwise. */
int gmq_uses_asm(void);

/* 4-layer GAIA fold result for C99 cross-lane parity. */
typedef struct {
    uint32_t seed;
    double gravity;
    double entropy;
    double weather;
    double resonance;
    double coherence;
} gmq_fold_t;

/* Pure-integer p/q (q = 2^20) 4-layer GAIA fold result (0 floats). */
typedef struct {
    uint32_t seed;
    int64_t q;
    int64_t gravity_p;
    int64_t entropy_p;
    int64_t weather_p;
    int64_t resonance_p;
    int64_t coherence_p;
} gmq_fold_pq_t;

/* 32-bit FNV-1a over prompt bytes. */
uint32_t gmq_seed_from(const char *prompt, size_t len);

/* Fold a prompt together with passive RF Shannon entropy (in bits). */
gmq_fold_t gmq_fold_spectrum(const char *prompt, size_t len, double rf_entropy_bits);

/* Fold a prompt together with passive RF Shannon entropy numerator (in q = 2^20, 0 floats). */
gmq_fold_pq_t gmq_fold_spectrum_pq(const char *prompt, size_t len, int64_t rf_entropy_p);

#ifdef __cplusplus
}
#endif

#endif /* GMQ_KERNEL_H */
