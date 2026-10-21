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
#define GMQ_PQ_DEFAULT_Q (1048576LL) /* 2^20: "Counting Divided" dyadic denominator */

/* Pack trits into a caller-owned, zero-initialized byte buffer. */
void gmq_pack_ternary(const int8_t *trits, size_t n, uint8_t *out);

/* Unpack bytes back into trits (out must hold `count` trits). */
void gmq_unpack_ternary(const uint8_t *bytes, size_t count, int8_t *out);

/* sum_j (pos[j] - neg[j]) * x[j] — the primitive the kernels are built on. */
double gmq_trit_dot(const int8_t *pos, const int8_t *neg, const double *x, size_t n);

/* Project double values onto int32 numerators p_out[k] = round(x[k] * q) in O(n). */
void gmq_quantize_pq(const double *x, size_t n, int64_t q, int32_t *p_out);

/* Pure integer-numerator dot product sum_j trits[j] * p_x[j] in Z (SSE4.1 pmulld / NEON smull). */
int64_t gmq_trit_dot_pq(const int8_t *trits, const int32_t *p_x, size_t n);

/* The masked kernel, j-major: out[i] = gamma * sum_j trit[j][i] * scale[j] * x[j]. */
void gmq_ternary_matmul(const uint8_t *packed, size_t rows, size_t cols,
                        const double *x, const double *scales, double gamma,
                        double *out);

/* Pure p/q ("Counting Divided") integer-numerator masked kernel: accumulates in Z (acc_scratch)
 * with zero inner-loop int->double conversions, dividing by q_x only once per output column. */
void gmq_ternary_matmul_pq(const uint8_t *packed, size_t rows, size_t cols,
                           const int32_t *p_x, int64_t q_x,
                           const double *scales, double gamma,
                           int64_t *acc_scratch, double *out);

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

/* 32-bit FNV-1a over prompt bytes. */
uint32_t gmq_seed_from(const char *prompt, size_t len);

/* Fold a prompt together with passive RF Shannon entropy (in bits). */
gmq_fold_t gmq_fold_spectrum(const char *prompt, size_t len, double rf_entropy_bits);

#ifdef __cplusplus
}
#endif

#endif /* GMQ_KERNEL_H */
