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

#define GMQ_TRITS_PER_BYTE 5

/* Pack trits into a caller-owned, zero-initialized byte buffer. */
void gmq_pack_ternary(const int8_t *trits, size_t n, uint8_t *out);

/* Unpack bytes back into trits (out must hold `count` trits). */
void gmq_unpack_ternary(const uint8_t *bytes, size_t count, int8_t *out);

/* sum_j (pos[j] - neg[j]) * x[j] — the primitive the kernels are built on. */
double gmq_trit_dot(const int8_t *pos, const int8_t *neg, const double *x, size_t n);

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

#endif /* GMQ_KERNEL_H */
