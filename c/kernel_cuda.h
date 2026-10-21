/*
 * kernel_cuda.h — CUDA GPU lane for the GAIA-MLX-QUANT substrate.
 *
 * Same 5-trits-per-byte base-3 packing and float64 masked accumulation as
 * the C99, Rust, Go, and Swift lanes, executed on local NVIDIA GPUs.
 */
#ifndef GMQ_KERNEL_CUDA_H
#define GMQ_KERNEL_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if a CUDA-capable GPU is present and initialized, 0 otherwise.
 * Writes a human-readable device description into `name_out` (up to `name_len`). */
int gmq_cuda_device_info(char *name_out, size_t name_len, int *sm_major, int *sm_minor);

/* GPU masked ternary matmul (j-major packed buffer, 5 trits/byte).
 * Returns 0 on success, non-zero CUDA error code on failure. */
int gmq_cuda_ternary_matmul(const uint8_t *packed, size_t rows, size_t cols,
                            const double *x, const double *scales, double gamma,
                            double *out);

/* GPU dense float64 reference matmul.
 * Returns 0 on success, non-zero CUDA error code on failure. */
int gmq_cuda_dense_matmul(const double *w, size_t rows, size_t cols,
                          const double *x, double *out);

/* GPU benchmark helper: runs `iters` launches on resident device buffers and
 * writes average kernel-only nanoseconds into `ternary_kernel_ns` and
 * `dense_kernel_ns`, plus copies the GPU ternary output into `out` for honest
 * error verification. */
int gmq_cuda_bench_resident(const uint8_t *packed, const double *w,
                            size_t rows, size_t cols,
                            const double *x, const double *scales, double gamma,
                            int iters, double *out,
                            double *ternary_kernel_ns, double *dense_kernel_ns);

#ifdef __cplusplus
}
#endif

#endif /* GMQ_KERNEL_CUDA_H */
