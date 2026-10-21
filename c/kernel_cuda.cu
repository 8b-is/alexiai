/*
 * kernel_cuda.cu — CUDA GPU lane for the GAIA-MLX-QUANT substrate.
 *
 * Implements the 5-trits-per-byte (1.585 bits/weight) masked accumulation
 * kernel and dense float64 reference on NVIDIA GPUs. Accumulates in float64
 * in exact j-order so results match the C99, Rust, Go, and Swift lanes.
 */

#include "kernel.h"
#include "kernel_cuda.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

__constant__ int8_t d_lut[5][256];
static int host_lut_uploaded = 0;

static int ensure_cuda_luts(void) {
    if (host_lut_uploaded) return 0;
    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    int8_t h_lut[5][256];
    for (int slot = 0; slot < 5; slot++) {
        for (int b = 0; b < 256; b++) {
            uint8_t digit = (uint8_t)((b / pow3[slot]) % 3);
            h_lut[slot][b] = (digit == 2) ? -1 : (int8_t)digit;
        }
    }
    cudaError_t err = cudaMemcpyToSymbol(d_lut, h_lut, sizeof(h_lut));
    if (err != cudaSuccess) return (int)err;
    host_lut_uploaded = 1;
    return 0;
}

__global__ void gmq_ternary_matmul_kernel(const uint8_t *__restrict__ packed,
                                          size_t rows, size_t cols,
                                          const double *__restrict__ x,
                                          const double *__restrict__ scales,
                                          double gamma,
                                          double *__restrict__ out) {
    __shared__ int8_t s_lut[5][256];
    for (unsigned int idx = threadIdx.x; idx < 5 * 256; idx += blockDim.x) {
        s_lut[idx >> 8][idx & 255u] = d_lut[idx >> 8][idx & 255u];
    }
    __syncthreads();

    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;

    uint32_t u_cols = (uint32_t)cols;
    uint32_t u_i = (uint32_t)i;
    double acc = 0.0;
    for (uint32_t j = 0; j < (uint32_t)rows; j++) {
        double xv = x[j];
        if (xv == 0.0) continue;
        double xj = xv * gamma;
        if (scales != NULL) xj *= scales[j];
        uint32_t flat = j * u_cols + u_i;
        uint32_t byte_idx = flat / 5u;
        uint32_t slot = flat - byte_idx * 5u;
        uint8_t byte = __ldg(&packed[byte_idx]);
        if (byte == 0) continue;
        int8_t trit = s_lut[slot][byte];
        acc += (double)trit * xj;
    }
    out[i] = acc;
}

__global__ void gmq_dense_matmul_kernel(const double *__restrict__ w,
                                        size_t rows, size_t cols,
                                        const double *__restrict__ x,
                                        double *__restrict__ out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;

    double acc = 0.0;
    for (size_t j = 0; j < rows; j++) {
        double xv = x[j];
        if (xv == 0.0) continue;
        acc += w[j * cols + i] * xv;
    }
    out[i] = acc;
}

extern "C" int gmq_cuda_device_info(char *name_out, size_t name_len, int *sm_major, int *sm_minor) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return 0;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return 0;
    }
    if (name_out && name_len > 0) {
        snprintf(name_out, name_len, "%s", prop.name);
    }
    if (sm_major) *sm_major = prop.major;
    if (sm_minor) *sm_minor = prop.minor;
    return 1;
}

extern "C" int gmq_cuda_ternary_matmul(const uint8_t *packed, size_t rows, size_t cols,
                                       const double *x, const double *scales, double gamma,
                                       double *out) {
    int rc = ensure_cuda_luts();
    if (rc != 0) return rc;

    size_t packed_bytes = (rows * cols + GMQ_TRITS_PER_BYTE - 1) / GMQ_TRITS_PER_BYTE;
    uint8_t *d_packed = NULL;
    double *d_x = NULL, *d_scales = NULL, *d_out = NULL;
    cudaError_t err;

    if ((err = cudaMalloc((void **)&d_packed, packed_bytes)) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc((void **)&d_x, rows * sizeof(double))) != cudaSuccess) goto cleanup;
    if (scales != NULL) {
        if ((err = cudaMalloc((void **)&d_scales, rows * sizeof(double))) != cudaSuccess) goto cleanup;
    }
    if ((err = cudaMalloc((void **)&d_out, cols * sizeof(double))) != cudaSuccess) goto cleanup;

    if ((err = cudaMemcpy(d_packed, packed, packed_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(d_x, x, rows * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if (scales != NULL) {
        if ((err = cudaMemcpy(d_scales, scales, rows * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    }

    {
        unsigned int threads = 256;
        unsigned int blocks = (unsigned int)((cols + threads - 1) / threads);
        gmq_ternary_matmul_kernel<<<blocks, threads>>>(d_packed, rows, cols, d_x, d_scales, gamma, d_out);
        if ((err = cudaGetLastError()) != cudaSuccess) goto cleanup;
        if ((err = cudaMemcpy(out, d_out, cols * sizeof(double), cudaMemcpyDeviceToHost)) != cudaSuccess) goto cleanup;
    }

cleanup:
    if (d_packed) cudaFree(d_packed);
    if (d_x) cudaFree(d_x);
    if (d_scales) cudaFree(d_scales);
    if (d_out) cudaFree(d_out);
    return (int)err;
}

extern "C" int gmq_cuda_dense_matmul(const double *w, size_t rows, size_t cols,
                                     const double *x, double *out) {
    double *d_w = NULL, *d_x = NULL, *d_out = NULL;
    cudaError_t err;

    if ((err = cudaMalloc((void **)&d_w, rows * cols * sizeof(double))) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc((void **)&d_x, rows * sizeof(double))) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc((void **)&d_out, cols * sizeof(double))) != cudaSuccess) goto cleanup;

    if ((err = cudaMemcpy(d_w, w, rows * cols * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(d_x, x, rows * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;

    {
        unsigned int threads = 256;
        unsigned int blocks = (unsigned int)((cols + threads - 1) / threads);
        gmq_dense_matmul_kernel<<<blocks, threads>>>(d_w, rows, cols, d_x, d_out);
        if ((err = cudaGetLastError()) != cudaSuccess) goto cleanup;
        if ((err = cudaMemcpy(out, d_out, cols * sizeof(double), cudaMemcpyDeviceToHost)) != cudaSuccess) goto cleanup;
    }

cleanup:
    if (d_w) cudaFree(d_w);
    if (d_x) cudaFree(d_x);
    if (d_out) cudaFree(d_out);
    return (int)err;
}

extern "C" int gmq_cuda_bench_resident(const uint8_t *packed, const double *w,
                                       size_t rows, size_t cols,
                                       const double *x, const double *scales, double gamma,
                                       int iters, double *out,
                                       double *ternary_kernel_ns, double *dense_kernel_ns) {
    int rc = ensure_cuda_luts();
    if (rc != 0) return rc;

    size_t packed_bytes = (rows * cols + GMQ_TRITS_PER_BYTE - 1) / GMQ_TRITS_PER_BYTE;
    uint8_t *d_packed = NULL;
    double *d_w = NULL, *d_x = NULL, *d_scales = NULL, *d_out = NULL;
    cudaEvent_t ev_start = NULL, ev_stop = NULL;
    cudaError_t err;

    if ((err = cudaMalloc((void **)&d_packed, packed_bytes)) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc((void **)&d_w, rows * cols * sizeof(double))) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc((void **)&d_x, rows * sizeof(double))) != cudaSuccess) goto cleanup;
    if (scales != NULL) {
        if ((err = cudaMalloc((void **)&d_scales, rows * sizeof(double))) != cudaSuccess) goto cleanup;
    }
    if ((err = cudaMalloc((void **)&d_out, cols * sizeof(double))) != cudaSuccess) goto cleanup;

    if ((err = cudaMemcpy(d_packed, packed, packed_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(d_w, w, rows * cols * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(d_x, x, rows * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if (scales != NULL) {
        if ((err = cudaMemcpy(d_scales, scales, rows * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    }

    if ((err = cudaEventCreate(&ev_start)) != cudaSuccess) goto cleanup;
    if ((err = cudaEventCreate(&ev_stop)) != cudaSuccess) goto cleanup;

    {
        unsigned int threads = 256;
        unsigned int blocks = (unsigned int)((cols + threads - 1) / threads);
        float ms_ternary = 0.0f, ms_dense = 0.0f;

        /* Warm-up */
        gmq_ternary_matmul_kernel<<<blocks, threads>>>(d_packed, rows, cols, d_x, d_scales, gamma, d_out);
        gmq_dense_matmul_kernel<<<blocks, threads>>>(d_w, rows, cols, d_x, d_out);
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        for (int k = 0; k < iters; k++) {
            gmq_dense_matmul_kernel<<<blocks, threads>>>(d_w, rows, cols, d_x, d_out);
        }
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms_dense, ev_start, ev_stop);

        cudaEventRecord(ev_start);
        for (int k = 0; k < iters; k++) {
            gmq_ternary_matmul_kernel<<<blocks, threads>>>(d_packed, rows, cols, d_x, d_scales, gamma, d_out);
        }
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms_ternary, ev_start, ev_stop);

        if ((err = cudaMemcpy(out, d_out, cols * sizeof(double), cudaMemcpyDeviceToHost)) != cudaSuccess) goto cleanup;

        if (ternary_kernel_ns) *ternary_kernel_ns = ((double)ms_ternary * 1e6) / (double)iters;
        if (dense_kernel_ns) *dense_kernel_ns = ((double)ms_dense * 1e6) / (double)iters;
    }

cleanup:
    if (ev_start) cudaEventDestroy(ev_start);
    if (ev_stop) cudaEventDestroy(ev_stop);
    if (d_packed) cudaFree(d_packed);
    if (d_w) cudaFree(d_w);
    if (d_x) cudaFree(d_x);
    if (d_scales) cudaFree(d_scales);
    if (d_out) cudaFree(d_out);
    return (int)err;
}
