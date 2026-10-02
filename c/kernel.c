/*
 * kernel.c — the C99 lane of the GAIA-MLX-QUANT substrate.
 *
 * Ternary b1.58 packing (5 trits/byte), the masked kernel, and a trit-dot
 * primitive with inline NEON (aarch64) and SSE4.1 (x86_64) assembly paths,
 * falling back to portable C99 elsewhere. Zero dependencies, no allocation
 * in the hot paths — the same math as the Rust core, the Go lane, and the
 * Swift lane.
 */

#include "kernel.h"

#include <math.h>

/* ── the five slot LUTs: byte → trit, no division in the hot path ───────── */

static const uint8_t gmq_pow3[GMQ_TRITS_PER_BYTE] = {1, 3, 9, 27, 81};

static int8_t gmq_slot_trit(uint8_t byte, int slot) {
    uint8_t digit = (uint8_t)((byte / gmq_pow3[slot]) % 3);
    return digit == 2 ? -1 : (int8_t)digit;
}

/* Static const LUTs built at compile time via designated initializers are
 * not computable in pure C99, so build them once at init and keep them in
 * static storage. */
static int8_t gmq_t0[256];
static int8_t gmq_t1[256];
static int8_t gmq_t2[256];
static int8_t gmq_t3[256];
static int8_t gmq_t4[256];
static int gmq_luts_ready = 0;

static void gmq_build_luts(void) {
    int b;
    if (gmq_luts_ready) return;
    for (b = 0; b < 256; b++) {
        gmq_t0[b] = gmq_slot_trit((uint8_t)b, 0);
        gmq_t1[b] = gmq_slot_trit((uint8_t)b, 1);
        gmq_t2[b] = gmq_slot_trit((uint8_t)b, 2);
        gmq_t3[b] = gmq_slot_trit((uint8_t)b, 3);
        gmq_t4[b] = gmq_slot_trit((uint8_t)b, 4);
    }
    gmq_luts_ready = 1;
}

/* ── packing ─────────────────────────────────────────────────────────────── */

/* The branchless digit mapping: -1 → 2, 0 → 0, +1 → 1.
 * For t = -1, (uint8_t)t = 0xFF, and 0xFF + (0xFF >> 7) * 3 wraps to 2. */
static uint8_t gmq_digit_of(int8_t t) {
    return (uint8_t)t + (uint8_t)(((uint8_t)t >> 7) * 3u);
}

void gmq_pack_ternary(const int8_t *trits, size_t n, uint8_t *out) {
    size_t i;
    for (i = 0; i < n; i++) {
        out[i / GMQ_TRITS_PER_BYTE] +=
            (uint8_t)(gmq_pow3[i % GMQ_TRITS_PER_BYTE] * gmq_digit_of(trits[i]));
    }
}

void gmq_unpack_ternary(const uint8_t *bytes, size_t count, int8_t *out) {
    size_t i;
    gmq_build_luts();
    for (i = 0; i < count / GMQ_TRITS_PER_BYTE; i++) {
        uint8_t b = bytes[i];
        size_t base = i * GMQ_TRITS_PER_BYTE;
        out[base] = gmq_t0[b];
        out[base + 1] = gmq_t1[b];
        out[base + 2] = gmq_t2[b];
        out[base + 3] = gmq_t3[b];
        out[base + 4] = gmq_t4[b];
    }
    /* tail: the final partial byte */
    {
        size_t base = (count / GMQ_TRITS_PER_BYTE) * GMQ_TRITS_PER_BYTE;
        if (base < count) {
            uint8_t b = bytes[count / GMQ_TRITS_PER_BYTE];
            if (base + 0 < count) out[base] = gmq_t0[b];
            if (base + 1 < count) out[base + 1] = gmq_t1[b];
            if (base + 2 < count) out[base + 2] = gmq_t2[b];
            if (base + 3 < count) out[base + 3] = gmq_t3[b];
            if (base + 4 < count) out[base + 4] = gmq_t4[b];
        }
    }
}

/* ── the trit-dot primitive, with platform assembly paths ────────────────── */

#if defined(__aarch64__)
/* NEON: eight trits per iteration, four f64 accumulators. */
double gmq_trit_dot(const int8_t *pos, const int8_t *neg, const double *x, size_t n) {
    double acc = 0.0;
    /* Saved originals: the asm advances register copies and burns its own
     * counter down to zero; the tail walks the untouched base pointers. */
    const int8_t *pos0 = pos;
    const int8_t *neg0 = neg;
    const double *x0 = x;
    size_t tail_start = n & ~(size_t)7;
    size_t k = tail_start;
    __asm__ volatile(
        "movi v0.2d, #0\n\t"
        "movi v1.2d, #0\n\t"
        "movi v16.2d, #0\n\t"
        "movi v17.2d, #0\n\t"
        "1:\n\t"
        "cmp %x[k], #0\n\t"
        "b.eq 2f\n\t"
        "ld1 {v2.8b}, [%x[p]], #8\n\t"
        "ld1 {v3.8b}, [%x[m]], #8\n\t"
        "sub v4.8b, v2.8b, v3.8b\n\t"
        "sxtl v5.8h, v4.8b\n\t"
        "sxtl v6.4s, v5.4h\n\t"
        "sxtl2 v7.4s, v5.8h\n\t"
        "sxtl v8.2d, v6.2s\n\t"
        "sxtl2 v9.2d, v6.4s\n\t"
        "sxtl v10.2d, v7.2s\n\t"
        "sxtl2 v11.2d, v7.4s\n\t"
        "scvtf v8.2d, v8.2d\n\t"
        "scvtf v9.2d, v9.2d\n\t"
        "scvtf v10.2d, v10.2d\n\t"
        "scvtf v11.2d, v11.2d\n\t"
        "ld1 {v12.2d, v13.2d, v14.2d, v15.2d}, [%x[x]], #64\n\t"
        "fmla v0.2d, v8.2d, v12.2d\n\t"
        "fmla v1.2d, v9.2d, v13.2d\n\t"
        "fmla v16.2d, v10.2d, v14.2d\n\t"
        "fmla v17.2d, v11.2d, v15.2d\n\t"
        "sub %x[k], %x[k], #8\n\t"
        "b 1b\n\t"
        "2:\n\t"
        "fadd v0.2d, v0.2d, v1.2d\n\t"
        "fadd v16.2d, v16.2d, v17.2d\n\t"
        "fadd v0.2d, v0.2d, v16.2d\n\t"
        "faddp d0, v0.2d\n\t"
        "str d0, %[acc]\n\t"
        : [acc] "=m"(acc), [p] "+r"(pos), [m] "+r"(neg), [x] "+r"(x), [k] "+r"(k)
        :
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
          "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17");
    /* tail */
    {
        size_t i;
        for (i = tail_start; i < n; i++) acc += (double)(pos0[i] - neg0[i]) * x0[i];
    }
    return acc;
}
int gmq_uses_asm(void) { return 1; }

#elif defined(__x86_64__) && defined(__SSE4_1__)
/* SSE4.1: four trits per iteration, two f64 accumulators. */
double gmq_trit_dot(const int8_t *pos, const int8_t *neg, const double *x, size_t n) {
    double acc = 0.0;
    const int8_t *pos0 = pos;
    const int8_t *neg0 = neg;
    const double *x0 = x;
    size_t tail_start = n & ~(size_t)3;
    size_t k = tail_start;
    __asm__ volatile(
        "xorpd %%xmm0, %%xmm0\n\t"
        "xorpd %%xmm1, %%xmm1\n\t"
        "1:\n\t"
        "cmp %[k], 0\n\t"
        "je 2f\n\t"
        "movd (%[p]), %%xmm2\n\t"
        "movd (%[m]), %%xmm3\n\t"
        "pmovsxbd %%xmm2, %%xmm2\n\t"
        "pmovsxbd %%xmm3, %%xmm3\n\t"
        "psubd %%xmm3, %%xmm2\n\t"
        "cvtdq2pd %%xmm2, %%xmm4\n\t"
        "pshufd $0x0E, %%xmm2, %%xmm2\n\t"
        "cvtdq2pd %%xmm2, %%xmm5\n\t"
        "movupd (%[x]), %%xmm6\n\t"
        "movupd 16(%[x]), %%xmm7\n\t"
        "mulpd %%xmm6, %%xmm4\n\t"
        "mulpd %%xmm7, %%xmm5\n\t"
        "addpd %%xmm4, %%xmm0\n\t"
        "addpd %%xmm5, %%xmm1\n\t"
        "add $4, %[p]\n\t"
        "add $4, %[m]\n\t"
        "add $32, %[x]\n\t"
        "sub $4, %[k]\n\t"
        "jmp 1b\n\t"
        "2:\n\t"
        "addpd %%xmm1, %%xmm0\n\t"
        "haddpd %%xmm0, %%xmm0\n\t"
        "movsd %%xmm0, %[acc]\n\t"
        : [acc] "=m"(acc), [p] "+r"(pos), [m] "+r"(neg), [x] "+r"(x), [k] "+r"(k)
        :
        : "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7");
    /* tail */
    {
        size_t i;
        for (i = tail_start; i < n; i++) acc += (double)(pos0[i] - neg0[i]) * x0[i];
    }
    return acc;
}
int gmq_uses_asm(void) { return 1; }

#else
/* Portable C99. */
double gmq_trit_dot(const int8_t *pos, const int8_t *neg, const double *x, size_t n) {
    double acc = 0.0;
    size_t i;
    for (i = 0; i < n; i++) acc += (double)(pos[i] - neg[i]) * x[i];
    return acc;
}
int gmq_uses_asm(void) { return 0; }
#endif

/* ── the kernels ─────────────────────────────────────────────────────────── */

void gmq_ternary_matmul(const uint8_t *packed, size_t rows, size_t cols,
                        const double *x, const double *scales, double gamma,
                        double *out) {
    size_t i, j;
    gmq_build_luts();
    for (i = 0; i < cols; i++) out[i] = 0.0;
    for (j = 0; j < rows; j++) {
        double xv = x[j];
        size_t start_t, end_t, start_b, end_b, b;
        double xj;
        if (xv == 0.0) continue;
        xj = xv * gamma;
        if (scales) xj *= scales[j];
        start_t = j * cols;
        end_t = start_t + cols;
        start_b = start_t / GMQ_TRITS_PER_BYTE;
        end_b = (end_t + GMQ_TRITS_PER_BYTE - 1) / GMQ_TRITS_PER_BYTE;
        for (b = start_b; b < end_b; b++) {
            uint8_t byte = packed[b];
            size_t base, t;
            if (byte == 0) continue;
            base = b * GMQ_TRITS_PER_BYTE;
            t = base;
            if (t >= start_t && t < end_t) out[t - start_t] += (double)gmq_t0[byte] * xj;
            t = base + 1;
            if (t >= start_t && t < end_t) out[t - start_t] += (double)gmq_t1[byte] * xj;
            t = base + 2;
            if (t >= start_t && t < end_t) out[t - start_t] += (double)gmq_t2[byte] * xj;
            t = base + 3;
            if (t >= start_t && t < end_t) out[t - start_t] += (double)gmq_t3[byte] * xj;
            t = base + 4;
            if (t >= start_t && t < end_t) out[t - start_t] += (double)gmq_t4[byte] * xj;
        }
    }
}

void gmq_dense_matmul(const double *w, size_t rows, size_t cols,
                      const double *x, double *out) {
    size_t i, j;
    for (i = 0; i < cols; i++) out[i] = 0.0;
    for (j = 0; j < rows; j++) {
        double xv = x[j];
        if (xv == 0.0) continue;
        for (i = 0; i < cols; i++) out[i] += w[j * cols + i] * xv;
    }
}

double gmq_relative_error(const double *a, const double *b, size_t n) {
    double num = 0.0, den = 0.0;
    size_t i;
    for (i = 0; i < n; i++) {
        double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    if (den == 0.0) return num == 0.0 ? 0.0 : INFINITY;
    return sqrt(num / den);
}
