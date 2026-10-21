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

/* NEON pure-integer p/q trit dot: 8 trits per iteration in 32-bit integer lanes (0 floats!). */
int64_t gmq_trit_dot_pq(const int8_t *trits, const int32_t *p_x, size_t n) {
    int64_t total = 0;
    size_t offset = 0;
    /* Process in chunks of up to 512 elements so 32-bit accumulators never overflow. */
    while (offset < n) {
        size_t rem = n - offset;
        size_t chunk = rem > 512 ? 512 : rem;
        size_t vec_len = chunk & ~(size_t)7;
        const int8_t *tp = trits + offset;
        const int32_t *xp = p_x + offset;
        const int8_t *tp0 = tp;
        const int32_t *xp0 = xp;
        size_t k = vec_len;
        int32_t chunk_sum = 0;
        if (k > 0) {
            __asm__ volatile(
                "movi v0.4s, #0\n\t"
                "movi v1.4s, #0\n\t"
                "1:\n\t"
                "ld1 {v2.8b}, [%x[t]], #8\n\t"
                "sxtl v3.8h, v2.8b\n\t"
                "sxtl v4.4s, v3.4h\n\t"
                "sxtl2 v5.4s, v3.8h\n\t"
                "ld1 {v6.4s, v7.4s}, [%x[x]], #32\n\t"
                "mla v0.4s, v4.4s, v6.4s\n\t"
                "mla v1.4s, v5.4s, v7.4s\n\t"
                "subs %x[k], %x[k], #8\n\t"
                "b.ne 1b\n\t"
                "add v0.4s, v0.4s, v1.4s\n\t"
                "addv s0, v0.4s\n\t"
                "fmov %w[out], s0\n\t"
                : [out] "=r"(chunk_sum), [t] "+r"(tp), [x] "+r"(xp), [k] "+r"(k)
                :
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
        }
        total += (int64_t)chunk_sum;
        {
            size_t i;
            for (i = vec_len; i < chunk; i++) {
                total += (int64_t)tp0[i] * (int64_t)xp0[i];
            }
        }
        offset += chunk;
    }
    return total;
}

/* NEON pure-integer Omni BFP-i8 dot: 16 trits per iteration via smlal/smlal2 (i8×i8->i16, 0 floats!). */
int64_t gmq_trit_dot_bfp_i8(const int8_t *trits, const int8_t *m_x, size_t n) {
    int64_t total = 0;
    size_t offset = 0;
    /* Process in chunks of up to 256 elements (256 * 127 = 32,512 <= 32,767, zero i16 overflow!). */
    while (offset < n) {
        size_t rem = n - offset;
        size_t chunk = rem > 256 ? 256 : rem;
        size_t vec_len = chunk & ~(size_t)15;
        const int8_t *tp = trits + offset;
        const int8_t *xp = m_x + offset;
        const int8_t *tp0 = tp;
        const int8_t *xp0 = xp;
        size_t k = vec_len;
        int32_t chunk_sum = 0;
        if (k > 0) {
            __asm__ volatile(
                "movi v0.8h, #0\n\t"
                "movi v1.8h, #0\n\t"
                "1:\n\t"
                "ld1 {v2.16b}, [%x[t]], #16\n\t"
                "ld1 {v3.16b}, [%x[x]], #16\n\t"
                "smlal v0.8h, v2.8b, v3.8b\n\t"
                "smlal2 v1.8h, v2.16b, v3.16b\n\t"
                "subs %x[k], %x[k], #16\n\t"
                "b.ne 1b\n\t"
                "saddlp v0.4s, v0.8h\n\t"
                "sadalp v0.4s, v1.8h\n\t"
                "addv s0, v0.4s\n\t"
                "fmov %w[out], s0\n\t"
                : [out] "=r"(chunk_sum), [t] "+r"(tp), [x] "+r"(xp), [k] "+r"(k)
                :
                : "cc", "memory", "v0", "v1", "v2", "v3");
        }
        total += (int64_t)chunk_sum;
        {
            size_t i;
            for (i = vec_len; i < chunk; i++) {
                total += (int64_t)tp0[i] * (int64_t)xp0[i];
            }
        }
        offset += chunk;
    }
    return total;
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
        "test %[k], %[k]\n\t"
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

/* SSE4.1 pure-integer p/q trit dot: 8 trits per iteration via pmovsxbd + pmulld + paddd (0 floats!). */
int64_t gmq_trit_dot_pq(const int8_t *trits, const int32_t *p_x, size_t n) {
    int64_t total = 0;
    size_t offset = 0;
    while (offset < n) {
        size_t rem = n - offset;
        size_t chunk = rem > 512 ? 512 : rem;
        size_t vec_len = chunk & ~(size_t)7;
        const int8_t *tp = trits + offset;
        const int32_t *xp = p_x + offset;
        const int8_t *tp0 = tp;
        const int32_t *xp0 = xp;
        size_t k = vec_len;
        int32_t lanes[4] = {0, 0, 0, 0};
        if (k > 0) {
            __asm__ volatile(
                "pxor %%xmm0, %%xmm0\n\t"
                "pxor %%xmm1, %%xmm1\n\t"
                "1:\n\t"
                "movd (%[t]), %%xmm2\n\t"
                "movd 4(%[t]), %%xmm3\n\t"
                "pmovsxbd %%xmm2, %%xmm2\n\t"
                "pmovsxbd %%xmm3, %%xmm3\n\t"
                "movdqu (%[x]), %%xmm4\n\t"
                "movdqu 16(%[x]), %%xmm5\n\t"
                "pmulld %%xmm4, %%xmm2\n\t"
                "pmulld %%xmm5, %%xmm3\n\t"
                "paddd %%xmm2, %%xmm0\n\t"
                "paddd %%xmm3, %%xmm1\n\t"
                "add $8, %[t]\n\t"
                "add $32, %[x]\n\t"
                "sub $8, %[k]\n\t"
                "jnz 1b\n\t"
                "paddd %%xmm1, %%xmm0\n\t"
                "movdqu %%xmm0, %[lanes]\n\t"
                : [lanes] "=m"(lanes), [t] "+r"(tp), [x] "+r"(xp), [k] "+r"(k)
                :
                : "cc", "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
            total += (int64_t)lanes[0] + (int64_t)lanes[1] + (int64_t)lanes[2] + (int64_t)lanes[3];
        }
        {
            size_t i;
            for (i = vec_len; i < chunk; i++) {
                total += (int64_t)tp0[i] * (int64_t)xp0[i];
            }
        }
        offset += chunk;
    }
    return total;
}

/* SSE4.1 pure-integer Omni BFP-i8 dot: 16 trits per iteration via pmovsxbw + pmullw + paddw (0 floats!). */
int64_t gmq_trit_dot_bfp_i8(const int8_t *trits, const int8_t *m_x, size_t n) {
    int64_t total = 0;
    size_t offset = 0;
    while (offset < n) {
        size_t rem = n - offset;
        size_t chunk = rem > 256 ? 256 : rem;
        size_t vec_len = chunk & ~(size_t)15;
        const int8_t *tp = trits + offset;
        const int8_t *xp = m_x + offset;
        const int8_t *tp0 = tp;
        const int8_t *xp0 = xp;
        size_t k = vec_len;
        int16_t lanes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (k > 0) {
            __asm__ volatile(
                "pxor %%xmm0, %%xmm0\n\t"
                "pxor %%xmm1, %%xmm1\n\t"
                "1:\n\t"
                "movq (%[t]), %%xmm2\n\t"
                "movq 8(%[t]), %%xmm3\n\t"
                "pmovsxbw %%xmm2, %%xmm2\n\t"
                "pmovsxbw %%xmm3, %%xmm3\n\t"
                "movq (%[x]), %%xmm4\n\t"
                "movq 8(%[x]), %%xmm5\n\t"
                "pmovsxbw %%xmm4, %%xmm4\n\t"
                "pmovsxbw %%xmm5, %%xmm5\n\t"
                "pmullw %%xmm4, %%xmm2\n\t"
                "pmullw %%xmm5, %%xmm3\n\t"
                "paddw %%xmm2, %%xmm0\n\t"
                "paddw %%xmm3, %%xmm1\n\t"
                "add $16, %[t]\n\t"
                "add $16, %[x]\n\t"
                "sub $16, %[k]\n\t"
                "jnz 1b\n\t"
                "paddw %%xmm1, %%xmm0\n\t"
                "movdqu %%xmm0, %[lanes]\n\t"
                : [lanes] "=m"(lanes), [t] "+r"(tp), [x] "+r"(xp), [k] "+r"(k)
                :
                : "cc", "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
            {
                int l;
                for (l = 0; l < 8; l++) total += (int64_t)lanes[l];
            }
        }
        {
            size_t i;
            for (i = vec_len; i < chunk; i++) {
                total += (int64_t)tp0[i] * (int64_t)xp0[i];
            }
        }
        offset += chunk;
    }
    return total;
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
int64_t gmq_trit_dot_pq(const int8_t *trits, const int32_t *p_x, size_t n) {
    int64_t acc = 0;
    size_t i;
    for (i = 0; i < n; i++) acc += (int64_t)trits[i] * (int64_t)p_x[i];
    return acc;
}
int64_t gmq_trit_dot_bfp_i8(const int8_t *trits, const int8_t *m_x, size_t n) {
    int64_t acc = 0;
    size_t i;
    for (i = 0; i < n; i++) acc += (int64_t)trits[i] * (int64_t)m_x[i];
    return acc;
}
int gmq_uses_asm(void) { return 0; }
#endif

/* ── the kernels ─────────────────────────────────────────────────────────── */

void gmq_quantize_per_row_pq(const int32_t *w_p, size_t rows, size_t cols,
                             int32_t q, int8_t *trits_out, int32_t *scales_p_out) {
    size_t r, c;
    int64_t cols_i64 = (int64_t)cols;
    int64_t two_cols = cols_i64 * 2;
    for (r = 0; r < rows; r++) {
        const int32_t *row = &w_p[r * cols];
        int8_t *dst = &trits_out[r * cols];
        int64_t sum_abs = 0;
        for (c = 0; c < cols; c++) {
            int64_t v = (int64_t)row[c];
            sum_abs += v < 0 ? -v : v;
        }
        if (sum_abs == 0) {
            scales_p_out[r] = q;
            for (c = 0; c < cols; c++) dst[c] = 0;
            continue;
        }
        scales_p_out[r] = (int32_t)((sum_abs + (cols_i64 >> 1)) / cols_i64);
        for (c = 0; c < cols; c++) {
            int64_t v = (int64_t)row[c];
            int64_t av = v < 0 ? -v : v;
            if (av * two_cols < sum_abs) {
                dst[c] = 0;
            } else {
                dst[c] = v > 0 ? 1 : -1;
            }
        }
    }
}

void gmq_ternary_matmul_pq(const uint8_t *packed, size_t rows, size_t cols,
                           const int32_t *p_x, const int32_t *p_scales,
                           int32_t q_scale, int64_t *p_out) {
    size_t i, j;
    int64_t q_s = q_scale > 0 ? (int64_t)q_scale : 1;
    int64_t half_q = q_s >> 1;
    gmq_build_luts();
    for (i = 0; i < cols; i++) p_out[i] = 0;
    for (j = 0; j < rows; j++) {
        int64_t xj = (int64_t)p_x[j];
        size_t start_t, end_t, start_b, end_b, b;
        if (xj == 0) continue;
        if (p_scales) {
            int64_t prod = xj * (int64_t)p_scales[j];
            xj = prod >= 0 ? (prod + half_q) / q_s : (prod - half_q) / q_s;
            if (xj == 0) continue;
        }
        start_t = j * cols;
        end_t = start_t + cols;
        start_b = start_t / GMQ_TRITS_PER_BYTE;
        end_b = (end_t + GMQ_TRITS_PER_BYTE - 1) / GMQ_TRITS_PER_BYTE;
        for (b = start_b; b < end_b; b++) {
            uint8_t byte = packed[b];
            size_t base;
            if (byte == 0) continue;
            base = b * GMQ_TRITS_PER_BYTE;
            if (base >= start_t && base + GMQ_TRITS_PER_BYTE <= end_t) {
                size_t col = base - start_t;
                p_out[col + 0] += (int64_t)gmq_t0[byte] * xj;
                p_out[col + 1] += (int64_t)gmq_t1[byte] * xj;
                p_out[col + 2] += (int64_t)gmq_t2[byte] * xj;
                p_out[col + 3] += (int64_t)gmq_t3[byte] * xj;
                p_out[col + 4] += (int64_t)gmq_t4[byte] * xj;
            } else {
                size_t t = base;
                if (t >= start_t && t < end_t) p_out[t - start_t] += (int64_t)gmq_t0[byte] * xj;
                t = base + 1;
                if (t >= start_t && t < end_t) p_out[t - start_t] += (int64_t)gmq_t1[byte] * xj;
                t = base + 2;
                if (t >= start_t && t < end_t) p_out[t - start_t] += (int64_t)gmq_t2[byte] * xj;
                t = base + 3;
                if (t >= start_t && t < end_t) p_out[t - start_t] += (int64_t)gmq_t3[byte] * xj;
                t = base + 4;
                if (t >= start_t && t < end_t) p_out[t - start_t] += (int64_t)gmq_t4[byte] * xj;
            }
        }
    }
}

void gmq_simd_matmul_pq(const int8_t *trits_i_major, size_t rows, size_t cols,
                        const int32_t *p_x, int64_t *p_out) {
    size_t i;
    for (i = 0; i < cols; i++) {
        p_out[i] = gmq_trit_dot_pq(&trits_i_major[i * rows], p_x, rows);
    }
}

void gmq_omni_bfp_i8_matmul_pq(const int8_t *trits_i_major, size_t rows, size_t cols,
                               const int8_t *m_x, int64_t *p_out) {
    size_t i = 0;
#if defined(__aarch64__)
    while (i + 4 <= cols) {
        int64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        size_t offset = 0;
        while (offset < rows) {
            size_t rem = rows - offset;
            size_t chunk = rem > 256 ? 256 : rem;
            size_t vec_len = chunk & ~(size_t)15;
            const int8_t *w0 = &trits_i_major[(i + 0) * rows + offset];
            const int8_t *w1 = &trits_i_major[(i + 1) * rows + offset];
            const int8_t *w2 = &trits_i_major[(i + 2) * rows + offset];
            const int8_t *w3 = &trits_i_major[(i + 3) * rows + offset];
            const int8_t *xp = m_x + offset;
            const int8_t *w0_0 = w0, *w1_0 = w1, *w2_0 = w2, *w3_0 = w3, *xp0 = xp;
            size_t k = vec_len;
            int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            if (k > 0) {
                __asm__ volatile(
                    "movi v0.8h, #0\n\t"
                    "movi v1.8h, #0\n\t"
                    "movi v2.8h, #0\n\t"
                    "movi v3.8h, #0\n\t"
                    "movi v4.8h, #0\n\t"
                    "movi v5.8h, #0\n\t"
                    "movi v6.8h, #0\n\t"
                    "movi v7.8h, #0\n\t"
                    "1:\n\t"
                    "ld1 {v16.16b}, [%x[x]], #16\n\t"
                    "ld1 {v17.16b}, [%x[c0]], #16\n\t"
                    "ld1 {v18.16b}, [%x[c1]], #16\n\t"
                    "ld1 {v19.16b}, [%x[c2]], #16\n\t"
                    "ld1 {v20.16b}, [%x[c3]], #16\n\t"
                    "smlal v0.8h, v17.8b, v16.8b\n\t"
                    "smlal2 v1.8h, v17.16b, v16.16b\n\t"
                    "smlal v2.8h, v18.8b, v16.8b\n\t"
                    "smlal2 v3.8h, v18.16b, v16.16b\n\t"
                    "smlal v4.8h, v19.8b, v16.8b\n\t"
                    "smlal2 v5.8h, v19.16b, v16.16b\n\t"
                    "smlal v6.8h, v20.8b, v16.8b\n\t"
                    "smlal2 v7.8h, v20.16b, v16.16b\n\t"
                    "subs %x[k], %x[k], #16\n\t"
                    "b.ne 1b\n\t"
                    "saddlp v0.4s, v0.8h\n\t"
                    "sadalp v0.4s, v1.8h\n\t"
                    "addv s0, v0.4s\n\t"
                    "fmov %w[o0], s0\n\t"
                    "saddlp v2.4s, v2.8h\n\t"
                    "sadalp v2.4s, v3.8h\n\t"
                    "addv s2, v2.4s\n\t"
                    "fmov %w[o1], s2\n\t"
                    "saddlp v4.4s, v4.8h\n\t"
                    "sadalp v4.4s, v5.8h\n\t"
                    "addv s4, v4.4s\n\t"
                    "fmov %w[o2], s4\n\t"
                    "saddlp v6.4s, v6.8h\n\t"
                    "sadalp v6.4s, v7.8h\n\t"
                    "addv s6, v6.4s\n\t"
                    "fmov %w[o3], s6\n\t"
                    : [o0] "=r"(s0), [o1] "=r"(s1), [o2] "=r"(s2), [o3] "=r"(s3),
                      [c0] "+r"(w0), [c1] "+r"(w1), [c2] "+r"(w2), [c3] "+r"(w3),
                      [x] "+r"(xp), [k] "+r"(k)
                    :
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                      "v16", "v17", "v18", "v19", "v20");
            }
            t0 += (int64_t)s0;
            t1 += (int64_t)s1;
            t2 += (int64_t)s2;
            t3 += (int64_t)s3;
            {
                size_t j;
                for (j = vec_len; j < chunk; j++) {
                    int64_t xv = (int64_t)xp0[j];
                    t0 += (int64_t)w0_0[j] * xv;
                    t1 += (int64_t)w1_0[j] * xv;
                    t2 += (int64_t)w2_0[j] * xv;
                    t3 += (int64_t)w3_0[j] * xv;
                }
            }
            offset += chunk;
        }
        p_out[i + 0] = t0;
        p_out[i + 1] = t1;
        p_out[i + 2] = t2;
        p_out[i + 3] = t3;
        i += 4;
    }
#endif
    for (; i < cols; i++) {
        p_out[i] = gmq_trit_dot_bfp_i8(&trits_i_major[i * rows], m_x, rows);
    }
}

void gmq_pack_quantum_bitplanes_i_major(const int8_t *trits, size_t rows, size_t cols,
                                        uint64_t *w_pos_u64, uint64_t *w_neg_u64) {
    size_t words = (rows + 63) / 64;
    size_t i, w, b;
    for (i = 0; i < cols; i++) {
        for (w = 0; w < words; w++) {
            uint64_t p_mask = 0, n_mask = 0;
            size_t j_base = w * 64;
            size_t j_end = j_base + 64 < rows ? j_base + 64 : rows;
            for (b = 0; b < j_end - j_base; b++) {
                int8_t t = trits[(j_base + b) * cols + i];
                if (t > 0) p_mask |= (1ull << b);
                else if (t < 0) n_mask |= (1ull << b);
            }
            w_pos_u64[i * words + w] = p_mask;
            w_neg_u64[i * words + w] = n_mask;
        }
    }
}

int32_t gmq_quantize_quantum_act_pq(const int32_t *p_x, size_t n,
                                    uint64_t *x_pos_u64, uint64_t *x_neg_u64) {
    size_t words = (n + 63) / 64;
    size_t i, w, b;
    int64_t sum_abs = 0, n64 = (int64_t)n, two_n = n64 * 2;
    for (w = 0; w < words; w++) {
        x_pos_u64[w] = 0;
        x_neg_u64[w] = 0;
    }
    if (n == 0) return 0;
    for (i = 0; i < n; i++) {
        int64_t v = (int64_t)p_x[i];
        sum_abs += v < 0 ? -v : v;
    }
    if (sum_abs == 0) return 0;
    for (w = 0; w < words; w++) {
        uint64_t pm = 0, nm = 0;
        size_t j_base = w * 64;
        size_t j_end = j_base + 64 < n ? j_base + 64 : n;
        for (b = 0; b < j_end - j_base; b++) {
            int64_t v = (int64_t)p_x[j_base + b];
            int64_t av = v < 0 ? -v : v;
            if (av * two_n >= sum_abs) {
                if (v > 0) pm |= (1ull << b);
                else if (v < 0) nm |= (1ull << b);
            }
        }
        x_pos_u64[w] = pm;
        x_neg_u64[w] = nm;
    }
    return (int32_t)((sum_abs + (n64 >> 1)) / n64);
}

void gmq_quantum_bitplane_matmul_pq(const uint64_t *w_pos_u64, const uint64_t *w_neg_u64,
                                    size_t rows, size_t cols,
                                    const uint64_t *x_pos_u64, const uint64_t *x_neg_u64,
                                    int32_t p_scale_x, int64_t *p_out) {
    size_t words = (rows + 63) / 64;
    int64_t sx = (int64_t)p_scale_x;
    size_t i = 0, w;
    while (i + 8 <= cols) {
        const uint64_t *wp0 = &w_pos_u64[(i + 0) * words], *wn0 = &w_neg_u64[(i + 0) * words];
        const uint64_t *wp1 = &w_pos_u64[(i + 1) * words], *wn1 = &w_neg_u64[(i + 1) * words];
        const uint64_t *wp2 = &w_pos_u64[(i + 2) * words], *wn2 = &w_neg_u64[(i + 2) * words];
        const uint64_t *wp3 = &w_pos_u64[(i + 3) * words], *wn3 = &w_neg_u64[(i + 3) * words];
        const uint64_t *wp4 = &w_pos_u64[(i + 4) * words], *wn4 = &w_neg_u64[(i + 4) * words];
        const uint64_t *wp5 = &w_pos_u64[(i + 5) * words], *wn5 = &w_neg_u64[(i + 5) * words];
        const uint64_t *wp6 = &w_pos_u64[(i + 6) * words], *wn6 = &w_neg_u64[(i + 6) * words];
        const uint64_t *wp7 = &w_pos_u64[(i + 7) * words], *wn7 = &w_neg_u64[(i + 7) * words];
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        for (w = 0; w < words; w++) {
            uint64_t xp = x_pos_u64[w], xn = x_neg_u64[w];
            a0 += __builtin_popcountll((wp0[w] & xp) | (wn0[w] & xn)) - __builtin_popcountll((wp0[w] & xn) | (wn0[w] & xp));
            a1 += __builtin_popcountll((wp1[w] & xp) | (wn1[w] & xn)) - __builtin_popcountll((wp1[w] & xn) | (wn1[w] & xp));
            a2 += __builtin_popcountll((wp2[w] & xp) | (wn2[w] & xn)) - __builtin_popcountll((wp2[w] & xn) | (wn2[w] & xp));
            a3 += __builtin_popcountll((wp3[w] & xp) | (wn3[w] & xn)) - __builtin_popcountll((wp3[w] & xn) | (wn3[w] & xp));
            a4 += __builtin_popcountll((wp4[w] & xp) | (wn4[w] & xn)) - __builtin_popcountll((wp4[w] & xn) | (wn4[w] & xp));
            a5 += __builtin_popcountll((wp5[w] & xp) | (wn5[w] & xn)) - __builtin_popcountll((wp5[w] & xn) | (wn5[w] & xp));
            a6 += __builtin_popcountll((wp6[w] & xp) | (wn6[w] & xn)) - __builtin_popcountll((wp6[w] & xn) | (wn6[w] & xp));
            a7 += __builtin_popcountll((wp7[w] & xp) | (wn7[w] & xn)) - __builtin_popcountll((wp7[w] & xn) | (wn7[w] & xp));
        }
        p_out[i + 0] = (int64_t)a0 * sx;
        p_out[i + 1] = (int64_t)a1 * sx;
        p_out[i + 2] = (int64_t)a2 * sx;
        p_out[i + 3] = (int64_t)a3 * sx;
        p_out[i + 4] = (int64_t)a4 * sx;
        p_out[i + 5] = (int64_t)a5 * sx;
        p_out[i + 6] = (int64_t)a6 * sx;
        p_out[i + 7] = (int64_t)a7 * sx;
        i += 8;
    }
    while (i < cols) {
        const uint64_t *wp = &w_pos_u64[i * words], *wn = &w_neg_u64[i * words];
        int32_t a0 = 0;
        for (w = 0; w < words; w++) {
            uint64_t xp = x_pos_u64[w], xn = x_neg_u64[w];
            a0 += __builtin_popcountll((wp[w] & xp) | (wn[w] & xn)) - __builtin_popcountll((wp[w] & xn) | (wn[w] & xp));
        }
        p_out[i] = (int64_t)a0 * sx;
        i += 1;
    }
}

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
            size_t base;
            if (byte == 0) continue;
            base = b * GMQ_TRITS_PER_BYTE;
            if (base >= start_t && base + GMQ_TRITS_PER_BYTE <= end_t) {
                size_t col = base - start_t;
                out[col + 0] += (double)gmq_t0[byte] * xj;
                out[col + 1] += (double)gmq_t1[byte] * xj;
                out[col + 2] += (double)gmq_t2[byte] * xj;
                out[col + 3] += (double)gmq_t3[byte] * xj;
                out[col + 4] += (double)gmq_t4[byte] * xj;
            } else {
                size_t t = base;
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

static double gmq_clamp(double v, double min_v, double max_v) {
    if (isnan(v) || v < min_v) return min_v;
    if (v > max_v) return max_v;
    return v;
}

uint32_t gmq_seed_from(const char *prompt, size_t len) {
    uint32_t h = 0x811c9dc5u;
    size_t i;
    for (i = 0; i < len; i++) {
        h ^= (uint32_t)(uint8_t)prompt[i];
        h *= 0x01000193u;
    }
    return h;
}

gmq_fold_t gmq_fold_spectrum(const char *prompt, size_t len, double rf_entropy_bits) {
    gmq_fold_t f;
    uint32_t s = gmq_seed_from(prompt, len);
    uint32_t a = s + 0x6d2b79f5u;
    uint32_t t = (a ^ (a >> 15)) * (a | 1u);
    double rand0, rf_factor, len_factor, spread;

    t ^= t + ((t ^ (t >> 7)) * (t | 61u));
    rand0 = (double)((t ^ (t >> 14)) >> 8) / 16777216.0;

    rf_factor = gmq_clamp((rf_entropy_bits - 2.5) * 0.05, -0.08, 0.14);
    len_factor = (double)len / 4096.0;
    if (len_factor > 0.35) len_factor = 0.35;

    f.seed = s;
    f.gravity = 1.0;
    f.entropy = gmq_clamp(0.62 + len_factor - 0.15 + rf_factor, 0.0, 1.0);
    f.weather = gmq_clamp(0.5 + (rand0 - 0.5) * 0.4, 0.0, 1.0);
    f.resonance = 1.618033988749894848204586834365638118;
    spread = fabs(f.entropy - 0.62) * 0.6 + fabs(f.weather - 0.5) * 0.4;
    f.coherence = gmq_clamp(1.0 - spread, 0.0, 1.0);
    return f;
}

static int64_t gmq_clamp_i64(int64_t v, int64_t min_v, int64_t max_v) {
    if (v < min_v) return min_v;
    if (v > max_v) return max_v;
    return v;
}

static int64_t gmq_abs_i64(int64_t v) {
    return v < 0 ? -v : v;
}

gmq_fold_pq_t gmq_fold_spectrum_pq(const char *prompt, size_t len, int64_t rf_entropy_p) {
    gmq_fold_pq_t f;
    uint32_t s = gmq_seed_from(prompt, len);
    uint32_t a = s + 0x6d2b79f5u;
    uint32_t t = (a ^ (a >> 15)) * (a | 1u);
    int64_t rand_p, rf_factor_p, len_p, spread_p;

    t ^= t + ((t ^ (t >> 7)) * (t | 61u));
    rand_p = (int64_t)((t ^ (t >> 14)) >> 12); /* in [0, 2^20) */

    rf_factor_p = gmq_clamp_i64((rf_entropy_p - 2621440) / 20, -83886, 146801);
    len_p = ((int64_t)len) << 8;
    if (len_p > 367002) len_p = 367002;

    f.seed = s;
    f.q = GMQ_PQ_DEFAULT_Q;
    f.gravity_p = 1048576;
    f.entropy_p = gmq_clamp_i64(650117 + len_p - 157286 + rf_factor_p, 0, GMQ_PQ_DEFAULT_Q);
    f.weather_p = gmq_clamp_i64(524288 + ((rand_p - 524288) * 2) / 5, 0, GMQ_PQ_DEFAULT_Q);
    f.resonance_p = 1696632;
    spread_p = (gmq_abs_i64(f.entropy_p - 650117) * 3 + gmq_abs_i64(f.weather_p - 524288) * 2) / 5;
    f.coherence_p = gmq_clamp_i64(GMQ_PQ_DEFAULT_Q - spread_p, 0, GMQ_PQ_DEFAULT_Q);
    return f;
}

uint64_t gmq_wupz_xor_fold(const uint8_t *ptr, size_t len) {
    uint64_t acc = 0;
    size_t i = 0;
    if (!ptr || len == 0) return 0;
    while (i + 8 <= len) {
        uint64_t w = (uint64_t)ptr[i + 0]
                   | ((uint64_t)ptr[i + 1] << 8)
                   | ((uint64_t)ptr[i + 2] << 16)
                   | ((uint64_t)ptr[i + 3] << 24)
                   | ((uint64_t)ptr[i + 4] << 32)
                   | ((uint64_t)ptr[i + 5] << 40)
                   | ((uint64_t)ptr[i + 6] << 48)
                   | ((uint64_t)ptr[i + 7] << 56);
        acc ^= w;
        i += 8;
    }
    while (i < len) {
        acc ^= (uint64_t)ptr[i];
        i += 1;
    }
    return acc;
}

uint64_t gmq_wupz_byte_diff(const uint8_t *a, const uint8_t *b, size_t len) {
    uint64_t count = 0;
    size_t i;
    if (!a || !b || len == 0) return 0;
    for (i = 0; i < len; i++) {
        if (a[i] != b[i]) count += 1;
    }
    return count;
}

uint64_t gmq_wupz_popcount(const uint8_t *ptr, size_t len) {
    uint64_t count = 0;
    size_t i = 0;
    if (!ptr || len == 0) return 0;
    while (i + 8 <= len) {
        uint64_t w = (uint64_t)ptr[i + 0]
                   | ((uint64_t)ptr[i + 1] << 8)
                   | ((uint64_t)ptr[i + 2] << 16)
                   | ((uint64_t)ptr[i + 3] << 24)
                   | ((uint64_t)ptr[i + 4] << 32)
                   | ((uint64_t)ptr[i + 5] << 40)
                   | ((uint64_t)ptr[i + 6] << 48)
                   | ((uint64_t)ptr[i + 7] << 56);
        count += (uint64_t)__builtin_popcountll(w);
        i += 8;
    }
    while (i < len) {
        count += (uint64_t)__builtin_popcount((unsigned int)ptr[i]);
        i += 1;
    }
    return count;
}

static int8_t gmq_quant_tern_trit_pq(int64_t p_x, int64_t q) {
    int64_t q_abs = q < 0 ? -q : q;
    int64_t lhs = p_x * 100;
    int64_t rhs = 33 * q_abs;
    if (lhs >= rhs) return 1;
    if (lhs <= -rhs) return -1;
    return 0;
}

uint8_t gmq_quant_tern_encode_pq(int64_t p_valence, int64_t p_arousal, int64_t p_dominance, int64_t q) {
    int8_t t0 = gmq_quant_tern_trit_pq(p_valence, q);
    int8_t t1 = gmq_quant_tern_trit_pq(p_arousal, q);
    int8_t t2 = gmq_quant_tern_trit_pq(p_dominance, q);
    uint8_t b0 = (uint8_t)((t0 + 1) & 3);
    uint8_t b1 = (uint8_t)(((t1 + 1) & 3) << 2);
    uint8_t b2 = (uint8_t)(((t2 + 1) & 3) << 4);
    uint8_t b3 = (uint8_t)(1u << 6); /* 4th slot is 0 -> (0 + 1) << 6 */
    return (uint8_t)(b0 | b1 | b2 | b3);
}
