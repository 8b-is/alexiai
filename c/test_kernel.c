/*
 * test_kernel.c — the C99 lane's test harness. No framework, no dependencies:
 * a handful of assertions and a non-zero exit on the first failure. The trit
 * dot test cross-checks the assembly path against the portable C reference,
 * so the platform asm is proven correct on whatever machine this runs.
 */

#include "kernel.h"

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

int main(void) {
    printf("gmq C99 test harness — asm path: %s\n", gmq_uses_asm() ? "compiled in" : "not compiled");
    test_round_trip();
    test_byte_242();
    test_asm_dot_matches_c();
    test_kernel_matches_dense();
    test_non_aligned_rows();
    test_relative_error();
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
