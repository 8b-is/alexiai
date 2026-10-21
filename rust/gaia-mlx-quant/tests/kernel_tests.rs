//! Kernel correctness: exact on trits, honest on quantization.

use gaia_mlx_quant::{
    dense_matmul, pack_ternary, pack_ternary_i_major_into, relative_error, ternary_matmul,
    ternary_matmul_i_major,
};

fn dense_weights_from_trits(trits: &[i8]) -> Vec<f64> {
    trits.iter().map(|&t| t as f64).collect()
}

#[test]
fn kernel_matches_dense_exactly_on_exact_trits() {
    let rows = 3usize;
    let cols = 5usize;
    let trits = [
        1i8, -1, 0, 1, -1, //
        -1, 1, 1, 0, 0, //
        0, 0, -1, 1, 1, //
    ];
    let x = [2.0, -1.0, 3.0];

    let packed = pack_ternary(&trits);
    let mut ternary = [0f64; 5];
    ternary_matmul(&packed, rows, cols, &x, None, 1.0, &mut ternary);

    let mut reference = [0f64; 5];
    dense_matmul(&dense_weights_from_trits(&trits), rows, cols, &x, &mut reference);

    assert_eq!(ternary, reference);
}

#[test]
fn kernel_respects_per_row_scales() {
    let rows = 2usize;
    let cols = 5usize;
    let trits = [1i8, -1, 0, 1, -1, 0, 1, 1, -1, 0];
    let x = [2.0, 3.0];
    let scales = [0.5, 2.0];

    let packed = pack_ternary(&trits);
    let mut out = [0f64; 5];
    ternary_matmul(&packed, rows, cols, &x, Some(&scales), 1.0, &mut out);

    // y[i] = scales[0]*trits[0][i]*2 + scales[1]*trits[1][i]*3
    let expected = [
        0.5 * 1.0 * 2.0 + 2.0 * 0.0 * 3.0,
        -0.5 * 2.0 + 2.0 * 1.0 * 3.0,
        0.5 * 0.0 * 2.0 + 2.0 * 1.0 * 3.0,
        0.5 * 1.0 * 2.0 + -2.0 * 3.0,
        -0.5 * 2.0 + 2.0 * 0.0 * 3.0,
    ];
    assert_eq!(out, expected);
}

#[test]
fn gamma_scales_the_whole_result() {
    let rows = 1usize;
    let cols = 5usize;
    let trits = [1i8, 1, 1, 1, 1];
    let x = [2.0];
    let packed = pack_ternary(&trits);
    let mut out = [0f64; 5];
    ternary_matmul(&packed, rows, cols, &x, None, 3.0, &mut out);
    assert_eq!(out, [6.0; 5]);
}

#[test]
fn non_aligned_row_boundaries_are_correct() {
    // cols=3 means row boundaries fall inside bytes (3 % 5 != 0).
    let rows = 4usize;
    let cols = 3usize;
    let trits: Vec<i8> = (0..rows * cols).map(|i| [-1, 0, 1][i % 3]).collect();
    let x = [1.0, 2.0, 3.0, 4.0];

    let packed = pack_ternary(&trits);
    let mut out = vec![0f64; cols];
    ternary_matmul(&packed, rows, cols, &x, None, 1.0, &mut out);

    let mut reference = vec![0f64; cols];
    dense_matmul(&dense_weights_from_trits(&trits), rows, cols, &x, &mut reference);

    assert_eq!(out, reference);
}

#[test]
fn zero_input_rows_are_skipped() {
    let rows = 3usize;
    let cols = 5usize;
    let trits = [1i8; 15];
    let x = [0.0, 2.0, 0.0];
    let packed = pack_ternary(&trits);
    let mut out = [0f64; 5];
    ternary_matmul(&packed, rows, cols, &x, None, 1.0, &mut out);
    // Only row 1 contributes.
    assert_eq!(out, [2.0; 5]);
}

#[test]
fn dense_reference_matches_naive_expectation() {
    let weights = [1.0, 2.0, 3.0, 4.0];
    let x = [2.0, -1.0];
    let mut out = [0f64; 2];
    dense_matmul(&weights, 2, 2, &x, &mut out);
    assert_eq!(out, [1.0 * 2.0 + -3.0, 2.0 * 2.0 + -4.0]);
}

#[test]
fn relative_error_is_zero_for_identical_vectors() {
    assert_eq!(relative_error(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    assert!(relative_error(&[2.0], &[1.0]) > 0.9);
}

#[test]
fn i_major_layout_matches_dense_and_j_major() {
    for (rows, cols) in [(4usize, 5usize), (5, 5), (64, 64), (7, 3), (3, 7)] {
        let trits: Vec<i8> = (0..rows * cols).map(|k| [-1, 0, 1][(k * 5 + 1) % 3]).collect();
        let x: Vec<f64> = (0..rows).map(|j| (j as f64 + 0.25) * 0.7).collect();
        let scales: Vec<f64> = (0..rows).map(|j| 0.4 + (j % 3) as f64 * 0.3).collect();

        let packed_j = pack_ternary(&trits);
        let mut out_j = vec![0f64; cols];
        ternary_matmul(&packed_j, rows, cols, &x, Some(&scales), 1.0, &mut out_j);

        let mut packed_i = vec![0u8; (rows * cols).div_ceil(5)];
        pack_ternary_i_major_into(&trits, rows, cols, &mut packed_i);
        let mut out_i = vec![0f64; cols];
        ternary_matmul_i_major(&packed_i, rows, cols, &x, Some(&scales), 1.0, &mut out_i);

        // Hand-computed reference: y[i] = sum_j trit[j][i] * x[j] * scale[j].
        let scaled_reference: Vec<f64> = (0..cols)
            .map(|i| {
                (0..rows)
                    .map(|j| trits[j * cols + i] as f64 * x[j] * scales[j])
                    .sum()
            })
            .collect();

        assert_eq!(out_j, out_i, "j-major and i-major disagree at {rows}x{cols}");
        assert_eq!(out_i, scaled_reference, "i-major vs reference at {rows}x{cols}");
    }
}

#[test]
fn i_major_gamma_scales_the_result() {
    let rows = 2usize;
    let cols = 5usize;
    let trits = [1i8; 10];
    let x = [2.0, 3.0];
    let mut packed = [0u8; 2];
    pack_ternary_i_major_into(&trits, rows, cols, &mut packed);
    let mut out = [0f64; 5];
    ternary_matmul_i_major(&packed, rows, cols, &x, None, 2.0, &mut out);
    assert_eq!(out, [10.0; 5]); // (2 + 3) * 2
}

#[test]
fn plane_kernel_matches_the_reference_at_odd_shapes() {
    use gaia_mlx_quant::{pack_ternary_planes_i_into, ternary_matmul_planes_i};

    for (rows, cols) in [(1usize, 5usize), (4, 5), (64, 64), (7, 3), (2, 2)] {
        let trits: Vec<i8> = (0..rows * cols).map(|k| [-1, 0, 1][(k * 3 + 2) % 3]).collect();
        let x: Vec<f64> = (0..rows).map(|j| (j as f64 - 1.0) * 0.6).collect();
        let scales: Vec<f64> = (0..rows).map(|j| 0.3 + (j % 5) as f64 * 0.2).collect();

        let mut pos = vec![0i8; rows * cols];
        let mut neg = vec![0i8; rows * cols];
        pack_ternary_planes_i_into(&trits, rows, cols, &mut pos, &mut neg);

        let mut out = vec![0f64; cols];
        ternary_matmul_planes_i(&pos, &neg, rows, cols, &x, Some(&scales), 1.0, &mut out);

        let expected: Vec<f64> = (0..cols)
            .map(|i| {
                (0..rows)
                    .map(|j| trits[j * cols + i] as f64 * x[j] * scales[j])
                    .sum()
            })
            .collect();
        assert_eq!(out, expected, "planes vs reference at {rows}x{cols}");
    }
}

#[test]
fn plane_kernel_applies_gamma() {
    use gaia_mlx_quant::{pack_ternary_planes_i_into, ternary_matmul_planes_i};

    let rows = 3usize;
    let cols = 2usize;
    let trits = [1i8, 1, 1, -1, -1, -1];
    let x = [1.0, 2.0, 3.0];
    let mut pos = vec![0i8; 6];
    let mut neg = vec![0i8; 6];
    pack_ternary_planes_i_into(&trits, rows, cols, &mut pos, &mut neg);
    let mut out = [0f64; 2];
    ternary_matmul_planes_i(&pos, &neg, rows, cols, &x, None, 0.5, &mut out);
    // col 0: 1*1 + 1*2 + (-1)*3 = 0; col 1: 1*1 + (-1)*2 + (-1)*3 = -4.
    assert_eq!(out, [0.0, -2.0]);
}

#[test]
fn pure_integer_pq_kernels_match_dense_pq_and_zero_floats() {
    use gaia_mlx_quant::{
        KERNEL_PQ_Q, dense_matmul_pq_into, isqrt_u128, pack_ternary_planes_i_into,
        quantize_ternary_pq_into, relative_error_pq_ppm, ternary_matmul_pq_i_major_into,
        ternary_matmul_pq_j_major_into, ternary_matmul_pq_planes_i_into, ternary_simd_i8_pq_into,
    };

    assert_eq!(isqrt_u128(0), 0);
    assert_eq!(isqrt_u128(1), 1);
    assert_eq!(isqrt_u128(1_000_000_000_000), 1_000_000);

    let rows = 8usize;
    let cols = 10usize;
    let q = KERNEL_PQ_Q;
    let p_w: Vec<i32> = (0..rows * cols)
        .map(|k| match (k * 5 + 1) % 3 {
            0 => -q,
            1 => 0,
            _ => q,
        })
        .collect();
    let mut trits = vec![0i8; rows * cols];
    let mut p_scales = vec![0i32; rows];
    quantize_ternary_pq_into(&p_w, rows, cols, KERNEL_PQ_Q, &mut trits, &mut p_scales);

    let packed_j = pack_ternary(&trits);
    let mut packed_i = vec![0u8; (rows * cols).div_ceil(5)];
    pack_ternary_i_major_into(&trits, rows, cols, &mut packed_i);
    let mut pos = vec![0i8; rows * cols];
    let mut neg = vec![0i8; rows * cols];
    pack_ternary_planes_i_into(&trits, rows, cols, &mut pos, &mut neg);
    let mut trits_i = vec![0i8; rows * cols];
    for j in 0..rows {
        for i in 0..cols {
            trits_i[i * rows + j] = trits[j * cols + i];
        }
    }

    let p_x: Vec<i32> = (0..rows).map(|j| ((j as i32) - 3) * (q / 4)).collect();
    let mut out_j = vec![0i64; cols];
    let mut out_i = vec![0i64; cols];
    let mut out_planes = vec![0i64; cols];
    let mut out_simd = vec![0i64; cols];

    ternary_matmul_pq_j_major_into(&packed_j, rows, cols, &p_x, Some(&p_scales), KERNEL_PQ_Q, &mut out_j);
    ternary_matmul_pq_i_major_into(&packed_i, rows, cols, &p_x, Some(&p_scales), KERNEL_PQ_Q, &mut out_i);
    ternary_matmul_pq_planes_i_into(&pos, &neg, rows, cols, &p_x, Some(&p_scales), KERNEL_PQ_Q, &mut out_planes);
    ternary_simd_i8_pq_into(&trits_i, rows, cols, &p_x, Some(&p_scales), KERNEL_PQ_Q, &mut out_simd);

    assert_eq!(out_j, out_i);
    assert_eq!(out_i, out_planes);
    assert_eq!(out_planes, out_simd);
    assert_eq!(relative_error_pq_ppm(&out_simd, &out_j), 0);

    // Unscaled integer equivalence against dense_matmul_pq_into on pure trit weights
    let p_trit_w: Vec<i32> = trits.iter().map(|&t| t as i32).collect();
    let mut out_dense = vec![0i64; cols];
    let mut out_unscaled = vec![0i64; cols];
    dense_matmul_pq_into(&p_trit_w, rows, cols, &p_x, &mut out_dense);
    ternary_simd_i8_pq_into(&trits_i, rows, cols, &p_x, None, KERNEL_PQ_Q, &mut out_unscaled);
    assert_eq!(out_unscaled, out_dense);
}
