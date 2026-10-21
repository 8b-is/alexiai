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

#[test]
fn quantum_bitplane_and_omni_bfp_i8_pq_kernels_match_reference() {
    use gaia_mlx_quant::{
        KERNEL_PQ_Q, omni_bfp_i8_simd_pq_into, pack_quantum_bitplanes_i_major_into,
        quantize_quantum_bitplane_i4_pq_into, quantize_quantum_ternary_act_pq_into,
        quantum_bitplane_i4_matmul_pq_into, quantum_ternary_matmul_pq_into,
        ternary_simd_i8_pq_into,
    };

    let rows = 68usize;
    let cols = 11usize;
    let words = rows.div_ceil(64);
    let trits: Vec<i8> = (0..rows * cols).map(|k| [-1, 0, 1][(k * 7 + 1) % 3]).collect();
    let mut trits_i = vec![0i8; rows * cols];
    for j in 0..rows {
        for i in 0..cols {
            trits_i[i * rows + j] = trits[j * cols + i];
        }
    }

    let mut w_pos_u64 = vec![0u64; cols * words];
    let mut w_neg_u64 = vec![0u64; cols * words];
    pack_quantum_bitplanes_i_major_into(&trits, rows, cols, &mut w_pos_u64, &mut w_neg_u64);

    // 1. Quantum 64-trit bitplane POPCNT exact match on ternary activations
    let p_x_trit: Vec<i32> = (0..rows)
        .map(|j| [-KERNEL_PQ_Q, 0, KERNEL_PQ_Q][(j * 5 + 2) % 3])
        .collect();
    let mut x_pos_u64 = vec![0u64; words];
    let mut x_neg_u64 = vec![0u64; words];
    let scale_x = quantize_quantum_ternary_act_pq_into(&p_x_trit, &mut x_pos_u64, &mut x_neg_u64);

    let mut out_q_u64 = vec![0i64; cols];
    quantum_ternary_matmul_pq_into(
        &w_pos_u64,
        &w_neg_u64,
        rows,
        cols,
        &x_pos_u64,
        &x_neg_u64,
        scale_x,
        None,
        KERNEL_PQ_Q,
        &mut out_q_u64,
    );

    let x_trits_only: Vec<i32> = p_x_trit.iter().map(|&v| v.signum()).collect();
    let mut out_trit_ref = vec![0i64; cols];
    ternary_simd_i8_pq_into(&trits_i, rows, cols, &x_trits_only, None, KERNEL_PQ_Q, &mut out_trit_ref);
    for i in 0..cols {
        assert_eq!(out_q_u64[i], out_trit_ref[i] * (scale_x as i64));
    }

    // 2. Quantum 4-bit bitplane POPCNT exact match on integer magnitudes in [-7, +7]
    let mut p_x_i4: Vec<i32> = (0..rows).map(|j| ((j as i32 % 15) - 7) * 700).collect();
    p_x_i4[64] = 7 * 700; // ensure tail group also has max_abs = 7 * 700 so step == 700 exact
    let mut x_sign_u64 = vec![0u64; words];
    let mut x_b0_u64 = vec![0u64; words];
    let mut x_b1_u64 = vec![0u64; words];
    let mut x_b2_u64 = vec![0u64; words];
    let mut block_scales_i4 = vec![0i32; words];
    quantize_quantum_bitplane_i4_pq_into(
        &p_x_i4,
        &mut x_sign_u64,
        &mut x_b0_u64,
        &mut x_b1_u64,
        &mut x_b2_u64,
        &mut block_scales_i4,
    );
    let mut out_q_i4 = vec![0i64; cols];
    quantum_bitplane_i4_matmul_pq_into(
        &w_pos_u64,
        &w_neg_u64,
        rows,
        cols,
        &x_sign_u64,
        &x_b0_u64,
        &x_b1_u64,
        &x_b2_u64,
        &block_scales_i4,
        &mut out_q_i4,
    );
    let mut out_i4_ref = vec![0i64; cols];
    ternary_simd_i8_pq_into(&trits_i, rows, cols, &p_x_i4, None, KERNEL_PQ_Q, &mut out_i4_ref);
    assert_eq!(out_q_i4, out_i4_ref);

    // 3. Omni BFP-i8 16-wide i16 SIMD exact match
    let m_x: Vec<i8> = (0..rows).map(|j| ((j as i32 * 13) % 251 - 125) as i8).collect();
    let m_x_i32: Vec<i32> = m_x.iter().map(|&v| v as i32).collect();
    let mut out_omni = vec![0i64; cols];
    let mut out_omni_ref = vec![0i64; cols];
    omni_bfp_i8_simd_pq_into(&trits_i, rows, cols, &m_x, None, &mut out_omni);
    ternary_simd_i8_pq_into(&trits_i, rows, cols, &m_x_i32, None, KERNEL_PQ_Q, &mut out_omni_ref);
    assert_eq!(out_omni, out_omni_ref);
}

#[test]
fn wupz_seam_quant_tern_and_plan_image_exact() {
    use gaia_mlx_quant::{
        GAIA_PQ_Q, WUPZ_ABI_VERSION, WUPZ_DEFAULT_WINDOW_SIZE, WUPZ_MAGIC, WUPZ_PLAN_VERSION,
        WupzBlockDesc, WupzSectionDesc, fold_pq, fold_to_quant_tern_pq, quant_tern_encode_vad_pq,
        quant_tern_pack_into, quant_tern_unpack_into, wupz_bitplane_diff_u64,
        wupz_bitplane_popcount_u64, wupz_byte_diff, wupz_changed_sections_into,
        wupz_pack_plan_into, wupz_plan_digest, wupz_plan_image_len, wupz_popcount,
        wupz_seal_frame_into, wupz_verify_frame_seal, wupz_xor_fold, wupz_xor_fold_u64,
    };

    // 1. xor_fold parity across every length 0..=260 + seal witness
    let mut buf = [0u8; 260];
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    for b in &mut buf {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (x >> 33) as u8;
    }
    for len in 0..=buf.len() {
        let slice = &buf[..len];
        let full_words_len = len & !7usize;
        let mut scalar_acc = 0u64;
        for (i, &byte) in slice[..full_words_len].iter().enumerate() {
            scalar_acc ^= (byte as u64) << ((i & 7) * 8);
        }
        for &byte in &slice[full_words_len..] {
            scalar_acc ^= byte as u64;
        }
        assert_eq!(wupz_xor_fold(slice), scalar_acc, "mismatch at len={len}");
    }

    let mut sealed = [0u8; 32];
    assert!(wupz_seal_frame_into(&buf[..24], &mut sealed).is_some());
    assert!(wupz_verify_frame_seal(&sealed));
    sealed[10] ^= 0x01;
    assert!(!wupz_verify_frame_seal(&sealed));

    // 2. byte_diff and popcount parity
    let mut buf2 = buf;
    assert_eq!(wupz_byte_diff(&buf, &buf2), 0);
    buf2[0] ^= 1;
    buf2[127] ^= 1;
    buf2[259] ^= 1;
    assert_eq!(wupz_byte_diff(&buf, &buf2), 3);

    let expected_pop: u64 = buf.iter().map(|b| b.count_ones() as u64).sum();
    assert_eq!(wupz_popcount(&buf), expected_pop);

    // 3. Quantum bitplane u64 fold, popcount & diff
    let wp_a = [0x0F0F_0F0F_0F0F_0F0Fu64, 0xAAAA_AAAA_AAAA_AAAAu64];
    let wn_a = [0xF0F0_F0F0_F0F0_F0F0u64, 0x5555_5555_5555_5555u64];
    let mut wp_b = wp_a;
    wp_b[0] ^= 0x03; // flip 2 positive trits
    assert_eq!(wupz_bitplane_popcount_u64(&wp_a, &wn_a), 128);
    assert_eq!(wupz_bitplane_diff_u64(&wp_a, &wn_a, &wp_b, &wn_a), 2);
    assert_eq!(wupz_xor_fold_u64(&wp_a), wp_a[0] ^ wp_a[1]);

    // 4. QuantTern (valence=0.8, arousal=0.1, dominance=0.0 -> [+1, 0, 0] -> byte 0x56 = "qt:56")
    let qt = quant_tern_encode_vad_pq((GAIA_PQ_Q * 8) / 10, GAIA_PQ_Q / 10, 0, GAIA_PQ_Q);
    assert_eq!(qt.trits, [1, 0, 0]);
    assert_eq!(qt.byte, 0x56);
    assert_eq!(qt.as_str(), "qt:56");
    assert_eq!(qt.magnitude, 1);
    assert_eq!(qt.glyphs(), ['+', '0', '0']);

    let mut unpacked_qt = [0i8; 4];
    quant_tern_unpack_into(&[qt.byte], 4, &mut unpacked_qt);
    assert_eq!(unpacked_qt, [1, 0, 0, 0]);

    let mut packed_qt = [0u8; 2];
    assert_eq!(quant_tern_pack_into(&[1, -1, 0, 1, -1], &mut packed_qt), 2);
    let mut roundtrip_qt = [0i8; 5];
    quant_tern_unpack_into(&packed_qt, 5, &mut roundtrip_qt);
    assert_eq!(roundtrip_qt, [1, -1, 0, 1, -1]);

    let fpq = fold_pq("wupz quanttern test", None);
    let f_qt = fold_to_quant_tern_pq(&fpq);
    assert!(f_qt.as_str().starts_with("qt:"));

    // 5. WUPZ v1 plan image wire parity with `wupz-rs/src/pack.rs`
    let sections = [
        WupzSectionDesc {
            id: 0,
            name: b".text",
            addr: 0x1000,
            len: 0x200,
            hash: 0xaa,
        },
        WupzSectionDesc {
            id: 1,
            name: b".data",
            addr: 0x3000,
            len: 0x40,
            hash: 0xbb,
        },
    ];
    let blocks = [
        WupzBlockDesc {
            index: 0,
            covers: &[0],
        },
        WupzBlockDesc {
            index: 1,
            covers: &[0, 1],
        },
    ];
    let expected_len = 4 + 1 + 4 + 4 + (4 + 8 + 8 + 8 + 2 + 5) * 2 + (4 + 4 + 4) + (4 + 4 + 8);
    assert_eq!(wupz_plan_image_len(&sections, &blocks), expected_len);
    let mut plan_buf = [0u8; 256];
    let img = wupz_pack_plan_into(&sections, &blocks, &mut plan_buf).unwrap();
    assert_eq!(&img[..4], &WUPZ_MAGIC);
    assert_eq!(img[4], WUPZ_PLAN_VERSION);
    assert_eq!(WUPZ_ABI_VERSION, 1);

    let d1 = wupz_plan_digest(WUPZ_DEFAULT_WINDOW_SIZE, &sections, &blocks);
    let mut sections_after = sections;
    sections_after[0].hash ^= 0xff;
    let d2 = wupz_plan_digest(WUPZ_DEFAULT_WINDOW_SIZE, &sections_after, &blocks);
    assert_ne!(d1, d2);

    let mut changed = [0u32; 4];
    assert_eq!(wupz_changed_sections_into(&sections, &sections_after, &mut changed), 1);
    assert_eq!(changed[0], 0);
}
