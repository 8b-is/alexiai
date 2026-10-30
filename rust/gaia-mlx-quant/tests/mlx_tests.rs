use gaia_mlx_quant::{
    MLX_GROUP64_COMPRESSION, MlxQuantizedMatrix, PQ_DEFAULT_Q, Pq, StridedView,
    b158_matmul_pq_i_major_into, gather_qmm_into, maybe_matmul_into, mergeq_matmul_into,
    mlx_code_of, mlx_matmul_pq_group64_into, mlx_quantized_matmul_into, mlx_trit_of,
    pack_ternary_i_major_into, quantize_pq_into, relative_error, rms_norm_into, rope_into,
    sdpa_single_query_into, ste_linear_backward_into, swiglu_into, ternary_matmul_i_major,
    ternary_simd_pq_matmul_into, ternary_transformer_step_into,
};

#[test]
fn mlx_group64_compression_is_12_80x() {
    assert!((MLX_GROUP64_COMPRESSION - 12.80).abs() < 1e-12);
    let w = vec![1.0f64; 64 * 64];
    let q = MlxQuantizedMatrix::quantize(&w, 64, 64).unwrap();
    assert!((q.compression_ratio() - 12.80).abs() < 1e-12);
}

#[test]
fn mlx_code_and_trit_round_trip() {
    assert_eq!(mlx_code_of(-1), 0);
    assert_eq!(mlx_code_of(0), 1);
    assert_eq!(mlx_code_of(1), 2);
    assert_eq!(mlx_trit_of(0), -1);
    assert_eq!(mlx_trit_of(1), 0);
    assert_eq!(mlx_trit_of(2), 1);
    assert_eq!(mlx_trit_of(3), 0);
}

#[test]
fn group64_quantize_dequantize_and_mergeq_agree_exactly() {
    let rows = 16;
    let cols = 128;
    let w: Vec<f64> = (0..rows * cols)
        .map(|i| ((i * 17 + 11) % 23) as f64 / 11.0 - 1.0)
        .collect();
    let x: Vec<f64> = (0..cols).map(|k| ((k * 7 + 3) % 13) as f64 / 6.0 - 1.0).collect();

    let q = MlxQuantizedMatrix::quantize(&w, rows, cols).unwrap();
    let y_q = q.matvec(&x);

    let mut y_mergeq = vec![0.0f64; rows];
    mergeq_matmul_into(&x, &w, rows, cols, 64, &mut y_mergeq);

    let err = relative_error(&y_q, &y_mergeq);
    assert!(err < 1e-12, "group64 matvec vs mergeq relative error = {err}");

    // Also verify dequantized dense matvec matches y_q
    let w_hat = q.dequantize();
    let mut y_dequant = vec![0.0f64; rows];
    for r in 0..rows {
        let mut s = 0.0;
        for c in 0..cols {
            s += w_hat[r * cols + c] * x[c];
        }
        y_dequant[r] = s;
    }
    assert!(relative_error(&y_q, &y_dequant) < 1e-12);
}

#[test]
fn maybe_matmul_is_unbiased_estimator_of_dense_matmul() {
    let rows = 8;
    let cols = 64;
    let w: Vec<f64> = (0..rows * cols)
        .map(|i| (((i * 31 + 7) % 101) as f64 / 100.0 - 0.25) * 0.8)
        .collect();
    let x: Vec<f64> = (0..cols).map(|k| (k as f64 + 1.0) / cols as f64).collect();

    let mut y_dense = vec![0.0f64; rows];
    for r in 0..rows {
        for c in 0..cols {
            y_dense[r] += w[r * cols + c] * x[c];
        }
    }

    let trials = 1024;
    let mut y_avg = vec![0.0f64; rows];
    let mut y_tmp = vec![0.0f64; rows];
    for t in 0..trials {
        maybe_matmul_into(&x, &w, rows, cols, 64, 1000 + (t as u64) * 7919, &mut y_tmp);
        for r in 0..rows {
            y_avg[r] += y_tmp[r] / (trials as f64);
        }
    }

    let err = relative_error(&y_avg, &y_dense);
    assert!(err < 0.03, "stochastic maybe_matmul mean error too large: {err}");
}

#[test]
fn gather_qmm_routes_selected_experts_accurately() {
    let num_experts = 4;
    let n_out = 8;
    let k_in = 64;
    let mut all_codes = Vec::new();
    let mut all_scales = Vec::new();
    let mut experts = Vec::new();

    for e in 0..num_experts {
        let w: Vec<f64> = (0..n_out * k_in)
            .map(|i| ((i * (e + 3) + 5) % 19) as f64 / 9.0 - 1.0)
            .collect();
        let q = MlxQuantizedMatrix::quantize(&w, n_out, k_in).unwrap();
        all_codes.extend_from_slice(&q.codes);
        all_scales.extend_from_slice(&q.scales);
        experts.push(q);
    }

    let x: Vec<f64> = (0..k_in).map(|k| (k as f64) * 0.05 - 1.0).collect();
    let active = [1usize, 3usize];
    let gates = [0.65f64, 0.35f64];
    let mut scratch = vec![0.0f64; n_out];
    let mut y_moe = vec![0.0f64; n_out];

    gather_qmm_into(
        &x,
        &all_codes,
        &all_scales,
        num_experts,
        n_out,
        k_in,
        &active,
        &gates,
        &mut scratch,
        &mut y_moe,
    );

    let y1 = experts[1].matvec(&x);
    let y3 = experts[3].matvec(&x);
    let y_ref: Vec<f64> = (0..n_out).map(|i| 0.65 * y1[i] + 0.35 * y3[i]).collect();

    assert!(relative_error(&y_moe, &y_ref) < 1e-12);
}

#[test]
fn strided_view_zero_copy_transpose_and_slice() {
    let data = [1, 2, 3, 4, 5, 6];
    let v = StridedView::row_major(&data, 2, 3);
    assert_eq!(v.get(0, 2), 3);
    assert_eq!(v.get(1, 0), 4);

    let vt = v.transpose();
    assert_eq!((vt.rows, vt.cols), (3, 2));
    assert_eq!(vt.get(2, 0), 3);
    assert_eq!(vt.get(0, 1), 4);

    let sub = v.slice(0, 2, 1, 2);
    assert_eq!(sub.get(0, 0), 2);
    assert_eq!(sub.get(1, 1), 6);
}

#[test]
fn rope_preserves_vector_norm_and_transformer_step_runs_cleanly() {
    let dim = 16;
    let hidden_dim = 32;
    let x: Vec<f64> = (0..dim).map(|i| (i as f64 + 1.0) * 0.1).collect();
    let mut x_rot = vec![0.0f64; dim];
    rope_into(&x, 5, 10000.0, &mut x_rot);

    let norm_before: f64 = x.iter().map(|v| v * v).sum();
    let norm_after: f64 = x_rot.iter().map(|v| v * v).sum();
    assert!(
        (norm_before - norm_after).abs() / norm_before < 1e-3,
        "RoPE should preserve Euclidean norm: {norm_before} vs {norm_after}"
    );

    // Test full Ternary Transformer step
    let trits_dd: Vec<i8> = (0..dim * dim).map(|i| [-1, 0, 1][(i * 5 + 1) % 3]).collect();
    let trits_dh: Vec<i8> = (0..dim * hidden_dim).map(|i| [-1, 0, 1][(i * 7 + 2) % 3]).collect();
    let mut w_dd = vec![0u8; (dim * dim).div_ceil(5)];
    let mut w_dh = vec![0u8; (dim * hidden_dim).div_ceil(5)];
    let mut w_hd = vec![0u8; (hidden_dim * dim).div_ceil(5)];
    pack_ternary_i_major_into(&trits_dd, dim, dim, &mut w_dd);
    pack_ternary_i_major_into(&trits_dh, dim, hidden_dim, &mut w_dh);
    pack_ternary_i_major_into(&trits_dh, hidden_dim, dim, &mut w_hd);

    let ones_d = vec![1.0f64; dim];
    let mut k_cache = vec![0.0f64; 4 * dim];
    let mut v_cache = vec![0.0f64; 4 * dim];
    let mut s_norm = vec![0.0f64; dim];
    let mut s_q = vec![0.0f64; dim];
    let mut s_k = vec![0.0f64; dim];
    let mut s_v = vec![0.0f64; dim];
    let mut s_attn = vec![0.0f64; dim];
    let mut s_scores = vec![0.0f64; 4];
    let mut s_gate = vec![0.0f64; hidden_dim];
    let mut s_up = vec![0.0f64; hidden_dim];
    let mut out = vec![0.0f64; dim];

    ternary_transformer_step_into(
        &x,
        &ones_d,
        &w_dd,
        &w_dd,
        &w_dd,
        &w_dd,
        &ones_d,
        &w_dh,
        &w_dh,
        &w_hd,
        dim,
        hidden_dim,
        0,
        &mut k_cache,
        &mut v_cache,
        &mut s_norm,
        &mut s_q,
        &mut s_k,
        &mut s_v,
        &mut s_attn,
        &mut s_scores,
        &mut s_gate,
        &mut s_up,
        &mut out,
    );
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(out.iter().any(|&v| v != 0.0));
}

#[test]
fn ste_linear_backward_produces_valid_gradients() {
    let n_out = 4;
    let k_in = 8;
    let x = [0.5, -0.25, 1.0, 0.75, -0.5, 0.2, 0.9, -0.8];
    let w: Vec<f64> = (0..n_out * k_in).map(|i| ((i as f64) - 15.0) * 0.05).collect();
    let grad_y = [1.0, -0.5, 0.25, 0.75];
    let mut grad_x = [0.0f64; 8];
    let mut grad_w = [0.0f64; 32];

    ste_linear_backward_into(&x, &w, &grad_y, n_out, k_in, &mut grad_x, &mut grad_w);
    assert!(grad_x.iter().all(|v| v.is_finite()));
    assert!(grad_w.iter().all(|v| v.is_finite()));
    assert!(grad_w.iter().any(|&v| v != 0.0));

    // Also exercise rms_norm, swiglu, sdpa, and mlx_quantized_matmul directly
    let mut norm_out = [0.0f64; 8];
    rms_norm_into(&x, None, 1e-5, &mut norm_out);
    let mut swi_out = [0.0f64; 8];
    swiglu_into(&x, &norm_out, &mut swi_out);
    let mut scores = [0.0f64; 2];
    let mut attn_out = [0.0f64; 4];
    sdpa_single_query_into(&x[..4], &w[..8], &w[8..16], 2, 4, &mut scores, &mut attn_out);
    let mut codes = [0x5555_5555u32; 4];
    codes[0] = 0xaaaa_0000;
    let scales = [1.0f64];
    let x64 = [0.5f64; 64];
    let mut y1 = [0.0f64; 1];
    mlx_quantized_matmul_into(&x64, &codes, &scales, 1, 64, &mut y1);
    assert!(y1[0].is_finite());
}

#[test]
fn pq_rational_and_integer_numerator_kernels_match_float_to_1e6() {
    // 1. Exact rational Pq arithmetic
    let a = Pq::reduced(6, 8).unwrap();
    assert_eq!((a.p, a.q), (3, 4));
    let b = Pq::reduced(1, 4).unwrap();
    assert_eq!(a.add(b), Pq { p: 1, q: 1 });
    assert_eq!(a.mul(b), Pq { p: 3, q: 16 });
    assert!(Pq::new(1, 0).is_none());

    // 2. MLX Group-64 p/q vs float Group-64
    let rows = 32;
    let cols = 128;
    let w: Vec<f64> = (0..rows * cols)
        .map(|i| ((i * 19 + 7) % 29) as f64 / 14.0 - 1.0)
        .collect();
    let x: Vec<f64> = (0..cols).map(|k| ((k * 11 + 3) % 17) as f64 / 8.0 - 1.0).collect();

    let q = MlxQuantizedMatrix::quantize(&w, rows, cols).unwrap();
    let y_float = q.matvec(&x);

    let mut p_x = vec![0i32; cols];
    let mut p_scales = vec![0i32; q.scales.len()];
    quantize_pq_into(&x, PQ_DEFAULT_Q, &mut p_x);
    quantize_pq_into(&q.scales, PQ_DEFAULT_Q, &mut p_scales);

    let mut y_pq = vec![0.0f64; rows];
    mlx_matmul_pq_group64_into(
        &p_x,
        PQ_DEFAULT_Q,
        &q.codes,
        &p_scales,
        PQ_DEFAULT_Q,
        rows,
        cols,
        &mut y_pq,
    );
    let err_g64 = relative_error(&y_pq, &y_float);
    assert!(err_g64 < 1e-5, "p/q Group-64 vs float error: {err_g64}");

    // 3. GAIA b1.58 i-major p/q & Wide-SIMD p/q vs float b1.58 i-major
    let trits_j: Vec<i8> = (0..rows * cols).map(|i| [-1, 0, 1][(i * 7 + 2) % 3]).collect();
    let mut trits_i = vec![0i8; rows * cols];
    for j in 0..rows {
        for i in 0..cols {
            trits_i[i * rows + j] = trits_j[j * cols + i];
        }
    }
    let mut packed_i = vec![0u8; (rows * cols).div_ceil(5)];
    pack_ternary_i_major_into(&trits_j, rows, cols, &mut packed_i);

    let x_row: Vec<f64> = (0..rows).map(|j| ((j * 5 + 1) % 13) as f64 / 6.0 - 1.0).collect();
    let mut p_x_row = vec![0i32; rows];
    quantize_pq_into(&x_row, PQ_DEFAULT_Q, &mut p_x_row);

    let mut y_b158_float = vec![0.0f64; cols];
    let mut y_b158_pq = vec![0.0f64; cols];
    let mut y_simd_pq = vec![0.0f64; cols];

    ternary_matmul_i_major(&packed_i, rows, cols, &x_row, None, 1.0, &mut y_b158_float);
    b158_matmul_pq_i_major_into(&packed_i, rows, cols, &p_x_row, PQ_DEFAULT_Q, 1.0, &mut y_b158_pq);
    ternary_simd_pq_matmul_into(&trits_i, rows, cols, &p_x_row, PQ_DEFAULT_Q, None, 1.0, &mut y_simd_pq);

    assert!(relative_error(&y_b158_pq, &y_b158_float) < 1e-6);
    assert!(relative_error(&y_simd_pq, &y_b158_float) < 1e-6);
}

#[test]
fn pure_integer_pq_mlx_and_transformer_primitives_operate_with_zero_floats() {
    use gaia_mlx_quant::{
        MLX_GROUP_SIZE, exp_pq, gather_qmm_pq_i64_into, gather_qmm_pq_into, maybe_matmul_pq_into,
        mergeq_matmul_pq_into, mlx_dequantize_group64_pq_into, mlx_matmul_pq_group64_i64_into,
        mlx_quantize_group64_pq_into, quantize_bfp_i8_group64_into,
        quantize_bfp_i8_pq_group64_into, quantize_bfp_i16_group64_into, quantize_pq_adaptive_into,
        relative_error_pq_ppm, rms_norm_pq_into, rope_pq_into, sdpa_pq_i32_single_query_into,
        sdpa_pq_single_query_into, sin_cos_pq, ste_linear_backward_pq_into, swiglu_pq_into,
        ternary_bfp_adaptive_mem8_matmul_into, ternary_bfp_i8_matmul_into,
        ternary_bfp_i16_matmul_into, ternary_transformer_step_pq_into,
    };

    let rows = 16usize;
    let cols = 64usize;
    let q_i32 = PQ_DEFAULT_Q as i32;
    let w_f64: Vec<f64> = (0..rows * cols)
        .map(|i| ((i * 17 + 11) % 23) as f64 / 11.0 - 1.0)
        .collect();
    let x_f64: Vec<f64> = (0..cols).map(|k| ((k * 7 + 3) % 13) as f64 / 6.0 - 1.0).collect();

    let mut p_w = vec![0i32; rows * cols];
    let mut p_x = vec![0i32; cols];
    let q_w = quantize_pq_adaptive_into(&w_f64, 22, &mut p_w);
    assert!(q_w >= PQ_DEFAULT_Q);
    quantize_pq_into(&w_f64, PQ_DEFAULT_Q, &mut p_w);
    quantize_pq_into(&x_f64, PQ_DEFAULT_Q, &mut p_x);

    let mut codes = vec![0u32; rows * (cols / 16)];
    let mut p_scales = vec![0i32; rows * (cols / 64)];
    mlx_quantize_group64_pq_into(&p_w, rows, cols, q_i32, &mut codes, &mut p_scales);

    let mut p_w_dequant = vec![0i32; rows * cols];
    mlx_dequantize_group64_pq_into(&codes, &p_scales, rows, cols, &mut p_w_dequant);
    assert!(p_w_dequant.iter().any(|&v| v != 0));

    let mut p_y_g64 = vec![0i64; rows];
    let mut p_y_mergeq = vec![0i64; rows];
    mlx_matmul_pq_group64_i64_into(&p_x, &codes, &p_scales, rows, cols, &mut p_y_g64);
    mergeq_matmul_pq_into(&p_x, &p_w, rows, cols, 64, &mut p_y_mergeq);
    assert_eq!(p_y_g64, p_y_mergeq);
    assert_eq!(relative_error_pq_ppm(&p_y_g64, &p_y_mergeq), 0);

    let mut p_y_maybe = vec![0i64; rows];
    maybe_matmul_pq_into(&p_x, &p_w, rows, cols, 64, 0x432_528, &mut p_y_maybe);
    assert!(p_y_maybe.iter().any(|&v| v != 0));

    let mut moe_scratch = vec![0i64; rows];
    let mut p_y_moe = vec![0i64; rows];
    gather_qmm_pq_i64_into(
        &p_x,
        &codes,
        &p_scales,
        1,
        rows,
        cols,
        &[0],
        &[q_i32],
        PQ_DEFAULT_Q,
        &mut moe_scratch,
        &mut p_y_moe,
    );
    assert_eq!(p_y_moe, p_y_g64);

    let mut f64_moe_scratch = vec![0.0f64; rows];
    let mut f64_y_moe = vec![0.0f64; rows];
    gather_qmm_pq_into(
        &p_x,
        PQ_DEFAULT_Q,
        &codes,
        &p_scales,
        PQ_DEFAULT_Q,
        1,
        rows,
        cols,
        &[0],
        &[1.0],
        &mut f64_moe_scratch,
        &mut f64_y_moe,
    );
    assert!(f64_y_moe.iter().any(|&v| v != 0.0));

    // Pure-integer sin/cos, exp, RMSNorm, RoPE, SwiGLU, SDPA, STE backward, and Transformer step
    let (s0, c0) = sin_cos_pq(0);
    assert_eq!(s0, 0);
    assert!((c0 - PQ_DEFAULT_Q).abs() <= 2);
    assert_eq!(exp_pq(0), PQ_DEFAULT_Q);
    assert!(exp_pq(-PQ_DEFAULT_Q) < PQ_DEFAULT_Q && exp_pq(-PQ_DEFAULT_Q) > 0);

    let dim = 16usize;
    let hidden_dim = 32usize;
    let mut p_norm = vec![0i32; dim];
    rms_norm_pq_into(&p_x[..dim], None, PQ_DEFAULT_Q, 10, &mut p_norm);
    let mut p_rot = vec![0i32; dim];
    rope_pq_into(&p_norm, 3, &mut p_rot);
    let mut p_swi = vec![0i32; dim];
    swiglu_pq_into(&p_norm, &p_rot, PQ_DEFAULT_Q, &mut p_swi);

    let mut p_scores = vec![0i64; 2];
    let mut p_attn = vec![0i32; dim];
    sdpa_pq_i32_single_query_into(
        &p_rot,
        &p_w[..2 * dim],
        &p_w[2 * dim..4 * dim],
        2,
        dim,
        PQ_DEFAULT_Q,
        &mut p_scores,
        &mut p_attn,
    );
    assert!(p_scores.iter().all(|&s| s > 0));

    let mut f64_scores = vec![0.0f64; 2];
    let mut f64_attn = vec![0.0f64; dim];
    sdpa_pq_single_query_into(
        &p_rot,
        &p_w[..2 * dim],
        &p_w[2 * dim..4 * dim],
        PQ_DEFAULT_Q,
        2,
        dim,
        &mut f64_scores,
        &mut f64_attn,
    );
    assert!(f64_scores.iter().all(|&s| s > 0.0));

    // Verify BFP-64 (i8, i16, adaptive MEM|8, and pure-integer pq i8)
    let num_groups = cols.div_ceil(MLX_GROUP_SIZE);
    let mut m_i8 = vec![0i8; cols];
    let mut s_i8 = vec![0.0f64; num_groups];
    let mut m_i16 = vec![0i16; cols];
    let mut s_i16 = vec![0.0f64; num_groups];
    let mut m_pq_i8 = vec![0i8; cols];
    let mut s_pq_i8 = vec![0i32; num_groups];
    quantize_bfp_i8_group64_into(&x_f64, &mut m_i8, &mut s_i8);
    quantize_bfp_i16_group64_into(&x_f64, &mut m_i16, &mut s_i16);
    quantize_bfp_i8_pq_group64_into(&p_x, &mut m_pq_i8, &mut s_pq_i8);
    assert!(s_pq_i8[0] > 0);

    let trits_bfp: Vec<i8> = (0..cols * rows).map(|i| [-1, 0, 1][(i * 3 + 1) % 3]).collect();
    let mut y_bfp8 = vec![0.0f64; rows];
    let mut y_bfp16 = vec![0.0f64; rows];
    let mut y_bfp_mem8 = vec![0.0f64; rows];
    let salience: Vec<f64> = (0..rows).map(|j| if j % 2 == 0 { 0.9 } else { 0.1 }).collect();
    ternary_bfp_i8_matmul_into(&trits_bfp, cols, rows, &m_i8, &s_i8, None, 1.0, &mut y_bfp8);
    ternary_bfp_i16_matmul_into(&trits_bfp, cols, rows, &m_i16, &s_i16, None, 1.0, &mut y_bfp16);
    ternary_bfp_adaptive_mem8_matmul_into(
        &trits_bfp,
        cols,
        rows,
        &m_i8,
        &s_i8,
        &m_i16,
        &s_i16,
        &salience,
        0.5,
        1.0,
        &mut y_bfp_mem8,
    );
    assert!(relative_error(&y_bfp8, &y_bfp16) < 0.02);
    assert!(relative_error(&y_bfp_mem8, &y_bfp16) < 0.02);

    let mut p_grad_x = vec![0i64; cols];
    let mut p_grad_w = vec![0i64; rows * cols];
    ste_linear_backward_pq_into(
        &p_x,
        &p_w,
        &p_x[..rows],
        rows,
        cols,
        PQ_DEFAULT_Q,
        &mut p_grad_x,
        &mut p_grad_w,
    );
    assert!(p_grad_w.iter().any(|&v| v != 0));

    let trits_dd: Vec<i8> = (0..dim * dim).map(|i| [-1, 0, 1][(i * 5 + 1) % 3]).collect();
    let trits_dh: Vec<i8> = (0..dim * hidden_dim).map(|i| [-1, 0, 1][(i * 7 + 2) % 3]).collect();
    let mut w_dd = vec![0u8; (dim * dim).div_ceil(5)];
    let mut w_dh = vec![0u8; (dim * hidden_dim).div_ceil(5)];
    let mut w_hd = vec![0u8; (hidden_dim * dim).div_ceil(5)];
    pack_ternary_i_major_into(&trits_dd, dim, dim, &mut w_dd);
    pack_ternary_i_major_into(&trits_dh, dim, hidden_dim, &mut w_dh);
    pack_ternary_i_major_into(&trits_dh, hidden_dim, dim, &mut w_hd);

    let p_ones = vec![q_i32; dim];
    let mut p_k_cache = vec![0i32; 4 * dim];
    let mut p_v_cache = vec![0i32; 4 * dim];
    let mut ps_norm = vec![0i32; dim];
    let mut ps_q = vec![0i32; dim];
    let mut ps_k = vec![0i32; dim];
    let mut ps_v = vec![0i32; dim];
    let mut ps_attn = vec![0i32; dim];
    let mut ps_scores = vec![0i64; 4];
    let mut ps_gate = vec![0i32; hidden_dim];
    let mut ps_up = vec![0i32; hidden_dim];
    let mut ps_i64 = vec![0i64; hidden_dim];
    let mut p_tf_out = vec![0i32; dim];

    ternary_transformer_step_pq_into(
        &p_x[..dim],
        &p_ones,
        &w_dd,
        &w_dd,
        &w_dd,
        &w_dd,
        &p_ones,
        &w_dh,
        &w_dh,
        &w_hd,
        dim,
        hidden_dim,
        0,
        &mut p_k_cache,
        &mut p_v_cache,
        &mut ps_norm,
        &mut ps_q,
        &mut ps_k,
        &mut ps_v,
        &mut ps_attn,
        &mut ps_scores,
        &mut ps_gate,
        &mut ps_up,
        &mut ps_i64,
        &mut p_tf_out,
    );
    assert!(p_tf_out.iter().any(|&v| v != 0));
}

#[test]
fn rational_wave_cell_16b_and_wave_memory_cell_24b_dds_shader_cascade() {
    use gaia_mlx_quant::{
        DDS_LUT_SIZE, DDS_SINE_LUT_I16, RationalWaveCell, WaveMemoryCell, rational_wave_dds_query,
    };

    assert_eq!(core::mem::size_of::<WaveMemoryCell>(), 24);
    assert_eq!(core::mem::align_of::<WaveMemoryCell>(), 8);
    assert_eq!(core::mem::size_of::<RationalWaveCell>(), 16);
    assert_eq!(core::mem::align_of::<RationalWaveCell>(), 16);
    assert_eq!(DDS_LUT_SIZE, 4096);

    // Verify 24-byte WaveMemoryCell construction
    let wmc = WaveMemoryCell::new(432.0, 1.0, 0.5, [200, 50, 20], 0x02, 1);
    assert_eq!(wmc.shape, 0x02);
    assert_eq!(wmc.quant, 1);
    assert_eq!(wmc.color, [200, 50, 20]);

    // Verify compile-time DDS sine LUT endpoints & quarter-wave symmetry
    assert_eq!(DDS_SINE_LUT_I16[0], 0);
    assert_eq!(DDS_SINE_LUT_I16[1024], 32767);
    assert_eq!(DDS_SINE_LUT_I16[2048], 0);
    assert_eq!(DDS_SINE_LUT_I16[3072], -32767);
    for k in 1..1024 {
        assert_eq!(DDS_SINE_LUT_I16[k], DDS_SINE_LUT_I16[2048 - k]);
        assert_eq!(DDS_SINE_LUT_I16[2048 + k], -DDS_SINE_LUT_I16[k]);
        assert_eq!(DDS_SINE_LUT_I16[4096 - k], -DDS_SINE_LUT_I16[k]);
    }

    // Verify RGB565 pack / unpack round-trip on 5-6-5 grid
    for r5 in [0u8, 7, 16, 25, 31] {
        for g6 in [0u8, 13, 32, 48, 63] {
            for b5 in [0u8, 5, 15, 24, 31] {
                let r = (r5 << 3) | (r5 >> 2);
                let g = (g6 << 2) | (g6 >> 4);
                let b = (b5 << 3) | (b5 >> 2);
                let packed = RationalWaveCell::pack_rgb565(r, g, b);
                let (ur, ug, ub) = RationalWaveCell::unpack_rgb565(packed);
                assert_eq!((ur, ug, ub), (r, g, b));
            }
        }
    }

    // Verify shape, shape_id, magnitude_octave, and quant helpers
    for &q in &[-1i8, 0, 1] {
        let sq = RationalWaveCell::pack_shape_quant(0x0042, q);
        let cell = RationalWaveCell {
            freq_p: 432,
            freq_q: 1,
            amp_p: 1,
            amp_q: 1,
            phase_p: 0,
            phase_q: 1,
            color: 0,
            shape_and_quant: sq,
        };
        assert_eq!(cell.quant(), q);
        assert_eq!(cell.shape(), 0x0042);
        assert_eq!(cell.shape_id(), 0x42);
        assert_eq!(cell.magnitude_octave(), 0);
    }

    // Reserved 0b11 decodes to 0
    let cell_res = RationalWaveCell {
        freq_p: 1,
        freq_q: 1,
        amp_p: 1,
        amp_q: 1,
        phase_p: 0,
        phase_q: 1,
        color: 0,
        shape_and_quant: 0b11,
    };
    assert_eq!(cell_res.quant(), 0);

    // Verify signed 6-bit magnitude_octave in [-32, 31]
    for oct in [-32i8, -7, -1, 0, 1, 5, 31] {
        let sq = RationalWaveCell::pack_shape_mag_quant(0x2A, oct, -1);
        let cell = RationalWaveCell {
            freq_p: 528,
            freq_q: 1,
            amp_p: 100,
            amp_q: 1,
            phase_p: 1,
            phase_q: 4,
            color: RationalWaveCell::pack_rgb565(200, 50, 20),
            shape_and_quant: sq,
        };
        assert_eq!(cell.shape_id(), 0x2A);
        assert_eq!(cell.magnitude_octave(), oct);
        assert_eq!(cell.quant(), -1);
    }

    // Verify 3-Stage DDS Shader Cascade:
    // - Cell 0: quant=0 -> culled at Stage 1
    // - Cell 1: quant=+1, shape=0x05 != target(0x02) -> survives Stage 1, culled at Stage 2 (shape)
    // - Cell 2: quant=+1, shape=0x02, distant color -> survives Stage 1, culled at Stage 2 (color)
    // - Cell 3: quant=+1, shape=0x02, matching color, phase=1/4 (peak +32767), amp=2/1, mag_octave=+1 -> +131068
    // - Cell 4: quant=-1, shape=0x02, matching color, phase=1/4 (peak +32767), amp=1/1, mag_octave=0 -> -32767
    let target_color = RationalWaveCell::pack_rgb565(200, 50, 20);
    let cells = [
        RationalWaveCell::new(432, 1, 10, 1, 1, 4, target_color, 0x02, 0),
        RationalWaveCell::new(432, 1, 10, 1, 1, 4, target_color, 0x05, 1),
        RationalWaveCell::new(
            432,
            1,
            10,
            1,
            1,
            4,
            RationalWaveCell::pack_rgb565(0, 255, 255),
            0x02,
            1,
        ),
        RationalWaveCell {
            freq_p: 0,
            freq_q: 1,
            amp_p: 2,
            amp_q: 1,
            phase_p: 1,
            phase_q: 4,
            color: target_color,
            shape_and_quant: RationalWaveCell::pack_shape_mag_quant(0x02, 1, 1),
        },
        RationalWaveCell {
            freq_p: 0,
            freq_q: 1,
            amp_p: 1,
            amp_q: 1,
            phase_p: 1,
            phase_q: 4,
            color: target_color,
            shape_and_quant: RationalWaveCell::pack_shape_mag_quant(0x02, 0, -1),
        },
    ];

    let res = rational_wave_dds_query(&cells, 10, 0x02, 0xFF, target_color, 24);
    assert_eq!(res.total_cells, 5);
    assert_eq!(res.stage1_survivors, 4);
    assert_eq!(res.stage2_survivors, 2);
    assert_eq!(res.total_energy_i64, (32767 * 2 * 2) - 32767);
    assert_eq!(res.blended_rgb565, target_color);

    // Determinism check: identical query yields identical state_digest
    let res2 = rational_wave_dds_query(&cells, 10, 0x02, 0xFF, target_color, 24);
    assert_eq!(res, res2);
}
