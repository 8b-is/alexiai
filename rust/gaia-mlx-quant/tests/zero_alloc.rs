//! The zero-allocation proof.
//!
//! A counting global allocator wraps the system allocator. The test snapshots
//! the counter, runs the pack + kernel + unpack hot paths on caller-owned
//! buffers, and asserts the counter did not move. This is how "zero
//! allocations" is a verified claim rather than a README sentence.
//!
//! The proof is a single test on purpose: the counter is process-global and
//! the test harness runs tests in parallel threads, so two concurrent tests
//! would see each other's allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded with the same pointer/layout the allocator handed out.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded with the same pointer/layout contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn hot_paths_perform_zero_allocations() {
    use gaia_mlx_quant::{
        Bfp64HeadRow, KERNEL_PQ_Q, Mem8AdaptiveHeadRow, PQ_DEFAULT_Q, PrimeBucketMod,
        WUPZ_DEFAULT_WINDOW_SIZE, WupzBlockDesc, WupzSectionDesc, b158_matmul_pq_i_major_into,
        engram_bfp64_int_aligned_matmul_into, engram_bfp64_prequant_matmul_into,
        engram_gather_bfp64_heads_into, engram_hash_ngram_pq_into,
        engram_mem8_adaptive_matmul_into, engram_step_bfp64_into, engram_step_pq_no_floats_into,
        fold, fold_pq, fold_spectrum,
        fold_spectrum_pq, fold_to_quant_tern_pq, gather_qmm_into, gather_qmm_pq_i64_into,
        maybe_matmul_into, maybe_matmul_pq_into, mergeq_matmul_into, mergeq_matmul_pq_into,
        mlx_dequantize_group64_into, mlx_dequantize_group64_pq_into,
        mlx_matmul_pq_group64_i64_into, mlx_matmul_pq_group64_into, mlx_quantize_group64_into,
        mlx_quantize_group64_pq_into, mlx_quantized_matmul_into, omni_bfp_i8_simd_pq_into,
        pack_quantum_bitplanes_i_major_into, pack_ternary_i_major_into, pack_ternary_into,
        pack_ternary_planes_i_into, quantize_bfp_i8_group64_into, quantize_bfp_i8_pq_group64_into,
        quantize_bfp_i16_group64_into, quantize_pq_into, quantize_quantum_bitplane_i4_pq_into,
        quantize_quantum_ternary_act_pq_into, quantize_ternary_pq_into,
        quantum_bitplane_i4_matmul_pq_into, quantum_ternary_matmul_pq_into, rms_norm_into,
        rms_norm_pq_into, rope_into, rope_pq_into, sampling, sampling_pq,
        sdpa_pq_i32_single_query_into, sdpa_single_query_into, ste_linear_backward_into,
        ste_linear_backward_pq_into, swiglu_into, swiglu_pq_into,
        ternary_bfp_adaptive_mem8_matmul_into, ternary_bfp_i8_matmul_into,
        ternary_bfp_i16_matmul_into, ternary_matmul, ternary_matmul_pq_i_major_into,
        ternary_matmul_pq_j_major_into, ternary_matmul_pq_planes_i_into, ternary_simd_i8_pq_into,
        ternary_simd_pq_matmul_into, ternary_transformer_step_into,
        ternary_transformer_step_pq_into, ultrabackyard_fanout_into, unpack_ternary_into,
        wupz_bitplane_and_nand_u64, wupz_bitplane_diff_u64, wupz_bitplane_popcount_u64,
        wupz_byte_diff, wupz_pack_plan_into, wupz_plan_digest, wupz_popcount, wupz_xor_fold,
        wupz_xor_fold_u64, UltraBackyardLane, kompress_super_ultra_lossless_decode_into,
        kompress_super_ultra_lossless_encode_into,
    };

    const ROWS: usize = 64;
    const COLS: usize = 64;

    // Setup (allocates — that is fine, it is not the hot path).
    let trits: Vec<i8> = (0..ROWS * COLS)
        .map(|i| [-1, 0, 1][(i * 7 + 3) % 3])
        .collect();
    let weights: Vec<f64> = (0..ROWS * COLS)
        .map(|i| ((i * 13 + 5) % 19) as f64 / 9.0 - 1.0)
        .collect();
    let x: Vec<f64> = (0..ROWS).map(|j| (j as f64 + 0.5) / ROWS as f64).collect();
    let scales: Vec<f64> = (0..ROWS).map(|j| 0.5 + (j % 7) as f64 * 0.1).collect();

    let mut packed = vec![0u8; (ROWS * COLS).div_ceil(5)];
    let mut packed_i = vec![0u8; (ROWS * COLS).div_ceil(5)];
    let mut out = vec![0f64; COLS];
    let mut unpacked = vec![0i8; ROWS * COLS];
    let mut mlx_codes = vec![0u32; ROWS * (COLS / 16)];
    let mut mlx_scales = vec![0f64; ROWS * (COLS / 64)];
    let mut p_x = vec![0i32; COLS];
    let mut p_scales = vec![0i32; ROWS * (COLS / 64)];
    let mut dequant_w = vec![0f64; ROWS * COLS];
    let mut grad_x = vec![0f64; COLS];
    let mut grad_w = vec![0f64; ROWS * COLS];
    let mut k_cache = vec![0f64; 4 * COLS];
    let mut v_cache = vec![0f64; 4 * COLS];
    let mut s_norm = vec![0f64; COLS];
    let mut s_q = vec![0f64; COLS];
    let mut s_k = vec![0f64; COLS];
    let mut s_v = vec![0f64; COLS];
    let mut s_attn = vec![0f64; COLS];
    let mut s_scores = vec![0f64; 4];
    let mut s_gate = vec![0f64; COLS];
    let mut s_up = vec![0f64; COLS];
    let mut pos_planes = vec![0i8; ROWS * COLS];
    let mut neg_planes = vec![0i8; ROWS * COLS];
    let mut p_w = vec![0i32; ROWS * COLS];
    let mut p_row_scales = vec![0i32; ROWS];
    let mut p_out_i64 = vec![0i64; COLS];
    let mut p_dequant_w = vec![0i32; ROWS * COLS];
    let mut p_grad_x = vec![0i64; COLS];
    let mut p_grad_w = vec![0i64; ROWS * COLS];
    let mut p_k_cache = vec![0i32; 4 * COLS];
    let mut p_v_cache = vec![0i32; 4 * COLS];
    let mut ps_norm = vec![0i32; COLS];
    let mut ps_q = vec![0i32; COLS];
    let mut ps_k = vec![0i32; COLS];
    let mut ps_v = vec![0i32; COLS];
    let mut ps_attn = vec![0i32; COLS];
    let mut ps_scores = vec![0i64; 4];
    let mut ps_gate = vec![0i32; COLS];
    let mut ps_up = vec![0i32; COLS];
    let mut ps_i64 = vec![0i64; COLS];
    let mut p_tf_out = vec![0i32; COLS];
    let mut w_pos_u64 = vec![0u64; COLS];
    let mut w_neg_u64 = vec![0u64; COLS];
    let mut x_pos_u64 = vec![0u64; 1];
    let mut x_neg_u64 = vec![0u64; 1];
    let mut x_sign_u64 = vec![0u64; 1];
    let mut x_b0_u64 = vec![0u64; 1];
    let mut x_b1_u64 = vec![0u64; 1];
    let mut x_b2_u64 = vec![0u64; 1];
    let mut q_scales_i4 = vec![0i32; 1];
    let mut m_x_i8 = vec![0i8; ROWS];
    let mut s_x_i8 = vec![0f64; 1];
    let mut m_x_i16 = vec![0i16; ROWS];
    let mut s_x_i16 = vec![0f64; 1];
    let mut s_x_pq_i8 = vec![0i32; 1];
    let mut exp_x = vec![0i8; ROWS / 64];

    let engram_buckets = [PrimeBucketMod::new(17, 0)];
    let engram_tokens = [3u64, 7, 11, 13];
    let engram_mults = [131u64, 997u64];
    let mut engram_indices = vec![0usize; 4];
    let bfp_head_table = vec![Bfp64HeadRow::from_f64_slice(&x); 17];
    let mem8_head_table = vec![Mem8AdaptiveHeadRow::from_f64_slice(&x, 0.005); 17];
    let mut engram_out_hc = vec![0f64; COLS];

    // Warm the paths (first-call branches and tables are settled).
    pack_ternary_into(&trits, &mut packed);
    pack_ternary_i_major_into(&trits, ROWS, COLS, &mut packed_i);
    pack_ternary_planes_i_into(&trits, ROWS, COLS, &mut pos_planes, &mut neg_planes);
    quantize_pq_into(&weights, PQ_DEFAULT_Q, &mut p_w);
    quantize_pq_into(&scales, PQ_DEFAULT_Q, &mut p_row_scales);
    ternary_matmul(&packed, ROWS, COLS, &x, Some(&scales), 1.0, &mut out);
    unpack_ternary_into(&packed, ROWS * COLS, &mut unpacked);
    mlx_quantize_group64_into(&weights, ROWS, COLS, &mut mlx_codes, &mut mlx_scales);
    mlx_dequantize_group64_into(&mlx_codes, &mlx_scales, ROWS, COLS, &mut dequant_w);
    mlx_quantized_matmul_into(&x, &mlx_codes, &mlx_scales, ROWS, COLS, &mut out);
    mergeq_matmul_into(&x, &weights, ROWS, COLS, 64, &mut out);
    maybe_matmul_into(&x, &weights, ROWS, COLS, 64, 42, &mut out);
    gather_qmm_into(&x, &mlx_codes, &mlx_scales, 1, ROWS, COLS, &[0], &[1.0], &mut s_norm, &mut out);
    rms_norm_into(&x, Some(&scales), 1e-5, &mut out);
    rope_into(&x, 1, 10000.0, &mut out);
    swiglu_into(&x, &scales, &mut out);
    sdpa_single_query_into(&x, &k_cache, &v_cache, 2, COLS, &mut s_scores, &mut out);
    ste_linear_backward_into(&x, &weights, &scales, ROWS, COLS, &mut grad_x, &mut grad_w);
    let _ = sampling(&fold("warmup", None));
    let _ = sampling(&fold_spectrum("warmup", 2.418, None));

    // The proof itself. `packed` is zeroed between iterations per the packing
    // contract (the accumulate is `+=`); zeroing is a memset, not an
    // allocation.
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..100 {
        packed.fill(0);
        pack_ternary_into(&trits, &mut packed);
        ternary_matmul(&packed, ROWS, COLS, &x, Some(&scales), 1.0, &mut out);
        unpack_ternary_into(&packed, ROWS * COLS, &mut unpacked);
        mlx_quantize_group64_into(&weights, ROWS, COLS, &mut mlx_codes, &mut mlx_scales);
        mlx_dequantize_group64_into(&mlx_codes, &mlx_scales, ROWS, COLS, &mut dequant_w);
        mlx_quantized_matmul_into(&x, &mlx_codes, &mlx_scales, ROWS, COLS, &mut out);
        quantize_pq_into(&x, PQ_DEFAULT_Q, &mut p_x);
        quantize_pq_into(&mlx_scales, PQ_DEFAULT_Q, &mut p_scales);
        mlx_matmul_pq_group64_into(&p_x, PQ_DEFAULT_Q, &mlx_codes, &p_scales, PQ_DEFAULT_Q, ROWS, COLS, &mut out);
        b158_matmul_pq_i_major_into(&packed_i, ROWS, COLS, &p_x, PQ_DEFAULT_Q, 1.0, &mut out);
        ternary_simd_pq_matmul_into(&trits, ROWS, COLS, &p_x, PQ_DEFAULT_Q, None, 1.0, &mut out);
        mergeq_matmul_into(&x, &weights, ROWS, COLS, 64, &mut out);
        maybe_matmul_into(&x, &weights, ROWS, COLS, 64, 42, &mut out);
        gather_qmm_into(&x, &mlx_codes, &mlx_scales, 1, ROWS, COLS, &[0], &[1.0], &mut s_norm, &mut out);
        rms_norm_into(&x, Some(&scales), 1e-5, &mut out);
        rope_into(&x, 1, 10000.0, &mut out);
        swiglu_into(&x, &scales, &mut out);
        sdpa_single_query_into(&x, &k_cache, &v_cache, 2, COLS, &mut s_scores, &mut out);
        ste_linear_backward_into(&x, &weights, &scales, ROWS, COLS, &mut grad_x, &mut grad_w);
        ternary_transformer_step_into(
            &x,
            &scales,
            &packed_i,
            &packed_i,
            &packed_i,
            &packed_i,
            &scales,
            &packed_i,
            &packed_i,
            &packed_i,
            COLS,
            COLS,
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
        let f1 = fold("Analyze Shinjuku 2.4GHz Ch 6 saturation", None);
        let f2 = fold_spectrum("Analyze Shinjuku 2.4GHz Ch 6 saturation", 2.418, None);
        let fpq1 = fold_pq("Analyze Shinjuku 2.4GHz Ch 6 saturation", None);
        let fpq2 = fold_spectrum_pq("Analyze Shinjuku 2.4GHz Ch 6 saturation", 2_535_457, None);
        quantize_ternary_pq_into(&p_w, ROWS, COLS, KERNEL_PQ_Q, &mut unpacked, &mut p_row_scales);
        ternary_matmul_pq_j_major_into(&packed, ROWS, COLS, &p_x, Some(&p_row_scales), KERNEL_PQ_Q, &mut p_out_i64);
        ternary_matmul_pq_i_major_into(&packed_i, ROWS, COLS, &p_x, Some(&p_row_scales), KERNEL_PQ_Q, &mut p_out_i64);
        ternary_matmul_pq_planes_i_into(&pos_planes, &neg_planes, ROWS, COLS, &p_x, Some(&p_row_scales), KERNEL_PQ_Q, &mut p_out_i64);
        ternary_simd_i8_pq_into(&trits, ROWS, COLS, &p_x, Some(&p_row_scales), KERNEL_PQ_Q, &mut p_out_i64);
        mlx_quantize_group64_pq_into(&p_w, ROWS, COLS, KERNEL_PQ_Q, &mut mlx_codes, &mut p_scales);
        mlx_dequantize_group64_pq_into(&mlx_codes, &p_scales, ROWS, COLS, &mut p_dequant_w);
        mlx_matmul_pq_group64_i64_into(&p_x, &mlx_codes, &p_scales, ROWS, COLS, &mut p_out_i64);
        mergeq_matmul_pq_into(&p_x, &p_w, ROWS, COLS, 64, &mut p_out_i64);
        maybe_matmul_pq_into(&p_x, &p_w, ROWS, COLS, 64, 42, &mut p_out_i64);
        gather_qmm_pq_i64_into(&p_x, &mlx_codes, &p_scales, 1, ROWS, COLS, &[0], &[1_048_576], PQ_DEFAULT_Q, &mut ps_i64, &mut p_out_i64);
        rms_norm_pq_into(&p_x, Some(&p_row_scales), PQ_DEFAULT_Q, 10, &mut ps_norm);
        rope_pq_into(&ps_norm, 1, &mut ps_q);
        swiglu_pq_into(&ps_norm, &ps_q, PQ_DEFAULT_Q, &mut ps_gate);
        sdpa_pq_i32_single_query_into(&ps_q, &p_k_cache, &p_v_cache, 2, COLS, PQ_DEFAULT_Q, &mut ps_scores, &mut ps_attn);
        ste_linear_backward_pq_into(&p_x, &p_w, &p_row_scales, ROWS, COLS, PQ_DEFAULT_Q, &mut p_grad_x, &mut p_grad_w);
        pack_quantum_bitplanes_i_major_into(&trits, ROWS, COLS, &mut w_pos_u64, &mut w_neg_u64);
        let q_sx = quantize_quantum_ternary_act_pq_into(&p_x, &mut x_pos_u64, &mut x_neg_u64);
        quantum_ternary_matmul_pq_into(&w_pos_u64, &w_neg_u64, ROWS, COLS, &x_pos_u64, &x_neg_u64, q_sx, None, KERNEL_PQ_Q, &mut p_out_i64);
        quantize_quantum_bitplane_i4_pq_into(&p_x, &mut x_sign_u64, &mut x_b0_u64, &mut x_b1_u64, &mut x_b2_u64, &mut q_scales_i4);
        quantum_bitplane_i4_matmul_pq_into(&w_pos_u64, &w_neg_u64, ROWS, COLS, &x_sign_u64, &x_b0_u64, &x_b1_u64, &x_b2_u64, &q_scales_i4, &mut p_out_i64);
        quantize_bfp_i8_pq_group64_into(&p_x, &mut m_x_i8, &mut s_x_pq_i8);
        omni_bfp_i8_simd_pq_into(&trits, ROWS, COLS, &m_x_i8, None, &mut p_out_i64);
        quantize_bfp_i8_group64_into(&x, &mut m_x_i8, &mut s_x_i8);
        quantize_bfp_i16_group64_into(&x, &mut m_x_i16, &mut s_x_i16);
        ternary_bfp_i8_matmul_into(&trits, ROWS, COLS, &m_x_i8, &s_x_i8, None, 1.0, &mut out);
        ternary_bfp_i16_matmul_into(&trits, ROWS, COLS, &m_x_i16, &s_x_i16, None, 1.0, &mut out);
        ternary_bfp_adaptive_mem8_matmul_into(&trits, ROWS, COLS, &m_x_i8, &s_x_i8, &m_x_i16, &s_x_i16, &scales, 0.7, 1.0, &mut out);
        ternary_transformer_step_pq_into(
            &p_x,
            &p_row_scales,
            &packed_i,
            &packed_i,
            &packed_i,
            &packed_i,
            &p_row_scales,
            &packed_i,
            &packed_i,
            &packed_i,
            COLS,
            COLS,
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
        engram_hash_ngram_pq_into(&engram_tokens, 2, &engram_mults, 1, &engram_buckets, &mut engram_indices);
        engram_gather_bfp64_heads_into(&bfp_head_table, &engram_indices[..1], &mut m_x_i8, &mut exp_x);
        engram_bfp64_prequant_matmul_into(&m_x_i8, &exp_x, &trits, &mlx_scales, ROWS, COLS, &mut out);
        engram_bfp64_int_aligned_matmul_into(&m_x_i8, &exp_x, &trits, &p_scales, ROWS, COLS, &mut out);
        engram_mem8_adaptive_matmul_into(&mem8_head_table, &engram_indices[..1], &trits, &mlx_scales, ROWS, COLS, &mut out);
        engram_step_bfp64_into(
            &bfp_head_table,
            &engram_indices[..1],
            &x,
            &trits,
            &mlx_scales,
            &trits,
            &mlx_scales,
            &scales,
            &scales,
            1,
            ROWS,
            COLS,
            &mut m_x_i8,
            &mut exp_x,
            &mut s_v,
            &mut s_k,
            &mut s_norm,
            &mut s_q,
            &mut engram_out_hc,
        );
        engram_step_pq_no_floats_into(
            &bfp_head_table,
            &engram_indices[..1],
            &p_x,
            &trits,
            &p_scales,
            &trits,
            &p_scales,
            &p_row_scales,
            &p_row_scales,
            1,
            ROWS,
            COLS,
            &mut m_x_i8,
            &mut exp_x,
            &mut ps_v,
            &mut ps_k,
            &mut p_tf_out,
        );
        let mut wupz_plan_buf = [0u8; 128];
        let wupz_secs = [WupzSectionDesc {
            id: 0,
            name: b".rodata.b158",
            addr: 0x1000,
            len: packed.len() as u64,
            hash: wupz_xor_fold(&packed),
        }];
        let wupz_blks = [WupzBlockDesc {
            index: 0,
            covers: &[0],
        }];
        let _ = wupz_pack_plan_into(&wupz_secs, &wupz_blks, &mut wupz_plan_buf);
        let _ = wupz_plan_digest(WUPZ_DEFAULT_WINDOW_SIZE, &wupz_secs, &wupz_blks);
        let _ = wupz_byte_diff(&packed, &packed_i);
        let _ = wupz_popcount(&packed);
        let _ = wupz_xor_fold_u64(&w_pos_u64);
        let _ = wupz_bitplane_popcount_u64(&w_pos_u64, &w_neg_u64);
        let _ = wupz_bitplane_diff_u64(&w_pos_u64, &w_neg_u64, &w_pos_u64, &w_neg_u64);
        let _ = wupz_bitplane_and_nand_u64(w_pos_u64[0], w_neg_u64[0], w_pos_u64[1], w_neg_u64[1]);
        let mut ksul_enc = [0u8; 256];
        let mut ksul_dec = [0u8; 128];
        if let Some(rcpt) = kompress_super_ultra_lossless_encode_into(&packed[..64], &mut ksul_enc) {
            let _ = kompress_super_ultra_lossless_decode_into(&ksul_enc[..rcpt.encoded_bytes], &mut ksul_dec);
        }
        let mut ub_lanes = [UltraBackyardLane {
            lane_id: 0,
            instruction: b"",
            xor_fold: 0,
            and_gate: 0,
            nand_gate: 0,
        }; 4];
        let _ = ultrabackyard_fanout_into(b"lane one; lane two\nlane three", 1, &mut ub_lanes);
        let _ = gaia_mlx_quant::dream_state_loop_into(
            b"clouds dream. lucid wake. mantra hallgat. bodhisattva szeret.",
            1,
            &mut ub_lanes,
        );
        let _ = fold_to_quant_tern_pq(&fpq1);
        std::hint::black_box((sampling(&f1), sampling(&f2), sampling_pq(&fpq1), sampling_pq(&fpq2)));
    }
    let after = ALLOCS.load(Ordering::Relaxed);

    assert_eq!(
        before, after,
        "the hot paths allocated {} times",
        after - before
    );
}
