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
        Bfp64HeadRow, Mem8AdaptiveHeadRow, PQ_DEFAULT_Q, PrimeBucketMod, b158_matmul_pq_i_major_into,
        engram_bfp64_int_aligned_matmul_into, engram_bfp64_prequant_matmul_into,
        engram_gather_bfp64_heads_into, engram_hash_ngram_pq_into,
        engram_mem8_adaptive_matmul_into, engram_step_bfp64_into, fold, fold_spectrum,
        gather_qmm_into, gather_qmm_pq_into, maybe_matmul_into, mergeq_matmul_into,
        mlx_dequantize_group64_into, mlx_matmul_pq_group64_into, mlx_quantize_group64_into,
        mlx_quantized_matmul_into, pack_ternary_i_major_into, pack_ternary_into,
        quantize_bfp_i8_group64_into, quantize_bfp_i16_group64_into, quantize_pq_adaptive_into,
        quantize_pq_into, rms_norm_into, rope_into, sampling, sdpa_pq_single_query_into,
        sdpa_single_query_into, ste_linear_backward_into, swiglu_into,
        ternary_bfp_adaptive_mem8_matmul_into, ternary_bfp_i8_matmul_into,
        ternary_bfp_i16_matmul_into, ternary_matmul, ternary_simd_pq_matmul_into,
        ternary_transformer_step_into, unpack_ternary_into,
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
    let mut m_i8 = vec![0i8; ROWS];
    let mut m_i16 = vec![0i16; ROWS];
    let mut exp_x = vec![0i8; ROWS / 64];
    let mut scale_x = vec![0f64; ROWS / 64];
    let mut promoted_mask = vec![false; ROWS / 64];

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
        let q_adapt = quantize_pq_adaptive_into(&x, 22, &mut p_x);
        quantize_pq_into(&mlx_scales, PQ_DEFAULT_Q, &mut p_scales);
        mlx_matmul_pq_group64_into(&p_x, q_adapt, &mlx_codes, &p_scales, PQ_DEFAULT_Q, ROWS, COLS, &mut out);
        b158_matmul_pq_i_major_into(&packed_i, ROWS, COLS, &p_x, q_adapt, 1.0, &mut out);
        ternary_simd_pq_matmul_into(&trits, ROWS, COLS, &p_x, q_adapt, None, 1.0, &mut out);
        quantize_bfp_i8_group64_into(&x, &mut m_i8, &mut exp_x, &mut scale_x);
        ternary_bfp_i8_matmul_into(&trits, ROWS, COLS, &m_i8, &scale_x, 1.0, &mut out);
        quantize_bfp_i16_group64_into(&x, &mut m_i16, &mut exp_x, &mut scale_x);
        ternary_bfp_i16_matmul_into(&trits, ROWS, COLS, &m_i16, &scale_x, 1.0, &mut out);
        let _ = ternary_bfp_adaptive_mem8_matmul_into(
            &trits,
            ROWS,
            COLS,
            &x,
            0.005,
            1.0,
            &mut m_i8,
            &mut m_i16,
            &mut exp_x,
            &mut scale_x,
            &mut promoted_mask,
            &mut out,
        );
        mergeq_matmul_into(&x, &weights, ROWS, COLS, 64, &mut out);
        maybe_matmul_into(&x, &weights, ROWS, COLS, 64, 42, &mut out);
        gather_qmm_into(&x, &mlx_codes, &mlx_scales, 1, ROWS, COLS, &[0], &[1.0], &mut s_norm, &mut out);
        gather_qmm_pq_into(&p_x, q_adapt, &mlx_codes, &p_scales, PQ_DEFAULT_Q, 1, ROWS, COLS, &[0], &[1.0], &mut s_norm, &mut out);
        rms_norm_into(&x, Some(&scales), 1e-5, &mut out);
        rope_into(&x, 1, 10000.0, &mut out);
        swiglu_into(&x, &scales, &mut out);
        sdpa_single_query_into(&x, &k_cache, &v_cache, 2, COLS, &mut s_scores, &mut out);
        sdpa_pq_single_query_into(&p_x, &p_x, &p_x, 1 << 14, 1, COLS, &mut s_scores, &mut out);
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
        engram_hash_ngram_pq_into(&engram_tokens, 2, &engram_mults, 1, &engram_buckets, &mut engram_indices);
        engram_gather_bfp64_heads_into(&bfp_head_table, &engram_indices[..1], &mut m_i8, &mut exp_x);
        engram_bfp64_prequant_matmul_into(&m_i8, &exp_x, &trits, &mlx_scales, ROWS, COLS, &mut out);
        engram_bfp64_int_aligned_matmul_into(&m_i8, &exp_x, &trits, &p_scales, ROWS, COLS, &mut out);
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
            &mut m_i8,
            &mut exp_x,
            &mut s_v,
            &mut s_k,
            &mut s_norm,
            &mut s_q,
            &mut engram_out_hc,
        );
        std::hint::black_box((sampling(&f1), sampling(&f2)));
    }
    let after = ALLOCS.load(Ordering::Relaxed);

    assert_eq!(
        before, after,
        "the hot paths allocated {} times",
        after - before
    );
}
