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
        fold, fold_spectrum, gather_qmm_into, maybe_matmul_into, mergeq_matmul_into,
        mlx_dequantize_group64_into, mlx_quantize_group64_into, mlx_quantized_matmul_into,
        pack_ternary_i_major_into, pack_ternary_into, rms_norm_into, rope_into, sampling,
        sdpa_single_query_into, ste_linear_backward_into, swiglu_into, ternary_matmul,
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
        std::hint::black_box((sampling(&f1), sampling(&f2)));
    }
    let after = ALLOCS.load(Ordering::Relaxed);

    assert_eq!(
        before, after,
        "the hot paths allocated {} times",
        after - before
    );
}
