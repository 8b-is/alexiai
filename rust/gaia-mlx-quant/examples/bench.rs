//! The honest bench — pure-integer `p/q` ("Counting Divided", zero floating point)
//! and packed ternary b1.58 versus dense float64, measured.
//!
//! ```bash
//! cargo run --release --example bench -- --dim 256
//! ```

use std::hint::black_box;
use std::time::Instant;

use gaia_mlx_quant::{
    BITS_PER_WEIGHT, GAIA_PQ_Q, KERNEL_PQ_Q, MLX_GROUP64_COMPRESSION, PQ_DEFAULT_Q,
    dense_matmul, dense_matmul_pq_into, fold, fold_pq, mlx_matmul_pq_group64_i64_into,
    mlx_quantize_group64_pq_into, pack_ternary, pack_ternary_i_major_into,
    pack_ternary_planes_i_into, quantize_ternary_pq_into, relative_error, relative_error_pq_ppm,
    sampling_pq, ternary_matmul, ternary_matmul_i_major, ternary_matmul_planes_i,
    ternary_matmul_pq_i_major_into, ternary_matmul_pq_j_major_into,
    ternary_matmul_pq_planes_i_into, ternary_simd_i8_pq_into,
};

/// Deterministic pure-integer uniform-ish generator in `[-q, +q]` (`q = 2^20`), zero floats.
fn weights_pq(n: usize, seed: u32) -> Vec<i32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x6d2b79f5);
            let mut t = s;
            t = (t ^ (t >> 15)).wrapping_mul(t | 1);
            t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
            let u20 = ((t ^ (t >> 14)) >> 12) as i32; // in [0, 2^20)
            (u20 << 1) - (PQ_DEFAULT_Q as i32)
        })
        .collect()
}

fn main() {
    let dim: usize = std::env::args()
        .nth(2)
        .and_then(|d| d.parse().ok())
        .unwrap_or(256);
    let rows = dim;
    let cols = dim;
    let aligned_cols = cols.max(64).div_ceil(64) * 64;

    let w_p = weights_pq(rows * cols, 0x5eed);
    let x_p = weights_pq(rows, 0xc0ffee);
    let mut trits = vec![0i8; rows * cols];
    let mut scales_p = vec![0i32; rows];
    quantize_ternary_pq_into(&w_p, rows, cols, KERNEL_PQ_Q, &mut trits, &mut scales_p);

    let mut trits_i_major = vec![0i8; rows * cols];
    for j in 0..rows {
        for i in 0..cols {
            trits_i_major[i * rows + j] = trits[j * cols + i];
        }
    }

    let packed_j = pack_ternary(&trits);
    let mut packed_i = vec![0u8; (rows * cols).div_ceil(5)];
    pack_ternary_i_major_into(&trits, rows, cols, &mut packed_i);
    let mut pos = vec![0i8; rows * cols];
    let mut neg = vec![0i8; rows * cols];
    pack_ternary_planes_i_into(&trits, rows, cols, &mut pos, &mut neg);

    // Group-64 pure-integer MLX quantization
    let w_p_al = weights_pq(rows * aligned_cols, 0x5eed);
    let x_p_al = weights_pq(aligned_cols, 0xc0ffee);
    let mut g64_codes = vec![0u32; rows * (aligned_cols / 16)];
    let mut g64_scales_p = vec![0i32; rows * (aligned_cols / 64)];
    mlx_quantize_group64_pq_into(
        &w_p_al,
        rows,
        aligned_cols,
        KERNEL_PQ_Q,
        &mut g64_codes,
        &mut g64_scales_p,
    );

    let inv_q = 1.0 / (PQ_DEFAULT_Q as f64);
    let w: Vec<f64> = w_p.iter().map(|&p| (p as f64) * inv_q).collect();
    let x: Vec<f64> = x_p.iter().map(|&p| (p as f64) * inv_q).collect();
    let scales: Vec<f64> = scales_p.iter().map(|&p| (p as f64) * inv_q).collect();

    let mut dense = vec![0f64; cols];
    let mut ternary = vec![0f64; cols];
    let mut ternary_i = vec![0f64; cols];
    let mut ternary_p = vec![0f64; cols];

    let mut dense_pq = vec![0i64; cols];
    let mut pq_simd = vec![0i64; cols];
    let mut pq_planes = vec![0i64; cols];
    let mut pq_b158_i = vec![0i64; cols];
    let mut pq_b158_j = vec![0i64; cols];
    let mut pq_g64 = vec![0i64; rows];

    // Pre-fuse x_p * scales_p once in O(rows) so pure-integer kernels measure pure matmul throughput
    let x_p_scaled: Vec<i32> = x_p
        .iter()
        .zip(scales_p.iter())
        .map(|(&xv, &sv)| (((xv as i64) * (sv as i64)) / (KERNEL_PQ_Q as i64)) as i32)
        .collect();

    // Warm-up.
    dense_matmul(&w, rows, cols, &x, &mut dense);
    dense_matmul_pq_into(&w_p, rows, cols, &x_p, &mut dense_pq);
    ternary_simd_i8_pq_into(&trits_i_major, rows, cols, &x_p_scaled, None, KERNEL_PQ_Q, &mut pq_simd);
    ternary_matmul_pq_planes_i_into(&pos, &neg, rows, cols, &x_p_scaled, None, KERNEL_PQ_Q, &mut pq_planes);
    ternary_matmul_pq_i_major_into(&packed_i, rows, cols, &x_p_scaled, None, KERNEL_PQ_Q, &mut pq_b158_i);
    ternary_matmul_pq_j_major_into(&packed_j, rows, cols, &x_p_scaled, None, KERNEL_PQ_Q, &mut pq_b158_j);
    mlx_matmul_pq_group64_i64_into(&x_p_al, &g64_codes, &g64_scales_p, rows, aligned_cols, &mut pq_g64);
    ternary_matmul(&packed_j, rows, cols, &x, Some(&scales), 1.0, &mut ternary);
    ternary_matmul_i_major(&packed_i, rows, cols, &x, Some(&scales), 1.0, &mut ternary_i);
    ternary_matmul_planes_i(&pos, &neg, rows, cols, &x, Some(&scales), 1.0, &mut ternary_p);

    let measure = |f: &mut dyn FnMut()| -> u128 {
        let mut iters = 16usize;
        loop {
            let start = Instant::now();
            for _ in 0..iters {
                f();
            }
            let elapsed = start.elapsed();
            if elapsed.as_millis() >= 150 {
                return elapsed.as_nanos() / iters as u128;
            }
            iters *= 2;
        }
    };

    let dense_ns = measure(&mut || {
        dense_matmul(black_box(&w), rows, cols, black_box(&x), &mut dense);
    });
    let pq_simd_ns = measure(&mut || {
        ternary_simd_i8_pq_into(
            black_box(&trits_i_major),
            rows,
            cols,
            black_box(&x_p_scaled),
            None,
            KERNEL_PQ_Q,
            black_box(&mut pq_simd),
        );
    });
    let pq_planes_ns = measure(&mut || {
        ternary_matmul_pq_planes_i_into(
            black_box(&pos),
            black_box(&neg),
            rows,
            cols,
            black_box(&x_p_scaled),
            None,
            KERNEL_PQ_Q,
            black_box(&mut pq_planes),
        );
    });
    let pq_g64_ns = measure(&mut || {
        mlx_matmul_pq_group64_i64_into(
            black_box(&x_p_al),
            black_box(&g64_codes),
            black_box(&g64_scales_p),
            rows,
            aligned_cols,
            black_box(&mut pq_g64),
        );
    });
    let pq_b158_i_ns = measure(&mut || {
        ternary_matmul_pq_i_major_into(
            black_box(&packed_i),
            rows,
            cols,
            black_box(&x_p_scaled),
            None,
            KERNEL_PQ_Q,
            black_box(&mut pq_b158_i),
        );
    });
    let pq_b158_j_ns = measure(&mut || {
        ternary_matmul_pq_j_major_into(
            black_box(&packed_j),
            rows,
            cols,
            black_box(&x_p_scaled),
            None,
            KERNEL_PQ_Q,
            black_box(&mut pq_b158_j),
        );
    });
    let planes_ns = measure(&mut || {
        ternary_matmul_planes_i(
            black_box(&pos),
            black_box(&neg),
            rows,
            cols,
            black_box(&x),
            Some(black_box(&scales)),
            1.0,
            &mut ternary_p,
        );
    });
    let ternary_i_ns = measure(&mut || {
        ternary_matmul_i_major(
            black_box(&packed_i),
            rows,
            cols,
            black_box(&x),
            Some(black_box(&scales)),
            1.0,
            &mut ternary_i,
        );
    });
    let ternary_j_ns = measure(&mut || {
        ternary_matmul(
            black_box(&packed_j),
            rows,
            cols,
            black_box(&x),
            Some(black_box(&scales)),
            1.0,
            &mut ternary,
        );
    });

    let field = fold("bench", None);
    let field_pq = fold_pq("bench", None);
    let samp_pq = sampling_pq(&field_pq);
    let pq_b158_scaled: Vec<i64> = pq_b158_i
        .iter()
        .map(|&v| v * (KERNEL_PQ_Q as i64))
        .collect();
    let err_ppm = relative_error_pq_ppm(&pq_b158_scaled, &dense_pq);
    let speedup_simd = (dense_ns as f64) / (pq_simd_ns.max(1) as f64);

    println!("GAIA-MLX-QUANT v0.3.0 bench — SOTA p/q pure-integer & ternary b1.58 numbers");
    println!(
        "  field      seed {} · dominant {:?} · coherence {:.4}",
        field.seed, field.dominant, field.coherence
    );
    println!(
        "  field-pq   q={} · coherence_p={} · T_p={} · top_k={} (0 floats)",
        GAIA_PQ_Q, field_pq.coherence_p, samp_pq.temperature_p, samp_pq.top_k
    );
    println!("  shape      {rows}x{cols}");
    println!(
        "  packing    {:.4} bits/weight · {} bytes vs {} f64 ({:.2}x) · group64 {:.2}x vs f32",
        BITS_PER_WEIGHT,
        packed_i.len(),
        w.len() * 8,
        (w.len() * 8) as f64 / packed_i.len() as f64,
        MLX_GROUP64_COMPRESSION,
    );
    println!("  dense      {dense_ns:>7} ns/iter  (f64, auto-vectorized reference)");
    println!(
        "  pq-simd-i8 {pq_simd_ns:>7} ns/iter  (i8×i32->i32 4-col NEON/AVX2 SIMD, 0 floats, {speedup_simd:.2}x vs dense)"
    );
    println!("  pq-planes  {pq_planes_ns:>7} ns/iter  (i8 sign-planes 4-col p/q integer, 0 floats)");
    println!("  pq-group64 {pq_g64_ns:>7} ns/iter  (16-trit/u32 4-row MLX_BYTE_TRITS_I32, 0 floats)");
    println!("  pq-b158-i  {pq_b158_i_ns:>7} ns/iter  (5-trits/byte 4-col BYTE_TRITS_I32, 0 floats)");
    println!("  pq-b158-j  {pq_b158_j_ns:>7} ns/iter  (5-trits/byte j-major BYTE_TRITS_I32, 0 floats)");
    println!("  planes     {planes_ns:>7} ns/iter  (i8 sign planes, pre-fused row scales)");
    println!("  packed-i   {ternary_i_ns:>7} ns/iter  (5-trits/byte LUT path, unguarded interior)");
    println!("  packed-j   {ternary_j_ns:>7} ns/iter  (j-major, JS-parity layout)");
    println!(
        "  error      {:.4} relative L2 vs dense ({err_ppm} ppm pure-integer isqrt_u128)",
        relative_error(&ternary_p, &dense)
    );
}
