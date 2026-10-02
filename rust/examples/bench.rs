//! The honest bench — packed ternary versus dense float64, measured.
//!
//! ```bash
//! cargo run --release --example bench -- --dim 256
//! ```

use std::hint::black_box;
use std::time::Instant;

use gaia_mlx_quant::{
    dense_matmul, fold, pack_ternary, pack_ternary_i_major_into, pack_ternary_planes_i_into,
    relative_error, ternary_matmul, ternary_matmul_i_major, ternary_matmul_planes_i,
};

/// A tiny deterministic gaussian-ish generator (sum of uniforms, no deps).
fn weights(n: usize, seed: u32) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x6d2b79f5);
            let mut t = s;
            t = (t ^ (t >> 15)).wrapping_mul(t | 1);
            t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
            let u = ((t ^ (t >> 14)) >> 8) as f64 / 16_777_216.0;
            u + u - 1.0 // ~uniform [-1, 1]
        })
        .collect()
}

fn quantize_per_row(w: &[f64], rows: usize, cols: usize) -> (Vec<i8>, Vec<f64>) {
    let mut trits = vec![0i8; w.len()];
    let mut scales = vec![0f64; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let abs_mean: f64 = row.iter().map(|v| v.abs()).sum::<f64>() / cols as f64;
        let scale = if abs_mean == 0.0 { 1.0 } else { abs_mean };
        scales[r] = scale;
        for (c, &v) in row.iter().enumerate() {
            let q = (v / scale).round();
            trits[r * cols + c] = if q > 0.0 { 1 } else if q < 0.0 { -1 } else { 0 };
        }
    }
    (trits, scales)
}

fn main() {
    let dim = std::env::args()
        .nth(2)
        .and_then(|d| d.parse().ok())
        .unwrap_or(256);
    let rows = dim;
    let cols = dim;

    let w = weights(rows * cols, 0x5eed);
    let x = weights(rows, 0xc0ffee);
    let (trits, scales) = quantize_per_row(&w, rows, cols);
    let packed_j = pack_ternary(&trits);
    let mut packed_i = vec![0u8; (rows * cols).div_ceil(5)];
    pack_ternary_i_major_into(&trits, rows, cols, &mut packed_i);
    let mut pos = vec![0i8; rows * cols];
    let mut neg = vec![0i8; rows * cols];
    pack_ternary_planes_i_into(&trits, rows, cols, &mut pos, &mut neg);

    let mut dense = vec![0f64; cols];
    let mut ternary = vec![0f64; cols];
    let mut ternary_i = vec![0f64; cols];
    let mut ternary_p = vec![0f64; cols];

    // Warm-up.
    dense_matmul(&w, rows, cols, &x, &mut dense);
    ternary_matmul(&packed_j, rows, cols, &x, Some(&scales), 1.0, &mut ternary);
    ternary_matmul_i_major(&packed_i, rows, cols, &x, Some(&scales), 1.0, &mut ternary_i);
    ternary_matmul_planes_i(&pos, &neg, rows, cols, &x, Some(&scales), 1.0, &mut ternary_p);

    // Time-based iteration counts: short fixed counts drown in scheduler
    // noise. Each kernel runs for at least ~400ms; the per-iter figure is
    // the honest average.
    let measure = |f: &mut dyn FnMut()| -> u128 {
        let mut iters = 8usize;
        loop {
            let start = Instant::now();
            for _ in 0..iters {
                f();
            }
            let elapsed = start.elapsed();
            if elapsed.as_millis() >= 400 {
                return elapsed.as_nanos() / iters as u128;
            }
            iters *= 2;
        }
    };

    let dense_ns = measure(&mut || {
        dense_matmul(black_box(&w), rows, cols, black_box(&x), &mut dense);
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

    println!("GAIA-MLX-QUANT bench — honest numbers");
    println!("  field      seed {} · dominant {:?} · coherence {:.4}", field.seed, field.dominant, field.coherence);
    println!("  shape      {rows}x{cols}");
    println!("  packing    {:.4} bits/weight · {} bytes vs {} f64 ({:.2}x)",
        (3f64).log2(),
        packed_i.len(),
        w.len() * 8,
        (w.len() * 8) as f64 / packed_i.len() as f64);
    println!("  dense      {dense_ns} ns/iter  (f64, auto-vectorized)");
    println!("  planes     {planes_ns} ns/iter  (i8 sign planes, 4-accumulator ILP, safe scalar)");
    println!("  packed-i   {ternary_i_ns} ns/iter  (5-trits/byte LUT path)");
    println!("  packed-j   {ternary_j_ns} ns/iter  (j-major, JS-parity layout)");
    println!("  error      {:.4} relative L2 vs dense (random weights — the worst case)",
        relative_error(&ternary_p, &dense));
}
