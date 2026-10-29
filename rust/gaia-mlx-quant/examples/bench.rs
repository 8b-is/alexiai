//! The honest bench — pure-integer `p/q` ("Counting Divided", zero floating point)
//! and packed ternary b1.58 versus dense float64, measured.
//!
//! ```bash
//! cargo run --release --example bench -- --dim 256
//! ```

use std::hint::black_box;
use std::time::Instant;

use gaia_mlx_quant::{
    BITS_PER_WEIGHT, GAIA_PQ_Q, KERNEL_PQ_Q, MLX_GROUP_SIZE, MLX_GROUP64_COMPRESSION, PQ_DEFAULT_Q,
    WUPZ_ABI_VERSION, dense_matmul, dense_matmul_pq_into, fold, fold_pq, fold_to_quant_tern_pq,
    mlx_matmul_pq_group64_i64_into, mlx_quantize_group64_pq_into, omni_bfp_i8_simd_pq_into,
    pack_quantum_bitplanes_i_major_into, pack_ternary, pack_ternary_i_major_into,
    pack_ternary_planes_i_into, quantize_bfp_i8_group64_into, quantize_bfp_i8_pq_group64_into,
    quantize_quantum_bitplane_i4_pq_into, quantize_quantum_ternary_act_pq_into,
    quantize_ternary_pq_into, quantum_bitplane_i4_matmul_pq_into, quantum_ternary_matmul_pq_into,
    relative_error, relative_error_pq_ppm, sampling_pq, ternary_bfp_i8_matmul_into, ternary_matmul,
    ternary_matmul_i_major, ternary_matmul_planes_i, ternary_matmul_pq_i_major_into,
    ternary_matmul_pq_j_major_into, ternary_matmul_pq_planes_i_into, ternary_simd_i8_pq_into,
    wupz_bitplane_popcount_u64, wupz_popcount, wupz_xor_fold,
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
    let x_f64_scaled: Vec<f64> = x
        .iter()
        .zip(scales.iter())
        .map(|(&xv, &sv)| xv * sv)
        .collect();

    // Quantum 64-trit bitplanes & 4-bit bitplanes & Omni BFP-i8 buffers
    let words_per_col = rows.div_ceil(64);
    let num_groups = rows.div_ceil(MLX_GROUP_SIZE);
    let mut w_pos_u64 = vec![0u64; cols * words_per_col];
    let mut w_neg_u64 = vec![0u64; cols * words_per_col];
    pack_quantum_bitplanes_i_major_into(&trits, rows, cols, &mut w_pos_u64, &mut w_neg_u64);

    let mut x_pos_u64 = vec![0u64; words_per_col];
    let mut x_neg_u64 = vec![0u64; words_per_col];
    let scale_x_qt = quantize_quantum_ternary_act_pq_into(&x_p_scaled, &mut x_pos_u64, &mut x_neg_u64);

    let mut x_sign_u64 = vec![0u64; words_per_col];
    let mut x_b0_u64 = vec![0u64; words_per_col];
    let mut x_b1_u64 = vec![0u64; words_per_col];
    let mut x_b2_u64 = vec![0u64; words_per_col];
    let mut scale_x_i4_p = vec![0i32; words_per_col];
    quantize_quantum_bitplane_i4_pq_into(
        &x_p_scaled,
        &mut x_sign_u64,
        &mut x_b0_u64,
        &mut x_b1_u64,
        &mut x_b2_u64,
        &mut scale_x_i4_p,
    );

    let mut m_x_pq_i8 = vec![0i8; rows];
    let mut scale_x_pq_i8 = vec![0i32; num_groups];
    quantize_bfp_i8_pq_group64_into(&x_p_scaled, &mut m_x_pq_i8, &mut scale_x_pq_i8);

    let mut m_x_bfp_i8 = vec![0i8; rows];
    let mut scale_x_bfp_i8 = vec![0f64; num_groups];
    quantize_bfp_i8_group64_into(&x_f64_scaled, &mut m_x_bfp_i8, &mut scale_x_bfp_i8);

    let mut pq_quantum_u64 = vec![0i64; cols];
    let mut pq_quantum_i4 = vec![0i64; cols];
    let mut pq_omni_bfp_i8 = vec![0i64; cols];
    let mut bfp_i8_out = vec![0f64; cols];

    // Warm-up.
    dense_matmul(&w, rows, cols, &x, &mut dense);
    dense_matmul_pq_into(&w_p, rows, cols, &x_p, &mut dense_pq);
    quantum_ternary_matmul_pq_into(
        &w_pos_u64,
        &w_neg_u64,
        rows,
        cols,
        &x_pos_u64,
        &x_neg_u64,
        scale_x_qt,
        None,
        KERNEL_PQ_Q,
        &mut pq_quantum_u64,
    );
    quantum_bitplane_i4_matmul_pq_into(
        &w_pos_u64,
        &w_neg_u64,
        rows,
        cols,
        &x_sign_u64,
        &x_b0_u64,
        &x_b1_u64,
        &x_b2_u64,
        &scale_x_i4_p,
        &mut pq_quantum_i4,
    );
    omni_bfp_i8_simd_pq_into(
        &trits_i_major,
        rows,
        cols,
        &m_x_pq_i8,
        Some(&scale_x_pq_i8),
        &mut pq_omni_bfp_i8,
    );
    ternary_bfp_i8_matmul_into(
        &trits_i_major,
        rows,
        cols,
        &m_x_bfp_i8,
        &scale_x_bfp_i8,
        None,
        1.0,
        &mut bfp_i8_out,
    );
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
            if elapsed.as_millis() >= 25 {
                return elapsed.as_nanos() / iters as u128;
            }
            iters *= 2;
        }
    };

    let dense_ns = measure(&mut || {
        dense_matmul(black_box(&w), rows, cols, black_box(&x), &mut dense);
    });
    let quantum_u64_ns = measure(&mut || {
        quantum_ternary_matmul_pq_into(
            black_box(&w_pos_u64),
            black_box(&w_neg_u64),
            rows,
            cols,
            black_box(&x_pos_u64),
            black_box(&x_neg_u64),
            scale_x_qt,
            None,
            KERNEL_PQ_Q,
            black_box(&mut pq_quantum_u64),
        );
    });
    let quantum_i4_ns = measure(&mut || {
        quantum_bitplane_i4_matmul_pq_into(
            black_box(&w_pos_u64),
            black_box(&w_neg_u64),
            rows,
            cols,
            black_box(&x_sign_u64),
            black_box(&x_b0_u64),
            black_box(&x_b1_u64),
            black_box(&x_b2_u64),
            black_box(&scale_x_i4_p),
            black_box(&mut pq_quantum_i4),
        );
    });
    let omni_bfp_i8_ns = measure(&mut || {
        omni_bfp_i8_simd_pq_into(
            black_box(&trits_i_major),
            rows,
            cols,
            black_box(&m_x_pq_i8),
            Some(black_box(&scale_x_pq_i8)),
            black_box(&mut pq_omni_bfp_i8),
        );
    });
    let bfp_i8_ns = measure(&mut || {
        ternary_bfp_i8_matmul_into(
            black_box(&trits_i_major),
            rows,
            cols,
            black_box(&m_x_bfp_i8),
            black_box(&scale_x_bfp_i8),
            None,
            1.0,
            black_box(&mut bfp_i8_out),
        );
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
    let qt = fold_to_quant_tern_pq(&field_pq);
    let wupz_fold = wupz_xor_fold(&packed_i);
    let wupz_pop = wupz_popcount(&packed_i);
    let wupz_qt_active = wupz_bitplane_popcount_u64(&w_pos_u64, &w_neg_u64);

    // ---- references -------------------------------------------------------
    // `dense` (f64) and `dense_pq` (i64, Q^2 units) cover the (rows x cols) -> cols
    // orientation. `pq-group64` is the transposed orientation
    // (`p_out[n] = sum_g scales[n,g] * dot(codes, x)`), so it needs its own reference.
    let mut dense_g64_pq = vec![0i64; rows];
    for r in 0..rows {
        let mut acc: i128 = 0;
        for c in 0..aligned_cols {
            acc += (w_p_al[r * aligned_cols + c] as i128) * (x_p_al[c] as i128);
        }
        dense_g64_pq[r] = acc as i64;
    }

    let q_scale = KERNEL_PQ_Q as i64;
    let to_q2 = |v: &[i64]| -> Vec<i64> { v.iter().map(|&x| x * q_scale).collect() };
    let rel_pq = |v: &[i64], reference: &[i64]| -> f64 {
        relative_error_pq_ppm(&to_q2(v), reference) as f64 / 1_000_000.0
    };

    // Every timed kernel reports a measured error against the reference it is
    // actually comparable to. No row is permitted to report speed alone: a
    // speedup without an accuracy figure is a cherry-pick.
    let table: Vec<(&str, u128, f64, &str)> = vec![
        ("dense", dense_ns, 0.0, "f64, auto-vectorized reference"),
        (
            "quantum-b158-u64",
            quantum_u64_ns,
            rel_pq(&pq_quantum_u64, &dense_pq),
            "64-trit POPCNT 8-col, 0 floats, ternary W x TERNARY A (hence the higher error)",
        ),
        (
            "quantum-bitplane4",
            quantum_i4_ns,
            rel_pq(&pq_quantum_i4, &dense_pq),
            "W1.58 x A4-bitplane POPCNT 4-col, 0 floats, ternary W x 4-bit A",
        ),
        (
            "omni-bfp-i8-pq",
            omni_bfp_i8_ns,
            rel_pq(&pq_omni_bfp_i8, &dense_pq),
            "i8xi8->i16 4-col SIMD, 0 floats, block scales applied",
        ),
        (
            "bfp-i8-4col",
            bfp_i8_ns,
            relative_error(&bfp_i8_out, &dense),
            "i8xi8->i16 64-group 4-col BFP-64 SIMD",
        ),
        (
            "pq-simd-i8",
            pq_simd_ns,
            rel_pq(&pq_simd, &dense_pq),
            "i8xi32->i32 4-col NEON/AVX2 SIMD, 0 floats",
        ),
        (
            "pq-planes",
            pq_planes_ns,
            rel_pq(&pq_planes, &dense_pq),
            "i8 sign-planes 4-col p/q integer, 0 floats",
        ),
        (
            "pq-group64",
            pq_g64_ns,
            relative_error_pq_ppm(&pq_g64, &dense_g64_pq) as f64 / 1_000_000.0,
            "16-trit/u32 4-row MLX_BYTE_TRITS_I32, 0 floats, transposed",
        ),
        (
            "pq-b158-i",
            pq_b158_i_ns,
            rel_pq(&pq_b158_i, &dense_pq),
            "5-trits/byte 4-col BYTE_TRITS_I32, 0 floats",
        ),
        (
            "pq-b158-j",
            pq_b158_j_ns,
            rel_pq(&pq_b158_j, &dense_pq),
            "5-trits/byte j-major BYTE_TRITS_I32, 0 floats",
        ),
        (
            "planes",
            planes_ns,
            relative_error(&ternary_p, &dense),
            "i8 sign planes, pre-fused row scales",
        ),
        (
            "packed-i",
            ternary_i_ns,
            relative_error(&ternary_i, &dense),
            "5-trits/byte LUT path, unguarded interior",
        ),
        (
            "packed-j",
            ternary_j_ns,
            relative_error(&ternary, &dense),
            "j-major, JS-parity layout",
        ),
    ];

    println!(
        "GAIA-MLX-QUANT v{} bench — ULTRA-SUPER-OMNI-Quantum Ternary, wupz & p/q pure-integer",
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "  field             seed {} · dominant {:?} · coherence {:.4}",
        field.seed, field.dominant, field.coherence
    );
    println!(
        "  field-pq          q={} · coherence_p={} · T_p={} · top_k={} (0 floats)",
        GAIA_PQ_Q, field_pq.coherence_p, samp_pq.temperature_p, samp_pq.top_k
    );
    println!(
        "  wupz-seam         ABI v{WUPZ_ABI_VERSION} · {} (byte 0x{:02x}) · xor_fold 0x{wupz_fold:016x} · popcount {wupz_pop}b · active_trits {wupz_qt_active}",
        qt.as_str(),
        qt.byte
    );
    println!("  shape             {rows}x{cols}");
    println!(
        "  packing           {:.4} bits/weight · {} bytes vs {} f64 ({:.2}x) · group64 {:.2}x vs f32",
        BITS_PER_WEIGHT,
        packed_i.len(),
        w.len() * 8,
        (w.len() * 8) as f64 / packed_i.len() as f64,
        MLX_GROUP64_COMPRESSION,
    );
    println!("  kernel             ns/iter  vs dense     rel-L2        ppm  notes");
    let mut worst: (f64, &str) = (0.0, "dense");
    for (name, ns, rel, note) in &table {
        // The ratio is printed for every row, including the ones slower than the
        // f64 dense reference. A table that only labels its wins is not honest.
        let ratio = (dense_ns as f64) / (*ns as f64).max(1.0);
        let ppm = (rel * 1_000_000.0).round() as u64;
        println!(
            "  {:<17}{:>9}  {:>8.2}x  {:>8.4}  {:>9}  {}",
            name, ns, ratio, rel, ppm, note
        );
        if *rel > worst.0 {
            worst = (*rel, name);
        }
    }
    println!(
        "  worst-case        {:.4} relative L2 ({} ppm) on `{}` — the largest reconstruction error of any kernel above",
        worst.0,
        (worst.0 * 1_000_000.0).round() as u64,
        worst.1
    );
}
