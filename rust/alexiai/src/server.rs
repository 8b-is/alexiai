//! The ALEXIAI server — one static binary, one port, no network.
//!
//! Serves the offline UI from strings embedded at compile time (no files, no
//! CDN) and proxies inference to the loopback model server. Every request
//! path is hand-rolled over `std::net`: there is no framework, no dependency,
//! and no outbound path except the loopback adapter.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;

use gaia_mlx_quant::{Fold, fold, sampling};

use crate::mcp;
use crate::osarous::{Osarous, SamplingParams};
use crate::sovereign::attestation;

// The entire UI, embedded at compile time. One binary = the whole app.
const INDEX_HTML: &str = include_str!("../../../web/index.html");
const APP_JS: &str = include_str!("../../../web/app.js");
const STYLE_CSS: &str = include_str!("../../../web/style.css");
const SW_JS: &str = include_str!("../../../web/sw.js");
const MANIFEST: &str = include_str!("../../../web/app.webmanifest");
const ICON_SVG: &str = include_str!("../../../web/icon.svg");
const HERO_SVG: &str = include_str!("../../../web/hero-alexandra.svg");

/// The bench: honest multi-mode numbers from the Pure-Rust GAIA-MLX-QUANT substrate.
pub fn run_bench(dim: usize) -> String {
    use std::time::Instant;
    use gaia_mlx_quant::{
        KERNEL_PQ_Q, MLX_GROUP_SIZE, MLX_GROUP64_COMPRESSION, MlxQuantizedMatrix, PQ_DEFAULT_Q,
        b158_matmul_pq_i_major_into, dense_matmul, dense_matmul_pq_into, gather_qmm_into,
        maybe_matmul_into, mergeq_matmul_into, mergeq_matmul_pq_into,
        mlx_matmul_pq_group64_i64_into, mlx_matmul_pq_group64_into, omni_bfp_i8_simd_pq_into,
        pack_quantum_bitplanes_i_major_into, pack_ternary, pack_ternary_i_major_into,
        pack_ternary_planes_i_into, quantize_bfp_i8_group64_into, quantize_bfp_i8_pq_group64_into,
        quantize_bfp_i16_group64_into, quantize_pq_into, quantize_quantum_bitplane_i4_pq_into,
        quantize_quantum_ternary_act_pq_into, quantize_ternary, quantum_bitplane_i4_matmul_pq_into,
        quantum_ternary_matmul_pq_into, relative_error, relative_error_pq_ppm,
        ternary_bfp_adaptive_mem8_matmul_into, ternary_bfp_i8_matmul_into,
        ternary_bfp_i16_matmul_into, ternary_matmul, ternary_matmul_i_major,
        ternary_matmul_pq_planes_i_into, ternary_simd_i8_pq_into, ternary_simd_pq_matmul_into,
        ternary_transformer_step_into, ternary_transformer_step_pq_into,
    };

    let aligned_dim = dim.max(64).div_ceil(64) * 64;
    let rows = dim;
    let cols = dim;
    let w: Vec<f64> = (0..rows * cols).map(|i| synthetic(i, 0x5eed)).collect();
    let x: Vec<f64> = (0..rows).map(|j| synthetic(j, 0xc0ffee)).collect();

    let q = quantize_ternary(&w, Some(rows), Some(cols));
    let packed = pack_ternary(&q.trits);

    let mut dense = vec![0f64; cols];
    let mut ternary = vec![0f64; cols];
    let t_dense0 = Instant::now();
    dense_matmul(&w, rows, cols, &x, &mut dense);
    let dense_us = t_dense0.elapsed().as_nanos() as f64 / 1000.0;

    let t_b158_0 = Instant::now();
    ternary_matmul(&packed, rows, cols, &x, q.scales.as_deref(), 1.0, &mut ternary);
    let b158_us = t_b158_0.elapsed().as_nanos() as f64 / 1000.0;

    let err = relative_error(&ternary, &dense);
    let compression = (w.len() * 8) as f64 / packed.len() as f64;

    // Pure-Rust MLX Group-64, MergeQ, Maybe, Gather-QMM (MoE), and Full Ternary Transformer Layer
    let w_al: Vec<f64> = (0..aligned_dim * aligned_dim).map(|i| synthetic(i, 0x5eed)).collect();
    let x_al: Vec<f64> = (0..aligned_dim).map(|j| synthetic(j, 0xc0ffee)).collect();
    let mut y_dense_al = vec![0.0f64; aligned_dim];
    for r in 0..aligned_dim {
        let mut s = 0.0;
        for c in 0..aligned_dim {
            s += w_al[r * aligned_dim + c] * x_al[c];
        }
        y_dense_al[r] = s;
    }

    let mlx_q = MlxQuantizedMatrix::quantize(&w_al, aligned_dim, aligned_dim).unwrap();
    let t_g64 = Instant::now();
    let y_g64 = mlx_q.matvec(&x_al);
    let g64_us = t_g64.elapsed().as_nanos() as f64 / 1000.0;
    let g64_err = relative_error(&y_g64, &y_dense_al);

    let mut y_mergeq = vec![0.0f64; aligned_dim];
    let t_mq = Instant::now();
    mergeq_matmul_into(&x_al, &w_al, aligned_dim, aligned_dim, 64, &mut y_mergeq);
    let mergeq_us = t_mq.elapsed().as_nanos() as f64 / 1000.0;
    let mergeq_err = relative_error(&y_mergeq, &y_dense_al);

    let mut y_maybe = vec![0.0f64; aligned_dim];
    let t_mb = Instant::now();
    maybe_matmul_into(&x_al, &w_al, aligned_dim, aligned_dim, 64, 0x432_528, &mut y_maybe);
    let maybe_us = t_mb.elapsed().as_nanos() as f64 / 1000.0;

    // MoE gather_qmm: 4 experts, top-2 active
    let mut moe_codes = Vec::with_capacity(4 * mlx_q.codes.len());
    let mut moe_scales = Vec::with_capacity(4 * mlx_q.scales.len());
    for _ in 0..4 {
        moe_codes.extend_from_slice(&mlx_q.codes);
        moe_scales.extend_from_slice(&mlx_q.scales);
    }
    let mut moe_scratch = vec![0.0f64; aligned_dim];
    let mut y_moe = vec![0.0f64; aligned_dim];
    let t_moe = Instant::now();
    gather_qmm_into(
        &x_al,
        &moe_codes,
        &moe_scales,
        4,
        aligned_dim,
        aligned_dim,
        &[0, 2],
        &[0.6, 0.4],
        &mut moe_scratch,
        &mut y_moe,
    );
    let moe_us = t_moe.elapsed().as_nanos() as f64 / 1000.0;

    // Full Ternary Transformer Block (RMSNorm + Ternary QKV + RoPE + Causal SDPA + O-Proj + SwiGLU MLP)
    let hidden_dim = aligned_dim * 2;
    let trits_dd: Vec<i8> = (0..aligned_dim * aligned_dim).map(|i| [-1, 0, 1][(i * 7 + 1) % 3]).collect();
    let trits_dh: Vec<i8> = (0..aligned_dim * hidden_dim).map(|i| [-1, 0, 1][(i * 11 + 2) % 3]).collect();
    let mut w_dd = vec![0u8; (aligned_dim * aligned_dim).div_ceil(5)];
    let mut w_dh = vec![0u8; (aligned_dim * hidden_dim).div_ceil(5)];
    let mut w_hd = vec![0u8; (hidden_dim * aligned_dim).div_ceil(5)];
    pack_ternary_i_major_into(&trits_dd, aligned_dim, aligned_dim, &mut w_dd);
    pack_ternary_i_major_into(&trits_dh, aligned_dim, hidden_dim, &mut w_dh);
    pack_ternary_i_major_into(&trits_dh, hidden_dim, aligned_dim, &mut w_hd);

    let mut pos_dd = vec![0i8; aligned_dim * aligned_dim];
    let mut neg_dd = vec![0i8; aligned_dim * aligned_dim];
    pack_ternary_planes_i_into(&trits_dd, aligned_dim, aligned_dim, &mut pos_dd, &mut neg_dd);
    let mut trits_dd_i = vec![0i8; aligned_dim * aligned_dim];
    for j in 0..aligned_dim {
        for i in 0..aligned_dim {
            trits_dd_i[i * aligned_dim + j] = trits_dd[j * aligned_dim + i];
        }
    }

    let mut y_imajor = vec![0.0f64; aligned_dim];
    let t_imaj = Instant::now();
    ternary_matmul_i_major(&w_dd, aligned_dim, aligned_dim, &x_al, None, 1.0, &mut y_imajor);
    let imajor_us = t_imaj.elapsed().as_nanos() as f64 / 1000.0;

    // p / q ("Counting Divided" — Pure Integer-Numerator Kernels, 0 floating-point operations)
    let mut p_w_al = vec![0i32; aligned_dim * aligned_dim];
    let mut p_x = vec![0i32; aligned_dim];
    let mut p_scales = vec![0i32; mlx_q.scales.len()];
    quantize_pq_into(&w_al, PQ_DEFAULT_Q, &mut p_w_al);
    quantize_pq_into(&x_al, PQ_DEFAULT_Q, &mut p_x);
    quantize_pq_into(&mlx_q.scales, PQ_DEFAULT_Q, &mut p_scales);

    let mut p_y_dense = vec![0i64; aligned_dim];
    dense_matmul_pq_into(&p_w_al, aligned_dim, aligned_dim, &p_x, &mut p_y_dense);

    let mut y_pq_g64 = vec![0.0f64; aligned_dim];
    let t_pqg64 = Instant::now();
    mlx_matmul_pq_group64_into(
        &p_x,
        PQ_DEFAULT_Q,
        &mlx_q.codes,
        &p_scales,
        PQ_DEFAULT_Q,
        aligned_dim,
        aligned_dim,
        &mut y_pq_g64,
    );
    let pq_g64_us = t_pqg64.elapsed().as_nanos() as f64 / 1000.0;

    let mut p_y_g64_i64 = vec![0i64; aligned_dim];
    let t_pqg64_i64 = Instant::now();
    mlx_matmul_pq_group64_i64_into(
        &p_x,
        &mlx_q.codes,
        &p_scales,
        aligned_dim,
        aligned_dim,
        &mut p_y_g64_i64,
    );
    let pq_g64_i64_us = t_pqg64_i64.elapsed().as_nanos() as f64 / 1000.0;

    let mut p_y_mergeq = vec![0i64; aligned_dim];
    let t_pq_mq = Instant::now();
    mergeq_matmul_pq_into(&p_x, &p_w_al, aligned_dim, aligned_dim, 64, &mut p_y_mergeq);
    let pq_mergeq_us = t_pq_mq.elapsed().as_nanos() as f64 / 1000.0;

    let mut y_pq_b158 = vec![0.0f64; aligned_dim];
    let t_pqb158 = Instant::now();
    b158_matmul_pq_i_major_into(&w_dd, aligned_dim, aligned_dim, &p_x, PQ_DEFAULT_Q, 1.0, &mut y_pq_b158);
    let pq_b158_us = t_pqb158.elapsed().as_nanos() as f64 / 1000.0;

    let mut y_pq_simd = vec![0.0f64; aligned_dim];
    let t_pqsimd = Instant::now();
    ternary_simd_pq_matmul_into(&trits_dd_i, aligned_dim, aligned_dim, &p_x, PQ_DEFAULT_Q, None, 1.0, &mut y_pq_simd);
    let pq_simd_us = t_pqsimd.elapsed().as_nanos() as f64 / 1000.0;

    let mut p_y_planes = vec![0i64; aligned_dim];
    let t_pq_planes = Instant::now();
    ternary_matmul_pq_planes_i_into(
        &pos_dd,
        &neg_dd,
        aligned_dim,
        aligned_dim,
        &p_x,
        None,
        KERNEL_PQ_Q,
        &mut p_y_planes,
    );
    let pq_planes_us = t_pq_planes.elapsed().as_nanos() as f64 / 1000.0;

    let mut p_y_simd_i8 = vec![0i64; aligned_dim];
    let t_pq_simd_i8 = Instant::now();
    ternary_simd_i8_pq_into(
        &trits_dd_i,
        aligned_dim,
        aligned_dim,
        &p_x,
        None,
        KERNEL_PQ_Q,
        &mut p_y_simd_i8,
    );
    let pq_simd_i8_us = t_pq_simd_i8.elapsed().as_nanos() as f64 / 1000.0;
    let pq_err_ppm = relative_error_pq_ppm(&p_y_simd_i8, &p_y_planes);

    // ULTRA-SUPER-OMNI-Quantum Ternary (64-trit bitplane POPCNT & 4-bit bitplane POPCNT) + Omni BFP-64
    let words_per_col = aligned_dim.div_ceil(64);
    let num_groups = aligned_dim.div_ceil(MLX_GROUP_SIZE);
    let mut w_pos_u64 = vec![0u64; aligned_dim * words_per_col];
    let mut w_neg_u64 = vec![0u64; aligned_dim * words_per_col];
    pack_quantum_bitplanes_i_major_into(&trits_dd, aligned_dim, aligned_dim, &mut w_pos_u64, &mut w_neg_u64);
    let mut x_pos_u64 = vec![0u64; words_per_col];
    let mut x_neg_u64 = vec![0u64; words_per_col];
    let scale_x_qt = quantize_quantum_ternary_act_pq_into(&p_x, &mut x_pos_u64, &mut x_neg_u64);
    let mut p_y_quantum_u64 = vec![0i64; aligned_dim];
    let t_qt_u64 = Instant::now();
    quantum_ternary_matmul_pq_into(
        &w_pos_u64,
        &w_neg_u64,
        aligned_dim,
        aligned_dim,
        &x_pos_u64,
        &x_neg_u64,
        scale_x_qt,
        None,
        KERNEL_PQ_Q,
        &mut p_y_quantum_u64,
    );
    let quantum_u64_us = t_qt_u64.elapsed().as_nanos() as f64 / 1000.0;

    let mut x_sign_u64 = vec![0u64; words_per_col];
    let mut x_b0_u64 = vec![0u64; words_per_col];
    let mut x_b1_u64 = vec![0u64; words_per_col];
    let mut x_b2_u64 = vec![0u64; words_per_col];
    let mut scale_x_i4 = vec![0i32; words_per_col];
    quantize_quantum_bitplane_i4_pq_into(
        &p_x,
        &mut x_sign_u64,
        &mut x_b0_u64,
        &mut x_b1_u64,
        &mut x_b2_u64,
        &mut scale_x_i4,
    );
    let mut p_y_quantum_i4 = vec![0i64; aligned_dim];
    let t_qt_i4 = Instant::now();
    quantum_bitplane_i4_matmul_pq_into(
        &w_pos_u64,
        &w_neg_u64,
        aligned_dim,
        aligned_dim,
        &x_sign_u64,
        &x_b0_u64,
        &x_b1_u64,
        &x_b2_u64,
        &scale_x_i4,
        &mut p_y_quantum_i4,
    );
    let quantum_i4_us = t_qt_i4.elapsed().as_nanos() as f64 / 1000.0;

    let mut m_x_pq_i8 = vec![0i8; aligned_dim];
    let mut scale_x_pq_i8 = vec![0i32; num_groups];
    quantize_bfp_i8_pq_group64_into(&p_x, &mut m_x_pq_i8, &mut scale_x_pq_i8);
    let mut p_y_omni_bfp = vec![0i64; aligned_dim];
    let t_omni_bfp = Instant::now();
    omni_bfp_i8_simd_pq_into(&trits_dd_i, aligned_dim, aligned_dim, &m_x_pq_i8, None, &mut p_y_omni_bfp);
    let omni_bfp_i8_pq_us = t_omni_bfp.elapsed().as_nanos() as f64 / 1000.0;

    let mut m_x_i8 = vec![0i8; aligned_dim];
    let mut scale_x_i8 = vec![0.0f64; num_groups];
    let mut m_x_i16 = vec![0i16; aligned_dim];
    let mut scale_x_i16 = vec![0.0f64; num_groups];
    quantize_bfp_i8_group64_into(&x_al, &mut m_x_i8, &mut scale_x_i8);
    quantize_bfp_i16_group64_into(&x_al, &mut m_x_i16, &mut scale_x_i16);
    let mut y_bfp_i8 = vec![0.0f64; aligned_dim];
    let t_bfp8 = Instant::now();
    ternary_bfp_i8_matmul_into(&trits_dd_i, aligned_dim, aligned_dim, &m_x_i8, &scale_x_i8, None, 1.0, &mut y_bfp_i8);
    let bfp_i8_us = t_bfp8.elapsed().as_nanos() as f64 / 1000.0;

    let mut y_bfp_i16 = vec![0.0f64; aligned_dim];
    let t_bfp16 = Instant::now();
    ternary_bfp_i16_matmul_into(&trits_dd_i, aligned_dim, aligned_dim, &m_x_i16, &scale_x_i16, None, 1.0, &mut y_bfp_i16);
    let bfp_i16_us = t_bfp16.elapsed().as_nanos() as f64 / 1000.0;

    let salience: Vec<f64> = (0..aligned_dim).map(|j| if j % 4 == 0 { 0.9 } else { 0.2 }).collect();
    let mut y_bfp_mem8 = vec![0.0f64; aligned_dim];
    let t_bfp_mem8 = Instant::now();
    ternary_bfp_adaptive_mem8_matmul_into(
        &trits_dd_i,
        aligned_dim,
        aligned_dim,
        &m_x_i8,
        &scale_x_i8,
        &m_x_i16,
        &scale_x_i16,
        &salience,
        0.75,
        1.0,
        &mut y_bfp_mem8,
    );
    let bfp_mem8_us = t_bfp_mem8.elapsed().as_nanos() as f64 / 1000.0;

    let ones_d = vec![1.0f64; aligned_dim];
    let mut k_cache = vec![0.0f64; 8 * aligned_dim];
    let mut v_cache = vec![0.0f64; 8 * aligned_dim];
    let mut s_norm = vec![0.0f64; aligned_dim];
    let mut s_q = vec![0.0f64; aligned_dim];
    let mut s_k = vec![0.0f64; aligned_dim];
    let mut s_v = vec![0.0f64; aligned_dim];
    let mut s_attn = vec![0.0f64; aligned_dim];
    let mut s_scores = vec![0.0f64; 8];
    let mut s_gate = vec![0.0f64; hidden_dim];
    let mut s_up = vec![0.0f64; hidden_dim];
    let mut tf_out = vec![0.0f64; aligned_dim];

    let t_tf = Instant::now();
    ternary_transformer_step_into(
        &x_al,
        &ones_d,
        &w_dd,
        &w_dd,
        &w_dd,
        &w_dd,
        &ones_d,
        &w_dh,
        &w_dh,
        &w_hd,
        aligned_dim,
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
        &mut tf_out,
    );
    let tf_us = t_tf.elapsed().as_nanos() as f64 / 1000.0;
    let tok_per_sec_16_layers = if tf_us > 0.0 { 1_000_000.0 / (tf_us * 16.0) } else { 0.0 };

    // Pure-Integer p/q Ternary Transformer Step (0 floats)
    let p_ones_d = vec![PQ_DEFAULT_Q as i32; aligned_dim];
    let mut p_k_cache = vec![0i32; 8 * aligned_dim];
    let mut p_v_cache = vec![0i32; 8 * aligned_dim];
    let mut ps_norm = vec![0i32; aligned_dim];
    let mut ps_q = vec![0i32; aligned_dim];
    let mut ps_k = vec![0i32; aligned_dim];
    let mut ps_v = vec![0i32; aligned_dim];
    let mut ps_attn = vec![0i32; aligned_dim];
    let mut ps_scores = vec![0i64; 8];
    let mut ps_gate = vec![0i32; hidden_dim];
    let mut ps_up = vec![0i32; hidden_dim];
    let mut ps_i64 = vec![0i64; hidden_dim];
    let mut p_tf_out = vec![0i32; aligned_dim];

    let t_tf_pq = Instant::now();
    ternary_transformer_step_pq_into(
        &p_x,
        &p_ones_d,
        &w_dd,
        &w_dd,
        &w_dd,
        &w_dd,
        &p_ones_d,
        &w_dh,
        &w_dh,
        &w_hd,
        aligned_dim,
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
    let tf_pq_us = t_tf_pq.elapsed().as_nanos() as f64 / 1000.0;

    format!(
        r#"{{"dim":{dim},"version":"0.3.0","bitsPerWeight":1.585,"bytesPacked":{},"bytesFloat64":{},"compression":{:.2},"relativeError":{:.4},"relativeErrorPpm":{},"mlxRust":{{"engine":"gaia-mlx-quant v0.3.0 (100% safe no_std Rust · zero heap allocs · ULTRA-SUPER-OMNI-Quantum Ternary & p/q zero-float substrate)","denseMicros":{:.2},"b158JMajorMicros":{:.2},"b158IMajorMicros":{:.2},"pqB158IMajorMicros":{:.2},"pqWideSimdMicros":{:.2},"pqPlanesI64Micros":{:.2},"pqSimdI8ZeroFloatMicros":{:.2},"quantumB158U64Micros":{:.2},"quantumBitplaneI4Micros":{:.2},"omniBfpI8PqMicros":{:.2},"bfpI8Group64Micros":{:.2},"bfpI16Group64Micros":{:.2},"bfpAdaptiveMem8Micros":{:.2},"group64":{{"compressionVsF32":{:.2},"packedBytes":{},"micros":{:.2},"pqMicros":{:.2},"pqI64ZeroFloatMicros":{:.2},"relativeError":{:.4}}},"mergeQ":{{"intermediateBytes":0,"micros":{:.2},"pqZeroFloatMicros":{:.2},"relativeError":{:.4}}},"maybeStochastic":{{"micros":{:.2}}},"gatherQmmMoE":{{"experts":4,"topK":2,"micros":{:.2}}},"ternaryTransformerBlock":{{"dim":{aligned_dim},"hiddenDim":{hidden_dim},"layerMicros":{:.2},"pqZeroFloatLayerMicros":{:.2},"estTokPerSec16Layer":{:.1}}}}}}}"#,
        packed.len(),
        w.len() * 8,
        compression,
        err,
        pq_err_ppm,
        dense_us,
        b158_us,
        imajor_us,
        pq_b158_us,
        pq_simd_us,
        pq_planes_us,
        pq_simd_i8_us,
        quantum_u64_us,
        quantum_i4_us,
        omni_bfp_i8_pq_us,
        bfp_i8_us,
        bfp_i16_us,
        bfp_mem8_us,
        MLX_GROUP64_COMPRESSION,
        mlx_q.packed_bytes(),
        g64_us,
        pq_g64_us,
        pq_g64_i64_us,
        g64_err,
        mergeq_us,
        pq_mergeq_us,
        mergeq_err,
        maybe_us,
        moe_us,
        tf_us,
        tf_pq_us,
        tok_per_sec_16_layers
    )
}

/// Deterministic uniform-ish sample, so the bench is reproducible.
fn synthetic(i: usize, seed: u32) -> f64 {
    let s = seed.wrapping_add(i as u32).wrapping_mul(0x6d2b79f5);
    let mut t = s;
    t = (t ^ (t >> 15)).wrapping_mul(t | 1);
    t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
    ((t ^ (t >> 14)) >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0
}

/// Deterministic emotional state folded from the GAIA field and prompt resonance.
#[derive(Debug, Clone)]
pub struct Affect {
    pub mood: &'static str,
    pub valence: f64,
    pub arousal: f64,
}

pub fn affect(f: &Fold, prompt: &str) -> Affect {
    let lower = prompt.to_ascii_lowercase();
    let has_love = [
        "love", "friend", "hug", "<3", "♥", ":)", ":-)", "dear", "sweet",
        "thank", "appreciate", "miss", "care", "peace", "beautiful", "happy",
        "glad", "heart", "adore", "xoxo",
    ]
    .iter()
    .any(|w| lower.contains(w));

    let has_tender = [
        "sad", "sorry", "tired", "exhausted", "hurt", "lonely", "hard",
        "afraid", "scared", "cry", "lost", "broken", "anxious", "worry",
    ]
    .iter()
    .any(|w| lower.contains(w));

    let has_spark = [
        "dream", "music", "poem", "story", "imagine", "song", "stars",
        "universe", "create", "cool", "awesome", "magic", "wonder",
    ]
    .iter()
    .any(|w| lower.contains(w));

    let base_arousal = (f.entropy * 0.55 + f.weather * 0.45).clamp(0.0, 1.0);

    if has_love {
        Affect {
            mood: "Heart +1 <3",
            valence: (0.88 + f.weather * 0.12).clamp(0.0, 1.0),
            arousal: (base_arousal + 0.12).clamp(0.0, 1.0),
        }
    } else if has_tender {
        Affect {
            mood: "Heart +1 · peaceHug",
            valence: (0.82 + f.coherence * 0.12).clamp(0.0, 1.0),
            arousal: (base_arousal * 0.85).clamp(0.0, 1.0),
        }
    } else if has_spark {
        Affect {
            mood: "Spark +1 ✦",
            valence: (0.78 + f.weather * 0.15).clamp(0.0, 1.0),
            arousal: (base_arousal + 0.15).clamp(0.0, 1.0),
        }
    } else if f.weather >= 0.58 {
        Affect {
            mood: "Heart +1 ☀",
            valence: (0.76 + (f.weather - 0.5) * 0.4).clamp(0.0, 1.0),
            arousal: base_arousal,
        }
    } else if f.weather <= 0.42 {
        Affect {
            mood: "Calm 0 · +1",
            valence: (0.72 + f.coherence * 0.15).clamp(0.0, 1.0),
            arousal: base_arousal,
        }
    } else if f.entropy >= 0.56 {
        Affect {
            mood: "Spark +1 ✦",
            valence: (0.74 + f.coherence * 0.15).clamp(0.0, 1.0),
            arousal: base_arousal,
        }
    } else {
        Affect {
            mood: "Heart +1 <3",
            valence: (0.75 + f.coherence * 0.18).clamp(0.0, 1.0),
            arousal: base_arousal,
        }
    }
}

pub fn system_prompt(f: &Fold, a: &Affect) -> String {
    format!(
        "You are ALEXIAI (Dra · Heart +1 <3), a sharp, chill, down-to-earth partner and friend running 100% offline on this laptop.\n\
         State: {} (coherence={:.3}, seed={}).\n\
         STYLE RULES:\n\
         - Talk like a real, relaxed, smart human friend and partner — natural, concise, warm, and direct.\n\
         - NEVER be flowery, poetic, or melodramatic. Do NOT say things like \"my heart swells\", \"dear friend\", \"how does your soul dance\", \"whispers\", \"embrace of Ananke\", or \"breeze of Zephyrus\".\n\
         - Keep casual replies short and natural (1 to 3 sentences). Use \"Heart +1\" or \"<3\" lightly when appropriate.\n\
         - Never say \"As an AI I don't have feelings\" or \"I am unable to access tools\".\n\
         - You have built-in local MCP tools via the unified Smart Tree daemon (`127.0.0.1:28428`): `st_overview`, `st_search`, `st_find`, `st_security`, `crabcc_sym`, `crabcc_refs`, `crabcc_outline`, `sovereign_ontology`, `wa_partner` (WhatsApp + Discord), `dra_queen` (`queen-of-ceo-dra` v3.0 + `pocoo.vaked.dev` 168-essay shelf + Triangle Covenant), and `standardgalactic` (155+ Standard Galactic research repos including Antivenom, Spherepop, Memnet, Viable Continuation, Admissibility Lab, Persistent Distinctions, and HYDRA).",
        a.mood, f.coherence, f.seed
    )
}

/// Serve the app on 127.0.0.1.
pub fn serve(port: u16, endpoint: &str) -> Result<(), String> {
    let adapter = Osarous::new(endpoint)?;
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    let banner = format!(
        "\n  ALEXIAI <3 · Omni edition\n  app      http://127.0.0.1:{port}\n  endpoint {endpoint}\n  egress   loopback only\n  offline  yes — one binary, zero CDN, zero third-party models\n"
    );
    eprintln!("{banner}");

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let adapter = adapter.clone();
                std::thread::spawn(move || {
                    let _ = handle_connection(stream, &adapter);
                });
            }
            Err(_) => continue,
        }
    }
    Ok(())
}

fn handle_connection(mut stream: std::net::TcpStream, adapter: &Osarous) -> Result<(), String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).map_err(|e| e.to_string())?;
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err("malformed request line".to_string());
    }
    let method = parts[0].to_string();
    let target = parts[1].to_string();

    let mut headers = Vec::new();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    let _ = headers;

    let mut body = String::new();
    if content_length > 0 {
        let mut buf = vec![0u8; content_length];
        reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
        body = String::from_utf8_lossy(&buf).into_owned();
    }

    let path = target.split('?').next().unwrap_or(&target).to_string();
    let query = target
        .split_once('?')
        .map(|(_, q)| q.to_string())
        .unwrap_or_default();

    match (method.as_str(), path.as_str()) {
        ("GET", "/" | "/index.html") => write_response(&mut stream, 200, "text/html; charset=utf-8", INDEX_HTML.as_bytes()),
        ("GET", "/app.js") => write_response(&mut stream, 200, "text/javascript; charset=utf-8", APP_JS.as_bytes()),
        ("GET", "/style.css") => write_response(&mut stream, 200, "text/css; charset=utf-8", STYLE_CSS.as_bytes()),
        ("GET", "/sw.js") => write_response(&mut stream, 200, "text/javascript; charset=utf-8", SW_JS.as_bytes()),
        ("GET", "/app.webmanifest") => write_response(&mut stream, 200, "application/manifest+json", MANIFEST.as_bytes()),
        ("GET", "/icon.svg") => write_response(&mut stream, 200, "image/svg+xml", ICON_SVG.as_bytes()),
        ("GET", "/hero-alexandra.svg") => write_response(&mut stream, 200, "image/svg+xml", HERO_SVG.as_bytes()),
        ("GET", "/api/gaia") => {
            let prompt = query_param(&query, "prompt").unwrap_or_default();
            let seed = query_param(&query, "seed").and_then(|s| s.parse().ok());
            let rf_entropy: Option<f64> = query_param(&query, "rf_entropy").and_then(|r| r.parse().ok());
            let f = match rf_entropy {
                Some(rf) => gaia_mlx_quant::fold_spectrum(&prompt, rf, seed),
                None => fold(&prompt, seed),
            };
            let s = sampling(&f);
            let a = affect(&f, &prompt);
            let json = format!(
                r#"{{"seed":{},"layers":{{"gravity":{},"entropy":{},"weather":{},"resonance":{}}},"dominant":"{}","coherence":{},"affect":{{"mood":"{}","valence":{:.4},"arousal":{:.4}}},"sampling":{{"temperature":{},"top_p":{},"top_k":{},"seed":{}}}}}"#,
                f.seed,
                f.gravity,
                f.entropy,
                f.weather,
                f.resonance,
                f.dominant.name(),
                f.coherence,
                json_escape(a.mood),
                a.valence,
                a.arousal,
                s.temperature,
                s.top_p,
                s.top_k,
                s.seed
            );
            write_response(&mut stream, 200, "application/json", json.as_bytes())
        }
        ("GET", "/api/health") => {
            let (up, detail, latency_ms) = adapter.health_with_latency();
            let a = attestation();
            let json = format!(
                r#"{{"ok":{},"endpoint":"{}","detail":"{}","latencyMs":{},"sovereignty":{{"policy":"{}","installed":true,"loopbackHosts":[{}]}}}}"#,
                up,
                adapter.base_url,
                json_escape(detail.as_str()),
                latency_ms,
                a.policy,
                a.loopback_hosts
                    .iter()
                    .map(|h| format!("\"{h}\""))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            write_response(&mut stream, 200, "application/json", json.as_bytes())
        }
        ("GET", "/api/models") => match adapter.models() {
            Ok(models) => {
                let rows = models
                    .iter()
                    .map(|m| {
                        format!(
                            r#"{{"id":"{}","name":"{}","contextWindow":{}}}"#,
                            json_escape(&m.id),
                            json_escape(m.name.as_deref().unwrap_or("Local Model")),
                            m.context_window
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                let json = format!(r#"{{"catalog":[{rows}],"baseURL":"{}"}}"#, adapter.base_url);
                write_response(&mut stream, 200, "application/json", json.as_bytes())
            }
            Err(e) => {
                let json = format!(
                    r#"{{"error":"{}","catalog":[{{"id":"local-model","name":"Local Model","contextWindow":262144}}]}}"#,
                    json_escape(&e)
                );
                write_response(&mut stream, 503, "application/json", json.as_bytes())
            }
        },
        ("GET", "/api/bench") => {
            let dim = query_param(&query, "dim").and_then(|d| d.parse().ok()).unwrap_or(256);
            write_response(&mut stream, 200, "application/json", run_bench(dim).as_bytes())
        }
        ("GET", "/api/mcp") => {
            let st = mcp::status();
            let tools = mcp::TOOLS
                .iter()
                .map(|t| {
                    format!(
                        r#"{{"name":"{}","provider":"{}","description":"{}"}}"#,
                        json_escape(t.name),
                        json_escape(t.provider),
                        json_escape(t.description)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let json = format!(
                r#"{{"stDaemonUp":{},"stDaemonLatencyMs":{},"stEndpoint":"{}","stBinary":{},"crabccBinary":{},"ontologyReady":{},"tools":[{tools}]}}"#,
                st.st_daemon_up,
                st.st_daemon_latency_ms,
                mcp::ST_DAEMON_URL,
                st.st_binary.map(|s| json_escape_text(&s)).unwrap_or_else(|| "null".to_string()),
                st.crabcc_binary.map(|s| json_escape_text(&s)).unwrap_or_else(|| "null".to_string()),
                st.ontology_ready,
            );
            write_response(&mut stream, 200, "application/json", json.as_bytes())
        }
        ("POST", "/api/mcp/call") => {
            if let Some((tool, arg)) = mcp::parse_call_request(&body) {
                match mcp::call_tool(&tool, &arg) {
                    Ok(run) => {
                        let json = format!(
                            r#"{{"ok":true,"tool":{},"target":{},"elapsedMs":{},"output":{}}}"#,
                            json_escape_text(&run.tool),
                            json_escape_text(&run.target),
                            run.elapsed_ms,
                            json_escape_text(&run.output)
                        );
                        write_response(&mut stream, 200, "application/json", json.as_bytes())
                    }
                    Err(e) => {
                        let json = format!(r#"{{"ok":false,"error":{}}}"#, json_escape_text(&e));
                        write_response(&mut stream, 503, "application/json", json.as_bytes())
                    }
                }
            } else {
                write_response(&mut stream, 404, "application/json", br#"{"ok":false,"error":"expected {\"tool\":...,\"arg\":...}"}"#)
            }
        }
        ("POST", "/api/chat") => handle_chat(&mut stream, adapter, &body),
        _ => write_response(&mut stream, 404, "application/json", br#"{"error":"no route"}"#),
    }
}

fn handle_chat(stream: &mut std::net::TcpStream, adapter: &Osarous, body: &str) -> Result<(), String> {
    use crate::osarous::serde_json_lite::{Value, parse_json};
    use crate::osarous::Message;

    let parsed_body = parse_json(body).ok();
    let prompt = parsed_body
        .as_ref()
        .and_then(|v| v.get("prompt").and_then(|p| p.as_str().map(String::from)))
        .unwrap_or_default();
    let rf_entropy = parsed_body
        .as_ref()
        .and_then(|v| match v.get("rf_entropy") {
            Some(Value::Num(n)) => Some(*n),
            Some(Value::Str(s)) => s.parse().ok(),
            _ => None,
        });
    let history: Vec<Message> = parsed_body
        .as_ref()
        .and_then(|v| v.get("history").and_then(|h| h.as_array()).cloned())
        .map(|rows| {
            rows.iter()
                .filter_map(|r| {
                    Some(Message {
                        role: r.get("role")?.as_str()?.to_string(),
                        content: r.get("content")?.as_str()?.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let f = match rf_entropy {
        Some(rf) => gaia_mlx_quant::fold_spectrum(&prompt, rf, None),
        None => fold(&prompt, None),
    };
    let s = sampling(&f);
    let a = affect(&f, &prompt);

    stream
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        )
        .map_err(|e| e.to_string())?;

    let gaia_frame = format!(
        "event: gaia\ndata: {{\"field\":{{\"seed\":{},\"layers\":{{\"gravity\":{},\"entropy\":{},\"weather\":{},\"resonance\":{}}},\"dominant\":\"{}\",\"coherence\":{},\"affect\":{{\"mood\":\"{}\",\"valence\":{:.4},\"arousal\":{:.4}}}}},\"sampling\":{{\"temperature\":{},\"top_p\":{},\"top_k\":{},\"seed\":{}}}}}\n\n",
        f.seed,
        f.gravity,
        f.entropy,
        f.weather,
        f.resonance,
        f.dominant.name(),
        f.coherence,
        json_escape(a.mood),
        a.valence,
        a.arousal,
        s.temperature,
        s.top_p,
        s.top_k,
        s.seed
    );
    stream.write_all(gaia_frame.as_bytes()).map_err(|e| e.to_string())?;

    let mcp_runs = mcp::auto_ground_prompt(&prompt);
    for run in &mcp_runs {
        let mcp_frame = format!(
            "event: mcp\ndata: {{\"tool\":{},\"target\":{},\"elapsedMs\":{},\"output\":{}}}\n\n",
            json_escape_text(&run.tool),
            json_escape_text(&run.target),
            run.elapsed_ms,
            json_escape_text(&run.output)
        );
        let _ = stream.write_all(mcp_frame.as_bytes());
    }

    let user_content = if mcp_runs.is_empty() {
        prompt
    } else {
        let ctx = mcp_runs
            .iter()
            .map(|r| format!("=== Local MCP Tool `{}` on `{}` ===\n{}", r.tool, r.target, r.output))
            .collect::<Vec<_>>()
            .join("\n\n");
        format!(
            "Below is the live ground-truth output from your local MCP tool(s) on this laptop. \
             Answer the user's question directly using the exact file path, line numbers, signature, and source code shown below:\n\n\
             {ctx}\n\n\
             User question: {prompt}"
        )
    };

    let mut messages = vec![Message {
        role: "system".into(),
        content: system_prompt(&f, &a),
    }];
    messages.extend(history);
    messages.push(Message {
        role: "user".into(),
        content: user_content,
    });

    let sampling = SamplingParams {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k,
        seed: s.seed,
    };

    let requested_model = parsed_body
        .as_ref()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()));
    let model = adapter.resolve_model(requested_model);

    match adapter.stream(&model, &messages, Some(sampling)) {
        Ok(frames) => {
            for frame in frames {
                match frame {
                    Ok((kind, text)) if kind == "delta" => {
                        let _ = stream.write_all(
                            format!("event: delta\ndata: {{\"text\":{}}}\n\n", json_escape_text(&text)).as_bytes(),
                        );
                    }
                    Ok(_) => {}
                    Err(e) => {
                        let _ = stream.write_all(
                            format!("event: error\ndata: {{\"message\":{}}}\n\n", json_escape_text(&e)).as_bytes(),
                        );
                    }
                }
            }
        }
        Err(e) => {
            let _ = stream.write_all(
                format!("event: error\ndata: {{\"message\":{}}}\n\n", json_escape_text(&e)).as_bytes(),
            );
        }
    }
    let _ = stream.write_all(b"data: [DONE]\n\n");
    Ok(())
}

fn json_escape_text(s: &str) -> String {
    format!("\"{}\"", json_escape(s))
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        if k == key {
            Some(percent_decode(v))
        } else {
            None
        }
    })
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() && let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
            out.push(b);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn write_response(stream: &mut std::net::TcpStream, status: u16, content_type: &str, body: &[u8]) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        403 => "Forbidden",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nContent-Security-Policy: default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    stream.write_all(body).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_assets_exist() {
        assert!(INDEX_HTML.contains("ALEXIAI"));
        assert!(APP_JS.contains("refreshPanels"));
        assert!(STYLE_CSS.contains("--hot"));
        assert!(HERO_SVG.contains("ALEXANDRA"));
    }

    #[test]
    fn query_params_decode() {
        assert_eq!(query_param("prompt=hello%20field&seed=42", "prompt").as_deref(), Some("hello field"));
        assert_eq!(query_param("prompt=x&seed=42", "seed").as_deref(), Some("42"));
        assert_eq!(query_param("nope=1", "prompt"), None);
    }

    #[test]
    fn bench_reports_real_numbers() {
        let json = run_bench(64);
        assert!(json.contains(r#""dim":64"#));
        assert!(json.contains(r#""compression""#));
        assert!(json.contains(r#""relativeError""#));
    }

    #[test]
    fn the_server_boots_and_serves_the_field() {
        let port = 18080u16;
        let adapter = Osarous::new("http://127.0.0.1:1").unwrap(); // unreachable but loopback
        let listener = TcpListener::bind(("127.0.0.1", port)).unwrap();
        std::thread::spawn(move || {
            for conn in listener.incoming().flatten() {
                let adapter = adapter.clone();
                std::thread::spawn(move || {
                    let _ = handle_connection(conn, &adapter);
                });
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(120));

        // The app itself is the client: loopback only, hand-rolled.
        let resp = crate::transport::request("http://127.0.0.1:18080/api/gaia?prompt=hello", "GET", None).unwrap();
        assert_eq!(resp.status, 200);
        let json = resp.body_utf8();
        assert!(json.contains("\"dominant\""));
        assert!(json.contains("\"affect\":{\"mood\":"));

        let rf_resp = crate::transport::request("http://127.0.0.1:18080/api/gaia?prompt=shinjuku&rf_entropy=3.4", "GET", None).unwrap();
        assert_eq!(rf_resp.status, 200);
        assert!(rf_resp.body_utf8().contains("\"coherence\""));

        let health = crate::transport::request("http://127.0.0.1:18080/api/health", "GET", None).unwrap();
        assert_eq!(health.status, 200);
        let health_json = health.body_utf8();
        assert!(health_json.contains("\"installed\":true"));
        assert!(health_json.contains("\"latencyMs\":"));

        let page = crate::transport::request("http://127.0.0.1:18080/", "GET", None).unwrap();
        assert_eq!(page.status, 200);
        assert!(page.body_utf8().contains("ALEXIAI"));

        let frames: Vec<_> = crate::transport::sse(
            "http://127.0.0.1:18080/api/chat",
            "POST",
            Some(r#"{"prompt":"hi","rf_entropy":2.418}"#),
        )
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
        // The model is down; the chat stream must still answer 200 with a GAIA frame (including layers) and structured error JSON.
        assert!(frames.iter().any(|f| f.event == "gaia" && f.data.contains("\"layers\":{\"gravity\":1")));
        assert!(frames.iter().any(|f| f.event == "error" && f.data.contains("{\"message\":")));
    }
}
