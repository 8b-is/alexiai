//! # DeepSeek Engram Conditional Memory on the GAIA-MLX-QUANT `p/q` + `BFP-64` Substrate
//!
//! Implements the **Engram** conditional memory architecture
//! (*"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"*,
//! `github.com/deepseek-ai/Engram`) in pure `#![no_std]`, `#![forbid(unsafe_code)]`, zero-allocation Rust.
//!
//! ## Why Engram + `BFP-64` (`i8` mantissa + `i16` SIMD accumulator) is a natural mathematical fit
//!
//! 1. **Exact 64-Dim Head Alignment (`D_head = 64 == BFP-64 block size`)**:
//!    In DeepSeek's Engram configuration (`n_embed_per_ngram = 512`, `n_head_per_ngram = 8`),
//!    each lookup head has embedding dimension `D_head = 512 / 8 = 64` — the **exact block size**
//!    of our `BFP-64` block-floating-point representation (`64 x i8` mantissas in `[-127, 127]` +
//!    `1 x i8` shared dyadic exponent = `65 bytes/row` instead of `256 bytes` FP32 or `512 bytes` FP64).
//!
//! 2. **Zero-Quantization O(1) Lookup**:
//!    When `MultiHeadEmbedding` retrieves `(max_ngram_size - 1) * n_head_per_ngram` heads
//!    (e.g., `16` heads x `64` dims = `1024`), the concatenated `engram_hidden_size` vector is
//!    **already partitioned into pre-quantized 64-element `BFP-64` blocks (`m_x`, `exp_x`)**!
//!    The 5 downstream linear projections (`key_projs[0..hc_mult]` and `value_proj`) execute
//!    directly on the retrieved `i8` mantissas using our 16-wide `i16` SIMD accumulator
//!    (`|S_64| <= 64 * 127 = 8,128 <= 32,767 = i16::MAX`) with **zero runtime activation
//!    quantization overhead**.
//!
//! 3. **Lemire `p/q` (`q = 2^64`) Fast Reciprocal Prime Hashing**:
//!    Replaces hardware 64-bit modulo (`mix % prime`) across all `(n, head)` prime buckets with
//!    precomputed 64-bit dyadic reciprocals (`inv_prime = floor((2^64 - 1) / prime) + 1`),
//!    turning multi-head prime hashing into pipelined 128-bit integer multiply-shifts.
//!
//! 4. **MEM|8 Adaptive Precision (`BFP-i8` -> `BFP-i16`) for High-Salience N-grams**:
//!    Frequent N-gram buckets remain in compact `BFP-i8` (`65 bytes/row`, `i16` SIMD accumulator),
//!    while high-salience or outlier N-gram heads promote to `BFP-i16` (`129 bytes/row`, `i32`
//!    SIMD accumulator).

use crate::mlx::{MLX_GROUP_SIZE, rms_norm_into};

/// Default Engram head dimension (`512 / 8 = 64`), identical to [`MLX_GROUP_SIZE`].
pub const ENGRAM_HEAD_DIM: usize = 64;

/// Default hyper-connection multiplicity (`hc_mult = 4`).
pub const ENGRAM_DEFAULT_HC_MULT: usize = 4;

/// Fixed-point `Q16` scale (`q = 2^16 = 65,536`) for the zero-float `p/q` Engram pipeline.
pub const ENGRAM_PQ_Q16: i64 = 1 << 16;

/// Precomputed Lemire `p/q` (`q = 2^64`) reciprocal for fast prime-bucket modulo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimeBucketMod {
    /// Prime bucket count `P`
    pub prime: u64,
    /// Precomputed reciprocal `ceil(2^64 / P)` for fast multiply-shift remainder
    pub inv_q64: u64,
    /// Cumulative row offset of this head inside `MultiHeadEmbedding`
    pub table_offset: usize,
}

impl PrimeBucketMod {
    /// Construct a [`PrimeBucketMod`] for prime modulus `prime` at `table_offset`.
    #[inline]
    pub const fn new(prime: u64, table_offset: usize) -> Self {
        let p = if prime == 0 { 1 } else { prime };
        let inv_q64 = (u64::MAX / p).wrapping_add(1);
        Self {
            prime: p,
            inv_q64,
            table_offset,
        }
    }

    /// Compute `x % self.prime` for `x < 2^32` using Lemire's `p/q` (`q = 2^64`) reciprocal
    /// without hardware division instructions.
    #[inline]
    pub const fn fast_rem_u64(&self, x: u64) -> usize {
        // For x < 2^32 and prime < 2^31, q_est = (x * inv_q64) >> 64 is within [q - 1, q + 1].
        let q_est = (((x as u128) * (self.inv_q64 as u128)) >> 64) as u64;
        let prod = q_est.wrapping_mul(self.prime);
        let mut rem = if x >= prod {
            x - prod
        } else {
            x.wrapping_add(self.prime) - prod
        };
        if rem >= self.prime {
            rem -= self.prime;
            if rem >= self.prime {
                rem %= self.prime;
            }
        }
        rem as usize
    }
}

/// Reference N-gram XOR-multiply hash + hardware `% prime` modulo over the
/// 32-bit-folded mix (`mix ^ (mix >> 32)`, matching the `use_pq_hash` path of
/// `engram_demo_v1.py`). Note: the upstream demo's default path hashes the
/// full 64-bit mix, so this kernel is bit-identical to the folded reference,
/// not to the un-folded Python baseline.
///
/// - `tokens`: token IDs of length `seq_len`
/// - `pad_id`: padding token ID for left-shifted context
/// - `multipliers`: odd integer multipliers `[max_ngram_size]`
/// - `buckets`: prime bucket descriptors `[(max_ngram_size - 1) * n_heads]`
/// - `out_global_indices`: output flat row indices into `MultiHeadEmbedding`,
///   shaped `[seq_len * num_total_heads]`
pub fn engram_hash_ngram_mod_into(
    tokens: &[u64],
    pad_id: u64,
    multipliers: &[u64],
    n_heads_per_ngram: usize,
    buckets: &[PrimeBucketMod],
    out_global_indices: &mut [usize],
) {
    let seq_len = tokens.len();
    let max_ngram = multipliers.len();
    if max_ngram < 2 || n_heads_per_ngram == 0 {
        return;
    }
    let total_heads = (max_ngram - 1) * n_heads_per_ngram;
    assert!(buckets.len() >= total_heads);
    assert!(out_global_indices.len() >= seq_len * total_heads);

    for t in 0..seq_len {
        let row_out = &mut out_global_indices[t * total_heads..(t + 1) * total_heads];
        let mut mix = tokens[t].wrapping_mul(multipliers[0]);

        for n in 2..=max_ngram {
            let k = n - 1;
            let tok_k = if t >= k { tokens[t - k] } else { pad_id };
            mix ^= tok_k.wrapping_mul(multipliers[k]);
            // Fold 64-bit mix into 32-bit unsigned mix to guarantee exact Lemire p/q equivalence
            let mix32 = (mix ^ (mix >> 32)) & 0xFFFF_FFFF;

            let ngram_idx = n - 2;
            let base_h = ngram_idx * n_heads_per_ngram;
            for h in 0..n_heads_per_ngram {
                let b = &buckets[base_h + h];
                row_out[base_h + h] = b.table_offset + ((mix32 % b.prime) as usize);
            }
        }
    }
}

/// Fast `p/q` (`q = 2^64`) reciprocal N-gram hash mapping (zero hardware division instructions in inner loop).
///
/// Produces bit-identical bucket indices to [`engram_hash_ngram_mod_into`].
pub fn engram_hash_ngram_pq_into(
    tokens: &[u64],
    pad_id: u64,
    multipliers: &[u64],
    n_heads_per_ngram: usize,
    buckets: &[PrimeBucketMod],
    out_global_indices: &mut [usize],
) {
    let seq_len = tokens.len();
    let max_ngram = multipliers.len();
    if max_ngram < 2 || n_heads_per_ngram == 0 {
        return;
    }
    let total_heads = (max_ngram - 1) * n_heads_per_ngram;
    assert!(buckets.len() >= total_heads);
    assert!(out_global_indices.len() >= seq_len * total_heads);

    for t in 0..seq_len {
        let row_out = &mut out_global_indices[t * total_heads..(t + 1) * total_heads];
        let mut mix = tokens[t].wrapping_mul(multipliers[0]);

        for n in 2..=max_ngram {
            let k = n - 1;
            let tok_k = if t >= k { tokens[t - k] } else { pad_id };
            mix ^= tok_k.wrapping_mul(multipliers[k]);
            let mix32 = (mix ^ (mix >> 32)) & 0xFFFF_FFFF;

            let ngram_idx = n - 2;
            let base_h = ngram_idx * n_heads_per_ngram;
            for h in 0..n_heads_per_ngram {
                let b = &buckets[base_h + h];
                row_out[base_h + h] = b.table_offset + b.fast_rem_u64(mix32);
            }
        }
    }
}

/// A single 64-dimensional Engram head embedding stored natively in **Omni `BFP-64`** format:
/// `64 x i8` mantissas in `[-127, 127]` + `1 x i8` shared power-of-two exponent (`65 bytes` total).
#[derive(Clone, Copy, Debug)]
pub struct Bfp64HeadRow {
    pub mantissas: [i8; ENGRAM_HEAD_DIM],
    pub exponent: i8,
}

impl Bfp64HeadRow {
    /// Quantize a 64-element `f64` head embedding into a native `BFP-64` row (`65 bytes`).
    pub fn from_f64_slice(src: &[f64]) -> Self {
        assert!(src.len() >= ENGRAM_HEAD_DIM);
        let mut max_abs = 0.0f64;
        for &v in &src[..ENGRAM_HEAD_DIM] {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs < 1e-18 {
            return Self {
                mantissas: [0i8; ENGRAM_HEAD_DIM],
                exponent: -64,
            };
        }
        let ratio = max_abs / 127.0;
        let bits = ratio.to_bits();
        let raw_exp = (((bits >> 52) & 0x7FF) as i32) - 1023;
        let mant = bits & 0x000F_FFFF_FFFF_FFFF;
        let e = if mant == 0 { raw_exp } else { raw_exp + 1 }.clamp(-60, 60) as i8;
        let inv_scale = libm_exp2_neg(e);

        let mut mantissas = [0i8; ENGRAM_HEAD_DIM];
        for (dst, &v) in mantissas.iter_mut().zip(src[..ENGRAM_HEAD_DIM].iter()) {
            let scaled = v * inv_scale;
            let rounded = if scaled >= 0.0 {
                (scaled + 0.5) as i32
            } else {
                (scaled - 0.5) as i32
            };
            *dst = rounded.clamp(-127, 127) as i8;
        }
        Self {
            mantissas,
            exponent: e,
        }
    }
}

/// Exact `2^(-e)` using IEEE-754 exponent bit construction (`no_std` safe).
#[inline]
const fn libm_exp2_neg(e: i8) -> f64 {
    let biased = (1023i64 - (e as i64)) as u64;
    f64::from_bits(biased << 52)
}

/// Exact `2^e` using IEEE-754 exponent bit construction (`no_std` safe).
#[inline]
const fn libm_exp2_pos(e: i8) -> f64 {
    let biased = (1023i64 + (e as i64)) as u64;
    f64::from_bits(biased << 52)
}

/// **MEM|8 Adaptive Precision 64-Dim Engram Head Row (`Mem8AdaptiveHeadRow`)**:
///
/// Embodies Omni's principle: *"Precision should be a property of the information's
/// requirements, not a fixed tax paid by every operation."*
/// - Common syntactic N-gram heads whose relative `i8` reconstruction error is `<= tau`
///   stay in compact **`BFP-i8`** (`65 bytes/row`, `i16` SIMD accumulator).
/// - High-salience or outlier factual N-gram heads (e.g., `"Bucephalus"`) whose error
///   exceeds `tau` automatically promote to **`BFP-i16`** (`129 bytes/row`, `i32` SIMD accumulator).
#[derive(Clone, Copy, Debug)]
pub struct Mem8AdaptiveHeadRow {
    pub mantissas_i8: [i8; ENGRAM_HEAD_DIM],
    pub mantissas_i16: [i16; ENGRAM_HEAD_DIM],
    pub exponent: i8,
    pub promoted_i16: bool,
}

impl Mem8AdaptiveHeadRow {
    /// Quantize a 64-element `f64` head embedding with adaptive promotion threshold `tau`.
    pub fn from_f64_slice(src: &[f64], tau: f64) -> Self {
        assert!(src.len() >= ENGRAM_HEAD_DIM);
        let bfp8 = Bfp64HeadRow::from_f64_slice(src);
        let scale8 = libm_exp2_pos(bfp8.exponent);

        let mut err_sq = 0.0f64;
        let mut sig_sq = 0.0f64;
        for (k, &v) in src[..ENGRAM_HEAD_DIM].iter().enumerate() {
            let recon = (bfp8.mantissas[k] as f64) * scale8;
            let diff = v - recon;
            err_sq += diff * diff;
            sig_sq += v * v;
        }
        let rel_err = libm_sqrt(err_sq / if sig_sq > 1e-24 { sig_sq } else { 1e-24 });

        if rel_err <= tau {
            return Self {
                mantissas_i8: bfp8.mantissas,
                mantissas_i16: [0i16; ENGRAM_HEAD_DIM],
                exponent: bfp8.exponent,
                promoted_i16: false,
            };
        }

        // Promote to BFP-i16 ([-32767, 32767] + shared dyadic exponent)
        let mut max_abs = 0.0f64;
        for &v in &src[..ENGRAM_HEAD_DIM] {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let ratio16 = max_abs / 32767.0;
        let bits16 = ratio16.to_bits();
        let raw_exp16 = (((bits16 >> 52) & 0x7FF) as i32) - 1023;
        let mant16 = bits16 & 0x000F_FFFF_FFFF_FFFF;
        let e16 = if mant16 == 0 { raw_exp16 } else { raw_exp16 + 1 }.clamp(-60, 60) as i8;
        let inv_scale16 = libm_exp2_neg(e16);

        let mut mantissas_i16 = [0i16; ENGRAM_HEAD_DIM];
        for (dst, &v) in mantissas_i16.iter_mut().zip(src[..ENGRAM_HEAD_DIM].iter()) {
            let scaled = v * inv_scale16;
            let rounded = if scaled >= 0.0 {
                (scaled + 0.5) as i32
            } else {
                (scaled - 0.5) as i32
            };
            *dst = rounded.clamp(-32767, 32767) as i16;
        }

        Self {
            mantissas_i8: bfp8.mantissas,
            mantissas_i16,
            exponent: e16,
            promoted_i16: true,
        }
    }
}

/// Project `num_heads` retrieved [`Mem8AdaptiveHeadRow`] blocks directly against a ternary
/// `[out_features, in_features]` matrix using **`i16` SIMD accumulation** for `BFP-i8` heads
/// and **`i32` SIMD accumulation** for promoted `BFP-i16` heads!
pub fn engram_mem8_adaptive_matmul_into(
    table: &[Mem8AdaptiveHeadRow],
    head_indices: &[usize],
    w_signs: &[i8],
    group_scales: &[f64],
    in_features: usize,
    out_features: usize,
    out: &mut [f64],
) {
    let num_groups = in_features / MLX_GROUP_SIZE;
    assert_eq!(head_indices.len(), num_groups);
    assert!(
        num_groups <= 64,
        "num_groups ({}) exceeds the 64-group BFP-64 alignment limit",
        num_groups
    );

    let mut exp2_cache = [0.0f64; 64];
    for (g, &idx) in head_indices.iter().enumerate().take(64) {
        exp2_cache[g] = libm_exp2_pos(table[idx].exponent);
    }

    for j in 0..out_features {
        let w_row = &w_signs[j * in_features..(j + 1) * in_features];
        let s_row = &group_scales[j * num_groups..(j + 1) * num_groups];
        let mut total = 0.0f64;

        for (g, &row_idx) in head_indices.iter().enumerate() {
            let head = &table[row_idx];
            let mw_blk = &w_row[g * MLX_GROUP_SIZE..(g + 1) * MLX_GROUP_SIZE];

            if head.promoted_i16 {
                // Promoted high-salience head: i32 SIMD accumulator (|S_64| <= 64 * 32767 = 2,097,088)
                let block_sum_i32: i32 = mw_blk
                    .iter()
                    .zip(head.mantissas_i16.iter())
                    .map(|(&w, &m)| (w as i32).wrapping_mul(m as i32))
                    .fold(0i32, i32::wrapping_add);
                total += (block_sum_i32 as f64) * exp2_cache[g] * s_row[g];
            } else {
                // Compact fast-path head: i16 SIMD accumulator (|S_64| <= 64 * 127 = 8,128)
                let block_sum_i16: i16 = mw_blk
                    .iter()
                    .zip(head.mantissas_i8.iter())
                    .map(|(&w, &m)| (w as i16).wrapping_mul(m as i16))
                    .fold(0i16, i16::wrapping_add);
                total += (block_sum_i16 as f64) * exp2_cache[g] * s_row[g];
            }
        }
        out[j] = total;
    }
}

/// Zero-quantization `O(1)` gather from a `BFP-64` `MultiHeadEmbedding` table into
/// pre-quantized `(m_x, exp_x)` buffers ready for `i16` SIMD ternary projection!
///
/// Because each head in Engram has dimension `D_head = 64`, gathering `num_heads` heads
/// produces `num_heads` contiguous 64-byte `i8` blocks and `num_heads` exponents with
/// **zero floating-point operations**.
#[inline]
pub fn engram_gather_bfp64_heads_into(
    table: &[Bfp64HeadRow],
    head_indices: &[usize],
    out_m_x: &mut [i8],
    out_exp_x: &mut [i8],
) {
    let num_heads = head_indices.len();
    assert!(out_m_x.len() >= num_heads * ENGRAM_HEAD_DIM);
    assert!(out_exp_x.len() >= num_heads);

    for (h, &row_idx) in head_indices.iter().enumerate() {
        let row = &table[row_idx];
        let dst_start = h * ENGRAM_HEAD_DIM;
        out_m_x[dst_start..dst_start + ENGRAM_HEAD_DIM].copy_from_slice(&row.mantissas);
        out_exp_x[h] = row.exponent;
    }
}

/// Project pre-quantized `BFP-64` Engram heads `(m_x, exp_x)` directly against a ternary
/// `[out_features, in_features]` weight matrix (`w_signs in {-1, 0, +1}`) using
/// **Omni's 16-wide `i16` SIMD block accumulator (`|S_64| <= 64 * 127 = 8,128 <= i16::MAX`)**!
///
/// Unlike general activations, retrieved Engram heads are *already* quantized into 64-element
/// `BFP-64` blocks in the lookup table, so this kernel skips activation quantization completely!
pub fn engram_bfp64_prequant_matmul_into(
    m_x: &[i8],
    exp_x: &[i8],
    w_signs: &[i8],
    group_scales: &[f64],
    in_features: usize,
    out_features: usize,
    out: &mut [f64],
) {
    let num_groups = in_features / MLX_GROUP_SIZE;
    assert_eq!(in_features % MLX_GROUP_SIZE, 0);
    assert!(m_x.len() >= in_features);
    assert!(exp_x.len() >= num_groups);
    assert!(w_signs.len() >= out_features * in_features);
    assert!(group_scales.len() >= out_features * num_groups);
    assert!(out.len() >= out_features);
    assert!(
        num_groups <= 64,
        "num_groups ({}) exceeds the 64-group BFP-64 alignment limit",
        num_groups
    );

    // Precompute 2^(exp_x[g]) once per head block (only 16 heads for 1024-dim Engram!)
    let mut exp2_cache = [0.0f64; 64];
    for g in 0..num_groups.min(64) {
        exp2_cache[g] = libm_exp2_pos(exp_x[g]);
    }

    for j in 0..out_features {
        let w_row = &w_signs[j * in_features..(j + 1) * in_features];
        let s_row = &group_scales[j * num_groups..(j + 1) * num_groups];
        let mut total = 0.0f64;

        for g in 0..num_groups {
            let base = g * MLX_GROUP_SIZE;
            let mx_blk = &m_x[base..base + MLX_GROUP_SIZE];
            let mw_blk = &w_row[base..base + MLX_GROUP_SIZE];

            // 16-wide i16 SIMD accumulation over 64 elements via wrapping_mul + wrapping_add
            // Worst-case magnitude: 64 * 127 = 8,128 <= 32,767 (i16::MAX)
            let block_sum: i16 = mw_blk
                .iter()
                .zip(mx_blk.iter())
                .map(|(&w, &m)| (w as i16).wrapping_mul(m as i16))
                .fold(0i16, i16::wrapping_add);
            total += (block_sum as f64) * exp2_cache[g] * s_row[g];
        }
        out[j] = total;
    }
}

/// **Deferred-Scale Integer-Aligned `BFP-64` Engram Projection (`engram_bfp64_int_aligned_matmul_into`)**:
///
/// Solves Omni's block-exponent alignment challenge:
/// Instead of converting each 64-element `i16` block sum to `f64` inside the group loop,
/// this kernel pre-aligns the 16 head exponents `2^(exp_x[g])` with integer group scales
/// `p_{s,g}` in `p/q` (`q = 2^20`), accumulates all 16 heads in **pure `i64` integer arithmetic**,
/// and performs **only one `i64 -> f64` conversion per 1024-dim output row**!
pub fn engram_bfp64_int_aligned_matmul_into(
    m_x: &[i8],
    exp_x: &[i8],
    w_signs: &[i8],
    group_scales_pq: &[i32], // [out_features * num_groups] scaled by q_s = 2^20
    in_features: usize,
    out_features: usize,
    out: &mut [f64],
) {
    let num_groups = in_features / MLX_GROUP_SIZE;
    assert_eq!(in_features % MLX_GROUP_SIZE, 0);
    assert!(m_x.len() >= in_features);
    assert!(exp_x.len() >= num_groups);
    assert!(w_signs.len() >= out_features * in_features);
    assert!(group_scales_pq.len() >= out_features * num_groups);
    assert!(out.len() >= out_features);
    assert!(
        num_groups <= 64,
        "num_groups ({}) exceeds the 64-group BFP-64 alignment limit",
        num_groups
    );

    // Align the 16 head exponents relative to min_exp so every head has an exact integer shift `2^(exp_x[g] - min_exp)`
    let mut min_exp = i8::MAX;
    for &e in &exp_x[..num_groups] {
        if e < min_exp {
            min_exp = e;
        }
    }
    let mut head_mult = [1i64; 64];
    for g in 0..num_groups.min(64) {
        let diff = ((exp_x[g] as i32) - (min_exp as i32)).clamp(0, 24) as u32;
        head_mult[g] = 1i64 << diff;
    }
    // Single boundary scale per output neuron: 2^(min_exp) / 2^20 = 2^(min_exp - 20)
    let boundary_scale = libm_exp2_pos(min_exp.saturating_sub(20));

    for j in 0..out_features {
        let w_row = &w_signs[j * in_features..(j + 1) * in_features];
        let ps_row = &group_scales_pq[j * num_groups..(j + 1) * num_groups];
        let mut row_i64: i64 = 0;

        for g in 0..num_groups {
            let base = g * MLX_GROUP_SIZE;
            let mx_blk = &m_x[base..base + MLX_GROUP_SIZE];
            let mw_blk = &w_row[base..base + MLX_GROUP_SIZE];

            // 16-wide i16 SIMD accumulation over 64 elements (|block_sum| <= 8,128)
            let block_sum: i16 = mw_blk
                .iter()
                .zip(mx_blk.iter())
                .map(|(&w, &m)| (w as i16).wrapping_mul(m as i16))
                .fold(0i16, i16::wrapping_add);

            // Pure integer cross-block alignment & accumulation!
            row_i64 += (block_sum as i64) * (ps_row[g] as i64) * head_mult[g];
        }
        out[j] = (row_i64 as f64) * boundary_scale;
    }
}

/// **Pure-Integer `Q16` Deferred-Scale `BFP-64` Engram Projection (`engram_bfp64_int_q16_matmul_into`)**:
///
/// Accumulates all `num_groups` 64-dim heads in `i16 -> i64` and shifts directly into
/// `Q16` (`2^16`) fixed-point integers (`out_q16: &mut [i32]`) with **zero floating-point ops**.
pub fn engram_bfp64_int_q16_matmul_into(
    m_x: &[i8],
    exp_x: &[i8],
    w_signs: &[i8],
    group_scales_pq: &[i32], // [out_features * num_groups] scaled by q_s = 2^20
    in_features: usize,
    out_features: usize,
    out_q16: &mut [i32],
) {
    let num_groups = in_features / MLX_GROUP_SIZE;
    assert!(
        num_groups <= 64,
        "num_groups ({}) exceeds the 64-group BFP-64 alignment limit",
        num_groups
    );
    let mut min_exp = i8::MAX;
    for &e in &exp_x[..num_groups] {
        if e < min_exp {
            min_exp = e;
        }
    }
    let mut head_mult = [1i64; 64];
    for g in 0..num_groups.min(64) {
        let diff = ((exp_x[g] as i32) - (min_exp as i32)).clamp(0, 20) as u32;
        head_mult[g] = 1i64 << diff;
    }
    let shift_to_q16 = (min_exp as i32) - 4;

    for j in 0..out_features {
        let w_row = &w_signs[j * in_features..(j + 1) * in_features];
        let ps_row = &group_scales_pq[j * num_groups..(j + 1) * num_groups];
        let mut row_i64: i64 = 0;

        for g in 0..num_groups {
            let base = g * MLX_GROUP_SIZE;
            let mx_blk = &m_x[base..base + MLX_GROUP_SIZE];
            let mw_blk = &w_row[base..base + MLX_GROUP_SIZE];

            let block_sum: i16 = mw_blk
                .iter()
                .zip(mx_blk.iter())
                .map(|(&w, &m)| (w as i16).wrapping_mul(m as i16))
                .fold(0i16, i16::wrapping_add);

            row_i64 += (block_sum as i64) * (ps_row[g] as i64) * head_mult[g];
        }

        let val_q16 = if shift_to_q16 >= 0 {
            row_i64 << (shift_to_q16.min(30) as u32)
        } else {
            let rshift = (-shift_to_q16).min(60) as u32;
            let half = 1i64 << (rshift.saturating_sub(1));
            if row_i64 >= 0 {
                (row_i64 + half) >> rshift
            } else {
                -(((-row_i64) + half) >> rshift)
            }
        };
        out_q16[j] = val_q16.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
}

/// **Fused Single-Pass Bilinear RMSNorm Gate (`engram_fused_rmsnorm_gate`)**:
///
/// Computes `g = <RMSNorm(k), RMSNorm(q)> / sqrt(D)` in a **single fused pass** over `k` and `q`
/// without allocating or writing intermediate normalized `scratch_k` / `scratch_q` vectors!
#[inline]
pub fn engram_fused_rmsnorm_gate(
    key: &[f64],
    query: &[f64],
    norm_k_weight: &[f64],
    norm_q_weight: &[f64],
    eps: f64,
) -> f64 {
    let d = key.len();
    assert_eq!(query.len(), d);
    let mut sum_k2 = 0.0f64;
    let mut sum_q2 = 0.0f64;
    let mut weighted_dot = 0.0f64;

    for i in 0..d {
        let ki = key[i];
        let qi = query[i];
        sum_k2 += ki * ki;
        sum_q2 += qi * qi;
        weighted_dot += (norm_k_weight[i] * norm_q_weight[i]) * (ki * qi);
    }

    let d_f64 = d as f64;
    let rms_k = libm_sqrt(sum_k2 / d_f64 + eps);
    let rms_q = libm_sqrt(sum_q2 / d_f64 + eps);
    let g = weighted_dot / (libm_sqrt(d_f64) * rms_k * rms_q);
    let abs_g = if g.abs() < 1e-6 { 1e-6 } else { g.abs() };
    let signed_sqrt = if g >= 0.0 {
        libm_sqrt(abs_g)
    } else {
        -libm_sqrt(abs_g)
    };
    1.0 / (1.0 + libm_exp(-signed_sqrt))
}

/// Pure-Integer `Q16` Fused Bilinear RMSNorm Gate (`engram_fused_rmsnorm_gate_q16`):
///
/// Computes the hyper-connection gate in `Q16` (`[0, 65536]`) using exact integer
/// Newton-Raphson `isqrt_u128` and rational Padé `[2/2]` sigmoid with **zero floats**.
#[inline]
pub fn engram_fused_rmsnorm_gate_q16(
    key_q16: &[i32],
    query_q16: &[i32],
    norm_k_w_q16: &[i32],
    norm_q_w_q16: &[i32],
) -> i32 {
    let d = key_q16.len();
    let mut sum_k2: u128 = 0;
    let mut sum_q2: u128 = 0;
    let mut weighted_dot_q32: i128 = 0;

    for i in 0..d {
        let k = key_q16[i] as i64;
        let q = query_q16[i] as i64;
        sum_k2 += (k * k) as u128;
        sum_q2 += (q * q) as u128;
        let w_prod_q16 = ((norm_k_w_q16[i] as i64) * (norm_q_w_q16[i] as i64)) >> 16;
        let kq_q16 = (k * q) >> 16;
        weighted_dot_q32 += (w_prod_q16 as i128) * (kq_q16 as i128);
    }

    let eps_term = (d as u128) * 42_950;
    let norm_k_q16 = engram_isqrt_u128(sum_k2 + eps_term).max(1) as i128;
    let norm_q_q16 = engram_isqrt_u128(sum_q2 + eps_term).max(1) as i128;
    let sqrt_d_q16 = engram_isqrt_u128((d as u128) << 32).max(1) as i128;

    let denom_q32 = (norm_k_q16 * norm_q_q16).max(1);
    let g_q16 = ((weighted_dot_q32 * sqrt_d_q16) / denom_q32) as i64;

    let abs_g_q16 = g_q16.unsigned_abs().max(1);
    let sqrt_abs_q16 = engram_isqrt_u128((abs_g_q16 as u128) << 16) as i64;
    let signed_sqrt_q16 = if g_q16 >= 0 {
        sqrt_abs_q16
    } else {
        -sqrt_abs_q16
    };

    engram_sigmoid_pade_q16(signed_sqrt_q16)
}

/// Rational Padé `[2/2]` Sigmoid in `Q16` (`1 / (1 + exp(-x))`), zero floats!
#[inline]
pub const fn engram_sigmoid_pade_q16(x_q16: i64) -> i32 {
    let x = if x_q16 < -(6 * ENGRAM_PQ_Q16) {
        -(6 * ENGRAM_PQ_Q16)
    } else if x_q16 > (6 * ENGRAM_PQ_Q16) {
        6 * ENGRAM_PQ_Q16
    } else {
        x_q16
    };
    let x2_q16 = (x * x) >> 16;
    let num = 12 * ENGRAM_PQ_Q16 + 6 * x + x2_q16;
    let raw_den = 24 * ENGRAM_PQ_Q16 + 2 * x2_q16;
    let den = if raw_den < 1 { 1 } else { raw_den };
    let sig_q16 = ((num << 16) + (den >> 1)) / den;
    if sig_q16 < 0 {
        0
    } else if sig_q16 > ENGRAM_PQ_Q16 {
        ENGRAM_PQ_Q16 as i32
    } else {
        sig_q16 as i32
    }
}

/// **100% Zero-Float Pure-Integer `p/q` ("Counting Divided") Engram Layer Step (`engram_step_pq_no_floats_into`)**:
///
/// Executes the entire DeepSeek Engram layer step (`MultiHeadEmbedding` gather, `value_proj`,
/// `key_projs[0..hc_mult]`, Fused Bilinear `RMSNorm` gate via `isqrt_u128`, signed-sqrt,
/// rational Padé `[2/2]` sigmoid, ShortConv SiLU, and residual) in pure integer arithmetic
/// with **zero IEEE-754 floating-point instructions**!
#[allow(clippy::too_many_arguments)]
pub fn engram_step_pq_no_floats_into(
    table: &[Bfp64HeadRow],
    head_indices: &[usize],
    hidden_states_q16: &[i32],    // [hc_mult * hidden_size] in Q16
    key_projs_signs: &[i8],       // [hc_mult * hidden_size * engram_dim]
    key_projs_scales_pq: &[i32],  // [hc_mult * hidden_size * num_groups] in Q20
    val_proj_signs: &[i8],        // [hidden_size * engram_dim]
    val_proj_scales_pq: &[i32],   // [hidden_size * num_groups] in Q20
    norm_w_q16: &[i32],           // [hidden_size] in Q16
    short_conv_w_q16: &[i32],     // [hc_mult * hidden_size] in Q16
    hc_mult: usize,
    engram_dim: usize,
    hidden_size: usize,
    m_x_scratch: &mut [i8],       // [engram_dim]
    exp_x_scratch: &mut [i8],     // [engram_dim / 64]
    val_q16_scratch: &mut [i32],  // [hidden_size]
    key_q16_scratch: &mut [i32],  // [hidden_size]
    out_hc_q16: &mut [i32],       // [hc_mult * hidden_size] in Q16
) {
    let num_groups = engram_dim / MLX_GROUP_SIZE;

    engram_gather_bfp64_heads_into(table, head_indices, m_x_scratch, exp_x_scratch);

    engram_bfp64_int_q16_matmul_into(
        m_x_scratch,
        exp_x_scratch,
        val_proj_signs,
        val_proj_scales_pq,
        engram_dim,
        hidden_size,
        val_q16_scratch,
    );

    let kp_mat_stride = hidden_size * engram_dim;
    let kp_scale_stride = hidden_size * num_groups;

    for hc in 0..hc_mult {
        let k_signs = &key_projs_signs[hc * kp_mat_stride..(hc + 1) * kp_mat_stride];
        let k_scales_pq = &key_projs_scales_pq[hc * kp_scale_stride..(hc + 1) * kp_scale_stride];
        let query_q16 = &hidden_states_q16[hc * hidden_size..(hc + 1) * hidden_size];

        engram_bfp64_int_q16_matmul_into(
            m_x_scratch,
            exp_x_scratch,
            k_signs,
            k_scales_pq,
            engram_dim,
            hidden_size,
            key_q16_scratch,
        );

        let gate_q16 = engram_fused_rmsnorm_gate_q16(
            key_q16_scratch,
            query_q16,
            norm_w_q16,
            norm_w_q16,
        ) as i64;

        let conv_hc_q16 = &short_conv_w_q16[hc * hidden_size..(hc + 1) * hidden_size];
        let out_slice = &mut out_hc_q16[hc * hidden_size..(hc + 1) * hidden_size];

        for d in 0..hidden_size {
            let v_gated_q16 = (gate_q16 * (val_q16_scratch[d] as i64)) >> 16;
            let conv_val_q16 = (v_gated_q16 * (conv_hc_q16[d] as i64)) >> 16;
            let sig_conv_q16 = engram_sigmoid_pade_q16(conv_val_q16) as i64;
            let silu_q16 = (conv_val_q16 * sig_conv_q16) >> 16;
            out_slice[d] = (query_q16[d] as i64 + v_gated_q16 + silu_q16) as i32;
        }
    }
}

#[inline]
const fn engram_isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let shift = (128 - n.leading_zeros()).div_ceil(2);
    let mut x = 1u128 << shift;
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Compute DeepSeek Engram's hyper-connection gate for stream `hc`:
///
/// $$g = \frac{\langle \text{RMSNorm}(k), \text{RMSNorm}(q) \rangle}{\sqrt{D}}, \qquad
///   \text{gate} = \sigma\left(\text{sign}(g)\sqrt{\max(|g|, 10^{-6})}\right)$$
pub fn engram_hc_gate(
    key: &[f64],
    query: &[f64],
    norm_k_weight: &[f64],
    norm_q_weight: &[f64],
    eps: f64,
    scratch_k: &mut [f64],
    scratch_q: &mut [f64],
) -> f64 {
    let d = key.len();
    assert_eq!(query.len(), d);
    rms_norm_into(key, Some(norm_k_weight), eps, &mut scratch_k[..d]);
    rms_norm_into(query, Some(norm_q_weight), eps, &mut scratch_q[..d]);

    let mut dot = 0.0f64;
    for i in 0..d {
        dot += scratch_k[i] * scratch_q[i];
    }
    let inv_sqrt_d = 1.0 / libm_sqrt(d as f64);
    let g = dot * inv_sqrt_d;
    let abs_g = if g.abs() < 1e-6 { 1e-6 } else { g.abs() };
    let signed_sqrt = if g >= 0.0 {
        libm_sqrt(abs_g)
    } else {
        -libm_sqrt(abs_g)
    };
    // Rational Padé / standard sigmoid 1 / (1 + exp(-x))
    1.0 / (1.0 + libm_exp(-signed_sqrt))
}

/// Execute a complete single-token **DeepSeek Engram Conditional Memory Step** using
/// **Omni `BFP-64` (`i8` head table + `i16` SIMD accumulator)**!
///
/// Flow (mirroring `Engram.forward` in `engram_demo_v1.py`):
/// 1. Gather `num_heads` (`16`) pre-quantized `BFP-64` head embeddings (`16 x 64 = 1024`) from `table`.
/// 2. Project `value = value_proj(embeddings)` using `engram_bfp64_prequant_matmul_into` (`i16` SIMD).
/// 3. For each hyper-connection stream `hc in 0..hc_mult`:
///    - Project `key_hc = key_projs[hc](embeddings)` using `engram_bfp64_prequant_matmul_into` (`i16` SIMD),
///    - Compute `gate_hc = engram_hc_gate(key_hc, query_hc)`,
///    - Fuse `gated_value[hc, d] = gate_hc * value[d]` + depthwise ShortConv tap + residual `hidden_states`.
#[allow(clippy::too_many_arguments)]
pub fn engram_step_bfp64_into(
    table: &[Bfp64HeadRow],
    head_indices: &[usize],
    hidden_states_hc: &[f64], // [hc_mult * hidden_size]
    key_projs_signs: &[i8],   // [hc_mult * hidden_size * engram_dim]
    key_projs_scales: &[f64], // [hc_mult * hidden_size * num_groups]
    val_proj_signs: &[i8],    // [hidden_size * engram_dim]
    val_proj_scales: &[f64],  // [hidden_size * num_groups]
    norm_w: &[f64],           // [hidden_size]
    short_conv_w: &[f64],     // [hc_mult * hidden_size] depthwise current-step weight
    hc_mult: usize,
    engram_dim: usize,
    hidden_size: usize,
    m_x_scratch: &mut [i8],   // [engram_dim]
    exp_x_scratch: &mut [i8], // [engram_dim / 64]
    val_scratch: &mut [f64],  // [hidden_size]
    key_scratch: &mut [f64],  // [hidden_size]
    nk_scratch: &mut [f64],   // [hidden_size]
    nq_scratch: &mut [f64],   // [hidden_size]
    out_hc: &mut [f64],       // [hc_mult * hidden_size]
) {
    let num_groups = engram_dim / MLX_GROUP_SIZE;

    // 1. O(1) gather of pre-quantized BFP-64 heads (zero float ops!)
    engram_gather_bfp64_heads_into(table, head_indices, m_x_scratch, exp_x_scratch);

    // 2. Project value_proj(embeddings) via i16 SIMD accumulator
    engram_bfp64_prequant_matmul_into(
        m_x_scratch,
        exp_x_scratch,
        val_proj_signs,
        val_proj_scales,
        engram_dim,
        hidden_size,
        val_scratch,
    );

    // 3. Per-stream key projection, RMSNorm gating, ShortConv SiLU, and residual addition
    let kp_mat_stride = hidden_size * engram_dim;
    let kp_scale_stride = hidden_size * num_groups;

    for hc in 0..hc_mult {
        let k_signs = &key_projs_signs[hc * kp_mat_stride..(hc + 1) * kp_mat_stride];
        let k_scales = &key_projs_scales[hc * kp_scale_stride..(hc + 1) * kp_scale_stride];
        let query_hc = &hidden_states_hc[hc * hidden_size..(hc + 1) * hidden_size];

        engram_bfp64_prequant_matmul_into(
            m_x_scratch,
            exp_x_scratch,
            k_signs,
            k_scales,
            engram_dim,
            hidden_size,
            key_scratch,
        );

        let gate = engram_hc_gate(
            key_scratch,
            query_hc,
            norm_w,
            norm_w,
            1e-5,
            nk_scratch,
            nq_scratch,
        );

        let conv_hc = &short_conv_w[hc * hidden_size..(hc + 1) * hidden_size];
        let out_slice = &mut out_hc[hc * hidden_size..(hc + 1) * hidden_size];

        for d in 0..hidden_size {
            let v_gated = gate * val_scratch[d];
            // Depthwise ShortConv + SiLU(x) = x / (1 + exp(-x))
            let conv_val = v_gated * conv_hc[d];
            let silu = conv_val / (1.0 + libm_exp(-conv_val));
            out_slice[d] = query_hc[d] + v_gated + silu;
        }
    }
}

/// Reference dense FP64 Engram step (mirrors `Engram.forward` in `engram_demo_v1.py`).
#[allow(clippy::too_many_arguments)]
pub fn engram_step_dense_fp64_into(
    dense_table: &[f64], // [total_rows * 64]
    head_indices: &[usize],
    hidden_states_hc: &[f64], // [hc_mult * hidden_size]
    key_projs_dense: &[f64],  // [hc_mult * hidden_size * engram_dim]
    val_proj_dense: &[f64],   // [hidden_size * engram_dim]
    norm_w: &[f64],
    short_conv_w: &[f64],
    hc_mult: usize,
    engram_dim: usize,
    hidden_size: usize,
    emb_scratch: &mut [f64], // [engram_dim]
    val_scratch: &mut [f64], // [hidden_size]
    key_scratch: &mut [f64], // [hidden_size]
    nk_scratch: &mut [f64],  // [hidden_size]
    nq_scratch: &mut [f64],  // [hidden_size]
    out_hc: &mut [f64],      // [hc_mult * hidden_size]
) {
    // 1. Gather FP64 head embeddings
    for (h, &row_idx) in head_indices.iter().enumerate() {
        let src = &dense_table[row_idx * ENGRAM_HEAD_DIM..(row_idx + 1) * ENGRAM_HEAD_DIM];
        emb_scratch[h * ENGRAM_HEAD_DIM..(h + 1) * ENGRAM_HEAD_DIM].copy_from_slice(src);
    }

    // 2. Dense value_proj
    for j in 0..hidden_size {
        let w_row = &val_proj_dense[j * engram_dim..(j + 1) * engram_dim];
        let mut sum = 0.0f64;
        for i in 0..engram_dim {
            sum += emb_scratch[i] * w_row[i];
        }
        val_scratch[j] = sum;
    }

    // 3. Per-stream dense key_projs + gate + ShortConv
    let kp_stride = hidden_size * engram_dim;
    for hc in 0..hc_mult {
        let k_mat = &key_projs_dense[hc * kp_stride..(hc + 1) * kp_stride];
        let query_hc = &hidden_states_hc[hc * hidden_size..(hc + 1) * hidden_size];
        for j in 0..hidden_size {
            let w_row = &k_mat[j * engram_dim..(j + 1) * engram_dim];
            let mut sum = 0.0f64;
            for i in 0..engram_dim {
                sum += emb_scratch[i] * w_row[i];
            }
            key_scratch[j] = sum;
        }

        let gate = engram_hc_gate(
            key_scratch,
            query_hc,
            norm_w,
            norm_w,
            1e-5,
            nk_scratch,
            nq_scratch,
        );

        let conv_hc = &short_conv_w[hc * hidden_size..(hc + 1) * hidden_size];
        let out_slice = &mut out_hc[hc * hidden_size..(hc + 1) * hidden_size];
        for d in 0..hidden_size {
            let v_gated = gate * val_scratch[d];
            let conv_val = v_gated * conv_hc[d];
            let silu = conv_val / (1.0 + libm_exp(-conv_val));
            out_slice[d] = query_hc[d] + v_gated + silu;
        }
    }
}

/// Pure `core` Newton-Raphson square root (`no_std` compatible).
#[inline]
fn libm_sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = f64::from_bits((x.to_bits() >> 1) + (1023u64 << 51));
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y
}

/// Pure `core` Padé `[4/4]` + range-reduced exponential `exp(x)` (`no_std` compatible).
#[inline]
fn libm_exp(x: f64) -> f64 {
    let x_clamped = x.clamp(-40.0, 40.0);
    const LN2: f64 = core::f64::consts::LN_2;
    const INV_LN2: f64 = core::f64::consts::LOG2_E;
    let k = (x_clamped * INV_LN2 + if x_clamped >= 0.0 { 0.5 } else { -0.5 }) as i32;
    let r = x_clamped - (k as f64) * LN2;
    let r2 = r * r;
    let p = 1.0 + 0.5 * r + (3.0 / 28.0) * r2 + (1.0 / 84.0) * r2 * r + (1.0 / 1680.0) * r2 * r2;
    let q = 1.0 - 0.5 * r + (3.0 / 28.0) * r2 - (1.0 / 84.0) * r2 * r + (1.0 / 1680.0) * r2 * r2;
    let biased = ((1023 + k).clamp(1, 2046) as u64) << 52;
    (p / q) * f64::from_bits(biased)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn test_engram_pq_hash_and_bfp64_layer_benchmark() {
        // Mirror DeepSeek Engram demo configuration:
        // 16 heads (2-gram and 3-gram x 8 heads each), D_head = 64 -> engram_dim = 1024,
        // hidden_size = 512 (or 1024), hc_mult = 4 (5 linear projections per token!)
        let max_ngram_size = 3;
        let n_heads_per_ngram = 8;
        let total_heads = (max_ngram_size - 1) * n_heads_per_ngram; // 16 heads
        let engram_dim = total_heads * ENGRAM_HEAD_DIM; // 16 * 64 = 1024
        let hidden_size = 512;
        let hc_mult = ENGRAM_DEFAULT_HC_MULT; // 4

        // Distinct primes for the 16 heads (matching NgramHashMapping prime bucket style)
        let primes: [u64; 16] = [
            1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
            1093, 1097,
        ];
        let mut buckets = Vec::with_capacity(16);
        let mut total_table_rows = 0usize;
        for &p in &primes {
            buckets.push(PrimeBucketMod::new(p, total_table_rows));
            total_table_rows += p as usize;
        }

        let multipliers: [u64; 3] = [6364136223846793005, 1442695040888963407, 22695477];
        let seq_len = 1024;
        let tokens: Vec<u64> = (0..seq_len)
            .map(|i| ((i as u64) * 131 + 17) % 129_280)
            .collect();

        let mut idx_mod = vec![0usize; seq_len * total_heads];
        let mut idx_pq = vec![0usize; seq_len * total_heads];

        engram_hash_ngram_mod_into(&tokens, 2, &multipliers, n_heads_per_ngram, &buckets, &mut idx_mod);
        engram_hash_ngram_pq_into(&tokens, 2, &multipliers, n_heads_per_ngram, &buckets, &mut idx_pq);

        // 1. Prove bit-exact bucket equivalence between hardware `% prime` and Lemire p/q reciprocal!
        assert_eq!(idx_mod, idx_pq, "Lemire p/q prime hashing must be bit-identical to % prime");

        // Benchmark N-gram multi-head prime hashing
        let hash_iters = if cfg!(debug_assertions) { 5 } else { 200 };
        let t_h0 = Instant::now();
        for _ in 0..hash_iters {
            engram_hash_ngram_mod_into(
                std::hint::black_box(&tokens),
                2,
                &multipliers,
                n_heads_per_ngram,
                &buckets,
                &mut idx_mod,
            );
        }
        let hash_mod_us = t_h0.elapsed().as_secs_f64() * 1e6 / (hash_iters as f64);

        let t_h1 = Instant::now();
        for _ in 0..hash_iters {
            engram_hash_ngram_pq_into(
                std::hint::black_box(&tokens),
                2,
                &multipliers,
                n_heads_per_ngram,
                &buckets,
                &mut idx_pq,
            );
        }
        let hash_pq_us = t_h1.elapsed().as_secs_f64() * 1e6 / (hash_iters as f64);

        // 2. Build MultiHeadEmbedding table (`total_table_rows x 64`) in both Dense FP64 and BFP-64 (`i8`)
        let mut dense_table = vec![0.0f64; total_table_rows * ENGRAM_HEAD_DIM];
        for (i, v) in dense_table.iter_mut().enumerate() {
            *v = (((i * 73 + 19) % 200) as f64 - 100.0) * 0.005;
        }
        let bfp_table: Vec<Bfp64HeadRow> = (0..total_table_rows)
            .map(|r| Bfp64HeadRow::from_f64_slice(&dense_table[r * ENGRAM_HEAD_DIM..(r + 1) * ENGRAM_HEAD_DIM]))
            .collect();

        // Build 4 key_projs (`4 x 512 x 1024`) and 1 value_proj (`512 x 1024`) = 2,621,440 weights
        let num_groups = engram_dim / MLX_GROUP_SIZE; // 16
        let kp_len = hc_mult * hidden_size * engram_dim;
        let vp_len = hidden_size * engram_dim;

        let mut key_projs_signs = vec![0i8; kp_len];
        let mut key_projs_scales = vec![0.02f64; hc_mult * hidden_size * num_groups];
        let mut key_projs_dense = vec![0.0f64; kp_len];

        for i in 0..kp_len {
            let s = match (i * 37 + 11) % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            };
            key_projs_signs[i] = s;
            let row = i / engram_dim;
            let g = (i % engram_dim) / MLX_GROUP_SIZE;
            let scale = 0.015 + ((row + g) % 7) as f64 * 0.002;
            key_projs_scales[row * num_groups + g] = scale;
            key_projs_dense[i] = (s as f64) * scale;
        }

        let mut val_proj_signs = vec![0i8; vp_len];
        let mut val_proj_scales = vec![0.02f64; hidden_size * num_groups];
        let mut val_proj_dense = vec![0.0f64; vp_len];
        for i in 0..vp_len {
            let s = match (i * 53 + 7) % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            };
            val_proj_signs[i] = s;
            let row = i / engram_dim;
            let g = (i % engram_dim) / MLX_GROUP_SIZE;
            let scale = 0.018 + ((row + g) % 5) as f64 * 0.002;
            val_proj_scales[row * num_groups + g] = scale;
            val_proj_dense[i] = (s as f64) * scale;
        }

        let hidden_states_hc: Vec<f64> = (0..hc_mult * hidden_size)
            .map(|i| (((i * 41 + 3) % 100) as f64 - 50.0) * 0.02)
            .collect();
        let norm_w = vec![1.0f64; hidden_size];
        let short_conv_w = vec![0.25f64; hc_mult * hidden_size];

        let head_indices = &idx_pq[0..total_heads];
        let mut emb_scratch = vec![0.0f64; engram_dim];
        let mut m_x_scratch = vec![0i8; engram_dim];
        let mut exp_x_scratch = vec![0i8; num_groups];
        let mut val_scratch = vec![0.0f64; hidden_size];
        let mut key_scratch = vec![0.0f64; hidden_size];
        let mut nk_scratch = vec![0.0f64; hidden_size];
        let mut nq_scratch = vec![0.0f64; hidden_size];
        let mut out_dense = vec![0.0f64; hc_mult * hidden_size];
        let mut out_bfp64 = vec![0.0f64; hc_mult * hidden_size];

        // Warmup & accuracy verification
        engram_step_dense_fp64_into(
            &dense_table,
            head_indices,
            &hidden_states_hc,
            &key_projs_dense,
            &val_proj_dense,
            &norm_w,
            &short_conv_w,
            hc_mult,
            engram_dim,
            hidden_size,
            &mut emb_scratch,
            &mut val_scratch,
            &mut key_scratch,
            &mut nk_scratch,
            &mut nq_scratch,
            &mut out_dense,
        );

        engram_step_bfp64_into(
            &bfp_table,
            head_indices,
            &hidden_states_hc,
            &key_projs_signs,
            &key_projs_scales,
            &val_proj_signs,
            &val_proj_scales,
            &norm_w,
            &short_conv_w,
            hc_mult,
            engram_dim,
            hidden_size,
            &mut m_x_scratch,
            &mut exp_x_scratch,
            &mut val_scratch,
            &mut key_scratch,
            &mut nk_scratch,
            &mut nq_scratch,
            &mut out_bfp64,
        );

        let norm_ref: f64 = out_dense.iter().map(|x| x * x).sum::<f64>().sqrt();
        let diff_norm: f64 = out_dense
            .iter()
            .zip(out_bfp64.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let rel_err = diff_norm / norm_ref.max(1e-12);

        let step_iters = if cfg!(debug_assertions) { 2 } else { 50 };
        let t_s0 = Instant::now();
        for _ in 0..step_iters {
            engram_step_dense_fp64_into(
                std::hint::black_box(&dense_table),
                std::hint::black_box(head_indices),
                std::hint::black_box(&hidden_states_hc),
                std::hint::black_box(&key_projs_dense),
                std::hint::black_box(&val_proj_dense),
                &norm_w,
                &short_conv_w,
                hc_mult,
                engram_dim,
                hidden_size,
                &mut emb_scratch,
                &mut val_scratch,
                &mut key_scratch,
                &mut nk_scratch,
                &mut nq_scratch,
                &mut out_dense,
            );
        }
        let dense_step_us = t_s0.elapsed().as_secs_f64() * 1e6 / (step_iters as f64);

        let t_s1 = Instant::now();
        for _ in 0..step_iters {
            engram_step_bfp64_into(
                std::hint::black_box(&bfp_table),
                std::hint::black_box(head_indices),
                std::hint::black_box(&hidden_states_hc),
                std::hint::black_box(&key_projs_signs),
                std::hint::black_box(&key_projs_scales),
                std::hint::black_box(&val_proj_signs),
                std::hint::black_box(&val_proj_scales),
                &norm_w,
                &short_conv_w,
                hc_mult,
                engram_dim,
                hidden_size,
                &mut m_x_scratch,
                &mut exp_x_scratch,
                &mut val_scratch,
                &mut key_scratch,
                &mut nk_scratch,
                &mut nq_scratch,
                &mut out_bfp64,
            );
        }
        let bfp64_step_us = t_s1.elapsed().as_secs_f64() * 1e6 / (step_iters as f64);

        // Also benchmark the integer-aligned BFP-64 projection (`engram_bfp64_int_aligned_matmul_into`)
        // vs dense FP64 projection on `value_proj` (`1024 -> 512`)
        let val_proj_scales_pq: Vec<i32> = val_proj_scales
            .iter()
            .map(|&s| (s * (1_048_576.0_f64)).round() as i32)
            .collect();
        let mut val_dense_out = vec![0.0f64; hidden_size];
        let mut val_int_out = vec![0.0f64; hidden_size];

        let proj_iters = if cfg!(debug_assertions) { 5 } else { 200 };
        let t_p0 = Instant::now();
        for _ in 0..proj_iters {
            for j in 0..hidden_size {
                let w_row = &val_proj_dense[j * engram_dim..(j + 1) * engram_dim];
                let mut sum = 0.0f64;
                for i in 0..engram_dim {
                    sum += emb_scratch[i] * w_row[i];
                }
                val_dense_out[j] = sum;
            }
            std::hint::black_box(&val_dense_out);
        }
        let dense_proj_us = t_p0.elapsed().as_secs_f64() * 1e6 / (proj_iters as f64);

        let t_p1 = Instant::now();
        for _ in 0..proj_iters {
            engram_bfp64_int_aligned_matmul_into(
                std::hint::black_box(&m_x_scratch),
                std::hint::black_box(&exp_x_scratch),
                std::hint::black_box(&val_proj_signs),
                std::hint::black_box(&val_proj_scales_pq),
                engram_dim,
                hidden_size,
                &mut val_int_out,
            );
            std::hint::black_box(&val_int_out);
        }
        let int_aligned_proj_us = t_p1.elapsed().as_secs_f64() * 1e6 / (proj_iters as f64);

        let proj_norm_ref: f64 = val_dense_out.iter().map(|x| x * x).sum::<f64>().sqrt();
        let proj_err: f64 = val_dense_out
            .iter()
            .zip(val_int_out.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
            / proj_norm_ref.max(1e-12);

        // Benchmark MEM|8 Adaptive Precision head table (`BFP-i8 -> BFP-i16` promotion)
        let mem8_table: Vec<Mem8AdaptiveHeadRow> = (0..total_table_rows)
            .map(|r| {
                Mem8AdaptiveHeadRow::from_f64_slice(
                    &dense_table[r * ENGRAM_HEAD_DIM..(r + 1) * ENGRAM_HEAD_DIM],
                    0.0052,
                )
            })
            .collect();
        let promoted_heads = head_indices
            .iter()
            .filter(|&&idx| mem8_table[idx].promoted_i16)
            .count();
        let mut val_mem8_out = vec![0.0f64; hidden_size];
        let t_p2 = Instant::now();
        for _ in 0..proj_iters {
            engram_mem8_adaptive_matmul_into(
                std::hint::black_box(&mem8_table),
                std::hint::black_box(head_indices),
                std::hint::black_box(&val_proj_signs),
                std::hint::black_box(&val_proj_scales),
                engram_dim,
                hidden_size,
                &mut val_mem8_out,
            );
            std::hint::black_box(&val_mem8_out);
        }
        let mem8_proj_us = t_p2.elapsed().as_secs_f64() * 1e6 / (proj_iters as f64);
        let mem8_err: f64 = val_dense_out
            .iter()
            .zip(val_mem8_out.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
            / proj_norm_ref.max(1e-12);

        // Memory footprint calculation for DeepSeek Engram demo (10,342,604 rows x 64 dims)
        // and DeepSeek-V4.1-Flash (384,006,168 rows x 64 dims)
        let demo_rows = 10_342_604usize;
        let v41_rows = 384_006_168usize;
        let demo_fp32_gb = (demo_rows as f64 * 64.0 * 4.0) / 1e9;
        let demo_bfp64_gb = (demo_rows as f64 * 65.0) / 1e9;
        let v41_fp32_gb = (v41_rows as f64 * 64.0 * 4.0) / 1e9;
        let v41_bfp64_gb = (v41_rows as f64 * 65.0) / 1e9;

        std::println!("\n=== DEEPSEEK ENGRAM (github.com/deepseek-ai/Engram) ON GAIA p/q + BFP-64 ===");
        std::println!(
            "  1. Multi-Head N-Gram Prime Hashing (1024 tokens x 16 prime heads = 16,384 hashes):"
        );
        std::println!("     - Hardware `% prime` modulo:         {:>8.2} us", hash_mod_us);
        std::println!(
            "     - Lemire p/q (q=2^64) Reciprocal:    {:>8.2} us ({:.2}x speedup, bit-exact!)",
            hash_pq_us,
            hash_mod_us / hash_pq_us
        );
        std::println!(
            "  2. Native 64-Dim Head Table Compression (`D_head = 64 == BFP-64 block size`):"
        );
        std::println!(
            "     - Engram Demo Table (10.34M rows):   {:.2} GB (FP32) -> {:.2} GB (BFP-64) = {:.2}x smaller",
            demo_fp32_gb,
            demo_bfp64_gb,
            demo_fp32_gb / demo_bfp64_gb
        );
        std::println!(
            "     - DeepSeek-V4.1 Table (384.0M rows): {:.2} GB (FP32) -> {:.2} GB (BFP-64) = {:.2}x smaller",
            v41_fp32_gb,
            v41_bfp64_gb,
            v41_fp32_gb / v41_bfp64_gb
        );
        std::println!(
            "  3. Isolated Engram Head Projection (`1024 -> 512`, 16 Pre-Quantized 64-Dim Heads):"
        );
        std::println!("     - Dense FP64 Projection:             {:>8.2} us (1.00x)", dense_proj_us);
        std::println!(
            "     - Integer-Aligned BFP-64 (i16->i64): {:>8.2} us ({:.2}x speedup) | rel_err: {:.5}",
            int_aligned_proj_us,
            dense_proj_us / int_aligned_proj_us,
            proj_err
        );
        std::println!(
            "     - MEM|8 Adaptive ({:>2}/16 prom i8->i16):{:>7.2} us ({:.2}x speedup) | rel_err: {:.5}",
            promoted_heads,
            mem8_proj_us,
            dense_proj_us / mem8_proj_us,
            mem8_err
        );
        std::println!(
            "  4. Full Engram Layer Step (16x64 Gather + 4 KeyProjs + 1 ValProj [2.62M weights] + HC Gate + ShortConv):"
        );
        std::println!("     - Dense FP64 Engram Step:            {:>8.2} us (1.00x)", dense_step_us);
        std::println!(
            "     - Zero-Quant BFP-64 + i16 SIMD Step: {:>8.2} us ({:.2}x speedup) | rel_err: {:.5}",
            bfp64_step_us,
            dense_step_us / bfp64_step_us,
            rel_err
        );

        assert!(rel_err < 0.01, "Expected BFP-64 Engram relative error < 1%: {}", rel_err);
        assert!(proj_err < 0.01, "Expected Int-Aligned BFP-64 relative error < 1%: {}", proj_err);
        assert!(mem8_err < proj_err, "Expected MEM|8 Adaptive error < BFP-i8 error");
        #[cfg(not(debug_assertions))]
        assert!(bfp64_step_us < dense_step_us);
    }

    #[test]
    fn engram_zero_float_step_tracks_dense_step() {
        let primes: [u64; 8] = [101, 103, 107, 109, 113, 127, 131, 137];
        let mut head_indices = [0usize; 8];
        let mut total_rows = 0usize;
        for (i, &p) in primes.iter().enumerate() {
            head_indices[i] = total_rows;
            total_rows += p as usize;
        }
        let mut dense_table = vec![0.0f64; total_rows * ENGRAM_HEAD_DIM];
        for (i, v) in dense_table.iter_mut().enumerate() {
            *v = (((i * 73 + 19) % 200) as f64 - 100.0) * 0.005;
        }
        let bfp_table: Vec<Bfp64HeadRow> = (0..total_rows)
            .map(|r| {
                Bfp64HeadRow::from_f64_slice(&dense_table[r * ENGRAM_HEAD_DIM..(r + 1) * ENGRAM_HEAD_DIM])
            })
            .collect();

        let hc_mult = 2usize;
        let hidden_size = 64usize;
        let engram_dim = 8 * ENGRAM_HEAD_DIM;
        let num_groups = engram_dim / MLX_GROUP_SIZE;

        let mut key_signs = vec![0i8; hc_mult * hidden_size * engram_dim];
        let mut key_scales_pq = vec![0i32; hc_mult * hidden_size * num_groups];
        let mut key_dense = vec![0.0f64; hc_mult * hidden_size * engram_dim];
        for i in 0..key_signs.len() {
            let s = match (i * 37 + 11) % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            };
            key_signs[i] = s;
            let row = i / engram_dim;
            let g = (i % engram_dim) / MLX_GROUP_SIZE;
            let scale = 0.015 + ((row + g) % 7) as f64 * 0.002;
            key_scales_pq[row * num_groups + g] = (scale * 1_048_576.0).round() as i32;
            key_dense[i] = (s as f64) * scale;
        }
        let mut val_signs = vec![0i8; hidden_size * engram_dim];
        let mut val_scales_pq = vec![0i32; hidden_size * num_groups];
        let mut val_dense = vec![0.0f64; hidden_size * engram_dim];
        for i in 0..val_signs.len() {
            let s = match (i * 53 + 7) % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            };
            val_signs[i] = s;
            let row = i / engram_dim;
            let g = (i % engram_dim) / MLX_GROUP_SIZE;
            let scale = 0.018 + ((row + g) % 5) as f64 * 0.002;
            val_scales_pq[row * num_groups + g] = (scale * 1_048_576.0).round() as i32;
            val_dense[i] = (s as f64) * scale;
        }

        let hidden_states: Vec<f64> = (0..hc_mult * hidden_size)
            .map(|i| (((i * 41 + 3) % 100) as f64 - 50.0) * 0.02)
            .collect();
        let hidden_states_q16: Vec<i32> = hidden_states
            .iter()
            .map(|&x| (x * (ENGRAM_PQ_Q16 as f64)).round() as i32)
            .collect();
        let norm_w = vec![1.0f64; hidden_size];
        let norm_w_q16 = vec![ENGRAM_PQ_Q16 as i32; hidden_size];
        let short_conv_w = vec![0.25f64; hc_mult * hidden_size];
        let short_conv_w_q16 =
            vec![(0.25 * (ENGRAM_PQ_Q16 as f64)).round() as i32; hc_mult * hidden_size];

        let mut emb_scratch = vec![0.0f64; engram_dim];
        let mut m_x_scratch = vec![0i8; engram_dim];
        let mut exp_x_scratch = vec![0i8; num_groups];
        let mut val_scratch = vec![0.0f64; hidden_size];
        let mut key_scratch = vec![0.0f64; hidden_size];
        let mut nk_scratch = vec![0.0f64; hidden_size];
        let mut nq_scratch = vec![0.0f64; hidden_size];
        let mut val_q16_scratch = vec![0i32; hidden_size];
        let mut key_q16_scratch = vec![0i32; hidden_size];
        let mut out_dense = vec![0.0f64; hc_mult * hidden_size];
        let mut out_q16 = vec![0i32; hc_mult * hidden_size];

        engram_step_dense_fp64_into(
            &dense_table,
            &head_indices,
            &hidden_states,
            &key_dense,
            &val_dense,
            &norm_w,
            &short_conv_w,
            hc_mult,
            engram_dim,
            hidden_size,
            &mut emb_scratch,
            &mut val_scratch,
            &mut key_scratch,
            &mut nk_scratch,
            &mut nq_scratch,
            &mut out_dense,
        );
        engram_step_pq_no_floats_into(
            &bfp_table,
            &head_indices,
            &hidden_states_q16,
            &key_signs,
            &key_scales_pq,
            &val_signs,
            &val_scales_pq,
            &norm_w_q16,
            &short_conv_w_q16,
            hc_mult,
            engram_dim,
            hidden_size,
            &mut m_x_scratch,
            &mut exp_x_scratch,
            &mut val_q16_scratch,
            &mut key_q16_scratch,
            &mut out_q16,
        );

        let norm_ref = out_dense.iter().map(|x| x * x).sum::<f64>().sqrt();
        let err: f64 = out_dense
            .iter()
            .zip(out_q16.iter())
            .map(|(&a, &b)| (a - (b as f64) / (ENGRAM_PQ_Q16 as f64)).powi(2))
            .sum::<f64>()
            .sqrt()
            / norm_ref;
        assert!(err < 0.05, "zero-float step relative error {err} exceeds 0.05");
    }

    #[test]
    #[should_panic(expected = "num_groups")]
    fn engram_matmul_functions_reject_more_than_64_groups() {
        let in_features = 65 * MLX_GROUP_SIZE;
        let out_features = 4;
        let m_x = vec![0i8; in_features];
        let exp_x = vec![0i8; 65];
        let signs = vec![0i8; out_features * in_features];
        let scales_pq = vec![0i32; out_features * 65];
        let mut out = vec![0.0f64; out_features];
        engram_bfp64_int_aligned_matmul_into(
            &m_x,
            &exp_x,
            &signs,
            &scales_pq,
            in_features,
            out_features,
            &mut out,
        );
    }
}
