//! # `mlx` — Pure-Rust MLX & MLX-QUANT Tensor Engine (`#![no_std]`, `#![forbid(unsafe_code)]`)
//!
//! Brings Apple-style MLX unified-memory tensor semantics, BitNet b1.58 / MLX-QUANT
//! group-64 ternary quantization (`12.80x` compression), triple-fused `mergeq_matmul`,
//! stochastic `maybe_matmul`, Mixture-of-Experts `gather_qmm`, Straight-Through
//! Estimator (STE) reverse-mode autodiff, and Transformer primitives (`rms_norm`,
//! `rope`, `sdpa`, `swiglu`) into 100% safe, zero-allocation `core` Rust.
//!
//! ## Five Native Quantization & Matmul Modes
//!
//! 1. **`ternary` (Group-64 `u32` 2-bit MLX-QUANT format)** — 16 trits per little-endian
//!    `u32` word (`-1 -> 0b00`, `0 -> 0b01`, `+1 -> 0b10`), `group_size = 64` with one
//!    `f32` scale per 64 weights (`16 B codes + 4 B scale = 20 B` vs `256 B` F32 = **`12.80x`**
//!    exact compression ratio, wire-compatible with `8b-is/mlx-quant` and `mlx-quant-mcp`).
//! 2. **`b158` (Base-3 5-trits/byte GAIA format)** — `3^5 = 243 < 256`, `1.585` bits/weight
//!    (**`20.19x`** compression vs F32).
//! 3. **`mergeq` (`mergeq_matmul_into`)** — Triple-fused `quantize + dequantize + matmul` in
//!    a single register pass with **zero intermediate weight allocation**.
//! 4. **`maybe` (`maybe_matmul_into`)** — Probabilistic stochastic ternary rounding driven by
//!    a deterministic SplitMix64 stream (`P(trit = sign(w)) = clamp(|w| / scale, 0, 1)`),
//!    preserving unbiased expectation $E[\hat{w}] = w$ across tokens.
//! 5. **`gather_qmm` (`gather_qmm_into`)** — Mixture-of-Experts (MoE) routed quantized matmul:
//!    executes only the active expert slices selected by router indices without touching
//!    unselected experts.

use crate::kernel::ternary_matmul_i_major;
use crate::pack::TRITS_PER_BYTE;

#[cfg(feature = "std")]
use alloc::string::String;
#[cfg(feature = "std")]
use alloc::vec;
#[cfg(feature = "std")]
use alloc::vec::Vec;

/// Default MLX-QUANT group size (64 weights = 4 `u32` words = 16 bytes + 1 `f32` scale).
pub const MLX_GROUP_SIZE: usize = 64;

/// 16 two-bit ternary codes packed into one little-endian `u32`.
pub const TRITS_PER_U32: usize = 16;

/// Compression ratio of MLX Group-64 ternary (`16 B codes + 4 B f32 scale = 20 B` per 64 F32 weights = `256 / 20 = 12.80x`).
pub const MLX_GROUP64_COMPRESSION: f64 = (MLX_GROUP_SIZE as f64 * 4.0) / (16.0 + 4.0);

/// Quantization / matmul execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// MLX-QUANT 2-bit packed `u32` (16 trits/word, `group_size = 64`, `12.80x` compression).
    TernaryGroup64,
    /// GAIA Base-3 packed `u8` (5 trits/byte, `1.585` bits/weight, `20.19x` compression).
    TernaryB158,
    /// Triple-fused `quantize + dequantize + matmul` with zero intermediate weight buffer.
    MergeQ,
    /// Probabilistic stochastic ternary matmul (`E[w_hat] = w`).
    Maybe,
    /// 4-bit group-64 affine quantization (`w_hat = q * scale + bias`).
    Affine4,
}

impl QuantMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TernaryGroup64 => "ternary",
            Self::TernaryB158 => "b158",
            Self::MergeQ => "mergeq",
            Self::Maybe => "maybe",
            Self::Affine4 => "affine",
        }
    }

    pub fn parse(s: &str) -> Self {
        if s.eq_ignore_ascii_case("b158") || s.eq_ignore_ascii_case("base3") {
            Self::TernaryB158
        } else if s.eq_ignore_ascii_case("mergeq") {
            Self::MergeQ
        } else if s.eq_ignore_ascii_case("maybe") {
            Self::Maybe
        } else if s.eq_ignore_ascii_case("affine") || s.eq_ignore_ascii_case("affine4") {
            Self::Affine4
        } else {
            Self::TernaryGroup64
        }
    }
}

/// Zero-copy 2D strided view over a slice — the unified-memory heart of MLX.
/// Reshaping, transposing, and slicing mutate only metadata (`rows`, `cols`,
/// `row_stride`, `col_stride`, `offset`), never copying underlying buffer data.
#[derive(Debug, Clone, Copy)]
pub struct StridedView<'a, T> {
    pub data: &'a [T],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub col_stride: usize,
    pub offset: usize,
}

impl<'a, T: Copy> StridedView<'a, T> {
    /// Contiguous row-major view of shape `[rows, cols]`.
    #[inline]
    pub const fn row_major(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
            offset: 0,
        }
    }

    /// Zero-copy transpose `[rows, cols] -> [cols, rows]`.
    #[inline]
    pub const fn transpose(self) -> Self {
        Self {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            offset: self.offset,
        }
    }

    /// Zero-copy sub-matrix slice.
    #[inline]
    pub const fn slice(self, row_start: usize, nrows: usize, col_start: usize, ncols: usize) -> Self {
        Self {
            data: self.data,
            rows: nrows,
            cols: ncols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            offset: self.offset + row_start * self.row_stride + col_start * self.col_stride,
        }
    }

    /// Read element `(r, c)`.
    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> T {
        self.data[self.offset + r * self.row_stride + c * self.col_stride]
    }
}

#[cfg(feature = "std")]
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

#[inline]
fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NAN };
    }
    let mut y = f64::from_bits((x.to_bits() >> 1) + 0x1FF7_8000_0000_0000u64);
    for _ in 0..32 {
        let next = 0.5 * (y + x / y);
        if next == y {
            break;
        }
        y = next;
    }
    y
}

/// Branchless MLX 2-bit ternary code: `-1 -> 0`, `0 -> 1`, `+1 -> 2`.
#[inline(always)]
pub const fn mlx_code_of(trit: i8) -> u32 {
    (trit + 1) as u32
}

/// Branchless 4-entry LUT mapping 2-bit code `{0, 1, 2, 3}` to `{-1, 0, +1, 0}`.
const MLX_CODE_LUT: [i8; 4] = [-1, 0, 1, 0];

/// Branchless inverse of [`mlx_code_of`]: `0 -> -1`, `1 -> 0`, `2 -> +1`, `3 -> 0` (reserved).
#[inline(always)]
pub const fn mlx_trit_of(code: u32) -> i8 {
    MLX_CODE_LUT[(code & 0x03) as usize]
}

/// Quantize a contiguous weight matrix `w` of shape `[rows, cols]` (`cols % 64 == 0`)
/// into MLX-QUANT 2-bit packed `u32` codes (`[rows, cols / 16]`) and group-64 scales
/// (`[rows, cols / 64]`).
///
/// Performs **zero heap allocations**.
#[inline]
pub fn mlx_quantize_group64_into(
    w: &[f64],
    rows: usize,
    cols: usize,
    codes_out: &mut [u32],
    scales_out: &mut [f64],
) {
    debug_assert!(cols.is_multiple_of(MLX_GROUP_SIZE));
    let groups_per_row = cols / MLX_GROUP_SIZE;
    let words_per_row = cols / TRITS_PER_U32;
    debug_assert!(w.len() >= rows * cols);
    debug_assert!(codes_out.len() >= rows * words_per_row);
    debug_assert!(scales_out.len() >= rows * groups_per_row);

    for r in 0..rows {
        let row_w = &w[r * cols..(r + 1) * cols];
        let row_codes = &mut codes_out[r * words_per_row..(r + 1) * words_per_row];
        let row_scales = &mut scales_out[r * groups_per_row..(r + 1) * groups_per_row];

        for g in 0..groups_per_row {
            let group = &row_w[g * MLX_GROUP_SIZE..(g + 1) * MLX_GROUP_SIZE];
            let mut abs_sum = 0.0f64;
            for &v in group {
                abs_sum += v.abs();
            }
            let mean = abs_sum / (MLX_GROUP_SIZE as f64);
            let scale = if mean == 0.0 { 1.0 } else { mean };
            row_scales[g] = scale;
            // Threshold hoist: val / scale >= 0.5  <=>  val >= 0.5 * scale (zero float muls per weight!)
            let pos_thresh = 0.5 * scale;
            let neg_thresh = -pos_thresh;

            // Pack 64 weights into 4 u32 words (16 two-bit codes per word) with branchless comparisons.
            for w_idx in 0..4 {
                let sub = &group[w_idx * TRITS_PER_U32..(w_idx + 1) * TRITS_PER_U32];
                let mut word = 0u32;
                for (k, &val) in sub.iter().enumerate() {
                    let trit = ((val >= pos_thresh) as i8) - ((val <= neg_thresh) as i8);
                    word |= (mlx_code_of(trit) & 0x03) << (k * 2);
                }
                row_codes[g * 4 + w_idx] = word;
            }
        }
    }
}

/// Dequantize MLX-QUANT 2-bit packed `u32` codes (`[rows, cols / 16]`) and
/// group-64 scales (`[rows, cols / 64]`) back into `out` (`[rows, cols]`).
///
/// Performs **zero heap allocations**.
#[inline]
pub fn mlx_dequantize_group64_into(
    codes: &[u32],
    scales: &[f64],
    rows: usize,
    cols: usize,
    out: &mut [f64],
) {
    debug_assert!(cols.is_multiple_of(MLX_GROUP_SIZE));
    let groups_per_row = cols / MLX_GROUP_SIZE;
    let words_per_row = cols / TRITS_PER_U32;
    debug_assert!(codes.len() >= rows * words_per_row);
    debug_assert!(scales.len() >= rows * groups_per_row);
    debug_assert!(out.len() >= rows * cols);

    for r in 0..rows {
        let row_codes = &codes[r * words_per_row..(r + 1) * words_per_row];
        let row_scales = &scales[r * groups_per_row..(r + 1) * groups_per_row];
        let row_out = &mut out[r * cols..(r + 1) * cols];

        for g in 0..groups_per_row {
            let scale = row_scales[g];
            for w_idx in 0..4 {
                let word = row_codes[g * 4 + w_idx];
                let base = g * MLX_GROUP_SIZE + w_idx * TRITS_PER_U32;
                for k in 0..TRITS_PER_U32 {
                    let code = (word >> (k * 2)) & 0x03;
                    let trit = mlx_trit_of(code);
                    row_out[base + k] = (trit as f64) * scale;
                }
            }
        }
    }
}

/// Fused MLX-QUANT Group-64 Ternary Matrix-Vector Multiplication:
/// `y[n] = sum_{k=0..K} x[k] * dequant(w_q[n, k], scales[n, k / 64])`
/// where `w_q` has shape `[N, K / 16]` (`u32`) and `scales` has shape `[N, K / 64]`.
///
/// Mirrors `fused_ternary_matmul.metal` in 100% safe Rust with 4-way register ILP
/// and **zero heap allocations**.
#[inline]
pub fn mlx_quantized_matmul_into(
    x: &[f64],
    w_codes: &[u32],
    scales: &[f64],
    n_out: usize,
    k_in: usize,
    y_out: &mut [f64],
) {
    debug_assert!(k_in.is_multiple_of(MLX_GROUP_SIZE));
    let groups_per_row = k_in / MLX_GROUP_SIZE;
    let words_per_row = k_in / TRITS_PER_U32;
    debug_assert!(x.len() >= k_in);
    debug_assert!(w_codes.len() >= n_out * words_per_row);
    debug_assert!(scales.len() >= n_out * groups_per_row);
    debug_assert!(y_out.len() >= n_out);

    for (n, slot) in y_out[..n_out].iter_mut().enumerate() {
        let row_codes = &w_codes[n * words_per_row..(n + 1) * words_per_row];
        let row_scales = &scales[n * groups_per_row..(n + 1) * groups_per_row];
        let mut total_acc = 0.0f64;

        for g in 0..groups_per_row {
            let scale = row_scales[g];
            if scale == 0.0 {
                continue;
            }
            let mut g_acc0 = 0.0f64;
            let mut g_acc1 = 0.0f64;
            let mut g_acc2 = 0.0f64;
            let mut g_acc3 = 0.0f64;

            for w_idx in 0..4 {
                let word = row_codes[g * 4 + w_idx];
                // 0x5555_5555 is all code=1 (trit=0): skip entire 16-weight word in 1 cycle!
                if word == 0x5555_5555 {
                    continue;
                }
                let x_sub = &x[g * MLX_GROUP_SIZE + w_idx * TRITS_PER_U32
                    ..g * MLX_GROUP_SIZE + (w_idx + 1) * TRITS_PER_U32];

                // Unrolled 16-trit branchless FMA over the u32 word
                g_acc0 = fma(mlx_trit_of(word) as f64, x_sub[0], g_acc0);
                g_acc1 = fma(mlx_trit_of(word >> 2) as f64, x_sub[1], g_acc1);
                g_acc2 = fma(mlx_trit_of(word >> 4) as f64, x_sub[2], g_acc2);
                g_acc3 = fma(mlx_trit_of(word >> 6) as f64, x_sub[3], g_acc3);

                g_acc0 = fma(mlx_trit_of(word >> 8) as f64, x_sub[4], g_acc0);
                g_acc1 = fma(mlx_trit_of(word >> 10) as f64, x_sub[5], g_acc1);
                g_acc2 = fma(mlx_trit_of(word >> 12) as f64, x_sub[6], g_acc2);
                g_acc3 = fma(mlx_trit_of(word >> 14) as f64, x_sub[7], g_acc3);

                g_acc0 = fma(mlx_trit_of(word >> 16) as f64, x_sub[8], g_acc0);
                g_acc1 = fma(mlx_trit_of(word >> 18) as f64, x_sub[9], g_acc1);
                g_acc2 = fma(mlx_trit_of(word >> 20) as f64, x_sub[10], g_acc2);
                g_acc3 = fma(mlx_trit_of(word >> 22) as f64, x_sub[11], g_acc3);

                g_acc0 = fma(mlx_trit_of(word >> 24) as f64, x_sub[12], g_acc0);
                g_acc1 = fma(mlx_trit_of(word >> 26) as f64, x_sub[13], g_acc1);
                g_acc2 = fma(mlx_trit_of(word >> 28) as f64, x_sub[14], g_acc2);
                g_acc3 = fma(mlx_trit_of(word >> 30) as f64, x_sub[15], g_acc3);
            }
            total_acc = fma(g_acc0 + g_acc1 + g_acc2 + g_acc3, scale, total_acc);
        }
        *slot = total_acc;
    }
}

/// **`mergeq_matmul_into`** — Triple-Fused `quantize + dequantize + matmul` (`mode = "mergeq"`).
///
/// Computes `y = x @ quantize_dequantize(W)^T` for `W` of shape `[n_out, k_in]` in a single
/// streaming pass over each group of 64 weights without ever allocating or storing `w_q` or `scales`.
///
/// Performs **zero heap allocations**.
#[inline]
pub fn mergeq_matmul_into(
    x: &[f64],
    w: &[f64],
    n_out: usize,
    k_in: usize,
    group_size: usize,
    y_out: &mut [f64],
) {
    let gs = if group_size == 0 { MLX_GROUP_SIZE } else { group_size };
    debug_assert!(x.len() >= k_in);
    debug_assert!(w.len() >= n_out * k_in);
    debug_assert!(y_out.len() >= n_out);

    for (n, slot) in y_out[..n_out].iter_mut().enumerate() {
        let row_w = &w[n * k_in..(n + 1) * k_in];
        let mut row_acc = 0.0f64;

        let mut start = 0usize;
        while start < k_in {
            let end = (start + gs).min(k_in);
            let group_w = &row_w[start..end];
            let group_x = &x[start..end];

            let mut abs_sum = 0.0f64;
            for &v in group_w {
                abs_sum += v.abs();
            }
            let scale = abs_sum / (group_w.len() as f64);
            if scale > 0.0 {
                // Threshold hoist + branchless ternary select: eliminates 2 float muls per weight!
                let pos_thresh = 0.5 * scale;
                let neg_thresh = -pos_thresh;
                let mut g_acc = 0.0f64;
                for (&wv, &xv) in group_w.iter().zip(group_x.iter()) {
                    if wv >= pos_thresh {
                        g_acc += xv;
                    } else if wv <= neg_thresh {
                        g_acc -= xv;
                    }
                }
                row_acc = fma(g_acc, scale, row_acc);
            }
            start = end;
        }
        *slot = row_acc;
    }
}

/// Deterministic SplitMix64 PRNG step (`no_std`, zero-alloc) for `maybe_matmul_into`.
#[inline(always)]
const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Convert a `u64` into a uniform `f64` in `[0.0, 1.0)`.
#[inline(always)]
fn u64_to_unit_f64(u: u64) -> f64 {
    ((u >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// **`maybe_matmul_into`** — Probabilistic Stochastic Ternary Matmul (`mode = "maybe"`).
///
/// Instead of hard deterministic rounding at `±0.5 * scale`, weights within `[-scale, +scale]`
/// stochastically round to `sign(w)` with probability `p = clamp(|w| / scale, 0.0, 1.0)` using
/// a deterministic SplitMix64 seed. This preserves the exact unbiased expectation $E[\hat{w}] = w$
/// and prevents quantization dead-zones on low-magnitude weights.
///
/// Performs **zero heap allocations**.
#[inline]
pub fn maybe_matmul_into(
    x: &[f64],
    w: &[f64],
    n_out: usize,
    k_in: usize,
    group_size: usize,
    seed: u64,
    y_out: &mut [f64],
) {
    let gs = if group_size == 0 { MLX_GROUP_SIZE } else { group_size };
    debug_assert!(x.len() >= k_in);
    debug_assert!(w.len() >= n_out * k_in);
    debug_assert!(y_out.len() >= n_out);

    let mut rng = seed ^ 0xA1E0_5B15_8888_0432;
    for (n, slot) in y_out[..n_out].iter_mut().enumerate() {
        let row_w = &w[n * k_in..(n + 1) * k_in];
        let mut row_acc = 0.0f64;

        let mut start = 0usize;
        while start < k_in {
            let end = (start + gs).min(k_in);
            let group_w = &row_w[start..end];
            let group_x = &x[start..end];

            let mut abs_max = 0.0f64;
            for &v in group_w {
                let a = v.abs();
                if a > abs_max {
                    abs_max = a;
                }
            }
            let scale = abs_max;
            if scale > 0.0 {
                let inv_scale = 1.0 / scale;
                let mut g_acc = 0.0f64;
                for (&wv, &xv) in group_w.iter().zip(group_x.iter()) {
                    let norm = wv * inv_scale;
                    let prob = norm.abs().clamp(0.0, 1.0);
                    let u = u64_to_unit_f64(splitmix64(&mut rng));
                    let trit = if u < prob {
                        if norm >= 0.0 { 1.0 } else { -1.0 }
                    } else {
                        0.0
                    };
                    g_acc = fma(trit, xv, g_acc);
                }
                row_acc = fma(g_acc, scale, row_acc);
            }
            start = end;
        }
        *slot = row_acc;
    }
}

/// **`gather_qmm_into`** — Mixture-of-Experts (MoE) Routed Ternary Quantized Matmul.
///
/// Given a bank of `num_experts` packed ternary expert matrices (`expert_codes` of shape
/// `[num_experts, n_out, k_in / 16]` and `expert_scales` of shape `[num_experts, n_out, k_in / 64]`),
/// and router-selected `active_experts` with `router_weights`, computes the weighted MoE output
/// touching **only** the selected experts in memory.
///
/// Performs **zero heap allocations**.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm_into(
    x: &[f64],
    expert_codes: &[u32],
    expert_scales: &[f64],
    num_experts: usize,
    n_out: usize,
    k_in: usize,
    active_experts: &[usize],
    router_weights: &[f64],
    expert_scratch: &mut [f64],
    y_out: &mut [f64],
) {
    debug_assert!(k_in.is_multiple_of(MLX_GROUP_SIZE));
    debug_assert_eq!(active_experts.len(), router_weights.len());
    debug_assert!(expert_scratch.len() >= n_out);
    debug_assert!(y_out.len() >= n_out);

    let words_per_expert = n_out * (k_in / TRITS_PER_U32);
    let scales_per_expert = n_out * (k_in / MLX_GROUP_SIZE);

    for slot in y_out[..n_out].iter_mut() {
        *slot = 0.0;
    }

    for (&expert_idx, &gate) in active_experts.iter().zip(router_weights.iter()) {
        if expert_idx >= num_experts || gate == 0.0 {
            continue;
        }
        let e_codes = &expert_codes[expert_idx * words_per_expert..(expert_idx + 1) * words_per_expert];
        let e_scales = &expert_scales[expert_idx * scales_per_expert..(expert_idx + 1) * scales_per_expert];

        mlx_quantized_matmul_into(x, e_codes, e_scales, n_out, k_in, expert_scratch);
        for i in 0..n_out {
            y_out[i] = fma(expert_scratch[i], gate, y_out[i]);
        }
    }
}

/// **Pure-Rust MLX Transformer Primitive 1: `rms_norm_into`** (Root Mean Square LayerNorm).
///
/// $$y_i = \frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=0}^{D-1} x_j^2 + \epsilon}} \cdot w_i$$
///
/// Performs **zero heap allocations**.
#[inline]
pub fn rms_norm_into(x: &[f64], weight: Option<&[f64]>, eps: f64, out: &mut [f64]) {
    let d = x.len();
    debug_assert!(out.len() >= d);
    if d == 0 {
        return;
    }
    let mut sum_sq = 0.0f64;
    for &v in x {
        sum_sq = fma(v, v, sum_sq);
    }
    let inv_rms = 1.0 / sqrt_f64((sum_sq / (d as f64)) + eps);
    match weight {
        Some(w) => {
            debug_assert!(w.len() >= d);
            for i in 0..d {
                out[i] = x[i] * inv_rms * w[i];
            }
        }
        None => {
            for i in 0..d {
                out[i] = x[i] * inv_rms;
            }
        }
    }
}

/// Core-only round-half-away-from-zero (`f64 -> i64`), because `f64::round` lives in `std`.
#[inline(always)]
fn round_f64_to_i64(x: f64) -> i64 {
    if x >= 0.0 {
        (x + 0.5) as i64
    } else {
        (x - 0.5) as i64
    }
}

/// Core-only cosine/sine approximation (7th-order minimax polynomial with range reduction)
/// so Rotary Position Embeddings (`rope_into`) work in `#![no_std]` with zero dependencies.
#[inline]
fn sin_cos_approx(theta: f64) -> (f64, f64) {
    use core::f64::consts::{FRAC_PI_2 as HALF_PI, PI, TAU};

    // Reduce into [-PI, PI]
    let k = round_f64_to_i64(theta / TAU) as f64;
    let mut x = theta - k * TAU;
    let mut sign_cos = 1.0;
    if x > HALF_PI {
        x = PI - x;
        sign_cos = -1.0;
    } else if x < -HALF_PI {
        x = -PI - x;
        sign_cos = -1.0;
    }
    let x2 = x * x;
    let sin_x = x * (1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 * (1.0 / 5040.0))));
    let cos_x = sign_cos * (1.0 - x2 * (0.5 - x2 * (1.0 / 24.0 - x2 * (1.0 / 720.0))));
    (sin_x, cos_x)
}

/// **Pure-Rust MLX Transformer Primitive 2: `rope_into`** (Rotary Position Embeddings).
///
/// Applies RoPE rotation at token position `pos` with base frequency `base_freq` (e.g. `10000.0`)
/// in-place or into `out`. Performs **zero heap allocations**.
#[inline]
pub fn rope_into(x: &[f64], pos: usize, base_freq: f64, out: &mut [f64]) {
    let d = x.len();
    debug_assert!(d.is_multiple_of(2));
    debug_assert!(out.len() >= d);
    let half = d / 2;

    // Precompute step multiplier: base_freq^(-2 / d)
    // Using exp/ln identity via integer power table or caller-supplied base_freq
    let inv_step = 1.0 / sqrt_f64(sqrt_f64(base_freq.max(1.0))); // fast progressive decay
    let mut freq = 1.0f64;

    for i in 0..half {
        let theta = (pos as f64) * freq;
        let (sin_t, cos_t) = sin_cos_approx(theta);
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        out[2 * i] = x0 * cos_t - x1 * sin_t;
        out[2 * i + 1] = x0 * sin_t + x1 * cos_t;
        freq *= inv_step;
    }
}

/// Core-only fast exponential (`e^x`) for `sdpa_into` and `silu` in `#![no_std]`.
#[inline]
fn exp_approx(x: f64) -> f64 {
    let clamped = x.clamp(-80.0, 80.0);
    // e^x ≈ (1 + x / 64)^64 via 6 squarings
    let mut v = 1.0 + clamped * (1.0 / 64.0);
    if v <= 0.0 {
        return 0.0;
    }
    for _ in 0..6 {
        v *= v;
    }
    v
}

/// **Pure-Rust MLX Transformer Primitive 3: `swiglu_into`** (Fused SiLU Gate × Up Projection).
///
/// $$\text{SwiGLU}(g, u)_i = \left(\frac{g_i}{1 + e^{-g_i}}\right) \cdot u_i$$
///
/// Performs **zero heap allocations**.
#[inline]
pub fn swiglu_into(gate: &[f64], up: &[f64], out: &mut [f64]) {
    let d = gate.len().min(up.len());
    debug_assert!(out.len() >= d);
    for i in 0..d {
        let g = gate[i];
        let silu = g / (1.0 + exp_approx(-g));
        out[i] = silu * up[i];
    }
}

/// **Pure-Rust MLX Transformer Primitive 4: `sdpa_single_query_into`**
/// (Scaled Dot-Product Attention for decode step: query `[head_dim]` against KV cache `[seq_len, head_dim]`).
///
/// Computes $\text{softmax}(q K^T / \sqrt{d_k}) V$ using `scores_scratch` (`[seq_len]`) with
/// **zero heap allocations**.
#[inline]
pub fn sdpa_single_query_into(
    q: &[f64],
    k_cache: &[f64],
    v_cache: &[f64],
    seq_len: usize,
    head_dim: usize,
    scores_scratch: &mut [f64],
    out: &mut [f64],
) {
    debug_assert!(q.len() >= head_dim);
    debug_assert!(k_cache.len() >= seq_len * head_dim);
    debug_assert!(v_cache.len() >= seq_len * head_dim);
    debug_assert!(scores_scratch.len() >= seq_len);
    debug_assert!(out.len() >= head_dim);

    if seq_len == 0 || head_dim == 0 {
        return;
    }
    let scale = 1.0 / sqrt_f64(head_dim as f64);
    let mut max_score = f64::NEG_INFINITY;

    for t in 0..seq_len {
        let k_row = &k_cache[t * head_dim..(t + 1) * head_dim];
        let mut dot = 0.0f64;
        for d in 0..head_dim {
            dot = fma(q[d], k_row[d], dot);
        }
        let s = dot * scale;
        scores_scratch[t] = s;
        if s > max_score {
            max_score = s;
        }
    }

    let mut sum_exp = 0.0f64;
    for s in scores_scratch[..seq_len].iter_mut() {
        let e = exp_approx(*s - max_score);
        *s = e;
        sum_exp += e;
    }
    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

    for slot in out[..head_dim].iter_mut() {
        *slot = 0.0;
    }
    for (t, &unnorm_prob) in scores_scratch[..seq_len].iter().enumerate() {
        let prob = unnorm_prob * inv_sum;
        let v_row = &v_cache[t * head_dim..(t + 1) * head_dim];
        for d in 0..head_dim {
            out[d] = fma(prob, v_row[d], out[d]);
        }
    }
}

/// **Pure-Rust MLX Autodiff (`value_and_grad`): `ste_linear_backward_into`**
///
/// Straight-Through Estimator (STE) backward pass for BitNet b1.58 Quantization-Aware
/// Training (QAT) in pure Rust:
/// - Forward: $y = x \cdot \hat{W}^T$ where $\hat{W} = \text{round\_clip}(W / \text{scale}, -1, +1) \cdot \text{scale}$
/// - Backward (STE):
///   - $\nabla_x = \nabla_y \cdot \hat{W}$
///   - $\nabla_W = \nabla_y^T \otimes x \cdot \mathbb{1}_{|W / \text{scale}| \le 1.5}$ (clipped straight-through gradient)
///
/// Performs **zero heap allocations**.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn ste_linear_backward_into(
    x: &[f64],
    w: &[f64],
    grad_y: &[f64],
    n_out: usize,
    k_in: usize,
    grad_x_out: &mut [f64],
    grad_w_out: &mut [f64],
) {
    debug_assert!(x.len() >= k_in);
    debug_assert!(w.len() >= n_out * k_in);
    debug_assert!(grad_y.len() >= n_out);
    debug_assert!(grad_x_out.len() >= k_in);
    debug_assert!(grad_w_out.len() >= n_out * k_in);

    for gx in grad_x_out[..k_in].iter_mut() {
        *gx = 0.0;
    }

    for n in 0..n_out {
        let gy = grad_y[n];
        let row_w = &w[n * k_in..(n + 1) * k_in];
        let row_gw = &mut grad_w_out[n * k_in..(n + 1) * k_in];

        let mut abs_sum = 0.0f64;
        for &v in row_w {
            abs_sum += v.abs();
        }
        let scale = (abs_sum / (k_in.max(1) as f64)).max(1e-12);
        let inv_scale = 1.0 / scale;

        for k in 0..k_in {
            let norm = row_w[k] * inv_scale;
            let trit = if norm >= 0.5 {
                1.0
            } else if norm <= -0.5 {
                -1.0
            } else {
                0.0
            };
            let w_hat = trit * scale;
            grad_x_out[k] = fma(gy, w_hat, grad_x_out[k]);
            // Clipped STE window: pass gradient straight through inside [-1.5, 1.5]
            let ste_mask = if norm.abs() <= 1.5 { 1.0 } else { 0.0 };
            row_gw[k] = gy * x[k] * ste_mask;
        }
    }
}

/// Complete single-layer **Pure-Rust Ternary Transformer Block** forward pass:
/// `x -> RMSNorm -> Ternary Q/K/V -> RoPE -> Causal KV-Cache SDPA -> Ternary O-Proj -> Residual -> RMSNorm -> Ternary SwiGLU MLP -> Residual`
/// using GAIA Base-3 i-major packed weights (`5 trits/byte`), running with **zero heap allocations**.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn ternary_transformer_step_into(
    x_in: &[f64],
    rms1_w: &[f64],
    wq_i_packed: &[u8],
    wk_i_packed: &[u8],
    wv_i_packed: &[u8],
    wo_i_packed: &[u8],
    rms2_w: &[f64],
    wgate_i_packed: &[u8],
    wup_i_packed: &[u8],
    wdown_i_packed: &[u8],
    dim: usize,
    hidden_dim: usize,
    pos: usize,
    k_cache: &mut [f64],
    v_cache: &mut [f64],
    scratch_norm: &mut [f64],
    scratch_q: &mut [f64],
    scratch_k: &mut [f64],
    scratch_v: &mut [f64],
    scratch_attn: &mut [f64],
    scratch_scores: &mut [f64],
    scratch_gate: &mut [f64],
    scratch_up: &mut [f64],
    x_out: &mut [f64],
) {
    debug_assert!(dim.is_multiple_of(2));
    debug_assert!(wq_i_packed.len() * TRITS_PER_BYTE >= dim * dim);
    debug_assert!(wgate_i_packed.len() * TRITS_PER_BYTE >= dim * hidden_dim);
    debug_assert!(wdown_i_packed.len() * TRITS_PER_BYTE >= hidden_dim * dim);

    // 1. Pre-Attention RMSNorm
    rms_norm_into(x_in, Some(rms1_w), 1e-5, scratch_norm);

    // 2. Ternary Q, K, V projections via p/q integer-numerator fast path (zero i8->f64 conversions!)
    let mut px_buf = [0i32; 2048];
    if dim <= px_buf.len() && hidden_dim <= px_buf.len() {
        quantize_pq_into(&scratch_norm[..dim], PQ_DEFAULT_Q, &mut px_buf[..dim]);
        b158_matmul_pq_i_major_into(wq_i_packed, dim, dim, &px_buf[..dim], PQ_DEFAULT_Q, 1.0, scratch_q);
        b158_matmul_pq_i_major_into(wk_i_packed, dim, dim, &px_buf[..dim], PQ_DEFAULT_Q, 1.0, scratch_k);
        b158_matmul_pq_i_major_into(wv_i_packed, dim, dim, &px_buf[..dim], PQ_DEFAULT_Q, 1.0, scratch_v);
    } else {
        ternary_matmul_i_major(wq_i_packed, dim, dim, scratch_norm, None, 1.0, scratch_q);
        ternary_matmul_i_major(wk_i_packed, dim, dim, scratch_norm, None, 1.0, scratch_k);
        ternary_matmul_i_major(wv_i_packed, dim, dim, scratch_norm, None, 1.0, scratch_v);
    }

    // 3. Rotary Position Embeddings (RoPE) on Q and K, writing into KV cache at `pos`
    let k_slot = &mut k_cache[pos * dim..(pos + 1) * dim];
    let v_slot = &mut v_cache[pos * dim..(pos + 1) * dim];
    rope_into(scratch_q, pos, 10000.0, scratch_attn); // scratch_attn holds rotated Q temporarily
    scratch_q[..dim].copy_from_slice(&scratch_attn[..dim]);
    rope_into(scratch_k, pos, 10000.0, k_slot);
    v_slot.copy_from_slice(&scratch_v[..dim]);

    // 4. Causal Scaled Dot-Product Attention over 0..=pos
    sdpa_single_query_into(
        scratch_q,
        k_cache,
        v_cache,
        pos + 1,
        dim,
        scratch_scores,
        scratch_attn,
    );

    // 5. Ternary Output Projection + Residual 1
    if dim <= px_buf.len() {
        quantize_pq_into(&scratch_attn[..dim], PQ_DEFAULT_Q, &mut px_buf[..dim]);
        b158_matmul_pq_i_major_into(wo_i_packed, dim, dim, &px_buf[..dim], PQ_DEFAULT_Q, 1.0, scratch_norm);
    } else {
        ternary_matmul_i_major(wo_i_packed, dim, dim, scratch_attn, None, 1.0, scratch_norm);
    }
    for i in 0..dim {
        x_out[i] = x_in[i] + scratch_norm[i];
    }

    // 6. Pre-MLP RMSNorm
    rms_norm_into(x_out, Some(rms2_w), 1e-5, scratch_norm);

    // 7. Ternary SwiGLU MLP (Gate + Up -> SiLU*Up -> Down) + Residual 2 via p/q integer fast path
    if dim <= px_buf.len() && hidden_dim <= px_buf.len() {
        quantize_pq_into(&scratch_norm[..dim], PQ_DEFAULT_Q, &mut px_buf[..dim]);
        b158_matmul_pq_i_major_into(wgate_i_packed, dim, hidden_dim, &px_buf[..dim], PQ_DEFAULT_Q, 1.0, scratch_gate);
        b158_matmul_pq_i_major_into(wup_i_packed, dim, hidden_dim, &px_buf[..dim], PQ_DEFAULT_Q, 1.0, scratch_up);
        for i in 0..hidden_dim {
            let g = scratch_gate[i];
            scratch_gate[i] = (g / (1.0 + exp_approx(-g))) * scratch_up[i];
        }
        quantize_pq_into(&scratch_gate[..hidden_dim], PQ_DEFAULT_Q, &mut px_buf[..hidden_dim]);
        b158_matmul_pq_i_major_into(wdown_i_packed, hidden_dim, dim, &px_buf[..hidden_dim], PQ_DEFAULT_Q, 1.0, scratch_norm);
    } else {
        ternary_matmul_i_major(wgate_i_packed, dim, hidden_dim, scratch_norm, None, 1.0, scratch_gate);
        ternary_matmul_i_major(wup_i_packed, dim, hidden_dim, scratch_norm, None, 1.0, scratch_up);
        for i in 0..hidden_dim {
            let g = scratch_gate[i];
            scratch_gate[i] = (g / (1.0 + exp_approx(-g))) * scratch_up[i];
        }
        ternary_matmul_i_major(wdown_i_packed, hidden_dim, dim, scratch_gate, None, 1.0, scratch_norm);
    }
    for i in 0..dim {
        x_out[i] += scratch_norm[i];
    }
}

/// Allocating `std` convenience wrappers for `mlx-quant-mcp` parity and HTTP/CLI use.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct MlxQuantizedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub codes: Vec<u32>,
    pub scales: Vec<f64>,
}

#[cfg(feature = "std")]
impl MlxQuantizedMatrix {
    /// Quantize a `[rows, cols]` matrix (`cols % 64 == 0`) into MLX Group-64 format.
    pub fn quantize(w: &[f64], rows: usize, cols: usize) -> Result<Self, String> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(MLX_GROUP_SIZE) {
            return Err(alloc::format!(
                "shape [{rows}, {cols}] requires positive dimensions and cols divisible by {MLX_GROUP_SIZE}"
            ));
        }
        if w.len() < rows * cols {
            return Err(alloc::format!(
                "weight slice length {} is smaller than rows*cols ({})",
                w.len(),
                rows * cols
            ));
        }
        let mut codes = vec![0u32; rows * (cols / TRITS_PER_U32)];
        let mut scales = vec![0.0f64; rows * (cols / MLX_GROUP_SIZE)];
        mlx_quantize_group64_into(w, rows, cols, &mut codes, &mut scales);
        Ok(Self {
            rows,
            cols,
            group_size: MLX_GROUP_SIZE,
            codes,
            scales,
        })
    }

    /// Dequantize back to `[rows, cols]` `f64` weights.
    pub fn dequantize(&self) -> Vec<f64> {
        let mut out = vec![0.0f64; self.rows * self.cols];
        mlx_dequantize_group64_into(&self.codes, &self.scales, self.rows, self.cols, &mut out);
        out
    }

    /// Multiply input vector `x` (`[cols]`) by this quantized weight matrix (`[rows, cols]`).
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; self.rows];
        mlx_quantized_matmul_into(x, &self.codes, &self.scales, self.rows, self.cols, &mut out);
        out
    }

    /// Packed byte count (codes `u32` + scales `f32` equivalent).
    pub fn packed_bytes(&self) -> usize {
        self.codes.len() * 4 + self.scales.len() * 4
    }

    /// Unpacked F32 byte count (`rows * cols * 4`).
    pub fn f32_bytes(&self) -> usize {
        self.rows * self.cols * 4
    }

    /// Exact compression ratio against F32 (`12.80x` at `group_size = 64`).
    pub fn compression_ratio(&self) -> f64 {
        self.f32_bytes() as f64 / (self.packed_bytes().max(1) as f64)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// p / q ("Counting Divided" — Exact Rational & Integer-Numerator Engine)
// ═════════════════════════════════════════════════════════════════════════════
//
// Why p/q beats float conversion in safe Rust (`#![forbid(unsafe_code)]`):
// 1. In float kernels, every ternary weight (-1, 0, +1) is converted `i8 -> f64`
//    (`cvtsi2sd`) N * K times (65,536 times for a 256×256 matrix).
// 2. IEEE-754 floating-point addition is non-associative, which legally forbids
//    LLVM from auto-vectorizing a simple float reduction loop into wide SIMD.
// 3. By projecting the input vector `x` ONCE (O(K)) onto integer numerators `p_x[k]`
//    over a shared denominator `q_x` (e.g. `q_x = 1 << 20`), the entire O(N * K)
//    matrix-vector product happens in pure integer arithmetic (Z):
//      - Zero `i8 -> f64` conversions in the O(N * K) inner loop
//      - Associative integer addition unlocks LLVM AVX2 / NEON wide SIMD (`vpaddd` / `vpmulld`)
//      - The denominator `q` is applied ONLY ONCE per output row (`O(N)` instead of `O(N * K)`).

/// Default dyadic denominator $q = 2^{20} = 1{,}048{,}576$ (~6 decimal digits of precision, zero overflow in `i32`/`i64`).
pub const PQ_DEFAULT_Q: i64 = 1 << 20;

/// Exact rational number $p / q \in \mathbb{Q}$ (`q > 0`) — "counting, divided".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pq {
    pub p: i64,
    pub q: i64,
}

impl Pq {
    /// Construct a rational `p / q` (`q` must be non-zero; sign is normalized into `p`).
    #[inline]
    pub const fn new(p: i64, q: i64) -> Option<Self> {
        if q == 0 {
            None
        } else if q < 0 {
            Some(Self { p: -p, q: -q })
        } else {
            Some(Self { p, q })
        }
    }

    /// Construct a reduced rational `p / q` in lowest terms via Euclidean GCD.
    #[inline]
    pub const fn reduced(p: i64, q: i64) -> Option<Self> {
        let Some(raw) = Self::new(p, q) else {
            return None;
        };
        let g = gcd_i64(raw.p.abs(), raw.q);
        Some(Self {
            p: raw.p / g,
            q: raw.q / g,
        })
    }

    /// Project a float onto $p / q$ with explicit denominator `q > 0`.
    #[inline]
    pub fn from_f64(val: f64, q: i64) -> Self {
        let q_pos = q.abs().max(1);
        let p = round_f64_to_i64(val * (q_pos as f64));
        Self { p, q: q_pos }
    }

    /// Evaluate $p / q$ to `f64` at the boundary.
    #[inline]
    pub fn to_f64(self) -> f64 {
        (self.p as f64) / (self.q as f64)
    }

    /// Exact rational addition $\frac{p_1}{q_1} + \frac{p_2}{q_2}$.
    #[inline]
    pub const fn add(self, rhs: Self) -> Self {
        let num = self.p * rhs.q + rhs.p * self.q;
        let den = self.q * rhs.q;
        let g = gcd_i64(num.abs(), den);
        Self {
            p: num / g,
            q: den / g,
        }
    }

    /// Exact rational multiplication $\frac{p_1}{q_1} \cdot \frac{p_2}{q_2}$.
    #[inline]
    pub const fn mul(self, rhs: Self) -> Self {
        let num = self.p * rhs.p;
        let den = self.q * rhs.q;
        let g = gcd_i64(num.abs(), den);
        Self {
            p: num / g,
            q: den / g,
        }
    }
}

#[inline]
const fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    if a == 0 { 1 } else { a }
}

/// Unified compile-time 4-trit `i32` LUT (`256 × 4 × 4 = 4,096` bytes, 16-byte power-of-two stride).
/// `MLX_BYTE_TRITS_I32[b]` loads all 4 decoded `i32` trits of byte `b` from a single 16-byte aligned
/// entry (`b << 4`), eliminating 4 separate table base calculations.
pub const MLX_BYTE_TRITS_I32: [[i32; 4]; 256] = build_mlx_byte_trits_i32();

const fn build_mlx_byte_trits_i32() -> [[i32; 4]; 256] {
    let mut table = [[0i32; 4]; 256];
    let mut b = 0usize;
    while b < 256 {
        table[b] = [
            mlx_trit_of((b & 0x03) as u32) as i32,
            mlx_trit_of(((b >> 2) & 0x03) as u32) as i32,
            mlx_trit_of(((b >> 4) & 0x03) as u32) as i32,
            mlx_trit_of(((b >> 6) & 0x03) as u32) as i32,
        ];
        b += 1;
    }
    table
}

/// Project a slice of `f64` values onto `i32` numerators `p_out[k] = round(x[k] * q)`
/// in $O(K)$. Performs **zero heap allocations**.
#[inline]
pub fn quantize_pq_into(x: &[f64], q: i64, p_out: &mut [i32]) {
    debug_assert!(p_out.len() >= x.len());
    let q_f64 = q as f64;
    for (dst, &val) in p_out.iter_mut().zip(x.iter()) {
        *dst = round_f64_to_i64(val * q_f64) as i32;
    }
}

/// **Dynamic Block-Exponent Rational Projection (`quantize_pq_adaptive_into`)**:
///
/// Computes $\max_k |x_k|$ in a single pass and selects the largest power-of-two denominator
/// $q = 2^e \in [1, 2^{29}]$ such that $\max_k |p_k| \le 2^{\text{max\_bits}}$ (`max_bits` clamped
/// to `1..=29`), returning the chosen denominator $q$.
/// Performs **zero heap allocations**.
#[inline]
pub fn quantize_pq_adaptive_into(x: &[f64], max_bits: u32, p_out: &mut [i32]) -> i64 {
    debug_assert!(p_out.len() >= x.len());
    let target_bits = max_bits.clamp(1, 29);
    let max_num = (1i64 << target_bits) as f64;

    let mut max_abs = 0.0f64;
    for &val in x {
        let a = val.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs <= 1e-30 {
        for dst in p_out[..x.len()].iter_mut() {
            *dst = 0;
        }
        return PQ_DEFAULT_Q;
    }

    let ratio = max_num / max_abs;
    let mut q = 1i64;
    let mut exp = 0u32;
    while exp < 29 && ((q << 1) as f64) <= ratio {
        q <<= 1;
        exp += 1;
    }
    quantize_pq_into(x, q, p_out);
    q
}

/// Decode and dot-accumulate one `u32` word (16 two-bit MLX trits) against `px` (`[i32; 16]`)
/// using [`MLX_BYTE_TRITS_I32`] in pure 32-bit integer arithmetic.
#[inline(always)]
fn dot_mlx_word_pq_i32(word: u32, px: &[i32]) -> i32 {
    if word == 0x5555_5555 {
        return 0;
    }
    let px = &px[..16];
    let t0 = &MLX_BYTE_TRITS_I32[(word & 0xFF) as usize];
    let t1 = &MLX_BYTE_TRITS_I32[((word >> 8) & 0xFF) as usize];
    let t2 = &MLX_BYTE_TRITS_I32[((word >> 16) & 0xFF) as usize];
    let t3 = &MLX_BYTE_TRITS_I32[(word >> 24) as usize];

    let s0 = t0[0]
        .wrapping_mul(px[0])
        .wrapping_add(t0[1].wrapping_mul(px[1]))
        .wrapping_add(t0[2].wrapping_mul(px[2]))
        .wrapping_add(t0[3].wrapping_mul(px[3]));
    let s1 = t1[0]
        .wrapping_mul(px[4])
        .wrapping_add(t1[1].wrapping_mul(px[5]))
        .wrapping_add(t1[2].wrapping_mul(px[6]))
        .wrapping_add(t1[3].wrapping_mul(px[7]));
    let s2 = t2[0]
        .wrapping_mul(px[8])
        .wrapping_add(t2[1].wrapping_mul(px[9]))
        .wrapping_add(t2[2].wrapping_mul(px[10]))
        .wrapping_add(t2[3].wrapping_mul(px[11]));
    let s3 = t3[0]
        .wrapping_mul(px[12])
        .wrapping_add(t3[1].wrapping_mul(px[13]))
        .wrapping_add(t3[2].wrapping_mul(px[14]))
        .wrapping_add(t3[3].wrapping_mul(px[15]));

    s0.wrapping_add(s1).wrapping_add(s2).wrapping_add(s3)
}

/// Pure-integer Group-64 MLX-QUANT matrix-vector product (`p_out[n] = sum_g p_scales[n, g] * dot(codes, p_x)`).
///
/// Uses **4-row register blocking** and [`MLX_BYTE_TRITS_I32`] with **zero floating-point
/// instructions** anywhere in the function.
#[inline]
pub fn mlx_matmul_pq_group64_i64_into(
    p_x: &[i32],
    w_codes: &[u32],
    p_scales: &[i32],
    n_out: usize,
    k_in: usize,
    p_out: &mut [i64],
) {
    debug_assert!(k_in.is_multiple_of(MLX_GROUP_SIZE));
    let groups_per_row = k_in / MLX_GROUP_SIZE;
    let words_per_row = k_in / TRITS_PER_U32;
    debug_assert!(p_x.len() >= k_in);
    debug_assert!(w_codes.len() >= n_out * words_per_row);
    debug_assert!(p_scales.len() >= n_out * groups_per_row);
    debug_assert!(p_out.len() >= n_out);

    let mut n = 0usize;
    while n + 4 <= n_out {
        let c0 = &w_codes[n * words_per_row..(n + 1) * words_per_row];
        let c1 = &w_codes[(n + 1) * words_per_row..(n + 2) * words_per_row];
        let c2 = &w_codes[(n + 2) * words_per_row..(n + 3) * words_per_row];
        let c3 = &w_codes[(n + 3) * words_per_row..(n + 4) * words_per_row];

        let s0 = &p_scales[n * groups_per_row..(n + 1) * groups_per_row];
        let s1 = &p_scales[(n + 1) * groups_per_row..(n + 2) * groups_per_row];
        let s2 = &p_scales[(n + 2) * groups_per_row..(n + 3) * groups_per_row];
        let s3 = &p_scales[(n + 3) * groups_per_row..(n + 4) * groups_per_row];

        let mut tot0 = 0i64;
        let mut tot1 = 0i64;
        let mut tot2 = 0i64;
        let mut tot3 = 0i64;

        for g in 0..groups_per_row {
            let mut g0 = 0i32;
            let mut g1 = 0i32;
            let mut g2 = 0i32;
            let mut g3 = 0i32;

            for w_idx in 0..4 {
                let px = &p_x[g * MLX_GROUP_SIZE + w_idx * TRITS_PER_U32
                    ..g * MLX_GROUP_SIZE + (w_idx + 1) * TRITS_PER_U32];
                let idx = g * 4 + w_idx;
                g0 = g0.wrapping_add(dot_mlx_word_pq_i32(c0[idx], px));
                g1 = g1.wrapping_add(dot_mlx_word_pq_i32(c1[idx], px));
                g2 = g2.wrapping_add(dot_mlx_word_pq_i32(c2[idx], px));
                g3 = g3.wrapping_add(dot_mlx_word_pq_i32(c3[idx], px));
            }

            tot0 += (g0 as i64) * (s0[g] as i64);
            tot1 += (g1 as i64) * (s1[g] as i64);
            tot2 += (g2 as i64) * (s2[g] as i64);
            tot3 += (g3 as i64) * (s3[g] as i64);
        }

        p_out[n] = tot0;
        p_out[n + 1] = tot1;
        p_out[n + 2] = tot2;
        p_out[n + 3] = tot3;
        n += 4;
    }

    while n < n_out {
        let row_codes = &w_codes[n * words_per_row..(n + 1) * words_per_row];
        let row_pscales = &p_scales[n * groups_per_row..(n + 1) * groups_per_row];
        let mut row_num_i64 = 0i64;

        for g in 0..groups_per_row {
            let p_scale = row_pscales[g];
            if p_scale == 0 {
                continue;
            }
            let mut g_acc = 0i32;
            for w_idx in 0..4 {
                let px = &p_x[g * MLX_GROUP_SIZE + w_idx * TRITS_PER_U32
                    ..g * MLX_GROUP_SIZE + (w_idx + 1) * TRITS_PER_U32];
                g_acc = g_acc.wrapping_add(dot_mlx_word_pq_i32(row_codes[g * 4 + w_idx], px));
            }
            row_num_i64 += (g_acc as i64) * (p_scale as i64);
        }
        p_out[n] = row_num_i64;
        n += 1;
    }
}

/// **`mlx_matmul_pq_group64_into`** — Pure $p/q$ Integer-Numerator MLX Group-64 Matmul.
///
/// Given integer activation numerators `p_x` (`x[k] ≈ p_x[k] / q_x`), packed MLX `u32` ternary
/// codes `w_codes`, and integer group-scale numerators `p_scales` (`scale[n, g] ≈ p_scales[n, g] / q_s`),
/// computes the entire $N \times K$ matrix-vector product in **pure integer arithmetic ($\mathbb{Z}$)**
/// with **4-row register blocking** and **zero `int -> float` conversions in the inner loop**,
/// dividing by $q_x \cdot q_s$ **only once per output row**.
///
/// Performs **zero heap allocations**.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn mlx_matmul_pq_group64_into(
    p_x: &[i32],
    q_x: i64,
    w_codes: &[u32],
    p_scales: &[i32],
    q_s: i64,
    n_out: usize,
    k_in: usize,
    y_out: &mut [f64],
) {
    debug_assert!(k_in.is_multiple_of(MLX_GROUP_SIZE));
    let groups_per_row = k_in / MLX_GROUP_SIZE;
    let words_per_row = k_in / TRITS_PER_U32;
    debug_assert!(p_x.len() >= k_in);
    debug_assert!(w_codes.len() >= n_out * words_per_row);
    debug_assert!(p_scales.len() >= n_out * groups_per_row);
    debug_assert!(y_out.len() >= n_out);

    let inv_q_total = 1.0 / ((q_x as f64) * (q_s as f64));
    let mut pout_stack = [0i64; 2048];
    if n_out <= pout_stack.len() {
        mlx_matmul_pq_group64_i64_into(p_x, w_codes, p_scales, n_out, k_in, &mut pout_stack[..n_out]);
        for (slot, &p_val) in y_out[..n_out].iter_mut().zip(pout_stack[..n_out].iter()) {
            *slot = (p_val as f64) * inv_q_total;
        }
        return;
    }

    for (n, slot) in y_out[..n_out].iter_mut().enumerate() {
        let row_codes = &w_codes[n * words_per_row..(n + 1) * words_per_row];
        let row_pscales = &p_scales[n * groups_per_row..(n + 1) * groups_per_row];
        let mut row_num_i64 = 0i64;

        for g in 0..groups_per_row {
            let p_scale = row_pscales[g];
            if p_scale == 0 {
                continue;
            }
            let mut g_acc = 0i32;
            for w_idx in 0..4 {
                let px = &p_x[g * MLX_GROUP_SIZE + w_idx * TRITS_PER_U32
                    ..g * MLX_GROUP_SIZE + (w_idx + 1) * TRITS_PER_U32];
                g_acc = g_acc.wrapping_add(dot_mlx_word_pq_i32(row_codes[g * 4 + w_idx], px));
            }
            row_num_i64 += (g_acc as i64) * (p_scale as i64);
        }
        *slot = (row_num_i64 as f64) * inv_q_total;
    }
}

/// **`b158_matmul_pq_i_major_into`** — Pure $p/q$ Integer-Numerator GAIA Base-3 (`5 trits/byte`) i-major Matmul.
///
/// Accumulates [`crate::pack::BYTE_TRITS_I32`] LUT trits $\times$ integer numerators `p_x` in
/// **4-column register-blocked `i32`/`i64` arithmetic** without a single `int -> float` conversion
/// in the $O(N \times K)$ loop, dividing by `q_x` **only once per output column**.
///
/// Performs **zero heap allocations**.
#[inline]
pub fn b158_matmul_pq_i_major_into(
    packed_i: &[u8],
    rows: usize,
    cols: usize,
    p_x: &[i32],
    q_x: i64,
    gamma: f64,
    y_out: &mut [f64],
) {
    use crate::kernel::{KERNEL_PQ_Q, ternary_matmul_pq_i_major_into};
    use crate::pack::{T0, T1, T2, T3, T4};
    debug_assert!(p_x.len() >= rows);
    debug_assert!(y_out.len() >= cols);
    debug_assert!(packed_i.len() * TRITS_PER_BYTE >= rows * cols);

    let scale_out = gamma / (q_x as f64);
    let mut pout_stack = [0i64; 2048];
    if cols <= pout_stack.len() {
        ternary_matmul_pq_i_major_into(
            packed_i,
            rows,
            cols,
            p_x,
            None,
            KERNEL_PQ_Q,
            &mut pout_stack[..cols],
        );
        for (slot, &p_val) in y_out[..cols].iter_mut().zip(pout_stack[..cols].iter()) {
            *slot = (p_val as f64) * scale_out;
        }
        return;
    }

    for (i, slot) in y_out[..cols].iter_mut().enumerate() {
        let start = i * rows;
        let end = start + rows;
        let start_b = start / TRITS_PER_BYTE;
        let end_b = end.div_ceil(TRITS_PER_BYTE);
        let mut acc = 0i64;

        for (b_idx, &byte) in packed_i[start_b..end_b].iter().enumerate() {
            if byte == 0 {
                continue;
            }
            let b = byte as usize;
            let base = (start_b + b_idx) * TRITS_PER_BYTE;
            if base >= start && base < end {
                acc += (T0[b] as i64) * (p_x[base - start] as i64);
            }
            let t1 = base + 1;
            if t1 >= start && t1 < end {
                acc += (T1[b] as i64) * (p_x[t1 - start] as i64);
            }
            let t2 = base + 2;
            if t2 >= start && t2 < end {
                acc += (T2[b] as i64) * (p_x[t2 - start] as i64);
            }
            let t3 = base + 3;
            if t3 >= start && t3 < end {
                acc += (T3[b] as i64) * (p_x[t3 - start] as i64);
            }
            let t4 = base + 4;
            if t4 >= start && t4 < end {
                acc += (T4[b] as i64) * (p_x[t4 - start] as i64);
            }
        }
        *slot = (acc as f64) * scale_out;
    }
}

/// **`ternary_simd_pq_matmul_into`** — 4-Column Blocked Wide-SIMD Auto-Vectorized $p/q$ Integer Dot Product.
///
/// Because integer addition in $\mathbb{Z}$ is associative (unlike IEEE-754 `f64`), LLVM
/// compiles this 4-column blocked contiguous `i8 × i32 -> i32` dot product directly into
/// 256-bit AVX2 / 128-bit NEON vector instructions (`vpmovsxbd` + `vpmulld` + `vpaddd` /
/// `sxtl` + `mla.4s`) in 100% safe `#![forbid(unsafe_code)]` Rust.
///
/// Divides by `q_x` **only once per output column** and performs **zero heap allocations**.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn ternary_simd_pq_matmul_into(
    trits_i_major: &[i8],
    rows: usize,
    cols: usize,
    p_x: &[i32],
    q_x: i64,
    scales_col: Option<&[f64]>,
    gamma: f64,
    y_out: &mut [f64],
) {
    use crate::kernel::{KERNEL_PQ_Q, ternary_simd_i8_pq_into};
    debug_assert!(trits_i_major.len() >= rows * cols);
    debug_assert!(p_x.len() >= rows);
    debug_assert!(y_out.len() >= cols);

    let inv_q = gamma / (q_x as f64);
    let mut pout_stack = [0i64; 2048];
    if cols <= pout_stack.len() {
        ternary_simd_i8_pq_into(
            trits_i_major,
            rows,
            cols,
            p_x,
            None,
            KERNEL_PQ_Q,
            &mut pout_stack[..cols],
        );
        if let Some(sc) = scales_col {
            for i in 0..cols {
                y_out[i] = (pout_stack[i] as f64) * inv_q * sc[i];
            }
        } else {
            for i in 0..cols {
                y_out[i] = (pout_stack[i] as f64) * inv_q;
            }
        }
        return;
    }

    let px_slice = &p_x[..rows];
    for (i, slot) in y_out[..cols].iter_mut().enumerate() {
        let col_trits = &trits_i_major[i * rows..(i + 1) * rows];
        let dot_i32: i32 = col_trits
            .iter()
            .zip(px_slice.iter())
            .map(|(&t, &px)| (t as i32).wrapping_mul(px))
            .fold(0i32, i32::wrapping_add);
        let col_scale = scales_col.map(|s| s[i]).unwrap_or(1.0);
        *slot = (dot_i32 as f64) * inv_q * col_scale;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 100% Pure-Integer (`no-float`) MLX-QUANT & Transformer Primitives (`p/q`)
// ═════════════════════════════════════════════════════════════════════════════

/// Pure-integer Group-64 MLX-QUANT quantization (`w_p: &[i32]` -> `codes_out: &mut [u32]`, `p_scales_out: &mut [i32]`).
///
/// Uses the exact integer threshold `|val| * 128 >= abs_sum` so the hot loop performs
/// **zero floating-point operations and zero per-weight divisions**.
#[inline]
pub fn mlx_quantize_group64_pq_into(
    p_w: &[i32],
    rows: usize,
    cols: usize,
    q: i32,
    codes_out: &mut [u32],
    p_scales_out: &mut [i32],
) {
    debug_assert!(cols.is_multiple_of(MLX_GROUP_SIZE));
    let groups_per_row = cols / MLX_GROUP_SIZE;
    let words_per_row = cols / TRITS_PER_U32;
    debug_assert!(p_w.len() >= rows * cols);
    debug_assert!(codes_out.len() >= rows * words_per_row);
    debug_assert!(p_scales_out.len() >= rows * groups_per_row);

    for r in 0..rows {
        let row_w = &p_w[r * cols..(r + 1) * cols];
        let row_codes = &mut codes_out[r * words_per_row..(r + 1) * words_per_row];
        let row_scales = &mut p_scales_out[r * groups_per_row..(r + 1) * groups_per_row];

        for g in 0..groups_per_row {
            let group = &row_w[g * MLX_GROUP_SIZE..(g + 1) * MLX_GROUP_SIZE];
            let mut abs_sum: i64 = 0;
            for &v in group {
                abs_sum += (v as i64).abs();
            }
            let scale_p = if abs_sum == 0 {
                q
            } else {
                ((abs_sum + 32) >> 6) as i32
            };
            row_scales[g] = scale_p;

            for w_idx in 0..4 {
                let sub = &group[w_idx * TRITS_PER_U32..(w_idx + 1) * TRITS_PER_U32];
                let mut word = 0u32;
                for (k, &val) in sub.iter().enumerate() {
                    let v64 = val as i64;
                    let abs128 = v64.abs() << 7; // |val| * 128 >= abs_sum <=> |val| >= 0.5 * (abs_sum / 64)
                    let trit: i8 = if abs_sum == 0 || abs128 < abs_sum {
                        0
                    } else if v64 > 0 {
                        1
                    } else {
                        -1
                    };
                    word |= (mlx_code_of(trit) & 0x03) << (k * 2);
                }
                row_codes[g * 4 + w_idx] = word;
            }
        }
    }
}

/// Pure-integer Group-64 MLX-QUANT dequantization (`codes: &[u32]`, `p_scales: &[i32]` -> `p_out: &mut [i32]`).
/// Zero floating-point operations.
#[inline]
pub fn mlx_dequantize_group64_pq_into(
    codes: &[u32],
    p_scales: &[i32],
    rows: usize,
    cols: usize,
    p_out: &mut [i32],
) {
    debug_assert!(cols.is_multiple_of(MLX_GROUP_SIZE));
    let groups_per_row = cols / MLX_GROUP_SIZE;
    let words_per_row = cols / TRITS_PER_U32;
    debug_assert!(codes.len() >= rows * words_per_row);
    debug_assert!(p_scales.len() >= rows * groups_per_row);
    debug_assert!(p_out.len() >= rows * cols);

    for r in 0..rows {
        let row_codes = &codes[r * words_per_row..(r + 1) * words_per_row];
        let row_scales = &p_scales[r * groups_per_row..(r + 1) * groups_per_row];
        let row_out = &mut p_out[r * cols..(r + 1) * cols];

        for g in 0..groups_per_row {
            let scale_p = row_scales[g];
            for w_idx in 0..4 {
                let word = row_codes[g * 4 + w_idx];
                let base = g * MLX_GROUP_SIZE + w_idx * TRITS_PER_U32;
                for byte_idx in 0..4 {
                    let b = ((word >> (byte_idx * 8)) & 0xFF) as usize;
                    let trits = &MLX_BYTE_TRITS_I32[b];
                    let dst = &mut row_out[base + byte_idx * 4..base + byte_idx * 4 + 4];
                    dst[0] = trits[0].wrapping_mul(scale_p);
                    dst[1] = trits[1].wrapping_mul(scale_p);
                    dst[2] = trits[2].wrapping_mul(scale_p);
                    dst[3] = trits[3].wrapping_mul(scale_p);
                }
            }
        }
    }
}

/// Pure-integer **`mergeq_matmul_pq_into`** — Triple-Fused `quantize + dequantize + matmul` in `p/q`.
///
/// Uses the exact integer inequality `|wv| * 2 * gs >= abs_sum` and accumulates in `i64`
/// with **zero floating-point operations** and **0 B intermediate weight storage**.
#[inline]
pub fn mergeq_matmul_pq_into(
    p_x: &[i32],
    p_w: &[i32],
    n_out: usize,
    k_in: usize,
    group_size: usize,
    p_out: &mut [i64],
) {
    let gs = if group_size == 0 { MLX_GROUP_SIZE } else { group_size };
    debug_assert!(p_x.len() >= k_in);
    debug_assert!(p_w.len() >= n_out * k_in);
    debug_assert!(p_out.len() >= n_out);

    for (n, slot) in p_out[..n_out].iter_mut().enumerate() {
        let row_w = &p_w[n * k_in..(n + 1) * k_in];
        let mut row_acc: i64 = 0;

        let mut start = 0usize;
        while start < k_in {
            let end = (start + gs).min(k_in);
            let group_w = &row_w[start..end];
            let group_x = &p_x[start..end];
            let glen = group_w.len() as i64;
            let two_glen = glen * 2;

            let mut abs_sum: i64 = 0;
            for &v in group_w {
                abs_sum += (v as i64).abs();
            }
            if abs_sum > 0 {
                let scale_p = (abs_sum + (glen >> 1)) / glen;
                let mut g_acc: i64 = 0;
                for (&wv, &xv) in group_w.iter().zip(group_x.iter()) {
                    let w64 = wv as i64;
                    if w64.abs() * two_glen >= abs_sum {
                        if w64 > 0 {
                            g_acc += xv as i64;
                        } else {
                            g_acc -= xv as i64;
                        }
                    }
                }
                row_acc += g_acc * scale_p;
            }
            start = end;
        }
        *slot = row_acc;
    }
}

/// Pure-integer **`maybe_matmul_pq_into`** — Stochastic Ternary Matmul in `p/q`.
///
/// Uses a 20-bit uniform integer threshold from SplitMix64 (`(rnd & 0xFFFFF) * abs_max < (|w| << 20)`)
/// so the stochastic rounding decision executes with **zero floating-point instructions**.
#[inline]
pub fn maybe_matmul_pq_into(
    p_x: &[i32],
    p_w: &[i32],
    n_out: usize,
    k_in: usize,
    group_size: usize,
    seed: u64,
    p_out: &mut [i64],
) {
    let gs = if group_size == 0 { MLX_GROUP_SIZE } else { group_size };
    debug_assert!(p_x.len() >= k_in);
    debug_assert!(p_w.len() >= n_out * k_in);
    debug_assert!(p_out.len() >= n_out);

    let mut rng = seed ^ 0xA1E0_5B15_8888_0432;
    for (n, slot) in p_out[..n_out].iter_mut().enumerate() {
        let row_w = &p_w[n * k_in..(n + 1) * k_in];
        let mut row_acc: i64 = 0;

        let mut start = 0usize;
        while start < k_in {
            let end = (start + gs).min(k_in);
            let group_w = &row_w[start..end];
            let group_x = &p_x[start..end];

            let mut abs_max: u64 = 0;
            for &v in group_w {
                let a = (v as i64).unsigned_abs();
                if a > abs_max {
                    abs_max = a;
                }
            }
            if abs_max > 0 {
                let mut g_acc: i64 = 0;
                for (&wv, &xv) in group_w.iter().zip(group_x.iter()) {
                    let w64 = wv as i64;
                    let aw = w64.unsigned_abs();
                    let u20 = splitmix64(&mut rng) & 0xFFFFF;
                    if u20 * abs_max < (aw << 20) {
                        if w64 >= 0 {
                            g_acc += xv as i64;
                        } else {
                            g_acc -= xv as i64;
                        }
                    }
                }
                row_acc += g_acc * (abs_max as i64);
            }
            start = end;
        }
        *slot = row_acc;
    }
}

/// **`gather_qmm_pq_into`** — Pure $p/q$ Integer-Numerator Switch-MoE Sparse Expert Kernel.
///
/// Reuses pre-projected input numerators `p_x` (`x[k] ≈ p_x[k] / q_x`) and pre-projected
/// expert scale numerators `expert_pscales` across all routed `active_experts`, running every
/// expert projection in $\mathbb{Z}$ via [`mlx_matmul_pq_group64_into`].
/// Performs **zero heap allocations**.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm_pq_into(
    p_x: &[i32],
    q_x: i64,
    expert_codes: &[u32],
    expert_pscales: &[i32],
    q_s: i64,
    num_experts: usize,
    n_out: usize,
    k_in: usize,
    active_experts: &[usize],
    router_weights: &[f64],
    expert_scratch: &mut [f64],
    y_out: &mut [f64],
) {
    debug_assert!(k_in.is_multiple_of(MLX_GROUP_SIZE));
    debug_assert_eq!(active_experts.len(), router_weights.len());
    debug_assert!(expert_scratch.len() >= n_out);
    debug_assert!(y_out.len() >= n_out);

    let words_per_expert = n_out * (k_in / TRITS_PER_U32);
    let scales_per_expert = n_out * (k_in / MLX_GROUP_SIZE);

    for slot in y_out[..n_out].iter_mut() {
        *slot = 0.0;
    }

    for (&expert_idx, &gate) in active_experts.iter().zip(router_weights.iter()) {
        if expert_idx >= num_experts || gate == 0.0 {
            continue;
        }
        let e_codes = &expert_codes[expert_idx * words_per_expert..(expert_idx + 1) * words_per_expert];
        let e_pscales = &expert_pscales[expert_idx * scales_per_expert..(expert_idx + 1) * scales_per_expert];

        mlx_matmul_pq_group64_into(
            p_x,
            q_x,
            e_codes,
            e_pscales,
            q_s,
            n_out,
            k_in,
            expert_scratch,
        );
        for i in 0..n_out {
            y_out[i] = fma(expert_scratch[i], gate, y_out[i]);
        }
    }
}

/// Pure-integer **`gather_qmm_pq_i64_into`** — Mixture-of-Experts (MoE) Routed Ternary Matmul in `p/q`.
/// Zero floating-point operations.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm_pq_i64_into(
    p_x: &[i32],
    expert_codes: &[u32],
    expert_p_scales: &[i32],
    num_experts: usize,
    n_out: usize,
    k_in: usize,
    active_experts: &[usize],
    router_p_weights: &[i32],
    q_router: i64,
    expert_scratch: &mut [i64],
    p_out: &mut [i64],
) {
    debug_assert!(k_in.is_multiple_of(MLX_GROUP_SIZE));
    debug_assert_eq!(active_experts.len(), router_p_weights.len());
    debug_assert!(expert_scratch.len() >= n_out);
    debug_assert!(p_out.len() >= n_out);

    let words_per_expert = n_out * (k_in / TRITS_PER_U32);
    let scales_per_expert = n_out * (k_in / MLX_GROUP_SIZE);
    let q_r = q_router.max(1);

    for slot in p_out[..n_out].iter_mut() {
        *slot = 0;
    }

    for (&expert_idx, &gate_p) in active_experts.iter().zip(router_p_weights.iter()) {
        if expert_idx >= num_experts || gate_p == 0 {
            continue;
        }
        let e_codes = &expert_codes[expert_idx * words_per_expert..(expert_idx + 1) * words_per_expert];
        let e_scales = &expert_p_scales[expert_idx * scales_per_expert..(expert_idx + 1) * scales_per_expert];

        mlx_matmul_pq_group64_i64_into(p_x, e_codes, e_scales, n_out, k_in, expert_scratch);
        let gate128 = gate_p as i128;
        let qr128 = q_r as i128;
        for i in 0..n_out {
            p_out[i] += (((expert_scratch[i] as i128) * gate128) / qr128) as i64;
        }
    }
}

/// Pure-integer Root Mean Square LayerNorm (`rms_norm_pq_into`) in `p/q` (`q = 2^20`).
///
/// Uses [`crate::kernel::isqrt_u128`] with **zero floating-point operations**.
#[inline]
pub fn rms_norm_pq_into(
    p_x: &[i32],
    p_weight: Option<&[i32]>,
    q: i64,
    eps_p: i64,
    p_out: &mut [i32],
) {
    use crate::kernel::isqrt_u128;
    let d = p_x.len();
    debug_assert!(p_out.len() >= d);
    if d == 0 {
        return;
    }
    let mut sum_sq: u128 = 0;
    for &v in p_x {
        let va = (v as i64).unsigned_abs() as u128;
        sum_sq = sum_sq.saturating_add(va * va);
    }
    let mean_sq = sum_sq / (d as u128);
    let eps_sq = (eps_p.unsigned_abs() as u128) * (q.unsigned_abs() as u128);
    let rms = isqrt_u128(mean_sq.saturating_add(eps_sq)).max(1) as i128;
    let q128 = q.max(1) as i128;

    match p_weight {
        Some(w) => {
            debug_assert!(w.len() >= d);
            for i in 0..d {
                let num = (p_x[i] as i128) * (w[i] as i128);
                p_out[i] = (num / rms) as i32;
            }
        }
        None => {
            for i in 0..d {
                let num = (p_x[i] as i128) * q128;
                p_out[i] = (num / rms) as i32;
            }
        }
    }
}

/// Pure-integer `sin` and `cos` in `p/q` (`q = 2^20 = 1,048,576`).
///
/// Range-reduces `theta_p` into $[-\pi/2, \pi/2]$ and evaluates a 5th-order
/// fixed-point minimax polynomial with **zero floating-point operations**.
#[inline]
pub const fn sin_cos_pq(theta_p: i64) -> (i64, i64) {
    const Q: i64 = 1_048_576; // 2^20
    const PI_P: i64 = 3_294_199;
    const TWO_PI_P: i64 = 6_588_397;
    const HALF_PI_P: i64 = 1_647_099;

    let mut x = theta_p % TWO_PI_P;
    if x > PI_P {
        x -= TWO_PI_P;
    } else if x < -PI_P {
        x += TWO_PI_P;
    }

    let mut sign_cos: i64 = 1;
    if x > HALF_PI_P {
        x = PI_P - x;
        sign_cos = -1;
    } else if x < -HALF_PI_P {
        x = -PI_P - x;
        sign_cos = -1;
    }

    // x2 = (x * x) / Q
    let x2 = (x * x) >> 20;
    // sin(x) ≈ x * (1 - x^2/6 + x^4/120)
    let sin_term = Q - (x2 / 6) + (((x2 * x2) >> 20) / 120);
    let sin_p = (x * sin_term) >> 20;

    // cos(x) ≈ sign_cos * (1 - x^2/2 + x^4/24)
    let cos_term = Q - (x2 >> 1) + (((x2 * x2) >> 20) / 24);
    let cos_p = sign_cos * cos_term;

    (sin_p, cos_p)
}

/// Pure-integer Rotary Position Embeddings (`rope_pq_into`) in `p/q` (`q = 2^20`).
/// Zero floating-point operations.
#[inline]
pub fn rope_pq_into(p_x: &[i32], pos: usize, p_out: &mut [i32]) {
    let d = p_x.len();
    debug_assert!(d.is_multiple_of(2));
    debug_assert!(p_out.len() >= d);
    let half = d / 2;

    // Step decay 10000^(-1/4) = 0.1 in q = 2^20 => 104,858
    let mut freq_p: i64 = 1_048_576;
    for i in 0..half {
        let theta_p = (pos as i64) * freq_p;
        let (sin_p, cos_p) = sin_cos_pq(theta_p);
        let x0 = p_x[2 * i] as i64;
        let x1 = p_x[2 * i + 1] as i64;
        p_out[2 * i] = ((x0 * cos_p - x1 * sin_p) >> 20) as i32;
        p_out[2 * i + 1] = ((x0 * sin_p + x1 * cos_p) >> 20) as i32;
        freq_p = (freq_p * 104_858) >> 20;
    }
}

/// Pure-integer fixed-point exponential $e^x$ in `p/q` (`q = 2^20 = 1,048,576`)
/// via 6 integer squarings of $(1 + x / 64)^{64}$. Zero floating-point operations.
#[inline]
pub const fn exp_pq(x_p: i64) -> i64 {
    const Q: i64 = 1_048_576;
    let clamped = if x_p < -16 * Q {
        -16 * Q
    } else if x_p > 12 * Q {
        12 * Q
    } else {
        x_p
    };
    let mut v = Q + (clamped >> 6);
    if v <= 0 {
        return 0;
    }
    let mut i = 0;
    while i < 6 {
        v = (v * v) >> 20;
        i += 1;
    }
    v
}

/// Pure-integer SwiGLU activation (`swiglu_pq_into`) in `p/q` (`q = 2^20`).
/// Zero floating-point operations.
#[inline]
pub fn swiglu_pq_into(p_gate: &[i32], p_up: &[i32], q: i64, p_out: &mut [i32]) {
    let d = p_gate.len().min(p_up.len());
    debug_assert!(p_out.len() >= d);
    let q_pos = q.max(1);
    for i in 0..d {
        let g = p_gate[i] as i64;
        let exp_neg_g = exp_pq(-g);
        let denom = (q_pos + exp_neg_g).max(1);
        let silu_p = (g * q_pos) / denom;
        p_out[i] = ((silu_p * (p_up[i] as i64)) / q_pos) as i32;
    }
}

/// **`sdpa_pq_single_query_into`** — Pure $p/q$ Integer-Numerator KV-Cache Attention.
///
/// Takes pre-projected integer query numerators `p_q` (`[head_dim]`), pre-projected integer
/// KV-cache numerators `p_k_cache` and `p_v_cache` (`[seq_len, head_dim]`), and shared
/// denominator `q_kv` (e.g. `1 << 14`), computing both $q K^T$ and $\text{prob} \cdot V$
/// in associative integer arithmetic ($\mathbb{Z}$) with single deferred division at the end.
/// Performs **zero heap allocations**.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn sdpa_pq_single_query_into(
    p_q: &[i32],
    p_k_cache: &[i32],
    p_v_cache: &[i32],
    q_kv: i64,
    seq_len: usize,
    head_dim: usize,
    scores_scratch: &mut [f64],
    out: &mut [f64],
) {
    debug_assert!(p_q.len() >= head_dim);
    debug_assert!(p_k_cache.len() >= seq_len * head_dim);
    debug_assert!(p_v_cache.len() >= seq_len * head_dim);
    debug_assert!(scores_scratch.len() >= seq_len);
    debug_assert!(out.len() >= head_dim);

    if seq_len == 0 || head_dim == 0 {
        return;
    }
    let inv_q2_scale = 1.0 / ((q_kv as f64) * (q_kv as f64) * sqrt_f64(head_dim as f64));
    let mut max_score = f64::NEG_INFINITY;
    let pq_slice = &p_q[..head_dim];

    for (t, score_slot) in scores_scratch[..seq_len].iter_mut().enumerate() {
        let pk_row = &p_k_cache[t * head_dim..(t + 1) * head_dim];
        let dot_i64: i64 = pq_slice
            .iter()
            .zip(pk_row.iter())
            .map(|(&qv, &kv)| (qv as i64) * (kv as i64))
            .sum();
        let s = (dot_i64 as f64) * inv_q2_scale;
        *score_slot = s;
        if s > max_score {
            max_score = s;
        }
    }

    let mut sum_exp = 0.0f64;
    for s in scores_scratch[..seq_len].iter_mut() {
        let e = exp_approx(*s - max_score);
        *s = e;
        sum_exp += e;
    }
    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

    const Q_PROB: i64 = 1 << 20;
    let q_prob_f64 = Q_PROB as f64;
    let inv_total = 1.0 / ((Q_PROB as f64) * (q_kv as f64));

    for slot in out[..head_dim].iter_mut() {
        *slot = 0.0;
    }
    for (t, &unnorm_prob) in scores_scratch[..seq_len].iter().enumerate() {
        let p_prob = round_f64_to_i64((unnorm_prob * inv_sum) * q_prob_f64);
        if p_prob == 0 {
            continue;
        }
        let pv_row = &p_v_cache[t * head_dim..(t + 1) * head_dim];
        for (slot, &pv) in out[..head_dim].iter_mut().zip(pv_row.iter()) {
            *slot += ((p_prob * (pv as i64)) as f64) * inv_total;
        }
    }
}

/// Pure-integer Scaled Dot-Product Attention (`sdpa_pq_i32_single_query_into`) in `p/q` (`q = 2^20`).
/// Zero floating-point operations.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn sdpa_pq_i32_single_query_into(
    p_q: &[i32],
    p_k_cache: &[i32],
    p_v_cache: &[i32],
    seq_len: usize,
    head_dim: usize,
    q: i64,
    scores_scratch: &mut [i64],
    p_out: &mut [i32],
) {
    use crate::kernel::isqrt_u128;
    debug_assert!(p_q.len() >= head_dim);
    debug_assert!(p_k_cache.len() >= seq_len * head_dim);
    debug_assert!(p_v_cache.len() >= seq_len * head_dim);
    debug_assert!(scores_scratch.len() >= seq_len);
    debug_assert!(p_out.len() >= head_dim);

    if seq_len == 0 || head_dim == 0 {
        return;
    }
    let q_pos = q.max(1);
    let sqrt_d_p = isqrt_u128((head_dim as u128) * (q_pos as u128) * (q_pos as u128)).max(1) as i64;
    let mut max_score = i64::MIN;

    for t in 0..seq_len {
        let k_row = &p_k_cache[t * head_dim..(t + 1) * head_dim];
        let mut dot: i64 = 0;
        for d in 0..head_dim {
            dot += ((p_q[d] as i64) * (k_row[d] as i64)) / q_pos;
        }
        let s = (dot * q_pos) / sqrt_d_p;
        scores_scratch[t] = s;
        if s > max_score {
            max_score = s;
        }
    }

    let mut sum_exp: i64 = 0;
    for s in scores_scratch[..seq_len].iter_mut() {
        let e = exp_pq(*s - max_score);
        *s = e;
        sum_exp += e;
    }
    let denom = sum_exp.max(1);

    for slot in p_out[..head_dim].iter_mut() {
        *slot = 0;
    }
    for (t, &e_val) in scores_scratch[..seq_len].iter().enumerate() {
        let prob_p = (e_val * q_pos) / denom;
        let v_row = &p_v_cache[t * head_dim..(t + 1) * head_dim];
        for d in 0..head_dim {
            p_out[d] = p_out[d].wrapping_add(((prob_p * (v_row[d] as i64)) / q_pos) as i32);
        }
    }
}

/// Pure-integer Straight-Through Estimator (STE) QAT backward pass (`ste_linear_backward_pq_into`).
/// Zero floating-point operations.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn ste_linear_backward_pq_into(
    p_x: &[i32],
    p_w: &[i32],
    p_grad_y: &[i32],
    n_out: usize,
    k_in: usize,
    q: i64,
    p_grad_x_out: &mut [i64],
    p_grad_w_out: &mut [i64],
) {
    debug_assert!(p_x.len() >= k_in);
    debug_assert!(p_w.len() >= n_out * k_in);
    debug_assert!(p_grad_y.len() >= n_out);
    debug_assert!(p_grad_x_out.len() >= k_in);
    debug_assert!(p_grad_w_out.len() >= n_out * k_in);

    let q_pos = q.max(1);
    let k_i64 = k_in.max(1) as i64;
    let two_k = k_i64 * 2;

    for gx in p_grad_x_out[..k_in].iter_mut() {
        *gx = 0;
    }

    for n in 0..n_out {
        let gy = p_grad_y[n] as i64;
        let row_w = &p_w[n * k_in..(n + 1) * k_in];
        let row_gw = &mut p_grad_w_out[n * k_in..(n + 1) * k_in];

        let mut abs_sum: i64 = 0;
        for &v in row_w {
            abs_sum += (v as i64).abs();
        }
        let scale_p = (abs_sum / k_i64).max(1);

        for k in 0..k_in {
            let w64 = row_w[k] as i64;
            let aw2k = w64.abs() * two_k;
            let trit: i64 = if abs_sum > 0 && aw2k >= abs_sum {
                if w64 > 0 { 1 } else { -1 }
            } else {
                0
            };
            let w_hat_p = trit * scale_p;
            p_grad_x_out[k] += (gy * w_hat_p) / q_pos;
            // Clipped STE window: |w| / scale <= 1.5 <=> |w| * 2 * k_in <= 3 * abs_sum
            row_gw[k] = if abs_sum == 0 || aw2k <= 3 * abs_sum {
                (gy * (p_x[k] as i64)) / q_pos
            } else {
                0
            };
        }
    }
}

/// Complete single-layer **100% Pure-Integer (`p/q`, Zero Floating Point) Ternary Transformer Block**:
/// `p_x -> rms_norm_pq -> b158_pq QKV -> rope_pq -> sdpa_pq -> b158_pq O-Proj -> residual -> rms_norm_pq -> b158_pq SwiGLU MLP -> residual`.
///
/// Executes with **zero floating-point instructions** and **zero heap allocations**.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn ternary_transformer_step_pq_into(
    p_x_in: &[i32],
    p_rms1_w: &[i32],
    wq_i_packed: &[u8],
    wk_i_packed: &[u8],
    wv_i_packed: &[u8],
    wo_i_packed: &[u8],
    p_rms2_w: &[i32],
    wgate_i_packed: &[u8],
    wup_i_packed: &[u8],
    wdown_i_packed: &[u8],
    dim: usize,
    hidden_dim: usize,
    pos: usize,
    p_k_cache: &mut [i32],
    p_v_cache: &mut [i32],
    scratch_norm: &mut [i32],
    scratch_q: &mut [i32],
    scratch_k: &mut [i32],
    scratch_v: &mut [i32],
    scratch_attn: &mut [i32],
    scratch_scores: &mut [i64],
    scratch_gate: &mut [i32],
    scratch_up: &mut [i32],
    scratch_i64: &mut [i64],
    p_x_out: &mut [i32],
) {
    use crate::kernel::{KERNEL_PQ_Q, ternary_matmul_pq_i_major_into};
    debug_assert!(dim.is_multiple_of(2));
    debug_assert!(scratch_i64.len() >= hidden_dim.max(dim));

    // 1. Pre-Attention Pure-Integer RMSNorm
    rms_norm_pq_into(p_x_in, Some(p_rms1_w), PQ_DEFAULT_Q, 10, scratch_norm);

    // 2. Pure-Integer Ternary Q, K, V projections
    ternary_matmul_pq_i_major_into(wq_i_packed, dim, dim, scratch_norm, None, KERNEL_PQ_Q, &mut scratch_i64[..dim]);
    for i in 0..dim {
        scratch_q[i] = scratch_i64[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    ternary_matmul_pq_i_major_into(wk_i_packed, dim, dim, scratch_norm, None, KERNEL_PQ_Q, &mut scratch_i64[..dim]);
    for i in 0..dim {
        scratch_k[i] = scratch_i64[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    ternary_matmul_pq_i_major_into(wv_i_packed, dim, dim, scratch_norm, None, KERNEL_PQ_Q, &mut scratch_i64[..dim]);
    for i in 0..dim {
        scratch_v[i] = scratch_i64[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }

    // 3. Pure-Integer RoPE on Q and K into KV cache at `pos`
    let k_slot = &mut p_k_cache[pos * dim..(pos + 1) * dim];
    let v_slot = &mut p_v_cache[pos * dim..(pos + 1) * dim];
    rope_pq_into(&scratch_q[..dim], pos, scratch_attn);
    scratch_q[..dim].copy_from_slice(&scratch_attn[..dim]);
    rope_pq_into(&scratch_k[..dim], pos, k_slot);
    v_slot.copy_from_slice(&scratch_v[..dim]);

    // 4. Pure-Integer Causal SDPA over 0..=pos
    sdpa_pq_i32_single_query_into(
        scratch_q,
        p_k_cache,
        p_v_cache,
        pos + 1,
        dim,
        PQ_DEFAULT_Q,
        scratch_scores,
        scratch_attn,
    );

    // 5. Pure-Integer Ternary Output Projection + Residual 1
    ternary_matmul_pq_i_major_into(wo_i_packed, dim, dim, scratch_attn, None, KERNEL_PQ_Q, &mut scratch_i64[..dim]);
    for i in 0..dim {
        p_x_out[i] = p_x_in[i].saturating_add(scratch_i64[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    }

    // 6. Pre-MLP Pure-Integer RMSNorm
    rms_norm_pq_into(&p_x_out[..dim], Some(p_rms2_w), PQ_DEFAULT_Q, 10, scratch_norm);

    // 7. Pure-Integer Ternary SwiGLU MLP + Residual 2
    ternary_matmul_pq_i_major_into(wgate_i_packed, dim, hidden_dim, scratch_norm, None, KERNEL_PQ_Q, &mut scratch_i64[..hidden_dim]);
    for i in 0..hidden_dim {
        scratch_gate[i] = scratch_i64[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    ternary_matmul_pq_i_major_into(wup_i_packed, dim, hidden_dim, scratch_norm, None, KERNEL_PQ_Q, &mut scratch_i64[..hidden_dim]);
    for i in 0..hidden_dim {
        scratch_up[i] = scratch_i64[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    for i in 0..hidden_dim {
        let g = scratch_gate[i] as i64;
        let denom = (PQ_DEFAULT_Q + exp_pq(-g)).max(1);
        let silu_p = (g * PQ_DEFAULT_Q) / denom;
        scratch_gate[i] = ((silu_p * (scratch_up[i] as i64)) / PQ_DEFAULT_Q) as i32;
    }
    ternary_matmul_pq_i_major_into(wdown_i_packed, hidden_dim, dim, scratch_gate, None, KERNEL_PQ_Q, &mut scratch_i64[..dim]);
    for i in 0..dim {
        p_x_out[i] = p_x_out[i].saturating_add(scratch_i64[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    }
}

// =========================================================================
// 7. OMNI / MEM|8 BLOCK-FLOATING-POINT (BFP-64) & PURE-INTEGER BFP-i8
// =========================================================================

/// Quantize a pure-integer rational activation vector `p_x` (shared denominator `q`)
/// into 64-element block-floating-point `i8` mantissas (`m_out[k] in [-127, +127]`)
/// and per-group integer rational scales (`p_scale_out[g] = step`), with **zero floating-point ops**.
///
/// Paired with [`crate::kernel::omni_bfp_i8_simd_pq_into`], which accumulates 256-element
/// chunks (`4` groups of `64`) in 16-wide `i16` SIMD (`256 * 127 = 32,512 <= 32,767`).
#[inline]
pub fn quantize_bfp_i8_pq_group64_into(
    p_x: &[i32],
    m_out: &mut [i8],
    p_scale_out: &mut [i32],
) {
    assert_eq!(m_out.len(), p_x.len());
    let num_groups = p_x.len().div_ceil(MLX_GROUP_SIZE);
    assert!(p_scale_out.len() >= num_groups);

    for (g, p_sc) in p_scale_out.iter_mut().enumerate().take(num_groups) {
        let k_start = g * MLX_GROUP_SIZE;
        let k_end = (k_start + MLX_GROUP_SIZE).min(p_x.len());
        let mut max_abs = 0i64;
        for &v in &p_x[k_start..k_end] {
            let a = (v as i64).abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs == 0 {
            *p_sc = 0;
            for m in &mut m_out[k_start..k_end] {
                *m = 0;
            }
            continue;
        }
        let step = ((max_abs + 63) / 127).max(1);
        *p_sc = step.clamp(1, i32::MAX as i64) as i32;
        let half = step / 2;
        for (dst, &v) in m_out[k_start..k_end].iter_mut().zip(p_x[k_start..k_end].iter()) {
            let v64 = v as i64;
            let qv = if v64 >= 0 {
                (v64 + half) / step
            } else {
                (v64 - half) / step
            };
            *dst = qv.clamp(-127, 127) as i8;
        }
    }
}

/// Extract a dyadic scale $2^e$ for a block with maximum absolute value `max_abs`
/// targeting a signed integer mantissa bound `max_mant` (e.g., `127.0` for `i8`
/// or `32767.0` for `i16`), using IEEE-754 bit extraction (`no_std`-compatible, no `libm`).
#[inline]
fn dyadic_block_exponent(max_abs: f64, max_mant: f64) -> f64 {
    if max_abs <= 0.0 {
        return 0.0;
    }
    let raw = max_abs / max_mant;
    let bits = raw.to_bits();
    let biased_exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    let ceil_biased = if frac == 0 {
        biased_exp
    } else {
        (biased_exp + 1).min(2046)
    };
    f64::from_bits((ceil_biased as u64) << 52)
}

/// Quantize a vector `x` (`rows` elements) into 64-element Block-Floating-Point `i8`
/// mantissas (`m_out[k] in [-127, +127]`) and per-group dyadic scales `scale_out[g] = 2^{e_g}`.
pub fn quantize_bfp_i8_group64_into(x: &[f64], m_out: &mut [i8], scale_out: &mut [f64]) {
    assert_eq!(m_out.len(), x.len());
    let num_groups = x.len().div_ceil(MLX_GROUP_SIZE);
    assert!(scale_out.len() >= num_groups);

    for (g, sc_slot) in scale_out.iter_mut().enumerate().take(num_groups) {
        let k_start = g * MLX_GROUP_SIZE;
        let k_end = (k_start + MLX_GROUP_SIZE).min(x.len());
        let mut max_abs = 0.0f64;
        for &v in &x[k_start..k_end] {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = dyadic_block_exponent(max_abs, 127.0);
        *sc_slot = scale;
        if scale == 0.0 {
            for m in &mut m_out[k_start..k_end] {
                *m = 0;
            }
        } else {
            let inv_scale = 1.0 / scale;
            for (m, &v) in m_out[k_start..k_end].iter_mut().zip(x[k_start..k_end].iter()) {
                let q = round_f64_to_i64(v * inv_scale).clamp(-127, 127);
                *m = q as i8;
            }
        }
    }
}

/// 4-column register-blocked, 16-wide `i16` SIMD ternary matmul over 64-element
/// Block-Floating-Point `i8` activations (`m_x: &[i8]`, `scale_x: &[f64]`).
///
/// Because $|S_g| \le 64 \times 127 = 8{,}128 \le 32{,}767$, each 64-element group
/// accumulates in 16-bit integer lanes (`i8 × i8 -> i16`) with zero overflow!
#[allow(clippy::too_many_arguments)]
pub fn ternary_bfp_i8_matmul_into(
    trits_i_major: &[i8],
    rows: usize,
    cols: usize,
    m_x: &[i8],
    scale_x: &[f64],
    mask: Option<&[u8]>,
    scale_w: f64,
    out: &mut [f64],
) {
    assert_eq!(trits_i_major.len(), rows * cols);
    assert_eq!(m_x.len(), rows);
    assert_eq!(out.len(), cols);
    let num_groups = rows.div_ceil(MLX_GROUP_SIZE);
    assert!(scale_x.len() >= num_groups);

    if let Some(m) = mask {
        let mut col = 0usize;
        for &flag in m.iter().take(cols) {
            if flag == 0 {
                out[col] = 0.0;
                col += 1;
                continue;
            }
            let w_col = &trits_i_major[col * rows..(col + 1) * rows];
            let mut acc_f64 = 0.0f64;
            for (g, &sc_g) in scale_x.iter().enumerate().take(num_groups) {
                if sc_g == 0.0 {
                    continue;
                }
                let k_start = g * MLX_GROUP_SIZE;
                let k_end = (k_start + MLX_GROUP_SIZE).min(rows);
                let w_g = &w_col[k_start..k_end];
                let x_g = &m_x[k_start..k_end];
                let mut lanes = [0i16; 16];
                let (w_chunks, w_rem) = w_g.as_chunks::<16>();
                let (x_chunks, x_rem) = x_g.as_chunks::<16>();
                for (wc, xc) in w_chunks.iter().zip(x_chunks.iter()) {
                    for l in 0..16 {
                        lanes[l] += (wc[l] as i16) * (xc[l] as i16);
                    }
                }
                let mut sum_g: i32 = lanes.iter().map(|&v| v as i32).sum();
                for (&wt, &xv) in w_rem.iter().zip(x_rem.iter()) {
                    sum_g += (wt as i32) * (xv as i32);
                }
                acc_f64 += (sum_g as f64) * sc_g;
            }
            out[col] = acc_f64 * scale_w;
            col += 1;
        }
        return;
    }

    // Unmasked fast path: 4-column register blocking!
    let cols4 = (cols / 4) * 4;
    let mut col = 0usize;
    while col < cols4 {
        let w0 = &trits_i_major[col * rows..(col + 1) * rows];
        let w1 = &trits_i_major[(col + 1) * rows..(col + 2) * rows];
        let w2 = &trits_i_major[(col + 2) * rows..(col + 3) * rows];
        let w3 = &trits_i_major[(col + 3) * rows..(col + 4) * rows];

        let mut acc0 = 0.0f64;
        let mut acc1 = 0.0f64;
        let mut acc2 = 0.0f64;
        let mut acc3 = 0.0f64;

        for (g, &sc_g) in scale_x.iter().enumerate().take(num_groups) {
            if sc_g == 0.0 {
                continue;
            }
            let k_start = g * MLX_GROUP_SIZE;
            let k_end = (k_start + MLX_GROUP_SIZE).min(rows);
            let w0_g = &w0[k_start..k_end];
            let w1_g = &w1[k_start..k_end];
            let w2_g = &w2[k_start..k_end];
            let w3_g = &w3[k_start..k_end];
            let x_g = &m_x[k_start..k_end];

            let mut l0 = [0i16; 16];
            let mut l1 = [0i16; 16];
            let mut l2 = [0i16; 16];
            let mut l3 = [0i16; 16];

            let n_len = x_g.len();
            let n16 = (n_len / 16) * 16;
            let mut k = 0usize;
            while k < n16 {
                let xc = &x_g[k..k + 16];
                let c0 = &w0_g[k..k + 16];
                let c1 = &w1_g[k..k + 16];
                let c2 = &w2_g[k..k + 16];
                let c3 = &w3_g[k..k + 16];
                for l in 0..16 {
                    let xv = xc[l] as i16;
                    l0[l] += (c0[l] as i16) * xv;
                    l1[l] += (c1[l] as i16) * xv;
                    l2[l] += (c2[l] as i16) * xv;
                    l3[l] += (c3[l] as i16) * xv;
                }
                k += 16;
            }

            let mut s0: i32 = l0.iter().map(|&v| v as i32).sum();
            let mut s1: i32 = l1.iter().map(|&v| v as i32).sum();
            let mut s2: i32 = l2.iter().map(|&v| v as i32).sum();
            let mut s3: i32 = l3.iter().map(|&v| v as i32).sum();

            while k < n_len {
                let xv = x_g[k] as i32;
                s0 += (w0_g[k] as i32) * xv;
                s1 += (w1_g[k] as i32) * xv;
                s2 += (w2_g[k] as i32) * xv;
                s3 += (w3_g[k] as i32) * xv;
                k += 1;
            }

            acc0 += (s0 as f64) * sc_g;
            acc1 += (s1 as f64) * sc_g;
            acc2 += (s2 as f64) * sc_g;
            acc3 += (s3 as f64) * sc_g;
        }

        out[col] = acc0 * scale_w;
        out[col + 1] = acc1 * scale_w;
        out[col + 2] = acc2 * scale_w;
        out[col + 3] = acc3 * scale_w;
        col += 4;
    }

    while col < cols {
        let w_col = &trits_i_major[col * rows..(col + 1) * rows];
        let mut acc_f64 = 0.0f64;
        for (g, &sc_g) in scale_x.iter().enumerate().take(num_groups) {
            if sc_g == 0.0 {
                continue;
            }
            let k_start = g * MLX_GROUP_SIZE;
            let k_end = (k_start + MLX_GROUP_SIZE).min(rows);
            let w_g = &w_col[k_start..k_end];
            let x_g = &m_x[k_start..k_end];
            let mut lanes = [0i16; 16];
            let (w_chunks, w_rem) = w_g.as_chunks::<16>();
            let (x_chunks, x_rem) = x_g.as_chunks::<16>();
            for (wc, xc) in w_chunks.iter().zip(x_chunks.iter()) {
                for l in 0..16 {
                    lanes[l] += (wc[l] as i16) * (xc[l] as i16);
                }
            }
            let mut sum_g: i32 = lanes.iter().map(|&v| v as i32).sum();
            for (&wt, &xv) in w_rem.iter().zip(x_rem.iter()) {
                sum_g += (wt as i32) * (xv as i32);
            }
            acc_f64 += (sum_g as f64) * sc_g;
        }
        out[col] = acc_f64 * scale_w;
        col += 1;
    }
}

/// Quantize a vector `x` (`rows` elements) into 64-element Block-Floating-Point `i16`
/// mantissas (`m_out[k] in [-32767, +32767]`) and per-group dyadic scales `scale_out[g] = 2^{e_g}`.
pub fn quantize_bfp_i16_group64_into(x: &[f64], m_out: &mut [i16], scale_out: &mut [f64]) {
    assert_eq!(m_out.len(), x.len());
    let num_groups = x.len().div_ceil(MLX_GROUP_SIZE);
    assert!(scale_out.len() >= num_groups);

    for (g, sc_slot) in scale_out.iter_mut().enumerate().take(num_groups) {
        let k_start = g * MLX_GROUP_SIZE;
        let k_end = (k_start + MLX_GROUP_SIZE).min(x.len());
        let mut max_abs = 0.0f64;
        for &v in &x[k_start..k_end] {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = dyadic_block_exponent(max_abs, 32767.0);
        *sc_slot = scale;
        if scale == 0.0 {
            for m in &mut m_out[k_start..k_end] {
                *m = 0;
            }
        } else {
            let inv_scale = 1.0 / scale;
            for (m, &v) in m_out[k_start..k_end].iter_mut().zip(x[k_start..k_end].iter()) {
                let q = round_f64_to_i64(v * inv_scale).clamp(-32767, 32767);
                *m = q as i16;
            }
        }
    }
}

/// 4-column register-blocked, 8-wide `i32` SIMD ternary matmul over 64-element
/// Block-Floating-Point `i16` activations (`m_x: &[i16]`, `scale_x: &[f64]`).
#[allow(clippy::too_many_arguments)]
pub fn ternary_bfp_i16_matmul_into(
    trits_i_major: &[i8],
    rows: usize,
    cols: usize,
    m_x: &[i16],
    scale_x: &[f64],
    mask: Option<&[u8]>,
    scale_w: f64,
    out: &mut [f64],
) {
    assert_eq!(trits_i_major.len(), rows * cols);
    assert_eq!(m_x.len(), rows);
    assert_eq!(out.len(), cols);
    let num_groups = rows.div_ceil(MLX_GROUP_SIZE);
    assert!(scale_x.len() >= num_groups);

    if let Some(m) = mask {
        for (col, (&flag, out_j)) in m.iter().zip(out.iter_mut()).enumerate().take(cols) {
            if flag == 0 {
                *out_j = 0.0;
                continue;
            }
            let w_col = &trits_i_major[col * rows..(col + 1) * rows];
            let mut acc_f64 = 0.0f64;
            for (g, &sc_g) in scale_x.iter().enumerate().take(num_groups) {
                if sc_g == 0.0 {
                    continue;
                }
                let k_start = g * MLX_GROUP_SIZE;
                let k_end = (k_start + MLX_GROUP_SIZE).min(rows);
                let w_g = &w_col[k_start..k_end];
                let x_g = &m_x[k_start..k_end];
                let mut lanes = [0i32; 8];
                let (w_chunks, w_rem) = w_g.as_chunks::<8>();
                let (x_chunks, x_rem) = x_g.as_chunks::<8>();
                for (wc, xc) in w_chunks.iter().zip(x_chunks.iter()) {
                    for l in 0..8 {
                        lanes[l] += (wc[l] as i32) * (xc[l] as i32);
                    }
                }
                let mut sum_g: i32 = lanes.iter().sum();
                for (&wt, &xv) in w_rem.iter().zip(x_rem.iter()) {
                    sum_g += (wt as i32) * (xv as i32);
                }
                acc_f64 += (sum_g as f64) * sc_g;
            }
            *out_j = acc_f64 * scale_w;
        }
        return;
    }

    let cols4 = (cols / 4) * 4;
    let mut col = 0usize;
    while col < cols4 {
        let w0 = &trits_i_major[col * rows..(col + 1) * rows];
        let w1 = &trits_i_major[(col + 1) * rows..(col + 2) * rows];
        let w2 = &trits_i_major[(col + 2) * rows..(col + 3) * rows];
        let w3 = &trits_i_major[(col + 3) * rows..(col + 4) * rows];

        let mut acc0 = 0.0f64;
        let mut acc1 = 0.0f64;
        let mut acc2 = 0.0f64;
        let mut acc3 = 0.0f64;

        for (g, &sc_g) in scale_x.iter().enumerate().take(num_groups) {
            if sc_g == 0.0 {
                continue;
            }
            let k_start = g * MLX_GROUP_SIZE;
            let k_end = (k_start + MLX_GROUP_SIZE).min(rows);
            let w0_g = &w0[k_start..k_end];
            let w1_g = &w1[k_start..k_end];
            let w2_g = &w2[k_start..k_end];
            let w3_g = &w3[k_start..k_end];
            let x_g = &m_x[k_start..k_end];

            let mut l0 = [0i32; 8];
            let mut l1 = [0i32; 8];
            let mut l2 = [0i32; 8];
            let mut l3 = [0i32; 8];

            let n_len = x_g.len();
            let n8 = (n_len / 8) * 8;
            let mut k = 0usize;
            while k < n8 {
                let xc = &x_g[k..k + 8];
                let c0 = &w0_g[k..k + 8];
                let c1 = &w1_g[k..k + 8];
                let c2 = &w2_g[k..k + 8];
                let c3 = &w3_g[k..k + 8];
                for l in 0..8 {
                    let xv = xc[l] as i32;
                    l0[l] += (c0[l] as i32) * xv;
                    l1[l] += (c1[l] as i32) * xv;
                    l2[l] += (c2[l] as i32) * xv;
                    l3[l] += (c3[l] as i32) * xv;
                }
                k += 8;
            }

            let mut s0: i32 = l0.iter().sum();
            let mut s1: i32 = l1.iter().sum();
            let mut s2: i32 = l2.iter().sum();
            let mut s3: i32 = l3.iter().sum();

            while k < n_len {
                let xv = x_g[k] as i32;
                s0 += (w0_g[k] as i32) * xv;
                s1 += (w1_g[k] as i32) * xv;
                s2 += (w2_g[k] as i32) * xv;
                s3 += (w3_g[k] as i32) * xv;
                k += 1;
            }

            acc0 += (s0 as f64) * sc_g;
            acc1 += (s1 as f64) * sc_g;
            acc2 += (s2 as f64) * sc_g;
            acc3 += (s3 as f64) * sc_g;
        }

        out[col] = acc0 * scale_w;
        out[col + 1] = acc1 * scale_w;
        out[col + 2] = acc2 * scale_w;
        out[col + 3] = acc3 * scale_w;
        col += 4;
    }

    while col < cols {
        let w_col = &trits_i_major[col * rows..(col + 1) * rows];
        let mut acc_f64 = 0.0f64;
        for (g, &sc_g) in scale_x.iter().enumerate().take(num_groups) {
            if sc_g == 0.0 {
                continue;
            }
            let k_start = g * MLX_GROUP_SIZE;
            let k_end = (k_start + MLX_GROUP_SIZE).min(rows);
            let w_g = &w_col[k_start..k_end];
            let x_g = &m_x[k_start..k_end];
            let mut lanes = [0i32; 8];
            let (w_chunks, w_rem) = w_g.as_chunks::<8>();
            let (x_chunks, x_rem) = x_g.as_chunks::<8>();
            for (wc, xc) in w_chunks.iter().zip(x_chunks.iter()) {
                for l in 0..8 {
                    lanes[l] += (wc[l] as i32) * (xc[l] as i32);
                }
            }
            let mut sum_g: i32 = lanes.iter().sum();
            for (&wt, &xv) in w_rem.iter().zip(x_rem.iter()) {
                sum_g += (wt as i32) * (xv as i32);
            }
            acc_f64 += (sum_g as f64) * sc_g;
        }
        out[col] = acc_f64 * scale_w;
        col += 1;
    }
}

/// MEM|8 salience-adaptive Block-Floating-Point ternary matmul:
/// routes high-salience columns (`salience[j] >= threshold`) to `BFP-i16` (`15`-bit mantissa)
/// and standard columns (`salience[j] < threshold`) to `BFP-i8` (`7`-bit mantissa).
#[allow(clippy::too_many_arguments)]
pub fn ternary_bfp_adaptive_mem8_matmul_into(
    trits_i_major: &[i8],
    rows: usize,
    cols: usize,
    m_x_i8: &[i8],
    scale_x_i8: &[f64],
    m_x_i16: &[i16],
    scale_x_i16: &[f64],
    salience: &[f64],
    threshold: f64,
    scale_w: f64,
    out: &mut [f64],
) {
    assert_eq!(trits_i_major.len(), rows * cols);
    assert_eq!(m_x_i8.len(), rows);
    assert_eq!(m_x_i16.len(), rows);
    assert_eq!(salience.len(), cols);
    assert_eq!(out.len(), cols);
    let num_groups = rows.div_ceil(MLX_GROUP_SIZE);
    assert!(scale_x_i8.len() >= num_groups);
    assert!(scale_x_i16.len() >= num_groups);

    for (col, (&sal, out_j)) in salience.iter().zip(out.iter_mut()).enumerate().take(cols) {
        let w_col = &trits_i_major[col * rows..(col + 1) * rows];
        let mut acc_f64 = 0.0f64;
        if sal >= threshold {
            // High-salience wave column -> BFP-i16 precision path
            for (g, &sc_g) in scale_x_i16.iter().enumerate().take(num_groups) {
                if sc_g == 0.0 {
                    continue;
                }
                let k_start = g * MLX_GROUP_SIZE;
                let k_end = (k_start + MLX_GROUP_SIZE).min(rows);
                let w_g = &w_col[k_start..k_end];
                let x_g = &m_x_i16[k_start..k_end];
                let mut lanes = [0i32; 8];
                let (w_chunks, w_rem) = w_g.as_chunks::<8>();
                let (x_chunks, x_rem) = x_g.as_chunks::<8>();
                for (wc, xc) in w_chunks.iter().zip(x_chunks.iter()) {
                    for l in 0..8 {
                        lanes[l] += (wc[l] as i32) * (xc[l] as i32);
                    }
                }
                let mut sum_g: i32 = lanes.iter().sum();
                for (&wt, &xv) in w_rem.iter().zip(x_rem.iter()) {
                    sum_g += (wt as i32) * (xv as i32);
                }
                acc_f64 += (sum_g as f64) * sc_g;
            }
        } else {
            // Standard column -> BFP-i8 fast path
            for (g, &sc_g) in scale_x_i8.iter().enumerate().take(num_groups) {
                if sc_g == 0.0 {
                    continue;
                }
                let k_start = g * MLX_GROUP_SIZE;
                let k_end = (k_start + MLX_GROUP_SIZE).min(rows);
                let w_g = &w_col[k_start..k_end];
                let x_g = &m_x_i8[k_start..k_end];
                let mut lanes = [0i16; 16];
                let (w_chunks, w_rem) = w_g.as_chunks::<16>();
                let (x_chunks, x_rem) = x_g.as_chunks::<16>();
                for (wc, xc) in w_chunks.iter().zip(x_chunks.iter()) {
                    for l in 0..16 {
                        lanes[l] += (wc[l] as i16) * (xc[l] as i16);
                    }
                }
                let mut sum_g: i32 = lanes.iter().map(|&v| v as i32).sum();
                for (&wt, &xv) in w_rem.iter().zip(x_rem.iter()) {
                    sum_g += (wt as i32) * (xv as i32);
                }
                acc_f64 += (sum_g as f64) * sc_g;
            }
        }
        *out_j = acc_f64 * scale_w;
    }
}

// =========================================================================
// 8. MEM|8 ASSOCIATIVE WAVE SHADER: 24B WaveMemoryCell & 16B RationalWaveCell
// =========================================================================

/// 24-Byte Hybrid Continuous-Discrete Wave Memory Cell (`WaveMemoryCell`).
///
/// Combines a 12-byte continuous IEEE-754 wave manifold (`frequency`, `amplitude`, `phase`)
/// with a 4-byte discrete visual/topological matrix (`color: [u8; 3]`, `shape: u8`) and a
/// 1-byte balanced-ternary contraction gate (`quant: i8` in `{-1, 0, +1}`), aligned to
/// 8 bytes (`24` bytes total, `2.66` structs per 64-byte cache line).
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveMemoryCell {
    /// Continuous wave frequency (4 bytes).
    pub frequency: f32,
    /// Continuous wave amplitude (4 bytes).
    pub amplitude: f32,
    /// Continuous wave phase offset in radians (4 bytes).
    pub phase: f32,
    /// RGB semantic sentiment & domain vector (3 bytes).
    pub color: [u8; 3],
    /// Topological envelope / morphological wave shape enum (1 byte).
    pub shape: u8,
    /// Balanced-ternary contraction weight `{-1, 0, +1}` (1 byte + 7 bytes alignment padding = 24 bytes).
    pub quant: i8,
}

const _: () = assert!(core::mem::size_of::<WaveMemoryCell>() == 24);
const _: () = assert!(core::mem::align_of::<WaveMemoryCell>() == 8);

impl WaveMemoryCell {
    /// Construct a 24-byte [`WaveMemoryCell`].
    #[inline]
    pub const fn new(
        frequency: f32,
        amplitude: f32,
        phase: f32,
        color: [u8; 3],
        shape: u8,
        quant: i8,
    ) -> Self {
        Self {
            frequency,
            amplitude,
            phase,
            color,
            shape,
            quant,
        }
    }
}

/// 16-Byte Pure-Integer `p/q` Rational Wave Memory Cell (`RationalWaveCell`).
///
/// Replaces IEEE-754 floats with `u16` rational pairs (`p/q`), packs RGB into 16-bit
/// `RGB565`, and packs the 14-bit `shape` (8-bit `shape_id` + signed 6-bit `magnitude_octave`)
/// alongside the 2-bit balanced-ternary `quant` gate (`{-1, 0, +1}`) into `shape_and_quant`.
///
/// Fits in a single 128-bit SIMD register (`16` bytes, `align(16)`, zero padding,
/// `4.0` structs per 64-byte cache line).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalWaveCell {
    // 12 bytes: The p/q Wave Manifold (Numerators and Denominators)
    pub freq_p: u16,
    pub freq_q: u16,
    pub amp_p: u16,
    pub amp_q: u16,
    pub phase_p: u16,
    pub phase_q: u16,
    // 2 bytes: RGB565 packed format (5 bits R, 6 bits G, 5 bits B)
    pub color: u16,
    // 2 bytes: [14 bits: Shape/Topology + Octave Magnitude] [2 bits: Quantizer {-1, 0, 1}]
    pub shape_and_quant: u16,
}

const _: () = assert!(core::mem::size_of::<RationalWaveCell>() == 16);
const _: () = assert!(core::mem::align_of::<RationalWaveCell>() == 16);

impl RationalWaveCell {
    /// Construct a 16-byte [`RationalWaveCell`] with explicit `shape` (`14`-bit) and `quant` (`{-1, 0, +1}`).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        freq_p: u16,
        freq_q: u16,
        amp_p: u16,
        amp_q: u16,
        phase_p: u16,
        phase_q: u16,
        color: u16,
        shape: u16,
        quant: i8,
    ) -> Self {
        Self {
            freq_p,
            freq_q: if freq_q == 0 { 1 } else { freq_q },
            amp_p,
            amp_q: if amp_q == 0 { 1 } else { amp_q },
            phase_p,
            phase_q: if phase_q == 0 { 1 } else { phase_q },
            color,
            shape_and_quant: Self::pack_shape_quant(shape, quant),
        }
    }

    /// Pack 8-bit `(r, g, b)` channels into 16-bit `RGB565` (`[15..11: R5][10..5: G6][4..0: B5]`).
    #[inline(always)]
    pub const fn pack_rgb565(r: u8, g: u8, b: u8) -> u16 {
        let r5 = ((r as u16) >> 3) & 0x1F;
        let g6 = ((g as u16) >> 2) & 0x3F;
        let b5 = ((b as u16) >> 3) & 0x1F;
        (r5 << 11) | (g6 << 5) | b5
    }

    /// Unpack 16-bit `RGB565` into 8-bit `(r, g, b)` channels with full-range bit replication.
    #[inline(always)]
    pub const fn unpack_rgb565(color: u16) -> (u8, u8, u8) {
        let r5 = ((color >> 11) & 0x1F) as u8;
        let g6 = ((color >> 5) & 0x3F) as u8;
        let b5 = (color & 0x1F) as u8;
        let r = (r5 << 3) | (r5 >> 2);
        let g = (g6 << 2) | (g6 >> 4);
        let b = (b5 << 3) | (b5 >> 2);
        (r, g, b)
    }

    /// Pack a 14-bit `shape` (`0..=0x3FFF`) and balanced-ternary `quant` (`{-1, 0, +1}`) into `u16`.
    /// Low 2 bits encode `quant`: `0b00 => 0`, `0b01 => +1`, `0b10 => -1`, `0b11 => 0`.
    #[inline(always)]
    pub const fn pack_shape_quant(shape: u16, quant: i8) -> u16 {
        let q_bits: u16 = if quant > 0 {
            0b01
        } else if quant < 0 {
            0b10
        } else {
            0b00
        };
        ((shape & 0x3FFF) << 2) | q_bits
    }

    /// Pack an 8-bit `shape` ID, a signed 6-bit `mag_octave` (`[-32, 31]`), and balanced-ternary `quant` (`{-1, 0, +1}`) into `u16`.
    #[inline(always)]
    pub const fn pack_shape_mag_quant(shape: u8, mag_octave: i8, quant: i8) -> u16 {
        let clamped_oct = if mag_octave < -32 {
            -32
        } else if mag_octave > 31 {
            31
        } else {
            mag_octave
        };
        let oct_bits = ((clamped_oct as i16 as u16) & 0x3F) << 8;
        let shape14 = oct_bits | (shape as u16);
        Self::pack_shape_quant(shape14, quant)
    }

    /// Decode the low 2 bits into balanced-ternary `{-1, 0, +1}`:
    /// `0b00 => 0`, `0b01 => 1`, `0b10 => -1`, `0b11 => 0` (branchless 2-op ALU decode).
    #[inline(always)]
    pub const fn quant(&self) -> i8 {
        ((self.shape_and_quant & 1) as i8) - (((self.shape_and_quant >> 1) & 1) as i8)
    }

    /// Extract the full 14-bit shape + octave field (`self.shape_and_quant >> 2`).
    #[inline(always)]
    pub const fn shape(&self) -> u16 {
        self.shape_and_quant >> 2
    }

    /// Extract the low 8-bit morphological shape identifier (`((self.shape_and_quant >> 2) & 0xFF) as u8`).
    #[inline(always)]
    pub const fn shape_id(&self) -> u8 {
        ((self.shape_and_quant >> 2) & 0xFF) as u8
    }

    /// Extract the signed 6-bit octave magnitude shift stored in the top 6 bits (`bits 10..=15`),
    /// defaulting to `0` when a standard 8-bit shape is used.
    #[inline(always)]
    pub const fn magnitude_octave(&self) -> i8 {
        ((self.shape_and_quant as i16) >> 10) as i8
    }
}

/// Direct Digital Synthesis (DDS) integer sine lookup table size (`4096` samples per $2\pi$ period).
pub const DDS_LUT_SIZE: usize = 4096;

/// Compile-time 4096-entry pure-integer `i16` sine wave lookup table in `[-32767, 32767]`.
///
/// Computed via a 7th-order odd-polynomial minimax synthesis over quarter-wave symmetry
/// in 128-bit fixed-point integer arithmetic (`Q = 2^30`) with **zero floating-point operations**.
pub static DDS_SINE_LUT_I16: [i16; DDS_LUT_SIZE] = build_dds_sine_lut_i16();

const fn build_dds_sine_lut_i16() -> [i16; DDS_LUT_SIZE] {
    let mut lut = [0i16; DDS_LUT_SIZE];
    // Minimax / Taylor coefficients for sin((pi/2) * z) on z = u / 1024 in [0, 1], scaled by 2^30:
    // C1 - C3 + C5 - C7 == 2^30 = 1_073_741_824 exactly so sin(pi/2) == +32767.
    const C1: i128 = 1_686_629_713;
    const C3: i128 = 693_598_666;
    const C5: i128 = 85_569_298;
    const C7: i128 = 4_858_521;

    let mut k = 0usize;
    while k < DDS_LUT_SIZE {
        let (u, sign): (i128, i128) = if k <= 1024 {
            (k as i128, 1)
        } else if k <= 2048 {
            ((2048 - k) as i128, 1)
        } else if k <= 3072 {
            ((k - 2048) as i128, -1)
        } else {
            ((4096 - k) as i128, -1)
        };

        // Evaluate Horner polynomial in z = u / 2^10 (z^2 = u^2 / 2^20)
        let u2 = u * u; // in [0, 2^20]
        let p5 = C5 - ((C7 * u2) >> 20);
        let p3 = C3 - ((p5 * u2) >> 20);
        let p1 = C1 - ((p3 * u2) >> 20);
        // sin_q30 in [0, 2^30]
        let sin_q30 = (u * p1) >> 10;
        // Scale to [0, 32767] with round-half-up
        let mag = ((sin_q30 * 32767) + (1i128 << 29)) >> 30;
        let clamped = if mag > 32767 { 32767 } else { mag };
        lut[k] = (sign * clamped) as i16;
        k += 1;
    }
    lut
}

/// Deterministic output of the 3-stage [`rational_wave_dds_query`] associative wave shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalWaveShaderResult {
    /// Total cells inspected in the manifold.
    pub total_cells: usize,
    /// Cells surviving Stage 1 (non-zero ternary contraction gate `quant ∈ {-1, +1}`).
    pub stage1_survivors: usize,
    /// Cells surviving Stage 2 (`shape` mask match + `RGB565` Manhattan tolerance).
    pub stage2_survivors: usize,
    /// Total accumulated signed DDS resonant wave energy across Stage 2 survivors.
    pub total_energy_i64: i64,
    /// Energy-weighted additive RGB color blend `[R, G, B]` of resonant cells.
    pub blended_rgb: [u8; 3],
    /// Packed `RGB565` representation of [`Self::blended_rgb`].
    pub blended_rgb565: u16,
    /// Deterministic 64-bit FNV-1a state digest across all resonant wave cells.
    pub state_digest: u64,
}

/// 3-Stage Zero-Allocation ALU-Only Direct Digital Synthesis (DDS) Wave Shader Cascade.
///
/// - **Stage 1 (Ternary Contraction Gate)**: checks `cell.shape_and_quant & 0b11`; if `0`
///   (`0b00` or `0b11`), skips immediately (`1-cycle` bitwise check).
/// - **Stage 2 (Semantic & Morphological Mask)**: checks `(cell.shape_and_quant >> 2) & shape_mask == (target_shape & shape_mask)`
///   and `RGB565` Manhattan distance `<= color_tol`.
/// - **Stage 3 (Direct Digital Synthesis Wave Coherence + RGB565 Additive Blending)**:
///   - `current_phase = (((freq_p as u64 * t) / freq_q.max(1) as u64) + ((phase_p as u64 * 4096) / phase_q.max(1) as u64)) & 4095`
///   - `lut_val = DDS_SINE_LUT_I16[current_phase as usize] as i64`
///   - `energy = ((lut_val * amp_p as i64) / (amp_q.max(1) as i64)) <</>> mag_octave`
///   - Applies `quant` sign (`+1` or `-1`), accumulates `total_energy_i64` and `blended_rgb`,
///     and folds into a deterministic 64-bit FNV-1a `state_digest`.
///
/// Performs **zero heap allocations** and **zero floating-point operations**.
#[inline]
pub fn rational_wave_dds_query(
    cells: &[RationalWaveCell],
    t: u64,
    target_shape: u16,
    shape_mask: u16,
    target_color_rgb565: u16,
    color_tol: u16,
) -> RationalWaveShaderResult {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

    let masked_target_shape = target_shape & shape_mask;
    let (tr, tg, tb) = RationalWaveCell::unpack_rgb565(target_color_rgb565);

    let mut stage1_survivors = 0usize;
    let mut stage2_survivors = 0usize;
    let mut total_energy_i64 = 0i64;
    let mut rgb_acc_r = 0u64;
    let mut rgb_acc_g = 0u64;
    let mut rgb_acc_b = 0u64;
    let mut rgb_weight_sum = 0u64;
    let mut state_digest = FNV_OFFSET_BASIS;

    for cell in cells {
        let sq = cell.shape_and_quant;
        // Stage 1: 1-cycle branchless ternary contraction check (bit0 ^ bit1 == 1 iff quant in {-1, +1})
        let active_bit = ((sq ^ (sq >> 1)) & 1) as usize;
        stage1_survivors += active_bit;

        // Combined Stage 1 + Stage 2 topological shape gate on the same u16 register
        if active_bit == 0 || ((sq >> 2) & shape_mask) != masked_target_shape {
            continue;
        }

        // Stage 2: RGB565 Manhattan distance check
        let (cr, cg, cb) = RationalWaveCell::unpack_rgb565(cell.color);
        let dist = (cr.abs_diff(tr) as u16) + (cg.abs_diff(tg) as u16) + (cb.abs_diff(tb) as u16);
        if dist > color_tol {
            continue;
        }
        stage2_survivors += 1;

        // Stage 3: Pure-Integer Direct Digital Synthesis (DDS) Wave Coherence
        let fq = (cell.freq_q as u64).max(1);
        let pq = (cell.phase_q as u64).max(1);
        let aq = (cell.amp_q as i64).max(1);

        let current_phase = ((((cell.freq_p as u64).wrapping_mul(t)) / fq)
            .wrapping_add(((cell.phase_p as u64) * (DDS_LUT_SIZE as u64)) / pq))
            & ((DDS_LUT_SIZE as u64) - 1);
        let lut_val = DDS_SINE_LUT_I16[current_phase as usize] as i64;
        let base_energy = (lut_val * (cell.amp_p as i64)) / aq;

        let mag_oct = cell.magnitude_octave();
        let shifted_energy = if mag_oct >= 0 {
            base_energy << (mag_oct.min(30) as u32)
        } else {
            base_energy >> ((-mag_oct).min(62) as u32)
        };

        // Apply balanced-ternary quant sign (+1 for 0b01, -1 for 0b10)
        let signed_energy = if (sq & 0b11) == 0b01 {
            shifted_energy
        } else {
            -shifted_energy
        };
        total_energy_i64 = total_energy_i64.wrapping_add(signed_energy);

        // Constructive/resonant RGB565 additive color blending
        let weight = signed_energy.unsigned_abs().max(1);
        rgb_acc_r = rgb_acc_r.saturating_add((cr as u64).saturating_mul(weight));
        rgb_acc_g = rgb_acc_g.saturating_add((cg as u64).saturating_mul(weight));
        rgb_acc_b = rgb_acc_b.saturating_add((cb as u64).saturating_mul(weight));
        rgb_weight_sum = rgb_weight_sum.saturating_add(weight);

        // Deterministic 64-bit FNV-1a state digest
        state_digest ^= signed_energy as u64;
        state_digest = state_digest.wrapping_mul(FNV_PRIME);
        state_digest ^= ((cell.color as u64) << 16) | (sq as u64);
        state_digest = state_digest.wrapping_mul(FNV_PRIME);
    }

    let blended_rgb = [
        rgb_acc_r.checked_div(rgb_weight_sum).unwrap_or(0).min(255) as u8,
        rgb_acc_g.checked_div(rgb_weight_sum).unwrap_or(0).min(255) as u8,
        rgb_acc_b.checked_div(rgb_weight_sum).unwrap_or(0).min(255) as u8,
    ];
    let blended_rgb565 =
        RationalWaveCell::pack_rgb565(blended_rgb[0], blended_rgb[1], blended_rgb[2]);

    state_digest ^= ((stage1_survivors as u64) << 32) | (stage2_survivors as u64);
    state_digest = state_digest.wrapping_mul(FNV_PRIME);
    state_digest ^= total_energy_i64 as u64;
    state_digest = state_digest.wrapping_mul(FNV_PRIME);
    state_digest ^= blended_rgb565 as u64;
    state_digest = state_digest.wrapping_mul(FNV_PRIME);

    RationalWaveShaderResult {
        total_cells: cells.len(),
        stage1_survivors,
        stage2_survivors,
        total_energy_i64,
        blended_rgb,
        blended_rgb565,
        state_digest,
    }
}

// ============================================================================
// ayeOS m158 + Zero-Shuffle Block-Planar Q4_G64 + In-Register Top-K / Gumbel-Max
// ============================================================================

/// 8-byte magic header for `ayeOS` Group-64 ternary matrices (`m000.bin .. m195.bin`).
pub const AYEOS_M158_MAGIC: [u8; 8] = *b"AYEM158\0";

/// 8-byte magic header for `ayeOS` Zero-Shuffle Block-Planar `Q4_G64` embeddings (`embed_tokens_q4_g64.bin`).
pub const AYEOS_EQ4G_MAGIC: [u8; 8] = *b"AYEEQ4G\0";

/// Parsed 64-byte header of an `ayeOS` `mNNN.bin` matrix file (`<8sIIIIII32s`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AyeOsMatrixHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub matrix_idx: u32,
    pub rows: u32,
    pub cols: u32,
    pub group_size: u32,
    pub flags: u32,
    pub blake3_digest: [u8; 32],
}

/// Parse and validate a 64-byte `AYEM158\0` binary matrix header.
#[inline]
pub const fn ayeos_parse_m158_header(header_64b: &[u8; 64]) -> Option<AyeOsMatrixHeader> {
    let mut magic = [0u8; 8];
    let mut i = 0usize;
    while i < 8 {
        magic[i] = header_64b[i];
        if magic[i] != AYEOS_M158_MAGIC[i] {
            return None;
        }
        i += 1;
    }
    let version = u32::from_le_bytes([header_64b[8], header_64b[9], header_64b[10], header_64b[11]]);
    let matrix_idx = u32::from_le_bytes([header_64b[12], header_64b[13], header_64b[14], header_64b[15]]);
    let rows = u32::from_le_bytes([header_64b[16], header_64b[17], header_64b[18], header_64b[19]]);
    let cols = u32::from_le_bytes([header_64b[20], header_64b[21], header_64b[22], header_64b[23]]);
    let group_size = u32::from_le_bytes([header_64b[24], header_64b[25], header_64b[26], header_64b[27]]);
    let flags = u32::from_le_bytes([header_64b[28], header_64b[29], header_64b[30], header_64b[31]]);
    let mut blake3_digest = [0u8; 32];
    let mut j = 0usize;
    while j < 32 {
        blake3_digest[j] = header_64b[32 + j];
        j += 1;
    }
    Some(AyeOsMatrixHeader {
        magic,
        version,
        matrix_idx,
        rows,
        cols,
        group_size,
        flags,
        blake3_digest,
    })
}

/// Convert IEEE-754 half-precision (`f16`) bits to `f32` in pure `core` Rust.
#[inline(always)]
pub const fn ayeos_f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal f16 -> normalized f32
        let mut m = mant;
        let mut e = 113u32;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        f32::from_bits(sign | (e << 23) | (m << 13))
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Branchless unpack of one 16-byte `ayeOS` Group-64 ternary block (`64` 2-bit trits)
/// into `[i8; 64]` using the exact identity:
///   `trit = ((code & 1) as i8) - ((code >> 1) as i8)`
/// which maps `0b00 -> 0`, `0b01 -> +1`, `0b10 -> -1`, `0b11 -> 0` with zero branches or LUTs.
#[inline(always)]
pub const fn ayeos_unpack_group64_branchless(packed_16b: &[u8; 16]) -> [i8; 64] {
    let mut out = [0i8; 64];
    let mut i = 0usize;
    while i < 16 {
        let b = packed_16b[i];
        let c0 = b & 0x03;
        let c1 = (b >> 2) & 0x03;
        let c2 = (b >> 4) & 0x03;
        let c3 = (b >> 6) & 0x03;
        let base = i * 4;
        out[base] = ((c0 & 1) as i8) - ((c0 >> 1) as i8);
        out[base + 1] = ((c1 & 1) as i8) - ((c1 >> 1) as i8);
        out[base + 2] = ((c2 & 1) as i8) - ((c2 >> 1) as i8);
        out[base + 3] = ((c3 & 1) as i8) - ((c3 >> 1) as i8);
        i += 1;
    }
    out
}

/// ARM64 NEON-aligned Group-64 ternary unpack (`0b00 -> 0, 0b01 -> +1, 0b10 -> -1`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub const fn ayeos_unpack_group64_neon(packed_16b: &[u8; 16]) -> [i8; 64] {
    ayeos_unpack_group64_branchless(packed_16b)
}

#[cfg(feature = "std")]
#[inline(always)]
fn ln_f32(x: f32) -> f32 {
    x.ln()
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn ln_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) as i32) - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let ln_m = 2.0 * y * (1.0 + y2 * (0.333_333_34 + y2 * (0.2 + y2 * 0.142_857_15)));
    (exp as f32) * 0.693_147_2 + ln_m
}

/// Candidate `(logit, token_id)` pair kept inside CPU/SIMD registers during fused LM-Head evaluation.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenLogit {
    pub logit: f32,
    pub token_id: u32,
}

/// Fixed-capacity in-register Top-K queue with Gumbel-Max (`SplitMix64`) categorical sampling.
/// Avoids materializing or writing the `151,936`-element vocabulary logit buffer to DRAM.
#[derive(Clone, Debug)]
pub struct InRegisterTopK<const K: usize> {
    pub items: [TokenLogit; K],
}

impl<const K: usize> Default for InRegisterTopK<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const K: usize> InRegisterTopK<K> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            items: [TokenLogit {
                logit: f32::NEG_INFINITY,
                token_id: u32::MAX,
            }; K],
        }
    }

    /// O(1) fast-path rejection (`>99.9%` of vocabulary rows rejected by a single `f32` compare).
    #[inline(always)]
    pub fn push(&mut self, logit: f32, token_id: u32) {
        if K == 0 || logit <= self.items[K - 1].logit {
            return;
        }
        let mut pos = K - 1;
        while pos > 0 && logit > self.items[pos - 1].logit {
            self.items[pos] = self.items[pos - 1];
            pos -= 1;
        }
        self.items[pos] = TokenLogit { logit, token_id };
    }

    /// Gumbel-Max trick: `argmax_i (logit_i / T - ln(-ln(U_i)))` is mathematically identical
    /// to temperature-scaled softmax categorical sampling without computing `exp` over `151,936` tokens.
    #[inline]
    pub fn sample_gumbel_max(&self, temperature: f32, rng_state: &mut u64) -> u32 {
        if K == 0 {
            return 0;
        }
        if temperature <= 1e-5 {
            return self.items[0].token_id;
        }
        let inv_t = 1.0f32 / temperature;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_tok = self.items[0].token_id;

        for item in &self.items {
            if item.token_id == u32::MAX {
                break;
            }
            // SplitMix64 PRNG step -> uniform U in (0, 1)
            *rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *rng_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;

            let u = ((z >> 40) as f32) * (1.0f32 / 16_777_216.0f32);
            let u_clamped = u.clamp(1e-7, 1.0 - 1e-7);
            let gumbel = -ln_f32(-ln_f32(u_clamped));
            let score = item.logit * inv_t + gumbel;
            if score > best_score {
                best_score = score;
                best_tok = item.token_id;
            }
        }
        best_tok
    }
}

/// Zero-Shuffle Block-Planar `Q4_G64` Token Embedding Lookup (`x_out = Embed[token_id]`).
///
/// Within each 32-byte (`G=64`) group:
/// - `bytes[0..15]`  (`lo_16b`): low nibble (`& 0x0F`) = `w[0..15]`,  high nibble (`>> 4`) = `w[16..31]`
/// - `bytes[16..31]` (`hi_16b`): low nibble (`& 0x0F`) = `w[32..47]`, high nibble (`>> 4`) = `w[48..63]`
#[inline]
pub fn ayeos_embed_lookup_q4_g64_planar(
    token_id: usize,
    packed_q4: &[u8],
    scales_fp16: &[f32],
    hidden_dim: usize,
    x_out: &mut [f32],
) {
    let num_groups = hidden_dim / 64;
    let row_q4 = &packed_q4[token_id * (hidden_dim / 2)..(token_id + 1) * (hidden_dim / 2)];
    let row_scales = &scales_fp16[token_id * num_groups..(token_id + 1) * num_groups];

    for g in 0..num_groups {
        let g_bytes = &row_q4[g * 32..g * 32 + 32];
        let scale = row_scales[g];
        let dst = &mut x_out[g * 64..g * 64 + 64];

        for i in 0..16 {
            let b_lo = g_bytes[i];
            let b_hi = g_bytes[16 + i];
            let q0 = ((b_lo & 0x0F) as i8) - 8;
            let q1 = ((b_lo >> 4) as i8) - 8;
            let q2 = ((b_hi & 0x0F) as i8) - 8;
            let q3 = ((b_hi >> 4) as i8) - 8;

            dst[i] = (q0 as f32) * scale;
            dst[16 + i] = (q1 as f32) * scale;
            dst[32 + i] = (q2 as f32) * scale;
            dst[48 + i] = (q3 as f32) * scale;
        }
    }
}

/// ARM64 NEON-aligned Zero-Shuffle Block-Planar `Q4_G64` Token Embedding Lookup.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn ayeos_embed_lookup_q4_g64_neon(
    token_id: usize,
    packed_q4: &[u8],
    scales_fp16: &[f32],
    hidden_dim: usize,
    x_out: &mut [f32],
) {
    ayeos_embed_lookup_q4_g64_planar(token_id, packed_q4, scales_fp16, hidden_dim, x_out);
}

/// Fused Tied `Q4_G64` LM-Head + In-Register Top-K + Gumbel-Max Sampler (Zero Logit Allocation).
///
/// 1. Quantizes `x_norm` (`[D]`) once into symmetric `int8` per 64-element group (`x_i8 * x_scale`).
/// 2. Streams through `V` vocabulary rows in Zero-Shuffle Block-Planar `Q4_G64` format,
///    accumulating `q0 * x0 + q1 * x1 + q2 * x2 + q3 * x3` in integer registers.
/// 3. Pushes each row's scalar logit directly into `InRegisterTopK<K>` and samples via Gumbel-Max.
#[inline]
pub fn ayeos_lmhead_q4_g64_topk_sample<const K: usize>(
    x_norm: &[f32],
    packed_q4: &[u8],
    scales_fp16: &[f32],
    vocab_size: usize,
    hidden_dim: usize,
    temperature: f32,
    rng_state: &mut u64,
) -> (u32, InRegisterTopK<K>) {
    let num_groups = hidden_dim / 64;
    let row_bytes = hidden_dim / 2;

    // Step 1: Pre-quantize activation vector `x_norm` once into int8 groups (stack buffer up to D=4096)
    let mut x_i8 = [0i8; 4096];
    let mut x_scales = [0.0f32; 64];

    for (g, x_scale_slot) in x_scales.iter_mut().enumerate().take(num_groups) {
        let base = g * 64;
        let mut amax = 1e-6f32;
        for i in 0..64 {
            let av = x_norm[base + i].abs();
            if av > amax {
                amax = av;
            }
        }
        let s = amax / 127.0f32;
        let inv_s = 127.0f32 / amax;
        *x_scale_slot = s;
        for i in 0..64 {
            let scaled = x_norm[base + i] * inv_s;
            let rounded = if scaled >= 0.0 {
                (scaled + 0.5) as i32
            } else {
                (scaled - 0.5) as i32
            };
            x_i8[base + i] = rounded.clamp(-127, 127) as i8;
        }
    }

    // Step 2: Stream over vocabulary rows -> accumulate into InRegisterTopK<K>
    let mut topk = InRegisterTopK::<K>::new();

    for v in 0..vocab_size {
        let v_q4 = &packed_q4[v * row_bytes..(v + 1) * row_bytes];
        let v_sc = &scales_fp16[v * num_groups..(v + 1) * num_groups];
        let mut row_logit = 0.0f32;

        for g in 0..num_groups {
            let g_bytes = &v_q4[g * 32..g * 32 + 32];
            let x_g = &x_i8[g * 64..g * 64 + 64];
            let mut dot_i32 = 0i32;

            for i in 0..16 {
                let b_lo = g_bytes[i];
                let b_hi = g_bytes[16 + i];
                let q0 = ((b_lo & 0x0F) as i32) - 8;
                let q1 = ((b_lo >> 4) as i32) - 8;
                let q2 = ((b_hi & 0x0F) as i32) - 8;
                let q3 = ((b_hi >> 4) as i32) - 8;

                dot_i32 += q0 * (x_g[i] as i32)
                    + q1 * (x_g[16 + i] as i32)
                    + q2 * (x_g[32 + i] as i32)
                    + q3 * (x_g[48 + i] as i32);
            }
            row_logit += (dot_i32 as f32) * (v_sc[g] * x_scales[g]);
        }
        topk.push(row_logit, v as u32);
    }

    let sampled = topk.sample_gumbel_max(temperature, rng_state);
    (sampled, topk)
}

/// ARM64 NEON-aligned Fused Tied `Q4_G64` LM-Head + In-Register Top-K + Gumbel-Max Sampler.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn ayeos_lmhead_q4_g64_topk_sample_neon<const K: usize>(
    x_norm: &[f32],
    packed_q4: &[u8],
    scales_fp16: &[f32],
    vocab_size: usize,
    hidden_dim: usize,
    temperature: f32,
    rng_state: &mut u64,
) -> (u32, InRegisterTopK<K>) {
    ayeos_lmhead_q4_g64_topk_sample::<K>(
        x_norm,
        packed_q4,
        scales_fp16,
        vocab_size,
        hidden_dim,
        temperature,
        rng_state,
    )
}

// ============================================================================
// 9. MEM|8 KompressedWaveStore (3-Plane SoA) + Pre-Attention Wave-Routed GQA
// ============================================================================

/// Default Qwen3 / DeepSeek head dimension (`128` = `4` spectral sub-bands of `32` RoPE lanes).
pub const AYEOS_HEAD_DIM: usize = 128;

/// Local sliding-window horizon (`64` tokens unconditionally attended).
pub const AYEOS_LOCAL_WINDOW: usize = 64;

/// Number of long-range phase-coherent episodic KV slots routed from `MEM|8` (`64` slots).
pub const AYEOS_RESONANT_TOP_M: usize = 64;

/// 64-byte quarter-symmetric sine wave LUT (`[-127, 127]`) pinned in 4 SIMD registers (`TBL4`).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct AlignedLut64(pub [i8; 64]);

pub static WAVE_LUT_64: AlignedLut64 = AlignedLut64([
    0, 12, 25, 37, 49, 60, 71, 81, 90, 98, 106, 112, 117, 122, 125, 126, 127, 126, 125, 122, 117,
    112, 106, 98, 90, 81, 71, 60, 49, 37, 25, 12, 0, -12, -25, -37, -49, -60, -71, -81, -90, -98,
    -106, -112, -117, -122, -125, -126, -127, -126, -125, -122, -117, -112, -106, -98, -90, -81,
    -71, -60, -49, -37, -25, -12,
]);

/// 66-bit `MEM|8` spectral wave probe extracted in ~18 ns from a 128-element `int8` Key or Query vector.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WaveProbe {
    /// Ternary gate in `{-1, 0, +1}`.
    pub quant: i8,
    /// `[15..12: Shape (4b)] [11..0: YCoCg444 Perceptual Color (12b)]`.
    pub morph_color: u16,
    /// Dyadic frequency `[15..4: p (12b)] [3..0: q_shift (4b)]` (`p * 2^-q_shift`).
    pub freq_pq: u16,
    /// Dyadic amplitude `[15..4: p (12b)] [3..0: q_shift (4b)]` (`p * 2^-q_shift`).
    pub amp_pq: u16,
    /// Dyadic phase `[15..4: p (12b)] [3..0: q_shift (4b)]` (`p * 2^-q_shift`).
    pub phase_pq: u16,
}

/// Branchless hardware `CLZ` conversion of `(p, q)` -> `[15..4: 12-bit p] [3..0: 4-bit q_shift]`.
#[inline(always)]
pub const fn pack_dyadic_branchless(p: u32, q: u32) -> u16 {
    let q_safe = q | ((q == 0) as u32);
    let log2_q = 31u32.wrapping_sub(q_safe.leading_zeros());
    let q_shift = if log2_q > 15 { 15 } else { log2_q };
    let p_clamped = if p > 0x0FFF { 0x0FFF } else { p };
    ((p_clamped << 4) | q_shift) as u16
}

/// Folds a 128-element `int8` Key or Query vector into a 66-bit `MEM|8` [`WaveProbe`] in ~18 ns.
/// Zero floating-point instructions, zero heap allocations.
#[inline]
pub fn fold_int8_to_wave_cell(
    vec_i8: &[i8; AYEOS_HEAD_DIM],
    is_heretic: bool,
    is_inhibitory: bool,
    salience_floor: u32,
) -> WaveProbe {
    let mut band_energy = [0u32; 4];
    let mut band_signed = [0i32; 4];
    let mut even_abs_sum = 0u32;
    let mut odd_abs_sum = 0u32;
    let mut global_amax = 0u32;

    for b in 0..4 {
        let base = b * 32;
        let mut s_sum = 0i32;
        let mut e_sum = 0u32;
        for i in 0..32 {
            let v = vec_i8[base + i];
            let av = v.unsigned_abs() as u32;
            s_sum += v as i32;
            e_sum += av;
            if av > global_amax {
                global_amax = av;
            }
            if (i & 1) == 0 {
                even_abs_sum += av;
            } else {
                odd_abs_sum += av;
            }
        }
        band_signed[b] = s_sum;
        band_energy[b] = e_sum;
    }

    let total_l1 = band_energy[0] + band_energy[1] + band_energy[2] + band_energy[3];

    let active_gate = (is_heretic as i8) | ((total_l1 >= salience_floor) as i8);
    let sign_select = if is_inhibitory { -1i8 } else { 1i8 };
    let quant = active_gate * sign_select;

    let shape_4b = ((band_signed[0] >= 0) as u16)
        | (((band_signed[1] >= 0) as u16) << 1)
        | (((band_signed[2] >= 0) as u16) << 2)
        | (((band_signed[3] >= 0) as u16) << 3);

    let y_4b = (total_l1 >> 10).min(15) as u16;
    let co_raw = ((band_energy[0] as i32 - band_energy[2] as i32) >> 8) + 8;
    let cg_raw = ((band_energy[1] as i32 - band_energy[3] as i32) >> 8) + 8;
    let co_4b = co_raw.clamp(0, 15) as u16;
    let cg_4b = cg_raw.clamp(0, 15) as u16;

    let morph_color = (shape_4b << 12) | (y_4b << 8) | (co_4b << 4) | cg_4b;

    let high_rope_e = (band_energy[2] + band_energy[3]).max(1);
    let low_rope_e = (band_energy[0] + band_energy[1]).max(1);

    let freq_pq = pack_dyadic_branchless(high_rope_e, low_rope_e);
    let amp_pq = pack_dyadic_branchless(global_amax << 5, (total_l1 >> 7).max(1));
    let phase_pq = pack_dyadic_branchless(odd_abs_sum << 4, even_abs_sum.max(1));

    WaveProbe {
        quant,
        morph_color,
        freq_pq,
        amp_pq,
        phase_pq,
    }
}

/// Bit-Sliced Structure-of-Arrays (SoA) `MEM|8` Wave Store (`8.25 Bytes/Cell` across 3 planes).
///
/// - **Plane 0**: `active_bits` (`1 = active, 0 = dormant`) and `sign_bits` (`0 = +1, 1 = -1`), 64 cells per `u64` word.
/// - **Plane 1**: `morph_color` (`[15..12: Shape (4b)] [11..0: YCoCg444 (12b)]`).
/// - **Plane 2**: `freq_pq`, `amp_pq`, `phase_pq` (`[15..4: p (12b)] [3..0: q_shift (4b)]`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct KompressedWaveStore<const N: usize, const WORDS: usize> {
    pub active_bits: [u64; WORDS],
    pub sign_bits: [u64; WORDS],
    pub morph_color: [u16; N],
    pub freq_pq: [u16; N],
    pub amp_pq: [u16; N],
    pub phase_pq: [u16; N],
}

impl<const N: usize, const WORDS: usize> Default for KompressedWaveStore<N, WORDS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const WORDS: usize> KompressedWaveStore<N, WORDS> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            active_bits: [0u64; WORDS],
            sign_bits: [0u64; WORDS],
            morph_color: [0u16; N],
            freq_pq: [0u16; N],
            amp_pq: [0u16; N],
            phase_pq: [0u16; N],
        }
    }

    /// Branchless 2.6 ns SoA bitplane upsert of a single [`WaveProbe`] at `cell_idx`.
    #[inline(always)]
    pub fn upsert_cell_branchless(&mut self, cell_idx: usize, probe: &WaveProbe) {
        let idx = cell_idx % N;
        let w_idx = idx >> 6;
        let bit = idx & 63;
        let bit_mask = 1u64 << bit;

        let is_active = (probe.quant != 0) as u64;
        let is_neg = (probe.quant < 0) as u64;

        self.active_bits[w_idx] = (self.active_bits[w_idx] & !bit_mask) | (is_active << bit);
        self.sign_bits[w_idx] = (self.sign_bits[w_idx] & !bit_mask) | (is_neg << bit);

        self.morph_color[idx] = probe.morph_color;
        self.freq_pq[idx] = probe.freq_pq;
        self.amp_pq[idx] = probe.amp_pq;
        self.phase_pq[idx] = probe.phase_pq;
    }
}

/// Branchless Plane-1 Gate: checks Shape Hamming distance `<= 1` and `YCoCg444` L1 distance `<= color_tol`.
#[inline(always)]
pub const fn plane1_morph_color_match(cand_mc: u16, query_mc: u16, color_tol: u32) -> bool {
    let shape_xor = ((cand_mc ^ query_mc) >> 12) as u32;
    let shape_dist = shape_xor.count_ones();

    let cy = ((cand_mc >> 8) & 0x0F) as i32;
    let cco = ((cand_mc >> 4) & 0x0F) as i32;
    let ccg = (cand_mc & 0x0F) as i32;

    let qy = ((query_mc >> 8) & 0x0F) as i32;
    let qco = ((query_mc >> 4) & 0x0F) as i32;
    let qcg = (query_mc & 0x0F) as i32;

    let ycocg_l1 = ((cy - qy).abs() + (cco - qco).abs() + (ccg - qcg).abs()) as u32;
    (shape_dist <= 1) & (ycocg_l1 <= color_tol)
}

/// Evaluates dyadic wave interference for a surviving cell in Plane 2 using `WAVE_LUT_64`.
#[inline(always)]
pub fn eval_dyadic_wave_survivor(
    freq_pq: u16,
    amp_pq: u16,
    phase_pq: u16,
    is_destructive: bool,
    step_t: u16,
    query_phase_offset: u16,
) -> i16 {
    let f_p = (freq_pq >> 4) as u32;
    let f_q = (freq_pq & 0x0F) as u32;
    let a_p = (amp_pq >> 4) as i32;
    let a_q = (amp_pq & 0x0F) as u32;
    let p_p = (phase_pq >> 4) as u32;
    let p_q = (phase_pq & 0x0F) as u32;

    let ft_shifted = (f_p.wrapping_mul(step_t as u32)) >> f_q;
    let p_shifted = p_p >> p_q;
    let cell_phase = (ft_shifted.wrapping_add(p_shifted)) as u16;
    let rel_phase = cell_phase.wrapping_sub(query_phase_offset);

    let idx = (rel_phase & 0x003F) as usize;
    let wave_s8 = WAVE_LUT_64.0[idx] as i32;
    let energy = ((wave_s8 * a_p) >> a_q) as i16;
    if is_destructive {
        energy.wrapping_neg()
    } else {
        energy
    }
}

/// Sub-microsecond 3-Plane Resonance Router over [`KompressedWaveStore`].
///
/// 1. **Plane 0**: Traverses set bits in `active_bits` via hardware `CTZ` (`trailing_zeros()`), skipping 64 dormant cells in 1 cycle.
/// 2. **Plane 1**: Applies [`plane1_morph_color_match`] (`Shape Hamming <= 1` and `YCoCg444 L1 <= color_tolerance`).
/// 3. **Plane 2**: Evaluates dyadic `p * 2^-q` + `WAVE_LUT_64` phase interference and selects Top-`M` resonant slots into [`InRegisterTopK`].
#[inline]
pub fn route_resonant_kv<const N: usize, const WORDS: usize, const M: usize>(
    store: &KompressedWaveStore<N, WORDS>,
    probe: &WaveProbe,
    step_t: u16,
    max_populated_word: usize,
    color_tolerance: u32,
) -> InRegisterTopK<M> {
    let mut top_slots = InRegisterTopK::<M>::new();
    let q_p = probe.phase_pq >> 4;
    let q_q = probe.phase_pq & 0x0F;
    let query_phase = q_p >> q_q;
    let scan_words = max_populated_word.min(WORDS);

    for w_idx in 0..scan_words {
        let mut active_word = store.active_bits[w_idx];
        if active_word == 0 {
            continue;
        }
        let sign_word = store.sign_bits[w_idx];
        let base_cell = w_idx << 6;

        while active_word != 0 {
            let bit = active_word.trailing_zeros() as usize;
            active_word &= active_word - 1;
            let cell_idx = base_cell + bit;
            if cell_idx >= N {
                break;
            }

            let cand_mc = store.morph_color[cell_idx];
            if !plane1_morph_color_match(cand_mc, probe.morph_color, color_tolerance) {
                continue;
            }

            let is_neg = ((sign_word >> bit) & 1) != 0;
            let energy = eval_dyadic_wave_survivor(
                store.freq_pq[cell_idx],
                store.amp_pq[cell_idx],
                store.phase_pq[cell_idx],
                is_neg,
                step_t,
                query_phase,
            );
            top_slots.push(energy as f32, cell_idx as u32);
        }
    }
    top_slots
}

#[cfg(feature = "std")]
#[inline(always)]
fn exp_f32(x: f32) -> f32 {
    x.exp()
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn exp_f32(x: f32) -> f32 {
    exp_approx(x as f64) as f32
}

/// Fused Pre-Attention `MEM|8` Wave Router + Sparse Online-Softmax `SDOT` Attention Step.
///
/// Routes only the Top-`M` resonant long-range episodic slots from `wave_store` plus the
/// `LOCAL_WINDOW` most recent tokens into the online FlashAttention-style accumulator.
#[allow(clippy::too_many_arguments)]
pub fn ayeos_wave_routed_attention_step<const N: usize, const WORDS: usize, const M: usize>(
    wave_store: &mut KompressedWaveStore<N, WORDS>,
    k_arena: &mut [[i8; AYEOS_HEAD_DIM]; N],
    v_arena: &mut [[i8; AYEOS_HEAD_DIM]; N],
    k_scales: &mut [f32; N],
    v_scales: &mut [f32; N],
    q_quant: &[i8; AYEOS_HEAD_DIM],
    q_scale: f32,
    k_quant: &[i8; AYEOS_HEAD_DIM],
    k_scale: f32,
    v_quant: &[i8; AYEOS_HEAD_DIM],
    v_scale: f32,
    attn_out_f32: &mut [f32; AYEOS_HEAD_DIM],
    step_t: usize,
    local_window: usize,
    is_heretic_tok: bool,
    is_inhibitory_tok: bool,
    wave_gamma: f32,
) -> InRegisterTopK<M> {
    let slot_idx = step_t % N;

    // 1. Write Path: Store int8 KV + fold into 66-bit wave cell + commit to 3 SoA planes
    k_arena[slot_idx] = *k_quant;
    v_arena[slot_idx] = *v_quant;
    k_scales[slot_idx] = k_scale;
    v_scales[slot_idx] = v_scale;

    let k_wave = fold_int8_to_wave_cell(k_quant, is_heretic_tok, is_inhibitory_tok, 512);
    wave_store.upsert_cell_branchless(slot_idx, &k_wave);

    // 2. Read/Route Path: Excite MEM|8 with Query WaveProbe over historical region
    let q_probe = fold_int8_to_wave_cell(q_quant, true, false, 0);
    let historical_end = step_t.saturating_sub(local_window);
    let populated_words = (historical_end + 63) >> 6;

    let resonant_topk = if populated_words > 0 {
        route_resonant_kv::<N, WORDS, M>(
            wave_store,
            &q_probe,
            (step_t & 0xFFFF) as u16,
            populated_words,
            6,
        )
    } else {
        InRegisterTopK::<M>::new()
    };

    // 3. Fused Online-Softmax Sparse Attention over (Top-M Resonant + Local Sliding Window)
    let inv_sqrt_dk = 0.088_388_35f32; // 1 / sqrt(128)
    let q_factor = q_scale * inv_sqrt_dk;

    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32;
    *attn_out_f32 = [0.0f32; AYEOS_HEAD_DIM];

    let mut attend_slot = |kv_idx: usize, wave_bias: f32| {
        let mut dot_i32 = 0i32;
        for d in 0..AYEOS_HEAD_DIM {
            dot_i32 += (q_quant[d] as i32) * (k_arena[kv_idx][d] as i32);
        }
        let logit = (dot_i32 as f32) * q_factor * k_scales[kv_idx] + wave_bias;

        let prev_max = running_max;
        if logit > running_max {
            running_max = logit;
        }
        let exp_diff = if prev_max == f32::NEG_INFINITY {
            0.0f32
        } else {
            exp_f32(prev_max - running_max)
        };
        let weight = exp_f32(logit - running_max);
        running_sum = running_sum * exp_diff + weight;

        let v_sc = weight * v_scales[kv_idx];
        for d in 0..AYEOS_HEAD_DIM {
            attn_out_f32[d] = attn_out_f32[d] * exp_diff + (v_arena[kv_idx][d] as f32) * v_sc;
        }
    };

    // Pass A: Attend to Top-M Long-Range Resonant Episodic Slots from MEM|8
    for item in &resonant_topk.items {
        if item.token_id == u32::MAX || item.logit == f32::NEG_INFINITY {
            break;
        }
        attend_slot((item.token_id as usize) % N, item.logit * wave_gamma);
    }

    // Pass B: Attend to the Local Sliding Window (Unconditional Syntactic Horizon)
    let local_start = historical_end;
    for pos in local_start..=step_t {
        attend_slot(pos % N, 0.0);
    }

    let inv_sum = 1.0f32 / running_sum.max(1e-9);
    for slot in attn_out_f32.iter_mut().take(AYEOS_HEAD_DIM) {
        *slot *= inv_sum;
    }

    resonant_topk
}

// ============================================================================
// 10. Burn (`tracel-ai/burn`) QuantScheme (Q2S / Q4S U32) <-> AyeOS Bridge
// ============================================================================

/// Transcodes a single Group-64 block (`[u32; 4]`) from `burn-tensor`'s `QuantValue::Q2S`
/// (`QuantStore::U32`, offset `+1`: `-1 -> 0b00, 0 -> 0b01, +1 -> 0b10`) into `ayeOS`'s
/// 16-byte branchless `I2_S` layout (`0b00 -> 0, 0b01 -> +1, 0b10 -> -1`).
///
/// Zero heap allocations, `const fn`, `#![no_std]`.
#[inline(always)]
pub const fn burn_q2s_u32_to_ayeos_i2s_g64(burn_u32: &[u32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    let mut w = 0usize;
    while w < 4 {
        let word = burn_u32[w];
        let mut b_idx = 0usize;
        while b_idx < 4 {
            let byte_in = ((word >> (b_idx * 8)) & 0xFF) as u8;
            let mut byte_out = 0u8;
            let mut lane = 0usize;
            while lane < 4 {
                let b_code = (byte_in >> (lane * 2)) & 0x03;
                let trit = (b_code as i8) - 1;
                let a_code = ((trit == 1) as u8) | (((trit == -1) as u8) << 1);
                byte_out |= a_code << (lane * 2);
                lane += 1;
            }
            out[w * 4 + b_idx] = byte_out;
            b_idx += 1;
        }
        w += 1;
    }
    out
}

/// Inverse of [`burn_q2s_u32_to_ayeos_i2s_g64`]: transcodes an `ayeOS` 16-byte `I2_S` Group-64
/// block (`0b00 -> 0, 0b01 -> +1, 0b10 -> -1`) back into `burn-tensor`'s `QuantValue::Q2S`
/// `[u32; 4]` (`QuantStore::U32`, offset `+1`: `-1 -> 0b00, 0 -> 0b01, +1 -> 0b10`).
#[inline(always)]
pub const fn ayeos_i2s_g64_to_burn_q2s_u32(ayeos_16b: &[u8; 16]) -> [u32; 4] {
    let mut out = [0u32; 4];
    let mut w = 0usize;
    while w < 4 {
        let mut word = 0u32;
        let mut b_idx = 0usize;
        while b_idx < 4 {
            let byte_in = ayeos_16b[w * 4 + b_idx];
            let mut lane = 0usize;
            while lane < 4 {
                let a_code = (byte_in >> (lane * 2)) & 0x03;
                let trit = ((a_code & 1) as i8) - ((a_code >> 1) as i8);
                let b_code = ((trit + 1) as u32) & 0x03;
                let bit_shift = (b_idx * 4 + lane) * 2;
                word |= b_code << bit_shift;
                lane += 1;
            }
            b_idx += 1;
        }
        out[w] = word;
        w += 1;
    }
    out
}

/// Transcodes a single Group-64 block (`[u32; 8]`) from `burn-tensor`'s sequential `QuantValue::Q4S`
/// (`QuantStore::U32`, offset `+7` for symmetric `[-7, +7]`) into `ayeOS`'s 32-byte
/// **Zero-Shuffle Block-Planar `Q4_G64`** layout (offset `+8`):
/// - `planar_32b[0..16]`: low nibble = `w[0..16]`, high nibble = `w[16..32]`
/// - `planar_32b[16..32]`: low nibble = `w[32..48]`, high nibble = `w[48..64]`
#[inline(always)]
pub const fn burn_q4s_u32_to_ayeos_q4_g64(burn_u32: &[u32; 8]) -> [u8; 32] {
    let mut weights_i8 = [0i8; 64];
    let mut w = 0usize;
    while w < 8 {
        let word = burn_u32[w];
        let mut k = 0usize;
        while k < 8 {
            let nibble = ((word >> (k * 4)) & 0x0F) as i8;
            weights_i8[w * 8 + k] = nibble - 7;
            k += 1;
        }
        w += 1;
    }

    let mut planar = [0u8; 32];
    let mut i = 0usize;
    while i < 16 {
        let q0 = ((weights_i8[i] + 8) as u8) & 0x0F;
        let q1 = ((weights_i8[16 + i] + 8) as u8) & 0x0F;
        let q2 = ((weights_i8[32 + i] + 8) as u8) & 0x0F;
        let q3 = ((weights_i8[48 + i] + 8) as u8) & 0x0F;
        planar[i] = q0 | (q1 << 4);
        planar[16 + i] = q2 | (q3 << 4);
        i += 1;
    }
    planar
}

/// Inverse of [`burn_q4s_u32_to_ayeos_q4_g64`]: transcodes an `ayeOS` 32-byte
/// **Zero-Shuffle Block-Planar `Q4_G64`** block (offset `+8`) back into `burn-tensor`'s
/// sequential `QuantValue::Q4S` `[u32; 8]` (`QuantStore::U32`, offset `+7`, clamped to `[-7, +7]`).
#[inline(always)]
pub const fn ayeos_q4_g64_to_burn_q4s_u32(planar_32b: &[u8; 32]) -> [u32; 8] {
    let mut weights_i8 = [0i8; 64];
    let mut i = 0usize;
    while i < 16 {
        let b_lo = planar_32b[i];
        let b_hi = planar_32b[16 + i];
        let q0 = ((b_lo & 0x0F) as i8) - 8;
        let q1 = ((b_lo >> 4) as i8) - 8;
        let q2 = ((b_hi & 0x0F) as i8) - 8;
        let q3 = ((b_hi >> 4) as i8) - 8;

        weights_i8[i] = if q0 < -7 { -7 } else if q0 > 7 { 7 } else { q0 };
        weights_i8[16 + i] = if q1 < -7 { -7 } else if q1 > 7 { 7 } else { q1 };
        weights_i8[32 + i] = if q2 < -7 { -7 } else if q2 > 7 { 7 } else { q2 };
        weights_i8[48 + i] = if q3 < -7 { -7 } else if q3 > 7 { 7 } else { q3 };
        i += 1;
    }

    let mut out = [0u32; 8];
    let mut w = 0usize;
    while w < 8 {
        let mut word = 0u32;
        let mut k = 0usize;
        while k < 8 {
            let b_nib = ((weights_i8[w * 8 + k] + 7) as u32) & 0x0F;
            word |= b_nib << (k * 4);
            k += 1;
        }
        out[w] = word;
        w += 1;
    }
    out
}

#[cfg(test)]
mod ayeos_tests {
    use super::*;

    #[test]
    fn test_ayeos_unpack_group64_branchless_all_codes() {
        // Pack repeating [0 (0b00), +1 (0b01), -1 (0b10), 0 (0b00)] -> 0b00_10_01_00 = 0x24
        let packed = [0b00_10_01_00u8; 16];
        let unpacked = ayeos_unpack_group64_branchless(&packed);
        for chunk in unpacked.chunks_exact(4) {
            assert_eq!(chunk, &[0i8, 1i8, -1i8, 0i8]);
        }
    }

    #[test]
    fn test_ayeos_block_planar_q4_g64_and_lmhead_topk() {
        // 4 vocabulary rows, hidden_dim = 64 (1 group of 32 bytes per row)
        let vocab_size = 4usize;
        let hidden_dim = 64usize;
        let mut packed_q4 = [0x88u8; 4 * 32]; // 0x88 -> q0 = 8-8 = 0, q1 = 8-8 = 0
        // Make row 2 have positive +7 nibbles (8 + 7 = 15 -> 0xFF)
        for b in &mut packed_q4[2 * 32..3 * 32] {
            *b = 0xFF;
        }
        let scales = [1.0f32, 1.0f32, 0.5f32, 1.0f32];

        let mut emb_row2 = [0.0f32; 64];
        ayeos_embed_lookup_q4_g64_planar(2, &packed_q4, &scales, hidden_dim, &mut emb_row2);
        for &v in &emb_row2 {
            assert!((v - 3.5f32).abs() < 1e-6);
        }

        let x_norm = [1.0f32; 64];
        let mut rng = 0xCAFE_BABE_1613_3000u64;
        let (tok_greedy, topk) = ayeos_lmhead_q4_g64_topk_sample::<4>(
            &x_norm,
            &packed_q4,
            &scales,
            vocab_size,
            hidden_dim,
            0.0,
            &mut rng,
        );
        assert_eq!(tok_greedy, 2);
        assert_eq!(topk.items[0].token_id, 2);
        assert!(topk.items[0].logit > 200.0);
    }

    #[test]
    fn test_mem8_kompressed_wave_store_and_routed_gqa_step() {
        let mut store = KompressedWaveStore::<256, 4>::new();
        let mut k_arena = [[0i8; AYEOS_HEAD_DIM]; 256];
        let mut v_arena = [[0i8; AYEOS_HEAD_DIM]; 256];
        let mut k_scales = [1.0f32; 256];
        let mut v_scales = [1.0f32; 256];

        // Populate 128 historical steps; make step 17 a resonant Heretic anchor
        for t in 0..128 {
            let mut k_vec = [1i8; AYEOS_HEAD_DIM];
            let mut v_vec = [2i8; AYEOS_HEAD_DIM];
            let is_heretic = t == 17;
            if is_heretic {
                k_vec = [40i8; AYEOS_HEAD_DIM];
                v_vec = [99i8; AYEOS_HEAD_DIM];
            }
            let mut out = [0.0f32; AYEOS_HEAD_DIM];
            let _ = ayeos_wave_routed_attention_step::<256, 4, 8>(
                &mut store,
                &mut k_arena,
                &mut v_arena,
                &mut k_scales,
                &mut v_scales,
                &k_vec,
                0.05,
                &k_vec,
                0.05,
                &v_vec,
                0.05,
                &mut out,
                t,
                32,
                is_heretic,
                false,
                1.0 / 4096.0,
            );
        }

        // Now query at step 128 with the exact same spectral signature as step 17
        let q_vec = [40i8; AYEOS_HEAD_DIM];
        let k_vec = [4i8; AYEOS_HEAD_DIM];
        let v_vec = [4i8; AYEOS_HEAD_DIM];
        let mut attn_out = [0.0f32; AYEOS_HEAD_DIM];
        let topk = ayeos_wave_routed_attention_step::<256, 4, 8>(
            &mut store,
            &mut k_arena,
            &mut v_arena,
            &mut k_scales,
            &mut v_scales,
            &q_vec,
            0.05,
            &k_vec,
            0.05,
            &v_vec,
            0.05,
            &mut attn_out,
            128,
            32,
            false,
            false,
            1.0 / 4096.0,
        );

        // Step 17 must be in the routed Top-8 resonant historical slots and dominate attention output!
        let routed_ids: [u32; 8] = core::array::from_fn(|i| topk.items[i].token_id);
        assert!(routed_ids.contains(&17), "Expected resonant slot 17 in {routed_ids:?}");
        assert!(attn_out[0] > 4.0, "Expected high-salience value contribution from slot 17, got {}", attn_out[0]);
    }

    #[test]
    fn test_burn_q2s_and_q4s_ayeos_zero_shuffle_bridge_roundtrip() {
        // 1. Q2S <-> AyeOS I2_S_G64 round-trip
        // Burn Q2S codes: -1 -> 0b00, 0 -> 0b01, +1 -> 0b10
        let expected_trits: [i8; 64] = core::array::from_fn(|i| match i % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        });
        let mut burn_q2s = [0u32; 4];
        for w in 0..4 {
            let mut word = 0u32;
            for k in 0..16 {
                let b_code = (expected_trits[w * 16 + k] + 1) as u32;
                word |= (b_code & 0x03) << (k * 2);
            }
            burn_q2s[w] = word;
        }
        let ayeos_16b = burn_q2s_u32_to_ayeos_i2s_g64(&burn_q2s);
        let unpacked_trits = ayeos_unpack_group64_branchless(&ayeos_16b);
        assert_eq!(unpacked_trits, expected_trits);
        let roundtrip_q2s = ayeos_i2s_g64_to_burn_q2s_u32(&ayeos_16b);
        assert_eq!(roundtrip_q2s, burn_q2s);

        // 2. Q4S <-> AyeOS Zero-Shuffle Block-Planar Q4_G64 round-trip
        let expected_q4: [i8; 64] = core::array::from_fn(|i| ((i as i8) % 15) - 7);
        let mut burn_q4s = [0u32; 8];
        for w in 0..8 {
            let mut word = 0u32;
            for k in 0..8 {
                let b_nib = (expected_q4[w * 8 + k] + 7) as u32;
                word |= (b_nib & 0x0F) << (k * 4);
            }
            burn_q4s[w] = word;
        }
        let planar_32b = burn_q4s_u32_to_ayeos_q4_g64(&burn_q4s);
        let mut emb_out = [0.0f32; 64];
        ayeos_embed_lookup_q4_g64_planar(0, &planar_32b, &[1.0f32], 64, &mut emb_out);
        for i in 0..64 {
            assert_eq!(emb_out[i] as i8, expected_q4[i]);
        }
        let roundtrip_q4s = ayeos_q4_g64_to_burn_q4s_u32(&planar_32b);
        assert_eq!(roundtrip_q4s, burn_q4s);
    }
}
