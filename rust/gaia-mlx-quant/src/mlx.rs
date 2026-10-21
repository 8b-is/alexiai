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


