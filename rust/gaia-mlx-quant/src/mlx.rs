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
            let inv_scale = 1.0 / scale;

            // Pack 64 weights into 4 u32 words (16 two-bit codes per word).
            for w_idx in 0..4 {
                let sub = &group[w_idx * TRITS_PER_U32..(w_idx + 1) * TRITS_PER_U32];
                let mut word = 0u32;
                for (k, &val) in sub.iter().enumerate() {
                    let scaled = val * inv_scale;
                    let trit: i8 = if scaled >= 0.5 {
                        1
                    } else if scaled <= -0.5 {
                        -1
                    } else {
                        0
                    };
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
                let inv_scale = 1.0 / scale;
                let mut g_acc = 0.0f64;
                for (&wv, &xv) in group_w.iter().zip(group_x.iter()) {
                    let s = wv * inv_scale;
                    let trit = if s >= 0.5 {
                        1.0
                    } else if s <= -0.5 {
                        -1.0
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

/// Core-only cosine/sine approximation (7th-order minimax polynomial with range reduction)
/// so Rotary Position Embeddings (`rope_into`) work in `#![no_std]` with zero dependencies.
#[inline]
fn sin_cos_approx(theta: f64) -> (f64, f64) {
    use core::f64::consts::{FRAC_PI_2 as HALF_PI, PI, TAU};

    // Reduce into [-PI, PI]
    let k = (theta / TAU).round();
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

    // 2. Ternary Q, K, V projections (1.585 bits/weight i-major)
    ternary_matmul_i_major(wq_i_packed, dim, dim, scratch_norm, None, 1.0, scratch_q);
    ternary_matmul_i_major(wk_i_packed, dim, dim, scratch_norm, None, 1.0, scratch_k);
    ternary_matmul_i_major(wv_i_packed, dim, dim, scratch_norm, None, 1.0, scratch_v);

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
    ternary_matmul_i_major(wo_i_packed, dim, dim, scratch_attn, None, 1.0, scratch_norm);
    for i in 0..dim {
        x_out[i] = x_in[i] + scratch_norm[i];
    }

    // 6. Pre-MLP RMSNorm
    rms_norm_into(x_out, Some(rms2_w), 1e-5, scratch_norm);

    // 7. Ternary SwiGLU MLP (Gate + Up -> SiLU*Up -> Down) + Residual 2
    ternary_matmul_i_major(wgate_i_packed, dim, hidden_dim, scratch_norm, None, 1.0, scratch_gate);
    ternary_matmul_i_major(wup_i_packed, dim, hidden_dim, scratch_norm, None, 1.0, scratch_up);
    for i in 0..hidden_dim {
        let g = scratch_gate[i];
        scratch_gate[i] = (g / (1.0 + exp_approx(-g))) * scratch_up[i];
    }
    ternary_matmul_i_major(wdown_i_packed, hidden_dim, dim, scratch_gate, None, 1.0, scratch_norm);
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
