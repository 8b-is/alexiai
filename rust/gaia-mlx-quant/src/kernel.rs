//! The kernel — masked ternary accumulation, zero allocations, zero `unsafe`.
//!
//! $$Y_i = \gamma \cdot \sum_j \left( \mathbf{W}^{+}_{j,i} - \mathbf{W}^{-}_{j,i} \right) \cdot \text{scale}_j \cdot x_j$$
//!
//! Row-major packed weights of shape `[rows, cols]` (trits, 5 per byte),
//! input `x` of length `rows`, optional per-row `scales`. The result is
//! written into a caller-owned `out` buffer — the hot path performs no heap
//! operations at all.
//!
//! ## the tricks, in order of appearance
//!
//! 1. **Aligned fast path** — when `cols % 5 == 0`, every byte belongs to
//!    exactly one row, so the per-slot bounds guards disappear entirely and
//!    the inner loop is pure unrolled LUT+FMA.
//! 2. **Const LUTs** — each byte's five trits come from the compile-time
//!    tables in [`crate::pack`]; no division, no modulo.
//! 3. **Per-row scalar hoist** — `xj = x[j] * scale_j * γ` is computed once
//!    per row, then applied to every slot.
//! 4. **Skip-zero-byte fast path** — a byte of all-zero trits contributes
//!    nothing to five accumulators, so it is skipped before any work.
//! 5. **FMA accumulation** — `fma(a, b, c)` compiles to a fused
//!    multiply-add on aarch64 and x86_64 std builds.
//! 6. **Unrolled slots** — the five slots are unrolled so each LUT load is a
//!    static direct access; there is no inner loop in the middle of the hot
//!    path.
//!
//! Accuracy note: accumulation is `f64`, so long columns never drift.

use crate::pack::{POW3, T0, T1, T2, T3, T4, TRITS_PER_BYTE, digit_of};

#[cfg(feature = "std")]
use alloc::vec;
#[cfg(feature = "std")]
use alloc::vec::Vec;

/// Fused multiply-add when std provides it; the portable fallback otherwise.
/// On aarch64 and x86_64 std builds, `a.mul_add(b, c)` compiles to a single
/// fused instruction; the `core`-only build keeps the portable two-op form.
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

/// `core`-only square root: a bit-level exponent-halving estimate, then
/// Newton refinement to a fixpoint. `f64::sqrt` lives in std; this is the
/// no-dependency stand-in. `0x1FF7_8000_0000_0000` is the classic f64 sqrt
/// magic constant (the exponent offset plus the mantissa tweak); the guess
/// lands within a few percent and Newton squares the error every step.
#[inline]
fn sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NAN };
    }
    let mut y = f64::from_bits((x.to_bits() >> 1) + 0x1FF7_8000_0000_0000u64);
    for _ in 0..64 {
        let next = 0.5 * (y + x / y);
        if next == y {
            break;
        }
        y = next;
    }
    y
}

/// The masked kernel.
///
/// # Panics
/// Only under `debug_assert!`s for shape violations; the kernel itself cannot
/// panic on valid shapes.
#[inline]
pub fn ternary_matmul(
    packed: &[u8],
    rows: usize,
    cols: usize,
    x: &[f64],
    scales: Option<&[f64]>,
    gamma: f64,
    out: &mut [f64],
) {
    debug_assert!(x.len() >= rows, "x shorter than rows");
    debug_assert!(out.len() >= cols, "out shorter than cols");
    debug_assert!(
        packed.len() * TRITS_PER_BYTE >= rows * cols,
        "packed buffer too small"
    );
    debug_assert!(scales.map(|s| s.len() >= rows).unwrap_or(true));

    for slot in out[..cols].iter_mut() {
        *slot = 0.0;
    }

    if cols.is_multiple_of(TRITS_PER_BYTE) {
        // ── aligned fast path: no guards, every byte fully inside its row ──
        let bytes_per_row = cols / TRITS_PER_BYTE;
        for (j, &xv) in x.iter().take(rows).enumerate() {
            if xv == 0.0 {
                continue;
            }
            let xj = xv * scales.map(|s| s[j]).unwrap_or(1.0) * gamma;
            let row = &packed[j * bytes_per_row..(j + 1) * bytes_per_row];
            let mut i = 0usize;
            for &byte in row {
                if byte == 0 {
                    i += TRITS_PER_BYTE;
                    continue;
                }
                out[i] = fma(T0[byte as usize] as f64, xj, out[i]);
                out[i + 1] = fma(T1[byte as usize] as f64, xj, out[i + 1]);
                out[i + 2] = fma(T2[byte as usize] as f64, xj, out[i + 2]);
                out[i + 3] = fma(T3[byte as usize] as f64, xj, out[i + 3]);
                out[i + 4] = fma(T4[byte as usize] as f64, xj, out[i + 4]);
                i += TRITS_PER_BYTE;
            }
        }
        return;
    }

    // ── general path: rows may cross byte boundaries ──
    for (j, &xv) in x.iter().take(rows).enumerate() {
        if xv == 0.0 {
            continue;
        }
        // Hoist everything that depends only on the row.
        let xj = xv * scales.map(|s| s[j]).unwrap_or(1.0) * gamma;

        // This row's trit range in the flat stream.
        let start_t = j * cols;
        let end_t = start_t + cols;
        let start_b = start_t / TRITS_PER_BYTE;
        let end_b = end_t.div_ceil(TRITS_PER_BYTE);

        for (b, &byte) in packed[start_b..end_b].iter().enumerate() {
            if byte == 0 {
                continue; // five zeros, five skipped accumulators
            }
            let base = (start_b + b) * TRITS_PER_BYTE;

            let t0 = base;
            if t0 >= start_t && t0 < end_t {
                let i0 = t0 - start_t;
                out[i0] = fma(T0[byte as usize] as f64, xj, out[i0]);
            }
            let t1 = base + 1;
            if t1 >= start_t && t1 < end_t {
                let i1 = t1 - start_t;
                out[i1] = fma(T1[byte as usize] as f64, xj, out[i1]);
            }
            let t2 = base + 2;
            if t2 >= start_t && t2 < end_t {
                let i2 = t2 - start_t;
                out[i2] = fma(T2[byte as usize] as f64, xj, out[i2]);
            }
            let t3 = base + 3;
            if t3 >= start_t && t3 < end_t {
                let i3 = t3 - start_t;
                out[i3] = fma(T3[byte as usize] as f64, xj, out[i3]);
            }
            let t4 = base + 4;
            if t4 >= start_t && t4 < end_t {
                let i4 = t4 - start_t;
                out[i4] = fma(T4[byte as usize] as f64, xj, out[i4]);
            }
        }
    }
}

/// Pack trits **i-major** (transposed): trit stream index `i * rows + j`.
///
/// The i-major layout is the layout the kernel wants: for a fixed output
/// index `i`, its `rows` trits are contiguous in the packed buffer, so the
/// accumulator stays in a register for the whole column and `out` is written
/// exactly once. Packing is offline (one-time per weight matrix), so the
/// per-element div/mod here is free in the steady state.
#[inline]
pub fn pack_ternary_i_major_into(
    trits_j_major: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [u8],
) {
    debug_assert!(trits_j_major.len() >= rows * cols);
    debug_assert!(out.len() * TRITS_PER_BYTE >= rows * cols);
    for j in 0..rows {
        for i in 0..cols {
            let idx = i * rows + j;
            let t = trits_j_major[j * cols + i];
            out[idx / TRITS_PER_BYTE] += POW3[idx % TRITS_PER_BYTE] * digit_of(t);
        }
    }
}

/// The i-major kernel — the layout trick that makes the ternary path
/// competitive with vectorized dense.
///
/// For every output column the accumulator lives in a register, the packed
/// trits stream once, `x` and `scales` stay L1-hot, and `out` is written
/// exactly once. On a 256² matrix this turns ~500KB of store traffic into
/// 2KB.
#[inline]
pub fn ternary_matmul_i_major(
    packed: &[u8],
    rows: usize,
    cols: usize,
    x: &[f64],
    scales: Option<&[f64]>,
    gamma: f64,
    out: &mut [f64],
) {
    debug_assert!(x.len() >= rows);
    debug_assert!(out.len() >= cols);
    debug_assert!(packed.len() * TRITS_PER_BYTE >= rows * cols);
    debug_assert!(scales.map(|s| s.len() >= rows).unwrap_or(true));

    let scale_at = |j: usize| scales.map(|s| s[j]).unwrap_or(1.0);

    for (i, slot) in out[..cols].iter_mut().enumerate() {
        let start = i * rows;
        let end = start + rows;
        let start_b = start / TRITS_PER_BYTE;
        let end_b = end.div_ceil(TRITS_PER_BYTE);

        let mut acc = 0.0f64;
        for (b, &byte) in packed[start_b..end_b].iter().enumerate() {
            if byte == 0 {
                continue;
            }
            let base = (start_b + b) * TRITS_PER_BYTE;

            if base < end && base >= start {
                acc = fma(T0[byte as usize] as f64, x[base - start] * scale_at(base - start), acc);
            }
            let t1 = base + 1;
            if t1 < end && t1 >= start {
                acc = fma(T1[byte as usize] as f64, x[t1 - start] * scale_at(t1 - start), acc);
            }
            let t2 = base + 2;
            if t2 < end && t2 >= start {
                acc = fma(T2[byte as usize] as f64, x[t2 - start] * scale_at(t2 - start), acc);
            }
            let t3 = base + 3;
            if t3 < end && t3 >= start {
                acc = fma(T3[byte as usize] as f64, x[t3 - start] * scale_at(t3 - start), acc);
            }
            let t4 = base + 4;
            if t4 < end && t4 >= start {
                acc = fma(T4[byte as usize] as f64, x[t4 - start] * scale_at(t4 - start), acc);
            }
        }
        *slot = acc * gamma;
    }
}

/// Unpack packed trits into **i-major sign planes**: `pos[k] = 1` where the
/// trit is `+1`, `neg[k] = 1` where it is `-1`, zero elsewhere, `k = i*rows+j`.
///
/// Storage stays packed at 1.585 bits/weight; the planes are the *runtime*
/// form (2 bytes/trit, still 4× smaller than f64) whose contiguous i8 loads
/// let the compiler vectorize the kernel with widen-and-FMA chains.
#[inline]
pub fn pack_ternary_planes_i_into(
    trits_j_major: &[i8],
    rows: usize,
    cols: usize,
    pos: &mut [i8],
    neg: &mut [i8],
) {
    debug_assert!(trits_j_major.len() >= rows * cols);
    debug_assert!(pos.len() >= rows * cols && neg.len() >= rows * cols);
    for j in 0..rows {
        for i in 0..cols {
            let k = i * rows + j;
            let t = trits_j_major[j * cols + i];
            pos[k] = if t == 1 { 1 } else { 0 };
            neg[k] = if t == -1 { 1 } else { 0 };
        }
    }
}

/// The plane kernel — branchless, register accumulators, 4-way ILP.
///
/// For each output column, **four independent accumulators** break the
/// serial f64 FMA dependency chain (the classic ILP trick: one chain is
/// latency-bound at ~4 cycles/fma; four chains approach the issue rate).
/// The inner loop walks the contiguous i8 planes; `(pos - neg) ∈ {-1, 0, 1}`
/// is widened to f64 and fused. The widen chain is scalar — the
/// auto-vectorizer is welcome but not assumed, and the bench reports
/// whatever it actually gets.
#[inline]
#[allow(clippy::too_many_arguments)] // a kernel takes its full context flat; a struct would hide it
pub fn ternary_matmul_planes_i(
    pos: &[i8],
    neg: &[i8],
    rows: usize,
    cols: usize,
    x: &[f64],
    scales: Option<&[f64]>,
    gamma: f64,
    out: &mut [f64],
) {
    debug_assert!(pos.len() >= rows * cols && neg.len() >= rows * cols);
    debug_assert!(x.len() >= rows);
    debug_assert!(out.len() >= cols);
    debug_assert!(scales.map(|s| s.len() >= rows).unwrap_or(true));

    let scale_at = |j: usize| scales.map(|s| s[j]).unwrap_or(1.0);

    for (i, slot) in out[..cols].iter_mut().enumerate() {
        let base = i * rows;

        // Four independent accumulator chains for ILP.
        let mut acc0 = 0.0f64;
        let mut acc1 = 0.0f64;
        let mut acc2 = 0.0f64;
        let mut acc3 = 0.0f64;

        let mut j = 0usize;
        while j + 4 <= rows {
            acc0 = fma((pos[base + j] - neg[base + j]) as f64, x[j] * scale_at(j), acc0);
            acc1 = fma(
                (pos[base + j + 1] - neg[base + j + 1]) as f64,
                x[j + 1] * scale_at(j + 1),
                acc1,
            );
            acc2 = fma(
                (pos[base + j + 2] - neg[base + j + 2]) as f64,
                x[j + 2] * scale_at(j + 2),
                acc2,
            );
            acc3 = fma(
                (pos[base + j + 3] - neg[base + j + 3]) as f64,
                x[j + 3] * scale_at(j + 3),
                acc3,
            );
            j += 4;
        }
        // Tail.
        while j < rows {
            acc0 = fma((pos[base + j] - neg[base + j]) as f64, x[j] * scale_at(j), acc0);
            j += 1;
        }
        *slot = (acc0 + acc1 + acc2 + acc3) * gamma;
    }
}

/// Absmean ternary quantization result.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct Quantized {
    pub trits: Vec<i8>,
    pub scales: Option<Vec<f64>>,
    pub gamma: f64,
}

/// Quantize a row-major weight matrix to `{-1, 0, +1}` trits with per-row
/// absmean scales — the scheme real b1.58 models are trained into via the
/// straight-through estimator. Without a shape, a single global scale is used.
#[cfg(feature = "std")]
pub fn quantize_ternary(weights: &[f64], rows: Option<usize>, cols: Option<usize>) -> Quantized {
    let n = weights.len();
    let mut trits = vec![0i8; n];

    fn quantize(v: f64, scale: f64) -> i8 {
        let q = (v / scale).round();
        if q > 0.0 {
            1
        } else if q < 0.0 {
            -1
        } else {
            0
        }
    }

    if let (Some(rows), Some(cols)) = (rows, cols)
        && rows * cols == n
        && rows > 0
        && cols > 0
    {
        let mut scales = vec![0f64; rows];
        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let mean: f64 = row.iter().map(|v| v.abs()).sum::<f64>() / cols as f64;
            let scale = if mean == 0.0 { 1.0 } else { mean };
            scales[r] = scale;
            for c in 0..cols {
                trits[r * cols + c] = quantize(row[c], scale);
            }
        }
        return Quantized {
            trits,
            scales: Some(scales),
            gamma: 1.0,
        };
    }

    let mean: f64 = weights.iter().map(|v| v.abs()).sum::<f64>() / n.max(1) as f64;
    let scale = if mean == 0.0 { 1.0 } else { mean };
    for (i, &v) in weights.iter().enumerate() {
        trits[i] = quantize(v, scale);
    }
    Quantized {
        trits,
        scales: None,
        gamma: 1.0,
    }
}

/// Dense float64 reference — the same math over unpacked weights, used by the
/// tests to prove the ternary path and by the bench as the accuracy baseline.
#[inline]
pub fn dense_matmul(weights: &[f64], rows: usize, cols: usize, x: &[f64], out: &mut [f64]) {
    debug_assert!(weights.len() >= rows * cols);
    debug_assert!(x.len() >= rows);
    debug_assert!(out.len() >= cols);

    for slot in out[..cols].iter_mut() {
        *slot = 0.0;
    }
    for (j, &xv) in x.iter().take(rows).enumerate() {
        if xv == 0.0 {
            continue;
        }
        let row = &weights[j * cols..(j + 1) * cols];
        for (i, &w) in row.iter().enumerate() {
            out[i] = fma(w, xv, out[i]);
        }
    }
}

/// Relative L2 error between two vectors, as a fraction of the reference's
/// L2 norm. Returns 0 for identical vectors.
#[inline]
pub fn relative_error(actual: &[f64], reference: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (a, r) in actual.iter().zip(reference.iter()) {
        let d = a - r;
        num += d * d;
        den += r * r;
    }
    if den == 0.0 {
        if num == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        sqrt(num / den)
    }
}
