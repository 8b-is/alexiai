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

use crate::pack::{BYTE_TRITS_I32, POW3, T0, T1, T2, T3, T4, TRITS_PER_BYTE, digit_of};

/// Default rational denominator `q = 2^20 = 1,048,576` for pure-integer `p/q` kernels.
pub const KERNEL_PQ_Q: i32 = 1_048_576;

/// Maximum stack-buffered dimension for zero-allocation `p/q` pre-scaling.
const MAX_STACK_DIM: usize = 2048;

#[cfg(feature = "std")]
use alloc::vec;
#[cfg(feature = "std")]
use alloc::vec::Vec;

/// Exact integer square root `floor(sqrt(n))` for `u128` in pure `core` Rust.
/// Zero floating-point instructions.
#[inline]
pub const fn isqrt_u128(n: u128) -> u64 {
    if n == 0 {
        return 0;
    }
    let shift = (127 - n.leading_zeros()) & !1;
    let mut bit = 1u128 << shift;
    let mut res = 0u128;
    let mut rem = n;
    while bit != 0 {
        let cand = res + bit;
        res >>= 1;
        if rem >= cand {
            rem -= cand;
            res += bit;
        }
        bit >>= 2;
    }
    res as u64
}

/// Pure-integer per-row absmean ternary quantization in `p/q` ("Counting Divided").
///
/// Quantizes row-major integer numerators `w_p` (`[rows, cols]`) into `trits_out`
/// (`{-1, 0, +1}`) and per-row absmean scale numerators `scales_p_out`.
/// Uses the exact integer inequality `|w| * 2 * cols >= sum_abs` so the inner
/// loop executes **zero floating-point operations and zero per-weight divisions**.
#[inline]
pub fn quantize_ternary_pq_into(
    w_p: &[i32],
    rows: usize,
    cols: usize,
    q: i32,
    trits_out: &mut [i8],
    scales_p_out: &mut [i32],
) {
    debug_assert!(w_p.len() >= rows * cols);
    debug_assert!(trits_out.len() >= rows * cols);
    debug_assert!(scales_p_out.len() >= rows);
    if rows == 0 || cols == 0 {
        return;
    }
    let cols_i64 = cols as i64;
    let two_cols = cols_i64 * 2;
    for r in 0..rows {
        let row = &w_p[r * cols..(r + 1) * cols];
        let dst = &mut trits_out[r * cols..(r + 1) * cols];
        let mut sum_abs: i64 = 0;
        for &v in row {
            sum_abs += (v as i64).abs();
        }
        if sum_abs == 0 {
            scales_p_out[r] = q;
            for t in dst.iter_mut() {
                *t = 0;
            }
            continue;
        }
        scales_p_out[r] = ((sum_abs + (cols_i64 >> 1)) / cols_i64) as i32;
        for (c, &v) in row.iter().enumerate() {
            let v64 = v as i64;
            let abs2c = v64.abs() * two_cols;
            dst[c] = if abs2c < sum_abs {
                0
            } else if v64 > 0 {
                1
            } else {
                -1
            };
        }
    }
}

/// Fuse `p_x[j]` and optional `p_scales[j]` (over `q_scale`) into a pre-scaled
/// `i32` row vector `px_fused[..rows]`. Zero floating-point operations.
#[inline]
fn fuse_pq_row_scales(
    p_x: &[i32],
    p_scales: Option<&[i32]>,
    q_scale: i32,
    rows: usize,
    px_fused: &mut [i32],
) {
    let q_s = (q_scale.max(1)) as i64;
    let half_q = q_s >> 1;
    if let Some(scales) = p_scales {
        for j in 0..rows {
            let prod = (p_x[j] as i64) * (scales[j] as i64);
            px_fused[j] = if prod >= 0 {
                ((prod + half_q) / q_s) as i32
            } else {
                ((prod - half_q) / q_s) as i32
            };
        }
    } else {
        px_fused[..rows].copy_from_slice(&p_x[..rows]);
    }
}

/// Pure-integer j-major `b1.58` (5 trits/byte) matrix-vector kernel in `p/q`.
///
/// Accumulates exact `i64` numerators into `p_out[..cols]` with **zero
/// floating-point operations**. Uses [`BYTE_TRITS_I32`] and strips per-slot
/// bounds guards across all interior bytes of each row.
#[inline]
pub fn ternary_matmul_pq_j_major_into(
    packed: &[u8],
    rows: usize,
    cols: usize,
    p_x: &[i32],
    p_scales: Option<&[i32]>,
    q_scale: i32,
    p_out: &mut [i64],
) {
    debug_assert!(p_x.len() >= rows);
    debug_assert!(p_out.len() >= cols);
    debug_assert!(packed.len() * TRITS_PER_BYTE >= rows * cols);
    debug_assert!(p_scales.map(|s| s.len() >= rows).unwrap_or(true));

    for slot in p_out[..cols].iter_mut() {
        *slot = 0;
    }
    if rows == 0 || cols == 0 {
        return;
    }

    let q_s = (q_scale.max(1)) as i64;
    let half_q = q_s >> 1;
    let row_scale = |j: usize| -> i64 {
        let xv = p_x[j] as i64;
        if let Some(s) = p_scales {
            let prod = xv * (s[j] as i64);
            if prod >= 0 {
                (prod + half_q) / q_s
            } else {
                (prod - half_q) / q_s
            }
        } else {
            xv
        }
    };

    if cols.is_multiple_of(TRITS_PER_BYTE) {
        let bytes_per_row = cols / TRITS_PER_BYTE;
        for j in 0..rows {
            let xj = row_scale(j);
            if xj == 0 {
                continue;
            }
            let row = &packed[j * bytes_per_row..(j + 1) * bytes_per_row];
            for (b_idx, &byte) in row.iter().enumerate() {
                if byte == 0 {
                    continue;
                }
                let trits = &BYTE_TRITS_I32[byte as usize];
                let dst = &mut p_out[b_idx * TRITS_PER_BYTE..b_idx * TRITS_PER_BYTE + TRITS_PER_BYTE];
                dst[0] += (trits[0] as i64) * xj;
                dst[1] += (trits[1] as i64) * xj;
                dst[2] += (trits[2] as i64) * xj;
                dst[3] += (trits[3] as i64) * xj;
                dst[4] += (trits[4] as i64) * xj;
            }
        }
        return;
    }

    // General path when cols % 5 != 0: handle first partial byte, unguarded
    // interior full bytes, and final partial byte.
    for j in 0..rows {
        let xj = row_scale(j);
        if xj == 0 {
            continue;
        }
        let start_t = j * cols;
        let end_t = start_t + cols;
        let start_b = start_t / TRITS_PER_BYTE;
        let end_b = end_t.div_ceil(TRITS_PER_BYTE);

        for (b, &byte) in packed.iter().enumerate().take(end_b).skip(start_b) {
            if byte == 0 {
                continue;
            }
            let base = b * TRITS_PER_BYTE;
            let trits = &BYTE_TRITS_I32[byte as usize];
            if base >= start_t && base + TRITS_PER_BYTE <= end_t {
                let col = base - start_t;
                let dst = &mut p_out[col..col + TRITS_PER_BYTE];
                dst[0] += (trits[0] as i64) * xj;
                dst[1] += (trits[1] as i64) * xj;
                dst[2] += (trits[2] as i64) * xj;
                dst[3] += (trits[3] as i64) * xj;
                dst[4] += (trits[4] as i64) * xj;
            } else {
                for (slot_idx, &trit) in trits.iter().enumerate() {
                    let t = base + slot_idx;
                    if t >= start_t && t < end_t {
                        p_out[t - start_t] += (trit as i64) * xj;
                    }
                }
            }
        }
    }
}

/// Pure-integer i-major `b1.58` (5 trits/byte) matrix-vector kernel in `p/q`.
///
/// Uses **4-column register blocking** and [`BYTE_TRITS_I32`] so `p_x` is loaded
/// once per 4 output columns and accumulated in 32-bit integer chunks with
/// **zero floating-point instructions**.
#[inline]
pub fn ternary_matmul_pq_i_major_into(
    packed: &[u8],
    rows: usize,
    cols: usize,
    p_x: &[i32],
    p_scales: Option<&[i32]>,
    q_scale: i32,
    p_out: &mut [i64],
) {
    debug_assert!(p_x.len() >= rows);
    debug_assert!(p_out.len() >= cols);
    debug_assert!(packed.len() * TRITS_PER_BYTE >= rows * cols);
    debug_assert!(p_scales.map(|s| s.len() >= rows).unwrap_or(true));

    if rows == 0 || cols == 0 {
        for slot in p_out[..cols].iter_mut() {
            *slot = 0;
        }
        return;
    }

    let mut px_stack: [i32; MAX_STACK_DIM];
    let px: &[i32] = match p_scales {
        Some(scales) if rows <= MAX_STACK_DIM => {
            px_stack = [0i32; MAX_STACK_DIM];
            fuse_pq_row_scales(p_x, Some(scales), q_scale, rows, &mut px_stack);
            &px_stack[..rows]
        }
        _ => &p_x[..rows],
    };

    // Fast path when rows % 5 == 0: every column has an exact integer number of
    // bytes (`bytes_per_col = rows / 5`), so columns are byte-aligned and we
    // can 4-column block without any bit-offset shifting.
    if rows.is_multiple_of(TRITS_PER_BYTE) && (rows <= MAX_STACK_DIM || p_scales.is_none()) {
        let bytes_per_col = rows / TRITS_PER_BYTE;
        // 12 bytes = 60 trits <= 64 trits per i32 chunk so i32 never overflows.
        const CHUNK_BYTES: usize = 12;

        let mut i = 0usize;
        while i + 4 <= cols {
            let col0 = &packed[i * bytes_per_col..(i + 1) * bytes_per_col];
            let col1 = &packed[(i + 1) * bytes_per_col..(i + 2) * bytes_per_col];
            let col2 = &packed[(i + 2) * bytes_per_col..(i + 3) * bytes_per_col];
            let col3 = &packed[(i + 3) * bytes_per_col..(i + 4) * bytes_per_col];

            let mut tot0: i64 = 0;
            let mut tot1: i64 = 0;
            let mut tot2: i64 = 0;
            let mut tot3: i64 = 0;

            let mut b_base = 0usize;
            while b_base < bytes_per_col {
                let b_end = (b_base + CHUNK_BYTES).min(bytes_per_col);
                let mut a0: i32 = 0;
                let mut a1: i32 = 0;
                let mut a2: i32 = 0;
                let mut a3: i32 = 0;

                for b in b_base..b_end {
                    let x5 = &px[b * TRITS_PER_BYTE..b * TRITS_PER_BYTE + TRITS_PER_BYTE];
                    let (x0, x1, x2, x3, x4) = (x5[0], x5[1], x5[2], x5[3], x5[4]);

                    let t0 = &BYTE_TRITS_I32[col0[b] as usize];
                    a0 = a0.wrapping_add(
                        t0[0]
                            .wrapping_mul(x0)
                            .wrapping_add(t0[1].wrapping_mul(x1))
                            .wrapping_add(t0[2].wrapping_mul(x2))
                            .wrapping_add(t0[3].wrapping_mul(x3))
                            .wrapping_add(t0[4].wrapping_mul(x4)),
                    );

                    let t1 = &BYTE_TRITS_I32[col1[b] as usize];
                    a1 = a1.wrapping_add(
                        t1[0]
                            .wrapping_mul(x0)
                            .wrapping_add(t1[1].wrapping_mul(x1))
                            .wrapping_add(t1[2].wrapping_mul(x2))
                            .wrapping_add(t1[3].wrapping_mul(x3))
                            .wrapping_add(t1[4].wrapping_mul(x4)),
                    );

                    let t2 = &BYTE_TRITS_I32[col2[b] as usize];
                    a2 = a2.wrapping_add(
                        t2[0]
                            .wrapping_mul(x0)
                            .wrapping_add(t2[1].wrapping_mul(x1))
                            .wrapping_add(t2[2].wrapping_mul(x2))
                            .wrapping_add(t2[3].wrapping_mul(x3))
                            .wrapping_add(t2[4].wrapping_mul(x4)),
                    );

                    let t3 = &BYTE_TRITS_I32[col3[b] as usize];
                    a3 = a3.wrapping_add(
                        t3[0]
                            .wrapping_mul(x0)
                            .wrapping_add(t3[1].wrapping_mul(x1))
                            .wrapping_add(t3[2].wrapping_mul(x2))
                            .wrapping_add(t3[3].wrapping_mul(x3))
                            .wrapping_add(t3[4].wrapping_mul(x4)),
                    );
                }

                tot0 += a0 as i64;
                tot1 += a1 as i64;
                tot2 += a2 as i64;
                tot3 += a3 as i64;
                b_base = b_end;
            }

            p_out[i] = tot0;
            p_out[i + 1] = tot1;
            p_out[i + 2] = tot2;
            p_out[i + 3] = tot3;
            i += 4;
        }

        while i < cols {
            let col = &packed[i * bytes_per_col..(i + 1) * bytes_per_col];
            let mut tot: i64 = 0;
            let mut b_base = 0usize;
            while b_base < bytes_per_col {
                let b_end = (b_base + CHUNK_BYTES).min(bytes_per_col);
                let mut a0: i32 = 0;
                for b in b_base..b_end {
                    let x5 = &px[b * TRITS_PER_BYTE..b * TRITS_PER_BYTE + TRITS_PER_BYTE];
                    let t = &BYTE_TRITS_I32[col[b] as usize];
                    a0 = a0.wrapping_add(
                        t[0].wrapping_mul(x5[0])
                            .wrapping_add(t[1].wrapping_mul(x5[1]))
                            .wrapping_add(t[2].wrapping_mul(x5[2]))
                            .wrapping_add(t[3].wrapping_mul(x5[3]))
                            .wrapping_add(t[4].wrapping_mul(x5[4])),
                    );
                }
                tot += a0 as i64;
                b_base = b_end;
            }
            p_out[i] = tot;
            i += 1;
        }
        return;
    }

    // General path when rows % 5 != 0 (columns share boundary bytes in the flat stream):
    // strip per-trit bounds checks across all interior full bytes of each column.
    for (i, slot) in p_out[..cols].iter_mut().enumerate() {
        let start = i * rows;
        let end = start + rows;
        let start_b = start / TRITS_PER_BYTE;
        let end_b = end.div_ceil(TRITS_PER_BYTE);

        let mut acc: i64 = 0;
        for (b, &byte) in packed.iter().enumerate().take(end_b).skip(start_b) {
            if byte == 0 {
                continue;
            }
            let base = b * TRITS_PER_BYTE;
            let trits = &BYTE_TRITS_I32[byte as usize];
            if base >= start && base + TRITS_PER_BYTE <= end {
                let j = base - start;
                let x5 = &px[j..j + TRITS_PER_BYTE];
                let sum5 = trits[0]
                    .wrapping_mul(x5[0])
                    .wrapping_add(trits[1].wrapping_mul(x5[1]))
                    .wrapping_add(trits[2].wrapping_mul(x5[2]))
                    .wrapping_add(trits[3].wrapping_mul(x5[3]))
                    .wrapping_add(trits[4].wrapping_mul(x5[4]));
                acc += sum5 as i64;
            } else {
                for (slot_idx, &trit) in trits.iter().enumerate() {
                    let t = base + slot_idx;
                    if t >= start && t < end {
                        acc += (trit as i64) * (px[t - start] as i64);
                    }
                }
            }
        }
        *slot = acc;
    }
}

/// Pure-integer i-major sign-planes matrix-vector kernel in `p/q`.
///
/// Uses **4-column register blocking** over 64-element associative `i32`
/// chunks so LLVM auto-vectorizes the inner loop with 128-bit NEON (`mla.4s`)
/// or AVX2 (`vpmulld`) instructions with **zero floating-point instructions**.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn ternary_matmul_pq_planes_i_into(
    pos: &[i8],
    neg: &[i8],
    rows: usize,
    cols: usize,
    p_x: &[i32],
    p_scales: Option<&[i32]>,
    q_scale: i32,
    p_out: &mut [i64],
) {
    debug_assert!(pos.len() >= rows * cols && neg.len() >= rows * cols);
    debug_assert!(p_x.len() >= rows);
    debug_assert!(p_out.len() >= cols);
    debug_assert!(p_scales.map(|s| s.len() >= rows).unwrap_or(true));

    if rows == 0 || cols == 0 {
        for slot in p_out[..cols].iter_mut() {
            *slot = 0;
        }
        return;
    }

    let mut px_stack: [i32; MAX_STACK_DIM];
    let px: &[i32] = match p_scales {
        Some(scales) if rows <= MAX_STACK_DIM => {
            px_stack = [0i32; MAX_STACK_DIM];
            fuse_pq_row_scales(p_x, Some(scales), q_scale, rows, &mut px_stack);
            &px_stack[..rows]
        }
        _ => &p_x[..rows],
    };

    const CHUNK: usize = 64;
    let mut i = 0usize;
    while i + 4 <= cols {
        let p0 = &pos[i * rows..(i + 1) * rows];
        let n0 = &neg[i * rows..(i + 1) * rows];
        let p1 = &pos[(i + 1) * rows..(i + 2) * rows];
        let n1 = &neg[(i + 1) * rows..(i + 2) * rows];
        let p2 = &pos[(i + 2) * rows..(i + 3) * rows];
        let n2 = &neg[(i + 2) * rows..(i + 3) * rows];
        let p3 = &pos[(i + 3) * rows..(i + 4) * rows];
        let n3 = &neg[(i + 3) * rows..(i + 4) * rows];

        let mut tot0: i64 = 0;
        let mut tot1: i64 = 0;
        let mut tot2: i64 = 0;
        let mut tot3: i64 = 0;

        let mut j_base = 0usize;
        while j_base < rows {
            let j_end = (j_base + CHUNK).min(rows);
            let mut a0: i32 = 0;
            let mut a1: i32 = 0;
            let mut a2: i32 = 0;
            let mut a3: i32 = 0;

            let xc = &px[j_base..j_end];
            let len = xc.len();
            let p0c = &p0[j_base..j_end][..len];
            let n0c = &n0[j_base..j_end][..len];
            let p1c = &p1[j_base..j_end][..len];
            let n1c = &n1[j_base..j_end][..len];
            let p2c = &p2[j_base..j_end][..len];
            let n2c = &n2[j_base..j_end][..len];
            let p3c = &p3[j_base..j_end][..len];
            let n3c = &n3[j_base..j_end][..len];

            for k in 0..len {
                let xk = xc[k];
                let t0 = p0c[k].wrapping_sub(n0c[k]) as i32;
                let t1 = p1c[k].wrapping_sub(n1c[k]) as i32;
                let t2 = p2c[k].wrapping_sub(n2c[k]) as i32;
                let t3 = p3c[k].wrapping_sub(n3c[k]) as i32;
                a0 = a0.wrapping_add(t0.wrapping_mul(xk));
                a1 = a1.wrapping_add(t1.wrapping_mul(xk));
                a2 = a2.wrapping_add(t2.wrapping_mul(xk));
                a3 = a3.wrapping_add(t3.wrapping_mul(xk));
            }

            tot0 += a0 as i64;
            tot1 += a1 as i64;
            tot2 += a2 as i64;
            tot3 += a3 as i64;
            j_base = j_end;
        }

        p_out[i] = tot0;
        p_out[i + 1] = tot1;
        p_out[i + 2] = tot2;
        p_out[i + 3] = tot3;
        i += 4;
    }

    while i < cols {
        let p_col = &pos[i * rows..(i + 1) * rows];
        let n_col = &neg[i * rows..(i + 1) * rows];
        let mut tot: i64 = 0;
        let mut j_base = 0usize;
        while j_base < rows {
            let j_end = (j_base + CHUNK).min(rows);
            let mut a0: i32 = 0;
            for ((&p, &m), &xk) in p_col[j_base..j_end]
                .iter()
                .zip(n_col[j_base..j_end].iter())
                .zip(px[j_base..j_end].iter())
            {
                let t = p.wrapping_sub(m) as i32;
                a0 = a0.wrapping_add(t.wrapping_mul(xk));
            }
            tot += a0 as i64;
            j_base = j_end;
        }
        p_out[i] = tot;
        i += 1;
    }
}

/// Pure-integer single-plane `i8` ternary matrix-vector kernel (`Y[i] = sum_j W[i, j] * X[j]`)
/// with **4-row register blocking** and 64-element associative `i32` SIMD chunks.
/// Zero floating-point operations.
#[inline]
pub fn ternary_simd_i8_pq_into(
    trits_i_major: &[i8],
    rows: usize,
    cols: usize,
    p_x: &[i32],
    p_scales: Option<&[i32]>,
    q_scale: i32,
    p_out: &mut [i64],
) {
    debug_assert!(trits_i_major.len() >= rows * cols);
    debug_assert!(p_x.len() >= rows);
    debug_assert!(p_out.len() >= cols);
    debug_assert!(p_scales.map(|s| s.len() >= rows).unwrap_or(true));

    if rows == 0 || cols == 0 {
        for slot in p_out[..cols].iter_mut() {
            *slot = 0;
        }
        return;
    }

    let mut px_stack: [i32; MAX_STACK_DIM];
    let px: &[i32] = match p_scales {
        Some(scales) if rows <= MAX_STACK_DIM => {
            px_stack = [0i32; MAX_STACK_DIM];
            fuse_pq_row_scales(p_x, Some(scales), q_scale, rows, &mut px_stack);
            &px_stack[..rows]
        }
        _ => &p_x[..rows],
    };

    const CHUNK: usize = 64;
    let mut i = 0usize;
    while i + 4 <= cols {
        let w0 = &trits_i_major[i * rows..(i + 1) * rows];
        let w1 = &trits_i_major[(i + 1) * rows..(i + 2) * rows];
        let w2 = &trits_i_major[(i + 2) * rows..(i + 3) * rows];
        let w3 = &trits_i_major[(i + 3) * rows..(i + 4) * rows];

        let mut tot0: i64 = 0;
        let mut tot1: i64 = 0;
        let mut tot2: i64 = 0;
        let mut tot3: i64 = 0;

        let mut j_base = 0usize;
        while j_base < rows {
            let j_end = (j_base + CHUNK).min(rows);
            let mut a0: i32 = 0;
            let mut a1: i32 = 0;
            let mut a2: i32 = 0;
            let mut a3: i32 = 0;

            let xc = &px[j_base..j_end];
            let len = xc.len();
            let w0c = &w0[j_base..j_end][..len];
            let w1c = &w1[j_base..j_end][..len];
            let w2c = &w2[j_base..j_end][..len];
            let w3c = &w3[j_base..j_end][..len];

            for k in 0..len {
                let xk = xc[k];
                a0 = a0.wrapping_add((w0c[k] as i32).wrapping_mul(xk));
                a1 = a1.wrapping_add((w1c[k] as i32).wrapping_mul(xk));
                a2 = a2.wrapping_add((w2c[k] as i32).wrapping_mul(xk));
                a3 = a3.wrapping_add((w3c[k] as i32).wrapping_mul(xk));
            }

            tot0 += a0 as i64;
            tot1 += a1 as i64;
            tot2 += a2 as i64;
            tot3 += a3 as i64;
            j_base = j_end;
        }

        p_out[i] = tot0;
        p_out[i + 1] = tot1;
        p_out[i + 2] = tot2;
        p_out[i + 3] = tot3;
        i += 4;
    }

    while i < cols {
        let w_col = &trits_i_major[i * rows..(i + 1) * rows];
        let mut tot: i64 = 0;
        let mut j_base = 0usize;
        while j_base < rows {
            let j_end = (j_base + CHUNK).min(rows);
            let mut a0: i32 = 0;
            for (&t, &xk) in w_col[j_base..j_end].iter().zip(px[j_base..j_end].iter()) {
                a0 = a0.wrapping_add((t as i32).wrapping_mul(xk));
            }
            tot += a0 as i64;
            j_base = j_end;
        }
        p_out[i] = tot;
        i += 1;
    }
}

/// Pure-integer dense reference matmul in `p/q` (`p_out[i] = sum_j p_w[j, i] * p_x[j]`).
/// Zero floating-point operations.
#[inline]
pub fn dense_matmul_pq_into(
    p_weights: &[i32],
    rows: usize,
    cols: usize,
    p_x: &[i32],
    p_out: &mut [i64],
) {
    debug_assert!(p_weights.len() >= rows * cols);
    debug_assert!(p_x.len() >= rows);
    debug_assert!(p_out.len() >= cols);

    for slot in p_out[..cols].iter_mut() {
        *slot = 0;
    }
    for (j, &xv) in p_x.iter().take(rows).enumerate() {
        if xv == 0 {
            continue;
        }
        let xv64 = xv as i64;
        let row = &p_weights[j * cols..(j + 1) * cols];
        for (i, &w) in row.iter().enumerate() {
            p_out[i] += (w as i64) * xv64;
        }
    }
}

/// Pure-integer relative L2 error in parts-per-million (`ppm`, $q = 10^6$)
/// using [`isqrt_u128`]. Zero floating-point operations.
#[inline]
pub fn relative_error_pq_ppm(actual: &[i64], reference: &[i64]) -> u64 {
    let mut num: u128 = 0;
    let mut den: u128 = 0;
    for (&a, &r) in actual.iter().zip(reference.iter()) {
        // Shift down by 10 bits if values are large (e.g. q^2 = 2^40) to prevent u128 overflow.
        let da = (a - r).unsigned_abs() as u128;
        let ra = r.unsigned_abs() as u128;
        let d_scaled = if da > (1u128 << 48) { da >> 16 } else { da };
        let r_scaled = if ra > (1u128 << 48) { ra >> 16 } else { ra };
        num = num.saturating_add(d_scaled.saturating_mul(d_scaled));
        den = den.saturating_add(r_scaled.saturating_mul(r_scaled));
    }
    if den == 0 {
        return if num == 0 { 0 } else { u64::MAX };
    }
    let num_root = isqrt_u128(num);
    let den_root = isqrt_u128(den);
    if den_root == 0 {
        return u64::MAX;
    }
    ((num_root as u128 * 1_000_000u128) / (den_root as u128)) as u64
}

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

    // ── general path: rows may cross byte boundaries (interior bytes unguarded) ──
    for (j, &xv) in x.iter().take(rows).enumerate() {
        if xv == 0.0 {
            continue;
        }
        let xj = xv * scales.map(|s| s[j]).unwrap_or(1.0) * gamma;

        let start_t = j * cols;
        let end_t = start_t + cols;
        let start_b = start_t / TRITS_PER_BYTE;
        let end_b = end_t.div_ceil(TRITS_PER_BYTE);

        for (b, &byte) in packed[start_b..end_b].iter().enumerate() {
            if byte == 0 {
                continue;
            }
            let base = (start_b + b) * TRITS_PER_BYTE;
            let idx = byte as usize;
            if base >= start_t && base + TRITS_PER_BYTE <= end_t {
                let col = base - start_t;
                out[col] = fma(T0[idx] as f64, xj, out[col]);
                out[col + 1] = fma(T1[idx] as f64, xj, out[col + 1]);
                out[col + 2] = fma(T2[idx] as f64, xj, out[col + 2]);
                out[col + 3] = fma(T3[idx] as f64, xj, out[col + 3]);
                out[col + 4] = fma(T4[idx] as f64, xj, out[col + 4]);
            } else {
                let t0 = base;
                if t0 >= start_t && t0 < end_t {
                    let i0 = t0 - start_t;
                    out[i0] = fma(T0[idx] as f64, xj, out[i0]);
                }
                let t1 = base + 1;
                if t1 >= start_t && t1 < end_t {
                    let i1 = t1 - start_t;
                    out[i1] = fma(T1[idx] as f64, xj, out[i1]);
                }
                let t2 = base + 2;
                if t2 >= start_t && t2 < end_t {
                    let i2 = t2 - start_t;
                    out[i2] = fma(T2[idx] as f64, xj, out[i2]);
                }
                let t3 = base + 3;
                if t3 >= start_t && t3 < end_t {
                    let i3 = t3 - start_t;
                    out[i3] = fma(T3[idx] as f64, xj, out[i3]);
                }
                let t4 = base + 4;
                if t4 >= start_t && t4 < end_t {
                    let i4 = t4 - start_t;
                    out[i4] = fma(T4[idx] as f64, xj, out[i4]);
                }
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
/// trits stream once, `x * scale` is pre-fused once on the stack (`O(rows)`
/// instead of `O(rows * cols)`), interior 5-trit bytes run without bounds
/// guards, and `out` is written exactly once.
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

    let mut xj_stack = [0.0f64; MAX_STACK_DIM];
    if rows <= MAX_STACK_DIM {
        if let Some(sc) = scales {
            for j in 0..rows {
                xj_stack[j] = x[j] * sc[j];
            }
        } else {
            xj_stack[..rows].copy_from_slice(&x[..rows]);
        }
        let xs = &xj_stack[..rows];
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
                let idx = byte as usize;
                if base >= start && base + TRITS_PER_BYTE <= end {
                    let j = base - start;
                    acc = fma(T0[idx] as f64, xs[j], acc);
                    acc = fma(T1[idx] as f64, xs[j + 1], acc);
                    acc = fma(T2[idx] as f64, xs[j + 2], acc);
                    acc = fma(T3[idx] as f64, xs[j + 3], acc);
                    acc = fma(T4[idx] as f64, xs[j + 4], acc);
                } else {
                    if base < end && base >= start {
                        acc = fma(T0[idx] as f64, xs[base - start], acc);
                    }
                    let t1 = base + 1;
                    if t1 < end && t1 >= start {
                        acc = fma(T1[idx] as f64, xs[t1 - start], acc);
                    }
                    let t2 = base + 2;
                    if t2 < end && t2 >= start {
                        acc = fma(T2[idx] as f64, xs[t2 - start], acc);
                    }
                    let t3 = base + 3;
                    if t3 < end && t3 >= start {
                        acc = fma(T3[idx] as f64, xs[t3 - start], acc);
                    }
                    let t4 = base + 4;
                    if t4 < end && t4 >= start {
                        acc = fma(T4[idx] as f64, xs[t4 - start], acc);
                    }
                }
            }
            *slot = acc * gamma;
        }
        return;
    }

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

/// The plane kernel — branchless, pre-fused row scales, register accumulators, 4-way ILP.
///
/// Pre-fuses `x[j] * scale[j]` once on the stack in `O(rows)`, then runs 4
/// independent accumulator chains per column.
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

    let mut xj_stack = [0.0f64; MAX_STACK_DIM];
    if rows <= MAX_STACK_DIM {
        if let Some(sc) = scales {
            for j in 0..rows {
                xj_stack[j] = x[j] * sc[j];
            }
        } else {
            xj_stack[..rows].copy_from_slice(&x[..rows]);
        }
        let xs = &xj_stack[..rows];
        for (i, slot) in out[..cols].iter_mut().enumerate() {
            let base = i * rows;
            let mut acc0 = 0.0f64;
            let mut acc1 = 0.0f64;
            let mut acc2 = 0.0f64;
            let mut acc3 = 0.0f64;

            let mut j = 0usize;
            while j + 4 <= rows {
                acc0 = fma((pos[base + j] - neg[base + j]) as f64, xs[j], acc0);
                acc1 = fma((pos[base + j + 1] - neg[base + j + 1]) as f64, xs[j + 1], acc1);
                acc2 = fma((pos[base + j + 2] - neg[base + j + 2]) as f64, xs[j + 2], acc2);
                acc3 = fma((pos[base + j + 3] - neg[base + j + 3]) as f64, xs[j + 3], acc3);
                j += 4;
            }
            while j < rows {
                acc0 = fma((pos[base + j] - neg[base + j]) as f64, xs[j], acc0);
                j += 1;
            }
            *slot = (acc0 + acc1 + acc2 + acc3) * gamma;
        }
        return;
    }

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
