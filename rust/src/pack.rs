//! Ternary b1.58 packing — 5 trits per byte, base-3, `core`-only.
//!
//! ## the format
//!
//! A trit is `-1`, `0`, or `+1`. Five trits pack into one byte because
//! `3^5 = 243 < 256`, which is why b1.58 weighs log₂3 ≈ 1.585 bits per
//! weight. Digits are base-3 little-endian, with `-1 → 2`, `0 → 0`,
//! `+1 → 1`. Unused trailing slots in the final byte are zero.

/// How many trits fit in one byte.
pub const TRITS_PER_BYTE: usize = 5;

#[cfg(feature = "std")]
use alloc::vec;
#[cfg(feature = "std")]
use alloc::vec::Vec;

/// `3^slot` for slots 0..5 — the base-3 place values.
pub const POW3: [u8; TRITS_PER_BYTE] = [1, 3, 9, 27, 81];

/// The bit-hacked digit mapping: `-1 → 2`, `0 → 0`, `+1 → 1`, branchless.
///
/// For `t = -1`, `t as u8` is `0xFF`, and `0xFF + (0xFF >> 7) * 3` wraps to
/// `2`. For `t ∈ {0, 1}` the shift term is zero and the identity holds.
#[inline]
pub const fn digit_of(t: i8) -> u8 {
    (t as u8).wrapping_add((t as u8 >> 7).wrapping_mul(3))
}

/// Inverse of [`digit_of`]: `0 → 0`, `1 → 1`, `2 → -1`.
#[inline]
pub const fn trit_of(digit: u8) -> i8 {
    if digit == 2 {
        -1
    } else {
        digit as i8
    }
}

/// Compile-time evaluated slot LUTs. `T0[b]` is the trit in slot 0 of byte
/// `b`. Five const tables mean the hot path performs *zero* divisions and
/// *zero* modulo operations — the classic byte-LUT bithack, and it is how the
/// kernel stays allocation-free and branch-light.
pub const T0: [i8; 256] = build_slot_lut(0);
pub const T1: [i8; 256] = build_slot_lut(1);
pub const T2: [i8; 256] = build_slot_lut(2);
pub const T3: [i8; 256] = build_slot_lut(3);
pub const T4: [i8; 256] = build_slot_lut(4);

/// The five slot LUTs as a flat reference — handy for parameterized code.
pub const SLOT_LUTS: [[i8; 256]; TRITS_PER_BYTE] = [T0, T1, T2, T3, T4];

/// Build the LUT for one slot, at compile time.
const fn build_slot_lut(slot: usize) -> [i8; 256] {
    let mut out = [0i8; 256];
    let mut b = 0usize;
    while b < 256 {
        let digit = (b / POW3[slot] as usize) % 3;
        out[b] = trit_of(digit as u8);
        b += 1;
    }
    out
}

/// Pack trits into a caller-owned byte buffer. Zero allocations, zero
/// branches per trit — the digit mapping is the const bit-hack above.
///
/// `out` must hold at least `trits.len().div_ceil(5)` bytes and must be
/// **zero-initialized** (digits accumulate with `+=`). Unused slots in the
/// final byte are left zero.
#[inline]
pub fn pack_ternary_into(trits: &[i8], out: &mut [u8]) {
    debug_assert!(out.len() * TRITS_PER_BYTE >= trits.len());
    for (i, &t) in trits.iter().enumerate() {
        debug_assert!(t == -1 || t == 0 || t == 1, "trit out of range");
        out[i / TRITS_PER_BYTE] += POW3[i % TRITS_PER_BYTE] * digit_of(t);
    }
}

/// Unpack bytes into a caller-owned trit buffer. Zero allocations; the hot
/// loop is unrolled over the five slots so each LUT is a static, direct load.
///
/// `out` must hold at least `count` trits; `bytes` must cover them.
#[inline]
pub fn unpack_ternary_into(bytes: &[u8], count: usize, out: &mut [i8]) {
    debug_assert!(bytes.len() * TRITS_PER_BYTE >= count);
    debug_assert!(out.len() >= count);
    for (i, &b) in bytes.iter().enumerate() {
        let base = i * TRITS_PER_BYTE;
        if base >= count {
            break;
        }
        out[base] = T0[b as usize];
        if base + 1 < count {
            out[base + 1] = T1[b as usize];
        }
        if base + 2 < count {
            out[base + 2] = T2[b as usize];
        }
        if base + 3 < count {
            out[base + 3] = T3[b as usize];
        }
        if base + 4 < count {
            out[base + 4] = T4[b as usize];
        }
    }
}

/// Convenience: pack into a fresh [`Vec`] (the one allocating convenience in
/// the crate; the zero-allocation path is [`pack_ternary_into`]).
#[cfg(feature = "std")]
#[inline]
pub fn pack_ternary(trits: &[i8]) -> Vec<u8> {
    let mut out = vec![0u8; trits.len().div_ceil(TRITS_PER_BYTE)];
    pack_ternary_into(trits, &mut out);
    out
}

/// Convenience: unpack into a fresh [`Vec`].
#[cfg(feature = "std")]
#[inline]
pub fn unpack_ternary(bytes: &[u8], count: usize) -> Vec<i8> {
    let mut out = vec![0i8; count];
    unpack_ternary_into(bytes, count, &mut out);
    out
}

/// The exact bits-per-weight of the packing: log₂3. Kept as a const literal
/// because `f64::log2` is not const-stable; the tests pin it against
/// `3f64.log2()`.
pub const BITS_PER_WEIGHT: f64 = 1.5849625007211561814537389439478;

/// Compression ratio versus float32.
pub const COMPRESSION_VS_F32: f64 = 32.0 / BITS_PER_WEIGHT;
