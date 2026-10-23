//! `wupz` — Incremental Pack-in-Pass Linker+Compressor & QuantTern Seam (`8b-is/wupz`).
//!
//! Wires the [`8b-is/wupz`](https://github.com/8b-is/wupz) (`w(ild) + UPX + z(ig)`)
//! C ABI v1 kernels (`wupz_xor_fold`, `wupz_byte_diff`, `wupz_popcount`), deterministic
//! `"WUPZ"` plan-image wire format, incremental section/window diffing, and the
//! `wupz.8b.is` `QuantTern` emotional ternary codec (`qt:<hex>`) directly into
//! `gaia-mlx-quant` with:
//! - `#![no_std]` and `#![forbid(unsafe_code)]`
//! - **Zero floating-point operations** (`p/q` rational thresholds with `q = 2^20`)
//! - **Zero heap allocations** on all hot paths (caller-provided slices & stack structs)

use crate::gaia::{FoldPq, GAIA_PQ_Q};

/// Wire magic for deterministic `wupz` layout plan images (`"WUPZ"`).
pub const WUPZ_MAGIC: [u8; 4] = *b"WUPZ";

/// Frozen C ABI version (`include/wupz_abi.h` `WUPZ_ABI_VERSION`).
pub const WUPZ_ABI_VERSION: u32 = 1;

/// Wire format version for `wupz` plan images (`wupz_core::pack::PLAN_VERSION`).
pub const WUPZ_PLAN_VERSION: u8 = 1;

/// Default address-independent packing window (`4096` bytes).
pub const WUPZ_DEFAULT_WINDOW_SIZE: u64 = 4096;

const HEADER_LEN: usize = 4 + 1 + 4 + 4;
const SECTION_FIXED: usize = 4 + 8 + 8 + 8 + 2;
const BLOCK_FIXED: usize = 4 + 4;

// ── 1. The Three `wupz_abi.h` Kernels (`xor_fold`, `byte_diff`, `popcount`) ──

/// Integrity fold (`wupz_xor_fold`): XOR of the little-endian `u64` words, then
/// the trailing bytes.
///
/// Matches `wupz-rs::fold::xor_fold` and `wupz-zig::xorFold` bit-for-bit across
/// every length `0..=usize::MAX`. Order-stable, allocation-free, and its own
/// witness: folding a frame with its 8-byte trailing seal yields `0`.
#[inline]
pub fn wupz_xor_fold(bytes: &[u8]) -> u64 {
    let mut acc: u64 = 0;
    let (chunks32, rem32) = bytes.as_chunks::<32>();
    for c in chunks32 {
        let w0 = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        let w1 = u64::from_le_bytes([c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]]);
        let w2 = u64::from_le_bytes([c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23]]);
        let w3 = u64::from_le_bytes([c[24], c[25], c[26], c[27], c[28], c[29], c[30], c[31]]);
        acc ^= (w0 ^ w1) ^ (w2 ^ w3);
    }
    let (chunks8, rem8) = rem32.as_chunks::<8>();
    for &c in chunks8 {
        acc ^= u64::from_le_bytes(c);
    }
    for &b in rem8 {
        acc ^= b as u64;
    }
    acc
}

/// Direct 64-bit word XOR fold over 64-trit quantum bitplanes (`w_pos_u64`, `w_neg_u64`).
///
/// Equivalent to `wupz_xor_fold` over the little-endian byte representation of `words`.
#[inline]
pub fn wupz_xor_fold_u64(words: &[u64]) -> u64 {
    let mut acc0: u64 = 0;
    let mut acc1: u64 = 0;
    let mut acc2: u64 = 0;
    let mut acc3: u64 = 0;
    let (chunks, rem) = words.as_chunks::<4>();
    for c in chunks {
        acc0 ^= c[0];
        acc1 ^= c[1];
        acc2 ^= c[2];
        acc3 ^= c[3];
    }
    let mut acc = (acc0 ^ acc1) ^ (acc2 ^ acc3);
    for &w in rem {
        acc ^= w;
    }
    acc
}

/// Seal a byte span into `out_frame` by appending its 1-byte or 8-byte `wupz_xor_fold` witness.
///
/// When `payload.len()` is a multiple of 8 and `out_frame.len() >= payload.len() + 8`,
/// appends the 8-byte little-endian `wupz_xor_fold(payload)` so that
/// `wupz_xor_fold(&out_frame[..payload.len() + 8]) == 0` (`#Commit = #ValidWitness`).
#[inline]
pub fn wupz_seal_frame_into(payload: &[u8], out_frame: &mut [u8]) -> Option<u64> {
    if !payload.len().is_multiple_of(8) || out_frame.len() < payload.len() + 8 {
        return None;
    }
    out_frame[..payload.len()].copy_from_slice(payload);
    let seal = wupz_xor_fold(payload);
    out_frame[payload.len()..payload.len() + 8].copy_from_slice(&seal.to_le_bytes());
    Some(seal)
}

/// Verify an 8-byte-aligned sealed frame (`#Commit = #ValidWitness` -> `wupz_xor_fold(frame) == 0`).
#[inline]
pub fn wupz_verify_frame_seal(frame: &[u8]) -> bool {
    frame.len() >= 8 && frame.len().is_multiple_of(8) && wupz_xor_fold(frame) == 0
}

/// Number of differing bytes between two spans (`wupz_byte_diff`).
///
/// Matches `wupz-rs::fold::byte_diff` and `wupz-zig::byteDiff` bit-for-bit.
/// Processes 16-byte chunks so LLVM auto-vectorizes the comparison into 128-bit
/// NEON (`cmeq.16b`) or AVX2 (`vpcmpeqb`) instructions in 100% safe Rust.
#[inline]
pub fn wupz_byte_diff(a: &[u8], b: &[u8]) -> u64 {
    let n = a.len().min(b.len());
    let a_slice = &a[..n];
    let b_slice = &b[..n];
    let mut count: u64 = 0;

    let (a_chunks, a_rem) = a_slice.as_chunks::<16>();
    let (b_chunks, b_rem) = b_slice.as_chunks::<16>();
    for (ca, cb) in a_chunks.iter().zip(b_chunks.iter()) {
        let mut block_diff: u32 = 0;
        for i in 0..16 {
            block_diff += u32::from(ca[i] != cb[i]);
        }
        count += block_diff as u64;
    }
    for (&ba, &bb) in a_rem.iter().zip(b_rem.iter()) {
        count += u64::from(ba != bb);
    }
    count
}

/// Quantum Bitplane Incremental Diff: exact number of differing trits between two
/// 64-trit quantum bitplane weight tensors `(w_pos_a, w_neg_a)` and `(w_pos_b, w_neg_b)`.
///
/// Evaluates 64 trits per cycle using bitwise XOR + OR + hardware `count_ones()` (`POPCNT`),
/// allowing `wupz` incremental repack to identify mutated weight blocks during QAT in $O(N/64)$.
#[inline]
pub fn wupz_bitplane_diff_u64(
    w_pos_a: &[u64],
    w_neg_a: &[u64],
    w_pos_b: &[u64],
    w_neg_b: &[u64],
) -> u64 {
    let n = w_pos_a
        .len()
        .min(w_neg_a.len())
        .min(w_pos_b.len())
        .min(w_neg_b.len());
    let mut count: u64 = 0;
    for (((&pa, &na), &pb), &nb) in w_pos_a[..n]
        .iter()
        .zip(w_neg_a[..n].iter())
        .zip(w_pos_b[..n].iter())
        .zip(w_neg_b[..n].iter())
    {
        count += ((pa ^ pb) | (na ^ nb)).count_ones() as u64;
    }
    count
}

/// Population count over a byte span (`wupz_popcount` density hint).
///
/// Matches `wupz-rs::fold::popcount` and `wupz-zig::popcount` bit-for-bit while
/// folding 8 bytes per `u64::count_ones()` instruction.
#[inline]
pub fn wupz_popcount(bytes: &[u8]) -> u64 {
    let mut count: u64 = 0;
    let (chunks8, rem8) = bytes.as_chunks::<8>();
    for &c in chunks8 {
        count += u64::from_le_bytes(c).count_ones() as u64;
    }
    for &b in rem8 {
        count += b.count_ones() as u64;
    }
    count
}

/// Active non-zero trit count across 64-trit quantum bitplanes `(w_pos_u64, w_neg_u64)`.
#[inline]
pub fn wupz_bitplane_popcount_u64(w_pos: &[u64], w_neg: &[u64]) -> u64 {
    let n = w_pos.len().min(w_neg.len());
    let mut count: u64 = 0;
    for (&pw, &nw) in w_pos[..n].iter().zip(w_neg[..n].iter()) {
        count += (pw | nw).count_ones() as u64;
    }
    count
}

// ── 2. `QuantTern` Emotional Ternary Codec (`qt:<hex>`, 0 Floats) ────────────

/// Numerator of the `wupz` / `qwave` `QuantTern` threshold (`0.33 = 33 / 100`).
pub const QUANT_TERN_THRESHOLD_NUM: i64 = 33;
/// Denominator of the `wupz` / `qwave` `QuantTern` threshold (`0.33 = 33 / 100`).
pub const QUANT_TERN_THRESHOLD_DEN: i64 = 100;

/// Quantize a single `p/q` rational value `p_x / q` to a `QuantTern` trit `{-1, 0, +1}`
/// at threshold `|x| >= 0.33` with **zero floating-point operations**.
///
/// A value with `|x| < 0.33` maps to `0` (a real *don't know*, not a small yes).
#[inline]
pub const fn quant_tern_trit_pq(p_x: i64, q: i64) -> i8 {
    let q_abs = if q < 0 { -q } else { q };
    let lhs = p_x * QUANT_TERN_THRESHOLD_DEN;
    let rhs = QUANT_TERN_THRESHOLD_NUM * q_abs;
    if lhs >= rhs {
        1
    } else if lhs <= -rhs {
        -1
    } else {
        0
    }
}

/// Pack trits `{-1, 0, +1}` into `out` at 4 trits per byte using the `wupz` / `qwave`
/// 2-bit code `((t + 1) & 3) << (2 * j)` (`-1 -> 0`, `0 -> 1`, `+1 -> 2`).
///
/// Trailing slots in the final byte are padded with `0` trits (`code = 1`), matching
/// `wupz.8b.is/main.js` `encode(vad)` bit-for-bit.
/// Returns the number of packed bytes written (`trits.len().div_ceil(4)`).
pub fn quant_tern_pack_into(trits: &[i8], out: &mut [u8]) -> usize {
    let needed = trits.len().div_ceil(4);
    assert!(out.len() >= needed, "quant_tern_pack_into: output slice too small");
    for (byte_idx, slot) in out[..needed].iter_mut().enumerate() {
        let base = byte_idx * 4;
        let mut b: u8 = 0;
        for j in 0..4 {
            let t = if base + j < trits.len() { trits[base + j] } else { 0 };
            let clamped = t.clamp(-1, 1);
            let bits = ((clamped + 1) as u8) & 0x03;
            b |= bits << (2 * j);
        }
        *slot = b;
    }
    needed
}

/// Unpack `count` trits from a `QuantTern` packed byte slice into `out`.
pub fn quant_tern_unpack_into(packed: &[u8], count: usize, out: &mut [i8]) {
    assert!(out.len() >= count, "quant_tern_unpack_into: output slice too small");
    assert!(
        packed.len() >= count.div_ceil(4),
        "quant_tern_unpack_into: packed slice too small"
    );
    for (idx, dst) in out[..count].iter_mut().enumerate() {
        let b = packed[idx / 4];
        let shift = (idx % 4) * 2;
        let code = ((b >> shift) & 0x03) as i8;
        *dst = (code - 1).clamp(-1, 1);
    }
}

/// Stack-allocated `QuantTern` 3-axis (`valence · arousal · dominance`) fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantTernVad {
    /// Quantized trits `[valence, arousal, dominance]` in `{-1, 0, +1}` (4th slot is `0`).
    pub trits: [i8; 3],
    /// Packed `QuantTern` byte (`((t0+1)&3) | (((t1+1)&3)<<2) | (((t2+1)&3)<<4) | (1<<6)`).
    pub byte: u8,
    /// Number of non-zero trits in `trits` (`0..=3`).
    pub magnitude: u8,
    /// ASCII tag `b"qt:XX"` (e.g. `*b"qt:59"`).
    pub hex: [u8; 5],
}

impl QuantTernVad {
    /// Borrow the 5-byte ASCII fingerprint (e.g. `"qt:56"` or `"qt:59"`) as a `&str`.
    #[inline]
    pub fn as_str(&self) -> &str {
        // Guaranteed valid ASCII (`"qt:"` + two lowercase hex digits).
        core::str::from_utf8(&self.hex).unwrap_or("qt:55")
    }

    /// Glyph representation (`'+'`, `'0'`, `'-'`) for each of the 3 axes.
    #[inline]
    pub const fn glyphs(&self) -> [char; 3] {
        [
            Self::trit_glyph(self.trits[0]),
            Self::trit_glyph(self.trits[1]),
            Self::trit_glyph(self.trits[2]),
        ]
    }

    const fn trit_glyph(t: i8) -> char {
        if t > 0 {
            '+'
        } else if t < 0 {
            '-'
        } else {
            '0'
        }
    }
}

const HEX_LOWER: [u8; 16] = *b"0123456789abcdef";

/// Encode 3 rational `p/q` axes (`valence`, `arousal`, `dominance` in `[-q, +q]`) into
/// a stack-allocated [`QuantTernVad`] (`qt:<hex>`) with **zero floats and zero allocations**.
#[inline]
pub fn quant_tern_encode_vad_pq(
    p_valence: i64,
    p_arousal: i64,
    p_dominance: i64,
    q: i64,
) -> QuantTernVad {
    let t0 = quant_tern_trit_pq(p_valence, q);
    let t1 = quant_tern_trit_pq(p_arousal, q);
    let t2 = quant_tern_trit_pq(p_dominance, q);
    let mut byte_buf = [0u8; 1];
    quant_tern_pack_into(&[t0, t1, t2], &mut byte_buf);
    let byte = byte_buf[0];
    let magnitude = u8::from(t0 != 0) + u8::from(t1 != 0) + u8::from(t2 != 0);
    let hex = [
        b'q',
        b't',
        b':',
        HEX_LOWER[(byte >> 4) as usize],
        HEX_LOWER[(byte & 0x0f) as usize],
    ];
    QuantTernVad {
        trits: [t0, t1, t2],
        byte,
        magnitude,
        hex,
    }
}

/// Derive a deterministic `wupz` [`QuantTernVad`] (`qt:<hex>`) directly from a
/// pure-integer [`FoldPq`] (`q = 2^20`) with **zero floats**.
///
/// Mapping:
/// - `valence`: centered coherence + weather warmth in `[-q, +q]`
/// - `arousal`: centered entropy + weather energy in `[-q, +q]`
/// - `dominance`: centered gravity - entropy balance in `[-q, +q]`
#[inline]
pub fn fold_to_quant_tern_pq(f: &FoldPq) -> QuantTernVad {
    let q = GAIA_PQ_Q;
    // Valence: coherence (in [0, q]) + weather deviation around 0.5*q
    let p_valence = ((f.coherence_p - (q / 2)) * 2 + (f.weather_p - (q / 2))).clamp(-q, q);
    // Arousal: weighted entropy + weather centered around 0.5*q
    let p_arousal = (((f.entropy_p * 55 + f.weather_p * 45) / 100) - (q / 2)) * 2;
    // Dominance: gravity stability relative to entropy dispersal
    let p_dominance = (f.gravity_p - q) * 4 + (f.coherence_p - f.entropy_p);
    quant_tern_encode_vad_pq(
        p_valence.clamp(-q, q),
        p_arousal.clamp(-q, q),
        p_dominance.clamp(-q, q),
        q,
    )
}

// ── 3. Zero-Alloc `"WUPZ"` Layout Plan Image & Incremental Section Receipt ───

/// Borrowed descriptor for a resolved `wupz` section (no heap `String` required).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WupzSectionDesc<'a> {
    pub id: u32,
    pub name: &'a [u8],
    pub addr: u64,
    pub len: u64,
    pub hash: u64,
}

/// Borrowed descriptor for a packed `wupz` block covering one or more section IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WupzBlockDesc<'a> {
    pub index: u32,
    pub covers: &'a [u32],
}

/// Compute the deterministic 64-bit FNV-1a plan digest matching
/// `wupz_core::plan::LayoutPlan::digest` byte-for-byte.
#[inline]
pub fn wupz_plan_digest(
    window_size: u64,
    sections: &[WupzSectionDesc<'_>],
    blocks: &[WupzBlockDesc<'_>],
) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(window_size.max(1));
    for s in sections {
        mix(s.id as u64);
        mix(s.addr);
        mix(s.len);
        mix(s.hash);
        for &b in s.name {
            mix(b as u64);
        }
    }
    for b in blocks {
        mix(b.index as u64);
        for &c in b.covers {
            mix(c as u64);
        }
    }
    h
}

/// Exact byte length of a serialized `"WUPZ"` v1 plan image.
#[inline]
pub fn wupz_plan_image_len(
    sections: &[WupzSectionDesc<'_>],
    blocks: &[WupzBlockDesc<'_>],
) -> usize {
    let sec_bytes: usize = sections.iter().map(|s| SECTION_FIXED + s.name.len()).sum();
    let blk_bytes: usize = blocks
        .iter()
        .map(|b| BLOCK_FIXED + b.covers.len() * 4)
        .sum();
    HEADER_LEN + sec_bytes + blk_bytes
}

/// Serialize a `"WUPZ"` v1 layout plan image directly into caller-provided `out`
/// with **zero heap allocations**, matching `wupz_core::pack::pack_into` byte-for-byte.
pub fn wupz_pack_plan_into<'a>(
    sections: &[WupzSectionDesc<'_>],
    blocks: &[WupzBlockDesc<'_>],
    out: &'a mut [u8],
) -> Option<&'a [u8]> {
    u32::try_from(sections.len()).ok()?;
    u32::try_from(blocks.len()).ok()?;
    for s in sections {
        u16::try_from(s.name.len()).ok()?;
    }
    for b in blocks {
        u32::try_from(b.covers.len()).ok()?;
    }
    let total_len = wupz_plan_image_len(sections, blocks);
    if out.len() < total_len {
        return None;
    }

    let mut at = 0usize;
    out[at..at + 4].copy_from_slice(&WUPZ_MAGIC);
    at += 4;
    out[at] = WUPZ_PLAN_VERSION;
    at += 1;
    out[at..at + 4].copy_from_slice(&(sections.len() as u32).to_le_bytes());
    at += 4;
    out[at..at + 4].copy_from_slice(&(blocks.len() as u32).to_le_bytes());
    at += 4;

    for s in sections {
        out[at..at + 4].copy_from_slice(&s.id.to_le_bytes());
        at += 4;
        out[at..at + 8].copy_from_slice(&s.addr.to_le_bytes());
        at += 8;
        out[at..at + 8].copy_from_slice(&s.len.to_le_bytes());
        at += 8;
        out[at..at + 8].copy_from_slice(&s.hash.to_le_bytes());
        at += 8;
        out[at..at + 2].copy_from_slice(&(s.name.len() as u16).to_le_bytes());
        at += 2;
        out[at..at + s.name.len()].copy_from_slice(s.name);
        at += s.name.len();
    }

    for b in blocks {
        out[at..at + 4].copy_from_slice(&b.index.to_le_bytes());
        at += 4;
        out[at..at + 4].copy_from_slice(&(b.covers.len() as u32).to_le_bytes());
        at += 4;
        for &c in b.covers {
            out[at..at + 4].copy_from_slice(&c.to_le_bytes());
            at += 4;
        }
    }

    debug_assert_eq!(at, total_len);
    Some(&out[..total_len])
}

/// Zero-allocation incremental rebuild rule (`wupz_core::incremental::changed_sections`):
/// writes the section IDs in `after` whose `(addr, len, hash)` differ from `before`
/// into `out_ids` and returns the number of changed sections.
pub fn wupz_changed_sections_into(
    before: &[WupzSectionDesc<'_>],
    after: &[WupzSectionDesc<'_>],
    out_ids: &mut [u32],
) -> usize {
    let mut count = 0usize;
    for s in after {
        let same = before
            .iter()
            .find(|prev| prev.id == s.id)
            .is_some_and(|prev| prev.hash == s.hash && prev.addr == s.addr && prev.len == s.len);
        if !same && count < out_ids.len() {
            out_ids[count] = s.id;
            count += 1;
        }
    }
    count
}

/// Build a `WupzSectionDesc` for a packed ternary (`b1.58`) or quantum bitplane weight buffer.
#[inline]
pub fn wupz_section_from_bytes<'a>(
    id: u32,
    name: &'a [u8],
    addr: u64,
    bytes: &[u8],
) -> WupzSectionDesc<'a> {
    WupzSectionDesc {
        id,
        name,
        addr,
        len: bytes.len() as u64,
        hash: wupz_xor_fold(bytes),
    }
}
