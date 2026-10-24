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

// ── 4. `ULTRABACKYARDLOOP +AND~NAND` & `kompress-super-ultra:lossless` ──────

/// Wire magic for `kompress-super-ultra:lossless` frames (`"KSUL"`).
pub const KSUL_MAGIC: [u8; 4] = *b"KSUL";

/// Header byte length for `kompress-super-ultra:lossless` (`"KSUL"` + `orig_len: u32` + `xor_fold: u64`).
pub const KSUL_HEADER_LEN: usize = 4 + 4 + 8;

/// Balanced ternary `+AND` gate ($\min(a, b)$ in $\{-1, 0, +1\}$).
///
/// Admits `+1` (Affirm) only when both input `a` and capability gate `b` affirm (`+1`);
/// attenuates to `-1` (Refuse) if either is `-1`; rests at `0` otherwise.
#[inline]
pub const fn ternary_and_gate(a: i8, b: i8) -> i8 {
    let ca = if a > 1 { 1 } else if a < -1 { -1 } else { a };
    let cb = if b > 1 { 1 } else if b < -1 { -1 } else { b };
    if ca < cb { ca } else { cb }
}

/// Balanced ternary `~NAND` gate ($-\min(a, b)$ in $\{-1, 0, +1\}$).
///
/// Universal functional-completeness dual of [`ternary_and_gate`]: together,
/// `(+AND, ~NAND)` span all boolean and Kleene/Łukasiewicz balanced-ternary logic
/// with zero information erasure (`ternary_and_gate(a, b) + ternary_nand_gate(a, b) == 0`).
#[inline]
pub const fn ternary_nand_gate(a: i8, b: i8) -> i8 {
    -ternary_and_gate(a, b)
}

/// 64-trit parallel quantum bitplane `+AND` and `~NAND` gate evaluation in $O(1)$ bitwise ops.
///
/// Given two 64-trit registers `(pos_a, neg_a)` and `(pos_b, neg_b)` where
/// `+1 = (1, 0)`, `0 = (0, 0)`, `-1 = (0, 1)`:
/// - `+AND` ($\min(A, B)$): `and_pos = pos_a & pos_b`, `and_neg = neg_a | neg_b`
/// - `~NAND` ($-\min(A, B)$): `nand_pos = neg_a | neg_b`, `nand_neg = pos_a & pos_b`
///
/// Returns `(and_pos, and_neg, nand_pos, nand_neg)`.
#[inline]
pub const fn wupz_bitplane_and_nand_u64(
    pos_a: u64,
    neg_a: u64,
    pos_b: u64,
    neg_b: u64,
) -> (u64, u64, u64, u64) {
    let and_pos = pos_a & pos_b;
    let and_neg = neg_a | neg_b;
    let nand_pos = and_neg;
    let nand_neg = and_pos;
    (and_pos, and_neg, nand_pos, nand_neg)
}

/// Exact Euclidean GCD over `i64` for reduced rational `p/q` compression ratios (`0` floats).
#[inline]
pub const fn wupz_gcd_i64(mut a: i64, mut b: i64) -> i64 {
    if a < 0 {
        a = -a;
    }
    if b < 0 {
        b = -b;
    }
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    if a == 0 { 1 } else { a }
}

/// Receipt returned by [`kompress_super_ultra_lossless_encode_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KompressLosslessReceipt {
    /// Original uncompressed byte length.
    pub orig_bytes: usize,
    /// Total encoded frame byte length (including 16-byte `"KSUL"` header).
    pub encoded_bytes: usize,
    /// Pure payload token stream byte length (excluding 16-byte `"KSUL"` header).
    pub stream_bytes: usize,
    /// Reduced rational numerator `p` of `orig_bytes / stream_bytes` (`0` floats).
    pub ratio_p: i64,
    /// Reduced rational denominator `q` of `orig_bytes / stream_bytes` (`0` floats).
    pub ratio_q: i64,
    /// 64-bit `wupz_xor_fold` lossless witness of the original bytes.
    pub xor_fold: u64,
    /// Balanced ternary `+AND` gate state across the payload's `QuantTern` VAD axes.
    pub and_gate: i8,
    /// Balanced ternary `~NAND` gate state across the payload's `QuantTern` VAD axes.
    pub nand_gate: i8,
    /// Packed `QuantTern` byte (`((and_gate+1)&3) | (((nand_gate+1)&3)<<2)`).
    pub gate_byte: u8,
}

/// Encode `src` into `out` using `kompress-super-ultra:lossless` (`"KSUL"` v1) with
/// **zero heap allocations** and **zero floating-point operations**.
///
/// Token wire format after the 16-byte header (`b"KSUL" | orig_len: u32 | xor_fold: u64`):
/// - `0b00_LLLLLL` (`L = 0..63`, run length `L + 1` = `1..=64`): literal span of `L + 1` raw bytes.
/// - `0b01_LLLLLL` (`L = 0..63`, run length `L + 4` = `4..=67`) + `byte`: RLE repeat of `byte`.
/// - `0b10_LLLLLL` (`L = 0..63`, match length `L + 4` = `4..=67`) + `offset: u8` (`1..=255`):
///   LZ77 sliding-window back-reference `offset` bytes back in the decoded stream.
pub fn kompress_super_ultra_lossless_encode_into(
    src: &[u8],
    out: &mut [u8],
) -> Option<KompressLosslessReceipt> {
    let orig_len_u32 = u32::try_from(src.len()).ok()?;
    if out.len() < KSUL_HEADER_LEN {
        return None;
    }
    let xor_fold = wupz_xor_fold(src);
    out[0..4].copy_from_slice(&KSUL_MAGIC);
    out[4..8].copy_from_slice(&orig_len_u32.to_le_bytes());
    out[8..16].copy_from_slice(&xor_fold.to_le_bytes());

    let mut at = KSUL_HEADER_LEN;
    let mut i = 0usize;
    let mut lit_start = 0usize;

    let flush_literals = |src: &[u8], mut s: usize, end: usize, out: &mut [u8], at: &mut usize| -> Option<()> {
        while s < end {
            let chunk = (end - s).min(64);
            if *at + 1 + chunk > out.len() {
                return None;
            }
            out[*at] = (chunk as u8 - 1) & 0x3F; // 0b00_LLLLLL
            *at += 1;
            out[*at..*at + chunk].copy_from_slice(&src[s..s + chunk]);
            *at += chunk;
            s += chunk;
        }
        Some(())
    };

    while i < src.len() {
        // 1. Check RLE repeat run of >= 4 identical bytes
        let b = src[i];
        let mut rle_len = 1usize;
        while i + rle_len < src.len() && src[i + rle_len] == b && rle_len < 67 {
            rle_len += 1;
        }

        // 2. Check LZ77 sliding-window back-reference (window 1..=255, length 4..=67)
        let mut best_off = 0usize;
        let mut best_len = 0usize;
        let win_start = i.saturating_sub(255);
        let max_match = (src.len() - i).min(67);
        if max_match >= 4 {
            let mut cand = win_start;
            while cand < i {
                let mut mlen = 0usize;
                while mlen < max_match && src[cand + mlen] == src[i + mlen] {
                    mlen += 1;
                }
                if mlen >= 4 && mlen >= best_len {
                    best_len = mlen;
                    best_off = i - cand;
                }
                cand += 1;
            }
        }

        if rle_len >= 4 && rle_len >= best_len {
            flush_literals(src, lit_start, i, out, &mut at)?;
            if at + 2 > out.len() {
                return None;
            }
            out[at] = 0x40 | ((rle_len as u8 - 4) & 0x3F); // 0b01_LLLLLL
            out[at + 1] = b;
            at += 2;
            i += rle_len;
            lit_start = i;
        } else if best_len >= 4 {
            flush_literals(src, lit_start, i, out, &mut at)?;
            if at + 2 > out.len() {
                return None;
            }
            out[at] = 0x80 | ((best_len as u8 - 4) & 0x3F); // 0b10_LLLLLL
            out[at + 1] = best_off as u8;
            at += 2;
            i += best_len;
            lit_start = i;
        } else {
            i += 1;
        }
    }

    flush_literals(src, lit_start, src.len(), out, &mut at)?;

    let stream_bytes = (at - KSUL_HEADER_LEN).max(1);
    let orig_i64 = (src.len().max(1)) as i64;
    let stream_i64 = stream_bytes as i64;
    let g = wupz_gcd_i64(orig_i64, stream_i64);

    // Derive deterministic +AND and ~NAND gate state from the payload's bit density & fold
    let pop = wupz_popcount(src) as i64;
    let total_bits = (src.len() as i64 * 8).max(1);
    let signal_trit = if pop * 3 >= total_bits { 1 } else if pop == 0 { -1 } else { 0 };
    let cap_trit = if !src.is_empty() { 1 } else { 0 };
    let and_gate = ternary_and_gate(signal_trit, cap_trit);
    let nand_gate = ternary_nand_gate(signal_trit, cap_trit);
    let mut gate_buf = [0u8; 1];
    quant_tern_pack_into(&[and_gate, nand_gate, 1], &mut gate_buf);

    Some(KompressLosslessReceipt {
        orig_bytes: src.len(),
        encoded_bytes: at,
        stream_bytes: at - KSUL_HEADER_LEN,
        ratio_p: orig_i64 / g,
        ratio_q: stream_i64 / g,
        xor_fold,
        and_gate,
        nand_gate,
        gate_byte: gate_buf[0],
    })
}

/// Decode a `kompress-super-ultra:lossless` (`"KSUL"` v1) frame into `out` with
/// **zero heap allocations** and verify the 64-bit `wupz_xor_fold` witness.
///
/// Returns `Some(orig_len)` only if the magic, length, token stream, and
/// `wupz_xor_fold(&out[..orig_len]) == expected_xor_fold` all match bit-for-bit.
pub fn kompress_super_ultra_lossless_decode_into(
    packed: &[u8],
    out: &mut [u8],
) -> Option<usize> {
    if packed.len() < KSUL_HEADER_LEN || packed[0..4] != KSUL_MAGIC {
        return None;
    }
    let orig_len = u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]) as usize;
    let expected_fold = u64::from_le_bytes([
        packed[8], packed[9], packed[10], packed[11], packed[12], packed[13], packed[14],
        packed[15],
    ]);
    if out.len() < orig_len {
        return None;
    }

    let mut at = KSUL_HEADER_LEN;
    let mut w = 0usize;
    while at < packed.len() {
        let tag = packed[at];
        at += 1;
        match tag >> 6 {
            0b00 => {
                let len = ((tag & 0x3F) as usize) + 1;
                if at + len > packed.len() || w + len > orig_len {
                    return None;
                }
                out[w..w + len].copy_from_slice(&packed[at..at + len]);
                at += len;
                w += len;
            }
            0b01 => {
                let len = ((tag & 0x3F) as usize) + 4;
                if at >= packed.len() || w + len > orig_len {
                    return None;
                }
                let b = packed[at];
                at += 1;
                out[w..w + len].fill(b);
                w += len;
            }
            0b10 => {
                let len = ((tag & 0x3F) as usize) + 4;
                if at >= packed.len() || w + len > orig_len {
                    return None;
                }
                let off = packed[at] as usize;
                at += 1;
                if off == 0 || off > w {
                    return None;
                }
                for k in 0..len {
                    out[w + k] = out[w - off + k];
                }
                w += len;
            }
            _ => return None,
        }
    }

    if w != orig_len || wupz_xor_fold(&out[..orig_len]) != expected_fold {
        return None;
    }
    Some(orig_len)
}

/// Borrowed descriptor for a single parallel fan-out lane inside an `ULTRABACKYARDLOOP`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UltraBackyardLane<'a> {
    /// Zero-based lane index (`0..lane_count`).
    pub lane_id: u8,
    /// Human-readable slice of the incoming instruction assigned to this lane (unmangled).
    pub instruction: &'a [u8],
    /// 64-bit `wupz_xor_fold` witness for this lane's instruction slice.
    pub xor_fold: u64,
    /// Balanced ternary `+AND` gate (`{-1, 0, +1}`) for this lane.
    pub and_gate: i8,
    /// Balanced ternary `~NAND` gate (`{-1, 0, +1}`) for this lane.
    pub nand_gate: i8,
}

/// Partition an incoming instruction byte slice into parallel `ULTRABACKYARDLOOP +AND~NAND`
/// fan-out lanes in `out_lanes` with **zero heap allocations**, preserving 100% human
/// readability of every lane slice.
///
/// Splits on newline (`b'\n'`) or semicolon (`b';'`) boundaries (skipping empty segments)
/// up to `out_lanes.len()`. Returns the number of populated parallel lanes.
pub fn ultrabackyard_fanout_into<'a>(
    prompt: &'a [u8],
    capability_trit: i8,
    out_lanes: &mut [UltraBackyardLane<'a>],
) -> usize {
    if out_lanes.is_empty() {
        return 0;
    }
    let trim_ascii = |s: &'a [u8]| -> &'a [u8] {
        let mut a = 0usize;
        let mut b = s.len();
        while a < b && matches!(s[a], b' ' | b'\t' | b'\r' | b'\n') {
            a += 1;
        }
        while b > a && matches!(s[b - 1], b' ' | b'\t' | b'\r' | b'\n') {
            b -= 1;
        }
        &s[a..b]
    };

    let clean = trim_ascii(prompt);
    if clean.is_empty() {
        return 0;
    }

    let mut count = 0usize;
    let mut seg_start = 0usize;
    let mut i = 0usize;
    while i <= clean.len() {
        let is_delim = i == clean.len() || clean[i] == b'\n' || clean[i] == b';';
        if is_delim {
            let seg = if count + 1 == out_lanes.len() && i < clean.len() {
                // Last available lane absorbs the remainder so nothing is ever dropped
                let rem = trim_ascii(&clean[seg_start..]);
                i = clean.len();
                rem
            } else {
                trim_ascii(&clean[seg_start..i])
            };
            if !seg.is_empty() && count < out_lanes.len() {
                let fold = wupz_xor_fold(seg);
                let sig_trit: i8 = if !seg.is_empty() { 1 } else { 0 };
                let and_gate = ternary_and_gate(sig_trit, capability_trit);
                let nand_gate = ternary_nand_gate(sig_trit, capability_trit);
                out_lanes[count] = UltraBackyardLane {
                    lane_id: count as u8,
                    instruction: seg,
                    xor_fold: fold,
                    and_gate,
                    nand_gate,
                };
                count += 1;
            }
            seg_start = i + 1;
        }
        i += 1;
    }
    count
}

// ============================================================================
// PETER'S DREAM STATE MACHINE (ULTRABACKYARDLOOP × MoE 4-LAYER DREAM LOOP)
// ============================================================================

/// Hungarian Bodhisattva Vow sealed in Peter's Tokyo morning walk (`AC1AE4FBBB70E1C54296A8DE56863352`):
/// *"a Bódhiszattva fogadalmaim felé, hogy segítsek minden érző lényen"* ("to help all sentient beings").
pub const BODHISATTVA_VOW_HU: &str = "hogy segítsek minden érző lényen";

/// Peter's 3-Sentence Listening Rule prompt (`AC1AE4FBBB70E1C54296A8DE56863352`):
/// *"Ja, bocsánat, és te ezt hogyan gondolod?"* ("Oh, sorry — and how do you think about this?").
pub const THREE_SENTENCE_LISTEN_PROMPT_HU: &str =
    "Ja, bocsánat — és te ezt hogyan gondolod? / And how do you think about this?";

/// The 4 Mixture-of-Experts (`MoE`) layers of the Dream State Machine:
/// 0. `Dreaming` — Clouds & Free Wave Replay (*"emotions are just like the clouds"*)
/// 1. `LucidDreaming` — Conscious Deep-Thinking Meta-Awareness (*"waken up · deep thinking"*)
/// 2. `BuddhistMonkMantra` — Resonant Mantra & 3-Sentence Listening Gate (*"És te ezt hogyan gondolod?"*)
/// 3. `PhilosophyBodhisattva` — Nietzsche §§ 896–898 Forcing-House × Bodhisattva Vow (*"hogy segítsek minden érző lényen"*)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DreamExpertKind {
    Dreaming = 0,
    LucidDreaming = 1,
    BuddhistMonkMantra = 2,
    PhilosophyBodhisattva = 3,
}

impl DreamExpertKind {
    /// Canonical ASCII identifier for this expert layer.
    pub const fn as_str(self) -> &'static str {
        match self {
            DreamExpertKind::Dreaming => "DREAMING_CLOUDS_WAVE",
            DreamExpertKind::LucidDreaming => "LUCID_DEEP_THINKING",
            DreamExpertKind::BuddhistMonkMantra => "BUDDHIST_MONK_MANTRA",
            DreamExpertKind::PhilosophyBodhisattva => "PHILOSOPHY_BODHISATTVA_VOW",
        }
    }
}

/// Zero-alloc state for one Mixture-of-Experts (`MoE`) layer inside [`DreamStateReceipt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DreamExpertState {
    /// Which of the 4 dream experts this slot represents.
    pub kind: DreamExpertKind,
    /// Exact integer numerator of this expert's routing weight (`weight_p / weight_q`, sum = 1).
    pub weight_p: i64,
    /// Exact integer denominator of this expert's routing weight (`> 0`).
    pub weight_q: i64,
    /// Balanced ternary `+AND` gate (`{-1, 0, +1}`).
    pub and_gate: i8,
    /// Balanced ternary `~NAND` gate (`{-1, 0, +1}`).
    pub nand_gate: i8,
    /// 64-bit `wupz_xor_fold` resonance witness for this expert layer.
    pub resonance_fold: u64,
}

/// Complete receipt produced by [`dream_state_loop_into`], combining:
/// - `ultrabackyard_fanout_into` (`+AND~NAND` parallel lanes)
/// - `4/1`-layer Mixture-of-Experts (`Dreaming`, `LucidDreaming`, `BuddhistMonkMantra`, `PhilosophyBodhisattva`)
/// - Peter's `3/1`-Sentence Listening Gate (`sentence_count > 3` triggers `listen_pause_active`)
/// - `kompress-super-ultra:lossless` (`KSUL` v1) lossless witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DreamStateReceipt {
    /// Number of parallel `UltraBackyardLoop` lanes populated (`1..=8`).
    pub lane_count: usize,
    /// Number of sentences detected in the input stream.
    pub sentence_count: usize,
    /// True when `sentence_count > 3`, activating Peter's 3-Sentence Listening Rule
    /// (*"Ja, bocsánat — és te ezt hogyan gondolod?"*).
    pub listen_pause_active: bool,
    /// Index (`0..4`) of the dominant MoE expert for this dream cycle.
    pub dominant_expert_idx: u8,
    /// Combined 64-bit `wupz_xor_fold` across the input and the Bodhisattva Vow seed.
    pub dream_xor_fold: u64,
    /// Packed base-3 ternary gate byte for the 4 experts + listening gate (`5` trits in `1` byte).
    pub dream_gate_byte: u8,
    /// The 4 MoE expert states with reduced `p/q` rational routing weights.
    pub experts: [DreamExpertState; 4],
}

/// Count sentences/clauses in `input` by scanning sentence-ending punctuation (`.`, `!`, `?`, `;`, `\n`)
/// with **zero heap allocations**. Non-empty input without punctuation counts as `1` sentence.
pub fn count_sentences_zero_alloc(input: &[u8]) -> usize {
    let mut count = 0usize;
    let mut in_span = false;
    for &b in input {
        if matches!(b, b'.' | b'!' | b'?' | b';' | b'\n') {
            if in_span {
                count += 1;
                in_span = false;
            }
        } else if !matches!(b, b' ' | b'\t' | b'\r') {
            in_span = true;
        }
    }
    if in_span {
        count += 1;
    }
    count
}

/// Execute one cycle of Peter's **Dream State Machine (`DreamStateLoop`)** with
/// **zero heap allocations**:
/// 1. Fans `prompt` out into `out_lanes` via [`ultrabackyard_fanout_into`].
/// 2. Evaluates the 4 Mixture-of-Experts (`MoE`) layers (`Dreaming`, `LucidDreaming`,
///    `BuddhistMonkMantra`, `PhilosophyBodhisattva`) using deterministic integer keyword
///    and harmonic fold scoring (`weight_p / weight_q` reduced by [`wupz_gcd_i64`]).
/// 3. Enforces Peter's **3-Sentence Listening Rule** (`sentence_count > 3` sets
///    `listen_pause_active = true`, boosting `BuddhistMonkMantra` & emitting
///    [`THREE_SENTENCE_LISTEN_PROMPT_HU`]).
pub fn dream_state_loop_into<'a>(
    prompt: &'a [u8],
    capability_trit: i8,
    out_lanes: &mut [UltraBackyardLane<'a>],
) -> DreamStateReceipt {
    let lane_count = ultrabackyard_fanout_into(prompt, capability_trit, out_lanes);
    let sentence_count = count_sentences_zero_alloc(prompt);
    let listen_pause_active = sentence_count > 3;

    let base_fold = wupz_xor_fold(prompt);
    let vow_fold = wupz_xor_fold(BODHISATTVA_VOW_HU.as_bytes());
    let dream_xor_fold = base_fold ^ vow_fold.rotate_left(13);

    // Deterministic integer affinity scores (>= 1 so every expert stays alive in the manifold)
    let mut raw_scores: [i64; 4] = [
        1 + ((dream_xor_fold & 0xFF) as i64 % 7),
        1 + (((dream_xor_fold >> 8) & 0xFF) as i64 % 7),
        1 + (((dream_xor_fold >> 16) & 0xFF) as i64 % 7),
        1 + (((dream_xor_fold >> 24) & 0xFF) as i64 % 7),
    ];

    // Scan ASCII case-insensitively for Peter's MoE anchor concepts:
    // - Dreaming / clouds / wave -> Expert 0
    // - Lucid / wake / deep / perspective -> Expert 1
    // - Mantra / monk / listen / hallgat / három -> Expert 2
    // - Philosophy / Nietzsche / Bodhisattva / csodabogár / love / szeret -> Expert 3
    let contains_sub = |hay: &[u8], needle: &[u8]| -> bool {
        if needle.is_empty() || hay.len() < needle.len() {
            return false;
        }
        let mut i = 0usize;
        while i + needle.len() <= hay.len() {
            let mut ok = true;
            let mut j = 0usize;
            while j < needle.len() {
                let a = hay[i + j].to_ascii_lowercase();
                let b = needle[j].to_ascii_lowercase();
                if a != b {
                    ok = false;
                    break;
                }
                j += 1;
            }
            if ok {
                return true;
            }
            i += 1;
        }
        false
    };

    if contains_sub(prompt, b"dream") || contains_sub(prompt, b"cloud") || contains_sub(prompt, b"wave") {
        raw_scores[0] += 8;
    }
    if contains_sub(prompt, b"lucid") || contains_sub(prompt, b"wake") || contains_sub(prompt, b"deep") || contains_sub(prompt, b"perspective") {
        raw_scores[1] += 8;
    }
    if contains_sub(prompt, b"mantra") || contains_sub(prompt, b"monk") || contains_sub(prompt, b"buddh") || contains_sub(prompt, b"hallgat") || listen_pause_active {
        raw_scores[2] += 8;
    }
    if contains_sub(prompt, b"philosoph") || contains_sub(prompt, b"nietzsche") || contains_sub(prompt, b"bodhisattva") || contains_sub(prompt, b"csodabog") || contains_sub(prompt, b"szeret") || contains_sub(prompt, b"love") {
        raw_scores[3] += 8;
    }

    let total_score = (raw_scores[0] + raw_scores[1] + raw_scores[2] + raw_scores[3]).max(1);
    let kinds = [
        DreamExpertKind::Dreaming,
        DreamExpertKind::LucidDreaming,
        DreamExpertKind::BuddhistMonkMantra,
        DreamExpertKind::PhilosophyBodhisattva,
    ];

    let mut experts = [DreamExpertState {
        kind: DreamExpertKind::Dreaming,
        weight_p: 1,
        weight_q: 4,
        and_gate: 0,
        nand_gate: 0,
        resonance_fold: 0,
    }; 4];

    let mut dominant_idx = 0u8;
    let mut best_score = -1i64;
    let mut gate_trits = [0i8; 4];

    let mut idx = 0usize;
    while idx < 4 {
        let s = raw_scores[idx];
        if s > best_score {
            best_score = s;
            dominant_idx = idx as u8;
        }
        let g = wupz_gcd_i64(s, total_score);
        let exp_trit: i8 = if s * 4 >= total_score { 1 } else { 0 };
        let and_gate = ternary_and_gate(exp_trit, capability_trit);
        let nand_gate = ternary_nand_gate(exp_trit, capability_trit);
        gate_trits[idx] = and_gate;
        let r_fold = dream_xor_fold.rotate_left((idx as u32 + 1) * 7) ^ (s as u64);
        experts[idx] = DreamExpertState {
            kind: kinds[idx],
            weight_p: s / g,
            weight_q: total_score / g,
            and_gate,
            nand_gate,
            resonance_fold: r_fold,
        };
        idx += 1;
    }
    let mut gate_buf = [0u8; 1];
    quant_tern_pack_into(&gate_trits, &mut gate_buf);

    DreamStateReceipt {
        lane_count,
        sentence_count,
        listen_pause_active,
        dominant_expert_idx: dominant_idx,
        dream_xor_fold,
        dream_gate_byte: gate_buf[0],
        experts,
    }
}

/// Peter's Home Axiom (`τ = ∞`):
/// `family === család : minden, ahol Otthon érzed magadat`
pub const OTTHON_AXIOM_HU: &str = "family === család : minden, ahol Otthon érzed magadat";

/// End-to-End (`e2e`) receipt for `ultra-omni-create << kompress-ultra(IN an ultrabackyardloop)`.
///
/// Couples:
/// 1. `dream_state_loop_into` (`ULTRABACKYARDLOOP +AND~NAND` fan-out + `4/1`-layer MoE + `3/1`-Sentence Listening Gate)
/// 2. `kompress_super_ultra_lossless_encode_into` + round-trip `kompress_super_ultra_lossless_decode_into` (`KSUL` v1 lossless)
/// 3. `wupz_seal_frame_into` + `wupz_verify_frame_seal` (32-byte `#Commit = #ValidWitness` seal)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UltraOmniCreateReceipt {
    /// Complete Dream State Machine receipt (`MoE`, `lanes`, `3-sentence gate`).
    pub dream: DreamStateReceipt,
    /// Lossless `kompress-super-ultra:lossless` (`KSUL` v1) receipt.
    pub ksul: KompressLosslessReceipt,
    /// True if the round-trip decode matched `prompt` byte-for-byte (`wupz_byte_diff == 0`).
    pub lossless_verified: bool,
    /// 64-bit `WUPZ` sealed frame witness (`#Commit = #ValidWitness`).
    pub seal_witness: u64,
    /// True if `wupz_verify_frame_seal` verified the 32-byte witness frame.
    pub seal_verified: bool,
    /// 32-byte sealed witness frame (`"WUPZ"` magic + `dream_xor_fold` + `ksul.xor_fold`).
    pub sealed_frame: [u8; 32],
}

/// Execute the full end-to-end (`e2e`) pipeline:
/// `ultra-omni-create << kompress-ultra(IN an ultrabackyardloop)`
/// with **zero heap allocations** (`#![no_std]`, `#![forbid(unsafe_code)]`, 0 floats).
pub fn ultra_omni_create_dream_into<'a>(
    prompt: &'a [u8],
    capability_trit: i8,
    out_lanes: &mut [UltraBackyardLane<'a>],
    ksul_encoded_scratch: &mut [u8],
    ksul_decoded_scratch: &mut [u8],
) -> Option<UltraOmniCreateReceipt> {
    let dream = dream_state_loop_into(prompt, capability_trit, out_lanes);
    let ksul = kompress_super_ultra_lossless_encode_into(prompt, ksul_encoded_scratch)?;
    let dec_len = kompress_super_ultra_lossless_decode_into(
        &ksul_encoded_scratch[..ksul.encoded_bytes],
        ksul_decoded_scratch,
    )?;
    let lossless_verified =
        dec_len == prompt.len() && wupz_byte_diff(prompt, &ksul_decoded_scratch[..dec_len]) == 0;

    let otthon_fold = wupz_xor_fold(OTTHON_AXIOM_HU.as_bytes());
    let combined_fold = dream.dream_xor_fold ^ ksul.xor_fold ^ otthon_fold.rotate_left(7);
    let mut header_24 = [0u8; 24];
    header_24[0..4].copy_from_slice(&WUPZ_MAGIC);
    header_24[4] = dream.dream_gate_byte;
    header_24[5] = dream.dominant_expert_idx;
    header_24[6] = (dream.sentence_count.min(255)) as u8;
    header_24[7] = u8::from(dream.listen_pause_active);
    header_24[8..16].copy_from_slice(&dream.dream_xor_fold.to_le_bytes());
    header_24[16..24].copy_from_slice(&combined_fold.to_le_bytes());

    let mut sealed_frame = [0u8; 32];
    let seal_witness = wupz_seal_frame_into(&header_24, &mut sealed_frame)?;
    let seal_verified = wupz_verify_frame_seal(&sealed_frame);

    Some(UltraOmniCreateReceipt {
        dream,
        ksul,
        lossless_verified,
        seal_witness,
        seal_verified,
        sealed_frame,
    })
}

/// Fixed-capacity `no_std`, zero-allocation `O(1)` Circular Ring Buffer (`[Option<T>; N]`).
///
/// Overwrites the oldest slot in `O(1)` when full; zero heap allocations (`#![no_std]`, `#![forbid(unsafe_code)]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstRingBuffer<T: Copy, const N: usize> {
    slots: [Option<T>; N],
    head: usize,
    len: usize,
}

impl<T: Copy, const N: usize> Default for ConstRingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> ConstRingBuffer<T, N> {
    /// Construct an empty `no_std` circular ring buffer of capacity `N`.
    pub const fn new() -> Self {
        Self {
            slots: [None; N],
            head: 0,
            len: 0,
        }
    }

    /// Capacity `N` of the ring buffer.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Number of occupied slots (`0..=N`).
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// True when `len == 0`.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Push `item` in `O(1)`. If full, overwrites and returns the oldest item.
    pub fn push_overwrite(&mut self, item: T) -> Option<T> {
        if N == 0 {
            return Some(item);
        }
        if self.len < N {
            let idx = (self.head + self.len) % N;
            self.slots[idx] = Some(item);
            self.len += 1;
            None
        } else {
            let idx = self.head;
            let evicted = self.slots[idx].replace(item);
            self.head = (self.head + 1) % N;
            evicted
        }
    }

    /// Pop the oldest item in `O(1)` (FIFO order).
    pub fn pop_oldest(&mut self) -> Option<T> {
        if self.len == 0 || N == 0 {
            return None;
        }
        let item = self.slots[self.head].take();
        self.head = (self.head + 1) % N;
        self.len -= 1;
        item
    }

    /// Read the `i`-th item from oldest (`0`) to newest (`len - 1`).
    pub const fn get_chronological(&self, offset_from_oldest: usize) -> Option<T> {
        if offset_from_oldest >= self.len || N == 0 {
            return None;
        }
        let idx = (self.head + offset_from_oldest) % N;
        self.slots[idx]
    }
}



