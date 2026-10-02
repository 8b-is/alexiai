//! Round-trip and format tests for the ternary packing.

use gaia_mlx_quant::{
    BITS_PER_WEIGHT, COMPRESSION_VS_F32, POW3, TRITS_PER_BYTE, pack_ternary, pack_ternary_into,
    unpack_ternary, unpack_ternary_into,
};

#[test]
fn bits_per_weight_is_log2_3() {
    assert!((BITS_PER_WEIGHT - 3f64.log2()).abs() < 1e-12);
}

// Checked at compile time: the packing must stay decisively denser than f32.
const _: () = assert!(COMPRESSION_VS_F32 > 20.0);

#[test]
fn place_values_are_base_three() {
    assert_eq!(POW3, [1, 3, 9, 27, 81]);
}

#[test]
fn digit_mapping_is_the_bithack() {
    for t in [-1i8, 0, 1] {
        let d = gaia_mlx_quant::digit_of(t);
        let expected = if t == -1 { 2 } else { t as u8 };
        assert_eq!(d, expected, "digit_of({t})");
        assert_eq!(gaia_mlx_quant::trit_of(d), t);
    }
}

#[test]
fn round_trips_exactly_at_every_length_modulo_5() {
    for n in 0..=32usize {
        let trits: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let packed = pack_ternary(&trits);
        assert_eq!(packed.len(), n.div_ceil(TRITS_PER_BYTE), "n={n}");
        assert_eq!(unpack_ternary(&packed, n), trits, "n={n}");
    }
}

#[test]
fn round_trips_a_five_byte_pattern() {
    // A deliberate mix including the trailing-byte edge.
    let trits = [
        1i8, -1, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 1, 1, 1, -1, 0, 1,
    ];
    let packed = pack_ternary(&trits);
    assert_eq!(unpack_ternary(&packed, trits.len()), trits);
}

#[test]
fn unused_trailing_slots_stay_zero() {
    let trits = [1i8, 1];
    let packed = pack_ternary(&trits);
    // digits 1 + 1*3 = 4; slots 2..4 must be zero.
    assert_eq!(packed[0], 4);
    assert_eq!(unpack_ternary(&packed, 5), [1, 1, 0, 0, 0]);
}

#[test]
fn zero_alloc_path_matches_convenience_path() {
    let trits: Vec<i8> = (0..1000).map(|i| [-1, 0, 1][i % 3]).collect();
    let mut into = vec![0u8; trits.len().div_ceil(TRITS_PER_BYTE)];
    pack_ternary_into(&trits, &mut into);
    assert_eq!(into, pack_ternary(&trits));

    let mut unpacked = vec![0i8; trits.len()];
    unpack_ternary_into(&into, trits.len(), &mut unpacked);
    assert_eq!(unpacked, trits);
}

#[test]
fn byte_zero_means_five_zero_trits() {
    let mut packed = [0u8; 1];
    pack_ternary_into(&[0i8; 5], &mut packed);
    assert_eq!(packed[0], 0);
    assert_eq!(unpack_ternary(&packed, 5), [0i8; 5]);
}

#[test]
fn byte_242_is_all_minus_one() {
    // 2 + 2*3 + 2*9 + 2*27 + 2*81 = 242 — five -1 digits.
    let packed = [242u8];
    assert_eq!(unpack_ternary(&packed, 5), [-1i8; 5]);
}
