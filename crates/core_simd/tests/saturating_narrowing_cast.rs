#![feature(portable_simd)]

use core_simd::simd::prelude::*;

#[test]
fn signed_narrowing_cast_saturates_at_destination_bounds() {
    let values = Simd::<i16, 8>::from_array([-400, -129, -128, -1, 0, 127, 128, 400]);
    let narrowed: Simd<i8, 8> = values
        .simd_clamp(Simd::splat(i8::MIN as i16), Simd::splat(i8::MAX as i16))
        .cast();

    assert_eq!(
        narrowed.to_array(),
        [i8::MIN, i8::MIN, i8::MIN, -1, 0, i8::MAX, i8::MAX, i8::MAX]
    );
}

#[test]
fn signed_to_unsigned_narrowing_cast_saturates_at_zero_and_maximum() {
    let values = Simd::<i16, 8>::from_array([-400, -1, 0, 1, 255, 256, 300, i16::MAX]);
    let narrowed: Simd<u8, 8> = values
        .simd_clamp(Simd::splat(u8::MIN as i16), Simd::splat(u8::MAX as i16))
        .cast();

    assert_eq!(
        narrowed.to_array(),
        [0, 0, 0, 1, u8::MAX, u8::MAX, u8::MAX, u8::MAX]
    );
}

#[test]
fn unsigned_narrowing_cast_saturates_at_destination_maximum() {
    let values = Simd::<u16, 8>::from_array([0, 1, 254, 255, 256, 300, 1024, u16::MAX]);
    let narrowed: Simd<u8, 8> = values
        .simd_clamp(Simd::splat(u8::MIN as u16), Simd::splat(u8::MAX as u16))
        .cast();

    assert_eq!(
        narrowed.to_array(),
        [0, 1, 254, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX]
    );
}

#[test]
fn four_lane_signed_narrowing_cast_saturates_at_destination_bounds() {
    // Four-lane casts have historically exposed backend code-generation issues.
    let values = Simd::<i16, 4>::from_array([-129, -128, 127, 128]);
    let narrowed: Simd<i8, 4> = values
        .simd_clamp(Simd::splat(i8::MIN as i16), Simd::splat(i8::MAX as i16))
        .cast();

    assert_eq!(narrowed.to_array(), [i8::MIN, i8::MIN, i8::MAX, i8::MAX]);
}

#[test]
fn four_lane_signed_to_unsigned_narrowing_cast_saturates_at_both_bounds() {
    let values = Simd::<i16, 4>::from_array([-1, 0, 255, 256]);
    let narrowed: Simd<u8, 4> = values
        .simd_clamp(Simd::splat(u8::MIN as i16), Simd::splat(u8::MAX as i16))
        .cast();

    assert_eq!(narrowed.to_array(), [0, 0, u8::MAX, u8::MAX]);
}

#[test]
fn four_lane_unsigned_narrowing_cast_saturates_at_maximum() {
    let values = Simd::<u16, 4>::from_array([0, 255, 256, u16::MAX]);
    let narrowed: Simd<u8, 4> = values
        .simd_clamp(Simd::splat(u8::MIN as u16), Simd::splat(u8::MAX as u16))
        .cast();

    assert_eq!(narrowed.to_array(), [0, u8::MAX, u8::MAX, u8::MAX]);
}

/// Exhaust every possible 16-bit input for the common saturating narrowing
/// conversions. This catches off-by-one bounds and signedness mistakes that
/// a handful of boundary samples can miss.
#[test]
fn exhaustive_i16_saturating_narrowing_matches_scalar_reference() {
    for base in (i16::MIN as i32..=i16::MAX as i32).step_by(8) {
        let input =
            Simd::<i16, 8>::from_array(core::array::from_fn(|lane| (base + lane as i32) as i16));
        let values = input.to_array();

        let signed: Simd<i8, 8> = input
            .simd_clamp(Simd::splat(i8::MIN as i16), Simd::splat(i8::MAX as i16))
            .cast();
        let signed_expected = values.map(|value| value.clamp(i8::MIN as i16, i8::MAX as i16) as i8);
        assert_eq!(signed.to_array(), signed_expected);

        let unsigned: Simd<u8, 8> = input
            .simd_clamp(Simd::splat(0), Simd::splat(u8::MAX as i16))
            .cast();
        let unsigned_expected = values.map(|value| value.clamp(0, u8::MAX as i16) as u8);
        assert_eq!(unsigned.to_array(), unsigned_expected);
    }
}

/// Exhaust every possible u16 input for unsigned saturating narrowing.
#[test]
fn exhaustive_u16_saturating_narrowing_matches_scalar_reference() {
    for base in (0_u32..=u16::MAX as u32).step_by(8) {
        let input =
            Simd::<u16, 8>::from_array(core::array::from_fn(|lane| (base + lane as u32) as u16));
        let values = input.to_array();

        let narrowed: Simd<u8, 8> = input
            .simd_clamp(Simd::splat(0), Simd::splat(u8::MAX as u16))
            .cast();
        let expected = values.map(|value| value.min(u8::MAX as u16) as u8);
        assert_eq!(narrowed.to_array(), expected);
    }
}
