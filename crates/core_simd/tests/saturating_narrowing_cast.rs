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
