#![feature(portable_simd)]

use core_simd::simd::Simd;

#[test]
fn to_mask_maps_zero_and_nonzero_signed_integers() {
    let values = Simd::from_array([0_i32, 4, -2, 0]);
    assert_eq!(values.to_mask().to_array(), [false, true, true, false]);
}

#[test]
fn to_mask_maps_zero_and_nonzero_unsigned_integers() {
    let values = Simd::from_array([0_u32, 4, u32::MAX, 0]);
    assert_eq!(values.to_mask().to_array(), [false, true, true, false]);
}

#[test]
fn to_mask_maps_zero_and_nonzero_floats() {
    let values = Simd::from_array([0.0_f32, -0.0, 1.5, -2.0]);
    assert_eq!(values.to_mask().to_array(), [false, false, true, true]);
}
