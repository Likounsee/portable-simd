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

#[test]
fn to_mask_preserves_f32_subnormals_and_classifies_special_values() {
    let values = Simd::from_array([
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
        -0.0,
        1.0,
    ]);
    assert_eq!(
        values.to_mask().to_array(),
        [true, true, true, true, true, false, false, true]
    );
}

#[test]
fn to_mask_preserves_f64_subnormals_and_classifies_special_values() {
    let values = Simd::from_array([
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x8000_0000_0000_0001),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        0.0,
        -0.0,
        1.0,
    ]);
    assert_eq!(
        values.to_mask().to_array(),
        [true, true, true, true, true, false, false, true]
    );
}
