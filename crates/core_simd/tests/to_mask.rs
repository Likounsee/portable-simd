#![feature(portable_simd)]

use core_simd::simd::{Simd, num::SimdFloat};

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


#[test]
fn float_classification_handles_f32_subnormals_and_boundaries() {
    let values = Simd::from_array([
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        0.0,
        f32::INFINITY,
        f32::NAN,
    ]);

    assert_eq!(
        values.is_subnormal().to_array(),
        [true, true, true, false, false, false, false, false]
    );
    assert_eq!(
        values.is_normal().to_array(),
        [false, false, false, true, true, false, false, false]
    );
}

#[test]
fn float_classification_handles_f64_subnormals_and_boundaries() {
    let values = Simd::from_array([
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x8000_0000_0000_0001),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        0.0,
        f64::INFINITY,
        f64::NAN,
    ]);

    assert_eq!(
        values.is_subnormal().to_array(),
        [true, true, true, false, false, false, false, false]
    );
    assert_eq!(
        values.is_normal().to_array(),
        [false, false, false, true, true, false, false, false]
    );
}


#[test]
fn float_classification_handles_f32_infinities_nans_and_finite_values() {
    let values = Simd::from_array([
        f32::NEG_INFINITY,
        -f32::MAX,
        -1.0,
        -0.0,
        0.0,
        1.0,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ]);

    assert_eq!(
        values.is_nan().to_array(),
        [false, false, false, false, false, false, false, false, true]
    );
    assert_eq!(
        values.is_infinite().to_array(),
        [true, false, false, false, false, false, false, true, false]
    );
    assert_eq!(
        values.is_finite().to_array(),
        [false, true, true, true, true, true, true, false, false]
    );
}

#[test]
fn float_classification_handles_f64_infinities_nans_and_finite_values() {
    let values = Simd::from_array([
        f64::NEG_INFINITY,
        -f64::MAX,
        -1.0,
        -0.0,
        0.0,
        1.0,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ]);

    assert_eq!(
        values.is_nan().to_array(),
        [false, false, false, false, false, false, false, false, true]
    );
    assert_eq!(
        values.is_infinite().to_array(),
        [true, false, false, false, false, false, false, true, false]
    );
    assert_eq!(
        values.is_finite().to_array(),
        [false, true, true, true, true, true, true, false, false]
    );
}
