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

#[test]
fn float_sign_predicates_preserve_f32_sign_bits_for_special_values() {
    let values = Simd::from_array([
        f32::from_bits(0x0000_0000),
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x7f80_0000),
        f32::from_bits(0xff80_0000),
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
    ]);

    assert_eq!(
        values.is_sign_positive().to_array(),
        [true, false, true, false, true, false, true, false]
    );
    assert_eq!(
        values.is_sign_negative().to_array(),
        [false, true, false, true, false, true, false, true]
    );
}

#[test]
fn float_sign_predicates_preserve_f64_sign_bits_for_special_values() {
    let values = Simd::from_array([
        f64::from_bits(0x0000_0000_0000_0000),
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(0x7ff0_0000_0000_0000),
        f64::from_bits(0xfff0_0000_0000_0000),
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_4321_8765_cba9),
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x8000_0000_0000_0001),
    ]);

    assert_eq!(
        values.is_sign_positive().to_array(),
        [true, false, true, false, true, false, true, false]
    );
    assert_eq!(
        values.is_sign_negative().to_array(),
        [false, true, false, true, false, true, false, true]
    );
}

#[test]
fn float_abs_preserves_f32_magnitude_bits_for_special_values() {
    let values = Simd::from_array([
        f32::from_bits(0x0000_0000),
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x7f80_0000),
        f32::from_bits(0xff80_0000),
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
    ]);
    let expected = values
        .to_array()
        .map(|value| f32::from_bits(value.to_bits() & 0x7fff_ffff));
    assert_eq!(
        values.abs().to_bits().to_array(),
        expected.map(f32::to_bits)
    );
}

#[test]
fn float_abs_preserves_f64_magnitude_bits_for_special_values() {
    let values = Simd::from_array([
        f64::from_bits(0x0000_0000_0000_0000),
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(0x7ff0_0000_0000_0000),
        f64::from_bits(0xfff0_0000_0000_0000),
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_4321_8765_cba9),
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x8000_0000_0000_0001),
    ]);
    let expected = values
        .to_array()
        .map(|value| f64::from_bits(value.to_bits() & 0x7fff_ffff_ffff_ffff));
    assert_eq!(
        values.abs().to_bits().to_array(),
        expected.map(f64::to_bits)
    );
}

#[test]
fn float_copysign_preserves_f32_payload_and_uses_each_sign_bit() {
    let magnitude = Simd::from_array([
        f32::from_bits(0x0000_0000),
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x7f80_0000),
        f32::from_bits(0xff80_0000),
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
    ]);
    let signs = Simd::from_array([-1.0_f32, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    let expected = magnitude
        .to_array()
        .into_iter()
        .zip(signs.to_array())
        .map(|(value, sign)| (value.to_bits() & 0x7fff_ffff) | (sign.to_bits() & 0x8000_0000));
    assert_eq!(magnitude.copysign(signs).to_bits().to_array(), expected);
}

#[test]
fn float_copysign_preserves_f64_payload_and_uses_each_sign_bit() {
    let magnitude = Simd::from_array([
        f64::from_bits(0x0000_0000_0000_0000),
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(0x7ff0_0000_0000_0000),
        f64::from_bits(0xfff0_0000_0000_0000),
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_4321_8765_cba9),
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x8000_0000_0000_0001),
    ]);
    let signs = Simd::from_array([-1.0_f64, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    let expected = magnitude
        .to_array()
        .into_iter()
        .zip(signs.to_array())
        .map(|(value, sign)| {
            (value.to_bits() & 0x7fff_ffff_ffff_ffff) | (sign.to_bits() & 0x8000_0000_0000_0000)
        });
    assert_eq!(magnitude.copysign(signs).to_bits().to_array(), expected);
}


#[test]
fn float_signum_preserves_f32_nan_bits_and_matches_scalar_semantics() {
    let values = Simd::from_array([
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
        f32::NEG_INFINITY,
        -3.5,
        -0.0,
        0.0,
        2.5,
        f32::INFINITY,
    ]);
    let expected = values.to_array().map(f32::signum).map(f32::to_bits);

    assert_eq!(values.signum().to_bits().to_array(), expected);
}

#[test]
fn float_signum_preserves_f64_nan_bits_and_matches_scalar_semantics() {
    let values = Simd::from_array([
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_4321_8765_cba9),
        f64::NEG_INFINITY,
        -3.5,
        -0.0,
        0.0,
        2.5,
        f64::INFINITY,
    ]);
    let expected = values.to_array().map(f64::signum).map(f64::to_bits);

    assert_eq!(values.signum().to_bits().to_array(), expected);
}
