#![feature(portable_simd)]

use core_simd::simd::{Simd, num::SimdFloat};

#[test]
fn float_classification_preserves_subnormal_and_zero_lanes_f32() {
    let values = Simd::from_array([
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -1.0,
    ]);

    assert_eq!(
        values.to_mask().to_array(),
        [true, true, false, false, true, true, true, true],
    );
    assert_eq!(
        values.is_subnormal().to_array(),
        [true, true, false, false, false, false, false, false],
    );
    assert_eq!(
        values.is_normal().to_array(),
        [false, false, false, false, false, false, false, true],
    );
    assert_eq!(
        values.is_finite().to_array(),
        [true, true, true, true, false, false, false, true],
    );
    assert_eq!(
        values.is_nan().to_array(),
        [false, false, false, false, false, false, true, false],
    );
}

#[test]
fn float_classification_preserves_subnormal_and_zero_lanes_f64() {
    let values = Simd::from_array([
        f64::from_bits(1),
        f64::from_bits(0x8000_0000_0000_0001),
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -1.0,
    ]);

    assert_eq!(
        values.to_mask().to_array(),
        [true, true, false, false, true, true, true, true],
    );
    assert_eq!(
        values.is_subnormal().to_array(),
        [true, true, false, false, false, false, false, false],
    );
    assert_eq!(
        values.is_normal().to_array(),
        [false, false, false, false, false, false, false, true],
    );
    assert_eq!(
        values.is_finite().to_array(),
        [true, true, true, true, false, false, false, true],
    );
    assert_eq!(
        values.is_nan().to_array(),
        [false, false, false, false, false, false, true, false],
    );
}

#[test]
fn float_sign_operations_preserve_f32_nan_payloads_and_signed_zero() {
    let values = Simd::from_array([
        f32::from_bits(0x0000_0000),
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
    ]);
    let signs = Simd::from_array([-1.0_f32, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    let bits = values.to_bits().to_array();
    let sign_bits = signs.to_bits().to_array();

    assert_eq!(
        values.is_sign_negative().to_array(),
        [false, true, false, true, false, true, false, true],
    );
    assert_eq!(
        values.abs().to_bits().to_array(),
        bits.map(|value| value & 0x7fff_ffff),
    );
    assert_eq!(
        values.copysign(signs).to_bits().to_array(),
        core::array::from_fn(|lane| {
            (bits[lane] & 0x7fff_ffff) | (sign_bits[lane] & 0x8000_0000)
        }),
    );
    assert_eq!(
        values.signum().to_bits().to_array(),
        values.to_array().map(|value| {
            if value.is_nan() {
                value.to_bits()
            } else {
                value.signum().to_bits()
            }
        }),
    );
}

#[test]
fn float_sign_operations_preserve_f64_nan_payloads_and_signed_zero() {
    let values = Simd::from_array([
        f64::from_bits(0x0000_0000_0000_0000),
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_4321_8765_cba9),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(1),
        f64::from_bits(0x8000_0000_0000_0001),
    ]);
    let signs = Simd::from_array([-1.0_f64, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    let bits = values.to_bits().to_array();
    let sign_bits = signs.to_bits().to_array();

    assert_eq!(
        values.is_sign_negative().to_array(),
        [false, true, false, true, false, true, false, true],
    );
    assert_eq!(
        values.abs().to_bits().to_array(),
        bits.map(|value| value & 0x7fff_ffff_ffff_ffff),
    );
    assert_eq!(
        values.copysign(signs).to_bits().to_array(),
        core::array::from_fn(|lane| {
            (bits[lane] & 0x7fff_ffff_ffff_ffff) | (sign_bits[lane] & 0x8000_0000_0000_0000)
        }),
    );
    assert_eq!(
        values.signum().to_bits().to_array(),
        values.to_array().map(|value| {
            if value.is_nan() {
                value.to_bits()
            } else {
                value.signum().to_bits()
            }
        }),
    );
}
