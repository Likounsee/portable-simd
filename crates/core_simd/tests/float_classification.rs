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
