#![feature(portable_simd)]

use core_simd::simd::{num::SimdFloat, Simd};

#[test]
fn float_cast_f32_to_i32_matches_scalar_saturating_casts() {
    let values = [
        f32::NEG_INFINITY,
        -2147483904.0,
        -19.75,
        -0.0,
        0.0,
        19.75,
        2147483648.0,
        f32::INFINITY,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as i32);
    assert_eq!(vector.cast::<i32>().to_array(), expected);
}

#[test]
fn float_cast_f32_to_u32_matches_scalar_saturating_casts() {
    let values = [
        f32::NEG_INFINITY,
        -19.75,
        -0.0,
        0.0,
        19.75,
        4294967296.0,
        f32::INFINITY,
        f32::NAN,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as u32);
    assert_eq!(vector.cast::<u32>().to_array(), expected);
}

#[test]
fn float_cast_f64_to_i64_matches_scalar_saturating_casts() {
    let values = [
        f64::NEG_INFINITY,
        -9223372036854777856.0,
        -19.75,
        -0.0,
        0.0,
        19.75,
        9223372036854775808.0,
        f64::INFINITY,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as i64);
    assert_eq!(vector.cast::<i64>().to_array(), expected);
}

#[test]
fn float_cast_f64_to_u64_matches_scalar_saturating_casts() {
    let values = [
        f64::NEG_INFINITY,
        -19.75,
        -0.0,
        0.0,
        19.75,
        18446744073709551616.0,
        f64::INFINITY,
        f64::NAN,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as u64);
    assert_eq!(vector.cast::<u64>().to_array(), expected);
}

#[test]
fn float_reductions_f32_match_ordered_scalar_references() {
    let values = [1.5_f32, -2.0, 4.0, 0.5, -0.25, 3.0, 2.0, -1.0];
    let vector = Simd::from_array(values);
    let expected_sum = values.into_iter().fold(-0.0, |sum, value| sum + value);
    let expected_product = values.into_iter().fold(1.0, |product, value| product * value);

    assert_eq!(vector.reduce_sum().to_bits(), expected_sum.to_bits());
    assert_eq!(vector.reduce_product().to_bits(), expected_product.to_bits());
}

#[test]
fn float_reductions_f64_match_ordered_scalar_references() {
    let values = [1.5_f64, -2.0, 4.0, 0.5, -0.25, 3.0, 2.0, -1.0];
    let vector = Simd::from_array(values);
    let expected_sum = values.into_iter().fold(-0.0, |sum, value| sum + value);
    let expected_product = values.into_iter().fold(1.0, |product, value| product * value);

    assert_eq!(vector.reduce_sum().to_bits(), expected_sum.to_bits());
    assert_eq!(vector.reduce_product().to_bits(), expected_product.to_bits());
}

#[test]
fn float_min_max_reductions_f32_skip_single_nan_lanes() {
    let values = Simd::from_array([
        f32::NAN,
        -4.0,
        2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        7.0,
        1.0,
    ]);

    assert_eq!(values.reduce_min(), f32::NEG_INFINITY);
    assert_eq!(values.reduce_max(), f32::INFINITY);
    assert!(Simd::splat(f32::NAN).reduce_min().is_nan());
    assert!(Simd::splat(f32::NAN).reduce_max().is_nan());
}

#[test]
fn float_min_max_reductions_f64_skip_single_nan_lanes() {
    let values = Simd::from_array([
        f64::NAN,
        -4.0,
        2.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        7.0,
        1.0,
    ]);

    assert_eq!(values.reduce_min(), f64::NEG_INFINITY);
    assert_eq!(values.reduce_max(), f64::INFINITY);
    assert!(Simd::splat(f64::NAN).reduce_min().is_nan());
    assert!(Simd::splat(f64::NAN).reduce_max().is_nan());
}
