#![feature(portable_simd)]
#![cfg_attr(target_has_reliable_f16_math, feature(f16))]
#![expect(internal_features)]
#![feature(cfg_target_has_reliable_f16_f128)]
#![allow(unused_features)]

use core_simd::simd::{Simd, num::SimdFloat};

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
    let expected_product = values
        .into_iter()
        .fold(1.0, |product, value| product * value);

    assert_eq!(vector.reduce_sum().to_bits(), expected_sum.to_bits());
    assert_eq!(
        vector.reduce_product().to_bits(),
        expected_product.to_bits()
    );
}

#[test]
fn float_reductions_f64_match_ordered_scalar_references() {
    let values = [1.5_f64, -2.0, 4.0, 0.5, -0.25, 3.0, 2.0, -1.0];
    let vector = Simd::from_array(values);
    let expected_sum = values.into_iter().fold(-0.0, |sum, value| sum + value);
    let expected_product = values
        .into_iter()
        .fold(1.0, |product, value| product * value);

    assert_eq!(vector.reduce_sum().to_bits(), expected_sum.to_bits());
    assert_eq!(
        vector.reduce_product().to_bits(),
        expected_product.to_bits()
    );
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
    assert!(Simd::<f32, 8>::splat(f32::NAN).reduce_min().is_nan());
    assert!(Simd::<f32, 8>::splat(f32::NAN).reduce_max().is_nan());
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
    assert!(Simd::<f64, 8>::splat(f64::NAN).reduce_min().is_nan());
    assert!(Simd::<f64, 8>::splat(f64::NAN).reduce_max().is_nan());
}

#[test]
fn float_cast_f32_to_i8_matches_scalar_saturating_casts() {
    let values = [
        f32::NEG_INFINITY,
        -129.0,
        -128.0,
        -1.9,
        0.0,
        1.9,
        127.0,
        f32::INFINITY,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as i8);
    assert_eq!(vector.cast::<i8>().to_array(), expected);
}

#[test]
fn float_cast_f32_to_u8_matches_scalar_saturating_casts() {
    let values = [
        f32::NEG_INFINITY,
        -1.9,
        -0.0,
        0.0,
        1.9,
        254.9,
        255.0,
        f32::NAN,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as u8);
    assert_eq!(vector.cast::<u8>().to_array(), expected);
}

#[test]
fn float_cast_f64_to_i16_matches_scalar_saturating_casts() {
    let values = [
        f64::NEG_INFINITY,
        -32769.0,
        -32768.0,
        -1.9,
        0.0,
        1.9,
        32767.0,
        f64::INFINITY,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as i16);
    assert_eq!(vector.cast::<i16>().to_array(), expected);
}

#[test]
fn float_cast_f64_to_u16_matches_scalar_saturating_casts() {
    let values = [
        f64::NEG_INFINITY,
        -1.9,
        -0.0,
        0.0,
        1.9,
        65534.9,
        65535.0,
        f64::NAN,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as u16);
    assert_eq!(vector.cast::<u16>().to_array(), expected);
}

#[cfg(all(
    target_has_reliable_f16_math,
    not(all(target_arch = "wasm32", target_feature = "simd128"))
))]
#[test]
fn float_cast_f16_to_i8_matches_scalar_saturating_casts() {
    let values = [
        f16::NEG_INFINITY,
        -129.0_f16,
        -128.0,
        -1.5,
        0.0,
        1.5,
        127.0,
        f16::INFINITY,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as i8);
    assert_eq!(vector.cast::<i8>().to_array(), expected);
}

#[cfg(all(
    target_has_reliable_f16_math,
    not(all(target_arch = "wasm32", target_feature = "simd128")),
))]
#[test]
fn float_cast_f16_to_u8_matches_scalar_saturating_casts() {
    let values = [
        f16::NEG_INFINITY,
        -1.5_f16,
        -0.0,
        0.0,
        1.5,
        254.0,
        255.0,
        f16::NAN,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as u8);
    assert_eq!(vector.cast::<u8>().to_array(), expected);
}

#[cfg(all(
    target_has_reliable_f16_math,
    not(all(target_arch = "wasm32", target_feature = "simd128")),
))]
#[test]
fn float_cast_f16_to_i16_matches_scalar_saturating_casts() {
    let values = [
        f16::NEG_INFINITY,
        -32768.0_f16,
        -1.5,
        -0.0,
        0.0,
        1.5,
        32752.0,
        f16::INFINITY,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as i16);
    assert_eq!(vector.cast::<i16>().to_array(), expected);
}

#[cfg(all(
    target_has_reliable_f16_math,
    not(all(target_arch = "wasm32", target_feature = "simd128")),
))]
#[test]
fn float_cast_f16_to_u16_matches_scalar_saturating_casts() {
    let values = [
        f16::NEG_INFINITY,
        -1.5_f16,
        -0.0,
        0.0,
        1.5,
        32752.0,
        65504.0,
        f16::NAN,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as u16);
    assert_eq!(vector.cast::<u16>().to_array(), expected);
}

#[test]
fn float_reductions_f32_preserve_order_for_cancellation_and_infinity() {
    let cancellation = [1.0e20_f32, -1.0e20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let vector = Simd::from_array(cancellation);
    let expected = cancellation
        .into_iter()
        .fold(0.0_f32, |sum, value| sum + value);
    assert_eq!(vector.reduce_sum().to_bits(), expected.to_bits());

    let exceptional = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    let vector = Simd::from_array(exceptional);
    let expected = exceptional
        .into_iter()
        .fold(0.0_f32, |sum, value| sum + value);
    assert_eq!(vector.reduce_sum().is_nan(), expected.is_nan());
}

#[test]
fn float_reductions_f64_preserve_order_for_cancellation_and_infinity() {
    let cancellation = [1.0e300_f64, -1.0e300, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let vector = Simd::from_array(cancellation);
    let expected = cancellation
        .into_iter()
        .fold(0.0_f64, |sum, value| sum + value);
    assert_eq!(vector.reduce_sum().to_bits(), expected.to_bits());

    let exceptional = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    let vector = Simd::from_array(exceptional);
    let expected = exceptional
        .into_iter()
        .fold(0.0_f64, |sum, value| sum + value);
    assert_eq!(vector.reduce_sum().is_nan(), expected.is_nan());
}

#[test]
fn float_to_int_unchecked_f32_to_i32_matches_scalar_truncation() {
    let values = [
        -2147483520.0_f32,
        -19.75,
        -0.5,
        0.0,
        19.75,
        2147483520.0,
        1.0,
        -1.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable i32.
    let actual = unsafe { vector.to_int_unchecked::<i32>() };
    assert_eq!(actual.to_array(), values.map(|value| value as i32));
}

#[test]
fn float_to_int_unchecked_f32_to_u32_matches_scalar_truncation() {
    let values = [
        -0.5_f32,
        0.0,
        1.0,
        19.75,
        255.5,
        65535.0,
        2147483648.0,
        4294967040.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable u32.
    let actual = unsafe { vector.to_int_unchecked::<u32>() };
    assert_eq!(actual.to_array(), values.map(|value| value as u32));
}

#[test]
fn float_to_int_unchecked_f64_to_i64_matches_scalar_truncation() {
    let values = [
        -9223372036854774784.0_f64,
        -19.75,
        -0.5,
        0.0,
        19.75,
        9223372036854774784.0,
        1.0,
        -1.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable i64.
    let actual = unsafe { vector.to_int_unchecked::<i64>() };
    assert_eq!(actual.to_array(), values.map(|value| value as i64));
}

#[test]
fn float_to_int_unchecked_f64_to_u64_matches_scalar_truncation() {
    let values = [
        -0.5_f64,
        0.0,
        1.0,
        19.75,
        255.5,
        65535.0,
        9223372036854774784.0,
        18446744073709549568.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable u64.
    let actual = unsafe { vector.to_int_unchecked::<u64>() };
    assert_eq!(actual.to_array(), values.map(|value| value as u64));
}

#[cfg(all(
    target_has_reliable_f16_math,
    not(all(target_arch = "wasm32", target_feature = "simd128")),
))]
#[test]
fn float_cast_f16_to_i32_matches_scalar_saturating_casts() {
    let values = [
        f16::NEG_INFINITY,
        -65504.0_f16,
        -19.5,
        -0.0,
        0.0,
        19.5,
        65504.0,
        f16::INFINITY,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as i32);
    assert_eq!(vector.cast::<i32>().to_array(), expected);
}

#[cfg(all(
    target_has_reliable_f16_math,
    not(all(target_arch = "wasm32", target_feature = "simd128")),
))]
#[test]
fn float_cast_f16_to_u32_matches_scalar_saturating_casts() {
    let values = [
        f16::NEG_INFINITY,
        -1.5_f16,
        -0.0,
        0.0,
        19.5,
        65504.0,
        f16::INFINITY,
        f16::NAN,
    ];
    let vector = Simd::from_array(values);
    let expected = values.map(|value| value as u32);
    assert_eq!(vector.cast::<u32>().to_array(), expected);
}

#[test]
fn float_cast_f32_to_i64_matches_scalar_saturating_casts() {
    let values = [
        f32::NEG_INFINITY,
        -1.0e18_f32,
        -19.75,
        -0.0,
        0.0,
        19.75,
        1.0e18,
        f32::INFINITY,
    ];
    let vector = Simd::from_array(values);
    assert_eq!(
        vector.cast::<i64>().to_array(),
        values.map(|value| value as i64)
    );
}

#[test]
fn float_cast_f32_to_u64_matches_scalar_saturating_casts() {
    let values = [
        f32::NEG_INFINITY,
        -19.75_f32,
        -0.0,
        0.0,
        19.75,
        1.0e18,
        f32::INFINITY,
        f32::NAN,
    ];
    let vector = Simd::from_array(values);
    assert_eq!(
        vector.cast::<u64>().to_array(),
        values.map(|value| value as u64)
    );
}

#[test]
fn float_cast_f64_to_i8_matches_scalar_saturating_casts() {
    let values = [
        f64::NEG_INFINITY,
        -129.0,
        -128.0,
        -1.9,
        0.0,
        1.9,
        127.0,
        f64::INFINITY,
    ];
    let vector = Simd::from_array(values);
    assert_eq!(
        vector.cast::<i8>().to_array(),
        values.map(|value| value as i8)
    );
}

#[test]
fn float_cast_f64_to_u8_matches_scalar_saturating_casts() {
    let values = [
        f64::NEG_INFINITY,
        -1.9,
        -0.0,
        0.0,
        1.9,
        254.9,
        255.0,
        f64::NAN,
    ];
    let vector = Simd::from_array(values);
    assert_eq!(
        vector.cast::<u8>().to_array(),
        values.map(|value| value as u8)
    );
}

#[cfg(all(
    target_has_reliable_f16_math,
    not(all(target_arch = "wasm32", target_feature = "simd128")),
))]
#[test]
fn float_cast_f16_to_i64_matches_scalar_saturating_casts() {
    let values = [
        f16::NEG_INFINITY,
        -65504.0_f16,
        -19.5,
        -0.0,
        0.0,
        19.5,
        65504.0,
        f16::INFINITY,
    ];
    let vector = Simd::from_array(values);
    assert_eq!(
        vector.cast::<i64>().to_array(),
        values.map(|value| value as i64)
    );
}

#[cfg(all(
    target_has_reliable_f16_math,
    not(all(target_arch = "wasm32", target_feature = "simd128")),
))]
#[test]
fn float_cast_f16_to_u64_matches_scalar_saturating_casts() {
    let values = [
        f16::NEG_INFINITY,
        -1.5_f16,
        -0.0,
        0.0,
        19.5,
        65504.0,
        f16::INFINITY,
        f16::NAN,
    ];
    let vector = Simd::from_array(values);
    assert_eq!(
        vector.cast::<u64>().to_array(),
        values.map(|value| value as u64)
    );
}

#[test]
fn float_cast_f32_to_i16_matches_scalar_saturating_casts() {
    let values = [
        f32::NEG_INFINITY,
        -32769.0,
        -32768.0,
        -1.9,
        0.0,
        1.9,
        32767.0,
        f32::INFINITY,
    ];
    let vector = Simd::from_array(values);
    assert_eq!(
        vector.cast::<i16>().to_array(),
        values.map(|value| value as i16)
    );
}

#[test]
fn float_cast_f32_to_u16_matches_scalar_saturating_casts() {
    let values = [
        f32::NEG_INFINITY,
        -1.9,
        -0.0,
        0.0,
        1.9,
        65534.9,
        65535.0,
        f32::NAN,
    ];
    let vector = Simd::from_array(values);
    assert_eq!(
        vector.cast::<u16>().to_array(),
        values.map(|value| value as u16)
    );
}

#[test]
fn float_cast_f64_to_i32_matches_scalar_saturating_casts() {
    let values = [
        f64::NEG_INFINITY,
        -2147483649.0,
        -2147483648.0,
        -19.75,
        0.0,
        19.75,
        2147483647.0,
        f64::INFINITY,
    ];
    let vector = Simd::from_array(values);
    assert_eq!(
        vector.cast::<i32>().to_array(),
        values.map(|value| value as i32)
    );
}

#[test]
fn float_cast_f64_to_u32_matches_scalar_saturating_casts() {
    let values = [
        f64::NEG_INFINITY,
        -1.9,
        -0.0,
        0.0,
        1.9,
        4294967294.0,
        4294967295.0,
        f64::NAN,
    ];
    let vector = Simd::from_array(values);
    assert_eq!(
        vector.cast::<u32>().to_array(),
        values.map(|value| value as u32)
    );
}

#[test]
fn float_to_int_unchecked_f32_to_i8_matches_scalar_truncation() {
    let values = [
        -127.9_f32, -100.5, -1.9, 0.0, 1.9, 42.7, 126.9, 127.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable i8.
    let actual = unsafe { vector.to_int_unchecked::<i8>() };
    assert_eq!(actual.to_array(), values.map(|value| value as i8));
}

#[test]
fn float_to_int_unchecked_f32_to_u8_matches_scalar_truncation() {
    let values = [0.0_f32, 0.9, 1.9, 42.7, 127.5, 200.9, 254.9, 255.0];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable u8.
    let actual = unsafe { vector.to_int_unchecked::<u8>() };
    assert_eq!(actual.to_array(), values.map(|value| value as u8));
}

#[test]
fn float_to_int_unchecked_f32_to_i16_matches_scalar_truncation() {
    let values = [
        -32767.9_f32,
        -1000.5,
        -1.9,
        0.0,
        1.9,
        1000.7,
        32766.9,
        32767.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable i16.
    let actual = unsafe { vector.to_int_unchecked::<i16>() };
    assert_eq!(actual.to_array(), values.map(|value| value as i16));
}

#[test]
fn float_to_int_unchecked_f32_to_u16_matches_scalar_truncation() {
    let values = [
        0.0_f32, 0.9, 1.9, 255.7, 1024.5, 32767.9, 65534.9, 65535.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable u16.
    let actual = unsafe { vector.to_int_unchecked::<u16>() };
    assert_eq!(actual.to_array(), values.map(|value| value as u16));
}

#[test]
fn float_to_int_unchecked_f64_to_i8_matches_scalar_truncation() {
    let values = [
        -127.9_f64, -100.5, -1.9, 0.0, 1.9, 42.7, 126.9, 127.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable i8.
    let actual = unsafe { vector.to_int_unchecked::<i8>() };
    assert_eq!(actual.to_array(), values.map(|value| value as i8));
}

#[test]
fn float_to_int_unchecked_f64_to_u8_matches_scalar_truncation() {
    let values = [0.0_f64, 0.9, 1.9, 42.7, 127.5, 200.9, 254.9, 255.0];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable u8.
    let actual = unsafe { vector.to_int_unchecked::<u8>() };
    assert_eq!(actual.to_array(), values.map(|value| value as u8));
}

#[test]
fn float_to_int_unchecked_f64_to_i16_matches_scalar_truncation() {
    let values = [
        -32767.9_f64,
        -1000.5,
        -1.9,
        0.0,
        1.9,
        1000.7,
        32766.9,
        32767.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable i16.
    let actual = unsafe { vector.to_int_unchecked::<i16>() };
    assert_eq!(actual.to_array(), values.map(|value| value as i16));
}

#[test]
fn float_to_int_unchecked_f64_to_u16_matches_scalar_truncation() {
    let values = [
        0.0_f64, 0.9, 1.9, 255.7, 1024.5, 32767.9, 65534.9, 65535.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable u16.
    let actual = unsafe { vector.to_int_unchecked::<u16>() };
    assert_eq!(actual.to_array(), values.map(|value| value as u16));
}

#[test]
fn float_to_int_unchecked_f64_to_i32_matches_scalar_truncation() {
    let values = [
        -2147483647.9_f64,
        -100000.5,
        -1.9,
        0.0,
        1.9,
        100000.7,
        2147483646.9,
        2147483647.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable i32.
    let actual = unsafe { vector.to_int_unchecked::<i32>() };
    assert_eq!(actual.to_array(), values.map(|value| value as i32));
}

#[test]
fn float_to_int_unchecked_f64_to_u32_matches_scalar_truncation() {
    let values = [
        0.0_f64,
        0.9,
        1.9,
        100000.7,
        2147483647.9,
        3000000000.9,
        4294967294.9,
        4294967295.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable u32.
    let actual = unsafe { vector.to_int_unchecked::<u32>() };
    assert_eq!(actual.to_array(), values.map(|value| value as u32));
}

#[cfg(target_has_reliable_f16_math)]
#[test]
fn float_to_int_unchecked_f16_to_i32_matches_scalar_truncation() {
    let values = [
        -65504.0_f16,
        -1000.5,
        -1.9,
        0.0,
        1.9,
        1000.7,
        32767.0,
        65504.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable i32.
    let actual = unsafe { vector.to_int_unchecked::<i32>() };
    assert_eq!(actual.to_array(), values.map(|value| value as i32));
}

#[cfg(target_has_reliable_f16_math)]
#[test]
fn float_to_int_unchecked_f16_to_u32_matches_scalar_truncation() {
    let values = [0.0_f16, 0.9, 1.9, 255.7, 1024.5, 32767.0, 60000.0, 65504.0];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable u32.
    let actual = unsafe { vector.to_int_unchecked::<u32>() };
    assert_eq!(actual.to_array(), values.map(|value| value as u32));
}

#[cfg(target_has_reliable_f16_math)]
#[test]
fn float_to_int_unchecked_f16_to_i64_matches_scalar_truncation() {
    let values = [
        -65504.0_f16,
        -1000.5,
        -1.9,
        0.0,
        1.9,
        1000.7,
        32767.0,
        65504.0,
    ];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable i64.
    let actual = unsafe { vector.to_int_unchecked::<i64>() };
    assert_eq!(actual.to_array(), values.map(|value| value as i64));
}

#[cfg(target_has_reliable_f16_math)]
#[test]
fn float_to_int_unchecked_f16_to_u64_matches_scalar_truncation() {
    let values = [0.0_f16, 0.9, 1.9, 255.7, 1024.5, 32767.0, 60000.0, 65504.0];
    let vector = Simd::from_array(values);
    // SAFETY: every value is finite and truncates to a representable u64.
    let actual = unsafe { vector.to_int_unchecked::<u64>() };
    assert_eq!(actual.to_array(), values.map(|value| value as u64));
}
