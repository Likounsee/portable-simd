#![feature(portable_simd)]

use core_simd::simd::{Simd, num::SimdFloat};

#[test]
fn f32_reduce_min_and_max_ignore_signaling_nan_lanes() {
    let values = [
        f32::from_bits(0x7f80_0001),
        -12.0,
        f32::from_bits(0xff80_0001),
        4.5,
        0.0,
        f32::INFINITY,
        -3.0,
        f32::from_bits(0x7f80_0001),
    ];
    let vector = Simd::from_array(values);
    let expected_min = values.into_iter().fold(f32::NAN, f32::min);
    let expected_max = values.into_iter().fold(f32::NAN, f32::max);

    assert_eq!(vector.reduce_min(), expected_min);
    assert_eq!(vector.reduce_max(), expected_max);
    assert_eq!(vector.reduce_min(), -12.0);
    assert_eq!(vector.reduce_max(), f32::INFINITY);
}

#[test]
fn f64_reduce_min_and_max_ignore_signaling_nan_lanes() {
    let values = [
        f64::from_bits(0x7ff0_0000_0000_0001),
        -12.0,
        f64::from_bits(0xfff0_0000_0000_0001),
        4.5,
        0.0,
        f64::INFINITY,
        -3.0,
        f64::from_bits(0x7ff0_0000_0000_0001),
    ];
    let vector = Simd::from_array(values);
    let expected_min = values.into_iter().fold(f64::NAN, f64::min);
    let expected_max = values.into_iter().fold(f64::NAN, f64::max);

    assert_eq!(vector.reduce_min(), expected_min);
    assert_eq!(vector.reduce_max(), expected_max);
    assert_eq!(vector.reduce_min(), -12.0);
    assert_eq!(vector.reduce_max(), f64::INFINITY);
}

#[test]
fn f32_reduce_min_and_max_return_nan_when_all_lanes_are_signaling_nan() {
    let values = Simd::from_array([
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xff80_0001),
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xff80_0001),
    ]);

    assert!(values.reduce_min().is_nan());
    assert!(values.reduce_max().is_nan());
}

#[test]
fn f64_reduce_min_and_max_return_nan_when_all_lanes_are_signaling_nan() {
    let values = Simd::from_array([
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::from_bits(0xfff0_0000_0000_0001),
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::from_bits(0xfff0_0000_0000_0001),
    ]);

    assert!(values.reduce_min().is_nan());
    assert!(values.reduce_max().is_nan());
}
