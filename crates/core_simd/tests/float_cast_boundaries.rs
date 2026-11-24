#![feature(portable_simd)]

use core_simd::simd::{Simd, prelude::*};

#[test]
fn f32_to_i32_cast_saturates_and_handles_nan() {
    let input = Simd::from_array([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.9]);
    let output: Simd<i32, 4> = input.cast();

    assert_eq!(output.to_array(), [0, i32::MAX, i32::MIN, 0]);
}

#[test]
fn f32_to_u32_cast_saturates_and_handles_nan() {
    let input = Simd::from_array([-1.0_f32, f32::INFINITY, f32::NAN, 42.9]);
    let output: Simd<u32, 4> = input.cast();

    assert_eq!(output.to_array(), [0, u32::MAX, 0, 42]);
}

#[test]
fn f64_to_i64_cast_saturates_at_exact_integer_bounds() {
    let input = Simd::from_array([
        i64::MIN as f64,
        i64::MAX as f64,
        -42.9,
        42.9,
    ]);
    let output: Simd<i64, 4> = input.cast();

    // i64::MAX as f64 rounds up to 2^63, so the saturating cast must clamp it.
    assert_eq!(output.to_array(), [i64::MIN, i64::MAX, -42, 42]);
}

#[test]
fn f64_to_u64_cast_saturates_at_unsigned_bounds() {
    let input = Simd::from_array([-1.0_f64, 0.0, 42.9, u64::MAX as f64]);
    let output: Simd<u64, 4> = input.cast();

    // u64::MAX as f64 rounds up to 2^64, so the saturating cast must clamp it.
    assert_eq!(output.to_array(), [0, 0, 42, u64::MAX]);
}
