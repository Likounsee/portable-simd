#![feature(portable_simd)]

use core_simd::simd::prelude::*;
use std_float::StdFloat;

#[test]
fn mul_sub_f32() {
    let a = f32x4::from_array([2.0, 3.0, 4.0, 5.0]);
    let b = f32x4::splat(10.0);
    let c = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);

    assert_eq!(a.mul_sub(b, c), f32x4::from_array([19.0, 28.0, 37.0, 46.0]));
}

#[test]
fn mul_sub_f64() {
    let a = f64x4::from_array([2.0, 3.0, 4.0, 5.0]);
    let b = f64x4::splat(10.0);
    let c = f64x4::from_array([1.0, 2.0, 3.0, 4.0]);

    assert_eq!(a.mul_sub(b, c), f64x4::from_array([19.0, 28.0, 37.0, 46.0]));
}

#[test]
fn mul_sub_preserves_fused_precision() {
    // The exact product is just below 1.0. A separately rounded multiplication
    // can round it to 1.0 before subtraction, losing the small negative result.
    let epsilon = 2.0_f64.powi(-27);
    let a = f64x2::splat(1.0 + epsilon);
    let b = f64x2::splat(1.0 - epsilon);
    let one = f64x2::splat(1.0);

    let fused = a.mul_sub(b, one);
    let scalar_fused = (1.0 + epsilon).mul_add(1.0 - epsilon, -1.0);
    assert_eq!(fused, f64x2::splat(scalar_fused));
    assert!(fused[0] < 0.0);
}
