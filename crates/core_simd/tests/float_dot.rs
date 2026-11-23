#![feature(portable_simd)]

use core_simd::simd::{num::SimdFloat, Simd};

#[test]
fn dot_f32_four_lanes() {
    let a = Simd::<f32, 4>::from_array([1.0, 2.0, 3.0, 4.0]);
    let b = Simd::<f32, 4>::from_array([5.0, 6.0, 7.0, 8.0]);
    assert_eq!(a.dot(b), 70.0);
}

#[test]
fn dot_f32_eight_lanes_with_negative_values() {
    let a = Simd::<f32, 8>::from_array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
    let b = Simd::<f32, 8>::from_array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(a.dot(b), -40.0);
}

#[test]
fn dot_f64_two_lanes() {
    let a = Simd::<f64, 2>::from_array([-2.0, 3.0]);
    let b = Simd::<f64, 2>::from_array([4.0, 5.0]);
    assert_eq!(a.dot(b), 7.0);
}

#[test]
fn dot_f64_four_lanes() {
    let a = Simd::<f64, 4>::from_array([0.5, 1.5, -2.0, 4.0]);
    let b = Simd::<f64, 4>::from_array([2.0, 2.0, 3.0, 0.5]);
    assert_eq!(a.dot(b), 0.0);
}
