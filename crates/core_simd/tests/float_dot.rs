#![feature(portable_simd)]

use core_simd::simd::{Simd, num::SimdFloat};

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

#[test]
fn dot3_ignores_fourth_lane_and_larger_vectors() {
    let a = Simd::<f32, 4>::from_array([1.0, 2.0, 3.0, 999.0]);
    let b = Simd::<f32, 4>::from_array([4.0, 5.0, 6.0, 888.0]);
    assert_eq!(a.dot3(b), 32.0);
    assert_eq!(a.dot4(b), 70.0);

    let a = Simd::<f32, 8>::from_array([1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0]);
    let b = Simd::<f32, 8>::from_array([4.0, 5.0, 6.0, 7.0, 999.0, 888.0, 777.0, 666.0]);
    assert_eq!(a.dot3(b), 32.0);
    assert_eq!(a.dot4(b), 60.0);
}

#[test]
fn dot3_and_dot4_support_f64() {
    let a = Simd::<f64, 4>::from_array([1.0, 2.0, 3.0, 4.0]);
    let b = Simd::<f64, 4>::from_array([4.0, 5.0, 6.0, 7.0]);
    assert_eq!(a.dot3(b), 32.0);
    assert_eq!(a.dot4(b), 60.0);
}
