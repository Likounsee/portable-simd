#![feature(f16, portable_simd)]

use core_simd::simd::{Simd, num::SimdFloat};

#[test]
fn mul_sub_relaxed_f32() {
    let a = Simd::<f32, 4>::from_array([2.0, -3.0, 0.5, 10.0]);
    let b = Simd::<f32, 4>::from_array([4.0, 2.0, -8.0, 0.25]);
    let c = Simd::<f32, 4>::from_array([1.0, -5.0, 2.0, 3.0]);

    assert_eq!(a.mul_sub_relaxed(b, c).to_array(), [7.0, -1.0, -6.0, -0.5]);
}

#[test]
fn mul_sub_relaxed_f64() {
    let a = Simd::<f64, 2>::from_array([1.5, -2.0]);
    let b = Simd::<f64, 2>::from_array([4.0, 3.0]);
    let c = Simd::<f64, 2>::from_array([0.5, 7.0]);

    assert_eq!(a.mul_sub_relaxed(b, c).to_array(), [5.5, -13.0]);
}

#[test]
fn mul_sub_relaxed_f16() {
    let a = Simd::<f16, 4>::from_array([2.0, -3.0, 0.5, 10.0]);
    let b = Simd::<f16, 4>::from_array([4.0, 2.0, -8.0, 0.25]);
    let c = Simd::<f16, 4>::from_array([1.0, -5.0, 2.0, 3.0]);

    assert_eq!(
        a.mul_sub_relaxed(b, c).to_array(),
        [7.0_f16, -1.0, -6.0, -0.5]
    );
}
