#![feature(f16, portable_simd)]

use core_simd::simd::{Simd, num::SimdFloat};

#[test]
fn pairwise_sum_f32() {
    let values = Simd::<f32, 4>::from_array([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(values.pairwise_reduce_sum(), 10.0);
}

#[test]
fn pairwise_product_f32() {
    let values = Simd::<f32, 4>::from_array([2.0, 3.0, 4.0, 5.0]);
    assert_eq!(values.pairwise_reduce_product(), 120.0);
}

#[test]
fn pairwise_reductions_f64() {
    let values = Simd::<f64, 8>::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(values.pairwise_reduce_sum(), 36.0);
    assert_eq!(values.pairwise_reduce_product(), 40_320.0);
}

#[test]
fn pairwise_reductions_f16() {
    let values = Simd::<f16, 4>::from_array([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(values.pairwise_reduce_sum(), 10.0_f16);
    assert_eq!(values.pairwise_reduce_product(), 24.0_f16);
}

#[test]
fn pairwise_reductions_single_lane() {
    let values = Simd::<f32, 1>::from_array([7.0]);
    assert_eq!(values.pairwise_reduce_sum(), 7.0);
    assert_eq!(values.pairwise_reduce_product(), 7.0);
}
