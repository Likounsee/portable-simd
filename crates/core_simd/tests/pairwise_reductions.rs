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

#[test]
fn pairwise_reductions_odd_lane_counts() {
    let three = Simd::<f32, 3>::from_array([1.0, 2.0, 3.0]);
    assert_eq!(three.pairwise_reduce_sum(), 6.0);
    assert_eq!(three.pairwise_reduce_product(), 6.0);

    let five = Simd::<f32, 5>::from_array([1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(five.pairwise_reduce_sum(), 15.0);
    assert_eq!(five.pairwise_reduce_product(), 120.0);
}

#[test]
fn pairwise_sum_uses_the_documented_tree_order() {
    // Pairwise grouping rounds the two pairs to 1e20 and -1e20, then sums to zero.
    // A sequential left fold instead produces 1.0.
    let values = Simd::<f32, 4>::from_array([1.0e20, 1.0, -1.0e20, 1.0]);
    assert_eq!(values.pairwise_reduce_sum(), 0.0);
    assert_eq!(values.reduce_sum(), 1.0);
}
