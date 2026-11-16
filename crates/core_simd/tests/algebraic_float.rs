#![feature(portable_simd)]

use core_simd::simd::{Simd, num::SimdFloat};

#[test]
fn f32_algebraic_lanewise_operations_match_scalar_contract() {
    let a = [1.25_f32, -8.0, 0.125, 64.0];
    let b = [0.5_f32, 2.0, -0.25, 4.0];
    let va = Simd::from_array(a);
    let vb = Simd::from_array(b);

    assert_eq!(va.algebraic_add(vb).to_array(), core::array::from_fn(|i| a[i].algebraic_add(b[i])));
    assert_eq!(va.algebraic_sub(vb).to_array(), core::array::from_fn(|i| a[i].algebraic_sub(b[i])));
    assert_eq!(va.algebraic_mul(vb).to_array(), core::array::from_fn(|i| a[i].algebraic_mul(b[i])));
    assert_eq!(va.algebraic_div(vb).to_array(), core::array::from_fn(|i| a[i].algebraic_div(b[i])));
    assert_eq!(va.algebraic_rem(vb).to_array(), core::array::from_fn(|i| a[i].algebraic_rem(b[i])));
}

#[test]
fn f64_algebraic_lanewise_operations_match_scalar_contract() {
    let a = [1.25_f64, -8.0, 0.125, 64.0];
    let b = [0.5_f64, 2.0, -0.25, 4.0];
    let va = Simd::from_array(a);
    let vb = Simd::from_array(b);

    assert_eq!(va.algebraic_add(vb).to_array(), core::array::from_fn(|i| a[i].algebraic_add(b[i])));
    assert_eq!(va.algebraic_sub(vb).to_array(), core::array::from_fn(|i| a[i].algebraic_sub(b[i])));
    assert_eq!(va.algebraic_mul(vb).to_array(), core::array::from_fn(|i| a[i].algebraic_mul(b[i])));
    assert_eq!(va.algebraic_div(vb).to_array(), core::array::from_fn(|i| a[i].algebraic_div(b[i])));
    assert_eq!(va.algebraic_rem(vb).to_array(), core::array::from_fn(|i| a[i].algebraic_rem(b[i])));
}

#[test]
fn f32_algebraic_reduce_sum_matches_scalar_algebraic_fold() {
    let values = [1.0_f32, 1.0e20, -1.0e20, 3.0];
    let expected = values.into_iter().fold(0.0_f32, |sum, lane| sum.algebraic_add(lane));
    assert_eq!(Simd::from_array(values).algebraic_reduce_sum().to_bits(), expected.to_bits());
}

#[test]
fn f64_algebraic_reduce_sum_matches_scalar_algebraic_fold() {
    let values = [1.0_f64, 1.0e300, -1.0e300, 3.0];
    let expected = values.into_iter().fold(0.0_f64, |sum, lane| sum.algebraic_add(lane));
    assert_eq!(Simd::from_array(values).algebraic_reduce_sum().to_bits(), expected.to_bits());
}
