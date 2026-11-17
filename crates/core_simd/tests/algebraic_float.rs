#![feature(portable_simd)]
#![feature(f16)]

use core_simd::simd::{Simd, num::SimdFloat};

fn assert_f32_close(actual: [f32; 4], expected: [f32; 4]) {
    for i in 0..4 {
        assert!((actual[i] - expected[i]).abs() <= 1e-6 * expected[i].abs().max(1.0));
    }
}

fn assert_f64_close(actual: [f64; 4], expected: [f64; 4]) {
    for i in 0..4 {
        assert!((actual[i] - expected[i]).abs() <= 1e-12 * expected[i].abs().max(1.0));
    }
}

fn assert_f16_close(actual: [f16; 4], expected: [f16; 4]) {
    for i in 0..4 {
        assert!((actual[i] - expected[i]).abs() <= 0.01_f16);
    }
}

fn assert_f32_lanewise(a: [f32; 4], b: [f32; 4]) {
    let va = Simd::from_array(a);
    let vb = Simd::from_array(b);
    assert_f32_close(
        va.algebraic_add(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_add(b[i])),
    );
    assert_f32_close(
        va.algebraic_sub(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_sub(b[i])),
    );
    assert_f32_close(
        va.algebraic_mul(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_mul(b[i])),
    );
    assert_f32_close(
        va.algebraic_div(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_div(b[i])),
    );
    assert_f32_close(
        va.algebraic_rem(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_rem(b[i])),
    );
}

fn assert_f64_lanewise(a: [f64; 4], b: [f64; 4]) {
    let va = Simd::from_array(a);
    let vb = Simd::from_array(b);
    assert_f64_close(
        va.algebraic_add(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_add(b[i])),
    );
    assert_f64_close(
        va.algebraic_sub(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_sub(b[i])),
    );
    assert_f64_close(
        va.algebraic_mul(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_mul(b[i])),
    );
    assert_f64_close(
        va.algebraic_div(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_div(b[i])),
    );
    assert_f64_close(
        va.algebraic_rem(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_rem(b[i])),
    );
}

fn assert_f16_lanewise(a: [f16; 4], b: [f16; 4]) {
    let va = Simd::from_array(a);
    let vb = Simd::from_array(b);
    assert_f16_close(
        va.algebraic_add(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_add(b[i])),
    );
    assert_f16_close(
        va.algebraic_sub(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_sub(b[i])),
    );
    assert_f16_close(
        va.algebraic_mul(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_mul(b[i])),
    );
    assert_f16_close(
        va.algebraic_div(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_div(b[i])),
    );
    assert_f16_close(
        va.algebraic_rem(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_rem(b[i])),
    );
}

#[test]
fn f16_algebraic_lanewise_operations_remain_close_to_scalar_results() {
    assert_f16_lanewise([1.25_f16, -8.0, 0.125, 64.0], [0.5_f16, 2.0, -0.25, 4.0]);
}

#[test]
fn f32_algebraic_lanewise_operations_match_scalar_contract() {
    assert_f32_lanewise([1.25, -8.0, 0.125, 64.0], [0.5, 2.0, -0.25, 4.0]);
}

#[test]
fn f64_algebraic_lanewise_operations_match_scalar_contract() {
    assert_f64_lanewise([1.25, -8.0, 0.125, 64.0], [0.5, 2.0, -0.25, 4.0]);
}

#[test]
fn f32_mul_add_relaxed_matches_a_valid_scalar_relaxed_result() {
    let x = [1.25_f32, -8.0, 0.125, 64.0];
    let a = [0.5_f32, 2.0, -0.25, 4.0];
    let b = [3.0_f32, -1.0, 2.0, 0.5];
    let actual = Simd::from_array(x)
        .mul_add_relaxed(Simd::from_array(a), Simd::from_array(b))
        .to_array();
    let fused: [f32; 4] = core::array::from_fn(|i| x[i].mul_add(a[i], b[i]));
    let separate: [f32; 4] = core::array::from_fn(|i| x[i] * a[i] + b[i]);
    for i in 0..4 {
        assert!(actual[i] == fused[i] || actual[i] == separate[i]);
    }
}

#[test]
fn f64_mul_add_relaxed_matches_a_valid_scalar_relaxed_result() {
    let x = [1.25_f64, -8.0, 0.125, 64.0];
    let a = [0.5_f64, 2.0, -0.25, 4.0];
    let b = [3.0_f64, -1.0, 2.0, 0.5];
    let actual = Simd::from_array(x)
        .mul_add_relaxed(Simd::from_array(a), Simd::from_array(b))
        .to_array();
    let fused: [f64; 4] = core::array::from_fn(|i| x[i].mul_add(a[i], b[i]));
    let separate: [f64; 4] = core::array::from_fn(|i| x[i] * a[i] + b[i]);
    for i in 0..4 {
        assert!(actual[i] == fused[i] || actual[i] == separate[i]);
    }
}

#[test]
fn f32_algebraic_reduce_sum_is_correct_for_exact_inputs() {
    assert_eq!(
        Simd::from_array([1.0_f32, 2.0, 4.0, 8.0]).algebraic_reduce_sum(),
        15.0
    );
}

#[test]
fn f64_algebraic_reduce_sum_is_correct_for_exact_inputs() {
    assert_eq!(
        Simd::from_array([1.0_f64, 2.0, 4.0, 8.0]).algebraic_reduce_sum(),
        15.0
    );
}

#[test]
fn f16_algebraic_reduce_sum_is_correct_for_exact_inputs() {
    assert_eq!(
        Simd::from_array([1.0_f16, 2.0, 4.0, 8.0]).algebraic_reduce_sum(),
        15.0_f16
    );
}

#[test]
fn f16_mul_add_relaxed_matches_a_valid_scalar_relaxed_result() {
    let x = [1.25_f16, -8.0, 0.125, 64.0];
    let a = [0.5_f16, 2.0, -0.25, 4.0];
    let b = [3.0_f16, -1.0, 2.0, 0.5];
    let actual = Simd::from_array(x)
        .mul_add_relaxed(Simd::from_array(a), Simd::from_array(b))
        .to_array();
    let fused: [f16; 4] = core::array::from_fn(|i| x[i].mul_add(a[i], b[i]));
    let separate: [f16; 4] = core::array::from_fn(|i| x[i] * a[i] + b[i]);
    for i in 0..4 {
        assert!(actual[i] == fused[i] || actual[i] == separate[i]);
    }
}

#[test]
fn f32_algebraic_reduce_sum_supports_eight_lanes() {
    assert_eq!(
        Simd::from_array([1.0_f32, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]).algebraic_reduce_sum(),
        255.0
    );
}

#[test]
fn f64_algebraic_reduce_sum_supports_two_lanes() {
    assert_eq!(Simd::from_array([1.0_f64, 2.0]).algebraic_reduce_sum(), 3.0);
}
