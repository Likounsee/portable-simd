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

fn assert_f16_close(actual: [f16; 4], expected: [f16; 4], operation: &str) {
    for i in 0..4 {
        // Algebraic operations may choose a different rounding path. Compare in
        // representable f16 steps instead of using a scale-dependent epsilon.
        let ordered_bits = |value: f16| {
            let bits = value.to_bits();
            if bits & 0x8000 != 0 {
                !bits
            } else {
                bits | 0x8000
            }
        };
        let actual_bits = ordered_bits(actual[i]);
        let expected_bits = ordered_bits(expected[i]);
        let ulp_distance = actual_bits.abs_diff(expected_bits);
        assert!(
            ulp_distance <= 2,
            "{operation}, lane {i}: actual={:?} (0x{:04x}), expected={:?} (0x{:04x}), distance={ulp_distance} ULP",
            actual[i],
            actual[i].to_bits(),
            expected[i],
            expected[i].to_bits(),
        );
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
        "add",
    );
    assert_f16_close(
        va.algebraic_sub(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_sub(b[i])),
        "sub",
    );
    assert_f16_close(
        va.algebraic_mul(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_mul(b[i])),
        "mul",
    );
    assert_f16_close(
        va.algebraic_div(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_div(b[i])),
        "div",
    );
    assert_f16_close(
        va.algebraic_rem(vb).to_array(),
        core::array::from_fn(|i| a[i].algebraic_rem(b[i])),
        "rem",
    );
}

// Miri's current f16 interpreter produces results several representable
// values away from the scalar f16 operation, so keep this numerical conformance
// test on native execution where the target's f16 arithmetic is exercised.
#[cfg_attr(
    miri,
    ignore = "Miri f16 arithmetic emulation is not numerically reliable"
)]
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
    let actual = Simd::from_array([1.0_f32, 2.0, 4.0, 8.0]).algebraic_reduce_sum();
    assert!((actual - 15.0).abs() <= 1e-3);
}

#[test]
fn f64_algebraic_reduce_sum_is_correct_for_exact_inputs() {
    let actual = Simd::from_array([1.0_f64, 2.0, 4.0, 8.0]).algebraic_reduce_sum();
    assert!((actual - 15.0).abs() <= 1e-12);
}

#[test]
fn f16_algebraic_reduce_sum_is_correct_for_exact_inputs() {
    let actual = Simd::from_array([1.0_f16, 2.0, 4.0, 8.0]).algebraic_reduce_sum();
    assert!((actual - 15.0_f16).abs() <= 0.1_f16);
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
        assert!(
            (actual[i] - fused[i]).abs() <= 0.1_f16 || (actual[i] - separate[i]).abs() <= 0.1_f16,
            "lane {i}: actual={:?}, fused={:?}, separate={:?}",
            actual[i],
            fused[i],
            separate[i],
        );
    }
}

#[test]
fn f32_algebraic_reduce_sum_supports_eight_lanes() {
    let actual =
        Simd::from_array([1.0_f32, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]).algebraic_reduce_sum();
    assert!((actual - 255.0).abs() <= 1e-3);
}

#[test]
fn f64_algebraic_reduce_sum_supports_two_lanes() {
    let actual = Simd::from_array([1.0_f64, 2.0]).algebraic_reduce_sum();
    assert!((actual - 3.0).abs() <= 1e-12);
}
