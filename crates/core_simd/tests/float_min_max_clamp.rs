#![feature(portable_simd)]

use core_simd::simd::{Simd, num::SimdFloat};

#[test]
fn simd_min_matches_scalar_nan_handling_for_f32() {
    let left_vector = Simd::from_array([
        f32::from_bits(0x7fc1_2345),
        -5.0,
        3.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.5,
        -2.0,
        8.0,
    ]);
    let right_vector = Simd::from_array([1.0, f32::NAN, 4.0, 7.0, -7.0, -0.5, 2.0, 9.0]);
    let left = left_vector.to_array();
    let right = right_vector.to_array();
    let expected = core::array::from_fn(|i| left[i].min(right[i]).to_bits());

    assert_eq!(
        left_vector.simd_min(right_vector).to_bits().to_array(),
        expected
    );
}

#[test]
fn simd_max_matches_scalar_nan_handling_for_f32() {
    let left_vector = Simd::from_array([
        f32::from_bits(0x7fc1_2345),
        -5.0,
        3.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.5,
        -2.0,
        8.0,
    ]);
    let right_vector = Simd::from_array([1.0, f32::NAN, 4.0, 7.0, -7.0, -0.5, 2.0, 9.0]);
    let left = left_vector.to_array();
    let right = right_vector.to_array();
    let expected = core::array::from_fn(|i| left[i].max(right[i]).to_bits());

    assert_eq!(
        left_vector.simd_max(right_vector).to_bits().to_array(),
        expected
    );
}

#[test]
fn simd_min_matches_scalar_nan_handling_for_f64() {
    let left_vector = Simd::from_array([
        f64::from_bits(0x7ff8_1234_5678_9abc),
        -5.0,
        3.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.5,
        -2.0,
        8.0,
    ]);
    let right_vector = Simd::from_array([1.0, f64::NAN, 4.0, 7.0, -7.0, -0.5, 2.0, 9.0]);
    let left = left_vector.to_array();
    let right = right_vector.to_array();
    let expected = core::array::from_fn(|i| left[i].min(right[i]).to_bits());

    assert_eq!(
        left_vector.simd_min(right_vector).to_bits().to_array(),
        expected
    );
}

#[test]
fn simd_max_matches_scalar_nan_handling_for_f64() {
    let left_vector = Simd::from_array([
        f64::from_bits(0x7ff8_1234_5678_9abc),
        -5.0,
        3.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.5,
        -2.0,
        8.0,
    ]);
    let right_vector = Simd::from_array([1.0, f64::NAN, 4.0, 7.0, -7.0, -0.5, 2.0, 9.0]);
    let left = left_vector.to_array();
    let right = right_vector.to_array();
    let expected = core::array::from_fn(|i| left[i].max(right[i]).to_bits());

    assert_eq!(
        left_vector.simd_max(right_vector).to_bits().to_array(),
        expected
    );
}

#[test]
fn simd_clamp_matches_scalar_behavior_for_f32() {
    let values_vector = Simd::from_array([
        f32::from_bits(0x7fc1_2345),
        f32::NEG_INFINITY,
        -3.0,
        -1.0,
        0.0,
        1.0,
        3.0,
        f32::INFINITY,
    ]);
    let min = Simd::from_array([-2.0, -2.0, -2.0, -1.0, -1.0, 0.0, 2.0, 2.0]);
    let max = Simd::from_array([2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let values = values_vector.to_array();
    let min = min.to_array();
    let max = max.to_array();
    let expected = core::array::from_fn(|i| values[i].clamp(min[i], max[i]).to_bits());

    assert_eq!(
        values_vector
            .simd_clamp(Simd::from_array(min), Simd::from_array(max))
            .to_bits()
            .to_array(),
        expected
    );
}

#[test]
fn simd_clamp_matches_scalar_behavior_for_f64() {
    let values_vector = Simd::from_array([
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::NEG_INFINITY,
        -3.0,
        -1.0,
        0.0,
        1.0,
        3.0,
        f64::INFINITY,
    ]);
    let min = Simd::from_array([-2.0, -2.0, -2.0, -1.0, -1.0, 0.0, 2.0, 2.0]);
    let max = Simd::from_array([2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let values = values_vector.to_array();
    let min = min.to_array();
    let max = max.to_array();
    let expected = core::array::from_fn(|i| values[i].clamp(min[i], max[i]).to_bits());

    assert_eq!(
        values_vector
            .simd_clamp(Simd::from_array(min), Simd::from_array(max))
            .to_bits()
            .to_array(),
        expected
    );
}

#[test]
fn simd_min_and_max_allow_either_signed_zero_for_f32_and_f64() {
    let f32_left = Simd::from_array([-0.0_f32, 0.0, -0.0, 0.0]);
    let f32_right = Simd::from_array([0.0_f32, -0.0, -0.0, 0.0]);
    for result in [
        f32_left.simd_min(f32_right).to_bits().to_array(),
        f32_left.simd_max(f32_right).to_bits().to_array(),
    ] {
        assert!(result.into_iter().all(|bits| bits & 0x7fff_ffff == 0));
    }

    let f64_left = Simd::from_array([-0.0_f64, 0.0, -0.0, 0.0]);
    let f64_right = Simd::from_array([0.0_f64, -0.0, -0.0, 0.0]);
    for result in [
        f64_left.simd_min(f64_right).to_bits().to_array(),
        f64_left.simd_max(f64_right).to_bits().to_array(),
    ] {
        assert!(result
            .into_iter()
            .all(|bits| bits & 0x7fff_ffff_ffff_ffff == 0));
    }
}

#[test]
fn simd_min_and_max_keep_both_nan_lanes_nan() {
    let f32_left = Simd::from_array([f32::from_bits(0x7fc1_2345); 4]);
    let f32_right = Simd::from_array([f32::from_bits(0xffc5_4321); 4]);
    assert!(f32_left.simd_min(f32_right).to_array().into_iter().all(f32::is_nan));
    assert!(f32_left.simd_max(f32_right).to_array().into_iter().all(f32::is_nan));

    let f64_left = Simd::from_array([f64::from_bits(0x7ff8_1234_5678_9abc); 4]);
    let f64_right = Simd::from_array([f64::from_bits(0xfff8_4321_8765_cba9); 4]);
    assert!(f64_left.simd_min(f64_right).to_array().into_iter().all(f64::is_nan));
    assert!(f64_left.simd_max(f64_right).to_array().into_iter().all(f64::is_nan));
}
