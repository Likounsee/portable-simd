#![feature(portable_simd)]
#![cfg(target_has_reliable_f16_math)]
#![feature(f16)]

use core_simd::simd::{Simd, num::SimdFloat};

#[test]
fn f16_simd_min_matches_scalar_nan_handling() {
    let left = Simd::from_array([
        f16::from_bits(0x7e01),
        -5.0_f16,
        3.0,
        f16::INFINITY,
        f16::NEG_INFINITY,
        0.5,
        -2.0,
        8.0,
    ]);
    let right = Simd::from_array([1.0_f16, f16::NAN, 4.0, 7.0, -7.0, -0.5, 2.0, 9.0]);
    let lhs = left.to_array();
    let rhs = right.to_array();
    let actual = left.simd_min(right).to_array();

    for lane in 0..8 {
        assert_eq!(actual[lane].to_bits(), lhs[lane].min(rhs[lane]).to_bits());
    }
}

#[test]
fn f16_simd_max_matches_scalar_nan_handling() {
    let left = Simd::from_array([
        f16::from_bits(0x7e01),
        -5.0_f16,
        3.0,
        f16::INFINITY,
        f16::NEG_INFINITY,
        0.5,
        -2.0,
        8.0,
    ]);
    let right = Simd::from_array([1.0_f16, f16::NAN, 4.0, 7.0, -7.0, -0.5, 2.0, 9.0]);
    let lhs = left.to_array();
    let rhs = right.to_array();
    let actual = left.simd_max(right).to_array();

    for lane in 0..8 {
        assert_eq!(actual[lane].to_bits(), lhs[lane].max(rhs[lane]).to_bits());
    }
}

#[test]
fn f16_simd_min_max_allow_either_signed_zero() {
    let left = Simd::from_array([-0.0_f16, 0.0, -0.0, 0.0]);
    let right = Simd::from_array([0.0_f16, -0.0, -0.0, 0.0]);

    for result in [
        left.simd_min(right).to_bits().to_array(),
        left.simd_max(right).to_bits().to_array(),
    ] {
        for bits in result {
            assert!(bits == 0x0000 || bits == 0x8000);
        }
    }
}

#[test]
fn f16_simd_clamp_matches_scalar_behavior() {
    let values = Simd::from_array([
        f16::NAN,
        f16::NEG_INFINITY,
        -3.0_f16,
        -1.0,
        0.0,
        1.0,
        3.0,
        f16::INFINITY,
    ]);
    let min = Simd::splat(-2.0_f16);
    let max = Simd::splat(2.0_f16);
    let scalar_values = values.to_array();
    let actual = values.simd_clamp(min, max).to_array();

    for lane in 0..8 {
        let expected = scalar_values[lane].clamp(-2.0_f16, 2.0_f16);
        if expected.is_nan() {
            assert!(actual[lane].is_nan());
        } else {
            assert_eq!(actual[lane].to_bits(), expected.to_bits());
        }
    }
}
