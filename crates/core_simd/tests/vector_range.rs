#![feature(portable_simd)]

use core_simd::simd::{Simd, cmp::SimdPartialOrd};

#[test]
fn integer_simd_in_range_is_inclusive() {
    let values = Simd::from_array([-1_i32, 0, 5, 10, 11]);
    let mask = values.simd_in_range(Simd::splat(0), Simd::splat(10));

    assert_eq!(mask.to_array(), [false, true, true, true, false]);
}

#[test]
fn simd_in_range_handles_per_lane_bounds() {
    let values = Simd::from_array([0_u32, 5, 10, 15]);
    let min = Simd::from_array([0, 4, 11, 10]);
    let max = Simd::from_array([0, 6, 20, 14]);

    assert_eq!(
        values.simd_in_range(min, max).to_array(),
        [true, true, false, false],
    );
}

#[test]
fn float_simd_in_range_rejects_nan() {
    let values = Simd::from_array([f32::NAN, -1.0, 0.5, 2.0]);
    let mask = values.simd_in_range(Simd::splat(-1.0), Simd::splat(1.0));

    assert_eq!(mask.to_array(), [false, true, true, false]);
}

#[test]
fn simd_in_range_rejects_inverted_intervals() {
    let values = Simd::from_array([0_i32, 5, 10, 15]);
    let min = Simd::from_array([0, 6, 10, 20]);
    let max = Simd::from_array([0, 4, 10, 19]);

    assert_eq!(
        values.simd_in_range(min, max).to_array(),
        [true, false, true, false],
    );
}

#[test]
fn simd_range_supports_exclusive_bounds() {
    let values = Simd::from_array([0_i32, 1, 2, 3, 4]);
    let min = Simd::splat(1);
    let max = Simd::splat(4);

    assert_eq!(
        values.simd_range(min..max).to_array(),
        [false, true, true, true, false],
    );
}

#[test]
fn simd_range_supports_one_sided_bounds() {
    let values = Simd::from_array([0_i32, 1, 2, 3, 4]);
    let two = Simd::splat(2);

    assert_eq!(
        values.simd_range(..=two).to_array(),
        [true, true, true, false, false],
    );
    assert_eq!(
        values.simd_range(two..).to_array(),
        [false, false, true, true, true],
    );
}

#[test]
fn fully_unbounded_simd_range_matches_nan_lanes() {
    let values = Simd::from_array([f32::NAN, f32::NEG_INFINITY, 0.0, f32::INFINITY]);

    assert_eq!(values.simd_range(..).to_array(), [true; 4]);
}

#[test]
fn simd_range_uses_per_lane_bounds() {
    let values = Simd::from_array([0_i32, 5, 10, 15]);
    let min = Simd::from_array([0, 4, 11, 10]);
    let max = Simd::from_array([0, 6, 20, 14]);

    assert_eq!(
        values.simd_range(min..=max).to_array(),
        [true, true, false, false],
    );
}
