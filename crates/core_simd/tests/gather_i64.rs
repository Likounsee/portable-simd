#![feature(portable_simd)]

use core_simd::simd::{Mask, Simd};

#[test]
fn gather_or_i64_handles_valid_negative_and_extreme_indices() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([3_i64, -1, 0, i64::MAX]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    assert_eq!(
        Simd::gather_or_i64(&values, indices, fallback),
        Simd::from_array([40, -2, 10, -4])
    );
}

#[test]
fn gather_or_default_i64_uses_default_for_invalid_indices() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([1_i64, i64::MIN, 3, i64::MAX]);

    assert_eq!(
        Simd::gather_or_default_i64(&values, indices),
        Simd::from_array([20, 0, 40, 0])
    );
}

#[test]
fn gather_select_i64_respects_mask_and_bounds() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([2_i64, i64::MAX, -1, 3]);
    let enable = Mask::from_array([true, true, false, true]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    assert_eq!(
        Simd::gather_select_i64(&values, enable, indices, fallback),
        Simd::from_array([30, -2, -3, 40])
    );
}

#[test]
fn gather_i64_on_empty_slice_returns_fallback() {
    let indices = Simd::from_array([0_i64, 1, i64::MIN, i64::MAX]);
    let fallback = Simd::from_array([10_i32, 20, 30, 40]);

    assert_eq!(
        Simd::<i32, 4>::gather_or_i64(&[], indices, fallback),
        fallback
    );
}
