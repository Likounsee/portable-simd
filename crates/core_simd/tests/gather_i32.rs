#![feature(portable_simd)]

use core_simd::simd::{Mask, Simd};

#[test]
fn gather_or_i32_handles_negative_and_out_of_bounds_indices() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([3_i32, -1, 0, 4]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    let actual = Simd::gather_or_i32(&values, indices, fallback);
    assert_eq!(actual, Simd::from_array([40, -2, 10, -4]));
}

#[test]
fn gather_or_default_i32_uses_default_for_invalid_indices() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([1_i32, -1, 3, i32::MAX]);

    let actual = Simd::gather_or_default_i32(&values, indices);
    assert_eq!(actual, Simd::from_array([20, 0, 40, 0]));
}

#[test]
fn gather_select_i32_respects_mask_and_signed_bounds() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([2_i32, 1, -1, 3]);
    let enable = Mask::from_array([true, false, true, true]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    let actual = Simd::gather_select_i32(&values, enable, indices, fallback);
    assert_eq!(actual, Simd::from_array([30, -2, -3, 40]));
}
