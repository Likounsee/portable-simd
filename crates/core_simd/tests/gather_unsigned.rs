#![feature(portable_simd)]

use core_simd::simd::{Mask, Simd};

#[test]
fn gather_or_u32_handles_valid_and_out_of_bounds_indices() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([3_u32, 0, 2, u32::MAX]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    assert_eq!(
        Simd::gather_or_u32(&values, indices, fallback),
        Simd::from_array([40, 10, 30, -4])
    );
}

#[test]
fn gather_or_default_u32_uses_default_for_invalid_indices() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([1_u32, u32::MAX, 3, 4]);

    assert_eq!(
        Simd::gather_or_default_u32(&values, indices),
        Simd::from_array([20, 0, 40, 0])
    );
}

#[test]
fn gather_select_u32_respects_mask_before_access() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([2_u32, u32::MAX, 1, 3]);
    let enable = Mask::from_array([true, true, false, true]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    assert_eq!(
        Simd::gather_select_u32(&values, enable, indices, fallback),
        Simd::from_array([30, -2, -3, 40])
    );
}

#[test]
fn gather_u32_on_empty_slice_returns_fallback() {
    let indices = Simd::from_array([0_u32, 1, u32::MAX, 4]);
    let fallback = Simd::from_array([10_i32, 20, 30, 40]);

    assert_eq!(
        Simd::<i32, 4>::gather_or_u32(&[], indices, fallback),
        fallback
    );
}

#[test]
fn gather_or_u64_handles_valid_and_unrepresentable_indices() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([3_u64, 0, 2, u64::MAX]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    assert_eq!(
        Simd::gather_or_u64(&values, indices, fallback),
        Simd::from_array([40, 10, 30, -4])
    );
}

#[test]
fn gather_or_default_u64_uses_default_for_invalid_indices() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([1_u64, u64::MAX, 3, 4]);

    assert_eq!(
        Simd::gather_or_default_u64(&values, indices),
        Simd::from_array([20, 0, 40, 0])
    );
}

#[test]
fn gather_select_u64_respects_mask_before_access() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([2_u64, u64::MAX, 1, 3]);
    let enable = Mask::from_array([true, true, false, true]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    assert_eq!(
        Simd::gather_select_u64(&values, enable, indices, fallback),
        Simd::from_array([30, -2, -3, 40])
    );
}

#[test]
fn gather_u64_on_empty_slice_returns_fallback() {
    let indices = Simd::from_array([0_u64, 1, u64::MAX, 4]);
    let fallback = Simd::from_array([10_i32, 20, 30, 40]);

    assert_eq!(
        Simd::<i32, 4>::gather_or_u64(&[], indices, fallback),
        fallback
    );
}
