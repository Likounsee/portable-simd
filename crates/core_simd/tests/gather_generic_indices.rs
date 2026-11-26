#![feature(portable_simd)]

use core_simd::simd::{Mask, Simd};

#[test]
fn generic_gather_accepts_narrow_signed_indices() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([3_i8, -1, 0, i8::MAX]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    assert_eq!(
        Simd::gather_or_index(&values, indices, fallback),
        Simd::from_array([40, -2, 10, -4])
    );
}

#[test]
fn generic_gather_default_accepts_narrow_signed_indices() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([1_i16, -1, 3, i16::MAX]);

    assert_eq!(
        Simd::gather_or_default_index(&values, indices),
        Simd::from_array([20, 0, 40, 0])
    );
}

#[test]
fn generic_gather_select_accepts_unsigned_indices_and_mask() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([2_u16, 1, u16::MAX, 3]);
    let enable = Mask::from_array([true, false, true, true]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    assert_eq!(
        Simd::gather_select_index(&values, enable, indices, fallback),
        Simd::from_array([30, -2, -3, 40])
    );
}

#[test]
fn generic_gather_handles_indices_wider_than_usize() {
    let values = [10_i32, 20, 30, 40];
    let indices = Simd::from_array([3_u128, 1, u128::MAX, 0]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    assert_eq!(
        Simd::gather_or_index(&values, indices, fallback),
        Simd::from_array([40, 20, -3, 10])
    );
}

#[test]
fn generic_gather_on_empty_slice_returns_fallback() {
    let indices = Simd::from_array([0_i16, -1, 1, i16::MAX]);
    let fallback = Simd::from_array([10_i32, 20, 30, 40]);

    assert_eq!(
        Simd::<i32, 4>::gather_or_index(&[], indices, fallback),
        fallback
    );
}
