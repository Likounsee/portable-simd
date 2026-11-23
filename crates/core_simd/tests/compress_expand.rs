#![feature(portable_simd)]

use core_simd::simd::{Mask, Simd};

#[test]
fn compress_store_preserves_selected_lane_order() {
    let values = Simd::from_array([10_i32, 20, 30, 40, 50, 60, 70, 80]);
    let mask = Mask::from_array([false, true, true, false, true, false, false, true]);
    let mut output = [-1; 8];

    let written = values.compress_store(&mut output, mask);

    assert_eq!(written, 4);
    assert_eq!(&output[..written], &[20, 30, 50, 80]);
    assert_eq!(&output[written..], &[-1; 4]);
}

#[test]
fn compress_store_does_not_write_past_short_destination() {
    let values = Simd::from_array([1_u32, 2, 3, 4]);
    let mask = Mask::splat(true);
    let mut output = [99; 2];

    let written = values.compress_store(&mut output, mask);

    assert_eq!(written, 2);
    assert_eq!(output, [1, 2]);
}

#[test]
fn expand_load_places_values_in_selected_lanes() {
    let input = [11_i16, 22, 33];
    let mask = Mask::from_array([false, true, false, true, true, false]);
    let fallback = Simd::splat(-1);

    let (result, consumed) = Simd::<i16, 6>::expand_load(&input, mask, fallback);

    assert_eq!(consumed, 3);
    assert_eq!(result.to_array(), [-1, 11, -1, 22, 33, -1]);
}

#[test]
fn expand_load_keeps_fallback_for_missing_values() {
    let input = [7_u8];
    let mask = Mask::from_array([true, false, true, true]);
    let fallback = Simd::from_array([90, 91, 92, 93]);

    let (result, consumed) = Simd::<u8, 4>::expand_load(&input, mask, fallback);

    assert_eq!(consumed, 1);
    assert_eq!(result.to_array(), [7, 91, 92, 93]);
}

#[test]
fn empty_mask_and_empty_slice_are_supported() {
    let values = Simd::from_array([1_u64, 2, 3, 4]);
    let mask = Mask::splat(false);
    let mut output = [];

    assert_eq!(values.compress_store(&mut output, mask), 0);

    let fallback = Simd::from_array([5, 6, 7, 8]);
    let (result, consumed) = Simd::<u64, 4>::expand_load(&[], mask, fallback);

    assert_eq!(consumed, 0);
    assert_eq!(result, fallback);
}
