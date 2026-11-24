#![feature(portable_simd)]

use core_simd::simd::{Mask, Select, Simd};

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

#[test]
fn compress_and_expand_round_trip_every_four_lane_mask() {
    let values = Simd::from_array([3_u16, 5, 7, 11]);
    for bits in 0_u8..16 {
        let selected = core::array::from_fn(|lane| bits & (1 << lane) != 0);
        let mask = Mask::from_array(selected);
        let mut packed = [0_u16; 4];

        let written = values.compress_store(&mut packed, mask);
        let fallback = Simd::splat(0);
        let (expanded, consumed) = Simd::<u16, 4>::expand_load(&packed[..written], mask, fallback);

        assert_eq!(consumed, written, "mask={bits:04b}");
        assert_eq!(expanded, mask.select(values, fallback), "mask={bits:04b}");
    }
}

#[test]
fn prefix_masks_use_contiguous_bounded_copies() {
    let values = Simd::from_array([10_u32, 20, 30, 40]);
    let prefix = Mask::from_array([true, true, true, false]);
    let mut output = [99_u32; 2];

    assert_eq!(values.compress_store(&mut output, prefix), 2);
    assert_eq!(output, [10, 20]);

    let fallback = Simd::from_array([90, 91, 92, 93]);
    let (expanded, consumed) = Simd::<u32, 4>::expand_load(&[7, 8], prefix, fallback);
    assert_eq!(consumed, 2);
    assert_eq!(expanded.to_array(), [7, 8, 92, 93]);
}

#[test]
fn compress_expand_cover_every_eight_lane_mask_and_slice_capacity() {
    let values = Simd::from_array([10_i32, 20, 30, 40, 50, 60, 70, 80]);
    let fallback = Simd::from_array([-1_i32, -2, -3, -4, -5, -6, -7, -8]);

    for bits in 0_u16..256 {
        let selected = core::array::from_fn(|lane| bits & (1 << lane) != 0);
        let mask = Mask::from_array(selected);
        let selected_count = selected.iter().filter(|&&lane| lane).count();

        for capacity in 0..=8 {
            let mut packed = [999_i32; 8];
            let written = values.compress_store(&mut packed[..capacity], mask);
            let expected_written = selected_count.min(capacity);

            assert_eq!(
                written, expected_written,
                "compress mask={bits:08b}, capacity={capacity}"
            );
            let expected_packed: Vec<i32> = values
                .to_array()
                .into_iter()
                .zip(selected)
                .filter_map(|(value, is_selected)| is_selected.then_some(value))
                .take(capacity)
                .collect();
            assert_eq!(
                &packed[..written],
                expected_packed.as_slice(),
                "packed values for mask={bits:08b}, capacity={capacity}"
            );
            assert!(
                packed[written..].iter().all(|&value| value == 999),
                "compress_store modified beyond its returned length: mask={bits:08b}, capacity={capacity}"
            );

            let input: Vec<i32> = (0..capacity).map(|index| 100 + index as i32).collect();
            let (expanded, consumed) = Simd::<i32, 8>::expand_load(&input, mask, fallback);
            let expected_consumed = selected_count.min(capacity);
            let mut expected = fallback.to_array();
            let mut input_index = 0;
            for lane in 0..8 {
                if selected[lane] && input_index < expected_consumed {
                    expected[lane] = input[input_index];
                    input_index += 1;
                }
            }

            assert_eq!(
                consumed, expected_consumed,
                "expand_load consumed count for mask={bits:08b}, capacity={capacity}"
            );
            assert_eq!(
                expanded.to_array(),
                expected,
                "expanded lanes for mask={bits:08b}, capacity={capacity}"
            );
        }
    }
}
