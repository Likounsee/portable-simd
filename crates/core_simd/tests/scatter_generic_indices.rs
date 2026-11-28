#![feature(portable_simd)]

use core_simd::simd::{Mask, Simd};

#[test]
fn generic_scatter_accepts_narrow_signed_indices() {
    let mut values = [0_i32; 4];
    let indices = Simd::from_array([3_i8, -1, 0, i8::MAX]);
    Simd::from_array([10, 20, 30, 40]).scatter_index(&mut values, indices);
    assert_eq!(values, [30, 0, 0, 10]);
}

#[test]
fn generic_scatter_select_accepts_unsigned_indices_and_mask() {
    let mut values = [0_i32; 4];
    let indices = Simd::from_array([2_u16, 1, u16::MAX, 3]);
    let enable = Mask::from_array([true, false, true, true]);
    Simd::from_array([10, 20, 30, 40]).scatter_select_index(&mut values, enable, indices);
    assert_eq!(values, [0, 0, 10, 40]);
}

#[test]
fn generic_scatter_ignores_large_and_unrepresentable_indices() {
    let mut values = [0_i32; 4];
    let indices = Simd::from_array([3_u64, 1, u64::MAX, 0]);
    Simd::from_array([10, 20, 30, 40]).scatter_index(&mut values, indices);
    assert_eq!(values, [40, 20, 0, 10]);
}

#[test]
fn generic_scatter_handles_pointer_sized_signed_extremes() {
    let mut values = [0_i32; 4];
    let indices = Simd::from_array([0_isize, 3, isize::MIN, isize::MAX]);
    Simd::from_array([10, 20, 30, 40]).scatter_index(&mut values, indices);
    assert_eq!(values, [10, 0, 0, 20]);
}

#[test]
fn generic_scatter_select_masks_invalid_pointer_sized_indices() {
    let mut values = [0_i32; 4];
    let indices = Simd::from_array([isize::MIN, 1, isize::MAX, 2]);
    let enable = Mask::from_array([false, true, true, false]);
    Simd::from_array([10, 20, 30, 40]).scatter_select_index(&mut values, enable, indices);
    assert_eq!(values, [0, 20, 0, 0]);
}

#[test]
fn generic_scatter_on_empty_slice_is_a_noop() {
    let indices = Simd::from_array([0_i16, -1, 1, i16::MAX]);
    Simd::from_array([10_i32, 20, 30, 40]).scatter_index(&mut [], indices);
}

#[test]
fn generic_scatter_duplicate_indices_follow_existing_scatter_contract() {
    let mut values = [0_i32; 4];
    let indices = Simd::from_array([1_i16, 1, 2, 3]);
    Simd::from_array([10, 20, 30, 40]).scatter_index(&mut values, indices);
    assert!(values[1] == 10 || values[1] == 20);
    assert_eq!(values[2..], [30, 40]);
}

test_helpers::test_lanes! {
    fn generic_scatter_signed_indices_cover_supported_lane_counts<const LANES: usize>() {
        let mut destination = [0_i32; 3];
        let values = Simd::from_array(core::array::from_fn(|lane| lane as i32 + 100));
        let indices = Simd::from_array(core::array::from_fn(|lane| {
            if lane < destination.len() { lane as i8 } else { -1_i8 }
        }));
        values.scatter_index(&mut destination, indices);

        let expected = core::array::from_fn(|index| {
            if index < LANES { index as i32 + 100 } else { 0 }
        });
        assert_eq!(destination, expected, "signed scatter, lanes={LANES}");
    }

    fn generic_scatter_unsigned_indices_cover_supported_lane_counts<const LANES: usize>() {
        let mut destination = [0_i32; 3];
        let values = Simd::from_array(core::array::from_fn(|lane| lane as i32 + 200));
        let indices = Simd::from_array(core::array::from_fn(|lane| {
            if lane < destination.len() { lane as u64 } else { u64::MAX }
        }));
        let enable = Mask::from_array(core::array::from_fn(|lane| lane % 2 == 0));
        values.scatter_select_index(&mut destination, enable, indices);

        let expected = core::array::from_fn(|index| {
            if index < LANES && index % 2 == 0 { index as i32 + 200 } else { 0 }
        });
        assert_eq!(destination, expected, "unsigned masked scatter, lanes={LANES}");
    }
}
