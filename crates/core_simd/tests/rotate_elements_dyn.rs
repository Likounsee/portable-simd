#![feature(portable_simd)]

use core_simd::simd::Simd;

fn check_runtime_rotations<const LANES: usize>() {
    let values: [u32; LANES] =
        core::array::from_fn(|lane| (lane as u32).wrapping_mul(0x9e37_79b9) ^ 0xa5a5_5a5a);
    let vector = Simd::<u32, LANES>::from_array(values);

    // Include offsets around the lane-count boundary and the largest usize.
    // The API promises modulo-LANES behavior for every runtime offset.
    for offset in [0, 1, LANES - 1, LANES, LANES + 1, usize::MAX] {
        let mut expected_left = values;
        expected_left.rotate_left(offset % LANES);
        assert_eq!(
            vector.rotate_elements_left_dyn(offset).to_array(),
            expected_left,
            "left rotation with {LANES} lanes and offset {offset}",
        );

        let mut expected_right = values;
        expected_right.rotate_right(offset % LANES);
        assert_eq!(
            vector.rotate_elements_right_dyn(offset).to_array(),
            expected_right,
            "right rotation with {LANES} lanes and offset {offset}",
        );
    }
}

test_helpers::test_lanes! {
    fn runtime_element_rotations_match_array<const LANES: usize>() {
        check_runtime_rotations::<LANES>();
    }
}

#[test]
fn runtime_element_rotations_preserve_single_lane_vectors() {
    check_runtime_rotations::<1>();
}

fn check_static_shuffle_helpers<const LANES: usize>() {
    let values: [u32; LANES] = core::array::from_fn(|lane| lane as u32 * 3 + 1);
    let other: [u32; LANES] = core::array::from_fn(|lane| lane as u32 * 5 + 101);
    let padding = u32::MAX - 7;
    let vector = Simd::<u32, LANES>::from_array(values);
    let other_vector = Simd::<u32, LANES>::from_array(other);

    assert_eq!(vector.reverse().to_array(), core::array::from_fn(|lane| values[LANES - lane - 1]));
    assert_eq!(vector.align_elements::<0>(other_vector).to_array(), values);
    assert_eq!(vector.align_elements::<LANES>(other_vector).to_array(), other);

    let align_one: [u32; LANES] = core::array::from_fn(|lane| {
        if lane + 1 < LANES {
            values[lane + 1]
        } else {
            other[0]
        }
    });
    assert_eq!(vector.align_elements::<1>(other_vector).to_array(), align_one);

    let shift_left_one: [u32; LANES] = core::array::from_fn(|lane| {
        if lane + 1 < LANES { values[lane + 1] } else { padding }
    });
    let shift_right_one: [u32; LANES] = core::array::from_fn(|lane| {
        if lane == 0 { padding } else { values[lane - 1] }
    });
    assert_eq!(vector.shift_elements_left::<1>(padding).to_array(), shift_left_one);
    assert_eq!(vector.shift_elements_right::<1>(padding).to_array(), shift_right_one);
    assert_eq!(vector.shift_elements_left::<LANES>(padding).to_array(), [padding; LANES]);
    assert_eq!(vector.shift_elements_right::<LANES>(padding).to_array(), [padding; LANES]);
}

test_helpers::test_lanes! {
    fn static_shuffle_helpers_match_array<const LANES: usize>() {
        check_static_shuffle_helpers::<LANES>();
    }
}
