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
