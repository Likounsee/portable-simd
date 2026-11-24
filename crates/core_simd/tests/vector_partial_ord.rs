#![feature(portable_simd)]

use core::cmp::Ordering;
use core_simd::simd::Simd;

#[test]
fn integer_partial_ord_matches_array_lexicographic_order() {
    let cases = [
        ([1_i32, 2, 3, 4], [1, 2, 3, 5]),
        ([1_i32, 3, 0, 0], [1, 2, 9, 9]),
        ([0_i32, 0, 0, 0], [0, 0, 0, 0]),
        ([-1_i32, 9, 9, 9], [0, -9, -9, -9]),
    ];

    for (left, right) in cases {
        assert_eq!(
            Simd::from_array(left).partial_cmp(&Simd::from_array(right)),
            left.partial_cmp(&right),
            "left={left:?}, right={right:?}",
        );
    }
}

#[test]
fn float_partial_ord_matches_array_lexicographic_order() {
    let cases = [
        ([1.0_f32, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0]),
        ([1.0_f32, 3.0, 0.0, 0.0], [1.0, 2.0, 9.0, 9.0]),
        ([-0.0_f32, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]),
        ([f32::NAN, 1.0, 2.0, 3.0], [f32::NAN, 1.0, 2.0, 3.0]),
        ([f32::NAN, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]),
        ([0.0_f32, f32::NAN, 2.0, 3.0], [0.0, f32::NAN, 2.0, 4.0]),
        ([0.0_f32, f32::NAN, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]),
    ];

    for (left, right) in cases {
        assert_eq!(
            Simd::from_array(left).partial_cmp(&Simd::from_array(right)),
            left.partial_cmp(&right),
            "left={left:?}, right={right:?}",
        );
    }
}

#[test]
fn integer_ord_is_lexicographic() {
    let left = Simd::from_array([1_i32, 2, 9, 9]);
    let right = Simd::from_array([1_i32, 3, 0, 0]);

    assert_eq!(left.cmp(&right), Ordering::Less);
    assert_eq!(right.cmp(&left), Ordering::Greater);
    assert_eq!(left.cmp(&left), Ordering::Equal);
}

#[test]
fn mask_partial_ord_uses_boolean_lane_ordering() {
    use core_simd::simd::Mask;

    let left = Mask::<i8, 4>::from_array([false, true, false, false]);
    let right = Mask::<i8, 4>::from_array([false, false, false, true]);

    assert_eq!(left.partial_cmp(&right), Some(Ordering::Greater));
    assert_eq!(right.partial_cmp(&left), Some(Ordering::Less));
    assert_eq!(left.partial_cmp(&left), Some(Ordering::Equal));
}

#[test]
fn mask_simd_partial_ord_matches_boolean_lane_ordering() {
    use core_simd::simd::{Mask, cmp::SimdPartialOrd};

    // Exhaust every pair of four-lane masks to guard the boolean ordering
    // semantics independently of the integer representation used by Mask.
    for left_bits in 0_u8..16 {
        for right_bits in 0_u8..16 {
            let left_lanes: [bool; 4] = core::array::from_fn(|lane| left_bits & (1 << lane) != 0);
            let right_lanes: [bool; 4] = core::array::from_fn(|lane| right_bits & (1 << lane) != 0);
            let left = Mask::<i8, 4>::from_array(left_lanes);
            let right = Mask::<i8, 4>::from_array(right_lanes);

            // PartialOrd on Mask must follow bool-array lexicographic ordering,
            // not the signed integer representation used internally by masks.
            assert_eq!(
                left.partial_cmp(&right),
                left_lanes.partial_cmp(&right_lanes),
                "partial_cmp: left={left_lanes:?}, right={right_lanes:?}",
            );

            let expected_lt = core::array::from_fn(|lane| left_lanes[lane] < right_lanes[lane]);
            let expected_le = core::array::from_fn(|lane| left_lanes[lane] <= right_lanes[lane]);
            let expected_gt = core::array::from_fn(|lane| left_lanes[lane] > right_lanes[lane]);
            let expected_ge = core::array::from_fn(|lane| left_lanes[lane] >= right_lanes[lane]);

            assert_eq!(
                left.simd_lt(right).to_array(),
                expected_lt,
                "simd_lt: left={left_lanes:?}, right={right_lanes:?}",
            );
            assert_eq!(
                left.simd_le(right).to_array(),
                expected_le,
                "simd_le: left={left_lanes:?}, right={right_lanes:?}",
            );
            assert_eq!(
                left.simd_gt(right).to_array(),
                expected_gt,
                "simd_gt: left={left_lanes:?}, right={right_lanes:?}",
            );
            assert_eq!(
                left.simd_ge(right).to_array(),
                expected_ge,
                "simd_ge: left={left_lanes:?}, right={right_lanes:?}",
            );
        }
    }
}
