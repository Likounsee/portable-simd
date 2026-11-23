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
