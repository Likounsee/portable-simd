#![feature(portable_simd)]

use core_simd::simd::prelude::*;

#[test]
fn generic_lane_count_aliases_are_available_from_the_prelude() {
    let integers: u32xN<3> = Simd::from_array([1, 2, 3]);
    let floats: f32xN<3> = Simd::from_array([1.0, 2.0, 3.0]);
    let mask: mask32xN<3> = integers.simd_eq(Simd::splat(2));

    assert_eq!(integers.to_array(), [1, 2, 3]);
    assert_eq!(floats.to_array(), [1.0, 2.0, 3.0]);
    assert_eq!(mask.to_array(), [false, true, false]);
}

#[test]
fn generic_aliases_support_non_power_of_two_lane_counts() {
    let signed: i16xN<5> = Simd::from_array([-2, -1, 0, 1, 2]);
    let unsigned: u8xN<5> = Simd::from_array([0, 1, 2, 3, 4]);

    assert_eq!(signed.to_array(), [-2, -1, 0, 1, 2]);
    assert_eq!(unsigned.to_array(), [0, 1, 2, 3, 4]);
}
