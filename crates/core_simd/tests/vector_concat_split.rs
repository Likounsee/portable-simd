#![feature(portable_simd)]

use core_simd::simd::Simd;

macro_rules! check_concat_split {
    ($small:literal, $large:literal) => {{
        let left = Simd::<u32, $small>::from_array(core::array::from_fn(|lane| lane as u32));
        let right =
            Simd::<u32, $small>::from_array(core::array::from_fn(|lane| (lane + $small) as u32));

        let joined: Simd<u32, $large> = left.concat::<$large, $small>(right);
        let expected: [u32; $large] = core::array::from_fn(|lane| lane as u32);
        assert_eq!(joined.to_array(), expected);

        let [first, second] = joined.split::<$small>();
        assert_eq!(first.to_array(), left.to_array());
        assert_eq!(second.to_array(), right.to_array());
    }};
}

#[test]
fn concat_and_split_round_trip_all_supported_equal_width_pairs() {
    check_concat_split!(1, 2);
    check_concat_split!(2, 4);
    check_concat_split!(4, 8);
    check_concat_split!(8, 16);
    check_concat_split!(16, 32);
    check_concat_split!(32, 64);
}

#[test]
fn concat_and_split_support_float_lanes() {
    let left = Simd::<f32, 4>::from_array([1.0, 2.0, 3.0, 4.0]);
    let right = Simd::<f32, 4>::from_array([5.0, 6.0, 7.0, 8.0]);

    let joined: Simd<f32, 8> = left.concat::<8, 4>(right);
    assert_eq!(joined.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let [first, second] = joined.split::<4>();
    assert_eq!(first, left);
    assert_eq!(second, right);
}
