#![feature(portable_simd)]
#![feature(f16)]

use core_simd::simd::{Simd, ToBytes, num::SimdFloat};

#[test]
fn byte_convert() {
    let int = Simd::<u32, 2>::from_array([0xdeadbeef, 0x8badf00d]);
    let ne_bytes = int.to_ne_bytes();
    let be_bytes = int.to_be_bytes();
    let le_bytes = int.to_le_bytes();
    assert_eq!(int.get(0).to_ne_bytes(), ne_bytes.as_array()[..4]);
    assert_eq!(int.get(1).to_ne_bytes(), ne_bytes.as_array()[4..]);
    assert_eq!(int.get(0).to_be_bytes(), be_bytes.as_array()[..4]);
    assert_eq!(int.get(1).to_be_bytes(), be_bytes.as_array()[4..]);
    assert_eq!(int.get(0).to_le_bytes(), le_bytes.as_array()[..4]);
    assert_eq!(int.get(1).to_le_bytes(), le_bytes.as_array()[4..]);
    assert_eq!(Simd::<u32, 2>::from_ne_bytes(ne_bytes), int);
    assert_eq!(Simd::<u32, 2>::from_be_bytes(be_bytes), int);
    assert_eq!(Simd::<u32, 2>::from_le_bytes(le_bytes), int);
}

#[test]
fn float_byte_conversions_preserve_f32_bit_patterns() {
    let values: Simd<f32, 4> = Simd::from_bits(Simd::<u32, 4>::from_array([
        0x0000_0000,
        0x8000_0000,
        0x7fc1_2345,
        0xff80_0001,
    ]));
    let native = values.to_ne_bytes();
    let big = values.to_be_bytes();
    let little = values.to_le_bytes();

    assert_eq!(
        Simd::<f32, 4>::from_ne_bytes(native).to_bits(),
        values.to_bits()
    );
    assert_eq!(
        Simd::<f32, 4>::from_be_bytes(big).to_bits(),
        values.to_bits()
    );
    assert_eq!(
        Simd::<f32, 4>::from_le_bytes(little).to_bits(),
        values.to_bits()
    );
}

#[test]
fn float_byte_conversions_preserve_f64_bit_patterns() {
    let values: Simd<f64, 2> = Simd::from_bits(Simd::<u64, 2>::from_array([
        0x0000_0000_0000_0000,
        0xfff8_1234_5678_9abc,
    ]));
    let native = values.to_ne_bytes();
    let big = values.to_be_bytes();
    let little = values.to_le_bytes();

    assert_eq!(
        Simd::<f64, 2>::from_ne_bytes(native).to_bits(),
        values.to_bits()
    );
    assert_eq!(
        Simd::<f64, 2>::from_be_bytes(big).to_bits(),
        values.to_bits()
    );
    assert_eq!(
        Simd::<f64, 2>::from_le_bytes(little).to_bits(),
        values.to_bits()
    );
}

#[test]
fn signed_integer_byte_conversions_roundtrip_all_endiannesses() {
    let values = Simd::<i64, 2>::from_array([i64::MIN, -0x0123_4567_89ab_cdef]);

    assert_eq!(Simd::<i64, 2>::from_ne_bytes(values.to_ne_bytes()), values);
    assert_eq!(Simd::<i64, 2>::from_be_bytes(values.to_be_bytes()), values);
    assert_eq!(Simd::<i64, 2>::from_le_bytes(values.to_le_bytes()), values);
}

#[test]
fn unsigned_integer_byte_conversions_roundtrip_all_endiannesses() {
    let values = Simd::<u16, 4>::from_array([0, 1, 0x8000, u16::MAX]);

    assert_eq!(Simd::<u16, 4>::from_ne_bytes(values.to_ne_bytes()), values);
    assert_eq!(Simd::<u16, 4>::from_be_bytes(values.to_be_bytes()), values);
    assert_eq!(Simd::<u16, 4>::from_le_bytes(values.to_le_bytes()), values);
}

#[test]
fn f16_byte_conversions_preserve_all_bit_patterns_and_endiannesses() {
    let values: Simd<f16, 4> =
        Simd::from_bits(Simd::<u16, 4>::from_array([0x0000, 0x8000, 0x7e11, 0xfc01]));

    assert_eq!(
        Simd::<f16, 4>::from_ne_bytes(values.to_ne_bytes()).to_bits(),
        values.to_bits()
    );
    assert_eq!(
        Simd::<f16, 4>::from_be_bytes(values.to_be_bytes()).to_bits(),
        values.to_bits()
    );
    assert_eq!(
        Simd::<f16, 4>::from_le_bytes(values.to_le_bytes()).to_bits(),
        values.to_bits()
    );
}

fn check_odd_lane_byte_roundtrip<const N: usize>()
where
    Simd<i32, N>: ToBytes,
{
    let values = Simd::<i32, N>::from_array(core::array::from_fn(|lane| {
        (lane as i32).wrapping_mul(0x1020_304) ^ i32::MIN
    }));

    assert_eq!(Simd::<i32, N>::from_ne_bytes(values.to_ne_bytes()), values);
    assert_eq!(Simd::<i32, N>::from_be_bytes(values.to_be_bytes()), values);
    assert_eq!(Simd::<i32, N>::from_le_bytes(values.to_le_bytes()), values);
}

#[test]
fn byte_conversions_support_non_power_of_two_lane_counts() {
    check_odd_lane_byte_roundtrip::<3>();
    check_odd_lane_byte_roundtrip::<5>();
    check_odd_lane_byte_roundtrip::<7>();
    check_odd_lane_byte_roundtrip::<9>();
    check_odd_lane_byte_roundtrip::<15>();
}

fn check_odd_lane_byte_layout<const N: usize>()
where
    Simd<i32, N>: ToBytes,
{
    let values = Simd::<i32, N>::from_array(core::array::from_fn(|lane| {
        (lane as i32).wrapping_mul(0x1020_304) ^ i32::MIN
    }));
    let scalars = values.to_array();

    let expected_ne: Vec<u8> = scalars
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let expected_be: Vec<u8> = scalars
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect();
    let expected_le: Vec<u8> = scalars
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    assert_eq!(
        values.to_ne_bytes().as_ref(),
        expected_ne.as_slice()
    );
    assert_eq!(
        values.to_be_bytes().as_ref(),
        expected_be.as_slice()
    );
    assert_eq!(
        values.to_le_bytes().as_ref(),
        expected_le.as_slice()
    );
}

#[test]
fn byte_conversions_preserve_lane_order_and_endian_layout_for_odd_lane_counts() {
    check_odd_lane_byte_layout::<3>();
    check_odd_lane_byte_layout::<5>();
    check_odd_lane_byte_layout::<7>();
    check_odd_lane_byte_layout::<9>();
    check_odd_lane_byte_layout::<15>();
}

macro_rules! check_byte_layout {
    ($ty:ty, $lanes:literal) => {{
        let values = Simd::<$ty, $lanes>::from_array(core::array::from_fn(|lane| {
            (lane as $ty).wrapping_mul(3).wrapping_add(1)
        }));
        let scalars = values.to_array();

        let expected_ne: Vec<u8> = scalars
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let expected_be: Vec<u8> = scalars
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect();
        let expected_le: Vec<u8> = scalars
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();

        let native = values.to_ne_bytes();
        let big = values.to_be_bytes();
        let little = values.to_le_bytes();

        assert_eq!(native.as_array().as_slice(), expected_ne.as_slice());
        assert_eq!(big.as_array().as_slice(), expected_be.as_slice());
        assert_eq!(little.as_array().as_slice(), expected_le.as_slice());

        assert_eq!(Simd::<$ty, $lanes>::from_ne_bytes(native), values);
        assert_eq!(Simd::<$ty, $lanes>::from_be_bytes(big), values);
        assert_eq!(Simd::<$ty, $lanes>::from_le_bytes(little), values);
    }};
}

#[test]
fn byte_conversions_cover_odd_lane_counts_across_integer_widths() {
    check_byte_layout!(u8, 3);
    check_byte_layout!(u8, 7);
    check_byte_layout!(u8, 9);
    check_byte_layout!(u8, 15);
    check_byte_layout!(u8, 31);
    check_byte_layout!(u8, 63);

    check_byte_layout!(i8, 5);
    check_byte_layout!(i8, 17);
    check_byte_layout!(i8, 33);
    check_byte_layout!(i8, 63);

    check_byte_layout!(u16, 3);
    check_byte_layout!(u16, 5);
    check_byte_layout!(u16, 7);
    check_byte_layout!(u16, 15);
    check_byte_layout!(u16, 31);

    check_byte_layout!(i16, 3);
    check_byte_layout!(i16, 9);
    check_byte_layout!(i16, 17);
    check_byte_layout!(i16, 31);

    check_byte_layout!(u32, 3);
    check_byte_layout!(u32, 5);
    check_byte_layout!(u32, 7);
    check_byte_layout!(u32, 9);
    check_byte_layout!(u32, 15);

    check_byte_layout!(i64, 3);
    check_byte_layout!(i64, 5);
    check_byte_layout!(i64, 7);
    check_byte_layout!(u64, 3);
    check_byte_layout!(u64, 5);
    check_byte_layout!(u64, 7);
}
