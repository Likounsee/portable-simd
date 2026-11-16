#![feature(portable_simd)]

use core_simd::simd::{Simd, ToBytes};

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
    let values = Simd::from_bits(Simd::<u32, 4>::from_array([
        0x0000_0000,
        0x8000_0000,
        0x7fc1_2345,
        0xff80_0001,
    ]));
    let native = values.to_ne_bytes();
    let big = values.to_be_bytes();
    let little = values.to_le_bytes();

    assert_eq!(Simd::<f32, 4>::from_ne_bytes(native).to_bits(), values.to_bits());
    assert_eq!(Simd::<f32, 4>::from_be_bytes(big).to_bits(), values.to_bits());
    assert_eq!(Simd::<f32, 4>::from_le_bytes(little).to_bits(), values.to_bits());
}

#[test]
fn float_byte_conversions_preserve_f64_bit_patterns() {
    let values = Simd::from_bits(Simd::<u64, 2>::from_array([
        0x0000_0000_0000_0000,
        0xfff8_1234_5678_9abc,
    ]));
    let native = values.to_ne_bytes();
    let big = values.to_be_bytes();
    let little = values.to_le_bytes();

    assert_eq!(Simd::<f64, 2>::from_ne_bytes(native).to_bits(), values.to_bits());
    assert_eq!(Simd::<f64, 2>::from_be_bytes(big).to_bits(), values.to_bits());
    assert_eq!(Simd::<f64, 2>::from_le_bytes(little).to_bits(), values.to_bits());
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

