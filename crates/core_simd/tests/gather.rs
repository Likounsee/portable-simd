#![feature(portable_simd)]

use core_simd::simd::Simd;

#[test]
fn gather_ptr_reads_every_lane() {
    let values = [10_i32, 20, 30, 40];
    let base = values.as_ptr();
    let pointers = Simd::from_array([
        base.wrapping_add(3),
        base,
        base.wrapping_add(2),
        base.wrapping_add(1),
    ]);

    // SAFETY: every pointer targets an initialized element of values.
    let actual = unsafe { Simd::gather_ptr(pointers) };
    assert_eq!(actual, Simd::from_array([40, 10, 30, 20]));
}

#[test]
fn gather_ptr_supports_single_lane() {
    let values = [42_i32];
    let pointers = Simd::from_array([values.as_ptr()]);

    // SAFETY: the only pointer targets the initialized element of values.
    let actual = unsafe { Simd::gather_ptr(pointers) };
    assert_eq!(actual, Simd::from_array([42]));
}

#[test]
fn gather_ptr_preserves_float_lane_bits() {
    let values = [
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x0000_0001),
        f32::INFINITY,
        f32::from_bits(0x7fc1_2345),
    ];
    let base = values.as_ptr();
    let pointers = Simd::from_array([
        base.wrapping_add(3),
        base.wrapping_add(1),
        base,
        base.wrapping_add(2),
    ]);

    // SAFETY: every pointer targets an initialized element of values.
    let actual = unsafe { Simd::gather_ptr(pointers) };
    assert_eq!(
        actual.to_array().map(f32::to_bits),
        [0x7fc1_2345, 0x0000_0001, 0x8000_0000, 0x7f80_0000]
    );
}
