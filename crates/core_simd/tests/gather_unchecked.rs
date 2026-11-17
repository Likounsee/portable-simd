#![feature(portable_simd)]

use core_simd::simd::Simd;

#[test]
fn gather_unchecked_reads_permuted_and_repeated_indices() {
    let values = [10_i32, 20, 30, 40, 50, 60, 70, 80];
    let indices = Simd::from_array([7_usize, 0, 3, 3, 5, 1, 6, 2]);

    // SAFETY: every lane index is strictly less than values.len().
    let actual = unsafe { Simd::gather_unchecked(&values, indices) };

    assert_eq!(actual.to_array(), [80, 10, 40, 40, 60, 20, 70, 30]);
}

#[test]
fn gather_unchecked_preserves_float_lane_bits() {
    let values = [
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x0000_0001),
        f32::INFINITY,
        f32::from_bits(0x7fc1_2345),
    ];
    let indices = Simd::from_array([3_usize, 1, 0, 2]);

    // SAFETY: all four indices are in bounds for values.
    let actual = unsafe { Simd::gather_unchecked(&values, indices) };

    assert_eq!(
        actual.to_array().map(f32::to_bits),
        [0x7fc1_2345, 0x0000_0001, 0x8000_0000, 0x7f80_0000]
    );
}

#[test]
fn gather_unchecked_supports_other_integer_element_widths() {
    let bytes = [3_u8, 5, 7, 11];
    let byte_indices = Simd::from_array([3_usize, 0, 2, 1]);
    // SAFETY: all indices are below the four-element slice length.
    assert_eq!(
        unsafe { Simd::gather_unchecked(&bytes, byte_indices) }.to_array(),
        [11, 3, 7, 5]
    );

    let words = [13_u64, 17, 19, 23];
    let word_indices = Simd::from_array([2_usize, 3, 0, 1]);
    // SAFETY: all indices are below the four-element slice length.
    let actual = unsafe { Simd::gather_unchecked(&words, word_indices) };
    assert_eq!(actual.to_array(), [19, 23, 13, 17]);
}

#[test]
fn gather_unchecked_preserves_f64_lane_bits() {
    let values = [
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(0x0000_0000_0000_0001),
        f64::INFINITY,
        f64::from_bits(0x7ff8_1234_5678_9abc),
    ];
    let indices = Simd::from_array([3_usize, 1, 0, 2]);
    // SAFETY: all four indices are in bounds for values.
    let actual = unsafe { Simd::gather_unchecked(&values, indices) };
    assert_eq!(
        actual.to_array().map(f64::to_bits),
        [
            0x7ff8_1234_5678_9abc,
            0x0000_0000_0000_0001,
            0x8000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
        ]
    );
}
