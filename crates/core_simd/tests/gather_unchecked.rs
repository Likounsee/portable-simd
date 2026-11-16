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
