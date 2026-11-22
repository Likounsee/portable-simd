#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

#[test]
fn signed_64_bit_shift_counts_keep_high_bits_for_overflow() {
    let value = Simd::from_array([1_i64, -2, 3, i64::MIN]);
    let amount = Simd::from_array([1_i64, 64, 1_i64 << 32, -(1_i64 << 32)]);

    let (left, left_overflow) = value.overflowing_shl(amount);
    let (right, right_overflow) = value.overflowing_shr(amount);

    assert_eq!(left.to_array(), [2, -2, 3, i64::MIN]);
    assert_eq!(right.to_array(), [0, -2, 3, i64::MIN]);
    assert_eq!(left_overflow.to_array(), [false, true, true, true]);
    assert_eq!(right_overflow.to_array(), [false, true, true, true]);
}

#[test]
fn unsigned_64_bit_shift_counts_keep_high_bits_for_overflow() {
    let value = Simd::from_array([1_u64, 2, 3, u64::MAX]);
    let amount = Simd::from_array([1_u64, 64, 1_u64 << 32, u64::MAX]);

    let (left, left_overflow) = value.overflowing_shl(amount);
    let (right, right_overflow) = value.overflowing_shr(amount);

    assert_eq!(left.to_array(), [2, 2, 3, 1_u64 << 63]);
    assert_eq!(right.to_array(), [0, 2, 3, 1]);
    assert_eq!(left_overflow.to_array(), [false, true, true, true]);
    assert_eq!(right_overflow.to_array(), [false, true, true, true]);
}
