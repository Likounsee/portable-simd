#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

#[test]
fn signed_64_bit_shift_counts_match_scalar_u32_conversion() {
    let value = Simd::from_array([1_i64, -2, 3, i64::MIN]);
    let amount = Simd::from_array([1_i64, 64, 1_i64 << 32, -1_i64]);

    let (left, left_overflow) = value.overflowing_shl(amount);
    let (right, right_overflow) = value.overflowing_shr(amount);

    assert_eq!(left.to_array(), [2, -2, 3, 0]);
    assert_eq!(right.to_array(), [0, -2, 3, -1]);
    assert_eq!(left_overflow.to_array(), [false, true, false, true]);
    assert_eq!(right_overflow.to_array(), [false, true, false, true]);
}

#[test]
fn unsigned_64_bit_shift_counts_match_scalar_u32_conversion() {
    let value = Simd::from_array([1_u64, 2, 3, u64::MAX]);
    let amount = Simd::from_array([1_u64, 64, 1_u64 << 32, u64::MAX]);

    let (left, left_overflow) = value.overflowing_shl(amount);
    let (right, right_overflow) = value.overflowing_shr(amount);

    assert_eq!(left.to_array(), [2, 2, 3, 1_u64 << 63]);
    assert_eq!(right.to_array(), [0, 2, 3, 1]);
    assert_eq!(left_overflow.to_array(), [false, true, false, true]);
    assert_eq!(right_overflow.to_array(), [false, true, false, true]);
}
