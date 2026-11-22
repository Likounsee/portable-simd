#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_shift_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;
            test_helpers::test_lanes! {
                fn unchecked_shifts<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|lane| {
                        if lane % 2 == 0 { -4 } else { 3 }
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|lane| (lane % 2) as $ty);
                    let value = Simd::<$ty, LANES>::from_array(lhs);
                    let amount = Simd::<$ty, LANES>::from_array(rhs);
                    // SAFETY: Every shift amount is either zero or one.
                    let left = unsafe { value.unchecked_shl(amount) };
                    // SAFETY: Every shift amount is either zero or one.
                    let right = unsafe { value.unchecked_shr(amount) };
                    let expected_left = core::array::from_fn(|lane| {
                        // SAFETY: Every shift amount is either zero or one.
                        unsafe { lhs[lane].unchecked_shl(rhs[lane] as u32) }
                    });
                    let expected_right = core::array::from_fn(|lane| {
                        // SAFETY: Every shift amount is either zero or one.
                        unsafe { lhs[lane].unchecked_shr(rhs[lane] as u32) }
                    });
                    assert_eq!(left.to_array(), expected_left);
                    assert_eq!(right.to_array(), expected_right);
                }
            }
        }
    };
}

macro_rules! unsigned_shift_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;
            test_helpers::test_lanes! {
                fn unchecked_shifts<const LANES: usize>() {
                    let lhs: [$ty; LANES] =
                        core::array::from_fn(|lane| (lane as $ty).wrapping_add(4));
                    let rhs: [$ty; LANES] = core::array::from_fn(|lane| (lane % 2) as $ty);
                    let value = Simd::<$ty, LANES>::from_array(lhs);
                    let amount = Simd::<$ty, LANES>::from_array(rhs);
                    // SAFETY: Every shift amount is either zero or one.
                    let left = unsafe { value.unchecked_shl(amount) };
                    // SAFETY: Every shift amount is either zero or one.
                    let right = unsafe { value.unchecked_shr(amount) };
                    let expected_left = core::array::from_fn(|lane| {
                        // SAFETY: Every shift amount is either zero or one.
                        unsafe { lhs[lane].unchecked_shl(rhs[lane] as u32) }
                    });
                    let expected_right = core::array::from_fn(|lane| {
                        // SAFETY: Every shift amount is either zero or one.
                        unsafe { lhs[lane].unchecked_shr(rhs[lane] as u32) }
                    });
                    assert_eq!(left.to_array(), expected_left);
                    assert_eq!(right.to_array(), expected_right);
                }
            }
        }
    };
}

signed_shift_tests! { i8_tests, i8 }
signed_shift_tests! { i16_tests, i16 }
signed_shift_tests! { i32_tests, i32 }
signed_shift_tests! { i64_tests, i64 }
signed_shift_tests! { isize_tests, isize }

unsigned_shift_tests! { u8_tests, u8 }
unsigned_shift_tests! { u16_tests, u16 }
unsigned_shift_tests! { u32_tests, u32 }
unsigned_shift_tests! { u64_tests, u64 }
unsigned_shift_tests! { usize_tests, usize }

#[test]
fn signed_unchecked_shifts_use_u32_converted_counts() {
    let value = Simd::from_array([1_i64, -2, 3, 4]);
    let amount = Simd::from_array([
        (1_i64 << 32) + 1,
        (1_i64 << 32) + 2,
        (1_i64 << 32) + 3,
        (1_i64 << 32) + 4,
    ]);

    // SAFETY: After conversion to u32, the shift amounts are 1, 2, 3, and 4,
    // all of which are less than the 64-bit lane width.
    let left = unsafe { value.unchecked_shl(amount) };
    // SAFETY: After conversion to u32, the shift amounts are 1, 2, 3, and 4,
    // all of which are less than the 64-bit lane width.
    let right = unsafe { value.unchecked_shr(amount) };

    assert_eq!(left.to_array(), [2, -8, 24, 64]);
    assert_eq!(right.to_array(), [0, -1, 0, 0]);
}
