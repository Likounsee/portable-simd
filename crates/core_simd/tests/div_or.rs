#![feature(portable_simd)]

use core_simd::simd::{
    Simd,
    num::{SimdInt, SimdUint},
};

macro_rules! signed_fallback_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn invalid_division_and_remainder_use_per_lane_fallbacks() {
                let lhs = Simd::<$ty, 4>::from_array([<$ty>::MIN, 12, -9, 7]);
                let rhs = Simd::<$ty, 4>::from_array([-1, 0, 2, 0]);
                let fallback = Simd::<$ty, 4>::from_array([91, 92, 93, 94]);

                assert_eq!(lhs.div_or(rhs, fallback).to_array(), [91, 92, -4, 94]);
                assert_eq!(lhs.rem_or(rhs, fallback).to_array(), [91, 92, -1, 94]);

                let all_zero = Simd::splat(0);
                assert_eq!(lhs.div_or(all_zero, fallback), fallback);
                assert_eq!(lhs.rem_or(all_zero, fallback), fallback);

                let all_overflow = Simd::splat(-1);
                let min = Simd::splat(<$ty>::MIN);
                assert_eq!(min.div_or(all_overflow, fallback), fallback);
                assert_eq!(min.rem_or(all_overflow, fallback), fallback);
            }
        }
    };
}

macro_rules! unsigned_fallback_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn zero_divisors_use_per_lane_fallbacks() {
                let lhs = Simd::<$ty, 4>::from_array([<$ty>::MAX, 12, 9, 7]);
                let rhs = Simd::<$ty, 4>::from_array([0, 3, 2, 0]);
                let fallback = Simd::<$ty, 4>::from_array([91, 92, 93, 94]);

                assert_eq!(lhs.div_or(rhs, fallback).to_array(), [91, 4, 4, 94]);
                assert_eq!(lhs.rem_or(rhs, fallback).to_array(), [91, 0, 1, 94]);

                let all_zero = Simd::splat(0);
                assert_eq!(lhs.div_or(all_zero, fallback), fallback);
                assert_eq!(lhs.rem_or(all_zero, fallback), fallback);
            }
        }
    };
}

signed_fallback_tests! { i8_tests, i8 }
signed_fallback_tests! { i16_tests, i16 }
signed_fallback_tests! { i32_tests, i32 }
signed_fallback_tests! { i64_tests, i64 }
signed_fallback_tests! { isize_tests, isize }

unsigned_fallback_tests! { u8_tests, u8 }
unsigned_fallback_tests! { u16_tests, u16 }
unsigned_fallback_tests! { u32_tests, u32 }
unsigned_fallback_tests! { u64_tests, u64 }
unsigned_fallback_tests! { usize_tests, usize }
