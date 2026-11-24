#![feature(portable_simd)]

use core_simd::simd::{
    Simd,
    num::{SimdInt, SimdUint},
};

macro_rules! signed_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn division_and_remainder_substitute_invalid_lanes() {
                let lhs = Simd::<$ty, 4>::from_array([12 as $ty, <$ty>::MIN, -9 as $ty, 8 as $ty]);
                let rhs = Simd::<$ty, 4>::from_array([3 as $ty, -1 as $ty, 0 as $ty, 2 as $ty]);
                let fallback = Simd::splat(77 as $ty);

                assert_eq!(
                    lhs.div_or(rhs, fallback).to_array(),
                    [4 as $ty, 77 as $ty, 77 as $ty, 4 as $ty]
                );
                assert_eq!(
                    lhs.rem_or(rhs, fallback).to_array(),
                    [0 as $ty, 77 as $ty, 77 as $ty, 0 as $ty]
                );
            }
        }
    };
}

macro_rules! unsigned_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn division_and_remainder_substitute_zero_divisors() {
                let lhs = Simd::<$ty, 4>::from_array([12 as $ty, 8 as $ty, 7 as $ty, <$ty>::MAX]);
                let rhs = Simd::<$ty, 4>::from_array([3 as $ty, 0 as $ty, 2 as $ty, 1 as $ty]);
                let fallback = Simd::splat(77 as $ty);

                assert_eq!(
                    lhs.div_or(rhs, fallback).to_array(),
                    [4 as $ty, 77 as $ty, 3 as $ty, <$ty>::MAX]
                );
                assert_eq!(
                    lhs.rem_or(rhs, fallback).to_array(),
                    [0 as $ty, 77 as $ty, 1 as $ty, 0 as $ty]
                );
            }
        }
    };
}

signed_tests!(i8, i8);
signed_tests!(i16, i16);
signed_tests!(i32, i32);
signed_tests!(i64, i64);
signed_tests!(isize, isize);

unsigned_tests!(u8, u8);
unsigned_tests!(u16, u16);
unsigned_tests!(u32, u32);
unsigned_tests!(u64, u64);
unsigned_tests!(usize, usize);
