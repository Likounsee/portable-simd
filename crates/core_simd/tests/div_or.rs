#![feature(portable_simd)]

use core_simd::simd::{
    Simd,
    num::{SimdInt, SimdUint},
};

macro_rules! signed_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn division_and_remainder_substitute_invalid_lanes<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => <$ty>::MIN,
                        1 => 12,
                        2 => -9,
                        _ => 8,
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => -1,
                        1 => 3,
                        2 => 0,
                        _ => 2,
                    });
                    let fallback = 77 as $ty;
                    let expected_div =
                        core::array::from_fn(|lane| lhs[lane].checked_div(rhs[lane]).unwrap_or(fallback));
                    let expected_rem =
                        core::array::from_fn(|lane| lhs[lane].checked_rem(rhs[lane]).unwrap_or(fallback));

                    let lhs = Simd::<$ty, LANES>::from_array(lhs);
                    let rhs = Simd::<$ty, LANES>::from_array(rhs);
                    let fallback = Simd::splat(fallback);

                    assert_eq!(lhs.div_or(rhs, fallback).to_array(), expected_div);
                    assert_eq!(lhs.rem_or(rhs, fallback).to_array(), expected_rem);
                }
            }
        }
    };
}

macro_rules! unsigned_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn division_and_remainder_substitute_zero_divisors<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => <$ty>::MAX,
                        1 => 12,
                        2 => 7,
                        _ => 8,
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 0,
                        1 => 3,
                        2 => 2,
                        _ => 1,
                    });
                    let fallback = 77 as $ty;
                    let expected_div =
                        core::array::from_fn(|lane| lhs[lane].checked_div(rhs[lane]).unwrap_or(fallback));
                    let expected_rem =
                        core::array::from_fn(|lane| lhs[lane].checked_rem(rhs[lane]).unwrap_or(fallback));

                    let lhs = Simd::<$ty, LANES>::from_array(lhs);
                    let rhs = Simd::<$ty, LANES>::from_array(rhs);
                    let fallback = Simd::splat(fallback);

                    assert_eq!(lhs.div_or(rhs, fallback).to_array(), expected_div);
                    assert_eq!(lhs.rem_or(rhs, fallback).to_array(), expected_rem);
                }
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
