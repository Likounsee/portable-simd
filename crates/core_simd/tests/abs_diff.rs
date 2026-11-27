#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_abs_diff_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {$(
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn abs_diff_matches_scalar_at_signed_boundaries<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 6 {
                        0 => <$ty>::MIN,
                        1 => <$ty>::MAX,
                        2 => -1,
                        3 => 0,
                        4 => <$ty>::MIN / 2,
                        _ => <$ty>::MAX / 2,
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 6 {
                        0 => <$ty>::MAX,
                        1 => <$ty>::MIN,
                        2 => 1,
                        3 => -1,
                        4 => <$ty>::MAX,
                        _ => <$ty>::MIN,
                    });

                    let actual = Simd::<$ty, LANES>::from_array(lhs)
                        .abs_diff(Simd::from_array(rhs))
                        .to_array();
                    let expected = core::array::from_fn(|lane| lhs[lane].abs_diff(rhs[lane]));

                    assert_eq!(actual, expected);
                }
            }
        }
    )*};
}

macro_rules! unsigned_abs_diff_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {$(
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn abs_diff_matches_scalar_at_unsigned_boundaries<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 5 {
                        0 => 0,
                        1 => <$ty>::MAX,
                        2 => <$ty>::MAX / 2,
                        3 => 1,
                        _ => <$ty>::MAX - 1,
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 5 {
                        0 => <$ty>::MAX,
                        1 => 0,
                        2 => <$ty>::MAX,
                        3 => <$ty>::MAX,
                        _ => 1,
                    });

                    let actual = Simd::<$ty, LANES>::from_array(lhs)
                        .abs_diff(Simd::from_array(rhs))
                        .to_array();
                    let expected = core::array::from_fn(|lane| lhs[lane].abs_diff(rhs[lane]));

                    assert_eq!(actual, expected);
                }
            }
        }
    )*};
}

signed_abs_diff_tests! {
    i8_cases: i8,
    i16_cases: i16,
    i32_cases: i32,
    i64_cases: i64,
    isize_cases: isize,
}

unsigned_abs_diff_tests! {
    u8_cases: u8,
    u16_cases: u16,
    u32_cases: u32,
    u64_cases: u64,
    usize_cases: usize,
}
