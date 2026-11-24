#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_pow_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn pow_matches_scalar<const LANES: usize>() {
                        let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 8 {
                            0 => <$ty>::MIN,
                            1 => <$ty>::MIN + 1,
                            2 => -3,
                            3 => -2,
                            4 => -1,
                            5 => 0,
                            6 => 1,
                            _ => <$ty>::MAX,
                        });
                        let vector = Simd::<$ty, LANES>::from_array(lhs);

                        for exp in [0, 1, 2, 3, 5, 8, 31, 32, u32::MAX] {
                            let wrapping = vector.wrapping_pow(exp).to_array();
                            let wrapping_expected = core::array::from_fn(|lane| {
                                lhs[lane].wrapping_pow(exp)
                            });
                            assert_eq!(wrapping, wrapping_expected, "wrapping_pow exponent {exp}");

                            let saturating = vector.saturating_pow(exp).to_array();
                            let saturating_expected = core::array::from_fn(|lane| {
                                lhs[lane].saturating_pow(exp)
                            });
                            assert_eq!(saturating, saturating_expected, "saturating_pow exponent {exp}");
                        }
                    }
                }
            }
        )*
    };
}

macro_rules! unsigned_pow_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn pow_matches_scalar<const LANES: usize>() {
                        let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 6 {
                            0 => 0,
                            1 => 1,
                            2 => 2,
                            3 => 3,
                            4 => <$ty>::MAX / 2,
                            _ => <$ty>::MAX,
                        });
                        let vector = Simd::<$ty, LANES>::from_array(lhs);

                        for exp in [0, 1, 2, 3, 5, 8, 31, 32, u32::MAX] {
                            let wrapping = vector.wrapping_pow(exp).to_array();
                            let wrapping_expected = core::array::from_fn(|lane| {
                                lhs[lane].wrapping_pow(exp)
                            });
                            assert_eq!(wrapping, wrapping_expected, "wrapping_pow exponent {exp}");

                            let saturating = vector.saturating_pow(exp).to_array();
                            let saturating_expected = core::array::from_fn(|lane| {
                                lhs[lane].saturating_pow(exp)
                            });
                            assert_eq!(saturating, saturating_expected, "saturating_pow exponent {exp}");
                        }
                    }
                }
            }
        )*
    };
}

signed_pow_tests! {
    i8_cases: i8,
    i16_cases: i16,
    i32_cases: i32,
    i64_cases: i64,
    isize_cases: isize,
}

unsigned_pow_tests! {
    u8_cases: u8,
    u16_cases: u16,
    u32_cases: u32,
    u64_cases: u64,
    usize_cases: usize,
}
