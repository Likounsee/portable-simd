#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

const EXPONENTS: [u32; 16] = [
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    15,
    31,
    32,
    63,
    127,
    255,
    256,
    u32::MAX,
];

macro_rules! signed_overflowing_pow_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn overflowing_pow_matches_scalar<const LANES: usize>() {
                        let values: [$ty; LANES] = core::array::from_fn(|lane| match lane % 10 {
                            0 => <$ty>::MIN,
                            1 => <$ty>::MAX,
                            2 => -3,
                            3 => -2,
                            4 => -1,
                            5 => 0,
                            6 => 1,
                            7 => 2,
                            8 => 3,
                            _ => 10,
                        });

                        for exponent in EXPONENTS {
                            let vector = Simd::<$ty, LANES>::from_array(values);
                            let wrapping = vector.wrapping_pow(exponent);
                            let expected_wrapping: [$ty; LANES] = core::array::from_fn(|lane| {
                                values[lane].wrapping_pow(exponent)
                            });
                            assert_eq!(wrapping.to_array(), expected_wrapping);

                            let saturating = vector.saturating_pow(exponent);
                            let expected_saturating: [$ty; LANES] = core::array::from_fn(|lane| {
                                values[lane].saturating_pow(exponent)
                            });
                            assert_eq!(saturating.to_array(), expected_saturating);

                            let (actual, actual_overflow) = vector.overflowing_pow(exponent);
                            let expected: [($ty, bool); LANES] = core::array::from_fn(|lane| {
                                values[lane].overflowing_pow(exponent)
                            });

                            assert_eq!(actual.to_array(), expected.map(|item| item.0));
                            assert_eq!(actual_overflow.to_array(), expected.map(|item| item.1));
                        }
                    }
                }
            }
        )*
    };
}

macro_rules! unsigned_overflowing_pow_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn overflowing_pow_matches_scalar<const LANES: usize>() {
                        let values: [$ty; LANES] = core::array::from_fn(|lane| match lane % 7 {
                            0 => 0,
                            1 => 1,
                            2 => 2,
                            3 => 3,
                            4 => 10,
                            5 => <$ty>::MAX / 2,
                            _ => <$ty>::MAX,
                        });

                        for exponent in EXPONENTS {
                            let vector = Simd::<$ty, LANES>::from_array(values);
                            let wrapping = vector.wrapping_pow(exponent);
                            let expected_wrapping: [$ty; LANES] = core::array::from_fn(|lane| {
                                values[lane].wrapping_pow(exponent)
                            });
                            assert_eq!(wrapping.to_array(), expected_wrapping);

                            let saturating = vector.saturating_pow(exponent);
                            let expected_saturating: [$ty; LANES] = core::array::from_fn(|lane| {
                                values[lane].saturating_pow(exponent)
                            });
                            assert_eq!(saturating.to_array(), expected_saturating);

                            let (actual, actual_overflow) = vector.overflowing_pow(exponent);
                            let expected: [($ty, bool); LANES] = core::array::from_fn(|lane| {
                                values[lane].overflowing_pow(exponent)
                            });

                            assert_eq!(actual.to_array(), expected.map(|item| item.0));
                            assert_eq!(actual_overflow.to_array(), expected.map(|item| item.1));
                        }
                    }
                }
            }
        )*
    };
}

signed_overflowing_pow_tests! {
    i8_cases: i8,
    i16_cases: i16,
    i32_cases: i32,
    i64_cases: i64,
    isize_cases: isize,
}

unsigned_overflowing_pow_tests! {
    u8_cases: u8,
    u16_cases: u16,
    u32_cases: u32,
    u64_cases: u64,
    usize_cases: usize,
}
