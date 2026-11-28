#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

const EXPONENTS: [u32; 12] = [0, 1, 2, 3, 7, 15, 31, 32, 63, 64, 127, u32::MAX];

macro_rules! signed_saturating_pow_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn saturating_pow_matches_scalar_at_boundaries<const LANES: usize>() {
                        let values: [$ty; LANES] = core::array::from_fn(|lane| match lane % 8 {
                            0 => <$ty>::MIN,
                            1 => <$ty>::MIN + 1,
                            2 => -2,
                            3 => -1,
                            4 => 0,
                            5 => 1,
                            6 => 2,
                            _ => <$ty>::MAX,
                        });

                        for exponent in EXPONENTS {
                            let actual = Simd::<$ty, LANES>::from_array(values)
                                .saturating_pow(exponent)
                                .to_array();
                            let expected = values.map(|value| value.saturating_pow(exponent));
                            assert_eq!(
                                actual, expected,
                                "saturating_pow mismatch for exponent {exponent}"
                            );
                        }
                    }
                }
            }
        )*
    };
}

macro_rules! unsigned_saturating_pow_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn saturating_pow_matches_scalar_at_boundaries<const LANES: usize>() {
                        let values: [$ty; LANES] = core::array::from_fn(|lane| match lane % 6 {
                            0 => 0,
                            1 => 1,
                            2 => 2,
                            3 => 3,
                            4 => <$ty>::MAX / 2,
                            _ => <$ty>::MAX,
                        });

                        for exponent in EXPONENTS {
                            let actual = Simd::<$ty, LANES>::from_array(values)
                                .saturating_pow(exponent)
                                .to_array();
                            let expected = values.map(|value| value.saturating_pow(exponent));
                            assert_eq!(
                                actual, expected,
                                "saturating_pow mismatch for exponent {exponent}"
                            );
                        }
                    }
                }
            }
        )*
    };
}

signed_saturating_pow_tests! {
    i8_cases: i8,
    i16_cases: i16,
    i32_cases: i32,
    i64_cases: i64,
    isize_cases: isize,
}

unsigned_saturating_pow_tests! {
    u8_cases: u8,
    u16_cases: u16,
    u32_cases: u32,
    u64_cases: u64,
    usize_cases: usize,
}
