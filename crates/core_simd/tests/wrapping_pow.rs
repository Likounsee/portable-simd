#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_wrapping_pow_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn wrapping_pow_matches_scalar<const LANES: usize>() {
                    let values: [$ty; LANES] = core::array::from_fn(|lane| {
                        match lane % 7 {
                            0 => <$ty>::MIN,
                            1 => -3,
                            2 => -2,
                            3 => -1,
                            4 => 0,
                            5 => 2,
                            _ => <$ty>::MAX,
                        }
                    });
                    let vector = Simd::<$ty, LANES>::from_array(values);

                    for exponent in [0, 1, 2, 3, 5, 17, 31] {
                        let actual = vector.wrapping_pow(exponent).to_array();
                        let expected =
                            values.map(|value| value.wrapping_pow(exponent));
                        assert_eq!(actual, expected, "wrapping exponent {exponent}");

                        let actual = vector.saturating_pow(exponent).to_array();
                        let expected =
                            values.map(|value| value.saturating_pow(exponent));
                        assert_eq!(actual, expected, "saturating exponent {exponent}");

                        let actual = vector.overflowing_pow(exponent);
                        let expected = values.map(|value| value.overflowing_pow(exponent));
                        assert_eq!(
                            actual.0.to_array(),
                            expected.map(|(value, _)| value),
                            "overflowing value for exponent {exponent}"
                        );
                        assert_eq!(
                            actual.1.to_array(),
                            expected.map(|(_, overflow)| overflow),
                            "overflowing mask for exponent {exponent}"
                        );
                    }
                }
            }
        }
    };
}

macro_rules! unsigned_wrapping_pow_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn wrapping_pow_matches_scalar<const LANES: usize>() {
                    let values: [$ty; LANES] = core::array::from_fn(|lane| {
                        match lane % 6 {
                            0 => 0,
                            1 => 1,
                            2 => 2,
                            3 => 3,
                            4 => <$ty>::MAX,
                            _ => <$ty>::MAX - 1,
                        }
                    });
                    let vector = Simd::<$ty, LANES>::from_array(values);

                    for exponent in [0, 1, 2, 3, 5, 17, 31] {
                        let actual = vector.wrapping_pow(exponent).to_array();
                        let expected =
                            values.map(|value| value.wrapping_pow(exponent));
                        assert_eq!(actual, expected, "exponent {exponent}");
                    }
                }
            }
        }
    };
}

signed_wrapping_pow_tests! { i8_tests, i8 }
signed_wrapping_pow_tests! { i16_tests, i16 }
signed_wrapping_pow_tests! { i32_tests, i32 }
signed_wrapping_pow_tests! { i64_tests, i64 }
signed_wrapping_pow_tests! { isize_tests, isize }

unsigned_wrapping_pow_tests! { u8_tests, u8 }
unsigned_wrapping_pow_tests! { u16_tests, u16 }
unsigned_wrapping_pow_tests! { u32_tests, u32 }
unsigned_wrapping_pow_tests! { u64_tests, u64 }
unsigned_wrapping_pow_tests! { usize_tests, usize }
