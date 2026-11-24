#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_saturating_mul_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn saturating_mul_matches_scalar<const LANES: usize>() {
                        let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                            0 => <$ty>::MIN,
                            1 => <$ty>::MAX,
                            2 => -1,
                            _ => <$ty>::MAX / 2,
                        });
                        let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                            0 => -1,
                            1 => 2,
                            2 => <$ty>::MIN,
                            _ => 3,
                        });

                        let actual = Simd::<$ty, LANES>::from_array(lhs)
                            .saturating_mul(Simd::from_array(rhs))
                            .to_array();
                        let expected = core::array::from_fn(|lane| {
                            lhs[lane].saturating_mul(rhs[lane])
                        });

                        assert_eq!(actual, expected);
                    }
                }
            }
        )*
    };
}

macro_rules! unsigned_saturating_mul_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn saturating_mul_matches_scalar<const LANES: usize>() {
                        let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                            0 => 0,
                            1 => <$ty>::MAX,
                            2 => <$ty>::MAX / 2,
                            _ => 3,
                        });
                        let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                            0 => <$ty>::MAX,
                            1 => 2,
                            2 => 3,
                            _ => <$ty>::MAX,
                        });

                        let actual = Simd::<$ty, LANES>::from_array(lhs)
                            .saturating_mul(Simd::from_array(rhs))
                            .to_array();
                        let expected = core::array::from_fn(|lane| {
                            lhs[lane].saturating_mul(rhs[lane])
                        });

                        assert_eq!(actual, expected);
                    }
                }
            }
        )*
    };
}

signed_saturating_mul_tests! {
    i8_cases: i8,
    i16_cases: i16,
    i32_cases: i32,
    i64_cases: i64,
    isize_cases: isize,
}

unsigned_saturating_mul_tests! {
    u8_cases: u8,
    u16_cases: u16,
    u32_cases: u32,
    u64_cases: u64,
    usize_cases: usize,
}
