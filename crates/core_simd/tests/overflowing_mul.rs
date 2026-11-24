#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_overflowing_mul_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn overflowing_mul_matches_scalar<const LANES: usize>() {
                        let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 6 {
                            0 => <$ty>::MIN,
                            1 => <$ty>::MAX,
                            2 => -1,
                            3 => 0,
                            4 => <$ty>::MAX / 2,
                            _ => 3,
                        });
                        let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 6 {
                            0 => -1,
                            1 => 2,
                            2 => <$ty>::MIN,
                            3 => <$ty>::MAX,
                            4 => 3,
                            _ => -7,
                        });

                        let (values, overflow) = Simd::<$ty, LANES>::from_array(lhs)
                            .overflowing_mul(Simd::from_array(rhs));
                        let expected: [($ty, bool); LANES] = core::array::from_fn(|lane| {
                            lhs[lane].overflowing_mul(rhs[lane])
                        });
                        assert_eq!(values.to_array(), expected.map(|item| item.0));
                        assert_eq!(overflow.to_array(), expected.map(|item| item.1));
                    }
                }
            }
        )*
    };
}

macro_rules! unsigned_overflowing_mul_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn overflowing_mul_matches_scalar<const LANES: usize>() {
                        let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 5 {
                            0 => 0,
                            1 => <$ty>::MAX,
                            2 => <$ty>::MAX / 2,
                            3 => 3,
                            _ => 17,
                        });
                        let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 5 {
                            0 => <$ty>::MAX,
                            1 => 2,
                            2 => 3,
                            3 => <$ty>::MAX,
                            _ => 19,
                        });

                        let (values, overflow) = Simd::<$ty, LANES>::from_array(lhs)
                            .overflowing_mul(Simd::from_array(rhs));
                        let expected: [($ty, bool); LANES] = core::array::from_fn(|lane| {
                            lhs[lane].overflowing_mul(rhs[lane])
                        });
                        assert_eq!(values.to_array(), expected.map(|item| item.0));
                        assert_eq!(overflow.to_array(), expected.map(|item| item.1));
                    }
                }
            }
        )*
    };
}

signed_overflowing_mul_tests! {
    i8_cases: i8,
    i16_cases: i16,
    i32_cases: i32,
    i64_cases: i64,
    isize_cases: isize,
}

unsigned_overflowing_mul_tests! {
    u8_cases: u8,
    u16_cases: u16,
    u32_cases: u32,
    u64_cases: u64,
    usize_cases: usize,
}
