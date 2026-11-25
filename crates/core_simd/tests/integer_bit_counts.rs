#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_bit_count_tests {
    ($module:ident, $ty:ty, $unsigned:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn bit_counts_match_scalar<const LANES: usize>() {
                    let values: [$ty; LANES] = core::array::from_fn(|lane| {
                        match lane % 8 {
                            0 => 0,
                            1 => -1,
                            2 => <$ty>::MIN,
                            3 => <$ty>::MAX,
                            4 => 1,
                            5 => -2,
                            6 => 0x55 as $ty,
                            _ => !0x55 as $ty,
                        }
                    });

                    let vector = Simd::<$ty, LANES>::from_array(values);
                    let ones = vector.count_ones().to_array();
                    let zeros = vector.count_zeros().to_array();

                    let expected_ones: [$unsigned; LANES] =
                        values.map(|value| value.count_ones() as $unsigned);
                    let expected_zeros: [$unsigned; LANES] =
                        values.map(|value| value.count_zeros() as $unsigned);

                    assert_eq!(ones, expected_ones, "count_ones");
                    assert_eq!(zeros, expected_zeros, "count_zeros");
                }
            }
        }
    };
}

macro_rules! unsigned_bit_count_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn bit_counts_match_scalar<const LANES: usize>() {
                    let values: [$ty; LANES] = core::array::from_fn(|lane| {
                        match lane % 8 {
                            0 => 0,
                            1 => <$ty>::MAX,
                            2 => 1,
                            3 => <$ty>::MAX / 2,
                            4 => 0x55 as $ty,
                            5 => !0x55 as $ty,
                            6 => <$ty>::MIN,
                            _ => 3,
                        }
                    });

                    let vector = Simd::<$ty, LANES>::from_array(values);
                    let ones = vector.count_ones().to_array();
                    let zeros = vector.count_zeros().to_array();

                    let expected_ones = values.map(|value| value.count_ones() as $ty);
                    let expected_zeros = values.map(|value| value.count_zeros() as $ty);

                    assert_eq!(ones, expected_ones, "count_ones");
                    assert_eq!(zeros, expected_zeros, "count_zeros");
                }
            }
        }
    };
}

signed_bit_count_tests!(i8_cases, i8, u8);
signed_bit_count_tests!(i16_cases, i16, u16);
signed_bit_count_tests!(i32_cases, i32, u32);
signed_bit_count_tests!(i64_cases, i64, u64);
signed_bit_count_tests!(isize_cases, isize, usize);

unsigned_bit_count_tests!(u8_cases, u8);
unsigned_bit_count_tests!(u16_cases, u16);
unsigned_bit_count_tests!(u32_cases, u32);
unsigned_bit_count_tests!(u64_cases, u64);
unsigned_bit_count_tests!(usize_cases, usize);
