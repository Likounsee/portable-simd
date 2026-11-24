#![feature(portable_simd)]

// Regression coverage for the horizontal min/max index operations added for
// https://github.com/rust-lang/portable-simd/issues/211.

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_index_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn min_max_index_returns_first_matching_lane<const LANES: usize>() {
                    let values: [$ty; LANES] = core::array::from_fn(|lane| {
                        match lane % 4 {
                            0 => <$ty>::MAX,
                            1 => <$ty>::MIN,
                            2 => 0,
                            _ => <$ty>::MIN,
                        }
                    });

                    let min = *values.iter().min().unwrap();
                    let max = *values.iter().max().unwrap();
                    let expected_min = values.iter().position(|&value| value == min).unwrap();
                    let expected_max = values.iter().position(|&value| value == max).unwrap();

                    let vector = Simd::<$ty, LANES>::from_array(values);
                    assert_eq!(vector.reduce_min_index(), expected_min);
                    assert_eq!(vector.reduce_max_index(), expected_max);
                }
            }
        }
    };
}

macro_rules! unsigned_index_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn min_max_index_returns_first_matching_lane<const LANES: usize>() {
                    let values: [$ty; LANES] = core::array::from_fn(|lane| {
                        match lane % 4 {
                            0 => <$ty>::MAX,
                            1 => 0,
                            2 => <$ty>::MAX / 2,
                            _ => 0,
                        }
                    });

                    let min = *values.iter().min().unwrap();
                    let max = *values.iter().max().unwrap();
                    let expected_min = values.iter().position(|&value| value == min).unwrap();
                    let expected_max = values.iter().position(|&value| value == max).unwrap();

                    let vector = Simd::<$ty, LANES>::from_array(values);
                    assert_eq!(vector.reduce_min_index(), expected_min);
                    assert_eq!(vector.reduce_max_index(), expected_max);
                }
            }
        }
    };
}

signed_index_tests! { i8_tests, i8 }
signed_index_tests! { i16_tests, i16 }
signed_index_tests! { i32_tests, i32 }
signed_index_tests! { i64_tests, i64 }
signed_index_tests! { isize_tests, isize }

unsigned_index_tests! { u8_tests, u8 }
unsigned_index_tests! { u16_tests, u16 }
unsigned_index_tests! { u32_tests, u32 }
unsigned_index_tests! { u64_tests, u64 }
unsigned_index_tests! { usize_tests, usize }
