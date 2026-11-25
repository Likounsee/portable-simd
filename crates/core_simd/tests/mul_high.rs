#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_mul_high_tests {
    ($module:ident, $ty:ty, $wide:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn mul_high_matches_full_width_product<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => <$ty>::MIN,
                        1 => <$ty>::MAX,
                        2 => -1,
                        _ => <$ty>::MAX / 3,
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => <$ty>::MIN,
                        1 => <$ty>::MAX,
                        2 => -1,
                        _ => <$ty>::MAX / 2,
                    });

                    let actual = Simd::<$ty, LANES>::from_array(lhs)
                        .mul_high(Simd::from_array(rhs))
                        .to_array();
                    let expected = core::array::from_fn(|lane| {
                        let product = (lhs[lane] as $wide) * (rhs[lane] as $wide);
                        (product >> <$ty>::BITS) as $ty
                    });

                    assert_eq!(actual, expected);
                }
            }
        }
    };
}

macro_rules! unsigned_mul_high_tests {
    ($module:ident, $ty:ty, $wide:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn mul_high_matches_full_width_product<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 0,
                        1 => <$ty>::MAX,
                        2 => <$ty>::MAX - 1,
                        _ => <$ty>::MAX / 2,
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => <$ty>::MAX,
                        1 => <$ty>::MAX,
                        2 => 3,
                        _ => <$ty>::MAX / 2,
                    });

                    let actual = Simd::<$ty, LANES>::from_array(lhs)
                        .mul_high(Simd::from_array(rhs))
                        .to_array();
                    let expected = core::array::from_fn(|lane| {
                        let product = (lhs[lane] as $wide) * (rhs[lane] as $wide);
                        (product >> <$ty>::BITS) as $ty
                    });

                    assert_eq!(actual, expected);
                }
            }
        }
    };
}

signed_mul_high_tests! { i8_tests, i8, i16 }
signed_mul_high_tests! { i16_tests, i16, i32 }
signed_mul_high_tests! { i32_tests, i32, i64 }
signed_mul_high_tests! { i64_tests, i64, i128 }
signed_mul_high_tests! { isize_tests, isize, i128 }

unsigned_mul_high_tests! { u8_tests, u8, u16 }
unsigned_mul_high_tests! { u16_tests, u16, u32 }
unsigned_mul_high_tests! { u32_tests, u32, u64 }
unsigned_mul_high_tests! { u64_tests, u64, u128 }
unsigned_mul_high_tests! { usize_tests, usize, u128 }

#[test]
fn i8_mul_high_exhaustively_matches_full_width_product() {
    for lhs in i16::from(i8::MIN)..=i16::from(i8::MAX) {
        let lhs = lhs as i8;
        let lhs_vector = Simd::<i8, 8>::splat(lhs);

        for rhs_start in (0_u16..=u8::MAX as u16).step_by(8) {
            let rhs = core::array::from_fn(|lane| (rhs_start + lane as u16) as u8 as i8);
            let actual = lhs_vector.mul_high(Simd::from_array(rhs)).to_array();
            let expected = rhs.map(|value| (((lhs as i16) * (value as i16)) >> i8::BITS) as i8);

            assert_eq!(actual, expected, "lhs={lhs}, rhs={rhs:?}");
        }
    }
}

#[test]
fn u8_mul_high_exhaustively_matches_full_width_product() {
    for lhs in u8::MIN..=u8::MAX {
        let lhs_vector = Simd::<u8, 8>::splat(lhs);

        for rhs_start in (0_u16..=u8::MAX as u16).step_by(8) {
            let rhs = core::array::from_fn(|lane| (rhs_start + lane as u16) as u8);
            let actual = lhs_vector.mul_high(Simd::from_array(rhs)).to_array();
            let expected = rhs.map(|value| (((lhs as u16) * (value as u16)) >> u8::BITS) as u8);

            assert_eq!(actual, expected, "lhs={lhs}, rhs={rhs:?}");
        }
    }
}
