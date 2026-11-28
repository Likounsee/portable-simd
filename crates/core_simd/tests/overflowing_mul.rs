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


// Exercise a broad deterministic distribution, not just hand-picked overflow
// boundaries. This covers both widened vector implementations and the scalar
// 64-bit fallback for every supported lane count.
macro_rules! randomized_signed_overflowing_mul_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn randomized_overflowing_mul_matches_scalar<const LANES: usize>() {
                    let mut lhs_state = 0x9e37_79b9_7f4a_7c15_u64;
                    let mut rhs_state = 0xd1b5_4a32_d192_ed03_u64;
                    let lhs: [$ty; LANES] = core::array::from_fn(|_| {
                        lhs_state = lhs_state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        lhs_state as $ty
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|_| {
                        rhs_state = rhs_state
                            .wrapping_mul(2_862_933_555_777_941_757)
                            .wrapping_add(3_037_000_493);
                        rhs_state as $ty
                    });

                    let left = Simd::<$ty, LANES>::from_array(lhs);
                    let right = Simd::<$ty, LANES>::from_array(rhs);
                    let (actual, actual_overflow) = left.overflowing_mul(right);
                    let actual_saturating = left.saturating_mul(right).to_array();
                    let expected: [($ty, bool); LANES] = core::array::from_fn(|lane| {
                        lhs[lane].overflowing_mul(rhs[lane])
                    });
                    let expected_saturating: [$ty; LANES] = core::array::from_fn(|lane| {
                        lhs[lane].saturating_mul(rhs[lane])
                    });

                    assert_eq!(actual.to_array(), expected.map(|item| item.0));
                    assert_eq!(actual_overflow.to_array(), expected.map(|item| item.1));
                    assert_eq!(actual_saturating, expected_saturating);
                }
            }
        }
    };
}

macro_rules! randomized_unsigned_overflowing_mul_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn randomized_overflowing_mul_matches_scalar<const LANES: usize>() {
                    let mut lhs_state = 0x9e37_79b9_7f4a_7c15_u64;
                    let mut rhs_state = 0xd1b5_4a32_d192_ed03_u64;
                    let lhs: [$ty; LANES] = core::array::from_fn(|_| {
                        lhs_state = lhs_state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        lhs_state as $ty
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|_| {
                        rhs_state = rhs_state
                            .wrapping_mul(2_862_933_555_777_941_757)
                            .wrapping_add(3_037_000_493);
                        rhs_state as $ty
                    });

                    let left = Simd::<$ty, LANES>::from_array(lhs);
                    let right = Simd::<$ty, LANES>::from_array(rhs);
                    let (actual, actual_overflow) = left.overflowing_mul(right);
                    let actual_saturating = left.saturating_mul(right).to_array();
                    let expected: [($ty, bool); LANES] = core::array::from_fn(|lane| {
                        lhs[lane].overflowing_mul(rhs[lane])
                    });
                    let expected_saturating: [$ty; LANES] = core::array::from_fn(|lane| {
                        lhs[lane].saturating_mul(rhs[lane])
                    });

                    assert_eq!(actual.to_array(), expected.map(|item| item.0));
                    assert_eq!(actual_overflow.to_array(), expected.map(|item| item.1));
                    assert_eq!(actual_saturating, expected_saturating);
                }
            }
        }
    };
}

randomized_signed_overflowing_mul_tests!(i8_randomized, i8);
randomized_signed_overflowing_mul_tests!(i16_randomized, i16);
randomized_signed_overflowing_mul_tests!(i32_randomized, i32);
randomized_signed_overflowing_mul_tests!(i64_randomized, i64);
randomized_signed_overflowing_mul_tests!(isize_randomized, isize);

randomized_unsigned_overflowing_mul_tests!(u8_randomized, u8);
randomized_unsigned_overflowing_mul_tests!(u16_randomized, u16);
randomized_unsigned_overflowing_mul_tests!(u32_randomized, u32);
randomized_unsigned_overflowing_mul_tests!(u64_randomized, u64);
randomized_unsigned_overflowing_mul_tests!(usize_randomized, usize);
