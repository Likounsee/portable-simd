#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {$(
        mod $module {
            use super::*;
            test_helpers::test_lanes! {
                fn overflowing_add_sub_matches_scalar<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|i| match i % 8 {
                        0 => <$ty>::MAX,
                        1 => <$ty>::MIN,
                        2 => <$ty>::MAX / 2,
                        3 => -1,
                        4 => <$ty>::MAX,
                        5 => <$ty>::MIN,
                        6 => 1,
                        _ => <$ty>::MIN,
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|i| match i % 8 {
                        0 => 1,
                        1 => -1,
                        2 => <$ty>::MAX / 2,
                        3 => <$ty>::MIN,
                        4 => -1,
                        5 => 1,
                        6 => <$ty>::MIN,
                        _ => -1,
                    });
                    let (add, add_mask) = Simd::<$ty, LANES>::from_array(lhs)
                        .overflowing_add(Simd::from_array(rhs));
                    let (sub, sub_mask) = Simd::<$ty, LANES>::from_array(lhs)
                        .overflowing_sub(Simd::from_array(rhs));
                    let expected_add: [($ty, bool); LANES] =
                        core::array::from_fn(|i| lhs[i].overflowing_add(rhs[i]));
                    let expected_sub: [($ty, bool); LANES] =
                        core::array::from_fn(|i| lhs[i].overflowing_sub(rhs[i]));
                    assert_eq!(add.to_array(), expected_add.map(|x| x.0));
                    assert_eq!(add_mask.to_array(), expected_add.map(|x| x.1));
                    assert_eq!(sub.to_array(), expected_sub.map(|x| x.0));
                    assert_eq!(sub_mask.to_array(), expected_sub.map(|x| x.1));
                    let (neg, neg_mask) = Simd::<$ty, LANES>::from_array(lhs).overflowing_neg();
                    let (abs, abs_mask) = Simd::<$ty, LANES>::from_array(lhs).overflowing_abs();
                    let expected_neg: [($ty, bool); LANES] =
                        core::array::from_fn(|i| lhs[i].overflowing_neg());
                    let expected_abs: [($ty, bool); LANES] =
                        core::array::from_fn(|i| lhs[i].overflowing_abs());
                    assert_eq!(neg.to_array(), expected_neg.map(|x| x.0));
                    assert_eq!(neg_mask.to_array(), expected_neg.map(|x| x.1));
                    assert_eq!(abs.to_array(), expected_abs.map(|x| x.0));
                    assert_eq!(abs_mask.to_array(), expected_abs.map(|x| x.1));
                }
            }
        }
    )*};
}

macro_rules! unsigned_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {$(
        mod $module {
            use super::*;
            test_helpers::test_lanes! {
                fn overflowing_add_sub_matches_scalar<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|i| match i % 4 {
                        0 => <$ty>::MAX,
                        1 => 0,
                        2 => <$ty>::MAX / 2,
                        _ => 3,
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|i| match i % 4 {
                        0 => 1,
                        1 => 1,
                        2 => <$ty>::MAX,
                        _ => <$ty>::MAX,
                    });
                    let (add, add_mask) = Simd::<$ty, LANES>::from_array(lhs)
                        .overflowing_add(Simd::from_array(rhs));
                    let (sub, sub_mask) = Simd::<$ty, LANES>::from_array(lhs)
                        .overflowing_sub(Simd::from_array(rhs));
                    let expected_add: [($ty, bool); LANES] =
                        core::array::from_fn(|i| lhs[i].overflowing_add(rhs[i]));
                    let expected_sub: [($ty, bool); LANES] =
                        core::array::from_fn(|i| lhs[i].overflowing_sub(rhs[i]));
                    assert_eq!(add.to_array(), expected_add.map(|x| x.0));
                    assert_eq!(add_mask.to_array(), expected_add.map(|x| x.1));
                    assert_eq!(sub.to_array(), expected_sub.map(|x| x.0));
                    assert_eq!(sub_mask.to_array(), expected_sub.map(|x| x.1));
                }
            }
        }
    )*};
}

signed_tests! {
    i8_cases: i8, i16_cases: i16, i32_cases: i32, i64_cases: i64, isize_cases: isize,
}
unsigned_tests! {
    u8_cases: u8, u16_cases: u16, u32_cases: u32, u64_cases: u64, usize_cases: usize,
}

// The i64/isize implementation uses sign-bit identities rather than widening
// to a larger SIMD lane. Exercise a broad deterministic set of bit patterns
// at every supported lane count against the scalar primitive reference.
macro_rules! randomized_signed_overflow_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn randomized_overflowing_add_sub<const LANES: usize>() {
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

                    let (actual_add, actual_add_overflow) =
                        Simd::<$ty, LANES>::from_array(lhs)
                            .overflowing_add(Simd::from_array(rhs));
                    let (actual_sub, actual_sub_overflow) =
                        Simd::<$ty, LANES>::from_array(lhs)
                            .overflowing_sub(Simd::from_array(rhs));
                    let expected_add: [($ty, bool); LANES] =
                        core::array::from_fn(|lane| lhs[lane].overflowing_add(rhs[lane]));
                    let expected_sub: [($ty, bool); LANES] =
                        core::array::from_fn(|lane| lhs[lane].overflowing_sub(rhs[lane]));

                    assert_eq!(actual_add.to_array(), expected_add.map(|item| item.0));
                    assert_eq!(actual_add_overflow.to_array(), expected_add.map(|item| item.1));
                    assert_eq!(actual_sub.to_array(), expected_sub.map(|item| item.0));
                    assert_eq!(actual_sub_overflow.to_array(), expected_sub.map(|item| item.1));
                }
            }
        }
    };
}

randomized_signed_overflow_tests!(i64_randomized, i64);
randomized_signed_overflow_tests!(isize_randomized, isize);

// Use a deterministic pseudo-random corpus to exercise carry and borrow
// detection over the full unsigned range, including the u64/usize paths that
// previously used scalar lane loops.
macro_rules! randomized_unsigned_overflow_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn randomized_overflowing_add_sub<const LANES: usize>() {
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

                    let (actual_add, actual_add_overflow) =
                        Simd::<$ty, LANES>::from_array(lhs)
                            .overflowing_add(Simd::from_array(rhs));
                    let (actual_sub, actual_sub_overflow) =
                        Simd::<$ty, LANES>::from_array(lhs)
                            .overflowing_sub(Simd::from_array(rhs));
                    let expected_add: [($ty, bool); LANES] =
                        core::array::from_fn(|lane| lhs[lane].overflowing_add(rhs[lane]));
                    let expected_sub: [($ty, bool); LANES] =
                        core::array::from_fn(|lane| lhs[lane].overflowing_sub(rhs[lane]));

                    assert_eq!(actual_add.to_array(), expected_add.map(|item| item.0));
                    assert_eq!(actual_add_overflow.to_array(), expected_add.map(|item| item.1));
                    assert_eq!(actual_sub.to_array(), expected_sub.map(|item| item.0));
                    assert_eq!(actual_sub_overflow.to_array(), expected_sub.map(|item| item.1));
                }
            }
        }
    };
}

randomized_unsigned_overflow_tests!(u8_randomized, u8);
randomized_unsigned_overflow_tests!(u16_randomized, u16);
randomized_unsigned_overflow_tests!(u32_randomized, u32);
randomized_unsigned_overflow_tests!(u64_randomized, u64);
randomized_unsigned_overflow_tests!(usize_randomized, usize);
