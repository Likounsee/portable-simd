#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {$(
        mod $module {
            use super::*;
            test_helpers::test_lanes! {
                fn overflowing_add_sub_matches_scalar<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|i| match i % 4 {
                        0 => <$ty>::MAX,
                        1 => <$ty>::MIN,
                        2 => <$ty>::MAX / 2,
                        _ => -1,
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|i| match i % 4 {
                        0 => 1,
                        1 => -1,
                        2 => <$ty>::MAX / 2,
                        _ => <$ty>::MIN,
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
