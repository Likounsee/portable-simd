#![feature(portable_simd)]

use core_simd::simd::{
    Simd,
    num::{SimdInt, SimdUint},
};

macro_rules! signed_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn division_and_remainder_substitute_invalid_lanes<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => <$ty>::MIN,
                        1 => 12,
                        2 => -9,
                        _ => 8,
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => -1,
                        1 => 3,
                        2 => 0,
                        _ => 2,
                    });
                    let fallback = 77 as $ty;
                    let expected_div =
                        core::array::from_fn(|lane| lhs[lane].checked_div(rhs[lane]).unwrap_or(fallback));
                    let expected_rem =
                        core::array::from_fn(|lane| lhs[lane].checked_rem(rhs[lane]).unwrap_or(fallback));

                    let lhs = Simd::<$ty, LANES>::from_array(lhs);
                    let rhs = Simd::<$ty, LANES>::from_array(rhs);
                    let fallback = Simd::splat(fallback);

                    assert_eq!(lhs.div_or(rhs, fallback).to_array(), expected_div);
                    assert_eq!(lhs.rem_or(rhs, fallback).to_array(), expected_rem);
                }
            }
        }
    };
}

macro_rules! unsigned_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn division_and_remainder_substitute_zero_divisors<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => <$ty>::MAX,
                        1 => 12,
                        2 => 7,
                        _ => 8,
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 0,
                        1 => 3,
                        2 => 2,
                        _ => 1,
                    });
                    let fallback = 77 as $ty;
                    let expected_div =
                        core::array::from_fn(|lane| lhs[lane].checked_div(rhs[lane]).unwrap_or(fallback));
                    let expected_rem =
                        core::array::from_fn(|lane| lhs[lane].checked_rem(rhs[lane]).unwrap_or(fallback));

                    let lhs = Simd::<$ty, LANES>::from_array(lhs);
                    let rhs = Simd::<$ty, LANES>::from_array(rhs);
                    let fallback = Simd::splat(fallback);

                    assert_eq!(lhs.div_or(rhs, fallback).to_array(), expected_div);
                    assert_eq!(lhs.rem_or(rhs, fallback).to_array(), expected_rem);
                }
            }
        }
    };
}

signed_tests!(i8, i8);
signed_tests!(i16, i16);
signed_tests!(i32, i32);
signed_tests!(i64, i64);
signed_tests!(isize, isize);

unsigned_tests!(u8, u8);
unsigned_tests!(u16, u16);
unsigned_tests!(u32, u32);
unsigned_tests!(u64, u64);
unsigned_tests!(usize, usize);


// AArch64 LLVM has historically ICE'd while lowering integer SIMD division and
// remainder for non-power-of-two lane counts when invalid divisors are present.
// Keep these concrete shapes in the target test matrix so the safe wrappers
// continue to replace zero and MIN / -1 divisors before reaching the operation.
fn assert_non_power_of_two_division<const LANES: usize>() {
    let lhs: [i32; LANES] = core::array::from_fn(|lane| match lane % 5 {
        0 => i32::MIN,
        1 => 12,
        2 => -9,
        3 => 17,
        _ => i32::MAX,
    });
    let rhs: [i32; LANES] = core::array::from_fn(|lane| match lane % 5 {
        0 => -1,
        1 => 0,
        2 => 3,
        3 => -5,
        _ => 7,
    });
    let fallback = 123_i32;
    let expected_div =
        core::array::from_fn(|lane| lhs[lane].checked_div(rhs[lane]).unwrap_or(fallback));
    let expected_rem =
        core::array::from_fn(|lane| lhs[lane].checked_rem(rhs[lane]).unwrap_or(fallback));

    let lhs = Simd::<i32, LANES>::from_array(lhs);
    let rhs = Simd::<i32, LANES>::from_array(rhs);
    let fallback = Simd::splat(fallback);

    assert_eq!(lhs.div_or(rhs, fallback).to_array(), expected_div);
    assert_eq!(lhs.rem_or(rhs, fallback).to_array(), expected_rem);
}

#[test]
fn non_power_of_two_division_and_remainder_three_lanes() {
    assert_non_power_of_two_division::<3>();
}

#[test]
fn non_power_of_two_division_and_remainder_five_lanes() {
    assert_non_power_of_two_division::<5>();
}
