#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_unchecked_arithmetic_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn unchecked_arithmetic<const LANES: usize>() {
                    let lhs: [$ty; LANES] = core::array::from_fn(|lane| {
                        match lane % 4 {
                            0 => -6,
                            1 => -3,
                            2 => 2,
                            _ => 5,
                        }
                    });
                    let rhs: [$ty; LANES] = core::array::from_fn(|lane| {
                        match lane % 4 {
                            0 => 2,
                            1 => -2,
                            2 => 3,
                            _ => -1,
                        }
                    });
                    let left = Simd::<$ty, LANES>::from_array(lhs);
                    let right = Simd::<$ty, LANES>::from_array(rhs);

                    // SAFETY: The chosen operands cannot overflow for any supported integer width.
                    let add = unsafe { left.unchecked_add(right) };
                    // SAFETY: The chosen operands cannot overflow for any supported integer width.
                    let sub = unsafe { left.unchecked_sub(right) };
                    // SAFETY: The chosen operands cannot overflow for any supported integer width.
                    let mul = unsafe { left.unchecked_mul(right) };
                    // SAFETY: Every divisor is nonzero and no signed MIN / -1 pair occurs.
                    let div = unsafe { left.unchecked_div(right) };
                    // SAFETY: Every divisor is nonzero and no signed MIN % -1 pair occurs.
                    let rem = unsafe { left.unchecked_rem(right) };

                    let expected_add = core::array::from_fn(|lane| {
                        // SAFETY: The chosen operands cannot overflow.
                        unsafe { lhs[lane].unchecked_add(rhs[lane]) }
                    });
                    let expected_sub = core::array::from_fn(|lane| {
                        // SAFETY: The chosen operands cannot overflow.
                        unsafe { lhs[lane].unchecked_sub(rhs[lane]) }
                    });
                    let expected_mul = core::array::from_fn(|lane| {
                        // SAFETY: The chosen operands cannot overflow.
                        unsafe { lhs[lane].unchecked_mul(rhs[lane]) }
                    });
                    // The scalar reference uses checked operators; the selected inputs are
                    // valid, so this independently verifies the SIMD unchecked operations.
                    let expected_div =
                        core::array::from_fn(|lane| lhs[lane] / rhs[lane]);
                    let expected_rem =
                        core::array::from_fn(|lane| lhs[lane] % rhs[lane]);

                    assert_eq!(add.to_array(), expected_add);
                    assert_eq!(sub.to_array(), expected_sub);
                    assert_eq!(mul.to_array(), expected_mul);
                    assert_eq!(div.to_array(), expected_div);
                    assert_eq!(rem.to_array(), expected_rem);
                }
            }
        }
    };
}

macro_rules! unsigned_unchecked_arithmetic_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn unchecked_arithmetic<const LANES: usize>() {
                    let lhs: [$ty; LANES] =
                        core::array::from_fn(|lane| (lane as $ty).wrapping_add(10));
                    let rhs: [$ty; LANES] =
                        core::array::from_fn(|lane| (lane % 3 + 1) as $ty);
                    let left = Simd::<$ty, LANES>::from_array(lhs);
                    let right = Simd::<$ty, LANES>::from_array(rhs);

                    // SAFETY: The chosen operands cannot overflow for any supported integer width.
                    let add = unsafe { left.unchecked_add(right) };
                    // SAFETY: lhs is always greater than rhs.
                    let sub = unsafe { left.unchecked_sub(right) };
                    // SAFETY: The chosen operands cannot overflow for any supported integer width.
                    let mul = unsafe { left.unchecked_mul(right) };
                    // SAFETY: rhs is always nonzero.
                    let div = unsafe { left.unchecked_div(right) };
                    // SAFETY: rhs is always nonzero.
                    let rem = unsafe { left.unchecked_rem(right) };

                    let expected_add = core::array::from_fn(|lane| {
                        // SAFETY: The chosen operands cannot overflow.
                        unsafe { lhs[lane].unchecked_add(rhs[lane]) }
                    });
                    let expected_sub = core::array::from_fn(|lane| {
                        // SAFETY: lhs is always greater than rhs.
                        unsafe { lhs[lane].unchecked_sub(rhs[lane]) }
                    });
                    let expected_mul = core::array::from_fn(|lane| {
                        // SAFETY: The chosen operands cannot overflow.
                        unsafe { lhs[lane].unchecked_mul(rhs[lane]) }
                    });
                    // The scalar reference uses checked operators; rhs is always nonzero.
                    let expected_div =
                        core::array::from_fn(|lane| lhs[lane] / rhs[lane]);
                    let expected_rem =
                        core::array::from_fn(|lane| lhs[lane] % rhs[lane]);

                    assert_eq!(add.to_array(), expected_add);
                    assert_eq!(sub.to_array(), expected_sub);
                    assert_eq!(mul.to_array(), expected_mul);
                    assert_eq!(div.to_array(), expected_div);
                    assert_eq!(rem.to_array(), expected_rem);
                }
            }
        }
    };
}

signed_unchecked_arithmetic_tests! { i8_tests, i8 }
signed_unchecked_arithmetic_tests! { i16_tests, i16 }
signed_unchecked_arithmetic_tests! { i32_tests, i32 }
signed_unchecked_arithmetic_tests! { i64_tests, i64 }
signed_unchecked_arithmetic_tests! { isize_tests, isize }

unsigned_unchecked_arithmetic_tests! { u8_tests, u8 }
unsigned_unchecked_arithmetic_tests! { u16_tests, u16 }
unsigned_unchecked_arithmetic_tests! { u32_tests, u32 }
unsigned_unchecked_arithmetic_tests! { u64_tests, u64 }
unsigned_unchecked_arithmetic_tests! { usize_tests, usize }
