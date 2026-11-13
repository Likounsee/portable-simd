#![feature(portable_simd)]
#![feature(f16)]
#![feature(cfg_target_has_reliable_f16_f128)]
#![allow(internal_features)]
#![allow(unused_features)]

macro_rules! unary_test {
    { $scalar:tt, $($func:tt),+ } => {
        test_helpers::test_lanes! {
            $(
            fn $func<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &core_simd::simd::Simd::<$scalar, LANES>::$func,
                    &$scalar::$func,
                    &|_| true,
                )
            }
            )*
        }
    }
}

macro_rules! unary_approx_test {
    { $scalar:tt, $($func:tt),+ } => {
        test_helpers::test_lanes! {
            $(
            fn $func<const LANES: usize>() {
                test_helpers::test_unary_elementwise_approx(
                    &core_simd::simd::Simd::<$scalar, LANES>::$func,
                    &$scalar::$func,
                    &|_| true,
                    16,
                )
            }
            )*
        }
    }
}

macro_rules! ternary_test {
    { $scalar:tt, $($func:tt),+ } => {
        test_helpers::test_lanes! {
            $(
            fn $func<const LANES: usize>() {
                test_helpers::test_ternary_elementwise(
                    &core_simd::simd::Simd::<$scalar, LANES>::$func,
                    &$scalar::$func,
                    &|_, _, _| true,
                )
            }
            )*
        }
    }
}

macro_rules! impl_tests {
    { $scalar:tt } => {
        mod $scalar {
            use std_float::StdFloat;

            unary_test! { $scalar, sqrt, ceil, floor, round, trunc, round_ties_even }
            ternary_test! { $scalar, mul_add }

            // Regression coverage for halfway cases: `round` rounds away from zero,
            // while `round_ties_even` selects the even integer.
            #[test]
            fn round_halfway_cases() {
                let input = core_simd::simd::Simd::<$scalar, 4>::from_array([
                    1.5 as $scalar,
                    2.5 as $scalar,
                    -1.5 as $scalar,
                    -2.5 as $scalar,
                ]);

                assert_eq!(
                    input.round().to_array(),
                    [2.0 as $scalar, 3.0 as $scalar, -2.0 as $scalar, -3.0 as $scalar],
                );
                assert_eq!(
                    input.round_ties_even().to_array(),
                    [2.0 as $scalar, 2.0 as $scalar, -2.0 as $scalar, -2.0 as $scalar],
                );
            }

            // https://github.com/rust-lang/miri/issues/3555
            unary_approx_test! { $scalar, sin, cos, exp, exp2, ln, log2, log10 }

            // The implementation of log is a.ln() / b.ln(), so there are 2 inexact operations,
            // hence a larger ulps is needed.
            test_helpers::test_lanes! {
                fn log<const LANES: usize>() {
                    test_helpers::test_binary_elementwise_approx(
                        &core_simd::simd::Simd::<$scalar, LANES>::log,
                        &$scalar::log,
                        &|_, _| true,
                        32,
                    )
                }
            }

            test_helpers::test_lanes! {
                fn fract<const LANES: usize>() {
                    test_helpers::test_unary_elementwise_flush_subnormals(
                        &core_simd::simd::Simd::<$scalar, LANES>::fract,
                        &$scalar::fract,
                        &|_| true,
                    )
                }
            }
        }
    }
}

#[cfg(target_has_reliable_f16_math)]
impl_tests! { f16 }
impl_tests! { f32 }
impl_tests! { f64 }
