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

            test_helpers::test_lanes! {
                fn mul_sub<const LANES: usize>() {
                    let a = [1.25 as $scalar; LANES];
                    let b = [4.0 as $scalar; LANES];
                    let c = [2.0 as $scalar; LANES];
                    let result = core_simd::simd::Simd::from_array(a)
                        .mul_sub(
                            core_simd::simd::Simd::from_array(b),
                            core_simd::simd::Simd::from_array(c),
                        );
                    let expected = core::array::from_fn(|lane| a[lane].mul_add(b[lane], -c[lane]));
                    assert_eq!(result.to_array(), expected);
                }
            }

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
                assert_eq!(input.round_fast(), input.round_ties_even());
            }

            test_helpers::test_lanes! {
                fn round_fast<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &core_simd::simd::Simd::<$scalar, LANES>::round_fast,
                        &$scalar::round_ties_even,
                        &|_| true,
                    )
                }
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


// These inputs distinguish a fused multiply-subtract from a separately rounded
// multiply followed by subtraction. The exact product is just below 1.0, but
// rounds to 1.0 in the intermediate f32/f64 representation.
#[test]
fn mul_sub_uses_single_rounding_f32() {
    use std_float::StdFloat;

    let epsilon = 2.0_f32.powi(-23);
    let a = core_simd::simd::Simd::<f32, 4>::splat(1.0 + epsilon);
    let b = core_simd::simd::Simd::<f32, 4>::splat(1.0 - epsilon);
    let c = core_simd::simd::Simd::<f32, 4>::splat(1.0);

    let fused = a.mul_sub(b, c);
    let unfused = (a * b) - c;
    let expected = -2.0_f32.powi(-46);

    assert_eq!(fused.to_array(), [expected; 4]);
    assert_eq!(unfused.to_array(), [0.0; 4]);
}

#[test]
fn mul_sub_uses_single_rounding_f64() {
    use std_float::StdFloat;

    let epsilon = 2.0_f64.powi(-52);
    let a = core_simd::simd::Simd::<f64, 4>::splat(1.0 + epsilon);
    let b = core_simd::simd::Simd::<f64, 4>::splat(1.0 - epsilon);
    let c = core_simd::simd::Simd::<f64, 4>::splat(1.0);

    let fused = a.mul_sub(b, c);
    let unfused = (a * b) - c;
    let expected = -2.0_f64.powi(-104);

    assert_eq!(fused.to_array(), [expected; 4]);
    assert_eq!(unfused.to_array(), [0.0; 4]);
}
