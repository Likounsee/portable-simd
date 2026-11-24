#![feature(portable_simd)]

macro_rules! float_rounding_test {
    { $scalar:tt, $int_scalar:tt } => {
        mod $scalar {
            use std_float::StdFloat;

            type Vector<const LANES: usize> = core_simd::simd::Simd<$scalar, LANES>;
            type Scalar = $scalar;
            type IntScalar = $int_scalar;

            test_helpers::test_lanes! {
                fn ceil<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::ceil,
                        &Scalar::ceil,
                        &|_| true,
                    )
                }

                fn floor<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::floor,
                        &Scalar::floor,
                        &|_| true,
                    )
                }

                fn round<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::round,
                        &Scalar::round,
                        &|_| true,
                    )
                }

                fn round_ties_away_from_zero<const LANES: usize>() {
                    let input: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 1.5,
                        1 => 2.5,
                        2 => -1.5,
                        _ => -2.5,
                    });
                    let expected: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 2.0,
                        1 => 3.0,
                        2 => -2.0,
                        _ => -3.0,
                    });

                    assert_eq!(
                        Vector::<LANES>::from_array(input).round().to_array(),
                        expected
                    );
                }

                fn trunc<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::trunc,
                        &Scalar::trunc,
                        &|_| true,
                    )
                }

                fn round_ties_even<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::round_ties_even,
                        &Scalar::round_ties_even,
                        &|_| true,
                    )
                }

                fn round_fast<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::round_fast,
                        &Scalar::round_ties_even,
                        &|_| true,
                    )
                }

                fn round_fast_ties_even_boundaries<const LANES: usize>() {
                    let input: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 8 {
                        0 => 0.5,
                        1 => 1.5,
                        2 => 2.5,
                        3 => 3.5,
                        4 => -0.5,
                        5 => -1.5,
                        6 => -2.5,
                        _ => -3.5,
                    });
                    let expected: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 8 {
                        0 | 4 => 0.0,
                        1 | 2 => 2.0,
                        3 => 4.0,
                        5 | 6 => -2.0,
                        _ => -4.0,
                    });

                    let actual =
                        Vector::<LANES>::from_array(input).round_fast().to_array();
                    for lane in 0..LANES {
                        assert_eq!(
                            actual[lane].to_bits(),
                            expected[lane].to_bits(),
                            "round_fast: lane={lane}",
                        );
                    }
                }

                fn fract<const LANES: usize>() {
                    test_helpers::test_unary_elementwise_flush_subnormals(
                        &Vector::<LANES>::fract,
                        &Scalar::fract,
                        &|_| true,
                    )
                }
            }

            test_helpers::test_lanes! {
                fn to_int_unchecked<const LANES: usize>() {
                    use core_simd::simd::num::SimdFloat;
                    // The maximum integer that can be represented by the equivalently sized float has
                    // all of the mantissa digits set to 1, pushed up to the MSB.
                    const ALL_MANTISSA_BITS: IntScalar = ((1 << <Scalar>::MANTISSA_DIGITS) - 1);
                    const MAX_REPRESENTABLE_VALUE: Scalar =
                        (ALL_MANTISSA_BITS << (size_of::<Scalar>() * 8 - <Scalar>::MANTISSA_DIGITS as usize - 1)) as Scalar;

                    let mut runner = test_helpers::make_runner();
                    runner.run(
                        &test_helpers::array::UniformArrayStrategy::new(-MAX_REPRESENTABLE_VALUE..MAX_REPRESENTABLE_VALUE),
                        |x| {
                            let result_1 = unsafe { Vector::from_array(x).to_int_unchecked::<IntScalar>().to_array() };
                            let result_2 = {
                                let mut result: [IntScalar; LANES] = [0; LANES];
                                for (i, o) in x.iter().zip(result.iter_mut()) {
                                    *o = unsafe { i.to_int_unchecked::<IntScalar>() };
                                }
                                result
                            };
                            test_helpers::prop_assert_biteq!(result_1, result_2);
                            Ok(())
                        },
                    ).unwrap();
                }
            }
        }
    }
}

float_rounding_test! { f32, i32 }
float_rounding_test! { f64, i64 }
