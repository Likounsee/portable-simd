/// Implements a test on a unary operation using proptest.
///
/// Compares the vector operation to the equivalent scalar operation.
#[macro_export]
macro_rules! impl_unary_op_test {
    { $scalar:ty, $trait:ident :: $fn:ident, $scalar_fn:expr } => {
        test_helpers::test_lanes! {
            fn $fn<const LANES: usize>() {
                test_helpers::test_unary_elementwise_flush_subnormals(
                    &<core_simd::simd::Simd<$scalar, LANES> as core::ops::$trait>::$fn,
                    &$scalar_fn,
                    &|_| true,
                );
            }
        }
    };
    { $scalar:ty, $trait:ident :: $fn:ident } => {
        impl_unary_op_test! { $scalar, $trait::$fn, <$scalar as core::ops::$trait>::$fn }
    };
}

/// Implements a test on a binary operation using proptest.
///
/// Compares the vector operation to the equivalent scalar operation.
#[macro_export]
macro_rules! impl_binary_op_test {
    { $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident, $scalar_fn:expr } => {
        mod $fn {
            use super::*;
            use core_simd::simd::Simd;

            test_helpers::test_lanes! {
                fn normal<const LANES: usize>() {
                    test_helpers::test_binary_elementwise_flush_subnormals(
                        &<Simd<$scalar, LANES> as core::ops::$trait>::$fn,
                        &$scalar_fn,
                        &|_, _| true,
                    );
                }

                fn assign<const LANES: usize>() {
                    test_helpers::test_binary_elementwise_flush_subnormals(
                        &|mut a, b| { <Simd<$scalar, LANES> as core::ops::$trait_assign>::$fn_assign(&mut a, b); a },
                        &$scalar_fn,
                        &|_, _| true,
                    );
                }
            }
        }
    };
    { $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident } => {
        impl_binary_op_test! { $scalar, $trait::$fn, $trait_assign::$fn_assign, <$scalar as core::ops::$trait>::$fn }
    };
}

/// Implements a test on a binary operation using proptest.
///
/// Like `impl_binary_op_test`, but allows providing a function for rejecting particular inputs
/// (like the `proptest_assume` macro).
///
/// Compares the vector operation to the equivalent scalar operation.
#[macro_export]
macro_rules! impl_binary_checked_op_test {
    { $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident, $scalar_fn:expr, $check_fn:expr } => {
        mod $fn {
            use super::*;
            use core_simd::simd::Simd;

            test_helpers::test_lanes! {
                fn normal<const LANES: usize>() {
                    #![allow(clippy::redundant_closure_call)]
                    test_helpers::test_binary_elementwise(
                        &<Simd<$scalar, LANES> as core::ops::$trait>::$fn,
                        &$scalar_fn,
                        &|x, y| x.iter().zip(y.iter()).all(|(x, y)| $check_fn(*x, *y)),
                    );
                }

                fn assign<const LANES: usize>() {
                    #![allow(clippy::redundant_closure_call)]
                    test_helpers::test_binary_elementwise(
                        &|mut a, b| { <Simd<$scalar, LANES> as core::ops::$trait_assign>::$fn_assign(&mut a, b); a },
                        &$scalar_fn,
                        &|x, y| x.iter().zip(y.iter()).all(|(x, y)| $check_fn(*x, *y)),
                    )
                }
            }
        }
    };
    { $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident, $check_fn:expr } => {
        impl_binary_checked_op_test! { $scalar, $trait::$fn, $trait_assign::$fn_assign, <$scalar as core::ops::$trait>::$fn, $check_fn }
    };
}

#[macro_export]
macro_rules! impl_common_integer_tests {
    { $vector:ident, $scalar:ident } => {
        test_helpers::test_lanes! {
            fn shr<const LANES: usize>() {
                use core::ops::Shr;
                let shr = |x: $scalar, y: $scalar| x.wrapping_shr(y as _);
                test_helpers::test_binary_elementwise(
                    &<$vector::<LANES> as Shr<$vector::<LANES>>>::shr,
                    &shr,
                    &|_, _| true,
                );
                test_helpers::test_binary_scalar_rhs_elementwise(
                    &<$vector::<LANES> as Shr<$scalar>>::shr,
                    &shr,
                    &|_, _| true,
                );
            }

            fn shl<const LANES: usize>() {
                use core::ops::Shl;
                let shl = |x: $scalar, y: $scalar| x.wrapping_shl(y as _);
                test_helpers::test_binary_elementwise(
                    &<$vector::<LANES> as Shl<$vector::<LANES>>>::shl,
                    &shl,
                    &|_, _| true,
                );
                test_helpers::test_binary_scalar_rhs_elementwise(
                    &<$vector::<LANES> as Shl<$scalar>>::shl,
                    &shl,
                    &|_, _| true,
                );
            }

            fn rotate_left<const LANES: usize>() {
                let values: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                    0 => 0,
                    1 => 1,
                    2 => Scalar::MAX,
                    _ => Scalar::MIN,
                });
                let vector = $vector::<LANES>::from_array(values);
                for shift in [0, 1, Scalar::BITS - 1, Scalar::BITS, Scalar::BITS + 1] {
                    let expected = core::array::from_fn(|lane| values[lane].rotate_left(shift));
                    assert_eq!(vector.rotate_left(shift).to_array(), expected);
                }
            }

            fn rotate_right<const LANES: usize>() {
                let values: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                    0 => 0,
                    1 => 1,
                    2 => Scalar::MAX,
                    _ => Scalar::MIN,
                });
                let vector = $vector::<LANES>::from_array(values);
                for shift in [0, 1, Scalar::BITS - 1, Scalar::BITS, Scalar::BITS + 1] {
                    let expected = core::array::from_fn(|lane| values[lane].rotate_right(shift));
                    assert_eq!(vector.rotate_right(shift).to_array(), expected);
                }
            }

            fn reduce_sum<const LANES: usize>() {
                test_helpers::test_1(&|x| {
                    use test_helpers::subnormals::{flush, flush_in};
                    test_helpers::prop_assert_biteq! (
                        $vector::<LANES>::from_array(x).reduce_sum(),
                        x.iter().copied().fold(0 as $scalar, $scalar::wrapping_add),
                        flush(x.iter().copied().map(flush_in).fold(0 as $scalar, $scalar::wrapping_add)),
                    );
                    Ok(())
                });
            }

            fn reduce_product<const LANES: usize>() {
                test_helpers::test_1(&|x| {
                    use test_helpers::subnormals::{flush, flush_in};
                    test_helpers::prop_assert_biteq! (
                        $vector::<LANES>::from_array(x).reduce_product(),
                        x.iter().copied().fold(1 as $scalar, $scalar::wrapping_mul),
                        flush(x.iter().copied().map(flush_in).fold(1 as $scalar, $scalar::wrapping_mul)),
                    );
                    Ok(())
                });
            }

            fn reduce_and<const LANES: usize>() {
                test_helpers::test_1(&|x| {
                    test_helpers::prop_assert_biteq! (
                        $vector::<LANES>::from_array(x).reduce_and(),
                        x.iter().copied().fold(-1i8 as $scalar, <$scalar as core::ops::BitAnd>::bitand),
                    );
                    Ok(())
                });
            }

            fn reduce_or<const LANES: usize>() {
                test_helpers::test_1(&|x| {
                    test_helpers::prop_assert_biteq! (
                        $vector::<LANES>::from_array(x).reduce_or(),
                        x.iter().copied().fold(0 as $scalar, <$scalar as core::ops::BitOr>::bitor),
                    );
                    Ok(())
                });
            }

            fn reduce_xor<const LANES: usize>() {
                test_helpers::test_1(&|x| {
                    test_helpers::prop_assert_biteq! (
                        $vector::<LANES>::from_array(x).reduce_xor(),
                        x.iter().copied().fold(0 as $scalar, <$scalar as core::ops::BitXor>::bitxor),
                    );
                    Ok(())
                });
            }

            fn reduce_max<const LANES: usize>() {
                test_helpers::test_1(&|x| {
                    test_helpers::prop_assert_biteq! (
                        $vector::<LANES>::from_array(x).reduce_max(),
                        x.iter().copied().max().unwrap(),
                    );
                    Ok(())
                });
            }

            fn reduce_min<const LANES: usize>() {
                test_helpers::test_1(&|x| {
                    test_helpers::prop_assert_biteq! (
                        $vector::<LANES>::from_array(x).reduce_min(),
                        x.iter().copied().min().unwrap(),
                    );
                    Ok(())
                });
            }

            fn swap_bytes<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &$vector::<LANES>::swap_bytes,
                    &$scalar::swap_bytes,
                    &|_| true,
                )
            }

            fn reverse_bits<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &$vector::<LANES>::reverse_bits,
                    &$scalar::reverse_bits,
                    &|_| true,
                )
            }

            fn count_ones<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &$vector::<LANES>::count_ones,
                    &|x| x.count_ones() as _,
                    &|_| true,
                )
            }

            fn count_zeros<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &$vector::<LANES>::count_zeros,
                    &|x| x.count_zeros() as _,
                    &|_| true,
                )
            }

            fn leading_zeros<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &$vector::<LANES>::leading_zeros,
                    &|x| x.leading_zeros() as _,
                    &|_| true,
                )
            }

            fn trailing_zeros<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &$vector::<LANES>::trailing_zeros,
                    &|x| x.trailing_zeros() as _,
                    &|_| true,
                )
            }

            fn leading_ones<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &$vector::<LANES>::leading_ones,
                    &|x| x.leading_ones() as _,
                    &|_| true,
                )
            }

            fn trailing_ones<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &$vector::<LANES>::trailing_ones,
                    &|x| x.trailing_ones() as _,
                    &|_| true,
                )
            }
        }
    }
}

/// Implement tests for signed integers.
#[macro_export]
macro_rules! impl_signed_tests {
    { $scalar:tt } => {
        mod $scalar {
            use core_simd::simd::num::SimdInt;
            type Vector<const LANES: usize> = core_simd::simd::Simd<Scalar, LANES>;
            type Scalar = $scalar;

            impl_common_integer_tests! { Vector, Scalar }

            test_helpers::test_lanes! {
                fn unchecked_add<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX - 1,
                        1 => Scalar::MIN + 1,
                        2 => -5,
                        _ => 7,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 1,
                        1 => -1,
                        2 => 5,
                        _ => 2,
                    });
                    // SAFETY: Every pair above has a representable sum.
                    let actual = unsafe {
                        Vector::<LANES>::from_array(lhs)
                            .unchecked_add(Vector::<LANES>::from_array(rhs))
                    };
                    let expected = core::array::from_fn(|lane| lhs[lane] + rhs[lane]);
                    assert_eq!(actual.to_array(), expected);
                }

                fn unchecked_sub<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MIN + 1,
                        1 => Scalar::MAX - 1,
                        2 => 5,
                        _ => -5,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 1,
                        1 => -1,
                        2 => 5,
                        _ => -5,
                    });
                    // SAFETY: Every pair above has a representable difference.
                    let actual = unsafe {
                        Vector::<LANES>::from_array(lhs)
                            .unchecked_sub(Vector::<LANES>::from_array(rhs))
                    };
                    let expected = core::array::from_fn(|lane| lhs[lane] - rhs[lane]);
                    assert_eq!(actual.to_array(), expected);
                }

                fn unchecked_mul<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX / 2,
                        1 => Scalar::MIN / 2,
                        2 => -7,
                        _ => 0,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 2,
                        1 => 2,
                        2 => -3,
                        _ => Scalar::MAX,
                    });
                    // SAFETY: Every pair above has a representable product.
                    let actual = unsafe {
                        Vector::<LANES>::from_array(lhs)
                            .unchecked_mul(Vector::<LANES>::from_array(rhs))
                    };
                    let expected = core::array::from_fn(|lane| lhs[lane] * rhs[lane]);
                    assert_eq!(actual.to_array(), expected);
                }

                fn mul_high<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        &Vector::<LANES>::mul_high,
                        &|a: Scalar, b: Scalar| (((a as i128) * (b as i128)) >> Scalar::BITS) as Scalar,
                        &|_, _| true,
                    );
                }

                fn saturating_mul<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        &Vector::<LANES>::saturating_mul,
                        &Scalar::saturating_mul,
                        &|_, _| true,
                    );
                }

                fn overflowing_add<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX,
                        1 => 0,
                        2 => Scalar::MAX / 2,
                        _ => 12,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 1,
                        1 => 1,
                        2 => 3,
                        _ => 3,
                    });
                    let expected = core::array::from_fn(|lane| lhs[lane].overflowing_add(rhs[lane]));
                    let (actual, overflow) =
                        Vector::<LANES>::from_array(lhs).overflowing_add(Vector::from_array(rhs));
                    assert_eq!(actual.to_array(), expected.map(|(value, _)| value));
                    assert_eq!(
                        overflow,
                        Vector::<LANES>::from_array(expected.map(|(_, did_overflow)| did_overflow as Scalar)).to_mask(),
                    );
                }

                fn overflowing_sub<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX,
                        1 => 0,
                        2 => Scalar::MAX / 2,
                        _ => 12,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 1,
                        1 => 1,
                        2 => Scalar::MAX,
                        _ => 3,
                    });
                    let expected = core::array::from_fn(|lane| lhs[lane].overflowing_sub(rhs[lane]));
                    let (actual, overflow) =
                        Vector::<LANES>::from_array(lhs).overflowing_sub(Vector::from_array(rhs));
                    assert_eq!(actual.to_array(), expected.map(|(value, _)| value));
                    assert_eq!(
                        overflow,
                        Vector::<LANES>::from_array(expected.map(|(_, did_overflow)| did_overflow as Scalar)).to_mask(),
                    );
                }

                fn overflowing_div<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MIN,
                        1 => Scalar::MAX,
                        2 => -21,
                        _ => 21,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => -1,
                        1 => 2,
                        2 => 5,
                        _ => -7,
                    });
                    let expected = core::array::from_fn(|lane| lhs[lane].overflowing_div(rhs[lane]));
                    let (actual, overflow) =
                        Vector::<LANES>::from_array(lhs).overflowing_div(Vector::from_array(rhs));
                    assert_eq!(actual.to_array(), expected.map(|(value, _)| value));
                    assert_eq!(
                        overflow,
                        Vector::<LANES>::from_array(expected.map(|(_, did_overflow)| did_overflow as Scalar)).to_mask(),
                    );
                }

                fn overflowing_rem<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MIN,
                        1 => Scalar::MAX,
                        2 => -21,
                        _ => 21,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => -1,
                        1 => 2,
                        2 => 5,
                        _ => -7,
                    });
                    let expected = core::array::from_fn(|lane| lhs[lane].overflowing_rem(rhs[lane]));
                    let (actual, overflow) =
                        Vector::<LANES>::from_array(lhs).overflowing_rem(Vector::from_array(rhs));
                    assert_eq!(actual.to_array(), expected.map(|(value, _)| value));
                    assert_eq!(
                        overflow,
                        Vector::<LANES>::from_array(expected.map(|(_, did_overflow)| did_overflow as Scalar)).to_mask(),
                    );
                }

                fn overflowing_mul<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX,
                        1 => 0,
                        2 => Scalar::MAX / 2,
                        _ => 12,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 2,
                        1 => 0,
                        2 => 3,
                        _ => 3,
                    });
                    let expected = core::array::from_fn(|lane| lhs[lane].overflowing_mul(rhs[lane]));
                    let (actual, overflow) =
                        Vector::<LANES>::from_array(lhs).overflowing_mul(Vector::from_array(rhs));
                    assert_eq!(actual.to_array(), expected.map(|(value, _)| value));
                    assert_eq!(
                        overflow,
                        Vector::<LANES>::from_array(expected.map(|(_, did_overflow)| did_overflow as Scalar)).to_mask(),
                    );
                }

                fn overflowing_add<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX,
                        1 => Scalar::MIN,
                        2 => Scalar::MAX / 2,
                        _ => -12,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 1,
                        1 => -1,
                        2 => 2,
                        _ => 12,
                    });
                    let expected = core::array::from_fn(|lane| lhs[lane].overflowing_add(rhs[lane]));
                    let (actual, overflow) =
                        Vector::<LANES>::from_array(lhs).overflowing_add(Vector::from_array(rhs));
                    assert_eq!(actual.to_array(), expected.map(|(value, _)| value));
                    assert_eq!(
                        overflow,
                        Vector::<LANES>::from_array(expected.map(|(_, did_overflow)| did_overflow as Scalar)).to_mask(),
                    );
                }

                fn overflowing_sub<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX,
                        1 => Scalar::MIN,
                        2 => Scalar::MIN / 2,
                        _ => -12,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => -1,
                        1 => 1,
                        2 => 2,
                        _ => -12,
                    });
                    let expected = core::array::from_fn(|lane| lhs[lane].overflowing_sub(rhs[lane]));
                    let (actual, overflow) =
                        Vector::<LANES>::from_array(lhs).overflowing_sub(Vector::from_array(rhs));
                    assert_eq!(actual.to_array(), expected.map(|(value, _)| value));
                    assert_eq!(
                        overflow,
                        Vector::<LANES>::from_array(expected.map(|(_, did_overflow)| did_overflow as Scalar)).to_mask(),
                    );
                }

                fn overflowing_div<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX,
                        1 => Scalar::MAX - 1,
                        2 => 21,
                        _ => 42,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 2,
                        1 => 3,
                        2 => 5,
                        _ => 7,
                    });
                    let expected = core::array::from_fn(|lane| lhs[lane].overflowing_div(rhs[lane]));
                    let (actual, overflow) =
                        Vector::<LANES>::from_array(lhs).overflowing_div(Vector::from_array(rhs));
                    assert_eq!(actual.to_array(), expected.map(|(value, _)| value));
                    assert_eq!(
                        overflow,
                        Vector::<LANES>::from_array(expected.map(|(_, did_overflow)| did_overflow as Scalar)).to_mask(),
                    );
                }

                fn overflowing_rem<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX,
                        1 => Scalar::MAX - 1,
                        2 => 21,
                        _ => 42,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 2,
                        1 => 3,
                        2 => 5,
                        _ => 7,
                    });
                    let expected = core::array::from_fn(|lane| lhs[lane].overflowing_rem(rhs[lane]));
                    let (actual, overflow) =
                        Vector::<LANES>::from_array(lhs).overflowing_rem(Vector::from_array(rhs));
                    assert_eq!(actual.to_array(), expected.map(|(value, _)| value));
                    assert_eq!(
                        overflow,
                        Vector::<LANES>::from_array(expected.map(|(_, did_overflow)| did_overflow as Scalar)).to_mask(),
                    );
                }

                fn overflowing_mul<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX,
                        1 => Scalar::MIN,
                        2 => Scalar::MAX / 2,
                        _ => -12,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 2,
                        1 => 2,
                        2 => 3,
                        _ => 3,
                    });
                    let expected = core::array::from_fn(|lane| lhs[lane].overflowing_mul(rhs[lane]));
                    let (actual, overflow) =
                        Vector::<LANES>::from_array(lhs).overflowing_mul(Vector::from_array(rhs));
                    assert_eq!(actual.to_array(), expected.map(|(value, _)| value));
                    assert_eq!(
                        overflow,
                        Vector::<LANES>::from_array(expected.map(|(_, did_overflow)| did_overflow as Scalar)).to_mask(),
                    );
                }

                fn neg<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &<Vector::<LANES> as core::ops::Neg>::neg,
                        &<Scalar as core::ops::Neg>::neg,
                        &|x| !x.contains(&Scalar::MIN),
                    );
                }

                fn to_mask<const LANES: usize>() {
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::to_mask,
                        &|x: Scalar| x != 0,
                        &|_| true,
                    );
                }

                fn is_positive<const LANES: usize>() {
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::is_positive,
                        &Scalar::is_positive,
                        &|_| true,
                    );
                }

                fn is_negative<const LANES: usize>() {
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::is_negative,
                        &Scalar::is_negative,
                        &|_| true,
                    );
                }

                fn signum<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::signum,
                        &Scalar::signum,
                        &|_| true,
                    )
                }

                fn abs_diff<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        &Vector::<LANES>::abs_diff,
                        &Scalar::abs_diff,
                        &|_, _| true,
                    )
                }

                fn simd_min<const LANES: usize>() {
                    use core_simd::simd::cmp::SimdOrd;
                    let a = Vector::<LANES>::splat(Scalar::MIN);
                    let b = Vector::<LANES>::splat(0);
                    assert_eq!(a.simd_min(b), a);
                    let a = Vector::<LANES>::splat(Scalar::MAX);
                    let b = Vector::<LANES>::splat(0);
                    assert_eq!(a.simd_min(b), b);
                }

                fn simd_max<const LANES: usize>() {
                    use core_simd::simd::cmp::SimdOrd;
                    let a = Vector::<LANES>::splat(Scalar::MIN);
                    let b = Vector::<LANES>::splat(0);
                    assert_eq!(a.simd_max(b), b);
                    let a = Vector::<LANES>::splat(Scalar::MAX);
                    let b = Vector::<LANES>::splat(0);
                    assert_eq!(a.simd_max(b), a);
                }

                fn simd_clamp<const LANES: usize>() {
                    use core_simd::simd::cmp::SimdOrd;
                    let min = Vector::<LANES>::splat(Scalar::MIN);
                    let max = Vector::<LANES>::splat(Scalar::MAX);
                    let zero = Vector::<LANES>::splat(0);
                    let one = Vector::<LANES>::splat(1);
                    let negone = Vector::<LANES>::splat(-1);
                    assert_eq!(zero.simd_clamp(min, max), zero);
                    assert_eq!(zero.simd_clamp(min, one), zero);
                    assert_eq!(zero.simd_clamp(one, max), one);
                    assert_eq!(zero.simd_clamp(min, negone), negone);
                }
            }

            test_helpers::test_lanes_panic! {
                fn div_by_all_zeros_panics<const LANES: usize>() {
                    let a = Vector::<LANES>::splat(42);
                    let b = Vector::<LANES>::splat(0);
                    let _ = a / b;
                }

                fn div_by_one_zero_panics<const LANES: usize>() {
                    let a = Vector::<LANES>::splat(42);
                    let mut b = Vector::<LANES>::splat(21);
                    b.set(0, 0 as _);
                    let _ = a / b;
                }

                fn rem_zero_panic<const LANES: usize>() {
                    let a = Vector::<LANES>::splat(42);
                    let b = Vector::<LANES>::splat(0);
                    let _ = a % b;
                }
            }

            test_helpers::test_lanes! {
                fn div_neg_one_no_panic<const LANES: usize>() {
                    let a = Vector::<LANES>::splat(42);
                    let b = Vector::<LANES>::splat(-1);
                    let _ = a / b;
                }

                fn rem_neg_one_no_panic<const LANES: usize>() {
                    let a = Vector::<LANES>::splat(42);
                    let b = Vector::<LANES>::splat(-1);
                    let _ = a % b;
                }

                // Issue #427: exercise the overflow boundary cases on every
                // generated lane width, including non-power-of-two widths 3
                // and 6 on AArch64 and under Miri.
                fn div_min_may_overflow<const LANES: usize>() {
                    let a = Vector::<LANES>::splat(Scalar::MIN);
                    let b = Vector::<LANES>::splat(-1);
                    assert_eq!(a / b, a);
                }

                fn rem_min_may_overflow<const LANES: usize>() {
                    let a = Vector::<LANES>::splat(Scalar::MIN);
                    let b = Vector::<LANES>::splat(-1);
                    assert_eq!(a % b, Vector::<LANES>::splat(0));
                }

                // The original report specifically identifies Scalar::MIN as
                // the divisor (rhs), not the dividend.
                fn rem_by_min_divisor<const LANES: usize>() {
                    let divisor = Vector::<LANES>::splat(Scalar::MIN);

                    // Exercise ordinary values and the signed boundaries.
                    for (dividend, expected) in [
                        (42, 42),
                        (-42, -42),
                        (Scalar::MIN, 0),
                        (Scalar::MAX, Scalar::MAX),
                        (-1, -1),
                        (0, 0),
                    ] {
                        let lhs = Vector::<LANES>::splat(dividend);
                        assert_eq!(
                            lhs % divisor,
                            Vector::<LANES>::splat(expected),
                            "dividend={dividend:?}, divisor=MIN",
                        );
                    }
                }

                // Mix the minimum signed divisor with ordinary divisors in one
                // vector, so the regression also exercises lane-wise
                // remainder selection rather than only splatted operands.
                fn rem_by_min_divisor_mixed_lanes<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| {
                        match lane % 6 {
                            0 => Scalar::MIN,
                            1 => Scalar::MAX,
                            2 => -42,
                            3 => 42,
                            4 => -1,
                            _ => 0,
                        }
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 6 {
                        0 | 1 | 4 => Scalar::MIN,
                        2 | 5 => 7,
                        _ => -7,
                    });
                    let expected: [Scalar; LANES] =
                        core::array::from_fn(|lane| lhs[lane].wrapping_rem(rhs[lane]));

                    assert_eq!(
                        Vector::<LANES>::from_array(lhs) % Vector::<LANES>::from_array(rhs),
                        Vector::<LANES>::from_array(expected),
                    );
                }
            }

            impl_binary_op_test!(Scalar, Add::add, AddAssign::add_assign, Scalar::wrapping_add);
            impl_binary_op_test!(Scalar, Sub::sub, SubAssign::sub_assign, Scalar::wrapping_sub);
            impl_binary_op_test!(Scalar, Mul::mul, MulAssign::mul_assign, Scalar::wrapping_mul);

            // Exclude Div and Rem panicking cases
            impl_binary_checked_op_test!(Scalar, Div::div, DivAssign::div_assign, Scalar::wrapping_div, |x, y| y != 0 && !(x == Scalar::MIN && y == -1));
            impl_binary_checked_op_test!(Scalar, Rem::rem, RemAssign::rem_assign, Scalar::wrapping_rem, |x, y| y != 0 && !(x == Scalar::MIN && y == -1));

            impl_unary_op_test!(Scalar, Not::not);
            impl_binary_op_test!(Scalar, BitAnd::bitand, BitAndAssign::bitand_assign);
            impl_binary_op_test!(Scalar, BitOr::bitor, BitOrAssign::bitor_assign);
            impl_binary_op_test!(Scalar, BitXor::bitxor, BitXorAssign::bitxor_assign);
        }
    }
}

/// Implement tests for unsigned integers.
#[macro_export]
macro_rules! impl_unsigned_tests {
    { $scalar:tt } => {
        mod $scalar {
            use core_simd::simd::num::SimdUint;
            type Vector<const LANES: usize> = core_simd::simd::Simd<Scalar, LANES>;
            type Scalar = $scalar;

            impl_common_integer_tests! { Vector, Scalar }

            test_helpers::test_lanes! {
                fn unchecked_add<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX - 1,
                        1 => 0,
                        2 => 10,
                        _ => 4,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 1,
                        1 => 0,
                        2 => 20,
                        _ => 5,
                    });
                    // SAFETY: Every pair above has a representable sum.
                    let actual = unsafe {
                        Vector::<LANES>::from_array(lhs)
                            .unchecked_add(Vector::<LANES>::from_array(rhs))
                    };
                    let expected = core::array::from_fn(|lane| lhs[lane] + rhs[lane]);
                    assert_eq!(actual.to_array(), expected);
                }

                fn unchecked_sub<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX,
                        1 => 1,
                        2 => 20,
                        _ => 5,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 1,
                        1 => 1,
                        2 => 10,
                        _ => 5,
                    });
                    // SAFETY: Every pair above has a non-underflowing difference.
                    let actual = unsafe {
                        Vector::<LANES>::from_array(lhs)
                            .unchecked_sub(Vector::<LANES>::from_array(rhs))
                    };
                    let expected = core::array::from_fn(|lane| lhs[lane] - rhs[lane]);
                    assert_eq!(actual.to_array(), expected);
                }

                fn unchecked_mul<const LANES: usize>() {
                    let lhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => Scalar::MAX / 2,
                        1 => 0,
                        2 => 10,
                        _ => 4,
                    });
                    let rhs: [Scalar; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => 2,
                        1 => Scalar::MAX,
                        2 => 3,
                        _ => 5,
                    });
                    // SAFETY: Every pair above has a representable product.
                    let actual = unsafe {
                        Vector::<LANES>::from_array(lhs)
                            .unchecked_mul(Vector::<LANES>::from_array(rhs))
                    };
                    let expected = core::array::from_fn(|lane| lhs[lane] * rhs[lane]);
                    assert_eq!(actual.to_array(), expected);
                }

            }

            test_helpers::test_lanes! {
                fn mul_high<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        &Vector::<LANES>::mul_high,
                        &|a: Scalar, b: Scalar| (((a as u128) * (b as u128)) >> Scalar::BITS) as Scalar,
                        &|_, _| true,
                    );
                }

                fn saturating_mul<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        &Vector::<LANES>::saturating_mul,
                        &Scalar::saturating_mul,
                        &|_, _| true,
                    );
                }
            }

            test_helpers::test_lanes_panic! {
                fn rem_zero_panic<const LANES: usize>() {
                    let a = Vector::<LANES>::splat(42);
                    let b = Vector::<LANES>::splat(0);
                    let _ = a % b;
                }
            }

            test_helpers::test_lanes! {
                fn to_mask<const LANES: usize>() {
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::to_mask,
                        &|x: Scalar| x != 0,
                        &|_| true,
                    );
                }

                fn wrapping_neg<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::wrapping_neg,
                        &Scalar::wrapping_neg,
                        &|_| true,
                    );
                }

                fn abs_diff<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        &Vector::<LANES>::abs_diff,
                        &Scalar::abs_diff,
                        &|_, _| true,
                    )
                }
            }

            impl_binary_op_test!(Scalar, Add::add, AddAssign::add_assign, Scalar::wrapping_add);
            impl_binary_op_test!(Scalar, Sub::sub, SubAssign::sub_assign, Scalar::wrapping_sub);
            impl_binary_op_test!(Scalar, Mul::mul, MulAssign::mul_assign, Scalar::wrapping_mul);

            // Exclude Div and Rem panicking cases
            impl_binary_checked_op_test!(Scalar, Div::div, DivAssign::div_assign, Scalar::wrapping_div, |_, y| y != 0);
            impl_binary_checked_op_test!(Scalar, Rem::rem, RemAssign::rem_assign, Scalar::wrapping_rem, |_, y| y != 0);

            impl_unary_op_test!(Scalar, Not::not);
            impl_binary_op_test!(Scalar, BitAnd::bitand, BitAndAssign::bitand_assign);
            impl_binary_op_test!(Scalar, BitOr::bitor, BitOrAssign::bitor_assign);
            impl_binary_op_test!(Scalar, BitXor::bitxor, BitXorAssign::bitxor_assign);
        }
    }
}

/// Implement tests for floating point numbers.
#[macro_export]
macro_rules! impl_float_tests {
    { $scalar:tt, $int_scalar:tt } => {
        mod $scalar {
            use core_simd::simd::num::SimdFloat;
            type Vector<const LANES: usize> = core_simd::simd::Simd<Scalar, LANES>;
            type Scalar = $scalar;

            impl_unary_op_test!(Scalar, Neg::neg);
            impl_binary_op_test!(Scalar, Add::add, AddAssign::add_assign);
            impl_binary_op_test!(Scalar, Sub::sub, SubAssign::sub_assign);
            impl_binary_op_test!(Scalar, Mul::mul, MulAssign::mul_assign);
            impl_binary_op_test!(Scalar, Div::div, DivAssign::div_assign);
            impl_binary_op_test!(Scalar, Rem::rem, RemAssign::rem_assign);

            test_helpers::test_lanes! {
                fn algebraic_add<const LANES: usize>() {
                    test_helpers::test_binary_elementwise_approx(
                        &Vector::<LANES>::algebraic_add,
                        &Scalar::algebraic_add,
                        &|_, _| true,
                        16,
                    );
                }

                fn algebraic_sub<const LANES: usize>() {
                    test_helpers::test_binary_elementwise_approx(
                        &Vector::<LANES>::algebraic_sub,
                        &Scalar::algebraic_sub,
                        &|_, _| true,
                        16,
                    );
                }

                fn algebraic_mul<const LANES: usize>() {
                    test_helpers::test_binary_elementwise_approx(
                        &Vector::<LANES>::algebraic_mul,
                        &Scalar::algebraic_mul,
                        &|_, _| true,
                        16,
                    );
                }

                fn algebraic_div<const LANES: usize>() {
                    test_helpers::test_binary_elementwise_approx(
                        &Vector::<LANES>::algebraic_div,
                        &Scalar::algebraic_div,
                        &|_, _| true,
                        16,
                    );
                }

                fn algebraic_rem<const LANES: usize>() {
                    test_helpers::test_binary_elementwise_approx(
                        &Vector::<LANES>::algebraic_rem,
                        &Scalar::algebraic_rem,
                        &|_, _| true,
                        16,
                    );
                }

                fn mul_add_relaxed<const LANES: usize>() {
                    let a = Vector::<LANES>::splat(2.0);
                    let b = Vector::<LANES>::splat(3.0);
                    let c = Vector::<LANES>::splat(1.0);
                    assert_eq!(a.mul_add_relaxed(b, c), Vector::<LANES>::splat(7.0));
                }

                fn to_mask<const LANES: usize>() {
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::to_mask,
                        &|x: Scalar| x != 0.0,
                        &|_| true,
                    );
                }

                fn is_sign_positive<const LANES: usize>() {
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::is_sign_positive,
                        &Scalar::is_sign_positive,
                        &|_| true,
                    );
                }

                fn is_sign_negative<const LANES: usize>() {
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::is_sign_negative,
                        &Scalar::is_sign_negative,
                        &|_| true,
                    );
                }

                fn is_finite<const LANES: usize>() {
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::is_finite,
                        &Scalar::is_finite,
                        &|_| true,
                    );
                }

                fn is_infinite<const LANES: usize>() {
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::is_infinite,
                        &Scalar::is_infinite,
                        &|_| true,
                    );
                }

                fn is_nan<const LANES: usize>() {
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::is_nan,
                        &Scalar::is_nan,
                        &|_| true,
                    );
                }

                fn is_normal<const LANES: usize>() {
                    // Arm v7 Neon violates float opsem re: subnormals, see
                    // https://github.com/rust-lang/portable-simd/issues/439
                    #[cfg(not(target_arch = "arm"))]
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::is_normal,
                        &Scalar::is_normal,
                        &|_| true,
                    );
                }

                fn is_subnormal<const LANES: usize>() {
                    // Arm v7 Neon violates float opsem re: subnormals, see
                    // https://github.com/rust-lang/portable-simd/issues/439
                    #[cfg(not(target_arch = "arm"))]
                    test_helpers::test_unary_mask_elementwise(
                        &Vector::<LANES>::is_subnormal,
                        &Scalar::is_subnormal,
                        &|_| true,
                    );
                }

                fn abs<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::abs,
                        &Scalar::abs,
                        &|_| true,
                    )
                }

                fn recip<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::recip,
                        &Scalar::recip,
                        &|_| true,
                    )
                }

                fn to_degrees<const LANES: usize>() {
                    test_helpers::test_unary_elementwise_flush_subnormals(
                        &Vector::<LANES>::to_degrees,
                        &Scalar::to_degrees,
                        &|_| true,
                    )
                }

                fn to_radians<const LANES: usize>() {
                    test_helpers::test_unary_elementwise_flush_subnormals(
                        &Vector::<LANES>::to_radians,
                        &Scalar::to_radians,
                        &|_| true,
                    )
                }

                fn signum<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::signum,
                        &Scalar::signum,
                        &|_| true,
                    )
                }

                fn copysign<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        &Vector::<LANES>::copysign,
                        &Scalar::copysign,
                        &|_, _| true,
                    )
                }

                fn simd_min<const LANES: usize>() {
                    // Regular conditions (both values aren't zero)
                    test_helpers::test_binary_elementwise(
                        &Vector::<LANES>::simd_min,
                        &Scalar::min,
                        // Reject the case where both values are zero with different signs
                        &|a, b| {
                            for (a, b) in a.iter().zip(b.iter()) {
                                if *a == 0. && *b == 0. && a.signum() != b.signum() {
                                    return false;
                                }
                            }
                            true
                        }
                    );

                    // Special case where both values are zero
                    let p_zero = Vector::<LANES>::splat(0.);
                    let n_zero = Vector::<LANES>::splat(-0.);
                    assert!(p_zero.simd_min(n_zero).to_array().iter().all(|x| *x == 0.));
                    assert!(n_zero.simd_min(p_zero).to_array().iter().all(|x| *x == 0.));
                }

                fn simd_max<const LANES: usize>() {
                    // Regular conditions (both values aren't zero)
                    test_helpers::test_binary_elementwise(
                        &Vector::<LANES>::simd_max,
                        &Scalar::max,
                        // Reject the case where both values are zero with different signs
                        &|a, b| {
                            for (a, b) in a.iter().zip(b.iter()) {
                                if *a == 0. && *b == 0. && a.signum() != b.signum() {
                                    return false;
                                }
                            }
                            true
                        }
                    );

                    // Special case where both values are zero
                    let p_zero = Vector::<LANES>::splat(0.);
                    let n_zero = Vector::<LANES>::splat(-0.);
                    assert!(p_zero.simd_max(n_zero).to_array().iter().all(|x| *x == 0.));
                    assert!(n_zero.simd_max(p_zero).to_array().iter().all(|x| *x == 0.));
                }

                fn simd_clamp<const LANES: usize>() {
                    if cfg!(all(target_arch = "powerpc64", target_feature = "vsx")) {
                        // https://gitlab.com/qemu-project/qemu/-/issues/1780
                        return;
                    }
                    test_helpers::test_3(&|value: [Scalar; LANES], mut min: [Scalar; LANES], mut max: [Scalar; LANES]| {
                        use test_helpers::subnormals::flush_in;
                        for (min, max) in min.iter_mut().zip(max.iter_mut()) {
                            if max < min {
                                core::mem::swap(min, max);
                            }
                            if min.is_nan() {
                                *min = Scalar::NEG_INFINITY;
                            }
                            if max.is_nan() {
                                *max = Scalar::INFINITY;
                            }
                        }

                        let mut result_scalar = [Scalar::default(); LANES];
                        for i in 0..LANES {
                            result_scalar[i] = value[i].clamp(min[i], max[i]);
                        }
                        let mut result_scalar_flush = [Scalar::default(); LANES];
                        for i in 0..LANES {
                            // Comparisons flush-to-zero, but return value selection is _not_ flushed.
                            let mut value = value[i];
                            if flush_in(value) < flush_in(min[i]) {
                                value = min[i];
                            }
                            if flush_in(value) > flush_in(max[i]) {
                                value = max[i];
                            }
                            result_scalar_flush[i] = value
                        }
                        let result_vector = Vector::from_array(value).simd_clamp(min.into(), max.into()).to_array();
                        test_helpers::prop_assert_biteq!(result_vector, result_scalar, result_scalar_flush);
                        Ok(())
                    })
                }

                fn reduce_sum<const LANES: usize>() {
                    test_helpers::test_1(&|x| {
                        test_helpers::prop_assert_biteq! (
                            Vector::<LANES>::from_array(x).reduce_sum(),
                            x.iter().sum(),
                        );
                        Ok(())
                    });
                }

                fn algebraic_reduce_sum<const LANES: usize>() {
                    test_helpers::test_1(&|x: [Scalar; LANES]| {
                        // Keep every possible reassociation within a finite range. In
                        // particular, f16 can overflow an intermediate pairwise sum even
                        // when the sequential scalar fold remains finite. Normalize the
                        // generated values instead of rejecting cases, which can exhaust
                        // proptest's global rejection limit for wider vectors.
                        let x = x.map(|value| {
                            if value.is_finite() {
                                value.clamp(-1.0, 1.0)
                            } else {
                                0.0
                            }
                        });
                        let actual = Vector::<LANES>::from_array(x).algebraic_reduce_sum();
                        let expected = x.iter().copied().fold(0.0, Scalar::algebraic_add);

                        // ULP distance is unstable around zero, especially for f16
                        // under Miri. Bound the absolute error by the input magnitude
                        // instead, which remains meaningful when cancellation occurs.
                        let magnitude = x.iter().copied().map(Scalar::abs).sum::<Scalar>();
                        // Keep a nonzero tolerance when the input magnitude is subnormal:
                        // multiplying a subnormal by EPSILON can underflow to zero.
                        let tolerance = magnitude.max(Scalar::MIN_POSITIVE) * Scalar::EPSILON * 128.0;
                        proptest::prop_assert!(
                            (actual - expected).abs() <= tolerance,
                            "actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}",
                        );
                        Ok(())
                    });
                }

                fn reduce_product<const LANES: usize>() {
                    test_helpers::test_1(&|x| {
                        test_helpers::prop_assert_biteq! (
                            Vector::<LANES>::from_array(x).reduce_product(),
                            x.iter().product(),
                        );
                        Ok(())
                    });
                }

                fn reduce_max<const LANES: usize>() {
                    test_helpers::test_1(&|x| {
                        let vmax = Vector::<LANES>::from_array(x).reduce_max();
                        let smax = x.iter().copied().fold(Scalar::NAN, Scalar::max);
                        // 0 and -0 are treated the same
                        if !(x.contains(&0.) && x.contains(&-0.) && vmax.abs() == 0. && smax.abs() == 0.) {
                            test_helpers::prop_assert_biteq!(vmax, smax);
                        }
                        Ok(())
                    });
                }

                fn reduce_min<const LANES: usize>() {
                    test_helpers::test_1(&|x| {
                        let vmax = Vector::<LANES>::from_array(x).reduce_min();
                        let smax = x.iter().copied().fold(Scalar::NAN, Scalar::min);
                        // 0 and -0 are treated the same
                        if !(x.contains(&0.) && x.contains(&-0.) && vmax.abs() == 0. && smax.abs() == 0.) {
                            test_helpers::prop_assert_biteq!(vmax, smax);
                        }
                        Ok(())
                    });
                }
            }

            #[cfg(feature = "std")]
            mod std {
                use std_float::StdFloat;

                use super::*;
                test_helpers::test_lanes! {
                    fn sqrt<const LANES: usize>() {
                        test_helpers::test_unary_elementwise(
                            &Vector::<LANES>::sqrt,
                            &Scalar::sqrt,
                            &|_| true,
                        )
                    }

                    fn mul_add<const LANES: usize>() {
                        test_helpers::test_ternary_elementwise(
                            &Vector::<LANES>::mul_add,
                            &Scalar::mul_add,
                            &|_, _, _| true,
                        )
                    }
                }
            }
        }
    }
}
