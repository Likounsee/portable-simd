#![feature(portable_simd)]
use core::{fmt, ops::RangeInclusive};
use test_helpers::{biteq, make_runner, prop_assert_biteq};

fn swizzle_dyn_scalar_ver<const N: usize>(values: [u8; N], idxs: [u8; N]) -> [u8; N] {
    let mut array = [0; N];
    for (i, k) in idxs.into_iter().enumerate() {
        if (k as usize) < N {
            array[i] = values[k as usize];
        };
    }
    array
}

test_helpers::test_lanes! {
    fn swizzle_dyn<const N: usize>() {
        match_simd_with_fallback(
            &core_simd::simd::Simd::<u8, N>::swizzle_dyn,
            &swizzle_dyn_scalar_ver,
            &|_, _| true,
        );
    }
}

fn match_simd_with_fallback<Scalar, ScalarResult, Vector, VectorResult, const N: usize>(
    fv: &dyn Fn(Vector, Vector) -> VectorResult,
    fs: &dyn Fn([Scalar; N], [Scalar; N]) -> [ScalarResult; N],
    check: &dyn Fn([Scalar; N], [Scalar; N]) -> bool,
) where
    Scalar: Copy + fmt::Debug + SwizzleStrategy,
    ScalarResult: Copy + biteq::BitEq + fmt::Debug + SwizzleStrategy,
    Vector: Into<[Scalar; N]> + From<[Scalar; N]> + Copy,
    VectorResult: Into<[ScalarResult; N]> + From<[ScalarResult; N]> + Copy,
{
    test_swizzles_2(&|x: [Scalar; N], y: [Scalar; N]| {
        proptest::prop_assume!(check(x, y));
        let result_v: [ScalarResult; N] = fv(x.into(), y.into()).into();
        let result_s: [ScalarResult; N] = fs(x, y);
        crate::prop_assert_biteq!(result_v, result_s);
        Ok(())
    });
}

fn test_swizzles_2<A: fmt::Debug + SwizzleStrategy, B: fmt::Debug + SwizzleStrategy>(
    f: &dyn Fn(A, B) -> proptest::test_runner::TestCaseResult,
) {
    let mut runner = make_runner();
    runner
        .run(
            &(A::swizzled_strategy(), B::swizzled_strategy()),
            |(a, b)| f(a, b),
        )
        .unwrap();
}

pub trait SwizzleStrategy {
    type Strategy: proptest::strategy::Strategy<Value = Self>;
    fn swizzled_strategy() -> Self::Strategy;
}

impl SwizzleStrategy for u8 {
    type Strategy = RangeInclusive<u8>;
    fn swizzled_strategy() -> Self::Strategy {
        0..=u8::MAX
    }
}

impl<T: fmt::Debug + SwizzleStrategy, const N: usize> SwizzleStrategy for [T; N] {
    type Strategy = test_helpers::array::UniformArrayStrategy<T::Strategy, Self>;
    fn swizzled_strategy() -> Self::Strategy {
        Self::Strategy::new(T::swizzled_strategy())
    }
}

fn check_exhaustive_swizzle_indices<const N: usize>() {
    let values: [u8; N] =
        core::array::from_fn(|lane| (lane as u8).wrapping_mul(37).wrapping_add(11));
    let vector = core_simd::simd::Simd::from_array(values);

    // Exercise every possible u8 index in every output lane. This covers all
    // in-bounds indices as well as every out-of-bounds byte value, independent
    // of the randomized property-test seed.
    for start in 0..=u8::MAX {
        let indices: [u8; N] = core::array::from_fn(|lane| start.wrapping_add(lane as u8));
        let actual = vector
            .swizzle_dyn(core_simd::simd::Simd::from_array(indices))
            .to_array();
        let expected = swizzle_dyn_scalar_ver(values, indices);
        assert_eq!(actual, expected, "N={N}, start={start}");
    }
}

#[test]
fn swizzle_dyn_exhaustive_indices_at_width_boundaries() {
    check_exhaustive_swizzle_indices::<1>();
    check_exhaustive_swizzle_indices::<7>();
    check_exhaustive_swizzle_indices::<8>();
    check_exhaustive_swizzle_indices::<9>();
    check_exhaustive_swizzle_indices::<15>();
    check_exhaustive_swizzle_indices::<16>();
    check_exhaustive_swizzle_indices::<17>();
    check_exhaustive_swizzle_indices::<23>();
    check_exhaustive_swizzle_indices::<24>();
    check_exhaustive_swizzle_indices::<31>();
    check_exhaustive_swizzle_indices::<32>();
    check_exhaustive_swizzle_indices::<33>();
    check_exhaustive_swizzle_indices::<47>();
    check_exhaustive_swizzle_indices::<48>();
    check_exhaustive_swizzle_indices::<49>();
    check_exhaustive_swizzle_indices::<63>();
    check_exhaustive_swizzle_indices::<64>();
}

fn check_unchecked_swizzle_high_bit_indices<const LANES: usize>() {
    let values: [u8; LANES] =
        core::array::from_fn(|lane| (lane as u8).wrapping_mul(13).wrapping_add(7));
    let indices: [u8; LANES] = core::array::from_fn(|lane| match lane % 4 {
        0 => (lane % LANES) as u8,
        1 => 0x80,
        2 => 0xff,
        _ => (LANES - 1) as u8,
    });

    let actual = core_simd::simd::Simd::from_array(values)
        .swizzle_dyn_unchecked(core_simd::simd::Simd::from_array(indices))
        .to_array();
    let expected = swizzle_dyn_scalar_ver(values, indices);

    assert_eq!(actual, expected, "LANES={LANES}");
}

#[test]
fn swizzle_dyn_unchecked_accepts_high_bit_out_of_bounds_indices() {
    check_unchecked_swizzle_high_bit_indices::<1>();
    check_unchecked_swizzle_high_bit_indices::<7>();
    check_unchecked_swizzle_high_bit_indices::<8>();
    check_unchecked_swizzle_high_bit_indices::<9>();
    check_unchecked_swizzle_high_bit_indices::<15>();
    check_unchecked_swizzle_high_bit_indices::<16>();
    check_unchecked_swizzle_high_bit_indices::<17>();
    check_unchecked_swizzle_high_bit_indices::<24>();
    check_unchecked_swizzle_high_bit_indices::<31>();
    check_unchecked_swizzle_high_bit_indices::<32>();
    check_unchecked_swizzle_high_bit_indices::<33>();
    check_unchecked_swizzle_high_bit_indices::<48>();
    check_unchecked_swizzle_high_bit_indices::<63>();
    check_unchecked_swizzle_high_bit_indices::<64>();
}

fn check_all_high_bit_swizzle_indices<const LANES: usize>() {
    let values = core::array::from_fn(|lane| (lane as u8).wrapping_mul(29).wrapping_add(3));
    let vector = core_simd::simd::Simd::from_array(values);

    // Every out-of-range byte with its most significant bit set must zero the
    // corresponding output lane, including on the SSSE3 u8x16 fast path.
    for index in 0x80_u8..=u8::MAX {
        let indices = core_simd::simd::Simd::splat(index);
        assert_eq!(
            vector.swizzle_dyn_unchecked(indices).to_array(),
            [0; LANES],
            "LANES={LANES}, index={index:#04x}"
        );
    }
}

#[test]
fn swizzle_dyn_unchecked_zeros_every_high_bit_index() {
    check_all_high_bit_swizzle_indices::<1>();
    check_all_high_bit_swizzle_indices::<7>();
    check_all_high_bit_swizzle_indices::<8>();
    check_all_high_bit_swizzle_indices::<9>();
    check_all_high_bit_swizzle_indices::<15>();
    check_all_high_bit_swizzle_indices::<16>();
    check_all_high_bit_swizzle_indices::<17>();
    check_all_high_bit_swizzle_indices::<24>();
    check_all_high_bit_swizzle_indices::<31>();
    check_all_high_bit_swizzle_indices::<32>();
    check_all_high_bit_swizzle_indices::<33>();
    check_all_high_bit_swizzle_indices::<48>();
    check_all_high_bit_swizzle_indices::<63>();
    check_all_high_bit_swizzle_indices::<64>();
}
