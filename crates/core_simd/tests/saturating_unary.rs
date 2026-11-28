#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::SimdInt;

macro_rules! signed_saturating_unary_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn saturating_abs_matches_scalar_at_signed_boundaries<const LANES: usize>() {
                        let values: [$ty; LANES] = core::array::from_fn(|lane| match lane % 6 {
                            0 => <$ty>::MIN,
                            1 => <$ty>::MIN + 1,
                            2 => -1,
                            3 => 0,
                            4 => 1,
                            _ => <$ty>::MAX,
                        });

                        let actual = Simd::<$ty, LANES>::from_array(values)
                            .saturating_abs()
                            .to_array();
                        let expected = values.map(<$ty>::saturating_abs);

                        assert_eq!(actual, expected);
                    }

                    fn saturating_neg_matches_scalar_at_signed_boundaries<const LANES: usize>() {
                        let values: [$ty; LANES] = core::array::from_fn(|lane| match lane % 6 {
                            0 => <$ty>::MIN,
                            1 => <$ty>::MIN + 1,
                            2 => -1,
                            3 => 0,
                            4 => 1,
                            _ => <$ty>::MAX,
                        });

                        let actual = Simd::<$ty, LANES>::from_array(values)
                            .saturating_neg()
                            .to_array();
                        let expected = values.map(<$ty>::saturating_neg);

                        assert_eq!(actual, expected);
                    }
                }
            }
        )*
    };
}

signed_saturating_unary_tests! {
    i8_cases: i8,
    i16_cases: i16,
    i32_cases: i32,
    i64_cases: i64,
    isize_cases: isize,
}
