#![feature(portable_simd)]
#![feature(f16)]
#![expect(internal_features)]
#![feature(cfg_target_has_reliable_f16_f128)]
#![allow(unused_features)]

use core_simd::simd::{Simd, num::SimdFloat};

#[cfg(all(
    target_has_reliable_f16_math,
    not(all(target_arch = "wasm32", target_feature = "simd128"))
))]
mod tests {
    use super::*;

    #[test]
    fn f16_reduce_sum_matches_ordered_scalar_fold() {
        let values = [
            1.0_f16,
            0.5,
            -0.25,
            2.0,
            -1.0,
            0.125,
            4.0,
            -2.0,
        ];
        let expected = values.into_iter().fold(0.0_f16, |sum, value| sum + value);

        assert_eq!(Simd::from_array(values).reduce_sum().to_bits(), expected.to_bits());
    }

    #[test]
    fn f16_reduce_product_matches_ordered_scalar_fold() {
        let values = [1.5_f16, 2.0, -0.5, 1.0, 0.25, 2.0, -1.0, 1.0];
        let expected = values.into_iter().fold(1.0_f16, |product, value| product * value);

        assert_eq!(
            Simd::from_array(values).reduce_product().to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    fn f16_reduce_min_and_max_skip_nan_lanes() {
        let values = Simd::from_array([
            f16::NAN,
            -3.0_f16,
            1.5,
            f16::INFINITY,
            f16::NEG_INFINITY,
            0.0,
            f16::from_bits(0x7e11),
            2.0,
        ]);

        assert_eq!(values.reduce_min(), f16::NEG_INFINITY);
        assert_eq!(values.reduce_max(), f16::INFINITY);
    }

    #[test]
    fn f16_reduce_min_and_max_return_nan_when_all_lanes_are_nan() {
        let values = Simd::from_array([
            f16::NAN,
            f16::from_bits(0x7e11),
            f16::from_bits(0xfe22),
            f16::NAN,
            f16::NAN,
            f16::NAN,
            f16::NAN,
            f16::NAN,
        ]);

        assert!(values.reduce_min().is_nan());
        assert!(values.reduce_max().is_nan());
    }

    #[test]
    fn f16_algebraic_reduce_sum_handles_exact_finite_values() {
        let values = [1.0_f16, 2.0, 4.0, 8.0, -0.5, 0.25, 0.125, 0.0];
        let expected = values.into_iter().sum::<f16>();

        assert_eq!(
            Simd::from_array(values).algebraic_reduce_sum().to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    fn f16_reduce_sum_preserves_order_for_cancellation() {
        let values = [2048.0_f16, -2048.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let expected = values.into_iter().fold(0.0_f16, |sum, value| sum + value);

        assert_eq!(
            Simd::from_array(values).reduce_sum().to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    fn f16_reduce_sum_propagates_invalid_infinity_sum() {
        let values = [
            f16::INFINITY,
            f16::NEG_INFINITY,
            1.0_f16,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let expected = values.into_iter().fold(0.0_f16, |sum, value| sum + value);
        let actual = Simd::from_array(values).reduce_sum();

        assert_eq!(actual.is_nan(), expected.is_nan());
    }

    #[test]
    fn f16_reduce_product_propagates_zero_times_infinity() {
        let values = [
            0.0_f16,
            f16::INFINITY,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ];
        let expected = values
            .into_iter()
            .fold(1.0_f16, |product, value| product * value);
        let actual = Simd::from_array(values).reduce_product();

        assert_eq!(actual.is_nan(), expected.is_nan());
    }
}
