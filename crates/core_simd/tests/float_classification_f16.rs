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
    fn f16_classification_handles_subnormals_zeros_normals_and_special_values() {
        let values = Simd::from_array([
            f16::from_bits(0x0001),
            f16::from_bits(0x8001),
            f16::from_bits(0x03ff),
            f16::MIN_POSITIVE,
            -f16::MIN_POSITIVE,
            0.0_f16,
            -0.0_f16,
            f16::INFINITY,
        ]);

        assert_eq!(
            values.to_mask().to_array(),
            [true, true, true, true, true, false, false, true]
        );
        assert_eq!(
            values.is_subnormal().to_array(),
            [true, true, true, false, false, false, false, false]
        );
        assert_eq!(
            values.is_normal().to_array(),
            [false, false, false, true, true, false, false, false]
        );
        assert_eq!(
            values.is_finite().to_array(),
            [true, true, true, true, true, true, true, false]
        );
    }

    #[test]
    fn f16_classification_distinguishes_infinities_and_nan() {
        let values = Simd::from_array([
            f16::NEG_INFINITY,
            f16::INFINITY,
            f16::from_bits(0x7e11),
            f16::from_bits(0xfe22),
            -f16::MAX,
            f16::MAX,
            0.0_f16,
            -0.0_f16,
        ]);

        assert_eq!(
            values.is_nan().to_array(),
            [false, false, true, true, false, false, false, false]
        );
        assert_eq!(
            values.is_infinite().to_array(),
            [true, true, false, false, false, false, false, false]
        );
        assert_eq!(
            values.is_finite().to_array(),
            [false, false, false, false, true, true, true, true]
        );
    }

    #[test]
    fn f16_sign_predicates_use_the_sign_bit_for_all_value_classes() {
        let values = Simd::from_array([
            f16::from_bits(0x0000),
            f16::from_bits(0x8000),
            f16::INFINITY,
            f16::NEG_INFINITY,
            f16::from_bits(0x7e11),
            f16::from_bits(0xfe22),
            f16::from_bits(0x0001),
            f16::from_bits(0x8001),
        ]);

        assert_eq!(
            values.is_sign_positive().to_array(),
            [true, false, true, false, true, false, true, false]
        );
        assert_eq!(
            values.is_sign_negative().to_array(),
            [false, true, false, true, false, true, false, true]
        );
    }

    #[test]
    fn f16_abs_copysign_and_signum_preserve_expected_bits() {
        let values = Simd::from_array([
            f16::from_bits(0x0000),
            f16::from_bits(0x8000),
            f16::INFINITY,
            f16::NEG_INFINITY,
            f16::from_bits(0x7e11),
            f16::from_bits(0xfe22),
            f16::from_bits(0x0001),
            f16::from_bits(0x8001),
        ]);
        let signs = Simd::from_array([
            -1.0_f16, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
        ]);
        let value_array = values.to_array();
        let sign_array = signs.to_array();

        let expected_abs = value_array.map(|value| value.to_bits() & 0x7fff);
        let expected_copy_sign = core::array::from_fn(|lane| {
            (value_array[lane].to_bits() & 0x7fff)
                | (sign_array[lane].to_bits() & 0x8000)
        });
        let expected_signum = value_array.map(|value| {
            if value.is_nan() {
                value.to_bits()
            } else {
                value.signum().to_bits()
            }
        });

        assert_eq!(values.abs().to_bits().to_array(), expected_abs);
        assert_eq!(
            values.copysign(signs).to_bits().to_array(),
            expected_copy_sign
        );
        assert_eq!(values.signum().to_bits().to_array(), expected_signum);
    }
}
