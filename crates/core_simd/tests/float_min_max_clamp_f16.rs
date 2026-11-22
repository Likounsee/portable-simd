#![feature(portable_simd)]
#![feature(f16)]
#![expect(internal_features)]
#![feature(cfg_target_has_reliable_f16_f128)]
#![allow(unused_features)]

// FIXME: f16 SIMD operations can hang on wasm simd; keep this suite gated
// consistently with the existing f16 operation tests.
#[cfg(all(
    target_has_reliable_f16_math,
    not(all(target_arch = "wasm32", target_feature = "simd128"))
))]
mod tests {
    use core_simd::simd::{Simd, num::SimdFloat};

    #[test]
    fn simd_min_matches_scalar_nan_handling_for_f16() {
        let left_vector = Simd::from_array([
            f16::from_bits(0x7e11),
            -5.0_f16,
            3.0,
            f16::INFINITY,
            f16::NEG_INFINITY,
            0.5,
            -2.0,
            8.0,
        ]);
        let right_vector = Simd::from_array([1.0_f16, f16::NAN, 4.0, 7.0, -7.0, -0.5, 2.0, 9.0]);
        let left = left_vector.to_array();
        let right = right_vector.to_array();
        let expected = core::array::from_fn(|i| left[i].min(right[i]).to_bits());

        assert_eq!(
            left_vector.simd_min(right_vector).to_bits().to_array(),
            expected
        );
    }

    #[test]
    fn simd_max_matches_scalar_nan_handling_for_f16() {
        let left_vector = Simd::from_array([
            f16::from_bits(0x7e11),
            -5.0_f16,
            3.0,
            f16::INFINITY,
            f16::NEG_INFINITY,
            0.5,
            -2.0,
            8.0,
        ]);
        let right_vector = Simd::from_array([1.0_f16, f16::NAN, 4.0, 7.0, -7.0, -0.5, 2.0, 9.0]);
        let left = left_vector.to_array();
        let right = right_vector.to_array();
        let expected = core::array::from_fn(|i| left[i].max(right[i]).to_bits());

        assert_eq!(
            left_vector.simd_max(right_vector).to_bits().to_array(),
            expected
        );
    }

    #[test]
    fn simd_clamp_matches_scalar_behavior_for_f16() {
        let values_vector = Simd::from_array([
            f16::from_bits(0x7e11),
            f16::NEG_INFINITY,
            -3.0_f16,
            -1.0,
            0.0,
            1.0,
            3.0,
            f16::INFINITY,
        ]);
        let min = [-2.0_f16, -2.0, -2.0, -1.0, -1.0, 0.0, 2.0, 2.0];
        let max = [2.0_f16, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let values = values_vector.to_array();
        let expected = core::array::from_fn(|i| values[i].clamp(min[i], max[i]).to_bits());

        assert_eq!(
            values_vector
                .simd_clamp(Simd::from_array(min), Simd::from_array(max))
                .to_bits()
                .to_array(),
            expected
        );
    }

    #[test]
    fn simd_min_and_max_allow_either_signed_zero_for_f16() {
        let left = Simd::from_array([-0.0_f16, 0.0, -0.0, 0.0]);
        let right = Simd::from_array([0.0_f16, -0.0, -0.0, 0.0]);
        assert_eq!(left.simd_min(right).to_array(), [0.0_f16; 4]);
        assert_eq!(left.simd_max(right).to_array(), [0.0_f16; 4]);
    }

}
