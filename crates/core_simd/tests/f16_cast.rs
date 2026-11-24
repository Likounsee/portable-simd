#![feature(portable_simd)]
#![cfg_attr(target_has_reliable_f16_math, feature(f16))]
#![expect(internal_features)]
#![feature(cfg_target_has_reliable_f16_f128)]
#![allow(unused_features)]

// The wasm32 Firefox test runner currently cannot deserialize the generated
// f16 cast test module (an i32/i64 type mismatch in the Wasm code section).
// Keep these cast tests enabled on native targets until that toolchain issue
// is resolved.
#[cfg(all(target_has_reliable_f16_math, not(target_arch = "wasm32")))]
mod tests {
    use core_simd::simd::{Simd, num::SimdFloat};

    #[test]
    fn f16_to_f32_and_f64_matches_scalar_casts() {
        let values = [
            0.0_f16,
            -0.0,
            1.0,
            -2.5,
            0.00006103515625,
            0.000000059604645,
            65504.0,
            -65504.0,
        ];

        let as_f32 = Simd::from_array(values).cast::<f32>().to_array();
        let expected_f32 = values.map(|value| value as f32);
        assert_eq!(as_f32, expected_f32);

        let as_f64 = Simd::from_array(values).cast::<f64>().to_array();
        let expected_f64 = values.map(|value| value as f64);
        assert_eq!(as_f64, expected_f64);
    }

    #[test]
    fn f32_to_f16_matches_scalar_rounding() {
        let values = [
            0.0_f32,
            -0.0,
            1.0,
            -2.5,
            0.00006103515625,
            0.000000059604645,
            65504.0,
            -65504.0,
        ];

        let actual = Simd::from_array(values).cast::<f16>().to_array();
        let expected = values.map(|value| value as f16);
        assert_eq!(actual, expected);
    }

    #[test]
    fn f64_to_f16_matches_scalar_rounding() {
        let values = [
            0.0_f64,
            -0.0,
            1.0,
            -2.5,
            0.00006103515625,
            0.000000059604645,
            65504.0,
            -65504.0,
        ];

        let actual = Simd::from_array(values).cast::<f16>().to_array();
        let expected = values.map(|value| value as f16);
        assert_eq!(actual, expected);
    }
}
