#![feature(portable_simd)]
#![feature(f16)]
#![expect(internal_features)]
#![feature(cfg_target_has_reliable_f16_f128)]

#[cfg(all(target_has_reliable_f16_math, not(all(target_arch = "wasm32", target_feature = "simd128"))))]
use core_simd::simd::{Simd, num::SimdFloat};

#[cfg(all(target_has_reliable_f16_math, not(all(target_arch = "wasm32", target_feature = "simd128"))))]
#[test]
fn float_classification_preserves_f16_subnormal_and_zero_lanes() {
    let values = Simd::from_array([
        f16::from_bits(0x0001),
        f16::from_bits(0x8001),
        0.0_f16,
        -0.0_f16,
        f16::INFINITY,
        f16::NEG_INFINITY,
        f16::NAN,
        -1.0_f16,
    ]);

    assert_eq!(
        values.to_mask().to_array(),
        [true, true, false, false, true, true, true, true],
    );
    assert_eq!(
        values.is_subnormal().to_array(),
        [true, true, false, false, false, false, false, false],
    );
    assert_eq!(
        values.is_normal().to_array(),
        [false, false, false, false, false, false, false, true],
    );
    assert_eq!(
        values.is_finite().to_array(),
        [true, true, true, true, false, false, false, true],
    );
    assert_eq!(
        values.is_nan().to_array(),
        [false, false, false, false, false, false, true, false],
    );
}
