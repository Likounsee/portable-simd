#![feature(portable_simd)]
#![feature(f16)]
#![feature(cfg_target_has_reliable_f16_f128)]
#![expect(internal_features)]

use core_simd::simd::{Simd, num::SimdFloat};

macro_rules! float_portability_tests {
    ($module:ident, $float:ty, $bits:ty, $sign_mask:expr, $exp_mask:expr, $min_normal:expr $(,)?) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn classification_and_sign_bit_operations_match_scalar<const LANES: usize>() {
                    // Exercise IEEE-754 classes without performing arithmetic on NaNs.
                    // This keeps the test meaningful on targets that canonicalize NaNs
                    // during arithmetic (notably WebAssembly).
                    let bits: [$bits; LANES] = core::array::from_fn(|lane| match lane % 12 {
                        0 => 0,
                        1 => $sign_mask,
                        2 => 1,
                        3 => $sign_mask | 1,
                        4 => $min_normal,
                        5 => $sign_mask | $min_normal,
                        6 => $exp_mask - 1,
                        7 => $sign_mask | ($exp_mask - 1),
                        8 => $exp_mask,
                        9 => $sign_mask | $exp_mask,
                        10 => $exp_mask | 1,
                        _ => $sign_mask | $exp_mask | 1,
                    });
                    let values = Simd::<$float, LANES>::from_bits(Simd::from_array(bits));
                    let scalars = bits.map(<$float>::from_bits);

                    assert_eq!(values.is_nan().to_array(), scalars.map(|x| x.is_nan()));
                    assert_eq!(values.is_finite().to_array(), scalars.map(|x| x.is_finite()));
                    assert_eq!(values.is_infinite().to_array(), scalars.map(|x| x.is_infinite()));
                    assert_eq!(values.is_normal().to_array(), scalars.map(|x| x.is_normal()));
                    assert_eq!(values.is_subnormal().to_array(), scalars.map(|x| x.is_subnormal()));
                    assert_eq!(
                        values.is_sign_negative().to_array(),
                        bits.map(|x| x & $sign_mask != 0),
                    );
                    assert_eq!(
                        values.is_sign_positive().to_array(),
                        bits.map(|x| x & $sign_mask == 0),
                    );
                    assert_eq!(
                        values.abs().to_bits().to_array(),
                        bits.map(|x| x & !$sign_mask),
                    );

                    let sign_bits: [$bits; LANES] = core::array::from_fn(|lane| {
                        if lane % 2 == 0 { 0 } else { $sign_mask }
                    });
                    let signs = Simd::<$float, LANES>::from_bits(Simd::from_array(sign_bits));
                    assert_eq!(
                        values.copysign(signs).to_bits().to_array(),
                        core::array::from_fn(|lane| {
                            (bits[lane] & !$sign_mask) | sign_bits[lane]
                        }),
                    );
                }
            }
        }
    };
}

float_portability_tests!(f32_cases, f32, u32, 0x8000_0000, 0x7f80_0000, 0x0080_0000);
#[cfg(target_has_reliable_f16_math)]
float_portability_tests!(f16_cases, f16, u16, 0x8000, 0x7c00, 0x0400);

float_portability_tests!(
    f64_cases,
    f64,
    u64,
    0x8000_0000_0000_0000,
    0x7ff0_0000_0000_0000,
    0x0010_0000_0000_0000,
);
