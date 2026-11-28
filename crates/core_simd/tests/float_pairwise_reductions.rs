#![feature(portable_simd)]
#![cfg_attr(target_has_reliable_f16_math, feature(f16))]
#![feature(cfg_target_has_reliable_f16_f128)]
#![expect(internal_features)]

use core_simd::simd::{Simd, num::SimdFloat};

macro_rules! pairwise_reduction_tests {
    ($module:ident, $float:ty, $large:expr, $small:expr $(,)?) => {
        mod $module {
            use super::*;

            fn pairwise_sum<const LANES: usize>(mut values: [$float; LANES]) -> $float {
                let mut len = LANES;
                while len > 1 {
                    let pairs = len / 2;
                    for index in 0..pairs {
                        values[index] = values[index * 2] + values[index * 2 + 1];
                    }
                    if len % 2 == 1 {
                        values[pairs] = values[len - 1];
                    }
                    len = pairs + len % 2;
                }
                values[0]
            }

            fn pairwise_product<const LANES: usize>(mut values: [$float; LANES]) -> $float {
                let mut len = LANES;
                while len > 1 {
                    let pairs = len / 2;
                    for index in 0..pairs {
                        values[index] = values[index * 2] * values[index * 2 + 1];
                    }
                    if len % 2 == 1 {
                        values[pairs] = values[len - 1];
                    }
                    len = pairs + len % 2;
                }
                values[0]
            }

            test_helpers::test_lanes! {
                fn pairwise_sum_and_product_match_the_documented_tree<const LANES: usize>() {
                    let values: [$float; LANES] = core::array::from_fn(|lane| match lane % 4 {
                        0 => $large,
                        1 => $small,
                        2 => -$large,
                        _ => 2.0,
                    });
                    let vector = Simd::from_array(values);

                    assert_eq!(
                        vector.pairwise_reduce_sum().to_bits(),
                        pairwise_sum(values).to_bits(),
                        "pairwise sum, lanes={LANES}",
                    );
                    assert_eq!(
                        vector.pairwise_reduce_product().to_bits(),
                        pairwise_product(values).to_bits(),
                        "pairwise product, lanes={LANES}",
                    );
                }

                fn pairwise_reductions_preserve_single_lane_values<const LANES: usize>() {
                    let values: [$float; LANES] = core::array::from_fn(|lane| {
                        if lane == 0 { -3.5 } else { 1.0 }
                    });
                    let vector = Simd::from_array(values);
                    assert_eq!(
                        vector.pairwise_reduce_sum().to_bits(),
                        pairwise_sum(values).to_bits(),
                    );
                    assert_eq!(
                        vector.pairwise_reduce_product().to_bits(),
                        pairwise_product(values).to_bits(),
                    );
                }
            }
        }
    };
}

pairwise_reduction_tests!(f32_cases, f32, 1.0e20_f32, 1.0_f32);
pairwise_reduction_tests!(f64_cases, f64, 1.0e300_f64, 1.0_f64);

#[cfg(target_has_reliable_f16_math)]
pairwise_reduction_tests!(f16_cases, f16, 1.0e4_f16, 1.0_f16);
