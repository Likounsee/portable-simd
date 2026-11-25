#![feature(f16, portable_simd)]

use core_simd::simd::{Simd, num::SimdFloat};

macro_rules! float_bitwise_tests {
    ($module:ident, $float:ty, $bits:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn bitwise_operators_preserve_ieee_bit_patterns<const LANES: usize>() {
                    let lhs_bits: [$bits; LANES] = core::array::from_fn(|lane| {
                        <$bits>::MAX.wrapping_sub((lane as $bits).wrapping_mul(13))
                    });
                    let rhs_bits: [$bits; LANES] = core::array::from_fn(|lane| {
                        (lane as $bits).wrapping_mul(29).wrapping_add(0x35 as $bits)
                    });
                    let expected_and =
                        core::array::from_fn(|lane| lhs_bits[lane] & rhs_bits[lane]);
                    let expected_or =
                        core::array::from_fn(|lane| lhs_bits[lane] | rhs_bits[lane]);
                    let expected_xor =
                        core::array::from_fn(|lane| lhs_bits[lane] ^ rhs_bits[lane]);
                    let expected_not = lhs_bits.map(|bits| !bits);

                    let lhs = Simd::<$float, LANES>::from_bits(Simd::from_array(lhs_bits));
                    let rhs = Simd::<$float, LANES>::from_bits(Simd::from_array(rhs_bits));

                    assert_eq!((lhs & rhs).to_bits().to_array(), expected_and);
                    assert_eq!((lhs | rhs).to_bits().to_array(), expected_or);
                    assert_eq!((lhs ^ rhs).to_bits().to_array(), expected_xor);
                    assert_eq!((!lhs).to_bits().to_array(), expected_not);

                    let mut and_assign = lhs;
                    and_assign &= rhs;
                    assert_eq!(and_assign.to_bits().to_array(), expected_and);

                    let mut or_assign = lhs;
                    or_assign |= rhs;
                    assert_eq!(or_assign.to_bits().to_array(), expected_or);

                    let mut xor_assign = lhs;
                    xor_assign ^= rhs;
                    assert_eq!(xor_assign.to_bits().to_array(), expected_xor);
                }
            }
        }
    };
}

float_bitwise_tests!(f16, f16, u16);
float_bitwise_tests!(f32, f32, u32);
float_bitwise_tests!(f64, f64, u64);
