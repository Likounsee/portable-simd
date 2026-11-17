#![feature(portable_simd)]

//! Regression tests for the boolean ordering semantics of SIMD masks.
//!
//! See https://github.com/rust-lang/portable-simd/issues/527.

use core_simd::simd::{
    Mask,
    cmp::{SimdOrd, SimdPartialOrd},
};

#[test]
fn mask_partial_order_matches_bool_ordering_for_64_lanes() {
    // Include the sign bit and mixed patterns. Comparing the underlying signed
    // mask lanes as integers would reverse the ordering of true and false.
    let cases = [
        (0, 0),
        (u64::MAX, 0),
        (0, u64::MAX),
        (0x8000_0000_0000_0000, 0),
        (0, 0x8000_0000_0000_0000),
        (0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555),
        (0xF0F0_0F0F_AAAA_5555, 0x0F0F_F0F0_5555_AAAA),
        (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
    ];

    for (a_bits, b_bits) in cases {
        let a = Mask::<i8, 64>::from_bitmask(a_bits);
        let b = Mask::<i8, 64>::from_bitmask(b_bits);

        assert_eq!(
            a.simd_lt(b).to_bitmask(),
            !a_bits & b_bits,
            "lt: {a_bits:#018x}, {b_bits:#018x}"
        );
        assert_eq!(
            a.simd_le(b).to_bitmask(),
            !a_bits | b_bits,
            "le: {a_bits:#018x}, {b_bits:#018x}"
        );
        assert_eq!(
            a.simd_gt(b).to_bitmask(),
            a_bits & !b_bits,
            "gt: {a_bits:#018x}, {b_bits:#018x}"
        );
        assert_eq!(
            a.simd_ge(b).to_bitmask(),
            a_bits | !b_bits,
            "ge: {a_bits:#018x}, {b_bits:#018x}"
        );

        assert_eq!(
            a.simd_max(b).to_bitmask(),
            a_bits | b_bits,
            "max: {a_bits:#018x}, {b_bits:#018x}"
        );
        assert_eq!(
            a.simd_min(b).to_bitmask(),
            a_bits & b_bits,
            "min: {a_bits:#018x}, {b_bits:#018x}"
        );
    }
}

#[test]
fn mask_partial_order_exhaustive_for_8_lanes() {
    // Exhaust all 256 x 256 pairs of 8-lane masks. This checks every possible
    // pair of boolean lane patterns, complementing the 64-lane sign-bit cases.
    for a_bits in 0..=u8::MAX {
        for b_bits in 0..=u8::MAX {
            let a = Mask::<i8, 8>::from_bitmask(a_bits as u64);
            let b = Mask::<i8, 8>::from_bitmask(b_bits as u64);
            let a_bits = a_bits as u64;
            let b_bits = b_bits as u64;
            let lane_mask = u8::MAX as u64;

            assert_eq!(a.simd_lt(b).to_bitmask(), (!a_bits & b_bits) & lane_mask);
            assert_eq!(a.simd_le(b).to_bitmask(), (!a_bits | b_bits) & lane_mask);
            assert_eq!(a.simd_gt(b).to_bitmask(), (a_bits & !b_bits) & lane_mask);
            assert_eq!(a.simd_ge(b).to_bitmask(), (a_bits | !b_bits) & lane_mask);
            assert_eq!(a.simd_max(b).to_bitmask(), (a_bits | b_bits) & lane_mask);
            assert_eq!(a.simd_min(b).to_bitmask(), (a_bits & b_bits) & lane_mask);
        }
    }
}
