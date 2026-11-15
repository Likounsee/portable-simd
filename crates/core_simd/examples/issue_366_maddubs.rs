#![feature(portable_simd)]

//! Portable fallback for the semantics of x86 `_mm_maddubs_epi16`
//! (issue #366).
//!
//! Each pair of unsigned bytes is multiplied by the corresponding signed byte,
//! adjacent products are added, and each sum is saturated to `i16`. This
//! expresses the operation with portable SIMD, but does not guarantee that
//! LLVM will select the x86 `pmaddubsw` instruction.
//!
//! Run with:
//!   cargo test -p core_simd --example issue_366_maddubs

use core_simd::simd::prelude::*;

/// Computes eight pairwise, signed-saturating sums from sixteen byte lanes.
#[inline]
pub fn maddubs(a: u8x16, b: i8x16) -> i16x8 {
    let products: i16x16 = a.cast() * b.cast();
    let even: i16x8 = simd_swizzle!(products, [0, 2, 4, 6, 8, 10, 12, 14]);
    let odd: i16x8 = simd_swizzle!(products, [1, 3, 5, 7, 9, 11, 13, 15]);
    even.saturating_add(odd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(a: [u8; 16], b: [i8; 16]) -> [i16; 8] {
        core::array::from_fn(|lane| {
            let i = lane * 2;
            let sum = i32::from(a[i]) * i32::from(b[i])
                + i32::from(a[i + 1]) * i32::from(b[i + 1]);
            sum.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
        })
    }

    #[test]
    fn matches_scalar_reference_including_saturation() {
        let cases = [
            ([0; 16], [0; 16]),
            ([u8::MAX; 16], [i8::MAX; 16]),
            ([u8::MAX; 16], [i8::MIN; 16]),
            ([u8::MAX; 16], [i8::MAX; 16]),
            ([u8::MAX; 16], [i8::MIN; 16]),
            (
                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                [127, 127, -128, -128, 127, -128, -128, 127, 127, 127, -128, -128, 127, -128, -128, 127],
            ),
            (
                [0, 255, 1, 254, 2, 253, 3, 252, 4, 251, 5, 250, 6, 249, 7, 248],
                [-128, 127, 127, -128, -1, 1, 2, -2, 3, -3, 64, -64, 127, -128, -128, 127],
            ),
        ];

        for (a, b) in cases {
            let actual = maddubs(u8x16::from_array(a), i8x16::from_array(b)).to_array();
            assert_eq!(actual, reference(a, b));
        }
    }

    #[test]
    fn covers_every_unsigned_and_signed_byte_value() {
        for base in 0..=u8::MAX {
            let a = core::array::from_fn(|lane| base.wrapping_add(lane as u8));
            let b = core::array::from_fn(|lane| (base as i8).wrapping_add(lane as i8));
            let actual = maddubs(u8x16::from_array(a), i8x16::from_array(b)).to_array();
            assert_eq!(actual, reference(a, b), "base={base}");
        }
    }
}
