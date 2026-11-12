//! Regression coverage for safe masked SIMD memory operations.
//!
//! These tests compare the safe APIs with scalar slice semantics, including empty
//! and short slices, disabled lanes, and out-of-bounds gather/scatter indices.

use core_simd::simd::{Mask, Simd};

#[test]
fn load_select_matches_scalar_slice_semantics_for_short_slices() {
    const LANES: usize = 4;
    let source = [10_i32, 20, 30, 40];
    let fallback = Simd::from_array([-1, -2, -3, -4]);
    let masks = [
        [false, false, false, false],
        [true, true, true, true],
        [true, false, true, false],
        [false, true, false, true],
    ];

    for len in 0..=LANES {
        for enabled in masks {
            let result = Simd::<i32, LANES>::load_select(
                &source[..len],
                Mask::from_array(enabled),
                fallback,
            )
            .to_array();

            let expected = core::array::from_fn(|lane| {
                if enabled[lane] && lane < len {
                    source[lane]
                } else {
                    fallback.to_array()[lane]
                }
            });
            assert_eq!(result, expected, "len={len}, enabled={enabled:?}");
        }
    }
}

#[test]
fn store_select_never_writes_disabled_or_out_of_bounds_lanes() {
    const LANES: usize = 4;
    let values = Simd::from_array([11_i32, 22, 33, 44]);
    let masks = [
        [false, false, false, false],
        [true, true, true, true],
        [true, false, true, false],
        [false, true, false, true],
    ];

    for len in 0..=LANES {
        for enabled in masks {
            let mut destination = [-1_i32; LANES];
            values.store_select(
                &mut destination[..len],
                Mask::from_array(enabled),
            );

            let expected = core::array::from_fn(|lane| {
                if lane < len && enabled[lane] {
                    values.to_array()[lane]
                } else {
                    -1
                }
            });
            assert_eq!(destination, expected, "len={len}, enabled={enabled:?}");
        }
    }
}

#[test]
fn gather_select_suppresses_disabled_and_out_of_bounds_indices() {
    let source = [10_i32, 20, 30];
    let indices = Simd::from_array([usize::MAX, 2, 0, 3]);
    let enabled = Mask::from_array([true, false, true, true]);
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    let result = Simd::gather_select(&source, enabled, indices, fallback);
    assert_eq!(result.to_array(), [-1, -2, 10, -4]);
}

#[test]
fn scatter_select_suppresses_disabled_and_out_of_bounds_indices() {
    let mut destination = [0_i32; 3];
    let values = Simd::from_array([11_i32, 22, 33, 44]);
    let indices = Simd::from_array([usize::MAX, 2, 0, 1]);
    let enabled = Mask::from_array([true, true, false, true]);

    values.scatter_select(&mut destination, enabled, indices);
    assert_eq!(destination, [0, 44, 22]);
}

#[test]
fn gather_or_default_handles_empty_slices_and_max_indices() {
    let source: [i32; 0] = [];
    let indices = Simd::from_array([0_usize, 1, usize::MAX, 42]);

    let result = Simd::gather_or_default(&source, indices);
    assert_eq!(result.to_array(), [0, 0, 0, 0]);
}
