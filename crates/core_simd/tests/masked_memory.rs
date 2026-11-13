#![feature(portable_simd)]

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
            values.store_select(&mut destination[..len], Mask::from_array(enabled));

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

#[test]
fn gather_select_matches_scalar_oracle_for_small_index_space() {
    const LANES: usize = 4;
    let source = [13_i32, 29, 47];
    let fallback = Simd::from_array([-1, -2, -3, -4]);

    // Enumerate every 4-lane index tuple in 0..=4. Index 3 is just
    // out of bounds and index 4 exercises a much larger invalid value below.
    for encoded_indices in 0..5_usize.pow(LANES as u32) {
        let mut encoded = encoded_indices;
        let indices = core::array::from_fn(|_| {
            let digit = encoded % 5;
            encoded /= 5;
            digit
        });
        let indices = Simd::from_array(indices.map(|index| {
            if index == 4 {
                usize::MAX
            } else {
                index
            }
        }));

        for mask_bits in 0..(1 << LANES) {
            let enabled = core::array::from_fn(|lane| mask_bits & (1 << lane) != 0);
            let mask = Mask::from_array(enabled);
            let actual = Simd::gather_select(&source, mask, indices, fallback).to_array();
            let index_array = indices.to_array();
            let expected = core::array::from_fn(|lane| {
                if enabled[lane] {
                    source
                        .get(index_array[lane])
                        .copied()
                        .unwrap_or(fallback.to_array()[lane])
                } else {
                    fallback.to_array()[lane]
                }
            });
            assert_eq!(
                actual, expected,
                "indices={index_array:?}, mask={enabled:?}"
            );
        }
    }
}

#[test]
fn scatter_select_matches_scalar_oracle_for_unique_valid_indices() {
    const LANES: usize = 4;
    let values = Simd::from_array([17_i32, 31, 53, 79]);

    // Enumerate every permutation of four distinct destinations in 0..4.
    for a in 0..LANES {
        for b in 0..LANES {
            for c in 0..LANES {
                for d in 0..LANES {
                    let indices = [a, b, c, d];
                    if (0..LANES).any(|i| (i + 1..LANES).any(|j| indices[i] == indices[j])) {
                        continue;
                    }

                    for mask_bits in 0..(1 << LANES) {
                        let enabled = core::array::from_fn(|lane| mask_bits & (1 << lane) != 0);
                        let mut actual = [-1_i32; LANES];
                        values.scatter_select(
                            &mut actual,
                            Mask::from_array(enabled),
                            Simd::from_array(indices),
                        );

                        let mut expected = [-1_i32; LANES];
                        for lane in 0..LANES {
                            if enabled[lane] {
                                expected[indices[lane]] = values.to_array()[lane];
                            }
                        }
                        assert_eq!(actual, expected, "indices={indices:?}, mask={enabled:?}");
                    }
                }
            }
        }
    }
}

fn check_contiguous_masked_memory_width<const LANES: usize>() {
    let source: [i32; LANES] =
        core::array::from_fn(|lane| (lane as i32).wrapping_mul(17).wrapping_add(5));
    let values = Simd::from_array(source);
    let fallback = Simd::from_array(core::array::from_fn(|lane| -1 - lane as i32));
    let patterns: [[bool; LANES]; 4] = [
        core::array::from_fn(|_| false),
        core::array::from_fn(|_| true),
        core::array::from_fn(|lane| lane % 2 == 0),
        core::array::from_fn(|lane| lane == 0 || lane + 1 == LANES),
    ];

    for len in 0..=LANES {
        for enabled in patterns {
            let actual = Simd::<i32, LANES>::load_select(
                &source[..len],
                Mask::from_array(enabled),
                fallback,
            )
            .to_array();
            let fallback_array = fallback.to_array();
            let expected = core::array::from_fn(|lane| {
                if enabled[lane] && lane < len {
                    source[lane]
                } else {
                    fallback_array[lane]
                }
            });
            assert_eq!(
                actual,
                expected,
                "load: LANES={LANES}, len={len}, mask={enabled:?}"
            );

            let mut destination = [-99_i32; LANES];
            values.store_select(&mut destination[..len], Mask::from_array(enabled));
            let expected = core::array::from_fn(|lane| {
                if enabled[lane] && lane < len {
                    source[lane]
                } else {
                    -99
                }
            });
            assert_eq!(destination, expected,
                "store: LANES={LANES}, len={len}, mask={enabled:?}"
            );
        }
    }
}

#[test]
fn contiguous_masked_memory_covers_lane_count_boundaries() {
    check_contiguous_masked_memory_width::<1>();
    check_contiguous_masked_memory_width::<2>();
    check_contiguous_masked_memory_width::<3>();
    check_contiguous_masked_memory_width::<4>();
    check_contiguous_masked_memory_width::<7>();
    check_contiguous_masked_memory_width::<8>();
    check_contiguous_masked_memory_width::<9>();
    check_contiguous_masked_memory_width::<15>();
    check_contiguous_masked_memory_width::<16>();
    check_contiguous_masked_memory_width::<17>();
    check_contiguous_masked_memory_width::<31>();
    check_contiguous_masked_memory_width::<32>();
    check_contiguous_masked_memory_width::<33>();
    check_contiguous_masked_memory_width::<63>();
    check_contiguous_masked_memory_width::<64>();
}

fn check_gather_scatter_masked_memory_width<const LANES: usize>() {
    let source: [i32; LANES] =
        core::array::from_fn(|lane| (lane as i32).wrapping_mul(23).wrapping_add(7));
    let values = Simd::from_array(core::array::from_fn(|lane| 100 + lane as i32));
    let fallback = Simd::from_array(core::array::from_fn(|lane| -10 - lane as i32));
    let indices = Simd::from_array(core::array::from_fn(|lane| {
        if lane % 2 == 0 { lane } else { usize::MAX }
    }));
    let index_array = indices.to_array();
    let patterns: [[bool; LANES]; 4] = [
        core::array::from_fn(|_| false),
        core::array::from_fn(|_| true),
        core::array::from_fn(|lane| lane % 2 == 0),
        core::array::from_fn(|lane| lane == 0 || lane + 1 == LANES),
    ];

    for len in 0..=LANES {
        for enabled in patterns {
            let actual =
                Simd::gather_select(&source[..len], Mask::from_array(enabled), indices, fallback)
                    .to_array();
            let fallback_array = fallback.to_array();
            let expected = core::array::from_fn(|lane| {
                if enabled[lane] {
                    source
                        .get(index_array[lane])
                        .copied()
                        .unwrap_or(fallback_array[lane])
                } else {
                    fallback_array[lane]
                }
            });
            assert_eq!(
                actual,
                expected,
                "gather: LANES={LANES}, len={len}, mask={enabled:?}"
            );

            let mut destination = [-77_i32; LANES];
            values.scatter_select(&mut destination[..len], Mask::from_array(enabled), indices);
            let mut expected = [-77_i32; LANES];
            for lane in 0..LANES {
                if enabled[lane] && index_array[lane] < len {
                    expected[index_array[lane]] = values.to_array()[lane];
                }
            }
            assert_eq!(
                destination,
                expected,
                "scatter: LANES={LANES}, len={len}, mask={enabled:?}"
            );
        }
    }
}

#[test]
fn gather_scatter_masked_memory_covers_lane_count_boundaries() {
    check_gather_scatter_masked_memory_width::<1>();
    check_gather_scatter_masked_memory_width::<2>();
    check_gather_scatter_masked_memory_width::<3>();
    check_gather_scatter_masked_memory_width::<4>();
    check_gather_scatter_masked_memory_width::<7>();
    check_gather_scatter_masked_memory_width::<8>();
    check_gather_scatter_masked_memory_width::<9>();
    check_gather_scatter_masked_memory_width::<15>();
    check_gather_scatter_masked_memory_width::<16>();
    check_gather_scatter_masked_memory_width::<17>();
    check_gather_scatter_masked_memory_width::<31>();
    check_gather_scatter_masked_memory_width::<32>();
    check_gather_scatter_masked_memory_width::<33>();
    check_gather_scatter_masked_memory_width::<63>();
    check_gather_scatter_masked_memory_width::<64>();
}
