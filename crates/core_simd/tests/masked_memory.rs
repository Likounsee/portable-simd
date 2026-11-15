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
fn contiguous_all_enabled_loads_copy_only_the_available_prefix() {
    const LANES: usize = 8;
    let source = [10_i32, 20, 30, 40, 50, 60, 70, 80, 90];
    let fallback = Simd::splat(-1_i32);

    for len in 0..=source.len() {
        let actual = Simd::<i32, LANES>::load_or_default(&source[..len]).to_array();
        let expected = core::array::from_fn(|lane| if lane < len { source[lane] } else { 0 });
        assert_eq!(actual, expected, "load_or_default: len={len}");

        let actual = Simd::<i32, LANES>::load_or(&source[..len], fallback).to_array();
        let expected = core::array::from_fn(|lane| if lane < len { source[lane] } else { -1 });
        assert_eq!(actual, expected, "load_or: len={len}");
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
fn all_enabled_full_width_store_preserves_the_destination_tail() {
    const LANES: usize = 8;
    let values = Simd::from_array([11_i32, 22, 33, 44, 55, 66, 77, 88]);
    let mut destination = [-1_i32; LANES + 2];

    values.store_select(&mut destination, Mask::splat(true));

    assert_eq!(&destination[..LANES], values.as_array());
    assert_eq!(&destination[LANES..], &[-1, -1]);
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
fn gather_unchecked_reads_all_valid_indices() {
    let source = [10_i32, 20, 30, 40, 50];
    let indices = Simd::from_array([4_usize, 0, 3, 1]);

    // SAFETY: every index is strictly less than source.len().
    let actual = unsafe { Simd::gather_unchecked(&source, indices) };

    assert_eq!(actual, Simd::from_array([50, 10, 40, 20]));
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
        let indices =
            Simd::from_array(indices.map(|index| if index == 4 { usize::MAX } else { index }));

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
    let patterns: [[bool; LANES]; 6] = [
        core::array::from_fn(|_| false),
        core::array::from_fn(|_| true),
        core::array::from_fn(|lane| lane % 2 == 0),
        core::array::from_fn(|lane| lane == 0 || lane + 1 == LANES),
        core::array::from_fn(|lane| lane < LANES / 2),
        core::array::from_fn(|lane| lane + 1 < LANES),
    ];

    for len in 0..=LANES {
        // Exercise every possible prefix length, not only half-width and
        // all-but-one masks. This covers the complete prefix fast-path boundary.
        for prefix_len in 0..=LANES {
            let enabled = core::array::from_fn(|lane| lane < prefix_len);
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
                actual, expected,
                "prefix load: LANES={LANES}, len={len}, prefix_len={prefix_len}"
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
            assert_eq!(
                destination, expected,
                "prefix store: LANES={LANES}, len={len}, prefix_len={prefix_len}"
            );
        }

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
                actual, expected,
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
            assert_eq!(
                destination, expected,
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
                    source[..len]
                        .get(index_array[lane])
                        .copied()
                        .unwrap_or(fallback_array[lane])
                } else {
                    fallback_array[lane]
                }
            });
            assert_eq!(
                actual, expected,
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
                destination, expected,
                "scatter: LANES={LANES}, len={len}, mask={enabled:?}"
            );
        }
    }
}

fn check_gather_index_boundaries<const LANES: usize>() {
    let source: Vec<i32> = (0..LANES).map(|lane| lane as i32 * 19 + 3).collect();
    let fallback = Simd::from_array(core::array::from_fn(|lane| -100 - lane as i32));
    let patterns: [[bool; LANES]; 4] = [
        core::array::from_fn(|_| false),
        core::array::from_fn(|_| true),
        core::array::from_fn(|lane| lane % 2 == 0),
        core::array::from_fn(|lane| lane % 3 == 1),
    ];

    // Rotate every lane through valid, boundary, and maximally invalid indices.
    // Re-run at each slice length so the same index can move in and out of bounds.
    for len in 0..=LANES {
        for shift in 0..LANES.max(1) {
            let indices = Simd::from_array(core::array::from_fn(|lane| match (lane + shift) % 4 {
                0 => (lane + shift) % LANES,
                1 => len,
                2 => LANES,
                _ => usize::MAX,
            }));
            let index_array = indices.to_array();
            for enabled in patterns {
                let actual = Simd::gather_select(
                    &source[..len],
                    Mask::from_array(enabled),
                    indices,
                    fallback,
                )
                .to_array();
                let fallback_array = fallback.to_array();
                let expected = core::array::from_fn(|lane| {
                    if enabled[lane] {
                        source[..len]
                            .get(index_array[lane])
                            .copied()
                            .unwrap_or(fallback_array[lane])
                    } else {
                        fallback_array[lane]
                    }
                });
                assert_eq!(
                    actual, expected,
                    "gather boundaries: LANES={LANES}, len={len}, shift={shift}, indices={index_array:?}, mask={enabled:?}"
                );
            }
        }
    }
}

#[test]
fn gather_index_boundaries_cover_supported_lane_widths() {
    check_gather_index_boundaries::<1>();
    check_gather_index_boundaries::<2>();
    check_gather_index_boundaries::<3>();
    check_gather_index_boundaries::<4>();
    check_gather_index_boundaries::<7>();
    check_gather_index_boundaries::<8>();
    check_gather_index_boundaries::<9>();
    check_gather_index_boundaries::<15>();
    check_gather_index_boundaries::<16>();
    check_gather_index_boundaries::<17>();
    check_gather_index_boundaries::<31>();
    check_gather_index_boundaries::<32>();
    check_gather_index_boundaries::<33>();
    check_gather_index_boundaries::<63>();
    check_gather_index_boundaries::<64>();
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

fn check_all_enabled_slice_memory<const LANES: usize>() {
    let source: Vec<i32> = (0..LANES + 2).map(|lane| lane as i32 * 3 + 1).collect();
    let values: Simd<i32, LANES> = Simd::from_array(core::array::from_fn(|lane| source[lane]));
    let fallback = Simd::splat(-7_i32);

    // Exercise the prefix boundary on both sides of the vector width. This
    // covers the full-width fast path as well as the bounded short-slice path.
    for len in 0..=LANES + 2 {
        let expected_load = core::array::from_fn(|lane| if lane < len { source[lane] } else { -7 });

        assert_eq!(
            Simd::<i32, LANES>::load_or(&source[..len], fallback).to_array(),
            expected_load,
            "load_or: lanes={LANES}, len={len}"
        );
        assert_eq!(
            Simd::<i32, LANES>::load_select(&source[..len], Mask::splat(true), fallback,)
                .to_array(),
            expected_load,
            "load_select: lanes={LANES}, len={len}"
        );

        let mut destination = vec![-1_i32; LANES + 2];
        values.store_select(&mut destination[..len], Mask::splat(true));

        let expected_store: Vec<i32> = (0..LANES + 2)
            .map(|lane| {
                if lane < len && lane < LANES {
                    source[lane]
                } else {
                    -1
                }
            })
            .collect();
        assert_eq!(
            destination, expected_store,
            "store_select: lanes={LANES}, len={len}"
        );
    }
}

#[test]
fn all_enabled_slice_memory_operations_cover_supported_lane_widths() {
    check_all_enabled_slice_memory::<1>();
    check_all_enabled_slice_memory::<2>();
    check_all_enabled_slice_memory::<3>();
    check_all_enabled_slice_memory::<4>();
    check_all_enabled_slice_memory::<7>();
    check_all_enabled_slice_memory::<8>();
    check_all_enabled_slice_memory::<9>();
    check_all_enabled_slice_memory::<15>();
    check_all_enabled_slice_memory::<16>();
    check_all_enabled_slice_memory::<17>();
    check_all_enabled_slice_memory::<31>();
    check_all_enabled_slice_memory::<32>();
    check_all_enabled_slice_memory::<33>();
    check_all_enabled_slice_memory::<63>();
    check_all_enabled_slice_memory::<64>();
}

fn check_full_width_slice_offsets<const LANES: usize>() {
    let source: Vec<i32> = (0..LANES * 2 + 1)
        .map(|index| index as i32 * 5 + 3)
        .collect();
    let values: Simd<i32, LANES> = Simd::from_array(core::array::from_fn(|lane| 100 + lane as i32));
    let fallback = Simd::splat(-7_i32);

    // Shift the slice start by each element offset to exercise full-width
    // operations at different addresses within the same allocation.
    for offset in 0..LANES {
        let source_slice = &source[offset..offset + LANES + 1];
        let expected_load = core::array::from_fn(|lane| source[offset + lane]);
        assert_eq!(
            Simd::<i32, LANES>::load_or(source_slice, fallback).to_array(),
            expected_load,
            "load_or: lanes={LANES}, offset={offset}"
        );
        assert_eq!(
            Simd::<i32, LANES>::load_select(source_slice, Mask::splat(true), fallback,).to_array(),
            expected_load,
            "load_select: lanes={LANES}, offset={offset}"
        );

        let mut destination = vec![-99_i32; LANES * 2 + 1];
        values.store_select(
            &mut destination[offset..offset + LANES + 1],
            Mask::splat(true),
        );
        let expected_store: Vec<i32> = (0..destination.len())
            .map(|index| {
                if (offset..offset + LANES).contains(&index) {
                    values.as_array()[index - offset]
                } else {
                    -99
                }
            })
            .collect();
        assert_eq!(
            destination, expected_store,
            "store_select: lanes={LANES}, offset={offset}"
        );
    }
}

#[test]
fn all_enabled_full_width_slice_memory_handles_element_offsets() {
    check_full_width_slice_offsets::<1>();
    check_full_width_slice_offsets::<2>();
    check_full_width_slice_offsets::<3>();
    check_full_width_slice_offsets::<4>();
    check_full_width_slice_offsets::<7>();
    check_full_width_slice_offsets::<8>();
    check_full_width_slice_offsets::<9>();
    check_full_width_slice_offsets::<15>();
    check_full_width_slice_offsets::<16>();
    check_full_width_slice_offsets::<17>();
    check_full_width_slice_offsets::<31>();
    check_full_width_slice_offsets::<32>();
    check_full_width_slice_offsets::<33>();
    check_full_width_slice_offsets::<63>();
    check_full_width_slice_offsets::<64>();
}

#[test]
fn scatter_select_handles_duplicate_and_invalid_indices_for_every_mask() {
    const LANES: usize = 4;
    const DEST_LEN: usize = 3;
    let values = Simd::from_array([17_i32, 31, 53, 79]);

    // Each lane can target any valid destination, the first invalid index,
    // or usize::MAX. Exhaust every tuple and every enable mask.
    for encoded_indices in 0..5_usize.pow(LANES as u32) {
        let mut encoded = encoded_indices;
        let indices = core::array::from_fn(|_| {
            let digit = encoded % 5;
            encoded /= 5;
            if digit == 4 { usize::MAX } else { digit }
        });
        let indices_simd = Simd::from_array(indices);

        for mask_bits in 0..(1 << LANES) {
            let enabled = core::array::from_fn(|lane| mask_bits & (1 << lane) != 0);
            let mut actual = [-99_i32; DEST_LEN];
            values.scatter_select(&mut actual, Mask::from_array(enabled), indices_simd);

            for (destination, &actual_value) in actual.iter().enumerate() {
                let last_writer = (0..LANES)
                    .rev()
                    .find(|&lane| enabled[lane] && indices[lane] == destination);

                if let Some(lane) = last_writer {
                    assert_eq!(
                        actual_value,
                        values.as_array()[lane],
                        "scatter must retain the highest enabled lane: indices={indices:?}, mask={enabled:?}, destination={destination}"
                    );
                } else {
                    assert_eq!(
                        actual_value, -99,
                        "unexpected write: indices={indices:?}, mask={enabled:?}, destination={destination}"
                    );
                }
            }
        }
    }
}

fn check_scatter_duplicate_winners<const LANES: usize>() {
    let dest_len = LANES / 2 + 1;
    let values = Simd::from_array(core::array::from_fn(|lane| 100 + lane as i32));
    let patterns: [[bool; LANES]; 4] = [
        core::array::from_fn(|_| false),
        core::array::from_fn(|_| true),
        core::array::from_fn(|lane| lane % 2 == 0),
        core::array::from_fn(|lane| lane == 0 || lane + 1 == LANES),
    ];

    // Repeated valid destinations force collisions; the other lanes include
    // the exact end boundary and usize::MAX to exercise safe suppression.
    let indices = Simd::from_array(core::array::from_fn(|lane| match lane % 4 {
        0 | 1 => lane % dest_len,
        2 => dest_len,
        _ => usize::MAX,
    }));
    let index_array = indices.to_array();

    for len in 0..=dest_len {
        for enabled in patterns {
            let mut actual = [-99_i32; LANES];
            values.scatter_select(&mut actual[..len], Mask::from_array(enabled), indices);

            for (destination, &actual_value) in actual.iter().enumerate() {
                let last_writer = if destination < len {
                    (0..LANES)
                        .rev()
                        .find(|&lane| enabled[lane] && index_array[lane] == destination)
                } else {
                    None
                };

                if let Some(lane) = last_writer {
                    assert_eq!(
                        actual_value,
                        values.as_array()[lane],
                        "scatter winner: LANES={LANES}, len={len}, indices={index_array:?}, mask={enabled:?}, destination={destination}"
                    );
                } else {
                    assert_eq!(
                        actual_value, -99,
                        "unexpected scatter write: LANES={LANES}, len={len}, indices={index_array:?}, mask={enabled:?}, destination={destination}"
                    );
                }
            }
        }
    }
}

#[test]
fn scatter_duplicate_winners_cover_supported_lane_widths() {
    check_scatter_duplicate_winners::<1>();
    check_scatter_duplicate_winners::<2>();
    check_scatter_duplicate_winners::<3>();
    check_scatter_duplicate_winners::<4>();
    check_scatter_duplicate_winners::<7>();
    check_scatter_duplicate_winners::<8>();
    check_scatter_duplicate_winners::<9>();
    check_scatter_duplicate_winners::<15>();
    check_scatter_duplicate_winners::<16>();
    check_scatter_duplicate_winners::<17>();
    check_scatter_duplicate_winners::<31>();
    check_scatter_duplicate_winners::<32>();
    check_scatter_duplicate_winners::<33>();
    check_scatter_duplicate_winners::<63>();
    check_scatter_duplicate_winners::<64>();
}
