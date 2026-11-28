#![feature(portable_simd)]

use core_simd::simd::{Mask, Select, Simd};

test_helpers::test_lanes! {
    fn compress_expand_round_trip_for_supported_lane_counts<const LANES: usize>() {
        let values_array: [i32; LANES] =
            core::array::from_fn(|lane| lane as i32 * 17 - 31);
        let fallback_array: [i32; LANES] =
            core::array::from_fn(|lane| -1000 - lane as i32);
        let values = Simd::from_array(values_array);
        let fallback = Simd::from_array(fallback_array);

        // Exercise sparse, dense, alternating, edge, and irregular masks for every
        // representative lane count, including odd and non-power-of-two vectors.
        for pattern in 0..6 {
            let selected: [bool; LANES] = core::array::from_fn(|lane| match pattern {
                0 => false,
                1 => true,
                2 => lane % 2 == 0,
                3 => lane % 2 == 1,
                4 => lane == 0 || lane + 1 == LANES,
                _ => lane % 3 == 1,
            });
            let mask = Mask::from_array(selected);
            let selected_count = selected.iter().filter(|&&value| value).count();

            let mut expected_compressed = fallback_array;
            let mut packed_count = 0;
            for lane in 0..LANES {
                if selected[lane] {
                    expected_compressed[packed_count] = values_array[lane];
                    packed_count += 1;
                }
            }

            let compressed = values.compress(mask, fallback);
            assert_eq!(
                compressed.to_array(),
                expected_compressed,
                "register compress: lanes={LANES}, pattern={pattern}"
            );
            assert_eq!(
                compressed.expand(mask, fallback),
                mask.select(values, fallback),
                "register expand round trip: lanes={LANES}, pattern={pattern}"
            );

            for capacity in 0..=LANES {
                let mut output = [i32::MIN; LANES];
                let written = values.compress_store(&mut output[..capacity], mask);
                let expected_written = selected_count.min(capacity);
                assert_eq!(
                    written, expected_written,
                    "compress count: lanes={LANES}, pattern={pattern}, capacity={capacity}"
                );

                let mut expected_output = [i32::MIN; LANES];
                let mut output_index = 0;
                for lane in 0..LANES {
                    if selected[lane] && output_index < expected_written {
                        expected_output[output_index] = values_array[lane];
                        output_index += 1;
                    }
                }
                assert_eq!(
                    output, expected_output,
                    "compress contents/bounds: lanes={LANES}, pattern={pattern}, capacity={capacity}"
                );

                let input: [i32; LANES] =
                    core::array::from_fn(|index| 10000 + index as i32);
                let (expanded, consumed) =
                    Simd::<i32, LANES>::expand_load(&input[..capacity], mask, fallback);
                assert_eq!(
                    consumed, expected_written,
                    "expand count: lanes={LANES}, pattern={pattern}, capacity={capacity}"
                );

                let mut expected_expanded = fallback_array;
                let mut input_index = 0;
                for lane in 0..LANES {
                    if selected[lane] && input_index < expected_written {
                        expected_expanded[lane] = input[input_index];
                        input_index += 1;
                    }
                }
                assert_eq!(
                    expanded.to_array(),
                    expected_expanded,
                    "expand contents/fallback: lanes={LANES}, pattern={pattern}, capacity={capacity}"
                );
            }
        }
    }
}
