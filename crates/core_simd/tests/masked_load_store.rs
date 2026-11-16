#![feature(portable_simd)]
use core_simd::simd::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn masked_load_store() {
    let mut arr = [u8::MAX; 7];

    u8x4::splat(0).store_select(&mut arr[5..], Mask::from_array([false, true, false, true]));
    // write to index 8 is OOB and dropped
    assert_eq!(arr, [255u8, 255, 255, 255, 255, 255, 0]);

    u8x4::from_array([0, 1, 2, 3]).store_select(&mut arr[1..], Mask::splat(true));
    assert_eq!(arr, [255u8, 0, 1, 2, 3, 255, 0]);

    // read from index 8 is OOB and dropped
    assert_eq!(
        u8x4::load_or(&arr[4..], u8x4::splat(42)),
        u8x4::from_array([3, 255, 0, 42])
    );
    assert_eq!(
        u8x4::load_select(
            &arr[4..],
            Mask::from_array([true, false, true, true]),
            u8x4::splat(42)
        ),
        u8x4::from_array([3, 42, 0, 42])
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn masked_load_store_boundary_masks() {
    // An all-false mask must not access even an empty slice.
    let empty: [u32; 0] = [];
    assert_eq!(
        u32x4::load_select(&empty, Mask::splat(false), u32x4::splat(17)),
        u32x4::splat(17)
    );

    // An all-true mask exactly matching the slice must access every lane.
    let source = [10u32, 20, 30, 40];
    assert_eq!(
        u32x4::load_select(&source, Mask::splat(true), u32x4::splat(99)),
        u32x4::from_array(source)
    );

    // Store-select with no active lanes must leave the destination untouched.
    let mut destination = [1u32, 2, 3, 4];
    u32x4::splat(88).store_select(&mut destination, Mask::splat(false));
    assert_eq!(destination, [1, 2, 3, 4]);

    // A partial mask at the end of the slice must only write active in-bounds lanes.
    u32x4::from_array([5, 6, 7, 8]).store_select(
        &mut destination[2..],
        Mask::from_array([true, false, false, false]),
    );
    assert_eq!(destination, [1, 2, 5, 4]);
}
