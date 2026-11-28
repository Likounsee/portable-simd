#![feature(portable_simd)]

use core_simd::simd::{Mask, Simd};

// Regression coverage for rust-lang/portable-simd#290. The original report
// failed during release LTO with LLVM 14 when lowering f64x8 scatter_select.
fn scatter_f64x8(target_slice: &mut [f64]) {
    let values = Simd::<f64, 8>::splat(1.0);
    values.scatter_select(target_slice, Mask::splat(true), Simd::splat(1));
}

#[test]
fn f64x8_scatter_select_writes_the_selected_element() {
    let mut values = vec![0.0_f64; 129];
    scatter_f64x8(&mut values);

    assert_eq!(values[1], 1.0);
    assert!(
        values
            .iter()
            .enumerate()
            .all(|(index, value)| { index == 1 || *value == 0.0 })
    );
}
