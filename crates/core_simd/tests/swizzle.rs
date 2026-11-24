#![feature(portable_simd)]
use core_simd::simd::{Simd, Swizzle};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);


#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn dynamic_get_set() {
    let mut vector = Simd::from_array([10_u32, 20, 30, 40]);
    let index = core::hint::black_box(2usize);

    assert_eq!(vector.get(index), 30);
    vector.set(index, 99);
    assert_eq!(vector.to_array(), [10, 20, 99, 40]);

    let index = core::hint::black_box(3usize);
    // SAFETY: index is less than the four lanes in the vector.
    assert_eq!(unsafe { vector.get_unchecked(index) }, 40);
    // SAFETY: index is less than the four lanes in the vector.
    unsafe { vector.set_unchecked(index, 77) };
    assert_eq!(vector.to_array(), [10, 20, 99, 77]);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn dynamic_get_checks_bounds() {
    let vector = Simd::from_array([1_u32, 2, 3, 4]);
    let _ = vector.get(4);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn dynamic_set_checks_bounds() {
    let mut vector = Simd::from_array([1_u32, 2, 3, 4]);
    vector.set(4, 5);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn swizzle() {
    struct Index;
    impl Swizzle<4> for Index {
        const INDEX: [usize; 4] = [2, 1, 3, 0];
    }
    impl Swizzle<2> for Index {
        const INDEX: [usize; 2] = [1, 1];
    }

    let vector = Simd::from_array([2, 4, 1, 9]);
    assert_eq!(Index::swizzle(vector).to_array(), [1, 4, 9, 2]);
    assert_eq!(Index::swizzle(vector).to_array(), [4, 4]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn reverse() {
    let a = Simd::from_array([1, 2, 3, 4]);
    assert_eq!(a.reverse().to_array(), [4, 3, 2, 1]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn rotate() {
    let a = Simd::from_array([1, 2, 3, 4]);
    assert_eq!(a.rotate_elements_left::<0>().to_array(), [1, 2, 3, 4]);
    assert_eq!(a.rotate_elements_left::<1>().to_array(), [2, 3, 4, 1]);
    assert_eq!(a.rotate_elements_left::<2>().to_array(), [3, 4, 1, 2]);
    assert_eq!(a.rotate_elements_left::<3>().to_array(), [4, 1, 2, 3]);
    assert_eq!(a.rotate_elements_left::<4>().to_array(), [1, 2, 3, 4]);
    assert_eq!(a.rotate_elements_left::<5>().to_array(), [2, 3, 4, 1]);
    assert_eq!(a.rotate_elements_right::<0>().to_array(), [1, 2, 3, 4]);
    assert_eq!(a.rotate_elements_right::<1>().to_array(), [4, 1, 2, 3]);
    assert_eq!(a.rotate_elements_right::<2>().to_array(), [3, 4, 1, 2]);
    assert_eq!(a.rotate_elements_right::<3>().to_array(), [2, 3, 4, 1]);
    assert_eq!(a.rotate_elements_right::<4>().to_array(), [1, 2, 3, 4]);
    assert_eq!(a.rotate_elements_right::<5>().to_array(), [4, 1, 2, 3]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn rotate_runtime() {
    let vector = Simd::from_array([0, 1, 2, 3]);
    let amount = core::hint::black_box(2usize);
    assert_eq!(
        vector.rotate_elements_left_dyn(amount).to_array(),
        [2, 3, 0, 1]
    );
    assert_eq!(
        vector.rotate_elements_right_dyn(amount).to_array(),
        [2, 3, 0, 1]
    );
    assert_eq!(vector.rotate_elements_left_dyn(0).to_array(), [0, 1, 2, 3]);
    assert_eq!(vector.rotate_elements_right_dyn(4).to_array(), [0, 1, 2, 3]);
    assert_eq!(vector.rotate_elements_left_dyn(5).to_array(), [1, 2, 3, 0]);
    assert_eq!(vector.rotate_elements_right_dyn(5).to_array(), [3, 0, 1, 2]);
    assert_eq!(
        vector.rotate_elements_left_dyn(usize::MAX).to_array(),
        [3, 0, 1, 2]
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn shift() {
    let a = Simd::from_array([1, 2, 3, 4]);
    assert_eq!(a.shift_elements_left::<0>(0).to_array(), [1, 2, 3, 4]);
    assert_eq!(a.shift_elements_left::<1>(0).to_array(), [2, 3, 4, 0]);
    assert_eq!(a.shift_elements_left::<2>(9).to_array(), [3, 4, 9, 9]);
    assert_eq!(a.shift_elements_left::<3>(8).to_array(), [4, 8, 8, 8]);
    assert_eq!(a.shift_elements_left::<4>(7).to_array(), [7, 7, 7, 7]);
    assert_eq!(a.shift_elements_left::<5>(6).to_array(), [6, 6, 6, 6]);
    assert_eq!(
        a.shift_elements_left::<{ usize::MAX }>(5).to_array(),
        [5, 5, 5, 5]
    );
    assert_eq!(a.shift_elements_right::<0>(0).to_array(), [1, 2, 3, 4]);
    assert_eq!(a.shift_elements_right::<1>(0).to_array(), [0, 1, 2, 3]);
    assert_eq!(a.shift_elements_right::<2>(-1).to_array(), [-1, -1, 1, 2]);
    assert_eq!(a.shift_elements_right::<3>(-2).to_array(), [-2, -2, -2, 1]);
    assert_eq!(a.shift_elements_right::<4>(-3).to_array(), [-3, -3, -3, -3]);
    assert_eq!(a.shift_elements_right::<5>(-4).to_array(), [-4, -4, -4, -4]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn interleave() {
    let a = Simd::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    let b = Simd::from_array([8, 9, 10, 11, 12, 13, 14, 15]);
    let (lo, hi) = a.interleave(b);
    assert_eq!(lo.to_array(), [0, 8, 1, 9, 2, 10, 3, 11]);
    assert_eq!(hi.to_array(), [4, 12, 5, 13, 6, 14, 7, 15]);
    let (even, odd) = lo.deinterleave(hi);
    assert_eq!(even, a);
    assert_eq!(odd, b);
}

// portable-simd#298
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn interleave_one() {
    let a = Simd::from_array([0]);
    let b = Simd::from_array([1]);
    let (lo, hi) = a.interleave(b);
    assert_eq!(lo.to_array(), [0]);
    assert_eq!(hi.to_array(), [1]);
    let (even, odd) = lo.deinterleave(hi);
    assert_eq!(even, a);
    assert_eq!(odd, b);
}

/// Regression coverage for portable-simd#486.
///
/// The unchecked variant requires every index to be in bounds or to have
/// its most significant bit set. Such out-of-bounds indices must produce zero.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn swizzle_dyn_unchecked_zeroes_high_bit_indices_u8x16() {
    let bytes = Simd::from_array(core::array::from_fn(|i| i as u8));
    let indices = Simd::from_array([
        0, 15, 128, 255, 1, 14, 129, 254, 2, 13, 130, 253, 3, 12, 131, 252,
    ]);

    assert_eq!(
        bytes.swizzle_dyn_unchecked(indices).to_array(),
        [0, 15, 0, 0, 1, 14, 0, 0, 2, 13, 0, 0, 3, 12, 0, 0]
    );
}

/// The same contract must hold when the implementation splits a 32-byte
/// vector into narrower native-width shuffles.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn swizzle_dyn_unchecked_zeroes_high_bit_indices_u8x32() {
    let bytes = Simd::from_array(core::array::from_fn(|i| i as u8));
    let indices = Simd::from_array([
        0, 31, 128, 255, 1, 30, 129, 254, 2, 29, 130, 253, 3, 28, 131, 252, 4, 27, 132, 251, 5, 26,
        133, 250, 6, 25, 134, 249, 7, 24, 135, 248,
    ]);

    assert_eq!(
        bytes.swizzle_dyn_unchecked(indices).to_array(),
        [
            0, 31, 0, 0, 1, 30, 0, 0, 2, 29, 0, 0, 3, 28, 0, 0, 4, 27, 0, 0, 5, 26, 0, 0, 6, 25, 0,
            0, 7, 24, 0, 0,
        ]
    );
}
