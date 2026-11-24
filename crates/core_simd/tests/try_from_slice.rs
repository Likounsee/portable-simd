#![feature(portable_simd)]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

use core_simd::simd::i32x4;

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn try_from_slice() {
    // Equal length
    assert_eq!(
        i32x4::try_from([1, 2, 3, 4].as_slice()).unwrap(),
        i32x4::from_array([1, 2, 3, 4])
    );

    // Slice length > vector length
    assert!(i32x4::try_from([1, 2, 3, 4, 5].as_slice()).is_err());

    // Slice length < vector length
    assert!(i32x4::try_from([1, 2, 3].as_slice()).is_err());
}

fn check_try_from_slice_lane_count<T, const LANES: usize>()
where
    T: Copy + core::fmt::Debug + PartialEq + From<u8> + core_simd::simd::SimdElement + 'static,
    for<'a> core_simd::simd::Simd<T, LANES>: core::convert::TryFrom<&'a [T]>,
{
    let values: [T; LANES] =
        core::array::from_fn(|lane| T::from((lane as u8).wrapping_mul(17).wrapping_add(3)));
    let vector = core_simd::simd::Simd::<T, LANES>::try_from(&values[..])
        .ok()
        .expect("exact-length slice must convert");
    assert_eq!(vector.to_array(), values, "exact length, lanes={LANES}");

    // Exercise the raw array-to-vector and vector-to-array copy paths with an
    // allocation containing exactly LANES elements. This guards the layout
    // assumptions used by Simd::from_slice and Simd::copy_to_slice.
    let loaded = core_simd::simd::Simd::<T, LANES>::from_slice(&values);
    assert_eq!(loaded.to_array(), values, "from_slice, lanes={LANES}");

    let mut copied = [T::from(0); LANES];
    loaded.copy_to_slice(&mut copied);
    assert_eq!(copied, values, "copy_to_slice, lanes={LANES}");

    assert!(
        core_simd::simd::Simd::<T, LANES>::try_from(&values[..LANES - 1]).is_err(),
        "short slice, lanes={LANES}"
    );

    let mut longer = values.to_vec();
    longer.push(T::from(251));
    assert!(
        core_simd::simd::Simd::<T, LANES>::try_from(&longer[..]).is_err(),
        "long slice, lanes={LANES}"
    );
}

macro_rules! try_from_slice_lane_coverage {
    ($name:ident, $ty:ty) => {
        test_helpers::test_lanes! {
            fn $name<const LANES: usize>() {
                check_try_from_slice_lane_count::<$ty, LANES>();
            }
        }
    };
}

try_from_slice_lane_coverage!(try_from_slice_u8_all_lane_counts, u8);
try_from_slice_lane_coverage!(try_from_slice_i16_all_lane_counts, i16);
try_from_slice_lane_coverage!(try_from_slice_u32_all_lane_counts, u32);

try_from_slice_lane_coverage!(try_from_slice_i8_all_lane_counts, i8);
try_from_slice_lane_coverage!(try_from_slice_i32_all_lane_counts, i32);
try_from_slice_lane_coverage!(try_from_slice_i64_all_lane_counts, i64);
try_from_slice_lane_coverage!(try_from_slice_isize_all_lane_counts, isize);
try_from_slice_lane_coverage!(try_from_slice_u16_all_lane_counts, u16);
try_from_slice_lane_coverage!(try_from_slice_u64_all_lane_counts, u64);
try_from_slice_lane_coverage!(try_from_slice_usize_all_lane_counts, usize);
try_from_slice_lane_coverage!(try_from_slice_f32_all_lane_counts, f32);
try_from_slice_lane_coverage!(try_from_slice_f64_all_lane_counts, f64);
