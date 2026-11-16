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
    T: Copy + core::fmt::Debug + PartialEq + From<u8> + core_simd::simd::SimdElement,
    for<'a> core_simd::simd::Simd<T, LANES>: core::convert::TryFrom<&'a [T]>,
{
    let values: [T; LANES] =
        core::array::from_fn(|lane| T::from((lane as u8).wrapping_mul(17).wrapping_add(3)));
    let vector = core_simd::simd::Simd::<T, LANES>::try_from(&values[..]).unwrap();
    assert_eq!(vector.to_array(), values, "exact length, lanes={LANES}");

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
