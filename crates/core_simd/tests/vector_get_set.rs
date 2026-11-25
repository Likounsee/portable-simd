#![feature(portable_simd)]

use core_simd::simd::Simd;

fn check_lane_access<const LANES: usize>() {
    let values = core::array::from_fn(|lane| lane as u32 * 17 + 5);
    let vector = Simd::<u32, LANES>::from_array(values);

    for index in 0..LANES {
        assert_eq!(vector.get(index), values[index], "get, lane={index}");
        assert_eq!(
            vector.get_checked(index),
            Some(values[index]),
            "get_checked, lane={index}"
        );
        // SAFETY: index is drawn from 0..LANES.
        assert_eq!(
            unsafe { vector.get_unchecked(index) },
            values[index],
            "get_unchecked, lane={index}"
        );

        let replacement = u32::MAX - index as u32;
        let mut updated = vector;
        updated.set(index, replacement);
        let mut expected = values;
        expected[index] = replacement;
        assert_eq!(updated.to_array(), expected, "set, lane={index}");

        let mut updated_unchecked = vector;
        // SAFETY: index is drawn from 0..LANES.
        unsafe { updated_unchecked.set_unchecked(index, replacement) };
        assert_eq!(
            updated_unchecked.to_array(),
            expected,
            "set_unchecked, lane={index}"
        );

        let mut updated_checked = vector;
        assert_eq!(
            updated_checked.set_checked(index, replacement),
            Some(()),
            "set_checked, lane={index}"
        );
        assert_eq!(
            updated_checked.to_array(),
            expected,
            "set_checked, lane={index}"
        );
    }

    assert_eq!(vector.get_checked(LANES), None);
    assert_eq!(vector.get_checked(usize::MAX), None);

    let mut unchanged = vector;
    assert_eq!(unchanged.set_checked(LANES, u32::MAX), None);
    assert_eq!(unchanged.set_checked(usize::MAX, u32::MAX), None);
    assert_eq!(
        unchanged, vector,
        "out-of-bounds set_checked must not mutate"
    );
}

test_helpers::test_lanes! {
    fn get_set_and_unchecked_access_cover_every_lane_count<const LANES: usize>() {
        check_lane_access::<LANES>();
    }
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn get_panics_when_index_is_out_of_bounds() {
    let vector = Simd::<u32, 4>::from_array([1, 2, 3, 4]);
    let _ = vector.get(4);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn set_panics_when_index_is_out_of_bounds() {
    let mut vector = Simd::<u32, 4>::from_array([1, 2, 3, 4]);
    vector.set(4, 99);
}
