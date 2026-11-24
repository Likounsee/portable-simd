#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

#[test]
fn signed_64_bit_shift_counts_match_scalar_u32_conversion() {
    let value = Simd::from_array([1_i64, -2, 3, i64::MIN]);
    let amount = Simd::from_array([1_i64, 64, 1_i64 << 32, -1_i64]);

    let (left, left_overflow) = value.overflowing_shl(amount);
    let (right, right_overflow) = value.overflowing_shr(amount);

    assert_eq!(left.to_array(), [2, -2, 3, 0]);
    assert_eq!(right.to_array(), [0, -2, 3, -1]);
    assert_eq!(left_overflow.to_array(), [false, true, false, true]);
    assert_eq!(right_overflow.to_array(), [false, true, false, true]);
}

#[test]
fn unsigned_64_bit_shift_counts_match_scalar_u32_conversion() {
    let value = Simd::from_array([1_u64, 2, 3, u64::MAX]);
    let amount = Simd::from_array([1_u64, 64, 1_u64 << 32, u64::MAX]);

    let (left, left_overflow) = value.overflowing_shl(amount);
    let (right, right_overflow) = value.overflowing_shr(amount);

    assert_eq!(left.to_array(), [2, 2, 3, 1_u64 << 63]);
    assert_eq!(right.to_array(), [0, 2, 3, 1]);
    assert_eq!(left_overflow.to_array(), [false, true, false, true]);
    assert_eq!(right_overflow.to_array(), [false, true, false, true]);
}

macro_rules! check_signed_shift_counts {
    ($ty:ty) => {{
        let value = Simd::<$ty, 4>::from_array([1, -2, <$ty>::MIN, <$ty>::MAX]);
        let amount = Simd::<$ty, 4>::from_array([1, <$ty>::BITS as $ty, -1, 0]);
        let (left, left_overflow) = value.overflowing_shl(amount);
        let (right, right_overflow) = value.overflowing_shr(amount);

        let expected_left = core::array::from_fn(|lane| {
            value.to_array()[lane].overflowing_shl(amount.to_array()[lane] as u32)
        });
        let expected_right = core::array::from_fn(|lane| {
            value.to_array()[lane].overflowing_shr(amount.to_array()[lane] as u32)
        });

        assert_eq!(left.to_array(), expected_left.map(|result| result.0));
        assert_eq!(right.to_array(), expected_right.map(|result| result.0));
        assert_eq!(
            left_overflow.to_array(),
            expected_left.map(|result| result.1)
        );
        assert_eq!(
            right_overflow.to_array(),
            expected_right.map(|result| result.1)
        );
    }};
}

macro_rules! check_unsigned_shift_counts {
    ($ty:ty) => {{
        let value = Simd::<$ty, 4>::from_array([1, 2, <$ty>::MIN, <$ty>::MAX]);
        let amount = Simd::<$ty, 4>::from_array([1, <$ty>::BITS as $ty, <$ty>::MAX, 0]);
        let (left, left_overflow) = value.overflowing_shl(amount);
        let (right, right_overflow) = value.overflowing_shr(amount);

        let expected_left = core::array::from_fn(|lane| {
            value.to_array()[lane].overflowing_shl(amount.to_array()[lane] as u32)
        });
        let expected_right = core::array::from_fn(|lane| {
            value.to_array()[lane].overflowing_shr(amount.to_array()[lane] as u32)
        });

        assert_eq!(left.to_array(), expected_left.map(|result| result.0));
        assert_eq!(right.to_array(), expected_right.map(|result| result.0));
        assert_eq!(
            left_overflow.to_array(),
            expected_left.map(|result| result.1)
        );
        assert_eq!(
            right_overflow.to_array(),
            expected_right.map(|result| result.1)
        );
    }};
}

#[test]
fn narrow_signed_shift_counts_match_scalar_semantics() {
    check_signed_shift_counts!(i8);
    check_signed_shift_counts!(i16);
    check_signed_shift_counts!(i32);
}

#[test]
fn narrow_unsigned_shift_counts_match_scalar_semantics() {
    check_unsigned_shift_counts!(u8);
    check_unsigned_shift_counts!(u16);
    check_unsigned_shift_counts!(u32);
}
macro_rules! signed_shift_count_lane_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn every_supported_lane_count_matches_scalar_semantics<const LANES: usize>() {
                    let value = Simd::<$ty, LANES>::from_array(core::array::from_fn(|lane| {
                        match lane % 4 {
                            0 => 1,
                            1 => -2,
                            2 => <$ty>::MIN,
                            _ => <$ty>::MAX,
                        }
                    }));
                    let amount = Simd::<$ty, LANES>::from_array(core::array::from_fn(|lane| {
                        match lane % 4 {
                            0 => 1,
                            1 => <$ty>::BITS as $ty,
                            2 => -1,
                            _ => 0,
                        }
                    }));

                    let (left, left_overflow) = value.overflowing_shl(amount);
                    let (right, right_overflow) = value.overflowing_shr(amount);
                    let expected_left = core::array::from_fn(|lane| {
                        value.to_array()[lane].overflowing_shl(amount.to_array()[lane] as u32)
                    });
                    let expected_right = core::array::from_fn(|lane| {
                        value.to_array()[lane].overflowing_shr(amount.to_array()[lane] as u32)
                    });

                    assert_eq!(left.to_array(), expected_left.map(|result| result.0));
                    assert_eq!(right.to_array(), expected_right.map(|result| result.0));
                    assert_eq!(left_overflow.to_array(), expected_left.map(|result| result.1));
                    assert_eq!(right_overflow.to_array(), expected_right.map(|result| result.1));
                }
            }
        }
    };
}

macro_rules! unsigned_shift_count_lane_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;

            test_helpers::test_lanes! {
                fn every_supported_lane_count_matches_scalar_semantics<const LANES: usize>() {
                    let value = Simd::<$ty, LANES>::from_array(core::array::from_fn(|lane| {
                        match lane % 4 {
                            0 => 1,
                            1 => 2,
                            2 => <$ty>::MIN,
                            _ => <$ty>::MAX,
                        }
                    }));
                    let amount = Simd::<$ty, LANES>::from_array(core::array::from_fn(|lane| {
                        match lane % 4 {
                            0 => 1,
                            1 => <$ty>::BITS as $ty,
                            2 => <$ty>::MAX,
                            _ => 0,
                        }
                    }));

                    let (left, left_overflow) = value.overflowing_shl(amount);
                    let (right, right_overflow) = value.overflowing_shr(amount);
                    let expected_left = core::array::from_fn(|lane| {
                        value.to_array()[lane].overflowing_shl(amount.to_array()[lane] as u32)
                    });
                    let expected_right = core::array::from_fn(|lane| {
                        value.to_array()[lane].overflowing_shr(amount.to_array()[lane] as u32)
                    });

                    assert_eq!(left.to_array(), expected_left.map(|result| result.0));
                    assert_eq!(right.to_array(), expected_right.map(|result| result.0));
                    assert_eq!(left_overflow.to_array(), expected_left.map(|result| result.1));
                    assert_eq!(right_overflow.to_array(), expected_right.map(|result| result.1));
                }
            }
        }
    };
}

signed_shift_count_lane_tests!(i8_lane_counts, i8);
signed_shift_count_lane_tests!(i16_lane_counts, i16);
signed_shift_count_lane_tests!(i32_lane_counts, i32);
unsigned_shift_count_lane_tests!(u8_lane_counts, u8);
unsigned_shift_count_lane_tests!(u16_lane_counts, u16);
unsigned_shift_count_lane_tests!(u32_lane_counts, u32);
