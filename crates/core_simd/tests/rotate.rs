#![feature(portable_simd)]

use core_simd::simd::Simd;
use core_simd::simd::num::{SimdInt, SimdUint};

macro_rules! signed_rotate_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn rotations_match_scalar<const LANES: usize>() {
                        let values: [$ty; LANES] = core::array::from_fn(|lane| match lane % 6 {
                            0 => <$ty>::MIN,
                            1 => <$ty>::MAX,
                            2 => -1,
                            3 => -2,
                            4 => 0,
                            _ => 0x35 as $ty,
                        });
                        let vector = Simd::<$ty, LANES>::from_array(values);

                        for shift in [
                            0,
                            1,
                            <$ty>::BITS - 1,
                            <$ty>::BITS,
                            <$ty>::BITS + 1,
                            u32::MAX,
                        ] {
                            assert_eq!(
                                vector.rotate_left(shift).to_array(),
                                values.map(|value| value.rotate_left(shift)),
                                "rotate_left shift={shift}",
                            );
                            assert_eq!(
                                vector.rotate_right(shift).to_array(),
                                values.map(|value| value.rotate_right(shift)),
                                "rotate_right shift={shift}",
                            );
                        }
                    }
                }
            }
        )*
    };
}

macro_rules! unsigned_rotate_tests {
    ($($module:ident: $ty:ty),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                test_helpers::test_lanes! {
                    fn rotations_match_scalar<const LANES: usize>() {
                        let values: [$ty; LANES] = core::array::from_fn(|lane| match lane % 5 {
                            0 => 0,
                            1 => <$ty>::MAX,
                            2 => 1,
                            3 => 0x35 as $ty,
                            _ => <$ty>::MAX / 2,
                        });
                        let vector = Simd::<$ty, LANES>::from_array(values);

                        for shift in [0, 1, <$ty>::BITS - 1, <$ty>::BITS, <$ty>::BITS + 1, u32::MAX] {
                            assert_eq!(
                                vector.rotate_left(shift).to_array(),
                                values.map(|value| value.rotate_left(shift)),
                                "rotate_left shift={shift}",
                            );
                            assert_eq!(
                                vector.rotate_right(shift).to_array(),
                                values.map(|value| value.rotate_right(shift)),
                                "rotate_right shift={shift}",
                            );
                        }
                    }
                }
            }
        )*
    };
}

signed_rotate_tests! {
    i8_tests: i8,
    i16_tests: i16,
    i32_tests: i32,
    i64_tests: i64,
    isize_tests: isize,
}

unsigned_rotate_tests! {
    u8_tests: u8,
    u16_tests: u16,
    u32_tests: u32,
    u64_tests: u64,
    usize_tests: usize,
}
