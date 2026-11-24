#![cfg(target_arch = "aarch64")]
#![cfg_attr(target_arch = "aarch64", feature(portable_simd))]

use core_simd::simd::Simd;

// AArch64 has historically ICE'd in LLVM for integer SIMD div/rem with
// non-power-of-two lane counts and particular divisor values (upstream #427).
// Keep these regressions on three lanes so they exercise that exact shape.

macro_rules! zero_divisor_tests {
    ($($name:ident: $ty:ty => $op:tt),* $(,)?) => {
        $(
            #[test]
            #[should_panic(expected = "attempt to divide by zero")]
            fn $name() {
                let lhs = Simd::<$ty, 3>::splat(12);
                let rhs = Simd::from_array([3, 0, 4]);
                let _ = lhs $op rhs;
            }
        )*
    };
}

zero_divisor_tests! {
    signed_i8_division_by_zero: i8 => /,
    signed_i16_division_by_zero: i16 => /,
    signed_i32_division_by_zero: i32 => /,
    signed_i64_division_by_zero: i64 => /,
    signed_isize_division_by_zero: isize => /,
    unsigned_u8_division_by_zero: u8 => /,
    unsigned_u16_division_by_zero: u16 => /,
    unsigned_u32_division_by_zero: u32 => /,
    unsigned_u64_division_by_zero: u64 => /,
    unsigned_usize_division_by_zero: usize => /,
    signed_i8_remainder_by_zero: i8 => %,
    signed_i16_remainder_by_zero: i16 => %,
    signed_i32_remainder_by_zero: i32 => %,
    signed_i64_remainder_by_zero: i64 => %,
    signed_isize_remainder_by_zero: isize => %,
    unsigned_u8_remainder_by_zero: u8 => %,
    unsigned_u16_remainder_by_zero: u16 => %,
    unsigned_u32_remainder_by_zero: u32 => %,
    unsigned_u64_remainder_by_zero: u64 => %,
    unsigned_usize_remainder_by_zero: usize => %,
}

macro_rules! remainder_by_min_tests {
    ($($name:ident: $ty:ty),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                let lhs = Simd::<$ty, 3>::from_array([<$ty>::MIN, -1, 17]);
                let rhs = Simd::<$ty, 3>::splat(<$ty>::MIN);

                assert_eq!((lhs % rhs).to_array(), [0, -1, 17]);
            }
        )*
    };
}

remainder_by_min_tests! {
    i8_remainder_by_min: i8,
    i16_remainder_by_min: i16,
    i32_remainder_by_min: i32,
    i64_remainder_by_min: i64,
    isize_remainder_by_min: isize,
}


macro_rules! signed_division_overflow_tests {
    ($($name:ident: $ty:ty),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                let lhs = Simd::<$ty, 3>::from_array([<$ty>::MIN, -1, 17]);
                let rhs = Simd::<$ty, 3>::from_array([-1, <$ty>::MIN, <$ty>::MIN]);

                assert_eq!((lhs / rhs).to_array(), [<$ty>::MIN, 0, 0]);
            }
        )*
    };
}

signed_division_overflow_tests! {
    i8_division_overflow_and_min_divisor: i8,
    i16_division_overflow_and_min_divisor: i16,
    i32_division_overflow_and_min_divisor: i32,
    i64_division_overflow_and_min_divisor: i64,
    isize_division_overflow_and_min_divisor: isize,
}

macro_rules! signed_remainder_overflow_tests {
    ($($name:ident: $ty:ty),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                let lhs = Simd::<$ty, 3>::splat(<$ty>::MIN);
                let rhs = Simd::<$ty, 3>::splat(-1);

                assert_eq!((lhs % rhs).to_array(), [0; 3]);
            }
        )*
    };
}

signed_remainder_overflow_tests! {
    i8_remainder_overflow: i8,
    i16_remainder_overflow: i16,
    i32_remainder_overflow: i32,
    i64_remainder_overflow: i64,
    isize_remainder_overflow: isize,
}
