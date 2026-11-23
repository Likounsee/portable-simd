#![feature(portable_simd)]

use core_simd::simd::{
    Simd,
    ptr::{SimdConstPtr, SimdMutPtr},
};

fn check_const_pointer_arithmetic<T, const N: usize>() {
    let base = core::ptr::without_provenance::<T>(0x1234);
    let pointers = Simd::<*const T, N>::splat(base);

    let counts = core::array::from_fn(|lane| match lane % 6 {
        0 => 0,
        1 => 1,
        2 => isize::MAX as usize,
        3 => (isize::MAX as usize) + 1,
        4 => usize::MAX,
        _ => usize::MAX - 1,
    });
    let actual_add = pointers
        .wrapping_add(Simd::from_array(counts))
        .addr()
        .to_array();
    let expected_add = core::array::from_fn(|lane| base.wrapping_add(counts[lane]).addr());
    assert_eq!(actual_add, expected_add, "const pointer wrapping_add, N={N}");

    let actual_sub = pointers
        .wrapping_sub(Simd::from_array(counts))
        .addr()
        .to_array();
    let expected_sub = core::array::from_fn(|lane| base.wrapping_sub(counts[lane]).addr());
    assert_eq!(actual_sub, expected_sub, "const pointer wrapping_sub, N={N}");

    let offsets = core::array::from_fn(|lane| match lane % 5 {
        0 => 0,
        1 => 1,
        2 => -1,
        3 => isize::MAX,
        _ => isize::MIN,
    });
    let actual_offset = pointers
        .wrapping_offset(Simd::from_array(offsets))
        .addr()
        .to_array();
    let expected_offset =
        core::array::from_fn(|lane| base.wrapping_offset(offsets[lane]).addr());
    assert_eq!(
        actual_offset, expected_offset,
        "const pointer wrapping_offset, N={N}"
    );
}

fn check_mut_pointer_arithmetic<T, const N: usize>() {
    let base = core::ptr::without_provenance_mut::<T>(0x5678);
    let pointers = Simd::<*mut T, N>::splat(base);

    let counts = core::array::from_fn(|lane| match lane % 6 {
        0 => 0,
        1 => 1,
        2 => isize::MAX as usize,
        3 => (isize::MAX as usize) + 1,
        4 => usize::MAX,
        _ => usize::MAX - 1,
    });
    let actual_add = pointers
        .wrapping_add(Simd::from_array(counts))
        .addr()
        .to_array();
    let expected_add = core::array::from_fn(|lane| base.wrapping_add(counts[lane]).addr());
    assert_eq!(actual_add, expected_add, "mut pointer wrapping_add, N={N}");

    let actual_sub = pointers
        .wrapping_sub(Simd::from_array(counts))
        .addr()
        .to_array();
    let expected_sub = core::array::from_fn(|lane| base.wrapping_sub(counts[lane]).addr());
    assert_eq!(actual_sub, expected_sub, "mut pointer wrapping_sub, N={N}");

    let offsets = core::array::from_fn(|lane| match lane % 5 {
        0 => 0,
        1 => 1,
        2 => -1,
        3 => isize::MAX,
        _ => isize::MIN,
    });
    let actual_offset = pointers
        .wrapping_offset(Simd::from_array(offsets))
        .addr()
        .to_array();
    let expected_offset =
        core::array::from_fn(|lane| base.wrapping_offset(offsets[lane]).addr());
    assert_eq!(
        actual_offset, expected_offset,
        "mut pointer wrapping_offset, N={N}"
    );
}

#[test]
fn pointer_wrapping_arithmetic_matches_scalar_at_integer_boundaries() {
    check_const_pointer_arithmetic::<u8, 1>();
    check_const_pointer_arithmetic::<u8, 2>();
    check_const_pointer_arithmetic::<u8, 4>();
    check_const_pointer_arithmetic::<u8, 8>();
    check_const_pointer_arithmetic::<u8, 16>();
    check_const_pointer_arithmetic::<u8, 32>();
    check_const_pointer_arithmetic::<u8, 64>();

    check_mut_pointer_arithmetic::<u8, 1>();
    check_mut_pointer_arithmetic::<u8, 2>();
    check_mut_pointer_arithmetic::<u8, 4>();
    check_mut_pointer_arithmetic::<u8, 8>();
    check_mut_pointer_arithmetic::<u8, 16>();
    check_mut_pointer_arithmetic::<u8, 32>();
    check_mut_pointer_arithmetic::<u8, 64>();

    check_const_pointer_arithmetic::<u32, 1>();
    check_mut_pointer_arithmetic::<u32, 1>();
    check_const_pointer_arithmetic::<u32, 2>();
    check_mut_pointer_arithmetic::<u32, 2>();
    check_const_pointer_arithmetic::<u32, 4>();
    check_mut_pointer_arithmetic::<u32, 4>();
    check_const_pointer_arithmetic::<u32, 8>();
    check_mut_pointer_arithmetic::<u32, 8>();
    check_const_pointer_arithmetic::<u32, 16>();
    check_mut_pointer_arithmetic::<u32, 16>();
    check_const_pointer_arithmetic::<u32, 32>();
    check_mut_pointer_arithmetic::<u32, 32>();
    check_const_pointer_arithmetic::<u32, 64>();
    check_mut_pointer_arithmetic::<u32, 64>();
}
