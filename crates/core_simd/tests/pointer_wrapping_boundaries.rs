#![feature(portable_simd)]

use core_simd::simd::{
    Simd,
    ptr::{SimdConstPtr, SimdMutPtr},
};

fn check_const_pointer_arithmetic<const N: usize>() {
    let base = core::ptr::without_provenance::<u8>(0x1234);
    let pointers = Simd::<*const u8, N>::splat(base);

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

fn check_mut_pointer_arithmetic<const N: usize>() {
    let base = core::ptr::without_provenance_mut::<u8>(0x5678);
    let pointers = Simd::<*mut u8, N>::splat(base);

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
    check_const_pointer_arithmetic::<1>();
    check_const_pointer_arithmetic::<2>();
    check_const_pointer_arithmetic::<4>();
    check_const_pointer_arithmetic::<8>();
    check_const_pointer_arithmetic::<16>();
    check_const_pointer_arithmetic::<32>();
    check_const_pointer_arithmetic::<64>();

    check_mut_pointer_arithmetic::<1>();
    check_mut_pointer_arithmetic::<2>();
    check_mut_pointer_arithmetic::<4>();
    check_mut_pointer_arithmetic::<8>();
    check_mut_pointer_arithmetic::<16>();
    check_mut_pointer_arithmetic::<32>();
    check_mut_pointer_arithmetic::<64>();
}
