#![feature(portable_simd)]

use core_simd::simd::{
    Simd,
    ptr::{self, SimdConstPtr, SimdMutPtr},
};

macro_rules! common_tests {
    { $constness:ident } => {
        test_helpers::test_lanes! {
            fn is_null<const LANES: usize>() {
                test_helpers::test_unary_mask_elementwise(
                    &Simd::<*$constness u32, LANES>::is_null,
                    &<*$constness u32>::is_null,
                    &|_| true,
                );

                let dangling = core::ptr::NonNull::<u32>::dangling().as_ptr() as *$constness u32;
                let pointers: [*$constness u32; LANES] =
                    core::array::from_fn(|lane| if lane % 2 == 0 { core::ptr::null_mut() as *$constness u32 } else { dangling });
                let actual = Simd::<*$constness u32, LANES>::from_array(pointers)
                    .is_null()
                    .to_array();
                let expected = core::array::from_fn(|lane| lane % 2 == 0);
                assert_eq!(actual, expected);
            }

            fn addr<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &Simd::<*$constness u32, LANES>::addr,
                    &<*$constness u32>::addr,
                    &|_| true,
                );
            }

            fn with_addr<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &Simd::<*$constness u32, LANES>::with_addr,
                    &<*$constness u32>::with_addr,
                    &|_, _| true,
                );

                let mut storage = [0u32; LANES];
                let base: [*$constness u32; LANES] = core::array::from_fn(|lane| {
                    storage.as_mut_ptr().wrapping_add(lane) as *$constness u32
                });
                let target_addresses: [usize; LANES] = core::array::from_fn(|lane| {
                    storage
                        .as_mut_ptr()
                        .wrapping_add(LANES - 1 - lane)
                        .addr()
                });

                let actual = Simd::<*$constness u32, LANES>::from_array(base)
                    .with_addr(Simd::from_array(target_addresses))
                    .addr()
                    .to_array();
                assert_eq!(actual, target_addresses);
            }

            fn expose_provenance<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &Simd::<*$constness u32, LANES>::expose_provenance,
                    &<*$constness u32>::expose_provenance,
                    &|_| true,
                );
            }

            fn wrapping_offset<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &Simd::<*$constness u32, LANES>::wrapping_offset,
                    &<*$constness u32>::wrapping_offset,
                    &|_, _| true,
                );

                let storage = [0u32; LANES];
                let midpoint = LANES / 2;
                let base = unsafe { storage.as_ptr().add(midpoint) } as *$constness u32;
                let offsets: [isize; LANES] =
                    core::array::from_fn(|lane| lane as isize - midpoint as isize);
                let actual = Simd::<*$constness u32, LANES>::splat(base)
                    .wrapping_offset(Simd::from_array(offsets))
                    .addr()
                    .to_array();
                let expected = core::array::from_fn(|lane| {
                    base.wrapping_offset(offsets[lane]).addr()
                });
                assert_eq!(actual, expected);
            }

            fn wrapping_add<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &Simd::<*$constness u32, LANES>::wrapping_add,
                    &<*$constness u32>::wrapping_add,
                    &|_, _| true,
                );

                let mut storage = [0u32; LANES];
                let base = storage.as_mut_ptr() as *$constness u32;
                let counts: [usize; LANES] = core::array::from_fn(|lane| lane);
                let actual = Simd::<*$constness u32, LANES>::splat(base)
                    .wrapping_add(Simd::from_array(counts))
                    .addr()
                    .to_array();
                let expected = core::array::from_fn(|lane| {
                    base.wrapping_add(counts[lane]).addr()
                });
                assert_eq!(actual, expected);
            }

            fn wrapping_sub<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &Simd::<*$constness u32, LANES>::wrapping_sub,
                    &<*$constness u32>::wrapping_sub,
                    &|_, _| true,
                );

                let mut storage = [0u32; LANES];
                let base = storage.as_mut_ptr().wrapping_add(LANES - 1) as *$constness u32;
                let counts: [usize; LANES] =
                    core::array::from_fn(|lane| LANES - 1 - lane);
                let actual = Simd::<*$constness u32, LANES>::splat(base)
                    .wrapping_sub(Simd::from_array(counts))
                    .addr()
                    .to_array();
                let expected = core::array::from_fn(|lane| {
                    base.wrapping_sub(counts[lane]).addr()
                });
                assert_eq!(actual, expected);
            }
        }
    }
}

mod const_ptr {
    use super::*;
    common_tests! { const }

    test_helpers::test_lanes! {
        fn cast_mut<const LANES: usize>() {
            test_helpers::test_unary_elementwise(
                &Simd::<*const u32, LANES>::cast_mut,
                &<*const u32>::cast_mut,
                &|_| true,
            );
        }

        fn cast_preserves_addresses<const LANES: usize>() {
            let storage = [0u32; LANES];
            let pointers = Simd::<*const u32, LANES>::from_array(
                core::array::from_fn(|lane| core::ptr::addr_of!(storage[lane])),
            );
            let cast = pointers.cast::<u8>();
            assert_eq!(cast.addr().to_array(), pointers.addr().to_array());
        }

        fn with_addr_preserves_provenance_for_dereference<const LANES: usize>() {
            let storage = core::array::from_fn::<_, LANES, _>(|lane| lane as u32 + 10);
            let base = storage.as_ptr();
            let pointers = Simd::<*const u32, LANES>::from_array(
                core::array::from_fn(|lane| base.wrapping_add(lane)),
            );
            let target_addresses = core::array::from_fn(|lane| {
                base.wrapping_add(LANES - 1 - lane).addr()
            });
            let relocated = pointers.with_addr(Simd::from_array(target_addresses)).to_array();

            // SAFETY: every target belongs to the same allocation and each
            // pointer retains provenance through with_addr.
            let actual = unsafe { relocated.map(|pointer| *pointer) };
            let expected = core::array::from_fn(|lane| storage[LANES - 1 - lane]);
            assert_eq!(actual, expected);
        }

        fn with_exposed_provenance<const LANES: usize>() {
            test_helpers::test_unary_elementwise(
                &ptr::with_exposed_provenance::<u32, LANES>,
                &core::ptr::with_exposed_provenance::<u32>,
                &|_| true,
            );
        }

        fn without_provenance<const LANES: usize>() {
            test_helpers::test_unary_elementwise(
                &ptr::without_provenance::<u32, LANES>,
                &core::ptr::without_provenance::<u32>,
                &|_| true,
            );
        }

    }
}

mod mut_ptr {
    use super::*;
    common_tests! { mut }

    test_helpers::test_lanes! {
        fn cast_const<const LANES: usize>() {
            test_helpers::test_unary_elementwise(
                &Simd::<*mut u32, LANES>::cast_const,
                &<*mut u32>::cast_const,
                &|_| true,
            );
        }

        fn cast_preserves_addresses<const LANES: usize>() {
            let mut storage = [0u32; LANES];
            let pointers = Simd::<*mut u32, LANES>::from_array(
                core::array::from_fn(|lane| core::ptr::addr_of_mut!(storage[lane])),
            );
            let cast = pointers.cast::<u8>();
            assert_eq!(cast.addr().to_array(), pointers.addr().to_array());
        }

        fn with_addr_preserves_provenance_for_writes<const LANES: usize>() {
            let mut storage = [0u32; LANES];
            let base = storage.as_mut_ptr();
            let pointers = Simd::<*mut u32, LANES>::from_array(
                core::array::from_fn(|lane| base.wrapping_add(lane)),
            );
            let target_addresses = core::array::from_fn(|lane| {
                base.wrapping_add(LANES - 1 - lane).addr()
            });
            let relocated = pointers.with_addr(Simd::from_array(target_addresses)).to_array();

            // SAFETY: the target addresses are distinct elements of the
            // allocation and each pointer retains provenance through with_addr.
            unsafe {
                for (lane, pointer) in relocated.into_iter().enumerate() {
                    pointer.write(lane as u32 + 100);
                }
            }

            let expected = core::array::from_fn(|lane| (LANES - 1 - lane) as u32 + 100);
            assert_eq!(storage, expected);
        }

        fn with_exposed_provenance<const LANES: usize>() {
            test_helpers::test_unary_elementwise(
                &ptr::with_exposed_provenance_mut::<u32, LANES>,
                &core::ptr::with_exposed_provenance_mut::<u32>,
                &|_| true,
            );
        }

        fn without_provenance<const LANES: usize>() {
            test_helpers::test_unary_elementwise(
                &ptr::without_provenance_mut::<u32, LANES>,
                &core::ptr::without_provenance_mut::<u32>,
                &|_| true,
            );
        }

    }
}
