#![feature(portable_simd)]
#![feature(f16)]

macro_rules! layout_tests {
    { $($mod:ident, $ty:ty,)* } => {
        $(
        mod $mod {
            fn check<const LANES: usize>() {
                assert_eq!(
                    core::mem::size_of::<core_simd::simd::Simd<$ty, LANES>>(),
                    core::mem::size_of::<[$ty; LANES]>(),
                );
                assert!(
                    core::mem::align_of::<core_simd::simd::Simd<$ty, LANES>>()
                        >= core::mem::align_of::<[$ty; LANES]>(),
                    "SIMD alignment must satisfy the array reference alignment for {} lanes",
                    LANES,
                );
            }

            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
            fn every_supported_lane_count_has_array_layout_compatibility() {
                check::<1>();
                check::<2>();
                check::<3>();
                check::<4>();
                check::<5>();
                check::<6>();
                check::<7>();
                check::<8>();
                check::<9>();
                check::<10>();
                check::<11>();
                check::<12>();
                check::<13>();
                check::<14>();
                check::<15>();
                check::<16>();
                check::<17>();
                check::<18>();
                check::<19>();
                check::<20>();
                check::<21>();
                check::<22>();
                check::<23>();
                check::<24>();
                check::<25>();
                check::<26>();
                check::<27>();
                check::<28>();
                check::<29>();
                check::<30>();
                check::<31>();
                check::<32>();
                check::<33>();
                check::<34>();
                check::<35>();
                check::<36>();
                check::<37>();
                check::<38>();
                check::<39>();
                check::<40>();
                check::<41>();
                check::<42>();
                check::<43>();
                check::<44>();
                check::<45>();
                check::<46>();
                check::<47>();
                check::<48>();
                check::<49>();
                check::<50>();
                check::<51>();
                check::<52>();
                check::<53>();
                check::<54>();
                check::<55>();
                check::<56>();
                check::<57>();
                check::<58>();
                check::<59>();
                check::<60>();
                check::<61>();
                check::<62>();
                check::<63>();
                check::<64>();
            }
        }
        )*
    }
}

layout_tests! {
    f16, f16,
    i8, i8,
    i16, i16,
    i32, i32,
    i64, i64,
    isize, isize,
    u8, u8,
    u16, u16,
    u32, u32,
    u64, u64,
    usize, usize,
    f32, f32,
    f64, f64,
    mut_ptr, *mut (),
    const_ptr, *const (),
}
