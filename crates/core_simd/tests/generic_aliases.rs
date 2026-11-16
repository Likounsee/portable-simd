#![feature(portable_simd)]
#![feature(f16)]

use core_simd::simd::*;

#[test]
fn generic_lane_aliases_match_simd() {
    let _: i8xN<4> = Simd::<i8, 4>::splat(0);
    let _: i16xN<4> = Simd::<i16, 4>::splat(0);
    let _: i32xN<4> = Simd::<i32, 4>::splat(0);
    let _: i64xN<4> = Simd::<i64, 4>::splat(0);
    let _: isizexN<4> = Simd::<isize, 4>::splat(0);

    let _: u8xN<4> = Simd::<u8, 4>::splat(0);
    let _: u16xN<4> = Simd::<u16, 4>::splat(0);
    let _: u32xN<4> = Simd::<u32, 4>::splat(0);
    let _: u64xN<4> = Simd::<u64, 4>::splat(0);
    let _: usizexN<4> = Simd::<usize, 4>::splat(0);

    let _: f16xN<4> = Simd::<f16, 4>::splat(0.0);
    let _: f32xN<4> = Simd::<f32, 4>::splat(0.0);
    let _: f64xN<4> = Simd::<f64, 4>::splat(0.0);
}

#[test]
fn generic_lane_aliases_work_with_const_generics() {
    fn double<const N: usize>(value: u32xN<N>) -> u32xN<N> {
        value + u32xN::<N>::splat(1)
    }

    assert_eq!(double(u32xN::<4>::splat(41)).to_array(), [42; 4]);
}

#[test]
fn generic_float_aliases_work_with_const_generics() {
    fn scale<const N: usize>(value: f32xN<N>) -> f32xN<N> {
        value * f32xN::<N>::splat(2.0)
    }

    let input: f32xN<8> = f32xN::<8>::splat(1.25);
    assert_eq!(scale(input).to_array(), [2.5; 8]);
}
