#![feature(portable_simd)]

//! Diagnostic benchmark for issue #475 (unchecked SIMD gather).
//!
//! Build the core_simd crate with the pinned nightly, then compile this file
//! against its rlib with optimization enabled. Inspect the optimized assembly
//! to confirm that the unchecked path does not retain per-lane bounds checks.
//! Timing is diagnostic only and must be recorded together with rustc, target,
//! CPU, and target-feature settings.

use core_simd::simd::Simd;
use std::hint::black_box;
use std::time::Instant;

const LANES: usize = 8;
const ITERATIONS: u32 = 1_000_000;

#[inline(never)]
fn checked(values: &[u32; LANES], indices: Simd<usize, LANES>) -> Simd<u32, LANES> {
    Simd::gather_or_default(values, indices)
}

#[inline(never)]
fn unchecked(values: &[u32; LANES], indices: Simd<usize, LANES>) -> Simd<u32, LANES> {
    // SAFETY: the fixed indices used by main are all below LANES.
    unsafe { Simd::gather_unchecked(values, indices) }
}

#[inline(never)]
fn pointer_gather(values: &[u32; LANES], indices: Simd<usize, LANES>) -> Simd<u32, LANES> {
    // SAFETY: the fixed indices used by main are all below LANES.
    let pointers = Simd::splat(values.as_ptr()).wrapping_add(indices);
    unsafe { Simd::gather_ptr(pointers) }
}


#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn avx2_gather(values: &[u32; LANES], indices: Simd<i32, LANES>) -> Simd<u32, LANES> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64 as x86;

    // SAFETY: The caller must ensure AVX2 is available. The indices are valid
    // element offsets, and scale=4 addresses u32 elements.
    unsafe {
        let indices: x86::__m256i = indices.into();
        x86::_mm256_i32gather_epi32::<4>(values.as_ptr().cast::<i32>(), indices).into()
    }
}

fn measure(label: &str, mut f: impl FnMut()) {
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        f();
    }
    let elapsed = start.elapsed();
    println!(
        "{label:24} {:>10.3} ns/iteration ({ITERATIONS} iterations)",
        elapsed.as_nanos() as f64 / f64::from(ITERATIONS)
    );
}

fn main() {
    let values = [11_u32, 23, 37, 41, 53, 67, 79, 83];
    let indices = Simd::from_array([7_usize, 0, 5, 2, 6, 1, 4, 3]);
    let expected = Simd::from_array([83, 11, 67, 37, 79, 23, 53, 41]);

    assert_eq!(checked(&values, indices), expected);
    assert_eq!(unchecked(&values, indices), expected);
    assert_eq!(pointer_gather(&values, indices), expected);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let avx2_indices = Simd::from_array([7_i32, 0, 5, 2, 6, 1, 4, 3]);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature detection proves AVX2 is available.
        assert_eq!(unsafe { avx2_gather(&values, avx2_indices) }, expected);
    }


    println!("portable-simd issue #475 gather diagnostic");
    println!("lanes={LANES}, iterations={ITERATIONS}");
    measure("checked gather", || {
        black_box(checked(black_box(&values), black_box(indices)));
    });
    measure("unchecked gather", || {
        black_box(unchecked(black_box(&values), black_box(indices)));
    });
    measure("pointer gather", || {
        black_box(pointer_gather(black_box(&values), black_box(indices)));
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        measure("AVX2 hardware gather", || {
            // SAFETY: runtime feature detection proves AVX2 is available.
            black_box(unsafe { avx2_gather(black_box(&values), black_box(avx2_indices)) });
        });
    }

}
