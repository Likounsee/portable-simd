#![feature(portable_simd)]

//! Portable SIMD slice-sum example for issue #110.
//!
//! Build core_simd with the pinned nightly and link this standalone example
//! against its rlib. The SIMD implementation handles full vectors and then
//! sums the tail scalarly, so every input length is supported.

use core_simd::simd::{num::SimdUint, Simd};
use std::hint::black_box;
use std::time::Instant;

const LANES: usize = 8;
const ITERATIONS: u32 = 200_000;

fn scalar_sum(values: &[u64]) -> u64 {
    values.iter().copied().sum()
}

fn simd_sum(values: &[u64]) -> u64 {
    let (chunks, remainder) = values.split_at(values.len() / LANES * LANES);
    let mut total = 0_u64;

    for chunk in chunks.chunks_exact(LANES) {
        let lanes = Simd::<u64, LANES>::from_array(
            chunk.try_into().expect("chunks_exact returns full SIMD-width chunks"),
        );
        total += lanes.reduce_sum();
    }

    total + remainder.iter().copied().sum::<u64>()
}

fn measure(label: &str, mut operation: impl FnMut() -> u64) {
    let start = Instant::now();
    let mut checksum = 0_u64;
    for _ in 0..ITERATIONS {
        checksum = checksum.wrapping_add(black_box(operation()));
    }
    let elapsed = start.elapsed();
    black_box(checksum);
    println!(
        "{label:18} {:>10.3} ns/iteration ({ITERATIONS} iterations)",
        elapsed.as_nanos() as f64 / f64::from(ITERATIONS)
    );
}

fn main() {
    let values: Vec<u64> = (0..10_003).map(|value| (value % 97) as u64).collect();
    let expected = scalar_sum(&values);
    assert_eq!(simd_sum(&values), expected);

    // Exercise empty input, exact vectors, and every short tail length.
    for len in 0..=(LANES * 2 + 1) {
        assert_eq!(
            simd_sum(&values[..len]),
            scalar_sum(&values[..len]),
            "sum mismatch for length {len}",
        );
    }

    println!("portable-simd issue #110: slice sum");
    println!("elements={}, lanes={LANES}, iterations={ITERATIONS}", values.len());
    measure("scalar sum", || scalar_sum(black_box(&values)));
    measure("SIMD sum", || simd_sum(black_box(&values)));
}
