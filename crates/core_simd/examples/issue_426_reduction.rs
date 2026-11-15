#![feature(portable_simd)]

//! Reproducer for upstream issue #426.
//!
//! Run with:
//!   cargo run --release -p core_simd --example issue_426_reduction
//!
//! The three implementations perform the same calculation over interleaved
//! [x, y, z, unused] records. The SIMD variants deliberately differ only in
//! where the horizontal reduction happens. This is a diagnostic benchmark,
//! not a promise that SIMD wins on every target.

use core_simd::simd::prelude::*;
use std::hint::black_box;
use std::time::Instant;

const POSITION: [f32; 4] = [2000.0, 2000.0, 2000.0, 0.0];
const DIRECTION: [f32; 4] = [0.8, 0.6, 0.0, 0.0];
const RECORD_WIDTH: usize = 4;

#[inline]
fn scalar(data: &[f32]) -> f32 {
    assert_eq!(data.len() % RECORD_WIDTH, 0);

    let mut sum = 0.0_f32;
    for record in data.chunks_exact(RECORD_WIDTH) {
        let x = record[0];
        let y = record[1];
        let z = record[2];
        sum += (x - POSITION[0]) * DIRECTION[0]
            + (y - POSITION[1]) * DIRECTION[1]
            + (z - POSITION[2]) * DIRECTION[2];
    }
    sum
}

#[inline]
fn simd_reduce_each_iteration(data: &[f32]) -> f32 {
    assert_eq!(data.len() % RECORD_WIDTH, 0);

    let position = f32x4::from_array(POSITION);
    let direction = f32x4::from_array(DIRECTION);
    let mut sum = 0.0_f32;

    for record in data.chunks_exact(RECORD_WIDTH) {
        let values = f32x4::from_array(record.try_into().unwrap());
        sum += ((values - position) * direction).reduce_sum();
    }
    sum
}

#[inline]
fn simd_accumulate_then_reduce(data: &[f32]) -> f32 {
    assert_eq!(data.len() % RECORD_WIDTH, 0);

    let position = f32x4::from_array(POSITION);
    let direction = f32x4::from_array(DIRECTION);
    let mut sums = f32x4::splat(0.0);

    for record in data.chunks_exact(RECORD_WIDTH) {
        let values = f32x4::from_array(record.try_into().unwrap());
        sums += (values - position) * direction;
    }

    sums.reduce_sum()
}

fn measure(label: &str, data: &[f32], operation: impl Fn(&[f32]) -> f32) {
    // Warm up and keep the result observable so the loop cannot be removed.
    black_box(operation(black_box(data)));

    let start = Instant::now();
    let result = black_box(operation(black_box(data)));
    let elapsed = start.elapsed();

    println!("{label:32} {elapsed:>12.3?}  result={result}");
}

fn main() {
    // Keep the default run quick enough for a local diagnostic. Increase this
    // value when comparing optimized code on a stable, otherwise idle machine.
    let records = 1_000_000;
    let mut data = Vec::with_capacity(records * RECORD_WIDTH);
    for i in 0..records {
        let x = 1800.0 + (i % 401) as f32 * 0.25;
        let y = 1900.0 + (i % 307) as f32 * 0.5;
        let z = 1700.0 + (i % 211) as f32 * 0.125;
        data.extend_from_slice(&[x, y, z, 0.0]);
    }

    let reference = scalar(&data);
    let reduce_each = simd_reduce_each_iteration(&data);
    let accumulate = simd_accumulate_then_reduce(&data);

    // Floating-point reassociation changes rounding, so compare with a
    // tolerance rather than requiring bit-for-bit equality.
    let tolerance = reference.abs().max(1.0) * 2.0e-5;
    assert!(
        (reference - reduce_each).abs() <= tolerance,
        "per-iteration SIMD reduction differs: scalar={reference}, SIMD={reduce_each}"
    );
    assert!(
        (reference - accumulate).abs() <= tolerance,
        "vector accumulation differs: scalar={reference}, SIMD={accumulate}"
    );

    println!("Records: {records}");
    measure("scalar", &data, scalar);
    measure("SIMD reduce every iteration", &data, simd_reduce_each_iteration);
    measure("SIMD accumulate then reduce", &data, simd_accumulate_then_reduce);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_variants_agree_for_small_input() {
        let data = [
            2001.0, 2002.0, 2003.0, 0.0, 1999.0, 2004.0, 2005.0, 0.0, 2010.0, 1990.0,
            2020.0, 0.0, 1980.0, 2030.0, 1970.0, 0.0,
        ];
        let expected = scalar(&data);
        let tolerance = expected.abs().max(1.0) * 2.0e-5;

        for actual in [
            simd_reduce_each_iteration(&data),
            simd_accumulate_then_reduce(&data),
        ] {
            assert!(
                (expected - actual).abs() <= tolerance,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    #[should_panic]
    fn rejects_incomplete_record() {
        let _ = simd_accumulate_then_reduce(&[1.0, 2.0, 3.0]);
    }
}
