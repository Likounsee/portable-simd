#![feature(portable_simd)]

//! Diagnostic benchmark for masked contiguous SIMD loads and stores.
//!
//! Reproducible baseline (use the toolchain pinned by this repository):
//!   rustc +nightly-2026-11-02 -O -C target-cpu=x86-64 -C target-feature=+sse2 \
//!     benchmarks/masked_load_store.rs -o /tmp/masked-load-store
//!   /tmp/masked-load-store
//!
//! Inspect optimized assembly:
//!   rustc +nightly-2026-11-02 -O -C target-cpu=x86-64 -C target-feature=+sse2 \
//!     --emit=asm -o /tmp/masked-load-store.s benchmarks/masked_load_store.rs
//!
//! Inspect optimized LLVM IR:
//!   rustc +nightly-2026-11-02 -O -C target-cpu=x86-64 -C target-feature=+sse2 \
//!     --emit=llvm-ir -o /tmp/masked-load-store.ll benchmarks/masked_load_store.rs
//!
//! Record rustc --version --verbose, target triple, CPU model, and flags with results.
//! This is a microbenchmark for code-generation diagnosis, not an end-to-end benchmark.

use core::simd::{Mask, Simd};
use std::hint::black_box;
use std::time::{Duration, Instant};

const LANES: usize = 8;
const ITERATIONS: u32 = 500_000;
const FALLBACK: f32 = -1.0;
const SENTINEL: f32 = 42.0;

#[inline(never)]
fn simd_load(slice: &[f32], enabled: [bool; LANES]) -> Simd<f32, LANES> {
    Simd::load_select(slice, Mask::from_array(enabled), Simd::splat(FALLBACK))
}

#[inline(never)]
fn scalar_load(slice: &[f32], enabled: [bool; LANES]) -> Simd<f32, LANES> {
    let mut result = [FALLBACK; LANES];
    for lane in 0..LANES {
        if enabled[lane] {
            if let Some(&value) = slice.get(lane) {
                result[lane] = value;
            }
        }
    }
    Simd::from_array(result)
}

#[inline(never)]
fn simd_store(value: Simd<f32, LANES>, slice: &mut [f32], enabled: [bool; LANES]) {
    value.store_select(slice, Mask::from_array(enabled));
}

#[inline(never)]
fn scalar_store(value: Simd<f32, LANES>, slice: &mut [f32], enabled: [bool; LANES]) {
    for lane in 0..LANES {
        if enabled[lane] {
            if let Some(destination) = slice.get_mut(lane) {
                *destination = value[lane];
            }
        }
    }
}

fn measure(label: &str, mut operation: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        operation();
    }
    let elapsed = start.elapsed();
    println!(
        "{label:28} {:>10.3} ns/iteration ({ITERATIONS} iterations)",
        elapsed.as_nanos() as f64 / f64::from(ITERATIONS)
    );
    elapsed
}

fn run_case(source: &[f32; LANES], len: usize, enabled: [bool; LANES], mask_name: &str) {
    let slice = &source[..len];
    let value = Simd::from_array([10.0_f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);

    // Check load semantics, including disabled and out-of-bounds lanes.
    let simd_loaded = simd_load(slice, enabled);
    let scalar_loaded = scalar_load(slice, enabled);
    assert_eq!(simd_loaded, scalar_loaded, "load mismatch: len={len}, mask={mask_name}");

    // Check store semantics without allowing disabled/out-of-bounds writes.
    let mut simd_output = [SENTINEL; LANES];
    let mut scalar_output = [SENTINEL; LANES];
    simd_store(value, &mut simd_output[..len], enabled);
    scalar_store(value, &mut scalar_output[..len], enabled);
    assert_eq!(
        simd_output, scalar_output,
        "store mismatch: len={len}, mask={mask_name}"
    );

    println!("\nlen={len}, mask={mask_name}");
    measure("SIMD load_select", || {
        black_box(simd_load(black_box(slice), black_box(enabled)));
    });
    measure("scalar masked load", || {
        black_box(scalar_load(black_box(slice), black_box(enabled)));
    });
    measure("SIMD store_select", || {
        simd_store(black_box(value), black_box(&mut simd_output[..len]), black_box(enabled));
        black_box(&simd_output);
    });
    measure("scalar masked store", || {
        scalar_store(black_box(value), black_box(&mut scalar_output[..len]), black_box(enabled));
        black_box(&scalar_output);
    });
}

fn main() {
    let source = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let masks = [
        ("all", [true; LANES]),
        ("alternating", [true, false, true, false, true, false, true, false]),
        ("sparse", [false, false, true, false, false, false, true, false]),
    ];

    println!("portable-simd masked load/store diagnostic");
    println!("lanes={LANES}, iterations={ITERATIONS}, fallback={FALLBACK}");
    for len in 0..=LANES {
        for &(mask_name, enabled) in &masks {
            run_case(&source, len, enabled, mask_name);
        }
    }
}
