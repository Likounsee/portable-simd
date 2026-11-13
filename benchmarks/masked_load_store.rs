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

use std::simd::{Mask, Simd};
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
fn simd_load_or_default(slice: &[f32]) -> Simd<f32, LANES> {
    Simd::load_or_default(slice)
}

#[inline(never)]
fn scalar_load_or_default(slice: &[f32]) -> Simd<f32, LANES> {
    let mut result = [0.0; LANES];
    for (&source, destination) in slice.iter().zip(result.iter_mut()) {
        *destination = source;
    }
    Simd::from_array(result)
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

#[inline(never)]
fn scalar_store_all(value: Simd<f32, LANES>, slice: &mut [f32]) {
    for (destination, &source) in slice.iter_mut().zip(value.as_array()) {
        *destination = source;
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

fn run_default_case(source: &[f32; LANES], len: usize) {
    let slice = &source[..len];
    let value = Simd::from_array([10.0_f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);

    // Match the exact load_or_default API and zip-based scalar loop from issue #494.
    let simd_loaded = simd_load_or_default(slice);
    let scalar_loaded = scalar_load_or_default(slice);
    assert_eq!(simd_loaded, scalar_loaded, "default load mismatch: len={len}");

    // Compare all-enabled store_select with the scalar zip loop from issue #494.
    let mut simd_output = [SENTINEL; LANES];
    let mut scalar_output = [SENTINEL; LANES];
    simd_store(value, &mut simd_output[..len], [true; LANES]);
    scalar_store_all(value, &mut scalar_output[..len]);
    assert_eq!(simd_output, scalar_output, "all-enabled store mismatch: len={len}");

    println!("\nlen={len}, mode=all-enabled/default");
    measure("SIMD load_or_default", || {
        black_box(simd_load_or_default(black_box(slice)));
    });
    measure("scalar zip load", || {
        black_box(scalar_load_or_default(black_box(slice)));
    });
    measure("SIMD store_select (all)", || {
        simd_store(black_box(value), black_box(&mut simd_output[..len]), [true; LANES]);
        black_box(&simd_output);
    });
    measure("scalar zip store", || {
        scalar_store_all(black_box(value), black_box(&mut scalar_output[..len]));
        black_box(&scalar_output);
    });
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
        run_default_case(&source, len);
        for &(mask_name, enabled) in &masks {
            run_case(&source, len, enabled, mask_name);
        }
    }
}
