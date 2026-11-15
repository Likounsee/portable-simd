#![feature(portable_simd)]

//! Diagnostic benchmark for masked contiguous SIMD loads and stores.
//!
//! Reproducible baseline (use the toolchain pinned by this repository):
//!   cargo build -p core_simd --features as_crate --lib --message-format=json > /tmp/core-simd-build.json
//!   mapfile -t CORE_SIMD_ARTIFACTS < <(python3 - <<'PY'
//!   import json
//!   with open("/tmp/core-simd-build.json") as build_output:
//!       for line in build_output:
//!           try:
//!               message = json.loads(line)
//!           except json.JSONDecodeError:
//!               continue
//!           target = message.get("target", {})
//!           if message.get("reason") == "compiler-artifact" and target.get("name") == "core_simd" and "lib" in target.get("kind", []):
//!               filenames = message.get("filenames", [])
//!               rlib = next((name for name in filenames if name.endswith(".rlib")), None)
//!               rmeta = next((name for name in filenames if name.endswith(".rmeta")), None)
//!               if rlib and rmeta:
//!                   print(rlib)
//!                   print(rmeta)
//!                   raise SystemExit(0)
//!   raise SystemExit("Cargo did not report both core_simd rlib and rmeta artifacts")
//!   PY
//!   )
//!   CORE_SIMD_RLIB="${CORE_SIMD_ARTIFACTS[0]}"
//!   CORE_SIMD_RMETA="${CORE_SIMD_ARTIFACTS[1]}"
//!   rustc --edition=2024 -O -C target-cpu=x86-64 -C target-feature=+sse2 \
//!     --extern core_simd="$CORE_SIMD_RLIB" --extern core_simd="$CORE_SIMD_RMETA" \
//!     -L dependency=target/debug/deps benchmarks/masked_load_store.rs -o /tmp/masked-load-store
//!
//! Inspect optimized assembly and LLVM IR with the same --extern / -L flags and
//! replace the output option with --emit=asm or --emit=llvm-ir.
//! This benchmark links the fork's core_simd crate; it is for code-generation diagnosis,
//! not an end-to-end benchmark.

use core_simd::simd::{Mask, Simd};
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
fn simd_load_all(slice: &[f32]) -> Simd<f32, LANES> {
    Simd::load_select(slice, Mask::splat(true), Simd::splat(FALLBACK))
}

#[inline(never)]
fn scalar_load_all(slice: &[f32]) -> Simd<f32, LANES> {
    let mut result = [FALLBACK; LANES];
    for (&source, destination) in slice.iter().zip(result.iter_mut()) {
        *destination = source;
    }
    Simd::from_array(result)
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
fn simd_store_all(value: Simd<f32, LANES>, slice: &mut [f32]) {
    value.store_select(slice, Mask::splat(true));
}

#[inline(never)]
fn scalar_store(value: Simd<f32, LANES>, slice: &mut [f32], enabled: [bool; LANES]) {
    for lane in 0..LANES {
        if enabled[lane] {
            if let Some(destination) = slice.get_mut(lane) {
                *destination = value.as_array()[lane];
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

    // Compare all-enabled load_select with the scalar prefix loop from issue #494.
    let simd_loaded_all = simd_load_all(slice);
    let scalar_loaded_all = scalar_load_all(slice);
    assert_eq!(
        simd_loaded_all, scalar_loaded_all,
        "all-enabled load mismatch: len={len}"
    );

    // Match the exact load_or_default API and zip-based scalar loop from issue #494.
    let simd_loaded = simd_load_or_default(slice);
    let scalar_loaded = scalar_load_or_default(slice);
    assert_eq!(
        simd_loaded, scalar_loaded,
        "default load mismatch: len={len}"
    );

    // Compare all-enabled store_select with the scalar zip loop from issue #494.
    let mut simd_output = [SENTINEL; LANES];
    let mut scalar_output = [SENTINEL; LANES];
    simd_store_all(value, &mut simd_output[..len]);
    scalar_store_all(value, &mut scalar_output[..len]);
    assert_eq!(
        simd_output, scalar_output,
        "all-enabled store mismatch: len={len}"
    );

    println!("\nlen={len}, mode=all-enabled/default");
    measure("SIMD load_select (all)", || {
        black_box(simd_load_all(black_box(slice)));
    });
    measure("scalar prefix load", || {
        black_box(scalar_load_all(black_box(slice)));
    });
    measure("SIMD load_or_default", || {
        black_box(simd_load_or_default(black_box(slice)));
    });
    measure("scalar zip load", || {
        black_box(scalar_load_or_default(black_box(slice)));
    });
    measure("SIMD store_select (all)", || {
        simd_store_all(black_box(value), black_box(&mut simd_output[..len]));
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
    assert_eq!(
        simd_loaded, scalar_loaded,
        "load mismatch: len={len}, mask={mask_name}"
    );

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
        simd_store(
            black_box(value),
            black_box(&mut simd_output[..len]),
            black_box(enabled),
        );
        black_box(&simd_output);
    });
    measure("scalar masked store", || {
        scalar_store(
            black_box(value),
            black_box(&mut scalar_output[..len]),
            black_box(enabled),
        );
        black_box(&scalar_output);
    });
}

fn main() {
    let source = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let masks = [
        ("all", [true; LANES]),
        ("prefix", [true, true, true, true, false, false, false, false]),
        (
            "alternating",
            [true, false, true, false, true, false, true, false],
        ),
        (
            "sparse",
            [false, false, true, false, false, false, true, false],
        ),
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
