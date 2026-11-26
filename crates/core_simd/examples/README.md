### `core_simd` examples

These examples demonstrate uses of the portable SIMD API. Several were originally adapted from the `packed_simd` crate.

Run the example regression tests with:

```sh
cargo test -p core_simd --example dot_product
cargo test -p core_simd --example mandelbrot
```

The `mandelbrot` example renders the escape-time fractal four pixels at a time
with portable SIMD, freezes lanes after they escape, and tests its results
against a scalar reference (including image dimensions that are not multiples
of the SIMD lane count).
