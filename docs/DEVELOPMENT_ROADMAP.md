# portable-simd development roadmap

This document tracks proposed contribution work in the `Likounsee/portable-simd` fork.
The upstream project remains `rust-lang/portable-simd`; changes intended for the Rust
project should be submitted there as focused pull requests.

## Working principles

- Preserve the upstream API and implementation conventions unless a change is explicitly
  intended to address an API design issue.
- Prefer small, independently reviewable changes over a broad rewrite.
- Reproduce a performance report before changing implementation or code generation.
- Keep correctness tests separate from benchmarks; never use benchmark results as proof
  of correctness.
- Test more than one target architecture when the change affects portable behavior.
- Do not claim a speedup without recording compiler version, target, flags, and benchmark
  methodology.
- Work only on this fork's `main` branch; do not create additional working branches.

## Repository baseline

The workspace currently contains three crates:

- `crates/core_simd`: the portable SIMD API and its implementation.
- `crates/std_float`: SIMD floating-point operations.
- `crates/test_helpers`: shared test support.

The workspace uses the nightly toolchain pinned in `rust-toolchain.toml`.
The CI workflow runs formatting, Clippy, and target-specific test/documentation jobs.

The fork's `main` is compared directly against `rust-lang:main`; see the
[live comparison](https://github.com/Likounsee/portable-simd/compare/rust-lang:main...main).
The fork-only work includes the diagnostic benchmark, regression tests, and this roadmap.
The comparison does not establish that the code compiles or passes CI.

## Priority work

### 1. Establish a reproducible baseline

- [x] Compare the fork's `main` with upstream `main` (live comparison linked above).
- [ ] Run the repository's documented test and formatting commands.
- [ ] Record the available local target and CPU features.
- [x] Build a diagnostic benchmark covering slice lengths 0 through 8 and all-enabled,
      alternating, and sparse masks, with scalar reference loops and semantic assertions.
- [ ] Run the benchmark on the pinned nightly and save compiler version, target triple,
      target features, timings, optimized LLVM IR, and generated assembly.

### 2. Investigate masked load/store code generation

Related upstream issue: [#494](https://github.com/rust-lang/portable-simd/issues/494).

- [x] Read the current implementations of `load_or_default`, `store_select`,
      `gather_or_default`, and `scatter`. The safe contiguous load/store APIs clamp
      their masks to the slice length before calling the pointer-level SIMD intrinsics.
- [x] Recheck upstream issue status and comments: still open, with no issue comments
      returned by the current GitHub check.
- [ ] Reproduce the reported x86-64 SSE2 behavior using the issue's examples.
- [ ] Compare the SIMD API implementation with equivalent scalar loops.
- [ ] Inspect optimized LLVM IR and assembly to identify the actual cost.
- [ ] Determine whether the fix belongs in this crate, LLVM, or rustc codegen.
- [ ] Only then propose a minimal implementation change and regression test.

The current benchmark is diagnostic only. It has not been compiled or executed, so it
provides no performance evidence and must not be used to claim an optimization.

### 3. Investigate mask-operation code generation

Related upstream issue: [#472](https://github.com/rust-lang/portable-simd/issues/472).

- [ ] Reproduce the hexadecimal-validation example on a current nightly compiler.
- [ ] Inspect comparison, bitwise mask combination, and reduction code generation.
- [ ] Compare against a hand-written target-specific implementation.
- [ ] Identify whether the missed optimization is caused by the API lowering,
      LLVM optimization, or target instruction selection.

### 4. Mask ordering regression coverage

Related upstream issue: [#527](https://github.com/rust-lang/portable-simd/issues/527) (closed
2026-10-26).

- [x] Confirm the current implementation uses boolean algebra for mask ordering
      (`!a & b`, `!a | b`, `a & !b`, `a | !b`) rather than signed integer comparisons.
- [x] Add a 64-lane regression test covering all four comparisons plus min/max,
      including the high bit and mixed patterns.
- [x] Add exhaustive coverage for all 65,536 pairs of 8-lane mask patterns,
      checking all four comparisons plus min/max.
- [x] Add bitmask round-trip coverage at every internal width transition (8, 16, 32, 64),
      plus odd lane counts and all-zero/all-one/patterned inputs. Expected lanes are
      constructed independently to avoid a round-trip-only false positive; all 65,536
      possible 16-lane bit patterns are also checked.
- [ ] Run the regression tests on the pinned nightly and across the CI target matrix.

### 5. Review API and soundness concerns before proposing changes

- [ ] Review the stabilization/API concerns in upstream issue #364.
- [x] Inspect the large-vector alignment report in #495 and the current `Simd` representation: `#[repr(simd, packed)]`; masked pointer loads/stores request element alignment. The issue remains open, and target layout measurements are still needed.
- [x] Review the visibility/API concern in upstream issue #537, including the full maintainer discussion.
- [ ] Review the unresolved Miri floating-point test determinism discussion in issue #463.
- [ ] Verify the current code and discussion before treating any old issue as unresolved.

Issue #537 is still open. The reported `PrivateMethods` supertrait exposure is present
in `masks.rs`. The full discussion narrows the concern: automatic stable-API exposure
is not established, because stabilization would require explicit stability annotations;
however, maintainers agree that the helper trait being callable downstream on nightly
is unintended. The issue author also clarified that the trait was previously named
`Sealed`. The implementation currently depends on its associated items from generic
`T: MaskElement` call sites (`TRUE`, `FALSE`, `valid`, `eq`, `Unsigned`, and conversion
helpers), so merely removing the supertrait would break internal code. A fix needs to
separate the sealing mechanism from implementation-only operations without re-exposing
them through the public trait, plus a downstream compile-fail test. No safe minimal
refactor has yet been established; do not change the trait hierarchy speculatively.

### 6. Floating-point test determinism

Related upstream issue: [#463](https://github.com/rust-lang/portable-simd/issues/463).

- [x] Inspect the issue and its five comments.
- [x] Confirm the current StdFloat test suite already uses approximate comparisons
      for transcendental functions such as sin, cos, exp, exp2, ln, log2, and log10,
      while retaining exact comparisons for several other operations.
- [ ] Determine whether the remaining Miri failures are still reproducible on the
      pinned nightly and identify the exact failing test cases.
- [ ] Discuss a narrowly scoped Miri-only strategy with maintainers before changing
      exact comparisons, because exact tests intentionally catch implementation regressions.

The discussion in #463 does not support a blanket switch to approximate assertions:
maintainers explicitly noted the value of exact comparisons for detecting regressions.
No test policy or tolerance change is made here.

### 7. Recent issue triage

- [x] #499: closed; current StdFloat::round documentation correctly says ties round away from zero.
- [x] #527: closed; regression coverage exists in this fork.
- [x] #524: closed after the reporter retested `swizzle_dyn` successfully on newer nightly for AArch64 macOS, x86-64 Linux, and RISC-V Linux. This was a toolchain/Miri-intrinsic support resolution, not a portable-simd source fix; AVX-512 and other Miri target coverage still need separate validation.
- [x] #519: closed; verified the fork's `crates/std_float/tests/float.rs` gates the `f16` test module with `#[cfg(target_has_reliable_f16_math)]`. The issue discussion attributes the original Windows GNU failure to missing f16 math symbols/ABI support, so retain the target gate unless compiler-builtins support changes.
- [x] #537: open; source and full discussion reviewed. A deliberate helper/sealing redesign and compile-fail test are still needed.
- [ ] #494: open; safe masked-memory semantics now have regression tests, but SSE2/AVX2 code generation remains unmeasured.
- [ ] #472: open; mask operation code generation remains uninvestigated.
- [ ] #505: open; API ergonomics request to rename `Select` to `SimdSelect` and consider prelude inclusion. This is a naming/export design question, not a demonstrated correctness bug; inspect the actual public export surface and naming conventions before proposing an API change.
- [ ] #495: open; large-vector alignment constraints.

## Contribution checklist

For each proposed change:

1. Link the upstream issue or explain the problem.
2. Add a minimal regression test where possible.
3. Include benchmark evidence for performance changes.
4. Run formatting, tests, and Clippy for relevant targets.
5. Document limitations and architecture-specific behavior.
6. Submit one focused pull request against upstream `main`.

## Status

This remains an investigation, not an optimization claim. The masked load/store call
chain has been inspected, scalar-oracle regression tests now cover short/empty slices
and masked/out-of-bounds gather/scatter lanes, and a diagnostic benchmark has been added.
Neither the tests nor the benchmark have yet been compiled or executed. Mask ordering now has both 64-lane
high-bit regression cases and exhaustive 8-lane pair coverage. Mask bitmask conversion
also has explicit round-trip coverage at the internal width transitions and odd lane
counts. These tests have not yet been run locally or through CI. No performance
results or implementation speedups are claimed.


## Masked memory semantic regression coverage

- [x] Add scalar-oracle tests for safe masked contiguous loads and stores over every
      slice length from zero through the vector width.
- [x] Cover disabled lanes and out-of-bounds gather/scatter indices, including
      `usize::MAX` and empty slices.
- [ ] Run these tests under the pinned nightly and Miri where supported.
- [ ] Continue codegen comparison for SSE2/AVX2; no performance claim until assembly
      and benchmark results are collected.

- [x] Extend masked gather coverage to every four-lane index tuple over valid,
      out-of-bounds, and `usize::MAX` indices, for all 16 enable masks.
- [x] Check masked scatter against a scalar oracle for every permutation of four
      distinct destinations and all 16 enable masks.


### 8. Dynamic byte swizzle correctness

Related upstream issue: [#486](https://github.com/rust-lang/portable-simd/issues/486).

- [x] Inspect the target-specific implementations in `swizzle_dyn.rs`, including
      x86 SSSE3/AVX2/AVX-512, AArch64 NEON, ARM NEON, WASM SIMD, and LoongArch.
- [x] Add deterministic scalar-oracle coverage that visits every possible `u8`
      index in every output lane at representative widths around the implementation
      boundaries (1, 7–9, 15–17, 23–24, 31–33, 47–49, 63–64).
- [ ] Run the test on the pinned nightly across the available architecture/feature
      matrix. The current test addition is not yet compiled or executed.
- [ ] Evaluate #486's proposed unchecked/implementation-defined out-of-bounds
      variant separately as an API design and safety-contract question; do not
      change the existing zeroing semantics.

The exhaustive test strengthens confidence in existing semantics. It does not
implement the optional API proposed in #486 and does not establish a performance
improvement.


- [x] Extend contiguous masked load/store scalar-oracle tests to lane counts around
      power-of-two boundaries (1–4, 7–9, 15–17, 31–33, 63–64), checking every
      slice length and all-disabled, all-enabled, alternating, and edge-lane masks.
- [ ] Run the expanded lane-width matrix on the pinned nightly and under Miri where
      supported.

- [x] Extend gather/scatter scalar-oracle checks to the same lane-count boundaries,
      every slice length, alternating valid and `usize::MAX` indices, and four mask
      patterns. Scatter destinations are unique so the expected result does not rely
      on unspecified duplicate-index write ordering.

### 9. Array/slice conversion code generation

Related upstream issues: [#343](https://github.com/rust-lang/portable-simd/issues/343) and [#341](https://github.com/rust-lang/portable-simd/issues/341). The older #341 report explains the padding/overread hazard and why a dedicated vector-load intrinsic was proposed; #343 asks for codegen regression tests.

- [x] Trace `Simd::from_array`, `to_array`, `from_slice`, and `copy_to_slice` to the internal `load`/`store` helpers.
- [x] Confirm the helpers use `ptr::copy_nonoverlapping` rather than a direct transmute or tuple-field access. The source explicitly documents that non-power-of-two `repr(simd)` vectors may contain trailing padding, so copying the full SIMD object from an `[T; N]` could read beyond the array.
- [ ] Inspect optimized LLVM IR for representative power-of-two and non-power-of-two lane counts, and verify that the element transfer becomes vector `load`/`store` operations without reading or writing padding as array elements.
- [ ] Add compiler-version-pinned codegen regression tests once the appropriate test harness and expected IR patterns are established.
- [ ] Check more than one element type and target; avoid brittle assembly assertions when an LLVM IR-level property is sufficient.

The implementation already contains the relevant safety rationale and an explicit FIXME about the current pointer-copy approach. The open question is whether the compiler consistently optimizes that approach as intended. No implementation rewrite is justified until IR is inspected.

### 10. `MaskElement` helper-trait visibility review

Related upstream issue: [#537](https://github.com/rust-lang/portable-simd/issues/537).

- [x] Confirm `private_methods::PrivateMethods` is declared with parent-module visibility and is a supertrait of public `MaskElement`.
- [x] Trace its current internal use: mask construction and validation use `TRUE`/`FALSE` and `valid`; mask indexing uses `eq`; bitmask conversion uses `Unsigned`, `max_unsigned`, `eq`, and `to_usize`.
- [x] Read the issue discussion: the reported concern is unintended nightly accessibility through the public subtrait, not an established stable-API break. A maintainer notes stabilization would require explicit stability annotations, but agrees nightly accessibility should be fixed.
- [ ] Determine a design that keeps the sealing contract while making implementation-only helpers inaccessible downstream. Any split must preserve the generic helper operations currently available under a `T: MaskElement` bound.
- [ ] Add a compile-fail/UI regression test demonstrating that downstream code cannot call the helper items, while internal mask operations continue to compile.

Do not simply delete `PrivateMethods` from the `MaskElement` supertrait: the current implementation relies on those associated items at several generic call sites. A safe fix needs a deliberate internal-helper design and compile-fail coverage.



### 11. Additional correctness and API issue triage

- [x] Recheck [#427](https://github.com/rust-lang/portable-simd/issues/427): the report concerns `simd_div` with zero divisors and `simd_rem` with minimum signed values for non-power-of-two vectors on AArch64. The issue still lacks a minimal reproducer; the only follow-up asks about the historical `repr(simd, packed)` layout and element type, and the reporter could not recall the exact type.
- [ ] Do not add a speculative workaround for #427. First construct a minimal reproducer on AArch64, establish whether the current representation still triggers the LLVM ICE, and preserve the exact compiler/LLVM versions and failing operation.
- [x] Triage [#539](https://github.com/rust-lang/portable-simd/issues/539): this is an API request for SIMD counterparts to scalar `algebraic_*` float operations, with a possible algebraic reduction. The request does not yet define the exact operation set, semantics, or interaction with the current `StdFloat` trait.
- [ ] Before implementing #539, compare the scalar algebraic float API and its stability/semantics with the existing `StdFloat` methods, then propose a minimal method set and tests for relaxed floating-point behavior. Do not silently give existing strict operations relaxed semantics.


### 12. Miri target-intrinsic issue follow-up

Related upstream issue: [#524](https://github.com/rust-lang/portable-simd/issues/524) (closed).

- [x] Read the full issue discussion, including the reporter's closing retest and maintainer comments.
- [x] Confirm the reporter retested `swizzle_dyn` successfully on newer nightly for `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`, and `riscv64a23-unknown-linux-gnu`.
- [x] Record that the resolution came from newer Miri/intrinsic support and not from a source change in this fork.
- [ ] Keep AVX-512 and other target configurations separate: a maintainer noted that some AVX-512 paths may still require Miri shims, particularly with `-Zbuild-std`.
- [ ] If Miri coverage is expanded, first test the repository's actual target/configuration matrix; do not add a broad `cfg(miri)` fallback solely on the basis of this closed report.

This closes the original reproducer as reported, but it does not establish universal Miri support for every architecture-specific `swizzle_dyn` implementation.


### 13. Float16 test-target gate verification

Related upstream issue: [#519](https://github.com/rust-lang/portable-simd/issues/519) (closed).

- [x] Read the full issue discussion, including the explanation that Windows GNU lacked the relevant reliable f16 math support/symbols.
- [x] Inspect the fork's current `crates/std_float/tests/float.rs`: the `f16` test instantiation is guarded by `#[cfg(target_has_reliable_f16_math)]`, while `f32` and `f64` remain unconditional.
- [x] Confirm the expected target gate is already present; no duplicate test workaround is needed.
- [ ] Revisit only if the compiler target capability or compiler-builtins support changes and the gate can safely be narrowed/removed.

This is a verified existing safeguard, not a new code change.


#### Design check: upstream PR #483 is not a narrow visibility fix

- [x] Inspect the full diff and discussion of [PR #483](https://github.com/rust-lang/portable-simd/pull/483), "Mask element v2".
- [x] Confirm its `Sealed` trait still declares `valid`, `eq`, `to_usize`, `max_unsigned`, `Unsigned`, `TRUE`, and `FALSE`, and remains a supertrait of `MaskElement`.
- [x] Reject a rename-only approach: changing `PrivateMethods` to `Sealed` does not remove the downstream-visible associated items described by #537.
- [x] Identify that PR #483 also changes the generic meaning of `Mask<T, N>` to use `T: SimdElement` and `T::Mask`, changes numerous public bounds and implementations, and is therefore a broad API redesign rather than a safe, isolated patch for #537.
- [ ] Develop a smaller design that keeps a method-free sealing supertrait and relocates internal operations without adding private bounds to public method signatures or changing the public `Mask<T, N>` API.
- [ ] Prove the design with a downstream compile-fail test for every leaked associated item category (constant, associated type, and method), plus ordinary internal mask tests.

This investigation narrows the solution space but does not yet justify a source edit. In particular, do not rename the trait and call the issue fixed.
