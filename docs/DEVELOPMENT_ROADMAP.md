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

Issue #537 remains open upstream. The fork now contains a focused candidate fix
in `crates/core_simd/src/masks.rs`: `PrivateMethods` is declared as an ordinary
module-private trait directly in `masks.rs`, rather than as a `pub` trait inside a
private child module. It remains a private supertrait of `MaskElement`, so the built-in
mask types stay sealed while the helper items are no longer publicly reachable through
the public trait. The existing generic mask operations continue to use the helper
inside the same module. Three compile-fail doctests attempt downstream access to
`TRUE`, `Unsigned`, and `eq`. This is a candidate implementation, not yet validated:
the doctests and crate have not been compiled locally or by CI. The distinction in the
upstream discussion remains important: maintainers did not establish an automatic
stable-API break, but agreed that nightly downstream access was unintended.

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
- [x] #537: open; source and full discussion reviewed. A focused private-supertrait
      candidate fix and three downstream compile-fail doctests are now present on the
      fork's `main`; execution under the pinned nightly is still pending.
- [ ] #494: open; safe masked-memory semantics now have regression tests, but SSE2/AVX2 code generation remains unmeasured.
- [ ] #472: open; mask operation code generation remains uninvestigated.
- [x] #505: open upstream, but the prelude concern is already resolved by upstream PR #506. The fork's `simd` module should be checked against that export before any work; the remaining discussion is only whether the generic, SIMD-independent `Select` trait should be renamed to `SimdSelect`. A maintainer explicitly considered the generic name reasonable. No rename is justified without broader API consensus.
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

- [x] Read the full issue discussion and distinguish the nightly visibility concern from the unproven stable-API compatibility concern.
- [x] Keep the sealing relationship while changing `PrivateMethods` from a publicly reachable child-module trait to an ordinary private trait declared directly in `masks.rs`.
- [x] Preserve the existing internal helper operations (`TRUE`, `FALSE`, `valid`, `eq`, `Unsigned`, `max_unsigned`, and `to_usize`) for generic mask code in that module.
- [x] Add three compile-fail doctests that attempt downstream access to `TRUE`, `Unsigned`, and `eq` through `MaskElement`.
- [ ] Compile the crate and run all three doctests on the pinned nightly; confirm each failure is specifically caused by privacy rather than an unrelated import or feature error.
- [ ] Run the full core_simd test suite and Clippy through CI before treating the candidate as validated.

The source-level candidate is committed on the fork's `main`, but no compiler has validated it yet. Do not describe #537 as fixed upstream or claim the regression tests pass until CI results are available.



### 11. Additional correctness and API issue triage

- [x] Recheck [#427](https://github.com/rust-lang/portable-simd/issues/427): the report concerns `simd_div` with zero divisors and `simd_rem` with minimum signed values for non-power-of-two vectors on AArch64. The issue still lacks a minimal reproducer; the only follow-up asks about the historical `repr(simd, packed)` layout and element type, and the reporter could not recall the exact type.
- [ ] Do not add a speculative workaround for #427. First construct a minimal reproducer on AArch64, establish whether the current representation still triggers the LLVM ICE, and preserve the exact compiler/LLVM versions and failing operation.
- [x] Triage [#539](https://github.com/rust-lang/portable-simd/issues/539) against the scalar float API: the current scalar surface is `algebraic_add`, `algebraic_sub`, `algebraic_mul`, `algebraic_div`, and `algebraic_rem`. The scalar documentation explicitly permits algebraic reassociation, reciprocal-based division, disregarding signed zero, and non-deterministic results; it does not promise bitwise reproducibility.
- [x] Compare the request with `StdFloat`: that trait currently exposes strict SIMD arithmetic through ordinary operators and strict `mul_add` (one-rounding FMA semantics). Relaxing those existing methods would be a breaking semantic change and is not an acceptable implementation of #539.
- [x] Check the documented `core::intrinsics::simd` surface: it exposes ordinary `simd_add`, `simd_sub`, `simd_mul`, `simd_div`, `simd_rem`, and `simd_reduce_add_unordered`, but no dedicated SIMD algebraic add/sub/mul/div/rem intrinsics. The scalar `f*_algebraic` intrinsics are not a substitute: their documented operand bound is `FloatPrimitive`, not a `repr(simd)` vector.
- [ ] Therefore, do not add a library-only `algebraic_*` SIMD API yet. The missing optimizer contract must first be provided by rustc/LLVM SIMD intrinsics (or another compiler-supported mechanism); forwarding to ordinary SIMD operators would not permit reassociation.
- [ ] `simd_reduce_add_unordered` already provides an unordered SIMD sum primitive at the intrinsic level. A public `algebraic_reduce_sum` wrapper still needs API review and confirmation of the intended floating-point contract; it should not be bundled with the missing elementwise arithmetic intrinsics.
- [ ] Treat `algebraic_reduce_sum` as a separate API design question: define reduction order, empty-vector behavior, and whether the result contract is intentionally nondeterministic before implementation.


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
- [x] Implement the narrow visibility fix in `crates/core_simd/src/masks.rs`: move `PrivateMethods` into the `masks` module with ordinary private visibility, retain it as the sealing supertrait, and explicitly allow the intentional `private_bounds` lint on `MaskElement`. No public `Mask<T, N>` signature or helper implementation changed.
- [x] Add three `compile_fail` doctests that attempt to access the helper associated constant, associated type, and method through `MaskElement` from downstream-style code.
- [ ] Compile the crate and run the three doctests plus the existing mask tests on the pinned nightly. The current environment cannot run `rustc`/`cargo`, so the source change is not yet compiler-verified.

The source-level fix is now committed, but #537 should not be considered fully verified until the compile-fail doctests and ordinary mask tests pass on the repository's pinned nightly. The key distinction is actual private visibility, not a rename to `Sealed`.
