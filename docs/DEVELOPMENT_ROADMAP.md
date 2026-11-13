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

The benchmark is now compiled and executed by the dedicated
`masked-memory-benchmark` CI job. Its semantic assertions are therefore checked on
every push to `main`. CI execution is not a controlled performance study: timings are
not persisted, and no optimized LLVM IR or assembly has yet been compared. Do not claim
an optimization from this job alone.

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
- [x] Run the regression tests on the pinned nightly and across the CI target matrix; the latest complete GitHub Actions run is green.

### 5. Review API and soundness concerns before proposing changes

- [ ] Review the stabilization/API concerns in upstream issue #364.
- [x] Inspect the large-vector alignment report in #495 and the current `Simd` representation: `#[repr(simd, packed)]`; masked pointer loads/stores request element alignment. The issue remains open, and target layout measurements are still needed.
- [x] Review the visibility/API concern in upstream issue #537, including the full maintainer discussion.
- [ ] Review the unresolved Miri floating-point test determinism discussion in issue #463.
- [ ] Verify the current code and discussion before treating any old issue as unresolved.

Issue #537 remains open upstream. The fork now contains a focused candidate fix
in `crates/core_simd/src/masks.rs`: `PrivateMethods` is declared directly in
`masks.rs` with `pub(crate)` visibility, rather than as a `pub` trait inside a
private child module. It remains inaccessible to downstream crates and is a crate-visible
supertrait of `MaskElement`, so the built-in
mask types stay sealed while the helper items are no longer publicly reachable through
the public trait. The existing generic mask operations continue to use the helper
inside the same module. Three compile-fail doctests attempt downstream access to
`TRUE`, `Unsigned`, and `eq`. The implementation has now passed the complete green CI run, including its downstream compile-fail doctests. Local compilation is still unavailable in this environment. The distinction in the
upstream discussion remains important: maintainers did not establish an automatic
stable-API break, but agreed that nightly downstream access was unintended.

### 6. Floating-point test determinism

Related upstream issue: [#463](https://github.com/rust-lang/portable-simd/issues/463).

- [x] Inspect the issue and its five comments.
- [x] Confirm the current StdFloat test suite already uses approximate comparisons
      for transcendental functions such as sin, cos, exp, exp2, ln, log2, and log10,
      while retaining exact comparisons for several other operations.
- [x] Determine whether the reported Miri failures are still reproducible on the
      pinned nightly: the current full Miri CI matrix is green. This does not prove the
      historical toolchain regression was fixed by portable-simd itself.
- [ ] Discuss a narrowly scoped Miri-only strategy with maintainers before changing
      exact comparisons, because exact tests intentionally catch implementation regressions.

The discussion in #463 does not support a blanket switch to approximate assertions:
maintainers explicitly noted the value of exact comparisons for detecting regressions.
The current pinned-nightly Miri matrix passes, so no test policy or tolerance change is
made here.

### 7. Recent issue triage

- [x] #427: open; the test suite already covers division by zero and the
      `MIN / -1` and `MIN % -1` overflow cases at the non-power-of-two lane widths.
      Added the missing exact reported remainder input: positive and negative dividends
      with `Scalar::MIN` as the divisor, including `MIN`, `MAX`, -1, and zero dividends.
      All three boundary tests are now explicitly registered through `test_lanes!`,
      so the generated 3- and 6-lane AArch64/Miri cases actually execute. This expands
      regression coverage but does not
      establish that the historical AArch64 LLVM ICE is fixed; a minimal compiler
      reproducer and the original toolchain details are still missing.
- [x] #499: closed; current StdFloat::round documentation correctly says ties round away from zero.
- [x] #527: closed; regression coverage exists in this fork.
- [x] #524: closed after the reporter retested `swizzle_dyn` successfully on newer nightly for AArch64 macOS, x86-64 Linux, and RISC-V Linux. This was a toolchain/Miri-intrinsic support resolution, not a portable-simd source fix; AVX-512 and other Miri target coverage still need separate validation.
- [x] #519: closed; verified the fork's `crates/std_float/tests/float.rs` gates the `f16` test module with `#[cfg(target_has_reliable_f16_math)]`. The issue discussion attributes the original Windows GNU failure to missing f16 math symbols/ABI support, so retain the target gate unless compiler-builtins support changes.
- [x] #537: open upstream; source and full discussion reviewed. The fork's
      crate-private `PrivateMethods` fix and all three downstream compile-fail doctests
      pass the latest complete green CI run.
- [x] #494: open; traced safe contiguous `load_select` and `store_select` to
      `simd_masked_load` / `simd_masked_store` using element alignment. Scalar-oracle
      semantic tests pass the green CI matrix. The diagnostic SSE2 benchmark now includes
      the exact `load_or_default` and all-enabled `store_select` paths plus the zip-based
      scalar load/store loops from the report, alongside masked cases. A dedicated CI
      job now compiles and runs it, checking its semantic assertions on every push.
      Controlled timings and optimized IR/assembly comparison are still required before
      changing implementation or claiming an improvement. Gather/scatter remain secondary,
      matching the issue author's stated priority.
- [x] #472: open; reviewed the full discussion. A maintainer traced the poor codegen
      to the validity `assume` emitted for full-width mask values and suggested compiler
      range metadata as a cleaner representation. The discussion also points to LLVM
      work on dropping useless assumptions. The fork's `Mask::from_simd_unchecked`
      currently emits `core::intrinsics::assume(<T as PrivateMethods>::valid(value))`;
      do not remove this safety invariant merely to improve codegen. The remaining
      work is to reproduce the original hexadecimal-validation example with the
      pinned nightly, compare optimized IR/assembly, and determine whether current
      rustc/LLVM still exhibits the issue.
- [x] #505: open upstream, but the prelude concern is already resolved by upstream PR #506. The fork's `simd` module already exports `Select` from the prelude; the remaining discussion is only whether the generic, SIMD-independent `Select` trait should be renamed to `SimdSelect`. A maintainer explicitly considered the generic name reasonable. No rename is justified without broader API consensus.
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
The regression tests have been compiled and executed successfully by CI. The #494 diagnostic benchmark is also now compiled and executed by its dedicated CI job, including its semantic assertions; controlled performance measurements and IR/assembly inspection remain outstanding. Mask ordering now has both 64-lane
high-bit regression cases and exhaustive 8-lane pair coverage. Mask bitmask conversion
also has explicit round-trip coverage at the internal width transitions and odd lane
counts. These tests have now passed through the latest green CI run; no local run is claimed. No performance
results or implementation speedups are claimed.


## Masked memory semantic regression coverage

- [x] Add scalar-oracle tests for safe masked contiguous loads and stores over every
      slice length from zero through the vector width.
- [x] Cover disabled lanes and out-of-bounds gather/scatter indices, including
      `usize::MAX` and empty slices.
- [x] Run these tests under the pinned nightly and Miri where supported; the latest complete CI run, including all four Miri shards, is green.
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
- [x] Run the test on the pinned nightly across the available architecture/feature
      matrix; the latest complete CI run is green.
- [ ] Evaluate #486's proposed unchecked/implementation-defined out-of-bounds
      variant separately as an API design and safety-contract question; do not
      change the existing zeroing semantics.

The exhaustive test strengthens confidence in existing semantics. It does not
implement the optional API proposed in #486 and does not establish a performance
improvement.


- [x] Extend contiguous masked load/store scalar-oracle tests to lane counts around
      power-of-two boundaries (1–4, 7–9, 15–17, 31–33, 63–64), checking every
      slice length and all-disabled, all-enabled, alternating, and edge-lane masks.
- [x] Run the expanded lane-width matrix on the pinned nightly and under Miri where
      supported; the latest complete CI run is green.

- [x] Extend gather/scatter scalar-oracle checks to the same lane-count boundaries,
      every slice length, alternating valid and `usize::MAX` indices, and four mask
      patterns. Scatter destinations are unique so the expected result does not rely
      on unspecified duplicate-index write ordering.

### 9. Array/slice conversion code generation

Related upstream issues: [#343](https://github.com/rust-lang/portable-simd/issues/343) and [#341](https://github.com/rust-lang/portable-simd/issues/341). The original #341 padding/overread concern predates the closure of #319. The latest #341 discussion still asks for guaranteed vector-typed LLVM loads/stores, including for non-power-of-two lane counts; #343 asks for codegen regression tests.

- [x] Trace `Simd::from_array`, `to_array`, `from_slice`, and `copy_to_slice` to the internal `load`/`store` helpers.
- [x] Confirm the helpers use `ptr::copy_nonoverlapping` rather than a direct transmute or tuple-field access. This copies exactly one `[T; N]` worth of elements and avoids reading or writing beyond the array.
- [ ] Reconcile the stale padding rationale in the source comments with the closure of #319; verify the current layout guarantee on the pinned compiler before changing those comments or the representation.
- [ ] Inspect optimized LLVM IR for representative power-of-two and non-power-of-two lane counts, and verify that the element transfer becomes vector `load`/`store` operations.
- [ ] Add compiler-version-pinned codegen regression tests once the appropriate test harness and expected IR patterns are established.
- [ ] Check more than one element type and target; avoid brittle assembly assertions when an LLVM IR-level property is sufficient.

The current copy helpers are memory-safe and deliberately copy only the array's element bytes. The remaining #341/#343 question is code generation: whether rustc/LLVM preserve vector-typed loads and stores for both power-of-two and non-power-of-two widths. The latest maintainer discussion suggests a compiler intrinsic may be needed to guarantee that property. This crate cannot safely emulate a missing compiler intrinsic by weakening the copy's bounds or alignment. No implementation rewrite is justified until IR is inspected.

### 10. `MaskElement` helper-trait visibility review

Related upstream issue: [#537](https://github.com/rust-lang/portable-simd/issues/537).

- [x] Read the full issue discussion and distinguish the nightly visibility concern from the unproven stable-API compatibility concern.
- [x] Keep the sealing relationship while declaring `PrivateMethods` directly in `masks.rs` with `pub(crate)` visibility: available to sibling implementation modules, but not nameable by downstream crates.
- [x] Preserve the existing internal helper operations (`TRUE`, `FALSE`, `valid`, `eq`, `Unsigned`, `max_unsigned`, and `to_usize`) for generic mask code in that module.
- [x] Add three compile-fail doctests that attempt downstream access to `TRUE`, `Unsigned`, and `eq` through `MaskElement`.
- [x] Compile the crate and run all three downstream compile-fail doctests on the pinned nightly; the complete CI run is green.
- [x] Run the full core_simd test suite and Clippy through CI; the complete CI run is green.

The first full GitHub Actions run exposed an internal-visibility regression: a fully private `PrivateMethods` trait prevented sibling modules from using its helpers. The final fix declares the trait `pub(crate)`, preserving access inside the crate while preventing downstream naming. The three external compile-fail doctests verify that the associated constant, type, and method cannot be reached through `MaskElement`. The latest complete CI run is green, so this fix is now validated.



### 11. Additional correctness and API issue triage

- [x] Recheck [#427](https://github.com/rust-lang/portable-simd/issues/427): the report concerns `simd_div` with zero divisors and `simd_rem` with minimum signed values for non-power-of-two vectors on AArch64. The upstream issue still lacks a minimal reproducer.
- [x] Trace existing signed-integer regression tests: `div_by_one_zero_panics`, `rem_min_may_overflow`, and the related overflow cases are generated by `test_lanes!`, which explicitly covers 3- and 6-lane vectors (non-power-of-two) on AArch64. The current green AArch64 CI run therefore exercises the reported input classes.
- [ ] Do not claim the historical LLVM ICE is fixed solely from semantic tests. A standalone compiler reproducer with the original rustc/LLVM versions is still needed before closing #427 or changing implementation code.
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
- [x] Implement the narrow visibility fix in `crates/core_simd/src/masks.rs`: declare `PrivateMethods` in the `masks` module with `pub(crate)` visibility, retain it as the sealing supertrait, and explicitly allow the intentional `private_bounds` lint on `MaskElement`. No public `Mask<T, N>` signature or helper implementation changed.
- [x] Add three `compile_fail` doctests that attempt to access the helper associated constant, associated type, and method through `MaskElement` from downstream-style code.
- [x] Compile the crate and run the three doctests plus the existing mask tests on the pinned nightly through CI; the latest complete CI run is green. Local compilation remains unavailable in this environment.

The source-level fix is committed and validated by the green CI run, including the compile-fail doctests and ordinary mask tests. The key distinction is crate-private visibility, not a rename to `Sealed`.


### 14. Running CI on the fork

The fork's `.github/workflows/ci.yml` supports `push` to `main`, pull requests,
and manual `workflow_dispatch`.

- [x] Confirm the workflow file is present on the default branch (`main`) and contains both the push trigger and `workflow_dispatch`.
- [x] Confirm development work targets `main`; the generated `gh-pages` branch is not used for development.
- [x] Enable GitHub Actions for the fork.
- [x] Run the `CI` workflow on `main`.
- [x] Inspect the complete run: all CI jobs are green.

GitHub disables workflows in forked repositories by default. To enable them, open
the fork's **Actions** tab and accept the enable-workflows prompt if it appears.
For a manual run, select **CI** in the workflow list, click **Run workflow**, choose
`main`, and confirm **Run workflow**. The workflow file must exist on the default
branch for the manual button to be available. See the
[GitHub manual workflow documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
and [fork workflow restrictions](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflows-in-forked-repositories).

The GitHub integration available for this work can inspect workflow jobs and logs
once a run ID is known, but it does not expose a workflow-dispatch action or a
general list-workflow-runs action. The commit-run helper only returns pull-request
runs, so an empty response from that helper must not be interpreted as proof that
a push-triggered run did not happen.


### CI correction — 2026-11-13

- Corrected the #427 signed remainder regression expectation: for a positive signed maximum and a minimum signed divisor, the remainder is the positive maximum (the divisor's magnitude is larger), not `-1`. This was the common failure across Miri, Windows, WebAssembly, and cross-target test jobs.
- Corrected the standalone #494 benchmark to import the nightly portable SIMD API through `std::simd`; `core` is not available as an extern-prelude path in this standalone `rustc` binary.
- These changes address the failures observed in CI run `43494021345`; a new run is required to confirm all targets.

### 15. Integer unchecked-operation API inventory

Related upstream issue: [#481](https://github.com/rust-lang/portable-simd/issues/481).

- [x] Verify the scalar stabilization referenced by the issue: Rust PR [#122520](https://github.com/rust-lang/rust/pull/122520) stabilized exactly `unchecked_add`, `unchecked_sub`, and `unchecked_mul`.
- [x] Distinguish these from `unchecked_neg` and `unchecked_shr`, which the stabilization PR explicitly did not include.
- [x] Check the current SIMD integer traits: neither `SimdInt` nor `SimdUint` exposes corresponding unchecked arithmetic methods.
- [ ] Discuss whether SIMD unchecked arithmetic belongs in the portable API and what compiler intrinsic/lowering should express the no-overflow precondition.
- [ ] Add methods and safety documentation only after the compiler contract and API design are agreed; add boundary tests that never violate the unsafe preconditions.

This is an API-inventory/design gap, not a demonstrated implementation bug. The scalar stabilization does not automatically imply that a SIMD wrapper can safely be implemented by calling the existing wrapping vector operators: the compiler must be able to rely on the documented no-overflow precondition. No unsafe SIMD API is added as part of this triage.
