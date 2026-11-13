# portable-simd development log

This is the change ledger for the `Likounsee/portable-simd` fork. It replaces a forward-looking roadmap: entries record what was changed, where it was changed, why, and what evidence exists. Line numbers refer to the files at this log's commit; later edits may shift them.

## Working rules

- Work only on this fork's `main`; never create a work branch or push to `rust-lang/portable-simd`.
- Keep correctness tests, performance measurements, and API proposals separate.
- Do not claim a compiler/backend issue is fixed by adding library tests alone.
- Do not claim performance improvements without reproducible timings and optimized IR/assembly.
- Record CI by run and job result. A cancelled job is not a passing job.

## Implemented changes

### 2026-11-12 — mask ordering and bitmask conversion tests

- Commits: `9f691991`, `5bd1dea8`, `12b9c14a`, `583beade`, `21efb0ac`, `5bba0c2b`.
- Added boolean-ordering regression coverage for SIMD masks and exhaustive comparison of all 65,536 pairs of 8-lane mask patterns.
- Added 64-lane high-bit/mixed-pattern cases and bitmask conversion tests around internal width transitions and odd lane counts. Expected lane values are constructed independently rather than relying only on a round trip.
- Files: `crates/core_simd/tests/mask_partial_ord.rs:1-86` (64-lane and exhaustive 8-lane ordering; exhaustive test begins at line 66); `crates/core_simd/tests/masks.rs:142-218` (round trips and lane-boundary conversion coverage).

### 2026-11-12 — masked memory correctness coverage

- Commits: `85a7e0f2`, `106ed04d`, `700c14ea`, `ca17dd7c` (plus formatting/documentation follow-ups).
- Added scalar-oracle tests for masked contiguous load/store, gather, and scatter. Cases cover empty and short slices, disabled lanes, out-of-bounds indices, `usize::MAX`, and unique scatter destinations.
- Expanded gather/scatter checks across all four-lane index tuples and enable masks, then across lane counts around power-of-two boundaries.
- File: `crates/core_simd/tests/masked_memory.rs:11-321`; primary tests begin at lines 11, 44, 72, 83, 94, 103, 144, and 304.

### 2026-11-12 — dynamic byte swizzle exhaustive indices

- Commit: `70c84848` (with tracking commit `b7f5facb`).
- Added a scalar-oracle test that visits every possible u8 index at representative lane widths around implementation boundaries.
- File: `crates/core_simd/tests/swizzle_dyn.rs:75-111`; helper begins at line 75 and width matrix at line 94.
- This verifies existing zeroing semantics; it does not add the optional unchecked/out-of-range API proposed in upstream issue #486.

### 2026-11-12 to 2026-11-13 — masked load/store diagnostic benchmark

- Commits: `e4247b00`, `1f6c6742`, `b7742b3d`, `87b450aa`.
- Added and expanded a diagnostic comparison for masked SIMD load/store versus scalar loops, including short lengths, mask patterns, `load_or_default`, and all-enabled `store_select`.
- Added CI compilation and semantic assertions for the benchmark. Corrected its standalone import from `core::simd` to `std::simd`.
- File: `benchmarks/masked_load_store.rs:1-184`; API wrappers are around lines 32-64, semantic checks around 100-110, and measurement loops around 115-159.
- The CI benchmark job compiles/runs semantic checks; it does not persist controlled timing results or inspect optimized IR/assembly. No speedup is claimed.

### 2026-11-13 — seal mask helper implementation without hiding it from sibling modules

- Commit: `e4632f7e` (followed by compile-fail tests and documentation corrections).
- Kept `PrivateMethods` crate-visible so sibling modules can use its helpers while downstream crates cannot name the helper trait through the public `MaskElement` interface.
- Added downstream compile-fail doctests for attempts to access `TRUE`, `Unsigned`, and `eq`.
- File: `crates/core_simd/src/masks.rs:37` (trait visibility) and line 223 (validity assumption); compile-fail coverage is in the module documentation/tests associated with `MaskElement`.
- This addresses the visibility concern in upstream #537; it does not establish that every stable-API compatibility concern in that discussion is resolved.

### 2026-11-13 — floating-point rounding regression tests

- Commit: `65d5231a`.
- Added halfway-value coverage for SIMD float rounding behavior.
- File: `crates/std_float/tests/float.rs`; the existing reliable-f16-math target gate is at line 114.
- Existing exact comparisons were not blanket-replaced with approximate comparisons; upstream #463 discussion does not justify weakening unrelated assertions.

### 2026-11-13 — signed division/remainder regression coverage (#427)

- Commits: `002df4ec`, `24ac2466`, `734c03f3`, `87b450aa`.
- Added tests for division/remainder overflow boundaries and the specifically reported case where the signed minimum is the remainder divisor (`rhs`).
- Registered the tests inside `test_helpers::test_lanes!`, which generates non-power-of-two widths 3 and 6 as well as the other configured widths.
- Corrected the scalar expected result for `Scalar::MAX % Scalar::MIN` from `-1` to `Scalar::MAX`.
- File: `crates/core_simd/tests/ops_macros.rs:377-429` (registered signed boundary cases; the mixed-lane addition below extends this block).
- These tests exercise the reported value classes but do not prove the historical AArch64 LLVM ICE is fixed. The original compiler versions and a minimal compiler-only reproducer are still needed.

### 2026-11-13 — unchecked integer SIMD API inventory (#481)

- Commit: `871ee034`.
- Verified that the referenced scalar stabilization covers `unchecked_add`, `unchecked_sub`, and `unchecked_mul`, not `unchecked_neg` or `unchecked_shr`.
- Confirmed the current SIMD integer traits do not expose those unchecked methods.
- Files inspected: `crates/core_simd/src/simd/num/int.rs:6-231` and `crates/core_simd/src/simd/num/uint.rs:4-117`.
- No unsafe SIMD method was added: API and compiler-contract review is required before an implementation can promise the no-overflow precondition.

### 2026-11-13 — mixed-lane signed remainder regression

- Included in this commit: mixed-lane coverage is added directly to the signed-integer `test_lanes!` block.
- Adds `rem_by_min_divisor_mixed_lanes`, mixing `Scalar::MIN`, ordinary positive/negative divisors, and signed boundary dividends in the same vector. Expected values are calculated lane by lane with scalar `wrapping_rem`.
- File: `crates/core_simd/tests/ops_macros.rs:431-458`, inside the existing `test_lanes!` expansion; it therefore generates the configured lane-width tests.
- This is additional regression coverage, not a claim that the upstream compiler ICE is fixed.

## Investigations and conclusions (no source fix claimed)

- **#494 — masked load/store code generation:** safe contiguous APIs were traced to masked pointer intrinsics using element alignment. The diagnostic benchmark is present, but controlled target-specific timings and optimized LLVM IR/assembly are still missing. Do not change implementation based on CI benchmark execution alone.
- **#472 — mask code generation:** `Mask::from_simd_unchecked` retains the validity `assume` at `crates/core_simd/src/masks.rs:223`. Removing a soundness invariant to chase codegen is not acceptable. Reproduce the original hexadecimal-validation example and inspect current IR/assembly first.
- **#341/#343 — array/slice conversion code generation:** current copy helpers use bounded element copies; the unresolved question is whether the compiler emits vector-typed loads/stores for non-power-of-two widths. Inspect IR before changing memory operations or alignment.
- **#495 — large-vector alignment:** likely involves compiler `repr(simd, packed)` layout; no library-side workaround is justified without target layout evidence.
- **#486 — dynamic swizzle:** exhaustive tests cover current semantics. An unchecked/out-of-range variant remains an API and safety-contract question.
- **#539 — algebraic float operations:** scalar algebraic intrinsics do not automatically provide SIMD semantics. A SIMD intrinsic/API contract is needed; existing strict operations must not silently change.
- **#505 — `Select` naming:** prelude export is already addressed upstream; renaming the generic trait has no established maintainer consensus.
- **#519 — f16 test gate:** existing `#[cfg(target_has_reliable_f16_math)]` gate was verified in `crates/std_float/tests/float.rs:114`; no change required.
- **#524 — Miri swizzle support:** reporter retested the reproducer successfully on newer nightly. This was a toolchain/Miri support resolution, not a source fix in this fork.
- **#499 and #527:** closed upstream; relevant rounding documentation and mask-ordering regression coverage were checked.

## CI and validation record

- The benchmark has a dedicated CI job that compiles and executes its semantic assertions.
- The signed remainder expected-value correction addressed failures observed in earlier CI jobs.
- On the log-preparation snapshot, run [43509125858](https://github.com/Likounsee/portable-simd/actions/runs/43509125858) is still in progress on attempt 2: most jobs succeeded, x86_64 macOS is running, and aarch64 macOS is cancelled. Therefore the complete matrix is **not recorded as green**.
- No local Rust compilation is claimed; a Rust toolchain is unavailable in the working environment.

## Remaining evidence needed

1. Finish or rerun both macOS jobs and record their actual conclusions.
2. Validate the mixed-lane test in the complete CI matrix after it is integrated.
3. Capture controlled #494 timings and optimized IR/assembly on the reported target before proposing a codegen change.
4. Reproduce #472 and #341/#343 with pinned compiler versions and retain the IR/assembly artifacts.
5. Resolve API/compiler contracts with maintainers before adding unchecked integer, algebraic float, or unchecked swizzle APIs.
