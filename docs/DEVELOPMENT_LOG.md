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
- Addressed the nondeterministic transcendental-comparison concern in upstream #463: `sin`, `cos`, `exp`, `exp2`, `ln`, `log2`, and `log10` use the existing ULP-based `test_unary_elementwise_approx` helper; `log` uses the binary approximate helper with a wider tolerance because it performs two logarithms and a division. Exact comparisons remain for operations whose contract is deterministic (rounding, truncation, square root, and fused multiply-add); unrelated assertions are not weakened.

### 2026-11-13 — signed division/remainder regression coverage (#427)

- Commits: `002df4ec`, `24ac2466`, `734c03f3`, `87b450aa`.
- Added tests for division/remainder overflow boundaries and the specifically reported case where the signed minimum is the remainder divisor (`rhs`).
- Registered the tests inside `test_helpers::test_lanes!`, which generates non-power-of-two widths 3 and 6 as well as the other configured widths.
- Corrected the scalar expected result for `Scalar::MAX % Scalar::MIN` from `-1` to `Scalar::MAX`.
- File: `crates/core_simd/tests/ops_macros.rs:377-429` (registered signed boundary cases; the mixed-lane addition below extends this block).
- Follow-up on the upstream discussion: maintainers recalled that the ICE occurred with `#[repr(simd, packed)]`; the original reporter thought it may have been `i8` but was not certain. The issue still has no compiler-only reproducer or pinned affected rustc/LLVM version.
- These tests exercise the reported value classes, including the likely narrow integer case and non-power-of-two widths, but do not prove the historical AArch64 LLVM ICE is fixed. A minimal compiler-only reproducer is still needed.

### 2026-11-13 — unchecked integer SIMD API inventory (#481)

- Commit: `871ee034`.
- Verified that the referenced scalar stabilization covers `unchecked_add`, `unchecked_sub`, and `unchecked_mul`, not `unchecked_neg` or `unchecked_shr`.
- Confirmed the current SIMD integer traits do not expose those unchecked methods.
- Files inspected: `crates/core_simd/src/simd/num/int.rs:6-231` and `crates/core_simd/src/simd/num/uint.rs:4-117`.
- No unsafe SIMD method was added: API and compiler-contract review is required before an implementation can promise the no-overflow precondition.

### 2026-11-13 — mixed-lane signed remainder regression

- Included in this commit: mixed-lane coverage is added directly to the signed-integer `test_lanes!` block.
- Adds `rem_by_min_divisor_mixed_lanes`, mixing `Scalar::MIN`, ordinary positive/negative divisors, and signed boundary dividends in the same vector. The lane pattern deliberately includes `Scalar::MAX % Scalar::MIN` and `-1 % Scalar::MIN` alongside ordinary divisors. Expected values are calculated lane by lane with scalar `wrapping_rem`.
- File: `crates/core_simd/tests/ops_macros.rs:431-458`, inside the existing `test_lanes!` expansion; it therefore generates the configured lane-width tests.
- This is additional regression coverage, not a claim that the upstream compiler ICE is fixed.

## Investigations and conclusions (no source fix claimed)

- **#494 — masked load/store code generation:** safe contiguous APIs were traced to masked pointer intrinsics using element alignment. The diagnostic benchmark is present, but controlled target-specific timings and optimized LLVM IR/assembly are still missing. Do not change implementation based on CI benchmark execution alone.
- **#472 — mask code generation:** `Mask::from_simd_unchecked` retains the validity `assume` at `crates/core_simd/src/masks.rs:223`. The upstream discussion also explored a compiler-level bit-sized representation (`Bits<N>`) as a longer-term direction; it is not a library-side patch available to this fork. Removing a soundness invariant to chase codegen is not acceptable. Reproduce the original hexadecimal-validation example and inspect current IR/assembly first.
- **#341/#343 — array/slice conversion code generation:** current copy helpers use bounded element copies. Upstream discussion confirms that ordinary array assignment can produce vector-typed LLVM loads/stores for power-of-two widths, while non-power-of-two widths may intentionally fall back to `memcpy`; maintainers still want an intrinsic or equivalent guarantee that SIMD-facing operations emit vector-typed IR. Inspect current IR for both width classes before changing memory operations or alignment.
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
- Run [43509125858](https://github.com/Likounsee/portable-simd/actions/runs/43509125858), attempt 3, completed successfully across the full matrix. Both macOS x86_64 and ARM64 passed debug, release, and documentation generation; Windows, Linux, WASM, Miri, Clippy, and the masked-memory benchmark job also passed.
- Run [43519315714](https://github.com/Likounsee/portable-simd/actions/runs/43519315714) completed all 30 jobs successfully for `55e04c0`.
- No local Rust compilation is claimed; a Rust toolchain is unavailable in the working environment.

## Remaining evidence needed

1. Confirm the push-triggered CI run for commit `55e04c0` validates the mixed-lane regression test across the full matrix.
2. Capture controlled #494 timings and optimized IR/assembly on the reported target before proposing a codegen change.
3. Reproduce #472 and #341/#343 with pinned compiler versions and retain the IR/assembly artifacts.
4. Resolve API/compiler contracts with maintainers before adding unchecked integer, algebraic float, or unchecked swizzle APIs.


### 2026-11-13 — follow-up review of compiler-owned issue reports

- Re-read the upstream maintainer comments for #427, #341, and #472 after the regression-test review.
- #427: the maintainer recollection narrows the historical trigger to the packed SIMD representation, possibly `i8`; the reporter could not confirm the exact element type. The current tests cover the value classes and lane widths but cannot substitute for the missing compiler reproducer.
- #341/#343: the discussion distinguishes power-of-two array copies, which can become LLVM vector loads/stores, from non-power-of-two copies that may lower to `memcpy`. The requested guarantee is specifically about vector-typed LLVM IR for SIMD-facing code. No unsafe overread or alignment workaround is justified.
- #472: the discussion points toward a compiler representation improvement for masks/bits. The fork must preserve the validity invariant and should not weaken it to force code generation.


### 2026-11-13 — issue #463 test audit and CI completion

- Rechecked `crates/std_float/tests/float.rs` and `crates/test_helpers/src/lib.rs` against upstream #463.
- Confirmed the transcendental functions implicated by platform-dependent libm results already use ULP-based approximate comparisons, while deterministic rounding operations retain exact assertions. The binary `log` test uses a 32-ULP allowance for its two-logarithm-plus-division implementation.
- The full pre-integration CI run completed green on attempt 3. The mixed-lane regression was added in commit `55e04c0`; its push-triggered CI must finish green before treating that test as validated.


### 2026-11-13 — fix lane-generic inference in mask reduction tests

- Commit: pending CI.
- The first full CI run for the new `first_set` / `last_set` lane-boundary test found that `core::array::from_fn` could not infer the const array length through the nested generic helper.
- Added explicit `check::<LANES>` arguments for every test pattern so the helper's const parameter is unambiguous on all targets.
- This is a test-compilation correction; it does not change the mask implementation.
- CI run [43521947560](https://github.com/Likounsee/portable-simd/actions/runs/43521947560) exposed the error on AArch64 and SPARC64 Clippy jobs. The corrected commit must pass the complete matrix before this coverage is considered validated.


### 2026-11-13 — clarify unaligned SIMD pointer safety (#341)

- Updated the `Simd<T, N>` unsafe-pointer documentation in `crates/core_simd/src/vector.rs`.
- Explicitly distinguishes alignment from access size: `read_unaligned` and `write_unaligned` still access `size_of::<Simd<T, N>>()` bytes.
- Warns against casting an array pointer to a SIMD pointer when non-power-of-two lane counts may give the SIMD type trailing padding, and directs users to the safe array/slice conversion APIs.
- This is a safety-documentation correction only. It does not change the implementation or claim to resolve the compiler intrinsic/codegen requests in #341/#343.

## 2026-11-14 — correct last-set implementation comment

- Corrected the internal `Mask::last_set` explanation: it selects the maximum matching lane index, not the first one.
- Documentation/comment-only cleanup; no behavior or generated code changed.


### 2026-11-14 — streamline CI workflows

- Added ref-scoped concurrency cancellation so a newer push or update supersedes an obsolete run instead of allowing multiple stale runs to consume runners.
- Added Cargo dependency/build caches to the compile-heavy CI jobs, including cross-target jobs (and the installed Cross binary).
- Removed repeated documentation builds from every x86 feature/OS matrix entry and both macOS entries. A dedicated documentation job now builds on pull requests and manual CI runs with `RUSTDOCFLAGS=-Dwarnings`; main pushes use the documentation deployment workflow for that same warning-checked build, avoiding a duplicate.
- Preserved the complete original x86 target/feature matrix: all 31 combinations (four targets, eight feature settings, excluding only the original i686 `native` cross-target case) still run in debug and release. No test target or feature combination was removed.
- Removed the repeated documentation build from every x86 and macOS matrix entry. Pull requests/manual CI build docs once in the dedicated docs job; main pushes use the separate documentation deployment workflow. Cargo caches and stale-run cancellation improve repeat-run efficiency without reducing test coverage.
- This is a workflow-efficiency change; its cache hit rate and elapsed-time improvement must be measured from the next completed Actions runs.


### 2026-11-14 — correct SIMD layout safety documentation

- Corrected the earlier raw-pointer documentation: the crate's `layout.rs` regression tests verify that every supported `Simd<T, N>` has the same size as its corresponding `[T; N]`, including non-power-of-two lane counts. The previous warning that these vectors may be larger because of trailing padding was inaccurate for this implementation.
- Updated the `as_array` / `as_mut_array` safety comments and the `load` / `store` safety comments, and removed obsolete `from_array` / `to_array` FIXME text that attributed the copies to non-power-of-two padding.
- Kept the important distinction that matching sizes do not make references interchangeable and do not waive pointer validity, initialization, aliasing, or provenance requirements.
- Documentation and safety-comment correction only; no runtime behavior changed.


### 2026-11-14 — cover f16 in SIMD layout regression tests

- Added `f16` to the existing no-padding layout test matrix, which already checks every supported lane count for the other SIMD element types.
- This closes a small coverage gap in the assertion used to support the array/SIMD size documentation: `f16` is also a `SimdElement` and should be checked explicitly.
- Test-only change; no implementation behavior changed.

### 2026-11-14 — fix cached Cross installation in cross-target CI

- The full-matrix run for commit `0a36778` exposed five cross-target job failures before any tests ran.
- All five failures had the same cause: the cache restored `~/.cargo/bin/cross` and `cross-util`, then `cargo install` failed because those binaries already existed.
- Updated the installation step to reuse the cached Cross binary when present and install the pinned revision only on a cache miss.
- No target, feature combination, debug/release run, or test was removed. This repairs CI setup and avoids recompiling Cross on cache hits.
- Evidence: run [43615304148](https://github.com/Likounsee/portable-simd/actions/runs/43615304148), failed step `Install Cross` in the AArch64, ARMv7, RISC-V, and PowerPC64LE jobs (all reported the same pre-test `binary already exists` error).
- Validation pending: the next full CI run must pass the affected cross-target jobs.
