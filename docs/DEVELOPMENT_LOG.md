### 2026-11-16 — simplify standalone benchmark linking

- Corrected the CI diagnostic build to pass the generated core_simd rlib once to rustc. The previous duplicate --extern entries for the same crate name could shadow the linkable artifact with metadata-only output.
- Updated both benchmark reproduction comments to use the rlib only.
- Clarified that the AVX2 unchecked swizzle fast path currently covers 32-byte vectors; wider x86 vectors intentionally retain the regular correctness-preserving fallback.

### 2026-11-16 — tighten unchecked swizzle split semantics

- Kept direct unchecked paths for architectures whose single instruction handles the full vector width (AArch64 TBL, ARMv7 VTBL, WebAssembly 16-byte swizzle, x86 SSSE3 16-byte and AVX2 32-byte).
- Removed an attempted unchecked split for wider vectors: a valid global index can be out of range for one half, and the API permits arbitrary results for those per-half indices, so OR-combining the halves could corrupt a valid result. WebAssembly's 32-byte split now uses the regular split helper, whose native 16-byte swizzles already provide the desired zeroing without x86-style index fixup. Wider x86 widths retain the correct fallback until a direct full-width implementation is available.

### 2026-11-16 — extend unchecked swizzle fast paths

- Extended `swizzle_dyn_unchecked` beyond the SSSE3 16-byte case: AArch64 NEON uses TBL directly, ARMv7 NEON uses VTBL directly, and WebAssembly SIMD128 uses its native swizzle instruction. These instructions naturally zero out-of-range high-bit indices, so no regular-path index fixup is needed.
- Added an AVX2 32-byte path using raw VPSHUFB controls. Since the API contract guarantees the high bit on out-of-range indices, VPSHUFB itself supplies zeroing and the regular-path saturating index adjustment is omitted. A 64-byte path composes two unchecked 32-byte shuffles.
- Existing exhaustive high-bit tests cover all byte values 0x80..=0xff across the relevant widths; CI's target-feature matrix includes SSSE3 and AVX2 configurations.
- Full validation is pending the queued CI run.

### 2026-11-16 — specialize algebraic SIMD reductions by float format

- Corrected the unordered reduction implementation after reviewing its history: the compiler intrinsic does not support f16 on all relevant backends.
- f32 and f64 now use the unordered SIMD reduction intrinsic; f16 retains the scalar algebraic fold. This avoids reintroducing the previously reverted f16 backend failure.
- Expanded the #475 diagnostic benchmark to compare checked slice gather, unchecked slice gather, and direct pointer gather while asserting identical results.
- Wired the gather benchmark into the quality-check CI job: rustfmt, optimized compilation, execution, assembly and LLVM IR emission, and artifact upload. This makes generated code inspectable under a fixed x86-64/SSE2 target.
- No performance improvement is claimed until CI artifacts and measurements are available.

### 2026-11-16 — use unordered SIMD reduction for algebraic float sums

- Replaced the scalar lane-by-lane fold in `SimdFloat::algebraic_reduce_sum` with the compiler's `simd_reduce_add_unordered` intrinsic.
- This matches the relaxed algebraic contract: the reduction may reassociate lanes and use a target-specific SIMD reduction instead of forcing a left-to-right scalar fold.
- Expanded issue #539 tests to cover f16/f32/f64 arithmetic, relaxed multiply-add, and exact representable reductions. Tests no longer require bit-identical results for a particular reduction order.
- Validation remains pending the CI run for this consolidated change.

### 2026-11-15 — rustfmt follow-up for issue #366

- The next CI run caught formatting differences in the scalar reference expression and long test arrays in `issue_366_maddubs.rs`.
- Applied the exact layout requested by rustfmt. This is a formatting-only correction; semantic and compilation validation remain in the active CI run.

### 2026-11-15 — fix issue #366 example compilation

- The first CI compile found two example-target errors: the file had no `main` function, and the destination type for the SIMD `cast` calls was not inferable from the product annotation.
- Added explicit `cast::<i16>()` types and a small runnable `main` demonstration. The scalar-reference tests remain unchanged.
- This corrects the example build only; it does not change the portable operation's semantics or claim that LLVM selects `pmaddubsw`. The follow-up CI run must validate the fix.

### 2026-11-15 — use full-width SIMD memory operations for all-enabled slices

- The CI assembly/LLVM IR for the all-enabled path showed that `load_or_default` and `store_select` always used a runtime `memcpy`, even when the slice contained at least a complete vector.
- Added a full-width fast path to `load_or`: when at least `N` elements are available, use the existing unaligned-safe SIMD `load` helper. Short slices retain the bounded prefix copy and fallback values.
- Added the matching full-width fast path to all-enabled `store_select`, using the existing unaligned-safe SIMD `store` helper. Short slices retain the prefix copy; masked stores are unchanged.
- Added a regression test proving a full-width store writes exactly the vector lanes and preserves any destination tail.
- This is a targeted response to the measured SSE2 codegen issue in upstream #494. CI assembly/IR and the full test matrix must validate the change before calling it complete.

### 2026-11-15 — pass both Cargo metadata and rlib artifacts to standalone rustc

- Passing only the `.rmeta` file failed because rustc also requires the crate's `.rlib` form; passing only the `.rlib` failed because Cargo emits it as a metadata-only stub for this crate.
- Updated CI and the local reproduction instructions to select both files from the same Cargo `compiler-artifact` record and pass both repeated `--extern core_simd=...` arguments, matching the form Cargo itself uses for this crate's integration tests.
- The benchmark still compiles against the fork and uses the public `as_array` accessor for its scalar reference. No SIMD implementation or test matrix changes.
- Validation is pending on the next quality-check run.

### 2026-11-15 — fix fork API usage in masked-memory codegen benchmark

- The first CI compile against the fork exposed one benchmark-only API mismatch: the scalar reference used `value[lane]`, but `core_simd::Simd` does not implement indexing. Replaced it with the supported `value.as_array()[lane]` accessor.
- Corrected the artifact-discovery error text to name the expected `.rmeta` file.
- Changed artifact upload's missing-file policy to warn: a failed compile remains the job's real failure, without a second redundant upload failure when no assembly/IR was produced.
- No library implementation or test matrix was changed. Validation is pending on the next full CI run.

### 2026-11-15 — align masked-memory benchmark reproduction instructions with CI

- Updated the benchmark's documented build/link command to select the `.rmeta` artifact reported by Cargo, matching the working CI diagnostic.
- The previous `.rlib` search could select the metadata-only stub produced for this crate build, so local reproduction did not match the CI invocation.
- This is a reproducibility/documentation correction; it does not change library behavior or test coverage.

### 2026-11-15 — remove extra blank line in masked-memory tests

- The follow-up rustfmt run identified one remaining extra blank line before the new contiguous-copy test.
- Removed the blank line; no code or test behavior changed. The next CI run is the validation for the full source optimization and native-cache fix.

### 2026-11-15 — apply rustfmt correction to contiguous-copy regression test

- The first CI run for the all-enabled memory fast paths reported only a formatting difference in the new expected-value closures.
- Applied rustfmt's exact single-line closure layout. No implementation or test semantics changed.
- The same run also exercised the native Linux job with the new dependency-only cache; full validation remains pending on the corrected commit.

### 2026-11-15 — use the full Cargo metadata artifact for inline codegen

- The selected Cargo `.rlib` is intentionally a metadata stub for this crate build; rustc reported that the matching `.rmeta` contains the full metadata needed by the standalone compilation.
- The diagnostic now selects the `.rmeta` entry from Cargo's `compiler-artifact` record. The benchmark exercises inline/generic `core_simd` methods, so the compiler can instantiate their MIR in the benchmark crate.
- Validation of this linkage is pending; if non-inline symbols are required, the benchmark will be moved to a Cargo example target instead.

### 2026-11-15 — select the actual Cargo compiler artifact

- The Cargo target-directory search selected an intermediate build-script rlib stub, which rustc correctly rejected because it contained metadata only.
- The CI step now captures Cargo's JSON build messages and selects the `.rlib` from the `compiler-artifact` record for the `core_simd` library target. This avoids confusing build-script outputs with the final crate artifact.
- The standalone compiler remains on edition 2024 and links the selected fork artifact through `--extern`.

### 2026-11-15 — compile the fork-linked benchmark with the workspace edition

- The fork rlib was found and passed to rustc, but the standalone compiler invocation defaulted to Rust 2015. That edition requires an explicit `extern crate` declaration and rejected the benchmark's Rust 2018+ `use core_simd::...` path.
- Added `--edition=2024` to all benchmark compiler commands and reproduction instructions. No library code changed.

### 2026-11-15 — locate the fork library through Cargo metadata

- The fork-linked benchmark passed both formatting checks, but its build step assumed Cargo wrote artifacts under the repository-local `target/debug/deps` directory.
- CI showed the active Cargo target directory differs from that assumption. The workflow now reads `target_directory` from `cargo metadata` and searches there for the built `core_simd` rlib.
- This changes only artifact discovery; the benchmark still links the exact fork library and the full test matrix remains unchanged.

### 2026-11-15 — format fork-linked masked-memory benchmark

- The new standalone `rustfmt --check` step caught six formatting differences in the expanded benchmark, including long assertions, calls, and mask literals.
- Applied rustfmt's suggested layout. No benchmark behavior or library code changed.
- The fork-linked compile and codegen artifact steps will run once formatting passes.

### 2026-11-15 — link masked-memory diagnostics to the fork implementation

- Audited the first codegen artifact and found that the standalone benchmark imported `std::simd`, so rustc measured the toolchain's sysroot implementation rather than `Likounsee/portable-simd`.
- The quality-check job now builds the workspace's `core_simd` crate with `as_crate` and passes that exact rlib to the diagnostic compiler through `--extern`; the benchmark imports `core_simd::simd`.
- Added dedicated all-enabled load/store wrappers so `#[inline(never)]` benchmark boundaries do not hide the constant all-true mask from the optimizer. Sparse-mask cases remain separately measured.
- The next artifact will be the first fork-specific SSE2 IR/assembly evidence. The all-enabled fast paths remain provisional until that artifact confirms their lowering and performance.

### 2026-11-15 — use contiguous copies for all-enabled masked memory operations

- The first standalone SSE2 artifact was later found to compile against the pinned toolchain's `std::simd`, not this fork's `core_simd` crate. It therefore did not establish fork-specific code generation; the benchmark and CI link are being corrected before making a performance claim.
- Changed `load_or` to initialize from its fallback vector and copy only the available slice prefix. `load_or_default` inherits this path.
- Added an all-enabled fast path to `load_select` and `store_select`; it uses the same bounded prefix-copy strategy. Sparse masks still use the existing masked load/store intrinsics.
- Added regression coverage for empty, short, exact-width, and longer input slices, including fallback preservation. Existing masked-memory tests continue to cover mixed masks and short destinations.
- This is a candidate fast path with correctness coverage. Its performance benefit and generated code remain unverified until the fork-linked diagnostic artifact is inspected; sparse-mask lowering is unchanged. Full CI validation is pending.

### 2026-11-15 — prevent native CPU artifacts from crossing runner hosts

- The native x86-64 job failed in the generated `cast` test binary with `SIGILL` after restoring its Cargo `target` cache.
- Compared the successful and failing runner diagnostics: the earlier cache producer advertised AVX-512 features (`avx512f`, `avx512bw`, `avx512dq`, and others), while the failing runner exposed AVX2 but no AVX-512. The native cache key was identical because it described only OS, target, feature label, and lockfile.
- The failure is therefore explained by reusing machine-code artifacts compiled under a different `-Ctarget-cpu=native` feature set; it is not evidence of a cast implementation defect.
- Native x86 matrix entries now cache only Cargo registry/Git dependencies. Their `target` build artifacts are compiled fresh on each runner. Non-native matrix entries retain the existing full build cache.
- No target, feature combination, debug/release run, or test was removed. Validation is pending on the next full CI run.
 
### 2026-11-15 — retain optimized codegen evidence for masked memory operations

- Extended the existing masked-memory diagnostic CI job to emit optimized x86-64/SSE2 assembly and LLVM IR using the repository-pinned nightly toolchain.
- Uploads both files as a commit-specific Actions artifact for 14 days, making the exact generated code available for review after each CI run.
- This closes the evidence gap in upstream #494: future changes can compare the actual `load_select` / `load_or_default` and `store_select` lowering against the scalar reference rather than relying on timings or source-level assumptions.
- The existing semantic benchmark still runs unchanged. No library implementation, unsafe code, target matrix, or test coverage was altered.
- Validation: the new CI run must confirm both compiler emissions and artifact upload succeed; the first artifact will be inspected before proposing any codegen implementation change.

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

### 2026-11-14 — consolidate CI quality and documentation workflows

- Merged the standalone formatting and masked-memory benchmark jobs into one Ubuntu job. Both checks still run as separate steps with their original commands and failure behavior.
- Moved the main-branch GitHub Pages deployment into the existing warning-checked documentation job and removed the separate documentation workflow. Documentation still builds with nightly Rust and `RUSTDOCFLAGS=-Dwarnings`; deployment runs only on pushes to `main` and only after a successful build.
- The workflow now launches one fewer job and one fewer Actions workflow per main push. The documentation build is not duplicated, and the quality checks still execute on every CI event.
- No test suite, target, target-feature combination, debug/release run, Clippy target, Miri shard, WASM configuration, or cross-target job was removed.
- This change consolidates orchestration; the next completed run should be used to compare elapsed time and verify the deployment step.

### 2026-11-14 — scope documentation deployment permissions to trusted pushes

- Follow-up hardening for the consolidated workflow: kept the pull-request/manual documentation check read-only and moved the GitHub Pages publishing step into a separate job that runs only for pushes to `main`.
- Only the deployment job receives `contents: write`; untrusted pull-request builds do not receive write permissions.
- The deployment job performs the same nightly documentation build with warnings denied before publishing. No documentation validation or deployment behavior was dropped.

### 2026-11-14 — avoid duplicate documentation builds after workflow consolidation

- Restricted the read-only documentation-check job to pull requests and manual runs.
- Main pushes now use only the dedicated `deploy-docs` job, which builds with warnings denied and deploys after that build succeeds. This avoids running the documentation build twice on pushes while keeping write permissions isolated to the trusted deployment job.


### 2026-11-14 — share the Cross host-tool cache across target jobs

- Replaced the per-target cache of `~/.cargo/bin` with one cache for the two Cross executables (`cross` and `cross-util`), shared by all cross-test matrix entries.
- Cross is a host-side tool: the same Ubuntu executable is used to test every emulated target. The previous key included the target and feature, causing up to eight independent Cross builds and cache copies.
- The shared cache key includes the pinned Cross Git revision (`CROSS_REV`), so changing the tool revision naturally invalidates the cached binaries. Target-specific Cargo registry, Git, and build-artifact caches remain separate.
- All seven default cross targets and the additional PowerPC64LE `+vsx` configuration still run both debug and release tests. No tests or target configurations were removed.
- Validation: the workflow syntax and resulting Actions run must be checked after this commit; cache reuse and elapsed-time gains require at least one subsequent warm-cache run to measure.


### 2026-11-15 — prepare Cross once before target fan-out

- Added a small `cross-tool` setup job that restores or installs the pinned Cross host executable once, then verifies it.
- Made the eight existing Cross test configurations depend on that setup job and restore the exact same revision-keyed executable cache. A cold cache can no longer trigger eight simultaneous builds of Cross.
- Kept target-specific Cargo dependency/build caches in each matrix job. All seven emulated targets plus the PowerPC64LE `+vsx` variant still execute both debug and release tests.
- This adds one setup job but does not remove or merge any test job, target, feature, or test mode. The tradeoff is one serialized Cross setup before the emulated test fan-out, avoiding duplicated CPU and network work on cache misses.
- Validation: the new workflow run must confirm the cache handoff and all downstream jobs; no local Rust toolchain is available for a local build.


### 2026-11-15 — fix Cross verification workspace context

- The first run of the single-install Cross setup exposed a verification-only failure: `cross --version` still probes Cargo metadata, but the setup job had not checked out the repository and therefore had no workspace manifest.
- Added a read-only checkout to the setup job before verifying Cross. This gives Cross the same valid workspace context it has in the downstream test jobs.
- The failure occurred before any emulated tests started; no test target or coverage was affected. The pinned Cross revision and shared executable cache remain unchanged.
- Validation: the next CI run must confirm that setup verification succeeds, the cache is restored by all eight Cross matrix entries, and their debug/release tests complete. No local Rust toolchain is available.

### 2026-11-15 — broaden all-enabled slice memory regression coverage

- Added a shared regression matrix for `load_or`, all-enabled `load_select`, and all-enabled `store_select` across representative supported lane counts from 1 through 64.
- Each lane count checks every slice length from empty through two elements beyond the vector width, covering the short-prefix path, exact full-width fast path, and oversized-slice tail preservation.
- The tests compare results against scalar slice semantics. No production code or unsafe behavior was changed in this test-only follow-up.
- Applied the formatting corrections reported by the first CI run. CI is required to execute the new matrix; no local Rust toolchain is available in this environment.

### 2026-11-15 — test full-width slice memory at shifted element offsets

- Added load/store regression coverage that advances slice starts through every element offset from zero to `LANES - 1` for representative vector widths from 1 to 64 lanes.
- The matrix checks `load_or`, all-enabled `load_select`, and `store_select` against scalar expectations, including preservation of the destination prefix and the element immediately after the vector write.
- This exercises the full-width fast path at different element-aligned addresses without assuming SIMD alignment. No production implementation was changed because no correctness defect has been demonstrated.
- Validation: pending CI on this commit; no local Rust toolchain is available in this environment.

### 2026-11-15 — exhaust duplicate-index scatter semantics

- Added an exhaustive four-lane `scatter_select` regression over every destination-index tuple (including duplicate indices, the first out-of-bounds index, and `usize::MAX`) and every enable mask.
- The oracle checks that disabled and out-of-bounds lanes never write, untouched destinations retain their sentinel, and destinations with duplicate enabled writers contain one of the permitted lane values. It deliberately does not assume a write order, matching the API contract.
- No production implementation change was made: the audit found no demonstrated correctness defect in the scatter implementation.
- Validation: pending GitHub Actions for this commit; no local Rust toolchain is available in this environment.


### 2026-11-15 — fix scatter regression test formatting

- The CI formatting job caught rustfmt differences in the shifted-offset and duplicate-index scatter regression tests.
- Applied the exact formatting changes reported by the pinned nightly rustfmt; test logic and SIMD implementation are unchanged.
- Validation: formatting will be rechecked by the CI run for this commit; no local Rust toolchain is available in this environment.

### 2026-11-15 — enforce documented duplicate scatter winner

- Tightened the exhaustive duplicate-index scatter_select regression oracle: for each destination, it now requires the highest enabled lane targeting that destination to win, matching the public API guarantee and its documented example.
- Removed the permissive assertion that accepted any colliding writer, which could hide a violation of the documented last-writer behavior.
- Kept exhaustive coverage over all 625 four-lane index tuples and all 16 masks, including out-of-bounds indices and usize::MAX.
- Validation: GitHub Actions must confirm formatting and all target test configurations; no local Rust toolchain is available in this environment.

### 2026-11-15 — expand gather bounds regression coverage

- Added gather_select boundary coverage for every supported lane width (1 through 64, including non-power-of-two widths).
- Rotates lanes through valid indices, the current slice-length boundary, the vector-width boundary, and usize::MAX while varying slice lengths and masks.
- Compares every result against scalar slice lookup and fallback semantics; production gather implementation remains unchanged because the audit found no demonstrated correctness defect.
- Validation is delegated to the repository CI; this environment has no local Rust toolchain.

### 2026-11-15 — record baseline-to-current stabilization assessment

- Added `docs/PROJECT_STATUS_AND_STABILIZATION.md` with the first-CI baseline, current validation evidence, the 16 upstream issues tracked in the development log, per-issue disposition, safety/codegen progress, stabilization roadmap, and explicitly qualified timeline ranges.
- The report distinguishes the fork's own issue tracker (disabled) from the upstream repository's open issue backlog, and distinguishes test/diagnostic progress from confirmed production fixes.
- The report records the last fully green run confirmed at assessment time and flags the current head CI as pending rather than claiming it is green.

### 2026-11-15 — fix gather boundary test formatting

- The latest head CI reported only rustfmt differences in the newly added gather-boundary test, not a Rust compilation or SIMD test failure.
- Applied the exact formatting suggested by the pinned nightly formatter and normalized the file's final newline.
- No test semantics or production SIMD code changed. The follow-up CI must confirm formatting and finish the remaining platform/Miri jobs; no local Rust toolchain is available here.


### 2026-11-15 — recheck current CI and stabilization blockers

- Rechecked branch main at 8cccd48d7acabbae6149991d30efff919b296b37 and the push-triggered run [43744268908](https://github.com/Likounsee/portable-simd/actions/runs/43744268908).
- The formatting and masked-memory checks passed, as did the Cross host-tool setup and multiple Windows, macOS, WASM, Clippy, and Miri jobs. The run was still unfinished at review time; no full-green claim is made.
- Reviewed upstream issue discussions #341, #427, and #472, including maintainer comments. #341 still explicitly asks for vector-typed LLVM IR loads/stores even when memcpy may be a better non-SIMD lowering; #427 still lacks a minimal reproducer and its historical report only tentatively recalls i8 with repr(simd, packed); #472 remains a compiler representation/codegen discussion, with maintainers considering range metadata / mask representation rather than removing validity invariants in the library.
- Audited upstream meta-issue #364, Critical issues before stabilization (still open; last updated 2026-08-04). It identifies API decisions not captured by the earlier 16-item implementation issue inventory: supported-lane-count bounds, mask element/representation design, swizzle API constraints, trait organization, and independently stabilizable feature boundaries. Added this as a separate cross-cutting blocker in docs/PROJECT_STATUS_AND_STABILIZATION.md.
- No production code was changed: the current CI has not exposed a new failure, and these upstream reports do not justify speculative unsafe/codegen changes. No local Rust toolchain is available, so validation remains delegated to GitHub Actions.


### 2026-11-15 — investigate hidden codegen regressions in upstream issue reports

- Reviewed upstream #321 in addition to the previously tracked issues. Its original AArch64 report shows an operation that is vectorized as a standalone function but scalarized after inlining into an iterator loop. The discussion points toward LLVM instruction-selection/reduction lowering, and a 2025 comment reports substantially improved output on a newer nightly. This is not evidence of a current library bug or a confirmed fix.
- Added #321 to the project status report as an investigated codegen regression candidate. The current fork's SSE2 masked-memory codegen artifact does not exercise this AArch64 reduction/inlining case; a future dedicated artifact should compile both standalone and inlined-loop forms and preserve LLVM IR plus assembly.
- Re-read #341 maintainer comments: the request is specifically for a vector-typed LLVM load/store where appropriate, even when array copies or memcpy are preferable for other cases. Existing fast-path evidence is useful but does not close the upstream request.
- Reviewed #486 source: x86 SSSE3 and AVX2 paths still use zeroing/clamping logic as documented; exhaustive semantic tests validate OOB zeroing but do not provide the optional implementation-defined-OOB API proposed by the issue. No new API was added without upstream consensus.
- No CI failure had been confirmed at the time of this review. No speculative production changes were made.


### 2026-11-15 — remove compiler-confirmed unused architecture feature gates

- Inspected complete Clippy logs rather than annotations: LoongArch Clippy reported `stdarch_loongarch` unused in `core_simd`; PowerPC64 Clippy reported `powerpc_target_feature` and `stdarch_powerpc` unused in `test_helpers`.
- Verified the LoongArch vendor module only imports the stable/current `core::arch::loongarch64` module and does not use APIs requiring that crate-level feature gate.
- Removed the LoongArch gate from `core_simd/src/lib.rs` and the PowerPC gates from `test_helpers/src/lib.rs`. Retained the separate PowerPC feature gate in `core_simd`, which is not the warning source.
- No target-specific algorithm or cfg selection was changed. Awaiting the matching CI jobs to verify the warning cleanup.


### 2026-11-15 — prepare GitHub Actions Node 24 runtime migration

- Full workflow logs repeatedly reported that pinned `actions/checkout@v4.4.0` and `actions/cache@v4.3.0` target Node.js 20 and are being forced to run on Node.js 24.
- Verified the official action metadata at `actions/checkout@v7.0.1` and `actions/cache@v6.1.0`: both explicitly declare `using: node24`.
- Verified the corresponding immutable tag commit SHAs before preparing the workflow update: checkout `3d3c42e5aac5ba805825da76410c181273ba90b1`; cache `55cc8345863c7cc4c66a329aec7e433d2d1c52a9`.
- This is a CI maintenance update only; no Rust code or cache key semantics are changed. Awaiting the current CI run before pushing to avoid cancelling it.

### 2026-11-15 — verify duplicate scatter winners across supported lane widths

- Added a scalar-oracle regression test for duplicate-index `scatter_select` behavior at representative lane counts from 1 through 64, including non-power-of-two widths.
- Exercises all-disabled, all-enabled, alternating, and edge-lane masks; duplicate valid destinations; the exact slice-length boundary; and `usize::MAX`.
- Checks the documented highest-enabled-lane winner and verifies that disabled, out-of-bounds, and destination-tail elements remain unchanged.
- This closes the test-coverage gap identified during the earlier scatter audit. It does not change the scatter implementation or claim a new performance improvement.
- Validation: pending the CI run triggered by this main-branch commit.

### 2026-11-15 — inspect fork-linked SSE2 assembly for the masked-memory fast paths

- Downloaded and inspected the actual `masked-load-store-codegen-8cccd48` artifact produced by CI, which links this fork's `core_simd` crate.
- Confirmed that the full-width all-enabled `load_select` path emits two 128-bit `movups` loads and the corresponding `store_select` path emits two `movups` stores for the eight-lane `f32` case.
- Short slices still use a bounded prefix-copy path, preserving fallback values and avoiding out-of-bounds access.
- Sparse `store_select` lowers to per-lane conditional scalar stores on SSE2. This is consistent with the absence of general masked-store instructions on SSE2; it remains a target-specific optimization candidate, not a correctness defect. No unsafe workaround was introduced.
- The artifact does not establish a measured end-to-end speedup. Controlled benchmark comparisons and AArch64/WASM codegen diagnostics remain outstanding.
- The previous scatter-collision test PR is now represented on `main`; no separate diff remains on its branch.


### 2026-11-15 — add the high-bit dynamic-swizzle fast path (#486)

- Added `Simd<u8, N>::swizzle_dyn_unchecked`, whose caller contract guarantees that every out-of-range index has bit 7 set.
- On x86/x86_64 builds with SSSE3 and exactly 16 lanes, the method now feeds indices directly to `PSHUFB`, avoiding the `zeroing_idxs` arithmetic required by the fully checked API.
- Other lane widths and targets fall back to `swizzle_dyn`, retaining the existing portable behavior rather than introducing target-specific assumptions.
- Added a regression test mixing valid indices with `0x80` and `0xff` out-of-range indices.
- This is a targeted implementation of upstream #486. Formatting, compile tests, the full target matrix, and generated-assembly verification are pending CI; no performance gain is claimed until the assembly is checked.

### 2026-11-15 — add a focused reproducer for upstream #426

- Added `crates/core_simd/examples/issue_426_reduction.rs` with the issue's interleaved XYZ calculation.
- The example compares a scalar loop, SIMD with a horizontal reduction on every record, and SIMD vector accumulation followed by one final reduction.
- It checks numerical agreement with a relative tolerance and reports timings for a local release-mode run. Timings are intentionally not asserted because they depend on the target CPU and compiler.
- The example is included in `cargo clippy --all-targets`; the new commit's CI must confirm it builds and passes before treating the reproducer as validated.

### 2026-11-15 — follow-up audit of upstream issues #524, #527, and #537

- #524 (Miri selecting unsupported architecture intrinsics) is closed upstream after the reported AArch64 swizzle path began working with newer Miri support. The discussion places the remaining missing-intrinsic work in Miri, not in a portable-simd software-fallback change; no speculative `cfg(miri)` branch was added.
- #527 (mask ordering) is already corrected in `crates/core_simd/src/simd/cmp/ord.rs`: mask ordering is expressed with boolean logic (`!self & other`, etc.) instead of signed integer comparisons. The fork also contains dedicated regression tests in `crates/core_simd/tests/mask_partial_ord.rs`, including exhaustive 8-lane patterns and 64-lane sign-bit cases.
- #537 (sealed-trait associated items leaking through a public supertrait) is already addressed in `crates/core_simd/src/masks.rs`: `PrivateMethods` is crate-private, and compile-fail doctests check that its associated items cannot be reached through `MaskElement`.
- These three reports were audited without changing production code: their fixes or resolution are already represented in the fork, and duplicating them would add risk without closing a remaining gap.


## Issue #366 — portable `maddubs` fallback

- Added `crates/core_simd/examples/issue_366_maddubs.rs` with a portable implementation of the semantics of x86 `_mm_maddubs_epi16`: unsigned-byte × signed-byte products, adjacent pair sums, and signed 16-bit saturation.
- Added scalar-reference regression tests covering saturation boundaries and varied signed/unsigned byte values.
- This is a semantic workaround/example, not a claim that LLVM emits `pmaddubsw`. The upstream discussion identifies instruction selection/codegen as the remaining limitation; a target-specific instruction is not appropriate as a portable API without broader architectural support.
- Validation is delegated to CI; no local Rust toolchain is available in this environment.

### 2026-11-15 — triage upstream issues #356 and #364

- #356 (bounds checks not eliminated) was reviewed together with its discussion and reproducer. The reported behavior is caused by bounds checks on ordinary slice indexing; the discussion confirms that an equivalent scalar implementation also retains checks, and that the original report used Rust 1.70 / LLVM 16. This is not evidence of a portable-SIMD correctness defect. No unsafe indexing change or SIMD API workaround was added.
- #364 (critical issues before stabilization) is an API-design/stabilization discussion covering supported lane counts, mask element types, and swizzle ergonomics. These are compatibility-sensitive design decisions rather than isolated implementation bugs. The fork should not silently redefine these public API contracts in a maintenance fix; the concerns remain tracked for upstream stabilization discussion.
- Outcome: both issues were triaged without production-code changes. Avoiding a speculative workaround is preferable to weakening memory safety or introducing an incompatible API change.

### 2026-11-15 — extend the #426 reduction reproducer with structure-of-arrays

- Added a fourth benchmark path that stores X, Y, and Z in separate contiguous slices, performs four records per SIMD iteration, accumulates in a vector, and reduces only once at the end.
- The example now contrasts the original interleaved (array-of-structures) layout with a structure-of-arrays layout, reflecting the upstream discussion that non-contiguous XYZ loads can limit vectorization independently of the horizontal-reduction cost.
- Added the SoA result to the scalar-reference tolerance checks and the small regression test. This is an explanatory benchmark/reproducer, not a guaranteed performance claim; actual timings remain target- and compiler-dependent.
- Validation: pending CI for the implementation commit.

### 2026-11-15 — audit upstream issue #486 (dynamic swizzle fast path)

- Confirmed the fork already contains `Simd<u8, N>::swizzle_dyn_unchecked`, which avoids `zeroing_idxs` on the x86 SSSE3 `u8x16` path and falls back to the regular zeroing implementation where that specialized path is unavailable.
- The method documents the contract: in-range indices select bytes; out-of-range indices with the high bit set yield zero; other out-of-range indices are implementation-defined.
- `crates/core_simd/tests/swizzle_dyn.rs` includes high-bit out-of-range regression coverage over lane widths from 1 through 64, alongside exhaustive byte-index coverage for the regular zeroing API.
- Outcome: #486's requested API/performance path is already present in this fork; no duplicate API or speculative target-specific implementation was added.

### 2026-11-15 — audit upstream issues #494, #286 and #321

- **#494 (masked memory codegen):** the fork already has explicit full-mask and contiguous-prefix fast paths in `Simd::load_select` and `Simd::store_select`. Sparse masks continue through the masked-load/store intrinsics. The diagnostic `benchmarks/masked_load_store.rs` compares SIMD and scalar paths, and `crates/core_simd/tests/masked_memory.rs` covers short/empty slices, every prefix length, lane-count boundaries, and gather/scatter edge cases. No further change was justified without target-specific assembly and timing evidence.
- **#286 (MaybeUninit SIMD lanes):** this is not a missing trait implementation that can safely be added in isolation. The upstream discussion identifies the mismatch between LLVM poison propagation and Rust's byte-level initialization model; exposing poison-like arithmetic risks unsoundness. Defer until the compiler/validity model provides a sound foundation.
- **#321 (scalarization after inlining):** the reproducer and discussion attribute the observed AArch64 regression to LLVM lowering/optimization of vector reductions and mask conversions. It is compiler/codegen behavior, not a portable-simd source-level correctness defect. Avoid target-specific source workarounds without reproducible compiler-version comparisons.

### 2026-11-15 — issue #539: algebraic SIMD reduction

- The fork already exposed elementwise `algebraic_add/sub/mul/div/rem` on `SimdFloat`, implemented by applying the scalar algebraic operation lane by lane.
- Added `SimdFloat::algebraic_reduce_sum` for f16/f32/f64 vectors. It folds lanes using scalar `algebraic_add`, explicitly opting into relaxed floating-point semantics so the compiler may reassociate the reduction.
- Added property-based tests for all float widths through the shared float test macro, comparing against the scalar algebraic fold with a 16-ULP tolerance.
- This API does not promise a particular vector instruction or speedup; target-specific codegen and benchmarks remain necessary to establish performance.


### 2026-11-15 — implement numeric SIMD `to_mask` conversion for issue #479

- Added `SimdInt::to_mask` and `SimdFloat::to_mask`, plus `SimdUint::to_mask`, converting each lane to a mask lane using comparison with zero.
- The unsigned trait does not expose its mask as an associated type, so its method returns the mask type from `SimdPartialEq` under an explicit trait bound rather than adding a new associated type to `SimdUint`.
- Float semantics are documented: NaN compares unequal to zero and therefore maps to `true`; both signed zeros map to `false`.
- This is a convenience API only; it does not reinterpret the underlying bits or promise a particular instruction sequence. CI validation is pending.

### 2026-11-16 — audit upstream issues #475, #211 and #369

- **#475 (`Simd::gather_unchecked`):** the fork already exposes this API in `crates/core_simd/src/vector.rs`. It documents the in-bounds precondition and delegates to the masked pointer gather with all lanes enabled. No duplicate implementation was needed.
- **#211 (horizontal min/max index):** no dedicated reduction-with-index API was added. The upstream discussion notes that the native instruction is not available consistently across targets or LLVM's portable IR. The fork already provides `Mask::first_set()`, which can find the first matching lane after comparing against `reduce_min()` or `reduce_max()`; this is the portable composition and keeps tie behavior explicit.
- **#369 (saturating narrowing casts):** no new public cast API was added. The upstream discussion confirms that `simd_clamp(...).cast()` already expresses the saturation pattern and that the relevant LLVM lowering fixes landed in LLVM 18/19. Adding a parallel API would duplicate the portable operation without a demonstrated compiler/codegen benefit.
- Outcome: all three requests were checked against the current fork and upstream discussion. Existing APIs or compiler lowering address the practical need; no speculative public API change was justified.

### 2026-11-16 — audit upstream issues #387 and #14

- **#387 (SIMD fused multiply-add):** the fork already exposes elementwise fused `mul_add` and `mul_sub` through `StdFloat` in `crates/std_float/src/lib.rs`, backed by the compiler's `simd_fma` intrinsic. This deliberately lives in the `std_float` layer because the current intrinsic may require runtime math-library support on targets without hardware support. The upstream request for a separate `no_std` operation with LLVM `fmuladd` semantics is not equivalent to the existing fused API; no substitute was added without a compiler intrinsic that guarantees the requested fallback semantics.
- **#14 (SIMD special functions):** the current `StdFloat` surface already provides `sqrt`, `sin`, `cos`, `exp`, `exp2`, `ln`, `log`, `log2`, and `log10`, alongside rounding and truncation operations. These are compiler/runtime-backed operations, not a missing set of simple portable arithmetic methods. Extending the set requires checking intrinsic availability and target runtime requirements individually; no speculative polynomial approximations were introduced.
- Outcome: the relevant functionality is already present in the `std_float` crate; the remaining no-std/runtime portability gap is compiler/intrinsic-level work rather than a safe local API duplication.


### 2026-11-16 — audit upstream issue #447 (generic-lane-count aliases)

- The fork already exports the requested shorthand aliases for every supported scalar lane type: `i8xN<N>`, `u32xN<N>`, `f32xN<N>`, and the corresponding signed, unsigned, pointer-sized, and floating-point variants.
- The aliases are generated in `crates/core_simd/src/alias.rs` and publicly re-exported from `core_simd::simd` through `crates/core_simd/src/mod.rs`.
- `crates/core_simd/tests/generic_aliases.rs` checks type equivalence for all scalar types and includes a generic function using `u32xN<N>` with a const-generic lane count.
- The upstream inverse aliases (`Tx1<T>`, `Tx2<T>`, etc.) are not implemented. They were not added: they are optional, less ergonomic in generic code, and would broaden the public API without a demonstrated need.
- Outcome: the central request in #447 is already implemented and covered by compile/runtime tests; no duplicate aliases were added.


### 2026-11-16 — audit upstream issues #373, #459, #146 and #297

- **#373 (maximum native lane count):** no portable, compile-time API can reliably infer a single "maximum" lane count from the target. The answer depends on enabled target features, ABI choices, and whether the caller wants the widest hardware vector or a width that performs best for a particular workload. The existing `LaneCount<N>: SupportedLaneCount` model lets callers choose N explicitly. No target-dependent public alias or hidden runtime dispatch policy was introduced.
- **#459 (vectors of pointers to struct/enum fields):** field projection can be expressed with pointer arithmetic and `offset_of!` where the layout contract permits it, but enum field projection and variant discrimination depend on layout/discriminant details that are not generally stable or safely introspectable. A generic helper would risk implying guarantees Rust does not provide. The linked upstream RFC for field projection is still a language-level dependency; no unsafe convenience API was added.
- **#146 (ARMv7 NEON boolean reduction codegen):** the report uses Rust/LLVM 1.55/12 from 2021. The follow-up discussion reports partial LLVM-side fixes with optimization-level sensitivity, so the historical assembly is not sufficient evidence of a current regression. This needs a current ARMv7 compiler/assembly reproducer before a source workaround is justified.
- **#297 (i64/u64 SIMD summation performance):** the upstream discussion explains that integer scalar loops are associative and can already auto-vectorize with target-aware widths, while manually fixed-width SIMD can inhibit that choice. This is not evidence of incorrect integer SIMD arithmetic or a missing API. Performance comparisons need optimized assembly and representative workloads; no redundant API was added.
- Outcome: these requests are either target/compiler policy questions or depend on Rust layout guarantees. The fork retains explicit lane-count selection and avoids promising unsupported layout or performance behavior.


### 2026-11-16 — audit upstream issue #242 (dynamic swizzling)

- The fork already implements byte-level runtime swizzling through `Simd<u8, N>::swizzle_dyn` in `crates/core_simd/src/swizzle_dyn.rs`.
- The implementation has target-specific paths for AArch64 NEON, ARMv7 NEON, WebAssembly SIMD128, LoongArch LSX/LASX, and x86 SSSE3/AVX2/AVX-512 VBMI, with a scalar fallback for unsupported widths/features.
- A separate `swizzle_dyn_unchecked` path documents the high-bit out-of-bounds contract and uses the SSSE3 byte-shuffle behavior where available.
- `crates/core_simd/tests/swizzle_dyn.rs` compares against a scalar reference, exhaustively checks all byte indices at lane-width boundaries from 1 to 64, and tests high-bit out-of-range indices for the unchecked variant.
- The remaining gap in #242 is the compiler/LLVM-level portable intrinsic and target-independent lowering, not the absence of usable library functionality in this fork. No duplicate intrinsic or inline-assembly workaround was added.
- Outcome: current library implementation and regression tests already cover the practical API request; upstream compiler integration remains separate work.

### 2026-11-16 — stabilize algebraic-reduction tests under Miri

- The Miri shards exposed a false-negative in the new algebraic_reduce_sum property test: a fixed 16-ULP comparison is too strict around zero after cancellation, particularly for f16.
- Replaced the ULP check with an absolute error bound scaled by the sum of input magnitudes and machine epsilon. The scale is floored at `Scalar::MIN_POSITIVE`, because multiplying a subnormal magnitude by `EPSILON` can underflow the tolerance to zero (reproduced by Miri on `f64` with a subnormal input).
- Generated values remain normalized to finite [-1, 1] values; no cases are discarded and no Miri tests are skipped.
- Validation: the first revised tolerance passed `f16` but exposed the subnormal-`f64` underflow described above. The new floor must be confirmed by CI; no local Rust test run was available.



### 2026-11-16 — refresh the stabilization report and begin the unsafe inventory

- Updated `docs/PROJECT_STATUS_AND_STABILIZATION.md` to the current main-line assessment and corrected stale issue dispositions for #475 (`gather_unchecked`), #479 (numeric `to_mask`), #486 (unchecked dynamic swizzle), and #539 (`algebraic_reduce_sum`).
- Added the implementation and triage work recorded through 2026-11-16, including #366 and #426, while distinguishing fork-side implementation from upstream issue closure and compiler-level resolution.
- Started a source inventory of unsafe operations in `vector.rs`, `masks.rs`, `select.rs`, pointer SIMD modules, and `std_float/src/lib.rs`. The inventory identifies pointer/memory, mask-validity, intrinsic, provenance, and unsafe-trait areas for review; it is not yet a soundness verdict.
- The last observed main CI run before this documentation update was run `43944365944` on `6d88cf5`, pending with 56 jobs. The current documentation commits must receive their own CI validation; no current-head green claim is made.
- No Rust implementation or unsafe block was changed in this documentation/audit-start commit. The next audit step is to inspect each unsafe API's contract against its implementation and compiler intrinsic requirements, then add focused tests or fixes only where a concrete defect is demonstrated.


### 2026-11-16 — clarify masked pointer load/store alignment contracts (#381)

- Compared the fork's public unsafe wrappers with the Rust SIMD intrinsic safety documentation for `simd_masked_load` and `simd_masked_store`.
- Both intrinsics use `SimdAlign::Element`, which requires the base pointer to be aligned to the SIMD element type independently of which lanes are enabled. The previous wrapper safety text described enabled-lane readability/writability but did not explicitly state this unconditional base-pointer alignment requirement.
- Updated `Simd::load_select_ptr` and `Simd::store_select_ptr` safety sections to require `align_of::<T>()` alignment even when the mask is all false, and clarified the matching internal SAFETY comments.
- This is a safety-contract/documentation correction for upstream #381; it does not change generated code or intrinsic behavior. No misaligned-pointer test was added because violating this contract would itself be undefined behavior. Full CI and Miri validation are pending.


### 2026-11-16 — add a focused code-generation diagnostic for issue #475

- Added `benchmarks/gather_unchecked.rs` to compare the safe `gather_or_default` path with `gather_unchecked` on identical, in-bounds indices.
- The benchmark asserts result equivalence before timing and documents that optimized assembly, compiler version, CPU, and target features must be captured before making a performance claim.
- This makes the original issue's bounds-check/codegen concern reproducible in the fork. It does not yet establish that the unchecked implementation emits better machine code; that requires running the diagnostic on the intended targets.


### 2026-11-16 — five-issue focused implementation pass (#475, #479, #486, #539, #447)

- Added `gather_unchecked.rs` regression coverage for permuted/repeated indices and bit-exact float lane movement (#475).
- Added a focused optimized-codegen/timing diagnostic for checked versus unchecked gathers. Performance conclusions are intentionally deferred until it is run and assembly is inspected (#475).
- Extended `to_mask.rs` with f16 subnormal, infinity, NaN, and signed-zero cases; existing f32/f64 tests already exercise the same classification edges (#479).
- Added `algebraic_float.rs` tests for all five lane-wise algebraic arithmetic operations on f32/f64 and for scalar-algebraic-fold equivalence of the reductions (#539).
- Extended dynamic swizzle regression tests to sweep every high-bit byte index from 0x80 through 0xff at widths around native vector boundaries (#486).
- Extended generic alias tests with a const-generic f32 computation; the pre-existing suite already verifies every scalar alias type (#447).
- CI for the latest main commit `48fc06f` was queued at the time of this entry. These changes are not considered validated until the complete run finishes. The benchmark has not yet produced measurements, so #475's performance claim remains open.


### 2026-11-16 — CI queue note for the five-issue sprint

- The workflow for commit `48fc06f` began, but the repository's concurrency policy cancelled that run after the subsequent documentation commit was pushed. This is a cancelled run, not a test failure and not a passing result.
- No further source changes are planned in this sprint pass so that the current-head CI run can execute without being superseded by another push.
