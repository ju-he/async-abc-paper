# Codebase Concerns

**Analysis Date:** 2026-04-08

## Tech Debt

**Pervasive `sys.path.insert` for imports:**
- Issue: 27+ files use `sys.path.insert(0, ...)` to resolve the `experiments/` package. No proper Python packaging (`setup.py`, `pyproject.toml` with `[project.scripts]`) for the experiments directory.
- Files: `experiments/scripts/scaling_runner.py:28`, `experiments/scripts/straggler_runner.py:18`, `experiments/scripts/sensitivity_runner.py:14`, `experiments/tests/conftest.py:20`, `experiments/jobs/submit_scaling.py:57`, and 22+ more.
- Impact: Fragile import resolution, order-dependent path manipulation, hard to run tests from arbitrary directories, IDE tooling breaks.
- Fix approach: Add `experiments/` as a proper installable package in `pyproject.toml` with `pip install -e .` or use a `src/` layout. Replace all `sys.path.insert` calls.

**Global mutable state for Propulate imports:**
- Issue: `Propulator` and `ABCPMC` are module-level globals set to `None` and lazily assigned by `_ensure_propulate_imports()`. This couples import resolution with runtime state.
- Files: `experiments/async_abc/inference/propulate_abc.py:29-34`, `experiments/async_abc/inference/propulate_abc.py:116-134`
- Impact: Not thread-safe, makes testing harder (need to reset global state between tests), fragile if import fails partway through.
- Fix approach: Use a factory function or lazy-import wrapper class instead of global reassignment.

**`reporters.py` is 3956 lines (largest non-test file):**
- Issue: Single module handles all experiment-type plot generation. Contains 64+ functions covering scaling, straggler, SBC, sensitivity, ablation, runtime heterogeneity, and more.
- Files: `experiments/async_abc/plotting/reporters.py`
- Impact: Hard to navigate, high merge conflict risk, unclear ownership boundaries between experiment types.
- Fix approach: Split into per-experiment-type modules (e.g., `reporters_scaling.py`, `reporters_sbc.py`) with a thin dispatch layer.

**Legacy `concurrent_futures` MPI sampler path kept in-tree:**
- Issue: The `concurrent_futures` sampler for pyABC MPI runs has a documented history of 10+ deadlock bugs across April 2026. It remains opt-in but is kept in code with warnings.
- Files: `experiments/async_abc/inference/pyabc_sampler.py`, `experiments/async_abc/inference/abc_smc_baseline.py:333`, `experiments/async_abc/inference/pyabc_wrapper.py:349`
- Impact: Code complexity from maintaining two parallel execution paths. Risk of someone re-enabling the broken path.
- Fix approach: Remove `concurrent_futures` path entirely if no longer needed for the paper. If kept, gate behind a more prominent warning or config validation.

**sim_backend private API access for resource cleanup:**
- Issue: realistic workload benchmark accesses sim_backend's name-mangled private attribute `_SimDir__con` to close SQLite connections. No public API exists.
- Files: `experiments/async_abc/benchmarks/realistic_workload.py:672-685`
- Impact: Will break silently if sim_backend renames its internals. A warning is logged but the connection leaks.
- Fix approach: File upstream sim_backend issue for a public `close()` API on `DataHandler`. Already noted in code comments.

## Known Bugs

**No known open bugs at analysis time.** The extensive bug-fix log in `.plans/bug-fixes/previous-fixes.md` documents 12+ resolved MPI-related deadlocks and timing issues from April 2026. The current `CommWorldMap` approach appears stable.

## Security Considerations

**No significant security risks.** This is a scientific computing codebase for HPC cluster use, not a user-facing service. No web endpoints, no user input parsing, no authentication.

**FPE state manipulation via ctypes:**
- Risk: Direct `ctypes.CDLL(None)` call to libc for floating-point exception control. Architecture-specific constants (`_FE_ALL_EXCEPT = 0x3D`) are x86-only.
- Files: `experiments/async_abc/benchmarks/realistic_workload.py:29-30`, `experiments/async_abc/benchmarks/realistic_workload.py:40-43`, `experiments/async_abc/benchmarks/realistic_workload.py:50-67`
- Current mitigation: Constants are documented as x86-specific. `_LIBC` gracefully falls back to `None` if load fails.
- Recommendations: Add a runtime architecture check that warns or errors on non-x86 platforms.

## Performance Bottlenecks

**O(n^2) quality curve evaluation (mitigated but not eliminated):**
- Problem: `_async_archive_rows` in `convergence.py` reconstructs the posterior at each evaluation point. With many records, this is O(n * eval_points).
- Files: `experiments/async_abc/analysis/convergence.py`
- Cause: Each evaluation point filters all records up to that time, then computes Wasserstein distance. The `max_eval_points=500` cap prevents the worst case but analysis of large runs is still slow.
- Improvement path: Pre-sort records by time once, use cumulative indexing instead of repeated filtering. Cache intermediate posteriors.

**`scaling_runner.py` inline summary computation on rank 0:**
- Problem: The scaling runner computes quality curves inline between workloads. At high worker counts, this blocks all workers waiting for rank 0.
- Files: `experiments/scripts/scaling_runner.py`
- Cause: `_final_summary_row` calls `posterior_quality_curve` which is CPU-bound on rank 0 while other ranks wait at MPI barriers.
- Improvement path: Defer all summary computation to post-processing. Write raw records only during the MPI run.

## Fragile Areas

**MPI lifecycle management (most fragile area in the codebase):**
- Files: `experiments/async_abc/inference/pyabc_sampler.py`, `experiments/async_abc/inference/abc_smc_baseline.py`, `experiments/async_abc/inference/pyabc_wrapper.py`, `experiments/async_abc/inference/propulate_abc.py`
- Why fragile: MPI communicator lifecycle, barrier placement, and message drain logic have caused 12+ deadlock bugs. The current `CommWorldMap` solution avoids inter-communicators but adds a custom blocking map implementation that must handle all edge cases (zero work items, worker crashes, etc.).
- Safe modification: Always test MPI changes with `mpirun -np 4` minimum. Run the MPI integration helpers at `experiments/tests/mpi_integration_helper.py` and `experiments/tests/mpi_abc_smc_baseline_helper.py`. Never add collective operations inside the Propulate hot loop.
- Test coverage: Unit tests mock MPI. Real MPI integration tests exist but require manual `mpirun` invocation -- they are not part of the standard test suite.

**Propulate wall-time drain loop:**
- Files: `experiments/async_abc/inference/propulate_abc.py:275-303`
- Why fragile: The `Allreduce`-based drain loop after the Propulate hot loop is the result of fixing an asymmetric-drain deadlock. It requires ALL ranks to participate collectively. Adding any early-exit path or barrier asymmetry here will deadlock.
- Safe modification: Never add `return` or `break` statements that only some ranks execute. Always use collective operations that all ranks enter.
- Test coverage: No automated MPI test for the drain path specifically.

**Propulate intra-request cleanup (`_cleanup_propulate_intra_requests`):**
- Files: `experiments/async_abc/inference/propulate_abc.py:60-86`
- Why fragile: Uses `MPI.Request.Testsome` to prune completed sends. Index-based deletion in reverse order is correct but easy to break. The `intra_buffers` list must stay synchronized with `intra_requests`.
- Safe modification: Do not change index arithmetic without verifying against the MPI specification for `Testsome` return values.
- Test coverage: No unit test for `Testsome` index edge cases.

**realistic workload benchmark (`realistic_workload.py`):**
- Files: `experiments/async_abc/benchmarks/realistic_workload.py` (694 lines)
- Why fragile: Depends on sim_backend private APIs, x86-specific FPE constants, runtime `sys.path` manipulation for venv resolution, file-system-based simulation I/O with UUID directories, and JSON config rewriting. Many failure modes return `float('nan')` silently.
- Safe modification: Always test with the sim_backend venv at `sim_backend_venv/.venv`. Check NaN rate after changes.
- Test coverage: `experiments/tests/test_benchmarks.py` covers the public API but not sim_backend integration (requires the full simulation stack).

## Scaling Limits

**Single-rank summary computation:**
- Current capacity: Works for runs producing up to ~50k records per workload.
- Limit: Quality curve computation on rank 0 blocks the entire MPI world. At 48+ workers with many replicates, this caused multi-minute stalls (documented in bug fixes).
- Scaling path: Move summary computation to post-processing scripts that run after the MPI job completes.

**MPI message volume in Propulate:**
- Current capacity: Bounded by `_cleanup_propulate_intra_requests` which prunes completed sends each iteration.
- Limit: High-throughput configurations (k=10, many workers) can still accumulate thousands of outstanding `isend` requests before `Testsome` can retire them.
- Scaling path: Rate-limit sends or use synchronous send for intra-island communication when backlog exceeds a threshold.

## Dependencies at Risk

**sim_backend (vendored at `sim_backend_venv/`):**
- Risk: Private API dependency (`_SimDir__con`). No public release, vendored as a local copy. Updates require manual sync.
- Impact: realistic workload benchmark breaks if internal API changes.
- Migration plan: Request public `close()` API upstream. Pin to known-working commit.

**propulate (vendored at `propulate/`):**
- Risk: Uses internal attributes (`intra_requests`, `intra_buffers`, `worker_sub_comm`, `island_comm`, `propulate_comm`, `generation`, `generations`). The wrapper reaches deep into Propulate internals for wall-time control and MPI cleanup.
- Impact: Any Propulate update that renames or restructures these attributes breaks the inference wrapper.
- Files: `experiments/async_abc/inference/propulate_abc.py:60-86`, `experiments/async_abc/inference/propulate_abc.py:237-308`
- Migration plan: Propose upstream API for wall-time-limited runs and clean communicator teardown. Keep vendored copy pinned until upstream supports these use cases.

## Missing Critical Features

**No automated MPI integration testing:**
- Problem: MPI integration tests (`mpi_integration_helper.py`, `mpi_abc_smc_baseline_helper.py`) must be run manually with `mpirun`. They are not in the pytest suite.
- Blocks: Regression detection for the most fragile code paths (MPI lifecycle, drain loops, communicator teardown).

**No CI pipeline:**
- Problem: No continuous integration configuration detected. Tests run only locally.
- Blocks: Automated regression detection, especially for the HPC-specific MPI paths that have historically broken.

## Test Coverage Gaps

**MPI drain and teardown paths:**
- What's not tested: The `Allreduce`-based drain loop in `propulate_abc.py:275-303`, `CommWorldMap` under real MPI, and the `concurrent_futures` legacy path under real MPI.
- Files: `experiments/async_abc/inference/propulate_abc.py`, `experiments/async_abc/inference/pyabc_sampler.py`
- Risk: These are the historically most bug-prone paths. Regressions would only surface on the HPC cluster.
- Priority: High

**realistic workload benchmark integration:**
- What's not tested: Full sim_backend simulation pipeline, FPE state restoration, config path rewriting with real sim_backend configs.
- Files: `experiments/async_abc/benchmarks/realistic_workload.py`
- Risk: NaN-returning simulations could silently degrade inference quality without detection.
- Priority: Medium

**Scaling runner workload orchestration:**
- What's not tested: Multi-workload MPI coordination in `scaling_runner.py`, shared executor reuse across k-values, shard merging under concurrent SLURM jobs.
- Files: `experiments/scripts/scaling_runner.py` (1094 lines)
- Risk: Workload ordering bugs or shard race conditions only manifest on the cluster.
- Priority: Medium

**Broad `except Exception` handlers:**
- What's not tested: 41 bare `except Exception:` handlers across 20 files. Many silently return defaults (`None`, `0`, `1`, `False`, empty DataFrame) or `continue` past errors.
- Files: See grep results -- notably `experiments/async_abc/utils/mpi.py:38-39`, `experiments/async_abc/utils/runner.py:283-284`, `experiments/async_abc/analysis/sensitivity.py:216-217`, `experiments/async_abc/plotting/export.py:46-47`
- Risk: Real errors (e.g., corrupted data files, MPI state corruption) are silently swallowed, making debugging harder.
- Priority: Medium -- many are intentional best-effort paths, but several (e.g., in `runner.py:283` and `sensitivity.py:216`) could hide data corruption.

---

*Concerns audit: 2026-04-08*
