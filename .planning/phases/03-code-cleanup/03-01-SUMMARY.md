---
phase: 03-code-cleanup
plan: 01
subsystem: inference-layer
tags: [dead-code-removal, mpi, pyabc, cleanup]
dependency_graph:
  requires: [02-03]
  provides: [clean inference layer — CommWorldMap sole MPI path]
  affects: [pyabc_sampler.py, pyabc_wrapper.py, abc_smc_baseline.py, scaling_runner.py, test_inference.py]
tech_stack:
  added: []
  patterns: [ValueError-on-invalid-sampler, single-bind-then-pass for resolve functions]
key_files:
  created: []
  modified:
    - experiments/async_abc/inference/pyabc_sampler.py
    - experiments/async_abc/inference/pyabc_wrapper.py
    - experiments/async_abc/inference/abc_smc_baseline.py
    - experiments/scripts/scaling_runner.py
    - experiments/async_abc/inference/method_registry.py
    - experiments/tests/test_inference.py
    - experiments/tests/mpi_integration_helper.py
    - experiments/tests/mpi_abc_smc_baseline_helper.py
    - .plans/bug-fixes/previous-fixes.md
decisions:
  - concurrent_futures and concurrent_futures_legacy now raise ValueError (loud failure, not silent fallback)
  - client_max_jobs kept in build_pyabc_sampler signature for forward-compat even though mapping path ignores it
  - TrackedFutureExecutor removed entirely — no callers post Phase 2
metrics:
  duration: "11 minutes"
  completed: "2026-04-14"
  tasks_completed: 3
  files_modified: 9
requirements_satisfied: [CODE-01, CODE-03]
---

# Phase 03 Plan 01: Dead MPICommExecutor/concurrent_futures Code Removal Summary

**One-liner:** Removed TrackedFutureExecutor class and all MPICommExecutor/concurrent_futures paths from inference layer; `resolve_pyabc_mpi_sampler` now raises ValueError for legacy sampler names; full test suite 596 passed.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Wave-0 test update — rewrite concurrent_futures tests to assert ValueError (TDD RED) | 1d8fdd2 |
| 2 | Remove dead MPICommExecutor/concurrent_futures paths from inference files (TDD GREEN) | 0079f70 |
| 3 | Clean stale MPICommExecutor comments in scaling_runner and method_registry | 0ca887f |

## Files Modified

### experiments/async_abc/inference/pyabc_sampler.py

**Net change:** removed ~55 lines (TrackedFutureExecutor class + concurrent_futures branch in build_pyabc_sampler + old resolve logic)

- Removed `from concurrent.futures import wait as _wait` import (line 11)
- Removed entire `TrackedFutureExecutor` class (lines 16–45)
- Simplified `resolve_pyabc_mpi_sampler`: removed concurrent_futures/concurrent_futures_legacy branches; unknown values now raise `ValueError("Valid values: 'mapping'.")` directly
- Simplified `build_pyabc_sampler`: removed `cfuture_executor` parameter and `concurrent_futures` branch; only `mapping` path remains for MPI backend

### experiments/async_abc/inference/pyabc_wrapper.py

**Net change:** removed ~40 lines (TrackedFutureExecutor import, mpi_executor param, MPICommExecutor branch)

- Removed `TrackedFutureExecutor` from import block
- Removed `mpi_executor=None` parameter from `run_pyabc_smc` signature
- Removed dead shared-MPICommExecutor branch and concurrent_futures branch
- Updated comment above CommWorldMap block

### experiments/async_abc/inference/abc_smc_baseline.py

**Net change:** removed ~40 lines + consolidated double resolve call

- Removed `TrackedFutureExecutor` from import block
- Removed `mpi_executor=None` parameter from `run_abc_smc_baseline` signature
- Consolidated double `resolve_pyabc_mpi_sampler` call into single bind-then-pass (Pitfall 5 fix)
- Removed dead shared-MPICommExecutor branch and concurrent_futures branch
- Updated comment above CommWorldMap block

### experiments/scripts/scaling_runner.py

**Net change:** comment-only updates

- `_run_workloads` docstring: removed "no shared MPICommExecutor" phrasing
- Pass 2 comment: removed "Unlike the old shared MPICommExecutor path" comparison

### experiments/async_abc/inference/method_registry.py

**Net change:** 1 line docstring update

- Removed `(e.g. ``mpi_executor``)` from `**kwargs` documentation

### experiments/tests/test_inference.py

**Net change:** removed ~250 lines, added ~30 lines

**Deleted test methods:**
- `test_mpi_concurrent_futures_backend_returns_concurrent_future_sampler` — exercised removed `concurrent_futures` branch of `build_pyabc_sampler`
- `test_mpi_concurrent_futures_backend_forwards_client_max_jobs` — exercised `ConcurrentFutureSampler` construction, path removed
- `test_mpi_concurrent_futures_backend_requires_existing_executor` — exercised removed branch
- `test_tracked_future_executor_waits_only_for_non_cancelled_pending_futures` — `TrackedFutureExecutor` class removed entirely
- `test_mpi_legacy_backend_is_forwarded` (TestPyabcWrapperMpiBackend) — passed `concurrent_futures` which now raises ValueError at resolve
- `test_mpi_legacy_backend_key_is_forwarded` (TestAbcSmcBaselineMpiBackend) — same reason
- `test_abc_smc_baseline_shutdown_does_not_hang` — used `concurrent_futures` helper arg
- `test_abc_smc_baseline_mpi_diagnostics_bounded_backlog` — parametrized on `concurrent_futures`
- `test_pyabc_smc_mpi_diagnostics_mapping` — despite name, passed `concurrent_futures`
- `test_pyabc_smc_mpi_legacy_remains_selectable` — passed `concurrent_futures_legacy`

**Added test methods:**
- `test_mpi_concurrent_futures_now_raises_value_error` — asserts `build_pyabc_sampler(..., mpi_sampler="concurrent_futures")` raises ValueError
- `test_resolve_pyabc_mpi_sampler_rejects_legacy_alias` — asserts `resolve_pyabc_mpi_sampler({"pyabc_mpi_sampler": "concurrent_futures_legacy"}, ...)` raises ValueError
- `test_resolve_pyabc_mpi_sampler_rejects_concurrent_futures` — asserts `resolve_pyabc_mpi_sampler({"pyabc_mpi_sampler": "concurrent_futures"}, ...)` raises ValueError

**Updated:**
- `_install_fake_mpi_executor`: removed `_FakeMPICommExecutor` / `fake_futures` / `mpi4py.futures` monkeypatching (dead code)
- `test_resolve_pyabc_client_max_jobs_defaults_to_worker_count_for_mpi`: changed `mpi_sampler="concurrent_futures"` → `"mapping"` (mapping returns None)
- `test_resolve_pyabc_client_max_jobs_honors_explicit_override`: same change (mapping always returns None regardless of explicit override)
- `test_non_root_returns_no_records_under_mpi_backend` docstring updated (removed stale concurrent_futures reference)

## Test Results

**Before (current HEAD before Task 1):** 103 tests collected (approximate)
**After Task 1 (RED):** 101 tests collected; new ValueError tests fail as expected
**After Task 2 (GREEN):** 98 passed, 3 skipped — all Task 1 new tests now pass
**After Task 3 (full suite):** 596 passed, 10 skipped, 1 warning — complete regression suite green

## Grep Verification Outputs

```
pyabc_sampler.py:   TrackedFutureExecutor=0 MPICommExecutor=0 mpi_executor=0 concurrent_futures=0
pyabc_wrapper.py:   TrackedFutureExecutor=0 MPICommExecutor=0 mpi_executor=0 concurrent_futures=0
abc_smc_baseline.py: TrackedFutureExecutor=0 MPICommExecutor=0 mpi_executor=0 concurrent_futures=0
scaling_runner.py: "shared MPICommExecutor" = 0 matches
method_registry.py: "mpi_executor" = 0 matches
```

## Bug Log Entry

`.plans/bug-fixes/previous-fixes.md` updated with new entry dated 2026-04-14:

```
## 2026-04-14 — Phase 3 Plan 01: Dead code removal

- Removed TrackedFutureExecutor class and MPICommExecutor / concurrent_futures
  branches from pyabc_sampler.py, pyabc_wrapper.py, abc_smc_baseline.py
- mpi_executor kwarg dropped from run_pyabc_smc and run_abc_smc_baseline
  (no callers post-Phase-2 scaling_runner migration)
- resolve_pyabc_mpi_sampler now rejects 'concurrent_futures' and
  'concurrent_futures_legacy' with ValueError
- Tests updated to match (test_mpi_concurrent_futures_now_raises_value_error,
  test_resolve_pyabc_mpi_sampler_rejects_*)
- Rationale: Phase 2 D-03 migrated scaling_runner to CommWorldMap. No config
  or caller references these paths.
```

## Requirements Satisfied

- **CODE-01**: MPI inference layer simplified — three inference files reduced by ~135 lines total; single CommWorldMap path remains
- **CODE-03**: Dead/legacy code removed — TrackedFutureExecutor class gone, concurrent_futures branches gone, mpi_executor parameter gone

## Deviations from Plan

**1. [Rule 1 - Bug] Consolidated double resolve_pyabc_mpi_sampler call in abc_smc_baseline.py**
- Found during: Task 2 (Edit 3c as described in plan)
- Issue: `client_max_jobs` resolution called `resolve_pyabc_mpi_sampler` inline, then `mpi_sampler` was resolved again immediately after — double invocation of the same resolve function
- Fix: Bind `mpi_sampler` first, then pass to `resolve_pyabc_client_max_jobs`
- Files modified: `experiments/async_abc/inference/abc_smc_baseline.py`
- Commit: 0079f70 (included in Task 2 commit)
- Note: This was actually documented in the plan as Pitfall 5 — executed as specified

**2. [Plan-driven] test_resolve_pyabc_client_max_jobs tests changed to expect None instead of int**
- The original tests passed `mpi_sampler="concurrent_futures"` which returned a non-None int. After the change to `mpi_sampler="mapping"`, the function always returns None for mapping (documented in `resolve_pyabc_client_max_jobs`). Updated assertions accordingly rather than keeping a now-misleading test name implying non-None return.

## Known Stubs

None — all data flows are wired through the real CommWorldMap path.

## Self-Check: PASSED
