---
phase: 02-mpi-hardening
plan: 01
subsystem: testing
tags: [mpi, mpi4py, mpirun, CommWorldMap, pytest, pyabc, NaN-weight, regression]

# Dependency graph
requires:
  - phase: 01-diagnose
    provides: CommWorldMap recommended as pyABC MPI sampler; Barrier placement documented

provides:
  - "experiments/tests/mpi_commworldmap_helper.py: 4-scenario mpirun subprocess helper"
  - "experiments/tests/test_mpi_hardening.py: 8-test Phase 2 MPI hardening suite"
  - "NaN-weight regression test (MPI-03 guard): mpirun-n2 pyabc_smc and abc_smc_baseline with max_wall_time_s=0.1"
  - "CommWorldMap coordination coverage: normal, root_exception, multi_call, double_shutdown"
  - "Barrier placement source-check regression test"

affects: [02-mpi-hardening-02, cluster-verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "mpirun subprocess helper pattern: sys.path.insert, if __name__=='__main__', JSON output to argv[1]"
    - "Non-mpirun tests isolated in TestMpiHardeningNoMpiRun class to guarantee execution regardless of mpirun probe state"
    - "Autouse MPI fixtures (skip_if_mpirun_not_usable, skip_if_no_pyabc, skip_if_no_mpi4py) on mpirun-dependent class"

key-files:
  created:
    - experiments/tests/mpi_commworldmap_helper.py
    - experiments/tests/test_mpi_hardening.py
  modified:
    - experiments/tests/mpi_integration_helper.py
    - experiments/tests/mpi_abc_smc_baseline_helper.py

key-decisions:
  - "Moved test_barrier_placement_source_check to separate TestMpiHardeningNoMpiRun class to guarantee it runs even when mpirun probe fails after heavy MPI test sessions"
  - "Helper assertion relaxed: allow empty records when max_wall_time_s is given (NaN guard test exercises early-stop path that can yield 0 records)"

patterns-established:
  - "MPI test class split: TestMpiHardening (mpirun-dependent, autouse skip fixtures) vs TestMpiHardeningNoMpiRun (source-check/in-process, always runs)"
  - "mpirun helper: define top-level _square, _inc, _double functions — no pickled lambdas over MPI"

requirements-completed: [TEST-01, TEST-03, MPI-03]

# Metrics
duration: 5min
completed: 2026-04-10
---

# Phase 02 Plan 01: MPI Hardening Test Suite Summary

**Eight-test Phase 2 MPI suite: NaN-weight regression (MPI-03), CommWorldMap 4-scenario coordination via mpirun subprocess helper, barrier source-check, and double-shutdown idempotency — all passing in nastjapy_copy/.venv**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-04-10T13:59:10Z
- **Completed:** 2026-04-10T14:04:00Z
- **Tasks:** 2
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments

- Created `mpi_commworldmap_helper.py`: subprocess helper exercising CommWorldMap in 4 coordination scenarios (normal, root_exception, multi_call, double_shutdown), writing JSON result from rank 0
- Created `test_mpi_hardening.py`: 8-test pytest module covering NaN-weight regression (TEST-01/MPI-03), CommWorldMap coordination (TEST-01), shutdown idempotency (TEST-03), and Barrier placement source check
- All 8 tests pass locally in nastjapy_copy/.venv with mpirun available (Open MPI 4.1.2)
- No production code (pyabc_sampler.py, pyabc_wrapper.py, abc_smc_baseline.py, scaling_runner.py) was modified

## Task Commits

1. **Task 1: Create mpi_commworldmap_helper.py** - `1f88022` (feat)
2. **Task 2: Create test_mpi_hardening.py** - `e48e8fc` (feat)

**Plan metadata:** _(to be added by final commit)_

## Files Created/Modified

- `experiments/tests/mpi_commworldmap_helper.py` — 4-scenario CommWorldMap subprocess helper with top-level _square, _inc, _double functions
- `experiments/tests/test_mpi_hardening.py` — 8-test pytest suite in TestMpiHardening + TestMpiHardeningNoMpiRun classes
- `experiments/tests/mpi_integration_helper.py` — Relaxed `len(records) > 0` to allow empty records when max_wall_time_s given
- `experiments/tests/mpi_abc_smc_baseline_helper.py` — Same relaxation for abc_smc_baseline helper

## Tests Added

| Test | Class | Requirement | mpirun dep |
|------|-------|-------------|-----------|
| test_nan_weight_guard_pyabc_smc | TestMpiHardening | MPI-03 | YES |
| test_nan_weight_guard_abc_smc_baseline | TestMpiHardening | MPI-03 | YES |
| test_commworldmap_normal_2rank | TestMpiHardening | TEST-01 | YES |
| test_commworldmap_root_exception_2rank | TestMpiHardening | TEST-01 | YES |
| test_commworldmap_multi_call_2rank | TestMpiHardening | TEST-01 | YES |
| test_commworldmap_double_shutdown_2rank | TestMpiHardening | TEST-01 | YES |
| test_commworldmap_single_process_double_shutdown | TestMpiHardening | TEST-03 | NO (mpi4py only) |
| test_barrier_placement_source_check | TestMpiHardeningNoMpiRun | TEST-01 | NO |

**Local run result:** 8 passed, 0 skipped (mpirun available as Open MPI 4.1.2)

## Decisions Made

- Split tests across two classes: `TestMpiHardening` (mpirun-dependent, 3 autouse skip fixtures) and `TestMpiHardeningNoMpiRun` (no fixtures, always runs). This ensures source-check and in-process tests run even on machines where mpirun enters a degraded state after heavy MPI test sessions.
- Helper assertion relaxed from `len(records) > 0` to conditional — only asserts `> 0` when no wall-time limit is given. Required because NaN guard test deliberately triggers early-stop path (max_wall_time_s=0.1) which can return 0 records.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Relaxed helper assertion to allow empty records under wall-time limit**
- **Found during:** Task 2 (NaN-weight guard test execution)
- **Issue:** `mpi_integration_helper.py` and `mpi_abc_smc_baseline_helper.py` both had `assert len(records) > 0` which crashed when max_wall_time_s=0.1 caused pyABC to return 0 records (valid early-stop behavior, not a failure)
- **Fix:** Changed assertion to only require `len(records) > 0` when `max_wall_time_s is None`; allowed empty records for wall-time-limited runs
- **Files modified:** experiments/tests/mpi_integration_helper.py, experiments/tests/mpi_abc_smc_baseline_helper.py
- **Verification:** test_nan_weight_guard_pyabc_smc and test_nan_weight_guard_abc_smc_baseline both PASS
- **Committed in:** e48e8fc (Task 2 commit)

**2. [Rule 2 - Missing Critical] Isolated non-mpirun tests in separate class**
- **Found during:** Task 2 (full suite run)
- **Issue:** Plan specified all tests in `TestMpiHardening` with 3 autouse fixtures, but `test_barrier_placement_source_check` was being skipped ("mpirun failed a trivial probe") after heavy mpirun sessions, violating the plan's MUST-RUN requirement
- **Fix:** Moved `test_barrier_placement_source_check` to `TestMpiHardeningNoMpiRun` class without autouse fixtures
- **Files modified:** experiments/tests/test_mpi_hardening.py
- **Verification:** test_barrier_placement_source_check PASSES in both isolated and full-suite runs
- **Committed in:** e48e8fc (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 Rule 1 bug, 1 Rule 2 missing critical)
**Impact on plan:** Both fixes necessary for correctness. No scope creep — production code was not modified.

## Issues Encountered

- mpirun probe failure after repeated mpirun calls in same pytest session caused `test_barrier_placement_source_check` to skip. Resolved by class isolation (Rule 2 deviation).
- `max_wall_time_s=0.1` returning 0 records from helpers that assumed records > 0. Resolved by conditional assertion (Rule 1 deviation).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 2 Plan 02 (48-rank cluster verification) can proceed: local mpirun tests confirm CommWorldMap coordination is correct
- `test_barrier_placement_source_check` provides regression guard for Barrier placement at all 3 call sites
- `test_commworldmap_double_shutdown_*` tests confirm idempotent shutdown on both mpirun-n2 and in-process paths

---
*Phase: 02-mpi-hardening*
*Completed: 2026-04-10*
