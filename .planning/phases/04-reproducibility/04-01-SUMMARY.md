---
phase: 04-reproducibility
plan: 01
subsystem: testing
tags: [pytest, csv, extend-mode, reproducibility, deterministic-seeds]

# Dependency graph
requires:
  - phase: 03-code-cleanup
    provides: gaussian_mean_runner.py test infrastructure, run_runner_main, patched_method_registry, timed_fake_method
provides:
  - pytest test asserting --extend on partial CSV produces identical row set as fresh run (REPR-01)
affects: [04-reproducibility]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-subdir pattern for fresh vs extend within single tmp_path (fresh_dir = tmp_path/'fresh', extend_dir = tmp_path/'extended')"
    - "Row-set equality on (method, replicate, seed, step, loss) tuples as primary REPR check"

key-files:
  created: []
  modified:
    - experiments/tests/test_extend.py

key-decisions:
  - "String equality holds for loss values across fresh/extend: no rounding needed — both runs go through identical deterministic seed path and CSV round-trip"
  - "Partial CSV fixture constructed inline (subset of fresh rows) rather than a hardcoded synthetic row — ensures structural validity and real column names"

patterns-established:
  - "REPR test pattern: run fresh, subset rows into second dir, extend, assert set equality on stable tuple key"

requirements-completed: [REPR-01]

# Metrics
duration: 8min
completed: 2026-04-14
---

# Phase 4, Plan 01: Extend-vs-Fresh Equivalence Test Summary

**Pytest test verifying --extend on a partial CSV produces the identical row set as a fresh run via set equality on (method, replicate, seed, step, loss) tuples — REPR-01 satisfied.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-14T17:32:00Z
- **Completed:** 2026-04-14T17:40:42Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Added `test_extend_matches_fresh_run_same_seed` (lines 167-223) inside `TestExtendBasicRunner` in `experiments/tests/test_extend.py`
- Test runs gaussian_mean fresh (ground truth), writes a partial CSV (rejection_abc replicate=0 only), runs --extend, then asserts row-set equality
- String equality holds for loss values: no rounding was needed (deterministic seed path, CSV round-trip identical in both runs)
- All 12 tests in `test_extend.py` pass with no regressions

## Task Commits

1. **Task 1: Add extend-vs-fresh-run equivalence test** - `0efe6dd` (feat)

## Files Created/Modified

- `experiments/tests/test_extend.py` - Added `test_extend_matches_fresh_run_same_seed` (57 lines inserted after line 165)

## Decisions Made

- String equality for loss values holds without rounding — both CSVs produced via same deterministic seed path and round-tripped through CSV serialization. Exact match confirmed in test run.
- Partial CSV fixture built inline from fresh run rows (filter to rejection_abc replicate=0) rather than writing a hardcoded synthetic row — this guarantees structural validity and correct fieldnames without duplication of the existing `test_extend_runs_only_missing_combinations` fixture pattern.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- REPR-01 is satisfied: `--extend` equivalence to fresh run is now enforced by a passing test
- Ready to proceed to plan 04-02 (seed consistency / seeding.py tests)

---
*Phase: 04-reproducibility*
*Completed: 2026-04-14*
