---
phase: 04-reproducibility
plan: 03
subsystem: testing
tags: [pytest, orchestrator, output-gate, csv-verification, repr-03]

# Dependency graph
requires:
  - phase: 04-reproducibility
    provides: "run_all_paper_experiments.py orchestrator, test infrastructure for orchestrator tests"
provides:
  - "_verify_outputs_exist(name, output_dir) helper in run_all_paper_experiments.py (lines 57-86)"
  - "Output-existence gate wired into main() per-experiment loop (lines 247-255)"
  - "TestOrchestratorOutputGate class with 6 tests in test_extend.py (lines 326-447)"
affects:
  - "04-reproducibility plan 04+ — gate now active on all future orchestrator runs"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Output-existence gate pattern: rc==0 AND is_root_rank() -> check data dir for non-empty CSVs"
    - "Generic CSV gate: any non-empty *.csv in data/ directory avoids brittle per-experiment manifests"

key-files:
  created: []
  modified:
    - "experiments/run_all_paper_experiments.py"
    - "experiments/tests/test_extend.py"

key-decisions:
  - "Generic gate rule (any non-empty CSV in data/) avoids brittle per-experiment filename manifests (D-07)"
  - "Gate layers on top of rc!=0 check, not replacing it — double-count prevented by elif branch (D-08)"
  - "Gate called only on root rank to avoid spurious failures on MPI worker ranks"
  - "Fake runner test needed a valid schema config (not just {}) because compute_scaling_factor validates in --test mode"

patterns-established:
  - "Output gate pattern: call _verify_outputs_exist only when rc==0 and is_root_rank(), append to failures on (False, reason)"

requirements-completed: [REPR-03]

# Metrics
duration: 15min
completed: 2026-04-14
---

# Phase 4 Plan 03: Output-Existence Gate Summary

**Output-existence gate added to orchestrator: rc=0 + missing/empty CSV outputs now fails the experiment, with 6 pytest tests covering all helper branches and pass/fail orchestrator paths**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-14T17:40:00Z
- **Completed:** 2026-04-14T17:55:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `_verify_outputs_exist(name, output_dir)` helper at lines 57-86 of `run_all_paper_experiments.py` returning `(True, None)` or `(False, reason)` with three distinct failure messages
- Wired gate into `main()` per-experiment loop (lines 247-255): called after `write_timing_csv` when `rc==0 and is_root_rank()`; failed gate appends to `failures` via `elif not outputs_ok:` branch
- Added `TestOrchestratorOutputGate` class (lines 326-447 of `test_extend.py`) with 6 tests: 4 unit tests for helper branches + 1 negative orchestrator path + 1 positive orchestrator path

## Task Commits

Each task was committed atomically:

1. **Task 1: Add _verify_outputs_exist helper and wire into main()** - `ea525d7` (feat)
2. **Task 2: Add TestOrchestratorOutputGate pytest coverage** - `cb1ccd5` (test)

**Plan metadata:** (docs commit follows)

_Note: TDD tasks — Task 1 modified source code; Task 2 added tests. Tests run and pass in both commits._

## Files Created/Modified
- `experiments/run_all_paper_experiments.py` - Added `_verify_outputs_exist` helper (lines 57-86) and output gate wiring in `main()` (lines 247-255)
- `experiments/tests/test_extend.py` - Added `TestOrchestratorOutputGate` class (lines 326-447) with 6 tests

## Decisions Made
- Generic "any non-empty CSV in data/" rule chosen over per-experiment manifest — avoids coupling gate to runner internals (D-07 intent)
- Gate placed after `write_timing_csv` call so timing is always recorded even when outputs are missing (debuggability)
- `elif not outputs_ok:` used (not `if`) to avoid double-counting failures when `rc != 0` (D-08 layering)
- Fake config in `test_orchestrator_fails_when_output_missing` needed full schema fields: `compute_scaling_factor` calls `load_config` which runs `_validate()` even for fake configs in `--test` mode

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fake runner config in test_orchestrator_fails_when_output_missing needed valid schema fields**
- **Found during:** Task 2 (test execution)
- **Issue:** Plan specified `config_path.write_text("{}")` but `compute_scaling_factor` calls `load_config` which invokes `_validate()` requiring `experiment_name`, `benchmark`, `methods`, `inference`, `execution` keys. Test crashed with `ValidationError: Config missing required top-level key: 'experiment_name'` before reaching the output gate assertion.
- **Fix:** Replaced `{}` with a minimal valid config containing all required schema keys. Benchmark name set to `gaussian_mean` (a valid entry in `VALID_BENCHMARK_NAMES`).
- **Files modified:** `experiments/tests/test_extend.py`
- **Verification:** All 6 `TestOrchestratorOutputGate` tests pass; `test_orchestrator_fails_when_output_missing` confirms `SystemExit(1)` with gate error logged.
- **Committed in:** `cb1ccd5` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in test fixture)
**Impact on plan:** Necessary fix for test to reach the gate under test. No scope creep.

## Issues Encountered
- `runtime_heterogeneity_runner.py` plot guard confirmed: runner still writes all data CSVs in `--test` mode (only plot generation is guarded by `if not test_mode:`). The gate works correctly in both modes — no special-casing needed.

## Next Phase Readiness
- REPR-03 gate is active: orchestrator now exits 1 when any runner produces missing or empty CSV outputs
- 18 tests in `test_extend.py` all pass (no regressions in `TestExtendOrchestrator` or other classes)
- Gate is generic (no per-experiment manifest) and will work for all 11 registered runners

---
*Phase: 04-reproducibility*
*Completed: 2026-04-14*
