---
phase: 04-reproducibility
plan: "02"
subsystem: testing/seeding
tags: [reproducibility, determinism, pytest, rejection-abc, seeding]
dependency_graph:
  requires: []
  provides: [REPR-02]
  affects: [experiments/tests/test_seeding.py]
tech_stack:
  added: []
  patterns: [run_runner_main end-to-end, set-equality row comparison, conftest-as-test-helpers]
key_files:
  created: []
  modified:
    - experiments/tests/test_seeding.py
decisions:
  - "Scoped to rejection_abc only per D-04 — MPI-based methods excluded by design"
  - "Key columns for comparison: method, replicate, seed, step, loss — wall_time excluded (timing-sensitive)"
  - "String equality holds for all key columns — no pytest.approx needed; rejection_abc is bit-deterministic from same seed"
metrics:
  duration: "~4 minutes"
  completed_date: "2026-04-14"
---

# Phase 4 Plan 2: Rejection ABC Seed Determinism Test Summary

End-to-end pytest test verifying rejection_abc produces identical CSV row sets across two runs with the same seed, satisfying REPR-02 scoped to single-process methods per D-04.

## What Was Built

Added `TestRunnerDeterminism` class to `experiments/tests/test_seeding.py` (lines 71-122). The class contains one test method `test_rejection_abc_same_seed_produces_same_rows` that:

1. Builds a gaussian_mean config with `methods=["rejection_abc"]`, `base_seed=1`, `n_replicates=2`, `max_simulations=80`, `k=10`
2. Runs `gaussian_mean_runner.py` into two separate tmp directories (run_a, run_b)
3. Reads both `raw_results.csv` outputs and asserts:
   - Both CSVs exist
   - Run A covers both replicates: `{("rejection_abc", "0"), ("rejection_abc", "1")}`
   - Row sets are equal on `["method", "replicate", "seed", "step", "loss"]`
   - Row counts are equal

Two import changes at the top of the file enabled conftest access:
- Added `import csv`, `import sys`, `from pathlib import Path`
- Added `sys.path.insert(0, ...)` + `import conftest as test_helpers` (mirroring test_extend.py pattern)

## Decisions Made

- **Key columns:** `["method", "replicate", "seed", "step", "loss"]` — excludes `wall_time` which is legitimately non-deterministic (timing measurement). All other key columns are deterministically derived from seed.
- **String equality sufficient:** rejection_abc with the same seed traverses identical numpy/random call sequences and the same CSV round-trip. No `pytest.approx` needed — exact string equality holds for all key columns.
- **No test_mode=True:** `max_simulations=80` override provides fast execution without needing test mode short-circuit.
- **Class-level organization:** New test placed in `TestRunnerDeterminism` class at end of file, keeping `TestMakeSeeds` and `TestSeedEverything` untouched.

## New class location

- `class TestRunnerDeterminism` — lines 71-122
- `def test_rejection_abc_same_seed_produces_same_rows` — line 72

## Deviations from Plan

None — plan executed exactly as written.

## Test Results

```
experiments/tests/test_seeding.py ...........    [100%]
11 passed in 1.12s
```

All 11 tests pass (10 pre-existing + 1 new). The new test runs both invocations in ~4 seconds total.

## REPR-02 Satisfaction

REPR-02 is satisfied: `rejection_abc` with the same seed produces deterministic output, enforced by a pytest test scoped per D-04 to single-process methods only. No audit document was produced per D-06.

## Known Stubs

None.

## Self-Check: PASSED

- `experiments/tests/test_seeding.py` modified — confirmed exists and contains `TestRunnerDeterminism`
- Commit 11aacf5 — confirmed
