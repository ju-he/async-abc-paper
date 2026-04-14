---
phase: 03-code-cleanup
plan: 03
subsystem: testing
tags: [end-to-end, smoke-test, orchestrator, test-mode, mpi-single-process]
dependency_graph:
  requires:
    - phase: 03-01
      provides: clean inference layer — CommWorldMap sole MPI path; no dead concurrent_futures branches
    - phase: 03-02
      provides: CommWorldMap coordination model and rank protocol documented inline
  provides:
    - All 11 EXPERIMENT_REGISTRY runners verified to exit 0 under --test (single-process orchestrator invocation)
    - TEST-02 satisfied — one-command smoke test works end-to-end
  affects: [Phase 4 — Reproducibility]
tech_stack:
  added: []
  patterns:
    - "Single-process --test mode exercises all 11 runners without mpirun — world_size=1, rank=0 throughout"
key_files:
  created: []
  modified: []
key_decisions:
  - "No source changes required — all 11 runners passed baseline without fixes (Task 2 was a no-op)"
  - "Pitfall 6 (straggler world_size=1 ValueError) confirmed to be a false alarm — the existing world_size<=1 guard in _resolve_effective_straggler_worker_id returns '0' before any offset arithmetic, so the ValueError is not reachable at world_size=1"
  - "--test mode exercises all 11 runners in single-process mode; true MPI parallelism is covered by experiments/tests/test_mpi_hardening.py and mpi_integration_helper.py, not by TEST-02"
requirements_completed: [TEST-02]
metrics:
  duration: "48m 54s"
  completed: "2026-04-14"
  tasks_completed: 3
  files_modified: 0
---

# Phase 03 Plan 03: End-to-end Orchestrator Smoke under --test Summary

**All 11 EXPERIMENT_REGISTRY runners pass `--test` in a single-process orchestrator invocation (exit 0, no source changes required); TEST-02 satisfied.**

## Performance

- **Duration:** 48m 54s (orchestrator wall time for all 11 experiments in test mode)
- **Completed:** 2026-04-14
- **Tasks:** 3 (1 baseline run, 1 no-op fix task, 1 human-verify checkpoint)
- **Files modified:** 0

## Accomplishments

- Task 1 baseline run: all 11 experiments in EXPERIMENT_REGISTRY exited 0 with output files present — no failures to fix
- Task 2: confirmed as a no-op — all 11 runners passed without any source changes
- Task 3: human reviewer ran the final orchestrator invocation, confirmed exit 0 and output files for all 11 experiments, and approved TEST-02
- Pitfall 6 (straggler ValueError at world_size=1) confirmed to be a false alarm — existing guard already handles this case

## Task Commits

No source-code commits were made in this plan (Task 2 was a no-op; no runners required fixes).

The final plan metadata commit covers this SUMMARY.md and updated STATE.md / ROADMAP.md.

## Baseline Run Results (Task 1)

Orchestrator invocation:
```
./nastjapy_copy/.venv/bin/python experiments/run_all_paper_experiments.py \
    --test --output-dir /tmp/test_paper_results_baseline
```
Log: `/tmp/run_all_baseline.log`

| Experiment           | Exit | Files | Failure |
|----------------------|------|-------|---------|
| gaussian_mean        | 0    | 83    | none    |
| gandk                | 0    | 87    | none    |
| lotka_volterra       | 0    | 80    | none    |
| cellular_potts       | 0    | 85    | none    |
| sbc                  | 0    | 29    | none    |
| straggler            | 0    | 34    | none    |
| runtime_heterogeneity| 0    | 27    | none    |
| scaling              | 0    | 213   | none    |
| sensitivity          | 0    | 10    | none    |
| sensitivity_gandk    | 0    | 10    | none    |
| ablation             | 0    | 24    | none    |

All 11 experiments: exit 0, at least one output file each, no unhandled exceptions.

## Task 2: No Fixes Required

All 11 runners passed the Task 1 baseline without modifications. Task 2 is a documented no-op per the plan's acceptance criteria: "all 11 runners passed baseline; no fixes required."

No entries were added to `.plans/bug-fixes/previous-fixes.md` (no bugs found).

## Final Orchestrator Invocation (Task 3 checkpoint)

Human reviewer executed:
```
./nastjapy_copy/.venv/bin/python experiments/run_all_paper_experiments.py \
    --test --output-dir /tmp/test_paper_results_baseline 2>&1 | tee /tmp/run_all_baseline.log
```
Exit code: **0**

Total wall time: **48m 54s**

Human checkpoint approval: **"approved — TEST-02 satisfied"** (2026-04-14)

## MPI vs Single-Process Coverage Note

`--test` mode exercises all 11 runners in single-process mode (world_size=1, rank=0). True MPI parallelism is covered by the MPI integration tests in `experiments/tests/test_mpi_hardening.py` and `mpi_integration_helper.py`, not by TEST-02. The single-process path exercises all runner logic, config parsing, pyABC invocation, and output writing — it does not exercise inter-rank coordination (CommWorldMap), which was verified in Phase 2 Plan 02 at 48 ranks on JUWELS ParaStationMPI.

## Requirements Satisfied

- **TEST-02**: All 11 EXPERIMENT_REGISTRY runners exit 0 under `--test` via the orchestrator, each writing at least one output file.

## Files Created/Modified

None — this plan required no source code changes.

## Decisions Made

- No fixes applied: Task 2 was a no-op because the Task 1 baseline was fully green.
- Pitfall 6 from RESEARCH.md was confirmed as a false alarm: `_resolve_effective_straggler_worker_id` in `straggler_runner.py` already has a `world_size <= 1` guard (lines 98-103 at time of execution) that returns `"0"` before any pyABC-method offset arithmetic, making the straggler runner safe at world_size=1.

## Deviations from Plan

None — plan executed exactly as written. Task 1 produced a clean baseline (all 11 pass), Task 2 became a no-op as the plan specified, and Task 3 was approved by the human reviewer.

## Issues Encountered

None.

## Next Phase Readiness

Phase 3 (Code Cleanup) is now complete:
- CODE-01 satisfied (Plan 03-01): dead MPICommExecutor/concurrent_futures paths removed
- CODE-02 satisfied (Plan 03-02): CommWorldMap coordination model documented inline
- TEST-02 satisfied (Plan 03-03): all 11 runners pass --test end-to-end

Phase 4 (Reproducibility) can begin: extend mode, seed determinism, and one-command end-to-end verification.

## Known Stubs

None — all 11 runners produce real output files in test mode.

---
*Phase: 03-code-cleanup*
*Completed: 2026-04-14*
