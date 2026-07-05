---
phase: 02-mpi-hardening
plan: "03"
subsystem: testing
tags: [mpi, commworldmap, scaling-runner, migration, mpicommexecutor, barrier]

# Dependency graph
requires:
  - phase: 02-02
    provides: 48-rank CommWorldMap verification (PASS) unblocking this migration
  - phase: 02-01
    provides: MPI hardening test suite including test_barrier_placement_source_check
provides:
  - scaling_runner.py migrated from shared MPICommExecutor to per-call CommWorldMap (D-03)
  - test_barrier_placement_source_check updated to match new structural invariant
affects: [03-code-cleanup]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Safer per-call CommWorldMap migration: all ranks enter _run_workloads, each pyABC call internally creates/tears down its own CommWorldMap via run_method_distributed"
    - "mpi_map plumbing check: grep mpi_map through run_method_distributed before using outer-wrapper migration — absent here, so per-call path used"
    - "Post-mpi_methods-pass Barrier: trailing Barrier after if mpi_methods block ensures all-ranks sync before finalization"

key-files:
  created: []
  modified:
    - experiments/scripts/scaling_runner.py
    - experiments/tests/test_mpi_hardening.py

key-decisions:
  - "Per-call CommWorldMap migration path chosen (Case B / safer alternative): mpi_map not plumbed through run_method_distributed, so outer-wrapper path would deadlock — per-call path is correct"
  - "MPICommExecutor completely removed from scaling_runner.py: import, use_shared_executor variable, and the shared-executor branch inside _run_workloads all deleted"
  - "test_barrier_placement_source_check regex updated: old MPICommExecutor pattern replaced with post-mpi_methods-pass Barrier pattern (Case B) — test still passes"

patterns-established:
  - "Migration decision gate: grep mpi_map plumbing before choosing outer-wrapper vs per-call CommWorldMap path"
  - "Barrier placement after MPI methods block: if int(n_workers) > 1 + if MPI.COMM_WORLD.Get_size() > 1 guards the post-pass Barrier"

requirements-completed: [MPI-01]

# Metrics
duration: 35min
completed: 2026-04-13
---

# Phase 02 Plan 03: Scaling Runner Migration to CommWorldMap Summary

**scaling_runner.py migrated from shared MPICommExecutor to per-call CommWorldMap (D-03); MPICommExecutor removed entirely; test_barrier_placement_source_check updated for new structural invariant**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-04-13T20:55:00Z
- **Completed:** 2026-04-13T21:30:00Z
- **Tasks:** 3 (Task 0: branch decision; Task 1: migration; Task 2: test update)
- **Files modified:** 2

## Plan 02-02 Outcome Read at Task 0

Both `.planning/phases/02-mpi-hardening/02-02-SUMMARY.md` and `.plans/diagnose/commworldmap-48rank-verification.md` were read. Both documents state:

- **Final status: PASS**
- 48-rank cluster job on JUWELS ParaStationMPI completed without hang
- n_records > 0, world_size=48, SLURM exit code 0
- MPI-01 closed in 02-02

**Decision: proceed with Tasks 1 and 2.**

## Migration Path Selected

The plan required checking `mpi_map` plumbing through `run_method_distributed`:

```bash
grep -n "mpi_map" experiments/async_abc/inference/pyabc_wrapper.py \
                 experiments/async_abc/inference/abc_smc_baseline.py \
                 experiments/async_abc/inference/pyabc_sampler.py \
                 experiments/async_abc/utils/runner.py \
                 experiments/async_abc/inference/method_registry.py
```

Result: `mpi_map` exists only within `pyabc_sampler.py` (internal to `build_pyabc_sampler`) and is used locally in `pyabc_wrapper.py` and `abc_smc_baseline.py`. It is **NOT** plumbed through `run_method_distributed`.

**Selected path: Safer per-call CommWorldMap (primary migration path per plan)**

This means all ranks enter `_run_workloads(mpi_methods)` together, and each pyABC call internally creates its own CommWorldMap via `run_method_distributed -> run_pyabc_smc / run_abc_smc_baseline`.

## Exact Changes Made

### Change 1: Remove `mpi_executor` kwarg from `_run_workloads` (lines ~934-1001)

- Deleted: `mpi_executor=None` kwarg
- Deleted: docstring mentioning MPICommExecutor
- Deleted: `if mpi_executor is not None:` branch (the entire shared-executor path with MethodProgressReporter, `run_method()` call, `progress.start/finish()`)
- Deleted: `else:` wrapper around `run_method_distributed` call
- Added: new docstring referencing Phase 2 D-03
- Result: `_run_workloads` directly calls `run_method_distributed` unconditionally

### Change 2: Replace shared-executor call site (lines ~1045-1071)

- Deleted: `use_shared_executor = bool(mpi_methods) and int(n_workers) > 1`
- Deleted: `if use_shared_executor:` block with `from mpi4py import MPI`, `from mpi4py.futures import MPICommExecutor`, `with MPICommExecutor(...)` context manager
- Deleted: `elif mpi_methods:` branch
- Added: simplified `if mpi_methods: _run_workloads(mpi_methods)` (all ranks participate)
- Added: post-pass Barrier: `if int(n_workers) > 1: from mpi4py import MPI; if MPI.COMM_WORLD.Get_size() > 1: MPI.COMM_WORLD.Barrier()`

### MPICommExecutor removal confirmed

```
grep -n "MPICommExecutor|mpi_executor" experiments/scripts/scaling_runner.py
```
Output: 2 matches, both in comments/docstrings. Zero code uses.

## Task 2: test_barrier_placement_source_check Update

The old `scaling_pattern` regex:
```python
scaling_pattern = re.compile(
    r"with MPICommExecutor\(MPI\.COMM_WORLD, root=0\) as executor:"
    r"[\s\S]{0,400}?"
    r"if MPI\.COMM_WORLD\.Get_size\(\) > 1:\s*\n"
    r"\s*MPI\.COMM_WORLD\.Barrier\(\)",
)
```

Replaced with Case B (safer per-call path):
```python
scaling_pattern = re.compile(
    r"if mpi_methods:\s*\n"
    r"(?:[^\n]*\n){1,10}?"
    r"\s*if MPI\.COMM_WORLD\.Get_size\(\) > 1:\s*\n"
    r"\s*MPI\.COMM_WORLD\.Barrier\(\)",
    re.MULTILINE,
)
```

Test result: `test_barrier_placement_source_check PASSED` (1 passed in 0.09s)

## Accomplishments

- scaling_runner.py no longer uses MPICommExecutor: import removed, shared-executor branch deleted, `mpi_executor` kwarg gone
- All ranks now participate uniformly in `_run_workloads` for both MPI and non-MPI methods
- Per-call CommWorldMap inside run_pyabc_smc handles coordination — no inter-communicator cycles
- Post-mpi_methods Barrier ensures all-ranks synchronization before finalization/shard writes
- `test_barrier_placement_source_check` updated to match new structural invariant; test passes

## Task Commits

1. **Task 0: Read Plan 02-02 outcome** — No commit (read-only; PASS confirmed, proceed)
2. **Task 1: Migrate scaling_runner.py** — `c7cdb70` (feat)
3. **Task 2: Update test_barrier_placement_source_check** — `a78853a` (fix)

**Plan metadata:** _(this commit)_ (docs: complete plan)

## Files Created/Modified

- `experiments/scripts/scaling_runner.py` — Migrated from MPICommExecutor to per-call CommWorldMap; `_run_workloads` simplified to single code path; post-mpi_methods Barrier added
- `experiments/tests/test_mpi_hardening.py` — `test_barrier_placement_source_check` regex updated for Case B (post-mpi_methods-pass pattern); wrapper/baseline regex unchanged; test passes

## Decisions Made

- Per-call CommWorldMap path selected (not outer-wrapper) because `mpi_map` is not plumbed through `run_method_distributed` — the outer-wrapper approach would have caused a deadlock (inner CommWorldMap inside blocked workers)
- MPICommExecutor import completely removed — no fallback, no dual path
- `use_shared_executor` variable removed — it was the old gatekeeper for the Candidate 2 path, now unnecessary

## Deviations from Plan

None - plan executed exactly as written. The plan explicitly specified the safer per-call path as the primary migration when `mpi_map` plumbing is absent, which is the case here.

## Issues Encountered

None. Task 2 test file changes were staged but not committed in a prior session. This session committed them as `a78853a`.

## MPI-01 Final State

- **MPI-01 status: Complete**
- CommWorldMap is now the sole pyABC MPI coordination model in the codebase
- scaling_runner.py, pyabc_wrapper.py, and abc_smc_baseline.py all use CommWorldMap (per-call) for MPI coordination
- MPICommExecutor has been removed from the experiment codebase
- Verified at 48 ranks on JUWELS ParaStationMPI (Plan 02-02 PASS)

## Next Phase Readiness

- Phase 2 MPI Hardening complete: CommWorldMap unified across all pyABC MPI paths
- Phase 3 (Code Cleanup) unblocked: the MPI model is now stable and documented
- No open MPI coordination concerns; the dual-code-path risk (Risk 3) is eliminated

## Self-Check: PASSED

- [x] experiments/scripts/scaling_runner.py — FOUND
- [x] experiments/tests/test_mpi_hardening.py — FOUND
- [x] .planning/phases/02-mpi-hardening/02-03-SUMMARY.md — FOUND
- [x] c7cdb70 (Task 1 commit) — CONFIRMED
- [x] a78853a (Task 2 commit) — CONFIRMED
- [x] MPICommExecutor removed from scaling_runner.py (only 2 comment-only references remain)
- [x] MPI.COMM_WORLD.Barrier() present in scaling_runner.py (1 match)
- [x] test_barrier_placement_source_check PASSED (1 passed in 0.03s)

---
*Phase: 02-mpi-hardening*
*Completed: 2026-04-13*
