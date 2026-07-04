# Phase 2: MPI Hardening - Context

**Gathered:** 2026-04-10
**Status:** Ready for planning

<domain>
## Phase Boundary

CommWorldMap is the chosen pyABC MPI sampler (Phase 1 unhedged recommendation). This phase verifies it at 48 ranks, hardens wall-time stopping, migrates the scaling runner if verification passes, and writes the full MPI test suite. No re-evaluation of sampler choice — that decision is locked.

</domain>

<decisions>
## Implementation Decisions

### 48-Rank Verification (MPI-01)

- **D-01:** Verification approach is **both local mpirun first, then cluster**: write local mpirun tests covering coordination paths at 2 ranks (and 4 ranks for stress), then submit a minimal cluster job to confirm 48-rank stability. Both must pass for MPI-01 to be done.
- **D-02:** Cluster test job: `gaussian_mean`, population=100, 3 generations, 48 ranks on JUWELS via SLURM. This is the Phase 1 recommended test recipe (Risk 2 reproduction recipe).
- **D-03:** Scaling runner migration: **if 48-rank cluster verification passes**, migrate the scaling runner from Candidate 2 (shared MPICommExecutor) to CommWorldMap. Unifies all pyABC paths, eliminates the dual-code-path Risk 3. If CommWorldMap fails the 48-rank test, scaling runner remains on Candidate 2.

### Worker Crash / Liveness (Risk 1)

- **D-04:** Worker crash handling: **rely on SLURM job timeout as outer safety net**. No liveness check, no MPI_Testsome polling, no watchdog. Document as known behavior: a worker crash means the job hangs until SLURM kills it. This is acceptable — worker crashes during ABC computation are rare, and SLURM is already the outer safety net on JUWELS. Risk 1 test stub from Phase 1 is **not implemented** (no liveness mechanism to test).

### Wall-Time Stopping Hardening (MPI-03)

- **D-05:** NaN weight catch guard stays **as-is** — no code changes to the existing `AssertionError` catch in `pyabc_wrapper.py:125-133` and `abc_smc_baseline.py:127-136`. The catch mechanism is sufficient; protection against future breakage comes from the regression test (D-06), not by broadening the guard.
- **D-06:** MPI-03 is satisfied by writing the Risk 4 regression test: run pyABC with `max_wall_time_s=0.1`, assert no exception escapes, returned records contain no NaN weights, log contains the wall-time stop warning. **Applies to both** `pyabc_smc` and `abc_smc_baseline` paths with the same pattern.

### Test Suite (TEST-01, TEST-03)

- **D-07:** All four test stubs from Phase 1 are implemented:
  1. **NaN weight guard regression** (Risk 4) — highest priority, protects the most fragile existing fix
  2. **CommWorldMap coordination test** — mpirun-based integration: normal run, root exception (try/finally path), multi-call; 2-rank default for CI
  3. **Barrier placement guard** (Risk 3) — **static source check** (not mpirun hang detection): pytest reads source and asserts Barrier() call sites exist at `pyabc_wrapper.py`, `abc_smc_baseline.py`, and `scaling_runner.py` expected lines. Fast, no flakiness, CI-safe.
  4. **Double-shutdown regression** (TEST-03) — assert CommWorldMap.shutdown() called twice does not hang or raise

- **D-08:** mpirun-based tests use **2 ranks for CI-facing tests, 4 ranks for stress variants** (marked `slow`/manual). Pattern follows existing `mpi_integration_helper.py` subprocess approach.

### Claude's Discretion

- Exact line numbers for the static Barrier source check (verify against current source before pinning)
- Whether to add the CommWorldMap coordination test to `test_inference.py` (alongside existing MPI tests) or create `test_commworldmap.py`
- Structure of the 48-rank cluster job script (can reuse existing SLURM helpers in `experiments/jobs/`)
- Order of tasks within plans

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 1 Deliverable
- `.plans/diagnose/mpi-evaluation.md` — Full inventory, characterization table, residual risks (Risk 1–4), test stub descriptions, and the unhedged CommWorldMap recommendation. **Primary reference for Phase 2.** Read the Residual Risks and Recommendation sections before planning.

### Bug History
- `.plans/bug-fixes/previous-fixes.md` — All MPI hang fixes Apr 4–8 2026; the evidence base for why CommWorldMap was chosen and what coordination paths have failed before

### MPI Coordination Code (primary targets)
- `experiments/async_abc/inference/pyabc_sampler.py` — CommWorldMap class, worker_loop(), shutdown(), build_pyabc_sampler(), resolve_pyabc_mpi_sampler()
- `experiments/async_abc/inference/pyabc_wrapper.py` — CommWorldMap usage (lines ~113–140, NaN catch at ~125–133, Barrier at ~413)
- `experiments/async_abc/inference/abc_smc_baseline.py` — CommWorldMap usage (same structure, NaN catch at ~127–136, Barrier at ~399)
- `experiments/scripts/scaling_runner.py` — Candidate 2 (shared MPICommExecutor) path; Barrier at ~1067–1068; migration target if D-03 applies

### Existing Test Infrastructure
- `experiments/tests/mpi_integration_helper.py` — subprocess helper pattern to follow for new mpirun tests
- `experiments/tests/mpi_abc_smc_baseline_helper.py` — same, for abc_smc_baseline path
- `experiments/tests/test_inference.py` — existing MPI-related test class (search for `test_mpi_backend_via_mpirun`, `test_abc_smc_baseline_shutdown_does_not_hang`)

### Requirements
- `.planning/REQUIREMENTS.md` §MPI-01, §MPI-03, §TEST-01, §TEST-03 — requirements this phase satisfies

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CommWorldMap` (`pyabc_sampler.py`) — fully implemented; no code changes needed to the core class
- `mpi_integration_helper.py` and `mpi_abc_smc_baseline_helper.py` — established subprocess mpirun test pattern; new tests should follow the same helper structure
- Existing `test_mpi_backend_via_mpirun` and `test_abc_smc_baseline_shutdown_does_not_hang` in `test_inference.py` — reference implementations for how mpirun subprocess tests are organized
- `experiments/jobs/` — SLURM job scripts; reuse for the 48-rank verification cluster job

### Established Patterns
- mpirun tests run as subprocess calls from pytest, output collected as JSON; assertion logic in the test, execution logic in the helper file
- CommWorldMap usage pattern in both `pyabc_wrapper.py` and `abc_smc_baseline.py` is identical: `cmap.is_root` → `cmap.map`, `finally: cmap.shutdown()`, workers in `cmap.worker_loop()`, followed by `COMM_WORLD.Barrier()` then allgather
- Tests that require mpirun are marked with a custom marker or placed in classes that skip when mpi4py unavailable

### Integration Points
- Scaling runner migration (D-03): `scaling_runner.py` creates a `MPICommExecutor` per `n_workers` value and passes it as `mpi_executor` kwarg through `run_method`/`run_method_distributed` — the CommWorldMap path does NOT use this kwarg, so migration requires removing the executor path and letting CommWorldMap handle coordination directly

</code_context>

<specifics>
## Specific Ideas

- The 48-rank cluster test is a new one-off SLURM script — not part of the regular test suite. It's a manual verification step that sets MPI-01 to done.
- Static Barrier source check should grep for the actual `COMM_WORLD.Barrier()` pattern at the expected call sites, not just check that `Barrier` appears anywhere in the file. Line numbers should be verified against current source before the test is written.
- If scaling runner migration (D-03) happens, the `mpi_executor` kwarg threading through `run_method_distributed` may become dead code — Phase 3 cleanup scope.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-mpi-hardening*
*Context gathered: 2026-04-10*
