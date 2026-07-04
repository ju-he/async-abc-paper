# Phase 3: Code Cleanup - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Simplify, document, and de-dead-code the MPI inference layer now that CommWorldMap is the sole coordination model and fully verified at 48 ranks. The concurrent_futures / MPICommExecutor paths are dead and must be removed. CommWorldMap coordination protocol must be documented inline. All experiment runners in the orchestrator must pass `--test`. No new features, no new experiments.

</domain>

<decisions>
## Implementation Decisions

### Dead path removal (CODE-01, CODE-03)

- **D-01:** Remove `concurrent_futures` and `concurrent_futures_legacy` branches from `pyabc_sampler.py:resolve_pyabc_mpi_sampler()`. No experiment config references these options — they are dead. The function should only handle the `"mapping"` / CommWorldMap path after cleanup.
- **D-02:** Remove the `mpi_executor=None` parameter and its associated branch from `pyabc_smc()` in `pyabc_wrapper.py` and `abc_smc_baseline()` in `abc_smc_baseline.py`. The scaling_runner was fully migrated to CommWorldMap in Phase 2 — no caller passes `mpi_executor` anymore. The `# Shared MPICommExecutor path` comment blocks at pyabc_wrapper.py:366-368 and abc_smc_baseline.py:349-353 are dead and should go.
- **D-03:** Remove the `concurrent_futures` branch in `build_pyabc_sampler()` / `pyabc_smc()` in `pyabc_wrapper.py` (lines ~370-382) and the equivalent branch in `abc_smc_baseline.py` (lines ~354-387). Both still import and use `MPICommExecutor` in this branch — these are the last live references to MPICommExecutor in the codebase.
- **D-04:** After removal, `resolve_pyabc_mpi_sampler()` in `pyabc_sampler.py` should validate only valid remaining options. Error message for invalid values should reflect the reduced option set.
- **D-05:** `scaling_runner.py` already fully migrated in Phase 2 — only comment-level references remain (lines ~940, ~1028). Update comments to remove stale MPICommExecutor mentions, but no functional changes needed.

### CommWorldMap inline documentation (CODE-02)

- **D-06:** Add a class-level docstring to `CommWorldMap` in `pyabc_sampler.py` describing: (a) the coordination model (root dispatches via bcast/map, workers spin in worker_loop), (b) the rank protocol sequence (bcast task → workers compute → root collects → Barrier → allgather), (c) known failure modes (worker crash → job hangs until SLURM timeout; no liveness check by design, documented in D-04 from Phase 2 context). The full evaluation is in `.plans/diagnose/mpi-evaluation.md` — inline comments should summarize, not duplicate.
- **D-07:** Key coordination methods (`worker_loop`, `shutdown`, `map`) should have docstrings or inline comments explaining their role in the rank protocol. Brief is fine — the class docstring covers the full picture.

### TEST-02: single-command end-to-end test

- **D-08:** TEST-02 is satisfied when `python experiments/run_all_paper_experiments.py --test --output-dir /tmp/test_paper_results` passes for all 11 experiments in `EXPERIMENT_REGISTRY` (gaussian_mean, gandk, lotka_volterra, realistic_workload, sbc, straggler, runtime_heterogeneity, scaling, sensitivity, sensitivity_gandk, ablation). No new pytest wrapper needed — the existing orchestrator script is the single command.
- **D-09:** "Pass" means: each runner exits 0, output files are written, no unhandled exceptions. If a runner currently fails in --test mode, fix the failure. If a runner requires mpirun and `--test` doesn't skip MPI, the planner should verify whether scaling/straggler need special handling (e.g., `mpirun -n 2` invocation from the orchestrator or a test-only bypass).

### Claude's Discretion

- Exact refactoring order across the three inference files
- Whether to split D-01 through D-05 into one plan or multiple
- How to handle `TrackedFutureExecutor` import in `pyabc_wrapper.py` if it becomes unused after the concurrent_futures branch is removed
- Minor docstring wording for CODE-02

</decisions>

<specifics>
## Specific Ideas

- Phase 2 context explicitly flagged: "if scaling runner migration (D-03) happens, the `mpi_executor` kwarg threading through `run_method_distributed` may become dead code — Phase 3 cleanup scope." Migration happened — remove it.
- The mpi-evaluation.md in `.plans/diagnose/` is the authoritative reference for the CommWorldMap design rationale — inline docs should point readers there for the full picture rather than duplicating.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### MPI coordination model (primary reference)
- `.plans/diagnose/mpi-evaluation.md` — Full CommWorldMap evaluation, rank protocol, residual risks (Risk 1–4), known failure modes. **Read before writing any CommWorldMap documentation (D-06, D-07).**

### Bug history
- `.plans/bug-fixes/previous-fixes.md` — All MPI hang fixes Apr 4–8 2026; context for why MPICommExecutor was removed and what coordination paths have failed before. Read before touching inference files.

### Inference layer (primary targets for CODE-01, CODE-02, CODE-03)
- `experiments/async_abc/inference/pyabc_sampler.py` — CommWorldMap class, `resolve_pyabc_mpi_sampler()`, `build_pyabc_sampler()`. Dead paths: concurrent_futures/concurrent_futures_legacy branches (lines ~296-315, ~382-394).
- `experiments/async_abc/inference/pyabc_wrapper.py` — `pyabc_smc()` with `mpi_executor=None` param and concurrent_futures branch (~258, ~367-382), NaN catch (~125-133), Barrier (~413).
- `experiments/async_abc/inference/abc_smc_baseline.py` — `abc_smc_baseline()` with `mpi_executor=None` param and concurrent_futures branch (~349-387), NaN catch (~127-136), Barrier (~399).
- `experiments/scripts/scaling_runner.py` — Fully migrated; comment-only cleanup needed (~940, ~1028).

### End-to-end test (TEST-02)
- `experiments/run_all_paper_experiments.py` — The single-command orchestrator. All 11 EXPERIMENT_REGISTRY entries must pass `--test`. Read the EXPERIMENT_REGISTRY dict and `_run_experiment()` before planning TEST-02 work.
- `experiments/configs/` — Config files for each runner (e.g., scaling.json, ablation.json). May need `test_mode`-specific overrides if runners fail.

### Requirements
- `.planning/REQUIREMENTS.md` §CODE-01, §CODE-02, §CODE-03, §TEST-02 — requirements this phase satisfies

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `experiments/async_abc/utils/benchmark_runner.py` — Shared runner entry point used by gandk, lotka_volterra, gaussian_mean, realistic_workload. Already calls `make_arg_parser` which defines `--test` and `--small`. These runners have --test support via the shared utility, not their own argparse.
- `experiments/async_abc/utils/runner.py:make_arg_parser()` — Defines `--test` and `--small` flags shared across all runners.

### Established Patterns
- CommWorldMap usage pattern is identical in both `pyabc_wrapper.py` and `abc_smc_baseline.py`: `cmap.is_root` → `cmap.map`, `finally: cmap.shutdown()`, workers in `cmap.worker_loop()`, followed by `COMM_WORLD.Barrier()` then allgather. Cleanup should preserve this pattern.
- After removing concurrent_futures branch, check if `TrackedFutureExecutor` import in `pyabc_wrapper.py` becomes unused — remove if so.

### Integration Points
- After removing `mpi_executor` param from `pyabc_smc()` and `abc_smc_baseline()`, verify no call sites pass this kwarg (scaling_runner was the only one and was migrated in Phase 2).
- `resolve_pyabc_mpi_sampler()` return value flows into `build_pyabc_sampler()` and then into `pyabc_smc()`/`abc_smc_baseline()` — changes to valid option set ripple through this call chain.

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-code-cleanup*
*Context gathered: 2026-04-14*
