# Phase 3: Code Cleanup - Research

**Researched:** 2026-04-14
**Domain:** Python refactoring, MPI inference layer dead-code removal, inline documentation, end-to-end test validation
**Confidence:** HIGH — all findings are based on direct code inspection of the actual source files

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Dead path removal (CODE-01, CODE-03)**

- **D-01:** Remove `concurrent_futures` and `concurrent_futures_legacy` branches from `pyabc_sampler.py:resolve_pyabc_mpi_sampler()`. No experiment config references these options — they are dead. The function should only handle the `"mapping"` / CommWorldMap path after cleanup.
- **D-02:** Remove the `mpi_executor=None` parameter and its associated branch from `pyabc_smc()` in `pyabc_wrapper.py` and `abc_smc_baseline()` in `abc_smc_baseline.py`. The scaling_runner was fully migrated to CommWorldMap in Phase 2 — no caller passes `mpi_executor` anymore. The `# Shared MPICommExecutor path` comment blocks at pyabc_wrapper.py:366-368 and abc_smc_baseline.py:349-353 are dead and should go.
- **D-03:** Remove the `concurrent_futures` branch in `build_pyabc_sampler()` / `pyabc_smc()` in `pyabc_wrapper.py` (lines ~370-382) and the equivalent branch in `abc_smc_baseline.py` (lines ~354-387). Both still import and use `MPICommExecutor` in this branch — these are the last live references to MPICommExecutor in the codebase.
- **D-04:** After removal, `resolve_pyabc_mpi_sampler()` in `pyabc_sampler.py` should validate only valid remaining options. Error message for invalid values should reflect the reduced option set.
- **D-05:** `scaling_runner.py` already fully migrated in Phase 2 — only comment-level references remain (lines ~940, ~1028). Update comments to remove stale MPICommExecutor mentions, but no functional changes needed.

**CommWorldMap inline documentation (CODE-02)**

- **D-06:** Add a class-level docstring to `CommWorldMap` in `pyabc_sampler.py` describing: (a) the coordination model (root dispatches via bcast/map, workers spin in worker_loop), (b) the rank protocol sequence (bcast task → workers compute → root collects → Barrier → allgather), (c) known failure modes (worker crash → job hangs until SLURM timeout; no liveness check by design, documented in D-04 from Phase 2 context). The full evaluation is in `.plans/diagnose/mpi-evaluation.md` — inline comments should summarize, not duplicate.
- **D-07:** Key coordination methods (`worker_loop`, `shutdown`, `map`) should have docstrings or inline comments explaining their role in the rank protocol. Brief is fine — the class docstring covers the full picture.

**TEST-02: single-command end-to-end test**

- **D-08:** TEST-02 is satisfied when `python experiments/run_all_paper_experiments.py --test --output-dir /tmp/test_paper_results` passes for all 11 experiments in `EXPERIMENT_REGISTRY` (gaussian_mean, gandk, lotka_volterra, realistic_workload, sbc, straggler, runtime_heterogeneity, scaling, sensitivity, sensitivity_gandk, ablation). No new pytest wrapper needed — the existing orchestrator script is the single command.
- **D-09:** "Pass" means: each runner exits 0, output files are written, no unhandled exceptions. If a runner currently fails in --test mode, fix the failure. If a runner requires mpirun and `--test` doesn't skip MPI, the planner should verify whether scaling/straggler need special handling (e.g., `mpirun -n 2` invocation from the orchestrator or a test-only bypass).

### Claude's Discretion

- Exact refactoring order across the three inference files
- Whether to split D-01 through D-05 into one plan or multiple
- How to handle `TrackedFutureExecutor` import in `pyabc_wrapper.py` if it becomes unused after the concurrent_futures branch is removed
- Minor docstring wording for CODE-02

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CODE-01 | pyabc_sampler.py, abc_smc_baseline.py, pyabc_wrapper.py simplified after multiple patch rounds | Direct code audit confirms dead branches are identifiable and removable without callers |
| CODE-02 | MPI coordination model documented inline (CommWorldMap design, rank protocol, known failure modes) | mpi-evaluation.md and bug history provide full content for inline docs |
| CODE-03 | Dead/legacy code removed (concurrent_futures_legacy paths, obsolete workarounds) | All MPICommExecutor import sites and dead branches identified by code inspection |
| TEST-02 | All experiment runners pass `--test` end-to-end in a single command | Orchestrator architecture and test-mode clamping understood; MPI concern for scaling/straggler documented |
</phase_requirements>

---

## Summary

Phase 3 is a focused refactoring and documentation pass over three inference files plus a single-command test validation. Phase 2 completed the functional migration — CommWorldMap is the sole pyABC MPI coordination model in the codebase. Phase 3 removes the now-dead `concurrent_futures` / `MPICommExecutor` branches that remain in `pyabc_wrapper.py` and `abc_smc_baseline.py`, removes the dead `mpi_executor` parameter threading from both function signatures, cleans up stale comments in `scaling_runner.py`, and adds inline documentation to `CommWorldMap`.

The dead code is well-bounded and safe to remove. No external callers pass `mpi_executor` after Phase 2's migration. The `concurrent_futures` branch in both wrapper files is guarded by `if mpi_sampler == "concurrent_futures"`, which requires an explicit config opt-in — no experiment config sets this. `resolve_pyabc_mpi_sampler()` still returns `"concurrent_futures"` for that opt-in config value, so after removing the branches the error path in `resolve_pyabc_mpi_sampler()` and `build_pyabc_sampler()` should reflect the reduced valid option set.

TEST-02 validation requires running the orchestrator without `mpirun` — the test-mode worker cap clamps `n_workers` to 8 locally and 48 on cluster via `get_test_mode_max_workers()`. On a local machine with a single process, `parallel_backend` will be `"mpi"` for straggler/scaling configs (which have `n_workers: 16`), but the CommWorldMap will fall back to sequential execution when `comm.Get_size() == 1`. This means the orchestrator should pass locally without `mpirun`, but the MPI code paths will run in single-process mode — functional but not testing real parallelism.

**Primary recommendation:** Remove dead branches in a single plan covering all three inference files (D-01 through D-05), then document CommWorldMap (D-06, D-07) in a second plan, then verify TEST-02 end-to-end (D-08, D-09) in a third plan. This separation keeps concerns clean and makes each plan's diff reviewable.

---

## Standard Stack

No new libraries are introduced. This phase modifies existing Python source files using the same stack as Phase 2.

### Core (already installed)
| Library | Purpose | Notes |
|---------|---------|-------|
| mpi4py | MPI coordination | Only import to remove: `from mpi4py.futures import MPICommExecutor` in dead branches |
| pyabc | ABC-SMC inference | `pyabc.ConcurrentFutureSampler` referenced in dead branches; `pyabc.MappingSampler` stays |
| Python stdlib | concurrent.futures, typing | `TrackedFutureExecutor` in pyabc_sampler.py — check if unused after dead-code removal |

### Dependencies to verify post-removal
| Symbol | File | Used by dead branch? | Action |
|--------|------|---------------------|--------|
| `TrackedFutureExecutor` | `pyabc_sampler.py` | Defined there; imported in `pyabc_wrapper.py:19` and `abc_smc_baseline.py:22` | Remove import if class removed; remove class only if no non-dead usage remains |
| `from concurrent.futures import wait as _wait` | `pyabc_sampler.py:11` | Used inside `TrackedFutureExecutor.wait_for_pending` | Remove if `TrackedFutureExecutor` is removed |
| `from mpi4py.futures import MPICommExecutor` | `pyabc_wrapper.py:374` and `abc_smc_baseline.py:359` | Both are inside the `concurrent_futures` dead branch | Remove with branch |

**TrackedFutureExecutor decision (Claude's discretion):** `TrackedFutureExecutor` is used only inside the `concurrent_futures` branch in both wrapper files. After removing those branches, the class has no non-dead callers. Remove the class from `pyabc_sampler.py` and its imports from `pyabc_wrapper.py` and `abc_smc_baseline.py`. Also remove `from concurrent.futures import wait as _wait` from `pyabc_sampler.py:11`.

---

## Architecture Patterns

### Dead Code Topology

The dead code follows a consistent pattern in both `pyabc_wrapper.py` and `abc_smc_baseline.py`:

```
run_pyabc_smc() / run_abc_smc_baseline()
│
├── [DEAD] if mpi_executor is not None:          ← D-02 remove
│       return _run_with_map_callable(mpi_executor.map)
│
├── [DEAD] if mpi_sampler == "concurrent_futures":  ← D-03 remove
│       from mpi4py.futures import MPICommExecutor
│       with MPICommExecutor(...) as executor:
│           ...
│       Barrier()
│       return result
│
└── [KEEP] # Default mapping path: CommWorldMap
        cmap = CommWorldMap(MPI.COMM_WORLD)
        ...
```

And in `resolve_pyabc_mpi_sampler()` in `pyabc_sampler.py`:

```
resolve_pyabc_mpi_sampler()
│
├── [KEEP]  if configured == "mapping": return "mapping"
├── [DEAD]  if configured == "concurrent_futures": warn; return "concurrent_futures"  ← D-01
├── [DEAD]  if configured == "concurrent_futures_legacy": warn; return "concurrent_futures"  ← D-01
└── [KEEP]  raise ValueError(invalid option)  ← update error message (D-04)
```

And in `build_pyabc_sampler()`:

```
build_pyabc_sampler()
│
├── [KEEP] if mpi_sampler == "mapping": return MappingSampler(...)
├── [DEAD] if mpi_sampler == "concurrent_futures": return ConcurrentFutureSampler(...)  ← D-03
└── [KEEP] raise ValueError(unknown mpi_sampler)  ← update error message
```

### CommWorldMap Rank Protocol (for D-06/D-07 inline docs)

The coordination sequence every planner and implementer must understand:

```
Root (rank 0)                    Workers (rank 1..N-1)
─────────────────────────────    ──────────────────────────
cmap = CommWorldMap(COMM_WORLD)  cmap = CommWorldMap(COMM_WORLD)
if cmap.is_root:                 else:
  try:                             cmap.worker_loop()  ← blocks here:
    result = run_abc(...)            while True:
      # each map() call:              tag, payload = bcast(None, root=0)
      #   bcast(("map", fn))  ──►     fn = payload
      #   send(idx, item)×N   ──►     while True:
      #   recv(result)×N      ◄──       item = recv(source=0, tag=0)
      #   send(sentinel)×N    ──►       if sentinel: break
                                        result = fn(work)
                                        send((idx, result), dest=0, tag=1)
  finally:
    cmap.shutdown()
    #   bcast(("shutdown",None)) ──►  break
                                 # exits worker_loop
COMM_WORLD.Barrier()             COMM_WORLD.Barrier()
# allgather in run_method_dist   # allgather in run_method_dist
```

**Known failure modes (required for D-06):**
1. **Worker crash during map():** Root receives `_WorkerError` wrapper — drains remaining workers, re-raises. Workers that haven't been sent a sentinel yet may wait at `recv(source=0, tag=0)` until root sends drain sentinels. Covered by drain logic in `CommWorldMap.map()` lines ~119-128.
2. **Root exception before shutdown():** Handled by `try/finally` in both wrapper files — `cmap.shutdown()` always fires (Apr 8 fix). Without this, workers block at `bcast(None, root=0)` forever.
3. **Worker crash between map() calls (during worker_loop bcast wait):** No liveness check. If a worker dies between map batches, root's next `bcast(("map", fn))` will hang waiting for all ranks. This is a known residual risk (see `.plans/diagnose/mpi-evaluation.md` Risk section). Job eventually killed by SLURM timeout. By design — no heartbeat mechanism.
4. **Single-process mode:** `CommWorldMap.map()` detects `self.size <= 1` and falls back to sequential `[fn(item) for item in items]`. No MPI calls. Used locally when running without `mpirun`.

### run_all_paper_experiments.py Architecture

The orchestrator does NOT use subprocess. It uses `importlib` to load each runner module and call `module.main(argv)`. All 11 runners share the same Python process. This means:

- MPI is initialized once for the whole process (if `mpirun` is used)
- Without `mpirun`, `get_world_size()` returns 1, `get_rank()` returns 0
- `n_workers` from straggler/scaling configs will be clamped to 8 (local test mode) by the test-mode override in `load_config()`
- CommWorldMap's single-process fallback handles `n_workers=1` via `size <= 1` check

**Key consequence for TEST-02:** Running `python run_all_paper_experiments.py --test` without `mpirun` will exercise CommWorldMap in single-process fallback mode for straggler/scaling. This is functionally correct (each experiment produces output files). True MPI parallelism is not tested by the orchestrator without `mpirun`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Finding all MPICommExecutor usages | Manual grep | `grep -r MPICommExecutor experiments/` | Already done — 2 sites confirmed (dead branches only) |
| Finding mpi_executor call sites | Manual scan | `grep -r mpi_executor experiments/` | Already done — no callers post-Phase 2 |
| Test-mode worker clamping | New logic | Existing `get_test_mode_overrides()` in schema.py | Already clamps n_workers, max_simulations, n_replicates, n_generations, max_wall_time_s |

---

## Common Pitfalls

### Pitfall 1: Removing mpi_executor without verifying all call sites
**What goes wrong:** If any caller still passes `mpi_executor=` kwarg after removal, it becomes an unexpected kwarg error at runtime.
**Why it happens:** The parameter was threaded through `run_method_distributed` via `**kwargs` in the old shared-executor path.
**How to avoid:** Before removing the parameter, grep for all call sites: `grep -r "mpi_executor" experiments/`. As of Phase 2, scaling_runner.py is the only historical caller and was migrated. Verify the grep returns no functional call sites (only comments/docs).
**Warning signs:** `TypeError: run_pyabc_smc() got an unexpected keyword argument 'mpi_executor'` at runtime.

### Pitfall 2: Leaving TrackedFutureExecutor import after removing usage
**What goes wrong:** `from .pyabc_sampler import TrackedFutureExecutor` in `pyabc_wrapper.py:19` and `abc_smc_baseline.py:22` will cause an `ImportError` if `TrackedFutureExecutor` is removed from `pyabc_sampler.py`.
**Why it happens:** The import list in both wrapper files explicitly names `TrackedFutureExecutor`.
**How to avoid:** Remove the import from `pyabc_sampler.py`, `pyabc_wrapper.py`, and `abc_smc_baseline.py` together as one atomic change.

### Pitfall 3: Updating error message in resolve_pyabc_mpi_sampler() incompletely
**What goes wrong:** After removing `concurrent_futures` and `concurrent_futures_legacy` branches, the `raise ValueError` at the bottom still lists them as valid options in its message: `"Valid values: 'concurrent_futures', 'mapping', 'concurrent_futures_legacy'."` (current text at line 314-315).
**Why it happens:** The error message is a separate string, not auto-generated from the branch structure.
**How to avoid:** Update the `raise ValueError` message to list only `"mapping"` as valid. Same fix needed in `build_pyabc_sampler()`'s `raise ValueError`.

### Pitfall 4: Test for resolve_pyabc_mpi_sampler expects concurrent_futures to be valid
**What goes wrong:** `test_inference.py::TestBuildPyabcSampler::test_resolve_pyabc_mpi_sampler_honors_legacy_alias_with_warning` and `test_resolve_pyabc_mpi_sampler_honors_explicit_mapping` may test behavior of the `concurrent_futures` / `concurrent_futures_legacy` branches.
**Why it happens:** Tests were written when those branches existed and were opt-in valid paths.
**How to avoid:** After removing the branches, update or remove tests that assert `concurrent_futures` is accepted. A test asserting that `concurrent_futures` raises `ValueError` would be appropriate to add (documents the intentional removal).

### Pitfall 5: client_max_jobs logic in abc_smc_baseline.py calls resolve_pyabc_mpi_sampler twice
**What goes wrong:** `abc_smc_baseline.py` calls `resolve_pyabc_mpi_sampler()` twice — once at lines ~291-299 (for `client_max_jobs`) and again at lines ~300-304 (for `mpi_sampler`). After cleanup, verify the order is still correct and no duplication issue arises.
**Why it happens:** The `client_max_jobs` call uses the sampler string before `mpi_sampler` is bound to the local variable.
**How to avoid:** Bind `mpi_sampler` first, then call `resolve_pyabc_client_max_jobs(... mpi_sampler=mpi_sampler)`. This is a minor cleanup opportunity during D-02.

### Pitfall 6: TEST-02 straggler runner uses n_workers=16 but local test runs single-process
**What goes wrong:** The straggler runner uses `_resolve_effective_straggler_worker_id()` which calls `get_world_size()`. With world_size=1 and `straggler_rank=0`, the effective worker id is `"0"` for non-pyABC methods but `"1"` for pyABC methods. When world_size=1, pyABC methods compute `effective = slot + 1 = 1`, then check `if effective >= active_world_size (1)` — this raises `ValueError: out of range`.
**Why it happens:** The straggler runner has a straggler_rank-to-effective-worker-id mapping that adds 1 for pyABC methods to account for root rank. With world_size=1, there is no worker rank 1.
**How to avoid:** Investigate `_resolve_effective_straggler_worker_id()` in `straggler_runner.py` and determine if the single-process path needs a guard. This is likely D-09's "fix the failure" case.
**Warning signs:** `ValueError: Configured straggler_rank=0 is out of range for method abc_smc_baseline with world_size=1` during TEST-02 validation.

---

## Code Examples

### Current dead branch in pyabc_wrapper.py (lines 366-401) — REMOVE entirely

```python
# Source: experiments/async_abc/inference/pyabc_wrapper.py:366-401 (current state)

# Shared MPICommExecutor path: caller manages lifecycle (scaling runner).
if mpi_executor is not None:
    return _run_with_map_callable(mpi_executor.map)

if mpi_sampler == "concurrent_futures":
    # Legacy futures path: still uses MPICommExecutor (opt-in only).
    from mpi4py.futures import MPICommExecutor

    result: List[ParticleRecord] = []
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            tracker = TrackedFutureExecutor(executor)
            sampler = build_pyabc_sampler(...)
            result = _run_pyabc_smc_with_sampler(...)
    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.Barrier()
    return result

# Default mapping path: CommWorldMap avoids MPICommExecutor entirely.
cmap = CommWorldMap(MPI.COMM_WORLD)
...
```

After cleanup, only the CommWorldMap block remains. Same structure in `abc_smc_baseline.py:349-386`.

### resolve_pyabc_mpi_sampler() after D-01/D-04

```python
# Source: experiments/async_abc/inference/pyabc_sampler.py (post-cleanup)

def resolve_pyabc_mpi_sampler(inference_cfg, *, parallel_backend, method_name) -> str | None:
    if parallel_backend != "mpi":
        return None
    configured = inference_cfg.get("pyabc_mpi_sampler")
    if configured in (None, "", "mapping"):
        return "mapping"
    raise ValueError(
        f"Unknown pyabc_mpi_sampler={configured!r} for {method_name}. "
        "Valid values: 'mapping'."
    )
```

### CommWorldMap class docstring (D-06)

The existing class docstring (lines 49-65) already describes the usage pattern. It needs augmentation to cover:
- The full rank protocol sequence (bcast → send/recv loop → shutdown bcast → Barrier → allgather)
- Known failure modes (worker crash between map calls = SLURM timeout, no liveness check by design)
- Reference to `.plans/diagnose/mpi-evaluation.md` for full evaluation

The existing usage block in the docstring is correct and should be kept.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Shared MPICommExecutor in scaling_runner | Per-call CommWorldMap in run_pyabc_smc / run_abc_smc_baseline | Phase 2 (2026-04-13) | scaling_runner.py now has no `mpi_executor` code paths — only comment remnants |
| MPICommExecutor per-call for non-scaling | CommWorldMap (Apr 8 2026) | Phase 1/2 | MPICommExecutor entirely removed from non-scaling code |
| concurrent_futures opt-in path | Dead (no config references it) | Phase 2 completion | Ready for removal in Phase 3 |

**Deprecated/outdated in current codebase (to be removed):**
- `concurrent_futures` and `concurrent_futures_legacy` branches in `resolve_pyabc_mpi_sampler()`: no config uses these, branch returns the same `"concurrent_futures"` value that the dead execution branch checks — both sides dead together
- `mpi_executor` parameter in `run_pyabc_smc()` and `run_abc_smc_baseline()`: no post-Phase-2 caller passes this
- `TrackedFutureExecutor` class: only used inside the `concurrent_futures` branch in both wrappers
- `from mpi4py.futures import MPICommExecutor`: only appears inside dead branches
- `from concurrent.futures import wait as _wait` in pyabc_sampler.py: only used by TrackedFutureExecutor

---

## Open Questions

1. **Straggler runner single-process world_size=1 + straggler_rank=0 for pyABC methods**
   - What we know: `_resolve_effective_straggler_worker_id` adds +1 offset for pyABC methods, resulting in effective=1 which is out of range for world_size=1
   - What's unclear: Whether this was intentionally excluded from local --test runs, or is an existing latent bug
   - Recommendation: Run `python experiments/run_all_paper_experiments.py --test --output-dir /tmp/test` locally before planning TEST-02 work to verify which runners actually fail. The straggler runner failure may be pre-existing.

2. **`client_max_jobs` double-call in abc_smc_baseline.py**
   - What we know: `resolve_pyabc_mpi_sampler()` is called twice before the `mpi_executor` check — once for the `client_max_jobs` call and once to bind `mpi_sampler`. The double-call is harmless but redundant.
   - What's unclear: Whether consolidating to one call is worth the diff complexity in Phase 3
   - Recommendation: Consolidate (bind `mpi_sampler` first at lines ~300-304, pass it to `resolve_pyabc_client_max_jobs`) as part of the D-02 cleanup. Low risk, cleaner code.

---

## Environment Availability

| Dependency | Required By | Available | Notes |
|------------|------------|-----------|-------|
| sim_backend_venv/.venv Python | All tests | Yes | Per CLAUDE.md |
| mpi4py | Dead-code removal verification | Yes (installed in venv) | Only to confirm imports work after removal |
| pyabc | Dead-code removal verification | Yes (installed in venv) | Test suite exercises pyabc paths |
| mpirun | TEST-02 MPI parallelism | Not required for local --test | CommWorldMap single-process fallback used locally |

**Local TEST-02 note:** The orchestrator can run all 11 experiments locally without `mpirun`. CommWorldMap's `size <= 1` guard runs `[fn(item) for item in items]` sequentially for MPI paths. This is sufficient to verify D-08/D-09 "exits 0, output files written, no unhandled exceptions."

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing, in sim_backend_venv/.venv) |
| Config file | none (pytest auto-discovers experiments/tests/) |
| Quick run command | `sim_backend_venv/.venv/bin/python -m pytest experiments/tests/test_inference.py experiments/tests/test_mpi_hardening.py -x -q` |
| Full suite command | `sim_backend_venv/.venv/bin/python -m pytest experiments/tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CODE-01/CODE-03 | Dead branches removed, tests still pass | unit | `pytest experiments/tests/test_inference.py -x -q` | Yes |
| CODE-03 | `concurrent_futures` config value now raises ValueError | unit | `pytest experiments/tests/test_inference.py::TestBuildPyabcSampler -x -q` | Yes (tests need update) |
| CODE-02 | CommWorldMap docstring present and accurate | manual | review | N/A |
| TEST-02 | All 11 runners pass `--test` | smoke | `sim_backend_venv/.venv/bin/python experiments/run_all_paper_experiments.py --test --output-dir /tmp/test_paper_results` | Yes (orchestrator exists) |

### Sampling Rate
- **Per task commit:** `sim_backend_venv/.venv/bin/python -m pytest experiments/tests/test_inference.py experiments/tests/test_mpi_hardening.py -x -q`
- **Per wave merge:** Full test suite
- **Phase gate:** Full suite green + orchestrator smoke pass before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] Update `experiments/tests/test_inference.py::TestBuildPyabcSampler::test_resolve_pyabc_mpi_sampler_honors_legacy_alias_with_warning` — this test asserts the deprecated legacy alias is accepted with a warning; after removal it should assert a `ValueError` instead
- [ ] (Optional) Add `test_resolve_pyabc_mpi_sampler_concurrent_futures_raises` to document the removal

---

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `experiments/async_abc/inference/pyabc_sampler.py` — full file read, all dead and live branches identified
- Direct code inspection: `experiments/async_abc/inference/pyabc_wrapper.py` — full file read, dead branches at lines 366-401
- Direct code inspection: `experiments/async_abc/inference/abc_smc_baseline.py` — full file read, dead branches at lines 349-386
- Direct code inspection: `experiments/scripts/scaling_runner.py` — lines 920-1040 read, comment-only stale references at ~940, ~1028
- Direct code inspection: `experiments/run_all_paper_experiments.py` — full file read, no subprocess, uses importlib
- Direct code inspection: `experiments/async_abc/io/schema.py` — test-mode clamping logic, `LOCAL_TEST_MAX_WORKERS=8`
- Direct code inspection: `experiments/async_abc/utils/runner.py` — `run_method_distributed` execution modes
- Direct code inspection: `experiments/scripts/straggler_runner.py` — `_resolve_effective_straggler_worker_id` world_size=1 risk
- `.planning/phases/03-code-cleanup/03-CONTEXT.md` — locked decisions D-01 through D-09
- `.plans/diagnose/mpi-evaluation.md` (partial read) — coordination points, rank protocol, residual risks
- `.plans/bug-fixes/previous-fixes.md` — full bug history for context on why MPICommExecutor was removed

### Secondary (HIGH confidence — live test run)
- `pytest experiments/tests/test_inference.py experiments/tests/test_mpi_hardening.py` — 103 passed, 15 skipped — confirms current test baseline before any changes

---

## Metadata

**Confidence breakdown:**
- Dead code identification: HIGH — all branches read directly, no ambiguity
- CommWorldMap documentation content: HIGH — mpi-evaluation.md and bug history provide all needed content
- TEST-02 local feasibility: HIGH — CommWorldMap single-process fallback confirmed in code
- Straggler/world_size=1 risk: MEDIUM — not yet verified by running the orchestrator (open question 1)
- Test update requirements: HIGH — legacy alias test confirmed by pytest --collect-only output

**Research date:** 2026-04-14
**Valid until:** Phase 3 completion (stable codebase, no external dependencies)
