# MPI Sampler Evaluation: Phase 1 Diagnose Inventory

**Status:** Draft (Plan 01 sections complete; Plan 02 will append characterization, paper sensitivity, residual risks, and recommendation)
**Author:** Phase 1 Diagnose
**Date:** 2026-04-10
**Requirements addressed:** MPI-02 (sampler evaluation), MPI-04 (coordination point inventory)
**Method:** Static code analysis + bug history review (per CONTEXT.md D-05). No new mpirun runs.

## Scope

This document evaluates four candidate MPI sampler approaches for pyABC under our cluster constraints (ParaStation MPI, up to 48 ranks). Each candidate is fully characterized for its MPI coordination points so that Phase 2 can implement the chosen approach with full knowledge of the failure surface.

The four candidates (per CONTEXT.md D-01) are:
1. CommWorldMap (current default) — custom bcast/send/recv map, no inter-communicators
2. MappingSampler + shared MPICommExecutor (scaling runner) — single Create_intercomm/Disconnect cycle
3. ConcurrentFutureSampler + per-call MPICommExecutor (legacy opt-in) — known teardown hangs
4. MappingSampler + per-call MPICommExecutor.map (the "vanilla pyABC MPI" path the original mapping sampler implemented before CommWorldMap)

## Shared Rank Protocol (all_ranks mode)

All pyABC-backed inference methods run in `all_ranks` execution mode. The protocol enforced by `run_method_distributed` (`experiments/async_abc/utils/runner.py:750`) is:

1. All ranks: `run_method()` is called
2. Inside `run_method`, `pyabc_wrapper` / `abc_smc_baseline` dispatch by `mpi_sampler` (sampler-specific coordination — see per-candidate sections below)
3. All ranks: `allgather(error_payload)` (`experiments/async_abc/utils/runner.py:876`) — shared exit gate
4. Root: returns records; workers return `[]`

The `allgather` at `runner.py:876` is the shared exit gate every sampler path must reach without hanging.

**Exception — scaling runner shared executor path:** When `mpi_executor is not None` (`experiments/scripts/scaling_runner.py:964`), the runner calls `run_method` directly on root only, bypassing `run_method_distributed` entirely. Workers stay blocked inside `MPICommExecutor`'s server recv loop and never call `allgather`. Synchronization happens once via `COMM_WORLD.Barrier()` after the entire `with MPICommExecutor(...)` context exits (`scaling_runner.py:1067-1068`).

## Candidate 1: CommWorldMap (current default)

**Entry points:**
- `experiments/async_abc/inference/pyabc_wrapper.py:404` (`run_pyabc_smc` MPI dispatch)
- `experiments/async_abc/inference/abc_smc_baseline.py:389` (`run_abc_smc_baseline` MPI dispatch)

**Implementation:** `experiments/async_abc/inference/pyabc_sampler.py:48-168` (`CommWorldMap` class)

### Coordination Points

| # | Direction | Operation | Call site | Purpose |
|---|-----------|-----------|-----------|---------|
| 1 | root only | (no MPI) `__init__` sets `self.is_root = True` | `pyabc_sampler.py:48` | Construct map |
| 2 | root only | `COMM_WORLD.bcast(("map", fn), root=0)` | `pyabc_sampler.py:97` | Broadcast batch function to all workers |
| 3 | root only | `COMM_WORLD.send((idx, item), dest=worker, tag=0)` × N | `pyabc_sampler.py:106` | Distribute work items dynamically |
| 4 | root only | `COMM_WORLD.recv(source=ANY_SOURCE, tag=1, status=status)` × N | `pyabc_sampler.py:115` | Collect results in completion order |
| 5 | root only | `COMM_WORLD.send(SENTINEL, dest=worker, tag=0)` | `pyabc_sampler.py:124` | Drain on worker error |
| 6 | root only | `COMM_WORLD.recv(source=ANY_SOURCE, tag=1)` × remaining | `pyabc_sampler.py:126` | Collect remaining drained results |
| 7 | root only | `COMM_WORLD.bcast(("shutdown", None), root=0)` | `pyabc_sampler.py:148` | Signal worker_loop to exit (idempotent via `self._shutdown`) |
| 8 | worker only | `COMM_WORLD.bcast(None, root=0)` | `pyabc_sampler.py:153` | Wait for command; "shutdown" tag breaks loop |
| 9 | worker only | `COMM_WORLD.recv(source=0, tag=0)` | `pyabc_sampler.py:158` | Receive work item or sentinel |
| 10 | worker only | `COMM_WORLD.send((idx, result_or_error), dest=0, tag=1)` | `pyabc_sampler.py:165` and `:167` | Send result/error back to root |
| 11 | all ranks | `COMM_WORLD.Barrier()` | `pyabc_wrapper.py:413` and `abc_smc_baseline.py:399` | Sync after CommWorldMap usage; required before allgather |
| 12 | all ranks | `allgather(error_payload)` | `runner.py:876` | Shared exit gate for `all_ranks` mode |

**Inter-communicator usage:** NONE. CommWorldMap uses only `COMM_WORLD`, no `Create_intercomm`, no `Disconnect`. This is the explicit design goal motivated by ParaStation MPI's `Disconnect` fragility (see Apr 8 entry).

**Shutdown safety:** Both `pyabc_wrapper.py:404-410` and `abc_smc_baseline.py:391-395` wrap the root path in `try/finally`, guaranteeing `cmap.shutdown()` even on root exception. This was the Apr 8 fix.

### Bug History Evidence

- **Apr 8, 2026 (CommWorldMap replaces MPICommExecutor):** CommWorldMap was added to fix `MPICommExecutor.__exit__` → `Disconnect()` hangs at 16 and 48 ranks for non-scaling experiments. Replaces self-managed `MPICommExecutor` path in both `abc_smc_baseline.py` and `pyabc_wrapper.py` for the default `mapping` sampler. Source: `.plans/bug-fixes/previous-fixes.md` (entry "2026-04-08: CommWorldMap replaces MPICommExecutor for default mapping sampler").
- **Apr 8, 2026 (CommWorldMap hang on root exception):** Without `try/finally`, root exceptions skipped `shutdown()`, leaving workers blocked at `bcast`, then root blocked at `allgather` — double deadlock. Fixed by wrapping root path in `try/finally`. Source: `.plans/bug-fixes/previous-fixes.md` (entry "2026-04-08: CommWorldMap hang on root exception + pyABC NaN weight crash").

### Status on cluster

- Verified working at 16 ranks for `lotka_volterra`, `straggler`, `gaussian_mean`, `gandk`, `sbc` (post-Apr 8 fix).
- **Unverified at 48 ranks** (per STATE.md). The scaling runner does NOT use CommWorldMap — it uses Candidate 2 (shared MPICommExecutor). Whether `bcast`-based coordination is stable under ParaStation MPI at 48 ranks is an open question feeding into Plan 02's residual risk section.

***

## Candidate 2: MappingSampler + Shared MPICommExecutor (scaling runner)

**Entry point:** `experiments/scripts/scaling_runner.py:1062` — `with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:` wraps all MPI workloads for one `n_workers` value.

**Implementation:** Standard pyABC `MappingSampler` with the executor's `executor.map` callable. The shared executor lives across all k-values and replicates within one `n_workers` configuration.

### Coordination Points

| # | Direction | Operation | Call site | Purpose |
|---|-----------|-----------|-----------|---------|
| 1 | all ranks | `MPICommExecutor.__enter__` → `Create_intercomm` on COMM_WORLD | `scaling_runner.py:1062` (mpi4py internals) | Create MPIPoolExecutor; one call per `n_workers` config |
| 2 | root only | `executor.map(fn, items)` per workload | inside `_run_workloads` (called from `scaling_runner.py:1064`) | Dispatch work to MPIPoolExecutor server loop |
| 3 | worker only | (blocking) MPIPoolExecutor server recv loop | mpi4py internals | Workers receive tasks via inter-communicator until context exits |
| 4 | root only | `MPICommExecutor.__exit__` → `executor.shutdown(wait=True)` + `Disconnect()` on inter-communicator | `scaling_runner.py:1062` (mpi4py internals) | One teardown total |
| 5 | all ranks | `COMM_WORLD.Barrier()` | `scaling_runner.py:1067-1068` | Post-context sync after `with` exits |
| 6 | all ranks | `allgather(error_payload)` per non-MPI workload | `runner.py:876` | Used only by non-MPI methods running before the shared executor block (`scaling_runner.py:1052-1053`) |

**Inter-communicator usage:** ONE `Create_intercomm` + ONE `Disconnect` per `n_workers` value, irrespective of how many k-values × replicates × methods run inside.

**Allgather skew avoidance:** When `mpi_executor is not None`, the runner calls `run_method` directly on root only (`scaling_runner.py:964-990`), bypassing `run_method_distributed`. This prevents the deadlock where workers stuck in the MPIPoolExecutor server loop cannot reach `allgather`.

### Bug History Evidence

- **Apr 7, 2026 (Shared MPICommExecutor across scaling workloads):** Each per-call `MPICommExecutor` triggered a full `Create_intercomm`/`Disconnect` cycle, and the second cycle deadlocked at 48 ranks. Fix: open ONE `MPICommExecutor` per `n_workers` value, reused across all k-values and replicates. Source: `.plans/bug-fixes/previous-fixes.md` (entry "2026-04-07: Shared MPICommExecutor across scaling workloads (48-rank repeated teardown hang)").
- **Apr 7, 2026 (test12 follow-up):** Even with the shared executor, `_run_workloads` still called `run_method_distributed` which expected workers in `allgather` — but workers were trapped in the server recv loop. Fixed by calling `run_method` directly when `mpi_executor` is provided, bypassing `allgather`. Source: same entry, "Follow-up (test12)" paragraph.
- **Apr 8, 2026 (lotka_volterra single cycle hang):** Even a SINGLE `Create_intercomm`/`Disconnect` cycle hung in some experiments (`lotka_volterra` `pyabc_smc` at 48 ranks) — motivating the move to CommWorldMap for non-scaling jobs. The shared executor path "happens to work" for the scaling runner because the server-loop usage pattern differs from per-call usage. Source: `.plans/bug-fixes/previous-fixes.md` (entry "2026-04-08: CommWorldMap replaces MPICommExecutor for default mapping sampler" — first paragraph).

### Status on cluster

- Currently used by the scaling runner only.
- Survives at 48 ranks for the scaling workload pattern (one shared executor per `n_workers`).
- Inter-communicator usage remains a known fragility on ParaStation MPI; this candidate is "lucky to work" rather than "robust by design."

***

## Candidate 3: ConcurrentFutureSampler + Per-call MPICommExecutor (legacy opt-in)

**Entry points:**
- `experiments/async_abc/inference/pyabc_wrapper.py:375` (when `mpi_sampler == "concurrent_futures"`)
- `experiments/async_abc/inference/abc_smc_baseline.py:359` (same path in baseline)

**Implementation:** `pyabc.ConcurrentFutureSampler` wrapping `MPICommExecutor`, with a per-call `with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:` block. The sampler maintains a speculative future queue via `MPIPoolExecutor.submit()`.

**Status:** Opt-in only with explicit warning (`pyabc_sampler.py:296-302`). Documented as having known teardown hangs at 48 ranks.

### Coordination Points

| # | Direction | Operation | Call site | Purpose |
|---|-----------|-----------|-----------|---------|
| 1 | all ranks | `MPICommExecutor.__enter__` → `Create_intercomm` on COMM_WORLD | `pyabc_wrapper.py:375` / `abc_smc_baseline.py:359` (mpi4py internals) | New inter-communicator per `abc.run()` call |
| 2 | root only | `executor.submit(simulate_fn, ...)` × `client_max_jobs` | inside `ConcurrentFutureSampler.sample` | Speculative future queue (pyABC internals) |
| 3 | worker only | (blocking) MPIPoolExecutor server recv loop | mpi4py internals | Workers process futures until shutdown |
| 4 | root only | `MPICommExecutor.__exit__` → `executor.shutdown(wait=True)` + `Disconnect()` on inter-communicator | `pyabc_wrapper.py:375` / `abc_smc_baseline.py:359` (mpi4py internals) | Teardown — known to deadlock |
| 5 | all ranks | `COMM_WORLD.Barrier()` | `pyabc_wrapper.py:399-400` / `abc_smc_baseline.py:384-385` | Post-context sync |
| 6 | all ranks | `allgather(error_payload)` | `runner.py:876` | Shared exit gate (only reached if Disconnect succeeds) |

**Inter-communicator usage:** ONE `Create_intercomm` + ONE `Disconnect` PER `abc.run()` call. Multiple methods × multiple replicates = many cycles, each one a hang risk.

### Bug History Evidence (this path is the historical hang battleground)

- **Apr 5–6, 2026:** `tracker.drain()` blocked indefinitely waiting on the speculative queue when `client_max_jobs=200`. Fix: bound `client_max_jobs` to `n_workers`. Source: `.plans/bug-fixes/previous-fixes.md` (Apr 5–6 entries).
- **Apr 5–6, 2026:** Double shutdown — explicit `executor.shutdown()` + `MPICommExecutor.__exit__` both called the same pool shutdown, racing internally. Fix: single-owner shutdown via context manager only.
- **Apr 6–7, 2026:** Even with bounded `client_max_jobs` and single-owner shutdown, `MPICommExecutor.__exit__` still deadlocked at 48 ranks on ParaStation MPI. The `Disconnect()` call hangs non-deterministically.
- **Apr 7, 2026 (Restore mapping as default):** `MPICommExecutor.__exit__` reproducibly hung after `abc.run()` completed at 48 ranks. Rank 0 finished in ~4.7s but 47 workers never exited the context. Fix: changed default `pyabc_mpi_sampler` from `concurrent_futures` to `mapping`. `concurrent_futures` retained as opt-in with warning. Source: `.plans/bug-fixes/previous-fixes.md` (entry "2026-04-07: Restore mapping as default pyABC MPI sampler (48-rank teardown hang)").

### Warning text in source

The opt-in warning lives at `pyabc_sampler.py:296-302` in `resolve_pyabc_mpi_sampler` and explicitly documents this as a known failure mode at scale. Plan 02 verification reads the warning text.

### Status on cluster

- **Known broken at 48 ranks on ParaStation MPI.** No new sub-bugs to fix — the entire path is the documented failure mode.
- Retained for diagnostic comparison and to allow opt-in fallback if a smaller-scale workload prefers the speculative future model.

***

## Candidate 4: MappingSampler + Per-call MPICommExecutor.map (the original "vanilla pyABC MPI" path)

**Entry point:** NOT currently instantiated in the codebase as a distinct path. This is the configuration that pyABC's official MPI documentation describes: `MappingSampler(map_=executor.map)` where `executor` is a per-call `MPICommExecutor`.

**Interpretation note (per RESEARCH.md Open Questions #1):** CONTEXT.md D-01 describes Candidate 4 as "native pyABC MappingSampler with plain mpi4py comm — no custom map adapter." The literal "no custom map adapter" reading is impossible: pyABC's `MappingSampler` requires a callable `map(fn, iterable) -> list`, and mpi4py does not ship a standalone `map` function. The closest match to "what pyABC's docs show" is `MappingSampler(map_=MPICommExecutor.map)` with a per-call executor lifecycle. This is what the codebase used between Apr 7 (mapping restored as default) and Apr 8 (CommWorldMap replaced it).

### Coordination Points (hypothetical reconstruction of the pre-CommWorldMap path)

| # | Direction | Operation | Call site (historical) | Purpose |
|---|-----------|-----------|------------------------|---------|
| 1 | all ranks | `MPICommExecutor.__enter__` → `Create_intercomm` on COMM_WORLD | per `abc.run()` (was at `pyabc_wrapper.py` and `abc_smc_baseline.py` pre-Apr 8) | New inter-communicator per `abc.run()` call |
| 2 | root only | `executor.map(simulate_fn, items)` (blocking, in-order results) | pyABC `MappingSampler.sample` | Synchronous parallel map per generation |
| 3 | worker only | (blocking) MPIPoolExecutor server recv loop | mpi4py internals | Workers process map items until shutdown |
| 4 | root only | `MPICommExecutor.__exit__` → `executor.shutdown(wait=True)` + `Disconnect()` | per `abc.run()` | Teardown — same Disconnect that hangs |
| 5 | all ranks | `COMM_WORLD.Barrier()` | post-context (would be at the same lines as Candidate 3) | Post-context sync |
| 6 | all ranks | `allgather(error_payload)` | `runner.py:876` | Shared exit gate |

**Difference from Candidate 3:** No speculative queue. `executor.map` is blocking and in-order, so there is no `client_max_jobs` queue depth and no `tracker.drain()` race. This is the only meaningful sampler-side difference. The MPI teardown sequence (`Create_intercomm` / `Disconnect`) is identical.

**Difference from Candidate 2:** Per-call executor lifecycle (one cycle per `abc.run()`) instead of one shared executor per `n_workers` value. Candidate 4 is structurally identical to "Candidate 2 without the scaling runner's shared-executor optimization."

### Bug History Evidence

- **Apr 7, 2026 (Restore mapping as default):** This is the path that was made default on Apr 7 after `concurrent_futures` was demoted. It worked for first invocations but suffered the same `Disconnect()` teardown hang on the second `abc.run()` call at 48 ranks. Source: `.plans/bug-fixes/previous-fixes.md` (entry "2026-04-07: Shared MPICommExecutor across scaling workloads" — describes the second-baseline hang affecting both `mapping` and `concurrent_futures`).
- **Apr 8, 2026 (CommWorldMap replaces MPICommExecutor):** Even a SINGLE `Create_intercomm`/`Disconnect` cycle hung in some experiments at 48 ranks (`lotka_volterra`). This is precisely Candidate 4's failure mode and is the direct motivation for CommWorldMap. Source: `.plans/bug-fixes/previous-fixes.md` (entry "2026-04-08: CommWorldMap replaces MPICommExecutor for default mapping sampler" — first paragraph).

### Status on cluster

- **Known broken at 48 ranks on ParaStation MPI**, for the same reason as Candidate 3: `MPICommExecutor.__exit__` → `Disconnect()` hangs non-deterministically.
- Cited by the user as the "closest to standard pyABC usage" candidate, which makes its evaluation important for the MPI-02 "correctness vs. standard pyABC usage" criterion. Plan 02 will discuss whether CommWorldMap deviates from how pyABC's `MappingSampler` is intended to be used (it does not — `MappingSampler` accepts any `map` callable, and CommWorldMap.map satisfies that contract).

***

## What Plan 02 Appends

Plan 01 provides the per-candidate inventory (coordination points + bug history). Plan 02 appends:

1. **Characterization table** — cross-candidate comparison across: cluster stability, correctness vs. standard pyABC usage, paper result sensitivity, inter-communicator dependency, residual risk.
2. **Paper sensitivity assessment** — does sampler choice affect which particles are accepted? Does it change wall-time semantics? Is overhead significant relative to the 900s budget?
3. **Residual risk discussion** — CommWorldMap at 48 ranks (unverified); shared MPICommExecutor fragility; whether Candidates 3 and 4 should be formally deprecated or retained as opt-in fallbacks.
4. **Recommendation** — a clear single-candidate recommendation with rationale, per CONTEXT.md D-02.

Plan 02 must NOT restructure or rewrite the sections above. Append-only past this separator.

**Cross-reference:** Coordination points in this document map directly to the test stubs that Phase 2 will implement. Every coordination point marked "known to deadlock" is a candidate test stub target for Phase 2's MPI integration tests.

**Note on Candidate 4 line number citations:** Candidate 4 is a historical reconstruction — no current call site exists. All citations for Candidates 1, 2, and 3 point to currently-live lines in the codebase (verified against source as of 2026-04-10).

***

*End of Plan 01 content. Requirements MPI-02 and MPI-04 inventory sections are complete.*

