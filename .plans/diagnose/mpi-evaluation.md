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
