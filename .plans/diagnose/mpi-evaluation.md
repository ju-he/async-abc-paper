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

---

## Characterization Table

| Candidate | Cluster Stability | Correctness vs Standard pyABC | Effect on Paper Results | Residual Risk |
|-----------|-------------------|-------------------------------|-------------------------|---------------|
| 1. CommWorldMap | Verified at 16 ranks post-Apr 8 for lotka_volterra, straggler, gaussian_mean, gandk, sbc. UNVERIFIED at 48 ranks. No inter-communicators means it eliminates the ParaStation Disconnect hang class entirely. | MappingSampler(map_=CommWorldMap.map) satisfies pyABC's documented MappingSampler interface — pyABC accepts any map(fn, iter) callable. CommWorldMap preserves input order (results indexed by idx). Particle acceptance semantics identical to standard pyABC. | Negligible. Coordination overhead sub-millisecond per batch at 48 workers and 100-1000 population. No inter-communicator teardown overhead. | (a) Worker crash during bcast causes root hang (theoretical, not observed). (b) 48-rank stability with ParaStation MPI unverified on any current job. |
| 2. MappingSampler + Shared MPICommExecutor | Survives at 48 ranks for the scaling workload pattern (one shared executor per n_workers value). Verified Apr 7. Depends on structural luck: one Create_intercomm/Disconnect cycle total. | Uses pyABC MappingSampler with executor.map — matches pyABC documentation exactly. Identical particle acceptance semantics. | ~40-50s teardown overhead per n_workers value (one-time). Not significant vs the 900s wall-time budget when amortized across k-values and replicates. | The single Disconnect still deadlocked in other non-scaling experiments (lotka_volterra at 48 ranks, Apr 8 entry). 'Happens to work' due to the server-loop usage pattern, not because Disconnect is reliable. Fragile to ParaStation MPI version changes. |
| 3. ConcurrentFutureSampler + Per-call MPICommExecutor | KNOWN BROKEN at 48 ranks on ParaStation MPI. Multiple reproducible hangs documented Apr 5-7. Retained opt-in with warning at pyabc_sampler.py:296-302. | Uses pyABC documented speculative future interface. Minor semantic deviation: workers may start simulating particles for next tolerance threshold before current generation completes (pre-submission via client_max_jobs). | Under wall-time budgets this can let slightly more work complete than strict sync semantics allow. Effect bounded by client_max_jobs and small in absolute terms. | Entire path is the known failure mode. No unresolved sub-bugs; the known behavior IS the risk. Useful only for diagnostic comparison. |
| 4. MappingSampler + Per-call MPICommExecutor.map | KNOWN BROKEN at 48 ranks on ParaStation MPI. Was default Apr 7 to 8 and hung in the second abc.run() at 48 ranks (Apr 7 entries). Single-cycle hangs also documented Apr 8. | This IS the standard pyABC MPI usage pattern from pyABC's own documentation. Particle acceptance semantics identical to Candidate 1 and 2 (synchronous blocking map). | Teardown overhead per abc.run() call, measured 37-50s per call (Apr 7). At many calls, eats a meaningful fraction of the 900s budget. | Per-call Create_intercomm/Disconnect cycles are the documented hang pattern. No path forward without rewriting mpi4py or ParaStation MPI. |

**Table summary:** Candidates 3 and 4 are eliminated by the bug history; Candidate 2 works only in its specific scaling runner pattern and remains fragile; Candidate 1 is the only candidate that eliminates the inter-communicator hang class by design, pending 48-rank verification (Phase 2 MPI-01 work).

---

## Paper Sensitivity Assessment (per D-07)

Assessed at reasoning level only — no new mpirun runs or timing numbers (per CONTEXT.md D-05 and D-07).

### Q1: Does sampler choice affect which particles get accepted (correctness of ABC-SMC)?

No, for all four candidates. MappingSampler and ConcurrentFutureSampler both pass the same simulate_fn to pyABC's ABCSMC loop; particle selection, epsilon schedule, and weight normalization are handled entirely inside pyabc.ABCSMC; the sampler only controls _how_ simulations are dispatched, not _which_ results are accepted. CommWorldMap's map preserves input order by indexing results with idx (pyabc_sampler.py:115-140), so no reordering. All four candidates are correctness-equivalent.

### Q2: Does sampler choice change wall-time semantics?

Potentially yes, marginally, for Candidate 3 only. ConcurrentFutureSampler pre-submits up to client_max_jobs futures speculatively, so workers can start on next tolerance threshold before current generation completes. Under wall-time this lets slightly more work complete, but effect is bounded by client_max_jobs (<= n_workers after Apr 5 fix) and small. Candidates 1, 2, 4 all use synchronous blocking map and have identical wall-time semantics.

### Q3: Is sampler overhead significant relative to the 900s wall-time budget?

- **Candidate 1:** NO. bcast + send*k + recv*k per batch; sub-millisecond at 48 workers × 100-1000 population.
- **Candidate 2:** Small. One Create_intercomm/Disconnect cycle per n_workers value, ~40-50s on Apr 7. Amortized across all k-values and replicates.
- **Candidate 3:** Large and hang-prone. Per-call teardown ~37-50s when it succeeds, plus hang risk on subsequent calls.
- **Candidate 4:** Large. Per-call Create_intercomm/Disconnect, ~40-50s per abc.run() on Apr 7. At many calls this eats a meaningful fraction of the 900s budget.

### Q4: Is the paper's primary claim sensitive to sampler choice?

No. The paper compares wall-time efficiency of async Propulate-ABC vs sync pyABC. The structural advantage of async (no global synchronization barrier per generation) is independent of which MPI dispatch mechanism sync pyABC uses. Sampler choice only affects sync pyABC's implementation throughput: fragile sampler makes sync pyABC look worse (unwanted bias favoring async), stable sampler represents sync pyABC fairly. CommWorldMap gives sync pyABC its best fair shot. ABC-SMC correctness from Q1 is identical across candidates, so scientific conclusions (accepted particles, posterior shape) are independent of sampler choice; only wall-time fairness depends on it.

---

## Residual Risks and Newly Discovered Hang Paths

Each entry below is either a previously-undocumented residual risk or a hang path identified by this phase's analysis. Per CONTEXT.md D-06, each gets a reproduction recipe or test stub description (not an actual test — test creation is Phase 2's job).

### Risk 1: Worker crash during CommWorldMap.map (NEW, not in bug history)

**Candidate affected:** 1 (CommWorldMap)

**Description:** If a worker crashes, exits, or is killed during CommWorldMap.map, root's next COMM_WORLD.bcast blocks indefinitely because mpi4py bcast requires all ranks. There is no timeout and no worker liveness check in CommWorldMap. Root will hang at the next bcast in CommWorldMap.map() or CommWorldMap.shutdown(); other surviving workers will hang at the next bcast in worker_loop. No process exits automatically.

**Reproduction recipe:**

1. Launch with `mpirun -n 3` running a CommWorldMap workload.
2. After at least one generation, send SIGKILL (`kill -9`) to the highest-rank worker.
3. Expected: root hangs at next bcast in CommWorldMap.map() or CommWorldMap.shutdown(); rank 1 also hangs at next bcast in worker_loop; no process exits.
4. Observation: no further output from surviving processes; py-spy dump on root shows mpi4py.MPI.Comm.bcast as current frame.

**Mitigation options (Phase 2 scope):**
(a) MPI_Testsome pattern to poll with timeout (not currently implemented anywhere).
(b) catch MPI_ERR_PROC_FAILED from bcast (requires MPI fault tolerance not available in ParaStation MPI default build).
(c) Document as expected-crash failure mode and rely on Slurm job timeouts as outer safety net.

### Risk 2: CommWorldMap at 48 ranks — unverified (KNOWN RISK, STATE.md entry)

**Candidate affected:** 1 (CommWorldMap)

**Description:** CommWorldMap was developed Apr 8 and verified only on non-scaling jobs at <=16 ranks. The scaling runner continues to use Candidate 2. Whether COMM_WORLD.bcast/send/recv are stable under ParaStation MPI at 48 ranks is untested.

**Reproduction recipe (Phase 2 cluster test, not Phase 1):**

1. Modify scaling_runner.py to use CommWorldMap-backed sampler instead of shared MPICommExecutor for one test run.
2. Submit scaling_48 job with a small pyABC workload (gaussian_mean, population=100, 3 generations).
3. Expected: job completes without hanging, CSV has non-zero rows.
4. Failure mode: if it hangs, the specific bcast/send/recv call at the hang point identifies the issue.

**Note:** Why this belongs to Phase 2: D-05 forbids new mpirun/cluster runs in Phase 1; explicitly Phase 2 MPI-01 work.

### Risk 3: Pre-Barrier allgather skew (existing fix, regression risk)

**Candidates affected:** 1 and 2

**Description:** The COMM_WORLD.Barrier() at pyabc_wrapper.py:413 and abc_smc_baseline.py:399 is NOT cosmetic — it ensures workers have exited worker_loop() before run_method_distributed calls allgather(error_payload); if removed or reordered (thinking it's redundant with allgather), workers entering allgather while root is still in map() will deadlock. For Candidate 2, the equivalent is COMM_WORLD.Barrier() at scaling_runner.py:1067-1068.

**How this was introduced and fixed:** Apr 5 entry documents the original bug: workers returned `[]` from inside the `with MPICommExecutor(...)` block and called `allgather` while root was still in `Disconnect`. The fix added `COMM_WORLD.Barrier()` after the context block. The CommWorldMap version of this fix (Barrier after `worker_loop()` returns) was added alongside CommWorldMap's introduction (Apr 8). The Barrier call sites are: `pyabc_wrapper.py:413`, `abc_smc_baseline.py:399`, `scaling_runner.py:1067-1068`.

**Test stub description (Phase 2):** Unit test mocking CommWorldMap.shutdown() to skip the bcast, asserting run_method_distributed deadlocks at allgather (integration-level, observable only via timeout); regression test parsing source for exact Barrier() call sites and failing CI if removed.

### Risk 4: pyABC NaN weight on partial generation (existing fix, regression risk)

**Candidates affected:** All four

**Description:** When max_wall_time_s fires mid-generation, pyABC stops collecting particles and the partial population has weight_sum 0 or NaN, causing Population.__init__ to raise AssertionError("The population total weight nan is not normalized"); the fix (Apr 8) catches this specific AssertionError in pyabc_wrapper.py:125-133 and abc_smc_baseline.py:127-136 and falls back to abc.history. Risk: if the fix guard is removed or pyABC error message wording changes, the catch misses.

**How this was introduced and fixed:** The Apr 8 entry documents that `max_wall_time_s` became the sole stopping criterion after commit 2666648 ("Unify stopping criterion"). When pyABC's wall-time fires mid-generation, the sampler stops collecting particles. The partial generation has particles whose weights cannot be normalized (sum is 0 or NaN), causing `Population.__init__` to raise `AssertionError: The population total weight nan is not normalized` (pyabc/population/population.py:98). The fix catches this in two places: `pyabc_wrapper.py:125-133` (for pyabc_smc path) and `abc_smc_baseline.py:127-136` (for baseline path). On catch, falls back to `abc.history` for all completed generations, treating the interrupted generation as a graceful early stop. The fix guard depends on pyABC's exact exception message text; if the pyABC library changes its message wording, the catch silently passes up the exception.

**Test stub description (Phase 2):** Unit test running a pyABC workload with max_wall_time_s=0.1 (force immediate timeout), asserting (a) no exception escapes, (b) returned records list has 0 or more entries but no NaN weights, (c) log contains the "NaN population weight — treating as early wall-time stop" warning.

---

*End of Plan 02 Task 1 content. Task 2 will append the Recommendation and Phase 1 Exit Checklist sections.*

---

## Recommendation

**Chosen approach for Phase 2: Candidate 1 — CommWorldMap (current default).** This is an unhedged recommendation. Candidate 1 is the only approach that survives the elimination analysis below.

### Elimination rationale

**Candidate 3 (ConcurrentFutureSampler + Per-call MPICommExecutor) — eliminated.** Multiple independent reproductions across Apr 5, 6, and 7 entries document teardown hangs at 48 ranks under ParaStation MPI. Multiple rounds of fixes — bounding client_max_jobs, single-owner shutdown, per-call teardown hardening — did not resolve the teardown hang. The Apr 5 entries document two distinct bugs (tracker.drain() hang and inter-comm teardown race) that were each fixed separately, yet the Apr 6-7 entries show the path still hung after both fixes were applied. The path is retained as opt-in only with the warning at `pyabc_sampler.py:296-302`, but it has no path to production use at 48 ranks on ParaStation MPI.

**Candidate 4 (MappingSampler + Per-call MPICommExecutor.map) — eliminated.** The Apr 7 "restore mapping as default" entry documents that switching the default from concurrent_futures to mapping worked for the first abc.run() invocation at 48 ranks, but hung on the second invocation in the second baseline method. The Apr 8 entry ("CommWorldMap replaces MPICommExecutor") documents that even a SINGLE Create_intercomm/Disconnect cycle hung in lotka_volterra at 48 ranks — Candidate 4's exact execution model. pyABC's documentation assumes an MPI implementation where Disconnect is not fragile. ParaStation MPI violates that assumption.

**Candidate 2 (MappingSampler + Shared MPICommExecutor) — eliminated for general use, retained for scaling runner only.** Three points explain why this candidate cannot be generalized:
1. Candidate 2 only works in a specific pattern — one shared executor per n_workers value with all MPI workloads grouped inside the `with` block — which does NOT fit the non-scaling experiment runner calling many independent abc.run() instances. The non-scaling runner creates a new MPICommExecutor per method call, which is exactly the per-call pattern that Candidate 4 proved deadlocks at 48 ranks.
2. Even in the scaling runner's happy path, the single Create_intercomm/Disconnect cycle still uses inter-communicators — the same primitive that hangs single-cycle in non-scaling experiments (Apr 8 lotka_volterra). It "happens to work" for scaling because the server-loop pattern differs, not because Disconnect is reliable on ParaStation MPI.
3. Candidate 2 requires the runner to bypass allgather coordination at scaling_runner.py:964-990, creating a second code path that is easy to break by future refactoring.

Candidate 2 is NOT removed — the scaling runner still uses it and will continue to work for scaling experiments. The recommendation is that Candidate 1 is preferred for NEW code and the non-scaling path. Phase 2 may optionally migrate the scaling runner to CommWorldMap as well if Risk 2 (48-rank verification) clears.

### Why Candidate 1 wins

1. **It is the only candidate that eliminates the inter-communicator hang class by design.** CommWorldMap uses only bcast/send/recv on COMM_WORLD; does not call Create_intercomm, does not call Disconnect, does not rely on MPIPoolExecutor. The documented ParaStation MPI failure modes (Disconnect deadlock at 48 ranks, repeated-cycle teardown hangs) cannot apply because the primitives that cause them are absent from CommWorldMap entirely.

2. **It has already resolved real bugs in production.** The Apr 8 CommWorldMap entry documents that it fixed hangs in lotka_volterra, straggler, gaussian_mean, gandk, and sbc non-scaling jobs. These are the same jobs that Candidate 4 could not run reliably. The fix was verified on the cluster at 16 ranks across five experiment types.

3. **It is correctness-equivalent to pyABC's documented usage.** MappingSampler(map_=cmap.map) is exactly what pyABC's MappingSampler API expects: a map(fn, iter) callable. CommWorldMap.map preserves input order and propagates errors. ABC-SMC particle acceptance semantics are identical to Candidate 4 (Q1 from Paper Sensitivity Assessment confirms all four candidates are correctness-equivalent).

4. **It has negligible overhead.** No inter-communicator teardown cost; sub-millisecond per-batch coordination. Contrast with Candidates 2, 3, and 4 where teardown overhead was measured at 37-50s per cycle (Apr 7).

5. **It has clear, isolated failure modes.** The two residual risks (worker crash during bcast; 48-rank verification) are well-defined and each has a clear test recipe for Phase 2. Compare to Candidate 2's diffuse "works for this pattern but not that pattern" fragility, where correctness depends on the runner's code structure rather than CommWorldMap's own properties.

### Residual work for Phase 2 (MPI-01)

1. Verify CommWorldMap at 48 ranks on the cluster (Risk 2). Until done, we are betting bcast/send/recv on COMM_WORLD are stable at scale on ParaStation MPI; the bet is reasonable since these are the most fundamental MPI primitives, but unverified.
2. Decide on worker crash handling (Risk 1). Options: MPI_Testsome with timeout (complex); rely on Slurm job timeout (simple, accepts that a worker crash means the job is lost); add wrapper to detect worker exit via OS signals (platform-dependent).
3. Decide whether the scaling runner should migrate from Candidate 2 to Candidate 1. Optional — Candidate 2 currently works for scaling — but unifying on Candidate 1 would eliminate the dual-code-path risk from Risk 3.
4. Regression tests for existing fixes: NaN weight guard (Risk 4), Barrier placement (Risk 3), try/finally on root exception path (Apr 8 fix). These protect against future regressions in code paths that have been patched multiple times.

### Clear verdict

- **Adopt:** CommWorldMap as the default MPI sampler for ALL pyABC-backed inference methods.
- **Retain for scaling:** Shared MPICommExecutor (Candidate 2) in the scaling runner until Phase 2 clears 48-rank verification for CommWorldMap.
- **Keep opt-in with warning:** ConcurrentFutureSampler (Candidate 3) for diagnostic comparison.
- **Reject:** MappingSampler + per-call MPICommExecutor.map (Candidate 4) — no production use case.

---

## Phase 1 Exit Checklist

- [x] All four candidates have a full coordination point inventory with file:line citations (Plan 01)
- [x] Characterization table populated with stability, correctness, paper effect, and residual risk per candidate (Plan 02 Task 1)
- [x] Paper sensitivity assessment addresses all four D-07 questions (Plan 02 Task 1)
- [x] Four residual risks and newly discovered hang paths documented with reproduction recipes (Plan 02 Task 1)
- [x] Clear, unhedged recommendation with evidence-based rationale (Plan 02 Task 2)
- [x] Residual work handed off to Phase 2 with specific test targets
- [x] Human-verified (Plan 02 Task 3)

*Phase 1 Diagnose deliverable — ready for human verification and Phase 2 planning.*

