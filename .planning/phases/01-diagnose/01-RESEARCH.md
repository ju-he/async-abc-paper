# Phase 1: Diagnose - Research

**Researched:** 2026-04-10
**Domain:** MPI coordination patterns in pyABC sampler paths; static code analysis of `pyabc_sampler.py`, `pyabc_wrapper.py`, `abc_smc_baseline.py`, `runner.py`, `scaling_runner.py`, and `mpi.py`; cross-referenced with `.plans/bug-fixes/previous-fixes.md`
**Confidence:** HIGH — all findings are grounded in the actual source code and documented bug history; no cluster runs required

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Evaluate four candidates:
  1. CommWorldMap — current default; custom bcast/send/recv map, no inter-communicators
  2. MappingSampler + shared MPICommExecutor — scaling runner's existing shared-executor path
  3. ConcurrentFutureSampler + MPICommExecutor — explicit opt-in legacy path; known teardown hangs
  4. Native pyABC MappingSampler with plain mpi4py comm — pyABC's documented recommended MPI path, no custom map adapter
- **D-02:** Full re-evaluation; recommendation must emerge from analysis, not be predetermined
- **D-03:** Inventory lives in `.plans/diagnose/mpi-evaluation.md` — structured markdown with per-candidate sections, characterization table, and recommendation
- **D-04:** No inline code changes in Phase 1; the document is the deliverable
- **D-05:** Static code analysis + bug history review only; no new mpirun runs, no cluster jobs
- **D-06:** Newly discovered hang paths get a reproduction recipe or test stub description added to inventory
- **D-07:** "Effect on paper results" assessed at reasoning level only — no new runs or timing numbers

### Claude's Discretion

- Structure of per-candidate inventory sections
- Order of candidates in the evaluation table
- Whether to include a "rejected paths" section for approaches definitively eliminated

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.

</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MPI-02 | All pyABC MPI sampler options evaluated for correctness, cluster stability, closeness to standard pyABC usage; best approach selected with rationale; paper conclusions assessed for sensitivity to sampler choice | Direct: all four candidates' coordination points have been mapped from source; bug history provides stability evidence; paper sensitivity is assessable by reasoning about particle semantics |
| MPI-04 | Remaining hang paths diagnosed systematically — all rank coordination points documented and tested for each candidate MPI approach | Direct: coordination points for all four candidates fully extracted from source code; residual risks identified for inventory |

</phase_requirements>

---

## Summary

Phase 1 is a code-reading and documentation exercise. All four candidate sampler paths are already present in the codebase and their MPI coordination points can be fully enumerated by static analysis. The primary work is producing the structured inventory document at `.plans/diagnose/mpi-evaluation.md`.

The bug history in `.plans/bug-fixes/previous-fixes.md` is the strongest source of evidence for stability characterization. It documents a complete progression: ConcurrentFutureSampler with per-call MPICommExecutor deadlocked reliably at 48 ranks (Apr 5-7 entries); MappingSampler + shared single-lifecycle MPICommExecutor (scaling runner path) survived but introduced allgather skew; CommWorldMap (Apr 8) was designed to eliminate all inter-communicator usage and resolved the hang for the non-scaling runner. The 4th candidate (native pyABC MappingSampler with a plain mpi4py map) is not currently instantiated in the codebase — evaluating it requires assessing pyABC's public API and what CommWorldMap adds beyond it.

**Primary recommendation for inventory structure:** Per-candidate sections should list each MPI collective or point-to-point operation by call site (file:line), direction (root-only, all-ranks, worker-only), and tag/communicator. The characterization table should have one row per candidate and four columns: cluster stability, correctness vs standard pyABC, effect on paper results, residual risk.

---

## Architecture Patterns

### Overall Rank Protocol (all_ranks mode)

All pyABC-backed inference methods run in `all_ranks` execution mode. The protocol enforced by `run_method_distributed` (runner.py:750) is:

```
All ranks: run_method() is called
  └── pyabc_wrapper / abc_smc_baseline dispatch by mpi_sampler
      └── sampler-specific coordination (see per-candidate below)
All ranks: allgather(error_payload)   ← runner.py:876
Root: return records
Workers: return []
```

The `allgather` at runner.py:876 is the shared exit gate for all `all_ranks` methods. Every sampler path must ensure all ranks reach it without hanging.

### Exception: Scaling Runner Shared Executor Path

When `mpi_executor is not None` (scaling_runner.py:964), the runner calls `run_method` directly on root only, bypassing `run_method_distributed` entirely. Workers are blocked inside `MPICommExecutor`'s server recv loop. This means `allgather` is never called mid-workload; synchronization happens only once via `COMM_WORLD.Barrier()` after the entire `with MPICommExecutor(...)` context exits (scaling_runner.py:1067-1068).

---

## Candidate Coordination Point Inventory (Source Evidence)

### Candidate 1: CommWorldMap (current default)

**Entry point:** `pyabc_wrapper.py:404` / `abc_smc_baseline.py:389`

**Root path:**
1. `CommWorldMap.__init__` — no MPI call; sets `self.is_root = True`
2. `CommWorldMap.map()` — per batch:
   - `COMM_WORLD.bcast(("map", fn), root=0)` — broadcasts function to all workers (pyabc_sampler.py:97)
   - `COMM_WORLD.send((idx, item), dest=worker, tag=0)` × N — distributes work items (pyabc_sampler.py:106)
   - `COMM_WORLD.recv(source=ANY_SOURCE, tag=1, status=status)` × N — collects results (pyabc_sampler.py:115)
   - On worker error: `COMM_WORLD.send(SENTINEL, dest=worker, tag=0)` to drain (pyabc_sampler.py:124)
   - `COMM_WORLD.recv(source=ANY_SOURCE, tag=1)` × remaining active (pyabc_sampler.py:126)
3. `CommWorldMap.shutdown()` — `COMM_WORLD.bcast(("shutdown", None), root=0)` (pyabc_sampler.py:148); idempotent guard via `self._shutdown`
4. `COMM_WORLD.Barrier()` — pyabc_wrapper.py:413 / abc_smc_baseline.py:399

**Worker path:**
1. `CommWorldMap.worker_loop()` — loops:
   - `COMM_WORLD.bcast(None, root=0)` — waits for command (pyabc_sampler.py:153); tag=="shutdown" breaks
   - `COMM_WORLD.recv(source=0, tag=0)` — receives work item or sentinel (pyabc_sampler.py:158)
   - `COMM_WORLD.send((idx, result_or_error), dest=0, tag=1)` — sends result (pyabc_sampler.py:165/167)
2. `COMM_WORLD.Barrier()` — same barrier as root

**Shutdown safety:** `try/finally` wraps root path (pyabc_wrapper.py:406-410, abc_smc_baseline.py:391-395) — `cmap.shutdown()` is guaranteed even on root exception (Apr 8 fix).

**Known hang scenario (resolved Apr 8):** If root exception skipped `shutdown()`, workers blocked at `bcast` forever, and root then blocked at `allgather` in runner — double deadlock. Fixed by the try/finally.

**Residual risk:** If a worker dies/exits before receiving the shutdown `bcast`, root's `bcast` will hang because mpi4py's `bcast` on `COMM_WORLD` requires all ranks to participate. Not yet documented in bug history. Reproduction path: kill a worker process mid-run.

---

### Candidate 2: MappingSampler + Shared MPICommExecutor (scaling runner)

**Entry point:** `scaling_runner.py:1062` — one `with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:` wrapping all pyABC workloads

**Root path (inside context):**
1. `MPICommExecutor.__enter__` — `Create_intercomm` on COMM_WORLD (mpi4py internals; one call per n_workers config)
2. For each workload: `run_method(... mpi_executor=executor)` — root calls `executor.map(fn, items)` which dispatches via MPIPoolExecutor
3. `MPICommExecutor.__exit__` — `executor.shutdown(wait=True)` + `Disconnect()` on inter-communicator — one call total

**Worker path (inside context):**
Workers enter the MPIPoolExecutor server recv loop. They never call `run_method`, never reach `allgather`. They stay blocked until root exits the context.

**Post-context sync:**
- `COMM_WORLD.Barrier()` at scaling_runner.py:1067-1068 — all ranks, after `with` block exits

**Non-MPI methods (Candidate 2's co-existing path):**
Before the shared executor block, non-MPI methods run via `run_method_distributed` normally, with full `allgather` coordination (scaling_runner.py:1052-1053).

**Known hang scenarios (resolved):**
- Apr 7: repeated `Create_intercomm`/`Disconnect` cycles deadlocked at 48 ranks (second cycle). Fixed by single shared executor.
- Apr 7 follow-up: workers in server loop couldn't reach `allgather`, so root calling `run_method_distributed` blocked forever. Fixed by calling `run_method` directly when `mpi_executor is not None`.

**Residual risk:** Only one `Create_intercomm`/`Disconnect` cycle, but it still uses inter-communicators. On ParaStation MPI, even a single cycle at 48 ranks has been documented to deadlock in some experiments (e.g. lotka_volterra, Apr 8 entry). The scaling runner happens to work because it exercises this path differently (server loop model vs. per-call model) — but this is still fragile.

---

### Candidate 3: ConcurrentFutureSampler + MPICommExecutor (legacy opt-in)

**Entry point:** `pyabc_wrapper.py:375` / `abc_smc_baseline.py:359` — activated when `mpi_sampler == "concurrent_futures"`

**Root path:**
1. `with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:` — `Create_intercomm`
2. `ConcurrentFutureSampler(cfuture_executor=tracker, client_max_jobs=N).sample(...)` — submits up to N concurrent futures via `executor.submit()`
3. `MPICommExecutor.__exit__` — `executor.shutdown(wait=True)` + `Disconnect()` per call
4. `COMM_WORLD.Barrier()` — pyabc_wrapper.py:399-400 / abc_smc_baseline.py:384-385

**Worker path:**
Workers block inside MPIPoolExecutor server loop until `Disconnect` on inter-communicator.

**Known hang scenarios (documented, unresolved for this path):**
- Apr 5-6: `tracker.drain()` blocked indefinitely waiting on speculative queue (`client_max_jobs=200`)
- Apr 5-6: double shutdown — explicit `executor.shutdown()` + `MPICommExecutor.__exit__` both called the same pool shutdown
- Apr 6-7: even with bounded `client_max_jobs=n_workers` and single-owner shutdown, `MPICommExecutor.__exit__` still deadlocked at 48 ranks on ParaStation MPI
- Apr 7 (final): the path was demoted to opt-in with a warning; `concurrent_futures` is documented as having known teardown hangs at scale

**Residual risk:** This path is known-broken at 48 ranks on ParaStation MPI. The warning in `resolve_pyabc_mpi_sampler` (pyabc_sampler.py:296-302) explicitly documents this. There are no unsolved sub-bugs — the entire path is the known failure mode.

---

### Candidate 4: Native pyABC MappingSampler with plain mpi4py comm

**Status:** NOT currently instantiated in the codebase. This is the hypothetical "closest to pyABC's own MPI documentation" path.

**What pyABC's MappingSampler expects:** According to pyABC's API, `MappingSampler(map_=map_fn)` accepts any callable with the signature `map(fn, iterable) -> list`. pyABC does not impose any MPI requirements on this callable — it just calls it each generation.

**What CommWorldMap provides vs. plain mpi4py:** The `CommWorldMap.map` method (pyabc_sampler.py:76-140) implements dynamic work distribution with `bcast`/`send`/`recv`. A "native pyABC with plain mpi4py" path would need to implement some equivalent: either `COMM_WORLD.scatter`/`gather`, `executor.map` from MPICommExecutor, or a custom map. pyABC itself does not ship an MPI map implementation — it expects the user to provide one.

**Implication:** Candidate 4 as described ("no custom map adapter") cannot be instantiated without providing a map callable. The question the CONTEXT.md is really asking is: does `CommWorldMap` deviate from what pyABC intends MappingSampler to receive? The answer is no — `MappingSampler` is designed for exactly this use case, and CommWorldMap's `map` method satisfies its interface.

**Residual risk:** If one interprets Candidate 4 as "use mpi4py's built-in `executor.map` from a per-call MPICommExecutor," that is exactly what the original mapping path did (Apr 7: "switch default MPI pyABC sampler from futures to synchronous mapping"). That path used `executor.map` from `MPICommExecutor` — and it still suffered `Disconnect` teardown hangs at 48 ranks, motivating CommWorldMap.

---

## Common Pitfalls

### Pitfall 1: bcast Requires All-Rank Participation

**What goes wrong:** `COMM_WORLD.bcast(None, root=0)` in mpi4py is a collective operation — it blocks until every rank in `COMM_WORLD` calls it. If any rank exits the process, crashes, or skips the call, root hangs forever.

**Why it happens:** Worker processes reach `worker_loop()` and enter the `bcast` wait. If root throws an exception before calling `shutdown()`, the shutdown `bcast` is never sent, and workers wait indefinitely. The Apr 8 fix (try/finally) addresses the exception case, but a worker crash would still cause root to hang at the next `bcast`.

**How to avoid:** Ensure `shutdown()` is always called from root (try/finally). For worker crashes, a timeout or `MPI_Testsome` pattern would be needed — not currently implemented.

**Warning signs:** All worker ranks print no output after the last `bcast` received; root appears to be in `bcast` with no progress.

### Pitfall 2: allgather After CommWorldMap Must Wait for Barrier

**What goes wrong:** In `run_method_distributed`, `allgather(error_payload)` is called after `run_method` returns. CommWorldMap's design requires the `COMM_WORLD.Barrier()` (pyabc_wrapper.py:413) to fire before allgather so all workers have exited `worker_loop()`. If `Barrier` is removed or misplaced, workers entering `allgather` while root is still in `map()` deadlocks.

**Why it happens:** Workers exit `worker_loop()` only after receiving the shutdown `bcast`. The `Barrier()` after `cmap.worker_loop()` ensures workers have exited before reaching `allgather`.

**How to avoid:** Never remove or reorder the `COMM_WORLD.Barrier()` that follows CommWorldMap usage.

### Pitfall 3: MPICommExecutor allgather Skew (Shared Executor Path)

**What goes wrong:** When workers are inside the MPIPoolExecutor server loop, they cannot participate in `COMM_WORLD.allgather`. Any call to `run_method_distributed` from root while workers are in the server loop will block forever.

**Why it happens:** `run_method_distributed` ends with `allgather(error_payload)` which is a collective requiring all ranks.

**How to avoid:** When `mpi_executor is not None`, call `run_method` directly (root only), not `run_method_distributed`. This is already implemented in scaling_runner.py:964-990.

### Pitfall 4: NaN Weight on Partial Generation

**What goes wrong:** When `max_wall_time_s` fires mid-generation, pyABC stops collecting particles. The resulting population has 0 or NaN weight sum, causing `AssertionError` in `Population.__init__`.

**Why it happens:** pyABC's `max_walltime` check occurs between generations, but a long final generation can overshoot. The fix (Apr 8) catches this specific `AssertionError` and falls back to `abc.history`.

**How to avoid:** Keep the `except AssertionError` guard that checks for "weight" and "nan" in both `pyabc_wrapper.py:125-133` and `abc_smc_baseline.py:127-136`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MPI map primitive | Custom scatter/gather | CommWorldMap.map | Dynamic load balance, ordered results, error propagation already implemented |
| Rank coordination helpers | Direct mpi4py calls scattered across files | `mpi.py` (`allgather`, `any_true`, `get_rank`) | Provides mpi4py-absent fallback for single-process runs |
| Shutdown safety | Ad-hoc flag checks | `try/finally: cmap.shutdown()` pattern | Already established and bug-tested; deviation is how Apr 8 hang was introduced |

---

## Runtime State Inventory

Step 2.5: SKIPPED — this is not a rename/refactor/migration phase.

---

## Environment Availability

Step 2.6: SKIPPED — Phase 1 is a pure documentation/analysis phase. No external tools, runtimes, or CLIs are needed beyond reading source files. No mpirun runs, no cluster jobs.

---

## Validation Architecture

Step 4: nyquist_validation — no `.planning/config.json` found. Treated as absent (enabled).

However, Phase 1's deliverable is a markdown document, not code. There is no automated test for document completeness. The success criteria from CONTEXT.md serve as the verification checklist:

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| MPI-02 | All four candidates evaluated with characterization table and recommendation | Manual review | — (document review) | No — Wave 0 gap |
| MPI-04 | Every rank coordination point documented per candidate | Manual review | — (document review) | No — Wave 0 gap |

### Wave 0 Gaps

- [ ] `.plans/diagnose/mpi-evaluation.md` — the primary deliverable; does not yet exist
- [ ] Directory `.plans/diagnose/` does not yet exist; must be created

---

## Code Examples

### CommWorldMap Shutdown Safety Pattern (established)

```python
# Source: pyabc_wrapper.py:404-414, abc_smc_baseline.py:389-400
cmap = CommWorldMap(MPI.COMM_WORLD)
result: List[ParticleRecord] = []
if cmap.is_root:
    try:
        result = _run_with_map_callable(cmap.map)
    finally:
        cmap.shutdown()   # guaranteed even on exception
else:
    cmap.worker_loop()    # blocks until shutdown bcast
if MPI.COMM_WORLD.Get_size() > 1:
    MPI.COMM_WORLD.Barrier()
return result
```

### Shared MPICommExecutor Pattern (scaling runner)

```python
# Source: scaling_runner.py:1058-1071
with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
    if executor is not None:
        _run_workloads(mpi_methods, mpi_executor=executor)
    # Workers block in server recv loop until context exits
if MPI.COMM_WORLD.Get_size() > 1:
    MPI.COMM_WORLD.Barrier()
```

### Shared Executor Worker-Skip Pattern (no allgather when workers in server loop)

```python
# Source: scaling_runner.py:964-990
if mpi_executor is not None:
    # root-only: workers are in MPIPoolExecutor server loop, cannot allgather
    records = run_method(
        base_method, ..., mpi_executor=mpi_executor
    )
else:
    records = run_method_distributed(...)  # full allgather coordination
```

### NaN Weight Guard Pattern

```python
# Source: pyabc_wrapper.py:115-133, abc_smc_baseline.py:116-136
try:
    history = abc.run(...)
except AssertionError as exc:
    if "weight" in str(exc) and "nan" in str(exc).lower():
        logger.warning("[...] NaN population weight — treating as early wall-time stop")
        history = abc.history
    else:
        raise
```

---

## Paper Sensitivity Assessment (D-07)

**Does sampler choice affect which particles get accepted (correctness)?**
No, for all candidates. `MappingSampler` and `ConcurrentFutureSampler` both pass the same `simulate_fn` to pyABC's ABC-SMC loop. The particle selection, epsilon schedule, and weight normalization are handled entirely by `pyabc.ABCSMC` — the sampler controls only _how_ simulations are dispatched to workers, not _which_ results are accepted. CommWorldMap's `map` function preserves input order (results returned in submission order via `results[idx]`), so there is no reordering of accepted particles.

**Does sampler choice change wall-time semantics?**
Potentially yes, in a small way. CommWorldMap uses dynamic work distribution (one item at a time per worker, root dispatches next item as each worker becomes idle). This is standard parallel map semantics and matches what pyABC's generation loop expects. ConcurrentFutureSampler pre-submits up to `client_max_jobs` futures speculatively — this means workers can start computing particles for the _next_ tolerance threshold before the current generation completes, which technically means some work counted in one generation was started before that generation's epsilon was known. Under wall-time semantics, this could let slightly more work complete within the 900s budget, but the effect is bounded and small.

**Is sampler overhead significant relative to 900s wall-time budget?**
No, for CommWorldMap. The coordination overhead per generation is proportional to `bcast + send*k + recv*k`, where k is population size. At 48 workers and k~100-1000, this is sub-millisecond compared to simulation durations. MPICommExecutor teardown overhead was documented as ~40-50s at 48 ranks (Apr 7 entry) — 5.6% of the 900s budget — but CommWorldMap eliminates this by avoiding inter-communicators entirely.

**Paper conclusion sensitivity:**
The paper's primary claim compares wall-time efficiency of async Propulate-ABC vs. sync pyABC. Sampler choice affects pyABC's internal throughput only marginally; the structural advantage of async over sync is independent of which MPI dispatch mechanism pyABC uses.

---

## Open Questions

1. **Candidate 4 interpretation**
   - What we know: pyABC's `MappingSampler` accepts any callable; CommWorldMap.map satisfies this interface; pyABC does not ship its own MPI map callable
   - What's unclear: The CONTEXT.md specifies "native pyABC MappingSampler with plain mpi4py comm — no custom map adapter." If "no custom map adapter" means no CommWorldMap, then what map callable would be provided? mpi4py does not have a standalone `map` function. The only mpi4py-backed map available is `MPICommExecutor.map`, which returns to Candidate 2.
   - Recommendation: In the inventory document, characterize Candidate 4 as "MappingSampler + executor.map from a per-call MPICommExecutor" (the original mapping path before CommWorldMap). This is what pyABC's official MPI examples show, and it is distinct from Candidate 2 (which uses a shared single-lifecycle executor). The bug history directly covers this path (Apr 7 entries).

2. **CommWorldMap at 48 ranks: unverified on cluster**
   - What we know: CommWorldMap was developed Apr 8 and is the current default. Bug history shows it fixed hangs in non-scaling jobs. STATE.md notes it is "unverified at 48 ranks."
   - What's unclear: Whether the `bcast`-based coordination is stable under ParaStation MPI at 48 ranks at scale (the scaling runner does not currently use CommWorldMap — it uses the shared MPICommExecutor path).
   - Recommendation: Document as a residual risk in the inventory. Note that verification belongs to Phase 2 (MPI-01).

3. **Worker crash during CommWorldMap.map**
   - What we know: If a worker process exits unexpectedly, root's next `bcast` call will block indefinitely because `COMM_WORLD.bcast` requires all ranks.
   - What's unclear: Whether this failure mode has been observed or whether worker crashes are possible in the current experimental setup.
   - Recommendation: Document as a theoretical hang path with reproduction recipe: "launch with n_workers=2, kill worker process mid-map, observe root hang at bcast."

---

## Sources

### Primary (HIGH confidence)

- `experiments/async_abc/inference/pyabc_sampler.py` — CommWorldMap class (lines 48-168), all `resolve_*` and `build_*` functions
- `experiments/async_abc/inference/pyabc_wrapper.py` — run_pyabc_smc MPI dispatch (lines 324-415)
- `experiments/async_abc/inference/abc_smc_baseline.py` — run_abc_smc_baseline MPI dispatch (lines 306-400)
- `experiments/async_abc/utils/runner.py` — run_method_distributed, rank protocol (lines 750-886)
- `experiments/async_abc/utils/mpi.py` — allgather, get_rank, any_true helpers
- `experiments/scripts/scaling_runner.py` — shared MPICommExecutor pattern (lines 934-1071)
- `.plans/bug-fixes/previous-fixes.md` — complete Apr 4-8 hang/crash history

### Secondary (MEDIUM confidence)

- CONTEXT.md Candidate 4 description cross-referenced with pyABC source interface expectations — MEDIUM because pyABC internals not directly read (would require pyabc package inspection)

### Tertiary (LOW confidence)

- None

---

## Metadata

**Confidence breakdown:**
- Candidate coordination points (CommWorldMap, MPICommExecutor paths): HIGH — read directly from source
- Bug history characterization: HIGH — read directly from `.plans/bug-fixes/previous-fixes.md`
- Candidate 4 evaluation: MEDIUM — inferred from pyABC API expectations and cross-referencing with existing code paths
- Paper sensitivity assessment: MEDIUM — reasoning-level only per D-07; no new runs

**Research date:** 2026-04-10
**Valid until:** Stable — only changes if source files are modified. Re-read pyabc_sampler.py, pyabc_wrapper.py, abc_smc_baseline.py before Phase 2 planning.
