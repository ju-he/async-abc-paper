# Phase 1: Diagnose - Context

**Gathered:** 2026-04-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce a written inventory of every rank coordination point for each candidate pyABC MPI sampler approach, evaluate each approach against the characterization criteria, and emit a clear recommendation. No code changes — the deliverable is documentation that Phase 2 consumes.

</domain>

<decisions>
## Implementation Decisions

### Candidate Approaches

- **D-01:** Evaluate **four** candidates (not three):
  1. **CommWorldMap** — current default; custom bcast/send/recv map, no inter-communicators (`pyabc_sampler.py`)
  2. **MappingSampler + shared MPICommExecutor** — scaling runner's existing shared-executor path; one Create_intercomm/Disconnect cycle
  3. **ConcurrentFutureSampler + MPICommExecutor** — explicit opt-in legacy path; known teardown hangs at scale (opt-in with warning)
  4. **Native pyABC MappingSampler with plain mpi4py comm** — pyABC's documented recommended MPI path, no custom map adapter; evaluate whether our CommWorldMap is actually necessary
- **D-02:** Treat this as a full re-evaluation — do not assume CommWorldMap is the winner going in. The recommendation must emerge from the analysis, not be predetermined.

### Inventory Format

- **D-03:** The written inventory lives in `.plans/diagnose/mpi-evaluation.md` — a structured markdown doc with:
  - Per-candidate section: every rank coordination point (bcast, send/recv, allgather, barrier, Create_intercomm, Disconnect)
  - Characterization table: cluster stability, correctness vs standard pyABC usage, effect on paper results
  - Clear recommendation with rationale
- **D-04:** No inline code changes in this phase. The doc is the deliverable.

### Verification Method

- **D-05:** Static code analysis + bug history review only. No new mpirun runs and no cluster jobs in Phase 1. Coordination points are identified by reading the code (`pyabc_sampler.py`, `abc_smc_baseline.py`, `pyabc_wrapper.py`, `runner.py`) and cross-referencing `.plans/bug-fixes/previous-fixes.md` for evidence of failure modes.
- **D-06:** Any newly discovered hang paths (not yet documented) get a reproduction recipe or test stub description added to the inventory — not an actual test (that's Phase 2).

### Paper Sensitivity Assessment

- **D-07:** "Effect on paper results" is assessed at reasoning level only — no new runs or timing numbers:
  - Does the sampler affect which particles get accepted (correctness of ABC-SMC)?
  - Does sampler choice change wall-time semantics (does work finished before the deadline change)?
  - Is sampler overhead significant relative to the 900s wall-time budget?
  - The paper's primary claim is wall-time comparison between async Propulate-ABC and sync pyABC; sampler choice is an implementation detail as long as it doesn't change accepted particle semantics.

### Claude's Discretion

- Structure of the per-candidate inventory sections (e.g., how to present bcast vs send/recv coordination points)
- Order of candidates in the evaluation table
- Whether to include a "rejected paths" section for approaches definitively eliminated

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Bug History
- `.plans/bug-fixes/previous-fixes.md` — full history of MPI hang fixes (Apr 4–8 2026); primary evidence base for the stability characterization of each candidate

### MPI Coordination Code
- `experiments/async_abc/inference/pyabc_sampler.py` — CommWorldMap, build_pyabc_sampler, resolve_pyabc_mpi_sampler; all candidate paths are controlled here
- `experiments/async_abc/inference/pyabc_wrapper.py` — CommWorldMap usage in pyabc_smc path (default mapping + legacy concurrent_futures)
- `experiments/async_abc/inference/abc_smc_baseline.py` — CommWorldMap usage in abc_smc_baseline path; identical structure to pyabc_wrapper

### Runner Coordination
- `experiments/async_abc/utils/runner.py` — run_method_distributed, run_method, allgather coordination; where rank modes (rank_zero, rank_parallel, all_ranks) are enforced
- `experiments/async_abc/utils/mpi.py` — get_rank, get_world_size, allgather, any_true; helper layer above mpi4py

### Requirements
- `.planning/REQUIREMENTS.md` §MPI-02, §MPI-04 — requirements this phase satisfies

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CommWorldMap` (`pyabc_sampler.py:48`) — fully implemented custom bcast/send/recv map; the primary candidate to verify
- `resolve_pyabc_mpi_sampler()` (`pyabc_sampler.py:279`) — controls which sampler path is selected; lists all supported values
- `build_pyabc_sampler()` (`pyabc_sampler.py:319`) — constructs the sampler; knows both mapping and concurrent_futures paths
- `TrackedFutureExecutor` (`pyabc_sampler.py:16`) — wraps executors for diagnostics (retained for concurrent_futures path)

### Established Patterns
- Both `abc_smc_baseline.py` and `pyabc_wrapper.py` have identical CommWorldMap usage patterns: `cmap.is_root` → run with `cmap.map`, `finally: cmap.shutdown()`, then workers in `cmap.worker_loop()`, followed by `COMM_WORLD.Barrier()`
- The scaling runner uses a shared `MPICommExecutor` pattern (separate from CommWorldMap) — this is the MappingSampler + shared MPICommExecutor candidate
- `run_method_distributed` determines execution mode (rank_zero, rank_parallel, all_ranks) and handles allgather coordination

### Integration Points
- The inventory doc integrates with Phase 2 planning: the recommendation becomes Phase 2's implementation target
- Newly discovered hang paths feed directly into Phase 2 test stubs

</code_context>

<specifics>
## Specific Ideas

- The 4th candidate (native pyABC MappingSampler with plain mpi4py comm) is specifically motivated by MPI-02's "correctness vs standard pyABC usage" criterion — characterize whether CommWorldMap deviates from how pyABC is designed to be used.
- The recommendation should be "clear" per the success criteria — not hedged. Evidence from the bug history is the primary basis.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-diagnose*
*Context gathered: 2026-04-10*
