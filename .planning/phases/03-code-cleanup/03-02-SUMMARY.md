---
phase: 03-code-cleanup
plan: 02
subsystem: inference
tags: [mpi, documentation, commworldmap, rank-protocol]

# Dependency graph
requires:
  - phase: 03-01
    provides: Dead MPICommExecutor code removed; CommWorldMap is sole pyABC MPI coordination model
provides:
  - CommWorldMap class docstring expanded with coordination model, rank protocol, and four failure modes
  - Per-method docstrings for map(), worker_loop(), shutdown() describing rank protocol role
  - References to .plans/diagnose/mpi-evaluation.md and .plans/bug-fixes/previous-fixes.md embedded in code
affects: [03-03, 04-reproducibility]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Inline documentation pattern: class-level docstring summarises design, references external evaluation doc rather than duplicating"
    - "Method docstrings enumerate numbered protocol steps matching method body"

key-files:
  created: []
  modified:
    - experiments/async_abc/inference/pyabc_sampler.py

key-decisions:
  - "Class docstring updated Usage block to match production caller pattern: try/finally around root branch, Barrier guarded by Get_size() > 1"
  - "Failure modes 1-4 summarised inline; full evaluation delegated to .plans/diagnose/mpi-evaluation.md (no duplication)"
  - "Human reviewer approved documentation accuracy (Task 3 checkpoint cleared)"

patterns-established:
  - "Documentation pattern: rank protocol steps numbered 1-N matching code flow, matching the coordination sequence in the class docstring"

requirements-completed: [CODE-02]

# Metrics
duration: ~25min
completed: 2026-04-14
---

# Phase 3 Plan 02: CommWorldMap Documentation Summary

**CommWorldMap class and method docstrings now describe coordination model, rank protocol, and four failure modes — CODE-02 satisfied**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-04-14
- **Tasks:** 3 (2 auto + 1 human checkpoint)
- **Files modified:** 1

## Accomplishments

- CommWorldMap class docstring expanded from a Usage-only block to a full design reference covering coordination model, rank protocol sequence, four known failure modes, and pointers to mpi-evaluation.md and previous-fixes.md
- Per-method docstrings added for map(), worker_loop(), and shutdown() describing each method's role in the rank protocol with numbered steps
- Usage example updated to match production caller pattern (try/finally, Barrier guarded by Get_size() > 1)
- Human reviewer approved documentation accuracy — CODE-02 satisfied

## Docstring Line Ranges

All edits are docstring-only. No method bodies or module structure changed.

| Location | Lines (post-edit) | Before (lines) | After (lines) |
|---|---|---|---|
| CommWorldMap class docstring | 16–95 | ~14 lines (Usage block only) | ~80 lines |
| map() docstring | 107–132 | 4 lines (one-liner + 2 sentences) | 26 lines |
| shutdown() docstring | 193–203 | 1 line | 11 lines |
| worker_loop() docstring | 211–231 | 1 line | 21 lines |

## Full Text of Expanded Class Docstring

```
COMM_WORLD-based blocking parallel map for pyABC's MappingSampler.

Replaces ``MPICommExecutor`` to avoid ``Create_intercomm`` /
``Disconnect`` fragility on ParaStation MPI at high rank counts.
Uses only ``bcast``, ``send``, and ``recv`` on ``COMM_WORLD`` —
no inter-communicators.

Coordination model
------------------
Root (rank 0) drives the computation. Workers (rank 1..N-1) spin
in :meth:`worker_loop` until root signals shutdown. Each ``map()``
call is a self-contained batch: root broadcasts the function,
distributes work items one at a time (for load balance), collects
results, and sends sentinels to park workers back in their
outer loop. The same ``CommWorldMap`` instance can service many
``map()`` calls between construction and ``shutdown()``.

Rank protocol (per ``map()`` call, ``size > 1``)
------------------------------------------------
::

    Root (rank 0)                   Workers (rank 1..N-1)
    ──────────────                  ─────────────────────
    bcast(("map", fn))  ──────►     tag, payload = bcast(None)
                                    fn = payload
    send(idx, item) ×N  ──────►     item = recv(source=0, tag=0)
                                    idx, work = item
                                    result = fn(work)
    recv(result) ×N     ◄──────     send((idx, result), tag=1)
    send(sentinel) ×N   ──────►     item is sentinel → break inner loop
                                    (wait on next bcast)
    shutdown():
      bcast(("shutdown", ...))──►   tag == "shutdown" → exit worker_loop
    COMM_WORLD.Barrier()            COMM_WORLD.Barrier()

After ``shutdown()`` the caller should call ``COMM_WORLD.Barrier()``
(the wrapper / baseline functions do this) and then proceed to
``allgather`` of results if needed.

Known failure modes
-------------------
1. **Worker crash during** ``map()``. Root receives a
   ``_WorkerError`` wrapper, drains remaining workers with sentinels,
   and re-raises. Workers that haven't yet received a sentinel wait
   at ``recv(source=0, tag=0)`` until the drain completes. See the
   drain block inside :meth:`map`.

2. **Root exception before** ``shutdown()``. The caller MUST wrap the
   root branch in ``try/finally`` and call ``cmap.shutdown()`` in
   ``finally``. Without this, workers block forever at
   ``bcast(None, root=0)``. This was the Apr 8 2026 teardown fix
   (see ``.plans/bug-fixes/previous-fixes.md``).

3. **Worker crash between** ``map()`` **calls.** No liveness check — by
   design, no heartbeat. Root's next ``bcast(("map", fn))`` hangs
   because not all ranks participate. Job is eventually killed by
   SLURM timeout. Trade-off rationale: see
   ``.plans/diagnose/mpi-evaluation.md`` (Residual Risks section).

4. **Single-process fallback** (``self.size <= 1``). :meth:`map`
   bypasses all MPI calls and runs ``[fn(item) for item in items]``
   sequentially. Used in local ``--test`` runs without ``mpirun``.

Full design evaluation: ``.plans/diagnose/mpi-evaluation.md``.
Bug history: ``.plans/bug-fixes/previous-fixes.md``.

Usage (all ranks must call the same code path)::

    cmap = CommWorldMap(MPI.COMM_WORLD)
    if cmap.is_root:
        try:
            sampler = build_pyabc_sampler(..., mpi_map=cmap.map)
            result = run_abc(sampler=sampler, ...)
        finally:
            cmap.shutdown()          # tells workers to exit
    else:
        cmap.worker_loop()           # blocks until shutdown
    if comm.Get_size() > 1:
        comm.Barrier()
```

## Method Docstring Line Counts (before/after)

| Method | Before | After |
|---|---|---|
| map() | 4 lines | 26 lines |
| shutdown() | 1 line | 11 lines |
| worker_loop() | 1 line | 21 lines |

## Human Reviewer Approval

- **Task 3 checkpoint:** Human review of CommWorldMap documentation accuracy
- **Resume signal received:** "approved"
- **Amendments requested:** None
- **Outcome:** CODE-02 satisfied — documentation rated accurate vs. code and consistent with mpi-evaluation.md

## Task Commits

1. **Task 1: Expand CommWorldMap class docstring** - `4e529d6` (docs)
2. **Task 2: Add per-method docstrings to map(), worker_loop(), shutdown()** - `2059381` (docs)
3. **Task 3: Human review checkpoint** - N/A (no code changes; approval signal: "approved")

## Files Created/Modified

- `experiments/async_abc/inference/pyabc_sampler.py` — CommWorldMap class docstring expanded; map(), shutdown(), worker_loop() method docstrings added

## Deviations from Plan

None — plan executed exactly as written. The two minor adjustments to the Usage block (try/finally, Barrier guard) were specified in the plan's action section.

## Requirements Completed

- **CODE-02**: CommWorldMap design, rank protocol, and known failure modes are described in inline docstrings sufficient to understand the coordination model without git history. Satisfied.

## Self-Check: PASSED

- `experiments/async_abc/inference/pyabc_sampler.py` exists and contains all required docstring content
- Commits 4e529d6 and 2059381 are present in git history
- ast.parse() passed during task execution (no syntax damage)
- Test suite passed: test_inference.py and test_mpi_hardening.py green
