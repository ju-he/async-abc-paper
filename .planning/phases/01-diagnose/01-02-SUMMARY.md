---
phase: 01-diagnose
plan: 02
subsystem: documentation
tags: [mpi, pyabc, diagnosis, evaluation, recommendation]

# Dependency graph
requires:
  - phase: 01-diagnose plan 01
    provides: Per-candidate MPI coordination point inventory (all 4 candidates) in .plans/diagnose/mpi-evaluation.md

provides:
  - Characterization table comparing all 4 pyABC MPI sampler candidates across cluster stability, correctness, paper effect, and residual risk
  - Paper sensitivity assessment answering D-07 questions (Q1-Q4): particle correctness, wall-time semantics, sampler overhead, paper conclusion sensitivity
  - Residual risks section with 4 documented hang paths and reproduction recipes / test stub descriptions
  - Unhedged recommendation: CommWorldMap (Candidate 1) is the clear winner — eliminates inter-communicator hang class by design
  - Phase 1 Exit Checklist fully completed and human-verified
  - Complete Phase 1 Diagnose deliverable ready to hand to Phase 2 planning

affects:
  - Phase 2 MPI Hardening (implementation target, CommWorldMap verified at 48 ranks)
  - Phase 3 Code Cleanup (dead code removal targets from Candidates 3 and 4 rejection)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Static analysis only during diagnosis phase (D-04/D-05): no inline code changes, no new mpirun runs"
    - "Characterization table + paper sensitivity + residual risks + recommendation as structured deliverable"

key-files:
  created: []
  modified:
    - .plans/diagnose/mpi-evaluation.md

key-decisions:
  - "CommWorldMap (Candidate 1) is the recommended pyABC MPI sampler — eliminates inter-communicator hang class by design (no Create_intercomm/Disconnect)"
  - "Candidates 3 (ConcurrentFutureSampler) and 4 (per-call MPICommExecutor.map) rejected: both known broken at 48 ranks on ParaStation MPI"
  - "Candidate 2 (Shared MPICommExecutor) works only in its specific scaling runner pattern and is fragile; should not be generalized"
  - "Sampler choice does not affect which particles get accepted (correctness-equivalent for Q1); only wall-time fairness depends on it (Q4)"
  - "Worker crash during CommWorldMap.map is a newly documented theoretical hang path (Risk 1) — Phase 2 work"

patterns-established:
  - "Phase 2 implementation target is CommWorldMap verified at 48 ranks (MPI-01) with regression tests for NaN weight crash, double shutdown, and barrier timing races"

requirements-completed:
  - MPI-02
  - MPI-04

# Metrics
duration: 10min (plus human verification pause)
completed: 2026-04-10
---

# Phase 01 Plan 02: Diagnose Summary

**Complete MPI sampler evaluation with characterization table, paper sensitivity assessment (D-07), residual risks, and unhedged recommendation: CommWorldMap over Shared MPICommExecutor**

## Performance

- **Duration:** ~10 min execution + human verification pause
- **Started:** 2026-04-10T13:33:47+02:00
- **Completed:** 2026-04-10T13:42:12+02:00
- **Tasks:** 3 (Task 1: append sections, Task 2: append recommendation, Task 3: human verify)
- **Files modified:** 1

## Accomplishments

- Appended Characterization Table (4 candidates x 5 columns), Paper Sensitivity Assessment (D-07 Q1-Q4), and Residual Risks (4 hang paths with reproduction recipes) to `.plans/diagnose/mpi-evaluation.md`
- Appended Recommendation section with single unhedged winner (CommWorldMap) and evidence-based rationale citing the bug history
- Human verified the complete document — all Phase 1 Exit Checklist items now checked
- Phase 1 Diagnose deliverable complete and ready for Phase 2 planning

## Task Commits

Each task was committed atomically:

1. **Task 1: Append Characterization Table, Paper Sensitivity Assessment, and Residual Risks** - `86e20b0` (feat)
2. **Task 2: Append Recommendation section** - `6ef7724` (feat)
3. **Task 3: Mark human-verified checkbox** - `ef1d3a8` (docs)

## Files Created/Modified

- `.plans/diagnose/mpi-evaluation.md` — Complete Phase 1 deliverable: per-candidate inventory (Plan 01) + characterization table + paper sensitivity assessment + residual risks + recommendation (Plan 02)

## Decisions Made

- CommWorldMap is the recommended approach: eliminates inter-communicator hang class entirely by design; Candidates 3 and 4 eliminated by bug history; Candidate 2 fragile to ParaStation MPI version changes
- All four candidates are correctness-equivalent for particle acceptance (Q1) — sampler choice does not affect scientific conclusions, only wall-time fairness (Q4)
- Worker crash during CommWorldMap.map (Risk 1) is newly documented as theoretical hang path not yet in bug history — mitigation options deferred to Phase 2
- CommWorldMap unverified at 48 ranks on ParaStation MPI (Risk 2) — explicit Phase 2 MPI-01 work item

## Deviations from Plan

None — plan executed exactly as written. No inline code changes per D-04; no new mpirun runs per D-05.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

Phase 2 MPI Hardening can now begin:
- Implementation target: CommWorldMap (Candidate 1)
- MPI-01: Verify CommWorldMap at 48 ranks on ParaStation MPI cluster
- MPI-03: Wall-time stopping robustness (NaN weight guard regression test)
- TEST-01, TEST-03: Unit tests for CommWorldMap sampler path and wall-time stopping
- Four residual risks have specific test stubs described (Risk 1: worker crash bcast, Risk 2: 48-rank cluster run, Risk 3: Barrier regression, Risk 4: NaN weight unit test)

No blockers for Phase 2 start.

---
*Phase: 01-diagnose*
*Completed: 2026-04-10*
