---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: verifying
stopped_at: Completed 01-02-PLAN.md — Phase 01 Diagnose complete
last_updated: "2026-04-10T11:43:28.812Z"
last_activity: 2026-04-10
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-10)

**Core value:** Experiments must run reliably to completion on the cluster — all paper results depend on this.
**Current focus:** Phase 01 — diagnose

## Current Position

Phase: 01 (diagnose) — EXECUTING
Plan: 2 of 2
Status: Phase complete — ready for verification
Last activity: 2026-04-10

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:** No data yet
| Phase 01-diagnose P01 | 30 | 2 tasks | 1 files |
| Phase 01-diagnose P02 | 10 | 3 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- CommWorldMap replaces MPICommExecutor (Create_intercomm deadlocks ParaStation MPI at scale) — unverified at 48 ranks
- Wall-time as sole stopping criterion — partial generations causing NaN weights, handled by catch-and-recover
- [Phase 01-diagnose]: Plan 01 scaffold + all 4 candidate inventory sections; Plan 02 appends characterization and recommendation — clean separation
- [Phase 01-diagnose]: Candidate 4 is historical reconstruction (pre-CommWorldMap path removed Apr 8); pyabc_sampler.py:158 is actually line 159 in current source
- [Phase 01-diagnose]: CommWorldMap (Candidate 1) is the recommended pyABC MPI sampler — eliminates inter-communicator hang class by design
- [Phase 01-diagnose]: Candidates 3 and 4 rejected: both known broken at 48 ranks on ParaStation MPI

### Pending Todos

None yet.

### Blockers/Concerns

- CommWorldMap fix unverified on cluster at 48 ranks — Phase 1 diagnosis should clarify risk
- Previous bug fixes in `.plans/bug-fixes/previous-fixes.md` — consult before touching MPI code

## Session Continuity

Last session: 2026-04-10T11:43:28.808Z
Stopped at: Completed 01-02-PLAN.md — Phase 01 Diagnose complete
Resume file: None
