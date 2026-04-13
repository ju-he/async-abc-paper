---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 02-03-PLAN.md (scaling_runner migration to CommWorldMap — PASS)
last_updated: "2026-04-13T21:30:00.000Z"
last_activity: 2026-04-13
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 5
  completed_plans: 5
  percent: 40
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-10)

**Core value:** Experiments must run reliably to completion on the cluster — all paper results depend on this.
**Current focus:** Phase 02 — mpi-hardening

## Current Position

Phase: 02 (mpi-hardening) — EXECUTING
Plan: 3 of 3 (Phase 2 complete)
Status: Phase 2 complete — ready for Phase 3
Last activity: 2026-04-13

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
| Phase 02-mpi-hardening P01 | 5 | 2 tasks | 4 files |
| Phase 02-mpi-hardening P02 | 3 days | 3 tasks | 3 files |

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
- [Phase 02-mpi-hardening]: Isolated non-mpirun tests (source-check, in-process shutdown) in TestMpiHardeningNoMpiRun class to guarantee execution regardless of mpirun probe state
- [Phase 02-mpi-hardening]: mpi helper assertion relaxed to allow empty records when max_wall_time_s given — early-stop is valid behavior not a failure
- [Phase 02-mpi-hardening]: CommWorldMap verified PASS at 48 ranks on JUWELS ParaStationMPI — MPI-01 satisfied; Plan 02-03 should proceed with scaling runner migration
- [Phase 02-mpi-hardening]: Local 2-rank smoke test (193 records, 2.6s) used as gate before cluster submission — confirms no local CommWorldMap regression
- [Phase 02-mpi-hardening]: scaling_runner.py migrated from shared MPICommExecutor to per-call CommWorldMap (D-03) — mpi_map not plumbed through run_method_distributed, safer per-call path used
- [Phase 02-mpi-hardening]: MPICommExecutor completely removed from scaling_runner.py; CommWorldMap is now the sole pyABC MPI coordination model in the codebase
- [Phase 02-mpi-hardening]: test_barrier_placement_source_check updated to Case B regex (post-mpi_methods-pass Barrier pattern); test passes after migration

### Pending Todos

None yet.

### Blockers/Concerns

- CommWorldMap fix unverified on cluster at 48 ranks — Phase 1 diagnosis should clarify risk
- Previous bug fixes in `.plans/bug-fixes/previous-fixes.md` — consult before touching MPI code

## Session Continuity

Last session: 2026-04-13T21:30:00.000Z
Stopped at: Completed 02-03-PLAN.md (scaling_runner migration to CommWorldMap — Phase 2 complete)
Resume file: None
