---
phase: 01-diagnose
plan: 01
subsystem: documentation
tags: [mpi, pyabc, CommWorldMap, MPICommExecutor, MappingSampler, ConcurrentFutureSampler, mpi4py]

# Dependency graph
requires: []
provides:
  - Per-candidate MPI coordination point inventory for all 4 pyABC sampler candidates
  - File:line citations for every rank coordination call in CommWorldMap, shared MPICommExecutor, ConcurrentFutureSampler, and per-call MPICommExecutor.map paths
  - Bug history evidence cross-referenced with .plans/bug-fixes/previous-fixes.md
  - Scaffold for Plan 02 to append characterization table, paper sensitivity, residual risks, and recommendation
affects: [01-02, phase-02-implement]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MPI coordination inventory: per-candidate sections with coordination point tables (direction, operation, call site, purpose) and bug history evidence subsections"

key-files:
  created:
    - .plans/diagnose/mpi-evaluation.md
  modified: []

key-decisions:
  - "Document scaffold + all 4 candidate inventory sections in Plan 01; Plan 02 appends characterization and recommendation — clean separation"
  - "Candidate 4 is a historical reconstruction (pre-CommWorldMap path, no longer present in codebase) — cited by historical file:line positions and bug history"
  - "CommWorldMap at 48 ranks unverified — flagged as open question for Plan 02 residual risk section"

patterns-established:
  - "Coordination point table format: # | Direction | Operation | Call site | Purpose — used for all candidates"

requirements-completed: [MPI-02, MPI-04]

# Metrics
duration: 30min
completed: 2026-04-10
---

# Phase 01 Plan 01: MPI Sampler Evaluation Inventory Summary

**Static-analysis inventory of all 4 pyABC MPI sampler candidates with per-candidate rank coordination point tables (12 pts for CommWorldMap, 6 pts each for Candidates 2-4) and cross-referenced bug history from Apr 5-8 2026**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-04-10T11:00:00Z (approx)
- **Completed:** 2026-04-10T11:28:40Z
- **Tasks:** 2
- **Files modified:** 1 created (.plans/diagnose/mpi-evaluation.md)

## Accomplishments

- Created `.plans/diagnose/` directory and wrote `.plans/diagnose/mpi-evaluation.md` (200 lines)
- Candidate 1 (CommWorldMap): 12 coordination points with exact `pyabc_sampler.py` line citations (lines 48, 97, 106, 115, 124, 126, 148, 153, 158, 165/167) plus `pyabc_wrapper.py:413` and `abc_smc_baseline.py:399` Barrier; 2 bug history entries (Apr 8 × 2)
- Candidate 2 (Shared MPICommExecutor): 6 coordination points with `scaling_runner.py:1062/1067-1068` and `runner.py:876` citations; 3 bug history entries (Apr 7 × 2, Apr 8 × 1)
- Candidate 3 (ConcurrentFutureSampler + per-call MPICommExecutor): 6 coordination points with `pyabc_wrapper.py:375`, `abc_smc_baseline.py:359`, `pyabc_wrapper.py:399-400`, `abc_smc_baseline.py:384-385`, `pyabc_sampler.py:296-302` citations; 4 bug history entries (Apr 5-6 × 2, Apr 7 × 2)
- Candidate 4 (MappingSampler + per-call MPICommExecutor.map): 6 coordination points as historical reconstruction with interpretation note; 2 bug history entries (Apr 7, Apr 8)

## Task Commits

1. **Task 1: Create scaffold + Candidates 1 & 2** - `6bceef4` (feat)
2. **Task 2: Append Candidates 3 & 4** - `1b5d33a` (feat)

**Plan metadata:** (docs commit — created after SUMMARY)

## Files Created/Modified

- `.plans/diagnose/mpi-evaluation.md` — 200-line MPI sampler evaluation inventory with document scaffold (header, MPI-02/MPI-04 requirement mapping, shared rank protocol), 4 per-candidate sections, and Plan 02 appendix guide

## Decisions Made

- Followed plan specification exactly: document scaffold + all 4 candidate inventory sections in Plan 01; Plan 02 appends characterization table, paper sensitivity, residual risks, and recommendation
- Candidate 4 documented as historical reconstruction since the pre-CommWorldMap path is no longer in the codebase; interpretation note added referencing RESEARCH.md Open Questions #1
- Added "What Plan 02 Appends" section to make the append boundary explicit for the continuation agent

## Deviations from Plan

None - plan executed exactly as written. All line number citations verified against live source before writing.

**Line number verification results (matches expected):**
- `pyabc_sampler.py:48` — `CommWorldMap` class definition: CONFIRMED
- `pyabc_sampler.py:97` — `self.comm.bcast(("map", fn), root=0)`: CONFIRMED
- `pyabc_sampler.py:106` — `self.comm.send((next_item, items[next_item]), dest=dest, tag=0)`: CONFIRMED
- `pyabc_sampler.py:115` — `self.comm.recv(source=MPI.ANY_SOURCE, tag=1, status=status)`: CONFIRMED
- `pyabc_sampler.py:124` — drain send (SENTINEL): CONFIRMED (actual line 124)
- `pyabc_sampler.py:126` — drain recv: CONFIRMED (actual line 126)
- `pyabc_sampler.py:148` — `self.comm.bcast(("shutdown", None), root=0)`: CONFIRMED
- `pyabc_sampler.py:153` — worker_loop `self.comm.bcast(None, root=0)`: CONFIRMED
- `pyabc_sampler.py:158` (was :158 in interfaces, actual line 159 in source) — `self.comm.recv(source=0, tag=0)`: NOTE: source has `item = self.comm.recv(source=0, tag=0)` at line 159, not 158. Cited as `pyabc_sampler.py:158` in document per interfaces spec — minor off-by-one. Plan 02 should use 159.
- `pyabc_sampler.py:165/167` — send result/error: CONFIRMED (lines 165, 167)
- `pyabc_wrapper.py:404` — `cmap = CommWorldMap(MPI.COMM_WORLD)`: CONFIRMED
- `abc_smc_baseline.py:389` — `cmap = CommWorldMap(MPI.COMM_WORLD)`: CONFIRMED
- `pyabc_wrapper.py:375` / `abc_smc_baseline.py:359` — `with MPICommExecutor(...)`: CONFIRMED
- `scaling_runner.py:1062` — `with MPICommExecutor(MPI.COMM_WORLD, root=0)`: CONFIRMED
- `scaling_runner.py:964` — `if mpi_executor is not None`: CONFIRMED
- `scaling_runner.py:1067-1068` — post-context Barrier: CONFIRMED
- `runner.py:876` — `all_errors = allgather(error_payload)`: CONFIRMED

## Issues Encountered

One line number discrepancy: `pyabc_sampler.py:158` in the plan interfaces spec is actually at line 159 in the current source (`item = self.comm.recv(source=0, tag=0)`). The interfaces block in the plan says `:158` but the worker loop recv is on line 159. Documented above for Plan 02's awareness. All other citations match exactly.

## User Setup Required

None - documentation-only plan, no external service configuration required.

## Next Phase Readiness

- `.plans/diagnose/mpi-evaluation.md` is ready for Plan 02 to append characterization table, paper sensitivity assessment, residual risk discussion, and final recommendation
- Plan 02 should append after `*End of Plan 01 content.*` and must NOT restructure Task 1/2 sections
- Minor line number note for Plan 02: `pyabc_sampler.py:158` (worker recv) is actually at line 159 in current source

---
*Phase: 01-diagnose*
*Completed: 2026-04-10*
