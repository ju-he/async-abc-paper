# Phase 1: Diagnose - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-10
**Phase:** 01-diagnose
**Areas discussed:** Candidate scope, Inventory format, Verification method, Paper sensitivity scope

---

## Candidate Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Verify CommWorldMap + document others | CommWorldMap is the clear winner based on fix history. Diagnosis confirms it's correct, documents the two rejected alternatives with evidence from bug log. | |
| Full re-evaluation of all 3 paths | Treat CommWorldMap, MappingSampler+shared MPICommExecutor, and ConcurrentFutureSampler as open candidates. | ✓ |
| Focus only on CommWorldMap gaps | CommWorldMap is already decided. Phase 1 just documents what CommWorldMap does and writes test stubs. | |

**User's choice:** Full re-evaluation of all 3 paths

**Follow-up — 4th candidate:**

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, include native pyABC MappingSampler | Evaluate pyABC's native MPI path (MappingSampler backed by mpi4py built-in map) — closest to standard pyABC usage as required by MPI-02. | ✓ |
| No, 3 candidates is enough | CommWorldMap, shared MPICommExecutor mapping, and ConcurrentFutureSampler cover the relevant design space. | |

**User's choice:** Yes, include it as a 4th candidate

**Notes:** The 4th candidate is motivated by MPI-02's "correctness vs standard pyABC usage" criterion. Evaluate whether CommWorldMap is necessary or if pyABC's own recommendation would work.

---

## Inventory Format

| Option | Description | Selected |
|--------|-------------|----------|
| Markdown doc in .plans/ | Structured .plans/diagnose/mpi-evaluation.md with inventory, characterization table, and recommendation. | ✓ |
| Inline in pyabc_sampler.py | Structured docstring/comment block at the top of pyabc_sampler.py. | |
| Both: .plans/ doc + inline summary | Full analysis in .plans/, key decisions summarized inline. | |

**User's choice:** Markdown doc in .plans/ (recommended)

---

## Verification Method

| Option | Description | Selected |
|--------|-------------|----------|
| Code analysis + bug history | Static analysis + .plans/bug-fixes/previous-fixes.md evidence. No new runs needed. | ✓ |
| Local mpirun tests per candidate | Write/run small mpirun integration tests (2-4 ranks) for each candidate. | |
| Cluster run at 48 ranks | Submit JUWELS job to actually test candidates. | |

**User's choice:** Code analysis + bug history (recommended)

**Notes:** Phase 2 handles actual cluster verification. Phase 1 stays pure documentation.

---

## Paper Sensitivity Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Reasoning-level assessment | Argue from sampler semantics — does it affect which particles get accepted? Wall-time semantics? No numbers needed. | ✓ |
| Timing comparison from existing runs | Extract timing deltas from existing cluster log files. | |
| Full rerun with each candidate | Run short test-mode experiments with each candidate. Phase 2 scope. | |

**User's choice:** Reasoning-level assessment (recommended)

---

## Claude's Discretion

- Structure of per-candidate inventory sections
- Order of candidates in evaluation table
- Whether to include a "rejected paths" section

## Deferred Ideas

None.
