# Roadmap: async-abc-paper

## Overview

Milestone v1.0 Stability & Reproducibility hardens the experiment framework so cluster runs complete reliably. The work proceeds in four phases: first diagnose what is actually broken, then fix and test the MPI coordination layer, then clean up the heavily-patched inference code, and finally verify reproducibility of experiment outputs. Every phase delivers a verifiable state of the codebase — not just tasks completed.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Diagnose** - Systematically document all rank coordination points and evaluate all candidate pyABC MPI sampler approaches; select the best one (completed 2026-04-10)
- [x] **Phase 2: MPI Hardening** - Implement the chosen MPI approach, verify it at scale, make wall-time stopping robust, backed by unit tests (completed 2026-04-13)
- [ ] **Phase 3: Code Cleanup** - Simplify and document the inference layer after the MPI model is fully understood
- [ ] **Phase 4: Reproducibility** - Verify extend mode, seed determinism, and one-command end-to-end test script

## Phase Details

### Phase 1: Diagnose
**Goal**: All rank coordination points are documented and all candidate pyABC MPI sampler approaches are evaluated — the best approach selected with evidence
**Depends on**: Nothing (first phase)
**Requirements**: MPI-02, MPI-04
**Success Criteria** (what must be TRUE):
  1. A written inventory of every point where ranks synchronize (bcast, send/recv, allgather, barrier) exists for each candidate MPI approach (CommWorldMap, MappingSampler, MPICommExecutor/ConcurrentFutures)
  2. Each approach is characterized for: cluster stability, correctness vs standard pyABC usage, effect on paper results — with a clear recommendation
  3. Any newly discovered hang paths are documented with a reproduction recipe or test stub
**Plans**: 2 plans
Plans:
- [x] 01-01-PLAN.md — Write per-candidate MPI coordination point inventory (all 4 candidates) to .plans/diagnose/mpi-evaluation.md
- [x] 01-02-PLAN.md — Append characterization table, paper sensitivity, residual risks, and unhedged recommendation; human-verify the complete deliverable

### Phase 2: MPI Hardening
**Goal**: The chosen pyABC MPI approach (selected in Phase 1) is implemented, does not hang at 48 ranks, and pyABC stops cleanly on wall-time without losing completed particle data
**Depends on**: Phase 1
**Requirements**: MPI-01, MPI-03, TEST-01, TEST-03
**Success Criteria** (what must be TRUE):
  1. A staged or instrumented cluster run at 48 ranks completes without hanging (or a local mpirun stress test passes all coordination paths) using the chosen approach
  2. Stopping pyABC mid-generation on wall-time deadline produces a CSV with all pre-deadline particles and no NaN weights
  3. MPI unit tests for the chosen sampler path and wall-time stopping run locally with mpirun and pass
  4. Regression tests covering NaN weight crash, double shutdown, and barrier timing races are present and passing
**Plans**: 3 plans
Plans:
- [x] 02-01-PLAN.md — Create MPI hardening test suite (NaN weight regression, CommWorldMap coordination, barrier source check, double-shutdown) — satisfies TEST-01, TEST-03, MPI-03
- [x] 02-02-PLAN.md — 48-rank cluster verification: SLURM script + Python driver + human checkpoint — satisfies MPI-01
- [x] 02-03-PLAN.md — Conditional scaling_runner migration to CommWorldMap (D-03, only if 02-02 PASSes) — completes MPI-01 unification

### Phase 3: Code Cleanup
**Goal**: The MPI inference layer is simplified, documented, and free of dead code so future patches can be made safely
**Depends on**: Phase 2
**Requirements**: CODE-01, CODE-02, CODE-03, TEST-02
**Success Criteria** (what must be TRUE):
  1. pyabc_sampler.py, abc_smc_baseline.py, and pyabc_wrapper.py have no commented-out legacy blocks or concurrent_futures fallback paths
  2. The CommWorldMap design, rank protocol, and known failure modes are described in inline comments sufficient to understand the coordination model without git history
  3. All experiment runners pass --test end-to-end in a single command without errors
**Plans**: 3 plans
- [x] 03-01-PLAN.md — Remove dead MPICommExecutor / concurrent_futures paths from the three inference files (CODE-01, CODE-03)
- [ ] 03-02-PLAN.md — Document CommWorldMap coordination model, rank protocol, and failure modes (CODE-02)
- [x] 03-03-PLAN.md — End-to-end orchestrator smoke under --test (TEST-02)

### Phase 4: Reproducibility
**Goal**: Experiment outputs are deterministic and the full pipeline can be verified in one command
**Depends on**: Phase 3
**Requirements**: REPR-01, REPR-02, REPR-03
**Success Criteria** (what must be TRUE):
  1. Running --extend on an existing CSV produces the same final outputs as a fresh run with the same seed (no silent incorrect merges)
  2. Running all benchmarks twice with the same seed produces bit-identical CSVs
  3. A single script invocation runs all runners in test mode and checks that output files exist and are non-empty
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Diagnose | 2/2 | Complete   | 2026-04-10 |
| 2. MPI Hardening | 3/3 | Complete   | 2026-04-13 |
| 3. Code Cleanup | 2/3 | In Progress|  |
| 4. Reproducibility | 0/? | Not started | - |
