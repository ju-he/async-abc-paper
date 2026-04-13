# async-abc-paper

## What This Is

A research paper codebase implementing and evaluating an **asynchronous steady-state ABC-SMC algorithm** for HPC environments. The algorithm eliminates generation-level synchronization barriers by reconstructing the active particle archive from evaluated history, enabling continuous execution under heterogeneous and unreliable worker conditions. Experiments run on JUWELS (ParaStationMPI) comparing async Propulate-ABC against synchronous pyABC baselines under fixed wall-time budgets.

## Core Value

Experiments must run reliably to completion on the cluster — all paper results depend on this.

## Current Milestone: v1.0 Stability & Reproducibility

**Goal:** Harden the experiment framework so cluster runs with fixed wall-time budgets complete reliably, and critical code paths are covered by tests.

**Target features:**
- Diagnose and conclusively fix pyABC/MPI coordination hangs (CommWorldMap path needs verification)
- Improve test coverage for MPI coordination, wall-time stopping, and pyABC integration
- Code structure cleanup: simplify the MPI inference layer which has been heavily patched
- Reproducibility: ensure experiment configs, seeds, and output structures are well-defined

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- [x] MPI/pyABC hang issues diagnosed: all 4 sampler candidates characterized, CommWorldMap recommended as unhedged choice — Validated in Phase 01: diagnose

### Validated

- [x] MPI/pyABC hang issues fixed conclusively under cluster conditions — CommWorldMap verified at 48 ranks on JUWELS ParaStationMPI (no hang, 193+ records). `MPICommExecutor` removed from all code paths. Validated in Phase 02: mpi-hardening
- [x] Test coverage for CommWorldMap coordination, wall-time stopping (NaN-weight guard), and pyABC MPI paths — 8 regression tests across `test_mpi_hardening.py`. Validated in Phase 02: mpi-hardening

### Active

<!-- Current scope. Building toward these. -->

- [ ] MPI inference layer simplified and documented after multiple patch rounds
- [ ] Experiment reproducibility verified (configs, seeds, output structure consistent)

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- New experiments or benchmarks — focus is stability first, new features later
- LaTeX/paper writing — separate concern from code stability
- nastjapy / Cellular Potts changes — not related to the MPI hang issue

## Context

- **HPC environment:** JUWELS supercomputer (Forschungszentrum Juelich), ParaStationMPI, 48 tasks/node, SLURM, account `tissuetwin`
- **MPI hang history:** Multiple rounds of fixes (Apr 4–8 2026) around pyABC MPI teardown. Key fix was replacing `MPICommExecutor` with `CommWorldMap` (bcast/send/recv only, no inter-communicators). Still not confident this holds under all cluster conditions.
- **Wall-time stopping:** All methods stop at a fixed wall-time budget. This is the paper's primary comparison axis. Getting this right is critical — partial generations with NaN weights, async teardown races, and barrier timing have all been bug sources.
- **Inference layer state:** `pyabc_sampler.py`, `abc_smc_baseline.py`, `pyabc_wrapper.py`, and `propulate_abc.py` have been heavily patched. The code works but is complex and not well-tested against MPI edge cases.
- **Algorithm:** Propulate-ABC propagator (`propulate/propulate/propagators/`) is stateless — reconstructs archive and tolerance from evaluated history each call.

## Constraints

- **Tech stack:** Python ≥3.10, mpi4py, pyABC, Propulate 1.2.2, ParaStationMPI (JUWELS)
- **Test env:** Local tests use `nastjapy_copy/.venv`; MPI integration tests run with `mpirun`
- **Reproducibility:** Experiments use deterministic seeds from `seeding.py`; `--extend` mode reads existing CSVs to skip completed runs

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CommWorldMap replaces MPICommExecutor | `Create_intercomm`/`Disconnect` deadlocks ParaStation MPI at scale | ✓ Confirmed — 48-rank JUWELS PASS (Phase 02) |
| Wall-time as sole stopping criterion | Fixed budget comparison is the paper's primary claim | ✓ Good |
| mapping sampler over concurrent_futures | Blocking map avoids speculative future teardown hangs | ✓ Confirmed via CommWorldMap verification |
| max_wall_time_s injected into pyABC | Allows apples-to-apples wall-time comparison | ⚠️ Revisit (partial generations cause NaN weights) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-13 — Phase 02 (mpi-hardening) complete*
