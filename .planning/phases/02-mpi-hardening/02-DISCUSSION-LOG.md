# Phase 2: MPI Hardening - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.

**Date:** 2026-04-10
**Participants:** User + Claude

---

## Area 1: 48-Rank Verification Strategy

**Q: How should we verify CommWorldMap at 48 ranks (MPI-01)?**
Options: Cluster run first / Local mpirun stress test only / Both: local first, then cluster
**Selected:** Both: local first, then cluster

**Q: For the cluster run — which experiment type and config?**
Options: gaussian_mean population=100 / Use existing scaling test config
**Selected:** gaussian_mean, population=100 (Phase 1 recommended recipe)

**Q: Should the scaling runner migrate from Candidate 2 to CommWorldMap if verification passes?**
Options: Yes, migrate if verification passes / No, leave on Candidate 2
**Selected:** Yes, migrate if verification passes

---

## Area 2: Worker Crash / Liveness Handling

**Q: How should CommWorldMap handle a worker process crash?**
Options: Rely on SLURM job timeout / MPI_Testsome polling / Watchdog via OS signals
**Selected:** Rely on SLURM job timeout — document as known behavior, no code change

---

## Area 3: Wall-Time Stopping Hardening

**Q: How much to harden the NaN weight catch?**
Options: Add regression test, no code change / Harden catch guard first, then test / Wrap pyABC's wall-time callback
**Selected:** Add regression test, no code change

**Q: Does MPI-03 apply to both pyabc_smc and abc_smc_baseline?**
Options: Both paths, same test for each / abc_smc_baseline only
**Selected:** Both paths, same test for each

---

## Area 4: Test Scope and Priorities

**Q: Which Phase 1 test stubs to implement?**
Options (multiselect): NaN weight guard, CommWorldMap coordination, Barrier placement guard, Double-shutdown regression
**Selected:** All four

**Q: Barrier placement guard — mpirun hang detection or static source check?**
Options: Static source check / mpirun hang detection via timeout
**Selected:** Static source check (fast, no flakiness, CI-safe)

**Q: mpirun integration test rank count?**
Options: 2 ranks for fast + 4 ranks stress / 4 ranks only / 2 ranks only
**Selected:** 2 ranks for CI-facing tests, 4 ranks for stress variants (marked slow)
