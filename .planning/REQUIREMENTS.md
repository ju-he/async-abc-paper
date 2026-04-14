# Requirements: async-abc-paper

**Defined:** 2026-04-10
**Core Value:** Experiments must run reliably to completion on the cluster — all paper results depend on this.

## v1 Requirements

### MPI Stability

- [x] **MPI-01**: pyABC MPI coordination does not hang at 48 ranks — the chosen approach (CommWorldMap, mapping sampler, MPICommExecutor, or other) is verified via staged test or instrumented run
- [x] **MPI-02**: All pyABC MPI sampler options (CommWorldMap, MappingSampler, ConcurrentFutureSampler/MPICommExecutor) are evaluated for correctness, cluster stability, and closeness to standard pyABC usage; the best approach is selected with rationale; paper conclusions assessed for sensitivity to sampler choice
- [x] **MPI-03**: pyABC stops cleanly on wall-time mid-generation without losing completed data (beyond current catch-and-recover)
- [x] **MPI-04**: Remaining hang paths diagnosed systematically — all rank coordination points documented and tested for each candidate MPI approach

### Test Coverage

- [x] **TEST-01**: MPI unit tests exist for CommWorldMap, pyabc_sampler, and wall-time stopping (locally runnable with mpirun)
- [x] **TEST-02**: All experiment runners pass `--test` end-to-end in a single command
- [x] **TEST-03**: Regression tests cover documented bugs (NaN weight, double shutdown, barrier timing races)

### Code Structure

- [x] **CODE-01**: pyabc_sampler.py, abc_smc_baseline.py, pyabc_wrapper.py simplified after multiple patch rounds
- [x] **CODE-02**: MPI coordination model documented inline (CommWorldMap design, rank protocol, known failure modes)
- [x] **CODE-03**: Dead/legacy code removed (concurrent_futures_legacy paths, obsolete workarounds)

### Reproducibility

- [ ] **REPR-01**: `--extend` mode verified to produce correct results (no silent incorrect merges)
- [ ] **REPR-02**: Config/seed audit confirms deterministic outputs for the same seed across all benchmarks
- [ ] **REPR-03**: One-command end-to-end test script runs all runners in test mode and verifies outputs exist

## v2 Requirements

### Future

- New experiments (scaling at >256 ranks, new benchmarks)
- Cellular Potts improvements
- Paper-ready figure polish

## Out of Scope

| Feature | Reason |
|---------|--------|
| New experiments / benchmarks | Focus is stability; new features come after cluster runs are reliable |
| LaTeX / paper writing | Separate concern from code stability |
| nastjapy / Cellular Potts internal changes | Not related to MPI hang issues |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MPI-02 | Phase 1 | Complete |
| MPI-04 | Phase 1 | Complete |
| MPI-01 | Phase 2 | Complete |
| MPI-03 | Phase 2 | Complete |
| TEST-01 | Phase 2 | Complete |
| TEST-03 | Phase 2 | Complete |
| CODE-01 | Phase 3 | Complete |
| CODE-02 | Phase 3 | Complete |
| CODE-03 | Phase 3 | Complete |
| TEST-02 | Phase 3 | Complete |
| REPR-01 | Phase 4 | Pending |
| REPR-02 | Phase 4 | Pending |
| REPR-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13 (100%)
- Unmapped: 0

---
*Requirements defined: 2026-04-10*
*Last updated: 2026-04-10 — traceability populated after roadmap creation*
