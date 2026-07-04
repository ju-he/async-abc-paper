---
phase: 1
slug: diagnose
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-10
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | sim_backend_venv/.venv (project venv) |
| **Quick run command** | `sim_backend_venv/.venv/bin/python -m pytest tests/ -x -q` |
| **Full suite command** | `sim_backend_venv/.venv/bin/python -m pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `sim_backend_venv/.venv/bin/python -m pytest tests/ -x -q`
- **After every plan wave:** Run `sim_backend_venv/.venv/bin/python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | MPI-02 | manual | `ls .plans/diagnose/mpi-evaluation.md` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | MPI-04 | manual | `grep -c "newly discovered" .plans/diagnose/mpi-evaluation.md` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `.plans/diagnose/` — output directory for evaluation document
- [ ] `.plans/diagnose/mpi-evaluation.md` — target output file (created by task, not pre-existing)

*Existing test infrastructure covers runtime validation; Phase 1 output is a documentation artifact, so automated verification is file-existence checks.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Inventory completeness for all 3 candidate approaches | MPI-02 | Requires human judgement on coordination point coverage | Review `mpi-evaluation.md` for CommWorldMap, MappingSampler, MPICommExecutor sections |
| Recommendation quality | MPI-04 | Requires human review of evidence quality | Check recommendation section for concrete evidence citations from bug history |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
