---
phase: 3
slug: code-cleanup
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-14
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing, in sim_backend_venv/.venv) |
| **Config file** | none — pytest auto-discovers experiments/tests/ |
| **Quick run command** | `sim_backend_venv/.venv/bin/python -m pytest experiments/tests/test_inference.py experiments/tests/test_mpi_hardening.py -x -q` |
| **Full suite command** | `sim_backend_venv/.venv/bin/python -m pytest experiments/tests/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `sim_backend_venv/.venv/bin/python -m pytest experiments/tests/test_inference.py experiments/tests/test_mpi_hardening.py -x -q`
- **After every plan wave:** Run `sim_backend_venv/.venv/bin/python -m pytest experiments/tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green + orchestrator smoke pass
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 0 | CODE-03 | unit | `pytest experiments/tests/test_inference.py::TestBuildPyabcSampler -x -q` | ✅ | ⬜ pending |
| 3-01-02 | 01 | 1 | CODE-01/CODE-03 | unit | `pytest experiments/tests/test_inference.py -x -q` | ✅ | ⬜ pending |
| 3-01-03 | 01 | 1 | CODE-01/CODE-03 | unit | `pytest experiments/tests/test_inference.py -x -q` | ✅ | ⬜ pending |
| 3-01-04 | 01 | 1 | CODE-01/CODE-03 | unit | `pytest experiments/tests/test_inference.py -x -q` | ✅ | ⬜ pending |
| 3-02-01 | 02 | 1 | CODE-02 | manual | review docstring | N/A | ⬜ pending |
| 3-03-01 | 03 | 1 | TEST-02 | smoke | `sim_backend_venv/.venv/bin/python experiments/run_all_paper_experiments.py --test --output-dir /tmp/test_paper_results` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `experiments/tests/test_inference.py::TestBuildPyabcSampler::test_resolve_pyabc_mpi_sampler_honors_legacy_alias_with_warning` — update to assert `ValueError` instead of warning (dead alias after CODE-03 removal)

*Note: Existing test infrastructure covers the rest — only the legacy alias test needs updating before removal tasks run.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CommWorldMap docstring accuracy | CODE-02 | Content review required | Read CommWorldMap class docstring in pyabc_sampler.py; verify it covers coordination model, rank protocol, and known failure modes |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
