---
phase: 04-reproducibility
verified: 2026-04-14T18:30:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 4: Reproducibility Verification Report

**Phase Goal:** Reproducibility — tests proving extend idempotency, seeding determinism, and output-existence gate for the orchestrator.
**Verified:** 2026-04-14T18:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A pytest test exists that verifies `--extend` on a partial CSV produces the same final row set as a fresh run with the same seed | VERIFIED | `test_extend_matches_fresh_run_same_seed` at line 167 of `experiments/tests/test_extend.py`, inside `TestExtendBasicRunner` |
| 2 | The extend test writes a partial CSV fixture, runs extend, and asserts row-set equality against a fresh run | VERIFIED | Lines 184-220: partial CSV written (rejection_abc replicate=0 only), `--extend` invoked, `assert extend_set == fresh_set` on key_cols present |
| 3 | A pytest test verifies that running `rejection_abc` twice with the same seed produces identical output row sets | VERIFIED | `TestRunnerDeterminism::test_rejection_abc_same_seed_produces_same_rows` at line 72 of `experiments/tests/test_seeding.py`; `assert set_a == set_b` at line 115 |
| 4 | The seeding test is scoped to single-process methods only (rejection_abc, no MPI methods) | VERIFIED | `grep -c "propulate\|pyabc\|abc_smc_baseline" test_seeding.py` returns 0; config sets `methods=["rejection_abc"]` only |
| 5 | After each runner exits with rc=0, the orchestrator verifies at least one non-empty CSV exists | VERIFIED | `_verify_outputs_exist` at lines 57-83 of `run_all_paper_experiments.py`; called at line 250 when `rc == 0 and is_root_rank()` |
| 6 | If expected outputs are missing or empty, the runner is added to `failures` and orchestrator exits non-zero | VERIFIED | `elif not outputs_ok: failures.append(name)` at lines 255-256; existing `if failures: sys.exit(1)` path unchanged |
| 7 | A pytest test proves the gate flags a runner whose outputs are missing (negative path) | VERIFIED | `test_orchestrator_fails_when_output_missing` at line 362 of `test_extend.py`; asserts `SystemExit(1)` |
| 8 | A pytest test proves the gate passes when outputs are present (positive path) | VERIFIED | `test_orchestrator_succeeds_when_outputs_present` at line 414 of `test_extend.py`; no SystemExit raised, CSVs confirmed non-empty |
| 9 | All relevant test suites pass with no regressions | VERIFIED | 29 tests across `test_extend.py` (18) and `test_seeding.py` (11) all pass |

**Score:** 9/9 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `experiments/tests/test_extend.py` | Extend-vs-fresh equivalence test (REPR-01) + output gate tests (REPR-03) | VERIFIED | Contains `test_extend_matches_fresh_run_same_seed` (line 167) inside `TestExtendBasicRunner`, and `TestOrchestratorOutputGate` class (lines 326-447) with 6 tests |
| `experiments/tests/test_seeding.py` | Seed determinism test for rejection_abc (REPR-02) | VERIFIED | Contains `TestRunnerDeterminism::test_rejection_abc_same_seed_produces_same_rows` (lines 71-121); pre-existing `TestMakeSeeds` and `TestSeedEverything` classes untouched |
| `experiments/run_all_paper_experiments.py` | Output-existence gate wired into per-experiment loop (REPR-03) | VERIFIED | `_verify_outputs_exist` helper at lines 57-83; gate wired at lines 247-256 after `write_timing_csv`, only when `rc==0 and is_root_rank()`; `elif not outputs_ok:` prevents double-counting |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `test_extend_matches_fresh_run_same_seed` | extend-mode runner invocation | `extra_args=("--extend",)` | WIRED | Line 207: `extra_args=("--extend",)` passed to `run_runner_main` |
| `test_extend_matches_fresh_run_same_seed` | two independent tmp_path runner outputs | set equality on (method, replicate, seed, step, loss) | WIRED | Lines 218-220: `fresh_set` and `extend_set` compared with `assert extend_set == fresh_set` |
| `test_rejection_abc_same_seed_produces_same_rows` | two independent gaussian_mean_runner.py invocations with base_seed=1 | set equality of row tuples | WIRED | Lines 87-115: two runs into `dir_a`/`dir_b`, `assert set_a == set_b` |
| `main() per-experiment loop` | `_verify_outputs_exist(name, output_dir)` | called after `write_timing_csv` when rc=0 | WIRED | Lines 249-252: `if rc == 0 and is_root_rank(): outputs_ok, outputs_reason = _verify_outputs_exist(name, output_dir)` |
| `_verify_outputs_exist` | `failures` list | appends name with reason when no non-empty CSV | WIRED | Lines 255-256: `elif not outputs_ok: failures.append(name)` |

---

### Data-Flow Trace (Level 4)

Not applicable — phase produces test infrastructure and a helper function, not data-rendering components. The test artifacts assert real data flows (CSV row sets from actual runner invocations) by design.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| REPR-01: extend-vs-fresh test passes | `pytest test_extend.py::TestExtendBasicRunner::test_extend_matches_fresh_run_same_seed -x` | 1 passed in 4.10s | PASS |
| REPR-02: seeding determinism test passes | `pytest test_seeding.py::TestRunnerDeterminism::test_rejection_abc_same_seed_produces_same_rows -x` | 1 passed in 1.08s | PASS |
| REPR-03: output gate tests pass (all 6) | `pytest test_extend.py::TestOrchestratorOutputGate -x` | 6 passed in 1.12s | PASS |
| Full suite no-regression | `pytest test_extend.py test_seeding.py` | 29 passed in 2.65s | PASS |
| Orchestrator syntax valid | `python -c "import ast; ast.parse(...); print('syntax ok')"` | syntax ok | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| REPR-01 | 04-01-PLAN.md | `--extend` mode verified to produce correct results (no silent incorrect merges) | SATISFIED | `test_extend_matches_fresh_run_same_seed` passes; partial CSV + extend produces identical row set to fresh run |
| REPR-02 | 04-02-PLAN.md | Config/seed audit confirms deterministic outputs for the same seed across all benchmarks (scoped: single-process only per D-04) | SATISFIED | `test_rejection_abc_same_seed_produces_same_rows` passes; two runs with same seed produce equal row sets on (method, replicate, seed, step, loss) |
| REPR-03 | 04-03-PLAN.md | One-command end-to-end test script runs all runners in test mode and verifies outputs exist | SATISFIED | `_verify_outputs_exist` gate active in orchestrator; negative-path test confirms `SystemExit(1)` when outputs missing; positive-path test confirms no false positive |

No orphaned requirements — all three REPR IDs declared in plans are accounted for, and REQUIREMENTS.md maps exactly these three to Phase 4.

---

### Anti-Patterns Found

No blocking anti-patterns. Scanned modified files for TODO/FIXME/placeholder comments, empty returns, and stub patterns — none found. The one notable deviation from the plan (fake runner config needing a valid schema rather than `{}`) was auto-fixed and documented in 04-03-SUMMARY.md; the resulting implementation is correct and substantive.

---

### Human Verification Required

None. All behavioral properties of this phase are verifiable programmatically via pytest. The tests themselves are the reproducibility contract.

---

## Gaps Summary

No gaps. All must-haves verified at all levels:

- **REPR-01** (extend idempotency): Test exists, is substantive, runs end-to-end, and passes.
- **REPR-02** (seeding determinism): Test exists, is substantive, is scoped to single-process methods per D-04, runs end-to-end, and passes.
- **REPR-03** (output-existence gate): Helper and wiring exist in the orchestrator, are substantive and correctly placed (after `write_timing_csv`, only on root rank, `elif` prevents double-counting), 6 pytest tests cover all branches and pass.

The phase goal is fully achieved.

---

_Verified: 2026-04-14T18:30:00Z_
_Verifier: Claude (gsd-verifier)_
