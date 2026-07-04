# Phase 4: Reproducibility - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-14
**Phase:** 04-reproducibility
**Areas discussed:** --extend correctness test, Seed determinism scope, One-command script (REPR-03), What 'same output' means

---

## --extend correctness test

| Option | Description | Selected |
|--------|-------------|----------|
| Pytest test: fresh vs extend diff | Run fresh, checkpoint mid-way, run extend from checkpoint, compare final CSVs. Add to test_extend.py. | ✓ |
| Pytest test: extend on full CSV is no-op | Verify extend when all combos complete produces identical output. Simpler but doesn't catch merge bugs. | |
| Both tests | No-op case AND fresh-vs-extend comparison. | |
| Manual audit doc | Written procedure, not automated. | |

**User's choice:** Pytest test: fresh vs extend diff
**Notes:** None

---

## Partial-run simulation approach

| Option | Description | Selected |
|--------|-------------|----------|
| Seed and write partial CSV manually | Construct partial CSV in fixture, run extend, compare. No actual crash needed. | ✓ |
| Run once, delete some rows, re-extend | Full run → delete rows → extend → compare to second fresh run. | |
| Mock/patch to crash mid-way | Inject failure after some combos. More realistic but complex. | |

**User's choice:** Seed and write partial CSV manually
**Notes:** None

---

## Seed determinism scope

| Option | Description | Selected |
|--------|-------------|----------|
| Single-process only: rejection_abc | Only rejection_abc is deterministic. MPI methods excluded. | ✓ |
| All methods, document limitations | Audit all methods, document which are/aren't deterministic. | |
| All methods, fix non-determinism | Seed MPI workers explicitly. High effort. | |

**User's choice:** Single-process only: rejection_abc
**Notes:** MPI methods (propulate_abc, pyabc) excluded — rank scheduling is non-deterministic by design on ParaStation MPI.

---

## Seed determinism verification form

| Option | Description | Selected |
|--------|-------------|----------|
| Pytest test: run twice, diff CSVs | Run rejection_abc twice same seed, assert identical. | ✓ |
| Audit doc only | Markdown listing which methods are deterministic. | |
| Both: test + audit doc | Automated test + short doc noting MPI non-determinism. | |

**User's choice:** Pytest test: run twice, diff CSVs
**Notes:** None

---

## One-command script (REPR-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Add output-existence checks to run_all | After each runner, verify expected CSVs exist and non-empty. Exit non-zero if missing. | ✓ |
| New standalone script | Separate verify_outputs.sh or verify_pipeline.py. | |
| Pytest test that calls run_all | Add pytest test in test_phase6.py calling orchestrator. | |

**User's choice:** Add output-existence checks to run_all
**Notes:** Phase 3 already confirmed exit-code checking. REPR-03 adds output-existence gate on top.

---

## What 'same output' means (REPR-01)

| Option | Description | Selected |
|--------|-------------|----------|
| Same rows (set equality, order-independent) | Set of (method, replicate, seed, params, loss) tuples identical. | ✓ |
| Bit-identical CSV files | Byte-for-byte identical including row order. | |
| Same summary statistics | Same mean/std per method/replicate. | |

**User's choice:** Same rows (set equality, order-independent)
**Notes:** Row order may vary between runs — strict byte diff is too fragile.

---

## Claude's Discretion

- Which CSV columns to include in row-set comparison
- Whether seed determinism test goes in test_seeding.py or test_extend.py
- Exact wording of output-existence failure messages

## Deferred Ideas

None.
