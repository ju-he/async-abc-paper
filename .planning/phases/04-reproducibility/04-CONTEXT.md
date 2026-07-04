# Phase 4: Reproducibility - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify that `--extend` mode produces correct results, confirm seed determinism for single-process methods, and add output-existence checks to the one-command pipeline runner. No new experiments, no new benchmarks, no MPI changes.

</domain>

<decisions>
## Implementation Decisions

### --extend correctness verification (REPR-01)

- **D-01:** Add a pytest test to `test_extend.py` that verifies `--extend` produces the same final rows as a fresh run with the same seed. Approach: construct a partial CSV fixture manually (write rows for some method/replicate combos, leave others missing), run extend, assert the resulting CSV matches a fresh run on the same config+seed.
- **D-02:** "Same output" means set equality of rows (order-independent). Compare the set of `(method, replicate, seed, params, loss)` tuples from the extend result against a fresh run. Row order may differ between runs — strict byte-level diff is too fragile.
- **D-03:** The partial-run checkpoint is simulated by constructing a partial CSV directly in the test fixture (no mid-run crash injection needed). Write a subset of expected rows, leave the rest missing, then run extend.

### Seed determinism audit (REPR-02)

- **D-04:** Scope is single-process methods only — `rejection_abc` is the only fully deterministic method. MPI-based methods (`propulate_abc`, `pyabc` variants) are excluded: rank scheduling on ParaStation MPI introduces non-determinism by design.
- **D-05:** Verification is a pytest test: run `rejection_abc` twice with the same seed in test mode, assert the resulting CSVs are set-equal (same row sets). Add to `test_seeding.py` or `test_extend.py` as appropriate.
- **D-06:** No audit document needed for REPR-02 — the pytest test is the verification artifact.

### One-command pipeline verification (REPR-03)

- **D-07:** REPR-03 is satisfied by adding output-existence checks to `run_all_paper_experiments.py`. After each runner finishes, verify expected output files (CSVs) exist and are non-empty. Exit non-zero if any are missing. No new standalone script.
- **D-08:** Phase 3 already confirmed all 11 runners pass `--test` mode (TEST-02 satisfied). REPR-03 adds the output-existence gate on top of exit-code checking.

### Claude's Discretion

- Which specific CSV columns to include in the row-set comparison (D-02) — use all numeric/key columns, exclude any metadata-only fields that may vary
- Whether the seed determinism test (D-05) goes in `test_seeding.py` or `test_extend.py`
- Exact wording of output-existence failure messages in `run_all_paper_experiments.py`

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Extend mode implementation
- `experiments/async_abc/utils/runner.py` — `find_completed_combinations()` and extend logic. Read before writing extend tests.
- `experiments/tests/test_extend.py` — Existing extend test suite (266 lines). New D-01 test goes here.

### Seeding
- `experiments/async_abc/utils/seeding.py` — `stable_seed()`, `make_seeds()`, `canonical_param_key()`. Read before writing seed determinism test.
- `experiments/tests/test_seeding.py` — Existing seeding tests (62 lines). D-05 test may go here.

### One-command orchestrator
- `experiments/run_all_paper_experiments.py` — Top-level orchestrator with `EXPERIMENT_REGISTRY`. D-07/D-08 changes go here.

### Requirements
- `.planning/REQUIREMENTS.md` §REPR-01, §REPR-02, §REPR-03 — requirements this phase satisfies

### Bug history (consult before touching runner or seeding code)
- `.plans/bug-fixes/previous-fixes.md` — All prior MPI and runner bug fixes

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `find_completed_combinations()` in `runner.py` — already parses completed (method, replicate) key tuples from CSV; extend test fixture can write rows that exercise this path
- `stable_seed()` / `make_seeds()` in `seeding.py` — deterministic seeding already in place; seed determinism test can use `base_seed=1` and run same config twice
- `conftest.py` `make_fast_runner_config()` and `run_runner_main()` — test infrastructure for running runners end-to-end in tests; D-01 and D-05 tests should use these

### Established Patterns
- `test_extend.py` fixture pattern: `extend_runner_config_file` builds a config with `rejection_abc` + `timed_fake`, uses `patched_method_registry` to inject `timed_fake` — D-01 test should follow the same pattern
- Row comparison in tests uses `_read_key_tuples()` and set comparison — D-02 set equality matches this existing pattern

### Integration Points
- `run_all_paper_experiments.py` already calls each runner's `main()` and checks exit codes — D-07 output-existence check should hook in after the exit-code check, before moving to the next experiment

</code_context>

<specifics>
## Specific Ideas

- The partial-CSV fixture for D-01 should use the `gaussian_mean` benchmark (already used in `test_extend.py`) with `rejection_abc` and `timed_fake` — no new benchmark setup needed
- For D-02, "same row set" comparison should normalize floats (use `pytest.approx` or round to N decimals) since loss values come from the same deterministic RNG path — but if truly bit-identical from same seed, exact equality should hold for `rejection_abc`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 04-reproducibility*
*Context gathered: 2026-04-14*
