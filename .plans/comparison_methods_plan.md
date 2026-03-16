# Comparison Methods Integration Plan

## Context

The async-ABC paper requires three comparison baselines to validate statistical accuracy and
computational performance against the primary `async_propulate_abc` method:

1. **`rejection_abc`** — simple rejection sampling, the most basic ABC baseline
2. **`pyabc_smc`** — already exists as a skeleton but has 4 correctness bugs that must be fixed
3. **`abc_smc_baseline`** — classical synchronous population ABC-SMC via pyABC, distinct from
   `pyabc_smc` in that it uses a fixed number of generations (`n_generations`) rather than an epsilon target

All 6 original TDD phases are complete (139 tests). This plan adds ~30 new tests and 2 new source
files while fixing the existing pyabc wrapper.

**Venv for testing:** `/home/juhe/bwSyncShare/Code/mirrors/nastjapy_copy/.venv` (has pyabc 0.12.17)

---

## Phase A: Fix `pyabc_wrapper.py` — ✅ COMPLETE

### Bugs fixed in `experiments/async_abc/inference/pyabc_wrapper.py`

- **Bug 1**: Added `sampler=` argument to `ABCSMC` (`SingleCoreSampler` or `MulticoreEvalParallelSampler`)
- **Bug 2**: Replaced non-deterministic `rng_counter` closure with param-hash seed + local `_distance_cache`
- **Bug 3**: Fixed `loss` field: stores actual distance from `_distance_cache`, not generation epsilon
- **Bug 4**: Fixed epsilon extraction via `all_pops.set_index("t")["epsilon"]` (robust to row ordering)
- **Bug 5**: Fixed `w.iloc[i]` positional indexing bug using `enumerate(df.iterrows())`

### Tests: `TestPyabcWrapperFixes` (7 tests in `test_inference.py`) — ✅

**Note:** pyabc requires `async-timeout` which was missing from the nastjapy_copy venv.
Fixed by running: `pip install async-timeout` in that venv.

---

## Phase B: Add `rejection_abc` — ✅ COMPLETE

### New file: `experiments/async_abc/inference/rejection_abc.py`

Pure numpy implementation (no pyABC dependency). Draws from prior, simulates, accepts if
`loss <= tol_init`. Stops at `max_simulations` or `k` accepted particles.

### Tests: `TestRejectionAbc` (8 tests in `test_inference.py`) — ✅

### Registry: added `"rejection_abc": run_rejection_abc` to `method_registry.py`

---

## Phase C: Add `abc_smc_baseline` — ✅ COMPLETE

### New file: `experiments/async_abc/inference/abc_smc_baseline.py`

Uses pyABC with `max_nr_populations=n_generations` (default 5). Key difference from `pyabc_smc`:
fixed generation count rather than epsilon target. Same distance-cache and sampler patterns.

```python
history = abc.run(
    minimum_epsilon=0.0,
    max_total_nr_simulations=max_sims,
    max_nr_populations=n_generations,
)
```

### Tests: `TestAbcSmcBaseline` (9 tests in `test_inference.py`) — ✅

### Registry: added `"abc_smc_baseline": run_abc_smc_baseline` to `method_registry.py`

---

## Phase D: Config + Schema + Integration — ✅ COMPLETE

### `experiments/async_abc/io/schema.py`

Added `n_generations: 3` clamp to `TEST_MODE_OVERRIDES["clamp"]["inference"]`.

### `experiments/configs/gaussian_mean.json`

- Added `"rejection_abc"` and `"abc_smc_baseline"` to `"methods"`
- Added `"n_generations": 5` to `"inference"`

### `experiments/configs/gandk.json`

- Added `"rejection_abc"` and `"abc_smc_baseline"` to `"methods"`
- Added `"n_generations": 5` to `"inference"`

### New tests (6 tests in `test_config.py` and `test_phase6.py`)

---

## Summary of All Changes

| File | Action |
|------|--------|
| `async_abc/inference/pyabc_wrapper.py` | Fixed 5 bugs |
| `async_abc/inference/rejection_abc.py` | Created (pure numpy) |
| `async_abc/inference/abc_smc_baseline.py` | Created (pyABC wrapper) |
| `async_abc/inference/method_registry.py` | 2 new imports + 2 registry entries |
| `async_abc/io/schema.py` | `n_generations: 3` in TEST_MODE_OVERRIDES |
| `experiments/configs/gaussian_mean.json` | 2 new methods + `n_generations` |
| `experiments/configs/gandk.json` | 2 new methods + `n_generations` |
| `experiments/tests/test_inference.py` | 24 new tests (3 classes) |
| `experiments/tests/test_config.py` | 3 new tests |
| `experiments/tests/test_phase6.py` | 3 new tests |

**Total: 41 new tests → final suite 180 tests (all passing)**

---

## Verification

```bash
cd /home/juhe/bwSyncShare/Code/async-abc-paper/experiments

# Full test suite
/home/juhe/bwSyncShare/Code/mirrors/nastjapy_copy/.venv/bin/pytest tests/ -v

# End-to-end runner smoke test
/home/juhe/bwSyncShare/Code/mirrors/nastjapy_copy/.venv/bin/python \
  scripts/gaussian_mean_runner.py \
  --config configs/gaussian_mean.json \
  --output-dir /tmp/test_comparison \
  --test
```
