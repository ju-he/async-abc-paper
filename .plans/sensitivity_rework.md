# Sensitivity Experiment Rework — TDD Plan

## Context

The sensitivity experiment has seven concrete problems identified in review:

1. **Wrong metric**: "mean final tolerance" conflates convergence with starting point when `tol_init` is a grid axis. Replace with Wasserstein distance at fixed simulation budget.
2. **`scheduler_type` in the grid**: a discrete algorithmic choice treated as a continuous hyperparameter. Separate into per-scheduler facet figures.
3. **Benchmark too easy**: only `gaussian_mean` — insensitive to k and perturbation_scale. Add `gandk` as the primary sensitivity benchmark.
4. **Grid mismatch**: paper-concept specifies `tol_init_multiplier = [0.5, 1.0, 2.0, 5.0]`; config has `[0.5, 1.0, 5.0]`.
5. **No uncertainty**: heatmap shows a point estimate with no cross-replicate spread.
6. **Fragile "last 10%" heuristic**: uses row count, not simulation budget position.
7. **`n_workers=48` hardcoded** in the inference config — misleads readers.

All phases follow strict TDD: write the failing test first, then implement, then refactor.

**Test runner**: `nastjapy_copy/.venv/bin/pytest experiments/tests/ -x -q`

---

## Phase 1 — Replace the summary metric with posterior quality ✅ COMPLETE

### What to test first

**File**: `experiments/tests/test_sensitivity_metric.py` (new file)

Tests to write (all failing before implementation):

```python
class TestSensitivityQualitySummary:
    def test_returns_dataframe_with_wasserstein_column(...)
        # Call compute_sensitivity_quality_summary() with synthetic CSVs
        # Assert result has columns: variant_name, wasserstein, wasserstein_std, n_replicates
        # + one column per grid key

    def test_wasserstein_uses_final_budget_window(...)
        # Two synthetic CSV files: one whose last N_sim/10 particles have low loss,
        # one with high loss throughout.
        # Assert the low-loss variant has lower wasserstein.

    def test_tol_init_variants_comparable_when_wasserstein_used(...)
        # Critical regression test: two variants differing only in tol_init_multiplier
        # but with identical posterior quality should yield similar wasserstein scores.
        # (Tolerance-based metric would make them look very different.)

    def test_per_replicate_std_is_computed(...)
        # CSV with 3 replicates having distinct seeds.
        # Assert wasserstein_std > 0.

    def test_missing_variant_csv_produces_nan_row(...)
        # Grid says variant exists but file is absent → row with NaN, no crash.
```

### Implementation

**New function** in `experiments/async_abc/analysis/sensitivity.py` (new file):

```python
def compute_sensitivity_quality_summary(
    data_dir: Path,
    grid: dict,
    true_params: dict,          # benchmark.true_params
    max_simulations: int,
    tail_fraction: float = 0.1, # fraction of budget to call "final"
) -> pd.DataFrame:
    ...
```

Steps:
1. For each variant CSV, load records.
2. Group by replicate.
3. For each replicate, take the last `max(1, int(max_simulations * tail_fraction))` rows by `step` (not row count — fixes issue 6).
4. Extract `param_*` columns as posterior samples.
5. Compute Wasserstein against `true_params` using the existing `_wasserstein_to_true_params()` from `analysis/convergence.py`.
6. Aggregate mean ± std across replicates.
7. Return DataFrame with one row per variant.

**Modify `plot_sensitivity_summary`** in `reporters.py`:
- Accept an optional `quality_df: pd.DataFrame | None` argument.
- If provided, use `wasserstein` column instead of the raw-tolerance computation.
- Preserve the old path as a fallback for backward compatibility (deprecated).

**Modify `sensitivity_runner.py`**:
- After all variant runs complete, call `compute_sensitivity_quality_summary()`.
- Write result to `output_dir.data / "sensitivity_quality_summary.csv"`.
- Pass it to `plot_sensitivity_summary()`.

### Verification

```bash
nastjapy_copy/.venv/bin/pytest experiments/tests/test_sensitivity_metric.py -x -q
nastjapy_copy/.venv/bin/pytest experiments/tests/test_runners.py::TestSensitivityRunner -x -q
nastjapy_copy/.venv/bin/pytest experiments/tests/test_plotting.py -k sensitivity -x -q
```

---

## Phase 2 — Separate `scheduler_type` from the sensitivity sweep ✅ COMPLETE

### What to test first

**In `experiments/tests/test_plotting.py`** (extend existing class):

```python
class TestSensitivityHeatmapPerScheduler:
    def test_generates_one_figure_per_scheduler(...)
        # Grid with 2 scheduler types and 2x2 k/perturbation_scale.
        # Call plot_sensitivity_summary().
        # Assert two PDF files exist:
        #   sensitivity_heatmap__scheduler_type=acceptance_rate.pdf
        #   sensitivity_heatmap__scheduler_type=quantile.pdf
        # Assert single combined sensitivity_heatmap.pdf does NOT exist.

    def test_scheduler_figure_has_correct_dimensions(...)
        # Each per-scheduler CSV should have n_tol_init_levels rows × n_k × n_perturb columns.

    def test_missing_scheduler_data_skips_file(...)
        # If no data exists for one scheduler, that file is skipped without crashing.
```

### Implementation

**Modify `plot_sensitivity_summary`**:
- If `scheduler_type` is in the grid, loop over its values.
- For each scheduler value, generate a separate figure:
  - filename: `sensitivity_heatmap__scheduler_type={value}.pdf/png`
  - rows=k, cols=perturbation_scale, facet=tol_init_multiplier
- Remove the 4D combined plot path (the 3×3 of 3×3 layout).
- The combined file is now only generated when `scheduler_type` is NOT in the grid.

**Config change** (`sensitivity.json`):
- Keep `scheduler_type` in the grid for now (data collection unchanged).
- Visualisation separates it automatically.

### Verification

```bash
nastjapy_copy/.venv/bin/pytest experiments/tests/test_plotting.py::TestSensitivityHeatmapPerScheduler -x -q
```

---

## Phase 3 — Add uncertainty display to heatmap cells ✅ COMPLETE

### What to test first

**In `experiments/tests/test_plotting.py`**:

```python
class TestSensitivityHeatmapUncertainty:
    def test_csv_output_contains_std_column(...)
        # After plot_sensitivity_summary(), the exported CSV has a wasserstein_std column
        # (or mean_tol_std if still using tolerance).

    def test_heatmap_metadata_records_n_replicates(...)
        # Metadata JSON contains n_replicates used per cell.

    def test_single_replicate_std_is_zero_or_nan(...)
        # With one replicate, std column should be 0 or NaN, not crash.
```

**In `experiments/tests/test_sensitivity_metric.py`**:

```python
    def test_three_replicates_produce_nonzero_std(...)
        # Three replicates with slightly different seeds → wasserstein_std > 0.
```

### Implementation

**Modify `sensitivity_heatmap()`** in `common.py`:
- Accept optional `std_matrix` of same shape as `matrix`.
- If provided, annotate each cell with ±std (small font below the mean).
- Extend the exported CSV to include a `wasserstein_std` column alongside `wasserstein_mean`.

**Modify `compute_sensitivity_quality_summary()`**:
- Already returns `wasserstein_std` (from Phase 1 implementation).

### Verification

```bash
nastjapy_copy/.venv/bin/pytest experiments/tests/test_plotting.py -k uncertainty -x -q
nastjapy_copy/.venv/bin/pytest experiments/tests/test_sensitivity_metric.py -x -q
```

---

## Phase 4 — Config fixes and benchmark upgrade ✅ COMPLETE

### What to test first

**In `experiments/tests/test_config.py`** (extend existing class):

```python
class TestSensitivityConfig:
    def test_tol_init_multiplier_has_four_levels(...)
        # cfg["sensitivity_grid"]["tol_init_multiplier"] has 4 elements
        # including 2.0.

    def test_sensitivity_config_has_gandk_variant(...)
        # A sensitivity_gandk.json exists in configs/
        # and its benchmark.name == "gandk".

    def test_gandk_sensitivity_has_adequate_budget(...)
        # sensitivity_gandk.json has max_simulations >= 20000
        # (g-and-k needs more budget than gaussian_mean).

    def test_n_workers_not_in_sensitivity_inference(...)
        # Neither sensitivity.json nor sensitivity_gandk.json
        # has "n_workers" as a top-level inference key.
```

### Implementation

1. **`sensitivity.json`**: Add `2.0` to `tol_init_multiplier` list. Remove `n_workers` from inference block. Bump `n_replicates` from 3 to 5 to match paper-concept.

2. **New `configs/sensitivity_gandk.json`**:
   - `benchmark.name = "gandk"`, `n_obs = 1000`, standard true params
   - `max_simulations = 20000` (g-and-k is harder)
   - Same sensitivity grid (minus `scheduler_type` or same 4D grid)
   - `n_replicates = 5`

3. **New `configs/small/sensitivity_gandk.json`**:
   - Reduced grid: k=[50], perturbation_scale=[0.4,0.8], scheduler_type=[acceptance_rate], tol_init_multiplier=[0.5,1.0]
   - `max_simulations = 2000`, `n_replicates = 2`

4. **`sensitivity_runner.py`**:
   - Remove `n_workers` injection from `inference_cfg` if present (or just don't include it in config)
   - Confirm `compute_sensitivity_quality_summary()` receives `bm.true_params`

### Verification

```bash
nastjapy_copy/.venv/bin/pytest experiments/tests/test_config.py::TestSensitivityConfig -x -q
# Then run end-to-end test mode:
nastjapy_copy/.venv/bin/python experiments/scripts/sensitivity_runner.py \
    --config experiments/configs/sensitivity.json --test --output-dir /tmp/sens_test
nastjapy_copy/.venv/bin/python experiments/scripts/sensitivity_runner.py \
    --config experiments/configs/sensitivity_gandk.json --test --output-dir /tmp/sens_gandk_test
```

---

## Phase 5 — Fix the "last 10% by row count" heuristic ✅ COMPLETE

This was partly addressed in Phase 1 (budget-keyed window), but needs explicit coverage.

### What to test first

**In `experiments/tests/test_sensitivity_metric.py`**:

```python
class TestBudgetKeyedTailWindow:
    def test_tail_window_uses_step_not_row_count(...)
        # Two CSVs: same particle count but one has steps 0..99, other 0..999
        # (simulating different simulation budgets).
        # Assert that "final" window is determined by max(step) * tail_fraction,
        # not len(rows) * tail_fraction.

    def test_few_accepted_particles_still_gives_estimate(...)
        # Only 5 accepted particles in the tail window → returns a value (not NaN or crash).
        # (Tight tol_init scenarios.)

    def test_tail_window_respects_max_simulations_parameter(...)
        # max_simulations=100, tail_fraction=0.1 → window covers steps 90-100.
        # Even if rows have steps beyond 100, those are excluded.
```

### Implementation

**In `compute_sensitivity_quality_summary()`** (Phase 1):
- Filter rows to `step >= max_simulations * (1 - tail_fraction)`
- Use this budget-position window, not `rows[-N:]`

### Verification

```bash
nastjapy_copy/.venv/bin/pytest experiments/tests/test_sensitivity_metric.py::TestBudgetKeyedTailWindow -x -q
```

---

## Phase 6 — Full regression pass ✅ COMPLETE

After all phases are complete, run the full test suite and an end-to-end small-mode run to confirm nothing regressed:

```bash
# Full test suite
nastjapy_copy/.venv/bin/pytest experiments/tests/ -x -q

# End-to-end sensitivity with gaussian_mean (small mode)
nastjapy_copy/.venv/bin/python experiments/scripts/sensitivity_runner.py \
    --config experiments/configs/sensitivity.json --small --output-dir /tmp/sens_small

# End-to-end sensitivity with gandk (small mode)
nastjapy_copy/.venv/bin/python experiments/scripts/sensitivity_runner.py \
    --config experiments/configs/sensitivity_gandk.json --small --output-dir /tmp/sens_gandk_small

# Verify outputs
ls /tmp/sens_small/data/sensitivity_quality_summary.csv
ls /tmp/sens_small/plots/sensitivity_heatmap__scheduler_type=acceptance_rate.pdf
ls /tmp/sens_gandk_small/data/sensitivity_quality_summary.csv
```

---

## Files to Create or Modify

| File | Action | Phase |
|------|--------|-------|
| `experiments/async_abc/analysis/sensitivity.py` | **Create** — `compute_sensitivity_quality_summary()` | 1 |
| `experiments/tests/test_sensitivity_metric.py` | **Create** — unit tests for metric function | 1, 5 |
| `experiments/async_abc/plotting/reporters.py` | **Modify** — `plot_sensitivity_summary()` accepts quality_df, per-scheduler loop | 1, 2 |
| `experiments/async_abc/plotting/common.py` | **Modify** — `sensitivity_heatmap()` accepts std_matrix, annotates cells | 3 |
| `experiments/scripts/sensitivity_runner.py` | **Modify** — call quality summary, pass to plotter | 1 |
| `experiments/configs/sensitivity.json` | **Modify** — add tol_init_multiplier=2.0, remove n_workers, bump replicates | 4 |
| `experiments/configs/sensitivity_gandk.json` | **Create** — g-and-k sensitivity config | 4 |
| `experiments/configs/small/sensitivity_gandk.json` | **Create** — small g-and-k sensitivity config | 4 |
| `experiments/tests/test_config.py` | **Modify** — extend TestSensitivityConfig | 4 |
| `experiments/tests/test_plotting.py` | **Modify** — add per-scheduler and uncertainty tests | 2, 3 |

---

## Non-Goals

- Changing how particle records are written (ParticleRecord schema unchanged)
- Modifying the scaling or benchmark runner pipelines
- Hardcoding column names in `common.py` is a separate cleanup; leave for later
- Running the full 81+ variant production grid — that is after tests pass
