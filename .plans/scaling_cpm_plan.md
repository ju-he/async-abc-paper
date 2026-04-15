# Plan: CPM Scaling Experiment + Quality Metrics

## Context

The existing `scaling` experiment uses Lotka-Volterra (LV). We want to add an analogous
`scaling_cpm` experiment using the Cellular Potts Model (CPM) benchmark, which is:
- The "expensive simulator" showcase for the paper (5–10 s/sim)
- Already proven to favor async over SMC in throughput
- Fully concurrent-safe (`MULTIPROCESSING_SAFE = True`)

Two quality metrics are needed (since no analytical posterior exists for CPM):
- **Option 1**: Posterior mean L2 to true params (fast, crude, always available)
- **Option 3**: Wasserstein distance to a precomputed reference posterior (proper distributional metric)

A prerequisite is fixing the existing `has_true_params=False` bug that currently blocks all
CPM quality metrics.

---

## Step 0: Fix `has_true_params=False` bug (prerequisite)

**Root cause (two sub-issues):**

**A.** `runtime_summary.py:_true_params_from_benchmark_cfg` (line 237–242) blindly does
`float(value)` for all `true_*` keys, which crashes on `true_params_scale: "normalized [0,1]"`
(a string). This silently corrupts or discards the true_params dict.

**Fix A** in `experiments/async_abc/reporting/runtime_summary.py:237`:
```python
def _true_params_from_benchmark_cfg(benchmark_cfg):
    true_params = {}
    for key, value in benchmark_cfg.items():
        if key.startswith("true_") and isinstance(value, (int, float)):
            true_params[key.removeprefix("true_")] = float(value)
    return true_params
```

**B.** `reporters.py:_true_params_from_cfg` reads param names from `record.params.keys()`.
If CPM params are stored with `param_` prefix in records (i.e. `record.params = {'param_division_rate': …}`),
then `f"true_{param}"` → `"true_param_division_rate"` which doesn't exist in config.

**Diagnosis step (before writing code):** Inspect a CPM ParticleRecord from small1 CSV to
confirm whether `record.params` keys have `param_` prefix.

**Fix B (if prefix confirmed)** in `experiments/async_abc/plotting/reporters.py:_true_params_from_cfg`
and `experiments/scripts/scaling_runner.py:_true_params_from_cfg` (local copy):
```python
clean = param.removeprefix("param_")
key = f"true_{clean}"
```

**Note on scale:** `true_division_rate=0.049905` and `true_motility=0.2` in the config are
already in [0,1] normalized space (matching parameter_space `"range": [0.0, 1.0]`).
No unit conversion needed.

**Files to edit:**
- `experiments/async_abc/reporting/runtime_summary.py`
- `experiments/async_abc/plotting/reporters.py` (if Fix B needed)
- `experiments/scripts/scaling_runner.py` (if Fix B needed — local copy of the function)

---

## Step 1: Add `posterior_mean_l2` metric (Option 1)

Add a new column `posterior_mean_l2` to the quality curve output.

**Where:** `experiments/async_abc/analysis/convergence.py`

**Implementation:** Add a helper and call it alongside `_wasserstein_to_true_params` in
`_quality_row()`:
```python
def _posterior_mean_l2(frame: pd.DataFrame, true_params: dict[str, float]) -> float | None:
    """Normalized L2 from posterior mean to true params (in [0,1] space)."""
    param_cols = [c for c in frame.columns if c in true_params]
    if not param_cols:
        return None
    means = frame[param_cols].mean()
    diffs = np.array([means[p] - true_params[p] for p in param_cols])
    return float(np.linalg.norm(diffs) / np.sqrt(len(param_cols)))
```

Emit as `posterior_mean_l2` in the DataFrame. The `posterior_quality_curve()` public
interface is unchanged — the new column is additive.

The scaling runner's `_final_summary_row()` and `_budget_summary_rows()` (scaling_runner.py ~line 417, 496)
also need to read and forward this column to the output CSVs.

---

## Step 2: Generate CPM reference posterior (Option 3 prerequisite)

**Script:** `experiments/scripts/generate_cpm_reference_posterior.py`

One-time cluster job. Runs parallel rejection ABC at tight tolerance.

**Approach:**
1. Load CPM benchmark from `cellular_potts.json`
2. Sample `(division_rate, motility)` uniformly from [0,1]² using MPI workers
3. Accept particles with `loss < target_tol` (target: ~0.1, expected ~3× tighter than
   SMC reaches in small1 at `final_tol≈0.165`)
4. Collect `n_reference=1000` accepted particles
5. Save to `experiments/assets/cellular_potts/reference_posterior_samples.csv`
   with columns `division_rate`, `motility`, `loss`

**Estimated cost:** ~100 k simulations, ~3 h at 48 workers. Run once, commit the CSV.

Add `"reference_posterior_path"` key to `cellular_potts.json` and `scaling_cpm.json`
benchmark sections pointing to this file.

---

## Step 3: Add `wasserstein_to_reference` metric (Option 3)

**Where:** `experiments/async_abc/analysis/convergence.py`

Extend `posterior_quality_curve()` signature:
```python
def posterior_quality_curve(
    records,
    true_params,
    *,
    reference_posterior: pd.DataFrame | None = None,  # NEW
    ...
) -> pd.DataFrame:
```

When `reference_posterior` is provided, compute sliced Wasserstein between checkpoint
posterior samples and reference samples (reusing the existing `ot.sliced_wasserstein_distance`
path). Emit as `wasserstein_to_reference` column alongside `wasserstein`.

**Loading in runners:** In `scaling_cpm_runner.py` (and optionally `cellular_potts_runner.py`),
load from config:
```python
ref_path = benchmark_cfg.get("reference_posterior_path")
reference_posterior = pd.read_csv(ref_path) if ref_path else None
```
Pass to `posterior_quality_curve(..., reference_posterior=reference_posterior)`.

---

## Step 4: Create `scaling_cpm_runner.py`

**File:** `experiments/scripts/scaling_cpm_runner.py`

Adapt from `experiments/scripts/scaling_runner.py`. Key differences:
- Benchmark loading: CPM instead of LV
- `_prepare_runtime_cfg()`: redirect CPM output dir (copy from `cellular_potts_runner.py:43-44`)
- Config section name: `"scaling_cpm"` (but same internal structure as `"scaling"`)
- Load and pass `reference_posterior` to `_quality_curve_by_wall_time()`
- `_true_params_from_cfg` local copy inherits Step 0 fixes

Do not refactor shared base class in this session (deferred to a later session).

---

## Step 5: Create configs

### `experiments/configs/scaling_cpm.json`
Based on `cellular_potts.json` benchmark section + scaling grid:
- `k_values: [10, 50, 100]`, `test_k_values: [10, 50]`
- `worker_counts: [1, 4, 16, 48, 96]`, `test_worker_counts: [1, 4, 48]`
- `wall_time_limit_s: 3600`, `wall_time_budgets_s: [900, 1800, 3600]`
- `n_replicates: 5`, `scheduler_type: "acceptance_rate"` (CPM has no extinction pathology)
- `max_simulations_policy: {min_total: 500, per_worker: 20, k_factor: 2}`
- Add `reference_posterior_path` to benchmark section

### `experiments/configs/small/scaling_cpm.json`
Same structure but:
- `n_replicates: 1`, `wall_time_limit_s: 1800`, `wall_time_budgets_s: [900, 1800]`
- `test_worker_counts: [1, 4, 48]`, `test_k_values: [10, 50]`

---

## Execution order

1. **Step 0** (diagnose + fix true_params bug) → verify with `--test` cellular_potts
2. **Step 1** (posterior_mean_l2) → unit test
3. **Step 5** (configs, no code)
4. **Step 4** (runner, depends on Steps 0–1)
5. **Step 2** (reference posterior generation, cluster job, independent)
6. **Step 3** (wasserstein_to_reference, can be coded before Step 2 finishes; tested once CSV exists)

Steps 2–3 can be deferred: first scaling_cpm test run uses only `posterior_mean_l2`.

---

## Verification

1. **Step 0:** Run `--test` cellular_potts; confirm `has_true_params=True` in `plot_audit.csv`
   and `quality_vs_wall_time_data.csv` is generated.
2. **Step 1:** Confirm `posterior_mean_l2` column is non-null in quality curve CSV.
3. **Steps 4–5:** Run `--test` scaling_cpm; confirm `budget_summary.csv` has valid
   `quality_posterior_mean_l2` rows for all (worker, k) combinations.
4. **Step 2:** Inspect reference posterior CSV histogram; check reasonable coverage of [0,1]² space.
5. **Step 3:** Run `--small` scaling_cpm; confirm `wasserstein_to_reference` column populated.
