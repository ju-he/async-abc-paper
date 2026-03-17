# Async-ABC Paper: Extended Experiments — Multi-Phase TDD Plan

## Context

This plan extends the existing experiment codebase with the analyses, visualizations, and experiments proposed after the initial config review. Every phase is TDD: tests are written (and must fail) before implementation begins.

**Venv:** `~/bwSyncShare/Code/mirrors/nastjapy_copy/.venv`
**Test runner:** `source ~/bwSyncShare/Code/mirrors/nastjapy_copy/.venv/bin/activate && cd experiments && python -m pytest tests/ -x -q`
**Existing tests:** `experiments/tests/` (9 files, ~700 lines)

### Core Invariants (carry over from existing plan)
- Every runner: `--config`, `--output-dir`, `--test` flags
- Test mode: ≤8 workers, max_simulations≤500, n_replicates≤2
- Every plot: `.pdf` + `.png` + data CSV + metadata JSON
- Config-driven, reproducible, provenance-tracked

---

## Phase Overview

| Phase | Goal | New Infrastructure | Key Files |
|-------|------|-------------------|-----------|
| 1 | Worker event logging | Extended `ParticleRecord` | `records.py`, inference wrappers |
| 2 | Analysis module | `async_abc/analysis/` | ESS, convergence, trajectory |
| 3 | New visualizations | Extended `plotting/` | Gantt, quality curve, corner plot |
| 4 | SBC / calibration | SBC runner + config | `sbc_runner.py`, `sbc.json` |
| 5 | Straggler experiment | New runner + config | `straggler_runner.py`, `straggler.json` |
| 6 | tol_init sensitivity | Extend sensitivity runner | `sensitivity_runner.py`, config update |

---

## Phase 1 — Extended ParticleRecord with Worker Event Fields

Status: completed on 2026-03-17

### Goal

Enable Gantt charts and generation-barrier timing by capturing when each simulation
started and finished, and on which worker it ran.

### New fields in `ParticleRecord`

```python
worker_id: Optional[str] = None         # rank / thread ID of the executing worker
sim_start_time: Optional[float] = None  # wall-clock seconds since run_start (sim began)
sim_end_time: Optional[float] = None    # wall-clock seconds since run_start (sim done)
generation: Optional[int] = None        # SMC generation index (for sync methods)
```

`wall_time` retains its meaning (= `sim_end_time` for completed sims).

### Implementation plan

**A. `experiments/async_abc/io/records.py`**
- Add 4 optional fields with defaults `None`
- Extend `to_csv_row()` to include them (empty string for None)
- Extend `from_csv_row()` to parse them back (with type coercion)
- Extend CSV header constant

**B. `experiments/async_abc/inference/propulate_abc.py`**
- Add closure-level dict `_timing_by_gen: dict[int, tuple[str, float, float]]` mapping `ind.generation → (worker_id, sim_start_time, sim_end_time)`
- Inside `loss_fn(ind)`:
  ```python
  rank = MPI.COMM_WORLD.Get_rank()
  sim_start = time.time() - run_start
  result = float(simulate_fn(params, seed=sim_seed))
  sim_end = time.time() - run_start
  _timing_by_gen[int(ind.generation)] = (str(rank), sim_start, sim_end)
  return result
  ```
  Timing is captured *around the actual simulate call* — more accurate than deducing from successive end times. Safe because each MPI process only writes entries for its own evaluated individuals.
- After `propulate()`, look up `_timing_by_gen.get(int(ind.generation), (None, None, None))` for each record
- `sim_end_time` replaces the existing `wall_time` computation (they measure the same thing); keep `wall_time = sim_end_time` for backward compatibility
- Populate `generation = int(ind.generation)` (already available on `ind`)

Implemented note:
- Used existing `propulate.Individual` metadata instead of an extra closure dict:
  `ind.rank -> worker_id`, `ind.evaltime -> sim_end_time`, and
  `ind.evalperiod -> sim_start_time` via `sim_end_time - evalperiod`.
  This keeps the implementation simpler while still using backend-provided timing data.

**C. `experiments/async_abc/inference/abc_smc_baseline.py`**
- Record `generation_start = time.time()` before each pyABC `run()` call per generation
- After each generation, compute `generation_end = time.time()`
- Set `sim_start_time ≈ generation_start - run_start` and `sim_end_time ≈ generation_end - run_start` as generation-level approximations
- Set `generation = t` for each particle in generation `t`
- Note: per-particle start/end within a generation is not accessible from pyABC's history; generation-level bounds are the best available approximation

Implemented note:
- Used `history.get_all_populations()["population_end_time"]` to recover actual
  generation end timestamps after the run. Generation start is approximated by
  the previous generation boundary, with generation 0 starting at `run_start`.

### TDD steps

**Step 1.1 — Write failing tests** (`experiments/tests/test_records.py`, extend existing): complete
```python
def test_particle_record_has_worker_event_fields():
    r = ParticleRecord(method="m", replicate=0, seed=1, step=0, params={}, loss=0.1)
    assert r.worker_id is None
    assert r.sim_start_time is None
    assert r.sim_end_time is None
    assert r.generation is None

def test_csv_roundtrip_with_worker_events():
    r = ParticleRecord(..., worker_id="rank_3", sim_start_time=1.2, sim_end_time=2.5, generation=2)
    # write to csv, read back, check equality

def test_csv_roundtrip_with_none_worker_events():
    # None fields roundtrip correctly (not garbled as 0.0)
```

**Step 1.2 — Write failing tests** (`experiments/tests/test_inference.py`, extend): complete
```python
def test_propulate_abc_records_have_sim_end_time():
    records = run_propulate_abc(minimal_config, ...)
    assert all(r.sim_end_time is not None for r in records)
    assert all(r.sim_end_time >= 0.0 for r in records)

def test_abc_smc_baseline_records_have_generation():
    records = run_abc_smc_baseline(minimal_config, ...)
    assert all(r.generation is not None for r in records)
    gens = sorted(set(r.generation for r in records))
    assert gens == list(range(len(gens)))  # contiguous from 0

def test_abc_smc_baseline_records_have_generation_timing():
    records = run_abc_smc_baseline(minimal_config, ...)
    for r in records:
        if r.sim_start_time is not None and r.sim_end_time is not None:
            assert r.sim_end_time >= r.sim_start_time
```

**Step 1.3 — Implement** fields in `records.py`, then `propulate_abc.py`, then `abc_smc_baseline.py`: complete

**Step 1.4 — Run tests**: complete

Verified:
- `python -m pytest tests/test_records.py -q` → 14 passed
- `python -m pytest tests/test_inference.py -q -k 'sim_end_time or generation_timing or records_have_generation'` → 3 passed
- `python -m pytest tests/test_inference.py -q` → 51 passed, 1 skipped
- `python -m pytest tests/test_runners.py -q` was started as a broader regression check but was stopped after running for several minutes; phase 1 correctness was verified by the focused schema and inference suites above.

### Commit message
``` 
feat: extend ParticleRecord with worker event fields

Enables Gantt chart visualization and generation barrier overhead analysis.
Propulate backend now exports worker rank and simulation timing from
Individual metadata, while the ABC-SMC baseline stamps generation-level
timing bounds and generation indices from pyABC history.
```

---

## Phase 2 — Analysis Module

Status: completed on 2026-03-17

### Goal

A standalone `async_abc/analysis/` module that computes derived metrics from
`raw_results.csv` — without needing to rerun experiments. All functions are
pure (input: list[ParticleRecord] or DataFrame, output: numeric result or DataFrame).

### New module: `experiments/async_abc/analysis/`

```
async_abc/analysis/
├── __init__.py
├── ess.py              # effective sample size from importance weights
├── convergence.py      # Wasserstein vs. wall_time; time-to-threshold
├── trajectory.py       # tolerance and loss over simulation step / wall_time
└── barrier.py          # generation barrier overhead from abc_smc_baseline data
```

Implemented note:
- Added `async_abc.analysis` with the planned public modules plus a small
  internal `_helpers.py` adapter to flatten `ParticleRecord` iterables into
  DataFrames for reuse across the pure analysis functions.

### `ess.py`

```python
def compute_ess(weights: np.ndarray) -> float:
    """ESS = (sum(w))^2 / sum(w^2). Weights need not be normalized."""

def ess_over_time(records: list[ParticleRecord], method: str) -> pd.DataFrame:
    """Return DataFrame(step, ess) for a single method+replicate."""
```

### `convergence.py`

```python
def wasserstein_at_checkpoints(
    records: list[ParticleRecord],
    true_params: dict[str, float],
    checkpoint_steps: list[int],
    n_projections: int = 50,
) -> pd.DataFrame:
    """Return DataFrame(method, replicate, step, wall_time, wasserstein) at each checkpoint.

    Uses sliced Wasserstein distance (ot.sliced_wasserstein_distance, n_projections=50)
    for multi-parameter posteriors; falls back to scipy 1D Wasserstein for single params.
    """

def time_to_threshold(
    records: list[ParticleRecord],
    true_params: dict[str, float],
    target_wasserstein: float,
) -> pd.DataFrame:
    """Return DataFrame(method, replicate, wall_time_to_threshold). NaN if never reached."""
```

### `trajectory.py`

```python
def tolerance_over_wall_time(records: list[ParticleRecord]) -> pd.DataFrame:
    """Return DataFrame(method, replicate, wall_time, tolerance)."""

def loss_over_steps(records: list[ParticleRecord]) -> pd.DataFrame:
    """Return DataFrame(method, replicate, step, loss)."""
```

### `barrier.py`

```python
def generation_spans(records: list[ParticleRecord]) -> pd.DataFrame:
    """For abc_smc_baseline records with generation field set:
    Return DataFrame(method, replicate, generation, gen_start, gen_end, gen_duration, n_particles)."""

def barrier_overhead_fraction(records: list[ParticleRecord]) -> pd.DataFrame:
    """Estimate fraction of wall time in barriers (min worker finishing a generation
    to last worker finishing) vs. total. Requires sim_start_time + sim_end_time."""
```

### TDD steps

**Step 2.1 — New file** `experiments/tests/test_analysis.py`: complete

```python
def test_compute_ess_uniform_weights():
    w = np.ones(10)
    assert abs(compute_ess(w) - 10.0) < 1e-6

def test_compute_ess_degenerate():
    w = np.zeros(10); w[0] = 1.0
    assert abs(compute_ess(w) - 1.0) < 1e-6

def test_ess_over_time_returns_dataframe(sample_records):
    df = ess_over_time(sample_records, method="async_propulate_abc")
    assert set(df.columns) >= {"step", "ess"}
    assert len(df) > 0

def test_wasserstein_at_checkpoints(sample_records):
    df = wasserstein_at_checkpoints(sample_records, true_params={"mu": 0.0}, checkpoint_steps=[10, 50])
    assert set(df.columns) >= {"method", "replicate", "step", "wall_time", "wasserstein"}

def test_time_to_threshold_returns_none_for_impossible(sample_records):
    df = time_to_threshold(sample_records, true_params={"mu": 0.0}, target_wasserstein=1e-9)
    assert df["wall_time_to_threshold"].isna().any()

def test_tolerance_over_wall_time(sample_records):
    df = tolerance_over_wall_time(sample_records)
    assert "wall_time" in df.columns and "tolerance" in df.columns

def test_generation_spans_requires_generation_field(abc_smc_records):
    df = generation_spans(abc_smc_records)
    assert "generation" in df.columns
    assert (df["gen_end"] >= df["gen_start"]).all()
```

**Step 2.2 — Implement** `ess.py`, `convergence.py`, `trajectory.py`, `barrier.py`: complete

**Step 2.3 — Run tests**: complete

Verified:
- `python -m pytest tests/test_analysis.py -q` → 9 passed
- `python -m pytest tests/test_records.py tests/test_inference.py -q` → 65 passed, 1 skipped

### Commit message
```
feat: add async_abc/analysis/ module (ESS, convergence, trajectory, barrier overhead)

Pure-function analysis layer that operates on ParticleRecord lists/DataFrames.
Enables time-to-threshold, Wasserstein checkpointing, ESS tracking, and
generation barrier overhead quantification without re-running experiments.
```

---

## Phase 3 — New Visualizations

Status: completed on 2026-03-17

### Goal

Four new plot types callable from runners and analysis scripts:

1. **Gantt / worker timeline** — one row per worker, colored blocks = simulation time
2. **Posterior quality vs. wall-clock time** — Wasserstein(t) curves per method
3. **Corner plot** — pairwise joint marginals for multi-parameter posteriors
4. **Tolerance trajectory** — ε(t) over wall-clock time, sync vs. async overlaid

### New functions in `experiments/async_abc/plotting/common.py`

```python
def gantt_plot(records: list[ParticleRecord], ax=None) -> Figure:
    """Horizontal bar chart: x=wall_time, y=worker_id, color=method.
    Requires sim_start_time and sim_end_time fields."""

def quality_vs_time_plot(quality_df: pd.DataFrame, ax=None) -> Figure:
    """Line plot: x=wall_time, y=wasserstein, hue=method.
    quality_df from convergence.wasserstein_at_checkpoints()."""

def corner_plot(records: list[ParticleRecord], param_names: list[str],
                true_params: dict = None, ax=None) -> Figure:
    """Grid of scatter/KDE plots for all pairs of parameters.
    Diagonal: marginal KDE. Off-diagonal: scatter with KDE contours."""

def tolerance_trajectory_plot(trajectory_df: pd.DataFrame, ax=None) -> Figure:
    """Line plot: x=wall_time, y=tolerance (log scale), hue=method.
    Shows how each method tightens the tolerance over time."""
```

### New reporters in `experiments/async_abc/plotting/reporters.py`

```python
def plot_worker_gantt(records, output_dir): ...
def plot_quality_vs_time(records, true_params, checkpoint_steps, output_dir): ...
def plot_corner(records, param_names, true_params, output_dir): ...
def plot_tolerance_trajectory(records, output_dir): ...
```

Implemented note:
- Added the four figure constructors in `plotting/common.py` as matplotlib
  figure builders and wired them into exporting reporters in
  `plotting/reporters.py`.
- Added `plot_benchmark_diagnostics(...)` so the benchmark runner scripts can
  honor the expanded `plots` config without duplicating dispatch logic.
- Updated `gaussian_mean.json`, `gandk.json`, `lotka_volterra.json`,
  `cellular_potts.json`, and `runtime_heterogeneity.json` with the new plot
  switches.
- Fixed a runtime integration bug during implementation by sizing the Gantt
  figure by worker count instead of record count.

### Config hooks (add to relevant experiment configs)

In benchmark configs (`plots` block):
```json
"plots": {
  "posterior": true,
  "archive_evolution": true,
  "corner": true,
  "tolerance_trajectory": true,
  "quality_vs_time": true
}
```

In `runtime_heterogeneity.json` (`plots` block):
```json
"plots": {
  "idle_fraction": true,
  "throughput_over_time": true,
  "idle_fraction_comparison": true,
  "gantt": true
}
```

### TDD steps

**Step 3.1 — Extend** `experiments/tests/test_plotting.py`: complete

```python
def test_gantt_plot_requires_sim_times(records_without_sim_times):
    with pytest.raises(ValueError, match="sim_start_time"):
        gantt_plot(records_without_sim_times)

def test_gantt_plot_returns_figure(records_with_sim_times):
    fig = gantt_plot(records_with_sim_times)
    assert isinstance(fig, Figure)

def test_quality_vs_time_plot_returns_figure(quality_df):
    fig = quality_vs_time_plot(quality_df)
    assert isinstance(fig, Figure)

def test_corner_plot_returns_figure(sample_records_multipar):
    fig = corner_plot(sample_records_multipar, param_names=["A", "B"])
    assert isinstance(fig, Figure)

def test_tolerance_trajectory_plot_returns_figure(trajectory_df):
    fig = tolerance_trajectory_plot(trajectory_df)
    assert isinstance(fig, Figure)
```

**Step 3.2 — Implement** plot functions in `common.py` and wire up reporters: complete

**Step 3.3 — Run tests**: complete

Verified:
- `python -m pytest tests/test_plotting.py -q` → 22 passed
- `python -m pytest tests/test_phase6.py -q -k 'Phase3ConfigPlots'` → 4 passed
- `python -m pytest tests/test_runners.py -q -k 'creates_phase3_plots or creates_gantt_plot'` → 2 passed

### Commit message
```
feat: add Gantt, quality-vs-time, corner, and tolerance trajectory plot types

Gantt plot visualizes per-worker simulation timelines (requires Phase 1 event
fields). Quality-vs-time and tolerance trajectory plots use Phase 2 analysis
functions. Corner plot shows pairwise joint posteriors for multi-parameter models.
```

---

## Phase 4 — Simulation-Based Calibration (SBC) Runner

Status: completed on 2026-03-17

### Goal

Empirical calibration check: draw θ* from prior, simulate observed data, run inference,
check whether θ* falls within α-credible intervals with frequency α across N trials.
Produces a rank-histogram (SBC plot) and empirical coverage table.

### New files

```
experiments/
├── configs/sbc.json
├── scripts/sbc_runner.py
└── async_abc/analysis/sbc.py
```

### `experiments/configs/sbc.json`

```json
{
  "experiment_name": "sbc",
  "benchmark": {
    "name": "gaussian_mean",
    "n_obs": 100,
    "sigma_obs": 1.0,
    "prior_low": -5.0,
    "prior_high": 5.0
  },
  "methods": ["async_propulate_abc", "abc_smc_baseline"],
  "inference": {
    "max_simulations": 5000,
    "n_workers": 48,
    "k": 100,
    "tol_init": 5.0,
    "n_generations": 5,
    "scheduler_type": "acceptance_rate",
    "perturbation_scale": 0.8
  },
  "execution": {
    "base_seed": 0
  },
  "sbc": {
    "n_trials": 200,
    "coverage_levels": [0.5, 0.8, 0.9, 0.95]
  },
  "plots": {
    "rank_histogram": true,
    "coverage_table": true
  }
}
```

Implemented note:
- Added `execution.n_replicates = 1` in the concrete config for compatibility
  with the existing config schema, while the SBC loop itself is driven by
  `sbc.n_trials`.
- Extended test-mode config overrides to clamp `sbc.n_trials`, keeping the
  SBC runner fast enough for subprocess tests.

### `experiments/async_abc/analysis/sbc.py`

```python
def compute_rank(posterior_samples: np.ndarray, true_value: float) -> int:
    """Rank of true_value among posterior samples (0 to len(samples))."""

def sbc_ranks(trials: list[dict]) -> pd.DataFrame:
    """trials: list of dicts with 'posterior_samples' and 'true_value'.
    Returns DataFrame(trial, param, rank, n_samples)."""

def empirical_coverage(
    trials: list[dict],
    coverage_levels: list[float],
) -> pd.DataFrame:
    """Returns DataFrame(param, coverage_level, empirical_coverage)."""
```

Implemented note:
- The implemented summaries preserve `method` and `param` columns when present,
  so multiple inference methods can be analyzed in one SBC run without losing
  provenance.

### `experiments/scripts/sbc_runner.py`

High-level flow:
1. Load config
2. For each trial (draw θ* from prior, set observed_data_seed=trial_idx):
   - Instantiate benchmark with θ* as true parameter
   - Run `run_method()` for each configured method
   - Compute posterior samples from accepted particles (weight ≥ threshold)
   - Record rank of θ* in posterior
3. Call `sbc.sbc_ranks()` and `sbc.empirical_coverage()`
4. Plot rank histogram and coverage table
5. Write results CSV + metadata

Implemented note:
- The runner samples true parameters from the benchmark limits, rebuilds the
  benchmark with `true_<param>` overrides per trial, runs each configured
  method via `run_method()`, writes `sbc_ranks.csv` and `coverage.csv`, and
  exports a rank histogram plus empirical-coverage figure.
- `run_all_paper_experiments.py` was updated to register the new `sbc`
  experiment.

### TDD steps

**Step 4.1 — New file** `experiments/tests/test_sbc.py`: complete

```python
def test_compute_rank_true_below_all():
    assert compute_rank(np.array([1.0, 2.0, 3.0]), 0.0) == 0

def test_compute_rank_true_above_all():
    assert compute_rank(np.array([1.0, 2.0, 3.0]), 4.0) == 3

def test_compute_rank_middle():
    rank = compute_rank(np.array([0.0, 1.0, 2.0]), 1.5)
    assert rank == 2

def test_sbc_ranks_returns_dataframe():
    trials = [
        {"posterior_samples": np.linspace(0, 1, 100), "true_value": 0.5}
        for _ in range(10)
    ]
    df = sbc_ranks(trials)
    assert "rank" in df.columns and "trial" in df.columns

def test_empirical_coverage_uniform_posterior():
    # Uniform posterior should give correct coverage
    rng = np.random.default_rng(42)
    trials = [
        {"posterior_samples": rng.uniform(0, 1, 100), "true_value": rng.uniform(0, 1)}
        for _ in range(500)
    ]
    df = empirical_coverage(trials, coverage_levels=[0.5, 0.9])
    row_50 = df[df["coverage_level"] == 0.5]["empirical_coverage"].iloc[0]
    assert abs(row_50 - 0.5) < 0.05  # within 5%

def test_sbc_runner_test_mode(tmp_path, sbc_config_file):
    result = subprocess.run(
        ["python", "scripts/sbc_runner.py", "--config", sbc_config_file,
         "--output-dir", str(tmp_path), "--test"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert (tmp_path / "sbc" / "data" / "sbc_ranks.csv").exists()
```

**Step 4.2 — Implement** `sbc.py`, then `sbc_runner.py`: complete

**Step 4.3 — Run tests**: complete

Verified:
- `python -m pytest tests/test_sbc.py -q` → 6 passed
- `python -m pytest tests/test_config.py tests/test_phase6.py -q -k 'sbc or config_passes_schema_validation or config_file_exists'` → 17 passed, 46 deselected

### Commit message
```
feat: add SBC (simulation-based calibration) runner and analysis

Empirically validates posterior calibration: true parameters should fall
within α-credible intervals with frequency α. Produces rank histograms and
coverage tables for async_propulate_abc vs abc_smc_baseline on gaussian_mean.
```

---

## Phase 5 — Straggler Tolerance Experiment

### Goal

Show that async ABC degrades gracefully when one worker permanently runs slowly
(straggler), while synchronous ABC-SMC blocks the entire generation on that worker.

This differs from the LogNormal heterogeneity experiment (statistical runtime noise)
by modelling a persistent, structural HPC failure mode.

### New files

```
experiments/
├── configs/straggler.json
└── scripts/straggler_runner.py
```

### `experiments/configs/straggler.json`

```json
{
  "experiment_name": "straggler",
  "benchmark": {
    "name": "gaussian_mean",
    "observed_data_seed": 42,
    "n_obs": 100,
    "true_mu": 0.0,
    "sigma_obs": 1.0,
    "prior_low": -5.0,
    "prior_high": 5.0
  },
  "methods": ["async_propulate_abc", "abc_smc_baseline"],
  "inference": {
    "max_simulations": 10000,
    "n_workers": 16,
    "k": 100,
    "tol_init": 5.0,
    "n_generations": 5,
    "scheduler_type": "acceptance_rate",
    "perturbation_scale": 0.8
  },
  "execution": {
    "n_replicates": 5,
    "base_seed": 0
  },
  "straggler": {
    "straggler_rank": 0,
    "base_sleep_s": 0.1,
    "slowdown_factor": [1, 5, 10, 20]
  },
  "plots": {
    "throughput_vs_slowdown": true,
    "gantt": true
  }
}
```

### `experiments/scripts/straggler_runner.py`

High-level flow:
1. Load config; extract `straggler.slowdown_factor` sweep
2. For each slowdown level:
   - Wrap benchmark's simulate: check `MPI.COMM_WORLD.Get_rank() == straggler_rank` (configurable, default rank 0 — all Propulate ranks are workers). If straggler: `time.sleep(slowdown_factor * base_sleep_s)`. In test mode: skip sleep entirely.
   - Run `run_experiment(cfg)` for both methods
   - Tag records: `method = f"{method}__slowdown{factor}x"`
3. Plot throughput vs. slowdown factor, and Gantt at worst slowdown level
4. Write results CSV + metadata
5. `compute_scaling_factor` already handles basic configs; add `straggler` block case analogous to `heterogeneity` (compute expected sleep overhead per sigma level)

### TDD steps

**Step 5.1 — Extend** `experiments/tests/test_runners.py`:

```python
def test_straggler_runner_test_mode(tmp_path, straggler_config_file):
    result = subprocess.run(
        ["python", "scripts/straggler_runner.py", "--config", straggler_config_file,
         "--output-dir", str(tmp_path), "--test"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    csv_path = tmp_path / "straggler" / "data" / "raw_results.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert "straggler_slowdown" in " ".join(df["method"].unique())

def test_straggler_runner_tags_records(tmp_path, straggler_config_file):
    # Slowdown levels should appear in method column
    ...
```

**Step 5.2 — Implement** `straggler_runner.py`

**Step 5.3 — Run tests** — all Phase 5 tests green

### Commit message
```
feat: add straggler experiment (persistent slow-worker fault tolerance test)

Sweeps slowdown_factor for a single straggler worker and compares async vs
sync throughput degradation. Async ABC routes work to available workers;
sync ABC-SMC blocks entire generations on the straggler.
```

---

## Phase 6 — tol_init Sensitivity Sweep

### Goal

Complete the hyperparameter sensitivity analysis by adding `tol_init` as a swept
dimension. The existing sensitivity grid covers k, perturbation_scale, and
scheduler_type, but not the initial tolerance threshold — often the most impactful
hyperparameter.

### Changes

**`experiments/configs/sensitivity.json`** — add `tol_init_multiplier` to grid:
```json
"sensitivity_grid": {
  "k": [50, 100, 200],
  "perturbation_scale": [0.4, 0.8, 1.5],
  "scheduler_type": ["acceptance_rate", "quantile", "geometric_decay"],
  "tol_init_multiplier": [0.5, 1.0, 2.0, 5.0]
}
```
`tol_init_multiplier` scales the base `tol_init` from the config. Total grid: 3 × 3 × 3 × 4 = 108 variants, 5 replicates each = 540 runs. May warrant reducing other dimensions or restricting to most informative subset.

**`experiments/scripts/sensitivity_runner.py`** — extend sweep loop:
- For each `tol_init_multiplier` in grid, set `cfg["inference"]["tol_init"] *= multiplier`
- Tag variant name with `_tol{multiplier}x` suffix
- Extend sensitivity heatmap to show tol_init as a separate panel or facet

**`experiments/async_abc/plotting/common.py`** — extend `sensitivity_heatmap()`:
- Support 3-way grid (k × perturbation_scale × tol_init) as a faceted figure

### TDD steps

**Step 6.1 — Extend** `experiments/tests/test_config.py`:
```python
def test_sensitivity_config_accepts_tol_init_multiplier():
    cfg = load_config("configs/sensitivity.json")
    assert "tol_init_multiplier" in cfg["sensitivity_grid"]
```

**Step 6.2 — Extend** `experiments/tests/test_runners.py`:
```python
def test_sensitivity_runner_applies_tol_init_multiplier(tmp_path, sensitivity_config):
    # Inject tol_init_multiplier=[0.5, 2.0] into config
    # Run in test mode, verify two tol_init levels appear in output
    ...
```

**Step 6.3 — Implement** runner extension and updated plot

**Step 6.4 — Run tests** — all Phase 6 tests green

### Commit message
```
feat: extend sensitivity sweep to include tol_init_multiplier dimension

Adds initial tolerance as a sensitivity dimension (×0.5, ×1, ×2, ×5 of base
tol_init). Sensitivity heatmap now facets over tol_init levels.
Grid size: 3×3×3×4 = 108 variants, completing the hyperparameter analysis.
```

---

## Progress Tracking

Mark each phase ✅ when its commit is made.

- [✅] Phase 1 — Extended ParticleRecord
- [✅] Phase 2 — Analysis module
- [✅] Phase 3 — New visualizations
- [✅] Phase 4 — SBC runner
- [ ] Phase 5 — Straggler experiment
- [ ] Phase 6 — tol_init sensitivity

## Resolved Design Decisions

1. **Phase 1 — Propulate worker_id + sim timing**: Propulate exposes no `starttime` on `Individual`. Instead, capture `sim_start_time` and `sim_end_time` directly inside `loss_fn` by measuring `time.time() - run_start` before and after the `simulate_fn()` call. Store `(worker_id, sim_start_time, sim_end_time)` in a closure dict `_timing_by_gen` keyed by `int(ind.generation)`. Look up after `propulate()` completes. All Propulate ranks are workers (no master/worker distinction) — `mpi4py` confirmed available.

2. **Phase 2 — Wasserstein for multi-dimensional posteriors**: Use **sliced Wasserstein distance** via `ot.sliced_wasserstein_distance()` from the POT library (installed: POT 0.9.6.post1). Use `n_projections=50` as default. For 1D posteriors (gaussian_mean), fall back to the existing `scipy.stats.wasserstein_distance` for exact computation.

3. **Phase 4 — SBC trial count**: 200 trials × 5000 sims on the cluster is acceptable. Test mode must clamp `n_trials` to a small value (e.g. 3). Add `sbc` block handling to `compute_scaling_factor()` in `runner.py` so `--test` produces an accurate full-run time estimate:
   ```python
   elif "sbc" in cfg:
       n_trials = cfg["sbc"]["n_trials"]
       test_trials = min(n_trials, 3)
       factor *= n_trials / test_trials
       note = f"{n_trials} SBC trials × {full_sims} sims, {full_workers} workers"
   ```

4. **Phase 5 — Worker targeting**: MPI rank is accessible inside the wrapped simulate function via `mpi4py.MPI.COMM_WORLD.Get_rank()`. All Propulate ranks are workers (no master/worker distinction). The straggler wrapper checks `if MPI.COMM_WORLD.Get_rank() == straggler_rank: time.sleep(slowdown_factor * base_sleep_s)`. Default `straggler_rank = 0`, configurable in `straggler.json`. In test mode, skip the sleep entirely.

5. **Phase 6 — Grid explosion**: Keep the full 108-variant grid as designed. Use `compute_scaling_factor()` (already handles `sensitivity_grid`) to estimate full runtime from a test run. Decision on whether to reduce the grid will be made based on that estimate.

---

## Open Questions (Remaining)

None — all design decisions resolved.
