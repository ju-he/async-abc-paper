# Detailed Code Checklist for Reproducible Experiments

This checklist describes the code structure needed to run, reproduce, and extend the experiments for the asynchronous steady-state ABC-SMC paper. This file has been updated to match the current implementation, especially the plotting pipeline, which now distinguishes paper-facing summary plots from replicate-level diagnostics and can explicitly skip paper plots when the recorded data fail an audit.

## 1. Repository-Level Layout

```text
project_root/
  experiments/
    async_abc/
      analysis/
      benchmarks/
      inference/
      io/
      plotting/
      utils/
    configs/
      *.json
      small/
        *.json
    scripts/
      gaussian_mean_runner.py
      gandk_runner.py
      lotka_volterra_runner.py
      cellular_potts_runner.py
      runtime_heterogeneity_runner.py
      scaling_runner.py
      sensitivity_runner.py
      ablation_runner.py
      straggler_runner.py
      sbc_runner.py
      replot.py
    jobs/
      submit_replicate_shards.py
  .plans/
    ressources/
  nastjapy_copy/.venv/
```

## 2. Core Requirements

The codebase should support these guarantees.

- Every experiment has a dedicated runner script.
- Every runner script accepts a `--config <json>` argument.
- Every runner script accepts a `--output-dir <path>` argument.
- Every runner script accepts a `--test` flag.
- Benchmark-style runners also support `--small`.
- The `--test` flag must:
  - use at most 8 CPUs
  - reduce simulation budgets drastically
  - reduce repetitions / seeds
  - reduce scaling ranges
  - shorten simulator runtimes or use lightweight simulator settings
  - still execute the full data-generation, aggregation, CSV export, metadata export, and plotting pipeline
- Every plot must produce:
  - an image file, preferably `.pdf` and `.png`
  - a plot CSV with the plotted data
  - a metadata JSON describing how the plot was created
- Benchmark plots support a paper-facing summary mode and, where useful, a replicate-level diagnostic mode.
- Paper-facing plots may be intentionally skipped when audit checks fail; this must still emit metadata documenting the skip.
- Plot regeneration from saved outputs is supported by `experiments/scripts/replot.py`.

## 3. Shared Infrastructure That Must Exist

### 3.1 Config loading and validation

Create a small config layer.

Required modules:

- `async_abc/io/config.py`
- `async_abc/io/schema.py`

Required functionality:

- load JSON config
- validate required fields
- merge user config with defaults
- apply `--test` overrides cleanly
- save resolved config into output directory for provenance

Recommended functions:

```python
load_config(path: str) -> dict
merge_with_defaults(user_cfg: dict, default_cfg: dict) -> dict
apply_test_overrides(cfg: dict) -> dict
save_resolved_config(cfg: dict, out_path: str) -> None
```

### 3.2 Output directory management

Required module:

- `async_abc/io/paths.py`

Functionality:

- create stable experiment run directory
- timestamped or hash-based subdirectories
- separate locations for raw results, processed data, plots, metadata, and logs

Suggested structure per run:

```text
<output-dir>/
  resolved_config.json
  logs/
  raw/
  processed/
  plots/
  metadata/
```

### 3.3 Reproducibility and randomness

Required module:

- `async_abc/utils/seeding.py`

Functionality:

- global seed initialization
- per-repetition seeds
- per-worker seeds if multiprocessing / MPI is used
- record all seeds in metadata

### 3.4 Common experiment record format

Required module:

- `async_abc/io/records.py`

Each experiment should write row-wise raw results to CSV or parquet with stable column names.

Suggested minimum columns:

- `experiment_name`
- `benchmark`
- `method`
- `seed`
- `replicate`
- `n_workers`
- `test_mode`
- `wallclock_seconds`
- `simulations_total`
- `accepted_total`
- `posterior_error`
- `wasserstein`
- `coverage`
- `throughput`
- `worker_utilization`
- `idle_fraction`
- `config_hash`

Not every experiment will use every field, but consistent schema helps aggregation.

### 3.5 Plotting helpers

Required modules:

- `async_abc/plotting/common.py`
- `async_abc/plotting/export.py`

Required functionality:

- load saved experiment outputs
- generate plot
- export plot data CSV
- export plot metadata JSON
- export explicit skip metadata for suppressed paper plots
- save figure as PDF and PNG

Recommended plot metadata fields:

```json
{
  "plot_name": "scaling_efficiency",
  "source_raw_files": ["raw/run_001.csv", "raw/run_002.csv"],
  "summary_plot": true,
  "diagnostic_plot": false,
  "x": "n_workers",
  "y": "efficiency_mean",
  "group_by": ["method", "benchmark"],
  "filters": {"test_mode": false},
  "aggregation": "mean_over_replicates",
  "ci_level": 0.95,
  "skipped": false,
  "skip_reason": null
}
```

### 3.6 Method wrappers

Required module:

- `async_abc/inference/method_registry.py`

This should standardize calls to:

- asynchronous Propulate ABC
- pyABC baseline
- optional classical ABC-SMC baseline implementation
- optional rejection ABC baseline for small models

All benchmark scripts should call methods through a common interface.

Suggested interface:

```python
run_inference(method_name: str, benchmark_name: str, cfg: dict, seed: int) -> dict
```

## 4. Benchmark Implementations Needed

### 4.1 Gaussian mean inference benchmark

Required files:

- `experiments/async_abc/benchmarks/gaussian_mean.py`
- `experiments/scripts/gaussian_mean_runner.py`
- `experiments/configs/gaussian_mean.json`
- `experiments/configs/small/gaussian_mean.json`

Required functionality:

- generate observed data for known ground-truth mean
- analytic posterior computation for comparison
- simulator returning Gaussian draws or summary statistics
- discrepancy function
- optional runtime delay hook for synthetic runtime studies

Minimum outputs:

- raw posterior summaries by seed and method
- processed comparison table
- posterior overlay plot data and figures
- accuracy metrics CSV
- metadata JSON for each plot

Default config should include:

```json
{
  "experiment_name": "gaussian_mean",
  "benchmark": {
    "n_obs": 100,
    "true_mu": 1.5,
    "sigma": 1.0,
    "prior": {"type": "uniform", "low": -5.0, "high": 5.0},
    "summary": "sample_mean"
  },
  "methods": ["async_propulate_abc", "pyabc_smc", "abc_smc_baseline"],
  "inference": {
    "population_size": 100,
    "max_simulations": 20000,
    "tolerance_schedule": "quantile",
    "quantile": 0.5
  },
  "execution": {
    "n_workers": 16,
    "n_replicates": 20,
    "seeds": [1,2,3,4,5]
  },
  "plots": {
    "make_posterior_overlay": true,
    "make_error_bars": true,
    "make_runtime_comparison": true
  }
}
```

`--test` overrides should reduce to something like:

- `n_workers <= 8`
- `max_simulations = 500`
- `n_replicates = 2`
- `seeds = [1]`
- fewer posterior draws

### 4.2 g-and-k benchmark

Required files:

- `experiments/async_abc/benchmarks/gandk.py`
- `experiments/scripts/gandk_runner.py`
- `experiments/configs/gandk.json`
- `experiments/configs/small/gandk.json`

Required functionality:

- simulator for g-and-k distribution
- summaries commonly used in ABC literature
- discrepancy metric
- metrics against known parameters

Outputs:

- posterior comparisons
- parameter recovery table
- Wasserstein / coverage summaries
- plot CSVs and metadata

### 4.3 Lotka-Volterra benchmark

Required files:

- `experiments/async_abc/benchmarks/lotka_volterra.py`
- `experiments/scripts/lotka_volterra_runner.py`
- `experiments/configs/lotka_volterra.json`
- `experiments/configs/small/lotka_volterra.json`

Required functionality:

- simulator wrapper
- observed trajectory generation
- summary statistics
- discrepancy function
- optional variable runtime instrumentation

Outputs:

- posterior recovery plots
- runtime and simulation budget comparisons
- Lotka-specific fallback / `tol_init` calibration diagnostics
- CSVs and metadata

### 4.4 Cellular Potts benchmark using cellsInSilico / nastjapy

Required files:

- `experiments/async_abc/benchmarks/cellular_potts.py`
- `experiments/async_abc/benchmarks/cellular_potts_wrappers.py`
- `experiments/scripts/cellular_potts_runner.py`
- `experiments/configs/cellular_potts.json`
- `experiments/configs/small/cellular_potts.json`

Required functionality:

- wrapper around `cellsInSilico` / `nastjapy`
- parameterized simulation entry point
- summary statistics extraction
- discrepancy function
- runtime logging per simulation
- graceful fallback or skip if dependencies are unavailable

Test mode requirements:

- small lattice / domain size
- short simulated time horizon
- few replicates
- at most 8 CPUs
- still produce all result files and plots

Outputs:

- posterior comparison
- runtime histogram / heterogeneity analysis
- scaling plots
- data CSVs and metadata JSONs

## 5. Experiment Sets and Required Scripts

### 5.1 Statistical benchmark scripts

Scripts:

- `gaussian_mean_runner.py`
- `gandk_runner.py`
- `lotka_volterra_runner.py`
- `cellular_potts_runner.py`

Each script must:

1. load config
2. apply `--test` overrides if requested
3. run configured methods over all seeds / replicates
4. save raw run table
5. aggregate processed summaries
6. generate plots
7. export plot CSVs
8. export plot metadata JSONs

Command-line interface each script should support:

```bash
python experiments/scripts/gaussian_mean_runner.py \
  --config experiments/configs/gaussian_mean.json \
  --output-dir results/gaussian
```

and

```bash
python experiments/scripts/gaussian_mean_runner.py \
  --config experiments/configs/gaussian_mean.json \
  --output-dir results/gaussian_test \
  --test
```

### 5.2 Runtime heterogeneity experiment

Purpose:

- quantify benefits of asynchronous execution when simulation runtime varies strongly
- measure idle worker fraction and posterior quality over wall-clock time under increasing variance

Required files:

- `experiments/scripts/runtime_heterogeneity_runner.py`
- `experiments/configs/runtime_heterogeneity.json`
- `experiments/configs/small/runtime_heterogeneity.json`

Required functionality:

- wrap the benchmark simulator with a per-evaluation LogNormal sleep to model heterogeneous HPC workloads
- sweep over multiple `sigma_levels` in a single run; each sigma level shares the same replicate budget
- per-replicate delay seeds derived via `stable_seed(base_seed, replicate_idx, sigma)` — no shared hardcoded seed
- sleep is injected **after** the simulation call, so Propulate `evaltime`/`evalperiod` records the full busy span
- in `--test` mode the sleep is skipped entirely
- record utilization and idle fractions, posterior quality curves, and speedup summary

Actual config structure:

```json
{
  "experiment_name": "runtime_heterogeneity",
  "benchmark": { "name": "gaussian_mean", ... },
  "methods": ["async_propulate_abc", "abc_smc_baseline"],
  "inference": { "max_simulations": 20000, "n_workers": 48, ... },
  "execution": { "n_replicates": 5, "base_seed": 0 },
  "heterogeneity": {
    "distribution": "lognormal",
    "base_delay_s": 1.0,
    "sigma_levels": [0.0, 0.5, 1.0, 1.5, 2.0]
  },
  "plots": {
    "idle_fraction": true,
    "throughput_over_time": true,
    "idle_fraction_comparison": true,
    "gantt": true,
    "quality_by_sigma": true
  }
}
```

`base_delay_s` sets the **median** delay per evaluation (`mu = log(base_delay_s)`).
`sigma_levels` sweeps the spread of the delay distribution.

Outputs:

- `data/raw_results.csv` — particle records; method names include `__sigma{X}` tag
- `data/speedup_summary.csv` — per-(sigma, base_method) median completion time and speedup ratio vs `abc_smc_baseline`
- `data/runtime_debug_summary.csv` — per-worker timing debug info
- `plots/quality_by_sigma.pdf` — **headline figure**: Wasserstein distance vs wall-clock time, one panel per sigma level, async vs sync overlaid
- `plots/idle_fraction_comparison.pdf` — utilization-loss fraction vs sigma, both methods
- `plots/idle_fraction.pdf` — per-method idle fraction summary
- `plots/throughput_over_time.pdf` — simulations/s over time, faceted by sigma
- `plots/worker_gantt.pdf` — diagnostic per-worker timeline (not a paper figure)
- each with accompanying CSV and metadata

Note: `plot_benchmark_diagnostics` is **not** called in this runner. The `quality_by_sigma` plot
replaces its function for the heterogeneity experiment (calling it on the combined
`method__sigma{X}` records would produce cluttered 10-method plots).

### 5.3 Scaling experiment

Purpose:

- show strong and weak scaling characteristics

Required files:

- `experiments/scripts/scaling_runner.py`
- `experiments/configs/scaling.json`
- `experiments/configs/small/scaling.json`

Required functionality:

- run selected benchmarks at increasing worker counts
- compare async method to pyABC
- compute speedup and efficiency

Suggested worker counts default:

```json
[1, 8, 32, 128, 256]
```

Test worker counts:

```json
[1, 2, 4, 8]
```

Outputs:

- wall-clock vs workers
- speedup vs workers
- efficiency vs workers
- simulations per second vs workers
- CSV and metadata for all plots

### 5.4 Sensitivity analysis

Purpose:

- show robustness to method hyperparameters

Implemented files:

- `experiments/scripts/sensitivity_runner.py`
- `experiments/configs/sensitivity.json`
- `experiments/configs/small/sensitivity.json`

Parameters to vary:

- archive size `k`
- perturbation scale
- tolerance scheduler type
- initial tolerance multiplier `tol_init_multiplier`

Current config shape:

```json
{
  "experiment_name": "sensitivity",
  "methods": ["async_propulate_abc"],
  "sensitivity_grid": {
    "k": [50, 100, 200],
    "perturbation_scale": [0.5, 0.8, 1.2],
    "scheduler_type": ["acceptance_rate", "quantile", "geometric_decay"],
    "tol_init_multiplier": [0.5, 1.0, 2.0, 5.0]
  }
}
```

Outputs:

- `sensitivity_heatmap.{pdf,png}`
- `sensitivity_heatmap_data.csv`
- `sensitivity_heatmap_meta.json`
- CSV and metadata

### 5.5 Ablation study

Purpose:

- isolate the contribution of algorithmic components

Implemented files:

- `experiments/scripts/ablation_runner.py`
- `experiments/configs/ablation.json`
- `experiments/configs/small/ablation.json`

Ablations to include:

- fixed covariance vs adaptive covariance
- fixed tolerance vs adaptive tolerance
- archive truncation variants
- weighting variants if applicable

Outputs:

- `ablation_comparison.{pdf,png}`
- `ablation_comparison_data.csv`
- `ablation_comparison_meta.json`
- mean final tolerance with 95% confidence intervals where enough data are available
- CSV and metadata

### 5.6 Replot and shard submission entry points

Implemented files:

- `experiments/scripts/replot.py`
- `experiments/jobs/submit_replicate_shards.py`

Current behavior:

- `replot.py` regenerates plots from saved outputs without rerunning inference.
- `submit_replicate_shards.py` prepares sharded submissions for benchmark and analysis runs.
- `--finalize-only` can now recover shard batches that failed during finalization if the shard payloads were already written.

## 6. Plot and Data Export Requirements

For every plot-producing script, the following must be emitted.

### 6.1 Plot image files

- `plot_name.pdf`
- `plot_name.png`

### 6.2 Plot data CSV

This CSV must contain the exact data used for the final figure.

Examples:

- `quality_vs_wall_time_data.csv`
- `quality_vs_wall_time_diagnostic_data.csv`
- `coverage_table_data.csv`
- `idle_fraction_data.csv`

Benchmark paper plots currently use canonical names:

- `archive_evolution`
- `tolerance_trajectory`
- `quality_vs_wall_time`
- `quality_vs_attempt_budget`
- `quality_vs_posterior_samples`
- `time_to_target_summary`
- `attempts_to_target_summary`

Benchmark diagnostics use explicit suffixes:

- `archive_evolution_diagnostic`
- `tolerance_trajectory_diagnostic`
- `quality_vs_wall_time_diagnostic`
- `quality_vs_attempt_budget_diagnostic`
- `quality_vs_posterior_samples_diagnostic`
- `time_to_target_diagnostic`
- `attempts_to_target_diagnostic`

### 6.3 Plot metadata JSON

Each plot metadata file should capture:

- source raw files
- plot kind (`summary_plot` / `diagnostic_plot`)
- aggregation recipe
- confidence interval level where relevant
- skip state and skip reason for audit-blocked plots
- git commit hash if available
- package versions if available

Benchmark outputs also emit:

- `plot_audit.csv`
- `plot_audit_summary.json`

Lotka-Volterra additionally emits:

- `lotka_tol_init_diagnostic.csv`
- `lotka_tol_init_diagnostic.json`

## 7. Configs That Currently Exist

Current experiment families use one main config plus a `small/` companion where applicable.

Key files:

```text
experiments/configs/gaussian_mean.json
experiments/configs/gandk.json
experiments/configs/lotka_volterra.json
experiments/configs/cellular_potts.json
experiments/configs/runtime_heterogeneity.json
experiments/configs/scaling.json
experiments/configs/sensitivity.json
experiments/configs/ablation.json
experiments/configs/straggler.json
experiments/configs/sbc.json
experiments/configs/small/gaussian_mean.json
experiments/configs/small/gandk.json
...
```

## 8. Logging and Provenance

Required module:

- `async_abc/io/provenance.py`

Each run should record:

- hostname
- CPU info if available
- MPI rank / world size if relevant
- package versions
- git commit hash
- start/end time
- duration
- resolved config path
- whether test mode was active

Outputs:

- `metadata/environment.json`
- `metadata/run_info.json`

## 9. Review-Ready Reproduction Workflow

A reviewer validating the plotting pipeline from existing outputs should be able to run:

```bash
python experiments/scripts/replot.py /path/to/run_root all
```

Small-budget execution currently relies on `--small` and `--test` rather than separate `paper/test.json` orchestration.

Quick-validation constraints remain:

- no more than 8 CPUs
- small simulation budgets
- reduced scaling grid
- reduced seeds and replicates
- short Cellular Potts runs

## 10. Minimum Testing Checklist

Before paper release, verify all of the following.

- [ ] Every script runs with a default config.
- [ ] Every script runs with `--test`.
- [ ] Every runner writes metadata and raw results.
- [ ] Every plot has both PDF and PNG outputs.
- [ ] Every plot has a CSV file with plotted data.
- [ ] Every plot has a metadata JSON file.
- [ ] Paper-facing benchmark plots emit summary outputs and diagnostic companions when enabled.
- [ ] Audit-blocked paper plots emit skip metadata instead of silent omission.
- [ ] pyABC comparisons are included where configured.
- [ ] Gaussian, g-and-k, Lotka-Volterra, Cellular Potts, SBC, and straggler experiments all run end-to-end.
- [ ] Runtime heterogeneity, scaling, sensitivity, and ablation experiments all run end-to-end.

## 11. Recommended Implementation Order

To minimize integration risk, implement in this order.

1. Shared config and output infrastructure
2. Gaussian benchmark end-to-end
3. Plot export and metadata export
4. g-and-k benchmark
5. Lotka-Volterra benchmark
6. pyABC method wrapper
7. Scaling and runtime heterogeneity scripts
8. Sensitivity and ablation scripts
9. Cellular Potts integration
10. Full `run_all_paper_experiments.py` orchestration
11. Test-mode validation for the entire pipeline

## 12. Final Deliverables Checklist

By the end, the repository should contain:

- [ ] All benchmark implementations
- [ ] All experiment runner scripts
- [ ] All default and test configs
- [ ] Common plotting and export utilities
- [ ] pyABC comparison wrapper
- [ ] One-command full reproduction script
- [ ] Reproduction manifests and provenance capture
- [ ] End-to-end test-mode execution path

This checklist is designed so that the experimental section of the paper is fully reproducible, modular, and reviewer-friendly without requiring manual regeneration of plots or ad hoc postprocessing.
