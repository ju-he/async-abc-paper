# Detailed Code Checklist for Reproducible Experiments

This checklist describes the code structure needed to run, reproduce, and extend all experiments for the asynchronous steady-state ABC-SMC paper. It is organized so that a reviewer can reproduce **all results and plots with one command**, while still allowing each benchmark or analysis to be run independently from a JSON config.

## 1. Repository-Level Layout

```text
project_root/
  async_abc/
    __init__.py
    propagators/
    schedulers/
    benchmarks/
    inference/
    runtime/
    plotting/
    io/
    utils/
  experiments/
    configs/
      gaussian/
      gandk/
      lotka_volterra/
      cellular_potts/
      runtime_heterogeneity/
      scaling/
      sensitivity/
      ablation/
      paper/
    scripts/
      run_gaussian_mean.py
      run_gandk.py
      run_lotka_volterra.py
      run_cellular_potts.py
      run_runtime_heterogeneity.py
      run_scaling.py
      run_sensitivity.py
      run_ablation.py
      run_all_paper_experiments.py
    templates/
      plot_styles.py
      report_helpers.py
  results/
    raw/
    processed/
    plots/
    metadata/
    logs/
  tests/
    test_configs/
    test_benchmarks/
    test_plotting/
    test_end_to_end/
  requirements.txt
  pyproject.toml
  README.md
```

## 2. Core Requirements

The codebase should support these guarantees.

- Every experiment has a dedicated runner script.
- Every runner script accepts a `--config <json>` argument.
- Every runner script accepts a `--output-dir <path>` argument.
- Every runner script accepts a `--test` flag.
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
- A single top-level script must run all experiments and regenerate all plots.

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

- load processed data
- generate plot
- export plot data CSV
- export plot metadata JSON
- save figure as PDF and PNG

Recommended plot metadata fields:

```json
{
  "plot_name": "scaling_efficiency",
  "source_processed_csv": "processed/scaling_summary.csv",
  "source_raw_files": ["raw/run_001.csv", "raw/run_002.csv"],
  "script": "experiments/scripts/run_scaling.py",
  "resolved_config": "resolved_config.json",
  "x": "n_workers",
  "y": "efficiency_mean",
  "group_by": ["method", "benchmark"],
  "filters": {"test_mode": false},
  "aggregation": "mean_over_seeds",
  "error_bars": "std",
  "figure_files": ["plots/scaling_efficiency.pdf", "plots/scaling_efficiency.png"]
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

- `async_abc/benchmarks/gaussian_mean.py`
- `experiments/scripts/run_gaussian_mean.py`
- `experiments/configs/gaussian/default.json`
- `experiments/configs/gaussian/test.json`

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

- `async_abc/benchmarks/gandk.py`
- `experiments/scripts/run_gandk.py`
- `experiments/configs/gandk/default.json`
- `experiments/configs/gandk/test.json`

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

- `async_abc/benchmarks/lotka_volterra.py`
- `experiments/scripts/run_lotka_volterra.py`
- `experiments/configs/lotka_volterra/default.json`
- `experiments/configs/lotka_volterra/test.json`

Required functionality:

- simulator wrapper
- observed trajectory generation
- summary statistics
- discrepancy function
- optional variable runtime instrumentation

Outputs:

- posterior recovery plots
- runtime and simulation budget comparisons
- CSVs and metadata

### 4.4 Cellular Potts benchmark using cellsInSilico / nastjapy

Required files:

- `async_abc/benchmarks/cellular_potts.py`
- `async_abc/benchmarks/cellular_potts_wrappers.py`
- `experiments/scripts/run_cellular_potts.py`
- `experiments/configs/cellular_potts/default.json`
- `experiments/configs/cellular_potts/test.json`

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

- `run_gaussian_mean.py`
- `run_gandk.py`
- `run_lotka_volterra.py`
- `run_cellular_potts.py`

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
python experiments/scripts/run_gaussian_mean.py \
  --config experiments/configs/gaussian/default.json \
  --output-dir results/gaussian
```

and

```bash
python experiments/scripts/run_gaussian_mean.py \
  --config experiments/configs/gaussian/default.json \
  --output-dir results/gaussian_test \
  --test
```

### 5.2 Runtime heterogeneity experiment

Purpose:

- quantify benefits of asynchronous execution when simulation runtime varies strongly

Required files:

- `experiments/scripts/run_runtime_heterogeneity.py`
- `experiments/configs/runtime_heterogeneity/default.json`
- `experiments/configs/runtime_heterogeneity/test.json`
- `async_abc/runtime/heterogeneity.py`

Required functionality:

- inject controlled synthetic runtime distributions into at least Gaussian and Lotka-Volterra benchmarks
- compare methods under increasing runtime variance
- record utilization and idle fractions

Suggested config fields:

```json
{
  "benchmarks": ["gaussian_mean", "lotka_volterra"],
  "methods": ["async_propulate_abc", "pyabc_smc"],
  "runtime_model": {
    "type": "lognormal",
    "variance_levels": [0.0, 0.5, 1.0, 1.5]
  },
  "execution": {
    "n_workers": [8, 32, 128],
    "seeds": [1, 2, 3]
  }
}
```

Outputs:

- efficiency vs runtime variance plot
- idle fraction vs variance plot
- throughput vs variance plot
- each with CSV and metadata

### 5.3 Scaling experiment

Purpose:

- show strong and weak scaling characteristics

Required files:

- `experiments/scripts/run_scaling.py`
- `experiments/configs/scaling/default.json`
- `experiments/configs/scaling/test.json`

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

Required files:

- `experiments/scripts/run_sensitivity.py`
- `experiments/configs/sensitivity/default.json`
- `experiments/configs/sensitivity/test.json`

Parameters to vary:

- archive size `k`
- perturbation scale
- tolerance scheduler type
- scheduler parameters
- optional archive selection strategy

Suggested config structure:

```json
{
  "benchmarks": ["gandk", "lotka_volterra"],
  "methods": ["async_propulate_abc"],
  "parameter_grid": {
    "archive_size": [50, 100, 200],
    "perturbation_scale": [0.5, 0.8, 1.2],
    "scheduler_type": ["quantile", "geometric_decay", "acceptance_rate"]
  },
  "execution": {
    "n_workers": 32,
    "seeds": [1,2,3,4,5]
  }
}
```

Outputs:

- heatmaps or line plots for posterior error vs hyperparameter
- runtime vs hyperparameter
- CSV and metadata

### 5.5 Ablation study

Purpose:

- isolate the contribution of algorithmic components

Required files:

- `experiments/scripts/run_ablation.py`
- `experiments/configs/ablation/default.json`
- `experiments/configs/ablation/test.json`

Ablations to include:

- fixed covariance vs adaptive covariance
- fixed tolerance vs adaptive tolerance
- archive truncation variants
- weighting variants if applicable

Outputs:

- component contribution plots
- summary tables
- CSV and metadata

### 5.6 One-command full reproduction script

Required file:

- `experiments/scripts/run_all_paper_experiments.py`

Required behavior:

- sequentially or selectively launch every experiment set
- accept global `--output-dir`
- accept global `--test`
- optionally accept `--only` to restrict to a subset
- generate a final manifest listing all produced result files

Suggested CLI:

```bash
python experiments/scripts/run_all_paper_experiments.py \
  --output-dir results/paper_full
```

and for pipeline validation:

```bash
python experiments/scripts/run_all_paper_experiments.py \
  --output-dir results/paper_test \
  --test
```

The script should invoke:

1. Gaussian mean benchmark
2. g-and-k benchmark
3. Lotka-Volterra benchmark
4. Cellular Potts benchmark
5. Runtime heterogeneity experiment
6. Scaling experiment
7. Sensitivity analysis
8. Ablation study

It should also write:

- `results_manifest.json`
- `plots_manifest.json`
- `environment_metadata.json`

## 6. Plot and Data Export Requirements

For every plot-producing script, the following must be emitted.

### 6.1 Plot image files

- `plot_name.pdf`
- `plot_name.png`

### 6.2 Plot data CSV

This CSV must contain the exact data used for the final figure.

Examples:

- `posterior_overlay_gaussian.csv`
- `scaling_efficiency.csv`
- `runtime_heterogeneity_idle_fraction.csv`

### 6.3 Plot metadata JSON

Each plot metadata file should capture:

- experiment script path
- benchmark name
- methods shown
- source raw files
- source processed CSVs
- resolved config path
- aggregation recipe
- filters used
- plotting columns
- error-bar definition
- random seeds included
- git commit hash if available
- package versions if available

## 7. Default Configs That Must Exist

Create a default and test config for each experiment family.

Required files:

```text
experiments/configs/gaussian/default.json
experiments/configs/gaussian/test.json
experiments/configs/gandk/default.json
experiments/configs/gandk/test.json
experiments/configs/lotka_volterra/default.json
experiments/configs/lotka_volterra/test.json
experiments/configs/cellular_potts/default.json
experiments/configs/cellular_potts/test.json
experiments/configs/runtime_heterogeneity/default.json
experiments/configs/runtime_heterogeneity/test.json
experiments/configs/scaling/default.json
experiments/configs/scaling/test.json
experiments/configs/sensitivity/default.json
experiments/configs/sensitivity/test.json
experiments/configs/ablation/default.json
experiments/configs/ablation/test.json
experiments/configs/paper/default.json
experiments/configs/paper/test.json
```

The `paper/default.json` should point to the default configs for all sub-experiments. The `paper/test.json` should point to the test configs.

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

A reviewer with sufficient compute resources should be able to run:

```bash
python experiments/scripts/run_all_paper_experiments.py \
  --config experiments/configs/paper/default.json \
  --output-dir results/reviewer_run
```

A reviewer validating the full pipeline quickly should be able to run:

```bash
python experiments/scripts/run_all_paper_experiments.py \
  --config experiments/configs/paper/test.json \
  --output-dir results/reviewer_test \
  --test
```

The latter must exercise the complete pipeline end-to-end while staying within the following constraints:

- no more than 8 CPUs
- small simulation budgets
- reduced scaling grid
- reduced seeds and replicates
- short Cellular Potts runs

## 10. Minimum Testing Checklist

Before paper release, verify all of the following.

- [ ] Every script runs with a default config.
- [ ] Every script runs with `--test`.
- [ ] Every script writes a resolved config.
- [ ] Every script writes raw results.
- [ ] Every script writes processed summaries.
- [ ] Every plot has both PDF and PNG outputs.
- [ ] Every plot has a CSV file with plotted data.
- [ ] Every plot has a metadata JSON file.
- [ ] The full reproduction script works in `--test` mode on one machine with at most 8 CPUs.
- [ ] The full reproduction script creates a manifest of outputs.
- [ ] pyABC comparisons are included where configured.
- [ ] Gaussian, g-and-k, Lotka-Volterra, and Cellular Potts benchmarks all run end-to-end.
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
