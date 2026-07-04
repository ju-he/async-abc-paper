# Codebase Structure

**Analysis Date:** 2026-04-08

## Directory Layout

```
async-abc-paper/
├── experiments/                    # Main Python package and experiment code
│   ├── run_all_paper_experiments.py  # Top-level orchestrator
│   ├── async_abc/                  # Core library package (installed via pyproject.toml)
│   │   ├── __init__.py
│   │   ├── analysis/               # Post-processing: convergence, ESS, SBC, trajectories
│   │   ├── benchmarks/             # Simulation models (gaussian_mean, gandk, etc.)
│   │   ├── inference/              # ABC method implementations + registry
│   │   ├── io/                     # Config, paths, records, schema
│   │   ├── plotting/               # Matplotlib figure generation
│   │   ├── reporting/              # Summary tables and runtime metrics
│   │   └── utils/                  # Runner harness, MPI, sharding, seeding
│   ├── scripts/                    # CLI runner scripts (one per experiment type)
│   ├── configs/                    # JSON experiment configurations
│   │   └── small/                  # Reduced-compute config variants
│   ├── jobs/                       # SLURM job scripts and submission helpers
│   ├── tests/                      # pytest test suite
│   ├── data/                       # Reference data (realistic workload model)
│   └── assets/                     # Static assets (realistic workload configs)
├── sim_backend_venv -> ...            # Symlink to sim_backend (cellular Potts simulator)
├── propulate -> ...                # Symlink to propulate (async evolutionary optimizer)
├── latex/                          # Paper manuscript
│   └── sn-article-template/        # Springer Nature article template
│       ├── sn-article.tex          # Main manuscript
│       ├── sn-bibliography.bib     # References
│       ├── figures/                # Paper figures (PNG/PDF)
│       └── tables/                 # Paper tables
│           └── generated/          # Auto-generated LaTeX tables
├── .plans/                         # Planning documents and bug fix history
│   ├── bug-fixes/
│   └── ressources/
├── .planning/                      # GSD planning output
│   └── codebase/                   # This document lives here
├── pyproject.toml                  # Package definition and dependencies
├── uv.lock                         # Dependency lockfile (uv)
├── CLAUDE.md                       # Claude Code project instructions
├── AGENTS.md                       # Agent instructions
└── README.md                       # Project overview
```

## Directory Purposes

**`experiments/async_abc/`:**
- Purpose: Installable Python package containing all library code
- Contains: Six sub-packages (analysis, benchmarks, inference, io, plotting, reporting, utils)
- Key files: `__init__.py` (package docstring only)
- Installed as `async_abc` via `pyproject.toml` (`[tool.setuptools.package-dir] "" = "experiments"`)

**`experiments/async_abc/benchmarks/`:**
- Purpose: Simulation model definitions
- Contains: One module per benchmark model, each exporting a class
- Key files: `__init__.py` (registry `_REGISTRY` + `make_benchmark()`), `gaussian_mean.py`, `gandk.py`, `lotka_volterra.py`, `realistic_workload.py`

**`experiments/async_abc/inference/`:**
- Purpose: ABC inference method implementations
- Contains: Method runners + registry dispatch
- Key files: `method_registry.py` (central dispatch), `propulate_abc.py`, `pyabc_wrapper.py`, `rejection_abc.py`, `abc_smc_baseline.py`, `_attempt_trace.py`, `_pyabc_common.py`, `_pyabc_history.py`, `pyabc_sampler.py`

**`experiments/async_abc/io/`:**
- Purpose: Configuration loading, validation, output path management, CSV record I/O
- Contains: Config pipeline and data serialization
- Key files: `config.py` (load_config), `schema.py` (validation constants), `paths.py` (OutputDir), `records.py` (ParticleRecord, RecordWriter)

**`experiments/async_abc/analysis/`:**
- Purpose: Statistical analysis of experiment results
- Contains: Convergence metrics, ESS, SBC diagnostics, trajectory extraction
- Key files: `convergence.py`, `sbc.py`, `ess.py`, `barrier.py`, `trajectory.py`, `final_state.py`, `sensitivity.py`, `audit.py`, `_helpers.py`

**`experiments/async_abc/plotting/`:**
- Purpose: Generate all figures (paper summaries + diagnostics)
- Contains: Reusable plot primitives and high-level orchestrators
- Key files: `common.py` (plot functions), `reporters.py` (plot orchestrators), `export.py` (save_figure with metadata), `sbc.py` (SBC-specific plots)

**`experiments/async_abc/reporting/`:**
- Purpose: Generate summary tables, runtime metrics, plot metadata
- Key files: `benchmark_reports.py`, `runtime_summary.py`, `plot_metadata.py`

**`experiments/async_abc/utils/`:**
- Purpose: Shared execution infrastructure
- Key files: `runner.py` (run_experiment, run_method_distributed, timing), `benchmark_runner.py` (run_benchmark_runner), `sharding.py` (ShardLayout, shard management), `mpi.py` (rank helpers), `seeding.py`, `progress.py`, `logging_utils.py`, `metadata.py`, `git.py`, `shard_finalizers.py`

**`experiments/scripts/`:**
- Purpose: CLI entry-point scripts for each experiment type
- Contains: `*_runner.py` files, utility scripts
- Key files: `gandk_runner.py`, `gaussian_mean_runner.py`, `lotka_volterra_runner.py`, `cellular_potts_runner.py`, `scaling_runner.py`, `sensitivity_runner.py`, `sbc_runner.py`, `straggler_runner.py`, `runtime_heterogeneity_runner.py`, `ablation_runner.py`, `replot.py`, `generate_cpm_reference.py`

**`experiments/configs/`:**
- Purpose: JSON experiment configuration files
- Contains: One `.json` per experiment, plus `small/` subdirectory for reduced-compute variants
- Key files: `gandk.json`, `gaussian_mean.json`, `lotka_volterra.json`, `realistic_workload.json`, `sbc.json`, `scaling.json`, `sensitivity.json`, `sensitivity_gandk.json`, `straggler.json`, `runtime_heterogeneity.json`, `ablation.json`

**`experiments/jobs/`:**
- Purpose: SLURM submission scripts and helpers
- Key files: `run_experiments.sh` (main SLURM batch script), `scaling_packed.sh`, `scaling_single.sh`, `submit_replicate_shards.py`, `submit_scaling.py`, `test_all.sh`, `test_sharded.sh`

**`experiments/tests/`:**
- Purpose: pytest test suite
- Key files: `conftest.py`, `test_benchmarks.py`, `test_config.py`, `test_inference.py`, `test_runners.py`, `test_sbc.py`, `test_analysis.py`, `test_plotting.py`, `test_records.py`, `test_paths.py`, `test_seeding.py`, `test_sharding.py`, `test_sensitivity_metric.py`, `test_parallel_coordination.py`, `test_progress.py`, `test_extend.py`, `test_phase6.py`

**`experiments/data/`:**
- Purpose: Reference data for benchmarks (primarily cellular Potts)
- Contains: `cpm_reference/`, `cpm_reference_debug/`

**`latex/sn-article-template/`:**
- Purpose: Paper manuscript and generated figures/tables
- Key files: `sn-article.tex`, `sn-bibliography.bib`, `figures/`, `tables/generated/`

## Key File Locations

**Entry Points:**
- `experiments/run_all_paper_experiments.py`: Top-level orchestrator for all experiments
- `experiments/scripts/*_runner.py`: Individual experiment runners

**Configuration:**
- `pyproject.toml`: Package definition, dependencies, pytest config
- `experiments/configs/*.json`: Experiment configurations
- `experiments/configs/small/*.json`: Reduced-compute config variants
- `experiments/async_abc/io/schema.py`: Config validation rules

**Core Logic:**
- `experiments/async_abc/utils/runner.py`: Central experiment execution (run_experiment, run_method_distributed)
- `experiments/async_abc/utils/benchmark_runner.py`: Shared runner entry point with sharding
- `experiments/async_abc/inference/method_registry.py`: Method dispatch (METHOD_REGISTRY, run_method)
- `experiments/async_abc/benchmarks/__init__.py`: Benchmark dispatch (make_benchmark)

**Testing:**
- `experiments/tests/conftest.py`: Shared fixtures
- `experiments/tests/test_*.py`: Test modules

## Naming Conventions

**Files:**
- Library modules: `snake_case.py` (e.g., `gaussian_mean.py`, `method_registry.py`)
- Private/internal modules: `_prefixed.py` (e.g., `_attempt_trace.py`, `_pyabc_common.py`, `_helpers.py`)
- Runner scripts: `<experiment_name>_runner.py` (e.g., `gandk_runner.py`)
- Config files: `<experiment_name>.json` (e.g., `gandk.json`)
- Test files: `test_<module>.py` (e.g., `test_benchmarks.py`)

**Directories:**
- Package directories: `snake_case` (e.g., `async_abc`, `io`)
- No nested packages beyond two levels (`async_abc/<subpackage>/`)

**Classes:**
- PascalCase: `GaussianMean`, `GandK`, `LotkaVolterra`, `RealisticWorkload`, `OutputDir`, `ParticleRecord`, `RecordWriter`, `ShardLayout`

**Functions:**
- snake_case: `run_experiment()`, `make_benchmark()`, `run_method()`, `load_config()`
- Private functions: `_prefixed()` (e.g., `_validate()`, `_apply_test_mode()`)

**Config keys:**
- snake_case strings: `"experiment_name"`, `"max_simulations"`, `"n_workers"`, `"base_seed"`

## Where to Add New Code

**New Benchmark Model:**
- Create: `experiments/async_abc/benchmarks/<name>.py` with a class implementing `simulate(params, seed) -> float` and `limits` attribute
- Register: Add to `_REGISTRY` dict in `experiments/async_abc/benchmarks/__init__.py`
- Add name to `VALID_BENCHMARK_NAMES` in `experiments/async_abc/io/schema.py`
- Create config: `experiments/configs/<name>.json` (and `experiments/configs/small/<name>.json`)
- Create runner: `experiments/scripts/<name>_runner.py` (delegate to `run_benchmark_runner`)
- Register in: `EXPERIMENT_REGISTRY` in `experiments/run_all_paper_experiments.py`
- Add tests: `experiments/tests/test_benchmarks.py`

**New Inference Method:**
- Create: `experiments/async_abc/inference/<name>.py` with a runner function matching the signature: `runner(simulate_fn, limits, inference_cfg, output_dir, replicate, seed, progress=None) -> List[ParticleRecord]`
- Register: Add to `METHOD_REGISTRY` and `METHOD_EXECUTION_MODE` in `experiments/async_abc/inference/method_registry.py`
- Add tests: `experiments/tests/test_inference.py`

**New Analysis Function:**
- Create: function in appropriate `experiments/async_abc/analysis/*.py` module
- Register: Add to `_EXPORTS` dict and `__all__` in `experiments/async_abc/analysis/__init__.py`

**New Plot Type:**
- Add plot primitive: `experiments/async_abc/plotting/common.py`
- Add orchestrator: `experiments/async_abc/plotting/reporters.py`
- Register: Add to `_EXPORTS` and `__all__` in `experiments/async_abc/plotting/__init__.py`

**New Experiment Type (non-standard):**
- Create runner: `experiments/scripts/<name>_runner.py` with custom main loop
- Create config: `experiments/configs/<name>.json`
- Register in: `EXPERIMENT_REGISTRY` in `experiments/run_all_paper_experiments.py`

**Utilities:**
- Shared helpers: `experiments/async_abc/utils/`
- MPI helpers: `experiments/async_abc/utils/mpi.py`

## Special Directories

**`sim_backend_venv` (symlink):**
- Purpose: External dependency for cellular Potts model simulation (sim_backend)
- Generated: No (symlink to `/home/juhe/bwSyncShare/Code/mirrors/sim_backend_venv`)
- Committed: Symlink committed, target is external
- Contains its own `.venv` used for testing

**`propulate` (symlink):**
- Purpose: External dependency for asynchronous evolutionary optimization
- Generated: No (symlink to `/home/juhe/bwSyncShare/Code/propulate/`)
- Committed: Symlink committed, target is external

**`experiments/data/`:**
- Purpose: Reference data for cellular Potts benchmark
- Generated: Yes, via `experiments/scripts/generate_cpm_reference.py`
- Committed: Yes

**`latex/sn-article-template/tables/generated/`:**
- Purpose: Auto-generated LaTeX tables from experiment results
- Generated: Yes
- Committed: Yes

**`latex/sn-article-template/figures/`:**
- Purpose: Paper figures copied from experiment output
- Generated: Partially (some auto-generated, some manual)
- Committed: Yes

---

*Structure analysis: 2026-04-08*
