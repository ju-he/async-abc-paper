# Async-ABC Paper: Experiments Codebase — Multi-Phase TDD Plan

## Context

This plan implements the full experiment codebase for the async-ABC paper. The paper benchmarks an asynchronous steady-state ABC-SMC algorithm (already implemented as `ABCPMC` in `propulate/propulate/propagators/abcpmc.py`) against classical baselines (rejection ABC, synchronous ABC-SMC, pyABC) across four benchmark models. The experiments measure both statistical accuracy and computational performance on HPC systems.

All code lives under `experiments/`. Tests use `/home/juhe/bwSyncShare/Code/propulate/.venv/`.

**Core design rules (from `experiment_code_checklist.md`):**
- Every runner accepts `--config <json>`, `--output-dir <path>`, `--test` flag
- Test mode: ≤8 CPUs, max_simulations=500, n_replicates=2, seeds=[1]
- Every plot produces `.pdf` + `.png`, plot CSV, metadata JSON
- Config-driven, reproducible, provenance-tracked

---

## Directory Layout

```
experiments/
├── async_abc/                     # importable library
│   ├── __init__.py
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── gaussian_mean.py       # analytic posterior sanity check
│   │   ├── gandk.py               # g-and-k distribution
│   │   ├── lotka_volterra.py      # stochastic population dynamics
│   │   └── cellular_potts.py      # stub — requires nastjapy
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── method_registry.py     # METHOD_REGISTRY dict + run_method()
│   │   ├── propulate_abc.py       # thin wrapper around ABCPMC propagator
│   │   └── pyabc_wrapper.py       # wrapper around pyabc (optional dep)
│   ├── io/
│   │   ├── __init__.py
│   │   ├── config.py              # load_config(), validate_config()
│   │   ├── schema.py              # CONFIG_SCHEMA (jsonschema dict)
│   │   ├── paths.py               # OutputDir helper
│   │   └── records.py             # ParticleRecord + RecordWriter
│   ├── plotting/
│   │   ├── __init__.py
│   │   ├── common.py              # posterior_plot(), scaling_plot(), etc.
│   │   └── export.py              # save_figure() → .pdf + .png + metadata
│   └── utils/
│       ├── __init__.py
│       └── seeding.py             # make_seeds(), seed_everything()
├── configs/
│   ├── gaussian_mean.json
│   ├── gandk.json
│   ├── lotka_volterra.json
│   ├── runtime_heterogeneity.json
│   ├── scaling.json
│   ├── sensitivity.json
│   └── ablation.json
├── scripts/
│   ├── gaussian_mean_runner.py
│   ├── gandk_runner.py
│   ├── lotka_volterra_runner.py
│   ├── runtime_heterogeneity_runner.py
│   ├── scaling_runner.py
│   ├── sensitivity_runner.py
│   └── ablation_runner.py
├── tests/
│   ├── conftest.py                # shared fixtures
│   ├── test_config.py
│   ├── test_records.py
│   ├── test_seeding.py
│   ├── test_benchmarks.py
│   ├── test_inference.py
│   ├── test_plotting.py
│   └── test_runners.py
└── run_all_paper_experiments.py
```

---

## Phase 1 — Core Infrastructure (io + utils) ✅ COMPLETE (41/41 tests passing)

**Goal:** Config loading, output paths, result records, reproducible seeding.

### Tests (`tests/test_config.py`, `test_records.py`, `test_seeding.py`)

```
test_load_config_valid        — loads a minimal JSON, returns dict
test_load_config_missing_key  — raises ValidationError for required fields
test_load_config_test_mode    — --test overrides clamp n_workers≤8, max_sims=500
test_output_dir_creates_tree  — OutputDir(base, name) creates subdirs
test_output_dir_idempotent    — calling twice doesn't error
test_records_csv_roundtrip    — write rows, read back, check dtypes
test_records_append           — append mode preserves existing rows
test_make_seeds               — len == n_replicates, deterministic given base seed
test_seed_everything          — numpy + random seeds set, results reproducible
```

### Implementation

- `async_abc/io/schema.py` — JSON schema dict
- `async_abc/io/config.py` — `load_config(path, test_mode=False) -> dict`
- `async_abc/io/paths.py` — `OutputDir` dataclass with `root/plots/data/logs`
- `async_abc/io/records.py` — `ParticleRecord` dataclass + `RecordWriter` (CSV)
- `async_abc/utils/seeding.py` — `make_seeds(n, base)`, `seed_everything(seed)`

---

## Phase 2 — Benchmark Models ✅ COMPLETE (30/30 tests passing, 71 total)

**Goal:** Each benchmark exposes `simulate(params: dict, seed: int) -> float`.

### Tests (`tests/test_benchmarks.py`)

```
test_gaussian_mean_known_params       — near-zero loss for true params
test_gaussian_mean_analytic_posterior — posterior mean ≈ analytic
test_gandk_output_range               — positive float output
test_gandk_deterministic_seed         — same seed → same result
test_lv_positive_populations          — populations never go negative
test_lv_extinction_handled            — large loss on extinction
test_lv_deterministic_seed            — same seed → same trajectory
test_cellular_potts_stub              — ImportError with helpful message
```

### Implementation

- `gaussian_mean.py` — N(θ, σ²), summary = |sample_mean - θ_obs|
- `gandk.py` — g-and-k quantile distribution, 8 summary stats
- `lotka_volterra.py` — stochastic LV, Gillespie/ODE, extinction handling
- `cellular_potts.py` — stub raising ImportError

---

## Phase 3 — Inference Methods & Registry ✅ COMPLETE (13 passed, 1 skipped; 84 total)

**Goal:** Uniform `run_method(name, simulate_fn, limits, config, output_dir, seed) -> List[ParticleRecord]`.

### Tests (`tests/test_inference.py`)

```
test_registry_contains_async_abc       — key present
test_propulate_abc_runs_test_mode      — completes < 60s on gaussian_mean
test_propulate_abc_records_schema      — all fields present in records
test_propulate_abc_tolerance_decreases — monotonically non-increasing
test_pyabc_wrapper_runs                — skip if not installed
test_run_method_unknown_raises         — KeyError with message
```

### Implementation

- `propulate_abc.py` — wraps `ABCPMC` + `Propulator`, collects `ParticleRecord`s
- `pyabc_wrapper.py` — optional pyabc wrapper (try/except import)
- `method_registry.py` — `METHOD_REGISTRY` dict + `run_method()` dispatcher

---

## Phase 4 — Experiment Runner Scripts ✅ COMPLETE (20/20 tests passing; 104 total)

**Goal:** CLI scripts, each complete end-to-end for one experiment.

### Tests (`tests/test_runners.py`)

```
test_*_runner_test_mode     — subprocess with --test completes, creates outputs
test_runner_creates_csv     — raw_results.csv exists
test_runner_creates_metadata — metadata.json with provenance
test_runner_idempotent      — re-run doesn't crash
```

### Implementation (7 scripts in `scripts/`)

Common pattern: argparse → load_config → OutputDir → loop seeds → run_method → plot → metadata

---

## Phase 5 — Plotting ✅ COMPLETE (14/14 tests passing; 118 total)

**Goal:** Reproducible figures: `.pdf` + `.png` + `_data.csv` + `_meta.json`.

### Tests (`tests/test_plotting.py`)

```
test_export_creates_pdf_and_png    — both formats saved
test_export_creates_data_csv       — CSV alongside figure
test_export_creates_meta_json      — JSON with git_hash, timestamp
test_posterior_plot_saves_files    — output files exist
test_scaling_plot_efficiency_curve — efficiency = throughput(n) / (n * throughput(1))
test_archive_evolution_plot        — tolerance over simulation count
test_wasserstein_metric            — returns float
```

### Implementation

- `export.py` — `save_figure()`, `get_git_hash()`
- `common.py` — `posterior_plot()`, `scaling_plot()`, `archive_evolution_plot()`,
  `sensitivity_heatmap()`, `compute_wasserstein()`

---

## Phase 6 — Configs + run_all ✅ COMPLETE (27/27 tests passing; 139 total)

**Goal:** JSON configs for all 7 experiments; single-command orchestration.

### Tests

```
test_all_configs_valid      — all JSON configs pass schema validation
test_run_all_test_mode      — run_all_paper_experiments.py --test completes
test_run_all_creates_outputs — all expected output directories created
```

### Implementation

- 7 JSON config files in `configs/`
- `run_all_paper_experiments.py` with `--test`, `--experiments`, `--output-dir`

---

## Verification

```bash
cd experiments
/home/juhe/bwSyncShare/Code/propulate/.venv/bin/pytest tests/ -v

/home/juhe/bwSyncShare/Code/propulate/.venv/bin/python scripts/gaussian_mean_runner.py \
  --config configs/gaussian_mean.json --output-dir /tmp/test_results --test

/home/juhe/bwSyncShare/Code/propulate/.venv/bin/python run_all_paper_experiments.py \
  --test --output-dir /tmp/paper_results
```

---

## Dependency Order

**Phase 1 → Phase 2 → Phase 3 → Phase 4 + Phase 5 (parallel) → Phase 6**

TDD cycle per phase: Write failing tests → implement to pass → refactor.
