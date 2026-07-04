# Testing Patterns

**Analysis Date:** 2026-04-08

## Test Framework

**Runner:**
- pytest (declared in `[project.optional-dependencies] test` in `pyproject.toml`)
- Config: `[tool.pytest.ini_options]` in `pyproject.toml`
- Test paths: `experiments/tests/`

**Assertion Library:**
- pytest built-in `assert` statements
- `pytest.approx()` for float comparisons
- `np.testing.assert_array_equal()` for numpy array comparisons

**Run Commands:**
```bash
# Use the project venv
/home/juhe/bwSyncShare/Code/mirrors/sim_backend_venv/.venv/bin/python -m pytest    # Run all tests
/home/juhe/bwSyncShare/Code/mirrors/sim_backend_venv/.venv/bin/python -m pytest -x  # Stop on first failure
/home/juhe/bwSyncShare/Code/mirrors/sim_backend_venv/.venv/bin/python -m pytest experiments/tests/test_sbc.py  # Single file
/home/juhe/bwSyncShare/Code/mirrors/sim_backend_venv/.venv/bin/python -m pytest -k "test_compute_rank"  # By name
```

## Test File Organization

**Location:**
- All tests in `experiments/tests/` (separate from source, configured via `testpaths` in pyproject.toml)

**Naming:**
- `test_<module_or_feature>.py`

**Files (by size):**
| File | Lines | Focus |
|------|-------|-------|
| `test_inference.py` | 2227 | Inference methods (propulate, rejection, pyabc, MPI) |
| `test_plotting.py` | 1578 | Plot generation, export, metadata |
| `test_runners.py` | 1345 | Runner script CLI and end-to-end |
| `test_sharding.py` | 1225 | Shard layout, distributed execution |
| `test_benchmarks.py` | 853 | Benchmark models (Gaussian, GandK, LV, realistic workload) |
| `test_sbc.py` | 831 | Simulation-based calibration |
| `test_config.py` | 580 | Config loading, validation, schema |
| `test_analysis.py` | 495 | Analysis functions (ESS, trajectories, barriers) |
| `test_parallel_coordination.py` | 395 | MPI coordination, method execution modes |
| `test_sensitivity_metric.py` | 304 | Sensitivity grid analysis |
| `test_phase6.py` | 303 | End-to-end config validation, run_all orchestration |
| `test_extend.py` | 266 | Extending/resuming experiments |
| `test_records.py` | 189 | ParticleRecord and RecordWriter |
| `test_progress.py` | 87 | Progress reporting |
| `test_seeding.py` | 62 | RNG seeding utilities |
| `test_paths.py` | 39 | OutputDir path management |

**Shared Test Infrastructure:**
- `experiments/tests/conftest.py` (655 lines) - Fixtures, helpers, session-scoped artifacts
- `experiments/tests/mpi_integration_helper.py` - MPI integration test helper (run as subprocess)
- `experiments/tests/mpi_abc_smc_baseline_helper.py` - ABC SMC MPI baseline helper

## Test Structure

**Suite Organization:**
```python
# Class-based grouping for related tests (used in test_records.py, test_config.py, test_seeding.py)
class TestParticleRecord:
    def test_creation(self):
        r = make_record()
        assert r.method == "async_propulate_abc"
        assert r.loss == pytest.approx(1.23)

    def test_params_stored(self):
        r = make_record(params={"mu": 1.0, "sigma": 2.0})
        assert r.params["mu"] == 1.0

# Function-based tests for standalone checks (used in test_sbc.py, test_analysis.py)
def test_compute_rank_true_below_all():
    assert compute_rank(np.array([1.0, 2.0, 3.0]), 0.0) == 0

def test_compute_rank_middle():
    rank = compute_rank(np.array([0.0, 1.0, 2.0]), 1.5)
    assert rank == 2
```

**Both patterns are used.** Use class-based grouping when testing a single component with many facets. Use standalone functions for independent unit tests.

## Fixtures

**Shared Fixtures (from `conftest.py`):**

```python
# Simple config fixtures
@pytest.fixture
def minimal_config():
    return copy.deepcopy(MINIMAL_CONFIG)

@pytest.fixture
def config_file(tmp_path, minimal_config):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(minimal_config))
    return p

@pytest.fixture
def tmp_output_dir(tmp_path):
    return tmp_path / "results"

# Session-scoped artifacts that run actual experiments once
@pytest.fixture(scope="session")
def gaussian_runner_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("gaussian_runner_artifact")
    cfg = make_fast_runner_config("gaussian_mean.json", ...)
    config_path = write_config(root, "gaussian_mean_fast.json", cfg)
    run_runner_main("gaussian_mean_runner.py", config_path, root)
    return {"root": root, "config": cfg, "config_path": config_path}
```

**Session-Scoped Artifacts:**
- `gaussian_runner_artifact` - Runs Gaussian mean experiment once for reuse
- `gandk_runner_artifact` - Runs G-and-k experiment once
- `lotka_runner_artifact` - Runs Lotka-Volterra experiment once
- `scaling_runner_artifact` - Runs scaling experiment once
- `sensitivity_runner_artifact` - Runs sensitivity experiment once
- `ablation_runner_artifact` - Runs ablation experiment once
- `runtime_heterogeneity_runner_artifact` - Runs heterogeneity experiment once
- `straggler_runner_artifact` - Runs straggler experiment once

These artifacts run with minimal budgets (`max_simulations=60-100`, `k=10-25`, `n_replicates=1`) to keep tests fast.

## Test Helpers

**Key helpers in `conftest.py`:**

```python
# Helper to build fast test configs from production configs
def make_fast_runner_config(config_name, *, methods=None, inference_overrides=None, ...):
    cfg = load_base_config(config_name)
    if methods is not None:
        cfg["methods"] = methods
    if inference_overrides:
        cfg["inference"].update(inference_overrides)
    return cfg

# Helper to run a runner script in-process
def run_runner_main(script_name, config_path, output_dir, *, test_mode=False):
    module = import_runner_module(script_name)
    argv = ["--config", str(config_path), "--output-dir", str(output_dir)]
    module.main(argv)

# Helper to run a runner as a subprocess
def run_runner_subprocess(script_name, config_path, output_dir, *, timeout=180):
    cmd = [PYTHON, str(SCRIPTS_DIR / script_name), "--config", str(config_path), ...]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

# Helper to create a local test record
def make_record(**kwargs):  # in test_records.py
    defaults = dict(method="async_propulate_abc", replicate=0, seed=42, ...)
    defaults.update(kwargs)
    return ParticleRecord(**defaults)
```

## Mocking

**Framework:** `unittest.mock` (MagicMock, monkeypatch)

**Patterns:**
```python
# monkeypatch for module-level replacements
monkeypatch.setattr(module, "configure_logging", lambda: None)
monkeypatch.setattr(module, "load_config", lambda path, test_mode=False: cfg)

# MagicMock for complex dependencies (realistic workload benchmark)
mock_sim = MagicMock()
mock_dist = MagicMock()
mock_sim.build_simulation_config.return_value = str(config_path)
mock_dist.calculate_distance.return_value = 2.5

# Fake MPI environment for unit tests (from test_inference.py)
def _install_fake_mpi_executor(monkeypatch, *, executor):
    fake_mpi = types.ModuleType("mpi4py")
    fake_mpi.MPI = types.SimpleNamespace(
        COMM_WORLD=types.SimpleNamespace(Get_rank=lambda: 0, Get_size=lambda: 1),
    )
    monkeypatch.setitem(sys.modules, "mpi4py", fake_mpi)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", fake_mpi.MPI)

# Context manager for method registry patching
with patched_method_registry({"timed_fake": timed_fake_method}):
    run_runner_main("straggler_runner.py", config_path, root)
```

**What to Mock:**
- MPI runtime (`mpi4py.MPI`) for local unit tests
- External simulation managers (realistic workload `SimulationManager`, `DistanceMetric`)
- Logging configuration in runner tests
- Config loading when testing runner logic in isolation

**What NOT to Mock:**
- Core analysis functions (test with real numpy/pandas operations)
- File I/O (use `tmp_path` fixtures instead)
- The benchmark models themselves (test with small budgets)

## Parametrize

```python
# Config validation across all config files
@pytest.mark.parametrize("config_file", CONFIG_FILES)
def test_config_loads(self, config_file):
    load_config(CONFIGS_DIR / config_file)

# Method execution modes
@pytest.mark.parametrize("method_name", ["pyabc_smc", "abc_smc_baseline"])
def test_method_coordination(self, method_name):
    ...
```

## Coverage

**Requirements:** No formal coverage target enforced
**Coverage tool:** Not configured in `pyproject.toml`

## Test Types

**Unit Tests:**
- Direct function calls with known inputs and expected outputs
- Example: `test_compute_rank_true_below_all()` in `experiments/tests/test_sbc.py`
- Example: `TestParticleRecord` class in `experiments/tests/test_records.py`

**Integration Tests (session-scoped artifacts):**
- Run full experiment pipelines with minimal budgets
- Reuse artifacts across multiple test functions via session-scoped fixtures
- Example: `gaussian_runner_artifact` runs the full Gaussian mean experiment, then individual tests verify outputs

**CLI Smoke Tests:**
- Run runner scripts as subprocesses and check return code
- Example: `TestRunnerCliSmoke` in `experiments/tests/test_runners.py`
- Uses `run_runner_subprocess()` with `timeout=180`

**MPI Integration Tests:**
- Separate helper scripts (`mpi_integration_helper.py`, `mpi_abc_smc_baseline_helper.py`)
- Run via `mpirun` subprocess from test functions
- Skipped when MPI is not available

## Common Patterns

**Async/Float Testing:**
```python
# Float comparisons always use pytest.approx
assert r.loss == pytest.approx(1.23)
assert float(rows[0]["loss"]) == pytest.approx(3.14)
assert abs(row_50 - 0.5) < 0.05  # also used for statistical tests

# Statistical tests with tolerance
rng = np.random.default_rng(42)
trials = [{"posterior_samples": rng.uniform(0, 1, 100), "true_value": rng.uniform(0, 1)} for i in range(500)]
df = empirical_coverage(trials, coverage_levels=[0.5, 0.9])
row_50 = df[df["coverage_level"] == 0.5]["empirical_coverage"].iloc[0]
assert abs(row_50 - 0.5) < 0.05  # statistical tolerance
```

**Error Testing:**
```python
# Expect specific exceptions with match patterns
with pytest.raises(ValidationError):
    load_config(p)

with pytest.raises(ValidationError, match="experiment_name"):
    load_config(p)

with pytest.raises(Exception):
    load_config(p)  # invalid JSON
```

**File Output Verification:**
```python
# Check that expected files were created
assert (tmp_path / "gaussian_mean" / "data" / "raw_results.csv").exists()
assert list((tmp_path / "sensitivity" / "data").glob("sensitivity_*.csv"))

# Verify file contents
with open(path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)
assert len(rows) == 5

# Verify JSON metadata
meta = json.loads((output_dir.plots / "rank_histogram_meta.json").read_text())
assert meta["plot_name"] == "rank_histogram"
assert meta["summary_plot"] is True
```

**Matplotlib Testing:**
```python
# Use Agg backend for non-interactive testing
import matplotlib
matplotlib.use("Agg")

# Verify figure creation
fig = _make_fig()
assert isinstance(fig, Figure)
```

**Optional Dependency Skipping:**
```python
def _nastjapy_available() -> bool:
    try:
        _ensure_nastjapy_on_path()
        return True
    except ImportError:
        return False

@pytest.fixture
def cpm_config(tmp_path):
    if not _NASTJAPY_AVAILABLE:
        pytest.skip("sim_backend not available -- run with sim_backend_venv/.venv")
    ...
```

---

*Testing analysis: 2026-04-08*
