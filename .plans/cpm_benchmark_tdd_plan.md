# TDD Implementation Plan: Realistic Simulator Workload Benchmark

## Status: Phases 1–6 implemented and passing (67/67 unit tests pass)
## Remaining: Phase 7 (integration test with the real external simulator) — requires reference data generation


## Overview

Replace the `RealisticWorkload` stub with a real benchmark class that wraps the simulation backend's
`SimulationManager` + `DistanceMetric`, exposing the standard `simulate(params, seed) -> float`
interface used by all inference methods (propulate, pyabc, rejection, smc-baseline).

**Key resources:**
- Simulation templates: `sim_backend_venv/templates/spheroid_inf_nanospheroids/`
- Venv for tests: `sim_backend_venv/.venv`
- Run tests with: `sim_backend_venv/.venv/bin/pytest experiments/tests/ -x`
- Parameter space: start with 2-param space (`theta_2`, `theta_1`) from
  `sim_backend_venv/templates/spheroid_inf_nanospheroids/parameter_space.json`
- Seed injection path: `Settings.randomseed` in the external simulator config (dotted path notation)

**TDD rhythm per phase: Red → Green → Refactor**

---

## Phase 1: Import Infrastructure

### Goal
Establish that the simulation backend modules can be reliably imported from within the benchmark package.

### 1.1 Red — Failing tests
Add to `experiments/tests/test_benchmarks.py`:

```python
def test_backend_importable():
    """sim_backend_venv/src must be importable via the import helper."""
    from async_abc.benchmarks.realistic_workload import _ensure_backend_on_path
    _ensure_backend_on_path()
    import simulation.manager  # noqa: F401
    import inference.distance  # noqa: F401
```

### 1.2 Green — Implementation
In `experiments/async_abc/benchmarks/realistic_workload.py`, add before class definition:

```python
from pathlib import Path
import sys

def _ensure_backend_on_path() -> None:
    """Insert sim_backend_venv/src onto sys.path so simulation backend modules are importable."""
    src = Path(__file__).parents[3] / "sim_backend_venv" / "src"
    if not src.is_dir():
        raise ImportError(
            f"sim_backend_venv/src not found at {src}. "
            "Ensure the sim_backend_venv symlink is present at the repo root."
        )
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
```

### 1.3 Refactor
Verify path computation is correct relative to the file's actual location:
`realistic_workload.py` is at `experiments/async_abc/benchmarks/realistic_workload.py`,
so `parents[3]` = repo root.

---

## Phase 2: RealisticWorkload.__init__ and .limits

### Goal
Replace the stub `__init__` so that:
- `.limits` is populated from the parameter space config
- `SimulationManager` and `DistanceMetric` are instantiated from config dicts
- Failures give clear error messages

### 2.1 Red — Failing tests
Add to `experiments/tests/test_benchmarks.py`:

```python
import pytest

REALISTIC_TEMPLATE_DIR = Path(__file__).parents[2] / "sim_backend_venv" / "templates" / "spheroid_inf_nanospheroids"

@pytest.fixture
def realistic_config():
    return {
        "name": "realistic_workload",
        "sim_config_template": str(REALISTIC_TEMPLATE_DIR / "sim_config.json"),
        "config_builder_params": str(REALISTIC_TEMPLATE_DIR / "config_builder_params.json"),
        "distance_metric_params": str(REALISTIC_TEMPLATE_DIR / "distance_metric_params.json"),
        "parameter_space": str(REALISTIC_TEMPLATE_DIR / "parameter_space.json"),
        "reference_data_path": "/tmp/realistic_reference",  # placeholder; not accessed in unit tests
        "engine_backend": "cis",   # or "srun"
        "output_dir": "/tmp/realistic_sims",
    }

def test_realistic_limits_populated(realistic_config, monkeypatch):
    """limits dict must match parameter_space ranges."""
    from async_abc.benchmarks.realistic_workload import RealisticWorkload
    # Monkeypatch SimulationManager and DistanceMetric to avoid needing the external simulator installed
    monkeypatch.setattr(
        "async_abc.benchmarks.realistic_workload.SimulationManager", MockSimManager
    )
    monkeypatch.setattr(
        "async_abc.benchmarks.realistic_workload.DistanceMetric", MockDistMetric
    )
    bm = RealisticWorkload(realistic_config)
    assert "theta_2" in bm.limits
    assert "theta_1" in bm.limits
    assert bm.limits["theta_2"] == (0.00006, 0.6)
    assert bm.limits["theta_1"] == (0, 10000)

def test_realistic_init_missing_key_raises(realistic_config):
    """Missing required realistic workload config keys raise ValueError, not ImportError."""
    from async_abc.benchmarks.realistic_workload import RealisticWorkload
    del realistic_config["reference_data_path"]
    with pytest.raises((ValueError, KeyError)):
        RealisticWorkload(realistic_config)
```

### 2.2 Green — Implementation
Replace the stub `RealisticWorkload` class:

```python
class RealisticWorkload:
    REQUIRED_KEYS = [
        "sim_config_template",
        "config_builder_params",
        "distance_metric_params",
        "parameter_space",
        "reference_data_path",
        "output_dir",
    ]

    def __init__(self, config: dict) -> None:
        _ensure_backend_on_path()
        # Import simulation backend modules lazily after path is set
        from simulation.manager import SimulationManager
        from simulation.engine_config import EngineBackendParams
        from inference.distance import DistanceMetric, DistanceMetricParams
        import importlib, os
        _paramspace_module = os.environ.get(
            "SIM_BACKEND_PARAMSPACE_MODULE", "sim_backend.parameter_space_config"
        )
        ParameterSpace = importlib.import_module(_paramspace_module).ParameterSpace

        for key in self.REQUIRED_KEYS:
            if key not in config:
                raise KeyError(f"RealisticWorkload config missing required key: '{key}'")

        # Load parameter space → derive limits
        param_space_path = Path(config["parameter_space"])
        with open(param_space_path) as f:
            ps_data = json.load(f)
        self._param_space_data = ps_data["parameters"]
        self.limits: Dict[str, Tuple[float, float]] = {
            name: tuple(p["range"]) for name, p in self._param_space_data.items()
        }
        self._parameter_space = ParameterSpace.model_validate(ps_data)
        self._seed_param_name = config.get("seed_param_name", "random_seed")
        self._seed_param_path = config.get("seed_param_path", "Settings.randomseed")
        self._output_dir = config["output_dir"]

        # Build SimulationManager
        cb_params_path = Path(config["config_builder_params"])
        with open(cb_params_path) as f:
            cb_raw = json.load(f)
        # Override config_template with the one from this config (allows portability)
        cb_raw["config_template"] = config["sim_config_template"]
        cb_raw["out_dir"] = self._output_dir
        from simulation.simulation_config_builder import SimulationConfigBuilderParams
        cb_params = SimulationConfigBuilderParams.model_validate(cb_raw)

        engine_backend = config.get("engine_backend", "cis")
        engine_params = None
        if engine_backend == "srun":
            ntasks = config.get("engine_ntasks", 1)
            engine_params = EngineBackendParams(backend="srun", ntasks=ntasks)

        self._sim_manager = SimulationManager(cb_params, self._parameter_space, engine_params)

        # Build DistanceMetric
        dm_params_path = Path(config["distance_metric_params"])
        with open(dm_params_path) as f:
            dm_raw = json.load(f)
        # Override reference_data with the benchmark-specific path
        dm_raw["reference_data"] = config["reference_data_path"]
        dm_params = DistanceMetricParams.model_validate(dm_raw)
        self._distance_metric = DistanceMetric(params=dm_params)

        self._eval_counter = 0
```

### 2.3 Refactor
- Ensure `ParameterSpace.model_validate` accepts `{"parameters": {...}}` (verify from the simulation backend source).
- Consider whether to load `config_builder_params.json` overrides via a merge helper.

---

## Phase 3: RealisticWorkload.simulate (unit level)

### Goal
`simulate(params, seed)` must:
1. Inject seed into params as a ParameterList entry
2. Call `SimulationManager.build_simulation_config`
3. Call `SimulationManager.run_simulation`
4. Call `DistanceMetric.calculate_distance` on the sim output dir
5. Call `SimulationManager.cleanup_simdir`
6. Return `float(distance_result)`

### 3.1 Red — Failing tests
Add to `experiments/tests/test_benchmarks.py` (using pytest-mock):

```python
def test_realistic_simulate_calls_pipeline(realistic_config, mocker):
    """simulate() must call build_config → run_simulation → distance → cleanup in order."""
    from async_abc.benchmarks.realistic_workload import RealisticWorkload

    mock_sim = mocker.MagicMock()
    mock_dist = mocker.MagicMock()
    mock_sim.build_simulation_config.return_value = "/tmp/realistic_sims/eval0/config.json"
    mock_dist.calculate_distance.return_value = 3.14  # float-like

    mocker.patch("async_abc.benchmarks.realistic_workload.SimulationManager", return_value=mock_sim)
    mocker.patch("async_abc.benchmarks.realistic_workload.DistanceMetric", return_value=mock_dist)

    bm = RealisticWorkload(realistic_config)
    result = bm.simulate({"theta_2": 0.01, "theta_1": 500.0}, seed=42)

    mock_sim.build_simulation_config.assert_called_once()
    mock_sim.run_simulation.assert_called_once()
    mock_dist.calculate_distance.assert_called_once()
    mock_sim.cleanup_simdir.assert_called_once()
    assert isinstance(result, float)

def test_realistic_simulate_seed_injected(realistic_config, mocker):
    """simulate() must include seed in ParameterList passed to build_simulation_config."""
    from async_abc.benchmarks.realistic_workload import RealisticWorkload
    from simulation.simulation_config import ParameterList
    ...
    # Capture the ParameterList argument and verify seed parameter is present
    captured = []
    mock_sim.build_simulation_config.side_effect = lambda pl, **kw: captured.append(pl) or "/tmp/x/cfg.json"
    bm.simulate({"theta_2": 0.01, "theta_1": 500.0}, seed=99)
    param_names = [p.name for p in captured[0].parameters]
    assert "random_seed" in param_names
    seed_val = next(p.value for p in captured[0].parameters if p.name == "random_seed")
    assert seed_val == 99

def test_realistic_simulate_returns_float_on_nan(realistic_config, mocker):
    """simulate() must return float('nan') if simulation raises, not re-raise."""
    ...
    mock_sim.run_simulation.side_effect = RuntimeError("the external simulator crashed")
    result = bm.simulate({"theta_2": 0.01, "theta_1": 500.0}, seed=0)
    assert result != result  # NaN check
```

### 3.2 Green — Implementation

```python
def simulate(self, params: dict, seed: int) -> float:
    from simulation.simulation_config import Parameter, ParameterList

    self._eval_counter += 1
    sim_dir_name = f"eval{self._eval_counter:06d}"

    # Build ParameterList (inference params + seed)
    param_entries = [
        Parameter(name=name, value=value, path=self._param_space_data[name]["path"])
        for name, value in params.items()
    ]
    param_entries.append(
        Parameter(name=self._seed_param_name, value=seed, path=self._seed_param_path)
    )
    param_list = ParameterList(parameters=param_entries)

    try:
        config_path = self._sim_manager.build_simulation_config(
            param_list, out_dir_name=sim_dir_name
        )
        self._sim_manager.run_simulation(config_path)
    except Exception as exc:
        logger.error("realistic workload simulation failed for params=%s seed=%d: %s", params, seed, exc)
        return float("nan")

    sim_dir = str(Path(config_path).parent)
    try:
        distance_result = self._distance_metric.calculate_distance(sim_dir)
        score = float(distance_result)
    except Exception as exc:
        logger.error("Distance computation failed for sim_dir=%s: %s", sim_dir, exc)
        score = float("nan")
    finally:
        try:
            self._sim_manager.cleanup_simdir(sim_dir)
        except Exception as exc:
            logger.warning("Cleanup failed for %s: %s", sim_dir, exc)

    return score
```

### 3.3 Refactor
- Consider thread-safety of `_eval_counter` if multiprocessing is used.
- Add logging at info level for each evaluation start.

---

## Phase 4: Config Schema Extension

### Goal
`load_config()` must validate realistic-workload-specific required fields when `benchmark.name == "realistic_workload"`.

### 4.1 Red — Failing tests
Add to `experiments/tests/test_config.py`:

```python
def test_realistic_config_missing_reference_data_raises(minimal_config):
    minimal_config["benchmark"] = {
        "name": "realistic_workload",
        "sim_config_template": "/some/template.json",
        # missing: reference_data_path, config_builder_params, etc.
    }
    with pytest.raises(ValidationError):
        load_config_from_dict(minimal_config)

def test_realistic_config_valid_passes(minimal_config, realistic_config_benchmark_section):
    minimal_config["benchmark"] = realistic_config_benchmark_section
    cfg = load_config_from_dict(minimal_config)
    assert cfg["benchmark"]["name"] == "realistic_workload"
```

### 4.2 Green — Implementation
In `experiments/async_abc/io/schema.py`, add realistic-workload-specific validation:

```python
REQUIRED_BENCHMARK_KEYS = [
    "sim_config_template",
    "config_builder_params",
    "distance_metric_params",
    "parameter_space",
    "reference_data_path",
    "output_dir",
]

def _validate_realistic_workload_benchmark(benchmark_cfg: dict) -> None:
    missing = [k for k in REQUIRED_BENCHMARK_KEYS if k not in benchmark_cfg]
    if missing:
        raise ValidationError(
            f"realistic_workload benchmark missing required keys: {missing}"
        )

# In the existing validate_config() or load_config() function:
if cfg["benchmark"]["name"] == "realistic_workload":
    _validate_realistic_workload_benchmark(cfg["benchmark"])
```

### 4.3 Refactor
- Ensure `realistic_workload` is also accepted by `make_benchmark()` in the benchmark factory.

---

## Phase 5: Reference Data Generation Script

### Goal
Provide a reusable one-off script that runs the external simulator with ground-truth parameters and
saves the output as the reference simulation for the distance metric.

### 5.1 Red — Failing tests
Add to `experiments/tests/test_runners.py`:

```python
def test_generate_realistic_reference_importable():
    """generate_realistic_reference.py must be importable without errors."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_realistic_reference",
        SCRIPTS_DIR / "generate_realistic_reference.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
```

### 5.2 Green — Implementation
`experiments/scripts/generate_realistic_reference.py`:

```python
"""Generate realistic workload reference simulation for use as observed data in benchmark."""
import argparse
import json
import sys
from pathlib import Path

def main(args=None):
    parser = argparse.ArgumentParser(description="Generate realistic workload reference simulation")
    parser.add_argument("--config-template", required=True)
    parser.add_argument("--config-builder-params", required=True)
    parser.add_argument("--true-params", required=True, help="JSON string or file of true parameter values")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine-backend", choices=["cis", "srun"], default="cis")
    parser.add_argument("--engine-ntasks", type=int, default=1)
    args = parser.parse_args(args)

    # Ensure the simulation backend is importable
    src = Path(__file__).parents[2] / "sim_backend_venv" / "src"
    sys.path.insert(0, str(src))

    from simulation.manager import SimulationManager
    from simulation.simulation_config_builder import SimulationConfigBuilderParams
    from simulation.simulation_config import Parameter, ParameterList
    from simulation.engine_config import EngineBackendParams

    with open(args.config_builder_params) as f:
        cb_raw = json.load(f)
    cb_raw["config_template"] = args.config_template
    cb_raw["out_dir"] = args.output_dir
    cb_params = SimulationConfigBuilderParams.model_validate(cb_raw)

    engine_params = None
    if args.engine_backend == "srun":
        engine_params = EngineBackendParams(backend="srun", ntasks=args.engine_ntasks)

    sim_manager = SimulationManager(cb_params, engine_backend_params=engine_params)

    # Load true params
    true_params_input = args.true_params
    if Path(true_params_input).is_file():
        with open(true_params_input) as f:
            true_params = json.load(f)
    else:
        true_params = json.loads(true_params_input)

    # Build default parameter space for path lookup
    # (true_params must include path metadata or a parameter_space file must be provided)
    param_entries = [Parameter(name=k, value=v["value"], path=v["path"]) for k, v in true_params.items()]
    param_entries.append(Parameter(name="random_seed", value=args.seed, path="Settings.randomseed"))
    param_list = ParameterList(parameters=param_entries)

    config_path = sim_manager.build_simulation_config(param_list, out_dir_name="reference")
    sim_manager.run_simulation(config_path)
    print(f"Reference simulation written to: {Path(config_path).parent}")

if __name__ == "__main__":
    main()
```

---

## Phase 6: Experiment Config and Runner Script

### Goal
Provide the JSON experiment config and runner script so the realistic workload benchmark runs identically
to other benchmarks.

### 6.1 Experiment config
`experiments/configs/realistic_workload.json`:

```json
{
  "experiment_name": "realistic_workload",
  "benchmark": {
    "name": "realistic_workload",
    "sim_config_template": "sim_backend_venv/templates/spheroid_inf_nanospheroids/sim_config.json",
    "config_builder_params": "sim_backend_venv/templates/spheroid_inf_nanospheroids/config_builder_params.json",
    "distance_metric_params": "sim_backend_venv/templates/spheroid_inf_nanospheroids/distance_metric_params.json",
    "parameter_space": "sim_backend_venv/templates/spheroid_inf_nanospheroids/parameter_space.json",
    "reference_data_path": "experiments/data/realistic_reference",
    "output_dir": "experiments/data/realistic_sims",
    "engine_backend": "cis",
    "seed_param_name": "random_seed",
    "seed_param_path": "Settings.randomseed"
  },
  "methods": ["async_propulate_abc", "pyabc_smc", "rejection_abc", "abc_smc_baseline"],
  "inference": {
    "max_simulations": 2000,
    "n_workers": 48,
    "k": 100,
    "tol_init": 10.0,
    "scheduler_type": "acceptance_rate",
    "perturbation_scale": 0.8
  },
  "execution": {
    "n_replicates": 5,
    "base_seed": 0
  }
}
```

Note: paths relative to repo root. Adjust before running.

### 6.2 Red — Failing tests
Add to `experiments/tests/test_runners.py`:

```python
def test_realistic_workload_runner_importable():
    spec = importlib.util.spec_from_file_location(
        "realistic_workload_runner",
        SCRIPTS_DIR / "realistic_workload_runner.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
```

### 6.3 Green — Implementation
`experiments/scripts/realistic_workload_runner.py`:
Copy `gaussian_mean_runner.py` verbatim, change the config path and experiment name.
No other logic changes needed — the benchmark callable handles all realistic workload specifics.

---

## Phase 7: Integration Test (requires the external simulator available)

### Goal
End-to-end test that a single realistic workload `simulate()` call returns a finite float using the
real simulation backend.

### 7.1 Test (marked `slow` / `integration`)

```python
@pytest.mark.slow
@pytest.mark.integration
def test_realistic_simulate_real_backend(realistic_config_with_real_reference):
    from async_abc.benchmarks.realistic_workload import RealisticWorkload
    bm = RealisticWorkload(realistic_config_with_real_reference)
    result = bm.simulate({"theta_2": 0.01, "theta_1": 500.0}, seed=0)
    assert isinstance(result, float)
    assert result > 0
    assert result == result  # not NaN
```

Run integration tests with:
```bash
sim_backend_venv/.venv/bin/pytest experiments/tests/ -m integration -x -v
```

---

## Execution Order

```
Phase 1  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5  →  Phase 6  →  Phase 7
imports      init         simulate    schema       gen_ref      runner      full e2e
             + limits      (mocked)   extension    script
```

**Prerequisite before Phase 7**: Run `generate_realistic_reference.py` with ground-truth
parameters to create the reference simulation data directory. The `reference_data_path`
in `realistic_workload.json` must point to this directory before any inference run.

## Files Modified / Created

| File | Change |
|------|--------|
| `experiments/async_abc/benchmarks/realistic_workload.py` | Replace stub |
| `experiments/async_abc/io/schema.py` | Add realistic workload validation |
| `experiments/configs/realistic_workload.json` | New |
| `experiments/scripts/realistic_workload_runner.py` | New |
| `experiments/scripts/generate_realistic_reference.py` | New |
| `experiments/tests/test_benchmarks.py` | Add realistic workload test cases |
| `experiments/tests/test_config.py` | Add realistic workload schema tests |
| `experiments/tests/test_runners.py` | Add realistic workload runner tests |
