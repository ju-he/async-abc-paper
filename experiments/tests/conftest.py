"""Shared pytest fixtures for the experiments test suite."""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

# Make async_abc importable from the experiments/ directory
EXPERIMENTS_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = EXPERIMENTS_DIR / "scripts"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
PYTHON = sys.executable

sys.path.insert(0, str(EXPERIMENTS_DIR))

from async_abc.io.records import ParticleRecord
from async_abc.inference.method_registry import METHOD_REGISTRY

MINIMAL_CONFIG = {
    "experiment_name": "test_experiment",
    "benchmark": {
        "name": "gaussian_mean",
        "observed_data_seed": 42,
        "n_obs": 50,
    },
    "methods": ["async_propulate_abc"],
    "inference": {
        "max_simulations": 5000,
        "n_workers": 4,
        "k": 20,
        "tol_init": 10.0,
        "scheduler_type": "acceptance_rate",
        "perturbation_scale": 0.8,
    },
    "execution": {
        "n_replicates": 3,
        "base_seed": 0,
    },
    "plots": {
        "posterior": True,
    },
}


def load_base_config(config_name: str) -> dict:
    return json.loads((CONFIGS_DIR / config_name).read_text())


def write_config(root: Path, name: str, cfg: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(cfg))
    return path


def import_runner_module(script_name: str):
    script_path = SCRIPTS_DIR / script_name
    module_name = f"tests_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_runner_main(
    script_name: str,
    config_path: Path,
    output_dir: Path,
    *,
    test_mode: bool = False,
    small_mode: bool = False,
    extra_args: tuple[str, ...] = (),
) -> Path:
    module = import_runner_module(script_name)
    argv = [
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if test_mode:
        argv.append("--test")
    if small_mode:
        argv.append("--small")
    argv.extend(extra_args)
    module.main(argv)
    return output_dir


def run_runner_subprocess(
    script_name: str,
    config_path: Path,
    output_dir: Path,
    *,
    test_mode: bool = False,
    small_mode: bool = False,
    extra_args: tuple[str, ...] = (),
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    cmd = [
        PYTHON,
        str(SCRIPTS_DIR / script_name),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if test_mode:
        cmd.append("--test")
    if small_mode:
        cmd.append("--small")
    cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def copy_output_tree(source_root: Path, dest_root: Path) -> Path:
    shutil.copytree(source_root, dest_root, dirs_exist_ok=True)
    return dest_root


@contextmanager
def patched_method_registry(extra_methods: dict[str, object]):
    original = {name: METHOD_REGISTRY.get(name) for name in extra_methods}
    try:
        METHOD_REGISTRY.update(extra_methods)
        yield
    finally:
        for name, value in original.items():
            if value is None:
                METHOD_REGISTRY.pop(name, None)
            else:
                METHOD_REGISTRY[name] = value


def make_fast_runner_config(
    config_name: str,
    *,
    methods: list[str] | None = None,
    inference_overrides: dict | None = None,
    execution_overrides: dict | None = None,
    plots: dict | None = None,
    top_level_updates: dict | None = None,
    replace_top_level: dict | None = None,
) -> dict:
    cfg = load_base_config(config_name)
    if methods is not None:
        cfg["methods"] = methods
    if inference_overrides:
        cfg["inference"].update(inference_overrides)
    if execution_overrides:
        cfg["execution"].update(execution_overrides)
    if plots is not None:
        cfg["plots"] = plots
    if top_level_updates:
        for key, value in top_level_updates.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    if replace_top_level:
        for key, value in replace_top_level.items():
            cfg[key] = value
    return cfg


def timed_fake_method(simulate_fn, limits, inference_cfg, output_dir, replicate, seed):
    base = float(replicate) * 0.5
    param_name = next(iter(limits))
    return [
        ParticleRecord(
            method="timed_fake",
            replicate=replicate,
            seed=seed,
            step=1,
            params={param_name: 0.1},
            loss=0.4,
            weight=0.5,
            tolerance=1.0,
            wall_time=base + 0.4,
            worker_id="0",
            sim_start_time=base,
            sim_end_time=base + 0.2,
            generation=0,
        ),
        ParticleRecord(
            method="timed_fake",
            replicate=replicate,
            seed=seed,
            step=2,
            params={param_name: -0.1},
            loss=0.2,
            weight=0.5,
            tolerance=0.5,
            wall_time=base + 0.4,
            worker_id="1",
            sim_start_time=base + 0.2,
            sim_end_time=base + 0.4,
            generation=1,
        ),
    ]


def clone_artifact_config(artifact: dict, dest_root: Path) -> Path:
    return write_config(dest_root, artifact["config_path"].name, copy.deepcopy(artifact["config"]))


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


@pytest.fixture
def sbc_config_file(tmp_path):
    cfg = {
        "experiment_name": "sbc",
        "benchmark": {
            "name": "gaussian_mean",
            "n_obs": 40,
            "sigma_obs": 1.0,
            "prior_low": -5.0,
            "prior_high": 5.0,
        },
        "methods": ["rejection_abc"],
        "inference": {
            "max_simulations": 100,
            "n_workers": 1,
            "k": 15,
            "tol_init": 5.0,
            "n_generations": 2,
            "scheduler_type": "acceptance_rate",
            "perturbation_scale": 0.8,
        },
        "execution": {
            "n_replicates": 1,
            "base_seed": 0,
        },
        "sbc": {
            "n_trials": 3,
            "coverage_levels": [0.5, 0.9],
        },
        "plots": {
            "rank_histogram": True,
            "coverage_table": True,
        },
    }
    path = tmp_path / "sbc.json"
    path.write_text(json.dumps(cfg))
    return path


@pytest.fixture
def straggler_config_file(tmp_path):
    cfg = {
        "experiment_name": "straggler",
        "benchmark": {
            "name": "gaussian_mean",
            "observed_data_seed": 42,
            "n_obs": 40,
            "true_mu": 0.0,
            "sigma_obs": 1.0,
            "prior_low": -5.0,
            "prior_high": 5.0,
        },
        "methods": ["rejection_abc"],
        "inference": {
            "max_simulations": 100,
            "n_workers": 1,
            "k": 15,
            "tol_init": 5.0,
            "n_generations": 2,
            "scheduler_type": "acceptance_rate",
            "perturbation_scale": 0.8,
        },
        "execution": {
            "n_replicates": 1,
            "base_seed": 0,
        },
        "straggler": {
            "straggler_rank": 0,
            "base_sleep_s": 0.1,
            "slowdown_factor": [1, 5],
        },
        "plots": {
            "throughput_vs_slowdown": True,
            "gantt": True,
        },
    }
    path = tmp_path / "straggler.json"
    path.write_text(json.dumps(cfg))
    return path


@pytest.fixture(scope="session")
def gaussian_runner_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("gaussian_runner_artifact")
    cfg = make_fast_runner_config(
        "gaussian_mean.json",
        methods=["rejection_abc"],
        inference_overrides={"max_simulations": 100, "k": 25},
        execution_overrides={"n_replicates": 1, "base_seed": 1},
        plots={
            "posterior": True,
            "archive_evolution": False,
            "corner": True,
            "tolerance_trajectory": True,
            "quality_vs_time": True,
        },
    )
    config_path = write_config(root, "gaussian_mean_fast.json", cfg)
    run_runner_main("gaussian_mean_runner.py", config_path, root, test_mode=False)
    return {"root": root, "config": cfg, "config_path": config_path}


@pytest.fixture(scope="session")
def gandk_runner_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("gandk_runner_artifact")
    cfg = make_fast_runner_config(
        "gandk.json",
        methods=["rejection_abc"],
        inference_overrides={"max_simulations": 60, "k": 12},
        execution_overrides={"n_replicates": 1, "base_seed": 1},
        plots={},
    )
    config_path = write_config(root, "gandk_fast.json", cfg)
    run_runner_main("gandk_runner.py", config_path, root)
    return {"root": root, "config": cfg, "config_path": config_path}


@pytest.fixture(scope="session")
def lotka_runner_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("lotka_runner_artifact")
    cfg = make_fast_runner_config(
        "lotka_volterra.json",
        methods=["rejection_abc"],
        inference_overrides={"max_simulations": 60, "k": 12, "tol_init": 1_000_000_000.0},
        execution_overrides={"n_replicates": 1, "base_seed": 1},
        plots={},
    )
    config_path = write_config(root, "lotka_fast.json", cfg)
    run_runner_main("lotka_volterra_runner.py", config_path, root)
    return {"root": root, "config": cfg, "config_path": config_path}


@pytest.fixture(scope="session")
def scaling_runner_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("scaling_runner_artifact")
    cfg = make_fast_runner_config(
        "scaling.json",
        methods=["rejection_abc"],
        inference_overrides={"max_simulations": 80, "k": 15, "tol_init": 1_000_000_000.0},
        execution_overrides={"n_replicates": 1, "base_seed": 1},
        plots={"scaling_curve": False, "efficiency": False},
        top_level_updates={
            "scaling": {
                "worker_counts": [1, 4],
                "test_worker_counts": [1, 4],
                "k_values": [10, 50],
                "test_k_values": [10, 50],
                "wall_time_budgets_s": [0.05, 0.1],
                "wall_time_limit_s": 0.1,
                "max_simulations_policy": {"min_total": 80, "per_worker": 10, "k_factor": 1},
            }
        },
    )
    config_path = write_config(root, "scaling_fast.json", cfg)
    run_runner_main("scaling_runner.py", config_path, root)
    return {"root": root, "config": cfg, "config_path": config_path}


@pytest.fixture(scope="session")
def sensitivity_runner_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("sensitivity_runner_artifact")
    cfg = make_fast_runner_config(
        "sensitivity.json",
        methods=["rejection_abc"],
        inference_overrides={"max_simulations": 80, "k": 10},
        execution_overrides={"n_replicates": 1, "base_seed": 1},
        plots={"sensitivity_heatmap": False},
        replace_top_level={
            "sensitivity_grid": {
                "k": [10],
                "perturbation_scale": [0.8],
                "scheduler_type": ["acceptance_rate"],
                "tol_init_multiplier": [1.0],
            }
        },
    )
    config_path = write_config(root, "sensitivity_fast.json", cfg)
    run_runner_main("sensitivity_runner.py", config_path, root)
    return {"root": root, "config": cfg, "config_path": config_path}


@pytest.fixture(scope="session")
def ablation_runner_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("ablation_runner_artifact")
    cfg = make_fast_runner_config(
        "ablation.json",
        methods=["rejection_abc"],
        inference_overrides={"max_simulations": 80, "k": 10},
        execution_overrides={"n_replicates": 1, "base_seed": 1},
        plots={"ablation_comparison": False},
        top_level_updates={
            "ablation_variants": [
                {
                    "name": "full_model",
                    "k": 10,
                    "scheduler_type": "acceptance_rate",
                    "perturbation_scale": 0.8,
                },
                {
                    "name": "small_archive",
                    "k": 5,
                    "scheduler_type": "acceptance_rate",
                    "perturbation_scale": 0.8,
                },
            ]
        },
    )
    config_path = write_config(root, "ablation_fast.json", cfg)
    run_runner_main("ablation_runner.py", config_path, root)
    return {"root": root, "config": cfg, "config_path": config_path}


@pytest.fixture(scope="session")
def runtime_heterogeneity_runner_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("runtime_heterogeneity_artifact")
    cfg = make_fast_runner_config(
        "runtime_heterogeneity.json",
        methods=["timed_fake"],
        inference_overrides={"max_simulations": 20, "k": 5},
        execution_overrides={"n_replicates": 1, "base_seed": 1},
        plots={
            "idle_fraction": False,
            "throughput_over_time": False,
            "idle_fraction_comparison": False,
            "gantt": True,
            "quality_by_sigma": True,
        },
        top_level_updates={
            "heterogeneity": {"distribution": "lognormal", "mu": 0.0, "sigma_levels": [0.0, 0.5]},
            "benchmark": {"true_mu": 0.0},
        },
    )
    config_path = write_config(root, "runtime_heterogeneity_fast.json", cfg)
    with patched_method_registry({"timed_fake": timed_fake_method}):
        run_runner_main("runtime_heterogeneity_runner.py", config_path, root)
    return {"root": root, "config": cfg, "config_path": config_path}


@pytest.fixture(scope="session")
def straggler_runner_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("straggler_runner_artifact")
    cfg = {
        "experiment_name": "straggler",
        "benchmark": {
            "name": "gaussian_mean",
            "observed_data_seed": 42,
            "n_obs": 40,
            "true_mu": 0.0,
            "sigma_obs": 1.0,
            "prior_low": -5.0,
            "prior_high": 5.0,
        },
        "methods": ["timed_fake"],
        "inference": {
            "max_simulations": 20,
            "n_workers": 1,
            "k": 5,
            "tol_init": 5.0,
            "n_generations": 2,
            "scheduler_type": "acceptance_rate",
            "perturbation_scale": 0.8,
        },
        "execution": {
            "n_replicates": 1,
            "base_seed": 1,
        },
        "straggler": {
            "straggler_rank": 0,
            "base_sleep_s": 0.1,
            "slowdown_factor": [1, 5],
        },
        "plots": {
            "throughput_vs_slowdown": True,
            "gantt": True,
        },
    }
    config_path = write_config(root, "straggler_fast.json", cfg)
    with patched_method_registry({"timed_fake": timed_fake_method}):
        run_runner_main("straggler_runner.py", config_path, root)
    return {"root": root, "config": cfg, "config_path": config_path}


@pytest.fixture
def sample_records():
    return [
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=1,
            params={"mu": 2.0},
            loss=2.0,
            weight=1.0,
            tolerance=5.0,
            wall_time=0.2,
            record_kind="simulation_attempt",
            time_semantics="event_end",
            attempt_count=1,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=10,
            params={"mu": 1.0},
            loss=1.0,
            weight=1.0,
            tolerance=2.5,
            wall_time=1.0,
            record_kind="simulation_attempt",
            time_semantics="event_end",
            attempt_count=10,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=25,
            params={"mu": 0.5},
            loss=0.5,
            weight=0.5,
            tolerance=1.0,
            wall_time=2.0,
            record_kind="simulation_attempt",
            time_semantics="event_end",
            attempt_count=25,
        ),
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=50,
            params={"mu": 0.1},
            loss=0.1,
            weight=0.5,
            tolerance=0.5,
            wall_time=3.0,
            record_kind="simulation_attempt",
            time_semantics="event_end",
            attempt_count=50,
        ),
    ]


@pytest.fixture
def abc_smc_records():
    return [
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=1,
            params={"mu": 0.8},
            loss=0.8,
            weight=0.5,
            tolerance=1.0,
            wall_time=1.2,
            worker_id="0",
            sim_start_time=0.0,
            sim_end_time=1.0,
            generation=0,
            record_kind="population_particle",
            time_semantics="generation_end",
            attempt_count=2,
        ),
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=2,
            params={"mu": 0.6},
            loss=0.6,
            weight=0.5,
            tolerance=1.0,
            wall_time=1.2,
            worker_id="1",
            sim_start_time=0.0,
            sim_end_time=1.2,
            generation=0,
            record_kind="population_particle",
            time_semantics="generation_end",
            attempt_count=2,
        ),
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=3,
            params={"mu": 0.3},
            loss=0.3,
            weight=0.5,
            tolerance=0.5,
            wall_time=2.6,
            worker_id="0",
            sim_start_time=1.2,
            sim_end_time=2.2,
            generation=1,
            record_kind="population_particle",
            time_semantics="generation_end",
            attempt_count=4,
        ),
        ParticleRecord(
            method="abc_smc_baseline",
            replicate=0,
            seed=2,
            step=4,
            params={"mu": 0.1},
            loss=0.1,
            weight=0.5,
            tolerance=0.5,
            wall_time=2.6,
            worker_id="1",
            sim_start_time=1.2,
            sim_end_time=2.6,
            generation=1,
            record_kind="population_particle",
            time_semantics="generation_end",
            attempt_count=4,
        ),
    ]
