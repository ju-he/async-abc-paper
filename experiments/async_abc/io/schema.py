"""Config schema definition.

REQUIRED_TOP_LEVEL lists the top-level keys that every config must contain.
REQUIRED_BENCHMARK, REQUIRED_INFERENCE, REQUIRED_EXECUTION list required sub-keys.
"""
import os
from copy import deepcopy

REQUIRED_TOP_LEVEL = ["experiment_name", "benchmark", "methods", "inference", "execution"]

REQUIRED_BENCHMARK = ["name"]

REQUIRED_INFERENCE = [
    "max_simulations",
    "k",
    "tol_init",
    "scheduler_type",
    "perturbation_scale",
]

REQUIRED_EXECUTION = ["n_replicates", "base_seed"]

VALID_SCHEDULER_TYPES = {"quantile", "geometric_decay", "acceptance_rate"}

VALID_BENCHMARK_NAMES = {"gaussian_mean", "gandk", "lotka_volterra", "cellular_potts"}

# Extra required benchmark keys when name == "cellular_potts"
CPM_REQUIRED_BENCHMARK_KEYS = [
    "nastja_config_template",
    "config_builder_params",
    "distance_metric_params",
    "parameter_space",
    "reference_data_path",
    "output_dir",
]


class ValidationError(ValueError):
    """Raised when a config dict fails validation."""


def _validate_cpm_benchmark(benchmark_cfg: dict) -> None:
    """Raise ValidationError if a cellular_potts benchmark config is missing required keys."""
    missing = [k for k in CPM_REQUIRED_BENCHMARK_KEYS if k not in benchmark_cfg]
    if missing:
        raise ValidationError(
            f"Config['benchmark'] missing required key(s) for cellular_potts: {missing}"
        )


# Test-mode worker limits.
LOCAL_TEST_MAX_WORKERS = 8
CLUSTER_TEST_MAX_WORKERS = 48
TEST_MAX_WORKERS_ENV_VAR = "ASYNC_ABC_TEST_MAX_WORKERS"


def get_test_mode_max_workers() -> int:
    """Return the worker cap for test mode in the current environment."""
    raw = os.getenv(TEST_MAX_WORKERS_ENV_VAR)
    if raw is not None:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{TEST_MAX_WORKERS_ENV_VAR} must be an integer, got {raw!r}"
            ) from exc
        if value < 1:
            raise ValueError(
                f"{TEST_MAX_WORKERS_ENV_VAR} must be >= 1, got {value}"
            )
        return value

    if os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_NTASKS"):
        return CLUSTER_TEST_MAX_WORKERS

    return LOCAL_TEST_MAX_WORKERS


_TEST_MODE_OVERRIDES_TEMPLATE = {
    "clamp": {
        "inference": {
            "n_workers": LOCAL_TEST_MAX_WORKERS,
            "max_simulations": 100,
            "n_generations": 2,
            "max_wall_time_s": 30,
        },
        "execution": {
            "n_replicates": 1,
        },
        "sbc": {
            "n_trials": 2,
        },
    },
    "set": {
        "execution": {
            "base_seed": 1,
        },
    },
}


def get_test_mode_overrides() -> dict:
    """Return test-mode overrides with an environment-appropriate worker cap."""
    overrides = deepcopy(_TEST_MODE_OVERRIDES_TEMPLATE)
    overrides["clamp"]["inference"]["n_workers"] = get_test_mode_max_workers()
    return overrides


# Backward-compatible snapshot for callers that import the constant directly.
TEST_MODE_OVERRIDES = get_test_mode_overrides()
