"""Config schema definition.

REQUIRED_TOP_LEVEL lists the top-level keys that every config must contain.
REQUIRED_BENCHMARK, REQUIRED_INFERENCE, REQUIRED_EXECUTION list required sub-keys.
"""

REQUIRED_TOP_LEVEL = ["experiment_name", "benchmark", "methods", "inference", "execution"]

REQUIRED_BENCHMARK = ["name"]

REQUIRED_INFERENCE = [
    "max_simulations",
    "n_workers",
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


# Test-mode override values.
# Keys under "clamp" are min(current, value); keys under "set" are forced to value.
TEST_MODE_OVERRIDES = {
    "clamp": {
        "inference": {
            "n_workers": 8,
            "max_simulations": 500,
            "n_generations": 3,
        },
        "execution": {
            "n_replicates": 2,
        },
    },
    "set": {
        "execution": {
            "base_seed": 1,
        },
    },
}
