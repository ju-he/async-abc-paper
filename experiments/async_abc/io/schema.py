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
