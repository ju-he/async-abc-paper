"""Config loading and validation."""
import copy
import json
import warnings
from pathlib import Path
from typing import Union

from .schema import (
    REQUIRED_BENCHMARK,
    REQUIRED_EXECUTION,
    REQUIRED_INFERENCE,
    REQUIRED_TOP_LEVEL,
    VALID_BENCHMARK_NAMES,
    VALID_SCHEDULER_TYPES,
    ValidationError,
    _validate_cpm_benchmark,
    get_test_mode_overrides,
)


def _resolve_config_path(path: Union[str, Path]) -> Path:
    """Resolve config paths from either the repo root or experiments/."""
    path = Path(path)
    if path.exists() or path.is_absolute():
        return path

    experiments_dir = Path(__file__).resolve().parents[2]
    candidate = experiments_dir / path
    if candidate.exists():
        return candidate

    return path


def _resolve_small_config_path(path: Path) -> Path:
    """Return the sibling small-tier config path for *path*."""
    if path.parent.name == "small":
        return path

    candidate = path.parent / "small" / path.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Small config not found for {path}. Expected {candidate}."
    )


def _validate(cfg: dict) -> None:
    """Raise ValidationError if cfg is missing required keys."""
    for key in REQUIRED_TOP_LEVEL:
        if key not in cfg:
            raise ValidationError(f"Config missing required top-level key: '{key}'")

    for key in REQUIRED_BENCHMARK:
        if key not in cfg["benchmark"]:
            raise ValidationError(f"Config['benchmark'] missing required key: '{key}'")

    for key in REQUIRED_INFERENCE:
        if key not in cfg["inference"]:
            raise ValidationError(f"Config['inference'] missing required key: '{key}'")

    for key in REQUIRED_EXECUTION:
        if key not in cfg["execution"]:
            raise ValidationError(f"Config['execution'] missing required key: '{key}'")

    if not isinstance(cfg["methods"], list) or len(cfg["methods"]) == 0:
        raise ValidationError("Config['methods'] must be a non-empty list.")

    # Validate scheduler_type against allowed values.
    scheduler_type = cfg["inference"].get("scheduler_type")
    if scheduler_type is not None and scheduler_type not in VALID_SCHEDULER_TYPES:
        raise ValidationError(
            f"Config['inference']['scheduler_type'] = {scheduler_type!r} "
            f"is not valid. Must be one of: {sorted(VALID_SCHEDULER_TYPES)}"
        )

    # Validate benchmark name against known benchmarks.
    benchmark_name = cfg["benchmark"].get("name")
    if benchmark_name is not None and benchmark_name not in VALID_BENCHMARK_NAMES:
        raise ValidationError(
            f"Config['benchmark']['name'] = {benchmark_name!r} "
            f"is not valid. Must be one of: {sorted(VALID_BENCHMARK_NAMES)}"
        )

    # Warn when wall-time is the intended stopping criterion but n_generations
    # is suspiciously low — it may become the binding constraint instead.
    max_wall_time_s = cfg["inference"].get("max_wall_time_s")
    n_generations = cfg["inference"].get("n_generations")
    if max_wall_time_s is not None and n_generations is not None and n_generations < 50:
        warnings.warn(
            f"n_generations={n_generations} is low for a wall-time-limited run "
            f"(max_wall_time_s={max_wall_time_s}). The generation cap may stop "
            f"the run before the wall-time budget is exhausted. Consider raising "
            f"n_generations to 1000 or higher.",
            stacklevel=3,
        )

    if cfg["benchmark"].get("name") == "cellular_potts":
        _validate_cpm_benchmark(cfg["benchmark"])


def _apply_test_mode(cfg: dict) -> dict:
    """Return a deep-copied config with test-mode overrides applied."""
    cfg = copy.deepcopy(cfg)
    test_mode_overrides = get_test_mode_overrides()
    # Clamp: use min(current, limit)
    for section, overrides in test_mode_overrides.get("clamp", {}).items():
        if section not in cfg:
            continue
        for key, limit in overrides.items():
            current = cfg[section].get(key)
            cfg[section][key] = min(current, limit) if current is not None else limit
    # Set: unconditionally assign
    for section, overrides in test_mode_overrides.get("set", {}).items():
        if section not in cfg:
            continue
        for key, val in overrides.items():
            cfg[section][key] = val
    # CPM runs are substantially heavier than the toy benchmarks. Shrink the
    # test budget further so cluster smoke tests stay cheap even under MPI.
    if cfg.get("benchmark", {}).get("name") == "cellular_potts":
        inference = cfg.setdefault("inference", {})
        inference["max_simulations"] = min(int(inference.get("max_simulations", 12)), 12)
        if inference.get("k") is not None:
            inference["k"] = min(int(inference["k"]), 5)
    return cfg


def compose_run_mode(config_tier: str, test_mode: bool) -> str:
    """Return the canonical run-mode label for a resolved config."""
    if config_tier not in {"full", "small"}:
        raise ValueError(f"Unsupported config tier: {config_tier!r}")
    if config_tier == "small":
        return "small_test" if test_mode else "small"
    return "test" if test_mode else "full"


def _annotate_mode(cfg: dict, *, config_tier: str, test_mode: bool) -> dict:
    """Persist run-tier metadata on the resolved config."""
    cfg = copy.deepcopy(cfg)
    inference = cfg.setdefault("inference", {})
    inference["test_mode"] = bool(test_mode)
    inference.setdefault("progress_log_interval_s", 10.0)
    # When wall-time is the stopping criterion and n_generations was not
    # explicitly configured, default to 1000 so it never binds.
    if inference.get("max_wall_time_s") is not None:
        inference.setdefault("n_generations", 1000)
    plots = cfg.setdefault("plots", {})
    plots.setdefault("emit_paper_summaries", True)
    plots.setdefault("emit_diagnostics", True)
    analysis = cfg.setdefault("analysis", {})
    analysis.setdefault("ci_level", 0.95)
    default_min_particles = inference.get("k")
    if default_min_particles is None:
        default_min_particles = 100
    analysis.setdefault("min_particles_for_threshold", int(default_min_particles))
    cfg.setdefault("execution", {})["config_tier"] = config_tier
    cfg["execution"]["run_mode"] = compose_run_mode(config_tier, bool(test_mode))
    return cfg


def is_test_mode(cfg: dict) -> bool:
    """Return whether a loaded config represents a test-mode run."""
    return bool(cfg.get("inference", {}).get("test_mode", False))


def is_small_mode(cfg: dict) -> bool:
    """Return whether a loaded config represents the small tier."""
    return cfg.get("execution", {}).get("config_tier", "full") == "small"


def get_run_mode(cfg: dict) -> str:
    """Return the canonical run mode for a loaded config."""
    execution = cfg.get("execution", {})
    run_mode = execution.get("run_mode")
    if isinstance(run_mode, str) and run_mode:
        return run_mode
    return compose_run_mode(str(execution.get("config_tier", "full")), is_test_mode(cfg))


def load_config(
    path: Union[str, Path],
    test_mode: bool = False,
    small_mode: bool = False,
) -> dict:
    """Load and validate a JSON experiment config.

    Parameters
    ----------
    path:
        Path to the JSON config file.
    test_mode:
        If True, apply test-mode overrides (reduced budgets, local max 8 workers,
        SLURM max 48 workers).
    small_mode:
        If True, load the sibling config from ``small/<filename>`` before
        applying test-mode overrides.

    Returns
    -------
    dict
        Validated (and optionally test-mode-clamped) config.

    Raises
    ------
    ValidationError
        If required keys are missing.
    json.JSONDecodeError
        If the file is not valid JSON.
    FileNotFoundError
        If the file does not exist.
    """
    path = _resolve_config_path(path)
    if small_mode:
        path = _resolve_small_config_path(path)
    with open(path) as f:
        cfg = json.load(f)

    _validate(cfg)

    if test_mode:
        cfg = _apply_test_mode(cfg)

    config_tier = "small" if path.parent.name == "small" else "full"
    return _annotate_mode(cfg, config_tier=config_tier, test_mode=test_mode)
