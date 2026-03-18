"""Config loading and validation."""
import copy
import json
from pathlib import Path
from typing import Union

from .schema import (
    REQUIRED_BENCHMARK,
    REQUIRED_EXECUTION,
    REQUIRED_INFERENCE,
    REQUIRED_TOP_LEVEL,
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
    return cfg


def load_config(path: Union[str, Path], test_mode: bool = False) -> dict:
    """Load and validate a JSON experiment config.

    Parameters
    ----------
    path:
        Path to the JSON config file.
    test_mode:
        If True, apply test-mode overrides (reduced budgets, local max 8 workers,
        SLURM max 48 workers).

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
    with open(path) as f:
        cfg = json.load(f)

    _validate(cfg)

    if test_mode:
        cfg = _apply_test_mode(cfg)

    return cfg
