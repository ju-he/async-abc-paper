"""Experiment metadata serialisation."""
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..io.config import get_run_mode, is_test_mode
from ..io.paths import OutputDir
from .git import get_git_hash

_VALIDITY_EXPERIMENTS = {
    "gaussian_mean",
    "gandk",
    "lotka_volterra",
    "cellular_potts",
    "sbc",
}
_HPC_PERFORMANCE_EXPERIMENTS = {
    "runtime_heterogeneity",
    "straggler",
    "scaling",
}
_METHOD_ANALYSIS_EXPERIMENTS = {
    "sensitivity",
    "sensitivity_gandk",
    "ablation",
}


def _get_git_hash() -> str:
    return get_git_hash(Path(__file__))


def _installed_packages() -> Dict[str, str]:
    """Return a dict of {package: version} for key scientific packages."""
    import importlib.metadata
    names = ["numpy", "scipy", "matplotlib", "propulate", "mpi4py"]
    out: Dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def infer_experiment_role(cfg: Dict[str, Any]) -> str:
    """Return the paper-facing role for an experiment config.

    Precedence: explicit ``paper.experiment_role`` overrides inferred defaults.
    """
    paper_cfg = cfg.get("paper", {})
    explicit = paper_cfg.get("experiment_role")
    if isinstance(explicit, str) and explicit:
        return explicit

    name = str(cfg.get("experiment_name", ""))
    if name in _VALIDITY_EXPERIMENTS:
        return "validity"
    if name in _HPC_PERFORMANCE_EXPERIMENTS:
        return "hpc_performance"
    if name in _METHOD_ANALYSIS_EXPERIMENTS:
        return "method_analysis"
    return "analysis"


def infer_method_comparison_roles(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Return the configured or inferred comparison role for each method.

    Precedence: explicit ``paper.method_comparison_roles`` overrides inferred defaults.
    """
    paper_cfg = cfg.get("paper", {})
    explicit = paper_cfg.get("method_comparison_roles")
    if isinstance(explicit, dict):
        return {str(key): str(val) for key, val in explicit.items()}

    roles: Dict[str, str] = {}
    for method in cfg.get("methods", []):
        method_name = str(method)
        if method_name == "async_propulate_abc":
            roles[method_name] = "proposed_async_method"
        elif method_name == "abc_smc_baseline":
            roles[method_name] = "main_sync_baseline"
        elif method_name == "pyabc_smc":
            roles[method_name] = "external_framework_reference"
        elif method_name == "rejection_abc":
            roles[method_name] = "small_model_reference"
        else:
            roles[method_name] = "comparison_method"
    return roles


def infer_stop_policy_by_method(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Return the effective stop-policy semantics for each configured method.

    Precedence: explicit ``paper.stop_policy_by_method`` overrides inferred defaults.
    """
    paper_cfg = cfg.get("paper", {})
    explicit = paper_cfg.get("stop_policy_by_method")
    if isinstance(explicit, dict):
        return {str(key): str(val) for key, val in explicit.items()}

    inference_cfg = cfg.get("inference", {})
    max_wall_time_s = inference_cfg.get("max_wall_time_s")
    by_method: Dict[str, str] = {}
    for method in cfg.get("methods", []):
        method_name = str(method)
        if method_name == "pyabc_smc":
            by_method[method_name] = "epsilon_target"
        elif method_name == "abc_smc_baseline":
            by_method[method_name] = "fixed_walltime" if max_wall_time_s not in (None, "") else "fixed_generations"
        elif method_name == "async_propulate_abc":
            by_method[method_name] = "fixed_walltime" if max_wall_time_s not in (None, "") else "fixed_budget"
        else:
            by_method[method_name] = "fixed_budget"
    return by_method


def infer_stop_policy(cfg: Dict[str, Any]) -> str:
    """Return the experiment-level stopping policy.

    Precedence: explicit ``paper.stop_policy`` overrides inferred defaults.
    """
    paper_cfg = cfg.get("paper", {})
    explicit = paper_cfg.get("stop_policy")
    if isinstance(explicit, str) and explicit:
        return explicit

    experiment_role = infer_experiment_role(cfg)
    if experiment_role == "hpc_performance":
        return "fixed_walltime"

    policies = sorted(set(infer_stop_policy_by_method(cfg).values()))
    if len(policies) == 1:
        return policies[0]
    if "fixed_walltime" in policies:
        return "fixed_walltime"
    return policies[0] if policies else "fixed_budget"


def write_metadata(
    output_dir: OutputDir,
    cfg: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write experiment provenance to ``output_dir.data/metadata.json``.

    Parameters
    ----------
    output_dir:
        Experiment output directory (must already exist).
    cfg:
        Full (validated) experiment config dict.
    extra:
        Optional additional key-value pairs to include.

    Returns
    -------
    Path
        Path to the written metadata file.
    """
    meta: Dict[str, Any] = {
        "experiment_name": cfg.get("experiment_name"),
        "timestamp": datetime.now().isoformat(),
        "git_hash": _get_git_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": _installed_packages(),
        "run_mode": get_run_mode(cfg),
        "test_mode": is_test_mode(cfg),
        "experiment_role": infer_experiment_role(cfg),
        "stop_policy": infer_stop_policy(cfg),
        "stop_policy_by_method": infer_stop_policy_by_method(cfg),
        "method_comparison_roles": infer_method_comparison_roles(cfg),
        "wall_time_limit_s": cfg.get("inference", {}).get("max_wall_time_s"),
        "wall_time_budgets_s": cfg.get("scaling", {}).get("wall_time_budgets_s"),
        "config": cfg,
    }
    if extra:
        meta.update(extra)

    path = output_dir.data / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path
