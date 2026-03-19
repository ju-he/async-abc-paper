"""Experiment metadata serialisation."""
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..io.config import get_run_mode, is_test_mode
from ..io.paths import OutputDir


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


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
        "config": cfg,
    }
    if extra:
        meta.update(extra)

    path = output_dir.data / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path
