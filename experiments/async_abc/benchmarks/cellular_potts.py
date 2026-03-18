"""Cellular Potts model benchmark backed by nastjapy's simulation machinery.

Requires nastjapy to run simulations. The active environment is preferred; the
repo-local ``nastjapy_copy/.venv`` is used only as a fallback.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_NASTJAPY_VENV = Path(__file__).resolve().parents[3] / "nastjapy_copy" / ".venv"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve project-relative CPM asset paths independently of the cwd."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _nastjapy_site_packages() -> Path:
    """Return the matching site-packages dir from the repo-local nastjapy venv."""
    return (
        _NASTJAPY_VENV
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _ensure_nastjapy_on_path() -> None:
    """Resolve nastjapy from the environment or the repo-local fallback.

    Raises
    ------
    ImportError
        If neither the active environment nor ``nastjapy_copy/.venv`` is usable.
    """
    try:
        import nastja.parameter_space_config  # noqa: F401
        return
    except Exception as env_exc:
        site_packages = _nastjapy_site_packages()
        if not site_packages.is_dir():
            raise ImportError(
                "The cellular_potts benchmark requires a working nastjapy/nastja "
                "installation in the active environment, or a repo-local "
                f"'nastjapy_copy/.venv' fallback with site-packages at {site_packages}."
            ) from env_exc

    site_packages_str = str(site_packages)
    if site_packages_str not in sys.path:
        sys.path.insert(0, site_packages_str)
    try:
        import nastja.parameter_space_config  # noqa: F401
    except Exception as path_exc:
        raise ImportError(
            "The cellular_potts benchmark requires a working nastjapy/nastja "
            "installation. Import failed from both the active environment and "
            f"the repo-local .venv site-packages at {site_packages}."
        ) from path_exc


def _normalize_generated_config_paths(config_path: str | Path) -> Path:
    """Rewrite generated include paths to absolute paths before launching NAStJA.

    The nastjapy templates can emit repo-root-relative include paths such as
    ``experiments/data/.../configs/filling.json``. NAStJA resolves those
    relative to the generated config directory, which duplicates the prefix and
    breaks the run. Converting include paths to absolute paths avoids that.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = json.load(f)

    include_key = None
    if "Include" in cfg:
        include_key = "Include"
    elif "include" in cfg:
        include_key = "include"

    if include_key is None:
        return config_path

    includes = cfg[include_key]
    include_items = [includes] if isinstance(includes, str) else includes
    if not isinstance(include_items, list):
        return config_path

    normalized = []
    changed = False
    for include in include_items:
        if not isinstance(include, str):
            normalized.append(include)
            continue
        path = Path(include)
        if path.is_absolute():
            normalized.append(include)
            continue
        config_relative = (config_path.parent / path).resolve()
        cwd_relative = (Path.cwd() / path).resolve()
        repo_relative = (_REPO_ROOT / path).resolve()
        if config_relative.exists():
            resolved = config_relative
        elif cwd_relative.exists():
            resolved = cwd_relative
        else:
            resolved = repo_relative
        resolved_str = str(resolved)
        normalized.append(resolved_str)
        changed = changed or resolved_str != include

    if changed:
        cfg[include_key] = normalized[0] if isinstance(includes, str) else normalized
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    return config_path


def _is_reference_data_dir(path: Path) -> bool:
    """Return whether ``path`` looks like a generated CPM reference directory."""
    if not path.is_dir():
        return False

    expected_files = [path / "config.json", path / "cis.out"]
    expected_dirs = [path / "configs"]
    if not all(file_path.is_file() for file_path in expected_files):
        return False
    if not all(dir_path.is_dir() for dir_path in expected_dirs):
        return False

    return (path / "000000" / "cellevents.log").is_file()


def _discover_reference_data_dirs(search_root: Path, target_name: str) -> list[Path]:
    """Find valid CPM reference directories below ``search_root``."""
    if not search_root.is_dir():
        return []

    candidates: list[Path] = []
    for candidate in search_root.rglob(target_name):
        if _is_reference_data_dir(candidate):
            candidates.append(candidate.resolve())
    return sorted(set(candidates), key=lambda path: (len(path.parts), str(path)))


def _resolve_reference_data_path(path_like: str | Path) -> Path:
    """Resolve the CPM reference directory, including nested generated layouts.

    Some NAStJA-generated reference datasets end up nested one level deeper than
    the intended ``.../reference`` directory, e.g.
    ``<root>/experiments/data/cpm_reference/reference``. Prefer the configured
    path, but fall back to that nested layout when present.
    """
    configured_path = _resolve_repo_path(path_like)
    if _is_reference_data_dir(configured_path):
        return configured_path

    search_roots = [configured_path.parent]
    repo_data_root = (_REPO_ROOT / "experiments" / "data").resolve()
    if repo_data_root not in search_roots:
        search_roots.append(repo_data_root)

    candidates: list[Path] = []
    for search_root in search_roots:
        candidates.extend(
            _discover_reference_data_dirs(
                search_root,
                configured_path.name,
            )
        )
    candidates = sorted(set(candidates), key=lambda path: (len(path.parts), str(path)))

    matching_parent_name = [
        candidate
        for candidate in candidates
        if configured_path.parent.name in candidate.parts
    ]
    if matching_parent_name:
        candidates = matching_parent_name

    if len(candidates) > 1:
        candidate_list = ", ".join(str(candidate) for candidate in candidates[:5])
        raise FileNotFoundError(
            "CPM reference_data_path is ambiguous because multiple valid reference "
            f"datasets were found while resolving {configured_path}: {candidate_list}"
        )

    for candidate in candidates:
        logger.warning(
            "Resolved CPM reference data path %s to discovered generated directory %s",
            configured_path,
            candidate,
        )
        return candidate

    searched_roots = ", ".join(str(root) for root in search_roots)
    raise FileNotFoundError(
        "CPM reference_data_path does not point to a valid reference dataset. "
        f"Configured path: {configured_path}. "
        f"Searched under: {searched_roots}. "
        "Expected files such as config.json, cis.out, configs/, and "
        "000000/cellevents.log. Regenerate the reference data or update "
        "'reference_data_path' to the generated directory."
    )


def _ensure_reference_alias(output_dir: Path, actual_reference_dir: Path) -> Path:
    """Create a stable ``output_dir/reference`` alias for generated reference data."""
    alias_path = output_dir / "reference"
    resolved_actual = actual_reference_dir.resolve()

    if alias_path.exists() or alias_path.is_symlink():
        try:
            if alias_path.resolve() == resolved_actual:
                return alias_path
        except FileNotFoundError:
            pass

        if alias_path.is_symlink() or alias_path.is_file():
            alias_path.unlink()
        elif alias_path.is_dir() and not any(alias_path.iterdir()):
            alias_path.rmdir()
        elif alias_path.is_dir():
            raise FileExistsError(
                f"Cannot create CPM reference alias at {alias_path}: directory is not empty."
            )

    alias_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        alias_path.symlink_to(resolved_actual, target_is_directory=True)
    except OSError:
        shutil.copytree(resolved_actual, alias_path)
    return alias_path


class CellularPotts:
    """Cellular Potts model benchmark using nastjapy's SimulationManager + DistanceMetric.

    Exposes the standard ``simulate(params, seed) -> float`` interface so all
    inference methods (propulate, pyabc, rejection, smc-baseline) can run on the
    CPM simulator without any changes.

    Parameters
    ----------
    config:
        Benchmark sub-config dict. Required keys:

        - ``nastja_config_template``: path to NAStJA sim_config.json template
        - ``config_builder_params``: path to config_builder_params.json
        - ``distance_metric_params``: path to distance_metric_params.json
        - ``parameter_space``: path to parameter_space JSON file
        - ``reference_data_path``: path to reference simulation directory
        - ``output_dir``: base directory for per-evaluation simulation outputs

        Optional keys:

        - ``engine_backend``: ``"cis"`` (default) or ``"srun"``
        - ``engine_ntasks``: number of MPI ranks for srun backend (default 1)
        - ``seed_param_name``: name of the seed parameter (default ``"random_seed"``)
        - ``seed_param_path``: NAStJA config path for the seed field
          (default ``"Settings.randomseed"``)

    _sim_manager:
        Pre-built SimulationManager — injected in tests to bypass real NAStJA.
    _distance_metric:
        Pre-built DistanceMetric — injected in tests to bypass real distance
        computation.
    """

    REQUIRED_KEYS = [
        "nastja_config_template",
        "config_builder_params",
        "distance_metric_params",
        "parameter_space",
        "reference_data_path",
        "output_dir",
    ]

    def __init__(
        self,
        config: dict,
        _sim_manager: Optional[Any] = None,
        _distance_metric: Optional[Any] = None,
    ) -> None:
        _ensure_nastjapy_on_path()

        for key in self.REQUIRED_KEYS:
            if key not in config:
                raise KeyError(f"CellularPotts config missing required key: '{key}'")

        # Load parameter space and derive limits dict
        param_space_path = _resolve_repo_path(config["parameter_space"])
        with open(param_space_path) as f:
            ps_data = json.load(f)

        from nastja.parameter_space_config import ParameterSpace

        self._parameter_space_data: Dict[str, Any] = ps_data["parameters"]
        self._parameter_space = ParameterSpace.model_validate(ps_data)
        self.limits: Dict[str, Tuple[float, float]] = {
            name: (float(p["range"][0]), float(p["range"][1]))
            for name, p in self._parameter_space_data.items()
        }
        self._seed_param_name: str = config.get("seed_param_name", "random_seed")
        self._seed_param_path: str = config.get("seed_param_path", "Settings.randomseed")
        self._output_dir: str = str(_resolve_repo_path(config["output_dir"]))
        self._eval_counter: int = 0  # for logging only; dir names use uuid4

        # SimulationManager
        if _sim_manager is not None:
            self._sim_manager = _sim_manager
        else:
            from simulation.engine_config import EngineBackendParams
            from simulation.manager import SimulationManager
            from simulation.simulation_config_builder import SimulationConfigBuilderParams

            cb_params_path = _resolve_repo_path(config["config_builder_params"])
            with open(cb_params_path) as f:
                cb_raw = json.load(f)
            # Override template path for portability (template JSON may have HPC paths)
            cb_raw["config_template"] = str(_resolve_repo_path(config["nastja_config_template"]))
            cb_raw["out_dir"] = self._output_dir
            cb_params = SimulationConfigBuilderParams.model_validate(cb_raw)

            engine_backend = config.get("engine_backend", "cis")
            engine_params: Optional[EngineBackendParams] = None
            if engine_backend == "srun":
                ntasks = config.get("engine_ntasks", 1)
                engine_params = EngineBackendParams(backend="srun", ntasks=ntasks)

            self._sim_manager = SimulationManager(
                cb_params, self._parameter_space, engine_params
            )

        # DistanceMetric
        if _distance_metric is not None:
            self._distance_metric = _distance_metric
        else:
            from inference.distance import DistanceMetric, DistanceMetricParams

            dm_params_path = _resolve_repo_path(config["distance_metric_params"])
            with open(dm_params_path) as f:
                dm_raw = json.load(f)
            dm_raw["reference_data"] = str(
                _resolve_reference_data_path(config["reference_data_path"])
            )
            dm_params = DistanceMetricParams.model_validate(dm_raw)
            self._distance_metric = DistanceMetric(params=dm_params)

    def simulate(self, params: dict, seed: int) -> float:
        """Run a CPM simulation and return the distance to reference data.

        The NAStJA random seed is injected as an extra parameter alongside the
        inference parameters so every call is reproducible.

        Returns ``float('nan')`` on simulation or scoring failure rather than
        raising, matching the contract expected by all inference methods.

        Parameters
        ----------
        params:
            Dict of parameter name → value (must match keys in ``limits``).
        seed:
            RNG seed injected into the NAStJA config.

        Returns
        -------
        float
            ABC distance (lower is better). ``nan`` on failure.
        """
        from simulation.simulation_config import Parameter, ParameterList

        self._eval_counter += 1
        sim_dir_name = f"eval_{uuid.uuid4().hex[:12]}"
        logger.debug("CPM eval #%d starting (dir=%s)", self._eval_counter, sim_dir_name)

        param_entries = [
            Parameter(
                name=name,
                value=value,
                path=self._parameter_space_data[name]["path"],
            )
            for name, value in params.items()
        ]
        param_entries.append(
            Parameter(
                name=self._seed_param_name,
                value=seed,
                path=self._seed_param_path,
            )
        )
        param_list = ParameterList(parameters=param_entries)

        try:
            config_path = self._sim_manager.build_simulation_config(
                param_list, out_dir_name=sim_dir_name
            )
            _normalize_generated_config_paths(config_path)
            self._sim_manager.run_simulation(config_path)
        except Exception as exc:
            logger.error(
                "CPM simulation failed for params=%s seed=%d: %s", params, seed, exc
            )
            return float("nan")

        sim_dir = str(Path(config_path).parent)
        score = float("nan")
        try:
            distance_result = self._distance_metric.calculate_distance(sim_dir)
            score = float(distance_result)
        except Exception as exc:
            logger.error(
                "Distance computation failed for sim_dir=%s: %s", sim_dir, exc
            )
        finally:
            try:
                self._sim_manager.cleanup_simdir(sim_dir)
            except Exception as exc:
                logger.warning("Cleanup failed for %s: %s", sim_dir, exc)

        return score
