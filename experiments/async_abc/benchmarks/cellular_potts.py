"""Cellular Potts model benchmark backed by nastjapy's simulation machinery.

Requires nastjapy to run simulations. The active environment is preferred; the
repo-local ``nastjapy_copy/.venv`` is used only as a fallback.
"""
from __future__ import annotations

import ctypes
import json
import logging
import re
import shutil
import sys
import uuid
import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_NASTJAPY_VENV = Path(__file__).resolve().parents[3] / "nastjapy_copy" / ".venv"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_CSV_RE = re.compile(r"^output_cells-\d{5}\.csv$")
# x86/x86-64 fenv.h constants. These values are architecture-specific (ARM
# uses different bit positions, e.g. FE_ALL_EXCEPT = 0x9F800000). This code
# is only expected to run on x86 HPC nodes; the values are intentionally
# hardcoded here rather than resolved at runtime because ctypes does not
# expose symbolic fenv constants.
_FE_ALL_EXCEPT = 0x3D   # FE_INVALID|FE_DENORMAL|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INEXACT (x86)
_FE_PYABC_MASK = 0x01 | 0x04 | 0x08  # FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW (x86)
# Physical limits are the authoritative source in parameter_space JSON ("physical_range").
# This module-level dict is used by the standalone normalize/denormalize helpers which
# operate without a config instance (e.g. in generate_cpm_reference.py).  Keep it in
# sync with the JSON values.
_CPM_PHYSICAL_LIMITS: Dict[str, Tuple[float, float]] = {
    "division_rate": (0.00006, 0.6),
    "motility": (0.0, 10000.0),
}

try:
    _LIBC = ctypes.CDLL(None)
except OSError:
    _LIBC = None

# Sentinel for getattr calls that need to distinguish "attribute absent" from
# "attribute present but falsy" (e.g. conn == 0 before a connection is opened).
_SENTINEL = object()


def _restore_default_fp_state() -> None:
    """Clear pending FP exceptions and disable traps enabled by native CPM code.

    The NAStJA/nastjapy stack can leave floating-point traps enabled after a
    simulation. SciPy/OpenBLAS intentionally executes IEEE edge-case checks
    during pyABC's covariance updates, which then crash with SIGFPE unless the
    default FP mask is restored before returning to Python code.
    """
    if _LIBC is None:
        return

    try:
        if hasattr(_LIBC, "feclearexcept"):
            _LIBC.feclearexcept(_FE_ALL_EXCEPT)
        if hasattr(_LIBC, "fedisableexcept"):
            _LIBC.fedisableexcept(_FE_PYABC_MASK)
    except Exception:
        logger.debug("Failed to restore default floating-point state", exc_info=True)


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


def _rewrite_generated_config_paths(config_path: str | Path) -> Path:
    """Rewrite generated include paths to absolute paths before launching NAStJA.

    The nastjapy templates can emit repo-root-relative include paths such as
    ``experiments/data/.../configs/filling.json``. NAStJA resolves those
    relative to the generated config directory, which duplicates the prefix and
    breaks the run. Converting include paths to absolute paths avoids that.

    This function modifies ``config_path`` in-place and returns it.  That is
    intentional and safe: ``config_path`` always points to a per-evaluation
    temporary directory (named with a UUID), so there is no concurrent access.
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


def _is_datahandler_compatible_reference_dir(path: Path) -> bool:
    """Return whether ``path`` looks like a directory DataHandler can load directly."""
    if not path.is_dir():
        return False

    for child in path.iterdir():
        if not child.is_file():
            continue
        if _OUTPUT_CSV_RE.match(child.name):
            return True
        if child.suffix.lower() == ".h5":
            return True
        if child.name == "data_files.zip":
            return True
    return False


def _is_supported_reference_path(path: Path) -> bool:
    """Return whether ``path`` can serve as CPM distance-metric reference data."""
    return _is_reference_data_dir(path) or _is_datahandler_compatible_reference_dir(path)


def _discover_reference_data_dirs(search_root: Path, target_name: str) -> list[Path]:
    """Find valid CPM reference directories below ``search_root``."""
    if not search_root.is_dir():
        return []

    candidates: list[Path] = []
    for candidate in search_root.rglob(target_name):
        if _is_supported_reference_path(candidate):
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
    if _is_supported_reference_path(configured_path):
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
        "CPM reference_data_path does not point to a supported reference dataset. "
        f"Configured path: {configured_path}. "
        f"Searched under: {searched_roots}. "
        "Expected either a generated CPM reference directory "
        "(config.json, cis.out, configs/, 000000/cellevents.log) or a "
        "DataHandler-compatible directory containing files such as "
        "output_cells-00000.csv, *.h5, or data_files.zip. Update "
        "'reference_data_path' accordingly."
    )


def _collect_reference_paths(configured_path: Path) -> list[str]:
    """Resolve one or more CPM reference directories from a configured path.

    Two layouts are supported:

    * **Single reference**: ``configured_path`` itself is a valid reference
      directory → returns ``[configured_path]``.
    * **Multi-reference container**: ``configured_path`` is a directory whose
      immediate children are valid reference directories (e.g. those produced
      by ``generate_cpm_reference.py --n-seeds N``) → returns all children
      sorted alphabetically.

    The second layout lets you point the config at a container directory and
    have all seed replicates picked up automatically.
    """
    resolved = _resolve_reference_data_path(configured_path)
    sub_refs = sorted(
        child for child in resolved.iterdir()
        if _is_supported_reference_path(child)
    )
    if sub_refs:
        return [str(p) for p in sub_refs]
    return [str(resolved)]


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


def _remove_eval_path(path_like: str | Path) -> None:
    """Remove a generated CPM evaluation path without archiving it."""
    path = Path(path_like)
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.exists():
        shutil.rmtree(path)


def normalize_cpm_param(
    name: str,
    value: float,
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """Map a CPM parameter from physical simulator units to [0, 1].

    Parameters
    ----------
    limits:
        Override physical limits dict. Defaults to the module-level
        ``_CPM_PHYSICAL_LIMITS`` constant, which must match the
        ``"physical_range"`` values in ``parameter_space_division_motility.json``.
    """
    lo, hi = (limits or _CPM_PHYSICAL_LIMITS)[name]
    if hi <= lo:
        raise ValueError(f"Invalid CPM physical range for {name!r}: {(lo, hi)}")
    return (float(value) - lo) / (hi - lo)


def denormalize_cpm_param(
    name: str,
    value: float,
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """Map a CPM parameter from [0, 1] to physical simulator units.

    Parameters
    ----------
    limits:
        Override physical limits dict. Defaults to the module-level
        ``_CPM_PHYSICAL_LIMITS`` constant, which must match the
        ``"physical_range"`` values in ``parameter_space_division_motility.json``.
    """
    lo, hi = (limits or _CPM_PHYSICAL_LIMITS)[name]
    return lo + float(value) * (hi - lo)


def normalize_cpm_params(
    params: Dict[str, float],
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Return CPM params normalized into the public [0, 1] parameter space."""
    return {name: normalize_cpm_param(name, float(value), limits) for name, value in params.items()}


def denormalize_cpm_params(
    params: Dict[str, float],
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Return CPM params converted from public [0, 1] values to physical units."""
    return {name: denormalize_cpm_param(name, float(value), limits) for name, value in params.items()}


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

    # The benchmark is safe for pyABC workers once the floating-point state is
    # restored after each native simulation call.
    PYABC_PARALLEL_SAFE = True

    # Keep the older flag for callers/tests that still check it directly.
    MULTIPROCESSING_SAFE = True

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
        self._physical_limits: Dict[str, Tuple[float, float]] = {}
        for name, entry in self._parameter_space_data.items():
            if "physical_range" not in entry:
                raise KeyError(
                    f"Parameter '{name}' in parameter_space JSON is missing 'physical_range'. "
                    "Add \"physical_range\": [lo, hi] to each parameter entry."
                )
            lo, hi = entry["physical_range"]
            self._physical_limits[name] = (float(lo), float(hi))
        self.limits: Dict[str, Tuple[float, float]] = {
            name: (0.0, 1.0) for name in self._parameter_space_data
        }
        self._seed_param_name: str = config.get("seed_param_name", "random_seed")
        self._seed_param_path: str = config.get("seed_param_path", "Settings.randomseed")
        self._output_dir: str = str(_resolve_repo_path(config["output_dir"]))
        self._keep_eval_dirs: bool = bool(config.get("keep_eval_dirs", False))
        self._eval_counter: int = 0  # for logging only; dir names use uuid4
        self._nan_counter: int = 0   # for NaN rate monitoring

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
            if isinstance(dm_raw.get("feature_space_model"), str):
                dm_raw["feature_space_model"] = str(
                    _resolve_repo_path(dm_raw["feature_space_model"])
                )
            ref_paths = _collect_reference_paths(
                _resolve_repo_path(config["reference_data_path"])
            )
            dm_raw["reference_data"] = ref_paths if len(ref_paths) > 1 else ref_paths[0]
            logger.info("CPM using %d reference simulation(s)", len(ref_paths))
            dm_params = DistanceMetricParams.model_validate(dm_raw)
            self._distance_metric = DistanceMetric(params=dm_params)

    def _cleanup_eval_dir(self, sim_dir: str) -> None:
        """Archive simulation output and optionally remove the directory."""
        path = Path(sim_dir)
        if not path.exists():
            return
        try:
            self._sim_manager.cleanup_simdir(sim_dir)
        except Exception as exc:
            logger.warning("Archive/cleanup failed for %s: %s", sim_dir, exc)
        if not self._keep_eval_dirs:
            try:
                _remove_eval_path(sim_dir)
            except Exception as exc:
                logger.warning("Removal failed for %s: %s", sim_dir, exc)

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

        physical_params = {
            name: denormalize_cpm_param(name, value, self._physical_limits)
            for name, value in params.items()
        }
        param_entries = [
            Parameter(
                name=name,
                value=physical_params[name],
                path=self._parameter_space_data[name]["path"],
            )
            for name in params
        ]
        param_entries.append(
            Parameter(
                name=self._seed_param_name,
                value=seed,
                path=self._seed_param_path,
            )
        )
        param_list = ParameterList(parameters=param_entries)

        # --- run simulation ---
        sim_dir: str | None = None
        try:
            config_path = self._sim_manager.build_simulation_config(
                param_list, out_dir_name=sim_dir_name
            )
            _rewrite_generated_config_paths(config_path)
            sim_dir = str(Path(config_path).parent)
            self._sim_manager.run_simulation(config_path)
        except Exception as exc:
            logger.error(
                "CPM simulation failed for params=%s seed=%d: %s", params, seed, exc
            )
            if sim_dir is None:
                sim_dir = str(Path(self._output_dir) / sim_dir_name)
            self._cleanup_eval_dir(sim_dir)
            _restore_default_fp_state()
            self._nan_counter += 1
            self._warn_if_high_nan_rate()
            return float("nan")

        # --- compute distance ---
        score = float("nan")
        try:
            distance_result = self._distance_metric.calculate_distance(sim_dir)
            score = float(distance_result)
        except Exception as exc:
            logger.error(
                "Distance computation failed for sim_dir=%s: %s", sim_dir, exc
            )
        finally:
            self._cleanup_eval_dir(sim_dir)
            _restore_default_fp_state()

        if score != score:  # isnan without importing math
            self._nan_counter += 1
            self._warn_if_high_nan_rate()
        return score

    def _warn_if_high_nan_rate(self) -> None:
        """Emit a warning when the NaN rate exceeds 15% after ≥ 20 evaluations."""
        if self._eval_counter < 20:
            return
        nan_rate = self._nan_counter / self._eval_counter
        if nan_rate > 0.15:
            logger.warning(
                "CPM NaN rate is high: %d/%d evaluations failed (%.0f%%). "
                "Check simulation stability or parameter ranges.",
                self._nan_counter,
                self._eval_counter,
                100 * nan_rate,
            )

    def close(self) -> None:
        """Best-effort teardown for CPM helper objects between experiment runs."""
        distance_metric = getattr(self, "_distance_metric", None)
        reference_data = getattr(distance_metric, "reference_data", []) if distance_metric else []
        for datahandler in reference_data:
            # nastjapy's DataHandler keeps an internal sqlite connection in the
            # private ``_SimDir__con`` attribute.  There is no public close() API;
            # this is a known workaround.  File an upstream nastjapy issue if the
            # attribute disappears and this warning fires.
            conn = getattr(datahandler, "_SimDir__con", _SENTINEL)
            if conn is _SENTINEL:
                logger.warning(
                    "Cannot close CPM reference-data connection: nastjapy DataHandler "
                    "no longer exposes '_SimDir__con'. Resource leak possible. "
                    "Request a public close() API from nastjapy."
                )
            elif conn not in (None, 0):
                try:
                    conn.close()
                except Exception:
                    logger.debug("Failed to close CPM reference-data connection", exc_info=True)

        self._distance_metric = None
        self._sim_manager = None
        gc.collect()

        if not self._keep_eval_dirs:
            output_dir = getattr(self, "_output_dir", None)
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)
