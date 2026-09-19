"""Cellular Potts model benchmark backed by nastjapy's simulation machinery.

Requires nastjapy to run simulations. The active environment is preferred; the
repo-local ``.venv`` is used only as a fallback.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import math
import re
import shutil
import sys
import uuid
import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Repo-local fallback venv. Was ``nastjapy_copy/.venv`` (a symlink into
# Code/mirrors/) until that directory disappeared from the sync share on
# 2026-07-30; the venv now lives at the repo root. Only consulted when the
# ACTIVE environment cannot import nastja, so it is inert whenever the run
# already uses .venv, and absent on the cluster checkout (deploys exclude
# dotfiles), where nastjapy comes from the module-loaded venv instead.
_NASTJAPY_VENV = Path(__file__).resolve().parents[3] / ".venv"
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
# Optional per-parameter prior scale, matching "scale" in the parameter_space JSON.
# A parameter whose response is confined to the bottom decade of its range is not
# identifiable under a uniform prior on that range however strong the response is --
# measured on division_rate, whose responsive window is 8% of [0.001, 0.2] linear and
# 50% of the same interval log-uniform. Absent means "linear", i.e. the original
# behaviour.
_CPM_PARAM_SCALES: Dict[str, str] = {}
# NAStJA takes the random seed as a positive 32-bit integer; derived replicate
# seeds are drawn from [1, 2**31 - 1].
_MAX_NASTJA_SEED = 2 ** 31 - 1
_LOG_SCALE = "log"
_LINEAR_SCALE = "linear"
_VALID_SCALES = (_LINEAR_SCALE, _LOG_SCALE)

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
        If neither the active environment nor the repo-local ``.venv`` is usable.
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
                f"'.venv' fallback with site-packages at {site_packages}."
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
    # A container must be recognised BEFORE the single-directory resolver runs.
    # ``_resolve_reference_data_path`` raises on anything that is not itself a
    # reference directory, and a container never is -- so asking it first made the
    # multi-seed layout this function documents (and that
    # ``generate_cpm_reference.py --n-seeds N`` produces) unusable.
    direct = _resolve_repo_path(configured_path)
    if direct.is_dir() and not _is_supported_reference_path(direct):
        children = sorted(
            child for child in direct.iterdir()
            if _is_supported_reference_path(child)
        )
        if children:
            return [str(child) for child in children]

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


def _resolve_scale(name: str, scales: Optional[Dict[str, str]]) -> str:
    """Prior scale for ``name``, defaulting to linear, rejecting anything else loudly."""
    scale = (scales if scales is not None else _CPM_PARAM_SCALES).get(name, _LINEAR_SCALE)
    if scale not in _VALID_SCALES:
        raise ValueError(
            f"CPM parameter {name!r} has scale {scale!r}; expected one of {_VALID_SCALES}"
        )
    return scale


def _check_log_range(name: str, lo: float, hi: float) -> None:
    if lo <= 0.0:
        raise ValueError(
            f"CPM parameter {name!r} is log-scaled but its physical_range starts at {lo}; "
            "a log-uniform prior needs a strictly positive lower bound"
        )


def normalize_cpm_param(
    name: str,
    value: float,
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
    scales: Optional[Dict[str, str]] = None,
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
    if _resolve_scale(name, scales) == _LOG_SCALE:
        _check_log_range(name, lo, hi)
        return (math.log(float(value)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (float(value) - lo) / (hi - lo)


def denormalize_cpm_param(
    name: str,
    value: float,
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
    scales: Optional[Dict[str, str]] = None,
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
    if _resolve_scale(name, scales) == _LOG_SCALE:
        _check_log_range(name, lo, hi)
        return math.exp(math.log(lo) + float(value) * (math.log(hi) - math.log(lo)))
    return lo + float(value) * (hi - lo)


def normalize_cpm_params(
    params: Dict[str, float],
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
    scales: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Return CPM params normalized into the public [0, 1] parameter space."""
    return {name: normalize_cpm_param(name, float(value), limits, scales)
            for name, value in params.items()}


def denormalize_cpm_params(
    params: Dict[str, float],
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
    scales: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Return CPM params converted from public [0, 1] values to physical units."""
    return {name: denormalize_cpm_param(name, float(value), limits, scales)
            for name, value in params.items()}


def replicate_seeds(seed: int, n_replicates: int) -> list[int]:
    """Return ``n_replicates`` distinct NAStJA seeds derived from one evaluation seed.

    A CPM evaluation at k replicates is k independent Monte Carlo realisations of
    the *same* theta, whose feature vectors are averaged before the discrepancy is
    taken (see ``CellularPotts.simulate``). The seeds must therefore be distinct,
    and they must be a deterministic function of the evaluation's own seed so a
    run is reproducible from its recorded seeds alone.

    The first seed is the evaluation seed itself, so k=1 reproduces the
    single-replicate behaviour exactly; the rest are SHA-256 derived, which
    spreads them over the seed range instead of handing NAStJA k consecutive
    integers.
    """
    if int(n_replicates) < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    seeds = [int(seed)]
    counter = 1
    # Distinctness is by construction (the digest varies with the counter); the
    # loop only skips the vanishingly rare collision with an earlier draw.
    while len(seeds) < int(n_replicates):
        digest = hashlib.sha256(f"{int(seed)}:{counter}".encode()).digest()
        candidate = int.from_bytes(digest[:8], "big") % _MAX_NASTJA_SEED + 1
        if candidate not in seeds:
            seeds.append(candidate)
        counter += 1
        if counter > int(n_replicates) + 1000:
            raise RuntimeError(
                f"Could not derive {n_replicates} distinct replicate seeds from seed {seed}"
            )
    return seeds


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
        # Parameters applied to every simulation but not inferred. A benchmark is not
        # defined by its inferred parameters alone: the shipped template sets
        # motilityamount[9] = 50, so "motility is not inferred" and "motility is held at
        # 1400" are different experiments and only the second is reproducible. Kept in
        # the parameter-space file so every simulator path lives in one place.
        self._fixed_params: Dict[str, Any] = {}
        for name, entry in (ps_data.get("fixed") or {}).items():
            if name in ps_data["parameters"]:
                raise ValueError(
                    f"Parameter '{name}' is both inferred and fixed in the parameter_space "
                    "JSON; it must be one or the other"
                )
            if "path" not in entry or "value" not in entry:
                raise KeyError(
                    f"Fixed parameter '{name}' needs both 'path' and 'value' in the "
                    "parameter_space JSON"
                )
            self._fixed_params[name] = (entry["path"], entry["value"])
        if self._fixed_params:
            logger.info("CPM holding %d parameter(s) fixed: %s",
                        len(self._fixed_params),
                        {k: v[1] for k, v in self._fixed_params.items()})
        self._parameter_space = ParameterSpace.model_validate(ps_data)
        self._physical_limits: Dict[str, Tuple[float, float]] = {}
        self._physical_scales: Dict[str, str] = {}
        for name, entry in self._parameter_space_data.items():
            if "physical_range" not in entry:
                raise KeyError(
                    f"Parameter '{name}' in parameter_space JSON is missing 'physical_range'. "
                    "Add \"physical_range\": [lo, hi] to each parameter entry."
                )
            lo, hi = entry["physical_range"]
            self._physical_limits[name] = (float(lo), float(hi))
            scale = str(entry.get("scale", _LINEAR_SCALE))
            if scale not in _VALID_SCALES:
                raise ValueError(
                    f"Parameter '{name}' in parameter_space JSON has scale {scale!r}; "
                    f"expected one of {_VALID_SCALES}"
                )
            if scale == _LOG_SCALE:
                _check_log_range(name, float(lo), float(hi))
            self._physical_scales[name] = scale
        self.limits: Dict[str, Tuple[float, float]] = {
            name: (0.0, 1.0) for name in self._parameter_space_data
        }
        self._seed_param_name: str = config.get("seed_param_name", "random_seed")
        self._seed_param_path: str = config.get("seed_param_path", "Settings.randomseed")
        self._output_dir: str = str(_resolve_repo_path(config["output_dir"]))
        self._keep_eval_dirs: bool = bool(config.get("keep_eval_dirs", False))
        self._eval_counter: int = 0  # for logging only; dir names use uuid4
        self._nan_counter: int = 0   # for NaN rate monitoring
        # One evaluation = k simulations at the same theta, feature-averaged. A single
        # 50^3 CPM realisation is too noisy for the discrepancy to order nearby thetas;
        # averaging k of them divides the within-theta feature variance by k, which is
        # what every forecast in .plans/cpm_setup_proposal_2026-09-19.md assumes.
        self._n_replicates: int = int(config.get("n_replicates_per_evaluation", 1))
        if self._n_replicates < 1:
            raise ValueError(
                "n_replicates_per_evaluation must be >= 1, got "
                f"{config['n_replicates_per_evaluation']!r}"
            )
        # With several reference directories the observed data can be read two ways:
        # as one observation whose features are the replicate average (True), or as
        # several independent observations whose distances are averaged (False, the
        # nastjapy default). They are different targets. The averaged observation is
        # the one the screening forecasts were computed under -- it halves the
        # reference's own noise, symmetric with the candidate averaging above.
        self._average_reference: bool = bool(config.get("average_reference_replicates", False))

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

        self._reference_handlers = list(getattr(self._distance_metric, "reference_data", []))
        if self._average_reference:
            self._collapse_reference_to_one_average()
        logger.info("CPM running %d simulation(s) per evaluation; reference read as %s",
                    self._n_replicates,
                    "one replicate-averaged observation" if self._average_reference
                    else "independent observations with averaged distances")

    def _collapse_reference_to_one_average(self) -> None:
        """Replace the reference replicates by their single feature-averaged observation.

        ``DistanceMetric`` averages the distance to each reference; this averages the
        references first and takes one distance, which is the operation the offline
        screening performed (``diag_cpm_screening._units``) and so the one the
        forecast posteriors were measured under.
        """
        metric = self._distance_metric
        references = list(getattr(metric, "reference_data", []))
        if len(references) < 2:
            return
        if getattr(metric, "feature_space_model", None) is None:
            raise ValueError(
                "average_reference_replicates requires a feature_space_model: the "
                "averaging is over raw feature arrays, which the legacy lambda_dict "
                "path does not expose."
            )
        metric.reference_data = [metric._average_feature_item(references)]
        logger.info("CPM averaged %d reference replicates into one observation",
                    len(references))

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

    def _build_param_list(self, params: dict, seed: int):
        """Physical parameters + fixed parameters + the NAStJA seed for one simulation."""
        from simulation.simulation_config import Parameter, ParameterList

        physical_params = {
            name: denormalize_cpm_param(name, value, self._physical_limits,
                                        self._physical_scales)
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
        param_entries += [
            Parameter(name=name, value=value, path=path)
            for name, (path, value) in self._fixed_params.items()
        ]
        param_entries.append(
            Parameter(
                name=self._seed_param_name,
                value=seed,
                path=self._seed_param_path,
            )
        )
        return ParameterList(parameters=param_entries)

    def _run_one_simulation(self, params: dict, seed: int) -> str:
        """Run one NAStJA simulation and return its output directory.

        Raises on failure, having first removed whatever directory it created --
        the caller cleans up the replicates that already succeeded.
        """
        sim_dir_name = f"eval_{uuid.uuid4().hex[:12]}"
        sim_dir: str | None = None
        try:
            config_path = self._sim_manager.build_simulation_config(
                self._build_param_list(params, seed), out_dir_name=sim_dir_name
            )
            _rewrite_generated_config_paths(config_path)
            sim_dir = str(Path(config_path).parent)
            self._sim_manager.run_simulation(config_path)
        except Exception:
            self._cleanup_eval_dir(
                sim_dir if sim_dir is not None
                else str(Path(self._output_dir) / sim_dir_name)
            )
            raise
        return sim_dir

    def _distance_over_replicates(self, sim_dirs: list) -> float:
        """Discrepancy of one evaluation from its replicate simulation directories."""
        if len(sim_dirs) == 1:
            return float(self._distance_metric.calculate_distance(sim_dirs[0]))
        return float(self._distance_metric.calculate_distance_replicates(sim_dirs))

    def simulate(self, params: dict, seed: int) -> float:
        """Run the CPM simulations of one evaluation and return the distance to reference.

        The evaluation runs ``n_replicates_per_evaluation`` simulations at the same
        parameters with distinct seeds derived from ``seed`` (see
        ``replicate_seeds``). Their raw feature arrays are averaged before the
        discrepancy is taken -- averaging features, not distances, which is what
        divides the within-theta noise by k.

        Returns ``float('inf')`` on simulation or scoring failure rather than
        raising. A failed simulation is, in ABC terms, an infinitely-bad
        discrepancy: ``inf`` is excluded from every archive (``loss < tol`` is
        False) exactly as a rejected sample should be, yet the ABCPMC
        ``_check_loss`` guard accepts it -- whereas ``nan`` is rejected by that
        guard (crash-loudly), so a single failed NAStJA run would otherwise
        abort a whole scaling combo. ``inf`` is behaviourally identical to the
        old ``nan`` for archive selection but keeps the run alive.

        A replicate that fails fails the whole evaluation: an evaluation averaged
        over fewer replicates than the rest carries more noise than the tolerance
        schedule was set for, so it is not the same evaluation.

        Parameters
        ----------
        params:
            Dict of parameter name → value (must match keys in ``limits``).
        seed:
            RNG seed for the evaluation; the replicate seeds are derived from it.

        Returns
        -------
        float
            ABC distance (lower is better). ``inf`` on failure.
        """
        self._eval_counter += 1
        seeds = replicate_seeds(seed, self._n_replicates)
        logger.debug("CPM eval #%d starting (%d replicate(s), seeds=%s)",
                     self._eval_counter, len(seeds), seeds)

        # --- run simulations ---
        sim_dirs: list = []
        try:
            for replicate_seed in seeds:
                sim_dirs.append(self._run_one_simulation(params, replicate_seed))
        except Exception as exc:
            logger.error(
                "CPM simulation failed for params=%s seed=%d (replicate seeds=%s): %s",
                params, seed, seeds, exc
            )
            for done_dir in sim_dirs:
                self._cleanup_eval_dir(done_dir)
            _restore_default_fp_state()
            self._nan_counter += 1
            self._warn_if_high_nan_rate()
            return float("inf")

        # --- compute distance ---
        score = float("inf")
        try:
            score = self._distance_over_replicates(sim_dirs)
        except Exception as exc:
            logger.error(
                "Distance computation failed for sim_dirs=%s: %s", sim_dirs, exc
            )
        finally:
            for sim_dir in sim_dirs:
                self._cleanup_eval_dir(sim_dir)
            _restore_default_fp_state()

        # A NaN distance (e.g. a degenerate feature vector) is treated as a
        # failed score: map it to inf so the ABCPMC crash-loudly guard does not
        # abort the run. The counter still tracks it as a failed evaluation.
        if score != score:  # isnan without importing math
            self._nan_counter += 1
            self._warn_if_high_nan_rate()
            score = float("inf")
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
        # The real DataHandlers, kept separately because averaging the reference
        # replaces ``reference_data`` with a plain feature container.
        reference_data = getattr(self, "_reference_handlers", [])
        for datahandler in reference_data:
            # The sqlite connection belongs to nastjapy's SimDir, which the
            # DataHandler holds as ``sim_dir``; it is private there
            # (``_SimDir__con``) and there is no public close() API, so this is a
            # known workaround. A CSV-backed reference never opens one (``__con``
            # stays 0), which is the usual case for CPM. File an upstream nastjapy
            # issue if the attribute disappears and this warning fires.
            owner = getattr(datahandler, "sim_dir", None) or datahandler
            conn = getattr(owner, "_SimDir__con", _SENTINEL)
            if conn is _SENTINEL:
                logger.warning(
                    "Cannot close CPM reference-data connection: nastjapy SimDir "
                    "no longer exposes '_SimDir__con'. Resource leak possible. "
                    "Request a public close() API from nastjapy."
                )
            elif conn not in (None, 0):
                try:
                    conn.close()
                except Exception:
                    logger.debug("Failed to close CPM reference-data connection", exc_info=True)

        self._reference_handlers = []
        self._distance_metric = None
        self._sim_manager = None
        gc.collect()

        if not self._keep_eval_dirs:
            output_dir = getattr(self, "_output_dir", None)
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)
