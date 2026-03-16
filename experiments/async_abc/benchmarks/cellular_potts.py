"""Cellular Potts model benchmark backed by nastjapy's simulation machinery.

Requires nastjapy (via the nastjapy_copy symlink) to run simulations.  Import
succeeds unconditionally; instantiation raises ``ImportError`` when nastjapy is
not importable from the repo-local symlink.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# nastjapy_copy/src is 3 parents above this file (repo root / nastjapy_copy / src)
_NASTJAPY_SRC = Path(__file__).resolve().parents[3] / "nastjapy_copy" / "src"


def _ensure_nastjapy_on_path() -> None:
    """Insert nastjapy_copy/src onto sys.path so nastjapy modules are importable.

    Raises
    ------
    ImportError
        If nastjapy_copy/src does not exist (symlink missing or broken).
    """
    if not _NASTJAPY_SRC.is_dir():
        raise ImportError(
            f"nastjapy_copy/src not found at {_NASTJAPY_SRC}. "
            "Ensure the nastjapy_copy symlink is present at the repo root."
        )
    src_str = str(_NASTJAPY_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


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
        param_space_path = Path(config["parameter_space"])
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
        self._output_dir: str = config["output_dir"]
        self._eval_counter: int = 0

        # SimulationManager
        if _sim_manager is not None:
            self._sim_manager = _sim_manager
        else:
            from simulation.engine_config import EngineBackendParams
            from simulation.manager import SimulationManager
            from simulation.simulation_config_builder import SimulationConfigBuilderParams

            cb_params_path = Path(config["config_builder_params"])
            with open(cb_params_path) as f:
                cb_raw = json.load(f)
            # Override template path for portability (template JSON may have HPC paths)
            cb_raw["config_template"] = config["nastja_config_template"]
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

            dm_params_path = Path(config["distance_metric_params"])
            with open(dm_params_path) as f:
                dm_raw = json.load(f)
            dm_raw["reference_data"] = config["reference_data_path"]
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
        sim_dir_name = f"eval{self._eval_counter:06d}"

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
