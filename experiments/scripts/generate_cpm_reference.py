#!/usr/bin/env python3
"""Generate a CPM reference simulation for use as observed data in the benchmark.

This is a one-off prerequisite script.  Run it once before any inference
experiment to produce the reference simulation directory that
``cellular_potts.json`` points to via ``reference_data_path``.

Example
-------
    python experiments/scripts/generate_cpm_reference.py \\
        --config-template nastjapy_copy/templates/spheroid_inf_nanospheroids/sim_config.json \\
        --config-builder-params nastjapy_copy/templates/spheroid_inf_nanospheroids/config_builder_params.json \\
        --parameter-space nastjapy_copy/templates/spheroid_inf_nanospheroids/parameter_space_division_motility.json \\
        --true-params '{"division_rate": 0.03, "motility": 2000}' \\
        --output-dir experiments/data/cpm_reference \\
        --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from async_abc.benchmarks.cellular_potts import (
    _ensure_nastjapy_on_path,
    _normalize_generated_config_paths,
)

try:
    _ensure_nastjapy_on_path()
except ImportError as exc:
    sys.exit(f"ERROR: {exc}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Generate CPM reference simulation (prerequisite for benchmark)."
    )
    parser.add_argument(
        "--config-template",
        required=True,
        help="Path to NAStJA sim_config.json template.",
    )
    parser.add_argument(
        "--config-builder-params",
        required=True,
        help="Path to config_builder_params.json.",
    )
    parser.add_argument(
        "--parameter-space",
        required=True,
        help="Path to parameter_space JSON file (used for path metadata).",
    )
    parser.add_argument(
        "--true-params",
        required=True,
        help=(
            'Ground-truth parameter values as a JSON object, e.g. '
            '\'{"division_rate": 0.03, "motility": 2000}\'.  '
            "Keys must match entries in --parameter-space."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the reference simulation will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="NAStJA random seed for the reference simulation (default: 0).",
    )
    parser.add_argument(
        "--engine-backend",
        choices=["cis", "srun"],
        default="cis",
        help="Simulation engine backend (default: cis).",
    )
    parser.add_argument(
        "--engine-ntasks",
        type=int,
        default=1,
        help="Number of MPI tasks for srun backend (default: 1).",
    )
    parser.add_argument(
        "--seed-param-path",
        default="Settings.randomseed",
        help="NAStJA config path for the RNG seed field (default: Settings.randomseed).",
    )
    args = parser.parse_args(args)

    from simulation.engine_config import EngineBackendParams
    from simulation.manager import SimulationManager
    from simulation.simulation_config import Parameter, ParameterList
    from simulation.simulation_config_builder import SimulationConfigBuilderParams

    # Load config_builder_params and override template path + output dir
    with open(args.config_builder_params) as f:
        cb_raw = json.load(f)
    cb_raw["config_template"] = args.config_template
    cb_raw["out_dir"] = args.output_dir
    cb_params = SimulationConfigBuilderParams.model_validate(cb_raw)

    # Engine backend
    engine_params = None
    if args.engine_backend == "srun":
        engine_params = EngineBackendParams(backend="srun", ntasks=args.engine_ntasks)

    sim_manager = SimulationManager(cb_params, engine_backend_params=engine_params)

    # Parse true params
    true_params: dict = json.loads(args.true_params)

    # Load parameter space for path metadata
    with open(args.parameter_space) as f:
        ps_data = json.load(f)
    param_space_data = ps_data["parameters"]

    # Validate that all true_params keys exist in the parameter space
    unknown = set(true_params) - set(param_space_data)
    if unknown:
        parser.error(f"--true-params contains unknown parameter(s): {sorted(unknown)}")

    # Build ParameterList (true params + seed)
    param_entries = [
        Parameter(
            name=name,
            value=value,
            path=param_space_data[name]["path"],
        )
        for name, value in true_params.items()
    ]
    param_entries.append(
        Parameter(
            name="random_seed",
            value=args.seed,
            path=args.seed_param_path,
        )
    )
    param_list = ParameterList(parameters=param_entries)

    print(f"Running reference simulation with params: {true_params}, seed={args.seed}")
    config_path = sim_manager.build_simulation_config(param_list, out_dir_name="reference")
    _normalize_generated_config_paths(config_path)
    sim_manager.run_simulation(config_path)

    output_path = Path(config_path).parent
    print(f"Reference simulation written to: {output_path}")
    print(
        f"Set 'reference_data_path' in cellular_potts.json to: {output_path}"
    )
    return str(output_path)


if __name__ == "__main__":
    main()
