#!/usr/bin/env python3
"""Generate a CPM reference simulation for use as observed data in the benchmark.

This is an optional utility for replacing the bundled CPM reference asset.
Run it to produce a new reference simulation directory and then point
``cellular_potts.json`` at it via ``reference_data_path``.

Example
-------
    python experiments/scripts/generate_cpm_reference.py \\
        --config-template experiments/assets/cellular_potts/sim_config.json \\
        --config-builder-params experiments/assets/cellular_potts/config_builder_params.json \\
        --parameter-space experiments/assets/cellular_potts/parameter_space_division_motility.json \\
        --true-params '{"division_rate": 0.049905, "motility": 0.2}' \\
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
    denormalize_cpm_params,
    _ensure_nastjapy_on_path,
    _ensure_reference_alias,
    _normalize_generated_config_paths,
    _resolve_repo_path,
)


DEFAULT_OUTPUT_DIR = "experiments/data/cpm_reference_generated"
BUNDLED_ASSET_ROOT = Path("experiments/assets/cellular_potts")


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
            '\'{"division_rate": 0.049905, "motility": 0.2}\'.  '
            "Keys must match entries in --parameter-space."
        ),
    )
    parser.add_argument(
        "--true-params-scale",
        choices=["normalized", "physical"],
        default="normalized",
        help="Interpret --true-params as normalized public parameters or physical simulator units (default: normalized).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where the reference simulation will be written "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
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

    output_dir_path = _resolve_repo_path(args.output_dir)
    bundled_asset_root = _resolve_repo_path(BUNDLED_ASSET_ROOT)
    if output_dir_path == bundled_asset_root or bundled_asset_root in output_dir_path.parents:
        parser.error(
            "--output-dir must not point into experiments/assets/cellular_potts. "
            "That directory contains bundled reference assets tracked by git. "
            f"Use the default generated-data location ({DEFAULT_OUTPUT_DIR}) or another "
            "path under experiments/data/."
        )

    try:
        _ensure_nastjapy_on_path()
    except ImportError as exc:
        sys.exit(f"ERROR: {exc}")

    from simulation.engine_config import EngineBackendParams
    from simulation.manager import SimulationManager
    from simulation.simulation_config import Parameter, ParameterList
    from simulation.simulation_config_builder import SimulationConfigBuilderParams

    config_template_path = _resolve_repo_path(args.config_template)
    config_builder_params_path = _resolve_repo_path(args.config_builder_params)
    parameter_space_path = _resolve_repo_path(args.parameter_space)

    # Load config_builder_params and override template path + output dir
    with open(config_builder_params_path) as f:
        cb_raw = json.load(f)
    cb_raw["config_template"] = str(config_template_path)
    cb_raw["out_dir"] = str(output_dir_path)
    cb_params = SimulationConfigBuilderParams.model_validate(cb_raw)

    # Engine backend
    engine_params = None
    if args.engine_backend == "srun":
        engine_params = EngineBackendParams(backend="srun", ntasks=args.engine_ntasks)

    sim_manager = SimulationManager(cb_params, engine_backend_params=engine_params)

    # Parse true params
    true_params: dict = json.loads(args.true_params)

    # Load parameter space for path metadata
    with open(parameter_space_path) as f:
        ps_data = json.load(f)
    param_space_data = ps_data["parameters"]

    # Validate that all true_params keys exist in the parameter space
    unknown = set(true_params) - set(param_space_data)
    if unknown:
        parser.error(f"--true-params contains unknown parameter(s): {sorted(unknown)}")

    physical_true_params = (
        denormalize_cpm_params(true_params)
        if args.true_params_scale == "normalized"
        else {name: float(value) for name, value in true_params.items()}
    )

    # Build ParameterList (true params + seed)
    param_entries = [
        Parameter(
            name=name,
            value=physical_true_params[name],
            path=param_space_data[name]["path"],
        )
        for name in true_params
    ]
    param_entries.append(
        Parameter(
            name="random_seed",
            value=args.seed,
            path=args.seed_param_path,
        )
    )
    param_list = ParameterList(parameters=param_entries)

    print(
        "Running reference simulation with params "
        f"(public={true_params}, physical={physical_true_params}), seed={args.seed}"
    )
    config_path = sim_manager.build_simulation_config(param_list, out_dir_name="reference")
    _normalize_generated_config_paths(config_path)
    sim_manager.run_simulation(config_path)

    output_path = Path(config_path).parent
    canonical_output_path = _ensure_reference_alias(output_dir_path, output_path)

    print(f"Reference simulation written to: {output_path}")
    if canonical_output_path.resolve() != output_path.resolve():
        print(f"Created canonical reference alias: {canonical_output_path} -> {output_path}")
    print(
        f"Set 'reference_data_path' in cellular_potts.json to: {canonical_output_path}"
    )
    return str(canonical_output_path)


if __name__ == "__main__":
    main()
