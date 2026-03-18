#!/usr/bin/env python3
"""Runner for the Cellular Potts model benchmark experiment.

Prerequisites
-------------
Before running this script, generate the reference simulation data once::

    python experiments/scripts/generate_cpm_reference.py \\
        --config-template experiments/assets/cellular_potts/sim_config.json \\
        --config-builder-params experiments/assets/cellular_potts/config_builder_params.json \\
        --parameter-space experiments/assets/cellular_potts/parameter_space_division_motility.json \\
        --true-params '{"division_rate": 0.03, "motility": 2000}' \\
        --output-dir experiments/data/cpm_reference \\
        --seed 0

Then update ``reference_data_path`` in ``experiments/configs/cellular_potts.json``
to match the generated directory before running this script.
"""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.metadata import write_metadata
from async_abc.utils.mpi import is_root_rank
from async_abc.utils.runner import (
    compute_corrected_estimate,
    format_duration,
    make_arg_parser,
    run_experiment,
    write_timing_csv,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = make_arg_parser("Cellular Potts model benchmark experiment.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, test_mode=args.test)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    t0 = time.time()
    records = run_experiment(cfg, output_dir)
    elapsed = time.time() - t0

    name = cfg["experiment_name"]
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", name, format_duration(elapsed))
    if args.test and is_root_rank():
        estimated = compute_corrected_estimate(
            elapsed, output_dir.data / "raw_results.csv", args.config
        )
        logger.info("[%s] Estimated full run: ~%s", name, format_duration(estimated))
    if is_root_rank():
        write_timing_csv(output_dir.data / "timing.csv", name, elapsed, estimated, args.test)

    if is_root_rank() and any(cfg.get("plots", {}).values()):
        from async_abc.plotting.reporters import plot_benchmark_diagnostics

        plot_benchmark_diagnostics(records, cfg, output_dir)
    if is_root_rank():
        write_metadata(output_dir, cfg)


if __name__ == "__main__":
    main()
