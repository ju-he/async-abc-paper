#!/usr/bin/env python3
"""Runner for the realistic-workload benchmark experiment.

Reference Data
--------------
The default config already points at bundled reference assets in
``experiments/assets/realistic_workload``. Regenerate them only if you want to
replace the default reference data::

    python experiments/scripts/generate_realistic_reference.py \\
        --config-template experiments/assets/realistic_workload/sim_config.json \\
        --config-builder-params experiments/assets/realistic_workload/config_builder_params.json \\
        --parameter-space experiments/assets/realistic_workload/parameter_space.json \\
        --true-params '{"theta_2": 0.049905, "theta_1": 0.2}' \\
        --seed 0

Then update ``reference_data_path`` in ``experiments/configs/realistic_workload.json``
if you want this runner to use the newly generated directory instead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.utils.benchmark_runner import run_benchmark_runner
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.metadata import write_metadata
from async_abc.utils.mpi import is_root_rank
from async_abc.utils.shard_finalizers import finalize_experiment_by_name
from async_abc.utils.runner import (
    compute_corrected_estimate,
    run_experiment,
    write_timing_comparison_csv,
    write_timing_csv,
)


def _prepare_runtime_cfg(cfg: dict, output_dir: OutputDir) -> dict:
    """Return cfg with simulation scratch output redirected into this experiment run."""
    cfg = dict(cfg)
    benchmark_cfg = dict(cfg["benchmark"])
    benchmark_cfg["output_dir"] = str(output_dir.root / "realistic_sims")
    cfg["benchmark"] = benchmark_cfg
    return cfg


def main(argv: list[str] | None = None) -> None:
    run_benchmark_runner(
        argv,
        description="Realistic-workload benchmark experiment.",
        runner_script_path=str(Path(__file__).resolve()),
        configure_logging_fn=configure_logging,
        load_config_fn=load_config,
        run_experiment_fn=run_experiment,
        compute_corrected_estimate_fn=compute_corrected_estimate,
        write_timing_csv_fn=write_timing_csv,
        write_timing_comparison_csv_fn=write_timing_comparison_csv,
        write_metadata_fn=write_metadata,
        finalize_experiment_by_name_fn=finalize_experiment_by_name,
        is_root_rank_fn=is_root_rank,
        prepare_runtime_cfg=_prepare_runtime_cfg,
    )


if __name__ == "__main__":
    main()
