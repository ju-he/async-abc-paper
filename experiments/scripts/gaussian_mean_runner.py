#!/usr/bin/env python3
"""Runner for the Gaussian mean benchmark experiment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import load_config
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.benchmark_runner import run_benchmark_runner
from async_abc.utils.metadata import write_metadata
from async_abc.utils.mpi import is_root_rank
from async_abc.utils.shard_finalizers import finalize_experiment_by_name
from async_abc.utils.runner import (
    compute_corrected_estimate,
    run_experiment,
    write_timing_comparison_csv,
    write_timing_csv,
)


def main(argv: list[str] | None = None) -> None:
    run_benchmark_runner(
        argv,
        description="Gaussian mean benchmark experiment.",
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
    )


if __name__ == "__main__":
    main()
