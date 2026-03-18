#!/usr/bin/env python3
"""Runner for the Gaussian mean benchmark experiment."""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.metadata import write_metadata
from async_abc.utils.mpi import is_root_rank
from async_abc.utils.runner import (
    compute_scaling_factor,
    format_duration,
    make_arg_parser,
    run_experiment,
    write_timing_csv,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = make_arg_parser("Gaussian mean benchmark experiment.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, test_mode=args.test)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    t0 = time.time()
    records = run_experiment(cfg, output_dir, extend=args.extend)
    elapsed = time.time() - t0

    name = cfg["experiment_name"]
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", name, format_duration(elapsed))
    if args.test and is_root_rank():
        factor, extra, note = compute_scaling_factor(args.config)
        estimated = elapsed * factor + extra
        logger.info(
            "[%s] Estimated full run: ~%s  (%s)",
            name,
            format_duration(estimated),
            note,
        )
    if is_root_rank():
        write_timing_csv(output_dir.data / "timing.csv", name, elapsed, estimated, args.test)

    if is_root_rank() and any(cfg.get("plots", {}).values()):
        from async_abc.plotting.reporters import plot_benchmark_diagnostics

        plot_benchmark_diagnostics(records, cfg, output_dir)
    if is_root_rank():
        write_metadata(output_dir, cfg)


if __name__ == "__main__":
    main()
