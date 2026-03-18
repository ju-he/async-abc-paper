#!/usr/bin/env python3
"""Orchestrate all paper experiments.

Usage
-----
Run every experiment in test mode::

    python run_all_paper_experiments.py --test --output-dir /tmp/paper_results

Run a subset::

    python run_all_paper_experiments.py --test \\
        --experiments gaussian_mean gandk \\
        --output-dir /tmp/paper_results

Full production run::

    python run_all_paper_experiments.py --output-dir /path/to/results
"""
import argparse
import importlib.util
import logging
import sys
import time
import traceback
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
SCRIPTS_DIR = EXPERIMENTS_DIR / "scripts"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"

sys.path.insert(0, str(EXPERIMENTS_DIR))
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.mpi import allreduce_max, is_root_rank
from async_abc.utils.runner import compute_scaling_factor, format_duration, write_timing_csv

logger = logging.getLogger(__name__)

# Ordered mapping: experiment_name → (runner_script, config_file)
EXPERIMENT_REGISTRY = {
    "gaussian_mean": ("gaussian_mean_runner.py", "gaussian_mean.json"),
    "gandk": ("gandk_runner.py", "gandk.json"),
    "lotka_volterra": ("lotka_volterra_runner.py", "lotka_volterra.json"),
    "sbc": ("sbc_runner.py", "sbc.json"),
    "straggler": ("straggler_runner.py", "straggler.json"),
    "runtime_heterogeneity": ("runtime_heterogeneity_runner.py", "runtime_heterogeneity.json"),
    "scaling": ("scaling_runner.py", "scaling.json"),
    "sensitivity": ("sensitivity_runner.py", "sensitivity.json"),
    "ablation": ("ablation_runner.py", "ablation.json"),
}


def _run_experiment(
    name: str,
    runner: str,
    config: str,
    output_dir: Path,
    test_mode: bool,
    extend: bool = False,
) -> tuple:
    """Run a single experiment script as a subprocess.

    Returns (returncode, elapsed_seconds).
    """
    runner_path = SCRIPTS_DIR / runner
    argv = [
        "--config", str(CONFIGS_DIR / config),
        "--output-dir", str(output_dir),
    ]
    if test_mode:
        argv.append("--test")
    if extend:
        argv.append("--extend")

    logger.info("[run_all] Starting: %s", name)
    t0 = time.time()
    rc_local = 0
    try:
        module_name = f"run_all_{runner_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, runner_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load runner module from {runner_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main(argv)
    except SystemExit as exc:
        code = exc.code
        rc_local = int(code) if isinstance(code, int) else 1
    except Exception:
        rc_local = 1
        if is_root_rank():
            logger.error(
                "[run_all] Runner %s crashed:\n%s",
                runner,
                traceback.format_exc(),
            )

    elapsed = time.time() - t0
    rc = allreduce_max(rc_local)
    if is_root_rank():
        if rc != 0:
            logger.error("[run_all] FAILED: %s (exit %s)", name, rc)
        else:
            logger.info("[run_all] Done:    %s", name)
    return rc, elapsed


def main(argv: list[str] | None = None) -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run all async-ABC paper experiments."
    )
    parser.add_argument(
        "--output-dir", required=True, dest="output_dir",
        help="Root directory for all experiment outputs.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: small budget, local max 8 workers, SLURM max 48 workers, 2 replicates.",
    )
    parser.add_argument(
        "--experiments", nargs="+", metavar="NAME",
        default=list(EXPERIMENT_REGISTRY.keys()),
        help=(
            "Experiments to run (default: all). "
            f"Choices: {', '.join(EXPERIMENT_REGISTRY)}"
        ),
    )
    parser.add_argument(
        "--extend", action="store_true",
        help="Skip parameter combinations already present in existing CSVs.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate requested experiment names up front
    unknown = [n for n in args.experiments if n not in EXPERIMENT_REGISTRY]
    if unknown:
        parser.error(f"Unknown experiment(s): {', '.join(unknown)}. "
                     f"Valid names: {', '.join(EXPERIMENT_REGISTRY)}")

    failures = []
    total_start = time.time()
    total_estimate = 0.0

    for name in args.experiments:
        runner, config = EXPERIMENT_REGISTRY[name]
        rc, elapsed = _run_experiment(name, runner, config, output_dir, test_mode=args.test, extend=args.extend)
        if is_root_rank():
            logger.info("[run_all] Time:    %s took %s", name, format_duration(elapsed))
        est = None
        if args.test:
            factor, extra, note = compute_scaling_factor(CONFIGS_DIR / config)
            est = elapsed * factor + extra
            if is_root_rank():
                total_estimate += est
                logger.info(
                    "[run_all] Estimate: %s full run ~%s  (%s)",
                    name,
                    format_duration(est),
                    note,
                )
        if is_root_rank():
            write_timing_csv(output_dir / "timing_summary.csv", name, elapsed, est, args.test)
        if rc != 0:
            failures.append(name)

    total_elapsed = time.time() - total_start
    if is_root_rank():
        logger.info("[run_all] Total runtime: %s", format_duration(total_elapsed))
    if args.test and total_estimate > 0 and is_root_rank():
        logger.info(
            "[run_all] Estimated total (full run): ~%s",
            format_duration(total_estimate),
        )

    if failures:
        if is_root_rank():
            logger.error("[run_all] %s experiment(s) failed: %s", len(failures), failures)
        sys.exit(1)

    if is_root_rank():
        logger.info("[run_all] All %s experiment(s) completed successfully.", len(args.experiments))


if __name__ == "__main__":
    main()
