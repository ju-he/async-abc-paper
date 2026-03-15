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
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
SCRIPTS_DIR = EXPERIMENTS_DIR / "scripts"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"

# Ordered mapping: experiment_name → (runner_script, config_file)
EXPERIMENT_REGISTRY = {
    "gaussian_mean": ("gaussian_mean_runner.py", "gaussian_mean.json"),
    "gandk": ("gandk_runner.py", "gandk.json"),
    "lotka_volterra": ("lotka_volterra_runner.py", "lotka_volterra.json"),
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
) -> int:
    """Run a single experiment script as a subprocess.

    Returns the process return code.
    """
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / runner),
        "--config", str(CONFIGS_DIR / config),
        "--output-dir", str(output_dir),
    ]
    if test_mode:
        cmd.append("--test")

    print(f"[run_all] Starting: {name}", flush=True)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"[run_all] FAILED: {name} (exit {result.returncode})", file=sys.stderr)
    else:
        print(f"[run_all] Done:    {name}", flush=True)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all async-ABC paper experiments."
    )
    parser.add_argument(
        "--output-dir", required=True, dest="output_dir",
        help="Root directory for all experiment outputs.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: small budget, ≤8 workers, 2 replicates.",
    )
    parser.add_argument(
        "--experiments", nargs="+", metavar="NAME",
        default=list(EXPERIMENT_REGISTRY.keys()),
        help=(
            "Experiments to run (default: all). "
            f"Choices: {', '.join(EXPERIMENT_REGISTRY)}"
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate requested experiment names up front
    unknown = [n for n in args.experiments if n not in EXPERIMENT_REGISTRY]
    if unknown:
        parser.error(f"Unknown experiment(s): {', '.join(unknown)}. "
                     f"Valid names: {', '.join(EXPERIMENT_REGISTRY)}")

    failures = []
    for name in args.experiments:
        runner, config = EXPERIMENT_REGISTRY[name]
        rc = _run_experiment(name, runner, config, output_dir, test_mode=args.test)
        if rc != 0:
            failures.append(name)

    if failures:
        print(f"\n[run_all] {len(failures)} experiment(s) failed: {failures}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[run_all] All {len(args.experiments)} experiment(s) completed successfully.")


if __name__ == "__main__":
    main()
