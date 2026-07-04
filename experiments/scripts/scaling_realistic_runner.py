#!/usr/bin/env python3
"""Scaling experiment runner for the realistic-workload benchmark.

Adapts the LV scaling runner for the realistic workload by redirecting the simulation
scratch directory under the experiment output directory, so parallel cluster
jobs for different (n_workers, k) combinations don't share a sim scratch path.

Usage mirrors scaling_runner.py; point --config at experiments/configs/scaling_realistic.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.paths import OutputDir

from scaling_runner import main as _scaling_main


def _prepare_runtime_cfg(cfg: dict, output_dir: OutputDir) -> dict:
    """Redirect simulation scratch files under the experiment output directory."""
    cfg = dict(cfg)
    benchmark_cfg = dict(cfg["benchmark"])
    benchmark_cfg["output_dir"] = str(output_dir.root / "realistic_sims")
    cfg["benchmark"] = benchmark_cfg
    return cfg


def main(argv: list[str] | None = None) -> None:
    _scaling_main(argv, prepare_runtime_cfg=_prepare_runtime_cfg)


if __name__ == "__main__":
    main()
