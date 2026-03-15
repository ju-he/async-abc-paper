#!/usr/bin/env python3
"""Runner for the g-and-k distribution benchmark experiment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.utils.metadata import write_metadata
from async_abc.utils.runner import make_arg_parser, run_experiment


def main() -> None:
    parser = make_arg_parser("G-and-k distribution benchmark experiment.")
    args = parser.parse_args()

    cfg = load_config(args.config, test_mode=args.test)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    run_experiment(cfg, output_dir)
    write_metadata(output_dir, cfg)


if __name__ == "__main__":
    main()
