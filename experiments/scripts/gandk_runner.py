#!/usr/bin/env python3
"""Runner for the g-and-k distribution benchmark experiment."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.utils.metadata import write_metadata
from async_abc.utils.runner import (
    compute_scaling_factor,
    format_duration,
    make_arg_parser,
    run_experiment,
    write_timing_csv,
)


def main(argv: list[str] | None = None) -> None:
    parser = make_arg_parser("G-and-k distribution benchmark experiment.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, test_mode=args.test)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    t0 = time.time()
    records = run_experiment(cfg, output_dir, extend=args.extend)
    elapsed = time.time() - t0

    name = cfg["experiment_name"]
    estimated = None
    print(f"[{name}] Done in {format_duration(elapsed)}", flush=True)
    if args.test:
        factor, extra, note = compute_scaling_factor(args.config)
        estimated = elapsed * factor + extra
        print(
            f"[{name}] Estimated full run: ~{format_duration(estimated)}  ({note})",
            flush=True,
        )
    write_timing_csv(output_dir.data / "timing.csv", name, elapsed, estimated, args.test)

    if any(cfg.get("plots", {}).values()):
        from async_abc.plotting.reporters import plot_benchmark_diagnostics

        plot_benchmark_diagnostics(records, cfg, output_dir)
    write_metadata(output_dir, cfg)


if __name__ == "__main__":
    main()
