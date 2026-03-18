#!/usr/bin/env python3
"""Persistent straggler experiment."""
import csv
import multiprocessing
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.inference.method_registry import run_method
from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.io.records import RecordWriter
from async_abc.plotting.export import save_figure
from async_abc.utils.metadata import write_metadata
from async_abc.utils.runner import compute_scaling_factor, find_completed_combinations, format_duration, make_arg_parser, write_timing_csv
from async_abc.utils.seeding import make_seeds


def _current_worker_rank() -> int:
    """Best-effort worker rank for MPI or multiprocessing workers."""
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank())
    except Exception:
        pass

    proc = multiprocessing.current_process()
    if proc._identity:
        return int(proc._identity[0] - 1)
    return 0


def _make_straggler_simulate(
    simulate_fn,
    straggler_rank: int,
    slowdown_factor: float,
    base_sleep_s: float,
    test_mode: bool = False,
):
    """Wrap simulate_fn with a persistent slowdown on one worker."""

    def wrapped(params, seed):
        result = simulate_fn(params, seed=seed)
        if not test_mode and _current_worker_rank() == straggler_rank:
            time.sleep(float(slowdown_factor) * float(base_sleep_s))
        return result

    return wrapped


def _plot_throughput_vs_slowdown(throughput_rows, output_dir: OutputDir) -> None:
    if not throughput_rows:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    methods = sorted({row["base_method"] for row in throughput_rows})
    for method in methods:
        rows = [row for row in throughput_rows if row["base_method"] == method]
        rows = sorted(rows, key=lambda row: row["slowdown_factor"])
        ax.plot(
            [row["slowdown_factor"] for row in rows],
            [row["throughput_sims_per_s"] for row in rows],
            marker="o",
            label=method,
        )

    ax.set_xlabel("slowdown factor")
    ax.set_ylabel("throughput (sim/s)")
    ax.set_title("Throughput vs. straggler slowdown")
    ax.legend(frameon=False)
    fig.tight_layout()

    data = {
        "slowdown_factor": [row["slowdown_factor"] for row in throughput_rows],
        "base_method": [row["base_method"] for row in throughput_rows],
        "replicate": [row["replicate"] for row in throughput_rows],
        "throughput_sims_per_s": [row["throughput_sims_per_s"] for row in throughput_rows],
        "wall_time_s": [row["wall_time_s"] for row in throughput_rows],
    }
    save_figure(fig, output_dir.plots / "throughput_vs_slowdown", data=data)


def main(argv: list[str] | None = None) -> None:
    parser = make_arg_parser("Persistent straggler experiment.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, test_mode=args.test)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    bm = make_benchmark(cfg["benchmark"])
    straggler_cfg = cfg["straggler"]
    slowdown_factors = [float(x) for x in straggler_cfg.get("slowdown_factor", [1.0])]
    straggler_rank = int(straggler_cfg.get("straggler_rank", 0))
    base_sleep_s = float(straggler_cfg.get("base_sleep_s", 0.1))

    csv_path = output_dir.data / "raw_results.csv"
    writer = RecordWriter(csv_path)
    seeds = make_seeds(cfg["execution"]["n_replicates"], cfg["execution"]["base_seed"])

    all_records = []
    throughput_rows = []
    worst_records = []
    original_simulate = bm.simulate

    experiment_start = time.time()
    for slowdown_factor in slowdown_factors:
        bm.simulate = _make_straggler_simulate(
            original_simulate,
            straggler_rank=straggler_rank,
            slowdown_factor=slowdown_factor,
            base_sleep_s=base_sleep_s,
            test_mode=args.test,
        )
        factor_records = []
        done = find_completed_combinations(csv_path, ["method", "replicate"]) if args.extend else set()
        for method in cfg["methods"]:
            tagged_method = f"{method}__straggler_slowdown{slowdown_factor:.4g}x"
            for replicate, seed in enumerate(seeds):
                if (tagged_method, str(replicate)) in done:
                    print(f"[straggler] --extend: skipping {tagged_method} replicate={replicate}", flush=True)
                    continue
                t0 = time.time()
                records = run_method(
                    method,
                    bm.simulate,
                    bm.limits,
                    cfg["inference"],
                    output_dir,
                    replicate,
                    seed,
                )
                elapsed = time.time() - t0
                for record in records:
                    record.method = tagged_method
                writer.write(records)
                all_records.extend(records)
                factor_records.extend(records)
                throughput_rows.append(
                    {
                        "slowdown_factor": slowdown_factor,
                        "base_method": method,
                        "method": tagged_method,
                        "replicate": replicate,
                        "seed": seed,
                        "n_simulations": len(records),
                        "wall_time_s": elapsed,
                        "throughput_sims_per_s": len(records) / elapsed if elapsed > 0 else float("nan"),
                    }
                )

        if slowdown_factor == max(slowdown_factors):
            worst_records = factor_records

    bm.simulate = original_simulate

    throughput_path = output_dir.data / "throughput_vs_slowdown_summary.csv"
    if throughput_rows:
        with open(throughput_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(throughput_rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(throughput_rows)

    elapsed = time.time() - experiment_start
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

    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("throughput_vs_slowdown"):
        _plot_throughput_vs_slowdown(throughput_rows, output_dir)
    if plots_cfg.get("gantt") and worst_records:
        from async_abc.plotting.reporters import plot_worker_gantt

        plot_worker_gantt(worst_records, output_dir)

    write_metadata(
        output_dir,
        cfg,
        extra={
            "slowdown_factors": slowdown_factors,
            "straggler_rank": straggler_rank,
            "base_sleep_s": base_sleep_s,
        },
    )


if __name__ == "__main__":
    main()
