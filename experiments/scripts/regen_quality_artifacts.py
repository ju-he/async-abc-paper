#!/usr/bin/env python3
"""Regenerate quality-vs-wall-time artifacts from existing raw records.

Post-hoc analysis only -- this runs **no simulation**. It rebuilds each
experiment's ``plots/quality_vs_wall_time{,_diagnostic}_data.csv`` from the
``data/raw_results.csv`` already on disk, using the kernel-correct archive
reconstruction (see the 2026-07-22 entry in ``.plans/bug-fixes/previous-fixes.md``).

The previous artifacts were built with the *hard*-kernel archive rule
(``loss < eps``) applied to smooth-kernel runs. That under-sized the archive for
most of every run -- Lotka--Volterra's curve used 8--51 particles instead of
k=100 throughout -- and, once the monotonically decreasing bandwidth fell below
the best achieved loss, emptied the archive so every later checkpoint was
silently dropped (Cellular Potts stopped at 28% of its budget).

Originals are preserved as ``*.prekernelfix.csv`` rather than overwritten.

Usage
-----
    python regen_quality_artifacts.py --root <campaign_root> \
        --experiments gandk lotka_volterra gaussian_mean cellular_potts

Memory: raw_results.csv reaches ~10 GB on the fast-simulator benchmarks, so only
the columns the reconstruction needs are read. Run the big ones on a large-memory
node (mem192, ``--exclusive``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from async_abc.analysis import posterior_quality_curve  # noqa: E402
from async_abc.plotting.reporters import _step_curve_summary  # noqa: E402

# Columns _prepare_quality_frame consumes; anything else in the 10 GB raw file is
# dead weight for this pass.
_BASE_COLS = [
    "method",
    "replicate",
    "step",
    "loss",
    "tolerance",
    "wall_time",
    "sim_start_time",
    "sim_end_time",
    "generation",
    "record_kind",
    "time_semantics",
    "attempt_count",
]


def _true_params(benchmark_cfg: dict) -> dict[str, float]:
    return {
        key.removeprefix("true_"): float(value)
        for key, value in benchmark_cfg.items()
        if key.startswith("true_") and isinstance(value, (int, float))
    }


def _load_records(raw_csv: Path) -> pd.DataFrame:
    header = pd.read_csv(raw_csv, nrows=0).columns.tolist()
    param_cols = [c for c in header if c.startswith("param_")]
    usecols = [c for c in _BASE_COLS if c in header] + param_cols
    return pd.read_csv(raw_csv, usecols=usecols)


def regen_experiment(root: Path, name: str) -> None:
    exp = root / name
    raw_csv = exp / "data" / "raw_results.csv"
    meta_path = exp / "data" / "metadata.json"
    if not raw_csv.exists() or not meta_path.exists():
        print(f"[{name}] SKIP: missing raw_results.csv or metadata.json")
        return

    cfg = json.loads(meta_path.read_text()).get("config", {})
    inference = cfg.get("inference", {})
    kernel = str(inference.get("kernel", "hard"))
    archive_size = inference.get("k")
    true_params = _true_params(cfg.get("benchmark", {}))
    if not true_params:
        print(f"[{name}] SKIP: no true_* params in benchmark config")
        return

    print(f"[{name}] loading {raw_csv} ({raw_csv.stat().st_size / 1e9:.1f} GB) ...", flush=True)
    records = _load_records(raw_csv)
    print(f"[{name}] {len(records):,} records; kernel={kernel} k={archive_size}", flush=True)

    plots = exp / "plots"
    plots.mkdir(exist_ok=True)
    # Two artifacts with *different* schemas, matching the reporters:
    #  - ``*_diagnostic_data.csv`` holds the raw per-replicate checkpoint rows,
    #  - ``*_data.csv`` is the replicate-aggregated summary (median + CI columns)
    #    produced by ``_step_curve_summary``, which the paper figures consume.
    for stem, strategy, aggregate in (
        ("quality_vs_wall_time_diagnostic_data", "quantile", False),
        ("quality_vs_wall_time_data", "time_uniform", True),
    ):
        out = plots / f"{stem}.csv"
        quality = posterior_quality_curve(
            records,
            true_params=true_params,
            axis_kind="wall_time",
            checkpoint_strategy=strategy,
            checkpoint_count=8,
            archive_size=archive_size,
            kernel=kernel,
        )
        if quality.empty:
            print(f"[{name}] {stem}: EMPTY -- left untouched")
            continue

        frame = quality
        if aggregate:
            frame = _step_curve_summary(
                quality,
                x_col="axis_value",
                y_col="wasserstein",
                ci_level=0.95,
                log_y=False,
                lower_bound=0.0,
            )
            if frame.empty:
                print(f"[{name}] {stem}: EMPTY summary -- left untouched")
                continue

        # Preserve the pre-fix artifact once; never clobber it on a re-run, or the
        # only copy of the original would be lost.
        backup = plots / f"{stem}.prekernelfix.csv"
        if out.exists() and not backup.exists():
            out.replace(backup)
        frame.to_csv(out, index=False)

        async_rows = quality[quality["method"].astype(str).str.startswith("async_propulate_abc")]
        if not async_rows.empty:
            span = async_rows.groupby("replicate")["wall_time"].max().median()
            npu = async_rows["n_particles_used"].median()
            print(f"[{name}] {stem}: {len(frame)} rows, async median last-obs "
                  f"{span:.0f}s, median archive {npu:.0f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--experiments", nargs="+", required=True)
    args = parser.parse_args()
    for name in args.experiments:
        regen_experiment(args.root, name)


if __name__ == "__main__":
    main()
