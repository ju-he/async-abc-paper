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


def _summary_paths(exp: Path) -> list[Path]:
    """Summary CSVs under ``exp`` carrying a ``final_quality_wasserstein`` column.

    Covers both shapes the runners emit: the straggler/heterogeneity single
    summary and the scaling runs' per-combination ``throughput_summary_w*_k*``
    shards plus their aggregate.
    """
    data = exp / "data"
    if not data.is_dir():
        return []
    found = []
    for path in sorted(data.glob("*.csv")):
        if path.name.endswith(".prekernelfix.csv"):
            continue
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:  # noqa: BLE001 - a malformed CSV is not this pass's business
            continue
        if "final_quality_wasserstein" in header:
            found.append(path)
    return found


def regen_summaries(root: Path, name: str) -> None:
    """Rewrite ``final_quality_wasserstein`` in an experiment's summary CSVs.

    ``regen_experiment`` above rebuilt the quality *curves*, but the summary
    CSVs carry their own copy of the curve's final value, computed by
    ``runtime_summary._final_quality_wasserstein`` /
    ``scaling_runner._quality_curve_by_wall_time`` at run time -- i.e. with the
    hard-kernel archive rule on any run that predates f62f143. Those columns are
    therefore still pre-fix on disk even where the curves have been corrected.

    Only the unweighted metric is affected. ``final_quality_wasserstein_weighted``
    and ``_analytic`` resample the reported posterior via
    ``runtime_summary._weighted_final_frame``, which never touches the
    kernel-dependent archive reconstruction, so they are left alone.

    Recomputes per (method, replicate) from the raw records with the run's own
    kernel, preserving the original as ``*.prekernelfix.csv``.
    """
    exp = root / name
    raw_csv = exp / "data" / "raw_results.csv"
    meta_path = exp / "data" / "metadata.json"
    if not raw_csv.exists() or not meta_path.exists():
        print(f"[{name}] summaries SKIP: missing raw_results.csv or metadata.json")
        return
    summaries = _summary_paths(exp)
    if not summaries:
        print(f"[{name}] summaries SKIP: no final_quality_wasserstein column on disk")
        return

    cfg = json.loads(meta_path.read_text()).get("config", {})
    inference = cfg.get("inference", {})
    kernel = str(inference.get("kernel", "hard"))
    true_params = _true_params(cfg.get("benchmark", {}))
    if not true_params:
        print(f"[{name}] summaries SKIP: no true_* params in benchmark config")
        return

    print(f"[{name}] summaries: loading {raw_csv} "
          f"({raw_csv.stat().st_size / 1e9:.1f} GB) ...", flush=True)
    records = _load_records(raw_csv)
    print(f"[{name}] summaries: {len(records):,} records; kernel={kernel}", flush=True)

    # The archive size varies per row on the scaling runs (k is swept), so the
    # curve is computed per (method, replicate) with that row's own k.
    cache: dict[tuple[str, int, int], float] = {}

    def _final(method: str, replicate: int, archive_size: int) -> float:
        key = (method, replicate, archive_size)
        if key in cache:
            return cache[key]
        subset = records[
            (records["method"].astype(str) == method)
            & (records["replicate"].astype(int) == int(replicate))
        ]
        value = float("nan")
        if not subset.empty:
            quality = posterior_quality_curve(
                subset,
                true_params=true_params,
                axis_kind="wall_time",
                checkpoint_strategy="quantile",
                checkpoint_count=8,
                archive_size=archive_size,
                kernel=kernel,
            )
            if not quality.empty:
                value = float(quality.sort_values("axis_value").iloc[-1]["wasserstein"])
        cache[key] = value
        return value

    for path in summaries:
        table = pd.read_csv(path)
        # The straggler/heterogeneity summaries name the tagged run ``method``;
        # the scaling summaries call the same thing ``method_variant`` and reserve
        # ``method``-less ``base_method`` for the family. Either way the value is
        # what the raw records' ``method`` column holds.
        method_col = next(
            (c for c in ("method", "method_variant") if c in table.columns), None
        )
        if method_col is None or "replicate" not in table.columns:
            print(f"[{path.name}] SKIP: no method/replicate columns to key on")
            continue
        old = table["final_quality_wasserstein"].astype(float).copy()
        updated = []
        for _, row in table.iterrows():
            archive_size = int(row["k"]) if "k" in table.columns and pd.notna(row["k"]) \
                else inference.get("k")
            updated.append(_final(str(row[method_col]), int(row["replicate"]), archive_size))
        table["final_quality_wasserstein"] = updated

        backup = path.with_suffix(".prekernelfix.csv")
        if not backup.exists():
            pd.read_csv(path).to_csv(backup, index=False)
        table.to_csv(path, index=False)

        new = table["final_quality_wasserstein"].astype(float)
        both = old.notna() & new.notna()
        shift = (new[both] - old[both]).abs()
        print(f"[{path.name}] {len(table)} rows rewritten; "
              f"median |change| {shift.median() if len(shift) else float('nan'):.4f}, "
              f"max {shift.max() if len(shift) else float('nan'):.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument(
        "--summaries",
        action="store_true",
        help=(
            "Also rewrite the stale final_quality_wasserstein column in the "
            "experiment's summary CSVs (see regen_summaries). Independent of the "
            "curve rebuild -- pass --summaries-only to skip the curves."
        ),
    )
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        dest="summaries_only",
        help="Rewrite only the summary CSVs; leave the quality curves untouched.",
    )
    args = parser.parse_args()
    for name in args.experiments:
        if not args.summaries_only:
            regen_experiment(args.root, name)
        if args.summaries or args.summaries_only:
            regen_summaries(args.root, name)


if __name__ == "__main__":
    main()
