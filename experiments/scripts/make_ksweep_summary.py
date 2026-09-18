#!/usr/bin/env python3
"""Vendor the (d, k, S) SBC calibration sweep behind Table~\\ref{tab:ks-sweep}.

``ksweep_summary.csv`` was committed without a generator, and it stopped at
d=16 -- so the paper's d=32 claim ("coverage 0.55-0.57, 0.83-0.84, 0.92, 0.96 at
the four levels for every k in {50,100,800}") and its doubled-budget control at
d=8, k=50 had no committed data behind them, even though both runs exist. This
script is that generator: it reads every ``sbc_nd_d<D>_k<K>_s<S>`` run in the
sweep campaign and writes the summary, d=32 and the 2x-budget variant included.

Each cell is the mean empirical coverage across the run's d parameters, matching
the aggregation the existing rows use (verified against d=8, k=50, S=20:
0.289375 / 0.517750 / 0.628625 / 0.713500).

Reads the campaign output from scratch, so like ``make_kfrontier_summary.py``
it has no vendored default path; it is a one-shot table builder, re-run only
when the runs change.

    python make_ksweep_summary.py [--root <campaign>] [--out <csv>]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("/home/juhe/remotes/scratch/herold2/async-abc/ksweep_20260729")
PAPER_DATA = Path(__file__).resolve().parents[1] / "data" / "paper_figures"
ASYNC = "async_propulate_abc"
_RUN_RE = re.compile(r"^sbc_nd_d(?P<d>\d+)_k(?P<k>\d+)_s(?P<S>\d+)(?P<variant>_.+)?$")


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(root.glob("sbc_nd_d*_k*_s*")):
        match = _RUN_RE.match(run.name)
        if match is None:
            continue
        coverage = run / "data" / "coverage.csv"
        if not coverage.exists():
            print(f"  skip {run.name}: no coverage.csv")
            continue
        frame = pd.read_csv(coverage)
        frame = frame[frame["method"] == ASYNC]
        if frame.empty:
            print(f"  skip {run.name}: no {ASYNC} rows")
            continue
        for level, group in frame.groupby("coverage_level"):
            rows.append({
                "d": int(match["d"]), "k": int(match["k"]), "S": int(match["S"]),
                # "" for the main sweep, "2x" for the doubled-budget control.
                "variant": (match["variant"] or "").lstrip("_"),
                "level": float(level),
                "cov": float(group["empirical_coverage"].mean()),
                "n_params": int(group["param"].nunique()),
                "n_trials": int(group["n_trials"].max()),
            })
    return pd.DataFrame(rows).sort_values(
        ["variant", "d", "k", "S", "level"]
    ).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(PAPER_DATA / "ksweep_summary.csv"))
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"campaign root not found: {root}")
    table = collect(root)
    if table.empty:
        raise SystemExit(f"no sbc_nd_* runs with coverage.csv under {root}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table.to_csv(index=False))
    print(f"wrote {out}  ({len(table)} rows)")
    print(f"  dimensions {sorted(table.d.unique())}")
    print(f"  archive sizes {sorted(table.k.unique())}")
    print(f"  snapshot counts {sorted(table.S.unique())}")
    print(f"  variants {sorted(x for x in table.variant.unique() if x)}")

    main_sweep = table[table.variant == ""]
    for d in sorted(main_sweep.d.unique()):
        sub = main_sweep[(main_sweep.d == d) & (main_sweep.S == 20)]
        if sub.empty:
            continue
        print(f"\n  d={d}, S=20 -- mean coverage by k:")
        print(sub.pivot_table(index="k", columns="level", values="cov")
              .round(3).to_string())


if __name__ == "__main__":
    main()
