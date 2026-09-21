#!/usr/bin/env python3
"""Repair the synchronous rows of the two-parameter Cellular Potts records.

``RecordWriter`` used to rebuild its column order from every batch it wrote and
hand that order to a fresh ``csv.DictWriter`` under the header of the *first*
batch. The asynchronous arm writes its parameters in config order
(``division_rate, cell_volume``); the synchronous baseline's come back
alphabetically (through a ``sort_keys=True`` trace), so every one of its rows
-- attempts and population particles alike -- was written with the two
parameter columns swapped. The writer is fixed (``async_abc/io/records.py``);
this script repairs the three stored runs that were written before the fix and
proves the repair against pyABC's own SQLite histories, which were never
touched by the bug.

It also extracts those histories' populations *with their importance weights*
(the record builder used to drop them, see ``population_weight``) into one
compact ``pyabc_populations.csv.gz``, which is what the paper's table scripts
score the synchronous posterior from.

    python experiments/scripts/repair_two_param_cpm_records.py --histories <dir>

``<dir>`` holds the ``abc_smc_baseline_rep*_seed*.db`` files extracted from
``cpm_two_param_fixed.tar.gz``, ``cpm_two_param_production.tar.gz`` and
``cpm_80_comparison.tar.gz`` (scratch), under their tarball-relative paths.
"""
from __future__ import annotations

import argparse
import gzip
import io
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "cpm_two_param_validation"
NAMES = ["division_rate", "cell_volume"]
RUNS = {
    "50_fixed": ("cpm_two_param_fixed", "cpm_two_param_fixed/cellular_potts_two_param"),
    "50_control": ("cpm_two_param_production", "cpm_two_param_production/cellular_potts_two_param"),
    "80": ("cpm_80_comparison", "cpm_80_comparison/cellular_potts_two_param_80"),
}
SWAPPED_METHODS = {"abc_smc_baseline"}


def _load_raw(run: str) -> tuple[pd.DataFrame, Path]:
    tar, rel = RUNS[run]
    gz = DATA / rel / "data" / "raw_results.csv.gz"
    if gz.exists():
        return pd.read_csv(gz), gz
    with tarfile.open(DATA / f"{tar}.tar.gz") as tf:
        member = f"{rel}/data/raw_results.csv"
        df = pd.read_csv(io.TextIOWrapper(tf.extractfile(member)))
    gz.parent.mkdir(parents=True, exist_ok=True)
    return df, gz


def _populations(histories: Path, rel: str, run: str) -> pd.DataFrame:
    import pyabc

    rows = []
    for db in sorted((histories / rel / "data").glob("abc_smc_baseline_rep*_seed*.db")):
        rep = int(re.search(r"rep(\d+)", db.name).group(1))
        h = pyabc.History(f"sqlite:///{db}", create=False)
        eps = h.get_all_populations().set_index("t")["epsilon"]
        for t in range(h.max_t + 1):
            df, w = h.get_distribution(m=0, t=t)
            dist = h.get_weighted_distances(t=t)["distance"].to_numpy(float)
            for j in range(len(df)):
                rows.append(dict(run=run, replicate=rep, generation=t, epsilon=float(eps[t]),
                                 **{n: float(df[n].iloc[j]) for n in NAMES},
                                 weight=float(np.asarray(w)[j]), distance=float(dist[j])))
    return pd.DataFrame(rows)


def repair(histories: Path) -> None:
    pops = []
    for run, (tar, rel) in RUNS.items():
        df, gz = _load_raw(run)
        pop_db = _populations(histories, rel, run)
        pops.append(pop_db)
        sw = df["method"].isin(SWAPPED_METHODS)
        a, b = f"param_{NAMES[0]}", f"param_{NAMES[1]}"
        df.loc[sw, [a, b]] = df.loc[sw, [b, a]].to_numpy()
        # Proof: every repaired population generation equals the history's, by name.
        pp = df[sw & (df["record_kind"] == "population_particle")]
        for (rep, t), g in pp.groupby(["replicate", "generation"]):
            ref = pop_db[(pop_db.replicate == rep) & (pop_db.generation == int(t))]
            for n in NAMES:
                if not np.allclose(np.sort(g[f"param_{n}"]), np.sort(ref[n]), atol=1e-9):
                    raise AssertionError(f"{run} rep {rep} t={int(t)}: repaired {n} differs from the pyABC history")
        with gzip.open(gz, "wt", newline="") as f:
            df.to_csv(f, index=False)
        print(f"{run}: repaired {int(sw.sum())} synchronous rows in {gz.relative_to(DATA)}; "
              f"{pp.groupby(['replicate', 'generation']).ngroups} populations verified against the history")
    out = DATA / "pyabc_populations.csv.gz"
    with gzip.open(out, "wt", newline="") as f:
        pd.concat(pops).to_csv(f, index=False)
    print(f"wrote {out.relative_to(DATA)} ({sum(len(p) for p in pops)} particles with weights)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=Path, required=True)
    args = parser.parse_args()
    repair(args.histories)


if __name__ == "__main__":
    main()
