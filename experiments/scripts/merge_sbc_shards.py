#!/usr/bin/env python3
"""Manually merge completed SBC shards into a coverage table + figures (WS3).

The sweep-submitted SBC run lost one shard to a plan-file startup race, so the
automatic 10-shard finalize will not trigger. This reproduces the finalizer's merge
(async_abc.utils.shard_finalizers.finalize_sbc_experiment) over whatever shards
actually completed: concatenate each shard's sbc_trials.jsonl, then compute empirical
coverage + SBC ranks with the SAME analysis functions. Prints the Table-4 numbers and
regenerates fig_sbc_coverage.pdf / fig_sbc_rank.pdf.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # experiments/ on path
from async_abc.analysis.sbc import empirical_coverage, sbc_ranks  # noqa: E402

RUNS = "/home/juhe/remotes/scratch/herold2/async-abc/sbc_1k_20260629/_shards/sbc/runs/sbc1k0629"
FIGDIR = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures"
LEVELS = [0.5, 0.8, 0.9, 0.95]
LABEL = {"abc_smc_baseline": "Synchronous baseline", "async_propulate_abc": "Asynchronous (ours)"}


def _load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            p["posterior_samples"] = np.asarray(p["posterior_samples"], dtype=float)
            if p.get("posterior_weights") is not None:
                p["posterior_weights"] = np.asarray(p["posterior_weights"], dtype=float)
            out.append(p)
    return out


def main() -> None:
    records: list[dict] = []
    shard_files = sorted(glob.glob(os.path.join(RUNS, "shard-*/sbc/data/sbc_trials.jsonl")))
    for jl in shard_files:
        recs = _load_jsonl(jl)
        records.extend(recs)
        print(f"  {jl.split('/runs/')[1]}: {len(recs)} trial-records")
    n_trials = len({(r.get("trial"), r.get("method")) for r in records})
    print(f"total trial-records: {len(records)}  (~{len(records)//2} trials x 2 methods)")

    cov = empirical_coverage(records, LEVELS)
    ranks = sbc_ranks(records)
    cov.to_csv(os.path.join(os.path.dirname(__file__), "sbc_coverage_merged.csv"), index=False)

    print("\n=== empirical coverage (Table 4) ===")
    print(f"{'method':>22} " + " ".join(f"{l:>6}" for l in LEVELS))
    for m in ["abc_smc_baseline", "async_propulate_abc"]:
        sub = cov[cov["method"] == m].set_index("coverage_level")["empirical_coverage"]
        if len(sub) == 0:
            continue
        print(f"{LABEL[m]:>22} " + " ".join(f"{sub.get(l, float('nan')):>6.3f}" for l in LEVELS))

    # --- fig_sbc_coverage.pdf : empirical vs nominal ---
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], ":", color="0.5", label="nominal")
    for m, color, mk in [("abc_smc_baseline", "#d62728", "s"), ("async_propulate_abc", "#1f77b4", "o")]:
        sub = cov[cov["method"] == m].set_index("coverage_level")["empirical_coverage"]
        if len(sub) == 0:
            continue
        xs = sorted(sub.index)
        ax.plot(xs, [sub[x] for x in xs], "-" + mk, color=color, label=LABEL[m], ms=7,
                mfc="white" if mk == "s" else color)
    ax.set_xlabel("nominal coverage level")
    ax.set_ylabel("empirical coverage")
    ax.set_title(f"SBC coverage ({len(records)//2} trials)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_sbc_coverage.pdf"), bbox_inches="tight")

    # --- fig_sbc_rank.pdf : rank histogram (async) ---
    rcol = "rank" if "rank" in ranks.columns else [c for c in ranks.columns if "rank" in c.lower()][0]
    a = ranks[ranks["method"] == "async_propulate_abc"][rcol].to_numpy(dtype=float)
    fig2, ax2 = plt.subplots(figsize=(4.6, 4.2))
    nbins = 20
    ax2.hist(a, bins=nbins, color="#1f77b4", alpha=0.8, edgecolor="white")
    exp = len(a) / nbins
    ax2.axhline(exp, color="0.4", ls="--", lw=1, label="uniform")
    ax2.set_xlabel("rank of true value")
    ax2.set_ylabel("count")
    ax2.set_title(f"SBC rank histogram (async, {len(a)} trials)")
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGDIR, "fig_sbc_rank.pdf"), bbox_inches="tight")
    print(f"\nwrote fig_sbc_coverage.pdf + fig_sbc_rank.pdf ({len(a)} async ranks)")


if __name__ == "__main__":
    main()
