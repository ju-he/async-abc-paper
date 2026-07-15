"""Shared plumbing for the paper-figure generators (``make_*_fig.py``).

Every paper figure now follows the same two-path contract so the repository
regenerates its figures on its own (the Declarations reproducibility promise):

* **Default path** — read the *vendored* per-figure CSVs committed under
  ``experiments/data/paper_figures/<name>/`` and draw. No cluster, no scratch.
* **``--refresh`` path** — re-derive those CSVs from the campaign output
  (``rerun_20260707`` mirrored locally at :data:`STAGING_ROOT`), re-vendor them,
  and draw. This is the only path that touches large/off-repo data.

A generator therefore splits into ``aggregate(root) -> {csv_name: DataFrame}``
(reads the campaign data, returns exactly the small frames it plots) and
``draw(frames)`` (pure plotting). :func:`run` wires the two together with the
argparse flag and :func:`async_abc.plotting.paper_style.save_paper_figure`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

# experiments/ on sys.path so ``async_abc`` imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from async_abc.plotting import paper_style as ps  # noqa: E402

# Local mirror of /p/scratch/.../async-abc/rerun_20260707 (small summaries + plots
# CSVs only; see the staging rsync). Override with --refresh <path>.
STAGING_ROOT = Path("/home/juhe/async-abc-rerun-staging")


def add_refresh_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--refresh",
        nargs="?",
        const=str(STAGING_ROOT),
        default=None,
        metavar="CAMPAIGN_ROOT",
        help=(
            "Re-derive and re-vendor this figure's CSVs from the campaign output "
            f"root (default: {STAGING_ROOT}). Without --refresh the figure is drawn "
            "from the committed vendored CSVs."
        ),
    )


def load_vendored(name: str) -> Dict[str, pd.DataFrame]:
    """Load the committed per-figure CSVs for figure ``name``."""
    vdir = ps.DATA_DIR / name
    csvs = sorted(vdir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No vendored data for '{name}' under {vdir}. Run with --refresh to "
            "derive it from the campaign output first."
        )
    return {p.stem: pd.read_csv(p) for p in csvs}


def run(
    name: str,
    description: str,
    aggregate: Callable[[Path], Dict[str, pd.DataFrame]],
    draw: Callable[[Dict[str, pd.DataFrame]], "object"],
    metadata: Dict[str, object] | None = None,
) -> None:
    """Standard entry point: parse args, obtain frames, draw, save/vendor.

    ``aggregate(campaign_root)`` returns ``{csv_name: DataFrame}`` (only on
    --refresh); ``draw(frames)`` returns the matplotlib ``Figure`` to save.
    """
    parser = argparse.ArgumentParser(description=description)
    add_refresh_arg(parser)
    args = parser.parse_args()

    ps.apply()
    if args.refresh is not None:
        frames = aggregate(Path(args.refresh))
        vendor = frames
    else:
        frames = load_vendored(name)
        vendor = None  # already committed; don't rewrite

    fig = draw(frames)
    saved = ps.save_paper_figure(fig, name, data=vendor, metadata=metadata)
    print(f"wrote {saved['pdf']}")
    if vendor is not None:
        for k, v in saved.items():
            if k.startswith("csv:"):
                print(f"  vendored {v}")
